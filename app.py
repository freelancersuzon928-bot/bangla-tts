import asyncio
import io
import re
import time
import zipfile

import edge_tts
import numpy as np
import streamlit as st
from pydub import AudioSegment

st.set_page_config(page_title="বাংলা ডকুমেন্টারি ভয়েস", page_icon="🎙️", layout="wide")

VOICES = {
    "নবনীতা (মহিলা - বাংলাদেশ)": "bn-BD-NabanitaNeural",
    "প্রদীপ (পুরুষ - বাংলাদেশ)": "bn-BD-PradeepNeural",
    "তানিশা (মহিলা - কলকাতা)": "bn-IN-TanishaaNeural",
    "ভাস্কর (পুরুষ - কলকাতা)": "bn-IN-BashkarNeural",
}

PRESETS = {
    "নিজে ঠিক করব": None,
    "ডকুমেন্টারি - গম্ভীর ভারী (প্রদীপ)": ("প্রদীপ (পুরুষ - বাংলাদেশ)", -20, -12, 0, 450, 1100),
    "ডকুমেন্টারি - নরম মহিলা (নবনীতা)": ("নবনীতা (মহিলা - বাংলাদেশ)", -5, -10, 0, 400, 1000),
    "আবৃত্তি - ধীর ও বিরতিসহ": ("প্রদীপ (পুরুষ - বাংলাদেশ)", -10, -18, 0, 800, 1500),
    "খবর - দ্রুত ও স্পষ্ট": ("নবনীতা (মহিলা - বাংলাদেশ)", 0, 8, 0, 250, 600),
}


def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text.strip()) if p.strip()]


def split_sentences(par, max_len=350):
    parts = [p.strip() for p in re.split(r"(?<=[।?!])\s+|\n", par) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        while len(p) > max_len:  # খুব লম্বা বাক্য কমা/স্পেসে ভাঙা
            cut = max(p.rfind(",", 0, max_len), p.rfind("،", 0, max_len), p.rfind(" ", 0, max_len))
            cut = cut if cut > 50 else max_len
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut].strip())
            p = p[cut:].lstrip(", ")
        if len(cur) + len(p) + 1 <= max_len:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


async def synth(text, voice, rate, pitch, volume):
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    data = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    return data


def synth_retry(text, voice, rate, pitch, volume, tries=5):
    last = None
    for i in range(tries):
        try:
            data = asyncio.run(synth(text, voice, rate, pitch, volume))
            if data:
                return data
        except Exception as e:
            last = e
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"ভয়েস তৈরি হয়নি। একটু পরে আবার চেষ্টা করুন। ({last})")


MUSIC_STYLES = {
    "কোনো মিউজিক নয়": None,
    "শান্ত গম্ভীর প্যাড (ডকুমেন্টারি/ইতিহাস)": [110.0, 164.5, 220.0, 329.0],
    "নরম উজ্জ্বল প্যাড (গল্প/প্রকৃতি)": [130.5, 196.0, 261.5, 392.0],
    "গভীর ড্রোন (রহস্য/ইতিহাসের ভারী সুর)": [55.0, 82.5, 110.0, 165.0],
    "নিজের মিউজিক ফাইল আপলোড করব": "upload",
}


def make_ambient(freqs, seconds=20, sr=44100):
    """কপিরাইটমুক্ত নরম প্যাড মিউজিক, লুপ করলে জোড়া ধরা পড়ে না।"""
    t = np.arange(int(seconds * sr)) / sr
    sig = np.zeros_like(t)
    for i, f in enumerate(freqs):
        f = round(f * seconds) / seconds  # পূর্ণ চক্র, তাই নিখুঁত লুপ
        lfo_f = (1 + i) / seconds
        env = 0.65 + 0.35 * np.sin(2 * np.pi * lfo_f * t + i)
        sig += env * (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) / (1 + i * 0.5))
    sig = sig / np.max(np.abs(sig)) * 0.8
    pcm = (sig * 32767).astype(np.int16)
    return AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)


def mix_music(voice, music, music_db, lead_ms=2500, tail_ms=4000):
    if music is None:
        return voice
    total = lead_ms + len(voice) + tail_ms
    loops = int(total / len(music)) + 1
    bg = (music * loops)[:total]
    bg = bg.apply_gain((voice.dBFS + music_db) - bg.dBFS)
    bg = bg.fade_in(2000).fade_out(min(tail_ms + 1500, total // 2))
    padded = AudioSegment.silent(duration=lead_ms) + voice + AudioSegment.silent(duration=tail_ms)
    return bg.overlay(padded)


def to_mp3(seg):
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate="128k")
    return buf.getvalue()


def build(text, voice, rate, pitch, volume, sent_pause, para_pause, part_minutes, progress, music=None, music_db=-22):
    paragraphs = split_paragraphs(text)
    jobs = []  # (chunk, is_last_in_paragraph)
    for par in paragraphs:
        chs = split_sentences(par)
        for i, c in enumerate(chs):
            jobs.append((c, i == len(chs) - 1))
    total = len(jobs)
    final = AudioSegment.empty()
    parts, part = [], AudioSegment.empty()
    limit = part_minutes * 60 * 1000 if part_minutes else None
    for n, (chunk, para_end) in enumerate(jobs, 1):
        data = synth_retry(chunk, voice, rate, pitch, volume)
        seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        seg += AudioSegment.silent(duration=para_pause if para_end else sent_pause)
        final += seg
        part += seg
        if limit and len(part) >= limit and para_end:
            parts.append(part)
            part = AudioSegment.empty()
        progress.progress(n / total, text=f"{n}/{total} টুকরো তৈরি হয়েছে")
    if len(part) > 0:
        parts.append(part)
    zip_bytes = None
    if limit and len(parts) > 1:
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w") as z:
            for i, p in enumerate(parts, 1):
                z.writestr(f"part_{i:02d}.mp3", to_mp3(mix_music(p, music, music_db)))
        zip_bytes = zb.getvalue()
    mixed = mix_music(final, music, music_db)
    return to_mp3(mixed), zip_bytes, len(mixed) / 1000


st.title("🎙️ বাংলা টেক্সট টু স্পিচ - ডকুমেন্টারি ভয়েস")
st.caption("বড় লেখাও দেওয়া যাবে। অনুচ্ছেদ আলাদা করতে মাঝে একটা ফাঁকা লাইন দিন।")

up = st.file_uploader("চাইলে .txt ফাইল আপলোড করুন (UTF-8)", type=["txt"])
default = up.read().decode("utf-8", errors="ignore") if up else ""
text = st.text_area("এখানে বাংলা লেখা পেস্ট করুন", value=default, height=350,
                    placeholder="আপনার ডকুমেন্টারির স্ক্রিপ্ট এখানে দিন...")
if text.strip():
    st.caption(f"অক্ষর: {len(text)} | অনুচ্ছেদ: {len(split_paragraphs(text))} | আনুমানিক সময়: {len(text) // 900 + 1}-{len(text) // 700 + 1} মিনিট")

preset = st.selectbox("রেডিমেড সেটিং", list(PRESETS.keys()))
p = PRESETS[preset]
if p:
    d_voice, d_pitch, d_rate, d_vol, d_sp, d_pp = p
else:
    d_voice, d_pitch, d_rate, d_vol, d_sp, d_pp = list(VOICES.keys())[1], 0, 0, 0, 400, 1000

voice_name = st.selectbox("ভয়েস", list(VOICES.keys()), index=list(VOICES.keys()).index(d_voice), key=f"v{preset}")
c1, c2 = st.columns(2)
with c1:
    pitch = st.slider("কণ্ঠ/সুর (Hz): মাইনাস = মোটা ও ভারী, প্লাস = চিকন", -50, 50, d_pitch, key=f"p{preset}")
    rate = st.slider("গতি (%): মাইনাস = ধীরে, প্লাস = দ্রুত", -50, 50, d_rate, key=f"r{preset}")
    volume = st.slider("ভলিউম (%)", -50, 50, d_vol, key=f"vol{preset}")
with c2:
    sent_pause = st.slider("বাক্যের মাঝে বিরতি (ms)", 0, 2000, d_sp, step=50, key=f"sp{preset}")
    para_pause = st.slider("অনুচ্ছেদের মাঝে বিরতি (ms)", 0, 4000, d_pp, step=100, key=f"pp{preset}")
    part_minutes = st.slider("আলাদা অংশে ভাগ (প্রতি কত মিনিট, 0 = ভাগ নয়)", 0, 30, 0)

st.subheader("🎵 ব্যাকগ্রাউন্ড মিউজিক")
m1, m2 = st.columns(2)
with m1:
    music_style = st.selectbox("মিউজিক", list(MUSIC_STYLES.keys()), index=1)
with m2:
    music_db = st.slider("মিউজিকের জোর (ভয়েসের চেয়ে কত ডেসিবেল নিচে)", -35, -10, -22,
                         help="সংখ্যা যত ছোট (মাইনাস বেশি), মিউজিক তত হালকা")
music_seg = None
choice = MUSIC_STYLES[music_style]
if choice == "upload":
    mf = st.file_uploader("মিউজিক ফাইল (mp3/wav)", type=["mp3", "wav", "m4a"], key="musicfile")
    if mf:
        music_seg = AudioSegment.from_file(io.BytesIO(mf.read()))
elif choice:
    music_seg = make_ambient(choice)
if music_seg is not None:
    st.audio(to_mp3(music_seg[:12000]), format="audio/mp3")
    st.caption("উপরে মিউজিকের নমুনা। ভয়েসের আগে ২.৫ সেকেন্ড মিউজিক শুরু হবে এবং শেষে ধীরে মিলিয়ে যাবে।")


if st.button("🔊 ভয়েস তৈরি করুন", type="primary"):
    if not text.strip():
        st.warning("আগে কিছু লেখা দিন।")
    else:
        bar = st.progress(0.0, text="শুরু হচ্ছে...")
        try:
            mp3, zp, secs = build(text, VOICES[voice_name], f"{rate:+d}%", f"{pitch:+d}Hz",
                                  f"{volume:+d}%", sent_pause, para_pause, part_minutes, bar,
                                  music_seg, music_db)
            st.session_state["result"] = (mp3, zp, secs)
        except Exception as e:
            st.error(str(e))

if "result" in st.session_state:
    mp3, zp, secs = st.session_state["result"]
    st.success(f"তৈরি! দৈর্ঘ্য: {int(secs // 60)} মিনিট {int(secs % 60)} সেকেন্ড")
    st.audio(mp3, format="audio/mp3")
    st.download_button("⬇️ সম্পূর্ণ MP3 ডাউনলোড", mp3, file_name="documentary_voice.mp3", mime="audio/mpeg")
    if zp:
        st.download_button("⬇️ অংশ অংশ করে ZIP ডাউনলোড", zp, file_name="voice_parts.zip", mime="application/zip")
