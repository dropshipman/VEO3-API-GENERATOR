"""
Veo 3 Batch Generator (Gemini API) — Streamlit app

What this does
- Simple batch UI to generate multiple Veo 3 videos via the Gemini API
- Polls long-running operations and lets you download MP4s
- Supports optional image-to-video (upload 1 image used for all prompts)
- Parameters: model (veo-3.0-generate-preview or veo-3.0-fast-generate-preview), aspect ratio, negative prompt, personGeneration (where allowed)

How to run locally
  export GEMINI_API_KEY=...  # from Google AI Studio / Vertex
  pip install streamlit google-genai python-dotenv
  streamlit run app.py

Notes
- Videos are saved to ./outputs with timestamped filenames
- Errors per item are shown inline without stopping the whole batch
- This demo keeps things minimal and avoids unsupported params
"""

import os
import io
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
except Exception as e:
    st.error("Failed to import google-genai SDK. Run: pip install google-genai")
    raise

# ---------- Config ----------
APP_TITLE = "(FONGSTUDIO) GENERATOR VEO 3"
DEFAULT_MODEL = "veo-3.0-generate-preview"
FAST_MODEL = "veo-3.0-fast-generate-preview"
VALID_MODELS = [DEFAULT_MODEL, FAST_MODEL]
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("🔑 API & Model")
    api_key = st.text_input("GEMINI_API_KEY", value=API_KEY, type="password")
    model = st.selectbox("Model", VALID_MODELS, index=0,
                         help="Use the *fast* preview model for quicker & cheaper runs (when available).")

    st.header("🎛️ Parameters")
    aspect = st.selectbox("Aspect ratio", ["16:9", "9:16", "1:1"], index=0)
    negative_prompt = st.text_input("Negative prompt", value="")
    person_generation = st.selectbox(
        "personGeneration (region-dependent)",
        ["allow_all", "allow_adult", "dont_allow"],
        index=0,
        help="Veo 3 text-to-video supports 'allow_all' in many regions; image-to-video may be restricted."
    )

    st.header("🖼️ Optional image → video")
    img_file = st.file_uploader("Upload a PNG/JPEG to animate (optional)", type=["png", "jpg", "jpeg"])    

st.markdown("""
Tempelkan **satu perintah per baris** di bawah. Anda dapat menyertakan isyarat audio, dialog dalam tanda kutip, dan arah bidikan/kamera.
Contoh: `Gang cyberpunk yang murung di malam hari, pergerakan lambat, hujan turun; SFX: guntur jauh; Dialog: \"Kita seharusnya tidak berada di sini.\"`
""")

raw_prompts = st.text_area("Prompts (one per line)", height=180)
prompts = [p.strip() for p in raw_prompts.splitlines() if p.strip()]

col_a, col_b = st.columns([1,1])
with col_a:
    start_btn = st.button("▶️ Generate Batch", type="primary", use_container_width=True)
with col_b:
    clear_btn = st.button("🗑️ Clear status", use_container_width=True)

if clear_btn:
    st.session_state.pop("jobs", None)

if start_btn:
    if not api_key:
        st.error("Please provide GEMINI_API_KEY in the sidebar.")
    elif not prompts:
        st.error("Please enter at least one prompt.")
    else:
        st.session_state.jobs = []
        client = genai.Client(api_key=api_key)
        image_bytes = None
        image_mime = None
        if img_file is not None:
            image_bytes = img_file.read()
            image_mime = "image/png" if img_file.name.lower().endswith(".png") else "image/jpeg"

        for idx, prompt in enumerate(prompts, start=1):
            with st.status(f"Submitting job {idx}/{len(prompts)}", expanded=False):
                cfg = genai_types.GenerateVideosConfig(
                    aspect_ratio=aspect,
                    negative_prompt=negative_prompt or None,
                    person_generation=person_generation or None,
                )

                # Build request
                if image_bytes:
                    image_obj = genai_types.Image(image_bytes=image_bytes, mime_type=image_mime)
                    op = client.models.generate_videos(
                        model=model,
                        prompt=prompt,
                        image=image_obj,
                        config=cfg,
                    )
                else:
                    op = client.models.generate_videos(
                        model=model,
                        prompt=prompt,
                        config=cfg,
                    )

                st.session_state.jobs.append({
                    "prompt": prompt,
                    "operation": op,
                    "status": "submitted",
                    "error": None,
                    "video_path": None,
                })

# ---------- Polling / Results ----------
if "jobs" in st.session_state and st.session_state.jobs:
    client = genai.Client(api_key=api_key or API_KEY or os.getenv("GEMINI_API_KEY", ""))

    for i, job in enumerate(st.session_state.jobs):
        with st.container(border=True):
            st.subheader(f"Job {i+1}")
            st.caption(job["prompt"])            
            try:
                op = job["operation"]
                # keep polling until done or error
                loops = 0
                while not op.done and loops < 999:
                    st.write("⏳ Waiting for video generation… (polling 10s)")
                    time.sleep(10)
                    op = client.operations.get(op)
                    loops += 1

                if not op.done:
                    st.warning("Operation not finished. Try again later or resubmit.")
                    job["status"] = "timeout"
                else:
                    # Success path
                    resp = getattr(op, "response", None) or getattr(op, "result", None)
                    vids = getattr(resp, "generated_videos", []) if resp else []
                    if not vids:
                        raise RuntimeError("No videos returned by the API.")
                    vid = vids[0]

                    # download
                    client.files.download(file=vid.video)
                    # The SDK exposes a helper .save(), but we'll ensure filesystem pathing
                    default_name = f"veo3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.mp4"
                    save_path = OUTPUT_DIR / default_name
                    try:
                        vid.video.save(str(save_path))  # SDK helper
                    except Exception:
                        # fallback: write bytes if present
                        b = getattr(vid.video, "video_bytes", None)
                        if b:
                            with open(save_path, "wb") as f:
                                f.write(b)
                        else:
                            raise

                    job["video_path"] = str(save_path)
                    job["status"] = "done"
                    st.success("✅ Done")
                    st.video(str(save_path))
                    with open(save_path, "rb") as f:
                        st.download_button("Download MP4", data=f, file_name=save_path.name, mime="video/mp4")
            except Exception as e:
                job["status"] = "error"
                job["error"] = str(e)
                st.error(f"Error: {e}")

st.markdown("""
---
**Tips**
- Bersikap eksplisit dengan *gerakan kamera* (dolly-in, crane, handheld), *pencahayaan*, *suasana hati*, *tempo*, *ketukan aksi*, dan isyarat *audio* (Dialog, SFX, Ambience).
- Perlu diingat Veo 3 saat ini menargetkan klip ~8 detik pada 24fps 720p. Waktu proses yang lebih lama atau resolusi yang lebih tinggi belum diekspos.
- Video yang dihasilkan dibatasi waktu di server; selalu unduh hasil Anda segera.
""")
