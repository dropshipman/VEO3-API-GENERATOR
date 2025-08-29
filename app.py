"""
Veo 3 Batch Generator (Gemini API) — Streamlit app (MINIMAL)

What this does
- Simple UI to generate multiple Veo 3 videos via the Gemini API
- Polls long-running operations and lets you download MP4s
- Optional image-to-video (one image used for all prompts)
- Parameters: model (veo-3.0-generate-preview or veo-3.0-fast-generate-preview), aspect ratio, negative prompt, personGeneration (where allowed)

How to run locally
  export GEMINI_API_KEY=...  # from Google AI Studio / Vertex
  pip install streamlit google-genai python-dotenv
  streamlit run app.py

Notes
- Videos are saved to ./outputs with timestamped filenames
- Errors per item are shown inline without stopping the whole batch
- Veo 3 preview currently supports 16:9; this app will auto-force 16:9 if needed
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    st.error("Failed to import google-genai SDK. Run: pip install google-genai")
    raise

# ---------- Config ----------
APP_TITLE = "Veo 3 Batch Generator (Gemini API)"
DEFAULT_MODEL = "veo-3.0-generate-preview"
FAST_MODEL = "veo-3.0-fast-generate-preview"
VEO2_MODEL = "veo-2.0-generate-001"
VALID_MODELS = [DEFAULT_MODEL, FAST_MODEL, VEO2_MODEL]
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
    model = st.selectbox(
        "Model",
        VALID_MODELS,
        index=0,
        help="Use the *fast* preview model for quicker & cheaper runs (when available). Veo 2 is silent (no audio)."
    )

    st.header("🎛️ Parameters")
    aspect = st.selectbox("Aspect ratio", ["16:9", "9:16", "1:1"], index=0)
    negative_prompt = st.text_input("Negative prompt", value="")
    person_generation = st.selectbox(
        "personGeneration (region-dependent)",
        ["allow_all", "allow_adult", "dont_allow"],
        index=0,
        help=(
            "Veo 3 text-to-video supports 'allow_all' in many regions; "
            "image-to-video requires 'allow_adult' in most regions."
        ),
    )

    st.header("🖼️ Optional image → video")
    img_file = st.file_uploader("Upload a PNG/JPEG to animate (optional)", type=["png", "jpg", "jpeg"])    

st.markdown(
    """
Paste **one prompt per line** below. You can include audio cues, dialogue in quotes, and shot/camera directions.
Example: `A moody cyberpunk alleyway at night, slow dolly-in, rain falling; SFX: distant thunder; Dialogue: \"We shouldn't be here.\"`
"""
)

raw_prompts = st.text_area("Prompts (one per line)", height=180)
prompts = [p.strip() for p in raw_prompts.splitlines() if p.strip()]

col_a, col_b = st.columns([1, 1])
with col_a:
    start_btn = st.button("▶️ Generate Batch", type="primary", use_container_width=True)
with col_b:
    clear_btn = st.button("🗑️ Clear status", use_container_width=True)

if clear_btn:
    st.session_state.pop("jobs", None)

# ---------- Submit ----------
if start_btn:
    if not api_key:
        st.error("Please provide GEMINI_API_KEY in the sidebar.")
    elif not prompts:
        st.error("Please enter at least one prompt.")
    else:
        st.session_state.jobs = []
        client = genai.Client(api_key=api_key)

        image_bytes: Optional[bytes] = None
        image_mime: Optional[str] = None
        if img_file is not None:
            image_bytes = img_file.read()
            image_mime = "image/png" if img_file.name.lower().endswith(".png") else "image/jpeg"

        # Guardrails for Veo 3 & Veo 2: enforce aspect + person rules
        aspect_param = aspect
        if model in (DEFAULT_MODEL, FAST_MODEL):
            if aspect != "16:9":
                st.warning("Veo 3 currently supports 16:9. Forcing aspect to 16:9.")
                aspect_param = "16:9"
        elif model == VEO2_MODEL:
            if aspect not in ("16:9", "9:16"):
                st.warning("Veo 2 supports 16:9 and 9:16. Forcing aspect to 16:9.")
                aspect_param = "16:9"

        person_effective = person_generation
        if model in (DEFAULT_MODEL, FAST_MODEL):
            if image_bytes is not None and person_generation != "allow_adult":
                st.warning("Veo 3 image→video requires 'allow_adult'. Overriding.")
                person_effective = "allow_adult"
        elif model == VEO2_MODEL:
            if image_bytes is not None and person_generation == "allow_all":
                st.warning("Veo 2 image→video requires 'allow_adult' or 'dont_allow'. Overriding to 'allow_adult'.")
                person_effective = "allow_adult"

        for idx, prompt in enumerate(prompts, start=1):
            with st.status(f"Submitting job {idx}/{len(prompts)}", expanded=False):
                cfg = genai_types.GenerateVideosConfig(
                    aspect_ratio=aspect_param,
                    negative_prompt=negative_prompt or None,
                    person_generation=person_effective or None,
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
                while not getattr(op, "done", False) and loops < 999:
                    st.write("⏳ Waiting for video generation… (polling 10s)")
                    time.sleep(10)
                    op = client.operations.get(op)
                    loops += 1

                if not getattr(op, "done", False):
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
                    default_name = f"veo3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.mp4"
                    save_path = OUTPUT_DIR / default_name
                    try:
                        vid.video.save(str(save_path))
                    except Exception:
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

st.markdown(
    """
---
**Tips**
- Be explicit with *camera moves* (dolly-in, crane, handheld), *lighting*, *mood*, *tempo*, *action beats*, and *audio* cues (Dialogue, SFX, Ambience).
- Keep in mind Veo 3 currently targets ~8s clips at 24fps 720p. Longer runtimes or higher resolutions are not yet exposed.
- Generated videos are time-limited on the server; always download your results promptly.
- For Veo 3 + image→video, use personGeneration = 'allow_adult'.
- For Veo 3 aspect ratio, 16:9 only (this app auto-forces 16:9 when needed).
- Veo 2 supports 16:9 and 9:16, but outputs silent video (no audio).
"""
)
