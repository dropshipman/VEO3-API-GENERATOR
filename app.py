"""
Veo 3 Batch Generator (Gemini API) — Streamlit app

What this does
- Simple batch UI to generate multiple Veo 3/Veo 2 videos via the Gemini API
- Polls long-running operations and lets you download MP4s
- Optional image-to-video (one image used for all prompts)
- Parameters: model, aspect ratio, negative prompt, personGeneration (region-aware)
- Extra: safer defaults + clearer error diagnostics
- NEW: Auto-convert Veo 3 16:9 → 9:16 (center crop + upscale)
- NEW: Show preview of uploaded still image

How to run locally
  export GEMINI_API_KEY=...  # from Google AI Studio (paid) / Vertex
  pip install streamlit google-genai python-dotenv moviepy imageio-ffmpeg
  streamlit run app.py

Deploy notes (Streamlit Cloud)
- Add a **requirements.txt** containing:  
  `streamlit
google-genai
python-dotenv
moviepy
imageio-ffmpeg`
- (Recommended) Add **runtime.txt** with `3.11` to avoid Python 3.13 compatibility issues with moviepy.

Notes
- Videos are saved to ./outputs with timestamped filenames
- Errors per item are shown inline without stopping the whole batch
- This demo keeps things minimal and avoids unsupported params
"""

import os
import io
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Google GenAI SDK
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

# Video post-process
# Try optional dependency for post-process (portrait conversion)
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

# ---------- Config ----------
APP_TITLE = "Veo 3 Batch Generator (Gemini API)"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model names (as of 2025-08)
VEO3 = "veo-3.0-generate-preview"
VEO3_FAST = "veo-3.0-fast-generate-preview"
VEO2 = "veo-2.0-generate-001"  # optional fallback (silent)
VALID_MODELS = [VEO3, VEO3_FAST, VEO2]

# ---------- Helpers ----------
def resolve_person_generation(model: str, has_image: bool, requested: str, region_mode: str) -> Optional[str]:
    """Return a person_generation value that matches Veo rules & region limits.
    Rules summary (AI Docs, Aug 2025):
      • Veo 3 (Preview) text→video: personGeneration = "allow_all"; in EU/UK/CH/MENA must be "allow_adult".
      • Veo 3 (Preview) image→video: personGeneration = "allow_adult" only.
      • Veo 2 (Stable): text→video {allow_all|allow_adult|dont_allow}; image→video {allow_adult|dont_allow}.
    """
    region_is_restricted = region_mode == "EU/UK/CH/MENA"

    if model in (VEO3, VEO3_FAST):
        if has_image:
            return "allow_adult"
        # text-only
        return "allow_adult" if region_is_restricted else "allow_all"

    # Veo 2
    if has_image:
        return requested if requested in ("allow_adult", "dont_allow") else "allow_adult"
    return requested if requested in ("allow_all", "allow_adult", "dont_allow") else "allow_adult"


def coerce_aspect_ratio(model: str, chosen: str) -> str:
    """Force allowed aspect ratios per model."""
    if model in (VEO3, VEO3_FAST):
        # Veo 3 Preview currently documents only "16:9".
        return "16:9"
    # Veo 2 supports 16:9 and 9:16; keep user's choice if valid.
    return chosen if chosen in ("16:9", "9:16") else "16:9"


def show_api_error(e: Exception):
    """Pretty-print Gemini API errors without letting Streamlit redact the message."""
    if isinstance(e, genai_errors.ClientError):
        code = getattr(e, "status_code", "?")
        body = getattr(e, "response", None)
        st.error(f"Gemini API ClientError (HTTP {code}). Lihat detail di bawah.")
        if body:
            try:
                import json
                st.code(json.dumps(body, indent=2)[:4000], language="json")
            except Exception:
                st.write(body)
        else:
            st.write(str(e))
    else:
        st.error(str(e))


def ar_string(w: int, h: int) -> str:
    g = math.gcd(w, h)
    return f"{w//g}:{h//g}"


def convert_16_9_to_9_16(input_path: str, out_height: int = 1280) -> str:
    """Center-crop a 16:9 video to 9:16 then upscale to target height (keeps audio)."""
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy belum terpasang. Tambahkan 'moviepy' & 'imageio-ffmpeg' ke requirements.txt.")
    clip = VideoFileClip(input_path)
    w, h = clip.size  # (width, height)
    # If already portrait-ish, just return same path
    if w * 16 <= h * 9:
        clip.close()
        return input_path

    # Compute crop box: width = h * 9/16, full height
    crop_w = int(h * 9 / 16)
    x1 = max(0, (w - crop_w) // 2)
    x2 = x1 + crop_w
    cropped = clip.crop(x1=x1, y1=0, x2=x2, y2=h)

    # Resize to target portrait resolution (e.g., 720x1280)
    target_w = int(out_height * 9 / 16)
    resized = cropped.resize(newsize=(target_w, out_height))

    # Output path
    stem = Path(input_path).stem
    out_path = OUTPUT_DIR / f"{stem}_9x16.mp4"

    # Preserve fps & audio if any
    fps = clip.fps or 24
    resized.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        preset="medium",
        threads=2,
        verbose=False,
        logger=None,
    )
    clip.close()
    resized.close()
    return str(out_path)


# ---------- UI ----------
load_dotenv()
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("🔑 API & Model")
    api_key = st.text_input("GEMINI_API_KEY", value=DEFAULT_API_KEY, type="password")
    model = ui_select("Model", VALID_MODELS, index=0,
                         help="*Fast* preview lebih cepat/hemat (jika tersedia). Veo 2 tidak ada audio."). Veo 2 tidak ada audio.")

    st.header("🎛️ Parameters")
    aspect_ui = ui_select("Aspect ratio", ["16:9", "9:16", "1:1"], index=0,
                             help="Veo 3 Preview saat ini mendukung 16:9 saja; opsi lain akan dipaksa 16:9.")
    negative_prompt = st.text_input("Negative prompt", value="")

    region_mode = ui_select(
        "Regional compliance",
        ["Default/Global", "EU/UK/CH/MENA"],
        index=0,
        help="Jika audiens kamu di wilayah terbatas, pilih EU/UK/CH/MENA agar parameter people sesuai kebijakan."
    )
    person_choice = ui_select(
        "Requested personGeneration",
        ["allow_all", "allow_adult", "dont_allow"],
        index=0,
        help="Akan disesuaikan otomatis sesuai model/mode & region."
    )

    st.header("🖼️ Optional image → video")
    img_file = st.file_uploader("Upload PNG/JPEG (opsional)", type=["png", "jpg", "jpeg"])    
    if img_file is not None:
        st.image(img_file, caption="Preview sumber (image→video)", use_column_width=True)

    st.header("📱 Post-process")
    if MOVIEPY_AVAILABLE:
        auto_to_vertical = st.checkbox(
            "Auto-convert output ke 9:16 (center crop + upscale)",
            value=True,
            help="Untuk Veo 3 yang 16:9, hasil akan dicrop tengah jadi portrait 9:16 (contoh 720×1280)."
        )
        target_h = st.select_slider(
            "Tinggi output 9:16",
            options=[960, 1280, 1440],
            value=1280,
            help="1280 cocok untuk vertical 720×1280."
        )
    else:
        st.info("""Post-process 9:16 nonaktif karena 'moviepy' belum terpasang. Tambahkan ke requirements.txt:
- moviepy
- imageio-ffmpeg""")
        auto_to_vertical = False
        target_h = 1280

st.markdown(
    """
Paste **satu prompt per baris**. Bisa sertakan audio cues, dialog dalam tanda kutip, dan arahan kamera.
Contoh: `Gang cyberpunk yang murung di malam hari, dolly-in perlahan, hujan; SFX: guntur jauh; Dialog: \"Kita seharusnya tidak berada di sini.\"`
    """
)

raw_prompts = st.text_area("Prompts (one per line)", height=180)
prompts = [p.strip() for p in raw_prompts.splitlines() if p.strip()]

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    start_btn = st.button("▶️ Generate Batch", type="primary", use_container_width=True)
with col_b:
    test_btn = st.button("🧪 Coba prompt aman", use_container_width=True)
with col_c:
    clear_btn = st.button("🗑️ Clear status", use_container_width=True)

if test_btn:
    prompts = [
        "Wide aerial of rolling green hills at sunrise, soft fog, gentle camera pan; Ambience: birdsong",
    ]

if clear_btn:
    st.session_state.pop("jobs", None)

# ---------- Submit ----------
if start_btn:
    if not api_key:
        st.error("Isi GEMINI_API_KEY di sidebar atau Secrets.")
    elif not prompts:
        st.error("Masukkan minimal satu prompt.")
    else:
        st.session_state.jobs = []
        client = genai.Client(api_key=api_key)

        image_bytes = img_file.read() if img_file is not None else None
        image_mime = None
        if img_file is not None:
            image_mime = "image/png" if img_file.name.lower().endswith(".png") else "image/jpeg"

        # Enforce allowed parameters per model/mode
        has_img = image_bytes is not None
        aspect = coerce_aspect_ratio(model, aspect_ui)
        effective_person = resolve_person_generation(
            model, has_img, person_choice,
            region_mode="EU/UK/CH/MENA" if region_mode.endswith("MENA") else "Default"
        )

        if aspect != aspect_ui and model in (VEO3, VEO3_FAST):
            st.warning("Veo 3 Preview saat ini hanya 16:9. Aspect ratio diubah otomatis ke 16:9.")

        for idx, prompt in enumerate(prompts, start=1):
            with st.status(f"Submitting job {idx}/{len(prompts)}", expanded=False):
                try:
                    cfg = genai_types.GenerateVideosConfig(
                        aspect_ratio=aspect,
                        negative_prompt=negative_prompt or None,
                        person_generation=effective_person,
                    )

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
                        "operation_name": op.name,
                        "status": "submitted",
                        "error": None,
                        "video_path": None,
                    })
                except Exception as e:
                    show_api_error(e)

# ---------- Polling / Results ----------
if "jobs" in st.session_state and st.session_state.jobs:
    client = genai.Client(api_key=api_key or DEFAULT_API_KEY or os.getenv("GEMINI_API_KEY", ""))

    for i, job in enumerate(st.session_state.jobs):
        with st.container(border=True):
            st.subheader(f"Job {i+1}")
            st.caption(job["prompt"])            
            if img_file is not None:
                st.image(img_file, caption="Preview sumber (image→video)", width=220)
            st.write(f"Operation: `{job.get('operation_name','?')}`")
            try:
                op = genai_types.GenerateVideosOperation(name=job["operation_name"]) if job.get("operation_name") else None
                loops = 0
                while op and not op.done and loops < 180:
                    st.write("⏳ Menunggu video… (poll 10s)")
                    time.sleep(10)
                    op = client.operations.get(op)
                    loops += 1

                if not op or not op.done:
                    st.warning("Operasi belum selesai. Coba refresh/poll lagi nanti.")
                else:
                    resp = getattr(op, "response", None) or getattr(op, "result", None)
                    vids = getattr(resp, "generated_videos", []) if resp else []
                    if not vids:
                        raise RuntimeError("API sukses tapi tidak ada video di response.")
                    vid = vids[0]

                    # Download video
                    client.files.download(file=vid.video)
                    default_name = f"veo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.mp4"
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

                    st.success("✅ Selesai (16:9)")
                    st.video(str(save_path))
                    col1, col2 = st.columns([1,1])
                    with col1:
                        with open(save_path, "rb") as f:
                            st.download_button("Download MP4 (16:9)", data=f, file_name=save_path.name, mime="video/mp4")

                    # Auto convert to 9:16 if requested or if user chose 9:16 but we coerced
                    need_vertical = auto_to_vertical and MOVIEPY_AVAILABLE and (model in (VEO3, VEO3_FAST))
                    if need_vertical:
                        try:
                            st.info("📐 Mengonversi ke 9:16 (center crop + upscale)…")
                            out_v = convert_16_9_to_9_16(str(save_path), out_height=target_h)
                            st.success("✅ Selesai (9:16)")
                            st.video(out_v)
                            with col2:
                                with open(out_v, "rb") as f:
                                    st.download_button("Download MP4 (9:16)", data=f, file_name=Path(out_v).name, mime="video/mp4")
                            # Show before/after sizes
                            try:
                                if MOVIEPY_AVAILABLE:
                                    v = VideoFileClip(out_v)
                                else:
                                    v = None
                                if v is not None:
                                    st.caption(f"Converted size: {v.w}x{v.h} ({ar_string(v.w, v.h)})")
                                    v.close()
                            except Exception:
                                pass
                        except Exception as e:
                            st.warning("Konversi 9:16 gagal. Lihat detail error di bawah.")
                            st.exception(e)

            except Exception as e:
                show_api_error(e)

st.markdown(
    """
---
**Tips**
- Tambahkan *camera moves*, *lighting*, *mood*, *tempo*, *audio cues* (Dialogue, SFX, Ambience).
- Veo 3 Preview menargetkan ~8s @ 24fps 720p (16:9). Opsi 9:16 dilakukan via post-process.
- Person generation tunduk batasan regional; untuk EU/UK/CH/MENA gunakan "allow_adult". Image→video di Veo 3: wajib "allow_adult".
- Hasil di server dibersihkan ±2 hari; selalu download.
- Konversi 9:16 di sini memakai crop tengah + upscale. Untuk framing dinamis/subjek bergerak, pertimbangkan tool reframing berbasis tracking.
    """
)
