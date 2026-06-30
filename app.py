"""Streamlit web app for the cricket highlight clipper.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile

import streamlit as st

from clipper.config import Settings
from clipper.ffmpeg_utils import ensure_ffmpeg
from clipper.pipeline import build_highlights, export_clips

st.set_page_config(page_title="Cricket Highlight Clipper", page_icon="🏏", layout="wide")

st.title("🏏 Cricket Highlight Clipper")
st.caption(
    "Detects exciting moments from crowd noise, snaps to scene cuts, and ranks "
    "by on-screen motion — then exports clips."
)

# --- ffmpeg check ---
try:
    ensure_ffmpeg()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")

    sensitivity = st.slider(
        "Sensitivity", 0.8, 3.0, 1.6, 0.1,
        help="Lower = more clips (catches quieter reactions). Higher = only the loudest moments.",
    )
    pre = st.slider("Lead-in before reaction (s)", 0.0, 15.0, 6.0, 0.5)
    post = st.slider("Tail after reaction (s)", 0.0, 15.0, 4.0, 0.5)
    max_clips = st.number_input(
        "Max clips (0 = all)", min_value=0, max_value=200, value=0, step=1
    )

    with st.expander("Advanced"):
        min_clip = st.slider("Min clip length (s)", 2.0, 20.0, 4.0, 0.5)
        max_clip = st.slider("Max clip length (s)", 8.0, 60.0, 25.0, 1.0)
        merge_gap = st.slider("Merge clips closer than (s)", 0.0, 15.0, 3.0, 0.5)
        use_scene = st.checkbox("Snap to scene cuts", value=True)
        use_motion = st.checkbox("Score by motion", value=True)
        accurate = st.checkbox("Frame-accurate cuts (slower)", value=True)


def make_settings() -> Settings:
    return Settings(
        sensitivity=sensitivity,
        pre_seconds=pre,
        post_seconds=post,
        max_clips=int(max_clips),
        min_clip_seconds=min_clip,
        max_clip_seconds=max_clip,
        merge_gap_seconds=merge_gap,
        use_scene_snap=use_scene,
        use_event_score=use_motion,
        accurate_cut=accurate,
        output_dir=st.session_state.get("output_dir", "clips"),
    )


# --- Input selection ---
st.subheader("1. Choose a video")
tab_path, tab_upload = st.tabs(["📁 Local file path", "⬆️ Upload"])

video_path = None
with tab_path:
    path_input = st.text_input(
        "Absolute path to a cricket video",
        placeholder="/Users/you/Videos/match.mp4",
    )
    if path_input:
        if os.path.isfile(path_input):
            video_path = path_input
        else:
            st.warning("File not found at that path.")

with tab_upload:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi", "m4v"])
    if uploaded is not None:
        tmp_dir = tempfile.mkdtemp(prefix="cricket_")
        tmp_path = os.path.join(tmp_dir, uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        video_path = tmp_path

out_dir = st.text_input("Output folder for clips", value="clips")
st.session_state["output_dir"] = out_dir

# --- Run ---
st.subheader("2. Find & export highlights")
if st.button("🚀 Process video", type="primary", disabled=video_path is None):
    settings = make_settings()
    progress_bar = st.progress(0.0, text="Starting...")

    def on_progress(frac: float, msg: str) -> None:
        progress_bar.progress(frac, text=msg)

    try:
        clips = build_highlights(video_path, settings, on_progress)
    except Exception as exc:  # surface ffmpeg/processing errors to the user
        st.error(f"Processing failed: {exc}")
        st.stop()

    if not clips:
        st.warning("No highlights detected. Try lowering the sensitivity.")
        st.stop()

    progress_bar.progress(0.0, text="Exporting clips...")
    clips = export_clips(video_path, clips, settings, on_progress)
    progress_bar.empty()

    st.session_state["clips"] = clips
    st.success(f"Exported {len(clips)} clip(s) to '{settings.output_dir}'.")

# --- Results ---
clips = st.session_state.get("clips")
if clips:
    st.subheader("3. Clips")
    cols = st.columns(2)
    for i, c in enumerate(clips):
        col = cols[i % 2]
        with col:
            st.markdown(
                f"**Clip {i + 1}** · {c.start:.1f}s → {c.end:.1f}s "
                f"({c.duration:.1f}s) · score {c.score:.2f} · {', '.join(c.reasons)}"
            )
            if c.out_path and os.path.isfile(c.out_path):
                st.video(c.out_path)
                with open(c.out_path, "rb") as f:
                    st.download_button(
                        "Download",
                        f.read(),
                        file_name=os.path.basename(c.out_path),
                        mime="video/mp4",
                        key=f"dl_{i}",
                    )
