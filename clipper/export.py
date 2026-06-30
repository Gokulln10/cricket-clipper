"""Export clips from the source video with ffmpeg."""
from __future__ import annotations

import subprocess

from .ffmpeg_utils import encoder_args, resolve_encoder


def export_clip(
    video_path: str,
    start: float,
    end: float,
    out_path: str,
    accurate: bool = True,
    encoder: str = "auto",
) -> str:
    """Write the segment [start, end] of ``video_path`` to ``out_path``.

    accurate=True  -> re-encode (frame-accurate cuts, slower).
    accurate=False -> stream copy (fast, but cuts land on keyframes).
    encoder        -> "auto"/"cpu"/explicit ffmpeg encoder (e.g. h264_nvenc).

    Returns the concrete video encoder used ("copy" when stream-copying).
    """
    duration = max(0.1, end - start)

    if accurate:
        enc_name = resolve_encoder(encoder)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
            *encoder_args(enc_name),
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            out_path,
        ]
    else:
        enc_name = "copy"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    return enc_name

