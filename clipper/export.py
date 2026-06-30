"""Export clips from the source video with ffmpeg."""
from __future__ import annotations

import subprocess


def export_clip(
    video_path: str,
    start: float,
    end: float,
    out_path: str,
    accurate: bool = True,
) -> None:
    """Write the segment [start, end] of ``video_path`` to ``out_path``.

    accurate=True  -> re-encode (frame-accurate cuts, slower).
    accurate=False -> stream copy (fast, but cuts land on keyframes).
    """
    duration = max(0.1, end - start)

    if accurate:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
