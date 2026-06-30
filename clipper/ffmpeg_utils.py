"""Small ffmpeg/ffprobe helpers shared across the package."""
from __future__ import annotations

import json
import shutil
import subprocess


def ensure_ffmpeg() -> None:
    """Raise a helpful error if ffmpeg/ffprobe are not installed."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg (e.g. `brew install ffmpeg`)."
        )


def probe_duration(video_path: str) -> float:
    """Return the duration of a media file in seconds."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", video_path,
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True,
    )
    data = json.loads(out.stdout or "{}")
    try:
        return float(data["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return 0.0
