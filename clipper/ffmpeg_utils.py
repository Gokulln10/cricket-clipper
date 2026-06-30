"""Small ffmpeg/ffprobe helpers shared across the package."""
from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from typing import List

# Preferred hardware H.264 encoders per platform, best-effort order.
_HW_ENCODERS = ["h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_amf"]


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


@lru_cache(maxsize=1)
def available_encoders() -> frozenset:
    """Return the set of encoder names this ffmpeg build supports."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, text=True,
        )
    except Exception:
        return frozenset()
    names = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        # Lines look like: " V....D h264_nvenc  NVIDIA NVENC H.264 encoder"
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


def resolve_encoder(encoder: str = "auto") -> str:
    """Resolve a Settings.encoder value to a concrete ffmpeg encoder name.

    ``auto`` picks the first available hardware encoder, else libx264.
    Returns ``libx264`` if the requested hardware encoder is unavailable.
    """
    have = available_encoders()
    if encoder in ("cpu", "libx264", "x264"):
        return "libx264"
    if encoder == "auto":
        for enc in _HW_ENCODERS:
            if enc in have:
                return enc
        return "libx264"
    # Explicit request: honour it if present, else fall back gracefully.
    return encoder if encoder in have else "libx264"


def encoder_args(encoder_name: str) -> List[str]:
    """ffmpeg quality/speed args for a given video encoder."""
    if encoder_name == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    if encoder_name == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    if encoder_name == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", "6M"]
    if encoder_name == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", "20"]
    if encoder_name == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"]
    # Unknown: pass through as a codec name.
    return ["-c:v", encoder_name]

