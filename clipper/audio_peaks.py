"""Detect exciting moments from crowd noise.

Boundaries and wickets in cricket are almost always followed by a sustained
spike in crowd / commentary volume. We extract a mono audio track, compute a
short-time RMS energy envelope, and flag windows whose energy stands out
(z-score) from the rest of the match.
"""
from __future__ import annotations

import subprocess
from typing import List, Tuple

import numpy as np


def extract_audio(video_path: str, sr: int = 16000) -> np.ndarray:
    """Decode the video's audio to a mono float32 waveform in [-1, 1]."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def energy_envelope(
    audio: np.ndarray, sr: int, win_seconds: float, hop_seconds: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, rms) for a sliding-window RMS energy envelope (O(n))."""
    win_n = max(1, int(win_seconds * sr))
    hop_n = max(1, int(hop_seconds * sr))
    if len(audio) < win_n:
        return np.array([]), np.array([])

    squared = audio.astype(np.float64) ** 2
    cumsum = np.concatenate([[0.0], np.cumsum(squared)])
    starts = np.arange(0, len(audio) - win_n + 1, hop_n)
    window_sums = cumsum[starts + win_n] - cumsum[starts]
    rms = np.sqrt(window_sums / win_n)
    times = (starts + win_n / 2.0) / sr
    return times, rms


def find_excitement(
    times: np.ndarray,
    rms: np.ndarray,
    sensitivity: float,
    min_event_seconds: float,
) -> List[Tuple[float, float, float]]:
    """Find contiguous high-energy regions.

    Returns a list of (start_time, end_time, score) where score is the peak
    z-score within the region.
    """
    if times.size == 0:
        return []

    # Robust normalisation using median / MAD so a few very loud moments don't
    # swamp the baseline.
    median = np.median(rms)
    mad = np.median(np.abs(rms - median)) + 1e-9
    z = (rms - median) / (1.4826 * mad)

    above = z > sensitivity
    regions: List[Tuple[float, float, float]] = []
    i, n = 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            t0, t1 = float(times[i]), float(times[j - 1])
            if (t1 - t0) >= min_event_seconds:
                regions.append((t0, t1, float(z[i:j].max())))
            i = j
        else:
            i += 1
    return regions
