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


def _local_baseline(
    rms: np.ndarray, hop_seconds: float, window_seconds: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling median / MAD baseline so detection adapts to local loudness.

    Returns (median, mad) arrays the same length as ``rms``. With
    ``window_seconds <= 0`` it falls back to a single global baseline.
    """
    n = len(rms)
    if window_seconds <= 0 or n == 0:
        med = np.full(n, np.median(rms) if n else 0.0)
        mad = np.full(n, (np.median(np.abs(rms - med)) if n else 0.0) + 1e-9)
        return med, mad

    win = max(1, int(window_seconds / hop_seconds))
    half = win // 2
    step = max(1, win // 10)
    grid = np.arange(0, n, step)
    med_g = np.empty(len(grid))
    mad_g = np.empty(len(grid))
    for k, i in enumerate(grid):
        seg = rms[max(0, i - half): min(n, i + half)]
        m = np.median(seg)
        med_g[k] = m
        mad_g[k] = np.median(np.abs(seg - m)) + 1e-9
    full = np.arange(n)
    med = np.interp(full, grid, med_g)
    mad = np.interp(full, grid, mad_g)
    return med, mad


def _regions_from_mask(
    times: np.ndarray, score: np.ndarray, mask: np.ndarray, min_event_seconds: float
) -> List[Tuple[float, float, float]]:
    regions: List[Tuple[float, float, float]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            t0, t1 = float(times[i]), float(times[j - 1])
            if (t1 - t0) >= min_event_seconds:
                regions.append((t0, t1, float(score[i:j].max())))
            i = j
        else:
            i += 1
    return regions


def find_excitement(
    times: np.ndarray,
    rms: np.ndarray,
    sensitivity: float,
    min_event_seconds: float,
    hop_seconds: float = 0.1,
    local_baseline_seconds: float = 45.0,
    use_onset: bool = True,
    onset_sensitivity: float = 2.5,
) -> List[Tuple[float, float, float]]:
    """Find exciting audio regions using adaptive level + onset detection.

    Two complementary detectors are combined to improve recall:

    * **Level** — energy that stands out from a *local* rolling baseline, so
      events are caught even during generally loud passages.
    * **Onset** — a sudden rise in loudness (positive derivative), which catches
      fast reactions that never reach the absolute peak level.

    Returns a list of (start_time, end_time, score).
    """
    if times.size == 0:
        return []

    median, mad = _local_baseline(rms, hop_seconds, local_baseline_seconds)
    z_level = (rms - median) / (1.4826 * mad)
    level_mask = z_level > sensitivity
    regions = _regions_from_mask(times, z_level, level_mask, min_event_seconds)

    if use_onset:
        # Smooth slightly, then look at positive change (rise) in loudness.
        kernel = max(1, int(0.3 / hop_seconds))
        smooth = np.convolve(rms, np.ones(kernel) / kernel, mode="same")
        delta = np.diff(smooth, prepend=smooth[:1])
        delta[delta < 0] = 0.0
        d_med = np.median(delta)
        d_mad = np.median(np.abs(delta - d_med)) + 1e-9
        z_onset = (delta - d_med) / (1.4826 * d_mad)
        onset_mask = z_onset > onset_sensitivity
        # Onsets are sharp; allow zero-length events (a single spike counts).
        regions += _regions_from_mask(times, z_onset, onset_mask, 0.0)

    regions.sort(key=lambda r: r[0])
    return regions

