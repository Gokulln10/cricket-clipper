"""Audio feature extraction: waveform -> fixed-size log-mel spectrogram."""
from __future__ import annotations

import subprocess

import numpy as np

# Feature geometry. Keep these in sync between training and inference; they are
# also saved into the model checkpoint so inference can validate them.
DEFAULT_SR = 16000
DEFAULT_N_MELS = 64
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 512
DEFAULT_DURATION = 3.0


def extract_audio_segment(
    path: str, start: float = 0.0, end: float | None = None, sr: int = DEFAULT_SR
) -> np.ndarray:
    """Decode a (segment of a) media file to a mono float32 waveform.

    Works for both audio and video files via ffmpeg. ``end=None`` reads to EOF.
    """
    cmd = ["ffmpeg"]
    if start and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if end is not None and end > start:
        cmd += ["-t", f"{max(0.05, end - start):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _fit_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Centre-crop or zero-pad a waveform to exactly ``target_len`` samples."""
    n = len(audio)
    if n == target_len:
        return audio
    if n > target_len:
        offset = (n - target_len) // 2
        return audio[offset: offset + target_len]
    pad = target_len - n
    left = pad // 2
    return np.pad(audio, (left, pad - left))


def extract_logmel(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    duration: float = DEFAULT_DURATION,
) -> np.ndarray:
    """Return a fixed-shape log-mel spectrogram [n_mels, T] in decibels."""
    import librosa

    target_len = int(duration * sr)
    audio = _fit_length(audio, target_len)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)
