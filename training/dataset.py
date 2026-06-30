"""Dataset loading for the audio-event classifier.

Two labelling layouts are supported:

1. **Folder layout** — one sub-folder per class, each containing audio/video
   clips of that event::

       data/
         four/    clip001.mp4  clip002.wav ...
         six/     ...
         wicket/  ...
         negative/ ...      # non-events (overs, gaps) -- recommended

2. **CSV layout** — a CSV with segment annotations against full videos::

       video,start,end,label
       match1.mp4,123.5,126.0,four
       match1.mp4,540.0,543.0,six
       match1.mp4,900.0,903.0,negative

Each sample is converted to a fixed-size log-mel spectrogram up front (datasets
are typically small), so training epochs are fast.
"""
from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np

from .features import (
    DEFAULT_DURATION,
    DEFAULT_N_MELS,
    DEFAULT_SR,
    extract_audio_segment,
    extract_logmel,
)

_MEDIA_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
               ".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def _iter_folder_samples(root: str) -> List[Tuple[str, float, float | None, str]]:
    samples: List[Tuple[str, float, float | None, str]] = []
    for label in sorted(os.listdir(root)):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        for name in sorted(os.listdir(label_dir)):
            if os.path.splitext(name)[1].lower() in _MEDIA_EXTS:
                samples.append((os.path.join(label_dir, name), 0.0, None, label))
    return samples


def _iter_csv_samples(csv_path: str) -> List[Tuple[str, float, float | None, str]]:
    base = os.path.dirname(os.path.abspath(csv_path))
    samples: List[Tuple[str, float, float | None, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row["video"]
            if not os.path.isabs(video):
                video = os.path.join(base, video)
            start = float(row.get("start", 0.0) or 0.0)
            end_raw = row.get("end", "")
            end = float(end_raw) if end_raw not in ("", None) else None
            samples.append((video, start, end, row["label"].strip()))
    return samples


def iter_samples(data_path: str) -> List[Tuple[str, float, float | None, str]]:
    """Return raw (path, start, end, label) samples without featurising them."""
    if os.path.isdir(data_path):
        return _iter_folder_samples(data_path)
    if data_path.lower().endswith(".csv"):
        return _iter_csv_samples(data_path)
    raise ValueError(f"Unsupported data path: {data_path}")


def load_dataset(
    data_path: str,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    n_mels: int = DEFAULT_N_MELS,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and featurise a dataset.

    Returns (X, y, label_names) where X has shape [N, 1, n_mels, T] and y holds
    integer class indices into ``label_names``.
    """
    if os.path.isdir(data_path):
        raw = _iter_folder_samples(data_path)
    elif data_path.lower().endswith(".csv"):
        raw = _iter_csv_samples(data_path)
    else:
        raise ValueError(f"Unsupported data path: {data_path}")

    if not raw:
        raise ValueError(f"No samples found under {data_path}")

    label_names = sorted({label for *_rest, label in raw})
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    feats: List[np.ndarray] = []
    labels: List[int] = []
    for path, start, end, label in raw:
        try:
            audio = extract_audio_segment(path, start, end, sr)
        except Exception as exc:  # skip unreadable samples but keep going
            print(f"  ! skipping {path} ({exc})")
            continue
        logmel = extract_logmel(audio, sr=sr, n_mels=n_mels, duration=duration)
        feats.append(logmel[np.newaxis, :, :])  # add channel dim
        labels.append(label_to_idx[label])

    if not feats:
        raise ValueError("All samples failed to load.")

    X = np.stack(feats).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, label_names
