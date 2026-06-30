"""Optional audio-event classifier integration for the clipper.

Wraps the model trained under ``training/`` so the pipeline can label each
candidate clip (e.g. ``four``/``six``/``wicket``) and boost confident events.
Degrades to a no-op if the model file or ML dependencies are unavailable.
"""
from __future__ import annotations

import os
from typing import Optional

# Labels considered "positive" events worth boosting in the ranking. Any label
# containing these substrings counts; everything else (e.g. "negative",
# "other") is treated as a non-event.
POSITIVE_HINTS = ("four", "six", "boundary", "wicket", "out", "fifty", "hundred", "appeal")


def is_available(model_path: str) -> bool:
    if not model_path or not os.path.isfile(model_path):
        return False
    try:
        import torch  # noqa: F401
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


def is_positive(label: str) -> bool:
    label = (label or "").lower()
    return any(h in label for h in POSITIVE_HINTS)


def load(model_path: str, device: str | None = None):
    """Return an EventClassifier or None if unavailable."""
    if not is_available(model_path):
        return None
    try:
        from training.infer import load_classifier
        return load_classifier(model_path, device or None)
    except Exception:
        return None
