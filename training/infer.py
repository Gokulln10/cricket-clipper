"""Inference for the trained audio-event classifier."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import numpy as np

from .features import extract_audio_segment, extract_logmel


@dataclass
class Prediction:
    label: str
    prob: float
    probs: Dict[str, float]


class EventClassifier:
    """Load a checkpoint and classify audio segments."""

    def __init__(self, model_path: str, device: str | None = None):
        import torch
        from .model import AudioCNN

        ckpt = torch.load(model_path, map_location="cpu")
        self.label_names: List[str] = ckpt["label_names"]
        self.sr: int = ckpt["sr"]
        self.duration: float = ckpt["duration"]
        self.n_mels: int = ckpt["n_mels"]
        self.feat_mean: float = ckpt["feat_mean"]
        self.feat_std: float = ckpt["feat_std"]

        self.device = self._pick_device(device)
        self.model = AudioCNN(len(self.label_names), n_mels=self.n_mels)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @staticmethod
    def _pick_device(name: str | None):
        import torch
        if name:
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def predict_audio(self, audio: np.ndarray) -> Prediction:
        import torch

        logmel = extract_logmel(audio, sr=self.sr, n_mels=self.n_mels, duration=self.duration)
        x = (logmel - self.feat_mean) / self.feat_std
        x = torch.from_numpy(x[np.newaxis, np.newaxis, :, :].astype(np.float32)).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
        probs_map = {name: float(p) for name, p in zip(self.label_names, probs)}
        best = int(np.argmax(probs))
        return Prediction(self.label_names[best], float(probs[best]), probs_map)

    def predict_segment(self, video_path: str, start: float, end: float) -> Prediction:
        audio = extract_audio_segment(video_path, start, end, self.sr)
        return self.predict_audio(audio)


@lru_cache(maxsize=2)
def load_classifier(model_path: str, device: str | None = None) -> EventClassifier:
    """Cached loader so the model is only read once per process."""
    return EventClassifier(model_path, device)
