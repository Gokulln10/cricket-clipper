"""Cricket player (person) and ball detection via YOLO (optional).

Uses the Ultralytics YOLO model to count people on the field and spot the ball
inside each candidate clip. People density and ball presence are strong cues
that real play is happening, which helps rank clips and recover events that the
audio detector alone might miss.

This is an OPTIONAL feature. It requires ``ultralytics`` (which pulls in
PyTorch):

    pip install -r requirements-ml.txt

If the package or model is unavailable the functions degrade to a no-op so the
rest of the pipeline keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import numpy as np

# COCO class indices used by the default YOLO models.
_PERSON_CLASS = 0
_BALL_CLASS = 32  # "sports ball"


@dataclass
class PlayerStats:
    avg_players: float = 0.0
    max_players: int = 0
    ball_seen: bool = False
    frames: int = 0

    @property
    def score(self) -> float:
        """0..1 signal: field-player density plus a bump if the ball is seen."""
        density = min(1.0, self.avg_players / 6.0)
        return min(1.0, 0.8 * density + (0.2 if self.ball_seen else 0.0))


def available() -> bool:
    """True if the optional ultralytics dependency is importable."""
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def analyze_segment(
    video_path: str,
    start: float,
    end: float,
    model_path: str = "yolov8n.pt",
    sample_fps: float = 2.0,
    conf: float = 0.35,
    device: Optional[str] = None,
) -> PlayerStats:
    """Detect players/ball across sampled frames of [start, end].

    Returns empty :class:`PlayerStats` if OpenCV/Ultralytics are unavailable.
    """
    try:
        import cv2
        model = _load_model(model_path)
    except Exception:
        return PlayerStats()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return PlayerStats()

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / max(0.5, sample_fps))))
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames: List[np.ndarray] = []
        idx = start_frame
        while idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start_frame) % step == 0:
                frames.append(frame)
            idx += 1

        if not frames:
            return PlayerStats()

        results = model.predict(
            frames, conf=conf, verbose=False,
            device=device if device else None,
            classes=[_PERSON_CLASS, _BALL_CLASS],
        )

        counts: List[int] = []
        ball_seen = False
        for res in results:
            if res.boxes is None or res.boxes.cls is None:
                counts.append(0)
                continue
            cls = res.boxes.cls.cpu().numpy()
            counts.append(int(np.sum(cls == _PERSON_CLASS)))
            if np.any(cls == _BALL_CLASS):
                ball_seen = True

        if not counts:
            return PlayerStats()
        return PlayerStats(
            avg_players=float(np.mean(counts)),
            max_players=int(max(counts)),
            ball_seen=ball_seen,
            frames=len(counts),
        )
    finally:
        cap.release()
