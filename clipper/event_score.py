"""Visual motion / event scoring.

A lightweight stand-in for a trained event-detection model: it measures how
much visual motion happens inside a candidate clip (bat swing, fielders
sprinting, batsmen running between wickets). Higher motion around a loud crowd
moment is a strong signal that something exciting actually happened on screen,
which lets us rank clips beyond audio alone.

To swap in a real ML model, replace ``score_segment`` with a call to your
classifier (e.g. a CNN/transformer over sampled frames) returning a 0..1 score.
"""
from __future__ import annotations

import numpy as np


def score_segment(video_path: str, start: float, end: float, sample_fps: float = 4.0) -> float:
    """Return a 0..1 motion-intensity score for the segment [start, end].

    Uses frame differencing on sampled frames. Returns 0.0 if OpenCV cannot
    read the video so callers can treat it as "no boost".
    """
    try:
        import cv2
    except Exception:
        return 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / max(0.5, sample_fps))))
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        prev = None
        diffs = []
        frame_idx = start_frame
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start_frame) % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))
                if prev is not None:
                    diffs.append(float(np.mean(np.abs(gray.astype(np.int16) - prev))))
                prev = gray.astype(np.int16)
            frame_idx += 1

        if not diffs:
            return 0.0
        # Normalise: typical inter-frame mean abs diff saturates well below 60.
        return float(min(1.0, np.mean(diffs) / 30.0))
    finally:
        cap.release()
