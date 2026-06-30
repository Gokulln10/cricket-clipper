"""Scene-change detection used to snap clip boundaries to natural cuts.

Cricket broadcasts cut between the live bowler-cam, replays and scoreboards.
Snapping the start of a clip to the nearest scene cut makes clips begin on a
clean shot instead of mid-frame.
"""
from __future__ import annotations

import bisect
from typing import List


def detect_scenes(video_path: str, threshold: float = 27.0) -> List[float]:
    """Return a sorted list of scene-cut timestamps (seconds).

    Falls back to an empty list if PySceneDetect is unavailable so the rest of
    the pipeline keeps working.
    """
    try:
        from scenedetect import detect, ContentDetector
    except Exception:
        return []

    try:
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
    except Exception:
        return []

    cuts: List[float] = []
    for start, _end in scene_list:
        cuts.append(start.get_seconds())
    return sorted(set(cuts))


def snap_to_scenes(t: float, scenes: List[float], max_distance: float) -> float:
    """Snap time ``t`` to the nearest scene cut within ``max_distance`` seconds."""
    if not scenes:
        return t
    idx = bisect.bisect_left(scenes, t)
    candidates = []
    if idx < len(scenes):
        candidates.append(scenes[idx])
    if idx > 0:
        candidates.append(scenes[idx - 1])
    best = min(candidates, key=lambda s: abs(s - t))
    return best if abs(best - t) <= max_distance else t
