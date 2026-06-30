"""End-to-end highlight pipeline: signals -> candidate clips -> ranked clips."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .audio_peaks import energy_envelope, extract_audio, find_excitement
from .config import Settings
from .event_score import score_segment
from .export import export_clip
from .ffmpeg_utils import ensure_ffmpeg, probe_duration
from .scene_detect import detect_scenes, snap_to_scenes

ProgressCb = Optional[Callable[[float, str], None]]


@dataclass
class Clip:
    start: float
    end: float
    audio_score: float
    motion_score: float = 0.0
    out_path: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def score(self) -> float:
        # Blend crowd reaction with on-screen motion.
        return self.audio_score * (1.0 + 0.5 * self.motion_score)

    @property
    def reasons(self) -> List[str]:
        tags = ["crowd-peak"]
        if self.motion_score >= 0.4:
            tags.append("high-motion")
        return tags


def _report(cb: ProgressCb, frac: float, msg: str) -> None:
    if cb:
        cb(max(0.0, min(1.0, frac)), msg)


def _merge_and_clamp(clips: List[Clip], s: Settings, duration: float) -> List[Clip]:
    clips.sort(key=lambda c: c.start)
    merged: List[Clip] = []
    for c in clips:
        if merged and c.start - merged[-1].end <= s.merge_gap_seconds:
            last = merged[-1]
            last.end = max(last.end, c.end)
            last.audio_score = max(last.audio_score, c.audio_score)
        else:
            merged.append(c)

    out: List[Clip] = []
    for c in merged:
        c.start = max(0.0, c.start)
        c.end = min(duration, c.end) if duration > 0 else c.end
        # Enforce minimum length by padding around the centre.
        if c.duration < s.min_clip_seconds:
            centre = (c.start + c.end) / 2.0
            half = s.min_clip_seconds / 2.0
            c.start = max(0.0, centre - half)
            c.end = c.start + s.min_clip_seconds
            if duration > 0:
                c.end = min(duration, c.end)
        # Enforce maximum length.
        if c.duration > s.max_clip_seconds:
            c.end = c.start + s.max_clip_seconds
        out.append(c)
    return out


def build_highlights(
    video_path: str,
    settings: Optional[Settings] = None,
    progress: ProgressCb = None,
) -> List[Clip]:
    """Analyse a video and return ranked highlight clips (not yet exported)."""
    s = settings or Settings()
    ensure_ffmpeg()

    _report(progress, 0.02, "Reading video metadata...")
    duration = probe_duration(video_path)

    _report(progress, 0.08, "Extracting audio...")
    audio = extract_audio(video_path, s.audio_sr)

    _report(progress, 0.25, "Analysing crowd noise...")
    times, rms = energy_envelope(audio, s.audio_sr, s.win_seconds, s.hop_seconds)
    regions = find_excitement(times, rms, s.sensitivity, s.min_event_seconds)

    scenes: List[float] = []
    if s.use_scene_snap:
        _report(progress, 0.45, "Detecting scene cuts...")
        scenes = detect_scenes(video_path, s.scene_threshold)

    candidates: List[Clip] = []
    for t0, t1, score in regions:
        start = t0 - s.pre_seconds
        end = t1 + s.post_seconds
        if scenes:
            start = snap_to_scenes(start, scenes, s.snap_max_seconds)
        candidates.append(Clip(start=start, end=end, audio_score=score))

    clips = _merge_and_clamp(candidates, s, duration)

    if s.use_event_score and clips:
        for i, c in enumerate(clips):
            _report(
                progress,
                0.55 + 0.35 * (i / max(1, len(clips))),
                f"Scoring motion {i + 1}/{len(clips)}...",
            )
            c.motion_score = score_segment(video_path, c.start, c.end, s.motion_sample_fps)

    clips.sort(key=lambda c: c.score, reverse=True)
    if s.max_clips and s.max_clips > 0:
        clips = clips[: s.max_clips]
    clips.sort(key=lambda c: c.start)

    _report(progress, 1.0, f"Found {len(clips)} highlight(s).")
    return clips


def export_clips(
    video_path: str,
    clips: List[Clip],
    settings: Optional[Settings] = None,
    progress: ProgressCb = None,
) -> List[Clip]:
    """Export each clip to disk, populating ``Clip.out_path``."""
    s = settings or Settings()
    os.makedirs(s.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    for i, c in enumerate(clips, start=1):
        out_path = os.path.join(s.output_dir, f"{base}_clip{i:02d}.mp4")
        _report(progress, i / max(1, len(clips)), f"Exporting clip {i}/{len(clips)}...")
        export_clip(video_path, c.start, c.end, out_path, accurate=s.accurate_cut)
        c.out_path = out_path
    return clips
