"""End-to-end highlight pipeline: signals -> candidate clips -> ranked clips."""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .audio_peaks import energy_envelope, extract_audio, find_excitement
from .config import Settings
from .event_score import score_segment
from .export import export_clip
from .ffmpeg_utils import ensure_ffmpeg, probe_duration, resolve_encoder
from .scene_detect import detect_scenes, snap_to_scenes

ProgressCb = Optional[Callable[[float, str], None]]


def _worker_count(settings: Settings) -> int:
    if settings.workers and settings.workers > 0:
        return settings.workers
    return max(1, os.cpu_count() or 1)



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


@dataclass
class Stats:
    """Timing and counters collected while processing a video."""
    video_path: str = ""
    video_seconds: float = 0.0
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    crowd_peaks: int = 0
    scene_cuts: int = 0
    candidate_clips: int = 0
    final_clips: int = 0
    clip_seconds_total: float = 0.0
    encoder_used: str = ""

    @property
    def analyse_seconds(self) -> float:
        keys = ("metadata", "audio_extract", "crowd_analysis", "scene_detect", "motion_score")
        return sum(self.stage_seconds.get(k, 0.0) for k in keys)

    @property
    def export_seconds(self) -> float:
        return self.stage_seconds.get("export", 0.0)

    @property
    def total_seconds(self) -> float:
        return sum(self.stage_seconds.values())

    @property
    def realtime_factor(self) -> float:
        """How many seconds of video processed per second of wall time."""
        if self.total_seconds <= 0:
            return 0.0
        return self.video_seconds / self.total_seconds

    def summary(self) -> str:
        lines = [
            f"Video duration   : {self.video_seconds:7.1f} s",
            f"Crowd peaks      : {self.crowd_peaks}",
            f"Scene cuts       : {self.scene_cuts}",
            f"Candidate clips  : {self.candidate_clips}",
            f"Final clips      : {self.final_clips} ({self.clip_seconds_total:.1f} s total)",
            f"Encoder used     : {self.encoder_used or 'n/a'}",
            "Stage timings:",
        ]
        for name, secs in self.stage_seconds.items():
            lines.append(f"  {name:<14}: {secs:7.2f} s")
        lines.append(f"Total processing : {self.total_seconds:7.2f} s "
                     f"({self.realtime_factor:.1f}x realtime)")
        return "\n".join(lines)


def _report(cb: ProgressCb, frac: float, msg: str) -> None:
    if cb:
        cb(max(0.0, min(1.0, frac)), msg)


@contextmanager
def _timed(stats: Stats, name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        stats.stage_seconds[name] = stats.stage_seconds.get(name, 0.0) + (
            time.perf_counter() - start
        )


def _score_motion_parallel(
    video_path: str, clips: List["Clip"], s: Settings, progress: ProgressCb
) -> None:
    """Score every clip's motion across multiple CPU cores."""
    workers = min(_worker_count(s), len(clips))
    if workers <= 1:
        for i, c in enumerate(clips):
            _report(progress, 0.55 + 0.35 * (i / max(1, len(clips))),
                    f"Scoring motion {i + 1}/{len(clips)}...")
            c.motion_score = score_segment(video_path, c.start, c.end, s.motion_sample_fps)
        return

    done = 0
    total = len(clips)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(score_segment, video_path, c.start, c.end, s.motion_sample_fps): c
                for c in clips
            }
            for fut in as_completed(futures):
                clip = futures[fut]
                try:
                    clip.motion_score = fut.result()
                except Exception:
                    clip.motion_score = 0.0
                done += 1
                _report(progress, 0.55 + 0.35 * (done / total),
                        f"Scoring motion {done}/{total} ({workers} workers)...")
    except Exception:
        # Pool unavailable (e.g. restricted env) -> fall back to sequential.
        for c in clips:
            c.motion_score = score_segment(video_path, c.start, c.end, s.motion_sample_fps)




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
    stats: Optional[Stats] = None,
) -> List[Clip]:
    """Analyse a video and return ranked highlight clips (not yet exported).

    If a ``Stats`` instance is provided it is populated with timings/counters.
    """
    s = settings or Settings()
    stats = stats if stats is not None else Stats()
    stats.video_path = video_path
    ensure_ffmpeg()

    _report(progress, 0.02, "Reading video metadata...")
    with _timed(stats, "metadata"):
        duration = probe_duration(video_path)
    stats.video_seconds = duration

    _report(progress, 0.08, "Extracting audio...")
    with _timed(stats, "audio_extract"):
        audio = extract_audio(video_path, s.audio_sr)

    _report(progress, 0.25, "Analysing crowd noise...")
    with _timed(stats, "crowd_analysis"):
        times, rms = energy_envelope(audio, s.audio_sr, s.win_seconds, s.hop_seconds)
        regions = find_excitement(times, rms, s.sensitivity, s.min_event_seconds)
    stats.crowd_peaks = len(regions)

    scenes: List[float] = []
    if s.use_scene_snap:
        _report(progress, 0.45, "Detecting scene cuts...")
        with _timed(stats, "scene_detect"):
            scenes = detect_scenes(
                video_path, s.scene_threshold, s.scene_frame_skip, s.scene_downscale
            )
    stats.scene_cuts = len(scenes)

    candidates: List[Clip] = []
    for t0, t1, score in regions:
        start = t0 - s.pre_seconds
        end = t1 + s.post_seconds
        if scenes:
            start = snap_to_scenes(start, scenes, s.snap_max_seconds)
        candidates.append(Clip(start=start, end=end, audio_score=score))

    clips = _merge_and_clamp(candidates, s, duration)
    stats.candidate_clips = len(clips)

    if s.use_event_score and clips:
        with _timed(stats, "motion_score"):
            _score_motion_parallel(video_path, clips, s, progress)

    clips.sort(key=lambda c: c.score, reverse=True)
    if s.max_clips and s.max_clips > 0:
        clips = clips[: s.max_clips]
    clips.sort(key=lambda c: c.start)

    stats.final_clips = len(clips)
    stats.clip_seconds_total = sum(c.duration for c in clips)

    _report(progress, 1.0, f"Found {len(clips)} highlight(s).")
    return clips


def export_clips(
    video_path: str,
    clips: List[Clip],
    settings: Optional[Settings] = None,
    progress: ProgressCb = None,
    stats: Optional[Stats] = None,
) -> List[Clip]:
    """Export each clip to disk in parallel, populating ``Clip.out_path``."""
    s = settings or Settings()
    os.makedirs(s.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    local_stats = stats if stats is not None else Stats()

    # CPU encodes are already multi-threaded, so don't oversubscribe; GPU/copy
    # exports are light, so concurrency there gives a real speedup.
    if s.accurate_cut and resolve_encoder(s.encoder) == "libx264":
        workers = min(len(clips), max(1, _worker_count(s) // 2)) or 1
    else:
        workers = min(len(clips), _worker_count(s)) or 1

    jobs = []
    for i, c in enumerate(clips, start=1):
        out_path = os.path.join(s.output_dir, f"{base}_clip{i:02d}.mp4")
        c.out_path = out_path
        jobs.append((c, out_path))

    done = 0
    total = len(jobs)

    def _run(job):
        clip, out_path = job
        return export_clip(video_path, clip.start, clip.end, out_path,
                           accurate=s.accurate_cut, encoder=s.encoder)

    with _timed(local_stats, "export"):
        if workers <= 1:
            for job in jobs:
                local_stats.encoder_used = _run(job)
                done += 1
                _report(progress, done / total, f"Exporting clip {done}/{total}...")
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run, job): job for job in jobs}
                for fut in as_completed(futures):
                    local_stats.encoder_used = fut.result()
                    done += 1
                    _report(progress, done / total,
                            f"Exporting clip {done}/{total} ({workers} workers)...")
    return clips
