"""Configuration for the cricket highlight clipper."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    # --- Audio crowd-noise detection ---
    audio_sr: int = 16000          # sample rate used for analysis (Hz)
    win_seconds: float = 0.5       # RMS analysis window length
    hop_seconds: float = 0.1       # step between analysis windows
    sensitivity: float = 1.6       # z-score threshold; lower = more clips
    min_event_seconds: float = 0.4 # ignore crowd spikes shorter than this

    # --- Clip framing ---
    pre_seconds: float = 6.0       # lead-in before the crowd reaction
    post_seconds: float = 4.0      # tail after the crowd reaction
    min_clip_seconds: float = 4.0  # enforce a sensible minimum length
    max_clip_seconds: float = 25.0 # cap a single clip's length
    merge_gap_seconds: float = 3.0 # merge clips closer than this together
    max_clips: int = 0             # 0 = keep all, otherwise keep top N by score

    # --- Scene-change snapping (PySceneDetect) ---
    use_scene_snap: bool = True
    scene_threshold: float = 27.0  # ContentDetector threshold
    snap_max_seconds: float = 2.5  # only snap a boundary within this distance
    scene_frame_skip: int = 0      # >0 skips frames during detection (faster, less precise)
    scene_downscale: int = 0       # 0 = auto downscale based on resolution

    # --- Motion / event scoring (visual excitement proxy) ---
    use_event_score: bool = True
    motion_sample_fps: float = 4.0 # frames/sec sampled when scoring motion

    # --- Parallelism ---
    workers: int = 0               # 0 = auto (os.cpu_count()); used for motion + export

    # --- Export ---
    accurate_cut: bool = True      # re-encode for frame-accurate cuts
    encoder: str = "auto"          # auto|cpu|h264_nvenc|h264_videotoolbox|h264_qsv|h264_amf
    output_dir: str = "clips"
