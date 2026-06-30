"""Command-line interface for the cricket highlight clipper.

Example:
    python cli.py match.mp4 --sensitivity 1.4 --max-clips 10 -o clips/
"""
from __future__ import annotations

import argparse
import sys

from clipper.config import Settings
from clipper.pipeline import build_highlights, export_clips, Stats, detect_cpu_count


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export highlight clips from a cricket video.")
    p.add_argument("video", help="Path to the input cricket video.")
    p.add_argument("-o", "--output-dir", default="clips", help="Folder for exported clips.")
    p.add_argument("--sensitivity", type=float, default=1.6,
                   help="Lower = more clips. Default 1.6.")
    p.add_argument("--pre", type=float, default=6.0, help="Lead-in seconds before reaction.")
    p.add_argument("--post", type=float, default=4.0, help="Tail seconds after reaction.")
    p.add_argument("--max-clips", type=int, default=0, help="Keep top N clips (0 = all).")
    p.add_argument("--no-scene-snap", action="store_true", help="Disable scene-cut snapping.")
    p.add_argument("--no-motion", action="store_true", help="Disable motion scoring.")
    p.add_argument("--fast", action="store_true", help="Stream-copy cuts (faster, keyframe-aligned).")
    p.add_argument("--encoder", default="auto",
                   help="Video encoder for re-encoded cuts: auto|cpu|h264_nvenc|"
                        "h264_videotoolbox|h264_qsv|h264_amf. 'auto' uses GPU if available.")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers for motion scoring & export (0 = all CPU cores).")
    p.add_argument("--scene-frame-skip", type=int, default=0,
                   help="Skip N frames between scene-detection samples (faster, less precise).")
    p.add_argument("--players", action="store_true",
                   help="Enable YOLO player/ball detection (needs `pip install -r requirements-ml.txt`).")
    p.add_argument("--player-model", default="yolov8n.pt",
                   help="YOLO weights to use (auto-downloaded on first run).")
    p.add_argument("--player-fps", type=float, default=2.0,
                   help="Frames/sec sampled for player detection.")
    p.add_argument("--device", default="",
                   help="Compute device for YOLO: ''(auto)|cpu|cuda|mps.")
    p.add_argument("--classifier", default="",
                   help="Path to a trained audio-event model (training/train.py output).")
    p.add_argument("--classifier-min-prob", type=float, default=0.5,
                   help="Minimum confidence for a positive event to boost a clip.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = Settings(
        sensitivity=args.sensitivity,
        pre_seconds=args.pre,
        post_seconds=args.post,
        max_clips=args.max_clips,
        use_scene_snap=not args.no_scene_snap,
        use_event_score=not args.no_motion,
        accurate_cut=not args.fast,
        encoder=args.encoder,
        workers=args.workers,
        scene_frame_skip=args.scene_frame_skip,
        use_player_detect=args.players,
        player_model=args.player_model,
        player_sample_fps=args.player_fps,
        device=args.device,
        classifier_model=args.classifier,
        classifier_min_prob=args.classifier_min_prob,
        output_dir=args.output_dir,
    )

    def progress(frac: float, msg: str) -> None:
        print(f"[{int(frac * 100):3d}%] {msg}", file=sys.stderr)

    cores = detect_cpu_count()
    chosen = args.workers if args.workers > 0 else cores
    print(f"Detected {cores} CPU core(s); using {chosen} worker(s).", file=sys.stderr)

    stats = Stats()
    clips = build_highlights(args.video, settings, progress, stats)
    if not clips:
        print("No highlights detected. Try a lower --sensitivity.", file=sys.stderr)
        return 1

    clips = export_clips(args.video, clips, settings, progress, stats)
    print(f"\nExported {len(clips)} clip(s) to '{settings.output_dir}':")
    for i, c in enumerate(clips, 1):
        print(f"  {i:2d}. {c.start:7.1f}s -> {c.end:7.1f}s "
              f"({c.duration:4.1f}s) score={c.score:.2f}  {c.out_path}")

    print("\n--- Stats ---")
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
