"""Command-line interface for the cricket highlight clipper.

Example:
    python cli.py match.mp4 --sensitivity 1.4 --max-clips 10 -o clips/
"""
from __future__ import annotations

import argparse
import sys

from clipper.config import Settings
from clipper.pipeline import build_highlights, export_clips, Stats


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
        output_dir=args.output_dir,
    )

    def progress(frac: float, msg: str) -> None:
        print(f"[{int(frac * 100):3d}%] {msg}", file=sys.stderr)

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
