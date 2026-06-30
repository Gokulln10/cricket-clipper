"""Bootstrap a labelling CSV from the clipper's own detections.

This speeds up building a training set: run the normal detection pipeline on a
match, export each candidate moment as a short clip, and write a CSV pre-filled
with the timestamps. You then just fill in the ``label`` column (e.g. four, six,
wicket, negative) and pass the CSV to ``python -m training.train``.

Example:
    python -m training.make_labels match.mp4 --out data/match1 --clip-seconds 3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

from clipper.config import Settings
from clipper.export import export_clip
from clipper.pipeline import build_highlights


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export candidate clips + a labelling CSV.")
    p.add_argument("video", help="Match video to mine for candidate events.")
    p.add_argument("--out", default="data/labelling", help="Output folder for clips + CSV.")
    p.add_argument("--sensitivity", type=float, default=1.0,
                   help="Low sensitivity = more candidates to label (default 1.0).")
    p.add_argument("--clip-seconds", type=float, default=3.0,
                   help="Length of each exported sample, centred on the moment.")
    p.add_argument("--no-export", action="store_true",
                   help="Only write the CSV, don't export sample clips.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    # Keep detection broad (no motion/player scoring needed) and high-recall.
    settings = Settings(
        sensitivity=args.sensitivity,
        use_event_score=False,
        use_player_detect=False,
        use_scene_snap=True,
    )

    def progress(frac, msg):
        print(f"[{int(frac * 100):3d}%] {msg}", file=sys.stderr)

    clips = build_highlights(args.video, settings, progress)
    if not clips:
        print("No candidates found; try a lower --sensitivity.", file=sys.stderr)
        return 1

    base = os.path.splitext(os.path.basename(args.video))[0]
    csv_path = os.path.join(args.out, f"{base}_labels.csv")
    half = args.clip_seconds / 2.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "start", "end", "label", "clip"])
        for i, c in enumerate(clips, 1):
            centre = (c.start + c.end) / 2.0
            start = max(0.0, centre - half)
            end = start + args.clip_seconds
            clip_path = ""
            if not args.no_export:
                clip_path = os.path.join(args.out, f"{base}_cand{i:03d}.mp4")
                try:
                    export_clip(args.video, start, end, clip_path, accurate=True)
                except Exception as exc:
                    print(f"  ! failed to export candidate {i}: {exc}", file=sys.stderr)
                    clip_path = ""
            writer.writerow([os.path.abspath(args.video), f"{start:.3f}", f"{end:.3f}", "", clip_path])

    print(f"\nWrote {len(clips)} candidates to {csv_path}")
    print("Fill in the 'label' column (four/six/wicket/negative/...), then run:")
    print(f"  python -m training.train --data {csv_path} --out models/event_clf.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
