"""Tuning / diagnostics tool for cricket-clipper.

Run the audio analysis on a real match and see *exactly* what each detector
fires on, without exporting any clips. This makes it fast to dial in the
right `--sensitivity`, `--onset-sensitivity` and `--swell-sensitivity` before
committing to a full run.

Examples
--------
Dump every detection to the console + a CSV you can scrub through:

    python tune.py match.mp4 --csv detections.csv

Sweep sensitivity to see how the candidate count changes:

    python tune.py match.mp4 --sweep 1.0 1.2 1.5 1.8 2.2

Score against known event timestamps (one "MM:SS" or seconds per line) to get
recall / false-positive numbers while you tune:

    python tune.py match.mp4 --truth events.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Tuple

import numpy as np

from clipper.audio_peaks import (
    extract_audio,
    energy_envelope,
    find_excitement,
    find_excitement_labeled,
)
from clipper.config import Settings


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _parse_truth(path: str) -> List[float]:
    """Read ground-truth event times: one per line, "MM:SS", "H:MM:SS" or seconds."""
    times: List[float] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split(",")[0].split()[0]
            if ":" in line:
                parts = [float(p) for p in line.split(":")]
                secs = 0.0
                for p in parts:
                    secs = secs * 60 + p
                times.append(secs)
            else:
                times.append(float(line))
    return sorted(times)


def _build_envelope(video: str, st: Settings):
    audio = extract_audio(video, sr=st.audio_sr)
    times, rms = energy_envelope(audio, st.audio_sr, st.win_seconds, st.hop_seconds)
    if times.size == 0:
        sys.exit("No audio detected (silent or unreadable track).")
    return times, rms


def _detect(times, rms, st: Settings, sensitivity: float):
    return find_excitement(
        times,
        rms,
        sensitivity=sensitivity,
        min_event_seconds=st.min_event_seconds,
        hop_seconds=st.hop_seconds,
        local_baseline_seconds=st.local_baseline_seconds,
        use_onset=st.use_onset,
        onset_sensitivity=st.onset_sensitivity,
        use_swell=st.use_swell,
        swell_window_seconds=st.swell_window_seconds,
        swell_sensitivity=st.swell_sensitivity,
    )


def _detect_labeled(times, rms, st: Settings, sensitivity: float):
    return find_excitement_labeled(
        times,
        rms,
        sensitivity=sensitivity,
        min_event_seconds=st.min_event_seconds,
        hop_seconds=st.hop_seconds,
        local_baseline_seconds=st.local_baseline_seconds,
        use_onset=st.use_onset,
        onset_sensitivity=st.onset_sensitivity,
        use_swell=st.use_swell,
        swell_window_seconds=st.swell_window_seconds,
        swell_sensitivity=st.swell_sensitivity,
    )


def _score_against_truth(
    regions: List[Tuple[float, float, float]],
    truth: List[float],
    tol: float,
) -> Tuple[int, int, int, List[float]]:
    """Match detections to truth times. A truth time is "hit" if it falls within
    a detection window (expanded by ``tol`` seconds on each side)."""
    hit = [False] * len(truth)
    matched_regions = [False] * len(regions)
    for ti, tt in enumerate(truth):
        for ri, (a, b, _s) in enumerate(regions):
            if (a - tol) <= tt <= (b + tol):
                hit[ti] = True
                matched_regions[ri] = True
                break
    hits = sum(hit)
    misses = len(truth) - hits
    false_pos = sum(1 for m in matched_regions if not m)
    missed_times = [truth[i] for i, h in enumerate(hit) if not h]
    return hits, misses, false_pos, missed_times


def cmd_dump(video: str, st: Settings, sensitivity: float, csv_path: str | None, truth_path: str | None, tol: float):
    times, rms = _build_envelope(video, st)
    duration = float(times[-1])
    labeled = _detect_labeled(times, rms, st, sensitivity)
    merged = _detect(times, rms, st, sensitivity)

    # Per-detector tallies from the labeled (pre-merge) view.
    by_det = {"level": 0, "onset": 0, "swell": 0}
    for _a, _b, _s, lbl in labeled:
        by_det[lbl] = by_det.get(lbl, 0) + 1

    print(f"Video length      : {_fmt(duration)} ({duration:.0f}s)")
    print(f"Sensitivity       : {sensitivity:.2f}")
    print(f"Raw detector hits : level={by_det['level']}  onset={by_det['onset']}  swell={by_det['swell']}")
    print(f"Merged candidates : {len(merged)}")
    print("-" * 60)
    for a, b, s in merged:
        # Which detectors overlapped this merged window?
        dets = sorted({lbl for la, lb, _ls, lbl in labeled if not (lb < a or la > b)})
        tag = "+".join(d[0] for d in dets) or "?"
        print(f"  {_fmt(a):>8}  ->  {_fmt(b):<8}  z={s:5.2f}  [{tag}]")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["start_s", "end_s", "start", "end", "zscore", "detectors"])
            for a, b, s in merged:
                dets = sorted({lbl for la, lb, _ls, lbl in labeled if not (lb < a or la > b)})
                w.writerow([f"{a:.2f}", f"{b:.2f}", _fmt(a), _fmt(b), f"{s:.3f}", "+".join(dets)])
        print("-" * 60)
        print(f"Wrote {len(merged)} rows to {csv_path}")

    if truth_path:
        truth = _parse_truth(truth_path)
        hits, misses, fp, missed = _score_against_truth(merged, truth, tol)
        recall = hits / len(truth) if truth else 0.0
        precision = hits / len(merged) if merged else 0.0
        print("-" * 60)
        print(f"Ground truth      : {len(truth)} events (±{tol:.0f}s tolerance)")
        print(f"Recall            : {recall:6.1%}  ({hits}/{len(truth)} caught)")
        print(f"Precision (approx): {precision:6.1%}  ({fp} candidates matched no event)")
        if missed:
            print("Missed events     : " + ", ".join(_fmt(t) for t in missed))


def cmd_sweep(video: str, st: Settings, values: List[float], truth_path: str | None, tol: float):
    times, rms = _build_envelope(video, st)
    truth = _parse_truth(truth_path) if truth_path else None
    header = f"{'sensitivity':>12} {'candidates':>12}"
    if truth:
        header += f" {'recall':>8} {'precision':>10} {'missed':>7}"
    print(header)
    print("-" * len(header))
    for v in values:
        merged = _detect(times, rms, st, v)
        line = f"{v:12.2f} {len(merged):12d}"
        if truth:
            hits, _miss, fp, _mt = _score_against_truth(merged, truth, tol)
            recall = hits / len(truth) if truth else 0.0
            precision = hits / len(merged) if merged else 0.0
            line += f" {recall:8.1%} {precision:10.1%} {len(truth) - hits:7d}"
        print(line)


def build_settings(args) -> Settings:
    st = Settings()
    st.onset_sensitivity = args.onset_sensitivity
    st.swell_sensitivity = args.swell_sensitivity
    st.local_baseline_seconds = args.local_baseline
    st.min_event_seconds = args.min_event
    if args.no_onset:
        st.use_onset = False
    if args.no_swell:
        st.use_swell = False
    return st


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tune cricket-clipper audio detection on a real match.")
    p.add_argument("video", help="Path to the match video.")
    p.add_argument("--sensitivity", type=float, default=1.3, help="Level z-score threshold (default 1.3).")
    p.add_argument("--onset-sensitivity", type=float, default=2.5, help="Onset z-score threshold.")
    p.add_argument("--swell-sensitivity", type=float, default=1.0, help="Swell z-score threshold.")
    p.add_argument("--local-baseline", type=float, default=45.0, help="Rolling baseline window (s).")
    p.add_argument("--min-event", type=float, default=0.3, help="Minimum event length (s).")
    p.add_argument("--no-onset", action="store_true", help="Disable the onset detector.")
    p.add_argument("--no-swell", action="store_true", help="Disable the swell detector.")
    p.add_argument("--csv", metavar="PATH", help="Write merged detections to a CSV.")
    p.add_argument("--truth", metavar="PATH", help="Ground-truth event times file (one per line).")
    p.add_argument("--tol", type=float, default=8.0, help="Match tolerance vs ground truth (s).")
    p.add_argument(
        "--sweep",
        nargs="+",
        type=float,
        metavar="S",
        help="Sweep these sensitivity values and report candidate counts (and recall if --truth given).",
    )
    args = p.parse_args(argv)

    st = build_settings(args)
    if args.sweep:
        cmd_sweep(args.video, st, args.sweep, args.truth, args.tol)
    else:
        cmd_dump(args.video, st, args.sensitivity, args.csv, args.truth, args.tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
