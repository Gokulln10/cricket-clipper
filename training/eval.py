"""Evaluate a trained audio-event classifier on a held-out labelled set.

Reports overall accuracy, a confusion matrix, and per-class precision / recall /
F1, plus a focused precision/recall for "positive" event classes (e.g. how
reliably boundaries are caught).

Example:
    python -m training.eval --model models/event_clf.pt --data data/val/
    python -m training.eval --model models/event_clf.pt --data data/val.csv
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an audio-event classifier.")
    p.add_argument("--model", required=True, help="Trained checkpoint (training.train output).")
    p.add_argument("--data", required=True, help="Held-out folder of class sub-dirs, or a CSV.")
    p.add_argument("--device", default="", help="''(auto)|cpu|cuda|mps")
    p.add_argument("--min-prob", type=float, default=0.0,
                   help="Predictions below this confidence are counted as 'unknown'.")
    return p.parse_args(argv)


def _confusion(y_true: List[int], y_pred: List[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if p >= 0:
            cm[t, p] += 1
    return cm


def _print_confusion(cm: np.ndarray, labels: List[str]) -> None:
    width = max(8, max(len(s) for s in labels) + 1)
    header = " " * width + "".join(f"{l[:width-1]:>{width}}" for l in labels)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(header)
    for i, row in enumerate(cm):
        line = f"{labels[i][:width-1]:>{width}}" + "".join(f"{v:>{width}}" for v in row)
        print(line)


def _print_per_class(cm: np.ndarray, labels: List[str]) -> Dict[str, float]:
    print("\nPer-class metrics:")
    print(f"{'label':>14}  {'precision':>9}  {'recall':>7}  {'f1':>6}  {'support':>7}")
    f1s = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        print(f"{label:>14}  {precision:>9.3f}  {recall:>7.3f}  {f1:>6.3f}  {support:>7d}")
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"{'macro avg':>14}  {'':>9}  {'':>7}  {macro_f1:>6.3f}")
    return {"macro_f1": macro_f1}


def main(argv=None) -> int:
    args = parse_args(argv)

    from .dataset import iter_samples
    from .infer import EventClassifier

    model = EventClassifier(args.model, args.device or None)
    labels = list(model.label_names)
    idx = {name: i for i, name in enumerate(labels)}

    samples = iter_samples(args.data)
    if not samples:
        print("No samples found.", file=sys.stderr)
        return 1

    y_true: List[int] = []
    y_pred: List[int] = []
    skipped_unknown_true = 0

    for k, (path, start, end, label) in enumerate(samples, 1):
        if label not in idx:
            # True label not in the model's vocabulary -> can't score it.
            skipped_unknown_true += 1
            continue
        try:
            pred = model.predict_segment(path, start, end if end is not None else start + model.duration)
        except Exception as exc:
            print(f"  ! skipping {path} ({exc})", file=sys.stderr)
            continue
        y_true.append(idx[label])
        y_pred.append(idx[pred.label] if pred.prob >= args.min_prob else -1)
        print(f"[{k}/{len(samples)}] true={label:<10} pred={pred.label:<10} p={pred.prob:.2f}",
              file=sys.stderr)

    if not y_true:
        print("No scorable samples (labels did not match the model's classes).", file=sys.stderr)
        return 1

    n = len(labels)
    cm = _confusion(y_true, y_pred, n)
    accuracy = float(np.trace(cm) / max(1, cm.sum()))

    _print_confusion(cm, labels)
    _print_per_class(cm, labels)
    print(f"\nOverall accuracy: {accuracy:.3f}  ({cm.sum()} samples)")

    # Focused view: how well do we catch *positive* events vs everything else.
    from clipper.classifier import is_positive
    pos = [i for i, l in enumerate(labels) if is_positive(l)]
    if pos:
        tp = sum(cm[i, j] for i in pos for j in pos)
        fn = sum(cm[i, j] for i in pos for j in range(n) if j not in pos)
        fp = sum(cm[i, j] for i in range(n) if i not in pos for j in pos)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        names = ", ".join(labels[i] for i in pos)
        print(f"\nPositive events ({names}):")
        print(f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
        print(f"  (recall = fraction of real events caught)")

    unscored = sum(1 for p in y_pred if p < 0)
    if unscored:
        print(f"\n{unscored} prediction(s) below --min-prob counted as misses.")
    if skipped_unknown_true:
        print(f"{skipped_unknown_true} sample(s) had labels not in the model and were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
