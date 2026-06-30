"""Train the cricket audio-event classifier.

Example:
    python -m training.train --data data/ --epochs 40 --out models/event_clf.pt

The checkpoint stores the model weights, the label names and the feature
geometry, so inference is fully self-describing.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an audio-event classifier.")
    p.add_argument("--data", required=True, help="Folder of class sub-dirs, or a CSV file.")
    p.add_argument("--out", default="models/event_clf.pt", help="Output checkpoint path.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--device", default="", help="''(auto)|cpu|cuda|mps")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _pick_device(name: str):
    import torch
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .dataset import load_dataset
    from .model import AudioCNN

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset from {args.data} ...")
    X, y, label_names = load_dataset(args.data, args.sr, args.duration, args.n_mels)
    print(f"  {len(y)} samples, {len(label_names)} classes: {label_names}")

    # Normalise features (store stats for inference).
    mean = float(X.mean())
    std = float(X.std() + 1e-6)
    X = (X - mean) / std

    # Stratified-ish split.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    n_val = max(1, int(len(y) * args.val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    device = _pick_device(args.device)
    print(f"  device: {device}")

    def loader(indices, shuffle):
        ds = TensorDataset(torch.from_numpy(X[indices]), torch.from_numpy(y[indices]))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_dl = loader(train_idx, True)
    val_dl = loader(val_idx, False)

    # Class weights to counter imbalance (events are rarer than negatives).
    counts = np.bincount(y[train_idx], minlength=len(label_names)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32, device=device)

    model = AudioCNN(len(label_names), n_mels=args.n_mels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_idx)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
        val_acc = correct / len(val_idx)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "label_names": label_names,
                "sr": args.sr,
                "duration": args.duration,
                "n_mels": args.n_mels,
                "feat_mean": mean,
                "feat_std": std,
            }, args.out)

    print(f"\nBest val accuracy: {best_acc:.3f}")
    print(f"Saved checkpoint -> {args.out}")
    # Also write a sidecar with the label list for convenience.
    with open(os.path.splitext(args.out)[0] + "_labels.json", "w") as f:
        json.dump(label_names, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
