#!/usr/bin/env python
# coding=utf-8
"""
Simplest possible view of the experiment.

Ground truth = the face detector (get_face): each image is face or no-face.
Rule        = predict NO-FACE if realistic_face_loss > threshold, else FACE.
We raise the threshold in small even steps and record the accuracy at each.

Output: <out_root>/analysis/threshold_accuracy.json

Usage:
    python realistic_thr_acc.py <out_root> [step]
    python realistic_thr_acc.py            # auto-pick newest realistic_thr_experiment_*
"""
import os
import sys
import csv
import glob
import json


def find_out_root(argv):
    if len(argv) > 1 and os.path.isdir(argv[1]):
        return argv[1].rstrip("/")
    cands = [c for c in glob.glob(os.path.join("outputs", "**", "realistic_thr_experiment_*"), recursive=True)
             if os.path.isdir(os.path.join(c, "_shards"))]
    if not cands:
        raise SystemExit("give <out_root> (a realistic_thr_experiment_* dir)")
    cands.sort(key=os.path.getmtime)
    return cands[-1]


def load(out_root):
    """Return list of (is_face: bool, realistic_loss: float)."""
    rows = []
    csv_path = os.path.join(out_root, "analysis", "per_image.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append((int(r["face"]) == 1, float(r["realistic_mean"])))
        return rows
    for p in sorted(glob.glob(os.path.join(out_root, "_shards", "shard_rank*.json"))):
        for rec in json.load(open(p)).get("records", []):
            rows.append((bool(rec["face"]), float(rec["realistic_mean"])))
    return rows


def main():
    out_root = find_out_root(sys.argv)
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    data = load(out_root)
    if not data:
        raise SystemExit("no data found")

    n = len(data)
    n_face = sum(1 for f, _ in data if f)
    n_noface = n - n_face
    losses = [l for _, l in data]
    lo = (int(min(losses) / step)) * step
    hi = (int(max(losses) / step) + 1) * step

    sweep = []
    t = lo
    while t <= hi + 1e-9:
        # predict no-face if loss > t, else face
        correct = 0
        for is_face, loss in data:
            pred_noface = loss > t
            gt_noface = not is_face
            if pred_noface == gt_noface:
                correct += 1
        sweep.append({"threshold": round(t, 4),
                      "accuracy": round(correct / n, 4),
                      "n_correct": correct})
        t += step

    best = max(sweep, key=lambda r: r["accuracy"])
    out = {
        "ground_truth": "face detector get_face (face vs no-face)",
        "classification_rule": "predict NO-FACE if realistic_face_loss > threshold, else FACE",
        "n_total": n, "n_face": n_face, "n_noface": n_noface,
        "loss_min": round(min(losses), 4), "loss_max": round(max(losses), 4),
        "chance_accuracy": round(max(n_face, n_noface) / n, 4),
        "best_threshold": best["threshold"], "best_accuracy": best["accuracy"],
        "step": step,
        "sweep": sweep,
    }
    dst = os.path.join(out_root, "analysis", "threshold_accuracy.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)

    print(f"n={n} (face={n_face}, no-face={n_noface})  chance_acc={out['chance_accuracy']}")
    print(f"best: threshold={best['threshold']}  accuracy={best['accuracy']}")
    print("threshold  accuracy")
    for r in sweep:
        bar = "#" * int(round(r["accuracy"] * 40))
        print(f"  {r['threshold']:.3f}    {r['accuracy']:.3f}  {bar}")
    print(f"\nwrote: {dst}")


if __name__ == "__main__":
    main()
