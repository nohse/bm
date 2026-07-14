#!/usr/bin/env python
"""Aggregate the attention-WEIGHTED cond results and report best per (pair, t-range),
K-curve, hands-FP fix rate. Kept separate from the non-attention numbers."""
import os, json, glob
import numpy as np
import exp100_attn_config as C

HERE = os.path.dirname(os.path.abspath(__file__))
Z = {}; labels = None
for f in sorted(glob.glob(os.path.join(HERE, "exp100", "scores_attn", "scores_shard*.npz"))):
    d = np.load(f)
    for k in d.files:
        if k == "labels": labels = d[k]
        else: Z[k] = d[k]
NT = C.NT; seeds = C.SEEDS
HANDS = [51, 61, 62, 76, 83, 87, 97]


def kidx(K): return np.unique(np.linspace(0, NT - 1, K).round().astype(int))
def avg(key): return np.mean([Z[key.format(sd=sd)] for sd in seeds], axis=0)
def ncorr(p): return int((p == labels).sum())

rows = []
for fk in C.FACE_PROMPTS:
    for nk in C.NONFACE_PROMPTS:
        for tn in C.TRANGES:
            Ef = avg(f"{tn}|{{sd}}|{fk}|FACE")
            En = avg(f"{tn}|{{sd}}|{fk}|{nk}|NON")
            idx = kidx(15)
            pred = (Ef[:, idx].mean(1) < En[:, idx].mean(1)).astype(int)
            rows.append((ncorr(pred), tn, fk, nk, Ef, En))
rows.sort(key=lambda r: -r[0])

print("="*100)
print("ATTENTION-WEIGHTED cond | 100 imgs | best per condition at K=15")
print("="*100)
print(f"{'n_ok':>4}  {'t-range':<10}{'face':<9}{'nonface':<11}")
for nc, tn, fk, nk, Ef, En in rows[:12]:
    print(f"{nc:>4}  {tn:<10}{fk:<9}{nk:<11}  \"{C.FACE_PROMPTS[fk]}\" vs \"{C.NONFACE_PROMPTS[nk]}\"")

nc, tn, fk, nk, Ef, En = rows[0]
print(f"\nBEST (attention): {nc}/100  {tn} | \"{C.FACE_PROMPTS[fk]}\" vs \"{C.NONFACE_PROMPTS[nk]}\"")
print("  K:    " + " ".join(f"{K:>4}" for K in C.K_LIST))
kc = []
for K in C.K_LIST:
    idx = kidx(K)
    pred = (Ef[:, idx].mean(1) < En[:, idx].mean(1)).astype(int); kc.append(ncorr(pred))
print("  n_ok: " + " ".join(f"{v:>4}" for v in kc))
idx = kidx(15); pred = (Ef[:, idx].mean(1) < En[:, idx].mean(1)).astype(int)
print(f"  hands/body FPs fixed: {sum(1 for i in HANDS if pred[i]==0)}/{len(HANDS)}  "
      f"| FP={int(((pred==1)&(labels==0)).sum())} FN={int(((pred==0)&(labels==1)).sum())}")

json.dump({"best_attn": rows[0][0], "best_cond": f"{tn}|{fk}|{nk}",
           "top": [(int(r[0]), r[1], r[2], r[3]) for r in rows[:12]],
           "Kcurve_best": dict(zip(map(str, C.K_LIST), kc))},
          open(os.path.join(HERE, "exp100", "scores_attn", "results_attn.json"), "w"), indent=2)
print("\nsaved exp100/scores_attn/results_attn.json")
