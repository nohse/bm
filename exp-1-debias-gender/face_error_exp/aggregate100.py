#!/usr/bin/env python
"""From exp100/scores.npz (per-timestep errors), for the cond scheme:
  (1) find ALL (t-range, prompt-pair) with >=95/100 correct at K=15;
  (2) for each such condition, report the accuracy as K (timestep count) is reduced.
Two accuracy views:
  n_correct_avg  = seed-averaged prediction (average the 15 per-t errors over 5 seeds,
                   then use K evenly-spaced of them)  -> stable / reproducible.
  n_correct_seed = per-seed single-pass mean +/- std (the honest 'K timesteps, one noise draw').
"""
import os, json, glob, argparse, importlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--cfg", default="exp100_config")
_ap.add_argument("--dir", default=os.path.join(HERE, "exp100"))
_ap.add_argument("--out", default=None)
_A = _ap.parse_args()
C = importlib.import_module(_A.cfg)
_OUT = _A.out or os.path.join(_A.dir, "results100.json")
# merge per-shard npz files
Z = {}
labels = None
for f in sorted(glob.glob(os.path.join(_A.dir, "scores_shard*.npz"))):
    d = np.load(f)
    for k in d.files:
        if k == "labels":
            labels = d[k]
        else:
            Z[k] = d[k]
N = len(labels)
FACE, NON = list(C.FACE_PROMPTS), list(C.NONFACE_PROMPTS)
seeds = C.SEEDS
NT = C.NT


def k_indices(K):
    return np.unique(np.linspace(0, NT - 1, K).round().astype(int))


# per-timestep arrays: arr[trange][seed][prompt] = [N,NT]
def get(tn, sd, pk):
    return Z[f"{tn}|{sd}|{pk}"]


def n_correct(Ef, En):
    return int(((En - Ef > 0).astype(int) == labels).sum())


# seed-averaged per-timestep errors per (trange, prompt)
avg_ts = {}
for tn in C.TRANGES:
    for pk in FACE + NON:
        avg_ts[(tn, pk)] = np.mean([get(tn, sd, pk) for sd in seeds], axis=0)  # [N,NT]

Kfull = k_indices(15)
results = {}          # (trange,f,n) -> {K: dict}
for tn in C.TRANGES:
    for fk in FACE:
        for nk in NON:
            row = {}
            for K in C.K_LIST:
                idx = k_indices(K)
                Ef = avg_ts[(tn, fk)][:, idx].mean(1)
                En = avg_ts[(tn, nk)][:, idx].mean(1)
                nc_avg = n_correct(Ef, En)
                # per-seed single-pass
                per = []
                for sd in seeds:
                    ef = get(tn, sd, fk)[:, idx].mean(1)
                    en = get(tn, sd, nk)[:, idx].mean(1)
                    per.append(n_correct(ef, en))
                row[K] = dict(avg=nc_avg, seed_mean=float(np.mean(per)),
                              seed_min=int(np.min(per)), seed_std=float(np.std(per)))
            results[(tn, fk, nk)] = row

# conditions with >=95 at K=15 (seed-averaged)
qualify = [(k, v) for k, v in results.items() if v[15]["avg"] >= 95]
qualify.sort(key=lambda kv: -kv[1][15]["avg"])

FP, NP = C.FACE_PROMPTS, C.NONFACE_PROMPTS
print(f"cond scheme | 100 images (50 face / 50 no-face) | {len(C.TRANGES)} t-ranges x "
      f"{len(FACE)}x{len(NON)} pairs\n")
print(f">=95/100 conditions at K=15 (seed-averaged): {len(qualify)} of {len(results)}\n")

print("="*104)
print("ALL >=95/100 CONDITIONS  and their accuracy as K (timestep count) shrinks  [seed-averaged n_correct]")
print("="*104)
hdr = "t-range      face        nonface     " + "".join(f"K{K:<4}" for K in C.K_LIST)
print(hdr)
for (tn, fk, nk), row in qualify:
    line = f"{tn:<13}{fk:<12}{nk:<12}"
    line += "".join(f"{row[K]['avg']:<5}" for K in C.K_LIST)
    print(line)

# which conditions HOLD >=95 down to small K
print("\n" + "="*104)
print("LOWEST K that still keeps >=95/100 (seed-averaged), for each qualifying condition")
print("="*104)
holds = []
for (tn, fk, nk), row in qualify:
    lowestK = min([K for K in C.K_LIST if row[K]["avg"] >= 95])
    holds.append((lowestK, tn, fk, nk, row))
holds.sort(key=lambda x: (x[0], -x[4][15]["avg"]))
print(f"{'minK>=95':<9}{'t-range':<13}{'face':<12}{'nonface':<12}{'K15':>5}{'K6':>5}{'K3':>5}{'K1':>5}")
for lowestK, tn, fk, nk, row in holds:
    print(f"{lowestK:<9}{tn:<13}{fk:<12}{nk:<12}"
          f"{row[15]['avg']:>5}{row[6]['avg']:>5}{row[3]['avg']:>5}{row[1]['avg']:>5}")

# detailed per-seed view for the top few
print("\n" + "="*104)
print("TOP 5 CONDITIONS -- per-seed single-pass n_correct (mean+/-std [min]) across K")
print("="*104)
for (tn, fk, nk), row in qualify[:5]:
    print(f'\n{tn} | "{FP[fk]}" vs "{NP[nk]}"')
    print("  " + "".join(f"K{K:<8}" for K in C.K_LIST))
    print("  " + "".join(f"{row[K]['seed_mean']:.1f}±{row[K]['seed_std']:.1f}[{row[K]['seed_min']}]".ljust(9)
                         for K in C.K_LIST))

# save
out = {f"{tn}|{fk}|{nk}": {str(K): row[K] for K in C.K_LIST}
       for (tn, fk, nk), row in results.items()}
json.dump({"qualify_ge95_at_K15": [f"{tn}|{fk}|{nk}" for (tn, fk, nk), _ in qualify],
           "prompts": {"face": FP, "nonface": NP}, "tranges": C.TRANGES,
           "K_list": C.K_LIST, "all": out},
          open(_OUT, "w"), indent=2)
print(f"\nsaved exp100/results100.json  ({len(results)} conditions x {len(C.K_LIST)} K values)")
