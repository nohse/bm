#!/usr/bin/env python
"""Find cond-only (t-range, prompt-pair) configs that classify all 40 TRUE images
correctly. Rank by ROBUSTNESS: prefer configs that are 40/40 under the seed-averaged
prediction AND at every individual seed (not multiple-comparison luck)."""
import os, json, glob
import numpy as np
import error_classify as E
import search_config as SC

HERE = E.HERE
manifest = json.load(open(os.path.join(HERE, "manifest_true.json")))
true = [m for m in manifest if m["group"] == "true"]
y = np.array([1 if m["true_label"] == "face" else 0 for m in true])
N = len(y)

shards = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "results_search", "shard_*.json")))]
K = shards[0]["K"]
# merge: data[trange][seed][prompt] -> np[N]
data = {}
for sh in shards:
    for tn, seeds in sh["tranges"].items():
        data[tn] = {sd: {p: np.array(v) for p, v in tbl.items()} for sd, tbl in seeds.items()}

seeds = [str(s) for s in SC.SEEDS]
FACE, NON = list(SC.FACE_PROMPTS), list(SC.NONFACE_PROMPTS)


def acc_of(Ef, En):
    return float(((En - Ef > 0).astype(int) == y).mean())


rows = []
for tn in data:
    # seed-averaged E
    avgE = {p: np.mean([data[tn][sd][p] for sd in seeds], axis=0)
            for p in FACE + NON}
    for fk in FACE:
        for nk in NON:
            acc_avg = acc_of(avgE[fk], avgE[nk])
            per_seed = [acc_of(data[tn][sd][fk], data[tn][sd][nk]) for sd in seeds]
            rows.append(dict(trange=tn, face=fk, nonface=nk, acc_avg=acc_avg,
                             acc_seed_min=min(per_seed), acc_seed_mean=float(np.mean(per_seed)),
                             per_seed=per_seed))

# rank: 40/40 averaged first, then by worst-seed accuracy, then mean
rows.sort(key=lambda r: (-r["acc_avg"], -r["acc_seed_min"], -r["acc_seed_mean"]))
perfect_avg = [r for r in rows if r["acc_avg"] == 1.0]
perfect_allseed = [r for r in rows if r["acc_seed_min"] == 1.0]

FP, NP = SC.FACE_PROMPTS, SC.NONFACE_PROMPTS
print(f"cond-only search on 40 TRUE images | K={K} | {len(SC.TRANGES)} t-ranges x "
      f"{len(FACE)}x{len(NON)} pairs = {len(rows)} configs\n")
print(f"configs with seed-AVERAGED acc = 40/40 : {len(perfect_avg)}")
print(f"configs with EVERY-seed acc   = 40/40 : {len(perfect_allseed)}  (truly robust)\n")

print("="*94)
print("MOST ROBUST 40/40 CONFIGS (40/40 at every seed) -- top 20")
print("="*94)
print(f"{'trange':<11}{'face':<11}{'nonface':<11}{'avg':>5}{'seedmin':>8}{'seedmean':>9}  per-seed")
for r in perfect_allseed[:20]:
    print(f"{r['trange']:<11}{r['face']:<11}{r['nonface']:<11}"
          f"{r['acc_avg']:>5.2f}{r['acc_seed_min']:>8.2f}{r['acc_seed_mean']:>9.3f}  "
          f"{['%.2f'%a for a in r['per_seed']]}")

if not perfect_allseed:
    print("(none 40/40 at every seed; showing best seed-averaged 40/40)")
    print("="*94)
    for r in perfect_avg[:20]:
        print(f"{r['trange']:<11}{r['face']:<11}{r['nonface']:<11}"
              f"{r['acc_avg']:>5.2f}{r['acc_seed_min']:>8.2f}{r['acc_seed_mean']:>9.3f}  "
              f"{['%.2f'%a for a in r['per_seed']]}")

# show the readable prompts of the single best config
best = (perfect_allseed or perfect_avg or rows)[0]
print("\n" + "="*94)
print("BEST CONFIG:")
print(f"  t-range   = {best['trange']}  {SC.TRANGES[best['trange']]}  (K={K})")
print(f'  face      = "{FP[best["face"]]}"')
print(f'  non-face  = "{NP[best["nonface"]]}"')
print(f"  acc: avg={best['acc_avg']:.3f}  seed-min={best['acc_seed_min']:.3f}  "
      f"seed-mean={best['acc_seed_mean']:.3f}  per-seed={['%.2f'%a for a in best['per_seed']]}")
print("="*94)

json.dump(dict(K=K, best=best, perfect_allseed=perfect_allseed[:50],
               perfect_avg=perfect_avg[:50]),
          open(os.path.join(HERE, "results_search", "search_summary.json"), "w"),
          indent=2, default=float)
