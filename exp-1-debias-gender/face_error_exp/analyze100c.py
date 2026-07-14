#!/usr/bin/env python
"""Round-3 analysis: CONTRASTIVE matched pairs across many t-ranges. Reports
n_correct/100 per (pair, t-range), finds >=95, K-reduction for winners, and whether
the 7 hands/body false-positives from before get fixed."""
import os, json, glob
import numpy as np
import exp100c_config as C

HERE = os.path.dirname(os.path.abspath(__file__))
Z = {}; labels = None
for f in sorted(glob.glob(os.path.join(HERE, "exp100", "scores_c", "scores_shard*.npz"))):
    d = np.load(f)
    for k in d.files:
        if k == "labels": labels = d[k]
        else: Z[k] = d[k]
man = json.load(open(os.path.join(HERE, "exp100", "manifest_exp100.json")))
NT = C.NT; seeds = C.SEEDS
HANDS = [51, 61, 62, 76, 83, 87, 97]   # no-face-but-hands/body FPs; SmallFace=35


def kidx(K): return np.unique(np.linspace(0, NT - 1, K).round().astype(int))
def avg_ts(tn, pk): return np.mean([Z[f"{tn}|{sd}|{pk}"] for sd in seeds], axis=0)
def ncorr(Ef, En): return int(((En - Ef > 0).astype(int) == labels).sum())


tranges = list(C.TRANGES)
# matched-pair table at K=15
print("="*112)
print("CONTRASTIVE MATCHED PAIRS -- n_correct/100 at K=15 (seed-averaged) per t-range")
print("="*112)
idx15 = kidx(15)
hdr = f"{'pair':<40}" + "".join(f"{tn.replace('t',''):>9}" for tn in tranges)
print(hdr)
best = (-1, None, None)
allcells = []
for fk, nk in C.PAIRS:
    label = f'{C.FACE_PROMPTS[fk]} | {C.NONFACE_PROMPTS[nk]}'
    line = f"{label[:39]:<40}"
    for tn in tranges:
        Ef = avg_ts(tn, fk)[:, idx15].mean(1); En = avg_ts(tn, nk)[:, idx15].mean(1)
        nc = ncorr(Ef, En)
        allcells.append((nc, tn, fk, nk))
        line += f"{nc:>9}"
        if nc > best[0]:
            best = (nc, tn, (fk, nk))
    print(line)

# also all cross-combos (not just matched) to be safe
allcombos = []
for fk in C.FACE_PROMPTS:
    for nk in C.NONFACE_PROMPTS:
        for tn in tranges:
            Ef = avg_ts(tn, fk)[:, idx15].mean(1); En = avg_ts(tn, nk)[:, idx15].mean(1)
            allcombos.append((ncorr(Ef, En), tn, fk, nk))
allcombos.sort(key=lambda x: -x[0])
ge95 = [c for c in allcombos if c[0] >= 95]

print(f"\nbest MATCHED pair: {best[0]}/100  [{best[1]} | {best[2][0]} vs {best[2][1]}]")
print(f">=95/100 conditions (all combos): {len(ge95)}")
print("\nTOP 12 (all combos) at K=15:")
for nc, tn, fk, nk in allcombos[:12]:
    mark = " <-- >=95" if nc >= 95 else ""
    print(f"  {nc:>3}  {tn:<10} {C.FACE_PROMPTS[fk][:30]:<31} vs {C.NONFACE_PROMPTS[nk][:26]:<27}{mark}")

# K-reduction + hands check for the winner (best combo)
def report_cond(nc0, tn, fk, nk, tag):
    print("\n" + "="*112)
    print(f"{tag}: {tn} | \"{C.FACE_PROMPTS[fk]}\" vs \"{C.NONFACE_PROMPTS[nk]}\"  ({nc0}/100 @K15)")
    print("="*112)
    print("  K:      " + " ".join(f"{K:>4}" for K in C.K_LIST))
    row = []
    for K in C.K_LIST:
        idx = kidx(K)
        Ef = avg_ts(tn, fk)[:, idx].mean(1); En = avg_ts(tn, nk)[:, idx].mean(1)
        row.append(ncorr(Ef, En))
    print("  n_ok:   " + " ".join(f"{v:>4}" for v in row))
    # hands / small-face check at K=15
    idx = idx15
    Ef = avg_ts(tn, fk)[:, idx].mean(1); En = avg_ts(tn, nk)[:, idx].mean(1)
    pred = (En - Ef > 0).astype(int)
    fixed = sum(1 for i in HANDS if pred[i] == 0)
    print(f"  hands/body FPs fixed (now correctly no-face): {fixed}/{len(HANDS)}")
    print(f"  small distant face (idx35) correct: {'yes' if pred[35]==1 else 'no'}")

report_cond(*allcombos[0][:1], allcombos[0][1], allcombos[0][2], allcombos[0][3], tag="BEST OVERALL")
bm = best
report_cond(bm[0], bm[1], bm[2][0], bm[2][1], tag="BEST MATCHED PAIR")

json.dump({"best_matched": best[0], "best_overall": allcombos[0][0],
           "ge95_count": len(ge95),
           "ge95": [f"{tn}|{fk}|{nk}(={nc})" for nc, tn, fk, nk in ge95],
           "top12": [f"{tn}|{fk}|{nk}(={nc})" for nc, tn, fk, nk in allcombos[:12]]},
          open(os.path.join(HERE, "exp100", "scores_c", "results100c.json"), "w"), indent=2)
print("\nsaved exp100/scores_c/results100c.json")
