#!/usr/bin/env python
"""Merge scoring shards for the TRUE-label set and report:
  (1) PRIMARY: error-classifier acc/AUC on 20 genuine-face vs 20 genuine-noface,
      swept over prompt pairs / schemes / timestep grids.
  (2) DIAGNOSTIC: on insightface-MISSED faces, does the best error condition call
      them 'face' (recover) -- i.e. does the error-method beat insightface there?
"""
import os, json, glob
import numpy as np
import error_classify as E

HERE = E.HERE
manifest = json.load(open(os.path.join(HERE, "manifest_true.json")))
Ntot = len(manifest)
group = np.array([m["group"] for m in manifest])
true_lab = np.array([1 if m["true_label"] == "face" else 0 for m in manifest])
insight_lab = np.array([1 if m["insight_label"] == "face" else 0 for m in manifest])

shards = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "results_true", "shard_*.json")))]
grids = list(E.GRIDS.keys())
schemes = ["cond", "eps_diff"] + [f"cfg{g}" for g in E.CFG_SCALES]
prompts = list(E.FACE_PROMPTS) + list(E.NONFACE_PROMPTS)

full = {g: {s: {p: np.full(Ntot, np.nan) for p in prompts} for s in schemes} for g in grids}
for sh in shards:
    for g in grids:
        for s in schemes:
            for p in prompts:
                for j, gi in enumerate(sh["idxs"]):
                    full[g][s][p][gi] = sh["grids"][g][s][p][j]

is_true = group == "true"
is_missed = group == "missed"
yt = true_lab[is_true]

# ---------- PRIMARY sweep on group=true ----------
allrows = []
per = {g: {} for g in grids}
for g in grids:
    for s in schemes:
        rows = []
        for fk in E.FACE_PROMPTS:
            for nk in E.NONFACE_PROMPTS:
                r = E.eval_pair(full[g][s][fk][is_true], full[g][s][nk][is_true], yt)
                rows.append(dict(face=fk, nonface=nk, **r))
                allrows.append((r["acc"], r["auc"], g, s, fk, nk))
        rows.sort(key=lambda r: (-r["acc"], -r["auc"]))
        per[g][s] = rows
allrows.sort(key=lambda x: (-x[0], -x[1]))

FP, NP = E.FACE_PROMPTS, E.NONFACE_PROMPTS
print(f"TRUE SETTING: {int(yt.sum())} genuine-face vs {int((1-yt).sum())} genuine-noface "
      f"(person context, 3-detector consensus)\n")

print("="*82)
print('USER LITERAL PAIR  "a photo of a face" vs "a photo of a non face"  [cond]')
print("="*82)
print(f"{'grid':<16}{'acc':>7}{'auc':>8}")
for g in grids:
    row = next(r for r in per[g]["cond"] if r["face"]=="F_face" and r["nonface"]=="N_nonface")
    print(f"{g:<16}{row['acc']:>7.3f}{row['auc']:>8.3f}")

print("\n" + "="*82)
print("TOP 25 CONDITIONS (grid, scheme, pair) ranked by acc, then auc")
print("="*82)
print(f"{'acc':>6}{'auc':>7}  {'grid':<15}{'scheme':<10}{'face':<13}{'nonface'}")
for acc, auc, g, s, fk, nk in allrows[:25]:
    print(f"{acc:>6.3f}{auc:>7.3f}  {g:<15}{s:<10}{fk:<13}{nk}")

print("\n" + "="*82)
print("BEST SCHEME per family (broad grid, best pair)")
print("="*82)
for s in schemes:
    b = per["broad_50_950"][s][0]
    print(f"  {s:<10} acc={b['acc']:.3f} auc={b['auc']:.3f}  ({b['face']} vs {b['nonface']})")

# best by AUC (threshold-free, robust on 40 imgs)
allrows_auc = sorted(allrows, key=lambda x: (-x[1], -x[0]))
bestA = allrows_auc[0]
bestAcc = allrows[0]
print("\n" + "="*82)
print(f"BEST BY AUC : auc={bestA[1]:.3f} acc={bestA[0]:.3f} | {bestA[2]} | {bestA[3]} | "
      f'"{FP.get(bestA[4],bestA[4])}" vs "{NP.get(bestA[5],bestA[5])}"')
print(f"BEST BY ACC : acc={bestAcc[0]:.3f} auc={bestAcc[1]:.3f} | {bestAcc[2]} | {bestAcc[3]} | "
      f'"{FP.get(bestAcc[4],bestAcc[4])}" vs "{NP.get(bestAcc[5],bestAcc[5])}"')
print("="*82)

# ---------- DIAGNOSTIC: insightface-missed faces ----------
if is_missed.sum() > 0:
    g, s, fk, nk = bestA[2], bestA[3], bestA[4], bestA[5]
    Ef = full[g][s][fk][is_missed]; En = full[g][s][nk][is_missed]
    pred_face = (En - Ef) > 0     # error-method predicts face
    print("\n" + "="*82)
    print(f"DIAGNOSTIC on {int(is_missed.sum())} insightface-MISSED faces "
          f"(truth=face, insightface said no-face)")
    print(f"  using best condition [{g} | {s} | {fk} vs {nk}]")
    print("="*82)
    print(f"  error-method labels them FACE (recovers real face): "
          f"{int(pred_face.sum())}/{len(pred_face)} = {pred_face.mean()*100:.0f}%")
    print(f"  insightface labels them FACE:                        0/{len(pred_face)} = 0%")
    print("  -> higher recovery = error-method tracks TRUE face presence better than insightface")

# ---------- insightface vs truth on full set (context) ----------
print("\n" + "="*82)
acc_ins = (insight_lab == true_lab).mean()
print(f"insightface accuracy vs TRUTH over all {Ntot} scored images: {acc_ins*100:.1f}%")
print("="*82)

json.dump({"primary_top": allrows[:25], "best_auc": bestA, "best_acc": bestAcc},
          open(os.path.join(HERE, "results_true", "results_true.json"), "w"), indent=2, default=float)
