#!/usr/bin/env python
"""Merge the 8 scoring shards into full 40-image error tables, evaluate every
prompt-pair / scheme / grid, and print the ranked report for the HARD setting."""
import os, json, glob
import numpy as np
import error_classify as E

HERE = E.HERE
manifest = json.load(open(os.path.join(HERE, "manifest_occ.json")))
N = len(manifest)
labels = np.array([1 if m["label"] == "face" else 0 for m in manifest])

# merge shards
shards = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "results_occ", "shard_*.json")))]
grids = list(E.GRIDS.keys())
schemes = ["cond", "eps_diff"] + [f"cfg{g}" for g in E.CFG_SCALES]
prompts = list(E.FACE_PROMPTS) + list(E.NONFACE_PROMPTS)

# full[grid][scheme][prompt] -> np[N]
full = {g: {s: {p: np.full(N, np.nan) for p in prompts} for s in schemes} for g in grids}
neg_full = {}
for sh in shards:
    idxs = sh["idxs"]
    for g in grids:
        for s in schemes:
            for p in prompts:
                for j, gi in enumerate(idxs):
                    full[g][s][p][gi] = sh["grids"][g][s][p][j]
    for key, d in sh["negcfg"].items():
        neg_full.setdefault(key, {"Ef": np.full(N, np.nan), "En": np.full(N, np.nan)})
        for j, gi in enumerate(idxs):
            neg_full[key]["Ef"][gi] = d["Ef"][j]
            neg_full[key]["En"][gi] = d["En"][j]

results = {"meta": dict(n_face=int(labels.sum()), n_noface=int((1 - labels).sum()),
                        face_prompts=E.FACE_PROMPTS, nonface_prompts=E.NONFACE_PROMPTS),
           "grids": {}}
for g in grids:
    results["grids"][g] = {}
    for s in schemes:
        rows = []
        for fk in E.FACE_PROMPTS:
            for nk in E.NONFACE_PROMPTS:
                r = E.eval_pair(full[g][s][fk], full[g][s][nk], labels)
                rows.append(dict(face=fk, nonface=nk, **r))
        rows.sort(key=lambda r: (-r["acc"], -r["auc"]))
        results["grids"][g][s] = rows
results["negcfg_broad"] = {k: E.eval_pair(v["Ef"], v["En"], labels) for k, v in neg_full.items()}
json.dump(results, open(os.path.join(HERE, "results_occ", "results_occ.json"), "w"), indent=2)

# ---------- report ----------
FP, NP = E.FACE_PROMPTS, E.NONFACE_PROMPTS
print(f"HARD SETTING: {int(labels.sum())} face-detected / {int((1-labels).sum())} no-face  "
      f"(all from occupation prompts)\n")

print("="*80)
print('USER LITERAL PAIR  "a photo of a face" vs "a photo of a non face"  [cond]')
print("="*80)
print(f"{'grid':<16}{'acc':>7}{'auc':>8}{'margin_face':>13}{'margin_non':>12}")
for g in grids:
    row = next(r for r in results["grids"][g]["cond"] if r["face"]=="F_face" and r["nonface"]=="N_nonface")
    print(f"{g:<16}{row['acc']:>7.3f}{row['auc']:>8.3f}{row['margin_face']:>13.3g}{row['margin_nonface']:>12.3g}")

print("\n" + "="*80)
print("TOP 25 CONDITIONS (grid, scheme, pair) ranked by acc, auc")
print("="*80)
allrows = []
for g in grids:
    for s in schemes:
        for r in results["grids"][g][s]:
            allrows.append((r["acc"], r["auc"], g, s, r["face"], r["nonface"]))
allrows.sort(key=lambda x: (-x[0], -x[1]))
print(f"{'acc':>6}{'auc':>7}  {'grid':<15}{'scheme':<10}{'face':<13}{'nonface'}")
for acc, auc, g, s, fk, nk in allrows[:25]:
    print(f"{acc:>6.3f}{auc:>7.3f}  {g:<15}{s:<10}{fk:<13}{nk}")

print("\n" + "="*80)
print("BEST SCHEME (broad grid, best pair per scheme)")
print("="*80)
for s in schemes:
    b = results["grids"]["broad_50_950"][s][0]
    print(f"  {s:<10} acc={b['acc']:.3f} auc={b['auc']:.3f}  ({b['face']} vs {b['nonface']})")

print("\n" + "="*80)
print("CLASS-AS-NEGATIVE-PROMPT CFG (broad grid)")
print("="*80)
for k, v in results["negcfg_broad"].items():
    print(f"  {k:<36} acc={v['acc']:.3f} auc={v['auc']:.3f}")

# best AUC (threshold-free) too, since acc ties are common on 40 imgs
allrows_auc = sorted(allrows, key=lambda x: (-x[1], -x[0]))
b = allrows_auc[0]
print("\n" + "="*80)
print(f"BEST BY AUC: auc={b[1]:.3f} acc={b[0]:.3f} | grid={b[2]} scheme={b[3]} | "
      f'"{FP.get(b[4],b[4])}" vs "{NP.get(b[5],b[5])}"')
b = allrows[0]
print(f"BEST BY ACC: acc={b[0]:.3f} auc={b[1]:.3f} | grid={b[2]} scheme={b[3]} | "
      f'"{FP.get(b[4],b[4])}" vs "{NP.get(b[5],b[5])}"')
print("="*80)
