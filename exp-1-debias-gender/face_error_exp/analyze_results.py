#!/usr/bin/env python
"""Summarize results.json: rank conditions by accuracy/AUC and print the headline findings."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "results.json")))
FP, NP = R["meta"]["face_prompts"], R["meta"]["nonface_prompts"]
print(f"Dataset: {R['meta']['n_face']} face / {R['meta']['n_nonface']} nonface  (100 total)\n")

# 1) The user's LITERAL pair, cond scheme, across grids
print("="*76)
print('USER LITERAL PAIR  "a photo of a face"  vs  "a photo of a non face"  [cond]')
print("="*76)
print(f"{'grid':<16}{'acc':>7}{'auc':>8}{'margin_face':>13}{'margin_non':>12}")
for g, gr in R["grids"].items():
    row = next(r for r in gr["cond"] if r["face"]=="F_face" and r["nonface"]=="N_nonface")
    print(f"{g:<16}{row['acc']:>7.3f}{row['auc']:>8.3f}{row['margin_face']:>13.4g}{row['margin_nonface']:>12.4g}")

# 2) Global best pair per (grid, scheme)
print("\n" + "="*76)
print("BEST PROMPT PAIR for each (grid, scheme)  -- ranked within cell by acc,auc")
print("="*76)
rows = []
for g, gr in R["grids"].items():
    for scheme, lst in gr.items():
        b = lst[0]
        rows.append((b["acc"], b["auc"], g, scheme, b["face"], b["nonface"]))
rows.sort(key=lambda x: (-x[0], -x[1]))
print(f"{'acc':>6}{'auc':>7}  {'grid':<15}{'scheme':<10}{'face':<14}{'nonface'}")
for acc, auc, g, scheme, fk, nk in rows[:25]:
    print(f"{acc:>6.3f}{auc:>7.3f}  {g:<15}{scheme:<10}{fk:<14}{nk}")

# 3) cond scheme: how does each grid do at the user's pair vs best pair
print("\n" + "="*76)
print("SCHEME COMPARISON on broad_50_950 grid (best pair per scheme)")
print("="*76)
gr = R["grids"]["broad_50_950"]
for scheme in ["cond","eps_diff","cfg2.0","cfg4.0","cfg7.5"]:
    if scheme in gr:
        b = gr[scheme][0]
        print(f"  {scheme:<10} acc={b['acc']:.3f} auc={b['auc']:.3f}  ({b['face']} vs {b['nonface']})")

# 4) negcfg (class-as-negative-prompt)
if "negcfg_broad" in R:
    print("\n" + "="*76)
    print("CLASS-AS-NEGATIVE-PROMPT CFG (broad grid)")
    print("="*76)
    for k, v in R["negcfg_broad"].items():
        print(f"  {k:<36} acc={v['acc']:.3f} auc={v['auc']:.3f}")

# 5) overall best
best = rows[0]
print("\n" + "="*76)
print(f"OVERALL BEST: acc={best[0]:.3f} auc={best[1]:.3f} | grid={best[2]} scheme={best[3]} | "
      f'"{FP.get(best[4],best[4])}" vs "{NP.get(best[5],best[5])}"')
print("="*76)
