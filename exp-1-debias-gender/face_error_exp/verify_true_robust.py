#!/usr/bin/env python
"""Multi-seed robustness on the TRUE 40-image set (20 genuine-face vs 20 genuine-noface).
Guards against the in-sample optimism of picking the top of a big sweep: re-scores the
leading conditions with independent noise draws and reports mean +/- std acc/AUC."""
import os, json, numpy as np
import error_classify as E

seeds = [111, 222, 333, 444, 555]
# (grid, scheme, face_key, nonface_key)
CONDS = [
    ("repo_400_800", "cond",     "F_face",      "N_withoutface"),  # best AUC in sweep
    ("broad_50_950", "eps_diff", "F_face",      "N_nonface"),      # best acc in sweep
    ("broad_50_950", "cond",     "F_face",      "N_nonface"),      # user literal, cond
    ("repo_400_800", "cond",     "F_humanface", "N_withoutface"),
    ("broad_50_950", "cond",     "F_face",      "N_withoutface"),
    ("high_600_950", "cond",     "F_portrait",  "N_object"),       # easy-setting winner: transfers?
    ("repo_400_800", "cond",     "F_face",      "N_nonface"),
]

def main():
    manifest = json.load(open(os.path.join(E.HERE, "manifest_true.json")))
    keep = [i for i, m in enumerate(manifest) if m["group"] == "true"]
    files = [os.path.join(E.HERE, manifest[i]["file"]) for i in keep]
    labels = np.array([1 if manifest[i]["true_label"] == "face" else 0 for i in keep])
    tok, te, vae, unet, E.SCHED = E.load_models()
    z0 = E.encode_images(vae, files)

    need = sorted({k for _, _, f, n in CONDS for k in (f, n)})
    embeds = {k: E.embed(tok, te, E.FACE_PROMPTS.get(k, E.NONFACE_PROMPTS.get(k))) for k in need}
    uncond = E.embed(tok, te, E.UNCOND)
    grids_needed = sorted({c[0] for c in CONDS})

    # per seed, per grid: score all needed prompts (cond+eps_diff+cfg) once
    rec = {c: {"acc": [], "auc": []} for c in CONDS}
    for sd in seeds:
        scored = {g: E.score_vocab(unet, z0, embeds, uncond, E.GRIDS[g], sd, chunk=8, cfg_scales=[])
                  for g in grids_needed}
        for c in CONDS:
            g, sch, fk, nk = c
            r = E.eval_pair(scored[g][sch][fk], scored[g][sch][nk], labels)
            rec[c]["acc"].append(r["acc"]); rec[c]["auc"].append(r["auc"])

    print(f"multi-seed ({len(seeds)} seeds) robustness on 40 TRUE images\n")
    print(f"{'grid':<14}{'scheme':<10}{'face':<12}{'nonface':<14}{'acc':>16}{'auc':>16}")
    out = {}
    for c in CONDS:
        g, sch, fk, nk = c
        a = np.array(rec[c]["acc"]); u = np.array(rec[c]["auc"])
        out["|".join(c)] = dict(acc_mean=float(a.mean()), acc_std=float(a.std()),
                                auc_mean=float(u.mean()), auc_std=float(u.std()))
        print(f"{g:<14}{sch:<10}{fk:<12}{nk:<14}"
              f"{a.mean():>8.3f}+/-{a.std():<5.3f}{u.mean():>8.3f}+/-{u.std():<5.3f}")
    json.dump(out, open(os.path.join(E.HERE, "results_true", "robust_true.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
