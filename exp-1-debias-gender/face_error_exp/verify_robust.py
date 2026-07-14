#!/usr/bin/env python
"""Adversarial robustness check: re-run the headline conditions across several
independent noise seeds (fresh eps draws) and report mean +/- std of acc/AUC,
so 'AUC=1.0' is confirmed to be the estimator, not one lucky noise draw."""
import os, json, numpy as np, torch
import error_classify as E

E.SCHED = None
seeds = [111, 222, 333, 444]

# conditions to stress-test: (grid_name, face_key, nonface_key)
CONDS = [
    ("high_600_950", "F_portrait",  "N_object"),     # overall best
    ("broad_50_950", "F_portrait",  "N_scenery"),    # robust broad best
    ("broad_50_950", "F_face",      "N_nonface"),    # user's LITERAL pair
    ("high_600_950", "F_face",      "N_nonface"),    # literal, best grid
    ("broad_50_950", "F_humanface", "N_object"),
    ("high_600_950", "F_closeface", "N_noface"),
]

def main():
    manifest = json.load(open(os.path.join(E.HERE, "manifest.json")))
    files = [os.path.join(E.HERE, m["file"]) for m in manifest]
    labels = np.array([1 if m["label"] == "face" else 0 for m in manifest])
    tok, te, vae, unet, E.SCHED = E.load_models()
    z0 = E.encode_images(vae, files)

    need = sorted({k for _, f, n in CONDS for k in (f, n)})
    embeds = {}
    for k in need:
        txt = E.FACE_PROMPTS.get(k, E.NONFACE_PROMPTS.get(k))
        embeds[k] = E.embed(tok, te, txt)
    uncond = E.embed(tok, te, E.UNCOND)

    print(f"{'grid':<14}{'face':<13}{'nonface':<12}{'acc mean+/-std':<18}{'auc mean+/-std'}")
    out = {}
    for gname, fk, nk in CONDS:
        grid = E.GRIDS[gname]
        accs, aucs = [], []
        for sd in seeds:
            tbl = E.score_vocab(unet, z0, {fk: embeds[fk], nk: embeds[nk]}, uncond,
                                grid, sd, chunk=10, cfg_scales=[])
            r = E.eval_pair(tbl["cond"][fk], tbl["cond"][nk], labels)
            accs.append(r["acc"]); aucs.append(r["auc"])
        accs, aucs = np.array(accs), np.array(aucs)
        key = f"{gname}|{fk}|{nk}"
        out[key] = dict(acc_mean=float(accs.mean()), acc_std=float(accs.std()),
                        auc_mean=float(aucs.mean()), auc_std=float(aucs.std()),
                        accs=accs.tolist(), aucs=aucs.tolist())
        print(f"{gname:<14}{fk:<13}{nk:<12}"
              f"{accs.mean():.3f}+/-{accs.std():.3f}     {aucs.mean():.3f}+/-{aucs.std():.3f}")
    json.dump(out, open(os.path.join(E.HERE, "results", "robust.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
