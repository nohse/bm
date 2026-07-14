#!/usr/bin/env python
"""Organize the 40 evaluated images and split by the BEST condition's prediction.

Best condition (TRUE/hard setting):
    grid = repo_400_800, scheme = cond,
    face prompt    = "a photo of a face"
    nonface prompt = "a photo without a face"

Prediction per image = averaged over several noise seeds (robust estimate):
    E_face   = mean_seed mean_t || eps_hat(z_t,t,"a photo of a face")   - eps ||^2
    E_noface = mean_seed mean_t || eps_hat(z_t,t,"a photo without a face") - eps ||^2
    predict FACE  iff  E_face < E_noface   (margin = E_noface - E_face > 0)

Writes:
  eval_set/true_face/     20 ground-truth face images
  eval_set/true_noface/   20 ground-truth no-face images
  eval_set/pred_face/     images the classifier called FACE   (filename marks truth+margin)
  eval_set/pred_noface/   images the classifier called NO-FACE
  eval_set/predictions.csv, montages, summary.txt
"""
import os, json, shutil, csv
import numpy as np
import error_classify as E
from PIL import Image

HERE = E.HERE
SEEDS = [111, 222, 333, 444, 555]
GRID = "repo_400_800"
FACE_TXT, NON_TXT = "a photo of a face", "a photo without a face"
OUT = os.path.join(HERE, "eval_set")

def occ_of(m):
    p = m.get("occupation", "") or ""
    if "face of a " in p:
        return p.split("face of a ")[1].split(", a person")[0]
    return p if p else os.path.basename(m["file"]).replace(".png", "")

def reset(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)

def main():
    manifest = json.load(open(os.path.join(HERE, "manifest_true.json")))
    true = [m for m in manifest if m["group"] == "true"]
    files = [os.path.join(HERE, m["file"]) for m in true]
    y = np.array([1 if m["true_label"] == "face" else 0 for m in true])  # 1=face

    tok, te, vae, unet, E.SCHED = E.load_models()
    z0 = E.encode_images(vae, files)
    ef = E.embed(tok, te, FACE_TXT)
    en = E.embed(tok, te, NON_TXT)
    uncond = E.embed(tok, te, "")

    # average E over seeds (cond scheme only)
    Ef = np.zeros(len(true)); En = np.zeros(len(true))
    for sd in SEEDS:
        tbl = E.score_vocab(unet, z0, {"F": ef, "N": en}, uncond, E.GRIDS[GRID], sd,
                            chunk=8, cfg_scales=[])
        Ef += tbl["cond"]["F"]; En += tbl["cond"]["N"]
    Ef /= len(SEEDS); En /= len(SEEDS)
    margin = En - Ef                 # >0 => predict face
    pred = (margin > 0).astype(int)

    # folders
    for sub in ["true_face", "true_noface", "pred_face", "pred_noface"]:
        reset(os.path.join(OUT, sub))

    rows = []
    for i, m in enumerate(true):
        occ = occ_of(m).replace("/", "-").replace(" ", "_")[:40]
        truth = "face" if y[i] == 1 else "noface"
        pr = "face" if pred[i] == 1 else "noface"
        correct = (pred[i] == y[i])
        # ground-truth folders (clean names)
        shutil.copy(files[i], os.path.join(OUT, f"true_{truth}", f"{occ}.png"))
        # prediction folders (name encodes truth + correctness + margin)
        tag = "OK" if correct else "WRONG"
        fname = f"{tag}__truth-{truth}__{occ}__margin{margin[i]:+.2e}.png"
        shutil.copy(files[i], os.path.join(OUT, f"pred_{pr}", fname))
        rows.append(dict(file=m["file"], occupation=occ_of(m), truth=truth,
                         pred=pr, correct=bool(correct),
                         E_face=round(float(Ef[i]), 8), E_noface=round(float(En[i]), 8),
                         margin=round(float(margin[i]), 10)))

    with open(os.path.join(OUT, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # montages of predicted groups
    def montage(recs, path, cols=8, thumb=140):
        if not recs:
            return
        rows_n = (len(recs) + cols - 1) // cols
        cv = Image.new("RGB", (cols * thumb, rows_n * thumb), "white")
        for i, r in enumerate(recs):
            im = Image.open(os.path.join(HERE, r["file"])).convert("RGB").resize((thumb, thumb))
            cv.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
        cv.save(path)
    montage([r for r in rows if r["pred"] == "face"], os.path.join(OUT, "montage_pred_face.png"))
    montage([r for r in rows if r["pred"] == "noface"], os.path.join(OUT, "montage_pred_noface.png"))

    # confusion
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    summ = (
        f"BEST condition: grid={GRID}, scheme=cond, "
        f'"{FACE_TXT}" vs "{NON_TXT}"  (prediction averaged over {len(SEEDS)} seeds)\n\n'
        f"                 pred FACE   pred NO-FACE\n"
        f"  truth FACE  :   {tp:>4}         {fn:>4}      (TP / FN)\n"
        f"  truth NOFACE:   {fp:>4}         {tn:>4}      (FP / TN)\n\n"
        f"  accuracy = {acc:.3f} ({tp+tn}/{len(y)})\n"
        f"  pred_face folder    : {tp+fp} images ({tp} correct faces, {fp} wrong no-faces)\n"
        f"  pred_noface folder  : {tn+fn} images ({tn} correct no-faces, {fn} missed faces)\n"
    )
    print(summ)
    open(os.path.join(OUT, "summary.txt"), "w").write(summ)

if __name__ == "__main__":
    main()
