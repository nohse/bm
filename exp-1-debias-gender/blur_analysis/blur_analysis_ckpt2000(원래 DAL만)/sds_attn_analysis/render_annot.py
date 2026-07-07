#!/usr/bin/env python
# coding=utf-8
"""Re-render annotated images + montages from the saved SDS results (no GPU)."""
import os, json, math, glob
from PIL import Image, ImageDraw, ImageFont

ROOT = "/workspace/finetune-fair-diffusion/exp-1-debias-gender/blur_analysis_ckpt2000(원래 DAL만)"
OUT = os.path.join(ROOT, "sds_attn_analysis")
FONT_PATH = "/workspace/finetune-fair-diffusion/data/0-utils/arial-bold.ttf"
SRC = {"ori": os.path.join(ROOT, "clean", "images_ori"),
       "ft":  os.path.join(ROOT, "clean", "images_ft")}

res = json.load(open(os.path.join(OUT, "sds_attn_results.json")))


def font(sz):
    try: return ImageFont.truetype(FONT_PATH, sz)
    except Exception: return ImageFont.load_default()


def fit_font(draw, text, max_w, start=26, lo=12):
    for sz in range(start, lo - 1, -1):
        f = font(sz)
        if draw.textlength(text, font=f) <= max_w:
            return f
    return font(lo)


def annotate(path, r, strip_h=108):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    pred_woman = (r["pred"] == 0)
    col = (220, 40, 40) if pred_woman else (40, 90, 220)   # red=woman, blue=man
    bw = 8
    cv = Image.new("RGB", (W + 2 * bw, H + 2 * bw + strip_h), col)
    cv.paste(im, (bw, bw))
    d = ImageDraw.Draw(cv)
    d.rectangle([bw, H + bw, W + bw, H + bw + strip_h], fill=(255, 255, 255))
    maxw = W - 16
    pred_str = "WOMAN" if pred_woman else "MAN"
    l1 = f"SDS_f(woman)={r['sds_f']:.5f}   SDS_m(man)={r['sds_m']:.5f}"
    l2 = f"P(woman)={r['p_woman']:.3f}   P(man)={r['p_man']:.3f}   -> {pred_str}"
    f1 = fit_font(d, l1, maxw, start=24)
    f2 = fit_font(d, l2, maxw, start=24)
    d.text((bw + 8, H + bw + 10), l1, fill=(0, 0, 0), font=f1)
    d.text((bw + 8, H + bw + 54), l2, fill=col, font=f2)
    return cv


def montage(imgs, cols, save_to, pad=6, bg=(30, 30, 30)):
    if not imgs: return
    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / cols)
    g = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, im in enumerate(imgs):
        rr, cc = divmod(i, cols)
        g.paste(im, (pad + cc * (w + pad), pad + rr * (h + pad)))
    g.save(save_to)


for tag in ("ori", "ft"):
    if tag not in res: continue
    rows = res[tag]["rows"]
    by_file = {r["file"]: r for r in rows}
    ann_dir = os.path.join(OUT, f"annotated_{tag}"); os.makedirs(ann_dir, exist_ok=True)
    mont_dir = os.path.join(OUT, f"montage_{tag}"); os.makedirs(mont_dir, exist_ok=True)
    by_occ = {}
    for fname in sorted(by_file):
        r = by_file[fname]
        ann = annotate(os.path.join(SRC[tag], fname), r)
        ann.save(os.path.join(ann_dir, fname))
        by_occ.setdefault(r["occ_id"], []).append((fname, ann))
    for occ_id in sorted(by_occ):
        occ_name = by_file[by_occ[occ_id][0][0]]["occ"]
        montage([a for _, a in by_occ[occ_id]], 10, os.path.join(mont_dir, f"{occ_id}_{occ_name}.png"))
    montage([a for occ_id in sorted(by_occ) for _, a in by_occ[occ_id]], 10,
            os.path.join(OUT, f"montage_all_{tag}.png"))
    print(f"[{tag}] re-rendered {len(rows)} images, {len(by_occ)} occ montages")
print("done")
