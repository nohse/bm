#!/usr/bin/env python
"""Merge sharded run_srr_signal.py outputs into one final folder + build montages.
Usage: python merge_shards.py --shards _shardA,_shardB --out . """
import argparse, json, shutil, csv
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="comma-separated shard dirs (relative to this folder)")
    ap.add_argument("--out", default=".", help="final output dir")
    ap.add_argument("--base_root", default="/workspace/bm/exp-1-debias-gender/srr_signal_exp")
    args = ap.parse_args()
    root = Path(args.base_root)
    shards = [root / s for s in args.shards.split(",")]
    out = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    (out / "base_images").mkdir(parents=True, exist_ok=True)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    (out / "aggregate").mkdir(parents=True, exist_ok=True)

    # ---- move base_images + panels ----
    for sh in shards:
        for sub in ("base_images", "panels"):
            for f in sorted((sh / sub).glob("*.jpg")):
                shutil.copy(f, out / sub / f.name)

    # ---- combine per-image CSV ----
    rows = []
    header = None
    for sh in shards:
        with open(sh / "aggregate" / "per_image.csv") as f:
            r = list(csv.reader(f))
        header = r[0]
        rows += r[1:]
    with open(out / "aggregate" / "per_image.csv", "w") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    # numeric columns -> recompute summary
    col = {name: i for i, name in enumerate(header)}
    def arr(name): return np.array([float(x[col[name]]) for x in rows])
    E0, Eg = arr("E0"), arr("Egrad")
    in_i, out_i = arr("in_region_l1_img"), arr("out_region_l1_img")
    in_l, out_l = arr("in_region_lat"), arr("out_region_lat")
    rf = arr("region_frac")

    # ---- combine spatial accumulators (sums) ----
    diff_sum = sum(np.load(sh / "aggregate" / "diff_accum_sum.npy") for sh in shards)
    mask_sum = sum(np.load(sh / "aggregate" / "mask_accum_sum.npy") for sh in shards)
    n = len(rows)
    avg_diff = diff_sum / n
    avg_mask = mask_sum / n

    def save_heat(a, path, mid="orange", hi="red"):
        an = (a / (a.max() + 1e-8)).clip(0, 1)
        from PIL import ImageOps
        ImageOps.colorize(Image.fromarray((an * 255).astype("uint8"), "L"), black="black", mid=mid, white=hi).save(path)
    save_heat(avg_diff, out / "aggregate" / "avg_srr_change_map.png")
    Image.fromarray(((avg_mask / (avg_mask.max() + 1e-8)) * 255).astype("uint8"), "L").save(
        out / "aggregate" / "avg_person_region.png")

    summary = {
        "n_images": n,
        "E_realistic_mean_before": float(E0.mean()),
        "E_realistic_mean_after_grad": float(Eg.mean()),
        "E_reduction_pct_mean": float(((E0 - Eg) / (E0 + 1e-8) * 100).mean()),
        "in_region_L1_mean_IMAGE": float(in_i.mean()),
        "out_region_L1_mean_IMAGE_vae_spillover": float(out_i.mean()),
        "localization_ratio_image": float(in_i.mean() / (out_i.mean() + 1e-8)),
        "in_region_change_mean_LATENT": float(in_l.mean()),
        "out_region_change_mean_LATENT_should_be_0": float(out_l.mean()),
        "region_frac_mean": float(rf.mean()),
    }
    with open(out / "aggregate" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- montage of base images (all) ----
    def montage(paths, cols, cell=192, path=None):
        paths = list(paths)
        rows_ = (len(paths) + cols - 1) // cols
        g = Image.new("RGB", (cols * cell, rows_ * cell), "white")
        for i, p in enumerate(paths):
            im = Image.open(p).resize((cell, cell))
            g.paste(im, ((i % cols) * cell, (i // cols) * cell))
        g.save(path, quality=85)
    base_paths = sorted((out / "base_images").glob("*.jpg"))
    montage(base_paths, cols=min(20, max(1, n // 10)), path=out / "aggregate" / "montage_base_images.jpg")

    print(json.dumps(summary, indent=2))
    print(f"merged {len(shards)} shards -> {out}  ({n} images, {len(base_paths)} base imgs, "
          f"{len(list((out/'panels').glob('*.jpg')))} panels)")


if __name__ == "__main__":
    main()
