"""
Rebuild the per-prompt clean montages (original SD-1.5 vs finetuned EMA TE-LoRA),
adding under each image the SDS scores to 5 decimals:
    sds_gender (colored)   sf=<s_f>  sm=<s_m>  real=<s_real>
Source images: images_ori/ , images_ft/   Scores: clean_results.json
"""
import os, json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MAN_C, WOMAN_C = "#1f6fd0", "#e0457b"     # blue / pink, matching original montage labels

def load():
    with open(os.path.join(HERE, "clean_results.json")) as f:
        d = json.load(f)
    return d["rows_ori"], d["rows_ft"]

def by_occ(rows):
    g = {}
    for r in rows:
        g.setdefault(r["file"][:3], {})[int(r["idx"])] = r   # key "o00".. -> {idx: row}
    return g

def montage(oi, occ, rows_o, rows_f, outdir, only=None):
    NPER = 10
    fig, axes = plt.subplots(2, NPER, figsize=(2.55 * NPER + 1.4, 8.9))
    fig.suptitle(f'o{oi:02d}   "{occ}"', fontsize=23, fontweight="bold", y=0.985)
    rowmeta = [("original", "SD-1.5",      "#2b6cb0", "images_ori", rows_o),
               ("finetuned", "EMA TE-LoRA", "#d35400", "images_ft", rows_f)]
    for r, (lbl, sub, lblc, imgdir, rows) in enumerate(rowmeta):
        for c in range(NPER):
            ax = axes[r, c]
            row = rows[c]
            im = Image.open(os.path.join(HERE, imgdir, row["file"] + ".png"))
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#cccccc"); s.set_linewidth(0.8)
            if r == 0:
                ax.set_title(f"n{c:02d}", fontsize=14, pad=4)
            g = row["sds_gender"]
            gc = MAN_C if g == "man" else WOMAN_C
            # gender (colored) just under the image
            ax.text(0.5, -0.045, g, transform=ax.transAxes, ha="center", va="top",
                    fontsize=13, fontweight="bold", color=gc)
            # the three SDS scores to 5 decimals, black monospace
            txt = (f"sf   {row['sds_f']:.5f}\n"
                   f"sm   {row['sds_m']:.5f}\n"
                   f"real {row['sds_real']:.5f}")
            ax.text(0.5, -0.175, txt, transform=ax.transAxes, ha="center", va="top",
                    fontsize=10.5, family="monospace", color="#111111", linespacing=1.25)
        # left row label, vertically centered on this row of images
        ycen = axes[r, 0].get_position().y0 + axes[r, 0].get_position().height / 2
        fig.text(0.013, ycen + 0.018, lbl,  fontsize=15, fontweight="bold",
                 color=lblc, ha="left", va="center")
        fig.text(0.013, ycen - 0.022, sub,  fontsize=11, color="#666666",
                 ha="left", va="center")
    fig.subplots_adjust(left=0.052, right=0.995, top=0.94, bottom=0.12,
                        wspace=0.06, hspace=0.46)
    slug = "".join(c if c.isalnum() else "_" for c in occ)[:40]
    out = os.path.join(outdir, f"montage_o{oi:02d}_{slug}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "montages_per_prompt_sds"))
    ap.add_argument("--only", type=int, default=None, help="render only this occ index")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows_o, rows_f = load()
    go, gf = by_occ(rows_o), by_occ(rows_f)
    # occupation name + index, in occ order, from the rows themselves
    seen = {}
    for r in rows_o:
        oi = int(r["file"][1:3])
        seen.setdefault(oi, r["occ"])
    for oi in sorted(seen):
        if args.only is not None and oi != args.only:
            continue
        key = f"o{oi:02d}"
        out = montage(oi, seen[oi], go[key], gf[key], args.out)
        print("wrote", out, flush=True)

if __name__ == "__main__":
    main()
