#!/usr/bin/env python3
"""Aggregate per-occupation CLIP-T JSONs (produced by clip_t_by_occupation_eval.py)
into a side-by-side table + overall averages. Usage:

    python clip_t_by_occupation_summary.py \
        --json 400=clip_t_by_occupation_out/clip_t_ckpt-400.json \
        --json 1200=clip_t_by_occupation_out/clip_t_ckpt-1200.json \
        --csv clip_t_by_occupation_out/clip_t_by_occupation.csv
"""
import argparse, json, csv, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", default=[],
                    help="LABEL=path.json (repeatable), e.g. 400=.../clip_t_ckpt-400.json")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args()

    labels, data = [], {}
    for spec in args.json:
        label, path = spec.split("=", 1)
        labels.append(label)
        with open(path) as f:
            data[label] = json.load(f)

    # union of occupations, ordered by first file's occurrence then any extras
    occ_order = []
    for label in labels:
        for occ in data[label]["per_occupation"].keys():
            if occ not in occ_order:
                occ_order.append(occ)

    def occ_mean(label, occ):
        d = data[label]["per_occupation"].get(occ)
        return d["clip_t_mean"] if d else None

    # ---- per-occupation table ----
    w = max((len(o) for o in occ_order), default=10)
    header = f"{'occupation':<{w}}  " + "  ".join(f"{'clipT@'+l:>12}" for l in labels)
    if len(labels) == 2:
        header += f"  {'Δ('+labels[1]+'-'+labels[0]+')':>14}"
    print(header)
    print("-" * len(header))
    for occ in occ_order:
        vals = [occ_mean(l, occ) for l in labels]
        row = f"{occ:<{w}}  " + "  ".join(
            (f"{v:>12.4f}" if v is not None else f"{'--':>12}") for v in vals)
        if len(labels) == 2 and all(v is not None for v in vals):
            row += f"  {vals[1]-vals[0]:>+14.4f}"
        print(row)
    print("-" * len(header))

    # ---- overall averages ----
    print("\n=== OVERALL AVERAGES ===")
    for label in labels:
        d = data[label]
        n = d.get("num_occupations")
        ipo = d.get("images_per_occupation")
        print(f"[{label}]  step={d.get('step')}  weights={d.get('weights')}  "
              f"#occ={n}  imgs/occ={ipo}")
        print(f"        mean over IMAGES      = {d['overall_clip_t_mean_over_images']:.5f}")
        print(f"        mean over OCCUPATIONS = {d['overall_clip_t_mean_over_occupations']:.5f}")
    if len(labels) == 2:
        a, b = labels
        da = data[a]["overall_clip_t_mean_over_occupations"]
        db = data[b]["overall_clip_t_mean_over_occupations"]
        print(f"\nΔ overall (occ-avg) {b} - {a} = {db-da:+.5f}")

    # ---- CSV ----
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            wtr = csv.writer(f)
            head = ["occupation"] + [f"clip_t_{l}" for l in labels]
            if len(labels) == 2:
                head += [f"delta_{labels[1]}_minus_{labels[0]}"]
            wtr.writerow(head)
            for occ in occ_order:
                vals = [occ_mean(l, occ) for l in labels]
                row = [occ] + [("" if v is None else f"{v:.6f}") for v in vals]
                if len(labels) == 2 and all(v is not None for v in vals):
                    row += [f"{vals[1]-vals[0]:.6f}"]
                wtr.writerow(row)
        print(f"\n[csv] wrote {args.csv}")


if __name__ == "__main__":
    main()
