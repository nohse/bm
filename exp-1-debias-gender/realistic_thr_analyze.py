#!/usr/bin/env python
# coding=utf-8
"""
Analyze the realistic-face-loss face/no-face threshold experiment.

Reads the per-rank shard JSONs written by chekc_SCRclip_attmap_grad.py
(--realistic_thr_experiment) and produces:

  analysis/
    per_image.csv                 one row per collected image
    threshold_table_realistic.csv per-threshold confusion metrics (score=realistic loss)
    summary.json / summary.md     best thresholds, AUC, separability, claim evidence
    hist_realistic.png            realistic-loss distribution by class
    scatter_frac_vs_realistic.png attn mask-fraction vs realistic loss (ties claim + fix)
    roc_realistic.png             ROC of "no-face if realistic loss > tau"
    threshold_curves.png          precision/recall/F1/accuracy vs tau
    attn_mask_frac_box.png        attn hard-mask fraction by class (evidence for over-masking)
    contact_face_sorted.jpg       face images sorted by realistic loss
    contact_noface_sorted.jpg     no-face images sorted by realistic loss
    contact_all_sorted.jpg        ALL images sorted by loss, border=class (does it split?)
    selected_50/{face,noface}/    balanced 25+25 (or as many as available) subset

Usage:
    python realistic_thr_analyze.py <out_root>
    python realistic_thr_analyze.py            # auto-pick newest realistic_thr_experiment_* under outputs/
"""
import os
import sys
import csv
import json
import glob
import math
import shutil

import numpy as np

# ---- optional plotting ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as e:  # pragma: no cover
    print(f"[analyze] matplotlib unavailable ({e}); plots skipped, CSV/summary still written.")
    HAVE_MPL = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


# ------------------------------------------------------------------ helpers
def find_out_root(argv):
    if len(argv) > 1:
        return argv[1].rstrip("/")
    cands = glob.glob(os.path.join("outputs", "**", "realistic_thr_experiment_*"), recursive=True)
    cands = [c for c in cands if os.path.isdir(os.path.join(c, "_shards"))]
    if not cands:
        raise SystemExit("No out_root given and none auto-found under outputs/**/realistic_thr_experiment_*")
    cands.sort(key=lambda p: os.path.getmtime(p))
    print(f"[analyze] auto-selected newest: {cands[-1]}")
    return cands[-1]


def load_records(out_root):
    shard_paths = sorted(glob.glob(os.path.join(out_root, "_shards", "shard_rank*.json")))
    if not shard_paths:
        raise SystemExit(f"No shards found under {out_root}/_shards")
    records, metas = [], []
    for p in shard_paths:
        with open(p) as f:
            d = json.load(f)
        metas.append(d.get("meta", {}))
        records.extend(d.get("records", []))
    return records, metas, shard_paths


def auc_mann_whitney(scores, labels):
    """AUC for score predicting label==1 (higher score -> label 1). Tie-corrected."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rank_pos_sum = ranks[labels == 1].sum()
    auc = (rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def threshold_sweep(scores, labels, direction="gt"):
    """Predict label==1 (no-face) if score>tau (direction='gt') or score<tau ('lt').
    Returns list of metric dicts over candidate taus."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    uniq = np.unique(scores)
    if len(uniq) == 1:
        cand = np.array([uniq[0] - 1e-6, uniq[0] + 1e-6])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        cand = np.concatenate([[uniq[0] - 1e-6], mids, [uniq[-1] + 1e-6]])
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    rows = []
    for tau in cand:
        pred = (scores > tau) if direction == "gt" else (scores < tau)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        recall = tp / P if P else float("nan")       # TPR / sensitivity for no-face
        specificity = tn / N if N else float("nan")  # 1-FPR
        fpr = fp / N if N else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        acc = (tp + tn) / (P + N) if (P + N) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and
              np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0) else 0.0
        youden = (recall + specificity - 1.0) if (np.isfinite(recall) and np.isfinite(specificity)) else float("nan")
        rows.append(dict(tau=float(tau), direction=direction, tp=tp, fp=fp, fn=fn, tn=tn,
                         recall_noface=recall, precision_noface=precision, specificity=specificity,
                         fpr=fpr, accuracy=acc, f1=f1, youden_j=youden))
    return rows


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((b.mean() - a.mean()) / sp)


def roc_points(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    P = max(int((labels == 1).sum()), 1)
    N = max(int((labels == 0).sum()), 1)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tps / P])
    fpr = np.concatenate([[0.0], fps / N])
    return fpr, tpr


# ------------------------------------------------------------------ main
def main():
    out_root = find_out_root(sys.argv)
    records, metas, shard_paths = load_records(out_root)
    ana = os.path.join(out_root, "analysis")
    os.makedirs(ana, exist_ok=True)

    if not records:
        raise SystemExit("No records collected.")

    face_mask = np.array([r["face"] for r in records], dtype=bool)
    label = (~face_mask).astype(int)           # 1 = no-face (positive class we detect)
    r_mean = np.array([r["realistic_mean"] for r in records], dtype=np.float64)
    r_std = np.array([r["realistic_std"] for r in records], dtype=np.float64)
    r0 = np.array([r["realistic_draw0"] for r in records], dtype=np.float64)
    frac = np.array([r["attn_mask_frac"] for r in records], dtype=np.float64)

    n_face = int(face_mask.sum())
    n_noface = int((~face_mask).sum())
    print(f"[analyze] records={len(records)} face={n_face} noface={n_noface}")

    # ---- per_image.csv ----
    with open(os.path.join(ana, "per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "idx", "face", "realistic_mean", "realistic_std", "realistic_draw0",
                    "attn_mask_frac", "prompt", "image"])
        for r in records:
            w.writerow([r["rank"], r["idx"], int(r["face"]), f'{r["realistic_mean"]:.6f}',
                        f'{r["realistic_std"]:.6f}', f'{r["realistic_draw0"]:.6f}',
                        f'{r["attn_mask_frac"]:.6f}', r["prompt"], r["image"]])

    # ---- metrics: realistic loss as no-face detector (direction-agnostic) ----
    # auc_raw = P(no-face scores higher than face). If <0.5 the separating rule is INVERTED
    # (no-face has LOWER realistic loss) -- a real, important finding, so we detect direction.
    auc_raw = auc_mann_whitney(r_mean, label)
    auc_raw0 = auc_mann_whitney(r0, label)            # single-draw (training-faithful noise)
    auc_raw_frac = auc_mann_whitney(frac, label)      # attn mask fraction as detector (claim side)

    hi_dir = auc_raw >= 0.5 if np.isfinite(auc_raw) else True
    real_dir = "gt" if hi_dir else "lt"               # 'no-face if loss > tau' vs '< tau'
    auc_real = max(auc_raw, 1.0 - auc_raw) if np.isfinite(auc_raw) else float("nan")
    auc_real0 = max(auc_raw0, 1.0 - auc_raw0) if np.isfinite(auc_raw0) else float("nan")
    auc_frac = max(auc_raw_frac, 1.0 - auc_raw_frac) if np.isfinite(auc_raw_frac) else float("nan")
    real_dir_text = ("no-face has HIGHER realistic loss" if hi_dir
                     else "no-face has LOWER realistic loss (INVERTED vs the naive hypothesis)")

    # sweep in the separating direction (report tau in original loss units)
    sweep = threshold_sweep(r_mean, label, direction=real_dir)

    with open(os.path.join(ana, "threshold_table_realistic.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["tau", "direction", "tp", "fp", "fn", "tn", "recall_noface", "precision_noface",
                "specificity", "fpr", "accuracy", "f1", "youden_j"]
        w.writerow(cols)
        for row in sweep:
            w.writerow([f'{row[c]:.6f}' if isinstance(row[c], float) else row[c] for c in cols])

    valid = [r for r in sweep if r["tp"] + r["fp"] > 0 and r["tp"] + r["fn"] > 0]
    best_f1 = max(valid, key=lambda r: (r["f1"] if np.isfinite(r["f1"]) else -1)) if valid else None
    best_j = max(sweep, key=lambda r: (r["youden_j"] if np.isfinite(r["youden_j"]) else -1))

    def cls_stats(x):
        xf = x[np.isfinite(x)]
        f = x[face_mask & np.isfinite(x)]
        nf = x[(~face_mask) & np.isfinite(x)]
        return dict(
            face_mean=float(f.mean()) if len(f) else float("nan"),
            face_median=float(np.median(f)) if len(f) else float("nan"),
            face_std=float(f.std(ddof=1)) if len(f) > 1 else float("nan"),
            noface_mean=float(nf.mean()) if len(nf) else float("nan"),
            noface_median=float(np.median(nf)) if len(nf) else float("nan"),
            noface_std=float(nf.std(ddof=1)) if len(nf) > 1 else float("nan"),
            cohens_d=cohens_d(f, nf),
        )

    st_real = cls_stats(r_mean)
    st_frac = cls_stats(frac)
    # relative measurement noise of the realistic loss (mean of per-image std / per-image mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_noise = float(np.nanmean(r_std / np.abs(r_mean)))

    summary = dict(
        out_root=out_root,
        counts=dict(total=len(records), face=n_face, noface=n_noface),
        config=metas[0] if metas else {},
        shard_metas=metas,
        realistic_loss=dict(
            auc_separation=auc_real,          # direction-agnostic (max(auc,1-auc))
            auc_raw_noface_if_higher=auc_raw,  # <0.5 means no-face has LOWER loss
            auc_single_draw=auc_real0,
            direction=real_dir_text,
            rule=f"predict no-face if realistic_loss {'>' if real_dir == 'gt' else '<'} tau",
            class_stats=st_real,
            mean_relative_measure_noise=rel_noise,
            best_threshold_f1=best_f1,
            best_threshold_youden=best_j,
        ),
        attn_mask_fraction=dict(
            auc_separation=auc_frac,
            auc_raw_noface_if_higher=auc_raw_frac,
            class_stats=st_frac,
            note="Over-masking claim: does the attn hard-mask cover MORE of no-face images? "
                 "auc_raw>0.5 -> yes (no-face masked more); ~0.5 -> similar; <0.5 -> face masked more.",
        ),
    )
    with open(os.path.join(ana, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- summary.md ----
    def fmt(x):
        return "nan" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.4f}"
    lines = []
    lines.append(f"# Realistic-face-loss threshold experiment\n")
    lines.append(f"- out_root: `{out_root}`")
    lines.append(f"- collected: **{len(records)}** images  (face={n_face}, no-face={n_noface})")
    m0 = metas[0] if metas else {}
    lines.append(f"- checkpoint step: {m0.get('ckpt_step')}  | weights: {m0.get('weights')} "
                 f"| denoise steps: {m0.get('num_denoise')} | skip_pct: {m0.get('skip_pct')} "
                 f"| realistic repeats R: {m0.get('R')}")
    hit = any(mm.get("hit_cap") for mm in metas)
    if n_noface < (m0.get("per_rank_target", 25) * len(metas)) or hit:
        lines.append(f"- ⚠️ note: pool cap hit on some rank={hit}; no-face harvested={n_noface} "
                     f"(target/class total={m0.get('per_rank_target', '?')}×{len(metas)} ranks).")
    lines.append("")
    lines.append("## Can the realistic-face SDS loss separate face vs no-face?\n")
    cmp = ">" if real_dir == "gt" else "<"
    lines.append(f"- **separation AUC (direction-agnostic), R-averaged loss: {fmt(auc_real)}**  "
                 f"(single-draw: {fmt(auc_real0)}; raw AUC[no-face if loss↑]={fmt(auc_raw)})")
    lines.append(f"- direction: **{real_dir_text}** → detection rule: `no-face if loss {cmp} τ`")
    lines.append(f"- realistic loss — face: mean {fmt(st_real['face_mean'])} ± {fmt(st_real['face_std'])}, "
                 f"median {fmt(st_real['face_median'])}")
    lines.append(f"- realistic loss — no-face: mean {fmt(st_real['noface_mean'])} ± {fmt(st_real['noface_std'])}, "
                 f"median {fmt(st_real['noface_median'])}")
    lines.append(f"- Cohen's d (no-face − face): {fmt(st_real['cohens_d'])}  "
                 f"| mean per-image relative measurement noise: {fmt(rel_noise)}")
    if best_f1:
        lines.append(f"- best threshold by F1: **loss {cmp} {fmt(best_f1['tau'])}** → "
                     f"F1={fmt(best_f1['f1'])}, recall(no-face)={fmt(best_f1['recall_noface'])}, "
                     f"precision={fmt(best_f1['precision_noface'])}, acc={fmt(best_f1['accuracy'])}")
    lines.append(f"- best threshold by Youden J: loss {cmp} {fmt(best_j['tau'])} → "
                 f"J={fmt(best_j['youden_j'])}, recall={fmt(best_j['recall_noface'])}, "
                 f"specificity={fmt(best_j['specificity'])}")
    lines.append("")
    lines.append("## Over-masking claim (attn hard-mask fraction)\n")
    lines.append(f"- attn mask fraction @thr={m0.get('mask_threshold')} — "
                 f"face: {fmt(st_frac['face_mean'])} ± {fmt(st_frac['face_std'])}  vs  "
                 f"no-face: {fmt(st_frac['noface_mean'])} ± {fmt(st_frac['noface_std'])}")
    lines.append(f"- no-face images get the img-loss gradient scaled by `factor2` over "
                 f"~{fmt(100*st_frac['noface_mean'])}% of the image, vs ~{fmt(100*st_frac['face_mean'])}% for faces "
                 f"(raw AUC[no-face if frac↑]={fmt(auc_raw_frac)}, separation AUC={fmt(auc_frac)}).")
    if np.isfinite(auc_raw_frac):
        if auc_raw_frac >= 0.65:
            lines.append("- → data supports the claim: no-face images ARE masked over a larger area.")
        elif auc_raw_frac <= 0.35:
            lines.append("- → data INVERTS the claim here: face images are masked more on average "
                         "(but note: even ~40%+ of *face* images being masked is itself large, "
                         "and the min-max mechanism is still magnitude-blind).")
        else:
            lines.append("- → mask fraction is SIMILAR across classes: the min-max mask does not "
                         "cleanly localize to faces even when a face is present (both get large masks); "
                         "the mechanism is magnitude-blind regardless.")
    lines.append("")
    lines.append("## Verdict\n")
    if np.isfinite(auc_real):
        if auc_real >= 0.9:
            v = "STRONG separation — a realistic-loss threshold is a viable face/no-face gate."
        elif auc_real >= 0.75:
            v = "MODERATE separation — a threshold works but with meaningful error; consider z-scoring / more repeats."
        else:
            v = "WEAK separation — a single global realistic-loss threshold is NOT a reliable gate as-is."
        lines.append(f"- {v}")
    with open(os.path.join(ana, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- plots ----
    if HAVE_MPL:
        _plots(ana, r_mean, r0, frac, face_mask, label, sweep, best_f1, best_j,
               auc_real, m0, real_dir)

    # ---- contact sheets + selected_50 ----
    if HAVE_PIL:
        _contact_sheets(out_root, ana, records, face_mask, r_mean)
        _selected_50(out_root, ana, records, face_mask, r_mean,
                     per_class=m0.get("per_rank_target", 25) and 25)

    print(f"\n[analyze] wrote analysis to: {ana}")


def _plots(ana, r_mean, r0, frac, face_mask, label, sweep, best_f1, best_j, auc_real, meta, real_dir="gt"):
    cmp = ">" if real_dir == "gt" else "<"
    f_vals = r_mean[face_mask]
    nf_vals = r_mean[~face_mask]

    # histogram
    plt.figure(figsize=(7, 4.5))
    lo, hi = np.nanmin(r_mean), np.nanmax(r_mean)
    bins = np.linspace(lo, hi, 30)
    plt.hist(f_vals, bins=bins, alpha=0.6, label=f"face (n={len(f_vals)})", color="#2c7fb8")
    plt.hist(nf_vals, bins=bins, alpha=0.6, label=f"no-face (n={len(nf_vals)})", color="#d95f0e")
    if best_f1:
        plt.axvline(best_f1["tau"], color="k", ls="--", lw=1.2, label=f"best-F1 τ={best_f1['tau']:.3f}")
    plt.xlabel("realistic-face SDS loss (R-averaged)")
    plt.ylabel("count")
    plt.title(f"Realistic-loss distribution by class (AUC={auc_real:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ana, "hist_realistic.png"), dpi=130)
    plt.close()

    # scatter attn frac vs realistic loss
    plt.figure(figsize=(7, 5))
    plt.scatter(r_mean[face_mask], frac[face_mask], s=22, alpha=0.7, color="#2c7fb8", label="face")
    plt.scatter(r_mean[~face_mask], frac[~face_mask], s=22, alpha=0.7, color="#d95f0e", label="no-face")
    if best_f1:
        plt.axvline(best_f1["tau"], color="k", ls="--", lw=1.0)
    plt.xlabel("realistic-face SDS loss (R-averaged)")
    plt.ylabel(f"attn hard-mask fraction @thr={meta.get('mask_threshold')}")
    plt.title("Fix signal (x) vs over-masking symptom (y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ana, "scatter_frac_vs_realistic.png"), dpi=130)
    plt.close()

    # ROC (oriented so the separating direction is above the diagonal)
    oriented = r_mean if real_dir == "gt" else -r_mean
    fpr, tpr = roc_points(oriented, label)
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(fpr, tpr, "-o", ms=2, color="#d95f0e",
             label=f"realistic loss, rule 'loss {cmp} τ' (AUC={auc_real:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR (face flagged as no-face)")
    plt.ylabel("TPR (no-face recall)")
    plt.title("ROC: detect no-face via realistic loss")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(ana, "roc_realistic.png"), dpi=130)
    plt.close()

    # threshold curves
    taus = np.array([r["tau"] for r in sweep])
    plt.figure(figsize=(7.5, 4.5))
    for key, col in [("recall_noface", "#d95f0e"), ("precision_noface", "#2c7fb8"),
                     ("f1", "#31a354"), ("accuracy", "#756bb1")]:
        plt.plot(taus, [r[key] for r in sweep], label=key, color=col)
    if best_f1:
        plt.axvline(best_f1["tau"], color="k", ls="--", lw=1.0, label=f"best-F1 τ={best_f1['tau']:.3f}")
    plt.xlabel(f"threshold τ  (predict no-face if loss {cmp} τ)")
    plt.ylabel("metric")
    plt.ylim(-0.02, 1.02)
    plt.title("Threshold sweep")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(ana, "threshold_curves.png"), dpi=130)
    plt.close()

    # attn frac box by class
    plt.figure(figsize=(5, 4.5))
    data = [frac[face_mask][np.isfinite(frac[face_mask])], frac[~face_mask][np.isfinite(frac[~face_mask])]]
    plt.boxplot(data, labels=["face", "no-face"], showmeans=True)
    plt.ylabel(f"attn hard-mask fraction @thr={meta.get('mask_threshold')}")
    plt.title("Over-masking: mask fraction by class")
    plt.tight_layout()
    plt.savefig(os.path.join(ana, "attn_mask_frac_box.png"), dpi=130)
    plt.close()


def _load_thumb(path, size=132):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    return img


def _sheet(items, out_path, cols=10, thumb=132, pad=6, title=""):
    """items: list of (image_path, caption, border_rgb or None)."""
    if not items:
        return
    n = len(items)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    cap_h = 16
    top = 24 if title else 0
    cell_w = thumb + pad
    cell_h = thumb + cap_h + pad
    W = cols * cell_w + pad
    H = rows * cell_h + pad + top
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    if title:
        draw.text((pad, 6), title, fill="black", font=font)
    for i, (p, cap, border) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * cell_w
        y = top + pad + r * cell_h
        try:
            th = _load_thumb(p, thumb)
        except Exception:
            th = Image.new("RGB", (thumb, thumb), "gray")
        sheet.paste(th, (x, y))
        if border is not None:
            draw.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline=border, width=3)
        draw.text((x, y + thumb + 1), cap, fill="black", font=font)
    sheet.save(out_path, quality=92)


def _contact_sheets(out_root, ana, records, face_mask, r_mean):
    order = np.argsort(r_mean)
    face_items, noface_items, all_items = [], [], []
    for i in order:
        r = records[i]
        p = os.path.join(out_root, r["image"])
        cap = f"{r['realistic_mean']:.3f}"
        if r["face"]:
            face_items.append((p, cap, (44, 127, 184)))
            all_items.append((p, cap, (44, 127, 184)))     # blue = face
        else:
            noface_items.append((p, cap, (217, 95, 14)))
            all_items.append((p, cap, (217, 95, 14)))       # orange = no-face
    _sheet(face_items, os.path.join(ana, "contact_face_sorted.jpg"),
           title="FACE images sorted by realistic loss (low->high)")
    _sheet(noface_items, os.path.join(ana, "contact_noface_sorted.jpg"),
           title="NO-FACE images sorted by realistic loss (low->high)")
    _sheet(all_items, os.path.join(ana, "contact_all_sorted.jpg"),
           title="ALL sorted by realistic loss; blue=face orange=no-face (clean split = clean color band)")


def _selected_50(out_root, ana, records, face_mask, r_mean, per_class=25):
    per_class = per_class or 25
    sel_dir = os.path.join(ana, "selected_50")
    for sub in ("face", "noface"):
        d = os.path.join(sel_dir, sub)
        os.makedirs(d, exist_ok=True)
    face_idx = [i for i in range(len(records)) if records[i]["face"]][:per_class]
    noface_idx = [i for i in range(len(records)) if not records[i]["face"]][:per_class]
    for i in face_idx + noface_idx:
        r = records[i]
        src = os.path.join(out_root, r["image"])
        sub = "face" if r["face"] else "noface"
        dst = os.path.join(sel_dir, sub, os.path.basename(r["image"]))
        try:
            shutil.copyfile(src, dst)
        except Exception as e:
            print(f"[analyze] copy fail {src}: {e}")
    print(f"[analyze] selected_50: face={len(face_idx)} noface={len(noface_idx)}")


if __name__ == "__main__":
    main()
