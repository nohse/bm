"""Toy experiment: replace the hard SCR gradient gate (min-max attn >= 0.15 -> x0.2) with soft masks.

Runs on CPU over 50 REAL saved scoring-time woman/man cross-attention maps (t=400-800, SD1.5) -- i.e.
exactly the `common_attn` the training file computes -- plus the 512x512 images they came from and
insightface face boxes as the localization ground truth.

    m = 1 - d * s        d = 1 - factor2 = 0.8      (multiplier at the attention peak == factor2 == 0.2)
    release r = 1 - m,   lambda = mean(r)           ("how much of the SCR gradient the gate removes")
    R_face = mean(r | face box)   -> how much the person is RELEASED to change   (debias pressure)
    R_bg   = mean(r | background) -> how much the background is released to drift (fidelity cost)

Gates compared (all monotone in the SAME map, so all share one pixel RANKING -- the design question is
therefore only how each ALLOCATES its damping budget over that ranking):
    hard          1[minmax(a) >= thr]                             <- CURRENT (thr=0.15)
    hard_topp     1[a >= nucleus cutoff at mass p]                <- same bang-bang, budget set by MASS
    softmax       exp((minmax(a)-1)/tau), floor-corrected         <- METHOD 1 (peakier for tau<1)
    topp          clamp((a-a_p)/(a_max-a_p),0,1)^gamma            <- METHOD 2 (soft in nucleus, 0 outside)
    soft_nucleus  1 in the top-p_core mass, smooth ramp to 0 at p_edge  <- HYBRID (recommended)
    center / shuffle / uniform                                    <- controls

Fair-comparison protocol: gates are compared at MATCHED R_face (equal debias pressure on the person),
never at their raw parameters, because R_face is what the release is FOR. lambda then tells you the
weight_loss_scr compensation, c_wscr = <m>_baseline / <m>_gate.
"""
import argparse
import json
import math
import os

import torch

import scr_softmask_lib as L

BASE_THR = 0.15        # args.attn_gate_thr
BASE_F2 = 0.2          # args.factor2
DEPTH = 1.0 - BASE_F2  # 0.8
SHELL_W = 0.25         # soft_nucleus shell width in mass units (p_edge = p_core + SHELL_W)


def face_masks(boxes, det, H=64, W=64):
    n = boxes.shape[0]
    fm = torch.zeros(n, H, W)
    yy = torch.arange(H).view(H, 1).float()
    xx = torch.arange(W).view(1, W).float()
    for i in range(n):
        if not det[i]:
            continue
        x1, y1, x2, y2 = boxes[i].tolist()
        fm[i] = (((xx + .5) >= x1) & ((xx + .5) < x2) & ((yy + .5) >= y1) & ((yy + .5) < y2)).float()
    return fm


def metrics(s, fm, det, depth=DEPTH):
    """Per-sample gate metrics from the damping-strength field s."""
    m = (1.0 - depth * s).clamp(0, 1)
    n = m.shape[0]
    r = 1.0 - m
    lam = r.flatten(1).mean(1)
    face_n = fm.flatten(1).sum(1).clamp_min(1)
    bg_n = (1 - fm).flatten(1).sum(1).clamp_min(1)
    R_face = (r * fm).flatten(1).sum(1) / face_n
    R_bg = (r * (1 - fm)).flatten(1).sum(1) / bg_n
    sel = R_face / R_bg.clamp_min(1e-6)
    nanify = lambda v: torch.where(det, v, torch.full_like(v, float("nan")))  # noqa: E731
    return {"lambda": lam, "mean_m": m.flatten(1).mean(1), "budget": s.flatten(1).mean(1),
            "R_face": nanify(R_face), "R_bg": nanify(R_bg), "selectivity": nanify(sel), "m": m}


def summarize(mt, mean_m_base):
    o = {}
    for k in ["lambda", "mean_m", "budget", "R_face", "R_bg", "selectivity"]:
        v = mt[k]
        v = v[~torch.isnan(v)]
        o[k] = {"mean": float(v.mean()), "std": float(v.std()),
                "cv": float(v.std() / v.mean().abs().clamp_min(1e-9))}
    o["c_wscr"] = mean_m_base / max(o["mean_m"]["mean"], 1e-9)
    return o


def bisect_param(fn, obj, target, lo, hi, iters=45):
    """Global bisection: find param so that obj(fn(param)) == target. obj must be monotone in param.
    Returns (param, achieved, feasible)."""
    o_lo, o_hi = obj(fn(lo)), obj(fn(hi))
    inc = o_hi >= o_lo
    if not (min(o_lo, o_hi) - 1e-6 <= target <= max(o_lo, o_hi) + 1e-6):
        best = lo if abs(o_lo - target) < abs(o_hi - target) else hi
        return best, obj(fn(best)), False
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (obj(fn(mid)) < target) == inc:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    return p, obj(fn(p)), True


def auc_face(score, fm, det):
    aucs = []
    for i in range(score.shape[0]):
        if not det[i]:
            continue
        y = fm[i].flatten() > 0.5
        if y.all() or (~y).all():
            continue
        r = score[i].flatten().argsort().argsort().float() + 1
        np_, nn = int(y.sum()), int((~y).sum())
        aucs.append(float((r[y].sum() - np_ * (np_ + 1) / 2) / (np_ * nn)))
    return float(torch.tensor(aucs).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="/workspace/finetune-fair-diffusion/exp-1-debias-gender/"
                    "attmap_experiments/attmap_threshold_mask_out_50/maps/attmaps_and_images.pt")
    ap.add_argument("--sources", default="/workspace/finetune-fair-diffusion/exp-1-debias-gender/"
                    "attmap_experiments/attmap_gen_vs_score_out/maps/attmaps.pt")
    ap.add_argument("--out", default="./scr_softmask_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    d = torch.load(args.maps, map_location="cpu", weights_only=False)
    a = d["attmaps"].float()
    fb = torch.load(os.path.join(args.out, "face_boxes.pt"), map_location="cpu", weights_only=False)
    fm, det = face_masks(fb["boxes"], fb["det"]), fb["det"]
    N = a.shape[0]
    R = {"n": N, "n_face": int(det.sum()), "base_thr": BASE_THR, "base_factor2": BASE_F2,
         "depth": DEPTH, "shell_w": SHELL_W,
         "face_area_frac": float(fm.flatten(1).mean(1)[det].mean())}
    print(f"[data] {N} maps | {int(det.sum())} with a face | face box = "
          f"{R['face_area_frac']:.3f} of the 64x64 latent")

    # -- 0. how much localization signal does the map even carry? -----------------------------------
    g, _ = L.minmax_norm(a)
    prob = L.to_prob(a)
    R["map"] = {
        "auc_attn_vs_face": auc_face(a, fm, det),
        "auc_center_prior_vs_face": auc_face(L.s_center(a, .30), fm, det),
        "auc_shuffled_attn": auc_face(L.s_shuffle(a, 0), fm, det),
        "normalized_entropy": float((-(prob * (prob + 1e-12).log()).flatten(1).sum(1) / math.log(4096)).mean()),
        "peak_over_mean": float((a.flatten(1).amax(1) / a.flatten(1).mean(1)).mean()),
        "mean_minmax_g": float(g.mean()),
    }
    print(f"[map ] AUC(attn->face)={R['map']['auc_attn_vs_face']:.3f}  "
          f"center-prior={R['map']['auc_center_prior_vs_face']:.3f}  "
          f"shuffled={R['map']['auc_shuffled_attn']:.3f}  entropy={R['map']['normalized_entropy']:.4f}")

    # -- 1. baseline ---------------------------------------------------------------------------------
    mt_base = metrics(L.s_hard(a, BASE_THR), fm, det)
    MEAN_M_BASE = float(mt_base["mean_m"].mean())
    LAM_BASE = 1 - MEAN_M_BASE
    RFACE_BASE = float(mt_base["R_face"][det].mean())
    base = summarize(mt_base, MEAN_M_BASE)
    R["baseline"] = base
    # the mass the current gate actually releases (this is what it "really" is: a top-p gate with a
    # p that swings sample to sample)
    mass = (prob * L.s_hard(a, BASE_THR)).flatten(1).sum(1)
    R["baseline"]["released_attn_mass"] = {"mean": float(mass.mean()), "std": float(mass.std()),
                                           "min": float(mass.min()), "max": float(mass.max())}
    print(f"[base] hard@{BASE_THR}: lambda={LAM_BASE:.3f} (cv {base['lambda']['cv']:.3f})  "
          f"R_face={RFACE_BASE:.3f}  R_bg={base['R_bg']['mean']:.3f}  sel={base['selectivity']['mean']:.2f}")
    print(f"       -> it releases {mass.mean():.1%} of the gender-attention MASS, but that swings "
          f"{mass.min():.1%}..{mass.max():.1%} per sample, and covers "
          f"{float(mt_base['budget'].mean()):.1%} +- {float(mt_base['budget'].std()):.1%} of the image")

    # -- 2. feasibility ------------------------------------------------------------------------------
    R["feasibility"] = {
        "baseline_lambda": LAM_BASE, "baseline_R_face": RFACE_BASE,
        "softmax_max_lambda(tau->inf)": DEPTH * float(L.budget(L.s_softmax(a, tau=1e6)).mean()),
        "softmax_max_R_face(tau->inf)": float(metrics(L.s_softmax(a, tau=1e6), fm, det)["R_face"][det].mean()),
        "topp_max_lambda(p=1,gamma->0)": DEPTH * float(L.budget(L.s_topp(a, 1.0, 1e-3)).mean()),
        "note": ("lambda = depth*mean(s) <= mean(s) and the softmax family is bounded by mean(minmax g)"
                 f" = {R['map']['mean_minmax_g']:.3f}: it can NEVER reach the baseline's lambda or R_face"
                 " at peak depth 0.2."),
    }
    f = R["feasibility"]
    print(f"[feas] softmax ceiling: lambda<={f['softmax_max_lambda(tau->inf)']:.3f} "
          f"R_face<={f['softmax_max_R_face(tau->inf)']:.3f}  vs baseline {LAM_BASE:.3f}/{RFACE_BASE:.3f}"
          f"  -> INFEASIBLE at depth {DEPTH}")

    # -- 3. sweeps -----------------------------------------------------------------------------------
    FAM = {
        "hard_thr":     (lambda p: L.s_hard(a, p),                      [.05, .10, .15, .20, .25, .30, .35, .40, .50]),
        "hard_topp":    (lambda p: (L.s_topp(a, p) > 0).float(),        [.2, .3, .4, .5, .6, .7, .8, .9]),
        "softmax_tau":  (lambda p: L.s_softmax(a, tau=p),               [.05, .1, .15, .2, .25, .3, .4, .5, .75, 1., 2., 5.]),
        "topp_p":       (lambda p: L.s_topp(a, p=p, gamma=1.0),         [.2, .3, .4, .5, .6, .7, .8, .9]),
        "topp_gamma_p50": (lambda p: L.s_topp(a, p=.5, gamma=p),        [.25, .5, .75, 1., 1.5, 2., 3.]),
        "soft_nucleus": (lambda p: L.s_soft_nucleus(a, p, min(p + SHELL_W, 1.0)), [.1, .2, .3, .4, .5, .6, .7]),
        "mass_sigmoid_pmid": (lambda p: L.s_mass_sigmoid(a, p_mid=p, tau=0.12), [.2, .3, .4, .5, .6, .65, .7, .8]),
        "mass_sigmoid_tau":  (lambda p: L.s_mass_sigmoid(a, p_mid=.65, tau=p), [.02, .05, .08, .12, .20, .30, .50]),
    }
    sweeps = {}
    for name, (fn, params) in FAM.items():
        rows = []
        for p in params:
            mt = metrics(fn(p), fm, det)
            s = summarize(mt, MEAN_M_BASE)
            rows.append({"param": p, "lambda": s["lambda"]["mean"], "lambda_cv": s["lambda"]["cv"],
                         "R_face": s["R_face"]["mean"], "R_bg": s["R_bg"]["mean"],
                         "selectivity": s["selectivity"]["mean"], "c_wscr": s["c_wscr"]})
        sweeps[name] = rows
        print(f"\n[sweep] {name}")
        print(f"   {'param':>7} {'lambda':>7} {'lam_cv':>7} {'R_face':>7} {'R_bg':>7} {'sel':>7} {'c_wscr':>7}")
        for r in rows:
            print(f"   {r['param']:>7.3g} {r['lambda']:>7.3f} {r['lambda_cv']:>7.3f} {r['R_face']:>7.3f} "
                  f"{r['R_bg']:>7.3f} {r['selectivity']:>7.2f} {r['c_wscr']:>7.3f}")
    R["sweeps"] = sweeps

    # -- 4. MATCHED-R_face comparison (equal debias pressure on the person) --------------------------
    print(f"\n[match] calibrating every gate to the baseline's face release R_face = {RFACE_BASE:.3f}")
    obj_rface = lambda s: float(metrics(s, fm, det)["R_face"][det].mean())  # noqa: E731
    cand = {
        "hard (current param.)":  (lambda p: L.s_hard(a, p),                      0.60, 0.02),
        "hard_topp (by mass)":    (lambda p: (L.s_topp(a, p) > 0).float(),        0.05, 0.99),
        "softmax tau":            (lambda p: L.s_softmax(a, tau=p),               0.02, 200.),
        "topp gamma (p=0.5)":     (lambda p: L.s_topp(a, p=.5, gamma=p),          8.0, 0.02),
        "soft_nucleus p_core":    (lambda p: L.s_soft_nucleus(a, p, min(p + SHELL_W, 1.0)), 0.01, 0.75),
    }
    matched = {}
    for name, (fn, lo, hi) in cand.items():
        p, got, ok = bisect_param(fn, obj_rface, RFACE_BASE, lo, hi)
        mt = metrics(fn(p), fm, det)
        s = summarize(mt, MEAN_M_BASE)
        s.update({"param": p, "R_face_achieved": got, "feasible": ok})
        matched[name] = s
    R["matched_R_face"] = {"target_R_face": RFACE_BASE, "gates": matched}
    print(f"   {'gate':<24} {'param':>7} {'R_face':>7} {'R_bg':>7} {'sel':>6} {'lambda':>7} "
          f"{'lam_cv':>7} {'c_wscr':>7}  feasible")
    for k, v in matched.items():
        print(f"   {k:<24} {v['param']:>7.3f} {v['R_face_achieved']:>7.3f} {v['R_bg']['mean']:>7.3f} "
              f"{v['selectivity']['mean']:>6.2f} {v['lambda']['mean']:>7.3f} {v['lambda']['cv']:>7.3f} "
              f"{v['c_wscr']:>7.3f}  {v['feasible']}")

    # -- 5. operating points the user would actually run ---------------------------------------------
    ops, m_ops = {}, {}
    OP = {
        "hard@0.15 (CURRENT)":        L.s_hard(a, BASE_THR),
        "hard_topp p=0.75":           (L.s_topp(a, .75) > 0).float(),
        "softmax tau=0.25 (asked)":   L.s_softmax(a, tau=.25),
        "softmax tau=0.5":            L.s_softmax(a, tau=.5),
        "topp p=0.5 g=1 (asked)":     L.s_topp(a, p=.5, gamma=1.),
        "topp p=0.5 g=0.5":           L.s_topp(a, p=.5, gamma=.5),
        "mass_sigmoid p=.65 t=.12 (REC)": L.s_mass_sigmoid(a, .65, .12),
        "mass_sigmoid p=.50 t=.12": L.s_mass_sigmoid(a, .50, .12),
        "soft_nucleus .30/.55 (REC)": L.s_soft_nucleus(a, .30, .55),
        "soft_nucleus .45/.70 (REC+)": L.s_soft_nucleus(a, .45, .70),
        "center prior (control)":     L.s_center(a, .30),
        "shuffled attn (control)":    L.s_soft_nucleus(L.s_shuffle(a), .30, .55),
        "uniform (control)":          L.s_uniform(a, .5),
    }
    print(f"\n[ops ] {'gate':<29} {'lambda':>7} {'lam_cv':>7} {'R_face':>7} {'R_bg':>7} {'sel':>6} {'c_wscr':>7}")
    for k, s in OP.items():
        mt = metrics(s, fm, det)
        ops[k] = summarize(mt, MEAN_M_BASE)
        m_ops[k] = mt["m"]
        v = ops[k]
        print(f"       {k:<29} {v['lambda']['mean']:>7.3f} {v['lambda']['cv']:>7.3f} "
              f"{v['R_face']['mean']:>7.3f} {v['R_bg']['mean']:>7.3f} "
              f"{v['selectivity']['mean']:>6.2f} {v['c_wscr']:>7.3f}")
    R["operating_points"] = ops

    # -- 5b. THE DECISIVE COMPARISON --------------------------------------------------------------
    # weight_loss_scr is a free GLOBAL knob, so the only w-invariant quality of a gate is the
    # PRESERVATION CONTRAST  P = m_bg / m_face  (how much harder the gate holds the background than
    # it holds the person). Comparing gates therefore means: fix the global SCR strength
    # <m> = 0.463 (the current gate's -- so weight_loss_scr NEVER needs re-tuning), then ask which
    # gate buys the most contrast at that price. Depth d = 1 - factor2 is swept too: it is the axis
    # the user pinned at 0.2, and it turns out to be the strongest lever.
    print(f"\n[FOM ] at FIXED global strength <m> = {MEAN_M_BASE:.3f} (no weight_loss_scr re-tune), "
          f"maximize preservation contrast P = m_bg / m_face")
    print(f"       {'gate':<20} {'factor2':>7} {'region':>7} {'m_face':>7} {'m_bg':>6} {'P':>6} "
          f"{'R_face':>7} {'R_bg':>6} {'lam_cv':>7}")
    fom = {}
    FAMS = {
        "hard (value thr)": (lambda p, x=None: L.s_hard(a, p), 0.60, 0.02),
        "hard_topp (mass)": (lambda p, x=None: (L.s_topp(a, p) > 0).float(), 0.02, 0.99),
        "soft_nucleus":     (lambda p, x=None: L.s_soft_nucleus(a, p, min(p + SHELL_W, 1.0)), 0.01, 0.75),
        "mass_sigmoid t=.12": (lambda p, x=None: L.s_mass_sigmoid(a, p_mid=p, tau=.12), 0.02, 0.99),
        "topp (gamma,p=.5)": (lambda p, x=None: L.s_topp(a, p=.5, gamma=p), 8.0, 0.02),
        "softmax (tau)":    (lambda p, x=None: L.s_softmax(a, tau=p), 0.02, 500.),
    }
    for gname, (gfn, lo, hi) in FAMS.items():
        for f2 in [0.2, 0.1, 0.0]:
            dep = 1.0 - f2
            obj = lambda s, dd=dep: float((1 - dd * s).clamp(0, 1).flatten(1).mean(1).mean())  # noqa: E731
            p, got, ok = bisect_param(gfn, obj, MEAN_M_BASE, lo, hi)
            mt = metrics(gfn(p), fm, det, depth=dep)
            rf, rb = float(mt["R_face"][det].mean()), float(mt["R_bg"][det].mean())
            m_f, m_b = 1 - rf, 1 - rb
            P = m_b / max(m_f, 1e-6)
            fom[f"{gname} | factor2={f2}"] = {
                "region_param": p, "feasible": ok, "mean_m": got, "m_face": m_f, "m_bg": m_b,
                "P": P, "R_face": rf, "R_bg": rb, "lambda_cv": float(mt["lambda"].std() / mt["lambda"].mean()),
            }
            flag = "" if ok else "  <- INFEASIBLE: cannot reach this <m> at any parameter"
            print(f"       {gname:<20} {f2:>7.1f} {p:>7.3f} {m_f:>7.3f} {m_b:>6.3f} {P:>6.2f} "
                  f"{rf:>7.3f} {rb:>6.3f} {float(mt['lambda'].std()/mt['lambda'].mean()):>7.3f}{flag}")
    # oracle: what a PERFECT face mask would score (the ceiling set by the attention map's quality)
    for f2 in [0.2, 0.0]:
        dep = 1 - f2
        s_or = fm.clone()
        mt = metrics(s_or, fm, det, depth=dep)
        rf, rb = float(mt["R_face"][det].mean()), float(mt["R_bg"][det].mean())
        fom[f"ORACLE face box | factor2={f2}"] = {"m_face": 1 - rf, "m_bg": 1 - rb,
                                                  "P": (1 - rb) / max(1 - rf, 1e-6),
                                                  "mean_m": float(mt["mean_m"].mean())}
        print(f"       {'ORACLE face box':<20} {f2:>7.1f} {'-':>7} {1-rf:>7.3f} {1-rb:>6.3f} "
              f"{(1-rb)/max(1-rf,1e-6):>6.2f} {rf:>7.3f} {rb:>6.3f} {'-':>7}   (<m>={float(mt['mean_m'].mean()):.3f})")
    R["fixed_budget_contrast"] = fom

    # -- 6. STABILITY: does the gate survive a change in the attention source? -----------------------
    # 10 images scored from three different cross-attn block sets (SCORE ALL / MID / RES16). Each gate
    # is first calibrated ON THE SAME SOURCE to the SAME lambda, so |dm| is compared at equal budget;
    # |dm|/lambda is the scale-free version.
    stab = {}
    try:
        S = torch.load(args.sources, map_location="cpu", weights_only=False)["SCORE"]
        ref = S["ALL"].float()
        LAM_T = 0.32
        gates = {
            "hard (thr)":       (lambda x, p: L.s_hard(x, p),                       0.60, 0.02),
            "hard_topp (mass)": (lambda x, p: (L.s_topp(x, p) > 0).float(),         0.02, 0.99),
            "softmax (tau)":    (lambda x, p: L.s_softmax(x, tau=p),                0.02, 200.),
            "topp (gamma,p=.5)": (lambda x, p: L.s_topp(x, p=.5, gamma=p),          8.0, 0.02),
            "soft_nucleus":     (lambda x, p: L.s_soft_nucleus(x, p, min(p + SHELL_W, 1.)), 0.01, 0.75),
            "mass_sigmoid t=.12": (lambda x, p: L.s_mass_sigmoid(x, p_mid=p, tau=.12), 0.02, 0.99),
            "mass_sigmoid t=.25": (lambda x, p: L.s_mass_sigmoid(x, p_mid=p, tau=.25), 0.02, 0.99),
        }
        obj_lam = lambda s: float((DEPTH * s).flatten(1).mean(1).mean())  # noqa: E731
        print(f"\n[stab] attention-source swap, every gate calibrated to lambda={LAM_T} on SCORE-ALL")
        print(f"       {'gate':<19} {'param':>7} {'lambda':>7} {'mean|dm|':>9} {'|dm|/lam':>9} {'fuzzyIoU':>9}")
        for gname, (gfn, lo, hi) in gates.items():
            p, got, ok = bisect_param(lambda q: gfn(ref, q), obj_lam, LAM_T, lo, hi)
            dm, fio = [], []
            for k1, k2 in [("ALL", "MID"), ("ALL", "RES16"), ("MID", "RES16")]:
                m1 = (1 - DEPTH * gfn(S[k1].float(), p)).clamp(0, 1)
                m2 = (1 - DEPTH * gfn(S[k2].float(), p)).clamp(0, 1)
                dm.append(float((m1 - m2).abs().mean()))
                r1, r2 = 1 - m1, 1 - m2
                fio.append(float((torch.minimum(r1, r2).flatten(1).sum(1) /
                                  torch.maximum(r1, r2).flatten(1).sum(1).clamp_min(1e-9)).mean()))
            adm = sum(dm) / len(dm)
            stab[gname] = {"param": p, "lambda": got, "feasible": ok, "mean_abs_delta_m": adm,
                           "rel_delta": adm / max(got, 1e-9), "fuzzy_iou": sum(fio) / len(fio)}
            v = stab[gname]
            flag = "" if ok else "  (lambda INFEASIBLE, at ceiling)"
            print(f"       {gname:<19} {p:>7.3f} {got:>7.3f} {adm:>9.4f} {v['rel_delta']:>9.3f} "
                  f"{v['fuzzy_iou']:>9.3f}{flag}")
    except Exception as e:  # noqa: BLE001
        print(f"[stab] SKIPPED ({type(e).__name__}: {e})")
    R["stability"] = stab

    # -- 7. edge cases -------------------------------------------------------------------------------
    edge = {}
    zero = torch.zeros(2, 64, 64)          # the no-face sample (training zeroes common_attn)
    for nm, s in [("hard", L.s_hard(zero, BASE_THR)), ("softmax", L.s_softmax(zero, tau=.25)),
                  ("softmax_NO_floorfix", L.s_softmax(zero, tau=.25, floor_correct=False)),
                  ("topp", L.s_topp(zero, .5)), ("soft_nucleus", L.s_soft_nucleus(zero, .3, .55)),
                  ("mass_sigmoid", L.s_mass_sigmoid(zero, .65, .12))]:
        edge[f"zero_map::{nm}"] = {"max_s": float(s.max()), "nan": bool(torch.isnan(s).any()),
                                   "ok": bool(s.max() == 0 and not torch.isnan(s).any())}
    a16 = a.half()
    for nm, f32, f16 in [("softmax", L.s_softmax(a, tau=.25), L.s_softmax(a16, tau=.25)),
                         ("topp", L.s_topp(a, .5), L.s_topp(a16, .5)),
                         ("soft_nucleus", L.s_soft_nucleus(a, .3, .55), L.s_soft_nucleus(a16, .3, .55)),
                         ("mass_sigmoid", L.s_mass_sigmoid(a, .65, .12), L.s_mass_sigmoid(a16, .65, .12))]:
        edge[f"fp16::{nm}"] = {"nan": bool(torch.isnan(f16).any()),
                               "max_dev": float((f16 - f32).abs().max())}
    b_tau = [float(L.budget(L.s_softmax(a, tau=t)).mean()) for t in [.05, .1, .25, .5, 1, 2, 5, 20]]
    b_gam = [float(L.budget(L.s_topp(a, .5, gamma=gm)).mean()) for gm in [.1, .25, .5, 1, 2, 4, 8]]
    b_pc = [float(L.budget(L.s_soft_nucleus(a, pc, min(pc + SHELL_W, 1.))).mean()) for pc in [.05, .1, .2, .3, .4, .5, .6]]
    edge["monotone_budget_in_tau"] = all(x <= y + 1e-9 for x, y in zip(b_tau, b_tau[1:]))
    edge["monotone_budget_in_gamma_desc"] = all(x >= y - 1e-9 for x, y in zip(b_gam, b_gam[1:]))
    edge["monotone_budget_in_p_core"] = all(x <= y + 1e-9 for x, y in zip(b_pc, b_pc[1:]))
    R["edge_cases"] = edge
    bad = [k for k, v in edge.items() if isinstance(v, dict) and v.get("ok") is False]
    print(f"\n[edge] all guards pass: {not bad}" + (f"  FAILURES: {bad}" if bad else "")
          + f" | fp16 max deviation {max(edge[k]['max_dev'] for k in edge if k.startswith('fp16')):.2e}"
          + f" | monotone(tau,gamma,p_core)="
          f"{edge['monotone_budget_in_tau'], edge['monotone_budget_in_gamma_desc'], edge['monotone_budget_in_p_core']}")
    print(f"       no-face guard: softmax WITHOUT the floor fix damps the whole image "
          f"(max_s={edge['zero_map::softmax_NO_floorfix']['max_s']:.3f}) -> must keep floor_correct=True")

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(R, fh, indent=2)
    torch.save({"m_ops": m_ops, "attn": a, "images": d["images"], "face_mask": fm, "det": det,
                "mean_m_base": MEAN_M_BASE}, os.path.join(args.out, "masks.pt"))
    print(f"\n[done] -> {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
