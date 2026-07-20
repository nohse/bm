"""Create 1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector_softgate.py from the nodetector
file, adding --scr_mask_mode {hard,softmax,topp,mass_sigmoid} for the SCR flip gradient gate.

Defaults reproduce the current gate BIT-IDENTICALLY (mode=hard):
    hard:  s = 1[minmax(common_attn) >= attn_gate_thr]           -> m = 1 - (1-factor2)*s = {0.2, 1.0}
The new modes replace only the DAMPING-STRENGTH field s; the release set, the no-face zeroing, the
hook on zt_ft, the SRR input mask and every other loss stay exactly as they are.
"""
import os
import re

SRC = "1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector.py"
DST = "1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector_softgate.py"

src = open(SRC).read()

# ---------------------------------------------------------------------------------------------- 1
# new argparse flags, right after --factor2
OLD_ARGS = """    parser.add_argument('--factor1', help="train, val, test batch size", type=float, default=0.2)
    parser.add_argument('--factor2', help="train, val, test batch size", type=float, default=0.2)
"""
NEW_ARGS = OLD_ARGS + '''
    # ---- SCR flip gradient-gate shape -----------------------------------------------------------
    # The gate damps the SCR (image-preservation) gradient by factor2 inside the person region for
    # flip/uncertain samples. `hard` (the default) is the CURRENT behaviour, bit-identical.
    #
    # MEASURED on 50 real scoring-time woman/man attn maps (scr_softmask_experiment.py):
    #   the `hard` gate thresholds an attention VALUE, and because the map is nearly uniform
    #   (normalized entropy 0.996) the region it selects swings from 25% to 90% of the image and
    #   releases 39%..92% of the gender-attention mass, sample to sample (CV of the released
    #   gradient energy = 0.19).
    #   `mass_sigmoid` cuts on cumulative attention MASS instead of value, so the released budget is
    #   the same for every sample (CV 0.193 -> 0.031, a 6.2x reduction) at the same average SCR
    #   strength (<m> 0.463 -> 0.466 at the default p_mid=0.80, so weight_loss_scr does NOT need
    #   re-tuning), and the gate is 24-33% more stable when the attention source is perturbed.
    parser.add_argument(
        '--scr_mask_mode',
        type=str, default="hard", choices=["hard", "softmax", "topp", "mass_sigmoid"],
        help="shape of the SCR flip gradient gate. 'hard' = CURRENT: s = 1[minmax(attn) >= "
             "--attn_gate_thr] (bit-identical default). 'softmax' = peak-normalized, floor-corrected "
             "exp((minmax(attn)-1)/tau) (temperature in the VALUE domain; measured to be far too weak "
             "-- cannot reach the hard gate's strength at ANY tau -- kept only for the ablation). "
             "'topp' = soft ramp inside the top --scr_mask_pmid attention-MASS nucleus, exactly 0 "
             "outside. 'mass_sigmoid' = RECOMMENDED: sigmoid((p_mid - cumulative_mass)/tau), i.e. a "
             "top-p cut with a temperature-controlled soft edge. In every mode the multiplier at the "
             "attention peak is exactly --factor2, so the DAMPING MAGNITUDE is unchanged.",
    )
    parser.add_argument(
        '--scr_mask_pmid',
        type=float, default=0.80,
        help="cumulative attention MASS released by the gate (modes topp / mass_sigmoid). MEASURED: "
             "p_mid=0.80 reproduces the current hard gate's AVERAGE SCR strength almost exactly "
             "(<m> 0.466 vs 0.463, i.e. weight_loss_scr needs NO re-tune) while cutting the "
             "per-sample swing 6x (CV 0.19 -> 0.03) -- the clean A/B where only the mask SHAPE changes. "
             "p_mid=0.65 releases less and leaks much less onto the background, but the SCR then acts "
             "~1.2x stronger, so scale weight_loss_scr by ~0.82 to keep the fidelity/debias balance.",
    )
    parser.add_argument(
        '--scr_mask_tau',
        type=float, default=0.12,
        help="softness of the gate edge. mass_sigmoid: temperature in cumulative-mass units "
             "(tau->0 = hard nucleus; 0.12 is a good compromise; larger = softer/more stable but less "
             "localized). softmax: temperature on the min-max attention value. topp: unused.",
    )
'''
assert OLD_ARGS in src
src = src.replace(OLD_ARGS, NEW_ARGS, 1)

# ---------------------------------------------------------------------------------------------- 2
# always tag the run/folder name (both modes), per the project convention
OLD_TAG = '        f"_wSCR-{args.weight_loss_scr}-{args.factor1}-{args.factor2}"\n'
NEW_TAG = (OLD_TAG +
           '        f"_scrMask-{args.scr_mask_mode}"\n'
           '        f"{(\'-p\'+format(args.scr_mask_pmid, \'g\')) if args.scr_mask_mode in (\'topp\',\'mass_sigmoid\') else \'\'}"\n'
           '        f"{(\'-t\'+format(args.scr_mask_tau, \'g\')) if args.scr_mask_mode in (\'softmax\',\'mass_sigmoid\') else \'\'}"\n')
assert OLD_TAG in src
src = src.replace(OLD_TAG, NEW_TAG, 1)

# ---------------------------------------------------------------------------------------------- 3
# the gate builder, inserted just before make_grad_hook
OLD_HOOK = "def make_grad_hook(coef):"
NEW_HELPER = '''def scr_damping_strength(common_attn, mode, thr, p_mid, tau):
    """Damping-strength field s in [0,1] for the SCR flip gradient gate.  m = 1 - (1-factor2)*s.

    common_attn: [n,H,W] sum-to-1 (detached) gender localization map from the residual scorer.
    Returns s [n,H,W], float32, detached. s == 1 at the attention peak in EVERY mode, so the
    multiplier there is exactly factor2 (the damping MAGNITUDE is mode-independent).

    NO-FACE SAMPLES: the caller zeroes common_attn for them. Every branch below must then return
    s == 0 (no damping) and never NaN -- the guards are load-bearing, not defensive noise:
      hard:         min-max of an all-zero map -> 0 -> below thr -> 0.            (as today)
      softmax:      exp((0-1)/tau) = exp(-1/tau) > 0 would damp the WHOLE image; the floor
                    correction maps g=0 -> s=0 exactly, and `degen` zeroes the constant map.
      topp/mass_*:  total mass 0 -> cumulative rank forced to 1 -> outside every nucleus -> 0.
    """
    a = common_attn.detach().float()
    n = a.shape[0]
    flat = a.reshape(n, -1)

    if mode in ("hard", "softmax"):
        lo = flat.amin(dim=1, keepdim=True)
        hi = flat.amax(dim=1, keepdim=True)
        degen = (hi - lo) <= 1e-12
        g = ((flat - lo) / (hi - lo + 1e-8)).clamp(0, 1)   # +1e-8 == the original expression, verbatim
        g = torch.where(degen, torch.zeros_like(g), g)
        if mode == "hard":
            s = (g >= thr).float()
        else:
            t = max(tau, 1e-6)
            fl = math.exp(-1.0 / t)                       # background floor: MUST be removed
            s = ((torch.exp((g - 1.0) / t) - fl) / (1.0 - fl)).clamp(0, 1)
        s = torch.where(degen, torch.zeros_like(s), s)
        return s.reshape_as(a)

    prob = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)      # all-zero map -> stays 0
    if mode == "topp":
        srt, _ = prob.sort(dim=1, descending=True)
        k = (srt.cumsum(dim=1) < p_mid).sum(dim=1, keepdim=True).clamp(max=prob.shape[1] - 1)
        a_p = srt.gather(1, k)                                        # nucleus cutoff value
        a_max = srt[:, :1]
        s = ((prob - a_p) / (a_max - a_p).clamp_min(1e-12)).clamp(0, 1)
        return s.reshape_as(a)

    if mode == "mass_sigmoid":
        order = prob.argsort(dim=1, descending=True)
        csum = prob.gather(1, order).cumsum(dim=1)
        C = torch.empty_like(csum).scatter_(1, order, csum)           # cumulative mass rank per pixel
        C = torch.where(prob.sum(dim=1, keepdim=True) <= 1e-12, torch.ones_like(C), C)
        t = max(tau, 1e-6)
        s0 = torch.sigmoid(torch.tensor(p_mid / t, device=a.device))          # value at C=0 (peak)
        s1 = torch.sigmoid(torch.tensor((p_mid - 1.0) / t, device=a.device))  # value at C=1
        s = torch.sigmoid((p_mid - C) / t)
        s = ((s - s1) / (s0 - s1).clamp_min(1e-12)).clamp(0, 1)        # C=0 -> 1, C=1 -> 0 exactly
        return s.reshape_as(a)

    raise ValueError(f"unknown scr_mask_mode: {mode}")


''' + OLD_HOOK
assert OLD_HOOK in src
src = src.replace(OLD_HOOK, NEW_HELPER, 1)

# ---------------------------------------------------------------------------------------------- 4
# the gate itself
OLD_GATE = """                cmin = common_attn_ij.amin(dim=(1, 2), keepdim=True)
                cmax = common_attn_ij.amax(dim=(1, 2), keepdim=True)
                attn_gate = ((common_attn_ij - cmin) / (cmax - cmin + 1e-8)).clamp(0, 1)          # [chunk,64,64] min-max
                release_ij = (targets_ij != preds_gender_ori_ij) | (targets_ij == -1)               # debias-aligned: flip OR uncertain(-1)
                scr_grad_mask = torch.ones_like(attn_gate)
                scr_grad_mask = torch.where((attn_gate >= args.attn_gate_thr) & release_ij[:, None, None],
                                            torch.full_like(scr_grad_mask, args.factor2), scr_grad_mask)
                scr_grad_mask = scr_grad_mask[:, None, :, :].to(z0_ij.dtype)                        # [chunk,1,64,64]
"""
NEW_GATE = """                # Damping-strength field s in [0,1] (mode-selectable; 'hard' == the original gate,
                # bit-identical: s = 1[minmax(attn) >= thr] -> m = {factor2, 1}). s == 1 at the
                # attention peak in every mode, so the damping MAGNITUDE stays factor2.
                attn_gate = scr_damping_strength(
                    common_attn_ij, args.scr_mask_mode, args.attn_gate_thr,
                    args.scr_mask_pmid, args.scr_mask_tau,
                )                                                                                  # [chunk,64,64]
                release_ij = (targets_ij != preds_gender_ori_ij) | (targets_ij == -1)               # debias-aligned: flip OR uncertain(-1)
                # m = 1 - (1 - factor2) * s, applied only to released samples. For mode 'hard' this is
                # EXACTLY the previous torch.where(gate >= thr & release, factor2, 1).
                s_ij = attn_gate * release_ij[:, None, None].to(attn_gate.dtype)
                if args.scr_mask_mode == "hard":
                    # keep the ORIGINAL expression verbatim so the default run is bit-identical
                    # (1 - (1-0.2)*1 = 0.19999998807907104 != float32(0.2); a 1.5e-8 drift otherwise)
                    scr_grad_mask = torch.where(s_ij > 0.5, torch.full_like(s_ij, args.factor2),
                                                torch.ones_like(s_ij))
                else:
                    scr_grad_mask = 1.0 - (1.0 - args.factor2) * s_ij
                scr_grad_mask = scr_grad_mask[:, None, :, :].to(z0_ij.dtype)                        # [chunk,1,64,64]
"""
assert OLD_GATE in src
src = src.replace(OLD_GATE, NEW_GATE, 1)

# ---------------------------------------------------------------------------------------------- 5
# the gradgate visualization: the middle panel is now the strength field s, not a min-max gate
src = src.replace(
    '        hard = (gate >= thr).float()                                        # [n,H,W] min-max hard mask',
    '        hard = (gate > 0).float()                                           # [n,H,W] support of the mask',
    1)
src = src.replace(
    '        labels = ["generated", "common-attn", "min-max gate", f"hard >= {thr:g}", f"applied x{factor2:g}"]',
    '        labels = ["generated", "common-attn", "mask s (strength)", "mask support", f"applied damp (min x{factor2:g})"]',
    1)
# `applied` was "damped iff mask < 1"; keep it, it is still exactly the damped region
assert "import math" in src, "math must already be imported (used by the softmax floor correction)"


# ---------------------------------------------------------------------------------------------- 6
# GRADIENT-SIDE LOGGING. The gate is a backward hook on zt_ft: it changes the GRADIENT, never the
# forward. loss_SCR is therefore IDENTICAL for every --scr_mask_mode, so no existing wandb panel can
# tell the arms apart. Log the two quantities that actually differ.
OLD_INIT = """            loss_SCR_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)"""
NEW_INIT = OLD_INIT + """
            # gate diagnostics (the gate is gradient-only, so loss_SCR cannot distinguish the modes):
            #   scr_mask_i    = mean SCR gradient multiplier <m> -> the effective SCR strength this step
            #   scr_release_i = fraction of samples the gate fires on (flip/uncertain)
            scr_mask_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            scr_release_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)"""
assert OLD_INIT in src
src = src.replace(OLD_INIT, NEW_INIT, 1)

OLD_STORE = """                with torch.no_grad():
                    loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)"""
NEW_STORE = """                with torch.no_grad():
                    scr_mask_i[idxs_ij] = scr_grad_mask.mean(dim=(1, 2, 3)).to(scr_mask_i.dtype)
                    scr_release_i[idxs_ij] = release_ij.to(scr_release_i.dtype)
                    loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)"""
assert OLD_STORE in src
src = src.replace(OLD_STORE, NEW_STORE, 1)

OLD_GATHER = """            loss_SCR_all = customized_all_gather(loss_SCR_i, accelerator)"""
NEW_GATHER = OLD_GATHER + """
            scr_mask_all = customized_all_gather(scr_mask_i, accelerator)
            scr_release_all = customized_all_gather(scr_release_i, accelerator)"""
assert OLD_GATHER in src
src = src.replace(OLD_GATHER, NEW_GATHER, 1)

OLD_APPEND = """                logs_i["loss_SCR"].append(loss_SCR_all)"""
NEW_APPEND = OLD_APPEND + """
                logs_i["scr_grad_mask_mean"].append(scr_mask_all)
                logs_i["scr_release_frac"].append(scr_release_all)"""
assert OLD_APPEND in src
src = src.replace(OLD_APPEND, NEW_APPEND, 1)

OLD_KEYS = """                    "loss_SCR": [],"""
NEW_KEYS = OLD_KEYS + """
                    "scr_grad_mask_mean": [],
                    "scr_release_frac": [],"""
assert OLD_KEYS in src
src = src.replace(OLD_KEYS, NEW_KEYS, 1)

OLD_PROC = """                for key in ["loss_fair", "loss_SRR", "loss_SCR", "loss"]:"""
NEW_PROC = """                for key in ["loss_fair", "loss_SRR", "loss_SCR", "loss",
                            "scr_grad_mask_mean", "scr_release_frac"]:"""
assert OLD_PROC in src
src = src.replace(OLD_PROC, NEW_PROC, 1)

open(DST, "w").write(src)
print(f"wrote {DST}  ({len(src.splitlines())} lines, +{len(src.splitlines()) - len(open(SRC).read().splitlines())})")
