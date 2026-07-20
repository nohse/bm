"""Soft SCR gradient-gate masks (drop-in replacements for the hard min-max >= 0.15 gate).

The SCR gate in 1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector.py multiplies the SCR
(image-preservation) gradient on zt_ft, per latent pixel, by

    m = 1 - d * s,      d = 1 - factor2  (default 0.8, i.e. the damped floor is factor2 = 0.2)

where s in [0,1] is the DAMPING STRENGTH field ("how much this pixel is released to change").
Today s is binary: s = 1[minmax(common_attn) >= 0.15]. This module provides continuous s fields.

Everything here is:
  * scale-invariant in the input map (works on the sum-to-1 common_attn or on any positive rescale),
  * NaN-safe on the degenerate all-zero map (the no-face case: common_attn is zeroed by
    face_indicators at line ~3102 -> must produce s == 0, i.e. NO damping),
  * fully vectorized over the batch, detached (the mask is a constant w.r.t. autograd),
  * exactly reproducible for the same input.

Definitions used by every metric downstream:
    release r = 1 - m = d * s        (fraction of the SCR gradient removed at that pixel)
    lambda   = mean_pixels(1 - m)    (RELEASED GRADIENT ENERGY; the "scale" of the gate)
    <m>      = mean_pixels(m)        (mean multiplier; lambda = 1 - <m>)
"""
import math

import torch

# A map is degenerate (-> no damping at all) when it is constant, in particular all-zero.
DEGEN_EPS = 1e-12


# --------------------------------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------------------------------
def minmax_norm(a, robust_q=0.0):
    """Per-image min-max normalize to [0,1]. Returns (g, degenerate_mask).

    a: [N,H,W] non-negative attention map, any positive scale (sum-to-1 or not -- min-max is
       invariant to a positive rescale, so this matches the training gate exactly).
    robust_q: if > 0, clip at the (q, 1-q) quantiles before normalizing (guards against a single hot
       pixel setting the max). robust_q=0 reproduces the training code's amin/amax exactly.
    """
    a = a.float()
    n = a.shape[0]
    flat = a.reshape(n, -1)
    if robust_q > 0:
        lo = torch.quantile(flat, robust_q, dim=1, keepdim=True)
        hi = torch.quantile(flat, 1.0 - robust_q, dim=1, keepdim=True)
    else:
        lo = flat.amin(dim=1, keepdim=True)
        hi = flat.amax(dim=1, keepdim=True)
    degen = (hi - lo) <= DEGEN_EPS                              # [N,1]
    g = ((flat - lo) / (hi - lo).clamp_min(DEGEN_EPS)).clamp(0, 1)
    g = torch.where(degen, torch.zeros_like(g), g)
    return g.reshape_as(a), degen.reshape(n)


def to_prob(a):
    """Per-image sum-to-1 (the common_attn convention). All-zero map -> all-zero (not NaN)."""
    a = a.float()
    n = a.shape[0]
    flat = a.reshape(n, -1)
    s = flat.sum(dim=1, keepdim=True)
    return (flat / s.clamp_min(DEGEN_EPS)).reshape_as(a)


# --------------------------------------------------------------------------------------------------
# the damping-strength fields s in [0,1]
# --------------------------------------------------------------------------------------------------
def s_hard(a, thr=0.15, robust_q=0.0):
    """BASELINE (current training code): binary s = 1[minmax(a) >= thr]."""
    g, degen = minmax_norm(a, robust_q)
    s = (g >= thr).float()
    s[degen] = 0.0                                              # degenerate map -> no damping
    return s


def s_softmax(a, tau=0.25, floor_correct=True, robust_q=0.0):
    """METHOD 1 -- softmax with temperature tau, peak-normalized.

    A softmax over the H*W pixels, softmax(g/tau), sums to 1 and so has values ~1/4096: it is NOT
    usable as a multiplicative mask. The only meaningful way to turn it into a mask is to normalize
    by its maximum, and because g is min-max normalized (max g == 1) that is EXACTLY

        s_i = softmax(g/tau)_i / max_j softmax(g/tau)_j = exp((g_i - 1) / tau)

    i.e. an exponential sharpening with s == 1 at the attention peak and tau < 1 making it peakier.
    (The identity is why no explicit softmax is computed here: it would cancel.)

    floor_correct: the raw form has a nonzero background floor exp(-1/tau) (e.g. 0.135 at tau=0.5),
        which would damp the ENTIRE image, including no-face samples. The corrected form
            s = (exp((g-1)/tau) - exp(-1/tau)) / (1 - exp(-1/tau))
        maps g=0 -> s=0 and g=1 -> s=1 exactly. Keep it on.

    Monotone limits (used by the budget calibrator): tau -> 0+  =>  s -> 0 except at the peak;
    tau -> inf  =>  s -> g. Hence mean(s) is increasing in tau and BOUNDED ABOVE BY mean(g)
    (this map: mean(g) ~= 0.257), which is what makes matching the hard gate's budget infeasible.
    """
    g, degen = minmax_norm(a, robust_q)
    s = torch.exp((g - 1.0) / max(tau, 1e-6))                   # (0,1], == 1 at the peak
    if floor_correct:
        fl = math.exp(-1.0 / max(tau, 1e-6))                    # underflows to 0.0 for tau < ~0.02
        s = (s - fl) / (1.0 - fl)
    s = s.clamp(0, 1)
    s[degen] = 0.0
    return s


def s_topp(a, p=0.5, gamma=1.0):
    """METHOD 2 -- cumulative-probability (nucleus / top-p) soft mask.

    Take the smallest set of pixels carrying a fraction p of the TOTAL attention mass (the map is
    sum-to-1, so this is a nucleus / top-p set). Inside it the mask is SOFT, outside it is exactly 0:

        a_p   = attention value of the last pixel included in the nucleus
        s     = clamp((a - a_p) / (a_max - a_p), 0, 1) ** gamma

    This is a min-max normalization whose LOWER anchor is the top-p cutoff instead of the global min.
    It is continuous at the nucleus boundary (s -> 0 there, no gradient-scale discontinuity), and it
    is defined by attention MASS, not by an attention VALUE -- so, unlike the hard gate, it adapts to
    each sample's peakiness instead of letting the sample's peakiness decide the masked area.

    gamma > 1 -> more peaked inside the nucleus (smaller budget); gamma < 1 -> flatter/fuller nucleus
    (larger budget, -> the hard nucleus as gamma -> 0). mean(s) is DECREASING in gamma, increasing in p.

    All-zero (no-face) map -> prob == 0 -> a_max == a_p == 0 -> s == 0. No NaN.
    """
    prob = to_prob(a)
    n = prob.shape[0]
    flat = prob.reshape(n, -1)
    srt, _ = flat.sort(dim=1, descending=True)
    csum = srt.cumsum(dim=1)
    # k = index of the last pixel included = first index whose cumulative mass reaches p
    k = (csum < p).sum(dim=1, keepdim=True).clamp(max=flat.shape[1] - 1)
    a_p = srt.gather(1, k)                                      # [N,1] nucleus cutoff value
    a_max = srt[:, :1]                                          # [N,1]
    denom = (a_max - a_p).clamp_min(DEGEN_EPS)
    s = ((flat - a_p) / denom).clamp(0, 1)
    if gamma != 1.0:
        s = s.clamp_min(0).pow(gamma)
    return s.reshape_as(a)


def cumulative_mass_rank(a):
    """C[i] in (0,1] = attention mass of every pixel ranked at or above pixel i (descending).

    C is the MASS-DOMAIN coordinate: C ~ 0 at the attention peak, C = 1 at the weakest pixel. Every
    mask defined as a function of C is automatically invariant to the map's value distribution, so
    its damped budget does NOT swing with the sample's peakiness -- which is the whole problem with
    thresholding the VALUE (the current gate: same 0.15 cut -> 25%..90% of the image, sample to sample).
    All-zero map -> C == 1 everywhere -> any nucleus mask evaluates to 0.
    """
    prob = to_prob(a)
    n = prob.shape[0]
    flat = prob.reshape(n, -1)
    order = flat.argsort(dim=1, descending=True)
    csum = flat.gather(1, order).cumsum(dim=1)
    C = torch.empty_like(csum).scatter_(1, order, csum)
    # a degenerate (all-zero) map has total mass 0 -> csum == 0 -> force C = 1 (outside every nucleus)
    dead = flat.sum(dim=1, keepdim=True) <= DEGEN_EPS
    C = torch.where(dead, torch.ones_like(C), C)
    return C.reshape_as(a)


def s_soft_nucleus(a, p_core=0.3, p_edge=0.7, smooth=True):
    """RECOMMENDED HYBRID -- a nucleus mask with a PLATEAU, defined purely in the mass domain.

        s = 1                              for C <= p_core     (person core: FULL damping, m = factor2)
        s = ramp((p_edge - C)/(p_edge - p_core))  for p_core < C <= p_edge   (soft shell)
        s = 0                              for C >  p_edge     (background: untouched, m = 1)

    It simultaneously satisfies every constraint the hard gate violates:
      * the core is damped by exactly factor2 (0.2)  -> the DAMPING MAGNITUDE is unchanged;
      * the boundary is soft                          -> no bang-bang flip when the attention jitters;
      * outside the nucleus s is EXACTLY 0            -> zero background leakage;
      * it is parameterized by attention MASS         -> the damped budget is near-constant per sample.

    p_core = p_edge reduces exactly to the hard nucleus (bang-bang at mass p).
    smooth=True uses the C1-continuous smoothstep u^2(3-2u) instead of a linear ramp.
    """
    C = cumulative_mass_rank(a)
    u = ((p_edge - C) / max(p_edge - p_core, 1e-6)).clamp(0, 1)
    if smooth:
        u = u * u * (3.0 - 2.0 * u)
    return u


def s_mass_sigmoid(a, p_mid=0.65, tau=0.12):
    """THE UNIFIED GATE -- the user's two ideas as the two knobs of one mask.

        C   = cumulative attention mass rank of the pixel (0 at the peak, 1 at the weakest pixel)
        s   = sigmoid((p_mid - C) / tau),  rescaled so that C=0 -> s=1 and C=1 -> s=0 exactly

      * p_mid  = METHOD 2: the CUMULATIVE-PROBABILITY cut. "Release the pixels carrying the top p_mid
                 of the gender-attention mass." Because it is a mass, not a value, the damped budget
                 no longer swings with the sample's peakiness.
      * tau    = METHOD 1: the TEMPERATURE. tau -> 0 recovers the hard nucleus (bang-bang); larger tau
                 fades the boundary. Applying the temperature in the MASS domain is what makes it work:
                 in the VALUE domain (s_softmax) the same idea is far too weak, because the min-max
                 attention values of face and background are barely separated (mean(g) ~ 0.26).

    The peak stays at s = 1, so the multiplier at the person core is exactly factor2 (the current 0.2):
    the DAMPING MAGNITUDE is preserved by construction.

    Rescaling to the exact endpoints also kills the background floor: sigmoid never reaches 0, so the
    raw form would damp the ENTIRE image (including no-face samples, whose common_attn is zeroed ->
    C == 1 -> s must be exactly 0). Here C=1 maps to s=0 exactly, so the guard is structural.
    """
    C = cumulative_mass_rank(a)
    t = max(tau, 1e-6)
    s0 = torch.sigmoid(torch.tensor(p_mid / t))              # value at C = 0 (the attention peak)
    s1 = torch.sigmoid(torch.tensor((p_mid - 1.0) / t))      # value at C = 1 (the weakest pixel)
    s = torch.sigmoid((p_mid - C) / t)
    return ((s - s1) / (s0 - s1).clamp_min(1e-12)).clamp(0, 1)


# ---- controls (do the attention maps carry any localization signal at all?) -----------------------
def s_center(a, sigma=0.30, thr=None):
    """CONTROL: an isotropic Gaussian CENTER PRIOR, ignoring the attention map entirely.

    If this localizes the face as well as the attention map does, the attention map adds nothing.
    sigma is in units of the image half-width. Returned as a [0,1] field peaked at the center.
    """
    n, H, W = a.shape
    yy = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
    xx = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
    d2 = xx ** 2 + yy ** 2
    s = torch.exp(-d2 / (2 * sigma ** 2))
    s = s / s.max()
    return s.unsqueeze(0).expand(n, H, W).clone()


def s_shuffle(a, seed=0):
    """CONTROL: the attention map with its pixels randomly permuted (same histogram, no structure)."""
    gen = torch.Generator().manual_seed(seed)
    n = a.shape[0]
    flat = a.reshape(n, -1)
    out = torch.empty_like(flat)
    for i in range(n):
        out[i] = flat[i][torch.randperm(flat.shape[1], generator=gen)]
    return out.reshape_as(a)


def s_uniform(a, level=1.0):
    """CONTROL: uniform damping everywhere (no localization); the "just lower wSCR" baseline."""
    return torch.full_like(a, float(level))


# --------------------------------------------------------------------------------------------------
# budget calibration
# --------------------------------------------------------------------------------------------------
def budget(s):
    """mean(s) per image -> [N]. With m = 1 - d*s this is lambda / d."""
    return s.reshape(s.shape[0], -1).mean(dim=1)


def calibrate_param(a, fn, target_budget, lo, hi, iters=40, per_sample=True):
    """Bisect a scalar parameter of `fn(a, param) -> s` so that mean(s) == target_budget.

    fn must be MONOTONE in param (increasing OR decreasing -- detected automatically from the
    bracket). Returns (param [N] or scalar, s [N,H,W], feasible [N] bool).

    per_sample=True solves an independent parameter for EVERY image -> the damped budget becomes
    IDENTICAL across samples (this is the direct fix for "the hard mask's extent swings per sample").
    per_sample=False solves one global parameter matching the dataset-mean budget.

    Infeasible samples (target outside [budget(lo), budget(hi)]) are flagged and clamped to the
    nearest endpoint -- the softmax family, for instance, cannot exceed mean(g) ~= 0.26.
    """
    n = a.shape[0]
    tgt = torch.as_tensor(target_budget, dtype=torch.float32)
    if tgt.ndim == 0:
        tgt = tgt.expand(n).clone()

    b_lo = budget(fn(a, lo))
    b_hi = budget(fn(a, hi))
    increasing = bool((b_hi.mean() >= b_lo.mean()).item())
    if not increasing:                                          # normalize to "increasing in param"
        lo, hi = hi, lo
        b_lo, b_hi = b_hi, b_lo

    feasible = (tgt >= b_lo - 1e-6) & (tgt <= b_hi + 1e-6)

    p_lo = torch.full((n,), float(lo))
    p_hi = torch.full((n,), float(hi))
    for _ in range(iters):
        mid = 0.5 * (p_lo + p_hi)
        b = torch.stack([budget(fn(a[i:i + 1], float(mid[i])))[0] for i in range(n)])
        too_small = b < tgt
        p_lo = torch.where(too_small, mid, p_lo)
        p_hi = torch.where(too_small, p_hi, mid)
    param = 0.5 * (p_lo + p_hi)

    if not per_sample:
        # one global parameter: bisect on the DATASET-MEAN budget instead
        g_lo, g_hi = float(min(lo, hi)), float(max(lo, hi))
        for _ in range(iters):
            mid = 0.5 * (g_lo + g_hi)
            b = budget(fn(a, mid)).mean().item()
            if (b < tgt.mean().item()) == increasing:
                g_lo = mid
            else:
                g_hi = mid
        param = torch.full((n,), 0.5 * (g_lo + g_hi))

    s = torch.cat([fn(a[i:i + 1], float(param[i])) for i in range(n)], dim=0)
    return param, s, feasible


# --------------------------------------------------------------------------------------------------
# strength field -> gradient multiplier
# --------------------------------------------------------------------------------------------------
def to_multiplier(s, depth=0.8, energy_match=None):
    """m = 1 - depth * s, optionally rescaled so that mean(m) equals `energy_match` per image.

    depth = 1 - factor2 (0.8 for the current factor2 = 0.2): the multiplier at the attention peak is
    exactly factor2, i.e. the DEPTH of the damping is preserved -- "the same 0.2 as before".

    energy_match (float, e.g. the baseline's <m> = 0.463): multiply m by c = target / mean(m) so the
    TOTAL SCR gradient energy per sample equals the baseline's. This is mathematically identical to
    keeping m untouched and scaling weight_loss_scr by c, so it is the drop-in way to change the
    mask's SHAPE without changing the loss balance (and it makes the SCR strength constant across
    samples). c < 1 for any peaked mask, so m stays in [0,1]; we clamp and report violations anyway.
    """
    m = (1.0 - depth * s).clamp(0, 1)
    if energy_match is None:
        return m, torch.ones(s.shape[0])
    n = m.shape[0]
    mean_m = m.reshape(n, -1).mean(dim=1).clamp_min(1e-6)
    c = torch.as_tensor(float(energy_match)) / mean_m
    m = (m * c.view(-1, 1, 1)).clamp(0, 1)
    return m, c
