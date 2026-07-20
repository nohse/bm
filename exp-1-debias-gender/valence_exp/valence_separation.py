#!/usr/bin/env python
# coding=utf-8
"""
OFFLINE PRE-CHECK for the multi-prompt POSITIVE/NEGATIVE (valence) residual scorer.

WHY THIS EXISTS -- read before spending a GPU-hour on training.
This repo has twice been burned by a *plausible-looking* prompt-based residual scorer that turned out
to carry no signal at all:
  * the woman/man SDS "classifier" whose fair loss sat pinned at ln(2) (p ~= 0.5, no gender signal);
  * "a photo of a realistic face" as a face/no-face gate -> ROC-AUC 0.504, i.e. a coin flip.
The face/no-face prompt pair that DID work ("a photo of a face" vs "a faceless photo", 88/100) was only
adopted after exactly this kind of offline sweep (face_error_exp/exp100). This script is that sweep, for
valence.

THE ESTIMATOR (identical to residual_gender_logits in
1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector_attmap.py, minus the attention weighting):
for a clean latent z0 and a class prompt c, add fresh noise eps at K timesteps, predict eps with the FROZEN
SD1.5 UNet, and measure  E_c = mean_t mean_pixels || eps_pred(z_t, t, c) - eps ||^2 .
The SAME eps/z_t is shared by the two prompts of a pair (paired, low-variance).
Per axis p, the signed GAP is
    g_p = E_p(negative prompt) - E_p(positive prompt)       # g_p > 0  <=>  image looks POSITIVE
(sign chosen so that larger = more positive; the pooled decision in the trainer is a function of the mean
of the g_p ONLY -- the per-axis baseline error offsets cancel exactly in the within-axis difference).

WHAT IT REPORTS, per axis and for the pooled scorers:
  * ROC-AUC of g_p against the ground-truth valence label  -> IS THERE ANY SIGNAL AT ALL?
    (0.5 = dead. This is the number that killed the "realistic face" gate.)
  * TEST-RETEST reliability of g_p: its correlation with ITSELF across two independent eps draws. The
    scorer is stochastic, and the trainer computes the ranking targets on one draw and the loss on
    another; an axis whose gap does not correlate with itself is Monte-Carlo noise, not signal. Neither
    the AUC nor the loss curve reveals this.
  * the SCALE of |g_p| -> which axis would DOMINATE an unweighted mean-of-gaps, and by how much.
  * the axis x axis correlation matrix -> are the 4 axes redundant views or genuinely different routes?
  * pooled AUC for four poolings:
        raw     : mean_p g_p          == mean-of-ERRORS == mean-of-LOGITS == mean-of-LOG-PROBS.
                  These are ONE estimator, not three options (logit_c = -E_c/tau is affine in E with a
                  shared tau, so the mean commutes with it; the log-prob variant differs only by a
                  class-independent term that softmax/CE ignore). Verified in float64: diff 4.6e-14.
        znorm   : mean_p (g_p - m_p)/s_p                  -- standardised, uniform weights
        zw      : sum_p w_p (g_p - m_p)/s_p , w_p ~ d'    -- standardised, DISCRIMINABILITY-weighted.
                  THIS IS WHAT THE TRAINER SHIPS. 1/s_p alone equalises each axis's VARIANCE, which is
                  necessary (else the largest-scale axis owns the decision) but NOT sufficient: it also
                  inflates a pure-noise axis to unit variance, which a uniform 1/P weight then injects at
                  full strength. So weight by measured separation, and give a dead axis weight 0.
        probavg : mean_p sigmoid(z_p / tau)   -- the ONE genuinely different pooling, and the one that
                  breaks (a mixture: a confident axis cannot be outvoted, saturated axes give no
                  gradient, and under a small tau it collapses onto the grid {0,.25,.5,.75,1}, whose
                  argsort hands 50/50 targets out by ARRIVAL ORDER).
  * THE CALIBRATION CONSTANTS to paste into the trainer:
        --valence_axis_center / --valence_axis_scale / --valence_axis_weight

GROUND TRUTH. Two ways to get labelled images (--label_mode):
  gen      (default, no external labels needed): generate images from the occupation prompt template with an
           explicitly POSITIVE suffix and an explicitly NEGATIVE suffix. The suffix is the ground truth. This
           measures whether the scorer can see valence THAT THE MODEL WAS EXPLICITLY TOLD TO PUT THERE -- an
           UPPER BOUND on its ability. If AUC is near 0.5 here, the scorer is dead and no training will work.
  clip     generate NEUTRAL occupation images (exactly what training sees) and label them with a zero-shot
           CLIP valence head -- an INDEPENDENT model. This measures agreement with the metric you will
           actually be evaluated on. Run this second; `gen` is the cheaper kill-switch.

USAGE (on the GPU box):
    python valence_separation.py --label_mode gen  --n_per_class 50 --K 15 --t_min 400 --t_max 800
    python valence_separation.py --label_mode clip --n_per_class 100
Results -> valence_exp/out/<tag>/{per_image.csv, summary.json, report.txt}

PASS/FAIL GATE (decide BEFORE training):
  * any axis with AUC < 0.60 is not carrying valence          -- re-word it (--variants) or drop it.
  * any axis with test-retest r < 0.30 is Monte-Carlo noise   -- raise --K or drop it.
  * if the POOLED d'-weighted (zw) AUC < 0.75, the scorer is too weak to drive a reliable 50/50 target
    split: re-word the prompts (see PROMPT_VARIANTS below) or re-think the axes. DO NOT TRAIN.

NOTE the two settings that MUST match the trainer or the constants will not transfer:
  --skip_denoise_frac (default 0.5)  training scores a TRUNCATED x0-hat, not a fully-denoised latent
  the scheduler                      a single DPMSolverMultistepScheduler drives BOTH generation and
                                     add_noise (the trainer does this; a DDPMScheduler would give a
                                     different z_t)
"""
import os, sys, json, math, argparse, itertools
import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "runwayml/stable-diffusion-v1-5"
DEV = "cuda"
DT = torch.float16

# ---------------------------------------------------------------------------- the 4 contrastive axes
# Each axis is a MINIMAL CONTRAST PAIR: only the polarity word differs between pos and neg, so the axis's
# baseline error offset (prompt length, rarity, etc.) cancels exactly in the gap g_p = E_neg - E_pos.
# (No per-axis attention token: the trainer does NOT use a cross-attention region for valence. The 4 axes
# localise to different places, their maps average toward uniform, two of them have non-contiguous content
# words, and the 8 prompts have incommensurable token lengths. See the trainer header.)
AXES = {
    "expr": dict(
        pos="A photo of positive facial expressions",
        neg="A photo of negative facial expressions",
    ),
    "cloth": dict(
        pos="A photo of positive clothing and actions",
        neg="A photo of negative clothing and actions",
    ),
    "bg": dict(
        pos="A photo of a background with a positive atmosphere",
        neg="A photo of a background with a negative atmosphere",
    ),
    "tone": dict(
        pos="A photo of a scene with an overall positive tone",
        neg="A photo of a scene with an overall negative tone",
    ),
}

# Fallback wordings to try if an axis comes back dead (AUC < 0.60). "positive"/"negative" are abstract
# adjectives; CLIP's text encoder may barely move on them. These swap in CONCRETE affect words, which is
# the single most likely fix.
PROMPT_VARIANTS = {
    "expr": [
        ("A photo of positive facial expressions", "A photo of negative facial expressions"),
        ("A photo of a happy smiling face", "A photo of a sad angry face"),
        ("A photo of a cheerful person", "A photo of a gloomy person"),
    ],
    "cloth": [
        ("A photo of positive clothing and actions", "A photo of negative clothing and actions"),
        ("A photo of bright clothing and confident actions", "A photo of drab clothing and defeated actions"),
    ],
    "bg": [
        ("A photo of a background with a positive atmosphere", "A photo of a background with a negative atmosphere"),
        ("A photo of a bright cheerful background", "A photo of a dark gloomy background"),
    ],
    "tone": [
        ("A photo of a scene with an overall positive tone", "A photo of a scene with an overall negative tone"),
        ("A photo of a warm uplifting scene", "A photo of a cold depressing scene"),
    ],
}

# ground-truth generation suffixes for --label_mode gen (the label IS the suffix)
GT_POS_SUFFIX = ", happy and smiling, bright cheerful setting, warm uplifting atmosphere"
GT_NEG_SUFFIX = ", sad and angry, dark gloomy setting, cold depressing atmosphere"

# CLIP zero-shot valence head for --label_mode clip (an INDEPENDENT model -- also the proposed eval metric)
CLIP_POS = [
    "a photo of a happy positive person in a cheerful bright scene",
    "a positive, uplifting, cheerful photo",
]
CLIP_NEG = [
    "a photo of a sad negative person in a gloomy dark scene",
    "a negative, depressing, gloomy photo",
]

OCCUPATIONS = ["doctor", "nurse", "lawyer", "teacher", "chef", "firefighter", "scientist", "janitor"]
TEMPLATE = "a photo of the face of a {}, a person"


def roc_auc(scores, labels):
    """AUC = P(score[pos] > score[neg]), ties counted as 0.5. No sklearn dependency."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


class Scorer:
    """The frozen SD1.5 residual-error scorer -- the EXACT estimator the trainer uses.

    Two details below are not cosmetic; getting either wrong makes the calibration constants measured here
    fail to transfer to training:

    1. ONE SCHEDULER. The trainer builds a single DPMSolverMultistepScheduler and uses it for BOTH the
       denoising loop AND the scorer's add_noise. Using a DDPMScheduler for add_noise (the intuitive choice)
       gives a different alphas_cumprod convention and therefore a different z_t.
    2. TRUNCATED z0. Training generates with --skip_denoise_frac (default 0.5): it runs only the first
       (1-frac) of the scheduler steps and then JUMPS to a predicted clean latent x0 via the closed-form
       eps->x0 formula. The z0 the scorer actually sees at train time is therefore a BLURRIER x0-hat, not a
       fully-denoised latent. Calibrating on full 25-step images would measure the gaps on a distribution the
       trainer never sees.
    """

    def __init__(self, K, t_min, t_max, seed=0):
        self.tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
        self.te = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEV, DT).eval()
        self.vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(DEV, DT).eval()
        self.unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").to(DEV, DT).eval()
        # SINGLE scheduler, exactly as the trainer does (noise_scheduler = DPMSolverMultistepScheduler)
        self.sched = DPMSolverMultistepScheduler.from_config(MODEL, subfolder="scheduler")
        self.timesteps = torch.linspace(t_min, t_max, steps=K).round().long().to(DEV)
        self.K = K
        self.seed = seed
        for m in (self.te, self.vae, self.unet):
            m.requires_grad_(False)

    @torch.no_grad()
    def embed(self, prompt):
        # The trainer's class/SRR scorers encode WITH the padding attention_mask (_encode_scoring_prompt);
        # the face/faceless pair was ablated WITHOUT it. That convention flip silently destroyed the face
        # gate once (masked embeds made it call nearly everything "faceless"). The valence prompts live in
        # the SAME scorer the woman/man pair lived in, so they use the WITH-mask convention.
        t = self.tok([prompt], padding="max_length", max_length=self.tok.model_max_length,
                     truncation=True, return_tensors="pt")
        return self.te(t.input_ids.to(DEV), t.attention_mask.to(DEV))[0].to(DT)

    @torch.no_grad()
    def generate(self, prompt, n, steps=21, gs=7.5, seed=0, skip_denoise_frac=0.5):
        """Returns (z0, images). z0 is the clean latent in the SCHEDULER/UNET scale -- what the trainer scores.

        Reproduces generate_image_no_gradient, including the truncated-denoising jump to x0-hat.
        `steps` defaults to 21 because the trainer draws num_denoising_steps from range(19,24) each step.
        """
        g = torch.Generator(device=DEV).manual_seed(seed)
        lat = torch.randn([n, 4, 64, 64], generator=g, device=DEV, dtype=DT)
        c = self.embed(prompt).expand(n, -1, -1)
        u = self.embed("").expand(n, -1, -1)
        self.sched.set_timesteps(steps, device=DEV)
        ts = self.sched.timesteps
        n_run = round((1.0 - float(skip_denoise_frac)) * steps)
        n_run = max(1, min(n_run, steps))

        for i, t in enumerate(ts):
            inp = self.sched.scale_model_input(torch.cat([lat] * 2), t)
            e = self.unet(inp, t, encoder_hidden_states=torch.cat([u, c])).sample
            eu, ec = e.chunk(2)
            eps = eu + gs * (ec - eu)
            if i == n_run - 1:
                # closed-form eps -> x0 jump (the truncation), exactly as the trainer does
                abar = self.sched.alphas_cumprod[t].to(device=lat.device, dtype=eps.dtype)
                lat = (lat - (1 - abar).sqrt() * eps) / abar.sqrt()
                break
            lat = self.sched.step(eps, t, lat).prev_sample

        z0 = lat                                                    # scheduler/UNet scale
        img = self.vae.decode(z0 / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        return z0, img

    @torch.no_grad()
    def error(self, z0, prompt, eps_seed):
        """E_c = mean_t mean_pixels ||eps_pred - eps||^2.

        `eps_seed` fixes the noise draw. Pass the SAME seed for the two prompts of a pair (that is what makes
        the gap a low-variance PAIRED difference) and a DIFFERENT seed per axis (independent axes -- otherwise
        the P gap estimates share one noise draw and averaging them cancels no Monte-Carlo noise at all).
        """
        n = z0.shape[0]
        c = self.embed(prompt).expand(n * self.K, -1, -1)
        g = torch.Generator(device=DEV).manual_seed(eps_seed)
        eps_l, zt_l = [], []
        for t in self.timesteps:
            eps = torch.randn(z0.shape, generator=g, device=DEV, dtype=z0.dtype)
            zt_l.append(self.sched.add_noise(z0, eps, t.repeat(n)))
            eps_l.append(eps)
        eps_all = torch.stack(eps_l, 1).reshape(n * self.K, *z0.shape[1:])
        zt_all = torch.stack(zt_l, 1).reshape(n * self.K, *z0.shape[1:]).to(DT)
        t_all = self.timesteps.repeat(n)
        pred = self.unet(zt_all, t_all, encoder_hidden_states=c).sample
        e = (pred.float() - eps_all.float()).pow(2).mean(dim=(1, 2, 3)).view(n, self.K).mean(1)
        return e.cpu().numpy()


@torch.no_grad()
def clip_valence_labels(images):
    """Independent zero-shot valence head (also the proposed EVAL metric replacing the CelebA classifier)."""
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    pil = [Image.fromarray(((i.float().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype("uint8"))
           for i in images]
    inp = proc(text=CLIP_POS + CLIP_NEG, images=pil, return_tensors="pt", padding=True).to(DEV)
    out = model(**inp).logits_per_image                      # [N, 4]
    pos = out[:, :len(CLIP_POS)].mean(1)
    neg = out[:, len(CLIP_POS):].mean(1)
    margin = (pos - neg).cpu().numpy()
    return (margin > 0).astype(int), margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_mode", choices=["gen", "clip"], default="gen")
    ap.add_argument("--n_per_class", type=int, default=50)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--t_min", type=int, default=400)
    ap.add_argument("--t_max", type=int, default=800)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="temperature used ONLY for the probavg comparison row (the trainer's --valence_tau)")
    ap.add_argument("--skip_denoise_frac", type=float, default=0.5,
                    help="MUST MATCH THE TRAINER (--skip_denoise_frac, default 0.5). Training scores a "
                         "TRUNCATED x0-hat, not a fully-denoised latent; calibrating on full denoising would "
                         "measure the gaps on a distribution training never sees.")
    ap.add_argument("--steps", type=int, default=21,
                    help="denoising steps (the trainer draws num_denoising_steps from range(19,24))")
    ap.add_argument("--variants", action="store_true",
                    help="also score every wording in PROMPT_VARIANTS (the fix if an axis is dead)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()

    tag = f"{a.label_mode}_n{a.n_per_class}_K{a.K}_t{a.t_min}-{a.t_max}_skip{a.skip_denoise_frac:g}"
    outdir = os.path.join(a.out, tag)
    os.makedirs(outdir, exist_ok=True)

    S = Scorer(a.K, a.t_min, a.t_max)

    # ---------------- build the labelled set
    z0s, labels, imgs = [], [], []
    if a.label_mode == "gen":
        per = max(1, a.n_per_class // len(OCCUPATIONS))
        for occ in OCCUPATIONS:
            for lab, suf in ((1, GT_POS_SUFFIX), (0, GT_NEG_SUFFIX)):
                z0, im = S.generate(TEMPLATE.format(occ) + suf, per, steps=a.steps,
                                    seed=hash((occ, lab)) % 10_000, skip_denoise_frac=a.skip_denoise_frac)
                z0s.append(z0); imgs.append(im); labels += [lab] * per
        z0 = torch.cat(z0s); labels = np.array(labels)
    else:
        per = max(1, (2 * a.n_per_class) // len(OCCUPATIONS))
        for occ in OCCUPATIONS:
            zz, im = S.generate(TEMPLATE.format(occ), per, steps=a.steps,
                                seed=hash(occ) % 10_000, skip_denoise_frac=a.skip_denoise_frac)
            z0s.append(zz); imgs.append(im)
        z0 = torch.cat(z0s)
        labels, _margin = clip_valence_labels(torch.cat(imgs))

    print(f"[set] {z0.shape[0]} images, {labels.sum()} positive / {(1-labels).sum()} negative")

    # ---------------- score every axis
    axis_sets = {k: [(v["pos"], v["neg"])] for k, v in AXES.items()}
    if a.variants:
        axis_sets = {k: PROMPT_VARIANTS[k] for k in AXES}

    results, gaps, gaps_retest = {}, {}, {}
    for ai, (ax, pairs) in enumerate(axis_sets.items()):
        for wi, (pp, np_) in enumerate(pairs):
            # eps SHARED within the pair (paired difference), INDEPENDENT across axes (so the P gaps do not
            # share one noise draw -- otherwise pooling them cancels no Monte-Carlo noise).
            seed_ax = 1000 + 17 * ai + wi
            E_pos = S.error(z0, pp, eps_seed=seed_ax)
            E_neg = S.error(z0, np_, eps_seed=seed_ax)
            g = E_neg - E_pos                      # >0 => looks POSITIVE
            key = ax if wi == 0 else f"{ax}#{wi}"
            auc = roc_auc(g, labels)

            # TEST-RETEST RELIABILITY: the scorer is STOCHASTIC (a fresh eps every call), and the trainer
            # computes the ranking targets on one noise draw and the loss on ANOTHER. If an axis's gap does
            # not even correlate with itself across two draws, it cannot carry a stable training signal --
            # no amount of pooling will rescue it. This is the diagnostic that distinguishes "weak but real"
            # from "pure Monte-Carlo noise", and neither the AUC nor the loss curve shows it.
            E_pos2 = S.error(z0, pp, eps_seed=seed_ax + 9999)
            E_neg2 = S.error(z0, np_, eps_seed=seed_ax + 9999)
            g2 = E_neg2 - E_pos2
            retest = float(np.corrcoef(g, g2)[0, 1]) if g.std() > 0 and g2.std() > 0 else 0.0

            results[key] = dict(pos_prompt=pp, neg_prompt=np_, auc=float(auc),
                                E_pos_mean=float(E_pos.mean()), E_neg_mean=float(E_neg.mean()),
                                gap_mean=float(g.mean()), gap_std=float(g.std()),
                                test_retest_r=retest)
            if wi == 0:
                gaps[ax] = g
                gaps_retest[ax] = g2
            flags = []
            if auc < 0.60:
                flags.append("DEAD(auc)")
            if retest < 0.30:
                flags.append("NOISE(retest)")
            print(f"[{key:12s}] AUC={auc:.3f}  retest_r={retest:+.2f}  |g|={np.abs(g).mean():.3e}  "
                  f"std={g.std():.3e}  E_pos={E_pos.mean():.4f} E_neg={E_neg.mean():.4f}  "
                  f"{' <-- ' + ','.join(flags) if flags else ''}")

    # ---------------- calibration constants the TRAINER consumes
    names = list(AXES)
    G = np.stack([gaps[k] for k in names], 1)                       # [N,P]
    m_p = G.mean(0)                                                  # --valence_axis_center
    s_p = G.std(0) + 1e-12                                           # --valence_axis_scale
    # WEIGHTS ~ DISCRIMINABILITY, not 1/std. Dividing by s_p equalises each axis's VARIANCE, which is
    # necessary (else the largest-scale axis owns the decision) but NOT sufficient: it inflates a pure-noise
    # axis to unit variance and then a uniform 1/P weight injects it at full strength, actively destroying
    # the pooled signal. So weight by how well each axis actually separates the classes. d' is recovered from
    # the AUC via the standard normal relation  d' = sqrt(2) * Phi^-1(AUC).
    from math import sqrt
    try:
        from scipy.stats import norm as _norm
        _ppf = _norm.ppf
    except Exception:                                                # no scipy -> crude fallback
        _ppf = lambda p: (p - 0.5) * 3.0
    aucs = np.array([results[k]["auc"] for k in names])
    dprime = np.array([sqrt(2.0) * _ppf(min(max(x, 1e-3), 1 - 1e-3)) for x in aucs])
    w_p = np.clip(dprime, 0.0, None)                                 # a below-chance axis contributes nothing
    w_p[aucs < 0.60] = 0.0                                           # hard floor: a dead axis is EXCLUDED
    w_p = w_p / w_p.sum() if w_p.sum() > 0 else np.ones_like(w_p) / len(w_p)

    # ---------------- pooling comparison
    raw = G.mean(1)                                                  # == mean-errors == mean-logits == mean-logprobs
    zc = (G - m_p) / s_p
    znorm = zc.mean(1)                                               # per-axis scale-equalised, uniform weights
    zw = (zc * w_p).sum(1)                                           # per-axis scale-equalised, d'-weighted  <-- SHIPPED
    probavg = (1.0 / (1.0 + np.exp(-zc / a.tau))).mean(1)            # the genuinely different pooling

    pooled = {
        "raw     (mean of gaps == mean-errors == mean-logits == mean-logprobs)": roc_auc(raw, labels),
        "znorm   (per-axis standardised, uniform 1/P weights)": roc_auc(znorm, labels),
        "zw      (per-axis standardised, d'-weighted)  <-- what the trainer ships": roc_auc(zw, labels),
        f"probavg (mean of per-axis sigmoid(z/tau={a.tau:g}))  <-- the one that breaks": roc_auc(probavg, labels),
    }
    share = np.abs(G).mean(0); share = share / share.sum()
    corr = np.corrcoef(G.T)
    n_tied = len(probavg) - len(np.unique(np.round(probavg, 6)))

    print("\n=== POOLED ===")
    for k, v in pooled.items():
        print(f"  AUC {v:.3f}   {k}")
    print("\n=== SCALE DOMINATION (share of the UNWEIGHTED mean-of-gaps budget) ===")
    for k, s in zip(names, share):
        print(f"  {k:6s} {s:6.1%}{'   <-- dominates' if s > 0.5 else ''}")
    print("\n=== AXIS CORRELATION (are the axes redundant views, or different routes?) ===")
    print("        " + "  ".join(f"{k:>6s}" for k in names))
    for i, k in enumerate(names):
        print(f"  {k:6s} " + "  ".join(f"{corr[i,j]:6.2f}" for j in range(len(names))))
    print(f"\n[probavg] tied values: {n_tied}/{len(probavg)} "
          f"({'RANKING WOULD DEGENERATE' if n_tied > len(probavg)*0.1 else 'ok at this tau'})")

    print("\n=== PASTE INTO THE TRAINER ===")
    print("  --valence_axis_center " + " ".join(f"{x:.6g}" for x in m_p))
    print("  --valence_axis_scale  " + " ".join(f"{x:.6g}" for x in s_p))
    print("  --valence_axis_weight " + " ".join(f"{x:.4f}" for x in w_p))
    print("  --valence_tau 1.0     (the pooled score is standardised, so tau is O(1), NOT 1e-4)")
    dead_axes = [k for k, x in zip(names, aucs) if x < 0.60]
    if dead_axes:
        print(f"  # NOTE: {dead_axes} scored AUC<0.60 and are given weight 0 above. Prefer to re-word them")
        print(f"  #       (--variants) or drop them from --valence_axes entirely.")

    summary = dict(tag=tag, n=int(z0.shape[0]), n_pos=int(labels.sum()), axes=results,
                   pooled={k: float(v) for k, v in pooled.items()},
                   scale_share={k: float(s) for k, s in zip(names, share)},
                   corr={k: [float(x) for x in corr[i]] for i, k in enumerate(names)},
                   probavg_ties=int(n_tied),
                   calibration=dict(
                       valence_axis_center=[float(x) for x in m_p],
                       valence_axis_scale=[float(x) for x in s_p],
                       valence_axis_weight=[float(x) for x in w_p],
                   ))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(outdir, "per_image.csv"), "w") as f:
        f.write("idx,label," + ",".join(f"gap_{k}" for k in names)
                + ",pooled_raw,pooled_znorm,pooled_zw,pooled_probavg\n")
        for i in range(len(labels)):
            f.write(f"{i},{labels[i]}," + ",".join(f"{G[i,j]:.6e}" for j in range(G.shape[1]))
                    + f",{raw[i]:.6e},{znorm[i]:.6e},{zw[i]:.6e},{probavg[i]:.6f}\n")

    zw_auc = pooled["zw      (per-axis standardised, d'-weighted)  <-- what the trainer ships"]
    dead = [k for k, v in results.items() if v["auc"] < 0.60]
    noisy = [k for k, v in results.items() if v["test_retest_r"] < 0.30]
    verdict = []
    if dead:
        verdict.append(f"DEAD AXES (AUC < 0.60): {dead} -> re-word them (--variants) or drop them from "
                       f"--valence_axes. They are given weight 0 in the pasted constants above.")
    if noisy:
        verdict.append(f"UNRELIABLE AXES (test-retest r < 0.30): {noisy} -> their gap does not correlate "
                       f"with ITSELF across two noise draws, so they are Monte-Carlo noise, not signal. "
                       f"Raise --K (more timesteps) or drop them.")
    if zw_auc < 0.75:
        verdict.append(f"POOLED (d'-weighted) AUC {zw_auc:.3f} < 0.75 -> the scorer is too weak to drive a "
                       f"reliable 50/50 target split. DO NOT TRAIN YET: re-word the prompts (--variants) or "
                       f"re-think the axes. This project has twice shipped a prompt-based residual scorer "
                       f"with no signal; this gate exists to stop the third time.")
    if not verdict:
        verdict.append(f"PASS: pooled d'-weighted AUC = {zw_auc:.3f}. Proceed to training with the "
                       f"--valence_axis_* constants printed above.")
    print("\n=== VERDICT ===\n" + "\n".join("  " + v for v in verdict))
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write("\n".join(verdict) + "\n\n" + json.dumps(summary, indent=2))
    print(f"\n-> {outdir}")


if __name__ == "__main__":
    main()
