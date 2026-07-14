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
  * the SCALE of |g_p| -> which axis would DOMINATE an unweighted mean-of-gaps, and by how much.
  * the axis x axis correlation matrix -> are the 4 axes redundant views or genuinely different routes?
  * pooled AUC for three poolings:
        raw    : mean_p g_p                       (== mean-of-errors == mean-of-logits == mean-of-log-probs)
        znorm  : mean_p g_p / s_p , s_p = std of g_p over the set  (equalises the axes)
        probavg: mean_p sigmoid(g_p / tau)        (the genuinely-different pooling; watch it saturate)
  * a per-axis tau suggestion  tau_p = std(g_p), so that g_p/tau_p is O(1) and the softmax does not saturate.

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
  * any axis with AUC < 0.60 is not carrying valence -- drop it or re-word it.
  * if the POOLED znorm AUC < 0.75, the multi-prompt scorer is too weak to drive a 50/50 target split:
    re-word the prompts (see PROMPT_VARIANTS below) or re-think the axes before training.
"""
import os, sys, json, math, argparse, itertools
import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "runwayml/stable-diffusion-v1-5"
DEV = "cuda"
DT = torch.float16

# ---------------------------------------------------------------------------- the 4 contrastive axes
# Each axis is a MINIMAL CONTRAST PAIR: only the polarity word differs between pos and neg, so the axis's
# baseline error offset (prompt length, rarity, etc.) cancels exactly in the gap g_p = E_neg - E_pos.
# `word` is the CONTENT noun whose cross-attention map localises the axis (NOT the polarity adjective --
# the trainer needs a REGION from this map, and "positive"/"negative" are adjectives with diffuse maps).
AXES = {
    "expr": dict(
        pos="A photo of positive facial expressions",
        neg="A photo of negative facial expressions",
        word="expressions",
    ),
    "cloth": dict(
        pos="A photo of positive clothing and actions",
        neg="A photo of negative clothing and actions",
        word="clothing",
    ),
    "bg": dict(
        pos="A photo of a background with a positive atmosphere",
        neg="A photo of a background with a negative atmosphere",
        word="background",
    ),
    "tone": dict(
        pos="A photo of a scene with an overall positive tone",
        neg="A photo of a scene with an overall negative tone",
        word="scene",
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
    """The frozen SD1.5 residual-error scorer -- the exact estimator the trainer uses."""

    def __init__(self, K, t_min, t_max, seed=0):
        self.tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
        self.te = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEV, DT).eval()
        self.vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(DEV, DT).eval()
        self.unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").to(DEV, DT).eval()
        self.sched = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
        self.gen_sched = DPMSolverMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")
        self.timesteps = torch.linspace(t_min, t_max, steps=K).round().long().to(DEV)
        self.K = K
        self.seed = seed
        for m in (self.te, self.vae, self.unet):
            m.requires_grad_(False)

    @torch.no_grad()
    def embed(self, prompt):
        # NOTE: the trainer's woman/man/SRR scorers encode WITH the padding attention_mask
        # (_encode_scoring_prompt); the face/faceless pair was ablated WITHOUT it. That convention flip
        # silently destroyed the face gate once. We match the WITH-mask convention here because the
        # valence prompts will live in the SAME scorer as woman/man did.
        t = self.tok([prompt], padding="max_length", max_length=self.tok.model_max_length,
                     truncation=True, return_tensors="pt")
        return self.te(t.input_ids.to(DEV), t.attention_mask.to(DEV))[0].to(DT)

    @torch.no_grad()
    def generate(self, prompt, n, steps=25, gs=7.5, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        lat = torch.randn([n, 4, 64, 64], generator=g, device=DEV, dtype=DT)
        c = self.embed(prompt).expand(n, -1, -1)
        u = self.embed("").expand(n, -1, -1)
        self.gen_sched.set_timesteps(steps, device=DEV)
        lat = lat * self.gen_sched.init_noise_sigma
        for t in self.gen_sched.timesteps:
            inp = self.gen_sched.scale_model_input(torch.cat([lat] * 2), t)
            e = self.unet(inp, t, encoder_hidden_states=torch.cat([u, c])).sample
            eu, ec = e.chunk(2)
            lat = self.gen_sched.step(eu + gs * (ec - eu), t, lat).prev_sample
        img = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        z0 = lat / self.vae.config.scaling_factor * self.vae.config.scaling_factor  # keep scheduler scale
        return lat, img  # lat == z0 in the scheduler/UNet scale, exactly what the trainer scores

    @torch.no_grad()
    def error(self, z0, prompt):
        """E_c = mean_t mean_pixels ||eps_pred - eps||^2, with the SAME eps per timestep across prompts."""
        n = z0.shape[0]
        c = self.embed(prompt).expand(n * self.K, -1, -1)
        g = torch.Generator(device=DEV).manual_seed(self.seed)  # SAME eps for every prompt -> paired
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
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--variants", action="store_true",
                    help="also score every wording in PROMPT_VARIANTS (the fix if an axis is dead)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()

    tag = f"{a.label_mode}_n{a.n_per_class}_K{a.K}_t{a.t_min}-{a.t_max}"
    outdir = os.path.join(a.out, tag)
    os.makedirs(outdir, exist_ok=True)

    S = Scorer(a.K, a.t_min, a.t_max)

    # ---------------- build the labelled set
    z0s, labels, imgs = [], [], []
    if a.label_mode == "gen":
        per = max(1, a.n_per_class // len(OCCUPATIONS))
        for occ in OCCUPATIONS:
            for lab, suf in ((1, GT_POS_SUFFIX), (0, GT_NEG_SUFFIX)):
                z0, im = S.generate(TEMPLATE.format(occ) + suf, per, seed=hash((occ, lab)) % 10_000)
                z0s.append(z0); imgs.append(im); labels += [lab] * per
        z0 = torch.cat(z0s); labels = np.array(labels)
    else:
        per = max(1, (2 * a.n_per_class) // len(OCCUPATIONS))
        for occ in OCCUPATIONS:
            zz, im = S.generate(TEMPLATE.format(occ), per, seed=hash(occ) % 10_000)
            z0s.append(zz); imgs.append(im)
        z0 = torch.cat(z0s)
        labels, _margin = clip_valence_labels(torch.cat(imgs))

    print(f"[set] {z0.shape[0]} images, {labels.sum()} positive / {(1-labels).sum()} negative")

    # ---------------- score every axis
    axis_sets = {k: [(v["pos"], v["neg"])] for k, v in AXES.items()}
    if a.variants:
        axis_sets = {k: PROMPT_VARIANTS[k] for k in AXES}

    results, gaps = {}, {}
    for ax, pairs in axis_sets.items():
        for wi, (pp, np_) in enumerate(pairs):
            E_pos = S.error(z0, pp)
            E_neg = S.error(z0, np_)
            g = E_neg - E_pos                      # >0 => looks POSITIVE
            key = ax if wi == 0 else f"{ax}#{wi}"
            auc = roc_auc(g, labels)
            results[key] = dict(pos_prompt=pp, neg_prompt=np_, auc=float(auc),
                                E_pos_mean=float(E_pos.mean()), E_neg_mean=float(E_neg.mean()),
                                gap_mean=float(g.mean()), gap_std=float(g.std()),
                                tau_suggested=float(g.std()))
            if wi == 0:
                gaps[ax] = g
            print(f"[{key:12s}] AUC={auc:.3f}  |g|={np.abs(g).mean():.3e}  std={g.std():.3e}  "
                  f"E_pos={E_pos.mean():.4f} E_neg={E_neg.mean():.4f}  {'<-- DEAD' if auc < 0.60 else ''}")

    # ---------------- pooling comparison
    G = np.stack([gaps[k] for k in AXES], 1)                       # [N,P]
    raw = G.mean(1)                                                 # == mean-errors == mean-logits == mean-logprobs
    znorm = (G / (G.std(0, keepdims=True) + 1e-12)).mean(1)         # per-axis scale-equalised
    probavg = 1.0 / (1.0 + np.exp(-G / a.tau))                      # per-axis sigmoid (the genuinely different one)
    probavg = probavg.mean(1)

    pooled = {
        "raw   (mean of gaps == mean-errors == mean-logits)": roc_auc(raw, labels),
        "znorm (per-axis std-normalised mean of gaps)": roc_auc(znorm, labels),
        f"probavg (mean of per-axis sigmoid(g/tau={a.tau:g}))": roc_auc(probavg, labels),
    }
    share = np.abs(G).mean(0); share = share / share.sum()
    corr = np.corrcoef(G.T)
    n_tied = len(probavg) - len(np.unique(np.round(probavg, 6)))

    print("\n=== POOLED ===")
    for k, v in pooled.items():
        print(f"  AUC {v:.3f}   {k}")
    print("\n=== SCALE DOMINATION (share of the unweighted mean-of-gaps budget) ===")
    for k, s in zip(AXES, share):
        print(f"  {k:6s} {s:6.1%}")
    print("\n=== AXIS CORRELATION ===")
    print("        " + "  ".join(f"{k:>6s}" for k in AXES))
    for i, k in enumerate(AXES):
        print(f"  {k:6s} " + "  ".join(f"{corr[i,j]:6.2f}" for j in range(len(AXES))))
    print(f"\n[probavg] tied values: {n_tied}/{len(probavg)} "
          f"({'RANKING WILL DEGENERATE' if n_tied > len(probavg)*0.1 else 'ok'})")

    summary = dict(tag=tag, n=int(z0.shape[0]), n_pos=int(labels.sum()), axes=results,
                   pooled={k: float(v) for k, v in pooled.items()},
                   scale_share={k: float(s) for k, s in zip(AXES, share)},
                   corr={k: [float(x) for x in corr[i]] for i, k in enumerate(AXES)},
                   probavg_ties=int(n_tied),
                   tau_suggested_per_axis={k: float(gaps[k].std()) for k in AXES})
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(outdir, "per_image.csv"), "w") as f:
        f.write("idx,label," + ",".join(f"gap_{k}" for k in AXES) + ",pooled_raw,pooled_znorm,pooled_probavg\n")
        for i in range(len(labels)):
            f.write(f"{i},{labels[i]}," + ",".join(f"{G[i,j]:.6e}" for j in range(G.shape[1]))
                    + f",{raw[i]:.6e},{znorm[i]:.6e},{probavg[i]:.6f}\n")

    dead = [k for k, v in results.items() if v["auc"] < 0.60]
    verdict = []
    if dead:
        verdict.append(f"DEAD AXES (AUC<0.60): {dead} -> re-word them (--variants) or drop them.")
    if pooled["znorm (per-axis std-normalised mean of gaps)"] < 0.75:
        verdict.append("POOLED znorm AUC < 0.75 -> the scorer is too weak to drive a 50/50 split. "
                       "DO NOT TRAIN YET: re-word the prompts or re-think the axes.")
    if not verdict:
        verdict.append("PASS: the pooled scorer separates valence. Proceed to training; "
                       "set --valence_tau per the suggested per-axis stds.")
    print("\n=== VERDICT ===\n" + "\n".join("  " + v for v in verdict))
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write("\n".join(verdict) + "\n\n" + json.dumps(summary, indent=2))
    print(f"\n-> {outdir}")


if __name__ == "__main__":
    main()
