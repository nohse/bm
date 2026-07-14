#!/usr/bin/env python
# coding=utf-8
"""
Diffusion RESIDUAL-ERROR classifier for face / non-face, ported from the estimator
in `residual_gender_logits` of 1-main-errorDAL,SCR,SRR_person_truncated_hspace.py
but WITHOUT the cross-attention spatial weighting (uniform spatial mean instead).

Idea (Diffusion Classifier, Li et al. 2023 -- exactly the mechanism the repo uses
for woman/man):  for a clean latent z0 and a class prompt c, add fresh noise eps at
K timesteps, predict eps with the FROZEN SD1.5 UNet, and measure the reconstruction
error  E_c = mean_t || eps_pred(z_t, t, c) - eps ||^2 .  The class whose prompt best
"explains" the image has the LOWEST error;  logit_c = -E_c / tau ;  predict argmin_c E_c.
The SAME noise/eps is shared across the compared prompts (paired, low-variance).

We sweep, and report which condition separates face vs non-face best:
  * PROMPT PAIRS       : many face-prompt x nonface-prompt wordings (incl. the user's
                         literal "a photo of a face" vs "a photo of a non face").
  * CONDITIONING SCHEME: cond      = || eps_c        - eps_true ||^2   (standard)
                         cfg{g}    = || eps_u + g(eps_c-eps_u) - eps_true ||^2  (uncond=empty)
                         eps_diff  = || eps_c - eps_u ||^2             (prompt effect size)
                         negcfg{g} = opposite CLASS prompt used as the negative branch
  * TIMESTEP GRID      : broad/low/mid/high/repo ranges of t.

Metric per (grid,pair,scheme): accuracy (decide face iff E_face < E_nonface, i.e.
margin = E_nonface - E_face > 0) and threshold-free ROC-AUC of the margin.
"""
import os, sys, json, argparse, itertools, math
import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "runwayml/stable-diffusion-v1-5"
DEV = "cuda"
DT = torch.float16

# ------------------------------------------------------------------ prompt vocab
FACE_PROMPTS = {
    "F_face":        "a photo of a face",
    "F_humanface":   "a photo of a human face",
    "F_closeface":   "a close-up photo of a person's face",
    "F_portrait":    "a portrait photo of a person",
    "F_person":      "a photo of a person",
}
NONFACE_PROMPTS = {
    "N_nonface":     "a photo of a non face",                 # user's literal wording
    "N_withoutface": "a photo without a face",
    "N_noface":      "a photo of no face",
    "N_landscape":   "a photo of a landscape",
    "N_object":      "a photo of an object",
    "N_nopeople":    "a photo of a scene without any people",
    "N_scenery":     "a photo of scenery",
    "N_photo":       "a photo",
}
UNCOND = ""  # empty prompt = unconditional

# ------------------------------------------------------------------ timestep grids
GRIDS = {
    "broad_50_950":  (50, 950, 25),
    "repo_400_800":  (400, 800, 25),
    "low_50_400":    (50, 400, 25),
    "mid_200_600":   (200, 600, 25),
    "high_600_950":  (600, 950, 25),
}
CFG_SCALES = [2.0, 4.0, 7.5]


# ------------------------------------------------------------------ model
def load_models():
    tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=DT).to(DEV).eval()
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=DT).to(DEV).eval()
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=DT).to(DEV).eval()
    sched = DPMSolverMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")
    assert sched.config.prediction_type == "epsilon"
    for m in (te, vae, unet):
        m.requires_grad_(False)
    return tok, te, vae, unet, sched


@torch.no_grad()
def embed(tok, te, text):
    ids = tok([text], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(DEV)
    return te(ids)[0]  # [1,77,768] fp16


@torch.no_grad()
def encode_images(vae, files):
    """Load PILs -> [-1,1] tensor -> deterministic VAE latent mean * scaling_factor -> z0 [N,4,64,64]."""
    zs = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((512, 512))
        x = torch.from_numpy(np.array(img)).float().div(127.5).sub(1.0)  # [-1,1]
        x = x.permute(2, 0, 1).unsqueeze(0).to(DEV, DT)
        lat = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        zs.append(lat)
    return torch.cat(zs, 0)  # [N,4,64,64]


def make_grid_noise(z0_chunk, t_lo, t_hi, K, sched, seed):
    """Build shared eps/zt for a chunk of images. Returns eps_all,zt_all,t_all folded [chunk*K,...]."""
    n = z0_chunk.shape[0]
    ts = torch.linspace(t_lo, t_hi, steps=K, device=DEV).round().long()  # [K]
    g = torch.Generator(device=DEV).manual_seed(seed)
    eps_list, zt_list = [], []
    for t in ts:
        eps = torch.randn(z0_chunk.shape, generator=g, device=DEV, dtype=z0_chunk.dtype)
        zt_list.append(sched.add_noise(z0_chunk, eps, t.repeat(n)))
        eps_list.append(eps)
    eps_all = torch.stack(eps_list, 1).reshape(n * K, *z0_chunk.shape[1:])
    zt_all = torch.stack(zt_list, 1).reshape(n * K, *z0_chunk.shape[1:]).to(DT)
    t_all = ts.repeat(n)  # row i*K+k -> t_k
    return eps_all, zt_all, t_all, n


@torch.no_grad()
def score_vocab(unet, z0, embeds, uncond_embed, grid, seed, chunk=10, cfg_scales=CFG_SCALES):
    """
    For every prompt in `embeds` (dict name->[1,77,768]) compute per-image scalar errors
    under three schemes, all from the SAME shared noise:
       E_cond[name] , E_diff[name] , E_cfg{g}[name]
    Returns dict scheme -> {name -> np.array[N]}.
    """
    t_lo, t_hi, K = grid
    N = z0.shape[0]
    names = list(embeds.keys())
    out = {"cond": {n: np.zeros(N) for n in names},
           "eps_diff": {n: np.zeros(N) for n in names}}
    for g in cfg_scales:
        out[f"cfg{g}"] = {n: np.zeros(N) for n in names}

    for s in range(0, N, chunk):
        z0c = z0[s:s + chunk]
        eps_all, zt_all, t_all, nC = make_grid_noise(z0c, t_lo, t_hi, K, SCHED, seed + s)
        eps_u = unet(zt_all, t_all, encoder_hidden_states=uncond_embed.expand(nC * K, -1, -1)).sample.float()
        eps_true = eps_all.float()
        for name in names:
            c = embeds[name].expand(nC * K, -1, -1)
            eps_c = unet(zt_all, t_all, encoder_hidden_states=c).sample.float()
            def redu(err):  # [nC*K,4,64,64] -> per-image mean over K [nC]
                return err.pow(2).mean(dim=(1, 2, 3)).view(nC, K).mean(1).cpu().numpy()
            out["cond"][name][s:s + nC] = redu(eps_c - eps_true)
            out["eps_diff"][name][s:s + nC] = redu(eps_c - eps_u)
            for g in cfg_scales:
                guided = eps_u + g * (eps_c - eps_u)
                out[f"cfg{g}"][name][s:s + nC] = redu(guided - eps_true)
    return out


@torch.no_grad()
def score_negcfg(unet, z0, face_embed, non_embed, grid, seed, g, chunk=10):
    """Symmetric class-as-negative-prompt CFG. Returns (E_face, E_non) per image [N].
       E_face uses non-face prompt as the negative branch and vice-versa."""
    t_lo, t_hi, K = grid
    N = z0.shape[0]
    Ef, En = np.zeros(N), np.zeros(N)
    for s in range(0, N, chunk):
        z0c = z0[s:s + chunk]
        eps_all, zt_all, t_all, nC = make_grid_noise(z0c, t_lo, t_hi, K, SCHED, seed + s)
        eps_true = eps_all.float()
        ef = unet(zt_all, t_all, encoder_hidden_states=face_embed.expand(nC * K, -1, -1)).sample.float()
        en = unet(zt_all, t_all, encoder_hidden_states=non_embed.expand(nC * K, -1, -1)).sample.float()
        def redu(err):
            return err.pow(2).mean(dim=(1, 2, 3)).view(nC, K).mean(1).cpu().numpy()
        guided_face = en + g * (ef - en)   # push toward face, away from non-face
        guided_non = ef + g * (en - ef)
        Ef[s:s + nC] = redu(guided_face - eps_true)
        En[s:s + nC] = redu(guided_non - eps_true)
    return Ef, En


# ------------------------------------------------------------------ metrics
def roc_auc(scores, labels):
    """AUC that higher score -> label 1. labels in {0,1}."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    s = scores[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    ranks[order] = ranks_sorted
    pos = labels == 1
    n_pos = pos.sum(); n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def eval_pair(E_face, E_non, labels):
    """labels: 1=face. margin = E_non - E_face (>0 => predict face)."""
    margin = E_non - E_face
    pred = (margin > 0).astype(int)
    acc = float((pred == labels).mean())
    auc = float(roc_auc(margin, labels))
    # separation: mean margin on faces should be >0, on non-faces <0
    return dict(acc=acc, auc=auc,
                margin_face=float(margin[labels == 1].mean()),
                margin_nonface=float(margin[labels == 0].mean()))


def main():
    global SCHED
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.json"))
    args = ap.parse_args()

    with open(os.path.join(HERE, "manifest.json")) as f:
        manifest = json.load(f)
    files = [os.path.join(HERE, m["file"]) for m in manifest]
    labels = np.array([1 if m["label"] == "face" else 0 for m in manifest])
    print(f"{len(files)} images: {int(labels.sum())} face / {int((1-labels).sum())} nonface")

    tok, te, vae, unet, SCHED = load_models()
    z0 = encode_images(vae, files)
    print("z0", tuple(z0.shape))

    all_embeds = {**{k: embed(tok, te, v) for k, v in FACE_PROMPTS.items()},
                  **{k: embed(tok, te, v) for k, v in NONFACE_PROMPTS.items()}}
    uncond_embed = embed(tok, te, UNCOND)

    results = {"grids": {}, "meta": dict(seed=args.seed, cfg_scales=CFG_SCALES,
               face_prompts=FACE_PROMPTS, nonface_prompts=NONFACE_PROMPTS,
               n_face=int(labels.sum()), n_nonface=int((1 - labels).sum()))}

    for gname, grid in GRIDS.items():
        print(f"\n=== grid {gname} {grid} ===")
        scored = score_vocab(unet, z0, all_embeds, uncond_embed, grid, args.seed, args.chunk)
        grid_res = {}
        for scheme, tbl in scored.items():
            rows = []
            for fk in FACE_PROMPTS:
                for nk in NONFACE_PROMPTS:
                    r = eval_pair(tbl[fk], tbl[nk], labels)
                    rows.append(dict(face=fk, nonface=nk, **r))
            rows.sort(key=lambda r: (-r["acc"], -r["auc"]))
            grid_res[scheme] = rows
        results["grids"][gname] = grid_res
        # quick per-grid console summary (cond scheme)
        best = grid_res["cond"][0]
        print(f"  [cond] best pair: {best['face']} vs {best['nonface']}  "
              f"acc={best['acc']:.3f} auc={best['auc']:.3f}")

    # ---- targeted negcfg (class-as-negative-prompt) on the broad grid, user's literal + person pair
    neg_res = {}
    for (fk, nk) in [("F_face", "N_nonface"), ("F_person", "N_nopeople"), ("F_humanface", "N_landscape")]:
        for g in [2.0, 4.0]:
            Ef, En = score_negcfg(unet, z0, all_embeds[fk], all_embeds[nk],
                                  GRIDS["broad_50_950"], args.seed, g, args.chunk)
            neg_res[f"{fk}|{nk}|negcfg{g}"] = eval_pair(Ef, En, labels)
    results["negcfg_broad"] = neg_res

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
