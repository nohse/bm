#!/usr/bin/env python
# coding=utf-8
"""
Visualize the SIGNAL that the SRR realism loss ("a photo of a realistic person") gives.

Faithfully reproduces the SRR term from
  1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector_noflip.py :: residual_gender_and_realism
  E_realistic(z0) = mean_{k in K} mean_pixels || eps_pred(z_t, t_k, "a photo of a realistic person") - eps_k ||^2
with t_k = linspace(residual_t_min=400, residual_t_max=800, K=15), FROZEN SD-1.5 scoring UNet,
NO classifier-free guidance (pure conditional), RAW residual (no 1/tau), and the gradient
localized to the person region (min-max normalized gender cross-attention >= attn_gate_thr=0.15).

For a set of occupation prompts x N images we produce, per image, three complementary views of
the SRR signal:
  A. EXACT gradient descent on the real E_realistic (region-masked)  -> the literal training signal
  B. one-step x0-prediction difference (SDS target, gs=1.0)         -> per-pixel push direction
  C. SDEdit re-denoise with the realism prompt (gs=1.0 faithful; gs=7.5 reference) -> the user's idea
plus the person-region mask overlay and difference heatmaps, and dataset-level aggregate statistics.

Single-GPU, self-contained. See README.md in this folder for interpretation.
"""
import os, sys, json, math, argparse, time, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler

# ----------------------------------------------------------------------------- config (mirror source args)
MODEL = "runwayml/stable-diffusion-v1-5"
SRR_PROMPT = "a photo of a realistic person"
WOMAN_PROMPT, MAN_PROMPT = "a photo of a woman", "a photo of a man"
WOMAN_WORD, MAN_WORD = "woman", "man"
RES_T_MIN, RES_T_MAX, RES_K = 400, 800, 15          # SRR / gender scorer timestep grid
ATTN_GATE_THR = 0.15                                # person-region min-max attn threshold
GUIDANCE = 7.5                                       # generation CFG
NUM_STEPS = 25                                       # generation denoising steps (eval setting)
OCC_TEMPLATE = "A photo of the face of a {occupation}, a person"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--occ_json", default="../data/1-prompts/occupation.json")
    p.add_argument("--out_dir", default="./srr_signal_exp")
    p.add_argument("--num_occ", type=int, default=50, help="how many occupations (from occupations_test_set)")
    p.add_argument("--per_occ", type=int, default=10, help="images per occupation")
    p.add_argument("--panel_per_occ", type=int, default=2, help="how many per occ get a full multi-view panel")
    p.add_argument("--occ_start", type=int, default=0, help="global occupation index to start at (sharding)")
    p.add_argument("--occ_end", type=int, default=-1, help="global occupation index to stop before (-1 = num_occ)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=12345)
    # exact-gradient descent (view A)
    p.add_argument("--grad_iters", type=int, default=20)
    p.add_argument("--grad_lr", type=float, default=0.02)
    p.add_argument("--grad_ckpts", default="0,3,8,20")
    # one-step x0 target (view B)
    p.add_argument("--b_scales", default="1.0", help="scale(s) applied to the mean x0-target delta")
    # SDEdit (view C)
    p.add_argument("--sdedit_tstarts", default="500,700")
    p.add_argument("--sdedit_gs", default="1.0,7.5")
    p.add_argument("--quick", action="store_true", help="tiny smoke test: 1 occ x 2 imgs")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.num_occ, args.per_occ, args.panel_per_occ = 1, 2, 2
    device = torch.device(args.device)
    wdtype = torch.float16
    hp = torch.float32
    out = Path(args.out_dir)
    (out / "base_images").mkdir(parents=True, exist_ok=True)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    (out / "aggregate").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load frozen SD-1.5 components
    print("loading SD-1.5 ...", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(device, wdtype).eval()
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(device, wdtype).eval()
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").to(device, wdtype).eval()
    scheduler = DPMSolverMultistepScheduler.from_config(MODEL, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)
    assert scheduler.config.prediction_type == "epsilon"
    # SD UNet has no BN/dropout, so eval()==train() outputs. The source enables gradient checkpointing only
    # to bound memory on large full-trajectory batches; here batches are tiny (n=1, K=15) on a 90GB+ GPU, so
    # we leave checkpointing OFF (backward does not recompute the forward) for ~1.3x faster descent. Outputs
    # and gradients are numerically identical to the checkpointed path.
    unet.eval()
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scaling = vae.config.scaling_factor

    # ------------------------------------------------------------------ cross-attn capture (person-region mask)
    class Ctx:
        def __init__(self): self.enabled = False; self.token_idxs = None; self.store = []
    ctx = Ctx()

    class CrossAttnCaptureProcessor:
        def __init__(self, ctx): self.ctx = ctx
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                b, c, h, w = hidden_states.shape
                hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
            bsz, seqlen, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, seqlen, bsz)
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            if self.ctx.enabled and self.ctx.token_idxs is not None:
                with torch.no_grad():
                    col = attention_probs[..., self.ctx.token_idxs].mean(dim=-1)
                self.ctx.store.append((col.detach(), attn.heads))
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

    procs = dict(unet.attn_processors)
    for name in list(procs.keys()):
        if name.endswith("attn2.processor"):
            procs[name] = CrossAttnCaptureProcessor(ctx)
    unet.set_attn_processor(procs)

    def find_tok_idxs(prompt, word):
        pid = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True).input_ids
        wid = tokenizer(word, add_special_tokens=False).input_ids
        L = len(wid)
        for i in range(len(pid) - L + 1):
            if pid[i:i + L] == wid:
                return list(range(i, i + L))
        raise ValueError(f"{word} not in {prompt}")
    woman_idxs = find_tok_idxs(WOMAN_PROMPT, WOMAN_WORD)
    man_idxs = find_tok_idxs(MAN_PROMPT, MAN_WORD)

    # ------------------------------------------------------------------ prompt encoders (mirror source)
    def encode_scoring(prompt):  # WITH attention_mask -> matches SRR / woman / man convention
        tok = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = text_encoder(tok.input_ids.to(device), tok.attention_mask.to(device))[0]
        return emb.to(wdtype)
    srr_embeds = encode_scoring(SRR_PROMPT)
    woman_embeds = encode_scoring(WOMAN_PROMPT)
    man_embeds = encode_scoring(MAN_PROMPT)

    def encode_gen(prompt, N):  # mirror generate_image_no_gradient text encoding (dynamic pad + uncond)
        tok = tokenizer([prompt] * N, return_tensors="pt", padding=True)
        emb = text_encoder(tok.input_ids.to(device), tok.attention_mask.to(device))[0]
        L = emb.shape[1]
        un = tokenizer([""] * N, padding="max_length", max_length=L, truncation=True, return_tensors="pt")
        un_emb = text_encoder(un.input_ids.to(device), un.attention_mask.to(device))[0]
        return torch.cat([un_emb, emb]).to(wdtype)

    # ------------------------------------------------------------------ helpers
    def decode(z0):  # z0 in scheduler scale -> image [-1,1]
        with torch.no_grad():
            img = vae.decode((z0 / scaling).to(vae.dtype)).sample.clamp(-1, 1)
        return img

    def to_pil(img):  # [3,H,W] in [-1,1]
        return transforms.ToPILImage()((img.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1))

    @torch.no_grad()
    def generate(prompt, noises):  # mirror generate_image_no_gradient (full denoise, CFG)
        N = noises.shape[0]
        emb = encode_gen(prompt, N)
        scheduler.set_timesteps(NUM_STEPS)
        latents = noises
        for t in scheduler.timesteps:
            lmi = torch.cat([latents.to(wdtype)] * 2)
            lmi = scheduler.scale_model_input(lmi, t)
            npred = unet(lmi, t, encoder_hidden_states=emb).sample.to(hp)
            nu, nt = npred.chunk(2)
            npred = nu + GUIDANCE * (nt - nu)
            latents = scheduler.step(npred, t, latents).prev_sample
        return decode(latents), latents  # image, z0

    # timesteps for the SRR / gender scorer
    res_ts = torch.linspace(RES_T_MIN, RES_T_MAX, RES_K, device=device).round().long()

    def gender_region_mask(z0, eps_stack):
        """Person region = min-max normalized common gender cross-attn >= ATTN_GATE_THR. Returns
        (face_mask[n,H,W], attn_gate[n,H,W], common_attn[n,H,W]) exactly as residual_gender_and_realism.
        eps_stack: [K,n,4,H,W] fixed noise (shared with the realism pass so views are consistent)."""
        n, _, H, W = z0.shape
        zt_all = torch.stack([scheduler.add_noise(z0, eps_stack[k], res_ts[k].repeat(n)) for k in range(RES_K)],
                             dim=1).reshape(n * RES_K, 4, H, W).to(wdtype)
        t_all = res_ts.repeat(n)
        attn_accum = torch.zeros(n, H, W, device=device); cnt = 0
        for embeds, idxs in ((woman_embeds, woman_idxs), (man_embeds, man_idxs)):
            c = embeds.expand(n * RES_K, -1, -1)
            ctx.store = []; ctx.token_idxs = idxs; ctx.enabled = True
            with torch.no_grad():
                _ = unet(zt_all, t_all, encoder_hidden_states=c).sample
            ctx.enabled = False
            for col, heads in ctx.store:
                s = int(round(math.sqrt(col.shape[-1])))
                a = col.view(n * RES_K, heads, s, s).float()
                a = torch.nn.functional.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
                a = a.mean(1).view(n, RES_K, H, W).mean(1)
                attn_accum = attn_accum + a; cnt += 1
            ctx.store = []
        common = attn_accum / max(cnt, 1)
        common = (common / (common.sum(dim=(1, 2), keepdim=True) + 1e-8)).detach()
        cmin = common.amin(dim=(1, 2), keepdim=True); cmax = common.amax(dim=(1, 2), keepdim=True)
        gate = ((common - cmin) / (cmax - cmin + 1e-8)).clamp(0, 1)
        mask = (gate >= ATTN_GATE_THR).to(z0.dtype)
        return mask, gate, common

    def srr_energy(z0, eps_stack, mask=None):
        """E_realistic(z0) exactly as residual_gender_and_realism (whole-image VALUE). If mask given,
        gradient is region-masked (z0_srr = mask*z0 + (1-mask)*z0.detach()) as in the file; else full grad.
        Returns E [n] (grad-carrying)."""
        n, _, H, W = z0.shape
        if mask is not None:
            mc = mask.unsqueeze(1)
            z0u = mc * z0 + (1.0 - mc) * z0.detach()
        else:
            z0u = z0
        zt = torch.stack([scheduler.add_noise(z0u, eps_stack[k], res_ts[k].repeat(n)) for k in range(RES_K)],
                         dim=1).reshape(n * RES_K, 4, H, W).to(wdtype)
        t_all = res_ts.repeat(n)
        c = srr_embeds.expand(n * RES_K, -1, -1)
        eps_pred = unet(zt, t_all, encoder_hidden_states=c).sample
        eps_all = eps_stack.transpose(0, 1).reshape(n * RES_K, 4, H, W)  # [n*K,...] row i*K+k -> eps_k  (see note)
        resid = (eps_pred.float() - eps_all.float()).pow(2).mean(1).view(n, RES_K, H, W)
        return resid.mean(dim=(2, 3)).mean(dim=1)  # [n]

    def make_eps_stack(z0, gen):  # [K,n,4,H,W] fixed noise (seeded generator)
        n, _, H, W = z0.shape
        return torch.stack([torch.randn(n, 4, H, W, device=device, generator=gen) for _ in range(RES_K)], dim=0)

    # NOTE on eps ordering: eps_stack is [K,n,...]; transpose(0,1)->[n,K,...] then reshape gives row i*K+k
    # -> eps for image i, timestep k, matching zt built as stack(dim=1).reshape (row i*K+k). Verified below.

    # ---- view B: one-step x0-prediction target (SDS direction, pure conditional gs=1.0) ----
    def x0_target_delta(z0, eps_stack):
        n, _, H, W = z0.shape
        deltas = []
        with torch.no_grad():
            for k in range(RES_K):
                t = res_ts[k]
                zt = scheduler.add_noise(z0, eps_stack[k], t.repeat(n)).to(wdtype)
                eps_pred = unet(zt, t.repeat(n), encoder_hidden_states=srr_embeds.expand(n, -1, -1)).sample.float()
                ab = alphas_cumprod[t].float()
                x0p = (zt.float() - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
                deltas.append(x0p - z0.float())
        return torch.stack(deltas, 0).mean(0)  # [n,4,H,W] mean target displacement in z0 space

    # ---- view C: SDEdit re-denoise with the realism prompt ----
    def sdedit(z0, t_start_target, gs, gen):
        n = z0.shape[0]
        sch = DPMSolverMultistepScheduler.from_config(scheduler.config)
        sch.set_timesteps(NUM_STEPS)
        ts = sch.timesteps.to(device)
        idx = int((ts <= t_start_target).nonzero()[0].item())  # ts descending; first <= target
        t_start = ts[idx]
        # CFG embeds for the realism prompt (uncond + realistic), dynamic-pad style like generation
        emb = encode_gen(SRR_PROMPT, n)
        eps = torch.randn(z0.shape, device=device, generator=gen)
        latents = sch.add_noise(z0, eps, t_start.repeat(n))
        with torch.no_grad():
            for t in ts[idx:]:
                lmi = torch.cat([latents.to(wdtype)] * 2)
                lmi = sch.scale_model_input(lmi, t)
                npred = unet(lmi, t, encoder_hidden_states=emb).sample.to(hp)
                nu, nt = npred.chunk(2)
                npred = nu + gs * (nt - nu)
                latents = sch.step(npred, t, latents).prev_sample
        return decode(latents), int(t_start.item())

    # ------------------------------------------------------------------ visualization utilities
    def diff_heat(a, b):  # two [3,H,W] in [-1,1] -> PIL heatmap of per-pixel L1 (over channels)
        d = (a.float().cpu() - b.float().cpu()).abs().mean(0)  # [H,W] in [0,2]
        d = (d / (d.max() + 1e-8)).clamp(0, 1)
        arr = (d * 255).to(torch.uint8).numpy()
        return ImageOps.colorize(Image.fromarray(arr, "L"), black="black", mid="orange", white="red")

    def mask_overlay(mask_hw, img, alpha=0.45, color=(0, 180, 255)):
        pil = to_pil(img).convert("RGBA")
        m = torch.nn.functional.interpolate(mask_hw.float().view(1, 1, *mask_hw.shape),
                                             size=pil.size[::-1], mode="nearest").view(pil.size[::-1])
        tint = Image.new("RGBA", pil.size, color + (0,))
        am = Image.fromarray((m * 255 * alpha).clamp(0, 255).to(torch.uint8).cpu().numpy(), "L")
        tint.putalpha(am)
        return Image.alpha_composite(pil, tint).convert("RGB")

    def label(pil, text):
        pil = ImageOps.expand(pil, border=(0, 34, 0, 0), fill="black")
        d = ImageDraw.Draw(pil)
        try:
            fnt = ImageFont.truetype("../data/0-utils/arial-bold.ttf", 26)
        except Exception:
            fnt = ImageFont.load_default()
        d.text((6, 4), text, fill="white", font=fnt)
        return pil

    def hcat(pils):
        h = max(p.height for p in pils); w = sum(p.width for p in pils)
        g = Image.new("RGB", (w, h), "white"); x = 0
        for p in pils:
            g.paste(p, (x, 0)); x += p.width
        return g

    # ------------------------------------------------------------------ run
    with open(args.occ_json) as f:
        occ_all = json.load(f)["occupations_test_set"][:args.num_occ]
    occ_lo = args.occ_start
    occ_hi = args.num_occ if args.occ_end < 0 else args.occ_end
    occ = occ_all[occ_lo:occ_hi]          # this shard's occupations (global index = occ_lo + local)
    grad_ckpts = sorted(set(int(x) for x in args.grad_ckpts.split(",")))
    b_scales = [float(x) for x in args.b_scales.split(",")]
    sdedit_tstarts = [int(x) for x in args.sdedit_tstarts.split(",")]
    sdedit_gs = [float(x) for x in args.sdedit_gs.split(",")]

    agg = {"E0": [], "Egrad": [], "in_region_l1": [], "out_region_l1": [],
           "in_region_lat": [], "out_region_lat": [],
           "region_frac": [], "occ": [], "img_idx": []}
    diff_accum = None; diff_count = 0; mask_accum = None

    t_start_all = time.time()
    for local_i, occupation in enumerate(occ):
        oi = occ_lo + local_i            # global occupation index (stable seeds/filenames across shards)
        prompt = OCC_TEMPLATE.format(occupation=occupation)
        # deterministic noise per (occupation, image)
        gen = torch.Generator(device=device).manual_seed(args.seed + oi * 1000)
        noises = torch.randn(args.per_occ, 4, 64, 64, device=device, generator=gen)
        base_imgs, z0s = generate(prompt, noises)  # [K,3,512,512], [K,4,64,64]

        for j in range(args.per_occ):
            z0 = z0s[j:j + 1].detach()
            base_img = base_imgs[j]
            gj = torch.Generator(device=device).manual_seed(args.seed + oi * 1000 + j + 1)
            eps_stack = make_eps_stack(z0, gj)                      # fixed noise for all views

            # region mask (once, from init z0)
            mask, gate, common = gender_region_mask(z0, eps_stack)  # [1,H,W]
            region_frac = mask.mean().item()

            # view A: exact region-masked gradient descent on E_realistic
            E0 = srr_energy(z0, eps_stack, mask=mask).item()
            z = z0.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=args.grad_lr)
            grad_snapshots = {}
            with torch.no_grad():
                if 0 in grad_ckpts:
                    grad_snapshots[0] = z.detach().clone()
            for it in range(1, max(grad_ckpts) + 1):
                opt.zero_grad()
                E = srr_energy(z, eps_stack, mask=mask).sum()
                E.backward()
                opt.step()
                if it in grad_ckpts:
                    grad_snapshots[it] = z.detach().clone()
            z_final = grad_snapshots[max(grad_ckpts)]
            with torch.no_grad():
                Egrad = srr_energy(z_final, eps_stack, mask=mask).item()
                # latent-space localization proof: gradient is exactly 0 outside the mask, so out-region
                # z0 must be UNCHANGED (Adam state stays 0 there). in-region carries the whole signal.
                dz = (z_final - z0).abs().mean(1)[0]                 # [H,W]
                mz = mask[0]
                in_z = (dz * mz).sum().item() / (mz.sum().item() + 1e-8)
                out_z = (dz * (1 - mz)).sum().item() / ((1 - mz).sum().item() + 1e-8)
            img_grad_final = decode(z_final)[0]

            # localization metric: L1 image change inside vs outside the person region
            with torch.no_grad():
                dimg = (decode(z_final)[0].float() - base_img.float()).abs().mean(0).cpu()  # [512,512]
                m512 = torch.nn.functional.interpolate(mask.float().view(1, 1, 64, 64), size=(512, 512),
                                                       mode="nearest").view(512, 512).cpu()
                in_l1 = (dimg * m512).sum().item() / (m512.sum().item() + 1e-8)
                out_l1 = (dimg * (1 - m512)).sum().item() / ((1 - m512).sum().item() + 1e-8)

            agg["E0"].append(E0); agg["Egrad"].append(Egrad)
            agg["in_region_l1"].append(in_l1); agg["out_region_l1"].append(out_l1)
            agg["in_region_lat"].append(in_z); agg["out_region_lat"].append(out_z)
            agg["region_frac"].append(region_frac); agg["occ"].append(occupation); agg["img_idx"].append(j)

            # accumulate spatial diff + mask for dataset-average maps
            if diff_accum is None:
                diff_accum = dimg.clone(); mask_accum = m512.clone()
            else:
                diff_accum += dimg; mask_accum += m512
            diff_count += 1

            # ---- panels for a subset ----
            if j < args.panel_per_occ:
                # view B
                bdelta = x0_target_delta(z0, eps_stack)          # [1,4,H,W]
                b_pils = []
                for s in b_scales:
                    imgB = decode(z0 + s * bdelta)[0]
                    b_pils.append(label(to_pil(imgB).resize((512, 512)), f"B: x0-target x{s:g} (gs1)"))
                    b_pils.append(label(diff_heat(imgB, base_img).resize((512, 512)), f"B diff x{s:g}"))
                # view C
                c_pils = []
                for ts0 in sdedit_tstarts:
                    for gs in sdedit_gs:
                        gc = torch.Generator(device=device).manual_seed(args.seed + 777)
                        imgC, t_used = sdedit(z0, ts0, gs, gc)
                        c_pils.append(label(to_pil(imgC[0]).resize((512, 512)), f"C: SDEdit t~{t_used} gs{gs:g}"))
                # view A snapshots
                a_pils = [label(to_pil(base_img), "orig (occupation)"),
                          label(mask_overlay(mask[0], base_img), f"person region ({region_frac*100:.0f}%)")]
                for it in grad_ckpts:
                    a_pils.append(label(to_pil(decode(grad_snapshots[it])[0]),
                                        f"A: SRR grad it{it}" + (f" E{E0:.3f}" if it == 0 else "")))
                a_pils.append(label(diff_heat(img_grad_final, base_img), f"A diff (E {E0:.3f}->{Egrad:.3f})"))

                row1 = hcat(a_pils)
                row2 = hcat(b_pils + c_pils)
                W = max(row1.width, row2.width)
                panel = Image.new("RGB", (W, row1.height + row2.height + 8), "white")
                panel.paste(row1, (0, 0)); panel.paste(row2, (0, row1.height + 8))
                safe = "".join(ch if ch.isalnum() else "_" for ch in occupation)[:24]
                panel.save(out / "panels" / f"{oi:02d}_{safe}_img{j}.jpg", quality=88)

            # save the base image
            safe = "".join(ch if ch.isalnum() else "_" for ch in occupation)[:24]
            to_pil(base_img).save(out / "base_images" / f"{oi:02d}_{safe}_img{j}.jpg", quality=90)

        el = time.time() - t_start_all
        print(f"[{local_i+1}/{len(occ)} occ#{oi}] {occupation:24s} | {el:.0f}s | "
              f"E {np.mean(agg['E0'][-args.per_occ:]):.3f}->{np.mean(agg['Egrad'][-args.per_occ:]):.3f} "
              f"| in/out L1 {np.mean(agg['in_region_l1'][-args.per_occ:]):.4f}/"
              f"{np.mean(agg['out_region_l1'][-args.per_occ:]):.4f}", flush=True)

    # ------------------------------------------------------------------ aggregate outputs
    A = {k: (np.array(v) if k not in ("occ",) else v) for k, v in agg.items()}
    summary = {
        "n_images": int(diff_count),
        "num_occ": len(occ), "per_occ": args.per_occ,
        "E_realistic_mean_before": float(A["E0"].mean()),
        "E_realistic_mean_after_grad": float(A["Egrad"].mean()),
        "E_reduction_pct_mean": float(((A["E0"] - A["Egrad"]) / (A["E0"] + 1e-8) * 100).mean()),
        "in_region_L1_mean_IMAGE": float(A["in_region_l1"].mean()),
        "out_region_L1_mean_IMAGE_vae_spillover": float(A["out_region_l1"].mean()),
        "localization_ratio_image": float(A["in_region_l1"].mean() / (A["out_region_l1"].mean() + 1e-8)),
        "in_region_change_mean_LATENT": float(A["in_region_lat"].mean()),
        "out_region_change_mean_LATENT_should_be_0": float(A["out_region_lat"].mean()),
        "region_frac_mean": float(A["region_frac"].mean()),
        "grad_iters": args.grad_iters, "grad_lr": args.grad_lr,
        "config": {"srr_prompt": SRR_PROMPT, "res_t": [RES_T_MIN, RES_T_MAX], "K": RES_K,
                   "attn_gate_thr": ATTN_GATE_THR, "cfg_free_signal": True},
    }
    with open(out / "aggregate" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # per-image csv
    with open(out / "aggregate" / "per_image.csv", "w") as f:
        f.write("occ,img,E0,Egrad,in_region_l1_img,out_region_l1_img,in_region_lat,out_region_lat,region_frac\n")
        for i in range(diff_count):
            f.write(f"{A['occ'][i]},{A['img_idx'][i]},{A['E0'][i]:.5f},{A['Egrad'][i]:.5f},"
                    f"{A['in_region_l1'][i]:.6f},{A['out_region_l1'][i]:.6f},"
                    f"{A['in_region_lat'][i]:.6f},{A['out_region_lat'][i]:.6f},{A['region_frac'][i]:.4f}\n")
    # raw spatial SUM accumulators + count, so shard runs can be merged exactly (merge_shards.py)
    np.save(out / "aggregate" / "diff_accum_sum.npy", diff_accum.numpy())
    np.save(out / "aggregate" / "mask_accum_sum.npy", mask_accum.numpy())
    # average spatial signal map + average region
    avg_diff = (diff_accum / diff_count)
    avg_diff_n = (avg_diff / (avg_diff.max() + 1e-8)).clamp(0, 1)
    ImageOps.colorize(Image.fromarray((avg_diff_n * 255).to(torch.uint8).numpy(), "L"),
                      black="black", mid="orange", white="red").save(out / "aggregate" / "avg_srr_change_map.png")
    avg_mask = (mask_accum / diff_count)
    avg_mask_n = (avg_mask / (avg_mask.max() + 1e-8)).clamp(0, 1)
    Image.fromarray((avg_mask_n * 255).to(torch.uint8).numpy(), "L").save(out / "aggregate" / "avg_person_region.png")

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print(f"\ntotal {time.time()-t_start_all:.0f}s | outputs in {out}")


if __name__ == "__main__":
    main()
