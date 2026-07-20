#!/usr/bin/env python
# coding=utf-8
"""
Actually TRAIN a text-encoder LoRA on the SRR realism loss ALONE (the model-level analog of View A's
latent-space descent) and produce before/during/after comparisons.

Faithful to 1-main-...noflip.py's mechanism:
  - trainable CLIP text-encoder LoRA (LoraLoaderMixin._modify_text_encoder, rank 50, AdamW lr 5e-5) -- exactly
    the user's "text encoder LoRA" setup. UNet/VAE frozen.
  - on-policy generation each step with gradient (generate_image_w_gradient: CFG 7.5, SDS grad-coef hooks,
    optional truncated denoising) using the trainable TE.
  - loss = weight_loss_face * E_realistic, where E_realistic is the SAME frozen-scorer SRR term as the source
    (residual_gender_and_realism), region-masked gradient (min-max gender-attn >= 0.15). The scorer uses a
    FROZEN copy of the TE (eval_text_encoder), never the trainable one -- exactly as the source.
  - NO fair / NO SCR loss: SRR is isolated so any change in generations is attributable to SRR only.

Outputs (under --out_dir): eval grids at each snapshot step, before/after compare panels, loss curve, log.
Single GPU, self-contained.
"""
import os, sys, json, math, argparse, time, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.loaders import LoraLoaderMixin

MODEL = "runwayml/stable-diffusion-v1-5"
SRR_PROMPT = "a photo of a realistic person"
WOMAN_PROMPT, MAN_PROMPT = "a photo of a woman", "a photo of a man"
WOMAN_WORD, MAN_WORD = "woman", "man"
RES_T_MIN, RES_T_MAX, RES_K = 400, 800, 15
ATTN_GATE_THR = 0.15
GUIDANCE = 7.5
OCC_TEMPLATE = "A photo of the face of a {occupation}, a person"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--occ_json", default="../data/1-prompts/occupation.json")
    p.add_argument("--out_dir", default="./srr_signal_exp/train_out")
    p.add_argument("--num_occ", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--rank", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight", type=float, default=1.0, help="weight_loss_face on E_realistic")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--bs", type=int, default=4, help="images per step (train_GPU_batch_size)")
    p.add_argument("--gen_steps", type=int, default=20, help="denoising steps for training generation")
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--skip_denoise_frac", type=float, default=0.0)
    p.add_argument("--snapshots", default="0,50,150,300", help="steps at which to eval + save grids")
    p.add_argument("--eval_occ", default="senator,cosmetologist,roustabout,violinist,butcher,interior designer",
                   help="occupations to eval (fixed seeds)")
    p.add_argument("--eval_per_occ", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.steps = 8; args.snapshots = "0,8"; args.num_occ = 6
        args.eval_occ = "senator,cosmetologist"; args.eval_per_occ = 1
    device = torch.device(args.device)
    wdtype = torch.float16
    out = Path(args.out_dir)
    (out / "eval_grids").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    snapshots = sorted(set(int(x) for x in args.snapshots.split(",")))
    logf = open(out / "train.log", "w")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    # ------------------------------------------------------------------ models
    log("loading SD-1.5 ...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(device, wdtype).eval().requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").to(device, wdtype).eval().requires_grad_(False)
    scheduler = DPMSolverMultistepScheduler.from_config(MODEL, subfolder="scheduler")
    assert scheduler.config.prediction_type == "epsilon"
    unet.enable_gradient_checkpointing(); unet.train()          # for multi-step generation backprop
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scaling = vae.config.scaling_factor

    # frozen TE copy = the SRR scorer's text encoder (fixed woman/man/realistic embeds), fp16
    te_frozen = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(device, wdtype).eval().requires_grad_(False)
    # trainable TE (fp32) with LoRA = the model we finetune; used ONLY for generation
    te_train = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(device, torch.float32)
    te_train.requires_grad_(False)
    lora_params = LoraLoaderMixin._modify_text_encoder(te_train, dtype=torch.float32, rank=args.rank, patch_mlp=True)
    te_train.train()
    n_lora = sum(p.numel() for p in lora_params)
    log(f"TE-LoRA params: {n_lora} across {len(lora_params)} tensors | lr={args.lr} weight={args.weight} steps={args.steps}")
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    # ------------------------------------------------------------------ cross-attn capture on unet (person-region mask)
    class Ctx:
        def __init__(self): self.enabled = False; self.token_idxs = None; self.store = []
    ctx = Ctx()
    class Cap:
        def __init__(self, ctx): self.ctx = ctx
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
            residual = hidden_states
            if attn.spatial_norm is not None: hidden_states = attn.spatial_norm(hidden_states, temb)
            ndim = hidden_states.ndim
            if ndim == 4:
                b, c, h, w = hidden_states.shape
                hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
            bsz, seq, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, seq, bsz)
            if attn.group_norm is not None: hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            q = attn.to_q(hidden_states)
            enc = hidden_states if encoder_hidden_states is None else (
                attn.norm_encoder_hidden_states(encoder_hidden_states) if attn.norm_cross else encoder_hidden_states)
            k = attn.to_k(enc); v = attn.to_v(enc)
            q = attn.head_to_batch_dim(q); k = attn.head_to_batch_dim(k); v = attn.head_to_batch_dim(v)
            probs = attn.get_attention_scores(q, k, attention_mask)
            if self.ctx.enabled and self.ctx.token_idxs is not None:
                with torch.no_grad():
                    col = probs[..., self.ctx.token_idxs].mean(dim=-1)
                self.ctx.store.append((col.detach(), attn.heads))
            hs = torch.bmm(probs, v); hs = attn.batch_to_head_dim(hs)
            hs = attn.to_out[0](hs); hs = attn.to_out[1](hs)
            if ndim == 4: hs = hs.transpose(-1, -2).reshape(b, c, h, w)
            if attn.residual_connection: hs = hs + residual
            return hs / attn.rescale_output_factor
    procs = dict(unet.attn_processors)
    for name in list(procs.keys()):
        if name.endswith("attn2.processor"): procs[name] = Cap(ctx)
    unet.set_attn_processor(procs)

    def find_tok_idxs(prompt, word):
        pid = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True).input_ids
        wid = tokenizer(word, add_special_tokens=False).input_ids
        for i in range(len(pid) - len(wid) + 1):
            if pid[i:i + len(wid)] == wid: return list(range(i, i + len(wid)))
        raise ValueError(word)
    woman_idxs, man_idxs = find_tok_idxs(WOMAN_PROMPT, WOMAN_WORD), find_tok_idxs(MAN_PROMPT, MAN_WORD)

    def encode_scoring(prompt):   # frozen TE, WITH attention mask
        tok = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = te_frozen(tok.input_ids.to(device), tok.attention_mask.to(device))[0]
        return emb.to(wdtype)
    srr_embeds, woman_embeds, man_embeds = encode_scoring(SRR_PROMPT), encode_scoring(WOMAN_PROMPT), encode_scoring(MAN_PROMPT)

    def encode_gen(prompt, N, te):  # trainable te for generation (grad); uncond+cond
        tok = tokenizer([prompt] * N, return_tensors="pt", padding=True)
        emb = te(tok.input_ids.to(device), tok.attention_mask.to(device))[0]
        L = emb.shape[1]
        un = tokenizer([""] * N, padding="max_length", max_length=L, truncation=True, return_tensors="pt")
        un_emb = te(un.input_ids.to(device), un.attention_mask.to(device))[0]
        return torch.cat([un_emb, emb])

    def decode(z0):
        with torch.no_grad():
            return vae.decode((z0 / scaling).to(vae.dtype)).sample.clamp(-1, 1)
    def to_pil(img):
        return transforms.ToPILImage()((img.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1))

    res_ts = torch.linspace(RES_T_MIN, RES_T_MAX, RES_K, device=device).round().long()

    def make_eps_stack(z0, gen=None):
        n, _, H, W = z0.shape
        return torch.stack([torch.randn(n, 4, H, W, device=device, generator=gen) for _ in range(RES_K)], dim=0)

    def gender_region_mask(z0, eps_stack):
        n, _, H, W = z0.shape
        zt = torch.stack([scheduler.add_noise(z0, eps_stack[k], res_ts[k].repeat(n)) for k in range(RES_K)],
                         dim=1).reshape(n * RES_K, 4, H, W).to(wdtype)
        t_all = res_ts.repeat(n); accum = torch.zeros(n, H, W, device=device); cnt = 0
        for emb, idxs in ((woman_embeds, woman_idxs), (man_embeds, man_idxs)):
            c = emb.expand(n * RES_K, -1, -1)
            ctx.store = []; ctx.token_idxs = idxs; ctx.enabled = True
            with torch.no_grad(): _ = unet(zt, t_all, encoder_hidden_states=c).sample
            ctx.enabled = False
            for col, heads in ctx.store:
                s = int(round(math.sqrt(col.shape[-1])))
                a = col.view(n * RES_K, heads, s, s).float()
                a = torch.nn.functional.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
                accum = accum + a.mean(1).view(n, RES_K, H, W).mean(1); cnt += 1
            ctx.store = []
        common = accum / max(cnt, 1)
        common = (common / (common.sum(dim=(1, 2), keepdim=True) + 1e-8)).detach()
        cmin, cmax = common.amin(dim=(1, 2), keepdim=True), common.amax(dim=(1, 2), keepdim=True)
        gate = ((common - cmin) / (cmax - cmin + 1e-8)).clamp(0, 1)
        return (gate >= ATTN_GATE_THR).to(z0.dtype)

    def srr_energy(z0, eps_stack, mask):
        n, _, H, W = z0.shape
        mc = mask.unsqueeze(1)
        z0u = mc * z0 + (1.0 - mc) * z0.detach()
        zt = torch.stack([scheduler.add_noise(z0u, eps_stack[k], res_ts[k].repeat(n)) for k in range(RES_K)],
                         dim=1).reshape(n * RES_K, 4, H, W).to(wdtype)
        t_all = res_ts.repeat(n)
        c = srr_embeds.expand(n * RES_K, -1, -1)
        eps_pred = unet(zt, t_all, encoder_hidden_states=c).sample
        eps_all = eps_stack.transpose(0, 1).reshape(n * RES_K, 4, H, W)
        resid = (eps_pred.float() - eps_all.float()).pow(2).mean(1).view(n, RES_K, H, W)
        return resid.mean(dim=(2, 3)).mean(dim=1)

    def gen_no_grad(prompt, noises, nsteps, te):
        N = noises.shape[0]
        emb = encode_gen(prompt, N, te).to(wdtype)
        scheduler.set_timesteps(nsteps)
        latents = noises
        with torch.no_grad():
            for t in scheduler.timesteps:
                lmi = scheduler.scale_model_input(torch.cat([latents.to(wdtype)] * 2), t)
                npred = unet(lmi, t, encoder_hidden_states=emb).sample.float()
                nu, nt = npred.chunk(2); npred = nu + GUIDANCE * (nt - nu)
                latents = scheduler.step(npred, t, latents).prev_sample
        return decode(latents), latents  # image, z0

    def view_a_descent(z0, iters=20, lr=0.02):
        """View A analog: descend E_realistic directly on the base latent (frozen model), region-masked."""
        eps_stack = make_eps_stack(z0, torch.Generator(device=device).manual_seed(4242))
        with torch.no_grad():
            mask = gender_region_mask(z0, eps_stack)
        z = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(iters):
            opt.zero_grad(); srr_energy(z, eps_stack, mask).sum().backward(); opt.step()
        return decode(z.detach())[0].cpu()

    def gen_w_grad(prompt, noises, nsteps):
        """mirror generate_image_w_gradient: trainable TE, CFG, SDS grad-coef hooks, optional truncation."""
        N = noises.shape[0]
        emb = encode_gen(prompt, N, te_train).to(wdtype)     # grad -> TE-LoRA
        scheduler.set_timesteps(nsteps)
        skip = float(args.skip_denoise_frac)
        n_total = len(scheduler.timesteps)
        n_run = max(1, int(round(n_total * (1.0 - skip)))) if skip > 0 else n_total
        gc = []
        for t in scheduler.timesteps[:n_run]:
            gc.append(alphas_cumprod[t].sqrt().item() * (1 - alphas_cumprod[t]).sqrt().item()
                      / (1 - scheduler.alphas[t].item()))
        gc = np.array(gc); gc /= (math.prod(gc) ** (1 / len(gc)))
        latents = noises
        for i, t in enumerate(scheduler.timesteps):
            lmi = scheduler.scale_model_input(torch.cat([latents.detach().to(wdtype)] * 2), t)
            npred = unet(lmi, t, encoder_hidden_states=emb).sample.float()
            nu, nt = npred.chunk(2); npred = nu + GUIDANCE * (nt - nu)
            npred.register_hook(lambda g, c=gc[i]: c * g)
            if skip > 0 and i == n_run - 1:
                ab = alphas_cumprod[t].to(latents.device, npred.dtype)
                latents = (latents - (1 - ab).sqrt() * npred) / ab.sqrt(); break
            latents = scheduler.step(npred, t, latents).prev_sample
        return latents  # z0 (grad-carrying)

    # ------------------------------------------------------------------ small viz helpers
    def label(pil, text):
        pil = ImageOps.expand(pil, border=(0, 30, 0, 0), fill="black"); d = ImageDraw.Draw(pil)
        try: fnt = ImageFont.truetype("../data/0-utils/arial-bold.ttf", 22)
        except Exception: fnt = ImageFont.load_default()
        d.text((5, 4), text, fill="white", font=fnt); return pil
    def hcat(pils):
        h = max(p.height for p in pils); g = Image.new("RGB", (sum(p.width for p in pils), h), "white"); x = 0
        for p in pils: g.paste(p, (x, 0)); x += p.width
        return g
    def vcat(pils):
        w = max(p.width for p in pils); g = Image.new("RGB", (w, sum(p.height for p in pils)), "white"); y = 0
        for p in pils: g.paste(p, (0, y)); y += p.height
        return g
    def diff_heat(a, b):
        d = (a.float().cpu() - b.float().cpu()).abs().mean(0); d = (d / (d.max() + 1e-8)).clamp(0, 1)
        return ImageOps.colorize(Image.fromarray((d * 255).to(torch.uint8).numpy(), "L"), black="black", mid="orange", white="red")

    # fixed eval set (occ, seed) -> noise; store per-snapshot decoded images
    eval_occ = [o.strip() for o in args.eval_occ.split(",")]
    eval_items = []
    for oi, occ in enumerate(eval_occ):
        for j in range(args.eval_per_occ):
            g = torch.Generator(device=device).manual_seed(90000 + oi * 100 + j)
            eval_items.append((occ, torch.randn(1, 4, 64, 64, device=device, generator=g)))
    eval_snapshots = {}   # step -> list of decoded [3,512,512] tensors aligned with eval_items

    def run_eval(step):
        imgs = []
        for occ, noise in eval_items:
            img_b, _ = gen_no_grad(OCC_TEMPLATE.format(occupation=occ), noise, args.eval_steps, te_train)
            imgs.append(img_b[0].cpu())
        eval_snapshots[step] = imgs
        # save a labeled grid
        pncol = args.eval_per_occ
        rows = []
        for r in range(len(eval_occ)):
            cells = [label(to_pil(imgs[r * pncol + c]), f"{eval_occ[r][:16]} s{c}") for c in range(pncol)] if False else \
                    [label(to_pil(imgs[r * args.eval_per_occ + c]), f"{eval_occ[r][:16]} s{c}") for c in range(args.eval_per_occ)]
            rows.append(hcat(cells))
        vcat(rows).save(out / "eval_grids" / f"eval_step{step:04d}.jpg", quality=88)

    # ------------------------------------------------------------------ TRAIN
    occ_all = json.load(open(args.occ_json))["occupations_test_set"][:args.num_occ]
    curve = []   # (step, loss_SRR_mean)
    t0 = time.time()
    if 0 in snapshots:
        run_eval(0); log(f"[eval] step 0 done ({time.time()-t0:.0f}s)")
    # capture base latents (LoRA is zero-init => this is the base model) for the View-A comparison
    base_z0 = []
    for occ, noise in eval_items:
        _, z0b = gen_no_grad(OCC_TEMPLATE.format(occupation=occ), noise, args.eval_steps, te_train)
        base_z0.append(z0b.detach())
    torch.manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        occ = occ_all[(step - 1) % len(occ_all)]
        prompt = OCC_TEMPLATE.format(occupation=occ)
        g = torch.Generator(device=device).manual_seed(args.seed + step)
        noises = torch.randn(args.bs, 4, 64, 64, device=device, generator=g)
        z0 = gen_w_grad(prompt, noises, args.gen_steps)          # grad -> TE-LoRA
        eps_stack = make_eps_stack(z0)                            # fresh SRR noise each step (stochastic, like training)
        with torch.no_grad():
            mask = gender_region_mask(z0.detach(), eps_stack)
        E = srr_energy(z0, eps_stack, mask)
        loss = args.weight * E.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 100.0)
        optimizer.step()
        curve.append((step, float(E.mean().item())))
        if step % 10 == 0 or step == 1:
            recent = np.mean([c[1] for c in curve[-10:]])
            log(f"step {step:4d}/{args.steps} | occ={occ[:18]:18s} | E_SRR={E.mean().item():.4f} "
                f"(avg10 {recent:.4f}) | {time.time()-t0:.0f}s")
        if step in snapshots:
            run_eval(step); log(f"[eval] step {step} done ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------ comparisons
    first, last = snapshots[0], snapshots[-1]
    # before/after compare panels per eval item
    for i, (occ, _) in enumerate(eval_items):
        cells = []
        for s in snapshots:
            cells.append(label(to_pil(eval_snapshots[s][i]), f"step{s}"))
        cells.append(label(diff_heat(eval_snapshots[last][i], eval_snapshots[first][i]), f"diff {first}->{last}"))
        hcat(cells).save(out / "compare" / f"{i:02d}_{occ.replace(' ','_')[:20]}.jpg", quality=90)
    # one big before/after grid (first vs last)
    ba = []
    for i, (occ, _) in enumerate(eval_items):
        ba.append(hcat([label(to_pil(eval_snapshots[first][i]), f"{occ[:14]} base"),
                        label(to_pil(eval_snapshots[last][i]), f"trained s{last}"),
                        label(diff_heat(eval_snapshots[last][i], eval_snapshots[first][i]), "diff")]))
    vcat(ba).save(out / "before_after.jpg", quality=88)

    # View A (latent descent, frozen model) vs actual TE-LoRA training, SAME seed/prompt
    va = []
    for i, (occ, _) in enumerate(eval_items):
        va_img = view_a_descent(base_z0[i])
        va.append(hcat([label(to_pil(eval_snapshots[first][i]), f"{occ[:14]} base"),
                        label(to_pil(va_img), "View A: latent descent"),
                        label(to_pil(eval_snapshots[last][i]), f"trained TE-LoRA s{last}")]))
    vcat(va).save(out / "viewA_vs_training.jpg", quality=88)

    # loss curve
    with open(out / "loss_curve.csv", "w") as f:
        f.write("step,E_SRR\n")
        for s, e in curve: f.write(f"{s},{e:.6f}\n")
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [c[0] for c in curve]; ys = [c[1] for c in curve]
        w = max(1, min(10, len(ys)))
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, alpha=0.3, label="E_SRR (per step)")
        if len(ys) >= w:
            ma = np.convolve(ys, np.ones(w) / w, mode="valid")
            plt.plot(xs[w - 1:], ma, label=f"moving avg ({w})")
        plt.xlabel("training step"); plt.ylabel("E_realistic (SRR loss)"); plt.legend(); plt.grid(alpha=0.3)
        plt.title(f"SRR-only TE-LoRA training (lr={args.lr}, weight={args.weight})")
        plt.tight_layout(); plt.savefig(out / "loss_curve.png", dpi=110)
    except Exception as e:
        log(f"(matplotlib unavailable: {e})")

    summary = {
        "steps": args.steps, "lr": args.lr, "weight": args.weight, "rank": args.rank, "bs": args.bs,
        "gen_steps": args.gen_steps, "skip_denoise_frac": args.skip_denoise_frac,
        "E_SRR_first10_mean": float(np.mean([c[1] for c in curve[:10]])) if curve else None,
        "E_SRR_last10_mean": float(np.mean([c[1] for c in curve[-10:]])) if curve else None,
        "snapshots": snapshots, "n_eval_items": len(eval_items),
    }
    if curve:
        summary["E_SRR_reduction_pct"] = (1 - summary["E_SRR_last10_mean"] / (summary["E_SRR_first10_mean"] + 1e-9)) * 100
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    log("==== SUMMARY ===="); log(json.dumps(summary, indent=2))
    log(f"total {time.time()-t0:.0f}s -> {out}")
    logf.close()


if __name__ == "__main__":
    main()
