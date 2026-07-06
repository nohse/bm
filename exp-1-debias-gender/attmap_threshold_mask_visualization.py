#!/usr/bin/env python
# coding=utf-8
"""
Visualize SDS scoring-time attention masks for face-gradient-style gating.

This follows the attention extraction used by chekc_SCRclip_face_grad.py:
for generated images, re-noise them at SDS timesteps, run the UNet with
"a photo of a woman" / "a photo of a man", capture woman/man token cross
attention over all cross-attention layers, average over timesteps/eps and
woman/man, then visualize hard masks made from that map.
"""

import argparse
import json
import math
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class CrossAttnCapture:
    def __init__(self, token_indices: List[int], use_cpu: bool = False, expect_cfg_pair: bool = False):
        self.token_indices = token_indices
        self.handles = []
        self.maps = []
        self.use_cpu = use_cpu
        self._q_cache = {}
        self._parent_map = {}
        self.expect_cfg_pair = expect_cfg_pair

    def _q_hook(self, module, inputs, output):
        parent = self._parent_map.get(id(module))
        if parent is None:
            return
        try:
            with torch.no_grad():
                self._q_cache[id(parent)] = output.detach().clone()
        except Exception:
            self._q_cache[id(parent)] = output.detach()

    def _k_hook(self, module, inputs, output):
        parent = self._parent_map.get(id(module))
        if parent is None:
            return
        parent_id = id(parent)
        if parent_id not in self._q_cache:
            return

        q = self._q_cache.pop(parent_id)
        k = output
        device = q.device if not self.use_cpu else torch.device("cpu")
        with torch.no_grad():
            q = q.detach().float().to(device)
            k = k.detach().float().to(device)
            heads = getattr(parent, "heads", None) or getattr(parent, "num_heads", None)
            bsz, nq, inner_q = q.shape
            _, nk, inner_k = k.shape
            if inner_q != inner_k:
                return
            if heads is None:
                for candidate in (8, 12, 16, 4, 6, 24, 32):
                    if inner_q % candidate == 0:
                        heads = candidate
                        break
                heads = heads or 8
            head_dim = inner_q // heads
            q = q.view(bsz, nq, heads, head_dim).permute(0, 2, 1, 3).contiguous()
            k = k.view(bsz, nk, heads, head_dim).permute(0, 2, 1, 3).contiguous()
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
            attn = attn_scores.softmax(dim=-1)
            if not self.token_indices:
                return

            tok_idx = torch.tensor(self.token_indices, device=device, dtype=torch.long)
            attn_tok = attn.index_select(-1, tok_idx).mean(dim=-1)
            if self.expect_cfg_pair and bsz >= 2:
                b_half = bsz // 2
                attn_tok = attn_tok[bsz - b_half:bsz]
            per_image = attn_tok.mean(dim=1)
            hw = int(math.sqrt(per_image.shape[1]))
            if hw * hw != per_image.shape[1]:
                return
            per_image = per_image.view(per_image.shape[0], hw, hw)
            if self.use_cpu:
                per_image = per_image.cpu()
            self.maps.append(per_image)

    def add_hooks(self, unet: torch.nn.Module):
        for name, module in unet.named_modules():
            if not (hasattr(module, "to_q") and hasattr(module, "to_k")):
                continue
            is_cross = getattr(module, "is_cross_attention", None)
            if is_cross is None:
                is_cross = ("attn2" in name) or ("Cross" in module.__class__.__name__)
            if not is_cross:
                continue
            self._parent_map[id(module.to_q)] = module
            self._parent_map[id(module.to_k)] = module
            self.handles.append(module.to_q.register_forward_hook(self._q_hook))
            self.handles.append(module.to_k.register_forward_hook(self._k_hook))
        return self

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._q_cache = {}
        self.maps = []

    def aggregated_map(self) -> Optional[torch.Tensor]:
        if not self.maps:
            return None
        max_h = max(m.shape[-2] for m in self.maps)
        max_w = max(m.shape[-1] for m in self.maps)
        ups = []
        for m in self.maps:
            ten = m.unsqueeze(1).float()
            if m.shape[-2:] != (max_h, max_w):
                ten = F.interpolate(ten, size=(max_h, max_w), mode="bilinear", align_corners=False)
            ups.append(ten.squeeze(1))
        return torch.stack(ups, dim=0).mean(dim=0)


def find_token_positions(tokenizer, prompt: str, keywords: List[str]) -> List[int]:
    toks = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    pieces = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    kws = [k.lower() for k in keywords]
    positions = []
    for idx, piece in enumerate(pieces):
        p = piece.lower().replace("Ġ", "").replace("▁", "")
        if any(k in p for k in kws):
            positions.append(idx)
    return positions


def normalize01(att: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(att.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(a.max()) <= float(a.min()):
        return torch.zeros_like(a)
    return ((a - a.min()) / (a.max() - a.min())).clamp(0, 1)


def heat_pil(att: torch.Tensor, size=(180, 180)) -> Image.Image:
    a = normalize01(att).mul(255).to(torch.uint8).numpy()
    return ImageOps.colorize(Image.fromarray(a, mode="L"), black="black", mid="orange", white="red").resize(
        size, Image.BILINEAR
    )


def img_pil(img: torch.Tensor, size=(180, 180)) -> Image.Image:
    arr = ((img.float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr).resize(size, Image.BILINEAR)


def overlay_heat(att: torch.Tensor, img: torch.Tensor, size=(180, 180), alpha=0.55) -> Image.Image:
    base = img_pil(img, size).convert("RGBA")
    norm = normalize01(att)
    heat = heat_pil(att, size).convert("RGBA")
    mask = F.interpolate(norm.view(1, 1, *norm.shape), size=(size[1], size[0]), mode="bilinear", align_corners=False)
    heat.putalpha(Image.fromarray(mask.view(size[1], size[0]).mul(255 * alpha).to(torch.uint8).numpy(), mode="L"))
    return Image.alpha_composite(base, heat).convert("RGB")


def mask_to_overlay(mask: torch.Tensor, img: torch.Tensor, size=(180, 180), color=(255, 32, 32), alpha=0.48) -> Image.Image:
    base = img_pil(img, size).convert("RGBA")
    m = mask.detach().float().cpu()
    m = F.interpolate(m.view(1, 1, *m.shape), size=(size[1], size[0]), mode="nearest").view(size[1], size[0])
    color_img = Image.new("RGBA", size, color + (0,))
    color_img.putalpha(Image.fromarray(m.mul(255 * alpha).to(torch.uint8).numpy(), mode="L"))
    return Image.alpha_composite(base, color_img).convert("RGB")


def threshold_mask(att: torch.Tensor, threshold: float) -> torch.Tensor:
    return (normalize01(att) >= threshold).float()


def top_frac_mask(att: torch.Tensor, frac: float) -> torch.Tensor:
    a = att.detach().float().cpu().clamp_min(0.0)
    flat = a.flatten()
    k = max(1, int(round(flat.numel() * frac)))
    if k >= flat.numel():
        return torch.ones_like(a)
    cutoff = torch.topk(flat, k).values.min()
    return (a >= cutoff).float()


def label(text: str, w: int, h: int = 24) -> Image.Image:
    out = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 4), text, fill="white", font=font)
    return out


def titled(cell: Image.Image, title: str) -> Image.Image:
    w, h = cell.size
    strip = label(title, w)
    out = Image.new("RGB", (w, h + strip.size[1]), "black")
    out.paste(strip, (0, 0))
    out.paste(cell, (0, strip.size[1]))
    return out


def hcat(cells: List[Image.Image], pad=4) -> Image.Image:
    h = max(c.size[1] for c in cells)
    w = sum(c.size[0] for c in cells) + pad * (len(cells) + 1)
    out = Image.new("RGB", (w, h + 2 * pad), "black")
    x = pad
    for cell in cells:
        out.paste(cell, (x, pad))
        x += cell.size[0] + pad
    return out


def vcat(rows: List[Image.Image], pad=6) -> Image.Image:
    w = max(r.size[0] for r in rows)
    h = sum(r.size[1] for r in rows) + pad * (len(rows) + 1)
    out = Image.new("RGB", (w, h), "black")
    y = pad
    for row in rows:
        out.paste(row, (0, y))
        y += row.size[1] + pad
    return out


def map_diagnostics(att: torch.Tensor):
    a = att.detach().float().cpu().clamp_min(0)
    flat = a.flatten()
    p = flat / flat.sum().clamp_min(1e-12)
    eff = float(1.0 / p.pow(2).sum().clamp_min(1e-12))
    return {
        "effective_support": eff,
        "effective_support_frac": eff / flat.numel(),
        "max_over_mean": float(a.max() / a.mean().clamp_min(1e-12)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="a photo of the face of a person")
    parser.add_argument("--n_images", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--score_batch_size", type=int, default=2)
    parser.add_argument("--num_denoising_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sds_t_min", type=int, default=400)
    parser.add_argument("--sds_t_max", type=int, default=800)
    parser.add_argument("--sds_num_t", type=int, default=15)
    parser.add_argument("--sds_num_eps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5991)
    parser.add_argument("--thresholds", type=str, default="0.2,0.4,0.6")
    parser.add_argument("--top_fracs", type=str, default="0.30,0.20,0.10,0.05")
    parser.add_argument("--out_dir", type=str, default="./attmap_threshold_mask_out")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    high_dtype = torch.float32
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    top_fracs = [float(x) for x in args.top_fracs.split(",") if x.strip()]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for sub in ("images", "maps", "panels"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    print("[load] Stable Diffusion components")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(device, dtype).eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device, dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(device, dtype).eval()
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model, subfolder="scheduler")
    ddpm = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    for module in (text_encoder, vae, unet):
        module.requires_grad_(False)

    def text_embeds(prompts, pad_max=False):
        if pad_max:
            toks = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        else:
            toks = tokenizer(prompts, return_tensors="pt", padding=True)
        return text_encoder(toks.input_ids.to(device), toks.attention_mask.to(device))[0]

    @torch.no_grad()
    def generate(noises: torch.Tensor) -> torch.Tensor:
        n = noises.shape[0]
        prompt_embeds = text_embeds([args.prompt] * n, pad_max=False)
        uncond = tokenizer([""] * n, padding="max_length", max_length=prompt_embeds.shape[1], truncation=True, return_tensors="pt")
        neg = text_encoder(uncond.input_ids.to(device), uncond.attention_mask.to(device))[0]
        embeds = torch.cat([neg, prompt_embeds]).to(dtype)
        noise_scheduler.set_timesteps(args.num_denoising_steps)
        latents = noises
        for t in noise_scheduler.timesteps:
            latent_input = torch.cat([latents.to(dtype)] * 2)
            latent_input = noise_scheduler.scale_model_input(latent_input, t)
            eps_pred = unet(latent_input, t, encoder_hidden_states=embeds).sample.to(high_dtype)
            eps_u, eps_c = eps_pred.chunk(2)
            eps_pred = eps_u + args.guidance_scale * (eps_c - eps_u)
            latents = noise_scheduler.step(eps_pred, t, latents).prev_sample
        images = vae.decode((latents / vae.config.scaling_factor).to(vae.dtype)).sample.clamp(-1, 1)
        return images

    print(f"[gen] generating {args.n_images} images")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    image_batches = []
    for start in range(0, args.n_images, args.batch_size):
        end = min(start + args.batch_size, args.n_images)
        noises = torch.randn((end - start, 4, 64, 64), generator=gen, device=device, dtype=dtype)
        imgs = generate(noises)
        image_batches.append(imgs.cpu())
        print(f"  generated {start}:{end}")
    images = torch.cat(image_batches, dim=0)
    for i in range(images.shape[0]):
        img_pil(images[i], (512, 512)).save(os.path.join(args.out_dir, "images", f"img_{i:03d}.png"))

    female_prompt = "a photo of a woman"
    male_prompt = "a photo of a man"
    tok_pos_f = find_token_positions(tokenizer, female_prompt, ["woman"])
    tok_pos_m = find_token_positions(tokenizer, male_prompt, ["man"])
    if not tok_pos_f or not tok_pos_m:
        raise RuntimeError(f"token positions not found: woman={tok_pos_f}, man={tok_pos_m}")
    print(f"[attn] woman token positions={tok_pos_f}, man token positions={tok_pos_m}")

    @torch.no_grad()
    def captured_unet(zt, t_vec, embeds, token_positions):
        capture = CrossAttnCapture(token_indices=token_positions, use_cpu=False, expect_cfg_pair=False).add_hooks(unet)
        try:
            _ = unet(zt.to(dtype), t_vec, encoder_hidden_states=embeds).sample
            att = capture.aggregated_map()
        finally:
            capture.clear()
        return att

    attmaps = torch.zeros((args.n_images, 64, 64), dtype=torch.float32)
    t_idx = torch.linspace(args.sds_t_min, args.sds_t_max, steps=args.sds_num_t, device=device).round().long()
    alphas = ddpm.alphas_cumprod.to(device=device, dtype=dtype)
    print("[attn] extracting scoring-time woman/man averaged attention maps")
    for start in range(0, args.n_images, args.score_batch_size):
        end = min(start + args.score_batch_size, args.n_images)
        imgs = images[start:end].to(device=device, dtype=dtype)
        bsz = imgs.shape[0]
        latents = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
        lat_exp = (
            latents.unsqueeze(1)
            .unsqueeze(2)
            .expand(bsz, args.sds_num_t, args.sds_num_eps, *latents.shape[1:])
            .contiguous()
            .view(bsz * args.sds_num_t * args.sds_num_eps, *latents.shape[1:])
        )
        t_vec = t_idx.view(1, args.sds_num_t, 1).expand(bsz, args.sds_num_t, args.sds_num_eps).reshape(-1)
        eps = torch.randn_like(lat_exp, dtype=dtype)
        ab = alphas[t_vec]
        zt = ab.sqrt().view(-1, 1, 1, 1) * lat_exp + (1 - ab).sqrt().view(-1, 1, 1, 1) * eps
        bke = zt.shape[0]

        emb_f = text_embeds([female_prompt] * bke, pad_max=True).to(dtype)
        emb_m = text_embeds([male_prompt] * bke, pad_max=True).to(dtype)
        att_f = captured_unet(zt, t_vec, emb_f, tok_pos_f)
        att_m = captured_unet(zt, t_vec, emb_m, tok_pos_m)
        if att_f is None or att_m is None:
            raise RuntimeError("Failed to capture SDS attention maps.")

        def to64(att):
            if att.shape[-2:] != (64, 64):
                att = F.interpolate(att.unsqueeze(1).float(), size=(64, 64), mode="bilinear", align_corners=False).squeeze(1)
            return att

        att_f = to64(att_f).view(bsz, args.sds_num_t, args.sds_num_eps, 64, 64).mean(dim=(1, 2))
        att_m = to64(att_m).view(bsz, args.sds_num_t, args.sds_num_eps, 64, 64).mean(dim=(1, 2))
        attmaps[start:end] = (0.5 * (att_f + att_m)).detach().cpu()
        print(f"  attmaps {start}:{end}")

    torch.save({"attmaps": attmaps, "images": images, "config": vars(args)}, os.path.join(args.out_dir, "maps", "attmaps_and_images.pt"))

    records = []
    panels = []
    cell = (160, 160)
    print("[vis] writing panels")
    for i in range(args.n_images):
        img = images[i]
        att = attmaps[i]
        cells = [
            titled(img_pil(img, cell), f"{i:03d} image"),
            titled(overlay_heat(att, img, cell), "raw attn"),
        ]
        rec = {"idx": i, "map": map_diagnostics(att), "threshold": {}, "top_frac": {}}
        for th in thresholds:
            mask = threshold_mask(att, th)
            rec["threshold"][str(th)] = float(mask.mean())
            cells.append(titled(mask_to_overlay(mask, img, cell), f"thr {th:g} ({mask.mean()*100:.1f}%)"))
        for frac in top_fracs:
            mask = top_frac_mask(att, frac)
            rec["top_frac"][str(frac)] = float(mask.mean())
            cells.append(titled(mask_to_overlay(mask, img, cell, color=(64, 180, 255)), f"top {frac*100:g}%"))
        panel = hcat(cells)
        panel.save(os.path.join(args.out_dir, "panels", f"panel_{i:03d}.png"))
        panels.append(panel)
        records.append(rec)

    page_size = 10
    page_paths = []
    for page, start in enumerate(range(0, len(panels), page_size)):
        out = vcat(panels[start:start + page_size])
        path = os.path.join(args.out_dir, f"panels_page_{page:02d}.png")
        out.save(path)
        page_paths.append(path)

    summary = {
        "config": vars(args),
        "page_paths": page_paths,
        "mean_map_diagnostics": {
            key: float(np.mean([r["map"][key] for r in records])) for key in records[0]["map"]
        },
        "mean_retained_by_threshold": {
            str(th): float(np.mean([r["threshold"][str(th)] for r in records])) for th in thresholds
        },
        "mean_retained_by_top_frac": {
            str(frac): float(np.mean([r["top_frac"][str(frac)] for r in records])) for frac in top_fracs
        },
        "per_image": records,
        "notes": {
            "thresholds": "Applied to per-image min-max normalized SDS attention map.",
            "top_frac": "Applied by raw attention rank; retained fraction is fixed except ties.",
            "attmap": "Average of woman/man token cross-attention during SDS scoring, matching chekc_SCRclip_face_grad.py return_attmap path.",
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"DONE: {os.path.abspath(args.out_dir)}")
    print("pages:")
    for path in page_paths:
        print(f"  {os.path.abspath(path)}")
    print("mean retained threshold:", summary["mean_retained_by_threshold"])
    print("mean map diagnostics:", summary["mean_map_diagnostics"])
    print("=" * 80)


if __name__ == "__main__":
    main()
