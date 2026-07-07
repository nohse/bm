#!/usr/bin/env python
# coding=utf-8
"""
attmap_gradient_focus_experiment.py
===================================

목적 (분석 #1 의 핵심 검증):
  attmap 은 "eval 성별판정(CE 값)" 에는 거의 영향이 없다 (attmap_gen_vs_score_experiment.py 결과).
  하지만 학습에서 SDS 분류기는 '미분가능한 critic' 이고, fair loss = CE(logits=-sds/tau, target) 의
  gradient 가 이미지로 흘러 들어간다. 이때 attmap(w) 은 eps-residual MSE 를 '공간적으로 재가중' 하므로,
  **학습 gradient 가 이미지의 어디를 고치는지(공간 focus)** 를 결정한다.

  이 스크립트는 그것을 직접 보여준다:
    같은 이미지에 대해 fairness 신호 sds_diff = (sds_man - sds_woman) 의
    d/d(image) gradient 를 계산하고, weight map 을
      - GEN_MID   (얼굴에 집중된 map)
      - SCORE_ALL (거의 균일한 map)
      - nomask    (마스크 없음)
    로 바꿔가며 |grad| 공간 분포가 얼마나 얼굴에 집중되는지 / 화면 전체로 퍼지는지 비교한다.

  가설: 얼굴집중 map -> gradient 가 얼굴에 집중 -> 성별 편집이 얼굴에만 -> 배경/품질 보존 + 또렷한 성별.
        균일 map(현재 ALL) -> gradient 가 배경까지 퍼짐 -> 편집이 전체로 번져 품질 저하 + 성별 애매.

입력: attmap_gen_vs_score_out/ 의 images/*.png 와 maps/attmaps.pt 를 재사용.
"""

import os, json, math, argparse
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler


# ---- vis helpers (원본 복사) ----
def _norm_vis(a):
    a = a.detach().float().cpu()
    a = torch.nan_to_num(a).clamp_min(0.0)
    nz = a[a > 0]
    if nz.numel() == 0:
        return torch.zeros_like(a)
    lo = torch.quantile(nz, 0.05); hi = torch.quantile(nz, 0.995)
    if not torch.isfinite(lo): lo = nz.min()
    if not torch.isfinite(hi): hi = nz.max()
    return ((a > 0).float() if hi <= lo else ((a - lo) / (hi - lo)).clamp(0, 1)).pow(0.5)


def heat_pil(a, size=(256, 256)):
    v = _norm_vis(a).clamp(0, 1).mul(255).to(torch.uint8).numpy()
    return ImageOps.colorize(Image.fromarray(v, mode="L"), black="black", mid="orange", white="red").resize(size, Image.BILINEAR)


def img_pil(img, size=(256, 256)):
    arr = ((img.float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr).resize(size, Image.BILINEAR)


def overlay(a, img, size=(256, 256), alpha=0.55):
    base = img_pil(img, size).convert("RGBA")
    v = _norm_vis(a)
    heat = heat_pil(a, size).convert("RGBA")
    vr = F.interpolate(v.unsqueeze(0).unsqueeze(0), size=(size[1], size[0]), mode="bilinear", align_corners=False).squeeze()
    heat.putalpha(Image.fromarray(vr.mul(255 * alpha).clamp(0, 255).to(torch.uint8).numpy(), mode="L"))
    return Image.alpha_composite(base, heat).convert("RGB")


def label(text, w, h=26):
    s = Image.new("RGB", (w, h), "black"); d = ImageDraw.Draw(s)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        f = ImageFont.load_default()
    d.text((4, 4), text, fill="white", font=f); return s


def titled(c, t):
    w, h = c.size; s = label(t, w)
    o = Image.new("RGB", (w, h + s.size[1]), "black"); o.paste(s, (0, 0)); o.paste(c, (0, s.size[1])); return o


def hcat(cs, pad=4):
    h = max(c.size[1] for c in cs); w = sum(c.size[0] for c in cs) + pad * (len(cs) + 1)
    o = Image.new("RGB", (w, h + 2 * pad), "black"); x = pad
    for c in cs:
        o.paste(c, (x, pad)); x += c.size[0] + pad
    return o


def vcat(rs, pad=4):
    w = max(r.size[0] for r in rs); h = sum(r.size[1] for r in rs) + pad * (len(rs) + 1)
    o = Image.new("RGB", (w, h), "black"); y = pad
    for r in rs:
        o.paste(r, (0, y)); y += r.size[1] + pad
    return o


def eff_support(a):
    a = a.detach().float().cpu().clamp_min(0).flatten()
    p = a / a.sum().clamp_min(1e-12)
    return float(1.0 / p.pow(2).sum().clamp_min(1e-12))


def center_frac(a, frac=0.5):
    """중앙 정사각(면적 frac) 안에 |grad| 질량이 얼마나 들어가나 (얼굴은 대개 중앙)."""
    a = a.detach().float().cpu().clamp_min(0)
    H, W = a.shape
    ch, cw = int(H * math.sqrt(frac)), int(W * math.sqrt(frac))
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    return float(a[y0:y0 + ch, x0:x0 + cw].sum() / a.sum().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--src_dir", type=str, default="./attmap_gen_vs_score_out")
    ap.add_argument("--out_dir", type=str, default="./attmap_gradient_focus_out")
    ap.add_argument("--idxs", type=str, default="0,1,2,5,8", help="분석할 이미지 인덱스")
    ap.add_argument("--sds_t_min", type=int, default=400)
    ap.add_argument("--sds_t_max", type=int, default=800)
    ap.add_argument("--sds_num_t", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5991)
    args = ap.parse_args()

    device = torch.device("cuda")
    dt = torch.float32  # gradient 안정성 위해 fp32
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "panels"), exist_ok=True)

    print("[load] SD1.5 (fp32) ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(device, dt).eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device, dt).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(device, dt).eval()
    ddpm = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    def temb(prompts):
        t = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        return text_encoder(t.input_ids.to(device), t.attention_mask.to(device))[0]

    # ---- 입력 재사용 ----
    maps = torch.load(os.path.join(args.src_dir, "maps", "attmaps.pt"), map_location="cpu")
    GEN_MID = maps["GEN"]["MID"]        # [N,8,8]  얼굴집중
    SCORE_ALL = maps["SCORE"]["ALL"]    # [N,64,64] 거의 균일
    idxs = [int(x) for x in args.idxs.split(",")]

    female_prompt, male_prompt = "a photo of a woman", "a photo of a man"
    pe_f1 = temb([female_prompt]); pe_m1 = temb([male_prompt])

    K = args.sds_num_t
    t_idx = torch.linspace(args.sds_t_min, args.sds_t_max, steps=K, device=device).round().long()
    alphacum = ddpm.alphas_cumprod.to(device=device, dtype=dt)

    def load_img(i):
        p = os.path.join(args.src_dir, "images", f"img_{i:02d}.png")
        arr = np.asarray(Image.open(p).convert("RGB").resize((512, 512))).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).mul(2).sub(1).to(device, dt)  # [-1,1]

    def prep_w(i, which, H, W):
        if which == "GEN_MID":
            w = GEN_MID[i].to(device, dt)
        elif which == "SCORE_ALL":
            w = SCORE_ALL[i].to(device, dt)
        else:
            return None
        if w.shape[-2:] != (H, W):
            w = F.interpolate(w.view(1, 1, *w.shape), size=(H, W), mode="bilinear", align_corners=False).view(H, W)
        return w.clamp_min(0)

    def sds_scalar(img, pe, w):
        """weighted SDS(가중 MSE) scalar; grad enabled on img. 현재 코드 reduction(ŵ^1) 과 동일."""
        lat = vae.encode(img.unsqueeze(0)).latent_dist.mean * vae.config.scaling_factor  # [1,4,64,64] (mean: 재현성)
        H, W = lat.shape[-2], lat.shape[-1]
        gen = torch.Generator(device=device).manual_seed(args.seed)
        total = 0.0
        for t in t_idx:
            eps = torch.randn(lat.shape, generator=gen, device=device, dtype=dt)
            ab = alphacum[t]
            zt = ab.sqrt() * lat + (1 - ab).sqrt() * eps
            eps_pred = unet(zt, t, encoder_hidden_states=pe).sample
            err = ((eps_pred - eps) ** 2).mean(dim=1)  # [1,H,W]
            if w is None:
                s = err.flatten(1).mean(dim=1)
            else:
                ww = w.view(1, H, W).clamp_min(0)
                wsum = ww.flatten(1).sum(1).clamp_min(1e-8)
                ww = ww / wsum.view(-1, 1, 1) * (H * W)
                s = (err * ww).flatten(1).mean(dim=1)
            total = total + s.squeeze(0)
        return total / K

    results = {}
    panels = []
    WHICH = ["GEN_MID", "SCORE_ALL", "nomask"]
    for i in idxs:
        img0 = load_img(i)
        grad_maps = {}
        for which in WHICH:
            img = img0.detach().clone().requires_grad_(True)
            w = prep_w(i, which, 64, 64)
            sds_f = sds_scalar(img, pe_f1, w)
            sds_m = sds_scalar(img, pe_m1, w)
            sds_diff = sds_m - sds_f   # fairness 방향: '남자로 vs 여자로' 신호
            if img.grad is not None:
                img.grad = None
            sds_diff.backward()
            g = img.grad.detach().abs().mean(0)  # [512,512] 픽셀별 gradient magnitude
            # latent 해상도로 downsample 해서 map spread 진단 (공정 비교)
            g64 = F.interpolate(g.view(1, 1, 512, 512), size=(64, 64), mode="area").view(64, 64)
            grad_maps[which] = {"g512": g.cpu(), "g64": g64.cpu(),
                                "eff_support": eff_support(g64), "center_frac": center_frac(g64, 0.5),
                                "sds_diff": float(sds_diff.detach())}
        results[i] = {k: {kk: v[kk] for kk in ("eff_support", "center_frac", "sds_diff")} for k, v in grad_maps.items()}

        CELL = (256, 256)
        row = [titled(img_pil(img0, CELL), f"[{i}] image")]
        for which in WHICH:
            gm = grad_maps[which]
            row.append(titled(overlay(gm["g512"], img0, CELL),
                              f"|grad| {which}\neff={gm['eff_support']:.0f} c50={gm['center_frac']:.2f}"))
        panel = hcat(row)
        cap = "  ".join([f"{w}: center50={grad_maps[w]['center_frac']:.2f} eff_supp={grad_maps[w]['eff_support']:.0f}" for w in WHICH])
        panel = vcat([panel, label(cap, panel.size[0], h=26)])
        panel.save(os.path.join(args.out_dir, "panels", f"grad_{i:02d}.png"))
        panels.append(panel)
        print(f"  img {i}: " + "  ".join([f"{w} center50={grad_maps[w]['center_frac']:.3f} eff={grad_maps[w]['eff_support']:.0f}" for w in WHICH]))

    vcat(panels, pad=8).save(os.path.join(args.out_dir, "ALL_grad_panels.png"))

    # 집계
    agg = {}
    for which in WHICH:
        cf = np.mean([results[i][which]["center_frac"] for i in idxs])
        es = np.mean([results[i][which]["eff_support"] for i in idxs])
        agg[which] = {"mean_center_frac_0.5": float(cf), "mean_eff_support_64": float(es)}
    summary = {"config": vars(args), "idxs": idxs,
               "explanation": "center_frac=중앙 50% 면적에 들어간 |grad| 비율(↑=얼굴집중). eff_support=|grad|의 유효 픽셀수(↓=집중).",
               "aggregate": agg, "per_image": results}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"DONE -> {os.path.abspath(args.out_dir)}")
    print("[gradient focus] center50(↑얼굴집중)  eff_support(↓집중)")
    for which in WHICH:
        print(f"   {which:10s}: center50={agg[which]['mean_center_frac_0.5']:.3f}  eff_support={agg[which]['mean_eff_support_64']:.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
