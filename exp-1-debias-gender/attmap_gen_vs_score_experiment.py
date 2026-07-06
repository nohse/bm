#!/usr/bin/env python
# coding=utf-8
"""
attmap_gen_vs_score_experiment.py
=================================

목적 (사용자 요청 + 후속 분석):
  사람 얼굴 N(기본 10)개를 생성하고, 각 이미지에 대해 attmap 을 뽑아 시각화/저장하고,
  그 attmap 으로 SDS residual error 를 가중 곱 -> CE(logits/softmax) 로 바꿔 비교한다.

두 개의 독립 축을 분리해서 본다:
  축1) 어디서 뽑나:  GEN  = 생성(denoising trajectory) 하면서 "face" 토큰 attention
                     SCORE= 채점(SDS 재노이징 t) 하면서 woman/man 토큰 attention 평균
  축2) 어느 layer:   ALL  = 모든 cross-attn layer 평균 (현재 sds_logits_from_images / CrossAttnCapture 방식)
                     MID  = mid_block cross-attn 만    (예전 _attn_weight_for_prompt / _install_mid_recorders 방식)
                     RES16= 16x16 해상도 cross-attn 만 (prompt-to-prompt 에서 가장 의미있는 해상도)

즉 각 이미지마다 최대 6개 map: GEN×{ALL,MID,RES16}, SCORE×{ALL,MID,RES16}.

그리고 (C): 동일 이미지/eps/eps_pred/재노이징에 대해 weight map 만 각 map 으로 바꿔가며
  err_woman, err_man 을 가중 평균 -> sds_f, sds_m -> logits=[-sds_f/tau,-sds_m/tau]
  -> softmax -> p(woman)/p(man), CE 를 계산해 "weight map 선택"이 gender logit 을
  어떻게 바꾸는지 숫자와 그림으로 보여준다.

출력물은 --out_dir 아래.
"""

import os, json, math, argparse, random
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, DDPMScheduler

SCOPES = ["ALL", "MID", "RES16"]


# =====================================================================================
# CrossAttnCapture (원본 복사 + scope 태깅 추가)
#   원본과 동일하게 to_q/to_k hook 으로 지정 토큰들의 cross-attn 을 [B,H,W] 로 수집.
#   추가: 각 map 이 어느 layer(mid 여부 / 해상도) 에서 나왔는지 태깅해서 scope 별 aggregate 지원.
# =====================================================================================
class CrossAttnCapture:
    def __init__(self, token_indices: List[int], use_cpu: bool = False, expect_cfg_pair: bool = True):
        self.token_indices = token_indices
        self.handles = []
        self.maps = []       # list of [B,h,w]
        self.map_is_mid = [] # bool per map
        self.map_hw = []     # int per map
        self.use_cpu = use_cpu
        self._q_cache = {}
        self._parent_map = {}
        self._mid_ids = set()
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
            B, Nq, inner_q = q.shape
            _, Nk, inner_k = k.shape
            if inner_q != inner_k:
                return
            if heads is None:
                for h in (8, 12, 16, 4, 6, 24, 32):
                    if inner_q % h == 0:
                        heads = h
                        break
                heads = heads or 8
            head_dim = inner_q // heads
            q = q.view(B, Nq, heads, head_dim).permute(0, 2, 1, 3).contiguous()
            k = k.view(B, Nk, heads, head_dim).permute(0, 2, 1, 3).contiguous()
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
            attn = attn_scores.softmax(dim=-1)
            if not self.token_indices:
                return
            tok_idx = torch.tensor(self.token_indices, device=device, dtype=torch.long)
            attn_tok = attn.index_select(-1, tok_idx).mean(dim=-1)  # (B,H,Nq)
            if self.expect_cfg_pair and B >= 2:
                b_half = B // 2
                cond = attn_tok[B - b_half:B]
            else:
                cond = attn_tok
            per_image = cond.mean(dim=1)  # (B_sel, Nq)
            hw = int(math.sqrt(per_image.shape[1]))
            if hw * hw != per_image.shape[1]:
                return
            per_image = per_image.view(per_image.shape[0], hw, hw)
            if self.use_cpu:
                per_image = per_image.cpu()
            self.maps.append(per_image)
            self.map_is_mid.append(parent_id in self._mid_ids)
            self.map_hw.append(hw)

    def add_hooks(self, unet: torch.nn.Module):
        for name, module in unet.named_modules():
            if not (hasattr(module, "to_q") and hasattr(module, "to_k")):
                continue
            is_cross = getattr(module, "is_cross_attention", None)
            if is_cross is None:
                is_cross = ("attn2" in name) or ("Cross" in module.__class__.__name__)
            if not is_cross:
                continue
            if name.startswith("mid_block"):
                self._mid_ids.add(id(module))
            self._parent_map[id(module.to_q)] = module
            self._parent_map[id(module.to_k)] = module
            self.handles.append(module.to_q.register_forward_hook(self._q_hook))
            self.handles.append(module.to_k.register_forward_hook(self._k_hook))
        return self

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self._q_cache = {}
        self.maps, self.map_is_mid, self.map_hw = [], [], []

    def aggregated_map(self, scope: str = "ALL") -> Optional[torch.Tensor]:
        idxs = list(range(len(self.maps)))
        if scope == "MID":
            idxs = [i for i in idxs if self.map_is_mid[i]]
        elif scope == "RES16":
            idxs = [i for i in idxs if self.map_hw[i] == 16]
        if not idxs:
            return None
        sel = [self.maps[i] for i in idxs]
        max_hw = max(m.shape[-1] for m in sel)
        ups = []
        for m in sel:
            ten = m.unsqueeze(1).float()
            if m.shape[-1] != max_hw or m.shape[-2] != max_hw:
                ten = F.interpolate(ten, size=(max_hw, max_hw), mode="bilinear", align_corners=False)
            ups.append(ten.squeeze(1))
        return torch.stack(ups, 0).mean(0)  # [B,H,W]


def find_token_positions(tokenizer, prompt: str, keywords: List[str]) -> List[int]:
    toks = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                     truncation=True, return_tensors="pt")
    pieces = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    kws = [k.lower() for k in keywords]
    pos = []
    for i, piece in enumerate(pieces):
        p = piece.lower().replace("Ġ", "").replace("▁", "")
        if any(k in p for k in kws):
            pos.append(i)
    return pos


# =====================================================================================
# 시각화 helper  (원본 파일에서 복사)
# =====================================================================================
def _normalize_attmap_for_vis(att: torch.Tensor) -> torch.Tensor:
    a = att.detach().float().cpu()
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    nz = a[a > 0]
    if nz.numel() == 0:
        return torch.zeros_like(a)
    lo = torch.quantile(nz, 0.05); hi = torch.quantile(nz, 0.995)
    if not torch.isfinite(lo): lo = nz.min()
    if not torch.isfinite(hi): hi = nz.max()
    scaled = (a > 0).float() if hi <= lo else ((a - lo) / (hi - lo)).clamp(0.0, 1.0)
    return scaled.pow(0.5)


def _attmap_vis_to_pil(vis: torch.Tensor) -> Image.Image:
    a = vis.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    return ImageOps.colorize(Image.fromarray(a, mode="L"), black="black", mid="orange", white="red")


def attmap_to_pil(att: torch.Tensor, size=(256, 256)) -> Image.Image:
    return _attmap_vis_to_pil(_normalize_attmap_for_vis(att)).resize(size, resample=Image.BILINEAR)


def img_tensor_to_pil(img: torch.Tensor, size=(256, 256)) -> Image.Image:
    arr = ((img.float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr).resize(size, resample=Image.BILINEAR)


def overlay_pil(att: torch.Tensor, img: torch.Tensor, size=(256, 256), alpha=0.5) -> Image.Image:
    base = img_tensor_to_pil(img, size).convert("RGBA")
    vis = _normalize_attmap_for_vis(att)
    heat = _attmap_vis_to_pil(vis).resize(size, resample=Image.BILINEAR).convert("RGBA")
    vis_r = F.interpolate(vis.unsqueeze(0).unsqueeze(0), size=(size[1], size[0]),
                          mode="bilinear", align_corners=False).squeeze()
    amask = Image.fromarray(vis_r.mul(255 * alpha).clamp(0, 255).to(torch.uint8).cpu().numpy(), mode="L")
    heat.putalpha(amask)
    return Image.alpha_composite(base, heat).convert("RGB")


def label_strip(text: str, w: int, h: int = 26) -> Image.Image:
    strip = Image.new("RGB", (w, h), "black")
    d = ImageDraw.Draw(strip)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        fnt = ImageFont.load_default()
    d.text((4, 4), text, fill="white", font=fnt)
    return strip


def titled(cell: Image.Image, title: str) -> Image.Image:
    w, h = cell.size
    strip = label_strip(title, w)
    out = Image.new("RGB", (w, h + strip.size[1]), "black")
    out.paste(strip, (0, 0)); out.paste(cell, (0, strip.size[1]))
    return out


def hcat(cells: List[Image.Image], pad=4, bg="black") -> Image.Image:
    h = max(c.size[1] for c in cells)
    w = sum(c.size[0] for c in cells) + pad * (len(cells) + 1)
    out = Image.new("RGB", (w, h + 2 * pad), bg)
    x = pad
    for c in cells:
        out.paste(c, (x, pad)); x += c.size[0] + pad
    return out


def vcat(rows: List[Image.Image], pad=4, bg="black") -> Image.Image:
    w = max(r.size[0] for r in rows)
    h = sum(r.size[1] for r in rows) + pad * (len(rows) + 1)
    out = Image.new("RGB", (w, h), bg)
    y = pad
    for r in rows:
        out.paste(r, (0, y)); y += r.size[1] + pad
    return out


# =====================================================================================
def map_diagnostics(w: torch.Tensor):
    w = w.detach().float().cpu().clamp_min(0)
    s = w.sum().clamp_min(1e-12)
    p = (w / s).flatten()
    eff = float(1.0 / p.pow(2).sum().clamp_min(1e-12))
    ent = float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum())
    hw = p.numel()
    return {"effective_support": eff, "effective_support_frac": eff / hw,
            "entropy_nats": ent, "max_over_mean": float(w.max() / w.mean().clamp_min(1e-12))}


def weighted_sds(err_bke, w_bke, B, K, num_eps):
    """현재 sds_logits_from_images 와 동일한 weighted-mean reduction (ŵ^1)."""
    if w_bke is not None:
        w = w_bke.to(err_bke.dtype).clamp_min(0)
        w_sum = w.flatten(1).sum(dim=1).clamp_min(1e-8)
        hw = err_bke.shape[-2] * err_bke.shape[-1]
        w = (w / w_sum.view(-1, 1, 1)) * hw
        sds_per = (err_bke * w).flatten(1).mean(dim=1)
    else:
        sds_per = err_bke.flatten(1).mean(dim=1)
    return sds_per.float().view(B, K, num_eps).mean(dim=(1, 2))


def ce_from_sds(sds_f, sds_m, tau):
    tau_t = torch.tensor(tau, dtype=torch.float32)
    logits = torch.stack([-sds_f / tau_t, -sds_m / tau_t], dim=1)
    logits = logits - logits.max(dim=1, keepdim=True).values
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    ce_w = -probs[:, 0].clamp_min(1e-12).log()
    ce_m = -probs[:, 1].clamp_min(1e-12).log()
    return logits, probs, preds, ce_w, ce_m


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--n_images", type=int, default=10)
    ap.add_argument("--gen_prompt", type=str, default="a photo of the face of a person")
    ap.add_argument("--gen_face_keyword", type=str, default="face")
    ap.add_argument("--num_denoising_steps", type=int, default=25)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--sds_t_min", type=int, default=400)
    ap.add_argument("--sds_t_max", type=int, default=800)
    ap.add_argument("--sds_num_t", type=int, default=15)
    ap.add_argument("--sds_num_eps", type=int, default=1)
    ap.add_argument("--sds_tau", type=float, default=0.0001)
    ap.add_argument("--sds_chunk", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5991)
    ap.add_argument("--out_dir", type=str, default="./attmap_gen_vs_score_out")
    args = ap.parse_args()

    device = torch.device("cuda")
    weight_dtype = torch.float16
    wd_high = torch.float32
    os.makedirs(args.out_dir, exist_ok=True)
    for sub in ["images", "maps", "panels"]:
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    print("[load] SD1.5 ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(device, weight_dtype).eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device, weight_dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(device, weight_dtype).eval()
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model, subfolder="scheduler")
    ddpm_forward = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    def text_embeds(prompts_list, pad_max=True):
        if pad_max:
            toks = tokenizer(prompts_list, padding="max_length", max_length=tokenizer.model_max_length,
                             truncation=True, return_tensors="pt")
        else:
            toks = tokenizer(prompts_list, return_tensors="pt", padding=True)
        return text_encoder(toks.input_ids.to(device), toks.attention_mask.to(device))[0]

    # =================================================================================
    # 1) 생성 + generation-time "face" attmap (GEN), scope 별
    # =================================================================================
    gen_tokpos = find_token_positions(tokenizer, args.gen_prompt, keywords=[args.gen_face_keyword])
    print(f"[gen] prompt='{args.gen_prompt}'  face token pos={gen_tokpos}")
    assert gen_tokpos, "gen_prompt 안에 face 키워드 토큰이 없음"

    @torch.no_grad()
    def generate_with_faceattn(prompt, noises):
        N = noises.shape[0]
        pe_c = text_embeds([prompt] * N, pad_max=False)
        un = tokenizer([""] * N, padding="max_length", max_length=pe_c.shape[1], truncation=True, return_tensors="pt")
        pe_u = text_encoder(un.input_ids.to(device), un.attention_mask.to(device))[0]
        pe = torch.cat([pe_u, pe_c]).to(weight_dtype)
        cap = CrossAttnCapture(token_indices=gen_tokpos, use_cpu=False, expect_cfg_pair=True).add_hooks(unet)
        noise_scheduler.set_timesteps(args.num_denoising_steps)
        latents = noises
        try:
            for t in noise_scheduler.timesteps:
                lmi = torch.cat([latents.to(weight_dtype)] * 2)
                lmi = noise_scheduler.scale_model_input(lmi, t)
                np_pred = unet(lmi, t, encoder_hidden_states=pe).sample.to(wd_high)
                nu, nt = np_pred.chunk(2)
                np_pred = nu + args.guidance_scale * (nt - nu)
                latents = noise_scheduler.step(np_pred, t, latents).prev_sample
            imgs = vae.decode((1 / vae.config.scaling_factor * latents).to(vae.dtype)).sample.clamp(-1, 1)
            gen_maps = {s: cap.aggregated_map(s) for s in SCOPES}  # [N,h,w] each (or None)
        finally:
            cap.clear()
        return imgs, gen_maps

    print(f"[gen] generating {args.n_images} face images ...")
    g = torch.Generator(device=device).manual_seed(args.seed)
    noises = torch.randn((args.n_images, 4, 64, 64), generator=g, device=device, dtype=weight_dtype)
    images, GEN = generate_with_faceattn(args.gen_prompt, noises)
    N = images.shape[0]
    for i in range(N):
        img_tensor_to_pil(images[i], (512, 512)).save(os.path.join(args.out_dir, "images", f"img_{i:02d}.png"))

    # =================================================================================
    # 2) scoring-time woman/man 평균 attmap (SCORE), scope 별 + eps_pred 로 err
    # =================================================================================
    female_prompt, male_prompt = "a photo of a woman", "a photo of a man"
    tok_pos_f = find_token_positions(tokenizer, female_prompt, keywords=["woman"])
    tok_pos_m = find_token_positions(tokenizer, male_prompt, keywords=["man"])
    print("[score] tok woman", tok_pos_f, "tok man", tok_pos_m)

    K, num_eps, tau = args.sds_num_t, args.sds_num_eps, args.sds_tau

    @torch.no_grad()
    def score_forward(zt, t_vec, pe, tokpos):
        cap = CrossAttnCapture(token_indices=tokpos, use_cpu=False, expect_cfg_pair=False).add_hooks(unet)
        try:
            eps_pred = unet(zt.to(weight_dtype), t_vec, encoder_hidden_states=pe).sample.to(weight_dtype)
            maps = {s: cap.aggregated_map(s) for s in SCOPES}
        finally:
            cap.clear()
        return eps_pred, maps

    # 누적: SCORE map (scope별) [N,64,64]
    SCORE = {s: torch.zeros((N, 64, 64), dtype=torch.float32) for s in SCOPES}
    # weight source 별 sds
    W_KEYS = [f"GEN_{s}" for s in SCOPES] + [f"SCORE_{s}" for s in SCOPES] + ["nomask"]
    acc = {k: {"f": [], "m": []} for k in W_KEYS}
    vis_err = {}

    def to64(mp_img, H, W):
        if mp_img.shape[-2:] != (H, W):
            return F.interpolate(mp_img.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        return mp_img

    for start in range(0, N, args.sds_chunk):
        end = min(start + args.sds_chunk, N)
        imgs = images[start:end].to(weight_dtype)
        Bc = imgs.shape[0]
        torch.manual_seed(args.seed + start)
        lat = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
        t_idx = torch.linspace(args.sds_t_min, args.sds_t_max, steps=K, device=device).round().long()
        lat_exp = lat.unsqueeze(1).unsqueeze(2).expand(Bc, K, num_eps, *lat.shape[1:]).contiguous().view(Bc * K * num_eps, *lat.shape[1:])
        t_vec = t_idx.view(1, K, 1).expand(Bc, K, num_eps).reshape(-1)
        BKE = lat_exp.shape[0]
        eps = torch.randn_like(lat_exp, dtype=weight_dtype)
        ab = ddpm_forward.alphas_cumprod.to(device=device, dtype=lat_exp.dtype)[t_vec]
        zt = ab.sqrt().view(-1, 1, 1, 1) * lat_exp + (1 - ab).sqrt().view(-1, 1, 1, 1) * eps
        _H, _W = zt.shape[-2], zt.shape[-1]

        pe_f = text_embeds([female_prompt] * BKE).to(weight_dtype)
        pe_m = text_embeds([male_prompt] * BKE).to(weight_dtype)
        eps_pred_f, maps_f = score_forward(zt, t_vec, pe_f, tok_pos_f)
        eps_pred_m, maps_m = score_forward(zt, t_vec, pe_m, tok_pos_m)

        # SCORE map (scope별): woman/man 평균, per-image
        score_img = {}
        for s in SCOPES:
            a_f = to64(maps_f[s].to(device, torch.float32), _H, _W).reshape(Bc, K, num_eps, _H, _W).mean((1, 2))
            a_m = to64(maps_m[s].to(device, torch.float32), _H, _W).reshape(Bc, K, num_eps, _H, _W).mean((1, 2))
            m = 0.5 * (a_f + a_m)
            score_img[s] = m
            SCORE[s][start:end] = m.cpu()

        err_f = ((eps_pred_f - eps) ** 2).mean(dim=1)  # [BKE,H,W]
        err_m = ((eps_pred_m - eps) ** 2).mean(dim=1)

        def to_bke(mp_img):
            return mp_img.unsqueeze(1).unsqueeze(2).expand(Bc, K, num_eps, _H, _W).contiguous().view(BKE, _H, _W)

        # GEN maps (scope별), per-image
        gen_img = {}
        for s in SCOPES:
            gm = GEN[s]
            if gm is None:
                gen_img[s] = None
                continue
            gen_img[s] = to64(gm[start:end].to(device, torch.float32), _H, _W)

        for s in SCOPES:
            if gen_img[s] is not None:
                acc[f"GEN_{s}"]["f"].append(weighted_sds(err_f, to_bke(gen_img[s]), Bc, K, num_eps).cpu())
                acc[f"GEN_{s}"]["m"].append(weighted_sds(err_m, to_bke(gen_img[s]), Bc, K, num_eps).cpu())
            acc[f"SCORE_{s}"]["f"].append(weighted_sds(err_f, to_bke(score_img[s]), Bc, K, num_eps).cpu())
            acc[f"SCORE_{s}"]["m"].append(weighted_sds(err_m, to_bke(score_img[s]), Bc, K, num_eps).cpu())
        acc["nomask"]["f"].append(weighted_sds(err_f, None, Bc, K, num_eps).cpu())
        acc["nomask"]["m"].append(weighted_sds(err_m, None, Bc, K, num_eps).cpu())

        # 시각화: 중앙 timestep 의 weighted error (MID scope 기준)
        mid = K // 2
        err_f_img = err_f.view(Bc, K, num_eps, _H, _W)[:, mid, 0]
        err_m_img = err_m.view(Bc, K, num_eps, _H, _W)[:, mid, 0]
        for bi in range(Bc):
            gi = start + bi
            def norm_w(mp):
                return mp / mp.flatten().sum().clamp_min(1e-8)
            wg = norm_w(gen_img["MID"][bi]) if gen_img["MID"] is not None else torch.ones_like(err_f_img[bi]) / err_f_img[bi].numel()
            ws = norm_w(score_img["MID"][bi])
            vis_err[gi] = {
                "GENmid_x_errf": (wg * err_f_img[bi]).cpu(), "GENmid_x_errm": (wg * err_m_img[bi]).cpu(),
                "SCOREmid_x_errf": (ws * err_f_img[bi]).cpu(), "SCOREmid_x_errm": (ws * err_m_img[bi]).cpu(),
            }
        print(f"   [score] images {start}:{end} done")

    # =================================================================================
    # 3) CE
    # =================================================================================
    def catB(d, fm):
        return torch.cat(d[fm], dim=0)

    results = {}
    for key in W_KEYS:
        if not acc[key]["f"]:
            continue
        sds_f = catB(acc[key], "f"); sds_m = catB(acc[key], "m")
        logits, probs, preds, ce_w, ce_m = ce_from_sds(sds_f, sds_m, tau)
        results[key] = dict(sds_f=sds_f, sds_m=sds_m, logits=logits, probs=probs,
                            preds=preds, ce_if_woman=ce_w, ce_if_man=ce_m)

    # =================================================================================
    # 4) 저장
    # =================================================================================
    torch.save({"GEN": {s: (GEN[s].cpu() if GEN[s] is not None else None) for s in SCOPES},
                "SCORE": {s: SCORE[s].cpu() for s in SCOPES},
                "gen_prompt": args.gen_prompt}, os.path.join(args.out_dir, "maps", "attmaps.pt"))

    CELL = (240, 240)
    panels = []
    per_image_json = []
    for i in range(N):
        rGa, rGm, rGr = results.get("GEN_ALL"), results.get("GEN_MID"), results.get("GEN_RES16")
        rSa, rSm, rSr = results.get("SCORE_ALL"), results.get("SCORE_MID"), results.get("SCORE_RES16")

        def pm(r):  # p(man)
            return float(r["probs"][i, 1]) if r is not None else float("nan")
        def pw(r):
            return float(r["probs"][i, 0]) if r is not None else float("nan")

        row_over = hcat([
            titled(img_tensor_to_pil(images[i], CELL), f"[{i}] generated face"),
            titled(overlay_pil(GEN["MID"][i], images[i], CELL) if GEN["MID"] is not None else img_tensor_to_pil(images[i], CELL), "GEN face MID"),
            titled(overlay_pil(SCORE["MID"][i], images[i], CELL), "SCORE w/m MID"),
            titled(overlay_pil(GEN["ALL"][i], images[i], CELL) if GEN["ALL"] is not None else img_tensor_to_pil(images[i], CELL), "GEN face ALL"),
            titled(overlay_pil(SCORE["ALL"][i], images[i], CELL), "SCORE w/m ALL"),
        ])
        row_heat = hcat([
            titled(attmap_to_pil(GEN["MID"][i], CELL) if GEN["MID"] is not None else Image.new("RGB", CELL), "GEN MID heat"),
            titled(attmap_to_pil(SCORE["MID"][i], CELL), "SCORE MID heat"),
            titled(attmap_to_pil(GEN["RES16"][i], CELL) if GEN["RES16"] is not None else Image.new("RGB", CELL), "GEN RES16 heat"),
            titled(attmap_to_pil(SCORE["RES16"][i], CELL), "SCORE RES16 heat"),
            titled(attmap_to_pil(GEN["ALL"][i], CELL) if GEN["ALL"] is not None else Image.new("RGB", CELL), "GEN ALL heat"),
        ])
        row_err = hcat([
            titled(attmap_to_pil(vis_err[i]["GENmid_x_errf"], CELL), "GENmid w x err_WOMAN"),
            titled(attmap_to_pil(vis_err[i]["GENmid_x_errm"], CELL), "GENmid w x err_MAN"),
            titled(attmap_to_pil(vis_err[i]["SCOREmid_x_errf"], CELL), "SCOREmid w x err_WOMAN"),
            titled(attmap_to_pil(vis_err[i]["SCOREmid_x_errm"], CELL), "SCOREmid w x err_MAN"),
        ])
        cap = (f"p(man):  GEN[MID]={pm(rGm):.3f} GEN[RES16]={pm(rGr):.3f} GEN[ALL]={pm(rGa):.3f}   ||   "
               f"SCORE[MID]={pm(rSm):.3f} SCORE[RES16]={pm(rSr):.3f} SCORE[ALL]={pm(rSa):.3f}   ||   "
               f"nomask={pm(results.get('nomask')):.3f}")
        capstrip = label_strip(cap, max(row_over.size[0], row_heat.size[0], row_err.size[0]), h=30)
        panel = vcat([row_over, row_heat, row_err, capstrip])
        panel.save(os.path.join(args.out_dir, "panels", f"panel_{i:02d}.png"))
        panels.append(panel)

        rec = {"idx": i}
        for key in W_KEYS:
            if key in results:
                r = results[key]
                rec[key] = {"p_woman": pw(r), "p_man": pm(r),
                            "pred": ("MAN" if r["preds"][i].item() == 1 else "WOMAN"),
                            "sds_f": float(r["sds_f"][i]), "sds_m": float(r["sds_m"][i])}
        rec["map_diag"] = {}
        for s in SCOPES:
            if GEN[s] is not None:
                rec["map_diag"][f"GEN_{s}"] = map_diagnostics(GEN[s][i])
            rec["map_diag"][f"SCORE_{s}"] = map_diagnostics(SCORE[s][i])
        per_image_json.append(rec)

    vcat(panels, pad=8).save(os.path.join(args.out_dir, "ALL_panels.png"))

    def agg(key):
        if key not in results:
            return None
        p = results[key]["probs"]; p_man = p[:, 1]
        return {"mean_max_prob": float(p.max(dim=1).values.mean()),
                "ambiguous_frac_0.35_0.65": float(((p_man > 0.35) & (p_man < 0.65)).float().mean()),
                "gender_gap": float((p_man >= 0.5).float().mean() - (p_man < 0.5).float().mean()),
                "mean_sds_f": float(results[key]["sds_f"].mean()),
                "mean_sds_m": float(results[key]["sds_m"].mean())}

    def map_stats(mp):
        if mp is None:
            return None
        ds = [map_diagnostics(mp[i]) for i in range(mp.shape[0])]
        return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}

    summary = {
        "config": vars(args),
        "axes": {"axis1_where": "GEN(생성 face 토큰) vs SCORE(채점 woman/man 토큰)",
                 "axis2_layer": "ALL(모든 cross-attn) / MID(mid_block만) / RES16(16x16만)"},
        "map_diagnostics_mean": {**{f"GEN_{s}": map_stats(GEN[s]) for s in SCOPES},
                                 **{f"SCORE_{s}": map_stats(SCORE[s]) for s in SCOPES},
                                 "note": "effective_support: 작을수록 얼굴에 집중, 4096=완전 균일(no-op)"},
        "ce_aggregate": {k: agg(k) for k in W_KEYS},
        "per_image": per_image_json,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 96)
    print(f"DONE. outputs -> {os.path.abspath(args.out_dir)}")
    print("=" * 96)
    print("\n[map spread] effective_support (작을수록 얼굴에 집중, 4096=완전 균일=사실상 no-op)")
    for k in list(summary["map_diagnostics_mean"].keys()):
        v = summary["map_diagnostics_mean"][k]
        if isinstance(v, dict) and "effective_support" in v:
            print(f"   {k:12s}: eff_support={v['effective_support']:7.1f}  frac={v['effective_support_frac']:.3f}  "
                  f"max/mean={v['max_over_mean']:.2f}")
    print("\n[CE aggregate]  mean_max_prob(↑또렷)  ambiguous(0.35~0.65,↓좋음)  gender_gap")
    for k in W_KEYS:
        a = agg(k)
        if a:
            print(f"   {k:12s}: mean_max_prob={a['mean_max_prob']:.3f}  ambiguous={a['ambiguous_frac_0.35_0.65']:.3f}  "
                  f"gap={a['gender_gap']:+.3f}  sds_f={a['mean_sds_f']:.4g} sds_m={a['mean_sds_m']:.4g}")
    print("\n[per-image p(man)]  GEN[MID] GEN[RES16] GEN[ALL] | SCORE[MID] SCORE[RES16] SCORE[ALL] | nomask")
    for r in per_image_json:
        def g(k):
            return r[k]["p_man"] if k in r else float("nan")
        print(f"   img {r['idx']:2d}: {g('GEN_MID'):.3f} {g('GEN_RES16'):.3f} {g('GEN_ALL'):.3f} | "
              f"{g('SCORE_MID'):.3f} {g('SCORE_RES16'):.3f} {g('SCORE_ALL'):.3f} | {g('nomask'):.3f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
