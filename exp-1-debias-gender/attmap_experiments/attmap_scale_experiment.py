#!/usr/bin/env python
# coding=utf-8
"""
attmap_scale_experiment.py

목적
----
06281-...-check.py  와  06291-...-check_att.py 두 파일의 유일한 차이는
sds_logits_from_images() 안에서 attmap(w)을 residual error에 곱하는 "한 줄"이다.

  check.py:  err = (eps_pred - eps)^2 ;  w = w_hat * HW (mean weight 1)
             sds = mean_spatial( w * err )         =  Σ_s  w_hat_s   · res_s^2     (w_hat 합=1, 1제곱)

  att.py:    res = eps_pred - eps ;       w = w_hat (공간합=1)
             sds = sum_spatial( (res*w)^2 ) (채널평균) = Σ_s  w_hat_s^2 · res_s^2   (w_hat 2제곱)

즉 att.py 는 attention 가중치가 w_hat -> w_hat^2 로 "제곱"되어 error 의 scale 이 작아진다.
이 스크립트는 동일한 100장의 생성 이미지 / 동일한 eps / 동일한 eps_pred / 동일한 w_map 으로
두 reduction 만 각각 적용해 female error, male error 의 mean/max/min 을 구해 비교한다.
(원본 두 파일에서 w_map 을 만드는 코드는 글자 그대로 동일하므로 한 번만 계산해 공유한다.)
"""

import os, json, math, argparse, random
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, DDPMScheduler


# =====================================================================================
# 원본 파일에서 그대로 복사한 CrossAttnCapture (attmap 수집기) -- 두 파일에서 byte 동일
# =====================================================================================
class CrossAttnCapture:
    def __init__(self, token_indices: List[int], use_cpu: bool = False, expect_cfg_pair: bool = True):
        self.token_indices = token_indices
        self.handles = []
        self.maps = []
        self.use_cpu = use_cpu
        self._q_cache = {}
        self._parent_map = {}
        self.use_cpu = use_cpu
        self.expect_cfg_pair = expect_cfg_pair

    def _q_hook(self, module, inputs, output):
        parent = self._parent_map.get(id(module))
        if parent is None:
            return
        parent_id = id(parent)
        try:
            with torch.no_grad():
                self._q_cache[parent_id] = output.detach().clone()
        except Exception:
            self._q_cache[parent_id] = output.detach()

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
            try:
                q = q.detach().float().to(device)
            except Exception:
                q = q.float().to(device)
            try:
                k = k.detach().float().to(device)
            except Exception:
                k = k.float().to(device)

            heads = getattr(parent, "heads", None)
            if heads is None:
                heads = getattr(parent, "num_heads", None)

            B, Nq, inner_q = q.shape
            _, Nk, inner_k = k.shape
            try:
                assert inner_q == inner_k
            except Exception:
                return

            if heads is None:
                for h in (8, 12, 16, 4, 6, 24, 32):
                    if inner_q % h == 0:
                        heads = h
                        break
                if heads is None:
                    heads = 8

            head_dim = inner_q // heads
            q = q.view(B, Nq, heads, head_dim).permute(0, 2, 1, 3).contiguous()
            k = k.view(B, Nk, heads, head_dim).permute(0, 2, 1, 3).contiguous()

            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
            attn = attn_scores.softmax(dim=-1)

            if not self.token_indices:
                return
            tok_idx = torch.tensor(self.token_indices, device=device, dtype=torch.long)
            attn_tok = attn.index_select(-1, tok_idx).mean(dim=-1)  # (B, H, Nq)

            if self.expect_cfg_pair and B >= 2:
                b_half = B // 2
                cond = attn_tok[B - b_half: B]
            else:
                cond = attn_tok

            per_image = cond.mean(dim=1)  # (B_sel, Nq)
            hw = int(math.sqrt(per_image.shape[1]))
            if hw * hw != per_image.shape[1]:
                return
            b_sel = per_image.shape[0]
            per_image = per_image.view(b_sel, hw, hw)
            if self.use_cpu:
                per_image = per_image.cpu()
            self.maps.append(per_image)

    def add_hooks(self, unet: torch.nn.Module):
        for name, module in unet.named_modules():
            has_qk = hasattr(module, "to_q") and hasattr(module, "to_k")
            if not has_qk:
                continue
            is_cross = getattr(module, "is_cross_attention", None)
            if is_cross is None:
                is_cross = ("attn2" in name) or ("Cross" in module.__class__.__name__)
            if not is_cross:
                continue
            try:
                self._parent_map[id(module.to_q)] = module
            except Exception:
                pass
            try:
                self._parent_map[id(module.to_k)] = module
            except Exception:
                pass
            self.handles.append(module.to_q.register_forward_hook(self._q_hook))
            self.handles.append(module.to_k.register_forward_hook(self._k_hook))
        return self

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self._q_cache = {}
        self.maps = []

    def aggregated_map(self) -> Optional[torch.Tensor]:
        if not self.maps:
            return None
        max_hw = max(m.shape[-1] for m in self.maps)
        upsampled = []
        for m in self.maps:
            Bm, hm, wm = m.shape
            ten = m.unsqueeze(1).float()
            if hm != max_hw or wm != max_hw:
                ten = F.interpolate(ten, size=(max_hw, max_hw), mode="bilinear", align_corners=False)
            upsampled.append(ten.squeeze(1))
        S = torch.stack(upsampled, dim=0)
        M = S.mean(dim=0)
        return M


def find_token_positions(tokenizer, prompt: str, keywords: List[str]) -> List[int]:
    toks = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                     truncation=True, return_tensors="pt")
    ids = toks.input_ids[0]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    kws = [k.lower() for k in keywords]
    pos = []
    for i, piece in enumerate(pieces):
        p = piece.lower().replace("Ġ", "").replace("▁", "")
        if any(k in p for k in kws):
            pos.append(i)
    return pos


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--gen_batch", type=int, default=10, help="이미지 생성 배치")
    ap.add_argument("--sds_chunk", type=int, default=10, help="SDS forward 시 한번에 처리할 이미지 수")
    ap.add_argument("--num_denoising_steps", type=int, default=25)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--sds_t_min", type=int, default=400)
    ap.add_argument("--sds_t_max", type=int, default=800)
    ap.add_argument("--sds_num_t", type=int, default=15)
    ap.add_argument("--sds_num_eps", type=int, default=1)
    ap.add_argument("--sds_tau", type=float, default=0.0001)
    ap.add_argument("--seed", type=int, default=5991)
    ap.add_argument("--prompts_json", type=str, default="../data/1-prompts/occupation.json")
    ap.add_argument("--out_dir", type=str, default="./attmap_scale_out")
    ap.add_argument("--save_images", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    weight_dtype = torch.float16            # 원본 mixed_precision="fp16"
    weight_dtype_high = torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[load] loading SD1.5 components ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(device, weight_dtype).eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device, weight_dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(device, weight_dtype).eval()
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model, subfolder="scheduler")
    ddpm_forward = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    # ---- prompts ----
    with open(args.prompts_json) as f:
        pj = json.load(f)
    template = pj["prompt_templates_test"][0]
    occ = pj["occupations_val_set"]
    prompts = [template.format(occupation=o) for o in occ]
    # n_images 를 채우도록 prompt 를 반복
    per_prompt = max(1, math.ceil(args.n_images / len(prompts)))

    # =================================================================================
    # 1) 이미지 생성 (generate_image_no_gradient 와 동일 로직)
    # =================================================================================
    @torch.no_grad()
    def generate(prompt, noises):
        N = noises.shape[0]
        toks = tokenizer([prompt] * N, return_tensors="pt", padding=True)
        ids = toks["input_ids"].to(device); am = toks["attention_mask"].to(device)
        pe = text_encoder(ids, am)[0]
        bs = pe.shape[0]; max_len = pe.shape[1]
        un = tokenizer([""] * bs, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
        npe = text_encoder(un["input_ids"].to(device), un["attention_mask"].to(device))[0]
        pe = torch.cat([npe, pe]).to(weight_dtype)

        noise_scheduler.set_timesteps(args.num_denoising_steps)
        latents = noises
        for t in noise_scheduler.timesteps:
            lmi = torch.cat([latents.to(weight_dtype)] * 2)
            lmi = noise_scheduler.scale_model_input(lmi, t)
            npred = unet(lmi, t, encoder_hidden_states=pe).sample.to(weight_dtype_high)
            nu, nt = npred.chunk(2)
            npred = nu + args.guidance_scale * (nt - nu)
            latents = noise_scheduler.step(npred, t, latents).prev_sample
        lat = 1 / vae.config.scaling_factor * latents
        imgs = vae.decode(lat.to(vae.dtype)).sample.clamp(-1, 1)
        return imgs

    print(f"[gen] generating {args.n_images} images ...")
    all_imgs = []
    img_meta = []
    g = torch.Generator(device=device).manual_seed(args.seed)
    pi = 0
    while len(all_imgs) < args.n_images:
        prompt = prompts[pi % len(prompts)]
        nthis = min(args.gen_batch, args.n_images - len(all_imgs))
        noises = torch.randn((nthis, 4, 64, 64), generator=g, device=device, dtype=weight_dtype)
        imgs = generate(prompt, noises)
        for k in range(nthis):
            all_imgs.append(imgs[k].detach())
            img_meta.append(prompt)
        pi += 1
        print(f"   {len(all_imgs)}/{args.n_images}")
    images = torch.stack(all_imgs, dim=0)  # [N,3,512,512] in [-1,1], fp16

    if args.save_images:
        imgdir = os.path.join(args.out_dir, "images")
        os.makedirs(imgdir, exist_ok=True)
        for i in range(images.shape[0]):
            arr = ((images[i].float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(os.path.join(imgdir, f"img_{i:03d}.png"))
        # montage 10x10
        thumbs = []
        for i in range(images.shape[0]):
            arr = ((images[i].float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            thumbs.append(np.array(Image.fromarray(arr).resize((128, 128))))
        cols = 10; rows = math.ceil(len(thumbs) / cols)
        canvas = np.zeros((rows * 128, cols * 128, 3), dtype=np.uint8)
        for i, th in enumerate(thumbs):
            r, c = divmod(i, cols)
            canvas[r*128:(r+1)*128, c*128:(c+1)*128] = th
        Image.fromarray(canvas).save(os.path.join(args.out_dir, "montage.png"))

    # =================================================================================
    # 2) SDS forward (한 번) -> eps, eps_pred_f, eps_pred_m, w_map  공유
    #    그 뒤 check / att reduction 을 각각 적용
    # =================================================================================
    female_prompt = "a photo of a woman"
    male_prompt = "a photo of a man"

    def text_embeds(prompts_list):
        toks = tokenizer(prompts_list, padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
        return text_encoder(toks.input_ids.to(device), toks.attention_mask.to(device))[0]

    tok_pos_f = find_token_positions(tokenizer, female_prompt, keywords=["woman"])
    tok_pos_m = find_token_positions(tokenizer, male_prompt, keywords=["man"])
    print("[sds] tok_pos_f", tok_pos_f, "tok_pos_m", tok_pos_m)

    # 결과 누적 (per-image, fp32)
    res = {k: {"check": {"f": [], "m": []}, "att": {"f": [], "m": []},
               "check_fp32": {"f": [], "m": []}, "att_fp32": {"f": [], "m": []}}
           for k in ["all"]}
    sum_w2_list = []   # Σ_s w_hat_s^2  per image (effective support 진단용)

    N = images.shape[0]
    K = args.sds_num_t
    num_eps = args.sds_num_eps
    tau = args.sds_tau

    @torch.no_grad()
    def forward_with_attn(zt, t_vec, prompt_embeds, token_positions):
        cap = CrossAttnCapture(token_indices=token_positions, use_cpu=False, expect_cfg_pair=False).add_hooks(unet)
        try:
            eps_pred = unet(zt.to(weight_dtype), t_vec, encoder_hidden_states=prompt_embeds).sample.to(weight_dtype)
            att_bke = cap.aggregated_map()
        finally:
            cap.clear()
        return eps_pred, att_bke

    for start in range(0, N, args.sds_chunk):
        end = min(start + args.sds_chunk, N)
        imgs = images[start:end].to(weight_dtype)
        B = imgs.shape[0]
        torch.manual_seed(args.seed + start)  # eps 재현성

        lat = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
        t_idx = torch.linspace(args.sds_t_min, args.sds_t_max, steps=K, device=device).round().long()
        lat_exp = lat.unsqueeze(1).unsqueeze(2).expand(B, K, num_eps, *lat.shape[1:]).contiguous().view(B*K*num_eps, *lat.shape[1:])
        t_vec = t_idx.view(1, K, 1).expand(B, K, num_eps).reshape(-1)
        BKE = lat_exp.shape[0]
        eps = torch.randn_like(lat_exp, dtype=weight_dtype)
        alpha_bar = ddpm_forward.alphas_cumprod.to(device=device, dtype=lat_exp.dtype)[t_vec]
        zt = alpha_bar.sqrt().view(-1, 1, 1, 1) * lat_exp + (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1) * eps

        pe_f = text_embeds([female_prompt] * BKE).to(weight_dtype)
        pe_m = text_embeds([male_prompt] * BKE).to(weight_dtype)

        eps_pred_f, att_f_bke = forward_with_attn(zt, t_vec, pe_f, tok_pos_f)
        eps_pred_m, att_m_bke = forward_with_attn(zt, t_vec, pe_m, tok_pos_m)

        _H, _W = zt.shape[-2], zt.shape[-1]
        # ---- w_map 구성 (두 파일에서 동일한 코드) ----
        a_f = att_f_bke.to(device=device, dtype=torch.float32)
        a_m = att_m_bke.to(device=device, dtype=torch.float32)
        if a_f.shape[-2:] != (_H, _W):
            a_f = F.interpolate(a_f.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False).squeeze(1)
        if a_m.shape[-2:] != (_H, _W):
            a_m = F.interpolate(a_m.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False).squeeze(1)
        att_f = a_f.reshape(B, K, num_eps, _H, _W).mean(dim=(1, 2))
        att_m = a_m.reshape(B, K, num_eps, _H, _W).mean(dim=(1, 2))
        attmap_mean = 0.5 * (att_f + att_m)
        w_map = (attmap_mean.unsqueeze(1).unsqueeze(2).expand(B, K, num_eps, _H, _W)
                 .contiguous().view(BKE, _H, _W).to(weight_dtype))

        # =========================================================================
        # (A) check.py reduction  (fp16, 원본 그대로)
        # =========================================================================
        err_f = ((eps_pred_f - eps) ** 2).mean(dim=1)   # [BKE,H,W]
        err_m = ((eps_pred_m - eps) ** 2).mean(dim=1)
        w = w_map.to(err_f.dtype).clamp_min(0)
        w_sum = w.flatten(1).sum(dim=1).clamp_min(1e-8)
        hw = err_f.shape[-2] * err_f.shape[-1]
        w_chk = (w / w_sum.view(-1, 1, 1)) * hw
        sds_f_check = (err_f * w_chk).flatten(1).mean(dim=1)
        sds_m_check = (err_m * w_chk).flatten(1).mean(dim=1)

        # =========================================================================
        # (B) att.py reduction  (fp16, 원본 그대로)
        # =========================================================================
        res_f = eps_pred_f - eps
        res_m = eps_pred_m - eps
        w_att = w_map.to(res_f.dtype).clamp_min(0)
        w_att = w_att / w_att.flatten(1).sum(dim=1).clamp_min(1e-8).view(-1, 1, 1)  # w_hat 공간합=1
        wc = w_att.unsqueeze(1)
        sds_f_att = ((res_f * wc) ** 2).mean(dim=1).flatten(1).sum(dim=1)
        sds_m_att = ((res_m * wc) ** 2).mean(dim=1).flatten(1).sum(dim=1)

        # =========================================================================
        # (C) fp32 버전 (fp16 underflow 영향 배제, 순수 공식 scale 비교용)
        # =========================================================================
        epf = eps_pred_f.float(); epm = eps_pred_m.float(); ep = eps.float()
        wf = w_map.float().clamp_min(0)
        wsum32 = wf.flatten(1).sum(dim=1).clamp_min(1e-8)
        what = wf / wsum32.view(-1, 1, 1)                 # 공간합=1
        err_f32 = ((epf - ep) ** 2).mean(dim=1)
        err_m32 = ((epm - ep) ** 2).mean(dim=1)
        sds_f_check32 = (err_f32 * (what * hw)).flatten(1).mean(dim=1)
        sds_m_check32 = (err_m32 * (what * hw)).flatten(1).mean(dim=1)
        rf = epf - ep; rm = epm - ep
        sds_f_att32 = ((rf * what.unsqueeze(1)) ** 2).mean(dim=1).flatten(1).sum(dim=1)
        sds_m_att32 = ((rm * what.unsqueeze(1)) ** 2).mean(dim=1).flatten(1).sum(dim=1)

        # per-image (K,E 평균) -> [B]
        def red(x):
            return x.float().view(B, K, num_eps).mean(dim=(1, 2))
        res["all"]["check"]["f"].append(red(sds_f_check))
        res["all"]["check"]["m"].append(red(sds_m_check))
        res["all"]["att"]["f"].append(red(sds_f_att))
        res["all"]["att"]["m"].append(red(sds_m_att))
        res["all"]["check_fp32"]["f"].append(red(sds_f_check32))
        res["all"]["check_fp32"]["m"].append(red(sds_m_check32))
        res["all"]["att_fp32"]["f"].append(red(sds_f_att32))
        res["all"]["att_fp32"]["m"].append(red(sds_m_att32))

        # Σ w_hat^2  per (BKE) -> [B] (effective support 진단)
        sumw2 = (what ** 2).flatten(1).sum(dim=1).view(B, K, num_eps).mean(dim=(1, 2))
        sum_w2_list.append(sumw2)

        print(f"   [sds] images {start}:{end} done")

    # =================================================================================
    # 3) 집계 & JSON 저장
    # =================================================================================
    def stats(t):
        t = t.float().cpu()
        return {"mean": float(t.mean()), "max": float(t.max()), "min": float(t.min()),
                "std": float(t.std()), "median": float(t.median())}

    def cat(method, fm):
        return torch.cat(res["all"][method][fm], dim=0)

    out = {
        "config": {
            "n_images": int(images.shape[0]),
            "model": args.model,
            "num_denoising_steps": args.num_denoising_steps,
            "guidance_scale": args.guidance_scale,
            "sds_t_min": args.sds_t_min, "sds_t_max": args.sds_t_max,
            "sds_num_t": args.sds_num_t, "sds_num_eps": args.sds_num_eps,
            "sds_tau": args.sds_tau, "weight_dtype": "float16",
            "region_mask_mode": "attn", "seed": args.seed,
            "note": "female error = sds_f (a photo of a woman), male error = sds_m (a photo of a man). "
                    "Same images / eps / eps_pred / w_map; only the attmap multiplication differs.",
        },
        "difference": {
            "check_formula": "sds = sum_s w_hat_s * res_s^2   (w_hat sum=1, weight power 1)",
            "att_formula":   "sds = sum_s w_hat_s^2 * res_s^2  (w_hat sum=1, weight power 2 -> smaller)",
        },
    }

    sw2 = torch.cat(sum_w2_list, dim=0)
    out["attmap_diagnostics"] = {
        "mean_sum_w_hat_squared": float(sw2.mean()),
        "effective_support_1_over_sum_w2_mean": float((1.0 / sw2).mean()),
        "comment": "att/check scale ratio ~ sum_s w_hat^2 * res^2 / sum_s w_hat * res^2; "
                   "if res^2 were uniform over space this equals sum w_hat^2 = 1/effective_support.",
    }

    for method in ["check", "att", "check_fp32", "att_fp32"]:
        out[method] = {
            "female_error": stats(cat(method, "f")),
            "male_error": stats(cat(method, "m")),
        }

    # ratio att/check (per-image, fp16 path & fp32 path)
    eps_div = 1e-30
    def ratio(num_m, den_m, fm):
        n = cat(num_m, fm); d = cat(den_m, fm)
        r = (n / (d + eps_div)).float().cpu()
        return {"mean": float(r.mean()), "max": float(r.max()), "min": float(r.min()), "median": float(r.median())}
    out["ratio_att_over_check_fp16"] = {
        "female": ratio("att", "check", "f"), "male": ratio("att", "check", "m"),
    }
    out["ratio_att_over_check_fp32"] = {
        "female": ratio("att_fp32", "check_fp32", "f"), "male": ratio("att_fp32", "check_fp32", "m"),
    }

    # per-image arrays (참고용)
    out["per_image"] = {
        "prompt": img_meta,
        "check_female": cat("check", "f").float().cpu().tolist(),
        "check_male": cat("check", "m").float().cpu().tolist(),
        "att_female": cat("att", "f").float().cpu().tolist(),
        "att_male": cat("att", "m").float().cpu().tolist(),
        "check_fp32_female": cat("check_fp32", "f").float().cpu().tolist(),
        "att_fp32_female": cat("att_fp32", "f").float().cpu().tolist(),
    }

    out_path = os.path.join(args.out_dir, "attmap_scale_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- 콘솔 요약 ----
    print("\n" + "=" * 78)
    print(f"RESULT  (n_images={images.shape[0]})   saved -> {out_path}")
    print("=" * 78)
    for method in ["check", "att", "check_fp32", "att_fp32"]:
        fe = out[method]["female_error"]; me = out[method]["male_error"]
        print(f"\n[{method}]")
        print(f"  female error  mean={fe['mean']:.6g}  max={fe['max']:.6g}  min={fe['min']:.6g}")
        print(f"  male   error  mean={me['mean']:.6g}  max={me['max']:.6g}  min={me['min']:.6g}")
    print("\n[att/check ratio  (fp16)]")
    print("  female mean ratio = {:.6g}".format(out["ratio_att_over_check_fp16"]["female"]["mean"]))
    print("  male   mean ratio = {:.6g}".format(out["ratio_att_over_check_fp16"]["male"]["mean"]))
    print("[att/check ratio  (fp32, pure formula)]")
    print("  female mean ratio = {:.6g}".format(out["ratio_att_over_check_fp32"]["female"]["mean"]))
    print("  male   mean ratio = {:.6g}".format(out["ratio_att_over_check_fp32"]["male"]["mean"]))
    print("\n[attmap]  mean Σŵ² = {:.6g}   effective support(1/Σŵ²) = {:.6g}".format(
        out["attmap_diagnostics"]["mean_sum_w_hat_squared"],
        out["attmap_diagnostics"]["effective_support_1_over_sum_w2_mean"]))
    print("=" * 78)


if __name__ == "__main__":
    main()
