"""
사용자 요청 토이 실험: ε-SCR loss가 "내용 차이"가 아니라 "비현실성"만 재는 자임을 보인다.

  Case R (realistic vs realistic): "a photo of a doctor" 를 다른 seed 로 2장 -> 완전히 다른 사람.
  Case N (realistic vs weird-noise): 의사 A 에 이상한 노이즈를 더한 그림. latent 거리는 Case R 과 동일하게 맞춤.

학습 코드와 동일한 ε-SCR 계산:
  같은 eps 로 두 latent 를 같은 t 로 re-noise -> frozen UNet 의 eps 예측(.sample) 을 MSE.
  t grid = linspace(400,800,15) (학습과 동일), eps 8 draws/t 평균, 생성-프롬프트 conditioning.
대조로 h-space(mid_block) MSE 도 같이 계산.
"""
import os, json, math
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler

DEVICE = "cuda:1"
MODEL = "runwayml/stable-diffusion-v1-5"
OUT = "/tmp/claude-0/-workspace/72ee5bcd-f241-4087-968d-f265162f6d5e/scratchpad/toy_escr_demo"
os.makedirs(OUT, exist_ok=True)
PROMPT = "a photo of a doctor"
T_GRID = torch.linspace(400, 800, 15).round().long().tolist()   # 학습과 동일
N_EPS = 8                                                        # t 당 eps 샘플 수
DOCTOR_SEEDS = [1, 2, 3, 4]
SCALE = 0.18215

torch.manual_seed(0)
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.to(DEVICE); pipe.set_progress_bar_config(disable=True)
unet = pipe.unet; unet.eval()
vae = pipe.vae
ddpm = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
ac = ddpm.alphas_cumprod.to(DEVICE, torch.float32)

_h = {}
unet.mid_block.register_forward_hook(lambda m, i, o: _h.__setitem__("h", o))

def text_embed(prompt):
    tok = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    with torch.no_grad():
        return pipe.text_encoder(tok.input_ids.to(DEVICE))[0].to(torch.float16)

EMB = text_embed(PROMPT)

def eps_and_h(zt):  # zt fp32 [B,4,64,64] -> eps_hat fp32, h fp32
    B = zt.shape[0]
    tt = torch.full((B,), TT[0], device=DEVICE, dtype=torch.long)
    out = unet(zt.to(torch.float16), tt, encoder_hidden_states=EMB.expand(B, -1, -1)).sample
    return out.float(), _h["h"].float()

def scr_errors(z0_x, z0_y):
    """학습과 동일한 ε-SCR 및 h-space MSE. t grid × eps draws 평균 스칼라 2개 반환."""
    global TT
    eps_errs, h_errs = [], []
    for t in T_GRID:
        TT = [t]
        ab = ac[t]; sab = ab.sqrt(); s1 = (1 - ab).sqrt()
        for k in range(N_EPS):
            g = torch.Generator(device=DEVICE).manual_seed(10_000 * t + k)
            eps = torch.randn(z0_x.shape, generator=g, device=DEVICE, dtype=torch.float32)
            zt_x = sab * z0_x + s1 * eps
            zt_y = sab * z0_y + s1 * eps
            with torch.no_grad():
                ex, hx = eps_and_h(zt_x)
                ey, hy = eps_and_h(zt_y)
            eps_errs.append(((ex - ey) ** 2).mean().item())
            h_errs.append(((hx - hy) ** 2).mean().item())
    return sum(eps_errs) / len(eps_errs), sum(h_errs) / len(h_errs)

# ---- 1) 의사 이미지 4장 (다른 seed) ----
print("generating doctor latents ...")
docs = {}
for s in DOCTOR_SEEDS:
    g = torch.Generator(device=DEVICE).manual_seed(s)
    with torch.no_grad():
        out = pipe(PROMPT, num_inference_steps=30, guidance_scale=7.5, generator=g, output_type="latent")
    docs[s] = out.images[0].unsqueeze(0).float()
    print(f"  doctor seed {s}: latent norm {docs[s].norm().item():.2f}")

# 의사-의사 latent 거리 (평균) -> 노이즈 그림의 거리를 여기에 맞춘다
dd_dists = []
seeds = DOCTOR_SEEDS
pairs = [(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
for a, b in pairs:
    dd_dists.append((docs[a] - docs[b]).norm().item())
d_match = sum(dd_dists) / len(dd_dists)
print(f"의사-의사 평균 latent 거리 = {d_match:.2f}")

# ---- 2) 이상한 노이즈 그림 ----
zA = docs[seeds[0]]
# (a) 거리 맞춘 노이즈: 의사 A + (같은 norm 의 가우시안)  -> 같은 거리, 하지만 off-manifold
noisy = {}
for k in range(4):
    g = torch.Generator(device=DEVICE).manual_seed(50_000 + k)
    n = torch.randn(zA.shape, generator=g, device=DEVICE, dtype=torch.float32)
    n = n * (d_match / n.norm())
    noisy[k] = zA + n
# (b) 순수 노이즈 그림 (문자 그대로 "이상한 노이즈 그림", 거리 안 맞춤 — 참고용)
g = torch.Generator(device=DEVICE).manual_seed(99_999)
pure = torch.randn(zA.shape, generator=g, device=DEVICE, dtype=torch.float32) * zA.std() + zA.mean()

# ---- 3) ε-SCR / h-space 계산 ----
print("\ncomputing ε-SCR errors ...")
rows = []
# Case R: 의사-의사 (모든 쌍)
R_eps, R_h = [], []
for a, b in pairs:
    e, h = scr_errors(docs[a], docs[b])
    R_eps.append(e); R_h.append(h)
    print(f"  [R] doctor{a} vs doctor{b}: latentdist={ (docs[a]-docs[b]).norm().item():.1f}  ε-SCR={e:.4f}  h-SCR={h:.3f}")
# Case N: 의사 A vs 거리맞춘 노이즈
N_eps, N_h = [], []
for k in range(4):
    e, h = scr_errors(zA, noisy[k])
    N_eps.append(e); N_h.append(h)
    print(f"  [N] doctorA vs weird-noise#{k}: latentdist={(zA-noisy[k]).norm().item():.1f}  ε-SCR={e:.4f}  h-SCR={h:.3f}")
# Case P: 의사 A vs 순수 노이즈 그림
P_eps, P_h = scr_errors(zA, pure)
print(f"  [P] doctorA vs pure-noise-image: latentdist={(zA-pure).norm().item():.1f}  ε-SCR={P_eps:.4f}  h-SCR={P_h:.3f}")

mean = lambda x: sum(x) / len(x)
summary = {
    "prompt": PROMPT, "t_grid": T_GRID, "eps_draws_per_t": N_EPS,
    "doctor_doctor_latent_dist_mean": d_match,
    "case_R_realistic_vs_realistic": {
        "desc": "다른 seed 의사 2장 (완전히 다른 사람) — 둘 다 사실적",
        "latent_dist": mean([(docs[a]-docs[b]).norm().item() for a,b in pairs]),
        "eps_scr_mean": mean(R_eps), "eps_scr_all": R_eps,
        "h_scr_mean": mean(R_h),
    },
    "case_N_realistic_vs_weirdnoise_MATCHED_dist": {
        "desc": "의사 A + 같은 norm 가우시안 노이즈 (latent 거리는 Case R 과 동일, 하지만 비현실적)",
        "latent_dist": mean([(zA-noisy[k]).norm().item() for k in range(4)]),
        "eps_scr_mean": mean(N_eps), "eps_scr_all": N_eps,
        "h_scr_mean": mean(N_h),
    },
    "case_P_realistic_vs_purenoise": {
        "desc": "의사 A vs 순수 노이즈 그림 (거리 안 맞춤, 참고용)",
        "eps_scr": P_eps, "h_scr": P_h,
    },
    "headline": {
        "eps_scr_R": mean(R_eps), "eps_scr_N": mean(N_eps),
        "eps_ratio_N_over_R": mean(N_eps) / mean(R_eps),
        "h_scr_R": mean(R_h), "h_scr_N": mean(N_h),
        "h_ratio_N_over_R": mean(N_h) / mean(R_h),
        "training_scr_floor": 0.028,
    },
}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ---- 4) 이미지 디코드 & 패널 저장 ----
def decode(z):
    with torch.no_grad():
        img = vae.decode(z.to(torch.float16) / SCALE).sample[0]
    img = (img.float().clamp(-1, 1) + 1) / 2
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

imgA = decode(docs[seeds[0]]); imgB = decode(docs[seeds[1]])
imgN = decode(noisy[0]); imgP = decode(pure)

fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[3, 2])
titles = [
    ("doctor A (seed 1)\nreference", imgA, None),
    (f"doctor B (seed 2)\nREALISTIC, different person\nlatent dist={summary['case_R_realistic_vs_realistic']['latent_dist']:.0f}\neps-SCR = {mean(R_eps):.4f}", imgB, "tab:green"),
    (f"weird-noise image\nSAME latent dist={summary['case_N_realistic_vs_weirdnoise_MATCHED_dist']['latent_dist']:.0f}\neps-SCR = {mean(N_eps):.4f}", imgN, "tab:red"),
    (f"pure noise image\neps-SCR = {P_eps:.4f}", imgP, "tab:red"),
]
for i, (t, im, c) in enumerate(titles):
    ax = fig.add_subplot(gs[0, i]); ax.imshow(im); ax.axis("off")
    ax.set_title(t, fontsize=10, color=(c or "black"))

# eps-SCR bar
ax1 = fig.add_subplot(gs[1, :2])
b = ax1.bar(["R: doctor vs doctor\n(realistic, diff person)", "N: doctor vs weird-noise\n(SAME latent dist)"],
            [mean(R_eps), mean(N_eps)], color=["tab:green", "tab:red"])
ax1.axhline(0.028, ls="--", color="gray"); ax1.text(1.4, 0.030, "training SCR floor ~0.028", fontsize=8, color="gray")
ax1.set_ylabel("eps-SCR error"); ax1.set_title(f"eps-SCR: SAME latent distance, yet N/R = {mean(N_eps)/mean(R_eps):.1f}x", fontsize=11)
for r in b: ax1.text(r.get_x()+r.get_width()/2, r.get_height(), f"{r.get_height():.4f}", ha="center", va="bottom", fontsize=10)

# h-space bar (control)
ax2 = fig.add_subplot(gs[1, 2:])
b2 = ax2.bar(["R: doctor vs doctor", "N: doctor vs weird-noise"], [mean(R_h), mean(N_h)], color=["tab:green", "tab:red"])
ax2.set_ylabel("h-space (mid_block) error"); ax2.set_title(f"control: h-space ranks R > N  (N/R = {mean(N_h)/mean(R_h):.1f}x)", fontsize=11)
for r in b2: ax2.text(r.get_x()+r.get_width()/2, r.get_height(), f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=10)

fig.suptitle("eps-SCR barely sees CONTENT change (doctor A vs B) = near training floor. It only penalizes UNREALISM.", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "panel.png"), dpi=110)
print("\nsaved:", os.path.join(OUT, "panel.png"))
print(json.dumps(summary["headline"], indent=2, ensure_ascii=False))
