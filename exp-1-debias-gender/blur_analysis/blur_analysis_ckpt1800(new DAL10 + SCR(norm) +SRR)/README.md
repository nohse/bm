# Blur / fairness diagnostic for checkpoint-1800 (TE-only LoRA, region-attn run)

Reproduces eval of
`outputs/gender_aaai/20260628-2336_gender_aaai_region-attn_skip-50pct_wImg-1.0_wRealFace-4.0_wFair-10.0_fcw-1.0_Th-0.2_lr-5e-05/ckpts/checkpoint-1800`
by applying the EMA TE-LoRA to base SD-1.5 (frozen UNet/VAE) and comparing **ori (base TE)** vs
**finetune (EMA-LoRA TE)** on the **same noise**.

## Files
- `repro_exp.py`   — standalone experiment (no dependency on the training script).
- `results.json`   — aggregated metrics for `full25` (eval condition) and `skip50` (training condition).
- `run.log`        — per-prompt log (realistic loss + Laplacian variance for every prompt).
- `montage_full25.png` / `montage_skip50.png` — left = ori (base), right = finetune (LoRA), 6 prompts.

## How to regenerate
```bash
cd exp-1-debias-gender
CKPT="outputs/gender_aaai/20260628-2336_gender_aaai_region-attn_skip-50pct_wImg-1.0_wRealFace-4.0_wFair-10.0_fcw-1.0_Th-0.2_lr-5e-05/ckpts/checkpoint-1800"
CUDA_VISIBLE_DEVICES=2 python blur_analysis_ckpt1800/repro_exp.py \
    --ckpt "$CKPT" --out blur_analysis_ckpt1800/full \
    --n_prompts 10 --n_per 5 --num_eps 4 --also_skip
```

## Key findings (full 25-step = eval condition, n=50)
| metric | ori (base) | finetune (LoRA) |
|---|---|---|
| realistic-SDS loss | 0.0748 | **0.0371** (블러일수록 ↓) |
| Laplacian variance (선명도) | 2981 | 524 |
| high-freq energy ratio | 0.0238 | 0.0017 |
| s_f (woman SDS err) ≈ s_m (man SDS err) | 0.0748 ≈ 0.0748 | 0.0370 ≈ 0.0370 |
| relative \|s_f−s_m\| separation | 0.24% | 0.73% |
| male ratio (SDS argmax) | 0.68 | 0.40 |

- **realistic loss rewards blur** (sharp 0.075 → blurry 0.037).
- **s_f ≈ s_m**: SDS classifier can't actually tell genders apart -> "50/50" is a coin flip, not real balance.

## Caveats
- SDS gender uses `region_mask_mode='none'` (unmasked); the production run used `attn` masking.
  The collapse `s_f ≈ s_m` is mask-independent in spirit, but absolute numbers differ from wandb.
- LoRA TE runs in fp32, others fp16 (training kept base fp16 / LoRA fp32); embeds cast to fp16 for UNet.
- Realistic SDS absolute value (~0.037) is the same order as wandb (~0.014-0.02) but not perfectly
  calibrated (generation path / step-count differences); the **ori-vs-ft direction** is the robust signal.
