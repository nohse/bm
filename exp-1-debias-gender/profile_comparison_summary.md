# Profile Comparison Summary

Numbers are approximate averages from the pasted profiling logs around iter 30-35.
Times are seconds per iteration. Memory values are per GPU.

Open the visual dashboard here:

- [profile_comparison_visual.html](/root/finetune-fair-diffusion/exp-1-debias-gender/profile_comparison_visual.html)

## Time Breakdown

| Module / bucket | ftdiff | ours fullstep | ours 50% skip | Notes |
|---|---:|---:|---:|---|
| Face detect | 7.2s | - | - | ftdiff includes generated/original/backward face detection. |
| Image generation | 9.6s | 9.9s | 5.7s | no-grad generated/target + no-grad original + gradient image generation. |
| SDS target scoring no-grad | - | 2.4s | 2.4s | ours computes target logits without grad; ftdiff uses face/MobileNet path instead. |
| SDS realistic scoring | - | 3.4s | 3.4s | ours backward-side realistic scoring. |
| CLIP / DINO / MobileNet / face feature | 0.2s | 0.1s | 0.1s | ftdiff uses more feature models; ours mostly uses CLIP image feature. |
| accelerator.backward | 6.3s | 12.9s | 10.1s | Actual autograd backward; LoRA gradients are produced here. |
| gather / plot / loss sync | 3.6s | 0.2s | 0.1s | ftdiff has noisy gather/plot cost; train plots are off in ours. |
| Other small overhead | 0.1s | 0.3s | 0.1s | prompt/noise prep, weights/loss assembly, optimizer logging, etc. |
| **Measured total** | **27.0s** | **29.0s** | **21.9s** | From `total_profiled`. |

## Memory Breakdown

| Module / bucket | ftdiff | ours fullstep | ours 50% skip | Notes |
|---|---:|---:|---:|---|
| End-of-iter live `cuda_alloc` | 4.48GB | 3.77GB | 3.77GB | Live PyTorch allocations after the iter. |
| Driver/cache `cuda_device_used` | 32-34GB | ~60GB | ~52GB | Includes CUDA/PyTorch cache; do not treat as live tensor memory. |
| LoRA params | 31.6MB | 31.6MB | 31.6MB | Same trainable parameter size. |
| LoRA grads | 31.6MB | 31.6MB | 31.6MB | Actual stored gradient tensor size. |
| AdamW optimizer state | 63.3MB | 63.3MB | 63.3MB | Two Adam moments for trainable LoRA params. |
| no-grad image generation peak | ~4.1GB | ~4.1GB | ~4.1GB | Peak is similar; 50% skip mainly reduces time. |
| gradient image generation peak | 25-26.5GB | 25-26.5GB | 19-20GB | Main memory saving from 50% skip. |
| SDS target scoring no-grad peak | - | ~11.8GB | ~11.8GB | Same in full and 50% skip. |
| SDS realistic scoring peak | - | ~24.4GB | ~24.4GB | Same in full and 50% skip. |
| CLIP gradient feature peak | ~1.0GB | ~1.0GB | ~1.0GB | Similar across runs when CLIP image loss is used. |
| DINO gradient feature peak | ~190MB | - | - | ftdiff only. |
| MobileNet peak | ~90MB | - | - | ftdiff only. |
| Face detect peak | ~36MB | - | - | GPU memory is small, but time can be large/noisy. |
| gather / plot / loss sync peak | ~72MB | ~1MB | ~1MB | ftdiff image gather/plot is heavier. |
| accelerator.backward behavior | frees ~26-28GB | frees ~47-49GB | frees ~41-42GB | Negative alloc delta means saved graph memory is released. |

## Short Interpretation

- ftdiff spends a lot of wall time in face detection and gather/plot/sync, but its backward graph is lighter than ours fullstep.
- ours fullstep removes face detection from training, but adds SDS target/realistic scoring. That makes `accelerator.backward` much heavier.
- ours 50% skip is fastest because diffusion image generation and its saved backward graph are shorter. It reduces gradient image generation peak from about 25-26.5GB to about 19-20GB.
- The actual LoRA gradient memory is the same in all runs: about 31.6MB. The huge memory numbers are mostly saved activations/autograd graph, not the LoRA gradient tensor itself.
