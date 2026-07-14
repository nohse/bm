#!/usr/bin/env python
"""Low-memory cond-only scorer for the 100-image set. Stores the 15 PER-TIMESTEP
squared errors per (t-range, seed, prompt, image) so that accuracy at ANY K<=15 is a
free sub-average of the same noise. Single GPU, small batch, hard memory cap, and the
VAE/text-encoder are freed after use so only the UNet (~1.7GB fp16) stays resident."""
import os, json, argparse, importlib
import numpy as np
import torch
import error_classify as E

HERE = E.HERE
C = None  # config module, set from --cfg in main


@torch.no_grad()
def per_timestep_err(unet, z0, emb, t_lo, t_hi, seed, chunk):
    """Return [N, NT] per-timestep squared-error means for one prompt (shared noise via seed)."""
    N = z0.shape[0]
    out = np.zeros((N, C.NT), dtype=np.float32)
    for s in range(0, N, chunk):
        z0c = z0[s:s + chunk]
        eps_all, zt_all, t_all, nC = E.make_grid_noise(z0c, t_lo, t_hi, C.NT, E.SCHED, seed + s)
        eps_c = unet(zt_all, t_all, encoder_hidden_states=emb.expand(nC * C.NT, -1, -1)).sample.float()
        e = (eps_c - eps_all.float()).pow(2).mean(dim=(1, 2, 3)).view(nC, C.NT)
        out[s:s + nC] = e.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_frac", type=float, default=0.13)   # ~12.5GB cap; actual use ~7GB (headroom avoids allocator thrash)
    ap.add_argument("--chunk", type=int, default=10)          # batch = chunk*15 = 150 (peak ~9GB)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--cfg", default="exp100_config")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global C
    C = importlib.import_module(args.cfg)
    outdir = args.outdir or os.path.join(HERE, "exp100")
    if args.out is None:
        args.out = os.path.join(outdir, f"scores_shard{args.shard}.npz")
    torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)

    manifest = json.load(open(os.path.join(HERE, "exp100", "manifest_exp100.json")))
    files = [os.path.join(HERE, m["file"]) for m in manifest]
    labels = np.array([1 if m["label"] == "face" else 0 for m in manifest])

    tok, te, vae, unet, E.SCHED = E.load_models()
    # encode images in small chunks then FREE the VAE
    zs = []
    from PIL import Image
    for s in range(0, len(files), 8):
        xb = []
        for f in files[s:s + 8]:
            img = Image.open(f).convert("RGB").resize((512, 512))
            x = torch.from_numpy(np.array(img)).float().div(127.5).sub(1.0).permute(2, 0, 1)
            xb.append(x)
        xb = torch.stack(xb).to(E.DEV, E.DT)
        zs.append((vae.encode(xb).latent_dist.mean * vae.config.scaling_factor).cpu())
    z0 = torch.cat(zs).to(E.DEV)
    del vae
    torch.cuda.empty_cache()

    embeds = {k: E.embed(tok, te, v) for k, v in {**C.FACE_PROMPTS, **C.NONFACE_PROMPTS}.items()}
    del te
    torch.cuda.empty_cache()

    # shard over (t-range, seed) jobs so all GPUs are used even with few t-ranges
    jobs = [(tn, sd) for tn in C.TRANGES for sd in C.SEEDS][args.shard::args.nshards]
    store = {}                       # "trange|seed|prompt" -> [N,NT]
    for tn, sd in jobs:
        t_lo, t_hi = C.TRANGES[tn]
        for pk, emb in embeds.items():
            store[f"{tn}|{sd}|{pk}"] = per_timestep_err(unet, z0, emb, t_lo, t_hi, sd, args.chunk)
        print(f"[score100] {tn} seed {sd} done  (peak GPU "
              f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB)", flush=True)
        np.savez_compressed(args.out, labels=labels, **store)   # incremental save

    np.savez_compressed(args.out, labels=labels, **store)
    print(f"[score100] saved {args.out}  ({len(store)} arrays); peak GPU "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
