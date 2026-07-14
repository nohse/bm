#!/usr/bin/env python
"""One GPU worker for the cond-only t-range x prompt search. Handles a SHARD of
t-ranges; for each, scores every prompt's cond error on the 40 TRUE images for
every seed. Saves per-seed E so the aggregator can find robust 40/40 configs."""
import os, json, argparse
import numpy as np
import torch
import error_classify as E
import search_config as SC


@torch.no_grad()
def score_cond(unet, z0, embeds, t_lo, t_hi, K, seed, chunk):
    N = z0.shape[0]
    out = {name: np.zeros(N) for name in embeds}
    for s in range(0, N, chunk):
        z0c = z0[s:s + chunk]
        eps_all, zt_all, t_all, nC = E.make_grid_noise(z0c, t_lo, t_hi, K, E.SCHED, seed + s)
        eps_true = eps_all.float()
        for name, emb in embeds.items():
            eps_c = unet(zt_all, t_all, encoder_hidden_states=emb.expand(nC * K, -1, -1)).sample.float()
            e = (eps_c - eps_true).pow(2).mean(dim=(1, 2, 3)).view(nC, K).mean(1)
            out[name][s:s + nC] = e.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--K", type=int, default=SC.K_MAIN)
    ap.add_argument("--outdir", default=os.path.join(E.HERE, "results_search"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    manifest = json.load(open(os.path.join(E.HERE, "manifest_true.json")))
    true = [m for m in manifest if m["group"] == "true"]
    files = [os.path.join(E.HERE, m["file"]) for m in true]

    tok, te, vae, unet, E.SCHED = E.load_models()
    z0 = E.encode_images(vae, files)
    all_prompts = {**SC.FACE_PROMPTS, **SC.NONFACE_PROMPTS}
    embeds = {k: E.embed(tok, te, v) for k, v in all_prompts.items()}

    tr_names = list(SC.TRANGES.keys())[args.shard::args.nshards]
    out = {"K": args.K, "tranges": {}}
    for tn in tr_names:
        t_lo, t_hi = SC.TRANGES[tn]
        out["tranges"][tn] = {}
        for sd in SC.SEEDS:
            tbl = score_cond(unet, z0, embeds, t_lo, t_hi, args.K, sd, chunk=8)
            out["tranges"][tn][str(sd)] = {p: v.tolist() for p, v in tbl.items()}
        print(f"[search shard {args.shard}] {tn} done", flush=True)

    json.dump(out, open(os.path.join(args.outdir, f"shard_{args.shard}.json"), "w"))
    print(f"[search shard {args.shard}] wrote {len(tr_names)} tranges", flush=True)


if __name__ == "__main__":
    main()
