#!/usr/bin/env python
"""One GPU worker: score a SHARD of the 40 occupation images with the diffusion
residual-error classifier (reusing error_classify machinery). Saves per-image
scalar errors for every (grid, scheme, prompt) so the aggregator can evaluate all
prompt pairs / schemes analytically."""
import os, json, argparse
import numpy as np
import error_classify as E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--manifest", default=os.path.join(E.HERE, "manifest_occ.json"))
    ap.add_argument("--outdir", default=os.path.join(E.HERE, "results_occ"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    manifest = json.load(open(args.manifest))
    files = [os.path.join(E.HERE, m["file"]) for m in manifest]
    idxs = list(range(len(files)))[args.shard::args.nshards]
    my_files = [files[i] for i in idxs]

    tok, te, vae, unet, E.SCHED = E.load_models()
    z0 = E.encode_images(vae, my_files)

    all_embeds = {**{k: E.embed(tok, te, v) for k, v in E.FACE_PROMPTS.items()},
                  **{k: E.embed(tok, te, v) for k, v in E.NONFACE_PROMPTS.items()}}
    uncond = E.embed(tok, te, E.UNCOND)

    out = {"idxs": idxs, "grids": {}}
    for gname, grid in E.GRIDS.items():
        scored = E.score_vocab(unet, z0, all_embeds, uncond, grid, args.seed,
                               chunk=len(my_files) if my_files else 1)
        # convert np arrays -> lists
        out["grids"][gname] = {sch: {p: v.tolist() for p, v in tbl.items()}
                               for sch, tbl in scored.items()}
    # negcfg targeted pairs on broad grid
    neg_pairs = [("F_face", "N_nonface"), ("F_person", "N_nopeople"),
                 ("F_humanface", "N_landscape")]
    out["negcfg"] = {}
    for fk, nk in neg_pairs:
        for g in [2.0, 4.0]:
            Ef, En = E.score_negcfg(unet, z0, all_embeds[fk], all_embeds[nk],
                                    E.GRIDS["broad_50_950"], args.seed, g,
                                    chunk=len(my_files) if my_files else 1)
            out["negcfg"][f"{fk}|{nk}|negcfg{g}"] = dict(Ef=Ef.tolist(), En=En.tolist())

    json.dump(out, open(os.path.join(args.outdir, f"shard_{args.shard}.json"), "w"))
    print(f"[score shard {args.shard}] done: {len(idxs)} images", flush=True)

if __name__ == "__main__":
    main()
