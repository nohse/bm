#!/usr/bin/env python
"""Attention-WEIGHTED cond scorer (original repo residual_gender_logits mechanism,
adapted to face/no-face). The 'face' cross-attention map from the FACE prompt is the
spatial weight; it is applied to BOTH the face and non-face per-pixel eps-error maps
(shared weight), then a weighted spatial SUM gives the error. Low memory, 8-GPU shard."""
import os, json, argparse, importlib, math
import numpy as np
import torch
import torch.nn.functional as F
import error_classify as E

HERE = E.HERE


class AttnCtx:
    def __init__(self):
        self.enabled = False
        self.token_idxs = None
        self.store = []


class CrossAttnCaptureProcessor:
    """Copied from the reference file: explicit get_attention_scores path; when enabled,
    stores the class-token attention column [B*heads, query_hw] (detached)."""
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.ctx.enabled and self.ctx.token_idxs is not None:
            with torch.no_grad():
                col = attention_probs[..., self.ctx.token_idxs].mean(dim=-1)   # [B*heads, query_hw]
            self.ctx.store.append((col.detach(), attn.heads))
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def find_token_idxs(tok, prompt, word):
    ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True).input_ids
    wids = tok(word, add_special_tokens=False).input_ids
    L = len(wids)
    for i in range(len(ids) - L + 1):
        if ids[i:i + L] == wids:
            return list(range(i, i + L))
    raise ValueError(f"'{word}' not in '{prompt}'")


def build_W(store, B, H=64, Wd=64):
    acc = torch.zeros(B, H, Wd, device=E.DEV)
    cnt = 0
    for col, heads in store:
        hw = col.shape[-1]; s = int(round(math.sqrt(hw)))
        a = col.view(B, heads, s, s).float()
        a = F.interpolate(a, size=(H, Wd), mode="bilinear", align_corners=False).mean(1)  # [B,H,W]
        acc = acc + a; cnt += 1
    m = acc / max(cnt, 1)
    m = m / (m.sum(dim=(1, 2), keepdim=True) + 1e-8)
    return m.detach()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--mem_frac", type=float, default=0.15)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--cfg", default="exp100_attn_config")
    ap.add_argument("--outdir", default=os.path.join(HERE, "exp100", "scores_attn"))
    args = ap.parse_args()
    C = importlib.import_module(args.cfg)
    os.makedirs(args.outdir, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)

    manifest = json.load(open(os.path.join(HERE, "exp100", "manifest_exp100.json")))
    files = [os.path.join(HERE, m["file"]) for m in manifest]
    labels = np.array([1 if m["label"] == "face" else 0 for m in manifest])

    tok, te, vae, unet, E.SCHED = E.load_models()
    ctx = AttnCtx()
    procs = dict(unet.attn_processors)
    for nm in list(procs):
        if nm.endswith("attn2.processor"):
            procs[nm] = CrossAttnCaptureProcessor(ctx)
    unet.set_attn_processor(procs)

    z0 = E.encode_images(vae, files); del vae; torch.cuda.empty_cache()
    face_emb = {k: E.embed(tok, te, v) for k, v in C.FACE_PROMPTS.items()}
    face_tok = {k: find_token_idxs(tok, v, C.ATTN_WORD) for k, v in C.FACE_PROMPTS.items()}
    non_emb = {k: E.embed(tok, te, v) for k, v in C.NONFACE_PROMPTS.items()}
    del te; torch.cuda.empty_cache()

    N = len(files); NT = C.NT
    jobs = [(tn, sd) for tn in C.TRANGES for sd in C.SEEDS][args.shard::args.nshards]
    store_out = {}
    for tn, sd in jobs:
        t_lo, t_hi = C.TRANGES[tn]
        errf = {fk: np.zeros((N, NT), np.float32) for fk in C.FACE_PROMPTS}
        errn = {(fk, nk): np.zeros((N, NT), np.float32) for fk in C.FACE_PROMPTS for nk in C.NONFACE_PROMPTS}
        for s in range(0, N, args.chunk):
            z0c = z0[s:s + args.chunk]
            eps_all, zt_all, t_all, nC = E.make_grid_noise(z0c, t_lo, t_hi, NT, E.SCHED, sd + s)
            B = nC * NT
            eps_true = eps_all.float()
            Ws = {}
            for fk, emb in face_emb.items():
                ctx.store = []; ctx.token_idxs = face_tok[fk]; ctx.enabled = True
                epf = unet(zt_all, t_all, encoder_hidden_states=emb.expand(B, -1, -1)).sample.float()
                ctx.enabled = False; st = ctx.store; ctx.store = []
                W = build_W(st, B)                                  # [B,64,64] sum-to-1
                Ws[fk] = W
                sqf = (epf - eps_true).pow(2).mean(1)               # [B,64,64]
                errf[fk][s:s + nC] = (W * sqf).sum(dim=(1, 2)).view(nC, NT).cpu().numpy()
            for nk, emb in non_emb.items():
                epn = unet(zt_all, t_all, encoder_hidden_states=emb.expand(B, -1, -1)).sample.float()
                sqn = (epn - eps_true).pow(2).mean(1)               # [B,64,64]
                for fk in C.FACE_PROMPTS:
                    errn[(fk, nk)][s:s + nC] = (Ws[fk] * sqn).sum(dim=(1, 2)).view(nC, NT).cpu().numpy()
        for fk in C.FACE_PROMPTS:
            store_out[f"{tn}|{sd}|{fk}|FACE"] = errf[fk]
            for nk in C.NONFACE_PROMPTS:
                store_out[f"{tn}|{sd}|{fk}|{nk}|NON"] = errn[(fk, nk)]
        np.savez_compressed(os.path.join(args.outdir, f"scores_shard{args.shard}.npz"),
                            labels=labels, **store_out)
        print(f"[score_attn] {tn} seed {sd} done (peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
