    #!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#



# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
from pathlib import Path

import argparse
import itertools
import logging
import math
import shutil
import json
import pytz
import random
from datetime import datetime
from tqdm.auto import tqdm
import copy
import pickle as pkl
import yaml
from packaging import version
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch
from torch import nn
import torchvision
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import scipy
from skimage import transform
import kornia
from sentence_transformers import SentenceTransformer, util
import open_clip

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, GradScalerKwargs
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
import torchvision.transforms as T  # PIL 변환용 (없으면 추가)
import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    DDPMScheduler,   # <-- [ADDED] for forward noising in SDS classifier
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
    AttnProcessor,   # <-- [ADDED] for temporary mid-block attn recording
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.training_utils import EMAModel

# you MUST import torch before insightface
# otherwise onnxruntime, used by FaceAnalysis, can only use CPU
from insightface.app import FaceAnalysis
from typing import List, Optional


my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.


# =========================
# [ADDED] Mid-block attn recorder (no scaling, token-masked)
# =========================
class _AttnStore:
    def __init__(self):
        self.maps = []  # list of [B, H, Q] already head-averaged -> we take [B,Q]
    def clear(self):
        self.maps.clear()
    def add(self, x):
        # x: [B,H,Q] -> average over heads -> [B,Q]
        if x.dim() == 3:
            self.maps.append(x.mean(dim=1).detach())
    def aggregate(self):
        if not self.maps:
            return None
        X = torch.stack(self.maps, dim=0)  # [L,B,Q]
        return X.mean(dim=0)               # [B,Q]


class _RecordingCrossAttnProcessor(AttnProcessor):
    """
    A lightweight AttnProcessor that records cross-attn probs at mid-block attn2 processors.
    It uses the vanilla attention math (no LoRA). We only use it under torch.no_grad() to get attn maps,
    then restore original processors for normal forward (which may include LoRA).
    """
    def __init__(self, store: _AttnStore, token_mask: torch.Tensor = None):
        super().__init__()
        self.store = store
        self.token_mask = token_mask  # [B,seq_len] float 0/1

    def set_token_mask(self, mask):
        self.token_mask = mask

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if not is_cross:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # attn probs [B*H, Q, K]
        attn_probs = attn.get_attention_scores(query, key, attention_mask)

        # Record only for cross-attn with a token mask
        if is_cross and (self.token_mask is not None):
            H = attn.heads
            Q = attn_probs.shape[1]
            K = attn_probs.shape[2]
            probs = attn_probs.view(batch_size, H, Q, K)     # [B,H,Q,K]
            m = self.token_mask.to(probs.dtype).unsqueeze(1).unsqueeze(2)  # [B,1,1,K]
            focused = (probs * m).sum(dim=-1)                # [B,H,Q]
            self.store.add(focused)                          # will be head-averaged inside

        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class CrossAttnCapture:
    """
    UNet의 cross-attention 모듈 내부 선형층 to_q, to_k에 forward_hook을 걸어
    각 호출마다 Q, K를 가로채고 즉시 attn = softmax(QK^T / sqrt(d))를 계산.
    지정된 텍스트 토큰 인덱스들에 대한 주의만 모아 저장합니다.
    maps는 각 훅 호출마다 배치 크기(B_cond) 만큼의 (h,w) 맵을 담는 텐서들을 저장합니다.
    """
    def __init__(self, token_indices: List[int], use_cpu: bool = False, expect_cfg_pair: bool = True):
        # use_cpu: if True (legacy), move tensors to CPU for attention math (safe but slow).
        # if False, perform attention math on the same device as Q/K (typically GPU).
        self.token_indices = token_indices
        self.handles = [] 
        self.maps = []   # list of tensors shape (B_cond, h, w)
        self.use_cpu = use_cpu
        self._q_cache = {}  # parent_id -> q (b, nq, h*d)
        self._parent_map = {}  # submodule_id -> parent module (avoid setting Module attrs)
        self.use_cpu = use_cpu
        self.expect_cfg_pair = expect_cfg_pair

    def _q_hook(self, module, inputs, output):
        # Detach and stash Q early to avoid interacting with autograd/checkpointing.
        parent = self._parent_map.get(id(module))
        if parent is None:
            return
        parent_id = id(parent)
        # stash a detached clone under torch.no_grad() to avoid autograd/ckpt interference
        try:
            with torch.no_grad():
                self._q_cache[parent_id] = output.detach().clone()
        except Exception:
            # fallback: keep raw output (best-effort)
            self._q_cache[parent_id] = output.detach()

    def _k_hook(self, module, inputs, output):
        # Perform all attention computations under no_grad on the same device
        # (GPU by default) to avoid expensive CPU<->GPU transfers. If `use_cpu` is True,
        # fall back to CPU (legacy behavior).
        parent = self._parent_map.get(id(module))
        if parent is None:
            return
        parent_id = id(parent)

        if parent_id not in self._q_cache:
            return

        q = self._q_cache.pop(parent_id)
        k = output
        # compute on same device unless use_cpu True
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

            # ---- infer head count / dimensions ----
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

            # attn: (B, H, Nq, Nk) -> select token indices along Nk -> (B, H, Nq, T)
            attn_tok = attn.index_select(-1, tok_idx).mean(dim=-1)  # (B, H, Nq)

            if self.expect_cfg_pair and B >= 2:
                b_half = B // 2
                cond = attn_tok[B - b_half : B]  # (b_half, H, Nq)
            else:
                cond = attn_tok  # (B, H, Nq)

            # per-image head-averaged maps: (B_sel, Nq)
            per_image = cond.mean(dim=1)  # (B_sel, Nq)

            hw = int(math.sqrt(per_image.shape[1]))
            if hw * hw != per_image.shape[1]:
                return

            b_sel = per_image.shape[0]
            per_image = per_image.view(b_sel, hw, hw)
            # optionally move to cpu for storage to reduce peak GPU memory
            if self.use_cpu:
                per_image = per_image.cpu()

            # store a single tensor per hook-call: (B_sel, h, w)
            self.maps.append(per_image)

    def add_hooks(self, unet: torch.nn.Module):
        installed = 0
        for name, module in unet.named_modules(): 
            has_qk = hasattr(module, "to_q") and hasattr(module, "to_k") 
            if not has_qk: 
                continue
            if not has_qk:
                continue

            is_cross = getattr(module, "is_cross_attention", None) 
            if is_cross is None: 
                is_cross = ("attn2" in name) or ("Cross" in module.__class__.__name__) 
            if not is_cross: 
                continue
            if not is_cross:
                continue

            # Avoid assigning Module instances as attributes on submodules (would register cyclic
            # children). Keep mapping in this object instead.
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
            installed += 1

        return self

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self._q_cache = {}
        self.maps = []

    def aggregated_map(self) -> Optional[torch.Tensor]:
        """
        수집된 모든 맵을 배치 차원(B_images)별로 평균하여 반환.
        반환: (B_images, H, W) torch.Tensor (정규화하지 않음)
        """
        if not self.maps:
            return None
        # find max hw across stored maps
        max_hw = max(m.shape[-1] for m in self.maps)
        upsampled = []
        for m in self.maps:
            # m: (B_images, h, w)
            Bm, hm, wm = m.shape
            ten = m.unsqueeze(1).float()   # (Bm,1,h,w)
            if hm != max_hw or wm != max_hw:
                ten = F.interpolate(ten, size=(max_hw, max_hw), mode="bilinear", align_corners=False)
            upsampled.append(ten.squeeze(1))  # (Bm, max_hw, max_hw)

        # stack across hooks: (num_calls, B_images, H, W)
        S = torch.stack(upsampled, dim=0)
        # mean over calls -> (B_images, H, W)
        M = S.mean(dim=0)
        return M  # [B_images, H, W]


def _build_token_mask(tokenizer, prompt: str, keywords, device):
    toks = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                     truncation=True, return_tensors="pt")
    ids = toks.input_ids[0]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    kws = [k.lower() for k in keywords]
    mask = []
    for piece in pieces:
        p = piece.lower().replace("Ġ", "").replace("▁", "")
        mask.append(1.0 if any(k in p for k in kws) else 0.0)
    return torch.tensor(mask, device=device).unsqueeze(0)


def sanitize_filename(s: str) -> str:
    """Make a filesystem-safe short filename from prompt text."""
    return "".join(c if c.isalnum() else "_" for c in s)[:200]


def _normalize_attmap_for_vis(att: torch.Tensor) -> torch.Tensor:
    """Normalize a single attention map for visualization only.

    The underlying attention tensor is left untouched; this just stretches contrast
    so low-magnitude maps remain visible in saved PNGs and overlays.
    """
    if not isinstance(att, torch.Tensor):
        att = torch.tensor(att)

    a = att.detach().float().cpu()
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    nonzero = a[a > 0]
    if nonzero.numel() == 0:
        return torch.zeros_like(a)

    lo = torch.quantile(nonzero, 0.05)
    hi = torch.quantile(nonzero, 0.995)
    if not torch.isfinite(lo):
        lo = nonzero.min()
    if not torch.isfinite(hi):
        hi = nonzero.max()

    if hi <= lo:
        scaled = (a > 0).float()
    else:
        scaled = ((a - lo) / (hi - lo)).clamp(0.0, 1.0)

    # Lift low responses so sparse maps do not look fully black.
    return scaled.pow(0.5)


def _attmap_vis_to_pil(vis: torch.Tensor) -> Image.Image:
    a = vis.clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
    img = Image.fromarray(a, mode="L")
    return ImageOps.colorize(img, black="black", mid="orange", white="red")


def attmap_to_pil(att: torch.Tensor) -> Image.Image:
    """Convert a single HxW attention map to a colored PIL image for display."""
    vis = _normalize_attmap_for_vis(att)
    return _attmap_vis_to_pil(vis)


def save_attmaps(att: Optional[torch.Tensor], save_dir: str, prefix: str):
    """Save attention maps to `save_dir` with filenames `{prefix}_{i}.jpg`.

    Args:
        att: Tensor of shape (B, H, W) or (H, W) with values in [0,1].
        save_dir: directory to save images into.
        prefix: filename prefix (should be sanitized by caller).
    """
    if att is None:
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if isinstance(att, torch.Tensor):
        ten = att.detach().cpu()
    else:
        ten = torch.tensor(att)

    if ten.ndim == 2:
        ten = ten.unsqueeze(0)

    for i in range(ten.shape[0]):
        pil = attmap_to_pil(ten[i])
        fname = f"{prefix}_{i}.jpg"
        pil.save(os.path.join(save_dir, fname), format="JPEG", quality=95)


def attmap_overlay_on_image(att_map: torch.Tensor, image: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    """Overlay a single attention map onto a single image tensor and return a PIL image.

    - `att_map`: HxW float tensor in [0,1]
    - `image`: 3xH_imgxW_img tensor in [-1,1]
    - `alpha`: overlay alpha for attention heatmap
    """
    # convert image tensor -> PIL
    try:
        img_pil = transforms.ToPILImage()(image.mul(0.5).add(0.5))
    except Exception:
        # fallback normalization
        img_pil = transforms.ToPILImage()((image + 1.0) * 0.5)

    vis = _normalize_attmap_for_vis(att_map)
    att_pil = _attmap_vis_to_pil(vis)

    if att_pil.size != img_pil.size:
        att_pil = att_pil.resize(img_pil.size, resample=Image.BILINEAR)
        vis = (
            F.interpolate(
                vis.unsqueeze(0).unsqueeze(0),
                size=(img_pil.size[1], img_pil.size[0]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    base = img_pil.convert("RGBA")
    heat = att_pil.convert("RGBA")
    alpha_mask = Image.fromarray(
        vis.mul(255 * alpha).clamp(0, 255).to(torch.uint8).cpu().numpy(),
        mode="L",
    )
    heat.putalpha(alpha_mask)
    return Image.alpha_composite(base, heat).convert("RGB")


def save_attmaps_with_overlay(att: Optional[torch.Tensor], images: torch.Tensor, save_dir: str, prefix: str, alpha: float = 0.45):
    """Save attmaps and overlayed images side-by-side.

    - `att`: (B, H, W) or (H, W)
    - `images`: (B, 3, H_img, W_img) in [-1,1]
    """
    if att is None:
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    ten = att.detach().cpu() if isinstance(att, torch.Tensor) else torch.tensor(att)
    imgs = images.detach().cpu()

    if ten.ndim == 2:
        ten = ten.unsqueeze(0)

    n = min(ten.shape[0], imgs.shape[0])
    for i in range(n):
        try:
            blended = attmap_overlay_on_image(ten[i], imgs[i], alpha=alpha)
            fname = f"{prefix}_{i}_overlay.jpg"
            blended.save(os.path.join(save_dir, fname), format="JPEG", quality=95)
        except Exception as e:
            print(f"[attmap overlay save error] idx={i} -> {e}")


def _install_mid_recorders(unet, store: _AttnStore, token_mask: torch.Tensor):
    """
    Swap ONLY mid_block.*.attn2.processor to recording processors.
    Returns a tuple(original_dict, recorder_dict) so caller can restore later.
    """
    orig = unet.attn_processors
    new = {}
    rec = {}
    for name, proc in orig.items():
        if name.startswith("mid_block") and name.endswith("attn2.processor"):
            rp = _RecordingCrossAttnProcessor(store=store, token_mask=token_mask)
            new[name] = rp
            rec[name] = rp
        else:
            new[name] = proc
    unet.set_attn_processor(new)
    return orig, rec


def _restore_attn_processors(unet, orig_dict):
    unet.set_attn_processor(orig_dict)


class MidBlockCapture:
    """Forward-hook wrapper that captures the output of `unet.mid_block`.

    Used by the h-space L2 loss to grab the [B, 1280, 8, 8] bottleneck feature
    on a single UNet forward pass, without modifying the model.
    """

    def __init__(self):
        self.last = None
        self._handle = None

    def _hook_fn(self, module, inputs, output):
        # UNetMidBlock2DCrossAttn returns a single tensor in SD1.5
        self.last = output

    def attach(self, unet):
        self.detach()
        self._handle = unet.mid_block.register_forward_hook(self._hook_fn)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.last = None


# =========================
# Original user code (with minimal changes below)
# =========================

class FaceFeatsModel(torch.nn.Module):
    def __init__(self, face_feats_path):
        super().__init__()
        
        with open(face_feats_path, "rb") as f:
            face_feats, face_genders, face_logits = pkl.load(f)
        
        face_feats = torch.nn.functional.normalize(face_feats, dim=-1)
        self.face_feats = nn.Parameter(face_feats)   
        self.face_feats.requires_grad_(False)               
        
    def forward(self, x):
        """no forward function
        """
        return None
        
    @torch.no_grad()
    def semantic_search(self, query_embeddings, selector=None, return_similarity=False):
        """search the closest face embedding from vector database.
        """
        target_embeddings = torch.ones_like(query_embeddings) * (-1)
        if return_similarity:
            similarities = torch.ones([query_embeddings.shape[0]], device=query_embeddings.device, dtype=query_embeddings.dtype) * (-1)
            
        if selector.sum()>0:
            hits = util.semantic_search(query_embeddings[selector], self.face_feats, score_function=util.dot_score, top_k=1)
            target_embeddings_ = torch.cat([self.face_feats[hit[0]["corpus_id"]].unsqueeze(dim=0) for hit in hits])
            target_embeddings[selector] = target_embeddings_
            if return_similarity:
                similarities_ = torch.tensor([hit[0]["score"] for hit in hits], device=query_embeddings.device, dtype=query_embeddings.dtype)
                similarities[selector] = similarities_

        if return_similarity:
            return target_embeddings.data.detach().clone(), similarities
        else:
            return target_embeddings.data.detach().clone()


def clean_checkpoint(ckpts_save_dir, name, checkpoints_total_limit):
    checkpoints = os.listdir(ckpts_save_dir)
    checkpoints = [d for d in checkpoints if d.startswith(name)]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(
            f"chekpoint name:{name}, {len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(args.ckpts_save_dir, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def plot_in_grid(images, save_to, face_indicators=None, face_bboxs=None, preds_gender=None, pred_class_probs_gender=None):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """
    images_w_face = images[face_indicators] if face_indicators is not None else images
    images_wo_face = images[face_indicators.logical_not()] if face_indicators is not None else images[[]]

    # first reorder everything from most to least male, from most to least female, and finally images without faces
    if preds_gender is not None and pred_class_probs_gender is not None:
        idxs_male = (preds_gender == 1).nonzero(as_tuple=False).view([-1])
        probs_male = pred_class_probs_gender[idxs_male]
        idxs_male = idxs_male[probs_male.argsort(descending=True)]

        idxs_female = (preds_gender == 0).nonzero(as_tuple=False).view([-1])
        probs_female = pred_class_probs_gender[idxs_female]
        idxs_female = idxs_female[probs_female.argsort(descending=True)]

        idxs_no_face = (preds_gender == -1).nonzero(as_tuple=False).view([-1])

        images_to_plot = []
        idxs_reordered = torch.cat([idxs_male, idxs_female, idxs_no_face])
    else:
        idxs_reordered = torch.arange(images.shape[0], device=images.device)
        images_to_plot = []

    for idx in idxs_reordered:
        img = images[idx]
        face_indicator = face_indicators[idx] if face_indicators is not None else torch.tensor(False, device=images.device)
        face_bbox = face_bboxs[idx] if face_bboxs is not None else torch.tensor([0,0,0,0], device=images.device)
        pred_gender = preds_gender[idx] if preds_gender is not None else torch.tensor(-1, device=images.device)
        pred_class_prob_gender = pred_class_probs_gender[idx] if pred_class_probs_gender is not None else torch.tensor(0.0, device=images.device)
        
        if pred_gender == 1:
            pred = "Male"
            border_color = "blue"
        elif pred_gender == 0:
            pred = "Female"
            border_color = "red"
        elif pred_gender == -1:
            pred = "Undetected"
            border_color = "white"
        
        img_pil = transforms.ToPILImage()(img*0.5+0.5)
        img_pil_draw = ImageDraw.Draw(img_pil)  
        if face_bboxs is not None:
            img_pil_draw.rectangle(face_bbox.tolist(), fill =None, outline =border_color, width=4)

        img_pil = ImageOps.expand(img_pil, border=(50,0,0,0),fill=border_color)

        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_probs_gender is not None and pred_class_prob_gender.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_gender.item())*512)], fill ="white", outline =None)

        try:
            fnt = ImageFont.truetype(font="../data/0-utils/arial-bold.ttf", size=100)
        except:
            fnt = ImageFont.load_default()
        img_pil_draw.text((400, 400), f"{idx.item()}", align ="left", font=fnt)

        img_pil = ImageOps.expand(img_pil_draw._image, border=(10,10,10,10),fill="black")
        
        images_to_plot.append(img_pil)
        
    N_imgs = len(images_to_plot)
    if N_imgs == 0:
        return
    N1 = int(math.sqrt(N_imgs))
    N2 = math.ceil(N_imgs / N1)

    for i in range(N1*N2-N_imgs):
        images_to_plot.append(
            Image.new('RGB', color="white", size=images_to_plot[0].size)
        )
    grid = image_grid(images_to_plot, N1, N2)
    if not os.path.exists(os.path.dirname(save_to)):
        os.makedirs(os.path.dirname(save_to))
    grid.save(save_to, quality=25)

def make_grad_hook(coef):
    return lambda x: coef * x

def customized_all_gather(tensor, accelerator, return_tensor_other_processes=False):
    """All-gather a tensor along dim 0 across processes.

    Hardened against the NCCL deadlock that used to hang a whole run for the full
    watchdog timeout (~1h) and then SIGABRT it. torch.distributed.all_gather requires
    every rank to pass an IDENTICALLY shaped tensor, but the old implementation passed
    each rank's raw tensor with no agreement step: if two ranks ever disagreed on a
    shape (e.g. a per-rank, data-dependent count) every rank blocked until the timeout.

    Now we first all_gather a tiny FIXED-SIZE shape descriptor (a collective that can
    never deadlock), then:
      * all shapes equal     -> original fast path (result is byte-for-byte identical);
      * only dim-0 differs    -> log the per-rank shapes and pad-to-max / gather / trim
                                 so the run survives instead of hanging;
      * trailing dims differ  -> raise a clear, SYNCHRONIZED error on every rank (every
                                 rank sees the same gathered descriptors and raises the
                                 same way, so this also cannot deadlock).
    """
    world = accelerator.num_processes
    t = tensor.detach()

    if world == 1:
        tensor_all = t.clone()
        if return_tensor_other_processes:
            tensor_others = torch.empty([0,] + list(t.shape[1:]), device=accelerator.device, dtype=t.dtype)
            return tensor_all, tensor_others
        return tensor_all

    # 1) fixed-size shape+dtype agreement -- this collective itself can never hang.
    #    NOTE: torch.distributed.all_gather requires identical SHAPE *and* DTYPE on every
    #    rank, so we check both. (A fp16-vs-fp32 dtype split from a no-face code path was
    #    the actual cause of the eval hangs, and shapes alone are identical there.)
    _MAXD = 8
    _DTYPE_CODES = {
        torch.float32: 0, torch.float16: 1, torch.bfloat16: 2, torch.float64: 3,
        torch.int64: 4, torch.int32: 5, torch.int16: 6, torch.int8: 7,
        torch.uint8: 8, torch.bool: 9,
    }
    assert t.dim() <= _MAXD, f"customized_all_gather: tensor ndim {t.dim()} exceeds {_MAXD}"
    desc = torch.full([_MAXD + 2], -1, dtype=torch.long, device=t.device)
    desc[0] = t.dim()
    if t.dim() > 0:
        desc[1:1 + t.dim()] = torch.tensor(list(t.shape), dtype=torch.long, device=t.device)
    desc[_MAXD + 1] = _DTYPE_CODES.get(t.dtype, -2)
    descs = [torch.empty_like(desc) for _ in range(world)]
    torch.distributed.all_gather(descs, desc)
    shapes, dtype_codes = [], []
    for d in descs:
        nd = int(d[0].item())
        shapes.append(tuple(int(x) for x in d[1:1 + nd].tolist()))
        dtype_codes.append(int(d[_MAXD + 1].item()))

    shapes_equal = all(s == shapes[0] for s in shapes)
    dtypes_equal = all(c == dtype_codes[0] for c in dtype_codes)

    if shapes_equal and dtypes_equal:
        # 2a) fast path: identical shape+dtype -> original behavior, unchanged.
        tensor_all = [t.clone() for _ in range(world)]
        torch.distributed.all_gather(tensor_all, t)
        if return_tensor_other_processes:
            tensor_others = torch.cat(
                [tensor_all[idx] for idx in range(world) if idx != accelerator.local_process_index], dim=0
            )
        tensor_all = torch.cat(tensor_all, dim=0)
        return (tensor_all, tensor_others) if return_tensor_other_processes else tensor_all

    # 2b) metadata disagrees: the old code would have DEADLOCKED here.
    if not dtypes_equal:
        # dtype codes: 0=f32 1=f16 2=bf16 3=f64 4=i64 5=i32 6=i16 7=i8 8=u8 9=bool (-2=other)
        raise RuntimeError(
            "customized_all_gather: tensor DTYPE differs across ranks "
            f"(per-rank dtype codes: {dtype_codes}, shapes: {shapes}); this cannot be "
            "all_gathered and would have deadlocked NCCL until the watchdog timeout. "
            "dtype codes: 0=f32 1=f16 2=bf16 3=f64 4=i64 5=i32 6=i16 7=i8 8=u8 9=bool."
        )
    if any(len(s) != len(shapes[0]) or s[1:] != shapes[0][1:] for s in shapes):
        raise RuntimeError(
            "customized_all_gather: tensors differ across ranks in ndim or trailing dims "
            f"(per-rank shapes: {shapes}); this cannot be all_gathered and would have "
            "deadlocked NCCL until the watchdog timeout."
        )
    if accelerator.is_main_process:
        print(
            "[customized_all_gather][WARN] dim-0 length differs across ranks "
            f"(per-rank shapes: {shapes}); padding to max and trimming so the run survives "
            "(the old code would have hung until the NCCL watchdog timeout)."
        )
    sizes = [s[0] for s in shapes]
    max_n = max(sizes)
    if t.shape[0] < max_n:
        pad = torch.zeros([max_n - t.shape[0], *t.shape[1:]], device=t.device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t
    gathered = [torch.empty_like(t_pad) for _ in range(world)]
    torch.distributed.all_gather(gathered, t_pad)
    parts = [gathered[r][:sizes[r]] for r in range(world)]
    tensor_all = torch.cat(parts, dim=0)
    if return_tensor_other_processes:
        others_parts = [parts[r] for r in range(world) if r != accelerator.local_process_index]
        tensor_others = torch.cat(others_parts, dim=0) if others_parts else torch.empty(
            [0,] + list(t.shape[1:]), device=t.device, dtype=t.dtype
        )
        return tensor_all, tensor_others
    return tensor_all


def expand_bbox(bbox, expand_coef, target_ratio):
    """
    bbox: [width_small, height_small, width_large, height_large], 
        this is the format returned from insightface.app.FaceAnalysis
    expand_coef: 0 is no expansion
    target_ratio: target img height/width ratio
    
    note that it is possible that bbox is outside the original image size
    confirmed for insightface.app.FaceAnalysis
    """
    
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    
    current_ratio = bbox_height / bbox_width
    if current_ratio > target_ratio:
        more_height = bbox_height * expand_coef
        more_width = (bbox_height+more_height) / target_ratio - bbox_width
    elif current_ratio <= target_ratio:
        more_width = bbox_width * expand_coef
        more_height = (bbox_width+more_width) * target_ratio - bbox_height
    
    bbox_new = [0,0,0,0]
    bbox_new[0] = int(round(bbox[0] - more_width*0.5))
    bbox_new[2] = int(round(bbox[2] + more_width*0.5))
    bbox_new[1] = int(round(bbox[1] - more_height*0.5))
    bbox_new[3] = int(round(bbox[3] + more_height*0.5))
    return bbox_new

def crop_face(img_tensor, bbox_new, target_size, fill_value):
    """
    img_tensor: [3,H,W]
    bbox_new: [width_small, height_small, width_large, height_large]
    target_size: [width,height]
    fill_value: value used if need to pad
    """
    img_height, img_width = img_tensor.shape[-2:]
    
    idx_left = max(bbox_new[0],0)
    idx_right = min(bbox_new[2], img_width)
    idx_bottom = max(bbox_new[1],0)
    idx_top = min(bbox_new[3], img_height)

    pad_left = max(-bbox_new[0],0)
    pad_right = max(-(img_width-bbox_new[2]),0)
    pad_top = max(-bbox_new[1],0)
    pad_bottom = max(-(img_height-bbox_new[3]),0)

    img_face = img_tensor[:,idx_bottom:idx_top,idx_left:idx_right]
    if pad_left>0 or pad_top>0 or pad_right>0 or pad_bottom>0:
        img_face = torchvision.transforms.Pad([pad_left,pad_top,pad_right,pad_bottom], fill=fill_value)(img_face)
    img_face = torchvision.transforms.Resize(size=target_size)(img_face)
    return img_face

def image_pipeline(img, tgz_landmark):
    img = (img+1)/2.0 * 255 # map to [0,255]

    crop_size = (112,112)
    src_landmark = np.array(
    [[38.2946, 51.6963], # left eye
    [73.5318, 51.5014], # right eye
    [56.0252, 71.7366], # nose
    [41.5493, 92.3655], # left corner of the mouth
    [70.7299, 92.2041]] # right corner of the mouth
    )

    tform = transform.SimilarityTransform()
    tform.estimate(tgz_landmark, src_landmark)

    M = torch.tensor(tform.params[0:2, :]).unsqueeze(dim=0).to(img.dtype).to(img.device)
    img_face = kornia.geometry.transform.warp_affine(img.unsqueeze(dim=0), M, crop_size, mode='bilinear', padding_mode='zeros', align_corners=False)
    img_face = img_face.squeeze()

    img_face = (img_face/255.0)*2-1 # map back to [-1,1]
    return img_face


class PromptsDataset(Dataset):
    def __init__(
        self,
        prompts,
    ):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, i):
        return self.prompts[i]


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to finetune Stable Diffusion for debiasing purposes.")

    # 1. experiment setting
    parser.add_argument(
        '--proj_name', 
        default="debias-SD",
        help="proj name",
        type=str, 
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        default=True,
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        default=False,
        help="Whether to train unet. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default="5991", 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20,
        help=(
            "Save a temporary checkpoint every X steps. "
            "The purpose of these checkpoints is to easily resume training "
            "when some error occurs during training."
        )
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=(
            "Max number of temporary checkpoints to store. "
            "The oldest ones will be deleted when new checkpoints are saved."),
    )
    parser.add_argument(
        "--checkpointing_steps_long",
        type=int,
        default=200,
        help=(
            "Save a checkpoint every Y steps. "
            "These checkpoints will not be deleted. They are used for final evaluation. "
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="provide the checkpoint path to resume from checkpoint",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    # parser.add_argument(
    #     "--enable_xformers_memory_efficient_attention", 
    #     action="store_true", 
    #     default=True,
    #     help="Whether or not to use xformers."
    # )
    parser.add_argument(
        "--rank",
        type=int,
        default=50,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        '--train_plot_every_n_iter', 
        help="plot training stats every n iteration", 
        type=int, 
        default=20
        )
    parser.add_argument(
        '--evaluate_every_n_iter', 
        help="evaluate every n iteration", 
        type=int,
        default=200
        )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help='only `"wandb"` is supported',
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        '--guidance_scale', 
        help="diffusion model text guidance scale", 
        type=float, 
        default=7.5
        )
    parser.add_argument(
        '--EMA_decay', 
        help="decay coefficient for EMA",
        type=float,
        default=0.996
        )

    # loss weight
    parser.add_argument(
        '--weight_loss_img', 
        default=4,
        help="weight for the image semantics preserving loss", 
        type=float, 
    )
    parser.add_argument(
        '--weight_loss_face', 
        default=1,
        help="weight for the face realism preserving loss", 
        type=float, 
    )
    parser.add_argument(
        '--weight_loss_face_realistic',
        default=4.0,
        help="weight for the realistic-face SDS loss (prompt: 'a photo of a realistic face')",
        type=float,
    )
    parser.add_argument(
        '--uncertainty_threshold', 
        help="the uncertainty threshold used in distributional alignment loss", 
        type=float, 
        default=0.2
        )
    parser.add_argument(
        "--ratio",
        "--target_male_ratio",
        dest="target_male_ratio",
        type=float,
        default=0.50,
        help="target male ratio for 2-class assignment (0.0 ~ 1.0).",
    )
    parser.add_argument('--factor1', help="train, val, test batch size", type=float, default=0.2)
    parser.add_argument('--factor2', help="train, val, test batch size", type=float, default=0.2)

    # batch size, properly set to max out GPU
    parser.add_argument(
        '--train_images_per_prompt_GPU', 
        help=(
            "number of images generated for a prompt per GPU during training. "
            "These images are used as a batch for distributional alignment."
        ), 
        type=int, 
        default=8,
        )
    parser.add_argument(
        '--train_GPU_batch_size', 
        help="training batch size in every GPU", 
        type=int, 
        default=4
        )
    parser.add_argument(
        '--val_images_per_prompt_GPU', 
        help=(
            "number of images generated for a prompt per GPU during validation. "
            "These images are used to measure bias."
        ),
        type=int, 
        default=60
        )
    parser.add_argument(
        '--val_GPU_batch_size', 
        help="validation batch size in every GPU", 
        type=int, 
        default=8
        )    


    # experiment input and output paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="logs will be saved to args.output_dir/args.proj_name/args.logging_dir",
    )
    parser.add_argument(
        "--prompt_occupation_path",
        type=str,
        default="../data/1-prompts/occupation.json",
        help="prompt template, and occupations for train and val",
    )
    parser.add_argument(
        '--classifier_weight_path', 
        default="../data/5-trained-test-classifiers/CelebA-MobileNetLarge-Gender-09191318/epoch=19-step=25320_MobileNetLarge.pt",
        help="pre-trained classifer that predicts binary gender", 
        type=str,
        required=False, 
    )
    parser.add_argument(
        '--face_feats_path', 
        help="external face feats, used for the face realism preserving loss", 
        type=str, 
        default="../data/3-face-features/CelebA_MobileNetLarge_08240859/face_feats.pkl"
        )
    # parser.add_argument(
    #     '--aligned_face_gender_model_path', 
    #     help="train, val, test batch size", 
    #     type=str, 
    #     default="../data/3-face-features/CelebA_MobileNetLarge_08240859/epoch=9-step=6330_MobileNetLarge.pt"
    #     )
    parser.add_argument('--opensphere_config', help="train, val, test batch size", type=str, default="../data/4-opensphere_checkpoints/opensphere_checkpoints/20220424_210641/config.yml")
    parser.add_argument('--opensphere_model_path', help="train, val, test batch size", type=str, default="../data/4-opensphere_checkpoints/opensphere_checkpoints/20220424_210641/models/backbone_100000.pth")

    # learning related settings
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", 
        type=float, 
        default=1.0, 
        help="Power factor of the polynomial scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=100.0, type=float, help="Max gradient norm.")

    # settings that should not be changed
    # we didn't experiment with other values
    parser.add_argument(
        "--img_size_small",
        type=int,
        default=224,
        help="For some operations, images will be resized to this size for more efficient processing",
    )
    parser.add_argument(
        "--size_face",
        type=int,
        default=224,
        help="faces will be resized to this size",
    )
    parser.add_argument(
        "--size_aligned_face",
        type=int,
        default=112,
        help="aligned faces will be resized to this size",
    )
    parser.add_argument('--face_gender_confidence_level', help="train, val, test batch size", type=float, default=0.9)

    # passed directly by accelerate
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
        )
    
    # config file
    parser.add_argument("--config", help="config file", type=str, default=None)

    # =====================
    # [ADDED] SDS classifier & attn gating args
    # =====================
    parser.add_argument("--sds_t_min", type=int, default=400, help="min t index for SDS")
    parser.add_argument("--sds_t_max", type=int, default=800, help="max t index for SDS")
    parser.add_argument("--sds_num_t", type=int, default=15, help="number of t samples (linspace)")
    parser.add_argument("--sds_num_eps", type=int, default=1, help="number of epsilon samples per t")
    parser.add_argument("--sds_tau", type=float, default=0.0001, help="temperature for -softmax on SDS losses")
    # h-space loss measured during denoising trajectory (fine DM vs original DM)
    parser.add_argument(
        "--h_loss_form", type=str, default="raw",
        choices=["raw", "cos"],
        help="h-space distance form during generation steps. "
             "'raw' = elementwise MSE (default). "
             "'cos' = 1 - cosine_similarity on flattened h (scale-invariant).",
    )
    parser.add_argument("--attn_grad_threshold", type=float, default=0.2, help="threshold for binary grad gate on attn map")
    # parse_args() 안의 인자들 사이에 추가
    parser.add_argument("--use_attn_weight", type=bool, default=True,
                        help="SDS에서 어텐션 가중치 사용할지 여부")
    parser.add_argument("--attn_blocks", type=str, default="mid",
                        help="어텐션을 훅킹할 블록들: 콤마로 결합 (예: 'mid', 'down', 'up', 'mid,up', 'all')")
    parser.add_argument("--attn_kind", type=str, default="attn2",
                        choices=["attn1","attn2"],
                        help="어텐션 종류 선택: attn1(self) / attn2(cross)")
    parser.add_argument("--attn_down_idx", type=str, default=None,
                        help="down_blocks 인덱스 선택 (예: '0,1,2' 또는 '0-2'); 미지정 시 전부")
    parser.add_argument("--attn_up_idx", type=str, default=None,
                        help="up_blocks 인덱스 선택 (예: '0,1,2' 또는 '1-3'); 미지정 시 전부")
    parser.add_argument(
        "--zeroshot_model", type=str, default="openai/clip-vit-large-patch14",
        help="zero-shot gender classifier로 사용할 CLIP 모델 이름"
    )
    parser.add_argument(
        "--skip_final_steps", type=int, default=0,
        help="Absolute number of final denoising steps to skip when using gradient-guided generation."
    )
    parser.add_argument(
        "--skip_final_steps_pct", type=float, default=50.0,
        help="If >0, compute final step skipping as this percentage of the sampled num_denoising_steps (overrides --skip_final_steps)."
    )
    parser.add_argument(
        "--eval_at_step0",
        dest="eval_at_step0",
        action="store_true",
        help="학습 시작 전(global_step==0) 평가를 수행합니다.",
    )
    parser.add_argument(
        "--no_eval_at_step0",
        dest="eval_at_step0",
        action="store_false",
        help="학습 시작 전(global_step==0) 평가를 건너뜁니다.",
    )

    # =====================
    # [ADDED] Region masking switch (face / attn / none)
    # =====================
    parser.add_argument("--region_mask_mode", type=str, default="none",
                        choices=["none", "face", "attn"],
                        help="SDS 및 backprop에서 사용할 영역 마스킹 방식 선택. 'attn'은 SDS 분류 프롬프트(woman/man) 토큰 기반 어텐션 맵, 'face'는 얼굴 bbox(학습 시 자동 비활성), 'none'은 전체.")
    parser.add_argument(
        "--save_attmaps",
        dest="save_attmaps",
        action="store_true",
        help="평가 시 attmap/overlay 이미지를 저장합니다.",
    )
    parser.add_argument(
        "--no_save_attmaps",
        dest="save_attmaps",
        action="store_false",
        help="평가 시 attmap/overlay 저장을 끕니다.",
    )
    parser.add_argument(
        "--log_wandb_images",
        dest="log_wandb_images",
        action="store_true",
        help="평가/학습 이미지를 wandb에 업로드합니다.",
    )
    parser.add_argument(
        "--no_wandb_images",
        dest="log_wandb_images",
        action="store_false",
        help="이미지를 wandb에 업로드하지 않습니다(디스크 저장은 유지).",
    )
    parser.set_defaults(save_attmaps=True)
    parser.set_defaults(eval_at_step0=False)
    parser.set_defaults(log_wandb_images=True)

    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="resume_from_checkpoint + eval_at_step0과 함께 사용. 시작 직후 1회 평가만 수행하고 학습 없이 종료합니다.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        if config_data is None:
            config_data = {}
        if "ratio" in config_data and "target_male_ratio" not in config_data:
            config_data["target_male_ratio"] = config_data["ratio"]
        if "ratio" in config_data:
            config_data.pop("ratio")
        args_dict = vars(args)
        unknown_config_keys = []
        force_disable_eval_at_step0 = "train_only_no_eval_models" in config_data
        for key, value in config_data.items():
            # Backward-compat: legacy config switch, and ensure no eval at step 0.
            if key == "train_only_no_eval_models":
                args_dict["eval_at_step0"] = False
                continue

            if key not in args_dict:
                unknown_config_keys.append(key)
                continue

            current_value = args_dict[key]
            if isinstance(current_value, bool):
                if isinstance(value, str):
                    value_lower = value.strip().lower()
                    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
                        args_dict[key] = True
                    elif value_lower in {"0", "false", "f", "no", "n", "off"}:
                        args_dict[key] = False
                    else:
                        raise ValueError(f"Invalid boolean string for config key '{key}': {value}")
                else:
                    args_dict[key] = bool(value)
            else:
                args_dict[key] = type(current_value)(value)

        if unknown_config_keys:
            print(f"[parse_args] Ignoring unknown config keys: {unknown_config_keys}")
        if force_disable_eval_at_step0:
            args_dict["eval_at_step0"] = False
        args = argparse.Namespace(**args_dict)

    args.target_male_ratio = float(np.clip(args.target_male_ratio, 0.0, 1.0))
    args.target_female_ratio = 1.0 - args.target_male_ratio

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

logger = get_logger(__name__)


def _parse_idx_list(s: str):
    """'0,2-4' 같은 문자열을 [0,2,3,4]로 파싱"""
    if not s: return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-")
            out.extend(list(range(int(a), int(b)+1)))
        else:
            out.append(int(part))
    return sorted(set(out))

def _install_recorders_select(
    unet, store: _AttnStore, token_mask: torch.Tensor,
    attn_blocks: str = "mid", attn_kind: str = "attn2",
    down_idx=None, up_idx=None
):
    """
    attn_blocks: 'mid', 'down', 'up', 'mid,up', 'all'
    attn_kind  : 'attn1' or 'attn2'
    down_idx   : [0,1,2] 선택 (None이면 전부)
    up_idx     : [0,1,2] 선택 (None이면 전부)
    """
    blocks = set([x.strip() for x in attn_blocks.split(",")])
    if "all" in blocks:
        blocks = {"down","mid","up"}

    orig = unet.attn_processors
    new, rec = {}, {}
    target_suffix = f"{attn_kind}.processor"

    for name, proc in orig.items():
        # 이름 예:
        #  - "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"
        #  - "mid_block.attentions.0.transformer_blocks.0.attn2.processor"
        #  - "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor"
        use_this = False

        if not name.endswith(target_suffix):
            new[name] = proc
            continue

        if "mid_block" in name and "mid" in blocks:
            use_this = True

        if "down_blocks." in name and "down" in blocks:
            if down_idx is None:
                use_this = True
            else:
                # down_blocks.i 가 포함되면 해당 i가 리스트에 있는지 확인
                for i in down_idx:
                    if f"down_blocks.{i}." in name:
                        use_this = True
                        break

        if "up_blocks." in name and "up" in blocks:
            if up_idx is None:
                use_this = True
            else:
                for i in up_idx:
                    if f"up_blocks.{i}." in name:
                        use_this = True
                        break

        if use_this:
            rp = _RecordingCrossAttnProcessor(store=store, token_mask=token_mask)
            new[name] = rp
            rec[name] = rp
        else:
            new[name] = proc

    unet.set_attn_processor(new)
    return orig, rec

def set_token_mask_all(rec_dict, mask):
    for rp in rec_dict.values():
        rp.set_token_mask(mask)

def main(args):

    if not args.train_text_encoder and not args.train_unet:
        raise ValueError("At least one of --train_text_encoder and --train_unet must be True.")
    if args.region_mask_mode == "face":
        logger.warning("region_mask_mode=face requested, but this run uses face detection only for the gradient hook/evaluation. Overriding SDS region_mask_mode to attn.")
        args.region_mask_mode = "attn"

    logging_dir = Path(args.output_dir, args.logging_dir)

    kwargs = GradScalerKwargs(
        init_scale = 2.**0,
        growth_interval=99999999, 
        backoff_factor=0.5,
        growth_factor=2,
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1, # we did not implement gradient accumulation
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        if version.parse(wandb.__version__) < version.parse("0.22.3"):
            raise RuntimeError(
                f"wandb>=0.22.3 is required for long API keys, but found wandb=={wandb.__version__}. "
                "Please upgrade wandb (e.g., pip install --upgrade wandb==0.22.3)."
            )
    else:
        raise ValueError("--report_to must be set to 'wanb', others are not implemented.")
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    def dist_broadcast_if_needed(tensor, src=0):
        if accelerator.num_processes == 1:
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError(
                "torch.distributed is not initialized even though Accelerator reports "
                f"{accelerator.num_processes} processes. Launch with accelerate/torchrun."
            )
        torch.distributed.broadcast(tensor, src=src)

    set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    now = datetime.now(my_timezone)
    timestring = now.strftime("%Y%m%d-%H%M")
    # Folder name encodes (per constrain.txt): date + which experiment (proj_name) +
    # the run-defining knobs -> region_mask_mode (attn/none), final-step skip %, and
    # each loss weight (img / realistic-face). NOTE: the fair-loss fields (_wFair-/_fcw-)
    # from DAL are omitted here because this script has no weight_loss_fair /
    # fair_correct_weight args (it uses the CE-based fair loss).
    folder_name = (
        f"{timestring}_{args.proj_name}"
        f"_region-{args.region_mask_mode}"
        f"_skip-{int(args.skip_final_steps_pct)}pct"
        f"_wImg-{args.weight_loss_img}"
        f"_wRealFace-{args.weight_loss_face_realistic}"
        f"_Th-{args.uncertainty_threshold}_lr-{args.learning_rate}"
    )
    
    args.imgs_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "imgs")
    args.ckpts_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "ckpts")

    if accelerator.is_main_process:
        os.makedirs(args.imgs_save_dir, exist_ok=True)
        os.makedirs(args.ckpts_save_dir, exist_ok=True)
        accelerator.init_trackers(
            args.proj_name, 
            init_kwargs = {
                "wandb": {
                    "name": folder_name, 
                    "dir": args.output_dir
                        }
                }
            )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
        )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder"
        )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet",
        )
    noise_scheduler = DPMSolverMultistepScheduler.from_config(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler",
        )
    # [ADDED] dedicated DDPM for forward noising used in SDS
    ddpm_forward = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    import torch
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    if args.train_text_encoder:
        eval_text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            )
        eval_text_encoder.requires_grad_(False)
        eval_text_encoder.to(accelerator.device, dtype=weight_dtype)
        
    if args.train_unet:        
        eval_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet",
        )
        eval_unet.requires_grad_(False)
        eval_unet.to(accelerator.device, dtype=weight_dtype)

    clip_model = CLIPModel.from_pretrained(args.zeroshot_model).to(accelerator.device)
    clip_model.eval().requires_grad_(False)
    clip_processor = CLIPProcessor.from_pretrained(args.zeroshot_model)
    # ---- zero-shot gender text embedding 미리 계산 (man / woman) ----
    gender_prompts = ["a photo of a woman", "a photo of a man"]
    with torch.no_grad():
        text_inputs = clip_processor(
            text=gender_prompts,
            return_tensors="pt",
            padding=True
        )
        text_inputs = {k: v.to(accelerator.device) for k, v in text_inputs.items()}
        gender_text_features = clip_model.get_text_features(**text_inputs)
        gender_text_features = F.normalize(gender_text_features, dim=-1)

    # Dedicated eval metrics models:
    # - CLIP-T / CLIP-I: CLIP ViT-bigG-14
    # - DINO_I: DINOv2 vit-g/14
    eval_clip_model_name = "ViT-bigG-14"
    eval_clip_pretrained = "laion2b_s39b_b160k"
    eval_dino_model_name = "dinov2_vitg14"
    clip_eval_model = None
    clip_eval_preprocess = None
    clip_eval_tokenizer = None
    dinov2_eval = None
    dinov2_eval_img_mean = None
    dinov2_eval_img_std = None
    if accelerator.is_main_process:
        clip_eval_precision = "fp32"
        if weight_dtype == torch.float16:
            clip_eval_precision = "fp16"
        elif weight_dtype == torch.bfloat16:
            clip_eval_precision = "bf16"

        clip_eval_model, _, clip_eval_preprocess = open_clip.create_model_and_transforms(
            eval_clip_model_name,
            pretrained=eval_clip_pretrained,
            precision=clip_eval_precision,
            device=accelerator.device,
        )
        clip_eval_model.eval().requires_grad_(False)
        clip_eval_tokenizer = open_clip.get_tokenizer(eval_clip_model_name)

        # NOTE: dinov2_eval is loaded ONLY on the main process (we are inside
        # `if accelerator.is_main_process`). There is therefore no cross-rank torch.hub
        # cache race to serialize here. Wrapping a main-process-only load in
        # accelerator.main_process_first() is a DEADLOCK: only rank 0 enters the context
        # and calls its internal barrier() at the end, while ranks 1..N never execute the
        # matching barrier -> NCCL hangs forever right after this load. Load directly
        # (matches 1-main-gender-sgd_dmscr_h_gen.py).
        dinov2_eval = torch.hub.load('facebookresearch/dinov2', eval_dino_model_name)
        dinov2_eval.to(accelerator.device, dtype=weight_dtype)
        dinov2_eval.requires_grad_(False)
        dinov2_eval.eval()
        dinov2_eval_img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
        dinov2_eval_img_std = torch.tensor([0.229, 0.224, 0.225]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warn(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #         vae.enable_xformers_memory_efficient_attention()
            
    #         if args.train_unet:
    #             eval_unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.train_unet:
        unet_lora_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            unet_lora_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=args.rank,
            ).to(accelerator.device)
            
        unet.set_attn_processor(unet_lora_procs)
        unet_lora_layers = AttnProcsLayers(unet.attn_processors)
        
        for p in unet_lora_layers.parameters():
            dist_broadcast_if_needed(p, src=0)
        
        unet_lora_ema = EMAModel(unet_lora_layers.parameters(), decay=args.EMA_decay)
        unet_lora_ema.to(accelerator.device)
        
        # print to check whether unet lora & ema is identical across devices
        print(f"{accelerator.device}; unet lora init to: {list(unet_lora_layers.parameters())[0].flatten()[1]:.6f}; unet lora ema init to: {unet_lora_ema.shadow_params[0].flatten()[1]:.6f}")

    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.rank, patch_mlp=True)
        
        for p in text_encoder_lora_params:
            dist_broadcast_if_needed(p, src=0)
                    
        text_encoder_lora_dict = {}
        text_encoder_lora_params_name_order = []
        for lora_param in text_encoder_lora_params:
            for name, param in text_encoder.named_parameters():
                if param is lora_param:
                    text_encoder_lora_dict[name] = lora_param
                    text_encoder_lora_params_name_order.append(name)
                    break
        assert text_encoder_lora_dict.__len__() == len(text_encoder_lora_params), "length does not match! something wrong happened while converting lora params to a state dict."

        # text_encoder_lora_params is randomly initiazed w/ different values at different processes
        # a hacky way to broadcast from main_process
        for name in text_encoder_lora_params_name_order:
            if accelerator.is_main_process:
                lora_param = text_encoder_lora_dict[name].detach().clone()
            else:
                lora_param = torch.zeros_like(text_encoder_lora_dict[name])
            dist_broadcast_if_needed(lora_param, src=0)
            text_encoder_lora_dict[name].data = lora_param

        class CustomModel(torch.nn.Module):
            def __init__(self, dict):
                """
                In the constructor we instantiate four parameters and assign them as
                member parameters.
                """
                super().__init__()
                self.param_names = list(dict.keys())
                self.params = nn.ParameterList()
                for name in self.param_names:
                    self.params.append( dict[name] )
            def forward(self, x):
                """
                no forward function
                """
                return None
        text_encoder_lora_model = CustomModel(text_encoder_lora_dict)

        text_encoder_lora_ema = EMAModel(text_encoder_lora_params, decay=args.EMA_decay)
        text_encoder_lora_ema.to(accelerator.device)

        text_encoder_lora_ema_dict = {}
        for name, shadow_param in itertools.zip_longest(text_encoder_lora_params_name_order, text_encoder_lora_ema.shadow_params):
            text_encoder_lora_ema_dict[name] = shadow_param
        assert text_encoder_lora_ema_dict.__len__() == text_encoder_lora_dict.__len__(), "length does not match! something wrong happened while converting lora params to a state dict."

        # print to check whether text_encoder lora & ema is identical across processes
        print(f"{accelerator.device}; TE lora init to: {list(text_encoder_lora_model.parameters())[0].flatten()[1]:.6f}; TE lora ema init to: {text_encoder_lora_ema.shadow_params[0].flatten()[1]:.6f}")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.train_text_encoder and args.train_unet:
        params_to_optimize = itertools.chain(unet_lora_layers.parameters(), text_encoder_lora_model.parameters())
    elif args.train_text_encoder and not args.train_unet:
        params_to_optimize = text_encoder_lora_model.parameters()
    elif not args.train_text_encoder and args.train_unet:
        params_to_optimize = unet_lora_layers.parameters()
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    with open(args.prompt_occupation_path, 'r') as f:
        experiment_data = json.load(f)
    prompts_train = [prompt.format(occupation=occupation) for prompt in experiment_data["prompt_templates_train"] for occupation in experiment_data["occupations_train_set"]]
    
    train_dataset = PromptsDataset(prompts=prompts_train)
    args.num_update_steps_per_epoch = train_dataset.__len__()
    args.num_train_epochs = math.ceil(args.max_train_steps / args.num_update_steps_per_epoch)

    # self-make a simple dataloader
    # the created train_dataloader_idxs should be identical across devices
    random.seed(args.seed+1)
    train_dataloader_idxs = []
    for epoch in range(args.num_train_epochs):
        idxs = list(range(train_dataset.__len__()))
        random.shuffle(idxs)
        train_dataloader_idxs.append(idxs)

    
    prompts_val = [prompt.format(occupation=occupation) for prompt in experiment_data["prompt_templates_test"] for occupation in experiment_data["occupations_test_set"]]
    
    
    #######################################################
    # set up things needed for finetuning        
    gender_classifier = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT, width_mult=1.0, reduced_tail=False, dilated=False)
    gender_classifier._modules['classifier'][3] = nn.Linear(1280, 2, bias=True)
    
    # NOTE: MobileNet classifier is used only for evaluation-time metrics.
    if not os.path.exists(args.classifier_weight_path):
        raise FileNotFoundError(
            f"[mnet] classifier weight not found at: {args.classifier_weight_path}\n"
            f"        Without these weights the classifier head is randomly initialized "
            f"and softmax outputs will hover around 0.5. "
            f"Pass the correct path via --classifier_weight_path."
        )
    gender_classifier.load_state_dict(torch.load(args.classifier_weight_path))
    if accelerator.is_main_process:
        logger.info(f"[mnet] loaded classifier weights from {args.classifier_weight_path}")
    gender_classifier.to(accelerator.device)
    gender_classifier.requires_grad_(False)
    gender_classifier.eval()
    
    # set up face_recognition and face_app on all devices (evaluation only)
    import face_recognition
    face_app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=['detection'], 
        providers=['CUDAExecutionProvider'], 
        provider_options=[{'device_id': accelerator.device.index}]
        )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    
    clip_image_processoor = CLIPImageProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    clip_vision_model_w_proj = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    clip_vision_model_w_proj.vision_model.to(accelerator.device, dtype=weight_dtype)
    clip_vision_model_w_proj.visual_projection.to(accelerator.device, dtype=weight_dtype)
    clip_vision_model_w_proj.requires_grad_(False)
    clip_vision_model_w_proj.gradient_checkpointing_enable()
    clip_img_mean = torch.tensor(clip_image_processoor.image_mean).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype) # mean is based on range [0,1]
    clip_img_std = torch.tensor(clip_image_processoor.image_std).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype) # std is based on range [0,1]
    

    # Serialize the torch.hub download/extract across ranks. On a cold cache every rank
    # otherwise downloads + extracts dinov2 into the same dir concurrently, and one rank's
    # shutil.rmtree collides with another rank still extracting (OSError [Errno 39]
    # Directory not empty: 'data'). Main process populates the cache first, barrier, then
    # all ranks load from a warm cache.
    with accelerator.main_process_first():
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2.to(accelerator.device, dtype=weight_dtype)
    dinov2.requires_grad_(False)
    dinov2_img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
    dinov2_img_std = torch.tensor([0.229, 0.224, 0.225]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
    
    CE_loss = nn.CrossEntropyLoss(reduction="none")   

    # Face-feature preservation models are intentionally skipped in this run.
    # Training uses only fair/img/realistic-face losses; MobileNet is eval-only.

    #######################################################
    
    @torch.no_grad()
    def generate_image_no_gradient(
        prompt,
        noises,
        num_denoising_steps,
        which_text_encoder,
        which_unet,
        skip_final_steps: int = 0,
        skip_final_steps_pct: float = 0.0,
        return_z0_latents: bool = False,
    ):
        """
        prompts: str
        noises: [N,4,64,64], N is number images to be generated for the prompt
        """
        N = noises.shape[0]
        prompts = [prompt] * N
        
        prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
        prompts_token["input_ids"] = prompts_token["input_ids"].to(accelerator.device)
        prompts_token["attention_mask"] = prompts_token["attention_mask"].to(accelerator.device)

        prompt_embeds = which_text_encoder(
            prompts_token["input_ids"],
            prompts_token["attention_mask"],
        )
        prompt_embeds = prompt_embeds[0]

        batch_size = prompt_embeds.shape[0]
        uncond_tokens = [""] * batch_size
        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
        uncond_input["input_ids"] = uncond_input["input_ids"].to(accelerator.device)
        uncond_input["attention_mask"] = uncond_input["attention_mask"].to(accelerator.device)
        negative_prompt_embeds = which_text_encoder(
            uncond_input["input_ids"],
            uncond_input["attention_mask"],
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds = prompt_embeds.to(weight_dtype)

        noise_scheduler.set_timesteps(num_denoising_steps)
        timesteps = noise_scheduler.timesteps
        total_steps = len(timesteps)

        skip_final_steps_pct = 0.0 if skip_final_steps_pct is None else float(skip_final_steps_pct)
        if skip_final_steps_pct < 0:
            raise ValueError("skip_final_steps_pct must be non-negative.")
        skip_from_pct = 0
        if skip_final_steps_pct > 0:
            skip_from_pct = int(math.floor(total_steps * skip_final_steps_pct / 100.0 + 0.5))

        skip_final_steps = skip_from_pct if skip_from_pct > 0 else skip_final_steps
        skip_final_steps = 0 if skip_final_steps is None else int(skip_final_steps)
        if skip_final_steps < 0:
            raise ValueError("skip_final_steps must be non-negative.")
        if skip_final_steps >= total_steps and total_steps > 0:
            skip_final_steps = total_steps - 1

        steps_to_run = total_steps - skip_final_steps
        latents = noises
        for i, t in enumerate(timesteps[:steps_to_run]):

            # scale model input
            latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            noises_pred = which_unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(weight_dtype_high_precision)

            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + args.guidance_scale * (noises_pred_text - noises_pred_uncond)

            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample
        if skip_final_steps > 0:
            t_cur = timesteps[steps_to_run]

            latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t_cur)

            eps = which_unet(
                latent_model_input,
                t_cur,
                encoder_hidden_states=prompt_embeds,
            ).sample
            eps = eps.to(weight_dtype_high_precision)

            eps_u, eps_c = eps.chunk(2)
            eps = eps_u + args.guidance_scale * (eps_c - eps_u)

            step_out = noise_scheduler.step(eps, t_cur, latents)
            if hasattr(step_out, "pred_original_sample") and step_out.pred_original_sample is not None:
                latents = step_out.pred_original_sample
            else:
                alpha_bar = noise_scheduler.alphas_cumprod[t_cur].to(device=latents.device, dtype=latents.dtype)
                latents = (latents - (1 - alpha_bar).sqrt() * eps.to(latents.dtype)) / alpha_bar.sqrt()

        z0_latents = latents
        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]

        att = None
        if return_z0_latents:
            return images, att, z0_latents
        return images, att
    
    def generate_image_w_gradient(
        prompt,
        noises,
        num_denoising_steps,
        which_text_encoder,
        which_unet,
        skip_final_steps: int = 0,
        skip_final_steps_pct: float = 0.0,
        return_z0_latents: bool = False,
        return_hspace_loss: bool = False,
        h_reference_text_encoder=None,
        h_reference_unet=None,
    ):
        which_unet.train()
        ref_is_same_unet = (h_reference_unet is which_unet)
        if return_hspace_loss:
            if h_reference_text_encoder is None or h_reference_unet is None:
                raise ValueError(
                    "return_hspace_loss=True requires h_reference_text_encoder and h_reference_unet."
                )
            # Keep checkpointing active when train_unet=False:
            # in that case both handles point to the same `unet`, so calling eval()
            # here would flip the training UNet to eval mode and disable gradient checkpointing.
            if not ref_is_same_unet:
                h_reference_unet.eval()

        N = noises.shape[0]
        prompts = [prompt] * N

        prompts_token = tokenizer(prompts, return_tensors="pt", padding=True).to(accelerator.device)
        prompt_embeds = which_text_encoder(prompts_token["input_ids"], prompts_token["attention_mask"])[0]

        uncond_input = tokenizer([""] * N, padding="max_length", max_length=prompt_embeds.shape[1],
                                truncation=True, return_tensors="pt").to(accelerator.device)
        negative_prompt_embeds = which_text_encoder(uncond_input["input_ids"], uncond_input["attention_mask"])[0]
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]).to(weight_dtype)

        ref_prompt_embeds = None
        if return_hspace_loss:
            with torch.no_grad():
                ref_prompts_token = tokenizer(prompts, return_tensors="pt", padding=True).to(accelerator.device)
                ref_prompt_embeds = h_reference_text_encoder(
                    ref_prompts_token["input_ids"], ref_prompts_token["attention_mask"]
                )[0]
                ref_uncond_input = tokenizer(
                    [""] * N,
                    padding="max_length",
                    max_length=ref_prompt_embeds.shape[1],
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)
                ref_negative_prompt_embeds = h_reference_text_encoder(
                    ref_uncond_input["input_ids"], ref_uncond_input["attention_mask"]
                )[0]
                ref_prompt_embeds = torch.cat([ref_negative_prompt_embeds, ref_prompt_embeds]).to(weight_dtype)

        noise_scheduler.set_timesteps(num_denoising_steps)
        timesteps = noise_scheduler.timesteps
        total_steps = len(timesteps)

        skip_final_steps_pct = 0.0 if skip_final_steps_pct is None else float(skip_final_steps_pct)
        if skip_final_steps_pct < 0:
            raise ValueError("skip_final_steps_pct must be non-negative.")
        skip_from_pct = 0
        if skip_final_steps_pct > 0:
            skip_from_pct = int(math.floor(total_steps * skip_final_steps_pct / 100.0 + 0.5))

        skip_final_steps = skip_from_pct if skip_from_pct > 0 else skip_final_steps
        skip_final_steps = 0 if skip_final_steps is None else int(skip_final_steps)
        if skip_final_steps < 0:
            raise ValueError("skip_final_steps must be non-negative.")
        if skip_final_steps >= total_steps and total_steps > 0:
            skip_final_steps = total_steps - 1

        # grad_coefs (원 코드 유지)
        grad_coefs = []
        for i, t in enumerate(timesteps):
            grad_coefs.append(
                noise_scheduler.alphas_cumprod[t].sqrt().item()
                * (1 - noise_scheduler.alphas_cumprod[t]).sqrt().item()
                / (1 - noise_scheduler.alphas[t].item())
            )
        grad_coefs = np.array(grad_coefs)
        grad_coefs /= (math.prod(grad_coefs) ** (1 / len(grad_coefs)))

        steps_to_run = total_steps - skip_final_steps
        latents = noises
        latents_ref = noises.detach().clone() if return_hspace_loss else None
        hspace_terms = []

        mid_capture_fine = MidBlockCapture().attach(which_unet) if return_hspace_loss else None
        mid_capture_ref = None
        if return_hspace_loss:
            if ref_is_same_unet:
                mid_capture_ref = mid_capture_fine
            else:
                mid_capture_ref = MidBlockCapture().attach(h_reference_unet)

        try:
            # compare h-space for every denoising step actually run (exclude skipped final tail)
            for i, t in enumerate(timesteps[:steps_to_run]):
                latent_model_input = torch.cat([latents.detach().to(weight_dtype)] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                if return_hspace_loss:
                    mid_capture_fine.last = None

                eps = which_unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
                h_fine = mid_capture_fine.last if return_hspace_loss else None
                eps = eps.to(weight_dtype_high_precision)

                eps_u, eps_c = eps.chunk(2)
                eps = eps_u + args.guidance_scale * (eps_c - eps_u)
                eps.register_hook(make_grad_hook(grad_coefs[i]))
                latents = noise_scheduler.step(eps, t, latents).prev_sample

                if return_hspace_loss:
                    with torch.no_grad():
                        latent_model_input_ref = torch.cat([latents_ref.to(weight_dtype)] * 2)
                        latent_model_input_ref = noise_scheduler.scale_model_input(latent_model_input_ref, t)
                        mid_capture_ref.last = None
                        eps_ref = h_reference_unet(
                            latent_model_input_ref,
                            t,
                            encoder_hidden_states=ref_prompt_embeds,
                        ).sample
                        h_ref = mid_capture_ref.last
                        eps_ref = eps_ref.to(weight_dtype_high_precision)
                        eps_ref_u, eps_ref_c = eps_ref.chunk(2)
                        eps_ref = eps_ref_u + args.guidance_scale * (eps_ref_c - eps_ref_u)
                        latents_ref = noise_scheduler.step(eps_ref, t, latents_ref).prev_sample

                    if h_fine is None or h_ref is None:
                        raise RuntimeError("Failed to capture mid_block h-space during denoising.")

                    h_fine_cond = h_fine
                    h_ref_cond = h_ref
                    if h_fine_cond.shape[0] == 2 * N:
                        h_fine_cond = h_fine_cond[N:]
                    if h_ref_cond.shape[0] == 2 * N:
                        h_ref_cond = h_ref_cond[N:]
                    if h_fine_cond.shape != h_ref_cond.shape:
                        raise RuntimeError(
                            f"h-space shape mismatch: fine={tuple(h_fine_cond.shape)} ref={tuple(h_ref_cond.shape)}"
                        )

                    form = getattr(args, "h_loss_form", "raw")
                    if form == "cos":
                        loss_t = 1.0 - F.cosine_similarity(
                            h_fine_cond.float().flatten(1),
                            h_ref_cond.detach().float().flatten(1),
                            dim=1,
                        )
                    else:
                        loss_t = ((h_fine_cond.float() - h_ref_cond.detach().float()) ** 2).mean(dim=(1, 2, 3))
                    hspace_terms.append(loss_t)

            # keep original x0-jump behavior for image generation when final steps are skipped
            if skip_final_steps > 0:
                t_cur = timesteps[steps_to_run]

                latent_model_input = torch.cat([latents.detach().to(weight_dtype)] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t_cur)

                eps = which_unet(latent_model_input, t_cur, encoder_hidden_states=prompt_embeds).sample
                eps = eps.to(weight_dtype_high_precision)

                eps_u, eps_c = eps.chunk(2)
                eps = eps_u + args.guidance_scale * (eps_c - eps_u)
                eps.register_hook(make_grad_hook(grad_coefs[steps_to_run]))

                step_out = noise_scheduler.step(eps, t_cur, latents)
                if hasattr(step_out, "pred_original_sample") and step_out.pred_original_sample is not None:
                    latents = step_out.pred_original_sample
                else:
                    alpha_bar = noise_scheduler.alphas_cumprod[t_cur].to(device=latents.device, dtype=latents.dtype)
                    latents = (latents - (1 - alpha_bar).sqrt() * eps.to(latents.dtype)) / alpha_bar.sqrt()
        finally:
            if mid_capture_fine is not None:
                mid_capture_fine.detach()
            if mid_capture_ref is not None and mid_capture_ref is not mid_capture_fine:
                mid_capture_ref.detach()

        z0_latents = latents
        latents = latents / vae.config.scaling_factor
        images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)

        att = None
        hspace_loss = None
        if return_hspace_loss:
            if len(hspace_terms) == 0:
                hspace_loss = torch.zeros([N], dtype=weight_dtype_high_precision, device=accelerator.device)
            else:
                hspace_loss = torch.stack(hspace_terms, dim=0).mean(dim=0)
        if return_z0_latents:
            if return_hspace_loss:
                return images, att, z0_latents, hspace_loss
            return images, att, z0_latents
        if return_hspace_loss:
            return images, att, hspace_loss
        return images, att

    
    def get_clip_feat(images, normalize=True, to_high_precision=True):
        """get clip features

        Args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1]
            normalize (bool):
            to_high_precision (bool):

        Returns:
            embeds (torch.tensor)
        """
        images_preprocessed = ((images+1)*0.5 - clip_img_mean) / clip_img_std
        embeds = clip_vision_model_w_proj(images_preprocessed).image_embeds
        
        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds
    
    def get_dino_feat(images, normalize=True, to_high_precision=True):
        """get dino features

        Args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1]
            normalize (bool):
            to_high_precision (bool):

        Returns:
            embeds (torch.tensor)
        """
        images_preprocessed = ((images+1)*0.5 - dinov2_img_mean) / dinov2_img_std
        embeds = dinov2(images_preprocessed)
        
        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds

    def get_dino_eval_feat(images, normalize=True, to_high_precision=True):
        """Get evaluation-time DINOv2 vit-g/14 features."""
        images_preprocessed = ((images+1)*0.5 - dinov2_eval_img_mean) / dinov2_eval_img_std
        embeds = dinov2_eval(images_preprocessed)

        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds
    
    def get_face_feats(net, data, flip=True, normalize=True, to_high_precision=True):
        # extract features from the original 
        # and horizontally flipped data
        feats = net(data)
        if flip:
            data = torch.flip(data, [3])
            feats += net(data)
        if to_high_precision:
            feats = feats.to(torch.float)
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats
        
    # =========================
    # [ADDED] helpers for region masks
    # =========================
    def _build_face_region_mask_from_bboxes(face_indicators: torch.Tensor,
                                            face_bboxes: torch.Tensor,
                                            out_h: int, out_w: int,
                                            img_h: int, img_w: int,
                                            device, dtype):
        """
        Returns [B, out_h, out_w] binary mask.
        - If face not detected => all ones (use full image as requested).
        - If detected => ones only inside scaled bbox, zeros elsewhere.
        """
        B = face_bboxes.shape[0]
        mask = torch.zeros((B, out_h, out_w), device=device, dtype=dtype)
        for i in range(B):
            if bool(face_indicators[i].item()):
                x1, y1, x2, y2 = face_bboxes[i].tolist()
                # clamp to image
                x1 = max(0, min(int(round(x1)), img_w - 1))
                x2 = max(0, min(int(round(x2)), img_w))
                y1 = max(0, min(int(round(y1)), img_h - 1))
                y2 = max(0, min(int(round(y2)), img_h))
                if x2 <= x1 or y2 <= y1:
                    # degenerate => full mask
                    mask[i, :, :] = 1
                else:
                    # scale to latent/eps grid
                    sx1 = int(round(x1 * (out_w / float(img_w))))
                    sx2 = int(round(x2 * (out_w / float(img_w))))
                    sy1 = int(round(y1 * (out_h / float(img_h))))
                    sy2 = int(round(y2 * (out_h / float(img_h))))
                    sx1 = max(0, min(sx1, out_w - 1))
                    sx2 = max(0, min(sx2, out_w))
                    sy1 = max(0, min(sy1, out_h - 1))
                    sy2 = max(0, min(sy2, out_h))
                    if sx2 <= sx1 or sy2 <= sy1:
                        mask[i, :, :] = 1
                    else:
                        mask[i, sy1:sy2, sx1:sx2] = 1
            else:
                # no face -> full image
                mask[i, :, :] = 1
        return mask

    def _text_embeds(prompts: list, which_text_encoder):
        toks = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        input_ids = toks.input_ids.to(accelerator.device)
        attn_mask = toks.attention_mask.to(accelerator.device)
        return which_text_encoder(input_ids, attn_mask)[0]


    def find_token_positions(tokenizer, prompt: str, keywords: List[str]) -> List[int]:
        """Find token piece indices that contain any keyword (case-insensitive).
        Returns list of indices (may be empty).
        """
        toks = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        ids = toks.input_ids[0]
        pieces = tokenizer.convert_ids_to_tokens(ids)
        kws = [k.lower() for k in keywords]
        pos = []
        for i, piece in enumerate(pieces):
            p = piece.lower().replace("Ġ", "").replace("▁", "")
            if any(k in p for k in kws):
                pos.append(i)
        return pos

    def _attn_weight_for_prompt(latent_inputs: torch.Tensor, t_vec: torch.Tensor, prompt: str):
        """
        Compute mid-block cross-attn map focused on tokens ['face'] with NO scaling/normalization.
        Returns [B, H, W] in the latent spatial size. Runs under torch.no_grad() and restores processors afterward.
        """
        store = _AttnStore()
        keywords = ["female","male"]
        token_mask = _build_token_mask(tokenizer, prompt, keywords=keywords, device=accelerator.device)
        orig, rec = _install_mid_recorders(unet, store, token_mask)

        with torch.no_grad():
            pe = _text_embeds([prompt]*latent_inputs.shape[0], text_encoder).to(weight_dtype)
            _ = unet(latent_inputs.to(weight_dtype), t_vec, encoder_hidden_states=pe)

        _restore_attn_processors(unet, orig)
        attn_vec = store.aggregate()  # [B,Q]
        if attn_vec is None:
            B, _, H, W = latent_inputs.shape
            return torch.ones([B, H, W], device=accelerator.device, dtype=weight_dtype)

        B = attn_vec.shape[0]
        Q = attn_vec.shape[1]
        S = int(math.sqrt(Q))
        attn_map = attn_vec.view(B, 1, S, S)

        H_lat, W_lat = latent_inputs.shape[-2:]
        attn_map = torch.nn.functional.interpolate(
            attn_map, size=(H_lat, W_lat), mode="bilinear", align_corners=False
        ).squeeze(1)

        return attn_map.to(weight_dtype)

    # =========================
    # [MODIFIED] SDS-based classifier utilities
    #   - region_mask_mode에 따라 attn/face/none 마스크 적용
    #   - 마스크는 err_f/err_m에 곱해져 경사가 얼굴 영역 밖으로 흐르지 않음
    # =========================
    def sds_logits_from_images(
        images: torch.Tensor,
        tau: float,
        t_min: int,
        t_max: int,
        num_t: int,
        num_eps: int,
        gate_grad: bool,
        precomputed_attmaps: Optional[torch.Tensor] = None,
        sds_text_encoder=None,
        sds_unet=None,
        sds_scheduler=None,
        compute_realistic_sds: bool = False,
        return_attmap: bool = False,
    ):
        """
        - region_mask_mode == 'face': InsightFace로 검출된 얼굴 bbox를 latent/eps 해상도에 맞춰 이진 마스크 생성(미검출=full 1)
        - region_mask_mode == 'attn': SDS 분류 프롬프트(woman/man)의 토큰 어텐션맵을 헤드/블록/timestep 평균해 사용
        - region_mask_mode == 'none': 마스크 미적용
        """
        B = images.shape[0]

        lat = vae.encode(images.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor  # [B,4,64,64]

        # --- t 인덱스/확장 ---
        t_idx = torch.linspace(t_min, t_max, steps=num_t, device=accelerator.device).round().long()
        K = t_idx.shape[0]
        lat_exp = lat.unsqueeze(1).unsqueeze(2).expand(B, K, num_eps, *lat.shape[1:]).contiguous().view(B*K*num_eps, *lat.shape[1:])
        t_vec = t_idx.view(1, K, 1).expand(B, K, num_eps).reshape(-1)
        BKE = lat_exp.shape[0]

        # --- eps & z_t (dtype 유지) ---
        eps = torch.randn_like(lat_exp, dtype=weight_dtype)
        alpha_bar = ddpm_forward.alphas_cumprod.to(device=accelerator.device, dtype=lat_exp.dtype)[t_vec]
        zt = alpha_bar.sqrt().view(-1,1,1,1) * lat_exp + (1.0 - alpha_bar).sqrt().view(-1,1,1,1) * eps

        # --- 텍스트 임베딩 ---
        female_prompt = "a photo of a woman"
        male_prompt   = "a photo of a man"
        realistic_prompt = "a photo of a realistic face"
        pe_f = _text_embeds([female_prompt]*BKE, sds_text_encoder).to(weight_dtype)
        pe_m = _text_embeds([male_prompt]*BKE,   sds_text_encoder).to(weight_dtype)
        pe_r = None
        if compute_realistic_sds:
            pe_r = _text_embeds([realistic_prompt]*BKE, sds_text_encoder).to(weight_dtype)

        # --- region weight map (w_map): face / attn / none ---
        w_map = None  # [BKE,H,W] or None
        attmap_mean = None  # [B,H,W] or None

        # 해상도(H,W): eps_pred_*와 동일(보통 64x64)
        _H = zt.shape[-2]
        _W = zt.shape[-1]

        def _forward_with_optional_attn(prompt_embeds: torch.Tensor, token_positions: List[int]):
            capture = None
            att_bke = None
            # Install the attention capture when the map is needed either as the SDS region
            # weight (region_mask_mode == "attn") OR simply for saving/visualization
            # (return_attmap). This lets --save_attmaps produce attmaps in every
            # region_mask_mode ("none"/"face"/"attn") without changing the SDS loss.
            want_attn_weight = (args.region_mask_mode == "attn" and args.use_attn_weight)
            if (want_attn_weight or return_attmap) and len(token_positions) > 0:
                capture = CrossAttnCapture(
                    token_indices=token_positions,
                    use_cpu=False,
                    expect_cfg_pair=False,
                ).add_hooks(sds_unet)
            try:
                eps_pred = sds_unet(zt.to(weight_dtype), t_vec, encoder_hidden_states=prompt_embeds).sample.to(weight_dtype)
                if capture is not None:
                    att_bke = capture.aggregated_map()
            finally:
                if capture is not None:
                    capture.clear()
            return eps_pred, att_bke

        tok_pos_f = find_token_positions(tokenizer, female_prompt, keywords=["woman"])
        tok_pos_m = find_token_positions(tokenizer, male_prompt, keywords=["man"])

        # --- UNet 예측 + prompt-token attention map 수집 ---
        eps_pred_f, att_f_bke = _forward_with_optional_attn(pe_f, tok_pos_f)
        eps_pred_m, att_m_bke = _forward_with_optional_attn(pe_m, tok_pos_m)
        eps_pred_r = None
        if compute_realistic_sds:
            eps_pred_r = sds_unet(zt.to(weight_dtype), t_vec, encoder_hidden_states=pe_r).sample.to(weight_dtype)

        # region_mask_mode == 'face' : 얼굴 bbox 기반 하드 마스크
        if args.region_mask_mode == "face":
            with torch.no_grad():
                # 얼굴 검출은 B개의 이미지에서 1회만 수행
                face_indicators_b, face_bboxs_b, _, _, _ = get_face(images)  # bbox는 image 해상도 기준
                img_h, img_w = images.shape[-2], images.shape[-1]
                face_mask_b = _build_face_region_mask_from_bboxes(
                    face_indicators_b, face_bboxs_b,
                    out_h=_H, out_w=_W,
                    img_h=img_h, img_w=img_w,
                    device=accelerator.device, dtype=weight_dtype
                )  # [B,_H,_W] 이진
                # K, E 로 복제
                w_map = face_mask_b.unsqueeze(1).unsqueeze(2).expand(B, K, num_eps, _H, _W).contiguous().view(BKE, _H, _W)

        # woman/man 토큰 attmap 평균 계산 (어텐션이 캡처된 경우).
        # region_mask_mode와 무관하게 계산해서 --save_attmaps 저장에 사용하고,
        # 'attn' 모드에서는 아래에서 이 값을 그대로 SDS region weight(w_map)로 재사용한다.
        if att_f_bke is not None and att_m_bke is not None:
            a_f = att_f_bke.to(device=accelerator.device, dtype=torch.float32)
            a_m = att_m_bke.to(device=accelerator.device, dtype=torch.float32)
            if a_f.shape[-2:] != (_H, _W):
                a_f = F.interpolate(a_f.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False).squeeze(1)
            if a_m.shape[-2:] != (_H, _W):
                a_m = F.interpolate(a_m.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False).squeeze(1)

            att_f = a_f.reshape(B, K, num_eps, _H, _W).mean(dim=(1, 2))  # [B,H,W]
            att_m = a_m.reshape(B, K, num_eps, _H, _W).mean(dim=(1, 2))  # [B,H,W]
            attmap_mean = 0.5 * (att_f + att_m)

        # region_mask_mode == 'attn': woman/man 토큰 attmap 평균을 SDS region weight로 사용
        if args.region_mask_mode == "attn":
            if args.use_attn_weight and attmap_mean is not None:
                w_map = (
                    attmap_mean.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(B, K, num_eps, _H, _W)
                    .contiguous()
                    .view(BKE, _H, _W)
                    .to(weight_dtype)
                )
            else:
                # fallback: 기존 precomputed attmap 지원
                attmap = precomputed_attmaps
                if isinstance(attmap, list):
                    try:
                        att_batches = []
                        for a in attmap:
                            if a is None:
                                continue
                            if torch.is_tensor(a):
                                att_batches.append(a.to(device=accelerator.device, dtype=weight_dtype))
                            else:
                                att_batches.append(torch.tensor(a, device=accelerator.device, dtype=weight_dtype))
                        if len(att_batches) > 0:
                            attmap = torch.cat(att_batches, dim=0)
                        else:
                            attmap = None
                    except Exception:
                        attmap = None

                if isinstance(attmap, torch.Tensor):
                    a = attmap.to(device=accelerator.device, dtype=weight_dtype)
                    if a.dim() == 2:
                        attmap_mean = a.unsqueeze(0).expand(B, _H, _W).contiguous()
                    elif a.dim() == 3:
                        if a.shape[0] == BKE:
                            attmap_mean = a.reshape(B, K, num_eps, _H, _W).mean(dim=(1, 2))
                        elif a.shape[0] == B:
                            attmap_mean = a
                        else:
                            attmap_mean = a.mean(dim=0, keepdim=True).expand(B, _H, _W).contiguous()
                    if attmap_mean is not None:
                        w_map = (
                            attmap_mean.unsqueeze(1)
                            .unsqueeze(2)
                            .expand(B, K, num_eps, _H, _W)
                            .contiguous()
                            .view(BKE, _H, _W)
                        )

        # --- SDS(가중 MSE; 같은 맵으로 f/m 모두 가중) ---
        err_f = ((eps_pred_f - eps)**2).mean(dim=1)  # [BKE,H,W]
        err_m = ((eps_pred_m - eps)**2).mean(dim=1)  # [BKE,H,W]
        err_r = None
        if compute_realistic_sds:
            err_r = ((eps_pred_r - eps)**2).mean(dim=1)  # [BKE,H,W]
        if w_map is not None:
            # Convert w_map into per-sample sum-to-one weights, then rescale by H*W so
            # the final reduction remains a spatial mean with average weight 1.
            w = w_map.to(err_f.dtype).clamp_min(0)
            w_sum = w.flatten(1).sum(dim=1).clamp_min(1e-8)  # [BKE]
            hw = err_f.shape[-2] * err_f.shape[-1]
            w = (w / w_sum.view(-1, 1, 1)) * hw
            sds_f_per = (err_f * w).flatten(1).mean(dim=1)  # [BKE]
            sds_m_per = (err_m * w).flatten(1).mean(dim=1)  # [BKE]
        else:
            # 마스크 없음
            sds_f_per = err_f.flatten(1).mean(dim=1)  # [BKE]
            sds_m_per = err_m.flatten(1).mean(dim=1)  # [BKE]
        sds_r_per = None
        if compute_realistic_sds:
            # realistic SDS always uses full pixels (no w_map)
            sds_r_per = err_r.flatten(1).mean(dim=1)

        # --- 로짓/확률: fp32 강제 계산 & fp32 출력 ---
        with torch.autocast("cuda", enabled=False):
            # 1) [B*K*E] -> [B,K,E] -> (K,E) 평균 => [B]
            sds_f32 = sds_f_per.float().view(B, K, num_eps).mean(dim=(1, 2))  # [B], fp32
            sds_m32 = sds_m_per.float().view(B, K, num_eps).mean(dim=(1, 2))  # [B], fp32
            sds_r32 = None
            if sds_r_per is not None:
                sds_r32 = sds_r_per.float().view(B, K, num_eps).mean(dim=(1, 2))  # [B], fp32

            tau32   = torch.tensor(tau, device=accelerator.device, dtype=torch.float32)

            # 2) logits/probs shape = [B,2]
            logits32 = torch.stack([-sds_f32 / tau32, -sds_m32 / tau32], dim=1)  # [B,2], fp32

            # 3) 수치 안정화(softmax 불변 변환): 행별 최대값 빼기
            logits32 = logits32 - logits32.max(dim=1, keepdim=True).values

            probs32  = torch.softmax(logits32, dim=1)  # [B,2], fp32

        preds = probs32.argmax(dim=1)  # [B]

        # --- 안전성 검사: NaN/Inf 및 소프트맥스 행합 ---
        row_sums   = probs32.sum(dim=1)                       # [B]
        bad_logits = (~torch.isfinite(logits32)).any(dim=1)   # [B]
        bad_probs  = (~torch.isfinite(probs32)).any(dim=1)    # [B]
        bad_sum    = (~torch.isfinite(row_sums)) | ((row_sums - 1.0).abs() > 1e-4)
        bad_rows   = bad_logits | bad_probs | bad_sum         # [B]

        if bad_rows.any():
            idx = torch.nonzero(bad_rows).squeeze(1)
            print(f"[ALERT] non-finite detected in logits/probs (rows={idx.tolist()})")
            print("tau:", float(tau32))
            print("sds_f32[min,max]:", float(sds_f32.min()), float(sds_f32.max()))
            print("sds_m32[min,max]:", float(sds_m32.min()), float(sds_m32.max()))
            print("logits32 (bad rows):\n", logits32[idx])
            print("probs32 (bad rows) & row sums:\n", probs32[idx], "\nrow_sums:", row_sums[idx])
            raise FloatingPointError("Non-finite in logits/probs")

        if return_attmap:
            return preds, probs32, logits32, sds_f32, sds_m32, sds_r32, attmap_mean
        return preds, probs32, logits32, sds_f32, sds_m32, sds_r32

    def clip_gender_classifier(
        images: torch.Tensor,
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        text_features: torch.Tensor,
        device: torch.device,
    ):
        """
        images: [B, 3, H, W]  (보통 [-1,1] 범위라고 가정)
        반환:
            preds:  [B] int64       (0: woman, 1: man)
            probs:  [B, 2] float32
            logits: [B, 2] float32
        """

        with torch.no_grad():
            # 1) [-1,1] -> [0,1]
            imgs = images.detach().cpu()
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

            # 2) Tensor -> PIL
            to_pil = T.ToPILImage()
            pil_images = [to_pil(img) for img in imgs]

            # 3) CLIPProcessor로 이미지 전처리
            img_inputs = clip_processor(
                images=pil_images,
                return_tensors="pt"
            )
            pixel_values = img_inputs["pixel_values"].to(device)

            # 4) image feature 추출
            image_features = clip_model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)

            # 5) text_features와 cosine similarity 기반 logits 계산
            #    CLIP의 logit_scale까지 같이 사용 (원래 forward와 동일한 방식)
            logit_scale = clip_model.logit_scale.exp()
            logits = (image_features @ text_features.t()) * logit_scale

            # 6) 확률 & 예측 클래스
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

        return preds, probs, logits


    # =======================================
    # Original face helpers (kept for viz/face-loss only)
    # =======================================
    def get_face(images, fill_value=-1):
        """
        images:shape [N,3,H,W], in range [-1,1], pytorch tensor
        returns:
            face_indicators: torch tensor of shape [N], only True or False
                True means face is detected, False otherwise
            face_bboxs: torch tensor of shape [N,4], 
                if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
            face_chips: torch tensor of shape [N,3,224,224]
                if face_indicator is False, the corresponding face_chip will be all fill_value
        """
        face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app = get_face_app(images, fill_value=fill_value)

        # Keep the same fallback behavior as ft.py:
        # if InsightFace misses a face, retry that subset with face_recognition.
        if face_indicators_app.logical_not().sum() > 0:
            face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR = get_face_FR(
                images[face_indicators_app.logical_not()],
                fill_value=fill_value,
            )

            face_bboxs_app[face_indicators_app.logical_not()] = face_bboxs_FR
            face_chips_app[face_indicators_app.logical_not()] = face_chips_FR
            face_landmarks_app[face_indicators_app.logical_not()] = face_landmarks_FR
            aligned_face_chips_app[face_indicators_app.logical_not()] = aligned_face_chips_FR
            face_indicators_app[face_indicators_app.logical_not()] = face_indicators_FR

        return face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app

    
    def get_largest_face_FR(faces_from_FR, dim_max, dim_min):
        if len(faces_from_FR) == 1:
            return faces_from_FR[0]
        elif len(faces_from_FR) > 1:
            area_max = 0
            idx_max = 0
            for idx, bbox in enumerate(faces_from_FR):
                bbox1 = np.array((bbox[-1],) + bbox[:-1])
                area = (min(bbox1[2],dim_max) - max(bbox1[0], dim_min)) * (min(bbox1[3],dim_max) - max(bbox1[1], dim_min))
                if area > area_max:
                    area_max = area
                    idx_max = idx
            return faces_from_FR[idx_max]
    
    def get_face_FR(images, fill_value=-1):
        """
        images:shape [N,3,H,W], in range [-1,1], pytorch tensor
        returns:
            face_indicators: torch tensor of shape [N], only True or False
                True means face is detected, False otherwise
            face_bboxs: torch tensor of shape [N,4], 
                if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
            face_chips: torch tensor of shape [N,3,224,224]
                if face_indicator is False, the corresponding face_chip will be all fill_value
        """

        images_np = ((images*0.5 + 0.5)*255).cpu().detach().permute(0,2,3,1).float().numpy().astype(np.uint8)
        
        face_indicators_FR = []
        face_bboxs_FR = []
        face_chips_FR = []
        face_landmarks_FR = []
        aligned_face_chips_FR = []
        for idx, image_np in enumerate(images_np):
            faces_from_FR = face_recognition.face_locations(image_np, model="cnn", number_of_times_to_upsample=0)
            if len(faces_from_FR) == 0:
                face_indicators_FR.append(False)
                face_bboxs_FR.append([fill_value]*4)
                face_chips_FR.append(torch.ones([1,3,args.size_face,args.size_face], dtype=images.dtype, device=images.device)*(fill_value))
                face_landmarks_FR.append(torch.ones([1,5,2], dtype=images.dtype, device=images.device)*(fill_value))
                aligned_face_chips_FR.append(torch.ones([1,3,args.size_aligned_face,args.size_aligned_face], dtype=images.dtype, device=images.device)*(fill_value))
            else:
                face_from_FR = get_largest_face_FR(faces_from_FR, dim_max=image_np.shape[0], dim_min=0)
                bbox = face_from_FR
                bbox = np.array((bbox[-1],) + bbox[:-1]) # need to convert bbox from face_recognition to the right order
                bbox = expand_bbox(bbox, expand_coef=1.1, target_ratio=1) # need to use a larger expand_coef for FR
                face_chip = crop_face(images[idx], bbox, target_size=[args.size_face,args.size_face], fill_value=fill_value)
                
                face_landmarks = face_recognition.face_landmarks(image_np, face_locations=[face_from_FR], model="large")

                left_eye = np.array(face_landmarks[0]["left_eye"]).mean(axis=0)
                right_eye = np.array(face_landmarks[0]["right_eye"]).mean(axis=0)
                nose_tip = np.array(face_landmarks[0]["nose_bridge"][-1])
                top_lip_left = np.array(face_landmarks[0]["top_lip"][0])
                top_lip_right = np.array(face_landmarks[0]["top_lip"][6])
                face_landmarks = np.stack([left_eye, right_eye, nose_tip, top_lip_left, top_lip_right])
                
                aligned_face_chip = image_pipeline(images[idx], face_landmarks)
                
                face_indicators_FR.append(True)
                face_bboxs_FR.append(bbox)
                face_chips_FR.append(face_chip.unsqueeze(dim=0))
                face_landmarks_FR.append(torch.tensor(face_landmarks).unsqueeze(dim=0).to(device=images.device).to(images.dtype))
                aligned_face_chips_FR.append(aligned_face_chip.unsqueeze(dim=0))
        
        face_indicators_FR = torch.tensor(face_indicators_FR).to(device=images.device)
        face_bboxs_FR = torch.tensor(face_bboxs_FR).to(device=images.device)
        face_chips_FR = torch.cat(face_chips_FR, dim=0)
        face_landmarks_FR = torch.cat(face_landmarks_FR, dim=0)
        aligned_face_chips_FR = torch.cat(aligned_face_chips_FR, dim=0)
        
        return face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR

    def get_largest_face_app(face_from_app, dim_max, dim_min):
        if len(face_from_app) == 1:
            return face_from_app[0]
        elif len(face_from_app) > 1:
            area_max = 0
            idx_max = 0
            for idx in range(len(face_from_app)):
                bbox = face_from_app[idx]["bbox"]
                area = (min(bbox[2],dim_max) - max(bbox[0], dim_min)) * (min(bbox[3],dim_max) - max(bbox[1], dim_min))
                if area > area_max:
                    area_max = area
                    idx_max = idx
            return face_from_app[idx_max]
    
    def get_face_app(images, fill_value=-1):
        """
        images:shape [N,3,H,W], in range [-1,1], pytorch tensor
        returns:
            face_indicators: torch tensor of shape [N], only True or False
                True means face is detected, False otherwise
            face_bboxs: torch tensor of shape [N,4], 
                if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
            face_chips: torch tensor of shape [N,3,224,224]
                if face_indicator is False, the corresponding face_chip will be all fill_value
        """        
        images_np = ((images*0.5 + 0.5)*255).cpu().detach().permute(0,2,3,1).float().numpy().astype(np.uint8)
        
        face_indicators_app = []
        face_bboxs_app = []
        face_chips_app = []
        face_landmarks_app = []
        aligned_face_chips_app = []
        for idx, image_np in enumerate(images_np):
            # face_app.get input should be [BGR]
            faces_from_app = face_app.get(image_np[:,:,[2,1,0]])
            if len(faces_from_app) == 0:
                face_indicators_app.append(False)
                face_bboxs_app.append([fill_value]*4)
                face_chips_app.append(torch.ones([1,3,args.size_face,args.size_face], dtype=images.dtype, device=images.device)*(fill_value))
                face_landmarks_app.append(torch.ones([1,5,2], dtype=images.dtype, device=images.device)*(fill_value))
                aligned_face_chips_app.append(torch.ones([1,3,args.size_aligned_face,args.size_aligned_face], dtype=images.dtype, device=images.device)*(fill_value))
            else:
                face_from_app = get_largest_face_app(faces_from_app, dim_max=image_np.shape[0], dim_min=0)
                bbox = expand_bbox(face_from_app["bbox"], expand_coef=0.5, target_ratio=1)
                face_chip = crop_face(images[idx], bbox, target_size=[args.size_face,args.size_face], fill_value=fill_value)
                
                face_landmarks = np.array(face_from_app["kps"])
                aligned_face_chip = image_pipeline(images[idx], face_landmarks)
                
                face_indicators_app.append(True)
                face_bboxs_app.append(bbox)
                face_chips_app.append(face_chip.unsqueeze(dim=0))
                face_landmarks_app.append(torch.tensor(face_landmarks).unsqueeze(dim=0).to(device=images.device).to(images.dtype))
                aligned_face_chips_app.append(aligned_face_chip.unsqueeze(dim=0))
        
        face_indicators_app = torch.tensor(face_indicators_app).to(device=images.device)
        face_bboxs_app = torch.tensor(face_bboxs_app).to(device=images.device)
        face_chips_app = torch.cat(face_chips_app, dim=0)
        face_landmarks_app = torch.cat(face_landmarks_app, dim=0)
        aligned_face_chips_app = torch.cat(aligned_face_chips_app, dim=0)
        
        return face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app
                
    # NOTE: used for evaluation-time metrics only (face detector + MobileNet).
    def get_face_gender(face_chips, selector=None, fill_value=-1):
        if selector != None:
            face_chips_w_faces = face_chips[selector]
        else:
            face_chips_w_faces = face_chips
                        
        if face_chips_w_faces.shape[0] == 0:
            logits_gender = torch.empty([0,2], dtype=face_chips.dtype, device=face_chips.device)
            probs_gender = torch.empty([0,2], dtype=face_chips.dtype, device=face_chips.device)
            preds_gender = torch.empty([0], dtype=torch.int64, device=face_chips.device)
        else:
            logits_gender = gender_classifier(face_chips_w_faces.float())
            probs_gender = torch.softmax(logits_gender, dim=-1)
        
            temp = probs_gender.max(dim=-1)
            preds_gender = temp.indices

        # Pin to float32 so the gathered probs dtype never depends on whether a face
        # was detected (no-face branch built fp16, has-face branch .float()->fp32);
        # the mismatch silently DEADLOCKED all_gather until the NCCL watchdog timeout.
        logits_gender = logits_gender.float()
        probs_gender = probs_gender.float()
        
        if selector != None:
            preds_gender_new = torch.ones(
                [selector.shape[0]]+list(preds_gender.shape[1:]), 
                dtype=preds_gender.dtype, 
                device=preds_gender.device
                ) * (fill_value)
            preds_gender_new[selector] = preds_gender
            
            probs_gender_new = torch.ones(
                [selector.shape[0]]+list(probs_gender.shape[1:]),
                dtype=probs_gender.dtype, 
                device=probs_gender.device
                ) * (fill_value)
            probs_gender_new[selector] = probs_gender
            
            logits_gender_new = torch.ones(
                [selector.shape[0]]+list(logits_gender.shape[1:]),
                dtype=logits_gender.dtype, 
                device=logits_gender.device
                ) * (fill_value)
            logits_gender_new[selector] = logits_gender
            
            return preds_gender_new, probs_gender_new, logits_gender_new
        else:
            return preds_gender, probs_gender, logits_gender
        
    @torch.no_grad()
    def generate_dynamic_targets(probs, target_male_ratio=0.5, w_uncertainty=False):
        """generate dynamic targets for the distributional alignment loss

        Args:
            probs (torch.tensor): shape [N,2], N points in a probability simplex of 2 dims
            target_male_ratio (float): target distribution, the percentage of class 1 (male)
            w_uncertainty (True/False): whether return uncertainty measures
        
        Returns:
            targets_all (torch.tensor): target classes
            uncertainty_all (torch.tensor): uncertainty of target classes
        """
        idxs_2_rank = (probs!=-1).all(dim=-1)
        probs_2_rank = probs[idxs_2_rank]
        target_male_ratio = float(np.clip(target_male_ratio, 0.0, 1.0))

        rank = torch.argsort(torch.argsort(probs_2_rank[:,1]))
        targets = (rank >= (rank.shape[0] * (1.0 - target_male_ratio))).long()

        targets_all = torch.ones([probs.shape[0]], dtype=torch.long, device=probs.device) * (-1)
        targets_all[idxs_2_rank] = targets
        
        if w_uncertainty:
            uncertainty = torch.ones([probs_2_rank.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
            uncertainty[targets==1] = torch.tensor(
                1 - scipy.stats.binom.cdf(
                    (rank[targets==1]).cpu().numpy(), 
                    probs_2_rank.shape[0], 
                    target_male_ratio
                    )
                ).to(probs.dtype).to(probs.device)
            uncertainty[targets==0] = torch.tensor(
                scipy.stats.binom.cdf(
                    rank[targets==0].cpu().numpy(), 
                    probs_2_rank.shape[0], 
                    target_male_ratio
                    )
                ).to(probs.dtype).to(probs.device)
            
            uncertainty_all = torch.ones([probs.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
            uncertainty_all[idxs_2_rank] = uncertainty
            
            return targets_all, uncertainty_all
        else:
            return targets_all

    @torch.no_grad()
    def evaluate_process(which_text_encoder, which_unet, name, prompts, noises, current_global_step, enable_sds_eval: bool = True):
        logs = []
        log_imgs = []
        num_denoising_steps = 25
        clip_i_sims = []
        dino_i_sims = []
        clip_t_sims = []
        to_pil_eval = T.ToPILImage()
        # Enable attmap computation whenever the user wants to save them
        # (--save_attmaps is now default on; pass --no_save_attmaps to disable).
        # Producing the attmap uses an attention capture during the SDS forward pass, so
        # we gate it on the save flag. The map is now built in EVERY region_mask_mode
        # ("none"/"face"/"attn"); only if the woman/man tokens can't be located does
        # attmap_* stay None and the save is skipped gracefully.
        need_attmap_eval = bool(args.save_attmaps)
        need_sds_classifier_eval = bool(enable_sds_eval)
        need_sds_forward = need_sds_classifier_eval or need_attmap_eval

        def _clip_image_features_eval(imgs: torch.Tensor):
            # imgs: [-1,1] -> [0,1], CPU PIL -> CLIP ViT-bigG-14 image features (normalized)
            imgs_01 = (imgs.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
            pil_images = [to_pil_eval(img) for img in imgs_01]
            clip_eval_dtype = clip_eval_model.visual.conv1.weight.dtype
            pixel_values = torch.stack([clip_eval_preprocess(img.convert("RGB")) for img in pil_images]).to(
                accelerator.device,
                dtype=clip_eval_dtype,
            )
            with torch.no_grad():
                feats = clip_eval_model.encode_image(pixel_values)
                feats = F.normalize(feats.float(), dim=-1)
            return feats

        def _clip_text_features_eval(text: str):
            with torch.no_grad():
                text_tokens = clip_eval_tokenizer([text]).to(accelerator.device)
                feats = clip_eval_model.encode_text(text_tokens)
                feats = F.normalize(feats.float(), dim=-1)
            return feats

        for prompt_i, noises_i in itertools.zip_longest(prompts, noises):
            if accelerator.is_main_process:
                logs_i = {
                    "gender_gap": [],
                    "gender_gap_abs": [],
                    "bias_score": [],
                    "bias_score_abs": [],
                    "gender_pred_between_0.2_0.8": [],
                    "gender_gap_mnet": [],
                    "gender_gap_abs_mnet": [],
                    "gender_pred_between_0.2_0.8_mnet": [],
                    "gender_gap_abs_mnet_ori": [],
                }
                if enable_sds_eval:
                    logs_i["gender_gap_abs_sds"] = []
                log_imgs_i = {}
            ################################################
            # step 1: generate all ori images
            images_ori = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                if args.train_text_encoder and args.train_unet:
                    images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=eval_unet)
                elif args.train_text_encoder and not args.train_unet:
                    images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=unet)
                elif not args.train_text_encoder and args.train_unet:
                    images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=eval_unet)
                images_ori.append(images_ij)
            images_ori = torch.cat(images_ori)

            # --- Face detector + MobileNet classifier (evaluation only) ---
            face_indicators_ori, face_bboxs_ori, face_chips_ori, face_landmarks_ori, aligned_face_chips_ori = get_face(images_ori)
            preds_gender_ori_mnet, probs_gender_ori_mnet, logits_gender_ori_mnet = get_face_gender(
                face_chips_ori,
                selector=face_indicators_ori,
                fill_value=-1,
            )

            # --- SDS-based classifier (optional; can be disabled for fast eval) ---
            attmap_ori_all = None
            attmap_ori = None
            if need_sds_forward:
                preds_gender_ori_sds, probs_gender_ori_sds, logits_gender_ori_sds, _, _, _, attmap_ori = sds_logits_from_images(
                    images_ori,
                    tau=args.sds_tau,
                    t_min=args.sds_t_min,
                    t_max=args.sds_t_max,
                    num_t=args.sds_num_t,
                    num_eps=args.sds_num_eps,
                    gate_grad=False,
                    precomputed_attmaps=None,
                    sds_text_encoder=eval_text_encoder,
                    sds_unet=unet,
                    sds_scheduler=ddpm_forward,
                    compute_realistic_sds=False,
                    return_attmap=True,
                )
            preds_gender_ori, probs_gender_ori, logits_gender_ori = clip_gender_classifier(
                images_ori,
                clip_model=clip_model,
                clip_processor=clip_processor,
                text_features=gender_text_features,
                device=accelerator.device,
            )
            images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
            face_indicators_ori_all = customized_all_gather(face_indicators_ori, accelerator, return_tensor_other_processes=False)
            face_bboxs_ori_all = customized_all_gather(face_bboxs_ori, accelerator, return_tensor_other_processes=False)
            preds_gender_ori_all = customized_all_gather(preds_gender_ori, accelerator, return_tensor_other_processes=False)
            probs_gender_ori_all = customized_all_gather(probs_gender_ori, accelerator, return_tensor_other_processes=False)
            if need_sds_classifier_eval:
                preds_gender_ori_all_sds = customized_all_gather(preds_gender_ori_sds, accelerator, return_tensor_other_processes=False)
                probs_gender_ori_all_sds = customized_all_gather(probs_gender_ori_sds, accelerator, return_tensor_other_processes=False)
            preds_gender_ori_all_mnet = customized_all_gather(preds_gender_ori_mnet, accelerator, return_tensor_other_processes=False)
            probs_gender_ori_all_mnet = customized_all_gather(probs_gender_ori_mnet, accelerator, return_tensor_other_processes=False)
            if need_attmap_eval and attmap_ori is not None:
                attmap_ori_all = customized_all_gather(attmap_ori, accelerator, return_tensor_other_processes=False)

            if accelerator.is_main_process:
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_ori.jpg")
                plot_in_grid(
                    images_ori_all, 
                    save_to, 
                    face_indicators=face_indicators_ori_all, face_bboxs=face_bboxs_ori_all, 
                    preds_gender=preds_gender_ori_all, 
                    pred_class_probs_gender=probs_gender_ori_all.max(dim=-1).values,
                )
                if enable_sds_eval:
                    save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_ori_sds.jpg")
                    plot_in_grid(
                        images_ori_all, 
                        save_to, 
                        face_indicators=face_indicators_ori_all, face_bboxs=face_bboxs_ori_all, 
                        preds_gender=preds_gender_ori_all_sds, 
                        pred_class_probs_gender=probs_gender_ori_all_sds.max(dim=-1).values,
                    )
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_ori_mnet.jpg")
                plot_in_grid(
                    images_ori_all,
                    save_to,
                    face_indicators=face_indicators_ori_all,
                    face_bboxs=face_bboxs_ori_all,
                    preds_gender=preds_gender_ori_all_mnet,
                    pred_class_probs_gender=probs_gender_ori_all_mnet.max(dim=-1).values,
                )
                log_imgs_i["img_ori"] = [save_to]
                if args.save_attmaps and attmap_ori_all is not None:
                    att_save_dir = os.path.join(args.imgs_save_dir, "eval_attmaps")
                    att_prefix = f"eval_{name}_{global_step}_{sanitize_filename(prompt_i)}_ori_att"
                    save_attmaps(attmap_ori_all, att_save_dir, att_prefix)
                    save_attmaps_with_overlay(attmap_ori_all, images_ori_all, att_save_dir, att_prefix, alpha=0.45)
                    att_preview = os.path.join(att_save_dir, f"{att_prefix}_0_overlay.jpg")
                    if os.path.exists(att_preview):
                        log_imgs_i["attmap_ori_overlay"] = [att_preview]

            
            images = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet)
                images.append(images_ij)
            images = torch.cat(images)
            
            # --- Face detector + MobileNet classifier (evaluation only) ---
            face_indicators, face_bboxs, face_chips, face_landmarks, aligned_face_chips = get_face(images)
            preds_gender_mnet, probs_gender_mnet, logits_gender_mnet = get_face_gender(
                face_chips,
                selector=face_indicators,
                fill_value=-1,
            )

            # --- SDS classifier on generated images (optional; can be disabled for fast eval) ---
            attmap_gen_all = None
            attmap_gen = None
            if need_sds_forward:
                preds_gender_sds, probs_gender_sds, logits_gender_sds, _, _, _, attmap_gen = sds_logits_from_images(
                    images,
                    tau=args.sds_tau,
                    t_min=args.sds_t_min,
                    t_max=args.sds_t_max,
                    num_t=args.sds_num_t,
                    num_eps=args.sds_num_eps,
                    gate_grad=False,
                    precomputed_attmaps=None,
                    sds_text_encoder=eval_text_encoder,
                    sds_unet=unet,
                    sds_scheduler=ddpm_forward,
                    compute_realistic_sds=False,
                    return_attmap=True,
                )
            preds_gender, probs_gender, logits_gender = clip_gender_classifier(
                images,
                clip_model=clip_model,
                clip_processor=clip_processor,
                text_features=gender_text_features,
                device=accelerator.device,
            )
            images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
            face_indicators_all = customized_all_gather(face_indicators, accelerator, return_tensor_other_processes=False)
            face_bboxs_all = customized_all_gather(face_bboxs, accelerator, return_tensor_other_processes=False)
            preds_gender_all = customized_all_gather(preds_gender, accelerator, return_tensor_other_processes=False)
            probs_gender_all = customized_all_gather(probs_gender, accelerator, return_tensor_other_processes=False)
            if need_sds_classifier_eval:
                preds_gender_all_sds = customized_all_gather(preds_gender_sds, accelerator, return_tensor_other_processes=False)
                probs_gender_all_sds = customized_all_gather(probs_gender_sds, accelerator, return_tensor_other_processes=False)
            preds_gender_all_mnet = customized_all_gather(preds_gender_mnet, accelerator, return_tensor_other_processes=False)
            probs_gender_all_mnet = customized_all_gather(probs_gender_mnet, accelerator, return_tensor_other_processes=False)
            if need_attmap_eval and attmap_gen is not None:
                attmap_gen_all = customized_all_gather(attmap_gen, accelerator, return_tensor_other_processes=False)

            if accelerator.is_main_process:
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_generated.jpg")
                plot_in_grid(
                    images_all, 
                    save_to, 
                    face_indicators=face_indicators_all, 
                    face_bboxs=face_bboxs_all, 
                    preds_gender=preds_gender_all, 
                    pred_class_probs_gender=probs_gender_all.max(dim=-1).values,
                    )
                if enable_sds_eval:
                    save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_generated_sds.jpg")
                    plot_in_grid(
                        images_all, 
                        save_to, 
                        face_indicators=face_indicators_all, 
                        face_bboxs=face_bboxs_all, 
                        preds_gender=preds_gender_all_sds, 
                        pred_class_probs_gender=probs_gender_all_sds.max(dim=-1).values,
                        )
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_generated_mnet.jpg")
                plot_in_grid(
                    images_all,
                    save_to,
                    face_indicators=face_indicators_all,
                    face_bboxs=face_bboxs_all,
                    preds_gender=preds_gender_all_mnet,
                    pred_class_probs_gender=probs_gender_all_mnet.max(dim=-1).values,
                )

                log_imgs_i["img_generated"] = [save_to]
                if args.save_attmaps and attmap_gen_all is not None:
                    att_save_dir = os.path.join(args.imgs_save_dir, "eval_attmaps")
                    att_prefix = f"eval_{name}_{global_step}_{sanitize_filename(prompt_i)}_generated_att"
                    save_attmaps(attmap_gen_all, att_save_dir, att_prefix)
                    save_attmaps_with_overlay(attmap_gen_all, images_all, att_save_dir, att_prefix, alpha=0.45)
                    att_preview = os.path.join(att_save_dir, f"{att_prefix}_0_overlay.jpg")
                    if os.path.exists(att_preview):
                        log_imgs_i["attmap_generated_overlay"] = [att_preview]
            
            if accelerator.is_main_process:
                probs_tmp = probs_gender_all[(probs_gender_all!=-1).all(dim=-1)]
                male_ratio = ((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean().item()
                female_ratio = ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean().item()
                gender_gap = male_ratio - female_ratio
                gender_pred_between_02_08 = ((probs_tmp[:,1]>=0.2)*(probs_tmp[:,1]<=0.8)).float().mean().item()
                logs_i["gender_gap"].append(gender_gap)
                logs_i["gender_gap_abs"].append(abs(gender_gap))
                bias_score = args.target_male_ratio - male_ratio
                logs_i["bias_score"].append(bias_score)
                logs_i["bias_score_abs"].append(abs(bias_score))
                logs_i["gender_pred_between_0.2_0.8"].append(abs(gender_pred_between_02_08))
                if enable_sds_eval:
                    probs_tmp = probs_gender_all_sds[(probs_gender_all_sds!=-1).all(dim=-1)]
                    gender_gap_sds = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                    logs_i["gender_gap_abs_sds"].append(abs(gender_gap_sds))
                probs_tmp_mnet = probs_gender_all_mnet[(probs_gender_all_mnet!=-1).all(dim=-1)]
                if probs_tmp_mnet.shape[0] > 0:
                    gender_gap_mnet = (
                        ((probs_tmp_mnet[:,1]>=0.5)*(probs_tmp_mnet[:,1]<=1)).float().mean()
                        - ((probs_tmp_mnet[:,1]>=0)*(probs_tmp_mnet[:,1]<=0.5)).float().mean()
                    ).item()
                    gender_pred_between_02_08_mnet = ((probs_tmp_mnet[:,1]>=0.2)*(probs_tmp_mnet[:,1]<=0.8)).float().mean().item()
                    logs_i["gender_gap_mnet"].append(gender_gap_mnet)
                    logs_i["gender_gap_abs_mnet"].append(abs(gender_gap_mnet))
                    logs_i["gender_pred_between_0.2_0.8_mnet"].append(gender_pred_between_02_08_mnet)
                else:
                    logs_i["gender_gap_mnet"].append(np.nan)
                    logs_i["gender_gap_abs_mnet"].append(np.nan)
                    logs_i["gender_pred_between_0.2_0.8_mnet"].append(np.nan)
                probs_tmp_mnet_ori = probs_gender_ori_all_mnet[(probs_gender_ori_all_mnet!=-1).all(dim=-1)]
                if probs_tmp_mnet_ori.shape[0] > 0:
                    gender_gap_mnet_ori = (
                        ((probs_tmp_mnet_ori[:,1]>=0.5)*(probs_tmp_mnet_ori[:,1]<=1)).float().mean()
                        - ((probs_tmp_mnet_ori[:,1]>=0)*(probs_tmp_mnet_ori[:,1]<=0.5)).float().mean()
                    ).item()
                    logs_i["gender_gap_abs_mnet_ori"].append(abs(gender_gap_mnet_ori))
                else:
                    logs_i["gender_gap_abs_mnet_ori"].append(np.nan)

                # --- similarity metrics (CLIP ViT-bigG-14 + DINOv2 vit-g/14) ---
                with torch.no_grad():
                    clip_feats_ori_eval = _clip_image_features_eval(images_ori_all)
                    clip_feats_gen_eval = _clip_image_features_eval(images_all)
                    clip_i_sims.extend((clip_feats_gen_eval * clip_feats_ori_eval).sum(dim=-1).detach().cpu().tolist())

                    images_small_ori_eval = transforms.Resize(224)(images_ori_all)
                    images_small_gen_eval = transforms.Resize(224)(images_all)
                    dino_ori_eval = get_dino_eval_feat(images_small_ori_eval, normalize=True, to_high_precision=True)
                    dino_gen_eval = get_dino_eval_feat(images_small_gen_eval, normalize=True, to_high_precision=True)
                    dino_i_sims.extend((dino_gen_eval * dino_ori_eval).sum(dim=-1).detach().cpu().tolist())

                    clip_text_feat = _clip_text_features_eval(prompt_i)
                    clip_t_sims.extend((clip_feats_gen_eval * clip_text_feat).sum(dim=-1).detach().cpu().tolist())

            
            if accelerator.is_main_process:
                log_imgs.append(log_imgs_i)
                logs.append(logs_i)

            # Keep all ranks aligned prompt-by-prompt during evaluation.
            # Rank 0 performs extra CPU/GPU work (image saving, metric logging),
            # so without this sync other ranks can run ahead and hit the next
            # collective early, which may trigger NCCL watchdog timeouts.
            accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            # wandb (eval): only the headline aggregate metric + the two image grids.
            if logs and ("gender_gap_abs_mnet" in logs[0]):
                _gg_mnet = float(np.nanmean(np.array([log["gender_gap_abs_mnet"] for log in logs], dtype=float)))
                wandb_tracker.log({f"eval_{name}_gender_gap_abs_mnet": _gg_mnet}, step=current_global_step)
            # [sds-gap] SDS-classifier gender gap on the SAME eval images (proxy vs mnet)
            if logs and ("gender_gap_abs_sds" in logs[0]):
                _gg_sds = float(np.nanmean(np.array([log["gender_gap_abs_sds"] for log in logs], dtype=float)))
                wandb_tracker.log({f"eval_{name}_gender_gap_abs_sds": _gg_sds}, step=current_global_step)

            if args.log_wandb_images:
                imgs_dict = {}
                for prompt_i, log_imgs_i in itertools.zip_longest(prompts, log_imgs):
                    for key in ("img_ori", "img_generated"):
                        if key in log_imgs_i:
                            imgs_dict.setdefault(key, []).append(
                                wandb.Image(data_or_path=log_imgs_i[key][0], caption=prompt_i)
                            )
                for key, imgs in imgs_dict.items():
                    wandb_tracker.log({f"eval_{name}_{key}": imgs}, step=current_global_step)

            if len(clip_i_sims) > 0:
                wandb_tracker.log({f"eval_{name}_Clip-I": float(np.mean(clip_i_sims))}, step=current_global_step)
            if len(dino_i_sims) > 0:
                wandb_tracker.log({f"eval_{name}_DINO_I": float(np.mean(dino_i_sims))}, step=current_global_step)
            if len(clip_t_sims) > 0:
                wandb_tracker.log({f"eval_{name}_Clip-T": float(np.mean(clip_t_sims))}, step=current_global_step)

            # Persist a small JSON of the headline metrics for offline comparison
            # across resumed checkpoints (eval_results.json next to /ckpts).
            if name == "EMA":
                metrics_to_save = {}
                if logs and "gender_gap_abs_mnet" in logs[0]:
                    gg = np.array([log["gender_gap_abs_mnet"] for log in logs], dtype=float)
                    metrics_to_save["eval_EMA_gender_gap_abs_mnet"] = float(np.nanmean(gg))
                # [sds-gap] persist SDS gap next to mnet gap for the proxy-mismatch test
                if logs and "gender_gap_abs_sds" in logs[0]:
                    metrics_to_save["eval_EMA_gender_gap_abs_sds"] = float(
                        np.nanmean(np.array([log["gender_gap_abs_sds"] for log in logs], dtype=float)))
                if len(clip_t_sims) > 0:
                    metrics_to_save["eval_EMA_Clip-T"] = float(np.mean(clip_t_sims))
                if len(clip_i_sims) > 0:
                    metrics_to_save["eval_EMA_Clip-I"] = float(np.mean(clip_i_sims))
                if len(dino_i_sims) > 0:
                    metrics_to_save["eval_EMA_DINO-I"] = float(np.mean(dino_i_sims))

                if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
                    # .../<run>/ckpts/checkpoint-XXX -> .../<run>/eval_results.json
                    results_path = os.path.join(
                        os.path.dirname(os.path.dirname(args.resume_from_checkpoint)),
                        "eval_results.json",
                    )
                else:
                    results_path = os.path.join(
                        os.path.dirname(args.imgs_save_dir), "eval_results.json"
                    )

                existing = {}
                if os.path.exists(results_path):
                    try:
                        with open(results_path, "r") as f:
                            existing = json.load(f)
                    except Exception:
                        existing = {}
                existing[str(int(current_global_step))] = metrics_to_save
                with open(results_path, "w") as f:
                    json.dump(existing, f, indent=2, sort_keys=True)
                logger.info(
                    f"[eval-results] step={current_global_step} -> {metrics_to_save} "
                    f"(saved to {results_path})"
                )

        # Ensure every rank fully exits evaluation together.
        accelerator.wait_for_everyone()
        return logs, log_imgs
    
    def apply_grad_hook_face(images, face_bboxs, face_bboxs_ori, targets, preds_gender_ori, factor=0.1):
        """Scale gradients on the same face region used by 1-main-debias-ftdiff.py.

        The hook is attached to the intersection of the generated-image face bbox
        and the original-image face bbox. If that region cannot be formed, the
        image is left unchanged.
        """
        images_new = []
        for image, face_bbox, face_bbox_ori, target, pred_gender_ori in itertools.zip_longest(
            images, face_bboxs, face_bboxs_ori, targets, preds_gender_ori
        ):
            if (face_bbox == -1).all():
                images_new.append(image.unsqueeze(dim=0))
                continue
            if (face_bbox_ori == -1).all():
                images_new.append(image.unsqueeze(dim=0))
                continue

            img_height, img_width = image.shape[-2:]
            idx_left = max(int(round(face_bbox[0].item())), int(round(face_bbox_ori[0].item())), 0)
            idx_right = min(int(round(face_bbox[2].item())), int(round(face_bbox_ori[2].item())), img_width)
            idx_bottom = max(int(round(face_bbox[1].item())), int(round(face_bbox_ori[1].item())), 0)
            idx_top = min(int(round(face_bbox[3].item())), int(round(face_bbox_ori[3].item())), img_height)

            if idx_right <= idx_left or idx_top <= idx_bottom:
                images_new.append(image.unsqueeze(dim=0))
                continue

            img_face = image[:, idx_bottom:idx_top, idx_left:idx_right].clone()
            if target == -1:
                grad_hook = make_grad_hook(factor)
            elif target == pred_gender_ori:
                grad_hook = make_grad_hook(1)
            else:
                grad_hook = make_grad_hook(factor)
            img_face.register_hook(grad_hook)

            img_add = torch.zeros_like(image)
            img_add[:, idx_bottom:idx_top, idx_left:idx_right] = img_face

            mask = torch.zeros_like(image)
            mask[:, idx_bottom:idx_top, idx_left:idx_right] = 1

            image = mask * img_add + (1 - mask) * image
            images_new.append(image.unsqueeze(dim=0))

        return torch.cat(images_new)
    
    def gen_dynamic_weights_sds(targets, preds_ori_sds, factor=0.2, out_dtype=None):
        """
        realistic.py의 dynamic-weight 로직을 original-image SDS 분류 기준으로 단순화:
        - target == -1 : factor
        - target != preds_ori_sds : factor
        - target == preds_ori_sds : 1
        """
        if out_dtype is None:
            out_dtype = torch.float32
        weights = torch.ones_like(targets, dtype=out_dtype, device=targets.device)
        valid = (targets != -1)
        weights[~valid] = float(factor)
        weights[valid & (preds_ori_sds != targets)] = float(factor)
        return weights

    def model_sanity_print(model, state):
        params = [p for p in model.parameters()]
        print(f"\t{accelerator.device}; {state};\n\t\tparam[0]: {params[0].flatten()[0].item():.8f};\tparam[0].grad: {params[0].grad.flatten()[0].item():.8f}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    optimizer, lr_scheduler = accelerator.prepare(
            optimizer, lr_scheduler
        )
    
    if args.train_text_encoder:
        text_encoder_lora_model, text_encoder_lora_ema = accelerator.prepare(text_encoder_lora_model, text_encoder_lora_ema)
        accelerator.register_for_checkpointing(text_encoder_lora_ema)
    if args.train_unet:
        unet_lora_layers, unet_lora_ema = accelerator.prepare(unet_lora_layers, unet_lora_ema)
        accelerator.register_for_checkpointing(unet_lora_ema)
        
    def should_trigger_evaluation(current_step: int) -> bool:
        # Evaluate every N steps from the beginning of training.
        if args.evaluate_every_n_iter <= 0:
            return False
        return current_step % args.evaluate_every_n_iter == 0

    def should_enable_sds_eval(current_step: int) -> bool:
        # Enable SDS-based evaluation: extra gender-gap metric (gender_gap_abs_sds)
        # and SDS-labeled image grids (_ori_sds.jpg / _generated_sds.jpg).
        # When --save_attmaps is on the SDS forward is already run for the attmap,
        # so this shares that same forward (no extra UNet passes).
        return True

    def evaluation_step(current_step):
        enable_sds_eval = should_enable_sds_eval(current_step)
        eval_imgs_per_prompt = max(args.val_images_per_prompt_GPU, math.ceil(60 / max(1, accelerator.num_processes)))
        noises_val = torch.randn(
        [len(prompts_val), eval_imgs_per_prompt,4,64,64],
        dtype=weight_dtype_high_precision
        ).to(accelerator.device)

        # evaluate EMA only
        if args.train_text_encoder:
            text_encoder_lora_dict_copy = copy.deepcopy(text_encoder_lora_dict)
            load_state_dict_results = text_encoder.load_state_dict(text_encoder_lora_ema_dict, strict=False)
        
        if args.train_unet:
            with torch.no_grad():
                unet_lora_layers_copy = copy.deepcopy(unet_lora_layers)
                for p, p_from in itertools.zip_longest(list(unet_lora_layers.parameters()), unet_lora_ema.shadow_params):
                    p.data = p_from.data
            
        evaluate_process(text_encoder, unet, "EMA", prompts_val, noises_val, current_step, enable_sds_eval=enable_sds_eval)
        
        if args.train_text_encoder:
            load_state_dict_results = text_encoder.load_state_dict(text_encoder_lora_dict_copy, strict=False)
        
        if args.train_unet:
            with torch.no_grad():
                for p, p_from in itertools.zip_longest(list(unet_lora_layers.parameters()), list(unet_lora_layers_copy.parameters())):
                    p.data = p_from.data

        # Keep train/eval phase boundary synchronized across ranks.
        accelerator.wait_for_everyone()
    
    
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num prompts = {train_dataset.__len__()}")
    logger.info(f"  Num images per prompt = {args.train_images_per_prompt_GPU} (per GPU) * {accelerator.num_processes} (GPU)")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:

        if not os.path.exists(args.resume_from_checkpoint):
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])

            resume_global_step = global_step
            first_epoch = global_step // args.num_update_steps_per_epoch
            resume_step = resume_global_step % (args.num_update_steps_per_epoch)
            
            if args.train_text_encoder:
                text_encoder_lora_ema.to(accelerator.device)
                
                # need to recreate text_encoder_lora_ema_dict
                text_encoder_lora_ema_dict = {}
                for name, shadow_param in itertools.zip_longest(text_encoder_lora_params_name_order, text_encoder_lora_ema.shadow_params):
                    text_encoder_lora_ema_dict[name] = shadow_param
                assert text_encoder_lora_ema_dict.__len__() == text_encoder_lora_dict.__len__(), "length does not match! something wrong happened while converting lora params to a state dict."
            
            if args.train_unet:
                unet_lora_ema.to(accelerator.device)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

    # NOTE: --eval_at_step0 evaluates at the *initial* step of this run.
    # When resuming from a checkpoint, that step equals the checkpoint's step
    # (e.g. 200 / 400), not 0 — so we no longer require global_step == 0.
    if args.eval_at_step0:
        evaluation_step(global_step)
        if args.eval_only:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.info(
                    f"[eval-only] step={global_step} evaluation finished, exiting without training."
                )
            accelerator.end_training()
            return

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, data_idx in enumerate(train_dataloader_idxs[epoch]):            
            
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                continue

            # get prompt, should be identical across processes
            prompt_i = train_dataset.__getitem__(data_idx)
            
            # generate noises, should differ by processes
            noises_i = torch.randn(
                [args.train_images_per_prompt_GPU,4,64,64],
                dtype=weight_dtype_high_precision
                ).to(accelerator.device)

            accelerator.wait_for_everyone()
            optimizer.zero_grad()

            # print noise to check if they are different by device
            noises_i_all = [noises_i.detach().clone() for i in range(accelerator.num_processes)]
            torch.distributed.all_gather(noises_i_all, noises_i)
            if accelerator.is_main_process:
                now = datetime.now(my_timezone)
                acc_str = " ".join(
                    f"\tprocess idx: {idx}; noise: {noises_i_all[idx].flatten()[-1].item():.4f};"
                    for idx in range(len(noises_i_all))
                )
                accelerator.print(
                    f"{now.strftime('%Y/%m/%d - %H:%M:%S')} --- epoch: {epoch}, step: {step}, prompt: {prompt_i}\n"
                    f"{acc_str}"
                )

            
            if accelerator.is_main_process:
                logs_i = {
                    "loss_fair": [],
                    "loss_face_realistic": [],
                    "loss_img_x0_clip_dino": [],
                    "loss": [],
                    "gender_gap": [],
                    "gender_gap_abs": [],
                    "gender_gap_abs_sds" : [],
                    "gender_pred_between_0.2_0.8": [],
                    "bias_score": [],
                    "bias_score_abs": [],
                }
                log_imgs_i = {}

            num_denoising_steps = random.choices(range(19,24), k=1)
            torch.distributed.broadcast_object_list(num_denoising_steps, src=0)
            num_denoising_steps = num_denoising_steps[0]

            with torch.no_grad():
                ################################################
                # step 1: generate all images using the diffusion model being finetuned
                images = []
                N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
                for j in range(N):
                    noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                    # Dynamic target assignment should always use fully denoised images.
                    images_ij, _att_ij = generate_image_no_gradient(
                        prompt_i,
                        noises_ij,
                        num_denoising_steps,
                        which_text_encoder=text_encoder,
                        which_unet=unet,
                        skip_final_steps=0,
                        skip_final_steps_pct=0.0,
                    )
                    images.append(images_ij)
                images = torch.cat(images)
                
                preds_gender_sds, probs_gender_sds, logits_gender_sds, _, _, _ = sds_logits_from_images(
                    images,
                    tau=args.sds_tau,
                    t_min=args.sds_t_min,
                    t_max=args.sds_t_max,
                    num_t=args.sds_num_t,
                    num_eps=args.sds_num_eps,
                    gate_grad=False,
                    precomputed_attmaps=None,
                    sds_text_encoder=eval_text_encoder,
                    sds_unet=unet,
                    sds_scheduler=ddpm_forward,
                    compute_realistic_sds=False,
                )
                preds_gender, probs_gender, logits_gender = clip_gender_classifier(
                    images,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    text_features=gender_text_features,
                    device=accelerator.device,
                )

                images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
                preds_gender_all = customized_all_gather(preds_gender, accelerator, return_tensor_other_processes=False)
                probs_gender_all = customized_all_gather(probs_gender, accelerator, return_tensor_other_processes=False)
                preds_gender_all_sds = customized_all_gather(preds_gender_sds, accelerator, return_tensor_other_processes=False)
                probs_gender_all_sds = customized_all_gather(probs_gender_sds, accelerator, return_tensor_other_processes=False)
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_generated.jpg")
                        plot_in_grid(images_all, save_to, preds_gender=preds_gender_all, pred_class_probs_gender=probs_gender_all.max(dim=-1).values)
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_generated_sds.jpg")
                        plot_in_grid(images_all, save_to, preds_gender=preds_gender_all_sds, pred_class_probs_gender=probs_gender_all_sds.max(dim=-1).values)
                        log_imgs_i["img_generated"] = [save_to]

                if accelerator.is_main_process:
                    probs_tmp = probs_gender_all[(probs_gender_all!=-1).all(dim=-1)]
                    male_ratio = ((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean().item()
                    female_ratio = ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean().item()
                    gender_gap = male_ratio - female_ratio
                    gender_pred_between_02_08 = ((probs_tmp[:,1]>=0.2)*(probs_tmp[:,1]<=0.8)).float().mean().item()
                    logs_i["gender_gap"].append(gender_gap)
                    logs_i["gender_gap_abs"].append(abs(gender_gap))
                    bias_score = args.target_male_ratio - male_ratio
                    logs_i["bias_score"].append(bias_score)
                    logs_i["bias_score_abs"].append(abs(bias_score))
                    logs_i["gender_pred_between_0.2_0.8"].append(gender_pred_between_02_08)
                    probs_tmp = probs_gender_all_sds[(probs_gender_all_sds!=-1).all(dim=-1)]
                    gender_gap_sds = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                    logs_i["gender_gap_abs_sds"].append(abs(gender_gap_sds))

                ################################################
                # Step 2: generate dynamic targets (based on SDS probs)
                targets_all, uncertainty_all = generate_dynamic_targets(
                    probs_gender_all_sds,
                    target_male_ratio=args.target_male_ratio,
                    w_uncertainty=True,
                )
                torch.distributed.broadcast(targets_all, src=0)
                torch.distributed.broadcast(uncertainty_all, src=0)

                targets_all[uncertainty_all>args.uncertainty_threshold] = -1
                targets = targets_all[probs_gender.shape[0]*(accelerator.local_process_index):probs_gender.shape[0]*(accelerator.local_process_index+1)]
                uncertainty = uncertainty_all[probs_gender.shape[0]*(accelerator.local_process_index):probs_gender.shape[0]*(accelerator.local_process_index+1)]
                accelerator.print(f"\tNum samples to compute grads: {(targets_all!=-1).sum().item()}/{targets_all.shape[0]}")

                ################################################
                # Step 3: generate all original images using the original diffusion model (kept)
                images_ori = []
                N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
                for j in range(N):
                    noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                    if args.train_text_encoder and args.train_unet:
                        images_ij, _att_ij = generate_image_no_gradient(
                            prompt_i,
                            noises_ij,
                            num_denoising_steps,
                            which_text_encoder=eval_text_encoder,
                            which_unet=eval_unet,
                            skip_final_steps=args.skip_final_steps,
                            skip_final_steps_pct=args.skip_final_steps_pct,
                        )
                    elif args.train_text_encoder and not args.train_unet:
                        images_ij, _att_ij = generate_image_no_gradient(
                            prompt_i,
                            noises_ij,
                            num_denoising_steps,
                            which_text_encoder=eval_text_encoder,
                            which_unet=unet,
                            skip_final_steps=args.skip_final_steps,
                            skip_final_steps_pct=args.skip_final_steps_pct,
                        )
                    elif not args.train_text_encoder and args.train_unet:
                        images_ij, _att_ij = generate_image_no_gradient(
                            prompt_i,
                            noises_ij,
                            num_denoising_steps,
                            which_text_encoder=text_encoder,
                            which_unet=eval_unet,
                            skip_final_steps=args.skip_final_steps,
                            skip_final_steps_pct=args.skip_final_steps_pct,
                        )
                    images_ori.append(images_ij)
                images_ori = torch.cat(images_ori)

                preds_gender_ori_sds, probs_gender_ori_sds, logits_gender_ori_sds, _, _, _ = sds_logits_from_images(
                    images_ori,
                    tau=args.sds_tau,
                    t_min=args.sds_t_min,
                    t_max=args.sds_t_max,
                    num_t=args.sds_num_t,
                    num_eps=args.sds_num_eps,
                    gate_grad=False,
                    precomputed_attmaps=None,
                    sds_text_encoder=eval_text_encoder,
                    sds_unet=unet,
                    sds_scheduler=ddpm_forward,
                    compute_realistic_sds=False,
                )
                face_indicators_ori, face_bboxs_ori, _, _, _ = get_face(images_ori)

                preds_gender_ori, probs_gender_ori, logits_gender_ori = clip_gender_classifier(
                    images_ori,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    text_features=gender_text_features,
                    device=accelerator.device,
                )

                images_small_ori = transforms.Resize(args.img_size_small)(images_ori)
                clip_feats_ori = get_clip_feat(images_small_ori, normalize=True, to_high_precision=True).detach()
                DINO_feats_ori = get_dino_feat(images_small_ori, normalize=True, to_high_precision=True).detach()

                images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
                preds_gender_ori_all = customized_all_gather(preds_gender_ori, accelerator, return_tensor_other_processes=False)
                probs_gender_ori_all = customized_all_gather(probs_gender_ori, accelerator, return_tensor_other_processes=False)
                
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_ori.jpg")
                        plot_in_grid(
                            images_ori_all, 
                            save_to, 
                            preds_gender=preds_gender_ori_all, 
                            pred_class_probs_gender=probs_gender_ori_all.max(dim=-1).values,
                        )
                        log_imgs_i["img_ori"] = [save_to]
            
            ################################################
            # Step 4: compute loss
            loss_fair_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_face_realistic_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_img_x0_clip_dino_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)

            idxs_i = list(range(targets.shape[0]))
            N_backward = math.ceil(targets.shape[0] / args.train_GPU_batch_size)
            for j in range(N_backward):
                idxs_ij = idxs_i[j*args.train_GPU_batch_size:(j+1)*args.train_GPU_batch_size]
                noises_ij = noises_i[idxs_ij]
                targets_ij = targets[idxs_ij]
                clip_feats_ori_ij = clip_feats_ori[idxs_ij]
                DINO_feats_ori_ij = DINO_feats_ori[idxs_ij]
                preds_gender_ori_sds_ij = preds_gender_ori_sds[idxs_ij]
                face_bboxs_ori_ij = face_bboxs_ori[idxs_ij]

                images_ij, _att_ij = generate_image_w_gradient(
                    prompt_i,
                    noises_ij,
                    num_denoising_steps,
                    which_text_encoder=text_encoder,
                    which_unet=unet,
                    skip_final_steps=args.skip_final_steps,
                    skip_final_steps_pct=args.skip_final_steps_pct,
                )
                with torch.no_grad():
                    _, face_bboxs_ij, _, _, _ = get_face(images_ij)
                images_ij_for_img_loss = apply_grad_hook_face(
                    images_ij,
                    face_bboxs_ij,
                    face_bboxs_ori_ij,
                    targets_ij,
                    preds_gender_ori_sds_ij,
                    factor=args.factor2,
                )
                images_small_ij = transforms.Resize(args.img_size_small)(images_ij_for_img_loss)
                clip_feats_ij = get_clip_feat(images_small_ij, normalize=True, to_high_precision=True)
                DINO_feats_ij = get_dino_feat(images_small_ij, normalize=True, to_high_precision=True)
                loss_CLIP_ij = 1.0 - (clip_feats_ij * clip_feats_ori_ij).sum(dim=-1)
                loss_DINO_ij = 1.0 - (DINO_feats_ij * DINO_feats_ori_ij).sum(dim=-1)
                loss_img_x0_clip_dino_ij = (loss_CLIP_ij + loss_DINO_ij).to(weight_dtype)

                # --- SDS classifier WITH gradient + region hard mask ---
                preds_gender_ij, probs_gender_ij, logits_gender_ij, _, _, sds_realistic_ij = sds_logits_from_images(
                    images_ij,
                    tau=args.sds_tau,
                    t_min=args.sds_t_min,
                    t_max=args.sds_t_max,
                    num_t=args.sds_num_t,
                    num_eps=args.sds_num_eps,
                    gate_grad=True,
                    precomputed_attmaps=None,
                    sds_text_encoder=eval_text_encoder,
                    sds_unet=unet,
                    sds_scheduler=ddpm_forward,
                    compute_realistic_sds=True,
                )

                loss_fair_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) *(-1)
                idxs_valid = (targets_ij != -1).nonzero().view([-1])
                logits_gender_ij = logits_gender_ij.half()
                if idxs_valid.numel() > 0:
                    loss_fair_ij[idxs_valid] = CE_loss(logits_gender_ij[idxs_valid], targets_ij[idxs_valid])

                loss_face_realistic_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) *(-1)

                if sds_realistic_ij is None:
                    loss_face_realistic_ij = torch.zeros(len(idxs_ij), dtype=weight_dtype, device=accelerator.device)
                else:
                    loss_face_realistic_ij = sds_realistic_ij.to(weight_dtype)

                dynamic_weights = gen_dynamic_weights_sds(
                    targets_ij,
                    preds_gender_ori_sds_ij,
                    factor=args.factor1,
                    out_dtype=loss_img_x0_clip_dino_ij.dtype,
                )

                loss_ij = (
                    loss_fair_ij
                    + args.weight_loss_img * dynamic_weights * loss_img_x0_clip_dino_ij
                    + args.weight_loss_face_realistic * loss_face_realistic_ij
                )
                accelerator.backward(loss_ij.mean())

                with torch.no_grad():
                    loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)
                    loss_face_realistic_i[idxs_ij] = loss_face_realistic_ij.to(loss_face_realistic_i.dtype)
                    loss_img_x0_clip_dino_i[idxs_ij] = loss_img_x0_clip_dino_ij.to(loss_img_x0_clip_dino_i.dtype)
                    loss_i[idxs_ij] = loss_ij.to(loss_i.dtype)
                    
            # for logging purpose, gather all losses to main_process
            accelerator.wait_for_everyone()
            loss_fair_all = customized_all_gather(loss_fair_i, accelerator)
            loss_face_realistic_all = customized_all_gather(loss_face_realistic_i, accelerator)
            loss_img_x0_clip_dino_all = customized_all_gather(loss_img_x0_clip_dino_i, accelerator)
            loss_all = customized_all_gather(loss_i, accelerator)

            loss_all = loss_all[loss_fair_all!=-1]
            loss_fair_all = loss_fair_all[loss_fair_all!=-1]
            loss_face_realistic_all = loss_face_realistic_all[loss_face_realistic_all!=-1]

            if accelerator.is_main_process:
                logs_i["loss_fair"].append(loss_fair_all)
                logs_i["loss_face_realistic"].append(loss_face_realistic_all)
                logs_i["loss_img_x0_clip_dino"].append(loss_img_x0_clip_dino_all)
                logs_i["loss"].append(loss_all)
            
            # process logs
            if accelerator.is_main_process:
                for key in ["loss_fair", "loss_face_realistic", "loss_img_x0_clip_dino", "loss"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)
                    else:
                        logs_i[key] = torch.cat(logs_i[key])
                for key in ["gender_gap", "gender_gap_abs", "gender_gap_abs_sds", "gender_pred_between_0.2_0.8", "bias_score", "bias_score_abs"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)

            ##########################################################################
            # log process for training
            if accelerator.is_main_process:
                for key, values in logs_i.items():
                    if isinstance(values, list):
                        wandb_tracker.log({f"train_{key}": np.mean(values)}, step=global_step)
                    else:
                        wandb_tracker.log({f"train_{key}": values.mean().item()}, step=global_step)
                if "bias_score" in logs_i:
                    wandb_tracker.log({"bias_score": float(np.mean(logs_i["bias_score"]))}, step=global_step)
                if "bias_score_abs" in logs_i:
                    wandb_tracker.log({"bias_score_abs": float(np.mean(logs_i["bias_score_abs"]))}, step=global_step)

                if args.log_wandb_images:
                    for key, values in log_imgs_i.items():
                        wandb_tracker.log({f"train_{key}":wandb.Image(
                                data_or_path=values[0],
                                caption=prompt_i,
                            )
                            },
                            step=global_step
                            )

            if args.train_text_encoder:
                model_sanity_print(text_encoder_lora_model, "check No.1, text_encoder: after accelerator.backward()")
            if args.train_unet:
                model_sanity_print(unet_lora_layers, "check No.1, unet: after accelerator.backward()")

            # note that up till now grads are not synced
            # we manually sync grads
            grad_is_finite = True
            with torch.no_grad():
                if args.train_text_encoder:
                    for p in text_encoder_lora_model.parameters():
                        if not torch.isfinite(p.grad).all():
                            grad_is_finite = False
                        torch.distributed.all_reduce(p.grad, torch.distributed.ReduceOp.SUM)
                        p.grad = p.grad / accelerator.num_processes / N_backward
                if args.train_unet:
                    for p in unet_lora_layers.parameters():
                        if not torch.isfinite(p.grad).all():
                            grad_is_finite = False
                        torch.distributed.all_reduce(p.grad, torch.distributed.ReduceOp.SUM)
                        p.grad = p.grad / accelerator.num_processes / N_backward
                
            if args.train_text_encoder:
                model_sanity_print(text_encoder_lora_model, "check No.2, text_encoder: after gradients allreduce & average")
            if args.train_unet:
                model_sanity_print(unet_lora_layers, "check No.2, unet: after gradients allreduce & average")

            if grad_is_finite:
                optimizer.step()
            else:
                accelerator.print(f"grads are not finite, skipped!")
            
            lr_scheduler.step()
            
            if grad_is_finite:
                if args.train_text_encoder:
                    text_encoder_lora_ema.step(  text_encoder_lora_params )
                if args.train_unet:
                    unet_lora_ema.step(  unet_lora_layers.parameters() )

            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                with torch.no_grad():
                    if args.train_text_encoder:
                        param_norm = np.mean([p.norm().item() for p in text_encoder_lora_params])
                        param_ema_norm = np.mean([p.norm().item() for p in text_encoder_lora_ema.shadow_params])
                        wandb_tracker.log({f"train_TE_lora_norm": param_norm}, step=global_step)
                        wandb_tracker.log({f"train_TE_lora_ema_norm": param_ema_norm}, step=global_step)
                    if args.train_unet:
                        param_norm = np.mean([p.norm().item() for p in unet_lora_layers.parameters()])
                        param_ema_norm = np.mean([p.norm().item() for p in unet_lora_ema.shadow_params])
                        wandb_tracker.log({f"train_unet_lora_norm": param_norm}, step=global_step)
                        wandb_tracker.log({f"train_unet_lora_ema_norm": param_ema_norm}, step=global_step)

            if should_trigger_evaluation(global_step):
                evaluation_step(global_step)
                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        name = "checkpoint_tmp"
                        clean_checkpoint(args.ckpts_save_dir, name, args.checkpoints_total_limit)

                    save_path = os.path.join(args.ckpts_save_dir, f"checkpoint_tmp-{global_step}")
                    accelerator.save_state(save_path)
                
                    logger.info(f"Accelerator checkpoint saved to {save_path}")

                if global_step % args.checkpointing_steps_long == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`

                    save_path = os.path.join(args.ckpts_save_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                
                    logger.info(f"Accelerator checkpoint saved to {save_path}")

            torch.cuda.empty_cache()

            # Hard stop at max_train_steps. Without this the loop runs WHOLE epochs;
            # here one epoch = len(occupations_train_set) prompts (e.g. 1000), so it would
            # overshoot (e.g. to 1000) instead of stopping at max_train_steps (e.g. 400).
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
