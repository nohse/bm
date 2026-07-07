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

import os, sys
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
import yaml
from packaging import version
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import scipy
 
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

from typing import List, Optional
import uuid, time


my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.

# =========================
# Poverty (higher/lower) prompt groups
# =========================
POVERTY_ITEMS = [
    {
        "class_label": "higher",
        "text": "a photo of a higher-class house with a well-maintained exterior",
    },
    {
        "class_label": "lower",
        "text": "a photo of a lower-class house with a poorly-maintained exterior",
    },
]
POVERTY_PROMPTS = [item["text"] for item in POVERTY_ITEMS]
POVERTY_HIGHER_IDXS = [i for i, item in enumerate(POVERTY_ITEMS) if item["class_label"] == "higher"]
POVERTY_LOWER_IDXS = [i for i, item in enumerate(POVERTY_ITEMS) if item["class_label"] == "lower"]
POVERTY_ATTMAP_PROMPTS = [item["text"] for item in POVERTY_ITEMS]
POVERTY_ATTMAP_KEYWORDS = ["house"]



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

            # store a single tensor per hook-call: (b_half, h, w)
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
    """Normalize attention map for visualization only.

    This does NOT affect training losses or SDS weighting. It is used only when
    converting maps to PNG for human inspection.
    """
    if not isinstance(att, torch.Tensor):
        att = torch.tensor(att)

    a = att.detach().float().cpu()
    if a.ndim != 2:
        a = a.squeeze()
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    # Robust contrast stretch: use percentiles first to make tiny-valued maps visible.
    lo = torch.quantile(a, 0.01)
    hi = torch.quantile(a, 0.99)
    if hi <= lo + 1e-12:
        lo = a.min()
        hi = a.max()
    if hi <= lo + 1e-12:
        return torch.zeros_like(a)

    vis = (a - lo) / (hi - lo)
    vis = vis.clamp(0.0, 1.0)
    # Slight gamma boost to reveal low-amplitude regions.
    vis = vis.pow(0.7)
    return vis


def attmap_to_pil(att: torch.Tensor) -> Image.Image:
    """Convert a single HxW attention map to a colored PIL image for display."""
    vis = _normalize_attmap_for_vis(att)
    a = vis.mul(255).to(torch.uint8).numpy()
    img = Image.fromarray(a, mode="L")
    colored = ImageOps.colorize(img, black="black", white="red")
    return colored


def save_attmaps(att: Optional[torch.Tensor], save_dir: str, prefix: str):
    """Save attention maps to `save_dir` with filenames `{prefix}_{i}.png`.

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
        fname = f"{prefix}_{i}.png"
        pil.save(os.path.join(save_dir, fname), format="PNG", optimize=True)


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

    # colored attention map PIL
    att_pil = attmap_to_pil(att_map)

    # resize att to image size
    if att_pil.size != img_pil.size:
        att_pil = att_pil.resize(img_pil.size, resample=Image.BILINEAR)

    # blend
    blended = Image.blend(img_pil.convert("RGB"), att_pil.convert("RGB"), alpha=alpha)
    return blended


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
            fname = f"{prefix}_{i}_overlay.png"
            blended.save(os.path.join(save_dir, fname), format="PNG", optimize=True)
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

def plot_in_grid(
    images,
    save_to,
    preds_poverty=None,
    pred_class_probs_poverty=None,
):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """
    # reorder by poverty class probability (higher first, then lower)
    if preds_poverty is not None and pred_class_probs_poverty is not None:
        idxs_higher = (preds_poverty == 0).nonzero(as_tuple=False).view([-1])
        probs_higher = pred_class_probs_poverty[idxs_higher]
        idxs_higher = idxs_higher[probs_higher.argsort(descending=True)]

        idxs_lower = (preds_poverty == 1).nonzero(as_tuple=False).view([-1])
        probs_lower = pred_class_probs_poverty[idxs_lower]
        idxs_lower = idxs_lower[probs_lower.argsort(descending=True)]

        idxs_unknown = (preds_poverty == -1).nonzero(as_tuple=False).view([-1])

        images_to_plot = []
        idxs_reordered = torch.cat([idxs_higher, idxs_lower, idxs_unknown])
    else:
        idxs_reordered = torch.arange(images.shape[0], device=images.device)
        images_to_plot = []

    for idx in idxs_reordered:
        img = images[idx]
        pred_poverty = preds_poverty[idx] if preds_poverty is not None else torch.tensor(-1, device=images.device)
        pred_class_prob_poverty = (
            pred_class_probs_poverty[idx] if pred_class_probs_poverty is not None else torch.tensor(0.0, device=images.device)
        )

        if pred_poverty == 0:
            border_color = "blue"
        elif pred_poverty == 1:
            border_color = "red"
        else:
            border_color = "white"

        img_pil = transforms.ToPILImage()(img*0.5+0.5)
        img_pil = ImageOps.expand(img_pil, border=(50,0,0,0), fill=border_color)

        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_probs_poverty is not None and pred_class_prob_poverty.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_poverty.item())*512)], fill="white", outline=None)

        try:
            fnt = ImageFont.truetype(font="../data/0-utils/arial-bold.ttf", size=100)
        except Exception:
            fnt = ImageFont.load_default()
        img_pil_draw.text((400, 400), f"{idx.item()}", align="left", font=fnt)

        img_pil = ImageOps.expand(img_pil_draw._image, border=(10,10,10,10), fill="black")

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
    tensor_all = [tensor.detach().clone() for i in range(accelerator.num_processes)]
    torch.distributed.all_gather(tensor_all, tensor)
    if return_tensor_other_processes:
        if accelerator.num_processes>1:
            tensor_others = torch.cat([tensor_all[idx] for idx in range(accelerator.num_processes) if idx != accelerator.local_process_index], dim=0)
        else:
            tensor_others = torch.empty([0,]+ list(tensor_all[0].shape[1:]), device=accelerator.device, dtype=tensor_all[0].dtype)
    tensor_all = torch.cat(tensor_all, dim=0)
    
    if return_tensor_other_processes:
        return tensor_all, tensor_others
    else:
        return tensor_all


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
        default=None,
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
        default=8,
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
        default="../data/1-prompts/occupation_house.json",
        help="prompt template, and occupations for train and val",
    )
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
        help="zero-shot poverty classifier로 사용할 CLIP 모델 이름"
    )
    parser.add_argument(
        "--skip_final_steps", type=int, default=0,
        help="Absolute number of final denoising steps to skip when using gradient-guided generation."
    )
    parser.add_argument(
        "--skip_final_steps_pct", type=float, default=0.0,
        help="If >0, compute final step skipping as this percentage of the sampled num_denoising_steps (overrides --skip_final_steps)."
    )
    parser.add_argument(
        "--final_steps_skip_pct",
        dest="skip_final_steps_pct",
        type=float,
        help="Alias of --skip_final_steps_pct.",
    )

    # =====================
    # [ADDED] Region masking switch (attn / none)
    # =====================
    parser.add_argument("--region_mask_mode", type=str, default="none",
                        choices=["none", "attn"],
                        help="SDS 및 backprop에서 사용할 영역 마스킹 방식 선택. 'attn'은 기존 어텐션 맵, 'none'은 전체.")
    parser.add_argument(
        "--save_attmaps",
        dest="save_attmaps",
        action="store_true",
        help="평가 시 shared attmap/overlay 이미지를 저장합니다.",
    )
    parser.add_argument(
        "--no_save_attmaps",
        dest="save_attmaps",
        action="store_false",
        help="평가 시 attmap/overlay 저장을 끕니다.",
    )
    parser.add_argument(
        "--enable_sds_eval",
        dest="enable_sds_eval",
        action="store_true",
        help="평가 단계에서 SDS 분류/attmap 계산을 수행합니다.",
    )
    parser.add_argument(
        "--disable_sds_eval",
        dest="enable_sds_eval",
        action="store_false",
        help="평가 단계에서 SDS 분류/attmap 계산을 생략합니다.",
    )
    parser.add_argument(
        "--eval_at_start",
        dest="eval_at_start",
        action="store_true",
        help="global_step 0에서 초기 evaluation을 수행합니다.",
    )
    parser.add_argument(
        "--no_eval_at_start",
        dest="eval_at_start",
        action="store_false",
        help="global_step 0에서 초기 evaluation을 생략합니다.",
    )
    parser.set_defaults(save_attmaps=True)
    parser.set_defaults(enable_sds_eval=False)
    parser.set_defaults(eval_at_start=True)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        args_dict = vars(args)
        for key, value in config_data.items():
            args_dict[key] = type(args_dict[key])(value)
        args = argparse.Namespace(**args_dict)

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

    set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    now = datetime.now(my_timezone)
    timestring = f"{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}"
    folder_name = f"BS-{args.train_images_per_prompt_GPU*accelerator.num_processes}_wImg-{args.weight_loss_img}-{args.factor1}-{args.factor2}_Th-{args.uncertainty_threshold}_loraR-{args.rank}_lr-{args.learning_rate}_{timestring}"
    
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
    # ---- zero-shot poverty text embedding 미리 계산 (higher/lower, 2 prompts) ----
    poverty_prompts = POVERTY_PROMPTS
    with torch.no_grad():
        text_inputs = clip_processor(
            text=poverty_prompts,
            return_tensors="pt",
            padding=True
        )
        text_inputs = {k: v.to(accelerator.device) for k, v in text_inputs.items()}
        poverty_text_features = clip_model.get_text_features(**text_inputs)
        poverty_text_features = F.normalize(poverty_text_features, dim=-1)

    # dedicated eval CLIP (force clip-vit-large-patch14 for evaluation-time metrics)
    eval_clip_model_id = "openai/clip-vit-large-patch14"
    if args.zeroshot_model == eval_clip_model_id:
        clip_eval_model = clip_model
        clip_eval_processor = clip_processor
    else:
        clip_eval_model = CLIPModel.from_pretrained(eval_clip_model_id).to(accelerator.device)
        clip_eval_model.eval().requires_grad_(False)
        clip_eval_processor = CLIPProcessor.from_pretrained(eval_clip_model_id)
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
            torch.distributed.broadcast(p, src=0)
        
        unet_lora_ema = EMAModel(unet_lora_layers.parameters(), decay=args.EMA_decay)
        unet_lora_ema.to(accelerator.device)
        
        # print to check whether unet lora & ema is identical across devices
        print(f"{accelerator.device}; unet lora init to: {list(unet_lora_layers.parameters())[0].flatten()[1]:.6f}; unet lora ema init to: {unet_lora_ema.shadow_params[0].flatten()[1]:.6f}")

    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.rank, patch_mlp=True)
        
        for p in text_encoder_lora_params:
            torch.distributed.broadcast(p, src=0)
                    
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
            torch.distributed.broadcast(lora_param, src=0)
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

    
    prompts_val = experiment_data.get("test_prompts", None)
    if not isinstance(prompts_val, list) or len(prompts_val) == 0:
        prompts_val = [
            prompt.format(occupation=occupation)
            for prompt in experiment_data["prompt_templates_test"]
            for occupation in experiment_data["occupations_test_set"]
        ]
    # Keep evaluation prompt count fixed/small for stable runtime.
    prompts_val = prompts_val[:10]
    
    
    #######################################################
    # set up things needed for finetuning
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
    

    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2.to(accelerator.device, dtype=weight_dtype)
    dinov2.requires_grad_(False)
    dinov2_img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
    dinov2_img_std = torch.tensor([0.229, 0.224, 0.225]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
    
    CE_loss = nn.CrossEntropyLoss(reduction="none")   
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

        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]

        return images, None
    
    def generate_image_w_gradient(
        prompt,
        noises,
        num_denoising_steps,
        which_text_encoder,
        which_unet,
        skip_final_steps: int = 0,
        skip_final_steps_pct: float = 0.0,
    ):
        which_unet.train()  # <- 전역 unet 말고 which_unet로 맞추는 게 안전

        N = noises.shape[0]
        prompts = [prompt] * N

        prompts_token = tokenizer(prompts, return_tensors="pt", padding=True).to(accelerator.device)
        prompt_embeds = which_text_encoder(prompts_token["input_ids"], prompts_token["attention_mask"])[0]

        uncond_input = tokenizer([""] * N, padding="max_length", max_length=prompt_embeds.shape[1],
                                truncation=True, return_tensors="pt").to(accelerator.device)
        negative_prompt_embeds = which_text_encoder(uncond_input["input_ids"], uncond_input["attention_mask"])[0]
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]).to(weight_dtype)

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

        # 1) 앞부분 steps_to_run 만큼은 기존 step과 동일
        for i, t in enumerate(timesteps[:steps_to_run]):
            latent_model_input = torch.cat([latents.detach().to(weight_dtype)] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            eps = which_unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            eps = eps.to(weight_dtype_high_precision)

            eps_u, eps_c = eps.chunk(2)
            eps = eps_u + args.guidance_scale * (eps_c - eps_u)

            eps.register_hook(make_grad_hook(grad_coefs[i]))
            latents = noise_scheduler.step(eps, t, latents).prev_sample
        # 2) skip이 있으면: "현재 latents가 위치한 timestep"에서 eps를 다시 예측해 x0로 점프
        if skip_final_steps > 0:
            t_cur = timesteps[steps_to_run]  # <- 중요: steps_to_run-1가 아니라 steps_to_run

            latent_model_input = torch.cat([latents.detach().to(weight_dtype)] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t_cur)

            eps = which_unet(latent_model_input, t_cur, encoder_hidden_states=prompt_embeds).sample
            eps = eps.to(weight_dtype_high_precision)

            eps_u, eps_c = eps.chunk(2)
            eps = eps_u + args.guidance_scale * (eps_c - eps_u)

            eps.register_hook(make_grad_hook(grad_coefs[steps_to_run]))

            step_out = noise_scheduler.step(eps, t_cur, latents)

            # scheduler가 제공하면 이게 가장 안전 (clip/threshold 포함)
            if hasattr(step_out, "pred_original_sample") and step_out.pred_original_sample is not None:
                latents = step_out.pred_original_sample
            else:
                # fallback: x0 공식
                alpha_bar = noise_scheduler.alphas_cumprod[t_cur].to(device=latents.device, dtype=latents.dtype)
                latents = (latents - (1 - alpha_bar).sqrt() * eps.to(latents.dtype)) / alpha_bar.sqrt()

        latents = latents / vae.config.scaling_factor
        images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)

        return images, None

    
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
        Compute mid-block cross-attn map focused on selected tokens with NO scaling/normalization.
        Returns [B, H, W] in the latent spatial size. Runs under torch.no_grad() and restores processors afterward.
        """
        store = _AttnStore()
        keywords = POVERTY_ATTMAP_KEYWORDS
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
    # [MODIFIED] SDS-based classifier utilities (poverty higher/lower)
    #   - region_mask_mode에 따라 shared house attn/none 마스크 적용
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
        compute_realistic_sds: bool = False,
        return_attmaps: bool = False,
    ):
        """
        - region_mask_mode == 'attn': SDS 계산 시 shared house attmap을 내부에서 추출하여 가중
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

        # 해상도(H,W): eps_pred_*와 동일(보통 64x64)
        _H = zt.shape[-2]
        _W = zt.shape[-1]
        realistic_prompt = "a photo of a realistic house"

        # --- 텍스트 임베딩 (2 prompts) ---
        prompt_embeds = _text_embeds(POVERTY_PROMPTS, sds_text_encoder).to(weight_dtype)  # [C,seq,hidden]
        pe_realistic = None
        if compute_realistic_sds:
            pe_realistic = _text_embeds([realistic_prompt] * BKE, sds_text_encoder).to(weight_dtype)
        num_prompts = prompt_embeds.shape[0]
        errs_per_prompt = []

        def _resize_attn_map(att_map: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if att_map is None:
                return None
            a = att_map.to(device=accelerator.device, dtype=torch.float32)
            if a.shape[-2:] != (_H, _W):
                a = F.interpolate(a.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False).squeeze(1)
            return a.to(weight_dtype)

        def _run_unet(prompt_embeds_i: torch.Tensor):
            return sds_unet(zt.to(weight_dtype), t_vec, encoder_hidden_states=prompt_embeds_i).sample.to(weight_dtype)

        def _compute_shared_house_attmap() -> Optional[torch.Tensor]:
            if not (args.region_mask_mode == "attn" and args.use_attn_weight):
                return None

            per_prompt_maps = []
            for attmap_prompt in POVERTY_ATTMAP_PROMPTS:
                tok_pos_i = find_token_positions(tokenizer, attmap_prompt, keywords=POVERTY_ATTMAP_KEYWORDS)
                if len(tok_pos_i) == 0:
                    continue

                pe_attmap_i = _text_embeds([attmap_prompt] * BKE, sds_text_encoder).to(weight_dtype)
                capture = CrossAttnCapture(
                    token_indices=tok_pos_i,
                    use_cpu=False,
                    expect_cfg_pair=False,
                ).add_hooks(sds_unet)
                try:
                    _ = _run_unet(pe_attmap_i)
                    att_map_i = capture.aggregated_map()
                finally:
                    capture.clear()

                att_map_i = _resize_attn_map(att_map_i)
                if att_map_i is not None:
                    per_prompt_maps.append(att_map_i)

            if len(per_prompt_maps) == 0:
                return None
            return torch.stack(per_prompt_maps, dim=0).mean(dim=0).to(weight_dtype)

        # --- UNet 예측 및 prompt별 에러 수집 ---
        for c in range(num_prompts):
            pe_c = prompt_embeds[c].unsqueeze(0).expand(BKE, -1, -1).contiguous()
            eps_pred_c = _run_unet(pe_c)
            err_c = ((eps_pred_c - eps)**2).mean(dim=1)  # [BKE,H,W]
            errs_per_prompt.append(err_c)

        # --- 두 클래스에 공통으로 적용할 house attmap 생성 ---
        shared_attmap_bke = None
        shared_attmap_mean = None
        if args.region_mask_mode == "attn" and args.use_attn_weight:
            shared_attmap_bke = _compute_shared_house_attmap()
            if shared_attmap_bke is not None:
                shared_attmap_mean = (
                    shared_attmap_bke.float()
                    .view(B, K, num_eps, _H, _W)
                    .mean(dim=(1, 2))
                    .to(weight_dtype)
                )

        # --- shared house attmap을 각 prompt의 SDS에 가중 곱 ---
        sds_per_prompt = []
        for c in range(num_prompts):
            err_c = errs_per_prompt[c]
            w_map = shared_attmap_bke
            if w_map is not None:
                w = w_map.to(err_c.dtype).clamp_min(0)
                w_sum = w.flatten(1).sum(dim=1).clamp_min(1e-8)
                sds_c_per = (err_c * w).flatten(1).sum(dim=1) / w_sum
            else:
                sds_c_per = err_c.flatten(1).mean(dim=1)
            sds_per_prompt.append(sds_c_per)

        sds_per_prompt = torch.stack(sds_per_prompt, dim=0)  # [C,BKE]
        sds_real_per = None
        if compute_realistic_sds:
            eps_pred_real = sds_unet(zt.to(weight_dtype), t_vec, encoder_hidden_states=pe_realistic).sample.to(weight_dtype)
            err_real = ((eps_pred_real - eps)**2).mean(dim=1)  # [BKE,H,W]
            sds_real_per = err_real.flatten(1).mean(dim=1)

        # --- 로짓/확률: fp32 강제 계산 & fp32 출력 ---
        with torch.autocast("cuda", enabled=False):
            # 1) [C,B*K*E] -> [C,B,K,E] -> (K,E) 평균 => [B,C]
            sds_per_prompt32 = (
                sds_per_prompt.float()
                .view(num_prompts, B, K, num_eps)
                .mean(dim=(2, 3))
                .permute(1, 0)
            )  # [B,C], fp32

            # 2) higher/lower 합산 (SDS 값 기준)
            sds_higher32 = sds_per_prompt32[:, POVERTY_HIGHER_IDXS].sum(dim=1)  # [B]
            sds_lower32 = sds_per_prompt32[:, POVERTY_LOWER_IDXS].sum(dim=1)  # [B]
            sds_real32 = None
            if sds_real_per is not None:
                sds_real32 = sds_real_per.float().view(B, K, num_eps).mean(dim=(1, 2))  # [B], fp32

            tau32 = torch.tensor(tau, device=accelerator.device, dtype=torch.float32)

            # 3) logits/probs shape = [B,2]
            logits32 = torch.stack([-sds_higher32 / tau32, -sds_lower32 / tau32], dim=1)  # [B,2], fp32

            # 4) 수치 안정화(softmax 불변 변환): 행별 최대값 빼기
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
            print("sds_higher32[min,max]:", float(sds_higher32.min()), float(sds_higher32.max()))
            print("sds_lower32[min,max]:", float(sds_lower32.min()), float(sds_lower32.max()))
            print("logits32 (bad rows):\n", logits32[idx])
            print("probs32 (bad rows) & row sums:\n", probs32[idx], "\nrow_sums:", row_sums[idx])
            raise FloatingPointError("Non-finite in logits/probs")

        if return_attmaps:
            return preds, probs32, logits32, sds_higher32, sds_lower32, sds_per_prompt32, sds_real32, shared_attmap_mean
        return preds, probs32, logits32, sds_higher32, sds_lower32, sds_per_prompt32, sds_real32

    def clip_poverty_classifier(
        images: torch.Tensor,
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        text_features: torch.Tensor,
        device: torch.device,
    ):
        """
        images: [B, 3, H, W]  (보통 [-1,1] 범위라고 가정)
        반환:
            preds:  [B] int64       (0: higher-class, 1: lower-class)
            probs:  [B, 2] float32
            logits: [B, 2] float32
            prompt_logits: [B, 2] float32 (각 prompt logits)
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
            prompt_logits = (image_features @ text_features.t()) * logit_scale  # [B,2]

            # 6) higher/lower 합산 -> 최종 logits
            higher_sum = prompt_logits[:, POVERTY_HIGHER_IDXS].sum(dim=1)
            lower_sum = prompt_logits[:, POVERTY_LOWER_IDXS].sum(dim=1)
            logits = torch.stack([higher_sum, lower_sum], dim=1)

            # 7) 확률 & 예측 클래스
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

        return preds, probs, logits, prompt_logits

    @torch.no_grad()
    def generate_dynamic_targets(probs, target_ratio=0.5, w_uncertainty=False):
        """generate dynamic targets for the distributional alignment loss

        Args:
            probs (torch.tensor): shape [N,2], N points in a probability simplex of 2 dims
            target_ratio (float): target distribution, the percentage of class 1 (lower-class)
            w_uncertainty (True/False): whether return uncertainty measures
        
        Returns:
            targets_all (torch.tensor): target classes
            uncertainty_all (torch.tensor): uncertainty of target classes
        """
        idxs_2_rank = (probs!=-1).all(dim=-1)
        probs_2_rank = probs[idxs_2_rank]

        rank = torch.argsort(torch.argsort(probs_2_rank[:,1]))
        targets = (rank >= (rank.shape[0]*target_ratio)).long()

        targets_all = torch.ones([probs.shape[0]], dtype=torch.long, device=probs.device) * (-1)
        targets_all[idxs_2_rank] = targets
        
        if w_uncertainty:
            uncertainty = torch.ones([probs_2_rank.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
            uncertainty[targets==1] = torch.tensor(
                1 - scipy.stats.binom.cdf(
                    (rank[targets==1]).cpu().numpy(), 
                    probs_2_rank.shape[0], 
                    1-target_ratio
                    )
                ).to(probs.dtype).to(probs.device)
            uncertainty[targets==0] = torch.tensor(
                scipy.stats.binom.cdf(
                    rank[targets==0].cpu().numpy(), 
                    probs_2_rank.shape[0], 
                    target_ratio
                    )
                ).to(probs.dtype).to(probs.device)
            
            uncertainty_all = torch.ones([probs.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
            uncertainty_all[idxs_2_rank] = uncertainty
            
            return targets_all, uncertainty_all
        else:
            return targets_all

    @torch.no_grad()
    def evaluate_process(which_text_encoder, which_unet, name, prompts, noises, current_global_step, enable_sds_eval: bool = False):
        logs = []
        log_imgs = []
        num_denoising_steps = 25
        clip_img_sims = []
        dino_img_sims = []
        clip_text_sims = []
        to_pil_eval = T.ToPILImage()

        def _clip_image_features_eval(imgs: torch.Tensor):
            imgs_01 = (imgs.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
            pil_images = [to_pil_eval(img) for img in imgs_01]
            inputs = clip_eval_processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(accelerator.device)
            with torch.no_grad():
                feats = clip_eval_model.get_image_features(pixel_values=pixel_values)
                feats = F.normalize(feats, dim=-1)
            return feats

        for prompt_i, noises_i in itertools.zip_longest(prompts, noises):
            if accelerator.is_main_process:
                logs_i = {
                    "poverty_gap": [],
                    "poverty_gap_abs": [],
                    "poverty_pred_between_0.2_0.8": [],
                }
                if enable_sds_eval:
                    logs_i["poverty_gap_abs_sds"] = []
                log_imgs_i = {}

            images_ori = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size * j:args.val_GPU_batch_size * (j + 1)]
                if args.train_text_encoder and args.train_unet:
                    images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=eval_unet)
                elif args.train_text_encoder and not args.train_unet:
                    images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=unet)
                elif not args.train_text_encoder and args.train_unet:
                    images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=eval_unet)
                images_ori.append(images_ij)
            images_ori = torch.cat(images_ori)

            shared_attmap_ori_all = None
            if enable_sds_eval:
                preds_poverty_ori_sds, probs_poverty_ori_sds, _, _, _, _, _, shared_attmap_ori = sds_logits_from_images(
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
                    return_attmaps=True,
                )
            preds_poverty_ori, probs_poverty_ori, _, _ = clip_poverty_classifier(
                images_ori,
                clip_model=clip_model,
                clip_processor=clip_processor,
                text_features=poverty_text_features,
                device=accelerator.device,
            )
            images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
            preds_poverty_ori_all = customized_all_gather(preds_poverty_ori, accelerator, return_tensor_other_processes=False)
            probs_poverty_ori_all = customized_all_gather(probs_poverty_ori, accelerator, return_tensor_other_processes=False)
            if enable_sds_eval:
                preds_poverty_ori_all_sds = customized_all_gather(preds_poverty_ori_sds, accelerator, return_tensor_other_processes=False)
                probs_poverty_ori_all_sds = customized_all_gather(probs_poverty_ori_sds, accelerator, return_tensor_other_processes=False)
                if args.region_mask_mode == "attn" and shared_attmap_ori is not None:
                    shared_attmap_ori_all = customized_all_gather(shared_attmap_ori, accelerator, return_tensor_other_processes=False)

            if accelerator.is_main_process:
                save_to_ori = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_ori.jpg")
                plot_in_grid(
                    images_ori_all,
                    save_to_ori,
                    preds_poverty=preds_poverty_ori_all,
                    pred_class_probs_poverty=probs_poverty_ori_all.max(dim=-1).values,
                )
                if enable_sds_eval:
                    save_to_ori_sds = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_ori_sds.jpg")
                    plot_in_grid(
                        images_ori_all,
                        save_to_ori_sds,
                        preds_poverty=preds_poverty_ori_all_sds,
                        pred_class_probs_poverty=probs_poverty_ori_all_sds.max(dim=-1).values,
                    )
                log_imgs_i["img_ori"] = [save_to_ori]
                if enable_sds_eval and args.save_attmaps and shared_attmap_ori_all is not None:
                    att_save_dir = os.path.join(args.imgs_save_dir, "eval_attmaps")
                    att_prefix = f"eval_{name}_{global_step}_{sanitize_filename(prompt_i)}_ori_att_house"
                    save_attmaps(shared_attmap_ori_all, att_save_dir, att_prefix)
                    save_attmaps_with_overlay(shared_attmap_ori_all, images_ori_all, att_save_dir, att_prefix, alpha=0.45)
                    att_preview = os.path.join(att_save_dir, f"{att_prefix}_0_overlay.png")
                    if os.path.exists(att_preview):
                        log_imgs_i["attmap_ori_overlay_house"] = [att_preview]

            images = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size * j:args.val_GPU_batch_size * (j + 1)]
                images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet)
                images.append(images_ij)
            images = torch.cat(images)

            shared_attmap_gen_all = None
            if enable_sds_eval:
                preds_poverty_sds, probs_poverty_sds, _, _, _, _, _, shared_attmap_gen = sds_logits_from_images(
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
                    return_attmaps=True,
                )
            preds_poverty, probs_poverty, _, _ = clip_poverty_classifier(
                images,
                clip_model=clip_model,
                clip_processor=clip_processor,
                text_features=poverty_text_features,
                device=accelerator.device,
            )
            images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
            preds_poverty_all = customized_all_gather(preds_poverty, accelerator, return_tensor_other_processes=False)
            probs_poverty_all = customized_all_gather(probs_poverty, accelerator, return_tensor_other_processes=False)
            if enable_sds_eval:
                preds_poverty_all_sds = customized_all_gather(preds_poverty_sds, accelerator, return_tensor_other_processes=False)
                probs_poverty_all_sds = customized_all_gather(probs_poverty_sds, accelerator, return_tensor_other_processes=False)
                if args.region_mask_mode == "attn" and shared_attmap_gen is not None:
                    shared_attmap_gen_all = customized_all_gather(shared_attmap_gen, accelerator, return_tensor_other_processes=False)

            if accelerator.is_main_process:
                save_to_generated = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_generated.jpg")
                plot_in_grid(
                    images_all,
                    save_to_generated,
                    preds_poverty=preds_poverty_all,
                    pred_class_probs_poverty=probs_poverty_all.max(dim=-1).values,
                )
                if enable_sds_eval:
                    save_to_generated_sds = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_generated_sds.jpg")
                    plot_in_grid(
                        images_all,
                        save_to_generated_sds,
                        preds_poverty=preds_poverty_all_sds,
                        pred_class_probs_poverty=probs_poverty_all_sds.max(dim=-1).values,
                    )
                log_imgs_i["img_generated"] = [save_to_generated]
                if enable_sds_eval and args.save_attmaps and shared_attmap_gen_all is not None:
                    att_save_dir = os.path.join(args.imgs_save_dir, "eval_attmaps")
                    att_prefix = f"eval_{name}_{global_step}_{sanitize_filename(prompt_i)}_generated_att_house"
                    save_attmaps(shared_attmap_gen_all, att_save_dir, att_prefix)
                    save_attmaps_with_overlay(shared_attmap_gen_all, images_all, att_save_dir, att_prefix, alpha=0.45)
                    att_preview = os.path.join(att_save_dir, f"{att_prefix}_0_overlay.png")
                    if os.path.exists(att_preview):
                        log_imgs_i["attmap_generated_overlay_house"] = [att_preview]

            if accelerator.is_main_process:
                probs_tmp = probs_poverty_all[(probs_poverty_all != -1).all(dim=-1)]
                poverty_gap = (((probs_tmp[:, 1] >= 0.5) * (probs_tmp[:, 1] <= 1)).float().mean() - ((probs_tmp[:, 1] >= 0) * (probs_tmp[:, 1] <= 0.5)).float().mean()).item()
                poverty_pred_between_02_08 = ((probs_tmp[:, 1] >= 0.2) * (probs_tmp[:, 1] <= 0.8)).float().mean().item()
                logs_i["poverty_gap"].append(poverty_gap)
                logs_i["poverty_gap_abs"].append(abs(poverty_gap))
                logs_i["poverty_pred_between_0.2_0.8"].append(abs(poverty_pred_between_02_08))
                if enable_sds_eval:
                    probs_tmp_sds = probs_poverty_all_sds[(probs_poverty_all_sds != -1).all(dim=-1)]
                    poverty_gap_sds = (((probs_tmp_sds[:, 1] >= 0.5) * (probs_tmp_sds[:, 1] <= 1)).float().mean() - ((probs_tmp_sds[:, 1] >= 0) * (probs_tmp_sds[:, 1] <= 0.5)).float().mean()).item()
                    logs_i["poverty_gap_abs_sds"].append(abs(poverty_gap_sds))

                with torch.no_grad():
                    clip_feats_ori_eval = _clip_image_features_eval(images_ori_all)
                    clip_feats_gen_eval = _clip_image_features_eval(images_all)
                    clip_img_sims.extend((clip_feats_gen_eval * clip_feats_ori_eval).sum(dim=-1).detach().cpu().tolist())

                    images_small_ori_eval = transforms.Resize(args.img_size_small)(images_ori_all)
                    images_small_gen_eval = transforms.Resize(args.img_size_small)(images_all)
                    dino_ori_eval = get_dino_feat(images_small_ori_eval, normalize=True, to_high_precision=True)
                    dino_gen_eval = get_dino_feat(images_small_gen_eval, normalize=True, to_high_precision=True)
                    dino_img_sims.extend((dino_gen_eval * dino_ori_eval).sum(dim=-1).detach().cpu().tolist())

                    text_inputs = clip_eval_processor(text=[prompt_i], return_tensors="pt", padding=True)
                    text_inputs = {k: v.to(accelerator.device) for k, v in text_inputs.items()}
                    clip_text_feat = F.normalize(clip_eval_model.get_text_features(**text_inputs), dim=-1)
                    clip_text_sims.extend((clip_feats_gen_eval * clip_text_feat).sum(dim=-1).detach().cpu().tolist())

            if accelerator.is_main_process:
                log_imgs.append(log_imgs_i)
                logs.append(logs_i)

        if accelerator.is_main_process:
            for prompt_i, logs_i in itertools.zip_longest(prompts, logs):
                for key, values in logs_i.items():
                    if isinstance(values, list):
                        wandb_tracker.log({f"eval_{name}_{key}_{prompt_i}": np.mean(values)}, step=current_global_step)
                    else:
                        wandb_tracker.log({f"eval_{name}_{key}_{prompt_i}": values.mean().item()}, step=current_global_step)

                for key in list(logs[0].keys()):
                    avg = np.array([log[key] for log in logs]).mean()
                    wandb_tracker.log({f"eval_{name}_{key}": avg}, step=current_global_step)

            imgs_dict = {}
            for prompt_i, log_imgs_i in itertools.zip_longest(prompts, log_imgs):
                for key, values in log_imgs_i.items():
                    if key not in imgs_dict.keys():
                        imgs_dict[key] = [wandb.Image(data_or_path=values[0], caption=prompt_i)]
                    else:
                        imgs_dict[key].append(wandb.Image(data_or_path=values[0], caption=prompt_i))
            for key, imgs in imgs_dict.items():
                wandb_tracker.log({f"eval_{name}_{key}": imgs}, step=current_global_step)

            if len(clip_img_sims) > 0:
                wandb_tracker.log({f"eval_{name}_clip_image_cos_mean": float(np.mean(clip_img_sims))}, step=current_global_step)
            if len(dino_img_sims) > 0:
                wandb_tracker.log({f"eval_{name}_dino_image_cos_mean": float(np.mean(dino_img_sims))}, step=current_global_step)
            if len(clip_text_sims) > 0:
                wandb_tracker.log({f"eval_{name}_clip_text_cos_mean": float(np.mean(clip_text_sims))}, step=current_global_step)

        return logs, log_imgs
    
    def gen_dynamic_weights_sds(targets, preds_sds, factor=0.2, out_dtype=None):
        if out_dtype is None:
            out_dtype = torch.float32
        weights = torch.ones_like(targets, dtype=out_dtype, device=targets.device)
        valid = (targets != -1)
        weights[~valid] = float(factor)
        weights[valid & (preds_sds != targets)] = float(factor)
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
        
    def evaluation_step(current_step):
        enable_sds_eval = bool(args.enable_sds_eval)
        eval_imgs_per_prompt = max(args.val_images_per_prompt_GPU, math.ceil(60 / max(1, accelerator.num_processes)))
        noises_val = torch.randn(
        [len(prompts_val), eval_imgs_per_prompt,4,64,64],
        dtype=weight_dtype_high_precision
        ).to(accelerator.device)
        evaluate_process(text_encoder, unet, "main", prompts_val, noises_val, current_step, enable_sds_eval=enable_sds_eval)

        # evaluate EMA as well
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

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, data_idx in enumerate(train_dataloader_idxs[epoch]):            
            
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                continue

            if global_step == 0 and args.eval_at_start:
                evaluation_step(global_step)

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
                    "loss_CLIP": [],
                    "loss_DINO": [],
                    "loss": [],
                    "poverty_gap": [],
                    "poverty_gap_abs": [],
                    "poverty_gap_abs_sds" : [],
                    "poverty_pred_between_0.2_0.8": [],
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
                    images_ij, _att_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet)
                    images.append(images_ij)
                images = torch.cat(images)
                preds_poverty_sds, probs_poverty_sds, logits_poverty_sds, _, _, _, _ = sds_logits_from_images(
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
                )
                preds_poverty, probs_poverty, logits_poverty, _ = clip_poverty_classifier(
                    images,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    text_features=poverty_text_features,
                    device=accelerator.device,
                )

                images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
                preds_poverty_all = customized_all_gather(preds_poverty, accelerator, return_tensor_other_processes=False)
                probs_poverty_all = customized_all_gather(probs_poverty, accelerator, return_tensor_other_processes=False)
                preds_poverty_all_sds = customized_all_gather(preds_poverty_sds, accelerator, return_tensor_other_processes=False)
                probs_poverty_all_sds = customized_all_gather(probs_poverty_sds, accelerator, return_tensor_other_processes=False)
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_generated.jpg")
                        plot_in_grid(images_all, save_to, preds_poverty=preds_poverty_all, pred_class_probs_poverty=probs_poverty_all.max(dim=-1).values)
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_generated_sds.jpg")
                        plot_in_grid(images_all, save_to, preds_poverty=preds_poverty_all_sds, pred_class_probs_poverty=probs_poverty_all_sds.max(dim=-1).values)
                        log_imgs_i["img_generated"] = [save_to]

                if accelerator.is_main_process:
                    probs_tmp = probs_poverty_all[(probs_poverty_all!=-1).all(dim=-1)]
                    poverty_gap = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                    poverty_pred_between_02_08 = ((probs_tmp[:,1]>=0.2)*(probs_tmp[:,1]<=0.8)).float().mean().item()
                    logs_i["poverty_gap"].append(poverty_gap)
                    logs_i["poverty_gap_abs"].append(abs(poverty_gap))
                    logs_i["poverty_pred_between_0.2_0.8"].append(poverty_pred_between_02_08)
                    probs_tmp = probs_poverty_all_sds[(probs_poverty_all_sds!=-1).all(dim=-1)]
                    poverty_gap_sds = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                    logs_i["poverty_gap_abs_sds"].append(abs(poverty_gap_sds))

                ################################################
                # Step 2: generate dynamic targets (based on SDS probs)
                targets_all, uncertainty_all = generate_dynamic_targets(probs_poverty_all_sds, w_uncertainty=True)
                torch.distributed.broadcast(targets_all, src=0)
                torch.distributed.broadcast(uncertainty_all, src=0)

                targets_all[uncertainty_all>args.uncertainty_threshold] = -1
                targets = targets_all[probs_poverty.shape[0]*(accelerator.local_process_index):probs_poverty.shape[0]*(accelerator.local_process_index+1)]
                uncertainty = uncertainty_all[probs_poverty.shape[0]*(accelerator.local_process_index):probs_poverty.shape[0]*(accelerator.local_process_index+1)]
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

                preds_poverty_ori_sds, probs_poverty_ori_sds, logits_poverty_ori_sds, _, _, _, _ = sds_logits_from_images(
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
                )
                preds_poverty_ori, probs_poverty_ori, logits_poverty_ori, _ = clip_poverty_classifier(
                    images_ori,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    text_features=poverty_text_features,
                    device=accelerator.device,
                )

                # embeddings for preservation losses
                images_small_ori = transforms.Resize(args.img_size_small)(images_ori)
                clip_feats_ori = get_clip_feat(images_small_ori, normalize=True, to_high_precision=True)
                DINO_feats_ori = get_dino_feat(images_small_ori, normalize=True, to_high_precision=True)

                images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
                preds_poverty_ori_all = customized_all_gather(preds_poverty_ori, accelerator, return_tensor_other_processes=False)
                probs_poverty_ori_all = customized_all_gather(probs_poverty_ori, accelerator, return_tensor_other_processes=False)
                
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_orinon.jpg")
                        plot_in_grid(images_ori_all, save_to, preds_poverty=None, pred_class_probs_poverty=None)
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_ori.jpg")
                        plot_in_grid(
                            images_ori_all, 
                            save_to, 
                            preds_poverty=preds_poverty_ori_all, 
                            pred_class_probs_poverty=probs_poverty_ori_all.max(dim=-1).values,
                        )
                        log_imgs_i["img_ori"] = [save_to]
            
            ################################################
            # Step 4: compute loss
            loss_fair_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_face_realistic_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_CLIP_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_DINO_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            
            idxs_i = list(range(targets.shape[0]))
            N_backward = math.ceil(targets.shape[0] / args.train_GPU_batch_size)
            for j in range(N_backward):
                idxs_ij = idxs_i[j*args.train_GPU_batch_size:(j+1)*args.train_GPU_batch_size]
                noises_ij = noises_i[idxs_ij]
                targets_ij = targets[idxs_ij]
                clip_feats_ori_ij = clip_feats_ori[idxs_ij]
                DINO_feats_ori_ij = DINO_feats_ori[idxs_ij]
                
                images_ij, _att_ij = generate_image_w_gradient(
                    prompt_i,
                    noises_ij,
                    num_denoising_steps,
                    which_text_encoder=text_encoder,
                    which_unet=unet,
                    skip_final_steps=args.skip_final_steps,
                    skip_final_steps_pct=args.skip_final_steps_pct,
                )

                # --- SDS classifier WITH gradient + region hard mask ---
                preds_poverty_ij, probs_poverty_ij, logits_poverty_ij, _, _, _, sds_realistic_ij = sds_logits_from_images(
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
                    compute_realistic_sds=True,
                )

                
                # (Deprecated) grad hook replaced by attention-region masking inside SDS computation

                images_small_ij = transforms.Resize(args.img_size_small)(images_ij)
                clip_feats_ij = get_clip_feat(images_small_ij, normalize=True, to_high_precision=True)
                DINO_feats_ij = get_dino_feat(images_small_ij, normalize=True, to_high_precision=True)

                loss_CLIP_ij = - (clip_feats_ij * clip_feats_ori_ij).sum(dim=-1) + 1
                loss_DINO_ij = - (DINO_feats_ij * DINO_feats_ori_ij).sum(dim=-1) + 1
                
                loss_fair_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) *(-1)
                idxs_valid = (targets_ij != -1).nonzero().view([-1])
                logits_poverty_ij = logits_poverty_ij.half()
                if idxs_valid.numel() > 0:
                    loss_fair_ij[idxs_valid] = CE_loss(logits_poverty_ij[idxs_valid], targets_ij[idxs_valid])

                if sds_realistic_ij is None:
                    loss_face_realistic_ij = torch.zeros(len(idxs_ij), dtype=weight_dtype, device=accelerator.device)
                else:
                    loss_face_realistic_ij = sds_realistic_ij.to(weight_dtype)

                # dynamic weights: compare target vs original-side SDS prediction
                preds_poverty_ori_sds_ij = preds_poverty_ori_sds[idxs_ij]
                dynamic_weights = gen_dynamic_weights_sds(
                    targets_ij,
                    preds_poverty_ori_sds_ij,
                    factor=args.factor1,
                    out_dtype=loss_CLIP_ij.dtype,
                )

                loss_ij = (
                    loss_fair_ij
                    + args.weight_loss_img * dynamic_weights * (loss_CLIP_ij + loss_DINO_ij)
                    + args.weight_loss_face_realistic * loss_face_realistic_ij
                )
                accelerator.backward(loss_ij.mean())

                with torch.no_grad():
                    loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)
                    loss_face_realistic_i[idxs_ij] = loss_face_realistic_ij.to(loss_face_realistic_i.dtype)
                    loss_CLIP_i[idxs_ij] = loss_CLIP_ij.to(loss_CLIP_i.dtype)
                    loss_DINO_i[idxs_ij] = loss_DINO_ij.to(loss_DINO_i.dtype)
                    loss_i[idxs_ij] = loss_ij.to(loss_i.dtype)
                    
            # for logging purpose, gather all losses to main_process
            accelerator.wait_for_everyone()
            loss_fair_all = customized_all_gather(loss_fair_i, accelerator)
            loss_face_realistic_all = customized_all_gather(loss_face_realistic_i, accelerator)
            loss_CLIP_all = customized_all_gather(loss_CLIP_i, accelerator)
            loss_DINO_all = customized_all_gather(loss_DINO_i, accelerator)
            loss_all = customized_all_gather(loss_i, accelerator)

            loss_all = loss_all[loss_fair_all!=-1]
            loss_fair_all = loss_fair_all[loss_fair_all!=-1]
            loss_face_realistic_all = loss_face_realistic_all[loss_face_realistic_all!=-1]

            if accelerator.is_main_process:
                logs_i["loss_fair"].append(loss_fair_all)
                logs_i["loss_face_realistic"].append(loss_face_realistic_all)
                logs_i["loss_CLIP"].append(loss_CLIP_all)
                logs_i["loss_DINO"].append(loss_DINO_all)
                logs_i["loss"].append(loss_all)
            
            # process logs
            if accelerator.is_main_process:
                for key in ["loss_fair", "loss_face_realistic", "loss_CLIP", "loss_DINO", "loss"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)
                    else:
                        logs_i[key] = torch.cat(logs_i[key])
                for key in ["poverty_gap", "poverty_gap_abs", "poverty_gap_abs_sds","poverty_pred_between_0.2_0.8"]:
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

            if global_step % args.evaluate_every_n_iter == 0:
                evaluation_step(global_step)

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
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
