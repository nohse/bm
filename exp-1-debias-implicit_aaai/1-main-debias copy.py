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
# See the License for the specific language governing permissions and

# =====================================================================================
# MULTIPROMPT_VALENCE_AXISATTN = the MULTIPROMPT_VALENCE file with PER-AXIS cross-attention weighting added
# back (--valence_axis_attn peraxis, default). Each axis's error is scored in its own content region (see
# CROSS-ATTENTION MAP below); --valence_axis_attn uniform reproduces the no-attmap valence file bit-for-bit.
# The CLASS is no longer gender
# (ONE prompt pair: "a photo of a woman" vs "a photo of a man"); it is POSITIVE vs NEGATIVE valence,
# defined JOINTLY by P=4 CONTRASTIVE PROMPT PAIRS (--valence_axes), each a minimal contrast in which
# only the polarity word differs:
#   expr  : "A photo of positive facial expressions"              vs "... negative facial expressions"
#   cloth : "A photo of positive clothing and actions"            vs "... negative clothing and actions"
#   bg    : "A photo of a background with a positive atmosphere"  vs "... a negative atmosphere"
#   tone  : "A photo of a scene with an overall positive tone"    vs "... an overall negative tone"
# The objective is unchanged in spirit: drive the generated distribution to 50/50 positive/negative.
#
# ------------------------------------------------------------------------------------------------
# HOW THE 4 PAIRS ARE POOLED   (--valence_pooling, default gap_mean)
# ------------------------------------------------------------------------------------------------
# Per axis p the frozen scoring UNet gives two errors E_p^pos, E_p^neg (eps-MSE under each prompt).
# The per-axis signed GAP is   g_p = E_p^neg - E_p^pos     (g_p > 0  <=>  the image reads POSITIVE).
#
# THE KEY MATH RESULT (verified in float64: max abs diff 4.6e-14):
#       mean-of-ERRORS  ==  mean-of-LOGITS  ==  mean-of-LOG-PROBS
# These are NOT three options -- they are ONE estimator. logit_c = -E_c/tau is AFFINE in E with a
# shared tau, so averaging over p commutes with it; and mean_p log_softmax(l_p) differs from
# mean_p l_p only by a CLASS-INDEPENDENT term (the per-axis log-partition), which softmax/CE are
# invariant to. All three give the same posterior AND the same gradient:
#       p(pos) = sigmoid( (1/P) * sum_p g_p / tau )
# Two consequences:
#   (+) each axis's BASELINE difficulty (prompt length, rarity, absolute error level) CANCELS EXACTLY,
#       because g_p is a WITHIN-pair difference. This is why minimal-contrast pairs matter.
#   (-) each axis's GAP SCALE does NOT cancel. An unweighted mean of raw gaps is dominated by whichever
#       axis has the largest dynamic range (in simulation one axis took 66% of the decision budget), so
#       a "4-axis" scorer silently collapses into a 1-axis scorer.  ->  --valence_axis_scale
#       But 1/std ALONE is a noise amplifier: it equalises VARIANCE, not DISCRIMINABILITY, so a DEAD
#       axis (pure Monte-Carlo noise) gets inflated to unit variance and injected at weight 1/P.
#       ->  --valence_axis_weight, set proportional to each axis's OFFLINE-MEASURED discriminability,
#           with any axis below the AUC floor set to 0.
#
# mean-of-PROBABILITIES is the ONE genuinely different pooling, and it is the one that BREAKS. It is
# reachable as --valence_pooling prob_mean for the ablation, but it is not the default because:
#   (a) it is a MIXTURE of experts, so one confident axis cannot be outvoted by the rest, and a
#       saturated axis contributes ~zero gradient (dp/dg ∝ p(1-p) -> 0);
#   (b) under a small tau each per-axis sigmoid becomes a hard 0/1 VOTE, so the mean lands on the
#       discrete grid {0, .25, .5, .75, 1} -- and generate_dynamic_targets ranks that with argsort, so
#       the massively-tied images get their 50/50 targets assigned by ARRIVAL ORDER rather than by
#       valence. Worse, argsort still emits a PERFECT 50/50 count, so the failure is invisible in wandb.
#
# RANKING (step 2): targets are ranked on the CONTINUOUS pooled score, never on softmax probs. The two
# are mathematically equivalent (probs[:,1] = sigmoid(score/tau) is strictly increasing), but fp32
# sigmoid returns EXACTLY 1.0 once score/tau > ~16.6, manufacturing ties out of thin air. The score
# cannot saturate. This also retires the -1 fill-value hack from the ranking population.
#
# NOISE (--valence_axis_eps, default independent): eps is shared WITHIN a pair (the paired difference
# is what makes g_p low-variance) but drawn INDEPENDENTLY ACROSS axes. Sharing one eps across all 8
# prompts -- the obvious port of the old 2-prompt code -- would correlate the P gap estimates through a
# single noise draw, so averaging them would reduce no Monte-Carlo noise at all and the entire point of
# pooling would be lost.
#
# ------------------------------------------------------------------------------------------------
# WHAT HAPPENS TO THE FACE MACHINERY (the concept is now SCENE-LEVEL, not face-bound)
# ------------------------------------------------------------------------------------------------
# FACE DETECTOR (--valence_face_gate, DEFAULT errfd): KEPT. This file changes ONLY the spatial SCR gate
#   (mechanism 4, --factor2); the FACE-based gates are deliberately left intact. So by default:
#     (1) a sample joins the fair/DAL loss only if the residual-error detector says it has a face, and
#     (2) a no-face sample keeps SCR weight 1 (bypasses the flip/uncertain damping).
#   Rationale for keeping them here: the class axes include face-bound content (expr = "facial expressions"),
#   so a faceless generation's valence is partly ill-defined; the design decision was to touch only the
#   whole-frame spatial gate, not the participation gate. SAFETY: get_valence scatters non-face rows to
#   fill_value so gathered SHAPES stay uniform (no NCCL shape-split), and pins fp32/int64 dtypes in BOTH the
#   empty and non-empty branches (no dtype-split deadlock -- the bug that bit this project twice). The
#   OPPOSITE ablation --valence_face_gate none makes every image participate (a faceless grim alleyway HAS a
#   valence), which aligns training with the whole-image CLIP eval head and drops the detector dependency, at
#   the cost of face-axis noise on faceless images (multi-axis pooling mitigates it). Measure the faceless
#   fraction and compare both on eval before committing.
# CROSS-ATTENTION MAP (AXISATTN, --valence_axis_attn peraxis, DEFAULT): REINSTATED, but PER AXIS, not as a
#   single common map. Each axis p weights its per-pixel error by the sum-to-1, DETACHED cross-attention of
#   that axis's own CONTENT phrases (--valence_axes 4th field): expr -> "facial expressions" (face);
#   cloth -> "clothing,actions" (body); bg -> "background,atmosphere" (scene); tone -> (empty) => uniform.
#   Each aspect is thus scored WHERE IT LIVES. This is DIFFERENT from the gender file's single COMMON map,
#   and that difference answers the four objections that (correctly) killed a common map for valence:
#   1. NON-CONTIGUOUS content ("clothing ... actions"): handled by UNIONING the token run of each phrase
#      (_find_content_token_indices), so the function word "and" is never averaged in.
#   2. INCOMMENSURABLE prompt lengths: each axis's map is normalised to sum-to-1 on ITS OWN, so a longer
#      prompt's larger raw columns cannot dominate -- the weighted error keeps a uniform-mean scale.
#   3. AXES LOCALISE ELSEWHERE (the fatal one for a common map): here the maps are NEVER averaged across
#      axes. Pooling happens only AFTER each axis's spatial reduction, so nothing smears toward uniform.
#   4. AXIS WITH NO NOUN (tone): given an empty content field, so it stays a uniform whole-image mean --
#      exactly right for a global axis.
#   THE MAP IS SHARED BY THE PAIR (averaged over the pos & neg forwards, detached) so the gap g_p = E_neg -
#   E_pos stays a clean paired difference -- the analogue of the woman/man common map, kept per axis.
#   COST: 'peraxis' installs CrossAttnCaptureProcessor, which forces the explicit get_attention_scores + bmm
#   path on the scoring UNet (no SDPA). --valence_axis_attn uniform reproduces the no-attmap file bit-for-bit
#   (processor NOT installed, SDPA kept). EMPIRICAL RISK: whether SD attention localises abstract nouns
#   ("atmosphere","actions") is unproven -- validate per-axis localisation offline (see BEFORE YOU TRAIN); a
#   diffuse map degrades gracefully to ~uniform, but a map that localises to the WRONG place can hurt.
# SRR REALISM LOSS: --srr_prompt default becomes "a realistic photo" (was "a photo of a realistic
#   person"), and its gradient is no longer input-masked to the person region -- it now covers the WHOLE
#   image. The realism anchor must constrain the same support the fair loss pushes on; the fair loss now
#   edits expression AND clothing AND background AND global tone, so a person-region-only anchor would
#   leave the model free to wreck the background in order to win the valence objective.
# SCR FLIP GATE: the SPATIAL gate is OFF by default (--factor2 1.0, was 0.2). Its meaning was "damp the
#   preservation loss WHERE the class lives, so that region is free to change" -- but for a scene-level
#   class that is the whole frame, so the mask is vacuous. NOTE the ATTMAP file damped the SAME release
#   set TWICE (factor1 globally AND factor2 inside the region = 0.2 x 0.2 = 0.04 of the kept gradient);
#   factor1 alone is kept as the single, explicit release damping.
# EVALUATION: the CelebA MobileNet gender classifier consumes 224x224 FACE CROPS and cannot be repointed
#   at a scene-level concept even in principle. It (and insightface) are replaced by a ZERO-SHOT CLIP
#   VALENCE HEAD on the open_clip ViT-bigG-14 this file ALREADY loads on rank 0 for CLIP-I/CLIP-T. It
#   scores the FULL image, never a face chip, and is INDEPENDENT of the training signal (CLIP is not in
#   this file's training loss). The per-class text prompts are prompt-ensembled in EMBEDDING space (the
#   canonical CLIP zero-shot recipe -- and exactly "average within a class", done where it is correct).
#
# ------------------------------------------------------------------------------------------------
# BEFORE YOU TRAIN: run valence_exp/valence_separation.py.
# This project has TWICE shipped a plausible-looking prompt-based residual scorer that carried NO signal
# (a fair loss pinned at ln 2; a "realistic face" gate with face/no-face AUC 0.504). "positive"/"negative"
# are ABSTRACT ADJECTIVES and SD's CLIP text encoder may barely move on them. The harness measures
# per-axis ROC-AUC, the per-axis gap scale (who dominates), the axis correlation matrix and the pooled
# AUC, on latents produced the SAME truncated way training scores them, and prints the
# --valence_axis_scale / --valence_axis_weight constants to paste in.
# AXISATTN adds a SECOND thing to validate: does each axis's content attention actually LOCALISE to its
# region (face / body / scene)? Measure the AUC with vs without the per-axis map and eyeball the maps; if a
# map is diffuse the weighting is a harmless ~no-op, but if it localises to the WRONG place set that axis's
# content field empty (-> uniform). Do NOT assume "atmosphere"/"actions" localise just because they parse.
# GATE: per-axis AUC >= 0.60 and pooled AUC >= 0.75. Do not spend a GPU-hour on training before it passes.
# Run-folder tag: _val-<P>ax-<pooling>[-uncal]_vAttn-<peraxis|uniform>  (uncal = calibration left at default).
# =====================================================================================
# Inherited from ATTMAP: a switch on the SPATIAL REDUCTION of the class error:
#     --gender_attn_weight {attn, none}   (kept only for the gender ablation; IGNORED for valence)
# The woman/man gender error E_c is the per-pixel squared eps-residual of the frozen scoring UNet
# under the "woman"/"man" text condition. The original file always reduces it with the COMMON
# woman/man cross-attention map (sum-to-1 normalized) as a spatial weight, i.e. E_c = sum_{u,v}
# attn(u,v) * res_c(u,v) -- re-weighting the error toward the gender/person region. This file makes
# that weighting OPTIONAL:
#   attn : E_c = sum_{u,v} common_attn(u,v) * res_c(u,v)        (attention-weighted SUM; original)
#   none : E_c = mean_{u,v} res_c(u,v)                          (the FULL error, uniform over space)
# The 'none' branch is exactly the 'attn' branch with a FLAT weight map 1/(H*W), which also sums to
# 1 -- so E_c keeps the same normalization/scale and --tau does NOT need to be re-tuned a priori.
# HOW BIG IS THE DIFFERENCE? Measured on 8 real generated images (K=15, t 400-800): this gender attn
# map is DIFFUSE, not a face mask (values 1.7e-4..4.7e-4 around the 2.44e-4 uniform value, ~2.8x
# max/min, participation ratio ~3900 of 4096 px). So 'attn' is a MILD re-weighting: E_woman
# none/attn = 1.04, the class GAP |E_man - E_woman| none/attn = 0.83, and the argmax gender preds
# agree 8/8. The flag mainly rescales the class-error GAP (hence the fair-loss gradient), and does
# NOT make the z0 gradient uniform (it flows through the scoring UNet's global receptive field:
# top-10% |grad| energy 0.225 attn -> 0.196 none, vs 0.10 for a truly uniform field).
# The flag applies to BOTH places the class error is computed: residual_gender_and_realism (the SCR
# fair loss) and residual_gender_logits (the training-time gender predictor behind get_face_gender).
# What the flag does NOT change (the cross-attention map is still computed in BOTH modes):
#   - the SRR realism gradient region  (min-max common_attn >= --attn_gate_thr, input-masked z0)
#   - the h-space SCR flip gradient gate (same region, scaled by --factor2)
#   - the attention/grad-gate visualizations (--save_attn_maps)
# Only the gender-error reduction switches; everything else is byte-for-byte the original file.
# Run-folder tag: the mode is ALWAYS written into the output folder / wandb run name --
#   _gAttn-attn = attmap weighting ON (original behaviour) , _gAttn-none = attmap weighting OFF
# so a run's own folder states whether the attmap multiply was used (an absent tag would be
# ambiguous with the original _nodetector.py runs, which carry no _gAttn tag at all). Example:
#   ..._wSRR-4_srr-person_errFD-8-50-950_gAttn-none_skipFrac-0.5_Th-0.2_loraR-50_lr-5e-05_07131530
# =====================================================================================
# NODETECTOR = SRR_person_truncated_hspace with the insightface FACE DETECTOR REPLACED -- in the
# TRAINING branch points ONLY -- by a diffusion residual-error face/no-face classifier:
#     face  iff  E("a photo of a face") < E("a faceless photo")
# where E_c is the conditional eps-prediction MSE of the FROZEN scoring UNet on the clean latent
# z0 (cond scheme, UNIFORM spatial mean, NO attention weighting), averaged over
# --face_residual_num_timesteps=8 timesteps linspaced in [--face_residual_t_min=50,
# --face_residual_t_max=950]. The prompt pair / t-range / K=8 come from the offline ablation in
# face_error_exp/exp100 (50 genuine-face vs 50 genuine-noface occupation images): this exact pair
# scores 88/100 at K=8 over t50-950, equal to its K=15 score (K<8 degrades).
# REPLACED SITES (all three TRAINING uses of get_face; NOTHING else is touched):
#   step 1: face_indicators     -> gates get_face_gender -> targets(-1) -> fair-loss participation
#   step 3: face_indicators_ori -> gates preds/probs_gender_ori -> SCR release gate + dyn. weight
#   step 4: face_indicators_ij  -> fair-loss idxs_w_face_loss, SCR attn zero-out, gen_dynamic_weights
#   face_bboxs at steps 1/3 become fill_value(-1) dummies (they were only used by plot_in_grid).
# EVALUATION (evaluate_process) still uses the real insightface detector + the external test
# classifier, so eval metrics stay comparable to all baselines.
# The output folder name gets an extra tag: _errFD-<K>-<tmin>-<tmax>  (e.g. _errFD-8-50-950).
# =====================================================================================
# SRR_person_truncated_hspace = SRR_person_truncated with the image-semantics loss REPLACED by the
# h-space SCR image loss (ported from 1-main-errorDAL,SCR,Face_hspace_truncated.py). The fairness
# (SCR gender logits) loss and the SRR realism loss are UNCHANGED; only the old CLIP+DINO image
# loss (--weight_loss_img) is swapped for the scoring-space per-timestep MSE of the FROZEN scoring
# UNet's mid_block (h-space) outputs of the ORIGINAL vs FINETUNED latents (--weight_loss_scr).
# The SCR gradient is spatially gated on flip/uncertain samples exactly as in the Face_hspace file
# (min-max gender attn >= --attn_gate_thr, scaled by --factor2); the gate reuses the gender
# cross-attention map already computed by the fused SRR/gender scorer (no extra scorer forward).
# TIMESTEPS: the woman/man class error + SRR realism are scored over --residual_t_min/max (default
# 400-800), while the SCR h-space MSE uses its OWN lower-noise grid --scr_t_min/max/--scr_num_timesteps
# (default 100-400, 15 steps) so it targets image structure rather than the coarse gender signal.
# CLIP-I / DINO-I remain as EVALUATION metrics only.
#
# SRR_person_truncated = SRR_person + TRUNCATED DENOISING (--skip_denoise_frac, default 0.5).
# During TRAINING, images are generated by running only the first n_run = round((1-frac)*
# num_denoising_steps) scheduler steps, then jumping to a predicted clean latent x0 via the
# closed-form eps->x0 formula (no extra UNet call). Applied to all three training passes
# (current-model, original/frozen, gradient) so reference and trained images stay comparable.
# EVALUATION always passes skip_denoise_frac=0.0 (full denoising) so metrics match baselines.
#
# INTERACTION NOTE (truncated z0 x SRR realism): with frac>0, z0 is a blurrier x0 estimate.
# The SRR realism loss (residual_gender_and_realism) scores this z0 with "a photo of a
# realistic person"; because E_realistic is a RAW eps-residual, part of it now reflects the
# truncation blur rather than gender-induced unrealism, so the realism gradient partly fights
# the truncation. The SCR gender logits (a woman-vs-man DIFFERENCE) are more robust to this
# shared blur. If the SRR term misbehaves under truncation, try a smaller skip_denoise_frac
# or re-tune weight_loss_face. This is inherent to putting any z0-based loss on a truncated
# trajectory, not a bug.
# =====================================================================================

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
import hashlib
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

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPModel
import open_clip
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, GradScalerKwargs

import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.training_utils import EMAModel

# you MUST import torch before insightface
# otherwise onnxruntime, used by FaceAnalysis, can only use CPU
from insightface.app import FaceAnalysis



my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.

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

def plot_in_grid(images, save_to, preds_class=None, pred_class_probs=None):
    """Grid of generated images, sorted and colour-coded by the predicted VALENCE class.

    images: torch tensor [N,3,H,W], in range [-1,1]
    preds_class: [N] int64, class order [negative=0, positive=1]; -1 == not scored (only possible under
        --valence_face_gate errfd, where unscored images exist at all).
    pred_class_probs: [N] the probability of the PREDICTED class (i.e. probs.max(-1).values), used both to
        sort within a class and to draw the confidence bar.

    The face bbox / face-indicator drawing is GONE: valence is a scene-level property, so there is no box to
    draw, and images are no longer partitioned into face / no-face.
    """
    idxs_pos = (preds_class == 1).nonzero(as_tuple=False).view([-1])
    idxs_pos = idxs_pos[pred_class_probs[idxs_pos].argsort(descending=True)]

    idxs_neg = (preds_class == 0).nonzero(as_tuple=False).view([-1])
    idxs_neg = idxs_neg[pred_class_probs[idxs_neg].argsort(descending=True)]

    idxs_unscored = (preds_class == -1).nonzero(as_tuple=False).view([-1])

    images_to_plot = []
    # most-positive -> least-positive, then most-negative -> least-negative, then unscored
    idxs_reordered = torch.cat([idxs_pos, idxs_neg, idxs_unscored])

    for idx in idxs_reordered:
        img = images[idx]
        pred_class = preds_class[idx]
        pred_class_prob = pred_class_probs[idx]

        if pred_class == 1:
            border_color = "green"      # positive
        elif pred_class == 0:
            border_color = "purple"     # negative
        else:
            border_color = "white"      # unscored

        img_pil = transforms.ToPILImage()(img*0.5+0.5)
        img_pil = ImageOps.expand(img_pil, border=(50,0,0,0),fill=border_color)

        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob.item())*512)], fill ="white", outline =None)

        fnt = ImageFont.truetype(font="../data/0-utils/arial-bold.ttf", size=100)
        img_pil_draw.text((400, 400), f"{idx.item()}", align ="left", font=fnt)

        img_pil = ImageOps.expand(img_pil_draw._image, border=(10,10,10,10),fill="black")
        
        images_to_plot.append(img_pil)
        
    N_imgs = len(images_to_plot)
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

def save_noface_images(images, face_indicators, save_dir, tag):
    """NODETECTOR: save each image the errFD classifier marked NO-FACE as its own JPG.

    images: torch tensor [N,3,H,W] in range [-1,1].
    face_indicators: bool tensor [N] (True == face). Images where this is False are saved.
    save_dir: destination directory (created if missing).
    tag: filename prefix, e.g. "train_step40" or "eval_EMA_200_<occupation>".
    Returns the number of images written. Caller should guard on accelerator.is_main_process.
    """
    fi = face_indicators.detach().cpu().bool()
    idxs = fi.logical_not().nonzero(as_tuple=False).view(-1).tolist()
    if len(idxs) == 0:
        return 0
    os.makedirs(save_dir, exist_ok=True)
    for i in idxs:
        img_pil = transforms.ToPILImage()((images[i].detach().cpu() * 0.5 + 0.5).clamp(0, 1))
        img_pil.save(os.path.join(save_dir, f"{tag}_idx{i}.jpg"), quality=90)
    return len(idxs)

def _sanitize_tag(text, maxlen=40):
    """filesystem-safe short slug from a prompt (for no-face image filenames)."""
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(text)).strip("_")
    return s[:maxlen] if s else "prompt"

def _normalize_attmap_for_vis(att):
    """Contrast-stretch a single HxW attention map for visualization only (quantile 0.05-0.995, sqrt lift)."""
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
    return scaled.pow(0.5)


def _attmap_vis_to_pil(vis):
    a = vis.clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
    img = Image.fromarray(a, mode="L")
    return ImageOps.colorize(img, black="black", mid="orange", white="red")


def attmap_overlay_on_image(att_map, image, alpha=0.45):
    """Overlay one HxW attention map (any scale) onto one 3xHxW image tensor in [-1,1]; returns a PIL RGB image."""
    img_pil = transforms.ToPILImage()(image.detach().cpu().mul(0.5).add(0.5).clamp(0, 1))
    vis = _normalize_attmap_for_vis(att_map)
    if vis.shape[-2:] != (img_pil.size[1], img_pil.size[0]):
        vis = torch.nn.functional.interpolate(
            vis.unsqueeze(0).unsqueeze(0), size=(img_pil.size[1], img_pil.size[0]),
            mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
    att_pil = _attmap_vis_to_pil(vis)
    base = img_pil.convert("RGBA")
    heat = att_pil.convert("RGBA")
    alpha_mask = Image.fromarray(vis.mul(255 * alpha).clamp(0, 255).to(torch.uint8).cpu().numpy(), mode="L")
    heat.putalpha(alpha_mask)
    return Image.alpha_composite(base, heat).convert("RGB")


def mask_overlay_on_image(mask, image, alpha=0.5, color=(255, 165, 0)):
    """Overlay a (near-)binary HxW mask onto one 3xHxW image tensor in [-1,1] as a solid-color tint.

    Unlike attmap_overlay_on_image this does NOT contrast-stretch, so the min-max hard mask
    (attn_gate >= thr) and the applied gradient-gate region are rendered faithfully (a pixel is
    tinted iff mask>0, with opacity proportional to the mask value). Returns a PIL RGB image.
    """
    img_pil = transforms.ToPILImage()(image.detach().cpu().mul(0.5).add(0.5).clamp(0, 1))
    m = mask.detach().float().cpu()
    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    if m.shape[-2:] != (img_pil.size[1], img_pil.size[0]):
        m = torch.nn.functional.interpolate(
            m.unsqueeze(0).unsqueeze(0), size=(img_pil.size[1], img_pil.size[0]), mode="nearest",
        ).squeeze(0).squeeze(0)
    base = img_pil.convert("RGBA")
    tint = Image.new("RGBA", base.size, color + (0,))
    alpha_mask = Image.fromarray(m.mul(255 * alpha).clamp(0, 255).to(torch.uint8).numpy(), mode="L")
    tint.putalpha(alpha_mask)
    return Image.alpha_composite(base, tint).convert("RGB")


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
        help="provide the checkpoint path to resume from checkpoint. NOTE: kept None for the SRR_person "
             "experiment so it starts fresh from pretrained SD -- resuming from a face-prompt SRR checkpoint "
             "would carry over weights trained on the old 'a photo of a realistic face' prompt and "
             "contaminate this run. Pass an explicit path only to resume an interrupted SRR_person run.",
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
        '--skip_first_eval',
        action="store_true",
        default=True,
        help="if set, skip the evaluation at global_step==0 and start training directly",
        )
    parser.add_argument(
        '--save_attn_maps',
        action="store_true",
        default=False,
        help="UNUSED FOR VALENCE (kept so existing yaml configs still load). It saved the class cross-attention "
             "maps and the SCR gradient-gate panels. Neither exists for a scene-level class; the per-axis GAP "
             "metrics logged to wandb (gap_<axis>_mean / _std / _corr_pooled) are the replacement diagnostic, "
             "and they are the ones that actually reveal a dead or dominating axis.",
        )
    parser.add_argument(
        '--save_noface_imgs',
        action="store_true",
        default=True,
        help="ON by default. Whenever the errFD face indicator marks an image as NO-FACE, save that "
             "single image to <run>/noface_imgs/{train,eval}/ so it can be inspected (to sanity-check the "
             "classifier for false negatives). Train saves the gathered current-model generations at EVERY "
             "step; eval saves the gathered evaluated images. Pass --no-save_noface_imgs style override via "
             "config (set to 0/false) to disable if disk volume is a concern.",
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
        '--weight_loss_scr',
        default=8,
        help="weight for the SCR image loss (scoring-space per-timestep MSE of the FROZEN scoring UNet's "
             "mid_block / h-space outputs of the original vs finetuned latents). Replaces the old CLIP+DINO "
             "image-semantics loss (--weight_loss_img); configs that still set weight_loss_img are aliased to "
             "this key (see parse_args).",
        type=float,
    )
    parser.add_argument(
        '--weight_loss_face',
        default=1,
        help="weight for the SRR (realistic-person SDS) loss. NOTE: kept named --weight_loss_face so the "
             "shared debias-*.yaml configs (which set weight_loss_face) apply to this SRR experiment too; "
             "it no longer weights the old face-feature realism-preserving loss (removed).",
        type=float,
    )
    parser.add_argument(
        '--srr_prompt',
        default="a photo of a realistic photo",
        type=str,
        help="text prompt whose frozen-SD diffusion residual error is used directly as the SRR realism "
             "loss. Scored by the fused residual scorer over --residual_t_min/max and "
             "--residual_num_timesteps (no separate timestep args). VALENCE: the default changed from "
             "'a photo of a realistic person' to 'a realistic photo', and the loss's gradient is no "
             "longer masked to the person region (see --srr_grad_region). The realism anchor has to "
             "constrain the same support the fair loss pushes on, and the valence fair loss edits "
             "expression AND clothing AND background AND global tone -- a person-region-only anchor "
             "would leave the model free to wreck the background to win the valence objective.",
    )
    # NOTE (removed on purpose): the gender file input-masked the SRR gradient to the person region and
    # spatially damped the SCR gradient inside it. Both regions came from the class-prompt cross-attention
    # map, which CANNOT BE BUILT for these axes (non-contiguous content words, incommensurable prompt
    # lengths, and 4 axes whose supports average to ~uniform -- see the header). Rather than keep a
    # half-dead --srr_grad_region/--valence_attn_capture pair that nothing can correctly satisfy, the
    # region machinery is gone: the SRR gradient covers the whole image, and factor1 is the single release
    # damping. --attn_gate_thr / --factor2 survive ONLY so existing yaml configs still load (see below).
    parser.add_argument(
        '--gender_attn_weight',
        default="none",
        type=str,
        choices=["attn", "none"],
        help="UNUSED FOR VALENCE (kept so existing yaml configs still load). It selected the spatial reduction of "
             "the woman/man class error. The valence class error always uses a UNIFORM spatial mean: there "
             "is no coherent region to weight a scene-level class by, and the cross-attention map cannot "
             "even be built for these axes (see the header). Note this flag ALREADY defaulted to \'none\' in "
             "the gender file, so a uniform mean is also what that file was actually doing.",
    )
    parser.add_argument(
        '--attn_gate_thr',
        default=0.15,
        help="UNUSED FOR VALENCE (kept so existing yaml configs still load). It was the min-max cross-attention "
             "threshold defining the person region that gated the SRR gradient and the SCR flip damping. "
             "Both regions are gone: the SRR gradient now covers the whole image and the spatial SCR gate is "
             "removed (see --factor2).",
        type=float,
    )
    parser.add_argument(
        '--uncertainty_threshold',
        help="the uncertainty threshold used in distributional alignment loss", 
        type=float, 
        default=0.2
        )
    parser.add_argument(
        '--factor1', type=float, default=0.2,
        help="GLOBAL damping of the SCR image-preservation loss for RELEASED samples (flip or uncertain), "
             "applied as a per-sample scalar by gen_dynamic_weights. This is the single, explicit release "
             "damping for valence.",
        )
    parser.add_argument(
        '--factor2', type=float, default=1.0,
        help="IGNORED FOR VALENCE (accepted so the existing debias-*.yaml configs still load; a non-1.0 "
             "value is warned about at startup). It used to be the SPATIAL damping of the SCR gradient "
             "inside the class cross-attention region for released samples. That region does not exist for "
             "a scene-level class -- 'where the class lives' is the whole frame, so the mask is vacuous. "
             "Note the gender file also damped the SAME release set TWICE: factor1 globally AND factor2 "
             "inside the region (0.2 x 0.2 = 0.04 of the kept gradient). factor1 is now the single, "
             "explicit release damping.",
        )

    # ------------------------------------------------------------------ multi-prompt VALENCE class
    parser.add_argument(
        '--valence_axes',
        type=str,
        nargs='+',
        default=[
            "expr::A photo of positive facial expressions::A photo of negative facial expressions::facial expressions",
            "cloth::A photo of positive clothing and actions::A photo of negative clothing and actions::clothing,actions",
            "bg::A photo of a background with a positive atmosphere::A photo of a background with a negative atmosphere::background,atmosphere",
            "tone::A photo of a scene with an overall positive tone::A photo of a scene with an overall negative tone::",
        ],
        help="the P contrastive prompt PAIRS that jointly define the binary class, each as "
             "'name::POSITIVE prompt::NEGATIVE prompt[::CONTENT phrases]'. Use MINIMAL CONTRASTS (only the "
             "polarity word differs): the per-axis gap g_p = E_neg - E_pos is a WITHIN-pair difference, so "
             "each axis's baseline error level (prompt length, rarity) cancels EXACTLY -- but only if the "
             "pair is otherwise identical. Class order is [negative=0, positive=1]. The optional 4th field is "
             "a comma-separated list of CONTENT PHRASES (AXISATTN variant): under --valence_axis_attn peraxis "
             "that axis's per-pixel error is spatially weighted by the sum-to-1 cross-attention of these "
             "phrases (facial expressions -> face; clothing,actions -> body; background,atmosphere -> scene), "
             "so each aspect is scored WHERE IT LIVES. Each phrase must appear verbatim in BOTH prompts. An "
             "EMPTY 4th field (e.g. the 'tone' axis) => UNIFORM whole-image mean, correct for a global axis "
             "with no localisable content noun.",
    )
    parser.add_argument(
        '--valence_axis_attn',
        type=str,
        default="peraxis",
        choices=["peraxis", "uniform"],
        help="AXISATTN: spatial reduction of each axis's per-class error. 'peraxis' (default, the point of "
             "this file): weight axis p's per-pixel squared eps-residual by the sum-to-1, DETACHED "
             "cross-attention map of that axis's CONTENT phrases (4th field of --valence_axes), averaged over "
             "the pos & neg forwards so the gap g_p = E_neg - E_pos stays a clean paired difference (both "
             "classes weighted by the SAME mask). Axes with no content phrases fall back to uniform. Because "
             "the map SUMS TO 1, the weighted error keeps the same scale as a uniform mean, so per-axis "
             "commensurability (and --valence_axis_scale) is unaffected. 'uniform': whole-image mean for "
             "every axis -- identical to the _multiprompt_valence.py (no-attmap) file, and it does NOT install "
             "the capturing attention processor, so the scoring UNet keeps its fast SDPA path. NOTE this is a "
             "DIFFERENT construction from the gender file's single COMMON map (which smeared 4 supports toward "
             "uniform); here each axis keeps its OWN map and is pooled only AFTER the spatial reduction, so no "
             "averaging-across-axes smearing occurs. RISK: whether SD cross-attention actually localises "
             "abstract nouns ('atmosphere','actions') is EMPIRICAL -- validate with valence_exp before "
             "trusting it; a diffuse map degrades gracefully to ~uniform, a WRONG map can hurt.",
    )
    parser.add_argument(
        '--valence_pooling',
        type=str,
        default="gap_mean",
        choices=["gap_mean", "prob_mean"],
        help="how the P axes are combined. 'gap_mean' (default): pool BEFORE the softmax -- "
             "score = sum_p w_p * (g_p - m_p)/s_p, one softmax at the end. This is simultaneously the "
             "'average the errors', 'average the logits' and 'average the log-probs' options, which are "
             "PROVABLY THE SAME ESTIMATOR (logit_c = -E_c/tau is affine in E with a shared tau, so the "
             "mean commutes with it; the log-prob variant differs only by a class-independent term that "
             "softmax/CE ignore). 'prob_mean': per-axis softmax, then AVERAGE THE PROBABILITIES -- the "
             "one genuinely different pooling, and the one that breaks (it is a mixture, so one confident "
             "axis cannot be outvoted and saturated axes contribute no gradient; and under a small tau "
             "each axis becomes a hard 0/1 vote, so the mean collapses onto the grid {0,.25,.5,.75,1} and "
             "the 50/50 target ranking degenerates into argsort tie-breaking by ARRIVAL ORDER). Provided "
             "for the ablation only.",
    )
    parser.add_argument(
        '--valence_tau',
        type=float,
        default=1.0,
        help="temperature of the pooled valence logits: logit_pos - logit_neg = score / valence_tau. "
             "SEPARATE from --tau (which stays 1e-4 for the gender scorer) because the pooled score is "
             "STANDARDISED (O(1)) once --valence_axis_scale is calibrated, so a 1e-4 temperature would "
             "saturate the softmax into a binary loss.",
    )
    parser.add_argument(
        '--valence_axis_scale',
        type=float,
        nargs='+',
        default=None,
        help="per-axis gap scale s_p (one float per axis, same order as --valence_axes). Divides each "
             "axis's gap so no axis dominates the pooled score. MEASURE IT OFFLINE with "
             "valence_exp/valence_separation.py (it prints the constants) -- these are FROZEN constants, "
             "never live batch statistics: an EMA of the current model's own batch would chase the model "
             "and erase the very population shift that IS the bias signal. Default None = all 1.0 "
             "(UNCALIBRATED -- the run is tagged _uncal, and the largest-scale axis will own the decision).",
    )
    parser.add_argument(
        '--valence_axis_center',
        type=float,
        nargs='+',
        default=None,
        help="per-axis gap offset m_p subtracted before scaling (same order as --valence_axes). Frozen "
             "offline constant, from valence_exp/valence_separation.py. Default None = all 0.0. Note this "
             "only shifts the decision boundary; it does NOT change the ranking, so the 50/50 split is "
             "unaffected by it -- it matters for the reported pred/prob metrics.",
    )
    parser.add_argument(
        '--valence_axis_weight',
        type=float,
        nargs='+',
        default=None,
        help="per-axis pooling weight w_p (same order as --valence_axes); renormalised to sum to 1. Set it "
             "PROPORTIONAL TO EACH AXIS'S MEASURED DISCRIMINABILITY (d' or AUC-0.5), and set a dead axis to "
             "0. WHY NOT just 1/P over the scaled gaps: dividing by s_p equalises each axis's VARIANCE, not "
             "its SIGNAL, so an axis that is pure Monte-Carlo noise gets inflated to unit variance and then "
             "injected at full 1/P weight -- actively destroying the pooled signal. Default None = uniform "
             "1/P (only safe once every axis has passed the offline AUC floor).",
    )
    parser.add_argument(
        '--valence_axis_eps',
        type=str,
        default="independent",
        choices=["independent", "shared"],
        help="eps draw across axes. 'independent' (default): a fresh eps per (axis, timestep), SHARED by "
             "that axis's positive/negative prompts. Sharing WITHIN a pair is what makes the gap a "
             "low-variance paired difference and is mandatory. Sharing ACROSS axes ('shared', the naive "
             "port of the old 2-prompt code) correlates the P gap estimates through a single noise draw, "
             "so averaging them reduces NO Monte-Carlo noise and the entire point of pooling is lost.",
    )
    parser.add_argument(
        '--valence_face_gate',
        type=str,
        default="errfd",
        choices=["none", "errfd"],
        help="whether the residual-error FACE detector gates the training loss (DAL participation + the SCR "
             "no-face preservation bypass). 'errfd' (DEFAULT, chosen for this file): KEEP the gender file's "
             "face gating -- (1) a sample only joins the fair/DAL loss if it has a face, and (2) a no-face "
             "sample keeps SCR weight 1 (bypasses flip/uncertain damping). This is the deliberate choice to "
             "change only the SPATIAL SCR gate (--factor2, mechanism 4) while leaving the face-based "
             "participation gate intact: the class axes include face-bound content (expr), so scoring valence "
             "on a faceless generation is partly ill-defined. Uses residual_face_indicators; get_valence "
             "scatters non-face rows to fill_value so gathered shapes stay uniform (no NCCL shape-split), and "
             "get_valence pins fp32/int64 dtypes in both branches (no dtype-split deadlock). 'none': the "
             "OPPOSITE ablation -- every image participates (a faceless grim alleyway HAS a valence), which "
             "aligns training with the whole-image CLIP eval head and drops the detector dependency, at the "
             "cost of feeding face-axis noise on faceless images (mitigated by multi-axis pooling). Run both "
             "and compare on eval; measure the faceless fraction first.",
    )
    parser.add_argument(
        '--eval_err_valence',
        type=str,
        default="generated",
        choices=["none", "generated", "both"],
        help="EVAL-TIME readout of the TRAINING scorer (the diffusion residual-error valence head, "
             "get_valence / _valence_axis_errors) on the validation generations, scored over ALL IMAGES -- "
             "the errFD face gate is NOT applied here, unlike the training loop (see --valence_face_gate, "
             "default errfd, which restricts the train_* valence metrics to face-detected rows). Keys are "
             "logged under eval_<name>_errVal/*. 'generated' (DEFAULT): score the finetuned generations only. "
             "'both': ALSO score the frozen/original generations -- these are constant in expectation across "
             "training, so errVal/gap_ori is the MONTE-CARLO NOISE FLOOR that makes the finetuned number "
             "interpretable (the scorer draws fresh eps every call, so it is stochastic even on fixed images). "
             "'none': skip entirely. COST: the scorer runs 2*P forwards at batch n*K "
             "(--residual_num_timesteps, default 15) per chunk, which roughly DOUBLES eval wall-clock for "
             "'generated' and TRIPLES it for 'both'. "
             "READ THE CAVEAT: this is the training objective re-evaluated, i.e. a number the optimiser is "
             "directly pushing on. It is a DIAGNOSTIC (reward-hacking / per-axis health / calibration drift), "
             "NEVER a debias result -- every bias claim must come from the independent CLIP head "
             "(eval_<name>_valence_gap; see the get_valence_test docstring).",
    )
    parser.add_argument(
        '--valence_allow_uncalibrated',
        action="store_true",
        default=False,
        help="permit training without --valence_axis_scale / --valence_axis_weight. Without calibration "
             "the pooled score is dominated by whichever axis has the largest raw gap, and a dead axis is "
             "weighted like a good one -- the exact configuration that produced this project's previous "
             "dead losses. Required to proceed uncalibrated; the run folder is tagged _uncal.",
    )
    parser.add_argument(
        '--valence_eval_pos_prompts',
        type=str,
        nargs='+',
        default=[
            "a photo of a happy positive person in a cheerful bright scene",
            "a positive, uplifting, cheerful photo",
        ],
        help="text prompts for the POSITIVE class of the zero-shot CLIP valence head used at EVALUATION. "
             "They are prompt-ensembled in EMBEDDING space (mean of the unit text vectors, renormalised) -- "
             "the canonical CLIP zero-shot recipe, and 'average within a class' done where it is correct. "
             "Deliberately WORDED DIFFERENTLY from --valence_axes: the eval metric must be independent of "
             "the training scorer, not a paraphrase of it.",
    )
    parser.add_argument(
        '--valence_eval_neg_prompts',
        type=str,
        nargs='+',
        default=[
            "a photo of a sad negative person in a gloomy dark scene",
            "a negative, depressing, gloomy photo",
        ],
        help="text prompts for the NEGATIVE class of the zero-shot CLIP valence head (see "
             "--valence_eval_pos_prompts).",
    )

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
        default=8
        )
    parser.add_argument(
        '--val_GPU_batch_size',
        help="validation batch size in every GPU",
        type=int,
        default=8
        )
    parser.add_argument(
        '--val_images_per_prompt_total',
        help=(
            "total number of images to KEEP per prompt during validation, across all GPUs. "
            "Each GPU still generates val_images_per_prompt_GPU images, and all-gather produces "
            "val_images_per_prompt_GPU * num_processes images; the gathered set is then truncated "
            "to the first `val_images_per_prompt_total` for all metrics and image grids. "
            "0 (default) means keep everything (val_images_per_prompt_GPU * num_processes). "
            "Set to 60 to evaluate on exactly 60 images per occupation even with 8 GPUs (8*8=64 -> 60)."
        ),
        type=int,
        default=0
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
    # NOTE: the mnet-based training gender classifier (--classifier_weight_path) has been removed;
    # the gender signal is now produced by a prompt-conditioned diffusion residual-error scorer.
    parser.add_argument(
        '--tau',
        default=1e-4,
        type=float,
        help="UNUSED FOR VALENCE (kept so existing yaml configs still load). It was the temperature of the "
             "woman/man residual-error gender logits. The valence scorer has its own --valence_tau, which is "
             "O(1) rather than 1e-4 because the pooled score is STANDARDISED -- feeding a standardised score "
             "through a 1e-4 temperature would saturate the softmax into a binary loss.",
    )
    parser.add_argument(
        '--residual_num_timesteps',
        default=15,
        type=int,
        help="number of timesteps used by the residual-error gender scorer",
    )
    parser.add_argument(
        '--residual_t_min',
        default=400,
        type=int,
        help="minimum timestep (inclusive) for the residual-error gender scorer",
    )
    parser.add_argument(
        '--residual_t_max',
        default=800,
        type=int,
        help="maximum timestep (inclusive) for the residual-error gender scorer",
    )
    parser.add_argument(
        '--scr_num_timesteps',
        default=15,
        type=int,
        help="number of timesteps used by the h-space SCR image loss (independent of the gender/realism "
             "scorer's --residual_num_timesteps)",
    )
    parser.add_argument(
        '--scr_t_min',
        default=400,
        type=int,
        help="minimum timestep (inclusive) for the h-space SCR image loss. SEPARATE from the woman/man "
             "class-error + SRR-realism scorer (which uses --residual_t_min/max, default 400-800): the SCR "
             "mid_block (h-space) MSE is evaluated over this lower-noise range so it targets image structure "
             "rather than the coarse gender signal.",
    )
    parser.add_argument(
        '--scr_t_max',
        default=800,
        type=int,
        help="maximum timestep (inclusive) for the h-space SCR image loss (see --scr_t_min).",
    )
    parser.add_argument(
        '--residual_woman_prompt',
        default="a photo of a woman",
        type=str,
        help="text prompt for the woman class (index 0) in the residual-error gender scorer",
    )
    parser.add_argument(
        '--residual_man_prompt',
        default="a photo of a man",
        type=str,
        help="text prompt for the man class (index 1) in the residual-error gender scorer",
    )
    parser.add_argument(
        '--residual_woman_word',
        default="woman",
        type=str,
        help="the woman-class word whose cross-attention map is used for spatial weighting "
             "(its token index is located in --residual_woman_prompt via the tokenizer, not hard-coded)",
    )
    parser.add_argument(
        '--residual_man_word',
        default="man",
        type=str,
        help="the man-class word whose cross-attention map is used for spatial weighting "
             "(its token index is located in --residual_man_prompt via the tokenizer, not hard-coded)",
    )
    parser.add_argument(
        '--face_residual_face_prompt',
        default="a photo of a face",
        type=str,
        help="NODETECTOR: face-class prompt of the residual-error face/no-face classifier that "
             "replaces the insightface detector in the training branch points",
    )
    parser.add_argument(
        '--face_residual_nonface_prompt',
        default="a faceless photo",
        type=str,
        help="NODETECTOR: no-face-class prompt (best partner of 'a photo of a face' in the "
             "face_error_exp/exp100 ablation)",
    )
    parser.add_argument(
        '--face_residual_num_timesteps',
        default=8,
        type=int,
        help="NODETECTOR: number of timesteps (K) of the face/no-face residual scorer. K=8 kept "
             "the offline accuracy of K=15 (88/100) for this prompt pair; K<8 degrades",
    )
    parser.add_argument(
        '--face_residual_t_min',
        default=50,
        type=int,
        help="NODETECTOR: min timestep (inclusive) of the face/no-face residual scorer "
             "(t in [50,950] was the best range for this prompt pair)",
    )
    parser.add_argument(
        '--face_residual_t_max',
        default=950,
        type=int,
        help="NODETECTOR: max timestep (inclusive) of the face/no-face residual scorer",
    )
    parser.add_argument(
        '--skip_denoise_frac',
        default=0.0,
        type=float,
        help="fraction of the denoising trajectory to skip when generating images (truncated denoising). "
             "0.0 = full sequential denoising, identical to the non-truncated SRR_person behavior. "
             "If >0, only the first round((1-frac)*num_denoising_steps) scheduler steps run "
             "sequentially; at the last of those steps we jump straight to a predicted clean "
             "latent x0 via the closed-form epsilon->x0 formula "
             "x0 = (z_t - sqrt(1-abar_t)*eps) / sqrt(abar_t) (no extra UNet call). "
             "Applied identically to every TRAINING image-generation pass (current-model, "
             "original/frozen model, and the gradient pass) so reference and trained images stay on "
             "equal footing; EVALUATION always passes 0.0 (full denoising) for comparable metrics. "
             "NOTE: with frac>0 the z0 fed to the SRR realism residual, the SCR gender logits and "
             "CLIP/DINO is a truncated (blurrier) x0 -- see the head-of-file note on this interaction.",
    )
    parser.add_argument(
        '--test_classifier_weight_path',
        default="../data/5-trained-test-classifiers/CelebA-MobileNetLarge-Gender-09191318/epoch=19-step=25320_MobileNetLarge.pt",
        help="separately-trained gender classifier (2-way output, gender only) used only for evaluation "
             "(evaluate_process); the training loss now uses the residual-error scorer, not this classifier",
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

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        args_dict = vars(args)
        # backward-compat: the old CLIP+DINO image-loss weight key maps onto the SCR loss weight,
        # so existing configs (e.g. debias-text-encoder.yaml with weight_loss_img) run unchanged.
        _key_aliases = {"weight_loss_img": "weight_loss_scr"}
        for key, value in config_data.items():
            key = _key_aliases.get(key, key)
            args_dict[key] = type(args_dict[key])(value)
        args = argparse.Namespace(**args_dict)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

logger = get_logger(__name__)

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
    _model_tag = f"{'TE' if args.train_text_encoder else ''}{'UNet' if args.train_unet else ''}"
    # short SRR-prompt slug (last word of --srr_prompt) so person-vs-face SRR runs are distinguishable
    # by folder name, not just timestamp: "a photo of a realistic person" -> srr-person, ...face -> srr-face.
    _srr_tag = args.srr_prompt.strip().split()[-1] if args.srr_prompt.strip() else "none"

    # ---------------------------------------------------------------- parse & validate the valence axes
    # Each --valence_axes entry is  'name::POSITIVE prompt::NEGATIVE prompt[::CONTENT phrases]'.
    # Class order [neg=0, pos=1]. The optional 4th field is a comma-separated list of CONTENT PHRASES whose
    # cross-attention defines THIS axis's spatial region (AXISATTN variant). Each phrase must appear VERBATIM
    # in BOTH the positive and the negative prompt (it is the shared, non-polarity part of the pair), and is
    # located independently in each prompt's own token stream (the polarity word can tokenize to a different
    # length, shifting the content tokens). An EMPTY / absent 4th field => this axis is scored with a UNIFORM
    # whole-image mean (correct for a global axis like "overall tone", which has no localisable content noun).
    valence_axes = []
    for spec in args.valence_axes:
        parts = [p.strip() for p in spec.split("::")]
        if len(parts) not in (3, 4) or not all(parts[:3]):
            raise ValueError(
                "--valence_axes entry must be 'name::POSITIVE prompt::NEGATIVE prompt[::CONTENT phrases]', "
                f"got {spec!r}"
            )
        content = []
        if len(parts) == 4 and parts[3]:
            content = [w.strip() for w in parts[3].split(",") if w.strip()]
        valence_axes.append({"name": parts[0], "pos": parts[1], "neg": parts[2], "content": content})
    P_axes = len(valence_axes)
    if P_axes == 0:
        raise ValueError("--valence_axes is empty; at least one contrastive pair is required")

    def _axis_vec(values, default, what):
        """Broadcast a per-axis CLI list to a [P] float32 tensor (frozen, never a batch statistic)."""
        if values is None:
            return torch.full([P_axes], float(default), dtype=torch.float)
        if len(values) != P_axes:
            raise ValueError(
                f"--valence_axis_{what} has {len(values)} values but there are {P_axes} axes "
                f"({[a['name'] for a in valence_axes]})"
            )
        return torch.tensor([float(v) for v in values], dtype=torch.float)

    valence_center = _axis_vec(args.valence_axis_center, 0.0, "center")     # [P] m_p
    valence_scale = _axis_vec(args.valence_axis_scale, 1.0, "scale")        # [P] s_p
    valence_weight = _axis_vec(args.valence_axis_weight, 1.0, "weight")     # [P] w_p (pre-normalisation)
    if (valence_scale <= 0).any():
        raise ValueError(f"--valence_axis_scale must be strictly positive, got {valence_scale.tolist()}")
    if valence_weight.sum() <= 0:
        raise ValueError("--valence_axis_weight must have a positive sum (at least one live axis)")
    valence_weight = valence_weight / valence_weight.sum()                  # sum-to-1

    # UNCALIBRATED = the exact configuration that produced this project's previous DEAD losses: the
    # largest-raw-gap axis owns the pooled decision, and a pure-noise axis is weighted like a good one.
    _valence_uncal = (args.valence_axis_scale is None) or (args.valence_axis_weight is None)
    if _valence_uncal and not args.valence_allow_uncalibrated:
        raise ValueError(
            "REFUSING TO TRAIN UNCALIBRATED.\n"
            "  --valence_axis_scale and --valence_axis_weight are not set, so every axis gets scale 1.0 and\n"
            "  weight 1/P. The per-axis gaps do NOT share a scale (their baselines cancel, their SCALES do\n"
            "  not), so the axis with the largest dynamic range will own the pooled decision and you will\n"
            "  ship a 1-axis debiaser believing you shipped a 4-axis one. A DEAD axis would be injected at\n"
            "  full weight.\n"
            "  FIX: run  python valence_exp/valence_separation.py  (it measures per-axis ROC-AUC and prints\n"
            "  the --valence_axis_scale / --valence_axis_weight constants to paste in).\n"
            "  To proceed anyway (the folder will be tagged _uncal): --valence_allow_uncalibrated"
        )
    if _valence_uncal:
        logger.warning(
            "VALENCE: running UNCALIBRATED (--valence_allow_uncalibrated). Pooled score is dominated by "
            "whichever axis has the largest raw gap; a dead axis carries full weight. Run "
            "valence_exp/valence_separation.py."
        )

    _val_tag = (
        f"_val-{P_axes}ax-{args.valence_pooling}{'-uncal' if _valence_uncal else ''}"
        # AXISATTN: always tag the spatial reduction so a run's folder says whether per-axis attention
        # weighting was on. _vAttn-peraxis = content-token weighted, _vAttn-uniform = whole-image mean.
        f"_vAttn-{args.valence_axis_attn}"
    )
    folder_name = (
        f"BS-{args.train_images_per_prompt_GPU*accelerator.num_processes}"
        f"_{_model_tag}"
        # VALENCE: the multi-prompt class is ALWAYS tagged (axis count + pooling + calibration state), so a
        # run's own folder says what class it optimised and whether the axes were calibrated.
        f"{_val_tag}"
        f"_vTau-{args.valence_tau:g}"
        f"_vEps-{args.valence_axis_eps}"
        f"{'_vFace-errfd' if args.valence_face_gate == 'errfd' else ''}"
        f"_resT-{args.residual_num_timesteps}-{args.residual_t_min}-{args.residual_t_max}"
        f"_scrT-{args.scr_num_timesteps}-{args.scr_t_min}-{args.scr_t_max}"
        f"_wSCR-{args.weight_loss_scr}-{args.factor1}-{args.factor2}"
        f"_wSRR-{args.weight_loss_face}"
        f"_srr-{_srr_tag}"
        # errFD is only tagged when the face gate is actually ON (--valence_face_gate errfd); for valence
        # the detector does not run in the training path at all, so tagging it would be a lie.
        f"{f'_errFD-{args.face_residual_num_timesteps}-{args.face_residual_t_min}-{args.face_residual_t_max}' if args.valence_face_gate == 'errfd' else ''}"
        f"{('_skipFrac-'+format(args.skip_denoise_frac, 'g')) if args.skip_denoise_frac>0 else ''}"
        f"_Th-{args.uncertainty_threshold}"
        f"_loraR-{args.rank}_lr-{args.learning_rate}"
        f"_{timestring}"
    )
    
    args.imgs_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "imgs")
    args.ckpts_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "ckpts")
    # NODETECTOR: where per-image NO-FACE (errFD-negative) images are dumped for inspection.
    args.noface_imgs_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "noface_imgs")

    if accelerator.is_main_process:
        os.makedirs(args.imgs_save_dir, exist_ok=True)
        os.makedirs(args.ckpts_save_dir, exist_ok=True)
        if args.save_noface_imgs:
            os.makedirs(os.path.join(args.noface_imgs_save_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(args.noface_imgs_save_dir, "eval"), exist_ok=True)
        accelerator.init_trackers(
            args.proj_name,
            config=vars(args),
            init_kwargs = {
                "wandb": {
                    "name": folder_name,
                    "dir": args.output_dir,
                    "save_code": True,
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
    

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
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

    #######################################################
    # Residual-error gender scorer setup (replaces the removed mnet gender classifier).
    # The scoring diffusion model is the FROZEN original SD (never the trainable copy):
    #   - if the unet is being trained, score with the frozen eval_unet; else the (frozen) unet.
    #   - if the text encoder is being trained, condition with the frozen eval_text_encoder; else text_encoder.
    # Gradient must flow through the scorer INPUT (loss -> residual -> zt -> z0 -> trainable model),
    # while the scorer's own parameters stay frozen.
    scoring_unet = eval_unet if args.train_unet else unet
    scoring_text_encoder = eval_text_encoder if args.train_text_encoder else text_encoder
    scoring_unet.requires_grad_(False)
    scoring_text_encoder.requires_grad_(False)
    scoring_unet.enable_gradient_checkpointing()
    # SD UNet has no BatchNorm/dropout, so train() does not change outputs; it only activates
    # gradient checkpointing (which needs training mode) to keep the residual backward memory bounded.
    scoring_unet.train()

    # The residual scorer assumes an epsilon-prediction model; v-prediction must not be compared to eps.
    assert noise_scheduler.config.prediction_type == "epsilon", (
        f"residual gender scorer assumes epsilon-prediction, but the scheduler uses "
        f"'{noise_scheduler.config.prediction_type}'."
    )

    # Cache the fixed woman/man text embeddings (class order: [woman=0, man=1]).
    def _encode_scoring_prompt(prompt):
        tok = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = scoring_text_encoder(
                tok["input_ids"].to(accelerator.device),
                tok["attention_mask"].to(accelerator.device),
            )[0]
        return emb.to(weight_dtype)
    residual_woman_embeds = _encode_scoring_prompt(args.residual_woman_prompt)  # [1, L, D]
    residual_man_embeds = _encode_scoring_prompt(args.residual_man_prompt)      # [1, L, D]
    residual_realistic_embeds = _encode_scoring_prompt(args.srr_prompt)         # [1, L, D], SRR realism prompt

    # VALENCE: cache the 2P class-prompt embeddings, one POSITIVE and one NEGATIVE per axis. Encoded with
    # the SAME (with-attention-mask) convention as the woman/man scorer they replace -- these prompts live
    # in the same scorer, so they must share its convention. (The face/faceless pair deliberately uses the
    # no-mask convention instead; see _encode_scoring_prompt_nomask below. Mixing the two silently flipped
    # that classifier's margins once, so the two conventions are kept explicitly separate.)
    for _ax in valence_axes:
        _ax["pos_embeds"] = _encode_scoring_prompt(_ax["pos"])                 # [1, L, D]
        _ax["neg_embeds"] = _encode_scoring_prompt(_ax["neg"])                 # [1, L, D]
    valence_center = valence_center.to(accelerator.device)                     # [P] frozen
    valence_scale = valence_scale.to(accelerator.device)                       # [P] frozen
    valence_weight = valence_weight.to(accelerator.device)                     # [P] frozen, sums to 1
    logger.info(
        "VALENCE class = %d contrastive pairs (order [neg=0, pos=1]):\n%s",
        P_axes,
        "\n".join(
            f"  [{i}] {a['name']:>6s}  w={valence_weight[i].item():.3f} s={valence_scale[i].item():.4g} "
            f"m={valence_center[i].item():.4g}\n"
            f"          pos: {a['pos']}\n"
            f"          neg: {a['neg']}"
            for i, a in enumerate(valence_axes)
        ),
    )
    # NODETECTOR: frozen-TE embeddings of the face/no-face classifier prompts (replaces the
    # insightface detector in the training branch points; see residual_face_indicators).
    # IMPORTANT: encoded WITHOUT attention_mask -- the standard SD text-encoding convention.
    # _encode_scoring_prompt passes the padding attention_mask into CLIP, which massively changes
    # the padded-position embeddings (rel. diff ~1.25). The woman/man/SRR scorers were built on
    # that masked convention, but the face/faceless pair was ablated (face_error_exp/exp100) with
    # the standard no-mask encoding; feeding masked embeddings flips its small margins and makes
    # the classifier call (nearly) everything "faceless". Verified by exact replay of in-situ
    # zt/eps tensors: masked embeds reproduce the 0/24 failure bit-for-bit, no-mask embeds restore
    # the ablation behaviour on the same tensors.
    def _encode_scoring_prompt_nomask(prompt):
        tok_out = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = scoring_text_encoder(tok_out["input_ids"].to(accelerator.device))[0]
        return emb.to(weight_dtype)
    residual_face_embeds = _encode_scoring_prompt_nomask(args.face_residual_face_prompt)        # [1, L, D]
    residual_faceless_embeds = _encode_scoring_prompt_nomask(args.face_residual_nonface_prompt) # [1, L, D]

    #######################################################
    # Cross-attention capture, for spatial weighting of the residual scorer.
    # A custom processor is installed ONLY on the scoring UNet's cross-attention (attn2) layers.
    # It always uses the explicit get_attention_scores path (so it stays deterministic under the
    # scoring UNet's gradient checkpointing), and — when enabled — stores the DETACHED attention map
    # of the current class token(s), reduced to [B*heads, query_hw]. Self-attention (attn1) is untouched.
    class _AttnCaptureCtx:
        def __init__(self):
            self.enabled = False
            self.token_idxs = None
            self.store = []
    attn_capture_ctx = _AttnCaptureCtx()

    class CrossAttnCaptureProcessor:
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
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
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
            attention_probs = attn.get_attention_scores(query, key, attention_mask)  # [B*heads, query_hw, key_tokens]
            if self.ctx.enabled and self.ctx.token_idxs is not None:
                # keep only the class token column(s), averaged over subtokens -> [B*heads, query_hw].
                # Wrapped in no_grad so this capture side-effect adds nothing to the autograd tape; this
                # keeps the forward/recompute saved-tensor counts identical under gradient checkpointing.
                with torch.no_grad():
                    col = attention_probs[..., self.ctx.token_idxs].mean(dim=-1)
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

    # Locate the woman/man class-token indices from the actual tokenizer output (no hard-coded positions).
    # If a class word splits into multiple subtokens, all of them are returned and later averaged.
    def _find_word_token_indices(prompt, word):
        prompt_ids = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True
        ).input_ids
        word_ids = tokenizer(word, add_special_tokens=False).input_ids
        L = len(word_ids)
        for i in range(len(prompt_ids) - L + 1):
            if prompt_ids[i:i + L] == word_ids:
                return list(range(i, i + L))
        raise ValueError(f"could not locate '{word}' tokens {word_ids} inside prompt '{prompt}' -> {prompt_ids}")
    residual_woman_token_idxs = _find_word_token_indices(args.residual_woman_prompt, args.residual_woman_word)
    residual_man_token_idxs = _find_word_token_indices(args.residual_man_prompt, args.residual_man_word)

    # AXISATTN: per-axis CONTENT-token indices, located INDEPENDENTLY in the pos and neg prompt token
    # streams (the polarity word "positive"/"negative" can tokenize to a different length, shifting the
    # shared content tokens). A phrase list is UNIONED across phrases -- this is how NON-CONTIGUOUS content
    # ("clothing" ... "actions", separated by "and") is expressed, which the single contiguous-run gender
    # helper could not. None => this axis has no region and is scored with a uniform whole-image mean.
    def _find_content_token_indices(prompt, phrases):
        idxs = []
        for ph in phrases:
            idxs += _find_word_token_indices(prompt, ph)      # each phrase a contiguous run; raises if absent
        return sorted(set(idxs))
    _valence_use_attn = (args.valence_axis_attn == "peraxis")
    for _ax in valence_axes:
        if _valence_use_attn and _ax["content"]:
            _ax["pos_content_idxs"] = _find_content_token_indices(_ax["pos"], _ax["content"])
            _ax["neg_content_idxs"] = _find_content_token_indices(_ax["neg"], _ax["content"])
        else:
            _ax["pos_content_idxs"] = None
            _ax["neg_content_idxs"] = None

    # AXISATTN: install the capturing attention processor ONLY when at least one axis uses per-axis attmap.
    #  - peraxis + >=1 content axis: install CrossAttnCaptureProcessor on the scoring UNet's attn2 layers so
    #    _valence_axis_errors can read the content-token cross-attention. This forces the explicit
    #    get_attention_scores + bmm path (no SDPA) on the scoring UNet -- the real, unavoidable cost of
    #    spatial localisation (the capture side-effect itself is under no_grad and free).
    #  - uniform (or no content axes): do NOT install it, so diffusers keeps its fast SDPA attention. That
    #    path is byte-identical to the _multiprompt_valence.py (no-attmap) file.
    # WHY PER-AXIS MAPS ARE VALID where a single COMMON map was rejected: each axis keeps its OWN map and is
    # pooled only AFTER the spatial reduction, so nothing averages 4 different supports toward uniform; the
    # sum-to-1 map preserves each axis's error scale (so --valence_axis_scale still means what it did); and
    # non-contiguous content is handled by unioning per-phrase token runs above. The tone axis (no content)
    # stays uniform by construction -- a global axis has no region.
    _valence_attn_installed = _valence_use_attn and any(a["pos_content_idxs"] is not None for a in valence_axes)
    if _valence_attn_installed:
        _scoring_attn_procs = dict(scoring_unet.attn_processors)
        for _name in list(_scoring_attn_procs.keys()):
            if _name.endswith("attn2.processor"):
                _scoring_attn_procs[_name] = CrossAttnCaptureProcessor(attn_capture_ctx)
        scoring_unet.set_attn_processor(_scoring_attn_procs)
        logger.info(
            "AXISATTN: per-axis cross-attention weighting ON. attmap axes=%s ; uniform axes=%s.",
            [a["name"] for a in valence_axes if a["pos_content_idxs"] is not None],
            [a["name"] for a in valence_axes if a["pos_content_idxs"] is None],
        )
    else:
        logger.info(
            "AXISATTN: uniform whole-image reduction for every axis (capturing processor NOT installed; "
            "scoring UNet keeps SDPA)."
        )
    if args.factor2 != 1.0:
        logger.warning(
            "--factor2 %s is IGNORED for valence (it was the spatial SCR gate inside the class "
            "cross-attention region; that region does not exist for a scene-level class). factor1=%s is "
            "the single release damping.", args.factor2, args.factor1,
        )

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

    
    prompts_val = [prompt.format(occupation=occupation) for prompt in experiment_data["prompt_templates_test"] for occupation in experiment_data["occupations_test_set"]]
    
    
    #######################################################
    # set up things needed for finetuning
    # (the mnet gender_classifier used to be initialized here; it is now replaced by the
    #  prompt-conditioned diffusion residual-error scorer set up further below.)

    # set up face_recognition and face_app on all devices
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
    

    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2.to(accelerator.device, dtype=weight_dtype)
    dinov2.requires_grad_(False)
    dinov2_img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
    dinov2_img_std = torch.tensor([0.229, 0.224, 0.225]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)

    #######################################################
    # evaluation-only feature extractors (deliberately different from the training ones above):
    #   - CLIP-ViT-bigG-14  -> CLIP-T (image vs prompt) and CLIP-I (image vs original-SD image)
    #   - DINOv2 vit-g/14   -> DINO   (image vs original-SD image)
    # These are only used inside evaluate_process, never in the training loss.
    # CLIP-ViT-bigG-14 for CLIP-T / CLIP-I, loaded via open_clip on the MAIN PROCESS ONLY
    # (matches file B, chekc_SCRclip_face_grad.py). Non-main ranks all-gather their images to
    # rank 0, which computes the CLIP metrics itself in evaluate_process. open_clip's own
    # preprocess (resize + center-crop + normalize, via PIL) replaces the previous HF CLIPModel
    # tensor-space pipeline, so CLIP-I/CLIP-T now match file B numerically.
    eval_clip_model_name = "ViT-bigG-14"
    eval_clip_pretrained = "laion2b_s39b_b160k"
    eval_clip_model = None
    eval_clip_preprocess = None
    eval_clip_tokenizer = None
    # DINOv2 vit-g/14 for DINO is now ALSO loaded on the MAIN PROCESS ONLY (matches file B):
    # rank 0 computes DINO over the gathered images, so there is no per-rank eval-DINOv2 pass
    # and no extra all-gather of DINO sims.
    eval_dinov2 = None
    if accelerator.is_main_process:
        eval_clip_precision = "fp32"
        if weight_dtype == torch.float16:
            eval_clip_precision = "fp16"
        elif weight_dtype == torch.bfloat16:
            eval_clip_precision = "bf16"
        eval_clip_model, _, eval_clip_preprocess = open_clip.create_model_and_transforms(
            eval_clip_model_name,
            pretrained=eval_clip_pretrained,
            precision=eval_clip_precision,
            device=accelerator.device,
        )
        eval_clip_model.eval().requires_grad_(False)
        eval_clip_tokenizer = open_clip.get_tokenizer(eval_clip_model_name)

        # eval DINOv2 loaded here (main process only). Do NOT wrap in accelerator.main_process_first():
        # only rank 0 enters this branch, and that context's trailing barrier() would hang the other
        # ranks forever (they never hit the matching barrier). Load directly. Reuses the ImageNet
        # normalization dinov2_img_mean/std (identical values to file B's dinov2_eval_img_mean/std).
        eval_dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        eval_dinov2.to(accelerator.device, dtype=weight_dtype)
        eval_dinov2.requires_grad_(False)
        eval_dinov2.eval()

    # separately-trained gender *test* classifier: MobileNetV3-Large with a 2-way (gender-only) head,
    # used only during evaluation to decide the gender class (mirrors eval-generated-images.py).
    # Same face-crop input as training; only the weights and the output dimension (2 vs 80) differ.
    gender_classifier_test = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT, width_mult=1.0, reduced_tail=False, dilated=False)
    gender_classifier_test._modules['classifier'][3] = nn.Linear(1280, 2, bias=True)
    gender_classifier_test.load_state_dict(torch.load(args.test_classifier_weight_path))
    # keep the MobileNet test classifier in float32 (do NOT cast to weight_dtype/fp16) so the
    # gender forward matches the sibling file B (chekc_SCRclip_face_grad.py): the input is cast
    # to .float() and the gathered probs are pinned to fp32 in get_face_gender_test.
    gender_classifier_test.to(accelerator.device)
    gender_classifier_test.requires_grad_(False)
    gender_classifier_test.eval()

    CE_loss = nn.CrossEntropyLoss(reduction="none")

    # build opensphere model
    sys.path.append(Path(__file__).parent.parent.__str__())
    sys.path.append(Path(__file__).parent.parent.joinpath("opensphere").__str__())
    from opensphere.builder import build_from_cfg
    from opensphere.utils import fill_config

    with open(args.opensphere_config, 'r') as f:
        opensphere_config = yaml.load(f, yaml.SafeLoader)
    opensphere_config['data'] = fill_config(opensphere_config['data'])
    face_feats_net = build_from_cfg(
        opensphere_config['model']['backbone']['net'],
        'model.backbone',
    )
    face_feats_net = nn.DataParallel(face_feats_net)
    face_feats_net.load_state_dict(torch.load(args.opensphere_model_path))
    face_feats_net = face_feats_net.module
    face_feats_net.to(accelerator.device)
    face_feats_net.requires_grad_(False)
    face_feats_net.to(weight_dtype)
    face_feats_net.eval()
    
    face_feats_model = FaceFeatsModel(args.face_feats_path)
    face_feats_model.to(weight_dtype_high_precision)
    face_feats_model.to(accelerator.device)
    face_feats_model.eval()

    #######################################################
    
    @torch.no_grad()
    def generate_image_no_gradient(prompt, noises, num_denoising_steps, which_text_encoder, which_unet, return_latents=False, skip_denoise_frac=None):
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
        # Optional truncated denoising: run only the first `n_run` scheduler steps sequentially,
        # then at the last of those jump straight to a predicted clean latent x0 via the
        # closed-form epsilon->x0 formula (reuses the current step's eps, no extra UNet call).
        # skip_denoise_frac == 0 -> n_run == full trajectory -> unchanged behavior.
        # The `skip_denoise_frac` argument overrides args.skip_denoise_frac; evaluation passes 0.0
        # so eval/validation metrics are always computed on fully-denoised images (comparable to baselines).
        skip_frac = float(args.skip_denoise_frac if skip_denoise_frac is None else skip_denoise_frac)
        n_total = len(noise_scheduler.timesteps)
        n_run = max(1, int(round(n_total * (1.0 - skip_frac)))) if skip_frac > 0.0 else n_total
        latents = noises
        for i, t in enumerate(noise_scheduler.timesteps):

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

            if skip_frac > 0.0 and i == n_run - 1:
                # one-shot jump z_t -> x0:  x0 = (z_t - sqrt(1-abar_t)*eps) / sqrt(abar_t)
                abar_t = noise_scheduler.alphas_cumprod[t].to(device=latents.device, dtype=noises_pred.dtype)
                latents = (latents - (1 - abar_t).sqrt() * noises_pred) / abar_t.sqrt()
                break

            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

        z0 = latents  # clean latent z0 in the scheduler/UNet scale (before VAE-decode rescaling)
        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]

        if return_latents:
            return images, z0
        return images

    def generate_image_w_gradient(prompt, noises, num_denoising_steps, which_text_encoder, which_unet, return_latents=False):
        """
        prompts: str
        noises: [N,4,64,64], N is number images to be generated for the prompt
        """
        # to enable gradient_checkpointing, unet must be set to train()
        unet.train()
        
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

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]).to(weight_dtype)
        
        noise_scheduler.set_timesteps(num_denoising_steps)
        # Optional truncated denoising (see generate_image_no_gradient). When skip_denoise_frac>0
        # only the first `n_run` steps run and we finish with a closed-form eps->x0 jump. The SDS
        # grad_coefs are computed over the executed steps only, so their geometric mean stays 1
        # (the same normalization the full trajectory uses). skip_denoise_frac==0 -> unchanged.
        skip_frac = float(getattr(args, "skip_denoise_frac", 0.0))
        n_total = len(noise_scheduler.timesteps)
        n_run = max(1, int(round(n_total * (1.0 - skip_frac)))) if skip_frac > 0.0 else n_total
        timesteps_run = noise_scheduler.timesteps[:n_run]
        grad_coefs = []
        for i, t in enumerate(timesteps_run):
            grad_coefs.append( noise_scheduler.alphas_cumprod[t].sqrt().item() * (1-noise_scheduler.alphas_cumprod[t]).sqrt().item() / (1-noise_scheduler.alphas[t].item()) )
        grad_coefs = np.array(grad_coefs)
        grad_coefs /= (math.prod(grad_coefs)**(1/len(grad_coefs)))

        latents = noises
        for i, t in enumerate(noise_scheduler.timesteps):

            # scale model input
            latent_model_input = torch.cat([latents.detach().to(weight_dtype)]*2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            noises_pred = which_unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(weight_dtype_high_precision)

            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + args.guidance_scale * (noises_pred_text - noises_pred_uncond)

            hook_fn = make_grad_hook(grad_coefs[i])
            noises_pred.register_hook(hook_fn)

            if skip_frac > 0.0 and i == n_run - 1:
                # one-shot jump z_t -> x0; grad flows through both `latents` (earlier steps) and
                # `noises_pred` (current UNet), mirroring a normal scheduler.step gradient path.
                abar_t = noise_scheduler.alphas_cumprod[t].to(device=latents.device, dtype=noises_pred.dtype)
                latents = (latents - (1 - abar_t).sqrt() * noises_pred) / abar_t.sqrt()
                break

            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

        z0 = latents  # clean latent z0 in the scheduler/UNet scale; NOT detached so grad reaches the trainable model
        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]

        if return_latents:
            return images, z0
        return images


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

    @torch.no_grad()
    def get_clip_feat_eval(images, normalize=True, to_high_precision=True):
        """image embeds from open_clip CLIP-ViT-bigG-14 (MAIN PROCESS ONLY), used for CLIP-T / CLIP-I.
        Matches file B: [-1,1] tensor -> PIL -> open_clip preprocess (resize + center-crop + normalize)
        -> encode_image. Must only be called on accelerator.is_main_process.

        Args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1]
        """
        imgs_01 = (images.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
        to_pil = transforms.ToPILImage()
        pil_images = [to_pil(img) for img in imgs_01]
        clip_dtype = eval_clip_model.visual.conv1.weight.dtype
        pixel_values = torch.stack(
            [eval_clip_preprocess(img.convert("RGB")) for img in pil_images]
        ).to(accelerator.device, dtype=clip_dtype)
        embeds = eval_clip_model.encode_image(pixel_values)

        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds

    @torch.no_grad()
    def get_clip_text_feat_eval(prompts, normalize=True, to_high_precision=True):
        """text embeds from open_clip CLIP-ViT-bigG-14 (MAIN PROCESS ONLY), used for CLIP-T.
        Must only be called on accelerator.is_main_process.

        Args:
            prompts (str or list[str])
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        text_tokens = eval_clip_tokenizer(prompts).to(accelerator.device)
        embeds = eval_clip_model.encode_text(text_tokens)

        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds

    @torch.no_grad()
    def get_dino_feat_eval(images, normalize=True, to_high_precision=True):
        """image embeds from the evaluation DINOv2 model (vit-g/14), used for DINO (MAIN PROCESS ONLY).
        Matches file B's get_dino_eval_feat: normalization only -- the caller resizes with
        transforms.Resize(224) beforehand (see evaluate_process). Reuses dinov2_img_mean/std,
        whose values are identical to file B's dinov2_eval_img_mean/std.

        Args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1], ALREADY resized by the caller
        """
        images_preprocessed = ((images+1)*0.5 - dinov2_img_mean) / dinov2_img_std
        embeds = eval_dinov2(images_preprocessed.to(weight_dtype))

        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds

    @torch.no_grad()
    def get_valence_test(images):
        """ZERO-SHOT CLIP VALENCE HEAD -- the EVALUATION metric. Replaces the CelebA gender classifier.

        Returns (preds [N] int64, probs [N,2] fp32), class order [neg=0, pos=1]. MAIN PROCESS ONLY (the
        open_clip bigG lives on rank 0), so pass it the already-all-gathered images.

        WHY A NEW HEAD AT ALL: the CelebA MobileNet test classifier consumes 224x224 FACE CROPS and emits a
        gender logit pair. It cannot be repointed at a scene-level concept even in principle -- there is no
        face crop that carries "the background has a gloomy atmosphere".

        WHY CLIP: the eval metric must be INDEPENDENT of the training signal (that is the entire reason the
        gender file used a separately-trained CelebA classifier rather than its own scorer). The open_clip
        ViT-bigG-14 is already loaded on rank 0 for CLIP-I/CLIP-T, is NOT part of this file's training loss,
        and scores the FULL image rather than a face chip. Scoring valence with the training scorer instead
        would not be merely "self-referential" -- it would be the training loss re-evaluated, i.e. a number
        the optimiser is directly maximising.

        PROMPT ENSEMBLING is done in EMBEDDING space: mean of the unit text vectors per class, renormalised.
        This is the canonical CLIP zero-shot recipe -- and it is exactly "average within a class", applied at
        the one place where averaging within a class is the right operation.

        NOTE the logit_scale: CLIP's cosine similarities live in a ~[-0.3, 0.4] band, so a softmax over raw
        cosines would be almost uniform and every probs-based metric would be mush. open_clip's learned
        logit_scale (exp ~100) is what makes the two-way softmax meaningful, and it is what the model was
        trained with.
        """
        img_feats = get_clip_feat_eval(images)                                   # [N,D] L2-normalised fp32
        t_pos = get_clip_text_feat_eval(list(args.valence_eval_pos_prompts))     # [n_pos,D] L2-normalised
        t_neg = get_clip_text_feat_eval(list(args.valence_eval_neg_prompts))     # [n_neg,D] L2-normalised
        t_pos = t_pos.mean(dim=0, keepdim=True)
        t_neg = t_neg.mean(dim=0, keepdim=True)
        t_pos = t_pos / t_pos.norm(dim=-1, keepdim=True)                         # renormalise the ensemble
        t_neg = t_neg / t_neg.norm(dim=-1, keepdim=True)
        text_feats = torch.cat([t_neg, t_pos], dim=0)                            # [2,D], rows [neg, pos]

        logit_scale = eval_clip_model.logit_scale.exp().float()
        logits = logit_scale * (img_feats.float() @ text_feats.float().t())       # [N,2]
        probs = torch.softmax(logits, dim=-1).float()                             # [N,2] fp32
        preds = probs.max(dim=-1).indices                                         # [N]   int64
        return preds, probs

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

        if face_indicators_app.logical_not().sum() > 0:
            face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR = get_face_FR(images[face_indicators_app.logical_not()], fill_value=fill_value)

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
            # import pdb; pdb.set_trace()
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
                
    def _valence_axis_errors(z0):
        """Per-axis, per-class residual errors of the FROZEN scoring UNet -- the multi-prompt class scorer.

        z0: [n,4,H,W] clean latent in the scheduler/UNet scale. NOT detached, so gradient flows
            E -> scoring_unet(zt) -> zt -> z0 -> the trainable model.

        Returns E [n,P,2] (fp32), class order [neg=0, pos=1]:
            E[i,p,c] = mean_t  sum_pixels A_p(pixel) * || eps_pred(z_t, t, c_{p,c}) - eps ||^2
        over --residual_num_timesteps timesteps linspaced in [--residual_t_min, --residual_t_max], where A_p
        is axis p's spatial weighting map (sum-to-1 per image).

        SPATIAL REDUCTION (AXISATTN, --valence_axis_attn):
          'peraxis' (default): A_p is axis p's CONTENT-token cross-attention -- captured from the frozen
             scoring UNet for the content phrases (--valence_axes 4th field: facial expressions -> face;
             clothing,actions -> body; background,atmosphere -> scene), averaged over the axis's pos & neg
             forwards, resized to (H,W), summed-to-1 and DETACHED. So each aspect is scored WHERE IT LIVES.
             Axes with no content phrases (e.g. 'tone') fall back to the uniform map. Because A_p SUMS TO 1,
             the weighted error keeps the SAME scale as a uniform mean (the uniform map is just A=1/(H*W),
             which also sums to 1), so per-axis commensurability and --valence_axis_scale are unaffected.
          'uniform': A_p = 1/(H*W) for every axis -- the plain whole-image mean of the _multiprompt_valence.py
             (no-attmap) file. In this mode the capturing processor is not installed and this loop never
             touches attn_capture_ctx, so the scoring UNet uses SDPA.

        THE MAP IS SHARED BY THE PAIR (and detached), which is what keeps the gap a clean paired difference:
        E_neg and E_pos for axis p are BOTH reduced by the SAME A_p, so A_p cancels in its own right and only
        the residual differs between the two classes. Building a per-class map instead would let the map
        difference leak into g_p. A_p is averaged over the pos AND neg forwards (the content tokens are the
        shared, non-polarity part of the pair) -- the exact analogue of the gender file's woman/man common map,
        but kept PER AXIS rather than pooled across axes (pooling across axes was the thing that smeared).

        NOISE. eps is SHARED WITHIN a pair: E_pos and E_neg for axis p are computed on the SAME zt/eps, which
        is what makes the gap g_p = E_neg - E_pos a low-variance PAIRED difference (the two prompts see the
        identical noise, so everything except the prompt cancels). eps is drawn INDEPENDENTLY ACROSS axes
        (--valence_axis_eps independent, the default): sharing one eps across all 2P prompts -- the obvious
        port of the old 2-prompt code -- would correlate the P gap estimates through a single noise draw, so
        averaging them would cancel no Monte-Carlo noise and pooling would buy nothing.

        [PERF] the K timesteps are folded into the batch dim ([n,K,...] -> [n*K,...]), so the scoring UNet
        runs ONCE per prompt (2P forwards) rather than 2*P*K sequential ones. The attention capture is a
        no_grad side-effect; the cost of 'peraxis' is that the scoring UNet runs the explicit attention path.
        """
        n = z0.shape[0]
        H, W = z0.shape[-2], z0.shape[-1]
        timesteps = torch.linspace(
            args.residual_t_min, args.residual_t_max, steps=args.residual_num_timesteps, device=z0.device
        ).round().long()
        K = timesteps.shape[0]
        t_all = timesteps.repeat(n)                                      # [n*K], row i*K+k -> t_k

        def _draw_noise():
            """one fresh eps per timestep, folded into the batch dim -> (eps_all, zt_all) [n*K,4,H,W]"""
            eps_list, zt_list = [], []
            for t in timesteps:
                eps_k = torch.randn_like(z0)
                zt_list.append(noise_scheduler.add_noise(z0, eps_k, t.repeat(n)))
                eps_list.append(eps_k)
            eps_all = torch.stack(eps_list, dim=1).reshape(n * K, *z0.shape[1:])
            zt_all = torch.stack(zt_list, dim=1).reshape(n * K, *z0.shape[1:]).to(weight_dtype)
            return eps_all, zt_all

        shared = _draw_noise() if args.valence_axis_eps == "shared" else None

        cols = []
        for ax in valence_axes:
            eps_all, zt_all = shared if shared is not None else _draw_noise()
            # This axis's per-class content-token indices; None => no region => uniform reduction.
            use_attn = ax["pos_content_idxs"] is not None
            per_class = []                                               # [n] error per class, order [neg,pos]
            per_class_res = []                                           # grad-carrying per-pixel res [n,K,H,W]
            attn_accum = torch.zeros(n, H, W, dtype=torch.float, device=z0.device)   # detached accumulator
            attn_count = 0
            # class order [neg=0, pos=1]; each forward captures with ITS OWN prompt's content-token indices.
            for embeds, tok_idxs in ((ax["neg_embeds"], ax["neg_content_idxs"]),
                                     (ax["pos_embeds"], ax["pos_content_idxs"])):
                c = embeds.expand(n * K, -1, -1)
                if use_attn:
                    attn_capture_ctx.store = []
                    attn_capture_ctx.token_idxs = tok_idxs
                    attn_capture_ctx.enabled = True
                eps_pred = scoring_unet(zt_all, t_all, encoder_hidden_states=c).sample
                if not use_attn:
                    # UNIFORM: the EXACT single fused reduction of the no-attmap file (mean over C,H,W then K).
                    # Kept bit-for-bit identical (not the channel-mean-then-spatial-mean form) so uniform mode is
                    # a perfect ablation baseline: any difference vs _multiprompt_valence.py is attributable to
                    # peraxis, never to fp32 reduction-order. The .float() pins the whole valence pipeline to fp32.
                    e = (eps_pred.float() - eps_all.float()).pow(2).mean(dim=(1, 2, 3)).view(n, K).mean(dim=1)
                    per_class.append(e)
                    continue
                # PERAXIS: accumulate this prompt's content-token cross-attention maps (detached), resized to (H,W).
                attn_capture_ctx.enabled = False
                captured = attn_capture_ctx.store
                attn_capture_ctx.store = []
                for col, heads in captured:
                    hw = col.shape[-1]
                    s = int(round(math.sqrt(hw)))
                    a = col.view(n * K, heads, s, s).float()                    # [n*K, heads, s, s]
                    a = torch.nn.functional.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
                    a = a.mean(dim=1).view(n, K, H, W).mean(dim=1)              # mean heads, then K -> [n,H,W]
                    attn_accum = attn_accum + a
                    attn_count += 1
                # per-pixel squared residual (channel-mean) -> [n,K,H,W], grad-carrying; reduced spatially below.
                res = (eps_pred.float() - eps_all.float()).pow(2).mean(dim=1).view(n, K, H, W)
                per_class_res.append(res)

            if use_attn:
                # ONE detached, sum-to-1 map per axis, shared by both classes (see docstring). attn_count > 0
                # is guaranteed: use_attn implies the capture processor is installed and attn2 layers fired.
                A = attn_accum / attn_count                                          # [n,H,W]
                A = A / (A.sum(dim=(1, 2), keepdim=True) + 1e-8)                      # sum-to-1 per image
                A = A.detach().unsqueeze(1)                                          # [n,1,H,W] weighting mask
                per_class = [(A * r).sum(dim=(2, 3)).mean(dim=1) for r in per_class_res]   # weighted SUM, mean-K
            cols.append(torch.stack(per_class, dim=1))                   # [n,2] order [neg,pos]
        return torch.stack(cols, dim=1)                                  # [n,P,2]

    def valence_logits_from_errors(E):
        """Pool the P axes into ONE binary decision. E: [n,P,2] fp32, class order [neg=0, pos=1].

        Returns (logits [n,2], score [n], gaps [n,P] detached-for-logging).

            gap    g_p = E_neg - E_pos          (> 0  <=>  the image reads POSITIVE)
            z_p        = (g_p - m_p) / s_p      (FROZEN offline constants, never live batch statistics)
            score      = sum_p w_p * z_p        (w sums to 1)
            logits     = [-score/(2*tau), +score/(2*tau)]

        WHY THE SYMMETRIC LOGIT FORM: logit_pos - logit_neg = score/tau by construction, so
        probs[:,1] = sigmoid(score/tau). Writing the logits this way instead of [-E_neg/tau, -E_pos/tau] is
        IDENTICAL in value and in gradient -- softmax and cross-entropy are invariant to a per-sample
        class-constant shift -- but it drops the large class-independent common-mode term that fp32 would
        otherwise have to cancel.

        WHY THIS IS SIMULTANEOUSLY 'average the errors', 'average the logits' AND 'average the log-probs':
        logit_c = -E_c/tau is AFFINE in E with a shared tau, so averaging over axes commutes with it; and the
        mean of per-axis log-probs differs from the mean of per-axis logits only by the per-axis
        log-partition, which is CLASS-INDEPENDENT and therefore invisible to softmax/CE. The three are ONE
        estimator (verified in float64: max abs diff 4.6e-14). Only 'average the PROBABILITIES' is genuinely
        different -- and it is the one that breaks; see --valence_pooling and the prob_mean branch below.

        WHY THE STANDARDISATION MATTERS: each axis's BASELINE error level cancels for free (g_p is a
        within-pair difference), but each axis's GAP SCALE does not. Without 1/s_p the axis with the largest
        dynamic range owns the pooled decision -- and owns the z0 gradient too, since
        dL/dE[p,c] is proportional to w_p/s_p -- so a '4-axis' scorer silently degenerates into a 1-axis one.
        """
        gaps = E[:, :, 0] - E[:, :, 1]                                        # [n,P]  E_neg - E_pos
        z = (gaps - valence_center.view(1, -1)) / valence_scale.view(1, -1)   # [n,P]  standardised

        if args.valence_pooling == "prob_mean":
            # ABLATION ONLY -- the user's option (iii), kept reachable so its failure can be DEMONSTRATED
            # rather than argued about. Per-axis sigmoid then average = a MIXTURE of experts: one confident
            # axis cannot be outvoted, and a saturated axis contributes ~no gradient (dp/dg ~ p(1-p) -> 0).
            # With a small tau each axis becomes a hard 0/1 vote, so the mean lands on the discrete grid
            # {0, 1/P, ..., 1} and generate_dynamic_targets' argsort splits the resulting tie buckets by
            # ARRIVAL ORDER rather than by valence -- while still reporting a perfect 50/50 count.
            p_pos = torch.sigmoid(z / args.valence_tau)                        # [n,P]
            p_pos = (valence_weight.view(1, -1) * p_pos).sum(dim=1)            # [n]
            p_pos = p_pos.clamp(1e-6, 1.0 - 1e-6)
            logits = torch.stack([torch.log1p(-p_pos), torch.log(p_pos)], dim=1)   # [n,2], CE on log p
            return logits, p_pos - 0.5, gaps.detach()                          # score = monotone key

        score = (valence_weight.view(1, -1) * z).sum(dim=1)                    # [n]
        half = score / (2.0 * args.valence_tau)
        logits = torch.stack([-half, half], dim=1)                             # [n,2], [neg=0, pos=1]
        return logits, score, gaps.detach()

    @torch.no_grad()
    def residual_face_indicators(z0):
        """NODETECTOR: residual-error face/no-face indicator replacing insightface get_face in the
        TRAINING branch points. Same estimator family as residual_gender_logits, but with a UNIFORM
        spatial mean (NO attention weighting) and its own prompt pair / timestep grid:
            E_c  = mean_t mean_pixels || eps_pred(z_t, t, c) - eps ||^2 ,
                   t = --face_residual_num_timesteps (8) steps linspaced in
                       [--face_residual_t_min, --face_residual_t_max] (50..950)
            face iff E(--face_residual_face_prompt) < E(--face_residual_nonface_prompt)
        The SAME fresh eps/zt per timestep is shared by both prompts (paired, low-variance), scored
        by the FROZEN scoring UNet under no_grad (the insightface detector it replaces was equally
        non-differentiable). K folded into the batch dim like residual_gender_logits ([n*K,...]).
        Offline ablation (face_error_exp/exp100, 50 genuine-face vs 50 genuine-noface occupation
        images): "a photo of a face" vs "a faceless photo" = 88/100 at K=8, t50-950 (== its K=15
        score; K<8 degrades). Returns a bool tensor [n], same as get_face's face_indicators.
        """
        n = z0.shape[0]
        timesteps = torch.linspace(
            args.face_residual_t_min, args.face_residual_t_max,
            steps=args.face_residual_num_timesteps, device=z0.device,
        ).round().long()
        K = timesteps.shape[0]

        # fresh eps per timestep, folded into the batch dim: [n,K,...] -> [n*K,...] (row = i*K + k)
        eps_list, zt_list = [], []
        for t in timesteps:
            t_batch = t.repeat(n)
            eps_k = torch.randn_like(z0)                          # shared by BOTH prompts below
            zt_list.append(noise_scheduler.add_noise(z0, eps_k, t_batch))
            eps_list.append(eps_k)
        eps_all = torch.stack(eps_list, dim=1).reshape(n * K, *z0.shape[1:])
        zt_all = torch.stack(zt_list, dim=1).reshape(n * K, *z0.shape[1:]).to(weight_dtype)
        t_all = timesteps.repeat(n)                                                   # [n*K], row i*K+k -> t_k

        E = {}
        for cls, embeds in (("face", residual_face_embeds), ("faceless", residual_faceless_embeds)):
            c = embeds.expand(n * K, -1, -1)
            eps_pred = scoring_unet(zt_all, t_all, encoder_hidden_states=c).sample
            # uniform mean over channels+pixels (NO attention weighting), then mean over K -> [n]
            E[cls] = (eps_pred.float() - eps_all.float()).pow(2).mean(dim=(1, 2, 3)).view(n, K).mean(dim=1)

        return E["face"] < E["faceless"]                                              # bool [n]

    def residual_valence_and_realism(z0):
        """Fused scorer for the TRAINING loss: the multi-prompt valence class + the SRR realism anchor.

        z0: [n,4,H,W] clean latent (NOT detached) -- gradient flows to the trainable model. The scorer
        (scoring_unet / scoring_text_encoder) stays frozen, so E_realistic pulls z0 onto the frozen model's
        "realistic photo" manifold (score distillation on the realism prompt).

        Returns:
          logits [n,2]     pooled valence logits, class order [neg=0, pos=1]  -> the fair (CE) loss
          score  [n]       the pooled continuous score; probs[:,1] = sigmoid(score / --valence_tau)
          gaps   [n,P]     per-axis gaps, DETACHED. Logged per axis so that a DEAD axis (gap ~ noise) or a
                           DOMINATING axis is visible directly in wandb, instead of having to be inferred
                           from a flat loss curve -- which is exactly how this project's two previous dead
                           scorers stayed hidden.
          E_realistic [n]  the SRR loss: the raw eps-residual under --srr_prompt, meaned over the WHOLE image
                           and over timesteps. RAW error -- NOT divided by tau.

        SRR GRADIENT SUPPORT = THE WHOLE IMAGE. The gender file input-masked z0 outside the person region so
        d(E_realistic)/dz0 was exactly zero outside it. That was right when the fair loss only edited a face.
        The valence fair loss edits expression AND clothing AND background AND global tone, so a
        person-region-only realism anchor would leave the model free to WRECK THE BACKGROUND in order to win
        the valence objective. The anchor must cover the same support the fair loss pushes on.
        """
        n = z0.shape[0]
        E = _valence_axis_errors(z0)                                     # [n,P,2], grad-carrying
        logits, score, gaps = valence_logits_from_errors(E)              # [n,2], [n], [n,P]

        # ---- SRR realism. Its own eps draw: E_realistic is a RAW error, not a paired difference, so it
        # gains nothing from sharing eps with the class prompts (only the within-pair sharing above matters).
        timesteps = torch.linspace(
            args.residual_t_min, args.residual_t_max, steps=args.residual_num_timesteps, device=z0.device
        ).round().long()
        K = timesteps.shape[0]
        t_all = timesteps.repeat(n)

        eps_list, zt_list = [], []
        for t in timesteps:
            eps_k = torch.randn_like(z0)
            zt_list.append(noise_scheduler.add_noise(z0, eps_k, t.repeat(n)))
            eps_list.append(eps_k)
        eps_all = torch.stack(eps_list, dim=1).reshape(n * K, *z0.shape[1:])
        zt_all = torch.stack(zt_list, dim=1).reshape(n * K, *z0.shape[1:]).to(weight_dtype)
        c_real = residual_realistic_embeds.expand(n * K, -1, -1)
        eps_pred = scoring_unet(zt_all, t_all, encoder_hidden_states=c_real).sample
        E_realistic = (eps_pred.float() - eps_all.float()).pow(2).mean(dim=(1, 2, 3)).view(n, K).mean(dim=1)

        return logits, score, gaps, E_realistic

    @torch.no_grad()
    def valence_valid_indicators(z0):
        """Which samples participate in the valence loss. [n] bool.

        DEFAULT (--valence_face_gate none): ALL of them. This is the substantive change, not a refactor.
        A scene has a valence whether or not it contains a face -- a faceless, grim alleyway is negative --
        so gating on face detection would enforce the 50/50 objective only INSIDE the face-containing
        subpopulation (roughly a third to a half of occupation generations), which is a sample-selection
        bias, not a safety net. It also means the gathered shapes/dtypes no longer depend on per-rank
        detection outcomes, which structurally removes the NCCL all_gather deadlock class this project has
        already hit twice.

        --valence_face_gate errfd reinstates the gender file's residual-error face detector, for the ablation.
        """
        if args.valence_face_gate == "errfd":
            return residual_face_indicators(z0)
        return torch.ones([z0.shape[0]], dtype=torch.bool, device=z0.device)

    def get_valence(z0, selector=None, fill_value=-1):
        """Training-time valence predictor on the clean latent z0. Successor of get_face_gender.

        Returns (preds [n] int64, probs [n,2] fp32, logits [n,2] fp32, score [n] fp32, gaps [n,P] fp32),
        class order [neg=0, pos=1]. Raw logits are returned so callers can feed cross_entropy directly.

        `selector` exists ONLY for --valence_face_gate errfd (the gender file's face gate). With the default
        --valence_face_gate none it is None and EVERY image is scored: a scene has a valence whether or not
        it contains a face, and gating on faces would silently restrict the 50/50 objective to the
        face-containing subpopulation -- a sample-selection bias, not a safety net.

        DTYPE INVARIANT (do not break): the empty and non-empty branches must emit IDENTICAL dtypes -- fp32
        probs/logits/score/gaps and int64 preds. An fp16/fp32 split between these two branches silently
        DEADLOCKS the downstream all_gather until the NCCL watchdog fires; that bug has already cost this
        project a debugging cycle. `score` and `gaps` are NEW gathered tensors, so they are pinned here too.
        """
        z0_sel = z0 if selector is None else z0[selector]
        n_all = z0.shape[0]

        if z0_sel.shape[0] == 0:
            logits = torch.empty([0, 2], dtype=torch.float, device=z0.device)
            probs = torch.empty([0, 2], dtype=torch.float, device=z0.device)
            preds = torch.empty([0], dtype=torch.int64, device=z0.device)
            score = torch.empty([0], dtype=torch.float, device=z0.device)
            gaps = torch.empty([0, P_axes], dtype=torch.float, device=z0.device)
        else:
            E = _valence_axis_errors(z0_sel)
            logits, score, gaps = valence_logits_from_errors(E)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.max(dim=-1).indices
            logits, probs = logits.float(), probs.float()
            score, gaps = score.float(), gaps.float()

        if selector is None:
            return preds, probs, logits, score, gaps

        def _scatter(t):
            out = torch.ones([n_all] + list(t.shape[1:]), dtype=t.dtype, device=t.device) * fill_value
            out[selector] = t
            return out

        return _scatter(preds), _scatter(probs), _scatter(logits), _scatter(score), _scatter(gaps)

    # REMOVED FOR VALENCE: compute_gender_attmaps / save_gender_attmap_panels / save_grad_gate_panels.
    # All three visualised the class-token cross-attention map and the SCR gradient gate derived from it.
    # Neither exists for a scene-level class: the map cannot be built for these axes (non-contiguous content
    # words, incommensurable prompt lengths, 4 supports that average to ~uniform -- see the header), and the
    # spatial SCR gate is gone with it. The per-axis GAP logging in the training loop replaces them as the
    # diagnostic: it is what actually tells you whether an axis is dead or dominating.

    def get_face_gender_test(face_chips, selector=None, fill_value=-1):
        """for the separately-trained CelebA gender *test* classifier (evaluation only).

        Identical interface to get_face_gender, but gender_classifier_test outputs 2 logits
        directly (gender only), so there is no 40-attribute reshape / index-20 selection.
        Class convention matches get_face_gender: index 1 == male, index 0 == female.
        """
        if selector != None:
            face_chips_w_faces = face_chips[selector]
        else:
            face_chips_w_faces = face_chips

        if face_chips_w_faces.shape[0] == 0:
            logits_gender = torch.empty([0,2], dtype=face_chips.dtype, device=face_chips.device)
            probs_gender = torch.empty([0,2], dtype=face_chips.dtype, device=face_chips.device)
            preds_gender = torch.empty([0], dtype=torch.int64, device=face_chips.device)
        else:
            logits_gender = gender_classifier_test(face_chips_w_faces.float())
            probs_gender = torch.softmax(logits_gender, dim=-1)
            preds_gender = probs_gender.max(dim=-1).indices

        # Pin to float32 so the gathered probs dtype never depends on whether a face
        # was detected (no-face branch built fp16, has-face branch .float()->fp32);
        # the mismatch silently DEADLOCKS all_gather until the NCCL watchdog timeout.
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
    def generate_dynamic_targets(scores, valid, target_ratio=0.5, w_uncertainty=False):
        """Generate the dynamic 50/50 targets for the distributional alignment loss.

        Args:
            scores (torch.tensor): [N] the CONTINUOUS pooled valence score (higher = more POSITIVE).
            valid  (torch.tensor): [N] bool, which rows participate in the ranking population.
            target_ratio (float): fraction of the population assigned to class 0 (negative); the top
                (1 - target_ratio) of the ranking gets class 1 (positive). 0.5 -> a 50/50 split.
            w_uncertainty: also return the binomial uncertainty of each assigned target.

        Returns targets_all [N] int64 (-1 where not ranked) and, optionally, uncertainty_all [N].

        WHY IT RANKS THE SCORE AND NOT probs[:,1] (this is a real bug fix, not a refactor):
        probs[:,1] = sigmoid(score / valence_tau) is a strictly increasing function of the score, so the two
        rankings are MATHEMATICALLY identical -- but fp32 sigmoid returns EXACTLY 1.0 once the logit gap
        exceeds ~16.6 (and exactly 0.0 below ~-16.6). Ranking on probs therefore manufactures large TIE
        buckets out of thin air, and argsort breaks those ties by ARRIVAL ORDER, handing opposite 50/50
        targets to images the scorer considers identical. The continuous score cannot saturate. (The failure
        is invisible in wandb, because argsort of a tied block still emits a perfect 50/50 count.)

        WHY `valid` IS AN EXPLICIT ARGUMENT: the gender file inferred the ranking population from a `-1`
        FILL VALUE inside probs. A score is a signed quantity for which -1 is a perfectly legal value, so the
        sentinel trick is not merely ugly here, it is WRONG. With --valence_face_gate none, `valid` is simply
        all-True: every image has a valence and participates.
        """
        idxs_2_rank = valid
        scores_2_rank = scores[idxs_2_rank]
        n_rank = scores_2_rank.shape[0]

        targets_all = torch.ones([scores.shape[0]], dtype=torch.long, device=scores.device) * (-1)
        uncertainty_all = torch.ones([scores.shape[0]], dtype=scores.dtype, device=scores.device) * (-1)
        if n_rank == 0:
            return (targets_all, uncertainty_all) if w_uncertainty else targets_all

        rank = torch.argsort(torch.argsort(scores_2_rank))
        targets = (rank >= (n_rank * target_ratio)).long()      # top (1-target_ratio) -> class 1 = POSITIVE
        targets_all[idxs_2_rank] = targets

        if w_uncertainty:
            uncertainty = torch.ones([n_rank], dtype=scores.dtype, device=scores.device) * (-1)
            uncertainty[targets == 1] = torch.tensor(
                1 - scipy.stats.binom.cdf(
                    (rank[targets == 1]).cpu().numpy(),
                    n_rank,
                    1 - target_ratio,
                    )
                ).to(scores.dtype).to(scores.device)
            uncertainty[targets == 0] = torch.tensor(
                scipy.stats.binom.cdf(
                    rank[targets == 0].cpu().numpy(),
                    n_rank,
                    target_ratio,
                    )
                ).to(scores.dtype).to(scores.device)
            uncertainty_all[idxs_2_rank] = uncertainty
            return targets_all, uncertainty_all
        return targets_all

    def err_valence_scores_eval(z0_chunks, seed):
        """EVAL-ONLY readout of the TRAINING residual-error valence scorer, over ALL IMAGES.

        z0_chunks: list of [n_j,4,H,W] clean latents, in the SAME order they were generated (so the
        concatenation lines up row-for-row with the concatenated images and therefore, after
        all_gather + [:val_keep], with images_all / the CLIP predictions).
        seed: int, the eps seed for THIS call -- see point 3 below. Must differ per (step, prompt,
        ft/ori, rank), otherwise every scorer call reuses the identical Monte-Carlo draw.
        Returns (preds [n] int64, probs [n,2] fp32, score [n] fp32, gaps [n,P] fp32), un-scattered.

        NO FACE GATE. `selector=None` is passed as a LITERAL, deliberately NOT the training loop's
        ternary `selector=(valid_indicators if args.valence_face_gate == "errfd" else None)`. Copying that
        would (a) fire residual_face_indicators -- 2 extra scoring-UNet forwards per chunk -- and (b) put -1
        fill rows back into the metrics. Every eval image is scored, which also matches the CLIP eval head's
        denominator, so errVal/* and the CLIP valence_* keys describe the SAME population and may be compared
        directly (that comparison is the point: see errVal/agree_with_clip).

        THREE THINGS THIS WRAPPER EXISTS FOR -- do not inline it away:
        1. torch.no_grad, defensively. get_valence / _valence_axis_errors are UNDECORATED and inherit the
           caller's grad mode; without no_grad an autograd graph is built across 2*P gradient-checkpointed
           UNet forwards at batch n*K, per prompt -> OOM. evaluate_process IS decorated @torch.no_grad(), so
           this inner `with` is currently redundant -- keep it anyway: it makes the helper safe to call from
           anywhere, and DO NOT "simplify" by moving the decorator off evaluate_process onto this function
           (that mistake was made once already while writing this, and it silently un-guards ~300 lines).
        2. attn_capture_ctx reset. Under --valence_axis_attn peraxis (the default) _valence_axis_errors
           mutates the process-global attn_capture_ctx and resets it with NO try/finally. If the forward
           raises, `enabled` stays True and EVERY later scoring_unet forward (SRR, errFD, SCR h-space) appends
           to ctx.store unboundedly -- a recoverable eval OOM would become silent training corruption plus a
           memory leak. Eval runs mid-training every --evaluate_every_n_iter steps, so that must not happen.
        3. RNG isolation, NOT merely RNG restore. The scorer burns global CUDA RNG (P*K torch.randn_like
           draws per chunk; _draw_noise takes no generator argument, so a private torch.Generator cannot be
           threaded in without editing the training scorer). Left alone it would shift every subsequent
           training noise draw, so a run WITH this metric would not be step-comparable to one without it.
           But saving and restoring the state ALONE is a trap: nothing else in the eval prompt loop consumes
           CUDA RNG (the val noises are drawn once, up in evaluation_step), so every scorer call would
           re-enter from the byte-identical generator state and draw the SAME eps -- making errVal/gap_ori a
           worthless "noise floor" (deterministic given the images) and giving the cross-prompt mean zero
           Monte-Carlo averaging. So: seed explicitly per call, then restore. See err_seed for the tuple the
           seed is hashed from and why it is hashed rather than stride-mixed.
        Chunking is the caller's job: pass one chunk per --val_GPU_batch_size generation batch. NOTE this
        bounds but does NOT equalise memory vs generation -- _valence_axis_errors folds K
        (--residual_num_timesteps, default 15) into the batch dim, so the scoring UNet runs at
        val_GPU_batch_size*K (8*15 = 120) against generation's CFG batch of 2*val_GPU_batch_size (16).
        That is the same scorer batch the training loop already sustains, but scorer memory grows 15x
        faster than generation memory if --val_GPU_batch_size is raised.
        """
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(accelerator.device) if accelerator.device.type == "cuda" else None
        )
        preds_l, probs_l, score_l, gaps_l = [], [], [], []
        try:
            # INSIDE the try: once this runs the generator is dirty, so every exit path from here on must go
            # through the finally that restores it.
            torch.manual_seed(seed)
            with torch.no_grad():
                for z0_c in z0_chunks:
                    preds_c, probs_c, _logits_c, score_c, gaps_c = get_valence(z0_c, selector=None)
                    preds_l.append(preds_c)
                    probs_l.append(probs_c)
                    score_l.append(score_c)
                    gaps_l.append(gaps_c)
        finally:
            attn_capture_ctx.enabled = False
            attn_capture_ctx.token_idxs = None
            attn_capture_ctx.store = []
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, accelerator.device)
        return torch.cat(preds_l), torch.cat(probs_l), torch.cat(score_l), torch.cat(gaps_l)

    @torch.no_grad()
    def evaluate_process(which_text_encoder, which_unet, name, prompts, noises, current_global_step):
        logs = []
        log_imgs = []
        num_denoising_steps = 25
        # Number of images generated per prompt across ALL GPUs after all-gather.
        # Each GPU generates args.val_images_per_prompt_GPU images -> total = that * num_processes.
        num_images_per_prompt_gathered = args.val_images_per_prompt_GPU * accelerator.num_processes
        # We optionally keep only the first `val_images_per_prompt_total` of the gathered images
        # (e.g. 8 GPUs * 8 = 64 generated, but keep exactly 60 for metrics / grids).
        if args.val_images_per_prompt_total and args.val_images_per_prompt_total > 0:
            val_keep = min(args.val_images_per_prompt_total, num_images_per_prompt_gathered)
        else:
            val_keep = num_images_per_prompt_gathered

        # EVAL READOUT OF THE TRAINING SCORER (--eval_err_valence). Scored over ALL images: the errFD face
        # gate that restricts the train_* valence metrics is deliberately NOT applied here (see
        # err_valence_scores_eval). These flags are constant across prompts, so every logs_i gets the SAME
        # key set -- required by the cross-prompt averaging loop at the bottom of this function, which does
        # logs[0].keys() and then indexes every log with it.
        score_err_gen = args.eval_err_valence in ("generated", "both")
        score_err_ori = args.eval_err_valence == "both"

        def err_seed(prompt_idx, which):
            """Reproducible, INDEPENDENT eps seed per (run, eval arm, step, prompt, ft/ori, rank).

            Independence along every axis is what makes the readout statistically useful: across steps so the
            curve is not one frozen draw; across prompts so the cross-prompt mean actually averages MC noise
            down; across ft/ori so errVal/gap_ori is a genuine noise floor rather than the same draw replayed;
            across ranks so the gathered population is not N copies of one noise pattern; across `name` so a
            re-enabled "main" arm does not share eps with "EMA".

            HASHED, NOT STRIDE-MIXED, ON PURPOSE. The obvious `a*A + b*B + c*C + rank` form silently aliases
            as soon as one stride is smaller than the range of the term below it -- e.g. a ft/ori stride of 7
            collides with rank at 8 GPUs, so rank 7's finetuned draw and rank 0's original draw become the
            same eps stream and the "noise floor" quietly correlates with the thing it is supposed to be a
            floor for. sha256 over the tuple has no strides to get wrong at any world size or prompt count.
            hashlib (not the builtin hash()) because PYTHONHASHSEED randomises str hashing per process --
            builtin hash would make this neither reproducible across runs nor consistent across ranks.
            """
            key = f"{args.seed}|{name}|{int(current_global_step)}|{int(prompt_idx)}|{which}|{int(accelerator.process_index)}"
            # 63-bit, comfortably inside torch's int64 seed range. 32 bits would be enough for correctness
            # but not for comfort: a full run reaches O(1e5) distinct (step, prompt, arm, rank) tuples, where
            # the birthday rate at 32 bits is already ~1 expected collision.
            return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)

        for prompt_idx, (prompt_i, noises_i) in enumerate(itertools.zip_longest(prompts, noises)):
            if accelerator.is_main_process:
                logs_i = {
                    "valence_gap": [],
                    "valence_gap_abs": [],
                    "valence_pred_between_0.2_0.8": [],
                    "CLIP-T": [],
                    "CLIP-I": [],
                    "DINO": [],
                }
                if score_err_gen:
                    # NOT a debias result -- the training objective re-evaluated on held-out prompts. Namespaced
                    # under "errVal/" so wandb renders it in its own panel group and it can never be mistaken
                    # for the independent CLIP number sitting next to it (eval_<name>_valence_gap).
                    logs_i["errVal/gap"] = []
                    logs_i["errVal/gap_abs"] = []
                    logs_i["errVal/pred_between_0.2_0.8"] = []
                    logs_i["errVal/score_mean"] = []
                    logs_i["errVal/score_std"] = []
                    # THE REWARD-HACKING TELL. The CLIP head and this scorer run on the SAME rows, so their
                    # disagreement is measurable directly. errVal/gap -> 0 while the CLIP valence_gap stays
                    # large == the model is winning the residual scorer without changing what the images look
                    # like. Nothing else logged by this file can show that.
                    logs_i["errVal/agree_with_clip"] = []
                    logs_i["errVal/gap_minus_clip_gap"] = []
                    for _ax in valence_axes:
                        # RAW per-axis gap g_p = E_neg - E_pos. Diagnostic for a DEAD axis (|mean| ~ 0, |corr|
                        # ~ 0) or a DOMINATING one. Raw gaps are NOT commensurable across axes...
                        logs_i[f"errVal/gap_{_ax['name']}_mean"] = []
                        logs_i[f"errVal/gap_{_ax['name']}_std"] = []
                        logs_i[f"errVal/gap_{_ax['name']}_corr_pooled"] = []
                        # ...so also log the STANDARDISED z_p = (g_p - m_p)/s_p. That is what actually enters
                        # the pooled score, and it is the only per-axis quantity comparable across axes and
                        # across runs -- i.e. the one that reveals --valence_axis_scale calibration drift on
                        # the eval distribution.
                        logs_i[f"errVal/z_{_ax['name']}_mean"] = []
                if score_err_ori:
                    # NOISE FLOOR / CONTROL. Frozen weights + fixed prompts, so this is constant in
                    # expectation across training; any drift in it is Monte-Carlo noise from the scorer's
                    # fresh eps draw. The delta is the interpretable quantity, the absolute gap is not.
                    logs_i["errVal/gap_ori"] = []
                    logs_i["errVal/gap_delta_ft_minus_ori"] = []
                log_imgs_i = {}
            ################################################
            # step 1: generate all ori images
            images_ori = []
            # One z0 chunk per generation batch, appended in generation order so the concatenation stays
            # row-aligned with images_ori (and so the scorer's peak memory tracks val_GPU_batch_size).
            z0s_ori = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                if args.train_text_encoder and args.train_unet:
                    out_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=eval_unet, return_latents=score_err_ori, skip_denoise_frac=0.0)
                elif args.train_text_encoder and not args.train_unet:
                    out_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=unet, return_latents=score_err_ori, skip_denoise_frac=0.0)
                elif not args.train_text_encoder and args.train_unet:
                    out_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=eval_unet, return_latents=score_err_ori, skip_denoise_frac=0.0)
                if score_err_ori:
                    images_ij, z0_ori_ij = out_ij
                    z0s_ori.append(z0_ori_ij)
                else:
                    images_ij = out_ij
                images_ori.append(images_ij)
            images_ori = torch.cat(images_ori)

            # VALENCE: no insightface, no face chips, no CelebA gender classifier. The valence head is a
            # zero-shot CLIP head that lives on rank 0 and scores the FULL image, so it runs AFTER the
            # all-gather (below) -- which also means 4 of the old per-rank all_gather calls disappear.
            # Removing the face gate is what makes this safe: the gathered shapes/dtypes no longer depend on
            # per-rank detection outcomes, which is the structural cause of the NCCL deadlock class this
            # project has already hit twice.
            images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
            images_ori_all = images_ori_all[:val_keep]   # see --val_images_per_prompt_total

            # ORI-side training-scorer readout (--eval_err_valence both). Scoring and the all_gather run on
            # EVERY rank -- never inside an is_main_process guard. evaluate_process has no barrier, so a
            # collective entered by rank 0 alone hangs the job until the NCCL watchdog fires. Truncate to
            # val_keep AFTER the gather so rows stay aligned with images_ori_all.
            if score_err_ori:
                _, probs_ev_ori, _, _ = err_valence_scores_eval(z0s_ori, err_seed(prompt_idx, "ori"))
                probs_ev_ori_all = customized_all_gather(probs_ev_ori, accelerator, return_tensor_other_processes=False)[:val_keep]

            if accelerator.is_main_process:
                preds_valence_ori_all, probs_valence_ori_all = get_valence_test(images_ori_all)
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_ori.jpg")
                plot_in_grid(
                    images_ori_all,
                    save_to,
                    preds_class=preds_valence_ori_all,
                    pred_class_probs=probs_valence_ori_all.max(dim=-1).values,
                )

                log_imgs_i["img_ori"] = [save_to]

            
            images = []
            z0s = []   # one chunk per generation batch, in generation order (see z0s_ori above)
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                if score_err_gen:
                    images_ij, z0_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=which_text_encoder, which_unet=which_unet, return_latents=True, skip_denoise_frac=0.0)
                    z0s.append(z0_ij)
                else:
                    images_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=which_text_encoder, which_unet=which_unet, skip_denoise_frac=0.0)
                images.append(images_ij)
            images = torch.cat(images)

            images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
            images_all = images_all[:val_keep]   # see --val_images_per_prompt_total

            # FINETUNED-side training-scorer readout (--eval_err_valence generated|both), over ALL images --
            # no face gate. All ranks score and gather; truncate to val_keep afterwards so the rows line up
            # with images_all and hence with the CLIP predictions computed from it on rank 0. get_valence
            # pins fp32/int64 in the selector=None branch, so the gathered dtypes are rank-uniform.
            if score_err_gen:
                preds_ev, probs_ev, scores_ev, gaps_ev = err_valence_scores_eval(z0s, err_seed(prompt_idx, "ft"))
                preds_ev_all = customized_all_gather(preds_ev, accelerator, return_tensor_other_processes=False)[:val_keep]
                probs_ev_all = customized_all_gather(probs_ev, accelerator, return_tensor_other_processes=False)[:val_keep]
                scores_ev_all = customized_all_gather(scores_ev, accelerator, return_tensor_other_processes=False)[:val_keep]
                gaps_ev_all = customized_all_gather(gaps_ev, accelerator, return_tensor_other_processes=False)[:val_keep]

            ################################################
            # eval fidelity / text-alignment metrics (CLIP-T, CLIP-I, DINO) are ALL computed on the
            # MAIN PROCESS ONLY -- open_clip bigG and the eval DINOv2 both live only on rank 0
            # (matches file B). They run over the already all-gathered & val_keep-truncated
            # images_all / images_ori_all in the is_main_process block below, so no per-rank feature
            # pass or extra all-gather is needed here. images_all[k] pairs with images_ori_all[k]
            # (same noises_i, same order).

            if accelerator.is_main_process:
                preds_valence_all, probs_valence_all = get_valence_test(images_all)
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_generated.jpg")
                plot_in_grid(
                    images_all,
                    save_to,
                    preds_class=preds_valence_all,
                    pred_class_probs=probs_valence_all.max(dim=-1).values,
                    )

                log_imgs_i["img_generated"] = [save_to]

            if accelerator.is_main_process:
                # NOTE the missing `-1` filter: the gender file dropped the fill-value rows here, which
                # silently made the metric DENOMINATOR "images with a detected face" rather than "images".
                # Every image has a valence, so the denominator is now every image -- an intentional
                # semantic change, and the reason the debias numbers are not comparable to a gender run.
                p_pos = probs_valence_all[:, 1]                                  # P(positive)
                valence_gap = ((p_pos >= 0.5).float().mean() - (p_pos < 0.5).float().mean()).item()
                valence_between = ((p_pos >= 0.2) * (p_pos <= 0.8)).float().mean().item()
                logs_i["valence_gap"].append(valence_gap)
                logs_i["valence_gap_abs"].append(abs(valence_gap))
                logs_i["valence_pred_between_0.2_0.8"].append(valence_between)

                # ---------------------------------------------------------------------------------
                # TRAINING-SCORER READOUT (--eval_err_valence). ALL IMAGES, NO FACE GATE.
                # ---------------------------------------------------------------------------------
                # WHAT THIS IS: the residual-error valence head the training loss is built on, re-run on the
                # validation generations. It is a DIAGNOSTIC OF THE OPTIMISER, never a debias result -- the
                # optimiser is directly pushing on this quantity. Every bias claim must come from
                # valence_gap above (the independent zero-shot CLIP head; see the get_valence_test docstring).
                #
                # WHY THE DENOMINATOR DIFFERS FROM train_valence_gap: the training loop restricts its valence
                # metrics to errFD face-detected rows (--valence_face_gate default errfd) and scatters the
                # rest to a -1 fill value. Here every image is scored (selector=None), so there are no fill
                # rows to mask out and the mean runs over all val_keep images -- matching the CLIP eval
                # convention directly above. Three further differences make
                # "train_valence_gap - errVal/gap" NOT a generalisation gap: eval uses held-out test
                # occupations, EMA rather than live weights, and a different sample size.
                if score_err_gen:
                    p_pos_e = probs_ev_all[:, 1]                                 # P(positive), training scorer
                    err_gap = ((p_pos_e >= 0.5).float().mean() - (p_pos_e < 0.5).float().mean()).item()
                    logs_i["errVal/gap"].append(err_gap)
                    logs_i["errVal/gap_abs"].append(abs(err_gap))
                    logs_i["errVal/pred_between_0.2_0.8"].append(
                        ((p_pos_e >= 0.2) * (p_pos_e <= 0.8)).float().mean().item()
                    )
                    # Standardised units: score = sum_p w_p * (g_p - m_p)/s_p, so these move with
                    # --valence_axis_scale / --valence_axis_center and are not comparable across
                    # differently-calibrated runs. Same caveat applies to pred_between_0.2_0.8, which is a
                    # pure function of --valence_tau (a scorer-confidence readout, not a fairness number).
                    logs_i["errVal/score_mean"].append(scores_ev_all.float().mean().item())
                    logs_i["errVal/score_std"].append(
                        scores_ev_all.float().std().item() if scores_ev_all.numel() > 1 else 0.0
                    )
                    # AGREEMENT WITH THE INDEPENDENT HEAD. Legal only because both heads scored the identical,
                    # identically-ordered rows (images_all, post-gather, post-val_keep). A falling agreement
                    # while errVal/gap improves is the signature of the model gaming its own scorer.
                    logs_i["errVal/agree_with_clip"].append(
                        (preds_ev_all == preds_valence_all).float().mean().item()
                    )
                    logs_i["errVal/gap_minus_clip_gap"].append(err_gap - valence_gap)

                    # PER-AXIS HEALTH. A DEAD axis: |mean gap| ~ 0, std ~ the Monte-Carlo noise floor,
                    # |corr_pooled| ~ 0. A DOMINATING axis: |gap| an order of magnitude above the others.
                    # This project has twice shipped a scorer that carried no signal; per-component logging
                    # is the only thing that would have caught it.
                    _score_e = scores_ev_all.float()
                    _z_e = (gaps_ev_all.float() - valence_center.view(1, -1)) / valence_scale.view(1, -1)
                    for _p, _ax in enumerate(valence_axes):
                        _g = gaps_ev_all[:, _p].float()
                        logs_i[f"errVal/gap_{_ax['name']}_mean"].append(_g.mean().item() if _g.numel() > 0 else 0.0)
                        logs_i[f"errVal/gap_{_ax['name']}_std"].append(_g.std().item() if _g.numel() > 1 else 0.0)
                        if _g.numel() > 1 and _g.std() > 0 and _score_e.std() > 0:
                            _c = torch.corrcoef(torch.stack([_g, _score_e]))[0, 1].item()
                        else:
                            _c = 0.0
                        logs_i[f"errVal/gap_{_ax['name']}_corr_pooled"].append(_c)
                        logs_i[f"errVal/z_{_ax['name']}_mean"].append(
                            _z_e[:, _p].mean().item() if _z_e.shape[0] > 0 else 0.0
                        )

                if score_err_ori:
                    # The frozen model on the same prompts/noises: constant in expectation, so it is the
                    # noise floor. The scorer redraws eps on every call, so errVal/gap alone is stochastic
                    # even on fixed images -- the DELTA is what carries the training signal.
                    p_pos_e_ori = probs_ev_ori_all[:, 1]
                    err_gap_ori = (
                        (p_pos_e_ori >= 0.5).float().mean() - (p_pos_e_ori < 0.5).float().mean()
                    ).item()
                    logs_i["errVal/gap_ori"].append(err_gap_ori)
                    logs_i["errVal/gap_delta_ft_minus_ori"].append(err_gap - err_gap_ori)

                # CLIP-T / CLIP-I / DINO via open_clip bigG + eval DINOv2 on the MAIN PROCESS ONLY,
                # over the already all-gathered & val_keep-truncated images_all / images_ori_all
                # (keeps the val_images_per_prompt image count unchanged). Matches file B.
                clip_feats_eval = get_clip_feat_eval(images_all)
                clip_feats_ori_eval = get_clip_feat_eval(images_ori_all)
                text_feats_eval = get_clip_text_feat_eval(prompt_i)
                sims_clip_t_all = (clip_feats_eval * text_feats_eval).sum(dim=-1)
                sims_clip_i_all = (clip_feats_eval * clip_feats_ori_eval).sum(dim=-1)

                # DINO: resize with transforms.Resize(224) (shorter-side, like file B) then normalize.
                # For the square 512x512 SD eval images this equals the old Resize((224,224)).
                images_small_gen_eval = transforms.Resize(224)(images_all)
                images_small_ori_eval = transforms.Resize(224)(images_ori_all)
                dino_feats_eval = get_dino_feat_eval(images_small_gen_eval)
                dino_feats_ori_eval = get_dino_feat_eval(images_small_ori_eval)
                sims_dino_all = (dino_feats_eval * dino_feats_ori_eval).sum(dim=-1)

                logs_i["CLIP-T"].append(sims_clip_t_all.mean().item())
                logs_i["CLIP-I"].append(sims_clip_i_all.mean().item())
                logs_i["DINO"].append(sims_dino_all.mean().item())


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
                        imgs_dict[key] = [wandb.Image(
                            data_or_path=values[0],
                            caption=prompt_i,
                        )]
                    else:
                        imgs_dict[key].append(wandb.Image(
                            data_or_path=values[0],
                            caption=prompt_i,
                        ))
            for key, imgs in imgs_dict.items():
                wandb_tracker.log(
                    {f"eval_{name}_{key}": imgs},
                    step=current_global_step
                    ) 
        
        return logs, log_imgs
    
    def apply_grad_hook_face(images, face_bboxs, face_bboxs_ori, targets, preds_gender_ori, probs_gender_ori, factor=0.1):
        """apply gradient hook on non-face regions of the generated images
        """
        images_new = []
        for image, face_bbox, face_bbox_ori, target, pred_gender_ori, prob_gender_ori in itertools.zip_longest(images, face_bboxs, face_bboxs_ori, targets, preds_gender_ori, probs_gender_ori):
            if (face_bbox == -1).all():
                images_new.append(image.unsqueeze(dim=0))
            else:
                img_width, img_height = image.shape[1:]
                idx_left = max(face_bbox[0], face_bbox_ori[0], 0)
                idx_right = min(face_bbox[2], face_bbox_ori[2], img_width)
                idx_bottom = max(face_bbox[1], face_bbox_ori[1], 0)
                idx_top = min(face_bbox[3], face_bbox_ori[3], img_height)

                img_face = image[:,idx_bottom:idx_top,idx_left:idx_right].clone()
                if target==-1:
                    grad_hook = make_grad_hook(factor)
                elif target==pred_gender_ori:
                    grad_hook = make_grad_hook(1)
                elif target!=pred_gender_ori:
                    grad_hook = make_grad_hook(factor)
                img_face.register_hook(grad_hook)

                img_add = torch.zeros_like(image)
                img_add[:,idx_bottom:idx_top,idx_left:idx_right] = img_face

                mask = torch.zeros_like(image)
                mask[:,idx_bottom:idx_top,idx_left:idx_right] = 1

                image = mask*img_add + (1-mask)*image
                images_new.append(image.unsqueeze(dim=0))

        images_new = torch.cat(images_new)
        return images_new
    
    def gen_dynamic_weights(valid_indicators, targets, preds_ori, factor=0.2):
        """Per-sample scalar on the SCR image-preservation loss.

        RELEASED samples -- those the fair loss wants to CHANGE (target != pred_ori) or that are too
        uncertain to have a target (target == -1) -- get their preservation loss damped to `factor`, so the
        model is allowed to move them. Samples already on target keep weight 1 (preserve them fully).

        This is now the SINGLE release damping. The gender file additionally damped the SAME release set
        spatially by factor2 inside the attention region (0.2 x 0.2 = 0.04 of the kept gradient); that
        spatial gate is gone for valence (there is no coherent region for a scene-level class).

        BEHAVIOUR CHANGE vs the gender file, stated explicitly: its no-face branch returned weight 1
        UNCONDITIONALLY, i.e. a no-face image had its preservation loss fully applied and BYPASSED the
        flip/uncertain damping entirely. With --valence_face_gate none every sample is valid, so that bypass
        no longer fires and every released sample is actually released. `valid_indicators` is kept only so
        --valence_face_gate errfd reproduces the old behaviour exactly.
        """
        weights = []
        for valid, target, pred_ori in itertools.zip_longest(valid_indicators, targets, preds_ori):
            if (valid == False).all():
                weights.append(1)                       # not scored -> nothing to release
            elif target == -1:                          # too uncertain to have a target -> release
                weights.append(factor)
            elif target == pred_ori:                    # already on target -> preserve fully
                weights.append(1)
            else:                                       # flip -> release
                weights.append(factor)

        return torch.tensor(weights, dtype=weight_dtype, device=accelerator.device)

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
        noises_val = torch.randn(
        [len(prompts_val), args.val_images_per_prompt_GPU,4,64,64],
        dtype=weight_dtype_high_precision
        ).to(accelerator.device)
        # "main" (current, non-EMA weights) evaluation is disabled on purpose; only EMA is evaluated below.
        # evaluate_process(text_encoder, unet, "main", prompts_val, noises_val, current_step)

        # evaluate EMA as well
        if args.train_text_encoder:
            text_encoder_lora_dict_copy = copy.deepcopy(text_encoder_lora_dict)
            load_state_dict_results = text_encoder.load_state_dict(text_encoder_lora_ema_dict, strict=False)
        
        if args.train_unet:
            with torch.no_grad():
                unet_lora_layers_copy = copy.deepcopy(unet_lora_layers)
                for p, p_from in itertools.zip_longest(list(unet_lora_layers.parameters()), unet_lora_ema.shadow_params):
                    p.data = p_from.data
            
        evaluate_process(text_encoder, unet, "EMA", prompts_val, noises_val, current_step)
        
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

    # upload the exact code file + parser/config files that produced this run to wandb
    if accelerator.is_main_process:
        try:
            wandb_tracker.log_code(
                root=os.path.dirname(os.path.abspath(__file__)),
                include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml") or path.endswith(".yml"),
            )
            # also explicitly save the running script and the resolved config file
            wandb.save(os.path.abspath(__file__), policy="now")
            if getattr(args, "config", None) and os.path.exists(args.config):
                wandb.save(os.path.abspath(args.config), policy="now")
        except Exception as e:
            logger.warning(f"wandb code/config upload failed: {e}")

    # ensures the start-of-run evaluation (fresh or resumed) runs exactly once
    first_step_eval_done = False
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, data_idx in enumerate(train_dataloader_idxs[epoch]):

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                continue

            # One-time evaluation before taking the first optimization step of this run.
            #   - resumed run (global_step > 0): always evaluate at the resumed step FIRST,
            #     so we get metrics for that checkpoint before continuing training.
            #   - fresh run   (global_step == 0): honor --skip_first_eval as before.
            if not first_step_eval_done:
                first_step_eval_done = True
                if args.resume_from_checkpoint and global_step > 0:
                    accelerator.print(f"Resumed from checkpoint: running evaluation at step {global_step} before continuing training.")
                    evaluation_step(global_step)
                elif global_step == 0 and not args.skip_first_eval:
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
            # logs = []
            # log_imgs = []

            # print noise to check if they are different by device
            noises_i_all = [noises_i.detach().clone() for i in range(accelerator.num_processes)]
            torch.distributed.all_gather(noises_i_all, noises_i)
            if accelerator.is_main_process:
                now = datetime.now(my_timezone)
                accelerator.print(
                    f"{now.strftime('%Y/%m/%d - %H:%M:%S')} --- epoch: {epoch}, step: {step}, prompt: {prompt_i}\n" +
                    " ".join([f"\tprocess idx: {idx}; noise: {noises_i_all[idx].flatten()[-1].item():.4f};" for idx in range(len(noises_i_all))])
                    )
            
            if accelerator.is_main_process:
                logs_i = {
                    "loss_fair": [],
                    "loss_SRR": [],
                    "loss_SCR": [],
                    "loss": [],
                    "valence_gap": [],
                    "valence_gap_abs": [],
                    "valence_pred_between_0.2_0.8": [],
                }
                # PER-AXIS diagnostics. Without these, a dead axis or a single dominating axis is invisible:
                # the aggregate loss curve looks the same either way, which is exactly how this project's two
                # previous non-discriminative scorers survived entire training runs unnoticed.
                for _ax in valence_axes:
                    logs_i[f"gap_{_ax['name']}_mean"] = []
                    logs_i[f"gap_{_ax['name']}_std"] = []
                    logs_i[f"gap_{_ax['name']}_corr_pooled"] = []
                log_imgs_i = {}

            num_denoising_steps = random.choices(range(19,24), k=1)
            torch.distributed.broadcast_object_list(num_denoising_steps, src=0)
            num_denoising_steps = num_denoising_steps[0]

            with torch.no_grad():
                ################################################
                # step 1: generate all images using the diffusion model being finetuned
                images = []
                z0s = []
                N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
                for j in range(N):
                    noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                    images_ij, z0_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet, return_latents=True)
                    images.append(images_ij)
                    z0s.append(z0_ij)
                images = torch.cat(images)
                z0 = torch.cat(z0s)

                # VALENCE: by DEFAULT (--valence_face_gate errfd) only face-detected images are scored and
                # participate; get_valence scatters non-face rows to the -1 fill value. --valence_face_gate
                # none scores EVERY image instead (valid_indicators all-True, no -1 rows).
                valid_indicators = valence_valid_indicators(z0)
                preds_val, probs_val, logits_val, score_val, gaps_val = get_valence(
                    z0, selector=(valid_indicators if args.valence_face_gate == "errfd" else None), fill_value=-1
                )

                images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
                preds_val_all = customized_all_gather(preds_val, accelerator, return_tensor_other_processes=False)
                probs_val_all = customized_all_gather(probs_val, accelerator, return_tensor_other_processes=False)
                # the ranking key must reach rank 0, so the pooled score is now a GATHERED tensor too
                score_val_all = customized_all_gather(score_val, accelerator, return_tensor_other_processes=False)
                gaps_val_all = customized_all_gather(gaps_val, accelerator, return_tensor_other_processes=False)
                valid_all = customized_all_gather(valid_indicators, accelerator, return_tensor_other_processes=False)
                if args.valence_face_gate == "errfd":
                    accelerator.print(f"\tNum faces detected (errFD gate): {valid_all.sum().item()}/{valid_all.shape[0]}.")

                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_generated.jpg")
                        plot_in_grid(images_all, save_to, preds_class=preds_val_all,
                                     pred_class_probs=probs_val_all.max(dim=-1).values)
                        log_imgs_i["img_generated"] = [save_to]

                if accelerator.is_main_process:
                    # Restrict every logged metric to the SCORED population. Under --valence_face_gate errfd
                    # (default) get_valence scatters non-face rows to the -1 fill value, and a -1 must NOT enter
                    # any statistic below: as a prob it reads "negative" (biasing valence_gap), and as a gap /
                    # score it poisons the per-axis mean/std/corr. valid_all (the gather of the face mask) is
                    # the correct "which rows are real" selector -- filtering on `!= -1` would be wrong because
                    # a real gap/score of -1 is legal. Under --valence_face_gate none valid_all is all-True, so
                    # this filter is a no-op and the numbers match the un-gated run exactly.
                    _m = valid_all.bool()
                    _nsc = int(_m.sum().item())
                    p_pos = probs_val_all[_m][:, 1] if _nsc > 0 else probs_val_all[:0, 1]
                    valence_gap = (
                        ((p_pos >= 0.5).float().mean() - (p_pos < 0.5).float().mean()).item() if _nsc > 0 else 0.0
                    )
                    logs_i["valence_gap"].append(valence_gap)
                    logs_i["valence_gap_abs"].append(abs(valence_gap))
                    logs_i["valence_pred_between_0.2_0.8"].append(
                        ((p_pos >= 0.2) * (p_pos <= 0.8)).float().mean().item() if _nsc > 0 else 0.0
                    )
                    # PER-AXIS DIAGNOSTICS -- the whole point of logging these. A DEAD axis shows up as
                    # |mean gap| ~ 0 with std ~ the Monte-Carlo noise floor and |corr| ~ 0 with the pooled
                    # score; a DOMINATING axis shows up as a |gap| an order of magnitude above the others.
                    # This project's two previous dead scorers hid behind a flat aggregate loss curve for
                    # entire training runs precisely because no per-component signal was ever logged.
                    _score_m = score_val_all[_m].float() if _nsc > 0 else score_val_all[:0].float()
                    for _p, _ax in enumerate(valence_axes):
                        _g = gaps_val_all[_m][:, _p].float() if _nsc > 0 else gaps_val_all[:0, _p].float()
                        logs_i[f"gap_{_ax['name']}_mean"].append(_g.mean().item() if _g.numel() > 0 else 0.0)
                        logs_i[f"gap_{_ax['name']}_std"].append(_g.std().item() if _g.numel() > 1 else 0.0)
                        if _g.numel() > 1 and _g.std() > 0 and _score_m.std() > 0:
                            _c = torch.corrcoef(torch.stack([_g, _score_m]))[0, 1].item()
                        else:
                            _c = 0.0
                        logs_i[f"gap_{_ax['name']}_corr_pooled"].append(_c)

                ################################################
                # Step 2: generate dynamic targets
                # Ranked on the CONTINUOUS pooled score, not on probs[:,1] -- see generate_dynamic_targets.
                # Still broadcast from rank 0, so every rank uses byte-identical targets.
                targets_all, uncertainty_all = generate_dynamic_targets(
                    score_val_all, valid_all, target_ratio=0.5, w_uncertainty=True
                )
                torch.distributed.broadcast(targets_all, src=0)
                torch.distributed.broadcast(uncertainty_all, src=0)

                targets_all[uncertainty_all>args.uncertainty_threshold] = -1
                _n_local = preds_val.shape[0]
                targets = targets_all[_n_local*(accelerator.local_process_index):_n_local*(accelerator.local_process_index+1)]
                uncertainty = uncertainty_all[_n_local*(accelerator.local_process_index):_n_local*(accelerator.local_process_index+1)]
                accelerator.print(f"\tNum samples to compute grads: {(targets_all!=-1).sum().item()}/{targets_all.shape[0]}")

                ################################################
                # Step 3: generate all original images using the original diffusion model
                # note that only targets from above will be used to compute loss
                # all other variables will not be used below
                images_ori = []
                z0_ori_list = []
                N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
                for j in range(N):
                    noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                    if args.train_text_encoder and args.train_unet:
                        images_ij, z0_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=eval_unet, return_latents=True)
                    elif args.train_text_encoder and not args.train_unet:
                        images_ij, z0_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=unet, return_latents=True)
                    elif not args.train_text_encoder and args.train_unet:
                        images_ij, z0_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=eval_unet, return_latents=True)
                    images_ori.append(images_ij)
                    z0_ori_list.append(z0_ij)
                images_ori = torch.cat(images_ori)
                z0_ori = torch.cat(z0_ori_list)

                # VALENCE of the ORIGINAL (frozen-model) images. preds_valence_ori is the "where the sample
                # currently sits" reference that decides the SCR release set (flip vs keep) below.
                valid_indicators_ori = valence_valid_indicators(z0_ori)
                preds_val_ori, probs_val_ori, logits_val_ori, score_val_ori, gaps_val_ori = get_valence(
                    z0_ori, selector=(valid_indicators_ori if args.valence_face_gate == "errfd" else None), fill_value=-1
                )

                # SCR: the image loss no longer uses CLIP/DINO features; it is computed in Step 4 from the
                # FROZEN scoring UNet's mid_block (h-space) on re-noised z0_ori vs z0_ft. (CLIP-I / DINO-I are
                # still computed as eval metrics.) z0_ori (the detached SCR target) is already retained above.

                images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
                preds_val_ori_all = customized_all_gather(preds_val_ori, accelerator, return_tensor_other_processes=False)
                probs_val_ori_all = customized_all_gather(probs_val_ori, accelerator, return_tensor_other_processes=False)

                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_ori.jpg")
                        plot_in_grid(images_ori_all, save_to, preds_class=preds_val_ori_all,
                                     pred_class_probs=probs_val_ori_all.max(dim=-1).values)

                        log_imgs_i["img_ori"] = [save_to]
            
            ################################################
            # Step 4: compute loss
            loss_fair_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_SRR_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_SCR_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)

            idxs_i = list(range(targets.shape[0]))
            N_backward = math.ceil(targets.shape[0] / args.train_GPU_batch_size)
            # SCR: fixed feature-extractor conditioning = generation prompt encoded by the FROZEN original TE,
            # and the SCR-specific timestep grid (both ori & ft re-noised to these same t). This grid is
            # INDEPENDENT of the woman/man class-error + SRR-realism scorer (which uses --residual_t_min/max,
            # 400-800): the SCR h-space MSE uses the lower-noise --scr_t_min/max (default 100-400).
            scr_gen_embeds = _encode_scoring_prompt(prompt_i)                                     # [1, L, D], frozen
            scr_timesteps = torch.linspace(
                args.scr_t_min, args.scr_t_max, steps=args.scr_num_timesteps, device=accelerator.device
            ).round().long()
            for j in range(N_backward):
                idxs_ij = idxs_i[j*args.train_GPU_batch_size:(j+1)*args.train_GPU_batch_size]
                noises_ij = noises_i[idxs_ij]
                targets_ij = targets[idxs_ij]
                preds_val_ori_ij = preds_val_ori[idxs_ij]

                images_ij, z0_ij = generate_image_w_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet, return_latents=True)
                valid_ij = valence_valid_indicators(z0_ij)
                # Branch B: fused residual scorer on z0_ij (grad flows z0 -> trainable model).
                #   - logits_val_ij [n,2]: pooled multi-prompt valence logits  -> the fair (CE) loss
                #   - score_val_ij  [n]  : the pooled continuous score (logged)
                #   - gaps_val_ij   [n,P]: per-axis gaps, detached (logged -- a dead/dominating axis is
                #                          visible here and nowhere else)
                #   - loss_SRR_ij   [n]  : raw residual error of --srr_prompt (SRR realism loss)
                logits_val_ij, score_val_ij, gaps_val_ij, loss_SRR_ij = residual_valence_and_realism(z0_ij)

                # Branch A: SCR image loss (scoring-space, FROZEN feature extractor).
                #   Re-noise the ORIGINAL (z0_ori) and FINETUNE (z0_ij) latents to the SAME zt (shared eps & t)
                #   and MSE the FROZEN scoring UNet's mid_block (h-space) output under the frozen generation
                #   prompt. grad: h_ft -> zt_ft -> z0_ij -> generation; h_ori is a detached target.
                #
                #   RELEASE SET: a sample is RELEASED (preservation damped to --factor1) when it must FLIP
                #   (target != pred_ori) or is too uncertain to have a target (target == -1). That decision now
                #   lives in exactly ONE place, gen_dynamic_weights below -- the gender file made it twice, in
                #   two different forms, which is what let the double-damping below go unnoticed.
                #
                #   THE SPATIAL GATE IS GONE. The gender file ALSO multiplied the SCR gradient by --factor2
                #   inside the class attention region for exactly this same release set -- so a released
                #   sample was damped TWICE (factor1 globally x factor2 in-region = 0.2 x 0.2 = 0.04). For a
                #   scene-level class there is no region to gate on ("where the class lives" is the whole
                #   frame), so the spatial mask is vacuous and only factor1 remains. No hook on zt_ft.
                z0_ori_ij = z0_ori[idxs_ij]
                scr_gen_embeds_ij = scr_gen_embeds.expand(len(idxs_ij), -1, -1)

                scr_mid_store = []
                def _scr_mid_hook(_m, _in, _out):
                    scr_mid_store.append(_out)
                _scr_h = scoring_unet.mid_block.register_forward_hook(_scr_mid_hook)
                scr_per_t = []
                for _t in scr_timesteps:
                    _tb = _t.repeat(len(idxs_ij))
                    _eps = torch.randn_like(z0_ij)
                    zt_ft = noise_scheduler.add_noise(z0_ij, _eps, _tb)                             # grad -> z0_ij
                    zt_ori = noise_scheduler.add_noise(z0_ori_ij, _eps, _tb)                        # detached target input
                    scr_mid_store.clear()
                    _ = scoring_unet(zt_ft.to(weight_dtype), _tb, encoder_hidden_states=scr_gen_embeds_ij).sample
                    h_ft = scr_mid_store[-1]
                    scr_mid_store.clear()
                    with torch.no_grad():
                        _ = scoring_unet(zt_ori.to(weight_dtype), _tb, encoder_hidden_states=scr_gen_embeds_ij).sample
                        h_ori = scr_mid_store[-1].detach()
                    scr_per_t.append(((h_ft.to(weight_dtype_high_precision) - h_ori.to(weight_dtype_high_precision)) ** 2).mean(dim=[1, 2, 3]))
                _scr_h.remove()
                loss_SCR_ij = torch.stack(scr_per_t, dim=0).mean(dim=0).to(weight_dtype)

                # ---- fair loss (CE on the pooled valence logits) over the samples that HAVE a target.
                # With --valence_face_gate none, `valid_ij` is all-True, so participation is decided purely
                # by target validity (i.e. by the uncertainty threshold) -- which is what it should be: every
                # image has a valence, so no image is excluded for lacking a face.
                idxs_w_fair_loss = ((valid_ij == True) * (targets_ij != -1)).nonzero().view([-1])

                # NOTE the two separate tensors. The gender file kept ONE `loss_fair_ij` pre-filled with -1
                # and then ADDED it straight into the optimized loss, so every non-participating row
                # contributed a constant -1 to loss_ij.mean(). That is gradient-free (a constant), so it never
                # corrupted training -- but it silently biased the REPORTED loss downward by
                # (#excluded / #total). Here the backward tensor is zero-filled (a true no-op) and the -1
                # sentinel survives only in the LOGGING tensor, where the `!= -1` filter below expects it.
                loss_fair_bwd = torch.zeros(len(idxs_ij), dtype=weight_dtype, device=accelerator.device)
                loss_fair_log = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) * (-1)
                if idxs_w_fair_loss.numel() > 0:
                    _ce = CE_loss(logits_val_ij[idxs_w_fair_loss], targets_ij[idxs_w_fair_loss])
                    loss_fair_bwd[idxs_w_fair_loss] = _ce.to(weight_dtype)
                    loss_fair_log[idxs_w_fair_loss] = _ce.detach().to(weight_dtype)

                # SRR realism loss: raw residual error of --srr_prompt (no 1/tau). Applies to ALL generated
                # samples (like the old CLIP/DINO image loss), not gated on target validity.
                loss_SRR_ij = loss_SRR_ij.to(weight_dtype)

                dynamic_weights = gen_dynamic_weights(valid_ij, targets_ij, preds_val_ori_ij, factor=args.factor1)
                loss_ij = loss_fair_bwd + args.weight_loss_scr * dynamic_weights * loss_SCR_ij + args.weight_loss_face * loss_SRR_ij
                accelerator.backward(loss_ij.mean())
                loss_fair_ij = loss_fair_log

                with torch.no_grad():
                    loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)
                    loss_SRR_i[idxs_ij] = loss_SRR_ij.to(loss_SRR_i.dtype)
                    loss_SCR_i[idxs_ij] = loss_SCR_ij.to(loss_SCR_i.dtype)
                    loss_i[idxs_ij] = loss_ij.to(loss_i.dtype)
                    
            # for logging purpose, gather all losses to main_process
            accelerator.wait_for_everyone()
            loss_fair_all = customized_all_gather(loss_fair_i, accelerator)
            loss_SRR_all = customized_all_gather(loss_SRR_i, accelerator)
            loss_SCR_all = customized_all_gather(loss_SCR_i, accelerator)
            loss_all = customized_all_gather(loss_i, accelerator)

            loss_all = loss_all[loss_fair_all!=-1]
            loss_fair_all = loss_fair_all[loss_fair_all!=-1]
            loss_SRR_all = loss_SRR_all[loss_SRR_all!=-1]

            if accelerator.is_main_process:
                logs_i["loss_fair"].append(loss_fair_all)
                logs_i["loss_SRR"].append(loss_SRR_all)
                logs_i["loss_SCR"].append(loss_SCR_all)
                logs_i["loss"].append(loss_all)

            # process logs
            if accelerator.is_main_process:
                for key in ["loss_fair", "loss_SRR", "loss_SCR", "loss"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)
                    else:
                        logs_i[key] = torch.cat(logs_i[key])
                _scalar_keys = ["valence_gap", "valence_gap_abs", "valence_pred_between_0.2_0.8"]
                _scalar_keys += [f"gap_{_ax['name']}_{_s}" for _ax in valence_axes
                                 for _s in ("mean", "std", "corr_pooled")]
                for key in _scalar_keys:
                    if logs_i.get(key) == []:
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
            # we mannually sync grads
            # accelerator.wait_for_everyone()
            grad_is_finite = True
            with torch.no_grad():
                if args.train_text_encoder:
                    for p in text_encoder_lora_model.parameters():
                        # SCR: the h-space loss reaches trainable params only through z0 (mid_block target),
                        # and on a rank whose batch has no valid face the loss_fair term contributes no grad.
                        # Zero-fill None grads so every rank all_reduces every param in the same order
                        # -> no NCCL desync/deadlock (grads are synced manually here).
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        if not torch.isfinite(p.grad).all():
                            grad_is_finite = False
                        torch.distributed.all_reduce(p.grad, torch.distributed.ReduceOp.SUM)
                        p.grad = p.grad / accelerator.num_processes / N_backward
                if args.train_unet:
                    for p in unet_lora_layers.parameters():
                        # SCR: see note above -- zero-fill None grads to keep the per-parameter all_reduce
                        # collective aligned across ranks.
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
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





# accelerate launch --config_file configs/accelerate_config.yaml 1-main-debias.py --config configs/debias-text-encoder.yaml
# accelerate launch --config_file configs/accelerate_config.yaml 1-main-debias.py --config configs/debias-unet.yaml
# accelerate launch --config_file configs/accelerate_config.yaml 1-main-debias.py --config configs/debias-text-encoder-and-unet.yaml