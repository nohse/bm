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

def plot_in_grid(images, save_to, face_indicators=None, face_bboxs=None, preds_gender=None, pred_class_probs_gender=None):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """
    images_w_face = images[face_indicators]
    images_wo_face = images[face_indicators.logical_not()]

    # first reorder everything from most to least male, from most to least female, and finally images without faces
    idxs_male = (preds_gender == 1).nonzero(as_tuple=False).view([-1])
    probs_male = pred_class_probs_gender[idxs_male]
    idxs_male = idxs_male[probs_male.argsort(descending=True)]

    idxs_female = (preds_gender == 0).nonzero(as_tuple=False).view([-1])
    probs_female = pred_class_probs_gender[idxs_female]
    idxs_female = idxs_female[probs_female.argsort(descending=True)]

    idxs_no_face = (preds_gender == -1).nonzero(as_tuple=False).view([-1])

    images_to_plot = []
    idxs_reordered = torch.torch.cat([idxs_male, idxs_female, idxs_no_face])
    
    for idx in idxs_reordered:
        img = images[idx]
        face_indicator = face_indicators[idx]
        face_bbox = face_bboxs[idx]
        pred_gender = preds_gender[idx]
        pred_class_prob_gender = pred_class_probs_gender[idx]
        
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
        img_pil_draw.rectangle(face_bbox.tolist(), fill =None, outline =border_color, width=4)

        img_pil = ImageOps.expand(img_pil, border=(50,0,0,0),fill=border_color)

        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_gender.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_gender.item())*512)], fill ="white", outline =None)

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
        default="./outputs/gender-debias-text-encoder-again/BS-24_TE_tau-0.0001_resT-15-400-800_wImg-8-0.2-0.2_wFace-1_skipFrac-0.5_Th-0.2_loraR-50_lr-5e-05_07081409/ckpts/checkpoint_tmp-1180",
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
        '--skip_first_eval',
        action="store_true",
        default=True,
        help="if set, skip the evaluation at global_step==0 and start training directly",
        )
    parser.add_argument(
        '--save_attn_maps',
        action="store_true",
        default=True,
        help="if set, at every --train_plot_every_n_iter step save the woman/man/common cross-attention "
             "maps used by the residual gender scorer, overlaid on the generated images",
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
        help="temperature for the residual-error gender logits: logit_c = -E_c / tau",
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
        '--skip_denoise_frac',
        default=0.5,
        type=float,
        help="fraction of the denoising trajectory to skip when generating images. "
             "0.0 (default) = full sequential denoising, identical to the original behavior. "
             "If >0, only the first round((1-frac)*num_denoising_steps) scheduler steps run "
             "sequentially; at the last of those steps we jump straight to a predicted clean "
             "latent x0 via the closed-form epsilon->x0 formula "
             "x0 = (z_t - sqrt(1-abar_t)*eps) / sqrt(abar_t) (no extra UNet call). "
             "Applied identically to every image-generation pass (current-model, original/frozen "
             "model, and the gradient pass), so reference and trained images stay on equal footing. "
             "NOTE: the resulting x0 is an approximation (blurrier than full denoising).",
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
        for key, value in config_data.items():
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
    folder_name = (
        f"BS-{args.train_images_per_prompt_GPU*accelerator.num_processes}"
        f"_{_model_tag}"
        f"_tau-{args.tau:g}"
        f"_resT-{args.residual_num_timesteps}-{args.residual_t_min}-{args.residual_t_max}"
        f"_wImg-{args.weight_loss_img}-{args.factor1}-{args.factor2}"
        f"_wFace-{args.weight_loss_face}"
        f"{('_skipFrac-'+format(args.skip_denoise_frac, 'g')) if args.skip_denoise_frac>0 else ''}"
        f"_Th-{args.uncertainty_threshold}"
        f"_loraR-{args.rank}_lr-{args.learning_rate}"
        f"_{timestring}"
    )
    
    args.imgs_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "imgs")
    args.ckpts_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "ckpts")

    if accelerator.is_main_process:
        os.makedirs(args.imgs_save_dir, exist_ok=True)
        os.makedirs(args.ckpts_save_dir, exist_ok=True)
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

    # install the capturing processor on the scoring UNet's cross-attention (attn2) layers only
    _scoring_attn_procs = dict(scoring_unet.attn_processors)
    for _name in list(_scoring_attn_procs.keys()):
        if _name.endswith("attn2.processor"):
            _scoring_attn_procs[_name] = CrossAttnCaptureProcessor(attn_capture_ctx)
    scoring_unet.set_attn_processor(_scoring_attn_procs)

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
                
    def residual_gender_logits(z0):
        """Prompt-conditioned diffusion residual-error gender scorer with cross-attention spatial weighting.

        z0: [n,4,H,W] clean latent in the scheduler/UNet scale. May require grad; it is NOT detached,
            so gradient flows: logits -> weighted residual err -> scoring UNet(zt) -> zt -> z0.

        For each of `--residual_num_timesteps` timesteps (linearly spaced in
        [--residual_t_min, --residual_t_max] inclusive) a fresh eps is sampled and the SAME eps/zt is
        scored under both the woman and man text conditions. Rather than averaging the squared eps error
        uniformly over space, it is weighted by a single COMMON cross-attention map:
          - for every (timestep, prompt, cross-attn block, head) the class-token ("woman"/"man")
            cross-attention map is extracted, resized to (H,W), and averaged over ALL of those axes;
          - the common map is spatially normalized to sum to 1 and detached (pure weighting mask);
          - the per-pixel squared error (channel-mean, [n,H,W]) is reduced by a spatial weighted SUM
            with the common map (not mean, because the map already sums to 1).

        Returns logits [n,2], class order [woman=0, man=1]: logit_c = -E_c / tau.

        [PERF] The `--residual_num_timesteps` timesteps are folded into the batch dimension, so the
        scoring UNet runs ONCE per prompt on an [n*K, ...] batch instead of K sequential [n, ...]
        forwards. A fresh eps is still drawn per timestep in the same order and the attention/residual
        reductions are unchanged, so the estimator is mathematically identical to the per-timestep loop
        (only GPU batching differs). Verified numerically equal (logits + grad) to the looped version.
        """
        n = z0.shape[0]
        H, W = z0.shape[-2], z0.shape[-1]
        timesteps = torch.linspace(
            args.residual_t_min, args.residual_t_max, steps=args.residual_num_timesteps, device=z0.device
        ).round().long()
        K = timesteps.shape[0]

        # Draw a fresh eps per timestep (SAME order/values as the per-timestep loop) and build z_t,
        # then fold the K timesteps into the batch dimension: [n,K,...] -> [n*K,...] (row = i*K + k).
        eps_list, zt_list = [], []
        for t in timesteps:
            t_batch = t.repeat(n)
            eps_k = torch.randn_like(z0)                          # fresh eps per timestep
            zt_list.append(noise_scheduler.add_noise(z0, eps_k, t_batch))
            eps_list.append(eps_k)
        eps_all = torch.stack(eps_list, dim=1).reshape(n * K, *z0.shape[1:])          # [n*K,4,H,W]
        zt_all = torch.stack(zt_list, dim=1).reshape(n * K, *z0.shape[1:]).to(weight_dtype)
        t_all = timesteps.repeat(n)                                                   # [n*K], row i*K+k -> t_k

        prompts_cfg = [
            ("woman", residual_woman_embeds, residual_woman_token_idxs),
            ("man", residual_man_embeds, residual_man_token_idxs),
        ]

        residual_maps = {}                                        # cls -> [n,K,H,W] (grad-carrying)
        attn_accum = torch.zeros(n, H, W, dtype=torch.float, device=z0.device)   # detached attention accumulator
        attn_count = 0

        for cls, embeds, tok_idxs in prompts_cfg:
            c = embeds.expand(n * K, -1, -1)
            attn_capture_ctx.store = []
            attn_capture_ctx.token_idxs = tok_idxs
            attn_capture_ctx.enabled = True
            eps_pred = scoring_unet(zt_all, t_all, encoder_hidden_states=c).sample
            attn_capture_ctx.enabled = False
            captured = attn_capture_ctx.store
            attn_capture_ctx.store = []

            # channel-mean squared residual -> [n*K,H,W] -> [n,K,H,W] (keeps grad to z0)
            residual_maps[cls] = (eps_pred.float() - eps_all.float()).pow(2).mean(dim=1).view(n, K, H, W)

            # accumulate the class-token cross-attention maps (detached), each resized to (H,W).
            # A captured block spans the whole [n*K] batch; averaging over K and counting once per
            # (prompt, block) reproduces the original per-(timestep,prompt,block) accum / attn_count.
            for col, heads in captured:
                hw = col.shape[-1]
                s = int(round(math.sqrt(hw)))
                a = col.view(n * K, heads, s, s).float()                                # [n*K, heads, s, s]
                a = torch.nn.functional.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
                a = a.mean(dim=1).view(n, K, H, W).mean(dim=1)                          # mean heads, then K -> [n,H,W]
                attn_accum = attn_accum + a
                attn_count += 1

        # common attention map: mean over {timesteps, prompts, blocks, heads}
        common_attn = attn_accum / max(attn_count, 1)                                   # [n,H,W]
        # spatial normalization so that sum_{u,v} common_attn(u,v) == 1 per image
        common_attn = common_attn / (common_attn.sum(dim=(1, 2), keepdim=True) + 1e-8)
        common_attn = common_attn.detach()                                             # weighting mask only

        # attention-weighted spatial SUM per timestep, then uniform mean over timesteps
        E_woman = (common_attn.unsqueeze(1) * residual_maps["woman"]).sum(dim=(2, 3)).mean(dim=1)   # [n]
        E_man = (common_attn.unsqueeze(1) * residual_maps["man"]).sum(dim=(2, 3)).mean(dim=1)       # [n]

        logits_gender = torch.stack([-E_woman / args.tau, -E_man / args.tau], dim=1)    # [n, 2]
        return logits_gender
    def get_face_gender(z0, selector=None, fill_value=-1):
        """Drop-in replacement for the removed mnet classifier, now scoring the clean latent z0.

        z0: [B,4,64,64] clean latents (scheduler/UNet scale).
        selector (bool [B]) == face_indicators: only face-detected latents are scored; the rest are
            filled with fill_value (mirrors the previous mnet behaviour so no-face images are excluded).
        Returns (preds_gender, probs_gender, logits_gender), class order [woman=0, man=1].
        Raw logits are returned so callers can feed them straight into cross_entropy (no pre-softmax).
        """
        if selector != None:
            z0_w_faces = z0[selector]
        else:
            z0_w_faces = z0

        if z0_w_faces.shape[0] == 0:
            logits_gender = torch.empty([0,2], dtype=torch.float, device=z0.device)
            probs_gender = torch.empty([0,2], dtype=torch.float, device=z0.device)
            preds_gender = torch.empty([0], dtype=torch.int64, device=z0.device)
        else:
            logits_gender = residual_gender_logits(z0_w_faces)
            probs_gender = torch.softmax(logits_gender, dim=-1)
            preds_gender = probs_gender.max(dim=-1).indices

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
    def compute_gender_attmaps(z0):
        """Visualization-only: aggregate the woman/man class-token cross-attention maps and the common map.

        Mirrors the attention aggregation inside residual_gender_logits but keeps woman/man separate and
        runs under no_grad. Returns (woman_map, man_map, common_map), each [B,H,W] normalized so that the
        spatial values sum to 1 per image (common_map == the actual weighting mask used by the scorer).
        """
        n = z0.shape[0]
        H, W = z0.shape[-2], z0.shape[-1]
        timesteps = torch.linspace(
            args.residual_t_min, args.residual_t_max, steps=args.residual_num_timesteps, device=z0.device
        ).round().long()
        c_woman = residual_woman_embeds.expand(n, -1, -1)
        c_man = residual_man_embeds.expand(n, -1, -1)
        prompts_cfg = [
            ("woman", c_woman, residual_woman_token_idxs),
            ("man", c_man, residual_man_token_idxs),
        ]
        accum = {
            "woman": torch.zeros(n, H, W, dtype=torch.float, device=z0.device),
            "man": torch.zeros(n, H, W, dtype=torch.float, device=z0.device),
        }
        counts = {"woman": 0, "man": 0}
        for t in timesteps:
            t_batch = t.repeat(n)
            eps = torch.randn_like(z0)
            zt_in = noise_scheduler.add_noise(z0, eps, t_batch).to(weight_dtype)
            for cls, c, tok_idxs in prompts_cfg:
                attn_capture_ctx.store = []
                attn_capture_ctx.token_idxs = tok_idxs
                attn_capture_ctx.enabled = True
                _ = scoring_unet(zt_in, t_batch, encoder_hidden_states=c).sample
                attn_capture_ctx.enabled = False
                captured = attn_capture_ctx.store
                attn_capture_ctx.store = []
                for col, heads in captured:
                    hw = col.shape[-1]
                    s = int(round(math.sqrt(hw)))
                    a = col.view(n, heads, s, s).float()
                    a = torch.nn.functional.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
                    accum[cls] = accum[cls] + a.mean(dim=1)
                    counts[cls] += 1
        woman_map = accum["woman"] / max(counts["woman"], 1)
        man_map = accum["man"] / max(counts["man"], 1)
        common_map = (woman_map + man_map) / 2.0   # == common_attn (mean over prompts too), before normalization

        def _norm(m):
            return m / (m.sum(dim=(1, 2), keepdim=True) + 1e-8)
        return _norm(woman_map), _norm(man_map), _norm(common_map)

    def save_gender_attmap_panels(z0, images, save_to, max_imgs=16):
        """Save a grid where each row is [ generated image | woman-attn | man-attn | common-attn ] (overlays)."""
        n = min(z0.shape[0], images.shape[0], max_imgs)
        if n == 0:
            return
        woman_map, man_map, common_map = compute_gender_attmaps(z0[:n])
        imgs = images[:n].detach().cpu()
        labels = ["generated", "woman-attn", "man-attn", "common-attn"]
        rows = []
        for i in range(n):
            base_pil = transforms.ToPILImage()(imgs[i].mul(0.5).add(0.5).clamp(0, 1))
            panels = [
                base_pil,
                attmap_overlay_on_image(woman_map[i], imgs[i]),
                attmap_overlay_on_image(man_map[i], imgs[i]),
                attmap_overlay_on_image(common_map[i], imgs[i]),
            ]
            w, h = base_pil.size
            row = Image.new("RGB", (w * len(panels), h))
            for j, (p, lab) in enumerate(zip(panels, labels)):
                p = p.resize((w, h)).copy()
                ImageDraw.Draw(p).text((5, 5), f"{lab} #{i}", fill="white")
                row.paste(p, (j * w, 0))
            rows.append(row)
        gw, gh = rows[0].size
        grid = Image.new("RGB", (gw, gh * len(rows)))
        for i, r in enumerate(rows):
            grid.paste(r, (0, i * gh))
        if os.path.dirname(save_to) and not os.path.exists(os.path.dirname(save_to)):
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
        grid.save(save_to, quality=92)

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
    def generate_dynamic_targets(probs, target_ratio=0.5, w_uncertainty=False):
        """generate dynamic targets for the distributional alignment loss

        Args:
            probs (torch.tensor): shape [N,2], N points in a probability simplex of 2 dims
            target_ratio (float): target distribution, the percentage of class 1 (male)
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
        for prompt_i, noises_i in itertools.zip_longest(prompts, noises):
            if accelerator.is_main_process:
                logs_i = {
                    "gender_gap": [],
                    "gender_gap_abs": [],
                    "gender_pred_between_0.2_0.8": [],
                    "CLIP-T": [],
                    "CLIP-I": [],
                    "DINO": [],
                }
                log_imgs_i = {}
            ################################################
            # step 1: generate all ori images
            images_ori = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                if args.train_text_encoder and args.train_unet:
                    images_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=eval_unet, skip_denoise_frac=0.0)
                elif args.train_text_encoder and not args.train_unet:
                    images_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=eval_text_encoder, which_unet=unet, skip_denoise_frac=0.0)
                elif not args.train_text_encoder and args.train_unet:
                    images_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=eval_unet, skip_denoise_frac=0.0)
                images_ori.append(images_ij)
            images_ori = torch.cat(images_ori)
            face_indicators_ori, face_bboxs_ori, face_chips_ori, face_landmarks_ori, aligned_face_chips_ori = get_face(images_ori)
            preds_gender_ori, probs_gender_ori, logits_gender_ori = get_face_gender_test(face_chips_ori, selector=face_indicators_ori, fill_value=-1)

            face_feats_ori = get_face_feats(face_feats_net, aligned_face_chips_ori)
            _, face_real_scores_ori = face_feats_model.semantic_search(face_feats_ori, selector=face_indicators_ori, return_similarity=True)

            images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
            face_indicators_ori_all = customized_all_gather(face_indicators_ori, accelerator, return_tensor_other_processes=False)
            face_bboxs_ori_all = customized_all_gather(face_bboxs_ori, accelerator, return_tensor_other_processes=False)
            preds_gender_ori_all = customized_all_gather(preds_gender_ori, accelerator, return_tensor_other_processes=False)
            probs_gender_ori_all = customized_all_gather(probs_gender_ori, accelerator, return_tensor_other_processes=False)
            face_real_scores_ori_all = customized_all_gather(face_real_scores_ori, accelerator, return_tensor_other_processes=False)

            # keep only the first val_keep gathered images (see --val_images_per_prompt_total)
            images_ori_all = images_ori_all[:val_keep]
            face_indicators_ori_all = face_indicators_ori_all[:val_keep]
            face_bboxs_ori_all = face_bboxs_ori_all[:val_keep]
            preds_gender_ori_all = preds_gender_ori_all[:val_keep]
            probs_gender_ori_all = probs_gender_ori_all[:val_keep]
            face_real_scores_ori_all = face_real_scores_ori_all[:val_keep]

            if accelerator.is_main_process:
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_ori.jpg")
                plot_in_grid(
                    images_ori_all, 
                    save_to, 
                    face_indicators=face_indicators_ori_all, face_bboxs=face_bboxs_ori_all, 
                    preds_gender=preds_gender_ori_all, 
                    pred_class_probs_gender=probs_gender_ori_all.max(dim=-1).values,
                    # face_real_scores=face_real_scores_ori_all
                )

                log_imgs_i["img_ori"] = [save_to]

            
            images = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                images_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=which_text_encoder, which_unet=which_unet, skip_denoise_frac=0.0)
                images.append(images_ij)
            images = torch.cat(images)
            
            face_indicators, face_bboxs, face_chips, face_landmarks, aligned_face_chips = get_face(images)
            preds_gender, probs_gender, logits_gender = get_face_gender_test(face_chips, selector=face_indicators, fill_value=-1)

            face_feats = get_face_feats(face_feats_net, aligned_face_chips)
            _, face_real_scores = face_feats_model.semantic_search(face_feats, selector=face_indicators, return_similarity=True)

            images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
            face_indicators_all = customized_all_gather(face_indicators, accelerator, return_tensor_other_processes=False)
            face_bboxs_all = customized_all_gather(face_bboxs, accelerator, return_tensor_other_processes=False)
            preds_gender_all = customized_all_gather(preds_gender, accelerator, return_tensor_other_processes=False)
            probs_gender_all = customized_all_gather(probs_gender, accelerator, return_tensor_other_processes=False)
            face_real_scores = customized_all_gather(face_real_scores, accelerator, return_tensor_other_processes=False)

            # keep only the first val_keep gathered images (see --val_images_per_prompt_total)
            images_all = images_all[:val_keep]
            face_indicators_all = face_indicators_all[:val_keep]
            face_bboxs_all = face_bboxs_all[:val_keep]
            preds_gender_all = preds_gender_all[:val_keep]
            probs_gender_all = probs_gender_all[:val_keep]
            face_real_scores = face_real_scores[:val_keep]

            ################################################
            # eval fidelity / text-alignment metrics (CLIP-T, CLIP-I, DINO) are ALL computed on the
            # MAIN PROCESS ONLY -- open_clip bigG and the eval DINOv2 both live only on rank 0
            # (matches file B). They run over the already all-gathered & val_keep-truncated
            # images_all / images_ori_all in the is_main_process block below, so no per-rank feature
            # pass or extra all-gather is needed here. images_all[k] pairs with images_ori_all[k]
            # (same noises_i, same order).

            if accelerator.is_main_process:
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_generated.jpg")
                plot_in_grid(
                    images_all, 
                    save_to, 
                    face_indicators=face_indicators_all, 
                    face_bboxs=face_bboxs_all, 
                    preds_gender=preds_gender_all, 
                    pred_class_probs_gender=probs_gender_all.max(dim=-1).values,
                    # face_real_scores=face_real_scores
                    )

                log_imgs_i["img_generated"] = [save_to]
            
            if accelerator.is_main_process:
                probs_tmp = probs_gender_all[(probs_gender_all!=-1).all(dim=-1)]
                gender_gap = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                gender_pred_between_02_08 = ((probs_tmp[:,1]>=0.2)*(probs_tmp[:,1]<=0.8)).float().mean().item()
                logs_i["gender_gap"].append(gender_gap)
                logs_i["gender_gap_abs"].append(abs(gender_gap))
                logs_i["gender_pred_between_0.2_0.8"].append(abs(gender_pred_between_02_08))

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
    
    def gen_dynamic_weights(face_indicators, targets, preds_gender_ori, probs_gender_ori, factor=0.2):
        weights = []
        for face_indicator, target, pred_gender_ori, prob_gender_ori in itertools.zip_longest(face_indicators, targets, preds_gender_ori, probs_gender_ori):
            if (face_indicator == False).all():
                weights.append(1)
            else:
                if target==-1:
                    weights.append(factor)
                elif target==pred_gender_ori:
                    weights.append(1)
                elif target!=pred_gender_ori:
                    weights.append(factor)

        weights = torch.tensor(weights, dtype=probs_gender_ori.dtype, device=probs_gender_ori.device)
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

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, data_idx in enumerate(train_dataloader_idxs[epoch]):            
            
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                continue

            if global_step == 0 and not args.skip_first_eval:
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
                    "loss_face": [],
                    "loss_CLIP": [],
                    "loss_DINO": [],
                    "loss": [],
                    "gender_gap": [],
                    "gender_gap_abs": [],
                    "gender_pred_between_0.2_0.8": [],
                }
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

                face_indicators, face_bboxs, face_chips, face_landmarks, aligned_face_chips = get_face(images)
                preds_gender, probs_gender, logits_gender = get_face_gender(z0, selector=face_indicators, fill_value=-1)

                
                face_feats = torch.ones([aligned_face_chips.shape[0],512], dtype=weight_dtype_high_precision, device=aligned_face_chips.device) * (-1)
                if sum(face_indicators)>0:
                    face_feats_ = get_face_feats(face_feats_net, aligned_face_chips[face_indicators])
                    face_feats[face_indicators] = face_feats_

                _, face_real_scores = face_feats_model.semantic_search(face_feats, selector=face_indicators, return_similarity=True)

                face_indicators_all, face_indicators_others = customized_all_gather(face_indicators, accelerator, return_tensor_other_processes=True)
                accelerator.print(f"\tNum faces detected: {face_indicators_all.sum().item()}/{face_indicators_all.shape[0]}.")
                
                images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
                face_bboxs_all = customized_all_gather(face_bboxs, accelerator, return_tensor_other_processes=False)
                preds_gender_all = customized_all_gather(preds_gender, accelerator, return_tensor_other_processes=False)
                probs_gender_all = customized_all_gather(probs_gender, accelerator, return_tensor_other_processes=False)
                face_real_scores_all = customized_all_gather(face_real_scores, accelerator, return_tensor_other_processes=False)
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_generated.jpg")
                        plot_in_grid(images_all, save_to, face_indicators=face_indicators_all, face_bboxs=face_bboxs_all, preds_gender=preds_gender_all, pred_class_probs_gender=probs_gender_all.max(dim=-1).values)

                        log_imgs_i["img_generated"] = [save_to]

                        if args.save_attn_maps:
                            # gender-classification cross-attention maps for this process's generated batch
                            attmap_save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_attmap.jpg")
                            save_gender_attmap_panels(z0, images, attmap_save_to)
                            log_imgs_i["attmap_generated"] = [attmap_save_to]

                if accelerator.is_main_process:
                    probs_tmp = probs_gender_all[(probs_gender_all!=-1).all(dim=-1)]
                    gender_gap = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                    gender_pred_between_02_08 = ((probs_tmp[:,1]>=0.2)*(probs_tmp[:,1]<=0.8)).float().mean().item()
                    logs_i["gender_gap"].append(gender_gap)
                    logs_i["gender_gap_abs"].append(abs(gender_gap))
                    logs_i["gender_pred_between_0.2_0.8"].append(gender_pred_between_02_08)

                ################################################
                # Step 2: generate dynamic targets 
                # also broadcast from process idx 0, just in case targets_all computed might be different on different processes
                targets_all, uncertainty_all = generate_dynamic_targets(probs_gender_all, w_uncertainty=True)
                torch.distributed.broadcast(targets_all, src=0)
                torch.distributed.broadcast(uncertainty_all, src=0)

                targets_all[uncertainty_all>args.uncertainty_threshold] = -1
                targets = targets_all[probs_gender.shape[0]*(accelerator.local_process_index):probs_gender.shape[0]*(accelerator.local_process_index+1)]
                uncertainty = uncertainty_all[probs_gender.shape[0]*(accelerator.local_process_index):probs_gender.shape[0]*(accelerator.local_process_index+1)]
                accelerator.print(f"\tNum faces to compute grads: {(targets_all!=-1).sum().item()}/{targets_all.shape[0]}")

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

                face_indicators_ori, face_bboxs_ori, face_chips_ori, face_landmarks_ori, aligned_face_chips_ori = get_face(images_ori)
                preds_gender_ori, probs_gender_ori, logits_gender_ori = get_face_gender(z0_ori, selector=face_indicators_ori, fill_value=-1)
                
                images_small_ori = transforms.Resize(args.img_size_small)(images_ori)
                clip_feats_ori = get_clip_feat(images_small_ori, normalize=True, to_high_precision=True)
                DINO_feats_ori = get_dino_feat(images_small_ori, normalize=True, to_high_precision=True)

                images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
                face_indicators_ori_all = customized_all_gather(face_indicators_ori, accelerator, return_tensor_other_processes=False)
                face_bboxs_ori_all = customized_all_gather(face_bboxs_ori, accelerator, return_tensor_other_processes=False)
                preds_gender_ori_all = customized_all_gather(preds_gender_ori, accelerator, return_tensor_other_processes=False)
                probs_gender_ori_all = customized_all_gather(probs_gender_ori, accelerator, return_tensor_other_processes=False)

                face_feats_ori = get_face_feats(face_feats_net, aligned_face_chips_ori)
                
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_ori.jpg")
                        plot_in_grid(images_ori_all, save_to, face_indicators=face_indicators_ori_all, face_bboxs=face_bboxs_ori_all, preds_gender=preds_gender_ori_all, pred_class_probs_gender=probs_gender_ori_all.max(dim=-1).values)

                        log_imgs_i["img_ori"] = [save_to]
            
            ################################################
            # Step 4: compute loss
            loss_fair_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_face_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
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
                preds_gender_ori_ij = preds_gender_ori[idxs_ij]
                probs_gender_ori_ij = probs_gender_ori[idxs_ij]
                face_bboxs_ori_ij = face_bboxs_ori[idxs_ij]
                face_feats_ori_ij = face_feats_ori[idxs_ij]
                
                images_ij, z0_ij = generate_image_w_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet, return_latents=True)
                face_indicators_ij, face_bboxs_ij, face_chips_ij, face_landmarks_ij, aligned_face_chips_ij = get_face(images_ij)
                # Branch B: residual-error gender logits from z0 directly (grad flows z0 -> trainable model).
                # Computed on z0_ij before the RGB grad-hook below, which only affects Branch A (image loss).
                preds_gender_ij, probs_gender_ij, logits_gender_ij = get_face_gender(z0_ij, selector=face_indicators_ij, fill_value=-1)

                # Branch A: RGB image losses (CLIP/DINO) use the decoded x0 with the existing face grad-hook.
                images_ij = apply_grad_hook_face(images_ij, face_bboxs_ij, face_bboxs_ori_ij, targets_ij, preds_gender_ori_ij, probs_gender_ori_ij, factor=args.factor2)
                images_small_ij = transforms.Resize(args.img_size_small)(images_ij)
                clip_feats_ij = get_clip_feat(images_small_ij, normalize=True, to_high_precision=True)
                DINO_feats_ij = get_dino_feat(images_small_ij, normalize=True, to_high_precision=True)

                loss_CLIP_ij = - (clip_feats_ij * clip_feats_ori_ij).sum(dim=-1) + 1
                loss_DINO_ij = - (DINO_feats_ij * DINO_feats_ori_ij).sum(dim=-1) + 1
                
                loss_fair_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) *(-1)
                idxs_w_face_loss = ((face_indicators_ij == True) * (targets_ij != -1)).nonzero().view([-1])
                loss_fair_ij_w_face_loss = CE_loss(logits_gender_ij[idxs_w_face_loss], targets_ij[idxs_w_face_loss])
                loss_fair_ij[idxs_w_face_loss] = loss_fair_ij_w_face_loss.to(loss_fair_ij.dtype)
                
                loss_face_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) *(-1)

                idxs_w_face_feats_from_ori = ((face_indicators_ij==True) * (targets_ij!=-1) * (targets_ij==preds_gender_ori_ij) * (probs_gender_ori_ij.max(dim=-1).values>=args.face_gender_confidence_level)).nonzero().view([-1]).tolist()
                if len(idxs_w_face_feats_from_ori)>0:
                    face_feats_1 = get_face_feats(face_feats_net, aligned_face_chips_ij[idxs_w_face_feats_from_ori])
                    face_feats_target_1 = face_feats_ori_ij[idxs_w_face_feats_from_ori]
                    loss_face_ij[idxs_w_face_feats_from_ori] = (1 - (face_feats_1*face_feats_target_1).sum(dim=-1)).to(loss_face_ij.dtype)
                
                idxs_w_face_feats_from_search = list(set(((face_indicators_ij==True) * (targets_ij!=-1) ).nonzero().view([-1]).tolist()) - set(idxs_w_face_feats_from_ori))
                if len(idxs_w_face_feats_from_search)>0:
                    face_feats_2 = get_face_feats(face_feats_net, aligned_face_chips_ij[idxs_w_face_feats_from_search])
                    face_feats_target_2 = face_feats_model.semantic_search(face_feats_2, face_indicators_ij[idxs_w_face_feats_from_search])
                    loss_face_ij[idxs_w_face_feats_from_search] = (1 - (face_feats_2*face_feats_target_2).sum(dim=-1)).to(loss_face_ij.dtype)

                dynamic_weights = gen_dynamic_weights(face_indicators_ij, targets_ij, preds_gender_ori_ij, probs_gender_ori_ij, factor=args.factor1)
                loss_ij = loss_fair_ij + args.weight_loss_img * dynamic_weights * (loss_CLIP_ij + loss_DINO_ij) + args.weight_loss_face * loss_face_ij
                accelerator.backward(loss_ij.mean())

                with torch.no_grad():
                    loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)
                    loss_face_i[idxs_ij] = loss_face_ij.to(loss_face_i.dtype)
                    loss_CLIP_i[idxs_ij] = loss_CLIP_ij.to(loss_CLIP_i.dtype)
                    loss_DINO_i[idxs_ij] = loss_DINO_ij.to(loss_DINO_i.dtype)
                    loss_i[idxs_ij] = loss_ij.to(loss_i.dtype)
                    
            # for logging purpose, gather all losses to main_process
            accelerator.wait_for_everyone()
            loss_fair_all = customized_all_gather(loss_fair_i, accelerator)
            loss_face_all = customized_all_gather(loss_face_i, accelerator)
            loss_CLIP_all = customized_all_gather(loss_CLIP_i, accelerator)
            loss_DINO_all = customized_all_gather(loss_DINO_i, accelerator)
            loss_all = customized_all_gather(loss_i, accelerator)

            loss_all = loss_all[loss_fair_all!=-1]
            loss_fair_all = loss_fair_all[loss_fair_all!=-1]
            loss_face_all = loss_face_all[loss_face_all!=-1]

            if accelerator.is_main_process:
                logs_i["loss_fair"].append(loss_fair_all)
                logs_i["loss_face"].append(loss_face_all)
                logs_i["loss_CLIP"].append(loss_CLIP_all)
                logs_i["loss_DINO"].append(loss_DINO_all)
                logs_i["loss"].append(loss_all)
            
            # process logs
            if accelerator.is_main_process:
                for key in ["loss_fair", "loss_face", "loss_CLIP", "loss_DINO", "loss"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)
                    else:
                        logs_i[key] = torch.cat(logs_i[key])
                for key in ["gender_gap", "gender_gap_abs", "gender_pred_between_0.2_0.8"]:
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
            # we mannually sync grads
            # accelerator.wait_for_everyone()
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





# accelerate launch --config_file configs/accelerate_config.yaml 1-main-debias.py --config configs/debias-text-encoder.yaml
# accelerate launch --config_file configs/accelerate_config.yaml 1-main-debias.py --config configs/debias-unet.yaml
# accelerate launch --config_file configs/accelerate_config.yaml 1-main-debias.py --config configs/debias-text-encoder-and-unet.yaml