"""
model_module.py

This file contains the PyTorch Lightning model setup for video understanding models.

Author: -
"""

# basic
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import os
import sys
import time

# torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary import ModelSummary

# timm
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# backbone
from src.models.probes.attentive_pooler import Probe
from src.models.backbones.video_swin import video_swin_tiny, video_swin_base, video_swin_base_ssv2
from src.models.backbones.video_mae import video_mae_small, video_mae_base, video_mae_large
from src.models.backbones.dinov3 import dinov3_small, dinov3_base, dinov3_large
from src.models.backbones.vjepa2 import vjepa2_large, vjepa2_1_large
from src.models.backbones.internvideo_next import internvideo_next_base, internvideo_next_large
from src.models.backbones.siglip2 import siglip2_large

# reference models
from transformers import AutoModel
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# utils
from src.utils.video_mae_utils import LayerDecayValueAssigner
from src.utils.internvideo_next_utils import interpolate_pos_embed
from src.utils.losses import SiLogLoss, BoxTrackingLoss

MODELS = {
    "dinov3": {
        "small": dinov3_small,
        "base": dinov3_base,
        "large": dinov3_large,
    },
    "video_mae": {
        "small": video_mae_small,
        "small_ssv2": video_mae_small,
        "base": video_mae_base,
        "base_ssv2": video_mae_base,
        "large": video_mae_large,
    },
    "video_swin": {
        "tiny": video_swin_tiny,
        "base": video_swin_base,
        "base_ssv2": video_swin_base_ssv2,
    },
    "vjepa2": {
        "large": vjepa2_large,
    },
    "vjepa2_1": {
        "large": vjepa2_1_large,
    },
    "internvideo_next": {
        "base": internvideo_next_base,
        "large": internvideo_next_large,
    },
    "siglip2": {
        "large": siglip2_large,
    },
}

class VideoModel(pl.LightningModule):
    """
    Video Transformer Lightning Module.

    Args:
        args (argparse.Namespace): Arguments.
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.test_num_views = self.hparams.test_num_crops * self.hparams.test_num_clips
        self.is_probe = self.hparams.run_type == "probe"

        # model
        if self.hparams.model_type == "video_swin":
            self.model = MODELS["video_swin"][self.hparams.model_size](
                num_classes=self.hparams.num_classes,
                task_type=self.hparams.task_type,
                num_verb_classes=self.hparams.num_verb_classes,
                num_noun_classes=self.hparams.num_noun_classes,
                is_probe=self.is_probe)
        elif self.hparams.model_type == "video_mae":
            self.model = MODELS["video_mae"][self.hparams.model_size](
                img_size=self.hparams.input_size,
                num_classes=self.hparams.num_classes,
                all_frames=self.hparams.clip_len,
                feature_type=self.hparams.feature_type,
                drop_path_rate=self.hparams.drop_path_rate,
                task_type=self.hparams.task_type,
                num_verb_classes=self.hparams.num_verb_classes,
                num_noun_classes=self.hparams.num_noun_classes,
                is_probe=self.is_probe)
        elif self.hparams.model_type == "dinov3":
            self.model = MODELS["dinov3"][self.hparams.model_size](
                img_size=self.hparams.input_size,
                num_frames=self.hparams.clip_len,
                drop_path_rate=self.hparams.drop_path_rate,
                peft_type=self.hparams.peft_type,
                peft_rank=self.hparams.peft_rank,
                peft_dropout=self.hparams.peft_dropout,
                n_prompt_tokens=self.hparams.n_prompt_tokens,
                prompt_tokens_type=self.hparams.prompt_tokens_type,
                prompt_fourier_ratio=self.hparams.prompt_fourier_ratio,
                bpt_rank=self.hparams.bpt_rank,
                peft_clip_len=self.hparams.peft_clip_len)
        elif self.hparams.model_type in ("vjepa2", "vjepa2_1"):
            self.model = MODELS[self.hparams.model_type][self.hparams.model_size](
                img_size=self.hparams.input_size,
                drop_path_rate=self.hparams.drop_path_rate,
                num_frames=self.hparams.clip_len,
                peft_type=self.hparams.peft_type,
                peft_rank=self.hparams.peft_rank,
                peft_dropout=self.hparams.peft_dropout,
                backbone_clip_len=self.hparams.backbone_clip_len,
                peft_clip_len=self.hparams.peft_clip_len)
        elif self.hparams.model_type == "internvideo_next":
            self.model = MODELS["internvideo_next"][self.hparams.model_size](
                img_size=self.hparams.input_size,
                num_frames=self.hparams.clip_len,
                drop_path_rate=self.hparams.drop_path_rate,
                peft_type=self.hparams.peft_type,
                peft_rank=self.hparams.peft_rank,
                peft_dropout=self.hparams.peft_dropout,
                n_prompt_tokens=self.hparams.n_prompt_tokens,
                prompt_fourier_ratio=self.hparams.prompt_fourier_ratio,
                bpt_rank=self.hparams.bpt_rank,
                backbone_clip_len=self.hparams.backbone_clip_len,
                peft_clip_len=self.hparams.peft_clip_len)
        elif self.hparams.model_type == "siglip2":
            _ckpt = torch.load(self.hparams.checkpoint_path, map_location='cpu', weights_only=True)
            self.model = MODELS["siglip2"][self.hparams.model_size](
                img_size=self.hparams.input_size,
                num_frames=self.hparams.clip_len,
                drop_path_rate=self.hparams.drop_path_rate,
                peft_type=self.hparams.peft_type,
                peft_rank=self.hparams.peft_rank,
                peft_dropout=self.hparams.peft_dropout,
                cfg=_ckpt['cfg'])
        else:
            raise ValueError(f"Unknown model type: {self.hparams.model_type}")

        # load checkpoint
        checkpoint = torch.load(self.hparams.checkpoint_path, map_location=self.device, weights_only=True)
        if self.hparams.state_type == "test_model":
            state_dict_model = {k[6:]: v for k, v in checkpoint['state_dict'].items() if k.startswith("model.")}
            strict = True
        else:
            state_dict_model = checkpoint['state_dict']        
            strict = True if ((self.hparams.peft_type is None) and (not self.hparams.model_type == "video_swin")) else False

        if self.hparams.model_type == "internvideo_next" and self.hparams.backbone_clip_len != 16:
            interpolate_pos_embed(state_dict_model, self.model)

        self.model.load_state_dict(state_dict_model, strict=strict)

        # probing
        if self.is_probe:
            if self.hparams.peft_type is None:
                self.model.eval()

            if self.hparams.peft_type != "full_finetune":
                for name, param in self.model.named_parameters():
                    if "prompt_tokens" not in name and "adapter" not in name and "lora" not in name and "st_lora" not in name and "dora" not in name and "ssf_scale" not in name and "ssf_shift" not in name and "zeroi2v" not in name:
                        param.requires_grad = False
                    if (self.hparams.peft_unfreeze_norm or self.hparams.peft_type == "layernorm") and "norm" in name:
                        param.requires_grad = True
                    if self.hparams.peft_type == "bitfit" and "bias" in name:
                        param.requires_grad = True

            self.pooler = Probe(embed_dim=self.model.embed_dim,
                                num_heads=self.model.num_heads,
                                depth=self.hparams.probe_depth,
                                num_classes=self.hparams.num_classes,
                                use_rope=self.hparams.probe_use_rope,
                                use_pos_embed=self.hparams.probe_add_pos_embed,
                                grid_size=self.model.grid_size,
                                num_frames=self.hparams.clip_len,
                                probe_clip_len=self.hparams.probe_clip_len,
                                input_size=self.hparams.input_size,
                                model_tubelet_size=self.model.tubelet_size,
                                drop_path_rate=self.hparams.probe_drop_path_rate,
                                task_type=self.hparams.task_type,
                                num_verb_classes=self.hparams.num_verb_classes,
                                num_noun_classes=self.hparams.num_noun_classes,
                                pooler_type=self.hparams.probe_pooler_type,
                                depth_patch_size=self.hparams.depth_patch_size,
                                depth_tubelet_size=self.hparams.depth_tubelet_size,
                                max_depth=self.hparams.max_depth,
                                min_depth=self.hparams.min_depth,
                                backbone_depth=self.model.get_num_layers() if hasattr(self.model, 'get_num_layers') else None,
                                max_queries=self.hparams.max_queries)

            if self.hparams.state_type == "test_model":
                state_dict_pooler = {k[7:]: v for k, v in checkpoint['state_dict'].items() if k.startswith("pooler.")}
                self.pooler.load_state_dict(state_dict_pooler)

        # extra reference model
        self.reference_model = None
        if self.hparams.reference_model_name == "optical_flow":
            self.reference_model = raft_large(weights=Raft_Large_Weights.C_T_V2, progress=False)
        elif self.hparams.reference_model_name == "hf_automodel":
            self.reference_model = AutoModel.from_pretrained(self.hparams.reference_model_path)
        elif self.hparams.reference_model_name == "video_mae_small":
            self.reference_model = VideoMAE(num_classes=self.hparams.num_classes)
            checkpoint_small = torch.load(self.hparams.reference_model_path, map_location=self.device, weights_only=True)['state_dict']
            checkpoint_small = {k.replace("model.", "", 1): v for k, v in checkpoint_small.items()}
            self.reference_model.load_state_dict(checkpoint_small, strict=False)
        elif self.hparams.reference_model_name == "video_swin_tiny":
            backbone_small = SwinTransformer3D()
            cls_head_small = I3DHead(
                            num_features=backbone_small.num_features, 
                            num_classes=self.hparams.num_classes)
            self.reference_model = nn.Sequential(OrderedDict([
                            ('backbone', backbone_small),
                            ('cls_head', cls_head_small)]))
            checkpoint_small = torch.load(self.hparams.reference_model_path, map_location=self.device, weights_only=True)
            self.reference_model.load_state_dict(checkpoint_small['state_dict'])
        elif self.hparams.reference_model_name == "pooler":
            self.reference_model = deepcopy(self.pooler)
            checkpoint_pooler = torch.load(self.hparams.reference_model_path, map_location=self.device, weights_only=True)
            state_dict_pooler = {k[7:]: v for k, v in checkpoint_pooler['state_dict'].items() if k.startswith("pooler.")}
            self.reference_model.load_state_dict(state_dict_pooler)
        elif self.hparams.reference_model_name == "ema":
            self.ema_decay_schedule = None
            if self.is_probe:
                self.reference_model = deepcopy(self.pooler)
                self._ema_param_pairs  = list(zip(self.reference_model.parameters(),
                                             self.pooler.parameters()))
                self._ema_buffer_pairs = list(zip(self.reference_model.buffers(),
                                             self.pooler.buffers()))
            else:
                self.reference_model = deepcopy(self.model)
                self._ema_param_pairs  = list(zip(self.reference_model.parameters(),
                                             self.model.parameters()))
                self._ema_buffer_pairs = list(zip(self.reference_model.buffers(),
                                             self.model.buffers()))

        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False 

        # learnability score
        self.learnability_model = None
        if (self.hparams.sample_strategy == "jest_videos") or (self.hparams.sample_strategy == "jest_frames"):
            if self.is_probe:
                self.learnability_model = deepcopy(self.pooler)
                self._learnability_param_pairs  = list(zip(self.learnability_model.parameters(),
                                             self.pooler.parameters()))
                self._learnability_buffer_pairs = list(zip(self.learnability_model.buffers(),
                                             self.pooler.buffers()))
            else:
                self.learnability_model = deepcopy(self.model)
                self._learnability_param_pairs  = list(zip(self.learnability_model.parameters(),
                                             self.model.parameters()))
                self._learnability_buffer_pairs = list(zip(self.learnability_model.buffers(),
                                             self.model.buffers()))

            for param in self.learnability_model.parameters():
                param.requires_grad = False 
            self.learnability_model.eval()

        # loss
        self.mixup_fn = None
        if self.hparams.task_type in ["classification", "classification_epic"]:
            if (self.hparams.mixup_enabled) and ((self.hparams.mixup > 0) or (self.hparams.cutmix > 0.0) or (self.hparams.cutmix_minmax is not None)):
                self.mixup_fn = Mixup(
                    mixup_alpha=self.hparams.mixup, cutmix_alpha=self.hparams.cutmix, cutmix_minmax=self.hparams.cutmix_minmax,
                    prob=self.hparams.mixup_prob, switch_prob=self.hparams.mixup_switch_prob, mode=self.hparams.mixup_mode,
                    label_smoothing=self.hparams.smoothing, num_classes=self.hparams.num_classes)
                self.loss_fn = SoftTargetCrossEntropy()
            elif self.hparams.smoothing > 0:
                self.loss_fn = LabelSmoothingCrossEntropy(smoothing=self.hparams.smoothing)
            else:
                self.loss_fn = nn.CrossEntropyLoss()

            self.loss_fn_val = nn.CrossEntropyLoss()
        elif self.hparams.task_type == "depth_estimation":
            self.loss_fn = SiLogLoss(min_depth=self.hparams.min_depth, max_depth=self.hparams.max_depth)
            self.loss_fn_val = SiLogLoss(min_depth=self.hparams.min_depth, max_depth=self.hparams.max_depth)
        elif self.hparams.task_type == "semantic_segmentation":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.hparams.seg_void_label)
            self.loss_fn_val = nn.CrossEntropyLoss(ignore_index=self.hparams.seg_void_label)
        elif self.hparams.task_type == "box_tracking":
            self.loss_fn = BoxTrackingLoss()
            self.loss_fn_val = BoxTrackingLoss()

        # metrics
        if self.hparams.task_type == "classification":
            self.metrics = {
                "val": {"top1_correct": 0, "top5_correct": 0, "ref_top1_correct": 0, "ref_top5_correct": 0, "samples": 0},
                "test": {"top1_correct": 0, "top5_correct": 0, "ref_top1_correct": 0, "ref_top5_correct": 0, "samples": 0}}
        elif self.hparams.task_type == "classification_epic":
            self.metrics = {
                "val": {"top1_correct": 0, "top5_correct": 0, "samples": 0, "ref_top1_correct": 0, "ref_top5_correct": 0, "top1_correct_verb": 0, "top5_correct_verb": 0, "top1_correct_noun": 0, "top5_correct_noun": 0},
                "test": {"top1_correct": 0, "top5_correct": 0, "ref_top1_correct": 0, "ref_top5_correct": 0, "samples": 0, "top1_correct_verb": 0, "top5_correct_verb": 0, "top1_correct_noun": 0, "top5_correct_noun": 0, "top1_hd_verb": 0, "top5_hd_verb": 0, "top1_hd_noun": 0, "top5_hd_noun": 0, "top1_hd": 0, "top5_hd": 0, "ref_top1_hd": 0, "ref_top5_hd": 0, "samples_hd": 0}}
        elif self.hparams.task_type == "depth_estimation":
            self.metrics = {
                "val":  {"abs_rel_sum": 0.0, "delta1_correct": 0.0, "valid_pixels": 0, "ref_abs_rel_sum": 0.0, "ref_delta1_correct": 0.0},
                "test": {"abs_rel_sum": 0.0, "delta1_correct": 0.0, "valid_pixels": 0, "ref_abs_rel_sum": 0.0, "ref_delta1_correct": 0.0}}
        elif self.hparams.task_type == "semantic_segmentation":
            nc = self.hparams.num_classes
            self.metrics = {
                "val":  {"intersection": torch.zeros(nc), "union": torch.zeros(nc),
                         "ref_intersection": torch.zeros(nc), "ref_union": torch.zeros(nc)},
                "test": {"intersection": torch.zeros(nc), "union": torch.zeros(nc),
                         "ref_intersection": torch.zeros(nc), "ref_union": torch.zeros(nc)},
            }
        elif self.hparams.task_type == "box_tracking":
            self.metrics = {
                "val":  {"iou_sum": 0.0, "boxes": 0, "ref_iou_sum": 0.0},
                "test": {"iou_sum": 0.0, "boxes": 0, "ref_iou_sum": 0.0}}
        else:
            raise NotImplementedError(f"Metrics for task type '{self.hparams.task_type}' is not implemented.")

        # inference throughput logging
        self.metrics["test"]["throughput_time"] = 0.0
        self.metrics["test"]["throughput_units"] = 0

        # count
        self.count = 0

    def on_train_start(self):
        self.log("trainable_params", float(ModelSummary(self).trainable_parameters), rank_zero_only=True)

    def forward(self, x, query_boxes=None):
        """Forward pass."""
        if self.hparams.task_type == "classification_epic":
            if self.is_probe:
                x_features = self.model(x)
                x_verb, x_noun = self.pooler(x_features)
            else:
                x_verb, x_noun = self.model(x)

            return x_verb, x_noun
        else:
            if self.is_probe and self.hparams.probe_pooler_type in ("dist", "dpt", "vda", "attentive_multilayer"):
                features = self.model(x, return_all_layers=True)
                images = x if self.hparams.probe_pooler_type == "dist" else None
                return self.pooler(features, images=images)
            x = self.model(x)
            if self.is_probe:
                x = self.pooler(x, query_boxes=query_boxes)
            return x

    def _timed_forward(self, x, query_boxes=None):
        """self(x) wrapped with cuda sync on both sides; accumulates wall-time
        and unit count (clips for classification, frames for seg/depth) into
        self.metrics['test']."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self(x, query_boxes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.metrics["test"]["throughput_time"] += time.perf_counter() - t0
        if self.hparams.task_type in ("depth_estimation", "semantic_segmentation"):
            # x: (B*num_views, C, clip_len, H, W); num_views==1 for these tasks
            self.metrics["test"]["throughput_units"] += x.shape[0] * x.shape[2]
        else:
            self.metrics["test"]["throughput_units"] += x.shape[0]
        return out

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        B, C, T, H, W = x.shape

        query_boxes = None
        if self.hparams.task_type == "classification_epic":
            y_verb = y[:, 0]
            y_noun = y[:, 1]
        elif self.hparams.task_type == "box_tracking":
            query_boxes = y['boxes'][:, :, 0]

        if self.hparams.sample_strategy == "repeated_augmentation":
            clip_len = T // self.hparams.num_samples
            x = x.view(B, C, self.hparams.num_samples, clip_len, H, W)  # [B, C, T, H, W] -> [B, C, num_views, clip_len, H, W]
            x = x.permute(0, 2, 1, 3, 4, 5).contiguous()                # reorder to [B, num_views, C, clip_len, H, W]      
            x = x.view(-1, C, clip_len, H, W)                           # [B, num_views, C, clip_len, H, W] -> [B * num_views, C, clip_len, H, W]
            if self.hparams.task_type == "classification_epic":
                y_verb = y_verb.repeat_interleave(self.hparams.num_samples, dim=0)
                y_noun = y_noun.repeat_interleave(self.hparams.num_samples, dim=0)
            else:
                y = y.repeat_interleave(self.hparams.num_samples, dim=0)
        elif (self.hparams.sample_strategy == "jest_videos") and (self.current_epoch >= self.hparams.jest_warmup_epochs):
            B_sub = B // self.hparams.jest_interval

            with torch.no_grad():
                if self.is_probe:
                    x = self.model(x)
                
                if self.hparams.task_type == "classification_epic":
                    y_easy_verb, y_easy_noun = self.reference_model(x)
                    y_hard_verb, y_hard_noun = self.learnability_model(x)
                else:
                    y_easy = self.reference_model(x)
                    y_hard = self.learnability_model(x)
            
            torch.cuda.empty_cache() 

            if self.hparams.learnability_reduction == "batch":
                if self.hparams.task_type == "classification_epic":
                    loss_easy_verb = torch.stack([self.loss_fn_val(y_easy_verb[i*B_sub:(i+1)*B_sub], y_verb[i*B_sub:(i+1)*B_sub]) 
                                        for i in range(self.hparams.jest_interval)])
                    loss_hard_verb = torch.stack([self.loss_fn_val(y_hard_verb[i*B_sub:(i+1)*B_sub], y_verb[i*B_sub:(i+1)*B_sub]) 
                                        for i in range(self.hparams.jest_interval)])
                    loss_easy_noun = torch.stack([self.loss_fn_val(y_easy_noun[i*B_sub:(i+1)*B_sub], y_noun[i*B_sub:(i+1)*B_sub]) 
                                        for i in range(self.hparams.jest_interval)])
                    loss_hard_noun = torch.stack([self.loss_fn_val(y_hard_noun[i*B_sub:(i+1)*B_sub], y_noun[i*B_sub:(i+1)*B_sub]) 
                                        for i in range(self.hparams.jest_interval)])

                    loss_tensor = loss_easy_verb - loss_hard_verb + loss_easy_noun - loss_hard_noun
                else:
                    loss_easy = torch.stack([self.loss_fn_val(y_easy[i*B_sub:(i+1)*B_sub], y[i*B_sub:(i+1)*B_sub]) 
                                            for i in range(self.hparams.jest_interval)])
                    loss_hard = torch.stack([self.loss_fn_val(y_hard[i*B_sub:(i+1)*B_sub], y[i*B_sub:(i+1)*B_sub]) 
                                            for i in range(self.hparams.jest_interval)])
                                        
                    loss_tensor = loss_easy - loss_hard

                best_view_index = torch.argmin(loss_tensor).item()
                x = x[best_view_index*B_sub:(best_view_index+1)*B_sub].contiguous()
                if self.hparams.task_type == "classification_epic":
                    y_verb = y_verb[best_view_index*B_sub:(best_view_index+1)*B_sub].contiguous()
                    y_noun = y_noun[best_view_index*B_sub:(best_view_index+1)*B_sub].contiguous()
                else:
                    y = y[best_view_index*B_sub:(best_view_index+1)*B_sub].contiguous()
            elif self.hparams.learnability_reduction == "sample":
                if self.hparams.task_type == "classification_epic":
                    loss_easy_verb = F.cross_entropy(y_easy_verb, y_verb, reduction='none') 
                    loss_hard_verb = F.cross_entropy(y_hard_verb, y_verb, reduction='none') 
                    loss_easy_noun = F.cross_entropy(y_easy_noun, y_noun, reduction='none') 
                    loss_hard_noun = F.cross_entropy(y_hard_noun, y_noun, reduction='none') 
                    loss_tensor = loss_easy_verb - loss_hard_verb + loss_easy_noun - loss_hard_noun

                    best_view_indices = torch.argsort(loss_tensor)[:B_sub]                
                    y_verb = y_verb[best_view_indices].contiguous()
                    y_noun = y_noun[best_view_indices].contiguous()
                else:
                    loss_easy = F.cross_entropy(y_easy, y, reduction='none') 
                    loss_hard = F.cross_entropy(y_hard, y, reduction='none') 
                    loss_tensor = loss_easy - loss_hard

                    best_view_indices = torch.argsort(loss_tensor)[:B_sub]                
                    y = y[best_view_indices].contiguous()

                x = x[best_view_indices].contiguous()
        elif (self.hparams.sample_strategy == "jest_frames") and (self.current_epoch >= self.hparams.jest_warmup_epochs):
            clip_len = T // self.hparams.jest_interval
                
            x = x.view(B, C, self.hparams.jest_interval, clip_len, H, W)  # [B, C, T, H, W] -> [B, C, num_views, clip_len, H, W]
            x = x.permute(0, 2, 1, 3, 4, 5).contiguous()                # reorder to [B, num_views, C, clip_len, H, W]      
            x = x.view(-1, C, clip_len, H, W)                           # [B, num_views, C, clip_len, H, W] -> [B * num_views, C, clip_len, H, W]

            with torch.no_grad():
                if self.is_probe:
                    x = self.model(x)
                
                if self.hparams.task_type == "classification_epic":
                    y_easy_verb, y_easy_noun = self.reference_model(x)
                    y_hard_verb, y_hard_noun = self.learnability_model(x)
                else:
                    y_easy = self.reference_model(x)
                    y_hard = self.learnability_model(x)
            
            torch.cuda.empty_cache() 

            if self.hparams.learnability_reduction == "sample_all":
                if self.hparams.task_type == "classification_epic":
                    y_verb = y_verb.repeat_interleave(self.hparams.jest_interval, dim=0)
                    y_noun = y_noun.repeat_interleave(self.hparams.jest_interval, dim=0)
                    loss_easy_verb = F.cross_entropy(y_easy_verb, y_verb, reduction='none') 
                    loss_hard_verb = F.cross_entropy(y_hard_verb, y_verb, reduction='none') 
                    loss_easy_noun = F.cross_entropy(y_easy_noun, y_noun, reduction='none') 
                    loss_hard_noun = F.cross_entropy(y_hard_noun, y_noun, reduction='none') 
                    loss_tensor = loss_easy_verb - loss_hard_verb + loss_easy_noun - loss_hard_noun

                    best_view_indices = torch.argsort(loss_tensor)[:B]                
                    y_verb = y_verb[best_view_indices].contiguous()
                    y_noun = y_noun[best_view_indices].contiguous()
                else:
                    y = y.repeat_interleave(self.hparams.jest_interval, dim=0)
                    loss_easy = F.cross_entropy(y_easy, y, reduction='none') 
                    loss_hard = F.cross_entropy(y_hard, y, reduction='none') 
                    loss_tensor = loss_easy - loss_hard

                    best_view_indices = torch.argsort(loss_tensor)[:B]                
                    y = y[best_view_indices].contiguous()

                x = x[best_view_indices].contiguous()
            else:
                if self.hparams.task_type == "classification_epic":
                    y_easy_verb = y_easy_verb.view(B, self.hparams.jest_interval, -1)
                    y_hard_verb = y_hard_verb.view(B, self.hparams.jest_interval, -1)
                    y_easy_noun = y_easy_noun.view(B, self.hparams.jest_interval, -1)
                    y_hard_noun = y_hard_noun.view(B, self.hparams.jest_interval, -1)
                else:
                    y_easy = y_easy.view(B, self.hparams.jest_interval, -1)
                    y_hard = y_hard.view(B, self.hparams.jest_interval, -1)

                x = x.view(B, self.hparams.jest_interval, *x.shape[1:])

                if self.hparams.learnability_reduction == "batch":
                    if self.hparams.task_type == "classification_epic":
                        loss_easy_verb = torch.stack([self.loss_fn_val(y_easy_verb[:, i], y_verb) 
                                    for i in range(self.hparams.jest_interval)])
                        loss_hard_verb = torch.stack([self.loss_fn_val(y_hard_verb[:, i], y_verb) 
                                    for i in range(self.hparams.jest_interval)])
                        loss_easy_noun = torch.stack([self.loss_fn_val(y_easy_noun[:, i], y_noun) 
                                    for i in range(self.hparams.jest_interval)])
                        loss_hard_noun = torch.stack([self.loss_fn_val(y_hard_noun[:, i], y_noun) 
                                    for i in range(self.hparams.jest_interval)])

                        loss_tensor = loss_easy_verb - loss_hard_verb + loss_easy_noun - loss_hard_noun
                    else:
                        loss_easy = torch.stack([self.loss_fn_val(y_easy[:, i], y) 
                                    for i in range(self.hparams.jest_interval)])

                        loss_hard = torch.stack([self.loss_fn_val(y_hard[:, i], y) 
                                    for i in range(self.hparams.jest_interval)])

                        loss_tensor = loss_easy - loss_hard

                    best_view_index = torch.argmin(loss_tensor).item()
                    x = x[:, best_view_index].contiguous()
                elif self.hparams.learnability_reduction == "sample":
                    if self.hparams.task_type == "classification_epic":
                        loss_easy_verb = torch.stack([
                            F.cross_entropy(y_easy_verb[:, i], y_verb, reduction='none') 
                            for i in range(self.hparams.jest_interval)
                        ], dim=1)
                        loss_hard_verb = torch.stack([
                            F.cross_entropy(y_hard_verb[:, i], y_verb, reduction='none') 
                            for i in range(self.hparams.jest_interval)
                        ], dim=1)
                        loss_easy_noun = torch.stack([
                            F.cross_entropy(y_easy_noun[:, i], y_noun, reduction='none') 
                            for i in range(self.hparams.jest_interval)
                        ], dim=1)
                        loss_hard_noun = torch.stack([
                            F.cross_entropy(y_hard_noun[:, i], y_noun, reduction='none') 
                            for i in range(self.hparams.jest_interval)
                        ], dim=1)

                        loss_tensor = loss_easy_verb - loss_hard_verb + loss_easy_noun - loss_hard_noun
                    else:
                        loss_easy = torch.stack([
                            F.cross_entropy(y_easy[:, i], y, reduction='none') 
                            for i in range(self.hparams.jest_interval)
                        ], dim=1)  
                        loss_hard = torch.stack([
                            F.cross_entropy(y_hard[:, i], y, reduction='none') 
                            for i in range(self.hparams.jest_interval)
                        ], dim=1)
                    
                        loss_tensor = loss_easy - loss_hard

                    best_view_indices = torch.argmin(loss_tensor, dim=1)
                    batch_indices = torch.arange(B, device=x.device)

                    x = x[batch_indices, best_view_indices].contiguous()
            

        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)

        if self.is_probe and self.hparams.peft_type is None:
            x_images = x if self.hparams.probe_pooler_type == "dist" else None
            if not (((self.hparams.sample_strategy == "jest_videos") or (self.hparams.sample_strategy == "jest_frames")) and (self.current_epoch >= self.hparams.jest_warmup_epochs)):
                with torch.no_grad():
                    if self.hparams.probe_pooler_type in ("dist", "dpt", "vda", "attentive_multilayer"):
                        x = self.model(x, return_all_layers=True)
                    else:
                        x = self.model(x)

            if self.hparams.probe_pooler_type == "dist":
                y_hat = self.pooler(x, images=x_images)
            else:
                y_hat = self.pooler(x, query_boxes=query_boxes)
        else:
            y_hat = self(x, query_boxes)

        if self.hparams.task_type == "classification_epic":
            y_hat_verb, y_hat_noun = y_hat
            loss_verb = self.loss_fn(y_hat_verb, y_verb)
            loss_noun = self.loss_fn(y_hat_noun, y_noun)
            loss = loss_verb + loss_noun

            self.log('train_loss_verb', loss_verb, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_loss_noun', loss_noun, on_step=False, on_epoch=True, sync_dist=True)
        elif self.hparams.task_type == "box_tracking":
            loss = self.loss_fn(y_hat, *self._box_targets(y))
        else:
            loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams.task_type == "classification":
            y_hat = self(x)
            loss = self.loss_fn_val(y_hat, y)
            top1_correct, top5_correct = self._topk_correct(y_hat, y, topk=(1, 5))

            self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
            self.metrics["val"]["top1_correct"] += top1_correct
            self.metrics["val"]["top5_correct"] += top5_correct
            self.metrics["val"]["samples"] += y.size(0)

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x)
                    y_hat_ref = self.reference_model(x_features)
                else:
                    y_hat_ref = self.reference_model(x)
         
                ref_top1_correct, ref_top5_correct = self._topk_correct(y_hat_ref, y, topk=(1, 5))

                self.metrics["val"]["ref_top1_correct"] += ref_top1_correct
                self.metrics["val"]["ref_top5_correct"] += ref_top5_correct
        elif self.hparams.task_type == "classification_epic":
            y_verb = y[:, 0]
            y_noun = y[:, 1]

            y_hat_verb, y_hat_noun = self(x)
            loss_verb = self.loss_fn_val(y_hat_verb, y_verb)
            loss_noun = self.loss_fn_val(y_hat_noun, y_noun)
            loss = loss_verb + loss_noun

            correct_action, correct_verb, correct_noun = self._topk_correct_epic(y_hat_verb, y_verb, y_hat_noun, y_noun, topk=(1, 5))
            top1_correct, top5_correct = correct_action[0], correct_action[1]
            top1_correct_verb, top5_correct_verb = correct_verb[0], correct_verb[1]
            top1_correct_noun, top5_correct_noun = correct_noun[0], correct_noun[1]

            self.log('val_loss_verb', loss_verb, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_loss_noun', loss_noun, on_step=False, on_epoch=True, sync_dist=True)
            self.metrics["val"]["top1_correct_verb"] += top1_correct_verb
            self.metrics["val"]["top5_correct_verb"] += top5_correct_verb
            self.metrics["val"]["top1_correct_noun"] += top1_correct_noun
            self.metrics["val"]["top5_correct_noun"] += top5_correct_noun

            self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
            self.metrics["val"]["top1_correct"] += top1_correct
            self.metrics["val"]["top5_correct"] += top5_correct
            self.metrics["val"]["samples"] += y.size(0)

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x)
                    y_verb_hat_ref, y_noun_hat_ref = self.reference_model(x_features)
                else:
                    y_verb_hat_ref, y_noun_hat_ref = self.reference_model(x)

                correct_action, correct_verb, correct_noun = self._topk_correct_epic(y_verb_hat_ref, y_verb, y_noun_hat_ref, y_noun, topk=(1, 5))
                ref_top1_correct, ref_top5_correct = correct_action[0], correct_action[1]

                self.metrics["val"]["ref_top1_correct"] += ref_top1_correct
                self.metrics["val"]["ref_top5_correct"] += ref_top5_correct
        elif self.hparams.task_type == "depth_estimation":
            y_hat = self(x)
            loss = self.loss_fn_val(y_hat, y)
            m = self._depth_metrics(y_hat, y)
            self.metrics["val"]["abs_rel_sum"]    += m["abs_rel"]
            self.metrics["val"]["delta1_correct"] += m["delta1"]
            self.metrics["val"]["valid_pixels"]   += m["count"]

            self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x)
                    y_hat_ref = self.reference_model(x_features)
                else:
                    y_hat_ref = self.reference_model(x)
                m_ref = self._depth_metrics(y_hat_ref, y)
                self.metrics["val"]["ref_abs_rel_sum"]    += m_ref["abs_rel"]
                self.metrics["val"]["ref_delta1_correct"] += m_ref["delta1"]
        elif self.hparams.task_type == "semantic_segmentation":
            y_hat = self(x)
            loss = self.loss_fn_val(y_hat, y)
            m = self._segmentation_metrics(y_hat, y)
            self.metrics["val"]["intersection"] += m["intersection"].cpu()
            self.metrics["val"]["union"]        += m["union"].cpu()

            self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x)
                    y_hat_ref = self.reference_model(x_features)
                else:
                    y_hat_ref = self.reference_model(x)
                m_ref = self._segmentation_metrics(y_hat_ref, y)
                self.metrics["val"]["ref_intersection"] += m_ref["intersection"].cpu()
                self.metrics["val"]["ref_union"]        += m_ref["union"].cpu()
        elif self.hparams.task_type == "box_tracking":
            query_boxes = y['boxes'][:, :, 0]
            target, valid = self._box_targets(y)

            y_hat = self(x, query_boxes)
            loss = self.loss_fn_val(y_hat, target, valid)
            m = self._box_metrics(y_hat, target, valid)
            self.metrics["val"]["iou_sum"] += m["iou_sum"]
            self.metrics["val"]["boxes"]   += m["count"]

            self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x)
                    y_hat_ref = self.reference_model(x_features, query_boxes=query_boxes)
                else:
                    y_hat_ref = self.reference_model(x, query_boxes=query_boxes)
                m_ref = self._box_metrics(y_hat_ref, target, valid)
                self.metrics["val"]["ref_iou_sum"] += m_ref["iou_sum"]
        else:
            raise NotImplementedError(f"Validation step for task type '{self.hparams.task_type}' is not implemented.")

        return loss

    def on_validation_epoch_end(self):
        self._aggregate_and_log_metrics(stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch        
        B, C, T, H, W = x.shape
        clip_len = self.hparams.clip_len 
        num_views = T // clip_len

        # [B, C, T*num_views, H, W] -> [B, C, num_views, T, H, W]
        x = x.view(B, C, num_views, clip_len, H, W)
        # -> [B, num_views, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()        
        # -> [B * num_views, C, T, H, W]
        x_input = x.view(-1, C, clip_len, H, W)

        if self.hparams.task_type == "classification":
            logits = self._timed_forward(x_input)
            probs = F.softmax(logits, dim=1)
            probs = probs.view(B, num_views, -1) 
            probs = probs.mean(dim=1) 

            # metrics
            top1_correct, top5_correct = self._topk_correct(probs, y, topk=(1, 5))

            self.metrics["test"]["top1_correct"] += top1_correct
            self.metrics["test"]["top5_correct"] += top5_correct
            self.metrics["test"]["samples"] += y.size(0)

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x_input)
                    y_hat_ref = self.reference_model(x_features)
                else:
                    y_hat_ref = self.reference_model(x_input)

                probs_ref = F.softmax(y_hat_ref, dim=1)         
                probs_ref = probs_ref.view(B, num_views, -1) 
                probs_ref = probs_ref.mean(dim=1)          
                ref_top1_correct, ref_top5_correct = self._topk_correct(probs_ref, y, topk=(1, 5))

                self.metrics["test"]["ref_top1_correct"] += ref_top1_correct
                self.metrics["test"]["ref_top5_correct"] += ref_top5_correct
        elif self.hparams.task_type == "classification_epic":
            y_verb = y[:, 0]
            y_noun = y[:, 1]

            # forward pass
            logits_verb, logits_noun = self._timed_forward(x_input)
            probs_verb = F.softmax(logits_verb, dim=1)
            probs_noun = F.softmax(logits_noun, dim=1)
            probs_verb = probs_verb.view(B, num_views, -1) 
            probs_noun = probs_noun.view(B, num_views, -1) 
            probs_verb = probs_verb.mean(dim=1)
            probs_noun = probs_noun.mean(dim=1)

            # metrics
            correct_action, correct_verb, correct_noun = self._topk_correct_epic(probs_verb, y_verb, probs_noun, y_noun, topk=(1, 5))
            top1_correct, top5_correct = correct_action[0], correct_action[1]
            top1_correct_verb, top5_correct_verb = correct_verb[0], correct_verb[1]
            top1_correct_noun, top5_correct_noun = correct_noun[0], correct_noun[1]

            if dataloader_idx == 0:
                self.metrics["test"]["top1_correct_verb"] += top1_correct_verb
                self.metrics["test"]["top5_correct_verb"] += top5_correct_verb
                self.metrics["test"]["top1_correct_noun"] += top1_correct_noun
                self.metrics["test"]["top5_correct_noun"] += top5_correct_noun
                self.metrics["test"]["top1_correct"] += top1_correct
                self.metrics["test"]["top5_correct"] += top5_correct
                self.metrics["test"]["samples"] += y.size(0)
            else:
                self.metrics["test"]["top1_hd_verb"] += top1_correct_verb
                self.metrics["test"]["top5_hd_verb"] += top5_correct_verb
                self.metrics["test"]["top1_hd_noun"] += top1_correct_noun
                self.metrics["test"]["top5_hd_noun"] += top5_correct_noun
                self.metrics["test"]["top1_hd"] += top1_correct
                self.metrics["test"]["top5_hd"] += top5_correct
                self.metrics["test"]["samples_hd"] += y.size(0)

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x_input)
                    y_verb_hat_ref, y_noun_hat_ref = self.reference_model(x_features)
                else:
                    y_verb_hat_ref, y_noun_hat_ref = self.reference_model(x_input)

                probs_verb_ref = F.softmax(y_verb_hat_ref, dim=1)         
                probs_verb_ref = probs_verb_ref.view(B, num_views, -1) 
                probs_verb_ref = probs_verb_ref.mean(dim=1) 
                probs_noun_ref = F.softmax(y_noun_hat_ref, dim=1)         
                probs_noun_ref = probs_noun_ref.view(B, num_views, -1) 
                probs_noun_ref = probs_noun_ref.mean(dim=1) 
                correct_action, correct_verb, correct_noun = self._topk_correct_epic(probs_verb_ref, y_verb, probs_noun_ref, y_noun, topk=(1, 5))
                ref_top1_correct, ref_top5_correct = correct_action[0], correct_action[1]

                if dataloader_idx == 0:
                    self.metrics["test"]["ref_top1_correct"] += ref_top1_correct
                    self.metrics["test"]["ref_top5_correct"] += ref_top5_correct
                else:
                    self.metrics["test"]["ref_top1_hd"] += ref_top1_correct
                    self.metrics["test"]["ref_top5_hd"] += ref_top5_correct
        elif self.hparams.task_type == "depth_estimation":
            assert num_views == 1, f"Depth estimation only supports 1 test view, got {num_views}"

            preds = self._timed_forward(x_input)  # (B, num_frames, H, W)

            m = self._depth_metrics(preds, y)
            self.metrics["test"]["abs_rel_sum"]    += m["abs_rel"]
            self.metrics["test"]["delta1_correct"] += m["delta1"]
            self.metrics["test"]["valid_pixels"]   += m["count"]

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x_input)
                    y_hat_ref = self.reference_model(x_features)
                else:
                    y_hat_ref = self.reference_model(x_input)
                m_ref = self._depth_metrics(y_hat_ref, y)
                self.metrics["test"]["ref_abs_rel_sum"]    += m_ref["abs_rel"]
                self.metrics["test"]["ref_delta1_correct"] += m_ref["delta1"]
        elif self.hparams.task_type == "semantic_segmentation":
            assert num_views == 1, f"Semantic segmentation only supports 1 test view, got {num_views}"

            preds = self._timed_forward(x_input)
            m = self._segmentation_metrics(preds, y)
            self.metrics["test"]["intersection"] += m["intersection"].cpu()
            self.metrics["test"]["union"]        += m["union"].cpu()

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x_input)
                    y_hat_ref = self.reference_model(x_features)
                else:
                    y_hat_ref = self.reference_model(x_input)
                m_ref = self._segmentation_metrics(y_hat_ref, y)
                self.metrics["test"]["ref_intersection"] += m_ref["intersection"].cpu()
                self.metrics["test"]["ref_union"]        += m_ref["union"].cpu()
        elif self.hparams.task_type == "box_tracking":
            assert num_views == 1, f"Box tracking only supports 1 test view, got {num_views}"

            query_boxes = y['boxes'][:, :, 0]
            target, valid = self._box_targets(y)

            preds = self._timed_forward(x_input, query_boxes)
            m = self._box_metrics(preds, target, valid)
            self.metrics["test"]["iou_sum"] += m["iou_sum"]
            self.metrics["test"]["boxes"]   += m["count"]

            if self.hparams.reference_model_name == "ema":
                if self.hparams.run_type == "probe":
                    x_features = self.model(x_input)
                    y_hat_ref = self.reference_model(x_features, query_boxes=query_boxes)
                else:
                    y_hat_ref = self.reference_model(x_input, query_boxes=query_boxes)
                m_ref = self._box_metrics(y_hat_ref, target, valid)
                self.metrics["test"]["ref_iou_sum"] += m_ref["iou_sum"]
        else:
            raise NotImplementedError(f"Test step for task type '{self.hparams.task_type}' is not implemented.")

    def on_test_epoch_start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_test_epoch_end(self):
        self._aggregate_and_log_metrics(stage="test")

    def configure_optimizers(self):
        if (self.hparams.reference_model_name == "ema") and (self.ema_decay_schedule is None):
            self._ema_configure_schedulers()

        if self.hparams.run_type == "probe":
            return self._pooler_configure_optimizers()
        elif self.hparams.run_type == "finetune":
            if self.hparams.model_type == "video_swin":
                return self._swin_configure_optimizers()
            elif self.hparams.model_type == "video_mae":
                return self._mae_configure_optimizers()
            else:
                raise ValueError(f"Optimizer setting for model type: {self.hparams.model_type} and run type: {self.hparams.run_type} not implemented")
    
    def _swin_configure_optimizers(self):
        param_groups = [
            {'params': [p for n, p in self.model.backbone.named_parameters() 
                        if 'patch_embed.proj' in n and p.requires_grad], 
            'weight_decay': 0.0,
            'lr': self.hparams.lr * self.hparams.lr_multiplier},
            
            {'params': [p for n, p in self.model.backbone.named_parameters() 
                        if 'relative_position_bias_table' in n and p.requires_grad], 
            'weight_decay': 0.0,
            'lr': self.hparams.lr * self.hparams.lr_multiplier},
            
            {'params': [p for n, p in self.model.backbone.named_parameters() 
                        if 'norm' in n and p.requires_grad], 
            'weight_decay': 0.0,
            'lr': self.hparams.lr * self.hparams.lr_multiplier},
            
            {'params': [p for n, p in self.model.backbone.named_parameters() 
                        if p.requires_grad and 
                        'patch_embed.proj' not in n and 
                        'relative_position_bias_table' not in n and 
                        'norm' not in n],  
            'weight_decay': self.hparams.weight_decay,
            'lr': self.hparams.lr * self.hparams.lr_multiplier},

            {'params': [p for p in self.model.cls_head.parameters() if p.requires_grad],
            'weight_decay': self.hparams.weight_decay,
            'lr': self.hparams.lr}
        ]

        # optimizers
        optimizer = optim.AdamW(param_groups)
        
        # schedulers
        total_steps = self.trainer.estimated_stepping_batches
        if self.hparams.max_steps > 0:
            warmup_steps = self.hparams.warmup_steps
        else:
            warmup_steps = int(int(total_steps / self.hparams.max_epochs) * self.hparams.warmup_epochs)

        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_factor,
            total_iters=warmup_steps
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.hparams.min_lr
        )

        sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sequential_scheduler,
                "interval": "step",
                "frequency": 1,
            }}

    def _pooler_configure_optimizers(self):
        param_groups = [
            {'params': [p for n, p in self.pooler.named_parameters()], 
            'weight_decay': self.hparams.weight_decay,
            'lr': self.hparams.lr}
        ]

        if self.hparams.peft_type is not None:
            if self.hparams.peft_type == "full_finetune":
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters()],
                    'weight_decay': self.hparams.weight_decay,
                    'lr': self.hparams.lr
                })

            if "prompt" in self.hparams.peft_type or "vpt" in self.hparams.peft_type or "vfpt" in self.hparams.peft_type or "bpt" in self.hparams.peft_type:
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters() if "prompt_tokens" in n],
                    'weight_decay': self.hparams.weight_decay,
                    'lr': self.hparams.lr
                })

            if "adapter" in self.hparams.peft_type or "frame2freq" in self.hparams.peft_type or self.hparams.peft_type == "zeroi2v":
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters() if "adapter" in n],
                    'weight_decay': self.hparams.weight_decay,
                    'lr': self.hparams.lr
                })

            if self.hparams.peft_type == "zeroi2v":
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters() if "zeroi2v" in n],
                    'weight_decay': 0.0,
                    'lr': self.hparams.lr
                })

            if "lora" in self.hparams.peft_type:
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters() if "lora" in n or "st_lora" in n],
                    'weight_decay': self.hparams.weight_decay,
                    'lr': self.hparams.lr
                })

            if "dora" in self.hparams.peft_type:
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters() if "dora" in n],
                    'weight_decay': self.hparams.weight_decay,
                    'lr': self.hparams.lr
                })

            if self.hparams.peft_type == "ssf":
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters() if "ssf_scale" in n or "ssf_shift" in n],
                    'weight_decay': 0.0,
                    'lr': self.hparams.lr
                })

            if self.hparams.peft_unfreeze_norm or self.hparams.peft_type == "layernorm":
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters() if "norm" in n],
                    'weight_decay': 0.0,
                    'lr': self.hparams.lr
                })

            if self.hparams.peft_type == "bitfit":
                param_groups.append({
                    'params': [p for n, p in self.model.named_parameters() if "bias" in n],
                    'weight_decay': 0.0,
                    'lr': self.hparams.lr
                })

        # optimizers
        optimizer = optim.AdamW(param_groups)
        
        # schedulers
        total_steps = self.trainer.estimated_stepping_batches
        if self.hparams.max_steps > 0:
            warmup_steps = self.hparams.warmup_steps
        else:
            warmup_steps = int(int(total_steps / self.hparams.max_epochs) * self.hparams.warmup_epochs)

        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_factor,
            total_iters=warmup_steps
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.hparams.min_lr
        )

        sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sequential_scheduler,
                "interval": "step",
                "frequency": 1,
            }}
    
    def _mae_configure_optimizers(self):
        #param groups
        num_layers = self.model.get_num_layers()         
        ld_scales  = [self.hparams.layer_decay ** (num_layers + 1 - i)
                    for i in range(num_layers + 2)]
        ld_assign  = LayerDecayValueAssigner(ld_scales)

        skip = set(self.model.no_weight_decay()) if hasattr(self.model, "no_weight_decay") else set()
          
        group_map    = {}          

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # weight-decay
            use_decay = not (param.ndim < 2 or "bias" in name or "norm" in name or name in skip)
            wd        = 0.0 if not use_decay else self.hparams.weight_decay

            # layer id & lr scale
            lid       = ld_assign.get_layer_id(name)
            lr_scale  = ld_assign.get_scale(lid)

            key = (lid, use_decay)      
            if key not in group_map:
                group_map[key] = {
                    "params":       [],
                    "weight_decay": wd,
                    "lr":           self.hparams.lr * lr_scale,
                    "lr_scale":     lr_scale,         
                }
            group_map[key]["params"].append(param)

        param_groups = list(group_map.values())

        # optimizers
        optimizer = optim.AdamW(param_groups, lr=self.hparams.lr, betas=(self.hparams.optimizer_beta_1, self.hparams.optimizer_beta_2), eps=self.hparams.optimizer_epsilon)

        # schedulers
        total_steps = self.trainer.estimated_stepping_batches
        if self.hparams.max_steps > 0:
            warmup_steps = self.hparams.warmup_steps
        else:
            warmup_steps = int(int(total_steps / self.hparams.max_epochs) * self.hparams.warmup_epochs)

        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_lr / self.hparams.lr,
            total_iters=warmup_steps
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.hparams.min_lr
        )

        sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sequential_scheduler,
                "interval": "step",  
                "frequency": 1,
            }}

    @torch.no_grad()
    def _topk_correct(self, outputs, targets, topk=(1,)):
        """
        Compute the number of correct predictions for top-k accuracy.
        """
        with torch.no_grad():
            maxk = max(topk)
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t() 
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            correct_counts = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                correct_counts.append(correct_k.item())
        return correct_counts
    
    @torch.no_grad()
    def _topk_correct_epic(self, verb_outputs, verb_targets, noun_outputs, noun_targets, topk=(1,)):
        """
        Compute the number of correct predictions for top-k accuracy.
        """
        with torch.no_grad():
            maxk = max(topk)
            
            # Get top-k predictions for verbs
            _, verb_pred = verb_outputs.topk(maxk, dim=1, largest=True, sorted=True)
            verb_pred = verb_pred.t()  # (maxk, B)
            verb_correct = verb_pred.eq(verb_targets.view(1, -1).expand_as(verb_pred))  # (maxk, B)
            
            # Get top-k predictions for nouns
            _, noun_pred = noun_outputs.topk(maxk, dim=1, largest=True, sorted=True)
            noun_pred = noun_pred.t()  # (maxk, B)
            noun_correct = noun_pred.eq(noun_targets.view(1, -1).expand_as(noun_pred))  # (maxk, B)
            
            # Compute accuracies for each k
            action_correct_counts = []
            verb_correct_counts = []
            noun_correct_counts = []
            
            for k in topk:
                # Verb top-k correct
                verb_correct_k = verb_correct[:k].reshape(-1).float().sum(0)
                verb_correct_counts.append(verb_correct_k.item())
                
                # Noun top-k correct
                noun_correct_k = noun_correct[:k].reshape(-1).float().sum(0)
                noun_correct_counts.append(noun_correct_k.item())
                
                # Action top-k correct 
                verb_any_correct = verb_correct[:k].any(dim=0)  # (B,)
                noun_any_correct = noun_correct[:k].any(dim=0)  # (B,)
                action_correct_k = (verb_any_correct & noun_any_correct).float().sum(0)
                action_correct_counts.append(action_correct_k.item())
            
        return action_correct_counts, verb_correct_counts, noun_correct_counts
    
    @torch.no_grad()
    def _depth_metrics(self, pred, target):
        valid = (target > self.hparams.min_depth) & (target < self.hparams.max_depth)
        pred = pred[valid].clamp(min=self.hparams.min_depth, max=self.hparams.max_depth)
        target = target[valid]
        n = target.numel()
        if n == 0:
            return {"abs_rel": 0.0, "delta1": 0.0, "count": 0}
        thresh = torch.max(pred / target, target / pred)
        return {
            "abs_rel": (torch.abs(pred - target) / target).sum().item(),
            "delta1":  (thresh < 1.25).float().sum().item(),
            "count":   n,
        }

    def _box_targets(self, y):
        """Boxes and mask for the frames after the query frame, normalized to [0, 1]."""
        return y['boxes'][:, :, 1:] / self.hparams.input_size, y['valid'][:, :, 1:]

    @torch.no_grad()
    def _box_metrics(self, pred, target, valid):
        # pred/target: (B, Q, T-1, 4) xyxy, valid: (B, Q, T-1)
        pred, target = pred[valid], target[valid]
        if pred.numel() == 0:
            return {"iou_sum": 0.0, "count": 0}

        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
        area_tgt = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = area_pred + area_tgt - inter

        iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
        return {"iou_sum": iou.sum().item(), "count": iou.numel()}

    @torch.no_grad()
    def _segmentation_metrics(self, pred, target):
        # pred:   (B, num_classes, T, H, W) 
        # target: (B, T, H, W), ignore_index void label
        num_classes = self.hparams.num_classes
        pred_labels = pred.argmax(dim=1)
        valid = target != self.hparams.seg_void_label
        pred_flat = pred_labels[valid]
        tgt_flat = target[valid]

        intersection = torch.zeros(num_classes, device=pred.device)
        union = torch.zeros(num_classes, device=pred.device)
        for c in range(num_classes):
            pred_c = pred_flat == c
            tgt_c = tgt_flat == c
            intersection[c] = (pred_c & tgt_c).sum()
            union[c] = (pred_c | tgt_c).sum()
        return {"intersection": intersection, "union": union}

    @torch.no_grad()
    def _aggregate_and_log_metrics(self, stage):
        """
        Gathers local counters across all ranks, computes final top-1/top-5,
        logs them (with sync_dist), and resets local counters.

        Args:
            stage (str): "val" or "test"
        """
        if stage not in self.metrics:
            raise ValueError(f"Unknown stage: {stage}")

        local_list = []

        if self.hparams.task_type in ["classification", "classification_epic"]:
            local_list += [self.metrics[stage]["samples"], self.metrics[stage]["top1_correct"], self.metrics[stage]["top5_correct"]]

            if self.hparams.task_type == "classification_epic":
                local_list += [self.metrics[stage]["top1_correct_verb"], self.metrics[stage]["top5_correct_verb"], self.metrics[stage]["top1_correct_noun"], self.metrics[stage]["top5_correct_noun"]]

                if self.hparams.reference_model_name == "ema" and stage != "test":
                    local_list += [self.metrics[stage]["ref_top1_correct"], self.metrics[stage]["ref_top5_correct"]]
                elif self.hparams.reference_model_name == "ema" and stage == "test":
                    local_list += [self.metrics[stage]["samples_hd"], self.metrics[stage]["top1_hd"], self.metrics[stage]["top5_hd"], self.metrics[stage]["top1_hd_verb"], self.metrics[stage]["top5_hd_verb"], self.metrics[stage]["top1_hd_noun"], self.metrics[stage]["top5_hd_noun"]]
                    local_list += [self.metrics[stage]["ref_top1_correct"], self.metrics[stage]["ref_top5_correct"], self.metrics[stage]["ref_top1_hd"], self.metrics[stage]["ref_top5_hd"]]
                elif stage == "test":
                    local_list += [self.metrics[stage]["samples_hd"], self.metrics[stage]["top1_hd"], self.metrics[stage]["top5_hd"], self.metrics[stage]["top1_hd_verb"], self.metrics[stage]["top5_hd_verb"], self.metrics[stage]["top1_hd_noun"], self.metrics[stage]["top5_hd_noun"]]
            else:
                if self.hparams.reference_model_name == "ema":
                    local_list += [self.metrics[stage]["ref_top1_correct"], self.metrics[stage]["ref_top5_correct"]]

            # Gather from all GPUs
            local_counts = torch.tensor(
                local_list,
                dtype=torch.float,
                device=self.device
            )
            global_counts = self.all_gather(local_counts)

            if global_counts.dim() > 1: 
                global_counts = global_counts.sum(dim=0)

            total_samples = global_counts[0].item()

            top1_acc = global_counts[1].item() / total_samples if total_samples > 0 else 0.0
            top5_acc = global_counts[2].item() / total_samples if total_samples > 0 else 0.0
            self.log(f"{stage}_top1_acc", top1_acc, sync_dist=False)
            self.log(f"{stage}_top5_acc", top5_acc, sync_dist=False)

            # Reset local counters
            self.metrics[stage]["top1_correct"] = 0
            self.metrics[stage]["top5_correct"] = 0

            if self.hparams.task_type == "classification_epic":
                verb_top1_acc = global_counts[3].item() / total_samples if total_samples > 0 else 0.0
                verb_top5_acc = global_counts[4].item() / total_samples if total_samples > 0 else 0.0
                noun_top1_acc = global_counts[5].item() / total_samples if total_samples > 0 else 0.0
                noun_top5_acc = global_counts[6].item() / total_samples if total_samples > 0 else 0.0

                self.log(f"{stage}_verb_top1_acc", verb_top1_acc, sync_dist=False)
                self.log(f"{stage}_verb_top5_acc", verb_top5_acc, sync_dist=False)
                self.log(f"{stage}_noun_top1_acc", noun_top1_acc, sync_dist=False)
                self.log(f"{stage}_noun_top5_acc", noun_top5_acc, sync_dist=False)

                self.metrics[stage]["top1_correct_verb"] = 0
                self.metrics[stage]["top5_correct_verb"] = 0
                self.metrics[stage]["top1_correct_noun"] = 0
                self.metrics[stage]["top5_correct_noun"] = 0

                if self.hparams.reference_model_name == "ema" and stage != "test":
                    ref_top1_acc = global_counts[7].item() / total_samples if total_samples > 0 else 0.0
                    ref_top5_acc = global_counts[8].item() / total_samples if total_samples > 0 else 0.0
                    self.log(f"{stage}_ref_top1_acc", ref_top1_acc, sync_dist=False)
                    self.log(f"{stage}_ref_top5_acc", ref_top5_acc, sync_dist=False)
                    self.metrics[stage]["ref_top1_correct"] = 0
                    self.metrics[stage]["ref_top5_correct"] = 0
                elif self.hparams.reference_model_name == "ema" and stage == "test":
                    total_samples_hd = global_counts[7].item()
                    hd_top1_acc = global_counts[8].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_top5_acc = global_counts[9].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_verb_top1_acc = global_counts[10].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_verb_top5_acc = global_counts[11].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_noun_top1_acc = global_counts[12].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_noun_top5_acc = global_counts[13].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    ref_top1_acc = global_counts[14].item() / total_samples if total_samples > 0 else 0.0
                    ref_top5_acc = global_counts[15].item() / total_samples if total_samples > 0 else 0.0
                    hd_ref_top1_acc = global_counts[16].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_ref_top5_acc = global_counts[17].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    self.log(f"{stage}_hd_top1_acc", hd_top1_acc, sync_dist=False)
                    self.log(f"{stage}_hd_top5_acc", hd_top5_acc, sync_dist=False)
                    self.log(f"{stage}_hd_verb_top1_acc", hd_verb_top1_acc, sync_dist=False)
                    self.log(f"{stage}_hd_verb_top5_acc", hd_verb_top5_acc, sync_dist=False)
                    self.log(f"{stage}_hd_noun_top1_acc", hd_noun_top1_acc, sync_dist=False)
                    self.log(f"{stage}_hd_noun_top5_acc", hd_noun_top5_acc, sync_dist=False)
                    self.log(f"{stage}_ref_top1_acc", ref_top1_acc, sync_dist=False)
                    self.log(f"{stage}_ref_top5_acc", ref_top5_acc, sync_dist=False)
                    self.log(f"{stage}_hd_ref_top1_acc", hd_ref_top1_acc, sync_dist=False)
                    self.log(f"{stage}_hd_ref_top5_acc", hd_ref_top5_acc, sync_dist=False)

                    self.metrics[stage]["samples_hd"] = 0
                    self.metrics[stage]["top1_hd"] = 0
                    self.metrics[stage]["top5_hd"] = 0
                    self.metrics[stage]["top1_hd_verb"] = 0
                    self.metrics[stage]["top5_hd_verb"] = 0
                    self.metrics[stage]["top1_hd_noun"] = 0
                    self.metrics[stage]["top5_hd_noun"] = 0
                    self.metrics[stage]["ref_top1_hd"] = 0
                    self.metrics[stage]["ref_top5_hd"] = 0
                    self.metrics[stage]["ref_top1_correct"] = 0
                    self.metrics[stage]["ref_top5_correct"] = 0
                elif stage == "test":
                    total_samples_hd = global_counts[7].item()
                    hd_top1_acc = global_counts[8].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_top5_acc = global_counts[9].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_verb_top1_acc = global_counts[10].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_verb_top5_acc = global_counts[11].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_noun_top1_acc = global_counts[12].item() / total_samples_hd if total_samples_hd > 0 else 0.0
                    hd_noun_top5_acc = global_counts[13].item() / total_samples_hd if total_samples_hd > 0 else 0.0

                    self.log(f"{stage}_hd_top1_acc", hd_top1_acc, sync_dist=False)
                    self.log(f"{stage}_hd_top5_acc", hd_top5_acc, sync_dist=False)
                    self.log(f"{stage}_hd_verb_top1_acc", hd_verb_top1_acc, sync_dist=False)
                    self.log(f"{stage}_hd_verb_top5_acc", hd_verb_top5_acc, sync_dist=False)
                    self.log(f"{stage}_hd_noun_top1_acc", hd_noun_top1_acc, sync_dist=False)
                    self.log(f"{stage}_hd_noun_top5_acc", hd_noun_top5_acc, sync_dist=False)

                    self.metrics[stage]["samples_hd"] = 0
                    self.metrics[stage]["top1_hd"] = 0
                    self.metrics[stage]["top5_hd"] = 0
                    self.metrics[stage]["top1_hd_verb"] = 0
                    self.metrics[stage]["top5_hd_verb"] = 0
                    self.metrics[stage]["top1_hd_noun"] = 0
                    self.metrics[stage]["top5_hd_noun"] = 0            
            else:
                if self.hparams.reference_model_name == "ema":
                    ref_top1_acc = global_counts[3].item() / total_samples if total_samples > 0 else 0.0
                    ref_top5_acc = global_counts[4].item() / total_samples if total_samples > 0 else 0.0
                    self.log(f"{stage}_ref_top1_acc", ref_top1_acc, sync_dist=False)
                    self.log(f"{stage}_ref_top5_acc", ref_top5_acc, sync_dist=False)
                    self.metrics[stage]["ref_top1_correct"] = 0
                    self.metrics[stage]["ref_top5_correct"] = 0
            
            self.metrics[stage]["samples"] = 0
        elif self.hparams.task_type == "depth_estimation":
            local_list += [
                self.metrics[stage]["valid_pixels"],
                self.metrics[stage]["abs_rel_sum"],
                self.metrics[stage]["delta1_correct"],
            ]
            if self.hparams.reference_model_name == "ema":
                local_list += [
                    self.metrics[stage]["ref_abs_rel_sum"],
                    self.metrics[stage]["ref_delta1_correct"],
                ]

            local_counts = torch.tensor(local_list, dtype=torch.float, device=self.device)
            global_counts = self.all_gather(local_counts)
            if global_counts.dim() > 1:
                global_counts = global_counts.sum(dim=0)

            N = global_counts[0].item()
            if N > 0:
                self.log(f"{stage}_abs_rel", global_counts[1].item() / N, sync_dist=False)
                self.log(f"{stage}_delta1",  global_counts[2].item() / N, sync_dist=False)
                if self.hparams.reference_model_name == "ema":
                    self.log(f"{stage}_ref_abs_rel", global_counts[3].item() / N, sync_dist=False)
                    self.log(f"{stage}_ref_delta1",  global_counts[4].item() / N, sync_dist=False)

            # reset
            self.metrics[stage]["abs_rel_sum"]        = 0.0
            self.metrics[stage]["delta1_correct"]     = 0.0
            self.metrics[stage]["valid_pixels"]       = 0
            self.metrics[stage]["ref_abs_rel_sum"]    = 0.0
            self.metrics[stage]["ref_delta1_correct"] = 0.0
        elif self.hparams.task_type == "semantic_segmentation":
            nc = self.hparams.num_classes
            inter = self.metrics[stage]["intersection"]
            union = self.metrics[stage]["union"]
            local_t = torch.cat([inter, union])  # (2*nc,)
            if self.hparams.reference_model_name == "ema":
                local_t = torch.cat([local_t,
                                     self.metrics[stage]["ref_intersection"],
                                     self.metrics[stage]["ref_union"]])  # (4*nc,)

            global_counts = self.all_gather(local_t.to(self.device))
            if global_counts.dim() > 1:
                global_counts = global_counts.sum(dim=0)

            g_inter = global_counts[:nc]
            g_union = global_counts[nc:2*nc]
            iou     = torch.where(g_union > 0, g_inter / g_union, torch.zeros_like(g_inter))
            valid   = g_union > 0
            mIoU    = iou[valid].mean().item() if valid.any() else 0.0
            self.log(f"{stage}_mIoU", mIoU, sync_dist=False)

            if self.hparams.reference_model_name == "ema":
                r_inter = global_counts[2*nc:3*nc]
                r_union = global_counts[3*nc:4*nc]
                r_iou   = torch.where(r_union > 0, r_inter / r_union, torch.zeros_like(r_inter))
                r_valid = r_union > 0
                ref_mIoU = r_iou[r_valid].mean().item() if r_valid.any() else 0.0
                self.log(f"{stage}_ref_mIoU", ref_mIoU, sync_dist=False)
                self.metrics[stage]["ref_intersection"] = torch.zeros(nc)
                self.metrics[stage]["ref_union"]        = torch.zeros(nc)

            self.metrics[stage]["intersection"] = torch.zeros(nc)
            self.metrics[stage]["union"]        = torch.zeros(nc)
        elif self.hparams.task_type == "box_tracking":
            local_list += [self.metrics[stage]["boxes"], self.metrics[stage]["iou_sum"]]
            if self.hparams.reference_model_name == "ema":
                local_list += [self.metrics[stage]["ref_iou_sum"]]

            local_counts = torch.tensor(local_list, dtype=torch.float, device=self.device)
            global_counts = self.all_gather(local_counts)
            if global_counts.dim() > 1:
                global_counts = global_counts.sum(dim=0)

            N = global_counts[0].item()
            if N > 0:
                self.log(f"{stage}_miou", global_counts[1].item() / N, sync_dist=False)
                if self.hparams.reference_model_name == "ema":
                    self.log(f"{stage}_ref_miou", global_counts[2].item() / N, sync_dist=False)

            # reset
            self.metrics[stage]["iou_sum"]     = 0.0
            self.metrics[stage]["boxes"]       = 0
            self.metrics[stage]["ref_iou_sum"] = 0.0

        # Inference throughput (test only): sum units across ranks (parallel
        # work), take max time as wall-clock (slowest rank dictates real elapsed).
        if stage == "test":
            t_local = torch.tensor(self.metrics[stage]["throughput_time"], dtype=torch.float64, device=self.device)
            u_local = torch.tensor(self.metrics[stage]["throughput_units"], dtype=torch.float64, device=self.device)
            t_all = self.all_gather(t_local)
            u_all = self.all_gather(u_local)
            wall = t_all.max().item()
            units = u_all.sum().item()
            if wall > 0 and units > 0:
                key = "test_frames_per_sec" if self.hparams.task_type in ("depth_estimation", "semantic_segmentation") else "test_clips_per_sec"
                self.log(key, units / wall, sync_dist=False)
            self.metrics[stage]["throughput_time"] = 0.0
            self.metrics[stage]["throughput_units"] = 0

            # Peak GPU memory (test only): max across ranks, since a single
            # rank OOMing sinks the whole run regardless of the others.
            if torch.cuda.is_available():
                alloc_local = torch.tensor(torch.cuda.max_memory_allocated(self.device), dtype=torch.float64, device=self.device)
                reserved_local = torch.tensor(torch.cuda.max_memory_reserved(self.device), dtype=torch.float64, device=self.device)
                alloc_all = self.all_gather(alloc_local)
                reserved_all = self.all_gather(reserved_local)
                self.log("test_peak_mem_allocated_gb", alloc_all.max().item() / (1024**3), sync_dist=False)
                self.log("test_peak_mem_reserved_gb", reserved_all.max().item() / (1024**3), sync_dist=False)

    @torch.no_grad()
    def _update_learnability_model(self):
        if self.learnability_model is None:
            return
        
        # in-place copy of params
        for tgt, src in self._learnability_param_pairs:
            if tgt.dtype == src.dtype:
                tgt.data.copy_(src.data, non_blocking=True)
            else:
                if torch.is_floating_point(tgt) and torch.is_floating_point(src):
                    tgt.data.copy_(src.data.to(tgt.dtype), non_blocking=True)
                else:
                    tgt.data.copy_(src.data, non_blocking=True)

        for tgt, src in self._learnability_buffer_pairs:
            if tgt.dtype == src.dtype:
                tgt.data.copy_(src.data, non_blocking=True)
            else:
                if torch.is_floating_point(tgt) and torch.is_floating_point(src):
                    tgt.data.copy_(src.data.to(tgt.dtype), non_blocking=True)
                else:
                    tgt.data.copy_(src.data, non_blocking=True)
    
    @torch.no_grad()
    def _update_reference_model(self):
        if self.hparams.reference_model_name != "ema":
            return

        ema_decay = self.ema_decay_schedule[self.global_step]
        
        # ema update parameters
        for tgt, src in self._ema_param_pairs:
            if tgt.dtype == src.dtype:
                tgt.data.mul_(ema_decay).add_(src.data, alpha=1 - ema_decay)
            else:
                src_cast = src.data.to(tgt.dtype)
                tgt.data.mul_(ema_decay).add_(src_cast, alpha=1 - ema_decay)
        
        for tgt, src in self._ema_buffer_pairs:
            if tgt.dtype == src.dtype:
                tgt.data.copy_(src.data)
            else:
                tgt.data.copy_(src.data.to(tgt.dtype))

    @torch.no_grad()
    def _ema_configure_schedulers(self):
        total_steps = self.trainer.estimated_stepping_batches
        steps = np.arange(total_steps)
        self.ema_decay_schedule = self.hparams.ema_decay_end + 0.5 * (self.hparams.ema_decay_start - self.hparams.ema_decay_end) * (1 + np.cos(np.pi * steps / total_steps))

    @torch.no_grad()
    def on_before_zero_grad(self, optimizer):
        self._update_learnability_model()
        self._update_reference_model()

    @torch.no_grad()
    def on_train_epoch_end(self):      
        if self.hparams.jest_warmup_epochs > 0:
            self.log('jest_interval', self.hparams.jest_interval, on_epoch=True, sync_dist=False)
            self.hparams.jest_interval = ((self.current_epoch + 1) // self.hparams.jest_warmup_epochs) + 1