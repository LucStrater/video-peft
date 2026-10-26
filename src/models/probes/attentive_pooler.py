"""
attentive_pooler.py

This file contains the source code for frozen pretrained 
image and video encoders with VJEPA2 attentive pooling.

Original Source: 
- VJEPA2: https://github.com/facebookresearch/vjepa2 
"""

# basic
import math
import numpy as np
import torch
import torch.nn as nn
import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

# utils
from src.utils.attentive_pooler_utils import Block, CrossAttention, CrossAttentionBlock, FourierFeatures, trunc_normal_
from src.utils.internvideo_next_utils import get_3d_sincos_pos_embed
from src.models.probes.dist import DiSTNetwork
from src.models.probes.dpt import DPTHead
from src.models.probes.vda import DPTHeadTemporal

class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
        use_rope=False,
        grid_size=16,
        drop_path=0.0,
        learned_queries=True,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim)) if learned_queries else None

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, drop_path=drop_path
            )
        else:
            self.cross_attention_block = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                        use_rope=use_rope,
                        grid_size=grid_size,
                        drop_path=drop_path,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        if self.query_tokens is not None:
            trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, queries=None):
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, False, None, use_reentrant=False)
                else:
                    x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1) if queries is None else queries
        q = self.cross_attention_block(q, x)
        return q

class EfficientPooler(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 32,
        d_out: int = 1,
        num_output_queries: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.d_out = d_out
        self.num_output_queries = num_output_queries
        self.num_queries = num_queries * num_output_queries
        
        self.v = nn.Linear(dim, (dim // d_out) * self.num_output_queries, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, self.num_queries, dim) * 0.02)
        
    def forward(self, x: torch.Tensor, cls=None, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        C_prime = C // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(B, -1, -1) 

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = (x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        q = q * self.scale
        v = (self.v(x).reshape(B, N, self.num_queries, C * self.num_output_queries // (self.d_out * self.num_queries)).permute(0, 2, 1, 3))

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        x_cls = x_cls.view(B, self.num_output_queries, C_prime)
        
        return x_cls

class LinearPooler(nn.Module):
    def __init__(self, num_queries: int, method: str = "global_average_pooling"):
        super().__init__()
        self.num_queries = num_queries
        self.method = method

    def forward(self, x, queries=None):
        if self.method == "global_average_pooling":
            x = x.mean(dim=1, keepdim=True)
            if self.num_queries > 1:
                x = x.repeat(1, self.num_queries, 1)             

        return x        

class Probe(nn.Module):
    """Attentive Classifier with optional dual-head support for EPIC datasets and linear probe mode"""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        use_activation_checkpointing=False,
        use_rope=False,
        use_pos_embed=True,
        num_frames=16,
        probe_clip_len=None,
        input_size=224,
        grid_size=16,
        model_tubelet_size=1,
        drop_path_rate=0.0,
        task_type="classification",
        num_verb_classes=97,
        num_noun_classes=300,
        pooler_type="attentive",
        depth_patch_size=8,
        depth_tubelet_size=2,
        max_depth=10.0,
        min_depth=0.001,
        backbone_depth=None,
        max_queries=25,
        box_fourier_bases=16,
    ):
        super().__init__()
        # probe_clip_len: frames the probe attends over per call (attentive-pooler context,
        # pos-embed size). Defaults to num_frames (= clip_len) when not set.
        _probe_clip_len = probe_clip_len if probe_clip_len is not None else num_frames
        self.probe_clip_len = _probe_clip_len
        self.decouple = (_probe_clip_len != num_frames)
        self.t_size = _probe_clip_len // model_tubelet_size
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.task_type = task_type
        self.pooler_type = pooler_type
        self.input_size = input_size
        self.max_depth = max_depth

        # dense prediction (depth / segmentation) 
        self.num_frames = num_frames
        self.depth_patch_size = depth_patch_size
        depth_tubelet_size = min(depth_tubelet_size, _probe_clip_len) if num_frames > 1 else 1
        self.depth_tubelet_size = depth_tubelet_size
        self.depth_grid_size = input_size // depth_patch_size
        self.depth_t_size = num_frames // depth_tubelet_size          # full output temporal size
        self.probe_depth_t_size = _probe_clip_len // depth_tubelet_size  # per-chunk query temporal size

        if pooler_type == "dist":
            lin_embed_dim = 384
        elif pooler_type in ("dpt", "vda"):
            # DPTHead.mid_channels = features // 2 (features = 256 if embed_dim>=1024 else 128)
            lin_embed_dim = 128 if embed_dim >= 1024 else 64
        else:
            lin_embed_dim = embed_dim

        if self.task_type == "classification":
            num_queries = 1
            self.linear = nn.Linear(lin_embed_dim, num_classes, bias=True)
        elif self.task_type == "classification_epic":
            num_queries = 2
            self.linear_verb = nn.Linear(lin_embed_dim, num_verb_classes, bias=True)
            self.linear_noun = nn.Linear(lin_embed_dim, num_noun_classes, bias=True)
        elif self.task_type == "depth_estimation":
            num_queries = self.probe_depth_t_size * self.depth_grid_size * self.depth_grid_size
            if pooler_type not in ("dpt", "vda"):
                self.linear = nn.Linear(lin_embed_dim, depth_tubelet_size * depth_patch_size * depth_patch_size, bias=True)
        elif self.task_type == "semantic_segmentation":
            num_queries = self.probe_depth_t_size * self.depth_grid_size * self.depth_grid_size
            self.num_classes = num_classes
            if pooler_type not in ("dpt", "vda"):
                self.linear = nn.Linear(lin_embed_dim, num_classes * depth_tubelet_size, bias=True)
        elif self.task_type == "box_tracking":
            assert pooler_type == "attentive", f"box tracking requires the attentive pooler, got {pooler_type}"
            num_queries = max_queries
            self.fourier = FourierFeatures(in_features=4, num_bases=box_fourier_bases)
            self.box_query_mlp = nn.Sequential(
                nn.Linear(self.fourier.out_features, embed_dim, bias=True),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim, bias=True)
            )
            # frame 0 is the query itself, so only the frames after it are predicted
            self.linear = nn.Linear(lin_embed_dim, (num_frames - 1) * 4, bias=True)
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

        if pooler_type in ("attentive", "attentive_multilayer"):
            if pooler_type == "attentive_multilayer":
                assert backbone_depth is not None, "attentive_multilayer pooler requires backbone_depth"
                self.ml_layer_indices = [(i + 1) * backbone_depth // 4 - 1 for i in range(4)]
            self.pooler = AttentivePooler(
                num_queries=num_queries,
                learned_queries=(self.task_type != "box_tracking"),
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=depth,
                norm_layer=norm_layer,
                init_std=init_std,
                qkv_bias=qkv_bias,
                complete_block=complete_block,
                use_activation_checkpointing=use_activation_checkpointing,
                use_rope=use_rope,
                grid_size=grid_size,
                drop_path=drop_path_rate,
            )
        elif pooler_type == "dist":
            self.dist_network = DiSTNetwork(
                num_frames=_probe_clip_len,
                d_model=embed_dim,
                num_layers=backbone_depth,
                grid_size=grid_size,
                spatial_patch_size=input_size // grid_size,
                sparse_sample_alpha=model_tubelet_size,
            )
            self.pooler = AttentivePooler(
                num_queries=num_queries,
                embed_dim=lin_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=depth,
                norm_layer=norm_layer,
                init_std=init_std,
                qkv_bias=qkv_bias,
                complete_block=complete_block,
                use_activation_checkpointing=use_activation_checkpointing,
                use_rope=use_rope,
                grid_size=grid_size,
                drop_path=drop_path_rate,
            )
        elif pooler_type == "efficient":
            self.pooler = EfficientPooler(dim=embed_dim, num_output_queries=num_queries)
        elif pooler_type == "linear":
            method = "spatial" if self.task_type in ("depth_estimation", "semantic_segmentation") else "global_average_pooling"
            self.pooler = LinearPooler(num_queries=num_queries, method=method)
        elif pooler_type in ("dpt", "vda"):
            assert backbone_depth is not None, f"{pooler_type} pooler requires backbone_depth"
            # Uniformly sample 4 layers from the backbone (including the last)
            self.dpt_layer_indices = [(i + 1) * backbone_depth // 4 - 1 for i in range(4)]
            head_cls = DPTHead if pooler_type == "dpt" else DPTHeadTemporal
            self.pooler = head_cls(
                in_channels=embed_dim,
                num_frames=num_frames,
                model_tubelet_size=model_tubelet_size,
                grid_size=grid_size,
                input_size=input_size,
                task_type=self.task_type,
                num_classes=num_classes,
                max_depth=max_depth,
            )
        else:
            raise ValueError(f"Invalid pooler type: {pooler_type}")

        if use_pos_embed and (pooler_type not in ("dist", "dpt", "vda")):
            self.pos_embed = nn.Parameter(torch.zeros(1, self.t_size * grid_size * grid_size, embed_dim))
            self.init_pos_embed()
            self.pos_embed.requires_grad_(False)
        else:
            self.pos_embed = None
            

    def init_pos_embed(self):
        pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.embed_dim, 
                grid_size=self.grid_size, 
                t_size=self.t_size, 
                cls_token=False
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, images=None, query_boxes=None):
        if self.pooler_type == "dist":
            B_orig = x[0].shape[0]
            HW = self.grid_size * self.grid_size
            true_T = x[0].shape[1] // HW  # actual backbone temporal-token count (independent of probe_clip_len)
            T = self.t_size

            if self.decouple and T < true_T:
                K = true_T // T
                mid_features = []
                for feat in x:
                    f = feat.reshape(B_orig, K, T, HW, -1).reshape(B_orig * K, T, HW, -1)
                    f = f.permute(2, 0, 1, 3).reshape(HW, B_orig * K * T, -1)  # (HW, B*K*T, C)
                    mid_features.append(f)
                C_img, H_img, W_img = images.shape[1], images.shape[3], images.shape[4]
                raw_per_chunk = images.shape[2] // K
                images_k = images.reshape(B_orig, C_img, K, raw_per_chunk, H_img, W_img)
                images_k = images_k.permute(0, 2, 1, 3, 4, 5).reshape(B_orig * K, C_img, raw_per_chunk, H_img, W_img)
                x = self.dist_network(mid_features, images_k)  # (B*K, T*HW, integration_dim)
                x = self.pooler(x)                              # (B*K, num_queries_per_chunk, D)
                if self.task_type in ("classification", "classification_epic"):
                    x = x.reshape(B_orig, K, x.shape[1], x.shape[2]).mean(dim=1)  # (B, num_queries, D)
                else:
                    x = x.reshape(B_orig, K * x.shape[1], x.shape[2])  # (B, K*num_queries_per_chunk, D)
            else:
                mid_features = []
                for feat in x:
                    f = feat.reshape(B_orig, T, HW, -1).permute(2, 0, 1, 3).reshape(HW, B_orig * T, -1)  # (HW, B*T, C)
                    mid_features.append(f)
                x = self.dist_network(mid_features, images)  # (B, T*HW, integration_dim)
                x = self.pooler(x)                            # (B, num_queries, integration_dim)
        elif self.pooler_type in ("dpt", "vda"):
            x = self.pooler([x[i] for i in self.dpt_layer_indices])
            if self.task_type == "classification":
                return self.linear(x)  # (B, mid) -> (B, num_classes)
            return x
        elif self.pooler_type == "attentive_multilayer":
            assert not self.decouple, "attentive_multilayer requires probe_clip_len == clip_len"
            assert self.task_type != "box_tracking", "attentive_multilayer does not support box tracking"
            feats = [x[i] for i in self.ml_layer_indices]
            if self.pos_embed is not None:
                feats = [f + self.pos_embed for f in feats]
            x = self.pooler(torch.cat(feats, dim=1))
        else:
            queries = None
            if self.task_type == "box_tracking":
                assert not self.decouple, "box tracking requires probe_clip_len == clip_len"
                queries = self.box_query_mlp(self.fourier(query_boxes / self.input_size))  # (B, Q, D)

            if self.decouple and self.probe_clip_len < self.num_frames:
                # N→B split: (B, K*chunk_tokens, D) → (B*K, chunk_tokens, D)
                B_orig = x.shape[0]
                chunk_tokens = self.t_size * self.grid_size * self.grid_size
                K = x.shape[1] // chunk_tokens
                x = x.reshape(B_orig * K, chunk_tokens, x.shape[-1])
                if self.pos_embed is not None:
                    x = x + self.pos_embed
                x = self.pooler(x)  # (B*K, num_queries_per_chunk, D)
                if self.task_type in ("classification", "classification_epic"):
                    x = x.reshape(B_orig, K, x.shape[1], x.shape[2]).mean(dim=1)  # (B, num_queries, D)
                else:
                    x = x.reshape(B_orig, K * x.shape[1], x.shape[2])  # (B, K*num_queries_per_chunk, D)
            else:
                if self.pos_embed is not None:
                    x = x + self.pos_embed
                x = self.pooler(x, queries=queries)

        if self.task_type == "classification":
            x = x.squeeze(1)
            x = self.linear(x)
            return x
        elif self.task_type == "classification_epic":
            x_verb = self.linear_verb(x[:, 0, :])
            x_noun = self.linear_noun(x[:, 1, :])
            return x_verb, x_noun
        elif self.task_type == "depth_estimation":
            B = x.shape[0]
            x = self.linear(x)  # (B, num_queries, depth_tubelet_size * depth_patch_size * depth_patch_size)

            # (B, T*H*W, pt*ph*pw) -> (B, T, H, W, pt, ph, pw)
            x = x.reshape(B, self.depth_t_size, self.depth_grid_size, self.depth_grid_size, self.depth_tubelet_size, self.depth_patch_size, self.depth_patch_size)
            # -> (B, T*pt, H*ph, W*pw)
            x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.depth_t_size * self.depth_tubelet_size, self.depth_grid_size * self.depth_patch_size, self.depth_grid_size * self.depth_patch_size)

            x = torch.sigmoid(x) * self.max_depth # scale to [0, max_depth] because negative depths are not possible and give NaN losses

            return x  # (B, num_frames, input_size, input_size)
        elif self.task_type == "semantic_segmentation":
            B = x.shape[0]
            x = self.linear(x)  # (B, num_queries, num_classes * depth_tubelet_size)
            x = x.reshape(B, self.depth_t_size, self.depth_grid_size, self.depth_grid_size, self.num_classes, self.depth_tubelet_size) # (B, T, depth_grid_size, depth_grid_size, num_classes, depth_tubelet_size)
            x = x.permute(0, 1, 5, 4, 2, 3).reshape(B * self.num_frames, self.num_classes, self.depth_grid_size, self.depth_grid_size)  # (B*T, num_classes, H_q, W_q)
            x = torch.nn.functional.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)       # (B*T, num_classes, input_size, input_size)
            x = x.reshape(B, self.num_frames, self.num_classes, self.input_size, self.input_size).permute(0, 2, 1, 3, 4)                # (B, num_classes, T, input_size, input_size)
            return x
        elif self.task_type == "box_tracking":
            B, Q = x.shape[0], x.shape[1]
            x = self.linear(x)  # (B, Q, (num_frames - 1) * 4), no activation
            return x.reshape(B, Q, self.num_frames - 1, 4)  # boxes normalized to [0, 1]
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
            