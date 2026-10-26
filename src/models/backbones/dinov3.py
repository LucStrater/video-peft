# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import sys
import numpy as np
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from src.utils.dinov3_layers import LayerScale, Mlp, PatchEmbed, RopePositionEmbedding, SelfAttention
from src.utils.dinov3_utils import named_apply, get_sinusoid_encoding_table, cat_keep_shapes, uncat_with_shapes
from src.models.peft.adaptformer import AdaptFormer
from src.models.peft.lora import LoRA, LoRAFormer
from src.models.peft.adapter import Adapter
from src.models.peft.aim import AIMAdapter
from src.models.peft.zeroi2v import LinearAdapter, TemporalEmbedding, temporal_head_shift


class LoRASelfAttention(SelfAttention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0, mask_k_bias=False, device=None,
                 peft_type=None, peft_rank=0, peft_dropout=0.0, num_frames=16):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                         attn_drop=attn_drop, proj_drop=proj_drop,
                         mask_k_bias=mask_k_bias, device=device)

        self.lora = None
        self.stdha = False
        self.loraformer = None
        self.loraformer_alpha = 1.0
        if peft_type == "lora":
            self.lora = LoRA(dim, num_heads, rank=peft_rank)
        elif peft_type == "zeroi2v":
            self.stdha = True
        elif peft_type == "loraformer":
            self.loraformer = LoRAFormer(dim, num_heads, rank=peft_rank, num_frames=num_frames)

    def forward(self, x: Tensor, attn_bias=None, rope=None, T=None) -> Tensor:
        qkv = self.qkv(x)
        if self.lora is not None:
            assert attn_bias is None
            B, N, _ = qkv.shape
            C = self.qkv.in_features
            qkv_r = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv_r, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            q, k, v = self.lora(x, q, k, v)
            if rope is not None:
                q, k = self.apply_rope(q, k, rope)
            attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn_v = attn_v.transpose(1, 2).reshape([B, N, C])
        elif self.stdha and T is not None:
            assert attn_bias is None
            B, N, _ = qkv.shape
            C = self.qkv.in_features
            qkv_r = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv_r, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            k = temporal_head_shift(k, T)
            v = temporal_head_shift(v, T)
            if rope is not None:
                q, k = self.apply_rope(q, k, rope)
            attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn_v = attn_v.transpose(1, 2).reshape([B, N, C])
        elif self.loraformer is not None:
            assert attn_bias is None
            B, N, _ = qkv.shape
            C = self.qkv.in_features
            qkv_r = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv_r, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            attn_v_loraformer = self.loraformer(x)
            if rope is not None:
                q, k = self.apply_rope(q, k, rope)
            attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn_v = attn_v.transpose(1, 2).reshape([B, N, C]) + self.loraformer_alpha * attn_v_loraformer
        else:
            attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        if self.lora is None:
            return super().forward_list(x_list, attn_bias=attn_bias, rope_list=rope_list)
        assert len(x_list) == len(rope_list)
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for x, qkv, rope in zip(x_list, qkv_list, rope_list):
            B, N, C = x.shape
            head_dim = C // self.num_heads
            qkv_r = qkv.reshape(B, N, 3, self.num_heads, head_dim)
            q, k, v = torch.unbind(qkv_r, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            q, k, v = self.lora(x, q, k, v)
            if rope is not None:
                q, k = self.apply_rope(q, k, rope)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            att_out.append(out.transpose(1, 2).reshape(B, N, C))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=LoRASelfAttention,
        ffn_layer=Mlp,
        mask_k_bias: bool = False,
        device=None,
        peft_type: str = None,
        peft_rank: int = 0,
        peft_dropout: float = 0.0,
        num_frames: int = 16,
        backbone_clip_len: int = 1,
        peft_clip_len: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
            peft_type=peft_type,
            peft_rank=peft_rank,
            peft_dropout=peft_dropout,
            num_frames=num_frames,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

        self.adapter_parallel_mlp = None
        self.adapter_pre_attn = None
        self.adapter_pre_mlp = None
        self.aim_t_adapter = None
        self.aim_s_adapter = None
        self.aim_mlp_adapter = None
        self.t_in_adapter = None
        self.s_adapter = None
        self.mlp_adapter = None
        self.mlp_out_adapter = None
        if peft_type == "adaptformer_adapter":
            self.adapter_parallel_mlp = AdaptFormer(in_dim=dim, bottleneck=peft_rank, dropout=peft_dropout)
        elif peft_type == "adapter":
            self.adapter_pre_attn = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="adapter", act_fn=nn.GELU)
            self.adapter_pre_mlp = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="adapter", act_fn=nn.GELU)
        elif peft_type == "st-adapter":
            self.adapter_pre_attn = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="st-adapter",
                                               backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
            self.adapter_pre_mlp = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="st-adapter",
                                              backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
        elif peft_type == "frame2freq_ms":
            self.adapter_pre_attn = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="frame2freq_ms",
                                               backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
            self.adapter_pre_mlp = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="frame2freq_ms",
                                               backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
        elif peft_type == "aim_adapter":
            self.aim_t_adapter = AIMAdapter(dim, peft_rank, skip_connect=False)
            self.aim_s_adapter = AIMAdapter(dim, peft_rank, skip_connect=True)
            self.aim_mlp_adapter = AIMAdapter(dim, peft_rank, skip_connect=False)
        elif peft_type == "zeroi2v":
            self.t_in_adapter = LinearAdapter(dim, peft_rank)
            self.s_adapter = LinearAdapter(dim, peft_rank)
            self.mlp_adapter = LinearAdapter(dim, peft_rank)
            self.mlp_out_adapter = LinearAdapter(dim, peft_rank)

    def _apply_adapter(self, adapter, x, T, hw):
        if T is None or hw is None:
            return x
        H_patches, W_patches = hw
        n_patches = H_patches * W_patches
        n_prefix = x.shape[1] - n_patches
        if n_prefix > 0:
            prefix, patches = x[:, :n_prefix], x[:, n_prefix:]
            patches = adapter(patches, T, H_patches, W_patches, frames_as_batch=True)
            return torch.cat([prefix, patches], dim=1)
        return adapter(x, T, H_patches, W_patches, frames_as_batch=True)

    def _apply_aim_temporal(self, x, T):
        BT, N1, D = x.shape
        B = BT // T
        # (BT, N1, D) -> (B*N1, T, D): each spatial token attends over T frames
        xt = x.view(B, T, N1, D).permute(0, 2, 1, 3).contiguous().view(B * N1, T, D)
        xt = self.aim_t_adapter(self.attn(self.norm1(xt)))  # no rope for temporal
        xt = xt.view(B, N1, T, D).permute(0, 2, 1, 3).contiguous().view(BT, N1, D)
        return x + xt

    @staticmethod
    def _maybe_index_rope(rope, indices):
        if rope is None:
            return None
        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        else:
            return sin, cos

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]
            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)
            x_attn = torch.index_add(
                x, dim=0, source=self.ls1(residual_1), index=indices_1, alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]
            x_subset_2 = x_attn[indices_2]
            x_norm2_subset = self.norm2(x_subset_2)
            residual_2 = self.mlp(x_norm2_subset)
            if self.adapter_parallel_mlp is not None:
                residual_2 = residual_2 + self.adapter_parallel_mlp(x_norm2_subset)
            x_ffn = torch.index_add(
                x_attn, dim=0, source=self.ls2(residual_2), index=indices_2, alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_norm2 = self.norm2(x_attn)
            x_mlp = self.mlp(x_norm2)
            if self.adapter_parallel_mlp is not None:
                x_mlp = x_mlp + self.adapter_parallel_mlp(x_norm2)
            x_ffn = x_attn + self.ls2(x_mlp)

        return x_ffn

    def _forward_list(self, x_list, rope_list=None, T=None, hw_list=None):
        if hw_list is None:
            hw_list = [None] * len(x_list)

        if self.adapter_pre_attn is not None:
            x_list = [
                self._apply_adapter(self.adapter_pre_attn, x, T, hw)
                for x, hw in zip(x_list, hw_list)
            ]
        if self.aim_t_adapter is not None and T is not None:
            x_list = [self._apply_aim_temporal(x, T) for x in x_list]

        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / s for b, s in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            if self.aim_s_adapter is not None:
                residual_1_list = [self.aim_s_adapter(r) for r in residual_1_list]

            x_attn_list = [
                torch.index_add(
                    x, dim=0, source=self.ls1(residual_1), index=indices_1, alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            if self.adapter_pre_mlp is not None:
                x_attn_list = [
                    self._apply_adapter(self.adapter_pre_mlp, x, T, hw)
                    for x, hw in zip(x_attn_list, hw_list)
                ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)
            if self.adapter_parallel_mlp is not None:
                residual_2_list = [
                    r + self.adapter_parallel_mlp(norm2)
                    for r, norm2 in zip(residual_2_list, norm2_list)
                ]
            if self.aim_mlp_adapter is not None:
                residual_2_list = [
                    r + 0.5 * self.aim_mlp_adapter(norm2)
                    for r, norm2 in zip(residual_2_list, norm2_list)
                ]

            x_ffn = [
                torch.index_add(
                    x_attn, dim=0, source=self.ls2(residual_2), index=indices_2, alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope, hw in zip(x_list, rope_list, hw_list):
                h = self.norm1(x)
                if self.t_in_adapter is not None:           
                    h = self.t_in_adapter(h)
                attn_out = self.attn(h, rope=rope, T=T)     
                if self.aim_s_adapter is not None:
                    attn_out = self.aim_s_adapter(attn_out)
                if self.s_adapter is not None:               
                    attn_out = self.s_adapter(attn_out)
                x_attn = x + self.ls1(attn_out)
                if self.adapter_pre_mlp is not None:
                    x_attn = self._apply_adapter(self.adapter_pre_mlp, x_attn, T, hw)
                x_norm2 = self.norm2(x_attn)
                if self.mlp_adapter is not None:             
                    x_norm2 = self.mlp_adapter(x_norm2)
                x_mlp = self.mlp(x_norm2)
                if self.adapter_parallel_mlp is not None:
                    x_mlp = x_mlp + self.adapter_parallel_mlp(x_norm2)
                if self.aim_mlp_adapter is not None:
                    x_mlp = x_mlp + 0.5 * self.aim_mlp_adapter(x_norm2)
                if self.mlp_out_adapter is not None:       
                    x_mlp = self.mlp_out_adapter(x_mlp)
                x_ffn = x_attn + self.ls2(x_mlp)
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None, T=None, hw_or_hw_list=None):
        if isinstance(x_or_x_list, Tensor):
            hw_list = [hw_or_hw_list] if hw_or_hw_list is not None else None
            return self._forward_list(
                [x_or_x_list], rope_list=[rope_or_rope_list], T=T, hw_list=hw_list,
            )[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            return self._forward_list(
                x_or_x_list, rope_list=rope_or_rope_list, T=T, hw_list=hw_or_hw_list,
            )
        else:
            raise AssertionError


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        device: Any | None = None,
        num_frames: int = 16,
        peft_type: str = None,
        peft_rank: int = 0,
        peft_dropout: float = 0.0,
        n_prompt_tokens: int = 0,
        prompt_tokens_type: str = "none",
        prompt_fourier_ratio: float = 0.0,
        bpt_rank: int = -1,
        peft_clip_len: int = 0,
    ):
        super().__init__()

        _peft_clip_len = peft_clip_len if peft_clip_len > 0 else num_frames

        norm_layer_cls = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.grid_size = img_size // patch_size
        self.tubelet_size = 1 # for compatibility with probe
        self.num_patches = self.grid_size ** 2
        self.prompt_tokens_type = prompt_tokens_type
        self.prompt_fourier_ratio = prompt_fourier_ratio
        self.n_fourier_tokens = 0
        self.prompt_tokens_conv = None

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = 4
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, embed_dim, device=device))
        self.n_prefix_tokens = 1 + self.n_storage_tokens
        self.n_prompt_tokens = n_prompt_tokens
        if self.n_prompt_tokens > 0:
            if prompt_tokens_type == "vpt_shallow":
                self.prompt_tokens = nn.Parameter(torch.empty(1, n_prompt_tokens*num_frames, embed_dim, device=device))
                nn.init.normal_(self.prompt_tokens, std=0.02)
                self.n_global_tokens = 1 + self.n_storage_tokens + self.n_prompt_tokens*num_frames
            elif prompt_tokens_type == "vpt_deep":
                vpt_dim = bpt_rank if bpt_rank > 0 else embed_dim
                self.prompt_tokens = nn.ParameterList([nn.Parameter(torch.empty(1, n_prompt_tokens*num_frames, vpt_dim, device=device)) for _ in range(depth)])
                for prompt_token in self.prompt_tokens:
                    nn.init.normal_(prompt_token, std=0.02)
                self.n_global_tokens = 1 + self.n_storage_tokens 
                
                if prompt_fourier_ratio > 0:
                    self.n_fourier_tokens = max(1, int(n_prompt_tokens * prompt_fourier_ratio * num_frames))

                if bpt_rank > 0:
                    self.prompt_tokens_conv =  nn.Sequential(*[nn.Conv2d(in_channels=bpt_rank, out_channels=embed_dim, kernel_size=1, bias=False) for _ in range(depth)])
                    for conv in self.prompt_tokens_conv:
                        w = conv.weight.data
                        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                    self.bpt_length = int(np.sqrt(n_prompt_tokens * num_frames))
            elif prompt_tokens_type == "vpt_shallow_per_frame":
                self.prompt_tokens = nn.Parameter(torch.empty(1, num_frames, n_prompt_tokens, embed_dim, device=device))
                nn.init.normal_(self.prompt_tokens, std=0.02)
                self.n_global_tokens = 1 + self.n_storage_tokens + self.n_prompt_tokens
            elif prompt_tokens_type == "vpt_deep_per_frame":
                self.prompt_tokens = nn.ParameterList([
                    nn.Parameter(torch.empty(1, num_frames, n_prompt_tokens, embed_dim, device=device)) 
                    for _ in range(depth)
                ])
                for prompt_token in self.prompt_tokens:
                    nn.init.normal_(prompt_token, std=0.02)
                self.n_global_tokens = 1 + self.n_storage_tokens
        else:
            self.n_global_tokens = 1 + self.n_storage_tokens
        
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100.0,
            min_period=None,
            max_period=None,
            normalize_coords="separate",
            shift_coords=None,
            jitter_coords=None,
            rescale_coords=None,
            dtype=torch.bfloat16,
            device=device,
        )
        ffn_ratio_sequence = [ffn_ratio] * depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=dpr[i],
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=1e-5,
                mask_k_bias=True,
                device=device,
                peft_type=peft_type,
                peft_rank=peft_rank,
                peft_dropout=peft_dropout,
                num_frames=num_frames,
                backbone_clip_len=num_frames,
                peft_clip_len=_peft_clip_len,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer_cls(embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        # ZeroI2V: learnable per-frame temporal encoding. STDHA needs the (B, T)
        # layout intact, so stochastic depth (which subsets the B*T batch) is off.
        self.zeroi2v_temporal_embedding = None
        if peft_type == "zeroi2v":
            assert drop_path_rate == 0, "ZeroI2V requires drop_path_rate=0 (STDHA needs the temporal structure intact)."
            self.zeroi2v_temporal_embedding = TemporalEmbedding(num_frames, embed_dim)

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)
        for m in self.modules():
            if isinstance(m, LoRA):
                nn.init.kaiming_uniform_(m.lora_a.weight, a=math.sqrt(5))
                nn.init.zeros_(m.lora_b.weight)
            elif isinstance(m, AIMAdapter):
                nn.init.zeros_(m.D_fc2.weight)
                nn.init.zeros_(m.D_fc2.bias)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        if self.prompt_tokens_type == "vpt_shallow":
            prompt_tokens = self.prompt_tokens
            x = torch.cat(
                [
                    cls_token.expand(B, -1, -1),
                    storage_tokens.expand(B, -1, -1),
                    prompt_tokens.expand(B, -1, -1),
                    x,
                ],
                dim=1,
            )
        elif self.prompt_tokens_type == "vpt_shallow_per_frame":
            _, F, N, D = self.prompt_tokens.shape
            prompt_tokens = self.prompt_tokens.expand(B//F, -1, -1, -1).reshape(B, N, D)
            x = torch.cat(
                [
                    cls_token.expand(B, -1, -1),
                    storage_tokens.expand(B, -1, -1),
                    prompt_tokens,
                    x,
                ],
                dim=1,
            )
        else:
            x = torch.cat(
                [
                    cls_token.expand(B, -1, -1),
                    storage_tokens.expand(B, -1, -1),
                    x,
                ],
                dim=1,
            )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor], T: int = None) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        if self.zeroi2v_temporal_embedding is not None:
            x = [self.zeroi2v_temporal_embedding(t_x, T) for t_x in x]
        for blk_idx, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]

            if self.prompt_tokens_type == "vpt_deep":
                x_with_prompts = []
                for t_x in x:
                    B = t_x.shape[0]
                    prompt_tokens = self.prompt_tokens[blk_idx].expand(B, -1, -1)

                    if self.prompt_tokens_conv is not None:
                        vectors = prompt_tokens.transpose(1, 2).contiguous().reshape(B, -1, self.bpt_length, self.bpt_length)
                        vectors = self.prompt_tokens_conv[blk_idx](vectors)
                        prompt_tokens = vectors.reshape(B, self.embed_dim, -1).transpose(1, 2).contiguous()

                    if self.n_fourier_tokens > 0:
                        fourier_prompts = prompt_tokens[:, :self.n_fourier_tokens, :]
                        regular_prompts = prompt_tokens[:, self.n_fourier_tokens:, :]
                        fourier_prompts = torch.fft.fft(torch.fft.fft(fourier_prompts, dim=-1), dim=-2).real
                        prompt_tokens = torch.cat([fourier_prompts, regular_prompts], dim=1)
                        
                    t_x = torch.cat([
                        t_x[:, :self.n_prefix_tokens],
                        prompt_tokens,
                        t_x[:, self.n_prefix_tokens:]
                    ], dim=1)
                    x_with_prompts.append(t_x)
                x = x_with_prompts
            elif self.prompt_tokens_type == "vpt_deep_per_frame":
                x_with_prompts = []
                for t_x in x:
                    B = t_x.shape[0]
                    _, F, N, D = self.prompt_tokens[blk_idx].shape
                    prompt_tokens = self.prompt_tokens[blk_idx].expand(B//F, -1, -1, -1).reshape(B, N, D)
                    t_x = torch.cat([
                        t_x[:, :self.n_prefix_tokens],
                        prompt_tokens,
                        t_x[:, self.n_prefix_tokens:]
                    ], dim=1)
                    x_with_prompts.append(t_x)
                x = x_with_prompts
            elif self.prompt_tokens_type == "reg_add":
                x_reg_add = []
                for t_x in x:
                    B = t_x.shape[0]
                    register_tokens = t_x[:, 1:self.n_prefix_tokens]
                    num_clips = B // self.num_frames
                    register_tokens_per_frame = register_tokens.reshape(num_clips, self.num_frames, self.n_storage_tokens, -1)
                    all_registers_in_clip = register_tokens_per_frame.reshape(num_clips, self.num_frames * self.n_storage_tokens, -1)
                    all_registers = all_registers_in_clip.unsqueeze(1).expand(-1, self.num_frames, -1, -1).reshape(B, self.num_frames * self.n_storage_tokens, -1)
                
                    t_x = torch.cat([
                        t_x[:, :1],
                        all_registers,
                        t_x[:, self.n_prefix_tokens:]
                    ], dim=1)
                    x_reg_add.append(t_x)
                x = x_reg_add
            elif self.prompt_tokens_type == "reg_shift":
                x_reg_shift = []
                for t_x in x:
                    B = t_x.shape[0]
                    register_tokens = t_x[:, 1:self.n_prefix_tokens]  
                    num_clips = B // self.num_frames
                    register_tokens_per_frame = register_tokens.reshape(num_clips, self.num_frames, self.n_storage_tokens, -1)
                    
                    reg_0 = register_tokens_per_frame[:, :, 0, :]
                    reg_1 = register_tokens_per_frame[:, :, 1, :]
                    remaining_regs = register_tokens_per_frame[:, :, 2:, :]
                    
                    reg_0_shifted = torch.zeros_like(reg_0)
                    reg_0_shifted[:, :-1, :] = reg_0[:, 1:, :]
                    
                    reg_1_shifted = torch.zeros_like(reg_1)
                    reg_1_shifted[:, 1:, :] = reg_1[:, :-1, :]
                    
                    shifted_registers = torch.cat([
                            reg_0_shifted.unsqueeze(2),
                            reg_1_shifted.unsqueeze(2),
                            remaining_regs
                        ], dim=2)
                    
                    shifted_registers = shifted_registers.reshape(B, self.n_storage_tokens, -1)
                    
                    t_x = torch.cat([
                            t_x[:, :1],
                            shifted_registers,
                            t_x[:, self.n_prefix_tokens:]
                        ], dim=1)
                    x_reg_shift.append(t_x)
                x = x_reg_shift

            x = blk(x, rope_sincos, T=T, hw_or_hw_list=rope)

            if (self.prompt_tokens_type == "vpt_deep") or (self.prompt_tokens_type == "vpt_deep_per_frame"):
                x_without_prompts = []
                for t_x in x:
                    n_prompt = self.n_prompt_tokens * self.num_frames if self.prompt_tokens_type == "vpt_deep" else self.n_prompt_tokens
                    t_x = torch.cat([
                        t_x[:, :self.n_prefix_tokens],
                        t_x[:, self.n_global_tokens + n_prompt:]
                    ], dim=1)
                    x_without_prompts.append(t_x)
                x = x_without_prompts
            elif self.prompt_tokens_type == "reg_add":
                x_reg_remove = []
                for t_x in x:
                    B = t_x.shape[0]
                    all_updated_registers = t_x[:, 1:1 + self.num_frames * self.n_storage_tokens]  
                    num_clips = B // self.num_frames
                    all_updated_registers = all_updated_registers.reshape(num_clips, self.num_frames, self.num_frames * self.n_storage_tokens, -1)

                    own_registers = []
                    for frame_idx in range(self.num_frames):
                        start_idx = frame_idx * self.n_storage_tokens
                        end_idx = (frame_idx + 1) * self.n_storage_tokens
                        own_registers.append(all_updated_registers[:, frame_idx, start_idx:end_idx, :])
                    
                    own_registers = torch.stack(own_registers, dim=1).reshape(B, self.n_storage_tokens, -1)
                    t_x = torch.cat([
                        t_x[:, :1],  
                        own_registers,  
                        t_x[:, 1 + self.num_frames * self.n_storage_tokens:]  
                    ], dim=1)
                    x_reg_remove.append(t_x)
                x = x_reg_remove
            # elif self.prompt_tokens_type == "tst":
            #     x_tst = []
            #     for t_x in x:
            #         D = t_x.shape[-1]
            #         prompt_tokens = t_x[:, self.n_prefix_tokens:self.n_global_tokens]
            #         prompt_tokens = prompt_tokens.view(B, self.num_frames, -1, D)
            #     pass

        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            x_norm = self.norm(x)
            x_norm_cls_reg = x_norm[:, : self.n_global_tokens]
            x_norm_patch = x_norm[:, self.n_global_tokens :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None, T: int = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks], T=T)[0]
        else:
            return self.forward_features_list(x, masks, T=T)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_global_tokens] for out in outputs]
        outputs = [out[:, self.n_global_tokens :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)  # Reorder to (B*T, C, H, W)
        features = self.forward_features(x, T=T)["x_norm_patchtokens"].reshape(B, T, self.num_patches, self.embed_dim).reshape(B, -1, self.embed_dim)
        return features


def dinov3_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model

def dinov3_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model

def dinov3_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


