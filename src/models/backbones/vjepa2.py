# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.vjepa2_utils import Attention, DropPath, MLP, SwiGLUFFN, PatchEmbed, PatchEmbed3D, get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, trunc_normal_, apply_masks, rotate_queries_or_keys
from src.models.peft.adapter import Adapter
from src.models.peft.lora import LoRA
from src.models.peft.adaptformer import AdaptFormer

class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        peft_type=None,
        peft_rank=0,
        peft_dropout=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

        self.lora = None
        if peft_type == "lora":
            self.lora = LoRA(dim, num_heads, rank=peft_rank, dropout=peft_dropout)

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if self.lora is not None:
            q, k, v = self.lora(x, q, k, v)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        peft_type=None,
        peft_rank=0,
        peft_dropout=0.1,
        backbone_clip_len=1,
        peft_clip_len=1,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
                peft_type=peft_type,
                peft_rank=peft_rank,
                peft_dropout=peft_dropout,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adapter_pre_attn = None
        self.adapter_pre_mlp = None
        self.adapter_parallel_mlp = None
        if peft_type == "st-adapter":
            self.adapter_pre_attn = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="st-adapter",
                                               backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
            self.adapter_pre_mlp = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="st-adapter",
                                              backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
        elif peft_type == "frame2freq_ms":
            self.adapter_pre_attn = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="frame2freq_ms",
                                               backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
            self.adapter_pre_mlp = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="frame2freq_ms",
                                               backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
        elif peft_type == "adapter":
            self.adapter_pre_attn = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="adapter", act_fn=nn.GELU)
            self.adapter_pre_mlp = Adapter(in_channels=dim, adapter_channels=peft_rank, adapter_type="adapter", act_fn=nn.GELU)
        elif peft_type == "adaptformer_adapter":
            self.adapter_parallel_mlp = AdaptFormer(in_dim=dim, bottleneck=peft_rank, dropout=peft_dropout)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, T, H_patches, W_patches)

        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)

        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, T, H_patches, W_patches)

        x_norm2 = self.norm2(x)
        x_mlp = self.mlp(x_norm2)

        if self.adapter_parallel_mlp is not None:
            x_mlp = x_mlp + self.adapter_parallel_mlp(x_norm2)

        x = x + self.drop_path(x_mlp)
        return x


class VJEPA2(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        use_rope=False,
        handle_nonsquare_inputs=True,
        use_mod_embed=False,
        num_out_norms=1,
        peft_type=None,
        peft_rank=0,
        peft_dropout=0.1,
        backbone_clip_len=0,
        peft_clip_len=0,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.handle_nonsquare_inputs = handle_nonsquare_inputs

        if type(img_size) is int:
            img_size = (img_size, img_size)

        self.grid_size = img_size[0] // patch_size
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        # backbone_clip_len: temporal context (in raw frames) per backbone evaluation; 0 -> same as num_frames.
        _backbone_clip_len = backbone_clip_len if backbone_clip_len > 0 else num_frames
        self.backbone_clip_len = _backbone_clip_len
        self.decouple = (self.backbone_clip_len != num_frames)

        # peft_clip_len: temporal context for PEFT modules; 0 -> same as backbone context.
        _peft_clip_len = peft_clip_len if peft_clip_len > 0 else _backbone_clip_len

        _backbone_clip_len_tokens = _backbone_clip_len // tubelet_size
        _peft_clip_len_tokens = _peft_clip_len // tubelet_size

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        _patch_embed_num_frames = _backbone_clip_len if use_rope else num_frames
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size, tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim
            )
            self.num_patches = (_patch_embed_num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # vjepa2.1: modality embeddings and separate image patch embed
        self.use_mod_embed = use_mod_embed
        if use_mod_embed:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # separate patch embed for images (tubelet_size=1)
            self.patch_embed_img = PatchEmbed3D(
                patch_size=patch_size, tubelet_size=1, in_chans=in_chans, embed_dim=embed_dim
            )

        # Position embedding
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        # Attention Blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    peft_type=peft_type,
                    peft_rank=peft_rank,
                    peft_dropout=peft_dropout,
                    backbone_clip_len=_backbone_clip_len_tokens,
                    peft_clip_len=_peft_clip_len_tokens,
                )
                for i in range(depth)
            ]
        )

        # vjepa2.1 uses multiple output norms (one per out_layer); vjepa2 uses a single norm
        if num_out_norms > 1:
            self.norm = None
            self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in range(num_out_norms)])
        else:
            self.norm = norm_layer(embed_dim)
            self.norms_block = None

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self._init_all_weights()
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; currently assumes square input
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_all_weights(self):
        peft_types = (AdaptFormer, Adapter, LoRA)
        skip_modules = set()
        for m in self.modules():
            if isinstance(m, peft_types):
                for child in m.modules():
                    skip_modules.add(id(child))

        def _init(m):
            if id(m) in skip_modules:
                return
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def _get_norm(self, idx=None):
        """Return the appropriate norm: per-layer norm for vjepa2.1, single norm for vjepa2."""
        if self.norms_block is not None:
            if idx is not None:
                return self.norms_block[idx]
            return self.norms_block[-1]
        return self.norm

    def forward(self, x, masks=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        B_orig = x.shape[0]
        is_image = (x.ndim == 4)
        T_input = 1 if is_image else x.shape[2]
        decouple_now = self.decouple and not is_image and self.backbone_clip_len < T_input

        # decoupling for temporal graphs
        if decouple_now:
            assert masks is None, "temporal-context decoupling is not supported together with masking"
            K = T_input // self.backbone_clip_len
            B, C, T, H, W = x.shape
            x = x.view(B, C, K, self.backbone_clip_len, H, W)
            x = x.permute(0, 2, 1, 3, 4, 5).reshape(B * K, C, self.backbone_clip_len, H, W)

        # Tokenize input
        # Image
        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        # Video
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            T = T // self.tubelet_size
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if self.use_mod_embed and is_image:
            # Images: use separate patch embed (tubelet_size=1) + image modality embedding
            x_5d = x.unsqueeze(2)  # (B, C, 1, H, W)
            x = self.patch_embed_img(x_5d)
            x = x + self.img_mod_embed
        elif self.use_mod_embed:
            # Videos: standard patch embed + video modality embedding
            x = self.patch_embed(x)
            x = x + self.video_mod_embed
        elif not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
            x = self.patch_embed(x)
            x += pos_embed
        else:
            x = self.patch_embed(x)

        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        # Fwd prop
        outs = []
        out_layer_idx = 0
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, masks, None, T=T, H_patches=H_patches, W_patches=W_patches, use_reentrant=False
                )
            else:
                x = blk(x, mask=masks, attn_mask=None, T=T, H_patches=H_patches, W_patches=W_patches)
            if self.out_layers is not None and i in self.out_layers:
                norm = self._get_norm(out_layer_idx if self.norms_block is not None else None)
                outs.append(norm(x))
                out_layer_idx += 1

        if self.out_layers is not None:
            if decouple_now:
                outs = [o.reshape(B_orig, -1, o.shape[-1]) for o in outs]
            return outs

        x = self._get_norm()(x)
        if decouple_now:
            x = x.reshape(B_orig, -1, x.shape[-1])
        return x

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already correct size, just return
            _, _, T, H, W = x.shape
            if H == self.img_height and W == self.img_width and T == self.num_frames:
                return pos_embed

            # Just chop off last N tokens of positional embedding
            elif H == self.img_height and W == self.img_width and T < self.num_frames:
                new_N = int((T // self.tubelet_size) * (H // self.patch_size) * (W // self.patch_size))
                return pos_embed[:, :new_N, :]

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = self.img_height // self.patch_size
            N_w = self.img_width // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T / N_t, H / N_h, W / N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already correct size, just return
            _, _, H, W = x.shape
            if H == self.img_height and W == self.img_width:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


def vjepa2_large(patch_size=16, **kwargs):
    model = VJEPA2(
        patch_size=patch_size,
        tubelet_size=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vjepa2_1_large(patch_size=16, **kwargs):
    model = VJEPA2(
        patch_size=patch_size,
        tubelet_size=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mod_embed=True,
        num_out_norms=4,
        **kwargs
    )
    return model










