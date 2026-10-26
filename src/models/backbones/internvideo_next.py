import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

import torch.utils.checkpoint as checkpoint
from functools import partial
import einops

from src.utils.internvideo_next_utils import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed, LayerScale, RMSNorm, PatchEmbed, Mlp
from src.models.peft.lora import LoRA, ST_LoRA
from src.models.peft.dora import DoRA
from src.models.peft.adaptformer import AdaptFormer, ST_AdaptFormer
from src.models.peft.adapter import Adapter
from src.models.peft.ssf import init_ssf_scale_shift, ssf_ada


class SSFMlp(Mlp):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 bias=True, drop=0.):
        super().__init__(in_features, hidden_features=hidden_features, out_features=out_features,
                         act_layer=act_layer, bias=bias, drop=drop)
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.ssf_scale_hidden, self.ssf_shift_hidden = init_ssf_scale_shift(hidden_features)
        self.ssf_scale_out, self.ssf_shift_out = init_ssf_scale_shift(out_features)

    def forward(self, x):
        x = ssf_ada(self.fc1(x), self.ssf_scale_hidden, self.ssf_shift_hidden)
        x = self.act(x)
        x = self.drop1(x)
        x = ssf_ada(self.fc2(x), self.ssf_scale_out, self.ssf_shift_out)
        x = self.drop2(x)
        return x


class SSFPatchEmbed(PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=8, tubelet_size=1, norm_layer=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                         embed_dim=embed_dim, num_frames=num_frames, tubelet_size=tubelet_size,
                         norm_layer=norm_layer)
        self.ssf_scale_proj, self.ssf_shift_proj = init_ssf_scale_shift(embed_dim)
        self.ssf_scale_norm, self.ssf_shift_norm = init_ssf_scale_shift(embed_dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = einops.rearrange(x, "b (t1 t2) (ht hp) (wt wp) c -> b (t1 ht wt) (t2 hp wp c)",
                             t2=self.tubelet_size, hp=self.patch_size[0], wp=self.patch_size[1])
        x = self.norm_before(x)
        x = einops.rearrange(x, "b (t1 ht wt) (t2 hp wp c) -> b (t1 t2) (ht hp) (wt wp) c",
                             t1=T // self.tubelet_size, ht=H // self.patch_size[0],
                             t2=self.tubelet_size, hp=self.patch_size[0], wp=self.patch_size[1])
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)
        x = ssf_ada(x, self.ssf_scale_proj, self.ssf_shift_proj)
        x = ssf_ada(self.norm(x), self.ssf_scale_norm, self.ssf_shift_norm)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.LayerNorm, qk_normalization=False,
                 peft_type=None, peft_rank=0, peft_dropout=0.1,
                 cls_token_num=0, backbone_clip_len=1, peft_clip_len=1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

        self.lora = None
        if peft_type == "lora":
            self.lora = LoRA(dim, num_heads, rank=peft_rank, dropout=peft_dropout)
        self.st_lora = None
        if peft_type in ("st_lora", "st_lora_adaptformer_adapter"):
            self.st_lora = ST_LoRA(dim, num_heads, rank=peft_rank,
                                   cls_token_num=cls_token_num,
                                   backbone_clip_len=backbone_clip_len,
                                   peft_clip_len=peft_clip_len)

        self.dora = None
        if peft_type == "dora":
            self.dora = DoRA(dim, num_heads, self.qkv.weight, rank=peft_rank, dropout=peft_dropout)

        self.ssf_scale_qkv = self.ssf_shift_qkv = None
        self.ssf_scale_proj = self.ssf_shift_proj = None
        if peft_type == "ssf":
            self.ssf_scale_qkv, self.ssf_shift_qkv = init_ssf_scale_shift(dim * 3)
            self.ssf_scale_proj, self.ssf_shift_proj = init_ssf_scale_shift(dim)

    def forward(self, x, T=None, H_patches=None, W_patches=None):
        B, N, C = x.shape
        qkv_out = self.qkv(x)  # [B, N, 3*C]
        if self.ssf_scale_qkv is not None:
            qkv_out = ssf_ada(qkv_out, self.ssf_scale_qkv, self.ssf_shift_qkv)

        if self.dora is not None:
            q, k, v = self.dora(x, qkv_out, self.qkv.weight)
        else:
            qkv = qkv_out.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if self.lora is not None:
                q, k, v = self.lora(x, q, k, v)
            if self.st_lora is not None:
                q, k, v = self.st_lora(x, q, k, v, T, H_patches, W_patches)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        with torch.backends.cuda.sdp_kernel():
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_prob if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.ssf_scale_proj is not None:
            x = ssf_ada(x, self.ssf_scale_proj, self.ssf_shift_proj)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, with_cp=False,
            qk_normalization=False, layerscale_no_force_fp32=False,
            peft_type=None, peft_rank=0, peft_dropout=0.1, cls_token_num=0,
            backbone_clip_len=1, peft_clip_len=1):
        super().__init__()
        self.cls_token_num = cls_token_num

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              norm_layer=norm_layer, qk_normalization=qk_normalization,
                              peft_type=peft_type, peft_rank=peft_rank, peft_dropout=peft_dropout,
                              cls_token_num=cls_token_num,
                              backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_cls = SSFMlp if peft_type == "ssf" else Mlp
        self.mlp = mlp_cls(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.with_cp = with_cp

        self.adapter_pre_attn = None
        self.adapter_pre_mlp = None
        self.adapter_parallel_mlp = None
        self.st_adapter_parallel_mlp = None
        if peft_type == "st-adapter":
            self.adapter_pre_attn = Adapter(in_channels=dim, adapter_channels=peft_rank,
                                               backbone_clip_len=backbone_clip_len, peft_clip_len=peft_clip_len)
            self.adapter_pre_mlp = Adapter(in_channels=dim, adapter_channels=peft_rank,
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
        elif peft_type in ("st_adaptformer_adapter", "st_lora_adaptformer_adapter"):
            self.st_adapter_parallel_mlp = ST_AdaptFormer(in_dim=dim, bottleneck=peft_rank,
                                                           cls_token_num=cls_token_num,
                                                           backbone_clip_len=backbone_clip_len,
                                                           peft_clip_len=peft_clip_len)

        self.ssf_scale_1 = self.ssf_shift_1 = None
        self.ssf_scale_2 = self.ssf_shift_2 = None
        if peft_type == "ssf":
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)

    def forward(self, x, T=None, H_patches=None, W_patches=None):
        def _inner_forward(x):
            if self.adapter_pre_attn is not None:
                x_cls, x_p = x[:, :self.cls_token_num], x[:, self.cls_token_num:]
                x_p = self.adapter_pre_attn(x_p, T, H_patches, W_patches)
                x = torch.cat([x_cls, x_p], dim=1)

            x_norm1 = self.norm1(x)
            if self.ssf_scale_1 is not None:
                x_norm1 = ssf_ada(x_norm1, self.ssf_scale_1, self.ssf_shift_1)
            x = x + self.drop_path1(self.ls1(self.attn(x_norm1, T=T, H_patches=H_patches, W_patches=W_patches)))

            if self.adapter_pre_mlp is not None:
                x_cls, x_p = x[:, :self.cls_token_num], x[:, self.cls_token_num:]
                x_p = self.adapter_pre_mlp(x_p, T, H_patches, W_patches)
                x = torch.cat([x_cls, x_p], dim=1)

            x_norm2 = self.norm2(x)
            if self.ssf_scale_2 is not None:
                x_norm2 = ssf_ada(x_norm2, self.ssf_scale_2, self.ssf_shift_2)
            x_mlp = self.mlp(x_norm2)
            if self.adapter_parallel_mlp is not None:
                x_mlp = x_mlp + self.adapter_parallel_mlp(x_norm2)
            if self.st_adapter_parallel_mlp is not None:
                x_mlp = x_mlp + self.st_adapter_parallel_mlp(x_norm2, T, H_patches, W_patches)
            x = x + self.drop_path2(self.ls2(x_mlp))
            return x

        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class InternVideo2(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.25,
            embed_dim: int = 1408,
            num_heads: int = 16,
            mlp_ratio: float = 4.3637,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 40,
            layerscale_no_force_fp32: bool = False,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            num_frames: int = 16,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,
            cls_token_num: int = 4,
            peft_type: str = None,
            peft_rank: int = 0,
            peft_dropout: float = 0.1,
            n_prompt_tokens: int = 0,
            prompt_fourier_ratio: float = 0.5,
            bpt_rank: int = 100,
            backbone_clip_len: int = 0,
            peft_clip_len: int = 0,
        ):
        super().__init__()
        # num_frames: full input temporal size (= dataloader clip_len, superseding).
        # backbone_clip_len: temporal context per backbone evaluation; 0 → same as num_frames.
        _backbone_clip_len = backbone_clip_len if backbone_clip_len > 0 else num_frames
        self.backbone_clip_len = _backbone_clip_len
        self.decouple = (self.backbone_clip_len != num_frames)

        # peft_clip_len: temporal context for PEFT modules; 0 → same as backbone context.
        _peft_clip_len = peft_clip_len if peft_clip_len > 0 else _backbone_clip_len

        self.cls_token_num = cls_token_num
        self.num_heads = num_heads
        self.tubelet_size = tubelet_size
        self.grid_size = img_size // patch_size
        self.embed_dim = embed_dim

        norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        patch_embed_cls = SSFPatchEmbed if peft_type == "ssf" else PatchEmbed
        self.patch_embed = patch_embed_cls(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=_backbone_clip_len, tubelet_size=tubelet_size, norm_layer=partial(RMSNorm, eps=1e-6)
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, cls_token_num, embed_dim))

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            print("Use seperable position embedding")
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, grid_size[1] * grid_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, grid_size[0], embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            print("Use joint position embedding")
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + cls_token_num, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(checkpoint_num):
                with_cp_list[idx] = True

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  peft_type=peft_type, peft_rank=peft_rank, peft_dropout=peft_dropout,
                  cls_token_num=cls_token_num,
                  backbone_clip_len=_backbone_clip_len, peft_clip_len=_peft_clip_len)
            for i in range(depth)])

        self.n_prompt_tokens = n_prompt_tokens
        self.n_fourier_tokens = 0
        self.prompt_tokens = None
        self.prompt_tokens_conv = None
        self.bpt_length = 0
        if peft_type in ("vpt_deep", "vfpt_deep") and n_prompt_tokens > 0:
            self.prompt_tokens = nn.ParameterList([
                nn.Parameter(torch.empty(1, n_prompt_tokens, embed_dim))
                for _ in range(depth)
            ])
            for p in self.prompt_tokens:
                nn.init.normal_(p, std=0.02)
            if peft_type == "vfpt_deep":
                self.n_fourier_tokens = max(1, int(n_prompt_tokens * prompt_fourier_ratio))
        elif peft_type == "bpt_deep" and n_prompt_tokens > 0 and bpt_rank > 0:
            self.bpt_length = int(math.sqrt(n_prompt_tokens))
            assert self.bpt_length ** 2 == n_prompt_tokens, "n_prompt_tokens must be a perfect square for bpt_deep"
            self.prompt_tokens = nn.ParameterList([
                nn.Parameter(torch.empty(1, n_prompt_tokens, bpt_rank))
                for _ in range(depth)
            ])
            for p in self.prompt_tokens:
                nn.init.normal_(p, std=0.02)
            self.prompt_tokens_conv = nn.Sequential(*[
                nn.Conv2d(bpt_rank, embed_dim, kernel_size=1, bias=False)
                for _ in range(depth)
            ])
            for conv in self.prompt_tokens_conv:
                nn.init.xavier_uniform_(conv.weight.view(conv.weight.shape[0], -1))

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self._init_all_weights()
        self.fix_init_weight()

    def init_pos_embed(self):
        print("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            pos_embed_spatial = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1],
                self.patch_embed.grid_size[1],
            )
            self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))
            pos_embed_temporal = get_1d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1],
                self.patch_embed.grid_size[0],
            )
            self.pos_embed_temporal.data.copy_(torch.from_numpy(pos_embed_temporal).float().unsqueeze(0))
        else:
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.patch_embed.grid_size[1],
                self.patch_embed.grid_size[0],
                cls_token=True,
                cls_token_num=self.cls_token_num
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_all_weights(self):
        peft_types = (LoRA, ST_LoRA, DoRA, AdaptFormer, ST_AdaptFormer, Adapter)
        skip_modules = set()
        for m in self.modules():
            if isinstance(m, peft_types):
                for child in m.modules():
                    skip_modules.add(id(child))

        def _init(m):
            if id(m) in skip_modules:
                return
            self._init_weights(m)

        self.apply(_init)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = {'pos_embed', 'pos_embed_spatial', 'pos_embed_temporal', 'pos_embed_cls', 'cls_token'}
        # SSF scale/shift are small affine params — weight decay would hurt them
        for name, _ in self.named_parameters():
            if 'ssf_scale' in name or 'ssf_shift' in name:
                no_decay.add(name)
        return no_decay

    def forward(self, x, return_all_layers=False):
        B_orig = x.shape[0]
        T_input = x.shape[2]

        if self.decouple and self.backbone_clip_len < T_input:
            K = T_input // self.backbone_clip_len
            B, C, T, H, W = x.shape
            x = x.view(B, C, K, self.backbone_clip_len, H, W)
            x = x.permute(0, 2, 1, 3, 4, 5).reshape(B * K, C, self.backbone_clip_len, H, W)

        x = self.patch_embed(x.type(self.dtype))
        B, T_patches, L, C = x.shape
        H_patches = W_patches = int(math.sqrt(L))
        x = x.view([B, T_patches * L, C])

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.grid_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.grid_size[1] * self.grid_size[2],
                dim=1,
            )
            pos_embed = torch.cat(
                [self.pos_embed_cls.expand(pos_embed.shape[0], -1, -1), pos_embed], 1
            )
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed

        all_layer_outputs = [] if return_all_layers else None

        for blk_idx, blk in enumerate(self.blocks):
            if self.prompt_tokens is not None:
                prompt = self.prompt_tokens[blk_idx].expand(B, -1, -1)
                if self.prompt_tokens_conv is not None:
                    vectors = prompt.transpose(1, 2).contiguous().reshape(B, -1, self.bpt_length, self.bpt_length)
                    vectors = self.prompt_tokens_conv[blk_idx](vectors)
                    prompt = vectors.reshape(B, self.embed_dim, -1).transpose(1, 2).contiguous()
                if self.n_fourier_tokens > 0:
                    fourier_part = torch.fft.fft(torch.fft.fft(prompt[:, :self.n_fourier_tokens], dim=-1), dim=-2).real
                    prompt = torch.cat([fourier_part, prompt[:, self.n_fourier_tokens:]], dim=1)
                x = torch.cat([x[:, :self.cls_token_num], prompt, x[:, self.cls_token_num:]], dim=1)
            x = blk(x, T=T_patches, H_patches=H_patches, W_patches=W_patches)
            if self.prompt_tokens is not None:
                x = torch.cat([x[:, :self.cls_token_num], x[:, self.cls_token_num + self.n_prompt_tokens:]], dim=1)

            if return_all_layers:
                all_layer_outputs.append(x[:, self.cls_token_num:, :])

        if return_all_layers:
            if self.decouple and self.backbone_clip_len < T_input:
                all_layer_outputs = [o.reshape(B_orig, -1, o.shape[-1]) for o in all_layer_outputs]
            return all_layer_outputs

        patches = x[:, self.cls_token_num:, :]
        if self.decouple and self.backbone_clip_len < T_input:
            patches = patches.reshape(B_orig, -1, patches.shape[-1])
        return patches


def internvideo_next_base(pretrained=False, img_size=224, **kwargs):
    model = InternVideo2(
        img_size=img_size, patch_size=14, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4,
        **kwargs
    )
    return model


def internvideo_next_large(pretrained=False, img_size=224, **kwargs):
    model = InternVideo2(
        img_size=img_size, patch_size=14, embed_dim=1024,
        depth=24, num_heads=16, mlp_ratio=4,
        **kwargs
    )
    return model
