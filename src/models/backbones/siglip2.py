import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.peft.lora import LoRA
from src.models.peft.adaptformer import AdaptFormer
from src.models.peft.adapter import Adapter


class _MLP(nn.Module):
    """Drop-in for Siglip2MLP; weight names match (fc1 / fc2)."""

    def __init__(self, embed_dim: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(intermediate_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SigLIP2Attention(nn.Module):
    """Self-attention with optional LoRA delta on q/k/v."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        peft_type: str = None,
        peft_rank: int = 0,
        peft_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.lora = None
        if peft_type == "lora":
            self.lora = LoRA(embed_dim, num_heads, rank=peft_rank, dropout=peft_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.lora is not None:
            q, k, v = self.lora(x, q, k, v)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)
        return self.out_proj(out)


class SigLIP2EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float = 1e-6,
        attn_dropout: float = 0.0,
        peft_type: str = None,
        peft_rank: int = 0,
        peft_dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.self_attn = SigLIP2Attention(
            embed_dim, num_heads,
            dropout=attn_dropout,
            peft_type=peft_type,
            peft_rank=peft_rank,
            peft_dropout=peft_dropout,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = _MLP(embed_dim, intermediate_size)

        self.adapter_pre_attn = None
        self.adapter_pre_mlp = None
        self.adapter_parallel_mlp = None
        if peft_type == "st-adapter":
            self.adapter_pre_attn = Adapter(in_channels=embed_dim, adapter_channels=peft_rank, adapter_type="st-adapter")
            self.adapter_pre_mlp = Adapter(in_channels=embed_dim, adapter_channels=peft_rank, adapter_type="st-adapter")
        elif peft_type == "frame2freq_ms":
            self.adapter_pre_attn = Adapter(in_channels=embed_dim, adapter_channels=peft_rank, adapter_type="frame2freq_ms")
            self.adapter_pre_mlp = Adapter(in_channels=embed_dim, adapter_channels=peft_rank, adapter_type="frame2freq_ms")
        elif peft_type == "adapter":
            self.adapter_pre_attn = Adapter(in_channels=embed_dim, adapter_channels=peft_rank, adapter_type="adapter", act_fn=nn.GELU)
            self.adapter_pre_mlp = Adapter(in_channels=embed_dim, adapter_channels=peft_rank, adapter_type="adapter", act_fn=nn.GELU)
        elif peft_type == "adaptformer_adapter":
            self.adapter_parallel_mlp = AdaptFormer(in_dim=embed_dim, bottleneck=peft_rank, dropout=peft_dropout)

    def forward(
        self,
        x: torch.Tensor,
        T: int = None,
        H_patches: int = None,
        W_patches: int = None,
    ) -> torch.Tensor:
        # Input layout: (B*T, H_p*W_p, C) — frames stacked along batch dim.
        if self.adapter_pre_attn is not None and T is not None:
            x = self.adapter_pre_attn(x, T, H_patches, W_patches, frames_as_batch=True)

        x = x + self.self_attn(self.layer_norm1(x))

        if self.adapter_pre_mlp is not None and T is not None:
            x = self.adapter_pre_mlp(x, T, H_patches, W_patches, frames_as_batch=True)

        x_ln = self.layer_norm2(x)
        x_mlp = self.mlp(x_ln)
        if self.adapter_parallel_mlp is not None:
            x_mlp = x_mlp + self.adapter_parallel_mlp(x_ln)
        x = x + x_mlp

        return x


class SigLIP2(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        num_frames: int = 16,
        drop_path_rate: float = 0.0,  # accepted for API compat; SigLIP2 has no stoch-depth
        peft_type: str = None,
        peft_rank: int = 0,
        peft_dropout: float = 0.0,
        cfg: dict = None,
    ):
        super().__init__()
        if cfg is None:
            raise ValueError("SigLIP2 requires a 'cfg' dict with architecture parameters.")

        patch_size       = cfg['patch_size']
        embed_dim        = cfg['hidden_size']
        num_heads        = cfg['num_attention_heads']
        intermediate_size = cfg['intermediate_size']
        depth            = cfg['num_hidden_layers']
        norm_eps         = cfg['layer_norm_eps']
        attn_dropout     = cfg['attention_dropout']
        pretrained_size  = cfg['image_size']

        assert img_size == pretrained_size, "img_size must be equal to pretrained_size"

        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = 1  # image-level model; required by Probe
        self.num_heads = num_heads
        self.embed_dim = self.num_features = embed_dim
        self.grid_size = img_size // patch_size       # actual input grid (for Probe)
        self.num_patches = self.grid_size ** 2
        self._pos_embed_size = pretrained_size // patch_size  # pretrained embedding grid

        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Embedding(self._pos_embed_size ** 2, embed_dim)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=norm_eps)

        self.layers = nn.ModuleList([
            SigLIP2EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
                attn_dropout=attn_dropout,
                peft_type=peft_type,
                peft_rank=peft_rank,
                peft_dropout=peft_dropout,
            )
            for _ in range(depth)
        ])

    def get_num_layers(self) -> int:
        return len(self.layers)

    def _embed(self, x: torch.Tensor) -> tuple:
        """(B, C, H, W) -> (B, H_p*W_p, D), H_p, W_p"""
        target_dtype = self.patch_embedding.weight.dtype
        x = self.patch_embedding(x.to(dtype=target_dtype))  # (B, D, H_p, W_p)
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)                    # (B, H_p*W_p, D)

        pos_size = self._pos_embed_size
        pos_w = self.position_embedding.weight               # (pos_size**2, D)
        if H_p == pos_size and W_p == pos_size:
            x = x + pos_w.unsqueeze(0)
        else:
            D = pos_w.shape[-1]
            pe = pos_w.reshape(pos_size, pos_size, D).permute(2, 0, 1).unsqueeze(0)
            pe = pe.to(torch.float32)
            pe = F.interpolate(pe, size=(H_p, W_p), mode="bilinear", align_corners=False, antialias=True)
            pe = pe.squeeze(0).permute(1, 2, 0).reshape(H_p * W_p, D).to(target_dtype)
            x = x + pe.unsqueeze(0)

        return x, H_p, W_p

    def forward(self, x: torch.Tensor, return_all_layers: bool = False):
        """
        Args:
            x: (B, C, T, H, W) video tensor
            return_all_layers: if True, return list of per-layer outputs
                (B, T*H_p*W_p, embed_dim) with post_layernorm applied.

        Returns:
            Tensor (B, T*H_p*W_p, embed_dim)  or  list of such tensors.
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        hidden, H_p, W_p = self._embed(x)   # (B*T, H_p*W_p, D)

        all_outputs = [] if return_all_layers else None

        for layer in self.layers:
            hidden = layer(hidden, T=T, H_patches=H_p, W_patches=W_p)
            if return_all_layers:
                normed = self.post_layernorm(hidden)
                all_outputs.append(normed.view(B, T * H_p * W_p, self.embed_dim))

        if return_all_layers:
            return all_outputs

        hidden = self.post_layernorm(hidden)
        return hidden.view(B, T * H_p * W_p, self.embed_dim)


def siglip2_large(img_size=256, **kwargs):
    model = SigLIP2(img_size=img_size, **kwargs)
    return model
