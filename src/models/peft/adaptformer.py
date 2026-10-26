import torch
import torch.nn as nn
import math


class ST_AdaptFormer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        bottleneck: int = 64,
        act_layer: type = nn.ReLU,
        dropout: float = 0.0,
        init_scale: float = 1e-3,
        cls_token_num: int = 0,
        backbone_clip_len: int = 1,
        peft_clip_len: int = 1,
    ):
        super().__init__()
        self.cls_token_num = cls_token_num
        self.peft_clip_len = peft_clip_len
        self.decouple = (peft_clip_len != backbone_clip_len)

        self.down_proj = nn.Linear(in_dim, bottleneck)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)
        self.temporal_conv = nn.Conv3d(
            bottleneck, bottleneck, kernel_size=(3, 1, 1),
            stride=1, padding=(1, 0, 0), groups=bottleneck
        )
        self.up_proj = nn.Linear(bottleneck, in_dim)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.constant_(self.temporal_conv.weight, 0.)
        nn.init.constant_(self.temporal_conv.bias, 0.)

    def forward(self, x: torch.Tensor, T: int, H_patches: int, W_patches: int) -> torch.Tensor:
        B = x.shape[0]
        out = self.down_proj(x)

        out_cls = out[:, :self.cls_token_num]   # [B, cls, bottleneck]
        out_p   = out[:, self.cls_token_num:]   # [B, T*H*W, bottleneck]
        b = self.temporal_conv.in_channels

        if self.decouple and self.peft_clip_len > T:
            K_merge = self.peft_clip_len // T
            BK = out_p.shape[0]
            out_p = out_p.view(BK // K_merge, K_merge * T, H_patches, W_patches, b)
            out_p = out_p.permute(0, 4, 1, 2, 3).contiguous()
            out_p = self.temporal_conv(out_p)
            out_p = out_p.permute(0, 2, 3, 4, 1).contiguous().view(BK, T * H_patches * W_patches, b)
        elif self.decouple and self.peft_clip_len < T:
            K_split = T // self.peft_clip_len
            B_orig = out_p.shape[0]
            out_p = out_p.view(B_orig * K_split, self.peft_clip_len, H_patches, W_patches, b)
            out_p = out_p.permute(0, 4, 1, 2, 3).contiguous()
            out_p = self.temporal_conv(out_p)
            out_p = out_p.permute(0, 2, 3, 4, 1).contiguous().view(B_orig, T * H_patches * W_patches, b)
        else:
            out_p = out_p.view(B, T, H_patches, W_patches, b).permute(0, 4, 1, 2, 3).contiguous()
            out_p = self.temporal_conv(out_p)
            out_p = out_p.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H_patches * W_patches, b)

        out = torch.cat([out_cls, out_p], dim=1)
        out = self.act(out)
        out = self.dropout(out)
        out = self.up_proj(out)
        return out * self.scale


class AdaptFormer(nn.Module):
    """
    AdaptFormer

    Args:
        in_dim:       Input / output feature dimension (embed_dim).
        bottleneck:   Hidden dimension of the bottleneck (default: 64).
        act_layer:    Activation function class (default: nn.ReLU).
        dropout:      Dropout probability inside the adapter (default: 0.0).
        init_scale:   Initial value for the learnable scaling factor
                      (default: 1e-3).  A small value ensures the adapter
                      is near-identity at the start of training.
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck: int = 64,
        act_layer: type = nn.ReLU,
        dropout: float = 0.0,
        init_scale: float = 1e-3,
    ):
        super().__init__()

        self.down_proj = nn.Linear(in_dim, bottleneck)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck, in_dim)

        # learnable scaling factor
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5)) 
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, C)  
        Returns:
            out: Tensor of shape (B, N, C)
        """
        out = self.down_proj(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.up_proj(out)
        return out * self.scale