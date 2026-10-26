import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRA(nn.Module):
    def __init__(self, dim, num_heads, qkv_weight, rank=128, merge_factor=1.0, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.merge_factor = merge_factor

        self.dora_drop = nn.Dropout(p=dropout)
        self.dora_a = nn.Linear(dim, rank, bias=False)
        self.dora_b = nn.Linear(rank, dim * 3, bias=False)

        # magnitude initialized to row norms of W_qkv; at init dora_b=0 so output = pretrained
        with torch.no_grad():
            self.magnitude = nn.Parameter(qkv_weight.norm(dim=1).clone())

        nn.init.kaiming_uniform_(self.dora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dora_b.weight)

    def forward(self, x, qkv_out, qkv_weight):
        """
        x:          [B, N, C]
        qkv_out:    [B, N, 3*C]  raw output of frozen qkv linear (no bias assumed)
        qkv_weight: [3*C, C]     frozen qkv weight (self.qkv.weight from Attention)
        Returns q, k, v: each [B, num_heads, N, head_dim]
        """
        B, N, C = x.shape

        lora_delta = self.dora_b(self.dora_drop(self.dora_a(x)))  # [B, N, 3*C]

        # row norms of effective weight W + BA
        ba = self.dora_b.weight @ self.dora_a.weight              # [3*C, C]
        weight_norm = (qkv_weight + self.merge_factor * ba).norm(dim=1)  # [3*C]
        scale = self.magnitude / weight_norm                       # [3*C]

        out = scale * (qkv_out + self.merge_factor * lora_delta)  # [B, N, 3*C]
        out = out.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = out.unbind(0)
        return q, k, v
