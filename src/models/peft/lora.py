import math
import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, dim, num_heads, rank=128, merge_factor=1.0, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.merge_factor = merge_factor

        self.lora_drop = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, dim * 3, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x, q, k, v):
        """
        x: [B, N, C]
        q, k, v: [B, num_heads, N, head_dim]
        Returns updated q, k, v with LoRA deltas added.
        """
        B, N, C = x.shape
        qkv_delta = self.lora_b(self.lora_drop(self.lora_a(x)))  # [B, N, 3*C]
        qkv_delta = (
            qkv_delta.reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)

        q = q + self.merge_factor * q_delta
        k = k + self.merge_factor * k_delta
        v = v + self.merge_factor * v_delta
        return q, k, v


class ST_LoRA(nn.Module):
    def __init__(self, dim, num_heads, rank=128, merge_factor=1.0, dropout=0.1,
                 cls_token_num=0, backbone_clip_len=1, peft_clip_len=1):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.merge_factor = merge_factor
        self.cls_token_num = cls_token_num
        self.peft_clip_len = peft_clip_len
        self.decouple = (peft_clip_len != backbone_clip_len)

        self.lora_drop = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, dim * 3, bias=False)
        self.temporal_conv = nn.Conv3d(
            rank, rank, kernel_size=(3, 1, 1),
            stride=1, padding=(1, 0, 0), groups=rank
        )

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        nn.init.constant_(self.temporal_conv.weight, 0.)
        nn.init.constant_(self.temporal_conv.bias, 0.)

    def forward(self, x, q, k, v, T, H_patches, W_patches):
        """
        x: [B, N, C]  (N includes cls tokens)
        q, k, v: [B, num_heads, N, head_dim]
        Returns updated q, k, v with ST-LoRA deltas added.
        """
        B, N, C = x.shape
        h = self.lora_drop(self.lora_a(x))  # [B, N, rank]

        h_cls = h[:, :self.cls_token_num]   # [B, cls, rank]
        h_p   = h[:, self.cls_token_num:]   # [B, T*H*W, rank]
        r = self.rank

        if self.decouple and self.peft_clip_len > T:
            K_merge = self.peft_clip_len // T
            BK = h_p.shape[0]
            h_p = h_p.view(BK // K_merge, K_merge * T, H_patches, W_patches, r)
            h_p = h_p.permute(0, 4, 1, 2, 3).contiguous()
            h_p = self.temporal_conv(h_p)
            h_p = h_p.permute(0, 2, 3, 4, 1).contiguous().view(BK, T * H_patches * W_patches, r)
        elif self.decouple and self.peft_clip_len < T:
            K_split = T // self.peft_clip_len
            B_orig = h_p.shape[0]
            h_p = h_p.view(B_orig * K_split, self.peft_clip_len, H_patches, W_patches, r)
            h_p = h_p.permute(0, 4, 1, 2, 3).contiguous()
            h_p = self.temporal_conv(h_p)
            h_p = h_p.permute(0, 2, 3, 4, 1).contiguous().view(B_orig, T * H_patches * W_patches, r)
        else:
            h_p = h_p.view(B, T, H_patches, W_patches, r).permute(0, 4, 1, 2, 3).contiguous()
            h_p = self.temporal_conv(h_p)
            h_p = h_p.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H_patches * W_patches, r)

        h = torch.cat([h_cls, h_p], dim=1)

        qkv_delta = self.lora_b(h)  # [B, N, 3*C]
        qkv_delta = (
            qkv_delta.reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)

        q = q + self.merge_factor * q_delta
        k = k + self.merge_factor * k_delta
        v = v + self.merge_factor * v_delta
        return q, k, v


class LoRAFormer(nn.Module):
    def __init__(self, dim, num_heads, rank=128, merge_factor=1.0, num_frames=16):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.merge_factor = merge_factor
        self.num_frames = num_frames
        
        self.lora_a = nn.Linear(dim, rank * 3, bias=False)
        self.lora_b = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        B, N, C = x.shape
        T = self.num_frames
        B_temp = B // T
        N_temp = N * T
        heads = self.num_heads
        r_h = self.rank // heads

        x = x.view(B_temp, N_temp, C)  # (T,N)-major

        qkv_delta = self.lora_a(x)
        qkv_delta = qkv_delta.reshape(B_temp, N_temp, 3, heads, r_h).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)  # [B_temp, heads, N_temp, r_h]

        attn = nn.functional.scaled_dot_product_attention(q_delta, k_delta, v_delta)
        # [B_temp, heads, N_temp, r_h]

        attn = attn.permute(0, 2, 1, 3).contiguous()      # [B_temp, N_temp, heads, r_h]
        attn = attn.view(B_temp, T, N, heads, r_h)        # split N_temp -> (T, N)
        attn = attn.reshape(B_temp * T, N, heads, r_h)    # merge (B_temp, T) -> B
        attn = attn.reshape(B, N, self.rank)              # merge heads -> rank

        out = self.lora_b(attn)                           # [B, N, C]
        return out