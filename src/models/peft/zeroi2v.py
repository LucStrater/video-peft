import torch
import torch.nn as nn

def default_head_offsets(num_frames):
    if num_frames <= 8:
        offsets = [1, -1]                 # RF 3
    elif num_frames <= 16:
        offsets = [1, -1, 2, -2]          # RF 5
    else:
        offsets = [1, -1, 2, -2, 3, -3]       # RF 6
    return {i: dt for i, dt in enumerate(offsets) if dt != 0}


def temporal_head_shift(t, T):
    head_offsets = default_head_offsets(T)
    BT, H, N, hd = t.shape
    B = BT // T
    t = t.view(B, T, H, N, hd)
    out = t.clone()
    for head_idx, s in head_offsets.items():
        if s > 0:                                   # frame i attends to i + s (look forward)
            out[:, :-s, head_idx] = t[:, s:, head_idx]
        elif s < 0:                                 # frame i attends to i - |s| (look back)
            a = -s
            out[:, a:, head_idx] = t[:, :-a, head_idx]
    return out.view(BT, H, N, hd)


class LinearAdapter(nn.Module):
    def __init__(self, dim, bottleneck, scale=0.5, skip_connect=True, dropout=0.1):
        super().__init__()
        self.skip_connect = skip_connect
        self.scale = scale
        self.adapter_fc1 = nn.Linear(dim, bottleneck)
        self.adapter_drop = nn.Dropout(dropout)
        self.adapter_fc2 = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.adapter_fc2.weight)
        nn.init.zeros_(self.adapter_fc2.bias)

    def forward(self, x):
        xs = self.adapter_fc2(self.adapter_drop(self.adapter_fc1(x)))
        return x + xs * self.scale if self.skip_connect else xs * self.scale


class TemporalEmbedding(nn.Module):
    def __init__(self, num_frames, dim):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, 1, dim))

    def forward(self, x, T):
        BT, N, D = x.shape
        B = BT // T
        x = x.view(B, T, N, D) + self.temporal_embedding
        return x.view(BT, N, D)
