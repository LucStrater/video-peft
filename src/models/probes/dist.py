import math
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange, repeat

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class IntegrationNetwork(nn.Module):
    def __init__(self, num_frames, d_model, integration_dim=384, t_kernel_size=3, mlp_ratio=1, spatial_temporal_mlp_ratio=0.25, alpha=1):
        super().__init__()
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(integration_dim, int(integration_dim * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(integration_dim * mlp_ratio), integration_dim))
        ]))
        self.temporal_ffn = nn.Sequential(OrderedDict([
            ("c_fc1", nn.Conv3d(integration_dim, int(integration_dim * spatial_temporal_mlp_ratio), kernel_size=(1, 1, 1), padding=(0, 0, 0))),
            ("c_fc2", nn.Conv3d(int(integration_dim * spatial_temporal_mlp_ratio), int(integration_dim * spatial_temporal_mlp_ratio), kernel_size=(t_kernel_size, 1, 1), padding=(t_kernel_size//2, 0, 0))),
            ("gelu1", QuickGELU()),
            ("c_proj", nn.Conv3d(int(integration_dim * spatial_temporal_mlp_ratio), integration_dim, kernel_size=(1), padding=(0), stride=1))
        ]))
        self.ln = LayerNorm(integration_dim)
        self.ln_temporal = LayerNorm(integration_dim)
        self.num_frames = num_frames
        self.alpha = int(alpha)

    def forward(self, x):
        l, bt, c = x.size()
        b, t = bt // (self.num_frames // self.alpha), self.num_frames // self.alpha
        t_x = self.ln_temporal(x).view(l, b, t, c).permute(1, 3, 2, 0).reshape(b, c, t, l, 1)
        t_x = self.temporal_ffn(t_x).flatten(3).permute(3, 0, 2, 1).flatten(1, 2)
        return self.ffn(self.ln(x)) + t_x


class TemporalNet(nn.Module):
    def __init__(self, num_frames, d_model, temporal_dim=96, t_kernel_size=3, mlp_ratio=1):
        super().__init__()
        self.temporal_net = nn.Sequential(OrderedDict([
            ("c_fc1", nn.Conv3d(temporal_dim, int(temporal_dim * mlp_ratio), kernel_size=(t_kernel_size, 1, 1), padding=(t_kernel_size//2, 0, 0))),
            ("gelu1", QuickGELU()),
            ("c_fc2", nn.Conv3d(int(temporal_dim * mlp_ratio), temporal_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))),
        ]))
        self.gelu = QuickGELU()
        self.ln = LayerNorm(temporal_dim)
        self.num_frames = num_frames

    def forward(self, x):
        assert hasattr(self, 'ln')
        return self.gelu(x + self.temporal_net(self.ln(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)))


class Temporal2IntegrationNetwork(nn.Module):
    def __init__(self, num_frames, d_model, temporal_dim=96, integration_dim=384, alpha=1):
        super().__init__()
        self.alpha = int(alpha)
        self.num_frames = num_frames
        self.linear_fuse = nn.Conv3d(temporal_dim, integration_dim, kernel_size=(self.alpha, 1, 1), padding=(0, 0, 0), stride=(self.alpha, 1, 1))

    def forward(self, x):
        x = self.linear_fuse(x).flatten(3)  # (B, C, T//alpha, HW)
        b, c, t, hw = x.size()
        x = x.permute(3, 0, 2, 1)  # (HW, B, T//alpha, C)
        x = x.flatten(1, 2)         # (HW, B*T//alpha, C)
        return x

class Integration2TemporalNetwork(nn.Module):
    def __init__(self, num_frames, d_model, integration_dim=384, temporal_dim=96, alpha=1):
        super().__init__()
        self.alpha = int(alpha)
        self.num_frames = num_frames
        self.linear_fuse = nn.Linear(integration_dim, temporal_dim)

    def forward(self, x):
        x = self.linear_fuse(x)
        l, bt, c = x.size()
        b, t = bt // (self.num_frames // self.alpha), self.num_frames // self.alpha
        x = x.view(l, b, t, c).permute(1, 3, 2, 0).reshape(b, c, t, int(math.sqrt(l)), int(math.sqrt(l)))
        # return x.repeat_interleave(self.alpha // self.temporal_alpha, dim=2)
        return torch.nn.functional.upsample_nearest(x, size=(x.size(2)*(self.alpha), x.size(3), x.size(4)))

class DiSTNetwork(nn.Module):
    """
    DiST feature enhancer: processes all-layer backbone features + raw images
    through a spatial-temporal integration network.
    Outputs (B, T*HW, output_embed_dim) — enriched spatiotemporal features
    in the same token layout as the backbone, ready for any downstream pooler.
    """
    def __init__(self, num_frames, d_model, num_layers,
                 grid_size=16, temporal_dim=96, integration_dim=384,
                 spatial_patch_size=14, temporal_patch_size=5, sparse_sample_alpha=1):
        super().__init__()
        self.selected_layers = list(range(num_layers))
        self.alpha = int(sparse_sample_alpha)
        self.num_frames = num_frames
        self.grid_size = grid_size
        self.integration_dim = integration_dim

        self.temporal_stem = nn.Conv3d(3, temporal_dim,
                                       kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                                       stride=(1, spatial_patch_size, spatial_patch_size),
                                       padding=(temporal_patch_size//2, 0, 0))

        self.input_linears = nn.ModuleList([nn.Linear(d_model, integration_dim) for i in range(num_layers)])
        self.integration2temporal_nets = nn.ModuleList([Integration2TemporalNetwork(num_frames, d_model=d_model, alpha=sparse_sample_alpha) for i in range(num_layers - 1)])
        self.temporal2integration_nets = nn.ModuleList([Temporal2IntegrationNetwork(num_frames, d_model=d_model, alpha=sparse_sample_alpha) for i in range(num_layers)])
        self.temporal_nets = nn.ModuleList([TemporalNet(num_frames, d_model=d_model) for i in self.selected_layers])
        self.integration_nets = nn.ModuleList([IntegrationNetwork(num_frames, d_model=d_model, alpha=sparse_sample_alpha) for i in self.selected_layers])
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'skip_init') and m.skip_init:
            return
        from timm.models.layers import trunc_normal_
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, mid_features, images):
        """
        Args:
            mid_features: list of per-layer features, each (HW, B*T, C)
            images: raw input frames (B, 3, T, H, W)
        Returns:
            (B, T*HW, output_embed_dim)
        """
        HW, bt, c = mid_features[self.selected_layers[0]].size()
        t = self.num_frames // self.alpha
        b = bt // t
        x_temporal = self.temporal_stem(images)
        res_feat = 0.0
        for idx, layer_id in enumerate(self.selected_layers):
            x_temporal = self.temporal_nets[idx](x_temporal)
            mid_feat = self.input_linears[idx](mid_features[layer_id]) + res_feat

            updated_mid_feat = mid_feat + self.temporal2integration_nets[idx](x_temporal)
            res_feat = self.integration_nets[idx](updated_mid_feat)

            if idx < len(self.selected_layers) - 1:
                x_temporal = self.integration2temporal_nets[idx](mid_feat) + x_temporal

        # spatial features: (HW, B*T, integration_dim) -> (B, T*HW, output_embed_dim)
        spatial_feat = res_feat + updated_mid_feat
        spatial_feat = spatial_feat.permute(1, 0, 2)            # (B*T, HW, integration_dim)
        features = spatial_feat.reshape(b, t * HW, -1)         # (B, T*HW, integration_dim)
        return features


