"""
video_swin.py

This file contains the source code of the backbone of Video Swin Transformer.

Original Sources:
- Video Swin Transformer: https://github.com/SwinTransformer/Video-Swin-Transformer
- Image Swin Transformer: https://github.com/microsoft/Swin-Transformer
"""

# torch
import torch
import torch.nn as nn

# efficient implementation
from einops import rearrange

# utils
from src.utils.video_swin_utils import PatchEmbed3D, BasicLayer, PatchMerging

# Swin Transformer 3D
class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (2,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: (8,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
    """

    def __init__(self,
                 patch_size=(2,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(8,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

    def forward(self, x):
        """Performs the forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, T, H, W).

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x


# Classification Head
class I3DHead(nn.Module):
    """Classification head for I3D.

    Args:
        in_channels (int): Number of channels in input feature.
        num_classes (int): Number of classes to be classified.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        spatial_pooling (bool): Whether to perform spatial pooling. Default: True.
    """

    def __init__(self,
                 num_features=768,
                 num_classes=400,
                 dropout_ratio=0.5,
                 spatial_pooling=True):
        super().__init__()

        # spatial pooling
        if spatial_pooling:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        # dropout
        if dropout_ratio != 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None

        # classification head
        self.fc_cls = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """Performs the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_features, T', H', W').

        Returns:
            torch.Tensor: The classification scores with shape (B, num_classes).
        """
        if self.avg_pool is not None:
            x = self.avg_pool(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        cls_score = self.fc_cls(x)

        return cls_score


# Video Swin 
class VideoSwin(nn.Module):
    """Video Swin Transformer.

    Args:
        patch_size (int | tuple(int)): 
        window_size (int): Window size.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        embed_dim (int): Number of linear projection output channels.
        num_heads (tuple[int]): Number of attention head of each stage.
        drop_path_rate (float): Stochastic depth rate.
        num_classes (int): Number of classes to be classified.  
    """
    def __init__(self, patch_size, window_size, depths, embed_dim, num_heads, drop_path_rate, num_classes, task_type, num_verb_classes, num_noun_classes, is_probe):
        super().__init__()
        self.tubelet_size = 1 # for compatibility with probe
        self.embed_dim = 1024
        self.num_heads = 8
        self.task_type = task_type
        self.is_epic = True if self.task_type == "classification_epic" else False
        self.is_probe = is_probe
        self.grid_size = 392
        self.backbone = SwinTransformer3D(
                            patch_size=patch_size, 
                            window_size=window_size, 
                            depths=depths, 
                            embed_dim=embed_dim, 
                            num_heads=num_heads, 
                            drop_path_rate=drop_path_rate)

        if not self.is_probe:
            if self.is_epic:
                self.verb_head = I3DHead(
                                num_features=self.backbone.num_features, 
                                num_classes=num_verb_classes)
                self.noun_head = I3DHead(
                                num_features=self.backbone.num_features, 
                                num_classes=num_noun_classes)
            else:
                self.cls_head = I3DHead(
                                num_features=self.backbone.num_features, 
                                num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        
        if not self.is_probe:
            if self.is_epic:
                verb_score = self.verb_head(x)
                noun_score = self.noun_head(x)
                x = torch.cat([verb_score, noun_score], dim=1)
            else:
                x = self.cls_head(x)
        else: 
            x = x.view(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

        return x

# Models
def video_swin_tiny(**kwargs):
    model = VideoSwin(
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.1,
        **kwargs,
    )
    return model

def video_swin_base(**kwargs):
    model = VideoSwin(
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.2,
        **kwargs,
    )
    return model

def video_swin_base_ssv2(**kwargs):
    model = VideoSwin(
        patch_size=(2, 4, 4),
        window_size=(16, 7, 7),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.4,
        **kwargs,
    )
    return model