# basic
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path, to_2tuple


# Helper Functions
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in the main path of residual blocks).

    Args:
        drop_prob (float, optional): Probability of dropping a path. Default is None.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

def get_position_angle_vec(position, d_hid): 
    """Get the position angle vector.

    Args:
        position (int): The position of the vector.
        d_hid (int): The dimension of the vector.
    """
    return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

def get_sinusoid_encoding_table(n_position, d_hid): 
    """Sinusoid position encoding table.

    Args:
        n_position (int): Number of positions.
        d_hid (int): Dimension of the hidden layer.
    """

    sinusoid_table = np.array([get_position_angle_vec(pos_i, d_hid) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 

def patch_scoring(M, threshold=0.):
    """
    LOST patch scoring based on the inverse degree.
    """
    # Cloning important
    A = M.clone()

    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    sel = torch.argsort(cent, descending=True)

    return sel, cent


# ViT Modules
class Mlp(nn.Module):
    """Multi-Layer Perceptron.

    Args:
        in_features (int): Input features.
        hidden_features (int, optional): Hidden features. Defaults to None.
        out_features (int, optional): Output features. Defaults to None.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        drop (float, optional): Dropout rate. Defaults to 0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """Attention module.

    Args:
        dim (int): Dimension of the input.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to use bias in the query, key, and value projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for the query-key dot product. Defaults to None.
        attn_drop (float, optional): Dropout rate for the attention weights. Defaults to 0.
        proj_drop (float, optional): Dropout rate for the projection. Defaults to 0.
        attn_head_dim (int, optional): Dimension of each attention head. Defaults to None.
    """

    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """Transformer block.

    Args:
        dim (int): Dimension of the input.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension. Defaults to 4.
        qkv_bias (bool, optional): Whether to use bias in the query, key, and value projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for the query-key dot product. Defaults to None.
        drop (float, optional): Dropout rate. Defaults to 0.
        attn_drop (float, optional): Dropout rate for the attention weights. Defaults to 0.
        drop_path (float, optional): Dropout rate for the path. Defaults to 0.
        init_values (float, optional): Initial values for the gamma parameters. Defaults to None.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        attn_head_dim (int, optional): Dimension of each attention head. Defaults to None.
    """
        
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int, optional): Image size. Defaults to 224.
        patch_size (int, optional): Patch size. Defaults to 16.
        in_chans (int, optional): Number of input channels. Defaults to 3.
        embed_dim (int, optional): Dimension of the embedding. Defaults to 768.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class LayerDecayValueAssigner:
    def __init__(self, values):
        self.values = values
        self.num_layers = len(self.values)

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        if var_name in ['cls_token', 'mask_token', 'pos_embed']:
            return 0
        elif var_name.startswith('patch_embed'):
            return 0
        elif var_name.startswith('rel_pos_bias'):
            return self.num_layers - 1
        elif var_name.startswith('blocks'):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return self.num_layers - 1
        
