"""
video_mae.py

This file contains the source code of the backbone of VideoMAE.

Original Source:
- VideoMAE: https://github.com/MCG-NJU/VideoMAE 
"""

# basic
from functools import partial
from contextlib import nullcontext

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

# utils
from src.utils.video_mae_utils import PatchEmbed, get_sinusoid_encoding_table, Block, patch_scoring

class VideoMAE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage.

    Args:
        img_size (int, optional): Image size. Defaults to 224.
        patch_size (int, optional): Patch size. Defaults to 16.
        in_chans (int, optional): Number of input channels. Defaults to 3.
        num_classes (int, optional): Number of classes. Defaults to 1000.
        embed_dim (int, optional): Dimension of the embedding. Defaults to 768.
        depth (int, optional): Depth of the transformer. Defaults to 12.
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension. Defaults to 4.
        qkv_bias (bool, optional): Whether to use bias in the query, key, and value projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for the query-key dot product. Defaults to None.
        fc_drop_rate (float, optional): Dropout rate for the final fully connected layer. Defaults to 0.
        drop_rate (float, optional): Dropout rate. Defaults to 0.
        attn_drop_rate (float, optional): Dropout rate for the attention weights. Defaults to 0.
        drop_path_rate (float, optional): Dropout rate for the path. Defaults to 0.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        init_values (float, optional): Initial values for the gamma parameters. Defaults to 0.
        use_learnable_pos_emb (bool, optional): Whether to use learnable positional embeddings. Defaults to False.
        init_scale (float, optional): Initial scale for the final fully connected layer. Defaults to 0.
        all_frames (int, optional): Number of frames. Defaults to 16.
        tubelet_size (int, optional): Size of the tubelet. Defaults to 2.
        use_checkpoint (bool, optional): Whether to use checkpointing. Defaults to False.
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=400, 
                 embed_dim=384, 
                 depth=12,
                 num_heads=6, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.1, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 init_values=0.,
                 init_scale=0.001,
                 all_frames=16,
                 tubelet_size=2,
                 feature_type="spatial",
                 task_type="classification",
                 num_verb_classes=97,
                 num_noun_classes=300,
                 is_probe=False):
        super().__init__()

        # params
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_side = self.grid_size = self.img_size // self.patch_size
        self.patches_per_time = all_frames // self.tubelet_size
        self.feature_type = feature_type
        self.task_type = task_type
        self.is_epic = True if self.task_type == "classification_epic" else False
        self.num_verb_classes = num_verb_classes
        self.num_noun_classes = num_noun_classes
        self.is_probe = is_probe
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # patch embed
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim, 
            num_frames=all_frames, 
            tubelet_size=self.tubelet_size)
        
        # pos embed
        num_patches = self.patch_embed.num_patches
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.norm = norm_layer(embed_dim)

        if not self.is_probe:
            if self.is_epic:
                self.verb_head = nn.Linear(embed_dim, num_verb_classes)
                self.noun_head = nn.Linear(embed_dim, num_noun_classes)

                trunc_normal_(self.verb_head.weight, std=.02)
                trunc_normal_(self.noun_head.weight, std=.02)
                self.apply(self._init_weights)

                self.verb_head.weight.data.mul_(init_scale)
                self.verb_head.bias.data.mul_(init_scale)
                self.noun_head.weight.data.mul_(init_scale)
                self.noun_head.bias.data.mul_(init_scale)
            else:
                self.head = nn.Linear(embed_dim, num_classes) 

                trunc_normal_(self.head.weight, std=.02)
                self.apply(self._init_weights)

                self.head.weight.data.mul_(init_scale)
                self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if (not self.is_probe) or (self.feature_type == "gap"):
            x = self.norm(x.mean(1))
        elif self.feature_type == "spatial":
            x = self.norm(x)
        else:
            x = x[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if not self.is_probe:
            if self.is_epic:
                verb_score = self.verb_head(x)
                noun_score = self.noun_head(x)
                x = (verb_score, noun_score)
            else:
                x = self.head(self.fc_dropout(x))
        return x
    
    @torch.no_grad()
    def forward_interpolate_pos_embed(self, x):
        # patch embed
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)
        
        # interpolate positional embedding
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if (H_patches != self.patches_per_side) or (W_patches != self.patches_per_side):
            pos_embed_reshaped = self.pos_embed.squeeze(0).reshape(self.patches_per_time, self.patches_per_side, self.patches_per_side, -1).permute(0, 3, 1, 2) 
            pos_embed_interpolated = F.interpolate(
                pos_embed_reshaped, 
                size=(H_patches, W_patches), 
                mode='bicubic', 
                align_corners=False
            )
            pos_embed_interpolated = pos_embed_interpolated.permute(0, 2, 3, 1).unsqueeze(0)
            new_pos_embed = pos_embed_interpolated.reshape(1, self.patches_per_time * H_patches * W_patches, -1)
            x = x + new_pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        else:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x.mean(1))
        x = self.head(self.fc_dropout(x))
        return x        
    
    @torch.no_grad()
    def forward_attention(self, x, which_features='attn'):
        """
        Forward pass to get column-wise attention maps from the last layer.
        Returns attention map of shape (batch, T, H, W)
        """
        # patch embed
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)
        
        # interpolate positional embedding
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if (H_patches != self.patches_per_side) or (W_patches != self.patches_per_side):
            pos_embed_reshaped = self.pos_embed.squeeze(0).reshape(self.patches_per_time, self.patches_per_side, self.patches_per_side, -1).permute(0, 3, 1, 2) 
            pos_embed_interpolated = F.interpolate(
                pos_embed_reshaped, 
                size=(H_patches, W_patches), 
                mode='bicubic', 
                align_corners=False
            )
            pos_embed_interpolated = pos_embed_interpolated.permute(0, 2, 3, 1).unsqueeze(0)
            new_pos_embed = pos_embed_interpolated.reshape(1, self.patches_per_time * H_patches * W_patches, -1)
            x = x + new_pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        else:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        x = self.pos_drop(x)
        
        # forward all blocks except the last one
        for i in range(len(self.blocks) - 1):
            x = self.blocks[i](x)
        
        # last block 
        last_block = self.blocks[-1]

        if which_features == 'features':
            features = last_block(x)
            features = self.norm(features)
            return features, H_patches, W_patches

        norm_x = last_block.norm1(x)
        
        # calculate QKV
        B, N, C = norm_x.shape
        qkv_bias = None
        if last_block.attn.q_bias is not None:
            qkv_bias = torch.cat((last_block.attn.q_bias, 
                                torch.zeros_like(last_block.attn.v_bias, requires_grad=False), 
                                last_block.attn.v_bias))
        
        qkv = F.linear(input=norm_x, weight=last_block.attn.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, last_block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        if which_features == 'q':
            q = q.transpose(1, 2).reshape(B, N, -1)
            return q
        elif which_features == 'k':
            k = k.transpose(1, 2).reshape(B, N, -1)
            return k, H_patches, W_patches
        elif which_features == 'v':
            v = v.transpose(1, 2).reshape(B, N, -1)
            return v, H_patches, W_patches

        q = q * last_block.attn.scale
        attn = (q @ k.transpose(-2, -1))  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        
        # # average across heads (B, N, N)
        # attn_avg_heads = attn.mean(dim=1)
        
        # randomly select a head for each batch item
        B, num_heads, N, _ = attn.shape
        random_head_indices = torch.randint(0, num_heads, (B,), device=attn.device)
        attn_avg_heads = attn[torch.arange(B), random_head_indices]  # (B, N, N)
      
        # sum of attention that each token receives 
        attn_per_token = attn_avg_heads.sum(dim=1)
        attn_map = attn_per_token.reshape(B, self.patches_per_time, H_patches, W_patches)

        if which_features == 'attn':
            return attn_map
        elif which_features == 'features+attn':
            features = last_block(x)
            features = self.norm(features)
            return features, attn_map, H_patches, W_patches
        elif which_features == 'q+attn':
            q = q.transpose(1, 2).reshape(B, N, -1)
            return q, attn_map, H_patches, W_patches
        elif which_features == 'k+attn':
            k = k.transpose(1, 2).reshape(B, N, -1)
            return k, attn_map, H_patches, W_patches
        elif which_features == 'v+attn':
            v = v.transpose(1, 2).reshape(B, N, -1)
            return v, attn_map, H_patches, W_patches
        
    @torch.no_grad()
    def forward_lost_score(self, x, k_patches=100, eps=1e-5):
        """
        Forward pass to get lost patches.
        """
        feats, attn_map, H_p, W_p = self.forward_attention(x, which_features='k+attn')   # (B,N,C)
        feats = F.normalize(feats, p=2, dim=-1)

        B, _, _ = feats.shape
        A = torch.bmm(feats, feats.transpose(1, 2))
        all_scores = []

        for i in range(B):
            sample_A = A[i]
            
            # Score patches and get sorted indices
            sorted_patches, scores = patch_scoring(sample_A)
            
            # Select seed (patch with lowest degree)
            seed = sorted_patches[0]
            
            # Seed expansion
            potentials = sorted_patches[:k_patches]
            similars = potentials[sample_A[seed, potentials] > 0.0]
            
            # Calculate expansion scores
            M = torch.sum(sample_A[similars, :], dim=0)
            
            all_scores.append(M)
        
        # lost score map
        v = torch.stack(all_scores)            

        # reshape
        v = v.to(feats.dtype).view(B, self.patches_per_time, H_p, W_p)
    
        # sign based on attention map correlation
        v_mean = v.mean(dim=(1, 2, 3), keepdim=True)                       # (B, 1, 1, 1)
        bipartition_mask = v > v_mean                                       # (B, T, H, W)
        
        # average attention for true and false partitions
        attn_true = torch.where(bipartition_mask, attn_map, 0.0).sum(dim=(1, 2, 3)) / (bipartition_mask.sum(dim=(1, 2, 3)) + eps)    # (B,)
        attn_false = torch.where(~bipartition_mask, attn_map, 0.0).sum(dim=(1, 2, 3)) / ((~bipartition_mask).sum(dim=(1, 2, 3)) + eps)  # (B,)
        
        # flip sign if false partition has higher attention
        should_flip = attn_false > attn_true                                # (B,)
        flip_factor = torch.where(should_flip, -1.0, 1.0).view(B, 1, 1, 1) # (B, 1, 1, 1)
        v = v * flip_factor
        
        return v       

    @torch.no_grad()
    def forward_ncut_score(self, x, tau=0.0, eps=1e-5, exact=False):
        """
        NCut score calculation. 
        1. A = binary affinity matrix of features
        2. D = diag(sum(A, dim=1))
        3. L = D - A
        4. Ĺ_norm = D^{-½} * L * D^{-½}
        5. w = 2nd-smallest eigvec(Ĺ_norm), v = D^{-½} w

        Returns
        -------
        ncut_score:
            (batch, T, H, W) map of the 2nd-smallest generalised-Laplacian
            eigenvector.
        """
        feats, attn_map, H_p, W_p = self.forward_attention(x, which_features='k+attn')   # (B,N,C)
        feats = F.normalize(feats, p=2, dim=-1)

        B, _, _ = feats.shape
        A = torch.bmm(feats, feats.transpose(1, 2))                         # (B,N,N)
        A = torch.where(A > tau, A.new_ones(()), A.new_full((), eps))

        # laplacian
        deg = A.sum(dim=-1) + eps                                           # avoid zeros
        L   = torch.diag_embed(deg) - A                                     # (B,N,N)
        D_inv_sqrt = deg.rsqrt()                                            # (B,N)

        L_norm = (D_inv_sqrt.unsqueeze(-1) * L.double() * D_inv_sqrt.unsqueeze(-2))   
        # eigendecomposition in full precision with AMP disabled
        amp_off = (torch.amp.autocast(device_type='cuda', enabled=False)
                if torch.is_autocast_enabled() else nullcontext())
        with amp_off:
            if exact:
                _, eigenvecs = torch.linalg.eigh(L_norm)                            # (B,N), (B,N,N)
            else:
                _, eigenvecs = torch.lobpcg(L_norm, k=2, largest=False)
                if torch.isnan(eigenvecs).any():
                    _, eigenvecs = torch.linalg.eigh(L_norm)
            w = eigenvecs[..., 1]                                               # 2nd-smallest eigenvector

        v = D_inv_sqrt * w              

        # reshape
        v = v.to(feats.dtype).view(B, self.patches_per_time, H_p, W_p)
    
        # sign based on attention map correlation
        v_mean = v.mean(dim=(1, 2, 3), keepdim=True)                       # (B, 1, 1, 1)
        bipartition_mask = v > v_mean                                       # (B, T, H, W)
        
        # average attention for true and false partitions
        attn_true = torch.where(bipartition_mask, attn_map, 0.0).sum(dim=(1, 2, 3)) / (bipartition_mask.sum(dim=(1, 2, 3)) + eps)    # (B,)
        attn_false = torch.where(~bipartition_mask, attn_map, 0.0).sum(dim=(1, 2, 3)) / ((~bipartition_mask).sum(dim=(1, 2, 3)) + eps)  # (B,)
        
        # flip sign if false partition has higher attention
        should_flip = attn_false > attn_true                                # (B,)
        flip_factor = torch.where(should_flip, -1.0, 1.0).view(B, 1, 1, 1) # (B, 1, 1, 1)
        v = v * flip_factor
        
        return v


# Models
def video_mae_small(patch_size=16, **kwargs):
    model = VideoMAE(
        patch_size=patch_size,
        tubelet_size=2,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        **kwargs,
    )
    return model

def video_mae_base(patch_size=16, **kwargs):
    model = VideoMAE(
        patch_size=patch_size,
        tubelet_size=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        **kwargs,
    )
    return model

def video_mae_large(patch_size=16, **kwargs):
    model = VideoMAE(
        patch_size=patch_size,
        tubelet_size=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )
    return model

