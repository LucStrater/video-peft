"""
vda.py

This file contains the source code for VDA head.

- source: https://github.com/DepthAnything/Video-Depth-Anything/tree/main
- they used in turn three different sources of which the license statement 
  is included in the file at the relevant code chunk.
- Modified by [anonymous author] for the purposes of this project.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from src.models.probes.dpt import DPTHead
from einops import rearrange, repeat
import math
from typing import Optional, Tuple

try:
    import xformers
    import xformers.ops

    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available")
    XFORMERS_AVAILABLE = False


# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.upcast_efficient_attention = False

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size).contiguous()
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size).contiguous()
        return tensor

    def reshape_heads_to_4d(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size).contiguous()
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim).contiguous()
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size).contiguous()
        return tensor

    def reshape_4d_to_heads(self, tensor):
        batch_size, seq_len, head_size, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, dim * head_size).contiguous()
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if XFORMERS_AVAILABLE and self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        if self.upcast_efficient_attention:
            org_dtype = query.dtype
            query = query.float()
            key = key.float()
            value = value.float()
            if attention_mask is not None:
                attention_mask = attention_mask.float()
        hidden_states = self._memory_efficient_attention_split(query, key, value, attention_mask)

        if self.upcast_efficient_attention:
            hidden_states = hidden_states.to(org_dtype)

        hidden_states = self.reshape_4d_to_heads(hidden_states)
        return hidden_states

        # print("Errror: no xformers")
        # raise NotImplementedError

    def _memory_efficient_attention_split(self, query, key, value, attention_mask):
        batch_size = query.shape[0]
        max_batch_size = 65535
        num_batches = (batch_size + max_batch_size - 1) // max_batch_size
        results = []
        for i in range(num_batches):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size, batch_size)
            query_batch = query[start_idx:end_idx]
            key_batch = key[start_idx:end_idx]
            value_batch = value[start_idx:end_idx]
            if attention_mask is not None:
                attention_mask_batch = attention_mask[start_idx:end_idx]
            else:
                attention_mask_batch = None
            result = xformers.ops.memory_efficient_attention(query_batch, key_batch, value_batch, attn_bias=attention_mask_batch)
            results.append(result)
        full_result = torch.cat(results, dim=0)
        return full_result


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous())
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous())
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)



# This file is originally from AnimateDiff/animatediff/models/motion_module.py at main · guoyww/AnimateDiff
# SPDX-License-Identifier: Apache-2.0 license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file was released under [ Apache-2.0 license], with the full license text available at [https://github.com/guoyww/AnimateDiff?tab=Apache-2.0-1-ov-file#readme].

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class TemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        num_attention_blocks               = 2,
        norm_num_groups                    = 32,
        temporal_max_len                   = 32,
        zero_initialize                    = True,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads,
            num_layers=num_transformer_block,
            num_attention_blocks=num_attention_blocks,
            norm_num_groups=norm_num_groups,
            temporal_max_len=temporal_max_len,
            pos_embedding_type=pos_embedding_type,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None, cached_hidden_state_list=None):
        hidden_states = input_tensor
        hidden_states, output_hidden_state_list = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask, cached_hidden_state_list)

        output = hidden_states
        return output, output_hidden_state_list  # list of hidden states


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        num_attention_blocks               = 2,
        norm_num_groups                    = 32,
        temporal_max_len                   = 32,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_attention_blocks=num_attention_blocks,
                    temporal_max_len=temporal_max_len,
                    pos_embedding_type=pos_embedding_type,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cached_hidden_state_list=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        output_hidden_state_list = []

        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim).contiguous()
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        if cached_hidden_state_list is not None:
            n = len(cached_hidden_state_list) // len(self.transformer_blocks)
        else:
            n = 0
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, hidden_state_list = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, attention_mask=attention_mask,
                                                     cached_hidden_state_list=cached_hidden_state_list[i*n:(i+1)*n] if n else None)
            output_hidden_state_list.extend(hidden_state_list)

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output, output_hidden_state_list


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        num_attention_blocks               = 2,
        temporal_max_len                   = 32,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        self.attention_blocks = nn.ModuleList(
            [
                TemporalAttention(
                        query_dim=dim,
                        heads=num_attention_heads,
                        dim_head=attention_head_dim,
                        temporal_max_len=temporal_max_len,
                        pos_embedding_type=pos_embedding_type,
                )
                for i in range(num_attention_blocks)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dim)
                for i in range(num_attention_blocks)
            ]
        )

        self.ff = FeedForward(dim, dropout=0.0, activation_fn="geglu")
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_state_list=None):
        output_hidden_state_list = []
        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            residual_hidden_states, output_hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                attention_mask=attention_mask,
                cached_hidden_states=cached_hidden_state_list[i] if cached_hidden_state_list is not None else None,
            )
            hidden_states = residual_hidden_states + hidden_states
            output_hidden_state_list.append(output_hidden_states)

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output, output_hidden_state_list


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.,
        max_len = 32
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.dtype)
        return self.dropout(x)

class TemporalAttention(CrossAttention):
    def __init__(
            self,
            temporal_max_len                   = 32,
            pos_embedding_type                 = "ape",
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.pos_embedding_type = pos_embedding_type
        self._use_memory_efficient_attention_xformers = True

        self.pos_encoder = None
        self.freqs_cis = None
        if self.pos_embedding_type == "ape":
            self.pos_encoder = PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.,
                max_len=temporal_max_len
            )

        elif self.pos_embedding_type == "rope":
            self.freqs_cis = precompute_freqs_cis(
                kwargs["query_dim"],
                temporal_max_len
            )

        else:
            raise NotImplementedError

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_states=None):
        # TODO: support cache for these
        assert encoder_hidden_states is None
        assert attention_mask is None

        d = hidden_states.shape[1]
        d_in = 0
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            input_hidden_states = hidden_states  # (bxd) f c
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            input_hidden_states = hidden_states
            d_in = cached_hidden_states.shape[1]
            hidden_states = torch.cat([cached_hidden_states, hidden_states], dim=1)

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states[:, d_in:, ...])
        dim = query.shape[-1]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if self.freqs_cis is not None:
            seq_len = query.shape[1]
            freqs_cis = self.freqs_cis[:seq_len].to(query.device)
            query, key = apply_rotary_emb(query, key, freqs_cis)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)


        use_memory_efficient = XFORMERS_AVAILABLE and self._use_memory_efficient_attention_xformers
        if use_memory_efficient and (dim // self.heads) % 8 != 0:
            # print('Warning: the dim {} cannot be divided by 8. Fall into normal attention'.format(dim // self.heads))
            use_memory_efficient = False

        # attention, what we cannot get enough of
        if use_memory_efficient:
            query = self.reshape_heads_to_4d(query)
            key = self.reshape_heads_to_4d(key)
            value = self.reshape_heads_to_4d(value)

            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                raise NotImplementedError
                # hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states, input_hidden_states

# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

class DPTHeadTemporal(DPTHead):
    """DPT head with temporal transformer mixing at 4 stages (Video Depth Anything)."""

    def __init__(
        self,
        in_channels,
        num_frames,
        model_tubelet_size,
        grid_size,
        input_size,
        task_type="depth_estimation",
        num_classes=1,
        max_depth=10.0,
        use_bn=False,
        use_clstoken=False,
        pe='ape',
    ):
        super().__init__(
            in_channels=in_channels,
            num_frames=num_frames,
            model_tubelet_size=model_tubelet_size,
            grid_size=grid_size,
            input_size=input_size,
            task_type=task_type,
            num_classes=num_classes,
            max_depth=max_depth,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
        )

        features = 256 if in_channels >= 1024 else 128
        out_channels = [256, 512, 1024, 1024] if in_channels >= 1024 else [96, 192, 384, 768]

        motion_kwargs = dict(
            num_attention_heads=8,
            num_transformer_block=1,
            num_attention_blocks=2,
            temporal_max_len=num_frames,
            zero_initialize=True,
            pos_embedding_type=pe,
        )

        # 4 temporal modules: layer_3, layer_4, path_4, path_3
        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], **motion_kwargs),
            TemporalModule(in_channels=out_channels[3], **motion_kwargs),
            TemporalModule(in_channels=features, **motion_kwargs),
            TemporalModule(in_channels=features, **motion_kwargs),
        ])

    def _apply_motion(self, idx, x, B, T):
        # (B*T, C, H, W) -> (B, C, T, H, W) -> motion -> (B*T, C, H, W)
        x = x.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        x, _ = self.motion_modules[idx](x, None, None, None)
        return x.permute(0, 2, 1, 3, 4).flatten(0, 1)

    def forward_features(self, out_features, patch_h, patch_w):
        T = self.t_size
        B = out_features[0].shape[0] // T

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_3 = self._apply_motion(0, layer_3, B, T)
        layer_4 = self._apply_motion(1, layer_4, B, T)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4 = self._apply_motion(2, path_4, B, T)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3 = self._apply_motion(3, path_3, B, T)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        return self.scratch.output_conv1(path_1)