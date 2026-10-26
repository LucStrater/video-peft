import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, in_channels, adapter_channels=384, kernel_size=(3, 1, 1),
                 backbone_clip_len=1, peft_clip_len=1, adapter_type="st-adapter", 
                 window_divisors=(1, 2, 4), act_fn=None):
        super().__init__()
        self.peft_clip_len = peft_clip_len
        self.decouple = (peft_clip_len != backbone_clip_len)
        self.adapter_type = adapter_type
        self.adapter_channels = adapter_channels
        self.window_divisors = tuple(window_divisors)
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.act_fn = act_fn() if act_fn is not None else None

        if adapter_type == "st-adapter":
            self.conv = nn.Conv3d(
                adapter_channels, adapter_channels,
                kernel_size=kernel_size,
                stride=(1, 1, 1),
                padding=tuple(k // 2 for k in kernel_size),
                groups=adapter_channels,
            )
            nn.init.constant_(self.conv.weight, 0.)
            nn.init.constant_(self.conv.bias, 0.)
            nn.init.constant_(self.fc1.bias, 0.)
            nn.init.constant_(self.fc2.bias, 0.)
        elif adapter_type == "frame2freq_ms":
            assert adapter_channels % 2 == 0, "adapter_channels must be even (split into freq/temp halves)"
            half = adapter_channels // 2
            self.conv_freq = nn.Conv3d(
                half, half, kernel_size=kernel_size, stride=(1, 1, 1),
                padding=tuple(k // 2 for k in kernel_size), groups=half,
            )
            self.conv_temp = nn.Conv3d(
                half, half, kernel_size=kernel_size, stride=(1, 1, 1),
                padding=tuple(k // 2 for k in kernel_size), groups=half,
            )

            for conv in (self.conv_freq, self.conv_temp):
                nn.init.constant_(conv.weight, 0.)
                nn.init.constant_(conv.bias, 0.)
            nn.init.constant_(self.fc1.bias, 0.)
            nn.init.constant_(self.fc2.bias, 0.)

    def _adapter(self, h):
        if self.act_fn is not None:
            h = self.act_fn(h)
        return h
    
    def _st_adapter(self, h):
        h = self.conv(h)
        if self.act_fn is not None:
            h = self.act_fn(h)
        return h
    
    def _frame2freq_ms(self, h, T):
        B, Ca, _, Hp, Wp = h.shape
        x_freq, x_temp = torch.split(h, Ca // 2, dim=1)  # (B, Ca/2, T, H, W)
        out_temp = self.conv_temp(x_temp)
        out_freq = torch.zeros_like(x_freq)
        # only use window divisors that evenly divide the current T; d=1 (global window) always
        # divides T, so small temporal contexts (e.g. Tc=1, 2) degrade to fewer scales 
        active_divisors = [d for d in self.window_divisors if T % d == 0]
        for d in active_divisors:
            wk = T // d
            xf = x_freq.view(B, Ca // 2, d, wk, Hp, Wp)

            # rFFT along the window (time) axis 
            spec = torch.fft.rfft(xf, n=wk, dim=3)          # complex
            Ffreq = spec.shape[3]

            spec = spec.permute(0, 2, 1, 3, 4, 5).reshape(B * d, Ca // 2, Ffreq, Hp, Wp)
            spec = torch.complex(self.conv_freq(spec.real), self.conv_freq(spec.imag))
            spec = spec.view(B, d, Ca // 2, Ffreq, Hp, Wp).permute(0, 2, 1, 3, 4, 5)

            # iFFT back to length wk and stitch windows 
            rec = torch.fft.irfft(spec, n=wk, dim=3)        # (B, Ca/2, d, wk, H, W)
            rec = rec.reshape(B, Ca // 2, T, Hp, Wp)
            out_freq = out_freq + rec
        out_freq = out_freq / len(active_divisors)

        h_out =  torch.cat([out_freq, out_temp], dim=1)

        if self.act_fn is not None:
            h_out = self.act_fn(h_out)
        return h_out

    def _forward_bottleneck(self, h, T=None):
        if self.adapter_type == "adapter":
            return self._adapter(h)
        elif self.adapter_type == "st-adapter":
            return self._st_adapter(h)
        elif self.adapter_type == "frame2freq_ms":
            return self._frame2freq_ms(h, T)

    def forward(self, x, T, H_patches, W_patches, frames_as_batch=False):
        x_id = x  
        h = self.fc1(x)
        Ca = self.adapter_channels

        if frames_as_batch:
            BT = h.shape[0]
            B = BT // T
            T_eff = self.peft_clip_len if (self.decouple and self.peft_clip_len < T) else T
            K = T // T_eff
            h = h.view(B * K, T_eff, H_patches, W_patches, Ca).permute(0, 4, 1, 2, 3).contiguous()
            h = self._forward_bottleneck(h, T_eff)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(BT, H_patches * W_patches, Ca)

        elif self.decouple and self.peft_clip_len > T:
            T_eff = self.peft_clip_len
            K_merge = T_eff // T
            BK, THW, _ = h.shape
            h = h.view(BK // K_merge, K_merge * T, H_patches, W_patches, Ca)
            h = h.permute(0, 4, 1, 2, 3).contiguous()
            h = self._forward_bottleneck(h, T_eff)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(BK, THW, Ca)

        elif self.decouple and self.peft_clip_len < T:
            T_eff = self.peft_clip_len
            K_split = T // T_eff
            B_orig = h.shape[0]
            h = h.view(B_orig * K_split, T_eff, H_patches, W_patches, Ca)
            h = h.permute(0, 4, 1, 2, 3).contiguous()
            h = self._forward_bottleneck(h, T_eff)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(B_orig, T * H_patches * W_patches, Ca)

        else:
            # standard: no decoupling.
            B = h.shape[0]
            h = h.view(B, T, H_patches, W_patches, Ca).permute(0, 4, 1, 2, 3).contiguous()
            h = self._forward_bottleneck(h, T)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H_patches * W_patches, Ca)

        h = self.fc2(h)
        return x_id + h

class ST_Adapter(nn.Module):
    def __init__(self, in_channels, adapter_channels=384, kernel_size=(3, 1, 1),
                 backbone_clip_len=1, peft_clip_len=1):
        super().__init__()
        self.peft_clip_len = peft_clip_len
        self.decouple = (peft_clip_len != backbone_clip_len)
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(k // 2 for k in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, T, H_patches, W_patches, frames_as_batch=False):
        x_id = x  # residual
        h = self.fc1(x)
        Ca = self.conv.in_channels

        if frames_as_batch:
            # legacy, this should not be used
            BT = h.shape[0]
            B = BT // T
            h = h.view(B, T, H_patches, W_patches, Ca).permute(0, 4, 1, 2, 3).contiguous()
            h = self.conv(h)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(BT, H_patches * W_patches, Ca)

        elif self.decouple and self.peft_clip_len > T:
            # B→N merge: outer loop packed K backbone segments into the batch dim.
            # x: (B*K, T*H*W, C) → expose K*T frames to the conv → restore (B*K, T*H*W, C).
            K_merge = self.peft_clip_len // T
            BK, THW, _ = h.shape
            h = h.view(BK // K_merge, K_merge * T, H_patches, W_patches, Ca)
            h = h.permute(0, 4, 1, 2, 3).contiguous()
            h = self.conv(h)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(BK, THW, Ca)

        elif self.decouple and self.peft_clip_len < T:
            # N→B split: divide the T_model sequence into shorter T_peft chunks.
            # x: (B, T*H*W, C) → (B*K_split, T_peft*H*W, C) → restore (B, T*H*W, C).
            K_split = T // self.peft_clip_len
            B_orig = h.shape[0]
            h = h.view(B_orig * K_split, self.peft_clip_len, H_patches, W_patches, Ca)
            h = h.permute(0, 4, 1, 2, 3).contiguous()
            h = self.conv(h)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(B_orig, T * H_patches * W_patches, Ca)

        else:
            # Standard: no decoupling.
            B = h.shape[0]
            h = h.view(B, T, H_patches, W_patches, Ca).permute(0, 4, 1, 2, 3).contiguous()
            h = self.conv(h)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H_patches * W_patches, Ca)

        h = self.fc2(h)
        return x_id + h


class Frame2Freq_MS(nn.Module):
    """Frame2Freq-MS: Multi-Scale Spectral Adapter (Ponbagavathi et al., 2026).
    """

    def __init__(self, in_channels, adapter_channels=384, kernel_size=(3, 1, 1),
                 backbone_clip_len=1, peft_clip_len=1, window_divisors=(1, 2, 4)):
        super().__init__()
        assert adapter_channels % 2 == 0, "adapter_channels must be even (split into freq/temp halves)"
        self.peft_clip_len = peft_clip_len
        self.decouple = (peft_clip_len != backbone_clip_len)
        self.window_divisors = tuple(window_divisors)
        self.adapter_channels = adapter_channels
        half = adapter_channels // 2

        self.fc1 = nn.Linear(in_channels, adapter_channels)

        self.conv_freq = nn.Conv3d(
            half, half, kernel_size=kernel_size, stride=(1, 1, 1),
            padding=tuple(k // 2 for k in kernel_size), groups=half,
        )
        self.conv_temp = nn.Conv3d(
            half, half, kernel_size=kernel_size, stride=(1, 1, 1),
            padding=tuple(k // 2 for k in kernel_size), groups=half,
        )

        self.fc2 = nn.Linear(adapter_channels, in_channels)

        for conv in (self.conv_freq, self.conv_temp):
            nn.init.constant_(conv.weight, 0.)
            nn.init.constant_(conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def _spectral(self, h, T):
        B, Ca, _, Hp, Wp = h.shape
        x_freq, x_temp = torch.split(h, Ca // 2, dim=1)  # (B, Ca/2, T, H, W)
        out_temp = self.conv_temp(x_temp)
        out_freq = torch.zeros_like(x_freq)
        # only use window divisors that evenly divide the current T; d=1 (global window) always
        # divides T, so small temporal contexts (e.g. Tc=1, 2) degrade to fewer scales instead of crashing.
        active_divisors = [d for d in self.window_divisors if T % d == 0]
        for d in active_divisors:
            wk = T // d
            xf = x_freq.view(B, Ca // 2, d, wk, Hp, Wp)

            # rFFT along the window (time) axis 
            spec = torch.fft.rfft(xf, n=wk, dim=3)          # complex
            Ffreq = spec.shape[3]

            spec = spec.permute(0, 2, 1, 3, 4, 5).reshape(B * d, Ca // 2, Ffreq, Hp, Wp)
            spec = torch.complex(self.conv_freq(spec.real), self.conv_freq(spec.imag))
            spec = spec.view(B, d, Ca // 2, Ffreq, Hp, Wp).permute(0, 2, 1, 3, 4, 5)

            # iFFT back to length wk and stitch windows 
            rec = torch.fft.irfft(spec, n=wk, dim=3)        # (B, Ca/2, d, wk, H, W)
            rec = rec.reshape(B, Ca // 2, T, Hp, Wp)
            out_freq = out_freq + rec
        out_freq = out_freq / len(active_divisors)

        return torch.cat([out_freq, out_temp], dim=1)

    def forward(self, x, T, H_patches, W_patches, frames_as_batch=False):
        x_id = x  # residual
        h = self.fc1(x)
        Ca = self.adapter_channels

        if frames_as_batch:
            BT = h.shape[0]
            B = BT // T
            h = h.view(B, T, H_patches, W_patches, Ca).permute(0, 4, 1, 2, 3).contiguous()
            h = self._spectral(h, T)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(BT, H_patches * W_patches, Ca)
        elif self.decouple and self.peft_clip_len > T:
            K_merge = self.peft_clip_len // T
            BK, THW, _ = h.shape
            T_eff = K_merge * T
            h = h.view(BK // K_merge, T_eff, H_patches, W_patches, Ca)
            h = h.permute(0, 4, 1, 2, 3).contiguous()
            h = self._spectral(h, T_eff)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(BK, THW, Ca)
        elif self.decouple and self.peft_clip_len < T:
            K_split = T // self.peft_clip_len
            B_orig = h.shape[0]
            T_eff = self.peft_clip_len
            h = h.view(B_orig * K_split, T_eff, H_patches, W_patches, Ca)
            h = h.permute(0, 4, 1, 2, 3).contiguous()
            h = self._spectral(h, T_eff)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(B_orig, T * H_patches * W_patches, Ca)
        else:
            B = h.shape[0]
            h = h.view(B, T, H_patches, W_patches, Ca).permute(0, 4, 1, 2, 3).contiguous()
            h = self._spectral(h, T)
            h = h.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H_patches * W_patches, Ca)

        h = self.fc2(h)
        return x_id + h