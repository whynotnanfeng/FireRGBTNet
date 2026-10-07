# -*- coding: utf-8 -*-
"""Multimodal Spatial Gated Fusion (MSGF) module.

Sec. 2.2.4 of the paper. MSGF couples two components:

* :class:`SGCA`  -- Spatial Gated Cross-Attention (Eqs. 19-25). Replaces the
  global attention used by prior work with multi-scale *local* receptive fields
  plus a pixel-level cross-modal gate, so tiny fire spots are neither diluted by
  the vast forest background nor amplified by cross-modal noise.
* :class:`MishGLU` -- Mish Gated Linear Unit (Eqs. 26-28). A large-kernel 9x9
  depthwise convolution supplies the global context that SGCA lacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import Conv

__all__ = ["SGCA", "MishGLU", "LayerNorm2d", "MSGF"]


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for ``[B, C, H, W]`` tensors."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class SGCA(nn.Module):
    """Spatial Gated Cross-Attention.

    Channels are split into one identity segment and three context segments
    (Eq. 19). The context segments go through depthwise convolutions with kernel
    sizes 3/5/7 and are concatenated (Eq. 20). A grouped 1x1 convolution then
    produces grouped Q/K/V (Eq. 21).

    Instead of a spatial correlation matrix, the cross-modal gate is computed
    pixel-wise along the *scale* dimension (Eqs. 22-24)::

        A_rgb = softmax(Q_rgb * K_th + Q_rgb + K_th)   # over the 3 scales
        delta_rgb = A_rgb * V_th

    The multiplicative term activates both modalities jointly at a location while
    the additive bias preserves each modality's independent confidence. Softmax
    over scales makes the three receptive fields compete at every pixel. The
    gated features are projected back and added residually to the context
    features, then concatenated with the untouched identity segment (Eq. 25).

    Args:
        dim: channel count of each modality.
        scales: kernel sizes. ``scales[0]`` is the identity segment and is not
            convolved; the remainder define the multi-scale context branches.
    """

    def __init__(self, dim, scales=(1, 3, 5, 7)):
        super().__init__()
        self.scales = tuple(scales)
        self.dim = dim
        self.num_scales = len(scales)

        base = dim // self.num_scales
        rem = dim % self.num_scales
        self.split_sizes = [base] * (self.num_scales - 1) + [base + rem]

        self.dim_identity = self.split_sizes[0]
        self.dim_context = dim - self.dim_identity
        self.num_context_scales = self.num_scales - 1

        self.rgb_dws = nn.ModuleList()
        self.thermal_dws = nn.ModuleList()
        for ch, k in zip(self.split_sizes[1:], self.scales[1:]):
            self.rgb_dws.append(nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=True))
            self.thermal_dws.append(nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=True))

        # Grouped 1x1 conv producing Q/K/V for the context segments.
        self.rgb_qkv = nn.Conv2d(
            self.dim_context, self.dim_context * 3, 1, groups=self.num_context_scales
        )
        self.thermal_qkv = nn.Conv2d(
            self.dim_context, self.dim_context * 3, 1, groups=self.num_context_scales
        )

        self.out_proj_rgb = nn.Conv2d(
            self.dim_context, self.dim_context, 1, groups=self.num_context_scales
        )
        self.out_proj_thermal = nn.Conv2d(
            self.dim_context, self.dim_context, 1, groups=self.num_context_scales
        )

    def forward(self, rgb, thermal):
        # Split off the identity segment, keep the rest as context.
        rgb_id, rgb_ctx = torch.split(rgb, [self.dim_identity, self.dim_context], dim=1)
        th_id, th_ctx = torch.split(thermal, [self.dim_identity, self.dim_context], dim=1)

        rgb_splits = torch.split(rgb_ctx, self.split_sizes[1:], dim=1)
        th_splits = torch.split(th_ctx, self.split_sizes[1:], dim=1)

        rgb_multi = torch.cat(
            [dw(r) for r, dw in zip(rgb_splits, self.rgb_dws)], dim=1
        )
        th_multi = torch.cat(
            [dw(t) for t, dw in zip(th_splits, self.thermal_dws)], dim=1
        )

        b, _, h, w = rgb_multi.shape

        # Grouped QKV -> (B, num_scales, 3, C_per_scale, H, W), then unbind Q/K/V.
        qkv_rgb = self.rgb_qkv(rgb_multi).view(
            b, self.num_context_scales, 3, -1, h, w
        ).contiguous()
        qkv_th = self.thermal_qkv(th_multi).view(
            b, self.num_context_scales, 3, -1, h, w
        ).contiguous()

        q_rgb, k_rgb, v_rgb = qkv_rgb.unbind(dim=2)
        q_th, k_th, v_th = qkv_th.unbind(dim=2)

        # Cross-modal pixel-level gate, softmax over the scale dimension.
        attn_rgb = F.softmax(q_rgb * k_th + q_rgb + k_th, dim=1)
        delta_rgb = attn_rgb * v_th

        attn_th = F.softmax(q_th * k_rgb + q_th + k_rgb, dim=1)
        delta_th = attn_th * v_rgb

        delta_rgb = self.out_proj_rgb(delta_rgb.view(b, -1, h, w))
        delta_th = self.out_proj_thermal(delta_th.view(b, -1, h, w))

        rgb_out = torch.cat([rgb_id, rgb_ctx + delta_rgb], dim=1)
        th_out = torch.cat([th_id, th_ctx + delta_th], dim=1)
        return rgb_out, th_out


class MishGLU(nn.Module):
    """Mish Gated Linear Unit (Eqs. 26-28).

    A 1x1 convolution expands the channels and splits them into a gating stream
    and an information stream. The gating stream applies a 9x9 depthwise
    convolution with a local residual connection followed by Mish activation
    (Eq. 27); the result gates the information stream (Eq. 28).

    Args:
        features: input/output channel count.
        hidden: hidden channel count before the ``2 * hidden / 3`` adjustment.
        dw_kernel: depthwise kernel size in the gating stream. The paper's
            MishGLU (Eq. 27) uses 9x9 inside MSGF; the RT-DETR decoder's FFN
            uses 3x3 in the released configuration.
    """

    def __init__(self, features, hidden=None, dw_kernel=9):
        super().__init__()
        out_features = features
        hidden = int(2 * hidden / 3)

        self.fc1 = nn.Conv2d(features, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden, hidden, kernel_size=dw_kernel, stride=1,
            padding=dw_kernel // 2, bias=True, groups=hidden,
        )
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden, out_features, kernel_size=1)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        return self.fc2(x)


class MSGF(nn.Module):
    """Multimodal Spatial Gated Fusion block.

    A post-norm transformer-style block: ``SGCA`` -> ``LayerNorm`` ->
    ``MishGLU`` -> ``LayerNorm`` -> 1x1 channel projection. The two modalities
    stay decoupled until the neck concatenates them, which keeps the fusion
    parameter-efficient.

    Args:
        rgb_in: input channels of the RGB stream.
        ir_in: input channels of the thermal stream.
        out_channels: output channels per stream. Defaults to ``rgb_in``.
        scales: receptive-field kernel sizes handed to :class:`SGCA`.
    """

    def __init__(self, rgb_in, ir_in, out_channels=None, scales=(1, 3, 5, 7)):
        super().__init__()
        out_channels = out_channels or rgb_in

        self.attn = SGCA(rgb_in, scales=scales)

        self.norm_rgb_attn = LayerNorm2d(rgb_in)
        self.norm_ir_attn = LayerNorm2d(ir_in)

        self.ffn_rgb = MishGLU(rgb_in, int(rgb_in * 2))
        self.ffn_ir = MishGLU(ir_in, int(ir_in * 2))

        self.norm_rgb_ffn = LayerNorm2d(rgb_in)
        self.norm_ir_ffn = LayerNorm2d(ir_in)

        self.rgb_reduce = Conv(rgb_in, out_channels)
        self.ir_reduce = Conv(ir_in, out_channels)

    def forward(self, rgb, ir):
        r_attn, i_attn = self.attn(rgb, ir)

        r_mid = self.norm_rgb_attn(r_attn)
        i_mid = self.norm_ir_attn(i_attn)

        r_out = self.norm_rgb_ffn(r_mid + self.ffn_rgb(r_mid))
        i_out = self.norm_ir_ffn(i_mid + self.ffn_ir(i_mid))

        return self.rgb_reduce(r_out), self.ir_reduce(i_out)
