# -*- coding: utf-8 -*-
"""Basic convolutional building blocks shared across FireRGBTNet.

The ``Conv`` wrapper (conv -> BN -> SiLU) is the fundamental building unit used
throughout the heterogeneous dual-stream backbone and the fusion neck.
"""

import math

import torch
import torch.nn as nn

__all__ = ["Conv", "autopad", "make_divisible"]


def autopad(k, p=None, d=1):
    """Pad ``k``x``k`` convolution to obtain 'same' output shape."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x, divisor=8):
    """Round ``x`` up to the nearest multiple of ``divisor``."""
    return int(math.ceil(x / divisor)) * divisor


class Conv(nn.Module):
    """Standard ``Conv2d`` + ``BatchNorm2d`` (+ optional SiLU) unit."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    # ------------------------------------------------------------------
    # Inference-time Conv+BN fusion helpers
    # ------------------------------------------------------------------
    def fuse_forward(self, x):
        """Forward pass used after :meth:`fuse` (BatchNorm has been folded in)."""
        return self.act(self.conv(x))

    def fuse(self):
        """Fold BatchNorm statistics into the convolution weights in-place."""
        fused_conv = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True,
        ).requires_grad_(False).to(self.conv.weight.device)

        w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
        b_conv = (
            torch.zeros(self.conv.weight.size(0), device=self.conv.weight.device)
            if self.conv.bias is None
            else self.conv.bias
        )

        w_bn = torch.diag(self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var)))
        b_bn = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(
            torch.sqrt(self.bn.running_var + self.bn.eps)
        )

        fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))
        fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        self.conv = fused_conv
        self.bn = nn.Identity()
        self.forward = self.fuse_forward
