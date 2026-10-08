# -*- coding: utf-8 -*-
"""Heterogeneous dual-stream backbone of FireRGBTNet.

Implements the three contributions described in Sec. 2.2.2 of the paper:

* :class:`RGBBlock`      -- ELAN-style aggregation with standard convolutions
                            (dense high-frequency visible-light textures).
* :class:`ThermalBlock`  -- ELAN-style aggregation with large-kernel depthwise
                            convolutions (sparse low-frequency thermal fields).
* :class:`TED`           -- Target-Enhanced Downsampling with Adaptive Weighted
                            Fusion, preserving fragile small-target responses.
"""

import torch
import torch.nn as nn

from .basic import Conv, make_divisible

__all__ = [
    "RGBUnit",
    "ThermalUnit",
    "RGBBlock",
    "ThermalBlock",
    "TED",
    "HeterogeneousDualStreamBackbone",
]


# ---------------------------------------------------------------------------
# Basic units  (Phi_RGB / Phi_Thermal in the paper)
# ---------------------------------------------------------------------------
class RGBUnit(nn.Module):
    """Bottleneck RGBUnit ``Phi_RGB``: two 3x3 standard convs, C -> C/2 -> C."""

    def __init__(self, ch_in, ch_out, e=0.5, k=3):
        super().__init__()
        c = int(ch_out * e)
        self.cv1 = Conv(ch_in, c, k, 1)
        self.cv2 = Conv(c, ch_out, k, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class ThermalUnit(nn.Module):
    """Bottleneck ThermalUnit ``Phi_Thermal``: 1x1 -> large-kernel DW -> 1x1."""

    def __init__(self, ch_in, ch_out, e=0.5, k=7):
        super().__init__()
        c = int(ch_out * e)
        self.cv1 = Conv(ch_in, c, 3, 1)
        self.cv2 = Conv(c, c, k, 1, g=c)
        self.cv3 = Conv(c, ch_out, 1, 1)

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x)))


class BottleneckUnit(nn.Module):
    """Modality-agnostic 1x1 -> kxk residual unit.

    Used by the neck and by the projection head, where neither the visible-light
    nor the thermal prior applies. Follows the same bottleneck layout as
    :class:`RGBUnit` but with a 1x1 channel-expanding first convolution.
    """

    def __init__(self, ch_in, ch_out, e=0.5, k=3):
        super().__init__()
        c = int(ch_out * e)
        self.cv1 = Conv(ch_in, c, 1, 1)
        self.cv2 = Conv(c, ch_out, k, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


# ---------------------------------------------------------------------------
# ELAN-style aggregation blocks  (RGBBlock / ThermalBlock, Eqs. 1-7)
# ---------------------------------------------------------------------------
class _ELANBlock(nn.Module):
    """Three-path ELAN aggregation: identity + 1x1 + two cascaded units.

    ``mode='rgb'`` selects :class:`RGBUnit`, ``mode='thermal'`` selects
    :class:`ThermalUnit`; ``enhance=True`` widens the aggregation to the
    two-branch variant used at the deeper C4/C5 stages.
    """

    def __init__(self, ch_in, ch_out, e=0.25, k=3, mode="common", enhance=False):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.c = int(ch_out * e)
        self.mode = mode
        self.enhance = enhance

        self.conv1 = Conv(ch_in, self.c * 2, k=1)
        self.conv2 = Conv(self.c * 3, ch_out, k=1)

        unit = self._make_unit(k)
        if enhance:
            self.branch = _ELANEnhance(self.c, k, mode)
        else:
            self.branch = unit

    def _make_unit(self, k):
        if self.mode == "rgb":
            return RGBUnit(self.c, self.c, k=k)
        if self.mode == "thermal":
            return ThermalUnit(self.c, self.c, k=k)
        return BottleneckUnit(self.c, self.c, k=k)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.split((self.c, self.c), dim=1)
        x3 = self.branch(x2)
        return self.conv2(torch.cat((x1, x2, x3), dim=1))


class _ELANEnhance(nn.Module):
    """Two-branch aggregation used at C4/C5 (parallel conv + unit chain)."""

    def __init__(self, c, k, mode):
        super().__init__()
        self.conv1 = Conv(c, c, k=1)
        self.conv2 = Conv(c, c, k=1)
        self.conv3 = Conv(c * 2, c, k=1)
        unit_cls = {"rgb": RGBUnit, "thermal": ThermalUnit}.get(mode, BottleneckUnit)
        self.b1 = unit_cls(c, c, k=k)
        self.b2 = unit_cls(c, c, k=k)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.b1(self.b2(self.conv2(x)))
        return self.conv3(torch.cat((x1, x2), dim=1))


class RGBBlock(_ELANBlock):
    """RGBBlock: ELAN aggregation built from standard convolutions.

    Args:
        ch_in: input channels.
        ch_out: output channels.
        e: expansion ratio of the hidden branch.
        enhance: use the deeper two-branch aggregation variant.
    """

    def __init__(self, ch_in, ch_out, e=0.25, enhance=False, **_):
        super().__init__(ch_in, ch_out, e=e, k=3, mode="rgb", enhance=enhance)


class ThermalBlock(_ELANBlock):
    """ThermalBlock: ELAN aggregation built from large-kernel DW convolutions.

    Args:
        ch_in: input channels.
        ch_out: output channels.
        e: expansion ratio of the hidden branch.
        k: large depthwise kernel size (5/7 in the released configuration).
        enhance: use the deeper two-branch aggregation variant.
    """

    def __init__(self, ch_in, ch_out, e=0.25, k=5, enhance=False, **_):
        super().__init__(ch_in, ch_out, e=e, k=k, mode="thermal", enhance=enhance)


# ---------------------------------------------------------------------------
# Target-Enhanced Downsampling  (TED, Eqs. 8-11)
# ---------------------------------------------------------------------------
class TED(nn.Module):
    """Target-Enhanced Downsampling with Adaptive Weighted Fusion (AWF).

    Branch 1: 2x2 max-pool followed by a 1x1 projection, which preserves the
    local extrema of tiny high-temperature / high-frequency responses.
    Branch 2: a 3x3 stride-2 convolution capturing the surrounding context.

    The two branches are recombined with learnable, normalised weights
    (Eqs. 10-11) instead of a plain concatenation, so the network can adapt
    the extremal-vs-local ratio to different fire scenarios.

    Args:
        ch_in: input channels.
        ch_out: output channels (split across the two branches).
    """

    def __init__(self, ch_in, ch_out):
        super().__init__()
        ch_out = make_divisible(ch_out, 8)

        c1 = ch_out // 2
        c2 = ch_out - c1

        self.branch_maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(ch_in, c1, k=1, s=1),
        )
        self.branch_conv = Conv(ch_in, c2, k=3, s=2)

        # Adaptive Weighted Fusion: w holds the branch importance weights and
        # v the learnable global scale (Eq. 10).
        #
        # w must start above zero. The normalisation divides by sum(w), so an
        # all-zero initialisation would drive every alpha to zero and zero out
        # the module output entirely. Starting from ones gives each branch an
        # equal weight of about 0.5, which keeps the signal alive and leaves the
        # network free to rebalance the two paths during training.
        self.weight = nn.Parameter(torch.ones(2))
        self.scale = nn.Parameter(torch.ones(2))
        self.eps = 1e-4

    def forward(self, x):
        x_pool = self.branch_maxpool(x)
        x_conv = self.branch_conv(x)

        # alpha_i = v_i * w_i / sum_j(w_j) + eps   (Eq. 10)
        w = self.weight
        alpha = (self.scale * w / (w.sum() + self.eps)).view(1, -1, 1, 1)

        # Y = Concat([alpha_1 * X_pool, alpha_2 * X_conv])   (Eq. 11)
        return torch.cat([alpha[:, 0] * x_pool, alpha[:, 1] * x_conv], dim=1)


# ---------------------------------------------------------------------------
# Full heterogeneous dual-stream backbone
# ---------------------------------------------------------------------------
class _SingleStreamBackbone(nn.Module):
    """Shared skeleton for the RGB and thermal branches.

    Target-Enhanced Downsampling replaces the strided convolution at the first
    two downsampling stages (stride 4 and stride 8); the deeper stages keep a
    plain 3x3 stride-2 convolution.
    """

    def __init__(self, block_cls, block_kwargs):
        super().__init__()
        self.stem = Conv(3, 16, k=3, s=2)

        # C2 (stride 4)
        self.layer2 = TED(16, 32)
        self.layer2_block = block_cls(32, 64, **block_kwargs["c2"])

        # C3 (stride 8)
        self.layer3_conv = TED(64, 64)
        self.layer3_block = block_cls(64, 128, **block_kwargs["c3"])

        # C4 (stride 16)
        self.layer4_conv = Conv(128, 128, k=3, s=2)
        self.layer4_block = block_cls(128, 128, enhance=True, **block_kwargs["c4"])

        # C5 (stride 32)
        self.layer5_conv = Conv(128, 256, k=3, s=2)
        self.layer5_block = block_cls(256, 256, enhance=True, **block_kwargs["c5"])

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        c2 = self.layer2_block(x)

        x = self.layer3_conv(c2)
        c3 = self.layer3_block(x)

        x = self.layer4_conv(c3)
        c4 = self.layer4_block(x)

        x = self.layer5_conv(c4)
        c5 = self.layer5_block(x)
        return c2, c3, c4, c5


class HeterogeneousDualStreamBackbone(nn.Module):
    """Heterogeneous dual-stream backbone: an RGB branch and a Thermal branch.

    The two branches deliberately differ. The RGB stream is built from
    :class:`RGBBlock`, which uses standard convolutions to preserve the dense,
    high-frequency texture of visible-light imagery. The thermal stream is built
    from :class:`ThermalBlock`, which uses large-kernel depthwise convolutions
    to widen the receptive field over sparse, low-frequency thermal backgrounds.
    Both branches use :class:`TED` at the first two downsampling stages.

    Returns the ``(C3, C4, C5)`` feature triple for both modalities. The C2
    (stride-4) stage is computed but dropped by the neck, which operates on P3-P5.
    """

    def __init__(self):
        super().__init__()

        rgb_kwargs = {
            "c2": dict(e=0.25),
            "c3": dict(e=0.25),
            "c4": dict(e=0.5),
            "c5": dict(e=0.5),
        }
        thermal_kwargs = {
            "c2": dict(e=0.25, k=5),
            "c3": dict(e=0.25, k=5),
            "c4": dict(e=0.5, k=5),
            "c5": dict(e=0.5, k=7),
        }
        self.rgb_backbone = _SingleStreamBackbone(RGBBlock, rgb_kwargs)
        self.thermal_backbone = _SingleStreamBackbone(ThermalBlock, thermal_kwargs)

    def forward(self, rgb, thermal):
        rgb_feats = self.rgb_backbone(rgb)[1:]  # drop C2 -> (C3, C4, C5)
        thermal_feats = self.thermal_backbone(thermal)[1:]
        return rgb_feats, thermal_feats
