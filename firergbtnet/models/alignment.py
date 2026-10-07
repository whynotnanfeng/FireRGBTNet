# -*- coding: utf-8 -*-
"""Multi-Scale Semantic Alignment Enhancement (MSAE) branch.

Sec. 2.2.3 of the paper. MSAE is an **auxiliary supervision branch**: it is
active during training only and can be dropped entirely at inference time,
hence zero deployment overhead.

Each selected backbone level is projected into a shared latent semantic space by
a :class:`SemanticProjectionHead` (Eqs. 12-15), then the two modalities are
aligned with the dual constraints

* Channel-level Wasserstein Distribution Alignment (CWA), Eq. 16 -- matches the
  per-channel first-order (mean) and second-order (std) statistics, and
* cosine loss, Eq. 17 -- matches the semantic direction along channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BottleneckUnit
from .basic import Conv

__all__ = ["SEBlock", "SemanticProjectionHead", "MSAE"]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel modulation (Eq. 15)."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channel = max(channel // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SemanticProjectionHead(nn.Module):
    """Project one modality's features into the shared latent semantic space.

    A 1x1 convolution doubles the channels and splits them into an identity
    branch ``X1`` and a projection branch ``X2`` (Eq. 12). ``X2`` passes through
    two cascaded units (Eqs. 13-14); all four tensors are concatenated, fused by
    a 1x1 convolution and modulated by SE (Eq. 15).

    Args:
        channels: input (and output) channel count of this level.
        modality: ``'rgb'`` or ``'thermal'``; kept for interface symmetry with the
            released single-modality baselines.
    """

    def __init__(self, channels, e=0.5, k=3, modality="rgb"):
        super().__init__()
        self.modality = modality
        self.c = int(channels * e)

        self.conv1 = Conv(channels, self.c * 2, k=1)
        self.conv2 = Conv(self.c * 4, channels, k=1)

        self.proj1 = BottleneckUnit(self.c, self.c, e=e, k=k)
        self.proj2 = BottleneckUnit(self.c, self.c, e=e, k=k)
        self.se = SEBlock(channels, reduction=16)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.split((self.c, self.c), dim=1)
        x3 = self.proj1(x2)
        x4 = self.proj2(x3)
        return self.se(self.conv2(torch.cat((x1, x2, x3, x4), dim=1)))


class MSAE(nn.Module):
    """Multi-Scale Semantic Alignment Enhancement branch (Eqs. 16-18).

    Args:
        channels_list: channel count of every backbone level ``(C3, C4, C5)``.
        active_indices: indices of ``channels_list`` that are aligned. The paper
            ablates all 2^3 combinations of this set (Table 8); the released
            configuration aligns all three scales.
        cwa_weight: weight ``w_cwa`` of the distribution-alignment term.
        cos_weight: weight ``w_cos`` of the cosine term.
    """

    def __init__(
        self,
        channels_list=(128, 128, 256),
        active_indices=(0, 1, 2),
        cwa_weight=0.5,
        cos_weight=0.5,
    ):
        super().__init__()
        self.active_indices = tuple(active_indices)
        self.cwa_weight = cwa_weight
        self.cos_weight = cos_weight

        # Only instantiate projectors for the selected levels to save parameters.
        self.rgb_projectors = nn.ModuleList(
            [SemanticProjectionHead(channels_list[i], modality="rgb") for i in self.active_indices]
        )
        self.thermal_projectors = nn.ModuleList(
            [
                SemanticProjectionHead(channels_list[i], modality="thermal")
                for i in self.active_indices
            ]
        )

    @staticmethod
    def channel_wasserstein_alignment(z_rgb, z_thermal, eps=1e-5):
        """CWA loss (Eq. 16): MSE on per-channel mean plus per-channel std.

        Equivalent to a channel-level Normalized Wasserstein Distance between the
        marginal distributions of the two modalities.
        """
        x = z_rgb.flatten(2)
        y = z_thermal.flatten(2)

        mean_x, mean_y = x.mean(dim=2), y.mean(dim=2)
        std_x = x.var(dim=2).clamp(min=eps).sqrt()
        std_y = y.var(dim=2).clamp(min=eps).sqrt()

        return F.mse_loss(mean_x, mean_y) + F.mse_loss(std_x, std_y)

    def forward(self, rgb_feats, thermal_feats):
        """Return the scalar multi-scale alignment loss (Eq. 18).

        Note:
            This branch is only invoked in training mode. At inference the whole
            module can be discarded, contributing zero FLOPs.
        """
        loss_align = rgb_feats[0].new_zeros(())
        if not self.training:
            return loss_align

        for slot, level in enumerate(self.active_indices):
            proj_rgb = self.rgb_projectors[slot](rgb_feats[level])
            proj_thermal = self.thermal_projectors[slot](thermal_feats[level])

            l_cwa = self.channel_wasserstein_alignment(proj_rgb, proj_thermal)
            l_cos = 1.0 - F.cosine_similarity(proj_rgb, proj_thermal, dim=1).mean()

            loss_align = loss_align + self.cwa_weight * l_cwa + self.cos_weight * l_cos

        return loss_align
