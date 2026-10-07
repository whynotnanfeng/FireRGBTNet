# -*- coding: utf-8 -*-
"""Feature Pyramid Neck of FireRGBTNet.

Section 2.2.1 ("Neck") of the paper. The neck performs top-down FPN
propagation over the MSGF-fused dual-stream features. At every level the RGB and
thermal streams are fused by MSGF, then upsampled and concatenated into the next
finer level, so the fused features accumulate both global semantics and local
spatial detail before reaching the detection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import MSGF

__all__ = ["FusionNeck"]


class FusionNeck(nn.Module):
    """Top-down dual-stream FPN neck built from MSGF blocks.

    Args:
        in_features: input channels of the ``(C3, C4, C5)`` backbone triple.
        out_features: per-level output channels after MSGF.
        scales: receptive-field kernel sizes forwarded to :class:`MSGF`, where
            ``scales[0]`` is the identity segment and the remainder define the
            multi-scale depthwise context branches of SGCA.
    """

    def __init__(self, in_features=(128, 128, 256), out_features=(64, 64, 64),
                 scales=(1, 3, 5, 7)):
        super().__init__()
        c3_in, c4_in, c5_in = in_features
        c3_out, c4_out, c5_out = out_features

        # Level 5 (top): fuse the deepest features.
        self.fuse5 = MSGF(rgb_in=c5_in, ir_in=c5_in, out_channels=c5_out, scales=scales)
        # Level 4 (mid): fuse C4 with the upsampled P5 features.
        self.fuse4 = MSGF(
            rgb_in=c4_in + c5_out, ir_in=c4_in + c5_out, out_channels=c4_out, scales=scales
        )
        # Level 3 (bottom): fuse C3 with the upsampled P4 features.
        self.fuse3 = MSGF(
            rgb_in=c3_in + c4_out, ir_in=c3_in + c4_out, out_channels=c3_out, scales=scales
        )

    def forward(self, rgb_feats, thermal_feats):
        """Args:
            rgb_feats: list/tuple ``(C3, C4, C5)`` RGB features.
            thermal_feats: list/tuple ``(C3, C4, C5)`` thermal features.

        Returns:
            List of three tensors ``[P3, P4, P5]``, each being the channel-wise
            concatenation of the fused RGB and thermal streams.
        """
        r3, r4, r5 = rgb_feats
        i3, i4, i5 = thermal_feats

        r5_f, i5_f = self.fuse5(r5, i5)

        r5_up = F.interpolate(r5_f, scale_factor=2, mode="nearest")
        i5_up = F.interpolate(i5_f, scale_factor=2, mode="nearest")

        r4_f, i4_f = self.fuse4(
            torch.cat([r4, r5_up], dim=1), torch.cat([i4, i5_up], dim=1)
        )

        r4_up = F.interpolate(r4_f, scale_factor=2, mode="nearest")
        i4_up = F.interpolate(i4_f, scale_factor=2, mode="nearest")

        r3_f, i3_f = self.fuse3(
            torch.cat([r3, r4_up], dim=1), torch.cat([i3, i4_up], dim=1)
        )

        return [
            torch.cat([r3_f, i3_f], dim=1),
            torch.cat([r4_f, i4_f], dim=1),
            torch.cat([r5_f, i5_f], dim=1),
        ]
