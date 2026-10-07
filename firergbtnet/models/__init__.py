# -*- coding: utf-8 -*-
"""Network components of FireRGBTNet."""

from .alignment import MSAE, SEBlock, SemanticProjectionHead
from .backbone import (
    TED,
    BottleneckUnit,
    HeterogeneousDualStreamBackbone,
    RGBBlock,
    RGBUnit,
    ThermalBlock,
    ThermalUnit,
)
from .basic import Conv, autopad, make_divisible
from .fusion import MSGF, SGCA, LayerNorm2d, MishGLU
from .head import RTDETRDecoder
from .neck import FusionNeck

__all__ = [
    # Basic building block
    "Conv",
    "autopad",
    "make_divisible",
    # Heterogeneous dual-stream backbone
    "HeterogeneousDualStreamBackbone",
    "RGBBlock",
    "ThermalBlock",
    "RGBUnit",
    "ThermalUnit",
    "BottleneckUnit",
    "TED",
    # Multi-scale semantic alignment enhancement
    "MSAE",
    "SemanticProjectionHead",
    "SEBlock",
    # Multimodal spatial gated fusion and the neck that hosts it
    "MSGF",
    "SGCA",
    "MishGLU",
    "LayerNorm2d",
    "FusionNeck",
    # Detection head
    "RTDETRDecoder",
]
