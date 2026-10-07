# -*- coding: utf-8 -*-
"""FireRGBTNet: a lightweight RGB-T forest fire detection model.

Reference implementation of the model published as:

    FireRGBTNet: A Lightweight Forest Fire Detection Model Based on
    Efficient RGB-Thermal Fusion
    Ma, Y.; Shan, W.; Wang, M.; Sui, Y.; Wang, M.
    Forests 2026, 17, 955. https://doi.org/10.3390/f17080955
"""

from .boxes import box_cxcywh_to_xyxy, box_iou, box_xyxy_to_cxcywh, generalized_box_iou
from .criterion import HungarianMatcher, SetCriterion, normalized_wasserstein_similarity
from .metrics import AdvancedDetMetrics, BoxF1Score
from .model import FireRGBTNet

__all__ = [
    "FireRGBTNet",
    "HungarianMatcher",
    "SetCriterion",
    "normalized_wasserstein_similarity",
    "AdvancedDetMetrics",
    "BoxF1Score",
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "box_iou",
    "generalized_box_iou",
    "__version__",
]

__version__ = "1.0.0"
