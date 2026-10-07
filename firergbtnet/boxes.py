# -*- coding: utf-8 -*-
"""Bounding-box conversion and IoU utilities."""

import torch
from torchvision.ops import box_area

__all__ = [
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "box_iou",
    "generalized_box_iou",
]


def box_cxcywh_to_xyxy(x):
    """Convert ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)``."""
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1
    )


def box_xyxy_to_cxcywh(x):
    """Convert ``(x1, y1, x2, y2)`` to ``(cx, cy, w, h)``."""
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def box_iou(boxes1, boxes2):
    """Pairwise IoU between two sets of ``xyxy`` boxes. Returns ``(iou, union)``."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/ for ``xyxy`` boxes."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
