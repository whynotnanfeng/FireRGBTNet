# -*- coding: utf-8 -*-
"""Set-prediction loss for FireRGBTNet (Sec. 2.2.5).

The joint objective combines

* Hungarian bipartite matching with an **NWD** auxiliary cost, which is far more
  tolerant of marginal position deviations on tiny blurry thermal targets than
  L1 / IoU costs alone (Eqs. 29-30);
* an IoU-aware binary cross-entropy branch that supervises a predicted quality
  score, later fused into the classification confidence at inference;
* deep supervision over the decoder layers and the encoder query-selection head;
* the MSAE alignment loss (see :mod:`firergbtnet.models.alignment`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss

from .boxes import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

__all__ = ["normalized_wasserstein_similarity", "HungarianMatcher", "SetCriterion"]


def normalized_wasserstein_similarity(box1, box2, eps=1e-7, scale=12.8):
    """Normalized Wasserstein Distance similarity between two sets of boxes.

    Models each box as a 2-D Gaussian ``N(cx, cy, w/2, h/2)`` and computes

    ``W2^2 = (dx)^2 + (dy)^2 + (dw/2)^2 + (dh/2)^2``  (Eq. 29)

    then maps it through ``exp(-sqrt(W2^2) / C)`` so that higher means more similar.

    Args:
        box1: predicted boxes, ``[N, 4]`` in ``(cx, cy, w, h)`` normalized format.
        box2: ground-truth boxes, ``[M, 4]``, same format.

    Returns:
        ``[N, M]`` similarity matrix.
    """
    b1_cx, b1_cy, b1_w, b1_h = box1.unbind(-1)
    b2_cx, b2_cy, b2_w, b2_h = box2.unbind(-1)

    b1_cx, b1_cy, b1_w, b1_h = (t.unsqueeze(1) for t in (b1_cx, b1_cy, b1_w, b1_h))
    b2_cx, b2_cy, b2_w, b2_h = (t.unsqueeze(0) for t in (b2_cx, b2_cy, b2_w, b2_h))

    w2_sq = (
        (b1_cx - b2_cx).pow(2)
        + (b1_cy - b2_cy).pow(2)
        + ((b1_w - b2_w) / 2.0).pow(2)
        + ((b1_h - b2_h) / 2.0).pow(2)
    )
    return torch.exp(-torch.sqrt(w2_sq + eps) / scale)


class HungarianMatcher(nn.Module):
    """Bipartite matcher producing the optimal query-to-ground-truth assignment.

    Args:
        cost_class: weight of the classification cost.
        cost_bbox: weight of the L1 box cost.
        cost_giou: weight of the GIoU box cost.
        cost_nwd: weight of the auxiliary Normalized Wasserstein Distance cost.
        use_focal_loss: use focal-loss formulation for the class cost.
        change_matcher: switch to a high-IoU-prioritised cost in late epochs.
        matcher_change_epoch: epoch at which the switch happens.
        iou_order_alpha: exponent of the IoU term in the late-stage cost.
    """

    def __init__(
        self,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=3.0,
        cost_nwd=2.0,
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
        change_matcher=False,
        matcher_change_epoch=100,
        iou_order_alpha=4.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_nwd = cost_nwd
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.change_matcher = change_matcher
        self.matcher_change_epoch = matcher_change_epoch
        self.iou_order_alpha = iou_order_alpha

    @torch.no_grad()
    @torch.compiler.disable
    def forward(self, outputs, targets, epoch=0):
        """Return a list of ``(query_idx, target_idx)`` index tensors per image."""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = outputs["pred_logits"][..., :-1].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            return [
                (torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64))
                for _ in range(bs)
            ]

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # Late stage: trust only high-quality IoU matches.
            class_score = out_prob[:, tgt_ids]
            iou_matrix, _ = box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )
            cost_matrix = -(class_score * torch.pow(iou_matrix, self.iou_order_alpha))
        else:
            if self.use_focal_loss:
                out_prob_cls = out_prob[:, tgt_ids]
                neg_cost_class = (
                    (1 - self.alpha) * (out_prob_cls**self.gamma) * (-(1 - out_prob_cls + 1e-8).log())
                )
                pos_cost_class = (
                    self.alpha * ((1 - out_prob_cls) ** self.gamma) * (-(out_prob_cls + 1e-8).log())
                )
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )
            cost_nwd = 1.0 - normalized_wasserstein_similarity(out_bbox, tgt_bbox)

            cost_matrix = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
                + self.cost_nwd * cost_nwd
            )

        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()
        cost_matrix = torch.nan_to_num(cost_matrix, nan=100.0, posinf=100.0, neginf=-100.0)
        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))
        ]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """DETR-style set criterion with IoU-aware supervision and deep supervision.

    Args:
        num_classes: number of foreground classes.
        matcher: a :class:`HungarianMatcher`.
        weight_dict: per-loss weights. Keys suffixed with ``_<i>`` (auxiliary
            decoder layers) and ``_enc`` (encoder head) are auto-generated.
        alpha: focal-loss alpha.
        gamma: focal-loss gamma.
        eos_coef: relative weight of the background/no-object class (CE path only).
    """

    def __init__(self, num_classes, matcher, weight_dict, alpha=0.25, gamma=2.0, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.eos_coef = eos_coef

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_src_permutation_idx(indices):
        """Flatten the per-image matched pairs into a single (batch, query) index."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Focal-loss classification loss (or CE when ``use_focal_loss=False``)."""
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.matcher.use_focal_loss:
            src_logits_fg = src_logits[..., :-1]
            target_onehot = torch.zeros_like(src_logits_fg)
            target_onehot[idx[0], idx[1], target_classes_o] = 1.0
            loss_ce = sigmoid_focal_loss(
                src_logits_fg, target_onehot, alpha=self.alpha, gamma=self.gamma, reduction="none"
            )
            loss_ce = loss_ce.sum() / num_boxes
        else:
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )

        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 regression loss plus GIoU loss on matched boxes."""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        loss_giou = loss_giou.sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_iouaware(self, outputs, targets, indices, num_boxes):
        """Binary cross-entropy on the predicted IoU-aware quality score."""
        idx = self._get_src_permutation_idx(indices)
        src_ious = outputs["pred_ious"][idx]
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        iou_matrix, _ = box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
        )
        target_ious = torch.diag(iou_matrix).view(-1, 1).detach().clamp(0, 1)

        loss_iou = F.binary_cross_entropy_with_logits(src_ious, target_ious, reduction="none")
        return {"loss_iouaware": loss_iou.sum() / num_boxes}

    def get_loss(self, outputs, targets, indices, num_boxes):
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        if "pred_ious" in outputs:
            losses.update(self.loss_iouaware(outputs, targets, indices, num_boxes))
        return losses

    @torch.compiler.disable
    def forward(self, outputs, targets, epoch=0):
        """Compute the full weighted loss dictionary.

        Returns:
            dict of scalar tensors including auxiliary (``*_<i>``) and encoder
            (``*_enc``) terms, plus ``loss_msae`` when present in ``outputs``.
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k not in ["aux_outputs", "enc_outputs"]
        }
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = self.get_loss(outputs, targets, indices, num_boxes)

        for i, aux_outputs in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(aux_outputs, targets, epoch=epoch)
            for k, v in self.get_loss(aux_outputs, targets, aux_indices, num_boxes).items():
                losses[f"{k}_{i}"] = v

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            enc_indices = self.matcher(enc_outputs, targets, epoch=epoch)
            for k, v in self.get_loss(enc_outputs, targets, enc_indices, num_boxes).items():
                losses[f"{k}_enc"] = v

        if "loss_msae" in outputs:
            losses["loss_msae"] = outputs["loss_msae"]

        return losses
