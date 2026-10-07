# -*- coding: utf-8 -*-
"""Detection metrics: mAP (COCO-style), Precision/Recall/F1 and confusion matrix.

Wraps ``torchmetrics`` so that the reported numbers follow exactly the protocol
described in Sec. 3.1 of the paper: mAP@0.5, mAP@0.75, mAP@0.5:0.95, plus
Precision / Recall / F1 at a fixed IoU threshold.
"""

import torch
import torchvision
from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision

__all__ = ["BoxF1Score", "AdvancedDetMetrics"]


class BoxF1Score(Metric):
    """Greedy matching F1 score at a fixed IoU and confidence threshold.

    Predictions are sorted by confidence and greedily matched to unmatched
    ground-truth boxes of the same class.
    """

    def __init__(self, iou_threshold=0.5, conf_threshold=0.1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            keep = pred["scores"] > self.conf_threshold
            p_boxes = pred["boxes"][keep]
            p_scores = pred["scores"][keep]
            p_labels = pred["labels"][keep]

            if len(gt_boxes) == 0:
                self.fp += len(p_boxes)
                continue
            if len(p_boxes) == 0:
                self.fn += len(gt_boxes)
                continue

            iou_matrix = torchvision.ops.box_iou(p_boxes, gt_boxes)
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=self.device)

            order = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[order]
            p_labels = p_labels[order]
            iou_matrix = iou_matrix[order]

            batch_tp = 0
            batch_fp = 0
            for i in range(len(p_boxes)):
                max_iou, max_idx = torch.max(iou_matrix[i], dim=0)
                if max_iou > self.iou_threshold:
                    if (p_labels[i] == gt_labels[max_idx]) and (not gt_matched[max_idx]):
                        batch_tp += 1
                        gt_matched[max_idx] = True
                    else:
                        batch_fp += 1
                else:
                    batch_fp += 1

            batch_fn = len(gt_boxes) - gt_matched.sum().item()
            self.tp += batch_tp
            self.fp += batch_fp
            self.fn += batch_fn

    def compute(self):
        """Return ``(precision, recall, f1)``."""
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        return precision, recall, f1


class AdvancedDetMetrics(Metric):
    """Aggregate mAP + P/R/F1 + normalised confusion matrix."""

    def __init__(self, num_classes, iou_threshold=0.5, conf_threshold=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
        self.f1_metric = BoxF1Score(iou_threshold=iou_threshold, conf_threshold=conf_threshold)
        self.conf_mat = MulticlassConfusionMatrix(num_classes=num_classes + 1, normalize="true")

    def update(self, preds, targets):
        self.map_metric.update(preds, targets)
        self.f1_metric.update(preds, targets)
        self._update_confusion_matrix(preds, targets)

    def _update_confusion_matrix(self, preds, targets):
        for pred, target in zip(preds, targets):
            valid_mask = pred["scores"] > 0.05
            p_boxes = pred["boxes"][valid_mask]
            p_labels = pred["labels"][valid_mask]
            t_boxes = target["boxes"]
            t_labels = target["labels"]

            if len(t_boxes) == 0:
                if len(p_boxes) > 0:
                    self.conf_mat.update(
                        p_labels, torch.full_like(p_labels, self.num_classes)
                    )
                continue

            if len(p_boxes) == 0:
                self.conf_mat.update(torch.full_like(t_labels, self.num_classes), t_labels)
                continue

            iou_matrix = torchvision.ops.box_iou(p_boxes, t_boxes)
            max_iou_val, max_iou_idx = iou_matrix.max(1)
            matched_mask = max_iou_val > self.iou_threshold

            if matched_mask.any():
                self.conf_mat.update(p_labels[matched_mask], t_labels[max_iou_idx[matched_mask]])
            if (~matched_mask).any():
                self.conf_mat.update(
                    p_labels[~matched_mask], torch.full_like(p_labels[~matched_mask], self.num_classes)
                )

            gt_iou_matrix = torchvision.ops.box_iou(t_boxes, p_boxes)
            if gt_iou_matrix.numel() > 0:
                gt_max_iou, _ = gt_iou_matrix.max(1)
                missed_mask = gt_max_iou < self.iou_threshold
                if missed_mask.any():
                    self.conf_mat.update(
                        torch.full_like(t_labels[missed_mask], self.num_classes),
                        t_labels[missed_mask],
                    )

    def compute(self):
        """Return a dict with ``map_50``, ``map_75``, ``map``, ``precision``, ``recall``, ``f1``, ``conf_mat``."""
        map_res = self.map_metric.compute()
        p, r, f1 = self.f1_metric.compute()
        conf_mat = self.conf_mat.compute()
        return {
            "map_50": map_res["map_50"],
            "map_75": map_res["map_75"],
            "map": map_res["map"],
            "precision": p,
            "recall": r,
            "f1": f1,
            "conf_mat": conf_mat,
        }

    def reset(self):
        self.map_metric.reset()
        self.f1_metric.reset()
        self.conf_mat.reset()
