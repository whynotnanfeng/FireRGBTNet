# -*- coding: utf-8 -*-
"""FireRGBTNet -- main LightningModule.

A lightweight end-to-end RGB-T forest fire detector built on three ideas:

1. a heterogeneous dual-stream backbone (RGBBlock / ThermalBlock + TED),
2. an auxiliary MSAE branch that pre-aligns the two modalities,
3. an MSGF neck fusing them via SGCA + MishGLU,

followed by an RT-DETR decoder with IoU-aware scoring.
"""

import math

import lightning as pl
import numpy as np
import torch
import torch.optim as optim

from .boxes import box_cxcywh_to_xyxy
from .criterion import HungarianMatcher, SetCriterion
from .metrics import AdvancedDetMetrics
from .models import FusionNeck, HeterogeneousDualStreamBackbone, MSAE, RTDETRDecoder

__all__ = ["FireRGBTNet"]


class FireRGBTNet(pl.LightningModule):
    """FireRGBTNet detector.

    Architecture, following the paper section by section:

    1. :class:`~firergbtnet.models.backbone.HeterogeneousDualStreamBackbone`
       -- RGBBlock / ThermalBlock with Target-Enhanced Downsampling.
    2. :class:`~firergbtnet.models.alignment.MSAE` -- auxiliary cross-modal
       alignment branch, active during training only.
    3. :class:`~firergbtnet.models.neck.FusionNeck` -- multi-modal spatial
       gated fusion built from SGCA and MishGLU.
    4. :class:`~firergbtnet.models.head.RTDETRDecoder` -- anchor-free decoder
       with IoU-aware scoring.

    Args:
        num_classes: number of foreground classes (smoke / fire / person).
        num_queries: number of object queries (300 in the paper).
        hidden_dim: decoder hidden dimension (128 in the paper).
        num_decoder_layers: number of decoder layers (2 in the paper).
        lr: base learning rate.
        weight_decay: AdamW weight decay.
        matcher_change_epoch: epoch at which the Hungarian matcher switches to
            the high-IoU-prioritised cost.
    """

    def __init__(
        self,
        num_classes=3,
        num_queries=300,
        hidden_dim=128,
        num_decoder_layers=2,
        lr=1e-3,
        weight_decay=1e-4,
        matcher_change_epoch=300,
    ):
        super().__init__()
        pl.seed_everything(42, workers=True)
        self.save_hyperparameters()

        self.num_classes = num_classes

        # 1. Heterogeneous dual-stream backbone -> (C3, C4, C5) per modality
        self.backbone = HeterogeneousDualStreamBackbone()

        # 2. MSAE auxiliary alignment branch (training only)
        self.msae = MSAE(channels_list=(128, 128, 256), active_indices=(0, 1, 2))

        # 3. MSGF fusion neck
        self.neck = FusionNeck(
            in_features=(128, 128, 256),
            out_features=(64, 64, 64),
            scales=(1, 3, 5, 7),
        )

        # 4. RT-DETR decoder head
        self.head = RTDETRDecoder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            in_channels=(128, 128, 128),
        )

        matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=3.0,
            use_focal_loss=True,
            change_matcher=True,
            matcher_change_epoch=matcher_change_epoch,
            iou_order_alpha=4.0,
        )
        weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 8.0,
            "loss_giou": 5.5,
            "loss_iouaware": 1.0,
            "loss_msae": 1.0,
        }
        # Auxiliary decoder-layer weights.
        for i in range(6):
            weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items()})
        # Encoder query-selection head weights.
        weight_dict.update({f"{k}_enc": v for k, v in weight_dict.items() if k in weight_dict})

        self.criterion = SetCriterion(
            num_classes, matcher, weight_dict, alpha=0.25, gamma=2.0, eos_coef=0.1
        )
        self.metrics = AdvancedDetMetrics(
            num_classes=num_classes, iou_threshold=0.5, conf_threshold=0.25
        )

        self._init_weights()

    def _init_weights(self):
        """Match the BatchNorm momentum used during training.

        Conv layers and activations keep PyTorch defaults; BatchNorm uses the
        YOLO-style ``eps``/``momentum`` pair.
        """
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, rgb, thermal):
        """Args:
            rgb: ``[B, 3, H, W]`` visible-light batch.
            thermal: ``[B, 3, H, W]`` thermal batch (3-channel replicated).

        Returns:
            Decoder output dict.
        """
        rgb_feats, thermal_feats = self.backbone(rgb, thermal)

        if self.msae is not None and self.training:
            self._msae_loss = self.msae(rgb_feats, thermal_feats)

        neck_feats = list(self.neck(rgb_feats, thermal_feats))
        return self.head(neck_feats)

    def post_process(self, outputs):
        """Fuse IoU-aware quality into the classification confidence.

        Returns:
            ``[B, num_queries, num_classes]`` score tensor.
        """
        pred_logits = outputs["pred_logits"]
        if self.criterion.matcher.use_focal_loss:
            scores = pred_logits.sigmoid()[..., :-1]
        else:
            scores = pred_logits.softmax(-1)[..., :-1]
        return scores * outputs["pred_ious"].sigmoid()

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------
    def _shared_step(self, batch):
        rgb, thermal, targets = batch
        outputs = self(rgb, thermal)

        if self.training and self.msae is not None:
            outputs = {**outputs, "loss_msae": self._msae_loss}

        loss_dict = self.criterion(outputs, targets, epoch=self.current_epoch)
        weight_dict = self.criterion.weight_dict
        total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        return total, loss_dict, outputs, targets

    def training_step(self, batch, batch_idx):
        total, loss_dict, _, _ = self._shared_step(batch)

        self.log("train_loss", total, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in loss_dict.items():
            if "_enc" in k:
                continue
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(k, v, prog_bar=True, on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        total, loss_dict, outputs, targets = self._shared_step(batch)
        self.log("val_loss", total, on_step=False, on_epoch=True)

        probas = self.post_process(outputs)

        formatted_preds = []
        formatted_targets = []
        for i in range(len(targets)):
            h, w = targets[i]["orig_size"]
            p_scores, p_labels = probas[i].max(-1)

            p_boxes_abs = box_cxcywh_to_xyxy(outputs["pred_boxes"][i])
            p_boxes_abs[:, 0::2] *= w
            p_boxes_abs[:, 1::2] *= h

            t_boxes_abs = box_cxcywh_to_xyxy(targets[i]["boxes"])
            t_boxes_abs[:, 0::2] *= w
            t_boxes_abs[:, 1::2] *= h

            keep = p_scores > 0.001
            formatted_preds.append(
                {"boxes": p_boxes_abs[keep], "scores": p_scores[keep], "labels": p_labels[keep]}
            )
            formatted_targets.append({"boxes": t_boxes_abs, "labels": targets[i]["labels"]})

            if self.current_epoch == self.trainer.max_epochs - 1 and batch_idx == 0 and i == 0:
                self.visualize_prediction(
                    batch[0][i], t_boxes_abs, p_boxes_abs[keep], p_labels[keep], p_scores[keep]
                )

        self.metrics.update(formatted_preds, formatted_targets)
        return total

    @torch.compiler.disable
    def on_validation_epoch_end(self):
        results = self.metrics.compute()

        self.log("val_map", results["map"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_map_50", results["map_50"], prog_bar=True, on_step=False, on_epoch=True)
        if "map_75" in results:
            self.log("val_map_75", results["map_75"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", results["precision"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recall", results["recall"], prog_bar=True, on_step=False, on_epoch=True)
        if "f1" in results:
            self.log("val_f1", results["f1"], prog_bar=False, on_step=False, on_epoch=True)

        self.metrics.reset()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    @torch.compiler.disable
    def visualize_prediction(self, img_tensor, gt_boxes, pred_boxes, pred_labels, scores,
                            class_names=("Smoke", "Fire", "Person")):
        """Log one validation image with GT (green) and prediction (red) boxes."""
        import cv2

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for box in gt_boxes:
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for box, score, label in zip(pred_boxes, scores, pred_labels):
            if score <= 0.25:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
            cv2.putText(
                img, f"{name} {score:.2f}", (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )

        if self.logger and hasattr(self.logger, "experiment") and hasattr(
            self.logger.experiment, "add_image"
        ):
            self.logger.experiment.add_image(
                "val_vis", cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                self.current_epoch, dataformats="HWC",
            )

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        """AdamW with linear warm-up (first 2% of steps) + cosine annealing.

        The minimum learning rate is constrained to 0.1% of the base value,
        matching Table 4 of the paper.
        """
        base_lr = self.hparams.lr
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "head" in n and p.requires_grad],
                "lr": base_lr,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if "head" not in n and p.requires_grad
                ],
                "lr": base_lr,
            },
        ]
        optimizer = optim.AdamW(param_dicts, weight_decay=self.hparams.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        if not total_steps:
            total_steps = 1
        warmup_steps = max(1, int(total_steps * 0.02))
        min_lr_factor = 0.001

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
