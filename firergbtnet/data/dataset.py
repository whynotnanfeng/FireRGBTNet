# -*- coding: utf-8 -*-
"""RGBT-3M dataset and data augmentation.

The dataset is expected in the YOLO-style layout produced by the official
RGBT-3M release (Zhang et al., *Remote Sensing* 2025, 17, 2593)::

    RGBT/
    |-- rgb/
    |   |-- train/      # 7854 image pairs
    |   `-- val/        # 3366 image pairs
    |-- ir/
    |   |-- train/
    |   `-- val/
    `-- labels/
        |-- train/      # YOLO txt: cls cx cy w h  (normalised)
        `-- val/

Augmentation combines HSV jitter (visible light only), brightness/contrast
jitter (thermal only), mosaic, CutMix and horizontal flip. Pairs are always
transformed jointly so the two modalities stay pixel-aligned.
"""

import math
import os
import random

import cv2
import lightning as pl
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)

__all__ = ["HYP", "RGBT3MDataset", "RGBTDataModule", "collate_fn"]


#: Augmentation hyper-parameters. HSV jitter only affects the visible light,
#: brightness/contrast jitter only affects the thermal image.
HYP = {
    "hsv_h": 0.01,   # hue gain
    "hsv_s": 0.4,    # saturation gain
    "hsv_v": 0.4,    # brightness (value) gain
    "ir_bri": 15,    # thermal brightness offset (+/-)
    "ir_con": 0.2,   # thermal contrast gain (1 +/- ir_con)
    "ir_p": 0.2,     # probability of applying thermal brightness/contrast jitter
    "degrees": 0.0,  # random rotation (+/- degrees)
    "translate": 0.1,  # random translation as a fraction of width/height
    "scale": 0.5,     # random scale (1 +/- scale)
    "shear": 0.0,     # random shear (+/- degrees)
    "mosaic": 0.4,    # mosaic probability
    "mixup": 0.0,     # CutMix probability (the original MixUp slot)
}


# Normalisation constants. The thermal statistics are dataset-specific and were
# computed over RGBT-3M, giving a more accurate cross-modal alignment.
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
IR_MEAN = [0.323, 0.323, 0.323]
IR_STD = [0.1281, 0.1281, 0.1281]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def augment_brightness_contrast(img, brightness_delta=32, contrast_range=(0.5, 1.5), p=0.5):
    """Apply ``dst = alpha * src + beta`` with probability ``p``."""
    if random.random() >= p:
        return
    beta = random.uniform(-brightness_delta, brightness_delta)
    alpha = random.uniform(*contrast_range)
    cv2.convertScaleAbs(img, alpha=alpha, beta=beta, dst=img)


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """Apply YOLO-style HSV jitter in-place."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert normalised ``(cx, cy, w, h)`` to pixel ``(x1, y1, x2, y2)``."""
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """Convert pixel ``(x1, y1, x2, y2)`` to normalised ``(cx, cy, w, h)``."""
    if clip:
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - eps)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - eps)
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """Filter boxes that became too small or too distorted after augmentation."""
    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    """Resize and pad an image to a fixed shape, preserving the aspect ratio.

    Returns:
        ``(padded_image, scale_ratio, (pad_w, pad_h))``
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RGBT3MDataset(Dataset):
    """Paired RGB / thermal detection dataset.

    Args:
        root: dataset root containing ``rgb/``, ``ir/`` and ``labels/``.
        split: ``'train'`` or ``'val'``.
        img_size: target ``(h, w)`` after letterboxing.
        augment: enable augmentation (forced off for the validation split).
    """

    CLASSES = ("Smoke", "Fire", "Person")

    def __init__(self, root, split="train", img_size=(640, 640), augment=True):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.hyp = HYP

        self.rgb_dir = os.path.join(root, "rgb", split)
        self.ir_dir = os.path.join(root, "ir", split)
        self.label_dir = os.path.join(root, "labels", split)

        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"Path not found: {self.rgb_dir}")

        self.filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.rgb_dir)
            if f.endswith((".jpg", ".png"))
        ]
        if not self.filenames:
            raise RuntimeError(f"No images found in {self.rgb_dir}")

        self.transform_rgb = T.Compose([T.ToTensor(), T.Normalize(RGB_MEAN, RGB_STD)])
        self.transform_ir = T.Compose([T.ToTensor(), T.Normalize(IR_MEAN, IR_STD)])

    def __len__(self):
        return len(self.filenames)

    def _resolve(self, directory, stem):
        """Return the existing path for ``stem`` with a .jpg or .png extension."""
        for ext in (".jpg", ".png"):
            path = os.path.join(directory, stem + ext)
            if os.path.exists(path):
                return path
        return None

    def load_image_pair(self, idx):
        """Load one aligned RGB/thermal pair. Returns ``(rgb, ir, (h, w))``."""
        fname = self.filenames[idx]

        rgb_path = self._resolve(self.rgb_dir, fname)
        if rgb_path is None:
            raise ValueError(f"Missing RGB image for {fname}")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"Failed to load {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        ir_path = self._resolve(self.ir_dir, fname)
        if ir_path is None:
            ir = np.zeros_like(rgb)
        else:
            ir = cv2.imread(ir_path)
            ir = np.zeros_like(rgb) if ir is None else cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

        return rgb, ir, (rgb.shape[0], rgb.shape[1])

    def load_labels(self, idx, h_orig, w_orig):
        """Load YOLO labels as ``[cls, x1, y1, x2, y2]`` in pixels."""
        label_path = os.path.join(self.label_dir, self.filenames[idx] + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) < 5:
                        continue
                    cls = int(data[0])
                    cx, cy, w, h = map(float, data[1:5])
                    labels.append(
                        [
                            cls,
                            (cx - w / 2) * w_orig,
                            (cy - h / 2) * h_orig,
                            (cx + w / 2) * w_orig,
                            (cy + h / 2) * h_orig,
                        ]
                    )
        return np.array(labels) if labels else np.zeros((0, 5))

    def load_mosaic(self, index):
        """Build a 4-image mosaic jointly for both modalities."""
        labels4 = []
        s = self.img_size[0]
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in [-s // 2, -s // 2])

        indices = [index] + random.choices(range(len(self)), k=3)
        img4_rgb = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        img4_ir = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)

        for i, index in enumerate(indices):
            rgb, ir, (h, w) = self.load_image_pair(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4_rgb[y1a:y2a, x1a:x2a] = rgb[y1b:y2b, x1b:x2b]
            img4_ir[y1a:y2a, x1a:x2a] = ir[y1b:y2b, x1b:x2b]

            labels = self.load_labels(index, h, w)
            if labels.size > 0:
                padw, padh = x1a - x1b, y1a - y1b
                labels[:, 1] += padw
                labels[:, 2] += padh
                labels[:, 3] += padw
                labels[:, 4] += padh
                labels4.append(labels)

        if labels4:
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
            w = labels4[:, 3] - labels4[:, 1]
            h = labels4[:, 4] - labels4[:, 2]
            labels4 = labels4[(w > 2) & (h > 2)]
        else:
            labels4 = np.zeros((0, 5))

        return self.random_affine(
            img4_rgb, img4_ir, labels4,
            degrees=self.hyp["degrees"], translate=self.hyp["translate"],
            scale=self.hyp["scale"], shear=self.hyp["shear"], border=-s // 2,
        )

    def random_affine(self, img_rgb, img_ir, targets=(), degrees=10, translate=0.1,
                      scale=0.1, shear=10, border=0):
        """Apply a random affine transform jointly to both modalities and the boxes."""
        height = img_rgb.shape[0] + border * 2
        width = img_rgb.shape[1] + border * 2

        c = np.eye(3)
        c[0, 2] = -img_rgb.shape[1] / 2
        c[1, 2] = -img_rgb.shape[0] / 2

        r = np.eye(3)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        r[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        sh = np.eye(3)
        sh[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
        sh[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

        t = np.eye(3)
        t[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
        t[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

        m = t @ sh @ r @ c
        if (border != 0) or (m != np.eye(3)).any():
            img_rgb = cv2.warpAffine(
                img_rgb, m[:2], dsize=(width, height), flags=cv2.INTER_LINEAR,
                borderValue=(114, 114, 114),
            )
            img_ir = cv2.warpAffine(
                img_ir, m[:2], dsize=(width, height), flags=cv2.INTER_LINEAR,
                borderValue=(114, 114, 114),
            )

        if len(targets) > 0:
            n = len(targets)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ m.T
            xy = xy[:, :2].reshape(n, 8)

            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            i = box_candidates(box1=targets[:, 1:5] * s, box2=new, area_thr=0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img_rgb, img_ir, targets

    def _apply_cutmix(self, rgb, ir, labels):
        """CutMix: paste a second mosaic into a random rectangle.

        Boxes whose centre falls inside the patch are removed from the original
        image; boxes of the donor image whose centre falls inside are kept.
        """
        rgb2, ir2, labels2 = self.load_mosaic(random.randint(0, len(self) - 1))

        h, w = rgb.shape[:2]
        lam = np.random.beta(1.0, 1.0)
        cut_w = int(w * np.sqrt(1.0 - lam))
        cut_h = int(h * np.sqrt(1.0 - lam))
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)

        rgb[y1:y2, x1:x2] = rgb2[y1:y2, x1:x2]
        ir[y1:y2, x1:x2] = ir2[y1:y2, x1:x2]

        if len(labels) > 0:
            cx_l = (labels[:, 1] + labels[:, 3]) / 2
            cy_l = (labels[:, 2] + labels[:, 4]) / 2
            inside = (cx_l > x1) & (cx_l < x2) & (cy_l > y1) & (cy_l < y2)
            labels = labels[~inside]

        if len(labels2) > 0:
            cx2 = (labels2[:, 1] + labels2[:, 3]) / 2
            cy2 = (labels2[:, 2] + labels2[:, 4]) / 2
            labels = np.concatenate((labels, labels2[(cx2 > x1) & (cx2 < x2) & (cy2 > y1) & (cy2 < y2)]), 0)

        return rgb, ir, labels

    def __getitem__(self, idx):
        if self.augment:
            if random.random() < self.hyp["mosaic"]:
                rgb, ir, labels = self.load_mosaic(idx)
                if random.random() < self.hyp["mixup"]:
                    rgb, ir, labels = self._apply_cutmix(rgb, ir, labels)
            else:
                rgb, ir, (h, w) = self.load_image_pair(idx)
                labels = self.load_labels(idx, h, w)
                rgb, ratio, (padw, padh) = letterbox(rgb, self.img_size, scaleup=True)
                ir, _, _ = letterbox(ir, self.img_size, scaleup=True)
                if len(labels):
                    labels[:, 1] = ratio * labels[:, 1] + padw
                    labels[:, 2] = ratio * labels[:, 2] + padh
                    labels[:, 3] = ratio * labels[:, 3] + padw
                    labels[:, 4] = ratio * labels[:, 4] + padh

            augment_hsv(rgb, hgain=self.hyp["hsv_h"], sgain=self.hyp["hsv_s"], vgain=self.hyp["hsv_v"])
            con_gap = self.hyp["ir_con"]
            augment_brightness_contrast(
                ir,
                brightness_delta=self.hyp["ir_bri"],
                contrast_range=(1.0 - con_gap, 1.0 + con_gap),
                p=self.hyp["ir_p"],
            )

            if random.random() < 0.5:
                rgb = cv2.flip(rgb, 1)
                ir = cv2.flip(ir, 1)
                if len(labels):
                    w_img = rgb.shape[1]
                    labels[:, [1, 3]] = w_img - labels[:, [3, 1]]
        else:
            rgb, ir, (h, w) = self.load_image_pair(idx)
            labels = self.load_labels(idx, h, w)
            rgb, ratio, (padw, padh) = letterbox(rgb, self.img_size, scaleup=False)
            ir, _, _ = letterbox(ir, self.img_size, scaleup=False)
            if len(labels):
                labels[:, 1] = ratio * labels[:, 1] + padw
                labels[:, 2] = ratio * labels[:, 2] + padh
                labels[:, 3] = ratio * labels[:, 3] + padw
                labels[:, 4] = ratio * labels[:, 4] + padh

        target = {}
        if len(labels) > 0:
            boxes_xyxy = np.sort(labels[:, 1:], axis=1)
            boxes_norm = xyxy2xywhn(
                boxes_xyxy.copy(), w=rgb.shape[1], h=rgb.shape[0], clip=True
            )
            valid = (boxes_norm[:, 2] > 0.001) & (boxes_norm[:, 3] > 0.001)
            target["boxes"] = torch.tensor(boxes_norm[valid], dtype=torch.float32)
            target["labels"] = torch.tensor(labels[valid, 0], dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        target["orig_size"] = torch.tensor([rgb.shape[0], rgb.shape[1]])
        target["image_id"] = torch.tensor([idx])

        return self.transform_rgb(rgb), self.transform_ir(ir), target


def collate_fn(batch):
    """Collate RGB/thermal tensors into batches, keeping targets as a list."""
    rgb = torch.stack([b[0] for b in batch])
    ir = torch.stack([b[1] for b in batch])
    return rgb, ir, [b[2] for b in batch]


class RGBTDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping the RGBT-3M train/val splits."""

    def __init__(self, data_dir, batch_size=32, num_workers=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = RGBT3MDataset(self.data_dir, split="train", augment=True)
        self.val_dataset = RGBT3MDataset(self.data_dir, split="val", augment=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
