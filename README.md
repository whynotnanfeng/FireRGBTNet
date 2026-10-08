# FireRGBTNet

**A Lightweight Forest Fire Detection Model Based on Efficient RGB-Thermal Fusion**

[![Paper](https://img.shields.io/badge/paper-Forests%202026%2C%2017%2C%20955-2f6b4a)](https://www.mdpi.com/1999-4907/17/8/955)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4.1%2Bcu118-ee4c2c)](https://pytorch.org/)

Official PyTorch implementation of FireRGBTNet, published in
[*Forests* 2026, 17, 955](https://www.mdpi.com/1999-4907/17/8/955).

UAV-based forest fire monitoring must detect very small targets under night-time
illumination, dense smoke and complex terrain, while running on edge hardware.
FireRGBTNet is an anchor-free RGB-T detector with **4.13 M parameters** and
**13.50 G FLOPs**, reaching **95.9 % mAP@0.5** and **62.8 % mAP@0.5:0.95** on
the RGBT-3M benchmark.

## Architecture

| Component | Module | File |
|---|---|---|
| Heterogeneous dual-stream backbone | `RGBBlock`, `ThermalBlock` | `firergbtnet/models/backbone.py` |
| Target-Enhanced Downsampling (with AWF) | `TED` | `firergbtnet/models/backbone.py` |
| Multi-Scale Semantic Alignment Enhancement | `MSAE` | `firergbtnet/models/alignment.py` |
| Multimodal Spatial Gated Fusion | `MSGF` (`SGCA` + `MishGLU`) | `firergbtnet/models/fusion.py` |
| Feature pyramid neck | `FusionNeck` | `firergbtnet/models/neck.py` |
| RT-DETR decoder with IoU-aware scoring | `RTDETRDecoder` | `firergbtnet/models/head.py` |
| Joint set-prediction loss (Hungarian + NWD) | `SetCriterion` | `firergbtnet/loss.py` |

MSAE is an auxiliary supervision branch: it is active only during training and is
dropped at inference, so it improves accuracy at zero deployment cost.

## Module map

Where each piece of the paper lives, including the parts that carry no name in
the publication and are therefore filed under conventional names:

| Paper / original name | Code location | Note |
|---|---|---|
| `RGBBlock`, `RGBUnit` | `models/backbone.py` | standard-convolution ELAN block |
| `ThermalBlock`, `ThermalUnit` | `models/backbone.py` | large-kernel depthwise ELAN block |
| `TED` (Target-Enhanced Downsampling) | `models/backbone.py` | dual-branch downsample with AWF |
| `MSAE` | `models/alignment.py` | auxiliary alignment branch, Eqs. 12-18 |
| `MSGF`, `SGCA`, `MishGLU` | `models/fusion.py` | spatial gated fusion, Eqs. 19-28 |
| Feature pyramid neck | `models/neck.py` | top-down dual-stream FPN |
| `RTDETRDecoder` | `models/head.py` | RT-DETR head with deformable attention |
| `Conv` and the base units | `models/basic.py` | shared conv wrapper |
| `FireRGBTNet` (was `RGBIRNet`) | `model.py` | LightningModule, train/val loops |
| `loss.py` (was `criterion.py`) | `loss.py` | `SetCriterion`, `HungarianMatcher`, NWD |
| `utils.py` | `boxes.py` | box conversions, IoU, GIoU |
| `metrics.py` | `metrics.py` | mAP, P/R/F1, confusion matrix |
| `data/datasets.py` | `data/dataset.py` | RGBT-3M dataset and augmentation |
| `train.py` | `tools/train.py` | training entry point and callbacks |

## Installation

Requires Python 3.10+ and PyTorch 2.0+. The paper used PyTorch 2.4.1 + CUDA 11.8
on a Tesla V100.

```bash
git clone https://github.com/whynotnanfeng/FireRGBTNet.git
cd FireRGBTNet

conda create -n firergbtnet python=3.10 -y
conda activate firergbtnet

pip install -r requirements.txt
```

## Dataset

Training and evaluation use **RGBT-3M**, a public UAV RGB-T forest fire dataset
with 11,220 registered image pairs (7,854 train / 3,366 val) labelled for three
classes: smoke, fire and person.

> Zhang, Y.; Rui, X.; Song, W. *A UAV-Based Multi-Scenario RGB-Thermal Dataset
> and Fusion Model for Enhanced Forest Fire Detection.* Remote Sens. 2025, 17,
> 2593.

Download the dataset from the following mirrors:

- [Quark Netdisk](https://pan.quark.cn/s/ce3e6450be11?pwd=hyx1) — access code `hyx1`
- [Google Drive](https://drive.google.com/file/d/1ZMti4vwcMg2xkdTN_PyyexeVPwFuKtrf/view?usp=sharing)

Note that Quark Netdisk requires a mainland China phone number to access.

Organise the files in this YOLO-style layout, where each RGB image has a
same-named thermal counterpart:

```
RGBT/
├── rgb/
│   ├── train/          # 7854 images
│   └── val/            # 3366 images
├── ir/
│   ├── train/
│   └── val/
└── labels/
    ├── train/          # <name>.txt  ->  cls cx cy w h   (normalised)
    └── val/
```

The dataset is not redistributed with this repository.

## Training

```bash
python tools/train.py --data-root /path/to/RGBT --output-dir runs/firergbtnet
```

Defaults reproduce the configuration reported in the paper:

| Parameter | Value |
|---|---|
| Input size | 640 x 640 |
| Epochs | 250 |
| Batch size | 32 |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| LR schedule | 2 % linear warm-up, then cosine annealing to 1e-6 |
| EMA decay | 0.9999 |
| Mosaic / CutMix | disabled over the final 30 epochs |
| Object queries | 300 |
| Decoder | hidden dim 128, 2 layers |

Useful overrides:

```bash
python tools/train.py --data-root /path/to/RGBT --epochs 300 --batch-size 16
python tools/train.py --data-root /path/to/RGBT --accelerator cpu --devices 1
```

## Results

RGBT-3M validation split, 640 x 640 input, batch size 1 for throughput.

| Model | P | R | F1 | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | FPS | Params (M) | FLOPs (G) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 87.6 | 89.3 | 88.4 | 93.3 | 60.4 | 57.3 | 54.2 | 3.07 | 8.79 |
| + ThermalBlock | 88.4 | 90.9 | 89.6 | 93.8 | 62.5 | 57.9 | 54.3 | 2.91 | 8.51 |
| + TED | 89.7 | 90.2 | 90.0 | 94.1 | 62.1 | 57.8 | 46.1 | 2.87 | 7.88 |
| + MSGF | 91.7 | 93.8 | 92.7 | 95.4 | 66.9 | 61.5 | 44.7 | 4.13 | 13.50 |
| **+ MSAE (FireRGBTNet)** | **91.8** | **94.7** | **93.2** | **95.9** | **68.9** | **62.8** | **44.7** | **4.13** | **13.50** |

## Repository layout

```
FireRGBTNet/
├── firergbtnet/
│   ├── model.py           # FireRGBTNet LightningModule
│   ├── loss.py            # Hungarian matching, NWD, IoU-aware set loss
│   ├── metrics.py         # mAP, precision, recall, F1, confusion matrix
│   ├── boxes.py           # box conversions, IoU, GIoU
│   ├── models/
│   │   ├── basic.py       # Conv wrapper with BN folding for inference
│   │   ├── backbone.py    # RGBBlock, ThermalBlock, TED, dual-stream backbone
│   │   ├── alignment.py   # MSAE branch (CWA + cosine losses)
│   │   ├── fusion.py      # SGCA, MishGLU, MSGF
│   │   ├── neck.py        # top-down dual-stream FPN
│   │   └── head.py        # RT-DETR decoder, deformable attention
│   └── data/
│       └── dataset.py     # RGBT-3M dataset and paired augmentation
└── tools/
    └── train.py           # training entry point with EMA and close-mosaic callbacks
```

## Notes

- `TED` implements the Adaptive Weighted Fusion of Eq. (10)-(11) with learnable
  branch weights, adding 16 learnable scalars in total.
- The neck operates on P3-P5; the stride-4 `C2` stage is computed by the backbone
  but not consumed downstream.
- `MishGLU` uses a 9x9 depthwise convolution inside MSGF and a 3x3 one inside the
  RT-DETR decoder FFN.
- Parameter counts: 5.38 M for the full model during training, 4.27 M for
  deployment once MSAE is discarded.

## Citation

```bibtex
@article{Ma_2026,
  title     = {FireRGBTNet: A Lightweight Forest Fire Detection Model Based on Efficient RGB–Thermal Fusion},
  author    = {Ma, Yifan and Shan, Weifeng and Wang, Maofa and Sui, Yanwei and Wang, Mengyu},
  journal   = {Forests},
  publisher = {MDPI AG},
  volume    = {17},
  number    = {8},
  pages     = {955},
  year      = {2026},
  month     = {Aug},
  issn      = {1999-4907},
  doi       = {10.3390/f17080955},
  url       = {https://doi.org/10.3390/f17080955}
}
```

Cite as: Ma, Y.; Shan, W.; Wang, M.; Sui, Y.; Wang, M. FireRGBTNet: A
Lightweight Forest Fire Detection Model Based on Efficient RGB-Thermal Fusion.
*Forests* **2026**, *17*(8), 955. https://doi.org/10.3390/f17080955

## License

Code is released under the [MIT License](LICENSE). The accompanying article is
published by MDPI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
