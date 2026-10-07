# -*- coding: utf-8 -*-
"""Train FireRGBTNet on the RGBT-3M dataset.

Reproduces the training configuration of Table 4 in the paper:
640x640 input, 250 epochs, batch size 32, 16 workers, AdamW
(lr 1e-3, weight decay 1e-4), 2% linear warm-up followed by cosine annealing
with a 1e-6 floor.

Example::

    python tools/train.py --data-root /path/to/RGBT --output-dir runs/firergbtnet
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import math

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from firergbtnet import FireRGBTNet
from firergbtnet.data import RGBTDataModule


class ModelEMA(pl.Callback):
    """Exponential moving average of the model weights.

    The averaged weights are swapped in during validation and swapped back
    afterwards, so training always continues from the raw optimiser state.
    """

    def __init__(self, decay=0.9999, use_num_updates=True):
        super().__init__()
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.ema_state_dict = {}
        self.steps = 0

    def on_fit_start(self, trainer, pl_module):
        self.ema_state_dict = {
            k: v.detach().clone() for k, v in pl_module.state_dict().items()
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.steps += 1
        decay = self.decay
        if self.use_num_updates:
            decay = min(self.decay, (1 + self.steps) / (10 + self.steps))

        with torch.no_grad():
            model_state = pl_module.state_dict()
            for k, ema_v in self.ema_state_dict.items():
                if k in model_state and ema_v.dtype.is_floating_point:
                    model_v = model_state[k]
                    ema_v.copy_(
                        ema_v * decay + model_v.to(ema_v.device) * (1.0 - decay)
                    )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.swap_weights(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.swap_weights(pl_module)

    def swap_weights(self, pl_module):
        """Exchange the live weights and the EMA weights."""
        with torch.no_grad():
            model_state = pl_module.state_dict()
            for k, ema_v in self.ema_state_dict.items():
                if k in model_state:
                    model_v = model_state[k]
                    temp = model_v.detach().clone()
                    model_v.copy_(ema_v.to(model_v.device))
                    ema_v.copy_(temp.to(ema_v.device))


class CloseMosaicCallback(pl.Callback):
    """Disable mosaic and CutMix during the final epochs of training.

    Closing the aggressive geometric augmentations lets the model settle on
    clean inputs, which usually gives a small but consistent mAP gain.
    """

    def __init__(self, close_epochs=15):
        super().__init__()
        self.close_epochs = close_epochs
        self.is_closed = False

    @staticmethod
    def _find_dataset_with_hyp(obj):
        """Depth-first search for the dataset object holding the ``hyp`` dict."""
        if hasattr(obj, "hyp"):
            return obj
        if hasattr(obj, "dataset"):
            return CloseMosaicCallback._find_dataset_with_hyp(obj.dataset)
        if isinstance(obj, (list, tuple)):
            for sub in obj:
                found = CloseMosaicCallback._find_dataset_with_hyp(sub)
                if found:
                    return found
        return None

    def on_train_epoch_start(self, trainer, pl_module):
        remaining = trainer.max_epochs - trainer.current_epoch
        if remaining <= self.close_epochs and not self.is_closed:
            print("=" * 40)
            print(
                f"[Strategy] Epoch {trainer.current_epoch}: closing mosaic & CutMix"
            )
            print("=" * 40)

            target_dataset = None
            if trainer.datamodule is not None and hasattr(trainer.datamodule, "train_dataset"):
                target_dataset = self._find_dataset_with_hyp(
                    trainer.datamodule.train_dataset
                )
            if target_dataset is None:
                loader = trainer.train_dataloader
                if hasattr(loader, "dataset"):
                    target_dataset = self._find_dataset_with_hyp(loader.dataset)

            if target_dataset is not None:
                print(f"Found dataset: {type(target_dataset).__name__}")
                target_dataset.hyp["mosaic"] = 0.0
                target_dataset.hyp["mixup"] = 0.0
                self.is_closed = True
                if hasattr(trainer, "reset_train_dataloader"):
                    trainer.reset_train_dataloader(pl_module)


class EpochSeparator(pl.Callback):
    """Print the current epoch index at the start of every epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f"[Training] Starting epoch {trainer.current_epoch}/{trainer.max_epochs - 1}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Train FireRGBTNet on RGBT-3M",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True, help="dataset root containing rgb/ ir/ labels/")
    p.add_argument("--output-dir", default="runs/firergbtnet", help="checkpoint / log directory")
    p.add_argument("--num-classes", type=int, default=3, help="smoke, fire, person")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3, help="initial AdamW learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accumulate-grad-batches", type=int, default=4)
    p.add_argument("--gradient-clip-val", type=float, default=0.1)
    p.add_argument("--num-queries", type=int, default=300, help="object queries")
    p.add_argument("--hidden-dim", type=int, default=128, help="decoder hidden dimension")
    p.add_argument("--num-decoder-layers", type=int, default=2)
    p.add_argument("--matcher-change-epoch", type=int, default=300)
    p.add_argument("--close-mosaic-epochs", type=int, default=30)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--accelerator", default="auto", help="'gpu', 'cpu' or 'auto'")
    p.add_argument("--devices", default=1, help="int, or comma-separated list of GPU ids")
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_devices(devices):
    """Normalise the ``--devices`` argument into a Lightning-compatible value."""
    if isinstance(devices, int):
        return devices
    text = str(devices)
    if "," in text:
        return [int(x) for x in text.split(",")]
    return int(text)


def main():
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    dm = RGBTDataModule(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            monitor="val_map",
            mode="max",
            filename="firergbtnet-{epoch:02d}-{val_map:.4f}",
            save_top_k=1,
        ),
        CloseMosaicCallback(close_epochs=args.close_mosaic_epochs),
        ModelEMA(decay=args.ema_decay),
        EpochSeparator(),
    ]

    model = FireRGBTNet(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        num_decoder_layers=args.num_decoder_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        matcher_change_epoch=args.matcher_change_epoch,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        sync_batchnorm=False,
        precision="16-mixed",
        default_root_dir=args.output_dir,
        log_every_n_steps=60,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
