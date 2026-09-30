"""
====================================================================
YOLO-O³ v4 Training Script

Purpose
-------
Train the configuration of YOLO-O³ on OrangeFruitNet+.
The implementation follows the  YOLO-O³ architecture and training protocol:

  * DWConv lightweight stem
  * C2f-Light-style reduced-depth backbone
  * alpha = 0.50 channel scaling
  * lightweight spatial refinement
  * P3/P4/P5 multi-scale representation
  * fixed weighted fusion (0.50, 0.30, 0.20)
  * anchor-free YOLOv8-style Detect head
  * 640x640, 300 epochs, batch 16
  * SGD, lr0=0.01, momentum=0.937, weight_decay=5e-4
  * cosine learning-rate decay
  * COCO initialization

Important implementation note
-----------------------------
Ultralytics model YAML syntax cannot directly express the two custom
operations used in our manuscript:

  A) spatial attention: A(F) = sigmoid(Conv(F)) and F' = F * A(F)
  B) fixed fusion: F = 0.50 P3 + 0.30 P4 + 0.20 P5

The YAML therefore contains parser-safe placeholders. This script
replaces those placeholders after model construction. The fusion
weights are frozen and are NOT learnable.

Hardware
--------
The main experiment used an NVIDIA RTX 4070 SUPER. Set DEVICE
below to "cpu" only for local architecture/inference testing.

Run
---
    python scripts/train_yolo_o3_v4.py
====================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from ultralytics import YOLO


# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------
ALPHA = 0.50
BETA = 0.50
FUSION_WEIGHTS = (0.50, 0.30, 0.20)
DEVICE = 0

MODEL_YAML = "models/yolo_o3_v4.yaml"
DATA_YAML = "configs/orangefruitnet.yaml"
PROJECT = "runs/yolo_o3"
RUN_NAME = "yolo_o3_v4"


class SpatialAttention(nn.Module):
    """Lightweight spatial attention described in Section 4.3.4.

    A(F) = sigmoid(Conv(F))
    F'  = F * A(F)

    The convolution produces a one-channel spatial attention map, so the
    additional parameter and compute overhead remain small.
    """

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.attn = nn.Conv2d(channels, 1, kernel_size, stride=1, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.attn(x))
        return x * a


class FixedWeightedFusion(nn.Module):
    """Non-learned P3/P4/P5 fusion with weights 0.50/0.30/0.20."""

    def __init__(self, channels: int, weights: Iterable[float] = FUSION_WEIGHTS):
        super().__init__()
        w = torch.tensor(tuple(weights), dtype=torch.float32)
        if w.numel() != 3:
            raise ValueError("YOLO-O³ v4 requires exactly three P3/P4/P5 fusion weights.")
        if not torch.isclose(w.sum(), torch.tensor(1.0)):
            raise ValueError("Fusion weights must sum to 1.0.")
        self.channels = channels
        self.register_buffer("weights", w.view(3, 1, 1, 1), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels * 3:
            raise RuntimeError(
                f"FixedWeightedFusion expected {self.channels * 3} channels, "
                f"received {x.shape[1]}."
            )
        p3, p4, p5 = torch.chunk(x, 3, dim=1)
        return (
            self.weights[0] * p3
            + self.weights[1] * p4
            + self.weights[2] * p5
        )


def _replace_manuscript_placeholders(model: YOLO) -> None:
    """Replace YAML placeholders with the manuscript operations.

    Layer 5/8/12 are CBAM placeholders in the YAML and become the exact
    lightweight spatial-attention block used here.
    Layer 19 is the concatenation-to-Conv placeholder and becomes fixed
    weighted P3/P4/P5 fusion.
    """

    net = model.model

    # The indices are deliberately documented in yolo_o3_v4.yaml.
    attention_layers = {5: 128, 8: 256, 12: 512}
    for idx, channels in attention_layers.items():
        net.model[idx] = SpatialAttention(channels)

    # Layer 18 is Concat(P3,P4,P5), hence layer 19 receives 3*128 channels.
    net.model[19] = FixedWeightedFusion(channels=128)



def _freeze_fixed_fusion(model: YOLO) -> None:
    """Keep the manuscript fusion coefficients fixed throughout training."""

    for module in model.model.modules():
        if isinstance(module, FixedWeightedFusion):
            for parameter in module.parameters():
                parameter.requires_grad_(False)



def main() -> None:
    print("=" * 72)
    print("YOLO-O³ v4 | manuscript-aligned training")
    print(f"alpha = {ALPHA:.2f}, beta = {BETA:.2f}")
    print(f"fusion weights = {FUSION_WEIGHTS}")
    print(f"device = {DEVICE}")
    print("=" * 72)

    if not Path(MODEL_YAML).exists():
        raise FileNotFoundError(MODEL_YAML)
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(DATA_YAML)

    # Build from YAML so the v4 architecture remains reproducible.
    model = YOLO(MODEL_YAML)

    # Replace parser-safe placeholders with manuscript operations.
    _replace_manuscript_placeholders(model)
    _freeze_fixed_fusion(model)

    # all models use MS COCO initialization.
    # For a YAML-created model, Ultralytics applies pretrained initialization
    # through its normal model-training pathway when pretrained=True.
    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=300,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        cos_lr=True,
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,
        pretrained=True,
        patience=100,
        workers=8,
        verbose=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
