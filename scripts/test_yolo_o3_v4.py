"""
YOLO-O³ v4 architecture-only validation.

This script does NOT train the model.

It verifies:
1. YAML model construction
2. Manuscript-specific SpatialAttention replacement
3. Fixed 0.50/0.30/0.20 P3/P4/P5 fusion replacement
4. Fusion weights and trainability
5. 640x640 CPU forward pass
6. Detection output construction

Run from the YOLO project root:

    python scripts/test_yolo_o3_v4.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


MODEL_YAML = "models/yolo_o3_v4.yaml"
IMG_SIZE = 640


# ============================================================
# Spatial Attention
# ============================================================

class SpatialAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()

        self.attn = nn.Conv2d(
            channels,
            1,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(self, x):
        attention = torch.sigmoid(self.attn(x))
        return x * attention


# ============================================================
# Fixed Weighted Multi-Scale Fusion
# ============================================================

class FixedWeightedFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        weights=(0.50, 0.30, 0.20),
    ):
        super().__init__()

        w = torch.tensor(
            weights,
            dtype=torch.float32
        )

        if w.numel() != 3:
            raise ValueError(
                "Fusion requires exactly three weights."
            )

        if not torch.isclose(
            w.sum(),
            torch.tensor(1.0)
        ):
            raise ValueError(
                "Fusion weights must sum to 1.0."
            )

        self.channels = channels

        self.register_buffer(
            "weights",
            w.view(3, 1, 1, 1)
        )

    def forward(self, x):

        expected_channels = self.channels * 3

        if x.shape[1] != expected_channels:
            raise RuntimeError(
                f"Fusion expected "
                f"{expected_channels} channels, "
                f"but received {x.shape[1]}."
            )

        p3, p4, p5 = torch.chunk(
            x,
            3,
            dim=1
        )

        return (
            self.weights[0] * p3
            + self.weights[1] * p4
            + self.weights[2] * p5
        )


# ============================================================
# Replace YAML placeholders with manuscript modules
# ============================================================

def replace_placeholders(model):

    net = model.model

    # --------------------------------------------------------
    # Spatial attention
    # --------------------------------------------------------

    attention_layers = {
        5: 128,
        8: 256,
        12: 512,
    }

    for layer_index, channels in attention_layers.items():

        net.model[layer_index] = SpatialAttention(
            channels
        )

    # --------------------------------------------------------
    # Fixed P3/P4/P5 weighted fusion
    #
    # P3 = 0.50
    # P4 = 0.30
    # P5 = 0.20
    # --------------------------------------------------------

    net.model[19] = FixedWeightedFusion(
        channels=128,
        weights=(0.50, 0.30, 0.20),
    )

    return model


# ============================================================
# Parameter count
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.model.parameters()
    )


# ============================================================
# Main test
# ============================================================

def main():

    print()
    print("=" * 72)
    print("YOLO-O³ v4 ARCHITECTURE-ONLY TEST")
    print("=" * 72)

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    print("Device  : CPU")
    print(f"Model   : {MODEL_YAML}")

    if not Path(MODEL_YAML).exists():

        raise FileNotFoundError(
            f"Model YAML not found: {MODEL_YAML}"
        )

    # ========================================================
    # 1. YAML construction
    # ========================================================

    print()
    print("[1/6] Constructing model from YAML...")

    model = YOLO(MODEL_YAML)

    print("PASS: YAML model construction")

    # ========================================================
    # 2. Replace manuscript placeholders
    # ========================================================

    print()
    print(
        "[2/6] Installing manuscript-specific "
        "custom modules..."
    )

    model = replace_placeholders(model)

    assert isinstance(
        model.model.model[5],
        SpatialAttention
    )

    assert isinstance(
        model.model.model[8],
        SpatialAttention
    )

    assert isinstance(
        model.model.model[12],
        SpatialAttention
    )

    assert isinstance(
        model.model.model[19],
        FixedWeightedFusion
    )

    print(
        "PASS: SpatialAttention "
        "at layers 5, 8, 12"
    )

    print(
        "PASS: FixedWeightedFusion "
        "at layer 19"
    )

    # ========================================================
    # 3. Verify fusion coefficients
    # ========================================================

    print()
    print(
        "[3/6] Verifying fixed fusion coefficients..."
    )

    fusion = model.model.model[19]

    weights = fusion.weights.flatten().tolist()

    print(
        f"Fusion weights: {weights}"
    )

    expected_weights = [
        0.50,
        0.30,
        0.20,
    ]

    assert all(
        abs(a - b) < 1e-6
        for a, b in zip(
            weights,
            expected_weights
        )
    )

    assert abs(
        sum(weights) - 1.0
    ) < 1e-6

    trainable_fusion_params = [
        p
        for p in fusion.parameters()
        if p.requires_grad
    ]

    assert len(
        trainable_fusion_params
    ) == 0

    print(
        "PASS: weights = "
        "0.50 / 0.30 / 0.20"
    )

    print(
        "PASS: fusion coefficients "
        "are non-trainable"
    )

    # ========================================================
    # 4. Parameter count
    # ========================================================

    print()
    print("[4/6] Counting parameters...")

    params = count_parameters(model)

    print(
        f"Total parameters: {params:,}"
    )

    print(
        "PASS: parameter count obtained"
    )

    # ========================================================
    # 5. CPU forward pass
    # ========================================================

    print()
    print(
        "[5/6] Running 640x640 CPU forward pass..."
    )

    x = torch.zeros(
        1,
        3,
        IMG_SIZE,
        IMG_SIZE
    )

    model.model.eval()

    with torch.no_grad():

        output = model.model(x)

    print(
        "PASS: CPU forward pass completed"
    )

    print(
        f"Output type: {type(output).__name__}"
    )

    if isinstance(
        output,
        (tuple, list)
    ):

        print(
            f"Output elements: {len(output)}"
        )

        for i, item in enumerate(output):

            if torch.is_tensor(item):

                print(
                    f"  output[{i}] "
                    f"shape = {tuple(item.shape)}"
                )

            elif isinstance(
                item,
                (tuple, list)
            ):

                print(
                    f"  output[{i}] = "
                    f"{type(item).__name__} "
                    f"({len(item)} items)"
                )

    elif torch.is_tensor(output):

        print(
            f"Output shape: "
            f"{tuple(output.shape)}"
        )

    # ========================================================
    # 6. Final checks
    # ========================================================

    print()
    print(
        "[6/6] Final architecture checks..."
    )

    print()
    print("Layer 5:")
    print(model.model.model[5])

    print()
    print("Layer 8:")
    print(model.model.model[8])

    print()
    print("Layer 12:")
    print(model.model.model[12])

    print()
    print("Layer 19:")
    print(model.model.model[19])

    print()
    print("=" * 72)
    print(
        "YOLO-O³ v4 ARCHITECTURE TEST: SUCCESS"
    )
    print("=" * 72)

    print()
    print("No training was performed.")


if __name__ == "__main__":
    main()