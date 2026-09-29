"""
============================================================
YOLO-O³ Training Script (Python API)
Author: Mohan Khedkar
Purpose:
    Train the proposed YOLO-O³ architecture on the expanded
    OrangeFruitNet dataset using Ultralytics YOLOv8 framework.

This script corresponds to:
- Section 4: Methodology
- Section 5: Experimental Setup
- Algorithm 1: YOLO-O³ Training Pipeline

Hardware:
- NVIDIA RTX 4070 SUPER

How to run
    (venv) python scripts/train_yolo_o3.py
OR  oneliner 
    yolo detect train model=models/yolo_o3.yaml data=configs/orangefruitnet.yaml imgsz=640 epochs=300 batch=16 optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=0.0005 cos_lr=True device=0 project=runs/yolo_o3 name=yolo_o3
============================================================
"""

from ultralytics import YOLO

def main():
    # --------------------------------------------------------
    # Load YOLO-O³ model definition
    # --------------------------------------------------------
    model = YOLO("models/yolo_o3.yaml")

    # --------------------------------------------------------
    # Training configuration (matches manuscript Section 5.1)
    # --------------------------------------------------------
    model.train(
        data="configs/orangefruitnet.yaml",
        imgsz=640,
        epochs=300,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        cos_lr=True,
        device=0,                     # GPU 0 (RTX 4070 SUPER)
        project="runs/yolo_o3",
        name="yolo_o3",
        pretrained=True,              # COCO initialization
        patience=100,                 # Early stopping
        workers=8,
        verbose=True,
        plots=True
    )

if __name__ == "__main__":
    main()
