# ============================================
# Baseline YOLOv8n Training Script
# Dataset: OrangeFruitNet (Multi-stage)
# ============================================
# Run YOLOv8n baseline training
# Single-line command
#       yolo detect train model=yolov8n.pt data=configs/orangefruitnet.yaml imgsz=640 epochs=300 batch=16 optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=0.0005 cos_lr=True device=0 project=runs/baseline name=yolov8n_orangefruitnet pretrained=True
# OR    run this file using following command
#       python scripts/train_baseline_yolov8n.py

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

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
    device=0,
    pretrained=True,
    project="runs/baseline",
    name="yolov8n_orangefruitnet"
)
