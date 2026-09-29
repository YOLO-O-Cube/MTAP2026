# ============================================
# Baseline YOLOv8n Training Script
# Fully aligned with Section 5.1 Training Configuration
# Dataset: OrangeFruitNet (Multi-stage)
# ============================================

# Activate virtual environment
$venvPath = "..\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
} else {
    Write-Error "Virtual environment not found!"
    exit 1
}

# Run YOLOv8n baseline training
yolo detect train `
    model=yolov8n.pt `
    data=configs/orangefruitnet.yaml `
    imgsz=640 `
    epochs=300 `
    batch=16 `
    optimizer=SGD `
    lr0=0.01 `
    momentum=0.937 `
    weight_decay=0.0005 `
    cos_lr=True `
    device=0 `
    pretrained=True `
    project=runs/baseline `
    name=yolov8n_orangefruitnet
