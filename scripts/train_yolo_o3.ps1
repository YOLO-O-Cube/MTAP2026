# ============================================================
# YOLO-O³ Training Script (PowerShell)
# Author: Mohan Khedkar
# Purpose: Train YOLO-O³ on OrangeFruitNet
# Environment: Windows 11 + PowerShell + venv
#   cd C:\Users\Admin\yolo_o3
#   .\scripts\train_yolo_o3.ps1
# ============================================================

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Verify environment
Write-Host "Using Python:"
python --version
Write-Host "Using YOLO:"
yolo version

# Run YOLO-O³ training
Write-Host "Starting YOLO-O³ training..."

yolo detect train `
  model=models/yolo_o3.yaml `
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
  project=runs/yolo_o3 `
  name=yolo_o3

Write-Host "Training completed."
