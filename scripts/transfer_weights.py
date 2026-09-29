from ultralytics import YOLO
import torch

print("Loading OLD trained model...")
old = YOLO("models/yolo_o3_coco_best.pt", task="detect").model

print("Building NEW P4-only architecture...")
new = YOLO("models/yolo_o3_v3.yaml", task="detect").model

old_sd = old.state_dict()
new_sd = new.state_dict()

transfer_sd = {}
skipped = []

for k, v in old_sd.items():
    if k in new_sd and v.shape == new_sd[k].shape:
        transfer_sd[k] = v
    else:
        skipped.append(k)

print(f"Old layers: {len(old_sd)}")
print(f"New layers: {len(new_sd)}")
print(f"Transferred layers: {len(transfer_sd)}")
print(f"Skipped layers (P3 + new P4_flower): {len(skipped)}")

new.load_state_dict(transfer_sd, strict=False)

out_path = "models/yolo_o3_v3_init_from_coco.pt"
torch.save(
    {
        "model": new,
        "args": {"task": "detect"}
    },
    out_path
)

print(f"✅ Weight-transferred model saved to: {out_path}")
