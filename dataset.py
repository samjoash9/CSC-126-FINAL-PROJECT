import os
import shutil

from roboflow import Roboflow
from ultralytics import YOLO

# Step 1: Download both datasets
rf = Roboflow(api_key="OrxOxzT55dMsUMHmo1bf")

# Civilian Dataset
civilian_proj = rf.workspace("folks").project("look-down-folks")
civilian_ver = civilian_proj.version(1)
civilian_ds = civilian_ver.download("yolov8")

# Soldier Dataset
soldier_proj = rf.workspace("prisoner-detection").project("my-first-project-4zuff")
soldier_ver = soldier_proj.version(3)
soldier_ds = soldier_ver.download("yolov8")

# Step 2: Setup merged dataset directory
merged_dir = "merged_dataset"
os.makedirs(f"{merged_dir}/images/train", exist_ok=True)
os.makedirs(f"{merged_dir}/labels/train", exist_ok=True)

def copy_images_and_labels(src_img, src_lbl, label_map_fn):
    for fname in os.listdir(src_img):
        shutil.copy(os.path.join(src_img, fname), f"{merged_dir}/images/train")

    for fname in os.listdir(src_lbl):
        src_label_path = os.path.join(src_lbl, fname)
        dst_label_path = os.path.join(f"{merged_dir}/labels/train", fname)

        with open(src_label_path, "r") as f:
            lines = f.readlines()

        # Map labels
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip malformed
            class_id = int(parts[0])
            new_class_id = label_map_fn(class_id)
            new_lines.append(" ".join([str(new_class_id)] + parts[1:]))

        with open(dst_label_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

# Step 3: Merge civilian (class 0 → 0)
copy_images_and_labels(
    src_img=f"{civilian_ds.location}/train/images",
    src_lbl=f"{civilian_ds.location}/train/labels",
    label_map_fn=lambda c: 0  # civilian
)

# Step 4: Merge soldier (class 0 → 1)
copy_images_and_labels(
    src_img=f"{soldier_ds.location}/train/images",
    src_lbl=f"{soldier_ds.location}/train/labels",
    label_map_fn=lambda c: 1  # soldier
)

# Step 5: Write data.yaml
data_yaml = f"""
train: {os.path.abspath(merged_dir)}/images/train
val: {os.path.abspath(merged_dir)}/images/train

nc: 2
names: ['civilian', 'soldier']
"""

with open(f"{merged_dir}/data.yaml", "w") as f:
    f.write(data_yaml.strip())

print("✅ Dataset merged successfully. Ready for training.")