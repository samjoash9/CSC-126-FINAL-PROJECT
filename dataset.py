import os
import random
import shutil
from pathlib import Path
from roboflow import Roboflow
from glob import glob

# Set random seed for reproducibility
random.seed(42)

# Directories
merged_dir = Path("merged_dataset")
merged_dir.mkdir(exist_ok=True)
for sub in ["train/images", "train/labels", "valid/images", "valid/labels", "test/images", "test/labels"]:
    (merged_dir / sub).mkdir(parents=True, exist_ok=True)

# Download datasets
rf = Roboflow(api_key="OrxOxzT55dMsUMHmo1bf")

# Civilian dataset
civil_proj = rf.workspace("carlapit-m3rsx").project("humandrone1")
civil_version = civil_proj.version(1)
civil_dataset = civil_version.download("yolov8")
civil_path = Path(civil_dataset.location)

# Soldier dataset
sold_proj = rf.workspace("magnesiumbased-lifeform").project("uav-mai")
sold_version = sold_proj.version(1)
sold_dataset = sold_version.download("yolov8")
sold_path = Path(sold_dataset.location)

def collect_pairs(base_path):
    images = glob(str(base_path / "**/images/*.jpg"), recursive=True)
    pairs = []
    for img in images:
        label = Path(img).as_posix().replace("/images/", "/labels/").replace(".jpg", ".txt")
        label_path = Path(label)
        if label_path.exists():
            pairs.append((str(Path(img)), str(label_path)))
    return pairs

def remap_labels_to_one(pairs):
    """Change soldier labels from class 0 to class 1 in their label txt files."""
    for _, label_path in pairs:
        label_path = Path(label_path)
        lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == '0':
                    parts[0] = '1'
                lines.append(" ".join(parts))
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

# Collect all image-label pairs
civil_pairs = collect_pairs(civil_path)
soldier_pairs = collect_pairs(sold_path)

# Balance by number of images (pairs)
min_images = min(len(civil_pairs), len(soldier_pairs))

civil_sample_bal = random.sample(civil_pairs, min_images)
soldier_sample_bal = random.sample(soldier_pairs, min_images)

# Remap soldier labels class 0 -> 1
remap_labels_to_one(soldier_sample_bal)

print(f"Balanced civilian pairs (images): {len(civil_sample_bal)}")
print(f"Balanced soldier pairs (images): {len(soldier_sample_bal)}")

# Combine and shuffle balanced dataset
balanced_combined = civil_sample_bal + soldier_sample_bal
random.shuffle(balanced_combined)

# Split 80% train, 10% valid, 10% test
n_total = len(balanced_combined)
n_train = int(0.8 * n_total)
n_valid = int(0.1 * n_total)

train_set = balanced_combined[:n_train]
valid_set = balanced_combined[n_train:n_train + n_valid]
test_set = balanced_combined[n_train + n_valid:]

def copy_to_subset(pairs, subset):
    for img_path, label_path in pairs:
        img_dest = merged_dir / subset / "images" / Path(img_path).name
        lbl_dest = merged_dir / subset / "labels" / Path(label_path).name
        shutil.copy(img_path, img_dest)
        shutil.copy(label_path, lbl_dest)

copy_to_subset(train_set, "train")
copy_to_subset(valid_set, "valid")
copy_to_subset(test_set, "test")

# Create data.yaml
yaml_path = merged_dir / "data.yaml"
yaml_path.write_text(f"""\
train: ./train/images
val: ./valid/images
test: ./test/images

names:
  0: civilian
  1: soldier
""")

print(f"✅ Balanced merged dataset created in: {merged_dir}")

# Optional: print counts per subset
print(f"Train set: {len(train_set)} images")
print(f"Validation set: {len(valid_set)} images")
print(f"Test set: {len(test_set)} images")