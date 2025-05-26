from pathlib import Path

def count_class_images(labels_folder, class_id):
    labels_path = Path(labels_folder)
    count = 0
    for label_file in labels_path.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
            # Check if any line corresponds to the target class_id
            if any(line.strip().startswith(str(class_id)) for line in lines):
                count += 1
    return count

merged_dir = Path("merged_dataset")

splits = ["train", "valid", "test"]

for split in splits:
    labels_path = merged_dir / split / "labels"
    civilian_count = count_class_images(labels_path, 0)
    soldier_count = count_class_images(labels_path, 1)
    total_images = len(list(labels_path.glob("*.txt")))
    print(f"{split.capitalize()} set:")
    print(f"  Total images: {total_images}")
    print(f"  Images with civilians: {civilian_count}")
    print(f"  Images with soldiers: {soldier_count}\n")