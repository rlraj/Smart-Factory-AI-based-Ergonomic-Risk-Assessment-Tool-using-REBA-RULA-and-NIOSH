import os

# Define dataset paths
dataset_root = r"C:\Users\220250572\OneDrive - Regal Rexnord\Desktop\Ergonmics\Dataset"
image_dirs = {
    "train": os.path.join(dataset_root, "Images", "Train"),
    "val": os.path.join(dataset_root, "Images", "Validation")
}
label_dirs = {
    "train": os.path.join(dataset_root, "labels", "train"),
    "val": os.path.join(dataset_root, "labels", "validation")
}

# Initialize result containers
missing_labels = {"train": [], "val": []}
empty_labels = {"train": [], "val": []}
total_images = {"train": 0, "val": 0}
total_labels = {"train": 0, "val": 0}

# Function to check label files
def check_labels(split):
    image_dir = image_dirs[split]
    label_dir = label_dirs[split]

    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return
    if not os.path.exists(label_dir):
        print(f"Label directory not found: {label_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images[split] = len(image_files)

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(label_dir, base_name + ".txt")
        if not os.path.exists(label_file):
            missing_labels[split].append(image_file)
        else:
            total_labels[split] += 1
            if os.path.getsize(label_file) == 0:
                empty_labels[split].append(image_file)

# Run checks for both train and val splits
check_labels("train")
check_labels("val")

# Print summary
print("\n=== YOLOv8 Pose Dataset Validation Summary ===")
for split in ["train", "val"]:
    print(f"\nSplit: {split}")
    print(f"Total images: {total_images[split]}")
    print(f"Total label files found: {total_labels[split]}")
    print(f"Missing label files: {len(missing_labels[split])}")
    if missing_labels[split]:
        print(" - " + "\n - ".join(missing_labels[split]))
    print(f"Empty label files: {len(empty_labels[split])}")
    if empty_labels[split]:
        print(" - " + "\n - ".join(empty_labels[split]))
