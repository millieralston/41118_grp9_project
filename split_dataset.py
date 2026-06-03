import random
import shutil
from pathlib import Path

"""
Dataset splitting utility for YOLO training.

This script prepares the generated dataset for training by randomly
partitioning images and labels into training and validation subsets.
Image-label pairs are copied into the directory structure expected by
Ultralytics YOLO, ensuring reproducible dataset splits through a fixed
random seed.
"""

# --------------------------
# Settings
# --------------------------

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

dataset_dir = Path("dataset")

images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

random.seed(SEED)

# --------------------------
# Find all images
# --------------------------

image_files = list(images_dir.glob("*.png"))

print("Images:", len(list(images_dir.glob("*.png"))))
print("Labels:", len(list(labels_dir.glob("*.txt"))))

print(f"Images found: {len(image_files)}")

label_files = list(labels_dir.glob("*.txt"))
print(f"Labels found: {len(label_files)}")

random.shuffle(image_files)

n_total = len(image_files)

n_train = int(TRAIN_RATIO * n_total)

train_files = image_files[:n_train]
val_files = image_files[n_train:]

# --------------------------
# Create folders
# --------------------------

for split in ["train", "val"]:

    (images_dir / split).mkdir(exist_ok=True)

    (labels_dir / split).mkdir(exist_ok=True)

# --------------------------
# Copy files
# --------------------------

def copy_split(files, split):

    for img_path in files:

        label_path = labels_dir / f"{img_path.stem}.txt"

        shutil.copy(
            img_path,
            images_dir / split / img_path.name
        )

        shutil.copy(
            label_path,
            labels_dir / split / label_path.name
        )

copy_split(train_files, "train")
copy_split(val_files, "val")

print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")