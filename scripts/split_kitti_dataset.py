import os
import shutil
import random
from pathlib import Path

# =========================
# CONFIG
# =========================

BASE_DIR = Path("/home/hs-coburg.de/lin4181s/kitti_yolo/dataset")

IMG_SRC = BASE_DIR / "raw_kitti/training/image_2"
LABEL_SRC = BASE_DIR / "labels/labels_all"

IMG_TRAIN = BASE_DIR / "images/train"
IMG_VAL   = BASE_DIR / "images/val"

LBL_TRAIN = BASE_DIR / "labels/train"
LBL_VAL   = BASE_DIR / "labels/val"

# split ratio
VAL_RATIO = 0.2
SEED = 42

# =========================
# SETUP
# =========================

random.seed(SEED)

IMG_TRAIN.mkdir(parents=True, exist_ok=True)
IMG_VAL.mkdir(parents=True, exist_ok=True)
LBL_TRAIN.mkdir(parents=True, exist_ok=True)
LBL_VAL.mkdir(parents=True, exist_ok=True)

# =========================
# GET ALL IMAGES
# =========================

images = sorted(list(IMG_SRC.glob("*.png")))

print(f"Total images found: {len(images)}")

# shuffle
random.shuffle(images)

split_idx = int(len(images) * (1 - VAL_RATIO))

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

# =========================
# COPY FUNCTION
# =========================

def copy_pair(img_list, img_dst, lbl_dst):
    missing_labels = 0

    for img_path in img_list:
        file_id = img_path.stem
        label_path = LABEL_SRC / f"{file_id}.txt"

        # copy image
        shutil.copy2(img_path, img_dst / img_path.name)

        # copy label if exists
        if label_path.exists():
            shutil.copy2(label_path, lbl_dst / label_path.name)
        else:
            missing_labels += 1

    return missing_labels

# =========================
# EXECUTE SPLIT
# =========================

print("\nCopying TRAIN set...")
train_missing = copy_pair(train_imgs, IMG_TRAIN, LBL_TRAIN)

print("Copying VAL set...")
val_missing = copy_pair(val_imgs, IMG_VAL, LBL_VAL)

print("\n===== DONE =====")
print(f"Missing train labels: {train_missing}")
print(f"Missing val labels: {val_missing}")
print("Dataset successfully prepared for YOLO 🚀")