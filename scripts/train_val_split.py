import os
import random
import shutil
from tqdm import tqdm

BASE = "/home/soham/cascade-detector/data/yolo_format"

IMG_TRAIN = os.path.join(BASE, "images/train")
LBL_TRAIN = os.path.join(BASE, "labels/train")

IMG_VAL = os.path.join(BASE, "images/val")
LBL_VAL = os.path.join(BASE, "labels/val")

os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

split_ratio = 0.2

extensions = [".png", ".jpg", ".jpeg"]

labels = [f for f in os.listdir(LBL_TRAIN) if f.endswith(".txt")]

random.shuffle(labels)

val_count = int(len(labels) * split_ratio)

val_labels = labels[:val_count]

for label in tqdm(val_labels, desc="Moving validation files"):
    name = os.path.splitext(label)[0]

    src_label = os.path.join(LBL_TRAIN, label)
    dst_label = os.path.join(LBL_VAL, label)

    image_found = False

    for ext in extensions:
        src_image = os.path.join(IMG_TRAIN, name + ext)
        if os.path.exists(src_image):
            dst_image = os.path.join(IMG_VAL, name + ext)
            shutil.move(src_image, dst_image)
            image_found = True
            break

    if image_found:
        shutil.move(src_label, dst_label)
    else:
        print(f"Image not found for {name}")

print("Validation split complete.")

