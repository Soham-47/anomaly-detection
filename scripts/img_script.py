import shutil
from pathlib import Path
from tqdm import tqdm

RAW_ROOT = Path("/home/soham/cascade-detector/data/raw/RDD2022 sample")
YOLO_ROOT = Path("/home/soham/cascade-detector/data/yolo_format")

splits = ["train", "test"]

for split in splits:

    src_dir = RAW_ROOT / split / "img"
    dst_dir = YOLO_ROOT / "images" / split

    dst_dir.mkdir(parents=True, exist_ok=True)

    images = list(src_dir.glob("*.jpg"))

    for img in tqdm(images, desc=f"Copying {split} images"):
        shutil.copy(img, dst_dir / img.name)

print("Images copied.")

