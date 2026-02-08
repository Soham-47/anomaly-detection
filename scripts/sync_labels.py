import os
from pathlib import Path

BASE_DIR = Path("/home/soham/cascade-detector/data/yolo_format")
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def sync():
    total_deleted = 0
    
    for split in SPLITS:
        image_dir = BASE_DIR / "images" / split
        label_dir = BASE_DIR / "labels" / split
        
        if not image_dir.exists() or not label_dir.exists():
            print(f"Skipping {split} - directory not found.")
            continue
            
        image_stems = {f.stem for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}
        
        label_files = list(label_dir.glob("*.txt"))
        
        for lf in label_files:
            if lf.stem not in image_stems:
                print(f"Deleting orphan label: {lf.name} (No matching image in {split})")
                lf.unlink()
                total_deleted += 1
                
    print(f"\nCleanup complete. Total orphan labels deleted: {total_deleted}")

if __name__ == "__main__":
    sync()
