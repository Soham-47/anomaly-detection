import json
from pathlib import Path
from tqdm import tqdm

DATA_ROOT = Path("/home/soham/cascade-detector/data/raw/RDD2022 sample")
OUTPUT_ROOT = Path("/home/soham/cascade-detector/data/yolo_format")

CLASS_MAP = {
    "alligator crack": 0,
    "block crack": 1,
    "longitudinal crack": 2,
    "other corruption": 3,
    "pothole": 4,
    "repair": 5,
    "transverse crack": 6
}

splits = ["train", "test"]

def convert_bbox(xmin, ymin, xmax, ymax, w, h):
    xc = (xmin + xmax) / 2 / w
    yc = (ymin + ymax) / 2 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return xc, yc, bw, bh

for split in splits:
    ann_dir = DATA_ROOT / split / "ann"
    out_label_dir = OUTPUT_ROOT / "labels" / split
    out_label_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(ann_dir.glob("*.json"))

    for jf in tqdm(json_files, desc=f"Converting {split}"):

        with open(jf) as f:
            data = json.load(f)

        w = data["size"]["width"]
        h = data["size"]["height"]

        image_name = jf.stem
        label_path = out_label_dir / f"{Path(image_name).stem}.txt"

        lines = []

        for obj in data["objects"]:

            cls = obj["classTitle"]

            if cls not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[cls]

            xmin, ymin = obj["points"]["exterior"][0]
            xmax, ymax = obj["points"]["exterior"][1]

            xc, yc, bw, bh = convert_bbox(xmin, ymin, xmax, ymax, w, h)

            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))





    