import os
import csv
import shutil
from pathlib import Path

# -------- CONFIG --------
BASE_DIR = Path("/path/to/datasets/CAER/CAER")
OUTPUT_VIDEO_DIR = Path("/path/to/datasets/CAER/videos_avi")
CSV_OUTPUT_DIR = Path("/path/to/datasets/CAER")


SPLITS = {
    "train": "train.csv",
    "validation": "val.csv",
    "test": "test.csv",
}

LABEL_MAP = {
    "Surprise": 0,
    "Sad": 1,
    "Neutral": 2,
    "Happy": 3,
    "Fear": 4,
    "Disgust": 5,
    "Anger": 6,
}

MOVE_FILES = True   # set False if you want to COPY instead
# ------------------------

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

global_id = 0

for split, csv_name in SPLITS.items():
    rows = []

    split_dir = BASE_DIR / split
    assert split_dir.exists(), f"{split_dir} does not exist"

    for class_name, label in LABEL_MAP.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        for video_path in sorted(class_dir.glob("*.avi")):
            unique_id = f"{global_id:07d}"   # zero-padded for cleanliness
            new_name = f"{unique_id}.avi"
            new_path = OUTPUT_VIDEO_DIR / new_name

            if MOVE_FILES:
                shutil.move(video_path, new_path)
            else:
                shutil.copy2(video_path, new_path)

            rows.append([unique_id, label])
            global_id += 1

    # write CSV
    csv_path = CSV_OUTPUT_DIR / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {csv_name}")

print(f"Done. Total videos processed: {global_id}")
