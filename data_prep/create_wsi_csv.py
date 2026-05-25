from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_ROOT / "wsi_metadata.csv"

LABEL_MAP = {
    "c": "chromophobe",
    "cc": "clearcell",
    "o": "oncocytoma",
    "p": "papillary"
}

rows = []

for split in ["train", "validate", "test"]:
    split_path = DATA_ROOT / split

    if not split_path.is_dir():
        print(f"Skipping missing split folder: {split_path}")
        continue

    for short_label, full_label in LABEL_MAP.items():
        class_path = split_path / short_label

        if not class_path.is_dir():
            print(f"Skipping missing class folder: {class_path}")
            continue

        for wsi_folder in sorted(class_path.iterdir()):
            if not wsi_folder.is_dir():
                continue

            patch_files = [
                f for f in wsi_folder.iterdir()
                if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
            ]

            rows.append({
                "wsi_id": wsi_folder.name,
                "label": full_label,
                "split": split,
                "patch_dir": str(Path(split) / short_label / wsi_folder.name),
                "num_patches": len(patch_files)
            })

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["wsi_id", "label", "split", "patch_dir", "num_patches"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV created: {OUTPUT_CSV}")
print(f"Total WSIs written: {len(rows)}")