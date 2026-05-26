from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_ROOT / "patch_metadata.csv"

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

            for patch_file in sorted(wsi_folder.iterdir()):
                if patch_file.is_file() and patch_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    rows.append({
                        "patch_name": patch_file.name,
                        "wsi_id": wsi_folder.name,
                        "label": full_label,
                        "split": split,
                        "patch_path": str(Path(split) / short_label / wsi_folder.name / patch_file.name)
                    })

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["patch_name", "wsi_id", "label", "split", "patch_path"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV created: {OUTPUT_CSV}")
print(f"Total patches written: {len(rows)}")