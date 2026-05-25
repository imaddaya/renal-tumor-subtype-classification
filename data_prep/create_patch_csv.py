import os
import csv

# Root dataset path
DATA_ROOT = "/home/hpdeadman/Grad_Project/data"

# Folder-name to full-label mapping
LABEL_MAP = {
    "c": "chromophobe",
    "cc": "clearcell",
    "o": "oncocytoma",
    "p": "papillary"
}

# Output CSV path
OUTPUT_CSV = "/home/hpdeadman/Grad_Project/data/patch_metadata.csv"

rows = []

for split in ["train", "validate", "test"]:
    split_path = os.path.join(DATA_ROOT, split)

    if not os.path.isdir(split_path):
        print(f"Skipping missing split folder: {split_path}")
        continue

    for short_label, full_label in LABEL_MAP.items():
        class_path = os.path.join(split_path, short_label)

        if not os.path.isdir(class_path):
            print(f"Skipping missing class folder: {class_path}")
            continue

        for wsi_folder in sorted(os.listdir(class_path)):
            wsi_path = os.path.join(class_path, wsi_folder)

            if not os.path.isdir(wsi_path):
                continue

            for patch_file in sorted(os.listdir(wsi_path)):
                if patch_file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    patch_path = os.path.join(wsi_path, patch_file)

                    rows.append({
                        "patch_name": patch_file,
                        "wsi_id": wsi_folder,
                        "label": full_label,
                        "split": split,
                        "patch_path": patch_path
                    })

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["patch_name", "wsi_id", "label", "split", "patch_path"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV created: {OUTPUT_CSV}")
print(f"Total patches written: {len(rows)}")