import os
import csv

# Root dataset path
DATA_ROOT = "/home/hpdeadman/Grad_Project/data"

# Your short class names mapped to full names
LABEL_MAP = {
    "c": "chromophobe",
    "cc": "clearcell",
    "o": "oncocytoma",
    "p": "papillary"
}

# Output CSV path
OUTPUT_CSV = "/home/hpdeadman/Grad_Project/data/wsi_metadata.csv"

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

            # Count image patches inside the WSI folder
            patch_files = [
                f for f in os.listdir(wsi_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ]

            rows.append({
                "wsi_id": wsi_folder,
                "label": full_label,
                "split": split,
                "patch_dir": wsi_path,
                "num_patches": len(patch_files)
            })

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["wsi_id", "label", "split", "patch_dir", "num_patches"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV created: {OUTPUT_CSV}")
print(f"Total WSIs written: {len(rows)}")