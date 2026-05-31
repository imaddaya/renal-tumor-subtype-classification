import csv
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "wsi_metadata.csv"
OUTPUT_CSV = PROJECT_ROOT / "patch_whiteness_audit.csv"

# -------------------------
# Settings
# -------------------------
# White threshold:
# A pixel is counted as "white/background" if ALL RGB channels are above this value.
WHITE_THRESHOLD = 220

# Optional split filter:
# Set to None to scan everything
# or use "train", "validate", "test"
SPLIT_FILTER = None

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# -------------------------
# Helpers
# -------------------------
def compute_white_percent(image_path: Path, white_threshold: int = 220):
    image = Image.open(image_path).convert("RGB")
    arr = np.array(image)

    white_mask = np.all(arr >= white_threshold, axis=2)
    white_percent = float(white_mask.mean() * 100.0)
    tissue_percent = 100.0 - white_percent

    return round(white_percent, 2), round(tissue_percent, 2)


# -------------------------
# Main
# -------------------------
def main():
    df = pd.read_csv(CSV_PATH)

    if SPLIT_FILTER is not None:
        df = df[df["split"] == SPLIT_FILTER].reset_index(drop=True)

    total_wsis = len(df)
    total_rows_written = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "label",
            "wsi_id",
            "patch_name",
            "patch_path",
            "white_percent",
            "tissue_percent"
        ])

        for wsi_idx, row in df.iterrows():
            split = row["split"]
            label = row["label"]
            wsi_id = row["wsi_id"]
            patch_dir = DATA_DIR / row["patch_dir"]

            if not patch_dir.exists():
                print(f"[WARNING] Missing patch directory for {wsi_id}: {patch_dir}")
                continue

            patch_files = sorted([
                p for p in patch_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTS
            ])

            if len(patch_files) == 0:
                print(f"[WARNING] No patch files found for {wsi_id}: {patch_dir}")
                continue

            print(f"[{wsi_idx + 1}/{total_wsis}] Scanning {wsi_id} | patches: {len(patch_files)}")

            for patch_path in patch_files:
                try:
                    white_percent, tissue_percent = compute_white_percent(
                        patch_path,
                        white_threshold=WHITE_THRESHOLD
                    )

                    writer.writerow([
                        split,
                        label,
                        wsi_id,
                        patch_path.name,
                        str(patch_path),
                        white_percent,
                        tissue_percent
                    ])
                    total_rows_written += 1

                except Exception as e:
                    print(f"[WARNING] Failed on {patch_path}: {e}")

    print("\nDone.")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Rows written: {total_rows_written}")


if __name__ == "__main__":
    main()