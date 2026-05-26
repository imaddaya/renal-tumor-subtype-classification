from pathlib import Path
from PIL import Image
from tqdm import tqdm
import csv

# --------------------------------------------------
# Portable paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_ROOT = PROJECT_ROOT / "raw_dataset"
DATA_ROOT = PROJECT_ROOT / "data"
WSI_METADATA_CSV = DATA_ROOT / "wsi_metadata.csv"

PATCH_SIZE = 224
BACKGROUND_THRESHOLD = 235

Image.MAX_IMAGE_PIXELS = None


def is_background(patch, threshold=BACKGROUND_THRESHOLD):
    gray = patch.convert("L")
    histogram = gray.histogram()
    pixels = sum(histogram)

    if pixels == 0:
        return True

    brightness = sum(i * w for i, w in enumerate(histogram)) / pixels
    return brightness > threshold


def load_wsi_metadata(csv_path: Path):
    """
    Reads data/wsi_metadata.csv and builds a lookup:
    wsi_id -> row info
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    mapping = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"wsi_id", "label", "split", "patch_dir"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")

        for row in reader:
            wsi_id = row["wsi_id"].strip()
            mapping[wsi_id] = {
                "label": row["label"].strip(),
                "split": row["split"].strip(),
                "patch_dir": row["patch_dir"].strip(),
            }

    return mapping


def process_one_wsi(file_path: Path, mapping: dict):
    slide_id = file_path.stem  # example: DHMC_0001

    if slide_id not in mapping:
        print(f"Skipping {slide_id}: not found in data/wsi_metadata.csv")
        return

    row = mapping[slide_id]

    # patch_dir already looks like: train/c/DHMC_0001
    save_dir = DATA_ROOT / row["patch_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(save_dir.glob("*.png"))
    if len(existing_files) > 10:
        print(f"Skipping already processed WSI: {slide_id}")
        return

    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            kept = 0
            for y in range(0, h, PATCH_SIZE):
                for x in range(0, w, PATCH_SIZE):
                    if x + PATCH_SIZE <= w and y + PATCH_SIZE <= h:
                        patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

                        if not is_background(patch):
                            patch_name = f"img{kept + 1:06d}.png"
                            patch.save(save_dir / patch_name)
                            kept += 1

            print(
                f"Done: {slide_id} -> "
                f"{row['split']} / {row['label']} -> "
                f"{kept} patches"
            )

    except Exception as e:
        print(f"Could not process {file_path.name}: {e}")


def process_dataset():
    if not RAW_DATASET_ROOT.exists():
        print(f"Error: raw dataset folder not found: {RAW_DATASET_ROOT}")
        return

    mapping = load_wsi_metadata(WSI_METADATA_CSV)

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs"}

    all_files = []
    for file_path in RAW_DATASET_ROOT.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
            all_files.append(file_path)

    if not all_files:
        print("No valid source files found.")
        return

    print(f"Found {len(all_files)} source files.")
    print(f"Loaded {len(mapping)} WSI entries from {WSI_METADATA_CSV}")

    for file_path in tqdm(all_files, desc="Slicing and splitting WSIs"):
        process_one_wsi(file_path, mapping)

    print(f"\nFinished. Output written into: {DATA_ROOT}")


if __name__ == "__main__":
    process_dataset()