from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --------------------------------------------------
# Portable paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Put all unzipped DHMC source files here
RAW_DATASET_ROOT = PROJECT_ROOT / "raw_dataset"

# Final output folder
DATA_ROOT = PROJECT_ROOT / "data"

PATCH_SIZE = 224
BACKGROUND_THRESHOLD = 235

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------
# Fixed split mapping
# --------------------------------------------------
TEST_SPLIT = {
    "c": {43, 44, 45, 46, 47},
    "cc": {52, 53, 54, 55, 56},
    "o": {24, 25, 26, 27, 28},
    "p": {72, 73, 74, 75, 76},
}

VALIDATE_SPLIT = {
    "c": {6, 7, 8, 9, 10},
    "cc": {11, 12, 13, 14, 15},
    "o": {1, 2, 3, 4, 5},
    "p": {19, 20, 21, 22, 23},
}

# Everything else goes to train
# --------------------------------------------------


def is_background(patch, threshold=BACKGROUND_THRESHOLD):
    gray = patch.convert("L")
    histogram = gray.histogram()
    pixels = sum(histogram)
    if pixels == 0:
        return True
    brightness = sum(i * w for i, w in enumerate(histogram)) / pixels
    return brightness > threshold


def parse_wsi_id_number(stem: str):
    """
    Example:
    DHMC_0001 -> 1
    DHMC_0043 -> 43
    """
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return None


def infer_class_from_number(wsi_num: int):
    """
    This uses your fixed class mapping logic.

    Known ranges from your split examples:
    - oncocytoma: low DHMC ids like 1..5 and 24..28
    - chromophobe: 6..10 and 43..47
    - clearcell: 11..15 and 52..56
    - papillary: 19..23 and 72..76

    For the rest, this function must be customized if your dataset contains
    more IDs outside these known groups and their class is not inferable from number alone.

    For now, this only supports the IDs that belong to your defined class mapping logic.
    """
    if wsi_num in TEST_SPLIT["c"] or wsi_num in VALIDATE_SPLIT["c"]:
        return "c"
    if wsi_num in TEST_SPLIT["cc"] or wsi_num in VALIDATE_SPLIT["cc"]:
        return "cc"
    if wsi_num in TEST_SPLIT["o"] or wsi_num in VALIDATE_SPLIT["o"]:
        return "o"
    if wsi_num in TEST_SPLIT["p"] or wsi_num in VALIDATE_SPLIT["p"]:
        return "p"

    # Everything else goes to train, but we still need the class.
    # You MUST extend this mapping rule if all remaining train IDs are not known by pattern.
    return None


def infer_split_and_class(wsi_num: int):
    for cls_name, nums in TEST_SPLIT.items():
        if wsi_num in nums:
            return "test", cls_name

    for cls_name, nums in VALIDATE_SPLIT.items():
        if wsi_num in nums:
            return "validate", cls_name

    cls_name = infer_class_from_number(wsi_num)
    if cls_name is not None:
        return "train", cls_name

    return None, None


def process_one_wsi(file_path: Path):
    slide_id = file_path.stem  # DHMC_0001
    wsi_num = parse_wsi_id_number(slide_id)

    if wsi_num is None:
        print(f"Skipping file with invalid name format: {file_path.name}")
        return

    split_name, class_name = infer_split_and_class(wsi_num)
    if split_name is None or class_name is None:
        print(f"Skipping {slide_id}: could not determine split/class")
        return

    save_dir = DATA_ROOT / split_name / class_name / slide_id
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

            print(f"Done: {slide_id} -> {split_name}/{class_name} -> {kept} patches")

    except Exception as e:
        print(f"Could not process {file_path.name}: {e}")


def process_dataset():
    if not RAW_DATASET_ROOT.exists():
        print(f"Error: raw dataset folder not found: {RAW_DATASET_ROOT}")
        return

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs"}

    all_files = []
    for file_path in RAW_DATASET_ROOT.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
            all_files.append(file_path)

    if not all_files:
        print("No valid source files found.")
        return

    print(f"Found {len(all_files)} source files.")

    for file_path in tqdm(all_files, desc="Slicing and splitting WSIs"):
        process_one_wsi(file_path)

    print(f"\nFinished. Output written into: {DATA_ROOT}")


if __name__ == "__main__":
    process_dataset()