import random
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_TO_IDX = {
    "chromophobe": 0,
    "clearcell": 1,
    "oncocytoma": 2,
    "papillary": 3,
}

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"


def _stable_seed(wsi_id: str, fixed_seed: int) -> int:
    text = f"{wsi_id}_{fixed_seed}"
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _white_to_weight(white_percent: float) -> float:
    if white_percent <= 40:
        return 1.0
    elif white_percent <= 60:
        return 0.7
    elif white_percent <= 75:
        return 0.4
    elif white_percent <= 85:
        return 0.15
    elif white_percent <= 95:
        return 0.05
    else:
        return 0.0


def _weighted_sample_without_replacement(items, weights, k, rng):
    """
    Weighted sampling without replacement.
    If k >= len(items), returns all items in weighted-random order.
    """
    items = list(items)
    weights = list(weights)

    chosen = []
    k = min(k, len(items))

    for _ in range(k):
        positive_weight_sum = sum(w for w in weights if w > 0)

        # If all remaining weights are zero, fall back to uniform choice
        if positive_weight_sum <= 0:
            idx = rng.randrange(len(items))
        else:
            idx = rng.choices(range(len(items)), weights=weights, k=1)[0]

        chosen.append(items.pop(idx))
        weights.pop(idx)

        if len(items) == 0:
            break

    return chosen


def _weighted_sample_with_replacement(items, weights, k, rng):
    """
    Weighted sampling with replacement.
    If all weights are zero, falls back to uniform sampling.
    """
    items = list(items)
    weights = list(weights)

    positive_weight_sum = sum(w for w in weights if w > 0)

    if positive_weight_sum <= 0:
        return rng.choices(items, k=k)

    return rng.choices(items, weights=weights, k=k)


class WSIDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split="train",
        num_patches=32,
        transform=None,
        sampling_mode="random",
        fixed_seed=42,
        patch_quality_csv=None,
        use_patch_quality_weights=False,
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.num_patches = num_patches
        self.transform = transform
        self.sampling_mode = sampling_mode
        self.fixed_seed = fixed_seed

        self.patch_quality_csv = patch_quality_csv
        self.use_patch_quality_weights = use_patch_quality_weights

        if self.sampling_mode not in ["random", "fixed"]:
            raise ValueError("sampling_mode must be either 'random' or 'fixed'")

        self.patch_white_map = {}
        if self.use_patch_quality_weights:
            if self.patch_quality_csv is None:
                raise ValueError(
                    "use_patch_quality_weights=True requires patch_quality_csv to be provided"
                )
            self.patch_white_map = self._load_patch_quality_map(self.patch_quality_csv)

    def _load_patch_quality_map(self, patch_quality_csv):
        patch_quality_csv = Path(patch_quality_csv)
        if not patch_quality_csv.exists():
            raise FileNotFoundError(f"Patch quality CSV not found: {patch_quality_csv}")

        quality_df = pd.read_csv(patch_quality_csv)

        required_cols = {"wsi_id", "patch_name", "white_percent"}
        missing = required_cols - set(quality_df.columns)
        if missing:
            raise ValueError(
                f"Patch quality CSV is missing required columns: {sorted(missing)}"
            )

        white_map = {}
        for _, row in quality_df.iterrows():
            key = (str(row["wsi_id"]), str(row["patch_name"]))
            white_map[key] = float(row["white_percent"])

        return white_map

    def _get_patch_weight(self, wsi_id, patch_path: Path):
        if not self.use_patch_quality_weights:
            return 1.0

        key = (str(wsi_id), patch_path.name)
        white_percent = self.patch_white_map.get(key, None)

        # If patch not found in audit CSV, keep it usable with neutral weight
        if white_percent is None:
            return 1.0

        return _white_to_weight(white_percent)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wsi_id = row["wsi_id"]
        label_name = row["label"]
        patch_dir = DATA_ROOT / row["patch_dir"]

        if not patch_dir.exists():
            raise FileNotFoundError(f"Patch directory not found: {patch_dir}")

        if label_name not in LABEL_TO_IDX:
            raise ValueError(f"Unknown label '{label_name}' for WSI {wsi_id}")

        label = LABEL_TO_IDX[label_name]

        patch_files = sorted([
            f for f in patch_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ])

        if len(patch_files) == 0:
            raise ValueError(f"No patch images found in {patch_dir}")

        patch_weights = [self._get_patch_weight(wsi_id, p) for p in patch_files]

        if self.sampling_mode == "random":
            rng = random
        else:
            rng = random.Random(_stable_seed(wsi_id, self.fixed_seed))

        if len(patch_files) >= self.num_patches:
            if self.use_patch_quality_weights:
                chosen = _weighted_sample_without_replacement(
                    patch_files,
                    patch_weights,
                    self.num_patches,
                    rng,
                )
            else:
                chosen = rng.sample(patch_files, self.num_patches)
        else:
            if self.use_patch_quality_weights:
                chosen = _weighted_sample_with_replacement(
                    patch_files,
                    patch_weights,
                    self.num_patches,
                    rng,
                )
            else:
                chosen = rng.choices(patch_files, k=self.num_patches)

        images = []
        for patch_path in chosen:
            image = Image.open(patch_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            images.append(image)

        images = torch.stack(images)
        label = torch.tensor(label, dtype=torch.long)

        return images, label, wsi_id


def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])