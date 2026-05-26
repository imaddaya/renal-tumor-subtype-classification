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


class WSIDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split="train",
        num_patches=32,
        transform=None,
        sampling_mode="random",
        fixed_seed=42,
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.num_patches = num_patches
        self.transform = transform
        self.sampling_mode = sampling_mode
        self.fixed_seed = fixed_seed

        if self.sampling_mode not in ["random", "fixed"]:
            raise ValueError("sampling_mode must be either 'random' or 'fixed'")

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

        if self.sampling_mode == "random":
            if len(patch_files) >= self.num_patches:
                chosen = random.sample(patch_files, self.num_patches)
            else:
                chosen = random.choices(patch_files, k=self.num_patches)

        else:  # fixed
            rng = random.Random(_stable_seed(wsi_id, self.fixed_seed))
            if len(patch_files) >= self.num_patches:
                chosen = rng.sample(patch_files, self.num_patches)
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