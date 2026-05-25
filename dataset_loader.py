import random
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


class WSIDataset(Dataset):
    def __init__(self, csv_path, split="train", num_patches=32, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.num_patches = num_patches
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wsi_id = row["wsi_id"]
        label_name = row["label"]
        patch_dir = DATA_ROOT / row["patch_dir"]

        label = LABEL_TO_IDX[label_name]

        patch_files = [
            f for f in patch_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ]

        if len(patch_files) == 0:
            raise ValueError(f"No patch images found in {patch_dir}")

        if len(patch_files) >= self.num_patches:
            chosen = random.sample(patch_files, self.num_patches)
        else:
            chosen = random.choices(patch_files, k=self.num_patches)

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