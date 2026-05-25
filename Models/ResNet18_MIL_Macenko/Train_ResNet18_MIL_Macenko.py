import csv
import sys
sys.path.append("/home/hpdeadman/Grad_Project")

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import torchstain

# -------------------------
# Settings
# -------------------------
CSV_PATH = "/home/hpdeadman/Grad_Project/data/wsi_metadata.csv"
TARGET_IMAGE_PATH = "/home/hpdeadman/Grad_Project/data/train/c/DHMC_0040/p_2688_3808.jpg"  
BATCH_SIZE = 2
NUM_PATCHES = 70
NUM_CLASSES = 4
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_model.pth"
HISTORY_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_history.csv"
REPORT_TXT = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_report.txt"

IDX_TO_LABEL = {
    0: "chromophobe",
    1: "clearcell",
    2: "oncocytoma",
    3: "papillary"
}

LABEL_TO_IDX = {
    "chromophobe": 0,
    "clearcell": 1,
    "oncocytoma": 2,
    "papillary": 3,
}

print("Using device:", DEVICE)

# -------------------------
# Macenko normalizer
# -------------------------
macenko_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")

target_img = Image.open(TARGET_IMAGE_PATH).convert("RGB")
target_tensor = macenko_transform(target_img)
normalizer.fit(target_tensor)

def macenko_normalize_pil(pil_img):
    src = macenko_transform(pil_img)
    norm = normalizer.normalize(src)[0]

    # torchstain may return CHW or HWC depending on backend/version
    if isinstance(norm, torch.Tensor):
        arr = norm.detach().cpu().numpy()
    else:
        arr = np.array(norm)

    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# -------------------------
# Dataset with Macenko
# -------------------------
class WSIDatasetMacenko(Dataset):
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
        patch_dir = row["patch_dir"]

        label = LABEL_TO_IDX[label_name]

        patch_files = [
            f for f in os.listdir(patch_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]

        if len(patch_files) == 0:
            raise ValueError(f"No patch images found in {patch_dir}")

        if len(patch_files) >= self.num_patches:
            chosen = random.sample(patch_files, self.num_patches)
        else:
            chosen = random.choices(patch_files, k=self.num_patches)

        images = []
        for patch_file in chosen:
            patch_path = os.path.join(patch_dir, patch_file)
            image = Image.open(patch_path).convert("RGB")

            try:
                image = macenko_normalize_pil(image)
            except Exception:
                pass

            if self.transform:
                image = self.transform(image)

            images.append(image)

        images = torch.stack(images)
        label = torch.tensor(label, dtype=torch.long)

        return images, label, wsi_id

# -------------------------
# Dataset / Loader
# -------------------------
train_dataset = WSIDatasetMacenko(
    csv_path=CSV_PATH,
    split="train",
    num_patches=NUM_PATCHES,
    transform=get_default_transform()
)

val_dataset = WSIDatasetMacenko(
    csv_path=CSV_PATH,
    split="validate",
    num_patches=NUM_PATCHES,
    transform=get_default_transform()
)

test_dataset = WSIDatasetMacenko(
    csv_path=CSV_PATH,
    split="test",
    num_patches=NUM_PATCHES,
    transform=get_default_transform()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------
# Model: ResNet18 + Attention MIL
# -------------------------
class AttentionMILModel(nn.Module):
    def __init__(self, num_classes=4, attn_dim=128):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.attention = nn.Sequential(
            nn.Linear(feat_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, N, -1)

        attn_scores = self.attention(feats)
        attn_weights = torch.softmax(attn_scores, dim=1)

        slide_feats = torch.sum(attn_weights * feats, dim=1)
        out = self.classifier(slide_feats)
        return out

model = AttentionMILModel(num_classes=NUM_CLASSES).to(DEVICE)

# -------------------------
# Loss / Optimizer
# -------------------------
class_counts = torch.tensor([13, 331, 85, 91], dtype=torch.float32)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training
# -------------------------
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels, _ in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss_avg = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss_avg = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss_avg,
        "train_acc": train_acc,
        "val_loss": val_loss_avg,
        "val_acc": val_acc,
    })

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss_avg:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss_avg:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

torch.save(model.state_dict(), MODEL_PATH)

with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    writer.writeheader()
    writer.writerows(history)

# -------------------------
# Test evaluation
# -------------------------
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

target_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n")

print("\nSaved:")
print(MODEL_PATH)
print(HISTORY_CSV)
print(REPORT_TXT)