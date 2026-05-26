import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader import WSIDataset, get_default_transform

# -------------------------
# Paths
# -------------------------
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "Models" / "ResNet18"
RESULTS_DIR = MODEL_DIR / "results"

# -------------------------
# Settings
# -------------------------
CSV_PATH = DATA_DIR / "wsi_metadata.csv"
BATCH_SIZE = 2
NUM_PATCHES = 70
NUM_CLASSES = 4
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = RESULTS_DIR / "ResNet18_model.pth"
HISTORY_CSV = RESULTS_DIR / "ResNet18_history.csv"
REPORT_TXT = RESULTS_DIR / "ResNet18_report.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IDX_TO_LABEL = {
    0: "chromophobe",
    1: "clearcell",
    2: "oncocytoma",
    3: "papillary"
}

print("Using device:", DEVICE)

# -------------------------
# Dataset / Loader
# -------------------------
train_dataset = WSIDataset(
    csv_path=CSV_PATH,
    split="train",
    num_patches=NUM_PATCHES,
    transform=get_default_transform()
)

val_dataset = WSIDataset(
    csv_path=CSV_PATH,
    split="validate",
    num_patches=NUM_PATCHES,
    transform=get_default_transform()
)

test_dataset = WSIDataset(
    csv_path=CSV_PATH,
    split="test",
    num_patches=NUM_PATCHES,
    transform=get_default_transform()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------
# Model: ResNet18 + mean pooling
# -------------------------
class SimpleWSIModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # x: [B, N, 3, 224, 224]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.backbone(x)          # [B*N, feat_dim]
        feats = feats.view(B, N, -1)      # [B, N, feat_dim]
        slide_feats = feats.mean(dim=1)   # mean pooling
        out = self.classifier(slide_feats)
        return out

model = SimpleWSIModel(num_classes=NUM_CLASSES).to(DEVICE)

# -------------------------
# Loss / Optimizer
# -------------------------
label_order = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

counts_series = train_dataset.df["label"].value_counts()
class_counts = torch.tensor(
    [float(counts_series.get(label_name, 0)) for label_name in label_order],
    dtype=torch.float32
)

if (class_counts == 0).any():
    raise ValueError(f"At least one class has zero samples in the training split: {class_counts.tolist()}")

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(DEVICE)

print("Training class counts:", class_counts.tolist())
print("Training class weights:", class_weights.tolist())

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# -------------------------
# idk
# -------------------------
def evaluate_model(model, loader, criterion, device, return_preds=False):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if return_preds:
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    if return_preds:
        return avg_loss, accuracy, all_labels, all_preds

    return avg_loss, accuracy

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

    val_loss_avg, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)

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

# save model
torch.save(model.state_dict(), MODEL_PATH)

# save history csv
with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    writer.writeheader()
    writer.writerows(history)

# -------------------------
# Test evaluation
# -------------------------
_, _, all_labels, all_preds = evaluate_model(
    model, test_loader, criterion, DEVICE, return_preds=True
)

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