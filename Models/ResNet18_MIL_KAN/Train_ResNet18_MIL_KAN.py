import csv
import sys
sys.path.append("/home/hpdeadman/Grad_Project")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader import WSIDataset, get_default_transform

# -------------------------
# Settings
# -------------------------
CSV_PATH = "/home/hpdeadman/Grad_Project/data/wsi_metadata.csv"
BATCH_SIZE = 2
NUM_PATCHES = 70
NUM_CLASSES = 4
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_KAN/results/ResNet18_MIL_KAN_model.pth"
HISTORY_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_KAN/results/ResNet18_MIL_KAN_history.csv"
REPORT_TXT = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_KAN/results/ResNet18_MIL_KAN_report.txt"

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
# Simple KAN-style layer
# -------------------------
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.poly2 = nn.Linear(in_features, out_features, bias=False)
        self.poly3 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.base(x) + self.poly2(x ** 2) + self.poly3(x ** 3)

class KANClassifier(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.layer1 = KANLinear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.act = nn.GELU()
        self.layer2 = KANLinear(hidden_features, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x

# -------------------------
# Model: ResNet18 + Attention MIL + KAN head
# -------------------------
class AttentionMILKANModel(nn.Module):
    def __init__(self, num_classes=4, attn_dim=128, kan_hidden=128):
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

        self.classifier = KANClassifier(
            in_features=feat_dim,
            hidden_features=kan_hidden,
            num_classes=num_classes
        )

    def forward(self, x):
        # x: [B, N, 3, 224, 224]
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        feats = self.backbone(x)          # [B*N, feat_dim]
        feats = feats.view(B, N, -1)      # [B, N, feat_dim]

        attn_scores = self.attention(feats)              # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1) # [B, N, 1]

        slide_feats = torch.sum(attn_weights * feats, dim=1)  # [B, feat_dim]
        out = self.classifier(slide_feats)
        return out

model = AttentionMILKANModel(num_classes=NUM_CLASSES).to(DEVICE)

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