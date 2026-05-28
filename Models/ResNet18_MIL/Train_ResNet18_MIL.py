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
MODEL_DIR = PROJECT_ROOT / "Models" / "ResNet18_MIL"
RESULTS_DIR = MODEL_DIR / "results"

TRAINING_DIR = RESULTS_DIR / "training"
FIXED_TEST_DIR = RESULTS_DIR / "fixed_test"

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
FIXED_TEST_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Settings
# -------------------------
CSV_PATH = DATA_DIR / "wsi_metadata.csv"

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
NUM_WORKERS = 2

TRAIN_PATCHES = 150
VAL_PATCHES = 400
TEST_PATCHES = 140

NUM_CLASSES = 4
EPOCHS = 20
LR = 1e-4
FIXED_SEED = 42
ATTN_DIM = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = TRAINING_DIR / "ResNet18_MIL_model.pth"
HISTORY_CSV = TRAINING_DIR / "ResNet18_MIL_history.csv"

REPORT_TXT = FIXED_TEST_DIR / "ResNet18_MIL_report.txt"
CM_CSV = FIXED_TEST_DIR / "ResNet18_MIL_confusion_matrix.csv"
PROBS_CSV = FIXED_TEST_DIR / "ResNet18_MIL_probabilities.csv"
CLASS_ERROR_CSV = FIXED_TEST_DIR / "ResNet18_MIL_class_error_rates.csv"

IDX_TO_LABEL = {
    0: "chromophobe",
    1: "clearcell",
    2: "oncocytoma",
    3: "papillary"
}

LABELS = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

print("Using device:", DEVICE)

# -------------------------
# Dataset / Loader
# -------------------------
train_dataset = WSIDataset(
    csv_path=CSV_PATH,
    split="train",
    num_patches=TRAIN_PATCHES,
    transform=get_default_transform(),
    sampling_mode="random",
)

val_dataset = WSIDataset(
    csv_path=CSV_PATH,
    split="validate",
    num_patches=VAL_PATCHES,
    transform=get_default_transform(),
    sampling_mode="fixed",
    fixed_seed=FIXED_SEED,
)

test_dataset = WSIDataset(
    csv_path=CSV_PATH,
    split="test",
    num_patches=TEST_PATCHES,
    transform=get_default_transform(),
    sampling_mode="fixed",
    fixed_seed=FIXED_SEED,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("\n=== Split summary ===")
print(f"Train WSIs: {len(train_dataset)}")
print(f"Validation WSIs: {len(val_dataset)}")
print(f"Test WSIs: {len(test_dataset)}")

print("\nTrain WSI IDs (first 10):", train_dataset.df["wsi_id"].tolist()[:10])
print("Validation WSI IDs:", val_dataset.df["wsi_id"].tolist())
print("Test WSI IDs:", test_dataset.df["wsi_id"].tolist())

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
        # x: [B, N, 3, 224, 224]
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        feats = self.backbone(x)                # [B*N, feat_dim]
        feats = feats.view(B, N, -1)            # [B, N, feat_dim]

        attn_scores = self.attention(feats)     # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)

        slide_feats = torch.sum(attn_weights * feats, dim=1)  # [B, feat_dim]
        out = self.classifier(slide_feats)
        return out


model = AttentionMILModel(num_classes=NUM_CLASSES, attn_dim=ATTN_DIM).to(DEVICE)

# -------------------------
# Loss / Optimizer
# -------------------------
counts_series = train_dataset.df["label"].value_counts()
class_counts = torch.tensor(
    [float(counts_series.get(label_name, 0)) for label_name in LABELS],
    dtype=torch.float32
)

if (class_counts == 0).any():
    raise ValueError(
        f"At least one class has zero samples in the training split: {class_counts.tolist()}"
    )

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(DEVICE)

print("\nTraining class counts:", class_counts.tolist())
print("Training class weights:", class_weights.tolist())

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Helpers
# -------------------------
def evaluate_model(model, loader, criterion, device, phase_name="", print_every=None):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels, wsi_ids) in enumerate(loader):
            if print_every is not None:
                if ((batch_idx + 1) % print_every == 0) or ((batch_idx + 1) == len(loader)):
                    print(f"[{phase_name}][Batch {batch_idx + 1}/{len(loader)}] WSI IDs: {list(wsi_ids)}")

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def build_class_error_rows(cm, labels):
    rows = []
    for i, class_name in enumerate(labels):
        row_sum = cm[i].sum()
        correct = cm[i, i]
        recall = correct / row_sum if row_sum > 0 else 0.0
        error_rate = 1.0 - recall if row_sum > 0 else 0.0

        rows.append({
            "class": class_name,
            "total_true_samples": int(row_sum),
            "correct_predictions": int(correct),
            "recall": round(recall, 4),
            "error_rate": round(error_rate, 4),
            "error_rate_percent": round(error_rate * 100, 2)
        })
    return rows

# -------------------------
# Training
# -------------------------
history = []

best_val_acc = -1.0
best_epoch = -1

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels, wsi_ids) in enumerate(train_loader):
        if ((batch_idx + 1) % 100 == 0) or ((batch_idx + 1) == len(train_loader)):
            print(
                f"[Train][Epoch {epoch + 1}][Batch {batch_idx + 1}/{len(train_loader)}] "
                f"WSI IDs: {list(wsi_ids)}"
            )

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

    val_loss_avg, val_acc = evaluate_model(
        model,
        val_loader,
        criterion,
        DEVICE,
        phase_name="Validation",
        print_every=10
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss_avg,
        "train_acc": train_acc,
        "val_loss": val_loss_avg,
        "val_acc": val_acc,
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved new best model at epoch {best_epoch} with val_acc={best_val_acc:.4f}")

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss_avg:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss_avg:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

print(f"\nBest model was from epoch {best_epoch} with validation accuracy {best_val_acc:.4f}")

# -------------------------
# Save history CSV
# -------------------------
with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    )
    writer.writeheader()
    writer.writerows(history)

# -------------------------
# Load best model before final test
# -------------------------
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Fixed test evaluation
# -------------------------
print("\n=== Fixed test evaluation ===")

all_labels = []
all_preds = []
prob_rows = []

with torch.no_grad():
    for batch_idx, (images, labels, wsi_ids) in enumerate(test_loader):
        if ((batch_idx + 1) % 10 == 0) or ((batch_idx + 1) == len(test_loader)):
            print(f"[Test][Batch {batch_idx + 1}/{len(test_loader)}] WSI IDs: {list(wsi_ids)}")

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        for b in range(len(wsi_ids)):
            prob_rows.append({
                "wsi_id": wsi_ids[b],
                "true_label": IDX_TO_LABEL[int(labels[b].cpu().item())],
                "predicted_label": IDX_TO_LABEL[int(preds[b].cpu().item())],
                "prob_chromophobe": round(float(probs[b, 0].cpu().item()), 6),
                "prob_clearcell": round(float(probs[b, 1].cpu().item()), 6),
                "prob_oncocytoma": round(float(probs[b, 2].cpu().item()), 6),
                "prob_papillary": round(float(probs[b, 3].cpu().item()), 6),
            })

report = classification_report(
    all_labels,
    all_preds,
    target_names=LABELS,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
class_error_rows = build_class_error_rows(cm, LABELS)

print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

print("\nClass-specific Error Rates:")
for row in class_error_rows:
    print(
        f"{row['class']}: "
        f"recall={row['recall']:.4f}, "
        f"error_rate={row['error_rate']:.4f} "
        f"({row['error_rate_percent']:.2f}%)"
    )

# -------------------------
# Save main report TXT
# -------------------------
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n")

# -------------------------
# Save confusion matrix CSV
# -------------------------
with open(CM_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true/pred"] + LABELS)
    for i, class_name in enumerate(LABELS):
        writer.writerow([class_name] + cm[i].tolist())

# -------------------------
# Save fixed-test probabilities CSV
# -------------------------
with open(PROBS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "wsi_id",
            "true_label",
            "predicted_label",
            "prob_chromophobe",
            "prob_clearcell",
            "prob_oncocytoma",
            "prob_papillary"
        ]
    )
    writer.writeheader()
    writer.writerows(prob_rows)

# -------------------------
# Save class error rates CSV
# -------------------------
with open(CLASS_ERROR_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "total_true_samples",
            "correct_predictions",
            "recall",
            "error_rate",
            "error_rate_percent"
        ]
    )
    writer.writeheader()
    writer.writerows(class_error_rows)

print("\nSaved:")
print(MODEL_PATH)
print(HISTORY_CSV)
print(REPORT_TXT)
print(CM_CSV)
print(PROBS_CSV)
print(CLASS_ERROR_CSV)