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
import numpy as np

from dataset_loader import WSIDataset, get_default_transform

# -------------------------
# Paths
# -------------------------
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "Models" / "ResNet18_MIL_KAN"
RESULTS_DIR = MODEL_DIR / "results"

TRAINING_DIR = RESULTS_DIR / "training"
MULTISAMPLE_DIR = RESULTS_DIR / "multisample"

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
MULTISAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Settings
# -------------------------
CSV_PATH = DATA_DIR / "wsi_metadata.csv"
MODEL_PATH = TRAINING_DIR / "ResNet18_MIL_KAN_model.pth"

NUM_CLASSES = 4
NUM_PATCHES = 140
NUM_RUNS = 10
BATCH_SIZE = 1
NUM_WORKERS = 2

ATTN_DIM = 128
KAN_HIDDEN = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPORT_TXT = MULTISAMPLE_DIR / "ResNet18_MIL_KAN_multisample_report.txt"
CM_CSV = MULTISAMPLE_DIR / "ResNet18_MIL_KAN_multisample_confusion_matrix.csv"
PROBS_CSV = MULTISAMPLE_DIR / "ResNet18_MIL_KAN_multisample_probabilities.csv"
CLASS_ERROR_CSV = MULTISAMPLE_DIR / "ResNet18_MIL_KAN_multisample_class_error_rates.csv"
AVG_CLASS_ERROR_CSV = MULTISAMPLE_DIR / "ResNet18_MIL_KAN_multisample_average_class_error_rates.csv"
PER_RUN_CSV = MULTISAMPLE_DIR / "ResNet18_MIL_KAN_multisample_per_run_predictions.csv"

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
test_dataset = WSIDataset(
    csv_path=CSV_PATH,
    split="test",
    num_patches=NUM_PATCHES,
    transform=get_default_transform(),
    sampling_mode="random",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("\n=== Multi-sample test summary ===")
print(f"Test WSIs: {len(test_dataset)}")
print("Test WSI IDs:", test_dataset.df["wsi_id"].tolist())

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
        feats = self.backbone(x)                # [B*N, feat_dim]
        feats = feats.view(B, N, -1)            # [B, N, feat_dim]

        attn_scores = self.attention(feats)     # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)

        slide_feats = torch.sum(attn_weights * feats, dim=1)
        out = self.classifier(slide_feats)
        return out


model = AttentionMILKANModel(
    num_classes=NUM_CLASSES,
    attn_dim=ATTN_DIM,
    kan_hidden=KAN_HIDDEN
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Helpers
# -------------------------
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
# Multi-sample inference
# -------------------------
all_run_probs = []
true_labels_reference = []
wsi_ids_reference = []
per_run_rows = []
per_run_error_rows = []

with torch.no_grad():
    for run_idx in range(NUM_RUNS):
        print(f"\n=== Running test-time sampling pass {run_idx + 1}/{NUM_RUNS} ===")

        run_probs = []
        run_labels = []
        run_wsi_ids = []

        for batch_idx, (images, labels, wsi_ids) in enumerate(test_loader):
            if ((batch_idx + 1) % 10 == 0) or ((batch_idx + 1) == len(test_loader)):
                print(f"[Run {run_idx + 1}][Batch {batch_idx + 1}/{len(test_loader)}] WSI IDs: {list(wsi_ids)}")

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            run_probs.append(probs.cpu().numpy())
            run_labels.extend(labels.cpu().numpy())
            run_wsi_ids.extend(wsi_ids)

            for b in range(len(wsi_ids)):
                per_run_rows.append({
                    "run": run_idx + 1,
                    "wsi_id": wsi_ids[b],
                    "true_label": IDX_TO_LABEL[int(labels[b].cpu().item())],
                    "predicted_label": IDX_TO_LABEL[int(preds[b].cpu().item())],
                    "prob_chromophobe": round(float(probs[b, 0].cpu().item()), 6),
                    "prob_clearcell": round(float(probs[b, 1].cpu().item()), 6),
                    "prob_oncocytoma": round(float(probs[b, 2].cpu().item()), 6),
                    "prob_papillary": round(float(probs[b, 3].cpu().item()), 6),
                })

        run_probs = np.concatenate(run_probs, axis=0)
        all_run_probs.append(run_probs)

        if run_idx == 0:
            true_labels_reference = run_labels
            wsi_ids_reference = run_wsi_ids

        run_preds = run_probs.argmax(axis=1)
        run_labels_np = np.array(run_labels)
        run_cm = confusion_matrix(run_labels_np, run_preds, labels=list(range(NUM_CLASSES)))

        run_error_rows = build_class_error_rows(run_cm, LABELS)
        for row in run_error_rows:
            row["run"] = run_idx + 1
            per_run_error_rows.append(row)

all_run_probs = np.stack(all_run_probs, axis=0)

# -------------------------
# Average probabilities across repeated runs
# -------------------------
avg_probs = all_run_probs.mean(axis=0)
final_preds = avg_probs.argmax(axis=1)
true_labels = np.array(true_labels_reference)

# -------------------------
# Final metrics from averaged probabilities
# -------------------------
report = classification_report(
    true_labels,
    final_preds,
    target_names=LABELS,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(true_labels, final_preds, labels=list(range(NUM_CLASSES)))
class_error_rows = build_class_error_rows(cm, LABELS)

print("\nMulti-sample Classification Report:")
print(report)
print("Multi-sample Confusion Matrix:")
print(cm)

print("\nFinal class-specific error rates (from averaged predictions):")
for row in class_error_rows:
    print(
        f"{row['class']}: "
        f"recall={row['recall']:.4f}, "
        f"error_rate={row['error_rate']:.4f} "
        f"({row['error_rate_percent']:.2f}%)"
    )

# -------------------------
# Average class error rates across runs
# -------------------------
avg_class_error_rows = []
for class_name in LABELS:
    class_rows = [row for row in per_run_error_rows if row["class"] == class_name]

    avg_recall = sum(row["recall"] for row in class_rows) / len(class_rows)
    avg_error_rate = sum(row["error_rate"] for row in class_rows) / len(class_rows)
    avg_error_rate_percent = avg_error_rate * 100

    avg_class_error_rows.append({
        "class": class_name,
        "avg_recall_across_runs": round(avg_recall, 4),
        "avg_error_rate_across_runs": round(avg_error_rate, 4),
        "avg_error_rate_percent_across_runs": round(avg_error_rate_percent, 2)
    })

print("\nAverage class-specific error rates across runs:")
for row in avg_class_error_rows:
    print(
        f"{row['class']}: "
        f"avg_recall={row['avg_recall_across_runs']:.4f}, "
        f"avg_error_rate={row['avg_error_rate_across_runs']:.4f} "
        f"({row['avg_error_rate_percent_across_runs']:.2f}%)"
    )

# -------------------------
# Save report
# -------------------------
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(f"Repeated test-time sampling runs per WSI: {NUM_RUNS}\n")
    f.write(f"Number of patches sampled per WSI per run: {NUM_PATCHES}\n\n")

    f.write("Multi-sample Classification Report:\n")
    f.write(report)

    f.write("\nMulti-sample Confusion Matrix:\n")
    f.write(str(cm))

    f.write("\n\nFinal Class-specific Error Rates (from averaged predictions):\n")
    for row in class_error_rows:
        f.write(
            f"{row['class']}: "
            f"recall={row['recall']:.4f}, "
            f"error_rate={row['error_rate']:.4f} "
            f"({row['error_rate_percent']:.2f}%)\n"
        )

    f.write("\nAverage Class-specific Error Rates Across Runs:\n")
    for row in avg_class_error_rows:
        f.write(
            f"{row['class']}: "
            f"avg_recall={row['avg_recall_across_runs']:.4f}, "
            f"avg_error_rate={row['avg_error_rate_across_runs']:.4f} "
            f"({row['avg_error_rate_percent_across_runs']:.2f}%)\n"
        )

# -------------------------
# Save confusion matrix CSV
# -------------------------
with open(CM_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true/pred"] + LABELS)
    for i, class_name in enumerate(LABELS):
        writer.writerow([class_name] + cm[i].tolist())

# -------------------------
# Save averaged probabilities per WSI
# -------------------------
with open(PROBS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "wsi_id",
        "true_label",
        "predicted_label",
        "prob_chromophobe",
        "prob_clearcell",
        "prob_oncocytoma",
        "prob_papillary"
    ])

    for i, wsi_id in enumerate(wsi_ids_reference):
        writer.writerow([
            wsi_id,
            IDX_TO_LABEL[int(true_labels[i])],
            IDX_TO_LABEL[int(final_preds[i])],
            round(float(avg_probs[i, 0]), 6),
            round(float(avg_probs[i, 1]), 6),
            round(float(avg_probs[i, 2]), 6),
            round(float(avg_probs[i, 3]), 6),
        ])

# -------------------------
# Save final class error rates CSV
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

# -------------------------
# Save average class error rates CSV
# -------------------------
with open(AVG_CLASS_ERROR_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "avg_recall_across_runs",
            "avg_error_rate_across_runs",
            "avg_error_rate_percent_across_runs"
        ]
    )
    writer.writeheader()
    writer.writerows(avg_class_error_rows)

# -------------------------
# Save per-run predictions CSV
# -------------------------
with open(PER_RUN_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run",
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
    writer.writerows(per_run_rows)

print("\nSaved:")
print(REPORT_TXT)
print(CM_CSV)
print(PROBS_CSV)
print(CLASS_ERROR_CSV)
print(AVG_CLASS_ERROR_CSV)
print(PER_RUN_CSV)