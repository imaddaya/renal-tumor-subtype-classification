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
MODEL_PATH = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_model.pth"

NUM_CLASSES = 4
NUM_PATCHES = 70
NUM_RUNS = 10
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPORT_TXT = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_multisample_report.txt"
CM_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_multisample_confusion_matrix.csv"
PROBS_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_multisample_probabilities.csv"
CLASS_ERROR_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_multisample_class_error_rates.csv"
AVG_CLASS_ERROR_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_multisample_average_class_error_rates.csv"
PER_RUN_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18_MIL_Macenko/results/ResNet18_MIL_Macenko_multisample_per_run_predictions.csv"

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

LABELS = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

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
# Test Dataset / Loader
# -------------------------
test_dataset = WSIDatasetMacenko(
    csv_path=CSV_PATH,
    split="test",
    num_patches=NUM_PATCHES,
    transform=get_default_transform()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

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
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

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
        print(f"\nRunning test-time sampling pass {run_idx + 1}/{NUM_RUNS} ...")

        run_probs = []
        run_labels = []
        run_wsi_ids = []

        for images, labels, wsi_ids in test_loader:
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

        for i, class_name in enumerate(LABELS):
            row_sum = run_cm[i].sum()
            correct = run_cm[i, i]
            recall = correct / row_sum if row_sum > 0 else 0.0
            error_rate = 1.0 - recall if row_sum > 0 else 0.0

            per_run_error_rows.append({
                "run": run_idx + 1,
                "class": class_name,
                "total_true_samples": int(row_sum),
                "correct_predictions": int(correct),
                "recall": round(recall, 4),
                "error_rate": round(error_rate, 4),
                "error_rate_percent": round(error_rate * 100, 2)
            })

all_run_probs = np.stack(all_run_probs, axis=0)

# Average probabilities across repeated runs
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

cm = confusion_matrix(true_labels, final_preds)

print("\nMulti-sample Classification Report:")
print(report)
print("Multi-sample Confusion Matrix:")
print(cm)

# -------------------------
# Final class-specific error rates
# -------------------------
class_error_rows = []
for i, class_name in enumerate(LABELS):
    row_sum = cm[i].sum()
    correct = cm[i, i]
    recall = correct / row_sum if row_sum > 0 else 0.0
    error_rate = 1.0 - recall if row_sum > 0 else 0.0

    class_error_rows.append({
        "class": class_name,
        "total_true_samples": int(row_sum),
        "correct_predictions": int(correct),
        "recall": round(recall, 4),
        "error_rate": round(error_rate, 4),
        "error_rate_percent": round(error_rate * 100, 2)
    })

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

with open(CM_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true/pred"] + LABELS)
    for i, class_name in enumerate(LABELS):
        writer.writerow([class_name] + cm[i].tolist())

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