import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

HISTORY_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_history.csv"
REPORT_TXT = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_report.txt"

ACC_PLOT = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_accuracy_curve.png"
LOSS_PLOT = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_loss_curve.png"
CM_PLOT = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_confusion_matrix.png"
REPORT_TABLE_PLOT = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_classification_report_table.png"
HISTORY_TABLE_PLOT = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_history_table.png"

# -------------------------
# Load history CSV
# -------------------------
df = pd.read_csv(HISTORY_CSV)

# -------------------------
# Plot 1: Accuracy curve
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
plt.plot(df["epoch"], df["val_acc"], marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet18 Baseline Accuracy Curve")
plt.xticks(df["epoch"])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(ACC_PLOT)
plt.close()

# -------------------------
# Plot 2: Loss curve
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet18 Baseline Loss Curve")
plt.xticks(df["epoch"])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_PLOT)
plt.close()

# -------------------------
# Read TXT
# -------------------------
with open(REPORT_TXT, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# Parse classification report
# -------------------------
report_section = text.split("Classification Report:")[-1].split("Confusion Matrix:")[0]
report_lines = [line.rstrip() for line in report_section.splitlines() if line.strip()]

report_rows = []
for line in report_lines:
    stripped = line.strip()

    # skip header line
    if stripped.startswith("precision"):
        continue

    parts = stripped.split()

    # normal class rows: class precision recall f1 support
    if len(parts) == 5:
        class_name = parts[0]
        precision, recall, f1_score, support = parts[1:]
        report_rows.append([class_name, precision, recall, f1_score, support])

    # rows like "macro avg", "weighted avg"
    elif len(parts) == 6:
        class_name = parts[0] + " " + parts[1]
        precision, recall, f1_score, support = parts[2:]
        report_rows.append([class_name, precision, recall, f1_score, support])

# -------------------------
# Parse confusion matrix
# -------------------------
cm_text = text.split("Confusion Matrix:")[-1].strip()
rows = []
for line in cm_text.splitlines():
    line = line.strip().replace("[", "").replace("]", "")
    if line:
        row = [int(x) for x in line.split()]
        rows.append(row)

cm = np.array(rows)

# -------------------------
# Plot 3: Confusion matrix
# -------------------------
labels = ["chromophobe", "clearcell", "oncocytoma", "papillary"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, colorbar=False)
plt.title("ResNet18 Baseline Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_PLOT)
plt.close()

# -------------------------
# Plot 4: Classification report table
# -------------------------
if report_rows:
    report_df = pd.DataFrame(
        report_rows,
        columns=["Class", "Precision", "Recall", "F1-Score", "Support"]
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(
        cellText=report_df.values,
        colLabels=report_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title("ResNet18 Baseline Classification Report", pad=20)
    plt.tight_layout()
    plt.savefig(REPORT_TABLE_PLOT, bbox_inches="tight")
    plt.close()
else:
    print("Warning: classification report rows were empty, report table image was not created.")

# -------------------------
# Plot 5: History table
# -------------------------
history_display_df = df.copy()
history_display_df.columns = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
table = ax.table(
    cellText=history_display_df.values,
    colLabels=history_display_df.columns,
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.3)
plt.title("ResNet18 Baseline Training History", pad=20)
plt.tight_layout()
plt.savefig(HISTORY_TABLE_PLOT, bbox_inches="tight")
plt.close()

print("Saved:")
print(ACC_PLOT)
print(LOSS_PLOT)
print(CM_PLOT)
if report_rows:
    print(REPORT_TABLE_PLOT)
print(HISTORY_TABLE_PLOT)