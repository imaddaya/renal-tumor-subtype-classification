import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# File paths
# -------------------------
CM_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_confusion_matrix.csv"
PROBS_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_probabilities.csv"
CLASS_ERROR_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_class_error_rates.csv"
AVG_CLASS_ERROR_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_average_class_error_rates.csv"
PER_RUN_CSV = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_per_run_predictions.csv"

CM_PNG = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_confusion_matrix.png"
PROBS_PNG = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_probabilities.png"
CLASS_ERROR_PNG = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_class_error_rates.png"
AVG_CLASS_ERROR_PNG = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_average_class_error_rates.png"
PER_RUN_PNG = "/home/hpdeadman/Grad_Project/Models/ResNet18/results/ResNet18_multisample_per_run_predictions.png"

# -------------------------
# 1) Confusion matrix image
# -------------------------
cm_df = pd.read_csv(CM_CSV)
labels = cm_df.iloc[:, 0].tolist()
cm = cm_df.iloc[:, 1:].values

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(len(labels)),
    yticks=np.arange(len(labels)),
    xticklabels=labels,
    yticklabels=labels,
    xlabel="Predicted label",
    ylabel="True label",
    title="ResNet18 Multi-Sample Confusion Matrix"
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, format(int(cm[i, j]), "d"),
            ha="center", va="center", color="black"
        )

fig.tight_layout()
plt.savefig(CM_PNG, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# 2) Final class error rates table image
# -------------------------
error_df = pd.read_csv(CLASS_ERROR_CSV)

fig, ax = plt.subplots(figsize=(10, 2 + len(error_df) * 0.5))
ax.axis("off")

table = ax.table(
    cellText=error_df.values,
    colLabels=error_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("ResNet18 Multi-Sample Final Class Error Rates", pad=20)
plt.savefig(CLASS_ERROR_PNG, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# 3) Average class error rates across runs table image
# -------------------------
avg_error_df = pd.read_csv(AVG_CLASS_ERROR_CSV)

fig, ax = plt.subplots(figsize=(10, 2 + len(avg_error_df) * 0.5))
ax.axis("off")

table = ax.table(
    cellText=avg_error_df.values,
    colLabels=avg_error_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("ResNet18 Multi-Sample Average Class Error Rates Across Runs", pad=20)
plt.savefig(AVG_CLASS_ERROR_PNG, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# 4) Averaged probabilities table image
# -------------------------
probs_df = pd.read_csv(PROBS_CSV)

fig, ax = plt.subplots(figsize=(16, 2 + len(probs_df) * 0.35))
ax.axis("off")

table = ax.table(
    cellText=probs_df.values,
    colLabels=probs_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.3)

plt.title("ResNet18 Multi-Sample Averaged Probabilities per WSI", pad=20)
plt.savefig(PROBS_PNG, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# 5) Per-run predictions table image
# -------------------------
per_run_df = pd.read_csv(PER_RUN_CSV)

fig, ax = plt.subplots(figsize=(18, 2 + len(per_run_df) * 0.22))
ax.axis("off")

table = ax.table(
    cellText=per_run_df.values,
    colLabels=per_run_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.15, 1.15)

plt.title("ResNet18 Per-Run Predictions (10 Test-Time Sampling Passes)", pad=20)
plt.savefig(PER_RUN_PNG, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print(CM_PNG)
print(CLASS_ERROR_PNG)
print(AVG_CLASS_ERROR_PNG)
print(PROBS_PNG)
print(PER_RUN_PNG)