import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "Models" / "ResNet18"
RESULTS_DIR = MODEL_DIR / "results"

TRAINING_DIR = RESULTS_DIR / "training"
FIXED_TEST_DIR = RESULTS_DIR / "fixed_test"
MULTISAMPLE_DIR = RESULTS_DIR / "multisample"
VISUALS_DIR = RESULTS_DIR / "visuals"

VISUALS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV = TRAINING_DIR / "ResNet18_history.csv"

FIXED_REPORT_TXT = FIXED_TEST_DIR / "ResNet18_report.txt"
FIXED_CM_CSV = FIXED_TEST_DIR / "ResNet18_confusion_matrix.csv"
FIXED_CLASS_ERROR_CSV = FIXED_TEST_DIR / "ResNet18_class_error_rates.csv"

MULTI_CM_CSV = MULTISAMPLE_DIR / "ResNet18_multisample_confusion_matrix.csv"

COMBINED_GRAPH_PNG = VISUALS_DIR / "ResNet18_training_overview.png"
REPORT_TABLE_PNG = VISUALS_DIR / "ResNet18_classification_report_table.png"
DETAILED_ERROR_TABLE_PNG = VISUALS_DIR / "ResNet18_detailed_error_table.png"
FIXED_CM_PNG = VISUALS_DIR / "ResNet18_fixed_confusion_matrix.png"
MULTI_CM_PNG = VISUALS_DIR / "ResNet18_multisample_confusion_matrix.png"
SUMMARY_TABLE_PNG = VISUALS_DIR / "ResNet18_summary_table.png"

LABELS = ["chromophobe", "clearcell", "oncocytoma", "papillary"]
SHORT_LABELS = ["C", "CC", "O", "P"]

# -------------------------
# Helpers
# -------------------------
def parse_classification_report_txt(report_txt_path: Path) -> pd.DataFrame:
    with open(report_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    report_section = text.split("Classification Report:")[-1].split("Confusion Matrix:")[0]
    report_lines = [line.rstrip() for line in report_section.splitlines() if line.strip()]

    report_rows = []
    for line in report_lines:
        stripped = line.strip()

        if stripped.startswith("precision"):
            continue

        parts = stripped.split()

        # class rows
        if len(parts) == 5:
            class_name = parts[0]
            precision, recall, f1_score, support = parts[1:]
            report_rows.append([class_name, float(precision), float(recall), float(f1_score), int(float(support))])

        # macro avg / weighted avg
        elif len(parts) == 6:
            class_name = parts[0] + " " + parts[1]
            precision, recall, f1_score, support = parts[2:]
            report_rows.append([class_name, float(precision), float(recall), float(f1_score), int(float(support))])

    return pd.DataFrame(
        report_rows,
        columns=["Class", "Precision", "Recall", "F1-Score", "Support"]
    )


def load_confusion_matrix_csv(cm_csv_path: Path):
    cm_df = pd.read_csv(cm_csv_path)
    labels = cm_df.iloc[:, 0].tolist()
    cm = cm_df.iloc[:, 1:].values
    return labels, cm


def save_table_png(df: pd.DataFrame, title: str, output_path: Path, font_size=10, x_scale=1.2, y_scale=1.5):
    fig_height = max(3, 1.2 + len(df) * 0.5)
    fig_width = max(10, len(df.columns) * 1.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(x_scale, y_scale)

    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=title
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center", color="black"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def format_number(x):
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def build_detailed_error_table(cm: np.ndarray):
    full_names = [
        "Chromophobe",
        "Clear Cell",
        "Oncocytoma",
        "Papillary"
    ]

    short_cols = ["As C (%)", "As CC (%)", "As O (%)", "As P (%)"]

    row_sums = cm.sum(axis=1)
    rows = []

    for i, class_name in enumerate(full_names):
        total_true = int(row_sums[i])

        percentages = []
        for j in range(cm.shape[1]):
            pct = (cm[i, j] / row_sums[i] * 100) if row_sums[i] > 0 else 0
            percentages.append(format_number(pct))

        correct_rate = format_number((cm[i, i] / row_sums[i] * 100) if row_sums[i] > 0 else 0)
        error_rate = format_number(100 - float(correct_rate))

        rows.append([
            class_name,
            percentages[0],
            percentages[1],
            percentages[2],
            percentages[3],
            correct_rate,
            error_rate
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "Actual Class",
            "Predicted as C (%)",
            "Predicted as CC (%)",
            "Predicted as O (%)",
            "Predicted as P (%)",
            "Correct Rate (%)",
            "Error Rate (%)"
        ]
    )
def build_summary_table(history_df, report_df, fixed_cm, fixed_class_error_df, multi_cm):
    best_val_acc = float(history_df["val_acc"].max())
    fixed_test_acc = float(np.trace(fixed_cm) / np.sum(fixed_cm)) if np.sum(fixed_cm) > 0 else 0.0
    multi_test_acc = float(np.trace(multi_cm) / np.sum(multi_cm)) if np.sum(multi_cm) > 0 else 0.0

    macro_row = report_df[report_df["Class"] == "macro avg"]
    if not macro_row.empty:
        macro_precision = float(macro_row["Precision"].values[0])
        macro_recall = float(macro_row["Recall"].values[0])
        macro_f1 = float(macro_row["F1-Score"].values[0])
    else:
        macro_precision = np.nan
        macro_recall = np.nan
        macro_f1 = np.nan

    avg_class_error_rate = float(fixed_class_error_df["error_rate_percent"].mean())

    best_class_row = fixed_class_error_df.loc[fixed_class_error_df["recall"].idxmax()]
    worst_class_row = fixed_class_error_df.loc[fixed_class_error_df["recall"].idxmin()]

    summary_rows = [
        ["Best Validation Accuracy", f"{best_val_acc:.4f}"],
        ["Fixed Test Accuracy", f"{fixed_test_acc:.4f}"],
        ["Multi-Sample Test Accuracy", f"{multi_test_acc:.4f}"],
        ["Macro Precision", f"{macro_precision:.4f}"],
        ["Macro Recall", f"{macro_recall:.4f}"],
        ["Macro F1-Score", f"{macro_f1:.4f}"],
        ["Average Class Error Rate (%)", f"{avg_class_error_rate:.2f}%"],
        ["Best Performing Class", f"{best_class_row['class']} (recall={best_class_row['recall']:.4f})"],
        ["Worst Performing Class", f"{worst_class_row['class']} (recall={worst_class_row['recall']:.4f})"],
    ]

    return pd.DataFrame(summary_rows, columns=["Metric", "Value"])


# -------------------------
# Load data
# -------------------------
history_df = pd.read_csv(HISTORY_CSV)
report_df = parse_classification_report_txt(FIXED_REPORT_TXT)
fixed_labels, fixed_cm = load_confusion_matrix_csv(FIXED_CM_CSV)
multi_labels, multi_cm = load_confusion_matrix_csv(MULTI_CM_CSV)
fixed_class_error_df = pd.read_csv(FIXED_CLASS_ERROR_CSV)

# -------------------------
# 1) Combined training overview graph
# -------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

line1 = ax1.plot(history_df["epoch"], history_df["train_acc"], marker="o", linewidth=2, label="Train Accuracy", color="tab:blue")
line2 = ax1.plot(history_df["epoch"], history_df["val_acc"], marker="o", linewidth=3.5, label="Validation Accuracy", color="tab:orange")

line3 = ax2.plot(history_df["epoch"], history_df["train_loss"], marker="s", linestyle="--", linewidth=2, label="Train Loss", color="tab:green")
line4 = ax2.plot(history_df["epoch"], history_df["val_loss"], marker="s", linestyle="--", linewidth=2.5, label="Validation Loss", color="tab:red")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax2.set_ylabel("Loss")
ax1.set_title("ResNet18 Training Overview")
ax1.set_xticks(history_df["epoch"])
ax1.grid(True, alpha=0.3)

lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.tight_layout()
plt.savefig(COMBINED_GRAPH_PNG, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# 2) Classification report table
# -------------------------
report_table_df = report_df.copy()
report_table_df["Precision"] = report_table_df["Precision"].map(lambda x: f"{x:.4f}")
report_table_df["Recall"] = report_table_df["Recall"].map(lambda x: f"{x:.4f}")
report_table_df["F1-Score"] = report_table_df["F1-Score"].map(lambda x: f"{x:.4f}")

save_table_png(
    report_table_df,
    "ResNet18 Fixed Test Classification Report",
    REPORT_TABLE_PNG,
    font_size=10,
    x_scale=1.2,
    y_scale=1.5
)

# -------------------------
# 3) Detailed prediction/error table
# -------------------------
detailed_error_df = build_detailed_error_table(fixed_cm)

save_table_png(
    detailed_error_df,
    "ResNet18 Fixed Test Detailed Prediction/Error Table",
    DETAILED_ERROR_TABLE_PNG,
    font_size=10,
    x_scale=1.2,
    y_scale=1.5
)

# -------------------------
# 4) Fixed confusion matrix
# -------------------------
plot_confusion_matrix(
    fixed_cm,
    fixed_labels,
    "ResNet18 Fixed Test Confusion Matrix",
    FIXED_CM_PNG
)

# -------------------------
# 5) Multi-sample confusion matrix
# -------------------------
plot_confusion_matrix(
    multi_cm,
    multi_labels,
    "ResNet18 Multi-Sample Confusion Matrix",
    MULTI_CM_PNG
)

# -------------------------
# 6) Summary table
# -------------------------
summary_df = build_summary_table(
    history_df,
    report_df,
    fixed_cm,
    fixed_class_error_df,
    multi_cm
)

save_table_png(
    summary_df,
    "ResNet18 Summary",
    SUMMARY_TABLE_PNG,
    font_size=10,
    x_scale=1.2,
    y_scale=1.5
)

print("Saved:")
print(COMBINED_GRAPH_PNG)
print(REPORT_TABLE_PNG)
print(DETAILED_ERROR_TABLE_PNG)
print(FIXED_CM_PNG)
print(MULTI_CM_PNG)
print(SUMMARY_TABLE_PNG)