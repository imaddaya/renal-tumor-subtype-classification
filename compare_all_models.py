import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Settings
# --------------------------------------------------
BASE_DIR = "/home/hpdeadman/Grad_Project/Models"

MODEL_INFO = {
    "ResNet18": {
        "report_txt": os.path.join(BASE_DIR, "ResNet18", "results", "ResNet18_multisample_report.txt"),
        "avg_error_csv": os.path.join(BASE_DIR, "ResNet18", "results", "ResNet18_multisample_average_class_error_rates.csv"),
    },
    "ResNet18 + MIL": {
        "report_txt": os.path.join(BASE_DIR, "ResNet18_MIL", "results", "ResNet18_MIL_multisample_report.txt"),
        "avg_error_csv": os.path.join(BASE_DIR, "ResNet18_MIL", "results", "ResNet18_MIL_multisample_average_class_error_rates.csv"),
    },
    "ResNet18 + MIL + Macenko": {
        "report_txt": os.path.join(BASE_DIR, "ResNet18_MIL_Macenko", "results", "ResNet18_MIL_Macenko_multisample_report.txt"),
        "avg_error_csv": os.path.join(BASE_DIR, "ResNet18_MIL_Macenko", "results", "ResNet18_MIL_Macenko_multisample_average_class_error_rates.csv"),
    },
    "ResNet18 + MIL + KAN": {
        "report_txt": os.path.join(BASE_DIR, "ResNet18_MIL_KAN", "results", "ResNet18_MIL_KAN_multisample_report.txt"),
        "avg_error_csv": os.path.join(BASE_DIR, "ResNet18_MIL_KAN", "results", "ResNet18_MIL_KAN_multisample_average_class_error_rates.csv"),
    },
    "ResNet18 + Vision Mamba": {
        "report_txt": os.path.join(BASE_DIR, "ResNet18_VisionMamba", "results", "ResNet18_VisionMamba_multisample_report.txt"),
        "avg_error_csv": os.path.join(BASE_DIR, "ResNet18_VisionMamba", "results", "ResNet18_VisionMamba_multisample_average_class_error_rates.csv"),
    },
    "ResNet18 + Vision Mamba + KAN": {
        "report_txt": os.path.join(BASE_DIR, "ResNet18_VisionMamba_KAN", "results", "ResNet18_VisionMamba_KAN_multisample_report.txt"),
        "avg_error_csv": os.path.join(BASE_DIR, "ResNet18_VisionMamba_KAN", "results", "ResNet18_VisionMamba_KAN_multisample_average_class_error_rates.csv"),
    },
}

CLASS_NAMES = ["chromophobe", "clearcell", "oncocytoma", "papillary"]

OUTPUT_CSV = os.path.join("/home/hpdeadman/Grad_Project", "all_models_comparison_table.csv")
OUTPUT_PNG = os.path.join("/home/hpdeadman/Grad_Project", "all_models_comparison_table.png")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def parse_report_metrics(report_txt_path):
    """
    Reads the multisample report txt and extracts:
    precision, recall, f1-score, support
    for each class.
    """
    if not os.path.exists(report_txt_path):
        raise FileNotFoundError(f"Missing report file: {report_txt_path}")

    with open(report_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    if "Multi-sample Classification Report:" in text:
        section = text.split("Multi-sample Classification Report:")[-1]
    elif "Classification Report:" in text:
        section = text.split("Classification Report:")[-1]
    else:
        raise ValueError(f"Could not find classification report in: {report_txt_path}")

    # stop before confusion matrix
    if "Multi-sample Confusion Matrix:" in section:
        section = section.split("Multi-sample Confusion Matrix:")[0]
    elif "Confusion Matrix:" in section:
        section = section.split("Confusion Matrix:")[0]

    lines = [line.rstrip() for line in section.splitlines() if line.strip()]

    metrics = {}

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("precision"):
            continue

        parts = stripped.split()

        # class rows: class_name precision recall f1 support
        if len(parts) == 5 and parts[0] in CLASS_NAMES:
            class_name = parts[0]
            precision, recall, f1_score, support = parts[1:]
            metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "support": int(float(support)),
            }

    return metrics


def load_avg_error_rates(avg_error_csv_path):
    """
    Reads average class-specific error rates across runs.
    """
    if not os.path.exists(avg_error_csv_path):
        raise FileNotFoundError(f"Missing avg error csv: {avg_error_csv_path}")

    df = pd.read_csv(avg_error_csv_path)

    error_map = {}
    for _, row in df.iterrows():
        error_map[row["class"]] = float(row["avg_error_rate_across_runs"])

    return error_map


# --------------------------------------------------
# Build comparison table
# --------------------------------------------------
rows = []

for model_name, paths in MODEL_INFO.items():
    report_metrics = parse_report_metrics(paths["report_txt"])
    avg_error_rates = load_avg_error_rates(paths["avg_error_csv"])

    for class_name in CLASS_NAMES:
        if class_name not in report_metrics:
            continue

        rows.append({
            "model": model_name,
            "class": class_name,
            "precision": round(report_metrics[class_name]["precision"], 4),
            "recall": round(report_metrics[class_name]["recall"], 4),
            "f1_score": round(report_metrics[class_name]["f1_score"], 4),
            "error_rate": round(avg_error_rates.get(class_name, 0.0), 4),
        })

comparison_df = pd.DataFrame(rows)

# nicer ordering
model_order = list(MODEL_INFO.keys())
comparison_df["model"] = pd.Categorical(comparison_df["model"], categories=model_order, ordered=True)
comparison_df["class"] = pd.Categorical(comparison_df["class"], categories=CLASS_NAMES, ordered=True)
comparison_df = comparison_df.sort_values(["model", "class"]).reset_index(drop=True)

# save csv
comparison_df.to_csv(OUTPUT_CSV, index=False)

# --------------------------------------------------
# Save PNG table
# --------------------------------------------------
display_df = comparison_df.copy()
display_df["precision"] = display_df["precision"].map(lambda x: f"{x:.4f}")
display_df["recall"] = display_df["recall"].map(lambda x: f"{x:.4f}")
display_df["f1_score"] = display_df["f1_score"].map(lambda x: f"{x:.4f}")
display_df["error_rate"] = display_df["error_rate"].map(lambda x: f"{x:.4f}")

fig_height = 2 + len(display_df) * 0.35
fig, ax = plt.subplots(figsize=(14, fig_height))
ax.axis("off")

table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    loc="center",
    cellLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.15, 1.2)

plt.title("Comparison of All Models Across All Tumor Classes", pad=20)
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print(OUTPUT_CSV)
print(OUTPUT_PNG)
print("\nPreview:")
print(comparison_df)