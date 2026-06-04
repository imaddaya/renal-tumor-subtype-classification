import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Settings
# --------------------------------------------------
PROJECT_DIR = Path("/home/hpdeadman/Grad_Project")
BASE_DIR = PROJECT_DIR / "Models"
OUTPUT_DIR = PROJECT_DIR / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_INFO = {
    "ResNet18": {
        "fixed_report_txt": BASE_DIR / "ResNet18" / "results" / "fixed_test" / "ResNet18_report.txt",
        "fixed_error_csv": BASE_DIR / "ResNet18" / "results" / "fixed_test" / "ResNet18_class_error_rates.csv",
        "multi_report_txt": BASE_DIR / "ResNet18" / "results" / "multisample" / "ResNet18_multisample_report.txt",
        "multi_avg_error_csv": BASE_DIR / "ResNet18" / "results" / "multisample" / "ResNet18_multisample_average_class_error_rates.csv",
    },
    "ResNet18 + MIL": {
        "fixed_report_txt": BASE_DIR / "ResNet18_MIL" / "results" / "fixed_test" / "ResNet18_MIL_report.txt",
        "fixed_error_csv": BASE_DIR / "ResNet18_MIL" / "results" / "fixed_test" / "ResNet18_MIL_class_error_rates.csv",
        "multi_report_txt": BASE_DIR / "ResNet18_MIL" / "results" / "multisample" / "ResNet18_MIL_multisample_report.txt",
        "multi_avg_error_csv": BASE_DIR / "ResNet18_MIL" / "results" / "multisample" / "ResNet18_MIL_multisample_average_class_error_rates.csv",
    },
    "ResNet18 + MIL + Macenko": {
        "fixed_report_txt": BASE_DIR / "ResNet18_MIL_Macenko" / "results" / "fixed_test" / "ResNet18_MIL_Macenko_report.txt",
        "fixed_error_csv": BASE_DIR / "ResNet18_MIL_Macenko" / "results" / "fixed_test" / "ResNet18_MIL_Macenko_class_error_rates.csv",
        "multi_report_txt": BASE_DIR / "ResNet18_MIL_Macenko" / "results" / "multisample" / "ResNet18_MIL_Macenko_multisample_report.txt",
        "multi_avg_error_csv": BASE_DIR / "ResNet18_MIL_Macenko" / "results" / "multisample" / "ResNet18_MIL_Macenko_multisample_average_class_error_rates.csv",
    },
    "ResNet18 + MIL + KAN": {
        "fixed_report_txt": BASE_DIR / "ResNet18_MIL_KAN" / "results" / "fixed_test" / "ResNet18_MIL_KAN_report.txt",
        "fixed_error_csv": BASE_DIR / "ResNet18_MIL_KAN" / "results" / "fixed_test" / "ResNet18_MIL_KAN_class_error_rates.csv",
        "multi_report_txt": BASE_DIR / "ResNet18_MIL_KAN" / "results" / "multisample" / "ResNet18_MIL_KAN_multisample_report.txt",
        "multi_avg_error_csv": BASE_DIR / "ResNet18_MIL_KAN" / "results" / "multisample" / "ResNet18_MIL_KAN_multisample_average_class_error_rates.csv",
    },
    "ResNet18 + Vision Mamba": {
        "fixed_report_txt": BASE_DIR / "ResNet18_VisionMamba" / "results" / "fixed_test" / "ResNet18_VisionMamba_report.txt",
        "fixed_error_csv": BASE_DIR / "ResNet18_VisionMamba" / "results" / "fixed_test" / "ResNet18_VisionMamba_class_error_rates.csv",
        "multi_report_txt": BASE_DIR / "ResNet18_VisionMamba" / "results" / "multisample" / "ResNet18_VisionMamba_multisample_report.txt",
        "multi_avg_error_csv": BASE_DIR / "ResNet18_VisionMamba" / "results" / "multisample" / "ResNet18_VisionMamba_multisample_average_class_error_rates.csv",
    },
    "ResNet18 + Vision Mamba + KAN": {
        "fixed_report_txt": BASE_DIR / "ResNet18_VisionMamba_KAN" / "results" / "fixed_test" / "ResNet18_VisionMamba_KAN_report.txt",
        "fixed_error_csv": BASE_DIR / "ResNet18_VisionMamba_KAN" / "results" / "fixed_test" / "ResNet18_VisionMamba_KAN_class_error_rates.csv",
        "multi_report_txt": BASE_DIR / "ResNet18_VisionMamba_KAN" / "results" / "multisample" / "ResNet18_VisionMamba_KAN_multisample_report.txt",
        "multi_avg_error_csv": BASE_DIR / "ResNet18_VisionMamba_KAN" / "results" / "multisample" / "ResNet18_VisionMamba_KAN_multisample_average_class_error_rates.csv",
    },
}

CLASS_NAMES = ["chromophobe", "clearcell", "oncocytoma", "papillary"]

OVERALL_CSV = OUTPUT_DIR / "all_models_overall_summary.csv"
OVERALL_PNG = OUTPUT_DIR / "all_models_overall_summary.png"

PER_CLASS_CSV = OUTPUT_DIR / "all_models_per_class_comparison.csv"
PER_CLASS_PNG = OUTPUT_DIR / "all_models_per_class_comparison.png"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def parse_report_metrics(report_txt_path: Path):
    if not report_txt_path.exists():
        raise FileNotFoundError(f"Missing report file: {report_txt_path}")

    with open(report_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    if "Multi-sample Classification Report:" in text:
        section = text.split("Multi-sample Classification Report:")[-1]
    elif "Classification Report:" in text:
        section = text.split("Classification Report:")[-1]
    else:
        raise ValueError(f"Could not find classification report in: {report_txt_path}")

    if "Multi-sample Confusion Matrix:" in section:
        section = section.split("Multi-sample Confusion Matrix:")[0]
    elif "Confusion Matrix:" in section:
        section = section.split("Confusion Matrix:")[0]

    lines = [line.rstrip() for line in section.splitlines() if line.strip()]

    result = {
        "classes": {},
        "accuracy": None,
        "macro_avg": {
            "precision": None,
            "recall": None,
            "f1_score": None,
            "support": None,
        },
        "weighted_avg": {
            "precision": None,
            "recall": None,
            "f1_score": None,
            "support": None,
        },
    }

    for line in lines:
        stripped = line.strip()
        parts = stripped.split()

        if stripped.startswith("precision"):
            continue

        if len(parts) == 5 and parts[0] in CLASS_NAMES:
            class_name = parts[0]
            precision, recall, f1_score, support = parts[1:]
            result["classes"][class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "support": int(float(support)),
            }

        elif len(parts) == 3 and parts[0] == "accuracy":
            accuracy, support = parts[1:]
            result["accuracy"] = {
                "value": float(accuracy),
                "support": int(float(support)),
            }

        elif len(parts) == 6 and parts[0] == "macro" and parts[1] == "avg":
            precision, recall, f1_score, support = parts[2:]
            result["macro_avg"] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "support": int(float(support)),
            }

        elif len(parts) == 6 and parts[0] == "weighted" and parts[1] == "avg":
            precision, recall, f1_score, support = parts[2:]
            result["weighted_avg"] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "support": int(float(support)),
            }

    return result


def load_fixed_error_rates(error_csv_path: Path):
    if not error_csv_path.exists():
        raise FileNotFoundError(f"Missing fixed error csv: {error_csv_path}")

    df = pd.read_csv(error_csv_path)

    error_map = {}
    for _, row in df.iterrows():
        error_map[row["class"]] = float(row["error_rate"])

    return error_map


def load_multi_avg_error_rates(avg_error_csv_path: Path):
    if not avg_error_csv_path.exists():
        raise FileNotFoundError(f"Missing avg error csv: {avg_error_csv_path}")

    df = pd.read_csv(avg_error_csv_path)

    error_map = {}
    for _, row in df.iterrows():
        error_map[row["class"]] = float(row["avg_error_rate_across_runs"])

    return error_map


def save_table_png(df: pd.DataFrame, title: str, output_path: Path, font_size=8, x_scale=1.2, y_scale=1.2):
    display_df = df.copy()

    fig_height = max(3.5, 1.5 + len(display_df) * 0.35)
    fig_width = max(12, len(display_df.columns) * 1.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(x_scale, y_scale)

    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# Build tables
# --------------------------------------------------
overall_rows = []
per_class_rows = []

for model_name, paths in MODEL_INFO.items():
    fixed_report = parse_report_metrics(paths["fixed_report_txt"])
    multi_report = parse_report_metrics(paths["multi_report_txt"])

    fixed_errors = load_fixed_error_rates(paths["fixed_error_csv"])
    multi_errors = load_multi_avg_error_rates(paths["multi_avg_error_csv"])

    fixed_avg_error = sum(fixed_errors.values()) / len(fixed_errors) if fixed_errors else 0.0
    multi_avg_error = sum(multi_errors.values()) / len(multi_errors) if multi_errors else 0.0

    overall_rows.append({
        "model": model_name,
        "fixed_accuracy": round(fixed_report["accuracy"]["value"], 4) if fixed_report["accuracy"] else None,
        "fixed_macro_precision": round(fixed_report["macro_avg"]["precision"], 4),
        "fixed_macro_recall": round(fixed_report["macro_avg"]["recall"], 4),
        "fixed_macro_f1": round(fixed_report["macro_avg"]["f1_score"], 4),
        "fixed_avg_error_rate": round(fixed_avg_error, 4),
        "multi_accuracy": round(multi_report["accuracy"]["value"], 4) if multi_report["accuracy"] else None,
        "multi_macro_precision": round(multi_report["macro_avg"]["precision"], 4),
        "multi_macro_recall": round(multi_report["macro_avg"]["recall"], 4),
        "multi_macro_f1": round(multi_report["macro_avg"]["f1_score"], 4),
        "multi_avg_error_rate": round(multi_avg_error, 4),
    })

    for class_name in CLASS_NAMES:
        fixed_class = fixed_report["classes"].get(class_name, {})
        multi_class = multi_report["classes"].get(class_name, {})

        per_class_rows.append({
            "model": model_name,
            "class": class_name,
            "fixed_precision": round(fixed_class.get("precision", 0.0), 4),
            "fixed_recall": round(fixed_class.get("recall", 0.0), 4),
            "fixed_f1": round(fixed_class.get("f1_score", 0.0), 4),
            "fixed_error_rate": round(fixed_errors.get(class_name, 0.0), 4),
            "multi_precision": round(multi_class.get("precision", 0.0), 4),
            "multi_recall": round(multi_class.get("recall", 0.0), 4),
            "multi_f1": round(multi_class.get("f1_score", 0.0), 4),
            "multi_error_rate": round(multi_errors.get(class_name, 0.0), 4),
        })

overall_df = pd.DataFrame(overall_rows)
per_class_df = pd.DataFrame(per_class_rows)

model_order = list(MODEL_INFO.keys())
overall_df["model"] = pd.Categorical(overall_df["model"], categories=model_order, ordered=True)
overall_df = overall_df.sort_values("model").reset_index(drop=True)

per_class_df["model"] = pd.Categorical(per_class_df["model"], categories=model_order, ordered=True)
per_class_df["class"] = pd.Categorical(per_class_df["class"], categories=CLASS_NAMES, ordered=True)
per_class_df = per_class_df.sort_values(["model", "class"]).reset_index(drop=True)

# Save CSVs
overall_df.to_csv(OVERALL_CSV, index=False)
per_class_df.to_csv(PER_CLASS_CSV, index=False)

# Format for PNGs
overall_display_df = overall_df.copy()
for col in overall_display_df.columns:
    if col != "model":
        overall_display_df[col] = overall_display_df[col].map(lambda x: f"{x:.4f}")

per_class_display_df = per_class_df.copy()
for col in per_class_display_df.columns:
    if col not in ["model", "class"]:
        per_class_display_df[col] = per_class_display_df[col].map(lambda x: f"{x:.4f}")

save_table_png(
    overall_display_df,
    "Overall Comparison of All Models",
    OVERALL_PNG,
    font_size=8,
    x_scale=1.2,
    y_scale=1.25,
)

save_table_png(
    per_class_display_df,
    "Per-Class Comparison of All Models",
    PER_CLASS_PNG,
    font_size=7,
    x_scale=1.15,
    y_scale=1.15,
)

print("Saved:")
print(OVERALL_CSV)
print(OVERALL_PNG)
print(PER_CLASS_CSV)
print(PER_CLASS_PNG)

print("\nOverall summary preview:")
print(overall_df)

print("\nPer-class summary preview:")
print(per_class_df.head(12))