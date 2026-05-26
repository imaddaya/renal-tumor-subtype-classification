from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

WSI_CSV = DATA_ROOT / "wsi_metadata.csv"
PATCH_CSV = DATA_ROOT / "patch_metadata.csv"

FULL_TO_SHORT = {
    "chromophobe": "c",
    "clearcell": "cc",
    "oncocytoma": "o",
    "papillary": "p",
}

LABEL_ORDER = ["c", "cc", "o", "p"]
SPLIT_ORDER = ["train", "test", "validate"]


def prepare_labels(df):
    df = df.copy()
    df["label_short"] = df["label"].map(FULL_TO_SHORT)
    df = df[df["label_short"].isin(LABEL_ORDER)]
    return df


def print_title(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def print_split_table(df, title):
    table = pd.crosstab(df["label_short"], df["split"])
    table = table.reindex(index=LABEL_ORDER, columns=SPLIT_ORDER, fill_value=0)

    print_title(title)
    print(table.to_string())


def print_total_table(df, title):
    counts = df["label_short"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    result = pd.DataFrame({"total": counts})

    print_title(title)
    print(result.to_string())


def main():
    if not WSI_CSV.exists():
        print(f"Missing file: {WSI_CSV}")
        return

    wsi_df = pd.read_csv(WSI_CSV)
    wsi_df = prepare_labels(wsi_df)

    print_split_table(wsi_df, "WSI counts per class per split")
    print_total_table(wsi_df, "Total WSI counts per class")

    if PATCH_CSV.exists():
        patch_df = pd.read_csv(PATCH_CSV)
        patch_df = prepare_labels(patch_df)

        print_split_table(patch_df, "Patch counts per class per split")
        print_total_table(patch_df, "Total patch counts per class")
    else:
        print_title("Patch metadata")
        print("patch_metadata.csv not found, so patch counts were skipped.")


if __name__ == "__main__":
    main()