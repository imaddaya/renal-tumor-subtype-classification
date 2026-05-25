import pandas as pd

wsi_csv = "/home/hpdeadman/Grad_Project/data/wsi_metadata.csv"
patch_csv = "/home/hpdeadman/Grad_Project/data/patch_metadata.csv"

wsi_df = pd.read_csv(wsi_csv)
patch_df = pd.read_csv(patch_csv)

print("\n=== WSI counts per class per split ===")
print(pd.crosstab(wsi_df["label"], wsi_df["split"]))

print("\n=== Total WSI counts per class ===")
print(wsi_df["label"].value_counts())

print("\n=== Patch counts per class per split ===")
print(pd.crosstab(patch_df["label"], patch_df["split"]))

print("\n=== Total patch counts per class ===")
print(patch_df["label"].value_counts())