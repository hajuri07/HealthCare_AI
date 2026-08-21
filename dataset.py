import pandas as pd

# Load HAM metadata
ham = pd.read_csv("C:\Ibrahim\Projects\Healthcare\Pytorch-Skin-Diseases-classification\HAM10000_metadata.csv")
ham = ham.rename(columns={
    "lesion_id": "lesion_id",
    "image_id": "image_id",
    "dx": "label",
    "age": "age",
    "sex": "sex",
    "localization": "site"
})

# Load ISIC 2024 (no header, so we add our own)
isic2024 = pd.read_csv("C:\Ibrahim\Projects\Healthcare\Pytorch-Skin-Diseases-classification\ISIC_2024_Training_GroundTruth.csv", header=None)
isic2024.columns = ["image_id", "label"]   # <-- this fixes the KeyError
isic2024["lesion_id"] = ""
isic2024["age"] = ""
isic2024["sex"] = ""
isic2024["site"] = ""

# Merge
merged = pd.concat([
    ham[["image_id","label","age","sex","site","lesion_id"]],
    isic2024[["image_id","label","age","sex","site","lesion_id"]]
], ignore_index=True)

merged.to_csv("merged_classification.csv", index=False)
import pandas as pd

# Load ISIC 2024 (no header)
isic2024 = pd.read_csv("C:\Ibrahim\Projects\Healthcare\Pytorch-Skin-Diseases-classification\ISIC_2024_Training_GroundTruth.csv", header=None)
isic2024.columns = ["image_id", "label"]

# Clean up labels
isic2024["label"] = (
    isic2024["label"]
    .replace({"malignant": 1})   # fix stray string
    .astype(str)                 # force everything to string
    .str.strip()                 # remove spaces
    .replace({"0.0": 0, "1.0": 1})  # map float-like strings to ints
    .astype(int)                 # finally cast to int
)

print(isic2024["label"].value_counts())




