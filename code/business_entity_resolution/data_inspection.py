import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"


FILES = {
    "train_source1": TRAIN_DIR / "train_source1.tsv",
    "train_source2": TRAIN_DIR / "train_source2.tsv",
    "train_source3": TRAIN_DIR / "train_source3.tsv",
    "train_ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
}


# ============================================================
# INSPECTION FUNCTION
# ============================================================

def inspect_file(name, path):

    print("\n" + "=" * 70)
    print(f"FILE: {name}")
    print("=" * 70)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    # IMPORTANT: dataset is TSV
    df = pd.read_csv(path, sep="\t")

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    if "entity_id" in df.columns:
        print("\nDuplicate entity IDs:")
        print(df["entity_id"].duplicated().sum())

        print("\nUnique entity IDs:")
        print(df["entity_id"].nunique())

    if "business_name" in df.columns:

        print("\nUnique business names:")
        print(df["business_name"].nunique())

        print("\nBusiness name examples:")
        print(df["business_name"].head(10).to_string(index=False))

        print("\nBusiness name length:")
        print(df["business_name"].astype(str).str.len().describe())

    if "business_address" in df.columns:

        print("\nBusiness address examples:")
        print(df["business_address"].head(10).to_string(index=False))

        print("\nBusiness address length:")
        print(df["business_address"].astype(str).str.len().describe())

    if "country" in df.columns:

        print("\nCountry distribution:")
        print(df["country"].value_counts(dropna=False))

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("BUSINESS ENTITY RESOLUTION - DATA INSPECTION")
    print("=" * 70)

    for name, path in FILES.items():
        inspect_file(name, path)

    print("\n")
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)