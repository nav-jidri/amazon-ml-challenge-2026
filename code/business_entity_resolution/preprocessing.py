import re
import unicodedata
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    # Preserve Unicode characters correctly
    value = unicodedata.normalize("NFC", value)

    # Case normalization
    value = value.casefold()

    result = []

    for char in value:

        category = unicodedata.category(char)

        # Keep letters, numbers, combining marks and spaces
        if category.startswith(("L", "N", "M")) or char.isspace():
            result.append(char)
        else:
            # Replace punctuation/symbols with spaces
            result.append(" ")

    value = "".join(result)

    # Remove repeated spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# BUSINESS NAME
# ============================================================

def normalize_business_name(value):
    return normalize_text(value)


# ============================================================
# ADDRESS
# ============================================================

def normalize_address(value):
    return normalize_text(value)


# ============================================================
# COUNTRY
# ============================================================

def normalize_country(value):

    if pd.isna(value):
        return ""

    return str(value).strip().casefold()


# ============================================================
# TOKEN KEY
# ============================================================

def sorted_token_key(value):

    if not value:
        return ""

    tokens = value.split()

    return " ".join(sorted(tokens))


# ============================================================
# PREPROCESS ONE DATAFRAME CHUNK
# ============================================================

def preprocess_dataframe(df):

    df = df.copy()

    # --------------------------------------------------------
    # BUSINESS NAME
    # --------------------------------------------------------

    df["name_clean"] = (
        df["business_name"]
        .map(normalize_business_name)
    )

    # Compact representation
    df["name_compact"] = (
        df["name_clean"]
        .str.replace(r"\s+", "", regex=True)
    )

    # Token-order-independent representation
    df["name_token_key"] = (
        df["name_clean"]
        .map(sorted_token_key)
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    df["address_clean"] = (
        df["business_address"]
        .map(normalize_address)
    )

    df["address_token_key"] = (
        df["address_clean"]
        .map(sorted_token_key)
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    df["country_clean"] = (
        df["country"]
        .map(normalize_country)
    )

    return df


# ============================================================
# READ + PREPROCESS IN CHUNKS
# ============================================================

def iter_preprocessed_file(input_path, chunksize=100000):

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        chunksize=chunksize
    ):

        processed = preprocess_dataframe(chunk)

        yield processed


# ============================================================
# VALIDATE PREPROCESSING
# ============================================================

def validate_file(input_path, rows=1000):

    print("\n" + "=" * 80)
    print(f"VALIDATING: {input_path.name}")
    print("=" * 80)

    chunk = pd.read_csv(
        input_path,
        sep="\t",
        nrows=rows
    )

    processed = preprocess_dataframe(chunk)

    print(f"Rows tested: {len(processed):,}")
    print(f"Columns: {list(processed.columns)}")

    print("\nSample:")
    print(
        processed[
            [
                "entity_id",
                "business_name",
                "name_clean",
                "name_compact",
                "name_token_key",
                "business_address",
                "address_clean",
                "country_clean"
            ]
        ].head(5).to_string(index=False)
    )

    print("\nMissing values after preprocessing:")

    print(
        processed[
            [
                "name_clean",
                "name_compact",
                "name_token_key",
                "address_clean",
                "address_token_key",
                "country_clean"
            ]
        ]
        .isna()
        .sum()
    )

    print("\nValidation complete.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    files = [

        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",

        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv",
    ]

    print("\n")
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("PREPROCESSING VALIDATION")
    print("=" * 80)

    for file_path in files:

        if not file_path.exists():

            print(f"\nERROR: File not found: {file_path}")

            continue

        validate_file(file_path)

    print("\n")
    print("=" * 80)
    print("PREPROCESSING VALIDATION COMPLETE")
    print("=" * 80)