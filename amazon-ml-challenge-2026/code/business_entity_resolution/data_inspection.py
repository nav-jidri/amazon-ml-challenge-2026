import pandas as pd
from pathlib import Path

DATASET = Path("dataset")

FILES = [
    DATASET / "train" / "train_source2.tsv",
    DATASET / "test" / "test_source1.tsv",
    DATASET / "test" / "test_source2.tsv",
    DATASET / "test" / "test_source3.tsv",
]


def inspect_file(path):
    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    total_rows = 0
    columns = None
    missing = None
    duplicate_ids = 0
    duplicate_rows = 0

    countries = {}
    name_lengths = []
    address_lengths = []

    sample = None

    for chunk in pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        chunksize=100_000,
        dtype=str,
        keep_default_na=True
    ):
        if sample is None:
            sample = chunk.head(5).copy()

        if columns is None:
            columns = chunk.columns.tolist()
            missing = pd.Series(0, index=columns, dtype="int64")

        total_rows += len(chunk)

        missing += chunk.isna().sum()

        duplicate_rows += chunk.duplicated().sum()

        if "entity_id" in chunk.columns:
            duplicate_ids += chunk["entity_id"].duplicated().sum()

        if "country" in chunk.columns:
            counts = chunk["country"].fillna("<MISSING>").value_counts()
            for country, count in counts.items():
                countries[country] = countries.get(country, 0) + int(count)

        if "business_name" in chunk.columns:
            lengths = chunk["business_name"].fillna("").str.len()
            name_lengths.extend(lengths.tolist())

        if "business_address" in chunk.columns:
            lengths = chunk["business_address"].fillna("").str.len()
            address_lengths.extend(lengths.tolist())

    print(f"\nROWS: {total_rows:,}")
    print(f"COLUMNS: {columns}")

    print("\nMISSING VALUES:")
    print(missing.to_string())

    print(f"\nDUPLICATE COMPLETE ROWS: {duplicate_rows:,}")
    print(f"DUPLICATE ENTITY IDs: {duplicate_ids:,}")

    print("\nCOUNTRIES:")
    for country, count in sorted(countries.items(), key=lambda x: -x[1]):
        print(f"  {country}: {count:,}")

    if name_lengths:
        s = pd.Series(name_lengths)
        print("\nBUSINESS NAME LENGTH:")
        print(f"  min:    {s.min()}")
        print(f"  max:    {s.max()}")
        print(f"  mean:   {s.mean():.2f}")
        print(f"  median: {s.median():.2f}")

    if address_lengths:
        s = pd.Series(address_lengths)
        print("\nBUSINESS ADDRESS LENGTH:")
        print(f"  min:    {s.min()}")
        print(f"  max:    {s.max()}")
        print(f"  mean:   {s.mean():.2f}")
        print(f"  median: {s.median():.2f}")

    print("\nSAMPLE:")
    print(sample.to_string(index=False))


for file in FILES:
    if file.exists():
        inspect_file(file)
    else:
        print(f"\nMISSING FILE: {file}")