import pandas as pd
from pathlib import Path


def inspect_dataset(file_path, domain):
    df = pd.read_parquet(file_path)

    print(f"\n===== {domain.upper()} =====")
    print(f"Number of samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nFirst 5 samples:")
    print(df.head())

    print("\nLabel distribution:")
    print(df["label"].value_counts())


def main():
    project_root = Path(__file__).resolve().parent.parent

    laptop_train = (
        project_root
        / "data"
        / "laptop"
        / "train-00000-of-00001.parquet"
    )

    restaurant_train = (
        project_root
        / "data"
        / "restaurant"
        / "train-00000-of-00001.parquet"
    )

    inspect_dataset(laptop_train, "laptop")
    inspect_dataset(restaurant_train, "restaurant")


if __name__ == "__main__":
    main()