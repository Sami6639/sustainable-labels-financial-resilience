from pathlib import Path
import pandas as pd


BASE_DIR = Path(
    r"C:\Users\User\Desktop\CPU_Project\sec_raw\2025Q4\extracted"
)

TARGET_ACCESSION = "0001049619-25-000491"

FILES = {
    "SUBMISSION": BASE_DIR / "SUBMISSION.tsv",
    "REGISTRANT": BASE_DIR / "REGISTRANT.tsv",
    "FUND_REPORTED_INFO": BASE_DIR / "FUND_REPORTED_INFO.tsv",
    "FUND_REPORTED_HOLDING": BASE_DIR / "FUND_REPORTED_HOLDING.tsv",
}


def search_small_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        low_memory=False
    )

    return df.loc[
        df["ACCESSION_NUMBER"] == TARGET_ACCESSION
    ].copy()


def search_large_file(file_path: Path) -> pd.DataFrame:
    matches = []

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=200_000,
        low_memory=False
    ):
        found = chunk.loc[
            chunk["ACCESSION_NUMBER"] == TARGET_ACCESSION
        ]

        if not found.empty:
            matches.append(found)

    if matches:
        return pd.concat(matches, ignore_index=True)

    return pd.DataFrame()


def main():
    print(f"Hedef ACCESSION_NUMBER: {TARGET_ACCESSION}")
    print("=" * 80)

    for table_name, file_path in FILES.items():
        if not file_path.exists():
            print(f"{table_name}: DOSYA BULUNAMADI")
            print(file_path)
            print("-" * 80)
            continue

        if table_name == "FUND_REPORTED_HOLDING":
            result = search_large_file(file_path)
        else:
            result = search_small_file(file_path)

        print(f"{table_name}: {len(result)} eşleşme")

        if not result.empty:
            print("Sütunlar:")
            print(result.columns.tolist())

            print("\nİlk eşleşme:")
            print(result.head(1).T)

        print("-" * 80)


if __name__ == "__main__":
    main()