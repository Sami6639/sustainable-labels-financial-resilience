from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

CLASSIFICATION_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_etf_refined_review_list.csv"
)

HOLDINGS_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_sustainable_etf_holdings.parquet"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_include_sample_summary.csv"
)


def main() -> None:
    if not CLASSIFICATION_FILE.exists():
        raise FileNotFoundError(
            f"Sınıflandırma dosyası bulunamadı:\n{CLASSIFICATION_FILE}"
        )

    if not HOLDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Holdings dosyası bulunamadı:\n{HOLDINGS_FILE}"
        )

    classification = pd.read_csv(
        CLASSIFICATION_FILE,
        dtype=str,
        low_memory=False,
    )

    holdings = pd.read_parquet(
        HOLDINGS_FILE
    )

    include_list = classification.loc[
        classification["REFINED_DECISION"] == "INCLUDE",
        [
            "ACCESSION_NUMBER",
            "SERIES_ID",
            "SERIES_NAME",
            "THEME_PRELIMINARY",
            "REFINED_REASON",
        ],
    ].copy()

    include_list = include_list.drop_duplicates(
        subset=["ACCESSION_NUMBER"]
    )

    holdings_include = holdings.merge(
        include_list,
        on=[
            "ACCESSION_NUMBER",
            "SERIES_ID",
            "SERIES_NAME",
        ],
        how="inner",
        validate="many_to_one",
    )

    holdings_include["PERCENTAGE_NUMERIC"] = pd.to_numeric(
        holdings_include["PERCENTAGE"],
        errors="coerce",
    )

    holdings_include["NET_ASSETS_NUMERIC"] = pd.to_numeric(
        holdings_include["NET_ASSETS"],
        errors="coerce",
    )

    holdings_include["HAS_TICKER"] = (
        holdings_include["IDENTIFIER_TICKER"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    holdings_include["HAS_ISIN"] = (
        holdings_include["IDENTIFIER_ISIN"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    summary = (
        holdings_include.groupby(
            [
                "ACCESSION_NUMBER",
                "SERIES_ID",
                "SERIES_NAME",
                "THEME_PRELIMINARY",
                "REFINED_REASON",
            ],
            dropna=False,
        )
        .agg(
            NET_ASSETS=(
                "NET_ASSETS_NUMERIC",
                "first",
            ),
            HOLDING_ROWS=(
                "HOLDING_ID",
                "size",
            ),
            UNIQUE_HOLDINGS=(
                "HOLDING_ID",
                "nunique",
            ),
            UNIQUE_ISSUERS=(
                "ISSUER_NAME",
                "nunique",
            ),
            TICKER_ROWS=(
                "HAS_TICKER",
                "sum",
            ),
            ISIN_ROWS=(
                "HAS_ISIN",
                "sum",
            ),
            REPORTED_WEIGHT_SUM=(
                "PERCENTAGE_NUMERIC",
                "sum",
            ),
            TOP_HOLDING_WEIGHT=(
                "PERCENTAGE_NUMERIC",
                "max",
            ),
        )
        .reset_index()
    )

    summary["TICKER_COVERAGE_RATE"] = (
        summary["TICKER_ROWS"]
        / summary["HOLDING_ROWS"]
    )

    summary["ISIN_COVERAGE_RATE"] = (
        summary["ISIN_ROWS"]
        / summary["HOLDING_ROWS"]
    )

    summary = summary.sort_values(
        [
            "THEME_PRELIMINARY",
            "SERIES_NAME",
        ]
    ).reset_index(drop=True)

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("INCLUDE ÖRNEKLEM ÖZETİ HAZIR")
    print("=" * 70)
    print(f"INCLUDE ETF sayısı: {len(include_list):,}")
    print(
        "Holdings içinde bulunan INCLUDE ETF sayısı:",
        summary["ACCESSION_NUMBER"].nunique(),
    )
    print(
        "INCLUDE holding satırı:",
        len(holdings_include),
    )
    print(
        "INCLUDE benzersiz issuer:",
        holdings_include["ISSUER_NAME"].nunique(),
    )
    print()
    print("Tema dağılımı:")
    print(
        summary["THEME_PRELIMINARY"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()
    print("Ticker kapsama oranı ortalaması:")
    print(
        round(
            summary["TICKER_COVERAGE_RATE"].mean(),
            4,
        )
    )
    print()
    print("ISIN kapsama oranı ortalaması:")
    print(
        round(
            summary["ISIN_COVERAGE_RATE"].mean(),
            4,
        )
    )
    print()
    print(f"Dosya kaydedildi:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()