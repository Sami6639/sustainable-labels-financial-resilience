from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

HOLDINGS_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_sustainable_etf_holdings.parquet"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_firm_universe_profile.csv"
)


def main():

    holdings = pd.read_parquet(HOLDINGS_FILE)

    # Sayısal ağırlık
    holdings["PERCENTAGE_NUMERIC"] = pd.to_numeric(
        holdings["PERCENTAGE"],
        errors="coerce"
    )

    summary = (
        holdings
        .groupby(
            [
                "ISSUER_NAME",
                "ISSUER_CUSIP",
                "IDENTIFIER_ISIN",
                "IDENTIFIER_TICKER",
                "INVESTMENT_COUNTRY",
                "ISSUER_TYPE",
                "ASSET_CAT",
            ],
            dropna=False
        )
        .agg(
            ETF_COUNT=("SERIES_NAME", "nunique"),
            HOLDING_ROWS=("HOLDING_ID", "size"),
            MAX_WEIGHT=("PERCENTAGE_NUMERIC", "max"),
            AVG_WEIGHT=("PERCENTAGE_NUMERIC", "mean"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        ["ETF_COUNT", "MAX_WEIGHT"],
        ascending=[False, False]
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("FIRM UNIVERSE PROFILING TAMAMLANDI")
    print("=" * 60)

    print("Toplam benzersiz firma:",
          len(summary))

    print("Ticker bulunan firma:",
          summary["IDENTIFIER_TICKER"].fillna("").ne("").sum())

    print("ISIN bulunan firma:",
          summary["IDENTIFIER_ISIN"].fillna("").ne("").sum())

    print("CUSIP bulunan firma:",
          summary["ISSUER_CUSIP"].fillna("").ne("").sum())

    print()

    print("Ülke dağılımı (ilk 15):")
    print(
        summary["INVESTMENT_COUNTRY"]
        .fillna("UNKNOWN")
        .value_counts()
        .head(15)
    )

    print()

    print("Asset Category:")
    print(
        summary["ASSET_CAT"]
        .fillna("UNKNOWN")
        .value_counts()
    )

    print()

    print("Issuer Type:")
    print(
        summary["ISSUER_TYPE"]
        .fillna("UNKNOWN")
        .value_counts()
    )

    print()

    print("Dosya kaydedildi:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()