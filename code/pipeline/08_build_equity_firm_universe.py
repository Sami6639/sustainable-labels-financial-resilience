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

OUTPUT_HOLDINGS_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_include_equity_corporate_holdings.parquet"
)

OUTPUT_FIRM_UNIVERSE_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_include_equity_firm_universe.csv"
)

OUTPUT_SUMMARY_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_include_equity_firm_universe_summary.csv"
)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
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

    print("1/4 — Dosyalar okunuyor...")

    classification = pd.read_csv(
        CLASSIFICATION_FILE,
        dtype=str,
        low_memory=False,
    )

    holdings = pd.read_parquet(
        HOLDINGS_FILE
    )

    include_accessions = set(
        classification.loc[
            classification["REFINED_DECISION"] == "INCLUDE",
            "ACCESSION_NUMBER",
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    print(f"INCLUDE ETF accession sayısı: {len(include_accessions):,}")

    print("\n2/4 — INCLUDE ETF holdingleri filtreleniyor...")

    holdings["ACCESSION_NUMBER_CLEAN"] = normalize_text(
        holdings["ACCESSION_NUMBER"]
    )

    holdings["ASSET_CAT_CLEAN"] = (
        normalize_text(holdings["ASSET_CAT"])
        .str.upper()
    )

    holdings["ISSUER_TYPE_CLEAN"] = (
        normalize_text(holdings["ISSUER_TYPE"])
        .str.upper()
    )

    include_holdings = holdings.loc[
        holdings["ACCESSION_NUMBER_CLEAN"].isin(
            include_accessions
        )
    ].copy()

    print(
        f"INCLUDE holding satırı: {len(include_holdings):,}"
    )

    print("\n3/4 — Equity-corporate holdingler seçiliyor...")

    equity_corporate = include_holdings.loc[
        (include_holdings["ASSET_CAT_CLEAN"] == "EC")
        & (include_holdings["ISSUER_TYPE_CLEAN"] == "CORP")
    ].copy()

    equity_corporate["PERCENTAGE_NUMERIC"] = pd.to_numeric(
        equity_corporate["PERCENTAGE"],
        errors="coerce",
    )

    equity_corporate["CURRENCY_VALUE_NUMERIC"] = pd.to_numeric(
        equity_corporate["CURRENCY_VALUE"],
        errors="coerce",
    )

    equity_corporate["ISSUER_NAME_CLEAN"] = (
        normalize_text(equity_corporate["ISSUER_NAME"])
        .str.upper()
    )

    equity_corporate["CUSIP_CLEAN"] = (
        normalize_text(equity_corporate["ISSUER_CUSIP"])
        .str.upper()
    )

    equity_corporate["ISIN_CLEAN"] = (
        normalize_text(equity_corporate["IDENTIFIER_ISIN"])
        .str.upper()
    )

    equity_corporate["TICKER_CLEAN"] = (
        normalize_text(equity_corporate["IDENTIFIER_TICKER"])
        .str.upper()
    )

    equity_corporate["COUNTRY_CLEAN"] = (
        normalize_text(equity_corporate["INVESTMENT_COUNTRY"])
        .str.upper()
    )

    equity_corporate.to_parquet(
        OUTPUT_HOLDINGS_FILE,
        index=False,
    )

    print(
        f"Equity-corporate holding satırı: "
        f"{len(equity_corporate):,}"
    )

    print("\n4/4 — Firma evreni oluşturuluyor...")

    firm_universe = (
        equity_corporate.groupby(
            [
                "ISSUER_NAME_CLEAN",
                "CUSIP_CLEAN",
                "ISIN_CLEAN",
                "TICKER_CLEAN",
                "COUNTRY_CLEAN",
            ],
            dropna=False,
        )
        .agg(
            DISPLAY_ISSUER_NAME=("ISSUER_NAME", "first"),
            ETF_COUNT=("SERIES_NAME", "nunique"),
            HOLDING_ROWS=("HOLDING_ID", "size"),
            MAX_PORTFOLIO_WEIGHT=(
                "PERCENTAGE_NUMERIC",
                "max",
            ),
            MEAN_PORTFOLIO_WEIGHT=(
                "PERCENTAGE_NUMERIC",
                "mean",
            ),
            TOTAL_REPORTED_WEIGHT=(
                "PERCENTAGE_NUMERIC",
                "sum",
            ),
            TOTAL_REPORTED_VALUE=(
                "CURRENCY_VALUE_NUMERIC",
                "sum",
            ),
        )
        .reset_index()
    )

    firm_universe["HAS_CUSIP"] = (
        firm_universe["CUSIP_CLEAN"].ne("")
    )

    firm_universe["HAS_ISIN"] = (
        firm_universe["ISIN_CLEAN"].ne("")
    )

    firm_universe["HAS_TICKER"] = (
        firm_universe["TICKER_CLEAN"].ne("")
    )

    firm_universe["US_FIRM_FLAG"] = (
        firm_universe["COUNTRY_CLEAN"] == "US"
    )

    firm_universe = firm_universe.sort_values(
        [
            "ETF_COUNT",
            "MAX_PORTFOLIO_WEIGHT",
            "DISPLAY_ISSUER_NAME",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    firm_universe.to_csv(
        OUTPUT_FIRM_UNIVERSE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        {
            "METRIC": [
                "INCLUDE_ETF_COUNT",
                "INCLUDE_HOLDING_ROWS",
                "EQUITY_CORPORATE_HOLDING_ROWS",
                "UNIQUE_FIRM_RECORDS",
                "FIRMS_WITH_CUSIP",
                "FIRMS_WITH_ISIN",
                "FIRMS_WITH_TICKER",
                "US_FIRM_RECORDS",
                "NON_US_FIRM_RECORDS",
            ],
            "VALUE": [
                len(include_accessions),
                len(include_holdings),
                len(equity_corporate),
                len(firm_universe),
                int(firm_universe["HAS_CUSIP"].sum()),
                int(firm_universe["HAS_ISIN"].sum()),
                int(firm_universe["HAS_TICKER"].sum()),
                int(firm_universe["US_FIRM_FLAG"].sum()),
                int((~firm_universe["US_FIRM_FLAG"]).sum()),
            ],
        }
    )

    summary.to_csv(
        OUTPUT_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nEQUITY FIRM UNIVERSE HAZIR")
    print("=" * 70)
    print(f"INCLUDE ETF sayısı: {len(include_accessions):,}")
    print(f"INCLUDE holding satırı: {len(include_holdings):,}")
    print(
        "Equity-corporate holding satırı:",
        f"{len(equity_corporate):,}",
    )
    print(
        "Benzersiz firma kaydı:",
        f"{len(firm_universe):,}",
    )
    print(
        "CUSIP bulunan firma:",
        f"{int(firm_universe['HAS_CUSIP'].sum()):,}",
    )
    print(
        "ISIN bulunan firma:",
        f"{int(firm_universe['HAS_ISIN'].sum()):,}",
    )
    print(
        "Ticker bulunan firma:",
        f"{int(firm_universe['HAS_TICKER'].sum()):,}",
    )
    print(
        "ABD firma kaydı:",
        f"{int(firm_universe['US_FIRM_FLAG'].sum()):,}",
    )
    print(
        "ABD dışı firma kaydı:",
        f"{int((~firm_universe['US_FIRM_FLAG']).sum()):,}",
    )

    print("\nOluşturulan dosyalar:")
    print(OUTPUT_HOLDINGS_FILE)
    print(OUTPUT_FIRM_UNIVERSE_FILE)
    print(OUTPUT_SUMMARY_FILE)


if __name__ == "__main__":
    main()