from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 14b_attach_identifiers_to_firm_characteristics.py
#
# Amaç:
# Firma-yıl finansal karakteristik paneline CUSIP, ISIN,
# ticker ve eşleştirme bilgilerini CIK üzerinden eklemek.
# ============================================================

PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")
OUTPUT_DIR = PROJECT_DIR / "output"

FIRM_FILE = OUTPUT_DIR / "firm_characteristics_annual.parquet"
MASTER_FILE = OUTPUT_DIR / "2025q4_us_company_master.csv"

ENRICHED_FILE = (
    OUTPUT_DIR / "firm_characteristics_annual_identifiers.parquet"
)

ENRICHED_CSV_FILE = (
    OUTPUT_DIR / "firm_characteristics_annual_identifiers.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR / "14b_identifier_attachment_validation.csv"
)

UNMATCHED_FILE = (
    OUTPUT_DIR / "14b_firm_characteristics_unmatched_cik.csv"
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun adlarını standartlaştırır."""

    result = df.copy()

    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return result


def normalize_cik(series: pd.Series) -> pd.Series:
    """
    CIK değerlerini 10 haneli metin formatına dönüştürür.

    Örnek:
        1750 -> 0000001750
    """

    numeric = pd.to_numeric(series, errors="coerce")

    return numeric.apply(
        lambda value: (
            f"{int(value):010d}"
            if pd.notna(value)
            else pd.NA
        )
    ).astype("string")


def clean_identifier(series: pd.Series) -> pd.Series:
    """
    CUSIP, ISIN ve ticker gibi kimlikleri temizler.
    """

    result = (
        series.astype("string")
        .str.strip()
        .str.upper()
    )

    invalid_values = {
        "",
        "NAN",
        "NONE",
        "<NA>",
        "NULL",
    }

    return result.mask(result.isin(invalid_values), pd.NA)


def select_company_master_record(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her CIK için tek ve en güvenilir company-master kaydını seçer.

    Öncelik:
    1. Company Facts için uygun kayıt
    2. CIK eşleşmesi doğrulanmış kayıt
    3. Geçerli CUSIP9
    4. Geçerli ISIN
    5. Primary ticker
    """

    master = master.copy()

    ranking_columns = [
        "COMPANY_FACTS_ELIGIBLE",
        "CIK_MATCHED",
        "HAS_VALID_CUSIP9",
        "HAS_VALID_ISIN",
    ]

    for column in ranking_columns:
        if column not in master.columns:
            master[column] = 0

        master[column] = (
            master[column]
            .astype("string")
            .str.upper()
            .map(
                {
                    "TRUE": 1,
                    "FALSE": 0,
                    "1": 1,
                    "0": 0,
                    "YES": 1,
                    "NO": 0,
                }
            )
            .fillna(
                pd.to_numeric(
                    master[column],
                    errors="coerce",
                )
            )
            .fillna(0)
            .astype(int)
        )

    if "PRIMARY_TICKER" not in master.columns:
        master["PRIMARY_TICKER"] = pd.NA

    master["_HAS_TICKER"] = (
        master["PRIMARY_TICKER"].notna().astype(int)
    )

    master = master.sort_values(
        by=[
            "CIK10",
            "COMPANY_FACTS_ELIGIBLE",
            "CIK_MATCHED",
            "HAS_VALID_CUSIP9",
            "HAS_VALID_ISIN",
            "_HAS_TICKER",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
        ],
        kind="mergesort",
    )

    duplicate_report = (
        master.groupby("CIK10", dropna=False)
        .size()
        .reset_index(name="MASTER_RECORDS_PER_CIK")
    )

    master = (
        master.dropna(subset=["CIK10"])
        .drop_duplicates(subset=["CIK10"], keep="first")
        .copy()
    )

    master = master.merge(
        duplicate_report,
        on="CIK10",
        how="left",
        validate="one_to_one",
    )

    master = master.drop(columns=["_HAS_TICKER"])

    return master


def main() -> None:

    print("=" * 72)
    print("14B - FİRMA KARAKTERİSTİKLERİNE IDENTIFIER EKLEME")
    print("=" * 72)

    if not FIRM_FILE.exists():
        raise FileNotFoundError(
            f"Firma karakteristikleri bulunamadı:\n{FIRM_FILE}"
        )

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Company master bulunamadı:\n{MASTER_FILE}"
        )

    # --------------------------------------------------------
    # 1. Verileri oku
    # --------------------------------------------------------

    print("\n1/6 - Dosyalar okunuyor...")

    firm = pd.read_parquet(FIRM_FILE)
    master = pd.read_csv(
        MASTER_FILE,
        low_memory=False,
        dtype=str,
    )

    firm = normalize_columns(firm)
    master = normalize_columns(master)

    print(f"Firma-yıl satırı: {len(firm):,}")
    print(f"Company-master satırı: {len(master):,}")

    # --------------------------------------------------------
    # 2. CIK standardizasyonu
    # --------------------------------------------------------

    print("\n2/6 - CIK değerleri standardize ediliyor...")

    if "CIK10" not in firm.columns:
        raise KeyError(
            "Firma karakteristikleri dosyasında CIK10 bulunamadı."
        )

    if "CIK" not in master.columns:
        raise KeyError(
            "Company-master dosyasında CIK bulunamadı."
        )

    firm["CIK10"] = normalize_cik(firm["CIK10"])
    master["CIK10"] = normalize_cik(master["CIK"])

    # --------------------------------------------------------
    # 3. Identifier sütunlarını temizle
    # --------------------------------------------------------

    print("\n3/6 - Identifier alanları temizleniyor...")

    identifier_columns = [
        "CUSIP9",
        "CUSIP8",
        "ISIN12",
        "PRIMARY_TICKER",
        "SEC_TICKER",
        "SEC_COMPANY_NAME_STANDARDIZED",
        "DISPLAY_ISSUER_NAME",
        "MATCH_METHOD",
    ]

    for column in identifier_columns:
        if column not in master.columns:
            master[column] = pd.NA
        else:
            master[column] = clean_identifier(
                master[column]
            )

    # --------------------------------------------------------
    # 4. Her CIK için tek master kayıt seç
    # --------------------------------------------------------

    print("\n4/6 - Her CIK için en güvenilir kayıt seçiliyor...")

    master_unique = select_company_master_record(master)

    keep_columns = [
        "CIK10",
        "CUSIP9",
        "CUSIP8",
        "ISIN12",
        "PRIMARY_TICKER",
        "SEC_TICKER",
        "SEC_COMPANY_NAME_STANDARDIZED",
        "DISPLAY_ISSUER_NAME",
        "MATCH_METHOD",
        "COMPANY_FACTS_ELIGIBLE",
        "CIK_MATCHED",
        "HAS_VALID_CUSIP9",
        "HAS_VALID_ISIN",
        "MASTER_RECORDS_PER_CIK",
    ]

    keep_columns = [
        column
        for column in keep_columns
        if column in master_unique.columns
    ]

    master_unique = master_unique[keep_columns].copy()

    print(
        f"Benzersiz master CIK sayısı: "
        f"{master_unique['CIK10'].nunique():,}"
    )

    # --------------------------------------------------------
    # 5. Firma-yıl paneliyle birleştir
    # --------------------------------------------------------

    print("\n5/6 - Firma-yıl paneli ile birleştiriliyor...")

    original_rows = len(firm)

    enriched = firm.merge(
        master_unique,
        on="CIK10",
        how="left",
        validate="many_to_one",
        indicator="_IDENTIFIER_MERGE",
    )

    if len(enriched) != original_rows:
        raise RuntimeError(
            "Merge sonrasında firma-yıl satır sayısı değişti. "
            "Many-to-one yapısı bozulmuş olabilir."
        )

    enriched["IDENTIFIER_MATCHED"] = (
        enriched["_IDENTIFIER_MERGE"].eq("both")
    )

    enriched["HAS_CUSIP9_ATTACHED"] = (
        enriched["CUSIP9"].notna()
    )

    enriched["HAS_CUSIP8_ATTACHED"] = (
        enriched["CUSIP8"].notna()
    )

    enriched["HAS_ISIN_ATTACHED"] = (
        enriched["ISIN12"].notna()
    )

    enriched["HAS_TICKER_ATTACHED"] = (
        enriched["PRIMARY_TICKER"].notna()
        | enriched["SEC_TICKER"].notna()
    )

    # Holdings dosyasıyla sonraki merge için ortak isimler
    enriched["CUSIP"] = enriched["CUSIP9"]
    enriched["ISIN"] = enriched["ISIN12"]
    enriched["TICKER"] = (
        enriched["PRIMARY_TICKER"]
        .fillna(enriched["SEC_TICKER"])
    )

    # --------------------------------------------------------
    # 6. Çıktılar ve doğrulama
    # --------------------------------------------------------

    print("\n6/6 - Çıktılar ve doğrulama hazırlanıyor...")

    validation_rows = [
        {
            "METRIC": "FIRM_YEAR_ROWS",
            "VALUE": len(enriched),
        },
        {
            "METRIC": "UNIQUE_CIK_IN_FIRM_PANEL",
            "VALUE": enriched["CIK10"].nunique(),
        },
        {
            "METRIC": "IDENTIFIER_MATCHED_ROWS",
            "VALUE": int(
                enriched["IDENTIFIER_MATCHED"].sum()
            ),
        },
        {
            "METRIC": "IDENTIFIER_MATCH_RATE",
            "VALUE": float(
                enriched["IDENTIFIER_MATCHED"].mean()
            ),
        },
        {
            "METRIC": "CUSIP9_COVERAGE",
            "VALUE": float(
                enriched["CUSIP9"].notna().mean()
            ),
        },
        {
            "METRIC": "CUSIP8_COVERAGE",
            "VALUE": float(
                enriched["CUSIP8"].notna().mean()
            ),
        },
        {
            "METRIC": "ISIN_COVERAGE",
            "VALUE": float(
                enriched["ISIN12"].notna().mean()
            ),
        },
        {
            "METRIC": "TICKER_COVERAGE",
            "VALUE": float(
                enriched["TICKER"].notna().mean()
            ),
        },
        {
            "METRIC": "DUPLICATE_CIK_FISCAL_YEAR_ROWS",
            "VALUE": int(
                enriched.duplicated(
                    subset=["CIK10", "FISCAL_YEAR"],
                    keep=False,
                ).sum()
            ),
        },
    ]

    validation = pd.DataFrame(validation_rows)

    unmatched = (
        enriched.loc[
            ~enriched["IDENTIFIER_MATCHED"],
            [
                "CIK10",
                "ENTITY_NAME",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            ["ENTITY_NAME", "CIK10"],
            na_position="last",
        )
    )

    enriched.to_parquet(
        ENRICHED_FILE,
        index=False,
    )

    enriched.to_csv(
        ENRICHED_CSV_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    unmatched.to_csv(
        UNMATCHED_FILE,
        index=False,
    )

    print("\nIDENTIFIER EKLEME TAMAMLANDI")
    print("=" * 72)

    print(validation.to_string(index=False))

    print("\nOluşturulan dosyalar:")
    print(ENRICHED_FILE)
    print(ENRICHED_CSV_FILE)
    print(VALIDATION_FILE)
    print(UNMATCHED_FILE)


if __name__ == "__main__":
    main()