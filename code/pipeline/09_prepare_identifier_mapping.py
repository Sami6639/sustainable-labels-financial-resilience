from pathlib import Path
import re

import pandas as pd


PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

INPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_include_equity_firm_universe.csv"
)

OUTPUT_ALL_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_firm_identifier_mapping_all.csv"
)

OUTPUT_US_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_us_firm_identifier_mapping.csv"
)

OUTPUT_REVIEW_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_identifier_manual_review.csv"
)

OUTPUT_SUMMARY_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_identifier_mapping_summary.csv"
)


def clean_identifier(value: object) -> str:
    """Kimlik alanlarından boşluk ve gereksiz karakterleri kaldırır."""
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if text in {"", "NAN", "NONE", "<NA>"}:
        return ""

    return re.sub(r"[^A-Z0-9|.\-]", "", text)


def split_identifier_values(value: object) -> list[str]:
    """Bir hücrede birden fazla kimlik varsa benzersiz liste üretir."""
    cleaned = clean_identifier(value)

    if not cleaned:
        return []

    values = [
        item.strip()
        for item in cleaned.split("|")
        if item.strip()
    ]

    return sorted(set(values))


def clean_cusip(value: object) -> str:
    """İlk geçerli 9 karakterli CUSIP değerini döndürür."""
    values = split_identifier_values(value)

    valid = [
        item
        for item in values
        if re.fullmatch(r"[A-Z0-9]{9}", item)
    ]

    return valid[0] if valid else ""


def clean_isin(value: object) -> str:
    """İlk geçerli 12 karakterli ISIN değerini döndürür."""
    values = split_identifier_values(value)

    valid = [
        item
        for item in values
        if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", item)
    ]

    return valid[0] if valid else ""


def clean_ticker(value: object) -> str:
    """Finansal veri eşleştirmesi için ilk kullanılabilir ticker'ı döndürür."""
    values = split_identifier_values(value)

    valid = []

    for item in values:
        ticker = item.strip().upper()

        if not ticker:
            continue

        if len(ticker) > 15:
            continue

        if not re.fullmatch(r"[A-Z0-9.\-]+", ticker):
            continue

        valid.append(ticker)

    return valid[0] if valid else ""


def normalize_company_name(value: object) -> str:
    """Şirket adını yardımcı eşleştirme için standartlaştırır."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    text = text.replace("&", " AND ")

    text = re.sub(
        r"\b(INCORPORATED|INCORPORATION|INC|CORPORATION|CORP|COMPANY|CO|"
        r"LIMITED|LTD|PLC|LLC|LP|SA|NV|AG|SE|ADR|ADS)\b",
        " ",
        text,
    )

    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı:\n{INPUT_FILE}"
        )

    print("1/4 — Firma evreni okunuyor...")

    firms = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        low_memory=False,
    )

    required_columns = {
        "DISPLAY_ISSUER_NAME",
        "CUSIP_CLEAN",
        "ISIN_CLEAN",
        "TICKER_CLEAN",
        "COUNTRY_CLEAN",
        "ETF_COUNT",
        "HOLDING_ROWS",
        "MAX_PORTFOLIO_WEIGHT",
    }

    missing = required_columns.difference(firms.columns)

    if missing:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing)}"
        )

    print(f"Başlangıç firma kaydı: {len(firms):,}")

    print("\n2/4 — Kimlik alanları temizleniyor...")

    firms["COMPANY_NAME_STANDARDIZED"] = (
        firms["DISPLAY_ISSUER_NAME"]
        .apply(normalize_company_name)
    )

    firms["CUSIP9"] = (
        firms["CUSIP_CLEAN"]
        .apply(clean_cusip)
    )

    firms["CUSIP8"] = (
        firms["CUSIP9"]
        .str[:8]
    )

    firms["ISIN12"] = (
        firms["ISIN_CLEAN"]
        .apply(clean_isin)
    )

    firms["PRIMARY_TICKER"] = (
        firms["TICKER_CLEAN"]
        .apply(clean_ticker)
    )

    firms["COUNTRY"] = (
        firms["COUNTRY_CLEAN"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    firms["US_BASELINE_FLAG"] = (
        firms["COUNTRY"] == "US"
    )

    firms["HAS_VALID_CUSIP9"] = (
        firms["CUSIP9"].ne("")
    )

    firms["HAS_VALID_CUSIP8"] = (
        firms["CUSIP8"].str.len().eq(8)
    )

    firms["HAS_VALID_ISIN"] = (
        firms["ISIN12"].ne("")
    )

    firms["HAS_TICKER"] = (
        firms["PRIMARY_TICKER"].ne("")
    )

    # ABD menkul kıymetlerinde ISIN genellikle US ile başlar.
    firms["US_ISIN_FLAG"] = (
        firms["ISIN12"].str.startswith("US", na=False)
    )

    firms["COUNTRY_ISIN_CONFLICT"] = (
        (firms["US_BASELINE_FLAG"] & firms["HAS_VALID_ISIN"])
        & (~firms["US_ISIN_FLAG"])
    )

    print("\n3/4 — Eşleştirme kalitesi sınıflandırılıyor...")

    def assign_match_priority(row: pd.Series) -> str:
        if (
            row["HAS_VALID_CUSIP8"]
            and row["HAS_VALID_ISIN"]
            and row["HAS_TICKER"]
        ):
            return "A_CUSIP_ISIN_TICKER"

        if row["HAS_VALID_CUSIP8"] and row["HAS_VALID_ISIN"]:
            return "B_CUSIP_ISIN"

        if row["HAS_VALID_CUSIP8"] and row["HAS_TICKER"]:
            return "C_CUSIP_TICKER"

        if row["HAS_VALID_CUSIP8"]:
            return "D_CUSIP_ONLY"

        if row["HAS_VALID_ISIN"] and row["HAS_TICKER"]:
            return "E_ISIN_TICKER"

        if row["HAS_VALID_ISIN"]:
            return "F_ISIN_ONLY"

        if row["HAS_TICKER"]:
            return "G_TICKER_ONLY"

        return "H_NAME_ONLY"

    firms["MATCH_PRIORITY"] = firms.apply(
        assign_match_priority,
        axis=1,
    )

    firms["MANUAL_REVIEW_FLAG"] = (
        firms["COUNTRY_ISIN_CONFLICT"]
        | firms["COMPANY_NAME_STANDARDIZED"].eq("")
        | (
            ~firms["HAS_VALID_CUSIP8"]
            & ~firms["HAS_VALID_ISIN"]
            & ~firms["HAS_TICKER"]
        )
    )

    # Aynı CUSIP8'in birden fazla adla görünmesini işaretle.
    cusip8_name_count = (
        firms.loc[firms["HAS_VALID_CUSIP8"]]
        .groupby("CUSIP8")["COMPANY_NAME_STANDARDIZED"]
        .nunique()
    )

    conflicting_cusip8 = set(
        cusip8_name_count[
            cusip8_name_count > 1
        ].index
    )

    firms["CUSIP8_NAME_CONFLICT"] = (
        firms["CUSIP8"].isin(conflicting_cusip8)
        & firms["HAS_VALID_CUSIP8"]
    )

    firms["MANUAL_REVIEW_FLAG"] = (
        firms["MANUAL_REVIEW_FLAG"]
        | firms["CUSIP8_NAME_CONFLICT"]
    )

    firms = firms.sort_values(
        [
            "US_BASELINE_FLAG",
            "MATCH_PRIORITY",
            "ETF_COUNT",
            "MAX_PORTFOLIO_WEIGHT",
            "DISPLAY_ISSUER_NAME",
        ],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)

    firms.to_csv(
        OUTPUT_ALL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    us_firms = firms.loc[
        firms["US_BASELINE_FLAG"]
    ].copy()

    us_firms.to_csv(
        OUTPUT_US_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    manual_review = firms.loc[
        firms["MANUAL_REVIEW_FLAG"]
    ].copy()

    manual_review.to_csv(
        OUTPUT_REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n4/4 — Özet hazırlanıyor...")

    summary = pd.DataFrame(
        {
            "METRIC": [
                "ALL_FIRM_RECORDS",
                "US_BASELINE_FIRM_RECORDS",
                "US_WITH_VALID_CUSIP8",
                "US_WITH_VALID_ISIN",
                "US_WITH_TICKER",
                "US_WITH_CUSIP8_AND_ISIN",
                "US_WITH_CUSIP8_ISIN_TICKER",
                "ALL_MANUAL_REVIEW_RECORDS",
                "CUSIP8_NAME_CONFLICT_RECORDS",
                "COUNTRY_ISIN_CONFLICT_RECORDS",
            ],
            "VALUE": [
                len(firms),
                len(us_firms),
                int(us_firms["HAS_VALID_CUSIP8"].sum()),
                int(us_firms["HAS_VALID_ISIN"].sum()),
                int(us_firms["HAS_TICKER"].sum()),
                int(
                    (
                        us_firms["HAS_VALID_CUSIP8"]
                        & us_firms["HAS_VALID_ISIN"]
                    ).sum()
                ),
                int(
                    (
                        us_firms["HAS_VALID_CUSIP8"]
                        & us_firms["HAS_VALID_ISIN"]
                        & us_firms["HAS_TICKER"]
                    ).sum()
                ),
                len(manual_review),
                int(firms["CUSIP8_NAME_CONFLICT"].sum()),
                int(firms["COUNTRY_ISIN_CONFLICT"].sum()),
            ],
        }
    )

    summary.to_csv(
        OUTPUT_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nIDENTIFIER MAPPING HAZIR")
    print("=" * 70)
    print(f"Toplam firma kaydı: {len(firms):,}")
    print(f"ABD ana örneklem firma kaydı: {len(us_firms):,}")
    print(
        "ABD — geçerli CUSIP8:",
        f"{int(us_firms['HAS_VALID_CUSIP8'].sum()):,}",
    )
    print(
        "ABD — geçerli ISIN:",
        f"{int(us_firms['HAS_VALID_ISIN'].sum()):,}",
    )
    print(
        "ABD — ticker:",
        f"{int(us_firms['HAS_TICKER'].sum()):,}",
    )
    print(
        "ABD — CUSIP8 + ISIN:",
        f"{int((us_firms['HAS_VALID_CUSIP8'] & us_firms['HAS_VALID_ISIN']).sum()):,}",
    )
    print(
        "Manuel inceleme gereken kayıt:",
        f"{len(manual_review):,}",
    )

    print("\nABD eşleştirme önceliği dağılımı:")
    print(
        us_firms["MATCH_PRIORITY"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nOluşturulan dosyalar:")
    print(OUTPUT_ALL_FILE)
    print(OUTPUT_US_FILE)
    print(OUTPUT_REVIEW_FILE)
    print(OUTPUT_SUMMARY_FILE)


if __name__ == "__main__":
    main()