"""
08b_build_historical_fixed_etf_universe.py
==========================================

Amaç
----
2025Q4'te tanımlanan sürdürülebilir ETF araştırma evrenini
dondurmak ve aynı ETF'leri 2023Q4, 2024Q4 ve 2025Q4
N-PORT snapshot'larında izlemek.

Eşleştirme sırası
-----------------
1. SERIES_ID
2. SERIES_LEI
3. Normalize edilmiş SERIES_NAME

Ana prensip
-----------
Geçmiş dönemlerde ETF'ler yeniden sınıflandırılmaz.
2025Q4 REFİNED_DECISION = INCLUDE evreni sabit tutulur.

Çıktılar
--------
- Dönem bazında eşleştirme raporu
- Dönem bazında fixed-universe holdings
- Dönem bazında equity-corporate holdings
- Ortak ETF coverage raporu
- Firma evreni özetleri
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys
import traceback
from typing import Dict, List, Tuple

import pandas as pd


# ============================================================
# 0. AYARLAR
# ============================================================

PROJECT_DIR = Path.home() / "Desktop" / "CPU_Project"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUARTERS = [
    "2023Q4",
    "2024Q4",
    "2025Q4",
]

REFERENCE_QUARTER = "2025Q4"

REFERENCE_CLASSIFICATION_FILE = (
    OUTPUT_DIR / "2025q4_etf_refined_review_list.csv"
)

REFERENCE_DECISION_COLUMN = "REFINED_DECISION"
REFERENCE_INCLUDE_VALUE = "INCLUDE"


# ============================================================
# 1. YARDIMCI FONKSİYONLAR
# ============================================================

def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def normalize_name(series: pd.Series) -> pd.Series:
    """
    Fon adını güvenli eşleştirme için normalize eder.
    """
    return (
        normalize_text(series)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean_identifier(series: pd.Series) -> pd.Series:
    """
    SERIES_ID ve SERIES_LEI gibi identifier alanlarını temizler.
    """
    result = (
        normalize_text(series)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    invalid = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "NA",
    }

    return result.mask(result.isin(invalid), "")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı:\n{path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Dosya boş görünüyor:\n{path}"
        )


def quarter_paths(quarter: str) -> Dict[str, Path]:
    q = quarter.lower()

    return {
        "candidates": (
            OUTPUT_DIR
            / f"{q}_sustainable_etf_candidates.csv"
        ),
        "holdings": (
            OUTPUT_DIR
            / f"{q}_sustainable_etf_holdings.parquet"
        ),
        "mapping": (
            OUTPUT_DIR
            / f"08b_{q}_fixed_universe_mapping.csv"
        ),
        "fixed_holdings": (
            OUTPUT_DIR
            / f"{q}_fixed_universe_holdings.parquet"
        ),
        "equity_holdings": (
            OUTPUT_DIR
            / f"{q}_fixed_universe_equity_corporate_holdings.parquet"
        ),
        "firm_universe": (
            OUTPUT_DIR
            / f"{q}_fixed_universe_equity_firm_universe.csv"
        ),
        "summary": (
            OUTPUT_DIR
            / f"08b_{q}_fixed_universe_summary.csv"
        ),
    }


# ============================================================
# 2. 2025Q4 REFERANS EVRENİ
# ============================================================

def load_reference_universe() -> pd.DataFrame:
    require_file(REFERENCE_CLASSIFICATION_FILE)

    reference = pd.read_csv(
        REFERENCE_CLASSIFICATION_FILE,
        dtype=str,
        low_memory=False,
    )

    required = {
        "ACCESSION_NUMBER",
        "SERIES_NAME",
        "SERIES_ID",
        "SERIES_LEI",
        REFERENCE_DECISION_COLUMN,
    }

    missing = required.difference(reference.columns)

    if missing:
        raise KeyError(
            "Referans sınıflandırma dosyasında eksik sütunlar:\n"
            + "\n".join(sorted(missing))
        )

    reference = reference.loc[
        reference[REFERENCE_DECISION_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq(REFERENCE_INCLUDE_VALUE)
    ].copy()

    if reference.empty:
        raise RuntimeError(
            "2025Q4 referans dosyasında INCLUDE ETF bulunamadı."
        )

    reference["REFERENCE_ACCESSION_NUMBER"] = normalize_text(
        reference["ACCESSION_NUMBER"]
    )

    reference["REFERENCE_SERIES_ID"] = clean_identifier(
        reference["SERIES_ID"]
    )

    reference["REFERENCE_SERIES_LEI"] = clean_identifier(
        reference["SERIES_LEI"]
    )

    reference["REFERENCE_SERIES_NAME"] = normalize_text(
        reference["SERIES_NAME"]
    )

    reference["REFERENCE_NAME_NORMALIZED"] = normalize_name(
        reference["SERIES_NAME"]
    )

    reference = reference[
        [
            "REFERENCE_ACCESSION_NUMBER",
            "REFERENCE_SERIES_ID",
            "REFERENCE_SERIES_LEI",
            "REFERENCE_SERIES_NAME",
            "REFERENCE_NAME_NORMALIZED",
        ]
    ].drop_duplicates(
        subset=["REFERENCE_SERIES_ID", "REFERENCE_SERIES_NAME"]
    )

    duplicate_series = reference.loc[
        reference["REFERENCE_SERIES_ID"].ne("")
        & reference.duplicated(
            subset=["REFERENCE_SERIES_ID"],
            keep=False,
        )
    ]

    if not duplicate_series.empty:
        raise RuntimeError(
            "Referans INCLUDE evreninde duplicate SERIES_ID bulundu."
        )

    print(
        f"2025Q4 sabit referans ETF evreni: "
        f"{len(reference):,}"
    )

    return reference.reset_index(drop=True)


# ============================================================
# 3. TARİHSEL ETF EŞLEŞTİRME
# ============================================================

def prepare_candidates(
    quarter: str,
    path: Path,
) -> pd.DataFrame:
    require_file(path)

    candidates = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    )

    required = {
        "ACCESSION_NUMBER",
        "SERIES_NAME",
        "SERIES_ID",
        "SERIES_LEI",
    }

    missing = required.difference(candidates.columns)

    if missing:
        raise KeyError(
            f"{quarter} candidate dosyasında eksik sütunlar:\n"
            + "\n".join(sorted(missing))
        )

    candidates = candidates.copy()

    candidates["HISTORICAL_ACCESSION_NUMBER"] = normalize_text(
        candidates["ACCESSION_NUMBER"]
    )

    candidates["HISTORICAL_SERIES_ID"] = clean_identifier(
        candidates["SERIES_ID"]
    )

    candidates["HISTORICAL_SERIES_LEI"] = clean_identifier(
        candidates["SERIES_LEI"]
    )

    candidates["HISTORICAL_SERIES_NAME"] = normalize_text(
        candidates["SERIES_NAME"]
    )

    candidates["HISTORICAL_NAME_NORMALIZED"] = normalize_name(
        candidates["SERIES_NAME"]
    )

    candidates["SNAPSHOT_QUARTER"] = quarter

    return candidates


def unique_lookup(
    candidates: pd.DataFrame,
    key_column: str,
) -> Dict[str, int]:
    """
    Yalnızca tekil identifier değerlerini lookup sözlüğüne alır.
    """
    valid = candidates.loc[
        candidates[key_column].ne("")
    ].copy()

    counts = valid[key_column].value_counts()

    unique_values = set(
        counts.loc[counts.eq(1)].index
    )

    return {
        value: int(index)
        for index, value in valid[key_column].items()
        if value in unique_values
    }


def match_reference_to_quarter(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    quarter: str,
) -> pd.DataFrame:
    """
    2025Q4 referans ETF evrenini tarihsel quarter evrenine eşleştirir.
    """
    series_id_lookup = unique_lookup(
        candidates,
        "HISTORICAL_SERIES_ID",
    )

    series_lei_lookup = unique_lookup(
        candidates,
        "HISTORICAL_SERIES_LEI",
    )

    name_lookup = unique_lookup(
        candidates,
        "HISTORICAL_NAME_NORMALIZED",
    )

    rows: List[Dict[str, object]] = []

    for _, ref in reference.iterrows():
        matched_index = None
        match_method = "UNRESOLVED"

        ref_series_id = ref["REFERENCE_SERIES_ID"]
        ref_series_lei = ref["REFERENCE_SERIES_LEI"]
        ref_name = ref["REFERENCE_NAME_NORMALIZED"]

        if (
            ref_series_id
            and ref_series_id in series_id_lookup
        ):
            matched_index = series_id_lookup[ref_series_id]
            match_method = "EXACT_SERIES_ID_MATCH"

        elif (
            ref_series_lei
            and ref_series_lei in series_lei_lookup
        ):
            matched_index = series_lei_lookup[ref_series_lei]
            match_method = "SERIES_LEI_MATCH"

        elif (
            ref_name
            and ref_name in name_lookup
        ):
            matched_index = name_lookup[ref_name]
            match_method = "NORMALIZED_NAME_MATCH"

        result = {
            "SNAPSHOT_QUARTER": quarter,
            "REFERENCE_SERIES_ID": ref_series_id,
            "REFERENCE_SERIES_LEI": ref_series_lei,
            "REFERENCE_SERIES_NAME": (
                ref["REFERENCE_SERIES_NAME"]
            ),
            "REFERENCE_ACCESSION_NUMBER": (
                ref["REFERENCE_ACCESSION_NUMBER"]
            ),
            "MATCH_METHOD": match_method,
            "MATCHED_FLAG": int(matched_index is not None),
            "HISTORICAL_ACCESSION_NUMBER": "",
            "HISTORICAL_SERIES_ID": "",
            "HISTORICAL_SERIES_LEI": "",
            "HISTORICAL_SERIES_NAME": "",
        }

        if matched_index is not None:
            hist = candidates.loc[matched_index]

            result.update(
                {
                    "HISTORICAL_ACCESSION_NUMBER": (
                        hist["HISTORICAL_ACCESSION_NUMBER"]
                    ),
                    "HISTORICAL_SERIES_ID": (
                        hist["HISTORICAL_SERIES_ID"]
                    ),
                    "HISTORICAL_SERIES_LEI": (
                        hist["HISTORICAL_SERIES_LEI"]
                    ),
                    "HISTORICAL_SERIES_NAME": (
                        hist["HISTORICAL_SERIES_NAME"]
                    ),
                }
            )

        rows.append(result)

    mapping = pd.DataFrame(rows)

    duplicate_historical_accessions = mapping.loc[
        mapping["MATCHED_FLAG"].eq(1)
        & mapping.duplicated(
            subset=["HISTORICAL_ACCESSION_NUMBER"],
            keep=False,
        )
    ]

    if not duplicate_historical_accessions.empty:
        raise RuntimeError(
            f"{quarter}: Aynı tarihsel accession birden fazla "
            "referans ETF ile eşleşti."
        )

    return mapping


# ============================================================
# 4. FIXED-UNIVERSE HOLDINGS
# ============================================================

def build_fixed_universe_holdings(
    quarter: str,
    holdings_path: Path,
    mapping: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_file(holdings_path)

    holdings = pd.read_parquet(
        holdings_path
    )

    required = {
        "ACCESSION_NUMBER",
        "SERIES_NAME",
        "SERIES_ID",
        "ASSET_CAT",
        "ISSUER_TYPE",
    }

    missing = required.difference(holdings.columns)

    if missing:
        raise KeyError(
            f"{quarter} holdings dosyasında eksik sütunlar:\n"
            + "\n".join(sorted(missing))
        )

    matched_mapping = mapping.loc[
        mapping["MATCHED_FLAG"].eq(1)
    ].copy()

    matched_accessions = set(
        matched_mapping["HISTORICAL_ACCESSION_NUMBER"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    holdings = holdings.copy()

    holdings["ACCESSION_NUMBER_CLEAN"] = normalize_text(
        holdings["ACCESSION_NUMBER"]
    )

    fixed_holdings = holdings.loc[
        holdings["ACCESSION_NUMBER_CLEAN"].isin(
            matched_accessions
        )
    ].copy()

    fixed_holdings = fixed_holdings.merge(
        matched_mapping[
            [
                "HISTORICAL_ACCESSION_NUMBER",
                "REFERENCE_SERIES_ID",
                "REFERENCE_SERIES_NAME",
                "MATCH_METHOD",
            ]
        ],
        left_on="ACCESSION_NUMBER_CLEAN",
        right_on="HISTORICAL_ACCESSION_NUMBER",
        how="left",
        validate="many_to_one",
    )

    fixed_holdings["SNAPSHOT_QUARTER"] = quarter

    fixed_holdings["ASSET_CAT_CLEAN"] = (
        normalize_text(fixed_holdings["ASSET_CAT"])
        .str.upper()
    )

    fixed_holdings["ISSUER_TYPE_CLEAN"] = (
        normalize_text(fixed_holdings["ISSUER_TYPE"])
        .str.upper()
    )

    equity_corporate = fixed_holdings.loc[
        fixed_holdings["ASSET_CAT_CLEAN"].eq("EC")
        & fixed_holdings["ISSUER_TYPE_CLEAN"].eq("CORP")
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

    return fixed_holdings, equity_corporate


# ============================================================
# 5. FİRMA EVRENİ VE ÖZET
# ============================================================

def build_firm_universe(
    equity_corporate: pd.DataFrame,
) -> pd.DataFrame:
    return (
        equity_corporate.groupby(
            [
                "SNAPSHOT_QUARTER",
                "REFERENCE_SERIES_ID",
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
            ETF_COUNT=("REFERENCE_SERIES_ID", "nunique"),
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


def build_summary(
    quarter: str,
    reference: pd.DataFrame,
    mapping: pd.DataFrame,
    fixed_holdings: pd.DataFrame,
    equity_corporate: pd.DataFrame,
    firm_universe: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SNAPSHOT_QUARTER": [quarter] * 10,
            "METRIC": [
                "REFERENCE_ETF_COUNT",
                "MATCHED_REFERENCE_ETFS",
                "UNMATCHED_REFERENCE_ETFS",
                "EXACT_SERIES_ID_MATCHES",
                "SERIES_LEI_MATCHES",
                "NORMALIZED_NAME_MATCHES",
                "FIXED_UNIVERSE_HOLDING_ROWS",
                "EQUITY_CORPORATE_HOLDING_ROWS",
                "UNIQUE_EQUITY_FIRM_RECORDS",
                "UNIQUE_REFERENCE_ETFS_WITH_EQUITY_HOLDINGS",
            ],
            "VALUE": [
                len(reference),
                int(mapping["MATCHED_FLAG"].sum()),
                int((mapping["MATCHED_FLAG"] == 0).sum()),
                int(
                    mapping["MATCH_METHOD"]
                    .eq("EXACT_SERIES_ID_MATCH")
                    .sum()
                ),
                int(
                    mapping["MATCH_METHOD"]
                    .eq("SERIES_LEI_MATCH")
                    .sum()
                ),
                int(
                    mapping["MATCH_METHOD"]
                    .eq("NORMALIZED_NAME_MATCH")
                    .sum()
                ),
                len(fixed_holdings),
                len(equity_corporate),
                len(firm_universe),
                equity_corporate[
                    "REFERENCE_SERIES_ID"
                ].nunique(),
            ],
        }
    )


# ============================================================
# 6. ANA PROGRAM
# ============================================================

def main() -> None:
    print("=" * 78)
    print("08B - HISTORICAL FIXED ETF UNIVERSE")
    print("=" * 78)

    reference = load_reference_universe()

    all_mappings = []
    all_summaries = []

    for quarter in QUARTERS:
        print("\n" + "-" * 78)
        print(f"Dönem işleniyor: {quarter}")
        print("-" * 78)

        paths = quarter_paths(quarter)

        candidates = prepare_candidates(
            quarter=quarter,
            path=paths["candidates"],
        )

        mapping = match_reference_to_quarter(
            reference=reference,
            candidates=candidates,
            quarter=quarter,
        )

        fixed_holdings, equity_corporate = (
            build_fixed_universe_holdings(
                quarter=quarter,
                holdings_path=paths["holdings"],
                mapping=mapping,
            )
        )

        firm_universe = build_firm_universe(
            equity_corporate
        )

        summary = build_summary(
            quarter=quarter,
            reference=reference,
            mapping=mapping,
            fixed_holdings=fixed_holdings,
            equity_corporate=equity_corporate,
            firm_universe=firm_universe,
        )

        mapping.to_csv(
            paths["mapping"],
            index=False,
            encoding="utf-8-sig",
        )

        fixed_holdings.to_parquet(
            paths["fixed_holdings"],
            index=False,
        )

        equity_corporate.to_parquet(
            paths["equity_holdings"],
            index=False,
        )

        firm_universe.to_csv(
            paths["firm_universe"],
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            paths["summary"],
            index=False,
            encoding="utf-8-sig",
        )

        all_mappings.append(mapping)
        all_summaries.append(summary)

        print(
            f"Matched ETF: "
            f"{int(mapping['MATCHED_FLAG'].sum())} / "
            f"{len(reference)}"
        )

        print(
            f"Equity-corporate holding satırı: "
            f"{len(equity_corporate):,}"
        )

        print(
            "Equity holdingi bulunan referans ETF: "
            f"{equity_corporate['REFERENCE_SERIES_ID'].nunique():,}"
        )

    combined_mapping = pd.concat(
        all_mappings,
        ignore_index=True,
    )

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined_mapping.to_csv(
        OUTPUT_DIR
        / "08b_historical_fixed_universe_mapping_all_quarters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_summary.to_csv(
        OUTPUT_DIR
        / "08b_historical_fixed_universe_summary_all_quarters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage_panel = (
        combined_mapping.pivot_table(
            index=[
                "REFERENCE_SERIES_ID",
                "REFERENCE_SERIES_NAME",
            ],
            columns="SNAPSHOT_QUARTER",
            values="MATCHED_FLAG",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    for quarter in QUARTERS:
        if quarter not in coverage_panel.columns:
            coverage_panel[quarter] = 0

    coverage_panel["N_QUARTERS_AVAILABLE"] = (
        coverage_panel[QUARTERS].sum(axis=1)
    )

    coverage_panel["AVAILABLE_ALL_THREE_QUARTERS"] = (
        coverage_panel["N_QUARTERS_AVAILABLE"].eq(3).astype(int)
    )

    coverage_panel.to_csv(
        OUTPUT_DIR
        / "08b_fixed_etf_universe_three_quarter_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("HISTORICAL FIXED ETF UNIVERSE HAZIR")
    print("=" * 78)

    print("\nDönem bazında özet:")
    print(
        combined_summary.to_string(index=False)
    )

    print(
        "\nÜç çeyrekte de bulunan ETF sayısı: "
        f"{coverage_panel['AVAILABLE_ALL_THREE_QUARTERS'].sum():,}"
    )

    print("\nAna çıktılar:")
    print(
        OUTPUT_DIR
        / "08b_historical_fixed_universe_mapping_all_quarters.csv"
    )
    print(
        OUTPUT_DIR
        / "08b_historical_fixed_universe_summary_all_quarters.csv"
    )
    print(
        OUTPUT_DIR
        / "08b_fixed_etf_universe_three_quarter_coverage.csv"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print("\nİŞLEM BAŞARISIZ")
        print(f"Hata türü: {type(exc).__name__}")
        print(f"Hata mesajı: {exc}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)