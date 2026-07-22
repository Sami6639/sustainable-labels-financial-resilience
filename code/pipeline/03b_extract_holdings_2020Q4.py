"""
03_extract_holdings_multi_quarter.py
====================================

SEC N-PORT holdings extraction for multiple quarterly snapshots.

Usage
-----
python scripts\\03_extract_holdings_multi_quarter.py 2023Q4
python scripts\\03_extract_holdings_multi_quarter.py 2024Q4
python scripts\\03_extract_holdings_multi_quarter.py 2025Q4

The script:
1. Reads the requested quarter from sec_raw/<QUARTER>/extracted.
2. Identifies a broad sustainable-ETF candidate universe.
3. Extracts holdings in chunks.
4. Attaches ticker and ISIN identifiers.
5. Writes quarter-specific CSV, Parquet, summary, validation, and log files.
6. Never overwrites another quarter's outputs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
import sys
import traceback
from typing import Dict, List

import pandas as pd


# ============================================================
# 0. GLOBAL SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
LOG_DIR = PROJECT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 200_000

VALID_QUARTERS = {"2020Q4"}

KEYWORDS = [
    "ESG",
    "SUSTAIN",
    "CLIMATE",
    "CLEAN",
    "SOLAR",
    "WIND",
    "BATTERY",
    "CARBON",
    "ENERGY",
    "GREEN",
]

INFO_COLUMNS = [
    "ACCESSION_NUMBER",
    "SERIES_NAME",
    "SERIES_ID",
    "SERIES_LEI",
    "TOTAL_ASSETS",
    "NET_ASSETS",
]

HOLDING_COLUMNS = [
    "ACCESSION_NUMBER",
    "HOLDING_ID",
    "ISSUER_NAME",
    "ISSUER_LEI",
    "ISSUER_TITLE",
    "ISSUER_CUSIP",
    "BALANCE",
    "UNIT",
    "CURRENCY_CODE",
    "CURRENCY_VALUE",
    "EXCHANGE_RATE",
    "PERCENTAGE",
    "PAYOFF_PROFILE",
    "ASSET_CAT",
    "ISSUER_TYPE",
    "INVESTMENT_COUNTRY",
    "IS_RESTRICTED_SECURITY",
    "FAIR_VALUE_LEVEL",
    "DERIVATIVE_CAT",
]

IDENTIFIER_COLUMNS = [
    "HOLDING_ID",
    "IDENTIFIER_ISIN",
    "IDENTIFIER_TICKER",
    "OTHER_IDENTIFIER",
    "OTHER_IDENTIFIER_DESC",
]


# ============================================================
# 1. QUARTER AND PATH MANAGEMENT
# ============================================================

def normalize_quarter(raw_value: str) -> str:
    """
    Normalize and validate a quarter string such as 2023Q4.
    """
    quarter = str(raw_value).strip().upper()

    if not re.fullmatch(r"20\d{2}Q[1-4]", quarter):
        raise ValueError(
            "DÃ¶nem biÃ§imi geÃ§ersiz. Ã–rnek kullanÄ±m: 2023Q4"
        )

    if quarter not in VALID_QUARTERS:
        raise ValueError(
            f"Bu Ã§alÄ±ÅŸma iÃ§in izin verilen dÃ¶nemler: "
            f"{', '.join(sorted(VALID_QUARTERS))}"
        )

    return quarter


def build_paths(quarter: str) -> Dict[str, Path]:
    """
    Build quarter-specific input and output paths.
    """
    quarter_lower = quarter.lower()


    extract_dir = (
        PROJECT_DIR
        / "data"
        / "raw"
        / "sec_raw"
        / quarter
        / "extracted"
    )

    paths = {
        "extract_dir": extract_dir,
        "info_file": extract_dir / "FUND_REPORTED_INFO.tsv",
        "holdings_file": extract_dir / "FUND_REPORTED_HOLDING.tsv",
        "identifiers_file": extract_dir / "IDENTIFIERS.tsv",
        "candidate_output": (
            OUTPUT_DIR
            / f"{quarter_lower}_sustainable_etf_candidates.csv"
        ),
        "holdings_parquet": (
            OUTPUT_DIR
            / f"{quarter_lower}_sustainable_etf_holdings.parquet"
        ),
        "holdings_csv": (
            OUTPUT_DIR
            / f"{quarter_lower}_sustainable_etf_holdings.csv"
        ),
        "summary_output": (
            OUTPUT_DIR
            / f"{quarter_lower}_sustainable_etf_holdings_summary.csv"
        ),
        "validation_output": (
            OUTPUT_DIR
            / f"{quarter_lower}_sustainable_etf_holdings_validation.csv"
        ),
        "metadata_output": (
            OUTPUT_DIR
            / f"{quarter_lower}_sustainable_etf_holdings_metadata.json"
        ),
        "log_output": (
            LOG_DIR
            / f"{quarter_lower}_extract_holdings.log"
        ),
    }

    return paths


# ============================================================
# 2. VALIDATION HELPERS
# ============================================================

def require_file(file_path: Path) -> None:
    """
    Confirm that a required input file exists and is not empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadÄ±:\n{file_path}"
        )

    if file_path.stat().st_size == 0:
        raise RuntimeError(
            f"Dosya boÅŸ gÃ¶rÃ¼nÃ¼yor:\n{file_path}"
        )


def write_log(log_file: Path, message: str) -> None:
    """
    Append a timestamped message to the quarter-specific log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with log_file.open(
        "a",
        encoding="utf-8",
    ) as output:
        output.write(f"[{timestamp}] {message}\n")


def safe_join_unique(series: pd.Series) -> str:
    """
    Join unique non-empty identifier values.
    """
    values = {
        str(value).strip()
        for value in series.dropna()
        if str(value).strip()
    }

    return "|".join(sorted(values))


# ============================================================
# 3. CANDIDATE ETF UNIVERSE
# ============================================================

def load_candidate_etfs(
    quarter: str,
    paths: Dict[str, Path],
) -> pd.DataFrame:
    """
    Identify the broad sustainable ETF candidate universe.
    """
    print("\n1/4 â€” Fon bilgileri okunuyor...")

    info = pd.read_csv(
        paths["info_file"],
        sep="\t",
        dtype=str,
        usecols=INFO_COLUMNS,
        low_memory=False,
    )

    keyword_pattern = "|".join(
        re.escape(keyword)
        for keyword in KEYWORDS
    )

    keyword_mask = info["SERIES_NAME"].str.contains(
        keyword_pattern,
        case=False,
        na=False,
        regex=True,
    )

    etf_mask = info["SERIES_NAME"].str.contains(
        "ETF",
        case=False,
        na=False,
        regex=False,
    )

    candidates = info.loc[
        keyword_mask & etf_mask
    ].copy()

    candidates = candidates.drop_duplicates(
        subset=["ACCESSION_NUMBER"]
    )

    candidates["SNAPSHOT_QUARTER"] = quarter

    candidates = candidates.sort_values(
        ["SERIES_NAME", "ACCESSION_NUMBER"]
    ).reset_index(drop=True)

    candidates.to_csv(
        paths["candidate_output"],
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Toplam fon satÄ±rÄ±: {len(info):,}")
    print(f"Aday ETF sayÄ±sÄ±: {len(candidates):,}")
    print(
        "Aday evren kaydedildi:\n"
        f"{paths['candidate_output']}"
    )

    write_log(
        paths["log_output"],
        (
            f"{quarter}: {len(info)} fund rows read; "
            f"{len(candidates)} candidate ETFs identified."
        ),
    )

    return candidates


# ============================================================
# 4. HOLDINGS EXTRACTION
# ============================================================

def extract_candidate_holdings(
    quarter: str,
    candidates: pd.DataFrame,
    paths: Dict[str, Path],
) -> pd.DataFrame:
    """
    Scan the large holdings table in chunks and retain candidate ETFs.
    """
    print("\n2/4 â€” Holdings dosyasÄ± taranÄ±yor...")

    target_accessions = set(
        candidates["ACCESSION_NUMBER"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    matched_chunks: List[pd.DataFrame] = []
    scanned_rows = 0
    matched_rows = 0

    reader = pd.read_csv(
        paths["holdings_file"],
        sep="\t",
        dtype=str,
        usecols=HOLDING_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        scanned_rows += len(chunk)

        accession_clean = (
            chunk["ACCESSION_NUMBER"]
            .astype(str)
            .str.strip()
        )

        matched = chunk.loc[
            accession_clean.isin(target_accessions)
        ].copy()

        if not matched.empty:
            matched_chunks.append(matched)
            matched_rows += len(matched)

        print(
            f"\rParÃ§a: {chunk_number:,} | "
            f"Taranan: {scanned_rows:,} | "
            f"EÅŸleÅŸen: {matched_rows:,}",
            end="",
            flush=True,
        )

    print()

    if not matched_chunks:
        raise RuntimeError(
            "Aday ETF accession numaralarÄ± iÃ§in hiÃ§bir holding "
            "kaydÄ± bulunamadÄ±."
        )

    holdings = pd.concat(
        matched_chunks,
        ignore_index=True,
    )

    holdings = holdings.merge(
        candidates[
            [
                "ACCESSION_NUMBER",
                "SERIES_NAME",
                "SERIES_ID",
                "SERIES_LEI",
                "TOTAL_ASSETS",
                "NET_ASSETS",
                "SNAPSHOT_QUARTER",
            ]
        ],
        on="ACCESSION_NUMBER",
        how="left",
        validate="many_to_one",
    )

    print(f"Ã‡Ä±karÄ±lan holding satÄ±rÄ±: {len(holdings):,}")
    print(
        "Holdingi bulunan benzersiz ETF sayÄ±sÄ±:",
        holdings["ACCESSION_NUMBER"].nunique(),
    )

    write_log(
        paths["log_output"],
        (
            f"{quarter}: {scanned_rows} holding rows scanned; "
            f"{len(holdings)} candidate holding rows retained."
        ),
    )

    return holdings


# ============================================================
# 5. IDENTIFIER ATTACHMENT
# ============================================================

def attach_identifiers(
    quarter: str,
    holdings: pd.DataFrame,
    paths: Dict[str, Path],
) -> pd.DataFrame:
    """
    Attach ISIN and ticker identifiers using HOLDING_ID.
    """
    print("\n3/4 â€” Ticker ve ISIN bilgileri eÅŸleÅŸtiriliyor...")

    target_holding_ids = set(
        holdings["HOLDING_ID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    identifier_chunks: List[pd.DataFrame] = []
    scanned_rows = 0
    matched_rows = 0

    reader = pd.read_csv(
        paths["identifiers_file"],
        sep="\t",
        dtype=str,
        usecols=IDENTIFIER_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        scanned_rows += len(chunk)

        holding_id_clean = (
            chunk["HOLDING_ID"]
            .astype(str)
            .str.strip()
        )

        matched = chunk.loc[
            holding_id_clean.isin(target_holding_ids)
        ].copy()

        if not matched.empty:
            identifier_chunks.append(matched)
            matched_rows += len(matched)

        print(
            f"\rParÃ§a: {chunk_number:,} | "
            f"Taranan: {scanned_rows:,} | "
            f"EÅŸleÅŸen: {matched_rows:,}",
            end="",
            flush=True,
        )

    print()

    if identifier_chunks:
        identifiers = pd.concat(
            identifier_chunks,
            ignore_index=True,
        )

        identifiers = (
            identifiers
            .groupby("HOLDING_ID", as_index=False)
            .agg(
                {
                    "IDENTIFIER_ISIN": safe_join_unique,
                    "IDENTIFIER_TICKER": safe_join_unique,
                    "OTHER_IDENTIFIER": safe_join_unique,
                    "OTHER_IDENTIFIER_DESC": safe_join_unique,
                }
            )
        )

        final = holdings.merge(
            identifiers,
            on="HOLDING_ID",
            how="left",
            validate="one_to_one",
        )

    else:
        print(
            "UyarÄ±: HiÃ§bir identifier kaydÄ± bulunamadÄ±. "
            "Holdings yine de kaydedilecek."
        )

        final = holdings.copy()

        final["IDENTIFIER_ISIN"] = pd.NA
        final["IDENTIFIER_TICKER"] = pd.NA
        final["OTHER_IDENTIFIER"] = pd.NA
        final["OTHER_IDENTIFIER_DESC"] = pd.NA

    write_log(
        paths["log_output"],
        (
            f"{quarter}: {scanned_rows} identifier rows scanned; "
            f"{matched_rows} identifier rows matched."
        ),
    )

    return final


# ============================================================
# 6. OUTPUT AND VALIDATION
# ============================================================

def save_outputs(
    quarter: str,
    final: pd.DataFrame,
    paths: Dict[str, Path],
) -> None:
    """
    Save holdings, summary, validation, and metadata outputs.
    """
    print("\n4/4 â€” Ã‡Ä±ktÄ±lar kaydediliyor...")

    final = final.copy()

    final["PERCENTAGE_NUMERIC"] = pd.to_numeric(
        final["PERCENTAGE"],
        errors="coerce",
    )

    final["CURRENCY_VALUE_NUMERIC"] = pd.to_numeric(
        final["CURRENCY_VALUE"],
        errors="coerce",
    )

    final = final.sort_values(
        [
            "SERIES_NAME",
            "PERCENTAGE_NUMERIC",
            "ISSUER_NAME",
        ],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    duplicate_holding_ids = int(
        final.duplicated(
            subset=[
                "ACCESSION_NUMBER",
                "HOLDING_ID",
            ],
            keep=False,
        ).sum()
    )

    final.to_parquet(
        paths["holdings_parquet"],
        index=False,
    )

    final.to_csv(
        paths["holdings_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        final.groupby(
            [
                "SNAPSHOT_QUARTER",
                "ACCESSION_NUMBER",
                "SERIES_ID",
                "SERIES_NAME",
            ],
            dropna=False,
        )
        .agg(
            HOLDING_ROWS=("HOLDING_ID", "size"),
            UNIQUE_HOLDING_IDS=("HOLDING_ID", "nunique"),
            UNIQUE_ISSUERS=("ISSUER_NAME", "nunique"),
            TICKER_AVAILABLE=(
                "IDENTIFIER_TICKER",
                lambda x: x.fillna("").ne("").sum(),
            ),
            ISIN_AVAILABLE=(
                "IDENTIFIER_ISIN",
                lambda x: x.fillna("").ne("").sum(),
            ),
            REPORTED_WEIGHT_SUM=(
                "PERCENTAGE_NUMERIC",
                "sum",
            ),
            TOTAL_CURRENCY_VALUE=(
                "CURRENCY_VALUE_NUMERIC",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("SERIES_NAME")
    )

    summary.to_csv(
        paths["summary_output"],
        index=False,
        encoding="utf-8-sig",
    )

    validation = pd.DataFrame(
        [
            {
                "SNAPSHOT_QUARTER": quarter,
                "HOLDING_ROWS": len(final),
                "UNIQUE_ACCESSIONS": (
                    final["ACCESSION_NUMBER"].nunique()
                ),
                "UNIQUE_SERIES_IDS": (
                    final["SERIES_ID"].nunique()
                ),
                "UNIQUE_SERIES_NAMES": (
                    final["SERIES_NAME"].nunique()
                ),
                "UNIQUE_HOLDING_IDS": (
                    final["HOLDING_ID"].nunique()
                ),
                "UNIQUE_ISSUERS": (
                    final["ISSUER_NAME"].nunique()
                ),
                "TICKER_AVAILABLE_ROWS": (
                    final["IDENTIFIER_TICKER"]
                    .fillna("")
                    .ne("")
                    .sum()
                ),
                "ISIN_AVAILABLE_ROWS": (
                    final["IDENTIFIER_ISIN"]
                    .fillna("")
                    .ne("")
                    .sum()
                ),
                "DUPLICATE_ACCESSION_HOLDING_ROWS": (
                    duplicate_holding_ids
                ),
                "NEGATIVE_WEIGHTS": int(
                    (
                        final["PERCENTAGE_NUMERIC"] < 0
                    ).fillna(False).sum()
                ),
                "MISSING_PERCENTAGE": int(
                    final["PERCENTAGE_NUMERIC"].isna().sum()
                ),
                "OUTPUT_CSV_EXISTS": (
                    paths["holdings_csv"].exists()
                ),
                "OUTPUT_PARQUET_EXISTS": (
                    paths["holdings_parquet"].exists()
                ),
            }
        ]
    )

    validation.to_csv(
        paths["validation_output"],
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "script": "03_extract_holdings_multi_quarter.py",
        "quarter": quarter,
        "created_at": datetime.now().isoformat(),
        "project_directory": str(PROJECT_DIR),
        "extract_directory": str(paths["extract_dir"]),
        "chunk_size": CHUNK_SIZE,
        "keywords": KEYWORDS,
        "candidate_etf_count": int(
            final["ACCESSION_NUMBER"].nunique()
        ),
        "holding_rows": int(len(final)),
        "unique_holding_ids": int(
            final["HOLDING_ID"].nunique()
        ),
        "unique_issuers": int(
            final["ISSUER_NAME"].nunique()
        ),
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key.endswith("output")
            or key.startswith("holdings_")
        },
    }

    with paths["metadata_output"].open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            metadata,
            output,
            indent=2,
            ensure_ascii=False,
        )

    write_log(
        paths["log_output"],
        (
            f"{quarter}: outputs saved successfully. "
            f"Rows={len(final)}, "
            f"ETFs={final['ACCESSION_NUMBER'].nunique()}."
        ),
    )

    print("\nÄ°ÅLEM BAÅARILI")
    print("=" * 72)
    print(f"DÃ¶nem: {quarter}")
    print(f"Nihai holding satÄ±rÄ±: {len(final):,}")
    print(
        "Holdingi bulunan ETF sayÄ±sÄ±:",
        final["ACCESSION_NUMBER"].nunique(),
    )
    print(
        "Benzersiz issuer adÄ±:",
        final["ISSUER_NAME"].nunique(),
    )
    print(
        "Ticker bulunan satÄ±r:",
        final["IDENTIFIER_TICKER"]
        .fillna("")
        .ne("")
        .sum(),
    )
    print(
        "ISIN bulunan satÄ±r:",
        final["IDENTIFIER_ISIN"]
        .fillna("")
        .ne("")
        .sum(),
    )
    print(
        "Duplicate accession-holding satÄ±rÄ±:",
        duplicate_holding_ids,
    )

    print("\nOluÅŸturulan dosyalar:")
    print(paths["candidate_output"])
    print(paths["holdings_parquet"])
    print(paths["holdings_csv"])
    print(paths["summary_output"])
    print(paths["validation_output"])
    print(paths["metadata_output"])


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError(
            "\nKullanÄ±m:\n"
            "python scripts\\03_extract_holdings_multi_quarter.py "
            "2023Q4\n"
        )

    quarter = normalize_quarter(sys.argv[1])
    paths = build_paths(quarter)

    print("=" * 72)
    print("MULTI-QUARTER SEC N-PORT HOLDINGS EXTRACTION")
    print("=" * 72)
    print(f"DÃ¶nem: {quarter}")
    print(f"Kaynak klasÃ¶r: {paths['extract_dir']}")
    print(f"Ã‡Ä±ktÄ± klasÃ¶rÃ¼: {OUTPUT_DIR}")

    require_file(paths["info_file"])
    require_file(paths["holdings_file"])
    require_file(paths["identifiers_file"])

    write_log(
        paths["log_output"],
        f"{quarter}: holdings extraction started.",
    )

    candidates = load_candidate_etfs(
        quarter=quarter,
        paths=paths,
    )

    holdings = extract_candidate_holdings(
        quarter=quarter,
        candidates=candidates,
        paths=paths,
    )

    final = attach_identifiers(
        quarter=quarter,
        holdings=holdings,
        paths=paths,
    )

    save_outputs(
        quarter=quarter,
        final=final,
        paths=paths,
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print("\n" + "=" * 72)
        print("Ä°ÅLEM BAÅARISIZ")
        print("=" * 72)
        print(f"Hata tÃ¼rÃ¼: {type(exc).__name__}")
        print(f"Hata mesajÄ±: {exc}")
        print("\nAyrÄ±ntÄ±lÄ± traceback:")
        traceback.print_exc()
        sys.exit(1)


