from pathlib import Path
import sys

import pandas as pd


# ============================================================
# DOSYA YOLLARI
# ============================================================

PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

EXTRACT_DIR = (
    PROJECT_DIR
    / "sec_raw"
    / "2025Q4"
    / "extracted"
)

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INFO_FILE = EXTRACT_DIR / "FUND_REPORTED_INFO.tsv"
HOLDINGS_FILE = EXTRACT_DIR / "FUND_REPORTED_HOLDING.tsv"
IDENTIFIERS_FILE = EXTRACT_DIR / "IDENTIFIERS.tsv"

ETF_UNIVERSE_OUTPUT = (
    OUTPUT_DIR / "2025q4_sustainable_etf_candidates.csv"
)

HOLDINGS_OUTPUT_PARQUET = (
    OUTPUT_DIR / "2025q4_sustainable_etf_holdings.parquet"
)

HOLDINGS_OUTPUT_CSV = (
    OUTPUT_DIR / "2025q4_sustainable_etf_holdings.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "2025q4_sustainable_etf_holdings_summary.csv"
)


# ============================================================
# ADAY EVREN İÇİN KEŞİF ANAHTAR KELİMELERİ
#
# Bu liste nihai örneklem değildir.
# Yalnızca geniş aday ETF evrenini belirler.
# ============================================================

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


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def require_file(file_path: Path) -> None:
    """Gerekli dosyanın gerçekten mevcut olduğunu kontrol eder."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı:\n{file_path}"
        )

    if file_path.stat().st_size == 0:
        raise RuntimeError(
            f"Dosya boş görünüyor:\n{file_path}"
        )


def load_candidate_etfs() -> pd.DataFrame:
    """2025 Q4 içinden geniş sürdürülebilir ETF aday evrenini çıkarır."""

    print("\n1/4 — Fon bilgileri okunuyor...")

    info = pd.read_csv(
        INFO_FILE,
        sep="\t",
        dtype=str,
        usecols=[
            "ACCESSION_NUMBER",
            "SERIES_NAME",
            "SERIES_ID",
            "SERIES_LEI",
            "TOTAL_ASSETS",
            "NET_ASSETS",
        ],
        low_memory=False,
    )

    keyword_pattern = "|".join(KEYWORDS)

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

    candidates = candidates.sort_values(
        ["SERIES_NAME", "ACCESSION_NUMBER"]
    ).reset_index(drop=True)

    candidates.to_csv(
        ETF_UNIVERSE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Toplam fon satırı: {len(info):,}")
    print(f"Aday ETF sayısı: {len(candidates):,}")
    print(f"Aday evren kaydedildi:\n{ETF_UNIVERSE_OUTPUT}")

    return candidates


def extract_candidate_holdings(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Büyük holdings dosyasını parçalar halinde tarar."""

    print("\n2/4 — Holdings dosyası taranıyor...")

    target_accessions = set(
        candidates["ACCESSION_NUMBER"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    selected_columns = [
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

    matched_chunks = []
    scanned_rows = 0
    matched_rows = 0

    reader = pd.read_csv(
        HOLDINGS_FILE,
        sep="\t",
        dtype=str,
        usecols=selected_columns,
        chunksize=200_000,
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
            f"\rParça: {chunk_number:,} | "
            f"Taranan: {scanned_rows:,} | "
            f"Eşleşen: {matched_rows:,}",
            end="",
            flush=True,
        )

    print()

    if not matched_chunks:
        raise RuntimeError(
            "Aday ETF accession numaraları için hiçbir holding "
            "kaydı bulunamadı."
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
            ]
        ],
        on="ACCESSION_NUMBER",
        how="left",
        validate="many_to_one",
    )

    print(f"Çıkarılan holding satırı: {len(holdings):,}")
    print(
        "Holdingi bulunan benzersiz ETF sayısı:",
        holdings["ACCESSION_NUMBER"].nunique(),
    )

    return holdings


def attach_identifiers(
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Holding ID üzerinden ISIN ve ticker bilgilerini ekler."""

    print("\n3/4 — Ticker ve ISIN bilgileri eşleştiriliyor...")

    target_holding_ids = set(
        holdings["HOLDING_ID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    identifier_chunks = []
    scanned_rows = 0
    matched_rows = 0

    reader = pd.read_csv(
        IDENTIFIERS_FILE,
        sep="\t",
        dtype=str,
        usecols=[
            "HOLDING_ID",
            "IDENTIFIER_ISIN",
            "IDENTIFIER_TICKER",
            "OTHER_IDENTIFIER",
            "OTHER_IDENTIFIER_DESC",
        ],
        chunksize=200_000,
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
            f"\rParça: {chunk_number:,} | "
            f"Taranan: {scanned_rows:,} | "
            f"Eşleşen: {matched_rows:,}",
            end="",
            flush=True,
        )

    print()

    if identifier_chunks:
        identifiers = pd.concat(
            identifier_chunks,
            ignore_index=True,
        )

        # Aynı holding için birden fazla identifier satırı bulunabilir.
        # Ticker ve ISIN değerlerini kaybetmeden tek satıra indiriyoruz.
        identifiers = (
            identifiers
            .groupby("HOLDING_ID", as_index=False)
            .agg(
                {
                    "IDENTIFIER_ISIN": lambda x: "|".join(
                        sorted(set(x.dropna().astype(str)))
                    ),
                    "IDENTIFIER_TICKER": lambda x: "|".join(
                        sorted(set(x.dropna().astype(str)))
                    ),
                    "OTHER_IDENTIFIER": lambda x: "|".join(
                        sorted(set(x.dropna().astype(str)))
                    ),
                    "OTHER_IDENTIFIER_DESC": lambda x: "|".join(
                        sorted(set(x.dropna().astype(str)))
                    ),
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
            "Uyarı: Hiçbir identifier kaydı bulunamadı. "
            "Holdings yine de kaydedilecek."
        )

        final = holdings.copy()

        final["IDENTIFIER_ISIN"] = pd.NA
        final["IDENTIFIER_TICKER"] = pd.NA
        final["OTHER_IDENTIFIER"] = pd.NA
        final["OTHER_IDENTIFIER_DESC"] = pd.NA

    return final


def save_outputs(final: pd.DataFrame) -> None:
    """Nihai dosyaları ve özet tabloyu kaydeder."""

    print("\n4/4 — Çıktılar kaydediliyor...")

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

    final.to_parquet(
        HOLDINGS_OUTPUT_PARQUET,
        index=False,
    )

    final.to_csv(
        HOLDINGS_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        final.groupby(
            [
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
        )
        .reset_index()
        .sort_values("SERIES_NAME")
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nİŞLEM BAŞARILI")
    print("=" * 70)
    print(f"Nihai holding satırı: {len(final):,}")
    print(
        "Holdingi bulunan ETF sayısı:",
        final["ACCESSION_NUMBER"].nunique(),
    )
    print(
        "Benzersiz issuer adı:",
        final["ISSUER_NAME"].nunique(),
    )
    print(
        "Ticker bulunan satır:",
        final["IDENTIFIER_TICKER"]
        .fillna("")
        .ne("")
        .sum(),
    )
    print(
        "ISIN bulunan satır:",
        final["IDENTIFIER_ISIN"]
        .fillna("")
        .ne("")
        .sum(),
    )

    print("\nOluşturulan dosyalar:")
    print(ETF_UNIVERSE_OUTPUT)
    print(HOLDINGS_OUTPUT_PARQUET)
    print(HOLDINGS_OUTPUT_CSV)
    print(SUMMARY_OUTPUT)


def main() -> None:
    try:
        require_file(INFO_FILE)
        require_file(HOLDINGS_FILE)
        require_file(IDENTIFIERS_FILE)

        candidates = load_candidate_etfs()
        holdings = extract_candidate_holdings(candidates)
        final = attach_identifiers(holdings)
        save_outputs(final)

    except Exception as exc:
        print("\nİŞLEM BAŞARISIZ")
        print(f"Hata türü: {type(exc).__name__}")
        print(f"Hata mesajı: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()