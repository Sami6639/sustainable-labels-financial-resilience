from pathlib import Path
import re
import sys

import pandas as pd
import requests


PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

INPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_us_firm_identifier_mapping.csv"
)

RAW_SEC_FILE = (
    PROJECT_DIR
    / "data"
    / "sec_company_tickers.json"
)

OUTPUT_COMPANY_MASTER = (
    PROJECT_DIR
    / "output"
    / "2025q4_us_company_master.csv"
)

OUTPUT_UNMATCHED = (
    PROJECT_DIR
    / "output"
    / "2025q4_us_company_master_unmatched.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_DIR
    / "output"
    / "2025q4_us_company_master_summary.csv"
)

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

HEADERS = {
    "User-Agent": (
        "Sami Kucukoglu academic research "
        "samikucukoglu@yahoo.com"
    ),
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


def normalize_company_name(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = text.replace("&", " AND ")

    text = re.sub(
        r"\b("
        r"INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|"
        r"LIMITED|LTD|PLC|LLC|LP|L P|SA|NV|AG|SE|"
        r"THE|HOLDINGS|HOLDING|GROUP"
        r")\b",
        " ",
        text,
    )

    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def download_sec_ticker_file() -> None:
    RAW_SEC_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1/4 — SEC company ticker listesi indiriliyor...")

    response = requests.get(
        SEC_TICKER_URL,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    RAW_SEC_FILE.write_bytes(response.content)

    if RAW_SEC_FILE.stat().st_size == 0:
        raise RuntimeError(
            "SEC ticker dosyası boş indirildi."
        )

    print(f"Dosya indirildi:\n{RAW_SEC_FILE}")


def load_sec_companies() -> pd.DataFrame:
    print("\n2/4 — SEC şirket listesi okunuyor...")

    raw = pd.read_json(
        RAW_SEC_FILE,
        orient="index",
        dtype=False,
    )

    required = {
        "cik_str",
        "ticker",
        "title",
    }

    missing = required.difference(raw.columns)

    if missing:
        raise ValueError(
            f"SEC dosyasında eksik sütunlar: {sorted(missing)}"
        )

    sec = raw[
        [
            "cik_str",
            "ticker",
            "title",
        ]
    ].copy()

    sec["CIK"] = (
        pd.to_numeric(
            sec["cik_str"],
            errors="coerce",
        )
        .astype("Int64")
        .astype(str)
        .str.zfill(10)
    )

    sec["SEC_TICKER"] = (
        sec["ticker"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    sec["SEC_COMPANY_NAME"] = (
        sec["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    sec["SEC_NAME_STANDARDIZED"] = (
        sec["SEC_COMPANY_NAME"]
        .apply(normalize_company_name)
    )

    sec = sec[
        [
            "CIK",
            "SEC_TICKER",
            "SEC_COMPANY_NAME",
            "SEC_NAME_STANDARDIZED",
        ]
    ].drop_duplicates()

    print(f"SEC şirket-ticker kaydı: {len(sec):,}")

    return sec


def build_company_master(
    sec: pd.DataFrame,
) -> pd.DataFrame:
    print("\n3/4 — Mevcut firma evreni SEC ile eşleştiriliyor...")

    firms = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        low_memory=False,
    )

    firms["PRIMARY_TICKER"] = (
        firms["PRIMARY_TICKER"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    firms["COMPANY_NAME_STANDARDIZED"] = (
        firms["DISPLAY_ISSUER_NAME"]
        .apply(normalize_company_name)
    )

    # Önceki denemelerden kalabilecek SEC sütunlarını temizle.
    # Böylece merge sırasında _x / _y sonekleri oluşmaz.
    firms = firms.drop(
        columns=[
            "CIK",
            "SEC_TICKER",
            "SEC_COMPANY_NAME",
            "SEC_NAME_STANDARDIZED",
        ],
        errors="ignore",
    )

    # İlk aşama: güvenilir ticker eşleşmesi
    ticker_map = (
        sec.loc[sec["SEC_TICKER"].ne("")]
        .sort_values(
            [
                "SEC_TICKER",
                "CIK",
            ]
        )
        .drop_duplicates(
            subset=["SEC_TICKER"],
            keep="first",
        )
    )

    master = firms.merge(
        ticker_map,
        left_on="PRIMARY_TICKER",
        right_on="SEC_TICKER",
        how="left",
        validate="many_to_one",
    )

    master["MATCH_METHOD"] = ""

    ticker_match = (
        master["PRIMARY_TICKER"].ne("")
        & master["CIK"].notna()
    )

    master.loc[
        ticker_match,
        "MATCH_METHOD",
    ] = "EXACT_TICKER"

    # İkinci aşama: yalnızca benzersiz standart şirket adı eşleşmesi
    unmatched_mask = master["CIK"].isna()

    unique_name_map = (
        sec.loc[
            sec["SEC_NAME_STANDARDIZED"].ne("")
        ]
        .groupby(
            "SEC_NAME_STANDARDIZED",
            as_index=False,
        )
        .filter(
            lambda group: group["CIK"].nunique() == 1
        )
        .sort_values(
            [
                "SEC_NAME_STANDARDIZED",
                "CIK",
            ]
        )
        .drop_duplicates(
            subset=["SEC_NAME_STANDARDIZED"],
            keep="first",
        )
    )

    name_lookup = unique_name_map.set_index(
        "SEC_NAME_STANDARDIZED"
    )

    for index in master.index[unmatched_mask]:
        company_name = master.at[
            index,
            "COMPANY_NAME_STANDARDIZED",
        ]

        if not company_name:
            continue

        if company_name not in name_lookup.index:
            continue

        matched = name_lookup.loc[company_name]

        master.at[index, "CIK"] = matched["CIK"]
        master.at[index, "SEC_TICKER"] = matched["SEC_TICKER"]
        master.at[index, "SEC_COMPANY_NAME"] = (
            matched["SEC_COMPANY_NAME"]
        )
        master.at[index, "SEC_NAME_STANDARDIZED"] = company_name
        master.at[index, "MATCH_METHOD"] = (
            "UNIQUE_STANDARDIZED_NAME"
        )

    master["CIK_MATCHED"] = master["CIK"].notna()

    master["TICKER_CONFLICT"] = (
        master["PRIMARY_TICKER"].ne("")
        & master["SEC_TICKER"].notna()
        & master["SEC_TICKER"].ne("")
        & (
            master["PRIMARY_TICKER"]
            != master["SEC_TICKER"]
        )
    )

    master["NAME_SIMILARITY_EXACT"] = (
        master["COMPANY_NAME_STANDARDIZED"].ne("")
        & master["SEC_NAME_STANDARDIZED"].notna()
        & (
            master["COMPANY_NAME_STANDARDIZED"]
            == master["SEC_NAME_STANDARDIZED"]
        )
    )

    master["COMPANY_FACTS_ELIGIBLE"] = (
        master["CIK_MATCHED"]
        & ~master["TICKER_CONFLICT"]
    )

    master = master.sort_values(
        [
            "COMPANY_FACTS_ELIGIBLE",
            "MATCH_METHOD",
            "ETF_COUNT",
            "MAX_PORTFOLIO_WEIGHT",
            "DISPLAY_ISSUER_NAME",
        ],
        ascending=[
            False,
            True,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return master


def save_outputs(master: pd.DataFrame) -> None:
    print("\n4/4 — Company master çıktıları kaydediliyor...")

    master.to_csv(
        OUTPUT_COMPANY_MASTER,
        index=False,
        encoding="utf-8-sig",
    )

    unmatched = master.loc[
        ~master["COMPANY_FACTS_ELIGIBLE"]
    ].copy()

    unmatched.to_csv(
        OUTPUT_UNMATCHED,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        {
            "METRIC": [
                "US_FIRM_RECORDS",
                "EXACT_TICKER_MATCHES",
                "UNIQUE_NAME_MATCHES",
                "TOTAL_CIK_MATCHES",
                "COMPANY_FACTS_ELIGIBLE",
                "TICKER_CONFLICTS",
                "UNMATCHED_RECORDS",
            ],
            "VALUE": [
                len(master),
                int(
                    (
                        master["MATCH_METHOD"]
                        == "EXACT_TICKER"
                    ).sum()
                ),
                int(
                    (
                        master["MATCH_METHOD"]
                        == "UNIQUE_STANDARDIZED_NAME"
                    ).sum()
                ),
                int(master["CIK_MATCHED"].sum()),
                int(
                    master[
                        "COMPANY_FACTS_ELIGIBLE"
                    ].sum()
                ),
                int(master["TICKER_CONFLICT"].sum()),
                len(unmatched),
            ],
        }
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nCOMPANY MASTER HAZIR")
    print("=" * 70)
    print(f"ABD firma kaydı: {len(master):,}")
    print(
        "Kesin ticker eşleşmesi:",
        f"{int((master['MATCH_METHOD'] == 'EXACT_TICKER').sum()):,}",
    )
    print(
        "Benzersiz isim eşleşmesi:",
        f"{int((master['MATCH_METHOD'] == 'UNIQUE_STANDARDIZED_NAME').sum()):,}",
    )
    print(
        "Toplam CIK eşleşmesi:",
        f"{int(master['CIK_MATCHED'].sum()):,}",
    )
    print(
        "Company Facts için uygun:",
        f"{int(master['COMPANY_FACTS_ELIGIBLE'].sum()):,}",
    )
    print(
        "Ticker çelişkisi:",
        f"{int(master['TICKER_CONFLICT'].sum()):,}",
    )
    print(
        "Eşleşmeyen / inceleme gereken:",
        f"{len(unmatched):,}",
    )

    print("\nOluşturulan dosyalar:")
    print(RAW_SEC_FILE)
    print(OUTPUT_COMPANY_MASTER)
    print(OUTPUT_UNMATCHED)
    print(OUTPUT_SUMMARY)


def main() -> None:
    try:
        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Girdi dosyası bulunamadı:\n{INPUT_FILE}"
            )

        download_sec_ticker_file()
        sec = load_sec_companies()
        master = build_company_master(sec)
        save_outputs(master)

    except Exception as exc:
        print("\nİŞLEM BAŞARISIZ")
        print(f"Hata türü: {type(exc).__name__}")
        print(f"Hata mesajı: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()