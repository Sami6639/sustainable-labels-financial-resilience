from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

INPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_us_company_master.csv"
)

OUTPUT_QUEUE = (
    PROJECT_DIR
    / "output"
    / "2025q4_sec_company_facts_download_queue.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_DIR
    / "output"
    / "2025q4_sec_company_facts_download_queue_summary.csv"
)


def normalize_cik(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    digits = "".join(character for character in text if character.isdigit())

    if not digits:
        return ""

    return digits.zfill(10)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Company master dosyası bulunamadı:\n{INPUT_FILE}"
        )

    print("1/3 — Company master okunuyor...")

    master = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        low_memory=False,
    )

    required_columns = {
        "CIK",
        "DISPLAY_ISSUER_NAME",
        "SEC_COMPANY_NAME",
        "SEC_TICKER",
        "MATCH_METHOD",
        "COMPANY_FACTS_ELIGIBLE",
        "ETF_COUNT",
        "MAX_PORTFOLIO_WEIGHT",
    }

    missing = required_columns.difference(master.columns)

    if missing:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing)}"
        )

    print(f"Toplam company master kaydı: {len(master):,}")

    print("\n2/3 — İndirme kuyruğu oluşturuluyor...")

    eligible_flag = (
        master["COMPANY_FACTS_ELIGIBLE"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin({"TRUE", "1", "YES", "Y"})
    )

    queue = master.loc[eligible_flag].copy()

    queue["CIK10"] = queue["CIK"].apply(normalize_cik)

    queue = queue.loc[
        queue["CIK10"].str.len().eq(10)
    ].copy()

    queue = queue.drop_duplicates(
        subset=["CIK10"]
    )

    queue["COMPANY_FACTS_URL"] = (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK"
        + queue["CIK10"]
        + ".json"
    )

    queue["DOWNLOAD_STATUS"] = "PENDING"
    queue["ATTEMPT_COUNT"] = 0
    queue["HTTP_STATUS"] = ""
    queue["LAST_ERROR"] = ""
    queue["DOWNLOADED_AT_UTC"] = ""
    queue["RAW_FILE_PATH"] = ""
    queue["FILE_SIZE_BYTES"] = ""
    queue["FACT_COUNT"] = ""
    queue["LAST_UPDATED_UTC"] = datetime.now(
        timezone.utc
    ).isoformat()

    queue = queue[
        [
            "CIK10",
            "DISPLAY_ISSUER_NAME",
            "SEC_COMPANY_NAME",
            "SEC_TICKER",
            "MATCH_METHOD",
            "ETF_COUNT",
            "MAX_PORTFOLIO_WEIGHT",
            "COMPANY_FACTS_URL",
            "DOWNLOAD_STATUS",
            "ATTEMPT_COUNT",
            "HTTP_STATUS",
            "LAST_ERROR",
            "DOWNLOADED_AT_UTC",
            "RAW_FILE_PATH",
            "FILE_SIZE_BYTES",
            "FACT_COUNT",
            "LAST_UPDATED_UTC",
        ]
    ].sort_values(
        [
            "ETF_COUNT",
            "MAX_PORTFOLIO_WEIGHT",
            "DISPLAY_ISSUER_NAME",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    queue.to_csv(
        OUTPUT_QUEUE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Kuyruktaki benzersiz CIK sayısı: {len(queue):,}")

    print("\n3/3 — Özet kaydediliyor...")

    summary = pd.DataFrame(
        {
            "METRIC": [
                "COMPANY_MASTER_RECORDS",
                "COMPANY_FACTS_ELIGIBLE_RECORDS",
                "VALID_UNIQUE_CIK10",
                "INITIAL_PENDING_DOWNLOADS",
            ],
            "VALUE": [
                len(master),
                int(eligible_flag.sum()),
                len(queue),
                int(
                    (
                        queue["DOWNLOAD_STATUS"]
                        == "PENDING"
                    ).sum()
                ),
            ],
        }
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDOWNLOAD QUEUE HAZIR")
    print("=" * 70)
    print(f"Company master kaydı: {len(master):,}")
    print(
        "Company Facts için uygun kayıt:",
        f"{int(eligible_flag.sum()):,}",
    )
    print(
        "Benzersiz geçerli CIK10:",
        f"{len(queue):,}",
    )
    print(
        "Başlangıç PENDING sayısı:",
        f"{int((queue['DOWNLOAD_STATUS'] == 'PENDING').sum()):,}",
    )

    print("\nOluşturulan dosyalar:")
    print(OUTPUT_QUEUE)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()