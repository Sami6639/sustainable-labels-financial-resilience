from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import time

import pandas as pd
import requests


# ============================================================
# DOSYA YOLLARI
# ============================================================

PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

QUEUE_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_sec_company_facts_download_queue.csv"
)

RAW_OUTPUT_DIR = (
    PROJECT_DIR
    / "sec_raw"
    / "companyfacts"
)

LOG_FILE = (
    PROJECT_DIR
    / "logs"
    / "sec_company_facts_download_log.csv"
)

SUMMARY_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_sec_company_facts_download_summary.csv"
)


# ============================================================
# SEC ERİŞİM AYARLARI
# ============================================================

HEADERS = {
    "User-Agent": (
        "Sami Kucukoglu academic research "
        "samikucukoglu@yahoo.com"
    ),
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

REQUEST_TIMEOUT_SECONDS = 60

# Muhafazakâr erişim hızı: saniyede yaklaşık 4 istek.
REQUEST_INTERVAL_SECONDS = 0.25

MAX_ATTEMPTS_PER_RUN = 3

RETRY_WAIT_SECONDS = {
    1: 3,
    2: 8,
    3: 20,
}

RETRYABLE_HTTP_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default

        text = str(value).strip()

        if not text:
            return default

        return int(float(text))

    except (TypeError, ValueError):
        return default


def normalize_cik(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    digits = "".join(
        character
        for character in text
        if character.isdigit()
    )

    if not digits:
        return ""

    return digits.zfill(10)


def count_company_facts(payload: dict) -> int:
    """
    Company Facts JSON içindeki toplam concept sayısını hesaplar.
    Bu sayı finansal gözlem sayısı değil, bildirilen XBRL kavram sayısıdır.
    """
    facts = payload.get("facts", {})

    if not isinstance(facts, dict):
        return 0

    total = 0

    for taxonomy_data in facts.values():
        if not isinstance(taxonomy_data, dict):
            continue

        total += len(taxonomy_data)

    return total


def is_valid_companyfacts_payload(
    payload: object,
    expected_cik: str,
) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "JSON root is not an object"

    if "facts" not in payload:
        return False, "JSON does not contain facts"

    if not isinstance(payload.get("facts"), dict):
        return False, "facts is not an object"

    returned_cik = normalize_cik(
        payload.get("cik")
    )

    if returned_cik and returned_cik != expected_cik:
        return (
            False,
            f"CIK mismatch: expected {expected_cik}, "
            f"received {returned_cik}",
        )

    return True, ""


def save_queue(queue: pd.DataFrame) -> None:
    queue.to_csv(
        QUEUE_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def append_log(log_row: dict) -> None:
    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_df = pd.DataFrame([log_row])

    if LOG_FILE.exists():
        log_df.to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        log_df.to_csv(
            LOG_FILE,
            index=False,
            encoding="utf-8-sig",
        )


def existing_file_is_valid(
    raw_file: Path,
    expected_cik: str,
) -> tuple[bool, int]:
    if not raw_file.exists():
        return False, 0

    if raw_file.stat().st_size == 0:
        return False, 0

    try:
        with raw_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        valid, _ = is_valid_companyfacts_payload(
            payload,
            expected_cik,
        )

        if not valid:
            return False, 0

        return True, count_company_facts(payload)

    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False, 0


def write_json_atomically(
    payload: dict,
    destination: Path,
) -> None:
    temporary_file = destination.with_suffix(
        ".json.tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    temporary_file.replace(destination)


def download_one_company(
    session: requests.Session,
    cik10: str,
    url: str,
    raw_file: Path,
) -> dict:
    last_error = ""
    last_http_status = ""

    for attempt in range(
        1,
        MAX_ATTEMPTS_PER_RUN + 1,
    ):
        started_at = utc_now()

        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            last_http_status = str(
                response.status_code
            )

            if response.status_code == 200:
                try:
                    payload = response.json()

                except requests.exceptions.JSONDecodeError as exc:
                    last_error = (
                        f"Invalid JSON: {exc}"
                    )

                else:
                    valid, validation_error = (
                        is_valid_companyfacts_payload(
                            payload,
                            cik10,
                        )
                    )

                    if valid:
                        write_json_atomically(
                            payload,
                            raw_file,
                        )

                        fact_count = count_company_facts(
                            payload
                        )

                        return {
                            "success": True,
                            "status": "SUCCESS",
                            "http_status": "200",
                            "error": "",
                            "attempts_this_run": attempt,
                            "fact_count": fact_count,
                            "file_size_bytes": (
                                raw_file.stat().st_size
                            ),
                            "downloaded_at_utc": utc_now(),
                        }

                    last_error = validation_error

            elif response.status_code == 404:
                return {
                    "success": False,
                    "status": "NOT_FOUND",
                    "http_status": "404",
                    "error": "SEC Company Facts record not found",
                    "attempts_this_run": attempt,
                    "fact_count": "",
                    "file_size_bytes": "",
                    "downloaded_at_utc": "",
                }

            else:
                response_message = (
                    response.text[:300]
                    .replace("\n", " ")
                    .strip()
                )

                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response_message}"
                )

                if (
                    response.status_code
                    not in RETRYABLE_HTTP_CODES
                ):
                    return {
                        "success": False,
                        "status": "FAILED_PERMANENT",
                        "http_status": last_http_status,
                        "error": last_error,
                        "attempts_this_run": attempt,
                        "fact_count": "",
                        "file_size_bytes": "",
                        "downloaded_at_utc": "",
                    }

        except requests.exceptions.RequestException as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

        append_log(
            {
                "TIMESTAMP_UTC": utc_now(),
                "CIK10": cik10,
                "ATTEMPT_IN_RUN": attempt,
                "STARTED_AT_UTC": started_at,
                "HTTP_STATUS": last_http_status,
                "SUCCESS": False,
                "ERROR": last_error,
                "URL": url,
            }
        )

        if attempt < MAX_ATTEMPTS_PER_RUN:
            time.sleep(
                RETRY_WAIT_SECONDS.get(
                    attempt,
                    20,
                )
            )

    return {
        "success": False,
        "status": "FAILED_RETRYABLE",
        "http_status": last_http_status,
        "error": last_error,
        "attempts_this_run": MAX_ATTEMPTS_PER_RUN,
        "fact_count": "",
        "file_size_bytes": "",
        "downloaded_at_utc": "",
    }


def prepare_queue() -> pd.DataFrame:
    if not QUEUE_FILE.exists():
        raise FileNotFoundError(
            f"İndirme kuyruğu bulunamadı:\n{QUEUE_FILE}"
        )

    queue = pd.read_csv(
        QUEUE_FILE,
        dtype=str,
        low_memory=False,
    )

    required_columns = {
        "CIK10",
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
    }

    missing = required_columns.difference(
        queue.columns
    )

    if missing:
        raise ValueError(
            f"Kuyrukta eksik sütunlar: "
            f"{sorted(missing)}"
        )

    queue["CIK10"] = (
        queue["CIK10"]
        .apply(normalize_cik)
    )

    invalid_cik = (
        ~queue["CIK10"].str.len().eq(10)
    )

    if invalid_cik.any():
        raise ValueError(
            "Kuyrukta geçersiz CIK10 bulundu. "
            f"Geçersiz satır sayısı: "
            f"{int(invalid_cik.sum())}"
        )

    return queue


def update_existing_downloads(
    queue: pd.DataFrame,
) -> pd.DataFrame:
    """
    Diskte geçerli JSON varsa kuyruk durumunu SUCCESS yapar.
    Böylece script kaldığı yerden devam eder.
    """
    recovered = 0

    for index, row in queue.iterrows():
        cik10 = row["CIK10"]

        raw_file = (
            RAW_OUTPUT_DIR
            / f"CIK{cik10}.json"
        )

        valid, fact_count = (
            existing_file_is_valid(
                raw_file,
                cik10,
            )
        )

        if not valid:
            continue

        queue.at[
            index,
            "DOWNLOAD_STATUS",
        ] = "SUCCESS"

        queue.at[
            index,
            "HTTP_STATUS",
        ] = "200"

        queue.at[
            index,
            "LAST_ERROR",
        ] = ""

        queue.at[
            index,
            "RAW_FILE_PATH",
        ] = str(raw_file)

        queue.at[
            index,
            "FILE_SIZE_BYTES",
        ] = str(raw_file.stat().st_size)

        queue.at[
            index,
            "FACT_COUNT",
        ] = str(fact_count)

        queue.at[
            index,
            "LAST_UPDATED_UTC",
        ] = utc_now()

        recovered += 1

    print(
        "Diskten doğrulanan mevcut başarılı dosya:",
        f"{recovered:,}",
    )

    return queue


def save_summary(queue: pd.DataFrame) -> None:
    status_counts = (
        queue["DOWNLOAD_STATUS"]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
    )

    summary_rows = [
        {
            "METRIC": "TOTAL_QUEUE_RECORDS",
            "VALUE": len(queue),
        },
        {
            "METRIC": "TOTAL_RAW_JSON_FILES",
            "VALUE": len(
                list(
                    RAW_OUTPUT_DIR.glob(
                        "CIK*.json"
                    )
                )
            ),
        },
    ]

    for status, count in status_counts.items():
        summary_rows.append(
            {
                "METRIC": (
                    f"STATUS_{status}"
                ),
                "VALUE": int(count),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    try:
        RAW_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        LOG_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("1/4 — İndirme kuyruğu okunuyor...")

        queue = prepare_queue()

        print(
            f"Toplam kuyruk kaydı: "
            f"{len(queue):,}"
        )

        print(
            "\n2/4 — Mevcut ham dosyalar "
            "doğrulanıyor..."
        )

        queue = update_existing_downloads(
            queue
        )

        save_queue(queue)

        pending_mask = ~queue[
            "DOWNLOAD_STATUS"
        ].isin(
            {
                "SUCCESS",
                "NOT_FOUND",
                "FAILED_PERMANENT",
            }
        )

        pending_indices = list(
            queue.index[pending_mask]
        )

        print(
            "\n3/4 — SEC Company Facts "
            "dosyaları indiriliyor..."
        )

        print(
            "Bu çalıştırmada indirilecek kayıt:",
            f"{len(pending_indices):,}",
        )

        session = requests.Session()
        session.headers.update(HEADERS)

        run_success = 0
        run_failed = 0

        total_to_process = len(
            pending_indices
        )

        for position, index in enumerate(
            pending_indices,
            start=1,
        ):
            row = queue.loc[index]

            cik10 = row["CIK10"]
            url = row["COMPANY_FACTS_URL"]

            raw_file = (
                RAW_OUTPUT_DIR
                / f"CIK{cik10}.json"
            )

            previous_attempt_count = safe_int(
                row["ATTEMPT_COUNT"]
            )

            result = download_one_company(
                session=session,
                cik10=cik10,
                url=url,
                raw_file=raw_file,
            )

            queue.at[
                index,
                "DOWNLOAD_STATUS",
            ] = result["status"]

            queue.at[
                index,
                "ATTEMPT_COUNT",
            ] = str(
                previous_attempt_count
                + result[
                    "attempts_this_run"
                ]
            )

            queue.at[
                index,
                "HTTP_STATUS",
            ] = result["http_status"]

            queue.at[
                index,
                "LAST_ERROR",
            ] = result["error"]

            queue.at[
                index,
                "LAST_UPDATED_UTC",
            ] = utc_now()

            if result["success"]:
                queue.at[
                    index,
                    "DOWNLOADED_AT_UTC",
                ] = result[
                    "downloaded_at_utc"
                ]

                queue.at[
                    index,
                    "RAW_FILE_PATH",
                ] = str(raw_file)

                queue.at[
                    index,
                    "FILE_SIZE_BYTES",
                ] = str(
                    result[
                        "file_size_bytes"
                    ]
                )

                queue.at[
                    index,
                    "FACT_COUNT",
                ] = str(
                    result["fact_count"]
                )

                run_success += 1

            else:
                run_failed += 1

            append_log(
                {
                    "TIMESTAMP_UTC": utc_now(),
                    "CIK10": cik10,
                    "ATTEMPT_IN_RUN": result[
                        "attempts_this_run"
                    ],
                    "STARTED_AT_UTC": "",
                    "HTTP_STATUS": result[
                        "http_status"
                    ],
                    "SUCCESS": result[
                        "success"
                    ],
                    "ERROR": result["error"],
                    "URL": url,
                }
            )

            # Her 25 şirkette kuyruğu diske yaz.
            if (
                position % 25 == 0
                or position == total_to_process
            ):
                save_queue(queue)
                save_summary(queue)

            print(
                f"\rİşlenen: "
                f"{position:,}/"
                f"{total_to_process:,} | "
                f"Başarılı: "
                f"{run_success:,} | "
                f"Başarısız: "
                f"{run_failed:,} | "
                f"CIK: {cik10}",
                end="",
                flush=True,
            )

            time.sleep(
                REQUEST_INTERVAL_SECONDS
            )

        print()

        session.close()

        print(
            "\n4/4 — Nihai kuyruk ve özet "
            "kaydediliyor..."
        )

        save_queue(queue)
        save_summary(queue)

        status_counts = (
            queue["DOWNLOAD_STATUS"]
            .fillna("UNKNOWN")
            .value_counts(dropna=False)
        )

        print("\nCOMPANY FACTS İNDİRME TAMAMLANDI")
        print("=" * 70)
        print(f"Toplam kuyruk: {len(queue):,}")
        print(
            "Bu çalıştırmada başarılı:",
            f"{run_success:,}",
        )
        print(
            "Bu çalıştırmada başarısız:",
            f"{run_failed:,}",
        )

        print("\nNihai durum dağılımı:")
        print(status_counts.to_string())

        print("\nHam JSON klasörü:")
        print(RAW_OUTPUT_DIR)

        print("\nGüncellenen kuyruk:")
        print(QUEUE_FILE)

        print("\nİndirme günlüğü:")
        print(LOG_FILE)

        print("\nÖzet dosyası:")
        print(SUMMARY_FILE)

    except KeyboardInterrupt:
        print(
            "\n\nİşlem kullanıcı tarafından durduruldu."
        )
        print(
            "Mevcut ilerleme kuyruk dosyasından "
            "devam ettirilebilir."
        )
        sys.exit(130)

    except Exception as exc:
        print("\nİŞLEM BAŞARISIZ")
        print(
            f"Hata türü: "
            f"{type(exc).__name__}"
        )
        print(f"Hata mesajı: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()