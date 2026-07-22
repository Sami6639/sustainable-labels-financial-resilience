from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd


# ============================================================
# DOSYA YOLLARI
# ============================================================

PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

RAW_DIR = (
    PROJECT_DIR
    / "sec_raw"
    / "companyfacts"
)

QUEUE_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_sec_company_facts_download_queue.csv"
)

OUTPUT_LONG = (
    PROJECT_DIR
    / "output"
    / "sec_company_fundamentals_annual_long.csv"
)

OUTPUT_WIDE = (
    PROJECT_DIR
    / "output"
    / "sec_company_fundamentals_annual_wide.parquet"
)

OUTPUT_COVERAGE = (
    PROJECT_DIR
    / "output"
    / "sec_company_fundamentals_coverage.csv"
)

OUTPUT_TAG_USAGE = (
    PROJECT_DIR
    / "output"
    / "sec_company_fundamentals_tag_usage.csv"
)

OUTPUT_ERRORS = (
    PROJECT_DIR
    / "logs"
    / "sec_company_fundamentals_extraction_errors.csv"
)


# ============================================================
# DÖNEM AYARLARI
# ============================================================

START_YEAR = 2010
END_YEAR = 2025

ALLOWED_FORMS = {
    "10-K",
    "10-K/A",
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
}


# ============================================================
# FİNANSAL DEĞİŞKENLER VE XBRL ETİKET ADAYLARI
#
# Etiketler öncelik sırasındadır.
# İlk kullanılabilir etiket seçilir.
# ============================================================

VARIABLE_TAGS = {
    "TOTAL_ASSETS": [
        "Assets",
    ],

    "TOTAL_LIABILITIES": [
        "Liabilities",
        "LiabilitiesCurrent",
    ],

    "STOCKHOLDERS_EQUITY": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
    ],

    "CASH_AND_EQUIVALENTS": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndDueFromBanks",
    ],

    "TOTAL_DEBT": [
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "DebtCurrent",
        "DebtAndCapitalLeaseObligations",
    ],

    "REVENUE": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],

    "NET_INCOME": [
        "NetIncomeLoss",
        "ProfitLoss",
        "IncomeLossFromContinuingOperations",
    ],

    "OPERATING_INCOME": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],

    "CAPEX": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],

    "R_AND_D": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],

    "DEPRECIATION_AMORTIZATION": [
        "DepreciationDepletionAndAmortization",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
        "Depreciation",
    ],

    "OPERATING_CASH_FLOW": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def normalize_cik(value: object) -> str:
    if pd.isna(value):
        return ""

    digits = "".join(
        character
        for character in str(value)
        if character.isdigit()
    )

    return digits.zfill(10) if digits else ""


def safe_numeric(value: object) -> float:
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return np.nan


def load_json(file_path: Path) -> dict:
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_us_gaap_facts(payload: dict) -> dict:
    facts = payload.get("facts", {})

    if not isinstance(facts, dict):
        return {}

    us_gaap = facts.get("us-gaap", {})

    if isinstance(us_gaap, dict):
        return us_gaap

    return {}


def choose_unit_entries(
    concept: dict,
) -> tuple[list[dict], str]:
    units = concept.get("units", {})

    if not isinstance(units, dict):
        return [], ""

    preferred_units = [
        "USD",
        "USD/shares",
        "shares",
        "pure",
    ]

    for unit in preferred_units:
        entries = units.get(unit)

        if isinstance(entries, list) and entries:
            return entries, unit

    for unit, entries in units.items():
        if isinstance(entries, list) and entries:
            return entries, str(unit)

    return [], ""


def fiscal_year_from_entry(entry: dict) -> int | None:
    fy = entry.get("fy")

    try:
        if fy is not None:
            return int(fy)
    except (TypeError, ValueError):
        pass

    end_date = pd.to_datetime(
        entry.get("end"),
        errors="coerce",
    )

    if pd.notna(end_date):
        return int(end_date.year)

    return None


def filter_annual_entries(
    entries: list[dict],
) -> pd.DataFrame:
    records = []

    for entry in entries:
        form = str(
            entry.get("form", "")
        ).strip()

        if form not in ALLOWED_FORMS:
            continue

        year = fiscal_year_from_entry(entry)

        if year is None:
            continue

        if year < START_YEAR or year > END_YEAR:
            continue

        value = safe_numeric(
            entry.get("val")
        )

        if pd.isna(value):
            continue

        filed = pd.to_datetime(
            entry.get("filed"),
            errors="coerce",
        )

        end_date = pd.to_datetime(
            entry.get("end"),
            errors="coerce",
        )

        start_date = pd.to_datetime(
            entry.get("start"),
            errors="coerce",
        )

        records.append(
            {
                "FISCAL_YEAR": year,
                "VALUE": value,
                "FORM": form,
                "FILED_DATE": filed,
                "START_DATE": start_date,
                "END_DATE": end_date,
                "FRAME": entry.get("frame", ""),
                "ACCESSION_NUMBER": entry.get("accn", ""),
                "FISCAL_PERIOD": entry.get("fp", ""),
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Yıllık akım değişkenlerinde FY kaydı tercih edilir.
    df["_FY_PRIORITY"] = (
        df["FISCAL_PERIOD"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("FY")
        .astype(int)
    )

    # Aynı mali yıl için:
    # 1. FY etiketi,
    # 2. en güncel filing,
    # 3. en güncel dönem sonu seçilir.
    df = (
        df.sort_values(
            [
                "FISCAL_YEAR",
                "_FY_PRIORITY",
                "FILED_DATE",
                "END_DATE",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=["FISCAL_YEAR"],
            keep="first",
        )
        .drop(columns="_FY_PRIORITY")
        .reset_index(drop=True)
    )

    return df


def extract_variable(
    us_gaap: dict,
    variable_name: str,
    candidate_tags: list[str],
) -> pd.DataFrame:
    for priority, tag in enumerate(
        candidate_tags,
        start=1,
    ):
        concept = us_gaap.get(tag)

        if not isinstance(concept, dict):
            continue

        entries, unit = choose_unit_entries(
            concept
        )

        if not entries:
            continue

        annual = filter_annual_entries(
            entries
        )

        if annual.empty:
            continue

        annual["VARIABLE"] = variable_name
        annual["XBRL_TAG"] = tag
        annual["TAG_PRIORITY"] = priority
        annual["UNIT"] = unit
        annual["CONCEPT_LABEL"] = concept.get(
            "label",
            "",
        )
        annual["CONCEPT_DESCRIPTION"] = concept.get(
            "description",
            "",
        )

        return annual

    return pd.DataFrame()


def extract_company(
    file_path: Path,
) -> tuple[pd.DataFrame, dict]:
    payload = load_json(file_path)

    cik10 = normalize_cik(
        payload.get("cik")
    )

    entity_name = str(
        payload.get("entityName", "")
    ).strip()

    us_gaap = find_us_gaap_facts(
        payload
    )

    company_frames = []

    for variable_name, tags in VARIABLE_TAGS.items():
        extracted = extract_variable(
            us_gaap=us_gaap,
            variable_name=variable_name,
            candidate_tags=tags,
        )

        if extracted.empty:
            continue

        extracted["CIK10"] = cik10
        extracted["ENTITY_NAME"] = entity_name
        extracted["SOURCE_FILE"] = str(file_path)

        company_frames.append(
            extracted
        )

    if company_frames:
        company_long = pd.concat(
            company_frames,
            ignore_index=True,
        )
    else:
        company_long = pd.DataFrame()

    metadata = {
        "CIK10": cik10,
        "ENTITY_NAME": entity_name,
        "SOURCE_FILE": str(file_path),
        "VARIABLES_FOUND": (
            company_long["VARIABLE"].nunique()
            if not company_long.empty
            else 0
        ),
        "OBSERVATIONS_EXTRACTED": len(
            company_long
        ),
    }

    return company_long, metadata


def build_wide_panel(
    long_data: pd.DataFrame,
) -> pd.DataFrame:
    if long_data.empty:
        return pd.DataFrame()

    wide = (
        long_data.pivot_table(
            index=[
                "CIK10",
                "ENTITY_NAME",
                "FISCAL_YEAR",
            ],
            columns="VARIABLE",
            values="VALUE",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None

    numeric_variables = list(
        VARIABLE_TAGS.keys()
    )

    for column in numeric_variables:
        if column not in wide.columns:
            wide[column] = np.nan

    wide = wide.sort_values(
        [
            "CIK10",
            "FISCAL_YEAR",
        ]
    ).reset_index(drop=True)

    return wide


def build_coverage(
    long_data: pd.DataFrame,
    queue: pd.DataFrame,
) -> pd.DataFrame:
    eligible_ciks = set(
        queue.loc[
            queue["DOWNLOAD_STATUS"]
            == "SUCCESS",
            "CIK10",
        ]
        .dropna()
        .apply(normalize_cik)
    )

    rows = []

    for variable_name in VARIABLE_TAGS:
        variable_data = long_data.loc[
            long_data["VARIABLE"]
            == variable_name
        ]

        firms_with_variable = (
            variable_data["CIK10"]
            .nunique()
        )

        observations = len(
            variable_data
        )

        coverage_rate = (
            firms_with_variable
            / len(eligible_ciks)
            if eligible_ciks
            else np.nan
        )

        rows.append(
            {
                "VARIABLE": variable_name,
                "SUCCESSFUL_COMPANY_FACTS_FILES": len(
                    eligible_ciks
                ),
                "FIRMS_WITH_VARIABLE": firms_with_variable,
                "ANNUAL_OBSERVATIONS": observations,
                "FIRM_COVERAGE_RATE": coverage_rate,
                "START_YEAR": (
                    variable_data["FISCAL_YEAR"].min()
                    if not variable_data.empty
                    else np.nan
                ),
                "END_YEAR": (
                    variable_data["FISCAL_YEAR"].max()
                    if not variable_data.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    try:
        if not RAW_DIR.exists():
            raise FileNotFoundError(
                f"Ham Company Facts klasörü bulunamadı:\n{RAW_DIR}"
            )

        if not QUEUE_FILE.exists():
            raise FileNotFoundError(
                f"Kuyruk dosyası bulunamadı:\n{QUEUE_FILE}"
            )

        OUTPUT_ERRORS.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("1/5 — İndirme kuyruğu okunuyor...")

        queue = pd.read_csv(
            QUEUE_FILE,
            dtype=str,
            low_memory=False,
        )

        queue["CIK10"] = (
            queue["CIK10"]
            .apply(normalize_cik)
        )

        success_queue = queue.loc[
            queue["DOWNLOAD_STATUS"]
            == "SUCCESS"
        ].copy()

        print(
            "Başarılı Company Facts kaydı:",
            f"{len(success_queue):,}",
        )

        raw_files = sorted(
            RAW_DIR.glob("CIK*.json")
        )

        print(
            "Diskteki JSON dosyası:",
            f"{len(raw_files):,}",
        )

        print("\n2/5 — Finansal kalemler çıkarılıyor...")

        all_frames = []
        metadata_rows = []
        error_rows = []

        for position, file_path in enumerate(
            raw_files,
            start=1,
        ):
            try:
                company_long, metadata = (
                    extract_company(file_path)
                )

                metadata_rows.append(
                    metadata
                )

                if not company_long.empty:
                    all_frames.append(
                        company_long
                    )

            except Exception as exc:
                error_rows.append(
                    {
                        "SOURCE_FILE": str(file_path),
                        "ERROR_TYPE": type(exc).__name__,
                        "ERROR_MESSAGE": str(exc),
                    }
                )

            print(
                f"\rİşlenen JSON: "
                f"{position:,}/{len(raw_files):,}",
                end="",
                flush=True,
            )

        print()

        if error_rows:
            pd.DataFrame(
                error_rows
            ).to_csv(
                OUTPUT_ERRORS,
                index=False,
                encoding="utf-8-sig",
            )

        if not all_frames:
            raise RuntimeError(
                "Hiçbir finansal gözlem çıkarılamadı."
            )

        long_data = pd.concat(
            all_frames,
            ignore_index=True,
        )

        long_data = long_data.sort_values(
            [
                "CIK10",
                "FISCAL_YEAR",
                "VARIABLE",
            ]
        ).reset_index(drop=True)

        print(
            "Çıkarılan uzun format gözlem:",
            f"{len(long_data):,}",
        )

        print("\n3/5 — Uzun format dosyası kaydediliyor...")

        long_data.to_csv(
            OUTPUT_LONG,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n4/5 — Yıllık geniş panel oluşturuluyor...")

        wide_data = build_wide_panel(
            long_data
        )

        wide_data.to_parquet(
            OUTPUT_WIDE,
            index=False,
        )

        coverage = build_coverage(
            long_data,
            queue,
        )

        coverage.to_csv(
            OUTPUT_COVERAGE,
            index=False,
            encoding="utf-8-sig",
        )

        tag_usage = (
            long_data.groupby(
                [
                    "VARIABLE",
                    "XBRL_TAG",
                    "UNIT",
                ],
                dropna=False,
            )
            .agg(
                FIRMS=("CIK10", "nunique"),
                OBSERVATIONS=("VALUE", "size"),
                MIN_YEAR=("FISCAL_YEAR", "min"),
                MAX_YEAR=("FISCAL_YEAR", "max"),
            )
            .reset_index()
            .sort_values(
                [
                    "VARIABLE",
                    "FIRMS",
                    "OBSERVATIONS",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
        )

        tag_usage.to_csv(
            OUTPUT_TAG_USAGE,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n5/5 — Veri bütünlüğü özeti...")

        print("\nFINANSAL VERİ ÇIKARIMI TAMAMLANDI")
        print("=" * 70)
        print(
            "Başarılı JSON dosyası:",
            f"{len(raw_files):,}",
        )
        print(
            "Firma sayısı:",
            f"{long_data['CIK10'].nunique():,}",
        )
        print(
            "Uzun format gözlem:",
            f"{len(long_data):,}",
        )
        print(
            "Geniş panel satırı:",
            f"{len(wide_data):,}",
        )
        print(
            "Dönem:",
            f"{int(wide_data['FISCAL_YEAR'].min())}"
            f"–"
            f"{int(wide_data['FISCAL_YEAR'].max())}",
        )
        print(
            "Okuma hatası:",
            f"{len(error_rows):,}",
        )

        print("\nDeğişken kapsaması:")
        print(
            coverage[
                [
                    "VARIABLE",
                    "FIRMS_WITH_VARIABLE",
                    "ANNUAL_OBSERVATIONS",
                    "FIRM_COVERAGE_RATE",
                ]
            ].to_string(
                index=False
            )
        )

        print("\nOluşturulan dosyalar:")
        print(OUTPUT_LONG)
        print(OUTPUT_WIDE)
        print(OUTPUT_COVERAGE)
        print(OUTPUT_TAG_USAGE)

        if error_rows:
            print(OUTPUT_ERRORS)

    except Exception as exc:
        print("\nİŞLEM BAŞARISIZ")
        print(
            f"Hata türü: {type(exc).__name__}"
        )
        print(
            f"Hata mesajı: {exc}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()