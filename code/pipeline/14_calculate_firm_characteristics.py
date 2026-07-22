from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# DOSYA YOLLARI
# ============================================================

PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

INPUT_FINANCIALS = (
    PROJECT_DIR
    / "output"
    / "sec_company_fundamentals_annual_wide.parquet"
)

VARIABLE_DICTIONARY = (
    PROJECT_DIR
    / "scripts"
    / "14_variable_dictionary.csv"
)

OUTPUT_PARQUET = (
    PROJECT_DIR
    / "output"
    / "firm_characteristics_annual.parquet"
)

OUTPUT_CSV = (
    PROJECT_DIR
    / "output"
    / "firm_characteristics_annual.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_DIR
    / "output"
    / "firm_characteristics_summary.csv"
)

OUTPUT_COVERAGE = (
    PROJECT_DIR
    / "output"
    / "firm_characteristics_coverage.csv"
)

OUTPUT_VALIDATION = (
    PROJECT_DIR
    / "output"
    / "firm_characteristics_validation.csv"
)


# ============================================================
# AYARLAR
# ============================================================

START_YEAR = 2010
END_YEAR = 2025

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

MIN_DENOMINATOR = 1e-9
MAX_GROWTH_GAP_YEARS = 1


CHARACTERISTIC_COLUMNS = [
    "LOG_ASSETS",
    "LEVERAGE",
    "DEBT_TO_EQUITY",
    "EQUITY_RATIO",
    "CASH_RATIO",
    "ROA",
    "OPERATING_PROFITABILITY",
    "OCF_TO_ASSETS",
    "CAPEX_INTENSITY",
    "RD_INTENSITY",
    "REVENUE_GROWTH",
    "ASSET_GROWTH",
    "SALES_TO_ASSETS",
    "CAPEX_TO_OCF",
    "EXTERNAL_FINANCE_DEPENDENCE",
    "FCF_TO_ASSETS",
]


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    require_positive_denominator: bool = True,
) -> pd.Series:
    numerator_numeric = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator_numeric = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    if require_positive_denominator:
        valid = (
            numerator_numeric.notna()
            & denominator_numeric.notna()
            & (denominator_numeric > MIN_DENOMINATOR)
        )
    else:
        valid = (
            numerator_numeric.notna()
            & denominator_numeric.notna()
            & (
                denominator_numeric.abs()
                > MIN_DENOMINATOR
            )
        )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype="float64",
    )

    result.loc[valid] = (
        numerator_numeric.loc[valid]
        / denominator_numeric.loc[valid]
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def winsorize_by_year(
    data: pd.DataFrame,
    column: str,
) -> tuple[pd.Series, pd.DataFrame]:
    output = pd.Series(
        np.nan,
        index=data.index,
        dtype="float64",
    )

    validation_rows = []

    for fiscal_year, group in data.groupby(
        "FISCAL_YEAR",
        dropna=False,
    ):
        values = pd.to_numeric(
            group[column],
            errors="coerce",
        )

        valid_values = values.dropna()

        if len(valid_values) < 20:
            output.loc[group.index] = values

            validation_rows.append(
                {
                    "VARIABLE": column,
                    "FISCAL_YEAR": fiscal_year,
                    "VALID_OBSERVATIONS": len(valid_values),
                    "LOWER_BOUND": np.nan,
                    "UPPER_BOUND": np.nan,
                    "BELOW_LOWER_COUNT": 0,
                    "ABOVE_UPPER_COUNT": 0,
                    "WINSORIZATION_APPLIED": False,
                }
            )

            continue

        lower_bound = valid_values.quantile(
            WINSOR_LOWER
        )

        upper_bound = valid_values.quantile(
            WINSOR_UPPER
        )

        below_count = int(
            (valid_values < lower_bound).sum()
        )

        above_count = int(
            (valid_values > upper_bound).sum()
        )

        output.loc[group.index] = values.clip(
            lower=lower_bound,
            upper=upper_bound,
        )

        validation_rows.append(
            {
                "VARIABLE": column,
                "FISCAL_YEAR": fiscal_year,
                "VALID_OBSERVATIONS": len(valid_values),
                "LOWER_BOUND": lower_bound,
                "UPPER_BOUND": upper_bound,
                "BELOW_LOWER_COUNT": below_count,
                "ABOVE_UPPER_COUNT": above_count,
                "WINSORIZATION_APPLIED": True,
            }
        )

    return output, pd.DataFrame(validation_rows)


def require_columns(
    data: pd.DataFrame,
    required_columns: set[str],
) -> None:
    missing = required_columns.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            f"Girdi panelinde eksik sütunlar: "
            f"{sorted(missing)}"
        )


# ============================================================
# VERİ OKUMA VE DOĞRULAMA
# ============================================================

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_FINANCIALS.exists():
        raise FileNotFoundError(
            f"Finansal panel bulunamadı:\n"
            f"{INPUT_FINANCIALS}"
        )

    if not VARIABLE_DICTIONARY.exists():
        raise FileNotFoundError(
            f"Değişken sözlüğü bulunamadı:\n"
            f"{VARIABLE_DICTIONARY}"
        )

    print(
        "1/6 — Finansal panel ve değişken sözlüğü okunuyor..."
    )

    financials = pd.read_parquet(
        INPUT_FINANCIALS
    )

    dictionary = pd.read_csv(
        VARIABLE_DICTIONARY,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_financial_columns = {
        "CIK10",
        "ENTITY_NAME",
        "FISCAL_YEAR",
        "TOTAL_ASSETS",
        "TOTAL_DEBT",
        "STOCKHOLDERS_EQUITY",
        "CASH_AND_EQUIVALENTS",
        "NET_INCOME",
        "OPERATING_INCOME",
        "OPERATING_CASH_FLOW",
        "CAPEX",
        "R_AND_D",
        "REVENUE",
    }

    require_columns(
        financials,
        required_financial_columns,
    )

    if "VARIABLE_NAME" not in dictionary.columns:
        raise ValueError(
            "Değişken sözlüğünde VARIABLE_NAME sütunu yok."
        )

    financials["CIK10"] = (
        financials["CIK10"]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.replace(
            r"\D",
            "",
            regex=True,
        )
        .str.zfill(10)
    )

    financials["FISCAL_YEAR"] = pd.to_numeric(
        financials["FISCAL_YEAR"],
        errors="coerce",
    ).astype("Int64")

    financials = financials.loc[
        financials["FISCAL_YEAR"].between(
            START_YEAR,
            END_YEAR,
        ).fillna(False)
    ].copy()

    numeric_columns = [
        "TOTAL_ASSETS",
        "TOTAL_DEBT",
        "STOCKHOLDERS_EQUITY",
        "CASH_AND_EQUIVALENTS",
        "NET_INCOME",
        "OPERATING_INCOME",
        "OPERATING_CASH_FLOW",
        "CAPEX",
        "R_AND_D",
        "REVENUE",
    ]

    for column in numeric_columns:
        financials[column] = pd.to_numeric(
            financials[column],
            errors="coerce",
        )

    duplicate_mask = financials.duplicated(
        subset=[
            "CIK10",
            "FISCAL_YEAR",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "CIK10–FISCAL_YEAR düzeyinde tekrar bulundu. "
            f"Tekrarlı satır sayısı: "
            f"{int(duplicate_mask.sum())}"
        )

    financials = financials.sort_values(
        [
            "CIK10",
            "FISCAL_YEAR",
        ]
    ).reset_index(drop=True)

    print(
        f"Firma-yıl satırı: {len(financials):,}"
    )
    print(
        f"Benzersiz firma: "
        f"{financials['CIK10'].nunique():,}"
    )
    print(
        f"Dönem: "
        f"{financials['FISCAL_YEAR'].min()}–"
        f"{financials['FISCAL_YEAR'].max()}"
    )
    print(
        f"Değişken sözlüğü satırı: "
        f"{len(dictionary):,}"
    )

    return financials, dictionary


# ============================================================
# FİRMA ÖZELLİKLERİ
# ============================================================

def calculate_characteristics(
    financials: pd.DataFrame,
) -> pd.DataFrame:
    print(
        "\n2/6 — Firma özellikleri hesaplanıyor..."
    )

    data = financials.copy()

    data["VALID_ASSETS_FLAG"] = (
        data["TOTAL_ASSETS"]
        .gt(MIN_DENOMINATOR)
        .fillna(False)
        .astype(int)
    )

    data["CAPEX_POSITIVE"] = (
        data["CAPEX"].abs()
    )

    data["RD_MISSING_FLAG"] = (
        data["R_AND_D"]
        .isna()
        .astype(int)
    )

    data["LOG_ASSETS"] = np.where(
        data["TOTAL_ASSETS"] > MIN_DENOMINATOR,
        np.log(data["TOTAL_ASSETS"]),
        np.nan,
    )

    data["LEVERAGE"] = safe_divide(
        data["TOTAL_DEBT"],
        data["TOTAL_ASSETS"],
    )

    data["DEBT_TO_EQUITY"] = safe_divide(
        data["TOTAL_DEBT"],
        data["STOCKHOLDERS_EQUITY"],
    )

    data["EQUITY_RATIO"] = safe_divide(
        data["STOCKHOLDERS_EQUITY"],
        data["TOTAL_ASSETS"],
    )

    data["CASH_RATIO"] = safe_divide(
        data["CASH_AND_EQUIVALENTS"],
        data["TOTAL_ASSETS"],
    )

    data["ROA"] = safe_divide(
        data["NET_INCOME"],
        data["TOTAL_ASSETS"],
    )

    data["OPERATING_PROFITABILITY"] = safe_divide(
        data["OPERATING_INCOME"],
        data["TOTAL_ASSETS"],
    )

    data["OCF_TO_ASSETS"] = safe_divide(
        data["OPERATING_CASH_FLOW"],
        data["TOTAL_ASSETS"],
    )

    data["CAPEX_INTENSITY"] = safe_divide(
        data["CAPEX_POSITIVE"],
        data["TOTAL_ASSETS"],
    )

    data["RD_INTENSITY"] = safe_divide(
        data["R_AND_D"],
        data["TOTAL_ASSETS"],
    )

    data["SALES_TO_ASSETS"] = safe_divide(
        data["REVENUE"],
        data["TOTAL_ASSETS"],
    )

    data["CAPEX_TO_OCF"] = safe_divide(
        data["CAPEX_POSITIVE"],
        data["OPERATING_CASH_FLOW"],
    )

    data["EXTERNAL_FINANCE_DEPENDENCE"] = safe_divide(
        (
            data["CAPEX_POSITIVE"]
            - data["OPERATING_CASH_FLOW"]
        ),
        data["TOTAL_ASSETS"],
    )

    data["FCF_TO_ASSETS"] = safe_divide(
        (
            data["OPERATING_CASH_FLOW"]
            - data["CAPEX_POSITIVE"]
        ),
        data["TOTAL_ASSETS"],
    )

    data["PREVIOUS_FISCAL_YEAR"] = (
        data.groupby("CIK10")[
            "FISCAL_YEAR"
        ]
        .shift(1)
    )

    year_difference = (
        data["FISCAL_YEAR"]
        - data["PREVIOUS_FISCAL_YEAR"]
    )

    data["CONSECUTIVE_YEAR_FLAG"] = (
        year_difference
        .eq(MAX_GROWTH_GAP_YEARS)
        .fillna(False)
        .astype(int)
    )

    previous_revenue = (
        data.groupby("CIK10")[
            "REVENUE"
        ]
        .shift(1)
    )

    previous_assets = (
        data.groupby("CIK10")[
            "TOTAL_ASSETS"
        ]
        .shift(1)
    )

    revenue_growth = (
        safe_divide(
            data["REVENUE"],
            previous_revenue,
        )
        - 1
    )

    asset_growth = (
        safe_divide(
            data["TOTAL_ASSETS"],
            previous_assets,
        )
        - 1
    )

    data["REVENUE_GROWTH"] = revenue_growth.where(
        data["CONSECUTIVE_YEAR_FLAG"].eq(1),
        np.nan,
    )

    data["ASSET_GROWTH"] = asset_growth.where(
        data["CONSECUTIVE_YEAR_FLAG"].eq(1),
        np.nan,
    )

    data[CHARACTERISTIC_COLUMNS] = (
        data[CHARACTERISTIC_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return data


# ============================================================
# WINSORIZATION
# ============================================================

def create_winsorized_variables(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(
        "\n3/6 — Yıllık winsorization uygulanıyor..."
    )

    output = data.copy()
    validation_frames = []

    for column in CHARACTERISTIC_COLUMNS:
        winsorized, validation = (
            winsorize_by_year(
                output,
                column,
            )
        )

        winsorized_column = (
            f"{column}_W"
        )

        output[winsorized_column] = (
            winsorized
        )

        validation[
            "WINSORIZED_VARIABLE"
        ] = winsorized_column

        validation_frames.append(
            validation
        )

    validation_data = pd.concat(
        validation_frames,
        ignore_index=True,
    )

    return output, validation_data


# ============================================================
# GECİKMELER
# ============================================================

def create_lagged_variables(
    data: pd.DataFrame,
) -> pd.DataFrame:
    print(
        "\n4/6 — Firma içi t−1 gecikmeler oluşturuluyor..."
    )

    output = data.sort_values(
        [
            "CIK10",
            "FISCAL_YEAR",
        ]
    ).copy()

    previous_year = (
        output.groupby("CIK10")[
            "FISCAL_YEAR"
        ]
        .shift(1)
    )

    nonconsecutive = (
        (
            output["FISCAL_YEAR"]
            - previous_year
        )
        .ne(1)
        .fillna(True)
    )

    for column in CHARACTERISTIC_COLUMNS:
        winsorized_column = (
            f"{column}_W"
        )

        lag_column = (
            f"{winsorized_column}_L1"
        )

        output[lag_column] = (
            output.groupby("CIK10")[
                winsorized_column
            ]
            .shift(1)
        )

        output.loc[
            nonconsecutive,
            lag_column,
        ] = np.nan

    return output


# ============================================================
# KALİTE RAPORLARI
# ============================================================

def build_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    print(
        "\n5/6 — Kapsama ve dağılım raporları hazırlanıyor..."
    )

    rows = []

    for column in CHARACTERISTIC_COLUMNS:
        versions = [
            column,
            f"{column}_W",
            f"{column}_W_L1",
        ]

        for version in versions:
            values = pd.to_numeric(
                data[version],
                errors="coerce",
            )

            valid = values.dropna()

            rows.append(
                {
                    "VARIABLE": version,
                    "TOTAL_FIRM_YEAR_ROWS": len(data),
                    "VALID_OBSERVATIONS": len(valid),
                    "COVERAGE_RATE": (
                        len(valid) / len(data)
                        if len(data) > 0
                        else np.nan
                    ),
                    "UNIQUE_FIRMS": (
                        data.loc[
                            values.notna(),
                            "CIK10",
                        ]
                        .nunique()
                    ),
                    "MEAN": valid.mean(),
                    "STD": valid.std(),
                    "MIN": valid.min(),
                    "P01": valid.quantile(0.01),
                    "P05": valid.quantile(0.05),
                    "MEDIAN": valid.median(),
                    "P95": valid.quantile(0.95),
                    "P99": valid.quantile(0.99),
                    "MAX": valid.max(),
                }
            )

    return pd.DataFrame(rows)


def build_coverage_by_year(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for fiscal_year, year_data in data.groupby(
        "FISCAL_YEAR"
    ):
        for column in CHARACTERISTIC_COLUMNS:
            version = f"{column}_W"

            valid = (
                year_data[version]
                .notna()
            )

            rows.append(
                {
                    "FISCAL_YEAR": fiscal_year,
                    "VARIABLE": version,
                    "TOTAL_ROWS": len(year_data),
                    "VALID_OBSERVATIONS": int(
                        valid.sum()
                    ),
                    "COVERAGE_RATE": (
                        float(valid.mean())
                        if len(year_data) > 0
                        else np.nan
                    ),
                    "UNIQUE_FIRMS": (
                        year_data.loc[
                            valid,
                            "CIK10",
                        ]
                        .nunique()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# ÇIKTILAR
# ============================================================

def save_outputs(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    print(
        "\n6/6 — Çıktılar kaydediliyor..."
    )

    data = data.sort_values(
        [
            "CIK10",
            "FISCAL_YEAR",
        ]
    ).reset_index(drop=True)

    data.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    data.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    coverage.to_csv(
        OUTPUT_COVERAGE,
        index=False,
        encoding="utf-8-sig",
    )

    validation.to_csv(
        OUTPUT_VALIDATION,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nFİRMA ÖZELLİKLERİ HAZIR"
    )
    print("=" * 70)

    print(
        f"Firma-yıl satırı: "
        f"{len(data):,}"
    )

    print(
        f"Benzersiz firma: "
        f"{data['CIK10'].nunique():,}"
    )

    print(
        f"Dönem: "
        f"{data['FISCAL_YEAR'].min()}–"
        f"{data['FISCAL_YEAR'].max()}"
    )

    print(
        "\nAna winsorize değişken kapsaması:"
    )

    display_variables = [
        "LEVERAGE_W",
        "CASH_RATIO_W",
        "ROA_W",
        "OCF_TO_ASSETS_W",
        "CAPEX_INTENSITY_W",
        "RD_INTENSITY_W",
        "REVENUE_GROWTH_W",
        "ASSET_GROWTH_W",
        "EXTERNAL_FINANCE_DEPENDENCE_W",
        "LOG_ASSETS_W",
    ]

    display_summary = summary.loc[
        summary["VARIABLE"].isin(
            display_variables
        ),
        [
            "VARIABLE",
            "VALID_OBSERVATIONS",
            "UNIQUE_FIRMS",
            "COVERAGE_RATE",
            "MEAN",
            "STD",
        ],
    ]

    print(
        display_summary.to_string(
            index=False
        )
    )

    print(
        "\nOluşturulan dosyalar:"
    )
    print(OUTPUT_PARQUET)
    print(OUTPUT_CSV)
    print(OUTPUT_SUMMARY)
    print(OUTPUT_COVERAGE)
    print(OUTPUT_VALIDATION)


# ============================================================
# ANA ÇALIŞMA
# ============================================================

def main() -> None:
    try:
        financials, dictionary = (
            load_inputs()
        )

        dictionary_variables = set(
            dictionary[
                "VARIABLE_NAME"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        missing_dictionary_entries = (
            set(
                CHARACTERISTIC_COLUMNS
            )
            .difference(
                dictionary_variables
            )
        )

        if missing_dictionary_entries:
            print(
                "Uyarı — sözlükte bulunmayan "
                "hesaplanmış değişkenler: "
                f"{sorted(missing_dictionary_entries)}"
            )

        characteristics = (
            calculate_characteristics(
                financials
            )
        )

        (
            characteristics,
            validation,
        ) = create_winsorized_variables(
            characteristics
        )

        characteristics = (
            create_lagged_variables(
                characteristics
            )
        )

        summary = build_summary(
            characteristics
        )

        coverage = (
            build_coverage_by_year(
                characteristics
            )
        )

        save_outputs(
            data=characteristics,
            summary=summary,
            coverage=coverage,
            validation=validation,
        )

    except Exception as exc:
        print(
            "\nİŞLEM BAŞARISIZ"
        )
        print(
            f"Hata türü: "
            f"{type(exc).__name__}"
        )
        print(
            f"Hata mesajı: {exc}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()