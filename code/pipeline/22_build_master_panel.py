from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 22_build_master_panel.py
#
# AMAÇ
# ----
# Aşağıdaki veri katmanlarını birleştirerek ekonometrik ana paneli
# oluşturmak:
#
# 1. Aylık ETF getirileri
# 2. Climate Policy Uncertainty
# 3. Piyasa kontrol değişkenleri
# 4. Holdings-based portfolio architecture faktörleri
#
# Örneklemler:
# - Full return pool
# - Moderate-quality architecture sample
# - Main-quality architecture sample
#
# ÖNEMLİ METODOLOJİK NOT
# ---------------------
# Portfolio architecture değişkenleri 2025Q4 holdings kesitinden
# üretilmiştir. Bu nedenle mevcut panel frozen-architecture
# cross-sectional exposure tasarımıdır.
#
# Gelecekte tarihsel N-PORT dönemleri eklendiğinde architecture
# değişkenleri zaman içinde değişen ETF-month özelliklerine
# dönüştürülecektir.
# ============================================================


# ============================================================
# 1. PROJE YOLLARI
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\User\Desktop\CPU_Project"
)

OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. GİRDİ DOSYALARI
# ============================================================

ETF_RETURN_FILE = (
    OUTPUT_DIR
    / "20_monthly_etf_returns.csv"
)

MARKET_CONTROL_FILE = (
    OUTPUT_DIR
    / "20_monthly_market_controls.csv"
)

CPU_FILE = (
    OUTPUT_DIR
    / "21_monthly_cpu.csv"
)

ARCHITECTURE_FILE = (
    OUTPUT_DIR
    / "18_quality_adjusted_architecture_factors.csv"
)

TICKER_MAPPING_FILE = (
    OUTPUT_DIR
    / "19a_etf_ticker_mapping.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

MASTER_PANEL_FILE = (
    OUTPUT_DIR
    / "22_master_cpu_return_architecture_panel.csv"
)

MASTER_PANEL_PARQUET_FILE = (
    OUTPUT_DIR
    / "22_master_cpu_return_architecture_panel.parquet"
)

MAIN_SAMPLE_FILE = (
    OUTPUT_DIR
    / "22_main_quality_panel.csv"
)

MAIN_SAMPLE_PARQUET_FILE = (
    OUTPUT_DIR
    / "22_main_quality_panel.parquet"
)

MODERATE_SAMPLE_FILE = (
    OUTPUT_DIR
    / "22_moderate_quality_panel.csv"
)

FULL_RETURN_POOL_FILE = (
    OUTPUT_DIR
    / "22_full_return_pool.csv"
)

ETF_COVERAGE_FILE = (
    OUTPUT_DIR
    / "22_master_panel_etf_coverage.csv"
)

MONTHLY_COVERAGE_FILE = (
    OUTPUT_DIR
    / "22_master_panel_monthly_coverage.csv"
)

SAMPLE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "22_master_panel_sample_summary.csv"
)

MERGE_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "22_master_panel_merge_diagnostics.csv"
)

DESCRIPTIVE_FILE = (
    OUTPUT_DIR
    / "22_master_panel_descriptive_statistics.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "22_master_panel_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "22_master_panel_variable_dictionary.csv"
)


# ============================================================
# 4. AMPİRİK AYARLAR
# ============================================================

START_DATE = pd.Timestamp(
    "2010-01-31"
)

END_DATE = pd.Timestamp(
    "2025-12-31"
)

ARCHITECTURE_SNAPSHOT = "2025Q4"

MIN_MONTHS_MAIN_ANALYSIS = 24

MIN_MONTHS_LONG_HISTORY = 60

RETURN_OUTLIER_LIMIT = 5.0


# ============================================================
# 5. ARCHITECTURE DEĞİŞKENLERİ
# ============================================================

RAW_ARCHITECTURE_FACTORS = [
    "FINANCIAL_RESILIENCE",
    "FINANCING_VULNERABILITY",
    "GROWTH_DURATION_EXPOSURE",
    "PORTFOLIO_CONCENTRATION",
    "CORE_TRANSITION_SENSITIVITY",
    "EXTENDED_TRANSITION_SENSITIVITY",
]

MAIN_ARCHITECTURE_FACTORS = [
    "FINANCIAL_RESILIENCE_MAIN",
    "FINANCING_VULNERABILITY_MAIN",
    "GROWTH_DURATION_EXPOSURE_MAIN",
    "PORTFOLIO_CONCENTRATION_MAIN",
    "CORE_TRANSITION_SENSITIVITY_MAIN",
    "EXTENDED_TRANSITION_SENSITIVITY_MAIN",
]

MODERATE_ARCHITECTURE_FACTORS = [
    "FINANCIAL_RESILIENCE_MODERATE",
    "FINANCING_VULNERABILITY_MODERATE",
    "GROWTH_DURATION_EXPOSURE_MODERATE",
    "PORTFOLIO_CONCENTRATION_MODERATE",
    "CORE_TRANSITION_SENSITIVITY_MODERATE",
    "EXTENDED_TRANSITION_SENSITIVITY_MODERATE",
]

PORTFOLIO_STRUCTURE_VARIABLES = [
    "HHI",
    "TOP5_WEIGHT",
    "TOP10_WEIGHT",
    "MAX_HOLDING_WEIGHT",
    "EFFECTIVE_NUMBER_OF_HOLDINGS",
    "FINANCIAL_MATCH_WEIGHT",
]

PORTFOLIO_CHARACTERISTICS = [
    "PW_ROA",
    "PW_LEVERAGE",
    "PW_CASH_RATIO",
    "PW_CAPEX_INTENSITY",
    "PW_RD_INTENSITY",
    "PW_REVENUE_GROWTH",
    "PW_EXTERNAL_FINANCE_DEPENDENCE",
    "PW_LOG_ASSETS",
]


# ============================================================
# 6. CPU DEĞİŞKENLERİ
# ============================================================

CPU_VARIABLES = [
    "CPU",
    "LOG_CPU",
    "CPU_Z",
    "LOG_CPU_Z",
    "CPU_DIFF",
    "CPU_CHANGE",
    "CPU_L1",
    "CPU_Z_L1",
    "CPU_DIFF_L1",
    "CPU_L2",
    "CPU_Z_L2",
    "CPU_BROAD",
    "CPU_LLM",
    "CPNEWS_NARROW",
    "CPNEWS_BROAD",
    "CPSENT",
    "CPU_INSTRUMENT",
    "CPU_SHOCK",
    "LOW_CPU_REGIME",
    "HIGH_CPU_REGIME",
    "EXTREME_CPU_REGIME",
    "CPU_ABOVE_MEDIAN",
    "CPU_REGIME",
]


# ============================================================
# 7. PİYASA KONTROLLERİ
# ============================================================

MARKET_CONTROL_VARIABLES = [
    "MARKET_RETURN",
    "ENERGY_RETURN",
    "TREASURY_RETURN",
    "VIX_LEVEL",
    "VIX_CHANGE",
    "SP500_RETURN",
]


# ============================================================
# 8. YARDIMCI FONKSİYONLAR
# ============================================================

def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sütun adlarını uppercase snake-case biçimine getirir.
    """

    result = df.copy()

    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            r"[^A-Z0-9]+",
            "_",
            regex=True,
        )
        .str.strip("_")
    )

    return result


def read_data(
    path: Path,
) -> pd.DataFrame:
    """
    CSV veya Parquet dosyasını okur.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı:\n{path}"
        )

    if path.suffix.lower() == ".csv":

        return pd.read_csv(
            path,
            low_memory=False,
        )

    if path.suffix.lower() == ".parquet":

        return pd.read_parquet(
            path
        )

    raise ValueError(
        f"Desteklenmeyen dosya türü: {path.suffix}"
    )


def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Sayısal olmayan değerleri NaN yapar.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def clean_text(
    series: pd.Series,
) -> pd.Series:
    """
    Metin değerlerini temizler.
    """

    result = (
        series.astype("string")
        .str.strip()
    )

    invalid_values = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "<NA>",
    }

    return result.mask(
        result.str.upper().isin(
            invalid_values
        ),
        pd.NA,
    )


def clean_ticker(
    series: pd.Series,
) -> pd.Series:
    """
    ETF ticker sembollerini standartlaştırır.
    """

    return (
        clean_text(series)
        .str.upper()
        .str.replace(
            ".",
            "-",
            regex=False,
        )
    )


def standardize_month_end(
    series: pd.Series,
) -> pd.Series:
    """
    Tarihleri ay sonuna dönüştürür.
    """

    dates = pd.to_datetime(
        series,
        errors="coerce",
    )

    return (
        dates
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """
    Zorunlu sütunların mevcut olduğunu kontrol eder.
    """

    missing_columns = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise KeyError(
            f"{dataset_name} dosyasında eksik sütunlar:\n"
            + "\n".join(
                missing_columns
            )
        )


def zscore(
    series: pd.Series,
) -> pd.Series:
    """
    Z-score standardizasyonu.
    """

    numeric = safe_numeric(
        series
    )

    mean_value = numeric.mean()

    std_value = numeric.std(
        ddof=1
    )

    if (
        pd.isna(std_value)
        or std_value <= 0
    ):

        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    return (
        numeric
        - mean_value
    ) / std_value


# ============================================================
# 9. ETF GETİRİ PANELİ
# ============================================================

def prepare_etf_returns(
    path: Path,
) -> pd.DataFrame:
    """
    Aylık ETF getirilerini hazırlar.
    """

    returns = normalize_columns(
        read_data(
            path
        )
    )

    require_columns(
        returns,
        [
            "DATE",
            "ETF_ID",
            "ETF_NAME",
            "ETF_RETURN",
        ],
        "ETF RETURNS",
    )

    returns = returns.copy()

    returns["DATE"] = standardize_month_end(
        returns["DATE"]
    )

    returns["ETF_ID"] = clean_text(
        returns["ETF_ID"]
    )

    returns["ETF_NAME"] = clean_text(
        returns["ETF_NAME"]
    )

    if "ETF_TICKER" in returns.columns:

        returns["ETF_TICKER"] = clean_ticker(
            returns["ETF_TICKER"]
        )

    else:

        returns["ETF_TICKER"] = pd.NA

    returns["ETF_RETURN"] = safe_numeric(
        returns["ETF_RETURN"]
    )

    if "MONTH_END_ADJ_CLOSE" in returns.columns:

        returns[
            "MONTH_END_ADJ_CLOSE"
        ] = safe_numeric(
            returns[
                "MONTH_END_ADJ_CLOSE"
            ]
        )

    returns = returns.loc[
        returns["DATE"].between(
            START_DATE,
            END_DATE,
        )
    ].copy()

    returns = returns.loc[
        returns["DATE"].notna()
        & returns["ETF_ID"].notna()
    ].copy()

    extreme_return_flag = (
        returns["ETF_RETURN"]
        .abs()
        > RETURN_OUTLIER_LIMIT
    )

    returns[
        "EXTREME_RETURN_FLAG"
    ] = extreme_return_flag.astype(
        int
    )

    returns.loc[
        extreme_return_flag,
        "ETF_RETURN",
    ] = np.nan

    duplicate_count = returns.duplicated(
        subset=[
            "ETF_ID",
            "DATE",
        ],
        keep=False,
    ).sum()

    if duplicate_count > 0:

        raise RuntimeError(
            "ETF return panelinde duplicate ETF_ID–DATE "
            f"satırı bulundu: {duplicate_count}"
        )

    return returns


# ============================================================
# 10. CPU PANELİ
# ============================================================

def prepare_cpu(
    path: Path,
) -> pd.DataFrame:
    """
    Aylık CPU panelini hazırlar.
    """

    cpu = normalize_columns(
        read_data(
            path
        )
    )

    require_columns(
        cpu,
        [
            "DATE",
            "CPU",
            "CPU_Z",
            "HIGH_CPU_REGIME",
        ],
        "CPU",
    )

    cpu = cpu.copy()

    cpu["DATE"] = standardize_month_end(
        cpu["DATE"]
    )

    available_columns = [
        "DATE",
    ]

    for variable in CPU_VARIABLES:

        if variable in cpu.columns:

            available_columns.append(
                variable
            )

    cpu = cpu[
        list(
            dict.fromkeys(
                available_columns
            )
        )
    ].copy()

    numeric_cpu_columns = [
        column
        for column in cpu.columns
        if column not in [
            "DATE",
            "CPU_REGIME",
        ]
    ]

    for column in numeric_cpu_columns:

        cpu[column] = safe_numeric(
            cpu[column]
        )

    cpu = cpu.loc[
        cpu["DATE"].between(
            START_DATE,
            END_DATE,
        )
    ].copy()

    duplicate_count = cpu.duplicated(
        subset=["DATE"],
        keep=False,
    ).sum()

    if duplicate_count > 0:

        raise RuntimeError(
            "CPU panelinde duplicate ay bulundu."
        )

    return cpu


# ============================================================
# 11. PİYASA KONTROL PANELİ
# ============================================================

def prepare_market_controls(
    path: Path,
) -> pd.DataFrame:
    """
    Aylık piyasa kontrollerini hazırlar.
    """

    controls = normalize_columns(
        read_data(
            path
        )
    )

    require_columns(
        controls,
        [
            "DATE",
            "MARKET_RETURN",
            "ENERGY_RETURN",
            "TREASURY_RETURN",
        ],
        "MARKET CONTROLS",
    )

    controls = controls.copy()

    controls["DATE"] = standardize_month_end(
        controls["DATE"]
    )

    available_columns = [
        "DATE",
    ]

    for variable in MARKET_CONTROL_VARIABLES:

        if variable in controls.columns:

            available_columns.append(
                variable
            )

    controls = controls[
        available_columns
    ].copy()

    for column in available_columns:

        if column != "DATE":

            controls[column] = safe_numeric(
                controls[column]
            )

    controls = controls.loc[
        controls["DATE"].between(
            START_DATE,
            END_DATE,
        )
    ].copy()

    duplicate_count = controls.duplicated(
        subset=["DATE"],
        keep=False,
    ).sum()

    if duplicate_count > 0:

        raise RuntimeError(
            "Piyasa kontrol panelinde duplicate ay bulundu."
        )

    return controls


# ============================================================
# 12. ARCHITECTURE PANELİ
# ============================================================

def prepare_architecture(
    path: Path,
) -> pd.DataFrame:
    """
    Quality-adjusted architecture panelini hazırlar.
    """

    architecture = normalize_columns(
        read_data(
            path
        )
    )

    require_columns(
        architecture,
        [
            "ETF_ID",
            "ETF_NAME",
            "FINANCIAL_MATCH_WEIGHT",
            "CORE_TRANSITION_SENSITIVITY_MAIN",
            "CORE_TRANSITION_SENSITIVITY_MODERATE",
        ],
        "ARCHITECTURE",
    )

    architecture = architecture.copy()

    architecture["ETF_ID"] = clean_text(
        architecture["ETF_ID"]
    )

    architecture["ETF_NAME"] = clean_text(
        architecture["ETF_NAME"]
    )

    if "ETF_TICKER" in architecture.columns:

        architecture["ETF_TICKER"] = clean_ticker(
            architecture["ETF_TICKER"]
        )

    available_columns = [
        "ETF_ID",
        "ETF_NAME",
    ]

    candidate_columns = (
        RAW_ARCHITECTURE_FACTORS
        + MAIN_ARCHITECTURE_FACTORS
        + MODERATE_ARCHITECTURE_FACTORS
        + PORTFOLIO_STRUCTURE_VARIABLES
        + PORTFOLIO_CHARACTERISTICS
    )

    candidate_columns.extend(
        [
            column
            for column in architecture.columns
            if (
                "_VALID_" in column
                or column.startswith(
                    "COV_"
                )
                or column.startswith(
                    "DISP_"
                )
            )
        ]
    )

    for column in candidate_columns:

        if column in architecture.columns:

            available_columns.append(
                column
            )

    architecture = architecture[
        list(
            dict.fromkeys(
                available_columns
            )
        )
    ].copy()

    architecture[
        "MAIN_ARCHITECTURE_SAMPLE"
    ] = (
        architecture[
            "CORE_TRANSITION_SENSITIVITY_MAIN"
        ].notna()
    ).astype(
        int
    )

    architecture[
        "MODERATE_ARCHITECTURE_SAMPLE"
    ] = (
        architecture[
            "CORE_TRANSITION_SENSITIVITY_MODERATE"
        ].notna()
    ).astype(
        int
    )

    architecture[
        "EXTENDED_MAIN_SAMPLE"
    ] = (
        architecture[
            "EXTENDED_TRANSITION_SENSITIVITY_MAIN"
        ].notna()
    ).astype(
        int
    )

    architecture[
        "ARCHITECTURE_SNAPSHOT"
    ] = ARCHITECTURE_SNAPSHOT

    architecture[
        "FROZEN_ARCHITECTURE_FLAG"
    ] = 1

    duplicate_count = architecture.duplicated(
        subset=["ETF_ID"],
        keep=False,
    ).sum()

    if duplicate_count > 0:

        raise RuntimeError(
            "Architecture panelinde duplicate ETF_ID bulundu."
        )

    return architecture


# ============================================================
# 13. TICKER MAPPING
# ============================================================

def prepare_ticker_mapping(
    path: Path,
) -> pd.DataFrame:
    """
    Resmî SEC ticker mapping tablosunu hazırlar.
    """

    mapping = normalize_columns(
        read_data(
            path
        )
    )

    require_columns(
        mapping,
        [
            "ETF_ID",
            "ETF_TICKER",
            "TICKER_MATCHED",
        ],
        "TICKER MAPPING",
    )

    mapping = mapping.copy()

    mapping["ETF_ID"] = clean_text(
        mapping["ETF_ID"]
    )

    mapping["ETF_TICKER"] = clean_ticker(
        mapping["ETF_TICKER"]
    )

    mapping = mapping.loc[
        safe_numeric(
            mapping["TICKER_MATCHED"]
        )
        == 1
    ].copy()

    mapping = (
        mapping[
            [
                "ETF_ID",
                "ETF_TICKER",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["ETF_ID"]
        )
    )

    return mapping


# ============================================================
# 14. ETKİLEŞİM DEĞİŞKENLERİ
# ============================================================

def add_interactions(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    CPU × portfolio architecture etkileşimlerini oluşturur.
    """

    result = panel.copy()

    architecture_variables = []

    for variable in (
        RAW_ARCHITECTURE_FACTORS
        + MAIN_ARCHITECTURE_FACTORS
        + MODERATE_ARCHITECTURE_FACTORS
        + PORTFOLIO_STRUCTURE_VARIABLES
        + PORTFOLIO_CHARACTERISTICS
    ):

        if variable in result.columns:

            architecture_variables.append(
                variable
            )

    for variable in architecture_variables:

        result[
            f"CPU_Z_X_{variable}"
        ] = (
            result["CPU_Z"]
            * result[variable]
        )

        result[
            f"LOG_CPU_Z_X_{variable}"
        ] = (
            result["LOG_CPU_Z"]
            * result[variable]
        )

        result[
            f"CPU_DIFF_X_{variable}"
        ] = (
            result["CPU_DIFF"]
            * result[variable]
        )

        result[
            f"CPU_Z_L1_X_{variable}"
        ] = (
            result["CPU_Z_L1"]
            * result[variable]
        )

        result[
            f"HIGH_CPU_X_{variable}"
        ] = (
            result["HIGH_CPU_REGIME"]
            * result[variable]
        )

        result[
            f"EXTREME_CPU_X_{variable}"
        ] = (
            result["EXTREME_CPU_REGIME"]
            * result[variable]
        )

        if "VIX_LEVEL_Z" in result.columns:

            result[
                f"VIX_X_{variable}"
            ] = (
                result["VIX_LEVEL_Z"]
                * result[variable]
            )

    return result


# ============================================================
# 15. ZAMAN SERİSİ VE STRES DEĞİŞKENLERİ
# ============================================================

def add_panel_variables(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ek zaman, stres ve dönüşüm değişkenlerini oluşturur.
    """

    result = panel.copy()

    result = result.sort_values(
        [
            "ETF_ID",
            "DATE",
        ]
    ).reset_index(
        drop=True
    )

    result["YEAR"] = (
        result["DATE"].dt.year
    )

    result["MONTH"] = (
        result["DATE"].dt.month
    )

    result["QUARTER"] = (
        result["DATE"].dt.quarter
    )

    result["YEAR_MONTH"] = (
        result["DATE"]
        .dt.to_period("M")
        .astype(str)
    )

    result["TIME_INDEX"] = (
        result["DATE"]
        .rank(
            method="dense"
        )
        .astype("int64")
    )

    result["ETF_RETURN_L1"] = (
        result.groupby(
            "ETF_ID"
        )["ETF_RETURN"]
        .shift(1)
    )

    result["ETF_RETURN_L2"] = (
        result.groupby(
            "ETF_ID"
        )["ETF_RETURN"]
        .shift(2)
    )

    result["ETF_RETURN_LEAD1"] = (
        result.groupby(
            "ETF_ID"
        )["ETF_RETURN"]
        .shift(-1)
    )

    result["ETF_RETURN_LEAD3"] = (
        result.groupby(
            "ETF_ID"
        )["ETF_RETURN"]
        .shift(-3)
    )

    result["CUM_RETURN_LEAD3"] = (
        (
            1.0
            + result["ETF_RETURN_LEAD1"]
        )
        * (
            1.0
            + result.groupby(
                "ETF_ID"
            )["ETF_RETURN"]
            .shift(-2)
        )
        * (
            1.0
            + result["ETF_RETURN_LEAD3"]
        )
        - 1.0
    )

    result["NEGATIVE_RETURN"] = (
        result["ETF_RETURN"] < 0
    ).astype(
        "Int64"
    )

    result["TAIL_RETURN_10"] = (
        result.groupby(
            "ETF_ID"
        )["ETF_RETURN"]
        .transform(
            lambda series: (
                series
                <= series.quantile(
                    0.10
                )
            ).astype(
                int
            )
        )
    )

    result["POST_2020"] = (
        result["DATE"]
        >= pd.Timestamp(
            "2020-01-31"
        )
    ).astype(
        int
    )

    result["COVID_PERIOD"] = (
        result["DATE"].between(
            pd.Timestamp(
                "2020-02-29"
            ),
            pd.Timestamp(
                "2021-12-31"
            ),
        )
    ).astype(
        int
    )

    if "VIX_LEVEL" in result.columns:

        monthly_vix = (
            result[
                [
                    "DATE",
                    "VIX_LEVEL",
                ]
            ]
            .drop_duplicates(
                subset=["DATE"]
            )
            .sort_values(
                "DATE"
            )
        )

        monthly_vix[
            "VIX_LEVEL_Z"
        ] = zscore(
            monthly_vix[
                "VIX_LEVEL"
            ]
        )

        vix_p75 = float(
            monthly_vix[
                "VIX_LEVEL"
            ]
            .quantile(
                0.75
            )
        )

        monthly_vix[
            "HIGH_VIX_REGIME"
        ] = (
            monthly_vix[
                "VIX_LEVEL"
            ]
            >= vix_p75
        ).astype(
            int
        )

        result = result.drop(
            columns=[
                "VIX_LEVEL_Z",
                "HIGH_VIX_REGIME",
            ],
            errors="ignore",
        )

        result = result.merge(
            monthly_vix[
                [
                    "DATE",
                    "VIX_LEVEL_Z",
                    "HIGH_VIX_REGIME",
                ]
            ],
            on="DATE",
            how="left",
            validate="many_to_one",
        )

    else:

        result["VIX_LEVEL_Z"] = np.nan
        result["HIGH_VIX_REGIME"] = 0

    result[
        "CPU_AND_MARKET_STRESS"
    ] = (
        (
            result[
                "HIGH_CPU_REGIME"
            ] == 1
        )
        & (
            result[
                "HIGH_VIX_REGIME"
            ] == 1
        )
    ).astype(
        int
    )

    result[
        "VALID_BASELINE_ROW"
    ] = (
        result[
            [
                "ETF_RETURN",
                "CPU_Z",
                "MARKET_RETURN",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    ).astype(
        int
    )

    result[
        "VALID_FULL_CONTROL_ROW"
    ] = (
        result[
            [
                "ETF_RETURN",
                "CPU_Z",
                "MARKET_RETURN",
                "ENERGY_RETURN",
                "TREASURY_RETURN",
                "VIX_CHANGE",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    ).astype(
        int
    )

    result[
        "VALID_MAIN_MODEL_ROW"
    ] = (
        (
            result[
                "MAIN_ARCHITECTURE_SAMPLE"
            ] == 1
        )
        & (
            result[
                "VALID_FULL_CONTROL_ROW"
            ] == 1
        )
    ).astype(
        int
    )

    result[
        "VALID_MODERATE_MODEL_ROW"
    ] = (
        (
            result[
                "MODERATE_ARCHITECTURE_SAMPLE"
            ] == 1
        )
        & (
            result[
                "VALID_FULL_CONTROL_ROW"
            ] == 1
        )
    ).astype(
        int
    )

    return result


# ============================================================
# 16. ETF COVERAGE RAPORU
# ============================================================

def build_etf_coverage(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    ETF bazında panel coverage raporu oluşturur.
    """

    rows = []

    for etf_id, group in panel.groupby(
        "ETF_ID",
        dropna=False,
    ):

        valid_returns = group.loc[
            group["ETF_RETURN"].notna()
        ]

        valid_main = group.loc[
            group[
                "VALID_MAIN_MODEL_ROW"
            ] == 1
        ]

        valid_moderate = group.loc[
            group[
                "VALID_MODERATE_MODEL_ROW"
            ] == 1
        ]

        rows.append(
            {
                "ETF_ID": etf_id,

                "ETF_NAME": (
                    group[
                        "ETF_NAME"
                    ].iloc[0]
                ),

                "ETF_TICKER": (
                    group[
                        "ETF_TICKER"
                    ].iloc[0]
                    if "ETF_TICKER"
                    in group.columns
                    else pd.NA
                ),

                "N_PANEL_ROWS": int(
                    len(group)
                ),

                "N_MONTHLY_RETURNS": int(
                    valid_returns.shape[0]
                ),

                "N_VALID_MAIN_ROWS": int(
                    valid_main.shape[0]
                ),

                "N_VALID_MODERATE_ROWS": int(
                    valid_moderate.shape[0]
                ),

                "START_DATE": (
                    valid_returns[
                        "DATE"
                    ].min()
                ),

                "END_DATE": (
                    valid_returns[
                        "DATE"
                    ].max()
                ),

                "AT_LEAST_24_MONTHS": int(
                    valid_returns.shape[0]
                    >= MIN_MONTHS_MAIN_ANALYSIS
                ),

                "AT_LEAST_60_MONTHS": int(
                    valid_returns.shape[0]
                    >= MIN_MONTHS_LONG_HISTORY
                ),

                "MAIN_ARCHITECTURE_SAMPLE": int(
                    group[
                        "MAIN_ARCHITECTURE_SAMPLE"
                    ].max()
                ),

                "MODERATE_ARCHITECTURE_SAMPLE": int(
                    group[
                        "MODERATE_ARCHITECTURE_SAMPLE"
                    ].max()
                ),

                "FINANCIAL_MATCH_WEIGHT": float(
                    group[
                        "FINANCIAL_MATCH_WEIGHT"
                    ].iloc[0]
                ),

                "CORE_TRANSITION_MAIN": (
                    group[
                        "CORE_TRANSITION_SENSITIVITY_MAIN"
                    ].iloc[0]
                ),

                "CORE_TRANSITION_MODERATE": (
                    group[
                        "CORE_TRANSITION_SENSITIVITY_MODERATE"
                    ].iloc[0]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. AYLIK COVERAGE RAPORU
# ============================================================

def build_monthly_coverage(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ay bazında panel coverage raporu oluşturur.
    """

    return (
        panel.groupby(
            "DATE",
            as_index=False,
        )
        .agg(
            N_ETFS=(
                "ETF_ID",
                "nunique",
            ),

            N_RETURN_OBSERVATIONS=(
                "ETF_RETURN",
                lambda series: (
                    series.notna().sum()
                ),
            ),

            N_MAIN_ETFS=(
                "MAIN_ARCHITECTURE_SAMPLE",
                "sum",
            ),

            N_MODERATE_ETFS=(
                "MODERATE_ARCHITECTURE_SAMPLE",
                "sum",
            ),

            N_VALID_MAIN_ROWS=(
                "VALID_MAIN_MODEL_ROW",
                "sum",
            ),

            N_VALID_MODERATE_ROWS=(
                "VALID_MODERATE_MODEL_ROW",
                "sum",
            ),

            CPU=(
                "CPU",
                "first",
            ),

            HIGH_CPU_REGIME=(
                "HIGH_CPU_REGIME",
                "first",
            ),

            VIX_LEVEL=(
                "VIX_LEVEL",
                "first",
            ),

            HIGH_VIX_REGIME=(
                "HIGH_VIX_REGIME",
                "first",
            ),
        )
    )


# ============================================================
# 18. SAMPLE SUMMARY
# ============================================================

def build_sample_summary(
    panel: pd.DataFrame,
    etf_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ana, moderate ve full örneklemleri özetler.
    """

    sample_definitions = {
        "FULL_RETURN_POOL": (
            panel["ETF_RETURN"].notna()
        ),

        "BASELINE_VALID_POOL": (
            panel[
                "VALID_BASELINE_ROW"
            ] == 1
        ),

        "FULL_CONTROL_POOL": (
            panel[
                "VALID_FULL_CONTROL_ROW"
            ] == 1
        ),

        "MAIN_QUALITY_SAMPLE": (
            panel[
                "VALID_MAIN_MODEL_ROW"
            ] == 1
        ),

        "MODERATE_QUALITY_SAMPLE": (
            panel[
                "VALID_MODERATE_MODEL_ROW"
            ] == 1
        ),

        "MAIN_HIGH_CPU_SAMPLE": (
            (
                panel[
                    "VALID_MAIN_MODEL_ROW"
                ] == 1
            )
            & (
                panel[
                    "HIGH_CPU_REGIME"
                ] == 1
            )
        ),

        "MAIN_CPU_MARKET_STRESS_SAMPLE": (
            (
                panel[
                    "VALID_MAIN_MODEL_ROW"
                ] == 1
            )
            & (
                panel[
                    "CPU_AND_MARKET_STRESS"
                ] == 1
            )
        ),
    }

    rows = []

    for sample_name, mask in (
        sample_definitions.items()
    ):

        subset = panel.loc[
            mask
        ]

        rows.append(
            {
                "SAMPLE": sample_name,

                "N_ROWS": int(
                    len(subset)
                ),

                "N_ETFS": int(
                    subset[
                        "ETF_ID"
                    ].nunique()
                ),

                "N_MONTHS": int(
                    subset[
                        "DATE"
                    ].nunique()
                ),

                "START_DATE": (
                    subset[
                        "DATE"
                    ].min()
                ),

                "END_DATE": (
                    subset[
                        "DATE"
                    ].max()
                ),

                "MEAN_ETF_RETURN": float(
                    subset[
                        "ETF_RETURN"
                    ].mean()
                ),

                "STD_ETF_RETURN": float(
                    subset[
                        "ETF_RETURN"
                    ].std(
                        ddof=1
                    )
                ),
            }
        )

    rows.append(
        {
            "SAMPLE": (
                "ETFS_AT_LEAST_24_MONTHS"
            ),

            "N_ROWS": np.nan,

            "N_ETFS": int(
                etf_coverage[
                    "AT_LEAST_24_MONTHS"
                ].sum()
            ),

            "N_MONTHS": np.nan,

            "START_DATE": pd.NaT,

            "END_DATE": pd.NaT,

            "MEAN_ETF_RETURN": np.nan,

            "STD_ETF_RETURN": np.nan,
        }
    )

    rows.append(
        {
            "SAMPLE": (
                "ETFS_AT_LEAST_60_MONTHS"
            ),

            "N_ROWS": np.nan,

            "N_ETFS": int(
                etf_coverage[
                    "AT_LEAST_60_MONTHS"
                ].sum()
            ),

            "N_MONTHS": np.nan,

            "START_DATE": pd.NaT,

            "END_DATE": pd.NaT,

            "MEAN_ETF_RETURN": np.nan,

            "STD_ETF_RETURN": np.nan,
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 19. DESCRIPTIVE STATISTICS
# ============================================================

def build_descriptive_statistics(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ana panel değişkenlerinin tanımlayıcı istatistiklerini üretir.
    """

    variables = [
        "ETF_RETURN",
        "CPU",
        "CPU_Z",
        "CPU_DIFF",
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_LEVEL",
        "VIX_CHANGE",
        "CORE_TRANSITION_SENSITIVITY_MAIN",
        "FINANCIAL_RESILIENCE_MAIN",
        "GROWTH_DURATION_EXPOSURE_MAIN",
        "PORTFOLIO_CONCENTRATION_MAIN",
        "EXTENDED_TRANSITION_SENSITIVITY_MAIN",
    ]

    rows = []

    for variable in variables:

        if variable not in panel.columns:
            continue

        values = safe_numeric(
            panel[
                variable
            ]
        )

        rows.append(
            {
                "VARIABLE": variable,

                "N": int(
                    values.notna().sum()
                ),

                "MEAN": float(
                    values.mean()
                ),

                "STD": float(
                    values.std(
                        ddof=1
                    )
                ),

                "MIN": float(
                    values.min()
                ),

                "P25": float(
                    values.quantile(
                        0.25
                    )
                ),

                "MEDIAN": float(
                    values.median()
                ),

                "P75": float(
                    values.quantile(
                        0.75
                    )
                ),

                "MAX": float(
                    values.max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 20. VARIABLE DICTIONARY
# ============================================================

def build_variable_dictionary() -> pd.DataFrame:
    """
    Master panel değişken sözlüğünü oluşturur.
    """

    rows = [
        {
            "VARIABLE": "ETF_RETURN",
            "TYPE": "Dependent variable",
            "SOURCE": (
                "Yahoo Finance adjusted month-end prices"
            ),
            "FORMULA": (
                "P_i,t / P_i,t-1 - 1"
            ),
            "RESEARCH_QUESTION": (
                "RQ1–RQ4"
            ),
        },

        {
            "VARIABLE": "CPU_Z",
            "TYPE": (
                "Standardized climate-policy uncertainty"
            ),
            "SOURCE": (
                "Official CPU narrow index"
            ),
            "FORMULA": (
                "(CPU_t - mean CPU) / standard deviation CPU"
            ),
            "RESEARCH_QUESTION": (
                "RQ1–RQ4"
            ),
        },

        {
            "VARIABLE": (
                "CORE_TRANSITION_SENSITIVITY_MAIN"
            ),
            "TYPE": (
                "Primary portfolio architecture factor"
            ),
            "SOURCE": (
                "2025Q4 SEC N-PORT holdings and FY2024 fundamentals"
            ),
            "FORMULA": (
                "-Financial Resilience "
                "+ Growth-Duration Exposure "
                "+ Portfolio Concentration"
            ),
            "RESEARCH_QUESTION": (
                "RQ1–RQ4"
            ),
        },

        {
            "VARIABLE": (
                "CPU_Z_X_CORE_TRANSITION_SENSITIVITY_MAIN"
            ),
            "TYPE": (
                "Primary conditional-pricing interaction"
            ),
            "SOURCE": (
                "CPU and portfolio architecture"
            ),
            "FORMULA": (
                "CPU_Z × Core Transition Sensitivity Main"
            ),
            "RESEARCH_QUESTION": (
                "RQ1, RQ2"
            ),
        },

        {
            "VARIABLE": (
                "CPU_AND_MARKET_STRESS"
            ),
            "TYPE": (
                "Joint stress-state indicator"
            ),
            "SOURCE": (
                "CPU and VIX"
            ),
            "FORMULA": (
                "1 when both CPU and VIX are in their upper quartiles"
            ),
            "RESEARCH_QUESTION": (
                "RQ3"
            ),
        },

        {
            "VARIABLE": (
                "CUM_RETURN_LEAD3"
            ),
            "TYPE": (
                "Forward recovery outcome"
            ),
            "SOURCE": (
                "ETF monthly returns"
            ),
            "FORMULA": (
                "Three-month cumulative forward ETF return"
            ),
            "RESEARCH_QUESTION": (
                "RQ4"
            ),
        },

        {
            "VARIABLE": (
                "FROZEN_ARCHITECTURE_FLAG"
            ),
            "TYPE": (
                "Research-design disclosure"
            ),
            "SOURCE": (
                "Architecture snapshot metadata"
            ),
            "FORMULA": (
                "1 for all observations using 2025Q4 architecture"
            ),
            "RESEARCH_QUESTION": (
                "Methodological limitation"
            ),
        },
    ]

    for variable in MARKET_CONTROL_VARIABLES:

        rows.append(
            {
                "VARIABLE": variable,
                "TYPE": "Time-series control",
                "SOURCE": (
                    "Yahoo Finance"
                ),
                "FORMULA": (
                    "Monthly return or level"
                ),
                "RESEARCH_QUESTION": (
                    "Control variable"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 21. VALIDATION
# ============================================================

def build_validation(
    panel: pd.DataFrame,
    main_sample: pd.DataFrame,
    moderate_sample: pd.DataFrame,
    etf_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Master panelin mekanik ve metodolojik kontrollerini yapar.
    """

    duplicate_rows = int(
        panel.duplicated(
            subset=[
                "ETF_ID",
                "DATE",
            ],
            keep=False,
        ).sum()
    )

    main_etfs = int(
        main_sample[
            "ETF_ID"
        ].nunique()
    )

    moderate_etfs = int(
        moderate_sample[
            "ETF_ID"
        ].nunique()
    )

    frozen_flag_values = (
        panel[
            "FROZEN_ARCHITECTURE_FLAG"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    rows = [
        {
            "CHECK": "MASTER_PANEL_ROWS",
            "VALUE": len(
                panel
            ),
            "PASS": int(
                len(
                    panel
                ) > 0
            ),
        },

        {
            "CHECK": "MASTER_PANEL_ETFS",
            "VALUE": panel[
                "ETF_ID"
            ].nunique(),
            "PASS": int(
                panel[
                    "ETF_ID"
                ].nunique()
                > 0
            ),
        },

        {
            "CHECK": "MASTER_PANEL_MONTHS",
            "VALUE": panel[
                "DATE"
            ].nunique(),
            "PASS": int(
                panel[
                    "DATE"
                ].nunique()
                >= 180
            ),
        },

        {
            "CHECK": (
                "DUPLICATE_ETF_MONTH_ROWS"
            ),
            "VALUE": duplicate_rows,
            "PASS": int(
                duplicate_rows == 0
            ),
        },

        {
            "CHECK": "MAIN_QUALITY_ETFS",
            "VALUE": main_etfs,
            "PASS": int(
                main_etfs > 0
            ),
        },

        {
            "CHECK": "MODERATE_QUALITY_ETFS",
            "VALUE": moderate_etfs,
            "PASS": int(
                moderate_etfs
                >= main_etfs
            ),
        },

        {
            "CHECK": (
                "ETFS_AT_LEAST_24_MONTHS"
            ),
            "VALUE": int(
                etf_coverage[
                    "AT_LEAST_24_MONTHS"
                ].sum()
            ),
            "PASS": int(
                etf_coverage[
                    "AT_LEAST_24_MONTHS"
                ].sum()
                > 0
            ),
        },

        {
            "CHECK": (
                "VALID_MAIN_MODEL_ROWS"
            ),
            "VALUE": int(
                panel[
                    "VALID_MAIN_MODEL_ROW"
                ].sum()
            ),
            "PASS": int(
                panel[
                    "VALID_MAIN_MODEL_ROW"
                ].sum()
                > 0
            ),
        },

        {
            "CHECK": (
                "VALID_MODERATE_MODEL_ROWS"
            ),
            "VALUE": int(
                panel[
                    "VALID_MODERATE_MODEL_ROW"
                ].sum()
            ),
            "PASS": int(
                panel[
                    "VALID_MODERATE_MODEL_ROW"
                ].sum()
                >= panel[
                    "VALID_MAIN_MODEL_ROW"
                ].sum()
            ),
        },

        {
            "CHECK": "CPU_MISSING_ROWS",
            "VALUE": int(
                panel[
                    "CPU"
                ].isna().sum()
            ),
            "PASS": int(
                panel[
                    "CPU"
                ].isna().sum()
                == 0
            ),
        },

        {
            "CHECK": (
                "MARKET_CONTROL_MISSING_ROWS"
            ),
            "VALUE": int(
                panel[
                    "MARKET_RETURN"
                ].isna().sum()
            ),
            "PASS": int(
                panel[
                    "MARKET_RETURN"
                ].isna().sum()
                == 0
            ),
        },

        {
            "CHECK": (
                "UNRELIABLE_MAIN_ARCHITECTURE_ROWS"
            ),
            "VALUE": int(
                (
                    panel[
                        "CORE_TRANSITION_SENSITIVITY_MAIN"
                    ].notna()
                    & (
                        panel[
                            "FINANCIAL_MATCH_WEIGHT"
                        ] < 0.80
                    )
                ).sum()
            ),
            "PASS": int(
                (
                    panel[
                        "CORE_TRANSITION_SENSITIVITY_MAIN"
                    ].notna()
                    & (
                        panel[
                            "FINANCIAL_MATCH_WEIGHT"
                        ] < 0.80
                    )
                ).sum()
                == 0
            ),
        },

        {
            "CHECK": (
                "FROZEN_ARCHITECTURE_DISCLOSED"
            ),
            "VALUE": str(
                frozen_flag_values
            ),
            "PASS": int(
                frozen_flag_values
                == [1]
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 22. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 82)
    print("22 - MASTER CPU–RETURN–PORTFOLIO ARCHITECTURE PANEL")
    print("=" * 82)

    # --------------------------------------------------------
    # 1. Girdiler
    # --------------------------------------------------------

    print(
        "\n1/10 - ETF aylık getirileri okunuyor..."
    )

    returns = prepare_etf_returns(
        ETF_RETURN_FILE
    )

    print(
        f"Return satırı: "
        f"{len(returns):,}"
    )

    print(
        f"Return ETF sayısı: "
        f"{returns['ETF_ID'].nunique():,}"
    )

    print(
        "\n2/10 - CPU paneli okunuyor..."
    )

    cpu = prepare_cpu(
        CPU_FILE
    )

    print(
        f"CPU ay sayısı: "
        f"{cpu['DATE'].nunique():,}"
    )

    print(
        "\n3/10 - Piyasa kontrolleri okunuyor..."
    )

    controls = prepare_market_controls(
        MARKET_CONTROL_FILE
    )

    print(
        f"Kontrol ay sayısı: "
        f"{controls['DATE'].nunique():,}"
    )

    print(
        "\n4/10 - Portfolio architecture paneli okunuyor..."
    )

    architecture = prepare_architecture(
        ARCHITECTURE_FILE
    )

    ticker_mapping = prepare_ticker_mapping(
        TICKER_MAPPING_FILE
    )

    architecture = architecture.merge(
        ticker_mapping,
        on="ETF_ID",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Architecture ETF sayısı: "
        f"{architecture['ETF_ID'].nunique():,}"
    )

    print(
        f"Main-quality ETF sayısı: "
        f"{architecture['MAIN_ARCHITECTURE_SAMPLE'].sum():,}"
    )

    print(
        f"Moderate-quality ETF sayısı: "
        f"{architecture['MODERATE_ARCHITECTURE_SAMPLE'].sum():,}"
    )

    # --------------------------------------------------------
    # 2. Return + architecture
    # --------------------------------------------------------

    print(
        "\n5/10 - ETF getirileri architecture ile birleştiriliyor..."
    )

    return_etf_count = returns[
        "ETF_ID"
    ].nunique()

    panel = returns.merge(
        architecture,
        on="ETF_ID",
        how="left",
        validate="many_to_one",
        suffixes=(
            "_RETURN",
            "_ARCH",
        ),
    )

    if "ETF_NAME_RETURN" in panel.columns:

        panel["ETF_NAME"] = (
            panel[
                "ETF_NAME_RETURN"
            ]
            .fillna(
                panel.get(
                    "ETF_NAME_ARCH"
                )
            )
        )

    if "ETF_TICKER_RETURN" in panel.columns:

        panel["ETF_TICKER"] = (
            panel[
                "ETF_TICKER_RETURN"
            ]
            .fillna(
                panel.get(
                    "ETF_TICKER_ARCH"
                )
            )
        )

    matched_architecture_etfs = panel.loc[
        panel[
            "FINANCIAL_MATCH_WEIGHT"
        ].notna(),
        "ETF_ID",
    ].nunique()

    # --------------------------------------------------------
    # 3. CPU ve kontroller
    # --------------------------------------------------------

    print(
        "\n6/10 - CPU ve piyasa kontrolleri ekleniyor..."
    )

    panel = panel.merge(
        cpu,
        on="DATE",
        how="left",
        validate="many_to_one",
    )

    panel = panel.merge(
        controls,
        on="DATE",
        how="left",
        validate="many_to_one",
    )

    panel = panel.loc[
        panel["DATE"].between(
            START_DATE,
            END_DATE,
        )
    ].copy()

    # --------------------------------------------------------
    # 4. Panel değişkenleri
    # --------------------------------------------------------

    print(
        "\n7/10 - Zaman, stres ve recovery değişkenleri oluşturuluyor..."
    )

    panel = add_panel_variables(
        panel
    )

    print(
        "\n8/10 - CPU × architecture etkileşimleri oluşturuluyor..."
    )

    panel = add_interactions(
        panel
    )

    # --------------------------------------------------------
    # 5. Örneklemler
    # --------------------------------------------------------

    full_return_pool = panel.loc[
        panel[
            "ETF_RETURN"
        ].notna()
    ].copy()

    main_sample = panel.loc[
        panel[
            "VALID_MAIN_MODEL_ROW"
        ] == 1
    ].copy()

    moderate_sample = panel.loc[
        panel[
            "VALID_MODERATE_MODEL_ROW"
        ] == 1
    ].copy()

    print(
        "\n9/10 - Coverage ve örneklem raporları hazırlanıyor..."
    )

    etf_coverage = build_etf_coverage(
        panel
    )

    monthly_coverage = (
        build_monthly_coverage(
            panel
        )
    )

    sample_summary = build_sample_summary(
        panel=panel,
        etf_coverage=etf_coverage,
    )

    descriptives = (
        build_descriptive_statistics(
            main_sample
        )
    )

    diagnostics = pd.DataFrame(
        [
            {
                "METRIC": (
                    "RETURN_INPUT_ROWS"
                ),
                "VALUE": len(
                    returns
                ),
            },

            {
                "METRIC": (
                    "RETURN_INPUT_ETFS"
                ),
                "VALUE": return_etf_count,
            },

            {
                "METRIC": (
                    "RETURN_ETFS_WITH_ARCHITECTURE"
                ),
                "VALUE": (
                    matched_architecture_etfs
                ),
            },

            {
                "METRIC": (
                    "ARCHITECTURE_MATCH_RATE"
                ),
                "VALUE": (
                    matched_architecture_etfs
                    / return_etf_count
                    if return_etf_count > 0
                    else np.nan
                ),
            },

            {
                "METRIC": (
                    "MASTER_PANEL_ROWS"
                ),
                "VALUE": len(
                    panel
                ),
            },

            {
                "METRIC": (
                    "MASTER_PANEL_ETFS"
                ),
                "VALUE": panel[
                    "ETF_ID"
                ].nunique(),
            },

            {
                "METRIC": (
                    "MASTER_PANEL_MONTHS"
                ),
                "VALUE": panel[
                    "DATE"
                ].nunique(),
            },

            {
                "METRIC": (
                    "FULL_RETURN_POOL_ROWS"
                ),
                "VALUE": len(
                    full_return_pool
                ),
            },

            {
                "METRIC": (
                    "MAIN_SAMPLE_ROWS"
                ),
                "VALUE": len(
                    main_sample
                ),
            },

            {
                "METRIC": (
                    "MAIN_SAMPLE_ETFS"
                ),
                "VALUE": main_sample[
                    "ETF_ID"
                ].nunique(),
            },

            {
                "METRIC": (
                    "MODERATE_SAMPLE_ROWS"
                ),
                "VALUE": len(
                    moderate_sample
                ),
            },

            {
                "METRIC": (
                    "MODERATE_SAMPLE_ETFS"
                ),
                "VALUE": moderate_sample[
                    "ETF_ID"
                ].nunique(),
            },

            {
                "METRIC": (
                    "FROZEN_ARCHITECTURE_SNAPSHOT"
                ),
                "VALUE": (
                    ARCHITECTURE_SNAPSHOT
                ),
            },
        ]
    )

    validation = build_validation(
        panel=panel,
        main_sample=main_sample,
        moderate_sample=moderate_sample,
        etf_coverage=etf_coverage,
    )

    variable_dictionary = (
        build_variable_dictionary()
    )

    # --------------------------------------------------------
    # 6. Kaydet
    # --------------------------------------------------------

    print(
        "\n10/10 - Master panel ve raporlar kaydediliyor..."
    )

    panel.to_csv(
        MASTER_PANEL_FILE,
        index=False,
    )

    panel.to_parquet(
        MASTER_PANEL_PARQUET_FILE,
        index=False,
    )

    main_sample.to_csv(
        MAIN_SAMPLE_FILE,
        index=False,
    )

    main_sample.to_parquet(
        MAIN_SAMPLE_PARQUET_FILE,
        index=False,
    )

    moderate_sample.to_csv(
        MODERATE_SAMPLE_FILE,
        index=False,
    )

    full_return_pool.to_csv(
        FULL_RETURN_POOL_FILE,
        index=False,
    )

    etf_coverage.to_csv(
        ETF_COVERAGE_FILE,
        index=False,
    )

    monthly_coverage.to_csv(
        MONTHLY_COVERAGE_FILE,
        index=False,
    )

    sample_summary.to_csv(
        SAMPLE_SUMMARY_FILE,
        index=False,
    )

    diagnostics.to_csv(
        MERGE_DIAGNOSTICS_FILE,
        index=False,
    )

    descriptives.to_csv(
        DESCRIPTIVE_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    variable_dictionary.to_csv(
        VARIABLE_DICTIONARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Ekran çıktıları
    # --------------------------------------------------------

    print(
        "\nMASTER PANEL HAZIR"
    )

    print("=" * 82)

    print(
        "\nMerge diagnostics:"
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    print(
        "\nÖrneklem özeti:"
    )

    print(
        sample_summary.to_string(
            index=False
        )
    )

    print(
        "\nETF coverage özeti:"
    )

    print(
        etf_coverage[
            [
                "N_MONTHLY_RETURNS",
                "N_VALID_MAIN_ROWS",
                "N_VALID_MODERATE_ROWS",
            ]
        ]
        .describe()
        .to_string()
    )

    print(
        "\nMain sample descriptive statistics:"
    )

    print(
        descriptives[
            [
                "VARIABLE",
                "N",
                "MEAN",
                "STD",
                "MEDIAN",
                "MIN",
                "MAX",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nValidation sonuçları:"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    failed_checks = validation.loc[
        validation[
            "PASS"
        ] == 0
    ]

    print(
        "\nBaşarısız kontrol sayısı: "
        f"{len(failed_checks):,}"
    )

    print(
        "\nMETODOLOJİK UYARI:"
    )

    print(
        "Architecture değişkenleri 2025Q4 holdings kesitine dayanır. "
        "Bu dosya frozen-architecture panelidir. Tarihsel holdings "
        "eklenmeden time-varying portfolio architecture iddiası "
        "kurulmamalıdır."
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        MASTER_PANEL_FILE
    )

    print(
        MAIN_SAMPLE_FILE
    )

    print(
        MODERATE_SAMPLE_FILE
    )

    print(
        FULL_RETURN_POOL_FILE
    )

    print(
        ETF_COVERAGE_FILE
    )

    print(
        MONTHLY_COVERAGE_FILE
    )

    print(
        SAMPLE_SUMMARY_FILE
    )

    print(
        MERGE_DIAGNOSTICS_FILE
    )

    print(
        DESCRIPTIVE_FILE
    )

    print(
        VALIDATION_FILE
    )

    print(
        VARIABLE_DICTIONARY_FILE
    )


if __name__ == "__main__":
    main()