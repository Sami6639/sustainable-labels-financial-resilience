from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 25_build_final_econometric_panel.py
#
# AMAÇ
# ----
# Nihai portfolio architecture kanallarını aşağıdaki verilerle
# birleştirerek ekonometrik analiz panelini oluşturmak:
#
# 1. ETF aylık getirileri
# 2. Climate Policy Uncertainty
# 3. Piyasa kontrolleri
# 4. Nihai holdings-based architecture kanalları
#
# ANA KANALLAR
# ------------
# - Internal Financial Capacity
# - External Financing Dependence
# - Growth-Duration Exposure
# - Portfolio Concentration
#
# ROBUSTNESS KANALLARI
# --------------------
# - Financial Architecture Risk
# - Extended Architecture Risk
# - Leverage Exposure
# - PCA Component 1
#
# RQ3
# ---
# CPU ve piyasa stresi altında koşullu fiyatlama.
#
# RQ4
# ---
# CPU şoklarından sonra 1, 3, 6 ve 12 aylık toparlanma.
#
# METODOLOJİK NOT
# ----------------
# Architecture değişkenleri 2025Q4 holdings kesitinden gelir.
# Bu nedenle FROZEN_ARCHITECTURE_FLAG = 1 olarak açıklanır.
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
    / "24b_final_architecture_channels.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

FINAL_PANEL_FILE = (
    OUTPUT_DIR
    / "25_final_econometric_panel.csv"
)

FINAL_PANEL_PARQUET_FILE = (
    OUTPUT_DIR
    / "25_final_econometric_panel.parquet"
)

PRIMARY_CHANNEL_PANEL_FILE = (
    OUTPUT_DIR
    / "25_primary_channel_panel.csv"
)

FINANCIAL_RISK_PANEL_FILE = (
    OUTPUT_DIR
    / "25_financial_architecture_risk_panel.csv"
)

PCA_PANEL_FILE = (
    OUTPUT_DIR
    / "25_pca_robustness_panel.csv"
)

ETF_COVERAGE_FILE = (
    OUTPUT_DIR
    / "25_final_panel_etf_coverage.csv"
)

CHANNEL_SAMPLE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "25_channel_sample_summary.csv"
)

MONTHLY_COVERAGE_FILE = (
    OUTPUT_DIR
    / "25_final_panel_monthly_coverage.csv"
)

DESCRIPTIVE_FILE = (
    OUTPUT_DIR
    / "25_final_panel_descriptive_statistics.csv"
)

CORRELATION_FILE = (
    OUTPUT_DIR
    / "25_final_panel_correlation_matrix.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "25_final_panel_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "25_final_panel_variable_dictionary.csv"
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

RETURN_ABSOLUTE_LIMIT = 5.0


# ============================================================
# 5. NİHAİ KANALLAR
# ============================================================

PRIMARY_CHANNELS = [
    "INTERNAL_FINANCIAL_CAPACITY",
    "EXTERNAL_FINANCING_DEPENDENCE",
    "GROWTH_DURATION_EXPOSURE_FINAL",
    "PORTFOLIO_CONCENTRATION_FINAL",
]

COMPOSITE_CHANNELS = [
    "FINANCIAL_ARCHITECTURE_RISK_FINAL",
    "EXTENDED_ARCHITECTURE_RISK_FINAL",
]

ROBUSTNESS_CHANNELS = [
    "LEVERAGE_EXPOSURE",
    "PCA_COMPONENT_1",
]

ALL_CHANNELS = (
    PRIMARY_CHANNELS
    + COMPOSITE_CHANNELS
    + ROBUSTNESS_CHANNELS
)


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

MARKET_CONTROLS = [
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
    Metin alanlarını temizler.
    """

    result = (
        series.astype("string")
        .str.strip()
    )

    invalid = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "<NA>",
    }

    return result.mask(
        result.str.upper().isin(
            invalid
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
    Tarihleri ay sonuna standardize eder.
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
    Zorunlu sütunları kontrol eder.
    """

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise KeyError(
            f"{dataset_name} içinde eksik sütunlar:\n"
            + "\n".join(
                missing
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
# 9. ETF GETİRİLERİ
# ============================================================

def prepare_returns(
    path: Path,
) -> pd.DataFrame:
    """
    ETF aylık getiri panelini hazırlar.
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

    returns[
        "EXTREME_RETURN_FLAG"
    ] = (
        returns["ETF_RETURN"].abs()
        > RETURN_ABSOLUTE_LIMIT
    ).astype(
        int
    )

    returns.loc[
        returns[
            "EXTREME_RETURN_FLAG"
        ] == 1,
        "ETF_RETURN",
    ] = np.nan

    duplicate_rows = int(
        returns.duplicated(
            subset=[
                "ETF_ID",
                "DATE",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_rows > 0:

        raise RuntimeError(
            "ETF return panelinde duplicate ETF_ID–DATE "
            f"satırı bulundu: {duplicate_rows}"
        )

    return returns


# ============================================================
# 10. CPU PANELİ
# ============================================================

def prepare_cpu(
    path: Path,
) -> pd.DataFrame:
    """
    CPU aylık panelini hazırlar.
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
            "EXTREME_CPU_REGIME",
        ],
        "CPU PANEL",
    )

    cpu["DATE"] = standardize_month_end(
        cpu["DATE"]
    )

    selected_columns = [
        "DATE",
    ]

    for variable in CPU_VARIABLES:

        if variable in cpu.columns:

            selected_columns.append(
                variable
            )

    cpu = cpu[
        list(
            dict.fromkeys(
                selected_columns
            )
        )
    ].copy()

    for column in cpu.columns:

        if column not in [
            "DATE",
            "CPU_REGIME",
        ]:

            cpu[column] = safe_numeric(
                cpu[column]
            )

    cpu = cpu.loc[
        cpu["DATE"].between(
            START_DATE,
            END_DATE,
        )
    ].copy()

    duplicate_rows = int(
        cpu.duplicated(
            subset=[
                "DATE",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_rows > 0:

        raise RuntimeError(
            "CPU panelinde duplicate ay bulundu."
        )

    return cpu


# ============================================================
# 11. PİYASA KONTROLLERİ
# ============================================================

def prepare_controls(
    path: Path,
) -> pd.DataFrame:
    """
    Piyasa kontrol panelini hazırlar.
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
            "VIX_LEVEL",
        ],
        "MARKET CONTROLS",
    )

    controls["DATE"] = standardize_month_end(
        controls["DATE"]
    )

    selected_columns = [
        "DATE",
    ]

    for variable in MARKET_CONTROLS:

        if variable in controls.columns:

            selected_columns.append(
                variable
            )

    controls = controls[
        selected_columns
    ].copy()

    for column in controls.columns:

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

    duplicate_rows = int(
        controls.duplicated(
            subset=[
                "DATE",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_rows > 0:

        raise RuntimeError(
            "Piyasa kontrol panelinde duplicate ay bulundu."
        )

    return controls


# ============================================================
# 12. ARCHITECTURE KANALLARI
# ============================================================

def prepare_architecture(
    path: Path,
) -> pd.DataFrame:
    """
    Nihai architecture kanallarını hazırlar.
    """

    architecture = normalize_columns(
        read_data(
            path
        )
    )

    required_columns = [
        "ETF_ID",
        "ETF_NAME",
        "FINANCIAL_MATCH_WEIGHT",
    ]

    for channel in (
        PRIMARY_CHANNELS
        + COMPOSITE_CHANNELS
    ):

        required_columns.extend(
            [
                channel,
                f"{channel}_MODERATE",
                f"{channel}_MAIN",
                f"{channel}_STRICT",
                f"{channel}_VALID_MODERATE",
                f"{channel}_VALID_MAIN",
                f"{channel}_VALID_STRICT",
            ]
        )

    required_columns.extend(
        [
            "LEVERAGE_EXPOSURE",
            "LEVERAGE_EXPOSURE_MAIN",
            "LEVERAGE_EXPOSURE_VALID_MAIN",
        ]
    )

    require_columns(
        architecture,
        required_columns,
        "FINAL ARCHITECTURE",
    )

    architecture["ETF_ID"] = clean_text(
        architecture["ETF_ID"]
    )

    architecture["ETF_NAME"] = clean_text(
        architecture["ETF_NAME"]
    )

    architecture = architecture.drop_duplicates(
        subset=[
            "ETF_ID",
        ]
    ).copy()

    architecture[
        "ARCHITECTURE_SNAPSHOT"
    ] = ARCHITECTURE_SNAPSHOT

    architecture[
        "FROZEN_ARCHITECTURE_FLAG"
    ] = 1

    return architecture


# ============================================================
# 13. ZAMAN VE GETİRİ DEĞİŞKENLERİ
# ============================================================

def add_time_and_return_variables(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zaman, gecikmeli getiri ve ileri toparlanma değişkenlerini
    oluşturur.
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

    result["DATE_GROUP"] = (
        result["YEAR_MONTH"]
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

    for horizon in [
        1,
        2,
        3,
        6,
        12,
    ]:

        result[
            f"ETF_RETURN_LEAD{horizon}"
        ] = (
            result.groupby(
                "ETF_ID"
            )["ETF_RETURN"]
            .shift(
                -horizon
            )
        )

    # --------------------------------------------------------
    # İleri kümülatif getiriler
    # --------------------------------------------------------

    def forward_cumulative_return(
        group: pd.DataFrame,
        horizon: int,
    ) -> pd.Series:

        return_series = group[
            "ETF_RETURN"
        ]

        forward_returns = [
            return_series.shift(
                -step
            )
            for step in range(
                1,
                horizon + 1,
            )
        ]

        forward_frame = pd.concat(
            forward_returns,
            axis=1,
        )

        valid_count = (
            forward_frame
            .notna()
            .sum(
                axis=1
            )
        )

        cumulative = (
            1.0
            + forward_frame
        ).prod(
            axis=1,
            min_count=horizon,
        ) - 1.0

        return cumulative.where(
            valid_count
            == horizon
        )

    for horizon in [
        3,
        6,
        12,
    ]:

        result[
            f"CUM_RETURN_LEAD{horizon}"
        ] = (
            result.groupby(
                "ETF_ID",
                group_keys=False,
            )
            .apply(
                lambda group: (
                    forward_cumulative_return(
                        group,
                        horizon,
                    )
                ),
                include_groups=False,
            )
            .reset_index(
                level=0,
                drop=True,
            )
            .reindex(
                result.index
            )
        )

    result[
        "NEGATIVE_RETURN_FLAG"
    ] = (
        result["ETF_RETURN"]
        < 0
    ).astype(
        "Int64"
    )

    result[
        "POST_2020"
    ] = (
        result["DATE"]
        >= pd.Timestamp(
            "2020-01-31"
        )
    ).astype(
        int
    )

    result[
        "COVID_PERIOD"
    ] = (
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

    return result


# ============================================================
# 14. STRES DEĞİŞKENLERİ
# ============================================================

def add_stress_variables(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    RQ3 için piyasa stresi ve ortak stres değişkenlerini üretir.
    """

    result = panel.copy()

    monthly_state = (
        result[
            [
                "DATE",
                "VIX_LEVEL",
                "CPU",
                "CPU_Z",
                "HIGH_CPU_REGIME",
                "EXTREME_CPU_REGIME",
            ]
        ]
        .drop_duplicates(
            subset=[
                "DATE",
            ]
        )
        .sort_values(
            "DATE"
        )
        .copy()
    )

    monthly_state[
        "VIX_LEVEL_Z"
    ] = zscore(
        monthly_state[
            "VIX_LEVEL"
        ]
    )

    vix_p75 = float(
        monthly_state[
            "VIX_LEVEL"
        ]
        .quantile(
            0.75
        )
    )

    vix_p90 = float(
        monthly_state[
            "VIX_LEVEL"
        ]
        .quantile(
            0.90
        )
    )

    monthly_state[
        "HIGH_VIX_REGIME"
    ] = (
        monthly_state[
            "VIX_LEVEL"
        ]
        >= vix_p75
    ).astype(
        int
    )

    monthly_state[
        "EXTREME_VIX_REGIME"
    ] = (
        monthly_state[
            "VIX_LEVEL"
        ]
        >= vix_p90
    ).astype(
        int
    )

    monthly_state[
        "CPU_AND_VIX_STRESS"
    ] = (
        (
            monthly_state[
                "HIGH_CPU_REGIME"
            ]
            == 1
        )
        & (
            monthly_state[
                "HIGH_VIX_REGIME"
            ]
            == 1
        )
    ).astype(
        int
    )

    monthly_state[
        "EXTREME_CPU_AND_VIX_STRESS"
    ] = (
        (
            monthly_state[
                "EXTREME_CPU_REGIME"
            ]
            == 1
        )
        & (
            monthly_state[
                "EXTREME_VIX_REGIME"
            ]
            == 1
        )
    ).astype(
        int
    )

    result = result.merge(
        monthly_state[
            [
                "DATE",
                "VIX_LEVEL_Z",
                "HIGH_VIX_REGIME",
                "EXTREME_VIX_REGIME",
                "CPU_AND_VIX_STRESS",
                "EXTREME_CPU_AND_VIX_STRESS",
            ]
        ],
        on="DATE",
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================
# 15. CPU × ARCHITECTURE ETKİLEŞİMLERİ
# ============================================================

def add_channel_interactions(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her architecture kanalı için CPU ve stres etkileşimlerini üretir.
    """

    result = panel.copy()

    for channel in ALL_CHANNELS:

        if channel not in result.columns:
            continue

        channel_variants = [
            channel,
            f"{channel}_MODERATE",
            f"{channel}_MAIN",
            f"{channel}_STRICT",
        ]

        for variant in channel_variants:

            if variant not in result.columns:
                continue

            result[
                f"CPU_Z_X_{variant}"
            ] = (
                result["CPU_Z"]
                * result[variant]
            )

            result[
                f"LOG_CPU_Z_X_{variant}"
            ] = (
                result["LOG_CPU_Z"]
                * result[variant]
            )

            result[
                f"CPU_DIFF_X_{variant}"
            ] = (
                result["CPU_DIFF"]
                * result[variant]
            )

            result[
                f"CPU_Z_L1_X_{variant}"
            ] = (
                result["CPU_Z_L1"]
                * result[variant]
            )

            result[
                f"HIGH_CPU_X_{variant}"
            ] = (
                result["HIGH_CPU_REGIME"]
                * result[variant]
            )

            result[
                f"EXTREME_CPU_X_{variant}"
            ] = (
                result["EXTREME_CPU_REGIME"]
                * result[variant]
            )

            result[
                f"HIGH_VIX_X_{variant}"
            ] = (
                result["HIGH_VIX_REGIME"]
                * result[variant]
            )

            result[
                f"CPU_VIX_STRESS_X_{variant}"
            ] = (
                result["CPU_AND_VIX_STRESS"]
                * result[variant]
            )

            result[
                f"EXTREME_CPU_VIX_X_{variant}"
            ] = (
                result[
                    "EXTREME_CPU_AND_VIX_STRESS"
                ]
                * result[variant]
            )

            if "CPU_SHOCK" in result.columns:

                result[
                    f"CPU_SHOCK_X_{variant}"
                ] = (
                    result["CPU_SHOCK"]
                    * result[variant]
                )

            if "CPU_INSTRUMENT" in result.columns:

                result[
                    f"CPU_INSTRUMENT_X_{variant}"
                ] = (
                    result["CPU_INSTRUMENT"]
                    * result[variant]
                )

    return result


# ============================================================
# 16. KANAL ÖRNEKLEM BAYRAKLARI
# ============================================================

def add_channel_sample_flags(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her kanal için valid model satırı bayraklarını üretir.
    """

    result = panel.copy()

    base_variables = [
        "ETF_RETURN",
        "CPU_Z",
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_CHANGE",
    ]

    result[
        "VALID_BASE_MODEL_ROW"
    ] = (
        result[
            base_variables
        ]
        .notna()
        .all(
            axis=1
        )
    ).astype(
        int
    )

    for channel in ALL_CHANNELS:

        for quality in [
            "MODERATE",
            "MAIN",
            "STRICT",
        ]:

            channel_column = (
                f"{channel}_{quality}"
            )

            interaction_column = (
                f"CPU_Z_X_{channel_column}"
            )

            if (
                channel_column
                not in result.columns
                or interaction_column
                not in result.columns
            ):
                continue

            result[
                f"VALID_{channel}_{quality}_ROW"
            ] = (
                (
                    result[
                        "VALID_BASE_MODEL_ROW"
                    ]
                    == 1
                )
                & result[
                    channel_column
                ].notna()
                & result[
                    interaction_column
                ].notna()
            ).astype(
                int
            )

    # PCA yalnızca mevcut olduğu ETF'lerde çalışır.
    if "PCA_COMPONENT_1" in result.columns:

        result[
            "VALID_PCA_MAIN_ROW"
        ] = (
            (
                result[
                    "VALID_BASE_MODEL_ROW"
                ]
                == 1
            )
            & result[
                "PCA_COMPONENT_1"
            ].notna()
            & result[
                "CPU_Z_X_PCA_COMPONENT_1"
            ].notna()
        ).astype(
            int
        )

    else:

        result[
            "VALID_PCA_MAIN_ROW"
        ] = 0

    # RQ3 stres örneklemleri
    result[
        "VALID_HIGH_CPU_ROW"
    ] = (
        (
            result[
                "VALID_BASE_MODEL_ROW"
            ]
            == 1
        )
        & (
            result[
                "HIGH_CPU_REGIME"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "VALID_CPU_VIX_STRESS_ROW"
    ] = (
        (
            result[
                "VALID_BASE_MODEL_ROW"
            ]
            == 1
        )
        & (
            result[
                "CPU_AND_VIX_STRESS"
            ]
            == 1
        )
    ).astype(
        int
    )

    # RQ4 ileri getiri örneklemleri
    for horizon in [
        1,
        3,
        6,
        12,
    ]:

        outcome = (
            f"ETF_RETURN_LEAD{horizon}"
            if horizon == 1
            else f"CUM_RETURN_LEAD{horizon}"
        )

        result[
            f"VALID_RECOVERY_{horizon}M_ROW"
        ] = (
            (
                result[
                    "VALID_BASE_MODEL_ROW"
                ]
                == 1
            )
            & result[
                outcome
            ].notna()
        ).astype(
            int
        )

    return result


# ============================================================
# 17. ETF COVERAGE RAPORU
# ============================================================

def build_etf_coverage(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    ETF bazında ekonometrik coverage raporu üretir.
    """

    rows = []

    for etf_id, group in panel.groupby(
        "ETF_ID",
        dropna=False,
    ):

        valid_returns = group.loc[
            group[
                "ETF_RETURN"
            ].notna()
        ]

        row = {
            "ETF_ID": etf_id,

            "ETF_NAME": (
                group[
                    "ETF_NAME"
                ].dropna().iloc[0]
                if group[
                    "ETF_NAME"
                ].notna().any()
                else pd.NA
            ),

            "ETF_TICKER": (
                group[
                    "ETF_TICKER"
                ].dropna().iloc[0]
                if (
                    "ETF_TICKER"
                    in group.columns
                    and group[
                        "ETF_TICKER"
                    ].notna().any()
                )
                else pd.NA
            ),

            "N_PANEL_ROWS": int(
                len(group)
            ),

            "N_MONTHLY_RETURNS": int(
                valid_returns.shape[0]
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

            "FINANCIAL_MATCH_WEIGHT": (
                safe_numeric(
                    group[
                        "FINANCIAL_MATCH_WEIGHT"
                    ]
                ).iloc[0]
            ),
        }

        for channel in ALL_CHANNELS:

            flag_column = (
                f"{channel}_VALID_MAIN"
            )

            if flag_column in group.columns:

                row[
                    f"{channel}_MAIN_VALID"
                ] = int(
                    safe_numeric(
                        group[
                            flag_column
                        ]
                    ).max()
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. KANAL ÖRNEKLEM ÖZETİ
# ============================================================

def build_channel_sample_summary(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her architecture kanalının moderate, main ve strict
    ekonometrik örneklemlerini özetler.
    """

    rows = []

    for channel in ALL_CHANNELS:

        for quality in [
            "MODERATE",
            "MAIN",
            "STRICT",
        ]:

            valid_column = (
                f"VALID_{channel}_{quality}_ROW"
            )

            channel_column = (
                f"{channel}_{quality}"
            )

            if (
                valid_column
                not in panel.columns
                or channel_column
                not in panel.columns
            ):
                continue

            subset = panel.loc[
                panel[
                    valid_column
                ]
                == 1
            ]

            rows.append(
                {
                    "CHANNEL": channel,
                    "QUALITY": quality,

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

                    "MEAN_CHANNEL_SCORE": float(
                        subset[
                            channel_column
                        ].mean()
                    ),

                    "STD_CHANNEL_SCORE": float(
                        subset[
                            channel_column
                        ].std(
                            ddof=1
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 19. AYLIK COVERAGE
# ============================================================

def build_monthly_coverage(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ay bazında panel kapsamını üretir.
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

            CPU=(
                "CPU",
                "first",
            ),

            CPU_Z=(
                "CPU_Z",
                "first",
            ),

            HIGH_CPU_REGIME=(
                "HIGH_CPU_REGIME",
                "first",
            ),

            EXTREME_CPU_REGIME=(
                "EXTREME_CPU_REGIME",
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

            CPU_AND_VIX_STRESS=(
                "CPU_AND_VIX_STRESS",
                "first",
            ),
        )
    )


# ============================================================
# 20. DESCRIPTIVE STATISTICS
# ============================================================

def build_descriptive_statistics(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nihai panel tanımlayıcı istatistiklerini üretir.
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
    ]

    variables.extend(
        [
            f"{channel}_MAIN"
            for channel in ALL_CHANNELS
            if f"{channel}_MAIN"
            in panel.columns
        ]
    )

    rows = []

    for variable in variables:

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
# 21. KORELASYON MATRİSİ
# ============================================================

def build_correlation_matrix(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    ETF-level architecture kesit korelasyonlarını üretir.
    """

    architecture_variables = [
        f"{channel}_MAIN"
        for channel in ALL_CHANNELS
        if f"{channel}_MAIN"
        in panel.columns
    ]

    cross_section = (
        panel[
            [
                "ETF_ID",
            ]
            + architecture_variables
        ]
        .drop_duplicates(
            subset=[
                "ETF_ID",
            ]
        )
    )

    correlation = (
        cross_section[
            architecture_variables
        ]
        .apply(
            safe_numeric
        )
        .corr(
            method="pearson",
            min_periods=5,
        )
    )

    correlation.index.name = (
        "VARIABLE"
    )

    return correlation.reset_index()


# ============================================================
# 22. VARIABLE DICTIONARY
# ============================================================

def build_variable_dictionary() -> pd.DataFrame:
    """
    Nihai ekonometrik panel değişken sözlüğünü oluşturur.
    """

    rows = [
        {
            "VARIABLE": "ETF_RETURN",
            "TYPE": "Dependent variable",
            "FORMULA": (
                "Monthly percentage change in adjusted month-end price"
            ),
            "THEORY": (
                "Conditional ETF pricing"
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
            "FORMULA": (
                "(CPU - sample mean) / sample standard deviation"
            ),
            "THEORY": (
                "Common external climate-policy uncertainty shock"
            ),
            "RESEARCH_QUESTION": (
                "RQ1–RQ4"
            ),
        },

        {
            "VARIABLE": "CPU_AND_VIX_STRESS",
            "TYPE": (
                "Joint stress-state indicator"
            ),
            "FORMULA": (
                "1 when both CPU and VIX are at or above "
                "their 75th percentiles"
            ),
            "THEORY": (
                "Market stress activates latent architecture exposure"
            ),
            "RESEARCH_QUESTION": (
                "RQ3"
            ),
        },

        {
            "VARIABLE": "CUM_RETURN_LEAD3",
            "TYPE": (
                "Forward recovery outcome"
            ),
            "FORMULA": (
                "Cumulative ETF return from t+1 through t+3"
            ),
            "THEORY": (
                "Post-shock recovery dynamics"
            ),
            "RESEARCH_QUESTION": (
                "RQ4"
            ),
        },

        {
            "VARIABLE": "CUM_RETURN_LEAD6",
            "TYPE": (
                "Forward recovery outcome"
            ),
            "FORMULA": (
                "Cumulative ETF return from t+1 through t+6"
            ),
            "THEORY": (
                "Medium-horizon recovery dynamics"
            ),
            "RESEARCH_QUESTION": (
                "RQ4"
            ),
        },

        {
            "VARIABLE": "CUM_RETURN_LEAD12",
            "TYPE": (
                "Forward recovery outcome"
            ),
            "FORMULA": (
                "Cumulative ETF return from t+1 through t+12"
            ),
            "THEORY": (
                "Longer-horizon recovery dynamics"
            ),
            "RESEARCH_QUESTION": (
                "RQ4"
            ),
        },

        {
            "VARIABLE": "FROZEN_ARCHITECTURE_FLAG",
            "TYPE": (
                "Research-design disclosure"
            ),
            "FORMULA": (
                "1 for all observations using the 2025Q4 "
                "holdings architecture snapshot"
            ),
            "THEORY": (
                "Current architecture used as persistent exposure proxy"
            ),
            "RESEARCH_QUESTION": (
                "Methodological disclosure"
            ),
        },
    ]

    for channel in ALL_CHANNELS:

        rows.append(
            {
                "VARIABLE": (
                    f"CPU_Z_X_{channel}_MAIN"
                ),
                "TYPE": (
                    "Conditional-pricing interaction"
                ),
                "FORMULA": (
                    f"CPU_Z × {channel}_MAIN"
                ),
                "THEORY": (
                    "Climate-policy uncertainty is transmitted through "
                    "embedded portfolio architecture"
                ),
                "RESEARCH_QUESTION": (
                    "RQ1 or RQ2"
                ),
            }
        )

        rows.append(
            {
                "VARIABLE": (
                    f"CPU_VIX_STRESS_X_{channel}_MAIN"
                ),
                "TYPE": (
                    "Stress-activation interaction"
                ),
                "FORMULA": (
                    f"CPU_AND_VIX_STRESS × {channel}_MAIN"
                ),
                "THEORY": (
                    "Market stress activates latent architecture exposure"
                ),
                "RESEARCH_QUESTION": (
                    "RQ3"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 23. VALIDATION
# ============================================================

def build_validation(
    panel: pd.DataFrame,
    channel_summary: pd.DataFrame,
    etf_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nihai ekonometrik panel validation kontrollerini yapar.
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

    frozen_values = (
        panel[
            "FROZEN_ARCHITECTURE_FLAG"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    rows = [
        {
            "CHECK": "FINAL_PANEL_ROWS",
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
            "CHECK": "FINAL_PANEL_ETFS",
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
            "CHECK": "FINAL_PANEL_MONTHS",
            "VALUE": panel[
                "DATE"
            ].nunique(),
            "PASS": int(
                panel[
                    "DATE"
                ].nunique()
                == 192
            ),
        },

        {
            "CHECK": (
                "DUPLICATE_ETF_MONTH_ROWS"
            ),
            "VALUE": duplicate_rows,
            "PASS": int(
                duplicate_rows
                == 0
            ),
        },

        {
            "CHECK": "CPU_MISSING_ROWS",
            "VALUE": int(
                panel[
                    "CPU"
                ].isna()
                .sum()
            ),
            "PASS": int(
                panel[
                    "CPU"
                ].isna()
                .sum()
                == 0
            ),
        },

        {
            "CHECK": (
                "MARKET_RETURN_MISSING_ROWS"
            ),
            "VALUE": int(
                panel[
                    "MARKET_RETURN"
                ].isna()
                .sum()
            ),
            "PASS": int(
                panel[
                    "MARKET_RETURN"
                ].isna()
                .sum()
                == 0
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
                "CHANNEL_SUMMARY_ROWS"
            ),
            "VALUE": len(
                channel_summary
            ),
            "PASS": int(
                len(
                    channel_summary
                )
                > 0
            ),
        },

        {
            "CHECK": (
                "FROZEN_ARCHITECTURE_DISCLOSED"
            ),
            "VALUE": str(
                frozen_values
            ),
            "PASS": int(
                frozen_values
                == [1]
            ),
        },

        {
            "CHECK": (
                "HIGH_CPU_MONTHS_AVAILABLE"
            ),
            "VALUE": int(
                panel.loc[
                    panel[
                        "HIGH_CPU_REGIME"
                    ]
                    == 1,
                    "DATE",
                ].nunique()
            ),
            "PASS": int(
                panel.loc[
                    panel[
                        "HIGH_CPU_REGIME"
                    ]
                    == 1,
                    "DATE",
                ].nunique()
                > 0
            ),
        },

        {
            "CHECK": (
                "CPU_VIX_STRESS_MONTHS_AVAILABLE"
            ),
            "VALUE": int(
                panel.loc[
                    panel[
                        "CPU_AND_VIX_STRESS"
                    ]
                    == 1,
                    "DATE",
                ].nunique()
            ),
            "PASS": int(
                panel.loc[
                    panel[
                        "CPU_AND_VIX_STRESS"
                    ]
                    == 1,
                    "DATE",
                ].nunique()
                > 0
            ),
        },

        {
            "CHECK": (
                "THREE_MONTH_RECOVERY_AVAILABLE"
            ),
            "VALUE": int(
                panel[
                    "CUM_RETURN_LEAD3"
                ]
                .notna()
                .sum()
            ),
            "PASS": int(
                panel[
                    "CUM_RETURN_LEAD3"
                ]
                .notna()
                .sum()
                > 0
            ),
        },

        {
            "CHECK": (
                "SIX_MONTH_RECOVERY_AVAILABLE"
            ),
            "VALUE": int(
                panel[
                    "CUM_RETURN_LEAD6"
                ]
                .notna()
                .sum()
            ),
            "PASS": int(
                panel[
                    "CUM_RETURN_LEAD6"
                ]
                .notna()
                .sum()
                > 0
            ),
        },

        {
            "CHECK": (
                "TWELVE_MONTH_RECOVERY_AVAILABLE"
            ),
            "VALUE": int(
                panel[
                    "CUM_RETURN_LEAD12"
                ]
                .notna()
                .sum()
            ),
            "PASS": int(
                panel[
                    "CUM_RETURN_LEAD12"
                ]
                .notna()
                .sum()
                > 0
            ),
        },
    ]

    for channel in PRIMARY_CHANNELS:

        valid_column = (
            f"VALID_{channel}_MAIN_ROW"
        )

        rows.append(
            {
                "CHECK": (
                    f"{channel}_MAIN_VALID_ROWS"
                ),
                "VALUE": int(
                    panel[
                        valid_column
                    ].sum()
                ),
                "PASS": int(
                    panel[
                        valid_column
                    ].sum()
                    > 0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 24. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 90)
    print("25 - FINAL ECONOMETRIC PANEL")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Girdiler
    # --------------------------------------------------------

    print(
        "\n1/10 - ETF aylık getirileri okunuyor..."
    )

    returns = prepare_returns(
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

    controls = prepare_controls(
        MARKET_CONTROL_FILE
    )

    print(
        f"Kontrol ay sayısı: "
        f"{controls['DATE'].nunique():,}"
    )

    print(
        "\n4/10 - Nihai architecture kanalları okunuyor..."
    )

    architecture = prepare_architecture(
        ARCHITECTURE_FILE
    )

    print(
        f"Architecture ETF sayısı: "
        f"{architecture['ETF_ID'].nunique():,}"
    )

    # --------------------------------------------------------
    # 2. Merge
    # --------------------------------------------------------

    print(
        "\n5/10 - ETF getirileri architecture ile birleştiriliyor..."
    )

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
                panel[
                    "ETF_NAME_ARCH"
                ]
            )
        )

    if "ETF_TICKER_RETURN" in panel.columns:

        panel["ETF_TICKER"] = (
            panel[
                "ETF_TICKER_RETURN"
            ]
        )

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
    # 3. Panel değişkenleri
    # --------------------------------------------------------

    print(
        "\n7/10 - Zaman, stres ve recovery değişkenleri oluşturuluyor..."
    )

    panel = add_time_and_return_variables(
        panel
    )

    panel = add_stress_variables(
        panel
    )

    print(
        "\n8/10 - CPU × architecture etkileşimleri oluşturuluyor..."
    )

    panel = add_channel_interactions(
        panel
    )

    panel = add_channel_sample_flags(
        panel
    )

    # --------------------------------------------------------
    # 4. Raporlar
    # --------------------------------------------------------

    print(
        "\n9/10 - Coverage ve diagnostics raporları hazırlanıyor..."
    )

    etf_coverage = build_etf_coverage(
        panel
    )

    channel_summary = (
        build_channel_sample_summary(
            panel
        )
    )

    monthly_coverage = (
        build_monthly_coverage(
            panel
        )
    )

    descriptives = (
        build_descriptive_statistics(
            panel
        )
    )

    correlations = (
        build_correlation_matrix(
            panel
        )
    )

    variable_dictionary = (
        build_variable_dictionary()
    )

    validation = build_validation(
        panel=panel,
        channel_summary=channel_summary,
        etf_coverage=etf_coverage,
    )

    # --------------------------------------------------------
    # 5. Özel paneller
    # --------------------------------------------------------

    primary_valid_columns = [
        f"VALID_{channel}_MAIN_ROW"
        for channel in PRIMARY_CHANNELS
    ]

    primary_channel_panel = panel.loc[
        panel[
            primary_valid_columns
        ]
        .sum(
            axis=1
        )
        > 0
    ].copy()

    financial_risk_panel = panel.loc[
        panel[
            "VALID_FINANCIAL_ARCHITECTURE_RISK_FINAL_MAIN_ROW"
        ]
        == 1
    ].copy()

    pca_panel = panel.loc[
        panel[
            "VALID_PCA_MAIN_ROW"
        ]
        == 1
    ].copy()

    # --------------------------------------------------------
    # 6. Kaydet
    # --------------------------------------------------------

    print(
        "\n10/10 - Nihai panel ve raporlar kaydediliyor..."
    )

    panel.to_csv(
        FINAL_PANEL_FILE,
        index=False,
    )

    panel.to_parquet(
        FINAL_PANEL_PARQUET_FILE,
        index=False,
    )

    primary_channel_panel.to_csv(
        PRIMARY_CHANNEL_PANEL_FILE,
        index=False,
    )

    financial_risk_panel.to_csv(
        FINANCIAL_RISK_PANEL_FILE,
        index=False,
    )

    pca_panel.to_csv(
        PCA_PANEL_FILE,
        index=False,
    )

    etf_coverage.to_csv(
        ETF_COVERAGE_FILE,
        index=False,
    )

    channel_summary.to_csv(
        CHANNEL_SAMPLE_SUMMARY_FILE,
        index=False,
    )

    monthly_coverage.to_csv(
        MONTHLY_COVERAGE_FILE,
        index=False,
    )

    descriptives.to_csv(
        DESCRIPTIVE_FILE,
        index=False,
    )

    correlations.to_csv(
        CORRELATION_FILE,
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
        "\nFINAL ECONOMETRIC PANEL HAZIR"
    )

    print("=" * 90)

    print(
        "\nGenel panel özeti:"
    )

    general_summary = pd.DataFrame(
        [
            {
                "METRIC": "PANEL_ROWS",
                "VALUE": len(
                    panel
                ),
            },
            {
                "METRIC": "PANEL_ETFS",
                "VALUE": panel[
                    "ETF_ID"
                ].nunique(),
            },
            {
                "METRIC": "PANEL_MONTHS",
                "VALUE": panel[
                    "DATE"
                ].nunique(),
            },
            {
                "METRIC": (
                    "VALID_BASE_MODEL_ROWS"
                ),
                "VALUE": int(
                    panel[
                        "VALID_BASE_MODEL_ROW"
                    ].sum()
                ),
            },
            {
                "METRIC": (
                    "PRIMARY_CHANNEL_PANEL_ROWS"
                ),
                "VALUE": len(
                    primary_channel_panel
                ),
            },
            {
                "METRIC": (
                    "PRIMARY_CHANNEL_PANEL_ETFS"
                ),
                "VALUE": primary_channel_panel[
                    "ETF_ID"
                ].nunique(),
            },
            {
                "METRIC": (
                    "FINANCIAL_RISK_PANEL_ROWS"
                ),
                "VALUE": len(
                    financial_risk_panel
                ),
            },
            {
                "METRIC": (
                    "FINANCIAL_RISK_PANEL_ETFS"
                ),
                "VALUE": financial_risk_panel[
                    "ETF_ID"
                ].nunique(),
            },
            {
                "METRIC": (
                    "PCA_PANEL_ROWS"
                ),
                "VALUE": len(
                    pca_panel
                ),
            },
            {
                "METRIC": (
                    "PCA_PANEL_ETFS"
                ),
                "VALUE": pca_panel[
                    "ETF_ID"
                ].nunique(),
            },
        ]
    )

    print(
        general_summary.to_string(
            index=False
        )
    )

    print(
        "\nMain-quality kanal örneklem özeti:"
    )

    print(
        channel_summary.loc[
            channel_summary[
                "QUALITY"
            ]
            == "MAIN",
            [
                "CHANNEL",
                "N_ROWS",
                "N_ETFS",
                "N_MONTHS",
                "START_DATE",
                "END_DATE",
                "MEAN_ETF_RETURN",
                "STD_ETF_RETURN",
            ],
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nStres ve recovery coverage:"
    )

    stress_recovery_summary = pd.DataFrame(
        [
            {
                "METRIC": "HIGH_CPU_MONTHS",
                "VALUE": panel.loc[
                    panel[
                        "HIGH_CPU_REGIME"
                    ]
                    == 1,
                    "DATE",
                ].nunique(),
            },
            {
                "METRIC": (
                    "CPU_VIX_STRESS_MONTHS"
                ),
                "VALUE": panel.loc[
                    panel[
                        "CPU_AND_VIX_STRESS"
                    ]
                    == 1,
                    "DATE",
                ].nunique(),
            },
            {
                "METRIC": (
                    "RECOVERY_1M_ROWS"
                ),
                "VALUE": int(
                    panel[
                        "VALID_RECOVERY_1M_ROW"
                    ].sum()
                ),
            },
            {
                "METRIC": (
                    "RECOVERY_3M_ROWS"
                ),
                "VALUE": int(
                    panel[
                        "VALID_RECOVERY_3M_ROW"
                    ].sum()
                ),
            },
            {
                "METRIC": (
                    "RECOVERY_6M_ROWS"
                ),
                "VALUE": int(
                    panel[
                        "VALID_RECOVERY_6M_ROW"
                    ].sum()
                ),
            },
            {
                "METRIC": (
                    "RECOVERY_12M_ROWS"
                ),
                "VALUE": int(
                    panel[
                        "VALID_RECOVERY_12M_ROW"
                    ].sum()
                ),
            },
        ]
    )

    print(
        stress_recovery_summary.to_string(
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
        "\nMETODOLOJİK NOT:"
    )

    print(
        "Architecture değişkenleri 2025Q4 holdings kesitine "
        "dayanmaktadır. Nihai panel frozen-architecture exposure "
        "tasarımıdır."
    )

    print(
        "Ana analizde dört primary channel ayrı ayrı tahmin "
        "edilecektir. Financial Architecture Risk, Extended Risk "
        "ve PCA yalnızca robustness amacıyla kullanılacaktır."
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        FINAL_PANEL_FILE
    )

    print(
        PRIMARY_CHANNEL_PANEL_FILE
    )

    print(
        FINANCIAL_RISK_PANEL_FILE
    )

    print(
        PCA_PANEL_FILE
    )

    print(
        ETF_COVERAGE_FILE
    )

    print(
        CHANNEL_SAMPLE_SUMMARY_FILE
    )

    print(
        MONTHLY_COVERAGE_FILE
    )

    print(
        DESCRIPTIVE_FILE
    )

    print(
        CORRELATION_FILE
    )

    print(
        VALIDATION_FILE
    )

    print(
        VARIABLE_DICTIONARY_FILE
    )


if __name__ == "__main__":
    main()