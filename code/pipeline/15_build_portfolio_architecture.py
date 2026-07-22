from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 15_build_portfolio_architecture.py
#
# Amaç
# ----
# 2025Q4 sürdürülebilir ETF holdings verisini, FY2024 firma
# karakteristikleriyle CUSIP üzerinden birleştirerek holdings-based
# portfolio architecture değişkenlerini üretmek.
#
# Üretilen ana değişkenler
# -----------------------
# Portfolio-weighted:
# - ROA
# - Leverage
# - Cash ratio
# - CapEx intensity
# - R&D intensity
# - Revenue growth
# - External finance dependence
# - Firm size
#
# Portfolio structure:
# - HHI
# - Top 5 weight
# - Top 10 weight
# - Maximum holding weight
# - Effective number of holdings
#
# Measurement quality:
# - Financial match weight
# - Characteristic-specific coverage
# - Characteristic-specific dispersion
# - Characteristic-specific sample flags
#
# Not:
# Architecture stability yalnızca tek holdings kesiti bulunduğu için
# bu aşamada hesaplanamaz.
# ============================================================


# ============================================================
# 1. PROJE YOLLARI
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\User\Desktop\CPU_Project"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. GİRDİ DOSYALARI
# ============================================================

HOLDINGS_FILE = (
    OUTPUT_DIR
    / "2025q4_include_equity_corporate_holdings.parquet"
)

FIRM_FILE = (
    OUTPUT_DIR
    / "firm_characteristics_annual_identifiers.parquet"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

MERGED_HOLDINGS_FILE = (
    OUTPUT_DIR
    / "15_holdings_firm_characteristics_merged.parquet"
)

MERGED_HOLDINGS_CSV_FILE = (
    OUTPUT_DIR
    / "15_holdings_firm_characteristics_merged.csv"
)

PORTFOLIO_ARCHITECTURE_FILE = (
    OUTPUT_DIR
    / "15_portfolio_architecture_panel.csv"
)

PORTFOLIO_ARCHITECTURE_PARQUET_FILE = (
    OUTPUT_DIR
    / "15_portfolio_architecture_panel.parquet"
)

CHARACTERISTICS_LONG_FILE = (
    OUTPUT_DIR
    / "15_portfolio_characteristics_long.csv"
)

MERGE_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "15_holdings_merge_diagnostics.csv"
)

ETF_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "15_etf_merge_diagnostics.csv"
)

UNMATCHED_HOLDINGS_FILE = (
    OUTPUT_DIR
    / "15_unmatched_holdings.csv"
)

CHARACTERISTIC_COVERAGE_REPORT_FILE = (
    OUTPUT_DIR
    / "15_characteristic_coverage_report.csv"
)

CONCENTRATION_REPORT_FILE = (
    OUTPUT_DIR
    / "15_portfolio_concentration_report.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "15_portfolio_architecture_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "15_portfolio_architecture_variable_dictionary.csv"
)


# ============================================================
# 4. METODOLOJİK AYARLAR
# ============================================================

CHARACTERISTIC_FISCAL_YEAR = 2024

FINANCIAL_MATCH_THRESHOLD_70 = 0.70
FINANCIAL_MATCH_THRESHOLD_80 = 0.80
FINANCIAL_MATCH_THRESHOLD_90 = 0.90

WEIGHT_TOLERANCE = 1e-8

MIN_HOLDINGS_FOR_DISPERSION = 2

ALLOW_NEGATIVE_WEIGHTS = False


# ============================================================
# 5. PORTFÖY KARAKTERİSTİK HARİTASI
# ============================================================

CHARACTERISTIC_MAP = {
    "PW_ROA": {
        "source": "ROA_W",
        "coverage": "COV_ROA",
        "dispersion": "DISP_ROA",
        "theory": (
            "Profitability and operating resilience"
        ),
        "research_question": (
            "RQ2, RQ3, RQ4"
        ),
        "expected_cpu_interaction": (
            "Positive or less negative"
        ),
    },

    "PW_LEVERAGE": {
        "source": "LEVERAGE_W",
        "coverage": "COV_LEVERAGE",
        "dispersion": "DISP_LEVERAGE",
        "theory": (
            "Financing constraints and fixed obligations"
        ),
        "research_question": (
            "RQ2, RQ3"
        ),
        "expected_cpu_interaction": (
            "Negative"
        ),
    },

    "PW_CASH_RATIO": {
        "source": "CASH_RATIO_W",
        "coverage": "COV_CASH_RATIO",
        "dispersion": "DISP_CASH_RATIO",
        "theory": (
            "Financial flexibility and liquidity buffers"
        ),
        "research_question": (
            "RQ2, RQ3, RQ4"
        ),
        "expected_cpu_interaction": (
            "Positive or less negative"
        ),
    },

    "PW_CAPEX_INTENSITY": {
        "source": "CAPEX_INTENSITY_W",
        "coverage": "COV_CAPEX_INTENSITY",
        "dispersion": "DISP_CAPEX_INTENSITY",
        "theory": (
            "Real options and irreversible investment"
        ),
        "research_question": (
            "RQ2, RQ4"
        ),
        "expected_cpu_interaction": (
            "Conditional"
        ),
    },

    "PW_RD_INTENSITY": {
        "source": "RD_INTENSITY_W",
        "coverage": "COV_RD_INTENSITY",
        "dispersion": "DISP_RD_INTENSITY",
        "theory": (
            "Growth options and equity duration"
        ),
        "research_question": (
            "RQ2, RQ4"
        ),
        "expected_cpu_interaction": (
            "Potentially negative"
        ),
    },

    "PW_REVENUE_GROWTH": {
        "source": "REVENUE_GROWTH_W",
        "coverage": "COV_REVENUE_GROWTH",
        "dispersion": "DISP_REVENUE_GROWTH",
        "theory": (
            "Growth exposure and operating expansion"
        ),
        "research_question": (
            "RQ2"
        ),
        "expected_cpu_interaction": (
            "Conditional"
        ),
    },

    "PW_EXTERNAL_FINANCE_DEPENDENCE": {
        "source": "EXTERNAL_FINANCE_DEPENDENCE_W",
        "coverage": "COV_EXTERNAL_FINANCE_DEPENDENCE",
        "dispersion": "DISP_EXTERNAL_FINANCE_DEPENDENCE",
        "theory": (
            "Dependence on external capital markets"
        ),
        "research_question": (
            "RQ2, RQ3"
        ),
        "expected_cpu_interaction": (
            "Negative"
        ),
    },

    "PW_LOG_ASSETS": {
        "source": "LOG_ASSETS_W",
        "coverage": "COV_LOG_ASSETS",
        "dispersion": "DISP_LOG_ASSETS",
        "theory": (
            "Firm size, information environment and financing access"
        ),
        "research_question": (
            "RQ2, RQ3"
        ),
        "expected_cpu_interaction": (
            "Positive or less negative"
        ),
    },
}


# ============================================================
# 6. GENEL YARDIMCI FONKSİYONLAR
# ============================================================

def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sütun adlarını uppercase snake-case biçimine getirir.
    """

    result = df.copy()

    result.columns = (
        result.columns
        .astype(str)
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

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(
            path
        )

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            low_memory=False,
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


def clean_text_identifier(
    series: pd.Series,
) -> pd.Series:
    """
    CUSIP, ISIN ve ticker değerlerini temizler.

    Yalnızca sıfırlardan oluşan identifier değerleri eksik kabul edilir.
    """

    result = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace(
            r"[^A-Z0-9]",
            "",
            regex=True,
        )
    )

    invalid_values = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "NA",
        "<NA>",
    }

    result = result.mask(
        result.isin(
            invalid_values
        ),
        pd.NA,
    )

    all_zero_flag = (
        result.notna()
        & result.str.fullmatch(
            r"0+"
        )
    )

    result = result.mask(
        all_zero_flag,
        pd.NA,
    )

    return result


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


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Eksik olmayan firma gözlemleri için ağırlıklı ortalama hesaplar.
    """

    values = safe_numeric(
        values
    )

    weights = safe_numeric(
        weights
    )

    valid = (
        values.notna()
        & weights.notna()
        & (weights >= 0)
    )

    if not valid.any():
        return np.nan

    valid_values = values.loc[
        valid
    ]

    valid_weights = weights.loc[
        valid
    ]

    weight_sum = valid_weights.sum()

    if weight_sum <= 0:
        return np.nan

    normalized_weights = (
        valid_weights
        / weight_sum
    )

    return float(
        np.sum(
            normalized_weights
            * valid_values
        )
    )


def weighted_dispersion(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Ağırlıklı portföy içi standart sapma hesaplar.
    """

    values = safe_numeric(
        values
    )

    weights = safe_numeric(
        weights
    )

    valid = (
        values.notna()
        & weights.notna()
        & (weights >= 0)
    )

    if int(
        valid.sum()
    ) < MIN_HOLDINGS_FOR_DISPERSION:
        return np.nan

    valid_values = (
        values.loc[
            valid
        ]
        .to_numpy(
            dtype=float
        )
    )

    valid_weights = (
        weights.loc[
            valid
        ]
        .to_numpy(
            dtype=float
        )
    )

    weight_sum = (
        valid_weights.sum()
    )

    if weight_sum <= 0:
        return np.nan

    normalized_weights = (
        valid_weights
        / weight_sum
    )

    weighted_average = np.sum(
        normalized_weights
        * valid_values
    )

    denominator = (
        1.0
        - np.sum(
            normalized_weights ** 2
        )
    )

    if denominator <= 0:
        return 0.0

    weighted_variance = (
        np.sum(
            normalized_weights
            * (
                valid_values
                - weighted_average
            ) ** 2
        )
        / denominator
    )

    return float(
        np.sqrt(
            max(
                weighted_variance,
                0.0,
            )
        )
    )


# ============================================================
# 7. HOLDINGS STANDARDİZASYONU
# ============================================================

def standardize_holdings(
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Holdings verisini temizler ve ortak kimlikleri oluşturur.
    """

    holdings = normalize_columns(
        holdings
    )

    required_columns = [
        "ACCESSION_NUMBER",
        "SERIES_ID",
        "SERIES_NAME",
        "ISSUER_NAME",
        "ISSUER_CUSIP",
    ]

    require_columns(
        holdings,
        required_columns,
        "HOLDINGS",
    )

    holdings = holdings.copy()

    holdings[
        "SOURCE_ROW_ID"
    ] = np.arange(
        len(holdings),
        dtype="int64",
    )

    holdings[
        "ETF_ID"
    ] = (
        holdings[
            "SERIES_ID"
        ]
        .astype("string")
        .str.strip()
    )

    holdings[
        "ETF_NAME"
    ] = (
        holdings[
            "SERIES_NAME"
        ]
        .astype("string")
        .str.strip()
    )

    holdings[
        "ACCESSION_NUMBER"
    ] = (
        holdings[
            "ACCESSION_NUMBER"
        ]
        .astype("string")
        .str.strip()
    )

    holdings[
        "ISSUER_NAME"
    ] = (
        holdings[
            "ISSUER_NAME"
        ]
        .astype("string")
        .str.strip()
    )

    if "HOLDING_ID" in holdings.columns:

        holdings[
            "HOLDING_ID_CLEAN"
        ] = (
            holdings[
                "HOLDING_ID"
            ]
            .astype("string")
            .str.strip()
        )

        holdings[
            "HOLDING_ID_CLEAN"
        ] = holdings[
            "HOLDING_ID_CLEAN"
        ].mask(
            holdings[
                "HOLDING_ID_CLEAN"
            ].isin(
                [
                    "",
                    "nan",
                    "NaN",
                    "<NA>",
                ]
            ),
            pd.NA,
        )

    else:

        holdings[
            "HOLDING_ID_CLEAN"
        ] = pd.NA

    if (
        "IDENTIFIER_ISIN"
        in holdings.columns
    ):

        holdings[
            "HOLDING_ISIN"
        ] = clean_text_identifier(
            holdings[
                "IDENTIFIER_ISIN"
            ]
        )

    else:

        holdings[
            "HOLDING_ISIN"
        ] = pd.NA

    if (
        "IDENTIFIER_TICKER"
        in holdings.columns
    ):

        holdings[
            "HOLDING_TICKER"
        ] = clean_text_identifier(
            holdings[
                "IDENTIFIER_TICKER"
            ]
        )

    else:

        holdings[
            "HOLDING_TICKER"
        ] = pd.NA

    holdings[
        "HOLDING_CUSIP"
    ] = clean_text_identifier(
        holdings[
            "ISSUER_CUSIP"
        ]
    )

    holdings[
        "HOLDING_CUSIP_LENGTH"
    ] = (
        holdings[
            "HOLDING_CUSIP"
        ]
        .str.len()
    )

    holdings[
        "HOLDING_CUSIP9"
    ] = (
        holdings[
            "HOLDING_CUSIP"
        ]
        .where(
            holdings[
                "HOLDING_CUSIP_LENGTH"
            ] >= 9
        )
        .str[:9]
    )

    holdings[
        "HOLDING_CUSIP8"
    ] = (
        holdings[
            "HOLDING_CUSIP"
        ]
        .where(
            holdings[
                "HOLDING_CUSIP_LENGTH"
            ] >= 8
        )
        .str[:8]
    )

    # --------------------------------------------------------
    # Güvenli aggregation key
    #
    # CUSIP9
    # -> ISIN
    # -> Holding ID
    # -> Kaynak satır kimliği
    # --------------------------------------------------------

    holdings[
        "HOLDING_AGGREGATION_KEY"
    ] = pd.NA

    cusip_mask = (
        holdings[
            "HOLDING_CUSIP9"
        ].notna()
    )

    holdings.loc[
        cusip_mask,
        "HOLDING_AGGREGATION_KEY",
    ] = (
        "CUSIP9:"
        + holdings.loc[
            cusip_mask,
            "HOLDING_CUSIP9",
        ]
    )

    isin_mask = (
        holdings[
            "HOLDING_AGGREGATION_KEY"
        ].isna()
        & holdings[
            "HOLDING_ISIN"
        ].notna()
    )

    holdings.loc[
        isin_mask,
        "HOLDING_AGGREGATION_KEY",
    ] = (
        "ISIN:"
        + holdings.loc[
            isin_mask,
            "HOLDING_ISIN",
        ]
    )

    holding_id_mask = (
        holdings[
            "HOLDING_AGGREGATION_KEY"
        ].isna()
        & holdings[
            "HOLDING_ID_CLEAN"
        ].notna()
    )

    holdings.loc[
        holding_id_mask,
        "HOLDING_AGGREGATION_KEY",
    ] = (
        "HOLDING_ID:"
        + holdings.loc[
            holding_id_mask,
            "HOLDING_ID_CLEAN",
        ].astype(
            "string"
        )
    )

    row_mask = (
        holdings[
            "HOLDING_AGGREGATION_KEY"
        ].isna()
    )

    holdings.loc[
        row_mask,
        "HOLDING_AGGREGATION_KEY",
    ] = (
        "ROW:"
        + holdings.loc[
            row_mask,
            "SOURCE_ROW_ID",
        ].astype(
            str
        )
    )

    # Percentage sütunu
    percentage_column = None

    for candidate in [
        "PERCENTAGE_NUMERIC",
        "PERCENTAGE",
    ]:

        if candidate in holdings.columns:

            percentage_column = candidate

            break

    if percentage_column is None:

        holdings[
            "REPORTED_PERCENTAGE"
        ] = np.nan

    else:

        holdings[
            "REPORTED_PERCENTAGE"
        ] = safe_numeric(
            holdings[
                percentage_column
            ]
        )

    # Holding value sütunu
    value_column = None

    for candidate in [
        "CURRENCY_VALUE_NUMERIC",
        "CURRENCY_VALUE",
        "FAIR_VALUE",
        "MARKET_VALUE",
    ]:

        if candidate in holdings.columns:

            value_column = candidate

            break

    if value_column is None:

        holdings[
            "HOLDING_VALUE"
        ] = np.nan

    else:

        holdings[
            "HOLDING_VALUE"
        ] = safe_numeric(
            holdings[
                value_column
            ]
        )

    if not ALLOW_NEGATIVE_WEIGHTS:

        holdings.loc[
            holdings[
                "REPORTED_PERCENTAGE"
            ] < 0,
            "REPORTED_PERCENTAGE",
        ] = np.nan

        holdings.loc[
            holdings[
                "HOLDING_VALUE"
            ] < 0,
            "HOLDING_VALUE",
        ] = np.nan

    return holdings


# ============================================================
# 8. HOLDING AĞIRLIKLARI
# ============================================================

def construct_holding_weights(
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Holding ağırlıklarını oluşturur.

    Öncelik:
    1. SEC reported percentage
    2. Holding value
    """

    holdings = holdings.copy()

    portfolio_keys = [
        "ACCESSION_NUMBER",
        "ETF_ID",
        "ETF_NAME",
    ]

    holdings[
        "RAW_WEIGHT_FROM_PERCENTAGE"
    ] = (
        holdings[
            "REPORTED_PERCENTAGE"
        ]
        / 100.0
    )

    total_holding_value = (
        holdings.groupby(
            portfolio_keys,
            dropna=False,
        )[
            "HOLDING_VALUE"
        ]
        .transform(
            "sum"
        )
    )

    holdings[
        "RAW_WEIGHT_FROM_VALUE"
    ] = np.where(
        total_holding_value > 0,
        holdings[
            "HOLDING_VALUE"
        ]
        / total_holding_value,
        np.nan,
    )

    holdings[
        "RAW_PORTFOLIO_WEIGHT"
    ] = holdings[
        "RAW_WEIGHT_FROM_PERCENTAGE"
    ].where(
        holdings[
            "RAW_WEIGHT_FROM_PERCENTAGE"
        ].notna(),
        holdings[
            "RAW_WEIGHT_FROM_VALUE"
        ],
    )

    holdings[
        "WEIGHT_SOURCE"
    ] = np.select(
        [
            holdings[
                "RAW_WEIGHT_FROM_PERCENTAGE"
            ].notna(),

            holdings[
                "RAW_WEIGHT_FROM_VALUE"
            ].notna(),
        ],
        [
            "REPORTED_PERCENTAGE",
            "HOLDING_VALUE",
        ],
        default="MISSING",
    )

    holdings[
        "VALID_WEIGHT_FLAG"
    ] = (
        holdings[
            "RAW_PORTFOLIO_WEIGHT"
        ].notna()
        & (
            holdings[
                "RAW_PORTFOLIO_WEIGHT"
            ] >= 0
        )
    )

    holdings[
        "VALID_RAW_WEIGHT"
    ] = holdings[
        "RAW_PORTFOLIO_WEIGHT"
    ].where(
        holdings[
            "VALID_WEIGHT_FLAG"
        ]
    )

    raw_weight_sum = (
        holdings.groupby(
            portfolio_keys,
            dropna=False,
        )[
            "VALID_RAW_WEIGHT"
        ]
        .transform(
            "sum"
        )
    )

    holdings[
        "EQUITY_NORMALIZED_WEIGHT"
    ] = np.where(
        holdings[
            "VALID_WEIGHT_FLAG"
        ]
        & (
            raw_weight_sum > 0
        ),
        holdings[
            "RAW_PORTFOLIO_WEIGHT"
        ]
        / raw_weight_sum,
        np.nan,
    )

    return holdings


# ============================================================
# 9. DUPLICATE HOLDINGS TOPLULAŞTIRMA
# ============================================================

def aggregate_duplicate_holdings(
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aynı ekonomik holdingi ETF içinde tek satırda toplulaştırır.
    """

    holdings = holdings.copy()

    group_columns = [
        "ACCESSION_NUMBER",
        "ETF_ID",
        "ETF_NAME",
        "HOLDING_AGGREGATION_KEY",
    ]

    aggregated = (
        holdings.groupby(
            group_columns,
            dropna=False,
            as_index=False,
        )
        .agg(
            ISSUER_NAME=(
                "ISSUER_NAME",
                "first",
            ),

            HOLDING_CUSIP9=(
                "HOLDING_CUSIP9",
                "first",
            ),

            HOLDING_CUSIP8=(
                "HOLDING_CUSIP8",
                "first",
            ),

            HOLDING_ISIN=(
                "HOLDING_ISIN",
                "first",
            ),

            HOLDING_TICKER=(
                "HOLDING_TICKER",
                "first",
            ),

            RAW_PORTFOLIO_WEIGHT=(
                "RAW_PORTFOLIO_WEIGHT",
                "sum",
            ),

            EQUITY_NORMALIZED_WEIGHT=(
                "EQUITY_NORMALIZED_WEIGHT",
                "sum",
            ),

            HOLDING_VALUE=(
                "HOLDING_VALUE",
                "sum",
            ),

            SOURCE_HOLDING_ROWS=(
                "SOURCE_ROW_ID",
                "size",
            ),
        )
    )

    portfolio_keys = [
        "ACCESSION_NUMBER",
        "ETF_ID",
        "ETF_NAME",
    ]

    total_weight = (
        aggregated.groupby(
            portfolio_keys,
            dropna=False,
        )[
            "EQUITY_NORMALIZED_WEIGHT"
        ]
        .transform(
            "sum"
        )
    )

    aggregated[
        "EQUITY_NORMALIZED_WEIGHT"
    ] = np.where(
        total_weight > 0,
        aggregated[
            "EQUITY_NORMALIZED_WEIGHT"
        ]
        / total_weight,
        np.nan,
    )

    return aggregated


# ============================================================
# 10. FİRMA KARAKTERİSTİKLERİ
# ============================================================

def prepare_firm_characteristics(
    firm: pd.DataFrame,
) -> pd.DataFrame:
    """
    FY2024 firma karakteristiklerini hazırlar.
    """

    firm = normalize_columns(
        firm
    )

    required_columns = [
        "CIK10",
        "ENTITY_NAME",
        "FISCAL_YEAR",
        "CUSIP9",
        "CUSIP8",
    ]

    source_variables = [
        settings[
            "source"
        ]
        for settings
        in CHARACTERISTIC_MAP.values()
    ]

    require_columns(
        firm,
        required_columns
        + source_variables,
        "FIRM_CHARACTERISTICS",
    )

    firm = firm.copy()

    firm[
        "FISCAL_YEAR"
    ] = safe_numeric(
        firm[
            "FISCAL_YEAR"
        ]
    )

    firm = firm.loc[
        firm[
            "FISCAL_YEAR"
        ]
        == CHARACTERISTIC_FISCAL_YEAR
    ].copy()

    if firm.empty:

        raise RuntimeError(
            f"FY{CHARACTERISTIC_FISCAL_YEAR} "
            "firma gözlemi bulunamadı."
        )

    firm[
        "CUSIP9"
    ] = clean_text_identifier(
        firm[
            "CUSIP9"
        ]
    )

    firm[
        "CUSIP8"
    ] = clean_text_identifier(
        firm[
            "CUSIP8"
        ]
    )

    if "ISIN12" in firm.columns:

        firm[
            "ISIN12"
        ] = clean_text_identifier(
            firm[
                "ISIN12"
            ]
        )

    if "TICKER" in firm.columns:

        firm[
            "TICKER"
        ] = clean_text_identifier(
            firm[
                "TICKER"
            ]
        )

    duplicate_cik_year = firm.duplicated(
        subset=[
            "CIK10",
            "FISCAL_YEAR",
        ],
        keep=False,
    )

    if duplicate_cik_year.any():

        raise RuntimeError(
            "Duplicate CIK10-FISCAL_YEAR gözlemi bulundu."
        )

    firm[
        "CUSIP9_DUPLICATE_FLAG"
    ] = (
        firm[
            "CUSIP9"
        ].notna()
        & firm.duplicated(
            subset=[
                "CUSIP9"
            ],
            keep=False,
        )
    )

    firm[
        "CUSIP8_DUPLICATE_FLAG"
    ] = (
        firm[
            "CUSIP8"
        ].notna()
        & firm.duplicated(
            subset=[
                "CUSIP8"
            ],
            keep=False,
        )
    )

    return firm


# ============================================================
# 11. HOLDINGS-FİRMA MERGE
# ============================================================

def merge_holdings_with_firm_characteristics(
    holdings: pd.DataFrame,
    firm: pd.DataFrame,
) -> pd.DataFrame:
    """
    İki aşamalı deterministik eşleştirme:

    1. CUSIP9
    2. CUSIP8 fallback
    """

    source_variables = [
        settings[
            "source"
        ]
        for settings
        in CHARACTERISTIC_MAP.values()
    ]

    metadata_columns = [
        "CIK10",
        "ENTITY_NAME",
        "FISCAL_YEAR",
        "CUSIP9",
        "CUSIP8",
    ]

    if "ISIN12" in firm.columns:

        metadata_columns.append(
            "ISIN12"
        )

    if "TICKER" in firm.columns:

        metadata_columns.append(
            "TICKER"
        )

    # --------------------------------------------------------
    # CUSIP9
    # --------------------------------------------------------

    firm_cusip9 = firm.loc[
        firm[
            "CUSIP9"
        ].notna()
        & ~firm[
            "CUSIP9_DUPLICATE_FLAG"
        ],
        metadata_columns
        + source_variables,
    ].copy()

    rename_map_cusip9 = {
        "CUSIP9": "HOLDING_CUSIP9",
        "CUSIP8": "FIRM_CUSIP8",
    }

    if "ISIN12" in firm_cusip9.columns:

        rename_map_cusip9[
            "ISIN12"
        ] = "FIRM_ISIN12"

    if "TICKER" in firm_cusip9.columns:

        rename_map_cusip9[
            "TICKER"
        ] = "FIRM_TICKER"

    firm_cusip9 = firm_cusip9.rename(
        columns=rename_map_cusip9
    )

    merged = holdings.merge(
        firm_cusip9,
        on="HOLDING_CUSIP9",
        how="left",
        validate="many_to_one",
        indicator="_CUSIP9_STATUS",
    )

    merged[
        "MATCH_METHOD"
    ] = pd.NA

    cusip9_success = (
        merged[
            "_CUSIP9_STATUS"
        ]
        .eq(
            "both"
        )
    )

    merged.loc[
        cusip9_success,
        "MATCH_METHOD",
    ] = "CUSIP9"

    # --------------------------------------------------------
    # CUSIP8 fallback
    # --------------------------------------------------------

    unmatched_index = merged.index[
        merged[
            "CIK10"
        ].isna()
        & merged[
            "HOLDING_CUSIP8"
        ].notna()
    ]

    firm_cusip8 = firm.loc[
        firm[
            "CUSIP8"
        ].notna()
        & ~firm[
            "CUSIP8_DUPLICATE_FLAG"
        ],
        metadata_columns
        + source_variables,
    ].copy()

    rename_map_cusip8 = {
        "CUSIP8": "HOLDING_CUSIP8",
        "CUSIP9": "FALLBACK_FIRM_CUSIP9",
    }

    if "ISIN12" in firm_cusip8.columns:

        rename_map_cusip8[
            "ISIN12"
        ] = "FALLBACK_FIRM_ISIN12"

    if "TICKER" in firm_cusip8.columns:

        rename_map_cusip8[
            "TICKER"
        ] = "FALLBACK_FIRM_TICKER"

    firm_cusip8 = firm_cusip8.rename(
        columns=rename_map_cusip8
    )

    if len(
        unmatched_index
    ) > 0:

        fallback_input = (
            merged.loc[
                unmatched_index,
                [
                    "HOLDING_CUSIP8"
                ],
            ]
            .copy()
        )

        fallback_input[
            "_ORIGINAL_INDEX"
        ] = fallback_input.index

        fallback = fallback_input.merge(
            firm_cusip8,
            on="HOLDING_CUSIP8",
            how="left",
            validate="many_to_one",
        )

        fallback = fallback.set_index(
            "_ORIGINAL_INDEX"
        )

        columns_to_fill = [
            "CIK10",
            "ENTITY_NAME",
            "FISCAL_YEAR",
        ] + source_variables

        for column in columns_to_fill:

            if column in fallback.columns:

                merged.loc[
                    fallback.index,
                    column,
                ] = fallback[
                    column
                ]

        successful_fallback = (
            fallback[
                "CIK10"
            ].notna()
        )

        successful_indexes = (
            fallback.index[
                successful_fallback
            ]
        )

        merged.loc[
            successful_indexes,
            "MATCH_METHOD",
        ] = "CUSIP8"

    merged[
        "FINANCIAL_MATCHED"
    ] = merged[
        "CIK10"
    ].notna()

    merged[
        "CHARACTERISTIC_FISCAL_YEAR"
    ] = CHARACTERISTIC_FISCAL_YEAR

    merged = merged.drop(
        columns=[
            "_CUSIP9_STATUS"
        ],
        errors="ignore",
    )

    return merged


# ============================================================
# 12. PORTFOLIO ARCHITECTURE HESAPLAMA
# ============================================================

def calculate_portfolio_architecture(
    merged: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    ETF düzeyinde portfolio architecture değişkenlerini hesaplar.
    """

    portfolio_keys = [
        "ACCESSION_NUMBER",
        "ETF_ID",
        "ETF_NAME",
    ]

    architecture_rows = []

    long_rows = []

    for (
        portfolio_values,
        group,
    ) in merged.groupby(
        portfolio_keys,
        dropna=False,
        sort=True,
    ):

        (
            accession_number,
            etf_id,
            etf_name,
        ) = portfolio_values

        group = group.copy()

        weights = safe_numeric(
            group[
                "EQUITY_NORMALIZED_WEIGHT"
            ]
        )

        valid_weight_mask = (
            weights.notna()
            & (
                weights >= 0
            )
        )

        weights = weights.where(
            valid_weight_mask
        )

        weight_sum = weights.sum()

        if weight_sum > 0:

            normalized_weights = (
                weights
                / weight_sum
            )

        else:

            normalized_weights = weights

        group[
            "_ANALYSIS_WEIGHT"
        ] = normalized_weights

        sorted_weights = (
            normalized_weights
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        hhi = float(
            (
                sorted_weights ** 2
            ).sum()
        )

        top5_weight = float(
            sorted_weights
            .head(
                5
            )
            .sum()
        )

        top10_weight = float(
            sorted_weights
            .head(
                10
            )
            .sum()
        )

        maximum_weight = (
            float(
                sorted_weights.iloc[
                    0
                ]
            )
            if len(
                sorted_weights
            ) > 0
            else np.nan
        )

        effective_number = (
            float(
                1.0
                / hhi
            )
            if hhi > 0
            else np.nan
        )

        financial_match_weight = float(
            group.loc[
                group[
                    "FINANCIAL_MATCHED"
                ],
                "_ANALYSIS_WEIGHT",
            ].sum()
        )

        row = {
            "ACCESSION_NUMBER": accession_number,

            "ETF_ID": etf_id,

            "ETF_NAME": etf_name,

            "CHARACTERISTIC_FISCAL_YEAR": (
                CHARACTERISTIC_FISCAL_YEAR
            ),

            "N_HOLDINGS": int(
                len(
                    group
                )
            ),

            "N_MATCHED_HOLDINGS": int(
                group[
                    "FINANCIAL_MATCHED"
                ].sum()
            ),

            "HOLDING_MATCH_RATE": float(
                group[
                    "FINANCIAL_MATCHED"
                ].mean()
            ),

            "FINANCIAL_MATCH_WEIGHT": (
                financial_match_weight
            ),

            "UNMATCHED_WEIGHT": float(
                1.0
                - financial_match_weight
            ),

            "HHI": hhi,

            "TOP5_WEIGHT": top5_weight,

            "TOP10_WEIGHT": top10_weight,

            "MAX_HOLDING_WEIGHT": (
                maximum_weight
            ),

            "EFFECTIVE_NUMBER_OF_HOLDINGS": (
                effective_number
            ),
        }

        characteristic_coverages = []

        for (
            output_name,
            settings,
        ) in CHARACTERISTIC_MAP.items():

            source_name = settings[
                "source"
            ]

            coverage_name = settings[
                "coverage"
            ]

            dispersion_name = settings[
                "dispersion"
            ]

            values = safe_numeric(
                group[
                    source_name
                ]
            )

            available_mask = (
                values.notna()
                & group[
                    "_ANALYSIS_WEIGHT"
                ].notna()
            )

            coverage_value = float(
                group.loc[
                    available_mask,
                    "_ANALYSIS_WEIGHT",
                ].sum()
            )

            weighted_average = weighted_mean(
                values=values,
                weights=group[
                    "_ANALYSIS_WEIGHT"
                ],
            )

            dispersion_value = weighted_dispersion(
                values=values,
                weights=group[
                    "_ANALYSIS_WEIGHT"
                ],
            )

            valid_firm_count = int(
                available_mask.sum()
            )

            row[
                output_name
            ] = weighted_average

            row[
                coverage_name
            ] = coverage_value

            row[
                dispersion_name
            ] = dispersion_value

            characteristic_coverages.append(
                coverage_value
            )

            # ------------------------------------------------
            # Characteristic-specific coverage sample flags
            #
            # Firma değişkenleri farklı SEC coverage oranlarına
            # sahip olduğu için ortak tek coverage kısıtı yoktur.
            # ------------------------------------------------

            short_name = output_name.replace(
                "PW_",
                "",
            )

            for (
                threshold_label,
                threshold_value,
            ) in [
                (
                    "COV50",
                    0.50,
                ),
                (
                    "COV60",
                    0.60,
                ),
                (
                    "COV70",
                    0.70,
                ),
                (
                    "COV80",
                    0.80,
                ),
                (
                    "COV90",
                    0.90,
                ),
            ]:

                row[
                    f"SAMPLE_{short_name}_{threshold_label}"
                ] = int(
                    pd.notna(
                        coverage_value
                    )
                    and (
                        coverage_value
                        >= threshold_value
                    )
                )

            long_rows.append(
                {
                    "ACCESSION_NUMBER": accession_number,

                    "ETF_ID": etf_id,

                    "ETF_NAME": etf_name,

                    "CHARACTERISTIC": (
                        output_name
                    ),

                    "SOURCE_VARIABLE": (
                        source_name
                    ),

                    "WEIGHTED_MEAN": (
                        weighted_average
                    ),

                    "DISPERSION": (
                        dispersion_value
                    ),

                    "COVERAGE": (
                        coverage_value
                    ),

                    "VALID_FIRMS": (
                        valid_firm_count
                    ),

                    "TOTAL_HOLDINGS": (
                        len(
                            group
                        )
                    ),

                    "THEORY": settings[
                        "theory"
                    ],

                    "RESEARCH_QUESTION": settings[
                        "research_question"
                    ],
                }
            )

        valid_coverages = [
            coverage_value
            for coverage_value
            in characteristic_coverages
            if pd.notna(
                coverage_value
            )
        ]

        if valid_coverages:

            row[
                "MEAN_CHARACTERISTIC_COVERAGE"
            ] = float(
                np.mean(
                    valid_coverages
                )
            )

            row[
                "MEDIAN_CHARACTERISTIC_COVERAGE"
            ] = float(
                np.median(
                    valid_coverages
                )
            )

            row[
                "MIN_CHARACTERISTIC_COVERAGE"
            ] = float(
                np.min(
                    valid_coverages
                )
            )

        else:

            row[
                "MEAN_CHARACTERISTIC_COVERAGE"
            ] = np.nan

            row[
                "MEDIAN_CHARACTERISTIC_COVERAGE"
            ] = np.nan

            row[
                "MIN_CHARACTERISTIC_COVERAGE"
            ] = np.nan

        row[
            "N_CHARACTERISTICS_ABOVE_50PCT"
        ] = int(
            sum(
                coverage_value
                >= 0.50
                for coverage_value
                in valid_coverages
            )
        )

        row[
            "N_CHARACTERISTICS_ABOVE_70PCT"
        ] = int(
            sum(
                coverage_value
                >= 0.70
                for coverage_value
                in valid_coverages
            )
        )

        row[
            "N_CHARACTERISTICS_ABOVE_80PCT"
        ] = int(
            sum(
                coverage_value
                >= 0.80
                for coverage_value
                in valid_coverages
            )
        )

        row[
            "N_CHARACTERISTICS_ABOVE_90PCT"
        ] = int(
            sum(
                coverage_value
                >= 0.90
                for coverage_value
                in valid_coverages
            )
        )

        # Genel financial-match sample flags

        row[
            "SAMPLE_FINANCIAL_MATCH_70"
        ] = int(
            financial_match_weight
            >= FINANCIAL_MATCH_THRESHOLD_70
        )

        row[
            "SAMPLE_FINANCIAL_MATCH_80"
        ] = int(
            financial_match_weight
            >= FINANCIAL_MATCH_THRESHOLD_80
        )

        row[
            "SAMPLE_FINANCIAL_MATCH_90"
        ] = int(
            financial_match_weight
            >= FINANCIAL_MATCH_THRESHOLD_90
        )

        # Tek holdings kesiti bulunduğu için stability yok

        row[
            "HOLDINGS_TURNOVER"
        ] = np.nan

        row[
            "ARCHITECTURE_STABILITY"
        ] = np.nan

        row[
            "STABILITY_AVAILABLE_FLAG"
        ] = 0

        architecture_rows.append(
            row
        )

    architecture = pd.DataFrame(
        architecture_rows
    )

    characteristics_long = pd.DataFrame(
        long_rows
    )

    return (
        architecture,
        characteristics_long,
    )


# ============================================================
# 13. MERGE DIAGNOSTICS
# ============================================================

def build_merge_diagnostics(
    raw_holdings: pd.DataFrame,
    aggregated_holdings: pd.DataFrame,
    merged: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Genel ve ETF bazlı eşleştirme raporlarını oluşturur.
    """

    portfolio_keys = [
        "ACCESSION_NUMBER",
        "ETF_ID",
        "ETF_NAME",
    ]

    total_weight = float(
        merged[
            "EQUITY_NORMALIZED_WEIGHT"
        ].sum()
    )

    matched_weight = float(
        merged.loc[
            merged[
                "FINANCIAL_MATCHED"
            ],
            "EQUITY_NORMALIZED_WEIGHT",
        ].sum()
    )

    general_rows = [
        {
            "METRIC": "RAW_HOLDING_ROWS",
            "VALUE": len(
                raw_holdings
            ),
        },

        {
            "METRIC": "AGGREGATED_HOLDING_ROWS",
            "VALUE": len(
                aggregated_holdings
            ),
        },

        {
            "METRIC": "UNIQUE_ETFS",
            "VALUE": merged[
                "ETF_ID"
            ].nunique(),
        },

        {
            "METRIC": "MATCHED_HOLDING_ROWS",
            "VALUE": int(
                merged[
                    "FINANCIAL_MATCHED"
                ].sum()
            ),
        },

        {
            "METRIC": "UNMATCHED_HOLDING_ROWS",
            "VALUE": int(
                (
                    ~merged[
                        "FINANCIAL_MATCHED"
                    ]
                ).sum()
            ),
        },

        {
            "METRIC": "HOLDING_ROW_MATCH_RATE",
            "VALUE": float(
                merged[
                    "FINANCIAL_MATCHED"
                ].mean()
            ),
        },

        {
            "METRIC": "CUSIP9_MATCHED_ROWS",
            "VALUE": int(
                merged[
                    "MATCH_METHOD"
                ]
                .eq(
                    "CUSIP9"
                )
                .sum()
            ),
        },

        {
            "METRIC": "CUSIP8_MATCHED_ROWS",
            "VALUE": int(
                merged[
                    "MATCH_METHOD"
                ]
                .eq(
                    "CUSIP8"
                )
                .sum()
            ),
        },

        {
            "METRIC": "MISSING_CUSIP9_ROWS",
            "VALUE": int(
                merged[
                    "HOLDING_CUSIP9"
                ].isna().sum()
            ),
        },

        {
            "METRIC": "TOTAL_NORMALIZED_WEIGHT",
            "VALUE": total_weight,
        },

        {
            "METRIC": "MATCHED_NORMALIZED_WEIGHT",
            "VALUE": matched_weight,
        },

        {
            "METRIC": "OVERALL_WEIGHTED_MATCH_RATE",
            "VALUE": (
                matched_weight
                / total_weight
                if total_weight > 0
                else np.nan
            ),
        },
    ]

    general_diagnostics = pd.DataFrame(
        general_rows
    )

    etf_rows = []

    for (
        keys,
        group,
    ) in merged.groupby(
        portfolio_keys,
        dropna=False,
    ):

        (
            accession_number,
            etf_id,
            etf_name,
        ) = keys

        portfolio_weight = float(
            group[
                "EQUITY_NORMALIZED_WEIGHT"
            ].sum()
        )

        portfolio_matched_weight = float(
            group.loc[
                group[
                    "FINANCIAL_MATCHED"
                ],
                "EQUITY_NORMALIZED_WEIGHT",
            ].sum()
        )

        etf_rows.append(
            {
                "ACCESSION_NUMBER": accession_number,

                "ETF_ID": etf_id,

                "ETF_NAME": etf_name,

                "N_HOLDINGS": len(
                    group
                ),

                "N_MATCHED_HOLDINGS": int(
                    group[
                        "FINANCIAL_MATCHED"
                    ].sum()
                ),

                "HOLDING_MATCH_RATE": float(
                    group[
                        "FINANCIAL_MATCHED"
                    ].mean()
                ),

                "TOTAL_WEIGHT": portfolio_weight,

                "MATCHED_WEIGHT": (
                    portfolio_matched_weight
                ),

                "WEIGHTED_MATCH_RATE": (
                    portfolio_matched_weight
                    / portfolio_weight
                    if portfolio_weight > 0
                    else np.nan
                ),

                "CUSIP9_MATCH_WEIGHT": float(
                    group.loc[
                        group[
                            "MATCH_METHOD"
                        ]
                        .eq(
                            "CUSIP9"
                        ),
                        "EQUITY_NORMALIZED_WEIGHT",
                    ].sum()
                ),

                "CUSIP8_MATCH_WEIGHT": float(
                    group.loc[
                        group[
                            "MATCH_METHOD"
                        ]
                        .eq(
                            "CUSIP8"
                        ),
                        "EQUITY_NORMALIZED_WEIGHT",
                    ].sum()
                ),

                "UNMATCHED_WEIGHT": float(
                    group.loc[
                        ~group[
                            "FINANCIAL_MATCHED"
                        ],
                        "EQUITY_NORMALIZED_WEIGHT",
                    ].sum()
                ),

                "MISSING_CUSIP_WEIGHT": float(
                    group.loc[
                        group[
                            "HOLDING_CUSIP9"
                        ].isna(),
                        "EQUITY_NORMALIZED_WEIGHT",
                    ].sum()
                ),

                "WEIGHT_SUM_DEVIATION": float(
                    abs(
                        portfolio_weight
                        - 1.0
                    )
                ),
            }
        )

    etf_diagnostics = pd.DataFrame(
        etf_rows
    )

    return (
        general_diagnostics,
        etf_diagnostics,
    )


# ============================================================
# 14. COVERAGE RAPORU
# ============================================================

def build_characteristic_coverage_report(
    architecture: pd.DataFrame,
) -> pd.DataFrame:
    """
    Portföy karakteristiklerinin coverage özetini üretir.
    """

    report_rows = []

    for (
        output_name,
        settings,
    ) in CHARACTERISTIC_MAP.items():

        coverage_column = settings[
            "coverage"
        ]

        coverage_values = safe_numeric(
            architecture[
                coverage_column
            ]
        )

        report_rows.append(
            {
                "PORTFOLIO_VARIABLE": (
                    output_name
                ),

                "SOURCE_VARIABLE": (
                    settings[
                        "source"
                    ]
                ),

                "COVERAGE_VARIABLE": (
                    coverage_column
                ),

                "MEAN_COVERAGE": float(
                    coverage_values.mean()
                ),

                "MEDIAN_COVERAGE": float(
                    coverage_values.median()
                ),

                "MIN_COVERAGE": float(
                    coverage_values.min()
                ),

                "MAX_COVERAGE": float(
                    coverage_values.max()
                ),

                "ETFS_ABOVE_50PCT": int(
                    (
                        coverage_values
                        >= 0.50
                    ).sum()
                ),

                "ETFS_ABOVE_60PCT": int(
                    (
                        coverage_values
                        >= 0.60
                    ).sum()
                ),

                "ETFS_ABOVE_70PCT": int(
                    (
                        coverage_values
                        >= 0.70
                    ).sum()
                ),

                "ETFS_ABOVE_80PCT": int(
                    (
                        coverage_values
                        >= 0.80
                    ).sum()
                ),

                "ETFS_ABOVE_90PCT": int(
                    (
                        coverage_values
                        >= 0.90
                    ).sum()
                ),

                "THEORY": settings[
                    "theory"
                ],

                "RESEARCH_QUESTION": settings[
                    "research_question"
                ],
            }
        )

    return pd.DataFrame(
        report_rows
    )


# ============================================================
# 15. VARIABLE DICTIONARY
# ============================================================

def build_variable_dictionary() -> pd.DataFrame:
    """
    Portfolio architecture değişken sözlüğünü oluşturur.
    """

    rows = [
        {
            "VARIABLE": "HHI",

            "SOURCE": "ETF holding weights",

            "FORMULA": "sum(weight_j^2)",

            "THEORY": (
                "Portfolio concentration"
            ),

            "RESEARCH_QUESTION": (
                "RQ1, RQ3"
            ),

            "EXPECTED_SIGN": (
                "Higher concentration may amplify CPU exposure"
            ),
        },

        {
            "VARIABLE": "TOP10_WEIGHT",

            "SOURCE": "ETF holding weights",

            "FORMULA": (
                "Sum of ten largest holding weights"
            ),

            "THEORY": (
                "Dominant-firm concentration"
            ),

            "RESEARCH_QUESTION": (
                "RQ1, RQ3"
            ),

            "EXPECTED_SIGN": (
                "Higher concentration may amplify CPU exposure"
            ),
        },

        {
            "VARIABLE": "EFFECTIVE_NUMBER_OF_HOLDINGS",

            "SOURCE": "ETF holding weights",

            "FORMULA": "1 / HHI",

            "THEORY": (
                "Effective diversification"
            ),

            "RESEARCH_QUESTION": (
                "RQ1, RQ3"
            ),

            "EXPECTED_SIGN": (
                "Greater diversification may attenuate exposure"
            ),
        },

        {
            "VARIABLE": "FINANCIAL_MATCH_WEIGHT",

            "SOURCE": (
                "Matched firm-characteristic holding weights"
            ),

            "FORMULA": (
                "Sum of matched holding weights"
            ),

            "THEORY": (
                "Measurement reliability"
            ),

            "RESEARCH_QUESTION": (
                "All"
            ),

            "EXPECTED_SIGN": (
                "Not an economic sign prediction"
            ),
        },

        {
            "VARIABLE": "ARCHITECTURE_STABILITY",

            "SOURCE": (
                "Consecutive holdings snapshots"
            ),

            "FORMULA": (
                "1 - 0.5 * sum(abs(weight_t - weight_t-1))"
            ),

            "THEORY": (
                "Persistence of embedded portfolio exposure"
            ),

            "RESEARCH_QUESTION": (
                "RQ1, RQ4"
            ),

            "EXPECTED_SIGN": (
                "Conditional"
            ),
        },
    ]

    for (
        output_name,
        settings,
    ) in CHARACTERISTIC_MAP.items():

        rows.append(
            {
                "VARIABLE": output_name,

                "SOURCE": settings[
                    "source"
                ],

                "FORMULA": (
                    "Weighted firm characteristic with "
                    "available-weight renormalization"
                ),

                "THEORY": settings[
                    "theory"
                ],

                "RESEARCH_QUESTION": settings[
                    "research_question"
                ],

                "EXPECTED_SIGN": settings[
                    "expected_cpu_interaction"
                ],
            }
        )

        rows.append(
            {
                "VARIABLE": settings[
                    "coverage"
                ],

                "SOURCE": settings[
                    "source"
                ],

                "FORMULA": (
                    "Sum of portfolio weights with available characteristic"
                ),

                "THEORY": (
                    "Characteristic-specific measurement coverage"
                ),

                "RESEARCH_QUESTION": (
                    "All"
                ),

                "EXPECTED_SIGN": (
                    "Not an economic sign prediction"
                ),
            }
        )

        rows.append(
            {
                "VARIABLE": settings[
                    "dispersion"
                ],

                "SOURCE": settings[
                    "source"
                ],

                "FORMULA": (
                    "Weighted within-portfolio standard deviation"
                ),

                "THEORY": (
                    "Firm-level heterogeneity embedded in the ETF"
                ),

                "RESEARCH_QUESTION": (
                    "RQ1, RQ2, RQ3"
                ),

                "EXPECTED_SIGN": (
                    "Conditional"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. VALIDATION
# ============================================================

def run_validation(
    holdings_raw: pd.DataFrame,
    holdings_aggregated: pd.DataFrame,
    architecture: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mekanik ve metodolojik doğrulama kontrollerini çalıştırır.
    """

    portfolio_keys = [
        "ACCESSION_NUMBER",
        "ETF_ID",
        "ETF_NAME",
    ]

    weight_sums = (
        holdings_aggregated.groupby(
            portfolio_keys,
            dropna=False,
        )[
            "EQUITY_NORMALIZED_WEIGHT"
        ]
        .sum()
    )

    maximum_weight_deviation = float(
        (
            weight_sums
            - 1.0
        )
        .abs()
        .max()
    )

    duplicate_architecture_rows = int(
        architecture.duplicated(
            subset=portfolio_keys,
            keep=False,
        ).sum()
    )

    hhi_outside_range = int(
        (
            (
                architecture[
                    "HHI"
                ] < 0
            )
            | (
                architecture[
                    "HHI"
                ] > 1
            )
        ).sum()
    )

    top10_outside_range = int(
        (
            (
                architecture[
                    "TOP10_WEIGHT"
                ] < 0
            )
            | (
                architecture[
                    "TOP10_WEIGHT"
                ] > 1
            )
        ).sum()
    )

    effective_number_problem = int(
        (
            architecture[
                "EFFECTIVE_NUMBER_OF_HOLDINGS"
            ]
            >
            architecture[
                "N_HOLDINGS"
            ]
            + 1e-6
        ).sum()
    )

    negative_weight_count = int(
        (
            holdings_aggregated[
                "EQUITY_NORMALIZED_WEIGHT"
            ] < 0
        ).sum()
    )

    validation_rows = [
        {
            "CHECK": "RAW_HOLDINGS_AVAILABLE",

            "VALUE": len(
                holdings_raw
            ),

            "PASS": int(
                len(
                    holdings_raw
                ) > 0
            ),
        },

        {
            "CHECK": "AGGREGATED_HOLDINGS_AVAILABLE",

            "VALUE": len(
                holdings_aggregated
            ),

            "PASS": int(
                len(
                    holdings_aggregated
                ) > 0
            ),
        },

        {
            "CHECK": "PORTFOLIOS_AVAILABLE",

            "VALUE": len(
                architecture
            ),

            "PASS": int(
                len(
                    architecture
                ) > 0
            ),
        },

        {
            "CHECK": "PORTFOLIO_WEIGHT_SUM_MAX_DEVIATION",

            "VALUE": maximum_weight_deviation,

            "PASS": int(
                maximum_weight_deviation
                <= WEIGHT_TOLERANCE
            ),
        },

        {
            "CHECK": "NEGATIVE_NORMALIZED_WEIGHTS",

            "VALUE": negative_weight_count,

            "PASS": int(
                negative_weight_count
                == 0
            ),
        },

        {
            "CHECK": "DUPLICATE_ARCHITECTURE_ROWS",

            "VALUE": duplicate_architecture_rows,

            "PASS": int(
                duplicate_architecture_rows
                == 0
            ),
        },

        {
            "CHECK": "HHI_OUTSIDE_ZERO_ONE",

            "VALUE": hhi_outside_range,

            "PASS": int(
                hhi_outside_range
                == 0
            ),
        },

        {
            "CHECK": "TOP10_OUTSIDE_ZERO_ONE",

            "VALUE": top10_outside_range,

            "PASS": int(
                top10_outside_range
                == 0
            ),
        },

        {
            "CHECK": "EFFECTIVE_N_EXCEEDS_HOLDINGS",

            "VALUE": effective_number_problem,

            "PASS": int(
                effective_number_problem
                == 0
            ),
        },

        {
            "CHECK": "PORTFOLIOS_FINANCIAL_MATCH_AT_LEAST_70PCT",

            "VALUE": int(
                architecture[
                    "SAMPLE_FINANCIAL_MATCH_70"
                ].sum()
            ),

            "PASS": int(
                architecture[
                    "SAMPLE_FINANCIAL_MATCH_70"
                ].sum()
                > 0
            ),
        },

        {
            "CHECK": "PORTFOLIOS_FINANCIAL_MATCH_AT_LEAST_80PCT",

            "VALUE": int(
                architecture[
                    "SAMPLE_FINANCIAL_MATCH_80"
                ].sum()
            ),

            "PASS": int(
                architecture[
                    "SAMPLE_FINANCIAL_MATCH_80"
                ].sum()
                > 0
            ),
        },

        {
            "CHECK": "PORTFOLIOS_FINANCIAL_MATCH_AT_LEAST_90PCT",

            "VALUE": int(
                architecture[
                    "SAMPLE_FINANCIAL_MATCH_90"
                ].sum()
            ),

            "PASS": int(
                architecture[
                    "SAMPLE_FINANCIAL_MATCH_90"
                ].sum()
                > 0
            ),
        },

        {
            "CHECK": "ARCHITECTURE_STABILITY_AVAILABLE",

            "VALUE": int(
                architecture[
                    "STABILITY_AVAILABLE_FLAG"
                ].sum()
            ),

            "PASS": 1,
        },
    ]

    return pd.DataFrame(
        validation_rows
    )


# ============================================================
# 17. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print(
        "=" * 78
    )

    print(
        "15 - HOLDINGS-BASED PORTFOLIO ARCHITECTURE"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # 1. Dosyaları oku
    # --------------------------------------------------------

    print(
        "\n1/10 - Holdings dosyası okunuyor..."
    )

    holdings_raw = read_data(
        HOLDINGS_FILE
    )

    print(
        f"Ham holdings satırı: "
        f"{len(holdings_raw):,}"
    )

    print(
        "\n2/10 - Firma karakteristikleri okunuyor..."
    )

    firm_raw = read_data(
        FIRM_FILE
    )

    print(
        f"Firma-yıl satırı: "
        f"{len(firm_raw):,}"
    )

    # --------------------------------------------------------
    # 2. Holdings standardize
    # --------------------------------------------------------

    print(
        "\n3/10 - Holdings standardize ediliyor..."
    )

    holdings = standardize_holdings(
        holdings_raw
    )

    print(
        f"Benzersiz ETF: "
        f"{holdings['ETF_ID'].nunique():,}"
    )

    # --------------------------------------------------------
    # 3. Ağırlıklar
    # --------------------------------------------------------

    print(
        "\n4/10 - Holding ağırlıkları oluşturuluyor..."
    )

    holdings = construct_holding_weights(
        holdings
    )

    missing_weight_count = int(
        holdings[
            "EQUITY_NORMALIZED_WEIGHT"
        ].isna().sum()
    )

    print(
        f"Ağırlığı oluşturulamayan satır: "
        f"{missing_weight_count:,}"
    )

    # --------------------------------------------------------
    # 4. Duplicate aggregation
    # --------------------------------------------------------

    print(
        "\n5/10 - Aynı ekonomik holdingler toplulaştırılıyor..."
    )

    holdings_aggregated = aggregate_duplicate_holdings(
        holdings
    )

    print(
        f"Toplulaştırılmış holding satırı: "
        f"{len(holdings_aggregated):,}"
    )

    # --------------------------------------------------------
    # 5. Firma özellikleri
    # --------------------------------------------------------

    print(
        f"\n6/10 - FY{CHARACTERISTIC_FISCAL_YEAR} "
        "firma karakteristikleri hazırlanıyor..."
    )

    firm = prepare_firm_characteristics(
        firm_raw
    )

    print(
        f"FY{CHARACTERISTIC_FISCAL_YEAR} firma sayısı: "
        f"{firm['CIK10'].nunique():,}"
    )

    # --------------------------------------------------------
    # 6. Merge
    # --------------------------------------------------------

    print(
        "\n7/10 - Holdings ile firma karakteristikleri "
        "CUSIP üzerinden eşleştiriliyor..."
    )

    merged = merge_holdings_with_firm_characteristics(
        holdings=holdings_aggregated,
        firm=firm,
    )

    print(
        f"Eşleşen holding satırı: "
        f"{merged['FINANCIAL_MATCHED'].sum():,}"
    )

    print(
        f"Eşleşmeyen holding satırı: "
        f"{(~merged['FINANCIAL_MATCHED']).sum():,}"
    )

    # --------------------------------------------------------
    # 7. Portfolio architecture
    # --------------------------------------------------------

    print(
        "\n8/10 - Portfolio architecture değişkenleri hesaplanıyor..."
    )

    (
        architecture,
        characteristics_long,
    ) = calculate_portfolio_architecture(
        merged
    )

    # --------------------------------------------------------
    # 8. Raporlar
    # --------------------------------------------------------

    print(
        "\n9/10 - Diagnostics ve validation raporları hazırlanıyor..."
    )

    (
        merge_diagnostics,
        etf_diagnostics,
    ) = build_merge_diagnostics(
        raw_holdings=holdings,
        aggregated_holdings=holdings_aggregated,
        merged=merged,
    )

    coverage_report = build_characteristic_coverage_report(
        architecture
    )

    concentration_columns = [
        "ACCESSION_NUMBER",
        "ETF_ID",
        "ETF_NAME",
        "N_HOLDINGS",
        "HHI",
        "TOP5_WEIGHT",
        "TOP10_WEIGHT",
        "MAX_HOLDING_WEIGHT",
        "EFFECTIVE_NUMBER_OF_HOLDINGS",
        "FINANCIAL_MATCH_WEIGHT",
    ]

    concentration_report = (
        architecture[
            concentration_columns
        ]
        .sort_values(
            by="HHI",
            ascending=False,
        )
        .copy()
    )

    validation = run_validation(
        holdings_raw=holdings,
        holdings_aggregated=holdings_aggregated,
        architecture=architecture,
    )

    variable_dictionary = build_variable_dictionary()

    unmatched = (
        merged.loc[
            ~merged[
                "FINANCIAL_MATCHED"
            ],
            [
                "ACCESSION_NUMBER",
                "ETF_ID",
                "ETF_NAME",
                "ISSUER_NAME",
                "HOLDING_CUSIP9",
                "HOLDING_CUSIP8",
                "HOLDING_ISIN",
                "HOLDING_TICKER",
                "EQUITY_NORMALIZED_WEIGHT",
            ],
        ]
        .sort_values(
            by=[
                "ETF_NAME",
                "EQUITY_NORMALIZED_WEIGHT",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    # --------------------------------------------------------
    # 9. Kaydet
    # --------------------------------------------------------

    print(
        "\n10/10 - Çıktılar kaydediliyor..."
    )

    merged.to_parquet(
        MERGED_HOLDINGS_FILE,
        index=False,
    )

    merged.to_csv(
        MERGED_HOLDINGS_CSV_FILE,
        index=False,
    )

    architecture.to_csv(
        PORTFOLIO_ARCHITECTURE_FILE,
        index=False,
    )

    architecture.to_parquet(
        PORTFOLIO_ARCHITECTURE_PARQUET_FILE,
        index=False,
    )

    characteristics_long.to_csv(
        CHARACTERISTICS_LONG_FILE,
        index=False,
    )

    merge_diagnostics.to_csv(
        MERGE_DIAGNOSTICS_FILE,
        index=False,
    )

    etf_diagnostics.to_csv(
        ETF_DIAGNOSTICS_FILE,
        index=False,
    )

    unmatched.to_csv(
        UNMATCHED_HOLDINGS_FILE,
        index=False,
    )

    coverage_report.to_csv(
        CHARACTERISTIC_COVERAGE_REPORT_FILE,
        index=False,
    )

    concentration_report.to_csv(
        CONCENTRATION_REPORT_FILE,
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
    # 10. Ekran çıktısı
    # --------------------------------------------------------

    print(
        "\nPORTFOLIO ARCHITECTURE HAZIR"
    )

    print(
        "=" * 78
    )

    print(
        "\nGenel merge sonuçları:"
    )

    print(
        merge_diagnostics.to_string(
            index=False
        )
    )

    print(
        "\nETF finansal eşleşme ağırlığı özeti:"
    )

    print(
        architecture[
            "FINANCIAL_MATCH_WEIGHT"
        ]
        .describe()
        .to_string()
    )

    print(
        "\nGenel örneklem eşikleri:"
    )

    print(
        "Financial match >= 70%: "
        f"{architecture['SAMPLE_FINANCIAL_MATCH_70'].sum():,}"
    )

    print(
        "Financial match >= 80%: "
        f"{architecture['SAMPLE_FINANCIAL_MATCH_80'].sum():,}"
    )

    print(
        "Financial match >= 90%: "
        f"{architecture['SAMPLE_FINANCIAL_MATCH_90'].sum():,}"
    )

    print(
        "\nKarakteristik bazlı coverage eşikleri:"
    )

    for output_name in CHARACTERISTIC_MAP:

        short_name = output_name.replace(
            "PW_",
            "",
        )

        count_50 = int(
            architecture[
                f"SAMPLE_{short_name}_COV50"
            ].sum()
        )

        count_70 = int(
            architecture[
                f"SAMPLE_{short_name}_COV70"
            ].sum()
        )

        count_80 = int(
            architecture[
                f"SAMPLE_{short_name}_COV80"
            ].sum()
        )

        print(
            f"{short_name:<32} "
            f">=50%: {count_50:>2} | "
            f">=70%: {count_70:>2} | "
            f">=80%: {count_80:>2}"
        )

    print(
        "\nConcentration özeti:"
    )

    print(
        architecture[
            [
                "HHI",
                "TOP10_WEIGHT",
                "EFFECTIVE_NUMBER_OF_HOLDINGS",
            ]
        ]
        .describe()
        .to_string()
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
        "\nBaşarısız mekanik kontrol sayısı: "
        f"{len(failed_checks):,}"
    )

    print(
        "\nNot: Architecture Stability şu anda NaN'dır. "
        "Çünkü yalnızca 2025Q4 holdings kesiti bulunmaktadır."
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        PORTFOLIO_ARCHITECTURE_FILE
    )

    print(
        CHARACTERISTICS_LONG_FILE
    )

    print(
        ETF_DIAGNOSTICS_FILE
    )

    print(
        CHARACTERISTIC_COVERAGE_REPORT_FILE
    )

    print(
        CONCENTRATION_REPORT_FILE
    )

    print(
        VALIDATION_FILE
    )

    print(
        VARIABLE_DICTIONARY_FILE
    )


if __name__ == "__main__":
    main()