from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 19_build_cpu_return_panel.py
#
# AMAÇ
# ----
# Quality-adjusted portfolio architecture faktörlerini:
#
# - Aylık ETF getirileri
# - Climate Policy Uncertainty (CPU)
# - Piyasa getirisi
# - Enerji sektörü getirisi
# - Uzun vadeli tahvil getirisi
#
# ile birleştirerek ana ekonometrik paneli oluşturmak.
#
# Temel model:
#
# Return_(i,t)
#   = alpha_i
#   + lambda_t
#   + beta_1 CPU_t
#   + beta_2 Architecture_i
#   + beta_3 CPU_t × Architecture_i
#   + Controls_t
#   + error_(i,t)
#
# Ana architecture faktörü:
# CORE_TRANSITION_SENSITIVITY_MAIN
#
# Robustness:
# - CORE_TRANSITION_SENSITIVITY_MODERATE
# - EXTENDED_TRANSITION_SENSITIVITY_MAIN
# - Mechanism factors
# ============================================================


# ============================================================
# 1. PROJE YOLLARI
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\User\Desktop\CPU_Project"
)

OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. ANA GİRDİ
# ============================================================

ARCHITECTURE_FILE = (
    OUTPUT_DIR
    / "18_quality_adjusted_architecture_factors.csv"
)


# ============================================================
# 3. OPSİYONEL AÇIK DOSYA YOLLARI
# ============================================================
#
# Dosyaların gerçek adlarını biliyorsan aşağıdaki değerleri
# doğrudan doldurabilirsin.
#
# None bırakıldığında script proje klasöründe otomatik arama yapar.

ETF_RETURN_FILE = None
CPU_FILE = None
CONTROL_FILE = None
ETF_TICKER_MAPPING_FILE = None


# Örnek:
#
# ETF_RETURN_FILE = (
#     OUTPUT_DIR / "monthly_etf_returns.csv"
# )
#
# CPU_FILE = (
#     DATA_DIR / "climate_policy_uncertainty_monthly.csv"
# )
#
# CONTROL_FILE = (
#     OUTPUT_DIR / "monthly_market_controls.csv"
# )


# ============================================================
# 4. ÇIKTI DOSYALARI
# ============================================================

PANEL_FILE = (
    OUTPUT_DIR
    / "19_cpu_return_architecture_panel.csv"
)

PANEL_PARQUET_FILE = (
    OUTPUT_DIR
    / "19_cpu_return_architecture_panel.parquet"
)

ETF_SAMPLE_FILE = (
    OUTPUT_DIR
    / "19_cpu_return_panel_etf_sample.csv"
)

MONTHLY_COVERAGE_FILE = (
    OUTPUT_DIR
    / "19_cpu_return_panel_monthly_coverage.csv"
)

ETF_COVERAGE_FILE = (
    OUTPUT_DIR
    / "19_cpu_return_panel_etf_coverage.csv"
)

MERGE_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "19_cpu_return_panel_merge_diagnostics.csv"
)

INPUT_INVENTORY_FILE = (
    OUTPUT_DIR
    / "19_input_file_inventory.csv"
)

COLUMN_INVENTORY_FILE = (
    OUTPUT_DIR
    / "19_input_column_inventory.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "19_cpu_return_panel_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "19_cpu_return_panel_variable_dictionary.csv"
)


# ============================================================
# 5. METODOLOJİK AYARLAR
# ============================================================

MIN_YEAR = 2010
MAX_YEAR = 2025

MIN_MONTHS_PER_ETF = 24

CPU_STANDARDIZATION_REFERENCE_START = "2010-01-01"
CPU_STANDARDIZATION_REFERENCE_END = "2025-12-31"

REQUIRE_MAIN_QUALITY = True


# ============================================================
# 6. SÜTUN ADI ADAYLARI
# ============================================================

DATE_CANDIDATES = [
    "DATE",
    "MONTH",
    "MONTH_DATE",
    "PERIOD",
    "TIME",
    "DATETIME",
    "OBSERVATION_DATE",
]

TICKER_CANDIDATES = [
    "TICKER",
    "ETF_TICKER",
    "SYMBOL",
    "FUND_TICKER",
]

ETF_ID_CANDIDATES = [
    "ETF_ID",
    "SERIES_ID",
]

ETF_NAME_CANDIDATES = [
    "ETF_NAME",
    "SERIES_NAME",
    "FUND_NAME",
    "NAME",
]

RETURN_CANDIDATES = [
    "RETURN",
    "ETF_RETURN",
    "MONTHLY_RETURN",
    "RET",
    "TOTAL_RETURN",
    "ADJ_RETURN",
]

CPU_CANDIDATES = [
    "CPU",
    "CPU_INDEX",
    "CLIMATE_POLICY_UNCERTAINTY",
    "CLIMATE_POLICY_UNCERTAINTY_INDEX",
    "GAVRIILIDIS_CPU",
]

MARKET_RETURN_CANDIDATES = [
    "MARKET_RETURN",
    "SPY_RETURN",
    "SPY",
    "MKT_RETURN",
    "MKT",
]

ENERGY_RETURN_CANDIDATES = [
    "ENERGY_RETURN",
    "XLE_RETURN",
    "XLE",
]

TREASURY_RETURN_CANDIDATES = [
    "TREASURY_RETURN",
    "TLT_RETURN",
    "TLT",
    "BOND_RETURN",
]


# ============================================================
# 7. ARCHITECTURE DEĞİŞKENLERİ
# ============================================================

ARCHITECTURE_VARIABLES = [
    "CORE_TRANSITION_SENSITIVITY_MAIN",
    "CORE_TRANSITION_SENSITIVITY_MODERATE",
    "EXTENDED_TRANSITION_SENSITIVITY_MAIN",
    "EXTENDED_TRANSITION_SENSITIVITY_MODERATE",
    "FINANCIAL_RESILIENCE_MAIN",
    "FINANCING_VULNERABILITY_MAIN",
    "GROWTH_DURATION_EXPOSURE_MAIN",
    "PORTFOLIO_CONCENTRATION_MAIN",
]

STATIC_CONTROL_VARIABLES = [
    "FINANCIAL_MATCH_WEIGHT",
    "HHI",
    "TOP10_WEIGHT",
    "EFFECTIVE_NUMBER_OF_HOLDINGS",
]


# ============================================================
# 8. GENEL YARDIMCI FONKSİYONLAR
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
            f"Dosya bulunamadı:\n{path}"
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
    Sayısal dönüşüm yapar.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def first_existing_column(
    columns,
    candidates,
):
    """
    Aday sütunlardan veri setinde ilk bulunanı döndürür.
    """

    available = set(columns)

    for candidate in candidates:

        if candidate in available:
            return candidate

    return None


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> None:
    """
    Zorunlu sütunları kontrol eder.
    """

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{dataset_name} içinde eksik sütunlar:\n"
            + "\n".join(missing)
        )


def standardize_month(
    series: pd.Series,
) -> pd.Series:
    """
    Tarihleri ayın son gününe standardize eder.
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


def clean_ticker(
    series: pd.Series,
) -> pd.Series:
    """
    Ticker değerlerini temizler.
    """

    result = (
        series.astype("string")
        .str.strip()
        .str.upper()
    )

    invalid = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "<NA>",
    }

    return result.mask(
        result.isin(invalid),
        pd.NA,
    )


def zscore(
    series: pd.Series,
) -> pd.Series:
    """
    Z-score standardizasyonu.
    """

    numeric = safe_numeric(series)

    mean_value = numeric.mean()
    std_value = numeric.std(ddof=1)

    if (
        pd.isna(std_value)
        or std_value <= 0
    ):
        return pd.Series(
            np.nan,
            index=series.index,
        )

    return (
        numeric - mean_value
    ) / std_value


# ============================================================
# 9. PROJE DOSYA ENVANTERİ
# ============================================================

def list_candidate_files() -> list[Path]:
    """
    Proje klasöründeki CSV ve Parquet dosyalarını listeler.
    """

    files = []

    for extension in [
        "*.csv",
        "*.parquet",
    ]:

        files.extend(
            PROJECT_DIR.rglob(extension)
        )

    excluded_prefixes = (
        "19_",
    )

    files = [
        file
        for file in files
        if not file.name.startswith(
            excluded_prefixes
        )
    ]

    return sorted(
        set(files)
    )


def inspect_file(
    path: Path,
) -> dict:
    """
    Dosyanın sütunlarını hızlı biçimde inceler.
    """

    try:

        if path.suffix.lower() == ".csv":

            sample = pd.read_csv(
                path,
                nrows=10,
                low_memory=False,
            )

        else:

            sample = pd.read_parquet(
                path
            ).head(10)

        sample = normalize_columns(
            sample
        )

        columns = list(
            sample.columns
        )

        return {
            "PATH": str(path),
            "FILE_NAME": path.name,
            "FILE_TYPE": path.suffix.lower(),
            "SIZE_BYTES": path.stat().st_size,
            "N_COLUMNS": len(columns),
            "HAS_DATE": int(
                first_existing_column(
                    columns,
                    DATE_CANDIDATES,
                )
                is not None
            ),
            "HAS_TICKER": int(
                first_existing_column(
                    columns,
                    TICKER_CANDIDATES,
                )
                is not None
            ),
            "HAS_ETF_ID": int(
                first_existing_column(
                    columns,
                    ETF_ID_CANDIDATES,
                )
                is not None
            ),
            "HAS_RETURN": int(
                first_existing_column(
                    columns,
                    RETURN_CANDIDATES,
                )
                is not None
            ),
            "HAS_CPU": int(
                first_existing_column(
                    columns,
                    CPU_CANDIDATES,
                )
                is not None
            ),
            "HAS_MARKET": int(
                first_existing_column(
                    columns,
                    MARKET_RETURN_CANDIDATES,
                )
                is not None
            ),
            "HAS_ENERGY": int(
                first_existing_column(
                    columns,
                    ENERGY_RETURN_CANDIDATES,
                )
                is not None
            ),
            "HAS_TREASURY": int(
                first_existing_column(
                    columns,
                    TREASURY_RETURN_CANDIDATES,
                )
                is not None
            ),
            "COLUMNS": " | ".join(columns),
        }

    except Exception as error:

        return {
            "PATH": str(path),
            "FILE_NAME": path.name,
            "FILE_TYPE": path.suffix.lower(),
            "SIZE_BYTES": path.stat().st_size,
            "N_COLUMNS": np.nan,
            "HAS_DATE": 0,
            "HAS_TICKER": 0,
            "HAS_ETF_ID": 0,
            "HAS_RETURN": 0,
            "HAS_CPU": 0,
            "HAS_MARKET": 0,
            "HAS_ENERGY": 0,
            "HAS_TREASURY": 0,
            "COLUMNS": f"READ_ERROR: {error}",
        }


def build_file_inventory() -> pd.DataFrame:
    """
    Projedeki veri dosyalarının envanterini oluşturur.
    """

    files = list_candidate_files()

    rows = [
        inspect_file(file)
        for file in files
    ]

    inventory = pd.DataFrame(rows)

    inventory.to_csv(
        INPUT_INVENTORY_FILE,
        index=False,
    )

    return inventory


def find_best_file(
    inventory: pd.DataFrame,
    required_flags: list[str],
    preferred_keywords: list[str],
) -> Path | None:
    """
    Gerekli sütun özelliklerini taşıyan en uygun dosyayı seçer.
    """

    candidates = inventory.copy()

    for flag in required_flags:

        candidates = candidates.loc[
            candidates[flag] == 1
        ]

    if candidates.empty:
        return None

    candidates = candidates.copy()

    candidates["KEYWORD_SCORE"] = 0

    for keyword in preferred_keywords:

        candidates["KEYWORD_SCORE"] += (
            candidates["FILE_NAME"]
            .str.lower()
            .str.contains(
                keyword.lower(),
                regex=False,
            )
            .astype(int)
        )

    candidates = candidates.sort_values(
        by=[
            "KEYWORD_SCORE",
            "SIZE_BYTES",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return Path(
        candidates.iloc[0]["PATH"]
    )


# ============================================================
# 10. ARCHITECTURE PANELİ
# ============================================================

def prepare_architecture_panel(
    path: Path,
) -> pd.DataFrame:
    """
    Quality-adjusted architecture panelini hazırlar.
    """

    architecture = normalize_columns(
        read_data(path)
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
        "ARCHITECTURE PANEL",
    )

    architecture["ETF_ID"] = (
        architecture["ETF_ID"]
        .astype("string")
        .str.strip()
    )

    architecture["ETF_NAME"] = (
        architecture["ETF_NAME"]
        .astype("string")
        .str.strip()
    )

    available_columns = [
        "ACCESSION_NUMBER",
        "ETF_ID",
        "ETF_NAME",
    ]

    for column in (
        ARCHITECTURE_VARIABLES
        + STATIC_CONTROL_VARIABLES
    ):

        if column in architecture.columns:
            available_columns.append(column)

    architecture = architecture[
        list(dict.fromkeys(available_columns))
    ].copy()

    architecture[
        "MAIN_ARCHITECTURE_SAMPLE"
    ] = (
        architecture[
            "CORE_TRANSITION_SENSITIVITY_MAIN"
        ].notna()
    ).astype(int)

    architecture[
        "MODERATE_ARCHITECTURE_SAMPLE"
    ] = (
        architecture[
            "CORE_TRANSITION_SENSITIVITY_MODERATE"
        ].notna()
    ).astype(int)

    return architecture


# ============================================================
# 11. ETF TICKER MAPPING
# ============================================================

def prepare_ticker_mapping(
    path: Path,
) -> pd.DataFrame:
    """
    ETF ID veya ETF adı ile ticker arasında mapping oluşturur.
    """

    mapping = normalize_columns(
        read_data(path)
    )

    ticker_column = first_existing_column(
        mapping.columns,
        TICKER_CANDIDATES,
    )

    etf_id_column = first_existing_column(
        mapping.columns,
        ETF_ID_CANDIDATES,
    )

    etf_name_column = first_existing_column(
        mapping.columns,
        ETF_NAME_CANDIDATES,
    )

    if ticker_column is None:
        raise KeyError(
            f"Ticker mapping dosyasında ticker sütunu yok:\n{path}"
        )

    mapping["ETF_TICKER"] = clean_ticker(
        mapping[ticker_column]
    )

    keep_columns = [
        "ETF_TICKER",
    ]

    if etf_id_column is not None:

        mapping["ETF_ID"] = (
            mapping[etf_id_column]
            .astype("string")
            .str.strip()
        )

        keep_columns.append(
            "ETF_ID"
        )

    if etf_name_column is not None:

        mapping["ETF_NAME"] = (
            mapping[etf_name_column]
            .astype("string")
            .str.strip()
        )

        keep_columns.append(
            "ETF_NAME"
        )

    mapping = (
        mapping[keep_columns]
        .dropna(
            subset=["ETF_TICKER"]
        )
        .drop_duplicates()
    )

    return mapping


def attach_tickers(
    architecture: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Architecture ETF'lerine ticker ekler.
    """

    result = architecture.copy()

    if "ETF_ID" in mapping.columns:

        id_mapping = (
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

        result = result.merge(
            id_mapping,
            on="ETF_ID",
            how="left",
            validate="one_to_one",
        )

    else:

        result["ETF_TICKER"] = pd.NA

    if (
        "ETF_NAME" in mapping.columns
        and result["ETF_TICKER"].isna().any()
    ):

        name_mapping = (
            mapping[
                [
                    "ETF_NAME",
                    "ETF_TICKER",
                ]
            ]
            .dropna()
            .drop_duplicates(
                subset=["ETF_NAME"]
            )
            .rename(
                columns={
                    "ETF_TICKER": (
                        "ETF_TICKER_NAME_MATCH"
                    )
                }
            )
        )

        result = result.merge(
            name_mapping,
            on="ETF_NAME",
            how="left",
            validate="one_to_one",
        )

        result["ETF_TICKER"] = (
            result["ETF_TICKER"]
            .fillna(
                result[
                    "ETF_TICKER_NAME_MATCH"
                ]
            )
        )

        result = result.drop(
            columns=[
                "ETF_TICKER_NAME_MATCH"
            ]
        )

    return result


# ============================================================
# 12. ETF GETİRİLERİ
# ============================================================

def prepare_etf_returns(
    path: Path,
) -> pd.DataFrame:
    """
    Aylık ETF getirilerini standartlaştırır.
    """

    returns = normalize_columns(
        read_data(path)
    )

    date_column = first_existing_column(
        returns.columns,
        DATE_CANDIDATES,
    )

    ticker_column = first_existing_column(
        returns.columns,
        TICKER_CANDIDATES,
    )

    etf_id_column = first_existing_column(
        returns.columns,
        ETF_ID_CANDIDATES,
    )

    return_column = first_existing_column(
        returns.columns,
        RETURN_CANDIDATES,
    )

    if date_column is None:
        raise KeyError(
            f"ETF return dosyasında tarih sütunu bulunamadı:\n{path}"
        )

    if return_column is None:
        raise KeyError(
            f"ETF return dosyasında getiri sütunu bulunamadı:\n{path}"
        )

    if (
        ticker_column is None
        and etf_id_column is None
    ):
        raise KeyError(
            "ETF return dosyasında ticker veya ETF_ID bulunamadı."
        )

    returns["DATE"] = standardize_month(
        returns[date_column]
    )

    returns["ETF_RETURN"] = safe_numeric(
        returns[return_column]
    )

    if ticker_column is not None:

        returns["ETF_TICKER"] = clean_ticker(
            returns[ticker_column]
        )

    if etf_id_column is not None:

        returns["ETF_ID"] = (
            returns[etf_id_column]
            .astype("string")
            .str.strip()
        )

    # Yüzde birimindeki getirileri decimal forma çevir.
    return_abs_95 = (
        returns["ETF_RETURN"]
        .abs()
        .quantile(0.95)
    )

    if pd.notna(return_abs_95) and return_abs_95 > 2:

        returns["ETF_RETURN"] = (
            returns["ETF_RETURN"] / 100.0
        )

    keep_columns = [
        "DATE",
        "ETF_RETURN",
    ]

    if "ETF_TICKER" in returns.columns:
        keep_columns.append("ETF_TICKER")

    if "ETF_ID" in returns.columns:
        keep_columns.append("ETF_ID")

    returns = returns[
        keep_columns
    ].copy()

    returns = returns.loc[
        returns["DATE"].notna()
        & returns["ETF_RETURN"].notna()
    ]

    returns = returns.loc[
        returns["DATE"].dt.year.between(
            MIN_YEAR,
            MAX_YEAR,
        )
    ]

    identifier_columns = [
        column
        for column in [
            "ETF_ID",
            "ETF_TICKER",
        ]
        if column in returns.columns
    ]

    returns = (
        returns.groupby(
            identifier_columns + ["DATE"],
            dropna=False,
            as_index=False,
        )["ETF_RETURN"]
        .mean()
    )

    return returns


# ============================================================
# 13. CPU VERİSİ
# ============================================================

def prepare_cpu(
    path: Path,
) -> pd.DataFrame:
    """
    Aylık CPU serisini hazırlar.
    """

    cpu = normalize_columns(
        read_data(path)
    )

    date_column = first_existing_column(
        cpu.columns,
        DATE_CANDIDATES,
    )

    cpu_column = first_existing_column(
        cpu.columns,
        CPU_CANDIDATES,
    )

    if date_column is None:
        raise KeyError(
            f"CPU dosyasında tarih sütunu bulunamadı:\n{path}"
        )

    if cpu_column is None:
        raise KeyError(
            f"CPU dosyasında CPU sütunu bulunamadı:\n{path}"
        )

    cpu["DATE"] = standardize_month(
        cpu[date_column]
    )

    cpu["CPU"] = safe_numeric(
        cpu[cpu_column]
    )

    cpu = (
        cpu[
            [
                "DATE",
                "CPU",
            ]
        ]
        .dropna()
        .groupby(
            "DATE",
            as_index=False,
        )
        .mean()
    )

    cpu = cpu.loc[
        cpu["DATE"].dt.year.between(
            MIN_YEAR,
            MAX_YEAR,
        )
    ].copy()

    cpu["LOG_CPU"] = np.log1p(
        cpu["CPU"].clip(lower=0)
    )

    cpu["CPU_CHANGE"] = (
        cpu["CPU"].pct_change(
            fill_method=None
        )
    )

    cpu["CPU_Z"] = zscore(
        cpu["CPU"]
    )

    cpu["LOG_CPU_Z"] = zscore(
        cpu["LOG_CPU"]
    )

    cpu["CPU_L1"] = cpu[
        "CPU"
    ].shift(1)

    cpu["CPU_Z_L1"] = cpu[
        "CPU_Z"
    ].shift(1)

    cpu["HIGH_CPU_REGIME"] = (
        cpu["CPU"]
        >= cpu["CPU"].quantile(0.75)
    ).astype(int)

    cpu["LOW_CPU_REGIME"] = (
        cpu["CPU"]
        <= cpu["CPU"].quantile(0.25)
    ).astype(int)

    return cpu


# ============================================================
# 14. KONTROL DEĞİŞKENLERİ
# ============================================================

def prepare_controls(
    path: Path,
) -> pd.DataFrame:
    """
    SPY, XLE ve TLT aylık getirilerini hazırlar.
    """

    controls = normalize_columns(
        read_data(path)
    )

    date_column = first_existing_column(
        controls.columns,
        DATE_CANDIDATES,
    )

    if date_column is None:
        raise KeyError(
            f"Kontrol dosyasında tarih sütunu bulunamadı:\n{path}"
        )

    controls["DATE"] = standardize_month(
        controls[date_column]
    )

    output = pd.DataFrame(
        {
            "DATE": controls["DATE"]
        }
    )

    variable_candidates = {
        "MARKET_RETURN": (
            MARKET_RETURN_CANDIDATES
        ),
        "ENERGY_RETURN": (
            ENERGY_RETURN_CANDIDATES
        ),
        "TREASURY_RETURN": (
            TREASURY_RETURN_CANDIDATES
        ),
    }

    for output_name, candidates in (
        variable_candidates.items()
    ):

        source_column = first_existing_column(
            controls.columns,
            candidates,
        )

        if source_column is not None:

            output[output_name] = safe_numeric(
                controls[source_column]
            )

            value_95 = (
                output[output_name]
                .abs()
                .quantile(0.95)
            )

            if (
                pd.notna(value_95)
                and value_95 > 2
            ):

                output[output_name] = (
                    output[output_name]
                    / 100.0
                )

        else:

            output[output_name] = np.nan

    output = (
        output.groupby(
            "DATE",
            as_index=False,
        )
        .mean()
    )

    output = output.loc[
        output["DATE"].dt.year.between(
            MIN_YEAR,
            MAX_YEAR,
        )
    ]

    return output


# ============================================================
# 15. PANEL BİRLEŞTİRME
# ============================================================

def merge_returns_and_architecture(
    returns: pd.DataFrame,
    architecture: pd.DataFrame,
) -> pd.DataFrame:
    """
    ETF getirilerini architecture verisiyle birleştirir.
    """

    if (
        "ETF_ID" in returns.columns
        and returns["ETF_ID"].notna().any()
    ):

        panel = returns.merge(
            architecture,
            on="ETF_ID",
            how="inner",
            validate="many_to_one",
            suffixes=(
                "",
                "_ARCH",
            ),
        )

        panel["RETURN_MATCH_METHOD"] = (
            "ETF_ID"
        )

        return panel

    if (
        "ETF_TICKER" not in returns.columns
        or "ETF_TICKER" not in architecture.columns
    ):

        raise RuntimeError(
            "Getiri ve architecture paneli arasında ortak "
            "ETF_ID veya ETF_TICKER bulunamadı."
        )

    architecture_ticker = (
        architecture.loc[
            architecture[
                "ETF_TICKER"
            ].notna()
        ]
        .drop_duplicates(
            subset=["ETF_TICKER"]
        )
    )

    panel = returns.merge(
        architecture_ticker,
        on="ETF_TICKER",
        how="inner",
        validate="many_to_one",
    )

    panel[
        "RETURN_MATCH_METHOD"
    ] = "ETF_TICKER"

    return panel


def add_interaction_variables(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    CPU × architecture interaction değişkenlerini oluşturur.
    """

    result = panel.copy()

    available_architecture_variables = [
        variable
        for variable in ARCHITECTURE_VARIABLES
        if variable in result.columns
    ]

    for variable in (
        available_architecture_variables
    ):

        result[
            f"CPU_X_{variable}"
        ] = (
            result["CPU_Z"]
            * result[variable]
        )

        result[
            f"HIGH_CPU_X_{variable}"
        ] = (
            result["HIGH_CPU_REGIME"]
            * result[variable]
        )

        result[
            f"CPU_L1_X_{variable}"
        ] = (
            result["CPU_Z_L1"]
            * result[variable]
        )

    return result


# ============================================================
# 16. RAPORLAR
# ============================================================

def build_etf_coverage(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    ETF bazında zaman serisi coverage raporu oluşturur.
    """

    rows = []

    for (
        etf_id,
        group,
    ) in panel.groupby(
        "ETF_ID",
        dropna=False,
    ):

        rows.append(
            {
                "ETF_ID": etf_id,
                "ETF_NAME": (
                    group[
                        "ETF_NAME"
                    ].iloc[0]
                    if "ETF_NAME"
                    in group.columns
                    else pd.NA
                ),
                "ETF_TICKER": (
                    group[
                        "ETF_TICKER"
                    ].iloc[0]
                    if "ETF_TICKER"
                    in group.columns
                    else pd.NA
                ),
                "N_MONTHS": len(group),
                "START_DATE": (
                    group["DATE"].min()
                ),
                "END_DATE": (
                    group["DATE"].max()
                ),
                "RETURN_COVERAGE": float(
                    group[
                        "ETF_RETURN"
                    ].notna().mean()
                ),
                "CPU_COVERAGE": float(
                    group["CPU"].notna().mean()
                ),
                "CONTROL_COVERAGE": float(
                    group[
                        [
                            "MARKET_RETURN",
                            "ENERGY_RETURN",
                            "TREASURY_RETURN",
                        ]
                    ]
                    .notna()
                    .all(axis=1)
                    .mean()
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
            }
        )

    return pd.DataFrame(rows)


def build_monthly_coverage(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ay bazında ETF coverage raporu oluşturur.
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
            N_OBSERVATIONS=(
                "ETF_RETURN",
                "size",
            ),
            RETURN_COVERAGE=(
                "ETF_RETURN",
                lambda x: x.notna().mean(),
            ),
            CPU_COVERAGE=(
                "CPU",
                lambda x: x.notna().mean(),
            ),
            MARKET_COVERAGE=(
                "MARKET_RETURN",
                lambda x: x.notna().mean(),
            ),
        )
    )


def build_variable_dictionary() -> pd.DataFrame:
    """
    Panel değişken sözlüğünü oluşturur.
    """

    rows = [
        {
            "VARIABLE": "ETF_RETURN",
            "TYPE": "Dependent variable",
            "FORMULA": (
                "Monthly decimal ETF total return"
            ),
            "PURPOSE": (
                "Conditional ETF pricing"
            ),
        },
        {
            "VARIABLE": "CPU",
            "TYPE": "Climate-policy shock",
            "FORMULA": (
                "Monthly Climate Policy Uncertainty index"
            ),
            "PURPOSE": (
                "Primary uncertainty state variable"
            ),
        },
        {
            "VARIABLE": "CPU_Z",
            "TYPE": "Standardized shock",
            "FORMULA": (
                "(CPU - mean CPU) / standard deviation CPU"
            ),
            "PURPOSE": (
                "Comparable interaction interpretation"
            ),
        },
        {
            "VARIABLE": "HIGH_CPU_REGIME",
            "TYPE": "Regime indicator",
            "FORMULA": (
                "1 when CPU is at or above the 75th percentile"
            ),
            "PURPOSE": (
                "Stress-activation test for RQ3"
            ),
        },
        {
            "VARIABLE": "MARKET_RETURN",
            "TYPE": "Control",
            "FORMULA": "Monthly SPY return",
            "PURPOSE": "Broad market control",
        },
        {
            "VARIABLE": "ENERGY_RETURN",
            "TYPE": "Control",
            "FORMULA": "Monthly XLE return",
            "PURPOSE": "Energy-sector control",
        },
        {
            "VARIABLE": "TREASURY_RETURN",
            "TYPE": "Control",
            "FORMULA": "Monthly TLT return",
            "PURPOSE": (
                "Long-duration and interest-rate control"
            ),
        },
    ]

    for variable in ARCHITECTURE_VARIABLES:

        rows.append(
            {
                "VARIABLE": variable,
                "TYPE": (
                    "Time-invariant portfolio architecture"
                ),
                "FORMULA": (
                    "Quality-adjusted holdings-based architecture score"
                ),
                "PURPOSE": (
                    "Cross-sectional exposure channel"
                ),
            }
        )

        rows.append(
            {
                "VARIABLE": (
                    f"CPU_X_{variable}"
                ),
                "TYPE": "Interaction",
                "FORMULA": (
                    f"CPU_Z × {variable}"
                ),
                "PURPOSE": (
                    "Primary conditional-pricing coefficient"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 17. VALIDATION
# ============================================================

def build_validation(
    panel: pd.DataFrame,
    etf_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Panelin temel doğrulama kontrollerini yapar.
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
        panel.loc[
            panel[
                "MAIN_ARCHITECTURE_SAMPLE"
            ] == 1,
            "ETF_ID",
        ].nunique()
    )

    moderate_etfs = int(
        panel.loc[
            panel[
                "MODERATE_ARCHITECTURE_SAMPLE"
            ] == 1,
            "ETF_ID",
        ].nunique()
    )

    rows = [
        {
            "CHECK": "PANEL_ROWS",
            "VALUE": len(panel),
            "PASS": int(
                len(panel) > 0
            ),
        },
        {
            "CHECK": "UNIQUE_ETFS",
            "VALUE": panel[
                "ETF_ID"
            ].nunique(),
            "PASS": int(
                panel[
                    "ETF_ID"
                ].nunique() > 0
            ),
        },
        {
            "CHECK": "UNIQUE_MONTHS",
            "VALUE": panel[
                "DATE"
            ].nunique(),
            "PASS": int(
                panel[
                    "DATE"
                ].nunique() >= 24
            ),
        },
        {
            "CHECK": "DUPLICATE_ETF_MONTH_ROWS",
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
                moderate_etfs >= main_etfs
            ),
        },
        {
            "CHECK": (
                "ETFS_WITH_AT_LEAST_24_MONTHS"
            ),
            "VALUE": int(
                (
                    etf_coverage[
                        "N_MONTHS"
                    ]
                    >= MIN_MONTHS_PER_ETF
                ).sum()
            ),
            "PASS": int(
                (
                    etf_coverage[
                        "N_MONTHS"
                    ]
                    >= MIN_MONTHS_PER_ETF
                ).sum()
                > 0
            ),
        },
        {
            "CHECK": "CPU_MISSING_ROWS",
            "VALUE": int(
                panel["CPU"].isna().sum()
            ),
            "PASS": int(
                panel["CPU"].isna().sum()
                == 0
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# 18. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 78)
    print("19 - CPU RETURN ARCHITECTURE PANEL")
    print("=" * 78)

    # --------------------------------------------------------
    # 1. Architecture
    # --------------------------------------------------------

    print(
        "\n1/10 - Quality-adjusted architecture paneli okunuyor..."
    )

    architecture = prepare_architecture_panel(
        ARCHITECTURE_FILE
    )

    print(
        f"Architecture ETF sayısı: "
        f"{architecture['ETF_ID'].nunique():,}"
    )

    # --------------------------------------------------------
    # 2. File inventory
    # --------------------------------------------------------

    print(
        "\n2/10 - Proje veri dosyaları taranıyor..."
    )

    inventory = build_file_inventory()

    # --------------------------------------------------------
    # 3. Return file
    # --------------------------------------------------------

    return_path = (
        Path(ETF_RETURN_FILE)
        if ETF_RETURN_FILE is not None
        else find_best_file(
            inventory=inventory,
            required_flags=[
                "HAS_DATE",
                "HAS_RETURN",
            ],
            preferred_keywords=[
                "etf",
                "return",
                "monthly",
                "panel",
            ],
        )
    )

    # --------------------------------------------------------
    # 4. CPU file
    # --------------------------------------------------------

    cpu_path = (
        Path(CPU_FILE)
        if CPU_FILE is not None
        else find_best_file(
            inventory=inventory,
            required_flags=[
                "HAS_DATE",
                "HAS_CPU",
            ],
            preferred_keywords=[
                "cpu",
                "climate",
                "uncertainty",
                "monthly",
            ],
        )
    )

    # --------------------------------------------------------
    # 5. Control file
    # --------------------------------------------------------

    control_path = (
        Path(CONTROL_FILE)
        if CONTROL_FILE is not None
        else find_best_file(
            inventory=inventory,
            required_flags=[
                "HAS_DATE",
                "HAS_MARKET",
            ],
            preferred_keywords=[
                "control",
                "spy",
                "xle",
                "tlt",
                "monthly",
                "return",
            ],
        )
    )

    # --------------------------------------------------------
    # 6. Ticker mapping
    # --------------------------------------------------------

    mapping_path = (
        Path(ETF_TICKER_MAPPING_FILE)
        if ETF_TICKER_MAPPING_FILE is not None
        else find_best_file(
            inventory=inventory,
            required_flags=[
                "HAS_TICKER",
            ],
            preferred_keywords=[
                "master",
                "mapping",
                "etf",
                "ticker",
                "universe",
            ],
        )
    )

    print("\nOtomatik seçilen dosyalar:")

    print(
        f"ETF returns: "
        f"{return_path}"
    )

    print(
        f"CPU: "
        f"{cpu_path}"
    )

    print(
        f"Controls: "
        f"{control_path}"
    )

    print(
        f"Ticker mapping: "
        f"{mapping_path}"
    )

    missing_inputs = []

    if return_path is None:
        missing_inputs.append(
            "ETF aylık getiri dosyası"
        )

    if cpu_path is None:
        missing_inputs.append(
            "CPU aylık veri dosyası"
        )

    if control_path is None:
        missing_inputs.append(
            "SPY/XLE/TLT kontrol dosyası"
        )

    # Mapping yalnızca ETF return dosyası ticker tabanlıysa gerekli
    # olacağından şimdilik zorunlu tutulmaz.

    if missing_inputs:

        print(
            "\nGEREKLİ GİRDİLERİN BİR KISMI OTOMATİK BULUNAMADI"
        )

        print("=" * 78)

        for item in missing_inputs:
            print(f"- {item}")

        print(
            "\nAyrıntılı dosya envanteri oluşturuldu:"
        )

        print(INPUT_INVENTORY_FILE)

        print(
            "\n19_build_cpu_return_panel.py dosyasının başındaki "
            "ETF_RETURN_FILE, CPU_FILE ve CONTROL_FILE yollarını "
            "gerçek dosya adlarıyla doldur."
        )

        return

    # --------------------------------------------------------
    # 7. Read inputs
    # --------------------------------------------------------

    print(
        "\n3/10 - ETF aylık getirileri hazırlanıyor..."
    )

    returns = prepare_etf_returns(
        return_path
    )

    print(
        f"ETF return satırı: "
        f"{len(returns):,}"
    )

    if (
        "ETF_ID" not in returns.columns
        or returns["ETF_ID"].isna().all()
    ):

        if mapping_path is None:

            raise RuntimeError(
                "ETF return dosyası ticker tabanlı; fakat ETF_ID–ticker "
                "mapping dosyası bulunamadı."
            )

        print(
            "\n4/10 - ETF ticker mapping hazırlanıyor..."
        )

        ticker_mapping = prepare_ticker_mapping(
            mapping_path
        )

        architecture = attach_tickers(
            architecture=architecture,
            mapping=ticker_mapping,
        )

    else:

        print(
            "\n4/10 - Return dosyasında ETF_ID bulundu; "
            "ticker mapping gerekmiyor."
        )

    print(
        "\n5/10 - CPU aylık serisi hazırlanıyor..."
    )

    cpu = prepare_cpu(
        cpu_path
    )

    print(
        f"CPU ay sayısı: "
        f"{len(cpu):,}"
    )

    print(
        "\n6/10 - Piyasa kontrol değişkenleri hazırlanıyor..."
    )

    controls = prepare_controls(
        control_path
    )

    # --------------------------------------------------------
    # 8. Merge
    # --------------------------------------------------------

    print(
        "\n7/10 - ETF getirileri architecture ile birleştiriliyor..."
    )

    panel = merge_returns_and_architecture(
        returns=returns,
        architecture=architecture,
    )

    print(
        f"Architecture ile eşleşen ETF sayısı: "
        f"{panel['ETF_ID'].nunique():,}"
    )

    print(
        "\n8/10 - CPU ve kontrol değişkenleri ekleniyor..."
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
        panel["DATE"].dt.year.between(
            MIN_YEAR,
            MAX_YEAR,
        )
    ].copy()

    panel = panel.sort_values(
        by=[
            "ETF_ID",
            "DATE",
        ]
    )

    # --------------------------------------------------------
    # 9. Interactions
    # --------------------------------------------------------

    print(
        "\n9/10 - CPU × architecture interactionları oluşturuluyor..."
    )

    panel = add_interaction_variables(
        panel
    )

    panel["YEAR"] = (
        panel["DATE"].dt.year
    )

    panel["MONTH"] = (
        panel["DATE"].dt.month
    )

    panel["YEAR_MONTH"] = (
        panel["DATE"]
        .dt.to_period("M")
        .astype(str)
    )

    panel[
        "VALID_RETURN_CPU_ROW"
    ] = (
        panel[
            [
                "ETF_RETURN",
                "CPU",
            ]
        ]
        .notna()
        .all(axis=1)
    ).astype(int)

    panel[
        "VALID_FULL_CONTROL_ROW"
    ] = (
        panel[
            [
                "ETF_RETURN",
                "CPU",
                "MARKET_RETURN",
                "ENERGY_RETURN",
                "TREASURY_RETURN",
            ]
        ]
        .notna()
        .all(axis=1)
    ).astype(int)

    # --------------------------------------------------------
    # 10. Reports and outputs
    # --------------------------------------------------------

    print(
        "\n10/10 - Panel ve diagnostics kaydediliyor..."
    )

    etf_coverage = build_etf_coverage(
        panel
    )

    monthly_coverage = build_monthly_coverage(
        panel
    )

    etf_sample = (
        architecture[
            [
                column
                for column in [
                    "ETF_ID",
                    "ETF_NAME",
                    "ETF_TICKER",
                    "FINANCIAL_MATCH_WEIGHT",
                    "MAIN_ARCHITECTURE_SAMPLE",
                    "MODERATE_ARCHITECTURE_SAMPLE",
                    "CORE_TRANSITION_SENSITIVITY_MAIN",
                    "CORE_TRANSITION_SENSITIVITY_MODERATE",
                    "EXTENDED_TRANSITION_SENSITIVITY_MAIN",
                ]
                if column in architecture.columns
            ]
        ]
        .copy()
    )

    diagnostics = pd.DataFrame(
        [
            {
                "METRIC": (
                    "ARCHITECTURE_ETFS"
                ),
                "VALUE": (
                    architecture[
                        "ETF_ID"
                    ].nunique()
                ),
            },
            {
                "METRIC": (
                    "RETURN_ETFS"
                ),
                "VALUE": (
                    returns[
                        "ETF_ID"
                    ].nunique()
                    if "ETF_ID"
                    in returns.columns
                    else returns[
                        "ETF_TICKER"
                    ].nunique()
                ),
            },
            {
                "METRIC": (
                    "MATCHED_PANEL_ETFS"
                ),
                "VALUE": (
                    panel[
                        "ETF_ID"
                    ].nunique()
                ),
            },
            {
                "METRIC": "PANEL_ROWS",
                "VALUE": len(panel),
            },
            {
                "METRIC": (
                    "PANEL_MONTHS"
                ),
                "VALUE": (
                    panel[
                        "DATE"
                    ].nunique()
                ),
            },
            {
                "METRIC": (
                    "MAIN_QUALITY_ETFS"
                ),
                "VALUE": (
                    panel.loc[
                        panel[
                            "MAIN_ARCHITECTURE_SAMPLE"
                        ]
                        == 1,
                        "ETF_ID",
                    ].nunique()
                ),
            },
            {
                "METRIC": (
                    "MODERATE_QUALITY_ETFS"
                ),
                "VALUE": (
                    panel.loc[
                        panel[
                            "MODERATE_ARCHITECTURE_SAMPLE"
                        ]
                        == 1,
                        "ETF_ID",
                    ].nunique()
                ),
            },
            {
                "METRIC": (
                    "VALID_RETURN_CPU_ROWS"
                ),
                "VALUE": (
                    panel[
                        "VALID_RETURN_CPU_ROW"
                    ].sum()
                ),
            },
            {
                "METRIC": (
                    "VALID_FULL_CONTROL_ROWS"
                ),
                "VALUE": (
                    panel[
                        "VALID_FULL_CONTROL_ROW"
                    ].sum()
                ),
            },
        ]
    )

    validation = build_validation(
        panel=panel,
        etf_coverage=etf_coverage,
    )

    variable_dictionary = (
        build_variable_dictionary()
    )

    panel.to_csv(
        PANEL_FILE,
        index=False,
    )

    panel.to_parquet(
        PANEL_PARQUET_FILE,
        index=False,
    )

    etf_sample.to_csv(
        ETF_SAMPLE_FILE,
        index=False,
    )

    monthly_coverage.to_csv(
        MONTHLY_COVERAGE_FILE,
        index=False,
    )

    etf_coverage.to_csv(
        ETF_COVERAGE_FILE,
        index=False,
    )

    diagnostics.to_csv(
        MERGE_DIAGNOSTICS_FILE,
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

    print(
        "\nCPU–RETURN–ARCHITECTURE PANELİ HAZIR"
    )

    print("=" * 78)

    print(
        "\nPanel diagnostics:"
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    print(
        "\nETF zaman serisi coverage özeti:"
    )

    print(
        etf_coverage[
            "N_MONTHS"
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
        validation["PASS"] == 0
    ]

    print(
        "\nBaşarısız kontrol sayısı: "
        f"{len(failed_checks):,}"
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(PANEL_FILE)
    print(ETF_SAMPLE_FILE)
    print(ETF_COVERAGE_FILE)
    print(MONTHLY_COVERAGE_FILE)
    print(MERGE_DIAGNOSTICS_FILE)
    print(VALIDATION_FILE)
    print(VARIABLE_DICTIONARY_FILE)


if __name__ == "__main__":
    main()