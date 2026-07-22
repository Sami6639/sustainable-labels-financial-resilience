from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 24b_finalize_architecture_channels.py
#
# AMAÇ
# ----
# Portfolio architecture kanallarını nihai teorik ve ampirik
# yapıya dönüştürmek.
#
# ANA KANALLAR
# ------------
# 1. INTERNAL_FINANCIAL_CAPACITY
#       ROA + Cash Ratio
#
# 2. EXTERNAL_FINANCING_DEPENDENCE
#       External Finance Dependence
#
# 3. GROWTH_DURATION_EXPOSURE
#       CapEx + R&D + Revenue Growth
#
# 4. PORTFOLIO_CONCENTRATION
#       HHI + Top10 Weight - Effective Number of Holdings
#
# RESTRICTED ROBUSTNESS
# ---------------------
# 5. LEVERAGE_EXPOSURE
#       Leverage
#
# ANA FINANCIAL ARCHITECTURE RISK
# -------------------------------
# - Internal Financial Capacity
# + External Financing Dependence
# + Growth-Duration Exposure
#
# Concentration ana financial composite'e dahil edilmez.
#
# EXTENDED ARCHITECTURE RISK
# --------------------------
# Financial Architecture Risk
# + Portfolio Concentration
#
# PCA
# ---
# Capacity, external finance dependence, growth-duration ve
# concentration kanalları üzerinde complete-case kesitte uygulanır.
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
# 2. GİRDİ DOSYASI
# ============================================================

INPUT_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_architecture_factors.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

FINAL_PANEL_FILE = (
    OUTPUT_DIR
    / "24b_final_architecture_channels.csv"
)

FINAL_PANEL_PARQUET_FILE = (
    OUTPUT_DIR
    / "24b_final_architecture_channels.parquet"
)

QUALITY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_quality_summary.csv"
)

DESCRIPTIVE_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_descriptive_statistics.csv"
)

CORRELATION_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_correlations.csv"
)

PCA_LOADINGS_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_pca_loadings.csv"
)

PCA_VARIANCE_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_pca_explained_variance.csv"
)

PCA_SCORES_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_pca_scores.csv"
)

RANKINGS_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_rankings.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "24b_final_channel_variable_dictionary.csv"
)


# ============================================================
# 4. KALİTE EŞİKLERİ
# ============================================================

MODERATE_MATCH_THRESHOLD = 0.70
MAIN_MATCH_THRESHOLD = 0.80
STRICT_MATCH_THRESHOLD = 0.90

MODERATE_COVERAGE_THRESHOLD = 0.50
MAIN_COVERAGE_THRESHOLD = 0.70
STRICT_COVERAGE_THRESHOLD = 0.80


# ============================================================
# 5. ANA KANALLAR
# ============================================================

PRIMARY_CHANNELS = [
    "INTERNAL_FINANCIAL_CAPACITY",
    "EXTERNAL_FINANCING_DEPENDENCE",
    "GROWTH_DURATION_EXPOSURE_FINAL",
    "PORTFOLIO_CONCENTRATION_FINAL",
]

ROBUSTNESS_CHANNELS = [
    "LEVERAGE_EXPOSURE",
]

COMPOSITE_CHANNELS = [
    "FINANCIAL_ARCHITECTURE_RISK_FINAL",
    "EXTENDED_ARCHITECTURE_RISK_FINAL",
]


# ============================================================
# 6. YARDIMCI FONKSİYONLAR
# ============================================================

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


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
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
            "Eksik sütunlar:\n"
            + "\n".join(
                missing
            )
        )


def row_mean_with_minimum(
    df: pd.DataFrame,
    columns: list[str],
    minimum_components: int,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Satır bazında eşit ağırlıklı ortalama üretir.
    """

    available_count = (
        df[
            columns
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    available_share = (
        available_count
        / len(
            columns
        )
    )

    score = (
        df[
            columns
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    score = score.where(
        available_count
        >= minimum_components
    )

    return (
        score,
        available_count,
        available_share,
    )


def quality_adjusted_score(
    raw_score: pd.Series,
    valid_flag: pd.Series,
) -> pd.Series:
    """
    Yalnızca geçerli kalite bayrağına sahip gözlemlerde skor tutar.
    """

    return raw_score.where(
        valid_flag == 1
    )


# ============================================================
# 7. ANA KANALLARIN OLUŞTURULMASI
# ============================================================

def build_final_channels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nihai ve non-overlapping architecture kanallarını oluşturur.
    """

    result = df.copy()

    # --------------------------------------------------------
    # 1. Internal Financial Capacity
    #
    # Script 24'te temiz biçimde oluşturuldu.
    # --------------------------------------------------------

    result[
        "INTERNAL_FINANCIAL_CAPACITY"
    ] = safe_numeric(
        result[
            "INTERNAL_FINANCIAL_CAPACITY"
        ]
    )

    # --------------------------------------------------------
    # 2. External Financing Dependence
    #
    # Tek değişkenli ana kanal.
    # Script 24'te oluşturulan standardize kaynak kullanılır.
    # --------------------------------------------------------

    result[
        "EXTERNAL_FINANCING_DEPENDENCE"
    ] = safe_numeric(
        result[
            "RB_Z_PW_EXTERNAL_FINANCE_DEPENDENCE"
        ]
    )

    # --------------------------------------------------------
    # 3. Growth-Duration Exposure
    # --------------------------------------------------------

    result[
        "GROWTH_DURATION_EXPOSURE_FINAL"
    ] = safe_numeric(
        result[
            "GROWTH_DURATION_EXPOSURE_REBUILT"
        ]
    )

    # --------------------------------------------------------
    # 4. Portfolio Concentration
    # --------------------------------------------------------

    result[
        "PORTFOLIO_CONCENTRATION_FINAL"
    ] = safe_numeric(
        result[
            "PORTFOLIO_CONCENTRATION_REBUILT"
        ]
    )

    # --------------------------------------------------------
    # 5. Restricted Leverage Exposure
    #
    # Ana composite'e dahil edilmez.
    # --------------------------------------------------------

    result[
        "LEVERAGE_EXPOSURE"
    ] = safe_numeric(
        result[
            "RB_Z_PW_LEVERAGE"
        ]
    )

    return result


# ============================================================
# 8. KALİTE BAYRAKLARI
# ============================================================

def add_channel_quality_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ana ve robustness kanalları için kalite bayrakları üretir.
    """

    result = df.copy()

    financial_match = safe_numeric(
        result[
            "FINANCIAL_MATCH_WEIGHT"
        ]
    )

    # --------------------------------------------------------
    # Internal Financial Capacity
    # --------------------------------------------------------

    for quality, match_threshold, coverage_threshold in [
        (
            "MODERATE",
            MODERATE_MATCH_THRESHOLD,
            MODERATE_COVERAGE_THRESHOLD,
        ),
        (
            "MAIN",
            MAIN_MATCH_THRESHOLD,
            MAIN_COVERAGE_THRESHOLD,
        ),
        (
            "STRICT",
            STRICT_MATCH_THRESHOLD,
            STRICT_COVERAGE_THRESHOLD,
        ),
    ]:

        capacity_valid = (
            result[
                "INTERNAL_FINANCIAL_CAPACITY"
            ].notna()
            & (
                safe_numeric(
                    result[
                        "COV_ROA"
                    ]
                )
                >= coverage_threshold
            )
            & (
                safe_numeric(
                    result[
                        "COV_CASH_RATIO"
                    ]
                )
                >= coverage_threshold
            )
            & (
                financial_match
                >= match_threshold
            )
        ).astype(
            int
        )

        result[
            f"INTERNAL_FINANCIAL_CAPACITY_VALID_{quality}"
        ] = capacity_valid

        result[
            f"INTERNAL_FINANCIAL_CAPACITY_{quality}"
        ] = quality_adjusted_score(
            raw_score=result[
                "INTERNAL_FINANCIAL_CAPACITY"
            ],
            valid_flag=capacity_valid,
        )

    # --------------------------------------------------------
    # External Financing Dependence
    #
    # Tek bileşenli kanal olduğu için kendi coverage değeri
    # üzerinden kalite kontrolü uygulanır.
    # --------------------------------------------------------

    for quality, match_threshold, coverage_threshold in [
        (
            "MODERATE",
            MODERATE_MATCH_THRESHOLD,
            MODERATE_COVERAGE_THRESHOLD,
        ),
        (
            "MAIN",
            MAIN_MATCH_THRESHOLD,
            MAIN_COVERAGE_THRESHOLD,
        ),
        (
            "STRICT",
            STRICT_MATCH_THRESHOLD,
            STRICT_COVERAGE_THRESHOLD,
        ),
    ]:

        external_valid = (
            result[
                "EXTERNAL_FINANCING_DEPENDENCE"
            ].notna()
            & (
                safe_numeric(
                    result[
                        "COV_EXTERNAL_FINANCE_DEPENDENCE"
                    ]
                )
                >= coverage_threshold
            )
            & (
                financial_match
                >= match_threshold
            )
        ).astype(
            int
        )

        result[
            f"EXTERNAL_FINANCING_DEPENDENCE_VALID_{quality}"
        ] = external_valid

        result[
            f"EXTERNAL_FINANCING_DEPENDENCE_{quality}"
        ] = quality_adjusted_score(
            raw_score=result[
                "EXTERNAL_FINANCING_DEPENDENCE"
            ],
            valid_flag=external_valid,
        )

    # --------------------------------------------------------
    # Growth-Duration Exposure
    #
    # Üç bileşenden en az ikisi coverage eşiğini sağlamalıdır.
    # --------------------------------------------------------

    growth_coverage_columns = [
        "COV_CAPEX_INTENSITY",
        "COV_RD_INTENSITY",
        "COV_REVENUE_GROWTH",
    ]

    growth_coverage = (
        result[
            growth_coverage_columns
        ]
        .apply(
            safe_numeric
        )
    )

    for quality, match_threshold, coverage_threshold in [
        (
            "MODERATE",
            MODERATE_MATCH_THRESHOLD,
            MODERATE_COVERAGE_THRESHOLD,
        ),
        (
            "MAIN",
            MAIN_MATCH_THRESHOLD,
            MAIN_COVERAGE_THRESHOLD,
        ),
        (
            "STRICT",
            STRICT_MATCH_THRESHOLD,
            STRICT_COVERAGE_THRESHOLD,
        ),
    ]:

        component_count = (
            growth_coverage
            >= coverage_threshold
        ).sum(
            axis=1
        )

        growth_valid = (
            result[
                "GROWTH_DURATION_EXPOSURE_FINAL"
            ].notna()
            & (
                component_count
                >= 2
            )
            & (
                financial_match
                >= match_threshold
            )
        ).astype(
            int
        )

        result[
            f"GROWTH_DURATION_EXPOSURE_FINAL_VALID_{quality}"
        ] = growth_valid

        result[
            f"GROWTH_DURATION_EXPOSURE_FINAL_{quality}"
        ] = quality_adjusted_score(
            raw_score=result[
                "GROWTH_DURATION_EXPOSURE_FINAL"
            ],
            valid_flag=growth_valid,
        )

    # --------------------------------------------------------
    # Portfolio Concentration
    #
    # Holdings tabanlı olduğu için finansal coverage gerekmez.
    # Ancak genel financial match örneklem tutarlılığı için
    # moderate/main/strict eşikleri korunur.
    # --------------------------------------------------------

    for quality, match_threshold in [
        (
            "MODERATE",
            MODERATE_MATCH_THRESHOLD,
        ),
        (
            "MAIN",
            MAIN_MATCH_THRESHOLD,
        ),
        (
            "STRICT",
            STRICT_MATCH_THRESHOLD,
        ),
    ]:

        concentration_valid = (
            result[
                "PORTFOLIO_CONCENTRATION_FINAL"
            ].notna()
            & (
                financial_match
                >= match_threshold
            )
        ).astype(
            int
        )

        result[
            f"PORTFOLIO_CONCENTRATION_FINAL_VALID_{quality}"
        ] = concentration_valid

        result[
            f"PORTFOLIO_CONCENTRATION_FINAL_{quality}"
        ] = quality_adjusted_score(
            raw_score=result[
                "PORTFOLIO_CONCENTRATION_FINAL"
            ],
            valid_flag=concentration_valid,
        )

    # --------------------------------------------------------
    # Leverage Exposure
    #
    # Restricted robustness kanalı.
    # --------------------------------------------------------

    for quality, match_threshold, coverage_threshold in [
        (
            "MODERATE",
            MODERATE_MATCH_THRESHOLD,
            MODERATE_COVERAGE_THRESHOLD,
        ),
        (
            "MAIN",
            MAIN_MATCH_THRESHOLD,
            MAIN_COVERAGE_THRESHOLD,
        ),
        (
            "STRICT",
            STRICT_MATCH_THRESHOLD,
            STRICT_COVERAGE_THRESHOLD,
        ),
    ]:

        leverage_valid = (
            result[
                "LEVERAGE_EXPOSURE"
            ].notna()
            & (
                safe_numeric(
                    result[
                        "COV_LEVERAGE"
                    ]
                )
                >= coverage_threshold
            )
            & (
                financial_match
                >= match_threshold
            )
        ).astype(
            int
        )

        result[
            f"LEVERAGE_EXPOSURE_VALID_{quality}"
        ] = leverage_valid

        result[
            f"LEVERAGE_EXPOSURE_{quality}"
        ] = quality_adjusted_score(
            raw_score=result[
                "LEVERAGE_EXPOSURE"
            ],
            valid_flag=leverage_valid,
        )

    return result


# ============================================================
# 9. COMPOSITE FAKTÖRLER
# ============================================================

def build_final_composites(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ana ve extended architecture composite'lerini oluşturur.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Ana Financial Architecture Risk
    #
    # Concentration ve leverage dahil edilmez.
    # --------------------------------------------------------

    result[
        "FINANCIAL_RISK_COMPONENT_CAPACITY"
    ] = (
        -1
        * result[
            "INTERNAL_FINANCIAL_CAPACITY"
        ]
    )

    result[
        "FINANCIAL_RISK_COMPONENT_EXTERNAL_FINANCE"
    ] = result[
        "EXTERNAL_FINANCING_DEPENDENCE"
    ]

    result[
        "FINANCIAL_RISK_COMPONENT_GROWTH_DURATION"
    ] = result[
        "GROWTH_DURATION_EXPOSURE_FINAL"
    ]

    financial_components = [
        "FINANCIAL_RISK_COMPONENT_CAPACITY",
        "FINANCIAL_RISK_COMPONENT_EXTERNAL_FINANCE",
        "FINANCIAL_RISK_COMPONENT_GROWTH_DURATION",
    ]

    (
        financial_score,
        financial_count,
        financial_share,
    ) = row_mean_with_minimum(
        df=result,
        columns=financial_components,
        minimum_components=2,
    )

    result[
        "FINANCIAL_ARCHITECTURE_RISK_FINAL"
    ] = financial_score

    result[
        "FINANCIAL_ARCHITECTURE_RISK_FINAL_AVAILABLE_COMPONENTS"
    ] = financial_count

    result[
        "FINANCIAL_ARCHITECTURE_RISK_FINAL_COMPONENT_SHARE"
    ] = financial_share

    # --------------------------------------------------------
    # Extended Architecture Risk
    #
    # Ana financial risk + concentration
    # --------------------------------------------------------

    extended_components = [
        "FINANCIAL_RISK_COMPONENT_CAPACITY",
        "FINANCIAL_RISK_COMPONENT_EXTERNAL_FINANCE",
        "FINANCIAL_RISK_COMPONENT_GROWTH_DURATION",
        "PORTFOLIO_CONCENTRATION_FINAL",
    ]

    (
        extended_score,
        extended_count,
        extended_share,
    ) = row_mean_with_minimum(
        df=result,
        columns=extended_components,
        minimum_components=3,
    )

    result[
        "EXTENDED_ARCHITECTURE_RISK_FINAL"
    ] = extended_score

    result[
        "EXTENDED_ARCHITECTURE_RISK_FINAL_AVAILABLE_COMPONENTS"
    ] = extended_count

    result[
        "EXTENDED_ARCHITECTURE_RISK_FINAL_COMPONENT_SHARE"
    ] = extended_share

    return result


def add_composite_quality_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Composite skorlar için kalite bayrakları oluşturur.
    """

    result = df.copy()

    for quality in [
        "MODERATE",
        "MAIN",
        "STRICT",
    ]:

        primary_valid_columns = [
            f"INTERNAL_FINANCIAL_CAPACITY_VALID_{quality}",
            f"EXTERNAL_FINANCING_DEPENDENCE_VALID_{quality}",
            f"GROWTH_DURATION_EXPOSURE_FINAL_VALID_{quality}",
        ]

        primary_valid_count = (
            result[
                primary_valid_columns
            ]
            .sum(
                axis=1
            )
        )

        financial_valid = (
            result[
                "FINANCIAL_ARCHITECTURE_RISK_FINAL"
            ].notna()
            & (
                primary_valid_count
                >= 2
            )
        ).astype(
            int
        )

        result[
            f"FINANCIAL_ARCHITECTURE_RISK_FINAL_VALID_{quality}"
        ] = financial_valid

        result[
            f"FINANCIAL_ARCHITECTURE_RISK_FINAL_{quality}"
        ] = quality_adjusted_score(
            raw_score=result[
                "FINANCIAL_ARCHITECTURE_RISK_FINAL"
            ],
            valid_flag=financial_valid,
        )

        extended_valid_columns = (
            primary_valid_columns
            + [
                f"PORTFOLIO_CONCENTRATION_FINAL_VALID_{quality}"
            ]
        )

        extended_valid_count = (
            result[
                extended_valid_columns
            ]
            .sum(
                axis=1
            )
        )

        extended_valid = (
            result[
                "EXTENDED_ARCHITECTURE_RISK_FINAL"
            ].notna()
            & (
                extended_valid_count
                >= 3
            )
        ).astype(
            int
        )

        result[
            f"EXTENDED_ARCHITECTURE_RISK_FINAL_VALID_{quality}"
        ] = extended_valid

        result[
            f"EXTENDED_ARCHITECTURE_RISK_FINAL_{quality}"
        ] = quality_adjusted_score(
            raw_score=result[
                "EXTENDED_ARCHITECTURE_RISK_FINAL"
            ],
            valid_flag=extended_valid,
        )

    return result


# ============================================================
# 10. PCA
# ============================================================

def orient_first_component(
    loadings: np.ndarray,
    columns: list[str],
) -> np.ndarray:
    """
    PCA işaretini ekonomik yoruma göre yönlendirir.
    """

    oriented = loadings.copy()

    if oriented.shape[1] == 0:
        return oriented

    expected_directions = {
        "INTERNAL_FINANCIAL_CAPACITY_MAIN": -1,
        "EXTERNAL_FINANCING_DEPENDENCE_MAIN": 1,
        "GROWTH_DURATION_EXPOSURE_FINAL_MAIN": 1,
        "PORTFOLIO_CONCENTRATION_FINAL_MAIN": 1,
    }

    orientation_score = 0.0

    for index, column in enumerate(
        columns
    ):

        orientation_score += (
            expected_directions.get(
                column,
                0,
            )
            * oriented[
                index,
                0,
            ]
        )

    if orientation_score < 0:

        oriented[
            :,
            0,
        ] *= -1

    return oriented


def run_pca(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Dört geniş main-quality kanal üzerinde PCA uygular.
    """

    pca_columns = [
        "INTERNAL_FINANCIAL_CAPACITY_MAIN",
        "EXTERNAL_FINANCING_DEPENDENCE_MAIN",
        "GROWTH_DURATION_EXPOSURE_FINAL_MAIN",
        "PORTFOLIO_CONCENTRATION_FINAL_MAIN",
    ]

    pca_sample = (
        df[
            [
                "ETF_ID",
                "ETF_NAME",
            ]
            + pca_columns
        ]
        .dropna(
            subset=pca_columns
        )
        .copy()
    )

    if len(pca_sample) < 5:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    matrix = pca_sample[
        pca_columns
    ].to_numpy(
        dtype=float
    )

    means = matrix.mean(
        axis=0
    )

    standard_deviations = matrix.std(
        axis=0,
        ddof=1,
    )

    if (
        standard_deviations <= 0
    ).any():

        raise RuntimeError(
            "PCA girdilerinden en az birinin standart sapması sıfır."
        )

    standardized = (
        matrix
        - means
    ) / standard_deviations

    correlation_matrix = np.corrcoef(
        standardized,
        rowvar=False,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        correlation_matrix
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    eigenvectors = orient_first_component(
        loadings=eigenvectors,
        columns=pca_columns,
    )

    scores = standardized @ eigenvectors

    explained_variance_ratio = (
        eigenvalues
        / eigenvalues.sum()
    )

    loading_rows = []

    for component_index in range(
        eigenvectors.shape[1]
    ):

        component_name = (
            f"PCA_COMPONENT_{component_index + 1}"
        )

        for variable_index, variable in enumerate(
            pca_columns
        ):

            loading_rows.append(
                {
                    "PCA_COMPONENT": component_name,
                    "VARIABLE": variable,
                    "LOADING": float(
                        eigenvectors[
                            variable_index,
                            component_index,
                        ]
                    ),
                }
            )

    variance_rows = []

    cumulative_variance = 0.0

    for component_index, ratio in enumerate(
        explained_variance_ratio
    ):

        cumulative_variance += ratio

        variance_rows.append(
            {
                "PCA_COMPONENT": (
                    f"PCA_COMPONENT_{component_index + 1}"
                ),
                "EIGENVALUE": float(
                    eigenvalues[
                        component_index
                    ]
                ),
                "EXPLAINED_VARIANCE_RATIO": float(
                    ratio
                ),
                "CUMULATIVE_EXPLAINED_VARIANCE": float(
                    cumulative_variance
                ),
                "N_ETFS": int(
                    len(
                        pca_sample
                    )
                ),
            }
        )

    score_data = pca_sample[
        [
            "ETF_ID",
            "ETF_NAME",
        ]
    ].copy()

    for component_index in range(
        scores.shape[1]
    ):

        score_data[
            f"PCA_COMPONENT_{component_index + 1}"
        ] = scores[
            :,
            component_index
        ]

    return (
        pd.DataFrame(
            loading_rows
        ),
        pd.DataFrame(
            variance_rows
        ),
        score_data,
    )


# ============================================================
# 11. RAPORLAR
# ============================================================

def build_quality_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kanal bazında geçerli ETF sayılarını üretir.
    """

    channels = (
        PRIMARY_CHANNELS
        + ROBUSTNESS_CHANNELS
        + COMPOSITE_CHANNELS
    )

    rows = []

    for channel in channels:

        rows.append(
            {
                "CHANNEL": channel,

                "MODERATE_VALID_ETFS": int(
                    df[
                        f"{channel}_VALID_MODERATE"
                    ].sum()
                ),

                "MAIN_VALID_ETFS": int(
                    df[
                        f"{channel}_VALID_MAIN"
                    ].sum()
                ),

                "STRICT_VALID_ETFS": int(
                    df[
                        f"{channel}_VALID_STRICT"
                    ].sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nihai kanal tanımlayıcı istatistiklerini üretir.
    """

    variables = (
        PRIMARY_CHANNELS
        + ROBUSTNESS_CHANNELS
        + COMPOSITE_CHANNELS
    )

    rows = []

    for variable in variables:

        values = safe_numeric(
            df[
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

                "SKEWNESS": float(
                    values.skew()
                ),

                "KURTOSIS": float(
                    values.kurt()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main-quality nihai kanal korelasyonlarını üretir.
    """

    variables = [
        "INTERNAL_FINANCIAL_CAPACITY_MAIN",
        "EXTERNAL_FINANCING_DEPENDENCE_MAIN",
        "GROWTH_DURATION_EXPOSURE_FINAL_MAIN",
        "PORTFOLIO_CONCENTRATION_FINAL_MAIN",
        "FINANCIAL_ARCHITECTURE_RISK_FINAL_MAIN",
        "EXTENDED_ARCHITECTURE_RISK_FINAL_MAIN",
        "LEVERAGE_EXPOSURE_MAIN",
    ]

    correlation = (
        df[
            variables
        ]
        .apply(
            safe_numeric
        )
        .corr(
            method="pearson",
            min_periods=5,
        )
    )

    correlation.index.name = "VARIABLE"

    return correlation.reset_index()


def build_rankings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main-quality nihai kanal sıralamalarını üretir.
    """

    variables = [
        "INTERNAL_FINANCIAL_CAPACITY_MAIN",
        "EXTERNAL_FINANCING_DEPENDENCE_MAIN",
        "GROWTH_DURATION_EXPOSURE_FINAL_MAIN",
        "PORTFOLIO_CONCENTRATION_FINAL_MAIN",
        "FINANCIAL_ARCHITECTURE_RISK_FINAL_MAIN",
        "EXTENDED_ARCHITECTURE_RISK_FINAL_MAIN",
    ]

    ranking = df[
        [
            "ETF_ID",
            "ETF_NAME",
            "FINANCIAL_MATCH_WEIGHT",
        ]
        + variables
    ].copy()

    for variable in variables:

        ranking[
            f"RANK_HIGH_{variable}"
        ] = ranking[
            variable
        ].rank(
            method="min",
            ascending=False,
        )

        ranking[
            f"PERCENTILE_{variable}"
        ] = ranking[
            variable
        ].rank(
            method="average",
            pct=True,
        )

    return ranking.sort_values(
        by=(
            "FINANCIAL_ARCHITECTURE_RISK_FINAL_MAIN"
        ),
        ascending=False,
        na_position="last",
    )


def build_variable_dictionary() -> pd.DataFrame:
    """
    Nihai architecture kanal sözlüğünü üretir.
    """

    return pd.DataFrame(
        [
            {
                "VARIABLE": (
                    "INTERNAL_FINANCIAL_CAPACITY"
                ),
                "TYPE": (
                    "Primary non-overlapping mechanism"
                ),
                "FORMULA": (
                    "Equal-weight mean of Z(ROA) and Z(Cash Ratio)"
                ),
                "THEORY": (
                    "Internal liquidity and profitability buffers"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Positive or less negative"
                ),
                "ROLE": "Primary channel",
            },

            {
                "VARIABLE": (
                    "EXTERNAL_FINANCING_DEPENDENCE"
                ),
                "TYPE": (
                    "Primary non-overlapping mechanism"
                ),
                "FORMULA": (
                    "Z(External Finance Dependence)"
                ),
                "THEORY": (
                    "Dependence on external capital and financing constraints"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Negative"
                ),
                "ROLE": "Primary channel",
            },

            {
                "VARIABLE": (
                    "GROWTH_DURATION_EXPOSURE_FINAL"
                ),
                "TYPE": (
                    "Primary non-overlapping mechanism"
                ),
                "FORMULA": (
                    "Equal-weight mean of Z(CapEx), Z(R&D), "
                    "and Z(Revenue Growth)"
                ),
                "THEORY": (
                    "Real options and equity duration"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Negative"
                ),
                "ROLE": "Primary channel",
            },

            {
                "VARIABLE": (
                    "PORTFOLIO_CONCENTRATION_FINAL"
                ),
                "TYPE": (
                    "Primary portfolio-structure mechanism"
                ),
                "FORMULA": (
                    "Equal-weight mean of Z(HHI), Z(Top10 Weight), "
                    "and -Z(Effective Number of Holdings)"
                ),
                "THEORY": (
                    "Diversification and dominant-holding exposure"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Negative"
                ),
                "ROLE": "Primary channel",
            },

            {
                "VARIABLE": (
                    "LEVERAGE_EXPOSURE"
                ),
                "TYPE": (
                    "Restricted robustness mechanism"
                ),
                "FORMULA": (
                    "Z(Portfolio-weighted Leverage)"
                ),
                "THEORY": (
                    "Existing debt burden and refinancing exposure"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Negative"
                ),
                "ROLE": (
                    "Restricted-sample robustness"
                ),
            },

            {
                "VARIABLE": (
                    "FINANCIAL_ARCHITECTURE_RISK_FINAL"
                ),
                "TYPE": (
                    "Non-overlapping financial composite"
                ),
                "FORMULA": (
                    "- Internal Capacity + External Financing Dependence "
                    "+ Growth-Duration Exposure"
                ),
                "THEORY": (
                    "Embedded firm-level financial architecture risk"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Negative"
                ),
                "ROLE": (
                    "Composite robustness"
                ),
            },

            {
                "VARIABLE": (
                    "EXTENDED_ARCHITECTURE_RISK_FINAL"
                ),
                "TYPE": (
                    "Extended architecture composite"
                ),
                "FORMULA": (
                    "- Internal Capacity + External Financing Dependence "
                    "+ Growth-Duration Exposure + Concentration"
                ),
                "THEORY": (
                    "Firm financial architecture and portfolio structure"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Negative"
                ),
                "ROLE": (
                    "Secondary robustness"
                ),
            },

            {
                "VARIABLE": "PCA_COMPONENT_1",
                "TYPE": (
                    "Data-driven robustness factor"
                ),
                "FORMULA": (
                    "First principal component of four main channels"
                ),
                "THEORY": (
                    "Common latent architecture dimension"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Based on economically oriented loadings"
                ),
                "ROLE": (
                    "PCA robustness"
                ),
            },
        ]
    )


# ============================================================
# 12. VALIDATION
# ============================================================

def build_validation(
    df: pd.DataFrame,
    quality_summary: pd.DataFrame,
    pca_variance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nihai kanal yapısının doğrulama kontrollerini yapar.
    """

    duplicate_etfs = int(
        df.duplicated(
            subset=[
                "ETF_ID",
            ],
            keep=False,
        ).sum()
    )

    main_columns = [
        "INTERNAL_FINANCIAL_CAPACITY_MAIN",
        "EXTERNAL_FINANCING_DEPENDENCE_MAIN",
        "GROWTH_DURATION_EXPOSURE_FINAL_MAIN",
        "PORTFOLIO_CONCENTRATION_FINAL_MAIN",
    ]

    channel_correlation = (
        df[
            main_columns
        ]
        .corr(
            min_periods=5
        )
    )

    off_diagonal = (
        channel_correlation
        .where(
            ~np.eye(
                len(
                    channel_correlation
                ),
                dtype=bool,
            )
        )
        .abs()
        .stack()
    )

    max_abs_correlation = (
        float(
            off_diagonal.max()
        )
        if not off_diagonal.empty
        else np.nan
    )

    rows = [
        {
            "CHECK": "INPUT_ETFS",
            "VALUE": len(
                df
            ),
            "PASS": int(
                len(
                    df
                ) > 0
            ),
        },

        {
            "CHECK": "DUPLICATE_ETF_IDS",
            "VALUE": duplicate_etfs,
            "PASS": int(
                duplicate_etfs
                == 0
            ),
        },

        {
            "CHECK": (
                "PRIMARY_CHANNELS_CREATED"
            ),
            "VALUE": sum(
                column in df.columns
                for column in PRIMARY_CHANNELS
            ),
            "PASS": int(
                all(
                    column in df.columns
                    for column in PRIMARY_CHANNELS
                )
            ),
        },

        {
            "CHECK": (
                "FINANCIAL_COMPOSITE_CREATED"
            ),
            "VALUE": int(
                (
                    "FINANCIAL_ARCHITECTURE_RISK_FINAL"
                    in df.columns
                )
            ),
            "PASS": int(
                (
                    "FINANCIAL_ARCHITECTURE_RISK_FINAL"
                    in df.columns
                )
            ),
        },

        {
            "CHECK": (
                "MAX_ABSOLUTE_PRIMARY_CHANNEL_CORRELATION"
            ),
            "VALUE": max_abs_correlation,
            "PASS": int(
                pd.notna(
                    max_abs_correlation
                )
                and max_abs_correlation
                < 0.95
            ),
        },

        {
            "CHECK": (
                "QUALITY_SUMMARY_ROWS"
            ),
            "VALUE": len(
                quality_summary
            ),
            "PASS": int(
                len(
                    quality_summary
                )
                == 7
            ),
        },

        {
            "CHECK": (
                "PCA_RESULTS_AVAILABLE"
            ),
            "VALUE": len(
                pca_variance
            ),
            "PASS": int(
                len(
                    pca_variance
                )
                > 0
            ),
        },

        {
            "CHECK": (
                "EXTERNAL_FINANCE_MAIN_VALID_ETFS"
            ),
            "VALUE": int(
                df[
                    "EXTERNAL_FINANCING_DEPENDENCE_VALID_MAIN"
                ].sum()
            ),
            "PASS": int(
                df[
                    "EXTERNAL_FINANCING_DEPENDENCE_VALID_MAIN"
                ].sum()
                >= 10
            ),
        },

        {
            "CHECK": (
                "FINANCIAL_RISK_MAIN_VALID_ETFS"
            ),
            "VALUE": int(
                df[
                    "FINANCIAL_ARCHITECTURE_RISK_FINAL_VALID_MAIN"
                ].sum()
            ),
            "PASS": int(
                df[
                    "FINANCIAL_ARCHITECTURE_RISK_FINAL_VALID_MAIN"
                ].sum()
                >= 10
            ),
        },

        {
            "CHECK": (
                "LEVERAGE_IDENTIFIED_AS_RESTRICTED"
            ),
            "VALUE": int(
                df[
                    "LEVERAGE_EXPOSURE_VALID_MAIN"
                ].sum()
            ),
            "PASS": 1,
        },
    ]

    for channel in PRIMARY_CHANNELS:

        rows.append(
            {
                "CHECK": (
                    f"{channel}_MAIN_VALID_ETFS"
                ),
                "VALUE": int(
                    df[
                        f"{channel}_VALID_MAIN"
                    ].sum()
                ),
                "PASS": int(
                    df[
                        f"{channel}_VALID_MAIN"
                    ].sum()
                    > 0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 13. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 88)
    print("24B - FINAL PORTFOLIO ARCHITECTURE CHANNELS")
    print("=" * 88)

    print(
        "\n1/9 - Script 24 architecture paneli okunuyor..."
    )

    architecture = normalize_columns(
        read_data(
            INPUT_FILE
        )
    )

    required_columns = [
        "ETF_ID",
        "ETF_NAME",
        "FINANCIAL_MATCH_WEIGHT",
        "INTERNAL_FINANCIAL_CAPACITY",
        "GROWTH_DURATION_EXPOSURE_REBUILT",
        "PORTFOLIO_CONCENTRATION_REBUILT",
        "RB_Z_PW_EXTERNAL_FINANCE_DEPENDENCE",
        "RB_Z_PW_LEVERAGE",
        "COV_ROA",
        "COV_CASH_RATIO",
        "COV_EXTERNAL_FINANCE_DEPENDENCE",
        "COV_CAPEX_INTENSITY",
        "COV_RD_INTENSITY",
        "COV_REVENUE_GROWTH",
        "COV_LEVERAGE",
    ]

    require_columns(
        architecture,
        required_columns,
    )

    print(
        f"ETF sayısı: "
        f"{len(architecture):,}"
    )

    print(
        "\n2/9 - Nihai non-overlapping kanallar oluşturuluyor..."
    )

    final_panel = build_final_channels(
        architecture
    )

    print(
        "\n3/9 - Kanal kalite bayrakları oluşturuluyor..."
    )

    final_panel = add_channel_quality_flags(
        final_panel
    )

    print(
        "\n4/9 - Nihai architecture composite'leri oluşturuluyor..."
    )

    final_panel = build_final_composites(
        final_panel
    )

    final_panel = add_composite_quality_flags(
        final_panel
    )

    print(
        "\n5/9 - PCA robustness analizi çalıştırılıyor..."
    )

    (
        pca_loadings,
        pca_variance,
        pca_scores,
    ) = run_pca(
        final_panel
    )

    if not pca_scores.empty:

        final_panel = final_panel.merge(
            pca_scores,
            on=[
                "ETF_ID",
                "ETF_NAME",
            ],
            how="left",
            validate="one_to_one",
        )

    print(
        "\n6/9 - Kalite ve descriptive raporları hazırlanıyor..."
    )

    quality_summary = build_quality_summary(
        final_panel
    )

    descriptives = build_descriptive_statistics(
        final_panel
    )

    correlations = build_correlations(
        final_panel
    )

    rankings = build_rankings(
        final_panel
    )

    variable_dictionary = (
        build_variable_dictionary()
    )

    print(
        "\n7/9 - Validation kontrolleri çalıştırılıyor..."
    )

    validation = build_validation(
        df=final_panel,
        quality_summary=quality_summary,
        pca_variance=pca_variance,
    )

    print(
        "\n8/9 - Nihai örneklem bayrakları ekleniyor..."
    )

    final_panel[
        "FINAL_PRIMARY_MAIN_SAMPLE"
    ] = (
        (
            final_panel[
                "INTERNAL_FINANCIAL_CAPACITY_VALID_MAIN"
            ] == 1
        )
        | (
            final_panel[
                "EXTERNAL_FINANCING_DEPENDENCE_VALID_MAIN"
            ] == 1
        )
        | (
            final_panel[
                "GROWTH_DURATION_EXPOSURE_FINAL_VALID_MAIN"
            ] == 1
        )
        | (
            final_panel[
                "PORTFOLIO_CONCENTRATION_FINAL_VALID_MAIN"
            ] == 1
        )
    ).astype(
        int
    )

    final_panel[
        "FINAL_FINANCIAL_COMPOSITE_MAIN_SAMPLE"
    ] = (
        final_panel[
            "FINANCIAL_ARCHITECTURE_RISK_FINAL_VALID_MAIN"
        ]
        == 1
    ).astype(
        int
    )

    final_panel[
        "FINAL_PCA_SAMPLE"
    ] = (
        final_panel[
            "PCA_COMPONENT_1"
        ].notna()
        if "PCA_COMPONENT_1"
        in final_panel.columns
        else False
    ).astype(
        int
    )

    print(
        "\n9/9 - Çıktılar kaydediliyor..."
    )

    final_panel.to_csv(
        FINAL_PANEL_FILE,
        index=False,
    )

    final_panel.to_parquet(
        FINAL_PANEL_PARQUET_FILE,
        index=False,
    )

    quality_summary.to_csv(
        QUALITY_SUMMARY_FILE,
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

    pca_loadings.to_csv(
        PCA_LOADINGS_FILE,
        index=False,
    )

    pca_variance.to_csv(
        PCA_VARIANCE_FILE,
        index=False,
    )

    pca_scores.to_csv(
        PCA_SCORES_FILE,
        index=False,
    )

    rankings.to_csv(
        RANKINGS_FILE,
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
        "\nFINAL ARCHITECTURE CHANNELS HAZIR"
    )

    print("=" * 88)

    print(
        "\nNihai kanal kalite özeti:"
    )

    print(
        quality_summary.to_string(
            index=False
        )
    )

    print(
        "\nNihai kanal descriptive statistics:"
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
        "\nMain-quality kanal korelasyonları:"
    )

    print(
        correlations.to_string(
            index=False
        )
    )

    print(
        "\nPCA explained variance:"
    )

    if pca_variance.empty:

        print(
            "PCA için yeterli complete-case ETF bulunamadı."
        )

    else:

        print(
            pca_variance.to_string(
                index=False
            )
        )

    print(
        "\nPCA loadings:"
    )

    if pca_loadings.empty:

        print(
            "PCA loading üretilemedi."
        )

    else:

        print(
            pca_loadings.to_string(
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
        "\nMETODOLOJİK KARAR:"
    )

    print(
        "External Financing Dependence ana mekanizma olarak, "
        "Leverage Exposure ise düşük coverage nedeniyle restricted "
        "robustness kanalı olarak kullanılacaktır."
    )

    print(
        "Ana Financial Architecture Risk composite'i concentration "
        "ve leverage içermemektedir."
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        FINAL_PANEL_FILE
    )

    print(
        QUALITY_SUMMARY_FILE
    )

    print(
        DESCRIPTIVE_FILE
    )

    print(
        CORRELATION_FILE
    )

    print(
        PCA_LOADINGS_FILE
    )

    print(
        PCA_VARIANCE_FILE
    )

    print(
        PCA_SCORES_FILE
    )

    print(
        RANKINGS_FILE
    )

    print(
        VALIDATION_FILE
    )

    print(
        VARIABLE_DICTIONARY_FILE
    )


if __name__ == "__main__":
    main()