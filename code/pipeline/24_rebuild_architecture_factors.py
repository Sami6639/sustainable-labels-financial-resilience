from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 24_rebuild_architecture_factors.py
#
# AMAÇ
# ----
# Önceki theory-based architecture faktörlerindeki bileşen
# örtüşmesini kaldırarak non-overlapping ve ekonomik olarak
# yorumlanabilir portfolio architecture kanalları oluşturmak.
#
# YENİ KANALLAR
# -------------
# 1. INTERNAL_FINANCIAL_CAPACITY
#       + ROA
#       + Cash Ratio
#
# 2. BALANCE_SHEET_VULNERABILITY
#       + Leverage
#       + External Finance Dependence
#
# 3. GROWTH_DURATION_EXPOSURE_REBUILT
#       + CapEx Intensity
#       + R&D Intensity
#       + Revenue Growth
#
# 4. PORTFOLIO_CONCENTRATION_REBUILT
#       + HHI
#       + Top10 Weight
#       - Effective Number of Holdings
#
# ANA TASARIM İLKESİ
# ------------------
# Bu dört kanal ana analizde ayrı ayrı test edilir.
#
# Concentration, firma finansal mimarisi composite'ine dahil
# edilmez. Böylece concentration'ın composite faktörü mekanik
# olarak domine etmesi engellenir.
#
# ROBUSTNESS COMPOSITE
# --------------------
# FINANCIAL_ARCHITECTURE_RISK
#       - Internal Financial Capacity
#       + Balance-Sheet Vulnerability
#       + Growth-Duration Exposure
#
# PCA yalnızca veri-temelli robustness amacıyla kullanılır.
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
    / "18_quality_adjusted_architecture_factors.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

REBUILT_PANEL_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_architecture_factors.csv"
)

REBUILT_PANEL_PARQUET_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_architecture_factors.parquet"
)

FACTOR_DESCRIPTIVE_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_factor_descriptive_statistics.csv"
)

FACTOR_CORRELATION_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_factor_correlations.csv"
)

FACTOR_COMPONENT_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_factor_components.csv"
)

STANDARDIZATION_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_standardization_parameters.csv"
)

QUALITY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_factor_quality_summary.csv"
)

LEAVE_ONE_OUT_FILE = (
    OUTPUT_DIR
    / "24_leave_one_component_out_factors.csv"
)

LEAVE_ONE_OUT_CORRELATION_FILE = (
    OUTPUT_DIR
    / "24_leave_one_out_correlations.csv"
)

PCA_LOADINGS_FILE = (
    OUTPUT_DIR
    / "24_architecture_pca_loadings.csv"
)

PCA_VARIANCE_FILE = (
    OUTPUT_DIR
    / "24_architecture_pca_explained_variance.csv"
)

PCA_SCORES_FILE = (
    OUTPUT_DIR
    / "24_architecture_pca_scores.csv"
)

RANKING_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_factor_rankings.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_factor_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "24_rebuilt_factor_variable_dictionary.csv"
)


# ============================================================
# 4. METODOLOJİK AYARLAR
# ============================================================

REFERENCE_MATCH_THRESHOLD = 0.80

MODERATE_MATCH_THRESHOLD = 0.70
MAIN_MATCH_THRESHOLD = 0.80
STRICT_MATCH_THRESHOLD = 0.90

LOWER_WINSOR_QUANTILE = 0.01
UPPER_WINSOR_QUANTILE = 0.99

MODERATE_COMPONENT_COVERAGE = 0.50
MAIN_COMPONENT_COVERAGE = 0.70
STRICT_COMPONENT_COVERAGE = 0.80


# ============================================================
# 5. YENİ NON-OVERLAPPING FAKTÖR TANIMLARI
# ============================================================

FACTOR_DEFINITIONS = {
    "INTERNAL_FINANCIAL_CAPACITY": {
        "components": {
            "PW_ROA": 1,
            "PW_CASH_RATIO": 1,
        },
        "coverage_columns": {
            "PW_ROA": "COV_ROA",
            "PW_CASH_RATIO": "COV_CASH_RATIO",
        },
        "minimum_components": 2,
        "theory": (
            "Profitability and internal liquidity buffers increase "
            "the ability of portfolio firms to absorb climate-policy "
            "and financing shocks."
        ),
        "expected_cpu_interaction": (
            "Positive or less negative"
        ),
        "research_questions": "RQ2, RQ3, RQ4",
    },

    "BALANCE_SHEET_VULNERABILITY": {
        "components": {
            "PW_LEVERAGE": 1,
            "PW_EXTERNAL_FINANCE_DEPENDENCE": 1,
        },
        "coverage_columns": {
            "PW_LEVERAGE": "COV_LEVERAGE",
            "PW_EXTERNAL_FINANCE_DEPENDENCE": (
                "COV_EXTERNAL_FINANCE_DEPENDENCE"
            ),
        },
        "minimum_components": 2,
        "theory": (
            "Leverage and dependence on external finance increase "
            "exposure to uncertainty-driven financing constraints "
            "and refinancing risk."
        ),
        "expected_cpu_interaction": "Negative",
        "research_questions": "RQ2, RQ3",
    },

    "GROWTH_DURATION_EXPOSURE_REBUILT": {
        "components": {
            "PW_CAPEX_INTENSITY": 1,
            "PW_RD_INTENSITY": 1,
            "PW_REVENUE_GROWTH": 1,
        },
        "coverage_columns": {
            "PW_CAPEX_INTENSITY": "COV_CAPEX_INTENSITY",
            "PW_RD_INTENSITY": "COV_RD_INTENSITY",
            "PW_REVENUE_GROWTH": "COV_REVENUE_GROWTH",
        },
        "minimum_components": 2,
        "theory": (
            "Investment, innovation and growth increase equity "
            "duration and the sensitivity of expected cash flows "
            "to policy and discount-rate uncertainty."
        ),
        "expected_cpu_interaction": (
            "Negative and potentially state dependent"
        ),
        "research_questions": "RQ2, RQ4",
    },

    "PORTFOLIO_CONCENTRATION_REBUILT": {
        "components": {
            "HHI": 1,
            "TOP10_WEIGHT": 1,
            "EFFECTIVE_NUMBER_OF_HOLDINGS": -1,
        },
        "coverage_columns": {},
        "minimum_components": 3,
        "theory": (
            "Portfolio concentration reduces diversification and "
            "increases exposure to dominant firms, sectors and "
            "transition-sensitive holdings."
        ),
        "expected_cpu_interaction": "Negative",
        "research_questions": "RQ1, RQ3",
    },
}


PRIMARY_CHANNELS = [
    "INTERNAL_FINANCIAL_CAPACITY",
    "BALANCE_SHEET_VULNERABILITY",
    "GROWTH_DURATION_EXPOSURE_REBUILT",
    "PORTFOLIO_CONCENTRATION_REBUILT",
]


# Financial composite excludes concentration.
FINANCIAL_RISK_COMPONENTS = {
    "INTERNAL_FINANCIAL_CAPACITY": -1,
    "BALANCE_SHEET_VULNERABILITY": 1,
    "GROWTH_DURATION_EXPOSURE_REBUILT": 1,
}


# Extended composite retained only as robustness.
EXTENDED_RISK_COMPONENTS = {
    "INTERNAL_FINANCIAL_CAPACITY": -1,
    "BALANCE_SHEET_VULNERABILITY": 1,
    "GROWTH_DURATION_EXPOSURE_REBUILT": 1,
    "PORTFOLIO_CONCENTRATION_REBUILT": 1,
}


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


def zscore_with_parameters(
    series: pd.Series,
    mean_value: float,
    std_value: float,
) -> pd.Series:
    """
    Sabit referans parametreleriyle standardizasyon yapar.
    """

    numeric = safe_numeric(
        series
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
    Satır bazında mevcut bileşenlerin eşit ağırlıklı ortalamasını
    hesaplar.
    """

    available = (
        df[
            columns
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    share = (
        available
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
        available
        >= minimum_components
    )

    return (
        score,
        available,
        share,
    )


# ============================================================
# 7. STANDARDİZASYON PARAMETRELERİ
# ============================================================

def collect_source_variables() -> list[str]:
    """
    Tüm factor source değişkenlerini listeler.
    """

    variables = []

    for definition in (
        FACTOR_DEFINITIONS.values()
    ):

        variables.extend(
            definition[
                "components"
            ].keys()
        )

    return sorted(
        set(
            variables
        )
    )


def build_standardization_parameters(
    df: pd.DataFrame,
    source_variables: list[str],
) -> pd.DataFrame:
    """
    Financial match >= %80 referans örnekleminde winsorization ve
    z-score parametrelerini hesaplar.
    """

    reference = df.loc[
        safe_numeric(
            df[
                "FINANCIAL_MATCH_WEIGHT"
            ]
        )
        >= REFERENCE_MATCH_THRESHOLD
    ].copy()

    if reference.empty:

        raise RuntimeError(
            "Financial match >= %80 referans örneklem bulunamadı."
        )

    rows = []

    for variable in source_variables:

        values = safe_numeric(
            reference[
                variable
            ]
        ).dropna()

        if len(values) < 5:

            raise RuntimeError(
                f"{variable} için yeterli referans gözlemi yok."
            )

        lower_limit = float(
            values.quantile(
                LOWER_WINSOR_QUANTILE
            )
        )

        upper_limit = float(
            values.quantile(
                UPPER_WINSOR_QUANTILE
            )
        )

        winsorized = values.clip(
            lower=lower_limit,
            upper=upper_limit,
        )

        mean_value = float(
            winsorized.mean()
        )

        std_value = float(
            winsorized.std(
                ddof=1
            )
        )

        rows.append(
            {
                "VARIABLE": variable,
                "REFERENCE_N": int(
                    len(values)
                ),
                "REFERENCE_MATCH_THRESHOLD": (
                    REFERENCE_MATCH_THRESHOLD
                ),
                "LOWER_WINSOR_LIMIT": (
                    lower_limit
                ),
                "UPPER_WINSOR_LIMIT": (
                    upper_limit
                ),
                "REFERENCE_MEAN": (
                    mean_value
                ),
                "REFERENCE_STD": (
                    std_value
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_standardized_sources(
    df: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kaynak değişkenlerin winsorize ve z-score sürümlerini oluşturur.
    """

    result = df.copy()

    parameter_lookup = (
        parameters
        .set_index(
            "VARIABLE"
        )
        .to_dict(
            orient="index"
        )
    )

    for variable, settings in (
        parameter_lookup.items()
    ):

        winsorized_name = (
            f"RB_W_{variable}"
        )

        z_name = (
            f"RB_Z_{variable}"
        )

        result[
            winsorized_name
        ] = safe_numeric(
            result[
                variable
            ]
        ).clip(
            lower=settings[
                "LOWER_WINSOR_LIMIT"
            ],
            upper=settings[
                "UPPER_WINSOR_LIMIT"
            ],
        )

        result[
            z_name
        ] = zscore_with_parameters(
            series=result[
                winsorized_name
            ],
            mean_value=settings[
                "REFERENCE_MEAN"
            ],
            std_value=settings[
                "REFERENCE_STD"
            ],
        )

    return result


# ============================================================
# 8. NON-OVERLAPPING FAKTÖRLER
# ============================================================

def build_rebuilt_factors(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Yeni non-overlapping faktörleri üretir.
    """

    result = df.copy()

    component_rows = []

    for factor_name, definition in (
        FACTOR_DEFINITIONS.items()
    ):

        signed_columns = []

        for variable, direction in (
            definition[
                "components"
            ].items()
        ):

            source_column = (
                f"RB_Z_{variable}"
            )

            component_column = (
                f"RB_COMP_{factor_name}_{variable}"
            )

            result[
                component_column
            ] = (
                direction
                * result[
                    source_column
                ]
            )

            signed_columns.append(
                component_column
            )

            component_rows.append(
                {
                    "FACTOR": factor_name,
                    "SOURCE_VARIABLE": variable,
                    "STANDARDIZED_SOURCE": source_column,
                    "DIRECTION": direction,
                    "EQUAL_WEIGHT": (
                        1.0
                        / len(
                            definition[
                                "components"
                            ]
                        )
                    ),
                    "THEORY": definition[
                        "theory"
                    ],
                    "EXPECTED_CPU_INTERACTION": definition[
                        "expected_cpu_interaction"
                    ],
                    "RESEARCH_QUESTIONS": definition[
                        "research_questions"
                    ],
                }
            )

        (
            factor_score,
            available_count,
            available_share,
        ) = row_mean_with_minimum(
            df=result,
            columns=signed_columns,
            minimum_components=definition[
                "minimum_components"
            ],
        )

        result[
            factor_name
        ] = factor_score

        result[
            f"{factor_name}_AVAILABLE_COMPONENTS"
        ] = available_count

        result[
            f"{factor_name}_COMPONENT_SHARE"
        ] = available_share

    return (
        result,
        pd.DataFrame(
            component_rows
        ),
    )


# ============================================================
# 9. COVERAGE QUALITY
# ============================================================

def add_factor_quality_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Yeni faktörler için moderate, main ve strict kalite bayrakları
    oluşturur.
    """

    result = df.copy()

    financial_match = safe_numeric(
        result[
            "FINANCIAL_MATCH_WEIGHT"
        ]
    )

    for factor_name, definition in (
        FACTOR_DEFINITIONS.items()
    ):

        coverage_columns = list(
            definition[
                "coverage_columns"
            ].values()
        )

        if not coverage_columns:

            result[
                f"{factor_name}_MEAN_COVERAGE"
            ] = 1.0

            result[
                f"{factor_name}_MIN_COVERAGE"
            ] = 1.0

            result[
                f"{factor_name}_COV50_COMPONENTS"
            ] = len(
                definition[
                    "components"
                ]
            )

            result[
                f"{factor_name}_COV70_COMPONENTS"
            ] = len(
                definition[
                    "components"
                ]
            )

            result[
                f"{factor_name}_COV80_COMPONENTS"
            ] = len(
                definition[
                    "components"
                ]
            )

        else:

            coverage_data = (
                result[
                    coverage_columns
                ]
                .apply(
                    safe_numeric
                )
            )

            result[
                f"{factor_name}_MEAN_COVERAGE"
            ] = coverage_data.mean(
                axis=1,
                skipna=True,
            )

            result[
                f"{factor_name}_MIN_COVERAGE"
            ] = coverage_data.min(
                axis=1,
                skipna=True,
            )

            result[
                f"{factor_name}_COV50_COMPONENTS"
            ] = (
                coverage_data
                >= MODERATE_COMPONENT_COVERAGE
            ).sum(
                axis=1
            )

            result[
                f"{factor_name}_COV70_COMPONENTS"
            ] = (
                coverage_data
                >= MAIN_COMPONENT_COVERAGE
            ).sum(
                axis=1
            )

            result[
                f"{factor_name}_COV80_COMPONENTS"
            ] = (
                coverage_data
                >= STRICT_COMPONENT_COVERAGE
            ).sum(
                axis=1
            )

        minimum_components = definition[
            "minimum_components"
        ]

        result[
            f"{factor_name}_VALID_MODERATE"
        ] = (
            result[
                factor_name
            ].notna()
            & (
                result[
                    f"{factor_name}_COV50_COMPONENTS"
                ]
                >= minimum_components
            )
            & (
                financial_match
                >= MODERATE_MATCH_THRESHOLD
            )
        ).astype(
            int
        )

        result[
            f"{factor_name}_VALID_MAIN"
        ] = (
            result[
                factor_name
            ].notna()
            & (
                result[
                    f"{factor_name}_COV70_COMPONENTS"
                ]
                >= minimum_components
            )
            & (
                financial_match
                >= MAIN_MATCH_THRESHOLD
            )
        ).astype(
            int
        )

        result[
            f"{factor_name}_VALID_STRICT"
        ] = (
            result[
                factor_name
            ].notna()
            & (
                result[
                    f"{factor_name}_COV80_COMPONENTS"
                ]
                >= minimum_components
            )
            & (
                financial_match
                >= STRICT_MATCH_THRESHOLD
            )
        ).astype(
            int
        )

        for quality in [
            "MODERATE",
            "MAIN",
            "STRICT",
        ]:

            result[
                f"{factor_name}_{quality}"
            ] = result[
                factor_name
            ].where(
                result[
                    f"{factor_name}_VALID_{quality}"
                ]
                == 1
            )

    return result


# ============================================================
# 10. FINANCIAL ARCHITECTURE COMPOSITES
# ============================================================

def build_signed_factor_composite(
    df: pd.DataFrame,
    component_map: dict[str, int],
    output_name: str,
    minimum_components: int,
) -> pd.DataFrame:
    """
    Faktor düzeyindeki bileşenlerden composite üretir.
    """

    result = df.copy()

    component_columns = []

    for factor_name, direction in (
        component_map.items()
    ):

        component_column = (
            f"RB_COMPOSITE_COMPONENT_"
            f"{output_name}_{factor_name}"
        )

        result[
            component_column
        ] = (
            direction
            * result[
                factor_name
            ]
        )

        component_columns.append(
            component_column
        )

    (
        score,
        count,
        share,
    ) = row_mean_with_minimum(
        df=result,
        columns=component_columns,
        minimum_components=minimum_components,
    )

    result[
        output_name
    ] = score

    result[
        f"{output_name}_AVAILABLE_COMPONENTS"
    ] = count

    result[
        f"{output_name}_COMPONENT_SHARE"
    ] = share

    return result


def add_composite_quality(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Composite kalite bayraklarını üretir.
    """

    result = df.copy()

    for quality in [
        "MODERATE",
        "MAIN",
        "STRICT",
    ]:

        financial_flags = [
            result[
                f"{factor}_VALID_{quality}"
            ]
            == 1
            for factor in (
                FINANCIAL_RISK_COMPONENTS.keys()
            )
        ]

        financial_valid_count = sum(
            flag.astype(int)
            for flag in financial_flags
        )

        result[
            f"FINANCIAL_ARCHITECTURE_RISK_VALID_{quality}"
        ] = (
            financial_valid_count
            >= 2
        ).astype(
            int
        )

        result[
            f"FINANCIAL_ARCHITECTURE_RISK_{quality}"
        ] = result[
            "FINANCIAL_ARCHITECTURE_RISK"
        ].where(
            result[
                f"FINANCIAL_ARCHITECTURE_RISK_VALID_{quality}"
            ]
            == 1
        )

        extended_flags = [
            result[
                f"{factor}_VALID_{quality}"
            ]
            == 1
            for factor in (
                EXTENDED_RISK_COMPONENTS.keys()
            )
        ]

        extended_valid_count = sum(
            flag.astype(int)
            for flag in extended_flags
        )

        result[
            f"EXTENDED_ARCHITECTURE_RISK_VALID_{quality}"
        ] = (
            extended_valid_count
            >= 3
        ).astype(
            int
        )

        result[
            f"EXTENDED_ARCHITECTURE_RISK_{quality}"
        ] = result[
            "EXTENDED_ARCHITECTURE_RISK"
        ].where(
            result[
                f"EXTENDED_ARCHITECTURE_RISK_VALID_{quality}"
            ]
            == 1
        )

    return result


# ============================================================
# 11. LEAVE-ONE-COMPONENT-OUT
# ============================================================

def build_leave_one_out_factors(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Her üç bileşenli faktör için bir bileşen çıkarılarak alternatif
    skorlar oluşturur.
    """

    result = df.copy()

    rows = []

    for factor_name, definition in (
        FACTOR_DEFINITIONS.items()
    ):

        components = list(
            definition[
                "components"
            ].items()
        )

        if len(components) < 3:
            continue

        for omitted_variable, _ in components:

            retained_columns = []

            for variable, direction in components:

                if variable == omitted_variable:
                    continue

                source = (
                    f"RB_Z_{variable}"
                )

                temporary_column = (
                    f"RB_LOO_COMPONENT_"
                    f"{factor_name}_"
                    f"OMIT_{omitted_variable}_"
                    f"{variable}"
                )

                result[
                    temporary_column
                ] = (
                    direction
                    * result[
                        source
                    ]
                )

                retained_columns.append(
                    temporary_column
                )

            output_name = (
                f"LOO_{factor_name}_"
                f"OMIT_{omitted_variable}"
            )

            result[
                output_name
            ] = result[
                retained_columns
            ].mean(
                axis=1,
                skipna=True,
            )

            result[
                output_name
            ] = result[
                output_name
            ].where(
                result[
                    retained_columns
                ]
                .notna()
                .sum(
                    axis=1
                )
                == len(
                    retained_columns
                )
            )

            correlation = (
                result[
                    [
                        factor_name,
                        output_name,
                    ]
                ]
                .corr(
                    min_periods=5
                )
                .iloc[
                    0,
                    1,
                ]
            )

            rows.append(
                {
                    "ORIGINAL_FACTOR": factor_name,
                    "OMITTED_VARIABLE": omitted_variable,
                    "LEAVE_ONE_OUT_FACTOR": output_name,
                    "CORRELATION_WITH_ORIGINAL": correlation,
                    "NON_MISSING_ETFS": int(
                        result[
                            output_name
                        ].notna().sum()
                    ),
                }
            )

    return (
        result,
        pd.DataFrame(
            rows
        ),
    )


# ============================================================
# 12. PCA ROBUSTNESS
# ============================================================

def orient_pca_loadings(
    loadings: np.ndarray,
    columns: list[str],
) -> np.ndarray:
    """
    PCA işaret belirsizliğini ekonomik yorum için stabilize eder.

    Birinci bileşenin vulnerability ve growth-duration ile pozitif,
    internal capacity ile negatif olması tercih edilir.
    """

    oriented = loadings.copy()

    if oriented.shape[1] == 0:

        return oriented

    orientation_score = 0.0

    direction_map = {
        "INTERNAL_FINANCIAL_CAPACITY_MAIN": -1,
        "BALANCE_SHEET_VULNERABILITY_MAIN": 1,
        "GROWTH_DURATION_EXPOSURE_REBUILT_MAIN": 1,
        "PORTFOLIO_CONCENTRATION_REBUILT_MAIN": 1,
    }

    for index, column in enumerate(
        columns
    ):

        orientation_score += (
            direction_map.get(
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


def run_pca_robustness(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Main-quality non-overlapping faktörlerde PCA uygular.

    PCA yalnızca complete-case ETF kesitinde çalıştırılır.
    """

    pca_columns = [
        "INTERNAL_FINANCIAL_CAPACITY_MAIN",
        "BALANCE_SHEET_VULNERABILITY_MAIN",
        "GROWTH_DURATION_EXPOSURE_REBUILT_MAIN",
        "PORTFOLIO_CONCENTRATION_REBUILT_MAIN",
    ]

    available_columns = [
        column
        for column in pca_columns
        if column in df.columns
    ]

    pca_sample = (
        df[
            [
                "ETF_ID",
                "ETF_NAME",
            ]
            + available_columns
        ]
        .dropna(
            subset=available_columns
        )
        .copy()
    )

    if (
        len(available_columns) < 2
        or len(pca_sample) < 5
    ):

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    matrix = pca_sample[
        available_columns
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

    eigenvectors = orient_pca_loadings(
        eigenvectors,
        available_columns,
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
            available_columns
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

    cumulative = 0.0

    for component_index, variance_ratio in enumerate(
        explained_variance_ratio
    ):

        cumulative += variance_ratio

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
                    variance_ratio
                ),
                "CUMULATIVE_EXPLAINED_VARIANCE": float(
                    cumulative
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
# 13. TANIMLAYICI İSTATİSTİKLER
# ============================================================

def build_descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Yeni faktörlerin tanımlayıcı istatistiklerini üretir.
    """

    variables = (
        PRIMARY_CHANNELS
        + [
            "FINANCIAL_ARCHITECTURE_RISK",
            "EXTENDED_ARCHITECTURE_RISK",
        ]
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


def build_factor_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Yeni main-quality faktör korelasyonlarını üretir.
    """

    variables = [
        "INTERNAL_FINANCIAL_CAPACITY_MAIN",
        "BALANCE_SHEET_VULNERABILITY_MAIN",
        "GROWTH_DURATION_EXPOSURE_REBUILT_MAIN",
        "PORTFOLIO_CONCENTRATION_REBUILT_MAIN",
        "FINANCIAL_ARCHITECTURE_RISK_MAIN",
        "EXTENDED_ARCHITECTURE_RISK_MAIN",
    ]

    available = [
        variable
        for variable in variables
        if variable in df.columns
    ]

    correlation = (
        df[
            available
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
# 14. KALİTE ÖZETİ
# ============================================================

def build_quality_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Faktör bazında geçerli ETF sayılarını raporlar.
    """

    factors = (
        PRIMARY_CHANNELS
        + [
            "FINANCIAL_ARCHITECTURE_RISK",
            "EXTENDED_ARCHITECTURE_RISK",
        ]
    )

    rows = []

    for factor in factors:

        rows.append(
            {
                "FACTOR": factor,
                "MODERATE_VALID_ETFS": int(
                    df[
                        f"{factor}_VALID_MODERATE"
                    ].sum()
                ),
                "MAIN_VALID_ETFS": int(
                    df[
                        f"{factor}_VALID_MAIN"
                    ].sum()
                ),
                "STRICT_VALID_ETFS": int(
                    df[
                        f"{factor}_VALID_STRICT"
                    ].sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. RANKINGS
# ============================================================

def build_rankings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main-quality ETF sıralamalarını üretir.
    """

    variables = [
        "INTERNAL_FINANCIAL_CAPACITY_MAIN",
        "BALANCE_SHEET_VULNERABILITY_MAIN",
        "GROWTH_DURATION_EXPOSURE_REBUILT_MAIN",
        "PORTFOLIO_CONCENTRATION_REBUILT_MAIN",
        "FINANCIAL_ARCHITECTURE_RISK_MAIN",
        "EXTENDED_ARCHITECTURE_RISK_MAIN",
    ]

    available = [
        variable
        for variable in variables
        if variable in df.columns
    ]

    ranking = df[
        [
            "ETF_ID",
            "ETF_NAME",
            "FINANCIAL_MATCH_WEIGHT",
        ]
        + available
    ].copy()

    for variable in available:

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
        by="FINANCIAL_ARCHITECTURE_RISK_MAIN",
        ascending=False,
        na_position="last",
    )


# ============================================================
# 16. VARIABLE DICTIONARY
# ============================================================

def build_variable_dictionary() -> pd.DataFrame:
    """
    Yeni faktörlerin metodolojik sözlüğünü oluşturur.
    """

    rows = []

    for factor_name, definition in (
        FACTOR_DEFINITIONS.items()
    ):

        formula_parts = []

        for variable, direction in (
            definition[
                "components"
            ].items()
        ):

            sign = (
                "+"
                if direction == 1
                else "-"
            )

            formula_parts.append(
                f"{sign} Z({variable})"
            )

        rows.append(
            {
                "VARIABLE": factor_name,
                "TYPE": (
                    "Non-overlapping theory-based composite"
                ),
                "FORMULA": (
                    "Equal-weight mean of "
                    + " ".join(
                        formula_parts
                    )
                ),
                "THEORY": definition[
                    "theory"
                ],
                "EXPECTED_CPU_INTERACTION": definition[
                    "expected_cpu_interaction"
                ],
                "RESEARCH_QUESTIONS": definition[
                    "research_questions"
                ],
                "PRIMARY_OR_ROBUSTNESS": (
                    "Primary mechanism channel"
                ),
            }
        )

    rows.extend(
        [
            {
                "VARIABLE": (
                    "FINANCIAL_ARCHITECTURE_RISK"
                ),
                "TYPE": (
                    "Non-overlapping theory-based composite"
                ),
                "FORMULA": (
                    "- Internal Financial Capacity "
                    "+ Balance-Sheet Vulnerability "
                    "+ Growth-Duration Exposure"
                ),
                "THEORY": (
                    "Firm-level financial architecture risk excluding "
                    "portfolio concentration."
                ),
                "EXPECTED_CPU_INTERACTION": "Negative",
                "RESEARCH_QUESTIONS": "RQ1, RQ2, RQ3",
                "PRIMARY_OR_ROBUSTNESS": (
                    "Composite robustness"
                ),
            },
            {
                "VARIABLE": (
                    "EXTENDED_ARCHITECTURE_RISK"
                ),
                "TYPE": (
                    "Non-overlapping theory-based composite"
                ),
                "FORMULA": (
                    "- Internal Financial Capacity "
                    "+ Balance-Sheet Vulnerability "
                    "+ Growth-Duration Exposure "
                    "+ Portfolio Concentration"
                ),
                "THEORY": (
                    "Extended architecture risk including portfolio "
                    "concentration."
                ),
                "EXPECTED_CPU_INTERACTION": "Negative",
                "RESEARCH_QUESTIONS": "RQ1, RQ2, RQ3",
                "PRIMARY_OR_ROBUSTNESS": (
                    "Secondary robustness only"
                ),
            },
            {
                "VARIABLE": "PCA_COMPONENT_1",
                "TYPE": "Data-driven principal component",
                "FORMULA": (
                    "First principal component of the four "
                    "non-overlapping main-quality channels"
                ),
                "THEORY": (
                    "Data-driven common architecture dimension"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Determined from economically oriented loadings"
                ),
                "RESEARCH_QUESTIONS": "Robustness",
                "PRIMARY_OR_ROBUSTNESS": (
                    "PCA robustness only"
                ),
            },
        ]
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. VALIDATION
# ============================================================

def build_validation(
    df: pd.DataFrame,
    parameters: pd.DataFrame,
    correlations: pd.DataFrame,
    quality_summary: pd.DataFrame,
    pca_variance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Yeni faktörlerin mekanik ve metodolojik doğrulamasını yapar.
    """

    main_channel_columns = [
        f"{factor}_MAIN"
        for factor in PRIMARY_CHANNELS
    ]

    duplicate_etfs = int(
        df.duplicated(
            subset=[
                "ETF_ID",
            ],
            keep=False,
        ).sum()
    )

    cross_correlation = (
        df[
            main_channel_columns
        ]
        .corr(
            min_periods=5
        )
    )

    off_diagonal = (
        cross_correlation
        .where(
            ~np.eye(
                len(
                    cross_correlation
                ),
                dtype=bool,
            )
        )
        .abs()
        .stack()
    )

    maximum_channel_correlation = (
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
                duplicate_etfs == 0
            ),
        },
        {
            "CHECK": (
                "STANDARDIZATION_PARAMETER_ROWS"
            ),
            "VALUE": len(
                parameters
            ),
            "PASS": int(
                len(
                    parameters
                )
                == len(
                    collect_source_variables()
                )
            ),
        },
        {
            "CHECK": (
                "PRIMARY_CHANNELS_CREATED"
            ),
            "VALUE": sum(
                factor in df.columns
                for factor in PRIMARY_CHANNELS
            ),
            "PASS": int(
                all(
                    factor in df.columns
                    for factor in PRIMARY_CHANNELS
                )
            ),
        },
        {
            "CHECK": (
                "FINANCIAL_COMPOSITE_CREATED"
            ),
            "VALUE": int(
                "FINANCIAL_ARCHITECTURE_RISK"
                in df.columns
            ),
            "PASS": int(
                "FINANCIAL_ARCHITECTURE_RISK"
                in df.columns
            ),
        },
        {
            "CHECK": (
                "MAX_ABSOLUTE_PRIMARY_CHANNEL_CORRELATION"
            ),
            "VALUE": maximum_channel_correlation,
            "PASS": int(
                pd.notna(
                    maximum_channel_correlation
                )
                and maximum_channel_correlation
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
                == 6
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
    ]

    for factor in PRIMARY_CHANNELS:

        rows.append(
            {
                "CHECK": (
                    f"{factor}_MAIN_VALID_ETFS"
                ),
                "VALUE": int(
                    df[
                        f"{factor}_VALID_MAIN"
                    ].sum()
                ),
                "PASS": int(
                    df[
                        f"{factor}_VALID_MAIN"
                    ].sum()
                    > 0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 86)
    print("24 - REBUILT NON-OVERLAPPING ARCHITECTURE FACTORS")
    print("=" * 86)

    print(
        "\n1/10 - Quality-adjusted architecture paneli okunuyor..."
    )

    architecture = normalize_columns(
        read_data(
            INPUT_FILE
        )
    )

    source_variables = (
        collect_source_variables()
    )

    required_columns = [
        "ETF_ID",
        "ETF_NAME",
        "FINANCIAL_MATCH_WEIGHT",
    ] + source_variables

    for definition in (
        FACTOR_DEFINITIONS.values()
    ):

        required_columns.extend(
            definition[
                "coverage_columns"
            ].values()
        )

    require_columns(
        architecture,
        sorted(
            set(
                required_columns
            )
        ),
    )

    print(
        f"ETF sayısı: "
        f"{len(architecture):,}"
    )

    print(
        "\n2/10 - Standardizasyon parametreleri hazırlanıyor..."
    )

    parameters = build_standardization_parameters(
        df=architecture,
        source_variables=source_variables,
    )

    print(
        "\n3/10 - Kaynak değişkenler winsorize ve standardize ediliyor..."
    )

    rebuilt_panel = create_standardized_sources(
        df=architecture,
        parameters=parameters,
    )

    print(
        "\n4/10 - Non-overlapping architecture faktörleri oluşturuluyor..."
    )

    (
        rebuilt_panel,
        factor_components,
    ) = build_rebuilt_factors(
        rebuilt_panel
    )

    print(
        "\n5/10 - Faktör coverage ve kalite bayrakları oluşturuluyor..."
    )

    rebuilt_panel = add_factor_quality_flags(
        rebuilt_panel
    )

    print(
        "\n6/10 - Financial architecture composites oluşturuluyor..."
    )

    rebuilt_panel = build_signed_factor_composite(
        df=rebuilt_panel,
        component_map=FINANCIAL_RISK_COMPONENTS,
        output_name=(
            "FINANCIAL_ARCHITECTURE_RISK"
        ),
        minimum_components=2,
    )

    rebuilt_panel = build_signed_factor_composite(
        df=rebuilt_panel,
        component_map=EXTENDED_RISK_COMPONENTS,
        output_name=(
            "EXTENDED_ARCHITECTURE_RISK"
        ),
        minimum_components=3,
    )

    rebuilt_panel = add_composite_quality(
        rebuilt_panel
    )

    print(
        "\n7/10 - Leave-one-component-out faktörleri oluşturuluyor..."
    )

    (
        rebuilt_panel,
        leave_one_out_summary,
    ) = build_leave_one_out_factors(
        rebuilt_panel
    )

    print(
        "\n8/10 - PCA robustness analizi uygulanıyor..."
    )

    (
        pca_loadings,
        pca_variance,
        pca_scores,
    ) = run_pca_robustness(
        rebuilt_panel
    )

    if not pca_scores.empty:

        rebuilt_panel = rebuilt_panel.merge(
            pca_scores,
            on=[
                "ETF_ID",
                "ETF_NAME",
            ],
            how="left",
            validate="one_to_one",
        )

    print(
        "\n9/10 - Diagnostics ve raporlar hazırlanıyor..."
    )

    descriptives = build_descriptive_statistics(
        rebuilt_panel
    )

    correlations = build_factor_correlations(
        rebuilt_panel
    )

    quality_summary = build_quality_summary(
        rebuilt_panel
    )

    rankings = build_rankings(
        rebuilt_panel
    )

    variable_dictionary = (
        build_variable_dictionary()
    )

    validation = build_validation(
        df=rebuilt_panel,
        parameters=parameters,
        correlations=correlations,
        quality_summary=quality_summary,
        pca_variance=pca_variance,
    )

    print(
        "\n10/10 - Çıktılar kaydediliyor..."
    )

    rebuilt_panel.to_csv(
        REBUILT_PANEL_FILE,
        index=False,
    )

    rebuilt_panel.to_parquet(
        REBUILT_PANEL_PARQUET_FILE,
        index=False,
    )

    descriptives.to_csv(
        FACTOR_DESCRIPTIVE_FILE,
        index=False,
    )

    correlations.to_csv(
        FACTOR_CORRELATION_FILE,
        index=False,
    )

    factor_components.to_csv(
        FACTOR_COMPONENT_FILE,
        index=False,
    )

    parameters.to_csv(
        STANDARDIZATION_FILE,
        index=False,
    )

    quality_summary.to_csv(
        QUALITY_SUMMARY_FILE,
        index=False,
    )

    leave_one_out_summary.to_csv(
        LEAVE_ONE_OUT_CORRELATION_FILE,
        index=False,
    )

    leave_one_out_columns = [
        "ETF_ID",
        "ETF_NAME",
    ] + [
        column
        for column in rebuilt_panel.columns
        if column.startswith(
            "LOO_"
        )
    ]

    rebuilt_panel[
        leave_one_out_columns
    ].to_csv(
        LEAVE_ONE_OUT_FILE,
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
        RANKING_FILE,
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
        "\nREBUILT ARCHITECTURE FACTORS HAZIR"
    )

    print("=" * 86)

    print(
        "\nFaktör kalite özeti:"
    )

    print(
        quality_summary.to_string(
            index=False
        )
    )

    print(
        "\nYeni faktör descriptive statistics:"
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
        "\nMain-quality faktör korelasyonları:"
    )

    print(
        correlations.to_string(
            index=False
        )
    )

    print(
        "\nLeave-one-component-out özeti:"
    )

    if leave_one_out_summary.empty:

        print(
            "Üç bileşenli faktör bulunamadı."
        )

    else:

        print(
            leave_one_out_summary.to_string(
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
        "Ana ekonometrik analizde dört non-overlapping kanal "
        "ayrı ayrı test edilecektir. Financial Architecture Risk "
        "composite'i robustness olarak kullanılacaktır. Portfolio "
        "Concentration, finansal capacity ve vulnerability "
        "composite'lerinden ayrı tutulacaktır."
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        REBUILT_PANEL_FILE
    )

    print(
        FACTOR_DESCRIPTIVE_FILE
    )

    print(
        FACTOR_CORRELATION_FILE
    )

    print(
        QUALITY_SUMMARY_FILE
    )

    print(
        LEAVE_ONE_OUT_CORRELATION_FILE
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
        VALIDATION_FILE
    )

    print(
        VARIABLE_DICTIONARY_FILE
    )


if __name__ == "__main__":
    main()