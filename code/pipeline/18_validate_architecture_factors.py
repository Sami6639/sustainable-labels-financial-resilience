from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 18_validate_architecture_factors.py
#
# Amaç
# ----
# Script 17 tarafından üretilen teori-temelli architecture
# faktörlerine ölçüm-kalitesi ve coverage koşulları eklemek.
#
# Temel sorun
# ------------
# Düşük financial-match coverage taşıyan ETF'lerde de faktör
# hesaplanabildiği için ham faktör sıralamaları ekonomik olarak
# güvenilir olmayabilir.
#
# Bu script:
# - Genel financial-match filtreleri
# - Faktör-bazlı coverage kalite ölçüleri
# - Moderate ve strict faktör geçerlilik bayrakları
# - Quality-adjusted faktör skorları
# - Güvenilir ETF sıralamaları
# - Nihai ekonometrik örneklem önerileri
# üretir.
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
# 2. GİRDİ DOSYASI
# ============================================================

FACTOR_FILE = (
    OUTPUT_DIR
    / "17_portfolio_architecture_factors.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

QUALITY_PANEL_FILE = (
    OUTPUT_DIR
    / "18_quality_adjusted_architecture_factors.csv"
)

QUALITY_PANEL_PARQUET_FILE = (
    OUTPUT_DIR
    / "18_quality_adjusted_architecture_factors.parquet"
)

QUALITY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "18_architecture_factor_quality_summary.csv"
)

RELIABLE_RANKINGS_FILE = (
    OUTPUT_DIR
    / "18_reliable_architecture_factor_rankings.csv"
)

SAMPLE_COUNTS_FILE = (
    OUTPUT_DIR
    / "18_architecture_factor_sample_counts.csv"
)

FACTOR_CORRELATION_FILE = (
    OUTPUT_DIR
    / "18_quality_adjusted_factor_correlations.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "18_architecture_factor_quality_validation.csv"
)


# ============================================================
# 4. METODOLOJİK EŞİKLER
# ============================================================

# Genel firma-finansal eşleşme kalitesi
MODERATE_FINANCIAL_MATCH = 0.70
MAIN_FINANCIAL_MATCH = 0.80
STRICT_FINANCIAL_MATCH = 0.90

# Faktör bileşenlerinin ağırlıklı coverage kalitesi
MODERATE_FACTOR_COVERAGE = 0.50
MAIN_FACTOR_COVERAGE = 0.70
STRICT_FACTOR_COVERAGE = 0.80


# ============================================================
# 5. FAKTÖR COVERAGE TANIMLARI
# ============================================================

FACTOR_COVERAGE_MAP = {
    "FINANCIAL_RESILIENCE": [
        "COV_ROA",
        "COV_CASH_RATIO",
        "COV_LOG_ASSETS",
    ],

    "FINANCING_VULNERABILITY": [
        "COV_LEVERAGE",
        "COV_EXTERNAL_FINANCE_DEPENDENCE",
        "COV_CASH_RATIO",
        "COV_ROA",
    ],

    "GROWTH_DURATION_EXPOSURE": [
        "COV_CAPEX_INTENSITY",
        "COV_RD_INTENSITY",
        "COV_REVENUE_GROWTH",
    ],
}


FACTOR_VARIABLES = [
    "FINANCIAL_RESILIENCE",
    "FINANCING_VULNERABILITY",
    "GROWTH_DURATION_EXPOSURE",
    "PORTFOLIO_CONCENTRATION",
    "CORE_TRANSITION_SENSITIVITY",
    "EXTENDED_TRANSITION_SENSITIVITY",
]


# ============================================================
# 6. YARDIMCI FONKSİYONLAR
# ============================================================

def read_data(
    path: Path,
) -> pd.DataFrame:
    """
    CSV dosyasını okur.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
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


def minimum_available_coverage(
    df: pd.DataFrame,
    coverage_columns: list[str],
) -> pd.Series:
    """
    İlgili faktörün mevcut coverage sütunları içindeki minimum
    coverage değerini hesaplar.
    """

    numeric = df[
        coverage_columns
    ].apply(
        safe_numeric
    )

    return numeric.min(
        axis=1,
        skipna=True,
    )


def mean_available_coverage(
    df: pd.DataFrame,
    coverage_columns: list[str],
) -> pd.Series:
    """
    İlgili faktör bileşenlerinin ortalama coverage değerini hesaplar.
    """

    numeric = df[
        coverage_columns
    ].apply(
        safe_numeric
    )

    return numeric.mean(
        axis=1,
        skipna=True,
    )


def count_coverage_above(
    df: pd.DataFrame,
    coverage_columns: list[str],
    threshold: float,
) -> pd.Series:
    """
    Belirlenen coverage eşiğini sağlayan bileşen sayısını hesaplar.
    """

    numeric = df[
        coverage_columns
    ].apply(
        safe_numeric
    )

    return (
        numeric
        >= threshold
    ).sum(
        axis=1
    )


# ============================================================
# 7. FAKTÖR KALİTE ÖLÇÜLERİ
# ============================================================

def add_factor_quality_measures(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her faktör için coverage kalitesi ve geçerlilik bayrakları
    oluşturur.
    """

    result = df.copy()

    result[
        "FINANCIAL_MATCH_MODERATE"
    ] = (
        safe_numeric(
            result[
                "FINANCIAL_MATCH_WEIGHT"
            ]
        )
        >= MODERATE_FINANCIAL_MATCH
    ).astype(
        int
    )

    result[
        "FINANCIAL_MATCH_MAIN"
    ] = (
        safe_numeric(
            result[
                "FINANCIAL_MATCH_WEIGHT"
            ]
        )
        >= MAIN_FINANCIAL_MATCH
    ).astype(
        int
    )

    result[
        "FINANCIAL_MATCH_STRICT"
    ] = (
        safe_numeric(
            result[
                "FINANCIAL_MATCH_WEIGHT"
            ]
        )
        >= STRICT_FINANCIAL_MATCH
    ).astype(
        int
    )

    for (
        factor_name,
        coverage_columns,
    ) in FACTOR_COVERAGE_MAP.items():

        result[
            f"{factor_name}_MEAN_COVERAGE"
        ] = mean_available_coverage(
            df=result,
            coverage_columns=coverage_columns,
        )

        result[
            f"{factor_name}_MIN_COVERAGE"
        ] = minimum_available_coverage(
            df=result,
            coverage_columns=coverage_columns,
        )

        result[
            f"{factor_name}_COMPONENTS_COV50"
        ] = count_coverage_above(
            df=result,
            coverage_columns=coverage_columns,
            threshold=0.50,
        )

        result[
            f"{factor_name}_COMPONENTS_COV70"
        ] = count_coverage_above(
            df=result,
            coverage_columns=coverage_columns,
            threshold=0.70,
        )

        result[
            f"{factor_name}_COMPONENTS_COV80"
        ] = count_coverage_above(
            df=result,
            coverage_columns=coverage_columns,
            threshold=0.80,
        )

    # --------------------------------------------------------
    # Financial Resilience
    #
    # Üç bileşenden en az ikisi:
    # - Moderate: coverage >= 50%
    # - Main: coverage >= 70%
    # - Strict: coverage >= 80%
    # --------------------------------------------------------

    result[
        "FINANCIAL_RESILIENCE_VALID_MODERATE"
    ] = (
        (
            result[
                "FINANCIAL_RESILIENCE_COMPONENTS_COV50"
            ]
            >= 2
        )
        & (
            result[
                "FINANCIAL_MATCH_MODERATE"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "FINANCIAL_RESILIENCE_VALID_MAIN"
    ] = (
        (
            result[
                "FINANCIAL_RESILIENCE_COMPONENTS_COV70"
            ]
            >= 2
        )
        & (
            result[
                "FINANCIAL_MATCH_MAIN"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "FINANCIAL_RESILIENCE_VALID_STRICT"
    ] = (
        (
            result[
                "FINANCIAL_RESILIENCE_COMPONENTS_COV80"
            ]
            >= 2
        )
        & (
            result[
                "FINANCIAL_MATCH_STRICT"
            ]
            == 1
        )
    ).astype(
        int
    )

    # --------------------------------------------------------
    # Financing Vulnerability
    #
    # Leverage coverage yapısal olarak düşük olduğu için:
    # - Moderate: dört bileşenden en az ikisi >= 50%
    # - Main: en az üç bileşen >= 50% ve overall match >= 80%
    # - Strict: en az iki bileşen >= 70% ve leverage >= 50%
    # --------------------------------------------------------

    result[
        "FINANCING_VULNERABILITY_VALID_MODERATE"
    ] = (
        (
            result[
                "FINANCING_VULNERABILITY_COMPONENTS_COV50"
            ]
            >= 2
        )
        & (
            result[
                "FINANCIAL_MATCH_MODERATE"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "FINANCING_VULNERABILITY_VALID_MAIN"
    ] = (
        (
            result[
                "FINANCING_VULNERABILITY_COMPONENTS_COV50"
            ]
            >= 3
        )
        & (
            result[
                "FINANCIAL_MATCH_MAIN"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "FINANCING_VULNERABILITY_VALID_STRICT"
    ] = (
        (
            result[
                "FINANCING_VULNERABILITY_COMPONENTS_COV70"
            ]
            >= 2
        )
        & (
            safe_numeric(
                result[
                    "COV_LEVERAGE"
                ]
            )
            >= 0.50
        )
        & (
            result[
                "FINANCIAL_MATCH_STRICT"
            ]
            == 1
        )
    ).astype(
        int
    )

    # --------------------------------------------------------
    # Growth-Duration Exposure
    #
    # R&D coverage yapısal olarak düşük olduğu için:
    # - Moderate: üç bileşenden en az ikisi >= 50%
    # - Main: en az iki bileşen >= 70%
    # - Strict: en az iki bileşen >= 80%
    # --------------------------------------------------------

    result[
        "GROWTH_DURATION_EXPOSURE_VALID_MODERATE"
    ] = (
        (
            result[
                "GROWTH_DURATION_EXPOSURE_COMPONENTS_COV50"
            ]
            >= 2
        )
        & (
            result[
                "FINANCIAL_MATCH_MODERATE"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "GROWTH_DURATION_EXPOSURE_VALID_MAIN"
    ] = (
        (
            result[
                "GROWTH_DURATION_EXPOSURE_COMPONENTS_COV70"
            ]
            >= 2
        )
        & (
            result[
                "FINANCIAL_MATCH_MAIN"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "GROWTH_DURATION_EXPOSURE_VALID_STRICT"
    ] = (
        (
            result[
                "GROWTH_DURATION_EXPOSURE_COMPONENTS_COV80"
            ]
            >= 2
        )
        & (
            result[
                "FINANCIAL_MATCH_STRICT"
            ]
            == 1
        )
    ).astype(
        int
    )

    # Concentration her ETF için holdings tabanlı ve eksiksizdir.
    result[
        "PORTFOLIO_CONCENTRATION_VALID_MODERATE"
    ] = 1

    result[
        "PORTFOLIO_CONCENTRATION_VALID_MAIN"
    ] = 1

    result[
        "PORTFOLIO_CONCENTRATION_VALID_STRICT"
    ] = 1

    # --------------------------------------------------------
    # Core Transition Sensitivity
    # --------------------------------------------------------

    result[
        "CORE_TRANSITION_VALID_MODERATE_QUALITY"
    ] = (
        (
            result[
                "FINANCIAL_RESILIENCE_VALID_MODERATE"
            ]
            == 1
        )
        & (
            result[
                "GROWTH_DURATION_EXPOSURE_VALID_MODERATE"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "CORE_TRANSITION_VALID_MAIN_QUALITY"
    ] = (
        (
            result[
                "FINANCIAL_RESILIENCE_VALID_MAIN"
            ]
            == 1
        )
        & (
            result[
                "GROWTH_DURATION_EXPOSURE_VALID_MAIN"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "CORE_TRANSITION_VALID_STRICT_QUALITY"
    ] = (
        (
            result[
                "FINANCIAL_RESILIENCE_VALID_STRICT"
            ]
            == 1
        )
        & (
            result[
                "GROWTH_DURATION_EXPOSURE_VALID_STRICT"
            ]
            == 1
        )
    ).astype(
        int
    )

    # --------------------------------------------------------
    # Extended Transition Sensitivity
    # --------------------------------------------------------

    result[
        "EXTENDED_TRANSITION_VALID_MODERATE_QUALITY"
    ] = (
        (
            result[
                "CORE_TRANSITION_VALID_MODERATE_QUALITY"
            ]
            == 1
        )
        & (
            result[
                "FINANCING_VULNERABILITY_VALID_MODERATE"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "EXTENDED_TRANSITION_VALID_MAIN_QUALITY"
    ] = (
        (
            result[
                "CORE_TRANSITION_VALID_MAIN_QUALITY"
            ]
            == 1
        )
        & (
            result[
                "FINANCING_VULNERABILITY_VALID_MAIN"
            ]
            == 1
        )
    ).astype(
        int
    )

    result[
        "EXTENDED_TRANSITION_VALID_STRICT_QUALITY"
    ] = (
        (
            result[
                "CORE_TRANSITION_VALID_STRICT_QUALITY"
            ]
            == 1
        )
        & (
            result[
                "FINANCING_VULNERABILITY_VALID_STRICT"
            ]
            == 1
        )
    ).astype(
        int
    )

    return result


# ============================================================
# 8. QUALITY-ADJUSTED FAKTÖRLER
# ============================================================

def add_quality_adjusted_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Yalnızca ilgili kalite bayrağını sağlayan ETF'ler için faktör
    skorlarını korur.
    """

    result = df.copy()

    base_factors = [
        "FINANCIAL_RESILIENCE",
        "FINANCING_VULNERABILITY",
        "GROWTH_DURATION_EXPOSURE",
        "PORTFOLIO_CONCENTRATION",
    ]

    for factor in base_factors:

        for quality_level in [
            "MODERATE",
            "MAIN",
            "STRICT",
        ]:

            flag = (
                f"{factor}_VALID_{quality_level}"
            )

            output = (
                f"{factor}_{quality_level}"
            )

            result[
                output
            ] = result[
                factor
            ].where(
                result[
                    flag
                ]
                == 1
            )

    for index_name in [
        "CORE_TRANSITION",
        "EXTENDED_TRANSITION",
    ]:

        raw_variable = (
            f"{index_name}_SENSITIVITY"
        )

        for quality_level in [
            "MODERATE",
            "MAIN",
            "STRICT",
        ]:

            flag = (
                f"{index_name}_VALID_"
                f"{quality_level}_QUALITY"
            )

            output = (
                f"{raw_variable}_{quality_level}"
            )

            result[
                output
            ] = result[
                raw_variable
            ].where(
                result[
                    flag
                ]
                == 1
            )

    return result


# ============================================================
# 9. KALİTE ÖZETİ
# ============================================================

def build_quality_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Faktör bazında kalite ve geçerli örneklem sayılarını raporlar.
    """

    rows = []

    factors = [
        "FINANCIAL_RESILIENCE",
        "FINANCING_VULNERABILITY",
        "GROWTH_DURATION_EXPOSURE",
        "PORTFOLIO_CONCENTRATION",
    ]

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

    rows.extend(
        [
            {
                "FACTOR": (
                    "CORE_TRANSITION_SENSITIVITY"
                ),

                "MODERATE_VALID_ETFS": int(
                    df[
                        "CORE_TRANSITION_VALID_MODERATE_QUALITY"
                    ].sum()
                ),

                "MAIN_VALID_ETFS": int(
                    df[
                        "CORE_TRANSITION_VALID_MAIN_QUALITY"
                    ].sum()
                ),

                "STRICT_VALID_ETFS": int(
                    df[
                        "CORE_TRANSITION_VALID_STRICT_QUALITY"
                    ].sum()
                ),
            },

            {
                "FACTOR": (
                    "EXTENDED_TRANSITION_SENSITIVITY"
                ),

                "MODERATE_VALID_ETFS": int(
                    df[
                        "EXTENDED_TRANSITION_VALID_MODERATE_QUALITY"
                    ].sum()
                ),

                "MAIN_VALID_ETFS": int(
                    df[
                        "EXTENDED_TRANSITION_VALID_MAIN_QUALITY"
                    ].sum()
                ),

                "STRICT_VALID_ETFS": int(
                    df[
                        "EXTENDED_TRANSITION_VALID_STRICT_QUALITY"
                    ].sum()
                ),
            },
        ]
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. SAMPLE COUNTS
# ============================================================

def build_sample_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genel ve faktör-bazlı örneklem sayılarını üretir.
    """

    rows = [
        {
            "SAMPLE": "ALL_ETFS",
            "N_ETFS": len(
                df
            ),
        },

        {
            "SAMPLE": (
                "FINANCIAL_MATCH_AT_LEAST_70"
            ),
            "N_ETFS": int(
                df[
                    "FINANCIAL_MATCH_MODERATE"
                ].sum()
            ),
        },

        {
            "SAMPLE": (
                "FINANCIAL_MATCH_AT_LEAST_80"
            ),
            "N_ETFS": int(
                df[
                    "FINANCIAL_MATCH_MAIN"
                ].sum()
            ),
        },

        {
            "SAMPLE": (
                "FINANCIAL_MATCH_AT_LEAST_90"
            ),
            "N_ETFS": int(
                df[
                    "FINANCIAL_MATCH_STRICT"
                ].sum()
            ),
        },

        {
            "SAMPLE": (
                "CORE_TRANSITION_MODERATE"
            ),
            "N_ETFS": int(
                df[
                    "CORE_TRANSITION_VALID_MODERATE_QUALITY"
                ].sum()
            ),
        },

        {
            "SAMPLE": (
                "CORE_TRANSITION_MAIN"
            ),
            "N_ETFS": int(
                df[
                    "CORE_TRANSITION_VALID_MAIN_QUALITY"
                ].sum()
            ),
        },

        {
            "SAMPLE": (
                "CORE_TRANSITION_STRICT"
            ),
            "N_ETFS": int(
                df[
                    "CORE_TRANSITION_VALID_STRICT_QUALITY"
                ].sum()
            ),
        },

        {
            "SAMPLE": (
                "EXTENDED_TRANSITION_MODERATE"
            ),
            "N_ETFS": int(
                df[
                    "EXTENDED_TRANSITION_VALID_MODERATE_QUALITY"
                ].sum()
            ),
        },

        {
            "SAMPLE": (
                "EXTENDED_TRANSITION_MAIN"
            ),
            "N_ETFS": int(
                df[
                    "EXTENDED_TRANSITION_VALID_MAIN_QUALITY"
                ].sum()
            ),
        },

        {
            "SAMPLE": (
                "EXTENDED_TRANSITION_STRICT"
            ),
            "N_ETFS": int(
                df[
                    "EXTENDED_TRANSITION_VALID_STRICT_QUALITY"
                ].sum()
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. GÜVENİLİR SIRALAMALAR
# ============================================================

def build_reliable_rankings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main-quality transition-sensitivity örneklemini sıralar.
    """

    columns = [
        "ETF_ID",
        "ETF_NAME",
        "FINANCIAL_MATCH_WEIGHT",
        "FINANCIAL_RESILIENCE_MAIN",
        "FINANCING_VULNERABILITY_MAIN",
        "GROWTH_DURATION_EXPOSURE_MAIN",
        "PORTFOLIO_CONCENTRATION_MAIN",
        "CORE_TRANSITION_SENSITIVITY_MAIN",
        "EXTENDED_TRANSITION_SENSITIVITY_MAIN",
    ]

    ranking = df[
        columns
    ].copy()

    ranking[
        "CORE_TRANSITION_MAIN_RANK"
    ] = ranking[
        "CORE_TRANSITION_SENSITIVITY_MAIN"
    ].rank(
        method="min",
        ascending=False,
    )

    ranking[
        "EXTENDED_TRANSITION_MAIN_RANK"
    ] = ranking[
        "EXTENDED_TRANSITION_SENSITIVITY_MAIN"
    ].rank(
        method="min",
        ascending=False,
    )

    return ranking.sort_values(
        by="CORE_TRANSITION_SENSITIVITY_MAIN",
        ascending=False,
        na_position="last",
    )


# ============================================================
# 12. QUALITY-ADJUSTED CORRELATIONS
# ============================================================

def build_quality_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main-quality faktörler arasındaki korelasyonları üretir.
    """

    columns = [
        "FINANCIAL_RESILIENCE_MAIN",
        "FINANCING_VULNERABILITY_MAIN",
        "GROWTH_DURATION_EXPOSURE_MAIN",
        "PORTFOLIO_CONCENTRATION_MAIN",
        "CORE_TRANSITION_SENSITIVITY_MAIN",
        "EXTENDED_TRANSITION_SENSITIVITY_MAIN",
    ]

    correlation = (
        df[
            columns
        ]
        .apply(
            safe_numeric
        )
        .corr(
            method="pearson",
            min_periods=10,
        )
    )

    correlation.index.name = (
        "VARIABLE"
    )

    return correlation.reset_index()


# ============================================================
# 13. VALIDATION
# ============================================================

def build_validation(
    df: pd.DataFrame,
    quality_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kalite filtrelerinin doğru uygulandığını kontrol eder.
    """

    unreliable_core_scores = int(
        (
            df[
                "CORE_TRANSITION_SENSITIVITY_MAIN"
            ].notna()
            & (
                df[
                    "FINANCIAL_MATCH_WEIGHT"
                ]
                < MAIN_FINANCIAL_MATCH
            )
        ).sum()
    )

    unreliable_extended_scores = int(
        (
            df[
                "EXTENDED_TRANSITION_SENSITIVITY_MAIN"
            ].notna()
            & (
                df[
                    "FINANCIAL_MATCH_WEIGHT"
                ]
                < MAIN_FINANCIAL_MATCH
            )
        ).sum()
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
                ) == 66
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
                ) == 6
            ),
        },

        {
            "CHECK": (
                "UNRELIABLE_CORE_MAIN_SCORES"
            ),
            "VALUE": unreliable_core_scores,
            "PASS": int(
                unreliable_core_scores
                == 0
            ),
        },

        {
            "CHECK": (
                "UNRELIABLE_EXTENDED_MAIN_SCORES"
            ),
            "VALUE": unreliable_extended_scores,
            "PASS": int(
                unreliable_extended_scores
                == 0
            ),
        },

        {
            "CHECK": (
                "CORE_MAIN_VALID_ETFS"
            ),
            "VALUE": int(
                df[
                    "CORE_TRANSITION_VALID_MAIN_QUALITY"
                ].sum()
            ),
            "PASS": int(
                df[
                    "CORE_TRANSITION_VALID_MAIN_QUALITY"
                ].sum()
                > 0
            ),
        },

        {
            "CHECK": (
                "EXTENDED_MAIN_VALID_ETFS"
            ),
            "VALUE": int(
                df[
                    "EXTENDED_TRANSITION_VALID_MAIN_QUALITY"
                ].sum()
            ),
            "PASS": int(
                df[
                    "EXTENDED_TRANSITION_VALID_MAIN_QUALITY"
                ].sum()
                > 0
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 14. ANA PIPELINE
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
        "18 - ARCHITECTURE FACTOR QUALITY VALIDATION"
    )

    print(
        "=" * 78
    )

    print(
        "\n1/7 - Architecture factor paneli okunuyor..."
    )

    factor_panel = read_data(
        FACTOR_FILE
    )

    required_columns = [
        "ETF_ID",
        "ETF_NAME",
        "FINANCIAL_MATCH_WEIGHT",
    ] + FACTOR_VARIABLES

    for coverage_columns in (
        FACTOR_COVERAGE_MAP.values()
    ):

        required_columns.extend(
            coverage_columns
        )

    require_columns(
        factor_panel,
        sorted(
            set(
                required_columns
            )
        ),
    )

    print(
        f"ETF sayısı: "
        f"{len(factor_panel):,}"
    )

    print(
        "\n2/7 - Faktör coverage kalitesi hesaplanıyor..."
    )

    quality_panel = add_factor_quality_measures(
        factor_panel
    )

    print(
        "\n3/7 - Quality-adjusted faktör skorları oluşturuluyor..."
    )

    quality_panel = add_quality_adjusted_scores(
        quality_panel
    )

    print(
        "\n4/7 - Kalite özeti ve örneklem sayıları hazırlanıyor..."
    )

    quality_summary = build_quality_summary(
        quality_panel
    )

    sample_counts = build_sample_counts(
        quality_panel
    )

    print(
        "\n5/7 - Güvenilir faktör sıralamaları hazırlanıyor..."
    )

    reliable_rankings = build_reliable_rankings(
        quality_panel
    )

    factor_correlations = build_quality_correlations(
        quality_panel
    )

    print(
        "\n6/7 - Validation kontrolleri çalıştırılıyor..."
    )

    validation = build_validation(
        df=quality_panel,
        quality_summary=quality_summary,
    )

    print(
        "\n7/7 - Çıktılar kaydediliyor..."
    )

    quality_panel.to_csv(
        QUALITY_PANEL_FILE,
        index=False,
    )

    quality_panel.to_parquet(
        QUALITY_PANEL_PARQUET_FILE,
        index=False,
    )

    quality_summary.to_csv(
        QUALITY_SUMMARY_FILE,
        index=False,
    )

    reliable_rankings.to_csv(
        RELIABLE_RANKINGS_FILE,
        index=False,
    )

    sample_counts.to_csv(
        SAMPLE_COUNTS_FILE,
        index=False,
    )

    factor_correlations.to_csv(
        FACTOR_CORRELATION_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    print(
        "\nARCHITECTURE FACTOR QUALITY VALIDATION HAZIR"
    )

    print(
        "=" * 78
    )

    print(
        "\nFaktör kalite özeti:"
    )

    print(
        quality_summary.to_string(
            index=False
        )
    )

    print(
        "\nÖrneklem sayıları:"
    )

    print(
        sample_counts.to_string(
            index=False
        )
    )

    print(
        "\nMain-quality Core Transition Sensitivity "
        "en yüksek 10 ETF:"
    )

    print(
        reliable_rankings.loc[
            reliable_rankings[
                "CORE_TRANSITION_SENSITIVITY_MAIN"
            ].notna(),
            [
                "ETF_NAME",
                "CORE_TRANSITION_SENSITIVITY_MAIN",
                "FINANCIAL_RESILIENCE_MAIN",
                "GROWTH_DURATION_EXPOSURE_MAIN",
                "PORTFOLIO_CONCENTRATION_MAIN",
                "FINANCIAL_MATCH_WEIGHT",
            ],
        ]
        .head(
            10
        )
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
        "\nAna çıktı dosyaları:"
    )

    print(
        QUALITY_PANEL_FILE
    )

    print(
        QUALITY_SUMMARY_FILE
    )

    print(
        RELIABLE_RANKINGS_FILE
    )

    print(
        SAMPLE_COUNTS_FILE
    )

    print(
        FACTOR_CORRELATION_FILE
    )

    print(
        VALIDATION_FILE
    )


if __name__ == "__main__":
    main()