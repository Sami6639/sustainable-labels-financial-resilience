from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 16_analyze_portfolio_architecture.py
#
# Amaç
# ----
# Script 15 tarafından oluşturulan portfolio architecture panelini
# analiz etmek ve makaledeki tanımlayıcı tabloları üretmek.
#
# Ana çıktılar
# -------------
# - Descriptive statistics
# - Correlation matrix
# - Coverage-adjusted descriptive statistics
# - Concentration rankings
# - Architecture extremes
# - Standardized architecture variables
# - Validation report
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

ARCHITECTURE_FILE = (
    OUTPUT_DIR
    / "15_portfolio_architecture_panel.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

DESCRIPTIVE_FILE = (
    OUTPUT_DIR
    / "16_portfolio_architecture_descriptive_statistics.csv"
)

CORRELATION_FILE = (
    OUTPUT_DIR
    / "16_portfolio_architecture_correlation_matrix.csv"
)

COVERAGE_ADJUSTED_FILE = (
    OUTPUT_DIR
    / "16_coverage_adjusted_descriptive_statistics.csv"
)

CONCENTRATION_RANKING_FILE = (
    OUTPUT_DIR
    / "16_portfolio_concentration_rankings.csv"
)

ARCHITECTURE_EXTREMES_FILE = (
    OUTPUT_DIR
    / "16_portfolio_architecture_extremes.csv"
)

STANDARDIZED_PANEL_FILE = (
    OUTPUT_DIR
    / "16_portfolio_architecture_standardized.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "16_portfolio_architecture_analysis_validation.csv"
)


# ============================================================
# 4. ANALİZ DEĞİŞKENLERİ
# ============================================================

ARCHITECTURE_VARIABLES = [
    "PW_ROA",
    "PW_LEVERAGE",
    "PW_CASH_RATIO",
    "PW_CAPEX_INTENSITY",
    "PW_RD_INTENSITY",
    "PW_REVENUE_GROWTH",
    "PW_EXTERNAL_FINANCE_DEPENDENCE",
    "PW_LOG_ASSETS",
    "HHI",
    "TOP10_WEIGHT",
    "EFFECTIVE_NUMBER_OF_HOLDINGS",
]

COVERAGE_MAP = {
    "PW_ROA": "COV_ROA",
    "PW_LEVERAGE": "COV_LEVERAGE",
    "PW_CASH_RATIO": "COV_CASH_RATIO",
    "PW_CAPEX_INTENSITY": "COV_CAPEX_INTENSITY",
    "PW_RD_INTENSITY": "COV_RD_INTENSITY",
    "PW_REVENUE_GROWTH": "COV_REVENUE_GROWTH",
    "PW_EXTERNAL_FINANCE_DEPENDENCE": (
        "COV_EXTERNAL_FINANCE_DEPENDENCE"
    ),
    "PW_LOG_ASSETS": "COV_LOG_ASSETS",
}


# ============================================================
# 5. YARDIMCI FONKSİYONLAR
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

    if pd.isna(
        std_value
    ) or std_value == 0:
        return pd.Series(
            np.nan,
            index=series.index,
        )

    return (
        numeric
        - mean_value
    ) / std_value


# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

def build_descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ana architecture değişkenleri için tanımlayıcı istatistikler.
    """

    rows = []

    for variable in ARCHITECTURE_VARIABLES:

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


# ============================================================
# 7. COVERAGE-ADJUSTED DESCRIPTIVES
# ============================================================

def build_coverage_adjusted_descriptives(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her karakteristik için farklı coverage eşiklerinde
    tanımlayıcı istatistikler üretir.
    """

    rows = []

    for variable, coverage_variable in COVERAGE_MAP.items():

        for threshold in [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ]:

            subset = df.loc[
                safe_numeric(
                    df[
                        coverage_variable
                    ]
                )
                >= threshold
            ].copy()

            values = safe_numeric(
                subset[
                    variable
                ]
            )

            rows.append(
                {
                    "VARIABLE": variable,
                    "COVERAGE_VARIABLE": (
                        coverage_variable
                    ),
                    "COVERAGE_THRESHOLD": (
                        threshold
                    ),
                    "N_ETFS": int(
                        len(
                            subset
                        )
                    ),
                    "NON_MISSING_N": int(
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
                    "MEDIAN": float(
                        values.median()
                    ),
                    "MIN": float(
                        values.min()
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
# 8. CORRELATION MATRIX
# ============================================================

def build_correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pearson correlation matrix.
    """

    numeric_data = df[
        ARCHITECTURE_VARIABLES
    ].apply(
        safe_numeric
    )

    correlation = numeric_data.corr(
        method="pearson",
        min_periods=10,
    )

    correlation.index.name = "VARIABLE"

    return correlation.reset_index()


# ============================================================
# 9. CONCENTRATION RANKINGS
# ============================================================

def build_concentration_rankings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ETF'leri yoğunlaşma ölçülerine göre sıralar.
    """

    columns = [
        "ETF_ID",
        "ETF_NAME",
        "N_HOLDINGS",
        "HHI",
        "TOP10_WEIGHT",
        "MAX_HOLDING_WEIGHT",
        "EFFECTIVE_NUMBER_OF_HOLDINGS",
        "FINANCIAL_MATCH_WEIGHT",
    ]

    ranking = df[
        columns
    ].copy()

    ranking[
        "HHI_RANK_HIGH_TO_LOW"
    ] = ranking[
        "HHI"
    ].rank(
        method="min",
        ascending=False,
    )

    ranking[
        "TOP10_RANK_HIGH_TO_LOW"
    ] = ranking[
        "TOP10_WEIGHT"
    ].rank(
        method="min",
        ascending=False,
    )

    ranking[
        "EFFECTIVE_N_RANK_HIGH_TO_LOW"
    ] = ranking[
        "EFFECTIVE_NUMBER_OF_HOLDINGS"
    ].rank(
        method="min",
        ascending=False,
    )

    return ranking.sort_values(
        by="HHI",
        ascending=False,
    )


# ============================================================
# 10. ARCHITECTURE EXTREMES
# ============================================================

def build_architecture_extremes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her architecture değişkeni için en yüksek ve en düşük ETF'leri seçer.
    """

    rows = []

    for variable in ARCHITECTURE_VARIABLES:

        valid = df.loc[
            df[
                variable
            ].notna()
        ].copy()

        if valid.empty:
            continue

        lowest = valid.nsmallest(
            5,
            variable,
        )

        highest = valid.nlargest(
            5,
            variable,
        )

        for _, row in lowest.iterrows():

            rows.append(
                {
                    "VARIABLE": variable,
                    "EXTREME_TYPE": "LOWEST",
                    "ETF_ID": row[
                        "ETF_ID"
                    ],
                    "ETF_NAME": row[
                        "ETF_NAME"
                    ],
                    "VALUE": row[
                        variable
                    ],
                    "FINANCIAL_MATCH_WEIGHT": row[
                        "FINANCIAL_MATCH_WEIGHT"
                    ],
                }
            )

        for _, row in highest.iterrows():

            rows.append(
                {
                    "VARIABLE": variable,
                    "EXTREME_TYPE": "HIGHEST",
                    "ETF_ID": row[
                        "ETF_ID"
                    ],
                    "ETF_NAME": row[
                        "ETF_NAME"
                    ],
                    "VALUE": row[
                        variable
                    ],
                    "FINANCIAL_MATCH_WEIGHT": row[
                        "FINANCIAL_MATCH_WEIGHT"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. STANDARDIZED PANEL
# ============================================================

def build_standardized_panel(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Architecture değişkenlerinin z-score sürümlerini üretir.
    """

    result = df.copy()

    for variable in ARCHITECTURE_VARIABLES:

        result[
            f"Z_{variable}"
        ] = zscore(
            result[
                variable
            ]
        )

    return result


# ============================================================
# 12. VALIDATION
# ============================================================

def build_validation(
    df: pd.DataFrame,
    descriptive: pd.DataFrame,
    correlation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analiz çıktılarının temel doğrulamasını yapar.
    """

    rows = [
        {
            "CHECK": "INPUT_ROWS",
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
            "CHECK": "UNIQUE_ETFS",
            "VALUE": df[
                "ETF_ID"
            ].nunique(),
            "PASS": int(
                df[
                    "ETF_ID"
                ].nunique()
                == len(
                    df
                )
            ),
        },
        {
            "CHECK": "DESCRIPTIVE_ROWS",
            "VALUE": len(
                descriptive
            ),
            "PASS": int(
                len(
                    descriptive
                )
                == len(
                    ARCHITECTURE_VARIABLES
                )
            ),
        },
        {
            "CHECK": "CORRELATION_ROWS",
            "VALUE": len(
                correlation
            ),
            "PASS": int(
                len(
                    correlation
                )
                == len(
                    ARCHITECTURE_VARIABLES
                )
            ),
        },
        {
            "CHECK": "MISSING_HHI",
            "VALUE": int(
                df[
                    "HHI"
                ].isna().sum()
            ),
            "PASS": int(
                df[
                    "HHI"
                ].isna().sum()
                == 0
            ),
        },
        {
            "CHECK": "MISSING_TOP10",
            "VALUE": int(
                df[
                    "TOP10_WEIGHT"
                ].isna().sum()
            ),
            "PASS": int(
                df[
                    "TOP10_WEIGHT"
                ].isna().sum()
                == 0
            ),
        },
        {
            "CHECK": "MISSING_EFFECTIVE_N",
            "VALUE": int(
                df[
                    "EFFECTIVE_NUMBER_OF_HOLDINGS"
                ].isna().sum()
            ),
            "PASS": int(
                df[
                    "EFFECTIVE_NUMBER_OF_HOLDINGS"
                ].isna().sum()
                == 0
            ),
        },
    ]

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

    print(
        "=" * 76
    )

    print(
        "16 - PORTFOLIO ARCHITECTURE ANALYSIS"
    )

    print(
        "=" * 76
    )

    print(
        "\n1/8 - Architecture paneli okunuyor..."
    )

    df = read_data(
        ARCHITECTURE_FILE
    )

    require_columns(
        df,
        [
            "ETF_ID",
            "ETF_NAME",
            "FINANCIAL_MATCH_WEIGHT",
        ]
        + ARCHITECTURE_VARIABLES
        + list(
            COVERAGE_MAP.values()
        ),
    )

    print(
        f"ETF sayısı: "
        f"{len(df):,}"
    )

    print(
        "\n2/8 - Descriptive statistics hazırlanıyor..."
    )

    descriptive = build_descriptive_statistics(
        df
    )

    print(
        "\n3/8 - Coverage-adjusted descriptives hazırlanıyor..."
    )

    coverage_adjusted = build_coverage_adjusted_descriptives(
        df
    )

    print(
        "\n4/8 - Correlation matrix hazırlanıyor..."
    )

    correlation = build_correlation_matrix(
        df
    )

    print(
        "\n5/8 - Concentration rankings hazırlanıyor..."
    )

    rankings = build_concentration_rankings(
        df
    )

    print(
        "\n6/8 - Architecture extremes hazırlanıyor..."
    )

    extremes = build_architecture_extremes(
        df
    )

    print(
        "\n7/8 - Standardized panel hazırlanıyor..."
    )

    standardized = build_standardized_panel(
        df
    )

    validation = build_validation(
        df=df,
        descriptive=descriptive,
        correlation=correlation,
    )

    print(
        "\n8/8 - Çıktılar kaydediliyor..."
    )

    descriptive.to_csv(
        DESCRIPTIVE_FILE,
        index=False,
    )

    correlation.to_csv(
        CORRELATION_FILE,
        index=False,
    )

    coverage_adjusted.to_csv(
        COVERAGE_ADJUSTED_FILE,
        index=False,
    )

    rankings.to_csv(
        CONCENTRATION_RANKING_FILE,
        index=False,
    )

    extremes.to_csv(
        ARCHITECTURE_EXTREMES_FILE,
        index=False,
    )

    standardized.to_csv(
        STANDARDIZED_PANEL_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    print(
        "\nPORTFOLIO ARCHITECTURE ANALYSIS HAZIR"
    )

    print(
        "=" * 76
    )

    print(
        "\nDescriptive statistics:"
    )

    print(
        descriptive[
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
        "\nEn yoğun 10 ETF:"
    )

    print(
        rankings[
            [
                "ETF_NAME",
                "HHI",
                "TOP10_WEIGHT",
                "EFFECTIVE_NUMBER_OF_HOLDINGS",
            ]
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
        "\nOluşturulan dosyalar:"
    )

    print(
        DESCRIPTIVE_FILE
    )

    print(
        CORRELATION_FILE
    )

    print(
        COVERAGE_ADJUSTED_FILE
    )

    print(
        CONCENTRATION_RANKING_FILE
    )

    print(
        ARCHITECTURE_EXTREMES_FILE
    )

    print(
        STANDARDIZED_PANEL_FILE
    )

    print(
        VALIDATION_FILE
    )


if __name__ == "__main__":
    main()