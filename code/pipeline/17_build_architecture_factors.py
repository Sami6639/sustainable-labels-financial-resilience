from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 17_build_architecture_factors.py
#
# Amaç
# ----
# Holdings-based portfolio architecture değişkenlerinden
# teori-temelli bileşik faktörler üretmek.
#
# Faktörler
# ---------
# 1. Financial Resilience
# 2. Financing Vulnerability
# 3. Growth and Duration Exposure
# 4. Portfolio Concentration
# 5. Core Transition Sensitivity
# 6. Extended Transition Sensitivity
#
# Tasarım ilkesi
# --------------
# Faktörler eşit ağırlıklı standardize edilmiş bileşenlerden
# oluşturulur. Standardizasyon parametreleri, financial match
# weight >= 80% olan güvenilir referans örneklemden hesaplanır.
#
# Not
# ---
# Bu faktörler teori-temelli composite indices'tir.
# PCA/factor analysis daha sonra robustness amacıyla yapılacaktır.
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

FACTOR_PANEL_FILE = (
    OUTPUT_DIR
    / "17_portfolio_architecture_factors.csv"
)

FACTOR_PANEL_PARQUET_FILE = (
    OUTPUT_DIR
    / "17_portfolio_architecture_factors.parquet"
)

FACTOR_DESCRIPTIVE_FILE = (
    OUTPUT_DIR
    / "17_architecture_factor_descriptive_statistics.csv"
)

FACTOR_CORRELATION_FILE = (
    OUTPUT_DIR
    / "17_architecture_factor_correlation_matrix.csv"
)

FACTOR_COMPONENTS_FILE = (
    OUTPUT_DIR
    / "17_architecture_factor_components.csv"
)

STANDARDIZATION_PARAMETERS_FILE = (
    OUTPUT_DIR
    / "17_architecture_standardization_parameters.csv"
)

FACTOR_RANKINGS_FILE = (
    OUTPUT_DIR
    / "17_architecture_factor_rankings.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "17_architecture_factor_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "17_architecture_factor_variable_dictionary.csv"
)


# ============================================================
# 4. METODOLOJİK AYARLAR
# ============================================================

REFERENCE_MATCH_THRESHOLD = 0.80

LOWER_WINSOR_QUANTILE = 0.01

UPPER_WINSOR_QUANTILE = 0.99

MIN_COMPONENT_SHARE = 0.50


# ============================================================
# 5. FAKTÖR TANIMLARI
# ============================================================

# direction:
# +1 = değişken arttıkça faktör artar
# -1 = değişken arttıkça faktör azalır

FACTOR_DEFINITIONS = {
    "FINANCIAL_RESILIENCE": {
        "components": {
            "PW_ROA": 1,
            "PW_CASH_RATIO": 1,
            "PW_LOG_ASSETS": 1,
        },
        "minimum_components": 2,
        "theory": (
            "Profitability, liquidity buffers and firm size "
            "increase the capacity of portfolio firms to absorb "
            "climate-policy and financing shocks."
        ),
        "research_questions": "RQ2, RQ3, RQ4",
        "expected_cpu_interaction": (
            "Positive or less negative"
        ),
    },

    "FINANCING_VULNERABILITY": {
        "components": {
            "PW_LEVERAGE": 1,
            "PW_EXTERNAL_FINANCE_DEPENDENCE": 1,
            "PW_CASH_RATIO": -1,
            "PW_ROA": -1,
        },
        "minimum_components": 2,
        "theory": (
            "High leverage, dependence on external capital and "
            "weak internal financial capacity increase exposure "
            "to uncertainty-driven financing constraints."
        ),
        "research_questions": "RQ2, RQ3",
        "expected_cpu_interaction": "Negative",
    },

    "GROWTH_DURATION_EXPOSURE": {
        "components": {
            "PW_CAPEX_INTENSITY": 1,
            "PW_RD_INTENSITY": 1,
            "PW_REVENUE_GROWTH": 1,
        },
        "minimum_components": 2,
        "theory": (
            "Investment intensity, R&D and growth increase the "
            "duration of expected cash flows and sensitivity to "
            "changes in discount rates and climate-policy regimes."
        ),
        "research_questions": "RQ2, RQ4",
        "expected_cpu_interaction": (
            "Potentially negative and state dependent"
        ),
    },

    "PORTFOLIO_CONCENTRATION": {
        "components": {
            "HHI": 1,
            "TOP10_WEIGHT": 1,
            "EFFECTIVE_NUMBER_OF_HOLDINGS": -1,
        },
        "minimum_components": 3,
        "theory": (
            "Greater concentration reduces diversification and "
            "increases exposure to dominant firms and sectors."
        ),
        "research_questions": "RQ1, RQ3",
        "expected_cpu_interaction": "Negative",
    },
}


CORE_FACTOR_COMPONENTS = {
    "FINANCIAL_RESILIENCE": -1,
    "GROWTH_DURATION_EXPOSURE": 1,
    "PORTFOLIO_CONCENTRATION": 1,
}


EXTENDED_FACTOR_COMPONENTS = {
    "FINANCIAL_RESILIENCE": -1,
    "FINANCING_VULNERABILITY": 1,
    "GROWTH_DURATION_EXPOSURE": 1,
    "PORTFOLIO_CONCENTRATION": 1,
}


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
    Gerekli sütunları kontrol eder.
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


def winsorize_with_limits(
    series: pd.Series,
    lower_limit: float,
    upper_limit: float,
) -> pd.Series:
    """
    Önceden belirlenmiş sınırlar kullanarak winsorization uygular.
    """

    numeric = safe_numeric(
        series
    )

    return numeric.clip(
        lower=lower_limit,
        upper=upper_limit,
    )


def standardize_with_parameters(
    series: pd.Series,
    mean_value: float,
    std_value: float,
) -> pd.Series:
    """
    Referans örneklemden gelen ortalama ve standart sapmayla
    standardizasyon yapar.
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
    dataframe: pd.DataFrame,
    columns: list[str],
    minimum_components: int,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Mevcut bileşenlerin satır bazlı eşit ağırlıklı ortalamasını
    hesaplar.

    Faktör yalnızca minimum bileşen sayısı sağlanıyorsa üretilir.
    """

    available_count = (
        dataframe[
            columns
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    component_share = (
        available_count
        / len(
            columns
        )
    )

    score = (
        dataframe[
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
        component_share,
    )


# ============================================================
# 7. STANDARDİZASYON PARAMETRELERİ
# ============================================================

def build_standardization_parameters(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    Financial match >= %80 referans örneklem üzerinden her değişken
    için winsorization ve z-score parametrelerini hesaplar.
    """

    reference_sample = df.loc[
        safe_numeric(
            df[
                "FINANCIAL_MATCH_WEIGHT"
            ]
        )
        >= REFERENCE_MATCH_THRESHOLD
    ].copy()

    if reference_sample.empty:
        raise RuntimeError(
            "Financial match >= %80 referans örneklem bulunamadı."
        )

    rows = []

    for variable in variables:

        values = safe_numeric(
            reference_sample[
                variable
            ]
        ).dropna()

        if len(
            values
        ) < 5:
            raise RuntimeError(
                f"{variable} için referans örneklemde "
                "yeterli gözlem bulunamadı."
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

        winsorized_values = values.clip(
            lower=lower_limit,
            upper=upper_limit,
        )

        mean_value = float(
            winsorized_values.mean()
        )

        std_value = float(
            winsorized_values.std(
                ddof=1
            )
        )

        rows.append(
            {
                "VARIABLE": variable,
                "REFERENCE_THRESHOLD": (
                    REFERENCE_MATCH_THRESHOLD
                ),
                "REFERENCE_N": int(
                    len(
                        values
                    )
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


# ============================================================
# 8. STANDARDİZE BİLEŞENLER
# ============================================================

def create_standardized_components(
    df: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ham portfolio architecture değişkenlerinden winsorize edilmiş
    z-score bileşenleri üretir.
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

    for variable, settings in parameter_lookup.items():

        winsorized_name = (
            f"W_{variable}"
        )

        standardized_name = (
            f"Z_{variable}"
        )

        result[
            winsorized_name
        ] = winsorize_with_limits(
            series=result[
                variable
            ],
            lower_limit=settings[
                "LOWER_WINSOR_LIMIT"
            ],
            upper_limit=settings[
                "UPPER_WINSOR_LIMIT"
            ],
        )

        result[
            standardized_name
        ] = standardize_with_parameters(
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
# 9. TEORİ-TEMELLİ FAKTÖRLER
# ============================================================

def build_theory_factors(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Standardize edilmiş bileşenlerden teori-temelli faktörleri üretir.
    """

    result = df.copy()

    component_rows = []

    for factor_name, definition in (
        FACTOR_DEFINITIONS.items()
    ):

        signed_component_columns = []

        for variable, direction in (
            definition[
                "components"
            ].items()
        ):

            source_z = (
                f"Z_{variable}"
            )

            signed_column = (
                f"COMP_{factor_name}_{variable}"
            )

            result[
                signed_column
            ] = (
                direction
                * result[
                    source_z
                ]
            )

            signed_component_columns.append(
                signed_column
            )

            component_rows.append(
                {
                    "FACTOR": factor_name,
                    "ARCHITECTURE_VARIABLE": variable,
                    "STANDARDIZED_SOURCE": source_z,
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
                    "RESEARCH_QUESTIONS": definition[
                        "research_questions"
                    ],
                    "EXPECTED_CPU_INTERACTION": definition[
                        "expected_cpu_interaction"
                    ],
                }
            )

        (
            factor_score,
            available_count,
            component_share,
        ) = row_mean_with_minimum(
            dataframe=result,
            columns=signed_component_columns,
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
        ] = component_share

        result[
            f"{factor_name}_VALID"
        ] = (
            factor_score.notna()
        ).astype(
            int
        )

    components = pd.DataFrame(
        component_rows
    )

    return (
        result,
        components,
    )


# ============================================================
# 10. TRANSITION SENSITIVITY COMPOSITES
# ============================================================

def build_transition_sensitivity_indices(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Core ve extended transition-sensitivity composite endekslerini
    üretir.
    """

    result = df.copy()

    # Core index
    core_columns = []

    for factor, direction in (
        CORE_FACTOR_COMPONENTS.items()
    ):

        component_name = (
            f"CORE_TRANSITION_COMPONENT_{factor}"
        )

        result[
            component_name
        ] = (
            direction
            * result[
                factor
            ]
        )

        core_columns.append(
            component_name
        )

    (
        core_score,
        core_available,
        core_share,
    ) = row_mean_with_minimum(
        dataframe=result,
        columns=core_columns,
        minimum_components=2,
    )

    result[
        "CORE_TRANSITION_SENSITIVITY"
    ] = core_score

    result[
        "CORE_TRANSITION_AVAILABLE_COMPONENTS"
    ] = core_available

    result[
        "CORE_TRANSITION_COMPONENT_SHARE"
    ] = core_share

    result[
        "CORE_TRANSITION_VALID"
    ] = (
        core_score.notna()
    ).astype(
        int
    )

    # Extended index
    extended_columns = []

    for factor, direction in (
        EXTENDED_FACTOR_COMPONENTS.items()
    ):

        component_name = (
            f"EXTENDED_TRANSITION_COMPONENT_{factor}"
        )

        result[
            component_name
        ] = (
            direction
            * result[
                factor
            ]
        )

        extended_columns.append(
            component_name
        )

    (
        extended_score,
        extended_available,
        extended_share,
    ) = row_mean_with_minimum(
        dataframe=result,
        columns=extended_columns,
        minimum_components=3,
    )

    result[
        "EXTENDED_TRANSITION_SENSITIVITY"
    ] = extended_score

    result[
        "EXTENDED_TRANSITION_AVAILABLE_COMPONENTS"
    ] = extended_available

    result[
        "EXTENDED_TRANSITION_COMPONENT_SHARE"
    ] = extended_share

    result[
        "EXTENDED_TRANSITION_VALID"
    ] = (
        extended_score.notna()
    ).astype(
        int
    )

    return result


# ============================================================
# 11. FACTOR DESCRIPTIVES
# ============================================================

def build_factor_descriptives(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Faktörlerin tanımlayıcı istatistiklerini üretir.
    """

    factor_variables = (
        list(
            FACTOR_DEFINITIONS.keys()
        )
        + [
            "CORE_TRANSITION_SENSITIVITY",
            "EXTENDED_TRANSITION_SENSITIVITY",
        ]
    )

    rows = []

    for variable in factor_variables:

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
# 12. CORRELATION MATRIX
# ============================================================

def build_factor_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Faktör korelasyon matrisini üretir.
    """

    variables = (
        list(
            FACTOR_DEFINITIONS.keys()
        )
        + [
            "CORE_TRANSITION_SENSITIVITY",
            "EXTENDED_TRANSITION_SENSITIVITY",
        ]
    )

    correlation = (
        df[
            variables
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
# 13. FACTOR RANKINGS
# ============================================================

def build_factor_rankings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ETF'leri ana architecture faktörlerine göre sıralar.
    """

    variables = [
        "FINANCIAL_RESILIENCE",
        "FINANCING_VULNERABILITY",
        "GROWTH_DURATION_EXPOSURE",
        "PORTFOLIO_CONCENTRATION",
        "CORE_TRANSITION_SENSITIVITY",
        "EXTENDED_TRANSITION_SENSITIVITY",
    ]

    base_columns = [
        "ETF_ID",
        "ETF_NAME",
        "FINANCIAL_MATCH_WEIGHT",
    ]

    ranking = df[
        base_columns
        + variables
    ].copy()

    for variable in variables:

        ranking[
            f"RANK_HIGH_{variable}"
        ] = (
            ranking[
                variable
            ]
            .rank(
                method="min",
                ascending=False,
            )
        )

        ranking[
            f"PERCENTILE_{variable}"
        ] = (
            ranking[
                variable
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

    return ranking.sort_values(
        by="CORE_TRANSITION_SENSITIVITY",
        ascending=False,
        na_position="last",
    )


# ============================================================
# 14. VARIABLE DICTIONARY
# ============================================================

def build_variable_dictionary() -> pd.DataFrame:
    """
    Faktörlerin teori ve formül sözlüğünü oluşturur.
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
                    "Theory-based equal-weight composite"
                ),
                "FORMULA": (
                    "Mean of: "
                    + " ".join(
                        formula_parts
                    )
                ),
                "MINIMUM_COMPONENTS": definition[
                    "minimum_components"
                ],
                "THEORY": definition[
                    "theory"
                ],
                "RESEARCH_QUESTIONS": definition[
                    "research_questions"
                ],
                "EXPECTED_CPU_INTERACTION": definition[
                    "expected_cpu_interaction"
                ],
            }
        )

    rows.extend(
        [
            {
                "VARIABLE": (
                    "CORE_TRANSITION_SENSITIVITY"
                ),
                "TYPE": (
                    "Theory-based equal-weight composite"
                ),
                "FORMULA": (
                    "- Financial Resilience "
                    "+ Growth Duration Exposure "
                    "+ Portfolio Concentration"
                ),
                "MINIMUM_COMPONENTS": 2,
                "THEORY": (
                    "Core exposure to climate-policy shocks through "
                    "weak shock-absorption capacity, long-duration "
                    "investment exposure and portfolio concentration."
                ),
                "RESEARCH_QUESTIONS": (
                    "RQ1, RQ2, RQ3, RQ4"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Negative"
                ),
            },
            {
                "VARIABLE": (
                    "EXTENDED_TRANSITION_SENSITIVITY"
                ),
                "TYPE": (
                    "Theory-based equal-weight composite"
                ),
                "FORMULA": (
                    "- Financial Resilience "
                    "+ Financing Vulnerability "
                    "+ Growth Duration Exposure "
                    "+ Portfolio Concentration"
                ),
                "MINIMUM_COMPONENTS": 3,
                "THEORY": (
                    "Extended transition sensitivity incorporating "
                    "financial constraints as an additional activation "
                    "channel."
                ),
                "RESEARCH_QUESTIONS": (
                    "RQ1, RQ2, RQ3, RQ4"
                ),
                "EXPECTED_CPU_INTERACTION": (
                    "Negative"
                ),
            },
        ]
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. VALIDATION
# ============================================================

def build_validation(
    df: pd.DataFrame,
    parameters: pd.DataFrame,
    descriptives: pd.DataFrame,
) -> pd.DataFrame:
    """
    Faktör üretim sürecini doğrular.
    """

    factor_variables = (
        list(
            FACTOR_DEFINITIONS.keys()
        )
        + [
            "CORE_TRANSITION_SENSITIVITY",
            "EXTENDED_TRANSITION_SENSITIVITY",
        ]
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
            "CHECK": (
                "STANDARDIZATION_PARAMETER_ROWS"
            ),
            "VALUE": len(
                parameters
            ),
            "PASS": int(
                len(
                    parameters
                ) > 0
            ),
        },
        {
            "CHECK": (
                "FACTOR_DESCRIPTIVE_ROWS"
            ),
            "VALUE": len(
                descriptives
            ),
            "PASS": int(
                len(
                    descriptives
                )
                == len(
                    factor_variables
                )
            ),
        },
        {
            "CHECK": (
                "CORE_TRANSITION_VALID_ETFS"
            ),
            "VALUE": int(
                df[
                    "CORE_TRANSITION_VALID"
                ].sum()
            ),
            "PASS": int(
                df[
                    "CORE_TRANSITION_VALID"
                ].sum()
                >= 30
            ),
        },
        {
            "CHECK": (
                "EXTENDED_TRANSITION_VALID_ETFS"
            ),
            "VALUE": int(
                df[
                    "EXTENDED_TRANSITION_VALID"
                ].sum()
            ),
            "PASS": int(
                df[
                    "EXTENDED_TRANSITION_VALID"
                ].sum()
                >= 20
            ),
        },
    ]

    for factor in factor_variables:

        rows.append(
            {
                "CHECK": (
                    f"{factor}_NON_MISSING"
                ),
                "VALUE": int(
                    df[
                        factor
                    ].notna().sum()
                ),
                "PASS": int(
                    df[
                        factor
                    ].notna().sum()
                    > 0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. ANA PIPELINE
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
        "17 - THEORY-BASED PORTFOLIO ARCHITECTURE FACTORS"
    )

    print(
        "=" * 78
    )

    print(
        "\n1/9 - Portfolio architecture paneli okunuyor..."
    )

    architecture = read_data(
        ARCHITECTURE_FILE
    )

    required_variables = {
        variable
        for definition
        in FACTOR_DEFINITIONS.values()
        for variable
        in definition[
            "components"
        ]
    }

    require_columns(
        architecture,
        [
            "ETF_ID",
            "ETF_NAME",
            "FINANCIAL_MATCH_WEIGHT",
        ]
        + sorted(
            required_variables
        ),
    )

    print(
        f"ETF sayısı: "
        f"{len(architecture):,}"
    )

    print(
        "\n2/9 - Güvenilir referans örneklem belirleniyor..."
    )

    reference_count = int(
        (
            safe_numeric(
                architecture[
                    "FINANCIAL_MATCH_WEIGHT"
                ]
            )
            >= REFERENCE_MATCH_THRESHOLD
        ).sum()
    )

    print(
        f"Financial match >= 80% referans ETF sayısı: "
        f"{reference_count:,}"
    )

    print(
        "\n3/9 - Standardizasyon parametreleri hesaplanıyor..."
    )

    parameters = build_standardization_parameters(
        df=architecture,
        variables=sorted(
            required_variables
        ),
    )

    print(
        "\n4/9 - Standardize architecture bileşenleri oluşturuluyor..."
    )

    factor_panel = create_standardized_components(
        df=architecture,
        parameters=parameters,
    )

    print(
        "\n5/9 - Teori-temelli architecture faktörleri oluşturuluyor..."
    )

    (
        factor_panel,
        factor_components,
    ) = build_theory_factors(
        factor_panel
    )

    print(
        "\n6/9 - Transition sensitivity endeksleri oluşturuluyor..."
    )

    factor_panel = build_transition_sensitivity_indices(
        factor_panel
    )

    print(
        "\n7/9 - Descriptive, correlation ve ranking tabloları hazırlanıyor..."
    )

    descriptives = build_factor_descriptives(
        factor_panel
    )

    correlation = build_factor_correlation(
        factor_panel
    )

    rankings = build_factor_rankings(
        factor_panel
    )

    variable_dictionary = build_variable_dictionary()

    validation = build_validation(
        df=factor_panel,
        parameters=parameters,
        descriptives=descriptives,
    )

    print(
        "\n8/9 - Faktör kalite kontrolleri yapılıyor..."
    )

    factor_names = (
        list(
            FACTOR_DEFINITIONS.keys()
        )
        + [
            "CORE_TRANSITION_SENSITIVITY",
            "EXTENDED_TRANSITION_SENSITIVITY",
        ]
    )

    for factor in factor_names:

        print(
            f"{factor:<35}: "
            f"{factor_panel[factor].notna().sum():>2} ETF"
        )

    print(
        "\n9/9 - Çıktılar kaydediliyor..."
    )

    factor_panel.to_csv(
        FACTOR_PANEL_FILE,
        index=False,
    )

    factor_panel.to_parquet(
        FACTOR_PANEL_PARQUET_FILE,
        index=False,
    )

    descriptives.to_csv(
        FACTOR_DESCRIPTIVE_FILE,
        index=False,
    )

    correlation.to_csv(
        FACTOR_CORRELATION_FILE,
        index=False,
    )

    factor_components.to_csv(
        FACTOR_COMPONENTS_FILE,
        index=False,
    )

    parameters.to_csv(
        STANDARDIZATION_PARAMETERS_FILE,
        index=False,
    )

    rankings.to_csv(
        FACTOR_RANKINGS_FILE,
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
        "\nARCHITECTURE FACTORS HAZIR"
    )

    print(
        "=" * 78
    )

    print(
        "\nFaktör descriptive statistics:"
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
        "\nCore transition sensitivity en yüksek 10 ETF:"
    )

    print(
        rankings[
            [
                "ETF_NAME",
                "CORE_TRANSITION_SENSITIVITY",
                "FINANCIAL_RESILIENCE",
                "GROWTH_DURATION_EXPOSURE",
                "PORTFOLIO_CONCENTRATION",
                "FINANCIAL_MATCH_WEIGHT",
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
        "\nAna çıktı dosyaları:"
    )

    print(
        FACTOR_PANEL_FILE
    )

    print(
        FACTOR_DESCRIPTIVE_FILE
    )

    print(
        FACTOR_CORRELATION_FILE
    )

    print(
        FACTOR_COMPONENTS_FILE
    )

    print(
        STANDARDIZATION_PARAMETERS_FILE
    )

    print(
        FACTOR_RANKINGS_FILE
    )

    print(
        VALIDATION_FILE
    )

    print(
        VARIABLE_DICTIONARY_FILE
    )


if __name__ == "__main__":
    main()