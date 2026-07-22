from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)
from statsmodels.stats.sandwich_covariance import (
    cov_cluster_2groups,
)


# ============================================================
# 23_estimate_baseline_models.py
#
# AMAÇ
# ----
# Climate Policy Uncertainty ile portfolio architecture
# etkileşiminin ETF getirileri üzerindeki koşullu fiyatlama
# etkisini test etmek.
#
# RQ1:
# Is climate policy uncertainty priced uniformly across
# sustainable portfolios, or does pricing depend on
# portfolio architecture?
#
# RQ2:
# Which embedded firm-level financial characteristics amplify
# or attenuate climate-policy uncertainty exposure?
#
# Temel model:
#
# R_i,t =
#   ETF FE
#   + beta_1 CPU_t
#   + beta_2 CPU_t x Architecture_i
#   + controls_t
#   + error_i,t
#
# Ana katsayı:
# CPU_Z_X_CORE_TRANSITION_SENSITIVITY_MAIN
#
# Önemli:
# Portfolio architecture 2025Q4 holdings kesitine dayalı,
# zaman içinde sabit bir exposure proxy'sidir.
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

MASTER_PANEL_FILE = (
    OUTPUT_DIR
    / "22_master_cpu_return_architecture_panel.csv"
)

MAIN_PANEL_FILE = (
    OUTPUT_DIR
    / "22_main_quality_panel.csv"
)

MODERATE_PANEL_FILE = (
    OUTPUT_DIR
    / "22_moderate_quality_panel.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

ALL_MODEL_RESULTS_FILE = (
    OUTPUT_DIR
    / "23_baseline_all_model_results.csv"
)

KEY_COEFFICIENTS_FILE = (
    OUTPUT_DIR
    / "23_baseline_key_coefficients.csv"
)

RQ1_SUMMARY_FILE = (
    OUTPUT_DIR
    / "23_rq1_core_transition_summary.csv"
)

RQ2_MECHANISM_FILE = (
    OUTPUT_DIR
    / "23_rq2_mechanism_models.csv"
)

MODEL_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "23_baseline_model_diagnostics.csv"
)

SAMPLE_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "23_baseline_sample_diagnostics.csv"
)

VIF_FILE = (
    OUTPUT_DIR
    / "23_baseline_vif_diagnostics.csv"
)

ARCHITECTURE_CORRELATION_FILE = (
    OUTPUT_DIR
    / "23_architecture_factor_correlations.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "23_baseline_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "23_baseline_model_dictionary.csv"
)


# ============================================================
# 4. MODEL AYARLARI
# ============================================================

DEPENDENT_VARIABLE = "ETF_RETURN"

TIME_CONTROLS = [
    "MARKET_RETURN",
    "ENERGY_RETURN",
    "TREASURY_RETURN",
    "VIX_CHANGE",
]

MAIN_ARCHITECTURE = (
    "CORE_TRANSITION_SENSITIVITY_MAIN"
)

MAIN_INTERACTION = (
    "CPU_Z_X_CORE_TRANSITION_SENSITIVITY_MAIN"
)

MODERATE_ARCHITECTURE = (
    "CORE_TRANSITION_SENSITIVITY_MODERATE"
)

MODERATE_INTERACTION = (
    "CPU_Z_X_CORE_TRANSITION_SENSITIVITY_MODERATE"
)


MECHANISM_MODELS_MAIN = {
    "FINANCIAL_RESILIENCE": {
        "architecture": (
            "FINANCIAL_RESILIENCE_MAIN"
        ),
        "interaction": (
            "CPU_Z_X_FINANCIAL_RESILIENCE_MAIN"
        ),
        "expected_sign": (
            "Positive or less negative"
        ),
    },

    "GROWTH_DURATION_EXPOSURE": {
        "architecture": (
            "GROWTH_DURATION_EXPOSURE_MAIN"
        ),
        "interaction": (
            "CPU_Z_X_GROWTH_DURATION_EXPOSURE_MAIN"
        ),
        "expected_sign": "Negative",
    },

    "PORTFOLIO_CONCENTRATION": {
        "architecture": (
            "PORTFOLIO_CONCENTRATION_MAIN"
        ),
        "interaction": (
            "CPU_Z_X_PORTFOLIO_CONCENTRATION_MAIN"
        ),
        "expected_sign": "Negative",
    },

    "FINANCING_VULNERABILITY": {
        "architecture": (
            "FINANCING_VULNERABILITY_MAIN"
        ),
        "interaction": (
            "CPU_Z_X_FINANCING_VULNERABILITY_MAIN"
        ),
        "expected_sign": "Negative",
    },

    "EXTENDED_TRANSITION_SENSITIVITY": {
        "architecture": (
            "EXTENDED_TRANSITION_SENSITIVITY_MAIN"
        ),
        "interaction": (
            "CPU_Z_X_EXTENDED_TRANSITION_SENSITIVITY_MAIN"
        ),
        "expected_sign": "Negative",
    },
}


# ============================================================
# 5. YARDIMCI FONKSİYONLAR
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
    dataset_name: str,
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
            f"{dataset_name} içinde eksik sütunlar:\n"
            + "\n".join(
                missing
            )
        )


def significance_stars(
    p_value,
) -> str:
    """
    Akademik tablo yıldızlarını üretir.
    """

    if pd.isna(p_value):
        return ""

    if p_value < 0.01:
        return "***"

    if p_value < 0.05:
        return "**"

    if p_value < 0.10:
        return "*"

    return ""


def prepare_panel(
    path: Path,
    sample_name: str,
) -> pd.DataFrame:
    """
    Ekonometrik paneli hazırlar.
    """

    panel = normalize_columns(
        read_data(
            path
        )
    )

    required_columns = [
        "DATE",
        "ETF_ID",
        "ETF_NAME",
        "ETF_RETURN",
        "CPU_Z",
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_CHANGE",
    ]

    require_columns(
        panel,
        required_columns,
        sample_name,
    )

    panel = panel.copy()

    panel["DATE"] = pd.to_datetime(
        panel["DATE"],
        errors="coerce",
    )

    panel["ETF_ID"] = (
        panel["ETF_ID"]
        .astype("string")
        .str.strip()
    )

    panel["DATE_GROUP"] = (
        panel["DATE"]
        .dt.to_period("M")
        .astype(str)
    )

    panel["CALENDAR_MONTH"] = (
        panel["DATE"]
        .dt.month
        .astype("Int64")
    )

    panel["YEAR"] = (
        panel["DATE"]
        .dt.year
        .astype("Int64")
    )

    numeric_candidates = [
        "ETF_RETURN",
        "CPU_Z",
        "CPU",
        "CPU_DIFF",
        "LOG_CPU_Z",
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_CHANGE",
        "VIX_LEVEL",
    ]

    numeric_candidates.extend(
        [
            MAIN_ARCHITECTURE,
            MAIN_INTERACTION,
            MODERATE_ARCHITECTURE,
            MODERATE_INTERACTION,
        ]
    )

    for settings in (
        MECHANISM_MODELS_MAIN.values()
    ):

        numeric_candidates.append(
            settings["architecture"]
        )

        numeric_candidates.append(
            settings["interaction"]
        )

    for column in set(
        numeric_candidates
    ):

        if column in panel.columns:

            panel[column] = safe_numeric(
                panel[column]
            )

    panel = panel.loc[
        panel["DATE"].notna()
        & panel["ETF_ID"].notna()
    ].copy()

    duplicate_rows = panel.duplicated(
        subset=[
            "ETF_ID",
            "DATE",
        ],
        keep=False,
    ).sum()

    if duplicate_rows > 0:

        raise RuntimeError(
            f"{sample_name} içinde duplicate ETF-month "
            f"satırı bulundu: {duplicate_rows}"
        )

    return panel


def build_formula(
    interaction_variable: str,
    include_etf_fe: bool,
    include_month_fe: bool,
    include_time_controls: bool,
    include_cpu_main: bool,
) -> str:
    """
    Model formülünü oluşturur.
    """

    regressors = []

    if include_cpu_main:

        regressors.append(
            "CPU_Z"
        )

    regressors.append(
        interaction_variable
    )

    if include_time_controls:

        regressors.extend(
            TIME_CONTROLS
        )

    if include_etf_fe:

        regressors.append(
            "C(ETF_ID)"
        )

    if include_month_fe:

        regressors.append(
            "C(DATE_GROUP)"
        )

    return (
        f"{DEPENDENT_VARIABLE} ~ "
        + " + ".join(
            regressors
        )
    )


def prepare_model_data(
    panel: pd.DataFrame,
    interaction_variable: str,
    include_time_controls: bool,
    include_cpu_main: bool,
) -> pd.DataFrame:
    """
    Model için complete-case veri setini oluşturur.
    """

    required = [
        DEPENDENT_VARIABLE,
        "ETF_ID",
        "DATE_GROUP",
        interaction_variable,
    ]

    if include_cpu_main:

        required.append(
            "CPU_Z"
        )

    if include_time_controls:

        required.extend(
            TIME_CONTROLS
        )

    require_columns(
        panel,
        required,
        "MODEL DATA",
    )

    model_data = panel[
        list(
            dict.fromkeys(
                required
            )
        )
    ].copy()

    for column in required:

        if column not in [
            "ETF_ID",
            "DATE_GROUP",
        ]:

            model_data[column] = safe_numeric(
                model_data[column]
            )

    model_data = model_data.dropna(
        subset=required
    ).copy()

    return model_data


# ============================================================
# 6. MODEL ESTIMATION
# ============================================================

def fit_model(
    panel: pd.DataFrame,
    model_name: str,
    sample_name: str,
    interaction_variable: str,
    covariance_type: str,
    include_etf_fe: bool,
    include_month_fe: bool,
    include_time_controls: bool,
    include_cpu_main: bool,
):
    """
    OLS ve fixed-effect modellerini tahmin eder.

    covariance_type:
    - HC3
    - CLUSTER_DATE
    - CLUSTER_ETF
    - TWO_WAY_CLUSTER
    """

    formula = build_formula(
        interaction_variable=interaction_variable,
        include_etf_fe=include_etf_fe,
        include_month_fe=include_month_fe,
        include_time_controls=include_time_controls,
        include_cpu_main=include_cpu_main,
    )

    model_data = prepare_model_data(
        panel=panel,
        interaction_variable=interaction_variable,
        include_time_controls=include_time_controls,
        include_cpu_main=include_cpu_main,
    )

    if model_data.empty:

        raise RuntimeError(
            f"{model_name} için geçerli gözlem bulunamadı."
        )

    base_result = smf.ols(
        formula=formula,
        data=model_data,
    ).fit()

    custom_covariance = None

    if covariance_type == "HC3":

        fitted = base_result.get_robustcov_results(
            cov_type="HC3"
        )

    elif covariance_type == "CLUSTER_DATE":

        fitted = base_result.get_robustcov_results(
            cov_type="cluster",
            groups=model_data[
                "DATE_GROUP"
            ],
            use_correction=True,
        )

    elif covariance_type == "CLUSTER_ETF":

        fitted = base_result.get_robustcov_results(
            cov_type="cluster",
            groups=model_data[
                "ETF_ID"
            ],
            use_correction=True,
        )

    elif covariance_type == "TWO_WAY_CLUSTER":

        group_etf = pd.factorize(
            model_data[
                "ETF_ID"
            ]
        )[0]

        group_date = pd.factorize(
            model_data[
                "DATE_GROUP"
            ]
        )[0]

        covariance_tuple = cov_cluster_2groups(
            base_result,
            group_etf,
            group_date,
        )

        custom_covariance = (
            covariance_tuple[0]
        )

        fitted = base_result

    else:

        raise ValueError(
            f"Bilinmeyen covariance türü: "
            f"{covariance_type}"
        )

    return {
        "model_name": model_name,
        "sample_name": sample_name,
        "formula": formula,
        "covariance_type": covariance_type,
        "model_data": model_data,
        "base_result": base_result,
        "fitted_result": fitted,
        "custom_covariance": custom_covariance,
        "interaction_variable": interaction_variable,
        "include_etf_fe": include_etf_fe,
        "include_month_fe": include_month_fe,
        "include_time_controls": include_time_controls,
        "include_cpu_main": include_cpu_main,
    }


# ============================================================
# 7. SONUÇ ÇIKARMA
# ============================================================

def extract_model_results(
    fitted_model: dict,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Model katsayılarını ve diagnostics bilgilerini çıkarır.
    """

    base_result = fitted_model[
        "base_result"
    ]

    fitted_result = fitted_model[
        "fitted_result"
    ]

    custom_covariance = fitted_model[
        "custom_covariance"
    ]

    parameter_names = list(
        base_result.params.index
    )

    coefficients = np.asarray(
        base_result.params
    )

    if custom_covariance is not None:

        standard_errors = np.sqrt(
            np.clip(
                np.diag(
                    custom_covariance
                ),
                a_min=0,
                a_max=None,
            )
        )

        test_statistics = np.divide(
            coefficients,
            standard_errors,
            out=np.full_like(
                coefficients,
                np.nan,
                dtype=float,
            ),
            where=(
                standard_errors > 0
            ),
        )

        p_values = (
            2.0
            * stats.norm.sf(
                np.abs(
                    test_statistics
                )
            )
        )

        confidence_lower = (
            coefficients
            - 1.96
            * standard_errors
        )

        confidence_upper = (
            coefficients
            + 1.96
            * standard_errors
        )

    else:

        standard_errors = np.asarray(
            fitted_result.bse
        )

        test_statistics = np.asarray(
            fitted_result.tvalues
        )

        p_values = np.asarray(
            fitted_result.pvalues
        )

        confidence_intervals = np.asarray(
            fitted_result.conf_int()
        )

        confidence_lower = (
            confidence_intervals[:, 0]
        )

        confidence_upper = (
            confidence_intervals[:, 1]
        )

    rows = []

    for index, parameter in enumerate(
        parameter_names
    ):

        p_value = float(
            p_values[index]
        )

        rows.append(
            {
                "MODEL": fitted_model[
                    "model_name"
                ],

                "SAMPLE": fitted_model[
                    "sample_name"
                ],

                "COVARIANCE": fitted_model[
                    "covariance_type"
                ],

                "PARAMETER": parameter,

                "COEFFICIENT": float(
                    coefficients[index]
                ),

                "STD_ERROR": float(
                    standard_errors[index]
                ),

                "TEST_STATISTIC": float(
                    test_statistics[index]
                ),

                "P_VALUE": p_value,

                "SIGNIFICANCE": (
                    significance_stars(
                        p_value
                    )
                ),

                "CI_LOWER_95": float(
                    confidence_lower[index]
                ),

                "CI_UPPER_95": float(
                    confidence_upper[index]
                ),

                "N_OBSERVATIONS": int(
                    base_result.nobs
                ),

                "N_ETFS": int(
                    fitted_model[
                        "model_data"
                    ][
                        "ETF_ID"
                    ].nunique()
                ),

                "N_MONTHS": int(
                    fitted_model[
                        "model_data"
                    ][
                        "DATE_GROUP"
                    ].nunique()
                ),

                "R_SQUARED": float(
                    base_result.rsquared
                ),

                "ADJ_R_SQUARED": float(
                    base_result.rsquared_adj
                ),
            }
        )

    diagnostic = {
        "MODEL": fitted_model[
            "model_name"
        ],

        "SAMPLE": fitted_model[
            "sample_name"
        ],

        "COVARIANCE": fitted_model[
            "covariance_type"
        ],

        "FORMULA": fitted_model[
            "formula"
        ],

        "N_OBSERVATIONS": int(
            base_result.nobs
        ),

        "N_ETFS": int(
            fitted_model[
                "model_data"
            ][
                "ETF_ID"
            ].nunique()
        ),

        "N_MONTHS": int(
            fitted_model[
                "model_data"
            ][
                "DATE_GROUP"
            ].nunique()
        ),

        "R_SQUARED": float(
            base_result.rsquared
        ),

        "ADJ_R_SQUARED": float(
            base_result.rsquared_adj
        ),

        "AIC": float(
            base_result.aic
        ),

        "BIC": float(
            base_result.bic
        ),

        "CONDITION_NUMBER": float(
            base_result.condition_number
        ),

        "ETF_FIXED_EFFECTS": int(
            fitted_model[
                "include_etf_fe"
            ]
        ),

        "MONTH_FIXED_EFFECTS": int(
            fitted_model[
                "include_month_fe"
            ]
        ),

        "TIME_CONTROLS": int(
            fitted_model[
                "include_time_controls"
            ]
        ),

        "CPU_MAIN_EFFECT": int(
            fitted_model[
                "include_cpu_main"
            ]
        ),

        "TARGET_INTERACTION": fitted_model[
            "interaction_variable"
        ],
    }

    return (
        pd.DataFrame(
            rows
        ),
        diagnostic,
    )


# ============================================================
# 8. RQ1 MODEL SETİ
# ============================================================

def estimate_rq1_models(
    main_panel: pd.DataFrame,
    moderate_panel: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Core Transition Sensitivity için ana modelleri tahmin eder.
    """

    model_specs = [
        {
            "panel": main_panel,
            "model_name": (
                "M1_MAIN_POOLED_HC3"
            ),
            "sample_name": "MAIN",
            "interaction": MAIN_INTERACTION,
            "covariance": "HC3",
            "etf_fe": False,
            "month_fe": False,
            "controls": True,
            "cpu_main": True,
        },

        {
            "panel": main_panel,
            "model_name": (
                "M2_MAIN_ETF_FE_CLUSTER_DATE"
            ),
            "sample_name": "MAIN",
            "interaction": MAIN_INTERACTION,
            "covariance": "CLUSTER_DATE",
            "etf_fe": True,
            "month_fe": False,
            "controls": True,
            "cpu_main": True,
        },

        {
            "panel": main_panel,
            "model_name": (
                "M3_MAIN_ETF_FE_CLUSTER_ETF"
            ),
            "sample_name": "MAIN",
            "interaction": MAIN_INTERACTION,
            "covariance": "CLUSTER_ETF",
            "etf_fe": True,
            "month_fe": False,
            "controls": True,
            "cpu_main": True,
        },

        {
            "panel": main_panel,
            "model_name": (
                "M4_MAIN_ETF_FE_TWO_WAY_CLUSTER"
            ),
            "sample_name": "MAIN",
            "interaction": MAIN_INTERACTION,
            "covariance": "TWO_WAY_CLUSTER",
            "etf_fe": True,
            "month_fe": False,
            "controls": True,
            "cpu_main": True,
        },

        {
            "panel": main_panel,
            "model_name": (
                "M5_MAIN_MONTH_FE_CLUSTER_ETF"
            ),
            "sample_name": "MAIN",
            "interaction": MAIN_INTERACTION,
            "covariance": "CLUSTER_ETF",
            "etf_fe": True,
            "month_fe": True,
            "controls": False,
            "cpu_main": False,
        },

        {
            "panel": moderate_panel,
            "model_name": (
                "M6_MODERATE_ETF_FE_CLUSTER_DATE"
            ),
            "sample_name": "MODERATE",
            "interaction": MODERATE_INTERACTION,
            "covariance": "CLUSTER_DATE",
            "etf_fe": True,
            "month_fe": False,
            "controls": True,
            "cpu_main": True,
        },

        {
            "panel": moderate_panel,
            "model_name": (
                "M7_MODERATE_ETF_FE_TWO_WAY_CLUSTER"
            ),
            "sample_name": "MODERATE",
            "interaction": MODERATE_INTERACTION,
            "covariance": "TWO_WAY_CLUSTER",
            "etf_fe": True,
            "month_fe": False,
            "controls": True,
            "cpu_main": True,
        },
    ]

    result_frames = []

    diagnostic_rows = []

    for specification in model_specs:

        print(
            f"  Tahmin ediliyor: "
            f"{specification['model_name']}"
        )

        fitted_model = fit_model(
            panel=specification[
                "panel"
            ],
            model_name=specification[
                "model_name"
            ],
            sample_name=specification[
                "sample_name"
            ],
            interaction_variable=specification[
                "interaction"
            ],
            covariance_type=specification[
                "covariance"
            ],
            include_etf_fe=specification[
                "etf_fe"
            ],
            include_month_fe=specification[
                "month_fe"
            ],
            include_time_controls=specification[
                "controls"
            ],
            include_cpu_main=specification[
                "cpu_main"
            ],
        )

        (
            coefficient_table,
            diagnostic,
        ) = extract_model_results(
            fitted_model
        )

        result_frames.append(
            coefficient_table
        )

        diagnostic_rows.append(
            diagnostic
        )

    return (
        pd.concat(
            result_frames,
            ignore_index=True,
        ),
        pd.DataFrame(
            diagnostic_rows
        ),
    )


# ============================================================
# 9. RQ2 MEKANİZMA MODELLERİ
# ============================================================

def estimate_mechanism_models(
    main_panel: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Her mekanizma faktörünü ayrı modelde test eder.

    Ayrı modeller, faktörler arası multicollinearity riskini azaltır
    ve teorik kanalların yorumlanmasını kolaylaştırır.
    """

    result_frames = []

    diagnostic_rows = []

    for (
        mechanism_name,
        settings,
    ) in MECHANISM_MODELS_MAIN.items():

        interaction = settings[
            "interaction"
        ]

        if interaction not in main_panel.columns:

            print(
                f"  Atlandı: {interaction} bulunamadı."
            )

            continue

        model_name = (
            f"RQ2_{mechanism_name}_"
            "ETF_FE_CLUSTER_DATE"
        )

        print(
            f"  Tahmin ediliyor: "
            f"{model_name}"
        )

        fitted_model = fit_model(
            panel=main_panel,
            model_name=model_name,
            sample_name="MAIN",
            interaction_variable=interaction,
            covariance_type=(
                "CLUSTER_DATE"
            ),
            include_etf_fe=True,
            include_month_fe=False,
            include_time_controls=True,
            include_cpu_main=True,
        )

        (
            coefficient_table,
            diagnostic,
        ) = extract_model_results(
            fitted_model
        )

        coefficient_table[
            "MECHANISM"
        ] = mechanism_name

        coefficient_table[
            "EXPECTED_SIGN"
        ] = settings[
            "expected_sign"
        ]

        result_frames.append(
            coefficient_table
        )

        diagnostic[
            "MECHANISM"
        ] = mechanism_name

        diagnostic_rows.append(
            diagnostic
        )

    if result_frames:

        result_table = pd.concat(
            result_frames,
            ignore_index=True,
        )

    else:

        result_table = pd.DataFrame()

    return (
        result_table,
        pd.DataFrame(
            diagnostic_rows
        ),
    )


# ============================================================
# 10. ANA KATSAYI ÖZETİ
# ============================================================

def build_key_coefficient_table(
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    CPU ana etkisi ve hedef interaction katsayılarını seçer.
    """

    target_parameters = [
        "CPU_Z",
        MAIN_INTERACTION,
        MODERATE_INTERACTION,
    ]

    target_parameters.extend(
        [
            settings[
                "interaction"
            ]
            for settings
            in MECHANISM_MODELS_MAIN.values()
        ]
    )

    result = all_results.loc[
        all_results[
            "PARAMETER"
        ].isin(
            target_parameters
        )
    ].copy()

    result[
        "COEFFICIENT_WITH_STARS"
    ] = (
        result[
            "COEFFICIENT"
        ]
        .map(
            lambda value: (
                f"{value:.6f}"
            )
        )
        + result[
            "SIGNIFICANCE"
        ]
    )

    return result.sort_values(
        by=[
            "SAMPLE",
            "MODEL",
            "PARAMETER",
        ]
    )


def build_rq1_summary(
    rq1_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    RQ1 ana interaction katsayılarını tek tabloda özetler.
    """

    interaction_parameters = [
        MAIN_INTERACTION,
        MODERATE_INTERACTION,
    ]

    summary = rq1_results.loc[
        rq1_results[
            "PARAMETER"
        ].isin(
            interaction_parameters
        )
    ].copy()

    summary[
        "ECONOMIC_INTERPRETATION"
    ] = np.where(
        summary[
            "COEFFICIENT"
        ] < 0,
        (
            "Higher transition sensitivity is associated with "
            "a more negative return response to CPU."
        ),
        (
            "Higher transition sensitivity is associated with "
            "a less negative or more positive return response to CPU."
        ),
    )

    return summary


# ============================================================
# 11. VIF VE KORELASYON
# ============================================================

def build_vif_diagnostics(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zaman serisi regresörleri için VIF hesaplar.
    """

    variables = [
        "CPU_Z",
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_CHANGE",
        MAIN_INTERACTION,
    ]

    available = [
        variable
        for variable in variables
        if variable in panel.columns
    ]

    data = (
        panel[
            available
        ]
        .apply(
            safe_numeric
        )
        .dropna()
        .copy()
    )

    if data.empty:

        return pd.DataFrame(
            columns=[
                "VARIABLE",
                "VIF",
                "N",
            ]
        )

    design = data.copy()

    design.insert(
        0,
        "INTERCEPT",
        1.0,
    )

    rows = []

    for index, column in enumerate(
        design.columns
    ):

        if column == "INTERCEPT":
            continue

        try:

            vif_value = (
                variance_inflation_factor(
                    design.values,
                    index,
                )
            )

        except Exception:

            vif_value = np.nan

        rows.append(
            {
                "VARIABLE": column,

                "VIF": float(
                    vif_value
                ),

                "N": len(
                    design
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_architecture_correlations(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main-quality architecture faktörlerinin ETF düzeyi
    korelasyon matrisini üretir.
    """

    variables = [
        "CORE_TRANSITION_SENSITIVITY_MAIN",
        "EXTENDED_TRANSITION_SENSITIVITY_MAIN",
        "FINANCIAL_RESILIENCE_MAIN",
        "FINANCING_VULNERABILITY_MAIN",
        "GROWTH_DURATION_EXPOSURE_MAIN",
        "PORTFOLIO_CONCENTRATION_MAIN",
    ]

    available = [
        variable
        for variable in variables
        if variable in panel.columns
    ]

    cross_section = (
        panel[
            [
                "ETF_ID",
            ]
            + available
        ]
        .drop_duplicates(
            subset=[
                "ETF_ID",
            ]
        )
    )

    correlation = (
        cross_section[
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
# 12. ÖRNEKLEM DIAGNOSTICS
# ============================================================

def build_sample_diagnostics(
    main_panel: pd.DataFrame,
    moderate_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ekonometrik örneklemleri özetler.
    """

    rows = []

    for (
        sample_name,
        panel,
        interaction,
    ) in [
        (
            "MAIN",
            main_panel,
            MAIN_INTERACTION,
        ),
        (
            "MODERATE",
            moderate_panel,
            MODERATE_INTERACTION,
        ),
    ]:

        valid_columns = [
            "ETF_RETURN",
            "CPU_Z",
            interaction,
        ] + TIME_CONTROLS

        available_columns = [
            column
            for column in valid_columns
            if column in panel.columns
        ]

        complete_case = panel[
            available_columns
        ].notna().all(
            axis=1
        )

        rows.append(
            {
                "SAMPLE": sample_name,

                "RAW_ROWS": len(
                    panel
                ),

                "RAW_ETFS": panel[
                    "ETF_ID"
                ].nunique(),

                "RAW_MONTHS": panel[
                    "DATE_GROUP"
                ].nunique(),

                "COMPLETE_CASE_ROWS": int(
                    complete_case.sum()
                ),

                "COMPLETE_CASE_RATE": float(
                    complete_case.mean()
                ),

                "START_DATE": (
                    panel[
                        "DATE"
                    ].min()
                ),

                "END_DATE": (
                    panel[
                        "DATE"
                    ].max()
                ),

                "MEAN_RETURN": float(
                    panel[
                        "ETF_RETURN"
                    ].mean()
                ),

                "STD_RETURN": float(
                    panel[
                        "ETF_RETURN"
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
# 13. MODEL SÖZLÜĞÜ
# ============================================================

def build_model_dictionary() -> pd.DataFrame:
    """
    Model tasarım sözlüğünü üretir.
    """

    return pd.DataFrame(
        [
            {
                "MODEL_FAMILY": "RQ1 Core",

                "DEPENDENT_VARIABLE": (
                    "ETF_RETURN"
                ),

                "KEY_REGRESSOR": (
                    MAIN_INTERACTION
                ),

                "FIXED_EFFECTS": (
                    "ETF fixed effects"
                ),

                "STANDARD_ERRORS": (
                    "Date-clustered, ETF-clustered and two-way clustered"
                ),

                "THEORY": (
                    "Conditional asset pricing"
                ),

                "INTERPRETATION": (
                    "Tests whether CPU pricing varies with embedded "
                    "portfolio transition sensitivity."
                ),
            },

            {
                "MODEL_FAMILY": (
                    "RQ1 Month FE"
                ),

                "DEPENDENT_VARIABLE": (
                    "ETF_RETURN"
                ),

                "KEY_REGRESSOR": (
                    MAIN_INTERACTION
                ),

                "FIXED_EFFECTS": (
                    "ETF and month fixed effects"
                ),

                "STANDARD_ERRORS": (
                    "ETF clustered"
                ),

                "THEORY": (
                    "Cross-sectional differential response"
                ),

                "INTERPRETATION": (
                    "Month effects absorb all common monthly shocks, "
                    "while the CPU–architecture interaction remains "
                    "identified through cross-sectional exposure."
                ),
            },

            {
                "MODEL_FAMILY": (
                    "RQ2 Mechanisms"
                ),

                "DEPENDENT_VARIABLE": (
                    "ETF_RETURN"
                ),

                "KEY_REGRESSOR": (
                    "CPU_Z × individual architecture mechanism"
                ),

                "FIXED_EFFECTS": (
                    "ETF fixed effects"
                ),

                "STANDARD_ERRORS": (
                    "Date clustered"
                ),

                "THEORY": (
                    "Real options, financing constraints, equity duration "
                    "and diversification"
                ),

                "INTERPRETATION": (
                    "Tests each architecture transmission channel "
                    "in a separate specification."
                ),
            },
        ]
    )


# ============================================================
# 14. VALIDATION
# ============================================================

def build_validation(
    rq1_results: pd.DataFrame,
    mechanism_results: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    sample_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ekonometrik çıktıların temel doğrulamasını yapar.
    """

    rq1_interaction_count = int(
        rq1_results[
            "PARAMETER"
        ].isin(
            [
                MAIN_INTERACTION,
                MODERATE_INTERACTION,
            ]
        ).sum()
    )

    main_two_way_available = int(
        (
            rq1_results[
                "MODEL"
            ]
            == (
                "M4_MAIN_ETF_FE_"
                "TWO_WAY_CLUSTER"
            )
        ).any()
    )

    month_fe_available = int(
        (
            rq1_results[
                "MODEL"
            ]
            == (
                "M5_MAIN_MONTH_FE_"
                "CLUSTER_ETF"
            )
        ).any()
    )

    finite_coefficients = int(
        np.isfinite(
            rq1_results[
                "COEFFICIENT"
            ]
        ).all()
    )

    rows = [
        {
            "CHECK": "RQ1_RESULT_ROWS",

            "VALUE": len(
                rq1_results
            ),

            "PASS": int(
                len(
                    rq1_results
                ) > 0
            ),
        },

        {
            "CHECK": (
                "RQ1_INTERACTION_ESTIMATES"
            ),

            "VALUE": rq1_interaction_count,

            "PASS": int(
                rq1_interaction_count
                >= 7
            ),
        },

        {
            "CHECK": (
                "MAIN_TWO_WAY_CLUSTER_AVAILABLE"
            ),

            "VALUE": (
                main_two_way_available
            ),

            "PASS": (
                main_two_way_available
            ),
        },

        {
            "CHECK": (
                "MONTH_FIXED_EFFECT_MODEL_AVAILABLE"
            ),

            "VALUE": month_fe_available,

            "PASS": month_fe_available,
        },

        {
            "CHECK": (
                "MECHANISM_RESULT_ROWS"
            ),

            "VALUE": len(
                mechanism_results
            ),

            "PASS": int(
                len(
                    mechanism_results
                ) > 0
            ),
        },

        {
            "CHECK": (
                "MODEL_DIAGNOSTIC_ROWS"
            ),

            "VALUE": len(
                model_diagnostics
            ),

            "PASS": int(
                len(
                    model_diagnostics
                ) >= 7
            ),
        },

        {
            "CHECK": (
                "SAMPLE_DIAGNOSTIC_ROWS"
            ),

            "VALUE": len(
                sample_diagnostics
            ),

            "PASS": int(
                len(
                    sample_diagnostics
                )
                == 2
            ),
        },

        {
            "CHECK": (
                "FINITE_RQ1_COEFFICIENTS"
            ),

            "VALUE": finite_coefficients,

            "PASS": finite_coefficients,
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 84)
    print("23 - BASELINE CONDITIONAL ASSET-PRICING MODELS")
    print("=" * 84)

    # --------------------------------------------------------
    # 1. Verileri oku
    # --------------------------------------------------------

    print(
        "\n1/8 - Main-quality panel okunuyor..."
    )

    main_panel = prepare_panel(
        path=MAIN_PANEL_FILE,
        sample_name="MAIN PANEL",
    )

    print(
        f"Main panel satırı: "
        f"{len(main_panel):,}"
    )

    print(
        f"Main panel ETF sayısı: "
        f"{main_panel['ETF_ID'].nunique():,}"
    )

    print(
        "\n2/8 - Moderate-quality panel okunuyor..."
    )

    moderate_panel = prepare_panel(
        path=MODERATE_PANEL_FILE,
        sample_name="MODERATE PANEL",
    )

    print(
        f"Moderate panel satırı: "
        f"{len(moderate_panel):,}"
    )

    print(
        f"Moderate ETF sayısı: "
        f"{moderate_panel['ETF_ID'].nunique():,}"
    )

    require_columns(
        main_panel,
        [
            MAIN_INTERACTION,
        ],
        "MAIN PANEL",
    )

    require_columns(
        moderate_panel,
        [
            MODERATE_INTERACTION,
        ],
        "MODERATE PANEL",
    )

    # --------------------------------------------------------
    # 2. RQ1
    # --------------------------------------------------------

    print(
        "\n3/8 - RQ1 core transition models tahmin ediliyor..."
    )

    (
        rq1_results,
        rq1_diagnostics,
    ) = estimate_rq1_models(
        main_panel=main_panel,
        moderate_panel=moderate_panel,
    )

    # --------------------------------------------------------
    # 3. RQ2
    # --------------------------------------------------------

    print(
        "\n4/8 - RQ2 mechanism models tahmin ediliyor..."
    )

    (
        mechanism_results,
        mechanism_diagnostics,
    ) = estimate_mechanism_models(
        main_panel=main_panel,
    )

    # --------------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------------

    print(
        "\n5/8 - VIF ve architecture korelasyonları hazırlanıyor..."
    )

    vif_diagnostics = build_vif_diagnostics(
        main_panel
    )

    architecture_correlations = (
        build_architecture_correlations(
            main_panel
        )
    )

    sample_diagnostics = (
        build_sample_diagnostics(
            main_panel=main_panel,
            moderate_panel=moderate_panel,
        )
    )

    # --------------------------------------------------------
    # 5. Sonuç tabloları
    # --------------------------------------------------------

    print(
        "\n6/8 - Ana katsayı tabloları hazırlanıyor..."
    )

    all_result_frames = [
        rq1_results
    ]

    if not mechanism_results.empty:

        all_result_frames.append(
            mechanism_results
        )

    all_results = pd.concat(
        all_result_frames,
        ignore_index=True,
        sort=False,
    )

    all_diagnostic_frames = [
        rq1_diagnostics
    ]

    if not mechanism_diagnostics.empty:

        all_diagnostic_frames.append(
            mechanism_diagnostics
        )

    model_diagnostics = pd.concat(
        all_diagnostic_frames,
        ignore_index=True,
        sort=False,
    )

    key_coefficients = (
        build_key_coefficient_table(
            all_results
        )
    )

    rq1_summary = build_rq1_summary(
        rq1_results
    )

    if not mechanism_results.empty:

        mechanism_key = (
            mechanism_results.loc[
                mechanism_results[
                    "PARAMETER"
                ].isin(
                    [
                        settings[
                            "interaction"
                        ]
                        for settings
                        in MECHANISM_MODELS_MAIN.values()
                    ]
                )
            ]
            .copy()
        )

    else:

        mechanism_key = pd.DataFrame()

    model_dictionary = (
        build_model_dictionary()
    )

    # --------------------------------------------------------
    # 6. Validation
    # --------------------------------------------------------

    print(
        "\n7/8 - Validation kontrolleri çalıştırılıyor..."
    )

    validation = build_validation(
        rq1_results=rq1_results,
        mechanism_results=mechanism_results,
        model_diagnostics=model_diagnostics,
        sample_diagnostics=sample_diagnostics,
    )

    # --------------------------------------------------------
    # 7. Kaydet
    # --------------------------------------------------------

    print(
        "\n8/8 - Ekonometrik çıktılar kaydediliyor..."
    )

    all_results.to_csv(
        ALL_MODEL_RESULTS_FILE,
        index=False,
    )

    key_coefficients.to_csv(
        KEY_COEFFICIENTS_FILE,
        index=False,
    )

    rq1_summary.to_csv(
        RQ1_SUMMARY_FILE,
        index=False,
    )

    mechanism_key.to_csv(
        RQ2_MECHANISM_FILE,
        index=False,
    )

    model_diagnostics.to_csv(
        MODEL_DIAGNOSTICS_FILE,
        index=False,
    )

    sample_diagnostics.to_csv(
        SAMPLE_DIAGNOSTICS_FILE,
        index=False,
    )

    vif_diagnostics.to_csv(
        VIF_FILE,
        index=False,
    )

    architecture_correlations.to_csv(
        ARCHITECTURE_CORRELATION_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    model_dictionary.to_csv(
        VARIABLE_DICTIONARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 8. Ekran çıktısı
    # --------------------------------------------------------

    print(
        "\nBASELINE MODELS HAZIR"
    )

    print("=" * 84)

    print(
        "\nRQ1 – Core Transition Sensitivity sonuçları:"
    )

    print(
        rq1_summary[
            [
                "MODEL",
                "SAMPLE",
                "COVARIANCE",
                "PARAMETER",
                "COEFFICIENT",
                "STD_ERROR",
                "P_VALUE",
                "SIGNIFICANCE",
                "N_OBSERVATIONS",
                "N_ETFS",
                "R_SQUARED",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nRQ2 – Mekanizma interaction sonuçları:"
    )

    if mechanism_key.empty:

        print(
            "Mekanizma sonucu üretilemedi."
        )

    else:

        print(
            mechanism_key[
                [
                    "MECHANISM",
                    "PARAMETER",
                    "COEFFICIENT",
                    "STD_ERROR",
                    "P_VALUE",
                    "SIGNIFICANCE",
                    "EXPECTED_SIGN",
                    "N_OBSERVATIONS",
                    "N_ETFS",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\nVIF diagnostics:"
    )

    print(
        vif_diagnostics.to_string(
            index=False
        )
    )

    print(
        "\nModel diagnostics:"
    )

    print(
        model_diagnostics[
            [
                "MODEL",
                "SAMPLE",
                "COVARIANCE",
                "N_OBSERVATIONS",
                "N_ETFS",
                "N_MONTHS",
                "R_SQUARED",
                "ADJ_R_SQUARED",
                "CONDITION_NUMBER",
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
        "\nMETODOLOJİK NOT:"
    )

    print(
        "Architecture seviyesi ETF fixed effects tarafından emilir. "
        "Bu nedenle ana tanımlayıcı katsayı CPU × Architecture "
        "interaction katsayısıdır."
    )

    print(
        "Ay fixed-effects modelinde CPU ve ortak piyasa kontrolleri "
        "ay etkileri tarafından emildiği için yalnızca cross-sectional "
        "CPU × Architecture interaction tanımlanır."
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        ALL_MODEL_RESULTS_FILE
    )

    print(
        KEY_COEFFICIENTS_FILE
    )

    print(
        RQ1_SUMMARY_FILE
    )

    print(
        RQ2_MECHANISM_FILE
    )

    print(
        MODEL_DIAGNOSTICS_FILE
    )

    print(
        SAMPLE_DIAGNOSTICS_FILE
    )

    print(
        VIF_FILE
    )

    print(
        ARCHITECTURE_CORRELATION_FILE
    )

    print(
        VALIDATION_FILE
    )


if __name__ == "__main__":
    main()