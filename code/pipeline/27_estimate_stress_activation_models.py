from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


# ============================================================
# 27_estimate_stress_activation_models.py
#
# AMAÇ
# ----
# RQ3'ü test etmek:
#
# Does market stress activate the pricing of embedded portfolio
# architecture?
#
# TEMEL TEORİK İDDİA
# ------------------
# Portfolio architecture normal dönemlerde latent olabilir.
# Climate Policy Uncertainty ile genel piyasa stresi aynı anda
# yükseldiğinde architecture exposure fiyatlanabilir.
#
# AYRIŞTIRILAN ETKİLER
# --------------------
# 1. Architecture × High CPU
# 2. Architecture × High VIX
# 3. Architecture × Joint CPU–VIX Stress
#
# SÜREKLİ ROBUSTNESS
# ------------------
# CPU_Z × VIX_LEVEL_Z × Architecture
#
# ANA KATSAYI
# ------------
# CPU_VIX_STRESS_X_<CHANNEL>_MAIN
#
# ÇIKARIM
# -------
# Ana çıkarım ETF fixed effects ve tarih/two-way clustered
# standart hatalara dayanır.
#
# Dört primary channel için main-quality örneklemde ayrıca
# tarih kümeli wild-cluster bootstrap uygulanır.
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

PANEL_FILE = (
    OUTPUT_DIR
    / "25_final_econometric_panel.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

ALL_RESULTS_FILE = (
    OUTPUT_DIR
    / "27_stress_activation_all_results.csv"
)

KEY_RESULTS_FILE = (
    OUTPUT_DIR
    / "27_stress_activation_key_coefficients.csv"
)

PRIMARY_RESULTS_FILE = (
    OUTPUT_DIR
    / "27_primary_channel_stress_results.csv"
)

ROBUSTNESS_RESULTS_FILE = (
    OUTPUT_DIR
    / "27_composite_pca_stress_results.csv"
)

CONTINUOUS_RESULTS_FILE = (
    OUTPUT_DIR
    / "27_continuous_triple_interaction_results.csv"
)

BOOTSTRAP_RESULTS_FILE = (
    OUTPUT_DIR
    / "27_joint_stress_wild_cluster_bootstrap.csv"
)

MODEL_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "27_stress_activation_model_diagnostics.csv"
)

SAMPLE_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "27_stress_activation_sample_diagnostics.csv"
)

VIF_FILE = (
    OUTPUT_DIR
    / "27_stress_activation_vif_diagnostics.csv"
)

SIGN_CONSISTENCY_FILE = (
    OUTPUT_DIR
    / "27_stress_activation_sign_consistency.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "27_stress_activation_validation.csv"
)

MODEL_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "27_stress_activation_model_dictionary.csv"
)


# ============================================================
# 4. AMPİRİK AYARLAR
# ============================================================

DEPENDENT_VARIABLE = "ETF_RETURN"

TIME_CONTROLS = [
    "MARKET_RETURN",
    "ENERGY_RETURN",
    "TREASURY_RETURN",
    "VIX_CHANGE",
]

BOOTSTRAP_REPLICATIONS = 399

BOOTSTRAP_SEED = 20260712

MIN_ETFS = 5

MIN_OBSERVATIONS = 100


# ============================================================
# 5. KANALLAR
# ============================================================

CHANNELS = {
    "INTERNAL_FINANCIAL_CAPACITY": {
        "role": "PRIMARY",
        "expected_joint_sign": "POSITIVE",
        "theory": (
            "Internal profitability and liquidity buffers should "
            "attenuate losses when climate-policy uncertainty and "
            "market stress coincide."
        ),
    },

    "EXTERNAL_FINANCING_DEPENDENCE": {
        "role": "PRIMARY",
        "expected_joint_sign": "NEGATIVE",
        "theory": (
            "Dependence on external capital should amplify losses "
            "when uncertainty and market stress tighten financing."
        ),
    },

    "GROWTH_DURATION_EXPOSURE_FINAL": {
        "role": "PRIMARY",
        "expected_joint_sign": "NEGATIVE",
        "theory": (
            "Long-duration investment and innovation exposure should "
            "become more vulnerable when policy and discount-rate "
            "uncertainty rise jointly."
        ),
    },

    "PORTFOLIO_CONCENTRATION_FINAL": {
        "role": "PRIMARY",
        "expected_joint_sign": "NEGATIVE",
        "theory": (
            "Portfolio concentration should amplify exposure to "
            "dominant holdings during joint climate-policy and "
            "market stress."
        ),
    },

    "FINANCIAL_ARCHITECTURE_RISK_FINAL": {
        "role": "COMPOSITE_ROBUSTNESS",
        "expected_joint_sign": "NEGATIVE",
        "theory": (
            "Higher embedded financial architecture risk should "
            "amplify joint-stress losses."
        ),
    },

    "EXTENDED_ARCHITECTURE_RISK_FINAL": {
        "role": "COMPOSITE_ROBUSTNESS",
        "expected_joint_sign": "NEGATIVE",
        "theory": (
            "Financial architecture combined with concentration "
            "should amplify joint-stress losses."
        ),
    },

    "PCA_COMPONENT_1": {
        "role": "PCA_ROBUSTNESS",
        "expected_joint_sign": "DATA_DRIVEN",
        "theory": (
            "The first principal component represents a common "
            "data-driven architecture dimension."
        ),
    },
}


# ============================================================
# 6. MODEL SPESİFİKASYONLARI
# ============================================================

REGIME_MODEL_SPECS = [
    {
        "suffix": "REGIME_ETF_FE_CLUSTER_DATE",
        "covariance": "CLUSTER_DATE",
        "etf_fe": True,
        "month_fe": False,
    },

    {
        "suffix": "REGIME_ETF_FE_CLUSTER_ETF",
        "covariance": "CLUSTER_ETF",
        "etf_fe": True,
        "month_fe": False,
    },

    {
        "suffix": "REGIME_ETF_FE_TWO_WAY_CLUSTER",
        "covariance": "TWO_WAY_CLUSTER",
        "etf_fe": True,
        "month_fe": False,
    },

    {
        "suffix": "REGIME_ETF_MONTH_FE_CLUSTER_ETF",
        "covariance": "CLUSTER_ETF",
        "etf_fe": True,
        "month_fe": True,
    },
]


CONTINUOUS_MODEL_SPECS = [
    {
        "suffix": "CONTINUOUS_ETF_FE_CLUSTER_DATE",
        "covariance": "CLUSTER_DATE",
        "etf_fe": True,
        "month_fe": False,
    },

    {
        "suffix": "CONTINUOUS_ETF_FE_CLUSTER_ETF",
        "covariance": "CLUSTER_ETF",
        "etf_fe": True,
        "month_fe": False,
    },

    {
        "suffix": "CONTINUOUS_ETF_FE_TWO_WAY_CLUSTER",
        "covariance": "TWO_WAY_CLUSTER",
        "etf_fe": True,
        "month_fe": False,
    },

    {
        "suffix": "CONTINUOUS_ETF_MONTH_FE_CLUSTER_ETF",
        "covariance": "CLUSTER_ETF",
        "etf_fe": True,
        "month_fe": True,
    },
]


# ============================================================
# 7. YARDIMCI FONKSİYONLAR
# ============================================================

def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

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

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> None:

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

    if pd.isna(p_value):
        return ""

    if p_value < 0.01:
        return "***"

    if p_value < 0.05:
        return "**"

    if p_value < 0.10:
        return "*"

    return ""


def expected_sign_match(
    coefficient: float,
    expected_sign: str,
) -> int:

    if pd.isna(coefficient):
        return 0

    if expected_sign == "POSITIVE":
        return int(
            coefficient > 0
        )

    if expected_sign == "NEGATIVE":
        return int(
            coefficient < 0
        )

    if expected_sign == "DATA_DRIVEN":
        return 1

    return 0


# ============================================================
# 8. PANELİ HAZIRLA
# ============================================================

def prepare_panel(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Panel dosyası bulunamadı:\n{path}"
        )

    panel = normalize_columns(
        pd.read_csv(
            path,
            low_memory=False,
        )
    )

    required = [
        "DATE",
        "DATE_GROUP",
        "ETF_ID",
        "ETF_RETURN",
        "CPU_Z",
        "HIGH_CPU_REGIME",
        "HIGH_VIX_REGIME",
        "CPU_AND_VIX_STRESS",
        "VIX_LEVEL_Z",
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_CHANGE",
    ]

    require_columns(
        panel,
        required,
        "FINAL ECONOMETRIC PANEL",
    )

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
        panel["DATE_GROUP"]
        .astype("string")
        .str.strip()
    )

    numeric_columns = [
        "ETF_RETURN",
        "CPU_Z",
        "HIGH_CPU_REGIME",
        "HIGH_VIX_REGIME",
        "CPU_AND_VIX_STRESS",
        "VIX_LEVEL_Z",
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_CHANGE",
    ]

    for channel in CHANNELS:

        if channel == "PCA_COMPONENT_1":

            candidate_columns = [
                "PCA_COMPONENT_1",
                "CPU_Z_X_PCA_COMPONENT_1",
                "VALID_PCA_MAIN_ROW",
            ]

        else:

            candidate_columns = []

            for quality in [
                "MAIN",
                "MODERATE",
            ]:

                variant = (
                    f"{channel}_{quality}"
                )

                candidate_columns.extend(
                    [
                        variant,
                        f"HIGH_CPU_X_{variant}",
                        f"HIGH_VIX_X_{variant}",
                        f"CPU_VIX_STRESS_X_{variant}",
                        f"VALID_{channel}_{quality}_ROW",
                    ]
                )

        numeric_columns.extend(
            [
                column
                for column in candidate_columns
                if column in panel.columns
            ]
        )

    for column in set(
        numeric_columns
    ):

        if column in panel.columns:

            panel[column] = safe_numeric(
                panel[column]
            )

    panel = panel.loc[
        panel["DATE"].notna()
        & panel["ETF_ID"].notna()
    ].copy()

    duplicate_rows = int(
        panel.duplicated(
            subset=[
                "ETF_ID",
                "DATE",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_rows > 0:

        raise RuntimeError(
            "Duplicate ETF-month satırı bulundu: "
            f"{duplicate_rows}"
        )

    return panel


# ============================================================
# 9. KANAL SÜTUNLARINI BELİRLE
# ============================================================

def resolve_channel_columns(
    panel: pd.DataFrame,
    channel: str,
    quality: str,
) -> dict:

    if channel == "PCA_COMPONENT_1":

        variant = "PCA_COMPONENT_1"

        valid_column = (
            "VALID_PCA_MAIN_ROW"
        )

    else:

        variant = (
            f"{channel}_{quality}"
        )

        valid_column = (
            f"VALID_{channel}_{quality}_ROW"
        )

    required = [
        variant,
        valid_column,
    ]

    require_columns(
        panel,
        required,
        f"{channel} {quality}",
    )

    return {
        "variant": variant,
        "valid_column": valid_column,

        "high_cpu_interaction": (
            f"HIGH_CPU_X_{variant}"
        ),

        "high_vix_interaction": (
            f"HIGH_VIX_X_{variant}"
        ),

        "joint_stress_interaction": (
            f"CPU_VIX_STRESS_X_{variant}"
        ),

        "cpu_interaction": (
            f"CPU_Z_X_{variant}"
        ),

        "vix_interaction": (
            f"VIX_Z_X_{variant}"
        ),

        "continuous_triple": (
            f"CPU_Z_X_VIX_Z_X_{variant}"
        ),
    }


# ============================================================
# 10. EKSİK ETKİLEŞİMLERİ OLUŞTUR
# ============================================================

def add_stress_interactions(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    result = panel.copy()

    result[
        "CPU_Z_X_VIX_Z"
    ] = (
        result["CPU_Z"]
        * result["VIX_LEVEL_Z"]
    )

    for channel in CHANNELS:

        qualities = (
            ["MAIN"]
            if channel == "PCA_COMPONENT_1"
            else [
                "MAIN",
                "MODERATE",
            ]
        )

        for quality in qualities:

            try:

                columns = resolve_channel_columns(
                    result,
                    channel,
                    quality,
                )

            except KeyError:

                continue

            variant = columns[
                "variant"
            ]

            result[
                columns[
                    "high_cpu_interaction"
                ]
            ] = (
                result[
                    "HIGH_CPU_REGIME"
                ]
                * result[
                    variant
                ]
            )

            result[
                columns[
                    "high_vix_interaction"
                ]
            ] = (
                result[
                    "HIGH_VIX_REGIME"
                ]
                * result[
                    variant
                ]
            )

            result[
                columns[
                    "joint_stress_interaction"
                ]
            ] = (
                result[
                    "CPU_AND_VIX_STRESS"
                ]
                * result[
                    variant
                ]
            )

            result[
                columns[
                    "cpu_interaction"
                ]
            ] = (
                result[
                    "CPU_Z"
                ]
                * result[
                    variant
                ]
            )

            result[
                columns[
                    "vix_interaction"
                ]
            ] = (
                result[
                    "VIX_LEVEL_Z"
                ]
                * result[
                    variant
                ]
            )

            result[
                columns[
                    "continuous_triple"
                ]
            ] = (
                result[
                    "CPU_Z"
                ]
                * result[
                    "VIX_LEVEL_Z"
                ]
                * result[
                    variant
                ]
            )

    return result


# ============================================================
# 11. ÖRNEKLEMİ HAZIRLA
# ============================================================

def prepare_channel_sample(
    panel: pd.DataFrame,
    channel: str,
    quality: str,
    model_family: str,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    columns = resolve_channel_columns(
        panel,
        channel,
        quality,
    )

    sample = panel.loc[
        panel[
            columns[
                "valid_column"
            ]
        ]
        == 1
    ].copy()

    if model_family == "REGIME":

        required = [
            "ETF_RETURN",
            "ETF_ID",
            "DATE_GROUP",
            "HIGH_CPU_REGIME",
            "HIGH_VIX_REGIME",
            "CPU_AND_VIX_STRESS",
            columns[
                "high_cpu_interaction"
            ],
            columns[
                "high_vix_interaction"
            ],
            columns[
                "joint_stress_interaction"
            ],
        ] + TIME_CONTROLS

    elif model_family == "CONTINUOUS":

        required = [
            "ETF_RETURN",
            "ETF_ID",
            "DATE_GROUP",
            "CPU_Z",
            "VIX_LEVEL_Z",
            "CPU_Z_X_VIX_Z",
            columns[
                "cpu_interaction"
            ],
            columns[
                "vix_interaction"
            ],
            columns[
                "continuous_triple"
            ],
        ] + TIME_CONTROLS

    else:

        raise ValueError(
            f"Bilinmeyen model ailesi: {model_family}"
        )

    require_columns(
        sample,
        required,
        f"{channel} {quality} {model_family}",
    )

    sample = sample.dropna(
        subset=required
    ).copy()

    sample = sample.sort_values(
        [
            "ETF_ID",
            "DATE",
        ]
    ).reset_index(
        drop=True
    )

    return (
        sample,
        columns,
    )


# ============================================================
# 12. REGİME MODEL FORMÜLÜ
# ============================================================

def build_regime_formula(
    columns: dict,
    month_fe: bool,
) -> str:

    interactions = [
        columns[
            "high_cpu_interaction"
        ],
        columns[
            "high_vix_interaction"
        ],
        columns[
            "joint_stress_interaction"
        ],
    ]

    regressors = []

    if not month_fe:

        regressors.extend(
            [
                "HIGH_CPU_REGIME",
                "HIGH_VIX_REGIME",
                "CPU_AND_VIX_STRESS",
            ]
        )

        regressors.extend(
            TIME_CONTROLS
        )

    regressors.extend(
        interactions
    )

    regressors.append(
        "C(ETF_ID)"
    )

    if month_fe:

        regressors.append(
            "C(DATE_GROUP)"
        )

    return (
        f"{DEPENDENT_VARIABLE} ~ "
        + " + ".join(
            regressors
        )
    )


# ============================================================
# 13. SÜREKLİ MODEL FORMÜLÜ
# ============================================================

def build_continuous_formula(
    columns: dict,
    month_fe: bool,
) -> str:

    interaction_terms = [
        columns[
            "cpu_interaction"
        ],
        columns[
            "vix_interaction"
        ],
        columns[
            "continuous_triple"
        ],
    ]

    regressors = []

    if not month_fe:

        regressors.extend(
            [
                "CPU_Z",
                "VIX_LEVEL_Z",
                "CPU_Z_X_VIX_Z",
            ]
        )

        regressors.extend(
            TIME_CONTROLS
        )

    regressors.extend(
        interaction_terms
    )

    regressors.append(
        "C(ETF_ID)"
    )

    if month_fe:

        regressors.append(
            "C(DATE_GROUP)"
        )

    return (
        f"{DEPENDENT_VARIABLE} ~ "
        + " + ".join(
            regressors
        )
    )


# ============================================================
# 14. MODEL TAHMİNİ
# ============================================================

def fit_model(
    sample: pd.DataFrame,
    formula: str,
    covariance_type: str,
):

    base_result = smf.ols(
        formula=formula,
        data=sample,
    ).fit()

    custom_covariance = None

    if covariance_type == "CLUSTER_DATE":

        robust_result = (
            base_result
            .get_robustcov_results(
                cov_type="cluster",
                groups=sample[
                    "DATE_GROUP"
                ],
                use_correction=True,
            )
        )

    elif covariance_type == "CLUSTER_ETF":

        robust_result = (
            base_result
            .get_robustcov_results(
                cov_type="cluster",
                groups=sample[
                    "ETF_ID"
                ],
                use_correction=True,
            )
        )

    elif covariance_type == "TWO_WAY_CLUSTER":

        etf_groups = pd.factorize(
            sample[
                "ETF_ID"
            ]
        )[0]

        date_groups = pd.factorize(
            sample[
                "DATE_GROUP"
            ]
        )[0]

        covariance_tuple = (
            cov_cluster_2groups(
                base_result,
                etf_groups,
                date_groups,
            )
        )

        custom_covariance = (
            covariance_tuple[0]
        )

        robust_result = base_result

    else:

        raise ValueError(
            "Bilinmeyen covariance türü: "
            f"{covariance_type}"
        )

    return (
        base_result,
        robust_result,
        custom_covariance,
    )


# ============================================================
# 15. SONUÇLARI ÇIKAR
# ============================================================

def extract_results(
    base_result,
    robust_result,
    custom_covariance,
    sample: pd.DataFrame,
    model_name: str,
    channel: str,
    channel_role: str,
    quality: str,
    model_family: str,
    covariance_type: str,
    expected_sign: str,
    target_parameter: str,
    formula: str,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    parameter_names = list(
        base_result.params.index
    )

    coefficients = np.asarray(
        base_result.params,
        dtype=float,
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
            out=np.full(
                len(coefficients),
                np.nan,
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
            robust_result.bse,
            dtype=float,
        )

        test_statistics = np.asarray(
            robust_result.tvalues,
            dtype=float,
        )

        p_values = np.asarray(
            robust_result.pvalues,
            dtype=float,
        )

        confidence_intervals = np.asarray(
            robust_result.conf_int(),
            dtype=float,
        )

        confidence_lower = (
            confidence_intervals[
                :,
                0,
            ]
        )

        confidence_upper = (
            confidence_intervals[
                :,
                1,
            ]
        )

    rows = []

    for index, parameter in enumerate(
        parameter_names
    ):

        coefficient = float(
            coefficients[index]
        )

        p_value = float(
            p_values[index]
        )

        rows.append(
            {
                "MODEL": model_name,

                "MODEL_FAMILY": model_family,

                "CHANNEL": channel,

                "CHANNEL_ROLE": channel_role,

                "QUALITY": quality,

                "COVARIANCE": covariance_type,

                "PARAMETER": parameter,

                "TARGET_PARAMETER": (
                    target_parameter
                ),

                "COEFFICIENT": coefficient,

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

                "EXPECTED_SIGN": expected_sign,

                "EXPECTED_SIGN_MATCH": (
                    expected_sign_match(
                        coefficient,
                        expected_sign,
                    )
                    if parameter
                    == target_parameter
                    else np.nan
                ),

                "N_OBSERVATIONS": int(
                    base_result.nobs
                ),

                "N_ETFS": int(
                    sample[
                        "ETF_ID"
                    ].nunique()
                ),

                "N_MONTHS": int(
                    sample[
                        "DATE_GROUP"
                    ].nunique()
                ),

                "N_JOINT_STRESS_MONTHS": int(
                    sample.loc[
                        sample[
                            "CPU_AND_VIX_STRESS"
                        ]
                        == 1,
                        "DATE_GROUP",
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
        "MODEL": model_name,

        "MODEL_FAMILY": model_family,

        "CHANNEL": channel,

        "CHANNEL_ROLE": channel_role,

        "QUALITY": quality,

        "COVARIANCE": covariance_type,

        "TARGET_PARAMETER": (
            target_parameter
        ),

        "FORMULA": formula,

        "N_OBSERVATIONS": int(
            base_result.nobs
        ),

        "N_ETFS": int(
            sample[
                "ETF_ID"
            ].nunique()
        ),

        "N_MONTHS": int(
            sample[
                "DATE_GROUP"
            ].nunique()
        ),

        "N_JOINT_STRESS_MONTHS": int(
            sample.loc[
                sample[
                    "CPU_AND_VIX_STRESS"
                ]
                == 1,
                "DATE_GROUP",
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
    }

    return (
        pd.DataFrame(
            rows
        ),
        diagnostic,
    )


# ============================================================
# 16. TÜM STRES MODELLERİ
# ============================================================

def estimate_all_models(
    panel: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    result_frames = []

    diagnostic_rows = []

    sample_rows = []

    for channel, settings in (
        CHANNELS.items()
    ):

        qualities = (
            ["MAIN"]
            if channel == "PCA_COMPONENT_1"
            else [
                "MAIN",
                "MODERATE",
            ]
        )

        for quality in qualities:

            for model_family in [
                "REGIME",
                "CONTINUOUS",
            ]:

                try:

                    (
                        sample,
                        columns,
                    ) = prepare_channel_sample(
                        panel=panel,
                        channel=channel,
                        quality=quality,
                        model_family=model_family,
                    )

                except KeyError as error:

                    print(
                        f"  Atlandı: "
                        f"{channel} {quality} "
                        f"{model_family}"
                    )

                    print(
                        f"  Neden: {error}"
                    )

                    continue

                n_rows = len(
                    sample
                )

                n_etfs = sample[
                    "ETF_ID"
                ].nunique()

                sample_rows.append(
                    {
                        "CHANNEL": channel,

                        "CHANNEL_ROLE": settings[
                            "role"
                        ],

                        "QUALITY": quality,

                        "MODEL_FAMILY": (
                            model_family
                        ),

                        "N_ROWS": int(
                            n_rows
                        ),

                        "N_ETFS": int(
                            n_etfs
                        ),

                        "N_MONTHS": int(
                            sample[
                                "DATE_GROUP"
                            ].nunique()
                        ),

                        "N_HIGH_CPU_MONTHS": int(
                            sample.loc[
                                sample[
                                    "HIGH_CPU_REGIME"
                                ]
                                == 1,
                                "DATE_GROUP",
                            ].nunique()
                        ),

                        "N_HIGH_VIX_MONTHS": int(
                            sample.loc[
                                sample[
                                    "HIGH_VIX_REGIME"
                                ]
                                == 1,
                                "DATE_GROUP",
                            ].nunique()
                        ),

                        "N_JOINT_STRESS_MONTHS": int(
                            sample.loc[
                                sample[
                                    "CPU_AND_VIX_STRESS"
                                ]
                                == 1,
                                "DATE_GROUP",
                            ].nunique()
                        ),

                        "START_DATE": (
                            sample[
                                "DATE"
                            ].min()
                        ),

                        "END_DATE": (
                            sample[
                                "DATE"
                            ].max()
                        ),
                    }
                )

                if (
                    n_rows < MIN_OBSERVATIONS
                    or n_etfs < MIN_ETFS
                ):

                    print(
                        f"  Tahmin edilmedi: "
                        f"{channel} {quality} "
                        f"{model_family} | "
                        f"ETF={n_etfs}, rows={n_rows}"
                    )

                    continue

                if model_family == "REGIME":

                    specifications = (
                        REGIME_MODEL_SPECS
                    )

                    target_parameter = (
                        columns[
                            "joint_stress_interaction"
                        ]
                    )

                else:

                    specifications = (
                        CONTINUOUS_MODEL_SPECS
                    )

                    target_parameter = (
                        columns[
                            "continuous_triple"
                        ]
                    )

                for specification in (
                    specifications
                ):

                    model_name = (
                        f"{channel}_"
                        f"{quality}_"
                        f"{specification['suffix']}"
                    )

                    print(
                        f"  Tahmin ediliyor: "
                        f"{model_name}"
                    )

                    if model_family == "REGIME":

                        formula = (
                            build_regime_formula(
                                columns=columns,
                                month_fe=(
                                    specification[
                                        "month_fe"
                                    ]
                                ),
                            )
                        )

                    else:

                        formula = (
                            build_continuous_formula(
                                columns=columns,
                                month_fe=(
                                    specification[
                                        "month_fe"
                                    ]
                                ),
                            )
                        )

                    (
                        base_result,
                        robust_result,
                        custom_covariance,
                    ) = fit_model(
                        sample=sample,
                        formula=formula,
                        covariance_type=(
                            specification[
                                "covariance"
                            ]
                        ),
                    )

                    (
                        coefficient_table,
                        diagnostic,
                    ) = extract_results(
                        base_result=base_result,
                        robust_result=robust_result,
                        custom_covariance=(
                            custom_covariance
                        ),
                        sample=sample,
                        model_name=model_name,
                        channel=channel,
                        channel_role=settings[
                            "role"
                        ],
                        quality=quality,
                        model_family=model_family,
                        covariance_type=(
                            specification[
                                "covariance"
                            ]
                        ),
                        expected_sign=settings[
                            "expected_joint_sign"
                        ],
                        target_parameter=(
                            target_parameter
                        ),
                        formula=formula,
                    )

                    result_frames.append(
                        coefficient_table
                    )

                    diagnostic_rows.append(
                        diagnostic
                    )

    if result_frames:

        all_results = pd.concat(
            result_frames,
            ignore_index=True,
        )

    else:

        all_results = pd.DataFrame()

    return (
        all_results,
        pd.DataFrame(
            diagnostic_rows
        ),
        pd.DataFrame(
            sample_rows
        ),
    )


# ============================================================
# 17. WILD-CLUSTER BOOTSTRAP
# ============================================================

def get_clustered_t_statistic(
    result,
    sample: pd.DataFrame,
    parameter: str,
) -> float:

    robust = (
        result
        .get_robustcov_results(
            cov_type="cluster",
            groups=sample[
                "DATE_GROUP"
            ],
            use_correction=True,
        )
    )

    parameter_names = list(
        result.params.index
    )

    parameter_index = (
        parameter_names.index(
            parameter
        )
    )

    return float(
        robust.tvalues[
            parameter_index
        ]
    )


def wild_cluster_bootstrap_test(
    sample: pd.DataFrame,
    unrestricted_formula: str,
    restricted_formula: str,
    target_parameter: str,
    replications: int,
    seed: int,
) -> dict:
    """
    Tarih kümeleri üzerinde Rademacher wild-cluster bootstrap.

    Null hipotezi:
        target interaction coefficient = 0

    Restricted model target interaction olmadan tahmin edilir.
    Restricted residual'lar tarih kümeleri düzeyinde +1/-1 ile
    çarpılır. Her bootstrap örnekleminde unrestricted model yeniden
    tahmin edilir.
    """

    unrestricted_result = smf.ols(
        unrestricted_formula,
        data=sample,
    ).fit()

    observed_t = (
        get_clustered_t_statistic(
            result=unrestricted_result,
            sample=sample,
            parameter=target_parameter,
        )
    )

    restricted_result = smf.ols(
        restricted_formula,
        data=sample,
    ).fit()

    fitted_null = np.asarray(
        restricted_result.fittedvalues,
        dtype=float,
    )

    residual_null = np.asarray(
        restricted_result.resid,
        dtype=float,
    )

    unique_clusters = (
        sample[
            "DATE_GROUP"
        ]
        .drop_duplicates()
        .tolist()
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_t_values = []

    bootstrap_data = sample.copy()

    original_y = bootstrap_data[
        DEPENDENT_VARIABLE
    ].copy()

    for replication in range(
        replications
    ):

        cluster_weights = {
            cluster: rng.choice(
                [
                    -1.0,
                    1.0,
                ]
            )
            for cluster
            in unique_clusters
        }

        observation_weights = (
            sample[
                "DATE_GROUP"
            ]
            .map(
                cluster_weights
            )
            .to_numpy(
                dtype=float
            )
        )

        bootstrap_y = (
            fitted_null
            + residual_null
            * observation_weights
        )

        bootstrap_data[
            DEPENDENT_VARIABLE
        ] = bootstrap_y

        try:

            bootstrap_result = smf.ols(
                unrestricted_formula,
                data=bootstrap_data,
            ).fit()

            bootstrap_t = (
                get_clustered_t_statistic(
                    result=bootstrap_result,
                    sample=bootstrap_data,
                    parameter=target_parameter,
                )
            )

            if np.isfinite(
                bootstrap_t
            ):

                bootstrap_t_values.append(
                    bootstrap_t
                )

        except Exception:

            continue

    bootstrap_data[
        DEPENDENT_VARIABLE
    ] = original_y

    valid_replications = len(
        bootstrap_t_values
    )

    if valid_replications == 0:

        bootstrap_p_value = np.nan

    else:

        bootstrap_array = np.asarray(
            bootstrap_t_values
        )

        bootstrap_p_value = (
            1.0
            + np.sum(
                np.abs(
                    bootstrap_array
                )
                >= abs(
                    observed_t
                )
            )
        ) / (
            valid_replications
            + 1.0
        )

    return {
        "OBSERVED_T_STATISTIC": (
            observed_t
        ),

        "BOOTSTRAP_P_VALUE": (
            bootstrap_p_value
        ),

        "REQUESTED_REPLICATIONS": (
            replications
        ),

        "VALID_REPLICATIONS": (
            valid_replications
        ),
    }


def run_primary_bootstraps(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    primary_channels = [
        channel
        for channel, settings
        in CHANNELS.items()
        if settings[
            "role"
        ]
        == "PRIMARY"
    ]

    for channel_index, channel in enumerate(
        primary_channels
    ):

        (
            sample,
            columns,
        ) = prepare_channel_sample(
            panel=panel,
            channel=channel,
            quality="MAIN",
            model_family="REGIME",
        )

        if (
            len(sample)
            < MIN_OBSERVATIONS
            or sample[
                "ETF_ID"
            ].nunique()
            < MIN_ETFS
        ):

            continue

        target = columns[
            "joint_stress_interaction"
        ]

        unrestricted_formula = (
            build_regime_formula(
                columns=columns,
                month_fe=False,
            )
        )

        restricted_regressors = [
            "HIGH_CPU_REGIME",
            "HIGH_VIX_REGIME",
            "CPU_AND_VIX_STRESS",
            columns[
                "high_cpu_interaction"
            ],
            columns[
                "high_vix_interaction"
            ],
        ] + TIME_CONTROLS + [
            "C(ETF_ID)",
        ]

        restricted_formula = (
            f"{DEPENDENT_VARIABLE} ~ "
            + " + ".join(
                restricted_regressors
            )
        )

        print(
            f"  Wild bootstrap: "
            f"{channel} | "
            f"B={BOOTSTRAP_REPLICATIONS}"
        )

        bootstrap_result = (
            wild_cluster_bootstrap_test(
                sample=sample,
                unrestricted_formula=(
                    unrestricted_formula
                ),
                restricted_formula=(
                    restricted_formula
                ),
                target_parameter=target,
                replications=(
                    BOOTSTRAP_REPLICATIONS
                ),
                seed=(
                    BOOTSTRAP_SEED
                    + channel_index
                ),
            )
        )

        rows.append(
            {
                "CHANNEL": channel,

                "QUALITY": "MAIN",

                "TARGET_PARAMETER": target,

                "N_OBSERVATIONS": len(
                    sample
                ),

                "N_ETFS": sample[
                    "ETF_ID"
                ].nunique(),

                "N_DATE_CLUSTERS": sample[
                    "DATE_GROUP"
                ].nunique(),

                "N_JOINT_STRESS_MONTHS": (
                    sample.loc[
                        sample[
                            "CPU_AND_VIX_STRESS"
                        ]
                        == 1,
                        "DATE_GROUP",
                    ].nunique()
                ),

                **bootstrap_result,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. ANA KATSAYILARI SEÇ
# ============================================================

def build_key_results(
    all_results: pd.DataFrame,
) -> pd.DataFrame:

    if all_results.empty:

        return pd.DataFrame()

    target_rows = (
        all_results[
            "PARAMETER"
        ]
        == all_results[
            "TARGET_PARAMETER"
        ]
    )

    component_rows = (
        all_results[
            "PARAMETER"
        ].str.startswith(
            (
                "HIGH_CPU_X_",
                "HIGH_VIX_X_",
                "CPU_VIX_STRESS_X_",
                "CPU_Z_X_",
                "VIX_Z_X_",
            ),
            na=False,
        )
    )

    key_results = all_results.loc[
        target_rows
        | component_rows
    ].copy()

    key_results[
        "COEFFICIENT_WITH_STARS"
    ] = (
        key_results[
            "COEFFICIENT"
        ]
        .map(
            lambda value: (
                f"{value:.6f}"
            )
        )
        + key_results[
            "SIGNIFICANCE"
        ]
    )

    return key_results


# ============================================================
# 19. İŞARET TUTARLILIĞI
# ============================================================

def build_sign_consistency(
    key_results: pd.DataFrame,
) -> pd.DataFrame:

    target_results = key_results.loc[
        key_results[
            "PARAMETER"
        ]
        == key_results[
            "TARGET_PARAMETER"
        ]
    ].copy()

    rows = []

    for (
        channel,
        quality,
        model_family,
    ), group in target_results.groupby(
        [
            "CHANNEL",
            "QUALITY",
            "MODEL_FAMILY",
        ],
        dropna=False,
    ):

        coefficients = group[
            "COEFFICIENT"
        ]

        rows.append(
            {
                "CHANNEL": channel,

                "QUALITY": quality,

                "MODEL_FAMILY": (
                    model_family
                ),

                "N_MODELS": len(
                    group
                ),

                "NEGATIVE_MODELS": int(
                    (
                        coefficients < 0
                    ).sum()
                ),

                "POSITIVE_MODELS": int(
                    (
                        coefficients > 0
                    ).sum()
                ),

                "EXPECTED_SIGN_MATCH_MODELS": int(
                    safe_numeric(
                        group[
                            "EXPECTED_SIGN_MATCH"
                        ]
                    )
                    .fillna(0)
                    .sum()
                ),

                "SIGNIFICANT_AT_10PCT": int(
                    (
                        group[
                            "P_VALUE"
                        ]
                        < 0.10
                    ).sum()
                ),

                "SIGNIFICANT_AT_5PCT": int(
                    (
                        group[
                            "P_VALUE"
                        ]
                        < 0.05
                    ).sum()
                ),

                "MEAN_COEFFICIENT": float(
                    coefficients.mean()
                ),

                "MIN_P_VALUE": float(
                    group[
                        "P_VALUE"
                    ].min()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 20. VIF
# ============================================================

def build_vif_diagnostics(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for channel, settings in (
        CHANNELS.items()
    ):

        if settings[
            "role"
        ] != "PRIMARY":

            continue

        try:

            (
                sample,
                columns,
            ) = prepare_channel_sample(
                panel=panel,
                channel=channel,
                quality="MAIN",
                model_family="REGIME",
            )

        except KeyError:

            continue

        variables = [
            "HIGH_CPU_REGIME",
            "HIGH_VIX_REGIME",
            "CPU_AND_VIX_STRESS",
            columns[
                "high_cpu_interaction"
            ],
            columns[
                "high_vix_interaction"
            ],
            columns[
                "joint_stress_interaction"
            ],
        ] + TIME_CONTROLS

        data = (
            sample[
                variables
            ]
            .apply(
                safe_numeric
            )
            .dropna()
            .copy()
        )

        if data.empty:
            continue

        design = data.copy()

        design.insert(
            0,
            "INTERCEPT",
            1.0,
        )

        for index, variable in enumerate(
            design.columns
        ):

            if variable == "INTERCEPT":
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
                    "CHANNEL": channel,

                    "VARIABLE": variable,

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


# ============================================================
# 21. MODEL SÖZLÜĞÜ
# ============================================================

def build_model_dictionary() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "MODEL_FAMILY": (
                    "REGIME_ACTIVATION"
                ),

                "DEPENDENT_VARIABLE": (
                    "ETF_RETURN"
                ),

                "KEY_PARAMETER": (
                    "Architecture × CPU_AND_VIX_STRESS"
                ),

                "LOWER_ORDER_TERMS": (
                    "High CPU, High VIX, Joint Stress, "
                    "Architecture × High CPU, "
                    "Architecture × High VIX"
                ),

                "FIXED_EFFECTS": (
                    "ETF fixed effects; alternative ETF and "
                    "month fixed effects"
                ),

                "STANDARD_ERRORS": (
                    "Date-clustered, ETF-clustered and "
                    "two-way clustered"
                ),

                "THEORY": (
                    "Joint market stress activates latent "
                    "portfolio architecture exposure."
                ),
            },

            {
                "MODEL_FAMILY": (
                    "CONTINUOUS_ACTIVATION"
                ),

                "DEPENDENT_VARIABLE": (
                    "ETF_RETURN"
                ),

                "KEY_PARAMETER": (
                    "CPU_Z × VIX_LEVEL_Z × Architecture"
                ),

                "LOWER_ORDER_TERMS": (
                    "CPU_Z, VIX_Z, CPU_Z × VIX_Z, "
                    "CPU_Z × Architecture, "
                    "VIX_Z × Architecture"
                ),

                "FIXED_EFFECTS": (
                    "ETF fixed effects; alternative ETF and "
                    "month fixed effects"
                ),

                "STANDARD_ERRORS": (
                    "Date-clustered, ETF-clustered and "
                    "two-way clustered"
                ),

                "THEORY": (
                    "Architecture pricing varies continuously "
                    "with the joint intensity of CPU and market stress."
                ),
            },

            {
                "MODEL_FAMILY": (
                    "WILD_CLUSTER_BOOTSTRAP"
                ),

                "DEPENDENT_VARIABLE": (
                    "ETF_RETURN"
                ),

                "KEY_PARAMETER": (
                    "Architecture × CPU_AND_VIX_STRESS"
                ),

                "LOWER_ORDER_TERMS": (
                    "Restricted null model excludes only the "
                    "joint architecture interaction."
                ),

                "FIXED_EFFECTS": (
                    "ETF fixed effects"
                ),

                "STANDARD_ERRORS": (
                    "Date-clustered bootstrap t-statistic"
                ),

                "THEORY": (
                    "Finite-sample inference for a limited number "
                    "of joint-stress months."
                ),
            },
        ]
    )


# ============================================================
# 22. VALIDATION
# ============================================================

def build_validation(
    all_results: pd.DataFrame,
    key_results: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    sample_diagnostics: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    vif_diagnostics: pd.DataFrame,
) -> pd.DataFrame:

    target_results = key_results.loc[
        key_results[
            "PARAMETER"
        ]
        == key_results[
            "TARGET_PARAMETER"
        ]
    ]

    primary_regime_channels = (
        target_results.loc[
            (
                target_results[
                    "CHANNEL_ROLE"
                ]
                == "PRIMARY"
            )
            & (
                target_results[
                    "MODEL_FAMILY"
                ]
                == "REGIME"
            ),
            "CHANNEL",
        ]
        .nunique()
    )

    primary_continuous_channels = (
        target_results.loc[
            (
                target_results[
                    "CHANNEL_ROLE"
                ]
                == "PRIMARY"
            )
            & (
                target_results[
                    "MODEL_FAMILY"
                ]
                == "CONTINUOUS"
            ),
            "CHANNEL",
        ]
        .nunique()
    )

    finite_coefficients = int(
        np.isfinite(
            all_results[
                "COEFFICIENT"
            ]
        ).all()
    )

    rows = [
        {
            "CHECK": "ALL_RESULT_ROWS",
            "VALUE": len(
                all_results
            ),
            "PASS": int(
                len(
                    all_results
                ) > 0
            ),
        },

        {
            "CHECK": "KEY_RESULT_ROWS",
            "VALUE": len(
                key_results
            ),
            "PASS": int(
                len(
                    key_results
                ) > 0
            ),
        },

        {
            "CHECK": (
                "PRIMARY_REGIME_CHANNELS_ESTIMATED"
            ),
            "VALUE": (
                primary_regime_channels
            ),
            "PASS": int(
                primary_regime_channels
                == 4
            ),
        },

        {
            "CHECK": (
                "PRIMARY_CONTINUOUS_CHANNELS_ESTIMATED"
            ),
            "VALUE": (
                primary_continuous_channels
            ),
            "PASS": int(
                primary_continuous_channels
                == 4
            ),
        },

        {
            "CHECK": (
                "BOOTSTRAP_PRIMARY_CHANNELS"
            ),
            "VALUE": len(
                bootstrap_results
            ),
            "PASS": int(
                len(
                    bootstrap_results
                )
                == 4
            ),
        },

        {
            "CHECK": (
                "VALID_BOOTSTRAP_REPLICATIONS"
            ),
            "VALUE": int(
                bootstrap_results[
                    "VALID_REPLICATIONS"
                ].sum()
                if not bootstrap_results.empty
                else 0
            ),
            "PASS": int(
                (
                    not bootstrap_results.empty
                )
                and (
                    bootstrap_results[
                        "VALID_REPLICATIONS"
                    ]
                    > 0
                ).all()
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
                )
                > 0
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
                > 0
            ),
        },

        {
            "CHECK": (
                "VIF_DIAGNOSTIC_ROWS"
            ),
            "VALUE": len(
                vif_diagnostics
            ),
            "PASS": int(
                len(
                    vif_diagnostics
                )
                > 0
            ),
        },

        {
            "CHECK": (
                "FINITE_COEFFICIENTS"
            ),
            "VALUE": finite_coefficients,
            "PASS": finite_coefficients,
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 23. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 94)
    print("27 - MARKET-STRESS ACTIVATION MODELS")
    print("=" * 94)

    print(
        "\n1/9 - Nihai ekonometrik panel okunuyor..."
    )

    panel = prepare_panel(
        PANEL_FILE
    )

    panel = add_stress_interactions(
        panel
    )

    print(
        f"Panel satırı: "
        f"{len(panel):,}"
    )

    print(
        f"ETF sayısı: "
        f"{panel['ETF_ID'].nunique():,}"
    )

    print(
        f"Ay sayısı: "
        f"{panel['DATE_GROUP'].nunique():,}"
    )

    print(
        f"Joint CPU–VIX stress ayı: "
        f"{panel.loc[panel['CPU_AND_VIX_STRESS'] == 1, 'DATE_GROUP'].nunique():,}"
    )

    print(
        "\n2/9 - Rejim ve sürekli stres modelleri tahmin ediliyor..."
    )

    (
        all_results,
        model_diagnostics,
        sample_diagnostics,
    ) = estimate_all_models(
        panel
    )

    if all_results.empty:

        raise RuntimeError(
            "Hiçbir stres modeli tahmin edilemedi."
        )

    print(
        "\n3/9 - Ana katsayı tabloları hazırlanıyor..."
    )

    key_results = build_key_results(
        all_results
    )

    target_results = key_results.loc[
        key_results[
            "PARAMETER"
        ]
        == key_results[
            "TARGET_PARAMETER"
        ]
    ].copy()

    primary_results = target_results.loc[
        target_results[
            "CHANNEL_ROLE"
        ]
        == "PRIMARY"
    ].copy()

    robustness_results = target_results.loc[
        target_results[
            "CHANNEL_ROLE"
        ]
        != "PRIMARY"
    ].copy()

    continuous_results = target_results.loc[
        target_results[
            "MODEL_FAMILY"
        ]
        == "CONTINUOUS"
    ].copy()

    sign_consistency = (
        build_sign_consistency(
            key_results
        )
    )

    print(
        "\n4/9 - Main-quality primary-channel wild bootstrap çalıştırılıyor..."
    )

    bootstrap_results = (
        run_primary_bootstraps(
            panel
        )
    )

    print(
        "\n5/9 - VIF diagnostics hazırlanıyor..."
    )

    vif_diagnostics = (
        build_vif_diagnostics(
            panel
        )
    )

    print(
        "\n6/9 - Model sözlüğü hazırlanıyor..."
    )

    model_dictionary = (
        build_model_dictionary()
    )

    print(
        "\n7/9 - Validation kontrolleri çalıştırılıyor..."
    )

    validation = build_validation(
        all_results=all_results,
        key_results=key_results,
        model_diagnostics=(
            model_diagnostics
        ),
        sample_diagnostics=(
            sample_diagnostics
        ),
        bootstrap_results=(
            bootstrap_results
        ),
        vif_diagnostics=(
            vif_diagnostics
        ),
    )

    print(
        "\n8/9 - Sonuç dosyaları kaydediliyor..."
    )

    all_results.to_csv(
        ALL_RESULTS_FILE,
        index=False,
    )

    key_results.to_csv(
        KEY_RESULTS_FILE,
        index=False,
    )

    primary_results.to_csv(
        PRIMARY_RESULTS_FILE,
        index=False,
    )

    robustness_results.to_csv(
        ROBUSTNESS_RESULTS_FILE,
        index=False,
    )

    continuous_results.to_csv(
        CONTINUOUS_RESULTS_FILE,
        index=False,
    )

    bootstrap_results.to_csv(
        BOOTSTRAP_RESULTS_FILE,
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

    sign_consistency.to_csv(
        SIGN_CONSISTENCY_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    model_dictionary.to_csv(
        MODEL_DICTIONARY_FILE,
        index=False,
    )

    print(
        "\n9/9 - Ana sonuçlar ekrana yazdırılıyor..."
    )

    print(
        "\nSTRESS ACTIVATION MODELS HAZIR"
    )

    print("=" * 94)

    print(
        "\nMain-quality primary-channel joint-stress sonuçları:"
    )

    main_regime = primary_results.loc[
        (
            primary_results[
                "QUALITY"
            ]
            == "MAIN"
        )
        & (
            primary_results[
                "MODEL_FAMILY"
            ]
            == "REGIME"
        )
    ].copy()

    print(
        main_regime[
            [
                "CHANNEL",
                "MODEL",
                "COVARIANCE",
                "COEFFICIENT",
                "STD_ERROR",
                "P_VALUE",
                "SIGNIFICANCE",
                "EXPECTED_SIGN",
                "EXPECTED_SIGN_MATCH",
                "N_OBSERVATIONS",
                "N_ETFS",
                "N_JOINT_STRESS_MONTHS",
                "R_SQUARED",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nModerate-quality primary-channel joint-stress sonuçları:"
    )

    moderate_regime = primary_results.loc[
        (
            primary_results[
                "QUALITY"
            ]
            == "MODERATE"
        )
        & (
            primary_results[
                "MODEL_FAMILY"
            ]
            == "REGIME"
        )
    ].copy()

    print(
        moderate_regime[
            [
                "CHANNEL",
                "MODEL",
                "COVARIANCE",
                "COEFFICIENT",
                "STD_ERROR",
                "P_VALUE",
                "SIGNIFICANCE",
                "EXPECTED_SIGN",
                "EXPECTED_SIGN_MATCH",
                "N_OBSERVATIONS",
                "N_ETFS",
                "N_JOINT_STRESS_MONTHS",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nMain-quality continuous triple-interaction sonuçları:"
    )

    main_continuous = primary_results.loc[
        (
            primary_results[
                "QUALITY"
            ]
            == "MAIN"
        )
        & (
            primary_results[
                "MODEL_FAMILY"
            ]
            == "CONTINUOUS"
        )
    ].copy()

    print(
        main_continuous[
            [
                "CHANNEL",
                "MODEL",
                "COVARIANCE",
                "COEFFICIENT",
                "STD_ERROR",
                "P_VALUE",
                "SIGNIFICANCE",
                "EXPECTED_SIGN",
                "EXPECTED_SIGN_MATCH",
                "N_OBSERVATIONS",
                "N_ETFS",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nComposite ve PCA stress robustness sonuçları:"
    )

    if robustness_results.empty:

        print(
            "Robustness sonucu üretilemedi."
        )

    else:

        print(
            robustness_results[
                [
                    "CHANNEL",
                    "QUALITY",
                    "MODEL_FAMILY",
                    "MODEL",
                    "COVARIANCE",
                    "COEFFICIENT",
                    "STD_ERROR",
                    "P_VALUE",
                    "SIGNIFICANCE",
                    "N_OBSERVATIONS",
                    "N_ETFS",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\nWild-cluster bootstrap sonuçları:"
    )

    print(
        bootstrap_results.to_string(
            index=False
        )
    )

    print(
        "\nİşaret ve anlamlılık tutarlılığı:"
    )

    print(
        sign_consistency.to_string(
            index=False
        )
    )

    print(
        "\nMain-quality rejim modeli VIF sonuçları:"
    )

    print(
        vif_diagnostics.to_string(
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
        ]
        == 0
    ]

    print(
        "\nBaşarısız kontrol sayısı: "
        f"{len(failed_checks):,}"
    )

    print(
        "\nMETODOLOJİK NOT:"
    )

    print(
        "Joint-stress katsayısı, Architecture × High CPU ve "
        "Architecture × High VIX etkileri kontrol edildikten sonra "
        "CPU ile piyasa stresinin eşzamanlı aktivasyon etkisini ölçer."
    )

    print(
        "Wild-cluster bootstrap yalnızca dört primary channel ve "
        "main-quality örneklem için uygulanmıştır."
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        PRIMARY_RESULTS_FILE
    )

    print(
        ROBUSTNESS_RESULTS_FILE
    )

    print(
        CONTINUOUS_RESULTS_FILE
    )

    print(
        BOOTSTRAP_RESULTS_FILE
    )

    print(
        KEY_RESULTS_FILE
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
        SIGN_CONSISTENCY_FILE
    )

    print(
        VALIDATION_FILE
    )


if __name__ == "__main__":
    main()