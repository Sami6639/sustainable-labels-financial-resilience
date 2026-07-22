from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


# ============================================================
# 26_estimate_final_channel_models.py
#
# AMAÇ
# ----
# Nihai portfolio architecture kanallarının Climate Policy
# Uncertainty ile etkileşimlerinin ETF getirileri üzerindeki
# koşullu fiyatlama etkisini tahmin etmek.
#
# RQ1
# Is climate policy uncertainty priced uniformly across
# sustainable portfolios, or does pricing depend on portfolio
# architecture?
#
# RQ2
# Which embedded portfolio-architecture channels amplify or
# attenuate climate-policy uncertainty exposure?
#
# ANA MODEL
# ---------
# ETF_RETURN_i,t =
#     ETF fixed effects
#     + CPU_Z_t
#     + CPU_Z_t × Architecture_i
#     + Market controls_t
#     + error_i,t
#
# ÖNEMLİ
# -------
# Architecture seviyeleri zaman içinde sabit olduğu için ETF fixed
# effects tarafından emilir. Ana tanımlayıcı katsayı:
#
# CPU_Z × Architecture
#
# katsayısıdır.
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
    / "26_final_channel_all_model_results.csv"
)

KEY_RESULTS_FILE = (
    OUTPUT_DIR
    / "26_final_channel_key_coefficients.csv"
)

PRIMARY_CHANNEL_RESULTS_FILE = (
    OUTPUT_DIR
    / "26_primary_channel_results.csv"
)

ROBUSTNESS_RESULTS_FILE = (
    OUTPUT_DIR
    / "26_composite_and_pca_robustness_results.csv"
)

MODEL_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "26_final_channel_model_diagnostics.csv"
)

SAMPLE_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "26_final_channel_sample_diagnostics.csv"
)

VIF_FILE = (
    OUTPUT_DIR
    / "26_final_channel_vif_diagnostics.csv"
)

SIGN_CONSISTENCY_FILE = (
    OUTPUT_DIR
    / "26_final_channel_sign_consistency.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "26_final_channel_validation.csv"
)

MODEL_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "26_final_channel_model_dictionary.csv"
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


# ============================================================
# 5. ANA VE ROBUSTNESS KANALLARI
# ============================================================

CHANNELS = {
    "INTERNAL_FINANCIAL_CAPACITY": {
        "role": "PRIMARY",
        "expected_sign": "POSITIVE_OR_LESS_NEGATIVE",
        "theory": (
            "Profitability and liquidity buffers should attenuate "
            "the negative pricing effect of climate-policy uncertainty."
        ),
    },

    "EXTERNAL_FINANCING_DEPENDENCE": {
        "role": "PRIMARY",
        "expected_sign": "NEGATIVE",
        "theory": (
            "Dependence on external capital should amplify uncertainty-"
            "driven financing constraints."
        ),
    },

    "GROWTH_DURATION_EXPOSURE_FINAL": {
        "role": "PRIMARY",
        "expected_sign": "NEGATIVE",
        "theory": (
            "Investment, innovation and growth-duration exposure should "
            "increase sensitivity to policy and discount-rate uncertainty."
        ),
    },

    "PORTFOLIO_CONCENTRATION_FINAL": {
        "role": "PRIMARY",
        "expected_sign": "NEGATIVE",
        "theory": (
            "Concentrated portfolios should exhibit stronger exposure to "
            "dominant holdings and transition-sensitive firms."
        ),
    },

    "FINANCIAL_ARCHITECTURE_RISK_FINAL": {
        "role": "COMPOSITE_ROBUSTNESS",
        "expected_sign": "NEGATIVE",
        "theory": (
            "A higher financial-architecture risk score should amplify "
            "the adverse pricing effect of CPU."
        ),
    },

    "EXTENDED_ARCHITECTURE_RISK_FINAL": {
        "role": "COMPOSITE_ROBUSTNESS",
        "expected_sign": "NEGATIVE",
        "theory": (
            "Financial architecture combined with concentration should "
            "amplify CPU sensitivity."
        ),
    },

    "PCA_COMPONENT_1": {
        "role": "PCA_ROBUSTNESS",
        "expected_sign": "DATA_DRIVEN",
        "theory": (
            "The first principal component captures a common data-driven "
            "portfolio-architecture dimension."
        ),
    },
}


# ============================================================
# 6. MODEL SPESİFİKASYONLARI
# ============================================================

MODEL_SPECIFICATIONS = [
    {
        "suffix": "POOLED_HC3",
        "covariance": "HC3",
        "etf_fe": False,
        "month_fe": False,
        "controls": True,
        "cpu_main": True,
    },

    {
        "suffix": "ETF_FE_CLUSTER_DATE",
        "covariance": "CLUSTER_DATE",
        "etf_fe": True,
        "month_fe": False,
        "controls": True,
        "cpu_main": True,
    },

    {
        "suffix": "ETF_FE_CLUSTER_ETF",
        "covariance": "CLUSTER_ETF",
        "etf_fe": True,
        "month_fe": False,
        "controls": True,
        "cpu_main": True,
    },

    {
        "suffix": "ETF_FE_TWO_WAY_CLUSTER",
        "covariance": "TWO_WAY_CLUSTER",
        "etf_fe": True,
        "month_fe": False,
        "controls": True,
        "cpu_main": True,
    },

    {
        "suffix": "ETF_AND_MONTH_FE_CLUSTER_ETF",
        "covariance": "CLUSTER_ETF",
        "etf_fe": True,
        "month_fe": True,
        "controls": False,
        "cpu_main": False,
    },
]


# ============================================================
# 7. YARDIMCI FONKSİYONLAR
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
            + "\n".join(
                missing
            )
        )


def significance_stars(
    p_value,
) -> str:
    """
    Akademik anlamlılık yıldızları.
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


def expected_sign_matches(
    coefficient: float,
    expected_sign: str,
) -> int:
    """
    Katsayı işaretinin teorik beklentiyle uyumunu kontrol eder.
    """

    if pd.isna(coefficient):
        return 0

    if expected_sign == "NEGATIVE":
        return int(
            coefficient < 0
        )

    if expected_sign == "POSITIVE_OR_LESS_NEGATIVE":
        return int(
            coefficient > 0
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
    """
    Nihai ekonometrik paneli okur ve temizler.
    """

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

    required_columns = [
        "DATE",
        "DATE_GROUP",
        "ETF_ID",
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
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_CHANGE",
    ]

    for channel in CHANNELS:

        for quality in [
            "MODERATE",
            "MAIN",
        ]:

            channel_column = (
                f"{channel}_{quality}"
            )

            interaction_column = (
                f"CPU_Z_X_{channel_column}"
            )

            valid_column = (
                f"VALID_{channel}_{quality}_ROW"
            )

            if channel_column in panel.columns:
                numeric_columns.append(
                    channel_column
                )

            if interaction_column in panel.columns:
                numeric_columns.append(
                    interaction_column
                )

            if valid_column in panel.columns:
                numeric_columns.append(
                    valid_column
                )

    if "PCA_COMPONENT_1" in panel.columns:

        numeric_columns.extend(
            [
                "PCA_COMPONENT_1",
                "CPU_Z_X_PCA_COMPONENT_1",
                "VALID_PCA_MAIN_ROW",
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
            "Panelde duplicate ETF-month satırı bulundu: "
            f"{duplicate_rows}"
        )

    return panel


# ============================================================
# 9. KANAL VE ÖRNEKLEM TANIMLARI
# ============================================================

def resolve_channel_columns(
    panel: pd.DataFrame,
    channel: str,
    quality: str,
) -> dict:
    """
    Kanal, interaction ve valid-row sütunlarını belirler.
    """

    if channel == "PCA_COMPONENT_1":

        return {
            "channel_column": (
                "PCA_COMPONENT_1"
            ),
            "interaction_column": (
                "CPU_Z_X_PCA_COMPONENT_1"
            ),
            "valid_column": (
                "VALID_PCA_MAIN_ROW"
            ),
            "quality": "MAIN",
        }

    channel_column = (
        f"{channel}_{quality}"
    )

    interaction_column = (
        f"CPU_Z_X_{channel_column}"
    )

    valid_column = (
        f"VALID_{channel}_{quality}_ROW"
    )

    require_columns(
        panel,
        [
            channel_column,
            interaction_column,
            valid_column,
        ],
        f"{channel} {quality}",
    )

    return {
        "channel_column": channel_column,
        "interaction_column": interaction_column,
        "valid_column": valid_column,
        "quality": quality,
    }


def prepare_channel_sample(
    panel: pd.DataFrame,
    channel: str,
    quality: str,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Kanal bazında ekonometrik örneklemi hazırlar.
    """

    columns = resolve_channel_columns(
        panel=panel,
        channel=channel,
        quality=quality,
    )

    sample = panel.loc[
        panel[
            columns[
                "valid_column"
            ]
        ]
        == 1
    ].copy()

    required_complete_case = [
        "ETF_RETURN",
        "CPU_Z",
        columns[
            "interaction_column"
        ],
    ] + TIME_CONTROLS

    sample = sample.dropna(
        subset=required_complete_case
    ).copy()

    sample = sample.sort_values(
        [
            "ETF_ID",
            "DATE",
        ]
    )

    return (
        sample,
        columns,
    )


# ============================================================
# 10. MODEL FORMÜLÜ
# ============================================================

def build_formula(
    interaction_variable: str,
    include_etf_fe: bool,
    include_month_fe: bool,
    include_controls: bool,
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

    if include_controls:

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


# ============================================================
# 11. MODEL TAHMİNİ
# ============================================================

def fit_model(
    sample: pd.DataFrame,
    formula: str,
    covariance_type: str,
):
    """
    OLS modelini tahmin eder ve istenen standart hata düzeltmesini
    uygular.
    """

    base_result = smf.ols(
        formula=formula,
        data=sample,
    ).fit()

    custom_covariance = None

    if covariance_type == "HC3":

        robust_result = (
            base_result
            .get_robustcov_results(
                cov_type="HC3"
            )
        )

    elif covariance_type == "CLUSTER_DATE":

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

        covariance_results = (
            cov_cluster_2groups(
                base_result,
                etf_groups,
                date_groups,
            )
        )

        custom_covariance = (
            covariance_results[0]
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
# 12. SONUÇLARI ÇIKAR
# ============================================================

def extract_results(
    base_result,
    robust_result,
    custom_covariance,
    sample: pd.DataFrame,
    model_name: str,
    channel: str,
    quality: str,
    channel_role: str,
    expected_sign: str,
    interaction_variable: str,
    covariance_type: str,
    formula: str,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Katsayı ve model diagnostics tablolarını üretir.
    """

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

                "CHANNEL": channel,

                "CHANNEL_ROLE": (
                    channel_role
                ),

                "QUALITY": quality,

                "COVARIANCE": (
                    covariance_type
                ),

                "PARAMETER": parameter,

                "COEFFICIENT": (
                    coefficient
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

                "EXPECTED_SIGN": (
                    expected_sign
                ),

                "EXPECTED_SIGN_MATCH": (
                    expected_sign_matches(
                        coefficient=coefficient,
                        expected_sign=expected_sign,
                    )
                    if parameter
                    == interaction_variable
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

        "CHANNEL": channel,

        "CHANNEL_ROLE": channel_role,

        "QUALITY": quality,

        "COVARIANCE": covariance_type,

        "FORMULA": formula,

        "TARGET_INTERACTION": (
            interaction_variable
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
# 13. TÜM KANAL MODELLERİ
# ============================================================

def estimate_all_models(
    panel: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Main ve moderate kanal modellerini tahmin eder.
    """

    result_frames = []

    diagnostic_rows = []

    sample_rows = []

    for channel, settings in (
        CHANNELS.items()
    ):

        if channel == "PCA_COMPONENT_1":

            qualities = [
                "MAIN",
            ]

        else:

            qualities = [
                "MAIN",
                "MODERATE",
            ]

        for quality in qualities:

            try:

                (
                    sample,
                    resolved_columns,
                ) = prepare_channel_sample(
                    panel=panel,
                    channel=channel,
                    quality=quality,
                )

            except KeyError as error:

                print(
                    f"  Atlandı: {channel} {quality}"
                )

                print(
                    f"  Neden: {error}"
                )

                continue

            interaction_variable = (
                resolved_columns[
                    "interaction_column"
                ]
            )

            n_etfs = int(
                sample[
                    "ETF_ID"
                ].nunique()
            )

            n_rows = int(
                len(
                    sample
                )
            )

            sample_rows.append(
                {
                    "CHANNEL": channel,

                    "CHANNEL_ROLE": settings[
                        "role"
                    ],

                    "QUALITY": quality,

                    "N_ROWS": n_rows,

                    "N_ETFS": n_etfs,

                    "N_MONTHS": int(
                        sample[
                            "DATE_GROUP"
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

                    "MEAN_RETURN": float(
                        sample[
                            "ETF_RETURN"
                        ].mean()
                    ),

                    "STD_RETURN": float(
                        sample[
                            "ETF_RETURN"
                        ].std(
                            ddof=1
                        )
                    ),
                }
            )

            # Çok küçük örneklemler tahmin edilmez.
            if (
                n_etfs < 5
                or n_rows < 100
            ):

                print(
                    f"  Tahmin edilmedi: "
                    f"{channel} {quality} | "
                    f"ETF={n_etfs}, rows={n_rows}"
                )

                continue

            for specification in (
                MODEL_SPECIFICATIONS
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

                formula = build_formula(
                    interaction_variable=(
                        interaction_variable
                    ),
                    include_etf_fe=(
                        specification[
                            "etf_fe"
                        ]
                    ),
                    include_month_fe=(
                        specification[
                            "month_fe"
                        ]
                    ),
                    include_controls=(
                        specification[
                            "controls"
                        ]
                    ),
                    include_cpu_main=(
                        specification[
                            "cpu_main"
                        ]
                    ),
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
                    quality=quality,
                    channel_role=settings[
                        "role"
                    ],
                    expected_sign=settings[
                        "expected_sign"
                    ],
                    interaction_variable=(
                        interaction_variable
                    ),
                    covariance_type=(
                        specification[
                            "covariance"
                        ]
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
# 14. ANA KATSAYILARI SEÇ
# ============================================================

def build_key_results(
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her modelde CPU ana etkisi ve hedef interaction katsayısını seçer.
    """

    if all_results.empty:
        return pd.DataFrame()

    interaction_mask = (
        all_results[
            "PARAMETER"
        ].str.startswith(
            "CPU_Z_X_",
            na=False,
        )
    )

    cpu_mask = (
        all_results[
            "PARAMETER"
        ]
        == "CPU_Z"
    )

    key_results = all_results.loc[
        interaction_mask
        | cpu_mask
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
# 15. İŞARET VE ANLAMLILIK TUTARLILIĞI
# ============================================================

def build_sign_consistency(
    key_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kanal bazında işaret ve anlamlılık tutarlılığını özetler.
    """

    interaction_results = key_results.loc[
        key_results[
            "PARAMETER"
        ].str.startswith(
            "CPU_Z_X_",
            na=False,
        )
    ].copy()

    rows = []

    for (
        channel,
        quality,
    ), group in interaction_results.groupby(
        [
            "CHANNEL",
            "QUALITY",
        ],
        dropna=False,
    ):

        coefficients = group[
            "COEFFICIENT"
        ]

        negative_count = int(
            (
                coefficients < 0
            ).sum()
        )

        positive_count = int(
            (
                coefficients > 0
            ).sum()
        )

        significant_10 = int(
            (
                group[
                    "P_VALUE"
                ] < 0.10
            ).sum()
        )

        significant_05 = int(
            (
                group[
                    "P_VALUE"
                ] < 0.05
            ).sum()
        )

        expected_matches = int(
            safe_numeric(
                group[
                    "EXPECTED_SIGN_MATCH"
                ]
            )
            .fillna(0)
            .sum()
        )

        rows.append(
            {
                "CHANNEL": channel,

                "QUALITY": quality,

                "N_MODELS": len(
                    group
                ),

                "NEGATIVE_COEFFICIENT_MODELS": (
                    negative_count
                ),

                "POSITIVE_COEFFICIENT_MODELS": (
                    positive_count
                ),

                "EXPECTED_SIGN_MATCH_MODELS": (
                    expected_matches
                ),

                "SIGNIFICANT_AT_10PCT_MODELS": (
                    significant_10
                ),

                "SIGNIFICANT_AT_5PCT_MODELS": (
                    significant_05
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
# 16. VIF DIAGNOSTICS
# ============================================================

def build_vif_diagnostics(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her main-quality kanalın temel regresyonu için VIF hesaplar.
    """

    rows = []

    for channel in CHANNELS:

        try:

            (
                sample,
                resolved_columns,
            ) = prepare_channel_sample(
                panel=panel,
                channel=channel,
                quality="MAIN",
            )

        except KeyError:

            continue

        if (
            sample[
                "ETF_ID"
            ].nunique()
            < 5
        ):

            continue

        interaction_variable = (
            resolved_columns[
                "interaction_column"
            ]
        )

        variables = [
            "CPU_Z",
            interaction_variable,
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
# 17. MODEL SÖZLÜĞÜ
# ============================================================

def build_model_dictionary() -> pd.DataFrame:
    """
    Ekonometrik model sözlüğünü üretir.
    """

    rows = []

    for channel, settings in (
        CHANNELS.items()
    ):

        rows.append(
            {
                "CHANNEL": channel,

                "ROLE": settings[
                    "role"
                ],

                "DEPENDENT_VARIABLE": (
                    "ETF_RETURN"
                ),

                "KEY_REGRESSOR": (
                    f"CPU_Z × {channel}"
                ),

                "EXPECTED_SIGN": settings[
                    "expected_sign"
                ],

                "THEORY": settings[
                    "theory"
                ],

                "BASE_CONTROLS": (
                    "Market, energy, Treasury and VIX change"
                ),

                "FIXED_EFFECTS": (
                    "ETF fixed effects; alternative ETF and month "
                    "fixed effects"
                ),

                "STANDARD_ERRORS": (
                    "HC3, date-clustered, ETF-clustered and "
                    "two-way clustered"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. VALIDATION
# ============================================================

def build_validation(
    all_results: pd.DataFrame,
    key_results: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    sample_diagnostics: pd.DataFrame,
    vif_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ekonometrik çıktıların mekanik doğrulamasını yapar.
    """

    interaction_results = key_results.loc[
        key_results[
            "PARAMETER"
        ].str.startswith(
            "CPU_Z_X_",
            na=False,
        )
    ].copy()

    primary_channels_estimated = (
        interaction_results.loc[
            interaction_results[
                "CHANNEL_ROLE"
            ]
            == "PRIMARY",
            "CHANNEL",
        ]
        .nunique()
    )

    main_two_way_models = int(
        (
            model_diagnostics[
                "COVARIANCE"
            ]
            == "TWO_WAY_CLUSTER"
        ).sum()
    )

    month_fe_models = int(
        model_diagnostics[
            "MODEL"
        ]
        .str.contains(
            "ETF_AND_MONTH_FE",
            regex=False,
            na=False,
        )
        .sum()
    )

    finite_coefficients = int(
        (
            np.isfinite(
                all_results[
                    "COEFFICIENT"
                ]
            )
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
                "PRIMARY_CHANNELS_ESTIMATED"
            ),
            "VALUE": (
                primary_channels_estimated
            ),
            "PASS": int(
                primary_channels_estimated
                == 4
            ),
        },

        {
            "CHECK": (
                "TWO_WAY_CLUSTER_MODELS"
            ),
            "VALUE": (
                main_two_way_models
            ),
            "PASS": int(
                main_two_way_models
                > 0
            ),
        },

        {
            "CHECK": (
                "ETF_AND_MONTH_FE_MODELS"
            ),
            "VALUE": month_fe_models,
            "PASS": int(
                month_fe_models
                > 0
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
                ) > 0
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
                ) > 0
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
                ) > 0
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
# 19. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 92)
    print("26 - FINAL PORTFOLIO ARCHITECTURE CHANNEL MODELS")
    print("=" * 92)

    # --------------------------------------------------------
    # 1. Panel
    # --------------------------------------------------------

    print(
        "\n1/8 - Nihai ekonometrik panel okunuyor..."
    )

    panel = prepare_panel(
        PANEL_FILE
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

    # --------------------------------------------------------
    # 2. Modeller
    # --------------------------------------------------------

    print(
        "\n2/8 - Main ve moderate kanal modelleri tahmin ediliyor..."
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
            "Hiçbir model tahmin edilemedi."
        )

    # --------------------------------------------------------
    # 3. Ana katsayılar
    # --------------------------------------------------------

    print(
        "\n3/8 - Ana interaction katsayıları hazırlanıyor..."
    )

    key_results = build_key_results(
        all_results
    )

    sign_consistency = (
        build_sign_consistency(
            key_results
        )
    )

    primary_results = (
        key_results.loc[
            (
                key_results[
                    "CHANNEL_ROLE"
                ]
                == "PRIMARY"
            )
            & (
                key_results[
                    "PARAMETER"
                ].str.startswith(
                    "CPU_Z_X_",
                    na=False,
                )
            )
        ]
        .copy()
    )

    robustness_results = (
        key_results.loc[
            (
                key_results[
                    "CHANNEL_ROLE"
                ]
                != "PRIMARY"
            )
            & (
                key_results[
                    "PARAMETER"
                ].str.startswith(
                    "CPU_Z_X_",
                    na=False,
                )
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # 4. VIF
    # --------------------------------------------------------

    print(
        "\n4/8 - VIF diagnostics hazırlanıyor..."
    )

    vif_diagnostics = (
        build_vif_diagnostics(
            panel
        )
    )

    # --------------------------------------------------------
    # 5. Model sözlüğü
    # --------------------------------------------------------

    print(
        "\n5/8 - Model sözlüğü hazırlanıyor..."
    )

    model_dictionary = (
        build_model_dictionary()
    )

    # --------------------------------------------------------
    # 6. Validation
    # --------------------------------------------------------

    print(
        "\n6/8 - Validation kontrolleri çalıştırılıyor..."
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
        vif_diagnostics=(
            vif_diagnostics
        ),
    )

    # --------------------------------------------------------
    # 7. Kaydet
    # --------------------------------------------------------

    print(
        "\n7/8 - Sonuç dosyaları kaydediliyor..."
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
        PRIMARY_CHANNEL_RESULTS_FILE,
        index=False,
    )

    robustness_results.to_csv(
        ROBUSTNESS_RESULTS_FILE,
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

    # --------------------------------------------------------
    # 8. Ekran sonuçları
    # --------------------------------------------------------

    print(
        "\n8/8 - Ana sonuç tabloları hazırlanıyor..."
    )

    print(
        "\nFINAL CHANNEL MODELS HAZIR"
    )

    print("=" * 92)

    print(
        "\nMain-quality primary-channel interaction sonuçları:"
    )

    main_primary = primary_results.loc[
        primary_results[
            "QUALITY"
        ]
        == "MAIN"
    ].copy()

    print(
        main_primary[
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
                "N_MONTHS",
                "R_SQUARED",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nModerate-quality primary-channel interaction sonuçları:"
    )

    moderate_primary = primary_results.loc[
        primary_results[
            "QUALITY"
        ]
        == "MODERATE"
    ].copy()

    print(
        moderate_primary[
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
                "R_SQUARED",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nComposite ve PCA robustness sonuçları:"
    )

    if robustness_results.empty:

        print(
            "Robustness interaction sonucu üretilemedi."
        )

    else:

        print(
            robustness_results[
                [
                    "CHANNEL",
                    "QUALITY",
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
        "\nİşaret ve anlamlılık tutarlılığı:"
    )

    print(
        sign_consistency.to_string(
            index=False
        )
    )

    print(
        "\nMain-quality VIF diagnostics:"
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
                "CHANNEL",
                "QUALITY",
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
        "Her portfolio architecture kanalı ayrı regresyonda "
        "tahmin edilmiştir. Bu yaklaşım yüksek kanal korelasyonunun "
        "ve composite mekanik örtüşmesinin ana sonuçları bozmasını "
        "engeller."
    )

    print(
        "Pooled HC3 sonuçları tanımlayıcıdır. Ana çıkarımlar ETF "
        "fixed effects ve clustered standart hatalara dayalı "
        "modellerden yapılmalıdır."
    )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        PRIMARY_CHANNEL_RESULTS_FILE
    )

    print(
        ROBUSTNESS_RESULTS_FILE
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