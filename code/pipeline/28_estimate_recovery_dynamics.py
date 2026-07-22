from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


# ============================================================
# 28_estimate_recovery_dynamics.py
#
# AMAÇ
# ----
# RQ4:
#
# Do portfolios with different embedded firm characteristics
# exhibit different recovery dynamics after climate-policy shocks?
#
# TEMEL MODEL
# -----------
# Forward_Return_i,t+h =
#       ETF fixed effects
#       + time fixed effects
#       + Shock_t × Architecture_i
#       + error_i,t+h
#
# ARCHITECTURE DEĞİŞKENLERİ
# -------------------------
# 1. Internal Financial Capacity
# 2. External Financing Dependence
# 3. Growth-Duration Exposure
# 4. Portfolio Concentration
#
# ROBUSTNESS
# ----------
# 5. Financial Architecture Risk
# 6. Extended Architecture Risk
# 7. PCA Component 1
#
# ŞOK TANIMLARI
# -------------
# 1. CPU_SHOCK
# 2. EXTREME_CPU_REGIME
# 3. CPU_AND_VIX_STRESS
#
# İLERİ GETİRİLER
# ---------------
# 1 ay  : ETF_RETURN_LEAD1
# 3 ay  : CUM_RETURN_LEAD3
# 6 ay  : CUM_RETURN_LEAD6
# 12 ay : CUM_RETURN_LEAD12
#
# ANA YORUM
# ---------
# Negatif Shock × Architecture katsayısı:
# Yüksek architecture exposure'a sahip ETF'lerin şok sonrasında
# daha zayıf toparlandığını gösterir.
#
# Pozitif katsayı:
# Yüksek architecture exposure'a sahip ETF'lerin şok sonrasında
# daha güçlü toparlandığını gösterir.
#
# METODOLOJİK NOT
# ----------------
# İleri kümülatif getiriler örtüşen gözlemler içerir. Bu nedenle
# ana çıkarım ETF ve tarih bazında iki yönlü kümelenmiş standart
# hatalara dayanmalıdır.
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
    / "28_recovery_all_model_results.csv"
)

KEY_RESULTS_FILE = (
    OUTPUT_DIR
    / "28_recovery_key_coefficients.csv"
)

PRIMARY_RESULTS_FILE = (
    OUTPUT_DIR
    / "28_primary_channel_recovery_results.csv"
)

ROBUSTNESS_RESULTS_FILE = (
    OUTPUT_DIR
    / "28_composite_pca_recovery_results.csv"
)

HORIZON_PROFILE_FILE = (
    OUTPUT_DIR
    / "28_recovery_horizon_profiles.csv"
)

MODEL_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "28_recovery_model_diagnostics.csv"
)

SAMPLE_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "28_recovery_sample_diagnostics.csv"
)

SIGN_CONSISTENCY_FILE = (
    OUTPUT_DIR
    / "28_recovery_sign_consistency.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "28_recovery_validation.csv"
)

MODEL_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "28_recovery_model_dictionary.csv"
)


# ============================================================
# 4. KONTROL DEĞİŞKENLERİ
# ============================================================

TIME_CONTROLS = [
    "MARKET_RETURN",
    "ENERGY_RETURN",
    "TREASURY_RETURN",
    "VIX_CHANGE",
]


# ============================================================
# 5. GETİRİ UFUKLARI
# ============================================================

HORIZONS = {
    1: {
        "outcome": "ETF_RETURN_LEAD1",
        "label": "ONE_MONTH",
    },

    3: {
        "outcome": "CUM_RETURN_LEAD3",
        "label": "THREE_MONTH",
    },

    6: {
        "outcome": "CUM_RETURN_LEAD6",
        "label": "SIX_MONTH",
    },

    12: {
        "outcome": "CUM_RETURN_LEAD12",
        "label": "TWELVE_MONTH",
    },
}


# ============================================================
# 6. ŞOK TANIMLARI
# ============================================================

SHOCKS = {
    "CPU_SHOCK": {
        "type": "CONTINUOUS",
        "primary_or_robustness": "PRIMARY",
        "description": (
            "Continuous innovation in climate-policy uncertainty"
        ),
    },

    "EXTREME_CPU_REGIME": {
        "type": "BINARY",
        "primary_or_robustness": "ROBUSTNESS",
        "description": (
            "Indicator for extreme climate-policy uncertainty months"
        ),
    },

    "CPU_AND_VIX_STRESS": {
        "type": "BINARY",
        "primary_or_robustness": "ROBUSTNESS",
        "description": (
            "Indicator for simultaneous high CPU and high market stress"
        ),
    },
}


# ============================================================
# 7. PORTFOLIO ARCHITECTURE KANALLARI
# ============================================================

CHANNELS = {
    "INTERNAL_FINANCIAL_CAPACITY": {
        "role": "PRIMARY",
        "expected_recovery_sign": "POSITIVE",
        "theory": (
            "Internal profitability and liquidity buffers should "
            "support stronger post-shock recovery."
        ),
    },

    "EXTERNAL_FINANCING_DEPENDENCE": {
        "role": "PRIMARY",
        "expected_recovery_sign": "NEGATIVE",
        "theory": (
            "Dependence on external capital should delay recovery "
            "after uncertainty shocks."
        ),
    },

    "GROWTH_DURATION_EXPOSURE_FINAL": {
        "role": "PRIMARY",
        "expected_recovery_sign": "NEGATIVE_OR_DELAYED",
        "theory": (
            "Long-duration investment and innovation exposure may "
            "produce deeper or slower post-shock adjustment."
        ),
    },

    "PORTFOLIO_CONCENTRATION_FINAL": {
        "role": "PRIMARY",
        "expected_recovery_sign": "NEGATIVE",
        "theory": (
            "Concentrated portfolios may recover more slowly because "
            "losses remain tied to dominant holdings."
        ),
    },

    "FINANCIAL_ARCHITECTURE_RISK_FINAL": {
        "role": "COMPOSITE_ROBUSTNESS",
        "expected_recovery_sign": "NEGATIVE",
        "theory": (
            "Higher embedded financial architecture risk should "
            "weaken post-shock recovery."
        ),
    },

    "EXTENDED_ARCHITECTURE_RISK_FINAL": {
        "role": "COMPOSITE_ROBUSTNESS",
        "expected_recovery_sign": "NEGATIVE",
        "theory": (
            "Financial vulnerability combined with concentration "
            "should weaken recovery."
        ),
    },

    "PCA_COMPONENT_1": {
        "role": "PCA_ROBUSTNESS",
        "expected_recovery_sign": "DATA_DRIVEN",
        "theory": (
            "The first principal component captures a common "
            "data-driven portfolio architecture dimension."
        ),
    },
}


# ============================================================
# 8. MODEL SPESİFİKASYONLARI
# ============================================================

MODEL_SPECIFICATIONS = [
    {
        "suffix": "ETF_FE_CLUSTER_DATE",
        "covariance": "CLUSTER_DATE",
        "month_fe": False,
        "controls": True,
    },

    {
        "suffix": "ETF_FE_CLUSTER_ETF",
        "covariance": "CLUSTER_ETF",
        "month_fe": False,
        "controls": True,
    },

    {
        "suffix": "ETF_FE_TWO_WAY_CLUSTER",
        "covariance": "TWO_WAY_CLUSTER",
        "month_fe": False,
        "controls": True,
    },

    {
        "suffix": "ETF_MONTH_FE_CLUSTER_ETF",
        "covariance": "CLUSTER_ETF",
        "month_fe": True,
        "controls": False,
    },
]


# ============================================================
# 9. MINIMUM ÖRNEKLEM EŞİKLERİ
# ============================================================

MIN_ETFS = 5

MIN_OBSERVATIONS = 100


# ============================================================
# 10. YARDIMCI FONKSİYONLAR
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

    if expected_sign == "NEGATIVE_OR_DELAYED":

        return int(
            coefficient < 0
        )

    if expected_sign == "DATA_DRIVEN":

        return 1

    return 0


# ============================================================
# 11. PANELİ HAZIRLA
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

    required_columns = [
        "DATE",
        "DATE_GROUP",
        "ETF_ID",
        "ETF_RETURN",
        "ETF_RETURN_LEAD1",
        "CUM_RETURN_LEAD3",
        "CUM_RETURN_LEAD6",
        "CUM_RETURN_LEAD12",
        "CPU_SHOCK",
        "EXTREME_CPU_REGIME",
        "CPU_AND_VIX_STRESS",
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
        "ETF_RETURN_LEAD1",
        "CUM_RETURN_LEAD3",
        "CUM_RETURN_LEAD6",
        "CUM_RETURN_LEAD12",
        "CPU_SHOCK",
        "EXTREME_CPU_REGIME",
        "CPU_AND_VIX_STRESS",
        "MARKET_RETURN",
        "ENERGY_RETURN",
        "TREASURY_RETURN",
        "VIX_CHANGE",
    ]

    for channel in CHANNELS:

        if channel == "PCA_COMPONENT_1":

            candidate_columns = [
                "PCA_COMPONENT_1",
                "VALID_PCA_MAIN_ROW",
            ]

        else:

            candidate_columns = [
                f"{channel}_MAIN",
                f"{channel}_MODERATE",
                f"VALID_{channel}_MAIN_ROW",
                f"VALID_{channel}_MODERATE_ROW",
            ]

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
# 12. KANAL SÜTUNLARINI BELİRLE
# ============================================================

def resolve_channel_columns(
    panel: pd.DataFrame,
    channel: str,
    quality: str,
) -> dict:

    if channel == "PCA_COMPONENT_1":

        architecture_column = (
            "PCA_COMPONENT_1"
        )

        valid_column = (
            "VALID_PCA_MAIN_ROW"
        )

        resolved_quality = "MAIN"

    else:

        architecture_column = (
            f"{channel}_{quality}"
        )

        valid_column = (
            f"VALID_{channel}_{quality}_ROW"
        )

        resolved_quality = quality

    require_columns(
        panel,
        [
            architecture_column,
            valid_column,
        ],
        f"{channel} {quality}",
    )

    return {
        "architecture_column": (
            architecture_column
        ),

        "valid_column": (
            valid_column
        ),

        "quality": (
            resolved_quality
        ),
    }


# ============================================================
# 13. ŞOK ETKİLEŞİMLERİNİ OLUŞTUR
# ============================================================

def add_recovery_interactions(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    result = panel.copy()

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

                resolved = (
                    resolve_channel_columns(
                        result,
                        channel,
                        quality,
                    )
                )

            except KeyError:

                continue

            architecture_column = (
                resolved[
                    "architecture_column"
                ]
            )

            for shock in SHOCKS:

                interaction_name = (
                    f"{shock}_X_"
                    f"{architecture_column}"
                )

                result[
                    interaction_name
                ] = (
                    result[
                        shock
                    ]
                    * result[
                        architecture_column
                    ]
                )

    return result


# ============================================================
# 14. MODEL ÖRNEKLEMİNİ HAZIRLA
# ============================================================

def prepare_model_sample(
    panel: pd.DataFrame,
    channel: str,
    quality: str,
    shock: str,
    outcome: str,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    resolved = resolve_channel_columns(
        panel=panel,
        channel=channel,
        quality=quality,
    )

    architecture_column = (
        resolved[
            "architecture_column"
        ]
    )

    valid_column = (
        resolved[
            "valid_column"
        ]
    )

    interaction_column = (
        f"{shock}_X_"
        f"{architecture_column}"
    )

    require_columns(
        panel,
        [
            outcome,
            shock,
            architecture_column,
            interaction_column,
            valid_column,
        ],
        (
            f"{channel} {quality} "
            f"{shock} {outcome}"
        ),
    )

    sample = panel.loc[
        panel[
            valid_column
        ]
        == 1
    ].copy()

    required_complete_case = [
        outcome,
        shock,
        interaction_column,
        "ETF_ID",
        "DATE_GROUP",
    ] + TIME_CONTROLS

    sample = sample.dropna(
        subset=required_complete_case
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
        {
            **resolved,
            "interaction_column": (
                interaction_column
            ),
        },
    )


# ============================================================
# 15. MODEL FORMÜLÜ
# ============================================================

def build_formula(
    outcome: str,
    shock: str,
    interaction_column: str,
    month_fe: bool,
    include_controls: bool,
) -> str:

    regressors = [
        interaction_column,
        "C(ETF_ID)",
    ]

    if month_fe:

        regressors.append(
            "C(DATE_GROUP)"
        )

    else:

        regressors.insert(
            0,
            shock,
        )

        if include_controls:

            regressors.extend(
                TIME_CONTROLS
            )

    return (
        f"{outcome} ~ "
        + " + ".join(
            regressors
        )
    )


# ============================================================
# 16. MODEL TAHMİNİ
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
# 17. SONUÇLARI ÇIKAR
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
    shock: str,
    horizon: int,
    outcome: str,
    covariance_type: str,
    target_parameter: str,
    expected_sign: str,
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

                "CHANNEL": channel,

                "CHANNEL_ROLE": channel_role,

                "QUALITY": quality,

                "SHOCK": shock,

                "HORIZON_MONTHS": horizon,

                "OUTCOME": outcome,

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

                "N_POSITIVE_SHOCK_MONTHS": int(
                    sample.loc[
                        sample[
                            shock
                        ]
                        > 0,
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

        "CHANNEL": channel,

        "CHANNEL_ROLE": channel_role,

        "QUALITY": quality,

        "SHOCK": shock,

        "HORIZON_MONTHS": horizon,

        "OUTCOME": outcome,

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

        "N_POSITIVE_SHOCK_MONTHS": int(
            sample.loc[
                sample[
                    shock
                ]
                > 0,
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
# 18. TÜM RECOVERY MODELLERİ
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

    for channel, channel_settings in (
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

            for shock in SHOCKS:

                for horizon, horizon_settings in (
                    HORIZONS.items()
                ):

                    outcome = horizon_settings[
                        "outcome"
                    ]

                    try:

                        (
                            sample,
                            resolved,
                        ) = prepare_model_sample(
                            panel=panel,
                            channel=channel,
                            quality=quality,
                            shock=shock,
                            outcome=outcome,
                        )

                    except KeyError as error:

                        print(
                            f"  Atlandı: "
                            f"{channel} {quality} "
                            f"{shock} {horizon}M"
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

                            "CHANNEL_ROLE": (
                                channel_settings[
                                    "role"
                                ]
                            ),

                            "QUALITY": quality,

                            "SHOCK": shock,

                            "HORIZON_MONTHS": (
                                horizon
                            ),

                            "OUTCOME": outcome,

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

                            "N_POSITIVE_SHOCK_MONTHS": int(
                                sample.loc[
                                    sample[
                                        shock
                                    ]
                                    > 0,
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

                            "MEAN_FORWARD_RETURN": float(
                                sample[
                                    outcome
                                ].mean()
                            ),

                            "STD_FORWARD_RETURN": float(
                                sample[
                                    outcome
                                ].std(
                                    ddof=1
                                )
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
                            f"{shock} {horizon}M | "
                            f"ETF={n_etfs}, rows={n_rows}"
                        )

                        continue

                    target_parameter = (
                        resolved[
                            "interaction_column"
                        ]
                    )

                    for specification in (
                        MODEL_SPECIFICATIONS
                    ):

                        model_name = (
                            f"{channel}_"
                            f"{quality}_"
                            f"{shock}_"
                            f"{horizon}M_"
                            f"{specification['suffix']}"
                        )

                        print(
                            f"  Tahmin ediliyor: "
                            f"{model_name}"
                        )

                        formula = build_formula(
                            outcome=outcome,
                            shock=shock,
                            interaction_column=(
                                target_parameter
                            ),
                            month_fe=(
                                specification[
                                    "month_fe"
                                ]
                            ),
                            include_controls=(
                                specification[
                                    "controls"
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
                            channel_role=(
                                channel_settings[
                                    "role"
                                ]
                            ),
                            quality=quality,
                            shock=shock,
                            horizon=horizon,
                            outcome=outcome,
                            covariance_type=(
                                specification[
                                    "covariance"
                                ]
                            ),
                            target_parameter=(
                                target_parameter
                            ),
                            expected_sign=(
                                channel_settings[
                                    "expected_recovery_sign"
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
# 19. ANA KATSAYILARI SEÇ
# ============================================================

def build_key_results(
    all_results: pd.DataFrame,
) -> pd.DataFrame:

    if all_results.empty:

        return pd.DataFrame()

    key_results = all_results.loc[
        all_results[
            "PARAMETER"
        ]
        == all_results[
            "TARGET_PARAMETER"
        ]
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
# 20. HORIZON PROFILE
# ============================================================

def build_horizon_profiles(
    key_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ana iki-yönlü clustered sonuçlardan horizon profili üretir.
    """

    profile = key_results.loc[
        (
            key_results[
                "QUALITY"
            ]
            == "MAIN"
        )
        & (
            key_results[
                "COVARIANCE"
            ]
            == "TWO_WAY_CLUSTER"
        )
    ].copy()

    return profile.sort_values(
        by=[
            "CHANNEL_ROLE",
            "CHANNEL",
            "SHOCK",
            "HORIZON_MONTHS",
        ]
    )


# ============================================================
# 21. İŞARET TUTARLILIĞI
# ============================================================

def build_sign_consistency(
    key_results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        channel,
        quality,
        shock,
        horizon,
    ), group in key_results.groupby(
        [
            "CHANNEL",
            "QUALITY",
            "SHOCK",
            "HORIZON_MONTHS",
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

                "SHOCK": shock,

                "HORIZON_MONTHS": horizon,

                "N_MODELS": int(
                    len(group)
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
# 22. MODEL SÖZLÜĞÜ
# ============================================================

def build_model_dictionary() -> pd.DataFrame:

    rows = []

    for shock, shock_settings in (
        SHOCKS.items()
    ):

        for horizon, horizon_settings in (
            HORIZONS.items()
        ):

            rows.append(
                {
                    "SHOCK": shock,

                    "SHOCK_TYPE": (
                        shock_settings[
                            "type"
                        ]
                    ),

                    "HORIZON_MONTHS": (
                        horizon
                    ),

                    "DEPENDENT_VARIABLE": (
                        horizon_settings[
                            "outcome"
                        ]
                    ),

                    "KEY_PARAMETER": (
                        f"{shock} × Portfolio Architecture"
                    ),

                    "FIXED_EFFECTS": (
                        "ETF fixed effects; alternative ETF and "
                        "month fixed effects"
                    ),

                    "STANDARD_ERRORS": (
                        "Date-clustered, ETF-clustered and "
                        "two-way clustered"
                    ),

                    "INTERPRETATION": (
                        "Tests whether embedded portfolio architecture "
                        "predicts differential post-shock recovery."
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 23. VALIDATION
# ============================================================

def build_validation(
    all_results: pd.DataFrame,
    key_results: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    sample_diagnostics: pd.DataFrame,
    horizon_profiles: pd.DataFrame,
) -> pd.DataFrame:

    primary_channels_estimated = (
        key_results.loc[
            key_results[
                "CHANNEL_ROLE"
            ]
            == "PRIMARY",
            "CHANNEL",
        ]
        .nunique()
    )

    horizons_estimated = (
        key_results[
            "HORIZON_MONTHS"
        ]
        .nunique()
    )

    shocks_estimated = (
        key_results[
            "SHOCK"
        ]
        .nunique()
    )

    two_way_models = int(
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
            "ETF_MONTH_FE",
            regex=False,
            na=False,
        )
        .sum()
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
                )
                > 0
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
                )
                > 0
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
                "RECOVERY_HORIZONS_ESTIMATED"
            ),
            "VALUE": (
                horizons_estimated
            ),
            "PASS": int(
                horizons_estimated
                == 4
            ),
        },

        {
            "CHECK": (
                "SHOCK_DEFINITIONS_ESTIMATED"
            ),
            "VALUE": (
                shocks_estimated
            ),
            "PASS": int(
                shocks_estimated
                == 3
            ),
        },

        {
            "CHECK": (
                "TWO_WAY_CLUSTER_MODELS"
            ),
            "VALUE": two_way_models,
            "PASS": int(
                two_way_models
                > 0
            ),
        },

        {
            "CHECK": (
                "ETF_MONTH_FE_MODELS"
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
                "HORIZON_PROFILE_ROWS"
            ),
            "VALUE": len(
                horizon_profiles
            ),
            "PASS": int(
                len(
                    horizon_profiles
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
# 24. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 96)
    print("28 - POST-SHOCK RECOVERY DYNAMICS")
    print("=" * 96)

    print(
        "\n1/8 - Nihai ekonometrik panel okunuyor..."
    )

    panel = prepare_panel(
        PANEL_FILE
    )

    panel = add_recovery_interactions(
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
        "\n2/8 - Recovery modelleri tahmin ediliyor..."
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
            "Hiçbir recovery modeli tahmin edilemedi."
        )

    print(
        "\n3/8 - Ana interaction katsayıları hazırlanıyor..."
    )

    key_results = build_key_results(
        all_results
    )

    primary_results = key_results.loc[
        key_results[
            "CHANNEL_ROLE"
        ]
        == "PRIMARY"
    ].copy()

    robustness_results = key_results.loc[
        key_results[
            "CHANNEL_ROLE"
        ]
        != "PRIMARY"
    ].copy()

    print(
        "\n4/8 - Recovery horizon profilleri hazırlanıyor..."
    )

    horizon_profiles = (
        build_horizon_profiles(
            key_results
        )
    )

    sign_consistency = (
        build_sign_consistency(
            key_results
        )
    )

    print(
        "\n5/8 - Model sözlüğü hazırlanıyor..."
    )

    model_dictionary = (
        build_model_dictionary()
    )

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
        horizon_profiles=(
            horizon_profiles
        ),
    )

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
        PRIMARY_RESULTS_FILE,
        index=False,
    )

    robustness_results.to_csv(
        ROBUSTNESS_RESULTS_FILE,
        index=False,
    )

    horizon_profiles.to_csv(
        HORIZON_PROFILE_FILE,
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
        "\n8/8 - Ana sonuçlar ekrana yazdırılıyor..."
    )

    print(
        "\nRECOVERY DYNAMICS MODELS HAZIR"
    )

    print("=" * 96)

    print(
        "\nMain-quality primary-channel CPU-shock horizon profilleri:"
    )

    primary_cpu_profiles = (
        horizon_profiles.loc[
            (
                horizon_profiles[
                    "CHANNEL_ROLE"
                ]
                == "PRIMARY"
            )
            & (
                horizon_profiles[
                    "SHOCK"
                ]
                == "CPU_SHOCK"
            )
        ]
        .copy()
    )

    print(
        primary_cpu_profiles[
            [
                "CHANNEL",
                "HORIZON_MONTHS",
                "COEFFICIENT",
                "STD_ERROR",
                "P_VALUE",
                "SIGNIFICANCE",
                "EXPECTED_SIGN",
                "EXPECTED_SIGN_MATCH",
                "N_OBSERVATIONS",
                "N_ETFS",
                "N_MONTHS",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nMain-quality extreme-CPU recovery sonuçları:"
    )

    primary_extreme_profiles = (
        horizon_profiles.loc[
            (
                horizon_profiles[
                    "CHANNEL_ROLE"
                ]
                == "PRIMARY"
            )
            & (
                horizon_profiles[
                    "SHOCK"
                ]
                == "EXTREME_CPU_REGIME"
            )
        ]
        .copy()
    )

    print(
        primary_extreme_profiles[
            [
                "CHANNEL",
                "HORIZON_MONTHS",
                "COEFFICIENT",
                "STD_ERROR",
                "P_VALUE",
                "SIGNIFICANCE",
                "EXPECTED_SIGN",
                "EXPECTED_SIGN_MATCH",
                "N_OBSERVATIONS",
                "N_ETFS",
                "N_POSITIVE_SHOCK_MONTHS",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nMain-quality joint-stress recovery sonuçları:"
    )

    primary_joint_profiles = (
        horizon_profiles.loc[
            (
                horizon_profiles[
                    "CHANNEL_ROLE"
                ]
                == "PRIMARY"
            )
            & (
                horizon_profiles[
                    "SHOCK"
                ]
                == "CPU_AND_VIX_STRESS"
            )
        ]
        .copy()
    )

    print(
        primary_joint_profiles[
            [
                "CHANNEL",
                "HORIZON_MONTHS",
                "COEFFICIENT",
                "STD_ERROR",
                "P_VALUE",
                "SIGNIFICANCE",
                "EXPECTED_SIGN",
                "EXPECTED_SIGN_MATCH",
                "N_OBSERVATIONS",
                "N_ETFS",
                "N_POSITIVE_SHOCK_MONTHS",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nComposite ve PCA recovery robustness sonuçları:"
    )

    if robustness_results.empty:

        print(
            "Robustness sonucu üretilemedi."
        )

    else:

        robustness_two_way = (
            robustness_results.loc[
                (
                    robustness_results[
                        "QUALITY"
                    ]
                    == "MAIN"
                )
                & (
                    robustness_results[
                        "COVARIANCE"
                    ]
                    == "TWO_WAY_CLUSTER"
                )
            ]
            .copy()
        )

        print(
            robustness_two_way[
                [
                    "CHANNEL",
                    "SHOCK",
                    "HORIZON_MONTHS",
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
        "Ana recovery yorumları main-quality ve two-way clustered "
        "sonuçlara dayanmalıdır. Üç, altı ve on iki aylık ileri "
        "kümülatif getiriler örtüşen dönemler içerir."
    )

    print(
        "CPU_SHOCK ana sürekli şok tanımıdır. Extreme CPU ve joint "
        "CPU–VIX stress sonuçları rejim robustness testleridir."
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
        HORIZON_PROFILE_FILE
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
        SIGN_CONSISTENCY_FILE
    )

    print(
        VALIDATION_FILE
    )


if __name__ == "__main__":
    main()