"""
30_final_persistent_architecture_analysis.py
============================================

TEK VE SON EKONOMETRİK SCRIPT

Amaç
----
1. 25_final_econometric_panel.csv ile
   29_persistent_architecture_scores.csv dosyalarını ETF_ID üzerinden birleştirir.
2. Persistent architecture kanallarını standardize eder.
3. İki yönlü sabit etkili ana CPU pricing modellerini tahmin eder.
4. Sürekli CPU × VIX stress activation modellerini tahmin eder.
5. Joint CPU–VIX stress modellerini tahmin eder.
6. 0, 3, 6 ve 12 aylık recovery modellerini tahmin eder.
7. Holm ve Benjamini–Hochberg düzeltmelerini uygular.
8. Nihai tablo, özet ve grafik dosyalarını üretir.

Metodolojik yaklaşım
--------------------
- ETF ve ay sabit etkileri iki yönlü within transformation ile uygulanır.
- Ana çıkarım date-clustered standart hatalara dayanır.
- Persistent architecture üç tarihsel N-PORT snapshot'ının ortalamasıdır.
- Nedensellik değil, conditional pricing ve exposure transmission test edilir.

Çalıştırma
----------
python scripts\\30_final_persistent_architecture_analysis.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ============================================================
# 0. YOLLAR VE AYARLAR
# ============================================================

PROJECT_DIR = Path.home() / "Desktop" / "CPU_Project"
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "30_final_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "25_final_econometric_panel.csv"
PERSISTENT_FILE = OUTPUT_DIR / "29_persistent_architecture_scores.csv"

MIN_ETFS = 10

BASE_CONTROLS = [
    "MARKET_RETURN",
    "ENERGY_RETURN",
    "TREASURY_RETURN",
]

CHANNEL_MAP = {
    "INTERNAL_FINANCIAL_CAPACITY": (
        "PERSISTENT_INTERNAL_FINANCIAL_CAPACITY"
    ),
    "EXTERNAL_FINANCING_DEPENDENCE": (
        "PERSISTENT_EXTERNAL_FINANCING_DEPENDENCE"
    ),
    "GROWTH_DURATION_EXPOSURE": (
        "PERSISTENT_GROWTH_DURATION_EXPOSURE_FINAL"
    ),
    "PORTFOLIO_CONCENTRATION": (
        "PERSISTENT_PORTFOLIO_CONCENTRATION_FINAL"
    ),
    "FINANCIAL_ARCHITECTURE_RISK": (
        "PERSISTENT_FINANCIAL_ARCHITECTURE_RISK_FINAL"
    ),
    "EXTENDED_ARCHITECTURE_RISK": (
        "PERSISTENT_EXTENDED_ARCHITECTURE_RISK_FINAL"
    ),
}

RECOVERY_OUTCOMES = {
    0: "ETF_RETURN",
    3: "CUM_RETURN_LEAD3",
    6: "CUM_RETURN_LEAD6",
    12: "CUM_RETURN_LEAD12",
}


# ============================================================
# 1. YARDIMCI FONKSİYONLAR
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Gerekli dosya bulunamadı:\n{path}")

    if path.stat().st_size == 0:
        raise RuntimeError(f"Dosya boş görünüyor:\n{path}")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def two_way_demean(
    df: pd.DataFrame,
    columns: List[str],
    entity_col: str = "ETF_ID",
    time_col: str = "DATE",
) -> pd.DataFrame:
    """
    x_it - x_i. - x_.t + x_..
    """
    result = df[[entity_col, time_col] + columns].copy()

    for column in columns:
        values = safe_numeric(result[column])

        entity_mean = values.groupby(
            result[entity_col]
        ).transform("mean")

        time_mean = values.groupby(
            result[time_col]
        ).transform("mean")

        grand_mean = values.mean()

        result[column] = (
            values
            - entity_mean
            - time_mean
            + grand_mean
        )

    return result


def fit_two_way_fe_date_cluster(
    data: pd.DataFrame,
    outcome: str,
    regressors: List[str],
    focal_term: str,
) -> Dict[str, object]:
    """
    İki yönlü FE + date-clustered SE.
    """
    required = list(
    dict.fromkeys(
        [
            "ETF_ID",
            "DATE",
            outcome,
        ]
        + regressors
    )
)

    sample = data[required].copy()

    for column in [outcome] + regressors:
        sample[column] = safe_numeric(sample[column])

    sample = sample.dropna()

    n_etfs = sample["ETF_ID"].nunique()
    n_dates = sample["DATE"].nunique()

    if (
        len(sample) < 50
        or n_etfs < MIN_ETFS
        or n_dates < 12
    ):
        return {
            "STATUS": "INSUFFICIENT_SAMPLE",
            "N_OBS": len(sample),
            "N_ETFS": n_etfs,
            "N_DATES": n_dates,
        }

    transformed = two_way_demean(
        sample,
        columns=[outcome] + regressors,
    )

    y = transformed[outcome].astype(float)
    x = transformed[regressors].astype(float)

    nonzero_columns = [
        column
        for column in regressors
        if x[column].std(ddof=1) > 1e-12
    ]

    if focal_term not in nonzero_columns:
        return {
            "STATUS": "FOCAL_TERM_ABSORBED_OR_CONSTANT",
            "N_OBS": len(sample),
            "N_ETFS": n_etfs,
            "N_DATES": n_dates,
        }

    x = x[nonzero_columns]

    model = sm.OLS(
        y.to_numpy(),
        x.to_numpy(),
        hasconst=False,
    )

    groups = pd.factorize(sample["DATE"])[0]

    fit = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
        },
    )

    position = nonzero_columns.index(focal_term)

    coefficient = float(fit.params[position])
    standard_error = float(fit.bse[position])
    t_stat = float(fit.tvalues[position])
    p_value = float(fit.pvalues[position])

    ci_low = coefficient - 1.96 * standard_error
    ci_high = coefficient + 1.96 * standard_error

    return {
        "STATUS": "OK",
        "N_OBS": len(sample),
        "N_ETFS": n_etfs,
        "N_DATES": n_dates,
        "COEFFICIENT": coefficient,
        "STD_ERROR": standard_error,
        "T_STAT": t_stat,
        "P_VALUE": p_value,
        "CI_LOW_95": ci_low,
        "CI_HIGH_95": ci_high,
        "R_SQUARED_WITHIN_TRANSFORMED": float(fit.rsquared),
    }


def standardize_persistent_channels(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    result = panel.copy()

    for label, source in CHANNEL_MAP.items():
        z_name = f"PERSISTENT_Z_{label}"

        values = safe_numeric(result[source])

        unique_values = (
            result[["ETF_ID", source]]
            .drop_duplicates("ETF_ID")[source]
            .pipe(safe_numeric)
            .dropna()
        )

        mean_value = unique_values.mean()
        std_value = unique_values.std(ddof=1)

        if pd.isna(std_value) or std_value <= 0:
            result[z_name] = np.nan
        else:
            result[z_name] = (
                values - mean_value
            ) / std_value

    return result


def prepare_panel() -> pd.DataFrame:
    require_file(PANEL_FILE)
    require_file(PERSISTENT_FILE)

    panel = pd.read_csv(
        PANEL_FILE,
        low_memory=False,
    )

    persistent = pd.read_csv(
        PERSISTENT_FILE,
        low_memory=False,
    )

    required_panel = {
        "DATE",
        "ETF_ID",
        "ETF_RETURN",
        "CPU_Z",
        "VIX_LEVEL_Z",
        "CPU_AND_VIX_STRESS",
        "EXTREME_CPU_REGIME",
    }.union(BASE_CONTROLS)

    missing_panel = required_panel.difference(panel.columns)

    if missing_panel:
        raise KeyError(
            "Final panelde eksik sütunlar:\n"
            + "\n".join(sorted(missing_panel))
        )

    required_persistent = {
        "ETF_ID",
        "N_QUARTERS_AVAILABLE",
    }.union(CHANNEL_MAP.values())

    missing_persistent = required_persistent.difference(
        persistent.columns
    )

    if missing_persistent:
        raise KeyError(
            "Persistent architecture dosyasında eksik sütunlar:\n"
            + "\n".join(sorted(missing_persistent))
        )

    if persistent["ETF_ID"].duplicated().any():
        raise RuntimeError(
            "Persistent architecture dosyasında duplicate ETF_ID bulundu."
        )

    keep_columns = [
        "ETF_ID",
        "N_QUARTERS_AVAILABLE",
        "AVAILABLE_ALL_THREE_QUARTERS",
        "MEAN_FINANCIAL_MATCH_WEIGHT",
    ] + list(CHANNEL_MAP.values())

    keep_columns = [
        column
        for column in keep_columns
        if column in persistent.columns
    ]

    merged = panel.merge(
        persistent[keep_columns],
        on="ETF_ID",
        how="inner",
        validate="many_to_one",
    )

    merged["DATE"] = pd.to_datetime(
        merged["DATE"],
        errors="coerce",
    )

    merged = merged.dropna(subset=["DATE"])

    merged = standardize_persistent_channels(
        merged
    )

    # Persistent kanallar ve etkileşimler
    for label in CHANNEL_MAP:
        architecture = f"PERSISTENT_Z_{label}"

        merged[
            f"CPU_X_{label}"
        ] = (
            safe_numeric(merged["CPU_Z"])
            * safe_numeric(merged[architecture])
        )

        merged[
            f"VIX_X_{label}"
        ] = (
            safe_numeric(merged["VIX_LEVEL_Z"])
            * safe_numeric(merged[architecture])
        )

        merged[
            f"CPU_VIX_X_{label}"
        ] = (
            safe_numeric(merged["CPU_Z"])
            * safe_numeric(merged["VIX_LEVEL_Z"])
            * safe_numeric(merged[architecture])
        )

        merged[
            f"JOINT_STRESS_X_{label}"
        ] = (
            safe_numeric(merged["CPU_AND_VIX_STRESS"])
            * safe_numeric(merged[architecture])
        )

        merged[
            f"EXTREME_CPU_X_{label}"
        ] = (
            safe_numeric(merged["EXTREME_CPU_REGIME"])
            * safe_numeric(merged[architecture])
        )

    return merged


# ============================================================
# 2. ANA MODELLER
# ============================================================

def run_all_models(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    controls = [
        column
        for column in BASE_CONTROLS
        if column in panel.columns
    ]

    for label in CHANNEL_MAP:
        architecture = f"PERSISTENT_Z_{label}"

        # ----------------------------------------------------
        # A. CPU × Persistent Architecture
        # ----------------------------------------------------
        focal = f"CPU_X_{label}"

        regressors = [
            architecture,
            focal,
        ] + controls

        result = fit_two_way_fe_date_cluster(
            data=panel,
            outcome="ETF_RETURN",
            regressors=regressors,
            focal_term=focal,
        )

        rows.append(
            {
                "MODEL_FAMILY": "CPU_PRICING",
                "CHANNEL": label,
                "OUTCOME": "ETF_RETURN",
                "HORIZON": 0,
                "FOCAL_TERM": focal,
                **result,
            }
        )

        # ----------------------------------------------------
        # B. Continuous CPU × VIX × Architecture
        # ----------------------------------------------------
        focal = f"CPU_VIX_X_{label}"

        regressors = [
            architecture,
            f"CPU_X_{label}",
            f"VIX_X_{label}",
            focal,
        ] + controls

        result = fit_two_way_fe_date_cluster(
            data=panel,
            outcome="ETF_RETURN",
            regressors=regressors,
            focal_term=focal,
        )

        rows.append(
            {
                "MODEL_FAMILY": "CONTINUOUS_STRESS_ACTIVATION",
                "CHANNEL": label,
                "OUTCOME": "ETF_RETURN",
                "HORIZON": 0,
                "FOCAL_TERM": focal,
                **result,
            }
        )

        # ----------------------------------------------------
        # C. Joint-stress activation
        # ----------------------------------------------------
        focal = f"JOINT_STRESS_X_{label}"

        regressors = [
            architecture,
            focal,
        ] + controls

        result = fit_two_way_fe_date_cluster(
            data=panel,
            outcome="ETF_RETURN",
            regressors=regressors,
            focal_term=focal,
        )

        rows.append(
            {
                "MODEL_FAMILY": "JOINT_STRESS_ACTIVATION",
                "CHANNEL": label,
                "OUTCOME": "ETF_RETURN",
                "HORIZON": 0,
                "FOCAL_TERM": focal,
                **result,
            }
        )

        # ----------------------------------------------------
        # D. Extreme CPU recovery
        # ----------------------------------------------------
        for horizon, outcome in RECOVERY_OUTCOMES.items():
            if outcome not in panel.columns:
                continue

            focal = f"EXTREME_CPU_X_{label}"

            regressors = [
                architecture,
                focal,
            ] + controls

            result = fit_two_way_fe_date_cluster(
                data=panel,
                outcome=outcome,
                regressors=regressors,
                focal_term=focal,
            )

            rows.append(
                {
                    "MODEL_FAMILY": "EXTREME_CPU_RECOVERY",
                    "CHANNEL": label,
                    "OUTCOME": outcome,
                    "HORIZON": horizon,
                    "FOCAL_TERM": focal,
                    **result,
                }
            )

        # ----------------------------------------------------
        # E. Joint-stress recovery
        # ----------------------------------------------------
        for horizon, outcome in RECOVERY_OUTCOMES.items():
            if outcome not in panel.columns:
                continue

            focal = f"JOINT_STRESS_X_{label}"

            regressors = [
                architecture,
                focal,
            ] + controls

            result = fit_two_way_fe_date_cluster(
                data=panel,
                outcome=outcome,
                regressors=regressors,
                focal_term=focal,
            )

            rows.append(
                {
                    "MODEL_FAMILY": "JOINT_STRESS_RECOVERY",
                    "CHANNEL": label,
                    "OUTCOME": outcome,
                    "HORIZON": horizon,
                    "FOCAL_TERM": focal,
                    **result,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 3. MULTIPLE TESTING
# ============================================================

def add_multiple_testing(
    results: pd.DataFrame,
) -> pd.DataFrame:
    output = results.copy()

    output["P_HOLM_FAMILY"] = np.nan
    output["Q_BH_FAMILY"] = np.nan

    valid = output.loc[
        output["STATUS"].eq("OK")
        & output["P_VALUE"].notna()
    ].copy()

    for family, family_rows in valid.groupby(
        "MODEL_FAMILY"
    ):
        indices = family_rows.index
        p_values = family_rows["P_VALUE"].to_numpy()

        _, holm_p, _, _ = multipletests(
            p_values,
            method="holm",
        )

        _, bh_p, _, _ = multipletests(
            p_values,
            method="fdr_bh",
        )

        output.loc[
            indices,
            "P_HOLM_FAMILY",
        ] = holm_p

        output.loc[
            indices,
            "Q_BH_FAMILY",
        ] = bh_p

    return output


# ============================================================
# 4. ÖZETLER VE GRAFİKLER
# ============================================================

def build_key_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    key_families = [
        "CPU_PRICING",
        "CONTINUOUS_STRESS_ACTIVATION",
        "JOINT_STRESS_ACTIVATION",
    ]

    columns = [
        "MODEL_FAMILY",
        "CHANNEL",
        "N_OBS",
        "N_ETFS",
        "N_DATES",
        "COEFFICIENT",
        "STD_ERROR",
        "T_STAT",
        "P_VALUE",
        "P_HOLM_FAMILY",
        "Q_BH_FAMILY",
        "CI_LOW_95",
        "CI_HIGH_95",
        "STATUS",
    ]

    available = [
        column
        for column in columns
        if column in results.columns
    ]

    return (
        results.loc[
            results["MODEL_FAMILY"].isin(
                key_families
            )
        ][available]
        .sort_values(
            [
                "MODEL_FAMILY",
                "CHANNEL",
            ]
        )
        .reset_index(drop=True)
    )


def create_coefficient_plot(
    key_results: pd.DataFrame,
) -> None:
    plot_data = key_results.loc[
        key_results["MODEL_FAMILY"].eq(
            "CONTINUOUS_STRESS_ACTIVATION"
        )
        & key_results["STATUS"].eq("OK")
    ].copy()

    if plot_data.empty:
        return

    plot_data = plot_data.sort_values(
        "COEFFICIENT"
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    y_positions = np.arange(
        len(plot_data)
    )

    x_error = np.vstack(
        [
            plot_data["COEFFICIENT"]
            - plot_data["CI_LOW_95"],
            plot_data["CI_HIGH_95"]
            - plot_data["COEFFICIENT"],
        ]
    )

    axis.errorbar(
        plot_data["COEFFICIENT"],
        y_positions,
        xerr=x_error,
        fmt="o",
        capsize=4,
    )

    axis.axvline(
        0,
        linestyle="--",
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        plot_data["CHANNEL"]
        .str.replace("_", " ")
        .str.title()
    )

    axis.set_xlabel(
        "CPU × VIX × Persistent Architecture coefficient"
    )

    axis.set_title(
        "Persistent Portfolio Architecture and Stress Activation"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR
        / "30_continuous_stress_coefficients.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_recovery_plots(
    results: pd.DataFrame,
) -> None:
    for family in [
        "EXTREME_CPU_RECOVERY",
        "JOINT_STRESS_RECOVERY",
    ]:
        family_data = results.loc[
            results["MODEL_FAMILY"].eq(family)
            & results["STATUS"].eq("OK")
        ].copy()

        for channel, channel_data in family_data.groupby(
            "CHANNEL"
        ):
            channel_data = channel_data.sort_values(
                "HORIZON"
            )

            if channel_data.empty:
                continue

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            axis.plot(
                channel_data["HORIZON"],
                channel_data["COEFFICIENT"],
                marker="o",
            )

            axis.fill_between(
                channel_data["HORIZON"],
                channel_data["CI_LOW_95"],
                channel_data["CI_HIGH_95"],
                alpha=0.2,
            )

            axis.axhline(
                0,
                linestyle="--",
            )

            axis.set_xlabel(
                "Horizon (months)"
            )

            axis.set_ylabel(
                "Cumulative return differential"
            )

            axis.set_title(
                f"{family.replace('_', ' ').title()}\n"
                f"{channel.replace('_', ' ').title()}"
            )

            figure.tight_layout()

            figure.savefig(
                FIGURE_DIR
                / (
                    f"30_{family.lower()}_"
                    f"{channel.lower()}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(figure)


def build_validation(
    panel: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    successful = results["STATUS"].eq("OK")

    return pd.DataFrame(
        [
            {
                "CHECK": "MERGED_PANEL_ROWS",
                "VALUE": len(panel),
                "PASS": int(len(panel) > 0),
            },
            {
                "CHECK": "MERGED_ETFS",
                "VALUE": panel["ETF_ID"].nunique(),
                "PASS": int(
                    panel["ETF_ID"].nunique() >= 40
                ),
            },
            {
                "CHECK": "DUPLICATE_ETF_DATE_ROWS",
                "VALUE": int(
                    panel.duplicated(
                        subset=[
                            "ETF_ID",
                            "DATE",
                        ],
                        keep=False,
                    ).sum()
                ),
                "PASS": int(
                    panel.duplicated(
                        subset=[
                            "ETF_ID",
                            "DATE",
                        ],
                        keep=False,
                    ).sum()
                    == 0
                ),
            },
            {
                "CHECK": "SUCCESSFUL_MODELS",
                "VALUE": int(successful.sum()),
                "PASS": int(successful.sum() > 0),
            },
            {
                "CHECK": "FAILED_MODELS",
                "VALUE": int((~successful).sum()),
                "PASS": 1,
            },
            {
                "CHECK": "CONTINUOUS_STRESS_MODELS",
                "VALUE": int(
                    (
                        results["MODEL_FAMILY"]
                        .eq(
                            "CONTINUOUS_STRESS_ACTIVATION"
                        )
                        & successful
                    ).sum()
                ),
                "PASS": int(
                    (
                        results["MODEL_FAMILY"]
                        .eq(
                            "CONTINUOUS_STRESS_ACTIVATION"
                        )
                        & successful
                    ).sum()
                    >= 4
                ),
            },
        ]
    )


# ============================================================
# 5. ANA PROGRAM
# ============================================================

def main() -> None:
    print("=" * 92)
    print(
        "30 - FINAL PERSISTENT ARCHITECTURE ANALYSIS"
    )
    print("=" * 92)

    print(
        "\n1/6 - Final panel ve persistent architecture "
        "birleştiriliyor..."
    )

    panel = prepare_panel()

    print(
        f"Birleşmiş panel satırı: {len(panel):,}"
    )

    print(
        f"Benzersiz ETF: {panel['ETF_ID'].nunique():,}"
    )

    print(
        f"Benzersiz ay: {panel['DATE'].nunique():,}"
    )

    print(
        "\n2/6 - Ana pricing, stress activation ve recovery "
        "modelleri tahmin ediliyor..."
    )

    results = run_all_models(
        panel
    )

    print(
        "\n3/6 - Multiple-testing düzeltmeleri uygulanıyor..."
    )

    results = add_multiple_testing(
        results
    )

    print(
        "\n4/6 - Nihai tablolar hazırlanıyor..."
    )

    key_results = build_key_results(
        results
    )

    validation = build_validation(
        panel=panel,
        results=results,
    )

    print(
        "\n5/6 - Grafikler oluşturuluyor..."
    )

    create_coefficient_plot(
        key_results
    )

    create_recovery_plots(
        results
    )

    print(
        "\n6/6 - Çıktılar kaydediliyor..."
    )

    panel.to_csv(
        OUTPUT_DIR
        / "30_persistent_architecture_merged_panel.csv",
        index=False,
    )

    results.to_csv(
        OUTPUT_DIR
        / "30_all_final_persistent_results.csv",
        index=False,
    )

    key_results.to_csv(
        OUTPUT_DIR
        / "30_key_persistent_results.csv",
        index=False,
    )

    results.loc[
        results["MODEL_FAMILY"]
        .str.contains("RECOVERY")
    ].to_csv(
        OUTPUT_DIR
        / "30_persistent_recovery_results.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR
        / "30_final_analysis_validation.csv",
        index=False,
    )

    print("\n" + "=" * 92)
    print("FINAL ANALİZ TAMAMLANDI")
    print("=" * 92)

    print("\nAna pricing ve stress activation sonuçları:")

    display_columns = [
        "MODEL_FAMILY",
        "CHANNEL",
        "N_OBS",
        "N_ETFS",
        "COEFFICIENT",
        "STD_ERROR",
        "P_VALUE",
        "P_HOLM_FAMILY",
        "Q_BH_FAMILY",
        "STATUS",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in key_results.columns
    ]

    print(
        key_results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\nValidation:")

    print(
        validation.to_string(
            index=False
        )
    )

    print("\nAna dosyalar:")

    print(
        OUTPUT_DIR
        / "30_all_final_persistent_results.csv"
    )

    print(
        OUTPUT_DIR
        / "30_key_persistent_results.csv"
    )

    print(
        OUTPUT_DIR
        / "30_persistent_recovery_results.csv"
    )

    print(
        OUTPUT_DIR
        / "30_final_analysis_validation.csv"
    )

    print(FIGURE_DIR)


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print("\n" + "=" * 92)
        print("FINAL SCRIPT HATA İLE DURDU")
        print("=" * 92)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)
