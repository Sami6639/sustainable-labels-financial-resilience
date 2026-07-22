"""
29_persistence_analysis.py
==========================

2023Q4, 2024Q4 ve 2025Q4 fixed-universe holdings verilerinden
karşılaştırılabilir tarihsel portfolio architecture kanalları üretir.

Ana çıktılar
------------
- 29_historical_architecture_channels.csv
- 29_architecture_pairwise_correlations.csv
- 29_architecture_icc.csv
- 29_architecture_rank_persistence.csv
- 29_architecture_transition_matrices.csv
- 29_persistent_architecture_scores.csv
- 29_persistence_validation.csv
- output/29_persistence_figures/*.png

Metodolojik not
---------------
Firma karakteristikleri FY2024 kesitinde sabit tutulur. Böylece
dönemler arasındaki değişim, firma finansallarındaki değişimden değil,
ETF holdings bileşiminden kaynaklanır.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import traceback
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 0. PROJE YOLLARI VE AYARLAR
# ============================================================

PROJECT_DIR = Path.home() / "Desktop" / "CPU_Project"
SCRIPT_DIR = PROJECT_DIR / "scripts"
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "29_persistence_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

QUARTERS = ["2023Q4", "2024Q4", "2025Q4"]
REFERENCE_QUARTER = "2025Q4"

FIRM_FILE = (
    OUTPUT_DIR
    / "firm_characteristics_annual_identifiers.csv"
)

HOLDINGS_FILES = {
    quarter: (
        OUTPUT_DIR
        / f"{quarter.lower()}_fixed_universe_equity_corporate_holdings.parquet"
    )
    for quarter in QUARTERS
}

CHANNELS = [
    "INTERNAL_FINANCIAL_CAPACITY",
    "EXTERNAL_FINANCING_DEPENDENCE",
    "GROWTH_DURATION_EXPOSURE_FINAL",
    "PORTFOLIO_CONCENTRATION_FINAL",
    "FINANCIAL_ARCHITECTURE_RISK_FINAL",
    "EXTENDED_ARCHITECTURE_RISK_FINAL",
]


# ============================================================
# 1. GENEL YARDIMCI FONKSİYONLAR
# ============================================================

def require_file(path: Path) -> None:
    """Dosyanın mevcut ve boş olmadığını doğrular."""
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı:\n{path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Dosya boş görünüyor:\n{path}"
        )


def load_module(
    module_name: str,
    file_path: Path,
):
    """Bir Python scriptini modül olarak güvenli biçimde yükler."""
    require_file(file_path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Script modül olarak yüklenemedi:\n{file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def safe_numeric(series: pd.Series) -> pd.Series:
    """Sayısal olmayan değerleri NaN yapar."""
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def prepare_historical_holdings(
    quarter: str,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tarihsel holdings içinde sabit 2025Q4 referans ETF kimliğini
    SERIES_ID ve SERIES_NAME olarak kullanır.
    """
    result = holdings.copy()

    required_columns = {
        "REFERENCE_SERIES_ID",
        "REFERENCE_SERIES_NAME",
    }

    missing = required_columns.difference(result.columns)

    if missing:
        raise KeyError(
            f"{quarter} holdings dosyasında eksik sütunlar:\n"
            + "\n".join(sorted(missing))
        )

    result["SERIES_ID"] = (
        result["REFERENCE_SERIES_ID"]
        .astype("string")
        .str.strip()
    )

    result["SERIES_NAME"] = (
        result["REFERENCE_SERIES_NAME"]
        .astype("string")
        .str.strip()
    )

    result["SNAPSHOT_QUARTER"] = quarter

    return result


# ============================================================
# 2. ÇEYREK BAZINDA PORTFOLIO ARCHITECTURE
# ============================================================

def build_quarter_architecture(
    quarter: str,
    holdings_raw: pd.DataFrame,
    firm_raw: pd.DataFrame,
    script15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Script 15 fonksiyonlarını kullanarak tek çeyrek için
    holdings-based architecture paneli üretir.
    """
    print(f"\n{quarter}: Holdings standardize ediliyor...")

    holdings = script15.standardize_holdings(
        holdings_raw
    )

    print(f"{quarter}: Holding ağırlıkları oluşturuluyor...")

    holdings = script15.construct_holding_weights(
        holdings
    )

    print(
        f"{quarter}: Aynı ekonomik holdingler "
        "toplulaştırılıyor..."
    )

    aggregated = script15.aggregate_duplicate_holdings(
        holdings
    )

    print(
        f"{quarter}: Firma karakteristikleri hazırlanıyor..."
    )

    firm = script15.prepare_firm_characteristics(
        firm_raw
    )

    print(
        f"{quarter}: Holdings-firma eşleştirmesi yapılıyor..."
    )

    merged = (
        script15
        .merge_holdings_with_firm_characteristics(
            holdings=aggregated,
            firm=firm,
        )
    )

    print(
        f"{quarter}: Portfolio architecture hesaplanıyor..."
    )

    architecture, _ = (
        script15.calculate_portfolio_architecture(
            merged
        )
    )

    architecture["SNAPSHOT_QUARTER"] = quarter

    diagnostics = pd.DataFrame(
        [
            {
                "SNAPSHOT_QUARTER": quarter,
                "RAW_HOLDING_ROWS": len(holdings_raw),
                "STANDARDIZED_HOLDING_ROWS": len(holdings),
                "AGGREGATED_HOLDING_ROWS": len(aggregated),
                "ETF_COUNT": int(
                    architecture["ETF_ID"].nunique()
                ),
                "MATCHED_HOLDING_ROWS": int(
                    merged["FINANCIAL_MATCHED"].sum()
                ),
                "UNMATCHED_HOLDING_ROWS": int(
                    (~merged["FINANCIAL_MATCHED"]).sum()
                ),
                "MEAN_FINANCIAL_MATCH_WEIGHT": float(
                    architecture[
                        "FINANCIAL_MATCH_WEIGHT"
                    ].mean()
                ),
                "MEDIAN_FINANCIAL_MATCH_WEIGHT": float(
                    architecture[
                        "FINANCIAL_MATCH_WEIGHT"
                    ].median()
                ),
            }
        ]
    )

    return architecture, diagnostics


# ============================================================
# 3. TARİHSEL NİHAİ ARCHITECTURE KANALLARI
# ============================================================

def build_historical_channels(
    architecture_all: pd.DataFrame,
    script24,
    script24b,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardizasyon parametrelerini yalnızca 2025Q4 referans
    kesitinden hesaplar ve aynı parametreleri üç döneme uygular.
    """
    source_variables = (
        script24.collect_source_variables()
    )

    reference = architecture_all.loc[
        architecture_all["SNAPSHOT_QUARTER"]
        .eq(REFERENCE_QUARTER)
    ].copy()

    if reference.empty:
        raise RuntimeError(
            "2025Q4 referans architecture kesiti bulunamadı."
        )

    print(
        "\n2025Q4 sabit standardizasyon parametreleri "
        "hesaplanıyor..."
    )

    parameters = (
        script24.build_standardization_parameters(
            df=reference,
            source_variables=source_variables,
        )
    )

    quarter_results: List[pd.DataFrame] = []

    for quarter in QUARTERS:
        print(
            f"\n{quarter}: Nihai architecture kanalları "
            "oluşturuluyor..."
        )

        subset = architecture_all.loc[
            architecture_all["SNAPSHOT_QUARTER"]
            .eq(quarter)
        ].copy()

        subset = script24.create_standardized_sources(
            df=subset,
            parameters=parameters,
        )

        subset, _ = script24.build_rebuilt_factors(
            subset
        )

        subset = script24.add_factor_quality_flags(
            subset
        )

        subset = script24.build_signed_factor_composite(
            df=subset,
            component_map=(
                script24.FINANCIAL_RISK_COMPONENTS
            ),
            output_name=(
                "FINANCIAL_ARCHITECTURE_RISK"
            ),
            minimum_components=2,
        )

        subset = script24.build_signed_factor_composite(
            df=subset,
            component_map=(
                script24.EXTENDED_RISK_COMPONENTS
            ),
            output_name=(
                "EXTENDED_ARCHITECTURE_RISK"
            ),
            minimum_components=3,
        )

        subset = script24.add_composite_quality(
            subset
        )

        subset = script24b.build_final_channels(
            subset
        )

        subset = script24b.add_channel_quality_flags(
            subset
        )

        subset = script24b.build_final_composites(
            subset
        )

        subset = script24b.add_composite_quality_flags(
            subset
        )

        quarter_results.append(subset)

    channels = pd.concat(
        quarter_results,
        ignore_index=True,
    )

    return channels, parameters


# ============================================================
# 4. PAIRWISE KORELASYONLAR
# ============================================================

def build_pairwise_correlations(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Pearson, Spearman ve Kendall korelasyonlarını hesaplar."""
    rows: List[Dict[str, object]] = []

    quarter_pairs = [
        ("2023Q4", "2024Q4"),
        ("2024Q4", "2025Q4"),
        ("2023Q4", "2025Q4"),
    ]

    for channel in CHANNELS:
        wide = panel.pivot_table(
            index="ETF_ID",
            columns="SNAPSHOT_QUARTER",
            values=channel,
            aggfunc="first",
        )

        for quarter_1, quarter_2 in quarter_pairs:
            if (
                quarter_1 not in wide.columns
                or quarter_2 not in wide.columns
            ):
                continue

            sample = wide[
                [quarter_1, quarter_2]
            ].dropna()

            if len(sample) < 5:
                continue

            pearson_r, pearson_p = stats.pearsonr(
                sample[quarter_1],
                sample[quarter_2],
            )

            spearman_rho, spearman_p = stats.spearmanr(
                sample[quarter_1],
                sample[quarter_2],
            )

            kendall_tau, kendall_p = stats.kendalltau(
                sample[quarter_1],
                sample[quarter_2],
            )

            rows.append(
                {
                    "CHANNEL": channel,
                    "QUARTER_1": quarter_1,
                    "QUARTER_2": quarter_2,
                    "N_ETFS": len(sample),
                    "PEARSON_R": float(pearson_r),
                    "PEARSON_P": float(pearson_p),
                    "SPEARMAN_RHO": float(spearman_rho),
                    "SPEARMAN_P": float(spearman_p),
                    "KENDALL_TAU": float(kendall_tau),
                    "KENDALL_P": float(kendall_p),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 5. ICC(3,1)
# ============================================================

def calculate_icc_3_1(
    matrix: np.ndarray,
) -> float:
    """
    ICC(3,1):
    Two-way mixed-effects, consistency, single measurement.

    Satırlar ETF, sütunlar çeyrektir.
    """
    values = np.asarray(
        matrix,
        dtype=float,
    )

    n_subjects, n_measurements = values.shape

    if n_subjects < 2 or n_measurements < 2:
        return np.nan

    grand_mean = float(values.mean())

    subject_means = values.mean(
        axis=1
    )

    measurement_means = values.mean(
        axis=0
    )

    ss_subjects = (
        n_measurements
        * np.sum(
            (subject_means - grand_mean) ** 2
        )
    )

    ss_error = np.sum(
        (
            values
            - subject_means[:, None]
            - measurement_means[None, :]
            + grand_mean
        )
        ** 2
    )

    ms_subjects = (
        ss_subjects
        / (n_subjects - 1)
    )

    ms_error = (
        ss_error
        / (
            (n_subjects - 1)
            * (n_measurements - 1)
        )
    )

    denominator = (
        ms_subjects
        + (n_measurements - 1) * ms_error
    )

    if denominator <= 0:
        return np.nan

    return float(
        (ms_subjects - ms_error)
        / denominator
    )


def build_icc_table(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Her kanal için üç çeyrekli ICC(3,1) hesaplar."""
    rows = []

    for channel in CHANNELS:
        wide = panel.pivot_table(
            index="ETF_ID",
            columns="SNAPSHOT_QUARTER",
            values=channel,
            aggfunc="first",
        )

        available_quarters = [
            quarter
            for quarter in QUARTERS
            if quarter in wide.columns
        ]

        if len(available_quarters) < 2:
            continue

        sample = wide[
            available_quarters
        ].dropna()

        if len(sample) < 5:
            continue

        icc_value = calculate_icc_3_1(
            sample.to_numpy(
                dtype=float
            )
        )

        if pd.isna(icc_value):
            interpretation = "Unavailable"
        elif icc_value >= 0.90:
            interpretation = "Excellent"
        elif icc_value >= 0.75:
            interpretation = "Good"
        elif icc_value >= 0.50:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"

        rows.append(
            {
                "CHANNEL": channel,
                "N_ETFS": len(sample),
                "N_QUARTERS": len(
                    available_quarters
                ),
                "ICC_3_1": icc_value,
                "INTERPRETATION": interpretation,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 6. RANK PERSISTENCE VE TRANSITION MATRIX
# ============================================================

def tercile_labels(
    series: pd.Series,
) -> pd.Series:
    """Seriyi LOW, MIDDLE ve HIGH tercillerine ayırır."""
    valid = safe_numeric(series).dropna()

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="object",
    )

    if len(valid) < 6:
        return result

    ranks = valid.rank(
        method="first"
    )

    groups = pd.qcut(
        ranks,
        q=3,
        labels=[
            "LOW",
            "MIDDLE",
            "HIGH",
        ],
    )

    result.loc[valid.index] = (
        groups.astype(str)
    )

    return result


def build_rank_and_transition_tables(
    panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Percentile değişimleri, high/low retention ve 2023Q4-2025Q4
    tercile transition matrix üretir.
    """
    rank_rows = []
    transition_rows = []

    quarter_pairs = [
        ("2023Q4", "2024Q4"),
        ("2024Q4", "2025Q4"),
        ("2023Q4", "2025Q4"),
    ]

    for channel in CHANNELS:
        wide = panel.pivot_table(
            index="ETF_ID",
            columns="SNAPSHOT_QUARTER",
            values=channel,
            aggfunc="first",
        )

        rank_wide = wide.rank(
            axis=0,
            method="average",
            pct=True,
        )

        for quarter_1, quarter_2 in quarter_pairs:
            if (
                quarter_1 not in rank_wide.columns
                or quarter_2 not in rank_wide.columns
            ):
                continue

            sample = rank_wide[
                [quarter_1, quarter_2]
            ].dropna()

            if len(sample) < 5:
                continue

            absolute_change = (
                sample[quarter_2]
                - sample[quarter_1]
            ).abs()

            same_half = (
                (sample[quarter_1] >= 0.50)
                == (sample[quarter_2] >= 0.50)
            )

            top_base = max(
                int(
                    (
                        sample[quarter_1]
                        >= (2 / 3)
                    ).sum()
                ),
                1,
            )

            bottom_base = max(
                int(
                    (
                        sample[quarter_1]
                        <= (1 / 3)
                    ).sum()
                ),
                1,
            )

            top_retention = (
                (
                    (sample[quarter_1] >= 2 / 3)
                    & (sample[quarter_2] >= 2 / 3)
                ).sum()
                / top_base
            )

            bottom_retention = (
                (
                    (sample[quarter_1] <= 1 / 3)
                    & (sample[quarter_2] <= 1 / 3)
                ).sum()
                / bottom_base
            )

            rank_rows.append(
                {
                    "CHANNEL": channel,
                    "QUARTER_1": quarter_1,
                    "QUARTER_2": quarter_2,
                    "N_ETFS": len(sample),
                    "MEAN_ABS_PERCENTILE_CHANGE": float(
                        absolute_change.mean()
                    ),
                    "MEDIAN_ABS_PERCENTILE_CHANGE": float(
                        absolute_change.median()
                    ),
                    "SAME_HIGH_LOW_HALF_SHARE": float(
                        same_half.mean()
                    ),
                    "TOP_TERCILE_RETENTION_SHARE": float(
                        top_retention
                    ),
                    "BOTTOM_TERCILE_RETENTION_SHARE": float(
                        bottom_retention
                    ),
                }
            )

        if {
            "2023Q4",
            "2025Q4",
        }.issubset(wide.columns):

            transition_sample = wide[
                ["2023Q4", "2025Q4"]
            ].dropna().copy()

            if len(transition_sample) >= 6:
                transition_sample[
                    "FROM_GROUP"
                ] = tercile_labels(
                    transition_sample["2023Q4"]
                )

                transition_sample[
                    "TO_GROUP"
                ] = tercile_labels(
                    transition_sample["2025Q4"]
                )

                table = pd.crosstab(
                    transition_sample["FROM_GROUP"],
                    transition_sample["TO_GROUP"],
                )

                for from_group in [
                    "LOW",
                    "MIDDLE",
                    "HIGH",
                ]:
                    for to_group in [
                        "LOW",
                        "MIDDLE",
                        "HIGH",
                    ]:
                        count = 0

                        if (
                            from_group in table.index
                            and to_group in table.columns
                        ):
                            count = int(
                                table.loc[
                                    from_group,
                                    to_group,
                                ]
                            )

                        transition_rows.append(
                            {
                                "CHANNEL": channel,
                                "FROM_QUARTER": "2023Q4",
                                "TO_QUARTER": "2025Q4",
                                "FROM_GROUP": from_group,
                                "TO_GROUP": to_group,
                                "ETF_COUNT": count,
                            }
                        )

    return (
        pd.DataFrame(rank_rows),
        pd.DataFrame(transition_rows),
    )


# ============================================================
# 7. PERSISTENT ARCHITECTURE SKORLARI
# ============================================================

def build_persistent_scores(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her ETF için 2023Q4-2025Q4 kanal ortalamalarını ve dönemsel
    değişkenlik ölçülerini üretir.
    """
    identity = (
        panel
        .sort_values("SNAPSHOT_QUARTER")
        .groupby(
            "ETF_ID",
            as_index=False,
        )
        .agg(
            ETF_NAME=(
                "ETF_NAME",
                "last",
            ),
            N_QUARTERS_AVAILABLE=(
                "SNAPSHOT_QUARTER",
                "nunique",
            ),
            MEAN_FINANCIAL_MATCH_WEIGHT=(
                "FINANCIAL_MATCH_WEIGHT",
                "mean",
            ),
        )
    )

    persistent = identity.copy()

    for channel in CHANNELS:
        summary = (
            panel.groupby(
                "ETF_ID"
            )[channel]
            .agg(
                [
                    "mean",
                    "std",
                    "count",
                    "min",
                    "max",
                ]
            )
            .reset_index()
            .rename(
                columns={
                    "mean": (
                        f"PERSISTENT_{channel}"
                    ),
                    "std": f"SD_{channel}",
                    "count": f"N_{channel}",
                    "min": f"MIN_{channel}",
                    "max": f"MAX_{channel}",
                }
            )
        )

        persistent = persistent.merge(
            summary,
            on="ETF_ID",
            how="left",
            validate="one_to_one",
        )

        persistent[
            f"RANGE_{channel}"
        ] = (
            persistent[f"MAX_{channel}"]
            - persistent[f"MIN_{channel}"]
        )

    persistent[
        "AVAILABLE_ALL_THREE_QUARTERS"
    ] = (
        persistent[
            "N_QUARTERS_AVAILABLE"
        ]
        .eq(3)
        .astype(int)
    )

    return persistent


# ============================================================
# 8. GRAFİKLER
# ============================================================

def create_figures(
    panel: pd.DataFrame,
    correlations: pd.DataFrame,
    icc_table: pd.DataFrame,
) -> None:
    """Publication-quality temel persistence grafiklerini üretir."""

    # 2023Q4 - 2025Q4 scatter grafikleri
    for channel in CHANNELS[:4]:
        wide = panel.pivot_table(
            index="ETF_ID",
            columns="SNAPSHOT_QUARTER",
            values=channel,
            aggfunc="first",
        )

        if not {
            "2023Q4",
            "2025Q4",
        }.issubset(wide.columns):
            continue

        sample = wide[
            ["2023Q4", "2025Q4"]
        ].dropna()

        if len(sample) < 5:
            continue

        figure, axis = plt.subplots(
            figsize=(7, 6)
        )

        axis.scatter(
            sample["2023Q4"],
            sample["2025Q4"],
        )

        slope, intercept = np.polyfit(
            sample["2023Q4"],
            sample["2025Q4"],
            1,
        )

        x_values = np.linspace(
            sample["2023Q4"].min(),
            sample["2023Q4"].max(),
            100,
        )

        axis.plot(
            x_values,
            intercept + slope * x_values,
        )

        axis.set_xlabel(
            "2023Q4 architecture score"
        )

        axis.set_ylabel(
            "2025Q4 architecture score"
        )

        axis.set_title(
            channel.replace(
                "_",
                " ",
            ).title()
        )

        figure.tight_layout()

        figure.savefig(
            FIGURE_DIR
            / f"29_scatter_{channel.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)

    # ICC özeti
    if not icc_table.empty:
        plot_data = (
            icc_table
            .sort_values("ICC_3_1")
            .copy()
        )

        figure, axis = plt.subplots(
            figsize=(9, 6)
        )

        axis.barh(
            plot_data["CHANNEL"]
            .str.replace("_", " ")
            .str.title(),
            plot_data["ICC_3_1"],
        )

        axis.axvline(
            0.75,
            linestyle="--",
        )

        axis.axvline(
            0.90,
            linestyle="--",
        )

        axis.set_xlabel("ICC(3,1)")
        axis.set_title(
            "Portfolio Architecture Persistence"
        )

        figure.tight_layout()

        figure.savefig(
            FIGURE_DIR
            / "29_architecture_icc_summary.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)

    # 2023Q4 - 2025Q4 Spearman özeti
    if not correlations.empty:
        pair = correlations.loc[
            correlations["QUARTER_1"].eq("2023Q4")
            & correlations["QUARTER_2"].eq("2025Q4")
        ].copy()

        if not pair.empty:
            pair = pair.sort_values(
                "SPEARMAN_RHO"
            )

            figure, axis = plt.subplots(
                figsize=(9, 6)
            )

            axis.barh(
                pair["CHANNEL"]
                .str.replace("_", " ")
                .str.title(),
                pair["SPEARMAN_RHO"],
            )

            axis.set_xlabel(
                "Spearman rank correlation"
            )

            axis.set_title(
                "Architecture Rank Persistence: "
                "2023Q4–2025Q4"
            )

            figure.tight_layout()

            figure.savefig(
                FIGURE_DIR
                / "29_architecture_spearman_summary.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(figure)


# ============================================================
# 9. VALIDATION
# ============================================================

def build_validation(
    panel: pd.DataFrame,
    correlations: pd.DataFrame,
    icc_table: pd.DataFrame,
    persistent: pd.DataFrame,
) -> pd.DataFrame:
    """Mekanik doğrulama kontrollerini üretir."""

    duplicate_count = int(
        panel.duplicated(
            subset=[
                "ETF_ID",
                "SNAPSHOT_QUARTER",
            ],
            keep=False,
        ).sum()
    )

    present_quarters = set(
        panel["SNAPSHOT_QUARTER"]
        .dropna()
        .astype(str)
        .unique()
    )

    all_three_count = int(
        persistent[
            "AVAILABLE_ALL_THREE_QUARTERS"
        ].sum()
    )

    rows = [
        {
            "CHECK": "THREE_QUARTERS_PRESENT",
            "VALUE": len(present_quarters),
            "PASS": int(
                set(QUARTERS).issubset(
                    present_quarters
                )
            ),
        },
        {
            "CHECK": "DUPLICATE_ETF_QUARTER_ROWS",
            "VALUE": duplicate_count,
            "PASS": int(
                duplicate_count == 0
            ),
        },
        {
            "CHECK": "PAIRWISE_CORRELATIONS_AVAILABLE",
            "VALUE": len(correlations),
            "PASS": int(
                len(correlations) > 0
            ),
        },
        {
            "CHECK": "ICC_RESULTS_AVAILABLE",
            "VALUE": len(icc_table),
            "PASS": int(
                len(icc_table) > 0
            ),
        },
        {
            "CHECK": "PERSISTENT_SCORES_AVAILABLE",
            "VALUE": len(persistent),
            "PASS": int(
                len(persistent) > 0
            ),
        },
        {
            "CHECK": "ETFS_AVAILABLE_ALL_THREE_QUARTERS",
            "VALUE": all_three_count,
            "PASS": int(
                all_three_count >= 40
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# 10. ANA PROGRAM
# ============================================================

def main() -> None:
    print("=" * 88)
    print(
        "29 - HISTORICAL PORTFOLIO "
        "ARCHITECTURE PERSISTENCE"
    )
    print("=" * 88)

    require_file(FIRM_FILE)

    for holdings_file in HOLDINGS_FILES.values():
        require_file(holdings_file)

    script15 = load_module(
        module_name="script15",
        file_path=(
            SCRIPT_DIR
            / "15_build_portfolio_architecture.py"
        ),
    )

    script24 = load_module(
        module_name="script24",
        file_path=(
            SCRIPT_DIR
            / "24_rebuild_architecture_factors.py"
        ),
    )

    script24b = load_module(
        module_name="script24b",
        file_path=(
            SCRIPT_DIR
            / "24b_finalize_architecture_channels.py"
        ),
    )

    print(
        "\nFirma karakteristikleri CSV dosyasından okunuyor..."
    )

    firm_raw = pd.read_csv(
        FIRM_FILE,
        low_memory=False,
    )

    architectures = []
    diagnostics = []

    for quarter in QUARTERS:
        print("\n" + "-" * 88)
        print(f"Dönem: {quarter}")
        print("-" * 88)

        holdings_raw = pd.read_parquet(
            HOLDINGS_FILES[quarter]
        )

        holdings_raw = prepare_historical_holdings(
            quarter=quarter,
            holdings=holdings_raw,
        )

        architecture, diagnostic = (
            build_quarter_architecture(
                quarter=quarter,
                holdings_raw=holdings_raw,
                firm_raw=firm_raw,
                script15=script15,
            )
        )

        architectures.append(
            architecture
        )

        diagnostics.append(
            diagnostic
        )

    architecture_all = pd.concat(
        architectures,
        ignore_index=True,
    )

    diagnostic_table = pd.concat(
        diagnostics,
        ignore_index=True,
    )

    channels, parameters = (
        build_historical_channels(
            architecture_all=architecture_all,
            script24=script24,
            script24b=script24b,
        )
    )

    correlations = (
        build_pairwise_correlations(
            channels
        )
    )

    icc_table = build_icc_table(
        channels
    )

    rank_persistence, transitions = (
        build_rank_and_transition_tables(
            channels
        )
    )

    persistent = build_persistent_scores(
        channels
    )

    validation = build_validation(
        panel=channels,
        correlations=correlations,
        icc_table=icc_table,
        persistent=persistent,
    )

    create_figures(
        panel=channels,
        correlations=correlations,
        icc_table=icc_table,
    )

    channels.to_csv(
        OUTPUT_DIR
        / "29_historical_architecture_channels.csv",
        index=False,
    )

    channels.to_parquet(
        OUTPUT_DIR
        / "29_historical_architecture_channels.parquet",
        index=False,
    )

    diagnostic_table.to_csv(
        OUTPUT_DIR
        / "29_historical_architecture_diagnostics.csv",
        index=False,
    )

    parameters.to_csv(
        OUTPUT_DIR
        / "29_fixed_standardization_parameters.csv",
        index=False,
    )

    correlations.to_csv(
        OUTPUT_DIR
        / "29_architecture_pairwise_correlations.csv",
        index=False,
    )

    icc_table.to_csv(
        OUTPUT_DIR
        / "29_architecture_icc.csv",
        index=False,
    )

    rank_persistence.to_csv(
        OUTPUT_DIR
        / "29_architecture_rank_persistence.csv",
        index=False,
    )

    transitions.to_csv(
        OUTPUT_DIR
        / "29_architecture_transition_matrices.csv",
        index=False,
    )

    persistent.to_csv(
        OUTPUT_DIR
        / "29_persistent_architecture_scores.csv",
        index=False,
    )

    persistent.to_parquet(
        OUTPUT_DIR
        / "29_persistent_architecture_scores.parquet",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR
        / "29_persistence_validation.csv",
        index=False,
    )

    print("\n" + "=" * 88)
    print("PERSISTENCE ANALİZİ TAMAMLANDI")
    print("=" * 88)

    print(
        "\nDönem bazında architecture diagnostics:"
    )

    print(
        diagnostic_table.to_string(
            index=False
        )
    )

    print("\nPairwise correlation özeti:")

    if correlations.empty:
        print(
            "Pairwise korelasyon üretilemedi."
        )
    else:
        print(
            correlations[
                [
                    "CHANNEL",
                    "QUARTER_1",
                    "QUARTER_2",
                    "N_ETFS",
                    "PEARSON_R",
                    "SPEARMAN_RHO",
                    "KENDALL_TAU",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nICC sonuçları:")

    if icc_table.empty:
        print("ICC sonucu üretilemedi.")
    else:
        print(
            icc_table.to_string(
                index=False
            )
        )

    print("\nRank persistence:")

    if rank_persistence.empty:
        print(
            "Rank persistence sonucu üretilemedi."
        )
    else:
        print(
            rank_persistence.to_string(
                index=False
            )
        )

    print("\nValidation:")

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        "\nÜç dönemde architecture skoru bulunan ETF sayısı:"
    )

    print(
        int(
            persistent[
                "AVAILABLE_ALL_THREE_QUARTERS"
            ].sum()
        )
    )

    print("\nAna çıktılar:")

    print(
        OUTPUT_DIR
        / "29_historical_architecture_channels.csv"
    )

    print(
        OUTPUT_DIR
        / "29_architecture_pairwise_correlations.csv"
    )

    print(
        OUTPUT_DIR
        / "29_architecture_icc.csv"
    )

    print(
        OUTPUT_DIR
        / "29_architecture_rank_persistence.csv"
    )

    print(
        OUTPUT_DIR
        / "29_persistent_architecture_scores.csv"
    )

    print(
        OUTPUT_DIR
        / "29_persistence_validation.csv"
    )

    print(FIGURE_DIR)


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print("\n" + "=" * 88)
        print("SCRIPT 29 HATA İLE DURDU")
        print("=" * 88)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)
