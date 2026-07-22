from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")
INPUT_FILE = PROJECT_DIR / "output" / "2025q4_sustainable_etf_candidates.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "2025q4_etf_preliminary_classification.csv"


def contains_any(text: str, words: list[str]) -> bool:
    text = str(text).lower()
    return any(word.lower() in text for word in words)


def classify_asset_type(name: str) -> str:
    if contains_any(
        name,
        [
            "bond",
            "fixed income",
            "treasury",
            "municipal",
            "muni ",
            "high yield",
            "credit",
        ],
    ):
        return "BOND"

    if contains_any(
        name,
        [
            "2x",
            "3x",
            "ultra",
            "leveraged",
            "inverse",
            "short ",
            "bear ",
            "bull ",
        ],
    ):
        return "LEVERAGED_OR_INVERSE"

    if contains_any(
        name,
        [
            "commodity",
            "futures",
            "oil fund",
            "natural gas",
            "uranium",
        ],
    ):
        return "COMMODITY_OR_FUTURES"

    if contains_any(
        name,
        [
            "reit",
            "real estate",
        ],
    ):
        return "REAL_ESTATE"

    return "POTENTIAL_EQUITY"


def classify_theme(name: str) -> str:
    if contains_any(name, ["solar"]):
        return "SOLAR"

    if contains_any(name, ["wind"]):
        return "WIND"

    if contains_any(
        name,
        [
            "battery",
            "lithium",
            "electric vehicle",
            "clean transportation",
            "future mobility",
        ],
    ):
        return "BATTERY_EV"

    if contains_any(
        name,
        [
            "water",
        ],
    ):
        return "WATER"

    if contains_any(
        name,
        [
            "clean energy",
            "renewable energy",
            "alternative energy",
            "clean power",
        ],
    ):
        return "CLEAN_ENERGY"

    if contains_any(
        name,
        [
            "climate",
            "carbon transition",
            "low carbon",
            "decarbon",
            "net zero",
            "paris-aligned",
        ],
    ):
        return "CLIMATE_TRANSITION"

    if contains_any(
        name,
        [
            "sustainable infrastructure",
            "clean infrastructure",
            "green infrastructure",
            "smart grid",
        ],
    ):
        return "SUSTAINABLE_INFRASTRUCTURE"

    if contains_any(
        name,
        [
            "environmental",
            "cleantech",
            "clean technology",
            "circular economy",
        ],
    ):
        return "ENVIRONMENTAL_TECHNOLOGY"

    if contains_any(
        name,
        [
            "esg",
            "sustainability",
            "sustainable",
            "socially responsible",
            "sri",
        ],
    ):
        return "BROAD_ESG"

    if contains_any(name, ["green bond"]):
        return "GREEN_BOND"

    return "OTHER_SUSTAINABLE"


def preliminary_decision(asset_type: str, theme: str) -> str:
    if asset_type != "POTENTIAL_EQUITY":
        return "EXCLUDE_PRELIMINARY"

    if theme == "GREEN_BOND":
        return "EXCLUDE_PRELIMINARY"

    return "REVIEW_FOR_EQUITY_SAMPLE"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        low_memory=False,
    )

    required_columns = {
        "ACCESSION_NUMBER",
        "SERIES_NAME",
        "SERIES_ID",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing)}"
        )

    df["ASSET_TYPE_PRELIMINARY"] = df["SERIES_NAME"].apply(
        classify_asset_type
    )

    df["THEME_PRELIMINARY"] = df["SERIES_NAME"].apply(
        classify_theme
    )

    df["SAMPLE_DECISION_PRELIMINARY"] = df.apply(
        lambda row: preliminary_decision(
            row["ASSET_TYPE_PRELIMINARY"],
            row["THEME_PRELIMINARY"],
        ),
        axis=1,
    )

    # Bu sütunlar manuel bilimsel inceleme için boş bırakılıyor.
    df["MANUAL_ASSET_TYPE"] = ""
    df["MANUAL_THEME"] = ""
    df["FINAL_INCLUDE"] = ""
    df["EXCLUSION_REASON"] = ""
    df["REVIEW_NOTES"] = ""

    df = df.sort_values(
        [
            "SAMPLE_DECISION_PRELIMINARY",
            "THEME_PRELIMINARY",
            "SERIES_NAME",
        ]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("SINIFLANDIRMA TAMAMLANDI")
    print("=" * 70)
    print(f"Toplam aday ETF: {len(df):,}")
    print()
    print("Ön sınıflandırma:")
    print(
        df["ASSET_TYPE_PRELIMINARY"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()
    print("Tema dağılımı:")
    print(
        df["THEME_PRELIMINARY"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()
    print("Ön karar:")
    print(
        df["SAMPLE_DECISION_PRELIMINARY"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()
    print(f"Dosya kaydedildi:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()