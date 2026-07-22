from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(r"C:\Users\User\Desktop\CPU_Project")

INPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_etf_preliminary_classification.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "2025q4_etf_refined_review_list.csv"
)


def contains_any(name: str, terms: list[str]) -> bool:
    text = str(name).lower()
    return any(term.lower() in text for term in terms)


def refine_decision(row: pd.Series) -> tuple[str, str]:
    name = str(row["SERIES_NAME"])
    asset_type = str(row["ASSET_TYPE_PRELIMINARY"])
    theme = str(row["THEME_PRELIMINARY"])

    # Önceden açık biçimde elenen varlık sınıfları
    if asset_type == "BOND":
        return "EXCLUDE", "Bond or fixed-income ETF"

    if asset_type == "LEVERAGED_OR_INVERSE":
        return "EXCLUDE", "Leveraged or inverse strategy"

    if asset_type == "COMMODITY_OR_FUTURES":
        return "EXCLUDE", "Commodity or futures-based strategy"

    if asset_type == "REAL_ESTATE":
        return "EXCLUDE", "Real-estate or REIT strategy"

    # Karma varlık / allocation ETF'leri
    if contains_any(
        name,
        [
            "allocation",
            "conservative",
            "moderate",
            "balanced",
            "aggressive allocation",
        ],
    ):
        return "EXCLUDE", "Multi-asset allocation ETF"

    # Geleneksel fosil enerji ve enerji gelir stratejileri
    if contains_any(
        name,
        [
            "energy income",
            "energy index",
            "energy producers",
            "u.s. energy",
            "global energy",
            "energy exploration",
            "energy momentum",
            "equal weight energy",
            "smallcap energy",
            "mlp",
            "oil",
            "natural gas",
            "american energy independence",
        ],
    ):
        return "EXCLUDE", "Conventional energy exposure"

    # Karbon allowance fonları hisse senedi değildir
    if contains_any(
        name,
        [
            "carbon allowance",
            "global carbon strategy",
        ],
    ):
        return "EXCLUDE", "Carbon allowance or emissions-market strategy"

    # Tematik sürdürülebilir equity ETF'leri
    if theme in {
        "CLEAN_ENERGY",
        "SOLAR",
        "WIND",
        "BATTERY_EV",
        "WATER",
        "CLIMATE_TRANSITION",
        "ENVIRONMENTAL_TECHNOLOGY",
        "SUSTAINABLE_INFRASTRUCTURE",
    }:
        return "INCLUDE", "Direct sustainable-equity or transition theme"

    # Geniş ESG equity fonları
    if theme == "BROAD_ESG":
        if contains_any(
            name,
            [
                "real assets",
                "dividend",
                "min vol",
                "small-cap",
                "mid-cap",
                "emerging markets",
                "international",
                "world",
                "eafe",
                "acwi",
            ],
        ):
            return (
                "MANUAL_REVIEW",
                "Broad ESG equity fund; geography or style requires review",
            )

        return "INCLUDE", "Broad ESG equity benchmark or strategy"

    # Enerji dönüşümüyle ilişkili fakat adı tek başına yeterli olmayan fonlar
    if contains_any(
        name,
        [
            "energy transition",
            "green metals",
            "green building",
            "energy storage",
            "remediation",
            "green alpha",
        ],
    ):
        return (
            "MANUAL_REVIEW",
            "Potential transition-equity exposure; mandate requires verification",
        )

    return (
        "MANUAL_REVIEW",
        "Name and preliminary theme are insufficient for final classification",
    )


def main() -> None:
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
        "ASSET_TYPE_PRELIMINARY",
        "THEME_PRELIMINARY",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing)}"
        )

    decisions = df.apply(
        refine_decision,
        axis=1,
        result_type="expand",
    )

    decisions.columns = [
        "REFINED_DECISION",
        "REFINED_REASON",
    ]

    df = pd.concat(
        [df, decisions],
        axis=1,
    )

    # Nihai karar henüz otomatik doldurulmuyor.
    df["FINAL_DECISION"] = ""
    df["FINAL_THEME"] = ""
    df["FINAL_REVIEW_NOTE"] = ""

    decision_order = {
        "MANUAL_REVIEW": 0,
        "INCLUDE": 1,
        "EXCLUDE": 2,
    }

    df["_ORDER"] = (
        df["REFINED_DECISION"]
        .map(decision_order)
        .fillna(9)
    )

    df = (
        df.sort_values(
            [
                "_ORDER",
                "THEME_PRELIMINARY",
                "SERIES_NAME",
            ]
        )
        .drop(columns="_ORDER")
        .reset_index(drop=True)
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("RAFİNE ÖRNEKLEM LİSTESİ HAZIR")
    print("=" * 70)
    print(f"Toplam ETF: {len(df):,}")
    print()
    print("Önerilen karar dağılımı:")
    print(
        df["REFINED_DECISION"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()
    print("Manuel inceleme gereken ETF sayısı:")
    print(
        int(
            (
                df["REFINED_DECISION"]
                == "MANUAL_REVIEW"
            ).sum()
        )
    )
    print()
    print(f"Dosya kaydedildi:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()