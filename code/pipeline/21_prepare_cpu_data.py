from pathlib import Path
import io
import warnings
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# ============================================================
# 21_prepare_cpu_data.py
#
# AMAÇ
# ----
# Resmî Climate Policy Uncertainty veri setini indirir,
# DATE sütunundaki 1985M01 biçimini doğru ayrıştırır ve
# 2010–2025 aylık CPU veri panelini oluşturur.
#
# Ana değişken:
# CPU_INDEX_NARROW
#
# Çıktılar:
# - CPU
# - LOG_CPU
# - CPU_Z
# - LOG_CPU_Z
# - CPU_DIFF
# - CPU_CHANGE
# - gecikmeli CPU değişkenleri
# - düşük, yüksek ve aşırı CPU rejimleri
# - CPU shock ve instrument serileri
# ============================================================


# ============================================================
# 1. PROJE YOLLARI
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\User\Desktop\CPU_Project"
)

OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. VERİ KAYNAĞI
# ============================================================

CPU_URLS = [
    (
        "https://raw.githubusercontent.com/"
        "dkaenzig/Climate-Policy-Uncertainty-Index/"
        "main/cpu_2026-05.xlsx"
    ),
    (
        "https://github.com/"
        "dkaenzig/Climate-Policy-Uncertainty-Index/"
        "raw/main/cpu_2026-05.xlsx"
    ),
]

USER_AGENT = (
    "Sami Kucukoglu academic research "
    "samikucukoglu@yahoo.com"
)

DOWNLOAD_TIMEOUT_SECONDS = 120


# ============================================================
# 3. AMPİRİK DÖNEM
# ============================================================

START_DATE = pd.Timestamp(
    "2010-01-31"
)

END_DATE = pd.Timestamp(
    "2025-12-31"
)


# ============================================================
# 4. DOSYA YOLLARI
# ============================================================

RAW_EXCEL_FILE = (
    DATA_DIR
    / "cpu_official_raw.xlsx"
)

RAW_CSV_FILE = (
    OUTPUT_DIR
    / "21_cpu_official_raw.csv"
)

MONTHLY_CPU_FILE = (
    OUTPUT_DIR
    / "21_monthly_cpu.csv"
)

MONTHLY_CPU_PARQUET_FILE = (
    OUTPUT_DIR
    / "21_monthly_cpu.parquet"
)

CPU_DESCRIPTIVE_FILE = (
    OUTPUT_DIR
    / "21_cpu_descriptive_statistics.csv"
)

CPU_REGIME_FILE = (
    OUTPUT_DIR
    / "21_cpu_regime_summary.csv"
)

COLUMN_AUDIT_FILE = (
    OUTPUT_DIR
    / "21_cpu_column_audit.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "21_cpu_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "21_cpu_variable_dictionary.csv"
)


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


# ============================================================
# 6. TARİH AYRIŞTIRMA
# ============================================================

def parse_cpu_date_value(
    value,
):
    """
    CPU veri setindeki aylık tarihleri ayrıştırır.

    Desteklenen örnekler:
    - 1985M01
    - 1985m01
    - 198501
    - 1985-01
    - 1985/01
    - standart Excel/Pandas tarihleri

    Bütün tarihler ay sonuna dönüştürülür.
    """

    if pd.isna(value):
        return pd.NaT

    text = str(value).strip().upper()

    if text in {
        "",
        "NAN",
        "NONE",
        "NULL",
        "<NA>",
    }:
        return pd.NaT

    # Örnek: 1985M01
    if (
        len(text) >= 7
        and "M" in text
    ):

        parts = text.split(
            "M",
            maxsplit=1,
        )

        try:

            year = int(
                parts[0]
            )

            month_text = (
                parts[1]
                .replace(
                    ".0",
                    "",
                )
                .strip()
            )

            month = int(
                month_text[:2]
            )

            date = pd.Timestamp(
                year=year,
                month=month,
                day=1,
            )

            return (
                date
                + pd.offsets.MonthEnd(0)
            )

        except (
            ValueError,
            TypeError,
        ):
            pass

    # Örnek: 198501
    digits_only = (
        text
        .replace(
            ".0",
            "",
        )
    )

    if (
        len(digits_only) == 6
        and digits_only.isdigit()
    ):

        try:

            year = int(
                digits_only[:4]
            )

            month = int(
                digits_only[4:6]
            )

            date = pd.Timestamp(
                year=year,
                month=month,
                day=1,
            )

            return (
                date
                + pd.offsets.MonthEnd(0)
            )

        except (
            ValueError,
            TypeError,
        ):
            pass

    # Örnek: 1985-01 veya 1985/01
    for separator in [
        "-",
        "/",
    ]:

        if separator in text:

            parts = text.split(
                separator
            )

            if len(parts) >= 2:

                try:

                    year = int(
                        parts[0]
                    )

                    month = int(
                        parts[1][:2]
                    )

                    date = pd.Timestamp(
                        year=year,
                        month=month,
                        day=1,
                    )

                    return (
                        date
                        + pd.offsets.MonthEnd(0)
                    )

                except (
                    ValueError,
                    TypeError,
                ):
                    pass

    # Son genel deneme
    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return pd.NaT

    return (
        parsed.to_period("M")
        .to_timestamp("M")
    )


def construct_date(
    series: pd.Series,
) -> pd.Series:
    """
    DATE sütununu ay sonu tarihine dönüştürür.
    """

    return series.apply(
        parse_cpu_date_value
    )


# ============================================================
# 7. VERİ İNDİRME
# ============================================================

def download_cpu_excel() -> bytes:
    """
    CPU Excel dosyasını indirir.

    Çevrim içi indirme başarısız olursa daha önce indirilen yerel
    Excel dosyasını kullanır.
    """

    errors = []

    for url in CPU_URLS:

        print(
            f"  Denenen kaynak: {url}"
        )

        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": (
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet,"
                    "application/octet-stream,*/*"
                ),
            },
        )

        try:

            with urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:

                raw_bytes = response.read()

            if not raw_bytes.startswith(
                b"PK"
            ):
                raise RuntimeError(
                    "İndirilen içerik geçerli XLSX biçiminde değil."
                )

            RAW_EXCEL_FILE.write_bytes(
                raw_bytes
            )

            print(
                f"  İndirme başarılı: "
                f"{len(raw_bytes):,} byte"
            )

            return raw_bytes

        except (
            HTTPError,
            URLError,
            TimeoutError,
            RuntimeError,
            OSError,
        ) as error:

            errors.append(
                f"{url}: {error}"
            )

    if RAW_EXCEL_FILE.exists():

        print(
            "\nÇevrim içi indirme başarısız."
        )

        print(
            "Yerel CPU Excel dosyası kullanılıyor."
        )

        raw_bytes = (
            RAW_EXCEL_FILE.read_bytes()
        )

        if not raw_bytes.startswith(
            b"PK"
        ):
            raise RuntimeError(
                "Yerel CPU Excel dosyası geçerli değil."
            )

        return raw_bytes

    raise RuntimeError(
        "CPU veri seti indirilemedi.\n"
        + "\n".join(
            errors
        )
    )


# ============================================================
# 8. EXCEL DOSYASINI OKUMA
# ============================================================

def read_cpu_excel(
    raw_bytes: bytes,
) -> tuple[
    pd.DataFrame,
    str,
]:
    """
    CPU endeksinin bulunduğu Excel sayfasını seçer.
    """

    excel_buffer = io.BytesIO(
        raw_bytes
    )

    excel_file = pd.ExcelFile(
        excel_buffer,
        engine="openpyxl",
    )

    candidates = []

    for sheet_name in excel_file.sheet_names:

        excel_buffer.seek(0)

        try:

            data = pd.read_excel(
                excel_buffer,
                sheet_name=sheet_name,
                engine="openpyxl",
            )

            data = normalize_columns(
                data
            )

            score = 0

            if "DATE" in data.columns:
                score += 5

            if (
                "CPU_INDEX_NARROW"
                in data.columns
            ):
                score += 10

            if len(data) >= 100:
                score += 1

            candidates.append(
                {
                    "SHEET": sheet_name,
                    "DATA": data,
                    "SCORE": score,
                    "ROWS": len(data),
                }
            )

        except Exception as error:

            print(
                f"  Sayfa okunamadı: "
                f"{sheet_name} | {error}"
            )

    if not candidates:

        raise RuntimeError(
            "Excel içinde okunabilir sayfa bulunamadı."
        )

    selected = sorted(
        candidates,
        key=lambda item: (
            item["SCORE"],
            item["ROWS"],
        ),
        reverse=True,
    )[0]

    data = selected[
        "DATA"
    ]

    if (
        "DATE" not in data.columns
        or "CPU_INDEX_NARROW"
        not in data.columns
    ):

        raise RuntimeError(
            "DATE ve CPU_INDEX_NARROW sütunlarını birlikte "
            "içeren sayfa bulunamadı."
        )

    return (
        data,
        selected["SHEET"],
    )


# ============================================================
# 9. CPU PANELİNİ HAZIRLAMA
# ============================================================

def prepare_cpu_panel(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aylık CPU panelini oluşturur.
    """

    data = normalize_columns(
        raw_data
    )

    required_columns = [
        "DATE",
        "CPU_INDEX_NARROW",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise KeyError(
            "CPU veri setinde eksik sütunlar:\n"
            + "\n".join(
                missing_columns
            )
        )

    panel = pd.DataFrame(
        {
            "DATE": construct_date(
                data["DATE"]
            ),

            "CPU": safe_numeric(
                data[
                    "CPU_INDEX_NARROW"
                ]
            ),
        }
    )

    optional_columns = {
        "CPU_BROAD": (
            "CPU_INDEX_BROAD"
        ),
        "CPU_LLM": (
            "CPU_INDEX_NARROW_LLM"
        ),
        "CPNEWS_NARROW": (
            "CPNEWS_INDEX_NARROW"
        ),
        "CPNEWS_BROAD": (
            "CPNEWS_INDEX_BROAD"
        ),
        "CPSENT": (
            "CPSENT_INDEX"
        ),
        "CPU_INSTRUMENT": (
            "CPU_INSTRUMENT"
        ),
        "CPU_SHOCK": (
            "CPU_SHOCK"
        ),
    }

    for output_name, source_name in (
        optional_columns.items()
    ):

        if source_name in data.columns:

            panel[
                output_name
            ] = safe_numeric(
                data[
                    source_name
                ]
            )

        else:

            panel[
                output_name
            ] = np.nan

    panel = panel.dropna(
        subset=[
            "DATE",
            "CPU",
        ]
    )

    panel = (
        panel.groupby(
            "DATE",
            as_index=False,
        )
        .mean(
            numeric_only=True
        )
        .sort_values(
            "DATE"
        )
        .reset_index(
            drop=True
        )
    )

    panel = panel.loc[
        panel["DATE"].between(
            START_DATE,
            END_DATE,
        )
    ].copy()

    if panel.empty:

        raise RuntimeError(
            "2010–2025 döneminde CPU gözlemi bulunamadı."
        )

    # --------------------------------------------------------
    # CPU dönüşümleri
    # --------------------------------------------------------

    panel[
        "LOG_CPU"
    ] = np.log1p(
        panel["CPU"].clip(
            lower=0
        )
    )

    panel[
        "CPU_Z"
    ] = zscore(
        panel["CPU"]
    )

    panel[
        "LOG_CPU_Z"
    ] = zscore(
        panel["LOG_CPU"]
    )

    panel[
        "CPU_DIFF"
    ] = panel[
        "CPU"
    ].diff()

    panel[
        "CPU_CHANGE"
    ] = panel[
        "CPU"
    ].pct_change(
        fill_method=None
    )

    panel[
        "CPU_L1"
    ] = panel[
        "CPU"
    ].shift(1)

    panel[
        "CPU_Z_L1"
    ] = panel[
        "CPU_Z"
    ].shift(1)

    panel[
        "CPU_DIFF_L1"
    ] = panel[
        "CPU_DIFF"
    ].shift(1)

    panel[
        "CPU_L2"
    ] = panel[
        "CPU"
    ].shift(2)

    panel[
        "CPU_Z_L2"
    ] = panel[
        "CPU_Z"
    ].shift(2)

    # --------------------------------------------------------
    # CPU rejimleri
    # --------------------------------------------------------

    cpu_p25 = float(
        panel["CPU"].quantile(
            0.25
        )
    )

    cpu_median = float(
        panel["CPU"].median()
    )

    cpu_p75 = float(
        panel["CPU"].quantile(
            0.75
        )
    )

    cpu_p90 = float(
        panel["CPU"].quantile(
            0.90
        )
    )

    panel[
        "LOW_CPU_REGIME"
    ] = (
        panel["CPU"]
        <= cpu_p25
    ).astype(
        int
    )

    panel[
        "HIGH_CPU_REGIME"
    ] = (
        panel["CPU"]
        >= cpu_p75
    ).astype(
        int
    )

    panel[
        "EXTREME_CPU_REGIME"
    ] = (
        panel["CPU"]
        >= cpu_p90
    ).astype(
        int
    )

    panel[
        "CPU_ABOVE_MEDIAN"
    ] = (
        panel["CPU"]
        >= cpu_median
    ).astype(
        int
    )

    panel[
        "CPU_REGIME"
    ] = pd.cut(
        panel["CPU"],
        bins=[
            -np.inf,
            cpu_p25,
            cpu_p75,
            np.inf,
        ],
        labels=[
            "LOW",
            "MIDDLE",
            "HIGH",
        ],
        include_lowest=True,
    ).astype(
        "string"
    )

    panel[
        "YEAR"
    ] = panel[
        "DATE"
    ].dt.year

    panel[
        "MONTH"
    ] = panel[
        "DATE"
    ].dt.month

    panel[
        "YEAR_MONTH"
    ] = (
        panel["DATE"]
        .dt.to_period("M")
        .astype(str)
    )

    return panel


# ============================================================
# 10. RAPORLAR
# ============================================================

def build_column_audit(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ham veri sütun audit raporu.
    """

    rows = []

    for column in raw_data.columns:

        rows.append(
            {
                "COLUMN": column,

                "DTYPE": str(
                    raw_data[
                        column
                    ].dtype
                ),

                "ROWS": len(
                    raw_data
                ),

                "NON_MISSING": int(
                    raw_data[
                        column
                    ]
                    .notna()
                    .sum()
                ),

                "COVERAGE_RATE": float(
                    raw_data[
                        column
                    ]
                    .notna()
                    .mean()
                ),

                "UNIQUE_VALUES": int(
                    raw_data[
                        column
                    ]
                    .nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_descriptive_statistics(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    CPU tanımlayıcı istatistikleri.
    """

    variables = [
        "CPU",
        "LOG_CPU",
        "CPU_Z",
        "CPU_DIFF",
        "CPU_CHANGE",
        "CPU_BROAD",
        "CPU_LLM",
        "CPNEWS_NARROW",
        "CPNEWS_BROAD",
        "CPSENT",
        "CPU_INSTRUMENT",
        "CPU_SHOCK",
    ]

    rows = []

    for variable in variables:

        values = safe_numeric(
            panel[
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

                "P90": float(
                    values.quantile(
                        0.90
                    )
                ),

                "MAX": float(
                    values.max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_regime_summary(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    CPU rejim özetini üretir.
    """

    return (
        panel.groupby(
            "CPU_REGIME",
            dropna=False,
            observed=True,
        )
        .agg(
            MONTHS=(
                "DATE",
                "size",
            ),

            START_DATE=(
                "DATE",
                "min",
            ),

            END_DATE=(
                "DATE",
                "max",
            ),

            CPU_MEAN=(
                "CPU",
                "mean",
            ),

            CPU_MEDIAN=(
                "CPU",
                "median",
            ),

            CPU_MIN=(
                "CPU",
                "min",
            ),

            CPU_MAX=(
                "CPU",
                "max",
            ),
        )
        .reset_index()
    )


def build_variable_dictionary() -> pd.DataFrame:
    """
    CPU değişken sözlüğü.
    """

    return pd.DataFrame(
        [
            {
                "VARIABLE": "CPU",

                "SOURCE": (
                    "Official Climate Policy Uncertainty dataset"
                ),

                "ORIGINAL_VARIABLE": (
                    "CPU_INDEX_NARROW"
                ),

                "FORMULA": (
                    "News-based narrow CPU index"
                ),

                "ROLE": (
                    "Primary uncertainty variable"
                ),
            },

            {
                "VARIABLE": "LOG_CPU",

                "SOURCE": "CPU",

                "ORIGINAL_VARIABLE": pd.NA,

                "FORMULA": "log(1 + CPU)",

                "ROLE": (
                    "Nonlinear robustness"
                ),
            },

            {
                "VARIABLE": "CPU_Z",

                "SOURCE": "CPU",

                "ORIGINAL_VARIABLE": pd.NA,

                "FORMULA": (
                    "(CPU - mean) / standard deviation"
                ),

                "ROLE": (
                    "Standardized interaction variable"
                ),
            },

            {
                "VARIABLE": "CPU_DIFF",

                "SOURCE": "CPU",

                "ORIGINAL_VARIABLE": pd.NA,

                "FORMULA": (
                    "CPU_t - CPU_t-1"
                ),

                "ROLE": (
                    "Monthly CPU innovation proxy"
                ),
            },

            {
                "VARIABLE": "HIGH_CPU_REGIME",

                "SOURCE": "CPU",

                "ORIGINAL_VARIABLE": pd.NA,

                "FORMULA": (
                    "1 when CPU >= 75th percentile"
                ),

                "ROLE": (
                    "Stress activation test"
                ),
            },

            {
                "VARIABLE": "EXTREME_CPU_REGIME",

                "SOURCE": "CPU",

                "ORIGINAL_VARIABLE": pd.NA,

                "FORMULA": (
                    "1 when CPU >= 90th percentile"
                ),

                "ROLE": (
                    "Extreme uncertainty test"
                ),
            },

            {
                "VARIABLE": "CPU_INSTRUMENT",

                "SOURCE": (
                    "Official CPU dataset"
                ),

                "ORIGINAL_VARIABLE": (
                    "CPU_INSTRUMENT"
                ),

                "FORMULA": (
                    "External CPU instrument"
                ),

                "ROLE": (
                    "Identification robustness"
                ),
            },

            {
                "VARIABLE": "CPU_SHOCK",

                "SOURCE": (
                    "Official CPU dataset"
                ),

                "ORIGINAL_VARIABLE": (
                    "CPU_SHOCK"
                ),

                "FORMULA": (
                    "Externally identified CPU shock"
                ),

                "ROLE": (
                    "External shock robustness"
                ),
            },
        ]
    )


# ============================================================
# 11. VALIDATION
# ============================================================

def build_validation(
    panel: pd.DataFrame,
    selected_sheet: str,
) -> pd.DataFrame:
    """
    CPU panelini doğrular.
    """

    expected_months = pd.period_range(
        start="2010-01",
        end="2025-12",
        freq="M",
    )

    observed_periods = set(
        panel["DATE"]
        .dt.to_period("M")
        .tolist()
    )

    missing_months = (
        set(expected_months)
        - observed_periods
    )

    duplicate_months = int(
        panel.duplicated(
            subset=[
                "DATE",
            ],
            keep=False,
        ).sum()
    )

    rows = [
        {
            "CHECK": (
                "SELECTED_EXCEL_SHEET"
            ),

            "VALUE": selected_sheet,

            "PASS": 1,
        },

        {
            "CHECK": "CPU_ROWS",

            "VALUE": len(
                panel
            ),

            "PASS": int(
                len(
                    panel
                ) > 0
            ),
        },

        {
            "CHECK": "UNIQUE_MONTHS",

            "VALUE": panel[
                "DATE"
            ].nunique(),

            "PASS": int(
                panel[
                    "DATE"
                ].nunique()
                == 192
            ),
        },

        {
            "CHECK": (
                "MISSING_EXPECTED_MONTHS"
            ),

            "VALUE": len(
                missing_months
            ),

            "PASS": int(
                len(
                    missing_months
                )
                == 0
            ),
        },

        {
            "CHECK": "DUPLICATE_MONTHS",

            "VALUE": duplicate_months,

            "PASS": int(
                duplicate_months
                == 0
            ),
        },

        {
            "CHECK": "MISSING_CPU_ROWS",

            "VALUE": int(
                panel[
                    "CPU"
                ].isna().sum()
            ),

            "PASS": int(
                panel[
                    "CPU"
                ].isna().sum()
                == 0
            ),
        },

        {
            "CHECK": "NEGATIVE_CPU_ROWS",

            "VALUE": int(
                (
                    panel[
                        "CPU"
                    ] < 0
                ).sum()
            ),

            "PASS": int(
                (
                    panel[
                        "CPU"
                    ] < 0
                ).sum()
                == 0
            ),
        },

        {
            "CHECK": (
                "CPU_STANDARD_DEVIATION"
            ),

            "VALUE": float(
                panel[
                    "CPU"
                ].std(
                    ddof=1
                )
            ),

            "PASS": int(
                panel[
                    "CPU"
                ].std(
                    ddof=1
                )
                > 0
            ),
        },

        {
            "CHECK": "HIGH_CPU_MONTHS",

            "VALUE": int(
                panel[
                    "HIGH_CPU_REGIME"
                ].sum()
            ),

            "PASS": int(
                panel[
                    "HIGH_CPU_REGIME"
                ].sum()
                > 0
            ),
        },

        {
            "CHECK": "EXTREME_CPU_MONTHS",

            "VALUE": int(
                panel[
                    "EXTREME_CPU_REGIME"
                ].sum()
            ),

            "PASS": int(
                panel[
                    "EXTREME_CPU_REGIME"
                ].sum()
                > 0
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 12. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 78)
    print("21 - CLIMATE POLICY UNCERTAINTY DATA")
    print("=" * 78)

    print(
        "\n1/7 - Resmi CPU veri seti indiriliyor..."
    )

    raw_bytes = download_cpu_excel()

    print(
        "\n2/7 - Excel sayfaları inceleniyor..."
    )

    (
        raw_data,
        selected_sheet,
    ) = read_cpu_excel(
        raw_bytes
    )

    print(
        f"Seçilen Excel sayfası: "
        f"{selected_sheet}"
    )

    print(
        f"Ham satır sayısı: "
        f"{len(raw_data):,}"
    )

    print(
        f"Ham sütun sayısı: "
        f"{len(raw_data.columns):,}"
    )

    print(
        "\n3/7 - Ham veri audit raporu hazırlanıyor..."
    )

    raw_data.to_csv(
        RAW_CSV_FILE,
        index=False,
    )

    column_audit = (
        build_column_audit(
            raw_data
        )
    )

    column_audit.to_csv(
        COLUMN_AUDIT_FILE,
        index=False,
    )

    print(
        "\n4/7 - Aylık CPU paneli hazırlanıyor..."
    )

    panel = prepare_cpu_panel(
        raw_data
    )

    print(
        f"2010–2025 CPU ay sayısı: "
        f"{len(panel):,}"
    )

    print(
        f"Başlangıç: "
        f"{panel['DATE'].min().date()}"
    )

    print(
        f"Bitiş: "
        f"{panel['DATE'].max().date()}"
    )

    print(
        "\n5/7 - Tanımlayıcı istatistikler hazırlanıyor..."
    )

    descriptive = (
        build_descriptive_statistics(
            panel
        )
    )

    regime_summary = (
        build_regime_summary(
            panel
        )
    )

    variable_dictionary = (
        build_variable_dictionary()
    )

    print(
        "\n6/7 - Validation kontrolleri çalıştırılıyor..."
    )

    validation = build_validation(
        panel=panel,
        selected_sheet=selected_sheet,
    )

    print(
        "\n7/7 - Çıktılar kaydediliyor..."
    )

    panel.to_csv(
        MONTHLY_CPU_FILE,
        index=False,
    )

    panel.to_parquet(
        MONTHLY_CPU_PARQUET_FILE,
        index=False,
    )

    descriptive.to_csv(
        CPU_DESCRIPTIVE_FILE,
        index=False,
    )

    regime_summary.to_csv(
        CPU_REGIME_FILE,
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
        "\nCPU DATA HAZIR"
    )

    print("=" * 78)

    print(
        "\nAna CPU tanımlayıcı istatistikleri:"
    )

    print(
        descriptive.loc[
            descriptive[
                "VARIABLE"
            ].isin(
                [
                    "CPU",
                    "LOG_CPU",
                    "CPU_DIFF",
                    "CPU_SHOCK",
                    "CPU_INSTRUMENT",
                ]
            )
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nCPU rejim özeti:"
    )

    print(
        regime_summary.to_string(
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
        MONTHLY_CPU_FILE
    )

    print(
        CPU_DESCRIPTIVE_FILE
    )

    print(
        CPU_REGIME_FILE
    )

    print(
        VALIDATION_FILE
    )

    print(
        VARIABLE_DICTIONARY_FILE
    )

    print(
        RAW_EXCEL_FILE
    )


if __name__ == "__main__":
    main()