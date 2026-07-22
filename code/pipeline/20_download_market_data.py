from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 20_download_market_data.py
#
# AMAÇ
# ----
# SEC series ID üzerinden eşleştirilen sürdürülebilir ETF'lerin
# günlük düzeltilmiş fiyatlarını indirir ve aylık getirilerini üretir.
#
# Ayrıca aşağıdaki piyasa kontrol serilerini indirir:
#
# SPY   : Broad U.S. equity market
# XLE   : Energy sector
# TLT   : Long-duration Treasury
# ^VIX  : Implied market volatility
# ^GSPC : S&P 500 index
#
# Çıktılar
# --------
# - ETF günlük fiyat paneli
# - ETF aylık getiri paneli
# - Piyasa kontrol aylık paneli
# - Ticker indirme raporu
# - ETF mapping ve coverage raporları
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

ETF_MAPPING_FILE = (
    OUTPUT_DIR
    / "19a_etf_ticker_mapping.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

DAILY_ETF_PRICES_FILE = (
    OUTPUT_DIR
    / "20_etf_daily_adjusted_prices.parquet"
)

DAILY_ETF_PRICES_CSV_FILE = (
    OUTPUT_DIR
    / "20_etf_daily_adjusted_prices.csv"
)

MONTHLY_ETF_RETURNS_FILE = (
    OUTPUT_DIR
    / "20_monthly_etf_returns.csv"
)

MONTHLY_ETF_RETURNS_PARQUET_FILE = (
    OUTPUT_DIR
    / "20_monthly_etf_returns.parquet"
)

MONTHLY_CONTROLS_FILE = (
    OUTPUT_DIR
    / "20_monthly_market_controls.csv"
)

MONTHLY_CONTROLS_PARQUET_FILE = (
    OUTPUT_DIR
    / "20_monthly_market_controls.parquet"
)

DOWNLOAD_REPORT_FILE = (
    OUTPUT_DIR
    / "20_market_data_download_report.csv"
)

ETF_COVERAGE_FILE = (
    OUTPUT_DIR
    / "20_etf_return_coverage.csv"
)

FAILED_TICKERS_FILE = (
    OUTPUT_DIR
    / "20_failed_market_tickers.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "20_market_data_validation.csv"
)

VARIABLE_DICTIONARY_FILE = (
    OUTPUT_DIR
    / "20_market_data_variable_dictionary.csv"
)


# ============================================================
# 4. TARİH VE İNDİRME AYARLARI
# ============================================================

START_DATE = "2009-12-01"

# 2025 tamamlanmış ampirik dönemini kapsar.
END_DATE = "2026-01-10"

MIN_MONTHLY_OBSERVATIONS = 12

DOWNLOAD_BATCH_SIZE = 15

DOWNLOAD_PAUSE_SECONDS = 2

MAX_DOWNLOAD_ATTEMPTS = 3


# ============================================================
# 5. KONTROL SERİLERİ
# ============================================================

CONTROL_TICKERS = {
    "SPY": "MARKET_RETURN",
    "XLE": "ENERGY_RETURN",
    "TLT": "TREASURY_RETURN",
    "^VIX": "VIX_LEVEL",
    "^GSPC": "SP500_RETURN",
}


# ============================================================
# 6. YARDIMCI FONKSİYONLAR
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


def clean_text(
    series: pd.Series,
) -> pd.Series:
    """
    Metin alanlarını temizler.
    """

    result = (
        series.astype("string")
        .str.strip()
    )

    invalid = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "<NA>",
    }

    return result.mask(
        result.str.upper().isin(invalid),
        pd.NA,
    )


def clean_ticker(
    series: pd.Series,
) -> pd.Series:
    """
    Yahoo Finance için ticker sembollerini temizler.

    Noktalı share-class ticker'ları Yahoo biçimine dönüştürür:
    BRK.B -> BRK-B
    """

    result = (
        clean_text(series)
        .str.upper()
        .str.replace(
            ".",
            "-",
            regex=False,
        )
    )

    return result


def standardize_month_end(
    series: pd.Series,
) -> pd.Series:
    """
    Tarihleri ay sonuna standardize eder.
    """

    dates = pd.to_datetime(
        series,
        errors="coerce",
    )

    return (
        dates
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )


def split_batches(
    items: list[str],
    batch_size: int,
) -> list[list[str]]:
    """
    Ticker listesini küçük indirme gruplarına ayırır.
    """

    return [
        items[index:index + batch_size]
        for index
        in range(
            0,
            len(items),
            batch_size,
        )
    ]


def extract_price_field(
    downloaded: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """
    yfinance çıktısından tek ticker için düzeltilmiş fiyat serisini çıkarır.

    auto_adjust=True kullanıldığı için Close alanı düzeltilmiş fiyattır.
    """

    if downloaded.empty:
        return pd.Series(
            dtype=float
        )

    if isinstance(
        downloaded.columns,
        pd.MultiIndex,
    ):

        # Olası yapı 1:
        # level 0 = Price field, level 1 = ticker
        if (
            "Close"
            in downloaded.columns.get_level_values(0)
            and ticker
            in downloaded.columns.get_level_values(1)
        ):

            return downloaded[
                ("Close", ticker)
            ].copy()

        # Olası yapı 2:
        # level 0 = ticker, level 1 = Price field
        if (
            ticker
            in downloaded.columns.get_level_values(0)
            and "Close"
            in downloaded.columns.get_level_values(1)
        ):

            return downloaded[
                (ticker, "Close")
            ].copy()

        return pd.Series(
            dtype=float
        )

    if "Close" in downloaded.columns:

        return downloaded[
            "Close"
        ].copy()

    return pd.Series(
        dtype=float
    )


def download_ticker_batch(
    tickers: list[str],
) -> pd.DataFrame:
    """
    Bir ticker grubunu indirir.
    """

    return yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=False,
        timeout=60,
    )


def download_with_retries(
    tickers: list[str],
) -> tuple[
    pd.DataFrame,
    list[dict],
]:
    """
    Ticker gruplarını retry mantığıyla indirir.

    Grup indirmesinde veri gelmeyen ticker'lar daha sonra tek tek denenir.
    """

    price_frames = []

    report_rows = []

    batches = split_batches(
        items=tickers,
        batch_size=DOWNLOAD_BATCH_SIZE,
    )

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        print(
            f"  Batch {batch_number}/{len(batches)}: "
            f"{len(batch)} ticker"
        )

        downloaded = pd.DataFrame()

        last_error = None

        for attempt in range(
            1,
            MAX_DOWNLOAD_ATTEMPTS + 1,
        ):

            try:

                downloaded = download_ticker_batch(
                    batch
                )

                if not downloaded.empty:
                    break

            except Exception as error:

                last_error = str(error)

            time.sleep(
                DOWNLOAD_PAUSE_SECONDS
                * attempt
            )

        for ticker in batch:

            price_series = extract_price_field(
                downloaded=downloaded,
                ticker=ticker,
            )

            if price_series.notna().sum() > 0:

                ticker_frame = (
                    price_series
                    .rename("ADJ_CLOSE")
                    .reset_index()
                )

                ticker_frame = ticker_frame.rename(
                    columns={
                        ticker_frame.columns[0]: "DATE"
                    }
                )

                ticker_frame[
                    "TICKER"
                ] = ticker

                price_frames.append(
                    ticker_frame[
                        [
                            "DATE",
                            "TICKER",
                            "ADJ_CLOSE",
                        ]
                    ]
                )

                report_rows.append(
                    {
                        "TICKER": ticker,
                        "DOWNLOAD_METHOD": "BATCH",
                        "SUCCESS": 1,
                        "OBSERVATIONS": int(
                            price_series.notna().sum()
                        ),
                        "START_DATE": (
                            price_series
                            .dropna()
                            .index.min()
                        ),
                        "END_DATE": (
                            price_series
                            .dropna()
                            .index.max()
                        ),
                        "ERROR": pd.NA,
                    }
                )

            else:

                report_rows.append(
                    {
                        "TICKER": ticker,
                        "DOWNLOAD_METHOD": "BATCH",
                        "SUCCESS": 0,
                        "OBSERVATIONS": 0,
                        "START_DATE": pd.NaT,
                        "END_DATE": pd.NaT,
                        "ERROR": last_error,
                    }
                )

        time.sleep(
            DOWNLOAD_PAUSE_SECONDS
        )

    initial_report = pd.DataFrame(
        report_rows
    )

    failed_tickers = (
        initial_report.loc[
            initial_report[
                "SUCCESS"
            ] == 0,
            "TICKER",
        ]
        .drop_duplicates()
        .tolist()
    )

    retry_rows = []

    for counter, ticker in enumerate(
        failed_tickers,
        start=1,
    ):

        print(
            f"  Tekli retry {counter}/{len(failed_tickers)}: "
            f"{ticker}"
        )

        successful = False

        last_error = None

        for attempt in range(
            1,
            MAX_DOWNLOAD_ATTEMPTS + 1,
        ):

            try:

                downloaded = download_ticker_batch(
                    [ticker]
                )

                price_series = extract_price_field(
                    downloaded=downloaded,
                    ticker=ticker,
                )

                if price_series.notna().sum() > 0:

                    ticker_frame = (
                        price_series
                        .rename("ADJ_CLOSE")
                        .reset_index()
                    )

                    ticker_frame = ticker_frame.rename(
                        columns={
                            ticker_frame.columns[0]: "DATE"
                        }
                    )

                    ticker_frame[
                        "TICKER"
                    ] = ticker

                    price_frames.append(
                        ticker_frame[
                            [
                                "DATE",
                                "TICKER",
                                "ADJ_CLOSE",
                            ]
                        ]
                    )

                    retry_rows.append(
                        {
                            "TICKER": ticker,
                            "DOWNLOAD_METHOD": "SINGLE_RETRY",
                            "SUCCESS": 1,
                            "OBSERVATIONS": int(
                                price_series.notna().sum()
                            ),
                            "START_DATE": (
                                price_series
                                .dropna()
                                .index.min()
                            ),
                            "END_DATE": (
                                price_series
                                .dropna()
                                .index.max()
                            ),
                            "ERROR": pd.NA,
                        }
                    )

                    successful = True

                    break

            except Exception as error:

                last_error = str(error)

            time.sleep(
                DOWNLOAD_PAUSE_SECONDS
                * attempt
            )

        if not successful:

            retry_rows.append(
                {
                    "TICKER": ticker,
                    "DOWNLOAD_METHOD": "SINGLE_RETRY",
                    "SUCCESS": 0,
                    "OBSERVATIONS": 0,
                    "START_DATE": pd.NaT,
                    "END_DATE": pd.NaT,
                    "ERROR": last_error,
                }
            )

        time.sleep(
            DOWNLOAD_PAUSE_SECONDS
        )

    if price_frames:

        daily_prices = pd.concat(
            price_frames,
            ignore_index=True,
        )

        daily_prices["DATE"] = pd.to_datetime(
            daily_prices["DATE"],
            errors="coerce",
        )

        daily_prices["ADJ_CLOSE"] = pd.to_numeric(
            daily_prices["ADJ_CLOSE"],
            errors="coerce",
        )

        daily_prices = (
            daily_prices
            .dropna(
                subset=[
                    "DATE",
                    "TICKER",
                    "ADJ_CLOSE",
                ]
            )
            .drop_duplicates(
                subset=[
                    "DATE",
                    "TICKER",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "TICKER",
                    "DATE",
                ]
            )
        )

    else:

        daily_prices = pd.DataFrame(
            columns=[
                "DATE",
                "TICKER",
                "ADJ_CLOSE",
            ]
        )

    final_report_rows = []

    all_reports = pd.concat(
        [
            initial_report,
            pd.DataFrame(retry_rows),
        ],
        ignore_index=True,
    )

    for ticker in tickers:

        ticker_attempts = all_reports.loc[
            all_reports[
                "TICKER"
            ] == ticker
        ]

        successful_attempts = ticker_attempts.loc[
            ticker_attempts[
                "SUCCESS"
            ] == 1
        ]

        if not successful_attempts.empty:

            selected = successful_attempts.iloc[
                -1
            ]

            final_report_rows.append(
                {
                    "TICKER": ticker,
                    "SUCCESS": 1,
                    "FINAL_DOWNLOAD_METHOD": selected[
                        "DOWNLOAD_METHOD"
                    ],
                    "OBSERVATIONS": selected[
                        "OBSERVATIONS"
                    ],
                    "START_DATE": selected[
                        "START_DATE"
                    ],
                    "END_DATE": selected[
                        "END_DATE"
                    ],
                    "ERROR": pd.NA,
                }
            )

        else:

            selected = ticker_attempts.iloc[
                -1
            ]

            final_report_rows.append(
                {
                    "TICKER": ticker,
                    "SUCCESS": 0,
                    "FINAL_DOWNLOAD_METHOD": selected[
                        "DOWNLOAD_METHOD"
                    ],
                    "OBSERVATIONS": 0,
                    "START_DATE": pd.NaT,
                    "END_DATE": pd.NaT,
                    "ERROR": selected[
                        "ERROR"
                    ],
                }
            )

    return (
        daily_prices,
        final_report_rows,
    )


# ============================================================
# 7. ETF MAPPING
# ============================================================

def prepare_etf_mapping(
    path: Path,
) -> pd.DataFrame:
    """
    SEC ETF ticker mapping dosyasını hazırlar.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"ETF mapping dosyası bulunamadı:\n{path}"
        )

    mapping = normalize_columns(
        pd.read_csv(
            path,
            low_memory=False,
        )
    )

    required_columns = [
        "ETF_ID",
        "ETF_NAME",
        "ETF_TICKER",
        "TICKER_MATCHED",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column not in mapping.columns
    ]

    if missing_columns:

        raise KeyError(
            "ETF mapping dosyasında eksik sütunlar:\n"
            + "\n".join(
                missing_columns
            )
        )

    mapping = mapping.loc[
        mapping[
            "TICKER_MATCHED"
        ] == 1
    ].copy()

    mapping[
        "ETF_TICKER_ORIGINAL"
    ] = clean_ticker(
        mapping[
            "ETF_TICKER"
        ]
    )

    mapping[
        "YAHOO_TICKER"
    ] = clean_ticker(
        mapping[
            "ETF_TICKER"
        ]
    )

    mapping = mapping.dropna(
        subset=[
            "YAHOO_TICKER",
        ]
    )

    mapping = mapping.drop_duplicates(
        subset=[
            "ETF_ID",
        ]
    )

    if mapping[
        "YAHOO_TICKER"
    ].duplicated().any():

        duplicates = mapping.loc[
            mapping[
                "YAHOO_TICKER"
            ].duplicated(
                keep=False
            ),
            [
                "ETF_ID",
                "ETF_NAME",
                "YAHOO_TICKER",
            ],
        ]

        raise RuntimeError(
            "Aynı ticker birden fazla ETF_ID ile eşleşti:\n"
            + duplicates.to_string(
                index=False
            )
        )

    return mapping


# ============================================================
# 8. AYLIK GETİRİLER
# ============================================================

def create_monthly_returns(
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Günlük düzeltilmiş fiyatlardan ay sonu fiyatları ve aylık
    decimal getirileri üretir.
    """

    prices = daily_prices.copy()

    prices[
        "MONTH_END"
    ] = standardize_month_end(
        prices[
            "DATE"
        ]
    )

    monthly_prices = (
        prices.sort_values(
            [
                "TICKER",
                "DATE",
            ]
        )
        .groupby(
            [
                "TICKER",
                "MONTH_END",
            ],
            as_index=False,
        )
        .tail(
            1
        )
        [
            [
                "TICKER",
                "MONTH_END",
                "ADJ_CLOSE",
            ]
        ]
        .rename(
            columns={
                "MONTH_END": "DATE",
                "ADJ_CLOSE": "MONTH_END_ADJ_CLOSE",
            }
        )
        .sort_values(
            [
                "TICKER",
                "DATE",
            ]
        )
    )

    monthly_prices[
        "ETF_RETURN"
    ] = (
        monthly_prices.groupby(
            "TICKER"
        )[
            "MONTH_END_ADJ_CLOSE"
        ]
        .pct_change(
            fill_method=None
        )
    )

    return monthly_prices


# ============================================================
# 9. KONTROL PANELİ
# ============================================================

def build_monthly_controls(
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    SPY, XLE, TLT, VIX ve S&P 500 kontrol panelini oluşturur.
    """

    controls = monthly_returns.loc[
        monthly_returns[
            "TICKER"
        ].isin(
            CONTROL_TICKERS.keys()
        )
    ].copy()

    return_frames = []

    # Return tabanlı kontroller
    for ticker in [
        "SPY",
        "XLE",
        "TLT",
        "^GSPC",
    ]:

        output_name = CONTROL_TICKERS[
            ticker
        ]

        ticker_data = controls.loc[
            controls[
                "TICKER"
            ] == ticker,
            [
                "DATE",
                "ETF_RETURN",
            ],
        ].rename(
            columns={
                "ETF_RETURN": output_name,
            }
        )

        return_frames.append(
            ticker_data
        )

    if return_frames:

        control_panel = return_frames[
            0
        ]

        for frame in return_frames[
            1:
        ]:

            control_panel = control_panel.merge(
                frame,
                on="DATE",
                how="outer",
                validate="one_to_one",
            )

    else:

        control_panel = pd.DataFrame(
            columns=[
                "DATE",
            ]
        )

    # VIX getiri değil, ay sonu seviye olarak kullanılacak
    vix = controls.loc[
        controls[
            "TICKER"
        ] == "^VIX",
        [
            "DATE",
            "MONTH_END_ADJ_CLOSE",
        ],
    ].rename(
        columns={
            "MONTH_END_ADJ_CLOSE": "VIX_LEVEL",
        }
    )

    control_panel = control_panel.merge(
        vix,
        on="DATE",
        how="outer",
        validate="one_to_one",
    )

    control_panel = control_panel.sort_values(
        "DATE"
    )

    control_panel[
        "VIX_CHANGE"
    ] = control_panel[
        "VIX_LEVEL"
    ].pct_change(
        fill_method=None
    )

    return control_panel


# ============================================================
# 10. COVERAGE RAPORU
# ============================================================

def build_etf_coverage(
    monthly_etf_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    ETF bazında getiri coverage raporu üretir.
    """

    rows = []

    for ticker, group in (
        monthly_etf_returns.groupby(
            "TICKER",
            dropna=False,
        )
    ):

        valid_returns = group.loc[
            group[
                "ETF_RETURN"
            ].notna()
        ]

        rows.append(
            {
                "YAHOO_TICKER": ticker,
                "N_MONTH_END_PRICES": int(
                    group[
                        "MONTH_END_ADJ_CLOSE"
                    ].notna().sum()
                ),
                "N_MONTHLY_RETURNS": int(
                    valid_returns.shape[0]
                ),
                "START_DATE": (
                    valid_returns[
                        "DATE"
                    ].min()
                ),
                "END_DATE": (
                    valid_returns[
                        "DATE"
                    ].max()
                ),
                "AT_LEAST_12_MONTHS": int(
                    valid_returns.shape[0]
                    >= MIN_MONTHLY_OBSERVATIONS
                ),
                "AT_LEAST_24_MONTHS": int(
                    valid_returns.shape[0]
                    >= 24
                ),
                "AT_LEAST_60_MONTHS": int(
                    valid_returns.shape[0]
                    >= 60
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. VARIABLE DICTIONARY
# ============================================================

def build_variable_dictionary() -> pd.DataFrame:
    """
    Piyasa verisi değişken sözlüğünü oluşturur.
    """

    return pd.DataFrame(
        [
            {
                "VARIABLE": "ETF_RETURN",
                "SOURCE": "Yahoo Finance via yfinance",
                "FREQUENCY": "Monthly",
                "FORMULA": (
                    "Month-end adjusted close percentage change"
                ),
                "ROLE": "Dependent variable",
            },
            {
                "VARIABLE": "MARKET_RETURN",
                "SOURCE": "SPY",
                "FREQUENCY": "Monthly",
                "FORMULA": (
                    "Month-end adjusted close percentage change"
                ),
                "ROLE": "Broad market control",
            },
            {
                "VARIABLE": "ENERGY_RETURN",
                "SOURCE": "XLE",
                "FREQUENCY": "Monthly",
                "FORMULA": (
                    "Month-end adjusted close percentage change"
                ),
                "ROLE": "Energy-sector control",
            },
            {
                "VARIABLE": "TREASURY_RETURN",
                "SOURCE": "TLT",
                "FREQUENCY": "Monthly",
                "FORMULA": (
                    "Month-end adjusted close percentage change"
                ),
                "ROLE": "Long-duration Treasury control",
            },
            {
                "VARIABLE": "VIX_LEVEL",
                "SOURCE": "^VIX",
                "FREQUENCY": "Monthly",
                "FORMULA": "Month-end closing level",
                "ROLE": "Market-stress state variable",
            },
            {
                "VARIABLE": "VIX_CHANGE",
                "SOURCE": "^VIX",
                "FREQUENCY": "Monthly",
                "FORMULA": (
                    "Monthly percentage change in VIX level"
                ),
                "ROLE": "Market-stress control",
            },
            {
                "VARIABLE": "SP500_RETURN",
                "SOURCE": "^GSPC",
                "FREQUENCY": "Monthly",
                "FORMULA": (
                    "Month-end index level percentage change"
                ),
                "ROLE": "Alternative market control",
            },
        ]
    )


# ============================================================
# 12. VALIDATION
# ============================================================

def build_validation(
    mapping: pd.DataFrame,
    daily_etf_prices: pd.DataFrame,
    monthly_etf_returns: pd.DataFrame,
    controls: pd.DataFrame,
    coverage: pd.DataFrame,
    failed_tickers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Piyasa veri panelinin mekanik kontrollerini yapar.
    """

    duplicated_daily = int(
        daily_etf_prices.duplicated(
            subset=[
                "DATE",
                "TICKER",
            ],
            keep=False,
        ).sum()
    )

    duplicated_monthly = int(
        monthly_etf_returns.duplicated(
            subset=[
                "DATE",
                "TICKER",
            ],
            keep=False,
        ).sum()
    )

    mapped_tickers = mapping[
        "YAHOO_TICKER"
    ].nunique()

    downloaded_etf_tickers = (
        monthly_etf_returns.loc[
            ~monthly_etf_returns[
                "TICKER"
            ].isin(
                CONTROL_TICKERS.keys()
            ),
            "TICKER",
        ].nunique()
    )

    rows = [
        {
            "CHECK": "MAPPED_ETF_TICKERS",
            "VALUE": mapped_tickers,
            "PASS": int(
                mapped_tickers > 0
            ),
        },
        {
            "CHECK": "DOWNLOADED_ETF_TICKERS",
            "VALUE": downloaded_etf_tickers,
            "PASS": int(
                downloaded_etf_tickers > 0
            ),
        },
        {
            "CHECK": "FAILED_ALL_TICKERS",
            "VALUE": len(
                failed_tickers
            ),
            "PASS": int(
                len(failed_tickers)
                < (
                    mapped_tickers
                    + len(
                        CONTROL_TICKERS
                    )
                )
            ),
        },
        {
            "CHECK": "DUPLICATE_DAILY_TICKER_DATE",
            "VALUE": duplicated_daily,
            "PASS": int(
                duplicated_daily == 0
            ),
        },
        {
            "CHECK": "DUPLICATE_MONTHLY_TICKER_DATE",
            "VALUE": duplicated_monthly,
            "PASS": int(
                duplicated_monthly == 0
            ),
        },
        {
            "CHECK": "ETFS_AT_LEAST_12_MONTHS",
            "VALUE": int(
                coverage[
                    "AT_LEAST_12_MONTHS"
                ].sum()
            ),
            "PASS": int(
                coverage[
                    "AT_LEAST_12_MONTHS"
                ].sum()
                > 0
            ),
        },
        {
            "CHECK": "MARKET_RETURN_AVAILABLE",
            "VALUE": int(
                controls[
                    "MARKET_RETURN"
                ].notna().sum()
            ),
            "PASS": int(
                controls[
                    "MARKET_RETURN"
                ].notna().sum()
                > 0
            ),
        },
        {
            "CHECK": "ENERGY_RETURN_AVAILABLE",
            "VALUE": int(
                controls[
                    "ENERGY_RETURN"
                ].notna().sum()
            ),
            "PASS": int(
                controls[
                    "ENERGY_RETURN"
                ].notna().sum()
                > 0
            ),
        },
        {
            "CHECK": "TREASURY_RETURN_AVAILABLE",
            "VALUE": int(
                controls[
                    "TREASURY_RETURN"
                ].notna().sum()
            ),
            "PASS": int(
                controls[
                    "TREASURY_RETURN"
                ].notna().sum()
                > 0
            ),
        },
        {
            "CHECK": "VIX_LEVEL_AVAILABLE",
            "VALUE": int(
                controls[
                    "VIX_LEVEL"
                ].notna().sum()
            ),
            "PASS": int(
                controls[
                    "VIX_LEVEL"
                ].notna().sum()
                > 0
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 13. ANA PIPELINE
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 78)
    print("20 - MARKET DATA DOWNLOAD")
    print("=" * 78)

    # --------------------------------------------------------
    # 1. Mapping
    # --------------------------------------------------------

    print(
        "\n1/8 - SEC ETF ticker mapping okunuyor..."
    )

    mapping = prepare_etf_mapping(
        ETF_MAPPING_FILE
    )

    etf_tickers = (
        mapping[
            "YAHOO_TICKER"
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(
        f"İndirilecek ETF ticker sayısı: "
        f"{len(etf_tickers):,}"
    )

    # --------------------------------------------------------
    # 2. Ticker evreni
    # --------------------------------------------------------

    all_tickers = list(
        dict.fromkeys(
            etf_tickers
            + list(
                CONTROL_TICKERS.keys()
            )
        )
    )

    print(
        f"Kontroller dahil toplam ticker: "
        f"{len(all_tickers):,}"
    )

    # --------------------------------------------------------
    # 3. İndirme
    # --------------------------------------------------------

    print(
        "\n2/8 - Günlük piyasa fiyatları indiriliyor..."
    )

    (
        daily_prices,
        download_report_rows,
    ) = download_with_retries(
        all_tickers
    )

    download_report = pd.DataFrame(
        download_report_rows
    )

    failed_tickers = download_report.loc[
        download_report[
            "SUCCESS"
        ] == 0
    ].copy()

    successful_tickers = download_report.loc[
        download_report[
            "SUCCESS"
        ] == 1
    ].copy()

    print(
        f"Başarılı ticker: "
        f"{len(successful_tickers):,}"
    )

    print(
        f"Başarısız ticker: "
        f"{len(failed_tickers):,}"
    )

    if daily_prices.empty:

        raise RuntimeError(
            "Hiçbir ticker için fiyat verisi indirilemedi."
        )

    # --------------------------------------------------------
    # 4. ETF mapping ekle
    # --------------------------------------------------------

    print(
        "\n3/8 - ETF kimlikleri fiyat paneline ekleniyor..."
    )

    mapping_for_merge = (
        mapping[
            [
                "ETF_ID",
                "ETF_NAME",
                "ETF_TICKER_ORIGINAL",
                "YAHOO_TICKER",
            ]
        ]
        .drop_duplicates(
            subset=[
                "YAHOO_TICKER",
            ]
        )
    )

    daily_prices = daily_prices.merge(
        mapping_for_merge,
        left_on="TICKER",
        right_on="YAHOO_TICKER",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 5. Aylık getiriler
    # --------------------------------------------------------

    print(
        "\n4/8 - Aylık ay-sonu fiyatları ve getiriler hesaplanıyor..."
    )

    monthly_all = create_monthly_returns(
        daily_prices[
            [
                "DATE",
                "TICKER",
                "ADJ_CLOSE",
            ]
        ]
    )

    monthly_etf_returns = monthly_all.loc[
        monthly_all[
            "TICKER"
        ].isin(
            etf_tickers
        )
    ].copy()

    monthly_etf_returns = (
        monthly_etf_returns.merge(
            mapping_for_merge,
            left_on="TICKER",
            right_on="YAHOO_TICKER",
            how="left",
            validate="many_to_one",
        )
    )

    monthly_etf_returns = monthly_etf_returns.rename(
        columns={
            "TICKER": "ETF_TICKER",
        }
    )

    monthly_etf_returns = monthly_etf_returns[
        [
            "DATE",
            "ETF_ID",
            "ETF_NAME",
            "ETF_TICKER",
            "ETF_TICKER_ORIGINAL",
            "MONTH_END_ADJ_CLOSE",
            "ETF_RETURN",
        ]
    ].sort_values(
        [
            "ETF_ID",
            "DATE",
        ]
    )

    # --------------------------------------------------------
    # 6. Kontroller
    # --------------------------------------------------------

    print(
        "\n5/8 - Aylık piyasa kontrol paneli oluşturuluyor..."
    )

    monthly_controls = build_monthly_controls(
        monthly_all
    )

    # --------------------------------------------------------
    # 7. Coverage
    # --------------------------------------------------------

    print(
        "\n6/8 - ETF coverage raporu hazırlanıyor..."
    )

    coverage_input = monthly_all.loc[
        monthly_all[
            "TICKER"
        ].isin(
            etf_tickers
        )
    ].copy()

    etf_coverage = build_etf_coverage(
        coverage_input
    )

    etf_coverage = etf_coverage.merge(
        mapping_for_merge,
        left_on="YAHOO_TICKER",
        right_on="YAHOO_TICKER",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 8. Validation ve kayıt
    # --------------------------------------------------------

    print(
        "\n7/8 - Validation çalıştırılıyor..."
    )

    validation = build_validation(
        mapping=mapping,
        daily_etf_prices=daily_prices,
        monthly_etf_returns=monthly_etf_returns.rename(
            columns={
                "ETF_TICKER": "TICKER",
            }
        ),
        controls=monthly_controls,
        coverage=etf_coverage,
        failed_tickers=failed_tickers,
    )

    variable_dictionary = (
        build_variable_dictionary()
    )

    print(
        "\n8/8 - Çıktılar kaydediliyor..."
    )

    daily_prices.to_parquet(
        DAILY_ETF_PRICES_FILE,
        index=False,
    )

    daily_prices.to_csv(
        DAILY_ETF_PRICES_CSV_FILE,
        index=False,
    )

    monthly_etf_returns.to_csv(
        MONTHLY_ETF_RETURNS_FILE,
        index=False,
    )

    monthly_etf_returns.to_parquet(
        MONTHLY_ETF_RETURNS_PARQUET_FILE,
        index=False,
    )

    monthly_controls.to_csv(
        MONTHLY_CONTROLS_FILE,
        index=False,
    )

    monthly_controls.to_parquet(
        MONTHLY_CONTROLS_PARQUET_FILE,
        index=False,
    )

    download_report.to_csv(
        DOWNLOAD_REPORT_FILE,
        index=False,
    )

    etf_coverage.to_csv(
        ETF_COVERAGE_FILE,
        index=False,
    )

    failed_tickers.to_csv(
        FAILED_TICKERS_FILE,
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
        "\nMARKET DATA HAZIR"
    )

    print("=" * 78)

    print(
        "\nİndirme özeti:"
    )

    print(
        download_report[
            "SUCCESS"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "SUCCESS"
        )
        .reset_index(
            name="TICKERS"
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nETF aylık getiri coverage özeti:"
    )

    print(
        etf_coverage[
            "N_MONTHLY_RETURNS"
        ]
        .describe()
        .to_string()
    )

    print(
        "\nKontrol değişkeni coverage:"
    )

    control_coverage = pd.DataFrame(
        {
            "VARIABLE": [
                "MARKET_RETURN",
                "ENERGY_RETURN",
                "TREASURY_RETURN",
                "VIX_LEVEL",
                "VIX_CHANGE",
                "SP500_RETURN",
            ],
            "NON_MISSING": [
                monthly_controls[
                    variable
                ].notna().sum()
                for variable in [
                    "MARKET_RETURN",
                    "ENERGY_RETURN",
                    "TREASURY_RETURN",
                    "VIX_LEVEL",
                    "VIX_CHANGE",
                    "SP500_RETURN",
                ]
            ],
        }
    )

    print(
        control_coverage.to_string(
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

    if not failed_tickers.empty:

        print(
            "\nİndirilemeyen ticker'lar:"
        )

        print(
            failed_tickers[
                [
                    "TICKER",
                    "ERROR",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\nAna çıktı dosyaları:"
    )

    print(
        MONTHLY_ETF_RETURNS_FILE
    )

    print(
        MONTHLY_CONTROLS_FILE
    )

    print(
        ETF_COVERAGE_FILE
    )

    print(
        DOWNLOAD_REPORT_FILE
    )

    print(
        FAILED_TICKERS_FILE
    )

    print(
        VALIDATION_FILE
    )


if __name__ == "__main__":
    main()