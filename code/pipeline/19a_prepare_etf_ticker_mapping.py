from pathlib import Path
import gzip
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


# ============================================================
# 19a_prepare_etf_ticker_mapping.py
#
# AMAÇ
# ----
# SEC'nin resmi company_tickers_mf.json dosyasını kullanarak
# architecture panelindeki ETF SERIES_ID değerlerini gerçek ETF
# ticker sembolleriyle eşleştirmek.
#
# Ana eşleştirme:
#
# ETF_ID = SEC SERIES_ID
#
# Bu script, holdings içindeki şirket ticker'larını kullanmaz.
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

ARCHITECTURE_FILE = (
    OUTPUT_DIR
    / "18_quality_adjusted_architecture_factors.csv"
)


# ============================================================
# 3. ÇIKTI DOSYALARI
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR
    / "19a_etf_ticker_mapping.csv"
)

UNMATCHED_FILE = (
    OUTPUT_DIR
    / "19a_etf_ticker_mapping_unmatched.csv"
)

MULTIPLE_CLASSES_FILE = (
    OUTPUT_DIR
    / "19a_etf_ticker_multiple_classes.csv"
)

SEC_RAW_FILE = (
    OUTPUT_DIR
    / "19a_sec_company_tickers_mf_raw.json"
)

SEC_PARSED_FILE = (
    OUTPUT_DIR
    / "19a_sec_company_tickers_mf_parsed.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "19a_etf_ticker_mapping_validation.csv"
)


# ============================================================
# 4. SEC AYARLARI
# ============================================================

SEC_URL = (
    "https://www.sec.gov/files/company_tickers_mf.json"
)

# SEC açıklayıcı bir User-Agent kullanılmasını ister.
USER_AGENT = (
    "Sami Kucukoglu academic research "
    "samikucukoglu@yahoo.com"
)

DOWNLOAD_TIMEOUT_SECONDS = 90


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

    invalid_values = {
        "",
        "nan",
        "NaN",
        "NAN",
        "none",
        "NONE",
        "null",
        "NULL",
        "<NA>",
    }

    return result.mask(
        result.isin(invalid_values),
        pd.NA,
    )


def clean_ticker(
    series: pd.Series,
) -> pd.Series:
    """
    Ticker sembollerini temizler ve büyük harfe çevirir.
    """

    return (
        clean_text(series)
        .str.upper()
    )


def normalize_series_id(
    series: pd.Series,
) -> pd.Series:
    """
    SEC series ID değerlerini standartlaştırır.

    Örnek:
        S000062205
    """

    result = (
        clean_text(series)
        .str.upper()
        .str.replace(
            r"\s+",
            "",
            regex=True,
        )
    )

    return result


def decode_sec_response(
    raw_bytes: bytes,
    content_encoding: str,
) -> bytes:
    """
    SEC yanıtı gzip sıkıştırılmışsa açar.

    Content-Encoding başlığı eksik olsa bile gzip magic bytes
    kontrol edilir.
    """

    encoding = (
        content_encoding
        or ""
    ).lower()

    is_gzip = (
        "gzip" in encoding
        or raw_bytes[:2] == b"\x1f\x8b"
    )

    if is_gzip:
        return gzip.decompress(
            raw_bytes
        )

    return raw_bytes


def load_cached_sec_json() -> dict:
    """
    İnternet indirmesi başarısız olursa daha önce kaydedilmiş SEC
    JSON dosyasını okumayı dener.
    """

    if not SEC_RAW_FILE.exists():
        raise FileNotFoundError(
            "Daha önce indirilmiş SEC JSON önbelleği bulunamadı."
        )

    raw_bytes = SEC_RAW_FILE.read_bytes()

    # Eski dosya sıkıştırılmış kaydedilmiş olabilir.
    if raw_bytes[:2] == b"\x1f\x8b":
        raw_bytes = gzip.decompress(
            raw_bytes
        )

    text = raw_bytes.decode(
        "utf-8-sig"
    )

    return json.loads(
        text
    )


def download_sec_json() -> dict:
    """
    SEC mutual-fund ticker JSON dosyasını indirir.

    Yanıt gzip biçiminde gelirse açılır. İndirme başarısız olursa
    mevcut yerel önbellek kullanılmaya çalışılır.
    """

    request = Request(
        SEC_URL,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "application/json,"
                "text/plain,*/*"
            ),
            "Accept-Encoding": (
                "gzip, deflate"
            ),
            "Connection": "close",
        },
    )

    try:

        with urlopen(
            request,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:

            raw_bytes = response.read()

            content_encoding = (
                response.headers.get(
                    "Content-Encoding",
                    "",
                )
            )

        decoded_bytes = decode_sec_response(
            raw_bytes=raw_bytes,
            content_encoding=content_encoding,
        )

        # Okunabilir JSON biçiminde kaydet.
        SEC_RAW_FILE.write_bytes(
            decoded_bytes
        )

        text = decoded_bytes.decode(
            "utf-8-sig"
        )

        return json.loads(
            text
        )

    except (
        HTTPError,
        URLError,
        TimeoutError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
    ) as error:

        print(
            "\nSEC indirme uyarısı:"
        )

        print(
            str(error)
        )

        print(
            "Yerel SEC JSON önbelleği deneniyor..."
        )

        return load_cached_sec_json()


def extract_record_value(
    normalized_record: dict,
    candidate_keys: list[str],
):
    """
    Bir SEC kaydında aday alan adlarından ilk bulunan değeri döndürür.
    """

    for key in candidate_keys:

        if key in normalized_record:

            value = normalized_record[
                key
            ]

            if value not in [
                None,
                "",
            ]:
                return value

    return None


def parse_sec_mutual_fund_json(
    raw_data,
) -> pd.DataFrame:
    """
    SEC mutual-fund JSON yapısını satırlara dönüştürür.

    Beklenen alanlar:
    - cik_str veya cik
    - seriesId
    - classId
    - ticker
    """

    if isinstance(
        raw_data,
        dict,
    ):

        # Bazı JSON sürümlerinde kayıtlar "data" altında olabilir.
        if (
            "data" in raw_data
            and isinstance(
                raw_data["data"],
                list,
            )
        ):

            records = raw_data[
                "data"
            ]

        else:

            records = list(
                raw_data.values()
            )

    elif isinstance(
        raw_data,
        list,
    ):

        records = raw_data

    else:

        raise TypeError(
            "SEC JSON yapısı tanınmadı."
        )

    rows = []

    for record in records:

        # Bazı veri yapıları liste biçiminde gelebilir.
        if isinstance(
            record,
            list,
        ):

            # Beklenen olası sıra:
            # CIK, Series ID, Class ID, Ticker
            if len(record) >= 4:

                rows.append(
                    {
                        "FUND_CIK": record[0],
                        "ETF_ID": record[1],
                        "CLASS_ID": record[2],
                        "ETF_TICKER": record[3],
                    }
                )

            continue

        if not isinstance(
            record,
            dict,
        ):
            continue

        normalized_record = {
            str(key)
            .strip()
            .lower()
            .replace(
                "_",
                "",
            )
            .replace(
                "-",
                "",
            ): value
            for key, value
            in record.items()
        }

        series_id = extract_record_value(
            normalized_record,
            [
                "seriesid",
                "series",
            ],
        )

        class_id = extract_record_value(
            normalized_record,
            [
                "classid",
                "class",
            ],
        )

        ticker = extract_record_value(
            normalized_record,
            [
                "ticker",
                "tickersymbol",
                "symbol",
            ],
        )

        cik = extract_record_value(
            normalized_record,
            [
                "cikstr",
                "cik",
            ],
        )

        if series_id is None:
            continue

        rows.append(
            {
                "FUND_CIK": cik,
                "ETF_ID": series_id,
                "CLASS_ID": class_id,
                "ETF_TICKER": ticker,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise RuntimeError(
            "SEC JSON içinden series/class/ticker kaydı çıkarılamadı."
        )

    result["ETF_ID"] = normalize_series_id(
        result["ETF_ID"]
    )

    result["CLASS_ID"] = (
        clean_text(
            result["CLASS_ID"]
        )
        .str.upper()
    )

    result["ETF_TICKER"] = clean_ticker(
        result["ETF_TICKER"]
    )

    result["FUND_CIK"] = clean_text(
        result["FUND_CIK"]
    )

    result = result.loc[
        result["ETF_ID"].notna()
    ].copy()

    return result


def choose_series_ticker(
    sec_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Her SEC SERIES_ID için deterministik biçimde tek ticker seçer.

    Bir series altında birden fazla ticker varsa:
    - bütün alternatifler ayrı dosyada saklanır,
    - alfabetik ilk geçerli ticker seçilir.
    """

    valid = sec_data.loc[
        sec_data["ETF_ID"].notna()
        & sec_data["ETF_TICKER"].notna()
    ].copy()

    valid = valid.drop_duplicates(
        subset=[
            "ETF_ID",
            "CLASS_ID",
            "ETF_TICKER",
        ]
    )

    ticker_counts = (
        valid.groupby(
            "ETF_ID",
            dropna=False,
        )["ETF_TICKER"]
        .nunique()
        .reset_index(
            name="N_TICKERS_PER_SERIES"
        )
    )

    valid = valid.merge(
        ticker_counts,
        on="ETF_ID",
        how="left",
        validate="many_to_one",
    )

    multiple_classes = valid.loc[
        valid[
            "N_TICKERS_PER_SERIES"
        ] > 1
    ].copy()

    selected = (
        valid.sort_values(
            by=[
                "ETF_ID",
                "ETF_TICKER",
                "CLASS_ID",
            ],
            ascending=[
                True,
                True,
                True,
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "ETF_ID",
            ],
            keep="first",
        )
        .copy()
    )

    return (
        selected,
        multiple_classes,
    )


# ============================================================
# 6. ANA PIPELINE
# ============================================================

def main() -> None:

    print("=" * 76)
    print("19A - OFFICIAL SEC ETF TICKER MAPPING")
    print("=" * 76)

    if not ARCHITECTURE_FILE.exists():

        raise FileNotFoundError(
            "Architecture dosyası bulunamadı:\n"
            f"{ARCHITECTURE_FILE}"
        )

    # --------------------------------------------------------
    # 1. Architecture ETF listesi
    # --------------------------------------------------------

    print(
        "\n1/6 - Architecture ETF listesi okunuyor..."
    )

    architecture = normalize_columns(
        pd.read_csv(
            ARCHITECTURE_FILE,
            low_memory=False,
        )
    )

    required_columns = [
        "ETF_ID",
        "ETF_NAME",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column not in architecture.columns
    ]

    if missing_columns:

        raise KeyError(
            "Architecture dosyasında eksik sütunlar:\n"
            + "\n".join(
                missing_columns
            )
        )

    architecture_base = (
        architecture[
            [
                "ETF_ID",
                "ETF_NAME",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    architecture_base[
        "ETF_ID"
    ] = normalize_series_id(
        architecture_base[
            "ETF_ID"
        ]
    )

    architecture_base[
        "ETF_NAME"
    ] = clean_text(
        architecture_base[
            "ETF_NAME"
        ]
    )

    architecture_base = (
        architecture_base
        .dropna(
            subset=[
                "ETF_ID",
            ]
        )
        .drop_duplicates(
            subset=[
                "ETF_ID",
            ]
        )
    )

    print(
        f"Architecture ETF sayısı: "
        f"{len(architecture_base):,}"
    )

    # --------------------------------------------------------
    # 2. SEC JSON indir
    # --------------------------------------------------------

    print(
        "\n2/6 - SEC mutual-fund ticker dosyası indiriliyor..."
    )

    raw_data = download_sec_json()

    time.sleep(
        0.2
    )

    # --------------------------------------------------------
    # 3. SEC kayıtlarını ayrıştır
    # --------------------------------------------------------

    print(
        "\n3/6 - SEC series/class/ticker kayıtları ayrıştırılıyor..."
    )

    sec_data = parse_sec_mutual_fund_json(
        raw_data
    )

    sec_data.to_csv(
        SEC_PARSED_FILE,
        index=False,
    )

    print(
        f"SEC series-class satırı: "
        f"{len(sec_data):,}"
    )

    print(
        f"SEC benzersiz series sayısı: "
        f"{sec_data['ETF_ID'].nunique():,}"
    )

    print(
        f"SEC ticker bulunan satır: "
        f"{sec_data['ETF_TICKER'].notna().sum():,}"
    )

    # --------------------------------------------------------
    # 4. Her series için tek ticker seç
    # --------------------------------------------------------

    print(
        "\n4/6 - Her SERIES_ID için ticker seçiliyor..."
    )

    (
        selected_tickers,
        multiple_classes,
    ) = choose_series_ticker(
        sec_data
    )

    print(
        f"Ticker taşıyan benzersiz SEC series: "
        f"{selected_tickers['ETF_ID'].nunique():,}"
    )

    # --------------------------------------------------------
    # 5. Architecture ile birleştir
    # --------------------------------------------------------

    print(
        "\n5/6 - Architecture ETF'leri SEC ticker'larıyla eşleştiriliyor..."
    )

    mapping_columns = [
        "ETF_ID",
        "FUND_CIK",
        "CLASS_ID",
        "ETF_TICKER",
        "N_TICKERS_PER_SERIES",
    ]

    result = architecture_base.merge(
        selected_tickers[
            mapping_columns
        ],
        on="ETF_ID",
        how="left",
        validate="one_to_one",
    )

    result[
        "TICKER_MATCHED"
    ] = result[
        "ETF_TICKER"
    ].notna().astype(
        int
    )

    unmatched = result.loc[
        result[
            "TICKER_MATCHED"
        ] == 0
    ].copy()

    # --------------------------------------------------------
    # 6. Validation ve çıktı
    # --------------------------------------------------------

    print(
        "\n6/6 - Çıktılar ve validation kaydediliyor..."
    )

    validation = pd.DataFrame(
        [
            {
                "METRIC": (
                    "TOTAL_ARCHITECTURE_ETFS"
                ),
                "VALUE": len(
                    result
                ),
            },
            {
                "METRIC": (
                    "TICKER_MATCHED_ETFS"
                ),
                "VALUE": int(
                    result[
                        "TICKER_MATCHED"
                    ].sum()
                ),
            },
            {
                "METRIC": (
                    "TICKER_UNMATCHED_ETFS"
                ),
                "VALUE": int(
                    (
                        result[
                            "TICKER_MATCHED"
                        ]
                        == 0
                    ).sum()
                ),
            },
            {
                "METRIC": (
                    "TICKER_MATCH_RATE"
                ),
                "VALUE": float(
                    result[
                        "TICKER_MATCHED"
                    ].mean()
                ),
            },
            {
                "METRIC": (
                    "MATCHED_SERIES_WITH_MULTIPLE_TICKERS"
                ),
                "VALUE": int(
                    (
                        result[
                            "N_TICKERS_PER_SERIES"
                        ].fillna(0)
                        > 1
                    ).sum()
                ),
            },
            {
                "METRIC": (
                    "DUPLICATE_ARCHITECTURE_ETF_IDS"
                ),
                "VALUE": int(
                    result.duplicated(
                        subset=[
                            "ETF_ID",
                        ],
                        keep=False,
                    ).sum()
                ),
            },
            {
                "METRIC": (
                    "DUPLICATE_SELECTED_TICKERS"
                ),
                "VALUE": int(
                    result.loc[
                        result[
                            "ETF_TICKER"
                        ].notna()
                    ]
                    .duplicated(
                        subset=[
                            "ETF_TICKER",
                        ],
                        keep=False,
                    )
                    .sum()
                ),
            },
        ]
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    unmatched.to_csv(
        UNMATCHED_FILE,
        index=False,
    )

    multiple_classes.to_csv(
        MULTIPLE_CLASSES_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    print(
        "\nOFFICIAL SEC ETF TICKER MAPPING HAZIR"
    )

    print("=" * 76)

    print(
        validation.to_string(
            index=False
        )
    )

    print(
        "\nİlk eşleşen ETF'ler:"
    )

    matched_preview = result.loc[
        result[
            "ETF_TICKER"
        ].notna(),
        [
            "ETF_ID",
            "ETF_NAME",
            "ETF_TICKER",
            "CLASS_ID",
        ],
    ].head(
        20
    )

    if matched_preview.empty:

        print(
            "Hiçbir ETF ticker ile eşleşmedi."
        )

    else:

        print(
            matched_preview.to_string(
                index=False
            )
        )

    if not unmatched.empty:

        print(
            "\nEşleşmeyen ilk ETF'ler:"
        )

        print(
            unmatched[
                [
                    "ETF_ID",
                    "ETF_NAME",
                ]
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    print(
        "\nOluşturulan dosyalar:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        UNMATCHED_FILE
    )

    print(
        MULTIPLE_CLASSES_FILE
    )

    print(
        VALIDATION_FILE
    )

    print(
        SEC_RAW_FILE
    )

    print(
        SEC_PARSED_FILE
    )


if __name__ == "__main__":
    main()