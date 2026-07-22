from pathlib import Path
import re
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_DIR = Path.home() / "Desktop" / "CPU_Project"
OUTPUT_DIR = PROJECT_DIR / "output"
LOG_DIR = PROJECT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEC_PAGE = (
    "https://www.sec.gov/data-research/"
    "sec-markets-data/form-n-port-data-sets"
)

# SEC, otomatik isteklerde tanımlayıcı bir User-Agent ister.
HEADERS = {
    "User-Agent": (
        "Sami Kucukoglu academic research "
        "samikucukoglu@yahoo.com"
    ),
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


def main() -> None:
    print("SEC N-PORT veri sayfasına bağlanılıyor...")

    response = requests.get(
        SEC_PAGE,
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    rows = []

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        label = " ".join(link.get_text(" ", strip=True).split())

        if ".zip" not in href.lower():
            continue

        if "nport" not in href.lower() and "n-port" not in href.lower():
            continue

        if href.startswith("/"):
            url = "https://www.sec.gov" + href
        elif href.startswith("http"):
            url = href
        else:
            url = "https://www.sec.gov/" + href.lstrip("/")

        text = f"{label} {href}"

        year_match = re.search(r"\b(20\d{2})\b", text)
        quarter_match = re.search(
            r"(?:Q|q|quarter[-_ ]?)([1-4])",
            text,
        )

        rows.append(
            {
                "label": label,
                "url": url,
                "year": (
                    int(year_match.group(1))
                    if year_match
                    else pd.NA
                ),
                "quarter": (
                    int(quarter_match.group(1))
                    if quarter_match
                    else pd.NA
                ),
            }
        )

    links = pd.DataFrame(rows).drop_duplicates(subset=["url"])

    if links.empty:
        raise RuntimeError(
            "SEC sayfasında N-PORT ZIP bağlantısı bulunamadı."
        )

    links = links.sort_values(
        by=["year", "quarter", "label"],
        na_position="last",
    ).reset_index(drop=True)

    output_file = OUTPUT_DIR / "sec_nport_available_files.csv"
    links.to_csv(output_file, index=False)

    print("\nBağlantı başarılı.")
    print(f"Bulunan ZIP sayısı: {len(links)}")
    print(f"Liste kaydedildi: {output_file}")

    print("\nSon 15 kayıt:")
    print(
        links.tail(15).to_string(
            index=False,
            columns=["year", "quarter", "label", "url"],
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nHATA: {exc}")
        sys.exit(1)