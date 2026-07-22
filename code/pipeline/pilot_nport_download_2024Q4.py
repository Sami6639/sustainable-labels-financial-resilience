from pathlib import Path
import hashlib
import sys
import zipfile

import requests


PROJECT_DIR = Path.home() / "Desktop" / "CPU_Project"
RAW_DIR = PROJECT_DIR / "sec_raw" / "2024Q4"
LOG_DIR = PROJECT_DIR / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ZIP_URL = (
    "https://www.sec.gov/files/dera/data/"
    "form-n-port-data-sets/2024q4_nport.zip"
)

ZIP_FILE = RAW_DIR / "2024q4_nport.zip"
EXTRACT_DIR = RAW_DIR / "extracted"

HEADERS = {
    "User-Agent": (
        "Sami Kucukoglu academic research "
        "samikucukoglu@yahoo.com"
    ),
    "Accept-Encoding": "gzip, deflate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_file() -> None:
    if ZIP_FILE.exists() and ZIP_FILE.stat().st_size > 100_000_000:
        print("ZIP dosyası daha önce indirilmiş; tekrar indirilmiyor.")
        return

    print("2024 Q4 N-PORT paketi indiriliyor...")
    print("Dosya birkaç yüz MB olabilir.")

    with requests.get(
        ZIP_URL,
        headers=HEADERS,
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with ZIP_FILE.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                output_file.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    percent = downloaded / total_size * 100
                    print(
                        f"\rİndirilen: {percent:6.2f}%",
                        end="",
                        flush=True,
                    )

    print("\nİndirme tamamlandı.")


def inspect_and_extract() -> None:
    if not zipfile.is_zipfile(ZIP_FILE):
        raise RuntimeError(
            "İndirilen dosya geçerli bir ZIP arşivi değil."
        )

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:
        members = archive.infolist()

        print(f"\nZIP içindeki dosya sayısı: {len(members)}")
        print("\nZIP içeriği:")

        for member in members:
            size_mb = member.file_size / (1024 * 1024)
            print(f"{member.filename:45s} {size_mb:10.2f} MB")

        archive.extractall(EXTRACT_DIR)

    print(f"\nDosyalar açıldı:\n{EXTRACT_DIR}")


def write_inventory() -> None:
    inventory_file = LOG_DIR / "2024q4_nport_inventory.txt"

    files = sorted(
        path
        for path in EXTRACT_DIR.rglob("*")
        if path.is_file()
    )

    with inventory_file.open(
        "w",
        encoding="utf-8",
    ) as output:
        output.write(f"ZIP URL: {ZIP_URL}\n")
        output.write(f"ZIP SHA256: {sha256_file(ZIP_FILE)}\n")
        output.write(f"Extracted files: {len(files)}\n\n")

        for path in files:
            relative_path = path.relative_to(EXTRACT_DIR)
            size_mb = path.stat().st_size / (1024 * 1024)

            output.write(
                f"{relative_path}\t{size_mb:.4f} MB\n"
            )

    print(f"\nEnvanter kaydedildi:\n{inventory_file}")


def main() -> None:
    download_file()
    inspect_and_extract()
    write_inventory()

    print("\n2024Q4 N-PORT İŞLEMİ BAŞARILI.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nHATA: {type(exc).__name__}: {exc}")
        sys.exit(1)