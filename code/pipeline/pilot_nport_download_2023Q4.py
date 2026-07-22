from pathlib import Path
import hashlib
import sys
import zipfile

import requests


PROJECT_DIR = Path.home() / "Desktop" / "CPU_Project"
RAW_DIR = PROJECT_DIR / "sec_raw" / "2023Q4"
LOG_DIR = PROJECT_DIR / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ZIP_URL = (
    "https://www.sec.gov/files/dera/data/"
    "form-n-port-data-sets/2023q4_nport.zip"
)

ZIP_FILE = RAW_DIR / "2023q4_nport.zip"
EXTRACT_DIR = RAW_DIR / "extracted"

HEADERS = {
    "User-Agent": (
        "Sami Kucukoglu academic research "
        "samikucukoglu@yahoo.com"
    ),
    "Accept-Encoding": "gzip, deflate",
}


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_file() -> None:
    """Download the official 2023Q4 SEC N-PORT ZIP package."""
    if ZIP_FILE.exists() and ZIP_FILE.stat().st_size > 100_000_000:
        print("ZIP dosyası daha önce indirilmiş; tekrar indirilmiyor.")
        return

    print("2023 Q4 N-PORT paketi indiriliyor...")
    print("Dosya birkaç yüz MB olabilir.")

    try:
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
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
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

    except Exception:
        if ZIP_FILE.exists() and ZIP_FILE.stat().st_size == 0:
            ZIP_FILE.unlink(missing_ok=True)

        raise


def inspect_and_extract() -> None:
    """Validate and extract the downloaded ZIP package."""
    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"ZIP dosyası bulunamadı: {ZIP_FILE}"
        )

    if not zipfile.is_zipfile(ZIP_FILE):
        raise RuntimeError(
            "İndirilen dosya geçerli bir ZIP arşivi değil."
        )

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:
        bad_member = archive.testzip()

        if bad_member is not None:
            raise RuntimeError(
                f"ZIP bütünlük kontrolü başarısız: {bad_member}"
            )

        members = archive.infolist()

        print(f"\nZIP içindeki dosya sayısı: {len(members)}")
        print("\nZIP içeriği:")

        for member in members:
            size_mb = member.file_size / (1024 * 1024)
            print(
                f"{member.filename:45s} "
                f"{size_mb:10.2f} MB"
            )

        archive.extractall(EXTRACT_DIR)

    print(f"\nDosyalar açıldı:\n{EXTRACT_DIR}")


def validate_required_files() -> None:
    """Confirm that the essential N-PORT tables were extracted."""
    required_files = [
        "SUBMISSION.tsv",
        "REGISTRANT.tsv",
        "FUND_REPORTED_INFO.tsv",
        "FUND_REPORTED_HOLDING.tsv",
        "IDENTIFIERS.tsv",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (EXTRACT_DIR / filename).exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Gerekli N-PORT dosyaları eksik: "
            + ", ".join(missing_files)
        )

    print("\nGerekli ana N-PORT tabloları doğrulandı.")

    for filename in required_files:
        file_path = EXTRACT_DIR / filename
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  {filename}: {size_mb:.2f} MB")


def write_inventory() -> None:
    """Write an inventory and checksum report."""
    inventory_file = LOG_DIR / "2023q4_nport_inventory.txt"

    files = sorted(
        path
        for path in EXTRACT_DIR.rglob("*")
        if path.is_file()
    )

    with inventory_file.open(
        "w",
        encoding="utf-8",
    ) as output:
        output.write("SEC N-PORT HISTORICAL HOLDINGS DOWNLOAD\n")
        output.write("=" * 60 + "\n")
        output.write("Quarter: 2023Q4\n")
        output.write(f"ZIP URL: {ZIP_URL}\n")
        output.write(f"ZIP file: {ZIP_FILE}\n")
        output.write(f"ZIP size bytes: {ZIP_FILE.stat().st_size}\n")
        output.write(f"ZIP SHA256: {sha256_file(ZIP_FILE)}\n")
        output.write(f"Extract directory: {EXTRACT_DIR}\n")
        output.write(f"Extracted files: {len(files)}\n\n")

        for path in files:
            relative_path = path.relative_to(EXTRACT_DIR)
            size_mb = path.stat().st_size / (1024 * 1024)

            output.write(
                f"{relative_path}\t{size_mb:.4f} MB\n"
            )

    print(f"\nEnvanter kaydedildi:\n{inventory_file}")


def main() -> None:
    print("=" * 72)
    print("2023Q4 SEC N-PORT HISTORICAL HOLDINGS DOWNLOAD")
    print("=" * 72)
    print(f"Proje klasörü: {PROJECT_DIR}")
    print(f"Hedef klasör: {RAW_DIR}")
    print(f"SEC dosyası: {ZIP_URL}")

    download_file()
    inspect_and_extract()
    validate_required_files()
    write_inventory()

    print("\n2023Q4 N-PORT İŞLEMİ BAŞARILI.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 72)
        print("2023Q4 N-PORT İŞLEMİ HATA İLE DURDU")
        print("=" * 72)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)