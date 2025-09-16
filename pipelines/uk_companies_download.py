"""
UK Companies House Data Download Pipeline using ZenML
Downloads the latest company data dump and extracts it to a local folder.
"""

import logging
import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPANIES_HOUSE_SNAPSHOT_PAGE = "https://download.companieshouse.gov.uk/en_output.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


@step(enable_cache=False)
def discover_snapshot_url() -> str:
    """Parse the Companies House snapshot page to find the current dump URL.

    Returns:
        str: Absolute URL to the latest "Company data as one file" ZIP.
    """
    logger.info("Fetching Companies House snapshot page: %s", COMPANIES_HOUSE_SNAPSHOT_PAGE)

    try:
        response = requests.get(COMPANIES_HOUSE_SNAPSHOT_PAGE, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch Companies House page: {e}")

    soup = BeautifulSoup(response.text, "lxml")

    # Find the heading that contains the text "Company data as one file"
    heading = soup.find(
        lambda tag: tag.name in {"h2", "h3"} and "Company data as one file" in tag.get_text()
    )

    link: str | None = None

    if heading:
        # The anchor usually lives in the next <ul>/<li>
        for a in heading.find_all_next("a", href=True):
            href: str = a["href"]
            if href.lower().endswith(".zip") and "BasicCompanyDataAsOneFile" in href:
                link = href
                break

    # Fallback – scan all anchors if heuristic above failed
    if not link:
        for a in soup.find_all("a", href=True):
            href: str = a["href"]
            if href.lower().endswith(".zip") and "BasicCompanyDataAsOneFile" in href:
                link = href
                break

    if not link:
        raise ValueError("Unable to locate Companies House dump link on page")

    # Ensure the link is absolute
    if not link.startswith("http"):
        link = f"https://download.companieshouse.gov.uk/{link.lstrip('/')}"

    logger.info("Discovered Companies House dump URL: %s", link)
    return link


@step(enable_cache=False)
def download_and_extract(url: str, output_dir: str = "data/uk_companies_house") -> Tuple[str, Dict]:
    """Download the ZIP file and extract the CSV to the specified directory.

    Args:
        url: URL to the Companies House ZIP file
        output_dir: Directory to extract the CSV file

    Returns:
        Tuple[str, Dict]: Path to extracted CSV file and download metadata
    """
    logger.info("Starting download from: %s", url)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract filename and date from URL if possible
    filename = url.split("/")[-1]
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    date_str = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")

    # Download with progress bar
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Save ZIP temporarily
        zip_path = output_path / f"temp_{filename}"

        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded {total_size / (1024*1024):.1f} MB to {zip_path}")

    except requests.RequestException as e:
        raise ValueError(f"Failed to download file: {e}")

    # Extract CSV from ZIP
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Find the CSV file in the archive
            csv_files = [name for name in zip_file.namelist() if name.lower().endswith('.csv')]

            if not csv_files:
                raise ValueError("No CSV file found in ZIP archive")

            csv_filename = csv_files[0]
            logger.info(f"Extracting CSV file: {csv_filename}")

            # Extract with a meaningful name
            extracted_name = f"BasicCompanyData_{date_str}.csv"
            csv_path = output_path / extracted_name

            with zip_file.open(csv_filename) as source, open(csv_path, 'wb') as target:
                total_extracted = 0
                with tqdm(desc="Extracting", unit='B', unit_scale=True) as pbar:
                    while True:
                        chunk = source.read(8192)
                        if not chunk:
                            break
                        target.write(chunk)
                        total_extracted += len(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Extracted {total_extracted / (1024*1024):.1f} MB to {csv_path}")

    except zipfile.ZipFileError as e:
        raise ValueError(f"Failed to extract ZIP file: {e}")

    finally:
        # Clean up temporary ZIP file
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Cleaned up temporary ZIP file")

    # Create download metadata
    metadata = {
        "download_url": url,
        "download_date": datetime.now().isoformat(),
        "original_filename": filename,
        "extracted_filename": extracted_name,
        "file_size_bytes": csv_path.stat().st_size,
        "file_size_mb": round(csv_path.stat().st_size / (1024*1024), 1),
        "data_date": date_str
    }

    # Save metadata
    metadata_path = output_path / "download_info.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved download metadata to {metadata_path}")
    logger.info(f"Download completed successfully: {csv_path}")

    return str(csv_path), metadata


@pipeline
def uk_companies_download_pipeline(output_dir: str = "data/uk_companies_house"):
    """Main pipeline for downloading UK Companies House data."""

    # Discover the latest dump URL
    snapshot_url = discover_snapshot_url()

    # Download and extract the data
    csv_path, metadata = download_and_extract(snapshot_url, output_dir)

    logger.info("Pipeline completed successfully!")
    logger.info("Check the output directory for downloaded files.")

    return {
        "csv_path": csv_path,
        "metadata": metadata
    }


if __name__ == "__main__":
    uk_companies_download_pipeline()
