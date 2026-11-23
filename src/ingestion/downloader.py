"""Dataset acquisition, extraction, and integrity verification."""
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from src.ingestion.config import IngestionConfig

logger = logging.getLogger("ecommerce_platform.downloader")


class DatasetDownloader:
    """Manages downloading, extracting, and locating the UCI Online Retail II dataset."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        """Initialize downloader with configuration.

        Args:
            config: IngestionConfig instance.
        """
        self.config = config or IngestionConfig()

    def get_or_download_dataset(self, force: bool = False) -> Path:
        """Ensures the raw Excel workbook is available locally.

        Checks in order:
        1. Existing uncompressed Excel file in data/raw/
        2. Existing zip file in data/raw/
        3. Existing zip file in user's Downloads folder
        4. Remote download from UCI Machine Learning Repository

        Args:
            force: If True, forces re-download and re-extraction.

        Returns:
            Path to the verified raw Excel file.

        Raises:
            FileNotFoundError: If the dataset cannot be retrieved.
        """
        excel_path = self.config.excel_path
        zip_path = self.config.zip_path

        if not force and excel_path.exists() and excel_path.stat().st_size > 0:
            logger.info(f"Raw dataset already present at: {excel_path} ({excel_path.stat().st_size:,} bytes)")
            return excel_path

        # Check local zip in data/raw
        if not force and zip_path.exists() and zip_path.stat().st_size > 0:
            logger.info(f"Found existing raw zip archive at: {zip_path}")
            return self._extract_zip(zip_path)

        # Check user's Downloads folder
        user_downloads_zip = Path.home() / "Downloads" / self.config.zip_filename
        if not force and user_downloads_zip.exists() and user_downloads_zip.stat().st_size > 0:
            logger.info(f"Found dataset archive in user Downloads: {user_downloads_zip}")
            try:
                shutil.copy2(user_downloads_zip, zip_path)
                logger.info(f"Copied dataset archive to: {zip_path}")
                return self._extract_zip(zip_path)
            except Exception as e:
                logger.warning(f"Could not copy from user Downloads ({e}). Proceeding to download.")

        # Download from UCI repository
        logger.info(f"Downloading UCI Online Retail II dataset from: {self.config.dataset_url}")
        self._download_file(self.config.dataset_url, zip_path)

        return self._extract_zip(zip_path)

    def _download_file(self, url: str, target_path: Path) -> None:
        """Downloads a remote file with streaming progress bar.

        Args:
            url: Remote URL.
            target_path: Local destination path.
        """
        target_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) E-Commerce Platform Ingestion"
        }

        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with open(target_path, "wb") as f, tqdm(
                    desc=f"Downloading {target_path.name}",
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)

            logger.info(f"Download complete: {target_path} ({target_path.stat().st_size:,} bytes)")
        except Exception as e:
            if target_path.exists():
                target_path.unlink()
            logger.error(
                f"Failed to download dataset: {e}\n"
                f"Manual Instructions:\n"
                f"1. Download zip from https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
                f"2. Place 'online+retail+ii.zip' or extracted 'online_retail_II.xlsx' into: {self.config.raw_dir}"
            )
            raise

    def _extract_zip(self, zip_path: Path) -> Path:
        """Extracts the Excel workbook from the downloaded zip file.

        Args:
            zip_path: Path to zip file.

        Returns:
            Path to the extracted Excel file.
        """
        logger.info(f"Extracting zip archive: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            # Find the xlsx file inside
            xlsx_files = [f for f in archive.namelist() if f.endswith(".xlsx")]
            if not xlsx_files:
                raise FileNotFoundError(f"No .xlsx file found in archive {zip_path}")

            target_member = xlsx_files[0]
            logger.info(f"Extracting member '{target_member}' into {self.config.raw_dir}")
            archive.extract(target_member, self.config.raw_dir)

            extracted_path = self.config.raw_dir / target_member
            final_path = self.config.excel_path

            # If extracted with a different name or case, normalize to config.excel_path
            if extracted_path != final_path and extracted_path.exists():
                if final_path.exists():
                    final_path.unlink()
                extracted_path.rename(final_path)

        if not final_path.exists() or final_path.stat().st_size == 0:
            raise FileNotFoundError(f"Extraction failed: {final_path} does not exist or is empty.")

        logger.info(f"Excel dataset verified at: {final_path} ({final_path.stat().st_size:,} bytes)")
        return final_path
