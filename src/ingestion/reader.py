"""Excel workbook reader for multi-sheet dataset ingestion."""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.ingestion.config import IngestionConfig

logger = logging.getLogger("ecommerce_platform.reader")


class ExcelDataReader:
    """Reads raw Excel workbook sheets while capturing runtime metrics and preserving raw state."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        """Initialize Excel reader.

        Args:
            config: IngestionConfig instance.
        """
        self.config = config or IngestionConfig()

    def inspect_sheet_names(self, excel_path: Optional[Path] = None) -> List[str]:
        """Inspects and returns all sheet names present in the workbook.

        Args:
            excel_path: Path to the excel file.

        Returns:
            List of sheet names.
        """
        path = excel_path or self.config.excel_path
        if not path.exists():
            raise FileNotFoundError(f"Workbook not found at: {path}")

        logger.info(f"Inspecting workbook sheets at: {path}")
        excel_file = pd.ExcelFile(path, engine="openpyxl")
        sheet_names = excel_file.sheet_names
        logger.info(f"Discovered {len(sheet_names)} sheet(s): {sheet_names}")
        return sheet_names

    def read_sheet(self, sheet_name: str, excel_path: Optional[Path] = None) -> pd.DataFrame:
        """Reads a single workbook sheet into a pandas DataFrame.

        Args:
            sheet_name: Name of sheet to read.
            excel_path: Path to the excel file.

        Returns:
            DataFrame containing raw sheet records.
        """
        path = excel_path or self.config.excel_path
        logger.info(f"Reading sheet '{sheet_name}' from {path.name}...")
        start_time = time.time()

        # Read using openpyxl engine
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            engine="openpyxl",
            dtype={
                "Invoice": str,
                "StockCode": str,
                "Description": str,
                "Country": str,
            },
        )
        duration = round(time.time() - start_time, 2)
        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        logger.info(
            f"Successfully read sheet '{sheet_name}': {len(df):,} rows, "
            f"{len(df.columns)} columns in {duration}s ({memory_mb} MB in memory)"
        )
        return df

    def read_all_sheets(
        self,
        excel_path: Optional[Path] = None,
        expected_sheets: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Reads all specified or discovered sheets from the workbook.

        Args:
            excel_path: Path to the Excel file.
            expected_sheets: Specific sheets to load (defaults to config.expected_sheets).

        Returns:
            Dictionary mapping sheet_name -> raw DataFrame.
        """
        path = excel_path or self.config.excel_path
        available_sheets = self.inspect_sheet_names(path)

        sheets_to_read = expected_sheets or self.config.expected_sheets
        missing_sheets = set(sheets_to_read) - set(available_sheets)
        if missing_sheets:
            logger.warning(
                f"Expected sheet(s) {missing_sheets} not found in workbook. "
                f"Available sheets: {available_sheets}. Reading available sheets."
            )
            sheets_to_read = [s for s in sheets_to_read if s in available_sheets]

        results: Dict[str, pd.DataFrame] = {}
        for sheet in sheets_to_read:
            results[sheet] = self.read_sheet(sheet, excel_path=path)

        return results
