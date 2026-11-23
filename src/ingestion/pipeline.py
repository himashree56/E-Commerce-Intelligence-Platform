"""End-to-end ingestion pipeline orchestrating raw data ingestion into Bronze layer."""
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ingestion.config import IngestionConfig
from src.ingestion.downloader import DatasetDownloader
from src.ingestion.reader import ExcelDataReader
from src.ingestion.schema import IngestionValidator, ValidationReport
from src.ingestion.writer import BronzeParquetWriter
from src.utils.logger import get_logger, setup_logging

logger = logging.getLogger("ecommerce_platform.pipeline")


@dataclass
class IngestionSummary:
    """Summary of complete ingestion pipeline execution."""

    run_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    status: str
    raw_source_file: str
    sheets_processed: List[str]
    sheet_row_counts: Dict[str, int]
    total_bronze_rows: int
    total_columns: int
    unified_parquet_path: str
    unified_parquet_size_bytes: int
    individual_parquet_paths: Dict[str, str]
    sheet_validation_reports: Dict[str, Dict[str, Any]]
    unified_validation_report: Dict[str, Any]
    preserved_metrics: Dict[str, Any]
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


class IngestionPipeline:
    """Orchestrates Phase 1 Bronze Ingestion for UCI Online Retail II dataset."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        """Initialize pipeline with configuration.

        Args:
            config: Optional IngestionConfig.
        """
        self.config = config or IngestionConfig()
        setup_logging(log_dir=self.config.logs_dir, log_file_name="ingestion.log")
        self.downloader = DatasetDownloader(self.config)
        self.reader = ExcelDataReader(self.config)
        self.writer = BronzeParquetWriter(self.config)

    def run(self, force: bool = False, raw_file_path: Optional[Path] = None) -> IngestionSummary:
        """Executes the full Bronze ingestion pipeline idempotently.

        Steps:
        1. Resolve raw Excel workbook path (existing, download, or extracted).
        2. Read both workbook sheets preserving raw state.
        3. Standardize column names into snake_case and append metadata audit fields.
        4. Validate schema, types, missing values, duplicates, and transaction counts.
        5. Verify preservation of cancellations, negative quantities, and missing customer IDs.
        6. Persist individual and unified datasets into data/bronze/ as Parquet.
        7. Record structured metadata in data/metadata/.

        Args:
            force: Re-download and overwrite existing outputs if True.
            raw_file_path: Optional direct path to raw Excel or zip file.

        Returns:
            IngestionSummary containing execution metrics.
        """
        start_time = datetime.now(timezone.utc)
        run_id = f"ingest_{start_time.strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"=== Starting Bronze Ingestion Pipeline (Run ID: {run_id}) ===")

        errors: List[str] = []
        sheet_summaries: Dict[str, Dict[str, Any]] = {}
        sheet_row_counts: Dict[str, int] = {}
        individual_parquets: Dict[str, str] = {}
        standardized_dfs: List[pd.DataFrame] = []

        try:
            # 1. Resolve raw data file
            if raw_file_path and Path(raw_file_path).exists():
                excel_path = Path(raw_file_path)
                logger.info(f"Using explicitly specified raw dataset at: {excel_path}")
            else:
                excel_path = self.downloader.get_or_download_dataset(force=force)

            # 2. Inspect available sheets
            available_sheets = self.reader.inspect_sheet_names(excel_path)
            logger.info(f"Sheets discovered in raw file: {available_sheets}")

            sheets_to_process = [
                s for s in self.config.expected_sheets if s in available_sheets
            ]
            if not sheets_to_process:
                sheets_to_process = available_sheets

            # 3. Read, Standardize, and Validate each sheet
            for sheet_name in sheets_to_process:
                logger.info(f"--- Processing Sheet: {sheet_name} ---")

                # Read raw sheet
                raw_df = self.reader.read_sheet(sheet_name, excel_path=excel_path)
                sheet_row_counts[sheet_name] = len(raw_df)

                # Validate raw columns
                is_valid_raw, raw_errors = IngestionValidator.validate_raw_columns(
                    raw_df, sheet_name
                )
                if not is_valid_raw:
                    errors.extend(raw_errors)

                # Standardize column names (without mutating raw_df)
                standardized_df = IngestionValidator.standardize_columns(
                    raw_df, sheet_name=sheet_name, ingestion_timestamp=start_time
                )
                standardized_dfs.append(standardized_df)

                # Generate validation report
                val_report = IngestionValidator.generate_validation_report(
                    standardized_df, sheet_name
                )
                sheet_summaries[sheet_name] = val_report.to_dict()

                # Write sheet-specific Parquet
                sheet_slug = sheet_name.lower().replace(" ", "_").replace("-", "_")
                sheet_parquet_path = self.config.bronze_dir / f"online_retail_{sheet_slug}.parquet"
                sheet_write_res = self.writer.write_parquet(
                    standardized_df, sheet_parquet_path
                )
                individual_parquets[sheet_name] = sheet_write_res["file_path"]

            # 4. Create Unified Bronze DataFrame
            logger.info("Combining all sheets into unified Bronze dataset...")
            unified_df = pd.concat(standardized_dfs, ignore_index=True)
            total_bronze_rows = len(unified_df)
            total_columns = len(unified_df.columns)

            logger.info(
                f"Unified Bronze DataFrame: {total_bronze_rows:,} total rows across "
                f"{len(standardized_dfs)} sheets."
            )

            # 5. Validate Unified Dataset
            unified_report = IngestionValidator.generate_validation_report(
                unified_df, sheet_name="unified_bronze"
            )

            # 6. Write Unified Bronze Parquet
            unified_write_res = self.writer.write_parquet(
                unified_df, self.config.unified_bronze_path
            )

            # 7. Summary metrics & preserved anomalies verification
            preserved_metrics = {
                "total_rows": total_bronze_rows,
                "cancellation_invoices_count": unified_report.cancellation_invoices_count,
                "negative_quantity_count": unified_report.negative_quantity_count,
                "zero_quantity_count": unified_report.zero_quantity_count,
                "missing_customer_count": unified_report.missing_customer_count,
                "duplicate_rows_count": unified_report.duplicate_rows_count,
                "total_unique_invoices": unified_report.total_unique_invoices,
                "unique_countries": unified_report.countries_count,
            }

            end_time = datetime.now(timezone.utc)
            duration = round((end_time - start_time).total_seconds(), 2)
            status = "SUCCESS" if not errors else "FAILED"

            summary = IngestionSummary(
                run_id=run_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                status=status,
                raw_source_file=str(excel_path),
                sheets_processed=sheets_to_process,
                sheet_row_counts=sheet_row_counts,
                total_bronze_rows=total_bronze_rows,
                total_columns=total_columns,
                unified_parquet_path=str(self.config.unified_bronze_path),
                unified_parquet_size_bytes=unified_write_res["file_size_bytes"],
                individual_parquet_paths=individual_parquets,
                sheet_validation_reports=sheet_summaries,
                unified_validation_report=unified_report.to_dict(),
                preserved_metrics=preserved_metrics,
                errors=errors,
            )

            # 8. Save structured metadata to disk
            self._save_metadata(summary, run_id)

            logger.info(
                f"=== Bronze Ingestion Completed Successfully! "
                f"Total rows: {total_bronze_rows:,} in {duration}s ==="
            )
            return summary

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)
            end_time = datetime.now(timezone.utc)
            duration = round((end_time - start_time).total_seconds(), 2)
            errors.append(str(e))

            failed_summary = IngestionSummary(
                run_id=run_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                status="FAILED",
                raw_source_file=str(self.config.excel_path),
                sheets_processed=[],
                sheet_row_counts={},
                total_bronze_rows=0,
                total_columns=0,
                unified_parquet_path="",
                unified_parquet_size_bytes=0,
                individual_parquet_paths={},
                sheet_validation_reports={},
                unified_validation_report={},
                preserved_metrics={},
                errors=errors,
            )
            self._save_metadata(failed_summary, run_id)
            raise

    def _save_metadata(self, summary: IngestionSummary, run_id: str) -> None:
        """Persists ingestion run metadata to JSON files."""
        run_file = self.config.metadata_dir / f"{run_id}.json"
        latest_file = self.config.metadata_dir / "latest_ingestion.json"

        data = summary.to_dict()
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Ingestion metadata saved to {run_file} and {latest_file}")
