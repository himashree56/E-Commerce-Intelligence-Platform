"""Idempotent, atomic Parquet writer for Bronze data layer."""
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.config import IngestionConfig

logger = logging.getLogger("ecommerce_platform.writer")


class BronzeParquetWriter:
    """Manages writing and verifying Bronze Parquet datasets with atomic file operations."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        """Initialize writer.

        Args:
            config: IngestionConfig instance.
        """
        self.config = config or IngestionConfig()

    def write_parquet(
        self,
        df: pd.DataFrame,
        destination_path: Path,
        compression: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically writes a DataFrame to a Parquet file.

        Uses a temporary UUID-tagged file first, then atomically replaces the destination.
        This guarantees idempotence and ensures no partial or corrupted files exist.

        Args:
            df: Standardized DataFrame to persist.
            destination_path: Target .parquet file path.
            compression: Compression algorithm (default from config, snappy).

        Returns:
            Dictionary with written file metadata (path, row_count, file_size_bytes).
        """
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        compression = compression or self.config.parquet_compression

        # Temporary atomic write path
        temp_file = destination_path.parent / f".tmp_{uuid.uuid4().hex}_{destination_path.name}"

        try:
            logger.info(
                f"Writing {len(df):,} rows to temporary Parquet file: {temp_file.name} "
                f"using {compression} compression..."
            )
            # Convert to PyArrow table for strict schema preservation
            table = pa.Table.from_pandas(df, preserve_index=False)

            pq.write_table(
                table,
                temp_file,
                compression=compression,
                use_dictionary=True,
                write_statistics=True,
            )

            # Atomically rename/replace target file
            if destination_path.exists():
                destination_path.unlink()
            temp_file.replace(destination_path)

            file_size = destination_path.stat().st_size
            logger.info(
                f"Successfully committed Bronze Parquet: {destination_path} "
                f"({len(df):,} rows, {file_size:,} bytes)"
            )

            return {
                "file_path": str(destination_path),
                "row_count": len(df),
                "columns_count": len(df.columns),
                "file_size_bytes": file_size,
                "compression": compression,
            }

        except Exception as e:
            logger.error(f"Failed writing Parquet to {destination_path}: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def inspect_parquet(self, parquet_path: Path) -> Dict[str, Any]:
        """Inspects written Parquet file metadata directly using PyArrow.

        Args:
            parquet_path: Path to Parquet file.

        Returns:
            Dictionary of parquet file metadata and stats.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        parquet_file = pq.ParquetFile(parquet_path)
        meta = parquet_file.metadata

        return {
            "num_rows": meta.num_rows,
            "num_columns": meta.num_columns,
            "num_row_groups": meta.num_row_groups,
            "serialized_size": meta.serialized_size,
            "schema": [f.name for f in parquet_file.schema_arrow],
        }
