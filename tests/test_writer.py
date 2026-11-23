"""Unit tests for BronzeParquetWriter."""
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.config import IngestionConfig
from src.ingestion.schema import IngestionValidator, StandardizedSchema
from src.ingestion.writer import BronzeParquetWriter


def test_write_and_inspect_parquet(sample_raw_dataframe: pd.DataFrame, tmp_path: Path):
    """Test atomic parquet writing and inspection."""
    config = IngestionConfig(base_dir=tmp_path)
    writer = BronzeParquetWriter(config=config)

    standardized = IngestionValidator.standardize_columns(
        sample_raw_dataframe, sheet_name="Year 2009-2010"
    )
    dest_path = tmp_path / "test_output.parquet"

    result = writer.write_parquet(standardized, dest_path)
    assert result["row_count"] == len(sample_raw_dataframe)
    assert dest_path.exists()
    assert dest_path.stat().st_size > 0

    # Inspect parquet metadata
    meta = writer.inspect_parquet(dest_path)
    assert meta["num_rows"] == 5
    assert meta["num_columns"] == len(standardized.columns)
    assert StandardizedSchema.INVOICE_NO in meta["schema"]


def test_write_parquet_idempotence(sample_raw_dataframe: pd.DataFrame, tmp_path: Path):
    """Test writing multiple times cleanly replaces destination without duplication."""
    config = IngestionConfig(base_dir=tmp_path)
    writer = BronzeParquetWriter(config=config)

    standardized = IngestionValidator.standardize_columns(
        sample_raw_dataframe, sheet_name="Year 2009-2010"
    )
    dest_path = tmp_path / "idempotent_test.parquet"

    # Write 1st time
    res1 = writer.write_parquet(standardized, dest_path)
    # Write 2nd time (should overwrite atomically)
    res2 = writer.write_parquet(standardized, dest_path)

    assert res1["row_count"] == res2["row_count"]
    df_read = pd.read_parquet(dest_path)
    assert len(df_read) == 5  # Row count remains 5, not doubled
