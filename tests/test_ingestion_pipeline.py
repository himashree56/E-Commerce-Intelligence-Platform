"""Integration tests for end-to-end IngestionPipeline."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.config import IngestionConfig
from src.ingestion.pipeline import IngestionPipeline, IngestionSummary
from src.ingestion.schema import StandardizedSchema


def test_pipeline_execution_end_to_end(mock_excel_file: Path, tmp_path: Path):
    """Test full pipeline execution producing individual & unified Bronze parquets and metadata."""
    config = IngestionConfig(base_dir=tmp_path)
    pipeline = IngestionPipeline(config=config)

    summary = pipeline.run(force=True, raw_file_path=mock_excel_file)

    assert isinstance(summary, IngestionSummary)
    assert summary.status == "SUCCESS"
    assert len(summary.sheets_processed) == 2
    assert summary.total_bronze_rows == 10  # 5 rows per sheet x 2 sheets

    # Verify unified Parquet exists and is readable
    unified_path = Path(summary.unified_parquet_path)
    assert unified_path.exists()
    assert unified_path.stat().st_size > 0

    df_bronze = pd.read_parquet(unified_path)
    assert len(df_bronze) == 10
    assert StandardizedSchema.INVOICE_NO in df_bronze.columns
    assert StandardizedSchema.IS_CANCELLATION in df_bronze.columns

    # Verify individual Parquets exist
    for sheet_name, p_path in summary.individual_parquet_paths.items():
        p = Path(p_path)
        assert p.exists()
        sheet_df = pd.read_parquet(p)
        assert len(sheet_df) == 5

    # Verify metadata JSON
    metadata_file = config.metadata_dir / "latest_ingestion.json"
    assert metadata_file.exists()
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data["total_bronze_rows"] == 10
    assert meta_data["status"] == "SUCCESS"
    assert meta_data["preserved_metrics"]["cancellation_invoices_count"] >= 2
