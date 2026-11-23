"""Pytest fixtures for ingestion and ETL tests."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.ingestion.config import IngestionConfig
from src.ingestion.schema import RawSchema


@pytest.fixture
def sample_raw_records() -> Dict[str, list]:
    """Provides sample records matching UCI Online Retail II raw structure."""
    return {
        RawSchema.INVOICE: ["489434", "C489435", "489436", "489437", "489438"],
        RawSchema.STOCK_CODE: ["85048", "22197", "84879", "79323P", "POST"],
        RawSchema.DESCRIPTION: [
            "15CM CHRISTMAS GLASS BALL 20 LIGHTS",
            "SMALL POPCORN HOLDER",
            "ASSORTED COLOUR BIRD ORNAMENT",
            "PINK CHERRY BLOSSOM CUP & SAUCER",
            None,  # missing description
        ],
        RawSchema.QUANTITY: [12, -1, 32, 0, 1],  # positive, negative, zero
        RawSchema.INVOICE_DATE: [
            "2009-12-01 07:45:00",
            "2009-12-01 09:12:00",
            "2009-12-01 10:28:00",
            "2009-12-01 11:45:00",
            "2009-12-01 12:00:00",
        ],
        RawSchema.PRICE: [6.95, 0.72, 1.69, 1.25, 18.0],
        RawSchema.CUSTOMER_ID: [13085.0, 13085.0, np.nan, 15332.0, np.nan],  # missing customer IDs
        RawSchema.COUNTRY: [
            "United Kingdom",
            "United Kingdom",
            "United Kingdom",
            "Germany",
            "France",
        ],
    }


@pytest.fixture
def sample_raw_dataframe(sample_raw_records) -> pd.DataFrame:
    """Returns sample raw DataFrame."""
    return pd.DataFrame(sample_raw_records)


@pytest.fixture
def mock_excel_file(tmp_path: Path, sample_raw_dataframe: pd.DataFrame) -> Path:
    """Creates a mock Excel workbook with two sheets in a temporary directory."""
    excel_path = tmp_path / "mock_online_retail.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sample_raw_dataframe.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        sample_raw_dataframe.to_excel(writer, sheet_name="Year 2010-2011", index=False)
    return excel_path


@pytest.fixture
def test_config(tmp_path: Path) -> IngestionConfig:
    """Provides an IngestionConfig scoped to a temporary directory."""
    return IngestionConfig(base_dir=tmp_path)


@pytest.fixture
def mock_bronze_records() -> pd.DataFrame:
    """Fixture providing diverse Bronze transaction records covering all edge cases."""
    data = {
        "invoice_no": [
            "100001",   # 0: Valid purchase
            "100002",   # 1: Valid purchase missing customer
            "C100003",  # 2: Standard cancellation
            "100004",   # 3: Return adjustment (no 'C', qty < 0, price > 0)
            "100005",   # 4: Inventory write-off (qty < 0, price = 0)
            "100006",   # 5: Zero price entry (qty > 0, price = 0)
            "100001",   # 6: Exact duplicate of row 0
            "100007",   # 7: Non-product fee (POST)
            "C100008",  # 8: Cancellation with positive quantity anomaly (qty > 0)
        ],
        "stock_code": ["85048", "22197", "85048", "22197", "35956", "21646", "85048", "POST", "M"],
        "description": [
            "BALLS", "POPCORN", "BALLS", "POPCORN", "damaged", "sample", "BALLS", "POSTAGE", "Manual"
        ],
        "quantity": [10, 5, -2, -1, -50, 10, 10, 1, 1],
        "invoice_date": pd.to_datetime([
            "2010-01-01 10:00:00",
            "2010-01-02 11:00:00",
            "2010-01-03 12:00:00",
            "2010-01-04 13:00:00",
            "2010-01-05 14:00:00",
            "2010-01-06 15:00:00",
            "2010-01-01 10:00:00",
            "2010-01-07 16:00:00",
            "2010-01-08 17:00:00",
        ]),
        "unit_price": [5.0, 2.0, 5.0, 2.0, 0.0, 0.0, 5.0, 15.0, 100.0],
        "customer_id": [12345, None, 12345, 12345, None, None, 12345, 12345, None],
        "country": ["United Kingdom"] * 9,
        "sheet_name": ["Year 2009-2010"] * 9,
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")] * 9,
        "is_cancellation": [False, False, True, True, True, False, False, False, True],
    }
    return pd.DataFrame(data)
