"""Unit tests for ExcelDataReader."""
from pathlib import Path

import pytest

from src.ingestion.config import IngestionConfig
from src.ingestion.reader import ExcelDataReader


def test_inspect_sheet_names(mock_excel_file: Path, test_config: IngestionConfig):
    """Test discovering sheet names in workbook."""
    reader = ExcelDataReader(config=test_config)
    sheet_names = reader.inspect_sheet_names(excel_path=mock_excel_file)
    assert "Year 2009-2010" in sheet_names
    assert "Year 2010-2011" in sheet_names
    assert len(sheet_names) == 2


def test_read_sheet(mock_excel_file: Path, test_config: IngestionConfig):
    """Test reading a single sheet from workbook."""
    reader = ExcelDataReader(config=test_config)
    df = reader.read_sheet("Year 2009-2010", excel_path=mock_excel_file)
    assert len(df) == 5
    assert "Invoice" in df.columns
    assert "Price" in df.columns


def test_read_all_sheets(mock_excel_file: Path, test_config: IngestionConfig):
    """Test reading both workbook sheets."""
    reader = ExcelDataReader(config=test_config)
    sheets_dict = reader.read_all_sheets(excel_path=mock_excel_file)
    assert len(sheets_dict) == 2
    assert "Year 2009-2010" in sheets_dict
    assert "Year 2010-2011" in sheets_dict
    assert len(sheets_dict["Year 2009-2010"]) == 5
    assert len(sheets_dict["Year 2010-2011"]) == 5


def test_missing_file_raises_error(test_config: IngestionConfig):
    """Test error raised when workbook path does not exist."""
    reader = ExcelDataReader(config=test_config)
    with pytest.raises(FileNotFoundError):
        reader.inspect_sheet_names(excel_path=Path("non_existent_file.xlsx"))
