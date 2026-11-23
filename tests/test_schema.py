"""Unit tests for schema validation, column standardization, and anomaly preservation."""
import numpy as np
import pandas as pd
import pytest

from src.ingestion.schema import (
    IngestionValidator,
    RawSchema,
    StandardizedSchema,
    ValidationReport,
)


def test_validate_raw_columns_success(sample_raw_dataframe):
    """Test raw column validation succeeds with complete dataset."""
    is_valid, errors = IngestionValidator.validate_raw_columns(
        sample_raw_dataframe, "Year 2009-2010"
    )
    assert is_valid is True
    assert len(errors) == 0


def test_validate_raw_columns_missing_field(sample_raw_dataframe):
    """Test raw column validation catches missing columns."""
    incomplete_df = sample_raw_dataframe.drop(columns=[RawSchema.CUSTOMER_ID])
    is_valid, errors = IngestionValidator.validate_raw_columns(
        incomplete_df, "Year 2009-2010"
    )
    assert is_valid is False
    assert any("Customer ID" in err for err in errors)


def test_standardize_columns_immutability(sample_raw_dataframe):
    """Test column standardization does not mutate original raw DataFrame."""
    raw_columns_before = list(sample_raw_dataframe.columns)
    standardized = IngestionValidator.standardize_columns(
        sample_raw_dataframe, sheet_name="Year 2009-2010"
    )
    # Original should be completely unmodified
    assert list(sample_raw_dataframe.columns) == raw_columns_before
    assert RawSchema.INVOICE in sample_raw_dataframe.columns
    # Standardized should have bronze columns
    assert StandardizedSchema.INVOICE_NO in standardized.columns
    assert StandardizedSchema.SHEET_NAME in standardized.columns
    assert StandardizedSchema.IS_CANCELLATION in standardized.columns


def test_anomaly_preservation(sample_raw_dataframe):
    """Test that cancellations, negative quantities, and missing customer IDs are preserved."""
    standardized = IngestionValidator.standardize_columns(
        sample_raw_dataframe, sheet_name="Year 2009-2010"
    )

    # 1. Negative quantity preserved (row index 1 has quantity -1)
    neg_row = standardized.iloc[1]
    assert neg_row[StandardizedSchema.QUANTITY] == -1

    # 2. Cancellation flagged (C489435 has invoice starting with 'C' and qty -1)
    assert bool(neg_row[StandardizedSchema.IS_CANCELLATION]) is True

    # 3. Missing customer ID preserved (rows 2 and 4 had missing customer ID)
    assert pd.isna(standardized.iloc[2][StandardizedSchema.CUSTOMER_ID])
    assert pd.isna(standardized.iloc[4][StandardizedSchema.CUSTOMER_ID])

    # 4. Total record count must remain identical (no dropped records)
    assert len(standardized) == len(sample_raw_dataframe)


def test_validation_report_metrics(sample_raw_dataframe):
    """Test metrics calculated in validation report."""
    standardized = IngestionValidator.standardize_columns(
        sample_raw_dataframe, sheet_name="Year 2009-2010"
    )
    report = IngestionValidator.generate_validation_report(
        standardized, sheet_name="Year 2009-2010"
    )

    assert isinstance(report, ValidationReport)
    assert report.total_rows == 5
    assert report.cancellation_invoices_count >= 1
    assert report.negative_quantity_count == 1
    assert report.missing_customer_count == 2
    assert report.is_valid is True
