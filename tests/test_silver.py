"""Unit tests for Silver layer transformations and business rules."""
import numpy as np
import pandas as pd
import pytest

from src.etl.config import ETLConfig
from src.etl.silver import QualityFlag, SilverTransformer, TransactionStatus


def test_silver_100_percent_preservation(mock_bronze_records):
    """Test that Silver transformation preserves exactly 100% of Bronze records."""
    transformer = SilverTransformer(ETLConfig())
    silver_df = transformer.transform_dataframe(mock_bronze_records)
    assert len(silver_df) == len(mock_bronze_records)


def test_transaction_status_hierarchy(mock_bronze_records):
    """Test mutually exclusive classification into TransactionStatus categories."""
    transformer = SilverTransformer(ETLConfig())
    silver_df = transformer.transform_dataframe(mock_bronze_records)

    statuses = silver_df["transaction_status"].tolist()
    assert statuses[0] == TransactionStatus.VALID_PURCHASE
    assert statuses[1] == TransactionStatus.VALID_PURCHASE  # Valid purchase even if customer is null
    assert statuses[2] == TransactionStatus.CANCELLATION
    assert statuses[3] == TransactionStatus.RETURN_ADJUSTMENT
    assert statuses[4] == TransactionStatus.INVENTORY_WRITE_OFF
    assert statuses[5] == TransactionStatus.ZERO_PRICE_ENTRY
    assert statuses[6] == TransactionStatus.DUPLICATE
    assert statuses[7] == TransactionStatus.VALID_PURCHASE  # Fee with positive qty and price
    assert statuses[8] == TransactionStatus.CANCELLATION    # Invoice starts with C


def test_financial_revenue_calculations(mock_bronze_records):
    """Test that cancellations and returns are not counted as gross sales, and net revenue is exact."""
    transformer = SilverTransformer(ETLConfig())
    silver_df = transformer.transform_dataframe(mock_bronze_records)

    # Row 0: Valid purchase: 10 * 5.0 = 50.0 gross, 0 return, 50.0 net
    r0 = silver_df.iloc[0]
    assert r0["gross_sales"] == 50.0
    assert r0["return_amount"] == 0.0
    assert r0["net_revenue"] == 50.0

    # Row 2: Cancellation: -2 * 5.0 -> gross 0.0, return 10.0, net -10.0
    r2 = silver_df.iloc[2]
    assert r2["gross_sales"] == 0.0
    assert r2["return_amount"] == 10.0
    assert r2["net_revenue"] == -10.0

    # Row 4: Inventory write-off: qty -50, price 0 -> gross 0.0, return 0.0, net 0.0
    r4 = silver_df.iloc[4]
    assert r4["gross_sales"] == 0.0
    assert r4["return_amount"] == 0.0
    assert r4["net_revenue"] == 0.0

    # Row 6: Duplicate of row 0 -> gross 0.0, return 0.0, net 0.0 (excluded from revenue!)
    r6 = silver_df.iloc[6]
    assert r6["gross_sales"] == 0.0
    assert r6["return_amount"] == 0.0
    assert r6["net_revenue"] == 0.0

    # Verify gross - return == net across entire dataframe
    diff = (silver_df["gross_sales"] - silver_df["return_amount"]) - silver_df["net_revenue"]
    assert (diff.abs() < 1e-6).all()


def test_bad_debt_accounting_adjustments():
    """Test that negative unit-price bad debt records are excluded from commercial revenue."""
    bad_debt_data = pd.DataFrame({
        "invoice_no": ["A506401"],
        "stock_code": ["B"],
        "description": ["Adjust bad debt"],
        "quantity": [1],
        "invoice_date": [pd.Timestamp("2010-04-29 13:36:00")],
        "unit_price": [-53594.36],
        "customer_id": [None],
        "country": ["United Kingdom"],
        "sheet_name": ["Year 2009-2010"],
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")],
        "is_cancellation": [False],
    })
    transformer = SilverTransformer(ETLConfig())
    silver_df = transformer.transform_dataframe(bad_debt_data)

    row = silver_df.iloc[0]
    assert bool(row["is_invalid_price"]) is True
    assert row["transaction_status"] == TransactionStatus.ZERO_PRICE_ENTRY
    assert row["quality_flag"] == QualityFlag.EXCLUDE_NON_REVENUE
    assert row["gross_sales"] == 0.0
    assert row["return_amount"] == 0.0
    assert row["net_revenue"] == 0.0


def test_cancellation_with_positive_quantity_anomaly():
    """Test invoice starting with 'C' having positive quantity (e.g. C496350).

    Verifies:
    1. Original positive quantity (+1) and unit price (£373.57) are strictly preserved.
    2. Explicitly flagged with is_ambiguous_cancellation == True.
    3. Accounting treatment: treated as credit note deduction (return_amount = 373.57, net_revenue = -373.57).
    4. Financial impact: including deduction yields -£373.57, excluding yields £0.00 (delta +£373.57).
    """
    canc_data = pd.DataFrame({
        "invoice_no": ["C496350"],
        "stock_code": ["M"],
        "description": ["Manual"],
        "quantity": [1],
        "invoice_date": [pd.Timestamp("2010-01-28 11:24:00")],
        "unit_price": [373.57],
        "customer_id": [None],
        "country": ["United Kingdom"],
        "sheet_name": ["Year 2009-2010"],
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")],
        "is_cancellation": [True],
    })
    transformer = SilverTransformer(ETLConfig())
    silver_df = transformer.transform_dataframe(canc_data)

    row = silver_df.iloc[0]
    # 1. Preservation of original raw fields
    assert row["quantity"] == 1
    assert row["unit_price"] == 373.57
    assert row["invoice_no"] == "C496350"

    # 2. Ambiguous cancellation flag
    assert bool(row["is_ambiguous_cancellation"]) is True
    assert bool(row["is_cancellation"]) is True
    assert row["transaction_status"] == TransactionStatus.CANCELLATION

    # 3. Current accounting treatment (deduction)
    assert row["gross_sales"] == 0.0
    assert row["return_amount"] == 373.57
    assert row["net_revenue"] == -373.57

    # 4. Financial impact verification
    included_net_impact = row["net_revenue"]
    excluded_net_impact = 0.0
    impact_delta = excluded_net_impact - included_net_impact
    assert round(impact_delta, 2) == 373.57


def test_missing_customer_preservation_and_revenue():
    """Test that records missing customer ID are preserved and count toward revenue."""
    guest_data = pd.DataFrame({
        "invoice_no": ["500001"],
        "stock_code": ["21099"],
        "description": ["PAPER CUPS"],
        "quantity": [20],
        "invoice_date": [pd.Timestamp("2010-03-01 10:00:00")],
        "unit_price": [1.50],
        "customer_id": [None],
        "country": ["United Kingdom"],
        "sheet_name": ["Year 2009-2010"],
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")],
        "is_cancellation": [False],
    })
    transformer = SilverTransformer(ETLConfig())
    silver_df = transformer.transform_dataframe(guest_data)

    assert len(silver_df) == 1
    row = silver_df.iloc[0]
    assert bool(row["is_missing_customer"]) is True
    assert row["quality_flag"] == QualityFlag.WARN_MISSING_CUSTOMER
    assert row["transaction_status"] == TransactionStatus.VALID_PURCHASE
    assert row["gross_sales"] == 30.0
    assert row["net_revenue"] == 30.0


def test_duplicate_subsequent_record_exclusion():
    """Test that subsequent duplicates are flagged and zeroed in revenue while the original is kept."""
    dup_data = pd.DataFrame({
        "invoice_no": ["500001", "500001", "500001"],
        "stock_code": ["21099", "21099", "21099"],
        "description": ["PAPER CUPS", "PAPER CUPS", "PAPER CUPS"],
        "quantity": [10, 10, 10],
        "invoice_date": [pd.Timestamp("2010-03-01 10:00:00")] * 3,
        "unit_price": [2.00, 2.00, 2.00],
        "customer_id": [12345, 12345, 12345],
        "country": ["United Kingdom"] * 3,
        "sheet_name": ["Year 2009-2010"] * 3,
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")] * 3,
        "is_cancellation": [False] * 3,
    })
    transformer = SilverTransformer(ETLConfig())
    silver_df = transformer.transform_dataframe(dup_data)

    assert len(silver_df) == 3
    assert bool(silver_df.iloc[0]["is_duplicate"]) is False
    assert silver_df.iloc[0]["transaction_status"] == TransactionStatus.VALID_PURCHASE
    assert silver_df.iloc[0]["net_revenue"] == 20.0

    assert bool(silver_df.iloc[1]["is_duplicate"]) is True
    assert silver_df.iloc[1]["transaction_status"] == TransactionStatus.DUPLICATE
    assert silver_df.iloc[1]["net_revenue"] == 0.0

    assert bool(silver_df.iloc[2]["is_duplicate"]) is True
    assert silver_df.iloc[2]["transaction_status"] == TransactionStatus.DUPLICATE
    assert silver_df.iloc[2]["net_revenue"] == 0.0

    assert silver_df["net_revenue"].sum() == 20.0


def test_extreme_outlier_balancing():
    """Test paired extreme outlier transactions (e.g. +80,995 and -80,995) balance cleanly."""
    outlier_data = pd.DataFrame({
        "invoice_no": ["581483", "C581484"],
        "stock_code": ["23843", "23843"],
        "description": ["PAPER CRAFT , LITTLE BIRDIE", "PAPER CRAFT , LITTLE BIRDIE"],
        "quantity": [80995, -80995],
        "invoice_date": [
            pd.Timestamp("2011-12-09 09:15:00"),
            pd.Timestamp("2011-12-09 09:27:00"),
        ],
        "unit_price": [2.08, 2.08],
        "customer_id": [16446, 16446],
        "country": ["United Kingdom", "United Kingdom"],
        "sheet_name": ["Year 2010-2011", "Year 2010-2011"],
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")] * 2,
        "is_cancellation": [False, True],
    })
    transformer = SilverTransformer(ETLConfig())
    silver_df = transformer.transform_dataframe(outlier_data)

    assert round(silver_df["gross_sales"].sum(), 2) == 168469.60
    assert round(silver_df["return_amount"].sum(), 2) == 168469.60
    assert round(silver_df["net_revenue"].sum(), 2) == 0.00
