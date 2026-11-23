"""
Regression and Quality Assurance Audit Tests for Phase 5 Isolation Forest Anomaly Detection.

Validates:
1. Exact source record verification for ambiguous cancellation C496350.
2. Temporal leakage invariance under chronological perturbation.
3. Deterministic lineage mapping to original workbook sheets and rows.
4. Database parity, model versioning, and scoring timestamps.
5. Exact financial non-interference against Phase 2 baseline.
"""
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.anomaly_features import (
    TransactionFeaturePipeline,
    compute_leakage_safe_prior_orders,
)


@pytest.fixture(scope="module")
def gold_anomalies_df() -> pd.DataFrame:
    path = Path("data/gold/gold_transaction_anomalies_isolation_forest.parquet")
    if not path.exists():
        pytest.skip(f"Dataset missing: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def silver_df() -> pd.DataFrame:
    path = Path("data/silver/online_retail_silver.parquet")
    if not path.exists():
        pytest.skip(f"Dataset missing: {path}")
    return pd.read_parquet(path)


def test_qa_audit_c496350_source_record(silver_df, gold_anomalies_df):
    """Verifies that invoice C496350 has true raw workbook values:
    unit_price = 373.57, quantity = 1, customer_id = None/NA, sheet = 'Year 2009-2010'.
    """
    c49_s = silver_df[silver_df["invoice_no"] == "C496350"]
    assert len(c49_s) == 1
    row_s = c49_s.iloc[0]

    assert row_s["stock_code"] == "M"
    assert row_s["quantity"] == 1
    assert row_s["unit_price"] == 373.57
    assert pd.isna(row_s["customer_id"])
    assert row_s["sheet_name"] == "Year 2009-2010"
    assert bool(row_s["is_ambiguous_cancellation"]) is True
    assert row_s["transaction_status"] == "CANCELLATION"

    c49_g = gold_anomalies_df[gold_anomalies_df["invoice_no"] == "C496350"]
    assert len(c49_g) == 1
    row_g = c49_g.iloc[0]

    assert row_g["source_row_id"] == 76799
    assert row_g["transaction_id"] == "TXN_0076799"
    assert row_g["unit_price"] == 373.57
    assert row_g["quantity"] == 1
    assert row_g["line_total"] == 373.57
    assert bool(row_g["is_anomaly"]) is True
    assert "HIGH_UNIT_PRICE" in row_g["diagnostic_reasons"]


def test_qa_audit_chronological_perturbation_invariance():
    """Verifies that appending future transactions does not perturb past feature representations."""
    df_past = pd.DataFrame({
        "customer_id": [100, 100, 200, 100],
        "invoice_no": ["INV1", "INV1", "INV2", "INV3"],
        "stock_code": ["A", "B", "A", "C"],
        "quantity": [10, 5, 2, 20],
        "unit_price": [2.0, 3.0, 2.0, 5.0],
        "invoice_date": pd.to_datetime([
            "2010-01-01 10:00:00",
            "2010-01-01 10:00:00",
            "2010-01-02 11:00:00",
            "2010-01-05 12:00:00",
        ]),
        "transaction_status": ["VALID_PURCHASE"] * 4,
    })

    pipeline = TransactionFeaturePipeline(use_scaler=True)
    pipeline.fit(df_past)
    X_past = pipeline.transform(df_past)

    # Future transactions for customer 100 and a new customer 300
    df_future = pd.DataFrame({
        "customer_id": [100, 300],
        "invoice_no": ["INV4", "INV5"],
        "stock_code": ["A", "D"],
        "quantity": [50, 1],
        "unit_price": [2.0, 15.0],
        "invoice_date": pd.to_datetime([
            "2010-02-01 10:00:00",
            "2010-02-05 15:00:00",
        ]),
        "transaction_status": ["VALID_PURCHASE"] * 2,
    })

    df_combined = pd.concat([df_past, df_future], ignore_index=True)
    X_combined = pipeline.transform(df_combined)

    diff = np.max(np.abs(X_combined[:len(df_past)] - X_past))
    assert diff < 1e-12, f"Feature leakage detected! Max past feature diff: {diff}"


def test_qa_audit_deterministic_lineage_mapping(silver_df, gold_anomalies_df):
    """Verifies that source_row_id preserves exact row ordering and sheet boundary."""
    assert len(gold_anomalies_df) == 1067371
    assert (gold_anomalies_df["source_row_id"].values == np.arange(1067371)).all()

    # Sheet boundary check (525,461 rows in sheet 1)
    boundary_idx = 525461
    assert silver_df.iloc[boundary_idx - 1]["sheet_name"] == "Year 2009-2010"
    assert silver_df.iloc[boundary_idx]["sheet_name"] == "Year 2010-2011"

    # Lineage check at boundary
    row_prev = gold_anomalies_df.iloc[boundary_idx - 1]
    row_curr = gold_anomalies_df.iloc[boundary_idx]
    assert row_prev["source_row_id"] == boundary_idx - 1
    assert row_curr["source_row_id"] == boundary_idx


def test_qa_audit_financial_non_interference_exact_penny(silver_df, gold_anomalies_df):
    """Enforces zero-penny tolerance: anomaly scoring must not change historical revenue."""
    silver_net = Decimal(str(round(float(silver_df["net_revenue"].sum()), 2)))
    assert silver_net == Decimal("19012910.85")

    silver_line_total = round(float(silver_df["line_total"].sum()), 2)
    gold_line_total = round(float(gold_anomalies_df["line_total"].sum()), 2)
    assert silver_line_total == gold_line_total


def test_qa_audit_database_metadata_columns():
    """Verifies model_version and scoring_timestamp presence in PostgreSQL table."""
    from src.database.connection import DatabaseManager
    from sqlalchemy import text

    db_mgr = DatabaseManager()
    try:
        health = db_mgr.check_health()
        if health.get("status") != "HEALTHY":
            pytest.skip("PostgreSQL database is not reachable.")
    except Exception:
        pytest.skip("PostgreSQL database connection failed.")

    with db_mgr.engine.connect() as conn:
        cols_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'transaction_anomalies_isolation_forest';
        """)
        cols = [r[0] for r in conn.execute(cols_query).fetchall()]
        assert "model_version" in cols
        assert "scoring_timestamp" in cols
        assert "transaction_id" in cols
        assert "source_row_id" in cols

        # Check total count and anomalies count
        cnt = conn.execute(text("SELECT COUNT(*) FROM transaction_anomalies_isolation_forest;")).scalar()
        assert cnt == 1067371

        anom_cnt = conn.execute(text("SELECT COUNT(*) FROM transaction_anomalies_isolation_forest WHERE is_anomaly = true;")).scalar()
        assert anom_cnt == 16664
