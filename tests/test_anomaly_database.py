"""
Integration and Schema Tests for Transaction Anomaly PostgreSQL Layer (Phase 5).
"""
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import func, select, text

from src.database.config import DatabaseConfig
from src.database.connection import DatabaseManager
from src.database.loader import GoldDatabaseLoader
from src.database.models import Base, TransactionAnomaliesIsolationForest
from src.database.schema_manager import SchemaManager


@pytest.fixture(scope="module")
def anomaly_test_db():
    """Initializes and returns a DatabaseManager for test database, or skips if unreachable."""
    config = DatabaseConfig()
    mgr = DatabaseManager(config=config, is_test=True)
    health = mgr.check_health()
    if health.get("status") != "HEALTHY":
        pytest.skip(f"PostgreSQL test database not available: {health.get('error')}")

    # Initialize schema
    schema_mgr = SchemaManager(db_manager=mgr)
    schema_mgr.create_all_tables(is_test=True)

    yield mgr

    # Cleanup
    schema_mgr.drop_all_tables(is_test=True)
    mgr.dispose()


def test_anomaly_table_schema_and_columns():
    """Verifies that SQLAlchemy model defines all required columns, primary keys, and types."""
    model = TransactionAnomaliesIsolationForest
    assert model.__tablename__ == "transaction_anomalies_isolation_forest"

    cols = {c.name: c for c in model.__table__.columns}
    assert "transaction_id" in cols
    assert cols["transaction_id"].primary_key is True

    required_cols = [
        "source_row_id", "invoice_no", "stock_code", "invoice_date",
        "transaction_status", "quantity", "unit_price", "line_total",
        "raw_decision_score", "anomaly_score", "is_anomaly",
        "diagnostic_reasons", "model_version", "scoring_timestamp"
    ]
    for c in required_cols:
        assert c in cols, f"Column '{c}' missing from model table!"


def test_anomaly_database_loading_and_idempotence(anomaly_test_db):
    """Verifies that GoldDatabaseLoader inserts and updates anomaly records idempotently."""
    loader = GoldDatabaseLoader(is_test=True, batch_size=10)

    sample_records = pd.DataFrame([
        {
            "transaction_id": "TXN_TEST_00001",
            "source_row_id": 1,
            "invoice_no": "TEST_INV1",
            "stock_code": "85048",
            "invoice_date": pd.Timestamp("2010-01-01 10:00:00", tz="UTC"),
            "transaction_status": "VALID_PURCHASE",
            "quantity": 10.0,
            "unit_price": 5.0,
            "line_total": 50.0,
            "raw_decision_score": -0.054321,
            "anomaly_score": 0.054321,
            "is_anomaly": True,
            "diagnostic_reasons": "EXTREME_QUANTITY",
            "model_version": "v1.0.0",
            "scoring_timestamp": pd.Timestamp.now(tz="UTC"),
        },
        {
            "transaction_id": "TXN_TEST_00002",
            "source_row_id": 2,
            "invoice_no": "TEST_INV2",
            "stock_code": "22197",
            "invoice_date": pd.Timestamp("2010-01-02 11:00:00", tz="UTC"),
            "transaction_status": "VALID_PURCHASE",
            "quantity": 2.0,
            "unit_price": 3.0,
            "line_total": 6.0,
            "raw_decision_score": 0.123456,
            "anomaly_score": -0.123456,
            "is_anomaly": False,
            "diagnostic_reasons": "",
            "model_version": "v1.0.0",
            "scoring_timestamp": pd.Timestamp.now(tz="UTC"),
        },
    ])

    # First load
    with anomaly_test_db.session_scope() as session:
        summary1 = loader.load_transaction_anomalies(session, sample_records)

    assert summary1.loaded_rows == 2

    with anomaly_test_db.engine.connect() as conn:
        cnt1 = conn.execute(
            select(func.count()).select_from(TransactionAnomaliesIsolationForest)
        ).scalar()
    assert cnt1 == 2

    # Second load (Idempotent upsert with same primary keys)
    with anomaly_test_db.session_scope() as session:
        summary2 = loader.load_transaction_anomalies(session, sample_records)

    assert summary2.loaded_rows == 2

    with anomaly_test_db.engine.connect() as conn:
        cnt2 = conn.execute(
            select(func.count()).select_from(TransactionAnomaliesIsolationForest)
        ).scalar()
    assert cnt2 == 2  # Must not duplicate rows
