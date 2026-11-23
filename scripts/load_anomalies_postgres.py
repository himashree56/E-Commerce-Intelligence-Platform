"""
CLI Script for Loading Transaction Anomaly Scores into PostgreSQL (Phase 5).

Creates the transaction_anomalies_isolation_forest table with proper indexes
and constraints, and performs batched, idempotent upsert of all 1,067,371 records.
"""
import os
import sys
import time
from pathlib import Path

# Silence loky warning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import func, select, text

from src.database.connection import DatabaseManager
from src.database.loader import GoldDatabaseLoader
from src.database.models import TransactionAnomaliesIsolationForest
from src.database.schema_manager import SchemaManager
from src.models.config import AnomalyDetectionConfig


def main() -> int:
    print("=" * 85)
    print("    E-COMMERCE INTELLIGENCE PLATFORM - LOAD ANOMALIES TO POSTGRESQL")
    print("=" * 85)

    config = AnomalyDetectionConfig()
    db_mgr = DatabaseManager()

    # 1. Health check
    health = db_mgr.check_health()
    if health.get("status") != "HEALTHY":
        print(f"[ERROR] Database connection unhealthy: {health.get('error')}", file=sys.stderr)
        return 1

    # 2. Ensure table schema exists
    schema_mgr = SchemaManager(db_manager=db_mgr)
    created_tables = schema_mgr.create_all_tables(is_test=False)
    print(f"[SCHEMA] Verified schema. Tables in DB: {len(created_tables)}")
    if TransactionAnomaliesIsolationForest.__tablename__ not in created_tables:
        print(f"[ERROR] Table '{TransactionAnomaliesIsolationForest.__tablename__}' not found in DB!", file=sys.stderr)
        return 1

    # 3. Load Scored Dataset
    parquet_path = config.gold_anomalies_path
    if not parquet_path.exists():
        print(f"[ERROR] Scored anomalies dataset not found at {parquet_path}. Run scoring first!", file=sys.stderr)
        return 1

    print(f"[DATA] Reading scored anomalies from: {parquet_path}...")
    t0 = time.time()
    scored_df = pd.read_parquet(parquet_path)
    total_records = len(scored_df)
    print(f"[DATA] Loaded {total_records:,} scored records in {time.time()-t0:.2f}s.")

    # 4. Load into PostgreSQL via batched upsert
    loader = GoldDatabaseLoader(is_test=False, batch_size=10000)
    print(f"[DATABASE] Loading {total_records:,} records into PostgreSQL in batches of {loader.batch_size:,}...")
    with db_mgr.session_scope() as session:
        load_summary = loader.load_transaction_anomalies(session, scored_df)

    print(f"[DATABASE] Load completed in {load_summary.duration_seconds:.2f}s. Loaded: {load_summary.loaded_rows:,} rows.")

    # 5. Verification and reconciliation
    with db_mgr.engine.connect() as conn:
        pg_count = conn.execute(select(func.count()).select_from(TransactionAnomaliesIsolationForest)).scalar()
        pg_anomalies = conn.execute(
            select(func.count())
            .select_from(TransactionAnomaliesIsolationForest)
            .where(TransactionAnomaliesIsolationForest.is_anomaly == True)
        ).scalar()

    print(f"[VERIFY] PostgreSQL Total Records: {pg_count:,} (Expected: {total_records:,})")
    print(f"[VERIFY] PostgreSQL Flagged Anomalies: {pg_anomalies:,} (Expected: {scored_df['is_anomaly'].sum():,})")

    assert pg_count == total_records, f"Row count mismatch: PG={pg_count} vs Parquet={total_records}"
    print("\n[SUCCESS] PostgreSQL transaction anomaly layer fully populated and verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
