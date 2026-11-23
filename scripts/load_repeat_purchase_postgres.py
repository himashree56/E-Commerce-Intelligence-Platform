"""
CLI Script for Loading XGBoost Repeat-Purchase Predictions into PostgreSQL (Phase 6).

Scores the test snapshot (2011-10-01) with the trained XGBoost model and persists
predictions into customer_repeat_purchase_prediction table via batched, idempotent upsert.
"""
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import func, select, text

from src.database.connection import DatabaseManager
from src.database.loader import GoldDatabaseLoader
from src.database.models import CustomerRepeatPurchasePrediction
from src.database.schema_manager import SchemaManager
from src.models.config import RepeatPurchaseConfig
from src.models.repeat_purchase import RepeatPurchasePredictor


def main() -> int:
    print("=" * 85)
    print("    E-COMMERCE INTELLIGENCE PLATFORM - LOAD REPEAT-PURCHASE PREDICTIONS")
    print("=" * 85)

    config = RepeatPurchaseConfig()
    db_mgr = DatabaseManager()

    # 1. Health Check
    health = db_mgr.check_health()
    if health.get("status") != "HEALTHY":
        print(f"[ERROR] Database connection unhealthy: {health.get('error')}", file=sys.stderr)
        return 1

    # 2. Schema Verification
    schema_mgr = SchemaManager(db_manager=db_mgr)
    created_tables = schema_mgr.create_all_tables(is_test=False)
    print(f"[SCHEMA] Verified schema. Tables in DB: {len(created_tables)}")
    if CustomerRepeatPurchasePrediction.__tablename__ not in created_tables:
        print(f"[ERROR] Table '{CustomerRepeatPurchasePrediction.__tablename__}' missing in DB!", file=sys.stderr)
        return 1

    # 3. Load Trained Model & Test Snapshot
    print(f"[MODEL] Loading trained RepeatPurchasePredictor from: {config.model_dir}...")
    predictor = RepeatPurchasePredictor.load(config.model_dir)

    print(f"[DATA] Reading test snapshot dataset from: {config.gold_test_path}...")
    test_df = pd.read_parquet(config.gold_test_path)
    total_records = len(test_df)
    print(f"[DATA] Loaded {total_records:,} customer snapshots as of {config.test_snapshot}.")

    # 4. Generate Predictions
    probs = predictor.predict_proba(test_df)
    preds = predictor.predict(test_df)

    scoring_ts = datetime.now(timezone.utc)
    feature_dicts = test_df[predictor.feature_names].to_dict(orient="records")

    pred_df = pd.DataFrame({
        "prediction_id": [f"PRED_{int(cid)}_{config.test_snapshot}" for cid in test_df["customer_id"]],
        "customer_id": test_df["customer_id"].astype(int),
        "prediction_as_of_date": pd.to_datetime(config.test_snapshot).date(),
        "prediction_horizon_days": config.horizon_days,
        "repeat_purchase_probability": [round(float(p), 6) for p in probs],
        "predicted_label": [bool(pred) for pred in preds],
        "model_version": config.model_version,
        "feature_snapshot": feature_dicts,
        "scoring_timestamp": scoring_ts,
    })

    # 5. Load into PostgreSQL
    loader = GoldDatabaseLoader(is_test=False, batch_size=2000)
    print(f"[DATABASE] Loading {len(pred_df):,} prediction records into PostgreSQL...")
    with db_mgr.session_scope() as session:
        load_summary = loader.load_repeat_purchase_predictions(session, pred_df)

    print(f"[DATABASE] Loaded {load_summary.loaded_rows:,} records in {load_summary.duration_seconds:.2f}s.")

    # 6. Verification and Idempotence Check
    with db_mgr.engine.connect() as conn:
        pg_count = conn.execute(select(func.count()).select_from(CustomerRepeatPurchasePrediction)).scalar()
        pg_pos = conn.execute(
            select(func.count())
            .select_from(CustomerRepeatPurchasePrediction)
            .where(CustomerRepeatPurchasePrediction.predicted_label == True)
        ).scalar()

    print(f"[VERIFY] PostgreSQL Total Records: {pg_count:,} (Expected: {total_records:,})")
    print(f"[VERIFY] PostgreSQL Predicted Positives: {pg_pos:,} (Expected: {int(pred_df['predicted_label'].sum()):,})")
    assert pg_count == total_records, f"Mismatch: PG={pg_count} vs Expected={total_records}"

    print("\n[SUCCESS] Customer repeat-purchase predictions successfully loaded and verified in PostgreSQL!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
