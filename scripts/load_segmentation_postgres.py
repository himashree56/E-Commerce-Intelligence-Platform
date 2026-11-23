"""CLI Script for Loading ML Customer Segmentation Predictions into PostgreSQL (Phase 4)."""
import os
import sys
from pathlib import Path

# Silence loky warning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import select, func, text

from src.database.connection import DatabaseManager
from src.database.loader import GoldDatabaseLoader
from src.database.models import CustomerSegmentationKMeans, CustomerRFMAnalytics
from src.database.schema_manager import SchemaManager
from src.models.config import SegmentationConfig
from src.models.customer_segmentation import CustomerSegmentationModel


def main() -> int:
    print("=" * 85)
    print("    E-COMMERCE INTELLIGENCE PLATFORM - LOAD SEGMENTATION TO POSTGRESQL")
    print("=" * 85)

    config = SegmentationConfig()
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
    if CustomerSegmentationKMeans.__tablename__ not in created_tables:
        print(f"[ERROR] Table '{CustomerSegmentationKMeans.__tablename__}' not found in DB!", file=sys.stderr)
        return 1

    # 3. Load or generate predictions
    predictions_path = config.base_dir / "data" / "gold" / "gold_customer_segmentation_kmeans.parquet"
    if predictions_path.exists():
        print(f"[DATA] Loading predictions from: {predictions_path}")
        preds_df = pd.read_parquet(predictions_path)
    else:
        print("[DATA] Generating fresh predictions from trained model...")
        model = CustomerSegmentationModel.load(config.model_dir)
        rfm_df = pd.read_parquet(config.gold_rfm_path)
        preds_df = model.predict(rfm_df)
        preds_df.to_parquet(predictions_path, index=False)

    print(f"[DATA] Prepared {len(preds_df):,} customer predictions for database loading.")

    # 4. Load into PostgreSQL via idempotent upsert
    loader = GoldDatabaseLoader(is_test=False)
    with db_mgr.session_scope() as session:
        load_summary = loader.load_customer_segmentation(session, preds_df)

    print(f"[DATABASE] Loaded {load_summary.loaded_rows:,} records in {load_summary.duration_seconds:.3f}s. Status: {load_summary.status}")

    # 5. Verification and reconciliation
    with db_mgr.engine.connect() as conn:
        pg_count = conn.execute(select(func.count()).select_from(CustomerSegmentationKMeans)).scalar()
        rfm_count = conn.execute(select(func.count()).select_from(CustomerRFMAnalytics)).scalar()
        null_clusters = conn.execute(
            select(func.count()).select_from(CustomerSegmentationKMeans).where(CustomerSegmentationKMeans.cluster_id.is_(None))
        ).scalar()

        # Cluster distribution in PostgreSQL
        q_dist = text("""
            SELECT cluster_id, cluster_label, COUNT(*) as cust_count,
                   ROUND(SUM(monetary_net), 2) as total_revenue,
                   ROUND(AVG(recency_days), 1) as avg_recency,
                   ROUND(AVG(frequency), 1) as avg_frequency
            FROM customer_segmentation_kmeans
            GROUP BY cluster_id, cluster_label
            ORDER BY cluster_id ASC;
        """)
        dist_df = pd.read_sql_query(q_dist, conn)

    print("\n" + "=" * 85)
    print("              POSTGRESQL SEGMENTATION RECONCILIATION")
    print("=" * 85)
    print(f"Eligible RFM Customers in DB:         {rfm_count:,}")
    print(f"ML Segmentation Records in DB:        {pg_count:,}")
    print(f"Null Clusters Count:                  {null_clusters}")
    print(f"Population Reconciliation Status:     {'PASS' if pg_count == rfm_count == 5939 else 'FAIL'}")

    print("\n--- PostgreSQL Cluster Breakdown ---")
    print(dist_df.to_string(index=False))
    print("=" * 85)

    if pg_count == rfm_count == 5939 and null_clusters == 0:
        print("[SUCCESS] All 5,939 customers successfully reconciled in PostgreSQL!")
        return 0
    else:
        print("[ERROR] Segmentation reconciliation failed!", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
