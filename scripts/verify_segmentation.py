"""CLI Verification Script for Phase 4 K-Means Customer Segmentation."""
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import select, func, text

from src.database.connection import DatabaseManager
from src.database.models import CustomerSegmentationKMeans, CustomerRFMAnalytics
from src.models.config import SegmentationConfig
from src.models.customer_segmentation import CustomerSegmentationModel


def main() -> int:
    print("=" * 85)
    print("    E-COMMERCE INTELLIGENCE PLATFORM - PHASE 4 SEGMENTATION VERIFICATION")
    print("=" * 85)

    config = SegmentationConfig()
    db_mgr = DatabaseManager()

    # 1. Health check
    health = db_mgr.check_health()
    if health.get("status") != "HEALTHY":
        print(f"[ERROR] Database is unhealthy: {health.get('error')}", file=sys.stderr)
        return 1

    with db_mgr.engine.connect() as conn:
        # Check table existence and counts
        rfm_count = conn.execute(select(func.count()).select_from(CustomerRFMAnalytics)).scalar()
        ml_count = conn.execute(select(func.count()).select_from(CustomerSegmentationKMeans)).scalar()

        # Check for orphan records (FK integrity)
        orphan_q = text("""
            SELECT COUNT(*) FROM customer_segmentation_kmeans ml
            LEFT JOIN customer_rfm_analytics rfm ON ml.customer_id = rfm.customer_id
            WHERE rfm.customer_id IS NULL;
        """)
        orphans = conn.execute(orphan_q).scalar()

        # Check for null clusters
        null_clusters = conn.execute(
            select(func.count()).select_from(CustomerSegmentationKMeans).where(
                CustomerSegmentationKMeans.cluster_id.is_(None)
            )
        ).scalar()

        # Check cluster distribution
        dist_q = text("""
            SELECT cluster_id, cluster_label,
                   COUNT(*) as customer_count,
                   ROUND((COUNT(*)::numeric / 5939) * 100, 2) as population_pct,
                   ROUND(SUM(monetary_net), 2) as total_revenue,
                   ROUND(AVG(recency_days), 1) as mean_recency,
                   ROUND(AVG(frequency), 1) as mean_frequency,
                   ROUND(AVG(distance_to_centroid), 4) as avg_centroid_distance
            FROM customer_segmentation_kmeans
            GROUP BY cluster_id, cluster_label
            ORDER BY cluster_id ASC;
        """)
        dist_df = pd.read_sql_query(dist_q, conn)

        # Check rule-based segment preservation (verify customer_rfm_analytics is untampered)
        rule_seg_count = conn.execute(
            text("SELECT COUNT(DISTINCT customer_segment) FROM customer_rfm_analytics;")
        ).scalar()

        # Check parity with saved parquet predictions
        pred_parquet = config.base_dir / "data" / "gold" / "gold_customer_segmentation_kmeans.parquet"
        parquet_matched = False
        if pred_parquet.exists():
            df_pq = pd.read_parquet(pred_parquet)
            df_pg = pd.read_sql_query(
                "SELECT customer_id, cluster_id, cluster_label FROM customer_segmentation_kmeans ORDER BY customer_id ASC;",
                conn,
            )
            df_pq_sorted = (
                df_pq[["customer_id", "cluster_id", "cluster_label"]]
                .astype({"customer_id": "int64", "cluster_id": "int64", "cluster_label": "str"})
                .sort_values("customer_id")
                .reset_index(drop=True)
            )
            df_pg_sorted = (
                df_pg[["customer_id", "cluster_id", "cluster_label"]]
                .astype({"customer_id": "int64", "cluster_id": "int64", "cluster_label": "str"})
                .sort_values("customer_id")
                .reset_index(drop=True)
            )
            parquet_matched = df_pq_sorted.equals(df_pg_sorted)

    print("\n--- Verification Checklist ---")
    print(f"1. Eligible RFM Customers in DB:      {rfm_count:,} (Expected: 5,939)")
    print(f"2. ML Segmentation Records in DB:     {ml_count:,} (Expected: 5,939)")
    print(f"3. Referential Integrity (Orphans):   {orphans} (Expected: 0)")
    print(f"4. Null Clusters Detected:            {null_clusters} (Expected: 0)")
    print(f"5. Rule-Based Segments Preserved:     {rule_seg_count} distinct segments (Expected: 8)")
    print(f"6. Parquet vs PostgreSQL Parity:      {'MATCHED (100% Identical)' if parquet_matched else 'MISMATCH'}")

    print("\n--- Verified PostgreSQL ML Clusters ---")
    print(dist_df.to_string(index=False))

    all_passed = (
        rfm_count == 5939
        and ml_count == 5939
        and orphans == 0
        and null_clusters == 0
        and rule_seg_count == 8
        and parquet_matched
    )

    print("\n" + "=" * 85)
    if all_passed:
        print("[SUCCESS] Phase 4 Customer Segmentation fully verified in PostgreSQL!")
        print("=" * 85)
        return 0
    else:
        print("[ERROR] Phase 4 verification checks failed!", file=sys.stderr)
        print("=" * 85)
        return 1


if __name__ == "__main__":
    sys.exit(main())
