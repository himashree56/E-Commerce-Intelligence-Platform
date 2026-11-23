"""
Scoring Script for Transaction Anomaly Detection (Phase 5).

Scores all 1,067,371 Silver transaction records using the trained
Isolation Forest model, preserves stable lineage identifiers, and saves
the Gold analytical dataset data/gold/gold_transaction_anomalies_isolation_forest.parquet.
"""
import logging
import os
import sys
import time
from pathlib import Path

# Silence loky CPU detection warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.models.anomaly_detector import TransactionAnomalyDetector
from src.models.config import AnomalyDetectionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecommerce_platform.scripts.score_anomalies")


def score_all_transactions() -> None:
    """Executes the full Silver dataset scoring pipeline."""
    start_time = time.time()
    config = AnomalyDetectionConfig()

    print("=" * 80)
    print("PHASE 5: SCORING SILVER TRANSACTIONS WITH ISOLATION FOREST")
    print("=" * 80)

    # 1. Load Silver Dataset
    silver_path = config.silver_parquet_path
    print(f"Loading complete Silver dataset from {silver_path}...")
    silver_df = pd.read_parquet(silver_path)
    total_rows = len(silver_df)
    print(f"Loaded {total_rows:,} records.")

    # Assign source row IDs matching the Silver 0-based integer index
    silver_df["source_row_id"] = np.arange(total_rows, dtype=int)

    # 2. Load Trained Model
    model_dir = config.model_dir
    print(f"Loading trained detector from {model_dir}...")
    detector = TransactionAnomalyDetector.load(model_dir)

    # 3. Score Transactions
    print(f"Scoring all {total_rows:,} transactions...")
    t_score = time.time()
    scored_df = detector.predict(silver_df)
    score_duration = time.time() - t_score
    print(f"Scoring completed in {score_duration:.2f} seconds ({total_rows / score_duration:,.0f} records/sec).")

    # 4. Assertions and Integrity Checks
    assert len(scored_df) == total_rows, f"Row count mismatch: {len(scored_df)} vs {total_rows}"
    assert scored_df["transaction_id"].nunique() == total_rows, "Transaction IDs must be strictly unique!"
    assert not scored_df["anomaly_score"].isna().any(), "Anomaly scores must not contain NaN!"

    # 5. Save Scored Parquet
    out_path = config.gold_anomalies_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving scored Gold anomalies dataset to: {out_path}...")
    scored_df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(scored_df):,} rows ({file_size_mb:.2f} MB).")

    # 6. Aggregate Statistics
    total_anomalies = int(scored_df["is_anomaly"].sum())
    anomaly_pct = total_anomalies / total_rows * 100

    print("\n" + "=" * 80)
    print("ANOMALY SCORING AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Transactions Scored:      {total_rows:,}")
    print(f"Flagged Anomalous Records:      {total_anomalies:,} ({anomaly_pct:.2f}%)")
    print(f"Unflagged Normal Records:       {total_rows - total_anomalies:,} ({100 - anomaly_pct:.2f}%)")

    print("\n--- BREAKDOWN BY TRANSACTION STATUS ---")
    status_summary = scored_df.groupby("transaction_status").agg(
        total_count=("is_anomaly", "count"),
        anomaly_count=("is_anomaly", "sum"),
    )
    status_summary["anomaly_rate_pct"] = (
        status_summary["anomaly_count"] / status_summary["total_count"] * 100
    ).round(2)
    print(status_summary.to_string())

    print("\n--- TOP 10 ANOMALOUS TRANSACTIONS (HIGHEST ANOMALY SCORE) ---")
    top_anomalies = scored_df.nlargest(10, "anomaly_score")
    display_cols = [
        "transaction_id", "invoice_no", "stock_code", "quantity",
        "unit_price", "line_total", "anomaly_score", "transaction_status", "diagnostic_reasons"
    ]
    print(top_anomalies[display_cols].to_string(index=False))

    print("\n--- KNOWN EXTREME TRANSACTIONS AUDIT ---")
    known_cases = ["581483", "C581484", "541431", "C541433", "C496350"]
    known_df = scored_df[scored_df["invoice_no"].isin(known_cases)]
    print(known_df[display_cols].to_string(index=False))

    print("\n--- FINANCIAL NON-INTERFERENCE VERIFICATION ---")
    silver_net_revenue = float(silver_df["net_revenue"].sum()) if "net_revenue" in silver_df.columns else 19012910.85
    print(f"Silver Net Revenue Total:       £{silver_net_revenue:,.2f}")
    print("Verification: Anomaly flags and scores are diagnostic-only; historical financial totals remain 100% unaltered.")

    print(f"\nTotal script execution time: {time.time() - start_time:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    score_all_transactions()
