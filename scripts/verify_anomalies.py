"""
Verification Script for Transaction Anomaly Detection (Phase 5).

Audits row count parity, lineage stability, score distributions,
flagged status allocations, and financial non-interference.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def verify_anomalies() -> int:
    print("=" * 85)
    print("    E-COMMERCE INTELLIGENCE PLATFORM - PHASE 5 ANOMALY AUDIT & VERIFICATION")
    print("=" * 85)

    silver_path = Path("data/silver/online_retail_silver.parquet")
    gold_anomalies_path = Path("data/gold/gold_transaction_anomalies_isolation_forest.parquet")
    model_dir = Path("models/anomaly_detection")

    # 1. Existence checks
    assert silver_path.exists(), f"Missing Silver dataset: {silver_path}"
    assert gold_anomalies_path.exists(), f"Missing Gold anomalies dataset: {gold_anomalies_path}"
    assert model_dir.exists(), f"Missing model artifacts: {model_dir}"

    print("[FILES] Verified existence of Silver, Gold Anomalies, and Model Artifacts.")

    # 2. Load Datasets
    t0 = time.time()
    silver_df = pd.read_parquet(silver_path)
    anomalies_df = pd.read_parquet(gold_anomalies_path)
    print(f"[DATA] Loaded Silver ({len(silver_df):,} rows) and Anomalies ({len(anomalies_df):,} rows) in {time.time()-t0:.2f}s.")

    # 3. Population & Lineage Invariants
    total_rows = len(silver_df)
    assert len(anomalies_df) == total_rows, f"Row count mismatch: {len(anomalies_df)} vs {total_rows}"
    assert anomalies_df["transaction_id"].nunique() == total_rows, "Duplicate transaction_id detected!"
    assert (anomalies_df["source_row_id"].values == np.arange(total_rows)).all(), "source_row_id does not match Silver index!"
    print(f"[LINEAGE] 100% row preservation and unique deterministic lineage verified ({total_rows:,} rows).")

    # 4. Anomaly Flags & Scores Breakdown
    total_anomalies = int(anomalies_df["is_anomaly"].sum())
    total_normal = total_rows - total_anomalies
    anomaly_pct = total_anomalies / total_rows * 100

    print(f"\n[SCORES] Anomaly Detection Breakdown:")
    print(f"  Total Flagged Anomalies: {total_anomalies:,} ({anomaly_pct:.2f}%)")
    print(f"  Total Normal Inliers:   {total_normal:,} ({100 - anomaly_pct:.2f}%)")

    # Check status-level anomaly distribution
    status_counts = anomalies_df.groupby("transaction_status")["is_anomaly"].agg(["count", "sum"])
    status_counts["pct"] = (status_counts["sum"] / status_counts["count"] * 100).round(2)
    print("\n[BREAKDOWN BY STATUS]")
    for st, row in status_counts.iterrows():
        print(f"  {st:<20}: Total={int(row['count']):>9,}, Anomalies={int(row['sum']):>6,} ({row['pct']:>6.2f}%)")

    # Non-commercial write-offs must be 100% flagged
    write_off_count = status_counts.loc["INVENTORY_WRITE_OFF", "count"]
    write_off_anomalies = status_counts.loc["INVENTORY_WRITE_OFF", "sum"]
    assert write_off_anomalies == write_off_count, "All inventory write-offs must be flagged as anomalous!"

    # Cancellations must NOT be 100% flagged (retail returns are normal operations)
    cancellation_pct = status_counts.loc["CANCELLATION", "pct"]
    assert cancellation_pct < 15.0, f"Excessive cancellation flagging: {cancellation_pct}%"
    print(f"[RETAIL VALIDATION] Normal retail returns preserved: only {cancellation_pct:.2f}% of cancellations flagged.")

    # 5. Diagnostic Explanations Check
    flagged_reasons = anomalies_df[anomalies_df["is_anomaly"]]["diagnostic_reasons"]
    assert (flagged_reasons != "").all(), "All flagged transactions must have diagnostic reason codes!"
    unflagged_reasons = anomalies_df[~anomalies_df["is_anomaly"]]["diagnostic_reasons"]
    assert (unflagged_reasons == "").all(), "Unflagged transactions must have empty reason codes!"
    print("[INTERPRETABILITY] 100% of flagged anomalies have valid diagnostic reason codes.")

    # 6. Specific Case Audit
    c49 = anomalies_df[anomalies_df["invoice_no"] == "C496350"]
    assert len(c49) == 1, "Invoice C496350 missing!"
    assert bool(c49["is_anomaly"].iloc[0]) is True, "C496350 must be flagged as anomalous!"
    assert "HIGH_UNIT_PRICE" in c49["diagnostic_reasons"].iloc[0], "C496350 must trigger HIGH_UNIT_PRICE!"
    print(f"[AUDIT C496350] Invoice C496350 verified: score={c49['anomaly_score'].iloc[0]:.6f}, reasons='{c49['diagnostic_reasons'].iloc[0]}'.")

    # 7. Financial Non-Interference
    silver_rev = float(silver_df["net_revenue"].sum())
    assert round(silver_rev, 2) == 19012910.85, f"Historical revenue altered: {silver_rev}"
    print(f"[FINANCIAL INTEGRITY] Historical Net Revenue £{silver_rev:,.2f} is 100% unaltered.")

    print("\n" + "=" * 85)
    print("    PHASE 5 VERIFICATION SUCCESSFUL — ALL QUALITY GATES PASSED")
    print("=" * 85)
    return 0


if __name__ == "__main__":
    sys.exit(verify_anomalies())
