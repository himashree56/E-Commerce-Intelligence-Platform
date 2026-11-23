"""
Generates and persists Gold customer repeat-purchase datasets for Phase 6.

Creates:
- data/gold/gold_repeat_purchase_train.parquet
- data/gold/gold_repeat_purchase_val.parquet
- data/gold/gold_repeat_purchase_test.parquet
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.features.repeat_purchase_features import extract_customer_snapshot
from src.models.config import RepeatPurchaseConfig


def main() -> int:
    print("=" * 85)
    print("    E-COMMERCE INTELLIGENCE PLATFORM - GENERATE REPEAT-PURCHASE DATASETS")
    print("=" * 85)

    config = RepeatPurchaseConfig()
    t_start = time.time()

    # 1. Load Silver Dataset
    print(f"Loading Silver Parquet from: {config.silver_parquet_path}...")
    silver_df = pd.read_parquet(config.silver_parquet_path)
    print(f"Loaded {len(silver_df):,} Silver transactions in {time.time()-t_start:.2f}s.")

    # 2. Generate Training Snapshots
    print(f"\n--- Extracting Training Snapshots: {config.train_snapshots} ---")
    train_dfs = []
    for snap in config.train_snapshots:
        t0 = time.time()
        snap_df = extract_customer_snapshot(silver_df, snap, horizon_days=config.horizon_days)
        train_dfs.append(snap_df)
        pos = int(snap_df["label_30d"].sum())
        print(f"Snapshot {snap}: {len(snap_df):,} customers, {pos:,} positive ({pos/len(snap_df)*100:.2f}%) [{time.time()-t0:.2f}s]")

    train_df = pd.concat(train_dfs, ignore_index=True)
    config.gold_train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(config.gold_train_path, index=False, engine="pyarrow", compression="snappy")
    print(f"Saved Train dataset to {config.gold_train_path} ({len(train_df):,} rows, {config.gold_train_path.stat().st_size / 1024 / 1024:.2f} MB).")

    # 3. Generate Validation Snapshot
    print(f"\n--- Extracting Validation Snapshot: {config.val_snapshot} ---")
    t0 = time.time()
    val_df = extract_customer_snapshot(silver_df, config.val_snapshot, horizon_days=config.horizon_days)
    val_pos = int(val_df["label_30d"].sum())
    print(f"Snapshot {config.val_snapshot}: {len(val_df):,} customers, {val_pos:,} positive ({val_pos/len(val_df)*100:.2f}%) [{time.time()-t0:.2f}s]")
    val_df.to_parquet(config.gold_val_path, index=False, engine="pyarrow", compression="snappy")
    print(f"Saved Validation dataset to {config.gold_val_path} ({len(val_df):,} rows, {config.gold_val_path.stat().st_size / 1024 / 1024:.2f} MB).")

    # 4. Generate Test Snapshot
    print(f"\n--- Extracting Test Snapshot: {config.test_snapshot} ---")
    t0 = time.time()
    test_df = extract_customer_snapshot(silver_df, config.test_snapshot, horizon_days=config.horizon_days)
    test_pos = int(test_df["label_30d"].sum())
    print(f"Snapshot {config.test_snapshot}: {len(test_df):,} customers, {test_pos:,} positive ({test_pos/len(test_df)*100:.2f}%) [{time.time()-t0:.2f}s]")
    test_df.to_parquet(config.gold_test_path, index=False, engine="pyarrow", compression="snappy")
    print(f"Saved Test dataset to {config.gold_test_path} ({len(test_df):,} rows, {config.gold_test_path.stat().st_size / 1024 / 1024:.2f} MB).")

    total_duration = time.time() - t_start
    print("\n" + "=" * 85)
    print("DATASET GENERATION SUMMARY")
    print("=" * 85)
    print(f"Train Dataset:      {len(train_df):,} rows ({train_df['label_30d'].sum():,} positive, {train_df['label_30d'].mean()*100:.2f}%)")
    print(f"Validation Dataset: {len(val_df):,} rows ({val_df['label_30d'].sum():,} positive, {val_df['label_30d'].mean()*100:.2f}%)")
    print(f"Test Dataset:       {len(test_df):,} rows ({test_df['label_30d'].sum():,} positive, {test_df['label_30d'].mean()*100:.2f}%)")
    print(f"Total Extraction Time: {total_duration:.2f} seconds.")
    print("=" * 85)
    return 0


if __name__ == "__main__":
    sys.exit(main())
