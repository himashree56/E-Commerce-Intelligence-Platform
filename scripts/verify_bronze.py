"""Verification script for the Bronze Parquet dataset."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow.parquet as pq

from src.ingestion.config import IngestionConfig
from src.ingestion.schema import StandardizedSchema


def verify_bronze_dataset(parquet_path: Path) -> int:
    """Verifies schema, counts, and preservation in Bronze Parquet file."""
    if not parquet_path.exists():
        print(f"[ERROR] Parquet file not found at: {parquet_path}")
        return 1

    print("=" * 70)
    print("           BRONZE PARQUET DATASET VERIFICATION")
    print("=" * 70)
    print(f"File Path: {parquet_path}")
    print(f"File Size: {parquet_path.stat().st_size:,} bytes")

    # Read PyArrow metadata
    parquet_file = pq.ParquetFile(parquet_path)
    meta = parquet_file.metadata
    print(f"Total Rows (PyArrow): {meta.num_rows:,}")
    print(f"Total Columns (PyArrow): {meta.num_columns}")
    print(f"Row Groups: {meta.num_row_groups}")

    # Read into pandas
    df = pd.read_parquet(parquet_path)
    print(f"\nDataFrame Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n--- Column Dtypes ---")
    for col, dtype in df.dtypes.items():
        print(f"  {col:22}: {dtype}")

    print("\n--- Null Value Counts & Percentages ---")
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = (null_count / len(df)) * 100
        print(f"  {col:22}: {null_count:7,} ({null_pct:5.2f}%)")

    # Preservation Checks
    print("\n--- Preserved Anomalies & Integrity Verification ---")
    cancellations = int(df[StandardizedSchema.IS_CANCELLATION].sum())
    neg_qty = int((df[StandardizedSchema.QUANTITY] < 0).sum())
    zero_qty = int((df[StandardizedSchema.QUANTITY] == 0).sum())
    pos_qty = int((df[StandardizedSchema.QUANTITY] > 0).sum())
    null_cust = int(df[StandardizedSchema.CUSTOMER_ID].isna().sum())
    unique_invoices = int(df[StandardizedSchema.INVOICE_NO].nunique())
    unique_customers = int(df[StandardizedSchema.CUSTOMER_ID].dropna().nunique())
    unique_countries = int(df[StandardizedSchema.COUNTRY].nunique())

    print(f"  Cancellations Flagged (is_cancellation) : {cancellations:,}")
    print(f"  Negative Quantities Preserved           : {neg_qty:,}")
    print(f"  Zero Quantities Preserved               : {zero_qty:,}")
    print(f"  Positive Quantities Preserved           : {pos_qty:,}")
    print(f"  Missing Customer IDs Preserved          : {null_cust:,}")
    print(f"  Unique Invoices                         : {unique_invoices:,}")
    print(f"  Unique Customer IDs                     : {unique_customers:,}")
    print(f"  Unique Countries                        : {unique_countries:,}")

    # Sheet breakdown
    if StandardizedSchema.SHEET_NAME in df.columns:
        print("\n--- Rows by Source Sheet ---")
        for sheet, count in df[StandardizedSchema.SHEET_NAME].value_counts().items():
            print(f"  {sheet:25}: {count:,}")

    print("\n--- Sample Records (First 3) ---")
    print(df.head(3).to_string())

    print("\n--- Sample Cancellations (First 3) ---")
    canc_df = df[df[StandardizedSchema.IS_CANCELLATION]].head(3)
    if not canc_df.empty:
        print(canc_df.to_string())
    else:
        print("  None found.")

    print("\n[SUCCESS] Bronze Parquet dataset verified successfully!")
    print("=" * 70)
    return 0


def main() -> int:
    config = IngestionConfig(base_dir=PROJECT_ROOT)
    return verify_bronze_dataset(config.unified_bronze_path)


if __name__ == "__main__":
    sys.exit(main())
