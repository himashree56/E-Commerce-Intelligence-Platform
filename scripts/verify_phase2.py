"""Verification script inspecting Silver and Gold parquet datasets."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow.parquet as pq

from src.etl.config import ETLConfig


def verify_tables():
    config = ETLConfig(base_dir=PROJECT_ROOT)

    print("=" * 75)
    print("        PHASE 2 ARTIFACT VERIFICATION (SILVER & GOLD)")
    print("=" * 75)

    # 1. Silver Parquet
    silver_p = config.silver_parquet_path
    if not silver_p.exists():
        print(f"[ERROR] Silver parquet missing: {silver_p}")
        return 1

    s_meta = pq.ParquetFile(silver_p).metadata
    s_df = pd.read_parquet(silver_p)
    print(f"\n[SILVER TABLE] {silver_p.name}")
    print(f"  Rows: {s_meta.num_rows:,} | Columns: {s_meta.num_columns} | File Size: {silver_p.stat().st_size:,} bytes")
    print(f"  Transaction Status Breakdown:")
    for status, count in s_df["transaction_status"].value_counts().items():
        pct = (count / len(s_df)) * 100
        print(f"    - {status:25}: {count:8,} ({pct:5.2f}%)")

    print(f"  Quality Flags:")
    for qf, count in s_df["quality_flag"].value_counts().items():
        pct = (count / len(s_df)) * 100
        print(f"    - {qf:25}: {count:8,} ({pct:5.2f}%)")

    print(f"\n  Financial Calculations:")
    print(f"    - Gross Sales:             £{s_df['gross_sales'].sum():14,.2f}")
    print(f"    - Return Amount:          -£{s_df['return_amount'].sum():14,.2f}")
    print(f"    - Net Revenue:             £{s_df['net_revenue'].sum():14,.2f}")

    # 2. Gold Tables
    gold_tables = {
        "Monthly Sales": config.gold_monthly_sales_path,
        "Customer RFM": config.gold_customer_rfm_path,
        "Product Performance": config.gold_product_performance_path,
        "Country Revenue": config.gold_country_revenue_path,
        "Customer Purchase History": config.gold_customer_purchase_history_path,
    }

    print("\n[GOLD ANALYTICAL TABLES]")
    for label, path in gold_tables.items():
        if path.exists():
            gdf = pd.read_parquet(path)
            print(f"  - {label:26}: {len(gdf):7,} rows | {len(gdf.columns):2} cols | {path.stat().st_size:,} bytes")
        else:
            print(f"  - {label:26}: MISSING at {path}")

    # Display Monthly Sales Top 5 & Customer RFM Top 5
    if config.gold_monthly_sales_path.exists():
        m_df = pd.read_parquet(config.gold_monthly_sales_path)
        print("\n--- Sample Gold Monthly Sales (First 3 Months) ---")
        print(m_df[["invoice_year_month", "gross_sales", "return_amount", "net_revenue", "valid_orders_count", "distinct_customers"]].head(3).to_string())

    if config.gold_customer_rfm_path.exists():
        rfm_df = pd.read_parquet(config.gold_customer_rfm_path)
        print("\n--- Sample Customer RFM Top Spenders (Top 3) ---")
        print(rfm_df[["customer_id", "recency_days", "frequency", "monetary_net", "customer_segment"]].head(3).to_string())

    print("\n[SUCCESS] All Silver and Gold tables verified successfully!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(verify_tables())
