"""
Audit Silver dataset for Phase 5 Isolation Forest Anomaly Detection.
"""
import pandas as pd
import numpy as np

def audit_silver():
    silver_path = "data/silver/online_retail_silver.parquet"
    print(f"Loading {silver_path}...")
    df = pd.read_parquet(silver_path)
    print(f"Dataset shape: {df.shape}")

    print("\n--- TRANSACTION STATUS DISTRIBUTION ---")
    ts = df["transaction_status"].value_counts(dropna=False)
    for k, v in ts.items():
        print(f"  {k}: {v:,} ({v / len(df) * 100:.2f}%)")

    print("\n--- QUALITY FLAG DISTRIBUTION ---")
    qf = df["quality_flag"].value_counts(dropna=False)
    for k, v in qf.items():
        print(f"  {k}: {v:,} ({v / len(df) * 100:.2f}%)")

    print("\n--- BOOLEAN FLAGS SUMMARY ---")
    flag_cols = [
        "is_cancellation", "is_duplicate", "is_return", 
        "is_invalid_price", "is_missing_customer", 
        "is_non_product", "is_ambiguous_cancellation"
    ]
    for col in flag_cols:
        count = df[col].sum()
        print(f"  {col}: {count:,} ({count / len(df) * 100:.2f}%)")

    print("\n--- QUANTITY DISTRIBUTION ---")
    print(df["quantity"].describe(percentiles=[0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999]))
    neg_q = (df["quantity"] < 0).sum()
    zero_q = (df["quantity"] == 0).sum()
    pos_q = (df["quantity"] > 0).sum()
    print(f"  Negative (<0): {neg_q:,}")
    print(f"  Zero (==0): {zero_q:,}")
    print(f"  Positive (>0): {pos_q:,}")

    print("\n--- UNIT PRICE DISTRIBUTION ---")
    print(df["unit_price"].describe(percentiles=[0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999]))
    neg_p = (df["unit_price"] < 0).sum()
    zero_p = (df["unit_price"] == 0).sum()
    pos_p = (df["unit_price"] > 0).sum()
    print(f"  Negative (<0): {neg_p:,}")
    print(f"  Zero (==0): {zero_p:,}")
    print(f"  Positive (>0): {pos_p:,}")

    print("\n--- LINE TOTAL DISTRIBUTION ---")
    print(df["line_total"].describe(percentiles=[0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999]))

    print("\n--- INVOICE C496350 ---")
    c49 = df[df["invoice_no"] == "C496350"]
    print(c49[["invoice_no", "stock_code", "description", "quantity", "unit_price", "line_total", "is_ambiguous_cancellation", "transaction_status"]])

    print("\n--- EXTREME POSITIVE TRANSACTIONS (Top 5 line_total) ---")
    top_pos = df.nlargest(5, "line_total")
    print(top_pos[["invoice_no", "stock_code", "description", "quantity", "unit_price", "line_total", "customer_id", "transaction_status"]])

    print("\n--- EXTREME NEGATIVE TRANSACTIONS (Bottom 5 line_total) ---")
    top_neg = df.nsmallest(5, "line_total")
    print(top_neg[["invoice_no", "stock_code", "description", "quantity", "unit_price", "line_total", "customer_id", "transaction_status"]])

    print("\n--- MISSING CUSTOMER TRANSACTIONS ---")
    missing_cust = df[df["is_missing_customer"]]
    print(f"  Count: {len(missing_cust):,}")
    print(f"  Status breakdown:")
    print(missing_cust["transaction_status"].value_counts())

    print("\n--- DUPLICATE RECORDS ---")
    dups = df[df["is_duplicate"]]
    print(f"  Count: {len(dups):,}")
    print(f"  Status breakdown:")
    print(dups["transaction_status"].value_counts())

if __name__ == "__main__":
    audit_silver()
