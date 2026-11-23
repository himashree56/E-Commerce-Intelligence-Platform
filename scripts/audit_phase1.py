"""Targeted audit script for Phase 1 Bronze dataset."""
from pathlib import Path
import pandas as pd

def run_audit():
    bronze_path = Path("data/bronze/online_retail_bronze.parquet")
    df = pd.read_parquet(bronze_path)
    
    print("=" * 70)
    print("1. AUDIT: CANCELLATIONS VS NEGATIVE QUANTITIES")
    print("=" * 70)
    c_invoice = df["invoice_no"].str.startswith(("C", "c"), na=False)
    neg_qty = df["quantity"] < 0
    zero_qty = df["quantity"] == 0
    both = c_invoice & neg_qty
    c_only = c_invoice & ~neg_qty
    neg_only = ~c_invoice & neg_qty
    
    print(f"Total rows in Bronze:                  {len(df):,}")
    print(f"Total flagged is_cancellation:         {df['is_cancellation'].sum():,}")
    print(f"Invoice starts with 'C':               {c_invoice.sum():,}")
    print(f"Quantity < 0:                          {neg_qty.sum():,}")
    print(f"Quantity == 0:                         {zero_qty.sum():,}")
    print(f"Both Invoice 'C' AND Quantity < 0:     {both.sum():,}")
    print(f"Invoice 'C' BUT Quantity >= 0:         {c_only.sum():,}")
    print(f"Quantity < 0 BUT Invoice NOT 'C':      {neg_only.sum():,}")
    
    if c_only.sum() > 0:
        print("\nRecords with Invoice 'C' BUT Quantity >= 0:")
        c_only_df = df[c_only][["invoice_no", "stock_code", "description", "quantity", "unit_price", "customer_id", "sheet_name"]]
        print(c_only_df.to_string())
        
    if neg_only.sum() > 0:
        print(f"\nBreakdown of {neg_only.sum():,} records with Quantity < 0 BUT Invoice NOT 'C':")
        neg_only_df = df[neg_only]
        print(f"Missing customer IDs among these:     {neg_only_df['customer_id'].isna().sum():,}")
        print(f"Zero price among these:               {(neg_only_df['unit_price'] == 0).sum():,}")
        print("\nSample records:")
        print(neg_only_df[["invoice_no", "stock_code", "description", "quantity", "unit_price", "customer_id", "sheet_name"]].head(10).to_string())
        
    print("\n" + "=" * 70)
    print("2. AUDIT: DUPLICATE RECORDS CALCULATION")
    print("=" * 70)
    
    business_cols = [
        "invoice_no",
        "stock_code",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
    ]
    all_raw_cols = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]
    
    dup_keep_first_biz = df.duplicated(subset=business_cols, keep="first").sum()
    dup_keep_last_biz = df.duplicated(subset=business_cols, keep="last").sum()
    dup_keep_false_biz = df.duplicated(subset=business_cols, keep=False).sum()
    
    dup_keep_first_raw = df.duplicated(subset=all_raw_cols, keep="first").sum()
    dup_keep_false_raw = df.duplicated(subset=all_raw_cols, keep=False).sum()
    
    print(f"Subset business_cols: {business_cols}")
    print(f"  - duplicated(keep='first') [Default count]:  {dup_keep_first_biz:,}")
    print(f"  - duplicated(keep='last'):                   {dup_keep_last_biz:,}")
    print(f"  - duplicated(keep=False) [All occurrences]:  {dup_keep_false_biz:,}")
    print(f"  - Unique duplicate groups:                   {dup_keep_false_biz - dup_keep_first_biz:,}")
    
    print(f"\nAll raw columns (including description & country):")
    print(f"  - duplicated(keep='first'):                  {dup_keep_first_raw:,}")
    print(f"  - duplicated(keep=False):                  {dup_keep_false_raw:,}")
    
    # Check if duplicates occur within the same sheet or across sheets
    s1 = pd.read_parquet("data/bronze/online_retail_year_2009_2010.parquet")
    s2 = pd.read_parquet("data/bronze/online_retail_year_2010_2011.parquet")
    dup_s1 = s1.duplicated(subset=business_cols, keep="first").sum()
    dup_s2 = s2.duplicated(subset=business_cols, keep="first").sum()
    print(f"\nDuplicate count per sheet (keep='first'):")
    print(f"  - Sheet 1 (Year 2009-2010):                  {dup_s1:,}")
    print(f"  - Sheet 2 (Year 2010-2011):                  {dup_s2:,}")
    print(f"  - Sum of per-sheet duplicates:               {dup_s1 + dup_s2:,}")
    print(f"  - Cross-sheet duplicates:                    {dup_keep_first_biz - (dup_s1 + dup_s2):,}")

if __name__ == "__main__":
    run_audit()
