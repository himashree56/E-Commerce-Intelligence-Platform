"""Investigation of unusual transactions across the 1,067,371 Bronze records."""
from pathlib import Path
import pandas as pd

def investigate():
    bronze_path = Path("data/bronze/online_retail_bronze.parquet")
    df = pd.read_parquet(bronze_path)

    print("=" * 75)
    print("      DETAILED INVESTIGATION OF UNUSUAL DATASET TRANSACTIONS")
    print("=" * 75)

    # 1. Non-product / Administrative Stock Codes
    non_numeric_codes = df[~df["stock_code"].str.match(r"^\d{4,5}[A-Za-z]?$", na=False)]["stock_code"].value_counts()
    print(f"\n1. Non-product or non-standard stock codes ({len(non_numeric_codes)} distinct):")
    print(non_numeric_codes.head(25))

    # 2. Negative unit prices
    neg_prices = df[df["unit_price"] < 0]
    print(f"\n2. Negative Unit Prices: {len(neg_prices)} records")
    if len(neg_prices) > 0:
        print(neg_prices[["invoice_no", "stock_code", "description", "quantity", "unit_price", "customer_id", "sheet_name"]].to_string())

    # 3. Zero unit prices
    zero_prices = df[df["unit_price"] == 0]
    print(f"\n3. Zero Unit Prices: {len(zero_prices):,} records")
    pos_qty_zero_price = zero_prices[zero_prices["quantity"] > 0]
    neg_qty_zero_price = zero_prices[zero_prices["quantity"] < 0]
    zero_qty_zero_price = zero_prices[zero_prices["quantity"] == 0]
    print(f"  - Zero Price with Quantity > 0: {len(pos_qty_zero_price):,}")
    print(f"  - Zero Price with Quantity < 0: {len(neg_qty_zero_price):,}")
    print(f"  - Zero Price with Quantity == 0: {len(zero_qty_zero_price):,}")

    print("\n  Top descriptions for Zero Price & Quantity > 0 (Giveaways/Samples/Found Stock):")
    print(pos_qty_zero_price["description"].value_counts().head(12))

    print("\n  Top descriptions for Zero Price & Quantity < 0 (Damaged/Lost/Scrapped/Internal):")
    print(neg_qty_zero_price["description"].value_counts().head(12))

    # 4. Extreme Quantities
    print("\n4. Extreme Outlier Quantities:")
    print("  Top 5 Largest Positive Quantities:")
    print(df.nlargest(5, "quantity")[["invoice_no", "stock_code", "description", "quantity", "unit_price", "customer_id", "invoice_date"]].to_string())
    print("\n  Top 5 Most Negative Quantities:")
    print(df.nsmallest(5, "quantity")[["invoice_no", "stock_code", "description", "quantity", "unit_price", "customer_id", "invoice_date"]].to_string())

    # 5. Cancellations with positive quantity
    c_pos = df[df["invoice_no"].str.startswith(("C", "c"), na=False) & (df["quantity"] > 0)]
    print(f"\n5. Invoices starting with 'C' with Quantity > 0: {len(c_pos)} records")
    if len(c_pos) > 0:
        print(c_pos[["invoice_no", "stock_code", "description", "quantity", "unit_price", "customer_id"]].to_string())

    # 6. Customer ID missing analysis
    missing_cust = df[df["customer_id"].isna()]
    print(f"\n6. Records Missing Customer ID: {len(missing_cust):,} ({len(missing_cust)/len(df)*100:.2f}%)")
    print(f"  - Cancellations among missing customer: {missing_cust['invoice_no'].str.startswith(('C', 'c'), na=False).sum():,}")
    print(f"  - Zero price among missing customer:    {(missing_cust['unit_price'] == 0).sum():,}")
    print(f"  - Positive sales with missing customer: {((missing_cust['quantity'] > 0) & (missing_cust['unit_price'] > 0)).sum():,}")

    print("=" * 75)

if __name__ == "__main__":
    investigate()
