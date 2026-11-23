import pandas as pd
from pathlib import Path

df = pd.read_parquet("data/silver/online_retail_silver.parquet")
print(f"Total Silver records: {len(df):,}")
print("Min invoice_date:", df["invoice_date"].min())
print("Max invoice_date:", df["invoice_date"].max())

cust_df = df.dropna(subset=["customer_id"])
n_cust_records = len(cust_df)
n_unique_cust = cust_df["customer_id"].nunique()
print(f"Records with customer_id: {n_cust_records:,} (Unique customers: {n_unique_cust:,})")

valid_purchases = cust_df[cust_df["transaction_status"] == "VALID_PURCHASE"]
print(f"Valid purchases with customer_id: {len(valid_purchases):,}")
print("Min valid purchase date:", valid_purchases["invoice_date"].min())
print("Max valid purchase date:", valid_purchases["invoice_date"].max())

# Unique invoices per month
cust_df_copy = cust_df.copy()
cust_df_copy["year_month"] = cust_df_copy["invoice_date"].dt.to_period("M")
monthly_purchases = valid_purchases.groupby(valid_purchases["invoice_date"].dt.to_period("M"))["invoice_no"].nunique()
monthly_active_cust = valid_purchases.groupby(valid_purchases["invoice_date"].dt.to_period("M"))["customer_id"].nunique()

summary = pd.DataFrame({
    "purchase_invoices": monthly_purchases,
    "active_customers": monthly_active_cust,
})
print("\nMonthly Activity Summary:")
print(summary)
