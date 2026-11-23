import pandas as pd

train_dates = ["2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01"]
val_date = "2011-08-01"
test_date = "2011-10-01"

df = pd.read_parquet("data/silver/online_retail_silver.parquet")
valid = df[(df["transaction_status"] == "VALID_PURCHASE") & (df["customer_id"].notna())].copy()
valid["invoice_date"] = pd.to_datetime(valid["invoice_date"])

def get_snapshot(t_str):
    t_obs = pd.Timestamp(t_str)
    t_fut = t_obs + pd.Timedelta(days=30)
    past = valid[valid["invoice_date"] < t_obs]
    eligible = sorted(list(set(past["customer_id"].unique())))
    fut = valid[(valid["invoice_date"] > t_obs) & (valid["invoice_date"] <= t_fut)]
    pos = set(fut["customer_id"].unique()).intersection(eligible)
    return pd.DataFrame({
        "customer_id": eligible,
        "snapshot_date": t_str,
        "label_30d": [1 if c in pos else 0 for c in eligible]
    })

train_dfs = [get_snapshot(d) for d in train_dates]
df_train = pd.concat(train_dfs, ignore_index=True)
df_val = get_snapshot(val_date)
df_test = get_snapshot(test_date)

n_train = len(df_train)
n_train_unique = df_train["customer_id"].nunique()
train_pos = int(df_train["label_30d"].sum())
train_neg = n_train - train_pos
train_pct = train_pos / n_train * 100

n_val = len(df_val)
n_val_unique = df_val["customer_id"].nunique()
val_pos = int(df_val["label_30d"].sum())
val_neg = n_val - val_pos
val_pct = val_pos / n_val * 100

n_test = len(df_test)
n_test_unique = df_test["customer_id"].nunique()
test_pos = int(df_test["label_30d"].sum())
test_neg = n_test - test_pos
test_pct = test_pos / n_test * 100

print("=== Temporal Split Summary ===")
print(f"Train Snapshots: {train_dates}")
print(f"  Total customer-snapshots: {n_train:,} | Unique customers: {n_train_unique:,}")
print(f"  Positive (Repeat): {train_pos:,} | Negative: {train_neg:,} | Prevalence: {train_pct:.2f}%")

print(f"\nValidation Snapshot: {val_date} (Future window: {val_date} to 2011-08-31)")
print(f"  Total customer-snapshots: {n_val:,} | Unique customers: {n_val_unique:,}")
print(f"  Positive (Repeat): {val_pos:,} | Negative: {val_neg:,} | Prevalence: {val_pct:.2f}%")

print(f"\nTest Snapshot: {test_date} (Future window: {test_date} to 2011-10-31)")
print(f"  Total customer-snapshots: {n_test:,} | Unique customers: {n_test_unique:,}")
print(f"  Positive (Repeat): {test_pos:,} | Negative: {test_neg:,} | Prevalence: {test_pct:.2f}%")

# Overlap of unique customers across splits
train_custs = set(df_train["customer_id"].unique())
val_custs = set(df_val["customer_id"].unique())
test_custs = set(df_test["customer_id"].unique())

print("\n=== Customer Identity Overlap Across Splits ===")
print(f"Unique customers in Train: {len(train_custs):,}")
print(f"Unique customers in Val:   {len(val_custs):,} (In Train: {len(val_custs.intersection(train_custs)):,}, New in Val: {len(val_custs - train_custs):,})")
print(f"Unique customers in Test:  {len(test_custs):,} (In Train/Val: {len(test_custs.intersection(train_custs.union(val_custs))):,}, New in Test: {len(test_custs - train_custs - val_custs):,})")

# Repeated customer appearances across snapshots within Train
train_appearances = df_train["customer_id"].value_counts()
print(f"\nWithin Train, customer snapshot appearances:")
print(train_appearances.value_counts().sort_index())
