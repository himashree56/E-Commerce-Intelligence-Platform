import pandas as pd
import numpy as np

df = pd.read_parquet("data/silver/online_retail_silver.parquet")
valid = df[(df["transaction_status"] == "VALID_PURCHASE") & (df["customer_id"].notna())].copy()
valid["invoice_date"] = pd.to_datetime(valid["invoice_date"])

print(f"Total valid purchase lines: {len(valid):,}")
print(f"Overall unique customers: {valid['customer_id'].nunique():,}")

candidate_snapshots = [
    # Year 1 snapshots (with 6+ months lookback)
    "2010-06-01",
    "2010-08-01",
    "2010-10-01",
    "2010-12-01",
    # Year 2 snapshots
    "2011-02-01",
    "2011-04-01",
    "2011-06-01",
    "2011-08-01",
    "2011-10-01",
    "2011-11-01",
]

results = []
for s_str in candidate_snapshots:
    t_obs = pd.Timestamp(s_str)
    t_fut = t_obs + pd.Timedelta(days=30)

    # Prior history: strictly before t_obs
    past = valid[valid["invoice_date"] < t_obs]
    eligible_cust = set(past["customer_id"].unique())

    # Future window: strictly after t_obs and <= t_fut
    fut = valid[(valid["invoice_date"] > t_obs) & (valid["invoice_date"] <= t_fut)]
    purchased_cust = set(fut["customer_id"].unique())

    repeat_buyers = eligible_cust.intersection(purchased_cust)
    n_eligible = len(eligible_cust)
    n_pos = len(repeat_buyers)
    pct = (n_pos / n_eligible * 100) if n_eligible > 0 else 0.0

    results.append({
        "snapshot_date": s_str,
        "future_window_end": t_fut.strftime("%Y-%m-%d"),
        "eligible_customers": n_eligible,
        "positive_repeat_buyers": n_pos,
        "negative_non_repeat": n_eligible - n_pos,
        "positive_rate_pct": round(pct, 2),
    })

res_df = pd.DataFrame(results)
print("\nCandidate Snapshots Overview (30-day window):")
print(res_df.to_string(index=False))
