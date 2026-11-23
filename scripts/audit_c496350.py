import pandas as pd
from pathlib import Path

# Check raw Excel if available
raw_excel_paths = list(Path("data/raw").glob("*.xlsx")) + list(Path("data/raw").glob("*.xls"))
print(f"Raw Excel files: {raw_excel_paths}")

for rep in raw_excel_paths:
    print(f"\n--- Checking raw workbook: {rep} ---")
    xls = pd.ExcelFile(rep)
    print(f"Sheet names: {xls.sheet_names}")
    for sheet_name in xls.sheet_names:
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)
        # Search all columns for 496350
        mask = df_raw.astype(str).apply(lambda row: row.str.contains("496350", case=False).any(), axis=1)
        matches = df_raw[mask]
        if len(matches) > 0:
            print(f"Found {len(matches)} match(es) in raw sheet '{sheet_name}':")
            for idx, r in matches.iterrows():
                print(f"Row {idx}: {dict(r)}")

# Check Bronze
bronze_files = list(Path("data/bronze").glob("*.parquet"))
for bf in bronze_files:
    df_b = pd.read_parquet(bf)
    mask = df_b.astype(str).apply(lambda row: row.str.contains("496350", case=False).any(), axis=1)
    matches = df_b[mask]
    if len(matches) > 0:
        print(f"\nFound {len(matches)} match(es) in Bronze '{bf.name}':")
        for idx, r in matches.iterrows():
            print(f"Row {idx}: {dict(r)}")

# Check Silver
silver_path = Path("data/silver/online_retail_silver.parquet")
if silver_path.exists():
    df_s = pd.read_parquet(silver_path)
    mask = df_s.astype(str).apply(lambda row: row.str.contains("496350", case=False).any(), axis=1)
    matches = df_s[mask]
    if len(matches) > 0:
        print(f"\nFound {len(matches)} match(es) in Silver:")
        for idx, r in matches.iterrows():
            print(f"Index {idx}:")
            for col in df_s.columns:
                print(f"  {col}: {r[col]}")

# Check Gold Anomalies
gold_path = Path("data/gold/gold_transaction_anomalies_isolation_forest.parquet")
if gold_path.exists():
    df_g = pd.read_parquet(gold_path)
    mask = df_g.astype(str).apply(lambda row: row.str.contains("496350", case=False).any(), axis=1)
    matches = df_g[mask]
    if len(matches) > 0:
        print(f"\nFound {len(matches)} match(es) in Gold Anomalies:")
        for idx, r in matches.iterrows():
            print(f"Index {idx}:")
            for col in df_g.columns:
                print(f"  {col}: {r[col]}")
