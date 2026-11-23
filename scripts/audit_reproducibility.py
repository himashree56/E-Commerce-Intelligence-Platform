import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from src.models.anomaly_detector import TransactionAnomalyDetector

def main():
    gold_path = Path("data/gold/gold_transaction_anomalies_isolation_forest.parquet")
    silver_path = Path("data/silver/online_retail_silver.parquet")
    model_dir = Path("models/anomaly_detection")

    print(f"Loading gold parquet from {gold_path}...")
    df_gold = pd.read_parquet(gold_path)

    print(f"Loading detector from {model_dir}...")
    detector = TransactionAnomalyDetector.load(model_dir)

    # Test sample of 50,000 rows
    print("Testing reproducibility on 50,000 Silver records...")
    df_silver_sample = pd.read_parquet(silver_path).head(50000)
    re_scored = detector.predict(df_silver_sample)
    orig_sample = df_gold.head(50000)

    score_diff = np.max(np.abs(re_scored["anomaly_score"].values - orig_sample["anomaly_score"].values))
    flags_match = (re_scored["is_anomaly"].values == orig_sample["is_anomaly"].values).all()
    reasons_match = (re_scored["diagnostic_reasons"].values == orig_sample["diagnostic_reasons"].values).all()

    print(f"Max score difference on re-scoring: {score_diff:.10f}")
    print(f"Flags match perfectly (50,000 rows): {flags_match}")
    print(f"Diagnostic reasons match perfectly: {reasons_match}")

    with open(model_dir / "evaluation_metrics.json") as f:
        metrics = json.load(f)

    seed_stats = metrics["sensitivity_and_stability"]["seed_stability"]
    print("\n--- Persisted Multi-Seed Stability Statistics ---")
    print(f"Tested seeds: {seed_stats['tested_seeds']}")
    print(f"Spearman Rank Correlation (rho): min={seed_stats['spearman_rank_correlation_min']}, mean={seed_stats['spearman_rank_correlation_mean']}, max={seed_stats['spearman_rank_correlation_max']}")
    print(f"Jaccard Top-1% Alert Similarity: min={seed_stats['jaccard_flag_similarity_min']}, mean={seed_stats['jaccard_flag_similarity_mean']}, max={seed_stats['jaccard_flag_similarity_max']}")

if __name__ == "__main__":
    main()
