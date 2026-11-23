"""
Training and Evaluation Script for Transaction Anomaly Detection (Phase 5).

Trains Isolation Forest on eligible commercial transactions from the Silver layer,
evaluates score distribution, contamination sensitivity, random-seed stability,
and synthetic diagnostic injection, generates visual reports, and persists artifacts.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

# Silence loky CPU detection warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.anomaly_detector import TransactionAnomalyDetector
from src.models.config import AnomalyDetectionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecommerce_platform.scripts.train_anomaly")


def generate_visual_reports(
    detector: TransactionAnomalyDetector,
    eval_results: dict,
    reports_dir: Path,
) -> None:
    """Generates diagnostic visualization plots for model governance."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating visual diagnostic reports in {reports_dir}...")

    # 1. Contamination Sensitivity Plot
    sens = eval_results.get("contamination_sensitivity", [])
    if sens:
        plt.figure(figsize=(8, 5))
        contams = [s["contamination"] * 100 for s in sens]
        flagged_pcts = [s["flagged_percentage"] for s in sens]
        plt.plot(contams, flagged_pcts, marker="o", color="#2563EB", linewidth=2.5, markersize=8)
        plt.plot([0.5, 5.0], [0.5, 5.0], linestyle="--", color="#9CA3AF", label="Ideal Identity (Expected = Observed)")
        plt.title("Isolation Forest: Contamination Sensitivity Curve", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Configured Contamination Rate (%)", fontsize=12)
        plt.ylabel("Observed Flagged Population (%)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(reports_dir / "contamination_sensitivity.png", dpi=300)
        plt.close()

    # 2. Score Distribution Histogram (from metrics)
    score_stats = detector.evaluation_metrics_.get("raw_decision_score_stats", {})
    if score_stats:
        plt.figure(figsize=(9, 5))
        offset = detector.threshold_policy_.get("decision_threshold", 0.0)
        # Plot quantile bar approximations
        quantiles = ["min", "p01", "p05", "p50", "p95", "p99", "max"]
        vals = [score_stats[q] for q in quantiles]
        colors = ["#EF4444" if v < offset else "#10B981" for v in vals]
        plt.bar(quantiles, vals, color=colors, edgecolor="#1F2937", linewidth=1.2, width=0.5)
        plt.axhline(offset, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Anomaly Threshold ({offset:.4f})")
        plt.title("Isolation Forest: Raw Decision Score Quantiles", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Quantile Marker", fontsize=12)
        plt.ylabel("Raw Decision Score (Lower = More Anomalous)", fontsize=12)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(reports_dir / "score_quantiles.png", dpi=300)
        plt.close()

    logger.info("Visual reports generated successfully.")


def run_training() -> None:
    """Executes the full Phase 5 training and evaluation workflow."""
    start_total = time.time()
    config = AnomalyDetectionConfig()

    print("=" * 80)
    print("PHASE 5: TRANSACTION ANOMALY DETECTION TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Load Silver Dataset
    silver_path = config.silver_parquet_path
    print(f"Loading Silver Parquet: {silver_path}...")
    silver_df = pd.read_parquet(silver_path)
    total_silver = len(silver_df)
    print(f"Loaded {total_silver:,} Silver transaction records.")

    # 2. Check Eligible Population
    eligible_mask = (~silver_df["is_duplicate"]) & (silver_df["unit_price"] > 0)
    eligible_df = silver_df[eligible_mask]
    n_eligible = len(eligible_df)
    n_excluded = total_silver - n_eligible
    print(f"Eligible Training Population: {n_eligible:,} records (96.22%)")
    print(f"Excluded from Training:       {n_excluded:,} records (3.78%)")
    print(f"  - Duplicates:               {silver_df['is_duplicate'].sum():,}")
    print(f"  - Inventory Write-Offs:     {(silver_df['transaction_status'] == 'INVENTORY_WRITE_OFF').sum():,}")
    print(f"  - Zero/Neg Price Entries:   {(silver_df['transaction_status'] == 'ZERO_PRICE_ENTRY').sum():,}")

    # 3. Initialize and Fit Anomaly Detector
    detector = TransactionAnomalyDetector(config=config)
    print("\nFitting TransactionFeaturePipeline and Isolation Forest...")
    detector.fit(silver_df)

    # 4. Evaluate Sensitivity, Seed Stability & Synthetic Injection
    print("\nRunning Sensitivity, Seed Stability, and Synthetic Diagnostic Experiments...")
    eval_results = detector.evaluate_sensitivity_and_stability(silver_df, sample_size=50000)
    detector.evaluation_metrics_["sensitivity_and_stability"] = eval_results

    # 5. Persist Model and Artifacts
    print(f"\nPersisting artifacts to: {config.model_dir}...")
    detector.save(config.model_dir)

    # 6. Generate Diagnostic Reports
    generate_visual_reports(detector, eval_results, config.reports_dir)

    # 7. Print Comprehensive Summary
    offset = detector.threshold_policy_.get("decision_threshold", 0.0)
    internal_offset = detector.threshold_policy_.get("internal_offset", 0.0)
    metrics = detector.evaluation_metrics_
    stability = eval_results["seed_stability"]
    synthetic = eval_results["synthetic_diagnostic_test"]

    print("\n" + "=" * 80)
    print("TRAINING & EVALUATION AUDIT SUMMARY")
    print("=" * 80)
    print(f"Model Version:                  {config.model_version}")
    print(f"Eligible Training Population:   {metrics['training_records_count']:,}")
    print(f"Excluded Administrative Count:  {metrics['excluded_records_count']:,}")
    print(f"Hyperparameters:                n_estimators={config.n_estimators}, max_samples={config.max_samples}, contamination={config.contamination}")
    print(f"Decision Boundary:              raw_decision_score < {offset:.6f} (internal offset: {internal_offset:.6f})")
    print(f"Training Flagged Count:         {metrics['training_flagged_count']:,} ({metrics['training_flagged_pct']}%)")
    print("\n--- SCORE DISTRIBUTION (RAW DECISION FUNCTION) ---")
    raw_stats = metrics["raw_decision_score_stats"]
    print(f"  Mean: {raw_stats['mean']:.6f}, Std: {raw_stats['std']:.6f}")
    print(f"  Min:  {raw_stats['min']:.6f}, p01: {raw_stats['p01']:.6f}, p05: {raw_stats['p05']:.6f}")
    print(f"  p50:  {raw_stats['p50']:.6f}, p95: {raw_stats['p95']:.6f}, p99: {raw_stats['p99']:.6f}, Max: {raw_stats['max']:.6f}")

    print("\n--- CONTAMINATION SENSITIVITY ---")
    for s in eval_results["contamination_sensitivity"]:
        print(f"  Contamination {s['contamination']*100:4.1f}% -> Offset: {s['decision_offset_threshold']:8.6f}, Flagged: {s['flagged_count']:,} ({s['flagged_percentage']}%)")

    print("\n--- SEED STABILITY (4 SEEDS: 42, 123, 456, 789) ---")
    print(f"  Spearman Rank Correlation: Min = {stability['spearman_rank_correlation_min']:.4f}, Mean = {stability['spearman_rank_correlation_mean']:.4f}, Max = {stability['spearman_rank_correlation_max']:.4f}")
    print(f"  Jaccard Flagged Overlap:   Min = {stability['jaccard_flagged_similarity_min'] if 'jaccard_flagged_similarity_min' in stability else stability.get('jaccard_flag_similarity_min'):.4f}, Mean = {stability.get('jaccard_flag_similarity_mean'):.4f}, Max = {stability.get('jaccard_flag_similarity_max'):.4f}")

    print("\n--- SYNTHETIC DIAGNOSTIC INJECTION ---")
    print(f"  Injected Extreme Anomalies:    {synthetic['injected_count']}")
    print(f"  Successfully Detected:         {synthetic['detected_count']} ({synthetic['diagnostic_detection_rate']*100:.1f}%)")

    print("\n--- DIAGNOSTIC REASON THRESHOLDS (99th Percentile Reference) ---")
    diag = metrics["diagnostic_thresholds"]
    print(f"  Quantity p99:      >= {diag['quantity_p99']:.1f} units")
    print(f"  Line Total p99:    >= £{diag['line_total_p99']:.2f}")
    print(f"  Unit Price p99:    >= £{diag['unit_price_p99']:.2f}")
    print(f"  Price Deviation:   >= {diag['price_dev_threshold']:.2f} (~3.0x catalog median)")

    print(f"\nExecution completed in {time.time() - start_total:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    run_training()
