"""
CLI Script for Training, Tuning, and Evaluating XGBoost Repeat-Purchase Prediction (Phase 6).

Trains:
- Dummy baseline
- Logistic Regression baseline
- Tuned XGBoost Classifier
Evaluates strictly on Validation (for tuning & threshold selection) and Held-out Test (for final reporting).
Generates SHAP and feature importance visualizations and persists all artifacts.
"""
import json
import os
import sys
import time
from pathlib import Path

# Silence loky CPU count warning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.config import RepeatPurchaseConfig
from src.models.repeat_purchase import RepeatPurchasePredictor


def main() -> int:
    print("=" * 85)
    print("    E-COMMERCE INTELLIGENCE PLATFORM - PHASE 6: XGBOOST REPEAT-PURCHASE TRAINING")
    print("=" * 85)

    config = RepeatPurchaseConfig()
    t_start = time.time()

    # 1. Load Gold Datasets
    print(f"Loading Gold Datasets from: {config.gold_train_path.parent}...")
    train_df = pd.read_parquet(config.gold_train_path)
    val_df = pd.read_parquet(config.gold_val_path)
    test_df = pd.read_parquet(config.gold_test_path)

    print(f"Train Dataset:      {len(train_df):,} rows | Positives: {train_df['label_30d'].sum():,} ({train_df['label_30d'].mean()*100:.2f}%)")
    print(f"Validation Dataset: {len(val_df):,} rows | Positives: {val_df['label_30d'].sum():,} ({val_df['label_30d'].mean()*100:.2f}%)")
    print(f"Test Dataset:       {len(test_df):,} rows | Positives: {test_df['label_30d'].sum():,} ({test_df['label_30d'].mean()*100:.2f}%)")

    # 2. Fit and Tune Predictor
    predictor = RepeatPurchasePredictor(config)
    print("\n--- Fitting Preprocessing, Baselines, and Tuning XGBoost ---")
    predictor.fit_and_tune(train_df, val_df)

    # 3. Evaluate Held-out Test Set
    print("\n--- Evaluating on Held-out Test Set ---")
    test_results = predictor.evaluate_test_set(test_df)

    # 4. Save Artifacts
    print(f"\n--- Persisting Artifacts to: {config.model_dir} ---")
    saved_dir = predictor.save()
    print(f"Saved artifacts to {saved_dir}.")

    # 5. Generate Visualizations (Feature Importance & SHAP)
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating Visualizations in: {reports_dir} ---")

    # Feature Importance Plot
    importances = predictor.compute_feature_importances()
    gain_scores = importances["gain"]
    top_feats = list(gain_scores.keys())[:12]
    top_gains = [gain_scores[f] for f in top_feats]

    plt.figure(figsize=(10, 6))
    plt.barh(top_feats[::-1], top_gains[::-1], color="#1f77b4", edgecolor="black")
    plt.xlabel("XGBoost Gain Feature Importance")
    plt.title("Top Feature Importance (Gain) - 30-Day Repeat Purchase Prediction")
    plt.tight_layout()
    fi_path = reports_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance plot to {fi_path}")

    # SHAP Summary Plot
    try:
        print("Computing SHAP values for 1,000 test customer snapshots...")
        shap_vals, explainer = predictor.compute_shap_explanations(test_df, max_samples=1000)
        X_sub = test_df[predictor.feature_names].iloc[:1000]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_sub, show=False)
        shap_path = reports_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(shap_path, dpi=300)
        plt.close()
        print(f"Saved SHAP Summary plot to {shap_path}")
    except Exception as e:
        print(f"[WARN] Could not generate SHAP plot: {e}")

    # 6. Display Metric Comparison Tables
    val_metrics = predictor.evaluation_metrics["validation"]
    test_metrics = predictor.evaluation_metrics["test"]

    print("\n" + "=" * 85)
    print("MODEL PERFORMANCE COMPARISON (HELD-OUT TEST PERIOD: 2011-10-01 to 2011-10-31)")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Dummy Baseline':<16} | {'Logistic Regression':<20} | {'XGBoost (Final)':<16}")
    print("-" * 85)

    metrics_to_show = [
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC (Avg Precision)", "pr_auc"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
        ("Brier Score (Calibration)", "brier_score"),
        ("Precision @ Top 5%", "precision_at_top_5_pct"),
        ("Precision @ Top 10%", "precision_at_top_10_pct"),
    ]

    for label, key in metrics_to_show:
        d_val = test_metrics["dummy_baseline"][key]
        lr_val = test_metrics["logistic_regression"][key]
        xgb_val = test_metrics["xgboost"][key]
        print(f"{label:<25} | {d_val:<16.4f} | {lr_val:<20.4f} | {xgb_val:<16.4f}")

    print("-" * 85)
    print(f"Optimal Decision Threshold (Validation F1): {predictor.optimal_threshold:.4f}")
    cm = test_metrics["xgboost"]["confusion_matrix"]
    print(f"Test Confusion Matrix: TN={cm['tn']:,}, FP={cm['fp']:,}, FN={cm['fn']:,}, TP={cm['tp']:,}")

    # Subgroups
    print("\n" + "=" * 85)
    print("XGBOOST PERFORMANCE BY CUSTOMER ACTIVITY SUBGROUP (TEST PERIOD)")
    print("=" * 85)
    subgroups = test_metrics["activity_subgroups_xgboost"]
    print(f"{'Subgroup':<30} | {'Count':<7} | {'Positives':<9} | {'Prevalence':<10} | {'ROC-AUC':<8} | {'PR-AUC':<8}")
    print("-" * 85)
    for grp_name, grp_data in subgroups.items():
        print(
            f"{grp_name:<30} | {grp_data['count']:<7} | {grp_data['positives']:<9} | "
            f"{grp_data['prevalence_pct']:<9.1f}% | {grp_data['roc_auc']:<8.4f} | {grp_data['pr_auc']:<8.4f}"
        )

    print("=" * 85)
    print(f"Total Phase 6 Model Training Pipeline completed in {time.time()-t_start:.2f} seconds.")
    print("=" * 85)
    return 0


if __name__ == "__main__":
    sys.exit(main())
