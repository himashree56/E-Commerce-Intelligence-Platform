"""CLI Script for Training, Profiling, and Persisting K-Means Customer Segmentation (Phase 4)."""
import os
import sys
from pathlib import Path

# Silence loky CPU detection warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models.config import SegmentationConfig
from src.models.customer_segmentation import CustomerSegmentationModel
from src.models.evaluation import SegmentationVisualizer


def main() -> int:
    print("=" * 85)
    print("    E-COMMERCE INTELLIGENCE PLATFORM - K-MEANS CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 85)

    config = SegmentationConfig()
    print(f"\n[CONFIG] Model Version:      {config.model_version}")
    print(f"[CONFIG] Random State:       {config.random_state}")
    print(f"[CONFIG] Initializations:    {config.n_init}")
    print(f"[CONFIG] Candidate Features: {config.feature_columns}")
    print(f"[CONFIG] Scaling:            {config.scaler_type.upper()}")
    print(f"[CONFIG] Target k Range:     {config.k_min} to {config.k_max} (Optimal k={config.optimal_k})")

    # 1. Load actual Gold Customer RFM dataset
    print(f"\n[DATA] Loading Gold RFM dataset from: {config.gold_rfm_path}")
    if not config.gold_rfm_path.exists():
        print(f"[ERROR] RFM dataset not found at {config.gold_rfm_path}", file=sys.stderr)
        return 1

    df_rfm = pd.read_parquet(config.gold_rfm_path)
    print(f"[DATA] Successfully loaded {len(df_rfm):,} customer records.")

    # 2. Train Customer Segmentation Model
    model = CustomerSegmentationModel(config=config)
    print("\n[TRAINING] Evaluating k in [2, 10] and fitting optimal K-Means model...")
    model.fit(df_rfm)

    # 3. Save Model Artifacts
    print(f"\n[PERSISTENCE] Saving model artifacts to: {config.model_dir}")
    saved_paths = model.save()
    for artifact_name, artifact_path in saved_paths.items():
        print(f"  - {artifact_name:20}: {artifact_path.name} ({artifact_path.stat().st_size:,} bytes)")

    # 4. Generate Visualizations
    print(f"\n[VISUALIZATION] Generating evaluation charts in: {config.reports_dir}")
    visualizer = SegmentationVisualizer(output_dir=config.reports_dir)

    # A. Evaluation curves
    eval_chart = visualizer.plot_k_evaluation_curves(
        model.evaluation_metrics_["tested_k_metrics"], optimal_k=config.optimal_k
    )
    print(f"  - Evaluation Curves:        {eval_chart.name}")

    # B. Cluster sizes
    sizes_chart = visualizer.plot_cluster_sizes(model.cluster_profiles_)
    print(f"  - Cluster Size Chart:       {sizes_chart.name}")

    # C. Score full dataset to generate profiled DataFrame
    predictions = model.predict(df_rfm)
    df_profiled = df_rfm.copy()
    df_profiled["cluster_id"] = predictions["cluster_id"].values
    df_profiled["cluster_label"] = predictions["cluster_label"].values
    df_profiled["distance_to_centroid"] = predictions["distance_to_centroid"].values

    # Save predictions to Gold Parquet
    out_parquet = config.base_dir / "data" / "gold" / "gold_customer_segmentation_kmeans.parquet"
    predictions.to_parquet(out_parquet, index=False)
    print(f"  - Scored Dataset Parquet:   {out_parquet.name} ({len(predictions):,} rows)")

    # D. RFM distributions
    dist_chart = visualizer.plot_rfm_distributions(df_profiled)
    print(f"  - Feature Distributions:    {dist_chart.name}")

    # E. PCA 2D projection
    X_scaled = model.pipeline.transform(df_rfm)
    pca_chart = visualizer.plot_pca_2d_projection(
        X_scaled=X_scaled,
        labels=predictions["cluster_id"].values,
        centroids=model.ordered_centroids_,
        cluster_labels_map=config.cluster_labels,
    )
    print(f"  - PCA 2D Projection:        {pca_chart.name}")

    # F. Cluster profile heatmap
    heatmap_chart = visualizer.plot_cluster_profile_heatmap(model.cluster_profiles_)
    print(f"  - Profile Heatmap:          {heatmap_chart.name}")

    # G. Silhouette samples
    sil_chart = visualizer.plot_silhouette_samples(
        X_scaled=X_scaled,
        labels=predictions["cluster_id"].values,
        optimal_k=config.optimal_k,
    )
    print(f"  - Silhouette Samples Plot:  {sil_chart.name}")

    # 5. Display Final Summary
    meta = model.training_metadata_
    print("\n" + "=" * 85)
    print("                    K-MEANS CLUSTERING SUMMARY (k=5)")
    print("=" * 85)
    print(f"Inertia:                 {meta['inertia']:,.2f}")
    print(f"Silhouette Score:        {meta['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Index:    {meta['davies_bouldin_index']:.4f}")
    print(f"Calinski-Harabasz Index: {meta['calinski_harabasz_index']:,.2f}")
    print(f"Total Customers Scored:  {meta['total_training_records']:,}")

    print("\n--- Cluster Profile Summary ---")
    for c_id_str, p in model.cluster_profiles_.items():
        print(
            f"Cluster {p['cluster_id']}: {p['cluster_label']:<34} | "
            f"Count={p['customer_count']:5,} ({p['population_pct']:5.2f}%) | "
            f"Rev=£{p['total_revenue_gbp']:12,.2f} ({p['revenue_pct']:5.2f}%) | "
            f"Med Rec={p['median_recency_days']:3.0f}d | "
            f"Med Freq={p['median_frequency']:2d} | "
            f"Med Spend=£{p['median_monetary_net_gbp']:8,.2f}"
        )

    print("=" * 85)
    print("[SUCCESS] Customer segmentation model trained, profiled, and persisted successfully!")
    print("=" * 85)
    return 0


if __name__ == "__main__":
    sys.exit(main())
