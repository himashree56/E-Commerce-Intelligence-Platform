"""Integration and unit tests for K-Means Customer Segmentation Model (Phase 4)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.config import SegmentationConfig
from src.models.customer_segmentation import CustomerSegmentationModel


@pytest.fixture(scope="module")
def rfm_dataset():
    """Loads actual 5,939-customer Gold RFM dataset."""
    config = SegmentationConfig()
    if not config.gold_rfm_path.exists():
        pytest.skip("Gold RFM parquet not found.")
    return pd.read_parquet(config.gold_rfm_path)


@pytest.fixture(scope="module")
def trained_model(rfm_dataset):
    """Provides a pre-trained CustomerSegmentationModel on the actual dataset."""
    config = SegmentationConfig(optimal_k=5, random_state=42, n_init=10)
    model = CustomerSegmentationModel(config=config)
    model.fit(rfm_dataset)
    return model


def test_segmentation_training_reproducibility(rfm_dataset):
    """Verifies that model training with fixed random_state produces identical cluster assignments."""
    config = SegmentationConfig(optimal_k=5, random_state=42, n_init=10)
    m1 = CustomerSegmentationModel(config=config).fit(rfm_dataset)
    m2 = CustomerSegmentationModel(config=config).fit(rfm_dataset)

    p1 = m1.predict(rfm_dataset)
    p2 = m2.predict(rfm_dataset)

    assert (p1["cluster_id"] == p2["cluster_id"]).all()
    assert (p1["cluster_label"] == p2["cluster_label"]).all()
    assert np.allclose(p1["distance_to_centroid"], p2["distance_to_centroid"], atol=1e-5)


def test_segmentation_k_range_evaluation(rfm_dataset):
    """Verifies that k evaluation correctly computes metrics for k in [2, 10]."""
    config = SegmentationConfig(k_min=2, k_max=6, n_init=5)
    model = CustomerSegmentationModel(config=config)
    eval_metrics = model.evaluate_k_range(rfm_dataset)

    assert len(eval_metrics) == 5
    for m in eval_metrics:
        assert 2 <= m["k"] <= 6
        assert m["inertia"] > 0
        assert -1.0 <= m["silhouette_score"] <= 1.0
        assert m["davies_bouldin_index"] > 0
        assert m["calinski_harabasz_index"] > 0
        assert sum(m["cluster_sizes"].values()) == 5939


def test_optimal_cluster_characteristics(trained_model):
    """Verifies the business-friendly cluster profiles and economic hierarchy of k=5."""
    profiles = trained_model.cluster_profiles_
    assert len(profiles) == 5

    # 1. Total population accounts for all 5,939 customers
    total_customers = sum(p["customer_count"] for p in profiles.values())
    assert total_customers == 5939

    # 2. Cluster 0 (Champions) has highest median monetary spend
    assert profiles["0"]["cluster_label"] == "High-Value Loyal Champions"
    assert profiles["0"]["median_monetary_net_gbp"] > 4000.0
    assert profiles["0"]["median_frequency"] >= 10
    assert profiles["0"]["revenue_pct"] > 65.0

    # 3. Cluster 1 (Active Recent Shoppers) has low recency (< 35 days)
    assert profiles["1"]["cluster_label"] == "Active Recent Shoppers"
    assert profiles["1"]["median_recency_days"] < 35.0

    # 4. Cluster 3 (Lost Low-Value Buyers) has highest recency (> 350 days)
    assert profiles["3"]["cluster_label"] == "Lost Low-Value Buyers"
    assert profiles["3"]["median_recency_days"] > 350.0

    # 5. Cluster 4 (Net Returners) has negative net monetary spend
    assert profiles["4"]["cluster_label"] == "Net Returners & Refund Adjusters"
    assert profiles["4"]["median_monetary_net_gbp"] <= 0.0


def test_model_persistence_and_reload(trained_model, rfm_dataset, tmp_path):
    """Verifies that saved model artifacts reload cleanly and yield identical predictions."""
    saved_files = trained_model.save(output_dir=tmp_path)
    assert Path(saved_files["model_path"]).exists()
    assert Path(saved_files["preprocessor_path"]).exists()
    assert Path(saved_files["schema_path"]).exists()
    assert Path(saved_files["profiles_path"]).exists()
    assert Path(saved_files["metrics_path"]).exists()
    assert Path(saved_files["metadata_path"]).exists()

    reloaded_model = CustomerSegmentationModel.load(tmp_path)
    assert reloaded_model.is_trained is True

    orig_preds = trained_model.predict(rfm_dataset.head(50))
    reloaded_preds = reloaded_model.predict(rfm_dataset.head(50))

    assert (orig_preds["cluster_id"] == reloaded_preds["cluster_id"]).all()
    assert (orig_preds["cluster_label"] == reloaded_preds["cluster_label"]).all()
    assert np.allclose(
        orig_preds["distance_to_centroid"],
        reloaded_preds["distance_to_centroid"],
        atol=1e-5,
    )


def test_single_record_inference(trained_model):
    """Verifies single dictionary record prediction and distance calculation."""
    record = {
        "customer_id": 99999,
        "recency_days": 10,
        "frequency": 25,
        "monetary_net": 12500.0,
    }

    pred = trained_model.predict(record)
    assert isinstance(pred, pd.DataFrame)
    assert len(pred) == 1
    assert pred["customer_id"].iloc[0] == 99999
    assert pred["cluster_id"].iloc[0] == 0  # Should be classified as Champion
    assert pred["cluster_label"].iloc[0] == "High-Value Loyal Champions"
    assert pred["distance_to_centroid"].iloc[0] >= 0.0


def test_negative_net_spenders_grouped_in_returners(trained_model):
    """Verifies that customers with net negative spend are placed into Net Returners cluster."""
    neg_customer = {
        "customer_id": 88888,
        "recency_days": 500,
        "frequency": 0,
        "monetary_net": -1500.0,
    }
    pred = trained_model.predict(neg_customer)
    assert pred["cluster_id"].iloc[0] == 4
    assert pred["cluster_label"].iloc[0] == "Net Returners & Refund Adjusters"


def test_extreme_spending_outlier_handling(trained_model):
    """Verifies that extreme revenue outliers (e.g. £500,000+) are robustly assigned to Champions."""
    outlier = {
        "customer_id": 77777,
        "recency_days": 2,
        "frequency": 150,
        "monetary_net": 500000.0,
    }
    pred = trained_model.predict(outlier)
    assert pred["cluster_id"].iloc[0] == 0
    assert pred["cluster_label"].iloc[0] == "High-Value Loyal Champions"


def test_invalid_inference_inputs_rejected(trained_model):
    """Verifies that invalid or incomplete inference records raise descriptive exceptions."""
    # Missing monetary_net
    with pytest.raises(ValueError, match="missing required feature columns"):
        trained_model.predict({"recency_days": 10, "frequency": 2})

    # Null value
    with pytest.raises(ValueError, match="null or NaN"):
        trained_model.predict({"recency_days": None, "frequency": 2, "monetary_net": 100.0})
