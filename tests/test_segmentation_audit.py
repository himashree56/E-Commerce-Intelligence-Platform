"""Targeted QA Audit Regression Tests for Phase 4 Customer Segmentation."""
import json
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.database.connection import DatabaseManager
from src.models.config import SegmentationConfig
from src.models.customer_segmentation import CustomerSegmentationModel


@pytest.fixture(scope="module")
def audit_data():
    """Provides Gold RFM and Scored Segmentation Parquet DataFrames."""
    config = SegmentationConfig()
    rfm_path = config.gold_rfm_path
    seg_path = config.base_dir / "data" / "gold" / "gold_customer_segmentation_kmeans.parquet"

    if not rfm_path.exists() or not seg_path.exists():
        pytest.skip("Required Parquet datasets not found.")

    rfm_df = pd.read_parquet(rfm_path)
    seg_df = pd.read_parquet(seg_path)
    merged = pd.merge(
        rfm_df,
        seg_df[["customer_id", "cluster_id", "cluster_label", "distance_to_centroid"]],
        on="customer_id",
    )
    return {
        "rfm": rfm_df,
        "seg": seg_df,
        "merged": merged,
    }


def test_audit_population_and_revenue_exact_totals(audit_data):
    """Verifies that all 5 cluster populations sum to 5,939 and revenues sum to £16,289,439.42."""
    merged = audit_data["merged"]

    expected_populations = {
        0: 1079,
        1: 1313,
        2: 1363,
        3: 2083,
        4: 101,
    }

    expected_revenues = {
        0: Decimal("11437580.80"),
        1: Decimal("1129609.12"),
        2: Decimal("2994506.26"),
        3: Decimal("820379.99"),
        4: Decimal("-92636.75"),
    }

    actual_pops = merged.groupby("cluster_id").size().to_dict()
    assert actual_pops == expected_populations
    assert sum(actual_pops.values()) == 5939

    # Financial sums using fixed Decimal precision
    actual_revs = {}
    for cid in range(5):
        s = merged.loc[merged["cluster_id"] == cid, "monetary_net"]
        actual_revs[cid] = sum(Decimal(str(round(v, 2))) for v in s)
        assert actual_revs[cid] == expected_revenues[cid]

    total_revenue = sum(actual_revs.values())
    assert total_revenue == Decimal("16289439.42")


def test_audit_crosstabulation_invariants(audit_data):
    """Verifies row sums, column sums, and grand total in RFM vs K-Means cross-tabulation."""
    merged = audit_data["merged"]
    rfm_df = audit_data["rfm"]

    ct = pd.crosstab(merged["cluster_id"], merged["customer_segment"], margins=True)

    # 1. Grand total
    assert ct.loc["All", "All"] == 5939

    # 2. Row totals equal cluster populations
    expected_populations = {0: 1079, 1: 1313, 2: 1363, 3: 2083, 4: 101}
    for cid, pop in expected_populations.items():
        assert ct.loc[cid, "All"] == pop

    # 3. Column totals equal rule-based segment populations
    segment_counts = rfm_df["customer_segment"].value_counts().to_dict()
    for seg_name, count in segment_counts.items():
        assert ct.loc["All", seg_name] == count


def test_audit_negative_and_zero_net_customers_distribution(audit_data):
    """Verifies the exact distribution of 81 negative-net and 21 zero-net customers."""
    merged = audit_data["merged"]

    neg_df = merged[merged["monetary_net"] < 0]
    zero_df = merged[merged["monetary_net"] == 0]

    assert len(neg_df) == 81
    assert len(zero_df) == 21
    assert len(neg_df) + len(zero_df) == 102

    # Negative revenue total
    neg_rev_sum = sum(Decimal(str(round(v, 2))) for v in neg_df["monetary_net"])
    assert neg_rev_sum == Decimal("-92636.75")

    # All 81 negative customers must be in Cluster 4
    assert (neg_df["cluster_id"] == 4).all()

    # Customer 12558 is the lone zero-net customer in Cluster 1 (Active Recent Shoppers)
    cust_12558 = merged[merged["customer_id"] == 12558]
    assert len(cust_12558) == 1
    assert cust_12558["monetary_net"].iloc[0] == 0.0
    assert cust_12558["recency_days"].iloc[0] == 2
    assert cust_12558["cluster_id"].iloc[0] == 1
    assert cust_12558["cluster_label"].iloc[0] == "Active Recent Shoppers"

    # Exactly 20 zero-net customers must be in Cluster 4
    zero_c4 = zero_df[zero_df["cluster_id"] == 4]
    assert len(zero_c4) == 20

    # Cluster 4 composition: exactly 101 customers (81 negative + 20 zero-net + 0 positive)
    c4_df = merged[merged["cluster_id"] == 4]
    assert len(c4_df) == 101
    assert (c4_df["monetary_net"] > 0).sum() == 0
    c4_rev_sum = sum(Decimal(str(round(v, 2))) for v in c4_df["monetary_net"])
    assert c4_rev_sum == Decimal("-92636.75")


def test_audit_model_selection_k_metrics():
    """Verifies k=2..10 evaluation metrics, silhouette tradeoff, and Calinski-Harabasz peak."""
    metrics_path = Path("models/customer_segmentation/evaluation_metrics.json")
    assert metrics_path.exists()

    with open(metrics_path) as f:
        data = json.load(f)

    k_metrics = {m["k"]: m for m in data["tested_k_metrics"]}
    assert len(k_metrics) == 9  # k in 2..10

    # Calinski-Harabasz reaches global peak at k=5
    ch_k5 = k_metrics[5]["calinski_harabasz_index"]
    for k, m in k_metrics.items():
        if k != 5:
            assert ch_k5 > m["calinski_harabasz_index"]
    assert ch_k5 == 4959.0

    # k=3 has higher silhouette (0.4473) than k=5 (0.3839)
    assert k_metrics[3]["silhouette_score"] > k_metrics[5]["silhouette_score"]
    assert round(k_metrics[3]["silhouette_score"], 4) == 0.4473
    assert round(k_metrics[5]["silhouette_score"], 4) == 0.3839


def test_audit_six_seed_stability(audit_data):
    """Verifies that k=5 clustering across 6 independent seeds has ARI > 0.95 and NMI > 0.95."""
    rfm_df = audit_data["rfm"]
    seeds = [42, 123, 456, 789, 999, 2024]
    assignments = {}

    for s in seeds:
        cfg = SegmentationConfig(optimal_k=5, random_state=s, n_init=20)
        m = CustomerSegmentationModel(config=cfg).fit(rfm_df, evaluate_k=False)
        preds = m.predict(rfm_df)
        assignments[s] = preds["cluster_id"].values

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            s1, s2 = seeds[i], seeds[j]
            ari = adjusted_rand_score(assignments[s1], assignments[s2])
            nmi = normalized_mutual_info_score(assignments[s1], assignments[s2])
            assert ari >= 0.95, f"Low ARI between seed {s1} and {s2}: {ari}"
            assert nmi >= 0.95, f"Low NMI between seed {s1} and {s2}: {nmi}"


def test_audit_model_serialization_and_db_parity(audit_data):
    """Verifies that reloaded model produces 100% parity with saved Parquet and PostgreSQL."""
    rfm_df = audit_data["rfm"]
    seg_df = audit_data["seg"]

    model = CustomerSegmentationModel.load("models/customer_segmentation")
    reloaded_preds = model.predict(rfm_df)

    # Parquet parity
    assert (reloaded_preds["customer_id"].values == seg_df["customer_id"].values).all()
    assert (reloaded_preds["cluster_id"].values == seg_df["cluster_id"].values).all()
    assert (reloaded_preds["cluster_label"].values == seg_df["cluster_label"].values).all()
    assert np.allclose(
        reloaded_preds["distance_to_centroid"].values,
        seg_df["distance_to_centroid"].values,
        atol=1e-5,
    )

    # PostgreSQL parity (if database is reachable)
    db_mgr = DatabaseManager()
    try:
        health = db_mgr.check_health()
        if health.get("status") != "HEALTHY":
            return
    except Exception:
        return

    with db_mgr.engine.connect() as conn:
        db_df = pd.read_sql_query(
            "SELECT customer_id, cluster_id, cluster_label, distance_to_centroid "
            "FROM customer_segmentation_kmeans ORDER BY customer_id ASC;",
            conn,
        )

    assert len(db_df) == 5939
    sorted_seg = seg_df.sort_values("customer_id").reset_index(drop=True)
    assert (db_df["customer_id"].values == sorted_seg["customer_id"].values).all()
    assert (db_df["cluster_id"].values == sorted_seg["cluster_id"].values).all()
    assert (db_df["cluster_label"].values == sorted_seg["cluster_label"].values).all()
    assert np.allclose(
        db_df["distance_to_centroid"].astype(float).values,
        sorted_seg["distance_to_centroid"].values,
        atol=1e-4,
    )


def test_audit_cluster_business_labels_consistency():
    """Verifies cluster business labels match their economic hierarchy and remain invariant."""
    profiles_path = Path("models/customer_segmentation/cluster_profiles.json")
    with open(profiles_path) as f:
        profiles = json.load(f)

    # 0: Champions
    assert profiles["0"]["cluster_label"] == "High-Value Loyal Champions"
    assert profiles["0"]["median_monetary_net_gbp"] > 4000.0

    # 1: Active Shoppers
    assert profiles["1"]["cluster_label"] == "Active Recent Shoppers"
    assert profiles["1"]["median_recency_days"] < 35.0

    # 2: Lapsed High-Potential
    assert profiles["2"]["cluster_label"] == "Lapsed High-Potential Customers"
    assert profiles["2"]["median_recency_days"] > 100.0
    assert profiles["2"]["median_monetary_net_gbp"] > 1000.0

    # 3: Lost Low-Value
    assert profiles["3"]["cluster_label"] == "Lost Low-Value Buyers"
    assert profiles["3"]["median_recency_days"] > 350.0

    # 4: Net Returners
    assert profiles["4"]["cluster_label"] == "Net Returners & Refund Adjusters"
    assert profiles["4"]["median_monetary_net_gbp"] < 0.0
