import json
import sys
from decimal import Decimal
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.database.connection import DatabaseManager
from src.models.config import SegmentationConfig
from src.models.customer_segmentation import CustomerSegmentationModel

def run_qa_audit():
    print("=" * 80)
    print("PHASE 4 TARGETED QA AUDIT SCRIPT")
    print("=" * 80)

    # Load Gold RFM and Scored Segmentation Parquet
    rfm_path = "data/gold/gold_customer_rfm.parquet"
    seg_path = "data/gold/gold_customer_segmentation_kmeans.parquet"

    rfm_df = pd.read_parquet(rfm_path)
    seg_df = pd.read_parquet(seg_path)

    print(f"Loaded Gold RFM: {len(rfm_df)} rows")
    print(f"Loaded Scored Segmentation: {len(seg_df)} rows")

    # 1 & 2: Recompute cluster populations and revenue totals
    merged = pd.merge(rfm_df, seg_df[['customer_id', 'cluster_id', 'cluster_label', 'distance_to_centroid']], on='customer_id')
    assert len(merged) == 5939, f"Merge row count {len(merged)} != 5939"

    pop_counts = merged.groupby(['cluster_id', 'cluster_label']).size()
    rev_totals = merged.groupby(['cluster_id', 'cluster_label'])['monetary_net'].sum()
    rev_dec = merged.groupby(['cluster_id', 'cluster_label'])['monetary_net'].apply(lambda s: sum(Decimal(str(round(v, 2))) for v in s))

    print("\n--- TASK 1 & 2: CLUSTER POPULATIONS & REVENUE TOTALS ---")
    total_pop = 0
    total_rev = Decimal("0.00")
    for (cid, clabel), count in pop_counts.items():
        rev = rev_dec[(cid, clabel)]
        rev_f = rev_totals[(cid, clabel)]
        total_pop += count
        total_rev += rev
        print(f"Cluster {cid} ({clabel}): Population = {count:,} ({count/len(merged)*100:.2f}%), Revenue = £{rev:,.2f} ({float(rev)/16289439.42*100:.2f}%)")

    print(f"\nTotal Population: {total_pop} (Expected: 5,939)")
    print(f"Total Net Revenue: £{total_rev} (Expected: £16,289,439.42)")
    assert total_pop == 5939
    assert total_rev == Decimal("16289439.42")

    # 3: Cross-tabulation Rule-Based RFM vs K-Means
    print("\n--- TASK 3: CROSS-TABULATION (K-Means vs Rule-Based RFM) ---")
    ct = pd.crosstab(merged['cluster_id'], merged['customer_segment'], margins=True)
    print(ct.to_string())

    # Assert row totals, col totals, grand total
    for cid in range(5):
        assert ct.loc[cid, 'All'] == pop_counts.xs(cid, level=0).values[0]
    for seg_name in rfm_df['customer_segment'].unique():
        assert ct.loc['All', seg_name] == (rfm_df['customer_segment'] == seg_name).sum()
    assert ct.loc['All', 'All'] == 5939
    print("Cross-tabulation assertions PASSED (all row, col, grand totals match).")

    # 4: Investigate 81 negative-net and 21 zero-net customers
    print("\n--- TASK 4: 81 NEGATIVE-NET & 21 ZERO-NET CUSTOMERS DISTRIBUTION ---")
    neg_customers = merged[merged['monetary_net'] < 0]
    zero_customers = merged[merged['monetary_net'] == 0]
    assert len(neg_customers) == 81, f"Negative count {len(neg_customers)} != 81"
    assert len(zero_customers) == 21, f"Zero count {len(zero_customers)} != 21"

    print(f"Negative-net count: {len(neg_customers)}, Net Sum: £{neg_customers['monetary_net'].sum():.2f}")
    print(f"Zero-net count: {len(zero_customers)}, Net Sum: £{zero_customers['monetary_net'].sum():.2f}")

    print("\nDistribution of negative-net customers across clusters:")
    print(neg_customers['cluster_id'].value_counts().to_dict())

    print("\nDistribution of zero-net customers across clusters:")
    print(zero_customers['cluster_id'].value_counts().to_dict())

    # Inspect the zero-net customer not in cluster 4
    non_c4_zero = zero_customers[zero_customers['cluster_id'] != 4]
    print("\nZero-net customers not in Cluster 4:")
    print(non_c4_zero[['customer_id', 'cluster_id', 'cluster_label', 'recency_days', 'frequency', 'monetary_net', 'distance_to_centroid']])

    c4_customers = merged[merged['cluster_id'] == 4]
    print(f"\nCluster 4 Breakdown: Total Count = {len(c4_customers)}")
    print(f"  Negative customers in C4: {len(c4_customers[c4_customers['monetary_net'] < 0])}")
    print(f"  Zero-net customers in C4: {len(c4_customers[c4_customers['monetary_net'] == 0])}")
    print(f"  Positive customers in C4: {len(c4_customers[c4_customers['monetary_net'] > 0])}")
    print(f"  Total C4 Revenue: £{c4_customers['monetary_net'].sum():.2f}")

    # 5: Model Selection Metrics (k=2..10)
    print("\n--- TASK 5: MODEL SELECTION METRICS (k=2..10) ---")
    with open("models/customer_segmentation/evaluation_metrics.json") as f:
        metrics_data = json.load(f)
    print(f"{'k':<4} {'Inertia':<10} {'Silhouette':<12} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<18} {'Cluster Sizes'}")
    for m in metrics_data.get('tested_k_metrics', metrics_data.get('k_metrics', [])):
        sizes = list(m['cluster_sizes'].values())
        print(f"{m['k']:<4} {m['inertia']:<10.2f} {m['silhouette_score']:<12.4f} {m['davies_bouldin_index']:<15.4f} {m['calinski_harabasz_index']:<18.2f} {sizes}")

    # 6: Six-Seed Stability
    print("\n--- TASK 6: SIX-SEED STABILITY EXPERIMENT ---")
    seeds = [42, 123, 456, 789, 999, 2024]
    models = {}
    assignments = {}
    for s in seeds:
        cfg = SegmentationConfig(optimal_k=5, random_state=s, n_init=20)
        m = CustomerSegmentationModel(config=cfg)
        m.fit(rfm_df)
        models[s] = m
        preds = m.predict(rfm_df)
        assignments[s] = preds['cluster_id'].values

    print("Pairwise Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI):")
    ari_list = []
    nmi_list = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            s1, s2 = seeds[i], seeds[j]
            ari = adjusted_rand_score(assignments[s1], assignments[s2])
            nmi = normalized_mutual_info_score(assignments[s1], assignments[s2])
            ari_list.append(ari)
            nmi_list.append(nmi)
            print(f"Seeds {s1:>4} vs {s2:>4}: ARI = {ari:.4f}, NMI = {nmi:.4f}")

    print(f"ARI: min = {min(ari_list):.4f}, mean = {np.mean(ari_list):.4f}, max = {max(ari_list):.4f}")
    print(f"NMI: min = {min(nmi_list):.4f}, mean = {np.mean(nmi_list):.4f}, max = {max(nmi_list):.4f}")

    # 7: Reload Saved Model Artifacts & Verify PostgreSQL Parity
    print("\n--- TASK 7: MODEL RELOAD & POSTGRESQL PARITY ---")
    loaded_model = CustomerSegmentationModel.load("models/customer_segmentation")
    reloaded_preds = loaded_model.predict(rfm_df)
    
    # Compare with saved parquet
    assert (reloaded_preds['customer_id'].values == seg_df['customer_id'].values).all()
    assert (reloaded_preds['cluster_id'].values == seg_df['cluster_id'].values).all()
    assert (reloaded_preds['cluster_label'].values == seg_df['cluster_label'].values).all()
    assert np.allclose(reloaded_preds['distance_to_centroid'].values, seg_df['distance_to_centroid'].values, atol=1e-5)
    print("Reloaded model predictions match gold_customer_segmentation_kmeans.parquet exactly (100% match across 5,939 rows).")

    # Compare with PostgreSQL table
    db_mgr = DatabaseManager()
    with db_mgr.engine.connect() as conn:
        db_df = pd.read_sql_query("SELECT customer_id, cluster_id, cluster_label, distance_to_centroid FROM customer_segmentation_kmeans ORDER BY customer_id ASC;", conn)
    
    assert len(db_df) == 5939
    sorted_seg = seg_df.sort_values('customer_id').reset_index(drop=True)
    assert (db_df['customer_id'].values == sorted_seg['customer_id'].values).all()
    assert (db_df['cluster_id'].values == sorted_seg['cluster_id'].values).all()
    assert (db_df['cluster_label'].values == sorted_seg['cluster_label'].values).all()
    assert np.allclose(db_df['distance_to_centroid'].astype(float).values, sorted_seg['distance_to_centroid'].values, atol=1e-4)
    print("PostgreSQL table matches gold_customer_segmentation_kmeans.parquet exactly (100% parity across 5,939 rows).")

    # 8: Verify Cluster Business Labels based on Actual Profiles
    print("\n--- TASK 8: VERIFY CLUSTER BUSINESS LABELS ---")
    with open("models/customer_segmentation/cluster_profiles.json") as f:
        profiles = json.load(f)
    for cid_str, p in profiles.items():
        print(f"Cluster {cid_str}: {p['cluster_label']}")
        print(f"  Median Recency: {p['median_recency_days']} days, Median Frequency: {p['median_frequency']}, Median Net Spend: £{p['median_monetary_net_gbp']:.2f}")

    print("\nALL QA AUDIT TASKS VERIFIED SUCCESSFULLY!")

if __name__ == '__main__':
    run_qa_audit()
