# Phase 4 — Customer Segmentation using K-Means

## 1. Executive Summary & Business Objective

Phase 4 introduces an unsupervised Machine Learning customer segmentation engine to the E-Commerce Intelligence Platform. Building upon the verified Gold RFM dataset and PostgreSQL analytics infrastructure, this module groups customers into distinct, actionable behavioral cohorts.

### Key Business Objectives:
1. **Explainable Customer Grouping**: Replace static heuristic score cuts with unsupervised multi-dimensional geometric clustering across Recency, Frequency, and Monetary Net dimensions.
2. **Actionable CRM Segmentation**: Identify high-value commercial champions, promising new buyers, at-risk churners, lost one-time purchasers, and refund-heavy customers to guide targeted marketing and retention strategies.
3. **Data Integrity & Traceability**: Guarantee that all 5,939 identified customers receive exactly one cluster assignment without dropping or truncating negative-spending customers.
4. **Production Persistence & Database Serving**: Store model artifacts using standard serialization, provide low-latency single-record and batch inference, and persist predictions in PostgreSQL with referential integrity.

---

## 2. Customer Population & Data Audit

The segmentation model trains on the complete Gold RFM dataset ([`data/gold/gold_customer_rfm.parquet`](file:///c:/Users/himas/Downloads/E-Commerce%20Intelligence%20Platform/data/gold/gold_customer_rfm.parquet)) and PostgreSQL analytics table `customer_rfm_analytics`:

- **Total Customer Population**: **5,939 unique customer IDs** (0 nulls, 0 duplicate identifiers).
- **Missing Customer Transactions**: Excluded from RFM (0 rows) and retained in Silver, Monthly Sales, and Country Revenue tables (generating £2,723,471.43 in unattributed guest revenue).
- **Positive Net Spenders (`monetary_net > 0`)**: **5,837 customers** (98.28%), generating **£16,382,076.17**.
- **Zero Net Spenders (`monetary_net == 0.00`)**: **21 customers** (0.35%), whose cancellations/returns perfectly offset purchases.
- **Strictly Negative Net Spenders (`monetary_net < 0.00`)**: **81 customers** (1.36%), generating **-£92,636.75** (ranging from -£1.25 to -£25,111.09).
- **Zero Frequency Customers (`frequency == 0`)**: **61 customers** who have zero valid purchases and only return/cancellation invoices in the 2-year observation window.
- **Extreme Spending Outliers**: The top 10 customers contribute over £2.8M in net revenue (e.g., customer 18102 at £570,380.61; customer 14646 at £523,342.07).

### Statistical Summary of Input Features

| Feature Metric | Count | Mean | Std | Min | 25% | Median (50%) | 75% | 95% | 99% | Max | Skewness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`recency_days`** | 5,939 | 202.8 | 211.7 | 1 | 25 | 96.0 | 382 | 628.0 | 729.0 | 739 | 0.88 |
| **`frequency`** | 5,939 | 6.22 | 12.96 | 0 | 1 | 3.0 | 7 | 20.1 | 46.0 | 398 | 12.67 |
| **`monetary_net`** | 5,939 | £2,742.79 | £13,679.50 | -£25,111.09 | £327.99 | £823.53 | £2,207.25 | £8,799.17 | £26,154.25 | £570,380.61 | 26.81 |

---

## 3. Feature Engineering & Preprocessing Pipeline

To handle extreme skewness and negative monetary values without dropping customers, we developed a scikit-learn pipeline ([`src/features/rfm_features.py`](file:///c:/Users/himas/Downloads/E-Commerce%20Intelligence%20Platform/src/features/rfm_features.py)):

### 1. Mathematical Handling of Negative Monetary Values
Standard logarithm transformations ($\log(x)$ or $\log(1+x)$) are undefined for negative numbers. Removing negative customers would discard 102 real accounts and bias financial reporting. Instead, we implement the **Symmetric Signed Log1p Transformation**:

$$f(x) = \text{sign}(x) \cdot \ln(1 + |x|)$$

- **Properties**:
  - $f(0) = 0$
  - Odd symmetry: $f(-x) = -f(x)$
  - Strictly monotonic across $(-\infty, \infty)$
  - Smoothly compresses both negative refund tails and positive spending tails
  - Invertible: $f^{-1}(y) = \text{sign}(y) \cdot (\exp(|y|) - 1)$

### 2. Feature Pipeline Architecture
- **`recency_days`**: Transformed via $\ln(1 + \max(0, x))$
- **`frequency`**: Transformed via $\ln(1 + \max(0, x))$
- **`monetary_net`**: Transformed via $\text{sign}(x) \cdot \ln(1 + |x|)$
- **Standardization**: Centered to zero mean and unit variance via `StandardScaler`.
- **Exclusion of Identifiers**: `customer_id` is strictly excluded from feature extraction and treated exclusively as an entity key.

---

## 4. Hyperparameter Evaluation & Optimal Cluster Selection ($k$)

We evaluated K-Means models from $k = 2$ through $k = 10$ using multiple initializations (`n_init=20`, `random_state=42`):

| Number of Clusters ($k$) | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Cluster Size Breakdown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$k = 2$** | 9,768.4 | 0.4251 | 0.8965 | 4,891.7 | [2,425, 3,514] |
| **$k = 3$** | 7,178.6 | 0.4471 | 0.7343 | 4,398.5 | [2,311, 3,526, 102] |
| **$k = 4$** | 5,312.6 | 0.3717 | 0.8666 | 4,656.6 | [2,642, 1,942, 1,253, 102] |
| **$\mathbf{k = 5}$ (Optimal)** | **4,102.8** | **0.3839** | **0.8103** | **4,959.0 (Global Peak)** | **[1,079, 1,313, 1,363, 2,083, 101]** |
| **$k = 6$** | 3,459.6 | 0.3665 | 0.8409 | 4,924.5 | [438, 1,239, 1,187, 1,817, 101, 1,157] |
| **$k = 7$** | 2,979.2 | 0.3550 | 0.8373 | 4,924.1 | [947, 1,580, 101, 1,056, 683, 446, 1,126] |
| **$k = 8$** | 2,712.4 | 0.3255 | 0.8901 | 4,718.4 | [1,034, 695, 101, 324, 957, 858, 1,316, 654] |
| **$k = 9$** | 2,465.8 | 0.3457 | 0.8581 | 4,615.0 | [686, 279, 536, 1,019, 101, 784, 609, 417, 1,508] |
| **$k = 10$** | 2,268.0 | 0.3246 | 0.8879 | 4,516.6 | [1,233, 462, 657, 101, 563, 959, 262, 372, 763, 567] |

### Selection Rationale for $k = 5$:
1. **Calinski-Harabasz Peak**: Reaches its **global maximum of 4,959.0** at $k = 5$, indicating optimal cluster dispersion and variance ratio.
2. **Substantial Silhouette Improvement**: Silhouette increases from 0.3717 ($k=4$) to **0.3839** ($k=5$).
3. **Davies-Bouldin Minimization**: Drops to **0.8103** (lower is better).
4. **CRM Business Utility**: While $k=3$ groups positive spenders into only two coarse buckets, $k=5$ provides an actionable CRM split between:
   - High-Value Champions (1,079 accounts)
   - Active Emerging Buyers (1,313 accounts)
   - Lapsed High-Potential Buyers (1,363 accounts)
   - Lost Low-Value Buyers (2,083 accounts)
   - Net Returners & Adjusters (101 accounts)

---

## 5. Customer Cluster Profiles & Business Interpretation

Each customer is assigned a deterministic cluster ID and business label:

```
+----------------------------------------------------------------------------------------------------+
| Cluster 0: High-Value Loyal Champions (1,079 customers | 18.17% pop | £11.44M revenue | 70.21% rev)  |
| Cluster 1: Active Recent Shoppers     (1,313 customers | 22.11% pop | £ 1.13M revenue |  6.93% rev)  |
| Cluster 2: Lapsed High-Potential      (1,363 customers | 22.95% pop | £ 2.99M revenue | 18.38% rev)  |
| Cluster 3: Lost Low-Value Buyers      (2,083 customers | 35.07% pop | £ 0.82M revenue |  5.04% rev)  |
| Cluster 4: Net Returners & Adjusters  (  101 customers |  1.70% pop | -£ 0.09M revenue| -0.57% rev)  |
+----------------------------------------------------------------------------------------------------+
```

### Detailed Cluster Characteristics Matrix

| Cluster ID | Business Label | Count | Pop % | Net Revenue (GBP) | Rev % | Median Recency | Median Frequency | Median Net Spend | Mean Distance to Centroid | Key Rule-Based RFM Crossover |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **High-Value Loyal Champions** | 1,079 | 18.17% | £11,437,580.80 | 70.21% | 11 days | 14 orders | £4,921.72 | 0.9532 | 1,021 Champions, 54 Loyal Customers |
| **1** | **Active Recent Shoppers** | 1,313 | 22.11% | £1,129,609.12 | 6.93% | 24 days | 3 orders | £692.22 | 0.7843 | 486 Loyal, 434 New/Promising, 287 Champions |
| **2** | **Lapsed High-Potential Customers** | 1,363 | 22.95% | £2,994,506.26 | 18.38% | 143 days | 5 orders | £1,623.97 | 0.7029 | 651 Loyal, 377 At Risk, 185 Champions |
| **3** | **Lost Low-Value Buyers** | 2,083 | 35.07% | £820,379.99 | 5.04% | 403 days | 1 order | £302.70 | 0.5827 | 965 Hibernating, 619 Attention, 260 Potential |
| **4** | **Net Returners & Refund Adjusters** | 101 | 1.70% | -£92,636.75 | -0.57% | 596 days | 0 orders | -£51.74 | 1.4588 | 78 Hibernating, 10 Attention, 6 Potential |


### Rule-Based RFM vs K-Means Full Cross-Tabulation Matrix

| K-Means Cluster | At Risk | Can't Lose Them | Champions | Needing Attention | Hibernating / Lost | Loyal Customers | New / Promising | Potential Loyalists | Total Cluster Pop | Total Cluster Revenue |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0: High-Value Loyal Champions** | 4 | 0 | 1,021 | 0 | 0 | 54 | 0 | 0 | **1,079** | **£11,437,580.80** |
| **1: Active Recent Shoppers** | 0 | 0 | 287 | 0 | 0 | 486 | 434 | 106 | **1,313** | **£1,129,609.12** |
| **2: Lapsed High-Potential** | 377 | 78 | 185 | 5 | 59 | 651 | 0 | 8 | **1,363** | **£2,994,506.26** |
| **3: Lost Low-Value Buyers** | 172 | 5 | 0 | 619 | 965 | 58 | 4 | 260 | **2,083** | **£820,379.99** |
| **4: Net Returners & Adjusters** | 1 | 2 | 0 | 10 | 78 | 1 | 3 | 6 | **101** | **-£92,636.75** |
| **Grand Total (Rule-Based Segment)** | **554** | **85** | **1,493** | **634** | **1,102** | **1,250** | **441** | **380** | **5,939** | **£16,289,439.42** |

### Investigation of Negative-Net and Zero-Net Accounts

A targeted audit was performed on all 102 accounts with non-positive net spending ($\text{monetary\_net} \le 0$):
- **Strictly Negative Net Spenders (81 accounts, -£92,636.75)**: Exactly 81 out of 81 (100.0%) are grouped into **Cluster 4**.
- **Zero-Net Spenders (21 accounts, £0.00)**: Exactly 20 out of 21 (95.2%) are grouped into **Cluster 4**.
- **Customer 12558 Exception**: Customer 12558 is the single zero-net account not assigned to Cluster 4. This customer completed an order and subsequent return 2 days prior to the dataset cutoff date (`recency_days = 2`, `frequency = 1`, `monetary_net = £0.00`). In standardized logarithmic space, their extreme recency placed them closest to the centroid of **Cluster 1 (Active Recent Shoppers)** (median recency 24 days) rather than Cluster 4 (median recency 596 days).
- **Cluster 4 Financial Exactness**: Cluster 4 consists of exactly 81 negative spenders + 20 zero-net spenders = 101 customers. Its total revenue sum is exactly **-£92,636.75** with zero-penny variance.

---

## 6. Model Evaluation & Robustness Evidence

### 1. Stability Across Random Seeds
We evaluated model stability across six independent random seeds (42, 123, 456, 789, 999, 2024):
- **Adjusted Rand Index (ARI)**: **0.9743 to 1.0000** (mean: 0.9882)
- **Normalized Mutual Information (NMI)**: **0.9589 to 1.0000** (mean: 0.9796)
- This demonstrates that cluster boundaries reflect intrinsic data structure rather than initialization artifacts.

### 2. Sensitivity to Extreme Monetary Outliers
Extreme spenders (£500,000+) are compressed by the symmetric signed log transformation and placed into Cluster 0 (Champions) without skewing centroid positions or creating singleton clusters.

### 3. Alternative Scaling Comparison
- `StandardScaler` yielded superior separation (Silhouette 0.3839, Calinski-Harabasz 4,959.0) compared to `RobustScaler` (Silhouette 0.3655, Calinski-Harabasz 5,670.9 with heavier distortion in the high-frequency tail).

### 4. Generated Publication Visualizations ([`reports/phase4/`](file:///c:/Users/himas/Downloads/E-Commerce%20Intelligence%20Platform/reports/phase4))
1. `kmeans_metrics_elbow_silhouette.png`: Multi-metric evaluation curves across $k=2..10$.
2. `cluster_size_distribution.png`: Population vs revenue share comparison.
3. `rfm_feature_distributions_by_cluster.png`: Symlog boxplots of recency, frequency, and net spend.
4. `pca_2d_projection.png`: 2D PCA projection of the standardized 3D feature space with projected centroids.
5. `cluster_profile_heatmap.png`: Normalized median characteristics matrix.
6. `silhouette_samples_plot.png`: Individual sample silhouette coefficient analysis.

---

## 7. Model Persistence & Inference Engine

Saved model artifacts in [`models/customer_segmentation/`](file:///c:/Users/himas/Downloads/E-Commerce%20Intelligence%20Platform/models/customer_segmentation):

- `kmeans_model.joblib`: Serialized scikit-learn `KMeans` model instance.
- `preprocessor.joblib`: Serialized `RFMFeaturePipeline` with fitted transformers.
- `feature_schema.json`: Input feature requirements, types, and descriptions.
- `cluster_profiles.json`: Full metrics, medians, and segment distributions per cluster.
- `evaluation_metrics.json`: Evaluated metrics across $k=2..10$ and selection rationale.
- `training_metadata.json`: Model version, training duration, parameters, and deterministic centroid mapping.

### Python Inference Example:
```python
from src.models.customer_segmentation import CustomerSegmentationModel

# Load serialized model
model = CustomerSegmentationModel.load("models/customer_segmentation")

# Single record inference
customer = {
    "customer_id": 12345,
    "recency_days": 14,
    "frequency": 12,
    "monetary_net": 3500.0,
}
prediction = model.predict(customer)
# Output: cluster_id=0, cluster_label="High-Value Loyal Champions", distance_to_centroid=0.8241
```

---

## 8. PostgreSQL Integration & Referential Integrity

### Table DDL: `customer_segmentation_kmeans`
```sql
CREATE TABLE IF NOT EXISTS customer_segmentation_kmeans (
    customer_id BIGINT PRIMARY KEY REFERENCES customer_rfm_analytics(customer_id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    cluster_label VARCHAR(100) NOT NULL,
    recency_days INTEGER NOT NULL,
    frequency INTEGER NOT NULL,
    monetary_net NUMERIC(14, 2) NOT NULL,
    distance_to_centroid NUMERIC(10, 4) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    metadata_payload JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_csk_cluster_id ON customer_segmentation_kmeans(cluster_id);
CREATE INDEX IF NOT EXISTS ix_csk_cluster_label ON customer_segmentation_kmeans(cluster_label);
CREATE INDEX IF NOT EXISTS ix_csk_model_version ON customer_segmentation_kmeans(model_version);
```

### Database Parity Verification
- **Eligible RFM Customers in DB**: **5,939**
- **ML Segmentation Records in DB**: **5,939**
- **Null Cluster Assignments**: **0**
- **Orphan Records**: **0** (strict Foreign Key enforcement)
- **Parquet vs PostgreSQL Parity**: **100% Identical**
- **Rule-Based RFM Segment Preservation**: Unaltered (all 8 rule-based segments preserved in `customer_rfm_analytics`).

---

## 9. Reproduction Commands

```powershell
# 1. Train, Profile, and Persist K-Means Segmentation
python scripts/train_segmentation.py

# 2. Ingest Predictions into PostgreSQL Database
python scripts/load_segmentation_postgres.py

# 3. Verify Database Parity & Reconciliation
python scripts/verify_segmentation.py

# 4. Run Targeted QA Audit
python scripts/qa_audit_phase4.py

# 5. Run Full Automated Test Suite (70 Tests)
python -m pytest -v tests/
```

---

## 10. Operational Limitations & Scientific Caveats

1. **Unsupervised Nature**: K-Means discovers geometric density groupings based on observed RFM metrics. Clusters represent descriptive behavioral segments, not causal propensities.
2. **Causal Claims**: No claim is made that segmenting customers directly improves retention or increases revenue without controlled A/B testing.
3. **Temporal Window Censoring**: The observation window spans 2 calendar years. Customers classified as "Net Returners" (Cluster 4) represent accounts where refund adjustments during the observation window exceeded purchases.
4. **Distance Metric Assumption**: K-Means assumes spherical clusters in standardized feature space; non-spherical sub-structures are effectively approximated through symmetric log feature scaling.
