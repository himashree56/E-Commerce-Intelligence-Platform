# Phase 6: XGBoost Repeat-Purchase Prediction & Independent QA Audit

## 1. Executive Summary & Business Objectives

Phase 6 introduces an enterprise-grade, supervised machine learning pipeline to predict customer repeat-purchase propensity over a 30-day forward window. Using an extreme gradient boosted decision tree ensemble (**XGBoost**), the platform generates calibrated, point-in-time purchase probabilities for commercial decision-making (such as targeted lifecycle marketing, retention outreach, and personalized re-engagement), while strictly enforcing leakage prevention and temporal boundary integrity.

### Key Architectural Commitments
1. **Strict Temporal Integrity & Non-Overlapping Purged Splits**: Models are trained and evaluated across chronological snapshots separated by a 31-day temporal purging gap to prevent future-label leakage across splits. Random `train_test_split` is strictly avoided.
2. **Right-Censoring Prevention**: Snapshots are placed strictly within fully observable historical periods ($T_{\text{obs}} + 30\text{d} \le T_{\text{dataset\_end}}$), preventing unobserved periods from being falsely recorded as negative instances.
3. **Point-in-Time Feature Engineering**: All 18 candidate features are computed using transactions timestamped strictly prior to the snapshot cutoff ($t < T_{\text{obs}}$). Global aggregates, full-dataset clusters, and future RFM metrics are explicitly excluded.
4. **Validation-Anchored Decision Threshold**: Operational decision thresholds ($t = 0.5900$) are determined strictly on validation data to maximize the $F_1$ score and applied out-of-sample to held-out test data.
5. **Clear Predictive Boundaries**: The model predicts *predictive propensity to repeat purchase*, not customer churn, customer lifetime value (LTV), or causal marketing uplift.
6. **Empirical Calibration Verification**: Raw vs. post-hoc probability calibration (Sigmoid / Platt scaling and Isotonic regression) is systematically audited, explaining the role of positive-class loss reweighting (`scale_pos_weight = 4.081`) on Brier scores.

---

## 2. Prediction Contract & Modeling Unit

### 2.1 Formal Prediction Problem
Given an observation cutoff date $T_{\text{obs}}$ and an existing customer $C_i$, predict whether $C_i$ will complete at least one valid commercial transaction within the forward 30-day prediction window:
$$\mathcal{W}_{\text{forward}} = [T_{\text{obs}}, T_{\text{obs}} + 30\text{ days}]$$

### 2.2 Modeling Unit & Primary Key
The modeling unit is the **customer-snapshot record**:
$$\text{Unit} = (C_i, T_{\text{obs}})$$
In relational persistence, this is identified by the composite key `(customer_id, snapshot_date)`.

### 2.3 Customer Eligibility Criteria
A customer $C_i$ is eligible for inclusion in the snapshot population at $T_{\text{obs}}$ if and only if:
1. **Identified Customer**: `customer_id` is non-null.
2. **Historical Commercial Activity**: The customer has completed at least one `VALID_PURCHASE` transaction with timestamp $t < T_{\text{obs}}$.
3. **Positive Historical Quantity & Price**: Historical orders reflect genuine commercial purchases rather than pure inventory adjustments or zero-price accounting records.

Guest checkouts (`customer_id` is null) cannot be tracked longitudinally and are excluded from customer-level repeat purchase modeling.

### 2.4 Label Construction & Cancellation Rules
The target label $Y \in \{0, 1\}$ is defined as:
$$Y = \begin{cases} 
1 & \text{if } \exists \text{ transaction } \tau \in \mathcal{W}_{\text{forward}} \text{ with customer\_id} = C_i \text{ and } \text{status} = \text{'VALID\_PURCHASE'} \\
0 & \text{otherwise}
\end{cases}$$

**Operational Rules**:
- **Cancellations (`C` prefix)** occurring in $\mathcal{W}_{\text{forward}}$ do **NOT** count as positive repeat purchases. A customer who only initiates returns in the window is assigned $Y = 0$.
- **Cancellations reversing an order placed within the window**: If an order is made and subsequently cancelled, the purchase event is recorded; however, return quantities are netted in financial feature aggregations.
- **Absence of Purchase vs. Censoring**: Customers with no purchases in $\mathcal{W}_{\text{forward}}$ are recorded as $Y = 0$ if and only if the full 30-day window is observable in the Silver dataset.

### 2.5 Scope & Non-Causality Boundaries
- **Not Customer Churn**: Churn implies permanent abandonment; non-purchase within 30 days is common in wholesale/specialty retail and does not indicate churn.
- **Not Customer Lifetime Value (LTV)**: The model predicts binary purchase occurrence, not future revenue magnitude.
- **Not Causal Uplift**: Predictions measure natural baseline repeat probability under historical conditions. They do not predict how a customer will respond to marketing interventions (uplift modeling).

---

## 3. Temporal Validation Strategy & Population Distribution

### 3.1 Historical Date Range & Split Windows
The Silver dataset spans from `2009-12-01 07:45:00` to `2011-12-09 12:50:00` (738 days). The latest possible snapshot date with a fully observable 30-day forward window is `2011-11-09`.

To guarantee complete separation and eliminate cross-split label overlap, the dataset is split into chronological partitions separated by a 31-day purging gap:

```
[--- Train Snapshots (4 dates) ---] [Gap 1] [Val Snapshot] [Gap 2] [Test Snapshot] [Safety Gap] [Dataset End]
2010-09-01, 12-01, 03-01, 06-01    Jul 2011  2011-08-01    Sep 2011  2011-10-01      39.5d      2011-12-09
```

| Partition | Snapshot Date(s) | Observation Cutoff | 30-Day Forward Label Window | Purging Buffer |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `2010-09-01`<br>`2010-12-01`<br>`2011-03-01`<br>`2011-06-01` | Cutoff at 00:00:00 on snapshot date | `[T, T + 30d]` (non-overlapping quarterly dates) | 31-day gap (`2011-07-01` to `2011-08-01`) before validation cutoff |
| **Validation** | `2011-08-01` | `2011-08-01 00:00:00` | `2011-08-01` to `2011-08-31 23:59:59` | 31-day gap (`2011-09-01` to `2011-10-01`) before test cutoff |
| **Held-Out Test** | `2011-10-01` | `2011-10-01 00:00:00` | `2011-10-01` to `2011-10-31 23:59:59` | Fully observable; ends 39.53 days before dataset end (`2011-12-09`) |

### 3.2 Population Statistics & Reconciliation Audit
The snapshot population exhibits stable positive class prevalence across chronological splits (~16% to 20%):

| Split | Customer Snapshots | Unique Customers | Positive Class ($Y=1$) | Negative Class ($Y=0$) | Class Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 17,057 | 4,933 | 3,357 | 13,700 | **19.68%** |
| **Validation** | 5,143 | 5,143 | 811 | 4,332 | **15.77%** |
| **Held-Out Test** | 5,438 | 5,438 | 1,110 | 4,328 | **20.41%** |
| **Total Model Corpus** | **27,638** | **5,438** | **5,278** | **22,360** | **19.10%** |

#### Population Reconciliation Findings
- **Total Snapshots Reconciled**: $17,057 + 5,143 + 5,438 = \mathbf{27,638}$. Duplicate `(customer_id, snapshot_date)` keys: **0**.
- **Unique Customers in Corpus vs. Full Silver Dataset**:
  - The total unique customers in the modeling corpus (Train $\cup$ Val $\cup$ Test) is **5,438**. Because eligibility requires $\ge 1$ past purchase and is monotonically non-decreasing over time, all customers active at earlier snapshots are eligible at the final test cutoff (`2011-10-01`). Thus, the unique customer count in Test equals the total unique customer count across the corpus.
  - Across the entire two-year Silver dataset (`2009-12-01` to `2011-12-09`), there are **5,878 unique customers** with at least one `VALID_PURCHASE`.
  - The **440 customer discrepancy** ($5,878 - 5,438 = 440$) represents newly acquired customers who completed their first purchase strictly between `2011-10-01 00:00:00` and `2011-12-09 12:50:00` (during the peak Q4 holiday surge) and were thus rightfully ineligible at the final snapshot date.

---

## 4. Leakage-Safe Feature Engineering Pipeline

### 4.1 Feature Specifications (18 Point-in-Time Features)
All features are derived strictly from transactions timestamped $t < T_{\text{obs}}$:

| # | Feature Name | Description | Calculation Formula ($t < T_{\text{obs}}$) | Safeguard Against Leakage |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `recency_days` | Days since customer's most recent valid purchase | $(T_{\text{obs}} - \max(t_{\text{valid}}))_{\text{days}}$ | Zero access to future purchase timestamps. |
| 2 | `tenure_days` | Days since customer's first observed valid purchase | $(T_{\text{obs}} - \min(t_{\text{valid}}))_{\text{days}}$ | Fixed historical anchor. |
| 3 | `purchase_frequency` | Cumulative distinct valid invoices | $\text{count}(\text{distinct invoice\_no where valid})$ | Strictly pre-cutoff orders. |
| 4 | `net_monetary_value` | Cumulative net customer spend | $\sum (\text{gross\_sales} - \text{return\_amount})$ | Includes only settled pre-cutoff transactions. |
| 5 | `gross_sales` | Cumulative gross spend | $\sum (\text{valid quantity} \times \text{unit\_price})$ | Strictly pre-cutoff orders. |
| 6 | `return_amount` | Cumulative monetary value of returns | $\sum (|\text{return quantity}| \times \text{unit\_price})$ | Cancellations post-cutoff are excluded. |
| 7 | `return_ratio` | Proportion of gross spend returned | $\text{clip}(\text{return\_amount} / \max(0.01, \text{gross\_sales}), 0.0, 10.0)$ | Bounded ratio $[0, 10.0]$, safe from division by zero. |
| 8 | `cancellation_frequency` | Cumulative distinct cancellation invoices | $\text{count}(\text{distinct cancellation invoices})$ | Strictly pre-cutoff returns. |
| 9 | `avg_order_value` | Historical average revenue per order | $\text{net\_monetary\_value} / \max(1, \text{purchase\_frequency})$ | Safe historical aggregation. |
| 10 | `avg_order_interval_days` | Average cadence between historical purchases | $\text{tenure\_days} / \max(1, \text{purchase\_frequency} - 1)$ | Single-purchase customers defaulted to recency. |
| 11 | `purchases_last_30d` | Invoices in recent 30-day lookback | Invoices in $[T_{\text{obs}} - 30\text{d}, T_{\text{obs}})$ | Strict backward lookback window. |
| 12 | `purchases_last_60d` | Invoices in recent 60-day lookback | Invoices in $[T_{\text{obs}} - 60\text{d}, T_{\text{obs}})$ | Strict backward lookback window. |
| 13 | `purchases_last_90d` | Invoices in recent 90-day lookback | Invoices in $[T_{\text{obs}} - 90\text{d}, T_{\text{obs}})$ | Strict backward lookback window. |
| 14 | `spend_last_30d` | Net spending in recent 30-day lookback | Spend in $[T_{\text{obs}} - 30\text{d}, T_{\text{obs}})$ | Strict backward lookback window. |
| 15 | `spend_last_60d` | Net spending in recent 60-day lookback | Spend in $[T_{\text{obs}} - 60\text{d}, T_{\text{obs}})$ | Strict backward lookback window. |
| 16 | `spend_last_90d` | Net spending in recent 90-day lookback | Spend in $[T_{\text{obs}} - 90\text{d}, T_{\text{obs}})$ | Strict backward lookback window. |
| 17 | `product_diversity` | Cumulative distinct product SKUs purchased | $\text{count}(\text{distinct stock\_code})$ | Strictly pre-cutoff items. |
| 18 | `total_quantity_purchased` | Cumulative physical items purchased | $\sum (\text{valid quantity})$ | Strictly pre-cutoff quantities. |

### 4.2 Preprocessing & Scaling
- Continuous features are normalized using `scikit-learn`'s `StandardScaler`.
- **Preconditioning Safeguard**: The scaler is fitted **strictly on the training dataset** and frozen. Validation and test partitions are transformed out-of-sample using the frozen mean and variance parameters, preventing data snooping.
- **Finite Value Audit**: Across all 27,638 snapshots, there are **0 NaNs** and **0 Infs**. Return ratios range from `0.0000` to `8.5724` (strictly within bounds), and minimum inter-order intervals are strictly positive.

---

## 5. Model Architecture, Baselines & Hyperparameters

### 5.1 Training Specifications & Selected Hyperparameters
The XGBoost model was tuned on validation PR-AUC over candidate hyperparameter configurations. The final optimal parameters are:

```json
{
  "n_estimators": 100,
  "max_depth": 3,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "scale_pos_weight": 4.081,
  "reg_alpha": 0.1,
  "reg_lambda": 1.0,
  "objective": "binary:logistic",
  "eval_metric": "aucpr",
  "random_state": 42
}
```

*Role of `scale_pos_weight`*:
`scale_pos_weight` was set to the empirical class imbalance ratio:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{13,700}{3,357} = 4.081$$
This places $4\times$ weight on correctly identifying positive repeat purchasers, maximizing $F_1$ and ranking precision in early deciles.

### 5.2 Baselines
1. **Dummy Baseline (`DummyClassifier(strategy='prior')`)**: Predicts the empirical training prior probability ($19.68\%$) across all test instances.
2. **Logistic Regression Baseline (`LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')`)**: A regularized linear baseline trained on standardized features.

---

## 6. Empirical Evaluation, Benchmarks & Calibration Audit

### 6.1 Performance Comparison on Held-Out Test Set (2011-10-01 to 2011-10-31)
Evaluated across **5,438 customers** on the strictly held-out test snapshot:

| Metric | Dummy Prior Baseline | Logistic Regression Baseline | **Raw XGBoost Model** | **Sigmoid Calibrated XGBoost** | Lift vs. LR Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.5000 | 0.7632 | **0.7750** | **0.7750** | +0.0118 |
| **PR-AUC (Avg Precision)** | 0.2041 | 0.4951 | **0.5180** | **0.5180** | **+0.0229 (+4.6%)** |
| **Precision** | 0.2041 (at base rate) | 0.3838 | **0.4832** | **0.4832** | **+0.0994 (+25.9%)** |
| **Recall** | 1.0000 | 0.6252 | **0.4793** | **0.4793** | Calibrated threshold |
| **F1 Score** | 0.3390 | 0.4757 | **0.4812** | **0.4812** | **+0.0055** |
| **Brier Score (Calibration)** | 0.1625 | 0.1854 | **0.1694** | **0.1348** | **-20.4% error** |
| **Precision @ Top 5%** | 0.2325 | 0.7491 | **0.7749** | **0.7749** | **+0.0258 (3.80x Base)** |
| **Precision @ Top 10%** | 0.2284 | 0.5930 | **0.6022** | **0.6022** | **+0.0092 (2.95x Base)** |

### 6.2 Decision Threshold Policy
- **Optimal Threshold Selection**: The operational classification threshold **$t = 0.5900$** was chosen strictly by sweeping $t \in [0.10, 0.90]$ with step $0.01$ on the **Validation Set** to maximize $F_1$ ($F_1 = 0.5217$).
- **Held-Out Generalization**: Applying $t = 0.5900$ to the test partition produced $F_1 = 0.4812$, confirming zero threshold data snooping.

### 6.3 Test Set Confusion Matrix (at $t = 0.5900$)
Across 5,438 test customer snapshots:
- **True Negatives (TN)**: **3,759** (Correctly identified non-repeat customers)
- **False Positives (FP)**: **569** (Non-repeat customers flagged as repeat)
- **False Negatives (FN)**: **578** (Repeat customers missed)
- **True Positives (TP)**: **532** (Correctly identified repeat customers)
- **Total Test Records**: $3,759 + 569 + 578 + 532 = \mathbf{5,438}$.

### 6.4 Probability Calibration Audit & Reliability Findings
#### Root Cause Analysis of Raw Brier Score (0.1694)
- Training with `scale_pos_weight = 4.081` penalizes false negatives $4\times$ more heavily, shifting raw predicted probabilities upward.
- On the test set, the true positive rate is **20.41%**, but the raw XGBoost model's mean predicted probability is **37.85%**.
- In contrast, the dummy prior baseline predicts a constant $p \approx 0.1968$ close to the base rate, yielding a naive Brier score of **0.1625**.
- Describing raw probabilities from a weighted tree model as "well-calibrated" is technically inaccurate without post-hoc adjustment.

#### Post-Hoc Calibration on Validation Data Only
To establish true calibration, post-hoc calibrators were fitted strictly on validation data:
1. **Sigmoid Calibration (Platt Scaling)**:
   - Validation Brier improves from $0.1551 \rightarrow \mathbf{0.0987}$.
   - Out-of-sample Test Brier improves from $0.1694 \rightarrow \mathbf{0.1348}$ (**20.4% lower quadratic error**).
   - Test ROC-AUC is strictly preserved at **0.7750** and PR-AUC at **0.5180**.
2. **Isotonic Calibration**:
   - Validation Brier improves from $0.1551 \rightarrow \mathbf{0.0969}$.
   - Out-of-sample Test Brier improves to **0.1348**; slight step-function discretization (Test ROC-AUC: 0.7731, PR-AUC: 0.5011).
3. **Verdict**: Sigmoid (Platt) calibration provides optimal monotonic probability scaling, outperforming both the uncalibrated model and the dummy prior baseline while maintaining peak discriminative ranking.
- **Reliability Diagram**: Saved to [`reports/phase6/calibration_reliability.png`](file:///c:/Users/himas/Downloads/E-Commerce%20Intelligence%20Platform/reports/phase6/calibration_reliability.png).

### 6.5 Subgroup Performance Analysis by Customer Activity
Model performance was evaluated across customer activity tiers on the test snapshot:

| Customer Activity Tier | Customer Count | Actual Repeat Rate | ROC-AUC | PR-AUC | Operational Characteristic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Activity (1–2 orders)** | 2,545 | 9.71% | **0.7282** | 0.2261 | Strong discrimination despite sparse transaction histories; captures dormant vs. single-purchase churn. |
| **Medium Activity (3–5 orders)** | 1,339 | 19.12% | **0.6329** | 0.3115 | Variable purchase cycles; relies heavily on recency and 60d/90d velocity. |
| **High Activity (6+ orders)** | 1,554 | 38.99% | **0.7318** | **0.6586** | Exceptional ranking power (PR-AUC 0.6586); cleanly segments active repeaters from core accounts. |

---

## 7. Model Interpretability & Feature Attributions

### 7.1 XGBoost Feature Importance Breakdown
Feature importance was analyzed across three standard tree metrics:

| Rank | Feature | Gain (Relative Information) | Weight (Frequency in Trees) | Cover (Relative Sample Influence) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `purchase_frequency` | **317.17** | 97 | 0.0827 |
| 2 | `purchases_last_90d` | **192.19** | 68 | 0.0841 |
| 3 | `net_monetary_value` | **154.91** | 102 | 0.0763 |
| 4 | `spend_last_90d` | **144.31** | 64 | 0.0610 |
| 5 | `avg_order_interval_days`| **129.14** | 76 | 0.0573 |
| 6 | `recency_days` | **109.13** | 88 | 0.0811 |
| 7 | `purchases_last_30d` | **104.75** | 67 | 0.0610 |
| 8 | `spend_last_30d` | **90.49** | 71 | 0.0526 |
| 9 | `product_diversity` | **85.04** | 84 | 0.0519 |
| 10 | `avg_order_value` | **73.22** | 90 | 0.0463 |

### 7.2 SHAP (SHapley Additive exPlanations) Global Insights
Using TreeSHAP on validation customer samples:
1. **Short-Term Cadence (`purchases_last_90d`, `purchases_last_30d`)**: High recent frequency strongly drives positive log-odds of immediate 30-day repeat purchase.
2. **Inter-Order Velocity (`avg_order_interval_days`)**: Customers with tight average purchase cycles (<14 days) exhibit substantial positive SHAP attributions; longer intervals (>60 days) contribute negatively.
3. **Monetary Weight (`net_monetary_value`, `spend_last_90d`)**: Higher spending amplifies repeat probability, but only when accompanied by recent order cadence.
4. **Recency Non-Linearity**: Low `recency_days` (<15 days) has a strong positive impact, which decays exponentially after 45 days.

---

## 8. Persisted Artifacts & Schemas

All model and evaluation artifacts are stored in `models/repeat_purchase/`:
- `xgb_model.joblib`: Serialized trained XGBoost booster (`RepeatPurchasePredictor`).
- `preprocessor.joblib`: Fitted `RepeatPurchaseFeaturePipeline` (StandardScaler).
- `logistic_regression_baseline.joblib`: Serialized Logistic Regression baseline model.
- `feature_schema.json`: Complete 18-feature name and ordering definition.
- `threshold_policy.json`: Selected decision threshold ($t = 0.5900$) and optimization record.
- `training_metadata.json`: Dataset cutoffs, population counts, hyperparameter specs (`scale_pos_weight = 4.081`).
- `evaluation_metrics.json`: Full benchmark metrics across Train, Validation, and Test.
- `feature_importances.json`: Gain, Weight, and Cover metrics for all features.

Visualizations:
- `reports/phase6/feature_importance.png`: 3-panel bar chart comparing Gain, Weight, and Cover.
- `reports/phase6/shap_summary.png`: SHAP beeswarm summary distribution.
- `reports/phase6/calibration_reliability.png`: Reliability diagram comparing Raw, Sigmoid, and Isotonic calibration.

---

## 9. PostgreSQL Schema & Serving Integration

### 9.1 Relational Schema Definition
The repeat purchase predictions are persisted in PostgreSQL via SQLAlchemy ORM (`CustomerRepeatPurchasePrediction`):

```sql
CREATE TABLE IF NOT EXISTS customer_repeat_purchase_prediction (
    prediction_id VARCHAR(64) PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customer_rfm_analytics(customer_id) ON DELETE CASCADE,
    prediction_as_of_date DATE NOT NULL,
    prediction_horizon_days INTEGER NOT NULL DEFAULT 30,
    repeat_purchase_probability NUMERIC(8, 6) NOT NULL,
    predicted_label BOOLEAN NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    feature_snapshot JSONB,
    scoring_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_crpp_cust_date_version UNIQUE (customer_id, prediction_as_of_date, model_version)
);

CREATE INDEX IF NOT EXISTS ix_crpp_customer_id ON customer_repeat_purchase_prediction (customer_id);
CREATE INDEX IF NOT EXISTS ix_crpp_as_of_date ON customer_repeat_purchase_prediction (prediction_as_of_date);
CREATE INDEX IF NOT EXISTS ix_crpp_predicted_label ON customer_repeat_purchase_prediction (predicted_label);
CREATE INDEX IF NOT EXISTS ix_crpp_prob ON customer_repeat_purchase_prediction (repeat_purchase_probability);
```

### 9.2 Loading & Upserting Contract
- **Idempotency**: Implemented via PostgreSQL `ON CONFLICT (prediction_id) DO UPDATE` or composite uniqueness. Re-running `load_repeat_purchase_postgres.py` preserves exact row counts with zero inflation.
- **Feature Snapshot Storage**: The 18 point-in-time features used to generate the score are stored in a `JSONB` column, ensuring complete auditability and reproducibility.
- **Transactional Rollback**: Any failure during batch loading rolls back the entire session without partial corruption.
- **Batch Loader**: Available via `src/database/loader.py` and CLI script `scripts/load_repeat_purchase_postgres.py`.

---

## 10. Automated Testing & Verification Suite

The pipeline is verified by a dedicated test suite in `tests/test_repeat_purchase.py` alongside the existing platform test suite:

| # | Test Function | Verified Property |
| :--- | :--- | :--- |
| 1 | `test_future_label_construction` | Valid purchases generate $Y=1$; cancellations do not count; empty windows generate $Y=0$. |
| 2 | `test_exact_boundary_timestamps_at_cutoff_and_horizon` | Exact timestamps at cutoff and 30d horizon are correctly captured in $[T_{\text{obs}}, T_{\text{obs}} + 30\text{d}]$. |
| 3 | `test_strict_point_in_time_feature_calculations` | Feature extraction strictly uses transactions $< T_{\text{obs}}$. |
| 4 | `test_chronological_perturbation_invariance` | Appending future transactions has zero effect on historical features ($t < T_{\text{obs}}$). |
| 5 | `test_chronological_splitting_no_label_overlap` | Enforces quarterly spacing and 31-day non-overlapping purging buffers between splits. |
| 6 | `test_customer_snapshot_uniqueness` | Composite key `(customer_id, snapshot_date)` is strictly unique across all splits. |
| 7 | `test_deterministic_test_metric_recomputation` | Deterministically recomputes ROC-AUC, PR-AUC, F1, Brier, and exact CM: TN=3759, FP=569, FN=578, TP=532. |
| 8 | `test_repeat_purchase_model_contract` | Probabilities bounded in $[0, 1]$; Precision @ Top 5% $\ge 0.70$ ($3\times$ baseline lift). |
| 9 | `test_threshold_selection_on_validation_only` | Validates threshold policy derivation on validation split ($t = 0.5900$). |
| 10 | `test_model_serialization_and_inference_consistency` | Deserialized model from disk reproduces probability predictions with numerical parity ($\text{tol} < 10^{-3}$). |
| 11 | `test_database_schema_contract` | Validates SQLAlchemy ORM columns, primary keys, and composite uniqueness constraint. |
| 12 | `test_repeat_purchase_postgres_foreign_key_and_jsonb_roundtrip` | Validates customer foreign key constraint and JSONB feature snapshot roundtrip in live PostgreSQL. |
| 13 | `test_repeat_purchase_postgres_idempotent_upsert` | Proves idempotent batch upserting without row inflation in live PostgreSQL. |
| 14 | `test_repeat_purchase_postgres_rollback_on_failed_load` | Proves transaction atomicity and rollback on constraint failure in live PostgreSQL. |
| 15 | `test_prediction_count_reconciliation` | Reconciles Gold test parquet with 5,438 rows, 1,110 positives (20.41%). |

### Full Test Suite Run (with live PostgreSQL)
```
collected 109 items
tests/test_anomaly_database.py ..                                        [  1%]
tests/test_anomaly_detector.py .......                                   [  8%]
tests/test_anomaly_features.py .....                                     [ 12%]
tests/test_anomaly_qa_audit.py .....                                     [ 17%]
tests/test_anomaly_reconciliation.py .....                               [ 22%]
tests/test_database.py .............                                     [ 33%]
tests/test_features.py .......                                           [ 40%]
tests/test_gold.py .....                                                 [ 44%]
tests/test_ingestion_pipeline.py .                                       [ 45%]
tests/test_reader.py ....                                                [ 49%]
tests/test_reconciliation.py .....                                       [ 54%]
tests/test_repeat_purchase.py ...............                            [ 67%]
tests/test_schema.py .....                                               [ 72%]
tests/test_segmentation.py ........                                      [ 79%]
tests/test_segmentation_audit.py .......                                 [ 86%]
tests/test_segmentation_database.py .....                                [ 90%]
tests/test_silver.py ........                                            [ 98%]
tests/test_writer.py ..                                                  [100%]

======================= 109 passed in 158.68s =======================
```
- **109 tests collected**.
- **109 passed (100% pass rate, 0 failed, 0 skipped, 0 warnings)**.

---

## 11. Production Limitations & Operational Governance

1. **Non-Causality Warning**: The model predicts baseline repeat propensity under observed marketing conditions. A customer with a high predicted probability ($P > 0.80$) would likely repeat purchase without any intervention; targeting them does not guarantee incremental lift.
2. **Wholesale vs. Retail Cadence Variance**: Customers in wholesale segments purchase in large, infrequent batches (60–120 day cycles). A 30-day window naturally flags them as negative instances even if they are active long-term customers. Separate window definitions (e.g., 60d or 90d) should be considered for wholesale accounts.
3. **Macroeconomic Seasonality (Q4 Holiday Peak)**: The held-out test snapshot (`2011-10-01`) covers October, leading into the peak November/December holiday season. Re-scoring snapshots in off-peak periods (e.g., February/March) should monitor for distribution shifts in frequency features.
4. **Retraining Policy**: The XGBoost model should be retrained quarterly using rolling chronological snapshot windows with automated drift monitoring on feature distributions (KS test) and calibration curves.
