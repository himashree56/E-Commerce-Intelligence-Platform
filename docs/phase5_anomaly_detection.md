# Phase 5: Transaction Anomaly Detection with Isolation Forest

## 1. Executive Summary & Business Objectives

Phase 5 introduces an enterprise-grade, unsupervised machine learning anomaly detection system to the E-Commerce Intelligence Platform. Leveraging an ensemble of isolation trees (**Isolation Forest**), the pipeline identifies multivariate anomalous transactions across 1,067,371 historical transaction lines while preserving the integrity of commercial operations (such as normal retail returns and high-volume wholesale orders).

### Key Architectural Commitments
1. **Financial Non-Interference**: Historical revenue calculations are strictly protected. Anomaly detection is a diagnostic and governance layer; it does not alter, drop, or mutate financial figures.
2. **Deterministic Lineage**: Every transaction receives a globally unique, deterministic lineage key (`TXN_{source_row_id:07d}`) anchored to its physical row position and dataset provenance, preventing collision issues associated with business hash keys.
3. **Preservation of Retail Operations**: Cancellations and returns are recognized as legitimate retail operations rather than treated unconditionally as fraud. Only 4.43% of cancellations exhibiting extreme characteristics are flagged.
4. **Diagnostic Interpretability**: Every flagged anomaly is mapped to human-interpretable diagnostic reason codes explaining *why* the transaction was isolated.
5. **Leakage-Safe Feature Pipeline**: Features are computed strictly using historical and contextual data without temporal leakage.

---

## 2. Silver Dataset Audit & Training Population

### 2.1 Full Silver Population Breakdown (1,067,371 Records)
The complete Silver dataset was audited across transaction statuses, price tiers, and data quality flags:

| Transaction Status | Record Count | % of Silver | Training Eligibility | Inclusion Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **VALID_PURCHASE** | 1,007,911 | 94.43% | **Eligible** | Core commercial operations representing baseline retail behavior. |
| **CANCELLATION** | 19,104 | 1.79% | **Eligible** | Normal return activity; model must learn legitimate return distributions. |
| **DUPLICATE** | 34,337 | 3.22% | **Excluded** | Identical re-transmitted lines distort local tree isolation density. |
| **INVENTORY_WRITE_OFF** | 3,393 | 0.32% | **Excluded** | Non-commercial stock adjustments (manual notes, damaged stock, lost goods). |
| **ZERO_PRICE_ENTRY** | 2,626 | 0.25% | **Excluded** | Non-commercial administrative entries (£0.00 accounting entries). |
| **Total Silver** | **1,067,371** | **100.00%** | **1,027,015 Trained / 40,356 Excluded** | **Complete population preserved and scored.** |

### 2.2 Audit of Ambiguous Cancellation C496350
- **Source Sheet**: `Year 2009-2010` (Row 76799 in raw workbook)
- **Invoice**: `C496350` (Stock Code: `M`, Description: `Manual`)
- **Quantity**: `+1` (Positive quantity despite cancellation prefix `C`)
- **Unit Price**: `£373.57` (Net Revenue: `-£373.57`)
- **Customer ID**: `None` (`NaN` / unregistered guest checkout in United Kingdom)
- **Status in Silver**: `CANCELLATION` (`is_ambiguous_cancellation = True`)
- **Phase 5 Scoring Result**: Raw Decision Score `-0.085739` $\rightarrow$ Anomaly Score **`+0.085739`** (`is_anomaly = True`).
- **Diagnostic Reasons**: `EXTREME_LINE_VALUE;HIGH_UNIT_PRICE;UNUSUAL_PRODUCT_PRICE;NEW_CUSTOMER_HIGH_VALUE`.
- **Verdict**: Successfully identified as a high-risk manual credit adjustment.
- **Audit Clarification**: Previously cited narrative values of `£1,354.18` and Customer ID `16321` were identified during QA auditing as documentation transcription errors; the true underlying raw workbook, Bronze, Silver, and Gold Parquet values are strictly verified at `£373.57` and `Customer ID: <NA>`. Source financial values were preserved without modification.

---

## 3. Detection Unit & Deterministic Lineage

### 3.1 Detection Unit Definition
- **Primary Detection Unit**: **Invoice-Line Level**. Each individual record in the Silver table corresponds to an item purchased or returned within an invoice.
- **Distinction from Whole-Invoice & Customer Fraud**: An anomalous line (e.g. an extreme single-item adjustment) does not necessarily imply that the entire cart or the associated customer is fraudulent. The system scores each line independently while aggregating customer and invoice context safely.

### 3.2 Deterministic Lineage Identifier
To preserve total auditability, deterministic identifiers were generated using physical row indices:
```python
transaction_id = f"TXN_{source_row_id:07d}"
```
This guarantees:
- 100% collision-free primary keys across all 1,067,371 rows.
- Complete bi-directional mapping back to Silver source parquet files and PostgreSQL tables.

---

## 4. Leakage-Safe Feature Engineering Pipeline

The `TransactionFeaturePipeline` extracts 10 numerical features covering scale, pricing relative to catalog medians, customer ordering maturity, and cyclical temporal patterns:

| # | Feature Name | Formula / Derivation | Transformation | Purpose & Leakage Safeguard |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `log_abs_quantity` | $\log(1 + |\text{quantity}|)$ | Natural Log (`log1p`) | Dampens extreme wholesale quantities (up to 80,995). |
| 2 | `log_unit_price` | $\log(1 + \max(0, \text{unit\_price}))$ | Natural Log (`log1p`) | Normalizes heavy right tail of unit prices (up to £38,970). |
| 3 | `log_abs_line_total` | $\log(1 + |\text{quantity} \times \text{unit\_price}|)$ | Natural Log (`log1p`) | Captures monetary magnitude of individual line item. |
| 4 | `product_price_deviation` | $|\log(1 + \text{unit\_price}) - \log(1 + \text{median\_price})|$ | Relative log absolute difference | Detects price tampering or abnormal discounting relative to learned product catalog median. |
| 5 | `log_prior_customer_orders` | $\log(1 + \text{prior\_invoices})$ | Sequential temporal cumcount | Strictly counts prior distinct invoices per customer; lines of the same invoice share the same count. |
| 6 | `is_missing_customer` | $1.0 \text{ if customer\_id is null else } 0.0$ | Binary indicator | Flags guest/unregistered checkouts without distorting numerical distributions. |
| 7 | `invoice_hour_sin` | $\sin(2\pi \times \text{hour} / 24)$ | Cyclical trigonometric | Captures time-of-day smoothness (continuity across midnight). |
| 8 | `invoice_hour_cos` | $\cos(2\pi \times \text{hour} / 24)$ | Cyclical trigonometric | Captures time-of-day smoothness. |
| 9 | `invoice_day_sin` | $\sin(2\pi \times \text{day} / 7)$ | Cyclical trigonometric | Day-of-week periodicity (0 = Monday, 6 = Sunday). |
| 10 | `invoice_day_cos` | $\cos(2\pi \times \text{day} / 7)$ | Cyclical trigonometric | Day-of-week periodicity. |

---

## 5. Model Architecture & Hyperparameters

### 5.1 Model Selection & Configuration
The Isolation Forest was implemented via `sklearn.ensemble.IsolationForest` using production-calibrated parameters:

```json
{
  "model_version": "v1.0.0",
  "n_estimators": 100,
  "max_samples": 256,
  "contamination": 0.01,
  "random_state": 42,
  "n_jobs": -1
}
```

### 5.2 Decision Boundary & Score Direction Policy
Scikit-learn's Isolation Forest implements:
$$\text{decision\_function}(X) = \text{score\_samples}(X) - \text{offset\_}$$
where `offset_` is set such that exactly `contamination` proportion of training records have $\text{decision\_function}(X) < 0.0$.

To align with intuitive business reporting where higher scores denote higher risk:
$$\text{anomaly\_score} = - \text{raw\_decision\_score}$$
$$\text{is\_anomaly} = \text{raw\_decision\_score} < 0.000000 \quad (\iff \text{anomaly\_score} > 0.000000)$$

---

## 6. Model Evaluation, Sensitivity & Stability

### 6.1 Contamination Parameter Sensitivity
The model was evaluated across 4 contamination thresholds on the 1,027,015 training set:

| Contamination Rate | Offset Threshold | Flagged Training Count | Flagged Training % | Operational Implication |
| :--- | :--- | :--- | :--- | :--- |
| **0.5%** | `-0.017486` | 5,135 | 0.50% | High precision, flags only extreme univariate and multivariate anomalies. |
| **1.0% (Selected)** | `0.000000` | **10,271** | **1.00%** | Optimal balance between alert volume and detection sensitivity. |
| **2.0%** | `+0.017990` | 20,541 | 2.00% | Broader review threshold capturing moderate pricing deviations. |
| **5.0%** | `+0.040946` | 51,322 | 5.00% | High recall, includes off-hours and low-frequency product lines. |

### 6.2 Multi-Seed Stability Experiment
Model stability was evaluated by training across 4 random seeds (`42, 123, 456, 789`):
- **Spearman Rank Correlation ($\rho$)**: Min `0.9546`, Mean **`0.9593`**, Max `0.9673`.
  - Proves that transaction risk ranking is exceptionally stable across tree forest variations.
- **Top 1% Jaccard Similarity**: Min `0.5480`, Mean **`0.5772`**, Max `0.6474`.
  - Reflects normal boundary jitter for marginal records near the 99th percentile cutoff.

### 6.3 Synthetic Diagnostic Injection Test
Five synthetic edge cases were injected into the trained detector to evaluate sensitivity:
1. Wholesale extreme quantity (50,000 units, £250.00 unit price): **Detected** (`score: 0.1015`)
2. Massive single unit price (£1,000.00 for £2.00 catalog item): **Detected** (`score: 0.0799`)
3. Massive cancellation (-30,000 units, £500.00 unit price): **Detected** (`score: 0.1559`)
4. Midnight bulk purchase (10,000 units at 03:00 AM): **Detected** (`score: 0.1395`)
5. Huge single charge (£5,000.00 unit price): **Detected** (`score: 0.0822`)
- **Synthetic Detection Rate**: **5 / 5 (100.0%)**.

---

## 7. Diagnostic Rule Explainer & Reason Codes

Unsupervised isolation tree scores provide a continuous outlier ranking but do not explain *why* an observation is isolated. To satisfy business explainability requirements, the `DiagnosticRuleExplainer` assigns one or more deterministic reason codes:

| Diagnostic Reason Code | Trigger Condition |
| :--- | :--- |
| `EXTREME_QUANTITY` | $|\text{quantity}| \ge 108.0$ (99th percentile of training data). |
| `EXTREME_LINE_VALUE` | $|\text{line\_total}| \ge £199.20$ (99th percentile of training data). |
| `HIGH_UNIT_PRICE` | $\text{unit\_price} \ge £18.68$ (99th percentile of training data). |
| `UNUSUAL_PRODUCT_PRICE` | Unit price deviates from catalog median by $> 1.1$ log units ($\approx 3\times$). |
| `OFF_HOURS_TRANSACTION` | Invoice hour $< 07:00$ or $\ge 20:00$. |
| `NEW_CUSTOMER_HIGH_VALUE` | Customer has 0 prior orders AND $|\text{line\_total}| \ge £61.20$ (95th percentile). |
| `EXACT_DUPLICATE_ROW` | Re-transmitted identical line marked `is_duplicate = True`. |
| `ADMINISTRATIVE_ADJUSTMENT` | Record has non-commercial status (`INVENTORY_WRITE_OFF` or `ZERO_PRICE_ENTRY`). |
| `MULTIVARIATE_ISOLATION_OUTLIER` | Anomaly score $> 0.0$ but no single univariate heuristic threshold was breached. |

---

## 8. Full Population Scoring & Gold Dataset Parity

The full Silver dataset (1,067,371 rows) was scored and persisted to `data/gold/gold_transaction_anomalies_isolation_forest.parquet` (25.10 MB).

### 8.1 Scoring Results by Status

| Transaction Status | Total Records | Anomalies Flagged | Anomaly Rate | Core Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **VALID_PURCHASE** | 1,007,911 | 9,424 | **0.94%** | Baseline commercial purchases exhibiting extreme order sizes or price anomalies. |
| **CANCELLATION** | 19,104 | 847 | **4.43%** | Retail return operations preserved; only extreme cancellations flagged. |
| **DUPLICATE** | 34,337 | 404 | **1.18%** | Duplicate lines scored independently; 404 flagged by tree isolation. |
| **INVENTORY_WRITE_OFF** | 3,393 | 3,393 | **100.00%** | Administrative stock adjustments correctly isolated from commercial sales. |
| **ZERO_PRICE_ENTRY** | 2,626 | 2,596 | **98.86%** | Non-commercial £0.00 accounting entries flagged. |
| **Total Population** | **1,067,371** | **16,664** | **1.56%** | **Overall anomaly rate is 1.56%, fully auditable.** |

### 8.2 Top 10 Anomalous Transactions in Silver Dataset

| Transaction ID | Invoice No | Customer ID | Quantity | Unit Price | Line Total | Anomaly Score | Diagnostic Reasons |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TXN_0587085` | `541431` | `12346` | +74,215 | £1.04 | £77,183.60 | `+0.165604` | `EXTREME_QUANTITY;EXTREME_LINE_VALUE;NEW_CUSTOMER_HIGH_VALUE` |
| `TXN_0587088` | `C541433` | `12346` | -74,215 | £1.04 | -£77,183.60 | `+0.165604` | `EXTREME_QUANTITY;EXTREME_LINE_VALUE` |
| `TXN_1065882` | `581483` | `16446` | +80,995 | £2.08 | £168,469.60 | `+0.165412` | `EXTREME_QUANTITY;EXTREME_LINE_VALUE` |
| `TXN_1065883` | `C581484` | `16446` | -80,995 | £2.08 | -£168,469.60 | `+0.165412` | `EXTREME_QUANTITY;EXTREME_LINE_VALUE` |
| `TXN_0607682` | `C543037` | `17940` | -1 | £4,287.63 | -£4,287.63 | `+0.119782` | `EXTREME_LINE_VALUE;HIGH_UNIT_PRICE;UNUSUAL_PRODUCT_PRICE` |
| `TXN_0677598` | `C549452` | `17940` | -1 | £4,265.73 | -£4,265.73 | `+0.119741` | `EXTREME_LINE_VALUE;HIGH_UNIT_PRICE;UNUSUAL_PRODUCT_PRICE` |
| `TXN_0411333` | `C527690` | `17940` | -1 | £4,189.38 | -£4,189.38 | `+0.119641` | `EXTREME_LINE_VALUE;HIGH_UNIT_PRICE;UNUSUAL_PRODUCT_PRICE` |
| `TXN_0579133` | `C540848` | `17940` | -1 | £3,837.45 | -£3,837.45 | `+0.119102` | `EXTREME_LINE_VALUE;HIGH_UNIT_PRICE;UNUSUAL_PRODUCT_PRICE` |
| `TXN_0680193` | `C549738` | `17940` | -1 | £3,767.12 | -£3,767.12 | `+0.119001` | `EXTREME_LINE_VALUE;HIGH_UNIT_PRICE;UNUSUAL_PRODUCT_PRICE` |
| `TXN_0749007` | `C555465` | `17940` | -1 | £3,649.34 | -£3,649.34 | `+0.118819` | `EXTREME_LINE_VALUE;HIGH_UNIT_PRICE;UNUSUAL_PRODUCT_PRICE` |

---

## 9. PostgreSQL Analytics Layer Integration

The PostgreSQL schema was extended with the `gold_transaction_anomalies_isolation_forest` table:

```sql
CREATE TABLE gold_transaction_anomalies_isolation_forest (
    transaction_id VARCHAR(32) PRIMARY KEY,
    source_row_id INTEGER NOT NULL UNIQUE,
    invoice_no VARCHAR(20) NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    description TEXT,
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(12, 4) NOT NULL,
    line_total NUMERIC(14, 4) NOT NULL,
    invoice_date TIMESTAMP NOT NULL,
    customer_id INTEGER,
    country VARCHAR(64) NOT NULL,
    transaction_status VARCHAR(32) NOT NULL,
    raw_decision_score NUMERIC(10, 6) NOT NULL,
    anomaly_score NUMERIC(10, 6) NOT NULL,
    is_anomaly BOOLEAN NOT NULL,
    diagnostic_reasons TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX ix_anomalies_is_anomaly ON gold_transaction_anomalies_isolation_forest (is_anomaly);
CREATE INDEX ix_anomalies_customer_id ON gold_transaction_anomalies_isolation_forest (customer_id);
CREATE INDEX ix_anomalies_invoice_no ON gold_transaction_anomalies_isolation_forest (invoice_no);
CREATE INDEX ix_anomalies_invoice_date ON gold_transaction_anomalies_isolation_forest (invoice_date);
```

---

## 10. Operational Guidelines, Caveats & Limitations

1. **Unsupervised Nature**: Isolation Forest flags statistical density outliers, NOT verified legal fraud. Scores should feed an operational alert queue for Human-in-the-Loop (HITL) investigation.
2. **Wholesale vs. Fraudulent Outliers**: Large commercial buyers (e.g. customer 16446 with 80,995 units) produce extreme line values. While correctly isolated statistically, business verification confirms legitimate B2B activity.
3. **Cancellation Symmetry**: Massive paired purchases and credit notes (e.g. invoice `541431` and cancellation `C541433`) receive identical high anomaly scores, representing operational error correction rather than credit theft.
4. **Model Retraining Schedule**: Retraining is recommended quarterly or when catalog pricing structures shift by $> 20\%$.
