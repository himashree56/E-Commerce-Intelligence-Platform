# Phase 7 — Production-Grade Interactive Streamlit Analytics Dashboard

## 1. Executive Summary

Phase 7 completes the delivery of the **E-Commerce Intelligence Platform** by providing an enterprise-grade interactive analytics and machine learning governance dashboard built with Streamlit and Plotly. The application unifies all previous project phases into an operational single pane of glass:
- **Phase 1 & 2**: Bronze ingestion and Silver/Gold medallion data transformations across 1,067,371 transaction lines.
- **Phase 3**: PostgreSQL 16 relational analytics lakehouse and zero-penny commercial ledger reconciliation (£19,012,910.85 net revenue).
- **Phase 4**: Customer RFM segmentation with unsupervised K-Means ($k=5$).
- **Phase 5**: Transaction anomaly detection with unsupervised Isolation Forest (16,664 statistical outliers).
- **Phase 6**: Supervised customer repeat-purchase prediction with XGBoost and Platt Sigmoid calibration ($t=0.27$ calibrated / $t=0.59$ raw).

The dashboard enforces a **Zero-Hardcoding Guarantee**: all displayed metrics, charts, tables, and distributions are dynamically queried from live PostgreSQL 16 database tables with automatic, seamless fallback to serialized Gold Parquet files and joblib models whenever database connectivity is unavailable.

---

## 2. Architecture & File Structure

```
src/dashboard/
├── __init__.py                     # Package initialization
├── config.py                       # App metadata, palette, formatters, cluster mappings
├── data_provider.py                # Dual PostgreSQL 16 & Gold Parquet query engine
├── app.py                          # Streamlit application entrypoint & router
├── components/                     # Reusable UI widgets
│   ├── __init__.py
│   ├── kpi_cards.py                # Single & multi-column metric cards
│   ├── charts.py                   # Responsive Plotly chart generators
│   └── tables.py                   # Dataframe displays with CSV exports
└── pages/                          # Modular domain pages
    ├── __init__.py
    ├── executive_overview.py       # Page 1: Top-level commercial & portfolio health
    ├── sales_analytics.py          # Page 2: Monthly trends, product velocity, returns
    ├── customer_segmentation.py    # Page 3: K-Means clusters, profiles, RFM scatter
    ├── anomaly_detection.py        # Page 4: Isolation Forest outliers & diagnostic reasons
    ├── repeat_purchase.py          # Page 5: XGBoost predictions & calibration policy
    ├── data_quality.py             # Page 6: Zero-penny balance & Medallion inventory
    └── model_governance.py         # Page 7: Model cards, activity tiers, AI ethics

scripts/
└── run_dashboard.py                # Standalone CLI launcher script
tests/
└── test_dashboard.py               # 31 automated unit & integration tests
```

---

## 3. Verified Financial & Commercial Baselines

Every displayed commercial metric strictly preserves the audited Phase 1–6 financial ledger:

| Commercial Metric | Audited Ledger Baseline | Live PostgreSQL Query | Parquet Fallback | Parity Discrepancy |
| :--- | :---: | :---: | :---: | :---: |
| **Gross Sales** | £20,475,708.60 | £20,475,708.60 | £20,475,708.60 | **£0.00** |
| **Customer Returns** | £1,462,797.75 | £1,462,797.75 | £1,462,797.75 | **£0.00** |
| **Net Commercial Revenue** | £19,012,910.85 | £19,012,910.85 | £19,012,910.85 | **£0.00** |
| **Silver Transaction Rows** | 1,067,371 | 1,067,371 | 1,067,371 | **0 rows** |
| **Valid Commercial Purchases** | 1,007,911 | 1,007,911 | 1,007,911 | **0 rows** |
| **Identified RFM Customers** | 5,939 | 5,939 | 5,939 | **0 accounts** |
| **Commercial Purchase Orders** | 44,870 | 44,870 | 44,870 | **0 orders** |
| **Average Order Value (AOV)** | £423.73 | £423.73 | £423.73 | **£0.00** |
| **Flagged Outlier Transactions** | 16,664 | 16,664 | 16,664 | **0 rows** |
| **Held-Out Test Predictions** | 5,438 | 5,438 | 5,438 | **0 rows** |

---

## 4. Detailed Page Specifications

### Page 1 — Executive Overview
- **Header & Connectivity**: Displays live indicator badge (`🟢 Connected: PostgreSQL 16 (Live)` or `🟡 Active: Parquet Fallback Mode`).
- **Primary KPIs**: Net Revenue (£19,012,910.85), Gross Sales (£20,475,708.60), Customer Returns (£1,462,797.75, 7.14%), AOV (£423.73).
- **Operational Volumes**: 1,067,371 Silver lines, 1,007,911 valid purchases, 44,870 purchase orders, 5,939 RFM customers.
- **Charts**:
  - 25-Month Commercial Revenue Trend (December 2009 to December 2011) combining Gross Sales, Returns, and Net Revenue.
  - Top 10 International Geographic Markets (United Kingdom: £16.39M, EIRE: £614.9K, Netherlands: £548.5K, Germany: £417.9K, France: £328.2K).
  - Top 10 Commercial Products by Net Revenue.

### Page 2 — Sales & Revenue Analytics
- **Interactive Controls**: Dual-slider date range selector (2009-12 through 2011-12) and country filter dropdown.
- **Dynamic Aggregation**: Dynamically sums period gross, returns, net, and calculates period cancellation rate.
- **Charts**: Dual-axis monthly revenue trend and monthly order volume (valid vs. cancelled orders).
- **Product Velocity Table**: Interactive grid with Gross Sales, Returns, Net Revenue, Gross Units, Units Returned, Net Units, and Distinct Customers with CSV export (`top_products_performance.csv`).

### Page 3 — Customer RFM Segmentation
- **Descriptive Non-Causal Policy**: Explicit disclaimer clarifying that cluster archetypes represent historical descriptive patterns, not deterministic future guarantees.
- **Cluster Profile Cards**:
  - **Cluster 0 (High-Value Loyal Champions)**: 1,079 customers (18.2%), £11,437,580.80 revenue (70.2%), Avg spend £10,600.17
  - **Cluster 1 (Active Recent Shoppers)**: 1,313 customers (22.1%), £1,129,609.12 revenue (6.9%), Avg spend £860.33
  - **Cluster 2 (Lapsed High-Potential)**: 1,363 customers (22.9%), £2,994,506.26 revenue (18.4%), Avg spend £2,197.00
  - **Cluster 3 (Lost Low-Value Buyers)**: 2,083 customers (35.1%), £820,379.99 revenue (5.0%), Avg spend £393.85
  - **Cluster 4 (Net Returners & Adjusters)**: 101 customers (1.7%), -£92,636.75 revenue (-0.6%), Avg spend -£917.20
- **Visualizations**: Cluster population and revenue contribution bar charts; 3D/2D interactive RFM scatter plot with logarithmic frequency scaling and monetary bubble sizing.
- **Reconciliation Cross-Tabulation**: Full cross-tabulation table reconciling rule-based quintile segments against K-Means clusters (summing to exactly 5,939).
- **Customer Lookup**: Interactive search by 5-digit Customer ID with instant retrieval of assigned cluster and RFM metrics.

### Page 4 — Transaction Anomaly Detection
- **Statistical Non-Fraud Disclaimer**: Prominent alert stating that flagged transactions are statistical outliers in high-dimensional feature space, NOT confirmed commercial fraud.
- **KPI Summary**: 1,067,371 scored lines, 16,664 flagged outliers (1.56% overall anomaly rate).
- **Status Breakdown**:
  - `VALID_PURCHASE`: 1,007,911 total, 9,424 anomalies (0.94%)
  - `CANCELLATION`: 19,104 total, 847 anomalies (4.43%)
  - `INVENTORY_WRITE_OFF`: 3,393 total, 3,393 anomalies (100.0%)
  - `ZERO_PRICE_ENTRY`: 2,626 total, 2,596 anomalies (98.86%)
  - `DUPLICATE`: 34,337 total, 404 anomalies (1.18%)
- **Audited Anomaly Explorer**: Interactive filtered table supporting search by Invoice No, Stock Code, or Customer ID, displaying Transaction ID, Invoice Date, Quantity, Unit Price, Line Total, Anomaly Score, and Diagnostic Reason codes (e.g., `EXTREME_UNIT_PRICE`, `ROUND_PRICE_ANOMALY`, `HIGH_QUANTITY_PURCHASE`).

### Page 5 — Repeat-Purchase Prediction
- **Historical Snapshot Notice**: Explicit documentation that predictions evaluate an out-of-time test cutoff at `2011-10-01` predicting 30-day forward re-engagement through `2011-10-31`.
- **Calibration Toggle**: Sidebar radio button switching between **Sigmoid Calibrated (Platt Scaling)** and **Raw XGBoost (Uncalibrated)**.
- **Model Performance Cards**:
  - Test ROC-AUC: **0.7750**
  - Test PR-AUC: **0.5180**
  - Precision @ Top 5%: **77.49%** (210/271 true positives)
  - Precision @ Top 10%: **60.22%** (327/543 true positives)
  - Calibrated Decision Threshold: $t = 0.2700$ (Test $F_1 = 0.4826$, Precision = 48.98%, Recall = 47.57%, Brier Score = 0.1348)
  - Raw Decision Threshold: $t = 0.5900$ (Test $F_1 = 0.4812$, Precision = 48.32%, Recall = 47.93%, Brier Score = 0.1694)
- **Visualizations**:
  - Held-out test ROC curve with chance baseline and Logistic Regression reference.
  - Calibration Reliability Diagram comparing raw vs. calibrated Sigmoid curves against perfect 45-degree calibration.
  - Feature Importance horizontal bar chart selectable by `Gain`, `Weight`, or `Cover`.
- **Prediction Cohort Explorer**: Filterable table by Propensity Tier (High $\ge 0.50$, Medium $[0.20, 0.50)$, Low $< 0.20$) or Customer ID search, displaying unpacked historical features and CSV export (`customer_repeat_predictions.csv`).

### Page 6 — Data Quality & Pipeline Monitoring
- **Zero-Penny Financial Reconciliation**:
  $$\text{Gross Sales (£20,475,708.60)} - \text{Returns (£1,462,797.75)} = \text{Net Revenue (£19,012,910.85)}$$
  $$\text{Ledger Discrepancy} = \mathbf{£0.00}\quad(\text{PERFECT ZERO-PENNY BALANCE})$$
- **Medallion Architecture Layer Inventory**: Detailed table tracking row counts and validation status across Bronze, Silver, Gold, and Scored layers.
- **Data Hygiene Audit**: Summary cards reporting 34,337 duplicate records, 230,808 missing customer accounts, 19,104 cancellations, and 2,626 zero-price entries.
- **ETL Execution Metadata**: Live PostgreSQL table display of execution runs logged in `etl_execution_metadata`.

### Page 7 — Model Performance & Governance
- **Production ML Registry**: Formal model cards for Customer Segmentation (K-Means), Anomaly Detection (Isolation Forest), and Repeat Purchase Prediction (XGBoost + Platt).
- **Subgroup Fairness Breakdown**: Disaggregated test performance across customer historical activity tiers:
  - **Low Activity (1–2 Orders)**: 2,545 accounts, 9.74% base prevalence, ROC-AUC 0.7282, PR-AUC 0.2261.
  - **Medium Activity (3–5 Orders)**: 1,339 accounts, 19.12% base prevalence, ROC-AUC 0.6329, PR-AUC 0.3115.
  - **High Activity (6+ Orders)**: 1,554 accounts, 39.00% base prevalence, ROC-AUC 0.7318, PR-AUC 0.6586, Recall 80.03%.
- **Operational Guidelines & Governance Guardrails**: Clear non-causal directives, human-in-the-loop triggers, and probability calibration enforcement.

---

## 5. Verification & Testing Evidence

### Automated Test Suite
- Comprehensive suite located in [`tests/test_dashboard.py`](file:///c:/Users/himas/Downloads/E-Commerce%20Intelligence%20Platform/tests/test_dashboard.py).
- **Results**: 31 passed in 4.43s.
- **Complete Repository Suite**: 140 passed out of 140 tests across Phases 1 through 7 (100% pass rate, 0 failures, 0 warnings).

### Live Browser Subagent Verification
- Executed via autonomous browser subagent (`conversationId: 555aeec8-6016-424c-878d-4047747f6cd5`) against running instance on `http://localhost:8501`.
- **Host-Verified Evidence**:
  - Visited origin: `http://localhost:8501`
  - Blocked origins: None
  - Evidence Screenshot: `browser_evidence/555aeec8-6016-424c-878d-4047747f6cd5/final_step_83.png`
  - Verification outcome: All 7 domain views confirmed operational with zero exceptions, responsive Plotly charts, and verified financial parity.

---

## 6. How to Run

### Option 1: Standalone CLI Launcher
```bash
python scripts/run_dashboard.py --port 8501
```

### Option 2: Streamlit CLI
```bash
streamlit run src/dashboard/app.py --server.port 8501
```
