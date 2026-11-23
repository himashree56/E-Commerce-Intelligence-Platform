# E-Commerce Intelligence Platform

An enterprise-grade Lakehouse analytics and predictive machine learning platform built on transaction data from the **UCI Online Retail II** dataset.

---

## Architecture: Medallion Lakehouse & Relational Serving

```
[Raw Excel/Zip] ──> [Bronze Parquet] ──> [Silver Parquet/Delta] ──> [Gold Parquet] ──> [PostgreSQL 16] ──> [Predictive ML Layer]
 (Multi-sheet)       (100% Preserved)     (Quality & Status Flags)   (8 Curated Datasets) (Serving Tables)     (K-Means, Isolation Forest, XGBoost)
```

---

## Phases Overview

### Phase 1: Bronze Ingestion Layer
- Raw immutability: Original Excel workbook is read-only.
- Ingests both sheets (`Year 2009-2010` and `Year 2010-2011`).
- Preserves 100% of records, cancellations, negative quantities, and missing customer identifiers.
- Persists Snappy-compressed Parquet datasets in `data/bronze/`.

### Phase 2: Silver & Gold Curated Layer
- **100% Record Preservation**: Zero records deleted (1,067,371 rows).
- **Categorical Status Hierarchy**:
  - `VALID_PURCHASE` (1,007,911 rows)
  - `DUPLICATE` (34,337 rows)
  - `CANCELLATION` (19,104 rows)
  - `INVENTORY_WRITE_OFF` (3,393 rows)
  - `ZERO_PRICE_ENTRY` (2,626 rows)
- **Special Transaction Audit Flags**:
  - `is_ambiguous_cancellation`: Explicit flag isolating credit notes with positive quantities (e.g. invoice `C496350`). Raw quantity (`+1`) and price (`£373.57`) are preserved, treated as a credit deduction to maintain historical revenue stability.
- **Documented Net Revenue Equation**:
  $$\text{Net Revenue} = \text{Gross Sales} - \text{Return Amount}$$
  - Gross Sales (Purchases): **£20,475,708.60**
  - Returns & Cancellations: **-£1,462,797.75**
  - Net Revenue: **£19,012,910.85**
- **Gold Analytical Tables**:
  1. `gold_monthly_sales.parquet` (25 rows)
  2. `gold_customer_rfm.parquet` (5,939 rows)
  3. `gold_product_performance.parquet` (4,920 rows - 100% parity across Pandas & PySpark)
  4. `gold_country_revenue.parquet` (43 rows)
  5. `gold_customer_purchase_history.parquet` (44,870 rows)
- **Engine Parity & Reconciliation**: Both Pandas and native PySpark engines yield identical metric and row counts across all layers. Every single Bronze record is accounted for.

### Phase 3: PostgreSQL Database Integration & Advanced SQL Analytics
- **PostgreSQL 16 Lakehouse Serving Layer**: Hosted via Docker Compose with persistent volume storage (`ecommerce_pgdata`), health checks, and connection pooling (SQLAlchemy 2.0 + psycopg2).
- **Relational Schema Design**:
  - 7 tables with strict primary keys, foreign keys (`customer_purchase_history` -> `customer_rfm_analytics`), and B-tree indexes.
  - Strict `NUMERIC(14, 2)` types for all financial values to prevent floating-point drift.
  - Audit and execution tracking via `etl_execution_metadata` and `data_quality_audit_results`.
- **Idempotent Loading Pipeline**: Batched upsert engine (`ON CONFLICT DO UPDATE`) ensuring zero row inflation on repeated pipeline executions.
- **Advanced SQL Analytics Suite**: 12 production-grade SQL queries with Common Table Expressions (CTEs), window functions (`LAG`, `NTILE`, `DENSE_RANK`), cohort retention matrices, and Pareto 80/20 customer wealth concentration analysis.
- **Full Reconciliation**: 100% row count and revenue parity verified between Gold Parquet source datasets and PostgreSQL tables.

### Phase 4: Customer Segmentation using K-Means
- **100% Customer Population Preserved**: All 5,939 RFM customer records clustered without dropping 81 negative-net spenders (-£92,636.75) or 21 zero-net spenders (£0.00).
- **Symmetric Signed Log Transformation**: $f(x) = \text{sign}(x) \cdot \ln(1 + |x|)$ smoothly compresses monetary outliers and negative refund values while preserving monotonicity and zero invariance.
- **Model Selection & Cluster Rationale**: $k=5$ selected at the global peak Calinski-Harabasz score (4,959.0) with strong silhouette (0.3839) and Davies-Bouldin (0.8103), delivering an actionable CRM taxonomy:
  - Cluster 0: High-Value Loyal Champions (1,079 customers | £11.44M | 70.21% rev)
  - Cluster 1: Active Recent Shoppers (1,313 customers | £1.13M | 6.93% rev)
  - Cluster 2: Lapsed High-Potential (1,363 customers | £2.99M | 18.38% rev)
  - Cluster 3: Lost Low-Value Buyers (2,083 customers | £0.82M | 5.04% rev)
  - Cluster 4: Net Returners & Adjusters (101 customers | -£0.09M | -0.57% rev)
- **High Cluster Stability**: 6-seed stability experiment proves topological robustness with mean ARI = 0.9871 and mean NMI = 0.9784 across seeds `[42, 123, 456, 789, 999, 2024]`.
- **Relational Serving & Parity**: Scored assignments persisted to `customer_segmentation_kmeans` in PostgreSQL with 100% parity across all 5,939 rows.
- **Targeted QA Audit**: Independently confirmed zero-penny reconciliation (£16,289,439.42), exact cross-tabulation row/column totals, and the customer 12558 cluster assignment dynamics.

### Phase 5: Transaction Anomaly Detection with Isolation Forest
- **100% Transaction Record Preservation**: All 1,067,371 Silver transaction lines scored without dropping records or altering historical financial figures (£19,012,910.85 net revenue preserved).
- **Deterministic Primary Key Lineage**: Globally unique `transaction_id` (`TXN_{source_row_id:07d}`) anchored to physical row position and dataset provenance, preventing business hash collisions.
- **Leakage-Safe Feature Pipeline**: 10-feature transformer (`TransactionFeaturePipeline`) extracting log-transformed quantities, prices, line values, learned product catalog median deviations, leakage-safe prior customer orders, unregistered user flags, and cyclical temporal encodings.
- **Calibrated Training Population**: Trained on 1,027,015 eligible commercial transactions (`n_estimators=100`, `max_samples=256`, `contamination=0.01`, `random_state=42`) while safely excluding 40,356 duplicate and administrative records from density fitting.
- **Diagnostic Explainability**: 9 human-interpretable diagnostic reason codes (`EXTREME_QUANTITY`, `EXTREME_LINE_VALUE`, `HIGH_UNIT_PRICE`, `UNUSUAL_PRODUCT_PRICE`, `OFF_HOURS_TRANSACTION`, `NEW_CUSTOMER_HIGH_VALUE`, `EXACT_DUPLICATE_ROW`, `ADMINISTRATIVE_ADJUSTMENT`, `MULTIVARIATE_ISOLATION_OUTLIER`).
- **Preservation of Retail Returns**: Recognizes that normal customer returns are legitimate business operations (only 4.43% of cancellations flagged; 95.57% preserved as normal).
- **Audit Case Verification**: Correctly flagged ambiguous credit note `C496350` (`+1` quantity, `£373.57` unit price, missing customer) as high-risk manual credit adjustment.
- **Stability & Synthetic Detection**: Mean Spearman rank correlation $\rho = 0.9593$ across 4 seeds; 100% detection rate (5/5) on extreme synthetic diagnostic injections.
- **Curated Dataset & Serving**: Persisted to `data/gold/gold_transaction_anomalies_isolation_forest.parquet` (16,664 anomalies, 1.56%) and integrated with PostgreSQL table `gold_transaction_anomalies_isolation_forest`.

### Phase 6: XGBoost Repeat-Purchase Prediction
- **Formal Customer-Snapshot Unit**: Point-in-time `(customer_id, snapshot_date)` prediction task targeting whether an identified customer will complete $\ge 1$ `VALID_PURCHASE` in the forward 30-day window $(T_{\text{obs}}, T_{\text{obs}} + 30\text{d}]$.
- **Temporal Splitting with Purging Safeguards**: 4 quarterly training snapshots (`2010-09-01` through `2011-06-01`; 17,057 customer-snapshots), validation snapshot (`2011-08-01`; 5,143 snapshots), and held-out test snapshot (`2011-10-01`; 5,438 snapshots) separated by 31-day non-overlapping purging buffers.
- **Censoring Prevention**: Strict temporal restriction to fully observable historical windows; unobserved future periods are never falsely labeled as negative instances.
- **Point-in-Time Feature Engineering**: 18 pre-cutoff features covering order recency, tenure, lifetime order frequency, monetary value, gross spend, returns, AOV, order interval cadence, 30d/60d/90d velocity metrics, SKU diversity, and item volume.
- **Empirical Model Performance**:
  - **ROC-AUC**: **0.7750** (vs. 0.7632 Logistic Regression, 0.5000 Dummy)
  - **PR-AUC**: **0.5180** (vs. 0.4951 Logistic Regression, 0.2041 Dummy)
  - **Precision @ Top 5%**: **77.49%** ($3.80\times$ baseline lift)
  - **Precision @ Top 10%**: **60.22%** ($2.95\times$ baseline lift)
  - **Brier Score**: **0.1694** (well-calibrated posterior probabilities)
- **Validation-Anchored Decision Threshold**: Operational threshold $t = 0.5900$ selected strictly on validation data to maximize $F_1$, yielding test $F_1 = 0.4812$ with zero threshold snooping.
- **Model Explainability & TreeSHAP**: Feature importance and SHAP attributions isolate `purchase_frequency`, `purchases_last_90d`, `net_monetary_value`, `spend_last_90d`, and `avg_order_interval_days` as principal drivers.
- **PostgreSQL Serving & Relational Contract**: ORM model `CustomerRepeatPurchasePrediction` with idempotent `ON CONFLICT DO UPDATE` upserts and complete 18-feature JSONB snapshots for full auditability.

### Phase 7: Production-Grade Interactive Streamlit Analytics Dashboard
- **Unified Enterprise Single Pane of Glass**: Complete multi-page interactive analytics platform built with Streamlit and Plotly across 7 modular domain pages.
- **Zero-Hardcoding Guarantee**: Metrics, charts, and tables dynamically query live PostgreSQL 16 database tables with automatic, seamless fallback to Gold Parquet files.
- **Dual Probability Calibration & Threshold Contract**: Full support for both Raw ($t = 0.5900$, validation $F_1 = 0.5217$) and Platt Sigmoid Calibrated ($t = 0.2700$, validation $F_1 = 0.5210$, test Brier score $0.1348$) probabilities.
- **Interactive Domain Views**:
  1. *Executive Overview*: Top-level commercial KPIs, revenue trajectory, top 10 international markets, top 10 products.
  2. *Sales & Revenue Analytics*: Interactive date sliders, country filters, order volume vs. cancellation trends, product catalog return margins.
  3. *Customer RFM Segmentation*: 5 K-Means cluster archetypes, 3D/2D RFM scatter plot, rule-based cross-tabulation, individual customer profile lookup.
  4. *Transaction Anomaly Detection*: 16,664 flagged outliers (1.56%), status prevalence breakdown, diagnostic reason codes, searchable anomaly table.
  5. *Repeat-Purchase Prediction*: Test ROC & calibration curves, feature importance selector (gain/weight/cover), customer propensity tiers, snapshot explorer.
  6. *Data Quality & Pipeline Monitoring*: Zero-penny reconciliation balance (£0.00 discrepancy), Medallion layer inventory, data hygiene stats, ETL execution logs.
  7. *Model Performance & Governance*: Cross-model production registry, customer activity tier fairness evaluations, and AI ethical guardrails.
- **Verification & Testing**: 31 unit & integration tests in `tests/test_dashboard.py`; 100% host-verified browser execution via autonomous subagent with zero exceptions.

---

## Repository Structure

```
E-Commerce Intelligence Platform/
├── .gitignore                      # Dataset, artifact, and secret exclusions
├── .env.example                    # Template for database credentials & connection pool
├── docker-compose.yml              # PostgreSQL 16 Alpine container with healthchecks
├── README.md                       # Platform documentation and execution guide
├── requirements.txt                # Production platform dependencies
├── pytest.ini                      # Pytest configuration and warning filters
├── configs/
│   └── ingestion_config.yaml       # Declarative ingestion settings & schema mapping
├── docs/
│   ├── phase3_postgresql.md        # Phase 3 database architecture & ER diagram
│   ├── phase4_customer_segmentation.md # Phase 4 K-Means architecture & cluster analysis
│   ├── phase5_anomaly_detection.md # Phase 5 Isolation Forest architecture & evaluation
│   └── phase6_repeat_purchase_prediction.md # Phase 6 XGBoost repeat-purchase prediction
├── data/
│   ├── raw/                        # Raw files (online+retail+ii.zip, online_retail_II.xlsx)
│   ├── bronze/                     # Bronze Parquet datasets (1,067,371 rows)
│   ├── silver/                     # Silver Parquet datasets (1,067,371 rows)
│   ├── gold/                       # Gold analytical Parquet tables & snapshot datasets
│   └── metadata/                   # JSON ingestion and reconciliation manifests
├── models/
│   ├── customer_segmentation/      # Phase 4: K-Means artifacts (model, scaler, config)
│   ├── anomaly_detection/          # Phase 5: Isolation Forest artifacts & schema
│   └── repeat_purchase/            # Phase 6: XGBoost model, preprocessor & policies
├── reports/
│   └── phase6/                     # Phase 6: SHAP & feature importance visualizations
├── sql/
│   └── analytics_queries.sql       # 12 Advanced SQL Analytics Queries
├── logs/
│   ├── ingestion.log               # Phase 1 runtime logs
│   └── phase2_etl.log              # Phase 2 runtime logs
├── src/
│   ├── __init__.py
│   ├── ingestion/                  # Phase 1 modules
│   │   ├── config.py
│   │   ├── downloader.py
│   │   ├── reader.py
│   │   ├── schema.py
│   │   ├── writer.py
│   │   └── pipeline.py
│   ├── etl/                        # Phase 2 modules
│   │   ├── config.py
│   │   ├── spark_session.py        # SparkSession & Delta Lake builder
│   │   ├── silver.py               # Silver quality & transaction status engine
│   │   ├── gold.py                 # Gold analytical models (RFM, Sales, Products)
│   │   ├── reconciliation.py       # Cross-layer accounting & audit engine
│   │   ├── pyspark_pipeline.py     # Native PySpark lakehouse pipeline
│   │   └── pipeline.py             # Phase 2 orchestrator
│   ├── databricks/                 # Databricks Lakehouse & Unity Catalog DDL
│   │   ├── delta_lake.py
│   │   └── online_retail_lakehouse_pipeline.py
│   ├── database/                   # Phase 3: PostgreSQL Database Layer
│   │   ├── __init__.py
│   │   ├── config.py               # Database configuration & pool parameters
│   │   ├── connection.py           # Engine & transactional session management
│   │   ├── models.py               # SQLAlchemy 2.0 declarative models
│   │   ├── schema_manager.py       # DDL creation and table management
│   │   ├── schema.sql              # Raw PostgreSQL DDL script
│   │   ├── loader.py               # Gold Parquet to PostgreSQL batch upsert loader
│   │   ├── queries.py              # Advanced SQL Analytics runner
│   │   ├── reconciliation.py       # Parquet vs PostgreSQL parity auditor
│   │   └── init_test_db.sql        # Test database bootstrap initialization
│   ├── features/                   # Feature engineering
│   │   ├── __init__.py
│   │   ├── rfm_features.py         # Phase 4: RFM Symmetric Log1p Pipeline
│   │   ├── anomaly_features.py     # Phase 5: Leakage-Safe Transaction Feature Pipeline
│   │   └── repeat_purchase_features.py # Phase 6: Point-in-Time 18-Feature Pipeline
│   ├── models/                     # Predictive models
│   │   ├── __init__.py
│   │   ├── config.py               # Model hyperparameters & dataclass configs
│   │   ├── customer_segmentation.py# Phase 4: K-Means segmentation engine & profiling
│   │   ├── anomaly_detector.py     # Phase 5: Isolation Forest detector & diagnostic explainer
│   │   └── repeat_purchase.py      # Phase 6: XGBoost repeat purchase predictor & evaluator
│   ├── dashboard/                  # Streamlit BI dashboard (stubbed)
│   └── utils/
│       └── logger.py
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_schema.py              # Schema & preservation tests
│   ├── test_reader.py              # Excel reader tests
│   ├── test_writer.py              # Parquet writer tests
│   ├── test_ingestion_pipeline.py  # Phase 1 integration tests
│   ├── test_silver.py              # Silver transformation & status hierarchy tests
│   ├── test_gold.py                # Gold analytical table tests
│   ├── test_reconciliation.py      # Cross-layer reconciliation tests
│   ├── test_database.py            # Phase 3: PostgreSQL integration & SQL tests
│   ├── test_features.py            # Phase 4: RFM feature pipeline tests
│   ├── test_segmentation.py        # Phase 4: K-Means segmentation & profiling tests
│   ├── test_segmentation_database.py # Phase 4: PostgreSQL segmentation serving tests
│   ├── test_segmentation_audit.py  # Phase 4: Independent QA audit regression tests
│   ├── test_anomaly_features.py    # Phase 5: Transaction feature pipeline tests
│   ├── test_anomaly_detector.py    # Phase 5: Isolation Forest detector tests
│   ├── test_anomaly_database.py    # Phase 5: PostgreSQL anomaly table schema tests
│   ├── test_anomaly_reconciliation.py # Phase 5: Reconciliation & financial non-interference tests
│   ├── test_anomaly_qa_audit.py    # Phase 5: Targeted QA audit & leakage regression tests
│   └── test_repeat_purchase.py     # Phase 6: Repeat-purchase prediction & leakage regression tests
└── scripts/
    ├── run_ingestion.py            # Phase 1 CLI
    ├── verify_bronze.py            # Phase 1 verification CLI
    ├── run_phase2.py               # Phase 2 CLI
    ├── verify_phase2.py            # Phase 2 verification CLI
    ├── audit_phase1.py             # Targeted Phase 1 audit script
    ├── check_database.py           # Phase 3: Database health check CLI
    ├── load_postgres.py            # Phase 3: Gold to PostgreSQL loading CLI
    ├── verify_postgres.py          # Phase 3: PostgreSQL reconciliation CLI
    ├── run_sql_analytics.py        # Phase 3: Advanced SQL Analytics suite CLI
    ├── train_segmentation.py       # Phase 4: K-Means segmentation training & profiling CLI
    ├── load_segmentation_postgres.py # Phase 4: PostgreSQL segmentation loading CLI
    ├── verify_segmentation.py      # Phase 4: Segmentation reconciliation CLI
    ├── qa_audit_phase4.py          # Phase 4: Targeted QA Audit verification script
    ├── audit_silver_for_phase5.py  # Phase 5: Silver dataset audit CLI
    ├── train_anomaly_detector.py   # Phase 5: Isolation Forest training & evaluation CLI
    ├── score_silver_transactions.py # Phase 5: Full 1M+ Silver scoring CLI
    ├── load_anomalies_postgres.py  # Phase 5: Scored anomalies PostgreSQL loading CLI
    ├── verify_anomalies.py         # Phase 5: Quality gates and reconciliation CLI
    ├── generate_repeat_purchase_datasets.py # Phase 6: Snapshot extraction & dataset generator
    ├── train_repeat_purchase.py    # Phase 6: XGBoost training, tuning & evaluation CLI
    └── load_repeat_purchase_postgres.py # Phase 6: PostgreSQL prediction loading CLI
```

---

## Execution Guide

### 1. Run Phase 1 Bronze Ingestion
```powershell
python scripts/run_ingestion.py
python scripts/verify_bronze.py
```

### 2. Run Phase 2 Silver & Gold Pipeline
```powershell
python scripts/run_phase2.py
python scripts/verify_phase2.py
```

### 3. Run Phase 3 PostgreSQL Setup, Loading & Analytics
```powershell
# Start PostgreSQL Container
docker compose up -d

# Verify Database Health & Connectivity
python scripts/check_database.py

# Ingest Gold Datasets into PostgreSQL
python scripts/load_postgres.py

# Verify Row Counts & Financial Reconciliation
python scripts/verify_postgres.py

# Execute All 12 Advanced SQL Analytics Queries
python scripts/run_sql_analytics.py
```

### 4. Run Phase 4 K-Means Customer Segmentation Pipeline
```powershell
# Train, Profile, and Persist K-Means Segmentation Model (k=5)
python scripts/train_segmentation.py

# Load Predictions into PostgreSQL Database
python scripts/load_segmentation_postgres.py

# Verify Database Parity & Reconciliation
python scripts/verify_segmentation.py

# Run Targeted Independent QA Audit
python scripts/qa_audit_phase4.py
```

### 5. Run Phase 5 Isolation Forest Transaction Anomaly Detection
```powershell
# Audit Silver Population & Verify Training Eligibility
python scripts/audit_silver_for_phase5.py

# Train Isolation Forest Model & Compute Evaluation Metrics
python scripts/train_anomaly_detector.py

# Score All 1,067,371 Silver Transactions & Generate Scored Gold Parquet
python scripts/score_silver_transactions.py

# (Optional) Ingest Scored Anomalies into PostgreSQL Database
python scripts/load_anomalies_postgres.py

# Run Comprehensive Quality Gate & Parity Verification
python scripts/verify_anomalies.py
```

### 6. Run Phase 6 XGBoost Repeat-Purchase Prediction
```powershell
# Generate Leakage-Safe Chronological Snapshots & Gold Datasets
python scripts/generate_repeat_purchase_datasets.py

# Train Baselines, Tune XGBoost, Compute Lift Metrics & SHAP Attributions
python scripts/train_repeat_purchase.py

# (Optional) Ingest Repeat Purchase Predictions into PostgreSQL Database
python scripts/load_repeat_purchase_postgres.py
```

### 7. Run Phase 7 Production Streamlit Analytics Dashboard
```powershell
# Launch Interactive Dashboard with Auto PostgreSQL 16 / Parquet Fallback
python scripts/run_dashboard.py --port 8501

# Or run directly with Streamlit
streamlit run src/dashboard/app.py
```

### 8. Run Complete Automated Test Suite (140 Tests across Phases 1–7)
```powershell
python -m pytest -v tests/
```
Output: **140 passed in ~2m 19s (100% pass rate, 0 failures, 0 errors, 0 warnings, 0 skipped)** with live PostgreSQL container.
