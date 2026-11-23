# Phase 8 — Docker Compose, Shell Automation & Operational Reliability

## 1. Executive Summary

Phase 8 establishes a hardened, cross-platform production operations, container orchestration, and automated pipeline execution framework for the **E-Commerce Intelligence Platform**.

### Key Deliverables:
1. **Docker Compose**: Containerized PostgreSQL 16 Alpine serving layer with persistent named volume storage (`ecommerce_pgdata`), health check validation, and zero secret embedding.
2. **Cross-Platform Automation**: Five matched pairs of production shell entrypoints for Windows PowerShell (`.ps1`) and Unix Bash (`.sh`):
   - `scripts/setup.ps1` / `scripts/setup.sh`
   - `scripts/start.ps1` / `scripts/start.sh`
   - `scripts/run_pipeline.ps1` / `scripts/run_pipeline.sh`
   - `scripts/run_tests.ps1` / `scripts/run_tests.sh`
   - `scripts/stop.ps1` / `scripts/stop.sh`
3. **Unified Pipeline Orchestration Engine (`src/orchestration/`)**:
   - Dependency-aware DAG coordinating Stages 1 through 9.
   - Separate execution modes: `FULL` (complete rebuild/train), `INFERENCE` (score with saved models), and `VALIDATE` (read-only audit).
   - Atomic file persistence (`atomic_save_parquet`) preventing dataset corruption upon interrupted execution.
4. **Structured Operational Audit Logging**: Real-time run tracking logged to both local JSON Lines manifest (`logs/pipeline_runs.jsonl`) and PostgreSQL table `etl_execution_metadata`.
5. **Environment Validator (`src/utils/env_validator.py`)**: Pre-flight verification of Python version, directory structure, disk space, `.env` parameters, and Docker CLI.

---

## 2. Execution Dependency Graph

```
Raw Multi-Sheet Excel (data/raw/online_retail_II.xlsx)
  │
  ▼ [Stage 1: BRONZE_INGESTION] (`scripts/run_ingestion.py`)
Bronze Dataset (`data/bronze/online_retail_bronze.parquet`) [1,067,371 rows]
  │
  ▼ [Stage 2 & 3: SILVER_AND_GOLD_ETL] (`scripts/run_phase2.py`)
Silver Transactions (`data/silver/online_retail_silver.parquet`) [1,067,371 rows]
  ├──> Gold Monthly Sales (`gold_monthly_sales.parquet`) [25 rows]
  ├──> Gold Product Performance (`gold_product_performance.parquet`) [4,920 rows]
  ├──> Gold Country Revenue (`gold_country_revenue.parquet`) [43 rows]
  ├──> Gold Customer RFM (`gold_customer_rfm.parquet`) [5,939 rows]
  └──> Gold Customer Purchase History (`gold_customer_purchase_history.parquet`) [44,870 rows]
  │
  ├──> [Stage 4: POSTGRES_CORE_LOAD] (`scripts/load_postgres.py`)
  │    Loads 5 core analytics tables into PostgreSQL 16
  │
  ├──> [Stage 5: CUSTOMER_SEGMENTATION] (`scripts/train_segmentation.py`)
  │    Generates `gold_customer_segmentation_kmeans.parquet` [5,939 rows]
  │
  ├──> [Stage 6: ANOMALY_SCORING] (`scripts/score_silver_transactions.py`)
  │    Generates `gold_transaction_anomalies_isolation_forest.parquet` [1,067,371 rows, 16,664 outliers]
  │
  ├──> [Stage 7: REPEAT_PURCHASE_PREDICTION] (`scripts/train_repeat_purchase.py`)
  │    Generates `gold_repeat_purchase_test.parquet` [5,438 rows]
  │
  ├──> [Stage 8: POSTGRES_ML_LOAD]
  │    Upserts segmentation, anomalies, and repeat-purchase predictions to PostgreSQL
  │
  └──> [Stage 9: RECONCILIATION_AUDIT]
       Verifies zero-penny commercial ledger reconciliation (£19,012,910.85 net revenue)
```

---

## 3. Cross-Platform Automation Runbook

### Windows PowerShell Commands
```powershell
# 1. First-time system setup & dependency installation
.\scripts\setup.ps1

# 2. Start PostgreSQL 16 container and verify database health
.\scripts\start.ps1

# 3. Optional: Start container and launch Streamlit analytics dashboard
.\scripts\start.ps1 -WithDashboard -DashboardPort 8501

# 4. Run pipeline in inference mode (scores using existing saved models)
.\scripts\run_pipeline.ps1 -Mode inference

# 5. Run pipeline in read-only validation mode
.\scripts\run_pipeline.ps1 -Mode validate

# 6. Run pipeline in full retrain mode
.\scripts\run_pipeline.ps1 -Mode full

# 7. Execute complete automated test suite (153 tests)
.\scripts\run_tests.ps1

# 8. Graceful service shutdown
.\scripts\stop.ps1
```

### Unix Bash Commands
```bash
# 1. First-time system setup & dependency installation
./scripts/setup.sh

# 2. Start PostgreSQL 16 container and verify database health
./scripts/start.sh

# 3. Optional: Start container and launch Streamlit dashboard
./scripts/start.sh --with-dashboard --port 8501

# 4. Run pipeline in inference mode
./scripts/run_pipeline.sh --mode inference

# 5. Run pipeline in validation mode
./scripts/run_pipeline.sh --mode validate

# 6. Run pipeline in full retrain mode
./scripts/run_pipeline.sh --mode full

# 7. Execute complete test suite
./scripts/run_tests.sh

# 8. Graceful service shutdown
./scripts/stop.sh
```

---

## 4. Idempotence & Recovery Architecture

1. **Relational Upserts**: PostgreSQL loaders enforce `ON CONFLICT DO UPDATE` or `ON CONFLICT DO NOTHING` on explicit primary keys (`invoice_no`, `stock_code`, `customer_id`, `transaction_id`, `prediction_id`), preventing duplicate row insertion upon repeated runs.
2. **Atomic File Serialization**: The orchestrator employs `atomic_save_parquet()`. Output datasets are serialized to `.tmp_<timestamp>.parquet` and atomically renamed via `os.replace()`, guaranteeing that an interrupted write cannot corrupt existing verified datasets.
3. **Structured Audit Trail**: Every run produces an immutable JSON Line event in `logs/pipeline_runs.jsonl` and a database record in `etl_execution_metadata`.
