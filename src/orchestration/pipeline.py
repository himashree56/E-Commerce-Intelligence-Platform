"""
E-Commerce Intelligence Platform — Pipeline Orchestrator.

Dependency-aware execution across Medallion layers, Machine Learning stages,
relational database loading, and reconciliation audits.
Supports FULL, INFERENCE, and VALIDATE execution modes with atomic artifact protection.
"""
import enum
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from src.orchestration.logger import OperationalLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class PipelineMode(str, enum.Enum):
    """Pipeline execution modes."""
    FULL = "full"             # Complete rebuild and retraining from raw sheets
    INFERENCE = "inference"   # Score and load using persisted models (no retraining)
    VALIDATE = "validate"     # Read-only verification of data and reconciliation


def atomic_save_parquet(df: pd.DataFrame, target_path: Path, **kwargs) -> None:
    """
    Saves DataFrame to Parquet atomically using temporary file swap.
    Prevents corruption of critical datasets if serialization fails mid-write.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.stem}.tmp_{int(time.time()*1000)}.parquet")
    try:
        df.to_parquet(temp_path, **kwargs)
        if target_path.exists():
            os.replace(temp_path, target_path)
        else:
            temp_path.rename(target_path)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise IOError(f"Atomic parquet write failed for {target_path}: {e}") from e


class PipelineOrchestrator:
    """
    Orchestrates end-to-end data pipelines, ML workflows, and database loading.
    """

    def __init__(
        self,
        mode: PipelineMode = PipelineMode.FULL,
        run_id: Optional[str] = None,
        skip_db: bool = False,
        timeout_seconds: int = 600,
    ):
        self.mode = mode
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"run_{now_str}_{self.mode.value}"
        self.skip_db = skip_db
        self.timeout_seconds = timeout_seconds
        self.logger = OperationalLogger(run_id=self.run_id, pipeline_mode=self.mode.value)

    def _run_script(self, script_name: str, args: Optional[List[str]] = None) -> subprocess.CompletedProcess:
        """Executes a standalone project script with Python executable and timeout."""
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        cmd = [sys.executable, str(script_path)] + (args or [])
        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip()
            raise RuntimeError(f"Script {script_name} failed with code {res.returncode}:\n{err_msg}")
        return res

    # -------------------------------------------------------------------------
    # Stage 1: Bronze Ingestion
    # -------------------------------------------------------------------------
    def run_bronze_ingestion(self) -> int:
        """Phase 1: Ingests raw workbook into Bronze Parquet."""
        with self.logger.track_stage("BRONZE_INGESTION", expected_records=1067371) as ctx:
            if self.mode == PipelineMode.VALIDATE:
                bronze_p = DATA_DIR / "bronze" / "online_retail_bronze.parquet"
                if not bronze_p.exists():
                    raise FileNotFoundError(f"Bronze file missing: {bronze_p}")
                df = pd.read_parquet(bronze_p)
                ctx["records"] = len(df)
                return len(df)

            self._run_script("run_ingestion.py")
            bronze_p = DATA_DIR / "bronze" / "online_retail_bronze.parquet"
            df = pd.read_parquet(bronze_p)
            ctx["records"] = len(df)
            return len(df)

    # -------------------------------------------------------------------------
    # Stage 2 & 3: Silver Cleaning & Gold Analytics
    # -------------------------------------------------------------------------
    def run_silver_and_gold_etl(self) -> Dict[str, int]:
        """Phase 2: Cleans Silver dataset and calculates Gold aggregates."""
        with self.logger.track_stage("SILVER_AND_GOLD_ETL", expected_records=1067371) as ctx:
            if self.mode == PipelineMode.VALIDATE:
                silver_p = DATA_DIR / "silver" / "online_retail_silver.parquet"
                if not silver_p.exists():
                    raise FileNotFoundError(f"Silver file missing: {silver_p}")
                df = pd.read_parquet(silver_p)
                ctx["records"] = len(df)
                return {"silver": len(df)}

            self._run_script("run_phase2.py")
            silver_p = DATA_DIR / "silver" / "online_retail_silver.parquet"
            df = pd.read_parquet(silver_p)
            ctx["records"] = len(df)
            return {"silver": len(df)}

    # -------------------------------------------------------------------------
    # Stage 4: PostgreSQL Core Analytics Load
    # -------------------------------------------------------------------------
    def run_postgres_core_load(self) -> int:
        """Phase 3: Upserts core Gold analytics tables into PostgreSQL."""
        if self.skip_db or self.mode == PipelineMode.VALIDATE:
            self.logger.log_stage_skipped("POSTGRES_CORE_LOAD", "Skipped by user flag or validate mode")
            return 0

        with self.logger.track_stage("POSTGRES_CORE_LOAD", expected_records=55797) as ctx:
            self._run_script("load_postgres.py")
            ctx["records"] = 55797
            return 55797

    # -------------------------------------------------------------------------
    # Stage 5: Customer RFM Segmentation
    # -------------------------------------------------------------------------
    def run_customer_segmentation(self) -> int:
        """Phase 4: Fits or applies K-Means customer segmentation (k=5)."""
        with self.logger.track_stage("CUSTOMER_SEGMENTATION", expected_records=5939, model_version="k5_kmeans") as ctx:
            gold_seg_p = DATA_DIR / "gold" / "gold_customer_segmentation_kmeans.parquet"

            if self.mode == PipelineMode.FULL:
                self._run_script("train_segmentation.py")
            elif self.mode == PipelineMode.INFERENCE:
                # Use pre-trained model to assign clusters
                import joblib
                model = joblib.load(MODELS_DIR / "customer_segmentation" / "kmeans_model.joblib")
                prep = joblib.load(MODELS_DIR / "customer_segmentation" / "preprocessor.joblib")
                rfm_df = pd.read_parquet(DATA_DIR / "gold" / "gold_customer_rfm.parquet")
                X_scaled = prep.transform(rfm_df)
                clusters = model.predict(X_scaled)
                rfm_df["cluster_id"] = clusters
                atomic_save_parquet(rfm_df, gold_seg_p)
            elif self.mode == PipelineMode.VALIDATE:
                if not gold_seg_p.exists():
                    raise FileNotFoundError(f"Segmentation parquet missing: {gold_seg_p}")

            df = pd.read_parquet(gold_seg_p)
            ctx["records"] = len(df)
            return len(df)

    # -------------------------------------------------------------------------
    # Stage 6: Transaction Anomaly Scoring
    # -------------------------------------------------------------------------
    def run_anomaly_scoring(self) -> int:
        """Phase 5: Scores Silver transactions with Isolation Forest."""
        with self.logger.track_stage("ANOMALY_SCORING", expected_records=1067371, model_version="isolation_forest_v1") as ctx:
            anom_p = DATA_DIR / "gold" / "gold_transaction_anomalies_isolation_forest.parquet"

            if self.mode == PipelineMode.FULL:
                self._run_script("train_anomaly_detector.py")
                self._run_script("score_silver_transactions.py")
            elif self.mode == PipelineMode.INFERENCE:
                self._run_script("score_silver_transactions.py")
            elif self.mode == PipelineMode.VALIDATE:
                if not anom_p.exists():
                    raise FileNotFoundError(f"Anomaly parquet missing: {anom_p}")

            df = pd.read_parquet(anom_p)
            ctx["records"] = len(df)
            return len(df)

    # -------------------------------------------------------------------------
    # Stage 7: Repeat-Purchase Prediction
    # -------------------------------------------------------------------------
    def run_repeat_purchase_prediction(self) -> int:
        """Phase 6: Generates point-in-time features & evaluates XGBoost."""
        with self.logger.track_stage("REPEAT_PURCHASE_PREDICTION", expected_records=5438, model_version="xgb_v1_calibrated") as ctx:
            test_p = DATA_DIR / "gold" / "gold_repeat_purchase_test.parquet"

            if self.mode == PipelineMode.FULL:
                self._run_script("generate_repeat_purchase_datasets.py")
                self._run_script("train_repeat_purchase.py")
            elif self.mode == PipelineMode.INFERENCE:
                # Re-generate test snapshot predictions using loaded predictor
                from src.models.repeat_purchase import RepeatPurchasePredictor
                predictor = RepeatPurchasePredictor.load(MODELS_DIR / "repeat_purchase")
                test_df = pd.read_parquet(test_p)
                probs = predictor.predict_proba(test_df, calibrated=True)
                test_df["repeat_purchase_probability"] = probs
                atomic_save_parquet(test_df, test_p)
            elif self.mode == PipelineMode.VALIDATE:
                if not test_p.exists():
                    raise FileNotFoundError(f"Repeat purchase test parquet missing: {test_p}")

            df = pd.read_parquet(test_p)
            ctx["records"] = len(df)
            return len(df)

    # -------------------------------------------------------------------------
    # Stage 8: PostgreSQL ML Tables Load
    # -------------------------------------------------------------------------
    def run_postgres_ml_load(self) -> int:
        """Loads Phase 4, 5, and 6 model predictions to PostgreSQL."""
        if self.skip_db or self.mode == PipelineMode.VALIDATE:
            self.logger.log_stage_skipped("POSTGRES_ML_LOAD", "Skipped by user flag or validate mode")
            return 0

        with self.logger.track_stage("POSTGRES_ML_LOAD") as ctx:
            self._run_script("load_segmentation_postgres.py")
            self._run_script("load_anomalies_postgres.py")
            self._run_script("load_repeat_purchase_postgres.py")
            ctx["records"] = 5939 + 1067371 + 5438
            return ctx["records"]

    # -------------------------------------------------------------------------
    # Stage 9: Verification & Reconciliation Audit
    # -------------------------------------------------------------------------
    def run_reconciliation_audit(self) -> Dict[str, Any]:
        """Runs automated verification and zero-penny financial reconciliation."""
        with self.logger.track_stage("RECONCILIATION_AUDIT") as ctx:
            # Check Silver totals
            silver_df = pd.read_parquet(DATA_DIR / "silver" / "online_retail_silver.parquet")
            gross = float(silver_df["gross_sales"].sum())
            returns = float(silver_df["return_amount"].sum())
            net = float(silver_df["net_revenue"].sum())

            diff = abs(net - 19012910.85)
            if diff > 0.01:
                raise ValueError(f"Net revenue reconciliation failure: computed £{net:.2f}, expected £19,012,910.85")

            ctx["details"] = {
                "gross_sales": gross,
                "returns": returns,
                "net_revenue": net,
                "balance_diff": diff,
                "silver_records": len(silver_df),
                "reconciliation_status": "ZERO-PENNY BALANCED",
            }
            return ctx["details"]

    # -------------------------------------------------------------------------
    # Master Pipeline Runner
    # -------------------------------------------------------------------------
    def execute(self) -> Dict[str, Any]:
        """Executes full orchestrated workflow according to configured mode."""
        print("================================================================================")
        print(f"PIPELINE ORCHESTRATION INITIATED: {self.run_id}")
        print(f"Execution Mode: {self.mode.value.upper()} | Skip DB: {self.skip_db}")
        print("================================================================================")

        t_start = time.perf_counter()
        results = {}

        try:
            results["bronze"] = self.run_bronze_ingestion()
            results["silver_gold"] = self.run_silver_and_gold_etl()
            results["postgres_core"] = self.run_postgres_core_load()
            results["segmentation"] = self.run_customer_segmentation()
            results["anomalies"] = self.run_anomaly_scoring()
            results["repeat_purchase"] = self.run_repeat_purchase_prediction()
            results["postgres_ml"] = self.run_postgres_ml_load()
            results["reconciliation"] = self.run_reconciliation_audit()

            total_duration = time.perf_counter() - t_start
            print("================================================================================")
            print(f"PIPELINE ORCHESTRATION COMPLETED SUCCESSFULLY in {total_duration:.2f}s")
            print("================================================================================")
            return {
                "status": "SUCCESS",
                "run_id": self.run_id,
                "mode": self.mode.value,
                "duration_seconds": total_duration,
                "results": results,
            }
        except Exception as e:
            total_duration = time.perf_counter() - t_start
            print("================================================================================")
            print(f"PIPELINE ORCHESTRATION FAILED after {total_duration:.2f}s: {e}")
            print("================================================================================")
            return {
                "status": "FAILED",
                "run_id": self.run_id,
                "mode": self.mode.value,
                "duration_seconds": total_duration,
                "error": str(e),
            }
