"""
CLI Entrypoint for E-Commerce Intelligence Pipeline Orchestration.

Usage:
    python scripts/run_pipeline.py [--mode full|inference|validate] [--skip-db] [--stage STAGE]
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.pipeline import PipelineMode, PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(description="E-Commerce Intelligence Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "inference", "validate"],
        default="inference",
        help="Pipeline execution mode: full (retrain), inference (score with saved models), validate (read-only audit)",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip PostgreSQL database load steps",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Execute single isolated stage (e.g., bronze, silver, segmentation, anomalies, repeat_purchase, reconciliation)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Subprocess script execution timeout in seconds",
    )
    args = parser.parse_args()

    mode_map = {
        "full": PipelineMode.FULL,
        "inference": PipelineMode.INFERENCE,
        "validate": PipelineMode.VALIDATE,
    }
    mode = mode_map[args.mode]

    orchestrator = PipelineOrchestrator(
        mode=mode,
        skip_db=args.skip_db,
        timeout_seconds=args.timeout,
    )

    if args.stage:
        stage_name = args.stage.lower().strip()
        print(f"Executing single stage: {stage_name} (Mode: {mode.value})")
        try:
            if stage_name in ("bronze", "ingestion"):
                res = orchestrator.run_bronze_ingestion()
            elif stage_name in ("silver", "gold", "etl"):
                res = orchestrator.run_silver_and_gold_etl()
            elif stage_name in ("postgres_core", "db_core"):
                res = orchestrator.run_postgres_core_load()
            elif stage_name in ("segmentation", "kmeans"):
                res = orchestrator.run_customer_segmentation()
            elif stage_name in ("anomalies", "isolation_forest"):
                res = orchestrator.run_anomaly_scoring()
            elif stage_name in ("repeat_purchase", "xgboost"):
                res = orchestrator.run_repeat_purchase_prediction()
            elif stage_name in ("postgres_ml", "db_ml"):
                res = orchestrator.run_postgres_ml_load()
            elif stage_name in ("reconciliation", "audit", "verify"):
                res = orchestrator.run_reconciliation_audit()
            else:
                print(f"Unknown stage name: {stage_name}")
                sys.exit(1)
            print(f"Stage {stage_name} completed: {res}")
            sys.exit(0)
        except Exception as e:
            print(f"Stage {stage_name} failed: {e}")
            sys.exit(1)

    result = orchestrator.execute()
    if result["status"] == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
