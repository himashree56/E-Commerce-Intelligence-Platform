"""Phase 2 ETL Pipeline Orchestrator: Bronze -> Silver -> Gold + Reconciliation."""
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.etl.config import ETLConfig
from src.etl.gold import GoldTransformationSummary, GoldTransformer
from src.etl.reconciliation import ReconciliationAuditor, ReconciliationReport
from src.etl.silver import SilverTransformationSummary, SilverTransformer
from src.utils.logger import setup_logging

logger = logging.getLogger("ecommerce_platform.etl_pipeline")


@dataclass
class Phase2PipelineSummary:
    """End-to-end execution summary for Phase 2."""

    run_id: str
    status: str
    start_time: str
    end_time: str
    duration_seconds: float
    bronze_records_ingested: int
    silver_records_produced: int
    gold_monthly_rows: int
    gold_rfm_rows: int
    gold_product_rows: int
    gold_country_rows: int
    gold_history_rows: int
    gross_sales_gbp: float
    return_amount_gbp: float
    net_revenue_gbp: float
    reconciliation_passed: bool
    silver_summary: Dict[str, Any]
    gold_summary: Dict[str, Any]
    reconciliation_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase2Pipeline:
    """Executes Bronze -> Silver -> Gold transformation and reconciliation."""

    def __init__(self, config: Optional[ETLConfig] = None) -> None:
        self.config = config or ETLConfig()
        setup_logging(log_dir=self.config.logs_dir, log_file_name="phase2_etl.log")
        self.silver_transformer = SilverTransformer(self.config)
        self.gold_transformer = GoldTransformer(self.config)
        self.auditor = ReconciliationAuditor(self.config)

    def run(self) -> Phase2PipelineSummary:
        """Executes complete Phase 2 transformation and verification."""
        start_time = datetime.now(timezone.utc)
        run_id = f"phase2_{start_time.strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"=== Starting Phase 2 ETL Pipeline (Run ID: {run_id}) ===")

        # 1. Run Silver Layer Transformation
        logger.info("Executing Silver layer transformation...")
        silver_df, silver_summary = self.silver_transformer.process_and_save()

        # Load bronze dataframe for reconciliation
        bronze_df = pd.read_parquet(self.config.bronze_parquet_path)

        # 2. Run Gold Layer Transformations
        logger.info("Executing Gold layer analytical transformations...")
        gold_tables, gold_summary = self.gold_transformer.process_and_save_all(silver_df)

        # 3. Run Reconciliation Audit
        logger.info("Executing cross-layer reconciliation audit...")
        recon_report = self.auditor.audit(
            bronze_df=bronze_df,
            silver_df=silver_df,
            gold_monthly_df=gold_tables["monthly_sales"],
            gold_country_df=gold_tables["country_revenue"],
        )

        end_time = datetime.now(timezone.utc)
        duration = round((end_time - start_time).total_seconds(), 2)

        pipeline_summary = Phase2PipelineSummary(
            run_id=run_id,
            status="SUCCESS" if recon_report.all_tests_passed else "FAILED_RECONCILIATION",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            bronze_records_ingested=recon_report.bronze_total_rows,
            silver_records_produced=recon_report.silver_total_rows,
            gold_monthly_rows=gold_summary.monthly_sales_rows,
            gold_rfm_rows=gold_summary.customer_rfm_rows,
            gold_product_rows=gold_summary.product_performance_rows,
            gold_country_rows=gold_summary.country_revenue_rows,
            gold_history_rows=gold_summary.customer_purchase_history_rows,
            gross_sales_gbp=silver_summary.gross_sales_gbp,
            return_amount_gbp=silver_summary.return_amount_gbp,
            net_revenue_gbp=silver_summary.net_revenue_gbp,
            reconciliation_passed=recon_report.all_tests_passed,
            silver_summary=silver_summary.to_dict(),
            gold_summary=gold_summary.to_dict(),
            reconciliation_summary=recon_report.to_dict(),
        )

        # Save metadata manifest
        meta_path = self.config.metadata_dir / f"{run_id}.json"
        latest_meta = self.config.metadata_dir / "latest_phase2.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_summary.to_dict(), f, indent=2)
        with open(latest_meta, "w", encoding="utf-8") as f:
            json.dump(pipeline_summary.to_dict(), f, indent=2)

        logger.info(
            f"=== Phase 2 Completed in {duration}s! Status: {pipeline_summary.status} ==="
        )
        return pipeline_summary
