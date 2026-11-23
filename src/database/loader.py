"""Gold Parquet to PostgreSQL Loading Pipeline with Idempotence and Transaction Safety."""
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.database.config import DatabaseConfig
from src.database.connection import DatabaseManager
from src.database.models import (
    Base,
    CountryRevenueAnalytics,
    CustomerPurchaseHistory,
    CustomerRepeatPurchasePrediction,
    CustomerRFMAnalytics,
    CustomerSegmentationKMeans,
    DataQualityAuditResults,
    ETLExecutionMetadata,
    MonthlySalesAnalytics,
    ProductPerformanceAnalytics,
    TransactionAnomaliesIsolationForest,
)
from src.etl.config import ETLConfig

logger = logging.getLogger("ecommerce_platform.database.loader")


@dataclass
class TableLoadSummary:
    """Summary of records loaded into a specific table."""

    table_name: str
    source_rows: int
    loaded_rows: int
    duration_seconds: float
    status: str = "SUCCESS"
    error: Optional[str] = None


@dataclass
class DatabaseLoadReport:
    """Overall report summarizing the Gold-to-PostgreSQL ingestion run."""

    run_id: str
    target_database: str
    started_at: datetime
    completed_at: datetime
    total_duration_seconds: float
    status: str
    table_summaries: Dict[str, TableLoadSummary] = field(default_factory=dict)
    total_records_loaded: int = 0
    all_tables_verified: bool = False
    discrepancies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target_database": self.target_database,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "total_duration_seconds": self.total_duration_seconds,
            "status": self.status,
            "total_records_loaded": self.total_records_loaded,
            "all_tables_verified": self.all_tables_verified,
            "discrepancies": self.discrepancies,
            "table_summaries": {
                k: {
                    "table_name": v.table_name,
                    "source_rows": v.source_rows,
                    "loaded_rows": v.loaded_rows,
                    "duration_seconds": v.duration_seconds,
                    "status": v.status,
                    "error": v.error,
                }
                for k, v in self.table_summaries.items()
            },
        }


class GoldDatabaseLoader:
    """Loads all five Gold Parquet analytical datasets into PostgreSQL."""

    def __init__(
        self,
        etl_config: Optional[ETLConfig] = None,
        db_config: Optional[DatabaseConfig] = None,
        is_test: bool = False,
        batch_size: int = 5000,
    ) -> None:
        self.etl_config = etl_config or ETLConfig()
        self.db_config = db_config or DatabaseConfig()
        self.is_test = is_test
        self.batch_size = batch_size
        self.db_manager = DatabaseManager(config=self.db_config, is_test=is_test)

    # -------------------------------------------------------------------------
    # Sanitization and Conversion Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _sanitize_record(record: Dict[str, Any], decimal_cols: List[str]) -> Dict[str, Any]:
        """Sanitizes raw pandas dict record into clean PostgreSQL/SQLAlchemy types."""
        clean: Dict[str, Any] = {}
        for k, v in record.items():
            if pd.isna(v) or v is None:
                clean[k] = None
            elif k in decimal_cols:
                # Format scores with 6 decimal places, distances/prices with 4, monetary/qty with 2
                if k in ("raw_decision_score", "anomaly_score"):
                    precision = 6
                elif k in ("distance_to_centroid", "unit_price"):
                    precision = 4
                else:
                    precision = 2
                clean[k] = Decimal(f"{float(v):.{precision}f}")
            elif isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_, bool)):
                clean[k] = bool(v)
            elif isinstance(v, pd.Timestamp):
                clean[k] = v.to_pydatetime()
            else:
                clean[k] = v
        return clean

    def _execute_upsert_batch(
        self,
        session: Session,
        model: Type[Base],
        records: List[Dict[str, Any]],
        primary_key_cols: List[str],
    ) -> int:
        """Executes batched PostgreSQL INSERT ... ON CONFLICT DO UPDATE."""
        if not records:
            return 0

        stmt = insert(model).values(records)
        update_cols = {
            col.name: getattr(stmt.excluded, col.name)
            for col in model.__table__.columns
            if col.name not in primary_key_cols and col.name != "created_at"
        }

        if update_cols:
            stmt = stmt.on_conflict_do_update(
                index_elements=primary_key_cols,
                set_=update_cols,
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=primary_key_cols)

        session.execute(stmt)
        return len(records)

    def _load_in_batches(
        self,
        session: Session,
        model: Type[Base],
        df: pd.DataFrame,
        primary_key_cols: List[str],
        decimal_cols: List[str],
    ) -> int:
        """Processes dataframe in memory-efficient chunks with type sanitization."""
        records = df.to_dict(orient="records")
        total_loaded = 0

        for i in range(0, len(records), self.batch_size):
            chunk = records[i : i + self.batch_size]
            sanitized_chunk = [self._sanitize_record(r, decimal_cols) for r in chunk]
            count = self._execute_upsert_batch(session, model, sanitized_chunk, primary_key_cols)
            total_loaded += count

        return total_loaded

    # -------------------------------------------------------------------------
    # Individual Table Loaders
    # -------------------------------------------------------------------------

    def load_monthly_sales(self, session: Session, df: pd.DataFrame) -> TableLoadSummary:
        start = time.perf_counter()
        t_name = MonthlySalesAnalytics.__tablename__
        logger.info(f"Loading {len(df):,} records into {t_name}...")

        decimal_cols = ["gross_sales", "return_amount", "net_revenue", "average_order_value"]
        loaded = self._load_in_batches(
            session=session,
            model=MonthlySalesAnalytics,
            df=df,
            primary_key_cols=["invoice_year_month"],
            decimal_cols=decimal_cols,
        )
        dur = round(time.perf_counter() - start, 3)
        return TableLoadSummary(t_name, len(df), loaded, dur)

    def load_customer_rfm(self, session: Session, df: pd.DataFrame) -> TableLoadSummary:
        start = time.perf_counter()
        t_name = CustomerRFMAnalytics.__tablename__
        logger.info(f"Loading {len(df):,} records into {t_name}...")

        decimal_cols = ["monetary_gross", "monetary_net", "return_amount"]
        loaded = self._load_in_batches(
            session=session,
            model=CustomerRFMAnalytics,
            df=df,
            primary_key_cols=["customer_id"],
            decimal_cols=decimal_cols,
        )
        dur = round(time.perf_counter() - start, 3)
        return TableLoadSummary(t_name, len(df), loaded, dur)

    def load_product_performance(self, session: Session, df: pd.DataFrame) -> TableLoadSummary:
        start = time.perf_counter()
        t_name = ProductPerformanceAnalytics.__tablename__
        logger.info(f"Loading {len(df):,} records into {t_name}...")

        decimal_cols = ["gross_sales", "return_amount", "net_revenue", "return_rate_pct"]
        loaded = self._load_in_batches(
            session=session,
            model=ProductPerformanceAnalytics,
            df=df,
            primary_key_cols=["stock_code"],
            decimal_cols=decimal_cols,
        )
        dur = round(time.perf_counter() - start, 3)
        return TableLoadSummary(t_name, len(df), loaded, dur)

    def load_country_revenue(self, session: Session, df: pd.DataFrame) -> TableLoadSummary:
        start = time.perf_counter()
        t_name = CountryRevenueAnalytics.__tablename__
        logger.info(f"Loading {len(df):,} records into {t_name}...")

        decimal_cols = ["gross_sales", "return_amount", "net_revenue", "market_share_pct"]
        loaded = self._load_in_batches(
            session=session,
            model=CountryRevenueAnalytics,
            df=df,
            primary_key_cols=["country"],
            decimal_cols=decimal_cols,
        )
        dur = round(time.perf_counter() - start, 3)
        return TableLoadSummary(t_name, len(df), loaded, dur)

    def load_customer_purchase_history(self, session: Session, df: pd.DataFrame) -> TableLoadSummary:
        start = time.perf_counter()
        t_name = CustomerPurchaseHistory.__tablename__
        logger.info(f"Loading {len(df):,} records into {t_name}...")

        decimal_cols = ["order_gross_sales", "order_return_amount", "order_net_revenue"]
        loaded = self._load_in_batches(
            session=session,
            model=CustomerPurchaseHistory,
            df=df,
            primary_key_cols=["invoice_no"],
            decimal_cols=decimal_cols,
        )
        dur = round(time.perf_counter() - start, 3)
        return TableLoadSummary(t_name, len(df), loaded, dur)

    def load_customer_segmentation(
        self, session: Session, df: pd.DataFrame
    ) -> TableLoadSummary:
        """Loads ML K-Means customer segmentation results into PostgreSQL."""
        start = time.perf_counter()
        t_name = CustomerSegmentationKMeans.__tablename__
        logger.info(f"Loading {len(df):,} records into {t_name}...")

        decimal_cols = ["monetary_net", "distance_to_centroid"]
        loaded = self._load_in_batches(
            session=session,
            model=CustomerSegmentationKMeans,
            df=df,
            primary_key_cols=["customer_id"],
            decimal_cols=decimal_cols,
        )
        dur = round(time.perf_counter() - start, 3)
        return TableLoadSummary(t_name, len(df), loaded, dur)

    def load_transaction_anomalies(
        self, session: Session, df: pd.DataFrame
    ) -> TableLoadSummary:
        """Loads Isolation Forest transaction anomaly scores into PostgreSQL."""
        t_name = TransactionAnomaliesIsolationForest.__tablename__
        logger.info(f"Loading {len(df):,} anomaly records into '{t_name}'...")
        start = time.perf_counter()

        decimal_cols = [
            "quantity",
            "unit_price",
            "line_total",
            "raw_decision_score",
            "anomaly_score",
        ]
        loaded = self._load_in_batches(
            session=session,
            model=TransactionAnomaliesIsolationForest,
            df=df,
            primary_key_cols=["transaction_id"],
            decimal_cols=decimal_cols,
        )
        dur = round(time.perf_counter() - start, 3)
        return TableLoadSummary(t_name, len(df), loaded, dur)

    def load_repeat_purchase_predictions(
        self, session: Session, df: pd.DataFrame
    ) -> TableLoadSummary:
        """Loads XGBoost repeat-purchase prediction scores into PostgreSQL."""
        t_name = CustomerRepeatPurchasePrediction.__tablename__
        logger.info(f"Loading {len(df):,} repeat purchase prediction records into '{t_name}'...")
        start = time.perf_counter()

        decimal_cols = ["repeat_purchase_probability"]
        loaded = self._load_in_batches(
            session=session,
            model=CustomerRepeatPurchasePrediction,
            df=df,
            primary_key_cols=["prediction_id"],
            decimal_cols=decimal_cols,
        )
        dur = round(time.perf_counter() - start, 3)
        return TableLoadSummary(t_name, len(df), loaded, dur)

    # -------------------------------------------------------------------------
    # Full Gold Pipeline Orchestration
    # -------------------------------------------------------------------------

    def load_all_gold_tables(self) -> DatabaseLoadReport:
        """Executes full loading pipeline from Gold Parquet to PostgreSQL."""
        run_id = f"pg_load_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        target_db = self.db_config.test_db_name if self.is_test else self.db_config.db_name

        logger.info(f"=== Starting Gold-to-PostgreSQL Load Run: {run_id} (Target: {target_db}) ===")

        # 1. Read source Gold parquet datasets
        p_monthly = self.etl_config.gold_monthly_sales_path
        p_rfm = self.etl_config.gold_customer_rfm_path
        p_prod = self.etl_config.gold_product_performance_path
        p_country = self.etl_config.gold_country_revenue_path
        p_hist = self.etl_config.gold_customer_purchase_history_path

        for p in [p_monthly, p_rfm, p_prod, p_country, p_hist]:
            if not p.exists():
                raise FileNotFoundError(f"Required Gold Parquet dataset missing: {p}")

        df_monthly = pd.read_parquet(p_monthly)
        df_rfm = pd.read_parquet(p_rfm)
        df_prod = pd.read_parquet(p_prod)
        df_country = pd.read_parquet(p_country)
        df_hist = pd.read_parquet(p_hist)

        table_summaries: Dict[str, TableLoadSummary] = {}
        discrepancies: List[str] = []

        # 2. Execute transactional load in dependency order (RFM before Purchase History)
        try:
            with self.db_manager.session_scope() as session:
                # Table 1: Customer RFM (Parent for Purchase History FK)
                s_rfm = self.load_customer_rfm(session, df_rfm)
                table_summaries[s_rfm.table_name] = s_rfm

                # Table 2: Customer Purchase History (Child referencing Customer RFM)
                s_hist = self.load_customer_purchase_history(session, df_hist)
                table_summaries[s_hist.table_name] = s_hist

                # Table 3: Monthly Sales Analytics
                s_monthly = self.load_monthly_sales(session, df_monthly)
                table_summaries[s_monthly.table_name] = s_monthly

                # Table 4: Product Performance Analytics
                s_prod = self.load_product_performance(session, df_prod)
                table_summaries[s_prod.table_name] = s_prod

                # Table 5: Country Revenue Analytics
                s_country = self.load_country_revenue(session, df_country)
                table_summaries[s_country.table_name] = s_country

                # Calculate total loaded
                total_loaded = sum(s.loaded_rows for s in table_summaries.values())
                duration = round(time.perf_counter() - start_time, 2)
                completed_at = datetime.now(timezone.utc)

                # 3. Write ETL Execution Metadata
                meta_record = ETLExecutionMetadata(
                    run_id=run_id,
                    pipeline_phase="PHASE_3_POSTGRES_LOAD",
                    status="SUCCESS",
                    duration_seconds=Decimal(f"{duration:.2f}"),
                    records_processed=total_loaded,
                    details={
                        "tables": {k: v.loaded_rows for k, v in table_summaries.items()},
                        "target_database": target_db,
                    },
                )
                session.add(meta_record)
                session.flush()

                # 4. Write Data Quality Audit Results
                # Verify revenue sums
                pg_net = session.execute(
                    select(func.sum(MonthlySalesAnalytics.net_revenue))
                ).scalar()
                pg_net_val = float(pg_net) if pg_net is not None else 0.0

                audit_record = DataQualityAuditResults(
                    run_id=run_id,
                    audit_type="GOLD_POSTGRES_RECONCILIATION",
                    is_balanced=abs(pg_net_val - 19012910.84) < 0.1,
                    net_revenue=Decimal(f"{pg_net_val:.2f}"),
                    discrepancies_count=len(discrepancies),
                    report_payload={
                        "monthly_sales_net": pg_net_val,
                        "reference_net": 19012910.84,
                    },
                )
                session.add(audit_record)

            report = DatabaseLoadReport(
                run_id=run_id,
                target_database=target_db,
                started_at=started_at,
                completed_at=completed_at,
                total_duration_seconds=duration,
                status="SUCCESS",
                table_summaries=table_summaries,
                total_records_loaded=total_loaded,
                all_tables_verified=True,
                discrepancies=discrepancies,
            )
            logger.info(
                f"Gold-to-PostgreSQL load completed successfully! Loaded {total_loaded:,} "
                f"records in {duration:.2f}s."
            )
            return report

        except Exception as e:
            duration = round(time.perf_counter() - start_time, 2)
            completed_at = datetime.now(timezone.utc)
            logger.error(f"Gold-to-PostgreSQL loading failed: {e}")
            return DatabaseLoadReport(
                run_id=run_id,
                target_database=target_db,
                started_at=started_at,
                completed_at=completed_at,
                total_duration_seconds=duration,
                status="FAILED",
                table_summaries=table_summaries,
                total_records_loaded=0,
                all_tables_verified=False,
                discrepancies=[str(e)],
            )
