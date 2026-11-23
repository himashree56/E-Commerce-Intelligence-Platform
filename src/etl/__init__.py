"""ETL Module (Phase 2).

Responsible for:
- Silver layer transformation: data cleaning, explicit transaction classification, and deduplication.
- Business rule enforcement: cancellations, negative quantities, returns reconciliation, accounting adjustments.
- Missing customer handling and auditability flags.
- Gold analytical modeling (Monthly sales, Customer RFM, Product performance, Country revenue, Customer history).
- Cross-layer reconciliation engine.
- PySpark distributed Lakehouse pipeline.
"""
from src.etl.config import ETLConfig
from src.etl.gold import GoldTransformationSummary, GoldTransformer
from src.etl.pipeline import Phase2Pipeline, Phase2PipelineSummary
from src.etl.pyspark_pipeline import PySparkLakehousePipeline
from src.etl.reconciliation import ReconciliationAuditor, ReconciliationReport
from src.etl.silver import (
    QualityFlag,
    SilverTransformationSummary,
    SilverTransformer,
    TransactionStatus,
)
from src.etl.spark_session import get_spark_session

__all__ = [
    "ETLConfig",
    "SilverTransformer",
    "TransactionStatus",
    "QualityFlag",
    "SilverTransformationSummary",
    "GoldTransformer",
    "GoldTransformationSummary",
    "ReconciliationAuditor",
    "ReconciliationReport",
    "Phase2Pipeline",
    "Phase2PipelineSummary",
    "PySparkLakehousePipeline",
    "get_spark_session",
]
