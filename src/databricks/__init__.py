"""Databricks & PySpark Module.

Responsible for:
- PySpark batch and streaming Lakehouse jobs.
- Delta Lake / Parquet table registration in Databricks Unity Catalog.
- Distributed feature computation and model training integration.
"""
from src.databricks.delta_lake import DeltaLakeManager

__all__ = ["DeltaLakeManager"]
