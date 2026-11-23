"""Databricks Delta Lake pipeline integration and Unity Catalog DDL generator."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("ecommerce_platform.databricks")


class DeltaLakeManager:
    """Manages Delta Lake format export, Z-ordering, and Unity Catalog DDL generation."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("data/delta")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_unity_catalog_ddl(self) -> str:
        """Generates Databricks Unity Catalog SQL DDL for Bronze, Silver, and Gold tables.

        Returns:
            SQL DDL script string.
        """
        ddl = """-- Databricks Unity Catalog DDL for E-Commerce Intelligence Platform
CREATE CATALOG IF NOT EXISTS ecommerce_catalog;
USE CATALOG ecommerce_catalog;

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Bronze Table
CREATE TABLE IF NOT EXISTS bronze.online_retail_raw (
    invoice_no STRING,
    stock_code STRING,
    description STRING,
    quantity BIGINT,
    invoice_date TIMESTAMP,
    unit_price DOUBLE,
    customer_id BIGINT,
    country STRING,
    sheet_name STRING,
    ingestion_timestamp TIMESTAMP,
    is_cancellation BOOLEAN
)
USING DELTA
LOCATION 'dbfs:/mnt/lakehouse/bronze/online_retail/'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

-- Silver Curated Table
CREATE TABLE IF NOT EXISTS silver.online_retail_transactions (
    invoice_no STRING,
    stock_code STRING,
    description STRING,
    quantity BIGINT,
    invoice_date TIMESTAMP,
    unit_price DOUBLE,
    customer_id BIGINT,
    country STRING,
    sheet_name STRING,
    ingestion_timestamp TIMESTAMP,
    is_cancellation BOOLEAN,
    is_ambiguous_cancellation BOOLEAN,
    is_duplicate BOOLEAN,
    is_return BOOLEAN,
    is_invalid_price BOOLEAN,
    is_missing_customer BOOLEAN,
    is_non_product BOOLEAN,
    transaction_status STRING,
    quality_flag STRING,
    line_total DOUBLE,
    gross_sales DOUBLE,
    return_amount DOUBLE,
    net_revenue DOUBLE,
    invoice_year_month STRING,
    invoice_year INT,
    invoice_month INT,
    invoice_day_of_week STRING,
    invoice_hour INT
)
USING DELTA
LOCATION 'dbfs:/mnt/lakehouse/silver/online_retail/'
PARTITIONED BY (invoice_year_month)
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);

-- Optimize Silver Table
OPTIMIZE silver.online_retail_transactions
ZORDER BY (customer_id, invoice_date, stock_code);

-- Gold Tables
CREATE TABLE IF NOT EXISTS gold.monthly_sales
USING DELTA
LOCATION 'dbfs:/mnt/lakehouse/gold/monthly_sales/';

CREATE TABLE IF NOT EXISTS gold.customer_rfm
USING DELTA
LOCATION 'dbfs:/mnt/lakehouse/gold/customer_rfm/';

CREATE TABLE IF NOT EXISTS gold.product_performance
USING DELTA
LOCATION 'dbfs:/mnt/lakehouse/gold/product_performance/';

CREATE TABLE IF NOT EXISTS gold.country_revenue
USING DELTA
LOCATION 'dbfs:/mnt/lakehouse/gold/country_revenue/';

CREATE TABLE IF NOT EXISTS gold.customer_purchase_history
USING DELTA
LOCATION 'dbfs:/mnt/lakehouse/gold/customer_purchase_history/';
"""
        return ddl

    def export_unity_catalog_script(self, script_path: Path) -> Path:
        """Saves Unity Catalog DDL script to disk."""
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(self.generate_unity_catalog_ddl())
        logger.info(f"Saved Databricks Unity Catalog DDL to: {script_path}")
        return script_path
