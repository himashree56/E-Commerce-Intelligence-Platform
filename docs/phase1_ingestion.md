# Phase 1: Raw Data Ingestion & Bronze Layer

## 1. Overview
Phase 1 implements a production-grade ingestion pipeline for the **UCI Online Retail II** dataset. The dataset contains transaction records from a UK-based online retail store from 01/12/2009 to 09/12/2011 across two workbook sheets (`Year 2009-2010` and `Year 2010-2011`).

## 2. Ingestion Principles
1. **Raw Immutability**: The original raw dataset file is treated as read-only and never modified.
2. **Anomaly Preservation**: Bronze layer does NOT filter out anomalies. Cancellations (invoice starting with `C`), negative quantities, and missing customer identifiers are strictly preserved and flagged for downstream processing in Phase 2.
3. **Idempotence & Atomic Operations**: All writes to Parquet are staged via temporary files and atomically renamed, guaranteeing consistency and zero corruption upon repeated runs.
4. **Structured Auditability**: Every run produces structured logs and a run-specific JSON metadata document capturing row counts, null counts, duplicate counts, schema types, and execution durations.

## 3. Schema Standardization

| Raw Column Name | Standardized Bronze Name | Target Type | Nullable | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `Invoice` | `invoice_no` | string | No | Trimmed string |
| `StockCode` | `stock_code` | string | No | Trimmed string |
| `Description` | `description` | string | Yes | Cleaned string, preserves nulls |
| `Quantity` | `quantity` | Int64 | No | Preserves negative and zero values |
| `InvoiceDate` | `invoice_date` | datetime64[ns] | No | Parsed timestamp |
| `Price` | `unit_price` | float64 | No | Unit price in GBP |
| `Customer ID` | `customer_id` | Int64 | Yes | Preserves null customer identifiers |
| `Country` | `country` | string | No | Destination country |
| *Added* | `sheet_name` | string | No | Originating workbook sheet |
| *Added* | `ingestion_timestamp` | datetime64[ns, UTC] | No | UTC timestamp of pipeline run |
| *Added* | `is_cancellation` | bool | No | `True` if `invoice_no` starts with 'C' or `quantity < 0` |

## 4. Output Artifacts
- **Unified Bronze Parquet**: `data/bronze/online_retail_bronze.parquet`
- **Sheet 1 Parquet**: `data/bronze/online_retail_year_2009_2010.parquet`
- **Sheet 2 Parquet**: `data/bronze/online_retail_year_2010_2011.parquet`
- **Run Metadata**: `data/metadata/ingest_{YYYYMMDD_HHMMSS}.json`
- **Latest Metadata Pointer**: `data/metadata/latest_ingestion.json`
- **Structured Log**: `logs/ingestion.log`
