"""Silver layer transformation: explicit quality & transaction-status flagging."""
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.etl.config import ETLConfig

logger = logging.getLogger("ecommerce_platform.silver")


class TransactionStatus:
    """Categorical classification of every transaction record in Silver layer."""

    VALID_PURCHASE = "VALID_PURCHASE"
    CANCELLATION = "CANCELLATION"
    RETURN_ADJUSTMENT = "RETURN_ADJUSTMENT"
    INVENTORY_WRITE_OFF = "INVENTORY_WRITE_OFF"
    ZERO_PRICE_ENTRY = "ZERO_PRICE_ENTRY"
    DUPLICATE = "DUPLICATE"


class QualityFlag:
    """Data quality classification for governance and auditability."""

    PASS = "PASS"
    WARN_MISSING_CUSTOMER = "WARN_MISSING_CUSTOMER"
    WARN_NON_PRODUCT = "WARN_NON_PRODUCT"
    EXCLUDE_NON_REVENUE = "EXCLUDE_NON_REVENUE"


@dataclass
class SilverTransformationSummary:
    """Summary of Silver layer transformations and row allocations."""

    total_bronze_records: int
    total_silver_records: int
    valid_purchases_count: int
    cancellations_count: int
    return_adjustments_count: int
    inventory_write_offs_count: int
    zero_price_entries_count: int
    duplicates_count: int
    missing_customer_count: int
    ambiguous_cancellations_count: int
    gross_sales_gbp: float
    return_amount_gbp: float
    net_revenue_gbp: float
    silver_parquet_path: str
    silver_parquet_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SilverTransformer:
    """Applies business rules, deduplication, and quality flags while preserving 100% of Bronze records."""

    def __init__(self, config: Optional[ETLConfig] = None) -> None:
        self.config = config or ETLConfig()

    def transform_dataframe(self, bronze_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms Bronze DataFrame into Silver schema with explicit status flags.

        Args:
            bronze_df: Raw-standardized Bronze DataFrame.

        Returns:
            Silver DataFrame with audit flags, financial metrics, and temporal dimensions.
        """
        logger.info(f"Transforming {len(bronze_df):,} Bronze records into Silver schema...")
        silver_df = bronze_df.copy()

        # 1. Deduplication flag (identifies subsequent duplicate records)
        business_cols = [
            c for c in self.config.duplicate_business_cols if c in silver_df.columns
        ]
        silver_df["is_duplicate"] = silver_df.duplicated(
            subset=business_cols, keep="first"
        )

        # 2. Anomaly flags
        c_invoice = silver_df["invoice_no"].str.startswith(("C", "c"), na=False)
        neg_qty = silver_df["quantity"] < 0
        zero_or_neg_price = (silver_df["unit_price"] <= 0) | silver_df["unit_price"].isna()
        missing_cust = silver_df["customer_id"].isna()
        non_product_code = silver_df["stock_code"].str.upper().isin(
            self.config.non_product_stock_codes
        )

        silver_df["is_cancellation"] = c_invoice
        silver_df["is_return"] = neg_qty
        silver_df["is_invalid_price"] = zero_or_neg_price
        silver_df["is_missing_customer"] = missing_cust
        silver_df["is_non_product"] = non_product_code
        # Ambiguous positive-quantity credit-note adjustment (e.g. C496350)
        # Preserves original positive quantity and price while explicitly flagging
        silver_df["is_ambiguous_cancellation"] = c_invoice & (~neg_qty) & (silver_df["quantity"] > 0)

        # 3. Explicit Transaction Status Assignment (Mutually Exclusive Hierarchy)
        # Hierarchy:
        # 1) Duplicate records -> DUPLICATE
        # 2) Invoices starting with 'C' -> CANCELLATION
        # 3) Quantity < 0 and Unit Price <= 0 -> INVENTORY_WRITE_OFF
        # 4) Quantity < 0 and Unit Price > 0 -> RETURN_ADJUSTMENT
        # 5) Quantity >= 0 and Unit Price <= 0 -> ZERO_PRICE_ENTRY
        # 6) Quantity > 0 and Unit Price > 0 -> VALID_PURCHASE

        conditions = [
            silver_df["is_duplicate"],
            c_invoice,
            neg_qty & zero_or_neg_price,
            neg_qty & (~zero_or_neg_price),
            (~neg_qty) & zero_or_neg_price,
            (~neg_qty) & (~zero_or_neg_price),
        ]
        choices = [
            TransactionStatus.DUPLICATE,
            TransactionStatus.CANCELLATION,
            TransactionStatus.INVENTORY_WRITE_OFF,
            TransactionStatus.RETURN_ADJUSTMENT,
            TransactionStatus.ZERO_PRICE_ENTRY,
            TransactionStatus.VALID_PURCHASE,
        ]

        silver_df["transaction_status"] = np.select(
            conditions, choices, default="UNKNOWN"
        )

        # 4. Data Quality Flag
        quality_conditions = [
            silver_df["transaction_status"].isin(
                [
                    TransactionStatus.DUPLICATE,
                    TransactionStatus.INVENTORY_WRITE_OFF,
                    TransactionStatus.ZERO_PRICE_ENTRY,
                ]
            ),
            silver_df["is_missing_customer"],
            silver_df["is_non_product"],
        ]
        quality_choices = [
            QualityFlag.EXCLUDE_NON_REVENUE,
            QualityFlag.WARN_MISSING_CUSTOMER,
            QualityFlag.WARN_NON_PRODUCT,
        ]
        silver_df["quality_flag"] = np.select(
            quality_conditions, quality_choices, default=QualityFlag.PASS
        )

        # 5. Financial & Revenue Calculations
        # - line_total: raw quantity * unit_price (signed)
        # - gross_sales: revenue from valid purchases only
        # - return_amount: absolute value of credit from cancellations and returns
        # - net_revenue: gross_sales - return_amount (commercial transactions only)
        silver_df["line_total"] = (
            silver_df["quantity"].astype(float) * silver_df["unit_price"]
        )

        is_valid = silver_df["transaction_status"] == TransactionStatus.VALID_PURCHASE
        is_canc = silver_df["transaction_status"] == TransactionStatus.CANCELLATION
        is_ret = silver_df["transaction_status"] == TransactionStatus.RETURN_ADJUSTMENT

        silver_df["gross_sales"] = np.where(is_valid, silver_df["line_total"], 0.0)
        silver_df["return_amount"] = np.where(
            is_canc | is_ret,
            np.abs(silver_df["quantity"].astype(float)) * silver_df["unit_price"],
            0.0,
        )
        silver_df["net_revenue"] = silver_df["gross_sales"] - silver_df["return_amount"]

        # 6. Temporal Dimensions
        invoice_dt = pd.to_datetime(silver_df["invoice_date"])
        silver_df["invoice_year_month"] = invoice_dt.dt.strftime("%Y-%m")
        silver_df["invoice_year"] = invoice_dt.dt.year.astype("Int32")
        silver_df["invoice_month"] = invoice_dt.dt.month.astype("Int32")
        silver_df["invoice_day_of_week"] = invoice_dt.dt.day_name()
        silver_df["invoice_hour"] = invoice_dt.dt.hour.astype("Int32")

        logger.info(
            f"Silver transformation complete. Status breakdown: "
            f"{silver_df['transaction_status'].value_counts().to_dict()}"
        )
        return silver_df

    def process_and_save(self) -> Tuple[pd.DataFrame, SilverTransformationSummary]:
        """Loads Bronze dataset, runs Silver transformations, and atomically writes Parquet."""
        bronze_path = self.config.bronze_parquet_path
        if not bronze_path.exists():
            raise FileNotFoundError(f"Bronze Parquet not found at: {bronze_path}")

        bronze_df = pd.read_parquet(bronze_path)
        total_bronze = len(bronze_df)

        silver_df = self.transform_dataframe(bronze_df)
        total_silver = len(silver_df)

        # Verification check: 100% preservation
        assert (
            total_bronze == total_silver
        ), f"Row count mismatch! Bronze={total_bronze:,}, Silver={total_silver:,}"

        # Write atomically using PyArrow Snappy
        dest_path = self.config.silver_parquet_path
        temp_file = dest_path.parent / f".tmp_{dest_path.name}"
        table = pa.Table.from_pandas(silver_df, preserve_index=False)
        pq.write_table(table, temp_file, compression="snappy")
        if dest_path.exists():
            dest_path.unlink()
        temp_file.replace(dest_path)

        file_size = dest_path.stat().st_size
        logger.info(
            f"Saved Silver Parquet: {dest_path} ({total_silver:,} rows, {file_size:,} bytes)"
        )

        status_counts = silver_df["transaction_status"].value_counts().to_dict()
        summary = SilverTransformationSummary(
            total_bronze_records=total_bronze,
            total_silver_records=total_silver,
            valid_purchases_count=int(
                status_counts.get(TransactionStatus.VALID_PURCHASE, 0)
            ),
            cancellations_count=int(status_counts.get(TransactionStatus.CANCELLATION, 0)),
            return_adjustments_count=int(
                status_counts.get(TransactionStatus.RETURN_ADJUSTMENT, 0)
            ),
            inventory_write_offs_count=int(
                status_counts.get(TransactionStatus.INVENTORY_WRITE_OFF, 0)
            ),
            zero_price_entries_count=int(
                status_counts.get(TransactionStatus.ZERO_PRICE_ENTRY, 0)
            ),
            duplicates_count=int(status_counts.get(TransactionStatus.DUPLICATE, 0)),
            missing_customer_count=int(silver_df["is_missing_customer"].sum()),
            ambiguous_cancellations_count=int(silver_df["is_ambiguous_cancellation"].sum()),
            gross_sales_gbp=round(float(silver_df["gross_sales"].sum()), 2),
            return_amount_gbp=round(float(silver_df["return_amount"].sum()), 2),
            net_revenue_gbp=round(float(silver_df["net_revenue"].sum()), 2),
            silver_parquet_path=str(dest_path),
            silver_parquet_size_bytes=file_size,
        )

        return silver_df, summary
