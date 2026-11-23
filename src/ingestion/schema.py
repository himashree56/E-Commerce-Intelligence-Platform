"""Schema definitions, standardization, and validation rules for Bronze ingestion."""
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("ecommerce_platform.schema")


class RawSchema:
    """Expected raw dataset column names and attributes from UCI Online Retail II."""

    INVOICE = "Invoice"
    STOCK_CODE = "StockCode"
    DESCRIPTION = "Description"
    QUANTITY = "Quantity"
    INVOICE_DATE = "InvoiceDate"
    PRICE = "Price"
    CUSTOMER_ID = "Customer ID"
    COUNTRY = "Country"

    REQUIRED_COLUMNS: List[str] = [
        INVOICE,
        STOCK_CODE,
        DESCRIPTION,
        QUANTITY,
        INVOICE_DATE,
        PRICE,
        CUSTOMER_ID,
        COUNTRY,
    ]


class StandardizedSchema:
    """Standardized column names for the Bronze layer."""

    INVOICE_NO = "invoice_no"
    STOCK_CODE = "stock_code"
    DESCRIPTION = "description"
    QUANTITY = "quantity"
    INVOICE_DATE = "invoice_date"
    UNIT_PRICE = "unit_price"
    CUSTOMER_ID = "customer_id"
    COUNTRY = "country"

    # Ingestion Metadata Columns
    SHEET_NAME = "sheet_name"
    INGESTION_TIMESTAMP = "ingestion_timestamp"
    IS_CANCELLATION = "is_cancellation"

    COLUMN_MAPPING: Dict[str, str] = {
        RawSchema.INVOICE: INVOICE_NO,
        RawSchema.STOCK_CODE: STOCK_CODE,
        RawSchema.DESCRIPTION: DESCRIPTION,
        RawSchema.QUANTITY: QUANTITY,
        RawSchema.INVOICE_DATE: INVOICE_DATE,
        RawSchema.PRICE: UNIT_PRICE,
        RawSchema.CUSTOMER_ID: CUSTOMER_ID,
        RawSchema.COUNTRY: COUNTRY,
    }

    REQUIRED_BRONZE_COLUMNS: List[str] = [
        INVOICE_NO,
        STOCK_CODE,
        DESCRIPTION,
        QUANTITY,
        INVOICE_DATE,
        UNIT_PRICE,
        CUSTOMER_ID,
        COUNTRY,
        SHEET_NAME,
        INGESTION_TIMESTAMP,
        IS_CANCELLATION,
    ]


@dataclass
class ValidationReport:
    """Detailed summary of data quality, completeness, and schema validation."""

    sheet_name: str
    total_rows: int
    total_columns: int
    column_names: List[str]
    column_dtypes: Dict[str, str]
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    duplicate_rows_count: int
    total_unique_invoices: int
    cancellation_invoices_count: int
    negative_quantity_count: int
    zero_quantity_count: int
    positive_quantity_count: int
    missing_customer_count: int
    min_invoice_date: Optional[str]
    max_invoice_date: Optional[str]
    countries_count: int
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to a serializable dictionary."""
        return asdict(self)


class IngestionValidator:
    """Validates raw and standardized datasets against strict schema and data quality expectations."""

    @staticmethod
    def validate_raw_columns(df: pd.DataFrame, sheet_name: str) -> Tuple[bool, List[str]]:
        """Verifies that all required raw columns exist in the DataFrame.

        Args:
            df: Input raw DataFrame.
            sheet_name: Name of workbook sheet.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []
        actual_cols = set(df.columns)
        expected_cols = set(RawSchema.REQUIRED_COLUMNS)

        missing = expected_cols - actual_cols
        if missing:
            msg = f"Sheet '{sheet_name}' is missing expected raw columns: {sorted(list(missing))}"
            logger.error(msg)
            errors.append(msg)

        return (len(errors) == 0, errors)

    @staticmethod
    def standardize_columns(
        df: pd.DataFrame,
        sheet_name: str,
        ingestion_timestamp: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Standardizes column names and adds ingestion audit columns without mutating the raw input.

        Args:
            df: Input raw DataFrame (will not be modified).
            sheet_name: Originating sheet name.
            ingestion_timestamp: UTC timestamp of ingestion.

        Returns:
            A new standardized DataFrame ready for Bronze ingestion.
        """
        # Create an explicit shallow/deep copy to avoid modifying original raw data
        bronze_df = df.copy()

        # Rename columns according to mapping
        rename_map = {
            col: StandardizedSchema.COLUMN_MAPPING[col]
            for col in bronze_df.columns
            if col in StandardizedSchema.COLUMN_MAPPING
        }
        bronze_df = bronze_df.rename(columns=rename_map)

        # Standardize data types without dropping anomalies or nulls
        # 1. Invoice No as string
        bronze_df[StandardizedSchema.INVOICE_NO] = (
            bronze_df[StandardizedSchema.INVOICE_NO].astype(str).str.strip()
        )

        # 2. Stock Code as string
        bronze_df[StandardizedSchema.STOCK_CODE] = (
            bronze_df[StandardizedSchema.STOCK_CODE].astype(str).str.strip()
        )

        # 3. Description: string / nullable
        bronze_df[StandardizedSchema.DESCRIPTION] = bronze_df[
            StandardizedSchema.DESCRIPTION
        ].astype("string")

        # 4. Quantity: numeric (preserve negative quantities!)
        bronze_df[StandardizedSchema.QUANTITY] = pd.to_numeric(
            bronze_df[StandardizedSchema.QUANTITY], errors="coerce"
        ).astype("Int64")

        # 5. Invoice Date: datetime
        bronze_df[StandardizedSchema.INVOICE_DATE] = pd.to_datetime(
            bronze_df[StandardizedSchema.INVOICE_DATE], errors="coerce"
        )

        # 6. Unit Price: float64
        bronze_df[StandardizedSchema.UNIT_PRICE] = pd.to_numeric(
            bronze_df[StandardizedSchema.UNIT_PRICE], errors="coerce"
        ).astype("float64")

        # 7. Customer ID: nullable Int64 (preserving nulls without coercing to float)
        bronze_df[StandardizedSchema.CUSTOMER_ID] = pd.to_numeric(
            bronze_df[StandardizedSchema.CUSTOMER_ID], errors="coerce"
        ).astype("Int64")

        # 8. Country: string
        bronze_df[StandardizedSchema.COUNTRY] = bronze_df[
            StandardizedSchema.COUNTRY
        ].astype("string")

        # Audit / Metadata columns
        ts = ingestion_timestamp or datetime.now(timezone.utc)
        bronze_df[StandardizedSchema.SHEET_NAME] = sheet_name
        bronze_df[StandardizedSchema.INGESTION_TIMESTAMP] = ts

        # Determine cancellations: invoice_no starting with 'C' or 'c' OR quantity < 0
        bronze_df[StandardizedSchema.IS_CANCELLATION] = (
            bronze_df[StandardizedSchema.INVOICE_NO].str.startswith(("C", "c"))
            | (bronze_df[StandardizedSchema.QUANTITY] < 0)
        ).astype(bool)

        return bronze_df

    @classmethod
    def generate_validation_report(
        cls, df: pd.DataFrame, sheet_name: str
    ) -> ValidationReport:
        """Computes comprehensive schema, missing value, duplicate, and transaction metrics.

        Args:
            df: Standardized DataFrame.
            sheet_name: Name of the sheet.

        Returns:
            ValidationReport instance.
        """
        errors = []
        total_rows = int(len(df))
        total_cols = int(len(df.columns))
        col_names = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Verify required bronze columns
        missing_required = set(StandardizedSchema.REQUIRED_BRONZE_COLUMNS) - set(
            col_names
        )
        if missing_required:
            errors.append(f"Missing required Bronze columns: {missing_required}")

        # Missing values
        missing_counts = {
            col: int(df[col].isna().sum()) for col in df.columns
        }
        missing_pcts = {
            col: round(float((missing_counts[col] / total_rows) * 100), 2)
            if total_rows > 0
            else 0.0
            for col in df.columns
        }

        # Duplicates across primary business fields
        business_cols = [
            StandardizedSchema.INVOICE_NO,
            StandardizedSchema.STOCK_CODE,
            StandardizedSchema.QUANTITY,
            StandardizedSchema.INVOICE_DATE,
            StandardizedSchema.UNIT_PRICE,
            StandardizedSchema.CUSTOMER_ID,
        ]
        available_business_cols = [c for c in business_cols if c in df.columns]
        duplicate_rows = int(df.duplicated(subset=available_business_cols).sum())

        # Unique invoices & cancellations
        total_unique_invoices = int(
            df[StandardizedSchema.INVOICE_NO].nunique()
        )
        cancellations = int(
            df[StandardizedSchema.IS_CANCELLATION].sum()
        )

        # Quantity checks (strictly preserving negative quantities)
        quantities = df[StandardizedSchema.QUANTITY]
        neg_qty = int((quantities < 0).sum())
        zero_qty = int((quantities == 0).sum())
        pos_qty = int((quantities > 0).sum())

        # Customer ID checks (preserving nulls)
        missing_cust = missing_counts.get(StandardizedSchema.CUSTOMER_ID, 0)

        # Date range
        min_date = None
        max_date = None
        if StandardizedSchema.INVOICE_DATE in df.columns:
            valid_dates = df[StandardizedSchema.INVOICE_DATE].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().isoformat()
                max_date = valid_dates.max().isoformat()

        # Country count
        countries_count = 0
        if StandardizedSchema.COUNTRY in df.columns:
            countries_count = int(df[StandardizedSchema.COUNTRY].nunique())

        # Preservation checks
        logger.info(
            f"Preserved data metrics for '{sheet_name}': "
            f"Cancellations={cancellations:,}, Negative Qty={neg_qty:,}, "
            f"Missing Customers={missing_cust:,}, Duplicates={duplicate_rows:,}"
        )

        is_valid = len(errors) == 0

        return ValidationReport(
            sheet_name=sheet_name,
            total_rows=total_rows,
            total_columns=total_cols,
            column_names=col_names,
            column_dtypes=dtypes,
            missing_counts=missing_counts,
            missing_percentages=missing_pcts,
            duplicate_rows_count=duplicate_rows,
            total_unique_invoices=total_unique_invoices,
            cancellation_invoices_count=cancellations,
            negative_quantity_count=neg_qty,
            zero_quantity_count=zero_qty,
            positive_quantity_count=pos_qty,
            missing_customer_count=missing_cust,
            min_invoice_date=min_date,
            max_invoice_date=max_date,
            countries_count=countries_count,
            is_valid=is_valid,
            validation_errors=errors,
        )
