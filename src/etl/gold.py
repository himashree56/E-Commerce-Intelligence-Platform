"""Gold layer analytical models: Monthly Sales, RFM, Products, Country, and History."""
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.etl.config import ETLConfig
from src.etl.silver import TransactionStatus

logger = logging.getLogger("ecommerce_platform.gold")


@dataclass
class GoldTransformationSummary:
    """Metrics summary of all Gold analytical tables."""

    monthly_sales_rows: int
    customer_rfm_rows: int
    product_performance_rows: int
    country_revenue_rows: int
    customer_purchase_history_rows: int
    total_gold_net_revenue_gbp: float
    total_gold_gross_sales_gbp: float
    total_gold_return_amount_gbp: float
    tables_saved: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoldTransformer:
    """Builds aggregated business and predictive analytical tables from Silver layer."""

    def __init__(self, config: Optional[ETLConfig] = None) -> None:
        self.config = config or ETLConfig()

    def build_monthly_sales(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates monthly sales performance and return trends.

        Args:
            silver_df: Silver DataFrame.

        Returns:
            Monthly sales DataFrame sorted chronologically.
        """
        logger.info("Building Gold Monthly Sales table...")
        comm_df = silver_df[
            ~silver_df["is_duplicate"]
            & ~silver_df["transaction_status"].isin(
                [
                    TransactionStatus.INVENTORY_WRITE_OFF,
                    TransactionStatus.ZERO_PRICE_ENTRY,
                ]
            )
        ].copy()

        # Precompute boolean slices for fast, vectorized aggregation
        is_purchase = comm_df["transaction_status"] == TransactionStatus.VALID_PURCHASE
        is_canc = comm_df["transaction_status"] == TransactionStatus.CANCELLATION
        is_return_any = comm_df["transaction_status"].isin(
            [TransactionStatus.CANCELLATION, TransactionStatus.RETURN_ADJUSTMENT]
        )

        comm_df["valid_invoice"] = np.where(is_purchase, comm_df["invoice_no"], np.nan)
        comm_df["canc_invoice"] = np.where(is_canc, comm_df["invoice_no"], np.nan)
        comm_df["sold_qty"] = np.where(is_purchase, comm_df["quantity"], 0)
        comm_df["ret_qty"] = np.where(is_return_any, comm_df["quantity"].abs(), 0)

        monthly = (
            comm_df.groupby("invoice_year_month", as_index=False)
            .agg(
                invoice_year=("invoice_year", "first"),
                invoice_month=("invoice_month", "first"),
                raw_gross=("gross_sales", "sum"),
                raw_return=("return_amount", "sum"),
                valid_orders_count=("valid_invoice", "nunique"),
                cancelled_orders_count=("canc_invoice", "nunique"),
                distinct_customers=("customer_id", "nunique"),
                total_units_sold=("sold_qty", "sum"),
                units_returned=("ret_qty", "sum"),
            )
            .sort_values("invoice_year_month")
            .reset_index(drop=True)
        )

        # Fixed-precision cumulative penny alignment to prevent periodic rounding drift
        cum_gross = monthly["raw_gross"].cumsum().round(2)
        cum_return = monthly["raw_return"].cumsum().round(2)
        prev_cum_gross = cum_gross.shift(1).fillna(0.0)
        prev_cum_return = cum_return.shift(1).fillna(0.0)

        monthly["gross_sales"] = (cum_gross - prev_cum_gross).round(2)
        monthly["return_amount"] = (cum_return - prev_cum_return).round(2)
        monthly["net_revenue"] = (monthly["gross_sales"] - monthly["return_amount"]).round(2)
        monthly.drop(columns=["raw_gross", "raw_return"], inplace=True)

        monthly["net_units"] = (
            monthly["total_units_sold"] - monthly["units_returned"]
        )
        monthly["average_order_value"] = np.where(
            monthly["valid_orders_count"] > 0,
            (monthly["gross_sales"] / monthly["valid_orders_count"]).round(2),
            0.0,
        )

        return monthly.sort_values("invoice_year_month").reset_index(drop=True)

    def build_customer_rfm(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Builds Customer RFM (Recency, Frequency, Monetary) segmentation model.

        Args:
            silver_df: Silver DataFrame.

        Returns:
            Customer-level analytical table with RFM scores and segments.
        """
        logger.info("Building Gold Customer RFM table...")
        cust_df = silver_df[
            silver_df["customer_id"].notna()
            & ~silver_df["is_duplicate"]
            & ~silver_df["transaction_status"].isin(
                [
                    TransactionStatus.INVENTORY_WRITE_OFF,
                    TransactionStatus.ZERO_PRICE_ENTRY,
                ]
            )
        ].copy()

        max_date = pd.to_datetime(cust_df["invoice_date"]).max()
        ref_date = max_date + pd.Timedelta(days=1)

        is_purchase = cust_df["transaction_status"] == TransactionStatus.VALID_PURCHASE
        is_ret_any = cust_df["transaction_status"].isin(
            [TransactionStatus.CANCELLATION, TransactionStatus.RETURN_ADJUSTMENT]
        )
        cust_df["valid_invoice"] = np.where(is_purchase, cust_df["invoice_no"], np.nan)
        cust_df["ret_invoice"] = np.where(is_ret_any, cust_df["invoice_no"], np.nan)

        rfm = (
            cust_df.groupby("customer_id", as_index=False)
            .agg(
                primary_country=("country", lambda x: x.mode().iloc[0] if not x.empty else "Unknown"),
                first_purchase_date=("invoice_date", "min"),
                last_purchase_date=("invoice_date", "max"),
                frequency=("valid_invoice", "nunique"),
                monetary_gross=("gross_sales", lambda s: round(float(s.sum()), 2)),
                monetary_net=("net_revenue", lambda s: round(float(s.sum()), 2)),
                return_amount=("return_amount", lambda s: round(float(s.sum()), 2)),
                return_orders_count=("ret_invoice", "nunique"),
            )
        )

        rfm["recency_days"] = (ref_date - pd.to_datetime(rfm["last_purchase_date"])).dt.days.astype(int)

        # Percentile-based robust scoring (works from 1 row to millions of rows)
        def score_percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
            if len(series) <= 1 or series.nunique() <= 1:
                return pd.Series(3, index=series.index, dtype=int)
            ranks = series.rank(pct=True, method="first", ascending=ascending)
            return (ranks * 5).apply(np.ceil).clip(lower=1, upper=5).astype(int)

        rfm["r_score"] = score_percentile(rfm["recency_days"], ascending=False)
        rfm["f_score"] = score_percentile(rfm["frequency"], ascending=True)
        rfm["m_score"] = score_percentile(rfm["monetary_net"], ascending=True)

        rfm["rfm_cell"] = (
            rfm["r_score"].astype(str)
            + rfm["f_score"].astype(str)
            + rfm["m_score"].astype(str)
        )

        def assign_segment(row) -> str:
            r = row["r_score"]
            f = row["f_score"]
            if r >= 4 and f >= 4:
                return "Champions"
            elif r >= 3 and f >= 3:
                return "Loyal Customers"
            elif r >= 4 and f <= 2:
                return "New / Promising"
            elif r == 3 and f <= 2:
                return "Potential Loyalists"
            elif r == 2 and f >= 3:
                return "At Risk"
            elif r <= 2 and f >= 4:
                return "Cant Lose Them"
            elif r == 2 and f <= 2:
                return "Customers Needing Attention"
            else:
                return "Hibernating / Lost"

        rfm["customer_segment"] = rfm.apply(assign_segment, axis=1)

        return rfm.sort_values("monetary_net", ascending=False).reset_index(drop=True)

    def build_product_performance(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates product catalog performance and return rates.

        Args:
            silver_df: Silver DataFrame.

        Returns:
            Product-level analytical table.
        """
        logger.info("Building Gold Product Performance table...")
        prod_df = silver_df[
            ~silver_df["is_duplicate"]
            & ~silver_df["is_non_product"]
            & ~silver_df["transaction_status"].isin(
                [
                    TransactionStatus.INVENTORY_WRITE_OFF,
                    TransactionStatus.ZERO_PRICE_ENTRY,
                ]
            )
        ].copy()

        is_purchase = prod_df["transaction_status"] == TransactionStatus.VALID_PURCHASE
        is_return_any = prod_df["transaction_status"].isin(
            [TransactionStatus.CANCELLATION, TransactionStatus.RETURN_ADJUSTMENT]
        )
        prod_df["valid_invoice"] = np.where(is_purchase, prod_df["invoice_no"], np.nan)
        prod_df["sold_qty"] = np.where(is_purchase, prod_df["quantity"], 0)
        prod_df["ret_qty"] = np.where(is_return_any, prod_df["quantity"].abs(), 0)

        products = (
            prod_df.groupby("stock_code", as_index=False)
            .agg(
                description=("description", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "Unknown"),
                gross_sales=("gross_sales", lambda s: round(float(s.sum()), 2)),
                return_amount=("return_amount", lambda s: round(float(s.sum()), 2)),
                net_revenue=("net_revenue", lambda s: round(float(s.sum()), 2)),
                units_sold=("sold_qty", "sum"),
                units_returned=("ret_qty", "sum"),
                distinct_orders=("valid_invoice", "nunique"),
                distinct_customers=("customer_id", "nunique"),
            )
        )

        products["net_units"] = products["units_sold"] - products["units_returned"]
        products["return_rate_pct"] = np.where(
            products["units_sold"] > 0,
            ((products["units_returned"] / products["units_sold"]) * 100).round(2),
            0.0,
        )

        return products.sort_values("net_revenue", ascending=False).reset_index(drop=True)

    def build_country_revenue(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates geographic revenue, return volume, and market share.

        Args:
            silver_df: Silver DataFrame.

        Returns:
            Country-level performance table.
        """
        logger.info("Building Gold Country Revenue table...")
        comm_df = silver_df[
            ~silver_df["is_duplicate"]
            & ~silver_df["transaction_status"].isin(
                [
                    TransactionStatus.INVENTORY_WRITE_OFF,
                    TransactionStatus.ZERO_PRICE_ENTRY,
                ]
            )
        ].copy()

        total_net_all = comm_df["net_revenue"].sum()

        is_purchase = comm_df["transaction_status"] == TransactionStatus.VALID_PURCHASE
        is_canc = comm_df["transaction_status"] == TransactionStatus.CANCELLATION
        comm_df["valid_invoice"] = np.where(is_purchase, comm_df["invoice_no"], np.nan)
        comm_df["canc_invoice"] = np.where(is_canc, comm_df["invoice_no"], np.nan)

        country_df = (
            comm_df.groupby("country", as_index=False)
            .agg(
                gross_sales=("gross_sales", lambda s: round(float(s.sum()), 2)),
                return_amount=("return_amount", lambda s: round(float(s.sum()), 2)),
                net_revenue=("net_revenue", lambda s: round(float(s.sum()), 2)),
                valid_orders_count=("valid_invoice", "nunique"),
                cancelled_orders_count=("canc_invoice", "nunique"),
                distinct_customers=("customer_id", "nunique"),
            )
        )

        country_df["market_share_pct"] = np.where(
            total_net_all != 0,
            ((country_df["net_revenue"] / total_net_all) * 100).round(2),
            0.0,
        )

        return country_df.sort_values("net_revenue", ascending=False).reset_index(drop=True)

    def build_customer_purchase_history(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Builds order-level timeline per customer for cohort & behavioral analytics.

        Args:
            silver_df: Silver DataFrame.

        Returns:
            Customer order history table.
        """
        logger.info("Building Gold Customer Purchase History table...")
        cust_orders = silver_df[
            silver_df["customer_id"].notna()
            & ~silver_df["is_duplicate"]
            & ~silver_df["transaction_status"].isin(
                [
                    TransactionStatus.INVENTORY_WRITE_OFF,
                    TransactionStatus.ZERO_PRICE_ENTRY,
                ]
            )
        ].copy()

        history = (
            cust_orders.groupby(["customer_id", "invoice_no"], as_index=False)
            .agg(
                invoice_date=("invoice_date", "first"),
                country=("country", "first"),
                is_cancellation=("is_cancellation", "first"),
                transaction_status=("transaction_status", "first"),
                line_items_count=("stock_code", "count"),
                total_quantity=("quantity", "sum"),
                order_gross_sales=("gross_sales", lambda s: round(float(s.sum()), 2)),
                order_return_amount=("return_amount", lambda s: round(float(s.sum()), 2)),
                order_net_revenue=("net_revenue", lambda s: round(float(s.sum()), 2)),
            )
        )

        return history.sort_values(["customer_id", "invoice_date"], ascending=[True, False]).reset_index(drop=True)

    def process_and_save_all(
        self, silver_df: pd.DataFrame
    ) -> Tuple[Dict[str, pd.DataFrame], GoldTransformationSummary]:
        """Generates and persists all 5 Gold analytical tables as Parquet."""
        monthly_df = self.build_monthly_sales(silver_df)
        rfm_df = self.build_customer_rfm(silver_df)
        prod_df = self.build_product_performance(silver_df)
        country_df = self.build_country_revenue(silver_df)
        history_df = self.build_customer_purchase_history(silver_df)

        tables = {
            "gold_monthly_sales": (monthly_df, self.config.gold_monthly_sales_path),
            "gold_customer_rfm": (rfm_df, self.config.gold_customer_rfm_path),
            "gold_product_performance": (
                prod_df,
                self.config.gold_product_performance_path,
            ),
            "gold_country_revenue": (country_df, self.config.gold_country_revenue_path),
            "gold_customer_purchase_history": (
                history_df,
                self.config.gold_customer_purchase_history_path,
            ),
        }

        saved_paths: Dict[str, str] = {}
        for name, (df, path) in tables.items():
            temp_p = path.parent / f".tmp_{path.name}"
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(tbl, temp_p, compression="snappy")
            if path.exists():
                path.unlink()
            temp_p.replace(path)
            saved_paths[name] = str(path)
            logger.info(
                f"Saved {name}: {path} ({len(df):,} rows, {path.stat().st_size:,} bytes)"
            )

        summary = GoldTransformationSummary(
            monthly_sales_rows=len(monthly_df),
            customer_rfm_rows=len(rfm_df),
            product_performance_rows=len(prod_df),
            country_revenue_rows=len(country_df),
            customer_purchase_history_rows=len(history_df),
            total_gold_net_revenue_gbp=round(float(monthly_df["net_revenue"].sum()), 2),
            total_gold_gross_sales_gbp=round(float(monthly_df["gross_sales"].sum()), 2),
            total_gold_return_amount_gbp=round(
                float(monthly_df["return_amount"].sum()), 2
            ),
            tables_saved=saved_paths,
        )

        return (
            {
                "monthly_sales": monthly_df,
                "customer_rfm": rfm_df,
                "product_performance": prod_df,
                "country_revenue": country_df,
                "customer_purchase_history": history_df,
            },
            summary,
        )
