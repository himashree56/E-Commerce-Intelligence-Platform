"""Native PySpark Bronze-to-Silver-to-Gold Pipeline."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.etl.config import ETLConfig
from src.etl.spark_session import get_spark_session

logger = logging.getLogger("ecommerce_platform.pyspark_pipeline")


class PySparkLakehousePipeline:
    """Executes Bronze -> Silver -> Gold transformations using native PySpark DataFrame API."""

    def __init__(self, config: Optional[ETLConfig] = None, spark_session=None) -> None:
        self.config = config or ETLConfig()
        self.spark = spark_session or get_spark_session()

    def build_silver(self, bronze_df):
        """Transforms Bronze PySpark DataFrame into Silver schema with audit flags."""
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        # Normalize timestamp if read as nanoseconds since epoch
        if dict(bronze_df.dtypes).get("invoice_date") in ["bigint", "long"]:
            bronze_df = bronze_df.withColumn(
                "invoice_date", (F.col("invoice_date") / 1e9).cast("timestamp")
            )

        window_spec = Window.partitionBy(
            "invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"
        ).orderBy("invoice_date")

        silver_df = (
            bronze_df
            .withColumn("row_num", F.row_number().over(window_spec))
            .withColumn("is_duplicate", F.col("row_num") > 1)
            .drop("row_num")
            .withColumn("is_cancellation", F.col("invoice_no").startswith("C") | F.col("invoice_no").startswith("c"))
            .withColumn("is_return", F.col("quantity") < 0)
            .withColumn("is_invalid_price", (F.col("unit_price") <= 0) | F.col("unit_price").isNull())
            .withColumn("is_missing_customer", F.col("customer_id").isNull())
            .withColumn(
                "is_non_product",
                F.upper(F.col("stock_code")).isin(self.config.non_product_stock_codes)
            )
            .withColumn(
                "is_ambiguous_cancellation",
                (F.col("invoice_no").startswith("C") | F.col("invoice_no").startswith("c")) & (F.col("quantity") > 0)
            )
            .withColumn(
                "transaction_status",
                F.when(F.col("is_duplicate"), F.lit("DUPLICATE"))
                .when(F.col("is_cancellation"), F.lit("CANCELLATION"))
                .when(F.col("is_return") & F.col("is_invalid_price"), F.lit("INVENTORY_WRITE_OFF"))
                .when(F.col("is_return") & ~F.col("is_invalid_price"), F.lit("RETURN_ADJUSTMENT"))
                .when(~F.col("is_return") & F.col("is_invalid_price"), F.lit("ZERO_PRICE_ENTRY"))
                .otherwise(F.lit("VALID_PURCHASE"))
            )
            .withColumn(
                "quality_flag",
                F.when(F.col("transaction_status").isin("DUPLICATE", "INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY"), F.lit("EXCLUDE_NON_REVENUE"))
                .when(F.col("is_missing_customer"), F.lit("WARN_MISSING_CUSTOMER"))
                .when(F.col("is_non_product"), F.lit("WARN_NON_PRODUCT"))
                .otherwise(F.lit("PASS"))
            )
            .withColumn("line_total", (F.col("quantity") * F.col("unit_price")).cast("double"))
            .withColumn(
                "gross_sales",
                F.when(F.col("transaction_status") == "VALID_PURCHASE", F.col("line_total")).otherwise(F.lit(0.0))
            )
            .withColumn(
                "return_amount",
                F.when(F.col("transaction_status").isin("CANCELLATION", "RETURN_ADJUSTMENT"), F.abs(F.col("quantity")) * F.col("unit_price")).otherwise(F.lit(0.0))
            )
            .withColumn("net_revenue", F.col("gross_sales") - F.col("return_amount"))
            .withColumn("invoice_year_month", F.date_format("invoice_date", "yyyy-MM"))
            .withColumn("invoice_year", F.year("invoice_date"))
            .withColumn("invoice_month", F.month("invoice_date"))
            .withColumn("invoice_day_of_week", F.date_format("invoice_date", "EEEE"))
            .withColumn("invoice_hour", F.hour("invoice_date"))
        )
        return silver_df

    def build_monthly_sales(self, silver_df):
        """Builds Gold Monthly Sales DataFrame with fixed-precision cumulative penny alignment."""
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        comm = silver_df.filter(
            ~F.col("is_duplicate") & ~F.col("transaction_status").isin("INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY")
        )
        base = (
            comm.groupBy("invoice_year_month", "invoice_year", "invoice_month")
            .agg(
                F.sum("gross_sales").alias("raw_gross"),
                F.sum("return_amount").alias("raw_return"),
                F.countDistinct(F.when(F.col("transaction_status") == "VALID_PURCHASE", F.col("invoice_no"))).alias("valid_orders_count"),
                F.countDistinct(F.when(F.col("transaction_status") == "CANCELLATION", F.col("invoice_no"))).alias("cancelled_orders_count"),
                F.countDistinct("customer_id").alias("distinct_customers"),
                F.sum(F.when(F.col("transaction_status") == "VALID_PURCHASE", F.col("quantity"))).alias("total_units_sold"),
                F.sum(F.when(F.col("transaction_status").isin("CANCELLATION", "RETURN_ADJUSTMENT"), F.abs(F.col("quantity")))).alias("units_returned")
            )
        )

        w_cum = Window.orderBy("invoice_year_month").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        w_lag = Window.orderBy("invoice_year_month")

        return (
            base
            .withColumn("cum_gross", F.round(F.sum("raw_gross").over(w_cum), 2))
            .withColumn("cum_return", F.round(F.sum("raw_return").over(w_cum), 2))
            .withColumn("prev_cum_gross", F.coalesce(F.lag("cum_gross", 1).over(w_lag), F.lit(0.0)))
            .withColumn("prev_cum_return", F.coalesce(F.lag("cum_return", 1).over(w_lag), F.lit(0.0)))
            .withColumn("gross_sales", F.round(F.col("cum_gross") - F.col("prev_cum_gross"), 2))
            .withColumn("return_amount", F.round(F.col("cum_return") - F.col("prev_cum_return"), 2))
            .withColumn("net_revenue", F.round(F.col("gross_sales") - F.col("return_amount"), 2))
            .withColumn("net_units", F.col("total_units_sold") - F.col("units_returned"))
            .withColumn(
                "average_order_value",
                F.when(F.col("valid_orders_count") > 0, F.round(F.col("gross_sales") / F.col("valid_orders_count"), 2)).otherwise(F.lit(0.0))
            )
            .drop("raw_gross", "raw_return", "cum_gross", "cum_return", "prev_cum_gross", "prev_cum_return")
            .orderBy("invoice_year_month")
        )

    def build_customer_rfm(self, silver_df):
        """Builds Gold Customer RFM DataFrame."""
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        comm = silver_df.filter(
            ~F.col("is_duplicate") & ~F.col("transaction_status").isin("INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY")
            & F.col("customer_id").isNotNull()
        )
        max_date_row = comm.select(F.max("invoice_date")).collect()
        max_date = max_date_row[0][0] if max_date_row and max_date_row[0][0] else None

        base_rfm = (
            comm.groupBy("customer_id")
            .agg(
                F.first("country").alias("primary_country"),
                F.min("invoice_date").alias("first_purchase_date"),
                F.max("invoice_date").alias("last_purchase_date"),
                F.countDistinct(F.when(F.col("transaction_status") == "VALID_PURCHASE", F.col("invoice_no"))).alias("frequency"),
                F.round(F.sum("gross_sales"), 2).alias("monetary_gross"),
                F.round(F.sum("net_revenue"), 2).alias("monetary_net"),
                F.round(F.sum("return_amount"), 2).alias("return_amount"),
                F.countDistinct(F.when(F.col("transaction_status").isin("CANCELLATION", "RETURN_ADJUSTMENT"), F.col("invoice_no"))).alias("return_orders_count")
            )
        )
        if max_date:
            base_rfm = base_rfm.withColumn("recency_days", F.datediff(F.lit(max_date), F.col("last_purchase_date")))
        else:
            base_rfm = base_rfm.withColumn("recency_days", F.lit(0))

        r_win = Window.orderBy(F.col("recency_days").desc())
        f_win = Window.orderBy(F.col("frequency").asc())
        m_win = Window.orderBy(F.col("monetary_net").asc())

        return (
            base_rfm
            .withColumn("r_score", F.ntile(5).over(r_win))
            .withColumn("f_score", F.ntile(5).over(f_win))
            .withColumn("m_score", F.ntile(5).over(m_win))
            .withColumn("rfm_cell", F.concat(F.col("r_score"), F.col("f_score"), F.col("m_score")))
            .withColumn(
                "customer_segment",
                F.when((F.col("r_score") >= 4) & (F.col("f_score") >= 4), F.lit("Champions"))
                .when((F.col("r_score") >= 3) & (F.col("f_score") >= 3), F.lit("Loyal Customers"))
                .when((F.col("r_score") >= 4) & (F.col("f_score") <= 2), F.lit("New / Promising"))
                .when((F.col("r_score") == 3) & (F.col("f_score") <= 2), F.lit("Potential Loyalists"))
                .when((F.col("r_score") == 2) & (F.col("f_score") >= 3), F.lit("At Risk"))
                .when((F.col("r_score") <= 2) & (F.col("f_score") >= 4), F.lit("Cant Lose Them"))
                .when((F.col("r_score") == 2) & (F.col("f_score") <= 2), F.lit("Customers Needing Attention"))
                .otherwise(F.lit("Hibernating / Lost"))
            )
            .orderBy(F.col("monetary_net").desc())
        )

    def build_product_performance(self, silver_df):
        """Builds Gold Product Performance DataFrame."""
        from pyspark.sql import functions as F

        comm = silver_df.filter(
            ~F.col("is_duplicate") & ~F.col("is_non_product")
            & ~F.col("transaction_status").isin("INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY")
        )
        return (
            comm.groupBy("stock_code")
            .agg(
                F.first("description", ignorenulls=True).alias("description"),
                F.coalesce(F.round(F.sum("gross_sales"), 2), F.lit(0.0)).alias("gross_sales"),
                F.coalesce(F.round(F.sum("return_amount"), 2), F.lit(0.0)).alias("return_amount"),
                F.coalesce(F.round(F.sum("net_revenue"), 2), F.lit(0.0)).alias("net_revenue"),
                F.coalesce(F.sum(F.when(F.col("transaction_status") == "VALID_PURCHASE", F.col("quantity"))), F.lit(0)).alias("units_sold"),
                F.coalesce(F.sum(F.when(F.col("transaction_status").isin("CANCELLATION", "RETURN_ADJUSTMENT"), F.abs(F.col("quantity")))), F.lit(0)).alias("units_returned"),
                F.countDistinct(F.when(F.col("transaction_status") == "VALID_PURCHASE", F.col("invoice_no"))).alias("distinct_orders"),
                F.countDistinct("customer_id").alias("distinct_customers")
            )
            .withColumn("net_units", F.col("units_sold") - F.col("units_returned"))
            .withColumn(
                "return_rate_pct",
                F.when(F.col("units_sold") > 0, F.round((F.col("units_returned") / F.col("units_sold")) * 100, 2)).otherwise(F.lit(0.0))
            )
            .orderBy(F.col("net_revenue").desc())
        )

    def build_country_revenue(self, silver_df):
        """Builds Gold Country Revenue DataFrame."""
        from pyspark.sql import functions as F

        comm = silver_df.filter(
            ~F.col("is_duplicate") & ~F.col("transaction_status").isin("INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY")
        )
        tot_net_row = comm.select(F.sum("net_revenue")).collect()
        tot_net = tot_net_row[0][0] if tot_net_row and tot_net_row[0][0] else 1.0

        return (
            comm.groupBy("country")
            .agg(
                F.round(F.sum("gross_sales"), 2).alias("gross_sales"),
                F.round(F.sum("return_amount"), 2).alias("return_amount"),
                F.round(F.sum("net_revenue"), 2).alias("net_revenue"),
                F.countDistinct(F.when(F.col("transaction_status") == "VALID_PURCHASE", F.col("invoice_no"))).alias("valid_orders_count"),
                F.countDistinct(F.when(F.col("transaction_status") == "CANCELLATION", F.col("invoice_no"))).alias("cancelled_orders_count"),
                F.countDistinct("customer_id").alias("distinct_customers")
            )
            .withColumn(
                "market_share_pct",
                F.round((F.col("net_revenue") / F.lit(tot_net)) * 100, 2)
            )
            .orderBy(F.col("net_revenue").desc())
        )

    def build_customer_purchase_history(self, silver_df):
        """Builds Gold Customer Purchase History DataFrame."""
        from pyspark.sql import functions as F

        comm = silver_df.filter(
            ~F.col("is_duplicate") & ~F.col("transaction_status").isin("INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY")
            & F.col("customer_id").isNotNull()
        )
        return (
            comm.groupBy("customer_id", "invoice_no")
            .agg(
                F.first("invoice_date").alias("invoice_date"),
                F.first("country").alias("country"),
                F.first("is_cancellation").alias("is_cancellation"),
                F.first("transaction_status").alias("transaction_status"),
                F.count("stock_code").alias("line_items_count"),
                F.sum("quantity").alias("total_quantity"),
                F.round(F.sum("gross_sales"), 2).alias("order_gross_sales"),
                F.round(F.sum("return_amount"), 2).alias("order_return_amount"),
                F.round(F.sum("net_revenue"), 2).alias("order_net_revenue")
            )
            .orderBy(F.col("customer_id").asc(), F.col("invoice_date").desc())
        )

    def run_pyspark_transformations(self) -> Dict[str, Any]:
        """Runs native PySpark transformations if SparkSession is active."""
        if self.spark is None:
            logger.warning("SparkSession is not available. Skipping PySpark run.")
            return {"status": "SKIPPED_SPARK_UNAVAILABLE"}

        from pyspark.sql import functions as F

        logger.info(f"Reading Bronze dataset via PySpark: {self.config.bronze_parquet_path}")
        bronze_df = self.spark.read.parquet(str(self.config.bronze_parquet_path))
        bronze_count = bronze_df.count()
        logger.info(f"PySpark Bronze record count: {bronze_count:,}")

        # 1. Silver
        silver_df = self.build_silver(bronze_df)
        silver_count = silver_df.count()
        logger.info(f"PySpark Silver record count: {silver_count:,}")

        # 2. Revenue Metrics
        rev_metrics = silver_df.select(
            F.round(F.sum("gross_sales"), 2).alias("gross"),
            F.round(F.sum("return_amount"), 2).alias("returns"),
            F.round(F.sum("net_revenue"), 2).alias("net")
        ).collect()[0]

        logger.info(
            f"PySpark Silver Metrics: Gross=£{rev_metrics['gross']:,.2f}, "
            f"Returns=-£{rev_metrics['returns']:,.2f}, Net=£{rev_metrics['net']:,.2f}"
        )

        # 3. Gold Tables
        monthly_df = self.build_monthly_sales(silver_df)
        rfm_df = self.build_customer_rfm(silver_df)
        prod_df = self.build_product_performance(silver_df)
        country_df = self.build_country_revenue(silver_df)
        hist_df = self.build_customer_purchase_history(silver_df)

        return {
            "status": "SUCCESS",
            "bronze_count": bronze_count,
            "silver_count": silver_count,
            "gross_sales": rev_metrics["gross"],
            "return_amount": rev_metrics["returns"],
            "net_revenue": rev_metrics["net"],
            "monthly_sales_count": monthly_df.count(),
            "customer_rfm_count": rfm_df.count(),
            "product_performance_count": prod_df.count(),
            "country_revenue_count": country_df.count(),
            "customer_history_count": hist_df.count(),
        }
