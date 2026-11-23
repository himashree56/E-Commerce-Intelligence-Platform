# Databricks notebook source
# MAGIC %md
# MAGIC # E-Commerce Intelligence Platform - Databricks Lakehouse Pipeline (Phase 2)
# MAGIC 
# MAGIC This notebook executes the Bronze-to-Silver-to-Gold Delta Lake medallion transformations.
# MAGIC Architecture:
# MAGIC - Bronze: Raw ingested Parquet / Delta records (100% preserved)
# MAGIC - Silver: Curated fact transactions with deduplication, quality flags, and financial metrics
# MAGIC - Gold: Five business & predictive analytical datasets:
# MAGIC   1. `monthly_sales`
# MAGIC   2. `customer_rfm`
# MAGIC   3. `product_performance`
# MAGIC   4. `country_revenue`
# MAGIC   5. `customer_purchase_history`
# MAGIC
# MAGIC Note: This notebook is Databricks Lakehouse & Unity Catalog compatible. Cloud execution is not claimed unless executed against an active Databricks cluster.

# COMMAND ----------
# DBTITLE 1,Configuration & Imports
from pyspark.sql import functions as F
from pyspark.sql.window import Window

BRONZE_PATH = "dbfs:/mnt/lakehouse/bronze/online_retail/"
SILVER_PATH = "dbfs:/mnt/lakehouse/silver/online_retail/"
GOLD_BASE_PATH = "dbfs:/mnt/lakehouse/gold/"

# COMMAND ----------
# DBTITLE 1,Read Bronze Delta / Parquet Table
bronze_df = spark.read.format("parquet").load(BRONZE_PATH)
display(bronze_df.limit(5))

# COMMAND ----------
# DBTITLE 1,Silver Transformation: Explicit Quality & Status Flags
window_spec = Window.partitionBy(
    "invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"
).orderBy("invoice_date")

silver_df = (
    bronze_df
    # 1. Deduplication window flag
    .withColumn("row_num", F.row_number().over(window_spec))
    .withColumn("is_duplicate", F.col("row_num") > 1)
    .drop("row_num")
    # 2. Flag anomalies
    .withColumn("is_cancellation", F.col("invoice_no").startswith("C") | F.col("invoice_no").startswith("c"))
    .withColumn("is_return", F.col("quantity") < 0)
    .withColumn("is_invalid_price", (F.col("unit_price") <= 0) | F.col("unit_price").isNull())
    .withColumn("is_missing_customer", F.col("customer_id").isNull())
    .withColumn(
        "is_non_product",
        F.upper(F.col("stock_code")).isin("POST", "D", "M", "BANK CHARGES", "CRUK", "PADS", "DOT", "C2", "TEST", "ADJUST", "ADJUST2")
    )
    .withColumn(
        "is_ambiguous_cancellation",
        (F.col("invoice_no").startswith("C") | F.col("invoice_no").startswith("c")) & (F.col("quantity") > 0)
    )
    # 3. Categorical Transaction Status (Mutually Exclusive Hierarchy)
    .withColumn(
        "transaction_status",
        F.when(F.col("is_duplicate"), F.lit("DUPLICATE"))
        .when(F.col("is_cancellation"), F.lit("CANCELLATION"))
        .when(F.col("is_return") & F.col("is_invalid_price"), F.lit("INVENTORY_WRITE_OFF"))
        .when(F.col("is_return") & ~F.col("is_invalid_price"), F.lit("RETURN_ADJUSTMENT"))
        .when(~F.col("is_return") & F.col("is_invalid_price"), F.lit("ZERO_PRICE_ENTRY"))
        .otherwise(F.lit("VALID_PURCHASE"))
    )
    # 4. Data Quality Flag
    .withColumn(
        "quality_flag",
        F.when(F.col("transaction_status").isin("DUPLICATE", "INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY"), F.lit("EXCLUDE_NON_REVENUE"))
        .when(F.col("is_missing_customer"), F.lit("WARN_MISSING_CUSTOMER"))
        .when(F.col("is_non_product"), F.lit("WARN_NON_PRODUCT"))
        .otherwise(F.lit("PASS"))
    )
    # 5. Financial Metrics
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
    # 6. Temporal Dimensions
    .withColumn("invoice_year_month", F.date_format("invoice_date", "yyyy-MM"))
    .withColumn("invoice_year", F.year("invoice_date"))
    .withColumn("invoice_month", F.month("invoice_date"))
    .withColumn("invoice_day_of_week", F.date_format("invoice_date", "EEEE"))
    .withColumn("invoice_hour", F.hour("invoice_date"))
)

# COMMAND ----------
# DBTITLE 1,Write Silver Table in Delta Format with Change Data Feed
(
    silver_df.write.format("delta")
    .mode("overwrite")
    .partitionBy("invoice_year_month")
    .option("delta.enableChangeDataFeed", "true")
    .save(SILVER_PATH)
)

# COMMAND ----------
# DBTITLE 1,Optimize Silver Table with Z-Ordering
# MAGIC %sql
# MAGIC OPTIMIZE delta.`dbfs:/mnt/lakehouse/silver/online_retail/`
# MAGIC ZORDER BY (customer_id, invoice_date, stock_code);

# COMMAND ----------
# DBTITLE 1,Filter Commercial Dataset for Gold Analytics
commercial_silver = silver_df.filter(
    ~F.col("is_duplicate") & ~F.col("transaction_status").isin("INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY")
)

# COMMAND ----------
# DBTITLE 1,Gold Table 1: Monthly Sales Performance
w_cum = Window.orderBy("invoice_year_month").rowsBetween(Window.unboundedPreceding, Window.currentRow)
w_lag = Window.orderBy("invoice_year_month")

gold_monthly_base = (
    commercial_silver.groupBy("invoice_year_month", "invoice_year", "invoice_month")
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

gold_monthly = (
    gold_monthly_base
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
gold_monthly.write.format("delta").mode("overwrite").save(GOLD_BASE_PATH + "monthly_sales/")

# COMMAND ----------
# DBTITLE 1,Gold Table 2: Customer RFM Segmentation
cust_base = commercial_silver.filter(F.col("customer_id").isNotNull())
max_date_val = cust_base.select(F.max("invoice_date")).collect()[0][0]

cust_rfm_agg = (
    cust_base.groupBy("customer_id")
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
    .withColumn("recency_days", F.datediff(F.lit(max_date_val), F.col("last_purchase_date")))
)

# RFM Quintile Scoring via Window NTILE
r_window = Window.orderBy(F.col("recency_days").desc())
f_window = Window.orderBy(F.col("frequency").asc())
m_window = Window.orderBy(F.col("monetary_net").asc())

gold_rfm = (
    cust_rfm_agg
    .withColumn("r_score", F.ntile(5).over(r_window))
    .withColumn("f_score", F.ntile(5).over(f_window))
    .withColumn("m_score", F.ntile(5).over(m_window))
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
gold_rfm.write.format("delta").mode("overwrite").save(GOLD_BASE_PATH + "customer_rfm/")

# COMMAND ----------
# DBTITLE 1,Gold Table 3: Product Performance & Return Rate
prod_base = commercial_silver.filter(~F.col("is_non_product"))

gold_product = (
    prod_base.groupBy("stock_code")
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
gold_product.write.format("delta").mode("overwrite").save(GOLD_BASE_PATH + "product_performance/")

# COMMAND ----------
# DBTITLE 1,Gold Table 4: Country Revenue & Market Share
total_net_rev = commercial_silver.select(F.sum("net_revenue")).collect()[0][0] or 1.0

gold_country = (
    commercial_silver.groupBy("country")
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
        F.round((F.col("net_revenue") / F.lit(total_net_rev)) * 100, 2)
    )
    .orderBy(F.col("net_revenue").desc())
)
gold_country.write.format("delta").mode("overwrite").save(GOLD_BASE_PATH + "country_revenue/")

# COMMAND ----------
# DBTITLE 1,Gold Table 5: Customer Purchase History Timeline
gold_history = (
    cust_base.groupBy("customer_id", "invoice_no")
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
gold_history.write.format("delta").mode("overwrite").save(GOLD_BASE_PATH + "customer_purchase_history/")
