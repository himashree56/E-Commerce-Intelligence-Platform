"""Unit tests for Gold layer analytical tables."""
import numpy as np
import pandas as pd
import pytest

from src.etl.config import ETLConfig
from src.etl.gold import GoldTransformer
from src.etl.silver import SilverTransformer


def test_gold_tables_generation(mock_bronze_records):
    """Test generating all 5 Gold analytical tables from Silver output."""
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(mock_bronze_records)

    g_transformer = GoldTransformer(ETLConfig())
    monthly_df = g_transformer.build_monthly_sales(silver_df)
    rfm_df = g_transformer.build_customer_rfm(silver_df)
    prod_df = g_transformer.build_product_performance(silver_df)
    country_df = g_transformer.build_country_revenue(silver_df)
    hist_df = g_transformer.build_customer_purchase_history(silver_df)

    # 1. Monthly Sales
    assert len(monthly_df) == 1  # All records are in 2010-01
    assert monthly_df.iloc[0]["invoice_year_month"] == "2010-01"
    # Commercial net revenue should match
    comm_silver = silver_df[
        ~silver_df["is_duplicate"]
        & ~silver_df["transaction_status"].isin(["INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY"])
    ]
    assert abs(monthly_df["net_revenue"].sum() - comm_silver["net_revenue"].sum()) < 1e-4

    # 2. Customer RFM
    assert len(rfm_df) == 1  # Only customer 12345 has non-null customer_id in mock
    cust = rfm_df.iloc[0]
    assert cust["customer_id"] == 12345
    assert cust["frequency"] >= 1
    assert cust["customer_segment"] is not None

    # 3. Product Performance
    assert len(prod_df) >= 2
    assert "return_rate_pct" in prod_df.columns

    # 4. Country Revenue
    assert len(country_df) == 1
    assert country_df.iloc[0]["country"] == "United Kingdom"
    assert country_df.iloc[0]["market_share_pct"] == 100.0

    # 5. Customer Purchase History
    assert len(hist_df) >= 1
    assert "order_net_revenue" in hist_df.columns


def test_gold_monthly_sales_financial_balance(mock_bronze_records):
    """Test that monthly sales gross, returns, and net revenue balance exactly."""
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(mock_bronze_records)
    g_transformer = GoldTransformer(ETLConfig())
    monthly_df = g_transformer.build_monthly_sales(silver_df)

    for _, row in monthly_df.iterrows():
        calc_net = round(row["gross_sales"] - row["return_amount"], 2)
        assert abs(calc_net - row["net_revenue"]) < 1e-4
        assert row["net_units"] == row["total_units_sold"] - row["units_returned"]


def test_customer_rfm_multi_customer_segmentation():
    """Test Customer RFM scoring and segmentation with multiple distinct customers."""
    now = pd.Timestamp("2011-12-01 10:00:00")
    data = pd.DataFrame({
        "invoice_no": ["101", "102", "103", "104", "105"],
        "stock_code": ["21099"] * 5,
        "description": ["PAPER CUPS"] * 5,
        "quantity": [10, 50, 5, 100, 2],
        "invoice_date": [
            now - pd.Timedelta(days=1),   # very recent
            now - pd.Timedelta(days=10),  # recent
            now - pd.Timedelta(days=100), # moderate
            now - pd.Timedelta(days=200), # old
            now - pd.Timedelta(days=350), # very old
        ],
        "unit_price": [10.0, 20.0, 5.0, 15.0, 2.0],
        "customer_id": [1001, 1002, 1003, 1004, 1005],
        "country": ["United Kingdom", "France", "Germany", "United Kingdom", "Spain"],
        "sheet_name": ["Year 2010-2011"] * 5,
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")] * 5,
        "is_cancellation": [False] * 5,
    })
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(data)

    g_transformer = GoldTransformer(ETLConfig())
    rfm_df = g_transformer.build_customer_rfm(silver_df)

    assert len(rfm_df) == 5
    # Recency scores must be between 1 and 5
    assert (rfm_df["r_score"] >= 1).all() and (rfm_df["r_score"] <= 5).all()
    assert (rfm_df["f_score"] >= 1).all() and (rfm_df["f_score"] <= 5).all()
    assert (rfm_df["m_score"] >= 1).all() and (rfm_df["m_score"] <= 5).all()

    # Most recent customer (1001) should have highest R score
    cust_1001 = rfm_df[rfm_df["customer_id"] == 1001].iloc[0]
    assert cust_1001["r_score"] >= 4


def test_product_performance_non_product_exclusion():
    """Test that administrative non-product codes (POST, D, M) are excluded from product table."""
    data = pd.DataFrame({
        "invoice_no": ["201", "202", "203"],
        "stock_code": ["POST", "21099", "M"],
        "description": ["POSTAGE", "PAPER CUPS", "Manual"],
        "quantity": [1, 10, 1],
        "invoice_date": [pd.Timestamp("2011-05-01 10:00:00")] * 3,
        "unit_price": [15.0, 2.0, 50.0],
        "customer_id": [12345, 12345, 12345],
        "country": ["United Kingdom"] * 3,
        "sheet_name": ["Year 2010-2011"] * 3,
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")] * 3,
        "is_cancellation": [False] * 3,
    })
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(data)

    g_transformer = GoldTransformer(ETLConfig())
    prod_df = g_transformer.build_product_performance(silver_df)

    assert len(prod_df) == 1
    assert prod_df.iloc[0]["stock_code"] == "21099"


def test_product_performance_pandas_pyspark_parity(tmp_path):
    """Regression test ensuring 100% parity between Pandas and PySpark Gold product tables.

    Tests:
    1. Case-insensitive non-product exclusion (e.g. lowercase 'm', 'post' vs 'M', 'POST').
    2. Null aggregation handling (zero returns or zero sales coalesce to 0/0.0).
    3. Row count, stock code keys, units, distinct orders, and revenue match exactly.
    """
    try:
        from src.etl.pyspark_pipeline import PySparkLakehousePipeline
        from src.etl.spark_session import get_spark_session

        spark = get_spark_session(enable_delta=False)
    except Exception:
        pytest.skip("PySpark unavailable or failed to initialize.")

    if spark is None:
        pytest.skip("SparkSession could not be initialized.")

    now = pd.Timestamp("2011-06-01 12:00:00")
    test_data = pd.DataFrame({
        "invoice_no": ["501", "502", "C503", "504", "505", "506"],
        "stock_code": ["21099", "21099", "21099", "85123A", "m", "POST"],
        "description": ["PAPER CUPS", "PAPER CUPS", "PAPER CUPS", "WHITE HANGING HEART", "Manual", "POSTAGE"],
        "quantity": [10, 5, -2, 8, 1, 1],
        "invoice_date": [now] * 6,
        "unit_price": [2.50, 2.50, 2.50, 3.00, 10.0, 15.0],
        "customer_id": [12345.0, 12346.0, 12345.0, 12347.0, None, 12345.0],
        "country": ["United Kingdom"] * 6,
        "sheet_name": ["Year 2010-2011"] * 6,
        "ingestion_timestamp": [pd.Timestamp.now(tz="UTC")] * 6,
        "is_cancellation": [False, False, True, False, False, False],
    })

    mock_parquet = tmp_path / "mock_bronze.parquet"
    test_data.to_parquet(mock_parquet)

    cfg = ETLConfig()

    # 1. Pandas Gold Product
    s_transformer = SilverTransformer(cfg)
    silver_df = s_transformer.transform_dataframe(test_data)
    g_transformer = GoldTransformer(cfg)
    pandas_prod = g_transformer.build_product_performance(silver_df)

    # 2. PySpark Gold Product
    pipeline = PySparkLakehousePipeline(cfg, spark_session=spark)
    spark_bronze = spark.read.parquet(str(mock_parquet))
    spark_silver = pipeline.build_silver(spark_bronze)
    spark_prod_df = pipeline.build_product_performance(spark_silver)
    spark_rows = [row.asDict() for row in spark_prod_df.collect()]
    spark_prod = pd.DataFrame(spark_rows)

    # 3. Assert parity
    assert len(pandas_prod) == len(spark_prod), f"Count divergence: Pandas={len(pandas_prod)}, Spark={len(spark_prod)}"
    assert set(pandas_prod["stock_code"]) == set(spark_prod["stock_code"])
    assert "m" not in spark_prod["stock_code"].values
    assert "M" not in spark_prod["stock_code"].values
    assert "POST" not in spark_prod["stock_code"].values

    for sc in pandas_prod["stock_code"].values:
        p_row = pandas_prod[pandas_prod["stock_code"] == sc].iloc[0]
        s_row = spark_prod[spark_prod["stock_code"] == sc].iloc[0]

        assert p_row["units_sold"] == s_row["units_sold"], f"units_sold mismatch for {sc}"
        assert p_row["units_returned"] == s_row["units_returned"], f"units_returned mismatch for {sc}"
        assert p_row["net_units"] == s_row["net_units"], f"net_units mismatch for {sc}"
        assert p_row["distinct_orders"] == s_row["distinct_orders"], f"distinct_orders mismatch for {sc}"
        assert abs(p_row["gross_sales"] - s_row["gross_sales"]) < 1e-2, f"gross_sales mismatch for {sc}"
        assert abs(p_row["return_amount"] - s_row["return_amount"]) < 1e-2, f"return_amount mismatch for {sc}"
        assert abs(p_row["net_revenue"] - s_row["net_revenue"]) < 1e-2, f"net_revenue mismatch for {sc}"
