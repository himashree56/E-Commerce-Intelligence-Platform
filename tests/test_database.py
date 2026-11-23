"""Integration and Unit Tests for PostgreSQL Database Layer (Phase 3)."""
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from src.database.config import DatabaseConfig
from src.database.connection import DatabaseManager
from src.database.loader import GoldDatabaseLoader
from src.database.models import (
    Base,
    CountryRevenueAnalytics,
    CustomerPurchaseHistory,
    CustomerRFMAnalytics,
    MonthlySalesAnalytics,
    ProductPerformanceAnalytics,
)
from src.database.queries import ALL_ANALYTICS_QUERIES, AnalyticsQueryRunner
from src.database.reconciliation import PostgresReconciliationAuditor
from src.database.schema_manager import SchemaManager


@pytest.fixture(scope="module")
def test_db_manager():
    """Initializes and returns a DatabaseManager pointing to the isolated test database."""
    config = DatabaseConfig()
    mgr = DatabaseManager(config=config, is_test=True)
    health = mgr.check_health()
    if health.get("status") != "HEALTHY":
        pytest.skip(f"PostgreSQL test database not available: {health.get('error')}")

    # Initialize schema
    schema_mgr = SchemaManager(db_manager=mgr)
    schema_mgr.create_all_tables(is_test=True)

    yield mgr

    # Cleanup after module
    schema_mgr.drop_all_tables(is_test=True)
    mgr.dispose()


@pytest.fixture(autouse=True)
def clean_test_tables(test_db_manager):
    """Truncates all tables in the test database prior to each test."""
    with test_db_manager.engine.connect() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE customer_purchase_history, customer_rfm_analytics, "
                "monthly_sales_analytics, product_performance_analytics, country_revenue_analytics, "
                "data_quality_audit_results, etl_execution_metadata RESTART IDENTITY CASCADE;"
            )
        )
        conn.commit()


def test_database_connectivity(test_db_manager):
    """Verifies active PostgreSQL connection, version inspection, and latency."""
    health = test_db_manager.check_health()
    assert health["status"] == "HEALTHY"
    assert "PostgreSQL 16" in health["version"]
    assert health["database"] == test_db_manager.config.test_db_name
    assert health["latency_ms"] >= 0.0


def test_schema_constraints_and_tables(test_db_manager):
    """Verifies that all 7 required tables are created in PostgreSQL."""
    schema_mgr = SchemaManager(db_manager=test_db_manager)
    tables = schema_mgr.get_table_row_counts(is_test=True)

    expected_tables = {
        "monthly_sales_analytics",
        "customer_rfm_analytics",
        "product_performance_analytics",
        "country_revenue_analytics",
        "customer_purchase_history",
        "etl_execution_metadata",
        "data_quality_audit_results",
    }
    assert expected_tables.issubset(set(tables.keys()))


def test_foreign_key_constraint_enforcement(test_db_manager):
    """Verifies that inserting into customer_purchase_history without a valid customer_id raises IntegrityError."""
    with pytest.raises(IntegrityError):
        with test_db_manager.session_scope() as session:
            # Attempt to insert order for non-existent customer 999999
            orphan_order = CustomerPurchaseHistory(
                invoice_no="INV_TEST_ORPHAN",
                customer_id=999999,
                invoice_date=datetime.now(timezone.utc),
                country="United Kingdom",
                is_cancellation=False,
                transaction_status="VALID_PURCHASE",
                line_items_count=1,
                total_quantity=5,
                order_gross_sales=Decimal("50.00"),
                order_return_amount=Decimal("0.00"),
                order_net_revenue=Decimal("50.00"),
            )
            session.add(orphan_order)


def test_duplicate_prevention_and_idempotence(test_db_manager):
    """Verifies that inserting the same record twice updates existing data without duplicating rows."""
    loader = GoldDatabaseLoader(is_test=True)

    # Initial mock record
    mock_df_1 = pd.DataFrame({
        "invoice_year_month": ["2010-01"],
        "invoice_year": [2010],
        "invoice_month": [1],
        "gross_sales": [1000.50],
        "return_amount": [50.00],
        "net_revenue": [950.50],
        "valid_orders_count": [10],
        "cancelled_orders_count": [1],
        "distinct_customers": [8],
        "total_units_sold": [500],
        "units_returned": [20],
        "net_units": [480],
        "average_order_value": [100.05],
    })

    with test_db_manager.session_scope() as session:
        loader.load_monthly_sales(session, mock_df_1)

    with test_db_manager.engine.connect() as conn:
        count_1 = conn.execute(select(func.count()).select_from(MonthlySalesAnalytics)).scalar()
        rev_1 = conn.execute(select(MonthlySalesAnalytics.net_revenue)).scalar()

    assert count_1 == 1
    assert rev_1 == Decimal("950.50")

    # Second load with updated net revenue
    mock_df_2 = mock_df_1.copy()
    mock_df_2["net_revenue"] = [999.00]

    with test_db_manager.session_scope() as session:
        loader.load_monthly_sales(session, mock_df_2)

    with test_db_manager.engine.connect() as conn:
        count_2 = conn.execute(select(func.count()).select_from(MonthlySalesAnalytics)).scalar()
        rev_2 = conn.execute(select(MonthlySalesAnalytics.net_revenue)).scalar()

    assert count_2 == 1, "Idempotence failed: Row count duplicated!"
    assert rev_2 == Decimal("999.00"), "Upsert failed to update existing record!"


def test_transaction_rollback_on_failure(test_db_manager):
    """Verifies that an unhandled exception inside session_scope rolls back the transaction entirely."""
    try:
        with test_db_manager.session_scope() as session:
            # 1. Valid insert
            country = CountryRevenueAnalytics(
                country="TestLand",
                gross_sales=Decimal("500.00"),
                return_amount=Decimal("0.00"),
                net_revenue=Decimal("500.00"),
                valid_orders_count=5,
                cancelled_orders_count=0,
                distinct_customers=3,
                market_share_pct=Decimal("100.00"),
            )
            session.add(country)
            session.flush()

            # 2. Trigger intentional exception
            raise RuntimeError("Intentional simulated crash before commit")
    except RuntimeError:
        pass

    # Verify that TestLand was rolled back and is not in the database
    with test_db_manager.engine.connect() as conn:
        res = conn.execute(
            select(CountryRevenueAnalytics).where(CountryRevenueAnalytics.country == "TestLand")
        ).first()
        assert res is None, "Rollback failed: partial record was committed!"


def test_null_handling_in_loader(test_db_manager):
    """Verifies that null descriptions and NaN values are correctly converted to SQL NULL."""
    loader = GoldDatabaseLoader(is_test=True)
    prod_df = pd.DataFrame({
        "stock_code": ["TEST_SKU_NULL"],
        "description": [None],  # Null description
        "gross_sales": [100.0],
        "return_amount": [0.0],
        "net_revenue": [100.0],
        "units_sold": [10],
        "units_returned": [0],
        "distinct_orders": [2],
        "distinct_customers": [2],
        "net_units": [10],
        "return_rate_pct": [0.0],
    })

    with test_db_manager.session_scope() as session:
        loader.load_product_performance(session, prod_df)

    with test_db_manager.engine.connect() as conn:
        prod = conn.execute(
            select(ProductPerformanceAnalytics).where(
                ProductPerformanceAnalytics.stock_code == "TEST_SKU_NULL"
            )
        ).first()

    assert prod is not None
    assert prod.description is None


def test_monetary_precision_numeric_type(test_db_manager):
    """Verifies that financial columns preserve exact 2 decimal precision without floating point drift."""
    loader = GoldDatabaseLoader(is_test=True)
    prod_df = pd.DataFrame({
        "stock_code": ["TEST_PRECISION_SKU"],
        "description": ["Precision Test Product"],
        "gross_sales": [19.99],
        "return_amount": [2.55],
        "net_revenue": [17.44],
        "units_sold": [1],
        "units_returned": [0],
        "distinct_orders": [1],
        "distinct_customers": [1],
        "net_units": [1],
        "return_rate_pct": [12.75],
    })

    with test_db_manager.session_scope() as session:
        loader.load_product_performance(session, prod_df)

    with test_db_manager.engine.connect() as conn:
        prod = conn.execute(
            select(ProductPerformanceAnalytics).where(
                ProductPerformanceAnalytics.stock_code == "TEST_PRECISION_SKU"
            )
        ).first()

    assert isinstance(prod.gross_sales, Decimal)
    assert prod.gross_sales == Decimal("19.99")
    assert prod.return_amount == Decimal("2.55")
    assert prod.net_revenue == Decimal("17.44")


def test_production_database_reconciliation():
    """Integration test verifying 100% reconciliation against the live PostgreSQL production database."""
    auditor = PostgresReconciliationAuditor(is_test=False)
    report = auditor.audit()

    assert report.all_checks_passed is True
    assert report.monetary_precision_verified is True
    assert report.idempotent_rerun_verified is True
    assert report.is_silver_gold_postgres_reconciled is True
    assert len(report.discrepancies) == 0

    # Verify exact reference row counts
    assert report.table_checks["monthly_sales_analytics"].postgres_rows == 25
    assert report.table_checks["customer_rfm_analytics"].postgres_rows == 5939
    assert report.table_checks["product_performance_analytics"].postgres_rows == 4920
    assert report.table_checks["country_revenue_analytics"].postgres_rows == 43
    assert report.table_checks["customer_purchase_history"].postgres_rows == 44870


def test_all_12_analytical_queries_execution():
    """Integration test verifying that all 12 Advanced SQL Analytics queries execute without error."""
    runner = AnalyticsQueryRunner()
    results = runner.run_all_queries()

    assert len(results) == 12
    for q_id, df in results.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, f"Query {q_id} returned empty DataFrame!"


# =============================================================================
# PHASE 3 AUDIT REGRESSION TESTS (FINANCIAL PRECISION & CUSTOMER CONSISTENCY)
# =============================================================================


def test_exact_monthly_revenue_zero_penny_discrepancy():
    """Regression test enforcing exact £0.00 discrepancy across Silver, Gold Monthly, Gold Country, and PostgreSQL."""
    auditor = PostgresReconciliationAuditor(is_test=False)
    report = auditor.audit()

    assert report.all_checks_passed is True
    assert report.is_silver_gold_postgres_reconciled is True

    # Validate exact reference figures (exact to the penny: £19,012,910.85)
    expected_net = Decimal("19012910.85")
    assert Decimal(str(report.postgres_monthly_net_revenue)) == expected_net
    assert Decimal(str(report.postgres_country_net_revenue)) == expected_net
    assert Decimal(str(report.silver_reference_net_revenue)) == expected_net

    # Zero penny difference
    assert abs(Decimal(str(report.postgres_monthly_net_revenue)) - expected_net) == Decimal("0.00")
    assert abs(Decimal(str(report.postgres_country_net_revenue)) - expected_net) == Decimal("0.00")


def test_customer_concentration_denominator_consistency():
    """Regression test verifying Pareto 80/20 customer concentration and multi-denominator metrics."""
    runner = AnalyticsQueryRunner()
    df = runner.run_query("revenue_concentration_pareto")

    assert len(df) == 4
    assert list(df["customer_bracket"]) == [
        "Top 1% Customers",
        "Top 5% Customers",
        "Top 10% Customers",
        "Top 20% Customers (Pareto)",
    ]

    # Positive spenders counts (N=5,837)
    assert df.loc[df["customer_bracket"] == "Top 1% Customers", "customer_count"].iloc[0] == 59
    assert df.loc[df["customer_bracket"] == "Top 5% Customers", "customer_count"].iloc[0] == 292
    assert df.loc[df["customer_bracket"] == "Top 10% Customers", "customer_count"].iloc[0] == 584
    assert df.loc[df["customer_bracket"] == "Top 20% Customers (Pareto)", "customer_count"].iloc[0] == 1168

    # Revenue generated per bracket
    top_1_rev = df.loc[df["customer_bracket"] == "Top 1% Customers", "bracket_revenue_gbp"].iloc[0]
    top_20_rev = df.loc[df["customer_bracket"] == "Top 20% Customers (Pareto)", "bracket_revenue_gbp"].iloc[0]
    assert abs(top_1_rev - 5091379.80) < 0.01
    assert abs(top_20_rev - 12550327.36) < 0.01

    # Verify percentages across the 3 documented denominators:
    # 1. Denominator 1 (Positive Spenders £16,382,076.17)
    top_20_pos_pct = df.loc[df["customer_bracket"] == "Top 20% Customers (Pareto)", "revenue_pct"].iloc[0]
    assert abs(top_20_pos_pct - 76.61) < 0.05

    # 2. Denominator 2 (All Identified Customers £16,289,439.42)
    top_20_all_cust_pct = df.loc[df["customer_bracket"] == "Top 20% Customers (Pareto)", "share_of_all_customer_net_pct"].iloc[0]
    assert abs(top_20_all_cust_pct - 77.05) < 0.05

    # 3. Denominator 3 (Total Commercial Company Net Revenue £19,012,910.85)
    top_20_comm_pct = df.loc[df["customer_bracket"] == "Top 20% Customers (Pareto)", "share_of_commercial_total_pct"].iloc[0]
    assert abs(top_20_comm_pct - 66.01) < 0.05


def test_idempotent_reload_verification(test_db_manager):
    """Regression test verifying that loading the same Gold datasets repeatedly causes zero duplicate rows."""
    loader = GoldDatabaseLoader(is_test=True)

    # First load
    report1 = loader.load_all_gold_tables()
    assert report1.status == "SUCCESS"
    counts1 = SchemaManager(db_manager=test_db_manager).get_table_row_counts(is_test=True)

    # Second load
    report2 = loader.load_all_gold_tables()
    assert report2.status == "SUCCESS"
    counts2 = SchemaManager(db_manager=test_db_manager).get_table_row_counts(is_test=True)

    gold_tables = [
        "monthly_sales_analytics",
        "customer_rfm_analytics",
        "product_performance_analytics",
        "country_revenue_analytics",
        "customer_purchase_history",
    ]
    # Row counts across all 5 Gold analytical tables must remain strictly identical (no row inflation)
    for tbl in gold_tables:
        assert counts1[tbl] == counts2[tbl], f"Row count changed for {tbl} on reload!"

    assert counts2["monthly_sales_analytics"] == 25
    assert counts2["customer_rfm_analytics"] == 5939
    assert counts2["product_performance_analytics"] == 4920
    assert counts2["country_revenue_analytics"] == 43
    assert counts2["customer_purchase_history"] == 44870

    # Operational audit tables log each pipeline execution
    assert counts2["etl_execution_metadata"] == counts1["etl_execution_metadata"] + 1
    assert counts2["data_quality_audit_results"] == counts1["data_quality_audit_results"] + 1


def test_decimal_monetary_precision_fixed_pence(test_db_manager):
    """Regression test verifying that sub-penny fractional inputs are rounded cleanly and stored as fixed NUMERIC(14,2)."""
    loader = GoldDatabaseLoader(is_test=True)
    # Test record with 3-decimal place sub-penny pricing (e.g. 0.001 like PADS stock_code)
    test_df = pd.DataFrame({
        "stock_code": ["SUB_PENNY_SKU"],
        "description": ["Sub penny unit test item"],
        "gross_sales": [0.001],
        "return_amount": [0.00],
        "net_revenue": [0.001],
        "units_sold": [1],
        "units_returned": [0],
        "distinct_orders": [1],
        "distinct_customers": [1],
        "net_units": [1],
        "return_rate_pct": [0.0],
    })

    with test_db_manager.session_scope() as session:
        loader.load_product_performance(session, test_df)

    with test_db_manager.engine.connect() as conn:
        row = conn.execute(
            select(ProductPerformanceAnalytics).where(
                ProductPerformanceAnalytics.stock_code == "SUB_PENNY_SKU"
            )
        ).first()

    assert row is not None
    assert isinstance(row.gross_sales, Decimal)
    assert row.gross_sales == Decimal("0.00")
    assert row.net_revenue == Decimal("0.00")
