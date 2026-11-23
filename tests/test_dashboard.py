"""
Phase 7 Dashboard Automated Test Suite.

Validates DataProvider integrity, PostgreSQL and Parquet fallback parity,
chart generation, metric formatters, and financial reconciliation.
"""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from unittest.mock import MagicMock, patch

from src.dashboard.config import (
    CLUSTER_COLORS,
    CLUSTER_NAMES,
    DashboardConfig,
    format_gbp,
    format_number,
    format_pct,
)
from src.dashboard.data_provider import DataProvider
from src.dashboard.components.charts import (
    create_calibration_curve_chart,
    create_cluster_population_chart,
    create_cluster_revenue_chart,
    create_feature_importance_chart,
    create_order_volume_trend_chart,
    create_revenue_trend_chart,
    create_rfm_scatter_chart,
    create_roc_curve_chart,
    create_top_countries_chart,
    create_top_products_chart,
    create_anomaly_distribution_chart,
)


@pytest.fixture(scope="module")
def data_provider():
    """Returns DataProvider instance."""
    return DataProvider()


class TestDashboardConfigAndFormatters:
    """Tests for dashboard formatting utilities and constants."""

    def test_format_gbp_positive(self):
        assert format_gbp(19012910.85) == "£19,012,910.85"
        assert format_gbp(423.733) == "£423.73"

    def test_format_gbp_negative(self):
        assert format_gbp(-36680.15) == "-£36,680.15"

    def test_format_gbp_none(self):
        assert format_gbp(None) == "£0.00"

    def test_format_number(self):
        assert format_number(1067371) == "1,067,371"
        assert format_number(5939) == "5,939"
        assert format_number(None) == "0"
        assert format_number(1234.56, decimals=1) == "1,234.6"

    def test_format_pct(self):
        assert format_pct(0.1968) == "19.68%"
        assert format_pct(0.7749) == "77.49%"
        assert format_pct(None) == "0.00%"

    def test_cluster_metadata(self):
        assert len(CLUSTER_NAMES) == 5
        assert len(CLUSTER_COLORS) == 5
        assert CLUSTER_NAMES[0] == "High-Value Loyal Champions"
        assert CLUSTER_NAMES[4] == "Net Returners & Adjusters"


class TestDataProviderFinancialBaselines:
    """Tests DataProvider returns exact verified commercial and ML numbers."""

    def test_executive_kpis_exact_values(self, data_provider):
        kpis = data_provider.get_executive_kpis()
        assert np.isclose(kpis["net_revenue"], 19012910.85, atol=0.01)
        assert np.isclose(kpis["gross_sales"], 20475708.60, atol=0.01)
        assert np.isclose(kpis["return_amount"], 1462797.75, atol=0.01)
        assert kpis["total_silver_transactions"] == 1067371
        assert kpis["unique_customers"] == 5939
        assert kpis["distinct_invoices"] == 44870
        assert kpis["anomalies_flagged"] == 16664

    def test_monthly_sales_reconciliation(self, data_provider):
        df = data_provider.get_monthly_sales()
        assert len(df) == 25
        assert np.isclose(df["net_revenue"].sum(), 19012910.85, atol=0.01)
        assert np.isclose(df["gross_sales"].sum(), 20475708.60, atol=0.01)
        assert np.isclose(df["return_amount"].sum(), 1462797.75, atol=0.01)

    def test_country_revenue_totals(self, data_provider):
        df = data_provider.get_country_revenue(limit=10)
        assert not df.empty
        assert "United Kingdom" in df["country"].values
        uk_row = df[df["country"] == "United Kingdom"].iloc[0]
        assert uk_row["net_revenue"] > 10000000

    def test_top_products_data(self, data_provider):
        df = data_provider.get_top_products(limit=15)
        assert len(df) == 15
        assert "net_revenue" in df.columns
        assert "gross_sales" in df.columns
        assert df.iloc[0]["net_revenue"] >= df.iloc[-1]["net_revenue"]

    def test_cluster_profiles_integrity(self, data_provider):
        df = data_provider.get_cluster_profiles()
        assert len(df) == 5
        assert df["customer_count"].sum() == 5939
        assert np.isclose(df["total_revenue"].sum(), 16289439.42, atol=0.05)
        # Cluster 4 must have negative total revenue
        c4 = df[df["cluster_id"] == 4].iloc[0]
        assert c4["total_revenue"] < 0
        assert c4["customer_count"] == 101

    def test_rfm_clusters_count(self, data_provider):
        df = data_provider.get_rfm_clusters()
        assert len(df) == 5939
        assert set(df["cluster_id"].unique()) == {0, 1, 2, 3, 4}

    def test_rfm_crosstab_balance(self, data_provider):
        df = data_provider.get_rfm_crosstab()
        assert "Total" in df.index
        assert "Total" in df.columns
        assert df.loc["Total", "Total"] == 5939

    def test_anomaly_summary_counts(self, data_provider):
        summary = data_provider.get_anomaly_summary()
        assert summary["total_scored_transactions"] == 1067371
        assert summary["total_anomalies"] == 16664
        assert np.isclose(summary["anomaly_rate_pct"], 1.56, atol=0.01)
        assert "VALID_PURCHASE" in summary["status_breakdown"]
        assert summary["status_breakdown"]["INVENTORY_WRITE_OFF"]["rate"] == 1.0

    def test_high_priority_anomalies_filters(self, data_provider):
        df_all = data_provider.get_high_priority_anomalies(limit=25)
        assert len(df_all) <= 25

        df_cancel = data_provider.get_high_priority_anomalies(
            limit=25, status_filter="CANCELLATION"
        )
        assert all(s == "CANCELLATION" for s in df_cancel["transaction_status"])

    def test_repeat_purchase_metrics(self, data_provider):
        m = data_provider.get_repeat_purchase_metrics()
        test_xgb = m["metrics"]["test"]["xgboost"]
        assert np.isclose(test_xgb["roc_auc"], 0.7750, atol=0.001)
        assert np.isclose(test_xgb["pr_auc"], 0.5180, atol=0.001)
        assert np.isclose(test_xgb["precision_at_top_5_pct"], 0.7749, atol=0.001)
        assert np.isclose(test_xgb["precision_at_top_10_pct"], 0.6022, atol=0.001)
        assert test_xgb["threshold"] == 0.59

        test_cal = m["metrics"]["test"]["xgboost_calibrated_sigmoid"]
        assert test_cal["threshold"] == 0.27
        assert np.isclose(test_cal["brier_score"], 0.1348, atol=0.001)

    def test_repeat_purchase_predictions(self, data_provider):
        # Raw predictions
        df_raw = data_provider.get_repeat_purchase_predictions(limit=50, use_calibrated=False)
        assert len(df_raw) == 50
        assert "repeat_purchase_probability" in df_raw.columns

        # Calibrated predictions
        df_cal = data_provider.get_repeat_purchase_predictions(limit=50, use_calibrated=True)
        assert len(df_cal) == 50
        # Calibrated probabilities should have lower mean than uncalibrated
        assert df_cal["repeat_purchase_probability"].mean() <= df_raw["repeat_purchase_probability"].mean() + 0.05

    def test_data_quality_inventory_reconciliation(self, data_provider):
        inv = data_provider.get_data_quality_inventory()
        assert inv["bronze_records"] == 1067371
        assert inv["silver_records"] == 1067371
        assert inv["gold_customer_rfm"] == 5939
        assert inv["anomalies_flagged"] == 16664
        assert inv["held_out_predictions"] == 5438
        # Zero penny balance
        diff = abs(inv["net_revenue"] - (inv["gross_sales"] - inv["return_amount"]))
        assert diff < 0.005

    def test_pipeline_execution_logs(self, data_provider):
        df = data_provider.get_pipeline_execution_logs()
        assert not df.empty
        assert "run_id" in df.columns
        assert "execution_status" in df.columns


class TestDataProviderParquetFallback:
    """Verifies that DataProvider gracefully falls back to Parquet when PostgreSQL is down."""

    @patch.object(DataProvider, "is_database_connected", return_value=False)
    def test_fallback_kpis(self, mock_conn):
        dp = DataProvider()
        kpis = dp.get_executive_kpis()
        assert kpis["source"] == "Gold Parquet Fallback"
        assert np.isclose(kpis["net_revenue"], 19012910.85, atol=0.01)
        assert kpis["unique_customers"] == 5939

    @patch.object(DataProvider, "is_database_connected", return_value=False)
    def test_fallback_monthly_sales(self, mock_conn):
        dp = DataProvider()
        df = dp.get_monthly_sales()
        assert len(df) == 25
        assert np.isclose(df["net_revenue"].sum(), 19012910.85, atol=0.01)

    @patch.object(DataProvider, "is_database_connected", return_value=False)
    def test_fallback_cluster_profiles(self, mock_conn):
        dp = DataProvider()
        df = dp.get_cluster_profiles()
        assert len(df) == 5
        assert df["customer_count"].sum() == 5939


class TestPlotlyChartsGeneration:
    """Verifies that all Plotly chart generators output valid go.Figure objects."""

    def test_revenue_trend_chart(self, data_provider):
        df = data_provider.get_monthly_sales()
        fig = create_revenue_trend_chart(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # Gross, Returns, Net

    def test_order_volume_chart(self, data_provider):
        df = data_provider.get_monthly_sales()
        fig = create_order_volume_trend_chart(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Valid, Cancelled

    def test_top_countries_chart(self, data_provider):
        df = data_provider.get_country_revenue(limit=10)
        fig = create_top_countries_chart(df)
        assert isinstance(fig, go.Figure)

    def test_top_products_chart(self, data_provider):
        df = data_provider.get_top_products(limit=10)
        fig = create_top_products_chart(df)
        assert isinstance(fig, go.Figure)

    def test_cluster_charts(self, data_provider):
        df = data_provider.get_cluster_profiles()
        fig_pop = create_cluster_population_chart(df)
        fig_rev = create_cluster_revenue_chart(df)
        assert isinstance(fig_pop, go.Figure)
        assert isinstance(fig_rev, go.Figure)

    def test_rfm_scatter_chart(self, data_provider):
        df = data_provider.get_rfm_clusters()
        fig = create_rfm_scatter_chart(df, sample_size=50)
        assert isinstance(fig, go.Figure)

    def test_anomaly_distribution_chart(self, data_provider):
        summary = data_provider.get_anomaly_summary()
        fig = create_anomaly_distribution_chart(summary)
        assert isinstance(fig, go.Figure)

    def test_feature_importance_chart(self, data_provider):
        fi = data_provider.get_feature_importances()
        fig = create_feature_importance_chart(fi, metric="gain")
        assert isinstance(fig, go.Figure)

    def test_roc_and_calibration_charts(self):
        fig_roc = create_roc_curve_chart()
        fig_cal = create_calibration_curve_chart()
        assert isinstance(fig_roc, go.Figure)
        assert isinstance(fig_cal, go.Figure)
