"""
Page 1: Executive Overview.

High-level commercial KPIs, revenue trajectory, top markets, and top products.
Preserves the Silver/Gold accounting baseline: £19,012,910.85 net revenue, 5,939 customers.
"""
import streamlit as st
from src.dashboard.components.charts import (
    create_revenue_trend_chart,
    create_top_countries_chart,
    create_top_products_chart,
)
from src.dashboard.components.kpi_cards import render_kpi_row
from src.dashboard.config import format_gbp, format_number
from src.dashboard.data_provider import DataProvider


def render_page(data_provider: DataProvider) -> None:
    """Renders Executive Overview page."""
    st.title("📈 Executive Commercial Overview")
    st.markdown(
        "Executive-level commercial performance and enterprise portfolio health across the "
        "reconciled 25-month transactional dataset (December 2009 – December 2011)."
    )

    kpis = data_provider.get_executive_kpis()

    # Source Indicator Badge
    col_badge, _ = st.columns([1, 3])
    with col_badge:
        badge_color = "green" if "PostgreSQL" in kpis.get("source", "") else "orange"
        st.caption(f"Data Source: **:{badge_color}[{kpis.get('source')}]**")

    st.markdown("---")

    # Primary Financial KPI Row
    st.subheader("Financial Performance")
    render_kpi_row([
        {
            "label": "Net Commercial Revenue",
            "value": format_gbp(kpis["net_revenue"]),
            "delta": "+£19.01M total",
            "delta_color": "normal",
            "help_text": "Gross commercial sales minus customer returns (£20.48M - £1.46M)",
        },
        {
            "label": "Gross Sales",
            "value": format_gbp(kpis["gross_sales"]),
            "help_text": "Total positive transaction value before cancellations",
        },
        {
            "label": "Customer Returns",
            "value": format_gbp(kpis["return_amount"]),
            "delta": "-£1.46M (7.14%)",
            "delta_color": "inverse",
            "help_text": "Total refund and cancellation adjustments",
        },
        {
            "label": "Average Order Value (AOV)",
            "value": format_gbp(kpis["average_order_value"]),
            "help_text": "Net revenue divided by 44,870 distinct customer purchase orders",
        },
    ])

    st.markdown(" ")

    # Operational Volume KPI Row
    st.subheader("Operational & Customer Volumes")
    render_kpi_row([
        {
            "label": "Silver Transaction Lines",
            "value": format_number(kpis["total_silver_transactions"]),
            "help_text": "Total audited transaction line items in Silver Parquet repository",
        },
        {
            "label": "Valid Commercial Purchases",
            "value": format_number(kpis["valid_purchases"]),
            "help_text": "Eligible commercial sales transactions (excluding cancellations/adjustments)",
        },
        {
            "label": "Distinct Commercial Orders",
            "value": format_number(kpis["distinct_invoices"]),
            "help_text": "Unique commercial order invoices across identified customer accounts",
        },
        {
            "label": "Identified RFM Customers",
            "value": format_number(kpis["unique_customers"]),
            "help_text": "Unique commercial accounts audited across Gold customer RFM layer",
        },
    ])

    st.markdown("---")

    # Revenue Trajectory
    monthly_sales_df = data_provider.get_monthly_sales()
    st.plotly_chart(create_revenue_trend_chart(monthly_sales_df), use_container_width=True)

    # Secondary Charts Row: Countries & Products
    col1, col2 = st.columns(2)
    with col1:
        country_df = data_provider.get_country_revenue(limit=10)
        st.plotly_chart(create_top_countries_chart(country_df), use_container_width=True)

    with col2:
        products_df = data_provider.get_top_products(limit=10)
        st.plotly_chart(create_top_products_chart(products_df), use_container_width=True)
