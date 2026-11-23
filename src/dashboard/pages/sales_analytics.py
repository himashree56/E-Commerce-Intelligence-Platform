"""
Page 2: Sales & Revenue Analytics.

Detailed monthly breakdown, product performance, international market analysis,
and cancellation/refund impact.
"""
import pandas as pd
import streamlit as st
from src.dashboard.components.charts import (
    create_order_volume_trend_chart,
    create_revenue_trend_chart,
    create_top_countries_chart,
)
from src.dashboard.components.kpi_cards import render_kpi_row
from src.dashboard.components.tables import render_table
from src.dashboard.config import format_gbp, format_number, format_pct
from src.dashboard.data_provider import DataProvider


def render_page(data_provider: DataProvider) -> None:
    """Renders Sales & Revenue Analytics page."""
    st.title("📊 Sales & Commercial Revenue Analytics")
    st.markdown(
        "Interactive exploration of monthly sales trajectories, order volume, "
        "product catalog velocity, and customer return margins."
    )

    monthly_df = data_provider.get_monthly_sales()
    country_df = data_provider.get_country_revenue(limit=50)

    # Filtering controls
    st.sidebar.markdown("### Sales Filters")
    available_months = sorted(monthly_df["invoice_year_month"].unique().tolist())
    selected_range = st.sidebar.select_slider(
        "Select Time Window",
        options=available_months,
        value=(available_months[0], available_months[-1]),
    )

    all_countries = ["All Markets"] + sorted(country_df["country"].unique().tolist())
    selected_country = st.sidebar.selectbox("Filter by Country", options=all_countries)

    # Filter monthly dataframe
    mask = (monthly_df["invoice_year_month"] >= selected_range[0]) & (
        monthly_df["invoice_year_month"] <= selected_range[1]
    )
    filtered_monthly = monthly_df[mask]

    # Metrics summary for the selected period
    period_gross = filtered_monthly["gross_sales"].sum()
    period_returns = filtered_monthly["return_amount"].sum()
    period_net = filtered_monthly["net_revenue"].sum()
    period_orders = filtered_monthly["valid_orders_count"].sum()
    period_cancels = filtered_monthly["cancelled_orders_count"].sum()
    cancellation_rate = period_cancels / max(1, period_orders + period_cancels)

    render_kpi_row([
        {"label": "Selected Period Net Revenue", "value": format_gbp(period_net)},
        {"label": "Selected Gross Sales", "value": format_gbp(period_gross)},
        {"label": "Selected Returns", "value": format_gbp(period_returns)},
        {"label": "Cancellation Rate", "value": format_pct(cancellation_rate), "delta_color": "inverse"},
    ])

    st.markdown("---")

    # Revenue and Order Volume Trends
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_revenue_trend_chart(filtered_monthly), use_container_width=True)
    with col2:
        st.plotly_chart(create_order_volume_trend_chart(filtered_monthly), use_container_width=True)

    st.markdown("---")

    # Product Performance Table
    st.subheader("Top Product Catalog Performance")
    st.caption("Top commercial products ranked by net sales with unit return margins.")
    
    products_df = data_provider.get_top_products(limit=50)
    
    # Format table columns for presentation
    disp_products = products_df[[
        "stock_code",
        "description",
        "gross_sales",
        "return_amount",
        "net_revenue",
        "units_sold",
        "units_returned",
        "net_units",
        "distinct_customers",
    ]].copy()

    disp_products["gross_sales"] = disp_products["gross_sales"].apply(lambda v: format_gbp(v, 2))
    disp_products["return_amount"] = disp_products["return_amount"].apply(lambda v: format_gbp(v, 2))
    disp_products["net_revenue"] = disp_products["net_revenue"].apply(lambda v: format_gbp(v, 2))
    disp_products["units_sold"] = disp_products["units_sold"].apply(format_number)
    disp_products["units_returned"] = disp_products["units_returned"].apply(format_number)
    disp_products["net_units"] = disp_products["net_units"].apply(format_number)
    disp_products["distinct_customers"] = disp_products["distinct_customers"].apply(format_number)

    disp_products.columns = [
        "Stock Code",
        "Description",
        "Gross Sales",
        "Return Value",
        "Net Revenue",
        "Gross Units",
        "Units Returned",
        "Net Units",
        "Customers",
    ]

    render_table(
        disp_products,
        download_filename="top_products_performance.csv",
        height=350,
    )

    st.markdown("---")

    # International Geographic Markets
    st.subheader("International Revenue Distribution")
    col_geo1, col_geo2 = st.columns([1, 1])
    with col_geo1:
        st.plotly_chart(create_top_countries_chart(country_df.head(15)), use_container_width=True)
    with col_geo2:
        disp_country = country_df[[
            "country",
            "net_revenue",
            "gross_sales",
            "return_amount",
            "distinct_customers",
            "valid_orders_count",
        ]].copy()
        disp_country["net_revenue"] = disp_country["net_revenue"].apply(format_gbp)
        disp_country["gross_sales"] = disp_country["gross_sales"].apply(format_gbp)
        disp_country["return_amount"] = disp_country["return_amount"].apply(format_gbp)
        disp_country["distinct_customers"] = disp_country["distinct_customers"].apply(format_number)
        disp_country["valid_orders_count"] = disp_country["valid_orders_count"].apply(format_number)
        disp_country.columns = ["Country", "Net Revenue", "Gross Sales", "Returns", "Customers", "Orders"]

        render_table(
            disp_country.head(15),
            download_filename="international_market_revenue.csv",
            height=380,
        )
