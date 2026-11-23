"""
Main Streamlit Application Entrypoint.

E-Commerce Intelligence Platform — Enterprise Lakehouse Analytics & Predictive Machine Learning.
Integrates live PostgreSQL 16 database and Gold Parquet fallback layers.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.dashboard.config import DashboardConfig
from src.dashboard.data_provider import DataProvider
from src.dashboard.pages import (
    anomaly_detection,
    customer_segmentation,
    data_quality,
    executive_overview,
    model_governance,
    repeat_purchase,
    sales_analytics,
)

# 1. Page Configuration
st.set_page_config(
    page_title=DashboardConfig.APP_TITLE,
    page_icon=DashboardConfig.APP_ICON,
    layout=DashboardConfig.LAYOUT,
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_data_provider() -> DataProvider:
    """Initializes and caches DataProvider instance across sessions."""
    return DataProvider()


def main() -> None:
    """Main application orchestrator and router."""
    data_provider = get_data_provider()
    is_live = data_provider.is_database_connected()

    # 2. Sidebar Navigation
    st.sidebar.title(f"{DashboardConfig.APP_ICON} E-Commerce Intelligence")
    st.sidebar.caption("Enterprise Lakehouse & Predictive ML")

    # Connection Status Indicator
    if is_live:
        st.sidebar.success("🟢 Connected: PostgreSQL 16 (Live)")
    else:
        st.sidebar.warning("🟡 Active: Parquet Fallback Mode")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    pages = {
        "Executive Overview": executive_overview.render_page,
        "Sales & Revenue Analytics": sales_analytics.render_page,
        "Customer RFM Segmentation": customer_segmentation.render_page,
        "Transaction Anomaly Detection": anomaly_detection.render_page,
        "Repeat-Purchase Prediction": repeat_purchase.render_page,
        "Data Quality & Pipeline Monitoring": data_quality.render_page,
        "Model Performance & Governance": model_governance.render_page,
    }

    selected_page_name = st.sidebar.radio(
        "Select Analytics Domain",
        options=list(pages.keys()),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 E-Commerce Intelligence Platform | Production v1.0.0")

    # 3. Page Rendering with Error Boundary
    render_fn = pages.get(selected_page_name)
    if render_fn:
        try:
            render_fn(data_provider)
        except Exception as e:
            st.error(f"⚠️ An error occurred while rendering the {selected_page_name} view.")
            st.exception(e)


if __name__ == "__main__":
    main()
