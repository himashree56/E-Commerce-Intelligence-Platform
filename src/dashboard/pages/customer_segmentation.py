"""
Page 3: Customer RFM Segmentation.

Explores Phase 4 K-Means customer segments (k=5), population & revenue distributions,
3D/2D RFM scatter, rule-based cross-tabulation, and individual customer profile lookup.
"""
import streamlit as st
from src.dashboard.components.charts import (
    create_cluster_population_chart,
    create_cluster_revenue_chart,
    create_rfm_scatter_chart,
)
from src.dashboard.components.kpi_cards import render_kpi_row
from src.dashboard.components.tables import render_table
from src.dashboard.config import CLUSTER_NAMES, format_gbp, format_number, format_pct
from src.dashboard.data_provider import DataProvider


def render_page(data_provider: DataProvider) -> None:
    """Renders Customer RFM Segmentation page."""
    st.title("👥 Customer RFM Segmentation (K-Means)")
    st.markdown(
        "Audited 5-cluster segmentation derived from log-transformed and standardized Recency, "
        "Frequency, and Monetary metrics across all 5,939 identified commercial customer accounts."
    )

    # Descriptive Non-Causal Notice
    st.info(
        "ℹ️ **Descriptive Notice**: Cluster names and profiles represent empirical behavioral groupings "
        "observed over historical transactions. They are non-causal descriptions, not guarantees of "
        "future individual purchasing behavior or operational intent."
    )

    profiles_df = data_provider.get_cluster_profiles()

    # Overview KPIs
    total_cust = profiles_df["customer_count"].sum()
    total_rev = profiles_df["total_revenue"].sum()

    render_kpi_row([
        {"label": "Total Segmented Accounts", "value": format_number(total_cust)},
        {"label": "Segmented Portfolio Revenue", "value": format_gbp(total_rev)},
        {"label": "Optimal Cluster Count (k)", "value": "5 Clusters"},
        {"label": "Model Algorithm", "value": "K-Means (k-means++)"},
    ])

    st.markdown("---")

    # Cluster Cards Breakdown
    st.subheader("Cluster Profile Archetypes")
    cols = st.columns(5)
    for idx, row in profiles_df.iterrows():
        cid = int(row["cluster_id"])
        with cols[cid]:
            with st.container(border=True):
                st.markdown(f"**Cluster {cid}**")
                st.markdown(f"*{row['cluster_name']}*")
                st.metric("Customers", f"{int(row['customer_count']):,}", f"{row['pct_of_customers']:.1f}%")
                st.metric("Revenue", format_gbp(row["total_revenue"]), f"{row['pct_of_revenue']:.1f}%")
                st.caption(
                    f"Avg Recency: **{int(round(row['mean_recency']))}d**<br>"
                    f"Avg Frequency: **{row['mean_frequency']:.1f}**<br>"
                    f"Avg Spend: **{format_gbp(row['mean_monetary'])}**",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Cluster Distribution Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(create_cluster_population_chart(profiles_df), use_container_width=True)
    with col_chart2:
        st.plotly_chart(create_cluster_revenue_chart(profiles_df), use_container_width=True)

    st.markdown("---")

    # Interactive RFM Scatter
    st.subheader("Interactive RFM Spatial Distribution")
    st.caption("Visualizing customer positioning across Recency, Frequency, and Monetary dimensions.")
    rfm_df = data_provider.get_rfm_clusters()
    st.plotly_chart(create_rfm_scatter_chart(rfm_df), use_container_width=True)

    st.markdown("---")

    # Cross-Tabulation Matrix: Rule-Based RFM vs K-Means
    st.subheader("Cross-Tabulation: Rule-Based RFM vs. K-Means Segments")
    st.caption(
        "Reconciling traditional rule-based quintile segments against unsupervised K-Means clusters (5,939 total accounts)."
    )
    crosstab_df = data_provider.get_rfm_crosstab()
    render_table(
        crosstab_df,
        download_filename="rfm_rule_vs_kmeans_crosstab.csv",
        hide_index=False,
    )

    st.markdown("---")

    # Individual Customer Lookup
    st.subheader("🔍 Individual Customer Profile Lookup")
    lookup_id = st.text_input("Enter 5-digit Customer ID (e.g., 18102, 14646, 12346):", "")
    if lookup_id.strip():
        try:
            cid_val = int(lookup_id.strip())
            matched = rfm_df[rfm_df["customer_id"] == cid_val]
            if not matched.empty:
                cust = matched.iloc[0]
                cluster_id = int(cust["cluster_id"])
                cname = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
                
                c_card1, c_card2, c_card3, c_card4 = st.columns(4)
                c_card1.metric("Assigned Cluster", f"{cluster_id} - {cname}")
                c_card2.metric("Recency", f"{int(cust['recency'])} days")
                c_card3.metric("Frequency", f"{int(cust['frequency'])} orders")
                c_card4.metric("Monetary Net Spend", format_gbp(cust["monetary"]))
            else:
                st.warning(f"Customer ID {cid_val} not found in the 5,939 Gold RFM customer cohort.")
        except ValueError:
            st.error("Please enter a valid numeric Customer ID.")
