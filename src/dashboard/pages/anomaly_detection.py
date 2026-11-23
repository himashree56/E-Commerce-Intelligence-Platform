"""
Page 4: Transaction Anomaly Detection.

Presents Phase 5 Isolation Forest unsupervised anomaly detection results.
Includes summary KPIs, status breakdown, score distribution, diagnostic reason codes,
transaction search, and statistical anomaly (non-fraud) disclaimers.
"""
import pandas as pd
import streamlit as st
from src.dashboard.components.charts import create_anomaly_distribution_chart
from src.dashboard.components.kpi_cards import render_kpi_row
from src.dashboard.components.tables import render_table
from src.dashboard.config import format_gbp, format_number, format_pct
from src.dashboard.data_provider import DataProvider


def render_page(data_provider: DataProvider) -> None:
    """Renders Transaction Anomaly Detection page."""
    st.title("🚨 Transaction Anomaly Detection (Isolation Forest)")
    st.markdown(
        "Unsupervised anomaly detection results across all 1,067,371 Silver transaction lines, "
        "trained on 1,027,015 valid commercial and cancellation events."
    )

    # Statistical Disclaimer
    st.warning(
        "⚠️ **Statistical Disclaimer**: Flagged transactions are multi-dimensional statistical outliers "
        "detected by an Isolation Forest model. An anomaly flag indicates statistical rarity in transaction "
        "patterns (such as extreme price, quantity, or unusual customer cadence); it does **NOT** constitute "
        "evidence or confirmation of commercial fraud, theft, or illicit behavior."
    )

    summary = data_provider.get_anomaly_summary()

    # Overview KPI row
    total_trans = summary["total_scored_transactions"]
    total_anoms = summary["total_anomalies"]
    rate_pct = summary["anomaly_rate_pct"]

    render_kpi_row([
        {"label": "Total Scored Transactions", "value": format_number(total_trans)},
        {"label": "Flagged Outliers", "value": format_number(total_anoms)},
        {"label": "Overall Anomaly Rate", "value": f"{rate_pct:.2f}%"},
        {"label": "Model Algorithm", "value": "Isolation Forest (100 Trees)"},
    ])

    st.markdown("---")

    # Status Breakdown Table & Chart
    st.subheader("Anomaly Prevalence by Transaction Status")
    st.caption("Distribution of flagged anomalies across Silver transaction status classifications.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(create_anomaly_distribution_chart(summary), use_container_width=True)

    with col2:
        breakdown_rows = []
        for status, stats in summary.get("status_breakdown", {}).items():
            breakdown_rows.append({
                "Status": status.replace("_", " ").title(),
                "Total Transactions": format_number(stats.get("total", 0)),
                "Flagged Anomalies": format_number(stats.get("anomalies", 0)),
                "Anomaly Rate": format_pct(stats.get("rate", 0)),
            })
        st.dataframe(pd.DataFrame(breakdown_rows), hide_index=True, use_container_width=True)

    st.markdown("---")

    # Interactive Transaction Filter & Table
    st.subheader("Audited Anomaly Sample & Diagnostic Reason Codes")
    
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        status_filter = st.selectbox(
            "Transaction Status",
            options=["ALL", "VALID_PURCHASE", "CANCELLATION", "INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY", "DUPLICATE"],
        )
    with col_f2:
        limit_val = st.selectbox("Sample Size", options=[25, 50, 100, 250], index=1)
    with col_f3:
        search_kw = st.text_input("Search Invoice No, Stock Code, or Customer ID:", "")

    anomalies_df = data_provider.get_high_priority_anomalies(
        limit=limit_val,
        status_filter=status_filter if status_filter != "ALL" else None,
        search_term=search_kw.strip() if search_kw.strip() else None,
    )

    if not anomalies_df.empty:
        # Format table for display
        cols_to_use = [
            "transaction_id",
            "invoice_no",
            "stock_code",
            "quantity",
            "unit_price",
            "line_total",
            "anomaly_score",
            "diagnostic_reasons",
            "transaction_status",
        ]
        # Include only available columns
        available_cols = [c for c in cols_to_use if c in anomalies_df.columns]
        disp_df = anomalies_df[available_cols].copy()

        if "unit_price" in disp_df.columns:
            disp_df["unit_price"] = disp_df["unit_price"].apply(format_gbp)
        if "line_total" in disp_df.columns:
            disp_df["line_total"] = disp_df["line_total"].apply(format_gbp)
        if "quantity" in disp_df.columns:
            disp_df["quantity"] = disp_df["quantity"].apply(format_number)
        if "anomaly_score" in disp_df.columns:
            disp_df["anomaly_score"] = disp_df["anomaly_score"].apply(lambda v: f"{v:.4f}")
        if "diagnostic_reasons" in disp_df.columns:
            disp_df["diagnostic_reasons"] = disp_df["diagnostic_reasons"].apply(
                lambda r: ", ".join(r) if isinstance(r, list) else str(r or "")
            )

        col_rename = {
            "transaction_id": "Transaction ID",
            "invoice_no": "Invoice No",
            "stock_code": "Stock Code",
            "quantity": "Quantity",
            "unit_price": "Unit Price",
            "line_total": "Line Total",
            "anomaly_score": "Anomaly Score",
            "diagnostic_reasons": "Diagnostic Reasons",
            "transaction_status": "Status",
        }
        disp_df.rename(columns=col_rename, inplace=True)

        render_table(
            disp_df,
            download_filename="audited_transaction_anomalies.csv",
            height=400,
        )
    else:
        st.info("No anomalies match the current filters.")
