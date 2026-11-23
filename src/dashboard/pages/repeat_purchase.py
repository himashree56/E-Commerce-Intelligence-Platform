"""
Page 5: Repeat-Purchase Prediction.

XGBoost supervised predictive modeling for 30-day customer re-engagement.
Supports Raw vs. Calibrated probabilities, decision thresholds (0.59 vs 0.27),
propensity tiers, individual customer predictions, and feature importance.
"""
import streamlit as st
from src.dashboard.components.charts import (
    create_calibration_curve_chart,
    create_feature_importance_chart,
    create_roc_curve_chart,
)
from src.dashboard.components.kpi_cards import render_kpi_row
from src.dashboard.components.tables import render_table
from src.dashboard.config import format_gbp, format_number, format_pct
from src.dashboard.data_provider import DataProvider


def render_page(data_provider: DataProvider) -> None:
    """Renders Repeat-Purchase Prediction page."""
    st.title("🎯 Customer Repeat-Purchase Prediction (XGBoost)")
    st.markdown(
        "Supervised machine learning pipeline predicting whether an active commercial customer "
        "will make at least one valid purchase within a 30-day forward window."
    )

    # Historical Snapshot Evaluation Notice
    st.info(
        "📅 **Historical Snapshot Notice**: Predictions and metrics presented on this page are derived from a "
        "rigorous temporal snapshot at **2011-10-01**, predicting customer re-orders through **2011-10-31**. "
        "They represent out-of-time test evaluations and are not live real-time signals."
    )

    metrics_data = data_provider.get_repeat_purchase_metrics()
    test_metrics = metrics_data["metrics"]["test"]

    # Calibration / Threshold Toggle Controls
    st.sidebar.markdown("### Model Probability Settings")
    prob_mode = st.sidebar.radio(
        "Probability Calibration",
        options=["Sigmoid Calibrated (Platt Scaling)", "Raw XGBoost (Uncalibrated)"],
        index=0,
        help="Calibrated probabilities map to true empirical base rates; uncalibrated probabilities are elevated by positive class reweighting.",
    )
    use_calibrated = "Sigmoid" in prob_mode
    active_model_key = "xgboost_calibrated_sigmoid" if use_calibrated else "xgboost"
    active_m = test_metrics[active_model_key]

    threshold_val = active_m["threshold"]

    # Model Performance KPIs
    render_kpi_row([
        {"label": "Held-Out Test ROC-AUC", "value": f"{active_m['roc_auc']:.4f}"},
        {"label": "Precision-Recall AUC", "value": f"{active_m['pr_auc']:.4f}"},
        {"label": "Precision @ Top 5%", "value": format_pct(active_m["precision_at_top_5_pct"])},
        {"label": "Decision Threshold (t)", "value": f"{threshold_val:.2f}"},
    ])

    render_kpi_row([
        {"label": "Test Classification F1", "value": f"{active_m['f1']:.4f}"},
        {"label": "Test Precision", "value": format_pct(active_m["precision"])},
        {"label": "Test Recall (Sensitivity)", "value": format_pct(active_m["recall"])},
        {"label": "Brier Score (Loss)", "value": f"{active_m['brier_score']:.4f}"},
    ])

    st.markdown("---")

    # Threshold Policy Callout
    policy = metrics_data.get("policy", {})
    st.subheader("Decision Threshold & Calibration Policy")
    st.markdown(
        f"- **Optimal Validation-Tuned Threshold**: `{threshold_val:.2f}` "
        f"({ 'Calibrated Platt threshold' if use_calibrated else 'Raw threshold tuned on 2011-08-01 validation set' })\n"
        f"- **Objective**: Maximize $F_1$ score balancing marketing intervention costs against recall.\n"
        f"- **Reliability Notice**: The raw XGBoost model uses `scale_pos_weight = 4.081` to handle class imbalance, "
        f"which shifts raw output probabilities rightward. Applying the Sigmoid calibrator restores monotonic empirical calibration."
    )

    st.markdown("---")

    # Evaluation Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_roc_curve_chart(), use_container_width=True)
    with col2:
        st.plotly_chart(create_calibration_curve_chart(), use_container_width=True)

    st.markdown("---")

    # Feature Importance Section
    st.subheader("Feature Importance Attribution")
    importance_type = st.radio(
        "Importance Metric",
        options=["gain", "weight", "cover"],
        index=0,
        horizontal=True,
        help="Gain: relative contribution per feature split. Weight: number of times a feature splits trees. Cover: relative count of observations.",
    )
    fi_dict = data_provider.get_feature_importances()
    st.plotly_chart(create_feature_importance_chart(fi_dict, metric=importance_type), use_container_width=True)

    st.markdown("---")

    # Prediction Cohort Explorer
    st.subheader("Held-Out Customer Cohort Predictions (Snapshot: 2011-10-01)")
    
    col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
    with col_c1:
        tier_choice = st.selectbox(
            "Propensity Tier",
            options=["ALL", "High Propensity", "Medium Propensity", "Low Propensity"],
        )
    with col_c2:
        limit_size = st.selectbox("Record Limit", options=[25, 50, 100, 250], index=1)
    with col_c3:
        search_cust_input = st.text_input("Search Specific Customer ID:", "")

    pred_df = data_provider.get_repeat_purchase_predictions(
        limit=limit_size,
        use_calibrated=use_calibrated,
        tier=tier_choice if tier_choice != "ALL" else None,
        search_customer=search_cust_input.strip() if search_cust_input.strip() else None,
    )

    if not pred_df.empty:
        disp_pred = pred_df.copy()
        if "repeat_purchase_probability" in disp_pred.columns:
            disp_pred["repeat_purchase_probability"] = disp_pred["repeat_purchase_probability"].apply(lambda p: f"{float(p):.4f}")
        if "predicted_label" in disp_pred.columns:
            disp_pred["predicted_label"] = disp_pred["predicted_label"].apply(lambda b: "YES" if b else "NO")
        if "net_monetary_value" in disp_pred.columns:
            disp_pred["net_monetary_value"] = disp_pred["net_monetary_value"].apply(format_gbp)
        if "purchase_frequency" in disp_pred.columns:
            disp_pred["purchase_frequency"] = disp_pred["purchase_frequency"].apply(format_number)
        if "recency_days" in disp_pred.columns:
            disp_pred["recency_days"] = disp_pred["recency_days"].apply(format_number)
        if "purchases_last_90d" in disp_pred.columns:
            disp_pred["purchases_last_90d"] = disp_pred["purchases_last_90d"].apply(format_number)

        drop_cols = [c for c in ["feature_snapshot", "scoring_timestamp", "created_at", "prediction_id"] if c in disp_pred.columns]
        if drop_cols:
            disp_pred.drop(columns=drop_cols, inplace=True)

        col_rename = {
            "customer_id": "Customer ID",
            "prediction_as_of_date": "Snapshot Date",
            "prediction_horizon_days": "Horizon (Days)",
            "repeat_purchase_probability": "Propensity Probability",
            "predicted_label": "Predicted Repeat Buyer",
            "model_version": "Model Version",
            "purchase_frequency": "Historical Frequency",
            "net_monetary_value": "Net Spend (£)",
            "recency_days": "Recency (Days)",
            "purchases_last_90d": "Orders (Last 90d)",
        }
        disp_pred.rename(columns=col_rename, inplace=True)

        render_table(
            disp_pred,
            download_filename="customer_repeat_predictions.csv",
            height=380,
        )
    else:
        st.info("No prediction records matched the filter criteria.")
