"""
Page 7: Model Performance & Governance.

Cross-model operational registry, fairness/subgroup evaluations,
calibration analysis, and production machine learning governance guardrails.
"""
import pandas as pd
import streamlit as st
from src.dashboard.components.charts import create_calibration_curve_chart, create_roc_curve_chart
from src.dashboard.components.kpi_cards import render_kpi_row
from src.dashboard.components.tables import render_table
from src.dashboard.config import format_number, format_pct
from src.dashboard.data_provider import DataProvider


def render_page(data_provider: DataProvider) -> None:
    """Renders Model Performance & Governance page."""
    st.title("⚖️ Model Governance & Operational Performance")
    st.markdown(
        "Production ML model cards, demographic & behavioral slice evaluations, "
        "decision thresholds, calibration contracts, and AI ethical guardrails."
    )

    # Cross-Model Registry Overview
    st.subheader("Production Machine Learning Model Registry")
    
    registry_data = [
        {
            "Model Name": "Phase 4 Customer Segmentation",
            "Algorithm": "K-Means (k-means++)",
            "Task / Type": "Unsupervised Clustering",
            "Evaluation Metric": "Silhouette (0.3957), Davies-Bouldin (0.8716)",
            "Operational Threshold": "k=5 Clusters",
            "Artifact Path": "models/kmeans/customer_segmentation_kmeans.joblib",
            "Audit Status": "AUDITED & SIGNED OFF",
        },
        {
            "Model Name": "Phase 5 Anomaly Detection",
            "Algorithm": "Isolation Forest (100 Trees)",
            "Task / Type": "Unsupervised Outlier Detection",
            "Evaluation Metric": "Contamination: 1.00% (16,664 anomalies)",
            "Operational Threshold": "Score <= 0.0 (offset = -0.5)",
            "Artifact Path": "models/isolation_forest/isolation_forest_model.joblib",
            "Audit Status": "AUDITED & SIGNED OFF",
        },
        {
            "Model Name": "Phase 6 Repeat-Purchase Predictor",
            "Algorithm": "XGBoost + Platt Sigmoid",
            "Task / Type": "Supervised Binary Classification",
            "Evaluation Metric": "Test ROC-AUC: 0.7750, PR-AUC: 0.5180",
            "Operational Threshold": "t=0.59 (Raw), t=0.27 (Calibrated)",
            "Artifact Path": "models/repeat_purchase/calibrated_xgb_model.joblib",
            "Audit Status": "AUDITED & SIGNED OFF",
        },
    ]
    st.dataframe(pd.DataFrame(registry_data), hide_index=True, use_container_width=True)

    st.markdown("---")

    # Repeat Purchase Behavioral Subgroup Analysis
    st.subheader("Subgroup Fairness & Sensitivity Breakdown (Activity Tiers)")
    st.markdown(
        "Performance evaluation across customer engagement tiers on the held-out test set (5,438 customers). "
        "Demonstrates model sensitivity and recall variations across historical order velocity."
    )

    metrics_dict = data_provider.get_repeat_purchase_metrics()
    subgroups = metrics_dict["metrics"]["test"].get("activity_subgroups_xgboost", {})

    subgroup_rows = []
    for grp_key, grp_vals in subgroups.items():
        name = grp_key.replace("_", " ").title()
        subgroup_rows.append({
            "Customer Activity Tier": name,
            "Cohort Size": format_number(grp_vals.get("count", 0)),
            "True Positive Repeaters": format_number(grp_vals.get("positives", 0)),
            "Base Prevalence": format_pct(grp_vals.get("prevalence_pct", 0) / 100),
            "Subgroup ROC-AUC": f"{grp_vals.get('roc_auc', 0):.4f}",
            "Subgroup PR-AUC": f"{grp_vals.get('pr_auc', 0):.4f}",
            "Subgroup Precision": format_pct(grp_vals.get("precision", 0)),
            "Subgroup Recall": format_pct(grp_vals.get("recall", 0)),
        })
    st.dataframe(pd.DataFrame(subgroup_rows), hide_index=True, use_container_width=True)

    st.markdown("---")

    # Model Performance Visualizations
    st.subheader("Model Diagnostic Curves")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_roc_curve_chart(), use_container_width=True)
    with col2:
        st.plotly_chart(create_calibration_curve_chart(), use_container_width=True)

    st.markdown("---")

    # Governance Guardrails & Operational Directives
    st.subheader("Operational Guidelines & AI Governance Contracts")

    with st.expander("📌 Customer Segmentation (Phase 4) Governance", expanded=True):
        st.markdown(
            "- **Descriptive Non-Causal Policy**: Cluster archetypes describe historical transactional behavior. "
            "They must **never** be used to deny service, alter pricing, or assume malicious intent.\n"
            "- **Drift Monitoring**: Clusters should be re-evaluated quarterly. If silhouette score drops below 0.30, "
            "trigger retrain evaluation pipeline.\n"
            "- **Reconciliation Parity**: Every customer assignment must reconcile against Gold RFM customer identifiers."
        )

    with st.expander("📌 Transaction Anomaly Detection (Phase 5) Governance", expanded=True):
        st.markdown(
            "- **Non-Fraud Directives**: Isolation Forest outliers are statistical deviations in high-dimensional feature space. "
            "Operational teams must **never** treat an anomaly flag as confirmed commercial fraud.\n"
            "- **Dual Review Gate**: Outlier flags with score < -0.10 require human-in-the-loop review before automated action.\n"
            "- **Preservation Contract**: All 1,067,371 Silver transactions remain intact; anomaly scores enrich records without row dropping."
        )

    with st.expander("📌 Repeat-Purchase Prediction (Phase 6) Governance", expanded=True):
        st.markdown(
            "- **Probability Calibration Mandate**: Always consume Sigmoid Calibrated probabilities when evaluating "
            "expected commercial return. Using raw uncalibrated probabilities will over-estimate conversion probability.\n"
            "- **Threshold Policy Contract**: If consuming raw probabilities, enforce decision threshold $t = 0.5900$; "
            "if consuming calibrated probabilities, enforce decision threshold $t = 0.2700$.\n"
            "- **Temporal Cutoff Contract**: Feature engineering must strictly enforce historical snapshot cutoffs. "
            "Predictions must display observation dates explicitly."
        )
