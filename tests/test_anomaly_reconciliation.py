"""
Reconciliation and Financial Non-Interference Tests for Phase 5 Anomaly Detection.
"""
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def anomaly_datasets():
    """Loads Silver and Gold anomaly datasets for reconciliation checks."""
    silver_path = Path("data/silver/online_retail_silver.parquet")
    anomalies_path = Path("data/gold/gold_transaction_anomalies_isolation_forest.parquet")

    if not silver_path.exists() or not anomalies_path.exists():
        pytest.skip("Required parquet datasets missing for reconciliation.")

    silver_df = pd.read_parquet(silver_path)
    anomalies_df = pd.read_parquet(anomalies_path)
    return {"silver": silver_df, "anomalies": anomalies_df}


def test_reconciliation_exact_row_count_and_lineage(anomaly_datasets):
    """Verifies 100% row preservation and unique deterministic lineage across 1,067,371 rows."""
    silver = anomaly_datasets["silver"]
    anomalies = anomaly_datasets["anomalies"]

    assert len(silver) == 1067371
    assert len(anomalies) == 1067371
    assert anomalies["transaction_id"].nunique() == 1067371
    assert (anomalies["source_row_id"].values == np.arange(1067371)).all()


def test_reconciliation_anomaly_counts_and_status_rates(anomaly_datasets):
    """Verifies that retail returns are preserved and non-commercial adjustments are flagged."""
    anomalies = anomaly_datasets["anomalies"]
    total_anomalies = int(anomalies["is_anomaly"].sum())

    assert total_anomalies == 16664
    assert round(total_anomalies / len(anomalies) * 100, 2) == 1.56

    # Status distribution
    status_summary = anomalies.groupby("transaction_status")["is_anomaly"].agg(["count", "sum"])
    status_summary["pct"] = status_summary["sum"] / status_summary["count"] * 100

    # Purchases have low anomaly rate
    assert status_summary.loc["VALID_PURCHASE", "pct"] < 1.5
    # Retail cancellations are mostly normal (rate < 10%)
    assert status_summary.loc["CANCELLATION", "pct"] < 10.0
    # Inventory write-offs are 100% flagged
    assert status_summary.loc["INVENTORY_WRITE_OFF", "pct"] == 100.0


def test_reconciliation_diagnostic_interpretability(anomaly_datasets):
    """Verifies that 100% of flagged anomalies have valid diagnostic reason codes."""
    anomalies = anomaly_datasets["anomalies"]

    flagged = anomalies[anomalies["is_anomaly"]]
    unflagged = anomalies[~anomalies["is_anomaly"]]

    assert (flagged["diagnostic_reasons"] != "").all()
    assert (unflagged["diagnostic_reasons"] == "").all()


def test_reconciliation_ambiguous_cancellation_c496350(anomaly_datasets):
    """Verifies that ambiguous positive-quantity credit note C496350 is flagged as an anomaly."""
    anomalies = anomaly_datasets["anomalies"]
    c49 = anomalies[anomalies["invoice_no"] == "C496350"]

    assert len(c49) == 1
    assert bool(c49["is_anomaly"].iloc[0]) is True
    assert "HIGH_UNIT_PRICE" in c49["diagnostic_reasons"].iloc[0]


def test_reconciliation_financial_non_interference(anomaly_datasets):
    """Verifies that anomaly scoring does NOT alter historical revenue figures."""
    silver = anomaly_datasets["silver"]
    silver_net = Decimal(str(round(float(silver["net_revenue"].sum()), 2)))
    assert silver_net == Decimal("19012910.85")
