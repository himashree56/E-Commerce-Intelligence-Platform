"""
Tests for Phase 5 Anomaly Detection Model (TransactionAnomalyDetector).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_detector import (
    DiagnosticRuleExplainer,
    TransactionAnomalyDetector,
)
from src.models.config import AnomalyDetectionConfig


@pytest.fixture
def synthetic_training_data() -> pd.DataFrame:
    """Fixture providing a synthetic commercial transaction dataset for testing."""
    np.random.seed(42)
    n = 1000
    quantities = np.random.randint(1, 30, size=n)
    prices = np.random.uniform(1.0, 20.0, size=n)

    # Inject 10 extreme transactions
    quantities[:5] = [5000, 10000, 20000, 500, 800]
    prices[5:10] = [500.0, 1200.0, 800.0, 300.0, 450.0]

    return pd.DataFrame({
        "invoice_no": [f"INV_{i:04d}" for i in range(n)],
        "stock_code": [f"STK_{i % 50:02d}" for i in range(n)],
        "quantity": quantities,
        "unit_price": prices,
        "invoice_date": pd.date_range("2010-01-01", periods=n, freq="h"),
        "customer_id": np.random.choice([1001, 1002, 1003, None], size=n),
        "transaction_status": ["VALID_PURCHASE"] * n,
        "is_duplicate": [False] * n,
    })


def test_detector_fit_and_score_direction(synthetic_training_data):
    """Verifies that Isolation Forest fits and score direction is consistent:

    raw_decision_score < 0 for anomalies, anomaly_score > 0 for anomalies.
    """
    config = AnomalyDetectionConfig(contamination=0.01, random_state=42)
    detector = TransactionAnomalyDetector(config=config)
    detector.fit(synthetic_training_data)

    assert detector.is_fitted
    scored = detector.predict(synthetic_training_data)

    # Score direction check
    anomalies = scored[scored["is_anomaly"]]
    inliers = scored[~scored["is_anomaly"]]

    assert len(anomalies) > 0
    assert (anomalies["raw_decision_score"] < 0.0).all()
    assert (anomalies["anomaly_score"] > 0.0).all()
    assert (inliers["raw_decision_score"] >= 0.0).all()
    assert (inliers["anomaly_score"] <= 0.0).all()


def test_detector_reproducibility(synthetic_training_data):
    """Verifies that identical random_state generates identical anomaly scores and flags."""
    cfg1 = AnomalyDetectionConfig(random_state=42, n_estimators=50)
    cfg2 = AnomalyDetectionConfig(random_state=42, n_estimators=50)

    m1 = TransactionAnomalyDetector(config=cfg1).fit(synthetic_training_data)
    m2 = TransactionAnomalyDetector(config=cfg2).fit(synthetic_training_data)

    s1 = m1.predict(synthetic_training_data)
    s2 = m2.predict(synthetic_training_data)

    assert np.allclose(s1["raw_decision_score"].values, s2["raw_decision_score"].values)
    assert (s1["is_anomaly"].values == s2["is_anomaly"].values).all()


def test_contamination_monotonicity(synthetic_training_data):
    """Verifies that increasing contamination increases flagged anomaly count monotonically."""
    c_low = AnomalyDetectionConfig(contamination=0.01, random_state=42, n_estimators=50)
    c_high = AnomalyDetectionConfig(contamination=0.05, random_state=42, n_estimators=50)

    m_low = TransactionAnomalyDetector(config=c_low).fit(synthetic_training_data)
    m_high = TransactionAnomalyDetector(config=c_high).fit(synthetic_training_data)

    s_low = m_low.predict(synthetic_training_data)
    s_high = m_high.predict(synthetic_training_data)

    assert s_high["is_anomaly"].sum() > s_low["is_anomaly"].sum()


def test_detector_serialization_and_reloading(synthetic_training_data, tmp_path):
    """Verifies that model artifacts persist and reload with 100% prediction parity."""
    model_dir = tmp_path / "test_anomaly_model"
    detector = TransactionAnomalyDetector().fit(synthetic_training_data)
    detector.save(model_dir)

    # Required artifact files must exist
    required_files = [
        "isolation_forest_model.joblib",
        "preprocessor.joblib",
        "feature_schema.json",
        "hyperparameters.json",
        "training_metadata.json",
        "threshold_policy.json",
        "evaluation_metrics.json",
    ]
    for rf in required_files:
        assert (model_dir / rf).exists(), f"Missing artifact: {rf}"

    # Reload model
    reloaded = TransactionAnomalyDetector.load(model_dir)
    assert reloaded.is_fitted

    preds_orig = detector.predict(synthetic_training_data)
    preds_reloaded = reloaded.predict(synthetic_training_data)

    assert np.allclose(
        preds_orig["raw_decision_score"].values,
        preds_reloaded["raw_decision_score"].values,
    )
    assert (preds_orig["is_anomaly"].values == preds_reloaded["is_anomaly"].values).all()
    assert (preds_orig["diagnostic_reasons"].values == preds_reloaded["diagnostic_reasons"].values).all()


def test_single_record_inference_and_validation(synthetic_training_data):
    """Verifies single-record inference and strict rejection of malformed or non-finite inputs."""
    detector = TransactionAnomalyDetector().fit(synthetic_training_data)

    # Valid normal record
    rec_normal = {
        "invoice_no": "TEST01",
        "stock_code": "STK_01",
        "quantity": 5,
        "unit_price": 4.50,
        "invoice_date": "2010-06-01 12:00:00",
    }
    res_normal = detector.predict_record(rec_normal)
    assert isinstance(res_normal, dict)
    assert "anomaly_score" in res_normal
    assert "is_anomaly" in res_normal

    # Valid anomalous record
    rec_anomaly = {
        "invoice_no": "TEST02",
        "stock_code": "STK_01",
        "quantity": 50000,
        "unit_price": 500.0,
        "invoice_date": "2010-06-01 12:00:00",
    }
    res_anomaly = detector.predict_record(rec_anomaly)
    assert res_anomaly["is_anomaly"] is True
    assert "EXTREME_QUANTITY" in res_anomaly["diagnostic_reasons"]

    # Rejection: missing required key
    with pytest.raises(ValueError, match="Missing required transaction field"):
        detector.predict_record({"quantity": 10})

    # Rejection: non-finite or non-numeric quantity
    with pytest.raises(ValueError, match="Invalid non-finite or non-numeric quantity"):
        detector.predict_record({"quantity": np.nan, "unit_price": 5.0})

    with pytest.raises(ValueError, match="Invalid non-finite or non-numeric unit_price"):
        detector.predict_record({"quantity": 10, "unit_price": np.inf})


def test_diagnostic_rule_explainer():
    """Verifies that diagnostic reason codes are generated accurately based on threshold triggers."""
    explainer = DiagnosticRuleExplainer(
        quantity_p99=100.0,
        line_total_p99=200.0,
        unit_price_p99=50.0,
        price_dev_threshold=1.10,
    )

    # Inlier should receive empty reason
    rec_inlier = {"quantity": 5, "unit_price": 10.0, "line_total": 50.0}
    assert explainer.explain_record(rec_inlier, is_anomaly=False) == ""

    # Extreme quantity
    rec_qty = {"quantity": 150, "unit_price": 1.0, "line_total": 150.0}
    assert "EXTREME_QUANTITY" in explainer.explain_record(rec_qty, is_anomaly=True)

    # High unit price
    rec_price = {"quantity": 1, "unit_price": 100.0, "line_total": 100.0}
    assert "HIGH_UNIT_PRICE" in explainer.explain_record(rec_price, is_anomaly=True)

    # Administrative duplicate
    rec_dup = {"quantity": 1, "unit_price": 5.0, "is_duplicate": True}
    assert "EXACT_DUPLICATE_ROW" in explainer.explain_record(rec_dup, is_anomaly=True)


def test_synthetic_diagnostic_injection(synthetic_training_data):
    """Verifies diagnostic synthetic anomaly injection detection."""
    detector = TransactionAnomalyDetector().fit(synthetic_training_data)
    results = detector.evaluate_sensitivity_and_stability(synthetic_training_data, sample_size=100)

    synth = results["synthetic_diagnostic_test"]
    assert synth["injected_count"] == 5
    assert synth["detected_count"] >= 4
    assert synth["diagnostic_detection_rate"] >= 0.80
