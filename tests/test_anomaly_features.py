"""
Tests for Phase 5 Feature Engineering Pipeline (TransactionFeaturePipeline).
"""
import numpy as np
import pandas as pd
import pytest

from src.features.anomaly_features import (
    FEATURE_NAMES,
    TransactionFeaturePipeline,
    compute_leakage_safe_prior_orders,
)


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Fixture providing diverse commercial and edge-case transaction lines."""
    return pd.DataFrame({
        "invoice_no": ["10001", "10001", "10002", "C10003", "10004"],
        "stock_code": ["85048", "22197", "85048", "85048", "NEW_ITEM"],
        "description": ["HEART", "POPCORN", "HEART", "HEART", "SPECIAL"],
        "quantity": [10, 5, 100, -2, 1],
        "unit_price": [2.50, 1.25, 2.50, 2.50, 250.0],
        "invoice_date": pd.to_datetime([
            "2010-01-01 10:00:00",
            "2010-01-01 10:00:00",
            "2010-01-05 14:00:00",
            "2010-01-10 09:00:00",
            "2010-01-12 22:00:00",
        ]),
        "customer_id": [12345, 12345, 12345, 12345, None],
        "country": ["United Kingdom"] * 5,
        "transaction_status": ["VALID_PURCHASE", "VALID_PURCHASE", "VALID_PURCHASE", "CANCELLATION", "VALID_PURCHASE"],
        "is_duplicate": [False, False, False, False, False],
        "invoice_hour": [10, 10, 14, 9, 22],
        "invoice_day_of_week": ["Friday", "Friday", "Tuesday", "Sunday", "Tuesday"],
    })


def test_leakage_safe_prior_orders_computation():
    """Verifies that prior order count uses only strictly historical invoices without future leakage."""
    df = pd.DataFrame({
        "customer_id": [100, 100, 100, 100, 200, None],
        "invoice_no": ["INV1", "INV1", "INV2", "INV3", "INV4", "INV5"],
        "invoice_date": pd.to_datetime([
            "2010-01-01 10:00:00",
            "2010-01-01 10:00:00",
            "2010-01-05 10:00:00",
            "2010-01-10 10:00:00",
            "2010-01-02 10:00:00",
            "2010-01-03 10:00:00",
        ]),
    })

    prior = compute_leakage_safe_prior_orders(df)
    # First invoice (both lines) should have 0 prior orders
    assert prior.iloc[0] == 0
    assert prior.iloc[1] == 0
    # Second invoice should have 1 prior order (INV1)
    assert prior.iloc[2] == 1
    # Third invoice should have 2 prior orders (INV1, INV2)
    assert prior.iloc[3] == 2
    # Separate customer 200 should have 0 prior orders
    assert prior.iloc[4] == 0
    # Missing customer should have 0 prior orders
    assert prior.iloc[5] == 0


def test_feature_pipeline_fit_transform_shape(sample_transactions):
    """Verifies that TransactionFeaturePipeline extracts expected 10 features with correct dimensions."""
    pipeline = TransactionFeaturePipeline(use_scaler=True)
    X = pipeline.fit_transform(sample_transactions)

    assert X.shape == (5, len(FEATURE_NAMES))
    assert pipeline.get_feature_names() == FEATURE_NAMES
    assert np.all(np.isfinite(X))


def test_feature_pipeline_product_median_learned(sample_transactions):
    """Verifies that catalog medians are learned from training data and applied to unknown products."""
    pipeline = TransactionFeaturePipeline(use_scaler=False)
    pipeline.fit(sample_transactions)

    # 85048 was present at 2.50
    assert "85048" in pipeline.catalog_median_prices_
    assert pipeline.catalog_median_prices_["85048"] == 2.50

    # Transform new data containing an unseen product
    new_data = pd.DataFrame({
        "quantity": [1],
        "unit_price": [10.0],
        "stock_code": ["UNSEEN_CODE_999"],
        "customer_id": [None],
        "invoice_date": ["2010-05-01 12:00:00"],
    })
    X_new = pipeline.transform(new_data)
    assert X_new.shape == (1, 10)
    assert np.all(np.isfinite(X_new))


def test_feature_pipeline_missing_and_zero_values():
    """Verifies robust handling of NaN values, zero price, and zero quantity."""
    df_edge = pd.DataFrame({
        "quantity": [0, -50, np.nan],
        "unit_price": [0.0, -10.0, np.nan],
        "customer_id": [None, np.nan, 99999],
        "stock_code": ["TEST", "TEST", "TEST"],
        "invoice_no": ["1", "2", "3"],
        "invoice_date": ["2010-01-01 00:00:00", "2010-01-01 12:00:00", "2010-01-01 23:59:59"],
    })

    pipeline = TransactionFeaturePipeline(use_scaler=True)
    pipeline.fit(df_edge)
    X = pipeline.transform(df_edge)

    assert X.shape == (3, 10)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_feature_pipeline_extreme_outliers():
    """Verifies that extreme quantities (e.g. 80,000 units) and prices (£50,000) are smoothly scaled."""
    df_extreme = pd.DataFrame({
        "quantity": [80995, -80995],
        "unit_price": [38970.0, 5000.0],
        "stock_code": ["EXTREME1", "EXTREME2"],
        "customer_id": [16446, 12346],
        "invoice_no": ["581483", "C581484"],
        "invoice_date": ["2011-12-09 09:15:00", "2011-12-09 09:27:00"],
    })

    pipeline = TransactionFeaturePipeline(use_scaler=True)
    pipeline.fit(df_extreme)
    X = pipeline.transform(df_extreme)

    assert X.shape == (2, 10)
    assert np.all(np.isfinite(X))
