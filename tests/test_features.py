"""Unit tests for RFM Feature Engineering Pipeline (Phase 4)."""
import numpy as np
import pandas as pd
import pytest

from src.features.rfm_features import (
    NonNegativeLog1pTransformer,
    RFMFeaturePipeline,
    SymmetricLogTransformer,
)


def test_symmetric_log_transform_properties():
    """Verifies odd symmetry, monotonicity, zero mapping, and invertibility of signed log1p."""
    transformer = SymmetricLogTransformer()
    x = np.array([-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0])

    y = transformer.transform(x)

    # 1. Zero maps to zero
    assert y[4] == 0.0

    # 2. Odd symmetry: f(-x) == -f(x)
    assert np.allclose(y[:4], -y[-4:][::-1])

    # 3. Monotonicity: strictly increasing
    assert np.all(np.diff(y) > 0)

    # 4. Invertibility
    x_rec = transformer.inverse_transform(y)
    assert np.allclose(x, x_rec)


def test_non_negative_log1p_clipping():
    """Verifies that non-negative log1p clips negative values cleanly without NaN."""
    transformer = NonNegativeLog1pTransformer()
    x = np.array([-50.0, -0.01, 0.0, 1.0, 10.0, 100.0])
    y = transformer.transform(x)

    assert not np.isnan(y).any()
    assert y[0] == 0.0
    assert y[1] == 0.0
    assert y[2] == 0.0
    assert y[3] == np.log1p(1.0)
    assert y[5] == np.log1p(100.0)


def test_rfm_feature_pipeline_fit_transform():
    """Verifies that RFMFeaturePipeline scales and normalizes candidate features."""
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "recency_days": [10, 50, 100, 200, 500],
        "frequency": [1, 2, 5, 10, 25],
        "monetary_net": [-50.0, 0.0, 150.0, 850.0, 5000.0],
    })

    pipe = RFMFeaturePipeline()
    X_scaled = pipe.fit_transform(df)

    assert X_scaled.shape == (5, 3)
    # Check standardization properties (mean ~ 0, std ~ 1)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-7)
    assert pipe.get_feature_names_out() == [
        "scaled_log_recency_days",
        "scaled_log_frequency",
        "scaled_log_monetary_net",
    ]


def test_rfm_feature_pipeline_excludes_customer_id():
    """Verifies that customer_id is strictly ignored and not treated as a predictive feature."""
    df = pd.DataFrame({
        "customer_id": [999999],
        "recency_days": [15],
        "frequency": [3],
        "monetary_net": [250.0],
        "other_metadata": ["dummy"],
    })

    pipe = RFMFeaturePipeline()
    pipe.fit(df)
    assert "customer_id" not in pipe.feature_names_in_
    assert pipe.n_features_in_ == 3


def test_rfm_feature_pipeline_missing_columns_raises_value_error():
    """Verifies that missing candidate features raise a descriptive ValueError."""
    df_missing = pd.DataFrame({
        "recency_days": [10],
        "frequency": [2],
        # monetary_net is missing
    })

    pipe = RFMFeaturePipeline()
    with pytest.raises(ValueError, match="Input missing required feature columns"):
        pipe.fit(df_missing)


def test_rfm_feature_pipeline_null_values_raises_value_error():
    """Verifies that null or NaN values in feature columns raise a descriptive ValueError."""
    df_null = pd.DataFrame({
        "recency_days": [10, np.nan],
        "frequency": [2, 4],
        "monetary_net": [100.0, 200.0],
    })

    pipe = RFMFeaturePipeline()
    with pytest.raises(ValueError, match="Input contains null or NaN values in feature columns"):
        pipe.fit(df_null)


def test_rfm_feature_pipeline_non_numeric_raises_value_error():
    """Verifies that non-numeric unconvertible strings raise a ValueError."""
    df_invalid = pd.DataFrame({
        "recency_days": ["not_a_number"],
        "frequency": [2],
        "monetary_net": [100.0],
    })

    pipe = RFMFeaturePipeline()
    with pytest.raises(ValueError, match="must be numeric"):
        pipe.fit(df_invalid)
