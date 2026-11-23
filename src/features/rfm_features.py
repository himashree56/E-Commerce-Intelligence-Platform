"""Feature Engineering Pipeline for Customer RFM Segmentation (Phase 4)."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger("ecommerce_platform.features.rfm")


class SymmetricLogTransformer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer applying symmetric signed log1p transformation:

    f(x) = sign(x) * log(1 + |x|)

    Preserves positive and negative directions while compressing heavy exponential tails.
    Continuously maps 0 to 0 with f'(0) = 1.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> "SymmetricLogTransformer":
        """No-op fit to satisfy scikit-learn API."""
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transforms input array or DataFrame via symmetric signed log1p."""
        X_arr = np.asarray(X, dtype=np.float64)
        return np.sign(X_arr) * np.log1p(np.abs(X_arr))

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Inverts symmetric signed log1p transformation:

        f^-1(y) = sign(y) * (exp(|y|) - 1)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        return np.sign(X_arr) * np.expm1(np.abs(X_arr))


class NonNegativeLog1pTransformer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer applying standard log(1 + x) for non-negative features.

    Clips negative inputs to 0 to prevent NaN generation on any anomalous values.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> "NonNegativeLog1pTransformer":
        """No-op fit to satisfy scikit-learn API."""
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transforms non-negative values via log1p, clipping negative values to 0."""
        X_arr = np.asarray(X, dtype=np.float64)
        return np.log1p(np.clip(X_arr, a_min=0.0, a_max=None))

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Inverts log1p transformation."""
        X_arr = np.asarray(X, dtype=np.float64)
        return np.expm1(X_arr)


class RFMFeaturePipeline(BaseEstimator, TransformerMixin):
    """Reproducible feature engineering pipeline for Customer RFM Segmentation.

    Transforms raw RFM metrics into normalized, scaled numerical features:
    - recency_days: NonNegativeLog1p -> StandardScaler
    - frequency: NonNegativeLog1p -> StandardScaler
    - monetary_net: SymmetricLogTransformer -> StandardScaler

    Guarantees:
    - Customer IDs are never used as predictive features.
    - Zero and negative net-spending customers are preserved without truncation or NaN.
    - Strict input validation on feature names, types, and null values.
    - Deterministic, serializable preprocessing state.
    """

    EXPECTED_FEATURES: List[str] = ["recency_days", "frequency", "monetary_net"]

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        scaler_type: str = "standard",
    ) -> None:
        self.feature_columns = feature_columns or self.EXPECTED_FEATURES
        self.scaler_type = scaler_type.lower()
        self.log_recency = NonNegativeLog1pTransformer()
        self.log_frequency = NonNegativeLog1pTransformer()
        self.log_monetary = SymmetricLogTransformer()
        self.scaler = StandardScaler() if self.scaler_type == "standard" else RobustScaler()
        self.is_fitted: bool = False
        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: int = len(self.feature_columns)

    def _validate_input(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """Validates input data schema, null values, and feature presence."""
        if isinstance(X, dict):
            # Convert single dictionary record or dict-of-lists to DataFrame
            if all(isinstance(v, (list, np.ndarray, pd.Series)) for v in X.values()):
                df = pd.DataFrame(X)
            else:
                df = pd.DataFrame([X])
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(
                f"Expected pandas DataFrame or dictionary, got {type(X).__name__}."
            )

        # Check for missing required feature columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input missing required feature columns: {missing_cols}. "
                f"Expected features: {self.feature_columns}."
            )

        # Extract only feature columns (excluding customer_id, dates, etc.)
        feature_df = df[self.feature_columns].copy()

        # Check for null / NaN values
        null_counts = feature_df.isnull().sum()
        if null_counts.any():
            failing_cols = null_counts[null_counts > 0].to_dict()
            raise ValueError(
                f"Input contains null or NaN values in feature columns: {failing_cols}. "
                f"All input feature records must be non-null."
            )

        # Verify numeric coercion
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(feature_df[col]):
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col])
                except Exception as e:
                    raise ValueError(
                        f"Column '{col}' must be numeric, but could not convert values: {e}"
                    )

        return feature_df

    def fit(self, X: Union[pd.DataFrame, Dict[str, Any]], y: Any = None) -> "RFMFeaturePipeline":
        """Fits transformers and scalers on the training RFM dataset."""
        df = self._validate_input(X)
        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        # Step 1: Compute logarithmic representations
        rec_trans = self.log_recency.fit_transform(df["recency_days"].values)
        freq_trans = self.log_frequency.fit_transform(df["frequency"].values)
        mon_trans = self.log_monetary.fit_transform(df["monetary_net"].values)

        trans_matrix = np.column_stack([rec_trans, freq_trans, mon_trans])

        # Step 2: Fit scaler
        self.scaler.fit(trans_matrix)
        self.is_fitted = True
        logger.info(
            f"Fitted RFMFeaturePipeline on {len(df)} samples with scaler={self.scaler_type}."
        )
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """Transforms raw RFM data into normalized, scaled numerical feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("RFMFeaturePipeline must be fitted before calling transform().")

        df = self._validate_input(X)

        rec_trans = self.log_recency.transform(df["recency_days"].values)
        freq_trans = self.log_frequency.transform(df["frequency"].values)
        mon_trans = self.log_monetary.transform(df["monetary_net"].values)

        trans_matrix = np.column_stack([rec_trans, freq_trans, mon_trans])
        scaled_matrix = self.scaler.transform(trans_matrix)
        return scaled_matrix

    def fit_transform(
        self, X: Union[pd.DataFrame, Dict[str, Any]], y: Any = None
    ) -> np.ndarray:
        """Fits transformers and transforms input in a single call."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Returns the list of output feature names."""
        return [f"scaled_log_{col}" for col in self.feature_columns]
