"""
Feature engineering pipeline for transaction anomaly detection with Isolation Forest.

Provides leakage-safe, robust feature extraction and scaling for invoice-line
e-commerce transactions.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ecommerce_platform.features.anomaly")


DAY_NAME_TO_INT: Dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}


FEATURE_NAMES: List[str] = [
    "log_abs_quantity",
    "log_unit_price",
    "log_abs_line_total",
    "product_price_deviation",
    "log_prior_customer_orders",
    "is_missing_customer",
    "invoice_hour_sin",
    "invoice_hour_cos",
    "invoice_day_sin",
    "invoice_day_cos",
]


def compute_leakage_safe_prior_orders(df: pd.DataFrame) -> pd.Series:
    """Computes the number of distinct invoices completed by the customer
    strictly PRIOR to the current transaction, preventing temporal data leakage.
    All lines belonging to the same invoice share the exact same prior order count.

    Args:
        df: DataFrame containing at least ['customer_id', 'invoice_no'].

    Returns:
        Series of integer prior order counts aligned with the original DataFrame index.
    """
    if "customer_id" not in df.columns or "invoice_no" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)

    orig_index = df.index

    # Check if invoice_date is available for chronological sequencing
    sort_cols = ["customer_id"]
    if "invoice_date" in df.columns:
        sort_cols.append("invoice_date")
    if "invoice_no" in df.columns:
        sort_cols.append("invoice_no")

    work = df[["customer_id", "invoice_no"] + ([c for c in ["invoice_date"] if c in df.columns])].copy()
    work["_orig_idx"] = np.arange(len(df))
    work = work.sort_values(sort_cols)

    # Exclude missing customer identifiers from prior order sequencing
    valid_work = work.dropna(subset=["customer_id"])
    unique_invoices = valid_work.drop_duplicates(subset=["customer_id", "invoice_no"], keep="first").copy()
    unique_invoices["_inv_prior"] = unique_invoices.groupby("customer_id").cumcount()

    # Left merge ensures every line in the invoice gets the exact prior order count
    work = work.merge(
        unique_invoices[["customer_id", "invoice_no", "_inv_prior"]],
        on=["customer_id", "invoice_no"],
        how="left",
    )
    work["_prior_orders"] = work["_inv_prior"].fillna(0).astype(int)

    # Restore original index order
    prior_orders = work.sort_values("_orig_idx")["_prior_orders"].astype(int)
    prior_orders.index = orig_index
    return prior_orders


class TransactionFeaturePipeline(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible feature transformer for transaction anomaly detection.

    Extracts leakage-safe behavioral, economic, and temporal features from
    invoice lines, applies non-linear log transformations to handle heavy tails,
    and scales features via StandardScaler.
    """

    def __init__(self, use_scaler: bool = True) -> None:
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self.catalog_median_prices_: Dict[str, float] = {}
        self.global_median_price_: float = 2.10
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame, y: Optional[Any] = None) -> "TransactionFeaturePipeline":
        """Fits catalog reference statistics and feature scaler on eligible training transactions.

        Args:
            df: Training DataFrame containing commercial transactions.
            y: Ignored (unsupervised).

        Returns:
            self
        """
        logger.info(f"Fitting TransactionFeaturePipeline on {len(df):,} records...")

        # 1. Learn catalog median unit price per stock code from positive prices
        pos_df = df[df["unit_price"] > 0] if "unit_price" in df.columns else df
        if len(pos_df) > 0 and "stock_code" in pos_df.columns:
            medians = pos_df.groupby("stock_code")["unit_price"].median().to_dict()
            self.catalog_median_prices_ = {str(k): float(v) for k, v in medians.items()}
            self.global_median_price_ = float(pos_df["unit_price"].median())
        else:
            self.catalog_median_prices_ = {}
            self.global_median_price_ = 2.10

        # 2. Extract unscaled features
        X_raw = self._extract_features(df, is_training=True)

        # 3. Fit scaler
        if self.use_scaler:
            self.scaler.fit(X_raw)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms incoming transactions into a standardized feature matrix.

        Args:
            df: DataFrame containing transaction records.

        Returns:
            2D numpy array of shape (N, len(FEATURE_NAMES)).
        """
        if not self.is_fitted_:
            raise ValueError("TransactionFeaturePipeline must be fitted before calling transform().")

        X_raw = self._extract_features(df, is_training=False)
        if self.use_scaler and self.scaler is not None:
            return self.scaler.transform(X_raw)
        return X_raw

    def _extract_features(self, df: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """Internal feature extraction logic handling missing values and edge cases."""
        n_rows = len(df)
        if n_rows == 0:
            return np.empty((0, len(self.feature_names_)))

        # 1. Absolute Quantity (heavy-tailed) -> log1p(abs(quantity))
        qty = df["quantity"].values if "quantity" in df.columns else np.zeros(n_rows)
        qty = np.nan_to_num(qty, nan=0.0)
        abs_qty = np.abs(qty)
        log_abs_qty = np.log1p(abs_qty)

        # 2. Unit Price -> log1p(max(0, unit_price))
        price = df["unit_price"].values if "unit_price" in df.columns else np.zeros(n_rows)
        price = np.nan_to_num(price, nan=0.0)
        clipped_price = np.maximum(0.0, price)
        log_unit_price = np.log1p(clipped_price)

        # 3. Absolute Line Total -> log1p(abs(line_total))
        if "line_total" in df.columns:
            line_tot = df["line_total"].values
            line_tot = np.nan_to_num(line_tot, nan=0.0)
        else:
            line_tot = qty * price
        abs_line_tot = np.abs(line_tot)
        log_abs_line_tot = np.log1p(abs_line_tot)

        # 4. Product Price Deviation: log(unit_price / catalog_median)
        # Ratio of transaction unit price to expected product median price
        if "stock_code" in df.columns:
            stock_codes = df["stock_code"].astype(str).values
            medians = np.array([
                self.catalog_median_prices_.get(sc, self.global_median_price_)
                for sc in stock_codes
            ])
        else:
            medians = np.full(n_rows, self.global_median_price_)

        medians = np.where(medians <= 0, self.global_median_price_, medians)
        # Avoid division by zero or negative:
        safe_price = np.maximum(0.001, clipped_price)
        safe_median = np.maximum(0.001, medians)
        price_ratio = safe_price / safe_median
        price_ratio_clipped = np.clip(price_ratio, 0.01, 100.0)
        product_price_dev = np.log(price_ratio_clipped)

        # 5. Prior Customer Orders (leakage-safe) -> log1p(prior_orders)
        if "prior_orders" in df.columns:
            prior_orders = df["prior_orders"].fillna(0).values
        elif "prior_customer_orders" in df.columns:
            prior_orders = df["prior_customer_orders"].fillna(0).values
        elif is_training or (n_rows > 1 and "customer_id" in df.columns and "invoice_no" in df.columns):
            prior_orders = compute_leakage_safe_prior_orders(df).values
        else:
            prior_orders = np.zeros(n_rows)

        prior_orders = np.nan_to_num(prior_orders, nan=0.0)
        log_prior_orders = np.log1p(np.maximum(0.0, prior_orders))

        # 6. Missing Customer Indicator
        if "is_missing_customer" in df.columns:
            is_missing = df["is_missing_customer"].astype(float).values
        elif "customer_id" in df.columns:
            is_missing = df["customer_id"].isna().astype(float).values
        else:
            is_missing = np.ones(n_rows, dtype=float)

        # 7 & 8. Invoice Hour (cyclical: sin & cos)
        if "invoice_hour" in df.columns:
            hours = df["invoice_hour"].fillna(12).values.astype(float)
        elif "invoice_date" in df.columns:
            hours = pd.to_datetime(df["invoice_date"]).dt.hour.fillna(12).values.astype(float)
        else:
            hours = np.full(n_rows, 12.0)

        hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hours / 24.0)

        # 9 & 10. Invoice Day of Week (cyclical: sin & cos)
        if "invoice_day_of_week" in df.columns:
            raw_days = df["invoice_day_of_week"].fillna(2)
            if raw_days.dtype == object or pd.api.types.is_string_dtype(raw_days):
                days = raw_days.astype(str).str.lower().map(DAY_NAME_TO_INT).fillna(2).values.astype(float)
            else:
                days = pd.to_numeric(raw_days, errors="coerce").fillna(2).values.astype(float)
        elif "invoice_date" in df.columns:
            days = pd.to_datetime(df["invoice_date"]).dt.dayofweek.fillna(2).values.astype(float)
        else:
            days = np.full(n_rows, 2.0)

        day_sin = np.sin(2.0 * np.pi * days / 7.0)
        day_cos = np.cos(2.0 * np.pi * days / 7.0)

        # Assemble 2D feature matrix
        feature_matrix = np.column_stack([
            log_abs_qty,
            log_unit_price,
            log_abs_line_tot,
            product_price_dev,
            log_prior_orders,
            is_missing,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
        ])

        # Validate finite values
        if not np.all(np.isfinite(feature_matrix)):
            logger.warning("Feature matrix contained non-finite values; replacing with 0.0.")
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

        return feature_matrix

    def get_feature_names(self) -> List[str]:
        """Returns the list of engineered feature names."""
        return list(self.feature_names_)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes pipeline parameters and learned catalog medians."""
        return {
            "use_scaler": self.use_scaler,
            "global_median_price": self.global_median_price_,
            "catalog_median_prices_count": len(self.catalog_median_prices_),
            "catalog_median_prices": self.catalog_median_prices_,
            "feature_names": self.feature_names_,
            "is_fitted": self.is_fitted_,
            "scaler_mean": self.scaler.mean_.tolist() if (self.scaler and hasattr(self.scaler, "mean_") and self.scaler.mean_ is not None) else None,
            "scaler_scale": self.scaler.scale_.tolist() if (self.scaler and hasattr(self.scaler, "scale_") and self.scaler.scale_ is not None) else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionFeaturePipeline":
        """Reconstructs pipeline from serialized dictionary."""
        pipeline = cls(use_scaler=data.get("use_scaler", True))
        pipeline.global_median_price_ = data.get("global_median_price", 2.10)
        pipeline.catalog_median_prices_ = data.get("catalog_median_prices", {})
        pipeline.feature_names_ = data.get("feature_names", list(FEATURE_NAMES))
        pipeline.is_fitted_ = data.get("is_fitted", False)

        if pipeline.use_scaler and data.get("scaler_mean") is not None:
            pipeline.scaler = StandardScaler()
            pipeline.scaler.mean_ = np.array(data["scaler_mean"])
            pipeline.scaler.scale_ = np.array(data["scaler_scale"])
            pipeline.scaler.var_ = pipeline.scaler.scale_ ** 2
            pipeline.scaler.n_samples_seen_ = 1000000

        return pipeline
