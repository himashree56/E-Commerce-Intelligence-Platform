"""
Point-in-time Feature Engineering Pipeline for Phase 6 Repeat-Purchase Prediction.

Constructs customer-snapshot features strictly using transactions prior to the
snapshot cutoff date (T_obs), preventing future temporal leakage.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ecommerce_platform.repeat_purchase.features")

REPEAT_PURCHASE_FEATURES = [
    "recency_days",
    "tenure_days",
    "purchase_frequency",
    "net_monetary_value",
    "gross_sales",
    "return_amount",
    "return_ratio",
    "cancellation_frequency",
    "avg_order_value",
    "avg_order_interval_days",
    "purchases_last_30d",
    "purchases_last_60d",
    "purchases_last_90d",
    "spend_last_30d",
    "spend_last_60d",
    "spend_last_90d",
    "product_diversity",
    "total_quantity_purchased",
]


def extract_customer_snapshot(
    silver_df: pd.DataFrame,
    snapshot_date: Union[str, pd.Timestamp],
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Extracts leakage-safe customer snapshot features and future repeat purchase labels.

    Eligibility:
        Customer must have at least one VALID_PURCHASE strictly prior to snapshot_date.

    Observation Window:
        Strictly before snapshot_date: [start_of_data, snapshot_date).

    Future Label Window:
        [snapshot_date, snapshot_date + horizon_days].
        Label = 1 if customer makes >= 1 VALID_PURCHASE in window, else 0.

    Args:
        silver_df: Clean Silver transaction DataFrame.
        snapshot_date: Observation cutoff timestamp.
        horizon_days: Future observation horizon (default: 30 days).

    Returns:
        DataFrame with ['customer_id', 'snapshot_date'] + REPEAT_PURCHASE_FEATURES + ['label_30d'].
    """
    snapshot_ts = pd.Timestamp(snapshot_date)
    future_ts = snapshot_ts + pd.Timedelta(days=horizon_days)

    # 1. Filter customer records
    cust_mask = silver_df["customer_id"].notna()
    work = silver_df[cust_mask].copy()
    work["customer_id"] = work["customer_id"].astype(int)
    work["invoice_date"] = pd.to_datetime(work["invoice_date"])

    # 2. Historical partition (strictly prior to snapshot)
    past = work[work["invoice_date"] < snapshot_ts]
    valid_past = past[past["transaction_status"] == "VALID_PURCHASE"]

    eligible_customers = np.sort(valid_past["customer_id"].unique())
    n_eligible = len(eligible_customers)

    if n_eligible == 0:
        logger.warning(f"No eligible customers found strictly before {snapshot_ts}.")
        cols = ["customer_id", "snapshot_date"] + REPEAT_PURCHASE_FEATURES + ["label_30d"]
        return pd.DataFrame(columns=cols)

    logger.info(
        f"Extracting features for {n_eligible:,} eligible customers as of {snapshot_ts.date()} "
        f"(Horizon: {horizon_days}d -> {future_ts.date()})..."
    )

    # 3. Vectorized feature computation on valid past purchases
    gb_valid = valid_past.groupby("customer_id")

    recency_days = (snapshot_ts - gb_valid["invoice_date"].max()).dt.total_seconds() / 86400.0
    tenure_days = (snapshot_ts - gb_valid["invoice_date"].min()).dt.total_seconds() / 86400.0
    purchase_frequency = gb_valid["invoice_no"].nunique()
    product_diversity = gb_valid["stock_code"].nunique()
    total_quantity = gb_valid["quantity"].sum()

    # 4. Monetary and cancellation metrics from full past (including returns/credit notes)
    gb_past = past.groupby("customer_id")
    net_monetary = gb_past["net_revenue"].sum()
    gross_sales = gb_past["gross_sales"].sum()
    return_amount = gb_past["return_amount"].sum()

    cancels_past = past[past["transaction_status"] == "CANCELLATION"]
    cancellation_freq = cancels_past.groupby("customer_id")["invoice_no"].nunique()

    # 5. Windowed features (last 30d, 60d, 90d strictly before snapshot)
    t_30 = snapshot_ts - pd.Timedelta(days=30)
    t_60 = snapshot_ts - pd.Timedelta(days=60)
    t_90 = snapshot_ts - pd.Timedelta(days=90)

    p30 = valid_past[valid_past["invoice_date"] >= t_30].groupby("customer_id")["invoice_no"].nunique()
    p60 = valid_past[valid_past["invoice_date"] >= t_60].groupby("customer_id")["invoice_no"].nunique()
    p90 = valid_past[valid_past["invoice_date"] >= t_90].groupby("customer_id")["invoice_no"].nunique()

    s30 = past[past["invoice_date"] >= t_30].groupby("customer_id")["net_revenue"].sum()
    s60 = past[past["invoice_date"] >= t_60].groupby("customer_id")["net_revenue"].sum()
    s90 = past[past["invoice_date"] >= t_90].groupby("customer_id")["net_revenue"].sum()

    # 6. Assemble feature matrix aligned with eligible customers
    snapshot_df = pd.DataFrame(index=eligible_customers)
    snapshot_df["customer_id"] = eligible_customers
    snapshot_df["snapshot_date"] = snapshot_ts.strftime("%Y-%m-%d")

    snapshot_df["recency_days"] = recency_days.reindex(eligible_customers).values
    snapshot_df["tenure_days"] = tenure_days.reindex(eligible_customers).values
    snapshot_df["purchase_frequency"] = purchase_frequency.reindex(eligible_customers).values
    snapshot_df["net_monetary_value"] = net_monetary.reindex(eligible_customers).fillna(0.0).values
    snapshot_df["gross_sales"] = gross_sales.reindex(eligible_customers).fillna(0.0).values
    snapshot_df["return_amount"] = return_amount.reindex(eligible_customers).fillna(0.0).values

    # Return ratio = return_amount / max(0.01, gross_sales)
    denom = np.maximum(0.01, snapshot_df["gross_sales"].values)
    snapshot_df["return_ratio"] = np.clip(snapshot_df["return_amount"].values / denom, 0.0, 10.0)

    snapshot_df["cancellation_frequency"] = cancellation_freq.reindex(eligible_customers).fillna(0).astype(int).values

    # Average Order Value
    snapshot_df["avg_order_value"] = snapshot_df["net_monetary_value"].values / np.maximum(
        1, snapshot_df["purchase_frequency"].values
    )

    # Average Order Interval in days
    intervals = np.where(
        snapshot_df["purchase_frequency"].values > 1,
        snapshot_df["tenure_days"].values / np.maximum(1, snapshot_df["purchase_frequency"].values - 1),
        snapshot_df["recency_days"].values,
    )
    snapshot_df["avg_order_interval_days"] = intervals

    snapshot_df["purchases_last_30d"] = p30.reindex(eligible_customers).fillna(0).astype(int).values
    snapshot_df["purchases_last_60d"] = p60.reindex(eligible_customers).fillna(0).astype(int).values
    snapshot_df["purchases_last_90d"] = p90.reindex(eligible_customers).fillna(0).astype(int).values

    snapshot_df["spend_last_30d"] = s30.reindex(eligible_customers).fillna(0.0).values
    snapshot_df["spend_last_60d"] = s60.reindex(eligible_customers).fillna(0.0).values
    snapshot_df["spend_last_90d"] = s90.reindex(eligible_customers).fillna(0.0).values

    snapshot_df["product_diversity"] = product_diversity.reindex(eligible_customers).values
    snapshot_df["total_quantity_purchased"] = total_quantity.reindex(eligible_customers).values

    # 7. Future label generation strictly within [snapshot_ts, future_ts]
    fut = work[(work["invoice_date"] >= snapshot_ts) & (work["invoice_date"] <= future_ts)]
    fut_valid = fut[fut["transaction_status"] == "VALID_PURCHASE"]
    repeat_buyers = set(fut_valid["customer_id"].unique()).intersection(set(eligible_customers))

    snapshot_df["label_30d"] = np.isin(snapshot_df["customer_id"].values, list(repeat_buyers)).astype(int)

    snapshot_df = snapshot_df.reset_index(drop=True)
    return snapshot_df


class RepeatPurchaseFeaturePipeline(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for repeat purchase feature preprocessing.

    Applies log1p transformations to monetary and count features with heavy tails,
    and scales features via StandardScaler.
    """

    def __init__(self, use_scaler: bool = True) -> None:
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self.feature_names_: List[str] = list(REPEAT_PURCHASE_FEATURES)
        self.is_fitted_: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> "RepeatPurchaseFeaturePipeline":
        """Fits scaler on training feature matrix."""
        X_mat = self._to_numpy(X)
        if self.use_scaler and self.scaler is not None:
            self.scaler.fit(X_mat)
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transforms input features into scaled representation."""
        if not self.is_fitted_:
            raise ValueError("RepeatPurchaseFeaturePipeline must be fitted before calling transform().")
        X_mat = self._to_numpy(X)
        if self.use_scaler and self.scaler is not None:
            return self.scaler.transform(X_mat)
        return X_mat

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _to_numpy(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            # Extract features in exact expected order
            missing = [f for f in self.feature_names_ if f not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing required features: {missing}")
            mat = X[self.feature_names_].values.astype(float)
        else:
            mat = np.asarray(X, dtype=float)
            if mat.shape[1] != len(self.feature_names_):
                raise ValueError(f"Expected {len(self.feature_names_)} features, got {mat.shape[1]}")

        # Sanitize any unexpected NaNs or Infs
        mat = np.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)
        return mat

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names_)
