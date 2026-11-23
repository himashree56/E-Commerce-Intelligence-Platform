"""Feature Engineering Module.

Responsible for:
- RFM feature extraction and symmetric log transformation (Phase 4).
- Transaction anomaly detection feature pipeline (Phase 5).
"""
from src.features.rfm_features import RFMFeaturePipeline, SymmetricLogTransformer
from src.features.anomaly_features import (
    FEATURE_NAMES,
    TransactionFeaturePipeline,
    compute_leakage_safe_prior_orders,
)
from src.features.repeat_purchase_features import (
    REPEAT_PURCHASE_FEATURES,
    RepeatPurchaseFeaturePipeline,
    extract_customer_snapshot,
)

__all__ = [
    "RFMFeaturePipeline",
    "SymmetricLogTransformer",
    "TransactionFeaturePipeline",
    "compute_leakage_safe_prior_orders",
    "FEATURE_NAMES",
    "REPEAT_PURCHASE_FEATURES",
    "RepeatPurchaseFeaturePipeline",
    "extract_customer_snapshot",
]
