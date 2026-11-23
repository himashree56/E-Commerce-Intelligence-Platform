"""Machine Learning Models Module.

Responsible for:
- Customer Segmentation (Phase 4): K-Means clustering on RFM.
- Transaction Anomaly Detection (Phase 5): Isolation Forest on invoice lines.
- Model registry, experiment tracking, and artifact persistence.
"""
from src.models.config import AnomalyDetectionConfig, RepeatPurchaseConfig, SegmentationConfig
from src.models.customer_segmentation import CustomerSegmentationModel
from src.models.anomaly_detector import (
    DiagnosticRuleExplainer,
    TransactionAnomalyDetector,
)
from src.models.repeat_purchase import RepeatPurchasePredictor

__all__ = [
    "SegmentationConfig",
    "CustomerSegmentationModel",
    "AnomalyDetectionConfig",
    "DiagnosticRuleExplainer",
    "TransactionAnomalyDetector",
    "RepeatPurchaseConfig",
    "RepeatPurchasePredictor",
]
