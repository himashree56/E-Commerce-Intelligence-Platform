"""Configuration settings for K-Means Customer Segmentation (Phase 4)."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class SegmentationConfig:
    """Hyperparameter and operational configuration for customer segmentation."""

    model_version: str = "v1.0.0"
    random_state: int = 42
    n_init: int = 20
    max_iter: int = 300
    scaler_type: str = "standard"  # 'standard' or 'robust'

    feature_columns: List[str] = field(
        default_factory=lambda: ["recency_days", "frequency", "monetary_net"]
    )

    k_min: int = 2
    k_max: int = 10
    optimal_k: int = 5

    # Standardized business-friendly cluster naming taxonomy
    cluster_labels: Dict[int, str] = field(
        default_factory=lambda: {
            0: "High-Value Loyal Champions",
            1: "Active Recent Shoppers",
            2: "Lapsed High-Potential Customers",
            3: "Lost Low-Value Buyers",
            4: "Net Returners & Refund Adjusters",
        }
    )

    # File and directory paths
    base_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    gold_rfm_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "gold" / "gold_customer_rfm.parquet"
    )
    model_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "customer_segmentation"
    )
    reports_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "reports" / "phase4"
    )

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AnomalyDetectionConfig:
    """Hyperparameter and operational configuration for Isolation Forest anomaly detection."""

    model_version: str = "v1.0.0"
    random_state: int = 42
    n_estimators: int = 100
    max_samples: int = 256  # Subsampling to prevent masking/swamping
    contamination: float = 0.01  # Assumed anomaly proportion (not measured fraud rate)
    n_jobs: int = -1

    # Diagnostic rule thresholds (derived from training data or domain defaults)
    quantity_p99: float = 100.0
    line_total_p99: float = 180.0
    unit_price_p99: float = 18.0
    price_deviation_threshold: float = 1.10  # log(3.0) ~ 1.0986

    # File and directory paths
    base_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    silver_parquet_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "silver" / "online_retail_silver.parquet"
    )
    gold_anomalies_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "gold" / "gold_transaction_anomalies_isolation_forest.parquet"
    )
    model_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "anomaly_detection"
    )
    reports_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "reports" / "phase5"
    )

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class RepeatPurchaseConfig:
    """Hyperparameter and operational configuration for XGBoost repeat-purchase prediction (Phase 6)."""

    model_version: str = "v1.0.0"
    random_state: int = 42
    horizon_days: int = 30

    train_snapshots: List[str] = field(
        default_factory=lambda: ["2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01"]
    )
    val_snapshot: str = "2011-08-01"
    test_snapshot: str = "2011-10-01"

    # Default XGBoost baseline hyperparameters
    xgb_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 4.08,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    # File and directory paths
    base_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    silver_parquet_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "silver" / "online_retail_silver.parquet"
    )
    gold_train_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "gold" / "gold_repeat_purchase_train.parquet"
    )
    gold_val_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "gold" / "gold_repeat_purchase_val.parquet"
    )
    gold_test_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "gold" / "gold_repeat_purchase_test.parquet"
    )
    model_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "repeat_purchase"
    )
    reports_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "reports" / "phase6"
    )

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

