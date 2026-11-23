"""
XGBoost Repeat-Purchase Prediction Engine (Phase 6).

Implements:
- Dummy baseline (prior/stratified)
- Logistic Regression baseline (standardized)
- XGBoost classifier with validation tuning and scale_pos_weight calibration
- Optimal decision threshold selection strictly on validation data
- Comprehensive metric reporting (ROC-AUC, PR-AUC, P@top5%, P@top10%, Brier, subgroup metrics)
- SHAP and global feature importance attribution
- Serialization, artifact persistence, and validated single/batch inference
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.repeat_purchase_features import (
    REPEAT_PURCHASE_FEATURES,
    RepeatPurchaseFeaturePipeline,
)
from src.models.config import RepeatPurchaseConfig

logger = logging.getLogger("ecommerce_platform.repeat_purchase.model")


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # Precision at top K%
    n_total = len(y_true)
    order = np.argsort(-y_prob)
    sorted_y_true = y_true[order]

    top_5_n = max(1, int(0.05 * n_total))
    top_10_n = max(1, int(0.10 * n_total))
    p_at_5 = float(np.mean(sorted_y_true[:top_5_n]))
    p_at_10 = float(np.mean(sorted_y_true[:top_10_n]))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "precision_at_top_5_pct": round(p_at_5, 4),
        "precision_at_top_10_pct": round(p_at_10, 4),
        "confusion_matrix": {
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
        "threshold": round(threshold, 4),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Finds decision threshold maximizing F1 score strictly on validation data."""
    best_thresh = 0.5
    best_score = -1.0

    thresholds = np.linspace(0.10, 0.90, 81)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return round(best_thresh, 4), round(best_score, 4)


class RepeatPurchasePredictor:
    """Production-grade repeat-purchase prediction pipeline using XGBoost and baselines."""

    def __init__(self, config: Optional[RepeatPurchaseConfig] = None) -> None:
        self.config = config or RepeatPurchaseConfig()
        self.preprocessor = RepeatPurchaseFeaturePipeline(use_scaler=True)
        self.dummy_model: Optional[DummyClassifier] = None
        self.lr_model: Optional[LogisticRegression] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.calibrated_model: Optional[Any] = None
        self.optimal_threshold: float = 0.59
        self.optimal_calibrated_threshold: float = 0.2350
        self.is_fitted: bool = False

        self.feature_names: List[str] = list(REPEAT_PURCHASE_FEATURES)
        self.evaluation_metrics: Dict[str, Any] = {}
        self.training_metadata: Dict[str, Any] = {}
        self.threshold_policy: Dict[str, Any] = {}

    def fit_and_tune(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> "RepeatPurchasePredictor":
        """Fits feature preprocessing and models, tuning hyperparameters strictly on validation data.

        Args:
            train_df: Customer snapshot training DataFrame.
            val_df: Customer snapshot validation DataFrame.

        Returns:
            self
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"Fitting RepeatPurchasePredictor on {len(train_df):,} training rows...")

        X_train_raw = train_df[self.feature_names]
        y_train = train_df["label_30d"].values.astype(int)

        X_val_raw = val_df[self.feature_names]
        y_val = val_df["label_30d"].values.astype(int)

        # 1. Fit Preprocessor (Scaler)
        self.preprocessor.fit(X_train_raw)
        X_train_scaled = self.preprocessor.transform(X_train_raw)
        X_val_scaled = self.preprocessor.transform(X_val_raw)

        # 2. Baseline 1: Dummy Classifier (Prior prevalence)
        logger.info("Training Baseline 1: DummyClassifier...")
        self.dummy_model = DummyClassifier(strategy="prior")
        self.dummy_model.fit(X_train_scaled, y_train)

        # 3. Baseline 2: Logistic Regression (L2 regularization)
        logger.info("Training Baseline 2: Logistic Regression...")
        self.lr_model = LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.config.random_state,
        )
        self.lr_model.fit(X_train_scaled, y_train)

        # 4. XGBoost Classifier Tuning on Validation Data
        logger.info("Tuning and training XGBoost Classifier...")
        neg_count = int(np.sum(y_train == 0))
        pos_count = int(np.sum(y_train == 1))
        scale_pos_weight = float(neg_count / max(1, pos_count))

        # Hyperparameter search grid on validation PR-AUC
        candidate_params = [
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
            {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
            {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7},
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.9, "colsample_bytree": 0.9},
        ]

        best_pr_auc = -1.0
        best_params = candidate_params[1]

        for params in candidate_params:
            candidate_xgb = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.config.random_state,
                eval_metric="aucpr",
                n_jobs=-1,
            )
            candidate_xgb.fit(X_train_raw, y_train)
            val_probs = candidate_xgb.predict_proba(X_val_raw)[:, 1]
            val_pr_auc = average_precision_score(y_val, val_probs)
            logger.info(f"Params: {params} -> Val PR-AUC: {val_pr_auc:.4f}")
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                best_params = params

        logger.info(f"Selected Best XGBoost Params: {best_params} (Val PR-AUC: {best_pr_auc:.4f})")

        # Fit best model
        self.xgb_model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.config.random_state,
            eval_metric="aucpr",
            n_jobs=-1,
        )
        self.xgb_model.fit(X_train_raw, y_train)

        # 5. Threshold Selection strictly on Validation Data (Raw)
        val_probs_xgb = self.xgb_model.predict_proba(X_val_raw)[:, 1]
        self.optimal_threshold, val_f1 = find_optimal_threshold(y_val, val_probs_xgb, metric="f1")
        logger.info(f"Selected Optimal Validation Decision Threshold (Raw): {self.optimal_threshold} (Val F1: {val_f1:.4f})")

        # 6. Fit Sigmoid Calibrator strictly on Validation Data
        logger.info("Fitting Sigmoid (Platt) Calibrator strictly on validation data...")
        self.calibrated_model = CalibratedClassifierCV(estimator=self.xgb_model, method="sigmoid", cv="prefit")
        self.calibrated_model.fit(X_val_raw, y_val)
        val_probs_cal = self.calibrated_model.predict_proba(X_val_raw)[:, 1]
        self.optimal_calibrated_threshold, val_cal_f1 = find_optimal_threshold(y_val, val_probs_cal, metric="f1")
        logger.info(f"Selected Optimal Calibrated Threshold: {self.optimal_calibrated_threshold} (Val F1: {val_cal_f1:.4f})")

        # 7. Compute Validation Metrics across models
        val_probs_dummy = self.dummy_model.predict_proba(X_val_scaled)[:, 1]
        val_probs_lr = self.lr_model.predict_proba(X_val_scaled)[:, 1]

        val_metrics = {
            "dummy_baseline": compute_metrics(y_val, val_probs_dummy, threshold=0.5),
            "logistic_regression": compute_metrics(y_val, val_probs_lr, threshold=0.5),
            "xgboost": compute_metrics(y_val, val_probs_xgb, threshold=self.optimal_threshold),
            "xgboost_calibrated_sigmoid": compute_metrics(y_val, val_probs_cal, threshold=self.optimal_calibrated_threshold),
        }

        duration_sec = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.is_fitted = True

        self.training_metadata = {
            "model_version": self.config.model_version,
            "trained_at": start_time.isoformat(),
            "duration_seconds": round(duration_sec, 2),
            "train_snapshots": self.config.train_snapshots,
            "val_snapshot": self.config.val_snapshot,
            "train_record_count": len(train_df),
            "val_record_count": len(val_df),
            "train_positive_count": pos_count,
            "train_negative_count": neg_count,
            "train_prevalence_pct": round(pos_count / len(train_df) * 100, 2),
            "scale_pos_weight": round(scale_pos_weight, 4),
            "best_hyperparameters": best_params,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "raw_decision_threshold": self.optimal_threshold,
            "calibrated_decision_threshold": self.optimal_calibrated_threshold,
        }

        self.threshold_policy = {
            "selected_threshold": self.optimal_threshold,
            "raw_threshold": self.optimal_threshold,
            "calibrated_threshold": self.optimal_calibrated_threshold,
            "selection_split": "validation",
            "validation_snapshot": self.config.val_snapshot,
            "optimization_objective": "f1_score",
            "validation_f1_at_threshold": val_f1,
            "validation_calibrated_f1_at_threshold": val_cal_f1,
            "decision_rule_raw": f"repeat_purchase_predicted = probability >= {self.optimal_threshold}",
            "decision_rule_calibrated": f"repeat_purchase_predicted = probability >= {self.optimal_calibrated_threshold}",
        }

        self.evaluation_metrics = {"validation": val_metrics}
        return self

    def evaluate_test_set(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates final model and baselines on held-out test snapshot once.

        Args:
            test_df: Customer snapshot test DataFrame.

        Returns:
            Dictionary of test evaluation metrics across all models and activity groups.
        """
        if not self.is_fitted or self.xgb_model is None:
            raise ValueError("Model must be fitted before evaluating test set.")

        logger.info(f"Evaluating models on Test Snapshot ({len(test_df):,} rows)...")
        X_test_raw = test_df[self.feature_names]
        X_test_scaled = self.preprocessor.transform(X_test_raw)
        y_test = test_df["label_30d"].values.astype(int)

        # Probabilities
        probs_dummy = self.dummy_model.predict_proba(X_test_scaled)[:, 1]
        probs_lr = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        probs_xgb = self.xgb_model.predict_proba(X_test_raw)[:, 1]
        probs_xgb_cal = (
            self.calibrated_model.predict_proba(X_test_raw)[:, 1] if self.calibrated_model is not None else None
        )

        # Overall Metrics
        test_metrics = {
            "dummy_baseline": compute_metrics(y_test, probs_dummy, threshold=0.5),
            "logistic_regression": compute_metrics(y_test, probs_lr, threshold=0.5),
            "xgboost": compute_metrics(y_test, probs_xgb, threshold=self.optimal_threshold),
        }
        if probs_xgb_cal is not None:
            test_metrics["xgboost_calibrated_sigmoid"] = compute_metrics(
                y_test, probs_xgb_cal, threshold=self.optimal_calibrated_threshold
            )

        # Subgroup Metrics by Customer Activity (Purchase Frequency)
        freq = test_df["purchase_frequency"].values
        subgroups = {
            "low_activity_1_2_orders": freq <= 2,
            "medium_activity_3_5_orders": (freq >= 3) & (freq <= 5),
            "high_activity_6_plus_orders": freq >= 6,
        }

        subgroup_metrics = {}
        for grp_name, mask in subgroups.items():
            if np.sum(mask) > 0 and len(np.unique(y_test[mask])) > 1:
                subgroup_metrics[grp_name] = {
                    "count": int(np.sum(mask)),
                    "positives": int(np.sum(y_test[mask])),
                    "prevalence_pct": round(float(np.mean(y_test[mask]) * 100), 2),
                    "roc_auc": round(float(roc_auc_score(y_test[mask], probs_xgb[mask])), 4),
                    "pr_auc": round(float(average_precision_score(y_test[mask], probs_xgb[mask])), 4),
                    "precision": round(float(precision_score(y_test[mask], (probs_xgb[mask] >= self.optimal_threshold).astype(int), zero_division=0)), 4),
                    "recall": round(float(recall_score(y_test[mask], (probs_xgb[mask] >= self.optimal_threshold).astype(int), zero_division=0)), 4),
                }

        test_metrics["activity_subgroups_xgboost"] = subgroup_metrics
        self.evaluation_metrics["test"] = test_metrics
        return test_metrics

    def predict_proba(self, df: pd.DataFrame, calibrated: bool = False) -> np.ndarray:
        """Computes repeat-purchase probabilities for incoming customer feature snapshots.
        
        Args:
            df: DataFrame containing the 18 repeat purchase features.
            calibrated: If True, returns posterior probabilities from the Sigmoid calibrator.
        """
        if not self.is_fitted or self.xgb_model is None:
            raise ValueError("Predictor is not fitted.")
        X = df[self.feature_names]
        if calibrated:
            if self.calibrated_model is not None:
                return self.calibrated_model.predict_proba(X)[:, 1]
            logger.warning("Calibrated model not available, using raw XGBoost probabilities.")
        return self.xgb_model.predict_proba(X)[:, 1]

    def predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        calibrated: bool = False,
    ) -> np.ndarray:
        """Computes binary repeat-purchase predictions using the selected threshold."""
        probs = self.predict_proba(df, calibrated=calibrated)
        if threshold is not None:
            thresh = threshold
        elif calibrated:
            thresh = self.optimal_calibrated_threshold
        else:
            thresh = self.optimal_threshold
        return (probs >= thresh).astype(int)

    def predict_single(
        self,
        features_dict: Dict[str, Any],
        calibrated: bool = False,
    ) -> Dict[str, Any]:
        """Validated single-customer inference for real-time CRM API endpoints."""
        df_single = pd.DataFrame([features_dict])
        prob = float(self.predict_proba(df_single, calibrated=calibrated)[0])
        thresh = self.optimal_calibrated_threshold if calibrated else self.optimal_threshold
        pred = bool(prob >= thresh)
        return {
            "customer_id": features_dict.get("customer_id"),
            "repeat_purchase_probability": round(prob, 4),
            "predicted_label": pred,
            "decision_threshold": thresh,
            "is_calibrated": calibrated,
            "model_version": self.config.model_version,
        }

    def compute_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """Returns XGBoost feature importance rankings across weight, gain, and cover."""
        if not self.is_fitted or self.xgb_model is None:
            raise ValueError("Predictor is not fitted.")

        booster = self.xgb_model.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        score_weight = booster.get_score(importance_type="weight")

        # Map to all features
        gain_dict = {f: round(float(score_gain.get(f, 0.0)), 4) for f in self.feature_names}
        weight_dict = {f: round(float(score_weight.get(f, 0.0)), 4) for f in self.feature_names}

        # Sort descending by gain
        sorted_gain = dict(sorted(gain_dict.items(), key=lambda item: -item[1]))
        return {"gain": sorted_gain, "weight": weight_dict}

    def compute_shap_explanations(
        self,
        sample_df: pd.DataFrame,
        max_samples: int = 1000,
    ) -> Tuple[np.ndarray, shap.Explainer]:
        """Computes TreeSHAP values for a sample of customer snapshots."""
        if not self.is_fitted or self.xgb_model is None:
            raise ValueError("Predictor is not fitted.")

        n_sub = min(max_samples, len(sample_df))
        X_sub = sample_df[self.feature_names].iloc[:n_sub]

        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X_sub)
        return shap_values, explainer

    def save(self, model_dir: Optional[Union[str, Path]] = None) -> Path:
        """Persists trained XGBoost model, preprocessor, and metadata JSON files."""
        if not self.is_fitted or self.xgb_model is None:
            raise ValueError("Cannot save an unfitted model.")

        out_dir = Path(model_dir) if model_dir else self.config.model_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving RepeatPurchasePredictor to {out_dir}...")

        # 1. XGBoost model
        joblib.dump(self.xgb_model, out_dir / "xgb_model.joblib")

        # 1b. Calibrated Model (Sigmoid / Platt scaling)
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, out_dir / "calibrated_xgb_model.joblib")

        # 2. Fitted Preprocessor
        joblib.dump(self.preprocessor, out_dir / "preprocessor.joblib")

        # 3. Baselines (optional for reference)
        if self.lr_model:
            joblib.dump(self.lr_model, out_dir / "logistic_regression_baseline.joblib")

        # 4. Feature Schema
        feature_schema = {
            "feature_names": self.feature_names,
            "feature_count": len(self.feature_names),
            "target": "label_30d",
            "horizon_days": self.config.horizon_days,
        }
        with open(out_dir / "feature_schema.json", "w") as f:
            json.dump(feature_schema, f, indent=2)

        # 5. Threshold Policy
        with open(out_dir / "threshold_policy.json", "w") as f:
            json.dump(self.threshold_policy, f, indent=2)

        # 6. Training Metadata
        with open(out_dir / "training_metadata.json", "w") as f:
            json.dump(self.training_metadata, f, indent=2)

        # 7. Evaluation Metrics
        with open(out_dir / "evaluation_metrics.json", "w") as f:
            json.dump(self.evaluation_metrics, f, indent=2)

        # 8. Feature Importances
        importances = self.compute_feature_importances()
        with open(out_dir / "feature_importances.json", "w") as f:
            json.dump(importances, f, indent=2)

        logger.info("Successfully persisted all repeat-purchase prediction artifacts.")
        return out_dir

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "RepeatPurchasePredictor":
        """Loads serialized XGBoost model, preprocessor, and metadata from disk."""
        m_dir = Path(model_dir)
        if not (m_dir / "xgb_model.joblib").exists():
            raise FileNotFoundError(f"Model artifact missing in {m_dir}")

        predictor = cls()
        predictor.xgb_model = joblib.load(m_dir / "xgb_model.joblib")
        predictor.preprocessor = joblib.load(m_dir / "preprocessor.joblib")

        if (m_dir / "logistic_regression_baseline.joblib").exists():
            predictor.lr_model = joblib.load(m_dir / "logistic_regression_baseline.joblib")

        if (m_dir / "calibrated_xgb_model.joblib").exists():
            predictor.calibrated_model = joblib.load(m_dir / "calibrated_xgb_model.joblib")

        with open(m_dir / "threshold_policy.json") as f:
            policy = json.load(f)
            predictor.optimal_threshold = policy.get("raw_threshold", policy.get("selected_threshold", 0.59))
            predictor.optimal_calibrated_threshold = policy.get("calibrated_threshold", 0.2350)
            predictor.threshold_policy = policy

        with open(m_dir / "training_metadata.json") as f:
            predictor.training_metadata = json.load(f)

        with open(m_dir / "evaluation_metrics.json") as f:
            predictor.evaluation_metrics = json.load(f)

        predictor.is_fitted = True
        return predictor
