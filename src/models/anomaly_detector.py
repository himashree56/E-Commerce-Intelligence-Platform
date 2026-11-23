"""
Transaction Anomaly Detection using Isolation Forest (Phase 5).

Provides unsupervised detection of invoice-line commercial anomalies,
interpretable rule-based diagnostic explanations, artifact persistence,
and single-record/batch inference.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

from src.features.anomaly_features import (
    FEATURE_NAMES,
    TransactionFeaturePipeline,
    compute_leakage_safe_prior_orders,
)
from src.models.config import AnomalyDetectionConfig

logger = logging.getLogger("ecommerce_platform.models.anomaly")


class DiagnosticRuleExplainer:
    """Generates human-readable, domain-informed diagnostic reason codes

    for transactions flagged as anomalous by Isolation Forest.
    """

    def __init__(
        self,
        quantity_p99: float = 100.0,
        line_total_p99: float = 180.0,
        unit_price_p99: float = 18.0,
        line_total_p95: float = 67.0,
        price_dev_threshold: float = 1.10,
    ) -> None:
        self.quantity_p99 = quantity_p99
        self.line_total_p99 = line_total_p99
        self.unit_price_p99 = unit_price_p99
        self.line_total_p95 = line_total_p95
        self.price_dev_threshold = price_dev_threshold

    def explain_record(
        self,
        record: Dict[str, Any],
        is_anomaly: bool,
        price_deviation: float = 0.0,
        prior_orders: int = 0,
    ) -> str:
        """Explains why a single transaction was flagged as anomalous."""
        if not is_anomaly:
            return ""

        reasons: List[str] = []

        # 1. Structural / Status checks
        status = str(record.get("transaction_status", ""))
        is_dup = bool(record.get("is_duplicate", False))
        if is_dup:
            reasons.append("EXACT_DUPLICATE_ROW")
        if status in ("INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY"):
            reasons.append("ADMINISTRATIVE_ADJUSTMENT")

        # 2. Extreme quantity
        qty = abs(float(record.get("quantity", 0.0)))
        if qty >= self.quantity_p99:
            reasons.append("EXTREME_QUANTITY")

        # 3. Extreme line value
        line_tot = abs(float(record.get("line_total", 0.0)))
        if line_tot >= self.line_total_p99:
            reasons.append("EXTREME_LINE_VALUE")

        # 4. High unit price
        price = float(record.get("unit_price", 0.0))
        if price >= self.unit_price_p99:
            reasons.append("HIGH_UNIT_PRICE")

        # 5. Unusual product price deviation
        if abs(price_deviation) >= self.price_dev_threshold:
            reasons.append("UNUSUAL_PRODUCT_PRICE")

        # 6. Off-hours transaction
        hour = record.get("invoice_hour")
        if hour is not None and (hour < 7 or hour > 19):
            reasons.append("OFF_HOURS_TRANSACTION")

        # 7. First-time customer high-value order
        if prior_orders == 0 and line_tot >= self.line_total_p95:
            reasons.append("NEW_CUSTOMER_HIGH_VALUE")

        # Fallback if isolation forest isolated but no single univariate rule triggered
        if not reasons:
            reasons.append("MULTIVARIATE_ISOLATION_OUTLIER")

        return ";".join(reasons)

    def explain_batch(
        self,
        df: pd.DataFrame,
        is_anomaly_array: np.ndarray,
        price_deviations: np.ndarray,
        prior_orders_array: np.ndarray,
    ) -> List[str]:
        """Generates explanations for a batch of transactions vectorized/iterated."""
        explanations: List[str] = []
        quantities = np.abs(df["quantity"].fillna(0).values)
        unit_prices = df["unit_price"].fillna(0).values
        if "line_total" in df.columns:
            line_totals = np.abs(df["line_total"].fillna(0).values)
        else:
            line_totals = quantities * np.maximum(0.0, unit_prices)

        hours = (
            df["invoice_hour"].values
            if "invoice_hour" in df.columns
            else np.full(len(df), 12)
        )
        statuses = (
            df["transaction_status"].values
            if "transaction_status" in df.columns
            else np.full(len(df), "")
        )
        duplicates = (
            df["is_duplicate"].values
            if "is_duplicate" in df.columns
            else np.zeros(len(df), dtype=bool)
        )

        for i in range(len(df)):
            if not is_anomaly_array[i]:
                explanations.append("")
                continue

            reasons: List[str] = []
            if duplicates[i]:
                reasons.append("EXACT_DUPLICATE_ROW")
            if statuses[i] in ("INVENTORY_WRITE_OFF", "ZERO_PRICE_ENTRY"):
                reasons.append("ADMINISTRATIVE_ADJUSTMENT")
            if quantities[i] >= self.quantity_p99:
                reasons.append("EXTREME_QUANTITY")
            if line_totals[i] >= self.line_total_p99:
                reasons.append("EXTREME_LINE_VALUE")
            if unit_prices[i] >= self.unit_price_p99:
                reasons.append("HIGH_UNIT_PRICE")
            if abs(price_deviations[i]) >= self.price_dev_threshold:
                reasons.append("UNUSUAL_PRODUCT_PRICE")
            if hours[i] < 7 or hours[i] > 19:
                reasons.append("OFF_HOURS_TRANSACTION")
            if prior_orders_array[i] == 0 and line_totals[i] >= self.line_total_p95:
                reasons.append("NEW_CUSTOMER_HIGH_VALUE")

            if not reasons:
                reasons.append("MULTIVARIATE_ISOLATION_OUTLIER")

            explanations.append(";".join(reasons))

        return explanations


class TransactionAnomalyDetector:
    """Production-grade Isolation Forest transaction anomaly detector.

    Implements leakage-safe feature preprocessing, model training,
    interpretable diagnostic rule generation, threshold calibration,
    persistence, and batch / single-record scoring.
    """

    def __init__(self, config: Optional[AnomalyDetectionConfig] = None) -> None:
        self.config = config or AnomalyDetectionConfig()
        self.pipeline = TransactionFeaturePipeline(use_scaler=True)
        self.model: Optional[IsolationForest] = None
        self.explainer: Optional[DiagnosticRuleExplainer] = None
        self.is_fitted: bool = False
        self.training_metadata_: Dict[str, Any] = {}
        self.evaluation_metrics_: Dict[str, Any] = {}
        self.threshold_policy_: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "TransactionAnomalyDetector":
        """Fits feature pipeline and Isolation Forest on eligible commercial transactions.

        Eligible population: non-duplicate records with valid price (unit_price > 0).

        Args:
            df: DataFrame containing Silver transaction records.

        Returns:
            self
        """
        logger.info(f"Fitting TransactionAnomalyDetector on {len(df):,} input records...")
        start_time = datetime.now(timezone.utc)

        # 1. Isolate eligible commercial transactions for training
        if "is_duplicate" in df.columns and "unit_price" in df.columns:
            eligible_mask = (~df["is_duplicate"]) & (df["unit_price"] > 0)
            train_df = df[eligible_mask].copy()
        else:
            train_df = df.copy()

        logger.info(
            f"Training on {len(train_df):,} eligible commercial records "
            f"({len(df) - len(train_df):,} excluded: duplicates / non-commercial adjustments)."
        )

        # 2. Fit feature pipeline on eligible training data
        X_train = self.pipeline.fit_transform(train_df)

        # 3. Fit Isolation Forest
        max_samples = (
            min(self.config.max_samples, len(train_df))
            if isinstance(self.config.max_samples, int)
            else self.config.max_samples
        )
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.model.fit(X_train)

        # 4. Calibrate diagnostic explainer thresholds from training data
        abs_qty = np.abs(train_df["quantity"].fillna(0).values)
        unit_prices = train_df["unit_price"].fillna(0).values
        line_tots = (
            np.abs(train_df["line_total"].fillna(0).values)
            if "line_total" in train_df.columns
            else abs_qty * unit_prices
        )

        self.config.quantity_p99 = float(np.percentile(abs_qty, 99))
        self.config.line_total_p99 = float(np.percentile(line_tots, 99))
        self.config.unit_price_p99 = float(np.percentile(unit_prices, 99))
        line_total_p95 = float(np.percentile(line_tots, 95))

        self.explainer = DiagnosticRuleExplainer(
            quantity_p99=self.config.quantity_p99,
            line_total_p99=self.config.line_total_p99,
            unit_price_p99=self.config.unit_price_p99,
            line_total_p95=line_total_p95,
            price_dev_threshold=self.config.price_deviation_threshold,
        )

        # 5. Compute training score distribution
        raw_train_scores = self.model.decision_function(X_train)
        anomaly_train_scores = -raw_train_scores
        offset = float(self.model.offset_)
        flagged_mask = raw_train_scores < 0.0

        end_time = datetime.now(timezone.utc)
        duration_sec = (end_time - start_time).total_seconds()

        self.threshold_policy_ = {
            "contamination_parameter": self.config.contamination,
            "internal_offset": offset,
            "decision_threshold": 0.0,
            "score_direction": (
                "raw_decision_score: scikit-learn decision_function(X) (higher is normal, negative is anomalous). "
                "anomaly_score: -raw_decision_score (higher is more anomalous, positive is anomalous)."
            ),
            "anomaly_decision_rule": "is_anomaly = raw_decision_score < 0.0 (or anomaly_score > 0.0)",
        }

        self.evaluation_metrics_ = {
            "training_records_count": len(train_df),
            "excluded_records_count": len(df) - len(train_df),
            "training_flagged_count": int(flagged_mask.sum()),
            "training_flagged_pct": round(float(flagged_mask.mean() * 100), 4),
            "raw_decision_score_stats": {
                "mean": round(float(raw_train_scores.mean()), 6),
                "std": round(float(raw_train_scores.std()), 6),
                "min": round(float(raw_train_scores.min()), 6),
                "p01": round(float(np.percentile(raw_train_scores, 1)), 6),
                "p05": round(float(np.percentile(raw_train_scores, 5)), 6),
                "p50": round(float(np.percentile(raw_train_scores, 50)), 6),
                "p95": round(float(np.percentile(raw_train_scores, 95)), 6),
                "p99": round(float(np.percentile(raw_train_scores, 99)), 6),
                "max": round(float(raw_train_scores.max()), 6),
            },
            "anomaly_score_stats": {
                "mean": round(float(anomaly_train_scores.mean()), 6),
                "std": round(float(anomaly_train_scores.std()), 6),
                "min": round(float(anomaly_train_scores.min()), 6),
                "p50": round(float(np.percentile(anomaly_train_scores, 50)), 6),
                "p99": round(float(np.percentile(anomaly_train_scores, 99)), 6),
                "max": round(float(anomaly_train_scores.max()), 6),
            },
            "diagnostic_thresholds": {
                "quantity_p99": self.config.quantity_p99,
                "line_total_p99": self.config.line_total_p99,
                "unit_price_p99": self.config.unit_price_p99,
                "line_total_p95": line_total_p95,
                "price_dev_threshold": self.config.price_deviation_threshold,
            },
        }

        self.training_metadata_ = {
            "model_version": self.config.model_version,
            "trained_at": start_time.isoformat(),
            "training_duration_seconds": round(duration_sec, 2),
            "n_estimators": self.config.n_estimators,
            "max_samples": self.config.max_samples,
            "contamination": self.config.contamination,
            "random_state": self.config.random_state,
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
        }

        self.is_fitted = True
        logger.info(
            f"Successfully trained Isolation Forest in {duration_sec:.2f}s. "
            f"Offset threshold: {offset:.6f}, Flagged: {flagged_mask.sum():,} ({flagged_mask.mean()*100:.2f}%)."
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scores transaction records and returns full dataset with scores and lineage.

        Preserves 100% of input rows and establishes stable source row lineage.

        Args:
            df: DataFrame of transaction records.

        Returns:
            Scored DataFrame containing transaction IDs, raw scores, anomaly scores,
            flags, and diagnostic reason codes.
        """
        if not self.is_fitted or self.model is None or self.explainer is None:
            raise ValueError("Model must be fitted or loaded before calling predict().")

        n_rows = len(df)
        logger.info(f"Scoring {n_rows:,} transactions with Isolation Forest...")

        # 1. Extract feature matrix
        X = self.pipeline.transform(df)

        # 2. Compute Isolation Forest scores
        raw_scores = self.model.decision_function(X)
        anomaly_scores = -raw_scores
        offset = float(self.model.offset_)
        is_anomaly = raw_scores < 0.0

        # 3. Extract diagnostic inputs
        # Price deviations
        if "stock_code" in df.columns and "unit_price" in df.columns:
            prices = df["unit_price"].fillna(0).values
            stock_codes = df["stock_code"].astype(str).values
            medians = np.array([
                self.pipeline.catalog_median_prices_.get(sc, self.pipeline.global_median_price_)
                for sc in stock_codes
            ])
            medians = np.where(medians <= 0, self.pipeline.global_median_price_, medians)
            safe_price = np.maximum(0.001, np.maximum(0.0, prices))
            safe_median = np.maximum(0.001, medians)
            price_devs = np.log(np.clip(safe_price / safe_median, 0.01, 100.0))
        else:
            price_devs = np.zeros(n_rows)

        # Prior orders
        if "prior_orders" in df.columns:
            prior_orders = df["prior_orders"].fillna(0).values.astype(int)
        elif "prior_customer_orders" in df.columns:
            prior_orders = df["prior_customer_orders"].fillna(0).values.astype(int)
        elif n_rows > 1 and "customer_id" in df.columns and "invoice_no" in df.columns:
            prior_orders = compute_leakage_safe_prior_orders(df).values.astype(int)
        else:
            prior_orders = np.zeros(n_rows, dtype=int)

        # 4. Generate diagnostic reason codes
        diagnostic_reasons = self.explainer.explain_batch(
            df, is_anomaly, price_devs, prior_orders
        )

        # 5. Build stable lineage identifiers
        if "source_row_id" in df.columns:
            source_row_ids = df["source_row_id"].values.astype(int)
        else:
            source_row_ids = np.arange(n_rows, dtype=int)

        transaction_ids = [f"TXN_{r:07d}" for r in source_row_ids]

        now_utc = datetime.now(timezone.utc)

        # 6. Assemble output DataFrame
        scored_df = pd.DataFrame({
            "transaction_id": transaction_ids,
            "source_row_id": source_row_ids,
            "invoice_no": df["invoice_no"].astype(str).values if "invoice_no" in df.columns else ["" for _ in range(n_rows)],
            "stock_code": df["stock_code"].astype(str).values if "stock_code" in df.columns else ["" for _ in range(n_rows)],
            "invoice_date": pd.to_datetime(df["invoice_date"]).values if "invoice_date" in df.columns else [now_utc for _ in range(n_rows)],
            "transaction_status": df["transaction_status"].astype(str).values if "transaction_status" in df.columns else ["VALID_PURCHASE" for _ in range(n_rows)],
            "quantity": df["quantity"].values if "quantity" in df.columns else np.zeros(n_rows),
            "unit_price": df["unit_price"].values if "unit_price" in df.columns else np.zeros(n_rows),
            "line_total": df["line_total"].values if "line_total" in df.columns else (df["quantity"].values * df["unit_price"].values if "quantity" in df.columns and "unit_price" in df.columns else np.zeros(n_rows)),
            "raw_decision_score": np.round(raw_scores, 6),
            "anomaly_score": np.round(anomaly_scores, 6),
            "is_anomaly": is_anomaly,
            "diagnostic_reasons": diagnostic_reasons,
            "model_version": self.config.model_version,
            "scoring_timestamp": now_utc,
        })

        return scored_df

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Performs validated single-record transaction anomaly scoring.

        Args:
            record: Dictionary containing transaction fields.

        Returns:
            Dictionary with prediction results and diagnostic reasons.

        Raises:
            ValueError: If required fields are missing or values are non-finite.
        """
        if not self.is_fitted or self.model is None or self.explainer is None:
            raise ValueError("Model must be fitted or loaded before calling predict_record().")

        required_keys = ["quantity", "unit_price"]
        for k in required_keys:
            if k not in record:
                raise ValueError(f"Missing required transaction field: '{k}'.")

        # Validate finite numeric values
        qty = record["quantity"]
        price = record["unit_price"]
        if not np.isfinite(qty) or not isinstance(qty, (int, float, np.integer, np.floating)):
            raise ValueError(f"Invalid non-finite or non-numeric quantity: {qty}")
        if not np.isfinite(price) or not isinstance(price, (int, float, np.integer, np.floating)):
            raise ValueError(f"Invalid non-finite or non-numeric unit_price: {price}")

        # Construct single-row DataFrame
        df_single = pd.DataFrame([record])
        scored_df = self.predict(df_single)
        row = scored_df.iloc[0]

        return {
            "transaction_id": row["transaction_id"],
            "raw_decision_score": float(row["raw_decision_score"]),
            "anomaly_score": float(row["anomaly_score"]),
            "is_anomaly": bool(row["is_anomaly"]),
            "diagnostic_reasons": str(row["diagnostic_reasons"]),
            "model_version": str(row["model_version"]),
        }

    def evaluate_sensitivity_and_stability(
        self, df: pd.DataFrame, sample_size: int = 50000
    ) -> Dict[str, Any]:
        """Performs sensitivity analysis across contamination settings,

        stability across random seeds, and synthetic diagnostic injection.
        """
        logger.info("Evaluating contamination sensitivity and random seed stability...")
        eligible = df[(~df["is_duplicate"]) & (df["unit_price"] > 0)].copy() if "is_duplicate" in df.columns else df.copy()
        X_full = self.pipeline.transform(eligible)

        # 1. Contamination sensitivity
        contaminations = [0.005, 0.01, 0.02, 0.05]
        sensitivity_results: List[Dict[str, Any]] = []
        raw_full_scores = self.model.decision_function(X_full)
        for c in contaminations:
            # Using current model's raw decision scores, evaluate threshold at (c * 100) percentile
            threshold = float(np.percentile(raw_full_scores, c * 100))
            flagged = int((raw_full_scores < threshold).sum())
            sensitivity_results.append({
                "contamination": c,
                "decision_offset_threshold": round(threshold, 6),
                "flagged_count": flagged,
                "flagged_percentage": round(flagged / len(eligible) * 100, 2),
            })

        # 2. Seed stability on representative sub-sample
        sample_n = min(sample_size, len(eligible))
        np.random.seed(42)
        idx = np.random.choice(len(eligible), size=sample_n, replace=False)
        X_sub = X_full[idx]

        seeds = [42, 123, 456, 789]
        seed_scores = {}
        seed_flags = {}
        max_samples_sub = (
            min(self.config.max_samples, len(X_sub))
            if isinstance(self.config.max_samples, int)
            else self.config.max_samples
        )
        for s in seeds:
            clf_s = IsolationForest(
                n_estimators=self.config.n_estimators,
                max_samples=max_samples_sub,
                contamination=self.config.contamination,
                random_state=s,
                n_jobs=-1,
            )
            clf_s.fit(X_sub)
            raw_s = clf_s.decision_function(X_sub)
            seed_scores[s] = raw_s
            seed_flags[s] = (raw_s < 0.0).astype(int)

        # Compute pairwise Spearman rank correlation and Jaccard overlap
        spearman_corrs = []
        jaccard_sims = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                s1, s2 = seeds[i], seeds[j]
                rho, _ = spearmanr(seed_scores[s1], seed_scores[s2])
                spearman_corrs.append(float(rho))

                # Jaccard overlap of flagged sets
                f1 = set(np.where(seed_flags[s1] == 1)[0])
                f2 = set(np.where(seed_flags[s2] == 1)[0])
                intersection = len(f1.intersection(f2))
                union = len(f1.union(f2))
                jaccard = intersection / union if union > 0 else 1.0
                jaccard_sims.append(float(jaccard))

        # 3. Synthetic anomaly injection diagnostic test
        # Inject deliberate extreme anomalies:
        synthetic_records = [
            {"invoice_no": "SYNTH01", "stock_code": "23843", "quantity": 50000, "unit_price": 250.0, "invoice_date": "2010-06-15 14:00:00", "transaction_status": "VALID_PURCHASE"},
            {"invoice_no": "SYNTH02", "stock_code": "85048", "quantity": 1, "unit_price": 1000.0, "invoice_date": "2010-06-15 14:00:00", "transaction_status": "VALID_PURCHASE"},
            {"invoice_no": "SYNTH03", "stock_code": "22502", "quantity": -30000, "unit_price": 500.0, "invoice_date": "2010-06-15 03:00:00", "transaction_status": "CANCELLATION"},
            {"invoice_no": "SYNTH04", "stock_code": "84347", "quantity": 10000, "unit_price": 50.0, "invoice_date": "2010-06-15 23:00:00", "transaction_status": "VALID_PURCHASE"},
            {"invoice_no": "SYNTH05", "stock_code": "M", "quantity": 1, "unit_price": 5000.0, "invoice_date": "2010-06-15 12:00:00", "transaction_status": "VALID_PURCHASE"},
        ]
        synth_df = pd.DataFrame(synthetic_records)
        scored_synth = self.predict(synth_df)
        synth_detected = int(scored_synth["is_anomaly"].sum())

        return {
            "contamination_sensitivity": sensitivity_results,
            "seed_stability": {
                "tested_seeds": seeds,
                "spearman_rank_correlation_min": round(min(spearman_corrs), 4),
                "spearman_rank_correlation_mean": round(float(np.mean(spearman_corrs)), 4),
                "spearman_rank_correlation_max": round(max(spearman_corrs), 4),
                "jaccard_flag_similarity_min": round(min(jaccard_sims), 4),
                "jaccard_flag_similarity_mean": round(float(np.mean(jaccard_sims)), 4),
                "jaccard_flag_similarity_max": round(max(jaccard_sims), 4),
            },
            "synthetic_diagnostic_test": {
                "injected_count": len(synthetic_records),
                "detected_count": synth_detected,
                "diagnostic_detection_rate": round(synth_detected / len(synthetic_records), 2),
                "synthetic_interpretations": scored_synth[["transaction_id", "quantity", "unit_price", "anomaly_score", "is_anomaly", "diagnostic_reasons"]].to_dict(orient="records"),
            },
        }

    def save(self, model_dir: Optional[Union[str, Path]] = None) -> Path:
        """Persists trained Isolation Forest model, preprocessor, and metadata.

        Args:
            model_dir: Directory where artifacts are saved.

        Returns:
            Path to the saved directory.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted TransactionAnomalyDetector.")

        out_dir = Path(model_dir) if model_dir else self.config.model_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Persisting TransactionAnomalyDetector to {out_dir}...")

        # 1. Scikit-learn model artifact
        joblib.dump(self.model, out_dir / "isolation_forest_model.joblib")

        # 2. Fitted preprocessor artifact
        joblib.dump(self.pipeline, out_dir / "preprocessor.joblib")

        # 3. Feature schema
        feature_schema = {
            "features": FEATURE_NAMES,
            "feature_count": len(FEATURE_NAMES),
            "expected_input_columns": [
                "invoice_no", "stock_code", "quantity", "unit_price",
                "invoice_date", "customer_id", "country", "transaction_status",
            ],
            "numeric_constraints": {
                "quantity": "finite float or integer",
                "unit_price": "finite non-negative float",
            },
        }
        with open(out_dir / "feature_schema.json", "w") as f:
            json.dump(feature_schema, f, indent=2)

        # 4. Hyperparameters
        hyperparameters = {
            "model_version": self.config.model_version,
            "n_estimators": self.config.n_estimators,
            "max_samples": self.config.max_samples,
            "contamination": self.config.contamination,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
        }
        with open(out_dir / "hyperparameters.json", "w") as f:
            json.dump(hyperparameters, f, indent=2)

        # 5. Training metadata
        with open(out_dir / "training_metadata.json", "w") as f:
            json.dump(self.training_metadata_, f, indent=2)

        # 6. Threshold policy
        with open(out_dir / "threshold_policy.json", "w") as f:
            json.dump(self.threshold_policy_, f, indent=2)

        # 7. Evaluation metrics
        with open(out_dir / "evaluation_metrics.json", "w") as f:
            json.dump(self.evaluation_metrics_, f, indent=2)

        logger.info(f"Artifacts successfully saved to {out_dir}")
        return out_dir

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "TransactionAnomalyDetector":
        """Loads and reconstructs an operational TransactionAnomalyDetector from disk.

        Args:
            model_dir: Path to directory containing persisted artifacts.

        Returns:
            Reconstructed TransactionAnomalyDetector instance.
        """
        in_dir = Path(model_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {in_dir}")

        logger.info(f"Loading TransactionAnomalyDetector from {in_dir}...")

        # 1. Hyperparameters
        with open(in_dir / "hyperparameters.json") as f:
            hp = json.load(f)

        config = AnomalyDetectionConfig(
            model_version=hp.get("model_version", "v1.0.0"),
            n_estimators=hp.get("n_estimators", 100),
            max_samples=hp.get("max_samples", 256),
            contamination=hp.get("contamination", 0.01),
            random_state=hp.get("random_state", 42),
            n_jobs=hp.get("n_jobs", -1),
        )

        detector = cls(config=config)

        # 2. Model & preprocessor
        detector.model = joblib.load(in_dir / "isolation_forest_model.joblib")
        detector.pipeline = joblib.load(in_dir / "preprocessor.joblib")

        # 3. Metadata, policy, and metrics
        with open(in_dir / "training_metadata.json") as f:
            detector.training_metadata_ = json.load(f)

        with open(in_dir / "threshold_policy.json") as f:
            detector.threshold_policy_ = json.load(f)

        with open(in_dir / "evaluation_metrics.json") as f:
            detector.evaluation_metrics_ = json.load(f)

        # 4. Diagnostic explainer
        diag_thresh = detector.evaluation_metrics_.get("diagnostic_thresholds", {})
        detector.explainer = DiagnosticRuleExplainer(
            quantity_p99=diag_thresh.get("quantity_p99", 100.0),
            line_total_p99=diag_thresh.get("line_total_p99", 180.0),
            unit_price_p99=diag_thresh.get("unit_price_p99", 18.0),
            line_total_p95=diag_thresh.get("line_total_p95", 67.0),
            price_dev_threshold=diag_thresh.get("price_dev_threshold", 1.10),
        )

        detector.is_fitted = True
        logger.info("Successfully reloaded TransactionAnomalyDetector.")
        return detector
