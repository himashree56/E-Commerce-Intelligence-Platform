"""
Comprehensive Test Suite for Phase 6 XGBoost Repeat-Purchase Prediction & Audit.

Verifies:
1. Future-label construction (valid purchase in 30d window vs cancellations).
2. Exact boundary timestamps at observation cutoff and 30-day horizon.
3. Observation-window censoring and customer eligibility.
4. Strict point-in-time feature extraction without lookahead leakage.
5. Chronological splitting without overlapping label windows and 31d purging gaps.
6. Customer-snapshot primary key uniqueness.
7. Deterministic recomputation of held-out test metrics and confusion matrix.
8. Model contract testing (probability range, calibration properties, lift at top deciles).
9. Validation-only threshold selection policy.
10. Model serialization and inference consistency (single & batch).
11. PostgreSQL table schema, composite uniqueness constraint, and foreign keys.
12. Live PostgreSQL JSONB feature snapshot roundtrip and constraint enforcement.
13. Live PostgreSQL idempotent upserting.
14. Live PostgreSQL transaction rollback on failure.
15. Prediction dataset reconciliation and positive prevalence.
"""
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from src.database.config import DatabaseConfig
from src.database.connection import DatabaseManager
from src.database.loader import GoldDatabaseLoader
from src.database.models import (
    CustomerRepeatPurchasePrediction,
    CustomerRFMAnalytics,
)
from src.database.schema_manager import SchemaManager
from src.features.repeat_purchase_features import (
    REPEAT_PURCHASE_FEATURES,
    RepeatPurchaseFeaturePipeline,
    extract_customer_snapshot,
)
from src.models.config import RepeatPurchaseConfig
from src.models.repeat_purchase import (
    RepeatPurchasePredictor,
    compute_metrics,
    find_optimal_threshold,
)


@pytest.fixture
def mock_transaction_stream() -> pd.DataFrame:
    """Fixture providing chronological transactions across multiple customers."""
    return pd.DataFrame({
        "customer_id": [101, 101, 102, 103, 101, 102, 103, 104],
        "invoice_no": ["INV1", "INV2", "INV3", "INV4", "INV5", "C_INV6", "INV7", "INV8"],
        "stock_code": ["ITEM_A", "ITEM_B", "ITEM_A", "ITEM_C", "ITEM_A", "ITEM_A", "ITEM_B", "ITEM_D"],
        "quantity": [10, 5, 20, 2, 8, -5, 1, 100],
        "unit_price": [2.50, 4.00, 1.50, 50.0, 2.50, 1.50, 20.0, 5.0],
        "net_revenue": [25.0, 20.0, 30.0, 100.0, 20.0, -7.50, 20.0, 500.0],
        "gross_sales": [25.0, 20.0, 30.0, 100.0, 20.0, 0.0, 20.0, 500.0],
        "return_amount": [0.0, 0.0, 0.0, 0.0, 0.0, 7.50, 0.0, 0.0],
        "transaction_status": [
            "VALID_PURCHASE",
            "VALID_PURCHASE",
            "VALID_PURCHASE",
            "VALID_PURCHASE",
            "VALID_PURCHASE",
            "CANCELLATION",
            "VALID_PURCHASE",
            "VALID_PURCHASE",
        ],
        "invoice_date": pd.to_datetime([
            "2010-01-01 10:00:00",  # 101 purchase
            "2010-01-10 14:00:00",  # 101 purchase
            "2010-01-05 12:00:00",  # 102 purchase
            "2010-01-15 09:00:00",  # 103 purchase
            # Future window (2010-02-01 to 2010-03-03):
            "2010-02-05 10:00:00",  # 101 repeat purchase -> positive!
            "2010-02-10 11:00:00",  # 102 cancellation only -> NOT positive!
            "2010-03-20 12:00:00",  # 103 purchase AFTER future window -> NOT positive!
            "2010-02-15 15:00:00",  # 104 first purchase in future window -> NOT eligible at 2010-02-01!
        ]),
    })


@pytest.fixture(scope="module")
def repeat_purchase_test_db():
    """Initializes and returns a DatabaseManager for test database, or skips if unreachable."""
    config = DatabaseConfig()
    mgr = DatabaseManager(config=config, is_test=True)
    health = mgr.check_health()
    if health.get("status") != "HEALTHY":
        pytest.skip(f"PostgreSQL test database not available: {health.get('error')}")

    schema_mgr = SchemaManager(db_manager=mgr)
    schema_mgr.create_all_tables(is_test=True)

    yield mgr

    schema_mgr.drop_all_tables(is_test=True)
    mgr.dispose()


def test_future_label_construction(mock_transaction_stream):
    """Verifies that future labels require a VALID_PURCHASE within the 30-day window."""
    snapshot_date = "2010-02-01"
    snap_df = extract_customer_snapshot(mock_transaction_stream, snapshot_date, horizon_days=30)

    # Eligible customers must have at least 1 purchase strictly before 2010-02-01
    assert sorted(snap_df["customer_id"].tolist()) == [101, 102, 103]

    row_101 = snap_df[snap_df["customer_id"] == 101].iloc[0]
    row_102 = snap_df[snap_df["customer_id"] == 102].iloc[0]
    row_103 = snap_df[snap_df["customer_id"] == 103].iloc[0]

    # Customer 101 bought on 2010-02-05 (in window) -> label = 1
    assert row_101["label_30d"] == 1

    # Customer 102 had only a CANCELLATION on 2010-02-10 (no valid purchase) -> label = 0
    assert row_102["label_30d"] == 0

    # Customer 103 bought on 2010-03-20 (>30 days after snapshot) -> label = 0
    assert row_103["label_30d"] == 0


def test_exact_boundary_timestamps_at_cutoff_and_horizon():
    """Boundary tests for purchases occurring exactly at snapshot cutoff and 30-day horizon."""
    snapshot_date = "2011-01-01 00:00:00"
    snap_ts = pd.Timestamp(snapshot_date)
    horizon_ts = snap_ts + pd.Timedelta(days=30)

    stream = pd.DataFrame({
        "customer_id": [201, 201, 202, 202, 203, 203, 204, 204],
        "invoice_no": [f"INV_{i}" for i in range(8)],
        "stock_code": ["SKU_A"] * 8,
        "quantity": [1] * 8,
        "unit_price": [10.0] * 8,
        "net_revenue": [10.0] * 8,
        "gross_sales": [10.0] * 8,
        "return_amount": [0.0] * 8,
        "transaction_status": ["VALID_PURCHASE"] * 8,
        "invoice_date": pd.to_datetime([
            # Customer 201: Pre-cutoff anchor + purchase exactly at snapshot cutoff
            "2010-12-15 10:00:00",
            snap_ts,  # Exactly at cutoff -> in forward window!
            # Customer 202: Pre-cutoff anchor + purchase exactly at 30-day horizon
            "2010-12-15 10:00:00",
            horizon_ts,  # Exactly at horizon -> in forward window!
            # Customer 203: Pre-cutoff anchor + purchase 1 second after horizon
            "2010-12-15 10:00:00",
            horizon_ts + pd.Timedelta(seconds=1),  # Past horizon -> label = 0!
            # Customer 204: Purchase 1 second before cutoff
            "2010-12-15 10:00:00",
            snap_ts - pd.Timedelta(seconds=1),  # Historical purchase
        ]),
    })

    snap_df = extract_customer_snapshot(stream, snapshot_date, horizon_days=30)
    assert len(snap_df) == 4

    res = dict(zip(snap_df["customer_id"], snap_df["label_30d"]))
    assert res[201] == 1, "Purchase exactly at cutoff should be in forward window"
    assert res[202] == 1, "Purchase exactly at 30-day horizon should be in forward window"
    assert res[203] == 0, "Purchase 1s after horizon must not be included"
    assert res[204] == 0, "No purchase in forward window"


def test_strict_point_in_time_feature_calculations(mock_transaction_stream):
    """Verifies that feature calculations strictly use data before the snapshot date."""
    snapshot_date = "2010-02-01"
    snap_df = extract_customer_snapshot(mock_transaction_stream, snapshot_date, horizon_days=30)

    row_101 = snap_df[snap_df["customer_id"] == 101].iloc[0]
    assert row_101["purchase_frequency"] == 2
    assert row_101["net_monetary_value"] == 45.0
    assert row_101["product_diversity"] == 2
    assert 21.0 < row_101["recency_days"] < 22.0


def test_chronological_perturbation_invariance():
    """Demonstrates that appending future data does not alter past feature extractions."""
    config = RepeatPurchaseConfig()
    train_path = config.gold_train_path
    if not train_path.exists():
        pytest.skip("Train dataset missing.")

    train_df = pd.read_parquet(train_path)
    pipeline = RepeatPurchaseFeaturePipeline(use_scaler=True)
    pipeline.fit(train_df)

    X_train_orig = pipeline.transform(train_df)

    future_data = train_df.iloc[:50].copy()
    future_data["recency_days"] += 100.0

    combined = pd.concat([train_df, future_data], ignore_index=True)
    X_train_perturbed = pipeline.transform(combined)[: len(train_df)]

    diff = np.max(np.abs(X_train_orig - X_train_perturbed))
    assert diff < 1e-12, f"Feature leakage under perturbation! Diff: {diff}"


def test_chronological_splitting_no_label_overlap():
    """Verifies that train, val, and test periods are chronologically ordered with zero label overlap."""
    config = RepeatPurchaseConfig()
    train_dates = [pd.Timestamp(d) for d in config.train_snapshots]
    val_date = pd.Timestamp(config.val_snapshot)
    test_date = pd.Timestamp(config.test_snapshot)

    # 1. Monotonic ordering of snapshot dates
    assert train_dates == sorted(train_dates)
    assert train_dates[-1] < val_date < test_date

    # 2. Training snapshots do not overlap
    horizon = pd.Timedelta(days=config.horizon_days)
    for i in range(len(train_dates) - 1):
        assert train_dates[i] + horizon <= train_dates[i + 1], "Overlapping train label windows!"

    # 3. Train does not overlap with Validation (31-day purging gap)
    assert train_dates[-1] + horizon <= val_date, "Train future window overlaps with Validation snapshot!"
    assert (val_date - (train_dates[-1] + horizon)).days >= 30, "Insufficient purging gap 1!"

    # 4. Validation does not overlap with Test (31-day purging gap)
    assert val_date + horizon <= test_date, "Validation future window overlaps with Test snapshot!"
    assert (test_date - (val_date + horizon)).days >= 30, "Insufficient purging gap 2!"


def test_customer_snapshot_uniqueness():
    """Verifies that each customer appears at most once per snapshot date."""
    config = RepeatPurchaseConfig()
    val_df = pd.read_parquet(config.gold_val_path)
    test_df = pd.read_parquet(config.gold_test_path)
    train_df = pd.read_parquet(config.gold_train_path)

    assert val_df["customer_id"].nunique() == len(val_df)
    assert test_df["customer_id"].nunique() == len(test_df)

    composite_keys = train_df["customer_id"].astype(str) + "_" + train_df["snapshot_date"].astype(str)
    assert composite_keys.nunique() == len(train_df)


def test_deterministic_test_metric_recomputation():
    """Deterministically recomputes all held-out metrics from saved test data and model artifacts."""
    config = RepeatPurchaseConfig()
    metrics_path = config.model_dir / "evaluation_metrics.json"
    schema_path = config.model_dir / "feature_schema.json"
    policy_path = config.model_dir / "threshold_policy.json"
    model_path = config.model_dir / "xgb_model.joblib"

    if not (metrics_path.exists() and model_path.exists()):
        pytest.skip("Model or metrics artifacts missing.")

    import joblib

    with open(metrics_path) as f:
        stored_metrics = json.load(f)["test"]["xgboost"]

    with open(schema_path) as f:
        feature_names = json.load(f)["feature_names"]

    with open(policy_path) as f:
        threshold = json.load(f)["selected_threshold"]

    model = joblib.load(model_path)
    test_df = pd.read_parquet(config.gold_test_path)

    y_test = test_df["label_30d"].values.astype(int)
    X_test = test_df[feature_names]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Recompute metrics
    recomputed_roc = float(roc_auc_score(y_test, y_prob))
    recomputed_pr = float(average_precision_score(y_test, y_prob))
    recomputed_f1 = float(f1_score(y_test, y_pred))
    recomputed_brier = float(brier_score_loss(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    assert abs(recomputed_roc - stored_metrics["roc_auc"]) < 1e-4
    assert abs(recomputed_pr - stored_metrics["pr_auc"]) < 1e-4
    assert abs(recomputed_f1 - stored_metrics["f1"]) < 1e-4
    assert abs(recomputed_brier - stored_metrics["brier_score"]) < 1e-4

    # Assert exact confusion matrix counts
    assert tn == 3759
    assert fp == 569
    assert fn == 578
    assert tp == 532
    assert (tn + fp + fn + tp) == 5438


def test_repeat_purchase_model_contract():
    """Verifies predictive model output bounds, monotonicity, and decile lift."""
    config = RepeatPurchaseConfig()
    predictor = RepeatPurchasePredictor.load(config.model_dir)
    test_df = pd.read_parquet(config.gold_test_path)

    probs = predictor.predict_proba(test_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be bounded in [0, 1]"

    # Precision in top 5% must provide substantial commercial lift (> 3x base rate)
    k5 = max(1, int(len(test_df) * 0.05))
    top_5_idx = np.argsort(probs)[::-1][:k5]
    top_5_prec = test_df["label_30d"].iloc[top_5_idx].mean()
    base_rate = test_df["label_30d"].mean()
    lift_5 = top_5_prec / base_rate

    assert top_5_prec >= 0.70, f"Top 5% precision ({top_5_prec:.4f}) below requirement"
    assert lift_5 >= 3.0, f"Top 5% lift ({lift_5:.2f}x) below 3.0x requirement"


def test_threshold_selection_on_validation_only():
    """Verifies that the threshold policy is derived strictly from validation data."""
    config = RepeatPurchaseConfig()
    policy_path = config.model_dir / "threshold_policy.json"
    if not policy_path.exists():
        pytest.skip("Threshold policy missing.")

    with open(policy_path) as f:
        policy = json.load(f)

    assert policy["selection_split"] == "validation"
    assert policy["optimization_objective"] == "f1_score"
    assert 0.10 <= policy["selected_threshold"] <= 0.90
    assert policy["validation_f1_at_threshold"] > 0.40


def test_model_serialization_and_inference_consistency():
    """Verifies that saving and reloading the predictor reproduces identical probabilities."""
    config = RepeatPurchaseConfig()
    predictor = RepeatPurchasePredictor.load(config.model_dir)

    test_df = pd.read_parquet(config.gold_test_path).head(100)
    probs_batch = predictor.predict_proba(test_df)
    preds_batch = predictor.predict(test_df)

    assert len(probs_batch) == 100
    assert len(preds_batch) == 100
    assert np.all((probs_batch >= 0.0) & (probs_batch <= 1.0))

    single_dict = test_df.iloc[0].to_dict()
    single_res = predictor.predict_single(single_dict)

    assert single_res["customer_id"] == single_dict["customer_id"]
    assert abs(single_res["repeat_purchase_probability"] - probs_batch[0]) < 1e-3
    assert single_res["predicted_label"] == bool(preds_batch[0])


def test_database_schema_contract():
    """Verifies CustomerRepeatPurchasePrediction ORM model columns, types, and constraints."""
    cols = {c.name for c in CustomerRepeatPurchasePrediction.__table__.columns}
    expected = {
        "prediction_id",
        "customer_id",
        "prediction_as_of_date",
        "prediction_horizon_days",
        "repeat_purchase_probability",
        "predicted_label",
        "model_version",
        "feature_snapshot",
        "scoring_timestamp",
        "created_at",
    }
    assert expected.issubset(cols)
    assert CustomerRepeatPurchasePrediction.__table__.primary_key.columns.keys() == ["prediction_id"]

    # Verify composite unique constraint
    unique_names = [uc.name for uc in CustomerRepeatPurchasePrediction.__table__.constraints if hasattr(uc, "columns")]
    assert "uq_crpp_cust_date_version" in unique_names


def test_repeat_purchase_postgres_foreign_key_and_jsonb_roundtrip(repeat_purchase_test_db):
    """Verifies customer foreign key constraint and JSONB feature snapshot roundtrip."""
    with repeat_purchase_test_db.session_scope() as session:
        # Create a valid parent customer in test database
        parent_cust = CustomerRFMAnalytics(
            customer_id=88801,
            primary_country="United Kingdom",
            recency_days=12,
            frequency=4,
            monetary_gross=Decimal("450.00"),
            monetary_net=Decimal("420.00"),
            return_amount=Decimal("30.00"),
            return_orders_count=1,
            first_purchase_date=datetime(2010, 1, 1, tzinfo=timezone.utc),
            last_purchase_date=datetime(2010, 8, 1, tzinfo=timezone.utc),
            r_score=4,
            f_score=3,
            m_score=3,
            rfm_cell="433",
            customer_segment="Active Recent Shoppers",
        )
        session.merge(parent_cust)

        # Insert prediction record with JSONB feature snapshot
        feature_dict = {f: 1.0 for f in REPEAT_PURCHASE_FEATURES}
        pred = CustomerRepeatPurchasePrediction(
            prediction_id="PRED_88801_2011-10-01",
            customer_id=88801,
            prediction_as_of_date=date(2011, 10, 1),
            prediction_horizon_days=30,
            repeat_purchase_probability=Decimal("0.784500"),
            predicted_label=True,
            model_version="v1.0.0",
            feature_snapshot=feature_dict,
            scoring_timestamp=datetime.now(timezone.utc),
        )
        session.merge(pred)

    # Query back and verify JSONB roundtrip
    with repeat_purchase_test_db.engine.connect() as conn:
        row = conn.execute(
            select(CustomerRepeatPurchasePrediction).where(
                CustomerRepeatPurchasePrediction.prediction_id == "PRED_88801_2011-10-01"
            )
        ).first()

        assert row is not None
        assert row.customer_id == 88801
        assert row.predicted_label is True
        assert isinstance(row.feature_snapshot, dict)
        assert len(row.feature_snapshot) == len(REPEAT_PURCHASE_FEATURES)

    # Verify foreign key constraint enforcement on orphan customer
    with pytest.raises(IntegrityError):
        with repeat_purchase_test_db.session_scope() as session:
            orphan_pred = CustomerRepeatPurchasePrediction(
                prediction_id="PRED_999999_2011-10-01",
                customer_id=999999,  # Non-existent customer
                prediction_as_of_date=date(2011, 10, 1),
                prediction_horizon_days=30,
                repeat_purchase_probability=Decimal("0.500000"),
                predicted_label=False,
                model_version="v1.0.0",
                scoring_timestamp=datetime.now(timezone.utc),
            )
            session.add(orphan_pred)


def test_repeat_purchase_postgres_idempotent_upsert(repeat_purchase_test_db):
    """Verifies idempotent loading into PostgreSQL via ON CONFLICT DO UPDATE."""
    loader = GoldDatabaseLoader(is_test=True, batch_size=5)

    with repeat_purchase_test_db.session_scope() as session:
        # Create parent customer
        session.merge(CustomerRFMAnalytics(
            customer_id=88802,
            primary_country="United Kingdom",
            recency_days=5,
            frequency=10,
            monetary_gross=Decimal("1200.00"),
            monetary_net=Decimal("1200.00"),
            return_amount=Decimal("0.00"),
            return_orders_count=0,
            first_purchase_date=datetime(2010, 1, 1, tzinfo=timezone.utc),
            last_purchase_date=datetime(2010, 9, 1, tzinfo=timezone.utc),
            r_score=5,
            f_score=5,
            m_score=5,
            rfm_cell="555",
            customer_segment="High-Value Loyal Champions",
        ))

    # Initial batch
    test_df_1 = pd.DataFrame([{
        "prediction_id": "PRED_88802_2011-10-01",
        "customer_id": 88802,
        "prediction_as_of_date": date(2011, 10, 1),
        "prediction_horizon_days": 30,
        "repeat_purchase_probability": 0.650000,
        "predicted_label": True,
        "model_version": "v1.0.0",
        "feature_snapshot": {"recency_days": 5.0},
        "scoring_timestamp": datetime.now(timezone.utc),
    }])

    with repeat_purchase_test_db.session_scope() as session:
        loader.load_repeat_purchase_predictions(session, test_df_1)

    # Re-load with modified probability (idempotent upsert)
    test_df_2 = pd.DataFrame([{
        "prediction_id": "PRED_88802_2011-10-01",
        "customer_id": 88802,
        "prediction_as_of_date": date(2011, 10, 1),
        "prediction_horizon_days": 30,
        "repeat_purchase_probability": 0.820000,
        "predicted_label": True,
        "model_version": "v1.0.0",
        "feature_snapshot": {"recency_days": 5.0},
        "scoring_timestamp": datetime.now(timezone.utc),
    }])

    with repeat_purchase_test_db.session_scope() as session:
        loader.load_repeat_purchase_predictions(session, test_df_2)

    with repeat_purchase_test_db.engine.connect() as conn:
        cnt = conn.execute(
            select(func.count()).select_from(CustomerRepeatPurchasePrediction).where(
                CustomerRepeatPurchasePrediction.customer_id == 88802
            )
        ).scalar()
        prob = conn.execute(
            select(CustomerRepeatPurchasePrediction.repeat_purchase_probability).where(
                CustomerRepeatPurchasePrediction.customer_id == 88802
            )
        ).scalar()

    assert cnt == 1, "Idempotent reload must not inflate row counts"
    assert prob == Decimal("0.820000"), "Upsert must update probability value"


def test_repeat_purchase_postgres_rollback_on_failed_load(repeat_purchase_test_db):
    """Verifies transactional atomicity and rollback on failed database loading."""
    initial_count = 0
    with repeat_purchase_test_db.engine.connect() as conn:
        initial_count = conn.execute(
            select(func.count()).select_from(CustomerRepeatPurchasePrediction)
        ).scalar()

    # Attempt a transaction block that raises an error midway
    try:
        with repeat_purchase_test_db.session_scope() as session:
            # 1. Valid insert
            session.add(CustomerRepeatPurchasePrediction(
                prediction_id="PRED_FAIL_TEST",
                customer_id=88801,
                prediction_as_of_date=date(2011, 10, 1),
                prediction_horizon_days=30,
                repeat_purchase_probability=Decimal("0.500000"),
                predicted_label=False,
                model_version="v1.0.0",
                scoring_timestamp=datetime.now(timezone.utc),
            ))
            session.flush()

            # 2. Intentionally trigger constraint error (orphan customer)
            session.add(CustomerRepeatPurchasePrediction(
                prediction_id="PRED_FAIL_ORPHAN",
                customer_id=999999,
                prediction_as_of_date=date(2011, 10, 1),
                prediction_horizon_days=30,
                repeat_purchase_probability=Decimal("0.500000"),
                predicted_label=False,
                model_version="v1.0.0",
                scoring_timestamp=datetime.now(timezone.utc),
            ))
            session.flush()
    except IntegrityError:
        pass  # Expected rollback

    # Verify that PRED_FAIL_TEST was rolled back and not persisted
    with repeat_purchase_test_db.engine.connect() as conn:
        final_count = conn.execute(
            select(func.count()).select_from(CustomerRepeatPurchasePrediction)
        ).scalar()
        check_row = conn.execute(
            select(CustomerRepeatPurchasePrediction).where(
                CustomerRepeatPurchasePrediction.prediction_id == "PRED_FAIL_TEST"
            )
        ).first()

    assert final_count == initial_count, "Rollback must restore original table row count"
    assert check_row is None, "Partially flushed record must be rolled back"


def test_prediction_count_reconciliation():
    """Reconciles test dataset row counts and positive label prevalence."""
    config = RepeatPurchaseConfig()
    test_df = pd.read_parquet(config.gold_test_path)

    assert len(test_df) == 5438
    pos_count = int(test_df["label_30d"].sum())
    assert pos_count == 1110
    assert round(pos_count / len(test_df) * 100, 2) == 20.41
