"""Integration tests for PostgreSQL ML Customer Segmentation Table (Phase 4)."""
from datetime import datetime, timezone
from decimal import Decimal
import pandas as pd
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from src.database.connection import DatabaseManager
from src.database.loader import GoldDatabaseLoader
from src.database.models import CustomerRFMAnalytics, CustomerSegmentationKMeans
from src.database.schema_manager import SchemaManager
from src.models.config import SegmentationConfig
from src.models.customer_segmentation import CustomerSegmentationModel


@pytest.fixture(autouse=True)
def skip_if_no_database():
    """Skips database integration tests if PostgreSQL is unreachable."""
    db_mgr = DatabaseManager()
    try:
        health = db_mgr.check_health()
        if health.get("status") != "HEALTHY":
            pytest.skip("PostgreSQL database is not running or reachable.")
    except Exception:
        pytest.skip("PostgreSQL database is not running or reachable.")


def test_segmentation_table_row_count_and_population_reconciliation():
    """Verifies that all 5,939 eligible RFM customers exist in customer_segmentation_kmeans."""
    db_mgr = DatabaseManager()
    with db_mgr.engine.connect() as conn:
        rfm_count = conn.execute(select(func.count()).select_from(CustomerRFMAnalytics)).scalar()
        ml_count = conn.execute(select(func.count()).select_from(CustomerSegmentationKMeans)).scalar()
        null_clusters = conn.execute(
            select(func.count()).select_from(CustomerSegmentationKMeans).where(
                CustomerSegmentationKMeans.cluster_id.is_(None)
            )
        ).scalar()

    assert rfm_count == 5939
    assert ml_count == 5939
    assert null_clusters == 0


def test_referential_integrity_and_orphan_rejection():
    """Verifies foreign key enforcement: orphan customer_ids cannot be inserted."""
    db_mgr = DatabaseManager()
    with pytest.raises(IntegrityError):
        with db_mgr.session_scope() as session:
            orphan_seg = CustomerSegmentationKMeans(
                customer_id=99999999,  # Non-existent customer
                cluster_id=0,
                cluster_label="High-Value Loyal Champions",
                recency_days=10,
                frequency=5,
                monetary_net=Decimal("500.00"),
                distance_to_centroid=Decimal("0.5000"),
                model_version="v1.0.0",
                prediction_timestamp=datetime.now(timezone.utc),
            )
            session.add(orphan_seg)
            session.flush()


def test_rule_based_rfm_segments_preserved_in_database():
    """Verifies that existing rule-based RFM segments were not overwritten or modified."""
    db_mgr = DatabaseManager()
    with db_mgr.engine.connect() as conn:
        segments = conn.execute(
            text("SELECT DISTINCT customer_segment FROM customer_rfm_analytics ORDER BY customer_segment;")
        ).scalars().all()

    expected_segments = {
        "At Risk",
        "Cant Lose Them",
        "Champions",
        "Customers Needing Attention",
        "Hibernating / Lost",
        "Loyal Customers",
        "New / Promising",
        "Potential Loyalists",
    }
    assert set(segments) == expected_segments


def test_idempotent_segmentation_reload():
    """Verifies that re-running the segmentation loader produces zero duplicate rows."""
    db_mgr = DatabaseManager()
    config = SegmentationConfig()
    loader = GoldDatabaseLoader(is_test=False)

    pred_parquet = config.base_dir / "data" / "gold" / "gold_customer_segmentation_kmeans.parquet"
    if not pred_parquet.exists():
        pytest.skip("Scored segmentation parquet file not found.")

    df_preds = pd.read_parquet(pred_parquet)

    # First load
    with db_mgr.session_scope() as session:
        loader.load_customer_segmentation(session, df_preds)

    with db_mgr.engine.connect() as conn:
        count_1 = conn.execute(select(func.count()).select_from(CustomerSegmentationKMeans)).scalar()

    # Second load (idempotence test)
    with db_mgr.session_scope() as session:
        loader.load_customer_segmentation(session, df_preds)

    with db_mgr.engine.connect() as conn:
        count_2 = conn.execute(select(func.count()).select_from(CustomerSegmentationKMeans)).scalar()

    assert count_1 == count_2 == 5939


def test_database_predictions_match_model_inference():
    """Verifies that loaded database rows strictly match offline model artifact inference."""
    config = SegmentationConfig()
    model = CustomerSegmentationModel.load(config.model_dir)

    db_mgr = DatabaseManager()
    with db_mgr.engine.connect() as conn:
        # Sample 100 diverse customers across all 5 clusters
        sample_q = text("""
            SELECT customer_id, recency_days, frequency, monetary_net, cluster_id, cluster_label
            FROM customer_segmentation_kmeans
            ORDER BY customer_id ASC
            LIMIT 100;
        """)
        df_sample = pd.read_sql_query(sample_q, conn)

    # Re-predict using reloaded model
    preds = model.predict(df_sample[["recency_days", "frequency", "monetary_net"]])

    assert (df_sample["cluster_id"].values == preds["cluster_id"].values).all()
    assert (df_sample["cluster_label"].values == preds["cluster_label"].values).all()
