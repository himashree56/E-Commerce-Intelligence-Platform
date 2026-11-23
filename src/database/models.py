"""SQLAlchemy 2.0 Declarative Models for PostgreSQL Lakehouse Serving Layer."""
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all PostgreSQL tables."""

    pass


class MonthlySalesAnalytics(Base):
    """Gold aggregate: Monthly sales revenue, returns, and order statistics."""

    __tablename__ = "monthly_sales_analytics"

    invoice_year_month: Mapped[str] = mapped_column(String(7), primary_key=True)
    invoice_year: Mapped[int] = mapped_column(Integer, nullable=False)
    invoice_month: Mapped[int] = mapped_column(Integer, nullable=False)

    gross_sales: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    return_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    net_revenue: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)

    valid_orders_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    cancelled_orders_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    distinct_customers: Mapped[int] = mapped_column(BigInteger, nullable=False)

    total_units_sold: Mapped[int] = mapped_column(BigInteger, nullable=False)
    units_returned: Mapped[int] = mapped_column(BigInteger, nullable=False)
    net_units: Mapped[int] = mapped_column(BigInteger, nullable=False)

    average_order_value: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_monthly_sales_year_month", "invoice_year_month"),
        Index("ix_monthly_sales_year", "invoice_year"),
    )


class CustomerRFMAnalytics(Base):
    """Gold aggregate: Customer-level RFM scoring and segmentation."""

    __tablename__ = "customer_rfm_analytics"

    customer_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    primary_country: Mapped[str] = mapped_column(String(100), nullable=False)

    first_purchase_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_purchase_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    frequency: Mapped[int] = mapped_column(BigInteger, nullable=False)
    monetary_gross: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    monetary_net: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    return_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    return_orders_count: Mapped[int] = mapped_column(BigInteger, nullable=False)

    recency_days: Mapped[int] = mapped_column(Integer, nullable=False)
    r_score: Mapped[int] = mapped_column(Integer, nullable=False)
    f_score: Mapped[int] = mapped_column(Integer, nullable=False)
    m_score: Mapped[int] = mapped_column(Integer, nullable=False)
    rfm_cell: Mapped[str] = mapped_column(String(10), nullable=False)
    customer_segment: Mapped[str] = mapped_column(String(50), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    purchase_history: Mapped[list["CustomerPurchaseHistory"]] = relationship(
        "CustomerPurchaseHistory", back_populates="customer", cascade="all, delete-orphan"
    )
    ml_segmentation: Mapped[Optional["CustomerSegmentationKMeans"]] = relationship(
        "CustomerSegmentationKMeans", back_populates="customer", uselist=False, cascade="all, delete-orphan"
    )
    repeat_purchase_predictions: Mapped[list["CustomerRepeatPurchasePrediction"]] = relationship(
        "CustomerRepeatPurchasePrediction", back_populates="customer", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_rfm_segment", "customer_segment"),
        Index("ix_rfm_recency_days", "recency_days"),
        Index("ix_rfm_monetary_net", "monetary_net"),
        Index("ix_rfm_frequency", "frequency"),
    )


class ProductPerformanceAnalytics(Base):
    """Gold aggregate: Product sales volume, returns, and distinct order metrics."""

    __tablename__ = "product_performance_analytics"

    stock_code: Mapped[str] = mapped_column(String(50), primary_key=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    gross_sales: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    return_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    net_revenue: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)

    units_sold: Mapped[int] = mapped_column(BigInteger, nullable=False)
    units_returned: Mapped[int] = mapped_column(BigInteger, nullable=False)
    net_units: Mapped[int] = mapped_column(BigInteger, nullable=False)

    distinct_orders: Mapped[int] = mapped_column(BigInteger, nullable=False)
    distinct_customers: Mapped[int] = mapped_column(BigInteger, nullable=False)

    return_rate_pct: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_prod_net_revenue", "net_revenue"),
        Index("ix_prod_units_sold", "units_sold"),
        Index("ix_prod_return_rate", "return_rate_pct"),
    )


class CountryRevenueAnalytics(Base):
    """Gold aggregate: Geographic market share and order breakdowns."""

    __tablename__ = "country_revenue_analytics"

    country: Mapped[str] = mapped_column(String(100), primary_key=True)

    gross_sales: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    return_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    net_revenue: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)

    valid_orders_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    cancelled_orders_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    distinct_customers: Mapped[int] = mapped_column(BigInteger, nullable=False)

    market_share_pct: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_country_net_revenue", "net_revenue"),
    )


class CustomerPurchaseHistory(Base):
    """Gold aggregate: Order-level summary per customer with financial metrics."""

    __tablename__ = "customer_purchase_history"

    invoice_no: Mapped[str] = mapped_column(String(50), primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("customer_rfm_analytics.customer_id", ondelete="CASCADE"), nullable=False
    )

    invoice_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    is_cancellation: Mapped[bool] = mapped_column(Boolean, nullable=False)
    transaction_status: Mapped[str] = mapped_column(String(50), nullable=False)

    line_items_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    total_quantity: Mapped[int] = mapped_column(BigInteger, nullable=False)

    order_gross_sales: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    order_return_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    order_net_revenue: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    customer: Mapped[CustomerRFMAnalytics] = relationship(
        "CustomerRFMAnalytics", back_populates="purchase_history"
    )

    __table_args__ = (
        Index("ix_cph_customer_id", "customer_id"),
        Index("ix_cph_invoice_date", "invoice_date"),
        Index("ix_cph_country", "country"),
        Index("ix_cph_status", "transaction_status"),
    )


class ETLExecutionMetadata(Base):
    """Audit table capturing pipeline runs across Bronze, Silver, Gold, and Database loading."""

    __tablename__ = "etl_execution_metadata"

    run_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    pipeline_phase: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    duration_seconds: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    records_processed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    audit_results: Mapped[list["DataQualityAuditResults"]] = relationship(
        "DataQualityAuditResults", back_populates="execution_metadata", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_etl_meta_phase", "pipeline_phase"),
        Index("ix_etl_meta_created_at", "created_at"),
    )


class DataQualityAuditResults(Base):
    """Audit table capturing data quality metrics, cross-layer balances, and anomaly counts."""

    __tablename__ = "data_quality_audit_results"

    audit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(
        String(100), ForeignKey("etl_execution_metadata.run_id", ondelete="SET NULL"), nullable=True
    )

    audit_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    audit_type: Mapped[str] = mapped_column(String(50), nullable=False)
    is_balanced: Mapped[bool] = mapped_column(Boolean, nullable=False)

    gross_sales: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2), nullable=True)
    return_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2), nullable=True)
    net_revenue: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2), nullable=True)
    discrepancies_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report_payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    execution_metadata: Mapped[Optional[ETLExecutionMetadata]] = relationship(
        "ETLExecutionMetadata", back_populates="audit_results"
    )

    __table_args__ = (
        Index("ix_audit_run_id", "run_id"),
        Index("ix_audit_type", "audit_type"),
    )


class CustomerSegmentationKMeans(Base):
    """Machine learning customer segmentation generated via K-Means clustering (Phase 4)."""

    __tablename__ = "customer_segmentation_kmeans"

    customer_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("customer_rfm_analytics.customer_id", ondelete="CASCADE"), primary_key=True
    )
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    cluster_label: Mapped[str] = mapped_column(String(100), nullable=False)
    recency_days: Mapped[int] = mapped_column(Integer, nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, nullable=False)
    monetary_net: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    distance_to_centroid: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    customer: Mapped["CustomerRFMAnalytics"] = relationship(
        "CustomerRFMAnalytics", back_populates="ml_segmentation"
    )

    __table_args__ = (
        Index("ix_csk_cluster_id", "cluster_id"),
        Index("ix_csk_cluster_label", "cluster_label"),
        Index("ix_csk_model_version", "model_version"),
    )


class TransactionAnomaliesIsolationForest(Base):
    """Transaction anomaly detection scores generated via Isolation Forest (Phase 5)."""

    __tablename__ = "transaction_anomalies_isolation_forest"

    transaction_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_row_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    invoice_no: Mapped[str] = mapped_column(String(20), nullable=False)
    stock_code: Mapped[str] = mapped_column(String(30), nullable=False)
    invoice_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    transaction_status: Mapped[str] = mapped_column(String(30), nullable=False)

    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    line_total: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)

    raw_decision_score: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    anomaly_score: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    diagnostic_reasons: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    scoring_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_taif_is_anomaly", "is_anomaly"),
        Index("ix_taif_anomaly_score", "anomaly_score"),
        Index("ix_taif_invoice_no", "invoice_no"),
        Index("ix_taif_model_version", "model_version"),
        Index("ix_taif_source_row_id", "source_row_id"),
    )


class CustomerRepeatPurchasePrediction(Base):
    """Customer 30-day repeat-purchase prediction scores generated via XGBoost (Phase 6)."""

    __tablename__ = "customer_repeat_purchase_prediction"

    prediction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("customer_rfm_analytics.customer_id", ondelete="CASCADE"), nullable=False
    )
    prediction_as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    prediction_horizon_days: Mapped[int] = mapped_column(Integer, default=30, nullable=False)

    repeat_purchase_probability: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    predicted_label: Mapped[bool] = mapped_column(Boolean, nullable=False)

    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    feature_snapshot: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    scoring_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    customer: Mapped["CustomerRFMAnalytics"] = relationship(
        "CustomerRFMAnalytics", back_populates="repeat_purchase_predictions"
    )

    __table_args__ = (
        UniqueConstraint("customer_id", "prediction_as_of_date", "model_version", name="uq_crpp_cust_date_version"),
        Index("ix_crpp_customer_id", "customer_id"),
        Index("ix_crpp_as_of_date", "prediction_as_of_date"),
        Index("ix_crpp_predicted_label", "predicted_label"),
        Index("ix_crpp_prob", "repeat_purchase_probability"),
    )


