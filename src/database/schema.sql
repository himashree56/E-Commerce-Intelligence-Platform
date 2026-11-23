-- ==============================================================================
-- E-Commerce Intelligence Platform - PostgreSQL Schema DDL
-- Phase 3 Lakehouse Serving & Analytics Tables
-- ==============================================================================

-- 1. Monthly Sales Analytics
CREATE TABLE IF NOT EXISTS monthly_sales_analytics (
    invoice_year_month VARCHAR(7) PRIMARY KEY,
    invoice_year INTEGER NOT NULL,
    invoice_month INTEGER NOT NULL,
    gross_sales NUMERIC(14, 2) NOT NULL,
    return_amount NUMERIC(14, 2) NOT NULL,
    net_revenue NUMERIC(14, 2) NOT NULL,
    valid_orders_count BIGINT NOT NULL,
    cancelled_orders_count BIGINT NOT NULL,
    distinct_customers BIGINT NOT NULL,
    total_units_sold BIGINT NOT NULL,
    units_returned BIGINT NOT NULL,
    net_units BIGINT NOT NULL,
    average_order_value NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_monthly_sales_year_month ON monthly_sales_analytics(invoice_year_month);
CREATE INDEX IF NOT EXISTS ix_monthly_sales_year ON monthly_sales_analytics(invoice_year);

-- 2. Customer RFM Analytics
CREATE TABLE IF NOT EXISTS customer_rfm_analytics (
    customer_id BIGINT PRIMARY KEY,
    primary_country VARCHAR(100) NOT NULL,
    first_purchase_date TIMESTAMPTZ NOT NULL,
    last_purchase_date TIMESTAMPTZ NOT NULL,
    frequency BIGINT NOT NULL,
    monetary_gross NUMERIC(14, 2) NOT NULL,
    monetary_net NUMERIC(14, 2) NOT NULL,
    return_amount NUMERIC(14, 2) NOT NULL,
    return_orders_count BIGINT NOT NULL,
    recency_days INTEGER NOT NULL,
    r_score INTEGER NOT NULL,
    f_score INTEGER NOT NULL,
    m_score INTEGER NOT NULL,
    rfm_cell VARCHAR(10) NOT NULL,
    customer_segment VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_rfm_segment ON customer_rfm_analytics(customer_segment);
CREATE INDEX IF NOT EXISTS ix_rfm_recency_days ON customer_rfm_analytics(recency_days);
CREATE INDEX IF NOT EXISTS ix_rfm_monetary_net ON customer_rfm_analytics(monetary_net);
CREATE INDEX IF NOT EXISTS ix_rfm_frequency ON customer_rfm_analytics(frequency);

-- 3. Product Performance Analytics
CREATE TABLE IF NOT EXISTS product_performance_analytics (
    stock_code VARCHAR(50) PRIMARY KEY,
    description TEXT,
    gross_sales NUMERIC(14, 2) NOT NULL,
    return_amount NUMERIC(14, 2) NOT NULL,
    net_revenue NUMERIC(14, 2) NOT NULL,
    units_sold BIGINT NOT NULL,
    units_returned BIGINT NOT NULL,
    net_units BIGINT NOT NULL,
    distinct_orders BIGINT NOT NULL,
    distinct_customers BIGINT NOT NULL,
    return_rate_pct NUMERIC(6, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_prod_net_revenue ON product_performance_analytics(net_revenue);
CREATE INDEX IF NOT EXISTS ix_prod_units_sold ON product_performance_analytics(units_sold);
CREATE INDEX IF NOT EXISTS ix_prod_return_rate ON product_performance_analytics(return_rate_pct);

-- 4. Country Revenue Analytics
CREATE TABLE IF NOT EXISTS country_revenue_analytics (
    country VARCHAR(100) PRIMARY KEY,
    gross_sales NUMERIC(14, 2) NOT NULL,
    return_amount NUMERIC(14, 2) NOT NULL,
    net_revenue NUMERIC(14, 2) NOT NULL,
    valid_orders_count BIGINT NOT NULL,
    cancelled_orders_count BIGINT NOT NULL,
    distinct_customers BIGINT NOT NULL,
    market_share_pct NUMERIC(6, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_country_net_revenue ON country_revenue_analytics(net_revenue);

-- 5. Customer Purchase History
CREATE TABLE IF NOT EXISTS customer_purchase_history (
    invoice_no VARCHAR(50) PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customer_rfm_analytics(customer_id) ON DELETE CASCADE,
    invoice_date TIMESTAMPTZ NOT NULL,
    country VARCHAR(100) NOT NULL,
    is_cancellation BOOLEAN NOT NULL,
    transaction_status VARCHAR(50) NOT NULL,
    line_items_count BIGINT NOT NULL,
    total_quantity BIGINT NOT NULL,
    order_gross_sales NUMERIC(14, 2) NOT NULL,
    order_return_amount NUMERIC(14, 2) NOT NULL,
    order_net_revenue NUMERIC(14, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_cph_customer_id ON customer_purchase_history(customer_id);
CREATE INDEX IF NOT EXISTS ix_cph_invoice_date ON customer_purchase_history(invoice_date);
CREATE INDEX IF NOT EXISTS ix_cph_country ON customer_purchase_history(country);
CREATE INDEX IF NOT EXISTS ix_cph_status ON customer_purchase_history(transaction_status);

-- 6. ETL Execution Metadata
CREATE TABLE IF NOT EXISTS etl_execution_metadata (
    run_id VARCHAR(100) PRIMARY KEY,
    pipeline_phase VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    duration_seconds NUMERIC(10, 2) NOT NULL,
    records_processed BIGINT NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_etl_meta_phase ON etl_execution_metadata(pipeline_phase);
CREATE INDEX IF NOT EXISTS ix_etl_meta_created_at ON etl_execution_metadata(created_at);

-- 7. Data Quality Audit Results
CREATE TABLE IF NOT EXISTS data_quality_audit_results (
    audit_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES etl_execution_metadata(run_id) ON DELETE SET NULL,
    audit_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
    audit_type VARCHAR(50) NOT NULL,
    is_balanced BOOLEAN NOT NULL,
    gross_sales NUMERIC(14, 2),
    return_amount NUMERIC(14, 2),
    net_revenue NUMERIC(14, 2),
    discrepancies_count INTEGER DEFAULT 0 NOT NULL,
    report_payload JSONB
);

CREATE INDEX IF NOT EXISTS ix_audit_run_id ON data_quality_audit_results(run_id);
CREATE INDEX IF NOT EXISTS ix_audit_type ON data_quality_audit_results(audit_type);

-- 8. Customer Segmentation K-Means (Phase 4 Machine Learning)
CREATE TABLE IF NOT EXISTS customer_segmentation_kmeans (
    customer_id BIGINT PRIMARY KEY REFERENCES customer_rfm_analytics(customer_id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    cluster_label VARCHAR(100) NOT NULL,
    recency_days INTEGER NOT NULL,
    frequency INTEGER NOT NULL,
    monetary_net NUMERIC(14, 2) NOT NULL,
    distance_to_centroid NUMERIC(10, 4) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    metadata_payload JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_csk_cluster_id ON customer_segmentation_kmeans(cluster_id);
CREATE INDEX IF NOT EXISTS ix_csk_cluster_label ON customer_segmentation_kmeans(cluster_label);
CREATE INDEX IF NOT EXISTS ix_csk_model_version ON customer_segmentation_kmeans(model_version);
