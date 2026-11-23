"""Advanced SQL Analytics Queries and Execution Engine."""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.database.connection import DatabaseManager

logger = logging.getLogger("ecommerce_platform.database.queries")


@dataclass
class AnalyticsQueryDefinition:
    """Metadata and SQL query definition for analytical reporting."""

    query_id: str
    name: str
    business_definition: str
    sql: str


# -----------------------------------------------------------------------------
# 12 ADVANCED SQL ANALYTICAL QUERIES
# -----------------------------------------------------------------------------

QUERY_MONTHLY_REVENUE_TRENDS = AnalyticsQueryDefinition(
    query_id="monthly_revenue_trends",
    name="Monthly Revenue Trends",
    business_definition=(
        "Evaluates high-level temporal trajectory of gross sales, returns, net revenue, "
        "and order counts across each calendar month in the dataset."
    ),
    sql="""
    SELECT
        invoice_year_month,
        invoice_year,
        invoice_month,
        gross_sales,
        return_amount,
        net_revenue,
        valid_orders_count,
        cancelled_orders_count,
        distinct_customers
    FROM monthly_sales_analytics
    ORDER BY invoice_year_month ASC;
    """,
)

QUERY_MOM_REVENUE_GROWTH = AnalyticsQueryDefinition(
    query_id="mom_revenue_growth",
    name="Month-over-Month Revenue Growth",
    business_definition=(
        "Measures the absolute GBP and percentage rate of net revenue growth relative "
        "to the preceding calendar month using the LAG() window function."
    ),
    sql="""
    WITH monthly_trend AS (
        SELECT
            invoice_year_month,
            net_revenue,
            LAG(net_revenue, 1) OVER (ORDER BY invoice_year_month ASC) AS prev_month_net_revenue
        FROM monthly_sales_analytics
    )
    SELECT
        invoice_year_month,
        net_revenue,
        prev_month_net_revenue,
        ROUND(net_revenue - prev_month_net_revenue, 2) AS mom_net_growth_gbp,
        ROUND(
            CASE
                WHEN prev_month_net_revenue IS NULL OR prev_month_net_revenue = 0 THEN NULL
                ELSE ((net_revenue - prev_month_net_revenue) / prev_month_net_revenue) * 100
            END, 2
        ) AS mom_growth_pct
    FROM monthly_trend
    ORDER BY invoice_year_month ASC;
    """,
)

QUERY_AVERAGE_ORDER_VALUE = AnalyticsQueryDefinition(
    query_id="average_order_value",
    name="Average Order Value (AOV)",
    business_definition=(
        "Tracks the mean monetary transaction value per valid order month-by-month, "
        "distinguishing gross basket size from net realized order value."
    ),
    sql="""
    SELECT
        invoice_year_month,
        valid_orders_count,
        gross_sales,
        net_revenue,
        average_order_value,
        ROUND(
            CASE WHEN valid_orders_count > 0 THEN gross_sales / valid_orders_count ELSE 0 END, 2
        ) AS aov_gross_gbp,
        ROUND(
            CASE WHEN valid_orders_count > 0 THEN net_revenue / valid_orders_count ELSE 0 END, 2
        ) AS aov_net_gbp
    FROM monthly_sales_analytics
    ORDER BY invoice_year_month ASC;
    """,
)

QUERY_TOP_20_PRODUCTS = AnalyticsQueryDefinition(
    query_id="top_20_products",
    name="Top 20 Products by Revenue",
    business_definition=(
        "Ranks catalog merchandise by total commercial net revenue, surfacing sales "
        "volumes, units returned, and return rates."
    ),
    sql="""
    SELECT
        stock_code,
        description,
        units_sold,
        units_returned,
        net_units,
        gross_sales,
        return_amount,
        net_revenue,
        return_rate_pct,
        DENSE_RANK() OVER (ORDER BY net_revenue DESC) AS revenue_rank
    FROM product_performance_analytics
    ORDER BY net_revenue DESC
    LIMIT 20;
    """,
)

QUERY_REVENUE_BY_COUNTRY = AnalyticsQueryDefinition(
    query_id="revenue_by_country",
    name="Revenue by Country",
    business_definition=(
        "Evaluates geographic performance, customer density, cancellation ratios, "
        "and running cumulative global market share."
    ),
    sql="""
    SELECT
        country,
        valid_orders_count,
        cancelled_orders_count,
        distinct_customers,
        gross_sales,
        return_amount,
        net_revenue,
        market_share_pct,
        ROUND(
            (SUM(net_revenue) OVER (ORDER BY net_revenue DESC) / SUM(net_revenue) OVER ()) * 100, 2
        ) AS cumulative_market_share_pct
    FROM country_revenue_analytics
    ORDER BY net_revenue DESC;
    """,
)

QUERY_CUSTOMER_LIFETIME_SPENDING = AnalyticsQueryDefinition(
    query_id="customer_lifetime_spending",
    name="Customer Lifetime Spending Distribution (Deciles)",
    business_definition=(
        "Segments the customer population into 10 equal deciles via NTILE(10) to quantify "
        "spending distribution and revenue contribution across value tiers."
    ),
    sql="""
    WITH customer_deciles AS (
        SELECT
            customer_id,
            primary_country,
            frequency,
            monetary_net,
            customer_segment,
            NTILE(10) OVER (ORDER BY monetary_net DESC) AS spend_decile
        FROM customer_rfm_analytics
    )
    SELECT
        spend_decile,
        COUNT(*) AS customer_count,
        ROUND(MIN(monetary_net), 2) AS min_spend_gbp,
        ROUND(MAX(monetary_net), 2) AS max_spend_gbp,
        ROUND(AVG(monetary_net), 2) AS avg_spend_gbp,
        ROUND(SUM(monetary_net), 2) AS total_decile_spend_gbp,
        ROUND(
            (SUM(monetary_net) / (SELECT SUM(monetary_net) FROM customer_rfm_analytics)) * 100, 2
        ) AS decile_revenue_share_pct
    FROM customer_deciles
    GROUP BY spend_decile
    ORDER BY spend_decile ASC;
    """,
)

QUERY_RFM_SEGMENT_DISTRIBUTION = AnalyticsQueryDefinition(
    query_id="rfm_segment_distribution",
    name="RFM Customer Segment Distribution",
    business_definition=(
        "Summarizes customer counts, average recency days, average order frequency, "
        "and total monetary net yield per behavioral segment."
    ),
    sql="""
    SELECT
        customer_segment,
        COUNT(*) AS customer_count,
        ROUND((COUNT(*)::NUMERIC / (SELECT COUNT(*) FROM customer_rfm_analytics)) * 100, 2) AS customer_share_pct,
        ROUND(AVG(recency_days), 1) AS avg_recency_days,
        ROUND(AVG(frequency), 1) AS avg_frequency,
        ROUND(SUM(monetary_net), 2) AS total_segment_revenue_gbp,
        ROUND(AVG(monetary_net), 2) AS avg_customer_value_gbp,
        ROUND(
            (SUM(monetary_net) / (SELECT SUM(monetary_net) FROM customer_rfm_analytics)) * 100, 2
        ) AS segment_revenue_share_pct
    FROM customer_rfm_analytics
    GROUP BY customer_segment
    ORDER BY total_segment_revenue_gbp DESC;
    """,
)

QUERY_REPEAT_PURCHASE_FREQUENCY = AnalyticsQueryDefinition(
    query_id="repeat_purchase_frequency",
    name="Repeat-Purchase Frequency Distribution",
    business_definition=(
        "Binned breakdown of customers based strictly on their total lifetime order count "
        "to distinguish single-order churn from repeat engagement."
    ),
    sql="""
    WITH frequency_brackets AS (
        SELECT
            customer_id,
            frequency,
            monetary_net,
            CASE
                WHEN frequency = 1 THEN '1 Order (One-Time Buyer)'
                WHEN frequency = 2 THEN '2 Orders'
                WHEN frequency BETWEEN 3 AND 5 THEN '3-5 Orders'
                WHEN frequency BETWEEN 6 AND 10 THEN '6-10 Orders'
                ELSE '11+ Orders (Loyal / Wholesale)'
            END AS frequency_tier
        FROM customer_rfm_analytics
    )
    SELECT
        frequency_tier,
        COUNT(*) AS customer_count,
        ROUND((COUNT(*)::NUMERIC / (SELECT COUNT(*) FROM customer_rfm_analytics)) * 100, 2) AS customer_pct,
        ROUND(SUM(monetary_net), 2) AS total_tier_revenue_gbp,
        ROUND(
            (SUM(monetary_net) / (SELECT SUM(monetary_net) FROM customer_rfm_analytics)) * 100, 2
        ) AS revenue_pct,
        ROUND(AVG(monetary_net), 2) AS avg_customer_spend_gbp
    FROM frequency_brackets
    GROUP BY frequency_tier
    ORDER BY MIN(frequency) ASC;
    """,
)

QUERY_CUSTOMER_COHORT_RETENTION = AnalyticsQueryDefinition(
    query_id="customer_cohort_retention",
    name="Customer Cohort Retention Analysis",
    business_definition=(
        "Measures monthly customer retention rate by tracking cohorts grouped strictly "
        "by their confirmed first purchase month across subsequent months."
    ),
    sql="""
    WITH customer_cohorts AS (
        SELECT
            customer_id,
            TO_CHAR(first_purchase_date, 'YYYY-MM') AS cohort_month
        FROM customer_rfm_analytics
    ),
    cohort_sizes AS (
        SELECT
            cohort_month,
            COUNT(*) AS cohort_size
        FROM customer_cohorts
        GROUP BY cohort_month
    ),
    monthly_activity AS (
        SELECT
            c.cohort_month,
            TO_CHAR(p.invoice_date, 'YYYY-MM') AS activity_month,
            COUNT(DISTINCT p.customer_id) AS active_customers,
            ROUND(SUM(p.order_net_revenue), 2) AS cohort_monthly_net_revenue
        FROM customer_purchase_history p
        JOIN customer_cohorts c ON p.customer_id = c.customer_id
        WHERE p.transaction_status = 'VALID_PURCHASE'
        GROUP BY c.cohort_month, TO_CHAR(p.invoice_date, 'YYYY-MM')
    )
    SELECT
        a.cohort_month,
        s.cohort_size,
        a.activity_month,
        a.active_customers,
        ROUND((a.active_customers::NUMERIC / s.cohort_size) * 100, 2) AS retention_rate_pct,
        a.cohort_monthly_net_revenue
    FROM monthly_activity a
    JOIN cohort_sizes s ON a.cohort_month = s.cohort_month
    WHERE a.cohort_month = '2009-12'  -- Focus on the foundational founding cohort
    ORDER BY a.cohort_month ASC, a.activity_month ASC;
    """,
)

QUERY_REVENUE_CONCENTRATION_PARETO = AnalyticsQueryDefinition(
    query_id="revenue_concentration_pareto",
    name="Revenue Concentration Among High-Value Customers (Pareto)",
    business_definition=(
        "Computes the proportion of net revenue generated by the top 1%, 5%, 10%, and 20% "
        "customer brackets across three documented financial denominators: (1) positive customer "
        "spenders (£16.38M), (2) all 5,939 identified customers (£16.29M, accounting for -£92.6K returns), "
        "and (3) total commercial company revenue (£19.01M, including £2.72M unattributed guest checkouts)."
    ),
    sql="""
    WITH ranked_customers AS (
        SELECT
            customer_id,
            monetary_net,
            ROW_NUMBER() OVER (ORDER BY monetary_net DESC) AS customer_rank,
            COUNT(*) OVER () AS total_positive_customers,
            SUM(monetary_net) OVER () AS total_positive_revenue
        FROM customer_rfm_analytics
        WHERE monetary_net > 0
    ),
    benchmarks AS (
        SELECT
            (SELECT COUNT(*) FROM customer_rfm_analytics) AS all_customers_count,
            (SELECT SUM(monetary_net) FROM customer_rfm_analytics) AS all_customer_revenue,
            (SELECT SUM(net_revenue) FROM monthly_sales_analytics) AS total_commercial_revenue
    )
    SELECT
        customer_bracket,
        customer_count,
        ROUND((customer_count::NUMERIC / total_positive_customers) * 100, 2) AS customer_pct,
        ROUND(bracket_revenue, 2) AS bracket_revenue_gbp,
        ROUND((bracket_revenue / total_positive_revenue) * 100, 2) AS revenue_pct,
        ROUND((bracket_revenue / b.all_customer_revenue) * 100, 2) AS share_of_all_customer_net_pct,
        ROUND((bracket_revenue / b.total_commercial_revenue) * 100, 2) AS share_of_commercial_total_pct
    FROM (
        SELECT
            'Top 1% Customers' AS customer_bracket,
            COUNT(*) AS customer_count,
            MAX(total_positive_customers) AS total_positive_customers,
            SUM(monetary_net) AS bracket_revenue,
            MAX(total_positive_revenue) AS total_positive_revenue
        FROM ranked_customers
        WHERE customer_rank <= CEIL(total_positive_customers * 0.01)
        UNION ALL
        SELECT
            'Top 5% Customers',
            COUNT(*),
            MAX(total_positive_customers),
            SUM(monetary_net),
            MAX(total_positive_revenue)
        FROM ranked_customers
        WHERE customer_rank <= CEIL(total_positive_customers * 0.05)
        UNION ALL
        SELECT
            'Top 10% Customers',
            COUNT(*),
            MAX(total_positive_customers),
            SUM(monetary_net),
            MAX(total_positive_revenue)
        FROM ranked_customers
        WHERE customer_rank <= CEIL(total_positive_customers * 0.10)
        UNION ALL
        SELECT
            'Top 20% Customers (Pareto)',
            COUNT(*),
            MAX(total_positive_customers),
            SUM(monetary_net),
            MAX(total_positive_revenue)
        FROM ranked_customers
        WHERE customer_rank <= CEIL(total_positive_customers * 0.20)
    ) tiers
    CROSS JOIN benchmarks b;
    """,
)

QUERY_PRODUCT_RETURN_RATES = AnalyticsQueryDefinition(
    query_id="product_return_rates",
    name="Product Return Rates",
    business_definition=(
        "Surfaces products with statistically significant volume (>= 50 units sold) "
        "experiencing the highest return percentages and financial refund amounts."
    ),
    sql="""
    SELECT
        stock_code,
        description,
        units_sold,
        units_returned,
        gross_sales,
        return_amount,
        return_rate_pct,
        DENSE_RANK() OVER (ORDER BY return_rate_pct DESC, return_amount DESC) AS return_severity_rank
    FROM product_performance_analytics
    WHERE units_sold >= 50
    ORDER BY return_rate_pct DESC, return_amount DESC
    LIMIT 20;
    """,
)

QUERY_CANCELLATION_TRENDS = AnalyticsQueryDefinition(
    query_id="cancellation_trends",
    name="Cancellation and Refund Trends",
    business_definition=(
        "Examines monthly cancellation rates and return-to-gross ratios to monitor "
        "operational and fulfillment health over time."
    ),
    sql="""
    SELECT
        invoice_year_month,
        valid_orders_count,
        cancelled_orders_count,
        (valid_orders_count + cancelled_orders_count) AS total_order_events,
        ROUND(
            (cancelled_orders_count::NUMERIC / NULLIF(valid_orders_count + cancelled_orders_count, 0)) * 100, 2
        ) AS cancellation_rate_pct,
        gross_sales,
        return_amount,
        ROUND(
            (return_amount / NULLIF(gross_sales, 0)) * 100, 2
        ) AS return_to_gross_pct
    FROM monthly_sales_analytics
    ORDER BY invoice_year_month ASC;
    """,
)

ALL_ANALYTICS_QUERIES: List[AnalyticsQueryDefinition] = [
    QUERY_MONTHLY_REVENUE_TRENDS,
    QUERY_MOM_REVENUE_GROWTH,
    QUERY_AVERAGE_ORDER_VALUE,
    QUERY_TOP_20_PRODUCTS,
    QUERY_REVENUE_BY_COUNTRY,
    QUERY_CUSTOMER_LIFETIME_SPENDING,
    QUERY_RFM_SEGMENT_DISTRIBUTION,
    QUERY_REPEAT_PURCHASE_FREQUENCY,
    QUERY_CUSTOMER_COHORT_RETENTION,
    QUERY_REVENUE_CONCENTRATION_PARETO,
    QUERY_PRODUCT_RETURN_RATES,
    QUERY_CANCELLATION_TRENDS,
]


QUERY_MAP: Dict[str, AnalyticsQueryDefinition] = {
    q.query_id: q for q in ALL_ANALYTICS_QUERIES
}


class AnalyticsQueryRunner:
    """Executes SQL analytics queries and formats results as Pandas DataFrames."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db_manager = db_manager or DatabaseManager()

    def run_query(self, query: Union[AnalyticsQueryDefinition, str]) -> pd.DataFrame:
        """Executes a single query definition or query_id and returns the result as a DataFrame."""
        if isinstance(query, str):
            query_def = QUERY_MAP.get(query)
            if not query_def:
                raise ValueError(
                    f"Unknown query ID '{query}'. Available queries: {list(QUERY_MAP.keys())}"
                )
        else:
            query_def = query

        logger.info(f"Executing analytical query: {query_def.name} ({query_def.query_id})...")
        with self.db_manager.engine.connect() as conn:
            df = pd.read_sql_query(text(query_def.sql), conn)
        return df

    def run_all_queries(self) -> Dict[str, pd.DataFrame]:
        """Executes all 12 analytical queries and returns a dictionary of DataFrames."""
        results: Dict[str, pd.DataFrame] = {}
        for q in ALL_ANALYTICS_QUERIES:
            results[q.query_id] = self.run_query(q)
        return results
