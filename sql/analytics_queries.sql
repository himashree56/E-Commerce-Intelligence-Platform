-- ==============================================================================
-- E-Commerce Intelligence Platform - Advanced SQL Analytics
-- Phase 3 PostgreSQL Analytical Query Suite
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Monthly Revenue Trends
-- Business Definition: High-level temporal trajectory of gross sales, returns,
-- net revenue, and transaction counts across each calendar month.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 2. Month-over-Month (MoM) Revenue Growth
-- Business Definition: Measures absolute GBP and percentage growth rate of net
-- revenue relative to the preceding calendar month using the LAG() window function.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 3. Average Order Value (AOV)
-- Business Definition: Tracks the mean monetary transaction value per valid order
-- month-by-month, distinguishing gross basket size from net realized order value.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 4. Top 20 Products by Revenue
-- Business Definition: Ranks catalog merchandise by total commercial net revenue,
-- surfacing sales volumes, units returned, and return rates.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 5. Revenue by Country
-- Business Definition: Evaluates geographic performance, customer density,
-- cancellation ratios, and running cumulative global market share.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 6. Customer Lifetime Spending Distribution (Deciles)
-- Business Definition: Segments the customer population into 10 equal deciles via
-- NTILE(10) to quantify spending distribution and revenue contribution across value tiers.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 7. RFM Customer Segment Distribution
-- Business Definition: Summarizes customer counts, average recency days,
-- average order frequency, and total monetary net yield per behavioral segment.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 8. Repeat-Purchase Frequency Distribution
-- Business Definition: Binned breakdown of customers based strictly on their total
-- lifetime order count to distinguish single-order churn from repeat engagement.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 9. Customer Cohort Retention Analysis
-- Business Definition: Measures monthly customer retention rate by tracking cohorts
-- grouped strictly by their confirmed first purchase month across subsequent months.
-- ------------------------------------------------------------------------------
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
WHERE a.cohort_month = '2009-12'
ORDER BY a.cohort_month ASC, a.activity_month ASC;

-- ------------------------------------------------------------------------------
-- 10. Revenue Concentration Among High-Value Customers (Pareto 80/20 Analysis)
-- Business Definition: Computes the proportion of net revenue generated by the top 1%, 5%, 10%,
-- and 20% customer brackets across three documented financial denominators:
-- (1) positive customer spenders (£16.38M),
-- (2) all 5,939 identified customers (£16.29M, accounting for -£92.6K returners),
-- (3) total commercial company revenue (£19.01M, including £2.72M unattributed guest checkouts).
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 11. Product Return Rates
-- Business Definition: Surfaces products with statistically significant volume
-- (>= 50 units sold) experiencing the highest return percentages and refund amounts.
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 12. Cancellation and Refund Trends
-- Business Definition: Examines monthly cancellation rates and return-to-gross
-- ratios to monitor operational fulfillment quality and customer returns over time.
-- ------------------------------------------------------------------------------
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
