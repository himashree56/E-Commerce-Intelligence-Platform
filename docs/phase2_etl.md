# Phase 2: Silver & Gold Medallion Lakehouse Layer

## 1. Executive Summary
Phase 2 promotes the raw-standardized Bronze dataset (1,067,371 rows) into a curated, auditable **Silver Layer** and builds 5 business-critical **Gold Analytical Tables**. Every Bronze transaction is accounted for without silent deletion.

---

## 2. Silver Layer Architecture & Quality Rules

### 100% Record Preservation Policy
No records are deleted from Bronze to Silver. Instead, records are tagged with explicit governance flags:
- `is_duplicate`: Identifies subsequent duplicate records across primary business keys (`invoice_no`, `stock_code`, `quantity`, `invoice_date`, `unit_price`, `customer_id`).
- `is_cancellation`: Identifies transactions where `invoice_no` starts with 'C'.
- `is_return`: Identifies transactions with negative quantities (`quantity < 0`).
- `is_invalid_price`: Identifies transactions with non-positive unit price (`unit_price <= 0`).
- `is_missing_customer`: Identifies unregistered transactions where `customer_id` is null.
- `is_non_product`: Identifies administrative / shipping codes (`POST`, `D`, `M`, `BANK CHARGES`, etc.).

### Categorical Transaction Status Hierarchy
Every record is assigned a single, mutually exclusive transaction status:

| Status | Definition | Row Count | % of Silver |
| :--- | :--- | :--- | :--- |
| `VALID_PURCHASE` | $quantity > 0 \land unit\_price > 0 \land \neg duplicate \land \neg cancellation$ | **1,007,911** | 94.43% |
| `DUPLICATE` | Subsequent duplicate across business keys | **34,337** | 3.22% |
| `CANCELLATION` | `invoice_no` starts with 'C' (credit notes/cancellations) | **19,104** | 1.79% |
| `INVENTORY_WRITE_OFF` | $quantity < 0 \land unit\_price \le 0$ (damaged, lost, audit adjustments) | **3,393** | 0.32% |
| `ZERO_PRICE_ENTRY` | $quantity \ge 0 \land unit\_price \le 0$ (samples, giveaways, gifts) | **2,626** | 0.25% |
| **Total** | **Exhaustive Sum of All Statuses** | **1,067,371** | **100.00%** |

---

## 3. Financial Metrics & Net Revenue Calculation

To avoid counting cancellation and return rows as ordinary positive purchases, financial calculations are strictly separated:

1. **Line Total** ($LT$):
   $$LT = \text{quantity} \times \text{unit\_price}$$
2. **Gross Sales** ($GS$):
   $$GS = \begin{cases} LT & \text{if } \text{transaction\_status} = \text{'VALID\_PURCHASE'} \\ 0.0 & \text{otherwise} \end{cases}$$
3. **Return Amount** ($RA$):
   $$RA = \begin{cases} |\text{quantity}| \times \text{unit\_price} & \text{if } \text{transaction\_status} \in \{\text{'CANCELLATION'}, \text{'RETURN\_ADJUSTMENT'}\} \\ 0.0 & \text{otherwise} \end{cases}$$
4. **Documented Net Revenue** ($NR$):
   $$NR = GS - RA$$

### Financial Reconciliation Totals
- **Gross Sales (Purchases)**: **£20,475,708.60**
- **Returns & Cancellations**: **-£1,462,797.75**
- **Net Revenue**: **£19,012,910.85**
- **Discrepancy**: **£0.00** ($20,475,708.60 - 1,462,797.75 = 19,012,910.85$)

---

## 4. Gold Analytical Tables

1. **Monthly Sales Table** (`data/gold/gold_monthly_sales.parquet`, 25 rows):
   - Aggregates sales, returns, order volume, customer counts, and average order value ($AOV$) by month from 2009-12 to 2011-12.
2. **Customer RFM Table** (`data/gold/gold_customer_rfm.parquet`, 5,939 rows):
   - Calculates Recency (days since last purchase), Frequency (distinct purchase invoices), and Monetary value (net revenue).
   - Generates RFM quintile scores (1-5) and assigns segments (`Champions`, `Loyal Customers`, `At Risk`, `Lost`, etc.).
3. **Product Performance Table** (`data/gold/gold_product_performance.parquet`, 4,920 rows):
   - Catalogs net revenue, gross sales, units sold, units returned, and return rate percentage per SKU.
4. **Country Revenue Table** (`data/gold/gold_country_revenue.parquet`, 43 rows):
   - Summarizes geographic distribution, order counts, and market share percentages.
5. **Customer Purchase History Table** (`data/gold/gold_customer_purchase_history.parquet`, 44,870 rows):
   - Order-level transaction history timeline per customer.

---

## 5. Databricks & Delta Lake Architecture

- **Unity Catalog DDL**: Exported in `src/databricks/delta_lake.py` and `src/databricks/online_retail_lakehouse_pipeline.py`.
- **Change Data Feed**: Enabled for downstream change capture.
- **Z-Ordering**: Optimized by `(customer_id, invoice_date)` for fast point queries.
- **Verification Disclaimer**: Local pipeline executed and verified with Parquet and Delta-compatible PySpark code. Live Databricks cloud execution was not tested against a remote Databricks cluster.
