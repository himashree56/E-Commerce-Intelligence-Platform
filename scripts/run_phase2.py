"""CLI entrypoint for Phase 2 ETL Pipeline (Bronze -> Silver -> Gold)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.config import ETLConfig
from src.etl.pipeline import Phase2Pipeline


def main() -> int:
    print("=" * 75)
    print("   E-COMMERCE INTELLIGENCE PLATFORM - PHASE 2 ETL PIPELINE")
    print("=" * 75)

    config = ETLConfig(base_dir=PROJECT_ROOT)
    pipeline = Phase2Pipeline(config=config)

    try:
        summary = pipeline.run()

        print("\n" + "=" * 75)
        print("                 PHASE 2 EXECUTION SUMMARY")
        print("=" * 75)
        print(f"Run ID:                      {summary.run_id}")
        print(f"Status:                      {summary.status}")
        print(f"Duration:                    {summary.duration_seconds}s")
        print(f"Bronze Records Ingested:     {summary.bronze_records_ingested:,}")
        print(f"Silver Records Produced:     {summary.silver_records_produced:,}")
        print(f"Reconciliation Passed:       {summary.reconciliation_passed}")

        print("\n--- Silver Layer Status Breakdown ---")
        for status, count in summary.silver_summary["status_breakdown"].items() if "status_breakdown" in summary.silver_summary else summary.reconciliation_summary["status_breakdown"].items():
            print(f"  - {status:25}: {count:8,}")

        print("\n--- Revenue Totals (GBP) ---")
        print(f"  - Gross Sales (Purchases):   £{summary.gross_sales_gbp:14,.2f}")
        print(f"  - Returns & Cancellations:  -£{summary.return_amount_gbp:14,.2f}")
        print(f"  - Documented Net Revenue:    £{summary.net_revenue_gbp:14,.2f}")

        print("\n--- Gold Analytical Tables ---")
        print(f"  - Monthly Sales:             {summary.gold_monthly_rows:6,} rows")
        print(f"  - Customer RFM:              {summary.gold_rfm_rows:6,} rows")
        print(f"  - Product Performance:       {summary.gold_product_rows:6,} rows")
        print(f"  - Country Revenue:           {summary.gold_country_rows:6,} rows")
        print(f"  - Customer Order History:    {summary.gold_history_rows:6,} rows")

        print("=" * 75)
        return 0 if summary.reconciliation_passed else 1

    except Exception as e:
        print(f"[ERROR] Phase 2 failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
