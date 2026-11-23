"""CLI Verification Script checking PostgreSQL table counts, revenue reconciliation, and precision."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.reconciliation import PostgresReconciliationAuditor


def main() -> int:
    print("=" * 80)
    print("   E-COMMERCE INTELLIGENCE PLATFORM - POSTGRESQL DATA RECONCILIATION")
    print("=" * 80)

    auditor = PostgresReconciliationAuditor(is_test=False)
    report = auditor.audit()

    print(f"\nTarget Database:              {report.target_database}")
    print(f"All Checks Passed:            {report.all_checks_passed}")
    print(f"Monetary Precision (NUMERIC): {report.monetary_precision_verified}")
    print(f"Idempotent Reruns Verified:   {report.idempotent_rerun_verified}")

    print("\n" + "-" * 80)
    print(" TABLE-LEVEL ROW & REVENUE PARITY (GOLD PARQUET vs POSTGRESQL)")
    print("-" * 80)
    print(
        f"{'Table Name':<32} | {'Parquet':<8} | {'Postgres':<8} | {'Count Parity':<12} | "
        f"{'Parquet Net':<14} | {'Postgres Net':<14}"
    )
    print("-" * 80)

    for tbl, chk in report.table_checks.items():
        pq_rev_str = f"£{chk.parquet_net_revenue:,.2f}" if chk.parquet_net_revenue is not None else "N/A"
        pg_rev_str = f"£{chk.postgres_net_revenue:,.2f}" if chk.postgres_net_revenue is not None else "N/A"
        parity_str = "[PASS]" if chk.is_count_matched else "[FAIL]"
        print(
            f"{tbl:<32} | {chk.parquet_rows:<8,} | {chk.postgres_rows:<8,} | {parity_str:<12} | "
            f"{pq_rev_str:<14} | {pg_rev_str:<14}"
        )

    print("-" * 80)
    print("\n--- Cross-Layer Financial Reconciliation (Silver vs PostgreSQL) ---")
    print(f"  - Silver Reference Net Revenue:  £{report.silver_reference_net_revenue:14,.2f}")
    print(f"  - PostgreSQL Monthly Net Rev:    £{report.postgres_monthly_net_revenue:14,.2f}")
    print(f"  - PostgreSQL Country Net Rev:    £{report.postgres_country_net_revenue:14,.2f}")
    print(f"  - Cross-Layer Balanced:          {report.is_silver_gold_postgres_reconciled}")

    if report.all_checks_passed:
        print("\n[SUCCESS] 100% data parity and financial reconciliation achieved in PostgreSQL!")
        print("=" * 80)
        return 0
    else:
        print("\n[ERROR] Reconciliation failed with discrepancies:")
        for d in report.discrepancies:
            print(f"  - {d}")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
