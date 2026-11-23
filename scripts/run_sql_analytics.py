"""CLI runner executing all 12 Advanced SQL Analytics queries against PostgreSQL."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.queries import ALL_ANALYTICS_QUERIES, AnalyticsQueryRunner


def main() -> int:
    print("=" * 80)
    print("     E-COMMERCE INTELLIGENCE PLATFORM - ADVANCED SQL ANALYTICS SUITE")
    print("=" * 80)

    runner = AnalyticsQueryRunner()
    success_count = 0

    for i, q in enumerate(ALL_ANALYTICS_QUERIES, 1):
        print(f"\n[{i:02d}/12] {q.name.upper()} ({q.query_id})")
        print(f"  Definition: {q.business_definition}")
        print("-" * 80)

        try:
            df = runner.run_query(q)
            success_count += 1
            print(f"  Rows returned: {len(df):,}")
            # Display sample rows
            display_rows = df.head(5) if len(df) > 5 else df
            print(display_rows.to_string(index=False))
            if len(df) > 5:
                print(f"  ... [and {len(df) - 5:,} more rows]")
        except Exception as e:
            print(f"  [ERROR] Query execution failed: {e}")

    print("\n" + "=" * 80)
    print(f"ANALYTICS EXECUTION SUMMARY: {success_count}/{len(ALL_ANALYTICS_QUERIES)} queries succeeded.")
    print("=" * 80)
    return 0 if success_count == len(ALL_ANALYTICS_QUERIES) else 1


if __name__ == "__main__":
    sys.exit(main())
