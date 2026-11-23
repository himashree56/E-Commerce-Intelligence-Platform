"""CLI Script executing Gold Parquet to PostgreSQL database loading pipeline."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.loader import GoldDatabaseLoader
from src.database.schema_manager import SchemaManager


def main() -> int:
    print("=" * 75)
    print("   E-COMMERCE INTELLIGENCE PLATFORM - GOLD TO POSTGRESQL PIPELINE")
    print("=" * 75)

    # 1. Ensure all tables are created
    schema_mgr = SchemaManager()
    created_tables = schema_mgr.create_all_tables(is_test=False)
    print(f"[SCHEMA] Verified {len(created_tables)} tables in PostgreSQL database.")

    # 2. Run loading pipeline
    loader = GoldDatabaseLoader(is_test=False)
    report = loader.load_all_gold_tables()

    print("\n" + "=" * 75)
    print("                     DATABASE LOAD SUMMARY")
    print("=" * 75)
    print(f"Run ID:                  {report.run_id}")
    print(f"Target Database:         {report.target_database}")
    print(f"Status:                  {report.status}")
    print(f"Duration:                {report.total_duration_seconds}s")
    print(f"Total Records Loaded:    {report.total_records_loaded:,}")

    print("\n--- Table-Level Load Breakdown ---")
    for tbl, summary in report.table_summaries.items():
        print(
            f"  - {tbl:30}: Source={summary.source_rows:6,} | "
            f"Loaded={summary.loaded_rows:6,} | Time={summary.duration_seconds:6.3f}s | "
            f"Status={summary.status}"
        )

    if report.status == "SUCCESS":
        print("\n[SUCCESS] All Gold analytical datasets loaded into PostgreSQL successfully!")
        print("=" * 75)
        return 0
    else:
        print("\n[ERROR] Pipeline failed with discrepancies:")
        for d in report.discrepancies:
            print(f"  - {d}")
        print("=" * 75)
        return 1


if __name__ == "__main__":
    sys.exit(main())
