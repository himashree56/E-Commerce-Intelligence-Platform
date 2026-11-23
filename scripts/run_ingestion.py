"""CLI entrypoint for Phase 1 Bronze Ingestion."""
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.config import IngestionConfig
from src.ingestion.pipeline import IngestionPipeline


def main() -> int:
    """Execute ingestion pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="E-Commerce Intelligence Platform - Phase 1 Bronze Ingestion Pipeline"
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Optional path to existing raw Excel or zip file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction or re-download of raw dataset",
    )

    args = parser.parse_args()

    config = IngestionConfig(base_dir=PROJECT_ROOT)
    pipeline = IngestionPipeline(config=config)

    print("=" * 70)
    print("   E-COMMERCE INTELLIGENCE PLATFORM - PHASE 1 INGESTION PIPELINE")
    print("=" * 70)

    try:
        summary = pipeline.run(force=args.force, raw_file_path=args.raw_path)

        print("\n" + "=" * 70)
        print("                 INGESTION RUN SUMMARY")
        print("=" * 70)
        print(f"Run ID:                    {summary.run_id}")
        print(f"Status:                    {summary.status}")
        print(f"Execution Duration:        {summary.duration_seconds}s")
        print(f"Raw Source File:           {summary.raw_source_file}")
        print(f"Sheets Processed:          {', '.join(summary.sheets_processed)}")
        for sheet, count in summary.sheet_row_counts.items():
            print(f"  - {sheet:22}: {count:,} records")
        print(f"Total Bronze Rows:         {summary.total_bronze_rows:,}")
        print(f"Total Bronze Columns:      {summary.total_columns}")
        print(f"Unified Parquet Path:      {summary.unified_parquet_path}")
        print(f"Unified Parquet Size:      {summary.unified_parquet_size_bytes:,} bytes")
        print("\nPreserved Quality Metrics:")
        for metric, val in summary.preserved_metrics.items():
            if isinstance(val, int):
                print(f"  - {metric:28}: {val:,}")
            else:
                print(f"  - {metric:28}: {val}")

        print("\nIndividual Sheet Parquets:")
        for sheet, path in summary.individual_parquet_paths.items():
            print(f"  - {sheet:22}: {path}")

        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
