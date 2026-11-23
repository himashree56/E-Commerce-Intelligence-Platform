"""Database connectivity and health check CLI script."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import DatabaseManager


def main() -> int:
    print("=" * 70)
    print("      E-COMMERCE INTELLIGENCE PLATFORM - DATABASE HEALTH CHECK")
    print("=" * 70)

    # 1. Primary Database
    prod_mgr = DatabaseManager(is_test=False)
    prod_health = prod_mgr.check_health()

    print("\n[PRIMARY ANALYTICS DATABASE]")
    print(f"  - Target DB:   {prod_health.get('database', 'UNKNOWN')}")
    print(f"  - Host/Port:   {prod_health.get('host')}:{prod_health.get('port')}")
    print(f"  - User:        {prod_health.get('user', 'UNKNOWN')}")
    print(f"  - Status:      {prod_health.get('status')}")
    print(f"  - Latency:     {prod_health.get('latency_ms')} ms")
    if prod_health.get("status") == "HEALTHY":
        print(f"  - Version:     {prod_health.get('version')}")
    else:
        print(f"  - Error:       {prod_health.get('error')}")

    # 2. Test Database
    test_mgr = DatabaseManager(is_test=True)
    test_health = test_mgr.check_health()

    print("\n[ISOLATED TEST DATABASE]")
    print(f"  - Target DB:   {test_health.get('database', 'UNKNOWN')}")
    print(f"  - Host/Port:   {test_health.get('host')}:{test_health.get('port')}")
    print(f"  - User:        {test_health.get('user', 'UNKNOWN')}")
    print(f"  - Status:      {test_health.get('status')}")
    print(f"  - Latency:     {test_health.get('latency_ms')} ms")
    if test_health.get("status") != "HEALTHY":
        print(f"  - Error:       {test_health.get('error')}")

    print("\n" + "=" * 70)
    is_healthy = (
        prod_health.get("status") == "HEALTHY"
        and test_health.get("status") == "HEALTHY"
    )
    if is_healthy:
        print("[SUCCESS] All database endpoints are operational and healthy!")
        print("=" * 70)
        return 0
    else:
        print("[FAILURE] One or more database endpoints failed health checks.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
