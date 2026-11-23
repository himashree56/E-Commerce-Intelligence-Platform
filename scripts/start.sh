#!/usr/bin/env bash
# ==============================================================================
# E-Commerce Intelligence Platform — Start Services Script (Bash)
# ==============================================================================
set -euo pipefail

WITH_DASHBOARD=false
TIMEOUT_SECONDS=60
DASHBOARD_PORT=8501

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-dashboard)
            WITH_DASHBOARD=true
            shift
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --port)
            DASHBOARD_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "E-COMMERCE INTELLIGENCE PLATFORM — SERVICE STARTUP (BASH)"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    VENV_PYTHON="python3"
fi

# 1. Validate Docker Compose config
echo -e "\n[1/3] Validating Docker Compose configuration..."
docker compose config > /dev/null
echo "  Docker Compose configuration validated."

# 2. Start PostgreSQL container
echo -e "\n[2/3] Starting PostgreSQL 16 container..."
docker compose up -d postgres

# 3. Wait for PostgreSQL readiness
echo "  Waiting for database health check (timeout: ${TIMEOUT_SECONDS}s)..."
elapsed=0
healthy=false

while [ "${elapsed}" -lt "${TIMEOUT_SECONDS}" ]; do
    if "${VENV_PYTHON}" scripts/check_database.py > /dev/null 2>&1; then
        healthy=true
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ "${healthy}" != "true" ]; then
    echo "ERROR: PostgreSQL failed to become healthy within ${TIMEOUT_SECONDS} seconds." >&2
    exit 1
fi
echo "  PostgreSQL 16 is ready and accepting connections."

# 4. Optional Dashboard Startup
if [ "${WITH_DASHBOARD}" = "true" ]; then
    echo -e "\n[3/3] Launching Streamlit analytics dashboard on port ${DASHBOARD_PORT}..."
    "${VENV_PYTHON}" scripts/run_dashboard.py --port "${DASHBOARD_PORT}"
else
    echo -e "\n[3/3] Platform services started successfully."
    echo "  To launch dashboard, run: python scripts/run_dashboard.py --port ${DASHBOARD_PORT}"
fi

echo "================================================================================"
echo "SERVICES OPERATIONAL [READY]"
echo "================================================================================"
exit 0
