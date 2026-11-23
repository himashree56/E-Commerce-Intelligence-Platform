#!/usr/bin/env bash
# ==============================================================================
# E-Commerce Intelligence Platform — Stop Services Script (Bash)
# ==============================================================================
set -euo pipefail

DOWN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --down)
            DOWN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "E-COMMERCE INTELLIGENCE PLATFORM — SERVICE SHUTDOWN (BASH)"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "${DOWN}" = "true" ]; then
    echo -e "\nRemoving containers and network (preserving persistent database volumes)..."
    docker compose down
else
    echo -e "\nStopping PostgreSQL container gracefully..."
    docker compose stop postgres
fi

echo -e "\n================================================================================"
echo "SERVICES STOPPED [CLEAN SHUTDOWN]"
echo "================================================================================"
exit 0
