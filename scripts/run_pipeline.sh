#!/usr/bin/env bash
# ==============================================================================
# E-Commerce Intelligence Platform — Run Pipeline Script (Bash)
# ==============================================================================
set -euo pipefail

MODE="inference"
SKIP_DB=false
STAGE=""
TIMEOUT_SECONDS=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --skip-db)
            SKIP_DB=true
            shift
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

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

ARGS=("scripts/run_pipeline.py" "--mode" "${MODE}" "--timeout" "${TIMEOUT_SECONDS}")
if [ "${SKIP_DB}" = "true" ]; then
    ARGS+=("--skip-db")
fi
if [ -n "${STAGE}" ]; then
    ARGS+=("--stage" "${STAGE}")
fi

"${VENV_PYTHON}" "${ARGS[@]}"
exit $?
