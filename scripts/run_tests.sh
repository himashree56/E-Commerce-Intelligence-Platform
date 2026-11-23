#!/usr/bin/env bash
# ==============================================================================
# E-Commerce Intelligence Platform — Run Tests Script (Bash)
# ==============================================================================
set -euo pipefail

VERBOSE=false
COVERAGE=false
FILTER=""
TARGET="tests/"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -k|--filter)
            FILTER="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "E-COMMERCE INTELLIGENCE PLATFORM — TEST SUITE EXECUTION (BASH)"
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

PYTEST_ARGS=("-m" "pytest" "${TARGET}")
if [ "${VERBOSE}" = "true" ]; then
    PYTEST_ARGS+=("-v")
else
    PYTEST_ARGS+=("-q")
fi

if [ -n "${FILTER}" ]; then
    PYTEST_ARGS+=("-k" "${FILTER}")
fi

if [ "${COVERAGE}" = "true" ]; then
    PYTEST_ARGS+=("--cov=src" "--cov-report=term-missing")
fi

set +e
"${VENV_PYTHON}" "${PYTEST_ARGS[@]}"
EXIT_CODE=$?
set -e

echo -e "\n================================================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "ALL TESTS PASSED SUCCESSFULLY [100% PASS RATE]"
else
    echo "TEST SUITE COMPLETED WITH FAILURES (Exit Code: ${EXIT_CODE})"
fi
echo "================================================================================"
exit ${EXIT_CODE}
