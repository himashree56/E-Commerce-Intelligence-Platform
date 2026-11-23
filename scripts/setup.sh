#!/usr/bin/env bash
# ==============================================================================
# E-Commerce Intelligence Platform — Unix/Linux Bash Setup Script
# ==============================================================================
set -euo pipefail

echo "================================================================================"
echo "E-COMMERCE INTELLIGENCE PLATFORM — SYSTEM SETUP (BASH)"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 1. Check Python
echo -e "\n[1/5] Checking Python runtime..."
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "ERROR: Neither python3 nor python found in PATH." >&2
    exit 1
fi
echo "  Found: $("${PYTHON_BIN}" --version)"

# 2. Check or create virtual environment
echo -e "\n[2/5] Checking virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment at .venv..."
    "${PYTHON_BIN}" -m venv .venv
fi

if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    VENV_PYTHON="${PYTHON_BIN}"
fi
echo "  Virtual environment ready: ${VENV_PYTHON}"

# 3. Install requirements
echo -e "\n[3/5] Installing/verifying project dependencies..."
if [ -f "requirements.txt" ]; then
    "${VENV_PYTHON}" -m pip install --upgrade pip --quiet
    "${VENV_PYTHON}" -m pip install -r requirements.txt --quiet
    echo "  Dependencies installed successfully."
fi

# 4. Initialize .env configuration
echo -e "\n[4/5] Checking environment configuration (.env)..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "  Created .env from .env.example with local defaults."
    fi
else
    echo "  .env configuration exists."
fi

# 5. Execute environment validator
echo -e "\n[5/5] Running environment validator..."
"${VENV_PYTHON}" src/utils/env_validator.py

echo "================================================================================"
echo "SETUP COMPLETED SUCCESSFULLY [PASS]"
echo "================================================================================"
exit 0
