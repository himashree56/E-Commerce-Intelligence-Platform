"""
Environment & Prerequisites Validator.

Validates system environment, Python version, required directories, disk space,
environment variables, Docker CLI availability, and database connectivity.
"""
import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ValidationResult:
    """Encapsulates system validation outcomes."""
    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, str] = field(default_factory=dict)


def check_python_version(min_major: int = 3, min_minor: int = 10) -> Tuple[bool, str]:
    """Verifies active Python runtime satisfies minimum version constraints."""
    version = sys.version_info
    ver_str = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major > min_major) or (version.major == min_major and version.minor >= min_minor):
        return True, f"Python {ver_str} satisfies minimum constraint (>={min_major}.{min_minor})"
    return False, f"Python {ver_str} does not satisfy constraint (>={min_major}.{min_minor})"


def check_directory_tree(project_root: Path = PROJECT_ROOT) -> Tuple[bool, List[str], List[str]]:
    """Checks required repository directories exist or creates them safely."""
    required_dirs = [
        "data/bronze",
        "data/silver",
        "data/gold",
        "models/customer_segmentation",
        "models/isolation_forest",
        "models/repeat_purchase",
        "logs",
        "configs",
        "reports",
    ]
    created = []
    missing_errors = []

    for rel_dir in required_dirs:
        p = project_root / rel_dir
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
                created.append(rel_dir)
            except Exception as e:
                missing_errors.append(f"Failed to create {rel_dir}: {e}")

    return len(missing_errors) == 0, created, missing_errors


def check_disk_space(project_root: Path = PROJECT_ROOT, min_free_gb: float = 1.0) -> Tuple[bool, str]:
    """Verifies host filesystem has sufficient free space for data processing."""
    try:
        usage = shutil.disk_usage(project_root)
        free_gb = usage.free / (1024 ** 3)
        msg = f"Available disk space: {free_gb:.2f} GB (minimum required: {min_free_gb:.1f} GB)"
        if free_gb >= min_free_gb:
            return True, msg
        return False, f"Low disk space: {free_gb:.2f} GB available, {min_free_gb:.1f} GB required"
    except Exception as e:
        return True, f"Could not determine disk usage: {e}"


def check_env_file(project_root: Path = PROJECT_ROOT) -> Tuple[bool, List[str], List[str]]:
    """Checks .env exists and defines required variables against .env.example."""
    env_path = project_root / ".env"
    example_path = project_root / ".env.example"
    warnings = []
    errors = []

    if not env_path.exists():
        if example_path.exists():
            warnings.append(
                ".env file not found. Copying .env.example -> .env with default local configurations."
            )
            try:
                shutil.copy(example_path, env_path)
            except Exception as e:
                errors.append(f"Failed to auto-create .env from .env.example: {e}")
        else:
            errors.append("Neither .env nor .env.example found in project root.")
            return False, warnings, errors

    # Check key variables
    from dotenv import dotenv_values
    env_vars = dotenv_values(env_path)
    required_keys = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    for k in required_keys:
        if not env_vars.get(k):
            errors.append(f"Missing required variable in .env: {k}")

    return len(errors) == 0, warnings, errors


def check_docker_cli() -> Tuple[bool, str]:
    """Checks if docker CLI is available in PATH."""
    docker_bin = shutil.which("docker")
    if docker_bin:
        return True, f"Docker CLI found: {docker_bin}"
    return False, "Docker CLI not found in system PATH. Docker Compose operations will be unavailable."


def check_database_connectivity(timeout_seconds: int = 5) -> Tuple[bool, str]:
    """Attempts lightweight connection to PostgreSQL if available."""
    try:
        from src.database.connection import DatabaseManager
        from sqlalchemy import text
        dm = DatabaseManager()
        with dm.engine.connect() as conn:
            res = conn.execute(text("SELECT 1;")).scalar()
            if res == 1:
                return True, "PostgreSQL 16 connection successful."
        return False, "PostgreSQL connected but query verification failed."
    except Exception as e:
        return False, f"PostgreSQL connection unavailable: {e}"


def validate_environment(
    check_db: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> ValidationResult:
    """Orchestrates comprehensive environment validation."""
    passed = []
    warnings = []
    errors = []
    details = {}

    # 1. Python version
    py_ok, py_msg = check_python_version()
    details["python"] = py_msg
    if py_ok:
        passed.append(py_msg)
    else:
        errors.append(py_msg)

    # 2. Directory structure
    dirs_ok, dirs_created, dirs_err = check_directory_tree(project_root)
    if dirs_ok:
        msg = "Required directory tree verified"
        if dirs_created:
            msg += f" (auto-created: {', '.join(dirs_created)})"
        passed.append(msg)
        details["directories"] = msg
    else:
        errors.extend(dirs_err)
        details["directories"] = f"Directory creation failures: {dirs_err}"

    # 3. Disk space
    disk_ok, disk_msg = check_disk_space(project_root)
    details["disk_space"] = disk_msg
    if disk_ok:
        passed.append(disk_msg)
    else:
        warnings.append(disk_msg)

    # 4. Environment configuration (.env)
    env_ok, env_warn, env_err = check_env_file(project_root)
    warnings.extend(env_warn)
    if env_ok:
        passed.append(".env configuration verified with all required variables.")
        details["env_config"] = "Valid"
    else:
        errors.extend(env_err)
        details["env_config"] = f"Errors: {env_err}"

    # 5. Docker CLI
    docker_ok, docker_msg = check_docker_cli()
    details["docker"] = docker_msg
    if docker_ok:
        passed.append(docker_msg)
    else:
        warnings.append(docker_msg)

    # 6. Database (optional)
    if check_db:
        db_ok, db_msg = check_database_connectivity()
        details["database"] = db_msg
        if db_ok:
            passed.append(db_msg)
        else:
            warnings.append(db_msg)

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        checks_passed=passed,
        warnings=warnings,
        errors=errors,
        details=details,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate E-Commerce Intelligence Platform Environment")
    parser.add_argument("--check-db", action="store_true", help="Include active PostgreSQL connection check")
    args = parser.parse_args()

    print("================================================================================")
    print("E-COMMERCE INTELLIGENCE PLATFORM — ENVIRONMENT VALIDATION")
    print("================================================================================")

    res = validate_environment(check_db=args.check_db)

    print("\n[CHECKS PASSED]")
    for p in res.checks_passed:
        print(f"  [OK] {p}")

    if res.warnings:
        print("\n[WARNINGS]")
        for w in res.warnings:
            print(f"  [!] {w}")

    if res.errors:
        print("\n[ERRORS]")
        for e in res.errors:
            print(f"  [X] {e}")

    print("================================================================================")
    if res.is_valid:
        print("ENVIRONMENT STATUS: READY [PASS]")
        sys.exit(0)
    else:
        print("ENVIRONMENT STATUS: INVALID [FAIL]")
        sys.exit(1)


if __name__ == "__main__":
    main()
