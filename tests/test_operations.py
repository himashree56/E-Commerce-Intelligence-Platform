"""
Phase 8 Operations & Automation Automated Test Suite.

Validates environment validator, structured operational logger, atomic file operations,
pipeline orchestration modes, and Docker Compose configuration.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.utils.env_validator import (
    check_python_version,
    check_directory_tree,
    check_disk_space,
    check_env_file,
    check_docker_cli,
    validate_environment,
)
from src.orchestration.logger import OperationalLogger, StageRunRecord
from src.orchestration.pipeline import (
    PipelineMode,
    PipelineOrchestrator,
    atomic_save_parquet,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestEnvironmentValidator:
    """Tests for system environment and prerequisites validator."""

    def test_python_version_valid(self):
        ok, msg = check_python_version(min_major=3, min_minor=10)
        assert ok is True
        assert "Python" in msg

    def test_directory_tree(self, tmp_path):
        ok, created, errors = check_directory_tree(tmp_path)
        assert ok is True
        assert len(created) > 0
        assert len(errors) == 0

    def test_disk_space_check(self):
        ok, msg = check_disk_space(min_free_gb=0.1)
        assert ok is True
        assert "Available disk space" in msg

    def test_env_file_check(self):
        ok, warnings, errors = check_env_file(PROJECT_ROOT)
        assert ok is True
        assert len(errors) == 0

    def test_docker_cli_check(self):
        ok, msg = check_docker_cli()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_validate_environment_composite(self):
        res = validate_environment(check_db=False, project_root=PROJECT_ROOT)
        assert res.is_valid is True
        assert len(res.checks_passed) >= 4
        assert len(res.errors) == 0


class TestOperationalLogger:
    """Tests for structured operational logger and JSONL audit trail."""

    def test_logger_lifecycle(self, tmp_path):
        manifest = tmp_path / "test_runs.jsonl"
        logger = OperationalLogger(
            run_id="test_run_123",
            pipeline_mode="test",
            manifest_path=manifest,
            enable_db_logging=False,
        )

        record = logger.log_stage_start("TEST_STAGE")
        assert record.status == "RUNNING"
        assert record.pipeline_stage == "TEST_STAGE"

        completed = logger.log_stage_complete(record, records_processed=100)
        assert completed.status == "SUCCESS"
        assert completed.records_processed == 100
        assert completed.duration_seconds >= 0.0

        skipped = logger.log_stage_skipped("SKIPPED_STAGE", "testing skip")
        assert skipped.status == "SKIPPED"

        # Verify lines in manifest
        with open(manifest) as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) >= 3

    def test_logger_track_stage_context_manager(self, tmp_path):
        manifest = tmp_path / "test_runs_ctx.jsonl"
        logger = OperationalLogger(
            run_id="test_run_ctx",
            pipeline_mode="test",
            manifest_path=manifest,
            enable_db_logging=False,
        )

        with logger.track_stage("CTX_STAGE", expected_records=50) as ctx:
            ctx["records"] = 55
            ctx["details"] = {"metric": "val"}

        assert len(logger.stage_records) == 1
        assert logger.stage_records[0].records_processed == 55
        assert logger.stage_records[0].status == "SUCCESS"


class TestAtomicFileOperations:
    """Tests atomic file writing to prevent dataset corruption on failure."""

    def test_atomic_save_parquet_success(self, tmp_path):
        target = tmp_path / "subfolder" / "data.parquet"
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        atomic_save_parquet(df, target)

        assert target.exists()
        loaded = pd.read_parquet(target)
        assert len(loaded) == 3

    def test_atomic_save_parquet_overwrites(self, tmp_path):
        target = tmp_path / "data.parquet"
        df1 = pd.DataFrame({"val": [10, 20]})
        atomic_save_parquet(df1, target)

        df2 = pd.DataFrame({"val": [10, 20, 30, 40]})
        atomic_save_parquet(df2, target)

        loaded = pd.read_parquet(target)
        assert len(loaded) == 4


class TestPipelineOrchestration:
    """Tests for pipeline orchestrator modes and stage dispatch."""

    def test_orchestrator_initialization(self):
        orchestrator = PipelineOrchestrator(mode=PipelineMode.VALIDATE, skip_db=True)
        assert orchestrator.mode == PipelineMode.VALIDATE
        assert orchestrator.skip_db is True
        assert "validate" in orchestrator.run_id

    def test_orchestrator_reconciliation_audit(self):
        orchestrator = PipelineOrchestrator(mode=PipelineMode.VALIDATE, skip_db=True)
        res = orchestrator.run_reconciliation_audit()
        assert res["reconciliation_status"] == "ZERO-PENNY BALANCED"
        assert np.isclose(res["net_revenue"], 19012910.85, atol=0.01)


class TestDockerComposeValidation:
    """Tests Docker Compose configuration parses cleanly."""

    def test_docker_compose_config_valid(self):
        res = subprocess.run(
            ["docker", "compose", "config"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "ecommerce_intelligence_postgres" in res.stdout
