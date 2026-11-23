"""
Operational Structured Logging for Pipeline Orchestration.

Records run metadata to local JSONL audit trail and operational PostgreSQL
database table (etl_execution_metadata).
"""
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = LOGS_DIR / "pipeline_runs.jsonl"


@dataclass
class StageRunRecord:
    """Structured record of a single pipeline stage execution."""
    run_id: str
    pipeline_stage: str
    status: str  # 'RUNNING', 'SUCCESS', 'FAILED', 'SKIPPED'
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    records_processed: int = 0
    model_version: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts record to JSON-serializable dictionary."""
        return asdict(self)


class OperationalLogger:
    """
    Manages structured operational audit trails across file and relational sinks.
    """

    def __init__(
        self,
        run_id: str,
        pipeline_mode: str = "full",
        manifest_path: Path = MANIFEST_PATH,
        enable_db_logging: bool = True,
    ):
        self.run_id = run_id
        self.pipeline_mode = pipeline_mode
        self.manifest_path = manifest_path
        self.enable_db_logging = enable_db_logging
        self.logger = logging.getLogger(f"ecommerce.pipeline.{run_id}")
        self.stage_records: List[StageRunRecord] = []

    def _append_to_file(self, record: StageRunRecord) -> None:
        """Appends record to local JSON Lines manifest."""
        try:
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            self.logger.warning(f"Could not append to JSONL manifest: {e}")

    def _persist_to_postgres(self, record: StageRunRecord) -> None:
        """Writes execution record to PostgreSQL etl_execution_metadata if available."""
        if not self.enable_db_logging:
            return

        try:
            from src.database.connection import DatabaseManager
            from sqlalchemy import text
            dm = DatabaseManager()
            with dm.engine.connect() as conn:
                q = text("""
                    INSERT INTO etl_execution_metadata (
                        run_id, pipeline_phase, status, duration_seconds, records_processed, details, created_at
                    ) VALUES (
                        :run_id, :phase, :status, :duration, :records, :details, :created_at
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        duration_seconds = EXCLUDED.duration_seconds,
                        records_processed = EXCLUDED.records_processed,
                        details = EXCLUDED.details;
                """)
                payload = {
                    "run_id": f"{record.run_id}_{record.pipeline_stage}",
                    "phase": record.pipeline_stage,
                    "status": record.status,
                    "duration": round(record.duration_seconds, 2),
                    "records": record.records_processed,
                    "details": json.dumps({
                        **record.details,
                        "overall_run_id": record.run_id,
                        "pipeline_mode": self.pipeline_mode,
                        "error_message": record.error_message,
                        "model_version": record.model_version,
                    }),
                    "created_at": record.started_at,
                }
                conn.execute(q, payload)
                conn.commit()
        except Exception as e:
            self.logger.debug(f"Database metadata persistence skipped: {e}")

    def log_stage_start(self, stage_name: str) -> StageRunRecord:
        """Logs initiation of a pipeline stage."""
        now_str = datetime.now(timezone.utc).isoformat()
        record = StageRunRecord(
            run_id=self.run_id,
            pipeline_stage=stage_name,
            status="RUNNING",
            started_at=now_str,
            details={"pipeline_mode": self.pipeline_mode},
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [START] Stage: {stage_name} (Mode: {self.pipeline_mode})")
        self._append_to_file(record)
        return record

    def log_stage_complete(
        self,
        record: StageRunRecord,
        records_processed: int = 0,
        details: Optional[Dict[str, Any]] = None,
        model_version: Optional[str] = None,
    ) -> StageRunRecord:
        """Logs successful completion of a stage."""
        now_str = datetime.now(timezone.utc).isoformat()
        record.status = "SUCCESS"
        record.completed_at = now_str
        start_dt = datetime.fromisoformat(record.started_at)
        record.duration_seconds = (datetime.fromisoformat(now_str) - start_dt).total_seconds()
        record.records_processed = records_processed
        record.model_version = model_version
        if details:
            record.details.update(details)

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] Stage: {record.pipeline_stage} "
            f"in {record.duration_seconds:.2f}s | Records: {records_processed:,}"
        )
        self._append_to_file(record)
        self._persist_to_postgres(record)
        self.stage_records.append(record)
        return record

    def log_stage_failure(
        self,
        record: StageRunRecord,
        error: Exception,
        details: Optional[Dict[str, Any]] = None,
    ) -> StageRunRecord:
        """Logs failure of a stage."""
        now_str = datetime.now(timezone.utc).isoformat()
        record.status = "FAILED"
        record.completed_at = now_str
        start_dt = datetime.fromisoformat(record.started_at)
        record.duration_seconds = (datetime.fromisoformat(now_str) - start_dt).total_seconds()
        record.error_message = str(error)
        if details:
            record.details.update(details)

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] [FAILED] Stage: {record.pipeline_stage} "
            f"after {record.duration_seconds:.2f}s | Error: {error}"
        )
        self._append_to_file(record)
        self._persist_to_postgres(record)
        self.stage_records.append(record)
        return record

    def log_stage_skipped(self, stage_name: str, reason: str) -> StageRunRecord:
        """Logs skipped stage."""
        now_str = datetime.now(timezone.utc).isoformat()
        record = StageRunRecord(
            run_id=self.run_id,
            pipeline_stage=stage_name,
            status="SKIPPED",
            started_at=now_str,
            completed_at=now_str,
            details={"skip_reason": reason, "pipeline_mode": self.pipeline_mode},
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [SKIPPED] Stage: {stage_name} | Reason: {reason}")
        self._append_to_file(record)
        self._persist_to_postgres(record)
        self.stage_records.append(record)
        return record

    @contextmanager
    def track_stage(
        self,
        stage_name: str,
        expected_records: int = 0,
        model_version: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager wrapping a stage execution with automatic timing and error capture.
        """
        record = self.log_stage_start(stage_name)
        ctx: Dict[str, Any] = {"records": expected_records, "details": {}}
        t0 = time.perf_counter()
        try:
            yield ctx
            rec_cnt = ctx.get("records", expected_records)
            self.log_stage_complete(
                record,
                records_processed=rec_cnt,
                details=ctx.get("details", {}),
                model_version=model_version,
            )
        except Exception as exc:
            self.log_stage_failure(record, error=exc, details=ctx.get("details", {}))
            raise
