"""Database connection management, session pooling, and health check utilities."""
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.config import DatabaseConfig

logger = logging.getLogger("ecommerce_platform.database.connection")


class DatabaseManager:
    """Singleton-style database connection and session manager."""

    def __init__(self, config: Optional[DatabaseConfig] = None, is_test: bool = False) -> None:
        self.config = config or DatabaseConfig()
        self.is_test = is_test
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        """Lazily initializes and returns the SQLAlchemy Engine with connection pooling."""
        if self._engine is None:
            url = self.config.get_connection_url(is_test=self.is_test)
            self._engine = create_engine(
                url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Health check connection prior to borrowing from pool
                future=True,
            )
            logger.info(
                f"Initialized SQLAlchemy Engine for database: "
                f"{self.config.test_db_name if self.is_test else self.config.db_name} "
                f"(pool_size={self.config.pool_size}, max_overflow={self.config.max_overflow})"
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Returns the configured sessionmaker."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
        return self._session_factory

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Context manager providing transactional session scope with auto-rollback on error."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            session.close()

    def check_health(self) -> Dict[str, Any]:
        """Tests active database connectivity, executes probe query, and measures latency."""
        start_time = time.perf_counter()
        try:
            with self.engine.connect() as conn:
                version = conn.execute(text("SELECT version();")).scalar()
                current_db = conn.execute(text("SELECT current_database();")).scalar()
                current_user = conn.execute(text("SELECT current_user;")).scalar()

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "status": "HEALTHY",
                "database": current_db,
                "user": current_user,
                "version": version,
                "latency_ms": latency_ms,
                "host": self.config.host,
                "port": self.config.port,
            }
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "UNHEALTHY",
                "error": str(e),
                "latency_ms": latency_ms,
                "host": self.config.host,
                "port": self.config.port,
            }

    def dispose(self) -> None:
        """Disposes the connection pool."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connection pool disposed.")
