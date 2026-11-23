"""Database Schema Manager for creating, dropping, and inspecting PostgreSQL tables."""
import logging
from typing import Dict, List, Optional

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from src.database.connection import DatabaseManager
from src.database.models import Base

logger = logging.getLogger("ecommerce_platform.database.schema_manager")


class SchemaManager:
    """Manages table creation, migration, and inspection."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db_manager = db_manager or DatabaseManager()

    def create_all_tables(self, is_test: bool = False) -> List[str]:
        """Creates all registered SQLAlchemy tables in the target database."""
        engine = self.db_manager.engine if not is_test else DatabaseManager(is_test=True).engine
        logger.info(f"Creating all database tables on engine: {engine.url.database}...")
        Base.metadata.create_all(bind=engine)
        inspector = inspect(engine)
        created_tables = inspector.get_table_names()
        logger.info(f"Tables successfully created/verified: {created_tables}")
        return created_tables

    def drop_all_tables(self, is_test: bool = False) -> None:
        """Drops all registered SQLAlchemy tables (primarily for test cleanup)."""
        engine = self.db_manager.engine if not is_test else DatabaseManager(is_test=True).engine
        logger.warning(f"Dropping all tables on database: {engine.url.database}...")
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped successfully.")

    def get_table_row_counts(self, is_test: bool = False) -> Dict[str, int]:
        """Inspects and returns row counts for all application tables."""
        from sqlalchemy import text

        engine = self.db_manager.engine if not is_test else DatabaseManager(is_test=True).engine
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        counts: Dict[str, int] = {}

        with engine.connect() as conn:
            for tbl in tables:
                cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl};")).scalar()
                counts[tbl] = int(cnt) if cnt is not None else 0

        return counts
