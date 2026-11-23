"""PostgreSQL Database Configuration and Environment Management."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file if available
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class DatabaseConfig:
    """Manages database connection parameters and pooling configuration."""

    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "ecommerce_intelligence"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "ecommerce_user"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", "ecommerce_secure_pass_2026"))
    schema: str = field(default_factory=lambda: os.getenv("DB_SCHEMA", "public"))

    # Connection pooling settings
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20")))
    pool_timeout: int = field(default_factory=lambda: int(os.getenv("DB_POOL_TIMEOUT", "30")))
    pool_recycle: int = field(default_factory=lambda: int(os.getenv("DB_POOL_RECYCLE", "1800")))

    # Test database name
    test_db_name: str = field(
        default_factory=lambda: os.getenv("TEST_DB_NAME", "ecommerce_intelligence_test")
    )

    def get_connection_url(self, is_test: bool = False) -> str:
        """Returns standard SQLAlchemy connection URL using psycopg2 driver."""
        target_db = self.test_db_name if is_test else self.db_name
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{target_db}"

    def get_dsn(self, is_test: bool = False) -> str:
        """Returns standard libpq DSN connection string."""
        target_db = self.test_db_name if is_test else self.db_name
        return (
            f"host={self.host} port={self.port} dbname={target_db} "
            f"user={self.user} password={self.password}"
        )
