"""Configuration parameters for dataset ingestion."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class IngestionConfig:
    """Configuration settings for raw data download, ingestion, and Bronze output."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    bronze_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # Dataset remote information
    dataset_url: str = (
        "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    )
    zip_filename: str = "online+retail+ii.zip"
    excel_filename: str = "online_retail_II.xlsx"

    # Workbook configuration
    expected_sheets: List[str] = field(
        default_factory=lambda: ["Year 2009-2010", "Year 2010-2011"]
    )

    # Raw expected columns
    expected_raw_columns: List[str] = field(
        default_factory=lambda: [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]
    )

    # Column standardization mapping (raw -> bronze)
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "Invoice": "invoice_no",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "unit_price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )

    # Parquet settings
    parquet_compression: str = "snappy"
    unified_bronze_file: str = "online_retail_bronze.parquet"

    def __post_init__(self) -> None:
        """Resolve all directory paths relative to base_dir."""
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.bronze_dir = self.data_dir / "bronze"
        self.metadata_dir = self.data_dir / "metadata"
        self.logs_dir = self.base_dir / "logs"

        # Ensure directories exist
        for directory in [
            self.raw_dir,
            self.bronze_dir,
            self.metadata_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def zip_path(self) -> Path:
        """Full path to local raw zip file."""
        return self.raw_dir / self.zip_filename

    @property
    def excel_path(self) -> Path:
        """Full path to local raw excel file."""
        return self.raw_dir / self.excel_filename

    @property
    def unified_bronze_path(self) -> Path:
        """Full path to unified bronze parquet output."""
        return self.bronze_dir / self.unified_bronze_file
