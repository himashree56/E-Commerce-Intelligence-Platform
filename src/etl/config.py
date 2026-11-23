"""Configuration for Silver and Gold ETL transformations."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ETLConfig:
    """Paths and business parameters for Silver and Gold layers."""

    base_dir: Path = field(default_factory=lambda: Path.cwd())
    bronze_dir: Path = field(init=False)
    silver_dir: Path = field(init=False)
    gold_dir: Path = field(init=False)
    delta_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # Input Bronze file
    bronze_file: str = "online_retail_bronze.parquet"

    # Silver output file
    silver_file: str = "online_retail_silver.parquet"

    # Gold output files
    gold_monthly_sales_file: str = "gold_monthly_sales.parquet"
    gold_customer_rfm_file: str = "gold_customer_rfm.parquet"
    gold_product_performance_file: str = "gold_product_performance.parquet"
    gold_country_revenue_file: str = "gold_country_revenue.parquet"
    gold_customer_purchase_history_file: str = "gold_customer_purchase_history.parquet"

    # Business rule parameters
    non_product_stock_codes: List[str] = field(
        default_factory=lambda: [
            "POST",
            "D",
            "M",
            "BANK CHARGES",
            "CRUK",
            "PADS",
            "DOT",
            "C2",
            "TEST",
            "ADJUST",
            "ADJUST2",
        ]
    )

    duplicate_business_cols: List[str] = field(
        default_factory=lambda: [
            "invoice_no",
            "stock_code",
            "quantity",
            "invoice_date",
            "unit_price",
            "customer_id",
        ]
    )

    def __post_init__(self) -> None:
        """Resolve all directories relative to base_dir."""
        data_dir = self.base_dir / "data"
        self.bronze_dir = data_dir / "bronze"
        self.silver_dir = data_dir / "silver"
        self.gold_dir = data_dir / "gold"
        self.delta_dir = data_dir / "delta"
        self.metadata_dir = data_dir / "metadata"
        self.logs_dir = self.base_dir / "logs"

        for d in [
            self.silver_dir,
            self.gold_dir,
            self.delta_dir,
            self.metadata_dir,
            self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def bronze_parquet_path(self) -> Path:
        return self.bronze_dir / self.bronze_file

    @property
    def silver_parquet_path(self) -> Path:
        return self.silver_dir / self.silver_file

    @property
    def gold_monthly_sales_path(self) -> Path:
        return self.gold_dir / self.gold_monthly_sales_file

    @property
    def gold_customer_rfm_path(self) -> Path:
        return self.gold_dir / self.gold_customer_rfm_file

    @property
    def gold_product_performance_path(self) -> Path:
        return self.gold_dir / self.gold_product_performance_file

    @property
    def gold_country_revenue_path(self) -> Path:
        return self.gold_dir / self.gold_country_revenue_file

    @property
    def gold_customer_purchase_history_path(self) -> Path:
        return self.gold_dir / self.gold_customer_purchase_history_file
