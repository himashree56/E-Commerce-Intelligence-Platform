"""Reconciliation engine verifying 100% accounting of Bronze records through Silver and Gold."""
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.etl.config import ETLConfig
from src.etl.silver import QualityFlag, TransactionStatus

logger = logging.getLogger("ecommerce_platform.reconciliation")


@dataclass
class ReconciliationReport:
    """Detailed reconciliation verification results across Bronze, Silver, and Gold."""

    bronze_total_rows: int
    silver_total_rows: int
    is_row_count_balanced: bool

    status_breakdown: Dict[str, int]
    status_sum: int
    is_status_exhaustive: bool

    quality_breakdown: Dict[str, int]

    silver_gross_sales_gbp: float
    silver_return_amount_gbp: float
    silver_net_revenue_gbp: float
    is_revenue_equation_balanced: bool

    gold_monthly_net_revenue_gbp: float
    gold_country_net_revenue_gbp: float
    is_gold_silver_reconciled: bool

    anomalies_accounted_for: Dict[str, int]
    all_tests_passed: bool
    discrepancies: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReconciliationAuditor:
    """Audits data lineage, record accounting, and financial integrity across layers."""

    def __init__(self, config: Optional[ETLConfig] = None) -> None:
        self.config = config or ETLConfig()

    def audit(
        self,
        bronze_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        gold_monthly_df: Optional[pd.DataFrame] = None,
        gold_country_df: Optional[pd.DataFrame] = None,
    ) -> ReconciliationReport:
        """Executes full audit verifying Bronze -> Silver -> Gold reconciliation."""
        discrepancies: List[str] = []

        # 1. Total row count preservation (Bronze == Silver)
        b_count = len(bronze_df)
        s_count = len(silver_df)
        row_balanced = b_count == s_count
        if not row_balanced:
            discrepancies.append(
                f"Row count mismatch: Bronze has {b_count:,} rows, Silver has {s_count:,} rows."
            )

        # 2. Transaction Status Exhaustiveness
        status_counts = silver_df["transaction_status"].value_counts().to_dict()
        status_sum = int(sum(status_counts.values()))
        status_exhaustive = status_sum == s_count
        if not status_exhaustive:
            discrepancies.append(
                f"Status mismatch: Sum of statuses ({status_sum:,}) != Silver rows ({s_count:,})."
            )

        # 3. Quality flag exhaustiveness
        quality_counts = silver_df["quality_flag"].value_counts().to_dict()

        # 4. Financial Equation Balance (gross_sales - return_amount == net_revenue)
        s_gross = round(float(silver_df["gross_sales"].sum()), 2)
        s_return = round(float(silver_df["return_amount"].sum()), 2)
        s_net = round(float(silver_df["net_revenue"].sum()), 2)

        calculated_net = round(s_gross - s_return, 2)
        revenue_balanced = abs(calculated_net - s_net) < 0.05
        if not revenue_balanced:
            discrepancies.append(
                f"Revenue equation failure: Gross ({s_gross:,.2f}) - Return ({s_return:,.2f}) "
                f"= {calculated_net:,.2f} != Net ({s_net:,.2f})."
            )

        # Row-level revenue validation
        row_rev_diff = (
            (silver_df["gross_sales"] - silver_df["return_amount"])
            - silver_df["net_revenue"]
        ).abs()
        if (row_rev_diff > 1e-4).any():
            discrepancies.append("Row-level gross - return != net discrepancy detected.")

        # 5. Gold vs Silver Reconciliation
        gold_monthly_net = 0.0
        gold_country_net = 0.0
        gold_reconciled = True

        if gold_monthly_df is not None:
            gold_monthly_net = round(float(gold_monthly_df["net_revenue"].sum()), 2)
            if abs(gold_monthly_net - s_net) >= 0.005:  # Zero-penny tolerance
                gold_reconciled = False
                discrepancies.append(
                    f"Gold Monthly net revenue ({gold_monthly_net:,.2f}) != Silver net revenue ({s_net:,.2f})."
                )

        if gold_country_df is not None:
            gold_country_net = round(float(gold_country_df["net_revenue"].sum()), 2)
            if abs(gold_country_net - s_net) >= 0.005:  # Zero-penny tolerance
                gold_reconciled = False
                discrepancies.append(
                    f"Gold Country net revenue ({gold_country_net:,.2f}) != Silver net revenue ({s_net:,.2f})."
                )

        # 6. Anomaly Accounting
        anomalies_accounted = {
            "cancellations_in_silver": int(
                (silver_df["transaction_status"] == TransactionStatus.CANCELLATION).sum()
            ),
            "return_adjustments_in_silver": int(
                (
                    silver_df["transaction_status"]
                    == TransactionStatus.RETURN_ADJUSTMENT
                ).sum()
            ),
            "inventory_write_offs_in_silver": int(
                (
                    silver_df["transaction_status"]
                    == TransactionStatus.INVENTORY_WRITE_OFF
                ).sum()
            ),
            "zero_price_entries_in_silver": int(
                (
                    silver_df["transaction_status"]
                    == TransactionStatus.ZERO_PRICE_ENTRY
                ).sum()
            ),
            "duplicates_in_silver": int(
                (silver_df["transaction_status"] == TransactionStatus.DUPLICATE).sum()
            ),
            "missing_customers_in_silver": int(
                silver_df["is_missing_customer"].sum()
            ),
            "ambiguous_cancellations_in_silver": int(
                silver_df["is_ambiguous_cancellation"].sum()
            )
            if "is_ambiguous_cancellation" in silver_df.columns
            else 0,
        }

        all_passed = (
            row_balanced
            and status_exhaustive
            and revenue_balanced
            and gold_reconciled
            and len(discrepancies) == 0
        )

        report = ReconciliationReport(
            bronze_total_rows=b_count,
            silver_total_rows=s_count,
            is_row_count_balanced=row_balanced,
            status_breakdown=status_counts,
            status_sum=status_sum,
            is_status_exhaustive=status_exhaustive,
            quality_breakdown=quality_counts,
            silver_gross_sales_gbp=s_gross,
            silver_return_amount_gbp=s_return,
            silver_net_revenue_gbp=s_net,
            is_revenue_equation_balanced=revenue_balanced,
            gold_monthly_net_revenue_gbp=gold_monthly_net,
            gold_country_net_revenue_gbp=gold_country_net,
            is_gold_silver_reconciled=gold_reconciled,
            anomalies_accounted_for=anomalies_accounted,
            all_tests_passed=all_passed,
            discrepancies=discrepancies,
        )

        # Save to disk
        out_file = self.config.metadata_dir / "reconciliation_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(
            f"Reconciliation audit completed. All checks passed: {all_passed}. "
            f"Report saved to: {out_file}"
        )
        return report
