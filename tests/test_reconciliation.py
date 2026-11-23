"""Reconciliation tests verifying that 100% of Bronze records are accounted for."""
import pandas as pd
import pytest

from src.etl.config import ETLConfig
from src.etl.gold import GoldTransformer
from src.etl.reconciliation import ReconciliationAuditor
from src.etl.silver import SilverTransformer


def test_reconciliation_audit(mock_bronze_records):
    """Test full reconciliation audit passes on mock dataset."""
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(mock_bronze_records)

    g_transformer = GoldTransformer(ETLConfig())
    monthly_df = g_transformer.build_monthly_sales(silver_df)
    country_df = g_transformer.build_country_revenue(silver_df)

    auditor = ReconciliationAuditor(ETLConfig())
    report = auditor.audit(
        bronze_df=mock_bronze_records,
        silver_df=silver_df,
        gold_monthly_df=monthly_df,
        gold_country_df=country_df,
    )

    assert report.all_tests_passed is True
    assert report.is_row_count_balanced is True
    assert report.is_status_exhaustive is True
    assert report.is_revenue_equation_balanced is True
    assert report.is_gold_silver_reconciled is True
    assert len(report.discrepancies) == 0


def test_reconciliation_detects_row_count_mismatch(mock_bronze_records):
    """Test that reconciliation fails if any Bronze record is dropped in Silver."""
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(mock_bronze_records)

    # Silently drop 1 record from Silver
    corrupted_silver = silver_df.iloc[:-1].copy()

    auditor = ReconciliationAuditor(ETLConfig())
    report = auditor.audit(bronze_df=mock_bronze_records, silver_df=corrupted_silver)

    assert report.is_row_count_balanced is False
    assert report.all_tests_passed is False
    assert any("Row count mismatch" in d for d in report.discrepancies)


def test_reconciliation_detects_revenue_mismatch(mock_bronze_records):
    """Test that reconciliation fails if gross - return != net revenue."""
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(mock_bronze_records)

    # Corrupt net_revenue in Silver
    corrupted_silver = silver_df.copy()
    corrupted_silver.loc[0, "net_revenue"] += 999.0

    auditor = ReconciliationAuditor(ETLConfig())
    report = auditor.audit(bronze_df=mock_bronze_records, silver_df=corrupted_silver)

    assert report.is_revenue_equation_balanced is False
    assert report.all_tests_passed is False
    assert any("Revenue equation failure" in d for d in report.discrepancies)


def test_reconciliation_detects_gold_silver_mismatch(mock_bronze_records):
    """Test that reconciliation fails if Gold totals do not match Silver totals."""
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(mock_bronze_records)

    g_transformer = GoldTransformer(ETLConfig())
    monthly_df = g_transformer.build_monthly_sales(silver_df)
    country_df = g_transformer.build_country_revenue(silver_df)

    # Corrupt monthly table
    corrupted_monthly = monthly_df.copy()
    corrupted_monthly.loc[0, "net_revenue"] += 500.0

    auditor = ReconciliationAuditor(ETLConfig())
    report = auditor.audit(
        bronze_df=mock_bronze_records,
        silver_df=silver_df,
        gold_monthly_df=corrupted_monthly,
        gold_country_df=country_df,
    )

    assert report.is_gold_silver_reconciled is False
    assert report.all_tests_passed is False
    assert any("Gold Monthly net revenue" in d for d in report.discrepancies)


def test_reconciliation_zero_penny_tolerance_enforcement(mock_bronze_records):
    """Test that even a single penny discrepancy (£0.01) triggers a reconciliation failure."""
    s_transformer = SilverTransformer(ETLConfig())
    silver_df = s_transformer.transform_dataframe(mock_bronze_records)

    g_transformer = GoldTransformer(ETLConfig())
    monthly_df = g_transformer.build_monthly_sales(silver_df)
    country_df = g_transformer.build_country_revenue(silver_df)

    # Induce exactly £0.01 divergence into monthly table
    divergent_monthly = monthly_df.copy()
    divergent_monthly.loc[0, "net_revenue"] += 0.01

    auditor = ReconciliationAuditor(ETLConfig())
    report = auditor.audit(
        bronze_df=mock_bronze_records,
        silver_df=silver_df,
        gold_monthly_df=divergent_monthly,
        gold_country_df=country_df,
    )

    assert report.is_gold_silver_reconciled is False
    assert report.all_tests_passed is False
    assert any("Gold Monthly net revenue" in d for d in report.discrepancies)
