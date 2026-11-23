import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.database.connection import DatabaseManager

from sqlalchemy import text

db = DatabaseManager()
with db.engine.connect() as conn:
    res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '%anomalies%'")).fetchall()
    table_names = [r[0] for r in res]
    print("Tables matching anomalies:", table_names)

    if "transaction_anomalies_isolation_forest" in table_names:
        df_pg = pd.DataFrame(conn.execute(text(
            """
            SELECT 
                transaction_status,
                COUNT(*) as total_count,
                SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count,
                ROUND(AVG(CASE WHEN is_anomaly THEN 1.0 ELSE 0.0 END) * 100, 4) as anomaly_pct
            FROM transaction_anomalies_isolation_forest
            GROUP BY transaction_status
            ORDER BY transaction_status
            """
        )).mappings().all())
        print("\nPostgreSQL Anomaly Counts by Status:")
        print(df_pg)
        
        total_pg = pd.DataFrame(conn.execute(text(
            """
            SELECT 
                COUNT(*) as total_rows,
                SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as total_anomalies
            FROM transaction_anomalies_isolation_forest
            """
        )).mappings().all())
        print("\nPostgreSQL Totals:")
        print(total_pg)
    else:
        print("Table gold_transaction_anomalies_isolation_forest does not exist yet in PostgreSQL.")
