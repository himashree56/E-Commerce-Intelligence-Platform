# E-Commerce Intelligence Platform - Architecture

## 1. Executive Summary
The E-Commerce Intelligence Platform is an enterprise-grade lakehouse and predictive analytics system designed to process transaction-level e-commerce data from the UCI Online Retail II dataset, enrich customer and product dimensions, extract behavioral features, and train machine learning models for customer segmentation, churn forecasting, and demand planning.

## 2. Medallion Lakehouse Architecture

```mermaid
flowchart TD
    subgraph Raw ["1. Raw Data Layer"]
        A["UCI Online Retail II (.xlsx / .zip)"]
    end

    subgraph Bronze ["2. Bronze Layer (Phase 1)"]
        B["Raw Multi-Sheet Ingestion"]
        C["Schema Standardization & Type Coercion"]
        D["Preserve Anomalies (Cancellations, Neg Qty, Null Customers)"]
        E["Parquet Storage (Snappy, PyArrow)"]
        F["Structured Ingestion Metadata (JSON)"]
        A --> B --> C --> D --> E
        D --> F
    end

    subgraph Silver ["3. Silver Layer (Phase 2)"]
        G["Data Cleansing & Validation"]
        H["Cancellations Reconciliation"]
        I["Customer Entity Resolution"]
        J["Curated Fact & Dimension Tables"]
        E --> G --> H --> I --> J
    end

    subgraph Gold ["4. Gold Layer (Phase 3 & 4)"]
        K["RFM Feature Store"]
        L["Customer Lifetime Value (CLV)"]
        M["Churn Risk Models (XGBoost)"]
        N["Customer Segmentation (K-Means)"]
        J --> K --> L
        J --> M
        J --> N
    end

    subgraph Serving ["5. Serving & Presentation"]
        O[("PostgreSQL Analytics Database")]
        P["Streamlit Executive Dashboard"]
        Q["Databricks Lakehouse Analytics"]
        J --> O
        K --> O
        M --> P
        N --> P
        O --> P
        E --> Q
    end
```

## 3. Technology Stack Justification
- **Python 3.12**: Modern language runtime with fast performance, typed dicts, and strict type safety.
- **Pandas & NumPy**: In-memory wrangling and vector math for sheet parsing and initial audit.
- **PyArrow & Parquet**: Columnar format with schema enforcement, Snappy compression, statistics metadata, and cross-platform compatibility.
- **PostgreSQL**: Serving store for dimensional tables, user access, and reporting queries.
- **PySpark & Databricks**: Distributed big data processing for lakehouse operations at scale.
- **Scikit-learn & XGBoost**: Production machine learning for segmentation and customer churn forecasting.
- **Streamlit**: Responsive executive analytics interface.
- **Docker**: Containerized deployment ensuring reproducibility across operating systems.
- **pytest**: Rigorous test-driven verification across all pipeline components.
