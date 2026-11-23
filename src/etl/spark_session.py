"""SparkSession builder with Delta Lake, Arrow optimization, and Java 21 compatibility."""
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger("ecommerce_platform.spark")


def get_spark_session(
    app_name: str = "EcommerceIntelligencePlatform",
    enable_delta: bool = True,
    driver_memory: str = "4g",
) -> Optional["pyspark.sql.SparkSession"]:  # type: ignore # noqa: F821
    """Constructs a local SparkSession configured for Delta Lake and PyArrow.

    Args:
        app_name: Name of the Spark application.
        enable_delta: Whether to enable Delta Lake extensions.
        driver_memory: Memory allocated to Spark driver.

    Returns:
        SparkSession instance or None if PySpark is unavailable.
    """
    try:
        import pyspark
        from pyspark.sql import SparkSession
    except ImportError:
        logger.warning("PySpark is not installed in the current environment.")
        return None

    # Ensure python executable is specified for worker processes on Windows (space-free 8.3 path)
    python_exec = sys.executable
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
            _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            _GetShortPathNameW.restype = wintypes.DWORD
            buf = ctypes.create_unicode_buffer(500)
            needed = _GetShortPathNameW(sys.executable, buf, 500)
            if 0 < needed < 500:
                python_exec = buf.value
        except Exception:
            pass
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec

    # Java 21+ JVM options for module reflection access
    java_opts = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
    )

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.driver.extraJavaOptions", java_opts)
        .config("spark.executor.extraJavaOptions", java_opts)
    )

    if enable_delta:
        try:
            from delta import configure_spark_with_delta_pip

            builder = configure_spark_with_delta_pip(builder)
            builder = (
                builder.config(
                    "spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension",
                ).config(
                    "spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog",
                )
            )
            logger.info("Delta Lake extensions configured for SparkSession.")
        except ImportError:
            logger.info(
                "delta-spark package not imported. Initializing standard SparkSession."
            )

    try:
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Initialized SparkSession successfully (Spark version: {spark.version})")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize SparkSession: {e}")
        return None
