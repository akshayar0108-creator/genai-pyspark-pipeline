"""PySpark analytics for synthetic e-commerce order data."""

import argparse
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import (
    col,
    count,
    date_format,
    lag,
    sum as spark_sum,
    when,
)
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.config import (
    CUSTOMERS_FILENAME,
    DEFAULT_SPARK_MASTER,
    ORDERS_FILENAME,
    PROCESSED_DATA_DIR,
    PRODUCTS_FILENAME,
    RAW_DATA_DIR,
)

LOGGER = logging.getLogger(__name__)

CUSTOMER_SCHEMA: StructType = StructType(
    [
        StructField("customer_id", IntegerType(), nullable=False),
        StructField("name", StringType(), nullable=True),
        StructField("email", StringType(), nullable=True),
        StructField("age", IntegerType(), nullable=True),
        StructField("city", StringType(), nullable=True),
        StructField("country", StringType(), nullable=True),
        StructField("registration_date", DateType(), nullable=True),
    ]
)
PRODUCT_SCHEMA: StructType = StructType(
    [
        StructField("product_id", IntegerType(), nullable=False),
        StructField("name", StringType(), nullable=True),
        StructField("category", StringType(), nullable=True),
        StructField("price", DoubleType(), nullable=True),
        StructField("stock", IntegerType(), nullable=True),
        StructField("rating", DoubleType(), nullable=True),
    ]
)
ORDER_SCHEMA: StructType = StructType(
    [
        StructField("order_id", IntegerType(), nullable=False),
        StructField("customer_id", IntegerType(), nullable=False),
        StructField("product_id", IntegerType(), nullable=False),
        StructField("quantity", IntegerType(), nullable=True),
        StructField("order_date", DateType(), nullable=True),
    ]
)


class SalesAnalytics:
    """Create a Spark session and calculate e-commerce sales metrics."""

    def __init__(self, spark: SparkSession | None = None) -> None:
        """Initialize analytics, optionally reusing an existing Spark session."""
        self.spark = spark

    def create_spark_session(
        self,
        app_name: str = "ecommerce-sales-analytics",
        master: str = DEFAULT_SPARK_MASTER,
    ) -> SparkSession:
        """Create a local Spark session with 4 GB driver memory, AQE, and Kryo.

        For local execution, the default master is ``local[*]``. Configure
        ``spark.driver.memory`` before Spark starts; Spark deployments may
        override it with their cluster-level settings.
        """
        self.spark = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.driver.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config(
                "spark.serializer",
                "org.apache.spark.serializer.KryoSerializer",
            )
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")
        LOGGER.info("Created Spark session for %s with master %s", app_name, master)
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet dataset from a local path or supported filesystem URI."""
        spark = self._require_spark()
        LOGGER.info("Loading Parquet dataset from %s", path)
        return spark.read.parquet(str(path))

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top N customers by revenue calculated from quantity and price."""
        if n < 1:
            raise ValueError("n must be a positive integer")
        revenue_by_customer = (
            orders_df.join(
                products_df.select("product_id", "price"),
                on="product_id",
                how="inner",
            )
            .withColumn("revenue", col("quantity") * col("price"))
            .groupBy("customer_id")
            .agg(spark_sum("revenue").alias("total_revenue"))
        )
        return revenue_by_customer.orderBy(col("total_revenue").desc()).limit(n)

    def sales_by_category(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Return revenue and units sold grouped by product category."""
        return (
            orders_df.join(
                products_df.select("product_id", "category", "price"),
                on="product_id",
                how="inner",
            )
            .withColumn("revenue", col("quantity") * col("price"))
            .groupBy("category")
            .agg(
                spark_sum("revenue").alias("total_revenue"),
                spark_sum("quantity").alias("units_sold"),
            )
            .orderBy(col("total_revenue").desc())
        )

    def monthly_trends(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Calculate monthly revenue and growth percentage versus the prior month.

        Growth is null for the first month and when the previous month's revenue
        is zero, since a percentage change cannot be calculated in those cases.
        """
        monthly_revenue = (
            orders_df.join(
                products_df.select("product_id", "price"),
                on="product_id",
                how="inner",
            )
            .withColumn("revenue", col("quantity") * col("price"))
            .withColumn("month", date_format(col("order_date"), "yyyy-MM"))
            .groupBy("month")
            .agg(spark_sum("revenue").alias("total_revenue"))
        )
        monthly_window = Window.orderBy("month")
        with_previous_month = monthly_revenue.withColumn(
            "previous_month_revenue",
            lag("total_revenue").over(monthly_window),
        )
        return (
            with_previous_month.withColumn(
                "growth_percentage",
                when(
                    col("previous_month_revenue").isNull()
                    | (col("previous_month_revenue") == 0),
                    None,
                ).otherwise(
                    (
                        (col("total_revenue") - col("previous_month_revenue"))
                        / col("previous_month_revenue")
                    )
                    * 100.0
                ),
            )
            .select(
                "month",
                "total_revenue",
                "previous_month_revenue",
                "growth_percentage",
            )
            .orderBy("month")
        )

    def _require_spark(self) -> SparkSession:
        """Return the configured session or report that one is required."""
        if self.spark is None:
            raise RuntimeError(
                "No Spark session is available; call create_spark_session() first"
            )
        return self.spark

    def stop(self) -> None:
        """Stop the associated Spark session if one has been created."""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None


def create_spark_session(
    app_name: str = "ecommerce-sales-analytics",
    master: str = DEFAULT_SPARK_MASTER,
) -> SparkSession:
    """Create a Spark session using the SalesAnalytics configuration."""
    return SalesAnalytics().create_spark_session(app_name=app_name, master=master)


def _read_csv(
    spark: SparkSession,
    file_path: Path,
    schema: StructType,
) -> DataFrame:
    """Read a headered CSV file using its declared Spark schema."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Required input CSV was not found: {file_path}")
    return (
        spark.read.option("header", "true")
        .option("dateFormat", "yyyy-MM-dd")
        .schema(schema)
        .csv(str(file_path))
    )


def analyze_orders(
    spark: SparkSession,
    input_dir: Path = RAW_DATA_DIR,
) -> dict[str, DataFrame]:
    """Build monthly trends, category sales, top customers, and top products."""
    orders = _read_csv(spark, input_dir / ORDERS_FILENAME, ORDER_SCHEMA)
    products = _read_csv(spark, input_dir / PRODUCTS_FILENAME, PRODUCT_SCHEMA)
    customers = _read_csv(spark, input_dir / CUSTOMERS_FILENAME, CUSTOMER_SCHEMA)
    analytics = SalesAnalytics(spark)

    top_customers = (
        analytics.top_customers_by_revenue(orders, products, n=20)
        .join(customers.select("customer_id", "name"), on="customer_id", how="inner")
        .select("customer_id", "name", "total_revenue")
    )
    top_products = (
        orders.join(
            products.select("product_id", "name", "category", "price"),
            on="product_id",
            how="inner",
        )
        .withColumn("revenue", col("quantity") * col("price"))
        .groupBy("product_id", "name", "category")
        .agg(
            count("order_id").alias("order_count"),
            spark_sum("quantity").alias("units_sold"),
            spark_sum("revenue").alias("total_revenue"),
        )
        .orderBy(col("total_revenue").desc())
        .limit(20)
    )
    return {
        "monthly_trends": analytics.monthly_trends(orders, products),
        "category_revenue": analytics.sales_by_category(orders, products),
        "top_customers": top_customers,
        "top_products": top_products,
    }


def run_analytics(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    master: str = DEFAULT_SPARK_MASTER,
) -> dict[str, Path]:
    """Run all analyses, save results as Spark CSV output directories, and stop Spark."""
    analytics = SalesAnalytics()
    spark = analytics.create_spark_session(master=master)
    try:
        results = analyze_orders(spark, input_dir=input_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}
        for name, result in results.items():
            path = output_dir / name
            result.write.mode("overwrite").option("header", "true").csv(str(path))
            LOGGER.info("Wrote %s insights to %s", name, path)
            output_paths[name] = path
        return output_paths
    finally:
        analytics.stop()


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for Spark analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--master", default=DEFAULT_SPARK_MASTER)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """Run the PySpark analytics command-line interface."""
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_analytics(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        master=args.master,
    )


if __name__ == "__main__":
    main()
