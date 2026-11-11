"""Load Parquet datasets and run the e-commerce sales analytics."""

import logging
from pathlib import Path
from time import perf_counter

from pyspark.sql import DataFrame

from src.spark_analytics import SalesAnalytics

RAW_DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


def _show_timed(label: str, result: DataFrame) -> None:
    """Display a DataFrame and print elapsed time, including Spark execution."""
    started_at = perf_counter()
    print(f"\n{label}")
    result.show(truncate=False)
    elapsed = perf_counter() - started_at
    print(f"{label} execution time: {elapsed:.2f} seconds")


def main() -> None:
    """Run all analytics against customers, products, and orders Parquet files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    analytics = SalesAnalytics()
    try:
        spark = analytics.create_spark_session()

        datasets: dict[str, DataFrame] = {}
        for name in ("customers", "products", "orders"):
            started_at = perf_counter()
            path = RAW_DATA_DIR / f"{name}.parquet"
            datasets[name] = analytics.load_parquet(path)
            print(f"Loaded {path.name} in {perf_counter() - started_at:.2f} seconds")

        top_customers = analytics.top_customers_by_revenue(
            datasets["orders"],
            datasets["products"],
        )
        if "name" in datasets["customers"].columns:
            top_customers = top_customers.join(
                datasets["customers"].select("customer_id", "name"),
                on="customer_id",
                how="left",
            ).select("customer_id", "name", "total_revenue")
        _show_timed("Top customers by revenue", top_customers)

        category_sales = analytics.sales_by_category(
            datasets["orders"],
            datasets["products"],
        )
        _show_timed("Sales by category", category_sales)

        monthly_trends = analytics.monthly_trends(
            datasets["orders"],
            datasets["products"],
        )
        _show_timed("Monthly revenue trends", monthly_trends)
    finally:
        analytics.stop()


if __name__ == "__main__":
    main()
