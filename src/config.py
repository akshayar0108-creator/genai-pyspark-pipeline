"""Shared paths and defaults for the e-commerce data pipeline."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

CUSTOMERS_FILENAME: str = "customers.csv"
PRODUCTS_FILENAME: str = "products.csv"
ORDERS_FILENAME: str = "orders.csv"

DEFAULT_SEED: int = 42
DEFAULT_CUSTOMER_COUNT: int = 1_000
DEFAULT_PRODUCT_COUNT: int = 100
DEFAULT_ORDER_COUNT: int = 5_000
DEFAULT_SPARK_MASTER: str = "local[*]"
