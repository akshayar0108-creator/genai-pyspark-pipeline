"""Generate synthetic e-commerce DataFrames and optionally save them as CSV."""

import argparse
import logging
import math
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import (
    CUSTOMERS_FILENAME,
    DEFAULT_SEED,
    RAW_DATA_DIR,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CUSTOMER_COUNT: Final[int] = 100_000
DEFAULT_PRODUCT_COUNT: Final[int] = 10_000
DEFAULT_ORDER_COUNT: Final[int] = 1_000_000
DEFAULT_BATCH_SIZE: Final[int] = 10_000
PRODUCT_CATEGORIES: Final[tuple[str, ...]] = (
    "Electronics",
    "Clothing",
    "Home",
    "Sports",
    "Books",
)


class SyntheticDataGenerator:
    """Create reproducible synthetic e-commerce customer, product, and order data."""

    def __init__(
        self,
        customer_count: int = DEFAULT_CUSTOMER_COUNT,
        product_count: int = DEFAULT_PRODUCT_COUNT,
        order_count: int = DEFAULT_ORDER_COUNT,
        seed: int = DEFAULT_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Set dataset sizes and initialize seeded Faker and NumPy generators."""
        if customer_count < 1 or product_count < 1 or order_count < 1:
            raise ValueError("customer_count, product_count, and order_count must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.customer_count = customer_count
        self.product_count = product_count
        self.order_count = order_count
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._fake = Faker()
        self._fake.seed_instance(seed)

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer IDs, Faker names/emails, normally distributed ages, and locations."""
        LOGGER.info("Generating %d customers", self.customer_count)
        today = date.today()
        records: list[dict[str, object]] = []
        for customer_id in tqdm(
            range(1, self.customer_count + 1),
            desc="Generating customers",
            unit="customer",
        ):
            registration_date = self._fake.date_between(
                start_date="-5y",
                end_date=today,
            )
            records.append(
                {
                    "customer_id": customer_id,
                    "name": self._fake.name(),
                    "email": self._fake.unique.email(),
                    "age": int(
                        np.clip(
                            round(self._rng.normal(loc=40.0, scale=12.0)),
                            18,
                            90,
                        )
                    ),
                    "city": self._fake.city(),
                    "country": self._fake.country(),
                    "registration_date": registration_date,
                }
            )
        return pd.DataFrame.from_records(
            records,
            columns=[
                "customer_id",
                "name",
                "email",
                "age",
                "city",
                "country",
                "registration_date",
            ],
        )

    def generate_products(self) -> pd.DataFrame:
        """Generate products with category, price, stock, and a 1-to-5 rating."""
        LOGGER.info("Generating %d products", self.product_count)
        records: list[dict[str, object]] = []
        for product_id in tqdm(
            range(1, self.product_count + 1),
            desc="Generating products",
            unit="product",
        ):
            records.append(
                {
                    "product_id": product_id,
                    "name": self._fake.catch_phrase(),
                    "category": self._rng.choice(PRODUCT_CATEGORIES),
                    "price": round(float(self._rng.uniform(10.0, 500.0)), 2),
                    "stock": int(self._rng.integers(0, 1001)),
                    "rating": round(float(self._rng.uniform(1.0, 5.0)), 1),
                }
            )
        return pd.DataFrame.from_records(
            records,
            columns=["product_id", "name", "category", "price", "stock", "rating"],
        )

    def generate_orders(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate orders in batches with an exact 80/20 customer-segment order split.

        Pareto-distributed weights skew orders within each segment. The top 20%
        of customers receive 80% of orders; the remaining customers share 20%.
        """
        if customers.empty or "customer_id" not in customers.columns:
            raise ValueError("customers must contain at least one customer_id")
        if products.empty or "product_id" not in products.columns:
            raise ValueError("products must contain at least one product_id")

        customer_ids = customers["customer_id"].to_numpy(dtype=np.int64)
        product_ids = products["product_id"].to_numpy(dtype=np.int64)
        if len(customer_ids) < 2:
            raise ValueError("at least two customers are required for an 80/20 order split")
        popular_count = max(1, math.ceil(len(customer_ids) * 0.2))
        pareto_order = np.argsort(self._rng.pareto(a=1.16, size=len(customer_ids)))
        popular_customers = customer_ids[pareto_order[-popular_count:]]
        other_customers = customer_ids[pareto_order[:-popular_count]]
        popular_weights = 1.0 + self._rng.pareto(a=1.16, size=len(popular_customers))
        other_weights = 1.0 + self._rng.pareto(
            a=1.16,
            size=max(1, len(other_customers)),
        )
        popular_order_count = int(self.order_count * 0.8)
        first_order_date = np.datetime64("today", "D") - np.timedelta64(365, "D")
        batches: list[pd.DataFrame] = []

        LOGGER.info(
            "Generating %d orders; assigning %d (80%%) to %d Pareto-weighted customers",
            self.order_count,
            popular_order_count,
            popular_count,
        )
        for start in tqdm(
            range(0, self.order_count, self.batch_size),
            desc="Generating orders",
            unit="batch",
        ):
            stop = min(start + self.batch_size, self.order_count)
            batch_popular_count = max(
                0,
                min(stop, popular_order_count) - start,
            )
            batch_other_count = stop - start - batch_popular_count
            batch_customer_ids = np.concatenate(
                [
                    self._rng.choice(
                        popular_customers,
                        size=batch_popular_count,
                        replace=True,
                        p=popular_weights / popular_weights.sum(),
                    ),
                    self._rng.choice(
                        other_customers if len(other_customers) else popular_customers,
                        size=batch_other_count,
                        replace=True,
                        p=(
                            other_weights / other_weights.sum()
                            if len(other_customers)
                            else popular_weights / popular_weights.sum()
                        ),
                    ),
                ]
            )
            day_offsets = self._rng.integers(0, 366, size=stop - start)
            batches.append(
                pd.DataFrame(
                    {
                        "order_id": np.arange(start + 1, stop + 1, dtype=np.int64),
                        "customer_id": batch_customer_ids,
                        "product_id": self._rng.choice(
                            product_ids,
                            size=stop - start,
                            replace=True,
                        ),
                        "quantity": self._rng.integers(1, 11, size=stop - start),
                        "order_date": pd.to_datetime(
                            first_order_date + day_offsets.astype("timedelta64[D]")
                        ),
                    }
                )
            )
        return pd.concat(batches, ignore_index=True)

    def generate_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customer, product, and order DataFrames in that order."""
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(customers, products)
        LOGGER.info("Generated all synthetic e-commerce datasets")
        return customers, products, orders


def generate_data(
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    output_dir: Path = RAW_DATA_DIR,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Generate datasets and save them as CSV files, returning their paths."""
    generator = SyntheticDataGenerator(
        customer_count=customer_count,
        product_count=product_count,
        order_count=order_count,
        seed=seed,
    )
    customers, products, orders = generator.generate_data()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "customers": output_dir / CUSTOMERS_FILENAME,
        "products": output_dir / "products.csv",
        "orders": output_dir / "orders.csv",
    }
    customers.to_csv(paths["customers"], index=False)
    products.to_csv(paths["products"], index=False)
    orders.to_csv(paths["orders"], index=False)
    LOGGER.info("Saved generated CSV datasets to %s", output_dir)
    return paths


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for data generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMER_COUNT)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--orders", type=int, default=DEFAULT_ORDER_COUNT)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """Run the CSV data-generation command-line interface."""
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    generate_data(
        customer_count=args.customers,
        product_count=args.products,
        order_count=args.orders,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
