"""Tests for fake e-commerce data generation."""

from pathlib import Path

import pytest

from src.data_generator import SyntheticDataGenerator, generate_data


def test_synthetic_generator_returns_valid_dataframes() -> None:
    """Generated DataFrames have the required fields, counts, and valid references."""
    generator = SyntheticDataGenerator(
        customer_count=10,
        product_count=4,
        order_count=25,
        seed=7,
        batch_size=6,
    )

    customers, products, orders = generator.generate_data()

    assert len(customers) == 10
    assert len(products) == 4
    assert len(orders) == 25
    assert customers.columns.tolist() == [
        "customer_id",
        "name",
        "email",
        "age",
        "city",
        "country",
        "registration_date",
    ]
    assert products.columns.tolist() == [
        "product_id",
        "name",
        "category",
        "price",
        "stock",
        "rating",
    ]
    assert orders.columns.tolist() == [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_date",
    ]
    assert orders["customer_id"].isin(customers["customer_id"]).all()
    assert orders["product_id"].isin(products["product_id"]).all()
    assert customers["age"].between(18, 90).all()
    assert products["category"].isin(
        {"Electronics", "Clothing", "Home", "Sports", "Books"}
    ).all()
    assert products["price"].between(10, 500).all()
    assert products["stock"].between(0, 1000).all()
    assert products["rating"].between(1, 5).all()
    assert orders["quantity"].between(1, 10).all()


def test_synthetic_generator_is_reproducible() -> None:
    """Repeated generation with the same seed produces identical DataFrames."""
    first = SyntheticDataGenerator(4, 3, 10, seed=19, batch_size=4).generate_data()
    second = SyntheticDataGenerator(4, 3, 10, seed=19, batch_size=4).generate_data()

    for first_frame, second_frame in zip(first, second):
        assert first_frame.equals(second_frame)


def test_generator_rejects_non_positive_counts() -> None:
    """Invalid dataset sizes fail clearly instead of creating empty data."""
    with pytest.raises(ValueError, match="must be positive"):
        SyntheticDataGenerator(customer_count=0)


def test_orders_require_at_least_two_customers() -> None:
    """The Pareto segment split needs both popular and long-tail customers."""
    generator = SyntheticDataGenerator(1, 1, 2, seed=2)
    customers = generator.generate_customers()
    products = generator.generate_products()

    with pytest.raises(ValueError, match="at least two customers"):
        generator.generate_orders(customers, products)


def test_generate_data_saves_csv_files(tmp_path: Path) -> None:
    """The legacy pipeline entry point writes all three CSV datasets."""
    paths = generate_data(3, 2, 5, output_dir=tmp_path, seed=11)

    assert set(paths) == {"customers", "products", "orders"}
    assert all(path.is_file() for path in paths.values())
