# E-commerce Data Pipeline

A small Python project that generates realistic fake customer, product, and order data, then uses PySpark to calculate useful e-commerce metrics. Generated customer and order data is synthetic and intended for development and testing only.

## Project structure

```text
.
├── data/
│   ├── raw/                    # Generated customers.csv, products.csv, orders.csv
│   └── processed/              # Spark CSV output directories
├── notebooks/
│   └── analysis.ipynb          # Interactive quick start
├── src/
│   ├── __init__.py
│   ├── config.py               # Shared paths and defaults
│   ├── data_generator.py       # Faker data generation
│   └── spark_analytics.py      # PySpark analysis
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10 or later
- Java 17 or later for current PySpark 4.x releases (Java 17 is also suitable for PySpark 3.5)

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

From the repository root, generate the default dataset (100,000 customers, 10,000 products, and 1,000,000 orders):

```bash
python -m src.data_generator
```

Customize row counts, seed, or output location:

```bash
python -m src.data_generator --customers 500 --products 50 --orders 2000 --seed 123
```

Analyze the generated data with local Spark:

```bash
python -m src.spark_analytics
```

Optionally customize paths or the Spark master:

```bash
python -m src.spark_analytics --input-dir data/raw --output-dir data/processed --master "local[*]"
```

The analytics command writes four Spark CSV output directories under `data/processed/`:

- `monthly_trends` — monthly revenue and month-over-month revenue growth
- `category_revenue` — revenue and units sold by product category
- `top_customers` — up to 20 customers ranked by spend
- `top_products` — up to 20 products ranked by revenue

Spark writes each result as a directory containing one or more part files; this is Spark's normal CSV output format. Generated CSV data and analytics outputs are ignored by Git.

## Tests

```bash
python -m pytest
```

## Notebook

Start JupyterLab from the repository root and open `notebooks/analysis.ipynb`:

```bash
jupyter lab
```
