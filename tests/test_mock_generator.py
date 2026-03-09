"""
tests/test_mock_generator.py – Unit tests for the synthetic data generator.
"""

import pandas as pd

from data.mock_generator import generate_transactions


def test_generate_default_rows():
    """Default call should return exactly 5 rows."""
    df = generate_transactions()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_generate_custom_rows():
    """Custom n should return the requested number of rows."""
    df = generate_transactions(n=20)
    assert len(df) == 20


def test_required_columns_present():
    """All expected columns must be present in the output DataFrame."""
    required = {
        "transaction_id",
        "merchant_id",
        "amount",
        "currency",
        "status",
        "error_code",
        "payment_method",
        "customer_name",
        "timestamp",
    }
    df = generate_transactions(n=3)
    assert required.issubset(set(df.columns))


def test_error_code_only_on_failed():
    """error_code should be None for non-FAILED transactions."""
    df = generate_transactions(n=50)
    non_failed = df[df["status"] != "FAILED"]
    assert non_failed["error_code"].isna().all()


def test_amount_positive():
    """All generated amounts should be positive numbers."""
    df = generate_transactions(n=10)
    assert (df["amount"] > 0).all()
