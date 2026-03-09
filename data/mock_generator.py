"""
data/mock_generator.py – Synthetic transaction log generator.

Uses the Faker library to produce realistic-looking (but entirely fake)
transaction records.  These records are used for:
  - Unit-testing the ML models without real payment data.
  - Populating demo dashboards during development.
  - Load-testing the API endpoints.
"""

from __future__ import annotations

import random
import uuid

import pandas as pd
from faker import Faker

# Seed for reproducible datasets during testing
_fake = Faker(locale="en_IN")   # Indian locale for INR amounts / names
Faker.seed(42)
random.seed(42)

# Possible values for categorical columns
_STATUSES = ["SUCCESS", "FAILED", "PENDING", "REFUNDED"]
_ERROR_CODES = [
    None,
    "INSUFFICIENT_FUNDS",
    "CARD_EXPIRED",
    "DO_NOT_HONOUR",
    "INVALID_CVV",
    "NETWORK_ERROR",
]
_PAYMENT_METHODS = ["UPI", "CARD", "NET_BANKING", "WALLET"]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def generate_transactions(n: int = 5) -> pd.DataFrame:
    """
    Generate ``n`` rows of synthetic transaction data.

    Each row contains:
      - ``transaction_id``:  UUID prefixed with "TXN-"
      - ``merchant_id``:     UUID prefixed with "MERCH-"
      - ``amount``:          Random float between ₹10 and ₹50,000
      - ``currency``:        Always "INR" for this demo
      - ``status``:          One of SUCCESS / FAILED / PENDING / REFUNDED
      - ``error_code``:      Gateway error code (only set when status is FAILED)
      - ``payment_method``:  UPI / CARD / NET_BANKING / WALLET
      - ``customer_name``:   Fake customer name
      - ``timestamp``:       Random datetime within the last 90 days

    Args:
        n: Number of transaction rows to generate (default: 5).

    Returns:
        A Pandas DataFrame with ``n`` rows and the columns described above.

    Example::

        >>> df = generate_transactions(n=10)
        >>> print(df.head())
    """
    records = []

    for _ in range(n):
        status = random.choice(_STATUSES)

        # Error codes only make sense when a transaction fails
        error_code = random.choice(_ERROR_CODES) if status == "FAILED" else None

        records.append(
            {
                "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
                "merchant_id": f"MERCH-{uuid.uuid4().hex[:6].upper()}",
                "amount": round(random.uniform(10.0, 50_000.0), 2),
                "currency": "INR",
                "status": status,
                "error_code": error_code,
                "payment_method": random.choice(_PAYMENT_METHODS),
                "customer_name": _fake.name(),
                "timestamp": _fake.date_time_between(
                    start_date="-90d", end_date="now"
                ).isoformat(),
            }
        )

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point – run directly to preview the generated data
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_transactions(n=5)
    print(df.to_string(index=False))
