"""
generate_realistic_data.py – Statistically realistic synthetic payment-gateway
                              data generator for the Intelligent Merchant Support
                              AI Agent project.

Generates three relational CSV files in ``data/output/``:

  • merchants.csv       – 25 unique merchant profiles
  • transactions.csv    – ~50,000 transactions over a 7-day rolling window
  • webhook_logs.csv    – One webhook delivery record per transaction

Statistical features
--------------------
* **Pareto distribution** for merchant transaction volumes (heavy-tail: a few
  merchants generate the majority of traffic) and for transaction amounts
  (heavy-tail: most amounts cluster low, with occasional large payments).
* **MCC-based amount profiles**: each Merchant Category Code has its own
  Pareto shape and scale parameters, producing realistic amount ranges
  (e.g., grocery ₹300–₹2k vs. electronics ₹2k–₹30k).  All amounts are
  clipped to a minimum of ₹50 to prevent unrealistic tiny values.
* **Time-of-day seasonality**: transaction timestamps follow an hourly
  weight curve modelling IST business hours (dual peaks at ~10 am–2 pm and
  ~5 pm–9 pm IST), with low activity at night.

CRITICAL – Preserved Demo Anomalies
------------------------------------
Three exact rows are injected into the final DataFrames so the Streamlit UI
hero-prompt tests and AppTest suite continue to pass:

  1. ``TXN-00194400`` → ``merchant_id_1``, DECLINED, ``93_Risk_Block``
  2. ``TXN-00000004`` → ``merchant_id_5``, SUCCESS;
     webhook ``WH-00000004`` with ``http_status=500``
  3. ``merchant_id_2`` → 150 consecutive webhook logs in a 1-hour window
     with ``http_status=401`` (auth-failure simulation)

Run directly::

    python generate_realistic_data.py
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED: int = 42
_rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)
_fake = Faker(locale="en_IN")
Faker.seed(SEED)

# ── Time window ───────────────────────────────────────────────────────────────
NOW: datetime = datetime.now(tz=timezone.utc).replace(microsecond=0)
WINDOW_DAYS: int = 7
WINDOW_START: datetime = NOW - timedelta(days=WINDOW_DAYS)
TARGET_TRANSACTIONS: int = 50_000
NUM_MERCHANTS: int = 25

# ── Lookup tables ─────────────────────────────────────────────────────────────
_MCC_CODES: list[str] = [
    "5411",  # Grocery stores
    "5812",  # Eating places & restaurants
    "5999",  # Miscellaneous retail
    "7372",  # Computer programming / software
    "5045",  # Computers & peripherals
    "4812",  # Telecom equipment
    "5912",  # Drug stores & pharmacies
    "7011",  # Hotels & lodging
    "5311",  # Department stores
    "5661",  # Shoe stores
]

# MCC → (pareto_alpha, scale_factor)
# Generated amount = (pareto(alpha) + 1) × scale, clipped to [50, ∞).
MCC_AMOUNT_PARAMS: dict[str, tuple[float, float]] = {
    "5411": (3.0, 300),    # Grocery: typical ₹300–₹2,000
    "5812": (2.8, 250),    # Restaurants: ₹250–₹1,500
    "5999": (2.0, 500),    # Misc retail: ₹500–₹5,000
    "7372": (1.8, 1500),   # Software: ₹1,500–₹20,000
    "5045": (1.5, 2000),   # Electronics: ₹2,000–₹30,000
    "4812": (2.5, 400),    # Telecom: ₹400–₹2,500
    "5912": (3.0, 200),    # Pharmacy: ₹200–₹1,000
    "7011": (2.0, 1000),   # Hotels: ₹1,000–₹15,000
    "5311": (2.0, 800),    # Department stores: ₹800–₹10,000
    "5661": (2.5, 600),    # Shoes: ₹600–₹4,000
}
_DEFAULT_AMOUNT_PARAMS: tuple[float, float] = (2.0, 500)

_NORMAL_DECLINE_CODES: list[str] = [
    "05_Do_Not_Honor",
    "51_Insufficient_Funds",
    "14_Invalid_Card_Number",
    "54_Expired_Card",
    "57_Transaction_Not_Permitted",
]

_ALL_BINS: list[str] = [
    "400011", "411000", "424242", "512345",
    "601100", "371449", "378282", "601782",
    "411111",
]
_BIN_WEIGHTS: list[float] = [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.02]

# Hourly traffic weights (UTC hours 0–23).
# IST = UTC + 5:30 → IST business-hour peaks map to approximately UTC 4–15.
_HOURLY_WEIGHTS: np.ndarray = np.array([
    0.30, 0.20, 0.20, 0.40, 0.70,  # UTC 0–4   (IST ~5:30–9:30 am)
    0.90, 1.00, 1.00, 1.00, 0.90,  # UTC 5–9   (IST ~10:30 am–2:30 pm)
    0.70, 0.80, 0.90, 1.00, 1.00,  # UTC 10–14 (IST ~3:30–7:30 pm)
    0.80, 0.60, 0.50, 0.40, 0.30,  # UTC 15–19 (IST ~8:30 pm–12:30 am)
    0.20, 0.20, 0.20, 0.30,        # UTC 20–23 (IST ~1:30–4:30 am)
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merchant_id(i: int) -> str:
    """Return the stable, human-readable merchant ID for 1-based index *i*."""
    return f"merchant_id_{i}"


def _sample_seasonal_timestamps(n: int) -> list[datetime]:
    """Return *n* timestamps sampled from the 7-day window with hourly seasonality."""
    total_hours = WINDOW_DAYS * 24  # 168

    # Tile the 24-hour weight pattern across all days, then normalise
    hour_probs = np.tile(_HOURLY_WEIGHTS, WINDOW_DAYS)
    hour_probs = hour_probs / hour_probs.sum()

    sampled_hours = _rng.choice(total_hours, size=n, p=hour_probs)
    second_offsets = _rng.integers(0, 3600, size=n)

    timestamps = [
        WINDOW_START + timedelta(hours=int(h), seconds=int(s))
        for h, s in zip(sampled_hours, second_offsets)
    ]
    return sorted(timestamps)


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 – Merchants
# ──────────────────────────────────────────────────────────────────────────────

def generate_merchants(n: int = NUM_MERCHANTS) -> pd.DataFrame:
    """
    Generate *n* unique merchant profiles.

    Columns: merchant_id, business_name, mcc_code, webhook_url
    """
    records = [
        {
            "merchant_id": _merchant_id(i),
            "business_name": _fake.company(),
            "mcc_code": random.choice(_MCC_CODES),
            "webhook_url": f"https://hooks.{_fake.domain_name()}/payment/events",
        }
        for i in range(1, n + 1)
    ]
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 – Transactions
# ──────────────────────────────────────────────────────────────────────────────

def generate_transactions(merchants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate ~50,000 transactions with Pareto merchant volumes,
    MCC-based Pareto amounts, and time-of-day seasonality.

    Columns: transaction_id, merchant_id, timestamp, amount, currency,
             status, decline_code, card_bin
    """
    merchant_ids = merchants_df["merchant_id"].tolist()
    n_merchants = len(merchant_ids)
    mcc_lookup = dict(zip(merchants_df["merchant_id"], merchants_df["mcc_code"]))

    # ── Pareto-based merchant assignment ──────────────────────────────────
    pareto_raw = _rng.pareto(a=1.16, size=n_merchants) + 1.0
    pareto_probs = pareto_raw / pareto_raw.sum()
    merchant_assignments = _rng.choice(
        merchant_ids, size=TARGET_TRANSACTIONS, p=pareto_probs,
    )

    # ── Seasonal timestamps ───────────────────────────────────────────────
    timestamps = _sample_seasonal_timestamps(TARGET_TRANSACTIONS)

    # ── MCC-based Pareto amounts (vectorised per MCC) ─────────────────────
    amounts = np.empty(TARGET_TRANSACTIONS)
    merchant_mccs = np.array([mcc_lookup[m] for m in merchant_assignments])
    for mcc in np.unique(merchant_mccs):
        mask = merchant_mccs == mcc
        n_mcc = int(mask.sum())
        alpha, scale = MCC_AMOUNT_PARAMS.get(mcc, _DEFAULT_AMOUNT_PARAMS)
        amounts[mask] = (_rng.pareto(a=alpha, size=n_mcc) + 1.0) * scale
    amounts = np.clip(amounts, 50, None).round(2)

    # ── Status & decline codes (~15 % decline rate) ───────────────────────
    is_declined = _rng.random(TARGET_TRANSACTIONS) > 0.85
    statuses = np.where(is_declined, "DECLINED", "SUCCESS")
    decline_codes: list[str | None] = [
        random.choice(_NORMAL_DECLINE_CODES) if is_declined[i] else None
        for i in range(TARGET_TRANSACTIONS)
    ]

    # ── Card BINs ─────────────────────────────────────────────────────────
    card_bins = random.choices(_ALL_BINS, weights=_BIN_WEIGHTS, k=TARGET_TRANSACTIONS)

    df = pd.DataFrame({
        "merchant_id": merchant_assignments,
        "timestamp": timestamps,
        "amount": amounts,
        "currency": "INR",
        "status": statuses,
        "decline_code": decline_codes,
        "card_bin": card_bins,
    })

    # Sort by timestamp, then assign sequential IDs
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN-{i:08d}" for i in range(len(df))])

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – Webhook Logs
# ──────────────────────────────────────────────────────────────────────────────

def generate_webhook_logs(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Generate one webhook delivery record per transaction.

    Columns: log_id, transaction_id, timestamp, event_type,
             http_status, delivery_attempts, latency_ms
    """
    n = len(transactions)

    event_map = {"SUCCESS": "payment.success", "DECLINED": "payment.failed"}
    event_types = transactions["status"].map(event_map).fillna("payment.pending")

    normal_statuses = random.choices([200, 200, 200, 200, 400, 500], k=n)
    latency_ms = _rng.integers(50, 501, size=n).tolist()
    delivery_attempts = [
        1 if s == 200 else random.randint(1, 3) for s in normal_statuses
    ]

    delays = _rng.integers(1, 31, size=n)
    wh_timestamps = [
        ts + timedelta(seconds=int(d))
        for ts, d in zip(transactions["timestamp"], delays)
    ]

    df = pd.DataFrame({
        "log_id": [f"WH-{i:08d}" for i in range(n)],
        "transaction_id": transactions["transaction_id"].values,
        "timestamp": wh_timestamps,
        "event_type": event_types.values,
        "http_status": normal_statuses,
        "delivery_attempts": delivery_attempts,
        "latency_ms": latency_ms,
    })
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – Demo anomaly injection
# ──────────────────────────────────────────────────────────────────────────────

def inject_demo_anomalies(
    transactions: pd.DataFrame,
    webhooks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject three critical demo rows that the Streamlit hero-prompt tests
    and UI AppTest suite depend on.

    1. TXN-00194400 → merchant_id_1, DECLINED, 93_Risk_Block
    2. TXN-00000004 → merchant_id_5, SUCCESS; WH-00000004 → http_status=500
    3. merchant_id_2 → 150 webhook logs with http_status=401 in a 1-hour block
    """

    # ── Demo 1: TXN-00194400 → merchant_id_1, DECLINED, 93_Risk_Block ────
    demo_ts = NOW - timedelta(hours=6)
    demo_txn = pd.DataFrame([{
        "transaction_id": "TXN-00194400",
        "merchant_id": "merchant_id_1",
        "timestamp": demo_ts,
        "amount": 15000.00,
        "currency": "INR",
        "status": "DECLINED",
        "decline_code": "93_Risk_Block",
        "card_bin": "411000",
    }])
    demo_wh = pd.DataFrame([{
        "log_id": "WH-00194400",
        "transaction_id": "TXN-00194400",
        "timestamp": demo_ts + timedelta(seconds=5),
        "event_type": "payment.failed",
        "http_status": 200,
        "delivery_attempts": 1,
        "latency_ms": 150,
    }])
    transactions = pd.concat([transactions, demo_txn], ignore_index=True)
    webhooks = pd.concat([webhooks, demo_wh], ignore_index=True)

    # ── Demo 2: TXN-00000004 → merchant_id_5, SUCCESS ────────────────────
    #            WH-00000004 → http_status=500 (dropped notification)
    mask_txn = transactions["transaction_id"] == "TXN-00000004"
    transactions.loc[mask_txn, "merchant_id"] = "merchant_id_5"
    transactions.loc[mask_txn, "status"] = "SUCCESS"
    transactions.loc[mask_txn, "decline_code"] = None

    mask_wh = webhooks["transaction_id"] == "TXN-00000004"
    webhooks.loc[mask_wh, "http_status"] = 500
    webhooks.loc[mask_wh, "event_type"] = "payment.success"
    webhooks.loc[mask_wh, "delivery_attempts"] = 3

    # ── Demo 3: merchant_id_2 – 150 webhook logs with 401 in 1-hour block ─
    a2_start = NOW - timedelta(hours=3)
    a2_offsets = sorted(_rng.integers(0, 3600, size=150))
    a2_timestamps = [a2_start + timedelta(seconds=int(s)) for s in a2_offsets]
    a2_amounts = np.clip(
        (_rng.pareto(a=2.0, size=150) + 1.0) * 500, 50, None,
    ).round(2)
    a2_is_declined = _rng.random(150) > 0.85
    a2_statuses = np.where(a2_is_declined, "DECLINED", "SUCCESS")
    a2_decline_codes: list[str | None] = [
        random.choice(_NORMAL_DECLINE_CODES) if a2_is_declined[i] else None
        for i in range(150)
    ]
    a2_txn_ids = [f"TXN-AUTH-{i:05d}" for i in range(150)]

    a2_txns = pd.DataFrame({
        "transaction_id": a2_txn_ids,
        "merchant_id": "merchant_id_2",
        "timestamp": a2_timestamps,
        "amount": a2_amounts,
        "currency": "INR",
        "status": a2_statuses,
        "decline_code": a2_decline_codes,
        "card_bin": random.choices(_ALL_BINS, weights=_BIN_WEIGHTS, k=150),
    })
    transactions = pd.concat([transactions, a2_txns], ignore_index=True)

    a2_wh_timestamps = [
        ts + timedelta(seconds=random.randint(1, 30)) for ts in a2_timestamps
    ]
    a2_wh = pd.DataFrame({
        "log_id": [f"WH-AUTH-{i:05d}" for i in range(150)],
        "transaction_id": a2_txn_ids,
        "timestamp": a2_wh_timestamps,
        "event_type": pd.Series(a2_statuses).map(
            {"SUCCESS": "payment.success", "DECLINED": "payment.failed"}
        ).fillna("payment.pending").values,
        "http_status": 401,
        "delivery_attempts": [random.randint(1, 3) for _ in range(150)],
        "latency_ms": _rng.integers(50, 501, size=150).tolist(),
    })
    webhooks = pd.concat([webhooks, a2_wh], ignore_index=True)

    return transactions, webhooks


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 – CSV reading helper
# ──────────────────────────────────────────────────────────────────────────────

def read_transactions_csv(path: str | Path) -> pd.DataFrame:
    """
    Read ``transactions.csv`` preserving ``card_bin`` as a string column.

    Always use this helper instead of a raw ``pd.read_csv()`` call when you
    need to filter or compare ``card_bin`` values.
    """
    return pd.read_csv(path, dtype={"card_bin": str})


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 – Orchestration & CSV export
# ──────────────────────────────────────────────────────────────────────────────

def main(output_dir: str | Path = Path(__file__).parent / "data" / "output") -> None:
    """
    Orchestrate generation of all three datasets and save them as CSV files.

    Parameters
    ----------
    output_dir:
        Directory where the three CSV files will be written.
        Created automatically if it does not exist.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating merchants …")
    merchants_df = generate_merchants()

    print(f"Generating transactions (~{TARGET_TRANSACTIONS:,} rows) …")
    transactions_df = generate_transactions(merchants_df)

    print(f"Generating webhook logs ({len(transactions_df):,} rows) …")
    webhooks_df = generate_webhook_logs(transactions_df)

    print("Injecting demo anomalies …")
    transactions_df, webhooks_df = inject_demo_anomalies(
        transactions_df, webhooks_df,
    )

    # Prepare for CSV export
    txn_save = transactions_df.copy()
    txn_save["timestamp"] = txn_save["timestamp"].astype(str)
    wh_save = webhooks_df.copy()
    wh_save["timestamp"] = wh_save["timestamp"].astype(str)

    merchants_path = output_path / "merchants.csv"
    transactions_path = output_path / "transactions.csv"
    webhooks_path = output_path / "webhook_logs.csv"

    merchants_df.to_csv(merchants_path, index=False)
    txn_save.to_csv(transactions_path, index=False)
    wh_save.to_csv(webhooks_path, index=False)

    print(f"\n✓ Saved {len(merchants_df):,} merchants       → {merchants_path}")
    print(f"✓ Saved {len(txn_save):,} transactions   → {transactions_path}")
    print(f"✓ Saved {len(wh_save):,} webhook logs  → {webhooks_path}")

    # ── Summary statistics ────────────────────────────────────────────────
    print("\n── Generation Summary ───────────────────────────────────────────")
    print(f"  Time window:          {WINDOW_DAYS} days ({WINDOW_START} → {NOW})")
    print(f"  Merchants:            {len(merchants_df)}")
    print(f"  Total transactions:   {len(txn_save):,}")
    print(
        f"  SUCCESS rate:         "
        f"{(transactions_df['status'] == 'SUCCESS').mean():.1%}"
    )
    print(
        f"  DECLINED rate:        "
        f"{(transactions_df['status'] == 'DECLINED').mean():.1%}"
    )
    print(f"  Min amount:           ₹{transactions_df['amount'].min():.2f}")
    print(f"  Max amount:           ₹{transactions_df['amount'].max():.2f}")
    print(f"  Median amount:        ₹{transactions_df['amount'].median():.2f}")
    print(f"  Demo anomalies:       3 (TXN-00194400, TXN-00000004/WH-00000004, 150×401)")
    print(
        "\n  Note: when loading transactions.csv, use "
        "read_transactions_csv() to preserve card_bin as str."
    )


if __name__ == "__main__":
    main()
