"""
data/telemetry_generator.py – Multimodal telemetry data generator for the
                               Intelligent Merchant Support AI Agent project.

Generates three relational Pandas DataFrames and saves them as CSV files:

  • merchants.csv       – 25 unique merchant profiles
  • transactions.csv    – ~259,200 transactions over the last 24 hours
                          (3 transactions/second across all merchants)
  • webhook_logs.csv    – One webhook delivery record per transaction

Six anomalies are injected to enable AI-agent diagnostics testing:

  1  merchant_id_1  – 50 consecutive DECLINED / 93_Risk_Block in 10-min window
  2  merchant_id_2  – All webhook deliveries in the last 2 hours → 401 Unauthorized
  3  merchant_id_3  – 100 card-testing transactions (₹1–₹5, 95 % declined) in 5-min window
  4  ALL merchants  – 90 % of BIN '411111' transactions fail in a 2-hour issuer-downtime window
  5  merchant_id_4  – High-volume spike 20:00–21:00 UTC; matching webhooks → 504 / latency > 5 s
  6  merchant_id_5  – Exactly 5 SUCCESS transactions whose webhooks are dropped (500)

Run directly:
    python data/telemetry_generator.py
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
_fake = Faker(locale="en_IN")
Faker.seed(SEED)

# ── Runtime anchor ────────────────────────────────────────────────────────────
# Freeze "now" so every function in this module shares the same reference point.
NOW: datetime = datetime.now(tz=timezone.utc).replace(microsecond=0)
WINDOW_HOURS: int = 24
WINDOW_START: datetime = NOW - timedelta(hours=WINDOW_HOURS)

# ── Scale constants ───────────────────────────────────────────────────────────
NUM_MERCHANTS: int = 25
TXN_RATE_PER_SECOND: int = 3           # 3 TPS  ×  86 400 s  =  259 200 transactions
ANOMALY_BIN: str = "411111"            # 6-digit BIN used for the issuer-downtime anomaly

# ── Lookup tables ─────────────────────────────────────────────────────────────
_MCC_CODES = [
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

_NORMAL_DECLINE_CODES = [
    "05_Do_Not_Honor",
    "51_Insufficient_Funds",
    "14_Invalid_Card_Number",
    "54_Expired_Card",
    "57_Transaction_Not_Permitted",
]

# Background BINs (anomaly BIN given a small natural weight of ~2 %)
_ALL_BINS = [
    "400011", "411000", "424242", "512345",
    "601100", "371449", "378282", "601782",
    ANOMALY_BIN,
]
_BIN_WEIGHTS = [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.02]

_WEBHOOK_EVENT_TYPES = [
    "payment.success",
    "payment.failed",
    "payment.pending",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _merchant_id(i: int) -> str:
    """Return the stable, human-readable merchant ID for index *i* (1-based)."""
    return f"merchant_id_{i}"


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 – Merchants
# ──────────────────────────────────────────────────────────────────────────────

def generate_merchants(n: int = NUM_MERCHANTS) -> pd.DataFrame:
    """
    Generate *n* unique merchant profiles.

    Columns
    -------
    merchant_id    – Stable ID of the form ``merchant_id_<n>``
    business_name  – Faker-generated company name
    mcc_code       – 4-digit ISO 18245 Merchant Category Code
    webhook_url    – HTTPS endpoint that receives payment events

    Parameters
    ----------
    n:
        Number of merchants to generate (default: 25).

    Returns
    -------
    pd.DataFrame
    """
    records = [
        {
            "merchant_id":    _merchant_id(i),
            "business_name":  _fake.company(),
            "mcc_code":       random.choice(_MCC_CODES),
            "webhook_url":    f"https://hooks.{_fake.domain_name()}/payment/events",
        }
        for i in range(1, n + 1)
    ]
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 – Transactions
# ──────────────────────────────────────────────────────────────────────────────

def generate_transactions(merchant_ids: list[str]) -> pd.DataFrame:
    """
    Generate synthetic transactions at ~3 TPS for the last 24 hours.

    Anomalies injected (in-place via boolean masks)
    ------------------------------------------------
    1  merchant_id_1  → 50 DECLINED / 93_Risk_Block  in a 10-min window
    3  merchant_id_3  → 100 tiny-amount card-test txns in a 5-min window
    4  ALL            → 90 % of BIN '411111' fail with 91_Issuer_Switch_Inoperative
                        in a 2-hour issuer-downtime window
    5  merchant_id_4  → Extra-high-volume spike 20:00–21:00; tagged for webhook mapping
    6  merchant_id_5  → First 5 SUCCESS txns tagged for dropped-webhook mapping

    Parameters
    ----------
    merchant_ids:
        Ordered list of merchant IDs to assign transactions to (round-robin).

    Returns
    -------
    pd.DataFrame with columns:
        transaction_id, merchant_id, timestamp (datetime), amount, currency,
        status, decline_code, card_bin, _anomaly_tag (internal, stripped on save)
    """
    total_seconds = WINDOW_HOURS * 3600                         # 86 400
    n_base = total_seconds * TXN_RATE_PER_SECOND                # 259 200

    # ── Build base timestamp array ─────────────────────────────────────────
    # Each second appears TXN_RATE_PER_SECOND times consecutively.
    second_offsets = np.repeat(np.arange(total_seconds), TXN_RATE_PER_SECOND)
    timestamps = pd.array(
        [WINDOW_START + timedelta(seconds=int(s)) for s in second_offsets],
        dtype="object",
    )

    # ── Round-robin merchant assignment ───────────────────────────────────
    merchant_col = [merchant_ids[i % len(merchant_ids)] for i in range(n_base)]

    # ── Random amounts, statuses, decline codes, card BINs ────────────────
    rng = np.random.default_rng(SEED)
    amounts      = rng.uniform(100.0, 50_000.0, n_base).round(2)
    is_declined  = rng.random(n_base) > 0.85                   # ~15 % decline rate
    card_bins    = random.choices(_ALL_BINS, weights=_BIN_WEIGHTS, k=n_base)

    decline_codes = [
        random.choice(_NORMAL_DECLINE_CODES) if is_declined[i] else None
        for i in range(n_base)
    ]
    statuses = ["DECLINED" if d else "SUCCESS" for d in is_declined]

    # Sequential transaction IDs (fast, unique, human-readable)
    txn_ids = [f"TXN-{i:08d}" for i in range(n_base)]

    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "merchant_id":    merchant_col,
        "timestamp":      timestamps,
        "amount":         amounts,
        "currency":       "INR",
        "status":         statuses,
        "decline_code":   decline_codes,
        "card_bin":       card_bins,
        "_anomaly_tag":   None,
    })

    # ── Anomaly window definitions ─────────────────────────────────────────
    # All windows are anchored inside the 24-hour generation period.

    # Anomaly 1: 10-min risk-block spike, starting 6 hours before NOW
    a1_start = NOW - timedelta(hours=6)
    a1_end   = a1_start + timedelta(minutes=10)

    # Anomaly 3: 5-min card-testing burst, starting 3 hours before NOW
    a3_start = NOW - timedelta(hours=3)
    a3_end   = a3_start + timedelta(minutes=5)

    # Anomaly 4: 2-hour issuer downtime, starting 8 hours before NOW
    a4_start = NOW - timedelta(hours=8)
    a4_end   = a4_start + timedelta(hours=2)

    # Anomaly 5: 20:00–21:00 UTC in the most-recent occurrence within the window
    # We find the 20:00 mark of the current UTC day; if it falls outside the
    # window, use the previous day's 20:00.
    today_20 = NOW.replace(hour=20, minute=0, second=0, microsecond=0)
    if today_20 > NOW:
        today_20 -= timedelta(days=1)
    a5_start = today_20
    a5_end   = a5_start + timedelta(hours=1)
    # Fallback: if the window is entirely outside the 24-h period, anchor at
    # 4 hours before NOW (always within range).
    if a5_start < WINDOW_START or a5_start > NOW:
        a5_start = NOW - timedelta(hours=4)
        a5_end   = a5_start + timedelta(hours=1)

    # ── Inject Anomaly 1 ──────────────────────────────────────────────────
    # merchant_id_1: first 50 transactions in the 10-min window → DECLINED/93_Risk_Block
    mask_a1_window = (
        (df["merchant_id"] == "merchant_id_1")
        & (df["timestamp"] >= a1_start)
        & (df["timestamp"] <= a1_end)
    )
    idx_a1 = df[mask_a1_window].head(50).index
    df.loc[idx_a1, "status"]       = "DECLINED"
    df.loc[idx_a1, "decline_code"] = "93_Risk_Block"
    df.loc[idx_a1, "_anomaly_tag"] = "ANOM1_RISK_BLOCK"

    # ── Inject Anomaly 3 ──────────────────────────────────────────────────
    # merchant_id_3: burst of exactly 100 extra card-testing transactions
    #   injected into the 5-min window (spike on top of baseline traffic)
    #   → amount ₹1–₹5, 95 % DECLINED with card-testing codes
    a3_window_seconds = int((a3_end - a3_start).total_seconds())
    a3_offsets  = np.random.randint(0, a3_window_seconds, size=100)
    a3_ts       = [a3_start + timedelta(seconds=int(s)) for s in a3_offsets]
    a3_amounts  = [round(random.uniform(1.0, 5.0), 2) for _ in range(100)]
    a3_declined = np.random.random(100) < 0.95
    a3_statuses = ["DECLINED" if d else "SUCCESS" for d in a3_declined]
    a3_codes    = [
        random.choice(["14_Invalid_Card_Number", "54_Expired_Card"]) if d else None
        for d in a3_declined
    ]
    a3_ids = [f"TXN-CARD-{i:05d}" for i in range(100)]

    a3_df = pd.DataFrame({
        "transaction_id": a3_ids,
        "merchant_id":    "merchant_id_3",
        "timestamp":      a3_ts,
        "amount":         a3_amounts,
        "currency":       "INR",
        "status":         a3_statuses,
        "decline_code":   a3_codes,
        "card_bin":       random.choices(_ALL_BINS, weights=_BIN_WEIGHTS, k=100),
        "_anomaly_tag":   "ANOM3_CARD_TESTING",
    })
    df = pd.concat([df, a3_df], ignore_index=True)

    # ── Inject Anomaly 4 ──────────────────────────────────────────────────
    # ALL merchants: 90 % of BIN '411111' transactions in the 2-hour window
    #   → DECLINED / 91_Issuer_Switch_Inoperative
    mask_a4_window = (
        (df["card_bin"] == ANOMALY_BIN)
        & (df["timestamp"] >= a4_start)
        & (df["timestamp"] <= a4_end)
    )
    idx_a4_all  = df[mask_a4_window].index
    # Use math.ceil to guarantee AT LEAST 90 % (not floor which could give 89.9 %)
    n_to_fail   = math.ceil(len(idx_a4_all) * 0.9)
    idx_a4_fail = np.random.choice(idx_a4_all, size=n_to_fail, replace=False)
    df.loc[idx_a4_fail, "status"]       = "DECLINED"
    df.loc[idx_a4_fail, "decline_code"] = "91_Issuer_Switch_Inoperative"
    # Only tag rows not already tagged by a higher-priority anomaly
    untagged_a4 = df.loc[idx_a4_fail, "_anomaly_tag"].isna()
    df.loc[idx_a4_fail[untagged_a4], "_anomaly_tag"] = "ANOM4_ISSUER_DOWN"

    # ── Tag Anomaly 5 (webhook mapping handled in generate_webhook_logs) ──
    # merchant_id_4 during the 20:00–21:00 spike window
    mask_a5 = (
        (df["merchant_id"] == "merchant_id_4")
        & (df["timestamp"] >= a5_start)
        & (df["timestamp"] <  a5_end)
    )
    df.loc[mask_a5 & df["_anomaly_tag"].isna(), "_anomaly_tag"] = "ANOM5_OVERLOAD"

    # ── Add extra high-volume transactions for Anomaly 5 ─────────────────
    # Inject 500 additional merchant_id_4 transactions in the 20:00–21:00 window
    # to simulate the volume spike (these sit on top of the baseline TPS).
    spike_seconds = int((a5_end - a5_start).total_seconds())
    spike_offsets = np.random.randint(0, spike_seconds, size=500)
    spike_ts      = [a5_start + timedelta(seconds=int(s)) for s in spike_offsets]
    spike_amounts = rng.uniform(100.0, 50_000.0, 500).round(2)
    spike_bins    = random.choices(_ALL_BINS, weights=_BIN_WEIGHTS, k=500)
    spike_status  = [
        "DECLINED" if rng.random() > 0.85 else "SUCCESS" for _ in range(500)
    ]
    spike_codes   = [
        random.choice(_NORMAL_DECLINE_CODES) if s == "DECLINED" else None
        for s in spike_status
    ]
    spike_ids = [f"TXN-SPIKE-{i:05d}" for i in range(500)]

    spike_df = pd.DataFrame({
        "transaction_id": spike_ids,
        "merchant_id":    "merchant_id_4",
        "timestamp":      spike_ts,
        "amount":         spike_amounts,
        "currency":       "INR",
        "status":         spike_status,
        "decline_code":   spike_codes,
        "card_bin":       spike_bins,
        "_anomaly_tag":   "ANOM5_OVERLOAD",
    })
    df = pd.concat([df, spike_df], ignore_index=True)

    # ── Tag Anomaly 6 ──────────────────────────────────────────────────────
    # merchant_id_5: first 5 SUCCESS transactions → webhooks will be dropped (500)
    mask_a6 = (df["merchant_id"] == "merchant_id_5") & (df["status"] == "SUCCESS")
    idx_a6  = df[mask_a6].head(5).index
    df.loc[idx_a6, "_anomaly_tag"] = "ANOM6_DROPPED_WEBHOOK"

    # Sort by timestamp for a realistic time-ordered log
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – Webhook Logs
# ──────────────────────────────────────────────────────────────────────────────

def generate_webhook_logs(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Generate one webhook delivery record for every transaction.

    Normal behaviour
    ----------------
    • ``http_status``       – weighted toward 200 (healthy); occasional 400 / 500
    • ``delivery_attempts`` – 1 for successes; 1–3 for failures
    • ``latency_ms``        – 50–500 ms
    • ``event_type``        – matches the transaction outcome

    Anomaly rules applied (using ``_anomaly_tag``)
    -----------------------------------------------
    2  merchant_id_2  → All webhooks in the last 2 hours → 401 Unauthorized
    5  ANOM5_OVERLOAD → http_status 504, latency_ms > 5000
    6  ANOM6_DROPPED  → http_status 500

    Parameters
    ----------
    transactions:
        DataFrame produced by ``generate_transactions()``.
        Must contain: transaction_id, merchant_id, timestamp, status, _anomaly_tag.

    Returns
    -------
    pd.DataFrame with columns:
        log_id, transaction_id, timestamp, event_type,
        http_status, delivery_attempts, latency_ms
    """
    n = len(transactions)

    # ── Derive event_type from transaction status ──────────────────────────
    event_map = {"SUCCESS": "payment.success", "DECLINED": "payment.failed"}
    event_types = transactions["status"].map(event_map).fillna("payment.pending")

    # ── Baseline webhook delivery (healthy) ───────────────────────────────
    normal_statuses = random.choices([200, 200, 200, 200, 400, 500], k=n)
    latency_ms      = np.random.randint(50, 501, size=n).tolist()
    delivery_attempts = [
        1 if s == 200 else random.randint(1, 3)
        for s in normal_statuses
    ]

    # Webhook timestamp = transaction timestamp + small delivery delay (1–30 s)
    delays = np.random.randint(1, 31, size=n)
    wh_timestamps = [
        ts + timedelta(seconds=int(d))
        for ts, d in zip(transactions["timestamp"], delays)
    ]

    log_ids = [f"WH-{i:08d}" for i in range(n)]

    df = pd.DataFrame({
        "log_id":             log_ids,
        "transaction_id":     transactions["transaction_id"].values,
        "timestamp":          wh_timestamps,
        "event_type":         event_types.values,
        "http_status":        normal_statuses,
        "delivery_attempts":  delivery_attempts,
        "latency_ms":         latency_ms,
        "_merchant_id":       transactions["merchant_id"].values,
        "_anomaly_tag":       transactions["_anomaly_tag"].values,
    })

    # ── Anomaly 2: merchant_id_2 – 401 for all webhooks in last 2 hours ──
    a2_cutoff = NOW - timedelta(hours=2)
    mask_a2 = (
        (df["_merchant_id"] == "merchant_id_2")
        & (df["timestamp"] >= a2_cutoff)
    )
    df.loc[mask_a2, "http_status"]        = 401
    df.loc[mask_a2, "delivery_attempts"]  = df.loc[mask_a2, "delivery_attempts"].apply(
        lambda _: random.randint(1, 3)
    )

    # ── Anomaly 5: ANOM5_OVERLOAD – 504 and high latency ─────────────────
    mask_a5 = df["_anomaly_tag"] == "ANOM5_OVERLOAD"
    df.loc[mask_a5, "http_status"]       = 504
    df.loc[mask_a5, "latency_ms"]        = np.random.randint(5001, 15001, size=mask_a5.sum())
    df.loc[mask_a5, "delivery_attempts"] = df.loc[mask_a5, "delivery_attempts"].apply(
        lambda _: random.randint(1, 3)
    )

    # ── Anomaly 6: ANOM6_DROPPED_WEBHOOK – 500 for 5 SUCCESS txns ────────
    mask_a6 = df["_anomaly_tag"] == "ANOM6_DROPPED_WEBHOOK"
    df.loc[mask_a6, "http_status"]       = 500
    df.loc[mask_a6, "delivery_attempts"] = 3    # Exhausted retries

    # Drop internal helper columns before returning
    df = df.drop(columns=["_merchant_id", "_anomaly_tag"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – CSV reading helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_transactions_csv(path: str | Path) -> pd.DataFrame:
    """
    Read ``transactions.csv`` back into a DataFrame with correct column types.

    ``card_bin`` is a 6-digit string (like a ZIP code or phone number) and
    must be loaded as ``str`` to prevent pandas from silently coercing it to
    ``int64``.  Always use this helper instead of a raw ``pd.read_csv()`` call
    when you need to filter or compare ``card_bin`` values.

    Parameters
    ----------
    path:
        Path to the ``transactions.csv`` file.

    Returns
    -------
    pd.DataFrame with ``card_bin`` typed as ``object`` (str).
    """
    return pd.read_csv(path, dtype={"card_bin": str})


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 – Orchestration & CSV export
# ──────────────────────────────────────────────────────────────────────────────

def main(output_dir: str | Path = Path(__file__).parent / "output") -> None:
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
    merchant_ids = merchants_df["merchant_id"].tolist()

    print(f"Generating transactions (~{WINDOW_HOURS * 3600 * TXN_RATE_PER_SECOND:,} rows + spike) …")
    transactions_df = generate_transactions(merchant_ids)

    print(f"Generating webhook logs ({len(transactions_df):,} rows) …")
    webhooks_df = generate_webhook_logs(transactions_df)

    # Strip the internal _anomaly_tag column from the saved CSV
    txn_save = transactions_df.drop(columns=["_anomaly_tag"])
    txn_save["timestamp"] = txn_save["timestamp"].astype(str)
    webhooks_save = webhooks_df.copy()
    webhooks_save["timestamp"] = webhooks_save["timestamp"].astype(str)

    merchants_path     = output_path / "merchants.csv"
    transactions_path  = output_path / "transactions.csv"
    webhooks_path      = output_path / "webhook_logs.csv"

    merchants_df.to_csv(merchants_path, index=False)
    txn_save.to_csv(transactions_path, index=False)
    webhooks_save.to_csv(webhooks_path, index=False)

    print(f"\n✓ Saved {len(merchants_df):,} merchants       → {merchants_path}")
    print(f"✓ Saved {len(txn_save):,} transactions   → {transactions_path}")
    print(f"✓ Saved {len(webhooks_save):,} webhook logs  → {webhooks_path}")

    # ── Anomaly summary (use the in-memory DataFrames – types are correct) ──
    tag_counts = transactions_df["_anomaly_tag"].value_counts()
    print("\n── Injected anomaly summary ──────────────────────────────────────")
    for tag, count in tag_counts.items():
        print(f"  {tag:<35} {count:>6} transactions")
    print(
        f"  {'ANOM2_UNAUTH_WEBHOOKS (last 2 h)':<35}"
        f" {int((webhooks_df['http_status'] == 401).sum()):>6} webhook logs (401)"
    )
    print(
        "\n  Note: when loading transactions.csv, always use "
        "read_transactions_csv() to preserve card_bin as str."
    )


if __name__ == "__main__":
    main()
