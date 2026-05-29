"""
train_isolation_forest.py – One-time training script for the Kafka ML Consumer.

Generates synthetic transaction data that mirrors the feature space produced by
kafka_producer.py, trains an IsolationForest anomaly detector, and serialises
the fitted model to:

    models/isolation_forest_model.joblib

Run this script once before starting kafka_ml_consumer.py:

    python train_isolation_forest.py

The generated model captures the normal operating distribution of the mock
gateway (majority healthy transactions with occasional 93_Risk_Block declines
and 401 webhook errors) so the consumer can detect genuine anomalies at runtime.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest

# ── Paths ──────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "isolation_forest_model.joblib"

# ── Feature columns (must match _DEFAULT_FEATURE_ORDER in kafka_ml_consumer) ──

FEATURE_COLUMNS: list[str] = [
    "amount",
    "is_declined",
    "is_risk_block",
    "is_webhook_401",
    "is_webhook_error",
    "hour_of_day",
]

# ── Training parameters ────────────────────────────────────────────────────────

N_SAMPLES: int = 5_000      # rows of synthetic training data
CONTAMINATION: float = 0.05  # expected anomaly fraction
RANDOM_STATE: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────────────────────────────────────


def _generate_training_data(n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Produce *n* synthetic transaction rows that mirror kafka_producer.py output.

    The distribution is intentionally skewed toward normal transactions so the
    IsolationForest learns a healthy baseline and can isolate the rare anomalies.

    Normal (~65 % of rows):
        - Random amount 100–50,000
        - status SUCCESS, no decline, webhook 200

    Normal-decline (~15 % of rows):
        - Random amount 100–50,000
        - status DECLINED with a routine code (05/51/54/14)
        - webhook 200

    Anomaly – 93_Risk_Block (~10 % of rows):
        - Small amount typical of card-testing (₹1–₹20)
        - status DECLINED, decline_code 93_Risk_Block
        - webhook 200

    Anomaly – 401 webhook (~10 % of rows):
        - Random status, webhook 401
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for _ in range(n):
        roll = rng.random()

        if roll < 0.65:                        # normal success
            rows.append({
                "amount": round(np_rng.uniform(100, 50_000), 2),
                "is_declined": 0.0,
                "is_risk_block": 0.0,
                "is_webhook_401": 0.0,
                "is_webhook_error": 0.0,
                "hour_of_day": float(rng.randint(0, 23)),
            })
        elif roll < 0.80:                      # normal decline (routine code)
            rows.append({
                "amount": round(np_rng.uniform(100, 50_000), 2),
                "is_declined": 1.0,
                "is_risk_block": 0.0,
                "is_webhook_401": 0.0,
                "is_webhook_error": 0.0,
                "hour_of_day": float(rng.randint(0, 23)),
            })
        elif roll < 0.90:                      # anomaly: 93_Risk_Block
            rows.append({
                "amount": round(np_rng.uniform(1, 20), 2),
                "is_declined": 1.0,
                "is_risk_block": 1.0,
                "is_webhook_401": 0.0,
                "is_webhook_error": 0.0,
                "hour_of_day": float(rng.randint(0, 23)),
            })
        else:                                  # anomaly: 401 webhook
            rows.append({
                "amount": round(np_rng.uniform(100, 50_000), 2),
                "is_declined": float(rng.random() < 0.5),
                "is_risk_block": 0.0,
                "is_webhook_401": 1.0,
                "is_webhook_error": 1.0,
                "hour_of_day": float(rng.randint(0, 23)),
            })

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


# ──────────────────────────────────────────────────────────────────────────────
# Training & persistence
# ──────────────────────────────────────────────────────────────────────────────


def train_and_save(
    n_samples: int = N_SAMPLES,
    contamination: float = CONTAMINATION,
    random_state: int = RANDOM_STATE,
    output_path: Path = MODEL_PATH,
) -> None:
    """Train an IsolationForest and persist it with joblib."""
    print(f"[train] Generating {n_samples:,} synthetic training samples …")
    df = _generate_training_data(n_samples, seed=random_state)

    print(f"[train] Feature columns : {FEATURE_COLUMNS}")
    print(f"[train] Contamination   : {contamination}")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    # Fit with a numpy array so sklearn does not set feature_names_in_.
    # The consumer falls back to _DEFAULT_FEATURE_ORDER (same 6-column order)
    # when feature_names_in_ is absent, which avoids sklearn "feature names"
    # warnings at prediction time.
    model.fit(df[FEATURE_COLUMNS].to_numpy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, output_path)
    print(f"[train] Model saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_save()
    print("[train] Done. You can now run kafka_ml_consumer.py.")
