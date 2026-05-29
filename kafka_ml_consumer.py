"""
kafka_ml_consumer.py – Real-time ML watcher for Phase-2 anomaly detection.

Consumes payloads from the `live_transactions` Kafka topic, loads a pre-trained
IsolationForest model, predicts anomaly status, and persists only anomalies to:
`data/ml_active_alerts.csv`

Storage logic:
- prediction == 1  -> anomaly, save payload JSON
- prediction == 0  -> normal, discard

Usage:
    python kafka_ml_consumer.py
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import load
from kafka import KafkaConsumer

TOPIC = "live_transactions"
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
GROUP_ID = "ml-watcher-group"

MODEL_PATH = Path(
    os.getenv(
        "IF_MODEL_PATH",
        str(Path(__file__).parent / "models" / "isolation_forest_model.joblib"),
    )
)
ALERTS_PATH = Path(__file__).parent / "data" / "ml_active_alerts.csv"

# Fallback order if model doesn't expose feature names.
_DEFAULT_FEATURE_ORDER = [
    "amount",
    "is_declined",
    "is_risk_block",
    "is_webhook_401",
    "is_webhook_error",
    "hour_of_day",
]


def _to_hour_of_day(iso_ts: str | None) -> int:
    """Safely convert ISO timestamp to UTC hour; return current hour on failure."""
    if not iso_ts:
        return datetime.now(tz=timezone.utc).hour
    try:
        return datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).astimezone(timezone.utc).hour
    except ValueError:
        return datetime.now(tz=timezone.utc).hour


def _build_feature_map(payload: dict) -> dict[str, float]:
    """Build model features from raw Kafka payload."""
    status = str(payload.get("status", "")).upper()
    decline_code = str(payload.get("decline_code") or "")
    webhook_http_status = int(payload.get("webhook_http_status", 200))

    return {
        "amount": float(payload.get("amount", 0.0)),
        "is_declined": float(status == "DECLINED"),
        "is_success": float(status == "SUCCESS"),
        "is_risk_block": float(decline_code == "93_Risk_Block"),
        "is_webhook_401": float(webhook_http_status == 401),
        "is_webhook_error": float(webhook_http_status >= 400),
        "webhook_http_status": float(webhook_http_status),
        "hour_of_day": float(_to_hour_of_day(payload.get("timestamp"))),
    }


def _prepare_model_input(model, payload: dict) -> np.ndarray:
    """Align payload features to model feature order."""
    feature_map = _build_feature_map(payload)
    feature_order = list(getattr(model, "feature_names_in_", _DEFAULT_FEATURE_ORDER))
    ordered_values = [float(feature_map.get(name, 0.0)) for name in feature_order]
    return np.array([ordered_values], dtype=float)


def _predict_anomaly_flag(model, payload: dict) -> tuple[int, float | None]:
    """Return (prediction, score) with prediction in {0, 1} where 1 means anomaly."""
    x = _prepare_model_input(model, payload)

    # IsolationForest raw labels: -1 anomaly, 1 normal.
    raw = int(model.predict(x)[0])
    prediction = 1 if raw == -1 else 0

    score = None
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(x)[0])

    return prediction, score


def _append_alert(payload: dict, score: float | None) -> None:
    """Append anomaly record to data/ml_active_alerts.csv."""
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = ALERTS_PATH.exists()

    row = {
        "detected_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "anomaly_score": score,
        "payload_json": json.dumps(payload, ensure_ascii=False),
    }

    with ALERTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["detected_at_utc", "anomaly_score", "payload_json"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Start Kafka consumer loop and persist only anomalous events."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pre-trained model not found at '{MODEL_PATH}'. "
            "Set IF_MODEL_PATH or place your IsolationForest model at that path."
        )

    model = load(MODEL_PATH)
    print(f"[consumer] Loaded model from: {MODEL_PATH}")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id=GROUP_ID,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    print(f"[consumer] Listening on topic '{TOPIC}' at {BOOTSTRAP_SERVERS}")

    try:
        for message in consumer:
            payload = message.value
            prediction, score = _predict_anomaly_flag(model, payload)

            if prediction == 1:
                _append_alert(payload, score)
                print(
                    "[consumer] anomaly saved:",
                    payload.get("transaction_id"),
                    f"score={score}",
                )
            else:
                print("[consumer] normal discarded:", payload.get("transaction_id"))
    except KeyboardInterrupt:
        print("\n[consumer] Stopped by user.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
