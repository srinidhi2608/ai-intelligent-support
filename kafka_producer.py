"""
kafka_producer.py – Mock Gateway publisher for Phase-2 real-time streaming.

Publishes one synthetic transaction/webhook payload per second to the
`live_transactions` Kafka topic. The stream includes occasional anomalies:
- 93_Risk_Block transaction declines
- 401 Unauthorized webhook failures

Usage:
    python kafka_producer.py
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from uuid import uuid4

from kafka import KafkaProducer

TOPIC = "live_transactions"
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
PUBLISH_INTERVAL_SECONDS = 1


def _generate_payload() -> dict:
    """Generate one synthetic transaction + webhook event payload."""
    merchant_id = f"merchant_id_{random.randint(1, 25)}"
    amount = round(random.uniform(100, 50_000), 2)

    status = "SUCCESS"
    decline_code = None
    webhook_http_status = 200
    anomaly_type = "none"

    anomaly_roll = random.random()
    if anomaly_roll < 0.10:
        # Decline anomaly with card-network risk block.
        status = "DECLINED"
        decline_code = "93_Risk_Block"
        webhook_http_status = 200
        anomaly_type = "93_Risk_Block"
    elif anomaly_roll < 0.20:
        # Webhook auth anomaly.
        status = random.choice(["SUCCESS", "DECLINED"])
        decline_code = "05_Do_Not_Honor" if status == "DECLINED" else None
        webhook_http_status = 401
        anomaly_type = "401_Unauthorized"
    elif anomaly_roll < 0.35:
        # Normal declined transactions.
        status = "DECLINED"
        decline_code = random.choice(
            [
                "05_Do_Not_Honor",
                "51_Insufficient_Funds",
                "54_Expired_Card",
                "14_Invalid_Card_Number",
            ]
        )
        webhook_http_status = 200

    return {
        "transaction_id": f"TXN-STREAM-{uuid4().hex[:12].upper()}",
        "merchant_id": merchant_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "amount": amount,
        "currency": "INR",
        "status": status,
        "decline_code": decline_code,
        "webhook_http_status": webhook_http_status,
        "event_type": "payment.failed" if status == "DECLINED" else "payment.success",
        "anomaly_type": anomaly_type,
    }


def main() -> None:
    """Continuously publish synthetic payloads to Kafka."""
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        retries=5,
        acks="all",
    )

    print(f"[producer] Connected to Kafka at {BOOTSTRAP_SERVERS}")
    print(f"[producer] Publishing to topic '{TOPIC}' every {PUBLISH_INTERVAL_SECONDS}s")

    try:
        while True:
            payload = _generate_payload()
            producer.send(TOPIC, value=payload).get(timeout=10)
            print(
                "[producer] sent",
                payload["transaction_id"],
                f"status={payload['status']}",
                f"decline={payload['decline_code']}",
                f"webhook_http_status={payload['webhook_http_status']}",
            )
            time.sleep(PUBLISH_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[producer] Stopped by user.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
