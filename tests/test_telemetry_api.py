"""
tests/test_telemetry_api.py – Integration tests for the /api/v1/ endpoints.

Injects a small in-memory ``DataLoader`` into ``app.state`` before each test
so no CSV files need to be present on disk.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.data_access import DataLoader
from main import app

# ──────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_loader() -> DataLoader:
    """Build a DataLoader from minimal in-memory fixtures."""
    merchants = pd.DataFrame(
        [
            {
                "merchant_id":   "merchant_id_1",
                "business_name": "Acme Payments Ltd.",
                "mcc_code":      "5812",
                "webhook_url":   "https://hooks.acme.com/payment/events",
            },
            {
                "merchant_id":   "merchant_id_2",
                "business_name": "Beta Commerce Pvt.",
                "mcc_code":      "5411",
                "webhook_url":   "https://hooks.beta.com/payment/events",
            },
        ]
    )

    transactions = pd.DataFrame(
        [
            {
                "transaction_id": "TXN-00000001",
                "merchant_id":    "merchant_id_1",
                "timestamp":      "2026-03-07T10:00:00+00:00",
                "amount":         1500.00,
                "currency":       "INR",
                "status":         "SUCCESS",
                "decline_code":   None,
                "card_bin":       "424242",
            },
            {
                "transaction_id": "TXN-00000002",
                "merchant_id":    "merchant_id_1",
                "timestamp":      "2026-03-07T10:01:00+00:00",
                "amount":         800.00,
                "currency":       "INR",
                "status":         "DECLINED",
                "decline_code":   "51_Insufficient_Funds",
                "card_bin":       "400011",
            },
        ]
    )

    webhook_logs = pd.DataFrame(
        [
            {
                "log_id":            "WH-00000001",
                "transaction_id":    "TXN-00000001",
                "timestamp":         "2026-03-07T10:00:05+00:00",
                "event_type":        "payment.success",
                "http_status":       200,
                "delivery_attempts": 1,
                "latency_ms":        120,
            },
            {
                "log_id":            "WH-00000002",
                "transaction_id":    "TXN-00000002",
                "timestamp":         "2026-03-07T10:01:05+00:00",
                "event_type":        "payment.failed",
                "http_status":       500,
                "delivery_attempts": 3,
                "latency_ms":        9800,
            },
        ]
    )

    return DataLoader(
        merchants_df=merchants,
        transactions_df=transactions,
        webhook_logs_df=webhook_logs,
    )


@pytest.fixture(autouse=True)
def inject_loader():
    """
    Inject a test DataLoader into app.state before each test and clean up
    afterwards.  ``autouse=True`` means this applies to every test in the
    module automatically.
    """
    app.state.data_loader = _make_loader()
    yield
    # Restore to a clean state so tests remain independent
    if hasattr(app.state, "data_loader"):
        del app.state.data_loader


@pytest.fixture
def client() -> TestClient:
    """Return a synchronous TestClient that does NOT trigger lifespan hooks
    (which would try to load real CSVs from disk)."""
    return TestClient(app, raise_server_exceptions=True)


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/v1/merchants/{merchant_id}
# ──────────────────────────────────────────────────────────────────────────────


class TestGetMerchantEndpoint:
    def test_found_returns_200(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1")
        assert r.status_code == 200
        data = r.json()
        assert data["merchant_id"] == "merchant_id_1"
        assert data["business_name"] == "Acme Payments Ltd."
        assert data["mcc_code"] == "5812"
        assert data["webhook_url"].startswith("https://")

    def test_not_found_returns_404(self, client):
        r = client.get("/api/v1/merchants/merchant_id_99")
        assert r.status_code == 404

    def test_response_schema_fields(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1")
        keys = set(r.json().keys())
        assert keys == {"merchant_id", "business_name", "mcc_code", "webhook_url"}


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/v1/merchants/{merchant_id}/transactions
# ──────────────────────────────────────────────────────────────────────────────


class TestGetMerchantTransactionsEndpoint:
    def test_returns_200_with_list(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1/transactions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_default_limit_is_respected(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1/transactions")
        # Only 2 transactions in fixture; both should be returned
        assert len(r.json()) == 2

    def test_custom_limit(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1/transactions?limit=1")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_unknown_merchant_returns_404(self, client):
        r = client.get("/api/v1/merchants/merchant_id_99/transactions")
        assert r.status_code == 404

    def test_merchant_with_no_transactions_returns_empty_list(self, client):
        # merchant_id_2 exists but has no transactions in the fixture
        r = client.get("/api/v1/merchants/merchant_id_2/transactions")
        assert r.status_code == 200
        assert r.json() == []

    def test_transaction_fields_present(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1/transactions")
        required_keys = {
            "transaction_id", "merchant_id", "timestamp",
            "amount", "currency", "status",
        }
        for row in r.json():
            assert required_keys.issubset(set(row.keys()))


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/v1/transactions/{transaction_id}
# ──────────────────────────────────────────────────────────────────────────────


class TestGetTransactionEndpoint:
    def test_found_returns_200(self, client):
        r = client.get("/api/v1/transactions/TXN-00000001")
        assert r.status_code == 200
        data = r.json()
        assert data["transaction_id"] == "TXN-00000001"
        assert data["amount"] == 1500.00
        assert data["status"] == "SUCCESS"
        assert data["decline_code"] is None

    def test_declined_transaction_has_decline_code(self, client):
        r = client.get("/api/v1/transactions/TXN-00000002")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "DECLINED"
        assert data["decline_code"] == "51_Insufficient_Funds"

    def test_not_found_returns_404(self, client):
        r = client.get("/api/v1/transactions/TXN-MISSING")
        assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/v1/merchants/{merchant_id}/webhooks
# ──────────────────────────────────────────────────────────────────────────────


class TestGetMerchantWebhooksEndpoint:
    def test_returns_200_with_list(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1/webhooks")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_logs_contain_correct_transactions(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1/webhooks")
        txn_ids = {log["transaction_id"] for log in r.json()}
        assert txn_ids == {"TXN-00000001", "TXN-00000002"}

    def test_unknown_merchant_returns_404(self, client):
        r = client.get("/api/v1/merchants/merchant_id_99/webhooks")
        assert r.status_code == 404

    def test_webhook_fields_present(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1/webhooks")
        required_keys = {
            "log_id", "transaction_id", "timestamp",
            "event_type", "http_status", "delivery_attempts", "latency_ms",
        }
        for log in r.json():
            assert required_keys.issubset(set(log.keys()))

    def test_limit_query_param(self, client):
        r = client.get("/api/v1/merchants/merchant_id_1/webhooks?limit=1")
        assert r.status_code == 200
        assert len(r.json()) == 1


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/v1/webhooks/{log_id}/retry
# ──────────────────────────────────────────────────────────────────────────────


class TestRetryWebhookEndpoint:
    def test_successful_retry_returns_200(self, client):
        r = client.post("/api/v1/webhooks/WH-00000002/retry")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["log_id"] == "WH-00000002"
        assert data["new_http_status"] == 200

    def test_retry_actually_updates_status(self, client):
        """Verify the in-memory status is updated and reflected in a subsequent GET."""
        # Before retry the log has http_status 500
        logs_before = client.get("/api/v1/merchants/merchant_id_1/webhooks").json()
        wh2_before = next(l for l in logs_before if l["log_id"] == "WH-00000002")
        assert wh2_before["http_status"] == 500

        # Perform retry
        client.post("/api/v1/webhooks/WH-00000002/retry")

        # After retry the log should reflect http_status 200
        logs_after = client.get("/api/v1/merchants/merchant_id_1/webhooks").json()
        wh2_after = next(l for l in logs_after if l["log_id"] == "WH-00000002")
        assert wh2_after["http_status"] == 200

    def test_unknown_log_returns_404(self, client):
        r = client.post("/api/v1/webhooks/WH-MISSING/retry")
        assert r.status_code == 404

    def test_response_contains_message(self, client):
        r = client.post("/api/v1/webhooks/WH-00000001/retry")
        assert "message" in r.json()
        assert len(r.json()["message"]) > 0
