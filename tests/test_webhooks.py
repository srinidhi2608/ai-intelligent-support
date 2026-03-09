"""
tests/test_webhooks.py – Unit tests for the payment webhook endpoint.
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_payment_webhook_success():
    """A valid SUCCESS payment event should return HTTP 200 and be acknowledged."""
    payload = {
        "transaction_id": "TXN-ABC123",
        "merchant_id": "MERCH-001",
        "amount": 1500.50,
        "currency": "INR",
        "status": "SUCCESS",
        "error_code": None,
        "metadata": {"gateway": "razorpay"},
    }
    response = client.post("/webhook/payment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["received"] is True
    assert data["transaction_id"] == "TXN-ABC123"


def test_payment_webhook_failed_transaction():
    """A FAILED payment event with an error code should be accepted."""
    payload = {
        "transaction_id": "TXN-FAIL999",
        "merchant_id": "MERCH-002",
        "amount": 500.00,
        "currency": "INR",
        "status": "FAILED",
        "error_code": "INSUFFICIENT_FUNDS",
    }
    response = client.post("/webhook/payment", json=payload)
    assert response.status_code == 200
    assert response.json()["transaction_id"] == "TXN-FAIL999"


def test_payment_webhook_missing_required_field():
    """Omitting a required field should return HTTP 422 Unprocessable Entity."""
    # Missing 'status' which is required
    payload = {
        "transaction_id": "TXN-BAD",
        "merchant_id": "MERCH-003",
        "amount": 100.0,
    }
    response = client.post("/webhook/payment", json=payload)
    assert response.status_code == 422
