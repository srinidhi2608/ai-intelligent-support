"""
tests/test_chat.py – Unit tests for the merchant support chat endpoint.
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_chat_message_returns_200():
    """A valid chat message should return HTTP 200 with a non-empty reply."""
    payload = {
        "merchant_id": "MERCH-001",
        "message": "Why did my last transaction fail?",
    }
    response = client.post("/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 0


def test_chat_message_with_session_id():
    """Providing a session_id should echo it back in the response."""
    payload = {
        "merchant_id": "MERCH-002",
        "message": "What is the status of TXN-XYZ?",
        "session_id": "test-session-42",
    }
    response = client.post("/chat/message", json=payload)
    assert response.status_code == 200
    assert response.json()["session_id"] == "test-session-42"


def test_chat_message_missing_required_field():
    """Omitting the message field should return HTTP 422."""
    payload = {"merchant_id": "MERCH-001"}
    response = client.post("/chat/message", json=payload)
    assert response.status_code == 422
