"""
tests/test_data_access.py – Unit tests for the DataLoader class.

Uses in-memory DataFrames (no CSV files required) so the test suite can
run on any machine without first generating the synthetic data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.data_access import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def merchants_df() -> pd.DataFrame:
    """Minimal merchants fixture."""
    return pd.DataFrame(
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


@pytest.fixture
def transactions_df() -> pd.DataFrame:
    """Minimal transactions fixture with two merchants."""
    return pd.DataFrame(
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
            {
                "transaction_id": "TXN-00000003",
                "merchant_id":    "merchant_id_2",
                "timestamp":      "2026-03-07T10:02:00+00:00",
                "amount":         200.00,
                "currency":       "INR",
                "status":         "SUCCESS",
                "decline_code":   None,
                "card_bin":       "411111",
            },
        ]
    )


@pytest.fixture
def webhook_logs_df() -> pd.DataFrame:
    """Minimal webhook logs fixture with one entry per transaction."""
    return pd.DataFrame(
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
            {
                "log_id":            "WH-00000003",
                "transaction_id":    "TXN-00000003",
                "timestamp":         "2026-03-07T10:02:05+00:00",
                "event_type":        "payment.success",
                "http_status":       401,
                "delivery_attempts": 2,
                "latency_ms":        300,
            },
        ]
    )


@pytest.fixture
def loader(merchants_df, transactions_df, webhook_logs_df) -> DataLoader:
    """DataLoader initialised from in-memory fixtures."""
    return DataLoader(
        merchants_df=merchants_df,
        transactions_df=transactions_df,
        webhook_logs_df=webhook_logs_df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# get_merchant
# ──────────────────────────────────────────────────────────────────────────────


class TestGetMerchant:
    def test_found(self, loader):
        result = loader.get_merchant("merchant_id_1")
        assert result is not None
        assert result["merchant_id"] == "merchant_id_1"
        assert result["business_name"] == "Acme Payments Ltd."

    def test_not_found_returns_none(self, loader):
        assert loader.get_merchant("merchant_id_99") is None

    def test_empty_dataframe_returns_none(self):
        loader = DataLoader(
            merchants_df=pd.DataFrame(),
            transactions_df=pd.DataFrame(),
            webhook_logs_df=pd.DataFrame(),
        )
        assert loader.get_merchant("merchant_id_1") is None


# ──────────────────────────────────────────────────────────────────────────────
# get_recent_transactions
# ──────────────────────────────────────────────────────────────────────────────


class TestGetRecentTransactions:
    def test_returns_correct_merchant_only(self, loader):
        rows = loader.get_recent_transactions("merchant_id_1")
        assert all(r["merchant_id"] == "merchant_id_1" for r in rows)

    def test_sorted_newest_first(self, loader):
        rows = loader.get_recent_transactions("merchant_id_1")
        assert rows[0]["timestamp"] >= rows[-1]["timestamp"]

    def test_limit_respected(self, loader):
        rows = loader.get_recent_transactions("merchant_id_1", limit=1)
        assert len(rows) == 1

    def test_unknown_merchant_returns_empty(self, loader):
        assert loader.get_recent_transactions("merchant_id_99") == []

    def test_empty_dataframe_returns_empty(self):
        loader = DataLoader(
            merchants_df=pd.DataFrame(),
            transactions_df=pd.DataFrame(),
            webhook_logs_df=pd.DataFrame(),
        )
        assert loader.get_recent_transactions("merchant_id_1") == []


# ──────────────────────────────────────────────────────────────────────────────
# get_transaction_details
# ──────────────────────────────────────────────────────────────────────────────


class TestGetTransactionDetails:
    def test_found(self, loader):
        result = loader.get_transaction_details("TXN-00000001")
        assert result is not None
        assert result["transaction_id"] == "TXN-00000001"
        assert result["amount"] == 1500.00

    def test_not_found_returns_none(self, loader):
        assert loader.get_transaction_details("TXN-MISSING") is None

    def test_decline_code_present_for_declined(self, loader):
        result = loader.get_transaction_details("TXN-00000002")
        assert result["status"] == "DECLINED"
        assert result["decline_code"] == "51_Insufficient_Funds"

    def test_decline_code_none_for_success(self, loader):
        result = loader.get_transaction_details("TXN-00000001")
        assert result["status"] == "SUCCESS"
        assert result["decline_code"] is None


# ──────────────────────────────────────────────────────────────────────────────
# get_webhook_logs_for_merchant
# ──────────────────────────────────────────────────────────────────────────────


class TestGetWebhookLogsForMerchant:
    def test_returns_logs_for_correct_merchant(self, loader):
        logs = loader.get_webhook_logs_for_merchant("merchant_id_1")
        # merchant_id_1 has TXN-00000001 and TXN-00000002
        txn_ids = {log["transaction_id"] for log in logs}
        assert txn_ids == {"TXN-00000001", "TXN-00000002"}

    def test_does_not_leak_other_merchant_logs(self, loader):
        logs = loader.get_webhook_logs_for_merchant("merchant_id_1")
        for log in logs:
            assert log["transaction_id"] != "TXN-00000003"

    def test_limit_respected(self, loader):
        logs = loader.get_webhook_logs_for_merchant("merchant_id_1", limit=1)
        assert len(logs) == 1

    def test_unknown_merchant_returns_empty(self, loader):
        assert loader.get_webhook_logs_for_merchant("merchant_id_99") == []


# ──────────────────────────────────────────────────────────────────────────────
# get_webhook_log_for_transaction
# ──────────────────────────────────────────────────────────────────────────────


class TestGetWebhookLogForTransaction:
    def test_found_returns_correct_log(self, loader):
        result = loader.get_webhook_log_for_transaction("TXN-00000001")
        assert result is not None
        assert result["log_id"] == "WH-00000001"
        assert result["transaction_id"] == "TXN-00000001"

    def test_not_found_returns_none(self, loader):
        assert loader.get_webhook_log_for_transaction("TXN-MISSING") is None

    def test_returns_correct_fields_for_failed_webhook(self, loader):
        result = loader.get_webhook_log_for_transaction("TXN-00000002")
        assert result is not None
        assert result["http_status"] == 500
        assert result["event_type"] == "payment.failed"

    def test_empty_dataframe_returns_none(self):
        loader = DataLoader(
            merchants_df=pd.DataFrame(),
            transactions_df=pd.DataFrame(),
            webhook_logs_df=pd.DataFrame(),
        )
        assert loader.get_webhook_log_for_transaction("TXN-00000001") is None

    def test_transaction_with_401_webhook(self, loader):
        result = loader.get_webhook_log_for_transaction("TXN-00000003")
        assert result is not None
        assert result["http_status"] == 401


# ──────────────────────────────────────────────────────────────────────────────
# update_webhook_status
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateWebhookStatus:
    def test_update_existing_log(self, loader):
        success = loader.update_webhook_status("WH-00000002", 200)
        assert success is True
        # Verify the change was applied
        row = loader.webhook_logs[loader.webhook_logs["log_id"] == "WH-00000002"]
        assert int(row.iloc[0]["http_status"]) == 200

    def test_update_non_existent_log_returns_false(self, loader):
        assert loader.update_webhook_status("WH-MISSING", 200) is False

    def test_update_does_not_affect_other_rows(self, loader):
        loader.update_webhook_status("WH-00000002", 200)
        row = loader.webhook_logs[loader.webhook_logs["log_id"] == "WH-00000001"]
        assert int(row.iloc[0]["http_status"]) == 200  # was already 200

    def test_update_on_empty_dataframe_returns_false(self):
        loader = DataLoader(
            merchants_df=pd.DataFrame(),
            transactions_df=pd.DataFrame(),
            webhook_logs_df=pd.DataFrame(),
        )
        assert loader.update_webhook_status("WH-00000001", 200) is False
