"""
tests/test_telemetry_generator.py – Focused tests for the multimodal telemetry
                                     data generator.

Validates:
  • Schema and row-count expectations for all three DataFrames
  • Correct injection of all six anomalies
  • Relational integrity (transaction_id links txn ↔ webhook_logs)
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from data.telemetry_generator import (
    ANOMALY_BIN,
    NOW,
    WINDOW_HOURS,
    generate_merchants,
    generate_transactions,
    generate_webhook_logs,
    read_transactions_csv,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures – generate once per test session for speed
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def merchants() -> pd.DataFrame:
    return generate_merchants()


@pytest.fixture(scope="module")
def transactions(merchants) -> pd.DataFrame:
    return generate_transactions(merchants["merchant_id"].tolist())


@pytest.fixture(scope="module")
def webhooks(transactions) -> pd.DataFrame:
    return generate_webhook_logs(transactions)


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 – Merchant schema
# ──────────────────────────────────────────────────────────────────────────────


class TestMerchants:
    def test_row_count(self, merchants):
        assert len(merchants) == 25

    def test_columns(self, merchants):
        assert set(merchants.columns) >= {"merchant_id", "business_name", "mcc_code", "webhook_url"}

    def test_merchant_ids_unique(self, merchants):
        assert merchants["merchant_id"].nunique() == 25

    def test_merchant_ids_follow_pattern(self, merchants):
        """IDs should be merchant_id_1 … merchant_id_25."""
        expected = {f"merchant_id_{i}" for i in range(1, 26)}
        assert set(merchants["merchant_id"]) == expected

    def test_webhook_urls_are_https(self, merchants):
        assert merchants["webhook_url"].str.startswith("https://").all()


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 – Transaction schema
# ──────────────────────────────────────────────────────────────────────────────


class TestTransactionSchema:
    REQUIRED_COLS = {
        "transaction_id", "merchant_id", "timestamp", "amount",
        "currency", "status", "decline_code", "card_bin",
    }

    def test_required_columns_present(self, transactions):
        assert self.REQUIRED_COLS.issubset(set(transactions.columns))

    def test_row_count_at_least_base(self, transactions):
        """At least 3 TPS × 24 h + anomaly spikes should be present."""
        base = WINDOW_HOURS * 3600 * 3   # 259,200
        assert len(transactions) >= base

    def test_currency_always_inr(self, transactions):
        assert (transactions["currency"] == "INR").all()

    def test_status_values(self, transactions):
        assert set(transactions["status"].unique()).issubset({"SUCCESS", "DECLINED"})

    def test_decline_code_none_for_success(self, transactions):
        success_rows = transactions[transactions["status"] == "SUCCESS"]
        assert success_rows["decline_code"].isna().all()

    def test_decline_code_set_for_declined(self, transactions):
        declined_rows = transactions[transactions["status"] == "DECLINED"]
        assert declined_rows["decline_code"].notna().all()

    def test_amounts_positive(self, transactions):
        assert (transactions["amount"] > 0).all()

    def test_transaction_ids_unique(self, transactions):
        assert transactions["transaction_id"].nunique() == len(transactions)


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – Anomaly 1: merchant_id_1 risk-block spike
# ──────────────────────────────────────────────────────────────────────────────


class TestAnomaly1RiskBlock:
    def _anom_rows(self, transactions) -> pd.DataFrame:
        return transactions[
            (transactions["merchant_id"] == "merchant_id_1")
            & (transactions["decline_code"] == "93_Risk_Block")
        ]

    def test_exactly_50_risk_block_rows(self, transactions):
        assert len(self._anom_rows(transactions)) == 50

    def test_all_declined(self, transactions):
        rows = self._anom_rows(transactions)
        assert (rows["status"] == "DECLINED").all()

    def test_within_10_minute_window(self, transactions):
        rows = self._anom_rows(transactions)
        ts = pd.to_datetime(rows["timestamp"])
        window_seconds = (ts.max() - ts.min()).total_seconds()
        assert window_seconds <= 600  # ≤ 10 minutes


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – Anomaly 2: merchant_id_2 unauthorized webhooks
# ──────────────────────────────────────────────────────────────────────────────


class TestAnomaly2UnauthorizedWebhooks:
    def test_401_webhooks_exist_for_merchant_2(self, transactions, webhooks):
        """All webhooks for merchant_id_2 in the last 2 hours must be 401."""
        cutoff = NOW - timedelta(hours=2)
        m2_recent_txns = transactions[
            (transactions["merchant_id"] == "merchant_id_2")
            & (pd.to_datetime(transactions["timestamp"]) >= cutoff)
        ]
        if len(m2_recent_txns) == 0:
            pytest.skip("No merchant_id_2 transactions in last 2 hours")

        m2_wh = webhooks[webhooks["transaction_id"].isin(m2_recent_txns["transaction_id"])]
        assert len(m2_wh) > 0
        assert (m2_wh["http_status"] == 401).all()


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 – Anomaly 3: merchant_id_3 card-testing burst
# ──────────────────────────────────────────────────────────────────────────────


class TestAnomaly3CardTesting:
    def _anom_rows(self, transactions) -> pd.DataFrame:
        return transactions[
            (transactions["merchant_id"] == "merchant_id_3")
            & (transactions["amount"] <= 5.0)
        ]

    def test_exactly_100_card_testing_rows(self, transactions):
        assert len(self._anom_rows(transactions)) == 100

    def test_amounts_between_1_and_5(self, transactions):
        rows = self._anom_rows(transactions)
        assert (rows["amount"] >= 1.0).all()
        assert (rows["amount"] <= 5.0).all()

    def test_at_least_90_percent_declined(self, transactions):
        rows = self._anom_rows(transactions)
        pct_declined = (rows["status"] == "DECLINED").mean()
        assert pct_declined >= 0.90

    def test_decline_codes_are_card_testing_codes(self, transactions):
        rows = self._anom_rows(transactions)
        declined = rows[rows["status"] == "DECLINED"]
        valid_codes = {"14_Invalid_Card_Number", "54_Expired_Card"}
        assert set(declined["decline_code"].unique()).issubset(valid_codes)

    def test_within_5_minute_window(self, transactions):
        rows = self._anom_rows(transactions)
        ts = pd.to_datetime(rows["timestamp"])
        window_seconds = (ts.max() - ts.min()).total_seconds()
        assert window_seconds <= 300  # ≤ 5 minutes


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 – Anomaly 4: BIN 411111 issuer downtime
# ──────────────────────────────────────────────────────────────────────────────


class TestAnomaly4IssuerDowntime:
    def test_anomaly_bin_constant(self):
        assert ANOMALY_BIN == "411111"

    def test_bin_411111_transactions_exist(self, transactions):
        """Some transactions should carry BIN 411111 (2 % probability)."""
        bin_rows = transactions[transactions["card_bin"].astype(str) == ANOMALY_BIN]
        assert len(bin_rows) > 0

    def test_90_percent_fail_in_downtime_window(self, transactions):
        """
        In the 2-hour issuer-downtime window, ≥ 90 % of BIN 411111 transactions
        should have decline_code '91_Issuer_Switch_Inoperative'.
        """
        a4_start = NOW - timedelta(hours=8)
        a4_end   = a4_start + timedelta(hours=2)

        ts = pd.to_datetime(transactions["timestamp"])
        mask = (
            (transactions["card_bin"].astype(str) == ANOMALY_BIN)
            & (ts >= a4_start)
            & (ts <= a4_end)
        )
        window_rows = transactions[mask]
        if len(window_rows) == 0:
            pytest.skip("No BIN 411111 transactions in the downtime window")

        fail_pct = (window_rows["decline_code"] == "91_Issuer_Switch_Inoperative").mean()
        assert fail_pct >= 0.90, f"Only {fail_pct:.0%} failed (expected ≥90 %)"


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 – Anomaly 5: merchant_id_4 server-overload
# ──────────────────────────────────────────────────────────────────────────────


class TestAnomaly5ServerOverload:
    def test_spike_transactions_exist(self, transactions):
        """merchant_id_4 should have an extra volume spike (500+ extra rows)."""
        m4 = transactions[transactions["merchant_id"] == "merchant_id_4"]
        # Baseline share: 1/25 of 259 200 ≈ 10 368; with spike should be more
        baseline = (WINDOW_HOURS * 3600 * 3) // 25
        assert len(m4) > baseline

    def test_504_webhooks_for_overload_transactions(self, transactions, webhooks):
        """Webhooks for merchant_id_4 overload transactions must return 504."""
        # Find the overload window (same logic as generator)
        today_20 = NOW.replace(hour=20, minute=0, second=0, microsecond=0)
        if today_20 > NOW:
            today_20 -= timedelta(days=1)
        a5_start = today_20
        a5_end   = a5_start + timedelta(hours=1)
        if a5_start < (NOW - timedelta(hours=WINDOW_HOURS)) or a5_start > NOW:
            a5_start = NOW - timedelta(hours=4)
            a5_end   = a5_start + timedelta(hours=1)

        ts = pd.to_datetime(transactions["timestamp"])
        m4_spike_txns = transactions[
            (transactions["merchant_id"] == "merchant_id_4")
            & (ts >= a5_start)
            & (ts < a5_end)
        ]
        if len(m4_spike_txns) == 0:
            pytest.skip("No merchant_id_4 transactions in the 20:00–21:00 window")

        m4_wh = webhooks[webhooks["transaction_id"].isin(m4_spike_txns["transaction_id"])]
        assert (m4_wh["http_status"] == 504).all()
        assert (m4_wh["latency_ms"] > 5000).all()


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 – Anomaly 6: merchant_id_5 dropped webhooks
# ──────────────────────────────────────────────────────────────────────────────


class TestAnomaly6DroppedWebhooks:
    def _dropped_wh(self, transactions, webhooks) -> pd.DataFrame:
        m5_success = transactions[
            (transactions["merchant_id"] == "merchant_id_5")
            & (transactions["status"] == "SUCCESS")
        ].head(5)
        return webhooks[webhooks["transaction_id"].isin(m5_success["transaction_id"])]

    def test_exactly_5_dropped_webhooks(self, transactions, webhooks):
        dropped = self._dropped_wh(transactions, webhooks)
        assert len(dropped) == 5

    def test_all_dropped_webhooks_are_500(self, transactions, webhooks):
        dropped = self._dropped_wh(transactions, webhooks)
        assert (dropped["http_status"] == 500).all()


# ──────────────────────────────────────────────────────────────────────────────
# Section 9 – Webhook schema and relational integrity
# ──────────────────────────────────────────────────────────────────────────────


class TestWebhookSchema:
    REQUIRED_COLS = {
        "log_id", "transaction_id", "timestamp", "event_type",
        "http_status", "delivery_attempts", "latency_ms",
    }

    def test_required_columns(self, webhooks):
        assert self.REQUIRED_COLS.issubset(set(webhooks.columns))

    def test_one_webhook_per_transaction(self, transactions, webhooks):
        assert len(webhooks) == len(transactions)

    def test_all_transaction_ids_present(self, transactions, webhooks):
        assert set(transactions["transaction_id"]) == set(webhooks["transaction_id"])

    def test_log_ids_unique(self, webhooks):
        assert webhooks["log_id"].nunique() == len(webhooks)

    def test_event_type_values(self, webhooks):
        valid = {"payment.success", "payment.failed", "payment.pending"}
        assert set(webhooks["event_type"].unique()).issubset(valid)

    def test_delivery_attempts_positive(self, webhooks):
        assert (webhooks["delivery_attempts"] >= 1).all()

    def test_latency_ms_positive(self, webhooks):
        assert (webhooks["latency_ms"] > 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# Section 10 – read_transactions_csv helper
# ──────────────────────────────────────────────────────────────────────────────


class TestReadTransactionsCsv:
    def test_card_bin_is_str_dtype(self, tmp_path, transactions):
        """read_transactions_csv() must return card_bin as a string dtype, not int."""
        # Save a small sample to a temp CSV
        sample = transactions[["transaction_id", "card_bin", "amount"]].head(20)
        csv_path = tmp_path / "sample.csv"
        sample.to_csv(csv_path, index=False)

        df = read_transactions_csv(csv_path)
        assert pd.api.types.is_string_dtype(df["card_bin"]), (
            "card_bin should be a string dtype, got "
            f"{df['card_bin'].dtype}. Use read_transactions_csv() "
            "to load transactions.csv."
        )

    def test_can_filter_anomaly_bin(self, tmp_path, transactions):
        """Filtering by ANOMALY_BIN string should work after read."""
        csv_path = tmp_path / "t.csv"
        transactions[["transaction_id", "card_bin"]].to_csv(csv_path, index=False)
        df = read_transactions_csv(csv_path)
        result = df[df["card_bin"] == ANOMALY_BIN]
        expected = transactions[transactions["card_bin"].astype(str) == ANOMALY_BIN]
        assert len(result) == len(expected)
