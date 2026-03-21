"""
tests/test_generate_realistic_data.py – Tests for the statistically realistic
                                         payment-gateway data generator.

Validates:
  • Schema and row-count expectations for all three DataFrames
  • Pareto distribution produces heavy-tailed merchant volumes
  • MCC-based amounts are clipped to a minimum of ₹50
  • Time-of-day seasonality is present
  • All three demo anomalies are correctly injected
  • Relational integrity (transaction_id links txn ↔ webhook_logs)
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from generate_realistic_data import (
    NOW,
    TARGET_TRANSACTIONS,
    WINDOW_DAYS,
    generate_merchants,
    generate_transactions,
    generate_webhook_logs,
    inject_demo_anomalies,
    read_transactions_csv,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures – generate once per test session for speed
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def merchants() -> pd.DataFrame:
    return generate_merchants()


@pytest.fixture(scope="module")
def raw_transactions(merchants) -> pd.DataFrame:
    """Transactions before demo anomaly injection."""
    return generate_transactions(merchants)


@pytest.fixture(scope="module")
def raw_webhooks(raw_transactions) -> pd.DataFrame:
    """Webhook logs before demo anomaly injection."""
    return generate_webhook_logs(raw_transactions)


@pytest.fixture(scope="module")
def transactions(raw_transactions, raw_webhooks) -> pd.DataFrame:
    """Transactions after demo anomaly injection."""
    txns, _ = inject_demo_anomalies(raw_transactions.copy(), raw_webhooks.copy())
    return txns


@pytest.fixture(scope="module")
def webhooks(raw_transactions, raw_webhooks) -> pd.DataFrame:
    """Webhook logs after demo anomaly injection."""
    _, wh = inject_demo_anomalies(raw_transactions.copy(), raw_webhooks.copy())
    return wh


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 – Merchant schema
# ──────────────────────────────────────────────────────────────────────────────


class TestMerchants:
    def test_row_count(self, merchants):
        assert len(merchants) == 25

    def test_columns(self, merchants):
        expected = {"merchant_id", "business_name", "mcc_code", "webhook_url"}
        assert expected.issubset(set(merchants.columns))

    def test_merchant_ids_unique(self, merchants):
        assert merchants["merchant_id"].nunique() == 25

    def test_merchant_ids_follow_pattern(self, merchants):
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

    def test_row_count_near_target(self, transactions):
        """~50,000 base + 1 demo txn + 150 auth-block = ~50,151."""
        assert TARGET_TRANSACTIONS <= len(transactions) <= TARGET_TRANSACTIONS + 500

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

    def test_amounts_at_least_50(self, transactions):
        assert (transactions["amount"] >= 50).all()

    def test_no_negative_amounts(self, transactions):
        assert (transactions["amount"] > 0).all()

    def test_transaction_ids_unique(self, transactions):
        assert transactions["transaction_id"].nunique() == len(transactions)


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – Pareto distribution verification
# ──────────────────────────────────────────────────────────────────────────────


class TestParetoDistribution:
    def test_merchant_volumes_are_heavy_tailed(self, transactions):
        """Top merchant should have significantly more transactions than bottom."""
        counts = transactions["merchant_id"].value_counts()
        top = counts.iloc[0]
        bottom = counts.iloc[-1]
        assert top > 2 * bottom, (
            f"Top merchant has {top}, bottom has {bottom} – "
            f"expected a heavy-tail ratio > 2x"
        )

    def test_not_all_merchants_equal(self, transactions):
        """Transaction counts should vary meaningfully across merchants."""
        counts = transactions["merchant_id"].value_counts()
        assert counts.std() > 100, "Merchant volumes should have significant variance"


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – Time-of-day seasonality
# ──────────────────────────────────────────────────────────────────────────────


class TestSeasonality:
    def test_peak_hours_have_more_transactions(self, transactions):
        """UTC 5–9 (IST peak) should have more transactions than UTC 20–23 (off-peak)."""
        ts = pd.to_datetime(transactions["timestamp"])
        hours = ts.dt.hour
        peak_count = int(hours.isin([5, 6, 7, 8, 9]).sum())
        off_count = int(hours.isin([20, 21, 22, 23]).sum())
        assert peak_count > off_count, (
            f"Peak hours: {peak_count}, off-peak hours: {off_count}"
        )

    def test_distribution_is_not_uniform(self, transactions):
        """Hourly transaction counts should not be uniform."""
        ts = pd.to_datetime(transactions["timestamp"])
        hourly_counts = ts.dt.hour.value_counts()
        assert hourly_counts.std() > 50, "Hourly distribution should not be uniform"


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 – Demo Anomaly 1: TXN-00194400
# ──────────────────────────────────────────────────────────────────────────────


class TestDemoAnomaly1:
    def _get_row(self, transactions):
        return transactions[transactions["transaction_id"] == "TXN-00194400"]

    def test_txn_exists(self, transactions):
        assert len(self._get_row(transactions)) == 1

    def test_merchant_is_merchant_id_1(self, transactions):
        row = self._get_row(transactions)
        assert row.iloc[0]["merchant_id"] == "merchant_id_1"

    def test_status_is_declined(self, transactions):
        row = self._get_row(transactions)
        assert row.iloc[0]["status"] == "DECLINED"

    def test_decline_code_is_93_risk_block(self, transactions):
        row = self._get_row(transactions)
        assert row.iloc[0]["decline_code"] == "93_Risk_Block"


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 – Demo Anomaly 2: TXN-00000004 / WH-00000004
# ──────────────────────────────────────────────────────────────────────────────


class TestDemoAnomaly2:
    def test_txn_00000004_exists(self, transactions):
        row = transactions[transactions["transaction_id"] == "TXN-00000004"]
        assert len(row) == 1

    def test_txn_00000004_merchant_is_merchant_id_5(self, transactions):
        row = transactions[transactions["transaction_id"] == "TXN-00000004"]
        assert row.iloc[0]["merchant_id"] == "merchant_id_5"

    def test_txn_00000004_status_is_success(self, transactions):
        row = transactions[transactions["transaction_id"] == "TXN-00000004"]
        assert row.iloc[0]["status"] == "SUCCESS"

    def test_wh_00000004_exists(self, webhooks):
        row = webhooks[webhooks["log_id"] == "WH-00000004"]
        assert len(row) == 1

    def test_wh_00000004_http_status_is_500(self, webhooks):
        row = webhooks[webhooks["log_id"] == "WH-00000004"]
        assert int(row.iloc[0]["http_status"]) == 500

    def test_wh_00000004_links_to_txn_00000004(self, webhooks):
        row = webhooks[webhooks["log_id"] == "WH-00000004"]
        assert row.iloc[0]["transaction_id"] == "TXN-00000004"


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 – Demo Anomaly 3: merchant_id_2 auth-failure block
# ──────────────────────────────────────────────────────────────────────────────


class TestDemoAnomaly3:
    def test_at_least_150_401_webhooks_for_merchant_2(self, transactions, webhooks):
        m2_txn_ids = transactions[
            transactions["merchant_id"] == "merchant_id_2"
        ]["transaction_id"]
        m2_wh = webhooks[webhooks["transaction_id"].isin(m2_txn_ids)]
        auth_failures = m2_wh[m2_wh["http_status"] == 401]
        assert len(auth_failures) >= 150

    def test_auth_block_has_exactly_150_injected_logs(self, webhooks):
        auth_wh = webhooks[webhooks["log_id"].str.startswith("WH-AUTH-")]
        assert len(auth_wh) == 150

    def test_all_auth_block_logs_are_401(self, webhooks):
        auth_wh = webhooks[webhooks["log_id"].str.startswith("WH-AUTH-")]
        assert (auth_wh["http_status"] == 401).all()

    def test_auth_block_within_reasonable_window(self, webhooks):
        """The 150 injected webhooks should span roughly one hour."""
        auth_wh = webhooks[webhooks["log_id"].str.startswith("WH-AUTH-")]
        ts = pd.to_datetime(auth_wh["timestamp"])
        window_seconds = (ts.max() - ts.min()).total_seconds()
        assert window_seconds <= 7200, (
            f"Auth-failure block spans {window_seconds}s (expected ≤ 7200s)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 – Webhook schema and relational integrity
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
# Section 9 – read_transactions_csv helper
# ──────────────────────────────────────────────────────────────────────────────


class TestReadTransactionsCsv:
    def test_card_bin_is_str_dtype(self, tmp_path, transactions):
        sample = transactions[["transaction_id", "card_bin", "amount"]].head(20)
        csv_path = tmp_path / "sample.csv"
        sample.to_csv(csv_path, index=False)

        df = read_transactions_csv(csv_path)
        assert pd.api.types.is_string_dtype(df["card_bin"]), (
            f"card_bin should be string, got {df['card_bin'].dtype}"
        )
