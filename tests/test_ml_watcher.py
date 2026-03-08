"""
tests/test_ml_watcher.py – Unit tests for the Proactive ML Watcher module.

All tests use in-memory DataFrames; no CSV files are required on disk.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from models.ml_watcher import (
    MerchantHealthMonitor,
    _FEATURE_COLUMNS,
    engineer_features,
    load_transactions,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

_BASE_TS = datetime(2026, 3, 7, 10, 0, 0, tzinfo=timezone.utc)


def _make_txn(
    n: int = 60,
    merchant_id: str = "merchant_id_1",
    status: str = "SUCCESS",
    decline_code: str | None = None,
    amount: float = 1500.0,
    start_ts: datetime = _BASE_TS,
    spread_seconds: int = 600,
) -> pd.DataFrame:
    """
    Build a small synthetic transactions DataFrame.

    Timestamps are evenly spread over *spread_seconds* seconds starting at
    *start_ts*.
    """
    rng = np.random.default_rng(0)
    offsets = np.linspace(0, spread_seconds, n, endpoint=False).astype(int)
    timestamps = [start_ts + timedelta(seconds=int(s)) for s in offsets]
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:05d}" for i in range(n)],
            "merchant_id":    merchant_id,
            "timestamp":      pd.to_datetime(timestamps, utc=True),
            "amount":         amount + rng.uniform(-10, 10, size=n),
            "currency":       "INR",
            "status":         status,
            "decline_code":   decline_code,
        }
    )


def _make_features(n_rows: int = 50) -> pd.DataFrame:
    """Build a synthetic feature DataFrame large enough for IsolationForest."""
    rng = np.random.default_rng(1)
    base = pd.DataFrame(
        {
            "merchant_id":        [f"merchant_id_{i % 5 + 1}" for i in range(n_rows)],
            "timestamp":          pd.date_range("2026-03-07", periods=n_rows, freq="10min", tz="UTC"),
            "total_transactions": rng.integers(5, 50, size=n_rows),
            "decline_count":      rng.integers(0, 10, size=n_rows),
            "avg_amount":         rng.uniform(100, 5000, size=n_rows),
            "risk_block_count":   rng.integers(0, 3, size=n_rows),
        }
    )
    base["decline_ratio"] = base["decline_count"] / base["total_transactions"]
    return base[["merchant_id", "timestamp"] + _FEATURE_COLUMNS]


# ──────────────────────────────────────────────────────────────────────────────
# load_transactions
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadTransactions:
    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_transactions(tmp_path / "missing.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"transaction_id": ["T1"], "amount": [100]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_transactions(csv)

    def test_timestamp_parsed_as_datetime(self, tmp_path):
        df = _make_txn(n=5)
        csv = tmp_path / "txn.csv"
        df.to_csv(csv, index=False)
        loaded = load_transactions(csv)
        assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])

    def test_returns_all_rows(self, tmp_path):
        df = _make_txn(n=10)
        csv = tmp_path / "txn.csv"
        df.to_csv(csv, index=False)
        loaded = load_transactions(csv)
        assert len(loaded) == 10


# ──────────────────────────────────────────────────────────────────────────────
# engineer_features
# ──────────────────────────────────────────────────────────────────────────────


class TestEngineerFeatures:
    def test_empty_input_returns_empty_dataframe(self):
        empty = pd.DataFrame(columns=["merchant_id", "timestamp", "amount", "status"])
        result = engineer_features(empty)
        assert result.empty
        assert set(_FEATURE_COLUMNS).issubset(set(result.columns))

    def test_output_columns(self):
        df = _make_txn(n=30)
        result = engineer_features(df, window="10min")
        expected = {"merchant_id", "timestamp"} | set(_FEATURE_COLUMNS)
        assert expected.issubset(set(result.columns))

    def test_total_transactions_is_positive(self):
        df = _make_txn(n=30)
        result = engineer_features(df, window="10min")
        assert (result["total_transactions"] > 0).all()

    def test_decline_count_all_zero_for_success_txns(self):
        df = _make_txn(n=20, status="SUCCESS")
        result = engineer_features(df, window="10min")
        assert (result["decline_count"] == 0).all()

    def test_decline_ratio_between_0_and_1(self):
        declined = _make_txn(n=10, status="DECLINED", decline_code="51_Insufficient_Funds")
        success  = _make_txn(n=10, status="SUCCESS")
        df = pd.concat([declined, success], ignore_index=True)
        result = engineer_features(df, window="10min")
        assert (result["decline_ratio"] >= 0.0).all()
        assert (result["decline_ratio"] <= 1.0).all()

    def test_decline_ratio_no_division_by_zero(self):
        """Empty windows must not produce NaN or inf in decline_ratio."""
        df = _make_txn(n=5, status="SUCCESS")
        result = engineer_features(df, window="1min")
        assert result["decline_ratio"].isna().sum() == 0

    def test_risk_block_count_correct(self):
        risk = _make_txn(n=15, status="DECLINED", decline_code="93_Risk_Block")
        other = _make_txn(n=5, status="DECLINED", decline_code="51_Insufficient_Funds")
        df = pd.concat([risk, other], ignore_index=True)
        result = engineer_features(df, window="10min")
        # All rows in a single window → risk_block_count should equal 15
        assert result["risk_block_count"].sum() == 15

    def test_avg_amount_approximately_correct(self):
        df = _make_txn(n=30, amount=2000.0)
        result = engineer_features(df, window="10min")
        # avg_amount should be close to 2000 (±10 from noise in _make_txn)
        assert (result["avg_amount"] > 1980).all()
        assert (result["avg_amount"] < 2020).all()

    def test_multiple_merchants_separated(self):
        m1 = _make_txn(n=20, merchant_id="merchant_id_1")
        m2 = _make_txn(n=20, merchant_id="merchant_id_2")
        df = pd.concat([m1, m2], ignore_index=True)
        result = engineer_features(df, window="10min")
        assert set(result["merchant_id"].unique()) == {"merchant_id_1", "merchant_id_2"}

    def test_missing_decline_code_column_handled(self):
        """decline_code is optional; should not raise even when absent."""
        df = _make_txn(n=20).drop(columns=["decline_code"])
        result = engineer_features(df, window="10min")
        assert "risk_block_count" in result.columns
        assert (result["risk_block_count"] == 0).all()

    def test_custom_window_produces_more_rows(self):
        """A narrower window produces more rows than a wider one."""
        df = _make_txn(n=60, spread_seconds=3600)
        wide  = engineer_features(df, window="60min")
        narrow = engineer_features(df, window="10min")
        assert len(narrow) >= len(wide)


# ──────────────────────────────────────────────────────────────────────────────
# MerchantHealthMonitor – train_and_predict
# ──────────────────────────────────────────────────────────────────────────────


class TestTrainAndPredict:
    def test_returns_tuple_of_df_and_array(self):
        features = _make_features(50)
        monitor = MerchantHealthMonitor()
        df_out, preds = monitor.train_and_predict(features)
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(preds, np.ndarray)

    def test_predictions_length_matches_rows(self):
        features = _make_features(50)
        monitor = MerchantHealthMonitor()
        _, preds = monitor.train_and_predict(features)
        assert len(preds) == len(features)

    def test_predictions_only_contain_1_and_minus1(self):
        features = _make_features(50)
        monitor = MerchantHealthMonitor()
        _, preds = monitor.train_and_predict(features)
        assert set(preds).issubset({1, -1})

    def test_is_trained_flag_set(self):
        features = _make_features(50)
        monitor = MerchantHealthMonitor()
        assert not monitor.is_trained
        monitor.train_and_predict(features)
        assert monitor.is_trained

    def test_raises_on_empty_dataframe(self):
        monitor = MerchantHealthMonitor()
        empty = pd.DataFrame(columns=["merchant_id", "timestamp"] + _FEATURE_COLUMNS)
        with pytest.raises(ValueError, match="empty"):
            monitor.train_and_predict(empty)

    def test_raises_on_missing_feature_columns(self):
        features = _make_features(30).drop(columns=["decline_ratio"])
        monitor = MerchantHealthMonitor()
        with pytest.raises(ValueError, match="missing required feature columns"):
            monitor.train_and_predict(features)

    def test_contamination_controls_anomaly_fraction(self):
        """At contamination=0.10 roughly 10 % of rows should be labelled -1."""
        features = _make_features(200)
        monitor = MerchantHealthMonitor(contamination=0.10)
        _, preds = monitor.train_and_predict(features)
        anomaly_frac = (preds == -1).mean()
        # IsolationForest guarantees exactly contamination * n anomalies
        assert abs(anomaly_frac - 0.10) < 0.02


# ──────────────────────────────────────────────────────────────────────────────
# MerchantHealthMonitor – generate_alerts
# ──────────────────────────────────────────────────────────────────────────────


class TestGenerateAlerts:
    def _run(self, features: pd.DataFrame) -> tuple[list, np.ndarray]:
        monitor = MerchantHealthMonitor(contamination=0.20)
        features, preds = monitor.train_and_predict(features)
        return monitor.generate_alerts(features, preds), preds

    def test_returns_list(self):
        features = _make_features(50)
        alerts, _ = self._run(features)
        assert isinstance(alerts, list)

    def test_alert_keys_present(self):
        features = _make_features(50)
        alerts, _ = self._run(features)
        required_keys = {"merchant_id", "timestamp", "alert_type", "metrics"}
        for alert in alerts:
            assert required_keys.issubset(set(alert.keys()))

    def test_metrics_keys_present(self):
        features = _make_features(50)
        alerts, _ = self._run(features)
        for alert in alerts:
            assert "decline_ratio" in alert["metrics"]
            assert "total_transactions" in alert["metrics"]

    def test_alert_count_matches_anomaly_count(self):
        features = _make_features(50)
        alerts, preds = self._run(features)
        assert len(alerts) == int((preds == -1).sum())

    def test_high_risk_block_alert_type(self):
        """Rows with risk_block_count > 10 should produce 'High Risk Block Spike'."""
        features = _make_features(50)
        # Force one row to look like a risk-block spike
        features = features.copy()
        features.loc[0, "risk_block_count"] = 15
        features.loc[0, "decline_ratio"]    = 0.9
        monitor = MerchantHealthMonitor(contamination=0.20)
        features, preds = monitor.train_and_predict(features)
        # Ensure row 0 is marked as anomaly
        preds[0] = -1
        alerts = monitor.generate_alerts(features, preds)
        risk_block_alerts = [a for a in alerts if a["alert_type"] == "High Risk Block Spike"]
        assert len(risk_block_alerts) >= 1

    def test_elevated_decline_ratio_alert_type(self):
        """Rows with decline_ratio > 0.6 but risk_block ≤ 10 → 'Elevated Decline Ratio'."""
        features = _make_features(50)
        features = features.copy()
        features.loc[1, "risk_block_count"] = 0
        features.loc[1, "decline_ratio"]    = 0.85
        monitor = MerchantHealthMonitor(contamination=0.20)
        features, preds = monitor.train_and_predict(features)
        preds[1] = -1
        alerts = monitor.generate_alerts(features, preds)
        target = [
            a for a in alerts
            if a["merchant_id"] == features.loc[1, "merchant_id"]
            and a["alert_type"] == "Elevated Decline Ratio"
        ]
        assert len(target) >= 1

    def test_general_anomaly_alert_type(self):
        """Rows with low risk_block and low decline_ratio → 'General Transaction Anomaly'."""
        features = _make_features(50)
        features = features.copy()
        features.loc[2, "risk_block_count"] = 0
        features.loc[2, "decline_ratio"]    = 0.1
        monitor = MerchantHealthMonitor(contamination=0.20)
        features, preds = monitor.train_and_predict(features)
        preds[2] = -1
        alerts = monitor.generate_alerts(features, preds)
        general = [a for a in alerts if a["alert_type"] == "General Transaction Anomaly"]
        assert len(general) >= 1

    def test_no_anomalies_returns_empty_list(self):
        features = _make_features(50)
        monitor = MerchantHealthMonitor()
        features, _ = monitor.train_and_predict(features)
        all_normal = np.ones(len(features), dtype=int)
        alerts = monitor.generate_alerts(features, all_normal)
        assert alerts == []

    def test_raises_on_mismatched_lengths(self):
        features = _make_features(50)
        monitor = MerchantHealthMonitor()
        features, preds = monitor.train_and_predict(features)
        with pytest.raises(ValueError, match="aligned"):
            monitor.generate_alerts(features, preds[:-1])

    def test_timestamp_is_string_in_alert(self):
        features = _make_features(50)
        monitor = MerchantHealthMonitor(contamination=0.20)
        features, preds = monitor.train_and_predict(features)
        alerts = monitor.generate_alerts(features, preds)
        for alert in alerts:
            assert isinstance(alert["timestamp"], str)

    def test_decline_ratio_rounded_to_4dp(self):
        features = _make_features(50)
        monitor = MerchantHealthMonitor(contamination=0.20)
        features, preds = monitor.train_and_predict(features)
        alerts = monitor.generate_alerts(features, preds)
        for alert in alerts:
            ratio = alert["metrics"]["decline_ratio"]
            assert ratio == round(ratio, 4)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end smoke test
# ──────────────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_full_pipeline_on_synthetic_data(self):
        """
        Simulate the complete pipeline: raw transactions → features → model
        → alerts, using in-memory data that mimics the real CSV schema.
        """
        # Mix of normal + a card-testing burst (anomaly 3 pattern)
        normal    = _make_txn(n=200, merchant_id="merchant_id_1", status="SUCCESS",
                               amount=2000.0, spread_seconds=7200)
        declined  = _make_txn(n=50,  merchant_id="merchant_id_1", status="DECLINED",
                               decline_code="51_Insufficient_Funds", amount=2000.0,
                               start_ts=_BASE_TS + timedelta(hours=1),
                               spread_seconds=1800)
        card_test = _make_txn(n=100, merchant_id="merchant_id_3", status="DECLINED",
                               decline_code="14_Invalid_Card_Number", amount=3.0,
                               start_ts=_BASE_TS + timedelta(hours=2),
                               spread_seconds=300)

        df = pd.concat([normal, declined, card_test], ignore_index=True)
        features = engineer_features(df, window="10min")

        monitor = MerchantHealthMonitor(contamination=0.10)
        features, preds = monitor.train_and_predict(features)
        alerts = monitor.generate_alerts(features, preds)

        # Basic sanity: pipeline ran without errors and produced some output
        assert isinstance(alerts, list)
        assert len(features) > 0
        assert set(preds).issubset({1, -1})
