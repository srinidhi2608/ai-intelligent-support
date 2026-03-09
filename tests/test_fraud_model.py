"""
tests/test_fraud_model.py – Unit tests for the FraudDetector ML model.
"""

import numpy as np
import pandas as pd
import pytest

from models.fraud_model import FraudDetector


def _make_df(n: int = 20) -> pd.DataFrame:
    """Create a minimal numeric DataFrame for testing."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "amount": rng.uniform(10, 50000, size=n),
            "hour_of_day": rng.integers(0, 24, size=n),
            "tx_count_last_hour": rng.integers(1, 50, size=n),
        }
    )


class TestFraudDetectorAnomaly:
    def test_init_default_mode(self):
        detector = FraudDetector()
        assert detector.mode == "anomaly"
        assert not detector.is_trained

    def test_train_sets_flag(self):
        detector = FraudDetector(mode="anomaly")
        detector.train(_make_df())
        assert detector.is_trained

    def test_predict_returns_array(self):
        detector = FraudDetector(mode="anomaly")
        df = _make_df()
        detector.train(df)
        predictions = detector.predict(df)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(df)

    def test_predict_before_train_raises(self):
        detector = FraudDetector(mode="anomaly")
        with pytest.raises(RuntimeError):
            detector.predict(_make_df())


class TestFraudDetectorSupervised:
    def test_train_and_predict_with_labels(self):
        detector = FraudDetector(mode="supervised")
        df = _make_df(n=50)
        rng = np.random.default_rng(1)
        labels = pd.Series(rng.integers(0, 2, size=50))
        detector.train(df, labels=labels)
        predictions = detector.predict(df)
        assert set(predictions).issubset({0, 1})

    def test_supervised_requires_labels(self):
        detector = FraudDetector(mode="supervised")
        with pytest.raises(ValueError):
            detector.train(_make_df())

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            FraudDetector(mode="unknown_mode")
