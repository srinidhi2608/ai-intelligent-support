"""
tests/test_ml_comparison.py – Unit tests for the ML Model Comparison Dashboard.

Strategy
--------
* Tests exercise data-loading helpers, ground-truth creation, feature
  engineering, model training, metric calculation, and prediction mapping
  using small controlled DataFrames.
* No CSV files or running Streamlit server are required.
* A Streamlit ``AppTest`` smoke-test verifies the page renders without error
  when CSV files are present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers – tiny DataFrames used across tests
# ---------------------------------------------------------------------------

def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "TXN-001", "TXN-002", "TXN-003", "TXN-004",
                "TXN-005", "TXN-006",
            ],
            "merchant_id": ["m1", "m1", "m2", "m2", "m1", "m2"],
            "timestamp": [
                "2025-06-01 10:00:00",
                "2025-06-01 11:00:00",
                "2025-06-01 10:30:00",
                "2025-06-01 12:00:00",
                "2025-06-01 13:00:00",
                "2025-06-01 14:00:00",
            ],
            "amount": [100.0, 200.0, 150.0, 50.0, 300.0, 75.0],
            "currency": ["INR"] * 6,
            "status": [
                "SUCCESS", "DECLINED", "SUCCESS", "DECLINED", "SUCCESS", "SUCCESS",
            ],
            "decline_code": [
                None, "51_Insufficient_Funds", None, "93_Risk_Block", None, None,
            ],
            "card_bin": ["411111"] * 6,
        }
    )


def _sample_webhook_logs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_id": [
                "WH-001", "WH-002", "WH-003", "WH-004", "WH-005", "WH-006",
            ],
            "transaction_id": [
                "TXN-001", "TXN-002", "TXN-003", "TXN-004",
                "TXN-005", "TXN-006",
            ],
            "timestamp": [
                "2025-06-01 10:00:05",
                "2025-06-01 11:00:03",
                "2025-06-01 10:30:02",
                "2025-06-01 12:00:10",
                "2025-06-01 13:00:01",
                "2025-06-01 14:00:08",
            ],
            "event_type": [
                "payment.success", "payment.failed", "payment.success",
                "payment.failed", "payment.success", "payment.success",
            ],
            "http_status": [200, 500, 200, 404, 401, 200],
            "delivery_attempts": [1, 3, 1, 2, 1, 1],
            "latency_ms": [120, 1500, 80, 950, 200, 150],
        }
    )


def _build_merged() -> pd.DataFrame:
    """Merge the sample DataFrames the same way the dashboard does."""
    txns = _sample_transactions()
    wh = _sample_webhook_logs()
    merged = txns.merge(wh, on="transaction_id", how="left", suffixes=("", "_wh"))
    return merged


# ---------------------------------------------------------------------------
# Tests – Ground truth creation
# ---------------------------------------------------------------------------

class TestGroundTruth:
    """Verify is_anomaly_actual is derived correctly."""

    def test_risk_block_flagged(self):
        """decline_code == '93_Risk_Block' → is_anomaly_actual == 1."""
        merged = _build_merged()
        risk_block = merged["decline_code"] == "93_Risk_Block"
        auth_fail = merged["http_status"] == 401
        merged["is_anomaly_actual"] = (risk_block | auth_fail).astype(int)

        # TXN-004 has 93_Risk_Block
        row = merged[merged["transaction_id"] == "TXN-004"].iloc[0]
        assert row["is_anomaly_actual"] == 1

    def test_http_401_flagged(self):
        """http_status == 401 → is_anomaly_actual == 1."""
        merged = _build_merged()
        risk_block = merged["decline_code"] == "93_Risk_Block"
        auth_fail = merged["http_status"] == 401
        merged["is_anomaly_actual"] = (risk_block | auth_fail).astype(int)

        # TXN-005 has http_status 401
        row = merged[merged["transaction_id"] == "TXN-005"].iloc[0]
        assert row["is_anomaly_actual"] == 1

    def test_normal_transaction_not_flagged(self):
        """A normal transaction should have is_anomaly_actual == 0."""
        merged = _build_merged()
        risk_block = merged["decline_code"] == "93_Risk_Block"
        auth_fail = merged["http_status"] == 401
        merged["is_anomaly_actual"] = (risk_block | auth_fail).astype(int)

        # TXN-001: SUCCESS, http_status 200
        row = merged[merged["transaction_id"] == "TXN-001"].iloc[0]
        assert row["is_anomaly_actual"] == 0

    def test_non_risk_decline_not_flagged(self):
        """A non-93_Risk_Block decline should not be flagged."""
        merged = _build_merged()
        risk_block = merged["decline_code"] == "93_Risk_Block"
        auth_fail = merged["http_status"] == 401
        merged["is_anomaly_actual"] = (risk_block | auth_fail).astype(int)

        # TXN-002: DECLINED with 51_Insufficient_Funds, http_status 500
        row = merged[merged["transaction_id"] == "TXN-002"].iloc[0]
        assert row["is_anomaly_actual"] == 0

    def test_total_anomaly_count(self):
        """Exactly 2 anomalies in sample data (TXN-004 + TXN-005)."""
        merged = _build_merged()
        risk_block = merged["decline_code"] == "93_Risk_Block"
        auth_fail = merged["http_status"] == 401
        merged["is_anomaly_actual"] = (risk_block | auth_fail).astype(int)

        assert merged["is_anomaly_actual"].sum() == 2

    def test_ground_truth_with_missing_decline_code(self):
        """Rows with NaN decline_code and non-401 http_status → normal."""
        merged = _build_merged()
        risk_block = merged["decline_code"] == "93_Risk_Block"
        auth_fail = merged["http_status"] == 401
        merged["is_anomaly_actual"] = (risk_block | auth_fail).astype(int)

        # TXN-003: SUCCESS, no decline_code, http_status 200
        row = merged[merged["transaction_id"] == "TXN-003"].iloc[0]
        assert row["is_anomaly_actual"] == 0


# ---------------------------------------------------------------------------
# Tests – Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Verify feature columns are prepared correctly."""

    def test_amount_is_numeric(self):
        merged = _build_merged()
        merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce").fillna(0.0)
        assert merged["amount"].dtype in (np.float64, np.float32)

    def test_http_status_is_int(self):
        merged = _build_merged()
        merged["http_status"] = (
            pd.to_numeric(merged["http_status"], errors="coerce").fillna(0).astype(int)
        )
        assert merged["http_status"].dtype in (np.int64, np.int32)

    def test_delivery_attempts_is_int(self):
        merged = _build_merged()
        merged["delivery_attempts"] = (
            pd.to_numeric(merged["delivery_attempts"], errors="coerce").fillna(0).astype(int)
        )
        assert merged["delivery_attempts"].dtype in (np.int64, np.int32)

    def test_nan_amount_filled_as_zero(self):
        df = pd.DataFrame({"amount": [100.0, None, "bad"]})
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        assert df["amount"].iloc[1] == 0.0
        assert df["amount"].iloc[2] == 0.0

    def test_missing_feature_column_defaults(self):
        """When a feature column is absent, it should default to 0."""
        df = pd.DataFrame({"transaction_id": ["TXN-001"]})
        if "amount" not in df.columns:
            df["amount"] = 0.0
        assert df["amount"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Tests – Prediction mapping
# ---------------------------------------------------------------------------

class TestPredictionMapping:
    """Verify sklearn's -1/1 output maps to 1/0 for anomaly/normal."""

    def test_outlier_maps_to_anomaly(self):
        raw = np.array([-1, 1, -1, 1])
        y_pred = (raw == -1).astype(int)
        assert y_pred[0] == 1
        assert y_pred[2] == 1

    def test_inlier_maps_to_normal(self):
        raw = np.array([-1, 1, -1, 1])
        y_pred = (raw == -1).astype(int)
        assert y_pred[1] == 0
        assert y_pred[3] == 0

    def test_mapping_preserves_length(self):
        raw = np.array([-1, 1, 1, -1, 1])
        y_pred = (raw == -1).astype(int)
        assert len(y_pred) == len(raw)


# ---------------------------------------------------------------------------
# Tests – Confusion count calculations
# ---------------------------------------------------------------------------

class TestConfusionCounts:
    """Verify TP, FP, FN, TN counts are computed correctly."""

    def test_tp_count(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        assert tp == 2  # indices 0, 4

    def test_fp_count(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        assert fp == 1  # index 3

    def test_fn_count(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        assert fn == 1  # index 2

    def test_tn_count(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        assert tn == 1  # index 1

    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        assert tp == 2
        assert fp == 0
        assert fn == 0
        assert tn == 2

    def test_counts_sum_to_total(self):
        """TP + FP + FN + TN should equal the total sample count."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        assert tp + fp + fn + tn == len(y_true)


# ---------------------------------------------------------------------------
# Tests – Metrics DataFrame structure
# ---------------------------------------------------------------------------

class TestMetricsDataFrame:
    """Verify the long-form metrics DataFrame has correct shape & content."""

    def test_metrics_df_columns(self):
        rows = [
            {"Model": "IsolationForest", "Metric": "Precision", "Score": 0.5},
            {"Model": "IsolationForest", "Metric": "Recall", "Score": 0.7},
            {"Model": "IsolationForest", "Metric": "F1-Score", "Score": 0.58},
        ]
        df = pd.DataFrame(rows)
        assert list(df.columns) == ["Model", "Metric", "Score"]

    def test_metrics_df_row_count(self):
        """3 models × 3 metrics = 9 rows."""
        model_names = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        metric_names = ["Precision", "Recall", "F1-Score"]
        rows = [
            {"Model": m, "Metric": met, "Score": 0.5}
            for m in model_names
            for met in metric_names
        ]
        df = pd.DataFrame(rows)
        assert len(df) == 9

    def test_pivot_produces_correct_shape(self):
        model_names = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        metric_names = ["Precision", "Recall", "F1-Score"]
        rows = [
            {"Model": m, "Metric": met, "Score": 0.5}
            for m in model_names
            for met in metric_names
        ]
        df = pd.DataFrame(rows)
        pivot = df.pivot(index="Model", columns="Metric", values="Score")
        assert pivot.shape == (3, 3)


# ---------------------------------------------------------------------------
# Tests – Confusion matrix & outcome labeling (Deep Dive visualisations)
# ---------------------------------------------------------------------------

class TestConfusionMatrixData:
    """Verify the confusion matrix data structures used by the heatmap."""

    def test_cm_array_shape(self):
        """Confusion matrix should be a 2×2 array."""
        counts = {"TP": 10, "FP": 5, "FN": 3, "TN": 82}
        cm = np.array([[counts["TN"], counts["FP"]],
                        [counts["FN"], counts["TP"]]])
        assert cm.shape == (2, 2)

    def test_cm_values_match_counts(self):
        counts = {"TP": 10, "FP": 5, "FN": 3, "TN": 82}
        cm = np.array([[counts["TN"], counts["FP"]],
                        [counts["FN"], counts["TP"]]])
        assert cm[0, 0] == 82   # TN (top-left)
        assert cm[0, 1] == 5    # FP (top-right)
        assert cm[1, 0] == 3    # FN (bottom-left)
        assert cm[1, 1] == 10   # TP (bottom-right)


class TestOutcomeLabeling:
    """Verify outcome labeling logic used for the box plot."""

    def test_outcome_labels_assigned_correctly(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        labels = ["TP (True Positive)", "FP (False Positive)",
                  "FN (False Negative)", "TN (True Negative)"]
        conditions = [
            (y_pred == 1) & (y_true == 1),
            (y_pred == 1) & (y_true == 0),
            (y_pred == 0) & (y_true == 1),
            (y_pred == 0) & (y_true == 0),
        ]
        outcomes = np.select(conditions, labels, default="TN (True Negative)")
        assert outcomes[0] == "TP (True Positive)"   # pred=1, true=1
        assert outcomes[1] == "FP (False Positive)"   # pred=1, true=0
        assert outcomes[2] == "FN (False Negative)"   # pred=0, true=1
        assert outcomes[3] == "TN (True Negative)"    # pred=0, true=0

    def test_all_outcomes_present(self):
        """When all four quadrants have data, all labels are present."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        labels = ["TP (True Positive)", "FP (False Positive)",
                  "FN (False Negative)", "TN (True Negative)"]
        conditions = [
            (y_pred == 1) & (y_true == 1),
            (y_pred == 1) & (y_true == 0),
            (y_pred == 0) & (y_true == 1),
            (y_pred == 0) & (y_true == 0),
        ]
        outcomes = np.select(conditions, labels, default="TN (True Negative)")
        assert set(outcomes) == set(labels)


# ---------------------------------------------------------------------------
# Tests – Scores are in valid range
# ---------------------------------------------------------------------------

class TestScoreRange:
    """Precision, recall, and F1-score should be between 0 and 1."""

    def test_precision_in_range(self):
        from sklearn.metrics import precision_score
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        score = precision_score(y_true, y_pred, zero_division=0)
        assert 0.0 <= score <= 1.0

    def test_recall_in_range(self):
        from sklearn.metrics import recall_score
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        score = recall_score(y_true, y_pred, zero_division=0)
        assert 0.0 <= score <= 1.0

    def test_f1_in_range(self):
        from sklearn.metrics import f1_score
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        score = f1_score(y_true, y_pred, zero_division=0)
        assert 0.0 <= score <= 1.0

    def test_zero_division_handled(self):
        """When there are no true positives, zero_division=0 avoids errors."""
        from sklearn.metrics import precision_score
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        score = precision_score(y_true, y_pred, zero_division=0)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests – End-to-end model training on small data
# ---------------------------------------------------------------------------

class TestModelTraining:
    """Smoke tests ensuring models run without errors on small DataFrames."""

    @pytest.fixture()
    def small_dataset(self):
        """Create a small merged dataset with ground truth."""
        merged = _build_merged()
        merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce").fillna(0.0)
        merged["http_status"] = (
            pd.to_numeric(merged["http_status"], errors="coerce").fillna(0).astype(int)
        )
        merged["delivery_attempts"] = (
            pd.to_numeric(merged["delivery_attempts"], errors="coerce").fillna(0).astype(int)
        )
        risk_block = merged["decline_code"] == "93_Risk_Block"
        auth_fail = merged["http_status"] == 401
        merged["is_anomaly_actual"] = (risk_block | auth_fail).astype(int)
        return merged

    def test_isolation_forest_runs(self, small_dataset):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(
            small_dataset[["amount", "http_status", "delivery_attempts"]]
        )
        model = IsolationForest(n_estimators=50, contamination=0.2, random_state=42)
        raw = model.fit_predict(X)
        y_pred = (raw == -1).astype(int)
        assert len(y_pred) == len(small_dataset)
        assert set(y_pred).issubset({0, 1})

    def test_one_class_svm_runs(self, small_dataset):
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(
            small_dataset[["amount", "http_status", "delivery_attempts"]]
        )
        model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.2)
        raw = model.fit(X).predict(X)
        y_pred = (raw == -1).astype(int)
        assert len(y_pred) == len(small_dataset)
        assert set(y_pred).issubset({0, 1})

    def test_lof_runs(self, small_dataset):
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(
            small_dataset[["amount", "http_status", "delivery_attempts"]]
        )
        model = LocalOutlierFactor(n_neighbors=2, contamination=0.2, novelty=False)
        raw = model.fit_predict(X)
        y_pred = (raw == -1).astype(int)
        assert len(y_pred) == len(small_dataset)
        assert set(y_pred).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests – AppTest smoke-test (page renders without error)
# ---------------------------------------------------------------------------

class TestMLComparisonPageRender:
    """Verify the ML Comparison page renders without exceptions via AppTest."""

    def test_page_renders_with_empty_data(self):
        """The page must not crash when CSV files are missing."""
        from streamlit.testing.v1 import AppTest
        import streamlit as st

        st.cache_data.clear()
        st.cache_resource.clear()

        at = AppTest.from_file("pages/2_🧠_ML_Comparison.py")
        at.run(timeout=10)

        assert not at.exception

    def test_page_renders_with_sample_data(self, tmp_path, monkeypatch):
        """The page renders correctly when CSV files contain data."""
        from streamlit.testing.v1 import AppTest
        import streamlit as st

        st.cache_data.clear()
        st.cache_resource.clear()

        _sample_transactions().to_csv(tmp_path / "transactions.csv", index=False)
        _sample_webhook_logs().to_csv(tmp_path / "webhook_logs.csv", index=False)

        monkeypatch.setenv("ANALYTICS_DATA_DIR", str(tmp_path))

        at = AppTest.from_file("pages/2_🧠_ML_Comparison.py")
        at.run(timeout=10)

        assert not at.exception
