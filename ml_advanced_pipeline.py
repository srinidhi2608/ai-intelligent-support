"""
ml_advanced_pipeline.py – Advanced ML Anomaly Detection Pipeline with
                          LangChain Agent Integration.

This module provides an end-to-end pipeline for detecting anomalous
merchant transaction behaviour using an Isolation Forest model, and
automatically triggering a LangChain agent to investigate and remediate
detected anomalies.

Typical usage::

    from ml_advanced_pipeline import detect_anomalies, trigger_agent_for_anomaly

    features_df = build_features(raw_df)
    results     = detect_anomalies(features_df)
    anomaly_row = results[results["prediction"] == 1].iloc[0]
    trigger_agent_for_anomaly(anomaly_row)

Or run the built-in edge-case tests::

    python ml_advanced_pipeline.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    raise ImportError(
        "scikit-learn is required but not installed. "
        "Install it with: pip install scikit-learn  "
        "(or install all project dependencies with: pip install -r requirements.txt)"
    ) from None

from agents.agent_orchestrator import get_agent

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Numeric feature columns fed to the Isolation Forest
FEATURE_COLUMNS: list[str] = [
    "total_transactions",
    "decline_count",
    "decline_ratio",
    "avg_amount",
]


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 – Feature engineering
# ──────────────────────────────────────────────────────────────────────────────


def build_features(df: pd.DataFrame, window: str = "10min") -> pd.DataFrame:
    """
    Aggregate raw transactions into merchant-level, rolling-window features.

    Parameters
    ----------
    df:
        Raw transactions DataFrame with at least ``merchant_id``,
        ``timestamp``, ``transaction_id``, ``amount``, and ``status``.
    window:
        Pandas offset alias for the aggregation window (default ``"10min"``).

    Returns
    -------
    pd.DataFrame
        One row per (merchant_id, time-window) with feature columns.
    """
    if df.empty:
        return pd.DataFrame(columns=["merchant_id"] + FEATURE_COLUMNS)

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    grouped = df.groupby(
        ["merchant_id", pd.Grouper(key="timestamp", freq=window)]
    )

    agg = grouped.agg(
        total_transactions=("transaction_id", "count"),
        decline_count=("status", lambda s: int((s == "DECLINED").sum())),
        avg_amount=("amount", "mean"),
    ).reset_index()

    agg = agg[agg["total_transactions"] > 0].copy()
    agg["decline_ratio"] = np.where(
        agg["total_transactions"] > 0,
        agg["decline_count"] / agg["total_transactions"],
        0.0,
    )

    return agg[["merchant_id"] + FEATURE_COLUMNS].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 – Anomaly detection
# ──────────────────────────────────────────────────────────────────────────────


def detect_anomalies(
    features_df: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run an Isolation Forest on *features_df* and return predictions.

    The returned DataFrame is a copy of *features_df* with an extra
    ``prediction`` column:

    * ``1`` → anomaly
    * ``0`` → normal

    Parameters
    ----------
    features_df:
        DataFrame with columns listed in :data:`FEATURE_COLUMNS`.
    contamination:
        Expected anomaly fraction (default 0.1).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        *features_df* augmented with a ``prediction`` column.
    """
    if features_df.empty:
        raise ValueError("features_df is empty — nothing to score.")

    X = features_df[FEATURE_COLUMNS].values
    model = IsolationForest(
        contamination=contamination, random_state=random_state
    )
    raw_preds = model.fit_predict(X)

    # Convert IsolationForest convention (-1=anomaly, 1=normal)
    # to a more intuitive convention   (1=anomaly,  0=normal).
    result = features_df.copy()
    result["prediction"] = np.where(raw_preds == -1, 1, 0)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – Agent integration
# ──────────────────────────────────────────────────────────────────────────────


def trigger_agent_for_anomaly(anomaly_row) -> None:
    """
    Trigger the LangChain agent to investigate a detected anomaly.

    Parameters
    ----------
    anomaly_row : dict or pd.Series
        A single row of anomaly data.  Must contain a ``merchant_id``
        key.  Typically this is a row from the DataFrame returned by
        :func:`detect_anomalies` where ``prediction == 1``.

    Raises
    ------
    ValueError
        If *anomaly_row* does not contain a ``merchant_id`` key.
    """
    agent_executor = get_agent()

    merchant_id = anomaly_row.get("merchant_id")
    if not merchant_id:
        raise ValueError(
            "anomaly_row must contain a 'merchant_id' key with a non-empty value."
        )

    alert_message = (
        f"SYSTEM ALERT: The ML Watcher has detected a high volume of "
        f"failed transactions for {merchant_id}. Please investigate the "
        f"root cause and take necessary action."
    )

    response = agent_executor.invoke({"input": alert_message})

    output = response.get("output", "No response from agent.")
    logger.info("Agent response for %s: %s", merchant_id, output)
    print(output)


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – Edge-case tests
# ──────────────────────────────────────────────────────────────────────────────


def test_edge_cases():
    """
    Validate the anomaly-detection pipeline with synthetic edge cases and
    demonstrate the agent handoff for a confirmed anomaly.
    """
    print("=" * 70)
    print("  ML Advanced Pipeline — Edge Case Tests")
    print("=" * 70)

    # ── Build synthetic feature data ──────────────────────────────────────
    # Normal merchants: low decline ratios, moderate transaction volumes.
    np.random.seed(42)
    normal_rows = []
    for i in range(1, 21):
        if i == 2:
            continue  # Reserve merchant_id_2 for the anomaly case
        normal_rows.append(
            {
                "merchant_id": f"merchant_id_{i}",
                "total_transactions": np.random.randint(50, 200),
                "decline_count": np.random.randint(0, 10),
                "decline_ratio": np.random.uniform(0.0, 0.1),
                "avg_amount": np.random.uniform(50.0, 500.0),
            }
        )

    # Positive Case 1: 401 webhook spike for merchant_id_2
    # A sudden burst of 150 transactions where 148 were declined — a
    # classic card-testing / auth-failure spike pattern.
    anomaly_case = {
        "merchant_id": "merchant_id_2",
        "total_transactions": 150,
        "decline_count": 148,
        "decline_ratio": 148 / 150,
        "avg_amount": 2.50,
    }
    normal_rows.append(anomaly_case)

    features_df = pd.DataFrame(normal_rows)

    # ── Run anomaly detection ─────────────────────────────────────────────
    results = detect_anomalies(features_df, contamination=0.1)

    # ── Positive Case 1: Assert merchant_id_2 is flagged ─────────────────
    merchant_2_rows = results[results["merchant_id"] == "merchant_id_2"]
    assert not merchant_2_rows.empty, (
        "Positive Case 1 FAILED: merchant_id_2 row not found in results."
    )

    merchant_2_row = merchant_2_rows.iloc[0]
    prediction = merchant_2_row["prediction"]

    assert prediction == 1, (
        f"Positive Case 1 FAILED: Expected merchant_id_2 to be flagged as "
        f"anomaly (1), but got {prediction}."
    )
    print(
        "\n✅ Positive Case 1 PASSED: merchant_id_2 (401 webhook spike) "
        "correctly flagged as anomaly."
    )

    # ── Trigger agent handoff for the detected anomaly ────────────────────
    print("\n🔗 Triggering LangChain Agent for merchant_id_2 anomaly...\n")
    trigger_agent_for_anomaly(merchant_2_row)

    print("\n" + "=" * 70)
    print("  All edge case tests passed.")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stderr,
    )
    test_edge_cases()
