"""
models/ml_watcher.py – Proactive ML Watcher: unsupervised anomaly detection
                        for merchant behaviour monitoring.

This module detects abnormal merchant behaviour — such as sudden decline
spikes, card-testing bot attacks, or risk-block surges — by applying an
Isolation Forest over time-series features aggregated in rolling windows.

Typical usage::

    from models.ml_watcher import engineer_features, MerchantHealthMonitor

    df        = pd.read_csv("data/output/transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    features  = engineer_features(df, window="10min")
    monitor   = MerchantHealthMonitor()
    features, predictions = monitor.train_and_predict(features)
    alerts    = monitor.generate_alerts(features, predictions)

Or run as a standalone script::

    python models/ml_watcher.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Columns that are required in the raw transactions DataFrame
_REQUIRED_COLUMNS: set[str] = {
    "transaction_id",
    "merchant_id",
    "timestamp",
    "amount",
    "status",
}

# Numeric feature columns fed to the Isolation Forest
# (must match the columns produced by engineer_features)
_FEATURE_COLUMNS: list[str] = [
    "total_transactions",
    "decline_count",
    "decline_ratio",
    "avg_amount",
    "risk_block_count",
]

# Alert-type thresholds
_RISK_BLOCK_SPIKE_THRESHOLD: int = 10
_ELEVATED_DECLINE_RATIO_THRESHOLD: float = 0.6

# Default path for the generated transactions CSV
_DEFAULT_TXN_CSV = Path(__file__).parent.parent / "data" / "output" / "transactions.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 – Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_transactions(csv_path: str | Path = _DEFAULT_TXN_CSV) -> pd.DataFrame:
    """
    Load ``transactions.csv`` into a Pandas DataFrame and parse the timestamp.

    The function validates that all required columns are present and converts
    the ``timestamp`` column from a string to timezone-aware
    ``datetime64[ns, UTC]`` objects.

    Parameters
    ----------
    csv_path:
        Path to ``transactions.csv``.  Defaults to ``data/output/transactions.csv``.

    Returns
    -------
    pd.DataFrame
        Raw transactions with ``timestamp`` as datetime.

    Raises
    ------
    FileNotFoundError:
        If the CSV file does not exist at *csv_path*.
    ValueError:
        If any required column is missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Transactions CSV not found at '{csv_path}'. "
            "Run `python data/telemetry_generator.py` to generate it."
        )

    df = pd.read_csv(csv_path, dtype={"card_bin": str})

    # Validate required columns
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"transactions.csv is missing required columns: {missing}"
        )

    # Parse timestamp → timezone-aware datetime (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info("Loaded %d transactions from '%s'", len(df), csv_path)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 – Feature engineering
# ──────────────────────────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame, window: str = "10min") -> pd.DataFrame:
    """
    Aggregate raw transactions into merchant-level, rolling-window features
    suitable for unsupervised anomaly detection.

    **Why rolling windows?**  Point-in-time transaction volumes are highly
    variable and hard to reason about in isolation.  Aggregating by a fixed
    time window (e.g. 10 minutes) converts the raw event stream into a
    compact, comparable summary for each merchant at each moment in time.
    This is the same approach used by real-time fraud-monitoring systems at
    payment networks such as Visa and Mastercard.

    Algorithm
    ---------
    1. Sort by ``merchant_id`` then ``timestamp`` so each merchant's events
       are contiguous and time-ordered.
    2. Use :func:`pandas.Grouper` to bucket timestamps into non-overlapping
       windows of size *window* (e.g. ``"10min"``).
    3. For each (merchant, window) bucket compute:

       ``total_transactions``
           Raw count of payment events.  A sudden spike relative to baseline
           is a signal of bot activity or a volume anomaly.

       ``decline_count``
           Count of transactions where ``status == 'DECLINED'``.

       ``decline_ratio``
           ``decline_count / total_transactions``.  Ratios above ~0.6 suggest
           either a card-testing attack or an issuer outage.
           Division-by-zero (empty bucket) is handled by filling with 0.

       ``avg_amount``
           Mean transaction amount.  Very small averages (e.g. ₹1–₹5) are
           a classic card-testing signature.

       ``risk_block_count``
           Count of transactions with ``decline_code == '93_Risk_Block'``.
           A large spike in this code indicates the card network has
           temporarily blocked transactions for this merchant.

    4. Drop empty buckets (windows with zero transactions) to avoid
       confusing the Isolation Forest with structural zeros.

    Parameters
    ----------
    df:
        Raw transactions DataFrame.  Must contain at least
        ``merchant_id``, ``timestamp``, ``amount``, ``status``,
        and ``decline_code`` (optional — treated as empty if absent).
    window:
        Pandas offset alias for the aggregation window (default: ``"10min"``).
        Examples: ``"5min"``, ``"1h"``, ``"30min"``.

    Returns
    -------
    pd.DataFrame
        One row per (merchant_id, time-window) combination, with columns:
        ``merchant_id``, ``timestamp``, ``total_transactions``,
        ``decline_count``, ``decline_ratio``, ``avg_amount``,
        ``risk_block_count``.

    Notes
    -----
    * ``decline_code`` is optional in the raw data.  If the column is absent
      it is synthesised as an empty string so ``risk_block_count`` returns 0.
    * Empty time windows (no transactions) are dropped from the output.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["merchant_id", "timestamp"] + _FEATURE_COLUMNS
        )

    # Ensure timestamp is datetime (caller may have already done this)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # decline_code may be absent in stripped-down test DataFrames
    if "decline_code" not in df.columns:
        df = df.copy()
        df["decline_code"] = ""

    # Boolean helper columns (computed once, cheaply)
    is_declined = df["status"] == "DECLINED"
    is_risk_block = df["decline_code"] == "93_Risk_Block"

    # ── Aggregate per (merchant_id × time-window) ─────────────────────────
    grouped = df.groupby(
        ["merchant_id", pd.Grouper(key="timestamp", freq=window)]
    )

    agg = grouped.agg(
        total_transactions=("transaction_id", "count"),
        decline_count=(
            "status",
            lambda s: int((s == "DECLINED").sum()),
        ),
        avg_amount=("amount", "mean"),
        risk_block_count=(
            "decline_code",
            lambda s: int((s == "93_Risk_Block").sum()),
        ),
    ).reset_index()

    # Drop windows where no transaction occurred (grouper creates empty bins)
    agg = agg[agg["total_transactions"] > 0].copy()

    # decline_ratio – safe division: defaults to 0.0 for empty windows
    agg["decline_ratio"] = np.where(
        agg["total_transactions"] > 0,
        agg["decline_count"] / agg["total_transactions"],
        0.0,
    )

    # Reorder columns for readability
    agg = agg[["merchant_id", "timestamp"] + _FEATURE_COLUMNS]
    agg = agg.reset_index(drop=True)

    logger.info(
        "Engineered %d feature rows from %d transactions (window=%s)",
        len(agg),
        len(df),
        window,
    )
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – Anomaly detection model
# ──────────────────────────────────────────────────────────────────────────────


class MerchantHealthMonitor:
    """
    Unsupervised anomaly detector for merchant transaction behaviour.

    Uses :class:`sklearn.ensemble.IsolationForest` to identify windows in
    which a merchant's aggregate metrics deviate from their baseline behaviour.
    Because no labelled ground-truth is required, the model can be deployed
    immediately on any new dataset.

    Parameters
    ----------
    contamination:
        Expected fraction of anomalous windows in the data (default: 0.05).
        This translates directly to the IsolationForest ``contamination``
        hyper-parameter.
    random_state:
        Random seed for reproducibility (default: 42).

    Attributes
    ----------
    model : IsolationForest
        The fitted scikit-learn estimator (available after
        :meth:`train_and_predict` has been called).
    is_trained : bool
        Whether :meth:`train_and_predict` has been called at least once.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        self.is_trained: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Training & inference
    # ──────────────────────────────────────────────────────────────────────────

    def train_and_predict(
        self, features_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Fit the Isolation Forest on *features_df* and return anomaly labels.

        The model is fit on the numeric feature columns only
        (``total_transactions``, ``decline_count``, ``decline_ratio``,
        ``avg_amount``, ``risk_block_count``).  Identifier columns
        (``merchant_id``, ``timestamp``) are excluded from training but are
        preserved in the returned DataFrame so callers can trace predictions
        back to a merchant and time window.

        IsolationForest convention:
          *  ``1``  → normal (inlier)
          * ``-1``  → anomaly (outlier)

        Parameters
        ----------
        features_df:
            DataFrame produced by :func:`engineer_features`.  Must contain
            all columns in :data:`_FEATURE_COLUMNS`.

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            A (features_df, predictions) pair where *predictions* is a
            1-D array of ``int`` with values ``1`` or ``-1``, aligned
            row-for-row with *features_df*.

        Raises
        ------
        ValueError:
            If *features_df* is empty or any required feature column is
            missing.
        """
        if features_df.empty:
            raise ValueError(
                "features_df is empty — nothing to train on. "
                "Ensure transactions.csv contains data and "
                "engineer_features() produced at least one row."
            )

        missing = set(_FEATURE_COLUMNS) - set(features_df.columns)
        if missing:
            raise ValueError(
                f"features_df is missing required feature columns: {missing}"
            )

        X = features_df[_FEATURE_COLUMNS].values  # shape (n_windows, 5)
        predictions: np.ndarray = self.model.fit_predict(X)
        self.is_trained = True

        n_anomalies = int((predictions == -1).sum())
        logger.info(
            "IsolationForest: %d windows scored, %d anomalies (%.1f%%)",
            len(predictions),
            n_anomalies,
            100 * n_anomalies / len(predictions),
        )
        return features_df, predictions

    # ──────────────────────────────────────────────────────────────────────────
    # Alert generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_alerts(
        self,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Convert anomalous windows into structured JSON-serialisable alerts.

        Each row where ``predictions == -1`` becomes one alert.  The
        ``alert_type`` is determined by the most-severe signal present in
        that window:

        1. **"High Risk Block Spike"** — ``risk_block_count > 10``
           Indicates the card network is actively blocking transactions for
           this merchant, likely due to a detected fraud pattern.

        2. **"Elevated Decline Ratio"** — ``decline_ratio > 0.6``
           More than 60 % of transactions in the window were declined.
           Possible causes: issuer outage, card-testing attack, or misconfigured
           merchant category code.

        3. **"General Transaction Anomaly"** — all other anomalous windows.
           The Isolation Forest flagged this window as unusual, but none of
           the specific rule-based signatures were triggered.

        Parameters
        ----------
        features_df:
            The same DataFrame returned by :meth:`train_and_predict`.
        predictions:
            1-D array of ``int`` (``1`` or ``-1``) aligned with *features_df*.

        Returns
        -------
        list[dict]
            A (possibly empty) list of alert dictionaries.  Each dict has:

            .. code-block:: python

                {
                    "merchant_id": "merchant_id_1",
                    "timestamp":   "2026-03-07T10:10:00+00:00",
                    "alert_type":  "High Risk Block Spike",
                    "metrics": {
                        "decline_ratio":      0.96,
                        "total_transactions": 52,
                    },
                }
        """
        if len(features_df) != len(predictions):
            raise ValueError(
                f"features_df has {len(features_df)} rows but predictions "
                f"has {len(predictions)} elements — they must be aligned."
            )

        # Attach predictions as a temporary column for easy filtering
        annotated = features_df.copy()
        annotated["_prediction"] = predictions
        anomalies = annotated[annotated["_prediction"] == -1].copy()

        alerts: list[dict[str, Any]] = []
        for _, row in anomalies.iterrows():
            # Determine alert type by priority
            if row["risk_block_count"] > _RISK_BLOCK_SPIKE_THRESHOLD:
                alert_type = "High Risk Block Spike"
            elif row["decline_ratio"] > _ELEVATED_DECLINE_RATIO_THRESHOLD:
                alert_type = "Elevated Decline Ratio"
            else:
                alert_type = "General Transaction Anomaly"

            # Serialise timestamp to ISO-8601 string
            ts = row["timestamp"]
            timestamp_str = (
                ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            )

            alerts.append(
                {
                    "merchant_id": str(row["merchant_id"]),
                    "timestamp":   timestamp_str,
                    "alert_type":  alert_type,
                    "metrics": {
                        "decline_ratio":      round(float(row["decline_ratio"]), 4),
                        "total_transactions": int(row["total_transactions"]),
                    },
                }
            )

        logger.info("Generated %d alerts from %d anomalous windows", len(alerts), len(anomalies))
        return alerts


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – Main execution block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stderr,
    )

    print("=" * 70)
    print("  Proactive ML Watcher — Merchant Anomaly Detection")
    print("=" * 70)

    # 1. Load transactions
    try:
        raw_df = load_transactions()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)

    # 2. Engineer time-series features (10-minute windows)
    features_df = engineer_features(raw_df, window="10min")
    print(f"\nFeature matrix: {len(features_df):,} windows × {len(_FEATURE_COLUMNS)} features")

    # 3. Fit Isolation Forest and predict anomalies
    monitor = MerchantHealthMonitor(contamination=0.05, random_state=42)
    features_df, predictions = monitor.train_and_predict(features_df)

    n_anomalies = int((predictions == -1).sum())
    print(f"Anomalies detected: {n_anomalies} / {len(predictions):,} windows "
          f"({100 * n_anomalies / len(predictions):.1f}%)")

    # 4. Generate structured alerts
    alerts = monitor.generate_alerts(features_df, predictions)

    # 5. Print nicely formatted JSON
    print(f"\nAlerts ({len(alerts)} total):")
    print("-" * 70)
    print(json.dumps(alerts, indent=2, ensure_ascii=False))
