"""
models/fraud_model.py – FraudDetector: Anomaly detection / XGBoost wrapper
                         for identifying fraudulent or anomalous transactions.

The class exposes a clean train() / predict() interface so it can be
swapped between an Isolation Forest (unsupervised) and an XGBoost classifier
(supervised) depending on the availability of labelled training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# FraudDetector
# ──────────────────────────────────────────────────────────────────────────────


class FraudDetector:
    """
    Wraps a scikit-learn compatible estimator for fraud / anomaly detection.

    Attributes:
        model:      The underlying ML estimator (IsolationForest or XGBoost).
        is_trained: Flag indicating whether the model has been fitted.

    Usage (unsupervised, no labels)::

        detector = FraudDetector(mode="anomaly")
        detector.train(df_transactions)
        predictions = detector.predict(df_new_transactions)

    Usage (supervised, with labels)::

        detector = FraudDetector(mode="supervised")
        detector.train(df_transactions, labels=df_transactions["is_fraud"])
        predictions = detector.predict(df_new_transactions)
    """

    def __init__(self, mode: str = "anomaly") -> None:
        """
        Initialise the FraudDetector.

        Args:
            mode: ``"anomaly"`` uses IsolationForest (unsupervised).
                  ``"supervised"`` uses XGBoostClassifier (requires labels).
        """
        self.mode = mode
        self.model = None       # populated by train()
        self.is_trained = False

        # Initialise the underlying estimator based on the chosen mode
        if mode == "anomaly":
            from sklearn.ensemble import IsolationForest

            # contamination: expected fraction of outliers in the dataset
            self.model = IsolationForest(contamination=0.05, random_state=42)

        elif mode == "supervised":
            from xgboost import XGBClassifier

            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'anomaly' or 'supervised'.")

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        transaction_data: pd.DataFrame,
        labels: pd.Series | None = None,
    ) -> None:
        """
        Fit the model on historical transaction data.

        Args:
            transaction_data: DataFrame of feature columns.  Categorical
                              columns should be encoded before calling train().
            labels:           Series of binary fraud labels (0 = legitimate,
                              1 = fraud).  Required for ``supervised`` mode,
                              ignored for ``anomaly`` mode.

        Raises:
            ValueError: If ``supervised`` mode is selected but no labels are
                        provided.
        """
        # TODO: add feature engineering (e.g. velocity, merchant category codes)
        #       and a preprocessing pipeline (scaling, encoding) before fitting.

        features = self._extract_features(transaction_data)

        if self.mode == "supervised":
            if labels is None:
                raise ValueError("Labels are required for supervised mode.")
            self.model.fit(features, labels)

        else:  # anomaly mode – unsupervised, no labels needed
            self.model.fit(features)

        self.is_trained = True

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, transaction_data: pd.DataFrame) -> np.ndarray:
        """
        Score new transactions for fraud risk.

        Args:
            transaction_data: DataFrame of the same feature columns used
                              during training.

        Returns:
            For ``anomaly`` mode: array of ``1`` (normal) or ``-1`` (anomaly).
            For ``supervised`` mode: array of ``0`` (legitimate) or ``1`` (fraud).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict().")

        features = self._extract_features(transaction_data)

        if self.mode == "supervised":
            return self.model.predict(features)

        # IsolationForest.predict returns 1 for inliers, -1 for outliers
        return self.model.predict(features)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_features(transaction_data: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare the numeric feature columns for model input.

        TODO: extend this method with richer feature engineering such as:
          - Rolling transaction velocity per merchant / card
          - Time-of-day and day-of-week cyclical encodings
          - Merchant category code one-hot encoding
          - Normalisation / standardisation

        Args:
            transaction_data: Raw transaction DataFrame.

        Returns:
            DataFrame containing only numeric feature columns.
        """
        # Keep only numeric columns as a sensible default
        numeric_cols = transaction_data.select_dtypes(include=[np.number]).columns.tolist()
        return transaction_data[numeric_cols]
