"""
data/data_access.py – DataLoader: in-memory data access layer for the
                       Intelligent Merchant Support AI Agent.

Loads the three synthetic telemetry CSVs (merchants, transactions,
webhook_logs) produced by ``data/telemetry_generator.py`` into Pandas
DataFrames on initialisation, then exposes clean query methods used by
the FastAPI endpoints and the AI agents.

Typical usage::

    loader = DataLoader()                              # loads from data/output/
    merchant = loader.get_merchant("merchant_id_1")
    txns     = loader.get_recent_transactions("merchant_id_1", limit=5)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default path for the generated CSV files
_DEFAULT_DATA_DIR = Path(__file__).parent / "output"


class DataLoader:
    """
    In-memory data access layer backed by three Pandas DataFrames.

    Parameters
    ----------
    data_dir:
        Directory that contains ``merchants.csv``, ``transactions.csv``,
        and ``webhook_logs.csv``.  Defaults to ``data/output/``.
    merchants_df:
        Optional pre-built merchants DataFrame (used in tests to bypass
        file I/O).
    transactions_df:
        Optional pre-built transactions DataFrame (used in tests).
    webhook_logs_df:
        Optional pre-built webhook logs DataFrame (used in tests).

    Attributes
    ----------
    merchants : pd.DataFrame
    transactions : pd.DataFrame
    webhook_logs : pd.DataFrame
    """

    def __init__(
        self,
        data_dir: str | Path = _DEFAULT_DATA_DIR,
        *,
        merchants_df: Optional[pd.DataFrame] = None,
        transactions_df: Optional[pd.DataFrame] = None,
        webhook_logs_df: Optional[pd.DataFrame] = None,
    ) -> None:
        data_path = Path(data_dir)

        # Allow callers to inject pre-built DataFrames (useful in unit tests)
        # so the class does not require CSV files to be present.
        if merchants_df is not None:
            self.merchants = merchants_df.copy()
        else:
            self.merchants = self._load_csv(
                data_path / "merchants.csv",
                dtype={"mcc_code": str},
            )

        if transactions_df is not None:
            self.transactions = transactions_df.copy()
        else:
            self.transactions = self._load_csv(
                data_path / "transactions.csv",
                # card_bin must be a string (6-digit BIN, not an integer)
                dtype={"card_bin": str},
            )

        if webhook_logs_df is not None:
            self.webhook_logs = webhook_logs_df.copy()
        else:
            self.webhook_logs = self._load_csv(data_path / "webhook_logs.csv")

        logger.info(
            "DataLoader ready — %d merchants, %d transactions, %d webhook logs",
            len(self.merchants),
            len(self.transactions),
            len(self.webhook_logs),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame, returning an empty DataFrame if the
        file does not exist so the rest of the application can still start.

        Parameters
        ----------
        path:
            Absolute or relative path to the CSV file.
        **kwargs:
            Extra arguments forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        pd.DataFrame
        """
        if not path.exists():
            logger.warning(
                "CSV file not found: %s — returning empty DataFrame. "
                "Run `python data/telemetry_generator.py` to generate the data.",
                path,
            )
            return pd.DataFrame()
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def _to_record(row: pd.Series) -> dict:
        """
        Convert a DataFrame row to a plain Python dict, replacing any
        ``float('nan')`` values (which pandas uses for missing data) with
        ``None`` so downstream Pydantic models can validate them correctly.

        Parameters
        ----------
        row:
            A single row from a DataFrame (as returned by ``.iloc[0]``).

        Returns
        -------
        dict
        """
        return {
            k: (None if (isinstance(v, float) and v != v) else v)
            for k, v in row.to_dict().items()
        }

    @classmethod
    def _to_records(cls, df: pd.DataFrame) -> list[dict]:
        """
        Convert every row in *df* to a plain Python dict via
        :meth:`_to_record`, replacing ``NaN`` with ``None``.
        """
        return [cls._to_record(row) for _, row in df.iterrows()]

    # ──────────────────────────────────────────────────────────────────────────
    # Query methods
    # ──────────────────────────────────────────────────────────────────────────

    def get_merchant(self, merchant_id: str) -> Optional[dict]:
        """
        Look up a single merchant by its ID.

        Parameters
        ----------
        merchant_id:
            The ``merchant_id`` value to find (e.g. ``"merchant_id_1"``).

        Returns
        -------
        dict | None
            A dictionary of the merchant's profile, or ``None`` if not found.
        """
        if self.merchants.empty:
            return None

        rows = self.merchants[self.merchants["merchant_id"] == merchant_id]
        if rows.empty:
            return None

        return self._to_record(rows.iloc[0])

    def get_recent_transactions(
        self, merchant_id: str, limit: int = 10
    ) -> list[dict]:
        """
        Return the *limit* most-recent transactions for a given merchant,
        sorted newest-first.

        Parameters
        ----------
        merchant_id:
            The merchant whose transactions to retrieve.
        limit:
            Maximum number of rows to return (default: 10).

        Returns
        -------
        list[dict]
            A list of transaction dicts (empty if the merchant has no
            transactions or is not found).
        """
        if self.transactions.empty:
            return []

        rows = self.transactions[self.transactions["merchant_id"] == merchant_id]
        if rows.empty:
            return []

        # Sort by timestamp descending so the most recent comes first
        if "timestamp" in rows.columns:
            rows = rows.sort_values("timestamp", ascending=False)

        return self._to_records(rows.head(limit))

    def get_transaction_details(self, transaction_id: str) -> Optional[dict]:
        """
        Retrieve full details for a single transaction.

        Parameters
        ----------
        transaction_id:
            The ``transaction_id`` to look up (e.g. ``"TXN-00000001"``).

        Returns
        -------
        dict | None
            A dictionary of the transaction's fields, or ``None`` if not found.
        """
        if self.transactions.empty:
            return None

        rows = self.transactions[
            self.transactions["transaction_id"] == transaction_id
        ]
        if rows.empty:
            return None

        return self._to_record(rows.iloc[0])

    def get_webhook_logs_for_merchant(
        self, merchant_id: str, limit: int = 10
    ) -> list[dict]:
        """
        Return the *limit* most-recent webhook delivery logs for a merchant.

        This requires an implicit join: ``webhook_logs`` does not contain
        ``merchant_id`` directly, so we first filter ``transactions`` by
        merchant, collect their ``transaction_id`` values, then filter
        ``webhook_logs`` on those IDs.

        Parameters
        ----------
        merchant_id:
            The merchant whose webhook logs to retrieve.
        limit:
            Maximum number of rows to return (default: 10).

        Returns
        -------
        list[dict]
            A list of webhook log dicts (empty if not found).
        """
        if self.transactions.empty or self.webhook_logs.empty:
            return []

        # Step 1: collect all transaction_ids for this merchant
        merchant_txn_ids = self.transactions.loc[
            self.transactions["merchant_id"] == merchant_id, "transaction_id"
        ]
        if merchant_txn_ids.empty:
            return []

        # Step 2: filter webhook_logs by those transaction_ids
        logs = self.webhook_logs[
            self.webhook_logs["transaction_id"].isin(merchant_txn_ids)
        ]
        if logs.empty:
            return []

        # Sort newest first if timestamp is available
        if "timestamp" in logs.columns:
            logs = logs.sort_values("timestamp", ascending=False)

        return self._to_records(logs.head(limit))

    def get_webhook_log_for_transaction(self, transaction_id: str) -> Optional[dict]:
        """
        Return the webhook log entry associated with a specific transaction.

        Parameters
        ----------
        transaction_id:
            The ``transaction_id`` to look up (e.g. ``"TXN-00000001"``).

        Returns
        -------
        dict | None
            A dictionary of the webhook log fields, or ``None`` if no log
            exists for this transaction.
        """
        if self.webhook_logs.empty:
            return None

        rows = self.webhook_logs[
            self.webhook_logs["transaction_id"] == transaction_id
        ]
        if rows.empty:
            return None

        return self._to_record(rows.iloc[0])

    def update_webhook_status(self, log_id: str, new_status: int) -> bool:
        """
        Simulate an agentic webhook retry by updating the ``http_status``
        column for a given log entry in-memory.

        Parameters
        ----------
        log_id:
            The ``log_id`` of the webhook log entry to update.
        new_status:
            The new HTTP status code to write (e.g. ``200``).

        Returns
        -------
        bool
            ``True`` if the row was found and updated, ``False`` otherwise.
        """
        if self.webhook_logs.empty:
            return False

        mask = self.webhook_logs["log_id"] == log_id
        if not mask.any():
            return False

        self.webhook_logs.loc[mask, "http_status"] = new_status
        logger.info(
            "Webhook log %s http_status updated to %d", log_id, new_status
        )
        return True
