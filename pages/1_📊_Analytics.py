"""
pages/1_📊_Analytics.py – Merchant Analytics Dashboard
========================================================

Streamlit multipage view that provides platform-wide and per-merchant
analytics for the Intelligent Merchant Support project.

Sections
--------
1. KPI metrics (total transactions, success rates, critical failures)
2. Decline-code distribution & transaction volume over time
3. Expandable table of recent failed transactions / webhooks
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Merchant Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Data directory ────────────────────────────────────────────────────────────
_DATA_DIR = Path(
    os.environ.get(
        "ANALYTICS_DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "output"),
    )
)


# ── Data loading & caching ────────────────────────────────────────────────────
@st.cache_data
def load_data(
    data_dir: str = str(_DATA_DIR),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and return (transactions, webhook_logs, merchants) DataFrames.

    Returns empty DataFrames when CSV files are missing so the dashboard
    can still render gracefully.
    """
    dp = Path(data_dir)

    def _safe_read(name: str, **kwargs: object) -> pd.DataFrame:
        path = dp / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, **kwargs)

    transactions = _safe_read("transactions.csv", dtype={"card_bin": str})
    webhook_logs = _safe_read("webhook_logs.csv")
    merchants = _safe_read("merchants.csv", dtype={"mcc_code": str})
    return transactions, webhook_logs, merchants


@st.cache_data
def build_merged_view(
    transactions: pd.DataFrame,
    webhook_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Merge transactions and webhook_logs on *transaction_id*.

    If either DataFrame is empty the other is returned as-is (with any
    missing columns filled by ``NaN``).
    """
    if transactions.empty and webhook_logs.empty:
        return pd.DataFrame()
    if transactions.empty:
        return webhook_logs.copy()
    if webhook_logs.empty:
        return transactions.copy()

    merged = transactions.merge(
        webhook_logs,
        on="transaction_id",
        how="left",
        suffixes=("", "_wh"),
    )
    return merged


# ── Load data ─────────────────────────────────────────────────────────────────
transactions_df, webhook_logs_df, merchants_df = load_data()
merged_df = build_merged_view(transactions_df, webhook_logs_df)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 Merchant Analytics Dashboard")

# ── Sidebar – merchant selector ───────────────────────────────────────────────
if not merchants_df.empty and "merchant_id" in merchants_df.columns:
    merchant_options = ["All Merchants"] + sorted(
        merchants_df["merchant_id"].unique().tolist()
    )
    # Use business_name as display label when available
    if "business_name" in merchants_df.columns:
        id_to_name = dict(
            zip(merchants_df["merchant_id"], merchants_df["business_name"])
        )
    else:
        id_to_name = {}
elif not merged_df.empty and "merchant_id" in merged_df.columns:
    merchant_options = ["All Merchants"] + sorted(
        merged_df["merchant_id"].unique().tolist()
    )
    id_to_name = {}
else:
    merchant_options = ["All Merchants"]
    id_to_name = {}

selected_merchant = st.sidebar.selectbox(
    "Select Merchant",
    options=merchant_options,
    format_func=lambda mid: (
        mid if mid == "All Merchants" else f"{mid} ({id_to_name[mid]})" if mid in id_to_name else mid
    ),
)

# ── Filter merged data ───────────────────────────────────────────────────────
if selected_merchant == "All Merchants" or merged_df.empty:
    view_df = merged_df.copy()
else:
    view_df = merged_df[merged_df["merchant_id"] == selected_merchant].copy()

# ── KPI helpers ───────────────────────────────────────────────────────────────


def _safe_pct(numerator: int, denominator: int) -> float:
    """Return percentage rounded to one decimal, guarding against zero division."""
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 1)


# ── KPI Metrics (top row) ────────────────────────────────────────────────────
total_txns = len(view_df)

if not view_df.empty and "status" in view_df.columns:
    success_count = int((view_df["status"] == "SUCCESS").sum())
else:
    success_count = 0
txn_success_rate = _safe_pct(success_count, total_txns)

if not view_df.empty and "http_status" in view_df.columns:
    webhook_ok = int(
        view_df["http_status"]
        .dropna()
        .apply(lambda s: 200 <= int(s) < 300)
        .sum()
    )
    webhook_total = int(view_df["http_status"].dropna().shape[0])
else:
    webhook_ok = 0
    webhook_total = 0
webhook_success_rate = _safe_pct(webhook_ok, webhook_total)

# Critical failures: 93_Risk_Block declines OR 500-status webhooks
critical_failures = 0
if not view_df.empty:
    if "decline_code" in view_df.columns:
        critical_failures += int(
            (view_df["decline_code"] == "93_Risk_Block").sum()
        )
    if "http_status" in view_df.columns:
        critical_failures += int(
            (view_df["http_status"].dropna().astype(int) == 500).sum()
        )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Transaction Success Rate", f"{txn_success_rate}%")
col3.metric("Webhook Success Rate", f"{webhook_success_rate}%")
col4.metric("Critical Failures", f"{critical_failures:,}")

# ── Visualisations (middle row) ───────────────────────────────────────────────
chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Decline Code Distribution")
    if (
        not view_df.empty
        and "decline_code" in view_df.columns
        and "status" in view_df.columns
    ):
        declined = view_df[view_df["status"] != "SUCCESS"].copy()
        if not declined.empty and declined["decline_code"].notna().any():
            code_counts = (
                declined["decline_code"]
                .dropna()
                .value_counts()
                .rename_axis("decline_code")
                .reset_index(name="count")
                .set_index("decline_code")
            )
            st.bar_chart(code_counts)
        else:
            st.info("No declined transactions to display.")
    else:
        st.info("No decline-code data available.")

with chart_right:
    st.subheader("Transaction Volume Over Time")
    # Use the transaction timestamp column (not the webhook one)
    ts_col = "timestamp" if "timestamp" in view_df.columns else None
    if ts_col and not view_df.empty:
        vol = view_df.copy()
        vol[ts_col] = pd.to_datetime(vol[ts_col], errors="coerce")
        vol = vol.dropna(subset=[ts_col])
        if not vol.empty:
            vol = vol.set_index(ts_col).resample("h").size().rename("transactions")
            st.line_chart(vol)
        else:
            st.info("No timestamp data available.")
    else:
        st.info("No transaction volume data available.")

# ── Raw data view (bottom row) ────────────────────────────────────────────────
with st.expander("View Recent Failed Transactions"):
    if view_df.empty:
        st.info("No data available.")
    else:
        failed_mask = pd.Series(False, index=view_df.index)
        if "status" in view_df.columns:
            failed_mask = failed_mask | (view_df["status"] != "SUCCESS")
        if "http_status" in view_df.columns:
            failed_mask = failed_mask | (
                view_df["http_status"]
                .dropna()
                .apply(lambda s: int(s) < 200 or int(s) >= 300)
                .reindex(view_df.index, fill_value=False)
            )
        failed = view_df[failed_mask].copy()

        if failed.empty:
            st.info("No failed transactions found for this selection.")
        else:
            # Sort most-recent first using transaction timestamp
            sort_col = "timestamp" if "timestamp" in failed.columns else None
            if sort_col:
                failed[sort_col] = pd.to_datetime(
                    failed[sort_col], errors="coerce"
                )
                failed = failed.sort_values(sort_col, ascending=False)
            st.dataframe(failed, use_container_width=True)
