"""
pages/2_🧠_ML_Comparison.py – ML Model Comparison Dashboard
=============================================================

Streamlit multipage view that trains, evaluates, and visually compares three
unsupervised anomaly-detection models on the merged transaction + webhook data.

Models
------
* Isolation Forest
* One-Class SVM
* Local Outlier Factor

Ground truth is derived from known anomaly indicators in the data:
``decline_code == '93_Risk_Block'`` **or** ``http_status == 401``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Model Comparison",
    page_icon="🧠",
    layout="wide",
)

# ── Data directory ────────────────────────────────────────────────────────────
_DATA_DIR = Path(
    os.environ.get(
        "ANALYTICS_DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "output"),
    )
)

# ── Plotly theme defaults ─────────────────────────────────────────────────────
_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    font=dict(size=13),
    margin=dict(l=40, r=20, t=50, b=40),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_prepare_data(
    data_dir: str = str(_DATA_DIR),
) -> pd.DataFrame:
    """Load transactions & webhook_logs, merge, engineer features, add ground truth.

    Returns an empty DataFrame when CSVs are missing so the page renders
    gracefully.
    """
    dp = Path(data_dir)

    def _safe_read(name: str, **kwargs: object) -> pd.DataFrame:
        path = dp / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, **kwargs)

    txns = _safe_read("transactions.csv", dtype={"card_bin": str})
    wh = _safe_read("webhook_logs.csv")

    if txns.empty and wh.empty:
        return pd.DataFrame()
    if txns.empty:
        return wh.copy()
    if wh.empty:
        merged = txns.copy()
    else:
        merged = txns.merge(
            wh, on="transaction_id", how="left", suffixes=("", "_wh")
        )

    # ── Feature engineering ───────────────────────────────────────────────
    if "amount" in merged.columns:
        merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce").fillna(0.0)
    else:
        merged["amount"] = 0.0

    if "http_status" in merged.columns:
        merged["http_status"] = (
            pd.to_numeric(merged["http_status"], errors="coerce").fillna(0).astype(int)
        )
    else:
        merged["http_status"] = 0

    if "delivery_attempts" in merged.columns:
        merged["delivery_attempts"] = (
            pd.to_numeric(merged["delivery_attempts"], errors="coerce").fillna(0).astype(int)
        )
    else:
        merged["delivery_attempts"] = 0

    # ── Ground truth ──────────────────────────────────────────────────────
    risk_block = (
        merged["decline_code"] == "93_Risk_Block"
        if "decline_code" in merged.columns
        else pd.Series(False, index=merged.index)
    )
    auth_fail = merged["http_status"] == 401

    merged["is_anomaly_actual"] = (risk_block | auth_fail).astype(int)

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

_FEATURE_COLS = ["amount", "http_status", "delivery_attempts"]
_MODEL_NAMES = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]


@st.cache_resource
def train_and_evaluate(
    _data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Train three anomaly-detection models and return evaluation artefacts.

    Parameters
    ----------
    _data : pd.DataFrame
        Merged DataFrame with feature columns and ``is_anomaly_actual``.

    Returns
    -------
    metrics_df : pd.DataFrame
        Long-form table with columns ``Model``, ``Metric``, ``Score``.
    predictions : dict[str, np.ndarray]
        Mapping of model name → binary prediction array (1 = anomaly, 0 = normal).
    confusion_counts : dict[str, dict[str, int]]
        Mapping of model name → {"TP": …, "FP": …, "FN": …}.
    """
    features = _data[_FEATURE_COLS].copy()
    y_true = _data["is_anomaly_actual"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    models = {
        "IsolationForest": IsolationForest(
            n_estimators=100, contamination=0.02, random_state=42
        ),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="auto", nu=0.02),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20, contamination=0.02, novelty=False
        ),
    }

    predictions: dict[str, np.ndarray] = {}
    confusion_counts: dict[str, dict[str, int]] = {}
    rows: list[dict[str, object]] = []

    for name, model in models.items():
        if name == "LocalOutlierFactor":
            raw = model.fit_predict(X_scaled)
        else:
            raw = model.fit(X_scaled).predict(X_scaled)

        # Map: -1 (outlier) → 1 (anomaly), 1 (inlier) → 0 (normal)
        y_pred = (raw == -1).astype(int)
        predictions[name] = y_pred

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        confusion_counts[name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

        rows.append({"Model": name, "Metric": "Precision", "Score": round(prec, 4)})
        rows.append({"Model": name, "Metric": "Recall", "Score": round(rec, 4)})
        rows.append({"Model": name, "Metric": "F1-Score", "Score": round(f1, 4)})

    metrics_df = pd.DataFrame(rows)
    return metrics_df, predictions, confusion_counts


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UI – TITLE & EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🧠 ML Model Comparison – Anomaly Detection")

st.markdown(
    """
**Unsupervised Anomaly Detection** identifies unusual patterns in data
*without* pre-labelled examples.  Each model below learns what "normal"
transaction behaviour looks like and flags deviations as potential
anomalies.

| Model | How it works |
|---|---|
| **Isolation Forest** | Recursively partitions data; anomalies are isolated quickly. |
| **One-Class SVM** | Finds a tight boundary around normal data in kernel space. |
| **Local Outlier Factor** | Compares local density of a point to its neighbours. |

**Ground truth**: a transaction is considered a *true anomaly* when
`decline_code == '93_Risk_Block'` **or** `http_status == 401`.
"""
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOAD → TRAIN → DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

data = load_and_prepare_data()

if data.empty or "is_anomaly_actual" not in data.columns:
    st.warning(
        "⚠️ No data found.  Run `python generate_realistic_data.py` first to "
        "create the CSV files under `data/output/`."
    )
    st.stop()

# Verify we have enough positive samples
if data["is_anomaly_actual"].sum() == 0:
    st.warning("⚠️ No anomalies detected in the ground truth. All labels are 0.")
    st.stop()

metrics_df, predictions, confusion_counts = train_and_evaluate(data)

# ── 4a. Metrics table ────────────────────────────────────────────────────────
st.subheader("📋 Evaluation Metrics")

pivot = metrics_df.pivot(index="Model", columns="Metric", values="Score")
pivot = pivot[["Precision", "Recall", "F1-Score"]]

st.dataframe(
    pivot.style.highlight_max(axis=0, color="#2ecc71"),
    use_container_width=True,
)

# ── 4b. Grouped bar chart ────────────────────────────────────────────────────
st.subheader("📊 Visual Comparison")

fig_bar = px.bar(
    metrics_df,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text_auto=".3f",
    color_discrete_sequence=["#3498db", "#e74c3c", "#2ecc71"],
    title="Precision / Recall / F1-Score by Model",
)
fig_bar.update_layout(**_PLOTLY_LAYOUT, yaxis_range=[0, 1])
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTERACTIVE EXPLORATION (DEEP DIVE)
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🔍 Deep Dive – Per-Model Exploration")

selected_model = st.selectbox("Select a model to investigate:", _MODEL_NAMES)

# ── Metric cards ──────────────────────────────────────────────────────────────
counts = confusion_counts[selected_model]
col_tp, col_fp, col_fn, col_tn = st.columns(4)
col_tp.metric("✅ True Positives (TP)", f"{counts['TP']:,}")
col_fp.metric("⚠️ False Positives (FP)", f"{counts['FP']:,}")
col_fn.metric("❌ False Negatives (FN)", f"{counts['FN']:,}")
col_tn.metric("🟢 True Negatives (TN)", f"{counts['TN']:,}")

# ── Two-column deep-dive charts ──────────────────────────────────────────────
chart_left, chart_right = st.columns(2)

# ── LEFT: Confusion Matrix Heatmap ───────────────────────────────────────────
with chart_left:
    cm_values = np.array(
        [[counts["TN"], counts["FP"]],
         [counts["FN"], counts["TP"]]]
    )
    cm_text = np.array(
        [[f"TN\n{counts['TN']:,}", f"FP\n{counts['FP']:,}"],
         [f"FN\n{counts['FN']:,}", f"TP\n{counts['TP']:,}"]]
    )

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_values,
        x=["Predicted Normal", "Predicted Anomaly"],
        y=["Actual Normal", "Actual Anomaly"],
        text=cm_text,
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        colorscale=[
            [0.0, "#1a1a2e"],
            [0.5, "#16213e"],
            [1.0, "#e74c3c"],
        ],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
    ))
    fig_cm.update_layout(
        title=dict(text=f"{selected_model} – Confusion Matrix", font=dict(size=16)),
        xaxis=dict(title="Predicted Label", side="bottom"),
        yaxis=dict(title="Actual Label", autorange="reversed"),
        **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ── RIGHT: Amount Distribution by Prediction Outcome ─────────────────────────
with chart_right:
    pred_series = pd.Series(predictions[selected_model], index=data.index)
    outcome_df = data[["amount"]].copy()
    y_true = data["is_anomaly_actual"]

    conditions = [
        (pred_series == 1) & (y_true == 1),
        (pred_series == 1) & (y_true == 0),
        (pred_series == 0) & (y_true == 1),
        (pred_series == 0) & (y_true == 0),
    ]
    labels = ["TP (True Positive)", "FP (False Positive)",
              "FN (False Negative)", "TN (True Negative)"]
    outcome_df["Outcome"] = np.select(
        conditions, labels, default="TN (True Negative)"
    )

    # Only show categories that have data
    present = outcome_df["Outcome"].unique().tolist()
    color_map = {
        "TP (True Positive)": "#2ecc71",
        "FP (False Positive)": "#e67e22",
        "FN (False Negative)": "#e74c3c",
        "TN (True Negative)": "#3498db",
    }
    category_order = [lbl for lbl in labels if lbl in present]

    fig_box = px.box(
        outcome_df,
        x="Outcome",
        y="amount",
        color="Outcome",
        color_discrete_map=color_map,
        category_orders={"Outcome": category_order},
        title=f"{selected_model} – Amount Distribution by Outcome",
        labels={"amount": "Transaction Amount (₹)", "Outcome": ""},
    )
    fig_box.update_layout(**_PLOTLY_LAYOUT, showlegend=False)
    fig_box.update_yaxes(type="log", title="Transaction Amount ₹ (log scale)")
    st.plotly_chart(fig_box, use_container_width=True)
