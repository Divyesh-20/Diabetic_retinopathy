"""
pages/admin/compare.py – Side-by-side model comparison
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

from utils.auth import require_role
from utils.metrics import list_evaluated_models, load_results
from config import MODELS, DR_STAGE_COLORS


def show_compare_page():
    require_role("admin")

    st.markdown("## ⚖️ Model Comparison")
    st.markdown("Select models to compare side-by-side across all evaluation metrics.")
    st.divider()

    evaluated = list_evaluated_models()
    if len(evaluated) < 2:
        st.info("You need at least **2 evaluated models** to compare. Go to **Train & Evaluate** first.")
        return

    selected = st.multiselect(
        "Select models to compare",
        options=evaluated,
        format_func=lambda k: MODELS.get(k, k),
        default=evaluated[:min(len(evaluated), 3)],
    )

    if len(selected) < 2:
        st.warning("Select at least 2 models.")
        return

    # ── Load all results ──────────────────────────────────────────────────────
    results = {}
    for mkey in selected:
        r = load_results(mkey)
        if r.get("metrics"):
            results[mkey] = r["metrics"]

    if len(results) < 2:
        st.error("Not enough evaluation data for selected models.")
        return

    metric_keys = [
        ("accuracy",          "Accuracy",          True,  100),
        ("f1_weighted",       "F1 Weighted",        True,  1),
        ("f1_macro",          "F1 Macro",           True,  1),
        ("precision_weighted","Precision (W)",      True,  1),
        ("recall_weighted",   "Recall (W)",         True,  1),
        ("quadratic_kappa",   "Quadratic Kappa",    True,  1),
        ("macro_auc",         "Macro AUC",          True,  1),
    ]

    # ── Comparison Table ──────────────────────────────────────────────────────
    st.markdown("#### 📋 Metrics Comparison Table")
    rows = []
    for mkey, m in results.items():
        row = {"Model": MODELS.get(mkey, mkey)}
        for mk, label, __, scale in metric_keys:
            val = m.get(mk) or 0
            row[label] = f"{val * scale:.2f}{'%' if scale == 100 else ''}"
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Export CSV ────────────────────────────────────────────────────────────
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export as CSV", data=csv, file_name="model_comparison.csv",
                        mime="text/csv")

    st.divider()

    # ── Grouped Bar Chart ─────────────────────────────────────────────────────
    st.markdown("#### 📊 Side-by-Side Performance Chart")
    palette = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC",
               "#26C6DA", "#D4E157", "#FF7043", "#8D6E63"]

    metric_options = [label for _, label, _, _ in metric_keys]
    chosen_metrics = st.multiselect(
        "Choose metrics to display",
        options=metric_options,
        default=["Accuracy", "F1 Weighted", "Quadratic Kappa", "Macro AUC"],
    )

    if chosen_metrics:
        metric_map = {label: (mk, scale) for mk, label, _, scale in metric_keys}
        fig = go.Figure()
        for i, (mkey, m) in enumerate(results.items()):
            ys = []
            for label in chosen_metrics:
                mk, scale = metric_map[label]
                val = m.get(mk) or 0
                ys.append(val * scale)
            fig.add_trace(go.Bar(
                name=MODELS.get(mkey, mkey),
                x=chosen_metrics,
                y=ys,
                marker_color=palette[i % len(palette)],
                text=[f"{v:.2f}" for v in ys],
                textposition="auto",
            ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Best Model Highlight ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🏆 Best Model")
    best = max(results.items(), key=lambda kv: kv[1].get("quadratic_kappa", 0))
    best_key, best_metrics = best
    st.success(
        f"**{MODELS.get(best_key, best_key)}** achieves the highest "
        f"Quadratic Kappa of **{best_metrics.get('quadratic_kappa', 0):.3f}** "
        f"and Accuracy of **{best_metrics.get('accuracy', 0)*100:.2f}%**"
    )

    # ── Per-class AUC Comparison ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📉 Per-Class AUC Comparison")
    auc_fig = go.Figure()
    for i, (mkey, m) in enumerate(results.items()):
        auc_pc = m.get("auc_per_class", {})
        ys = [auc_pc.get(str(c), auc_pc.get(c, 0)) or 0 for c in range(5)]
        auc_fig.add_trace(go.Bar(
            name=MODELS.get(mkey, mkey),
            x=[f"Stage {c}" for c in range(5)],
            y=ys,
            marker_color=palette[i % len(palette)],
            text=[f"{v:.2f}" for v in ys],
            textposition="auto",
        ))
    auc_fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, range=[0, 1]),
        margin=dict(l=0, r=0, t=10, b=0),
        height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(auc_fig, use_container_width=True)
