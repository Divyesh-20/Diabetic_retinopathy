"""
pages/admin/dashboard.py – Admin overview dashboard
"""

import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json

from utils.auth import require_role
from utils.metrics import list_evaluated_models, load_results
from models.model_factory import list_available_models, list_all_models
from utils.dataset import get_dataset_stats
from config import MODELS, DR_STAGES, DR_STAGE_COLORS, DATA_DIR


def show_dashboard_page():
    require_role("admin")

    st.markdown("## 🏠 Admin Dashboard")
    st.markdown("Overview of the DR Detection System.")
    st.divider()

    # ── Summary Cards ─────────────────────────────────────────────────────────
    evaluated  = list_evaluated_models()
    available  = list_available_models()
    all_models = list_all_models()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Models", len(all_models))
    with col2:
        st.metric("Trained Models", len(available))
    with col3:
        st.metric("Evaluated Models", len(evaluated))
    with col4:
        stats = get_dataset_stats(DATA_DIR)
        st.metric("Dataset Images", stats.get("total", 0))

    st.divider()

    # ── Dataset Distribution ──────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📊 Dataset Class Distribution")
        stats = get_dataset_stats(DATA_DIR)
        if stats.get("total", 0) == 0:
            st.info("No dataset found. Place images in `data/0`, `data/1`, ..., `data/4` folders.")
        else:
            labels = [f"Stage {i}: {DR_STAGES[i]}" for i in range(5)]
            counts = [stats.get(i, 0) for i in range(5)]
            colors_list = [DR_STAGE_COLORS[i] for i in range(5)]
            fig = go.Figure(go.Bar(
                x=labels, y=counts,
                marker_color=colors_list,
                text=counts, textposition="auto",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False, title="Images"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🤖 Model Status")
        rows = []
        for name, display in MODELS.items():
            trained    = "✅ Trained" if name in available else "⬛ Not Trained"
            evaluated  = "✅ Evaluated" if name in list_evaluated_models() else "—"
            rows.append({"Model": display, "Status": trained, "Evaluated": evaluated})

        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Quick Access ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### ⚡ Quick Actions")
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        if st.button("🏋️ Train a Model", use_container_width=True):
            st.session_state["page"] = "admin_train"
            st.rerun()
    with qcol2:
        if st.button("📈 Evaluate Models", use_container_width=True):
            st.session_state["page"] = "admin_evaluate"
            st.rerun()
    with qcol3:
        if st.button("⚖️ Compare Models", use_container_width=True):
            st.session_state["page"] = "admin_compare"
            st.rerun()

    # ── Recent Results Summary ─────────────────────────────────────────────────
    evaluated_list = list_evaluated_models()
    if evaluated_list:
        st.divider()
        st.markdown("#### 📋 Latest Evaluation Summary")
        summary_rows = []
        for mname in evaluated_list:
            res = load_results(mname)
            m   = res.get("metrics", {})
            summary_rows.append({
                "Model":    MODELS.get(mname, mname),
                "Accuracy": f"{m.get('accuracy', 0)*100:.1f}%",
                "F1 (W)":   f"{m.get('f1_weighted', 0):.3f}",
                "Kappa":    f"{m.get('quadratic_kappa', 0):.3f}",
                "AUC":      f"{m.get('macro_auc', 0) or 0:.3f}",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
