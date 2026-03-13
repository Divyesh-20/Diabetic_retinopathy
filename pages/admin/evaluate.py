"""
pages/admin/evaluate.py – Per-model evaluation metrics and visualisations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd

from utils.auth import require_role
from utils.metrics import load_results, list_evaluated_models, compute_metrics
from utils.gradcam import generate_gradcam
from utils.preprocessing import preprocess_for_model, load_image_pil
from models.model_factory import load_model, list_available_models
from config import MODELS, DR_STAGES, DR_STAGE_COLORS, NUM_CLASSES


def show_evaluate_page():
    require_role("admin")

    st.markdown("## 📈 Model Evaluation")
    st.divider()

    evaluated = list_evaluated_models()
    available = list_available_models()

    if not evaluated and not available:
        st.info("No trained or evaluated models found. Go to **Train Model** first.")
        return

    # ── Model selector ────────────────────────────────────────────────────────
    selectable = list(set(evaluated) | set(available))
    model_key  = st.selectbox(
        "Select Model to Evaluate",
        options=selectable,
        format_func=lambda k: MODELS.get(k, k),
    )

    # ── Load saved results ─────────────────────────────────────────────────────
    saved = load_results(model_key)
    metrics = saved.get("metrics", {})
    history = saved.get("history", {})

    # Option: evaluate on newly uploaded images
    with st.expander("🔄 Re-evaluate on uploaded images"):
        test_files = st.file_uploader(
            "Upload test images (multiple)", accept_multiple_files=True,
            type=["png", "jpg", "jpeg"], key="eval_upload"
        )
        test_labels_str = st.text_input(
            "True labels (comma-separated, same order as images)",
            placeholder="e.g. 0,1,2,3,4,0"
        )
        if st.button("Run Evaluation", key="run_eval") and test_files:
            if not test_labels_str.strip():
                st.error("Please provide ground truth labels.")
            else:
                try:
                    true_labels = [int(x.strip()) for x in test_labels_str.split(",")]
                    if len(true_labels) != len(test_files):
                        st.error("Number of labels must match number of images.")
                    else:
                        model = load_model(model_key)
                        preds_list = []
                        for f in test_files:
                            f.seek(0)
                            batch = preprocess_for_model(f, model_key)
                            preds_list.append(model.predict(batch, verbose=0)[0])
                        y_pred = np.array(preds_list)
                        y_true = np.array(true_labels)
                        from utils.metrics import compute_metrics, save_results
                        metrics = compute_metrics(y_true, y_pred)
                        save_results(model_key, metrics)
                        st.success("Evaluation complete! Results saved.")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    if not metrics:
        st.info("No evaluation results found for this model. Train and evaluate it first.")
        return

    # ── Key Metrics ───────────────────────────────────────────────────────────
    st.markdown("#### 🎯 Key Metrics")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Accuracy",   f"{metrics.get('accuracy', 0)*100:.2f}%")
    mc2.metric("F1 Weighted",f"{metrics.get('f1_weighted', 0):.3f}")
    mc3.metric("F1 Macro",   f"{metrics.get('f1_macro', 0):.3f}")
    mc4.metric("Kappa (QW)", f"{metrics.get('quadratic_kappa', 0):.3f}")
    mc5.metric("Macro AUC",  f"{metrics.get('macro_auc') or 0:.3f}")

    st.divider()
    col_cm, col_roc = st.columns(2)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    with col_cm:
        st.markdown("#### 🔲 Confusion Matrix")
        cm = metrics.get("confusion_matrix")
        if cm:
            cm_arr = np.array(cm)
            labels = [f"Stage {i}" for i in range(NUM_CLASSES)]
            fig_cm = ff.create_annotated_heatmap(
                cm_arr,
                x=labels, y=labels,
                colorscale="Blues",
                showscale=True,
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=0, r=0, t=40, b=0),
                height=350,
                xaxis=dict(title="Predicted"),
                yaxis=dict(title="Actual", autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curves ────────────────────────────────────────────────────────────
    with col_roc:
        st.markdown("#### 📉 ROC Curves (per class)")
        roc_data = metrics.get("roc_curves", {})
        if roc_data:
            fig_roc = go.Figure()
            for c_str, curve in roc_data.items():
                c = int(c_str)
                if curve:
                    auc_val = metrics["auc_per_class"].get(c_str, metrics["auc_per_class"].get(c, 0)) or 0
                    fig_roc.add_trace(go.Scatter(
                        x=curve["fpr"], y=curve["tpr"],
                        mode="lines",
                        name=f"{DR_STAGES[c]} (AUC={auc_val:.2f})",
                        line=dict(color=DR_STAGE_COLORS[c]),
                    ))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                          line=dict(dash="dash", color="gray"),
                                          showlegend=False))
            fig_roc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(title="FPR", showgrid=False),
                yaxis=dict(title="TPR", showgrid=False),
                margin=dict(l=0, r=0, t=10, b=0),
                height=350, legend=dict(x=0.01, y=0.01),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # ── Per-class Classification Report ───────────────────────────────────────
    cls_report = metrics.get("classification_report", {})
    if cls_report:
        st.divider()
        st.markdown("#### 📋 Per-Class Classification Report")
        rows = []
        for label, vals in cls_report.items():
            if isinstance(vals, dict):
                rows.append({
                    "Class":     label,
                    "Precision": f"{vals.get('precision', 0):.3f}",
                    "Recall":    f"{vals.get('recall',    0):.3f}",
                    "F1-score":  f"{vals.get('f1-score',  0):.3f}",
                    "Support":   int(vals.get("support", 0)),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Training History ──────────────────────────────────────────────────────
    if history and history.get("train_acc"):
        st.divider()
        st.markdown("#### 📈 Training History")
        eps = list(range(1, len(history["train_acc"]) + 1))
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=eps, y=[a*100 for a in history["train_acc"]],
                                    name="Train Acc", line=dict(color="#42A5F5")))
        fig_h.add_trace(go.Scatter(x=eps, y=[a*100 for a in history["val_acc"]],
                                    name="Val Acc",   line=dict(color="#66BB6A")))
        fig_h.add_trace(go.Scatter(x=eps, y=history["train_loss"],
                                    name="Train Loss", line=dict(color="#EF5350"), yaxis="y2"))
        fig_h.add_trace(go.Scatter(x=eps, y=history["val_loss"],
                                    name="Val Loss",   line=dict(color="#FF7043"), yaxis="y2"))
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Accuracy (%)", showgrid=False),
            yaxis2=dict(title="Loss", overlaying="y", side="right", showgrid=False),
            margin=dict(l=0, r=0, t=10, b=0),
            height=300, legend=dict(x=0, y=1),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # ── GradCAM on uploaded sample ────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🎯 Grad-CAM Visualisation")
    gc_file = st.file_uploader("Upload a sample image for Grad-CAM", type=["png","jpg","jpeg"],
                                key="eval_gradcam")
    if gc_file:
        if model_key in list_available_models():
            with st.spinner("Generating GradCAM…"):
                try:
                    model = load_model(model_key)
                    gc_file.seek(0)
                    orig, overlay, _ = generate_gradcam(model, gc_file, model_key)
                    gc_col1, gc_col2 = st.columns(2)
                    gc_col1.image(orig,    caption="Original", use_column_width=True)
                    gc_col2.image(overlay, caption="Grad-CAM", use_column_width=True)
                except Exception as e:
                    st.error(f"GradCAM failed: {e}")
        else:
            st.warning("Model weights not found. Cannot generate GradCAM.")
