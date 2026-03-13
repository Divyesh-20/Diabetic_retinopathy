"""
pages/admin/train.py – Model training page for admin
"""

import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go

from utils.auth import require_role
from utils.dataset import get_image_paths_labels, split_dataset, make_tf_dataset, get_class_weights
from utils.trainer import train_model, is_model_trained
from utils.metrics import compute_metrics, save_results, load_results
from models.model_factory import build_model, load_model
from config import MODELS, DATA_DIR, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_BATCH_SIZE


def show_train_page():
    require_role("admin")

    st.markdown("## 🏋️ Train Model")
    st.markdown("Select a model, configure hyperparameters, and start training.")
    st.divider()

    # ── Model selection ───────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        model_key = st.selectbox(
            "Select Model",
            options=list(MODELS.keys()),
            format_func=lambda k: MODELS[k],
            key="train_model_select",
        )
    with col2:
        already_trained = is_model_trained(model_key)
        if already_trained:
            st.info("✅ Pretrained weights exist. Training will fine-tune.")
        else:
            st.warning("⬛ No saved weights. Will train from scratch.")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Hyperparameters")
    hp_col1, hp_col2, hp_col3 = st.columns(3)
    with hp_col1:
        epochs = st.slider("Epochs", min_value=5, max_value=100, value=DEFAULT_EPOCHS, step=5)
    with hp_col2:
        lr = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=DEFAULT_LR,
            format_func=lambda v: f"{v:.0e}",
        )
    with hp_col3:
        batch_size = st.select_slider(
            "Batch Size", options=[8, 16, 32, 64], value=DEFAULT_BATCH_SIZE
        )

    use_class_weights = st.checkbox("Use class weights (recommended for imbalanced data)", value=True)
    trainable_base = st.checkbox(
        "Fine-tune backbone (unfreeze pretrained layers)", value=False
    )

    st.markdown("#### 📂 Dataset")
    data_dir = st.text_input("Dataset Directory", value=DATA_DIR,
                              help="Folder with sub-folders named 0, 1, 2, 3, 4")

    # ── Start Training ────────────────────────────────────────────────────────
    st.divider()
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Validate dataset
        import os
        if not os.path.isdir(data_dir):
            st.error(f"Dataset directory not found: `{data_dir}`")
            return

        with st.spinner("Loading dataset…"):
            try:
                paths, labels = get_image_paths_labels(data_dir)
                if len(paths) == 0:
                    st.error("No images found. Check folder structure (sub-folders 0–4).")
                    return
                (Xtr, ytr), (Xv, yv), (Xte, yte) = split_dataset(paths, labels)
            except Exception as e:
                st.error(f"Dataset loading failed: {e}")
                return

        lstm_models = {"cnn_lstm", "inception_resnet_lstm", "mobilenet_lstm"}
        is_lstm = model_key in lstm_models

        train_ds = make_tf_dataset(Xtr, ytr, batch_size, augment=True,  lstm=is_lstm)
        val_ds   = make_tf_dataset(Xv,  yv,  batch_size, augment=False, lstm=is_lstm)
        test_ds  = make_tf_dataset(Xte, yte, batch_size, augment=False, lstm=is_lstm)

        cw = get_class_weights(ytr) if use_class_weights else None

        # Build model
        with st.spinner("Building model…"):
            if already_trained:
                try:
                    model = load_model(model_key)
                    st.success("Loaded pretrained weights.")
                except Exception:
                    model = build_model(model_key, trainable_base=trainable_base)
                    st.info("Starting fresh (weight load failed).")
            else:
                model = build_model(model_key, trainable_base=trainable_base)

        # Live training progress
        st.markdown("#### 📈 Training Progress")
        epoch_ph    = st.empty()
        progress_bar = st.progress(0)
        chart_ph    = st.empty()

        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        epoch_counter = [0]

        def epoch_callback(epoch_num, logs):
            epoch_counter[0] = epoch_num
            progress_bar.progress(int(epoch_num / epochs * 100))
            train_accs.append(logs.get("accuracy", 0))
            val_accs.append(logs.get("val_accuracy", 0))
            train_losses.append(logs.get("loss", 0))
            val_losses.append(logs.get("val_loss", 0))

            epoch_ph.markdown(
                f"**Epoch {epoch_num}/{epochs}** — "
                f"Loss: `{logs.get('loss', 0):.4f}` | "
                f"Acc: `{logs.get('accuracy', 0)*100:.2f}%` | "
                f"Val Loss: `{logs.get('val_loss', 0):.4f}` | "
                f"Val Acc: `{logs.get('val_accuracy', 0)*100:.2f}%`"
            )

            if len(train_accs) > 1:
                eps = list(range(1, len(train_accs) + 1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=eps, y=[a*100 for a in train_accs],
                                          mode="lines+markers", name="Train Acc",
                                          line=dict(color="#42A5F5")))
                fig.add_trace(go.Scatter(x=eps, y=[a*100 for a in val_accs],
                                          mode="lines+markers", name="Val Acc",
                                          line=dict(color="#66BB6A")))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    xaxis=dict(title="Epoch"), yaxis=dict(title="Accuracy (%)"),
                    margin=dict(l=0, r=0, t=10, b=0), height=250, legend=dict(x=0, y=1),
                )
                chart_ph.plotly_chart(fig, use_container_width=True)

        try:
            history = train_model(model, model_key, train_ds, val_ds,
                                   epochs=epochs, learning_rate=lr,
                                   class_weights=cw,
                                   progress_callback=epoch_callback)
        except Exception as e:
            st.error(f"Training failed: {e}")
            return

        progress_bar.progress(100)
        st.success(f"✅ Training complete! Weights saved to `saved_models/{model_key}.h5`")

        # ── Auto-evaluate on test set ─────────────────────────────────────────
        st.markdown("#### 📊 Test Set Evaluation")
        with st.spinner("Evaluating on test set…"):
            from utils.trainer import evaluate_model_on_dataset
            y_true, y_pred_proba = evaluate_model_on_dataset(model, test_ds)
            from utils.metrics import compute_metrics, save_results
            metrics = compute_metrics(y_true, y_pred_proba)
            save_results(model_key, metrics, history)

        m = metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
        mc2.metric("F1 (Wt.)",  f"{m['f1_weighted']:.3f}")
        mc3.metric("Kappa",     f"{m['quadratic_kappa']:.3f}")
        mc4.metric("Macro AUC", f"{m.get('macro_auc') or 0:.3f}")
        st.info("Full metrics available in the **Evaluate** page.")
