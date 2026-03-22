"""
pages/admin/train.py – Model training page for admin (FIXED)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os

from tensorflow.keras.callbacks import EarlyStopping

from utils.auth import require_role
from utils.dataset import (
    get_image_paths_labels,
    split_dataset,
    make_tf_dataset,
    get_class_weights,
)
from utils.trainer import train_model, is_model_trained
from utils.metrics import compute_metrics, save_results
from models.model_factory import build_model, load_model
from config import MODELS, DATA_DIR, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_BATCH_SIZE


def show_train_page():
    require_role("admin")

    st.markdown("## 🏋️ Train Model")
    st.divider()

    # ── Model selection ─────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        model_key = st.selectbox(
            "Select Model",
            options=list(MODELS.keys()),
            format_func=lambda k: MODELS[k],
        )

    with col2:
        already_trained = is_model_trained(model_key)
        if already_trained:
            st.info("Pretrained weights found → Fine-tuning mode")
        else:
            st.warning("Training from scratch")

    # ── Hyperparameters ─────────────────────────────────────
    st.markdown("#### ⚙️ Hyperparameters")

    hp1, hp2, hp3 = st.columns(3)

    with hp1:
        epochs = st.slider("Epochs", 5, 50, DEFAULT_EPOCHS, step=5)

    with hp2:
        lr = st.select_slider(
            "Learning Rate",
            options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
            value=DEFAULT_LR,
            format_func=lambda v: f"{v:.0e}",
        )

    with hp3:
        batch_size = st.select_slider(
            "Batch Size", options=[8, 16, 32], value=DEFAULT_BATCH_SIZE
        )

    use_class_weights = st.checkbox("Use class weights", value=True)
    trainable_base = st.checkbox("Unfreeze backbone (NOT recommended)", value=False)

    st.markdown("#### 📂 Dataset")

    data_dir = st.text_input("Dataset Directory", value=DATA_DIR)

    # ── Train button ────────────────────────────────────────
    st.divider()

    if st.button("🚀 Start Training", use_container_width=True):

        if not os.path.isdir(data_dir):
            st.error(f"Invalid dataset path: {data_dir}")
            return

        # ── Load dataset ─────────────────────────────────────
        with st.spinner("Loading dataset..."):
            try:
                paths, labels = get_image_paths_labels(data_dir)
                (Xtr, ytr), (Xv, yv), (Xte, yte) = split_dataset(paths, labels)
            except Exception as e:
                st.error(f"Dataset error: {e}")
                return

        # ── Dataset mode ─────────────────────────────────────
        lstm_models = {"cnn_lstm", "mobilenet_lstm", "inception_resnet_lstm"}
        is_lstm = model_key in lstm_models

        # 🔴 CLAHE FIX
        use_clahe = True

        train_ds = make_tf_dataset(
            Xtr, ytr, batch_size, augment=True, lstm=is_lstm, apply_clahe=use_clahe
        )

        val_ds = make_tf_dataset(
            Xv, yv, batch_size, augment=False, lstm=is_lstm, apply_clahe=use_clahe
        )

        test_ds = make_tf_dataset(
            Xte, yte, batch_size, augment=False, lstm=is_lstm, apply_clahe=use_clahe
        )

        cw = get_class_weights(ytr) if use_class_weights else None

        # ── Build / Load model ───────────────────────────────
        fine_tuning = False

        with st.spinner("Preparing model..."):
            if already_trained:
                try:
                    model = load_model(model_key)
                    fine_tuning = True

                   

                    # 🔴 Freeze most layers
                    freeze_until = int(len(model.layers) * 0.8)
                    for layer in model.layers[:freeze_until]:
                        layer.trainable = False

                    st.success(f"Fine-tuning with LR={lr:.1e}")

                except Exception as e:
                    st.warning("Failed to load model → training from scratch")
                    model = build_model(model_key, trainable_base=False)

            else:
                model = build_model(model_key, trainable_base=trainable_base)

        # ── Callbacks ────────────────────────────────────────
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        # ── Training UI ─────────────────────────────────────
        st.markdown("#### 📈 Training Progress")

        progress_bar = st.progress(0)
        chart_ph = st.empty()

        train_accs, val_accs = [], []

        def epoch_callback(epoch, logs):
            progress_bar.progress(int(epoch / epochs * 100))

            train_accs.append(logs.get("accuracy", 0))
            val_accs.append(logs.get("val_accuracy", 0))

            if len(train_accs) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=train_accs, name="Train"))
                fig.add_trace(go.Scatter(y=val_accs, name="Val"))
                chart_ph.plotly_chart(fig, use_container_width=True)

        # ── Train ───────────────────────────────────────────
        try:
            history = train_model(
                model,
                model_key,
                train_ds,
                val_ds,
                epochs=epochs,
                learning_rate=lr,
                class_weights=cw,
                progress_callback=epoch_callback,
                fine_tuning=fine_tuning,
                extra_callbacks=[early_stop],
            )
        except Exception as e:
            st.error(f"Training failed: {e}")
            return

        progress_bar.progress(100)
        st.success("Training complete")

        # ── Evaluation ──────────────────────────────────────
        st.markdown("#### 📊 Test Evaluation")

        from utils.trainer import evaluate_model_on_dataset

        y_true, y_pred = evaluate_model_on_dataset(model, test_ds)
        metrics = compute_metrics(y_true, y_pred)

        save_results(model_key, metrics, history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        c2.metric("F1", f"{metrics['f1_weighted']:.3f}")
        c3.metric("Kappa", f"{metrics['quadratic_kappa']:.3f}")