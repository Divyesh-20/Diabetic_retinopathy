"""
utils/trainer.py – Training loop for DR detection models
"""

import os
import numpy as np
import tensorflow as tf
from config import SAVED_MODELS_DIR, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_BATCH_SIZE
from models.model_factory import get_model_save_path


def get_callbacks(model_name: str, patience: int = 7, fine_tuning: bool = False):
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    save_path = get_model_save_path(model_name)

    monitor = "val_loss"
    # Fine-tuning is more volatile → be a little more patient before stopping
    eff_patience = patience + 3 if fine_tuning else patience

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            save_best_only=True,
            save_weights_only=False,
            monitor=monitor,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=eff_patience,
            restore_best_weights=True,   # ← CRITICAL: restores best weights on stop
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,          # gentler reduction during fine-tuning
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def _freeze_batch_norm_layers(model: tf.keras.Model):
    """
    Keep BatchNormalization layers frozen during fine-tuning.
    BN running statistics are calibrated on ImageNet; re-training them
    with a small domain-specific dataset destabilises the backbone.
    """
    bn_frozen = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            bn_frozen += 1
        # Handle nested models (e.g. TimeDistributed wrapping InceptionResNetV2)
        elif hasattr(layer, "layer") and isinstance(layer.layer, tf.keras.Model):
            for sub in layer.layer.layers:
                if isinstance(sub, tf.keras.layers.BatchNormalization):
                    sub.trainable = False
                    bn_frozen += 1
        elif hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.BatchNormalization):
                    sub.trainable = False
                    bn_frozen += 1
    return bn_frozen


def train_model(
    model: tf.keras.Model,
    model_name: str,
    train_ds,
    val_ds,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    class_weights: dict = None,
    progress_callback=None,
    fine_tuning: bool = False,
    extra_callbacks: list = None,
) -> dict:
    """
    Compile and train the model.

    Fine-tuning fixes applied:
      1. LR is scaled down to 1/10th (prevents catastrophic forgetting)
      2. BatchNorm layers are frozen (preserves ImageNet statistics)
      3. Label smoothing is lower for fine-tuning (model is already calibrated)
      4. EarlyStopping patience is increased (fine-tuning converges slower)
      5. ReduceLROnPlateau uses a gentler factor (0.5 instead of 0.3)

    Returns: history dict
    """
    # ── Learning rate ──────────────────────────────────────────────────────────
    # Fine-tuning MUST use a much smaller LR to avoid destroying pre-trained weights
    actual_lr = learning_rate * 0.1 if fine_tuning else learning_rate

    # ── Freeze BatchNorm during fine-tuning ───────────────────────────────────
    if fine_tuning:
        bn_count = _freeze_batch_norm_layers(model)
        print(f"[Fine-Tuning] Frozen {bn_count} BatchNorm layer(s) to preserve ImageNet statistics.")

    # ── Compile ───────────────────────────────────────────────────────────────
    # Label smoothing: lower during fine-tuning (model already calibrated)
    label_smoothing = 0.02 if fine_tuning else 0.05
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=actual_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0,        # ← gradient clipping prevents weight explosions
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = get_callbacks(model_name, fine_tuning=fine_tuning)

    class StreamlitProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, logs or {})

    if progress_callback:
        callbacks.append(StreamlitProgressCallback())

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # ── Fit ───────────────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return {
        "train_acc":  history.history.get("accuracy", []),
        "val_acc":    history.history.get("val_accuracy", []),
        "train_loss": history.history.get("loss", []),
        "val_loss":   history.history.get("val_loss", []),
        "train_auc":  history.history.get("auc", []),
        "val_auc":    history.history.get("val_auc", []),
    }


def evaluate_model_on_dataset(model: tf.keras.Model, test_ds) -> tuple:
    """
    Run model.predict on the test dataset.
    Returns (y_true_array, y_pred_proba_array).
    """
    y_true_list, y_pred_list = [], []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_pred_list.append(preds)
        y_true_list.append(np.argmax(batch_y.numpy(), axis=1))
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


def is_model_trained(model_name: str) -> bool:
    """Check if a saved model exists (either .keras or .h5)."""
    from models.model_factory import get_model_save_path, get_model_legacy_path
    keras_path = get_model_save_path(model_name)
    h5_path    = get_model_legacy_path(model_name)
    return os.path.exists(keras_path) or os.path.exists(h5_path)
