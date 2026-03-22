"""
utils/trainer.py – Training loop for DR detection models
"""

import os
import numpy as np
import tensorflow as tf
from config import SAVED_MODELS_DIR, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_BATCH_SIZE
from models.model_factory import get_model_save_path


def get_callbacks(model_name: str, patience: int = 7):
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    save_path = get_model_save_path(model_name)

    monitor = "val_loss"

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
            patience=patience,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

def train_model(model: tf.keras.Model,
    model_name: str,
    train_ds,
    val_ds,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    class_weights: dict = None,
    progress_callback=None,
    fine_tuning: bool = False,
    extra_callbacks: list = None, ) -> dict:
    """
    Compile and train the model.
    fine_tuning: if True, halve the learning rate and add label smoothing.
    progress_callback: optional callable(epoch, logs) for Streamlit progress updates.
    Returns: history dict
    """
    DEFAULT_LR = 1e-4
    actual_lr = learning_rate * 0.1 if fine_tuning else learning_rate
    min_lr = 1e-6

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=actual_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05 if fine_tuning else 0.02           # ← Prevents overconfidence, improves generalization
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = get_callbacks(model_name)

    # Streamlit live progress callback
    class StreamlitProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, logs or {})

    if progress_callback:
        callbacks.append(StreamlitProgressCallback())

    if extra_callbacks:
        callbacks.extend(extra_callbacks)    

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,    # ← prints epoch progress in terminal (loss, acc, val_loss, val_acc)
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
