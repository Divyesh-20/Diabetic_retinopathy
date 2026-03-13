"""
utils/trainer.py – Training loop for DR detection models
"""

import os
import numpy as np
import tensorflow as tf
from config import SAVED_MODELS_DIR, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_BATCH_SIZE


def get_callbacks(model_name: str, patience: int = 7):
    """Standard training callbacks."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    weight_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}.h5")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weight_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=0,
        ),
    ]


def train_model(model: tf.keras.Model,
                model_name: str,
                train_ds,
                val_ds,
                epochs: int = DEFAULT_EPOCHS,
                learning_rate: float = DEFAULT_LR,
                class_weights: dict = None,
                progress_callback=None) -> dict:
    """
    Compile and train the model.
    progress_callback: optional callable(epoch, logs) for Streamlit progress updates.
    Returns: history dict {train_acc, val_acc, train_loss, val_loss}
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    callbacks = get_callbacks(model_name)

    # Wrap progress callback
    class StreamlitProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, logs or {})

    if progress_callback:
        callbacks.append(StreamlitProgressCallback())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
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
    """Check if saved weights exist for this model."""
    path = os.path.join(SAVED_MODELS_DIR, f"{model_name}.h5")
    return os.path.exists(path)
