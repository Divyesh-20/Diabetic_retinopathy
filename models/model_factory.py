"""
models/model_factory.py – Unified model build and load interface
"""

import os
import tensorflow as tf
from config import SAVED_MODELS_DIR, MODELS


def build_model(model_name: str, trainable_base: bool = False) -> tf.keras.Model:
    """
    Instantiate a fresh model by name.
    model_name: one of the keys in config.MODELS
    """
    if model_name == "alexnet":
        from models.alexnet import build_alexnet
        return build_alexnet()

    elif model_name == "densenet":
        from models.densenet import build_densenet
        return build_densenet(trainable_base=trainable_base)

    elif model_name == "inceptionnet":
        from models.inceptionnet import build_inceptionnet
        return build_inceptionnet(trainable_base=trainable_base)

    elif model_name == "efficientnet":
        from models.efficientnet import build_efficientnet
        return build_efficientnet(trainable_base=trainable_base)

    elif model_name == "resnet":
        from models.resnet import build_resnet
        return build_resnet(trainable_base=trainable_base)

    elif model_name == "mobilenet":
        from models.mobilenet import build_mobilenet
        return build_mobilenet(trainable_base=trainable_base)

    elif model_name == "cnn_lstm":
        from models.cnn_lstm import build_cnn_lstm
        return build_cnn_lstm()

    elif model_name == "inception_resnet_lstm":
        from models.inception_resnet_lstm import build_inception_resnet_lstm
        return build_inception_resnet_lstm(trainable_base=trainable_base)

    elif model_name == "mobilenet_lstm":
        from models.mobilenet_lstm import build_mobilenet_lstm
        return build_mobilenet_lstm(trainable_base=trainable_base)

    else:
        raise ValueError(f"Unknown model name: '{model_name}'. "
                         f"Choose from: {list(MODELS.keys())}")


def get_model_save_path(model_name: str) -> str:
    """Return the preferred save path (.keras format)."""
    return os.path.join(SAVED_MODELS_DIR, f"{model_name}.keras")


def get_model_legacy_path(model_name: str) -> str:
    """Return the legacy .h5 path for backward compatibility."""
    return os.path.join(SAVED_MODELS_DIR, f"{model_name}.h5")


def load_model(model_name: str) -> tf.keras.Model:
    """
    Load a saved model. Tries .keras (full model) first, then .h5 (weights-only fallback).
    Uses tf.keras.models.load_model for full state restoration (architecture + weights + optimizer).
    Raises FileNotFoundError if no saved model exists.
    """
    keras_path = get_model_save_path(model_name)
    h5_path    = get_model_legacy_path(model_name)

    # 1. Prefer .keras full-model save
    if os.path.exists(keras_path):
        try:
            return tf.keras.models.load_model(keras_path)
        except Exception as e:
            raise ValueError(f"Failed to load full model '{model_name}' from {keras_path}: {e}")

    # 2. Fallback: legacy weights-only .h5
    if os.path.exists(h5_path):
        try:
            model = build_model(model_name)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            model.load_weights(h5_path)
            return model
        except Exception as e:
            raise ValueError(
                f"Failed to load weights for '{model_name}' from {h5_path}. "
                f"Delete the .h5 file and retrain the model. Error: {e}"
            )

    raise FileNotFoundError(
        f"No saved model found for '{model_name}'. "
        f"Looked for: {keras_path} and {h5_path}. "
        f"Please train the model first."
    )


def list_available_models() -> list:
    """Return model names that have saved weights/models."""
    if not os.path.isdir(SAVED_MODELS_DIR):
        return []
    available = []
    for name in MODELS.keys():
        keras_path = get_model_save_path(name)
        h5_path    = get_model_legacy_path(name)
        if os.path.exists(keras_path) or os.path.exists(h5_path):
            available.append(name)
    return available


def list_all_models() -> list:
    """Return all registered model names."""
    return list(MODELS.keys())
