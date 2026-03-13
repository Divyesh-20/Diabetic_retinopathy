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


def load_model(model_name: str) -> tf.keras.Model:
    """
    Build the architecture and load saved weights (.h5) from SAVED_MODELS_DIR.
    Raises FileNotFoundError if weights don't exist yet.
    """
    weight_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}.h5")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"No saved weights found for '{model_name}' at {weight_path}. "
            f"Please train the model first."
        )
    model = build_model(model_name)
    model.load_weights(weight_path)
    return model


def list_available_models() -> list:
    """Return model names that have saved weights."""
    if not os.path.isdir(SAVED_MODELS_DIR):
        return []
    return [
        name for name in MODELS.keys()
        if os.path.exists(os.path.join(SAVED_MODELS_DIR, f"{name}.h5"))
    ]


def list_all_models() -> list:
    """Return all registered model names."""
    return list(MODELS.keys())
