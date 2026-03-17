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
    if model_name == "cnn_lstm":
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
    # 1. Try loading the entire model first (most accurate for recovery)
    try:
        model = tf.keras.models.load_model(weight_path)
        print(f"DEBUG: Successfully loaded full model '{model_name}'")
        return model
    except Exception as e:
        print(f"DEBUG: Full model load failed ({e}), falling back to building architecture...")

    # 2. Build and load
    model = build_model(model_name)
    
    try:
        model.load_weights(weight_path)
        print(f"DEBUG: Successfully loaded EXACT weights for '{model_name}'")
    except Exception:
        print(f"DEBUG: Exact load failed. Attempting Shape-Based Alignment...")
        try:
            import h5py
            import numpy as np
            
            # Open H5 directly
            with h5py.File(weight_path, "r") as f:
                # Get all datasets recursively
                h5_datasets = {}
                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        h5_datasets[obj.shape] = h5_datasets.get(obj.shape, []) + [obj[()]]
                f.visititems(visitor)
                
                # Assign to model layers by shape matching
                assigned = 0
                for weight in model.weights:
                    shape = weight.shape.as_list()
                    shape_tuple = tuple(shape)
                    if shape_tuple in h5_datasets and h5_datasets[shape_tuple]:
                        # Pop the first matching weight
                        new_val = h5_datasets[shape_tuple].pop(0)
                        weight.assign(new_val)
                        assigned += 1
                
                print(f"DEBUG: Shape-Based Recovery: Assigned {assigned}/{len(model.weights)} weights.")
                
                if assigned < (len(model.weights) * 0.5):
                    print("WARNING: Less than 50% of weights were recovered.")
        except Exception as e:
            print(f"DEBUG: Shape-Based Recovery FAILED: {e}")
            # Final fallback: by_name
            model.load_weights(weight_path, by_name=True, skip_mismatch=True)

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
