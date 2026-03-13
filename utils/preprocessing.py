"""
utils/preprocessing.py – Image preprocessing utilities for fundus images
"""

import numpy as np
import cv2
from PIL import Image
from config import IMG_SIZE


def load_image_pil(image_source) -> Image.Image:
    """Load image from file path or uploaded file object."""
    if isinstance(image_source, str):
        return Image.open(image_source).convert("RGB")
    return Image.open(image_source).convert("RGB")


def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_clahe(image_array: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve visibility in fundus images.
    """
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_merged = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
    return result


def preprocess_image(image_source, apply_enhancement: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline:
      1. Load image
      2. Resize to IMG_SIZE
      3. Optional CLAHE enhancement
      4. Normalize to [0, 1]
    Returns: float32 array of shape (H, W, 3)
    """
    img = load_image_pil(image_source)
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    if apply_enhancement:
        arr_uint8 = arr.astype(np.uint8)
        arr_uint8 = apply_clahe(arr_uint8)
        arr = arr_uint8.astype(np.float32)

    arr = arr / 255.0
    return arr


def preprocess_for_model(image_source, model_name: str = "") -> np.ndarray:
    """
    Returns a batch-ready array.
    For LSTM-based models: shape (1, 1, H, W, 3) — time-distributed single frame
    For plain CNN models:  shape (1, H, W, 3)
    """
    arr = preprocess_image(image_source)   # (H, W, 3)
    lstm_models = {"cnn_lstm", "inception_resnet_lstm", "mobilenet_lstm"}
    if model_name in lstm_models:
        return arr[np.newaxis, np.newaxis, ...]   # (1, 1, H, W, 3)
    return arr[np.newaxis, ...]                   # (1, H, W, 3)


def augment_image(image_array: np.ndarray) -> np.ndarray:
    """
    Basic augmentation for training diversity.
    Expects array in [0,1] range, shape (H, W, 3).
    """
    arr = (image_array * 255).astype(np.uint8)
    # Random horizontal flip
    if np.random.rand() > 0.5:
        arr = cv2.flip(arr, 1)
    # Random vertical flip
    if np.random.rand() > 0.5:
        arr = cv2.flip(arr, 0)
    # Random brightness adjustment
    factor = np.random.uniform(0.8, 1.2)
    arr = np.clip(arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return arr.astype(np.float32) / 255.0
