"""
utils/gradcam.py – Grad-CAM heatmap generation for Keras models
"""

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from config import GRADCAM_LAYERS, IMG_SIZE


def get_last_conv_layer_name(model_name: str, model: tf.keras.Model) -> str:
    """
    Return the target conv layer name for Grad-CAM.
    First checks GRADCAM_LAYERS registry, then auto-detects.
    """
    if model_name in GRADCAM_LAYERS:
        layer_name = GRADCAM_LAYERS[model_name]
        try:
            model.get_layer(layer_name)
            return layer_name
        except ValueError:
            pass  # fallback to auto-detect

    # Auto-detect: find last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return model.layers[-3].name   # fallback


def make_gradcam_heatmap(img_array: np.ndarray,
                          model: tf.keras.Model,
                          last_conv_layer_name: str,
                          pred_index: int = None) -> np.ndarray:
    """
    Generate raw GradCAM heatmap (H, W) in [0, 1].
    img_array: preprocessed image(s), shape accepted by the model
    """
    # Build gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output,
                 model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # Handle TimeDistributed outputs (LSTM models)
    if len(conv_outputs.shape) == 5:   # (batch, time, H, W, C)
        conv_outputs = conv_outputs[0, -1]   # last time step
        grads         = grads[0, -1]
    else:
        conv_outputs = conv_outputs[0]
        grads         = grads[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap: np.ndarray,
                     original_img: np.ndarray,
                     alpha: float = 0.4,
                     colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay GradCAM heatmap on original image.
    original_img: uint8 RGB array (H, W, 3)
    Returns: uint8 RGB overlay array
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def generate_gradcam(model: tf.keras.Model,
                      image_source,
                      model_name: str,
                      pred_index: int = None):
    """
    High-level GradCAM interface.
    image_source: PIL Image, file path, or uploaded file.
    Returns: (original_pil, gradcam_pil, heatmap_array)
    """
    from utils.preprocessing import preprocess_for_model, load_image_pil

    pil_img = load_image_pil(image_source)
    pil_resized = pil_img.resize(IMG_SIZE, Image.LANCZOS)
    original_array = np.array(pil_resized)   # uint8

    img_batch = preprocess_for_model(image_source, model_name)

    last_conv = get_last_conv_layer_name(model_name, model)

    try:
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv, pred_index)
    except Exception:
        # Fallback: return blank heatmap
        heatmap = np.zeros(IMG_SIZE)

    overlay = overlay_heatmap(heatmap, original_array)
    return pil_resized, Image.fromarray(overlay), heatmap
