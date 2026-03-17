"""
models/inception_resnet_lstm.py – MAIN MODEL
Hybrid InceptionResNetV2 + LSTM for 5-class DR classification

Architecture:
  InceptionResNetV2 (ImageNet) → GlobalAveragePooling2D
  → Reshape to (1, features) → LSTM(256) → Dense(5)

Input shape: (1, 224, 224, 3)  — single image as 1-frame time sequence
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionResNetV2
from config import IMG_SIZE, NUM_CLASSES


def build_inception_resnet_lstm(num_classes: int = NUM_CLASSES,
                                 input_shape: tuple = (1, *IMG_SIZE, 3),
                                 trainable_base: bool = False) -> tf.keras.Model:
    """
    InceptionResNetV2 + LSTM hybrid.
    This is the primary model used for patient-facing inference.
    """
    # ── CNN Backbone ──────────────────────────────────────────────────────────
    base = InceptionResNetV2(weights="imagenet",
                              include_top=False,
                              input_shape=(*IMG_SIZE, 3))
    base.trainable = trainable_base

    # ── Time-sequence input ───────────────────────────────────────────────────
    inp = layers.Input(shape=input_shape, name="input")   # (batch, 1, 224, 224, 3)

    # Apply CNN feature extraction to each frame via TimeDistributed
    x = layers.TimeDistributed(base, name="inception_resnetv2")(inp)
    
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(),
                                name="td_gap")(x)          # (batch, 1, 1536)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    x = layers.LSTM(256, return_sequences=False,
                    dropout=0.3, name="lstm")(x)           # (batch, 256)
    x = layers.BatchNormalization(name="bn_lstm")(x)

    # ── Classification Head ───────────────────────────────────────────────────
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs=inp, outputs=out, name="InceptionResNetV2_LSTM")
