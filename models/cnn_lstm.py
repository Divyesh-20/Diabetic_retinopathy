"""
models/cnn_lstm.py – Hybrid CNN + LSTM model for DR classification
Uses a custom CNN feature extractor wrapped in TimeDistributed → LSTM
Input shape: (1, H, W, 3)  — single fundus image as a 1-frame sequence
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE, NUM_CLASSES


def build_cnn_lstm(num_classes: int = NUM_CLASSES,
                   input_shape: tuple = (1, *IMG_SIZE, 3)) -> tf.keras.Model:
    """
    Custom CNN wrapped with TimeDistributed, followed by LSTM.
    The CNN learns spatial features; LSTM adds temporal/contextual modelling.
    """
    inp = layers.Input(shape=input_shape, name="input")   # (batch, 1, H, W, 3)

    # TimeDistributed CNN Feature Extractor
    x = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu"),
                                name="time_distributed_0")(inp)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)

    x = layers.TimeDistributed(layers.Conv2D(64, 3, padding="same", activation="relu"),
                                name="time_distributed_1")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)

    x = layers.TimeDistributed(layers.Conv2D(128, 3, padding="same", activation="relu"),
                                name="time_distributed_2")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)

    x = layers.TimeDistributed(layers.Conv2D(256, 3, padding="same", activation="relu"),
                                name="time_distributed_3")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)

    x = layers.TimeDistributed(layers.Conv2D(256, 3, padding="same", activation="relu"),
                                name="time_distributed_4")(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)  # (batch, 1, 256)

    # LSTM
    x = layers.LSTM(256, return_sequences=False, dropout=0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs=inp, outputs=out, name="CNN_LSTM")
