"""
models/mobilenet_lstm.py – Hybrid MobileNetV2 + LSTM for DR classification
Input shape: (1, 224, 224, 3)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from config import IMG_SIZE, NUM_CLASSES


def build_mobilenet_lstm(num_classes: int = NUM_CLASSES,
                          input_shape: tuple = (1, *IMG_SIZE, 3),
                          trainable_base: bool = False) -> tf.keras.Model:
    """
    Lightweight hybrid: MobileNetV2 feature extractor + LSTM.
    Good balance of accuracy and speed.
    """
    base = MobileNetV2(weights="imagenet",
                       include_top=False,
                       input_shape=(*IMG_SIZE, 3))
    base.trainable = trainable_base

    inp = layers.Input(shape=input_shape, name="input")   # (batch, 1, 224, 224, 3)

    x = layers.TimeDistributed(base, name="mobilenetv2")(inp)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(),
                                name="td_gap")(x)          # (batch, 1, 1280)

    x = layers.LSTM(128, return_sequences=False,
                    dropout=0.3, name="lstm")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs=inp, outputs=out, name="MobileNetV2_LSTM")
