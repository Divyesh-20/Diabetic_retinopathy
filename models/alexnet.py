"""
models/alexnet.py – Custom AlexNet for 5-class DR classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE, NUM_CLASSES


def build_alexnet(num_classes: int = NUM_CLASSES,
                  input_shape: tuple = (*IMG_SIZE, 3)) -> tf.keras.Model:
    """
    Custom AlexNet adapted for 224×224 fundus images.
    Architecture mirrors the original paper but uses BatchNorm for stability.
    """
    inp = layers.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(96, 11, strides=4, padding="same", activation="relu", name="conv2d_0")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Block 2
    x = layers.Conv2D(256, 5, padding="same", activation="relu", name="conv2d_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Block 3
    x = layers.Conv2D(384, 3, padding="same", activation="relu", name="conv2d_2")(x)
    x = layers.BatchNormalization()(x)

    # Block 4
    x = layers.Conv2D(384, 3, padding="same", activation="relu", name="conv2d_3")(x)
    x = layers.BatchNormalization()(x)

    # Block 5
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv2d_4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Fully connected
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs=inp, outputs=out, name="AlexNet")
