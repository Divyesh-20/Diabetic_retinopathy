"""
models/inceptionnet.py – InceptionV3 pretrained backbone for DR classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from config import IMG_SIZE, NUM_CLASSES


def build_inceptionnet(num_classes: int = NUM_CLASSES,
                        input_shape: tuple = (*IMG_SIZE, 3),
                        trainable_base: bool = False) -> tf.keras.Model:
    """
    InceptionV3 with ImageNet weights, custom classification head.
    """
    base = InceptionV3(weights="imagenet",
                       include_top=False,
                       input_shape=input_shape)
    base.trainable = trainable_base

    inp = layers.Input(shape=input_shape, name="input")
    x = base(inp, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs=inp, outputs=out, name="InceptionV3")
