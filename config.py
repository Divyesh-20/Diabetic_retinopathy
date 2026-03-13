"""
config.py – Global constants and settings for the DR Detection System
"""

import os

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "dataset")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

# ── Auth Credentials ──────────────────────────────────────────────────────────
CREDENTIALS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user":  {"password": "user123",  "role": "user"},
}

# ── DR Stage Labels ───────────────────────────────────────────────────────────
DR_STAGES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

DR_STAGE_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy are present. Continue regular eye exams.",
    1: "Mild non-proliferative diabetic retinopathy. Microaneurysms are present. Routine monitoring recommended.",
    2: "Moderate non-proliferative diabetic retinopathy. More blood vessels are affected. Closer monitoring required.",
    3: "Severe non-proliferative diabetic retinopathy. Many blood vessels are blocked. High risk of progression.",
    4: "Proliferative diabetic retinopathy. New abnormal blood vessels grow. Immediate medical attention required.",
}

DR_STAGE_COLORS = {
    0: "#28a745",   # green
    1: "#ffc107",   # yellow
    2: "#fd7e14",   # orange
    3: "#dc3545",   # red
    4: "#7b1fa2",   # purple
}

# ── Model Registry ────────────────────────────────────────────────────────────
MODELS = {
    "alexnet":               "AlexNet",
    "densenet":              "DenseNet121",
    "inceptionnet":          "InceptionV3",
    "efficientnet":          "EfficientNetB0",
    "resnet":                "ResNet50",
    "mobilenet":             "MobileNetV2",
    "cnn_lstm":              "CNN + LSTM Hybrid",
    "inception_resnet_lstm": "InceptionResNetV2 + LSTM (Main)",
    "mobilenet_lstm":        "MobileNetV2 + LSTM Hybrid",
}

MAIN_MODEL = "inception_resnet_lstm"

# ── Image Settings ────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
NUM_CLASSES = 5
BATCH_SIZE  = 16

# ── Training Defaults ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS      = 30
DEFAULT_LR          = 1e-4
DEFAULT_BATCH_SIZE  = 16

# ── GradCAM Last-Conv Layer Names per Model ───────────────────────────────────
GRADCAM_LAYERS = {
    "alexnet":               "conv2d_4",
    "densenet":              "conv5_block16_1_conv",
    "inceptionnet":          "mixed10",
    "efficientnet":          "top_conv",
    "resnet":                "conv5_block3_out",
    "mobilenet":             "Conv_1",
    "cnn_lstm":              "time_distributed_4",
    "inception_resnet_lstm": "mixed_7a",
    "mobilenet_lstm":        "Conv_1",
}
