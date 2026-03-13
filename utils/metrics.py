"""
utils/metrics.py – Evaluation metrics for multi-class DR classification
"""

import numpy as np
import json
import os
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
from config import RESULTS_DIR, NUM_CLASSES, DR_STAGES


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.
    y_true:       integer labels (N,)
    y_pred_proba: softmax probabilities (N, 5)
    Returns a comprehensive metrics dict.
    """
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc     = accuracy_score(y_true, y_pred)
    f1_w    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_mac  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_w  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w   = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa   = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Per-class AUC (one-vs-rest) – requires ≥2 classes present per class
    auc_per_class = {}
    roc_curves    = {}
    y_true_bin = np.eye(NUM_CLASSES)[y_true]        # one-hot
    for c in range(NUM_CLASSES):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_pred_proba[:, c])
            auc_val = roc_auc_score(y_true_bin[:, c], y_pred_proba[:, c])
            auc_per_class[c] = float(auc_val)
            roc_curves[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        except ValueError:
            auc_per_class[c] = None
            roc_curves[c]    = None

    try:
        macro_auc = roc_auc_score(y_true_bin, y_pred_proba,
                                   average="macro", multi_class="ovr")
    except ValueError:
        macro_auc = None

    cls_report = classification_report(
        y_true, y_pred,
        target_names=[DR_STAGES[i] for i in range(NUM_CLASSES)],
        output_dict=True, zero_division=0
    )

    return {
        "accuracy":         float(acc),
        "f1_weighted":      float(f1_w),
        "f1_macro":         float(f1_mac),
        "precision_weighted": float(prec_w),
        "recall_weighted":  float(rec_w),
        "quadratic_kappa":  float(kappa),
        "macro_auc":        float(macro_auc) if macro_auc is not None else None,
        "auc_per_class":    auc_per_class,
        "confusion_matrix": cm.tolist(),
        "roc_curves":       roc_curves,
        "classification_report": cls_report,
    }


def save_results(model_name: str, metrics: dict, history: dict = None):
    """Save metrics (and optional training history) to RESULTS_DIR."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"metrics": metrics}
    if history:
        payload["history"] = history
    path = os.path.join(RESULTS_DIR, f"{model_name}_results.json")
    # Sanitise: convert ndarray → list
    payload_str = json.dumps(payload, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(path, "w") as f:
        f.write(payload_str)
    return path


def load_results(model_name: str) -> dict:
    """Load saved results for a model. Returns {} if not found."""
    path = os.path.join(RESULTS_DIR, f"{model_name}_results.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def list_evaluated_models() -> list:
    """Return list of model names that have saved results."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    return [
        f.replace("_results.json", "")
        for f in os.listdir(RESULTS_DIR)
        if f.endswith("_results.json")
    ]
