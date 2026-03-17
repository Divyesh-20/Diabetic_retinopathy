"""
utils/class_reversal_fix.py – Detect and fix class inversion issues
"""

import numpy as np
import os
from config import SAVED_MODELS_DIR

def should_reverse_classes():
    """
    Check if the trained models have reversed class mappings.
    
    This happens when:
    - No_DR (folder 0) → model outputs class 4 (Proliferative_DR)
    - Proliferate_DR (folder 4) → model outputs class 0 (No_DR)
    
    Returns: bool - True if classes should be reversed
    """
    # This can be stored as a flag file or environment variable
    # For now, check if a reversal flag file exists
    reversal_flag = os.path.join(SAVED_MODELS_DIR, ".class_reversal_flag")
    return os.path.exists(reversal_flag)

def fix_predictions(predictions_array, invert=None):
    """
    Fix inverted class predictions if needed.
    
    Args:
        predictions_array: numpy array of shape (5,) with class probabilities
        invert: bool or None. If None, auto-detect from flag file
    
    Returns:
        Fixed predictions array
    """
    if invert is None:
        invert = should_reverse_classes()
    
    if invert:
        return predictions_array[::-1]
    return predictions_array

def set_class_reversal(should_reverse: bool):
    """
    Set the class reversal flag.
    
    Args:
        should_reverse: bool - True to enable class reversal
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    reversal_flag = os.path.join(SAVED_MODELS_DIR, ".class_reversal_flag")
    
    if should_reverse:
        with open(reversal_flag, 'w') as f:
            f.write("1")
    else:
        if os.path.exists(reversal_flag):
            os.remove(reversal_flag)
