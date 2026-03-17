import tensorflow as tf
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from models.model_factory import build_model
from config import MODELS

weight_path = r"c:\Users\shind\project\saved_models\inception_resnet_lstm.h5"

if not os.path.exists(weight_path):
    print(f"File not found: {weight_path}")
    exit()

print(f"Scanning all architectures for weights: {os.path.basename(weight_path)}")

for model_key in list(MODELS.keys()):
    print(f"\n--- Testing: {model_key} ---")
    try:
        model = build_model(model_key)
        model.load_weights(weight_path)
        print(f"MATCH FOUND: {model_key} loads weights successfully!")
    except Exception as e:
        err_msg = str(e).lower()
        if "shape mismatch" in err_msg or "size mismatch" in err_msg:
            # Print first 200 chars of mismatch
            print(f"Mismatch: {str(e)[:200]}")
        else:
            print(f"Error: {str(e)[:100]}...")
