import os
import sys
import tensorflow as tf
import numpy as np

print("--- AUDIT START ---")

# 1. Imports
try:
    import config
    from models.model_factory import build_model
    from utils.dataset import get_image_paths_labels
    print("[OK] Imports successful.")
except Exception as e:
    print(f"[FAIL] Imports: {e}")
    sys.exit(1)

# 2. Paths
if os.path.exists(config.DATA_DIR):
    print(f"[OK] DATA_DIR exists: {config.DATA_DIR}")
    paths, labels = get_image_paths_labels(config.DATA_DIR)
    print(f"     Images found: {len(paths)}")
else:
    print(f"[FAIL] DATA_DIR missing: {config.DATA_DIR}")

# 3. Model Build (MobileNet-LSTM only)
print("\nTesting MobileNet-LSTM build...")
try:
    # Use weights=None to avoid downloading if internet is slow during audit
    from models.mobilenet_lstm import build_mobilenet_lstm
    # Actually, model_factory.build_model('mobilenet_lstm') calls build_mobilenet_lstm()
    # Let's see if we can build it without pretrained weights first to see if it's the download hanging
    m = build_mobilenet_lstm()
    print("[OK] MobileNet-LSTM built.")
    
    dummy_img = np.random.rand(1, 1, 224, 224, 3)
    pred = m.predict(dummy_img, verbose=0)
    print(f"[OK] Inference test successful. Prediction shape: {pred.shape}")
except Exception as e:
    print(f"[FAIL] Model/Inference: {e}")

print("\n--- AUDIT END ---")
