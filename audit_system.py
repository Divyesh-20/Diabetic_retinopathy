import os
import sys
import tensorflow as tf
import streamlit as st
import numpy as np

def run_audit():
    print("="*50)
    print("DIABETIC RETINOPATHY SYSTEM - READINESS AUDIT")
    print("="*50)
    
    # 1. Base Imports
    try:
        from config import DATA_DIR, IMG_SIZE, MODELS, GRADCAM_LAYERS
        from models.model_factory import build_model
        from utils.dataset import get_image_paths_labels
        from utils.gradcam import generate_gradcam
        print("[OK] Core modules imported successfully.")
    except Exception as e:
        print(f"[FAIL] Core import error: {e}")
        return

    # 2. Path Verification
    if os.path.isdir(DATA_DIR):
        print(f"[OK] DATA_DIR exists: {DATA_DIR}")
    else:
        print(f"[WARN] DATA_DIR not found: {DATA_DIR}. Make sure you provide the correct path in the Admin Dashboard.")

    # 3. Model Architecture Verification
    print("\n--- Model Building Test ---")
    test_models = ["cnn_lstm", "inception_resnet_lstm", "mobilenet_lstm", "mobilenet"]
    for m_key in test_models:
        try:
            m = build_model(m_key)
            print(f"  [OK] Model '{m_key}' built successfully.")
            
            # Check Grad-CAM targets
            gc_target = GRADCAM_LAYERS.get(m_key)
            if gc_target:
                found = any(layer.name == gc_target for layer in m.layers)
                if found:
                    print(f"    - GradCAM layer '{gc_target}' found.")
                else:
                    print(f"    - [FAIL] GradCAM layer '{gc_target}' NOT FOUND in model layers.")
                    # List layers to help debug
                    # print([l.name for l in m.layers[-10:]])
            else:
                print(f"    - No GradCAM target defined in config.")
        except Exception as e:
            print(f"  [FAIL] Failed to build '{m_key}': {e}")

    # 4. Dataset Logic Verification
    print("\n--- Dataset Scan ---")
    try:
        paths, labels = get_image_paths_labels(DATA_DIR)
        print(f"  [OK] Dataset scan complete: {len(paths)} images found.")
        if len(paths) > 0:
            unique_labels = set(labels)
            print(f"  [OK] Classes found: {sorted(list(unique_labels))}")
    except Exception as e:
        print(f"  [FAIL] Dataset scan failed: {e}")

    # 5. Prediction Pipeline (Dummy)
    print("\n--- Inference Pipeline Test ---")
    try:
        dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        # Testing MobileNet_LSTM (for the speed-run)
        m_lstm = build_model("mobilenet_lstm")
        dummy_input = np.expand_dims(dummy_img, axis=0) # (1, 1, 224, 224, 3)
        pred = m_lstm.predict(dummy_input, verbose=0)
        print(f"  [OK] Model 'mobilenet_lstm' prediction successful. Output shape: {pred.shape}")
    except Exception as e:
        print(f"  [FAIL] Inference test failed: {e}")

    print("\n" + "="*50)
    print("AUDIT COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_audit()
