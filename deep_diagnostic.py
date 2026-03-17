import tensorflow as tf
import os
import sys
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

from models.model_factory import load_model, build_model

def deep_check(model_name):
    print(f"\n{'='*40}")
    print(f"DEEP DIAGNOSTIC: {model_name}")
    print(f"{'='*40}")
    
    # 1. Build a fresh model for comparison
    fresh = build_model(model_name)
    fresh_w = fresh.weights[0].numpy()
    print(f"Fresh model weight[0] mean: {fresh_w.mean():.8f}, std: {fresh_w.std():.8f}")
    
    # 2. Try loading
    try:
        loaded = load_model(model_name)
        loaded_w = loaded.weights[0].numpy()
        print(f"Loaded model weight[0] mean: {loaded_w.mean():.8f}, std: {loaded_w.std():.8f}")
        
        diff = np.abs(fresh_w - loaded_w).sum()
        if diff < 1e-6:
            print("❌ WARNING: Loaded weights are IDENTICAL to fresh random weights. Load failed (skipped everything).")
        else:
            print("✅ SUCCESS: Weights have changed from random initialization.")
            
    except Exception as e:
        print(f"❌ LOAD FAILED: {e}")

if __name__ == "__main__":
    deep_check("inception_resnet_lstm")
    deep_check("mobilenet_lstm")
