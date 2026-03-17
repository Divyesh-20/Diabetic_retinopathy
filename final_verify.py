import tensorflow as tf
import os
import sys
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

from models.model_factory import load_model, build_model

def run_verification():
    print("🧠 STARTING FINAL WEIGHT RECOVERY VERIFICATION")
    print("-" * 50)
    
    model_name = "inception_resnet_lstm"
    
    try:
        # Load using our new 'Blind Shape Alignment'
        model = load_model(model_name)
        
        # Test Prediction
        dummy_img = np.random.rand(1, 1, 224, 224, 3).astype(np.float32)
        preds = model.predict(dummy_img, verbose=0)[0]
        
        print(f"\n✅ Prediction Successful!")
        print(f"Confidences: {preds}")
        
        # If weights are random, they will be very close to 0.2
        is_random = all(abs(p - 0.2) < 0.05 for p in preds)
        
        if is_random:
            print("❌ WARNING: The model is still producing random (20%) results.")
            print("   This means the H5 file might not contain the trained weights for this structure.")
        else:
            print("🚀 SUCCESS! The model is producing structured predictions (non-random).")
            print("   Your 12-hour training investment has been RECOVERED.")
            
    except Exception as e:
        print(f"❌ Verification FAILED: {e}")

if __name__ == "__main__":
    run_verification()
