import h5py
import os

h5_path = r"c:\Users\shind\project\saved_models\inception_resnet_lstm.h5"
if not os.path.exists(h5_path):
    print(f"File not found: {h5_path}")
else:
    print(f"Inspecting: {h5_path}")
    print(f"Size: {os.path.getsize(h5_path) / 1024 / 1024:.2f} MB")
    
    with h5py.File(h5_path, "r") as f:
        print("\nTop level keys:")
        for key in list(f.keys())[:20]:
            print(f" - {key}")
        
        # Check if 'model_weights' or similar exists (Keras 2 vs 3 structure)
        if "model_weights" in f:
            print("\nModel Weights keys:")
            for key in list(f["model_weights"].keys())[:20]:
                print(f"   - {key}")
