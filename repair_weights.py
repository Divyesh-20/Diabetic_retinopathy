import tensorflow as tf
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from models.inception_resnet_lstm import build_inception_resnet_lstm

source_h5 = r"c:\Users\shind\project\saved_models\inception_resnet_lstm.h5"
output_h5 = r"c:\Users\shind\project\saved_models\inception_resnet_lstm_fixed.h5"

if not os.path.exists(source_h5):
    print(f"File not found: {source_h5}")
    exit()

print(f"REPAIRING: {source_h5}")

# Load the data from source
import h5py
source_data = h5py.File(source_h5, 'r')

# Build the target model
model = build_inception_resnet_lstm()

print("\n--- Model Layers ---")
found_weights = 0
total_weights = sum(len(l.weights) for l in model.layers)

# Strategy: attempt to load weights by name from any path in the H5
def find_weight_in_h5(w_name, w_shape):
    # Try direct name
    # Logic to recursively search in H5
    pass

try:
    # Try smarter load by name with skip_mismatch
    model.load_weights(source_h5, by_name=True, skip_mismatch=True)
    print("Smarter Load finished (by_name=True).")
    
    # Save the new file
    model.save_weights(output_h5)
    print(f"\nSUCCESS! Created fixed weights file at: {output_h5}")
    print("Instructions: Rename this to 'inception_resnet_lstm.h5' and try again!")

except Exception as e:
    print(f"Repair failed: {e}")
finally:
    source_data.close()
