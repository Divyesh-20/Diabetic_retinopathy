import h5py
import os
import tensorflow as tf
import numpy as np

h5_path = r"c:\Users\shind\project\saved_models\inception_resnet_lstm.h5"

def get_h5_shapes(path):
    shapes = {}
    with h5py.File(path, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                shapes[name] = obj.shape
        f.visititems(visitor)
    return shapes

print("Reading H5 shapes...")
h5_shapes = get_h5_shapes(h5_path)
print(f"Found {len(h5_shapes)} datasets in H5.")

# Now get model shapes without building weights where possible
# Or just build a small part
print("\nFirst 10 H5 datasets:")
for k in list(h5_shapes.keys())[:10]:
    print(f"  {k}: {h5_shapes[k]}")

# Let's count how many match common InceptionResNet shapes
# e.g. (3, 3, 32, 32)
match_3_3_32_32 = sum(1 for s in h5_shapes.values() if s == (3, 3, 32, 32))
print(f"\nCount of (3,3,32,32) weights: {match_3_3_32_32}")
