"""
debug_predictions.py – Debug model predictions to identify label inversion
"""

import os
import numpy as np
from models.model_factory import load_model
from utils.preprocessing import preprocess_for_model
from config import SAVED_MODELS_DIR, MAIN_MODEL, DR_STAGES
from utils.dataset import get_image_paths_labels
from PIL import Image

# Load model
print(f"Loading model: {MAIN_MODEL}")
try:
    model = load_model(MAIN_MODEL)
    print(f"✓ Model loaded successfully")
    print(f"  Model output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Get test images
print("\nScanning for test images...")
paths, labels = get_image_paths_labels()
print(f"Found {len(paths)} images total")

if len(paths) == 0:
    print("✗ No images found!")
    exit(1)

# Group by class
class_images = {i: [] for i in range(5)}
for path, label in zip(paths, labels):
    class_images[label].append(path)

print("\nImages per class:")
for cls_id, stage_name in DR_STAGES.items():
    count = len(class_images[cls_id])
    print(f"  Class {cls_id} ({stage_name}): {count} images")
    
# Test one image from each class
print("\n" + "="*70)
print("TESTING: One image per class")
print("="*70)

for cls_id in range(5):
    if not class_images[cls_id]:
        print(f"\n✗ No images for class {cls_id}")
        continue
    
    img_path = class_images[cls_id][0]
    print(f"\n📁 Class {cls_id} ({DR_STAGES[cls_id]})")
    print(f"   File: {os.path.basename(img_path)}")
    
    try:
        # Preprocess and predict
        img_batch = preprocess_for_model(img_path, MAIN_MODEL)
        print(f"   Batch shape: {img_batch.shape}")
        
        preds = model.predict(img_batch, verbose=0)[0]
        predicted_cls = np.argmax(preds)
        confidence = preds[predicted_cls] * 100
        
        print(f"   ✓ Predictions:")
        for i, score in enumerate(preds):
            marker = "→" if i == predicted_cls else " "
            print(f"     {marker} {DR_STAGES[i]}: {score*100:.2f}%")
        
        if predicted_cls == cls_id:
            print(f"   ✅ CORRECT (predicted class {predicted_cls})")
        else:
            print(f"   ❌ WRONG (predicted class {predicted_cls} instead of {cls_id})")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nIf most predictions are inverted (e.g., No_DR → Proliferative_DR),")
print("the class labels may be reversed in the trained model.")
