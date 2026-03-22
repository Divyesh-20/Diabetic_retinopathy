"""
utils/dataset.py – Dataset loading and splitting utilities
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_CLASSES


def _apply_clahe_tf(img_tensor):
    """Apply CLAHE enhancement to a normalized float32 tensor. Returns float32 in [0,1]."""
    # Convert float tensor to uint8 numpy, apply CLAHE, return float tensor
    def _clahe_np(img_np):
        img_np = img_np.numpy()              # EagerTensor → numpy array
        img_uint8 = (img_np * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_merged = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
        return result.astype(np.float32) / 255.0

    return tf.py_function(func=_clahe_np, inp=[img_tensor], Tout=tf.float32)


def get_image_paths_labels(data_dir: str = DATA_DIR):
    """
    Scans data_dir expecting structures like:
    dataset/colored_images/train/ [No_DR, Mild, Moderate, Severe, Proliferate_DR]
    dataset/colored_images/validation/ [No_DR, Mild, Moderate, Severe, Proliferate_DR]
    
    Deduplication is by FULL PATH to prevent silent data loss from same filenames
    in different subdirectories.
    """
    # Mapping of user's folder names to official stage IDs
    folder_to_class = {
        "No_DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferate_DR": 4
    }

    paths, labels = [], []
    seen_paths = set()   # ← FIXED: deduplicate by full path, not filename

    # 1. Look for root level train, validation, test
    search_dirs = []
    for sub in ["train", "validation", "test"]:
        sub_d = os.path.join(data_dir, sub)
        if os.path.isdir(sub_d):
            search_dirs.append(sub_d)

    # 2. If nothing found, check inside colored_images
    if not search_dirs:
        colored_dir = os.path.join(data_dir, "colored_images")
        if os.path.isdir(colored_dir):
            for sub in ["train", "validation", "test"]:
                sub_d = os.path.join(colored_dir, sub)
                if os.path.isdir(sub_d):
                    search_dirs.append(sub_d)
            if not search_dirs:
                search_dirs = [colored_dir]

    # 3. Ultimate fallback
    if not search_dirs:
        search_dirs = [data_dir]

    for s_dir in search_dirs:
        for folder_name, class_id in folder_to_class.items():
            class_dir = os.path.join(s_dir, folder_name)
            if not os.path.isdir(class_dir):
                # Try case-insensitive fallback
                for item in os.listdir(s_dir):
                    if item.lower() == folder_name.lower() and os.path.isdir(os.path.join(s_dir, item)):
                        class_dir = os.path.join(s_dir, item)
                        break

            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                        full_path = os.path.normpath(os.path.join(class_dir, fname))
                        if full_path not in seen_paths:    # ← FIXED: use full path
                            paths.append(full_path)
                            labels.append(class_id)
                            seen_paths.add(full_path)

    return paths, labels


def split_dataset(paths, labels, val_size=0.15, test_size=0.15, seed=42):
    """Split into train/val/test with stratification."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(val_size + test_size), random_state=seed, stratify=labels
    )
    ratio = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=ratio, random_state=seed, stratify=y_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def make_tf_dataset(paths, labels, batch_size=BATCH_SIZE, augment=False,
                    lstm=False, apply_clahe=True):
    """
    Build a tf.data.Dataset from file paths and labels.
    lstm=True    → output shape (1, H, W, 3) time-distributed
    apply_clahe  → apply CLAHE enhancement (MUST match inference-time preprocessing)
    """
    SHUFFLE_BUFFER = 1000

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0

        if apply_clahe:
            img = _apply_clahe_tf(img)
            img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])

        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.15)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)

        if lstm:
            img = tf.expand_dims(img, axis=0)   # (1, H, W, 3)

        label = tf.one_hot(label, NUM_CLASSES)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if augment:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_class_weights(labels):
    """Compute inverse-frequency class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


def get_dataset_stats(data_dir: str = DATA_DIR):
    """Return dict with count per class based on folder structure."""
    stats = {i: 0 for i in range(NUM_CLASSES)}
    total = 0

    paths, labels = get_image_paths_labels(data_dir)
    for lbl in labels:
        stats[lbl] += 1
        total += 1

    stats_dict = {k: v for k, v in stats.items()}
    stats_dict["total"] = total
    return stats_dict
