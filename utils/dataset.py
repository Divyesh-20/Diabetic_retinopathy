"""
utils/dataset.py – Dataset loading and splitting utilities
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_CLASSES


def get_image_paths_labels(data_dir: str = DATA_DIR):
    """
    Scans data_dir expecting structures like:
    dataset/colored_images/train/ [No_DR, Mild, Moderate, Severe, Proliferate_DR]
    dataset/colored_images/validation/ [No_DR, Mild, Moderate, Severe, Proliferate_DR]
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
    seen_filenames = set()
    
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
                # Try case-insensitive fallback just in case
                for item in os.listdir(s_dir):
                    if item.lower() == folder_name.lower() and os.path.isdir(os.path.join(s_dir, item)):
                        class_dir = os.path.join(s_dir, item)
                        break
            
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                        if fname not in seen_filenames:
                            paths.append(os.path.join(class_dir, fname))
                            labels.append(class_id)
                            seen_filenames.add(fname)
                        
    return paths, labels


def split_dataset(paths, labels, val_size=0.15, test_size=0.15, seed=42):
    """Split into train/val/test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(val_size + test_size), random_state=seed, stratify=labels
    )
    ratio = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=ratio, random_state=seed, stratify=y_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def make_tf_dataset(paths, labels, batch_size=BATCH_SIZE, augment=False, lstm=False):
    """
    Build a tf.data.Dataset from file paths and labels.
    lstm=True → output shape (1, H, W, 3) time-distributed
    """
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
        if lstm:
            img = tf.expand_dims(img, axis=0)   # (1, H, W, 3)
        label = tf.one_hot(label, NUM_CLASSES)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
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
    """Return dict with count per class based on new folder structure."""
    stats = {i: 0 for i in range(NUM_CLASSES)}
    total = 0
    
    paths, labels = get_image_paths_labels(data_dir)
    for lbl in labels:
        stats[lbl] += 1
        total += 1
        
    stats_dict = {k: v for k, v in stats.items()}
    stats_dict["total"] = total
    return stats_dict
