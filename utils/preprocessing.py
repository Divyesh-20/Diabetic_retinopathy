"""
utils/preprocessing.py – Image preprocessing utilities for fundus images
"""

import numpy as np
import cv2
from PIL import Image
from config import IMG_SIZE


def load_image_pil(image_source) -> Image.Image:
    """Load image from file path or uploaded file object."""
    if isinstance(image_source, str):
        return Image.open(image_source).convert("RGB")
    return Image.open(image_source).convert("RGB")


def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_clahe(image_array: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve visibility in fundus images.
    Expects uint8 RGB array. Returns uint8 RGB array.
    """
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_merged = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
    return result


def validate_fundus_image(image_array: np.ndarray) -> tuple:
    """
    Validate whether an image is a genuine retinal fundus image.

    A fundus image has these specific visual characteristics:
      1. Dark circular/oval border (dark background surrounding the retina)
      2. Dominant red/orange colour channel (R > G > B on average)
      3. High saturation — fundus is never grayscale or near-gray
      4. Characteristic circular bright region (the retina disc)
      5. Blood-vessel-like edge structure (moderate edge density)

    Each check is scored independently. Requires at least 3/5 checks to pass.
    This is significantly stricter than the previous 3/4 scheme.

    Returns:
        (is_valid: bool, reason: str)
    """
    if image_array is None or image_array.size == 0:
        return False, "Empty image provided"

    # Ensure uint8
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)

    h, w = image_array.shape[:2]
    total_pixels = h * w

    hsv  = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    checks_passed = []
    check_details = []

    # ── Check 1: Dark circular border (characteristic of fundus) ──────────────
    # Real fundus images have a very dark border making up >10% of pixels
    very_dark_mask = gray < 30
    dark_border_ratio = np.sum(very_dark_mask) / total_pixels
    if dark_border_ratio >= 0.08:
        checks_passed.append(1)
        check_details.append(f"✓ Dark border: {dark_border_ratio:.1%}")
    else:
        check_details.append(f"✗ Dark border too low: {dark_border_ratio:.1%} (need ≥8%)")

    # ── Check 2: Red/Orange channel dominance ──────────────────────────────────
    # Fundus images have dominant red channel: R > G > B
    r_mean = float(image_array[:, :, 0].mean())
    g_mean = float(image_array[:, :, 1].mean())
    b_mean = float(image_array[:, :, 2].mean())

    # Also check for warm hue in non-dark pixels
    non_dark = ~very_dark_mask
    red_lower = np.array([0, 40, 40])
    red_upper = np.array([25, 255, 255])
    red_mask  = cv2.inRange(hsv, red_lower, red_upper)

    # Wrap-around: also catch reds at high hue (355–180 deg)
    red_upper2 = np.array([180, 255, 255])
    red_lower2 = np.array([165, 40, 40])
    red_mask2  = cv2.inRange(hsv, red_lower2, red_upper2)
    combined_red_mask = cv2.bitwise_or(red_mask, red_mask2)

    non_dark_total = np.sum(non_dark)
    if non_dark_total > 0:
        red_in_nondark = np.sum((combined_red_mask > 0) & non_dark) / non_dark_total
    else:
        red_in_nondark = 0.0

    if (r_mean > g_mean) and (red_in_nondark >= 0.15):
        checks_passed.append(2)
        check_details.append(f"✓ Red dominance: R={r_mean:.0f} > G={g_mean:.0f}, warm ratio={red_in_nondark:.1%}")
    else:
        check_details.append(f"✗ Insufficient red: R={r_mean:.0f} G={g_mean:.0f} B={b_mean:.0f}, warm={red_in_nondark:.1%} (need ≥15%)")

    # ── Check 3: Saturation (not grayscale / near-gray) ───────────────────────
    # Fundus images have rich colour — saturation channel is high in non-dark region
    sat_channel = hsv[:, :, 1]   # 0-255 saturation
    if non_dark_total > 0:
        mean_saturation = float(np.mean(sat_channel[non_dark]))
    else:
        mean_saturation = 0.0

    if mean_saturation >= 60:   # saturation > ~24% of 255
        checks_passed.append(3)
        check_details.append(f"✓ Saturation OK: {mean_saturation:.1f}")
    else:
        check_details.append(f"✗ Low saturation (grayscale-like): {mean_saturation:.1f} (need ≥60)")

    # ── Check 4: Edge density (blood vessels) ─────────────────────────────────
    # Fundus has fine blood vessel edges. Too few = solid color. Too many = noise.
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(np.sum(edges > 0)) / total_pixels

    if 0.02 <= edge_density <= 0.35:
        checks_passed.append(4)
        check_details.append(f"✓ Edge density: {edge_density:.2%}")
    else:
        check_details.append(f"✗ Edge density out of range: {edge_density:.2%} (need 2%–35%)")

    # ── Check 5: Circular bright region (retina disc shape) ───────────────────
    # The bright region of a fundus image (retina) should be roughly circular/oval.
    # We check if the bright region has aspect ratio close to 1 and is convex.
    bright_mask = (gray > 40).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright_closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bright_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circular_ok = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > total_pixels * 0.15:   # bright region must be > 15% of image
            x, y, cw, ch = cv2.boundingRect(largest)
            aspect_ratio = min(cw, ch) / max(cw, ch + 1e-5)
            hull_area = cv2.contourArea(cv2.convexHull(largest))
            solidity = area / (hull_area + 1e-5)
            # Good fundus: aspect ratio 0.6–1.0, high solidity (compact shape)
            if aspect_ratio >= 0.55 and solidity >= 0.70:
                circular_ok = True

    if circular_ok:
        checks_passed.append(5)
        check_details.append(f"✓ Circular retina region detected")
    else:
        check_details.append(f"✗ No clear circular retina region found")

    # ── Final decision ────────────────────────────────────────────────────────
    # MANDATORY: checks 1 (dark border) AND 2 (red dominance) MUST both pass.
    # These two together are what uniquely identify a fundus image.
    # Then at least 1 of the remaining 3 must also pass.
    num_passed     = len(checks_passed)
    mandatory_pass = (1 in checks_passed) and (2 in checks_passed)
    optional_pass  = len([c for c in checks_passed if c in {3, 4, 5}]) >= 1
    is_valid = mandatory_pass and optional_pass

    if is_valid:
        return True, "Valid fundus image"
    else:
        failed = [d for d in check_details if d.startswith("✗")]
        summary = f"Only {num_passed}/5 checks passed. Failed: " + " | ".join(failed)
        return False, summary


def preprocess_image(image_source, apply_enhancement: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline:
      1. Load image
      2. Resize to IMG_SIZE
      3. Optional CLAHE enhancement
      4. Normalize to [0, 1]
    Returns: float32 array of shape (H, W, 3)
    """
    img = load_image_pil(image_source)
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)

    if apply_enhancement:
        arr = apply_clahe(arr)

    return arr.astype(np.float32) / 255.0


def preprocess_for_model(image_source, model_name: str = "") -> np.ndarray:
    """
    Returns a batch-ready array.
    For LSTM-based models: shape (1, 1, H, W, 3) — time-distributed single frame
    For plain CNN models:  shape (1, H, W, 3)
    """
    arr = preprocess_image(image_source)   # (H, W, 3) float32 [0,1] with CLAHE
    lstm_models = {"cnn_lstm", "inception_resnet_lstm", "mobilenet_lstm"}
    if model_name in lstm_models:
        return arr[np.newaxis, np.newaxis, ...]   # (1, 1, H, W, 3)
    return arr[np.newaxis, ...]                   # (1, H, W, 3)


def augment_image(image_array: np.ndarray) -> np.ndarray:
    """
    Basic augmentation for training diversity.
    Expects array in [0,1] range, shape (H, W, 3).
    """
    arr = (image_array * 255).astype(np.uint8)
    # Random horizontal flip
    if np.random.rand() > 0.5:
        arr = cv2.flip(arr, 1)
    # Random vertical flip
    if np.random.rand() > 0.5:
        arr = cv2.flip(arr, 0)
    # Random brightness adjustment
    factor = np.random.uniform(0.8, 1.2)
    arr = np.clip(arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return arr.astype(np.float32) / 255.0
