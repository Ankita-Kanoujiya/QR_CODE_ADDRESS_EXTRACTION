# utils/image.py
# Image preprocessing helpers: tilt correction, CLAHE, sharpening, blur detection.

import cv2
import numpy as np


# ------------------------------------------------------------------------------
# Tilt correction
# ------------------------------------------------------------------------------

def correct_tilt(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Detect and correct skew using Hough line transform.

    Returns:
        (rotated_image, angle_degrees)
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return img, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 90:
                angles.append(a)

    if not angles:
        return img, 0.0

    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return img, angle

    h, w   = img.shape[:2]
    cx, cy = w // 2, h // 2
    M      = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    ca, sa = abs(M[0, 0]), abs(M[0, 1])
    nw     = int(h * sa + w * ca)
    nh     = int(h * ca + w * sa)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy

    rotated = cv2.warpAffine(img, M, (nw, nh),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


# ------------------------------------------------------------------------------
# Enhancement
# ------------------------------------------------------------------------------

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Apply a simple unsharp-mask sharpening kernel."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale; no-op if already single-channel."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


# ------------------------------------------------------------------------------
# Blur detection
# ------------------------------------------------------------------------------

def is_blurry(img: np.ndarray, threshold: int = 50) -> tuple[bool, float, float]:
    """Estimate blur using Laplacian variance.

    Returns:
        (is_blurry, laplacian_score, blur_percentage)
    """
    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    score       = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_pct = min(score / 500.0 * 100, 100.0)
    blur_pct    = round(100.0 - clarity_pct, 1)
    return score < threshold, round(score, 2), blur_pct


# ------------------------------------------------------------------------------
# Upscaling
# ------------------------------------------------------------------------------

def ensure_min_width(img: np.ndarray, min_width: int = 1500) -> np.ndarray:
    """Upscale image if its width is below min_width."""
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        img   = cv2.resize(img,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    return img
