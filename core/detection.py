# core/detection.py
# QR code region detection using YOLOv5 (primary) and contour analysis (fallback).

import cv2
import numpy as np
import streamlit as st
import torch

from config import MODEL_PATH, YOLO_CONF_THRESHOLD, YOLO_INPUT_SIZE, QR_ROI_PADDING


# ------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------

@st.cache_resource
def load_yolo_model() -> torch.nn.Module:
    """Load and cache the custom YOLOv5 QR detection model."""
    import os
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: '{MODEL_PATH}'")
        st.stop()
    model = torch.hub.load("ultralytics/yolov5", "custom",
                           path=MODEL_PATH, force_reload=False)
    model.eval()
    return model


# ------------------------------------------------------------------------------
# YOLOv5 detection
# ------------------------------------------------------------------------------

def detect_qr_yolo(
    img: np.ndarray,
    model: torch.nn.Module,
    conf_threshold: float = YOLO_CONF_THRESHOLD,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """Detect the QR region using the YOLOv5 model.

    Returns:
        (qr_roi, annotated_image, best_confidence)
        qr_roi is None when no detection exceeds conf_threshold.
    """
    img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results    = model(img_rgb, size=YOLO_INPUT_SIZE)
    detections = results.xyxy[0].tolist()

    box_img   = img.copy()
    qr_roi    = None
    best_conf = 0.0

    for *xyxy, conf, cls in detections:
        conf = float(conf)
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, xyxy)

        if conf > best_conf:
            best_conf = conf
            h, w      = img.shape[:2]
            x1p = max(0,     x1 - QR_ROI_PADDING)
            y1p = max(0,     y1 - QR_ROI_PADDING)
            x2p = min(w,     x2 + QR_ROI_PADDING)
            y2p = min(h,     y2 + QR_ROI_PADDING)
            qr_roi = img[y1p:y2p, x1p:x2p]

        cv2.rectangle(box_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(box_img, f"QR {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return qr_roi, box_img, best_conf


# ------------------------------------------------------------------------------
# Contour-based fallback detection
# ------------------------------------------------------------------------------

def detect_qr_contour(
    img: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Fallback QR detection using contour analysis (no model required).

    Looks for large, roughly-square contours that are likely QR codes.

    Returns:
        (qr_roi, annotated_image)
        qr_roi is None if nothing suitable is found.
    """
    original     = img.copy()
    img_h, img_w = img.shape[:2]
    gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur   = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    qr_roi    = None
    best_area = 0
    box_img   = img.copy()
    max_area  = 0.85 * img_w * img_h

    for c in cnts:
        peri       = cv2.arcLength(c, True)
        approx     = cv2.approxPolyDP(c, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area       = cv2.contourArea(c)
        ar         = w / float(h)

        if len(approx) == 4 and area > 1000 and area < max_area and 0.7 < ar < 1.4:
            if area > best_area:
                best_area = area
                qr_roi    = original[y:y + h, x:x + w]
                cv2.rectangle(box_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(box_img, "QR (contour)",
                            (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return qr_roi, box_img


# ------------------------------------------------------------------------------
# Unified detection (YOLO → contour → full image)
# ------------------------------------------------------------------------------

def detect_qr(
    img_tilt: np.ndarray,
    model: torch.nn.Module,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run YOLO detection, fall back to contour, then fall back to full image.

    Returns:
        (qr_roi, annotated_image, detection_method_label)
    """
    qr_roi, detected_img, _ = detect_qr_yolo(img_tilt.copy(), model)
    if qr_roi is not None:
        return qr_roi, detected_img, "YOLOv5"

    qr_roi, detected_img = detect_qr_contour(img_tilt.copy())
    if qr_roi is not None:
        return qr_roi, detected_img, "Contour"

    return img_tilt, img_tilt.copy(), "Full image"
