# core/decoding.py
# QR decode cascade: tries multiple backends × multiple image preprocessings.
# Order: pyzbar → ZXing → OpenCV  ×  ROI → tilt-corrected full → original full.

import cv2
import numpy as np

from helpers.image import to_gray, apply_clahe, sharpen


# ------------------------------------------------------------------------------
# Image variant generator
# ------------------------------------------------------------------------------

def _get_versions(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return a list of (label, grayscale_variant) to try for decoding."""
    gray     = to_gray(img)
    versions = [("gray",         gray),
                ("inverted",     cv2.bitwise_not(gray))]

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions += [("otsu",         otsu),
                 ("otsu_inv",     cv2.bitwise_not(otsu))]

    ada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    versions += [("adaptive",     ada),
                 ("adaptive_inv", cv2.bitwise_not(ada)),
                 ("clahe",        to_gray(apply_clahe(img))),
                 ("sharp",        to_gray(sharpen(img)))]

    for scale in (2, 3):
        up = cv2.resize(gray,
                        (gray.shape[1] * scale, gray.shape[0] * scale),
                        interpolation=cv2.INTER_LANCZOS4)
        _, up_otsu = cv2.threshold(up, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        up_sharp = to_gray(sharpen(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)))
        versions += [(f"{scale}x",       up),
                     (f"{scale}x_otsu",  up_otsu),
                     (f"{scale}x_sharp", up_sharp)]

    for angle, code in [(90,  cv2.ROTATE_90_CLOCKWISE),
                        (180, cv2.ROTATE_180),
                        (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        versions.append((f"rot{angle}", cv2.rotate(gray, code)))

    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    versions.append(("denoised", denoised))

    brightness = gray.mean()
    if brightness > 180:
        alpha    = 180 / brightness
        darkened = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)
        _, dark_otsu = cv2.threshold(darkened, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions += [("darkened",      darkened),
                     ("darkened_otsu", dark_otsu)]

    gamma     = 0.5
    gamma_lut = np.array([((i / 255.0) ** gamma) * 255
                           for i in range(256)], dtype=np.uint8)
    versions.append(("gamma", cv2.LUT(gray, gamma_lut)))

    return versions


# ------------------------------------------------------------------------------
# Individual backends
# ------------------------------------------------------------------------------

def _decode_pyzbar(gray: np.ndarray) -> bytes | None:
    from pyzbar.pyzbar import decode as pyzbar_decode
    for obj in pyzbar_decode(gray):
        if obj.type == "QRCODE":
            return obj.data
    return None


def _decode_zxing(gray: np.ndarray) -> bytes | None:
    try:
        import zxingcpp
        for r in zxingcpp.read_barcodes(gray):
            if "QR" in str(r.format):
                return r.text.encode("utf-8")
    except Exception:
        pass
    return None


def _decode_opencv(gray: np.ndarray) -> bytes | None:
    try:
        det  = cv2.QRCodeDetector()
        data, _, _ = det.detectAndDecode(gray)
        if data:
            return data.encode("utf-8")
        try:
            det2 = cv2.QRCodeDetectorAruco()
            data2, _, _ = det2.detectAndDecode(gray)
            if data2:
                return data2.encode("utf-8")
        except Exception:
            pass
    except Exception:
        pass
    return None


_BACKENDS = [
    ("pyzbar",    _decode_pyzbar),
    ("ZXing",     _decode_zxing),
    ("OpenCV QR", _decode_opencv),
]


# ------------------------------------------------------------------------------
# Cascade decoder
# ------------------------------------------------------------------------------

def _try_all_backends(
    versions: list[tuple[str, np.ndarray]],
    stage_label: str,
) -> tuple[bytes | None, str | None]:
    """Try every backend against every image variant; return first success."""
    for backend_name, backend_fn in _BACKENDS:
        for ver_name, ver_img in versions:
            try:
                raw = backend_fn(ver_img)
                if raw:
                    return raw, f"{stage_label}+{backend_name}+{ver_name}"
            except Exception:
                pass
    return None, None


def decode_cascade(
    qr_roi:   np.ndarray,
    img_tilt: np.ndarray,
    img_bgr:  np.ndarray,
) -> tuple[bytes | None, str | None]:
    """Attempt QR decoding in order: ROI → tilt-corrected full → original full.

    Returns:
        (raw_bytes, method_label)  — raw_bytes is None on failure.
    """
    for label, base_img in [("ROI",       qr_roi),
                              ("full_tilt", img_tilt),
                              ("full_orig", img_bgr)]:
        raw, method = _try_all_backends(_get_versions(base_img), label)
        if raw:
            return raw, method
    return None, None
