# # pipeline.py
import numpy as np

from config   import DB_PATH, MATCH_THRESHOLD, BLUR_THRESHOLD, ADDRESS_ORDER
from core     import load_yolo_model, detect_qr, decode_cascade, parse_qr
from helpers  import correct_tilt, is_blurry, ensure_min_width, build_excel
from matching import match_address


class AadhaarPipeline:
    """
    result keys on success:
        status, personal_info, extracted_address, db_address,
        match_status, excel_bytes, qr_roi, detected_img, detection_method

    result keys on error:
        status, message
        + qr_roi, detected_img, detection_method  (only when blur error)
    """

    def __init__(self):
        self.model = load_yolo_model()

    def run(self, img_bgr: np.ndarray) -> dict:
        # 1. Ensure minimum width
        img_bgr = ensure_min_width(img_bgr)

        # 2. Tilt correction
        img_tilt, _ = correct_tilt(img_bgr.copy())

        # 3. QR detection
        try:
            qr_roi, detected_img, detection_method = detect_qr(img_tilt.copy(), self.model)
        except Exception as e:
            return {"status": "error", "message": f"QR detection failed: {e}"}

        # 4. Blur check
        qr_blurry, qr_blur_score, qr_blur_pct = is_blurry(qr_roi, threshold=BLUR_THRESHOLD)
        if qr_blurry:
            return {
                "status":           "error",
                "message":          "QR code clarity is too low. Cannot extract data. Please upload a clear image.",
                "qr_roi":           qr_roi,
                "detected_img":     detected_img,
                "detection_method": detection_method,
            }

        # 5. Decode QR
        qr_raw, _ = decode_cascade(qr_roi, img_tilt, img_bgr)
        print("qr_raw:", bool(qr_raw), "| detected_img in result will be:", True)

        if not qr_raw:
            return {
                "status":           "error",
                "message":          "QR code clarity is too low. Cannot extract data. Please upload a clear image.",
                "qr_roi":           qr_roi,
                "detected_img":     detected_img,
                "detection_method": detection_method,
            }
        # if not qr_raw:
        #     return {"status": "error", "message": "QR code could not be decoded. Please upload a clearer image."}

        # 6. Parse Aadhaar data
        try:
            info, address = parse_qr(qr_raw)
        except Exception as e:
            return {"status": "error", "message": f"Parse error: {e}"}

        if not info:
            return {"status": "error", "message": "QR decoded but no Aadhaar data found. This may be a sample/demo card."}
        if not address:
            return {"status": "error", "message": "No address fields were extracted from QR code."}

        # 7. Address verification
        try:
            match_result = match_address(
                address_parts=address,
                db_path=DB_PATH,
                threshold=MATCH_THRESHOLD,
                address_col="Address",
            )
        except Exception as e:
            return {"status": "error", "message": f"Address matching error: {e}"}

        best_match     = match_result["matched_row"]
        is_match       = match_result["status"] == "Valid Address"
        extracted_full = ", ".join(address[f] for f in ADDRESS_ORDER if f in address)
        db_addr_raw    = best_match.get("Address", "—") if best_match else "—"
        row_status     = "Match" if is_match else "Mismatch"

        # 8. Build Excel
        excel_bytes = build_excel(
            extracted_full=extracted_full,
            db_addr=db_addr_raw,
            row_status=row_status,
        )

        return {
            "status":            "success",
            "personal_info":     info,
            "extracted_address": extracted_full,
            "db_address":        db_addr_raw,
            "match_status":      row_status,
            "excel_bytes":       excel_bytes,
            "qr_roi":            qr_roi,
            "detected_img":      detected_img,
            "detection_method":  detection_method,
        }