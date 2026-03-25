# config.py
# All configurable paths and constants for the Aadhaar QR Decoder.

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
MODEL_PATH = "/app/best_qr_det_v3.pt"
DB_PATH    = "/app/aadhaar.db"

CSV_PATH   = None


# ------------------------------------------------------------------------------
# Matching
# ------------------------------------------------------------------------------
MATCH_THRESHOLD = 0.90

# ------------------------------------------------------------------------------
# Detection
# ------------------------------------------------------------------------------
YOLO_CONF_THRESHOLD = 0.25
YOLO_INPUT_SIZE     = 640
QR_ROI_PADDING      = 10       # pixels of padding around detected QR bounding box

# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------
MIN_IMAGE_WIDTH  = 1500        # Images narrower than this are upscaled
BLUR_THRESHOLD   = 50          # Laplacian variance below this → image is blurry

# ------------------------------------------------------------------------------
# Aadhaar field positions (secure QR format, 0-indexed after split on 0xFF)
# ------------------------------------------------------------------------------
FIELD_MAP: dict[int, str] = {
    3:  "Name",
    4:  "Date of Birth",
    5:  "Gender",
    7:  "District",
    9:  "House",
    10: "Locality",
    11: "Pincode",
    12: "Post Office",
    13: "State",
    14: "Street",
    16: "Village/Town/City",
}

ADDRESS_POSITIONS: set[int] = {7, 9, 10, 11, 12, 13, 14, 16}

# Ordered keys used to build the full address string
ADDRESS_ORDER = [
    "House", "Street", "Locality", "Village/Town/City",
    "Post Office", "Sub District", "District", "State", "Pincode",
]
