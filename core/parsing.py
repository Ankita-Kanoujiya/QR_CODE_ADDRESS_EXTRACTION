# core/parsing.py
# Parsers for both Aadhaar QR formats:
#   - Secure QR  : large integer → zlib-compressed binary blob
#   - Old QR     : XML / PrintLetterBarcodeData

import zlib
import xml.etree.ElementTree as ET

from config import FIELD_MAP, ADDRESS_POSITIONS


# ------------------------------------------------------------------------------
# Secure QR parser (post-2019 Aadhaar cards)
# ------------------------------------------------------------------------------

def parse_secure_qr(raw_bytes: bytes) -> tuple[dict, dict]:
    """Parse the compressed binary Secure QR format.

    The QR payload is a large decimal integer. We convert it to bytes,
    strip leading zero-bytes, and zlib-decompress the result. Fields are
    separated by 0xFF bytes.

    Returns:
        (info_dict, address_dict)
    """
    qr_int       = int(raw_bytes.decode("utf-8").strip())
    byte_data    = qr_int.to_bytes(5000, "big").lstrip(b"\x00")
    decompressed = zlib.decompress(byte_data, wbits=47)
    fields       = decompressed.split(b"\xff")

    info: dict    = {}
    address: dict = {}

    for pos, label in FIELD_MAP.items():
        if pos >= len(fields):
            continue
        try:
            val = fields[pos].decode("utf-8", errors="replace").strip()
            val = "".join(c for c in val if c.isprintable()).strip()
            if val:
                info[label] = val
                if pos in ADDRESS_POSITIONS:
                    address[label] = val
        except Exception:
            pass

    return info, address


# ------------------------------------------------------------------------------
# Old XML QR parser (pre-2019 Aadhaar cards)
# ------------------------------------------------------------------------------

_OLD_FIELD_MAP = {
    "name":    "Name",
    "dob":     "Date of Birth",
    "gender":  "Gender",
    "co":      "House",
    "house":   "House",
    "street":  "Street",
    "lm":      "Landmark",
    "loc":     "Locality",
    "vtc":     "Village/Town/City",
    "po":      "Post Office",
    "subdist": "Sub District",
    "dist":    "District",
    "state":   "State",
    "pc":      "Pincode",
}

_OLD_ADDRESS_KEYS = {
    "co", "house", "street", "lm", "loc", "vtc",
    "po", "subdist", "dist", "state", "pc",
}


def parse_old_qr(raw_bytes: bytes) -> tuple[dict, dict]:
    """Parse the XML-based QR format (PrintLetterBarcodeData).

    Returns:
        (info_dict, address_dict)
    """
    text = raw_bytes.decode("utf-8").strip()
    root = ET.fromstring(text)

    info: dict    = {}
    address: dict = {}

    for xml_key, label in _OLD_FIELD_MAP.items():
        val = root.attrib.get(xml_key, "").strip()
        if val:
            info[label] = val
            if xml_key in _OLD_ADDRESS_KEYS:
                address[label] = val

    return info, address


# ------------------------------------------------------------------------------
# Auto-dispatch
# ------------------------------------------------------------------------------

def parse_qr(raw_bytes: bytes) -> tuple[dict, dict]:
    """Detect QR format and delegate to the correct parser.

    Returns:
        (info_dict, address_dict)
    Raises:
        ValueError on parse failure.
    """
    qr_str = raw_bytes.decode("utf-8", errors="replace").strip()

    if qr_str.startswith("<") or "PrintLetterBarcodeData" in qr_str:
        return parse_old_qr(raw_bytes)

    return parse_secure_qr(raw_bytes)
