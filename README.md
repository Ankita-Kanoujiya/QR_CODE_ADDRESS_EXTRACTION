# Aadhaar QR Decoder

A Streamlit app to extract and verify information from Aadhaar card QR codes.

## Project Structure

```
aadhaar_qr_decoder/
├── app.py                   # Entry point — Streamlit UI
├── config.py                # All configuration constants
├── core/
│   ├── __init__.py
│   ├── detection.py         # QR detection (YOLOv5 + contour fallback)
│   ├── decoding.py          # QR decode cascade (pyzbar / ZXing / OpenCV)
│   └── parsing.py           # Aadhaar data parsers (secure QR + old XML QR)
├── utils/
│   ├── __init__.py
│   ├── image.py             # Image preprocessing (tilt, CLAHE, sharpen, blur check)
│   └── export.py            # Excel report builder
└── README.md
```

## Setup

```bash
pip install streamlit opencv-python-headless numpy torch pillow pyzbar zxingcpp openpyxl pandas
```

> Place your `best_qr_det_v3.pt` YOLOv5 weights and `aadhaar.db` database at the paths set in `config.py`.

## Run

```bash
streamlit run app.py
```
