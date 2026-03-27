import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pipeline import AadhaarPipeline

# ------------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Aadhaar QR Decoder", layout="wide")


VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

# ------------------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ------------------------------------------------------------------------------
# Login page
# ------------------------------------------------------------------------------
if not st.session_state.authenticated:
    st.title("Login")
    username = st.text_input("Login ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Login ID or Password.")
    st.stop()

# ------------------------------------------------------------------------------
# Main app (only reached after login)
# ------------------------------------------------------------------------------
st.title("Aadhaar QR Code Decoder")
st.write("Upload your Aadhaar card image to extract information from the QR code.")

# ------------------------------------------------------------------------------
# Load pipeline once
# ------------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return AadhaarPipeline()

pipeline = get_pipeline()

# ------------------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Aadhaar Card Image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear photo of your Aadhaar card. QR code must be visible and sharp.",
)

if uploaded_file is None:
    st.stop()

# ------------------------------------------------------------------------------
# Load image
# ------------------------------------------------------------------------------
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img_bgr is None:
    img_bgr = cv2.cvtColor(
        np.array(Image.open(uploaded_file).convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )

# ------------------------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------------------------
with st.spinner("Processing image..."):
    result = pipeline.run(img_bgr)

# ------------------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------------------
if result["status"] == "error":
    print("error keys:", result.keys())
    if "detected_img" in result:
        st.subheader("QR Detection")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                     caption="Original Image", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(result["detected_img"], cv2.COLOR_BGR2RGB),
                     caption=f"Detected — {result['detection_method']}", use_container_width=True)
        with col3:
            qr_display = cv2.copyMakeBorder(
                result["qr_roi"].copy(), 5, 5, 5, 5,
                cv2.BORDER_CONSTANT, value=(0, 255, 0),
            )
            st.image(cv2.cvtColor(qr_display, cv2.COLOR_BGR2RGB),
                     caption="Cropped QR Region", use_container_width=True)
        st.markdown("---")
        st.error(result["message"])
    else:
        st.error(result["message"])
    st.stop()

# ------------------------------------------------------------------------------
# Unpack result
# ------------------------------------------------------------------------------
extracted_full   = result["extracted_address"]
db_addr_raw      = result["db_address"]
row_status_plain = result["match_status"]
is_match         = row_status_plain == "Match"
row_status_emoji = "✅ Match" if is_match else "❌ Mismatch"

# ------------------------------------------------------------------------------
# QR Detection images
# ------------------------------------------------------------------------------
st.subheader("QR Detection")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
             caption="Original Image", use_container_width=True)
with col2:
    st.image(cv2.cvtColor(result["detected_img"], cv2.COLOR_BGR2RGB),
             caption=f"Detected — {result['detection_method']}", use_container_width=True)
with col3:
    qr_display = cv2.copyMakeBorder(
        result["qr_roi"].copy(), 5, 5, 5, 5,
        cv2.BORDER_CONSTANT, value=(0, 255, 0),
    )
    st.image(cv2.cvtColor(qr_display, cv2.COLOR_BGR2RGB),
             caption="Cropped QR Region", use_container_width=True)

# ------------------------------------------------------------------------------
# Address comparison table
# ------------------------------------------------------------------------------
st.markdown("#### Address Comparison")

status_color  = "#C6EFCE" if is_match else "#FFC7CE"
status_fcolor = "#276221" if is_match else "#9C0006"

table_html = f"""
<style>
.addr-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    font-family: sans-serif;
    margin-top: 8px;
}}
.addr-table th {{
    background-color: #2C3E50;
    color: white;
    padding: 12px 16px;
    text-align: left;
    font-size: 15px;
}}
.addr-table td {{
    padding: 14px 16px;
    border: 1px solid #ddd;
    vertical-align: top;
    word-wrap: break-word;
    white-space: pre-wrap;
    line-height: 1.6;
}}
.addr-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
.status-cell {{
    background-color: {status_color};
    color: {status_fcolor};
    font-weight: bold;
    font-size: 15px;
    text-align: center;
    white-space: nowrap;
}}
</style>
<table class="addr-table">
  <thead>
    <tr>
      <th style="width:42%">Extracted Address</th>
      <th style="width:42%">DB Address</th>
      <th style="width:16%">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>{extracted_full}</td>
      <td>{db_addr_raw}</td>
      <td class="status-cell">{row_status_emoji}</td>
    </tr>
  </tbody>
</table>
"""
st.markdown(table_html, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Excel download
# ------------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Download Report")

st.download_button(
    label     = "⬇️ Download Address Comparison as Excel",
    data      = result["excel_bytes"],
    file_name = "aadhaar_address_comparison.xlsx",
    mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)