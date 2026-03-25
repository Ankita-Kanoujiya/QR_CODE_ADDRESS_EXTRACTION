FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    libzbar-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Project files ──────────────────────────────────────────────────────────────
COPY config.py     .
COPY app.py        .
COPY matching.py   .
COPY pipeline.py   .

# ── Packages ───────────────────────────────────────────────────────────────────
COPY core/         ./core/
COPY helpers/      ./helpers/

# ── Model & Database ───────────────────────────────────────────────────────────
COPY best_qr_det_v3.pt  .
COPY aadhaar.db        .

EXPOSE 5021

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=5021
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py"]