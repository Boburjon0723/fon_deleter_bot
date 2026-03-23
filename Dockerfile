# Railway / Docker: Telegram bot + FastSAM (CPU)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/data/cache \
    TORCH_HOME=/data/cache/torch

WORKDIR /app

# OpenCV (cv2) uchun: libxcb.so.1 va bog‘liq X11 kutubxonalari (GUI emas, headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libxcb-xfixes0 \
    libxcb-render0 \
    libxcb-shm0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Build vaqtida tekshiruv (libxcb / libgl xatolari shu yerda chiqadi)
RUN python -c "import cv2; import numpy; print('opencv OK', cv2.__version__)"

COPY segment_products.py telegram_bot.py ./

# /data bo‘sh bo‘lmasa cache shu yerda saqlanadi (Railway Volume tutash)
RUN mkdir -p /data/cache

CMD ["python", "telegram_bot.py"]
