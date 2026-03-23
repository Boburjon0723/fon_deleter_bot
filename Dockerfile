# Railway: CPU-only PyTorch (kichikroq, tezroq yuklash) + bot
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    XDG_CACHE_HOME=/data/cache \
    TORCH_HOME=/data/cache/torch

WORKDIR /app

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

# 1) PyTorch CPU — PyPI dagi to‘liq CUDA build emas (Railway build tezroq)
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
# 2) Qolgan paketlar (ultralytics allaqachon o‘rnatilgan torch dan foydalanadi)
RUN pip install -r requirements-docker.txt

RUN python -c "import torch; import cv2; print('OK torch', torch.__version__, 'cv2', __import__('cv2').__version__)"

COPY segment_products.py telegram_bot.py ./
RUN mkdir -p /data/cache

CMD ["python", "telegram_bot.py"]
