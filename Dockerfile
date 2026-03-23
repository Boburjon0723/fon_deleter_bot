# Railway / Docker: Telegram bot + FastSAM (CPU)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/data/cache \
    TORCH_HOME=/data/cache/torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY segment_products.py telegram_bot.py ./

# /data bo‘sh bo‘lmasa cache shu yerda saqlanadi (Railway Volume tutash)
RUN mkdir -p /data/cache

CMD ["python", "telegram_bot.py"]
