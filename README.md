# fon_deleter_bot

Telegram orqali rasm yuboriladi, har bir mahsulot alohida oq fonli PNG qilib qaytadi (FastSAM + `segment_products.py`).

## Repository

https://github.com/Boburjon0723/fon_deleter_bot

## Maxfiylik

- **`.env` fayl gitga kiritilmaydi** (`.gitignore`).
- Token va maxfiy ma’lumotlarni faqat **Railway Variables** yoki mahalliy `.env` da saqlang.

## Railway deploy

1. Reponi GitHub ga push qiling.
2. [Railway](https://railway.app) → **New Project** → **Deploy from GitHub** → `fon_deleter_bot` ni tanlang.
3. **Variables** (Environment) qo‘shing:

| O‘zgaruvchi | Majburiy | Tavsif |
|-------------|----------|--------|
| `TELEGRAM_BOT_TOKEN` | Ha | @BotFather dan olingan token |
| `BOT_QUALITY` | Yo‘q | `draft` · `normal` · `high` · `max` (standart: `normal`) |
| `BOT_DEVICE` | Yo‘q | `cpu` yoki `cuda` (Railway odatda `cpu`) |

4. **Volumes** (ixtiyoriy): mount path `/data` — modellarni qayta yuklamaslik uchun.
5. Deploy tugagach, bot Telegramda javob berishini tekshiring.

Docker image `Dockerfile` orqali yig‘iladi; start: `python telegram_bot.py`.

## Mahalliy ishga tushirish

```bash
pip install -r requirements.txt
copy .env.example .env
# .env ichiga TELEGRAM_BOT_TOKEN yozing
python telegram_bot.py
```

## Loyiha fayllari

- `segment_products.py` — segmentatsiya
- `telegram_bot.py` — Telegram bot
- `gui.py` — grafik interfeys (ixtiyoriy)
