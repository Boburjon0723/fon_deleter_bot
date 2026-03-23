"""
Telegram bot: rasm yuboriladi, har bir mahsulot alohida PNG qaytadi.

1) @BotFather dan token oling.
2) Muhit o‘zgaruvchisi: TELEGRAM_BOT_TOKEN
3) Ishga tushirish: python telegram_bot.py

Ixtiyoriy: BOT_QUALITY=normal|high|max, BOT_DEVICE=cpu|cuda
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from functools import partial
from pathlib import Path

import pytz
import telegram._utils.datetime as _ptb_dt

# PTB UTC = datetime.timezone.utc; APScheduler 3.x faqat pytz qabul qiladi (Windows xato)
_ptb_dt.UTC = pytz.UTC

from telegram import InputMediaPhoto, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from segment_products import SegmentResult, get_inference_params, process_image

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Bir marta yuboriladigan rasmlar soni (Telegram media group limiti 10)
MAX_MEDIA_GROUP = 10


def _env_quality() -> str:
    q = os.environ.get("BOT_QUALITY", "normal").lower()
    return q if q in ("draft", "normal", "high", "max") else "normal"


def _env_device() -> str:
    d = os.environ.get("BOT_DEVICE", "cpu").lower()
    return d if d in ("cpu", "cuda") else "cpu"


def _run_segmentation(input_path: Path, work_dir: Path) -> SegmentResult:
    inf = get_inference_params(_env_quality())
    return process_image(
        input_path,
        work_dir,
        conf=inf["conf"],
        iou=inf["iou"],
        imgsz=inf["imgsz"],
        device=_env_device(),
        min_area_ratio=0.002,
        mode="white",
        filter_subsumed=True,
        max_objects=15,
        layout="centered",
        canvas_side=1024,
        pad_ratio=0.06,
        model_key=inf["model_key"],
        refine_mask=bool(inf["refine_mask"]),
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Salom! Rasm yuboring (📎 yoki siqib tushirish).\n"
        "Men har bir mahsulotni alohida oq fonli PNG qilib qaytaraman.\n"
        "Birinchi marta sekin bo‘lishi mumkin (model yuklanadi).\n\n"
        "/help — qisqa yordam"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "• Rasm: foto yoki hujjat sifatida yuboring.\n"
        "• Ko‘p ob’yekt: har biri alohida fayl.\n"
        "• Serverda BOT_QUALITY va BOT_DEVICE sozlash mumkin.\n"
        "• Juda katta rasm yubormang (Telegram cheklovi)."
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return

    if msg.photo:
        tg_file = await msg.photo[-1].get_file()
        suffix = ".jpg"
    elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"):
        tg_file = await msg.document.get_file()
        suf = Path(msg.document.file_name or "image").suffix.lower()
        suffix = suf if suf in (".png", ".jpg", ".jpeg", ".webp", ".bmp") else ".png"
    else:
        await msg.reply_text("Iltimos, rasm yuboring (foto yoki rasm fayli).")
        return

    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)
    status = await msg.reply_text("Rasm qabul qilindi, segmentatsiya boshlandi… (bir necha daqiqa)")

    with tempfile.TemporaryDirectory(prefix="tg_seg_") as tmp:
        tmp_path = Path(tmp)
        in_file = tmp_path / f"in{suffix}"
        await tg_file.download_to_drive(custom_path=str(in_file))

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                partial(_run_segmentation, in_file, tmp_path),
            )
        except Exception as e:
            logger.exception("segmentatsiya xatosi")
            await status.edit_text(f"Xato: {e}")
            return

        if not result.ok:
            await status.edit_text(result.message)
            return

        files = result.saved_files
        await status.edit_text(
            f"Tayyor: {len(files)} ta ob’yekt. Rasmlar yuborilmoqda…"
        )

        # Albomlarda 10 tagacha; qolganlari ketma-ket
        for batch_start in range(0, len(files), MAX_MEDIA_GROUP):
            chunk = files[batch_start : batch_start + MAX_MEDIA_GROUP]
            if len(chunk) == 1:
                with open(chunk[0], "rb") as fh:
                    await msg.reply_photo(photo=fh)
            else:
                media = [
                    InputMediaPhoto(media=str(p), filename=p.name) for p in chunk
                ]
                await msg.reply_media_group(media=media)
            await asyncio.sleep(0.3)

        await status.edit_text(f"Tayyor: {len(files)} ta PNG yuborildi.")


def main() -> None:
    try:
        from dotenv import load_dotenv

        # Ishchi katalog emas, skript yonidagi .env (Git Bash / boshqa cwd uchun)
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN o‘rnatilmagan.\n"
            "Loyiha papkasida `.env` yarating (yoki: python setup_env.py).\n"
            "Git Bash / PowerShell: export TELEGRAM_BOT_TOKEN='...'  yoki  $env:TELEGRAM_BOT_TOKEN='...'\n"
            "Keyin: python telegram_bot.py"
        )

    app = Application.builder().token(token).job_queue(None).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(
        MessageHandler(filters.Document.IMAGE & ~filters.COMMAND, handle_image)
    )

    logger.info("Bot ishga tushmoqda… (Ctrl+C to‘xtatish)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
