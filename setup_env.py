"""
Mahalliy `.env` faylini yaratish (GitHub ga kirmaydi).

Ishlatish:
  python setup_env.py
  python setup_env.py --token "123456:ABC..."

Railway CLI: https://docs.railway.com/guides/cli
  railway variable set TELEGRAM_BOT_TOKEN=123456:ABC...
  railway variable set BOT_QUALITY=normal
  railway variable set BOT_DEVICE=cpu
  yoki: .\\railway_vars.ps1
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"

VALID_QUALITY = ("draft", "normal", "high", "max")
VALID_DEVICE = ("cpu", "cuda")


def _validate_token(token: str) -> bool:
    token = token.strip()
    if not token or ":" not in token or len(token) < 40:
        return False
    # 123456789:AAH... — taxminiy tekshiruv
    return bool(re.match(r"^\d+:[A-Za-z0-9_-]+$", token))


def write_env(token: str, quality: str, device: str, path: Path) -> None:
    if quality not in VALID_QUALITY:
        raise ValueError(f"BOT_QUALITY: {VALID_QUALITY}")
    if device not in VALID_DEVICE:
        raise ValueError(f"BOT_DEVICE: {VALID_DEVICE}")

    lines = [
        "# setup_env.py orqali yaratilgan — gitga kiritilmaydi",
        f"TELEGRAM_BOT_TOKEN={token.strip()}",
        f"BOT_QUALITY={quality}",
        f"BOT_DEVICE={device}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Yozildi: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="`.env` faylida o‘zgaruvchilarni yozish")
    p.add_argument("--token", default="", help="Telegram bot token (bo‘sh bo‘lsa, yashirin so‘raydi)")
    p.add_argument(
        "--quality",
        default="normal",
        choices=VALID_QUALITY,
        help="Segmentatsiya sifati",
    )
    p.add_argument("--device", default="cpu", choices=VALID_DEVICE, help="cpu yoki cuda")
    p.add_argument(
        "--output",
        type=Path,
        default=ENV_PATH,
        help="Chiqish fayli (standart: loyiha/.env)",
    )
    p.add_argument("--force", action="store_true", help=".env bor bo‘lsa ham qayta yozish")
    args = p.parse_args()

    out: Path = args.output
    if out.exists() and not args.force:
        print(f"Fayl bor: {out}\nQayta yozish uchun: --force", file=sys.stderr)
        return 1

    token = args.token.strip()
    if not token:
        print("Telegram bot token (@BotFather):")
        token = getpass.getpass("Token: ").strip()

    if not _validate_token(token):
        print(
            "Token formati noto‘g‘ri ko‘rinadi (123456789:AAH... kabi bo‘lishi kerak).",
            file=sys.stderr,
        )
        return 1

    try:
        write_env(token, args.quality, args.device, out)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    print("\nKeyingi qadam:")
    print(f"  python telegram_bot.py")
    print("\nRailway (CLI):")
    print(f'  railway variables set TELEGRAM_BOT_TOKEN="{token[:8]}..."')
    print("  (to‘liq tokenni Railway dashboard → Variables dan qo‘ying)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
