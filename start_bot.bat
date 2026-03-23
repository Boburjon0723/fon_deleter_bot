@echo off
chcp 65001 >nul
cd /d "%~dp0"
if "%TELEGRAM_BOT_TOKEN%"=="" (
  echo Avval token qo'ying, masalan PowerShell:
  echo   $env:TELEGRAM_BOT_TOKEN="123456:ABC..."
  echo Keyin qayta ishga tushiring.
  pause
  exit /b 1
)
python telegram_bot.py
