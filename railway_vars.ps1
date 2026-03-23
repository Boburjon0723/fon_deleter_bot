# Railway CLI orqali o'zgaruvchilar (PowerShell)
# Hujjat: https://docs.railway.com/guides/cli — `railway variable set KEY=value`
#
# 1) npm i -g @railway/cli
# 2) railway login
# 3) Loyiha papkasida: railway link
# 4) .\railway_vars.ps1

$ErrorActionPreference = "Stop"

if (-not (Get-Command railway -ErrorAction SilentlyContinue)) {
    Write-Host "Railway CLI yo'q. O'rnating: npm i -g @railway/cli" -ForegroundColor Yellow
    exit 1
}

if (-not $env:TELEGRAM_BOT_TOKEN) {
    $secure = Read-Host "TELEGRAM_BOT_TOKEN" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    $env:TELEGRAM_BOT_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
}

if (-not $env:TELEGRAM_BOT_TOKEN) {
    Write-Host "Token kiritilmadi." -ForegroundColor Red
    exit 1
}

if (-not $env:BOT_QUALITY) { $env:BOT_QUALITY = "normal" }
if (-not $env:BOT_DEVICE) { $env:BOT_DEVICE = "cpu" }

railway variable set "TELEGRAM_BOT_TOKEN=$($env:TELEGRAM_BOT_TOKEN)"
railway variable set "BOT_QUALITY=$($env:BOT_QUALITY)"
railway variable set "BOT_DEVICE=$($env:BOT_DEVICE)"

Write-Host "Tayyor. railway variable list bilan tekshiring." -ForegroundColor Green
