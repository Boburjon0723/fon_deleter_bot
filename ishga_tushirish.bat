@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Grafik interfeys: konsol oynasiz (pythonw / pyw)
where pyw >nul 2>&1
if %errorlevel%==0 (
  start "" pyw -3 "%~dp0gui.py"
  exit /b 0
)

where pythonw >nul 2>&1
if %errorlevel%==0 (
  start "" pythonw "%~dp0gui.py"
  exit /b 0
)

where py >nul 2>&1
if %errorlevel%==0 (
  start "" py -3 "%~dp0gui.py"
  exit /b 0
)

where python >nul 2>&1
if %errorlevel%==0 (
  start "" python "%~dp0gui.py"
  exit /b 0
)

echo.
echo Python topilmadi. Avval o'rnating: https://www.python.org/downloads/
echo Keyin: pip install -r requirements.txt
echo.
pause
exit /b 1
