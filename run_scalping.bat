@echo off
REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ========================================
echo   High-Probability Scalping Bot
echo   Targeting 80%+ Win Rate
echo ========================================
echo.
echo Working directory: %cd%
echo.

REM Clear Python cache to ensure latest code is used
if exist "bot\__pycache__" (
    echo Clearing Python cache...
    rmdir /s /q "bot\__pycache__"
)

REM Check if config argument provided
if "%1"=="" (
    echo Using default config: configs/scalping.yaml
    set CONFIG=configs/scalping.yaml
) else (
    set CONFIG=%1
)

echo Config: %CONFIG%
echo.
echo Starting bot...
echo Logs will be written to logs\ folder
echo Press Ctrl+C to stop
echo.

.venv\Scripts\python.exe -m bot.scalping_bot --config %CONFIG%

pause
