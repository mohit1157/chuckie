@echo off
REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ========================================
echo   CHUCKIE - Forex Scalping Bot
echo ========================================
echo.
echo Select trading mode:
echo.
echo   [1] Regular   - 5%% risk per trade, 10%% daily max loss
echo   [2] Safe      - 2%% risk per trade, 5%% daily max loss (recommended for overnight)
echo.

set /p MODE="Enter choice (1 or 2): "

if "%MODE%"=="1" (
    set CONFIG=configs/scalping.yaml
    set MODE_NAME=REGULAR
    echo.
    echo Selected: REGULAR mode
    echo   - Risk per trade: 5%%
    echo   - Max daily loss: 10%%
) else if "%MODE%"=="2" (
    set CONFIG=configs/scalping_overnight.yaml
    set MODE_NAME=SAFE
    echo.
    echo Selected: SAFE mode
    echo   - Risk per trade: 2%%
    echo   - Max daily loss: 5%%
) else (
    echo Invalid choice. Using SAFE mode by default.
    set CONFIG=configs/scalping_overnight.yaml
    set MODE_NAME=SAFE
)

echo.
echo ========================================
echo   Mode: %MODE_NAME%
echo   Config: %CONFIG%
echo ========================================
echo.

REM Clear Python cache to ensure latest code is used
if exist "bot\__pycache__" (
    echo Clearing Python cache...
    rmdir /s /q "bot\__pycache__"
)

echo Working directory: %cd%
echo.
echo Starting bot...
echo Logs will be written to logs\ folder
echo Press Ctrl+C to stop
echo.

.venv\Scripts\python.exe -m bot.scalping_bot --config %CONFIG%

echo.
echo Bot stopped. Press any key to close...
pause >nul
