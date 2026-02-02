@echo off
echo ========================================
echo   OVERNIGHT SAFE Scalping Bot
echo   Max Loss: 5%% | Risk per Trade: 2%%
echo ========================================
echo.

REM Clear Python cache to ensure latest code is used
if exist "bot\__pycache__" (
    echo Clearing Python cache...
    rmdir /s /q "bot\__pycache__"
)

set CONFIG=configs/scalping_overnight.yaml

echo Config: %CONFIG%
echo.
echo SAFETY LIMITS:
echo   - Max daily loss: 5%% (bot stops trading)
echo   - Risk per trade: 2%% (smaller positions)
echo   - Same strategy, just safer sizing
echo.
echo Starting bot...
echo Logs will be written to logs\ folder
echo Press Ctrl+C to stop
echo.

.venv\Scripts\python.exe -m bot.scalping_bot --config %CONFIG%

pause
