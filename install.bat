@echo off
REM VELAS Trading System - Install Dependencies
echo ========================================
echo   VELAS Installation
echo ========================================
echo.

echo Installing Python dependencies...
echo.

pip install pandas numpy pyyaml sqlalchemy aiohttp websockets loguru python-telegram-bot

echo.
echo Optional dependencies:
pip install pyarrow fastapi uvicorn scipy

echo.
echo Installation complete!
echo.
echo Next steps:
echo 1. Copy config/secrets.yaml.example to config/secrets.yaml
echo 2. Add your Telegram bot token and Binance API keys
echo 3. Run: start_bot.bat

pause
