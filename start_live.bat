@echo off
REM VELAS Trading System - Start Live Trading
echo ========================================
echo   VELAS Live Trading
echo ========================================
echo.

echo WARNING: This will start live trading!
echo Make sure your config/secrets.yaml is configured.
echo.

set /p CONFIRM="Start in DRY RUN mode? (y/n): "

if /i "%CONFIRM%"=="y" (
    echo Starting in DRY RUN mode...
    python main.py --dry-run
) else (
    echo Starting LIVE trading...
    python main.py
)

pause
