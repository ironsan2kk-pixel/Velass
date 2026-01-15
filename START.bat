@echo off
chcp 65001 >nul
title VELAS Trading System

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║         VELAS TRADING SYSTEM v2.0                     ║
echo ║         Strategy Bot + Web Admin                      ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.11+
    pause
    exit /b 1
)

echo Starting Web Dashboard...
echo.
echo ═══════════════════════════════════════════════════════
echo   Dashboard:  http://localhost:5000
echo   Settings:   http://localhost:5000/settings
echo   Strategies: http://localhost:5000/strategies
echo   Backtest:   http://localhost:5000/backtest
echo ═══════════════════════════════════════════════════════
echo.
echo Configure your API keys at: http://localhost:5000/settings
echo.

REM Open browser
start "" http://localhost:5000/settings

REM Start dashboard on port 5000
python -c "from monitor.dashboard import Dashboard; d = Dashboard(port=5000); d.run()"

pause
