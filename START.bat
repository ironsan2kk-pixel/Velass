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
echo   Dashboard:  http://localhost:8080
echo   Settings:   http://localhost:8080/settings
echo   Strategies: http://localhost:8080/strategies
echo   Backtest:   http://localhost:8080/backtest
echo ═══════════════════════════════════════════════════════
echo.
echo Configure your API keys at: http://localhost:8080/settings
echo.

REM Open browser
start "" http://localhost:8080/settings

REM Start dashboard
python main.py --dashboard

pause
