@echo off
REM VELAS Trading System - Run Backtest
echo ========================================
echo   VELAS Backtest Runner
echo ========================================
echo.

set /p PAIR="Enter pair (e.g., BTCUSDT): "
set /p STRATEGY="Enter strategy (velas/ema_cross/bollinger/rsi_divergence): "
set /p DAYS="Enter days (default 30): "

if "%DAYS%"=="" set DAYS=30

echo.
echo Running backtest: %PAIR% - %STRATEGY% (%DAYS% days)
echo.

python main.py --backtest --pair %PAIR% --strategy %STRATEGY% --days %DAYS%

pause
