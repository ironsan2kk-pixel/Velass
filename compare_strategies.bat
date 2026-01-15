@echo off
REM VELAS Trading System - Compare Strategies
echo ========================================
echo   VELAS Strategy Comparison
echo ========================================
echo.

set /p PAIR="Enter pair (e.g., BTCUSDT): "
set /p DAYS="Enter days (default 30): "

if "%DAYS%"=="" set DAYS=30

echo.
echo Comparing all strategies on %PAIR% (%DAYS% days)
echo.

python main.py --compare velas,ema_cross,bollinger,rsi_divergence --pair %PAIR% --days %DAYS%

pause
