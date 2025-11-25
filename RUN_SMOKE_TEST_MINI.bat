@echo off
echo ================================================================================
echo SMOKE TEST MINI - AGENT 7 V2.1
echo ================================================================================
echo.
echo Ultra-fast test (~1 minute, 100 steps)
echo.
echo Verifies:
echo   [1] Model loads
echo   [2] 3 actions (SELL, HOLD, BUY)
echo   [3] Opens AND closes positions
echo   [4] No mode collapse
echo.
echo ================================================================================

cd /d "%~dp0"

python smoke_test_MINI.py

pause
