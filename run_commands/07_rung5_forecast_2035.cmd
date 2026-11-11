@echo off
rem Rung 5: backtests and the 2035 forecast with error bar
title 07_rung5_forecast_2035
cd /d "%~dp0.."
echo Rung 5: backtests and the 2035 forecast with error bar
echo.
python scripts\run_results.py forecast
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
