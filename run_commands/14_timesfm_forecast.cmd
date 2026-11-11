@echo off
rem TimesFM forecast (uses the timesfm_forecast venv)
title 14_timesfm_forecast
cd /d "%~dp0.."
echo TimesFM forecast (uses the timesfm_forecast venv)
echo.
"timesfm_forecast\.venv\Scripts\python.exe" timesfm_forecast\run_forecast.py %*
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
