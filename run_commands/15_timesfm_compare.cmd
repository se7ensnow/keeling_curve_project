@echo off
rem TimesFM vs our models (uses the timesfm_forecast venv)
title 15_timesfm_compare
cd /d "%~dp0.."
echo TimesFM vs our models (uses the timesfm_forecast venv)
echo.
"timesfm_forecast\.venv\Scripts\python.exe" timesfm_forecast\compare.py %*
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
