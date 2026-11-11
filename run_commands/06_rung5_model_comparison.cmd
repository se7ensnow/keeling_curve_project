@echo off
rem Rung 5: held-out table, errors by year, 1 ppm horizon
title 06_rung5_model_comparison
cd /d "%~dp0.."
echo Rung 5: held-out table, errors by year, 1 ppm horizon
echo.
python scripts\run_results.py compare
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
