@echo off
rem Write frontend\data.js and results\params JSON
title 10_export_explorer_data
cd /d "%~dp0.."
echo Write frontend\data.js and results\params JSON
echo.
python scripts\export_frontend.py
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
