@echo off
rem Write frontend\presentation-data.js
title 11_export_presentation_data
cd /d "%~dp0.."
echo Write frontend\presentation-data.js
echo.
python scripts\export_presentation.py
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
