@echo off
rem All results: rungs 1-5, figures, tables, JSON
title 01_all_results
cd /d "%~dp0.."
echo All results: rungs 1-5, figures, tables, JSON
echo.
python scripts\run_results.py all
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
