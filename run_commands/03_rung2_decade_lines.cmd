@echo off
rem Rung 2: a straight line per decade
title 03_rung2_decade_lines
cd /d "%~dp0.."
echo Rung 2: a straight line per decade
echo.
python scripts\run_results.py rung2
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
