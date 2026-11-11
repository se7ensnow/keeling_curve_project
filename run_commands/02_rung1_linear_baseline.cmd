@echo off
rem Rung 1: one straight line and residual plot
title 02_rung1_linear_baseline
cd /d "%~dp0.."
echo Rung 1: one straight line and residual plot
echo.
python scripts\run_results.py rung1
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
