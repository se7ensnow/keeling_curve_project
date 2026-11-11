@echo off
rem Rung 4: exponential trend, r fitted non-linearly
title 05_rung4_exponential
cd /d "%~dp0.."
echo Rung 4: exponential trend, r fitted non-linearly
echo.
python scripts\run_results.py rung4
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
