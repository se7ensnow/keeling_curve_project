@echo off
rem Run all pytest tests
title 12_run_tests
cd /d "%~dp0.."
echo Run all pytest tests
echo.
python -m pytest
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
