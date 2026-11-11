@echo off
rem Rung 3: polynomial + harmonics, chosen on 2011-2015
title 04_rung3_polynomial_harmonics
cd /d "%~dp0.."
echo Rung 3: polynomial + harmonics, chosen on 2011-2015
echo.
python scripts\run_results.py rung3
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
