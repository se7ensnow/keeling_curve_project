@echo off
rem Rung 5: models fitted on 1958-1985 only
title 08_rung5_failure_case
cd /d "%~dp0.."
echo Rung 5: models fitted on 1958-1985 only
echo.
python scripts\run_results.py failure
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
