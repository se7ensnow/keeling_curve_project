@echo off
rem Tool: trend/season split, growth rate now, 2035 forecast
title 09_tool_decomposition_forecast
cd /d "%~dp0.."
echo Tool: trend/season split, growth rate now, 2035 forecast
echo.
python scripts\run_tool.py %*
echo.
if errorlevel 1 (echo FAILED with exit code %errorlevel%) else (echo Done.)
pause
