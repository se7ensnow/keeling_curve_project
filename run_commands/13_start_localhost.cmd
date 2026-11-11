@echo off
rem Local web server for the explorer and the presentation page
title 13_start_localhost
cd /d "%~dp0.."
set PORT=8000
echo Explorer:      http://localhost:%PORT%/
echo Presentation:  http://localhost:%PORT%/presentation.html
echo Close this window (or press Ctrl+C) to stop the server.
echo.
start "" cmd /c "timeout /t 2 /nobreak >nul & start "" http://localhost:%PORT%/ & start "" http://localhost:%PORT%/presentation.html"
python -m http.server %PORT% --directory frontend
pause
