@echo off
echo ==============================================
echo      Starting GeoChange AI Pipeline...
echo ==============================================

:: Kill any existing process running on port 8000 to prevent crash
FOR /F "tokens=5 delims= " %%P IN ('netstat -a -n -o ^| findstr :8000') DO TaskKill.exe /F /PID %%P >nul 2>&1

:: Start the FastAPI backend in a separate window
echo Starting FastAPI Backend...
start cmd /k "cd api && python main.py"

:: Give the server 3 seconds to boot up
timeout /t 3 /nobreak >nul

:: Open the Dashboard in the default web browser
echo Launching Dashboard...
start "" "http://localhost:8000/"

echo ==============================================
echo Done! You can close this black window.
