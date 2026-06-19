@echo off
REM ============================================================================
REM  run_dashboard.bat  -  AI Sound Advisor dashboard launcher (hardware-free)
REM
REM  Double-click this file to start the dashboard. It FORCES Python 3.10 via the
REM  "py" launcher, because that is the interpreter the project's packages are
REM  installed into. (Running the .py directly can pick a newer Python -- e.g.
REM  3.14 -- that does not have uvicorn, which fails with "No module named
REM  'uvicorn'".)
REM
REM  Audio is simulated; the X32 Connect / Test Connection panel talks to a real
REM  mixer over the network using the IP/Port in the Settings window.
REM  Dashboard opens at http://127.0.0.1:8001
REM ============================================================================

setlocal

REM --- Always work from this batch file's own folder (handles spaces in path) -
cd /d "%~dp0"

REM --- Make sure Python 3.10 is available before launching --------------------
py -3.10 -c "import sys" 1>nul 2>nul
if errorlevel 1 (
    echo [!] Python 3.10 was not found via the "py" launcher.
    echo     Install Python 3.10, or edit this file to point at your interpreter.
    echo.
    pause
    exit /b 1
)

REM --- Verify the core dependency is present; offer to install if missing -----
py -3.10 -c "import uvicorn" 1>nul 2>nul
if errorlevel 1 (
    echo [!] Required packages are not installed in Python 3.10.
    echo     Installing from requirements.txt ...
    py -3.10 -m pip install -r requirements.txt
    echo.
)

REM --- Free port 8001 if a previous dashboard instance is still holding it ---
REM    (avoids "[Errno 10048] only one usage of each socket address" -- the old
REM     server keeps running until killed even after you close its window.)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "127.0.0.1:8001" ^| findstr "LISTENING"') do (
    echo [i] Port 8001 is in use by PID %%p - stopping the old instance...
    taskkill /PID %%p /F >nul 2>nul
)

echo ============================================================
echo   AI Sound Advisor - Dashboard  (Python 3.10)
echo   Open: http://127.0.0.1:8001
echo   Press Ctrl+C in this window to stop.
echo ============================================================
echo.

py -3.10 run_dashboard.py

REM --- Keep the window open if the server exits / errors ---------------------
echo.
pause
endlocal
