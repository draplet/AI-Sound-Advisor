@echo off
REM ============================================================================
REM  run_complete_demo.bat  -  AI Sound Advisor one-click demo launcher
REM
REM  Double-click to bring the whole hardware-free demo to life at once:
REM    1. Ollama        - launches "ollama run qwen2.5:1.5b" in its own window
REM                       so the local AI brain is awake and loaded.
REM    2. AI switches    - sets the LLM env vars the dashboard reads at startup
REM                       (SOUND_ADVISOR_LLM_BACKEND / _LLM_MODEL).
REM    3. Demo server    - runs run_dashboard.py via Uvicorn (fake mixer faders,
REM                       simulated clipping / feedback / vocal-masking) and
REM                       opens the browser dashboard automatically.
REM ============================================================================

setlocal

REM --- Always work from this batch file's own folder (handles spaces in path) -
cd /d "%~dp0"

echo ============================================================
echo   AI Sound Advisor - Complete Demo Launcher
echo ============================================================
echo.

REM --- 1. Start Ollama (local AI brain) in a separate window -----------------
echo [1/3] Waking the local AI brain (ollama run qwen2.5:1.5b)...
start "Ollama - qwen2.5:1.5b" cmd /k "ollama run qwen2.5:1.5b"

REM --- 2. Turn on the AI switches (env vars for the server in THIS window) ----
echo [2/3] Setting AI switches (backend=ollama, model=qwen2.5:1.5b)...
set "SOUND_ADVISOR_LLM_BACKEND=ollama"
set "SOUND_ADVISOR_LLM_MODEL=qwen2.5:1.5b"

REM --- 3. Open the dashboard in the browser once the server has had a moment --
echo [3/3] Starting demo server and opening the dashboard...
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start "" http://127.0.0.1:8000"

echo.
echo Demo server is starting on http://127.0.0.1:8000
echo Close this window (or press Ctrl+C) to stop the demo.
echo ------------------------------------------------------------
echo.

REM --- Launch the hardware-free demo server via Uvicorn ----------------------
REM run_dashboard.py exposes a module-level "app"; it inherits the env vars set
REM above. Reload is intentionally OFF (single deterministic demo process).
python -m uvicorn run_dashboard:app --host 127.0.0.1 --port 8000

endlocal
