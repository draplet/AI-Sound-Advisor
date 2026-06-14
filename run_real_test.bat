@echo off
REM ============================================================================
REM  run_real_test.bat  -  AI Sound Advisor REAL-HARDWARE launcher
REM
REM  Like run_complete_demo.bat, but instead of the fake demo it runs the LIVE
REM  app against real hardware:
REM    - real audio capture (SoundDeviceAudioSource)
REM    - real Behringer X32 over OSC (UdpOscTransport)
REM    - real local Ollama LLM (qwen2.5:1.5b - light enough for an old laptop)
REM
REM  Entry point: uvicorn src.web_server:build_default_app --factory
REM
REM  >>> EDIT THE HARDWARE SETTINGS IN THE CONFIG BLOCK BELOW <<<
REM  The values shipped here are the project defaults; change X32_IP and
REM  AUDIO_DEVICE to match your room before serious testing.
REM ============================================================================

setlocal

REM --- Always work from this batch file's own folder (needed so "src" imports) -
cd /d "%~dp0"

REM ===========================================================================
REM  DEPENDENCY CHECK  --  real mic capture needs the 'sounddevice' package
REM  (not required by the demo). If it's missing the app still runs failsafe,
REM  but it can't hear the room, so offer to install it before launching.
REM ===========================================================================
python -c "import sounddevice" 1>nul 2>nul
if errorlevel 1 (
    echo.
    echo [!] The 'sounddevice' package is NOT installed.
    echo     Without it the app cannot capture audio from your mic / interface.
    echo.
    choice /c YN /n /m "Install it now with pip?  [Y/N]: "
    if errorlevel 2 (
        echo     Skipping install - continuing WITHOUT live audio capture.
    ) else (
        echo     Installing sounddevice...
        python -m pip install sounddevice
    )
    echo.
)

REM ===========================================================================
REM  HARDWARE CONFIG  --  edit these to match your rig
REM ===========================================================================

REM -- Behringer X32 mixer (OSC over your network) ----------------------------
set "SOUND_ADVISOR_X32_IP=192.168.0.2"
set "SOUND_ADVISOR_X32_PORT=10023"
set "SOUND_ADVISOR_OSC_TIMEOUT=1.0"

REM -- Audio input (the room / main-mix capture) -----------------------------
REM Leave AUDIO_DEVICE blank to use the Windows default input device, OR set it
REM to a device index (e.g. 1) or a device-name substring (e.g. "Scarlett").
REM To list devices: python -c "import sounddevice as sd; print(sd.query_devices())"
set "SOUND_ADVISOR_AUDIO_DEVICE="
set "SOUND_ADVISOR_SAMPLE_RATE=48000"
set "SOUND_ADVISOR_AUDIO_CHANNELS=1"

REM -- Optional second input = recorder / broadcast feed (enables the Broadcast
REM    analysis panel). Leave blank to disable. --------------------------------
set "SOUND_ADVISOR_RECORDING_DEVICE="

REM -- Where profiles and session logs are written ---------------------------
set "SOUND_ADVISOR_PROFILE_DIR=profiles"
set "SOUND_ADVISOR_LOG_DIR=logs"

REM ===========================================================================
REM  AI / LLM CONFIG  --  real local Ollama brain
REM ===========================================================================
set "SOUND_ADVISOR_LLM_BACKEND=ollama"
set "SOUND_ADVISOR_LLM_MODEL=qwen2.5:1.5b"
REM Old laptop -> give generation extra head-room before falling back to alerts.
set "SOUND_ADVISOR_LLM_TIMEOUT=60"

REM ===========================================================================
REM  Server host / port
REM ===========================================================================
set "HOST=127.0.0.1"
set "PORT=8000"

echo ============================================================
echo   AI Sound Advisor - REAL HARDWARE Test Launcher
echo ============================================================
echo   X32 mixer    : %SOUND_ADVISOR_X32_IP%:%SOUND_ADVISOR_X32_PORT%
echo   Audio device : %SOUND_ADVISOR_AUDIO_DEVICE% (blank = system default)
echo   LLM          : ollama / %SOUND_ADVISOR_LLM_MODEL%
echo   Dashboard    : http://%HOST%:%PORT%
echo ============================================================
echo.

REM --- 1. Start Ollama (local AI brain) in a separate window -----------------
echo [1/3] Waking the local AI brain (ollama run %SOUND_ADVISOR_LLM_MODEL%)...
start "Ollama - %SOUND_ADVISOR_LLM_MODEL%" cmd /k "ollama run %SOUND_ADVISOR_LLM_MODEL%"

REM --- 2. (env vars already set above - the server inherits them) ------------
echo [2/3] Hardware + AI switches set.

REM --- 3. Open the dashboard in the browser once the server has booted -------
echo [3/3] Starting live server and opening the dashboard...
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start "" http://%HOST%:%PORT%"

echo.
echo Live server is starting on http://%HOST%:%PORT%
echo Close this window (or press Ctrl+C) to stop the server.
echo ------------------------------------------------------------
echo.

REM --- Launch the LIVE app via Uvicorn (factory builds real hardware wiring) --
python -m uvicorn src.web_server:build_default_app --factory --host %HOST% --port %PORT%

endlocal
