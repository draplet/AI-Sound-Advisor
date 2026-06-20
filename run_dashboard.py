"""
run_dashboard.py

Hardware-free launcher for the AI Sound Advisor dashboard.

There is no real microphone (sounddevice) attached in this environment, so the
audio side is simulated. It feeds a STEADY, clean mix so the dashboard sits
quietly on "Your mix is sounding good" — no rolling demo alerts. (To replay the
old cycling demo of clipping / vocal-masking, set SIMULATE_ISSUES = True below.)
The X32 Connect / Test Connection panel talks to a real mixer over the network.

Run:
    python run_dashboard.py
Then open http://127.0.0.1:8001
"""
from __future__ import annotations

import os
import sys

print("[DEBUG] Starting imports...", file=sys.stderr)

try:
    import numpy as np
    import uvicorn
    print("[DEBUG] Core imports OK", file=sys.stderr)

    from src.audio_analysis import AudioAnalysisEngine, AudioMetrics, EnergyLevel
    from src.detection import DetectionEngine
    from src.interaction import UserInteractionAgent
    from src.mixer_state import MixerStateAgent, X32OscTransport
    from src.pacing import SuggestionPacingAgent
    from src.profile_store import ContextProfileAgent
    from src.recording_analysis import RecordingAnalysisEngine
    from src.suggestion import SuggestionGenerator
    from src.llm_client import LLMBackend, LLMConfig, build_llm_client
    from src.system_loop import (
        AudioSource,
        SoundDeviceAudioSource,
        SystemLoopOrchestrator,
    )
    from src.network_config import NetworkConfigStore
    from src.web_server import create_app
    from src.x32_connection import UdpOscChannel, X32ConnectionManager
    print("[DEBUG] All imports OK", file=sys.stderr)
except Exception as e:
    print(f"[ERROR] Import failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    print("\nPress Enter to exit...")
    input()
    sys.exit(1)

HOST = "127.0.0.1"
PORT = 8001

#: Local Ollama model the AI panel uses by default. Must be installed
#: (check with `ollama list`; pull with e.g. `ollama pull qwen2.5:1.5b`).
#: Override at runtime with SOUND_ADVISOR_LLM_MODEL / _LLM_BACKEND.
OLLAMA_MODEL = "qwen2.5:1.5b"

#: Banner shown while there is no audio input yet or the X32 is not connected.
#: Until real communication is established the dashboard does NOT claim the mix
#: is good — it waits.
WAITING_STATUS = "Waiting for communication — connect the X32 to begin."

#: Analyze a REAL audio input (the room mic / an interface / the X32 over USB)
#: instead of a simulated feed. True by default so readings + suggestions reflect
#: the live signal. Set False to fall back to the simulated audio below.
USE_REAL_AUDIO = os.environ.get("SOUND_ADVISOR_USE_REAL_AUDIO", "1") not in ("0", "", "false", "False")

#: Which input device to capture. None = the Windows default input. Override with
#: SOUND_ADVISOR_AUDIO_DEVICE = a device index (e.g. 1) or a name substring
#: (e.g. "X32" or "Scarlett"). List devices:
#:   py -3.10 -c "import sounddevice as sd; print(sd.query_devices())"
def _audio_device():
    raw = os.environ.get("SOUND_ADVISOR_AUDIO_DEVICE")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return raw

AUDIO_DEVICE = _audio_device()
SAMPLE_RATE = int(os.environ.get("SOUND_ADVISOR_SAMPLE_RATE") or 48_000)
AUDIO_CHANNELS = int(os.environ.get("SOUND_ADVISOR_AUDIO_CHANNELS") or 1)

#: When False (default) the simulated engine holds a steady clean mix -> no alerts.
#: Set True to replay the old rolling demo (clean -> clipping -> vocal masking).
#: Only used when USE_REAL_AUDIO is False.
SIMULATE_ISSUES = False
SECONDS_PER_STATE = 6  # how long each simulated condition lasts (when simulating)


class DemoX32Transport(X32OscTransport):
    """A connected X32 with one channel: a lowered 'Lead Vocal' (fader ~-10 dB)."""

    def query(self, address: str):
        if address.endswith("/config/name"):
            return "Lead Vocal"
        if address.endswith("/mix/fader"):
            return 0.5      # ~ -10 dB -> below the prominence line
        if address.endswith("/mix/on"):
            return 1        # unmuted / audible
        if "/headamp/" in address:
            return 0.4
        return 0


#: A steady, clean mix that sits inside every detection threshold (Conservative
#: target -23 LUFS +/-3 dB; clip line -1 dBFS; energy NORMAL) -> zero alerts.
CLEAN_METRICS = AudioMetrics(
    loudness_lufs=-22.0, peak_level=-10.0,
    low_mid_energy=EnergyLevel.NORMAL,
    high_freq_energy=EnergyLevel.NORMAL,
)


class SteadyCleanAudioEngine:
    """Always reports the same clean mix -> the dashboard shows 'mix sounding good'."""

    def analyze(self, samples, sample_rate) -> AudioMetrics:
        return CLEAN_METRICS


class DemoAudioEngine:
    """Cycles through realistic metric states so the dashboard looks live."""

    _STATES = [
        # Clean mix -> "Your mix is sounding good"
        AudioMetrics(loudness_lufs=-22.0, peak_level=-10.0,
                     low_mid_energy=EnergyLevel.NORMAL,
                     high_freq_energy=EnergyLevel.NORMAL),
        # Hot peaks -> CLIPPING (red)
        AudioMetrics(loudness_lufs=-12.0, peak_level=0.0,
                     low_mid_energy=EnergyLevel.NORMAL,
                     high_freq_energy=EnergyLevel.NORMAL),
        # Heavy low-mid + lowered vocal -> VOCAL MASKING (+ eq mud)
        AudioMetrics(loudness_lufs=-18.0, peak_level=-6.0,
                     low_mid_energy=EnergyLevel.HIGH,
                     high_freq_energy=EnergyLevel.NORMAL),
    ]

    def __init__(self, ticks_per_state: int):
        self._ticks_per_state = max(1, ticks_per_state)
        self._calls = 0

    def _current_idx(self) -> int:
        return (self._calls // self._ticks_per_state) % len(self._STATES)

    def analyze(self, samples, sample_rate) -> AudioMetrics:
        # The recorder feed is tagged with a non-zero buffer (see
        # DemoRecordingSource): return a deliberately mismatched broadcast
        # version of the current room state so the recording panel shows
        # findings — quieter (loudness imbalance) and dull (vocal clarity).
        first = float(np.asarray(samples).reshape(-1)[0])
        if first != 0.0:
            room = self._STATES[self._current_idx()]
            return AudioMetrics(
                loudness_lufs=room.loudness_lufs - 8.0,
                peak_level=room.peak_level,
                low_mid_energy=room.low_mid_energy,
                high_freq_energy=EnergyLevel.LOW,
            )
        idx = self._current_idx()
        self._calls += 1          # only the room capture advances the cycle
        return self._STATES[idx]


class SilentAudioSource(AudioSource):
    """The room/main capture: returns a zero buffer (the DemoAudioEngine cycles
    states by call count, not buffer content)."""

    def capture(self):
        return np.zeros(2_048), 48_000


class DemoRecordingSource(AudioSource):
    """The recorder/stream feed: a non-zero buffer tags it so DemoAudioEngine
    returns the mismatched broadcast metrics."""

    def capture(self):
        return np.ones(2_048), 48_000


class WaitingAwareOrchestrator(SystemLoopOrchestrator):
    """Gates the all-clear banner on real communication.

    The demo feeds simulated-but-always-present audio, so without this the
    dashboard would claim "Your mix is sounding good" the instant it starts —
    before anything is actually connected. Instead, after each tick we check the
    real master X32 connection and whether audio metrics were produced; until
    BOTH are true the frame shows a "waiting for communication" banner and no
    issues/suggestions. "Your mix is sounding good" therefore only appears once
    a connection is up and a good mix has genuinely been measured.
    """

    def __init__(self, *args, connection_manager: X32ConnectionManager, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection_manager = connection_manager

    def tick(self, now_ms: int):
        frame = super().tick(now_ms)
        try:
            connected = bool(self._connection_manager.status().connected)
        except Exception:  # noqa: BLE001 — failsafe: treat as not connected
            connected = False
        has_input = frame.metrics is not None
        if not connected or not has_input:
            frame = frame.model_copy(update={
                "all_clear": False,
                "status_message": WAITING_STATUS,
                "issues": [],
                "suggestions": [],
                "mixer_connected": connected,
            })
            self._last_frame = frame
        return frame


def build_demo_app():
    profile_agent = ContextProfileAgent.in_memory("demo")
    # Local AI brain (Agent 4). Honour any env-configured backend (e.g. set by
    # run_complete_demo.bat); otherwise default to local Ollama so the AI prompt
    # panel works out of the box. Failsafe: if Ollama is not running / the model
    # is missing, the generator silently falls back to built-in instructor alerts.
    llm_config = LLMConfig.from_env()
    if llm_config.backend is LLMBackend.NONE:
        llm_config = LLMConfig(backend=LLMBackend.OLLAMA, model=OLLAMA_MODEL)
    generator = SuggestionGenerator(build_llm_client(llm_config))
    interaction_agent = UserInteractionAgent(generator, profile_agent)

    if USE_REAL_AUDIO:
        # Live analysis of a real input (room mic / interface / X32-over-USB).
        # Readings + clipping/loudness/EQ suggestions reflect the actual signal.
        # If the device can't be read, the orchestrator degrades to "audio lost"
        # and the waiting banner — it never claims the mix is good.
        audio_source = SoundDeviceAudioSource(
            sample_rate=SAMPLE_RATE, channels=AUDIO_CHANNELS, device=AUDIO_DEVICE,
        )
        audio_engine = AudioAnalysisEngine.default()
        recording_source = None
        recording_engine = None
    elif SIMULATE_ISSUES:
        # Old rolling demo: cycle clean -> clipping -> vocal masking, plus a
        # mismatched broadcast feed so the recording panel shows findings.
        ticks_per_state = int(SECONDS_PER_STATE * 1000 / 300)  # ~3.3 ticks/sec
        audio_source = SilentAudioSource()
        audio_engine = DemoAudioEngine(ticks_per_state)
        recording_source = DemoRecordingSource()
        recording_engine = RecordingAnalysisEngine.default()
    else:
        # Quiet simulated default: a steady clean mix and no recording feed -> no
        # rolling alerts; the dashboard rests on "Your mix is sounding good".
        audio_source = SilentAudioSource()
        audio_engine = SteadyCleanAudioEngine()
        recording_source = None
        recording_engine = None

    # Connect panel + Settings "Test Connection" wired to a REAL UDP/OSC channel
    # so /info is actually sent over the network to the configured X32. The probe
    # only reports success on a genuine reply; a wrong IP / missing mixer / no
    # network times out and surfaces as an error (spec: "if no return from /info
    # display error"). The audio side stays simulated (no mic in this env).
    network_store = NetworkConfigStore()        # real project config.json
    net = network_store.load()                  # creates it from X32 factory defaults on first run
    x32_connection = X32ConnectionManager(
        channel_factory=lambda ip, port: UdpOscChannel(ip, port),
        info_timeout=net.timeout_s,             # honour the saved Timeout setting
    )

    orchestrator = WaitingAwareOrchestrator(
        audio_source=audio_source,
        audio_engine=audio_engine,
        mixer_agent=MixerStateAgent(DemoX32Transport()),
        detection_engine=DetectionEngine.default(),
        profile_agent=profile_agent,
        pacing_agent=SuggestionPacingAgent(),
        suggestion_generator=generator,
        interaction_agent=interaction_agent,
        channels=(3,),
        recording_source=recording_source,
        recording_analysis_engine=recording_engine,
        connection_manager=x32_connection,      # gate the banner on real comms
    )
    return create_app(
        orchestrator=orchestrator,
        profile_agent=profile_agent,
        interaction_agent=interaction_agent,
        x32_connection=x32_connection,
        network_store=network_store,
    )


app = build_demo_app()

if __name__ == "__main__":
    try:
        print(f"AI Sound Advisor dashboard -> http://{HOST}:{PORT}")
        print("Press Ctrl+C to stop")
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()

