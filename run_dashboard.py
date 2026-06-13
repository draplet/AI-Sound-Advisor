"""
run_dashboard.py

Hardware-free demo launcher for the AI Sound Advisor dashboard.

There is no real microphone (sounddevice) or X32 attached in this environment,
so this script wires the SystemLoopOrchestrator with in-memory fakes that
simulate a live mix. The audio "engine" cycles through a few states every few
seconds (clean -> clipping -> vocal masking) so the dashboard visibly comes
alive: colour-coded alerts, confidence, the controls, the prompt box and the
Ignore button all work against real agent logic.

Run:
    python run_dashboard.py
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import numpy as np
import uvicorn

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.detection import DetectionEngine
from src.interaction import UserInteractionAgent
from src.mixer_state import MixerStateAgent, X32OscTransport
from src.pacing import SuggestionPacingAgent
from src.profile_store import ContextProfileAgent
from src.suggestion import SuggestionGenerator
from src.system_loop import AudioSource, SystemLoopOrchestrator
from src.web_server import create_app

HOST = "127.0.0.1"
PORT = 8000
SECONDS_PER_STATE = 6  # how long each simulated condition lasts


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

    def analyze(self, samples, sample_rate) -> AudioMetrics:
        idx = (self._calls // self._ticks_per_state) % len(self._STATES)
        self._calls += 1
        return self._STATES[idx]


class SilentAudioSource(AudioSource):
    """Returns a dummy buffer; the DemoAudioEngine ignores it anyway."""

    def capture(self):
        return np.zeros(2_048), 48_000


def build_demo_app():
    profile_agent = ContextProfileAgent.in_memory("demo")
    generator = SuggestionGenerator()  # no LLM -> built-in instructor fallbacks
    interaction_agent = UserInteractionAgent(generator, profile_agent)

    # ~3.3 ticks/sec on the WebSocket -> convert seconds-per-state to ticks.
    ticks_per_state = int(SECONDS_PER_STATE * 1000 / 300)

    orchestrator = SystemLoopOrchestrator(
        audio_source=SilentAudioSource(),
        audio_engine=DemoAudioEngine(ticks_per_state),
        mixer_agent=MixerStateAgent(DemoX32Transport()),
        detection_engine=DetectionEngine.default(),
        profile_agent=profile_agent,
        pacing_agent=SuggestionPacingAgent(),
        suggestion_generator=generator,
        interaction_agent=interaction_agent,
        channels=(3,),
    )
    return create_app(
        orchestrator=orchestrator,
        profile_agent=profile_agent,
        interaction_agent=interaction_agent,
    )


app = build_demo_app()

if __name__ == "__main__":
    print(f"AI Sound Advisor dashboard -> http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
