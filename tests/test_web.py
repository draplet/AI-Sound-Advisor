"""
tests/test_web.py

AI Sound Advisor — Web UI / Dashboard Agent
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — UI / FRONTEND AGENT):
  - "Web-based UI hosted locally" / "Backend: Python (FastAPI)" / "HTML/JS V1"
  - Suggestions panel (priority colour-coded, confidence, all-clear message)
  - Controls panel: Mode (Sermon/Worship/Play) + Loudness mode selectors,
    X32 + Audio connection indicators
  - AI prompt panel: free-text instruction box -> Agent 7
  - "Real-time updates 2-4 times per second" via WebSocket or fast polling
  - "Ignore: temporary suppression (e.g. 3 minutes)" -> Agent 7
  - "System must never crash during service"

The server hosts a FastAPI app wrapping the SystemLoopOrchestrator (all 7
agents). Each state request runs one orchestrator tick over an injected
(deterministic) clock — no real sleeps, no hardware.

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.web_server  ->  create_app(orchestrator, profile_agent, interaction_agent)

Run with:
  pytest tests/test_web.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.detection import DetectionEngine, IssueType
from src.interaction import UserInteractionAgent
from src.mixer_state import MixerStateAgent, X32OscTransport
from src.pacing import SuggestionPacingAgent
from src.profile_store import ContextProfileAgent
from src.suggestion import SuggestionGenerator
from src.state_manager import LoudnessMode, Scene
from src.system_loop import SystemLoopOrchestrator, AudioSource

from src.web_server import create_app, ServerConfig, build_default_app, DEFAULT_X32_IP


# ===========================================================================
# Test doubles (in-memory — no hardware, no network, no LLM)
# ===========================================================================

class FakeX32Transport(X32OscTransport):
    def __init__(self, name="Lead Vocal", fader=0.25, on=1, gain=0.5, fail=False):
        self.name, self.fader, self.on, self.gain, self.fail = (
            name, fader, on, gain, fail
        )

    def query(self, address: str):
        if self.fail:
            raise ConnectionError("X32 not responding")
        if address.endswith("/config/name"):
            return self.name
        if address.endswith("/mix/fader"):
            return self.fader
        if address.endswith("/mix/on"):
            return self.on
        if "/headamp/" in address:
            return self.gain
        return 0


class StubAudioEngine:
    def __init__(self, metrics: AudioMetrics):
        self.metrics = metrics

    def analyze(self, samples, sample_rate) -> AudioMetrics:
        return self.metrics


class FakeAudioSource(AudioSource):
    def __init__(self, fail=False):
        self.fail = fail

    def capture(self):
        if self.fail:
            raise OSError("audio device disconnected")
        return np.zeros(2_048), 48_000


def metrics(loudness=-22.0, peak=-10.0,
            low_mid=EnergyLevel.NORMAL, high=EnergyLevel.NORMAL) -> AudioMetrics:
    return AudioMetrics(loudness_lufs=loudness, peak_level=peak,
                        low_mid_energy=low_mid, high_freq_energy=high)


def clean_metrics():
    return metrics()


def clipping_metrics():
    return metrics(peak=0.0)


def make_clock(start=1_000, step=1_000):
    """Deterministic monotonically-increasing millisecond clock."""
    box = {"t": start - step}

    def clock():
        box["t"] += step
        return box["t"]

    return clock


# ===========================================================================
# App + client factory
# ===========================================================================

def make_env(audio_metrics=None, transport=None, audio_source=None):
    transport = transport or FakeX32Transport()
    audio_source = audio_source or FakeAudioSource()
    audio_engine = StubAudioEngine(audio_metrics or clean_metrics())

    profile_agent = ContextProfileAgent.in_memory("web-test")
    generator = SuggestionGenerator()  # no LLM -> deterministic fallbacks
    interaction_agent = UserInteractionAgent(generator, profile_agent)

    orch = SystemLoopOrchestrator(
        audio_source=audio_source,
        audio_engine=audio_engine,
        mixer_agent=MixerStateAgent(transport),
        detection_engine=DetectionEngine.default(),
        profile_agent=profile_agent,
        pacing_agent=SuggestionPacingAgent(),
        suggestion_generator=generator,
        interaction_agent=interaction_agent,
        channels=(3,),
    )
    app = create_app(
        orchestrator=orch,
        profile_agent=profile_agent,
        interaction_agent=interaction_agent,
        clock=make_clock(),
    )
    return SimpleNamespace(
        app=app, client=TestClient(app),
        profile_agent=profile_agent, interaction_agent=interaction_agent,
        orch=orch,
    )


# ===========================================================================
# SECTION 1 — Serving the dashboard page
# ===========================================================================

class TestPageServing:

    def test_root_serves_html(self):
        env = make_env()
        resp = env.client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_page_contains_dashboard_panels(self):
        env = make_env()
        body = env.client.get("/").text.lower()
        # Core panels named in the spec should be present in the markup.
        assert "suggestion" in body
        assert "prompt" in body
        # Connection indicators for X32 + audio input.
        assert "x32" in body


# ===========================================================================
# SECTION 2 — Real-time state endpoint (HTTP polling fallback)
# ===========================================================================

class TestStateEndpoint:

    def test_state_returns_expected_shape(self):
        env = make_env(clean_metrics())
        data = env.client.get("/api/state").json()
        for key in ("all_clear", "status_message", "suggestions", "issues",
                    "audio_ok", "mixer_connected", "scene", "loudness_mode",
                    "warnings", "metrics", "tick_index", "now_ms"):
            assert key in data

    def test_clean_mix_reports_all_clear(self):
        env = make_env(clean_metrics())
        data = env.client.get("/api/state").json()
        assert data["all_clear"] is True
        assert data["status_message"] == "Your mix is sounding good"
        assert data["suggestions"] == []

    def test_detected_issue_appears_with_message(self):
        env = make_env(clipping_metrics())
        data = env.client.get("/api/state").json()
        issue_types = {i["issue"] for i in data["issues"]}
        assert IssueType.CLIPPING.value in issue_types
        # The suggestions panel carries colour + confidence + a message.
        clip = next(s for s in data["suggestions"]
                    if s["issue"] == IssueType.CLIPPING.value)
        assert clip["message"]
        assert clip["priority"]
        assert "confidence" in clip

    def test_connection_indicators_reflect_hardware(self):
        env = make_env(clean_metrics())
        data = env.client.get("/api/state").json()
        assert data["audio_ok"] is True
        assert data["mixer_connected"] is True

    def test_disconnected_mixer_is_reported(self):
        env = make_env(clean_metrics(), transport=FakeX32Transport(fail=True))
        data = env.client.get("/api/state").json()
        assert data["mixer_connected"] is False
        assert data["warnings"]

    def test_lost_audio_is_reported(self):
        env = make_env(clean_metrics(), audio_source=FakeAudioSource(fail=True))
        data = env.client.get("/api/state").json()
        assert data["audio_ok"] is False
        # Must never falsely claim the mix is good when it cannot hear.
        assert data["all_clear"] is False


# ===========================================================================
# SECTION 3 — Controls panel: Mode + Loudness selectors (Agent 5)
# ===========================================================================

class TestControls:

    def test_set_mode_updates_scene(self):
        env = make_env(clean_metrics())
        resp = env.client.post("/api/mode", json={"mode": "worship"})
        assert resp.status_code == 200
        assert env.profile_agent.active.event_type == Scene.WORSHIP
        # And it surfaces in the next state poll.
        assert env.client.get("/api/state").json()["scene"] == "worship"

    def test_set_loudness_mode_updates_profile(self):
        env = make_env(clean_metrics())
        resp = env.client.post("/api/loudness",
                               json={"loudness_mode": "energetic"})
        assert resp.status_code == 200
        assert env.profile_agent.active.loudness_mode == LoudnessMode.ENERGETIC
        assert env.client.get("/api/state").json()["loudness_mode"] == "energetic"

    def test_invalid_mode_is_rejected(self):
        env = make_env(clean_metrics())
        resp = env.client.post("/api/mode", json={"mode": "bogus"})
        assert resp.status_code == 422


# ===========================================================================
# SECTION 4 — AI prompt panel routes into Agent 7 / Agent 5
# ===========================================================================

class TestPromptPanel:

    def test_prompt_is_recorded_as_instruction(self):
        env = make_env(clean_metrics())
        resp = env.client.post("/api/prompt",
                               json={"text": "The piano is always loud"})
        assert resp.status_code == 200
        # Agent 7 -> Agent 5: instruction stored, source override learned.
        assert "The piano is always loud" in env.profile_agent.active.user_instructions
        assert "piano" in env.profile_agent.active.overrides

    def test_prompt_returns_acknowledgement(self):
        env = make_env(clean_metrics())
        data = env.client.post("/api/prompt",
                               json={"text": "This is a quiet service"}).json()
        assert data["action"] == "instruction"
        assert data["message"]


# ===========================================================================
# SECTION 5 — Ignore action: temporary 3-minute suppression (Agent 7)
# ===========================================================================

class TestIgnoreAction:

    def test_ignore_suppresses_issue(self):
        env = make_env(clipping_metrics())
        # Confirm the issue is present first.
        before = env.client.get("/api/state").json()
        assert IssueType.CLIPPING.value in {i["issue"] for i in before["issues"]}

        resp = env.client.post("/api/ignore",
                               json={"issue": "clipping", "channel": None})
        assert resp.status_code == 200

        after = env.client.get("/api/state").json()
        assert IssueType.CLIPPING.value not in {i["issue"] for i in after["issues"]}

    def test_ignore_returns_acknowledgement(self):
        env = make_env(clipping_metrics())
        data = env.client.post("/api/ignore",
                               json={"issue": "clipping", "channel": None}).json()
        assert data["action"] == "ignore"


# ===========================================================================
# SECTION 6 — Real-time WebSocket channel
# ===========================================================================

class TestWebSocket:

    def test_websocket_pushes_state_frames(self):
        env = make_env(clipping_metrics())
        with env.client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
        for key in ("all_clear", "suggestions", "issues",
                    "audio_ok", "mixer_connected"):
            assert key in data
        assert IssueType.CLIPPING.value in {i["issue"] for i in data["issues"]}


# ===========================================================================
# SECTION 7 — Failsafe: a broken orchestrator never takes the server down
# ===========================================================================

# ===========================================================================
# SECTION 8 — Configuration via environment variables
# ===========================================================================

class TestServerConfig:

    def test_defaults_when_no_env(self):
        cfg = ServerConfig.from_env({})
        assert cfg.x32_ip == DEFAULT_X32_IP
        assert cfg.x32_port == 10023
        assert cfg.sample_rate == 48_000
        assert cfg.audio_channels == 1
        assert cfg.audio_device is None
        assert cfg.profile_dir == "profiles"

    def test_env_overrides_are_applied(self):
        cfg = ServerConfig.from_env({
            "SOUND_ADVISOR_X32_IP": "10.0.0.5",
            "SOUND_ADVISOR_X32_PORT": "10024",
            "SOUND_ADVISOR_OSC_TIMEOUT": "2.5",
            "SOUND_ADVISOR_AUDIO_DEVICE": "3",
            "SOUND_ADVISOR_SAMPLE_RATE": "44100",
            "SOUND_ADVISOR_AUDIO_CHANNELS": "2",
            "SOUND_ADVISOR_PROFILE_DIR": "/srv/profiles",
        })
        assert cfg.x32_ip == "10.0.0.5"
        assert cfg.x32_port == 10024
        assert cfg.osc_timeout == 2.5
        assert cfg.audio_device == 3            # numeric device index
        assert cfg.sample_rate == 44100
        assert cfg.audio_channels == 2
        assert cfg.profile_dir == "/srv/profiles"

    def test_audio_device_accepts_a_name_string(self):
        cfg = ServerConfig.from_env({"SOUND_ADVISOR_AUDIO_DEVICE": "Scarlett 2i2"})
        assert cfg.audio_device == "Scarlett 2i2"

    def test_invalid_numbers_fall_back_to_defaults(self):
        cfg = ServerConfig.from_env({
            "SOUND_ADVISOR_X32_PORT": "not-a-port",
            "SOUND_ADVISOR_SAMPLE_RATE": "",
        })
        assert cfg.x32_port == 10023
        assert cfg.sample_rate == 48_000


class TestBuildDefaultApp:

    def test_build_with_no_config_reads_env(self, monkeypatch):
        monkeypatch.setenv("SOUND_ADVISOR_X32_IP", "192.168.1.77")
        monkeypatch.setenv("SOUND_ADVISOR_X32_PORT", "10024")
        app = build_default_app()
        orch = app.state.orchestrator
        assert orch._mixer_agent._transport._ip == "192.168.1.77"
        assert orch._mixer_agent._transport._port == 10024

    def test_build_applies_explicit_config(self):
        cfg = ServerConfig(x32_ip="172.16.0.9", x32_port=10024,
                           audio_device=2, sample_rate=44100, audio_channels=2)
        app = build_default_app(cfg)
        orch = app.state.orchestrator
        assert orch._mixer_agent._transport._ip == "172.16.0.9"
        assert orch._audio_source.sample_rate == 44100
        assert orch._audio_source.device == 2
        assert orch._audio_source.channels == 2
        # The app is fully wired and serves its routes.
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert {"/", "/api/state", "/ws"} <= paths


class TestServerFailsafe:

    def test_state_endpoint_survives_orchestrator_error(self):
        env = make_env(clean_metrics())

        class Boom:
            def tick(self, now_ms):
                raise RuntimeError("kaboom")

        env.orch.__dict__  # touch to ensure attribute access is fine
        env.client.app  # sanity
        # Swap the orchestrator the app holds for one that always explodes.
        env.app.state.orchestrator = Boom()
        resp = env.client.get("/api/state")
        assert resp.status_code == 200
        assert resp.json()["status_message"]
