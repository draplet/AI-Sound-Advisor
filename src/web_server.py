"""
src/web_server.py

Web UI / Dashboard Agent for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — UI / FRONTEND AGENT):
  - "Web-based UI hosted locally" / "Backend: Python (FastAPI)" / "HTML/JS (V1)"
  - Suggestions panel: priority colour-coding, confidence, all-clear message
  - Controls panel: Mode (Sermon/Worship/Play) + Loudness mode selectors,
    X32 + Audio connection indicators
  - AI prompt panel: free-text instruction box routed into Agent 7
  - "Real-time updates 2-4 times per second" via WebSocket (with HTTP-poll
    fallback at /api/state)
  - "Ignore: temporary suppression (e.g. 3 minutes)" -> Agent 7
  - Failsafe: "System must never crash during service" — every endpoint
    degrades gracefully instead of raising.

Design
------
The server is a thin HTTP/WebSocket shell around the SystemLoopOrchestrator
(which already integrates all 7 agents). Each state request runs ONE orchestrator
tick and returns the resulting LoopFrame as JSON. The orchestrator, the agents the
UI mutates (profile + interaction), the clock and a small presentation cache all
live on ``app.state`` so they can be inspected or swapped in tests.

Presentation cache (LLM-friendly display)
------------------------------------------
The orchestrator only generates a suggestion's text on the tick an issue is new
or due for a repeat (the spec's "LLM not every loop" rule). To keep the panel
stable between those ticks, the server caches the latest suggestion text per
issue and re-attaches it to the currently-active issues, pruning entries whose
issue has cleared. No extra LLM calls are made.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.detection import Issue, IssueType, Priority, confidence_level
from src.event_log import SessionLogger
from src.interaction import IGNORE_DEFAULT_MS, UserInteractionAgent
from src.mixer_state import X32_OSC_PORT
from src.profile_store import ContextProfileAgent
from src.state_manager import LoudnessMode, Scene
from src.suggestion import PRIORITY_ICONS
from src.system_loop import SystemLoopOrchestrator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Where the dashboard template lives (spec: src/templates/index.html).
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"

#: WebSocket push cadence (ms). Spec: "2-4 times per second" -> 300ms ~= 3.3 Hz.
DEFAULT_WS_INTERVAL_MS = 300

#: Status shown if even the failsafe frame cannot be built.
SERVER_DEGRADED_STATUS = "Dashboard running in degraded mode."

#: Default X32 mixer IP (override with SOUND_ADVISOR_X32_IP).
DEFAULT_X32_IP = "192.168.0.2"

#: Prefix for all dashboard environment-variable settings.
ENV_PREFIX = "SOUND_ADVISOR_"

IssueKey = Tuple[object, Optional[str]]


# ===========================================================================
# Configuration (environment-driven, so no code edits to point at real gear)
# ===========================================================================

def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    """Parse an int env var, falling back to ``default`` on missing/invalid."""
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_device(env: Mapping[str, str], key: str):
    """Audio device: an int index when numeric, else a device-name string."""
    raw = env.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


class ServerConfig(BaseModel):
    """Runtime configuration for the live dashboard, sourced from env vars.

    Environment variables (all optional; sensible defaults shown):
      SOUND_ADVISOR_X32_IP          X32 mixer IP        (default 192.168.0.2)
      SOUND_ADVISOR_X32_PORT        X32 OSC UDP port    (default 10023)
      SOUND_ADVISOR_OSC_TIMEOUT     OSC reply timeout s (default 1.0)
      SOUND_ADVISOR_AUDIO_DEVICE    input device index or name (default system)
      SOUND_ADVISOR_SAMPLE_RATE     capture sample rate (default 48000)
      SOUND_ADVISOR_AUDIO_CHANNELS  capture channels    (default 1)
      SOUND_ADVISOR_PROFILE_DIR     where profiles save (default "profiles")
    """

    x32_ip: str = DEFAULT_X32_IP
    x32_port: int = X32_OSC_PORT
    osc_timeout: float = 1.0
    audio_device: Optional[Union[int, str]] = None
    sample_rate: int = 48_000
    audio_channels: int = 1
    profile_dir: str = "profiles"
    log_dir: str = "logs"
    #: Optional recorder/stream feed input device (enables recording analysis).
    recording_device: Optional[Union[int, str]] = None

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "ServerConfig":
        """Build a config from a mapping (defaults to ``os.environ``)."""
        env = os.environ if environ is None else environ
        return cls(
            x32_ip=env.get(f"{ENV_PREFIX}X32_IP") or DEFAULT_X32_IP,
            x32_port=_env_int(env, f"{ENV_PREFIX}X32_PORT", X32_OSC_PORT),
            osc_timeout=_env_float(env, f"{ENV_PREFIX}OSC_TIMEOUT", 1.0),
            audio_device=_env_device(env, f"{ENV_PREFIX}AUDIO_DEVICE"),
            sample_rate=_env_int(env, f"{ENV_PREFIX}SAMPLE_RATE", 48_000),
            audio_channels=_env_int(env, f"{ENV_PREFIX}AUDIO_CHANNELS", 1),
            profile_dir=env.get(f"{ENV_PREFIX}PROFILE_DIR") or "profiles",
            log_dir=env.get(f"{ENV_PREFIX}LOG_DIR") or "logs",
            recording_device=_env_device(env, f"{ENV_PREFIX}RECORDING_DEVICE"),
        )


# ===========================================================================
# Request models (FastAPI validates these — bad input -> 422, never a crash)
# ===========================================================================

class ModeRequest(BaseModel):
    mode: Scene


class LoudnessRequest(BaseModel):
    loudness_mode: LoudnessMode


class PromptRequest(BaseModel):
    text: str
    issue: Optional[IssueType] = None
    channel: Optional[str] = None


class IgnoreRequest(BaseModel):
    issue: IssueType
    channel: Optional[str] = None
    duration_ms: int = IGNORE_DEFAULT_MS


class IssueActionRequest(BaseModel):
    """Identifies the issue an operator acted on (More / Tell me how / Mark normal)."""

    issue: IssueType
    channel: Optional[str] = None


class FocusRequest(BaseModel):
    """Toggle for Focus Mode (show only the highest-priority issue)."""

    enabled: bool


class ChatTurn(BaseModel):
    """One prior turn of the AI chat conversation."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """A chat message plus the conversation so far (for multi-turn context)."""

    message: str
    history: List[ChatTurn] = Field(default_factory=list)


def _issue_from(req: "IssueActionRequest") -> Issue:
    """Rebuild a minimal Issue for an Agent 7 action.

    The action handlers key only on ``(issue, channel)``; priority/confidence
    are placeholders (mirrors how /api/ignore reconstructs an Issue).
    """
    return Issue(
        issue=req.issue, channel=req.channel,
        priority=Priority.LOW, confidence=0.0,
    )


# ===========================================================================
# Presentation helpers
# ===========================================================================

def _now_ms_realtime() -> int:
    """Default monotonic millisecond clock for live operation."""
    return int(time.monotonic() * 1000)


def _build_payload(frame, cache: Dict[IssueKey, object]) -> dict:
    """Merge the orchestrator frame with cached suggestion text for the UI.

    Updates ``cache`` with any freshly generated suggestions, prunes cleared
    issues, then renders one display entry per currently-active issue.
    """
    # Absorb this tick's freshly generated suggestions.
    for suggestion in frame.suggestions:
        cache[(suggestion.issue, suggestion.channel)] = suggestion

    current_keys = {(i.issue, i.channel) for i in frame.issues}
    for key in [k for k in cache if k not in current_keys]:
        del cache[key]

    display = []
    for issue in frame.issues:
        cached = cache.get((issue.issue, issue.channel))
        display.append(
            {
                "issue": issue.issue.value,
                "channel": issue.channel,
                "priority": issue.priority.value,
                "confidence": issue.confidence,
                "confidence_label": confidence_level(issue.confidence).value,
                "priority_icon": PRIORITY_ICONS.get(issue.priority, "🔵"),
                "message": cached.message if cached is not None else "",
            }
        )

    payload = frame.model_dump(mode="json")
    payload["suggestions"] = display
    return payload


def _degraded_payload() -> dict:
    """Last-resort safe payload if a tick blows up entirely."""
    return {
        "tick_index": 0,
        "now_ms": 0,
        "metrics": None,
        "issues": [],
        "suggestions": [],
        "all_clear": False,
        "status_message": SERVER_DEGRADED_STATUS,
        "audio_ok": False,
        "mixer_connected": False,
        "warnings": [SERVER_DEGRADED_STATUS],
        "scene": Scene.SERMON.value,
        "loudness_mode": LoudnessMode.CONSERVATIVE.value,
        "calibrated": False,
        "focus_mode": False,
        "recording": None,
    }


def _apply_focus(payload: dict) -> None:
    """Trim the payload to only the highest-priority issue (Focus Mode).

    The orchestrator already returns issues sorted by spec priority (then
    confidence), and the display suggestions are built in that same order, so
    the most urgent item is the first of each list.
    """
    payload["issues"] = (payload.get("issues") or [])[:1]
    payload["suggestions"] = (payload.get("suggestions") or [])[:1]


def _mix_summary(app: FastAPI) -> str:
    """A short natural-language summary of the current mix, to ground the chat."""
    try:
        frame = app.state.orchestrator.last_frame
        if frame is None:
            return ""
        if getattr(frame, "all_clear", False):
            return "the mix currently sounds good"
        parts = []
        for issue in getattr(frame, "issues", [])[:4]:
            where = f" on {issue.channel}" if issue.channel else " on the main mix"
            parts.append(f"{issue.issue.value}{where}")
        return "active issues: " + ", ".join(parts) if parts else ""
    except Exception:  # noqa: BLE001 — failsafe
        return ""


def _tick_payload(app: FastAPI) -> dict:
    """Run one orchestrator tick and build the UI payload (never raises)."""
    focus = bool(getattr(app.state, "focus_mode", False))
    try:
        orchestrator = app.state.orchestrator
        frame = orchestrator.tick(app.state.clock())
        payload = _build_payload(frame, app.state.suggestion_cache)
    except Exception:  # noqa: BLE001 — failsafe: never take the dashboard down
        payload = _degraded_payload()
    payload["focus_mode"] = focus
    if focus:
        _apply_focus(payload)
    return payload


# ===========================================================================
# App factory
# ===========================================================================

def create_app(
    orchestrator: SystemLoopOrchestrator,
    profile_agent: ContextProfileAgent,
    interaction_agent: UserInteractionAgent,
    *,
    clock: Optional[Callable[[], int]] = None,
    ws_interval_ms: int = DEFAULT_WS_INTERVAL_MS,
    template_path: Optional[Path] = None,
    event_logger: Optional[SessionLogger] = None,
) -> FastAPI:
    """Build the dashboard FastAPI app around a wired orchestrator."""
    app = FastAPI(title="AI Sound Advisor")

    app.state.orchestrator = orchestrator
    app.state.profile_agent = profile_agent
    app.state.interaction_agent = interaction_agent
    app.state.clock = clock or _now_ms_realtime
    app.state.ws_interval_ms = ws_interval_ms
    app.state.template_path = Path(template_path) if template_path else TEMPLATE_PATH
    app.state.suggestion_cache: Dict[IssueKey, object] = {}
    #: Optional logging system (spec LOGGING SYSTEM); None disables logging.
    app.state.event_logger = event_logger
    #: Focus Mode: when True, only the highest-priority issue is shown.
    app.state.focus_mode = False

    def _log_action(action: str, channel: Optional[str] = None,
                    detail: Optional[str] = None) -> None:
        """Record a user action if a logger is attached (never raises)."""
        logger = app.state.event_logger
        if logger is None:
            return
        try:
            logger.log_user_action(
                action, now_ms=app.state.clock(), channel=channel, detail=detail
            )
        except Exception:  # noqa: BLE001 — failsafe
            pass

    # ------------------------------------------------------------------
    # Page (step 9: display in UI)
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        try:
            html = app.state.template_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001 — failsafe
            html = "<h1>AI Sound Advisor</h1><p>Dashboard template missing.</p>"
        return HTMLResponse(content=html)

    # ------------------------------------------------------------------
    # Real-time state (HTTP polling fallback for the WebSocket)
    # ------------------------------------------------------------------

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        return JSONResponse(_tick_payload(app))

    # ------------------------------------------------------------------
    # Controls panel (Agent 5: scene + loudness mode)
    # ------------------------------------------------------------------

    @app.post("/api/mode")
    def set_mode(req: ModeRequest) -> JSONResponse:
        try:
            app.state.profile_agent.set_event_type(req.mode)
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse({"ok": False}, status_code=200)
        return JSONResponse({"ok": True, "scene": req.mode.value})

    @app.post("/api/loudness")
    def set_loudness(req: LoudnessRequest) -> JSONResponse:
        try:
            app.state.profile_agent.set_loudness_mode(req.loudness_mode)
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse({"ok": False}, status_code=200)
        return JSONResponse({"ok": True, "loudness_mode": req.loudness_mode.value})

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> JSONResponse:
        try:
            history = [{"role": t.role, "content": t.content} for t in req.history]
            reply, source = app.state.interaction_agent.chat(
                req.message, history=history, mix_summary=_mix_summary(app)
            )
            _log_action("chat", detail=req.message)
            return JSONResponse({
                "reply": reply,
                "source": getattr(source, "value", str(source)),
            })
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse(
                {"reply": "Sorry, I couldn't process that right now.",
                 "source": "fallback"},
                status_code=200,
            )

    @app.post("/api/focus")
    def set_focus(req: FocusRequest) -> JSONResponse:
        app.state.focus_mode = bool(req.enabled)
        _log_action("focus_on" if app.state.focus_mode else "focus_off")
        return JSONResponse({"ok": True, "focus_mode": app.state.focus_mode})

    # ------------------------------------------------------------------
    # AI prompt panel (Agent 7 -> Agent 5)
    # ------------------------------------------------------------------

    @app.post("/api/prompt")
    def submit_prompt(req: PromptRequest) -> JSONResponse:
        issue = None
        if req.issue is not None:
            issue = Issue(
                issue=req.issue, channel=req.channel,
                priority=Priority.LOW, confidence=0.0,
            )
        try:
            result = app.state.interaction_agent.handle_prompt(
                req.text, now_ms=app.state.clock(), issue=issue
            )
            _log_action(result.action.value, detail=req.text)
            return JSONResponse(result.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse(
                {"action": "instruction", "message": "Could not process that."},
                status_code=200,
            )

    # ------------------------------------------------------------------
    # Ignore action (Agent 7: temporary 3-minute suppression)
    # ------------------------------------------------------------------

    @app.post("/api/ignore")
    def ignore_issue(req: IgnoreRequest) -> JSONResponse:
        issue = Issue(
            issue=req.issue, channel=req.channel,
            priority=Priority.LOW, confidence=0.0,
        )
        try:
            result = app.state.interaction_agent.ignore(
                issue, now_ms=app.state.clock(), duration_ms=req.duration_ms
            )
            _log_action("ignore", channel=req.channel)
            return JSONResponse(result.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse(
                {"action": "ignore", "message": "Could not ignore that."},
                status_code=200,
            )

    # ------------------------------------------------------------------
    # Suggestion actions (Agent 7: More / Tell me how / Mark as normal)
    # ------------------------------------------------------------------

    @app.post("/api/more")
    def more_detail(req: IssueActionRequest) -> JSONResponse:
        try:
            result = app.state.interaction_agent.more(_issue_from(req))
            _log_action("more", channel=req.channel)
            return JSONResponse(result.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse(
                {"action": "more", "message": "Could not load more detail."},
                status_code=200,
            )

    @app.post("/api/tellmehow")
    def tell_me_how(req: IssueActionRequest) -> JSONResponse:
        try:
            result = app.state.interaction_agent.how_to(_issue_from(req))
            _log_action("tell_me_how", channel=req.channel)
            return JSONResponse(result.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse(
                {"action": "tell_me_how", "message": "Could not load the steps."},
                status_code=200,
            )

    @app.post("/api/mark-normal")
    def mark_normal(req: IssueActionRequest) -> JSONResponse:
        try:
            result = app.state.interaction_agent.mark_normal(_issue_from(req))
            _log_action("mark_normal", channel=req.channel)
            return JSONResponse(result.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse(
                {"action": "mark_normal", "message": "Could not mark that as normal."},
                status_code=200,
            )

    # ------------------------------------------------------------------
    # End-of-event profile save flow (Agent 5: summary + persist)
    # ------------------------------------------------------------------

    @app.get("/api/profile/summary")
    def profile_summary() -> JSONResponse:
        try:
            agent = app.state.profile_agent
            return JSONResponse({
                "has_unsaved_changes": bool(agent.has_unsaved_changes),
                "changes": list(agent.pending_changes()),
            })
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse({"has_unsaved_changes": False, "changes": []})

    @app.post("/api/profile/save")
    def profile_save() -> JSONResponse:
        try:
            path = app.state.profile_agent.save()
            _log_action("save_profile")
            return JSONResponse({
                "ok": True,
                "message": f"Profile saved to {path.name}.",
            })
        except Exception:  # noqa: BLE001 — failsafe (e.g. no storage configured)
            return JSONResponse(
                {"ok": False, "message": "Could not save the profile."},
                status_code=200,
            )

    # ------------------------------------------------------------------
    # Calibration mode (Agent 5: capture a "good mix" baseline)
    # ------------------------------------------------------------------

    @app.post("/api/calibrate")
    def calibrate() -> JSONResponse:
        try:
            frame = app.state.orchestrator.tick(app.state.clock())
            if frame.metrics is None:
                return JSONResponse(
                    {"ok": False, "message": "No audio to calibrate from."},
                    status_code=200,
                )
            baseline = app.state.profile_agent.calibrate(frame.metrics)
            _log_action("calibrate")
            return JSONResponse({
                "ok": True,
                "message": "Captured this mix as the good-mix baseline.",
                "baseline": baseline.model_dump(mode="json"),
            })
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse(
                {"ok": False, "message": "Could not calibrate."}, status_code=200
            )

    # ------------------------------------------------------------------
    # Logging system review (spec: post-service review)
    # ------------------------------------------------------------------

    @app.get("/api/log/recent")
    def recent_log(limit: int = 100) -> JSONResponse:
        logger = app.state.event_logger
        if logger is None:
            return JSONResponse({"records": []})
        try:
            records = [r.model_dump(mode="json") for r in logger.recent(limit)]
            return JSONResponse({"records": records})
        except Exception:  # noqa: BLE001 — failsafe
            return JSONResponse({"records": []})

    # ------------------------------------------------------------------
    # Real-time WebSocket push
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def state_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        interval = app.state.ws_interval_ms / 1000.0
        try:
            while True:
                await websocket.send_json(_tick_payload(app))
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 — failsafe: drop the socket, keep serving
            return

    return app


# ===========================================================================
# Production entry point: `uvicorn src.web_server:app`
# ===========================================================================

def build_default_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Wire a live dashboard from default/real components.

    Uses the real microphone capture (SoundDeviceAudioSource) and the standard
    agent set, configured from ``config`` (defaults to ``ServerConfig.from_env()``
    so the mixer IP and audio device come from environment variables — no code
    edits needed). The local Mistral 7B is attached automatically when
    SOUND_ADVISOR_LLM_BACKEND selects one (ollama/lmstudio); otherwise the app
    runs on Agent 4's built-in basic alerts.

    Run live with:
        uvicorn src.web_server:build_default_app --factory
    """
    from src.audio_analysis import AudioAnalysisEngine
    from src.detection import DetectionEngine
    from src.event_log import JsonlFileSink, SessionLogger
    from src.llm_client import build_llm_client
    from src.mixer_state import MixerStateAgent, UdpOscTransport
    from src.pacing import SuggestionPacingAgent
    from src.recording_analysis import RecordingAnalysisEngine
    from src.suggestion import SuggestionGenerator
    from src.system_loop import SoundDeviceAudioSource

    cfg = config or ServerConfig.from_env()

    profile_agent = ContextProfileAgent(cfg.profile_dir)
    # Local Mistral 7B (Ollama/LM Studio) when configured via env, else None
    # -> SuggestionGenerator falls back to basic alerts. Failsafe either way.
    generator = SuggestionGenerator(build_llm_client())
    interaction_agent = UserInteractionAgent(generator, profile_agent)
    # Append-only session log for post-service review (spec LOGGING SYSTEM).
    event_logger = SessionLogger(
        JsonlFileSink(Path(cfg.log_dir) / "session.jsonl")
    )

    # Optional recorder/stream feed analysis (spec RECORDING ANALYSIS MODULE):
    # only enabled when a recording input device is configured.
    recording_source = None
    recording_engine = None
    if cfg.recording_device is not None:
        recording_source = SoundDeviceAudioSource(
            sample_rate=cfg.sample_rate,
            channels=cfg.audio_channels,
            device=cfg.recording_device,
        )
        recording_engine = RecordingAnalysisEngine.default()

    orchestrator = SystemLoopOrchestrator(
        audio_source=SoundDeviceAudioSource(
            sample_rate=cfg.sample_rate,
            channels=cfg.audio_channels,
            device=cfg.audio_device,
        ),
        audio_engine=AudioAnalysisEngine.default(),
        mixer_agent=MixerStateAgent(
            UdpOscTransport(
                ip=cfg.x32_ip, port=cfg.x32_port, timeout=cfg.osc_timeout
            )
        ),
        detection_engine=DetectionEngine.default(),
        profile_agent=profile_agent,
        pacing_agent=SuggestionPacingAgent(),
        suggestion_generator=generator,
        interaction_agent=interaction_agent,
        event_logger=event_logger,
        recording_source=recording_source,
        recording_analysis_engine=recording_engine,
    )
    return create_app(
        orchestrator=orchestrator,
        profile_agent=profile_agent,
        interaction_agent=interaction_agent,
        event_logger=event_logger,
    )
