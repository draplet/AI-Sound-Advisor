"""
src/system_loop.py

System Loop Orchestrator for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — SYSTEM LOOP):

    Loop every 0.25-0.5 seconds:
      1. Capture audio data
      2. Read X32 mixer state
      3. Analyze audio (LUFS, FFT, Peaks)
      4. Detect issues
      5. Apply profile + user overrides
      6. Score priority + confidence
      7. Check pacing rules
      8. Generate suggestions (LLM)
      9. Display in UI

This module is the conductor: it owns no analysis logic of its own. It wires the
seven agents already built and runs them in the spec order, one ``LoopFrame`` per
tick, then hands that frame to the UI (step 9).

Agent integration (exactly as outlined in the spec loop):
    Step 1  AudioSource              — capture audio (injectable, like Agent 2's
                                       transport; real impl: SoundDeviceAudioSource)
    Step 2  Agent 2  MixerStateAgent — read_state() (failsafe, never raises)
    Step 3  Agent 1  AudioAnalysisEngine — analyze() -> AudioMetrics
    Step 4  Agent 3  DetectionEngine — detect() -> prioritised Issues
    Step 5  Agent 5  ContextProfileAgent — detection_profile() steers detection;
            Agent 7  UserInteractionAgent — filter_ignored() drops dismissed issues
    Step 6  Agent 3  (priority + confidence are produced inside detect())
    Step 7  Agent 6  SuggestionPacingAgent — tick() -> what to suggest now
    Step 8  Agent 4  SuggestionGenerator — generate_all() ONLY on Agent 6's output
    Step 9           LoopFrame — the structured object the UI renders

FAILSAFE — "System must never crash during service"
----------------------------------------------------
``tick`` is guaranteed never to raise. Every external interaction is wrapped:
  - Lost audio capture / analysis -> a degraded frame with a warning, no
    suggestions, and ``all_clear=False`` (the loop must never claim the mix is
    "sounding good" when it cannot actually hear it).
  - A disconnected X32 surfaces as ``mixer_connected=False`` + warning (Agent 2
    already returns this without raising); the audio path keeps working.
  - Any other unexpected error degrades to a safe frame so the live service is
    never interrupted.

PERFORMANCE — "AI (LLM) must NOT run every loop"
------------------------------------------------
Agent 4 (the LLM) is only ever called with Agent 6's ``to_suggest`` list — new
issues and issues due for a throttled repeat. On a steady issue most ticks send
nothing to the LLM, keeping it off the hot loop.

DETERMINISM
-----------
Like Agents 6/7 and the lifecycle, time is injected as ``now_ms``; the loop holds
no wall clock. ``run()`` advances a simulated clock with no real sleeps so the
whole loop is reproducible in tests. A production driver supplies real timestamps
and paces itself to ``DEFAULT_LOOP_INTERVAL_MS``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from src.audio_analysis import AudioAnalysisEngine, AudioMetrics
from src.detection import DetectionEngine, Issue
from src.event_log import SessionLogger
from src.interaction import UserInteractionAgent
from src.mixer_state import MixerStateAgent, X32_CHANNEL_COUNT
from src.pacing import SuggestionPacingAgent
from src.profile_store import ContextProfileAgent
from src.recording_analysis import RecordingAnalysis, RecordingAnalysisEngine
from src.state_manager import LoudnessMode, Scene
from src.suggestion import SuggestionGenerator, Suggestion

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Loop cadence (ms). Spec: "Loop every 0.25-0.5 seconds." We default to the
#: fast end so the UI feels responsive ("real-time updates 2-4 times/second").
DEFAULT_LOOP_INTERVAL_MS = 250

#: Warning shown when the audio device cannot be read (spec: "Audio input lost").
AUDIO_LOST_WARNING = "Audio input signal lost. Check the audio interface."

#: Warning shown when the recorder/stream feed cannot be read (spec RECORDING
#: ANALYSIS MODULE — the room analysis keeps running regardless).
RECORDING_LOST_WARNING = "Recording/stream feed lost. Check the recorder input."

#: Status shown when an unexpected error forces a degraded iteration.
DEGRADED_STATUS = "System running in degraded mode; some checks were skipped."

#: How often to health-probe the local LLM (ms). The spec wants the AI/LLM
#: status to "update in real time" but the PERFORMANCE RULES forbid touching the
#: model every loop, so we probe at most this often and cache the result.
LLM_PROBE_INTERVAL_MS = 5_000

#: Fader move (dB) between ticks that counts as an operator change (spec CHANGE
#: DETECTION: "detect user adjustment on the mixer").
CHANGE_FADER_DB = 1.0

#: Sudden integrated-loudness jump (LUFS) between ticks that counts as a change
#: in the audio (spec: "sudden change in audio").
CHANGE_LOUDNESS_LUFS = 3.0


# ===========================================================================
# Step 1 — Audio capture interface (injectable, like Agent 2's OSC transport)
# ===========================================================================

class AudioSource(ABC):
    """Interface for capturing a real-time audio buffer (spec step 1).

    ``capture`` returns ``(samples, sample_rate)`` where ``samples`` is a numpy
    buffer (mono 1-D, or 2-D ``(frames, channels)`` which Agent 1 downmixes).
    Implementations raise on device failure; the orchestrator translates that
    into a graceful "audio lost" frame rather than crashing.
    """

    @abstractmethod
    def capture(self) -> Tuple[np.ndarray, int]:
        """Return the latest ``(samples, sample_rate)`` (raises on failure)."""


class SoundDeviceAudioSource(AudioSource):
    """Real microphone/line capture via the ``sounddevice`` library.

    Records a short blocking buffer each call. ``sounddevice`` is imported
    lazily so this module loads even where the optional dependency is absent
    (mirroring Agent 2's lazy python-osc import).
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        block_ms: int = DEFAULT_LOOP_INTERVAL_MS,
        channels: int = 1,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.channels = channels
        self.device = device

    def capture(self) -> Tuple[np.ndarray, int]:
        import sounddevice as sd  # lazy: optional dependency

        frames = max(1, int(self.sample_rate * self.block_ms / 1000))
        buffer = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float64",
            device=self.device,
        )
        sd.wait()
        # Squeeze a single-channel (frames, 1) capture down to mono 1-D.
        if buffer.ndim == 2 and buffer.shape[1] == 1:
            buffer = buffer[:, 0]
        return buffer, self.sample_rate


# ===========================================================================
# Step 9 — Structured output for the UI
# ===========================================================================

class LoopFrame(BaseModel):
    """One iteration's result — everything the UI needs to render this tick.

    Spec (Web UI Agent): suggestions panel, connection-status indicators, scene
    and loudness selectors, and the "Your mix is sounding good" all-clear state
    are all derived from these fields.
    """

    tick_index: int
    now_ms: int

    # Step 3 output (None when audio was unavailable this tick).
    metrics: Optional[AudioMetrics] = None

    # Steps 4-6: the detected, profile-/override-filtered issues this tick.
    issues: List[Issue] = Field(default_factory=list)

    # Step 8: suggestions freshly generated this tick (only for paced issues).
    suggestions: List[Suggestion] = Field(default_factory=list)

    # Step 7: all-clear + status banner.
    all_clear: bool = False
    status_message: Optional[str] = None

    # Connection-status indicators (failsafe / spec status panel).
    audio_ok: bool = True
    mixer_connected: bool = True
    #: AI/LLM availability for the status indicator (None = not yet probed).
    llm_available: Optional[bool] = None
    warnings: List[str] = Field(default_factory=list)

    #: Recording/broadcast analysis vs the room (None when no recorder feed is
    #: wired). Spec RECORDING ANALYSIS MODULE.
    recording: Optional[RecordingAnalysis] = None

    # Current scene / loudness (spec controls panel).
    scene: Scene = Scene.SERMON
    loudness_mode: LoudnessMode = LoudnessMode.CONSERVATIVE

    #: Whether a calibration baseline has been captured (spec CALIBRATION MODE).
    calibrated: bool = False


# ===========================================================================
# The orchestrator
# ===========================================================================

class SystemLoopOrchestrator:
    """Runs the 9-step system loop, integrating all seven agents.

    Every dependency is injected so the loop is fully testable with in-memory
    fakes — no hardware, no network, no LLM, no real clock.
    """

    def __init__(
        self,
        audio_source: AudioSource,
        audio_engine: AudioAnalysisEngine,
        mixer_agent: MixerStateAgent,
        detection_engine: DetectionEngine,
        profile_agent: ContextProfileAgent,
        pacing_agent: SuggestionPacingAgent,
        suggestion_generator: SuggestionGenerator,
        interaction_agent: UserInteractionAgent,
        channels: Optional[Iterable[int]] = None,
        event_logger: Optional["SessionLogger"] = None,
        recording_source: Optional[AudioSource] = None,
        recording_analysis_engine: Optional[RecordingAnalysisEngine] = None,
    ) -> None:
        self._audio_source = audio_source
        self._audio_engine = audio_engine
        self._mixer_agent = mixer_agent
        self._detection_engine = detection_engine
        self._profile_agent = profile_agent
        self._pacing_agent = pacing_agent
        self._suggestion_generator = suggestion_generator
        self._interaction_agent = interaction_agent
        #: Optional logging system (spec LOGGING SYSTEM); None disables logging.
        self._event_logger = event_logger
        #: Optional recorder/stream feed + its analysis engine (spec RECORDING
        #: ANALYSIS MODULE). Both must be set for recording analysis to run.
        self._recording_source = recording_source
        self._recording_analysis_engine = recording_analysis_engine
        self._channels = (
            list(channels) if channels is not None
            else list(range(1, X32_CHANNEL_COUNT + 1))
        )
        self._tick_index = 0
        self._last_frame: Optional[LoopFrame] = None
        # Cached AI/LLM health (probed at most every LLM_PROBE_INTERVAL_MS).
        self._llm_available: Optional[bool] = None
        self._llm_probed_ms: Optional[int] = None
        # Previous mixer/audio state for automatic change detection.
        self._prev_channels: dict = {}
        self._prev_loudness: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_frame(self) -> Optional[LoopFrame]:
        """The most recent frame produced (None before the first tick)."""
        return self._last_frame

    def _detect_user_changes(self, mixer, metrics, now_ms: int) -> None:
        """Pause suggestions briefly after an operator move or sudden audio change.

        Compares this tick's mixer faders/mutes and integrated loudness against
        the previous tick. A fader move (>= ``CHANGE_FADER_DB``) or a mute toggle
        pauses just that channel; a sudden loudness jump (>= ``CHANGE_LOUDNESS_LUFS``)
        pauses globally. The first tick (no baseline) never triggers. Failsafe.
        """
        try:
            channels = getattr(mixer, "channels", {}) or {}
            for key, channel in channels.items():
                previous = self._prev_channels.get(key)
                if previous is not None:
                    fader_moved = abs(channel.fader - previous[0]) >= CHANGE_FADER_DB
                    mute_toggled = channel.mute != previous[1]
                    if fader_moved or mute_toggled:
                        self._pacing_agent.notify_user_change(
                            now_ms, channel=channel.name
                        )
            self._prev_channels = {
                key: (channel.fader, channel.mute)
                for key, channel in channels.items()
            }

            if metrics is not None:
                if (
                    self._prev_loudness is not None
                    and abs(metrics.loudness_lufs - self._prev_loudness)
                    >= CHANGE_LOUDNESS_LUFS
                ):
                    self._pacing_agent.notify_user_change(now_ms, channel=None)
                self._prev_loudness = metrics.loudness_lufs
        except Exception:  # noqa: BLE001 — failsafe: detection never crashes the loop
            pass

    def _analyze_recording(
        self, room_metrics: AudioMetrics, warnings: List[str]
    ) -> Optional[RecordingAnalysis]:
        """Capture and analyse the recorder/stream feed against the room.

        Returns None when no recorder feed is wired. If the feed is wired but
        cannot be read this tick, returns an ``unavailable`` analysis and adds a
        warning — the room analysis (the main loop) is never affected. Failsafe.
        """
        if self._recording_source is None or self._recording_analysis_engine is None:
            return None
        try:
            samples, sample_rate = self._recording_source.capture()
            recording_metrics = self._audio_engine.analyze(samples, sample_rate)
            return self._recording_analysis_engine.analyze(
                room_metrics, recording_metrics
            )
        except Exception:  # noqa: BLE001 — failsafe: feed lost, keep the room running
            if RECORDING_LOST_WARNING not in warnings:
                warnings.append(RECORDING_LOST_WARNING)
            return RecordingAnalysis(
                available=False, findings=[], room_metrics=room_metrics,
                recording_metrics=None,
            )

    def _log_frame_safe(self, frame: LoopFrame, now_ms: int) -> None:
        """Hand the frame to the logging system, if one is wired (never raises)."""
        if self._event_logger is None:
            return
        try:
            self._event_logger.log_frame(frame, now_ms)
        except Exception:  # noqa: BLE001 — failsafe: logging never crashes the loop
            pass

    def _refresh_llm_available(self, now_ms: int) -> Optional[bool]:
        """Return cached AI/LLM availability, re-probing only past the interval.

        Honours "AI (LLM) must NOT run every loop": the health probe runs at
        most once per ``LLM_PROBE_INTERVAL_MS``; other ticks reuse the cache.
        Never raises — a failing probe leaves the previous value in place.
        """
        last = self._llm_probed_ms
        if last is None or (now_ms - last) >= LLM_PROBE_INTERVAL_MS:
            try:
                self._llm_available = self._suggestion_generator.llm_available()
            except Exception:  # noqa: BLE001 — failsafe: keep last known value
                pass
            self._llm_probed_ms = now_ms
        return self._llm_available

    def notify_user_change(
        self,
        now_ms: int,
        channel: Optional[str] = None,
        pause_ms: Optional[int] = None,
    ) -> None:
        """Tell Agent 6 the operator just made a change (spec: pause 2-4 s).

        Pass-through to the pacing agent so the UI can register a mixer move
        through the orchestrator it already holds.
        """
        self._pacing_agent.notify_user_change(
            now_ms, channel=channel, pause_ms=pause_ms
        )

    def tick(self, now_ms: int) -> LoopFrame:
        """Run one full iteration of the system loop.

        FAILSAFE: never raises. Any failure degrades to a safe ``LoopFrame``.
        """
        self._tick_index += 1
        try:
            frame = self._run_tick(now_ms)
        except Exception as exc:  # noqa: BLE001 — failsafe: never crash during service
            frame = self._degraded_frame(now_ms, exc)
            self._last_frame = frame
        self._log_frame_safe(frame, now_ms)
        return frame

    def run(
        self,
        ticks: int,
        interval_ms: int = DEFAULT_LOOP_INTERVAL_MS,
        start_ms: int = 0,
    ) -> List[LoopFrame]:
        """Run ``ticks`` iterations over a *simulated* clock (no real sleeps).

        Deterministic helper for tests and offline replay. A production driver
        instead calls :meth:`tick` on a real timer paced to ``interval_ms``.
        """
        frames: List[LoopFrame] = []
        now = start_ms
        for _ in range(ticks):
            frames.append(self.tick(now))
            now += interval_ms
        return frames

    # ------------------------------------------------------------------
    # The 9 steps
    # ------------------------------------------------------------------

    def _run_tick(self, now_ms: int) -> LoopFrame:
        # --- Step 2: read X32 mixer state (Agent 2 — already failsafe) -------
        mixer = self._mixer_agent.read_state(self._channels)
        warnings = list(mixer.warnings)

        # --- Step 1: capture audio data --------------------------------------
        try:
            samples, sample_rate = self._audio_source.capture()
            # --- Step 3: analyze audio (LUFS / FFT / Peaks) (Agent 1) --------
            metrics = self._audio_engine.analyze(samples, sample_rate)
        except Exception as exc:  # noqa: BLE001 — audio lost: warn, keep running
            frame = self._audio_lost_frame(now_ms, mixer, warnings, exc)
            self._last_frame = frame
            return frame

        # --- Step 5a: apply learned profile (Agent 5) ------------------------
        profile = self._profile_agent.detection_profile()

        # --- Steps 4 + 6: detect issues, scored by priority + confidence -----
        issues = self._detection_engine.detect(metrics, mixer, profile)

        # --- Step 5b: apply user overrides — temporary ignores (Agent 7) -----
        issues = self._interaction_agent.filter_ignored(issues, now_ms)

        # --- Change detection: an operator move / sudden audio change pauses
        #     the affected suggestions briefly (spec CHANGE DETECTION). -------
        self._detect_user_changes(mixer, metrics, now_ms)

        # --- Step 7: pacing rules (Agent 6) ----------------------------------
        decision = self._pacing_agent.tick(issues, now_ms)

        # --- Step 8: generate suggestions — LLM ONLY on paced issues (Agent 4)
        suggestions = self._suggestion_generator.generate_all(decision.to_suggest)

        # --- Recording / broadcast analysis vs the room (optional module) ----
        recording = self._analyze_recording(metrics, warnings)

        # --- Step 9: assemble the frame for the UI ---------------------------
        frame = LoopFrame(
            tick_index=self._tick_index,
            now_ms=now_ms,
            metrics=metrics,
            issues=issues,
            suggestions=suggestions,
            all_clear=decision.all_clear,
            status_message=decision.status_message,
            audio_ok=True,
            mixer_connected=mixer.connected,
            llm_available=self._refresh_llm_available(now_ms),
            warnings=warnings,
            scene=self._safe_scene(),
            loudness_mode=self._safe_loudness(),
            calibrated=self._safe_calibrated(),
            recording=recording,
        )
        self._last_frame = frame
        return frame

    # ------------------------------------------------------------------
    # Degraded-frame builders (failsafe)
    # ------------------------------------------------------------------

    def _audio_lost_frame(
        self, now_ms: int, mixer, warnings: List[str], exc: Exception
    ) -> LoopFrame:
        """Audio capture/analysis failed: warn, suggest nothing, never claim
        the mix is good. The mixer (and its connection state) is still known."""
        out = list(warnings)
        if AUDIO_LOST_WARNING not in out:
            out.append(AUDIO_LOST_WARNING)
        return LoopFrame(
            tick_index=self._tick_index,
            now_ms=now_ms,
            metrics=None,
            issues=[],
            suggestions=[],
            all_clear=False,
            status_message=AUDIO_LOST_WARNING,
            audio_ok=False,
            mixer_connected=bool(getattr(mixer, "connected", False)),
            llm_available=self._llm_available,
            warnings=out,
            scene=self._safe_scene(),
            loudness_mode=self._safe_loudness(),
            calibrated=self._safe_calibrated(),
        )

    def _degraded_frame(self, now_ms: int, exc: Exception) -> LoopFrame:
        """Last-resort safe frame for any unexpected loop failure."""
        return LoopFrame(
            tick_index=self._tick_index,
            now_ms=now_ms,
            metrics=None,
            issues=[],
            suggestions=[],
            all_clear=False,
            status_message=DEGRADED_STATUS,
            audio_ok=False,
            mixer_connected=False,
            llm_available=self._llm_available,
            warnings=[f"Loop iteration error ({type(exc).__name__}: {exc})."],
            scene=self._safe_scene(),
            loudness_mode=self._safe_loudness(),
            calibrated=self._safe_calibrated(),
        )

    # ------------------------------------------------------------------
    # Safe state accessors (never raise even if the profile agent is faulty)
    # ------------------------------------------------------------------

    def _safe_scene(self) -> Scene:
        try:
            return self._profile_agent.active.event_type
        except Exception:  # noqa: BLE001 — failsafe
            return Scene.SERMON

    def _safe_loudness(self) -> LoudnessMode:
        try:
            return self._profile_agent.active.loudness_mode
        except Exception:  # noqa: BLE001 — failsafe
            return LoudnessMode.CONSERVATIVE

    def _safe_calibrated(self) -> bool:
        try:
            return bool(self._profile_agent.is_calibrated)
        except Exception:  # noqa: BLE001 — failsafe
            return False
