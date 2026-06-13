"""
tests/test_system_loop.py

AI Sound Advisor — System Loop Orchestrator
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — SYSTEM LOOP):
  Loop every 0.25-0.5 seconds:
    1. Capture audio data            (AudioSource — new injectable input)
    2. Read X32 mixer state          (Agent 2: MixerStateAgent)
    3. Analyze audio (LUFS/FFT/Peaks)(Agent 1: AudioAnalysisEngine)
    4. Detect issues                 (Agent 3: DetectionEngine)
    5. Apply profile + user overrides(Agent 5: ContextProfileAgent + Agent 7)
    6. Score priority + confidence   (Agent 3)
    7. Check pacing rules            (Agent 6: SuggestionPacingAgent)
    8. Generate suggestions (LLM)    (Agent 4: SuggestionGenerator)
    9. Display in UI                 (LoopFrame output object)

Cross-cutting spec requirements exercised here:
  - "System must never crash during service" (FAILSAFE — tick never raises)
  - "If Audio input lost -> show warning" / "If X32 disconnects -> show warning"
  - Performance rule: "AI (LLM) must NOT run every loop — only on new/changed
    issues" (the orchestrator hands ONLY Agent 6's `to_suggest` to Agent 4)

Time is driven by an explicit simulated clock (now_ms), exactly like
src/lifecycle.py, src/pacing.py and src/interaction.py — no real sleeps, fully
deterministic.

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.system_loop  ->  SystemLoopOrchestrator, AudioSource, LoopFrame,
                       DEFAULT_LOOP_INTERVAL_MS

Run with:
  pytest tests/test_system_loop.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.audio_analysis import AudioAnalysisEngine, AudioMetrics, EnergyLevel
from src.detection import DetectionEngine, IssueType, Priority
from src.interaction import UserInteractionAgent
from src.mixer_state import MixerStateAgent, X32OscTransport
from src.pacing import ALL_CLEAR_MESSAGE, SuggestionPacingAgent
from src.profile_store import ContextProfileAgent
from src.suggestion import SuggestionGenerator
from src.state_manager import LoudnessMode, Scene

from src.system_loop import (
    SystemLoopOrchestrator,
    AudioSource,
    LoopFrame,
    DEFAULT_LOOP_INTERVAL_MS,
)


# ===========================================================================
# Test doubles
# ===========================================================================

class FakeX32Transport(X32OscTransport):
    """In-memory X32 returning one configurable channel (default: a lowered
    'Lead Vocal'), or raising to simulate a disconnected mixer."""

    def __init__(self, name="Lead Vocal", fader=0.25, on=1, gain=0.5, fail=False):
        self.name = name
        self.fader = fader      # 0.25 -> -30 dB (well below the prominence line)
        self.on = on            # 1 == audible (unmuted)
        self.gain = gain
        self.fail = fail

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
    """Duck-typed stand-in for Agent 1 returning fixed metrics (deterministic
    detection) and counting analysis calls."""

    def __init__(self, metrics: AudioMetrics):
        self.metrics = metrics
        self.calls = 0

    def analyze(self, samples, sample_rate) -> AudioMetrics:
        self.calls += 1
        return self.metrics


class FakeAudioSource(AudioSource):
    """In-memory capture source; can simulate a lost audio device."""

    def __init__(self, samples=None, sample_rate=48_000, fail=False):
        self.samples = np.zeros(2_048) if samples is None else samples
        self.sample_rate = sample_rate
        self.fail = fail
        self.calls = 0

    def capture(self):
        self.calls += 1
        if self.fail:
            raise OSError("audio device disconnected")
        return self.samples, self.sample_rate


class SpyGenerator(SuggestionGenerator):
    """Agent 4 with no LLM (deterministic fallbacks) that records how many
    issues it was asked to turn into suggestions — proves LLM gating."""

    def __init__(self):
        super().__init__(client=None)
        self.generated_counts = []

    def generate_all(self, issues):
        self.generated_counts.append(len(issues))
        return super().generate_all(issues)


# ===========================================================================
# Metric helpers
# ===========================================================================

def metrics(loudness=-22.0, peak=-10.0,
            low_mid=EnergyLevel.NORMAL, high=EnergyLevel.NORMAL) -> AudioMetrics:
    return AudioMetrics(
        loudness_lufs=loudness, peak_level=peak,
        low_mid_energy=low_mid, high_freq_energy=high,
    )


def masking_metrics() -> AudioMetrics:
    # Heavy low-mid (music) energy -> vocal-masking detector fires when a
    # lowered vocal channel is present on the mixer.
    return metrics(low_mid=EnergyLevel.HIGH)


def clipping_metrics() -> AudioMetrics:
    return metrics(peak=0.0)  # at full scale -> clipping


def clean_metrics() -> AudioMetrics:
    return metrics()  # nothing fires -> all clear


# ===========================================================================
# Wiring helper — 6 REAL agents + a stub Agent 1 (or a real one when asked)
# ===========================================================================

def make(audio_metrics: AudioMetrics = None,
         transport: FakeX32Transport = None,
         audio_source: FakeAudioSource = None,
         real_engine: bool = False,
         detection_engine=None,
         channels=(3,),
         audio_engine=None,
         recording_source=None,
         recording_engine=None):
    transport = transport or FakeX32Transport()
    audio_source = audio_source or FakeAudioSource()

    if audio_engine is None:
        if real_engine:
            audio_engine = AudioAnalysisEngine.default()
        else:
            audio_engine = StubAudioEngine(audio_metrics or clean_metrics())

    mixer_agent = MixerStateAgent(transport)
    detection = detection_engine or DetectionEngine.default()
    profile_agent = ContextProfileAgent.in_memory("test")
    pacing_agent = SuggestionPacingAgent()
    generator = SpyGenerator()
    interaction_agent = UserInteractionAgent(generator, profile_agent)

    orch = SystemLoopOrchestrator(
        audio_source=audio_source,
        audio_engine=audio_engine,
        mixer_agent=mixer_agent,
        detection_engine=detection,
        profile_agent=profile_agent,
        pacing_agent=pacing_agent,
        suggestion_generator=generator,
        interaction_agent=interaction_agent,
        channels=channels,
        recording_source=recording_source,
        recording_analysis_engine=recording_engine,
    )
    return SimpleNamespace(
        orch=orch, transport=transport, audio_source=audio_source,
        audio_engine=audio_engine, mixer_agent=mixer_agent,
        detection=detection, profile_agent=profile_agent,
        pacing_agent=pacing_agent, generator=generator,
        interaction_agent=interaction_agent,
        recording_source=recording_source,
    )


def _issue_types(frame: LoopFrame):
    return {i.issue for i in frame.issues}


# ===========================================================================
# SECTION 1 — A single tick runs all 9 steps and returns a LoopFrame
# ===========================================================================

class TestTickHappyPath:

    def test_tick_returns_a_loopframe(self):
        env = make(clean_metrics())
        frame = env.orch.tick(now_ms=1_000)
        assert isinstance(frame, LoopFrame)

    def test_tick_captures_audio_and_reads_mixer(self):
        env = make(clean_metrics())
        env.orch.tick(now_ms=1_000)
        assert env.audio_source.calls == 1          # step 1
        assert env.audio_engine.calls == 1          # step 3

    def test_frame_exposes_health_and_state(self):
        env = make(clean_metrics())
        frame = env.orch.tick(now_ms=1_000)
        assert frame.audio_ok is True
        assert frame.mixer_connected is True
        assert frame.scene == Scene.SERMON
        assert frame.loudness_mode == LoudnessMode.CONSERVATIVE
        assert frame.now_ms == 1_000
        assert frame.tick_index == 1

    def test_detected_issue_flows_into_suggestions(self):
        env = make(masking_metrics())
        frame = env.orch.tick(now_ms=1_000)
        assert IssueType.VOCAL_MASKING in _issue_types(frame)
        assert any(s.issue == IssueType.VOCAL_MASKING for s in frame.suggestions)
        assert frame.all_clear is False

    def test_clipping_issue_detected_end_to_end(self):
        env = make(clipping_metrics())
        frame = env.orch.tick(now_ms=1_000)
        assert IssueType.CLIPPING in _issue_types(frame)
        assert any(s.issue == IssueType.CLIPPING for s in frame.suggestions)

    def test_rendered_suggestion_uses_spec_output_format(self):
        env = make(masking_metrics())
        frame = env.orch.tick(now_ms=1_000)
        rendered = frame.suggestions[0].render()
        assert "Confidence:" in rendered


# ===========================================================================
# SECTION 2 — No issues => "Your mix is sounding good"
# ===========================================================================

class TestAllClear:

    def test_clean_mix_reports_all_clear(self):
        env = make(clean_metrics())
        frame = env.orch.tick(now_ms=1_000)
        assert frame.all_clear is True
        assert frame.status_message == ALL_CLEAR_MESSAGE
        assert frame.suggestions == []
        assert frame.issues == []


# ===========================================================================
# SECTION 3 — Performance rule: LLM (Agent 4) does NOT run every loop
# ===========================================================================

class TestLLMGating:

    def test_llm_only_runs_on_new_and_repeat_boundaries(self):
        """A steady single issue ticked once per second for 20s should be sent
        to Agent 4 exactly twice (new + one 15s repeat) — not every loop.

        Clipping metrics yield exactly one issue, isolating the gating count."""
        env = make(clipping_metrics())
        for t in range(0, 21):
            env.orch.tick(now_ms=t * 1_000)
        total_sent_to_llm = sum(env.generator.generated_counts)
        assert total_sent_to_llm == 2

    def test_clean_loop_never_calls_llm(self):
        env = make(clean_metrics())
        for t in range(0, 5):
            env.orch.tick(now_ms=t * 1_000)
        assert sum(env.generator.generated_counts) == 0


# ===========================================================================
# SECTION 4 — FAILSAFE: the loop must never crash during service
# ===========================================================================

class TestFailsafeAudioLost:

    def test_lost_audio_does_not_raise_and_warns(self):
        env = make(clean_metrics(), audio_source=FakeAudioSource(fail=True))
        frame = env.orch.tick(now_ms=1_000)          # must not raise
        assert frame.audio_ok is False
        assert frame.suggestions == []
        assert frame.all_clear is False              # NOT "sounding good"
        assert any("audio" in w.lower() for w in frame.warnings)

    def test_audio_engine_failure_is_contained(self):
        class BrokenEngine:
            calls = 0
            def analyze(self, samples, sample_rate):
                raise ValueError("bad buffer")
        env = make(clean_metrics())
        env.orch._audio_engine = BrokenEngine()      # swap in a faulty Agent 1
        frame = env.orch.tick(now_ms=1_000)
        assert isinstance(frame, LoopFrame)
        assert frame.audio_ok is False


class TestFailsafeMixerDisconnected:

    def test_disconnected_mixer_warns_but_loop_continues(self):
        env = make(clean_metrics(), transport=FakeX32Transport(fail=True))
        frame = env.orch.tick(now_ms=1_000)          # must not raise
        assert frame.mixer_connected is False
        assert frame.warnings                        # at least one warning
        assert frame.audio_ok is True                # audio path still works


class TestFailsafeNeverRaises:

    def test_any_internal_failure_degrades_gracefully(self):
        class BrokenDetection:
            def detect(self, audio, mixer, profile=None):
                raise RuntimeError("detector exploded")
        env = make(masking_metrics(), detection_engine=BrokenDetection())
        frame = env.orch.tick(now_ms=1_000)          # must not raise
        assert isinstance(frame, LoopFrame)
        assert frame.suggestions == []
        assert frame.status_message is not None


# ===========================================================================
# SECTION 5 — Agent 7 integration: temporary user "ignore"
# ===========================================================================

class TestUserIgnoreIntegration:

    def test_ignored_issue_is_filtered_from_the_frame(self):
        env = make(masking_metrics())
        first = env.orch.tick(now_ms=0)
        masking = next(i for i in first.issues
                       if i.issue == IssueType.VOCAL_MASKING)

        env.interaction_agent.ignore(masking, now_ms=0)   # user dismisses it

        frame = env.orch.tick(now_ms=1_000)
        assert IssueType.VOCAL_MASKING not in _issue_types(frame)

    def test_ignore_expires_and_issue_returns(self):
        env = make(masking_metrics())
        first = env.orch.tick(now_ms=0)
        masking = next(i for i in first.issues
                       if i.issue == IssueType.VOCAL_MASKING)
        env.interaction_agent.ignore(masking, now_ms=0, duration_ms=180_000)

        env.orch.tick(now_ms=1_000)                       # still ignored
        late = env.orch.tick(now_ms=200_000)              # past the 3-min window
        assert IssueType.VOCAL_MASKING in _issue_types(late)


# ===========================================================================
# SECTION 6 — Agent 5 integration: profile "mark as normal" suppression
# ===========================================================================

class TestProfileOverrideIntegration:

    def test_marked_normal_issue_is_suppressed(self):
        env = make(masking_metrics())
        env.profile_agent.mark_normal(IssueType.VOCAL_MASKING)
        frame = env.orch.tick(now_ms=1_000)
        assert IssueType.VOCAL_MASKING not in _issue_types(frame)

    def test_scene_and_loudness_changes_reflect_in_frame(self):
        env = make(clean_metrics())
        env.profile_agent.set_event_type(Scene.WORSHIP)
        env.profile_agent.set_loudness_mode(LoudnessMode.ENERGETIC)
        frame = env.orch.tick(now_ms=1_000)
        assert frame.scene == Scene.WORSHIP
        assert frame.loudness_mode == LoudnessMode.ENERGETIC


# ===========================================================================
# SECTION 7 — Agent 6 integration: post-change pause via the orchestrator
# ===========================================================================

class TestUserChangePauseIntegration:

    def test_notify_user_change_suppresses_during_pause(self):
        env = make(masking_metrics())
        # Isolate the channel-scoped vocal-masking issue (heavy low-mid energy
        # also legitimately raises a mix-wide eq_mud issue, which a *channel*
        # pause would not cover) by treating muddiness as normal here.
        env.profile_agent.mark_normal(IssueType.EQ_MUD)
        env.orch.notify_user_change(now_ms=1_000, channel="Lead Vocal")
        frame = env.orch.tick(now_ms=1_500)          # inside the 3s pause
        assert frame.suggestions == []

    def test_suggestions_resume_after_pause(self):
        env = make(masking_metrics())
        env.profile_agent.mark_normal(IssueType.EQ_MUD)
        env.orch.notify_user_change(now_ms=1_000, channel="Lead Vocal")
        env.orch.tick(now_ms=1_500)                  # paused
        frame = env.orch.tick(now_ms=5_000)          # pause expired
        assert any(s.issue == IssueType.VOCAL_MASKING for s in frame.suggestions)


# ===========================================================================
# SECTION 8 — Deterministic run() over a simulated clock
# ===========================================================================

class TestDeterministicRun:

    def test_run_advances_simulated_clock_without_sleeping(self):
        env = make(clean_metrics())
        frames = env.orch.run(ticks=3, interval_ms=250, start_ms=0)
        assert [f.now_ms for f in frames] == [0, 250, 500]
        assert [f.tick_index for f in frames] == [1, 2, 3]

    def test_default_loop_interval_is_within_spec_window(self):
        # Spec: "Loop every 0.25-0.5 seconds".
        assert 250 <= DEFAULT_LOOP_INTERVAL_MS <= 500


# ===========================================================================
# SECTION 9 — Real Agent 1 integration (no stub)
# ===========================================================================

class TestRealAudioEngineIntegration:

    def test_real_engine_runs_through_the_loop(self):
        sr = 48_000
        t = np.arange(sr // 4) / sr
        buffer = 0.3 * np.sin(2 * np.pi * 220.0 * t)   # a real tone
        env = make(real_engine=True,
                   audio_source=FakeAudioSource(samples=buffer, sample_rate=sr))
        frame = env.orch.tick(now_ms=1_000)
        assert frame.audio_ok is True
        assert frame.metrics is not None
        assert isinstance(frame.metrics, AudioMetrics)


# ===========================================================================
# SECTION 10 — AI/LLM availability indicator
# (spec CONNECTION STATUS INDICATOR: "AI/LLM availability ... update in real time")
# ===========================================================================

class FakeHealthClient:
    """A stand-in LLM client exposing a health probe and counting probes."""

    def __init__(self, available=True):
        self.available = available
        self.probes = 0

    def generate(self, prompt):
        return "ok"

    def is_available(self):
        self.probes += 1
        return self.available


def _make_with_client(client):
    """Wire an orchestrator whose Agent 4 uses ``client``."""
    env = make(clean_metrics())
    env.orch._suggestion_generator._client = client
    return env


class TestLLMAvailabilityIndicator:

    def test_no_client_reports_unavailable(self):
        env = make(clean_metrics())              # SpyGenerator has no client
        frame = env.orch.tick(now_ms=1_000)
        assert frame.llm_available is False

    def test_reachable_client_reports_available(self):
        env = _make_with_client(FakeHealthClient(available=True))
        frame = env.orch.tick(now_ms=1_000)
        assert frame.llm_available is True

    def test_unreachable_client_reports_unavailable(self):
        env = _make_with_client(FakeHealthClient(available=False))
        frame = env.orch.tick(now_ms=1_000)
        assert frame.llm_available is False

    def test_probe_is_cached_not_run_every_tick(self):
        # PERFORMANCE: must not health-probe the model on every 250ms loop.
        client = FakeHealthClient(available=True)
        env = _make_with_client(client)
        env.orch.run(ticks=10, interval_ms=250, start_ms=0)
        assert client.probes < 10               # cached between probes

    def test_probe_refreshes_after_interval(self):
        client = FakeHealthClient(available=True)
        env = _make_with_client(client)
        env.orch.tick(now_ms=0)
        env.orch.tick(now_ms=60_000)            # well past the probe interval
        assert client.probes >= 2


# ===========================================================================
# SECTION 11 — Automatic change detection
# (spec CHANGE DETECTION: "pause associated suggestions 2-4s after a change")
# ===========================================================================

class TestChangeDetection:

    def test_no_pause_on_the_first_tick(self):
        env = make(masking_metrics())
        env.orch.tick(now_ms=1_000)
        # Nothing to compare against yet -> no spurious pause.
        assert env.pacing_agent.is_paused("Lead Vocal", 1_000) is False

    def test_fader_move_pauses_that_channel(self):
        env = make(masking_metrics())
        env.orch.tick(now_ms=1_000)             # baseline (fader 0.25)
        env.transport.fader = 0.75              # operator pushes the fader up
        env.orch.tick(now_ms=1_250)
        assert env.pacing_agent.is_paused("Lead Vocal", 1_250) is True

    def test_mute_toggle_pauses_that_channel(self):
        env = make(masking_metrics())
        env.orch.tick(now_ms=1_000)
        env.transport.on = 0                    # operator mutes the channel
        env.orch.tick(now_ms=1_250)
        assert env.pacing_agent.is_paused("Lead Vocal", 1_250) is True

    def test_sudden_loudness_jump_pauses_globally(self):
        env = make(clean_metrics())
        env.orch.tick(now_ms=1_000)             # baseline (-22 LUFS)
        env.audio_engine.metrics = metrics(loudness=-8.0)   # big sudden jump
        env.orch.tick(now_ms=1_250)
        assert env.pacing_agent.is_paused(None, 1_250) is True

    def test_steady_state_does_not_pause(self):
        env = make(masking_metrics())
        env.orch.tick(now_ms=1_000)
        env.orch.tick(now_ms=1_250)             # nothing changed
        assert env.pacing_agent.is_paused("Lead Vocal", 1_250) is False


# ===========================================================================
# SECTION 12 — Recording / Broadcast Analysis integration
# (spec RECORDING ANALYSIS MODULE: compare the recorder feed against the room)
# ===========================================================================

from src.recording_analysis import RecordingAnalysisEngine


class FakeRecordingSource(AudioSource):
    """Second audio input: the recorder/stream feed. ``level`` tags the buffer
    so a TaggedEngine can return distinct metrics for room vs recording."""

    def __init__(self, level=1.0, fail=False):
        self.level = level
        self.fail = fail
        self.calls = 0

    def capture(self):
        self.calls += 1
        if self.fail:
            raise OSError("recorder feed lost")
        return np.full(2_048, self.level), 48_000


class TaggedEngine:
    """Returns ``recording`` metrics for non-zero buffers (the feed) and
    ``room`` metrics for zero buffers (the room source)."""

    def __init__(self, room, recording):
        self.room = room
        self.recording = recording

    def analyze(self, samples, sample_rate):
        first = float(np.asarray(samples).reshape(-1)[0])
        return self.recording if first != 0.0 else self.room


class TestRecordingAnalysisIntegration:

    def test_no_recording_source_means_no_recording_field(self):
        env = make(clean_metrics())
        frame = env.orch.tick(now_ms=1_000)
        assert frame.recording is None

    def test_recording_feed_is_analysed_against_the_room(self):
        # Room at -23 LUFS, recorder feed 12 dB louder -> a loudness imbalance.
        engine = TaggedEngine(room=metrics(loudness=-23.0),
                              recording=metrics(loudness=-11.0))
        env = make(audio_engine=engine,
                   recording_source=FakeRecordingSource(level=1.0),
                   recording_engine=RecordingAnalysisEngine.default())
        frame = env.orch.tick(now_ms=1_000)
        assert frame.recording is not None
        assert frame.recording.available is True
        assert len(frame.recording.findings) >= 1

    def test_matched_feed_is_all_clear(self):
        engine = TaggedEngine(room=metrics(loudness=-20.0),
                              recording=metrics(loudness=-20.0))
        env = make(audio_engine=engine,
                   recording_source=FakeRecordingSource(level=1.0),
                   recording_engine=RecordingAnalysisEngine.default())
        frame = env.orch.tick(now_ms=1_000)
        assert frame.recording.available is True
        assert frame.recording.all_clear is True

    def test_lost_recording_feed_is_failsafe(self):
        env = make(clean_metrics(),
                   recording_source=FakeRecordingSource(fail=True),
                   recording_engine=RecordingAnalysisEngine.default())
        frame = env.orch.tick(now_ms=1_000)
        # The loop still produces a frame; the room analysis is unaffected.
        assert frame.audio_ok is True
        assert frame.recording is not None
        assert frame.recording.available is False
        assert any("record" in w.lower() for w in frame.warnings)
