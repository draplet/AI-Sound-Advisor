"""
tests/test_detection.py

AI Sound Advisor — Agent 3: Detection Engine
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — AGENT 3):
  - "Combine audio + mixer data to detect sound issues"
  - Detect: Feedback risk, Vocal masking, Clipping, Loudness issues,
            EQ problems (mud, harshness)
  - "Assign priority" / "Calculate confidence score"
  - Output Example:
        {"issue": "vocal_masking", "channel": "Lead Vocal",
         "priority": "high", "confidence": 0.9}
  - Confidence: High (>0.8), Medium (0.5-0.8), Low (<0.5)
  - Priority order: 1 Feedback, 2 Vocal clarity (Sermon highest, Worship next),
                    3 Clipping, 4 Loudness, 5 Balance

Inputs are produced by Agent 1 (AudioMetrics) and Agent 2 (MixerState); the
Detection Engine combines them with Profile data. No real hardware involved.

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.detection  →  DetectionEngine, IssueDetector, Issue, Profile,
                    IssueType, Priority, ConfidenceLevel, confidence_level,
                    spec_rank, priority_for, and the concrete detectors.

Run with:
  pytest tests/test_detection.py -v
"""

import pytest

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.mixer_state import ChannelState, MixerState
from src.state_manager import Scene, LoudnessMode

from src.detection import (
    DetectionEngine,
    IssueDetector,
    Issue,
    Profile,
    IssueType,
    Priority,
    ConfidenceLevel,
    confidence_level,
    spec_rank,
    priority_for,
    ClippingDetector,
    LoudnessDetector,
    FeedbackDetector,
    EqMudDetector,
    EqHarshnessDetector,
    VocalMaskingDetector,
)


# ===========================================================================
# Builders — assemble Agent 1 / Agent 2 outputs for the engine
# ===========================================================================

def audio(
    loudness_lufs=-23.0,
    peak_level=-12.0,
    low_mid_energy=EnergyLevel.NORMAL,
    high_freq_energy=EnergyLevel.NORMAL,
):
    return AudioMetrics(
        loudness_lufs=loudness_lufs,
        peak_level=peak_level,
        low_mid_energy=low_mid_energy,
        high_freq_energy=high_freq_energy,
    )


def mixer_with_channel(name="Lead Vocal", index=3, fader=-10.0, mute=False):
    return MixerState(
        connected=True,
        channels={
            f"channel_{index}": ChannelState(
                index=index, name=name, fader=fader, mute=mute, gain=12.0
            )
        },
    )


def empty_mixer(connected=True):
    return MixerState(connected=connected, channels={})


# ===========================================================================
# SECTION 1 — confidence_level categorisation (spec thresholds)
# ===========================================================================

class TestConfidenceLevel:

    @pytest.mark.parametrize(
        "score, expected",
        [
            (0.95, ConfidenceLevel.HIGH),
            (0.81, ConfidenceLevel.HIGH),
            (0.80, ConfidenceLevel.MEDIUM),   # not > 0.8
            (0.50, ConfidenceLevel.MEDIUM),
            (0.4999, ConfidenceLevel.LOW),
            (0.0, ConfidenceLevel.LOW),
        ],
    )
    def test_confidence_level_thresholds(self, score, expected):
        assert confidence_level(score) == expected


# ===========================================================================
# SECTION 2 — Issue model + priority helpers
# ===========================================================================

class TestIssueModelAndPriority:

    def test_issue_exposes_spec_fields(self):
        issue = Issue(
            issue=IssueType.VOCAL_MASKING,
            channel="Lead Vocal",
            channel_index=1,
            priority=Priority.HIGH,
            confidence=0.9,
        )
        dumped = issue.model_dump()
        # channel_index is optional (None for main-mix issues) and carries the
        # 1-based channel number so the UI can show an OSC-style tag.
        assert set(dumped.keys()) == {
            "issue", "channel", "channel_index", "priority", "confidence",
        }
        assert dumped["channel_index"] == 1

    def test_issue_confidence_label_matches_score(self):
        issue = Issue(
            issue=IssueType.CLIPPING,
            channel=None,
            priority=Priority.HIGH,
            confidence=0.95,
        )
        assert issue.confidence_label == ConfidenceLevel.HIGH

    def test_feedback_is_critical_priority(self):
        assert priority_for(IssueType.FEEDBACK, Scene.SERMON) == Priority.CRITICAL

    def test_feedback_outranks_everything(self):
        ranks = [
            spec_rank(t, Scene.SERMON)
            for t in (
                IssueType.VOCAL_MASKING,
                IssueType.CLIPPING,
                IssueType.LOUDNESS,
            )
        ]
        assert all(spec_rank(IssueType.FEEDBACK, Scene.SERMON) > r for r in ranks)

    def test_vocal_clarity_ranks_higher_in_sermon_than_worship(self):
        """Spec: 'Sermon clarity highest, Worship vocals next.'"""
        assert spec_rank(IssueType.VOCAL_MASKING, Scene.SERMON) > spec_rank(
            IssueType.VOCAL_MASKING, Scene.WORSHIP
        )

    def test_spec_priority_ordering_vocal_above_clipping_above_loudness(self):
        s = Scene.SERMON
        assert (
            spec_rank(IssueType.VOCAL_MASKING, s)
            > spec_rank(IssueType.CLIPPING, s)
            > spec_rank(IssueType.LOUDNESS, s)
        )


# ===========================================================================
# SECTION 3 — ClippingDetector  (Spec rank 3)
# ===========================================================================

class TestClippingDetector:

    def test_clipping_detected_near_full_scale(self):
        issue = ClippingDetector().detect(
            _ctx(audio(peak_level=-0.5))
        )
        assert issue is not None
        assert issue.issue == IssueType.CLIPPING
        assert issue.priority == Priority.HIGH
        assert issue.confidence > 0.8  # High confidence

    def test_no_clipping_well_below_threshold(self):
        assert ClippingDetector().detect(_ctx(audio(peak_level=-12.0))) is None


# ===========================================================================
# SECTION 4 — LoudnessDetector  (Spec rank 4; loudness-mode aware)
# ===========================================================================

class TestLoudnessDetector:

    def test_too_loud_for_conservative_mode(self):
        ctx = _ctx(audio(loudness_lufs=-5.0),
                   profile=Profile(loudness_mode=LoudnessMode.CONSERVATIVE))
        issue = LoudnessDetector().detect(ctx)
        assert issue is not None and issue.issue == IssueType.LOUDNESS

    def test_on_target_conservative_is_no_issue(self):
        ctx = _ctx(audio(loudness_lufs=-23.0),
                   profile=Profile(loudness_mode=LoudnessMode.CONSERVATIVE))
        assert LoudnessDetector().detect(ctx) is None

    def test_too_quiet_is_flagged(self):
        ctx = _ctx(audio(loudness_lufs=-45.0),
                   profile=Profile(loudness_mode=LoudnessMode.CONSERVATIVE))
        assert LoudnessDetector().detect(ctx) is not None

    def test_energetic_mode_tolerates_louder_signal(self):
        """A level that is 'too loud' under Conservative is fine under Energetic."""
        loud = audio(loudness_lufs=-14.0)
        assert LoudnessDetector().detect(
            _ctx(loud, profile=Profile(loudness_mode=LoudnessMode.ENERGETIC))
        ) is None
        assert LoudnessDetector().detect(
            _ctx(loud, profile=Profile(loudness_mode=LoudnessMode.CONSERVATIVE))
        ) is not None


# ===========================================================================
# SECTION 5 — EQ detectors (mud / harshness)
# ===========================================================================

class TestEqDetectors:

    def test_mud_detected_on_high_low_mid_energy(self):
        issue = EqMudDetector().detect(_ctx(audio(low_mid_energy=EnergyLevel.HIGH)))
        assert issue is not None and issue.issue == IssueType.EQ_MUD

    def test_no_mud_on_normal_low_mid(self):
        assert EqMudDetector().detect(
            _ctx(audio(low_mid_energy=EnergyLevel.NORMAL))
        ) is None

    def test_harshness_detected_on_high_high_freq_energy(self):
        issue = EqHarshnessDetector().detect(
            _ctx(audio(high_freq_energy=EnergyLevel.HIGH))
        )
        assert issue is not None and issue.issue == IssueType.EQ_HARSHNESS

    def test_no_harshness_on_normal_high_freq(self):
        assert EqHarshnessDetector().detect(
            _ctx(audio(high_freq_energy=EnergyLevel.NORMAL))
        ) is None


# ===========================================================================
# SECTION 6 — FeedbackDetector  (Spec rank 1, critical)
# ===========================================================================

class TestFeedbackDetector:

    def test_feedback_risk_on_sustained_loud_high_freq(self):
        issue = FeedbackDetector().detect(
            _ctx(audio(high_freq_energy=EnergyLevel.HIGH, peak_level=-1.0))
        )
        assert issue is not None
        assert issue.issue == IssueType.FEEDBACK
        assert issue.priority == Priority.CRITICAL

    def test_no_feedback_when_high_freq_quiet(self):
        assert FeedbackDetector().detect(
            _ctx(audio(high_freq_energy=EnergyLevel.HIGH, peak_level=-20.0))
        ) is None

    def test_no_feedback_when_high_freq_normal(self):
        assert FeedbackDetector().detect(
            _ctx(audio(high_freq_energy=EnergyLevel.NORMAL, peak_level=-1.0))
        ) is None


# ===========================================================================
# SECTION 7 — VocalMaskingDetector  (flagship: combines audio + mixer)
# ===========================================================================

class TestVocalMaskingDetector:

    def test_masking_detected_when_vocal_low_and_music_heavy(self):
        """SPEC PRIMARY example: vocals buried under music."""
        ctx = _ctx(
            audio(low_mid_energy=EnergyLevel.HIGH),
            mixer=mixer_with_channel(name="Lead Vocal", fader=-20.0),
        )
        issue = VocalMaskingDetector().detect(ctx)

        assert issue is not None
        assert issue.issue == IssueType.VOCAL_MASKING
        assert issue.channel == "Lead Vocal"
        assert issue.confidence > 0.8  # clearly buried => high confidence
        assert issue.priority in (Priority.HIGH, Priority.CRITICAL)

    def test_no_masking_when_music_energy_normal(self):
        ctx = _ctx(
            audio(low_mid_energy=EnergyLevel.NORMAL),
            mixer=mixer_with_channel(name="Lead Vocal", fader=-20.0),
        )
        assert VocalMaskingDetector().detect(ctx) is None

    def test_no_masking_when_vocal_is_prominent(self):
        ctx = _ctx(
            audio(low_mid_energy=EnergyLevel.HIGH),
            mixer=mixer_with_channel(name="Lead Vocal", fader=2.0),
        )
        assert VocalMaskingDetector().detect(ctx) is None

    def test_no_masking_when_vocal_channel_muted(self):
        ctx = _ctx(
            audio(low_mid_energy=EnergyLevel.HIGH),
            mixer=mixer_with_channel(name="Lead Vocal", fader=-20.0, mute=True),
        )
        assert VocalMaskingDetector().detect(ctx) is None

    def test_no_masking_when_no_vocal_channel_present(self):
        ctx = _ctx(
            audio(low_mid_energy=EnergyLevel.HIGH),
            mixer=mixer_with_channel(name="Kick Drum", fader=-20.0),
        )
        assert VocalMaskingDetector().detect(ctx) is None


# ===========================================================================
# SECTION 8 — DetectionEngine aggregation
# ===========================================================================

class TestDetectionEngine:

    def test_default_engine_has_six_detectors(self):
        assert len(DetectionEngine.default().detectors) == 6

    def test_no_issues_returns_empty_list(self):
        engine = DetectionEngine.default()
        clean = audio(
            loudness_lufs=-23.0,
            peak_level=-20.0,
            low_mid_energy=EnergyLevel.NORMAL,
            high_freq_energy=EnergyLevel.NORMAL,
        )
        assert engine.detect(clean, empty_mixer()) == []

    def test_issues_are_sorted_by_priority(self):
        """Feedback, then vocal masking, then clipping must lead the list."""
        engine = DetectionEngine.default()
        a = audio(
            loudness_lufs=-5.0,                  # loudness issue
            peak_level=-0.5,                     # clipping + feedback peak
            low_mid_energy=EnergyLevel.HIGH,     # mud + vocal masking
            high_freq_energy=EnergyLevel.HIGH,   # harshness + feedback
        )
        m = mixer_with_channel(name="Lead Vocal", fader=-20.0)

        issues = engine.detect(a, m)  # default profile => Scene.SERMON
        types = [i.issue for i in issues]

        assert types[0] == IssueType.FEEDBACK
        assert types[1] == IssueType.VOCAL_MASKING
        assert types[2] == IssueType.CLIPPING

    def test_spec_example_is_reproduced(self):
        """Reproduces the AGENT 3 output example structure."""
        engine = DetectionEngine.default()
        a = audio(low_mid_energy=EnergyLevel.HIGH)
        m = mixer_with_channel(name="Lead Vocal", fader=-18.0)

        masking = [i for i in engine.detect(a, m) if i.issue == IssueType.VOCAL_MASKING]
        assert len(masking) == 1
        issue = masking[0]
        assert issue.channel == "Lead Vocal"
        assert issue.priority == Priority.HIGH
        assert 0.0 < issue.confidence <= 1.0

    def test_profile_can_suppress_an_issue_type(self):
        """Spec: 'Apply profile + user overrides.'"""
        engine = DetectionEngine.default()
        a = audio(peak_level=-0.5)  # would normally clip
        profile = Profile(suppressed_issues=[IssueType.CLIPPING])

        issues = engine.detect(a, empty_mixer(), profile)
        assert all(i.issue != IssueType.CLIPPING for i in issues)

    def test_engine_works_with_disconnected_mixer(self):
        """Audio-only detectors still run when the X32 is unavailable."""
        engine = DetectionEngine.default()
        a = audio(peak_level=-0.5)  # clipping from audio alone
        issues = engine.detect(a, empty_mixer(connected=False))
        assert any(i.issue == IssueType.CLIPPING for i in issues)

    def test_engine_never_crashes_if_a_detector_raises(self):
        """Failsafe: one faulty detector must not crash the whole pass."""

        class BoomDetector(IssueDetector):
            @property
            def name(self):
                return "boom"

            def detect(self, context):
                raise RuntimeError("detector exploded")

        engine = DetectionEngine([BoomDetector(), ClippingDetector()])
        a = audio(peak_level=-0.5)

        issues = engine.detect(a, empty_mixer())
        assert any(i.issue == IssueType.CLIPPING for i in issues)


# ===========================================================================
# Local helper: build a DetectionContext for single-detector tests
# ===========================================================================

def _ctx(audio_metrics, mixer=None, profile=None):
    from src.detection import DetectionContext

    return DetectionContext(
        audio=audio_metrics,
        mixer=mixer if mixer is not None else empty_mixer(),
        profile=profile if profile is not None else Profile(),
    )
