"""
tests/test_profile_agent.py

AI Sound Advisor — Agent 5: Context & Profile Agent
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — AGENT 5):
  - "Manage profiles (load/save)"
  - "Store user instructions and overrides"
  - "Track 'normal' conditions"
  - "Apply learned behavior"
  - Profile Contents: event type, room size/characteristics, acoustic sources,
                      user overrides (e.g. "piano is normally loud")
  - Data Storage: "JSON files for profiles and logs (V1)"
  - Profile Management: real-time temporary overrides; end-of-event prompt to
                        save updates + "Show summary of learned changes"
  - Calibration Mode: capture LUFS / frequency distribution / balance baseline

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.profile_store  →  ContextProfileAgent, EnvironmentProfile,
                        CalibrationBaseline, RoomSize

Run with:
  pytest tests/test_profile_agent.py -v
"""

import pytest

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.detection import IssueType, Profile as DetectionProfile
from src.state_manager import Scene, LoudnessMode

from src.profile_store import (
    ContextProfileAgent,
    EnvironmentProfile,
    CalibrationBaseline,
    RoomSize,
)


def sample_audio(
    loudness_lufs=-18.0,
    peak_level=-3.0,
    low_mid_energy=EnergyLevel.NORMAL,
    high_freq_energy=EnergyLevel.NORMAL,
):
    return AudioMetrics(
        loudness_lufs=loudness_lufs,
        peak_level=peak_level,
        low_mid_energy=low_mid_energy,
        high_freq_energy=high_freq_energy,
    )


# ===========================================================================
# SECTION 1 — EnvironmentProfile model + defaults
# ===========================================================================

class TestEnvironmentProfile:

    def test_profile_defaults(self):
        profile = EnvironmentProfile(name="default")
        assert profile.event_type == Scene.SERMON
        assert profile.loudness_mode == LoudnessMode.CONSERVATIVE
        assert profile.acoustic_sources == []
        assert profile.user_instructions == []
        assert profile.overrides == {}
        assert profile.suppressed_issues == []
        assert profile.calibration is None

    def test_profile_holds_spec_contents(self):
        profile = EnvironmentProfile(
            name="main-room",
            event_type=Scene.WORSHIP,
            room_size=RoomSize.LARGE,
            room_characteristics="reverberant stone hall",
            acoustic_sources=["drums", "piano"],
            overrides={"piano": "normally loud"},
        )
        assert profile.room_size == RoomSize.LARGE
        assert "piano" in profile.acoustic_sources
        assert profile.overrides["piano"] == "normally loud"


# ===========================================================================
# SECTION 2 — Apply learned behavior: build the detection Profile
# ===========================================================================

class TestDetectionProfileMapping:

    def test_detection_profile_maps_scene_and_loudness(self):
        profile = EnvironmentProfile(
            name="x", event_type=Scene.WORSHIP, loudness_mode=LoudnessMode.ENERGETIC
        )
        detection = profile.detection_profile()
        assert isinstance(detection, DetectionProfile)
        assert detection.scene == Scene.WORSHIP
        assert detection.loudness_mode == LoudnessMode.ENERGETIC

    def test_detection_profile_carries_suppressed_issues(self):
        profile = EnvironmentProfile(
            name="x", suppressed_issues=[IssueType.BALANCE]
        )
        assert IssueType.BALANCE in profile.detection_profile().suppressed_issues


# ===========================================================================
# SECTION 3 — CalibrationBaseline (capture a "good mix" reference)
# ===========================================================================

class TestCalibrationBaseline:

    def test_baseline_from_audio_metrics(self):
        metrics = sample_audio(
            loudness_lufs=-16.0,
            low_mid_energy=EnergyLevel.HIGH,
            high_freq_energy=EnergyLevel.NORMAL,
        )
        baseline = CalibrationBaseline.from_audio(metrics)
        assert baseline.loudness_lufs == -16.0
        assert baseline.low_mid_energy == EnergyLevel.HIGH
        assert baseline.high_freq_energy == EnergyLevel.NORMAL


# ===========================================================================
# SECTION 4 — Persistence: JSON load / save (V1 storage)
# ===========================================================================

class TestPersistence:

    def test_save_writes_a_json_file(self, tmp_path):
        agent = ContextProfileAgent(tmp_path)
        agent.active.name = "evening-service"

        path = agent.save()

        assert path.exists()
        assert path.suffix == ".json"

    def test_save_then_load_round_trips_fields(self, tmp_path):
        agent = ContextProfileAgent(tmp_path)
        agent.active.name = "main"
        agent.set_event_type(Scene.PLAY)
        agent.set_loudness_mode(LoudnessMode.BALANCED)
        agent.add_acoustic_source("drums")
        agent.set_override("piano", "normally loud")
        agent.save()

        reloaded = ContextProfileAgent(tmp_path).load("main")

        assert reloaded.event_type == Scene.PLAY
        assert reloaded.loudness_mode == LoudnessMode.BALANCED
        assert "drums" in reloaded.acoustic_sources
        assert reloaded.overrides["piano"] == "normally loud"

    def test_load_missing_profile_raises(self, tmp_path):
        agent = ContextProfileAgent(tmp_path)
        with pytest.raises(FileNotFoundError):
            agent.load("does-not-exist")

    def test_load_or_create_returns_default_when_missing(self, tmp_path):
        agent = ContextProfileAgent(tmp_path)
        profile = agent.load_or_create("fresh")
        assert isinstance(profile, EnvironmentProfile)
        assert profile.name == "fresh"

    def test_list_profiles_returns_saved_names(self, tmp_path):
        agent = ContextProfileAgent(tmp_path)
        agent.active.name = "a"
        agent.save()
        agent.active.name = "b"
        agent.save()

        names = ContextProfileAgent(tmp_path).list_profiles()
        assert set(names) == {"a", "b"}


# ===========================================================================
# SECTION 5 — Real-time learning + change tracking
# ===========================================================================

class TestRealtimeLearning:

    def test_add_user_instruction_is_stored_and_tracked(self):
        agent = ContextProfileAgent.in_memory()
        agent.add_user_instruction("This is a quiet service")

        assert "This is a quiet service" in agent.active.user_instructions
        assert agent.has_unsaved_changes is True
        assert any("quiet service" in c for c in agent.pending_changes())

    def test_mark_normal_suppresses_issue_in_detection_profile(self):
        """Spec: 'Mark condition as normal' -> learned behavior applied."""
        agent = ContextProfileAgent.in_memory()
        agent.mark_normal(IssueType.LOUDNESS)

        assert IssueType.LOUDNESS in agent.active.suppressed_issues
        assert IssueType.LOUDNESS in agent.detection_profile().suppressed_issues

    def test_set_override_records_change(self):
        agent = ContextProfileAgent.in_memory()
        agent.set_override("piano", "normally loud")
        assert agent.active.overrides["piano"] == "normally loud"
        assert agent.has_unsaved_changes is True

    def test_saving_clears_pending_changes(self, tmp_path):
        agent = ContextProfileAgent(tmp_path)
        agent.add_user_instruction("Ignore choir balance for now")
        assert agent.has_unsaved_changes is True

        agent.save()

        assert agent.has_unsaved_changes is False
        assert agent.pending_changes() == []

    def test_loading_resets_change_tracking(self, tmp_path):
        agent = ContextProfileAgent(tmp_path)
        agent.active.name = "svc"
        agent.save()

        other = ContextProfileAgent(tmp_path)
        other.add_user_instruction("temp note")  # dirty
        other.load("svc")

        assert other.has_unsaved_changes is False


# ===========================================================================
# SECTION 6 — Calibration mode
# ===========================================================================

class TestCalibration:

    def test_calibrate_captures_baseline_from_audio(self):
        agent = ContextProfileAgent.in_memory()
        assert agent.is_calibrated is False

        agent.calibrate(sample_audio(loudness_lufs=-15.0,
                                     low_mid_energy=EnergyLevel.HIGH))

        assert agent.is_calibrated is True
        assert agent.active.calibration is not None
        assert agent.active.calibration.loudness_lufs == -15.0
        assert agent.has_unsaved_changes is True


# ===========================================================================
# SECTION 7 — End-of-event summary
# ===========================================================================

class TestEndOfEventSummary:

    def test_summary_lists_all_learned_changes(self, tmp_path):
        """Spec: end of event 'Show summary of learned changes'."""
        agent = ContextProfileAgent(tmp_path)
        agent.set_event_type(Scene.WORSHIP)
        agent.add_user_instruction("The piano is always loud")
        agent.mark_normal(IssueType.EQ_MUD)

        changes = agent.pending_changes()

        assert len(changes) >= 3
        assert agent.has_unsaved_changes is True
