"""
tests/test_core.py

AI Sound Advisor — Core Stability Layer
TDD Test Suite (written BEFORE application code exists)

Specification Reference:
  - Failsafe: "System must never crash during service"
  - Issue Lifecycle: detected → active (after 500ms) → resolved/ignored
  - State Manager: "thread-safe registry of: current_scene, loudness_state,
                    active_issues, ignored_issues"

Modules under test (DO NOT EXIST YET — that is intentional):
  src.setup_assistant  →  SystemSetupAssistant, HardwareWarning,
                          ConnectionStatus, SetupReport
  src.lifecycle        →  IssueLifecycle, IssueState
  src.state_manager    →  SystemStateManager, Scene, LoudnessMode

Run with:
  pytest tests/test_core.py -v
"""

import threading
import pytest

# ---------------------------------------------------------------------------
# Imports from modules that do not yet exist.
# These WILL raise ImportError until the application code is written.
# That is the correct TDD red state.
# ---------------------------------------------------------------------------
from src.setup_assistant import (
    SystemSetupAssistant,
    HardwareWarning,
    ConnectionStatus,
    SetupReport,
)
from src.lifecycle import IssueLifecycle, IssueState
from src.state_manager import SystemStateManager, Scene, LoudnessMode


# ===========================================================================
# SECTION 1 — SystemSetupAssistant
#
# Specification requirements covered:
#   • "If X32 disconnects → show warning"
#   • "If Audio input lost → show warning"
#   • "System must never crash during service"
#   • "Continue operating where possible"
#   • "Connection Status Indicator: Must update in real time"
# ===========================================================================

class TestSystemSetupAssistant:
    """
    Tests that the SystemSetupAssistant correctly generates HardwareWarning
    payloads for missing hardware connections and that the process NEVER
    raises an unhandled exception regardless of hardware state.
    """

    # ------------------------------------------------------------------
    # SPEC TEST 1 (PRIMARY) — X32 disconnect generates a warning payload
    # ------------------------------------------------------------------

    def test_x32_disconnect_generates_hardware_warning(self):
        """
        SPEC TEST 1 (Primary Requirement):
        When the mock X32 hardware flag is False, verify_connections() must:
          1. Return a SetupReport whose x32_status equals DISCONNECTED.
          2. Include at least one HardwareWarning whose component field
             identifies the X32.
          3. Never raise an unhandled exception (covered by the call itself
             completing without error).
        """
        assistant = SystemSetupAssistant(
            x32_connected=False,
            audio_input_connected=True,
        )

        report = assistant.verify_connections()

        # Connection status must be DISCONNECTED for the X32
        assert report.x32_status == ConnectionStatus.DISCONNECTED

        # At least one warning must be present
        assert len(report.warnings) >= 1

        # Exactly one warning must identify the X32 component
        x32_warnings = [w for w in report.warnings if w.component == "X32"]
        assert len(x32_warnings) == 1, (
            "Expected exactly one HardwareWarning with component='X32', "
            f"got: {[w.component for w in report.warnings]}"
        )

        # System must mark itself as still operational (not crashed/halted)
        assert report.is_operational is True

    def test_x32_disconnect_does_not_raise_exception(self):
        """
        SPEC TEST 1 (No-Crash Guarantee):
        Isolated from the assertion logic above so that a crash failure is
        reported as a distinct, unambiguous test failure rather than buried
        inside an assert block.
        """
        assistant = SystemSetupAssistant(
            x32_connected=False,
            audio_input_connected=True,
        )

        try:
            assistant.verify_connections()
        except Exception as exc:
            pytest.fail(
                f"verify_connections() raised an unexpected exception when "
                f"X32 is disconnected. Exception: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Audio input disconnect
    # ------------------------------------------------------------------

    def test_audio_input_disconnect_generates_hardware_warning(self):
        """
        Spec: "If Audio input lost → show warning."
        When audio_input_connected=False, the report must contain a
        HardwareWarning identifying the AudioInput component and the
        system must stay operational.
        """
        assistant = SystemSetupAssistant(
            x32_connected=True,
            audio_input_connected=False,
        )

        report = assistant.verify_connections()

        assert report.audio_status == ConnectionStatus.DISCONNECTED

        audio_warnings = [w for w in report.warnings if w.component == "AudioInput"]
        assert len(audio_warnings) == 1, (
            "Expected exactly one HardwareWarning with component='AudioInput', "
            f"got: {[w.component for w in report.warnings]}"
        )

        assert report.is_operational is True

    def test_audio_input_disconnect_does_not_raise_exception(self):
        """
        No-crash guarantee mirrored for audio input loss.
        """
        assistant = SystemSetupAssistant(
            x32_connected=True,
            audio_input_connected=False,
        )

        try:
            assistant.verify_connections()
        except Exception as exc:
            pytest.fail(
                f"verify_connections() raised an unexpected exception when "
                f"AudioInput is disconnected. Exception: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Clean (fully connected) state
    # ------------------------------------------------------------------

    def test_all_connected_produces_no_warnings(self):
        """
        When all hardware flags are True, the SetupReport must contain
        zero warnings and both status fields must be CONNECTED.
        """
        assistant = SystemSetupAssistant(
            x32_connected=True,
            audio_input_connected=True,
        )

        report = assistant.verify_connections()

        assert report.x32_status == ConnectionStatus.CONNECTED
        assert report.audio_status == ConnectionStatus.CONNECTED
        assert len(report.warnings) == 0
        assert report.is_operational is True

    # ------------------------------------------------------------------
    # Both components disconnected simultaneously
    # ------------------------------------------------------------------

    def test_both_disconnected_generates_two_distinct_warnings(self):
        """
        When both hardware flags are False, the report must contain one
        warning per component. The system must still not crash.
        """
        assistant = SystemSetupAssistant(
            x32_connected=False,
            audio_input_connected=False,
        )

        report = assistant.verify_connections()

        assert report.x32_status == ConnectionStatus.DISCONNECTED
        assert report.audio_status == ConnectionStatus.DISCONNECTED

        warned_components = {w.component for w in report.warnings}
        assert "X32" in warned_components
        assert "AudioInput" in warned_components

        # Process must remain alive
        assert report.is_operational is True

    # ------------------------------------------------------------------
    # HardwareWarning payload structure (Pydantic validation)
    # ------------------------------------------------------------------

    def test_hardware_warning_payload_has_required_fields(self):
        """
        Spec: Pydantic is used for data validation.
        A HardwareWarning instance must expose non-empty 'component'
        and 'message' string fields.
        """
        assistant = SystemSetupAssistant(
            x32_connected=False,
            audio_input_connected=True,
        )
        report = assistant.verify_connections()

        warning = report.warnings[0]

        assert isinstance(warning, HardwareWarning)
        assert isinstance(warning.component, str) and warning.component != ""
        assert isinstance(warning.message, str) and warning.message != ""

    def test_setup_report_is_a_pydantic_validated_object(self):
        """
        SetupReport must be a structured, validated object (Pydantic model).
        Verify that accessing its fields does not raise AttributeError.
        """
        assistant = SystemSetupAssistant(
            x32_connected=True,
            audio_input_connected=True,
        )

        report = assistant.verify_connections()

        # All four required fields from the spec must exist
        _ = report.x32_status
        _ = report.audio_status
        _ = report.warnings
        _ = report.is_operational


# ===========================================================================
# SECTION 2 — IssueLifecycle State Machine
#
# Specification requirements covered:
#   • "Issue must persist for X ms before showing" (mocked at 500ms)
#   • "Issue must clear for X ms before removal"
#   • States: detected, active, improving, resolved, ignored
#   • "An issue cannot move from detected to active unless an internal
#     persistence timer (mocked at 500ms) has expired"
# ===========================================================================

class TestIssueLifecycle:
    """
    Tests the IssueLifecycle state machine using simulated clock ticks
    via advance_time(elapsed_ms). No real-time sleeps are used.

    State machine under test:
        DETECTED ──(500ms)──► ACTIVE ──► IMPROVING
                                    └──► RESOLVED
                                    └──► IGNORED
    """

    PERSISTENCE_THRESHOLD_MS = 500  # Spec: "mocked at 500ms"

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def test_newly_created_issue_starts_in_detected_state(self):
        """
        Every new IssueLifecycle instance must begin in the DETECTED state.
        """
        issue = IssueLifecycle(issue_id="test_001", issue_type="vocal_masking")

        assert issue.state == IssueState.DETECTED

    # ------------------------------------------------------------------
    # Persistence timer: below threshold
    # ------------------------------------------------------------------

    def test_issue_stays_detected_before_threshold_is_crossed(self):
        """
        SPEC TEST 2 (Partial — pre-threshold):
        Advancing time by 300ms (< 500ms threshold) must NOT transition
        the issue out of DETECTED.
        """
        issue = IssueLifecycle(issue_id="test_002", issue_type="clipping")

        issue.advance_time(elapsed_ms=300)

        assert issue.state == IssueState.DETECTED, (
            f"Issue transitioned to {issue.state} before 500ms threshold. "
            "The persistence gate is not being enforced."
        )

    # ------------------------------------------------------------------
    # Persistence timer: at exact threshold
    # ------------------------------------------------------------------

    def test_issue_becomes_active_at_exact_threshold(self):
        """
        SPEC TEST 2 (Partial — at threshold):
        Advancing time to exactly 500ms must transition the issue to ACTIVE.
        """
        issue = IssueLifecycle(issue_id="test_003", issue_type="clipping")

        issue.advance_time(elapsed_ms=self.PERSISTENCE_THRESHOLD_MS)

        assert issue.state == IssueState.ACTIVE, (
            f"Issue did not transition to ACTIVE at the 500ms threshold. "
            f"Current state: {issue.state}"
        )

    # ------------------------------------------------------------------
    # SPEC TEST 2 (PRIMARY) — Full lifecycle: detected → active → resolved
    # ------------------------------------------------------------------

    def test_full_lifecycle_detected_to_active_to_resolved(self):
        """
        SPEC TEST 2 (Primary Requirement):
        Simulated clock ticks must drive a complete lifecycle:
            DETECTED → ACTIVE → RESOLVED
        """
        issue = IssueLifecycle(issue_id="test_004", issue_type="vocal_masking")

        # ── Step 1: Verify initial DETECTED state ──────────────────────
        assert issue.state == IssueState.DETECTED

        # ── Step 2: Advance past threshold → expect ACTIVE ─────────────
        issue.advance_time(elapsed_ms=600)

        assert issue.state == IssueState.ACTIVE, (
            f"Expected ACTIVE after 600ms, got {issue.state}"
        )

        # ── Step 3: Resolve the issue → expect RESOLVED ─────────────────
        issue.resolve()

        assert issue.state == IssueState.RESOLVED, (
            f"Expected RESOLVED after resolve(), got {issue.state}"
        )

    # ------------------------------------------------------------------
    # Cumulative tick accumulation
    # ------------------------------------------------------------------

    def test_cumulative_advance_time_calls_cross_threshold(self):
        """
        Multiple advance_time() calls with smaller increments must
        accumulate to cross the 500ms threshold correctly.

        Scenario: 250ms + 300ms = 550ms > 500ms → must reach ACTIVE.
        """
        issue = IssueLifecycle(issue_id="test_005", issue_type="feedback_risk")

        # First tick: 250ms total — must still be DETECTED
        issue.advance_time(elapsed_ms=250)
        assert issue.state == IssueState.DETECTED, (
            "Issue should still be DETECTED at 250ms cumulative."
        )

        # Second tick: 300ms more (total 550ms) — must now be ACTIVE
        issue.advance_time(elapsed_ms=300)
        assert issue.state == IssueState.ACTIVE, (
            f"Expected ACTIVE at 550ms cumulative, got {issue.state}"
        )

    # ------------------------------------------------------------------
    # Ignored state
    # ------------------------------------------------------------------

    def test_active_issue_can_be_set_to_ignored(self):
        """
        An ACTIVE issue must transition to IGNORED when ignore() is called.
        """
        issue = IssueLifecycle(issue_id="test_006", issue_type="loudness")
        issue.advance_time(elapsed_ms=600)  # Push to ACTIVE
        assert issue.state == IssueState.ACTIVE

        issue.ignore()

        assert issue.state == IssueState.IGNORED

    # ------------------------------------------------------------------
    # Terminal state: RESOLVED must not re-activate
    # ------------------------------------------------------------------

    def test_resolved_issue_stays_resolved_on_further_time_advances(self):
        """
        Once an issue is RESOLVED, further advance_time() calls must
        NOT push it back to ACTIVE. RESOLVED is a terminal state.
        """
        issue = IssueLifecycle(issue_id="test_007", issue_type="eq_harshness")
        issue.advance_time(elapsed_ms=600)
        issue.resolve()
        assert issue.state == IssueState.RESOLVED

        # Additional time advancement — must not revert state
        issue.advance_time(elapsed_ms=1000)

        assert issue.state == IssueState.RESOLVED, (
            "RESOLVED issue incorrectly transitioned to another state "
            f"after further time advances. Got: {issue.state}"
        )

    # ------------------------------------------------------------------
    # elapsed_ms tracking
    # ------------------------------------------------------------------

    def test_elapsed_ms_accumulates_correctly_across_multiple_ticks(self):
        """
        The total elapsed_ms must accurately reflect the sum of all
        advance_time() calls made since creation.
        """
        issue = IssueLifecycle(issue_id="test_008", issue_type="eq_mud")

        issue.advance_time(elapsed_ms=200)
        issue.advance_time(elapsed_ms=150)
        issue.advance_time(elapsed_ms=75)

        assert issue.elapsed_ms == 425, (
            f"Expected elapsed_ms=425, got {issue.elapsed_ms}"
        )


# ===========================================================================
# SECTION 3 — SystemStateManager
#
# Specification requirements covered:
#   • "thread-safe registry of: current_scene, loudness_state,
#     active_issues, ignored_issues"
#   • "System detects or allows selection: Sermon / Worship / Play / Custom"
#   • "Loudness Modes: Conservative / Balanced / Energetic"
#   • "Store temporary overrides" / suppression of ignored issues
# ===========================================================================

class TestSystemStateManager:
    """
    Tests the SystemStateManager's thread-safe state registry, scene/mode
    management, and correct suppression of ignored issues.
    """

    # ------------------------------------------------------------------
    # Default initialization
    # ------------------------------------------------------------------

    def test_state_manager_initializes_with_correct_defaults(self):
        """
        SystemStateManager defaults must match the system's startup spec:
          - current_scene:  Scene.SERMON   (highest priority in spec)
          - loudness_mode:  LoudnessMode.CONSERVATIVE  (spec default)
          - active_issues:  empty dict
          - ignored_issues: empty dict
        """
        manager = SystemStateManager()

        assert manager.current_scene == Scene.SERMON
        assert manager.loudness_mode == LoudnessMode.CONSERVATIVE
        assert manager.active_issues == {}
        assert manager.ignored_issues == {}

    # ------------------------------------------------------------------
    # add_issue
    # ------------------------------------------------------------------

    def test_add_issue_registers_issue_in_active_issues(self):
        """
        Calling add_issue() must place the issue_id key into active_issues.
        """
        manager = SystemStateManager()

        manager.add_issue(
            issue_id="issue_001",
            issue_type="vocal_masking",
            channel="Lead Vocal",
        )

        assert "issue_001" in manager.active_issues

    def test_add_issue_stores_correct_metadata(self):
        """
        The stored active issue entry must contain the correct type and
        channel fields that were passed to add_issue().
        """
        manager = SystemStateManager()

        manager.add_issue(
            issue_id="issue_002",
            issue_type="clipping",
            channel="Drum Bus",
        )

        entry = manager.active_issues["issue_002"]
        assert entry["issue_type"] == "clipping"
        assert entry["channel"] == "Drum Bus"

    # ------------------------------------------------------------------
    # SPEC TEST 3 (PRIMARY) — Ignored issues are suppressed
    # ------------------------------------------------------------------

    def test_ignored_issue_is_removed_from_active_issues(self):
        """
        SPEC TEST 3 (Primary Requirement):
        After ignore_issue() is called, the issue_id must:
          1. No longer appear in active_issues.
          2. Appear in ignored_issues.
        """
        manager = SystemStateManager()

        # Arrange: add issue to active registry
        manager.add_issue(
            issue_id="issue_003",
            issue_type="clipping",
            channel="Drum Bus",
        )
        assert "issue_003" in manager.active_issues

        # Act: ignore the issue
        manager.ignore_issue("issue_003")

        # Assert: suppressed from active
        assert "issue_003" not in manager.active_issues, (
            "Ignored issue must be removed from active_issues immediately."
        )

        # Assert: registered in ignored
        assert "issue_003" in manager.ignored_issues, (
            "Ignored issue must be present in ignored_issues registry."
        )

    def test_ignored_issue_is_not_re_added_to_active_on_duplicate_add(self):
        """
        Spec: Ignored issues must remain suppressed.
        If add_issue() is called again with the same issue_id that has
        already been ignored, the issue must NOT reappear in active_issues.
        """
        manager = SystemStateManager()

        manager.add_issue(
            issue_id="issue_004",
            issue_type="feedback_risk",
            channel="Monitor 1",
        )
        manager.ignore_issue("issue_004")

        # Attempt to re-add the same ignored issue_id
        manager.add_issue(
            issue_id="issue_004",
            issue_type="feedback_risk",
            channel="Monitor 1",
        )

        assert "issue_004" not in manager.active_issues, (
            "A currently ignored issue must not re-enter active_issues "
            "via a duplicate add_issue() call."
        )
        assert "issue_004" in manager.ignored_issues

    # ------------------------------------------------------------------
    # resolve_issue
    # ------------------------------------------------------------------

    def test_resolve_issue_removes_it_from_active_issues(self):
        """
        Calling resolve_issue() must remove the issue from active_issues.
        """
        manager = SystemStateManager()

        manager.add_issue(
            issue_id="issue_005",
            issue_type="loudness",
            channel="Main LR",
        )
        manager.resolve_issue("issue_005")

        assert "issue_005" not in manager.active_issues

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def test_set_scene_updates_current_scene(self):
        """
        set_scene() must update the current_scene to the supplied Scene value.
        """
        manager = SystemStateManager()

        manager.set_scene(Scene.WORSHIP)

        assert manager.current_scene == Scene.WORSHIP

    def test_set_scene_cycles_through_all_valid_scenes(self):
        """
        All four valid Scene values must be accepted without error.
        """
        manager = SystemStateManager()

        for scene in [Scene.SERMON, Scene.WORSHIP, Scene.PLAY, Scene.CUSTOM]:
            manager.set_scene(scene)
            assert manager.current_scene == scene

    # ------------------------------------------------------------------
    # Loudness mode management
    # ------------------------------------------------------------------

    def test_set_loudness_mode_updates_loudness_mode(self):
        """
        set_loudness_mode() must update the loudness_mode to the supplied
        LoudnessMode value.
        """
        manager = SystemStateManager()

        manager.set_loudness_mode(LoudnessMode.ENERGETIC)

        assert manager.loudness_mode == LoudnessMode.ENERGETIC

    def test_set_loudness_mode_cycles_through_all_valid_modes(self):
        """
        All three valid LoudnessMode values must be accepted without error.
        """
        manager = SystemStateManager()

        for mode in [
            LoudnessMode.CONSERVATIVE,
            LoudnessMode.BALANCED,
            LoudnessMode.ENERGETIC,
        ]:
            manager.set_loudness_mode(mode)
            assert manager.loudness_mode == mode

    # ------------------------------------------------------------------
    # get_active_issues — filtered view
    # ------------------------------------------------------------------

    def test_get_active_issues_excludes_ignored_issues(self):
        """
        get_active_issues() must return ONLY issues that are not ignored.
        Ignored issues must be filtered out of the returned collection.
        """
        manager = SystemStateManager()

        manager.add_issue(issue_id="issue_006", issue_type="eq_mud", channel="Piano")
        manager.add_issue(issue_id="issue_007", issue_type="clipping", channel="Kick")
        manager.ignore_issue("issue_007")

        active = manager.get_active_issues()

        assert "issue_006" in active, "Active issue must appear in get_active_issues()."
        assert "issue_007" not in active, (
            "Ignored issue must NOT appear in get_active_issues()."
        )

    def test_get_active_issues_returns_empty_when_all_resolved(self):
        """
        After all issues are resolved, get_active_issues() must return
        an empty collection.
        """
        manager = SystemStateManager()

        manager.add_issue(issue_id="issue_008", issue_type="loudness", channel="Main LR")
        manager.resolve_issue("issue_008")

        active = manager.get_active_issues()

        assert len(active) == 0

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_concurrent_add_issue_calls_do_not_corrupt_state(self):
        """
        Spec: "thread-safe registry"
        5 threads each adding 10 unique issues concurrently must result
        in exactly 50 issues in active_issues with no data corruption
        and no exceptions raised.
        """
        manager = SystemStateManager()
        errors: list[Exception] = []

        def worker(thread_index: int) -> None:
            try:
                for i in range(10):
                    manager.add_issue(
                        issue_id=f"issue_t{thread_index}_{i}",
                        issue_type="test_concurrent",
                        channel=f"Channel {i}",
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(t,), daemon=True)
            for t in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No thread should have raised an exception
        assert len(errors) == 0, (
            f"Thread-safety violation: {len(errors)} exception(s) raised. "
            f"First: {errors[0] if errors else 'N/A'}"
        )

        # All 50 unique issue_ids must have been registered
        assert len(manager.active_issues) == 50, (
            f"Expected 50 issues after concurrent writes, "
            f"got {len(manager.active_issues)}. "
            "This may indicate a race condition or dropped write."
        )