"""
tests/test_recording_analysis.py

AI Sound Advisor — Recording / Broadcast Analysis Module
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — RECORDING ANALYSIS MODULE):
    Analyzes:
      - Audio sent to recording/stream
    Detects:
      - Vocal clarity differences
      - Loudness imbalance
      - Mix inconsistencies vs room
    Purpose:
      Improve broadcast/recording quality

Design under test
-----------------
``src.recording_analysis`` compares the recorder/broadcast feed's metrics
against the room metrics (both produced by Agent 1). Like Agents 1 and 3 it uses
the Strategy Pattern: a list of interchangeable ``RecordingComparator`` objects,
each emitting at most one ``RecordingFinding``. ``RecordingAnalysisEngine``
combines them into a sorted ``RecordingAnalysis``. Everything is deterministic
and failsafe.

Run with:
  pytest tests/test_recording_analysis.py -v
"""
from __future__ import annotations

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.detection import Priority

from src.recording_analysis import (
    LOUDNESS_IMBALANCE_LUFS,
    RecordingAnalysis,
    RecordingAnalysisEngine,
    RecordingComparator,
    RecordingFinding,
    RecordingFindingType,
)

LOW, NORMAL, HIGH = EnergyLevel.LOW, EnergyLevel.NORMAL, EnergyLevel.HIGH


def m(loudness=-23.0, peak=-3.0, low_mid=NORMAL, high=NORMAL) -> AudioMetrics:
    return AudioMetrics(loudness_lufs=loudness, peak_level=peak,
                        low_mid_energy=low_mid, high_freq_energy=high)


# ===========================================================================
# SECTION 1 — Loudness imbalance (recording vs room)
# ===========================================================================

class TestLoudnessImbalance:

    def test_matched_loudness_has_no_finding(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(loudness=-23.0), recording=m(loudness=-23.0))
        kinds = {f.kind for f in result.findings}
        assert RecordingFindingType.LOUDNESS_IMBALANCE not in kinds

    def test_recording_louder_than_room_is_flagged(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(loudness=-23.0), recording=m(loudness=-13.0))
        finding = next(f for f in result.findings
                       if f.kind == RecordingFindingType.LOUDNESS_IMBALANCE)
        assert "louder" in finding.message.lower()

    def test_recording_quieter_than_room_is_flagged(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(loudness=-18.0), recording=m(loudness=-28.0))
        finding = next(f for f in result.findings
                       if f.kind == RecordingFindingType.LOUDNESS_IMBALANCE)
        assert "quieter" in finding.message.lower()

    def test_small_difference_below_threshold_is_ignored(self):
        engine = RecordingAnalysisEngine.default()
        small = LOUDNESS_IMBALANCE_LUFS / 2.0
        result = engine.analyze(room=m(loudness=-23.0),
                                recording=m(loudness=-23.0 + small))
        kinds = {f.kind for f in result.findings}
        assert RecordingFindingType.LOUDNESS_IMBALANCE not in kinds


# ===========================================================================
# SECTION 2 — Vocal clarity differences (presence/high-frequency loss)
# ===========================================================================

class TestVocalClarity:

    def test_clarity_loss_when_recording_has_fewer_highs(self):
        engine = RecordingAnalysisEngine.default()
        # Room has normal presence; the broadcast feed is dull -> clarity lost.
        result = engine.analyze(room=m(high=NORMAL), recording=m(high=LOW))
        kinds = {f.kind for f in result.findings}
        assert RecordingFindingType.VOCAL_CLARITY in kinds

    def test_no_clarity_finding_when_highs_match(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(high=NORMAL), recording=m(high=NORMAL))
        kinds = {f.kind for f in result.findings}
        assert RecordingFindingType.VOCAL_CLARITY not in kinds

    def test_more_highs_in_recording_is_not_a_clarity_loss(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(high=NORMAL), recording=m(high=HIGH))
        kinds = {f.kind for f in result.findings}
        assert RecordingFindingType.VOCAL_CLARITY not in kinds


# ===========================================================================
# SECTION 3 — Mix inconsistency vs room (low-mid tonal balance)
# ===========================================================================

class TestMixInconsistency:

    def test_low_mid_mismatch_is_flagged(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(low_mid=NORMAL), recording=m(low_mid=HIGH))
        kinds = {f.kind for f in result.findings}
        assert RecordingFindingType.MIX_INCONSISTENCY in kinds

    def test_matched_low_mid_has_no_inconsistency(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(low_mid=NORMAL), recording=m(low_mid=NORMAL))
        kinds = {f.kind for f in result.findings}
        assert RecordingFindingType.MIX_INCONSISTENCY not in kinds


# ===========================================================================
# SECTION 4 — Engine behaviour
# ===========================================================================

class TestEngine:

    def test_perfect_match_is_all_clear(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(), recording=m())
        assert isinstance(result, RecordingAnalysis)
        assert result.available is True
        assert result.findings == []
        assert result.all_clear is True

    def test_findings_sorted_by_severity(self):
        engine = RecordingAnalysisEngine.default()
        # Big loudness gap (HIGH) + a low-mid mismatch (MEDIUM) both fire.
        result = engine.analyze(
            room=m(loudness=-23.0, low_mid=NORMAL),
            recording=m(loudness=-13.0, low_mid=HIGH),
        )
        assert len(result.findings) >= 2
        # The most severe finding sorts first (loudness gap is HIGH here).
        severity = {Priority.CRITICAL: 3, Priority.HIGH: 2,
                    Priority.MEDIUM: 1, Priority.LOW: 0}
        ranks = [severity[f.severity] for f in result.findings]
        assert ranks == sorted(ranks, reverse=True)
        assert result.findings[0].kind == RecordingFindingType.LOUDNESS_IMBALANCE

    def test_engine_is_failsafe_if_a_comparator_raises(self):
        class BoomComparator(RecordingComparator):
            def compare(self, room, recording):
                raise RuntimeError("boom")

        engine = RecordingAnalysisEngine([BoomComparator()])
        # A broken comparator must not crash the analysis.
        result = engine.analyze(room=m(), recording=m())
        assert isinstance(result, RecordingAnalysis)
        assert result.findings == []

    def test_findings_carry_a_human_readable_message_and_confidence(self):
        engine = RecordingAnalysisEngine.default()
        result = engine.analyze(room=m(loudness=-23.0), recording=m(loudness=-10.0))
        for f in result.findings:
            assert f.message
            assert 0.0 <= f.confidence <= 1.0
            assert isinstance(f.severity, Priority)
