"""
tests/test_event_log.py

AI Sound Advisor — Logging System
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — LOGGING SYSTEM):
    Record:
      - Issues detected
      - User actions
      - Audio metrics
    Allows:
      - Post-service review
      - Training improvements

Design under test
-----------------
``src.event_log`` provides an append-only event log:
  - ``LogSink`` ABC (external I/O behind an interface, project convention) with
    a real ``JsonlFileSink`` (newline-delimited JSON on disk — V1 storage) and
    an ``InMemoryLogSink`` for tests.
  - ``SessionLogger`` records issues, user actions and audio metrics. Time is
    injected (now_ms) for determinism, exactly like the rest of the system.
  - Logging is failsafe: a broken sink must NEVER crash a live service.

Run with:
  pytest tests/test_event_log.py -v
"""
from __future__ import annotations

import numpy as np

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.detection import Issue, IssueType, Priority
from src.system_loop import LoopFrame

from src.event_log import (
    InMemoryLogSink,
    JsonlFileSink,
    LogEventType,
    LogRecord,
    SessionLogger,
)


# ===========================================================================
# Helpers
# ===========================================================================

def an_issue(issue=IssueType.CLIPPING, channel="Lead Vocal",
             priority=Priority.HIGH, confidence=0.9) -> Issue:
    return Issue(issue=issue, channel=channel,
                 priority=priority, confidence=confidence)


def some_metrics(loudness=-18.0, peak=-1.0) -> AudioMetrics:
    return AudioMetrics(loudness_lufs=loudness, peak_level=peak,
                        low_mid_energy=EnergyLevel.NORMAL,
                        high_freq_energy=EnergyLevel.NORMAL)


# ===========================================================================
# SECTION 1 — Recording the three required record kinds
# ===========================================================================

class TestRecordingEvents:

    def test_log_issue_writes_a_record(self):
        sink = InMemoryLogSink()
        logger = SessionLogger(sink)
        logger.log_issue(an_issue(), now_ms=1_000)
        records = sink.read_all()
        assert len(records) == 1
        rec = records[0]
        assert rec.type == LogEventType.ISSUE_DETECTED
        assert rec.timestamp_ms == 1_000
        assert rec.data["issue"] == IssueType.CLIPPING.value
        assert rec.data["channel"] == "Lead Vocal"
        assert rec.data["priority"] == Priority.HIGH.value

    def test_log_user_action_writes_a_record(self):
        sink = InMemoryLogSink()
        logger = SessionLogger(sink)
        logger.log_user_action("ignore", now_ms=2_000, channel="Lead Vocal")
        rec = sink.read_all()[0]
        assert rec.type == LogEventType.USER_ACTION
        assert rec.data["action"] == "ignore"
        assert rec.data["channel"] == "Lead Vocal"

    def test_log_metrics_writes_a_record(self):
        sink = InMemoryLogSink()
        logger = SessionLogger(sink)
        logger.log_metrics(some_metrics(loudness=-15.0), now_ms=3_000)
        rec = sink.read_all()[0]
        assert rec.type == LogEventType.METRICS
        assert rec.data["loudness_lufs"] == -15.0

    def test_records_are_returned_in_order(self):
        sink = InMemoryLogSink()
        logger = SessionLogger(sink)
        logger.log_issue(an_issue(), now_ms=1_000)
        logger.log_user_action("more", now_ms=1_500)
        logger.log_metrics(some_metrics(), now_ms=2_000)
        kinds = [r.type for r in logger.records()]
        assert kinds == [
            LogEventType.ISSUE_DETECTED,
            LogEventType.USER_ACTION,
            LogEventType.METRICS,
        ]


# ===========================================================================
# SECTION 2 — Logging a whole LoopFrame (issue lifecycle + throttled metrics)
# ===========================================================================

class TestFrameLogging:

    def _frame(self, issues, metrics, now_ms):
        return LoopFrame(tick_index=1, now_ms=now_ms,
                         metrics=metrics, issues=issues)

    def test_new_issue_is_logged_once_not_every_tick(self):
        sink = InMemoryLogSink()
        logger = SessionLogger(sink)
        issue = an_issue()
        # Same issue present across three ticks -> logged once (on appearance).
        logger.log_frame(self._frame([issue], some_metrics(), 0), now_ms=0)
        logger.log_frame(self._frame([issue], some_metrics(), 250), now_ms=250)
        logger.log_frame(self._frame([issue], some_metrics(), 500), now_ms=500)
        detected = [r for r in sink.read_all()
                    if r.type == LogEventType.ISSUE_DETECTED]
        assert len(detected) == 1

    def test_issue_clearing_is_logged_as_resolved(self):
        sink = InMemoryLogSink()
        logger = SessionLogger(sink)
        issue = an_issue()
        logger.log_frame(self._frame([issue], some_metrics(), 0), now_ms=0)
        logger.log_frame(self._frame([], some_metrics(), 250), now_ms=250)
        resolved = [r for r in sink.read_all()
                    if r.type == LogEventType.ISSUE_RESOLVED]
        assert len(resolved) == 1
        assert resolved[0].data["issue"] == IssueType.CLIPPING.value

    def test_metrics_are_throttled(self):
        sink = InMemoryLogSink()
        logger = SessionLogger(sink, metrics_interval_ms=5_000)
        # 21 ticks across 5 seconds at 250ms -> metrics logged ~ twice, not 21x.
        for i in range(21):
            logger.log_frame(self._frame([], some_metrics(), i * 250),
                             now_ms=i * 250)
        metric_records = [r for r in sink.read_all()
                          if r.type == LogEventType.METRICS]
        assert 1 <= len(metric_records) <= 3


# ===========================================================================
# SECTION 3 — Failsafe: a broken sink never crashes the service
# ===========================================================================

class TestFailsafe:

    def test_broken_sink_does_not_raise(self):
        class BoomSink(InMemoryLogSink):
            def write(self, record):
                raise IOError("disk full")

        logger = SessionLogger(BoomSink())
        # None of these may raise.
        logger.log_issue(an_issue(), now_ms=1_000)
        logger.log_user_action("ignore", now_ms=1_000)
        logger.log_metrics(some_metrics(), now_ms=1_000)
        logger.log_frame(
            LoopFrame(tick_index=1, now_ms=0, metrics=some_metrics(),
                      issues=[an_issue()]),
            now_ms=0,
        )


# ===========================================================================
# SECTION 4 — JSONL file persistence (post-service review)
# ===========================================================================

class TestJsonlPersistence:

    def test_records_persist_to_disk_and_read_back(self, tmp_path):
        path = tmp_path / "session.jsonl"
        logger = SessionLogger(JsonlFileSink(path))
        logger.log_issue(an_issue(), now_ms=1_000)
        logger.log_user_action("tell_me_how", now_ms=1_200)
        logger.log_metrics(some_metrics(), now_ms=1_500)

        assert path.exists()
        # A fresh reader over the same file recovers every record.
        reread = JsonlFileSink(path).read_all()
        assert [r.type for r in reread] == [
            LogEventType.ISSUE_DETECTED,
            LogEventType.USER_ACTION,
            LogEventType.METRICS,
        ]

    def test_corrupt_lines_are_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "session.jsonl"
        sink = JsonlFileSink(path)
        sink.write(LogRecord(timestamp_ms=1, type=LogEventType.SYSTEM, data={}))
        # Append a junk line then a good one.
        with path.open("a", encoding="utf-8") as f:
            f.write("not json at all\n")
        sink.write(LogRecord(timestamp_ms=2, type=LogEventType.SYSTEM, data={}))
        records = sink.read_all()
        assert len(records) == 2     # the junk line is skipped, not fatal
