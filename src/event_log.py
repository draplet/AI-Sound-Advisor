"""
src/event_log.py

Logging System for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — LOGGING SYSTEM):
    Record:
      - Issues detected
      - User actions
      - Audio metrics
    Allows:
      - Post-service review
      - Training improvements

Design
------
The log is an append-only stream of structured ``LogRecord`` objects. The actual
storage sits behind an injectable ``LogSink`` ABC — the project's standard
pattern for external I/O — so the logger is exercised in tests with an in-memory
fake (no disk), while production uses ``JsonlFileSink`` (newline-delimited JSON,
the V1 storage choice: appendable, human-readable, trivially re-readable for
post-service review).

``SessionLogger`` is the façade the rest of the system talks to. Time is injected
as ``now_ms`` for determinism, exactly like Agents 6/7, the lifecycle and the
system loop — the logger holds no wall clock.

FAILSAFE — "System must never crash during service"
----------------------------------------------------
Every write is wrapped: a full disk or a broken sink can never take down a live
service. A failed write is silently dropped (the show goes on); reading back a
log skips any corrupt line rather than raising.

PERFORMANCE
-----------
``log_frame`` is the hot-loop entry point. It logs each issue once when it first
appears (and once when it clears), and throttles audio-metric snapshots to at
most one per ``metrics_interval_ms`` — so a 4 Hz loop does not write thousands of
near-identical metric rows per minute.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from src.audio_analysis import AudioMetrics
from src.detection import Issue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default minimum spacing (ms) between logged audio-metric snapshots.
DEFAULT_METRICS_INTERVAL_MS = 5_000

IssueKey = Tuple[str, Optional[str]]


# ===========================================================================
# Record model
# ===========================================================================

class LogEventType(str, Enum):
    """The kind of event a record captures (spec: issues, actions, metrics)."""

    ISSUE_DETECTED = "issue_detected"
    ISSUE_RESOLVED = "issue_resolved"
    USER_ACTION = "user_action"
    METRICS = "metrics"
    SYSTEM = "system"


class LogRecord(BaseModel):
    """A single timestamped log entry."""

    timestamp_ms: int
    type: LogEventType
    data: Dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# Sink interface + implementations
# ===========================================================================

class LogSink(ABC):
    """Where log records are written (and read back for review)."""

    @abstractmethod
    def write(self, record: LogRecord) -> None:
        """Append ``record`` to the log (may raise; the logger guards calls)."""

    @abstractmethod
    def read_all(self) -> List[LogRecord]:
        """Return every record written so far, in order."""


class InMemoryLogSink(LogSink):
    """A transient, in-process sink — used in tests and for live review buffers."""

    def __init__(self) -> None:
        self._records: List[LogRecord] = []

    def write(self, record: LogRecord) -> None:
        self._records.append(record)

    def read_all(self) -> List[LogRecord]:
        return list(self._records)


class JsonlFileSink(LogSink):
    """Append-only newline-delimited JSON log on disk (V1 storage).

    One JSON object per line keeps the log appendable and crash-tolerant: a
    partially written final line never corrupts earlier records, and review
    tooling can stream it line by line.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: LogRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def read_all(self) -> List[LogRecord]:
        if not self._path.exists():
            return []
        out: List[LogRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(LogRecord.model_validate_json(line))
            except Exception:  # noqa: BLE001 — skip corrupt lines, never crash review
                continue
        return out


# ===========================================================================
# Session logger (façade)
# ===========================================================================

class SessionLogger:
    """Records issues, user actions and audio metrics for post-service review.

    Pass any :class:`LogSink`. All public methods are failsafe — a sink failure
    is swallowed so logging can never interrupt a live service.
    """

    def __init__(
        self,
        sink: LogSink,
        *,
        metrics_interval_ms: int = DEFAULT_METRICS_INTERVAL_MS,
    ) -> None:
        self._sink = sink
        self._metrics_interval_ms = metrics_interval_ms
        self._active_issues: Set[IssueKey] = set()
        self._last_metrics_ms: Optional[int] = None

    # ------------------------------------------------------------------
    # Generic write (failsafe)
    # ------------------------------------------------------------------

    def log_event(
        self, event_type: LogEventType, data: Dict[str, Any], now_ms: int
    ) -> None:
        """Write a single record; never raises (a failed write is dropped)."""
        try:
            self._sink.write(
                LogRecord(timestamp_ms=now_ms, type=event_type, data=dict(data))
            )
        except Exception:  # noqa: BLE001 — failsafe: logging must never crash service
            pass

    # ------------------------------------------------------------------
    # The three spec record kinds
    # ------------------------------------------------------------------

    def log_issue(self, issue: Issue, now_ms: int) -> None:
        """Record that an issue was detected."""
        self.log_event(LogEventType.ISSUE_DETECTED, self._issue_data(issue), now_ms)

    def log_user_action(
        self,
        action: str,
        now_ms: int,
        *,
        channel: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """Record an operator action (ignore, mark-normal, more, etc.)."""
        data: Dict[str, Any] = {"action": action}
        if channel is not None:
            data["channel"] = channel
        if detail is not None:
            data["detail"] = detail
        self.log_event(LogEventType.USER_ACTION, data, now_ms)

    def log_metrics(self, metrics: AudioMetrics, now_ms: int) -> None:
        """Record an audio-metrics snapshot."""
        self.log_event(
            LogEventType.METRICS, metrics.model_dump(mode="json"), now_ms
        )

    # ------------------------------------------------------------------
    # Hot-loop entry point: log a whole frame (dedup + throttle)
    # ------------------------------------------------------------------

    def log_frame(self, frame: Any, now_ms: int) -> None:
        """Log a LoopFrame: new/resolved issues plus throttled metric snapshots.

        Logs each issue once on appearance and once on resolution, so a steady
        issue does not flood the log every tick. Metric snapshots are written at
        most once per ``metrics_interval_ms``. Fully failsafe.
        """
        try:
            current: Dict[IssueKey, Issue] = {}
            for issue in getattr(frame, "issues", []) or []:
                current[self._key(issue)] = issue

            for key, issue in current.items():
                if key not in self._active_issues:
                    self.log_issue(issue, now_ms)

            for key in self._active_issues - set(current):
                self.log_event(
                    LogEventType.ISSUE_RESOLVED,
                    {"issue": key[0], "channel": key[1]},
                    now_ms,
                )
            self._active_issues = set(current)

            metrics = getattr(frame, "metrics", None)
            if metrics is not None and self._metrics_due(now_ms):
                self.log_metrics(metrics, now_ms)
                self._last_metrics_ms = now_ms
        except Exception:  # noqa: BLE001 — failsafe: never crash the loop on logging
            pass

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def records(self) -> List[LogRecord]:
        """Return every record logged so far (for post-service review)."""
        try:
            return self._sink.read_all()
        except Exception:  # noqa: BLE001 — failsafe
            return []

    def recent(self, limit: int = 100) -> List[LogRecord]:
        """Return the most recent ``limit`` records (newest last)."""
        records = self.records()
        if limit <= 0:
            return []
        return records[-limit:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(issue: Issue) -> IssueKey:
        return (issue.issue.value, issue.channel)

    @staticmethod
    def _issue_data(issue: Issue) -> Dict[str, Any]:
        return {
            "issue": issue.issue.value,
            "channel": issue.channel,
            "priority": issue.priority.value,
            "confidence": issue.confidence,
        }

    def _metrics_due(self, now_ms: int) -> bool:
        last = self._last_metrics_ms
        return last is None or (now_ms - last) >= self._metrics_interval_ms
