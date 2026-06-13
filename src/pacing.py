"""
src/pacing.py

Agent 6: Suggestion Pacing Agent for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — AGENT 6):
  - "New issue -> immediate suggestion"
  - "Ongoing issue -> repeat every 10-30 seconds"
  - "After user change -> pause 2-4 seconds"
  - "No issues -> display 'Your mix is sounding good'"
  - "Prevents excessive or rapid suggestions."
  Performance rule: "LLM calls limited to new issues only, not every loop."
  Change detection: "Pause associated suggestions for 2-4 seconds after change."

Design
------
Each loop tick the system passes the currently-detected issues plus the current
time (ms). The agent returns a ``PacingDecision`` whose ``to_suggest`` list is
exactly the issues that should be handed to Agent 4 (the LLM) this tick — new
issues and issues due for a throttled repeat, minus anything inside a
post-change pause window. Everything else is held back, which is what keeps the
LLM off the hot loop.

Time is supplied explicitly (now_ms) so the agent is deterministic and testable
with no real sleeps, matching src/lifecycle.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.detection import Issue

#: Shown when there are no active issues (spec exact text).
ALL_CLEAR_MESSAGE = "Your mix is sounding good"

#: Default repeat cadence for an ongoing issue (spec window: 10-30 s).
DEFAULT_REPEAT_INTERVAL_MS = 15_000

#: Default pause after a user change before suggesting again (spec: 2-4 s).
DEFAULT_PAUSE_MS = 3_000

#: An issue's identity for pacing purposes.
IssueKey = Tuple[object, Optional[str]]


class PacingDecision(BaseModel):
    """Result of one pacing tick."""

    to_suggest: List[Issue] = Field(default_factory=list)
    all_clear: bool = False
    status_message: Optional[str] = None


@dataclass
class _Track:
    """Per-issue pacing state."""

    first_seen_ms: int
    last_suggested_ms: Optional[int] = None


class SuggestionPacingAgent:
    """Decides which detected issues are released as suggestions, and when."""

    def __init__(
        self,
        repeat_interval_ms: int = DEFAULT_REPEAT_INTERVAL_MS,
        pause_ms: int = DEFAULT_PAUSE_MS,
    ) -> None:
        self.repeat_interval_ms = repeat_interval_ms
        self.pause_ms = pause_ms
        self._tracked: Dict[IssueKey, _Track] = {}
        self._global_pause_until: Optional[int] = None
        self._channel_pause_until: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Main loop entry point
    # ------------------------------------------------------------------

    def tick(self, issues: List[Issue], now_ms: int) -> PacingDecision:
        """Process the current issue set and return what to suggest now."""
        if not issues:
            # No active issues: drop all tracking so a returning issue is "new".
            self._tracked.clear()
            return PacingDecision(
                to_suggest=[], all_clear=True, status_message=ALL_CLEAR_MESSAGE
            )

        present: set[IssueKey] = set()
        to_suggest: List[Issue] = []

        for current in issues:
            key = self._key(current)
            present.add(key)

            entry = self._tracked.get(key)
            if entry is None:
                entry = _Track(first_seen_ms=now_ms)
                self._tracked[key] = entry

            # Hold back anything inside a post-change pause window.
            if self.is_paused(current.channel, now_ms):
                continue

            # "New" (never suggested) fires immediately; "ongoing" only when due.
            due = (
                entry.last_suggested_ms is None
                or (now_ms - entry.last_suggested_ms) >= self.repeat_interval_ms
            )
            if due:
                entry.last_suggested_ms = now_ms
                to_suggest.append(current)

        # Forget issues that are no longer present (resolved).
        for key in [k for k in self._tracked if k not in present]:
            del self._tracked[key]

        return PacingDecision(to_suggest=to_suggest, all_clear=False, status_message=None)

    # ------------------------------------------------------------------
    # User-change pause (spec: "pause 2-4 seconds after change")
    # ------------------------------------------------------------------

    def notify_user_change(
        self,
        now_ms: int,
        channel: Optional[str] = None,
        pause_ms: Optional[int] = None,
    ) -> None:
        """Register a user change; suppress related suggestions briefly.

        ``channel=None`` pauses mix-wide (global) suggestions; a channel name
        pauses only that channel's suggestions.
        """
        until = now_ms + (pause_ms if pause_ms is not None else self.pause_ms)
        if channel is None:
            self._global_pause_until = until
        else:
            previous = self._channel_pause_until.get(channel)
            self._channel_pause_until[channel] = (
                until if previous is None else max(previous, until)
            )

    def is_paused(self, channel: Optional[str], now_ms: int) -> bool:
        """Whether suggestions for ``channel`` are currently paused."""
        if self._global_pause_until is not None and now_ms < self._global_pause_until:
            return True
        if channel is not None:
            until = self._channel_pause_until.get(channel)
            if until is not None and now_ms < until:
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(issue: Issue) -> IssueKey:
        return (issue.issue, issue.channel)

    def reset(self) -> None:
        """Clear all tracking and pause state."""
        self._tracked.clear()
        self._global_pause_until = None
        self._channel_pause_until.clear()
