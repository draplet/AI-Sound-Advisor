"""
src/lifecycle.py

Issue lifecycle state machine for the AI Sound Advisor.

Spec requirements implemented here:
  - "Issue must persist for X ms before showing" (mocked at 500ms)
  - States: detected, active, improving, resolved, ignored
  - "An issue cannot move from detected to active unless an internal
     persistence timer (mocked at 500ms) has expired"

Time is driven by simulated clock ticks via advance_time(elapsed_ms); no real
wall-clock sleeps are used, which keeps the state machine deterministic.

State diagram:
    DETECTED --(>=500ms)--> ACTIVE --> IMPROVING
                                  +---> RESOLVED  (terminal)
                                  +---> IGNORED   (terminal)
"""
from __future__ import annotations

from enum import Enum


class IssueState(str, Enum):
    """Lifecycle states an issue can occupy."""

    DETECTED = "detected"
    ACTIVE = "active"
    IMPROVING = "improving"
    RESOLVED = "resolved"
    IGNORED = "ignored"


# Persistence gate: an issue must be observed for this long before it surfaces.
PERSISTENCE_THRESHOLD_MS = 500

# States from which the machine no longer reacts to time advancement.
_TERMINAL_STATES = frozenset({IssueState.RESOLVED, IssueState.IGNORED})


class IssueLifecycle:
    """
    Tracks a single issue's progression through the lifecycle states.

    The persistence timer accumulates via advance_time(); the issue only
    promotes from DETECTED to ACTIVE once the threshold has been crossed.
    """

    def __init__(self, issue_id: str, issue_type: str) -> None:
        self.issue_id = issue_id
        self.issue_type = issue_type
        self.state: IssueState = IssueState.DETECTED
        self.elapsed_ms: int = 0

    def advance_time(self, elapsed_ms: int) -> IssueState:
        """
        Add elapsed_ms to the persistence timer and apply any due transition.

        The only time-driven transition is DETECTED -> ACTIVE once the
        cumulative elapsed time reaches the persistence threshold. Terminal
        states (RESOLVED / IGNORED) are never disturbed by time advancement.
        """
        self.elapsed_ms += elapsed_ms

        if (
            self.state == IssueState.DETECTED
            and self.elapsed_ms >= PERSISTENCE_THRESHOLD_MS
        ):
            self.state = IssueState.ACTIVE

        return self.state

    def resolve(self) -> None:
        """Move the issue to the terminal RESOLVED state."""
        self.state = IssueState.RESOLVED

    def ignore(self) -> None:
        """Move the issue to the terminal IGNORED state."""
        self.state = IssueState.IGNORED

    def improve(self) -> None:
        """Mark an active issue as trending towards resolution."""
        if self.state == IssueState.ACTIVE:
            self.state = IssueState.IMPROVING
