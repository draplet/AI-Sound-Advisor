"""
src/interaction.py

Agent 7: User Interaction Agent for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — AGENT 7):
  - "Ignore suggestion (temporary, e.g. 3 minutes)"
  - "Mark condition as normal (profile learning)"
  - "'More' button for detailed explanation"
  - "'Tell Me How' button for step-by-step instructions"
  - "Prompt interface for user input"

Design
------
Agent 7 is the controller for operator actions. It owns the short-lived
"ignore" timers itself and delegates everything else to the agents already
built:
  - More / Tell me how  -> Agent 4 (SuggestionGenerator.explain / .how_to)
  - Mark as normal / prompt instructions -> Agent 5 (ContextProfileAgent)

Ignore timing is driven by an explicit simulated clock (now_ms) so it stays
deterministic and testable, matching Agent 6 and src/lifecycle.py.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from src.detection import Issue, IssueType

#: Default temporary-ignore duration (spec: "e.g. 3 minutes").
IGNORE_DEFAULT_MS = 180_000

#: Acoustic-source words that, when mentioned in a prompt, become an override.
_SOURCE_WORDS = ["piano", "drums", "choir", "bass", "guitar", "organ", "kick", "vocal"]
#: Words that signal a "this is normal/expected" statement worth learning.
_NORMAL_WORDS = ["loud", "quiet", "normal", "normally", "always"]

IssueKey = Tuple[object, Optional[str]]


class UserActionType(str, Enum):
    """The kind of action the agent performed in response to the user."""

    IGNORE = "ignore"
    MARK_NORMAL = "mark_normal"
    MORE = "more"
    TELL_ME_HOW = "tell_me_how"
    INSTRUCTION = "instruction"


class InteractionResult(BaseModel):
    """The outcome of a user action, for display by the UI."""

    action: UserActionType
    message: str
    detail: Optional[str] = None


class UserInteractionAgent:
    """Handles operator controls, delegating to the suggestion + profile agents."""

    def __init__(self, suggestions, profile) -> None:
        # Duck-typed: `suggestions` needs explain()/how_to(); `profile` needs
        # mark_normal()/add_user_instruction()/set_override().
        self._suggestions = suggestions
        self._profile = profile
        self._ignored_until: Dict[IssueKey, int] = {}

    # ------------------------------------------------------------------
    # Ignore (temporary suppression)
    # ------------------------------------------------------------------

    def ignore(
        self, an_issue: Issue, now_ms: int, duration_ms: int = IGNORE_DEFAULT_MS
    ) -> InteractionResult:
        """Temporarily suppress an issue's suggestions for ``duration_ms``."""
        self._ignored_until[self._key(an_issue)] = now_ms + duration_ms
        minutes = duration_ms / 60_000
        return InteractionResult(
            action=UserActionType.IGNORE,
            message=f"Ignoring this {an_issue.issue.value} suggestion for "
            f"{minutes:.0f} minute(s).",
        )

    def is_ignored(self, an_issue: Issue, now_ms: int) -> bool:
        """Whether the issue is currently within an active ignore window."""
        until = self._ignored_until.get(self._key(an_issue))
        return until is not None and now_ms < until

    def filter_ignored(self, issues: List[Issue], now_ms: int) -> List[Issue]:
        """Return the issues that are not currently ignored (purges expired)."""
        self._purge_expired(now_ms)
        return [i for i in issues if not self.is_ignored(i, now_ms)]

    # ------------------------------------------------------------------
    # Mark as normal (delegates to Agent 5)
    # ------------------------------------------------------------------

    def mark_normal(self, an_issue: Issue) -> InteractionResult:
        """Record the issue's condition as normal so it is suppressed."""
        self._profile.mark_normal(an_issue.issue)
        return InteractionResult(
            action=UserActionType.MARK_NORMAL,
            message=f"Got it — treating {an_issue.issue.value} as normal here.",
        )

    # ------------------------------------------------------------------
    # More / Tell me how (delegate to Agent 4)
    # ------------------------------------------------------------------

    def more(self, an_issue: Issue) -> InteractionResult:
        """The 'More' action: a fuller explanation of the issue."""
        detail = self._suggestions.explain(an_issue)
        return InteractionResult(
            action=UserActionType.MORE, message="Here's more detail.", detail=detail
        )

    def how_to(self, an_issue: Issue) -> InteractionResult:
        """The 'Tell me how' action: step-by-step instructions."""
        detail = self._suggestions.how_to(an_issue)
        return InteractionResult(
            action=UserActionType.TELL_ME_HOW,
            message="Here are the steps.",
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Conversational chat (delegates to Agent 4's LLM)
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        mix_summary: str = "",
    ):
        """Answer an operator question conversationally via Agent 4.

        Returns ``(reply_text, source)``. This is the prompt interface's
        conversational mode; it does not mutate the profile (use
        :meth:`handle_prompt` for instructions/overrides that should be learned).
        """
        return self._suggestions.chat(
            message, history=history, mix_summary=mix_summary
        )

    # ------------------------------------------------------------------
    # Free-text prompt interface
    # ------------------------------------------------------------------

    def handle_prompt(
        self, text: str, now_ms: int = 0, issue: Optional[Issue] = None
    ) -> InteractionResult:
        """Route a free-text operator instruction to the right action."""
        lowered = text.strip().lower()

        # "Ignore this for now" — needs an issue in context.
        if "ignore" in lowered and issue is not None:
            return self.ignore(issue, now_ms=now_ms)

        # Always keep the raw instruction for the profile / LLM context.
        self._profile.add_user_instruction(text)

        # "The piano is always loud" -> learn an override for that source.
        if any(w in lowered for w in _NORMAL_WORDS):
            source = next((w for w in _SOURCE_WORDS if w in lowered), None)
            if source is not None:
                self._profile.set_override(source, text)

        return InteractionResult(
            action=UserActionType.INSTRUCTION,
            message="Got it — I'll remember that.",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(an_issue: Issue) -> IssueKey:
        return (an_issue.issue, an_issue.channel)

    def _purge_expired(self, now_ms: int) -> None:
        for key in [k for k, until in self._ignored_until.items() if now_ms >= until]:
            del self._ignored_until[key]
