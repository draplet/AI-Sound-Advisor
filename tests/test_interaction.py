"""
tests/test_interaction.py

AI Sound Advisor — Agent 7: User Interaction Agent
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — AGENT 7):
  - "Ignore suggestion (temporary, e.g. 3 minutes)"
  - "Mark condition as normal (profile learning)"
  - "'More' button for detailed explanation"
  - "'Tell Me How' button for step-by-step instructions"
  - "Prompt interface for user input"
  Example inputs: "The piano is always loud" / "This is a quiet service" /
                  "Ignore this for now"

Agent 7 is a controller: it owns the temporary ignore timers and delegates the
rest to Agent 4 (SuggestionGenerator: More / Tell me how) and Agent 5
(ContextProfileAgent: mark-normal / prompt instructions). Ignore timing uses an
explicit simulated clock (now_ms), like Agents 6 and the lifecycle.

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.interaction  →  UserInteractionAgent, InteractionResult, UserActionType,
                      IGNORE_DEFAULT_MS

Run with:
  pytest tests/test_interaction.py -v
"""

from src.detection import Issue, IssueType, Priority
from src.suggestion import SuggestionGenerator
from src.profile_store import ContextProfileAgent

from src.interaction import (
    UserInteractionAgent,
    InteractionResult,
    UserActionType,
    IGNORE_DEFAULT_MS,
)


def issue(issue_type=IssueType.VOCAL_MASKING, channel="Lead Vocal",
          priority=Priority.HIGH, confidence=0.9):
    return Issue(issue=issue_type, channel=channel, priority=priority,
                 confidence=confidence)


class FakeGenerator:
    """Duck-typed stand-in for SuggestionGenerator (Agent 4)."""

    def __init__(self):
        self.explained = None
        self.how_to_arg = None

    def explain(self, an_issue):
        self.explained = an_issue
        return "EXPLANATION TEXT"

    def how_to(self, an_issue):
        self.how_to_arg = an_issue
        return "1. Select the channel\n2. Raise the fader"


def make_agent(generator=None, profile=None):
    return UserInteractionAgent(
        suggestions=generator or FakeGenerator(),
        profile=profile or ContextProfileAgent.in_memory(),
    )


# ===========================================================================
# SECTION 1 — Temporary ignore (3 minutes)
# ===========================================================================

class TestIgnore:

    def test_default_ignore_duration_is_three_minutes(self):
        assert IGNORE_DEFAULT_MS == 180_000

    def test_ignore_returns_result_and_marks_issue_ignored(self):
        agent = make_agent()
        result = agent.ignore(issue(), now_ms=0)

        assert isinstance(result, InteractionResult)
        assert result.action == UserActionType.IGNORE
        assert agent.is_ignored(issue(), now_ms=1_000) is True

    def test_ignore_expires_after_duration(self):
        agent = make_agent()
        agent.ignore(issue(), now_ms=0)

        assert agent.is_ignored(issue(), now_ms=179_000) is True
        assert agent.is_ignored(issue(), now_ms=181_000) is False

    def test_custom_ignore_duration(self):
        agent = make_agent()
        agent.ignore(issue(), now_ms=0, duration_ms=5_000)
        assert agent.is_ignored(issue(), now_ms=4_000) is True
        assert agent.is_ignored(issue(), now_ms=6_000) is False

    def test_filter_ignored_removes_only_ignored_issues(self):
        agent = make_agent()
        masking = issue(IssueType.VOCAL_MASKING, channel="Lead Vocal")
        clipping = issue(IssueType.CLIPPING, channel="Kick")
        agent.ignore(masking, now_ms=0)

        remaining = agent.filter_ignored([masking, clipping], now_ms=1_000)

        assert masking not in remaining
        assert clipping in remaining

    def test_filter_ignored_returns_all_after_expiry(self):
        agent = make_agent()
        target = issue()
        agent.ignore(target, now_ms=0)

        remaining = agent.filter_ignored([target], now_ms=200_000)
        assert target in remaining


# ===========================================================================
# SECTION 2 — Mark condition as normal (delegates to Agent 5)
# ===========================================================================

class TestMarkNormal:

    def test_mark_normal_suppresses_issue_in_profile(self):
        profile = ContextProfileAgent.in_memory()
        agent = make_agent(profile=profile)

        result = agent.mark_normal(issue(IssueType.LOUDNESS))

        assert result.action == UserActionType.MARK_NORMAL
        assert IssueType.LOUDNESS in profile.active.suppressed_issues
        # Learned behavior reaches the detection profile.
        assert IssueType.LOUDNESS in profile.detection_profile().suppressed_issues


# ===========================================================================
# SECTION 3 — "More" and "Tell me how" (delegate to Agent 4)
# ===========================================================================

class TestMoreAndHowTo:

    def test_more_delegates_to_generator_explain(self):
        generator = FakeGenerator()
        agent = make_agent(generator=generator)

        result = agent.more(issue())

        assert result.action == UserActionType.MORE
        assert result.detail == "EXPLANATION TEXT"
        assert generator.explained is not None

    def test_how_to_delegates_to_generator_how_to(self):
        generator = FakeGenerator()
        agent = make_agent(generator=generator)

        result = agent.how_to(issue())

        assert result.action == UserActionType.TELL_ME_HOW
        assert "1." in result.detail and "2." in result.detail

    def test_more_works_with_real_generator_fallback(self):
        """Integration: real Agent 4 with no LLM still returns usable text."""
        agent = make_agent(generator=SuggestionGenerator())
        result = agent.more(issue())
        assert isinstance(result.detail, str) and result.detail.strip() != ""


# ===========================================================================
# SECTION 4 — Prompt interface (free-text routing)
# ===========================================================================

class TestPromptInterface:

    def test_ignore_phrase_with_issue_context_ignores_it(self):
        agent = make_agent()
        result = agent.handle_prompt("Ignore this for now", now_ms=0, issue=issue())

        assert result.action == UserActionType.IGNORE
        assert agent.is_ignored(issue(), now_ms=1_000) is True

    def test_general_statement_is_stored_as_instruction(self):
        profile = ContextProfileAgent.in_memory()
        agent = make_agent(profile=profile)

        result = agent.handle_prompt("This is a quiet service")

        assert result.action == UserActionType.INSTRUCTION
        assert "This is a quiet service" in profile.active.user_instructions

    def test_statement_about_a_source_creates_an_override(self):
        """'The piano is always loud' should be learned as a piano override."""
        profile = ContextProfileAgent.in_memory()
        agent = make_agent(profile=profile)

        agent.handle_prompt("The piano is always loud")

        assert "piano" in profile.active.overrides
        assert "The piano is always loud" in profile.active.user_instructions

    def test_ignore_phrase_without_issue_is_recorded_not_crashing(self):
        agent = make_agent()
        result = agent.handle_prompt("Ignore this for now")  # no issue context
        assert isinstance(result, InteractionResult)
