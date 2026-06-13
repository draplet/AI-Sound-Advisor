"""
tests/test_pacing.py

AI Sound Advisor — Agent 6: Suggestion Pacing Agent
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — AGENT 6):
  - "New issue -> immediate suggestion"
  - "Ongoing issue -> repeat every 10-30 seconds"
  - "After user change -> pause 2-4 seconds"
  - "No issues -> display 'Your mix is sounding good'"
  - "Prevents excessive or rapid suggestions."
  Performance: "LLM calls limited to new issues only, not every loop."
  Change detection: "Pause associated suggestions for 2-4 seconds after change."

Time is driven by an explicit simulated clock (now_ms), exactly like
src/lifecycle.py — no real sleeps, fully deterministic.

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.pacing  →  SuggestionPacingAgent, PacingDecision, ALL_CLEAR_MESSAGE

Run with:
  pytest tests/test_pacing.py -v
"""

from src.detection import Issue, IssueType, Priority
from src.pacing import SuggestionPacingAgent, PacingDecision, ALL_CLEAR_MESSAGE


def issue(issue_type=IssueType.VOCAL_MASKING, channel="Lead Vocal",
          priority=Priority.HIGH, confidence=0.9):
    return Issue(issue=issue_type, channel=channel, priority=priority,
                 confidence=confidence)


def _keys(decision: PacingDecision):
    return {(i.issue, i.channel) for i in decision.to_suggest}


# ===========================================================================
# SECTION 1 — New issue => immediate suggestion
# ===========================================================================

class TestNewIssue:

    def test_new_issue_is_suggested_immediately(self):
        agent = SuggestionPacingAgent()
        decision = agent.tick([issue()], now_ms=1_000)

        assert decision.all_clear is False
        assert (IssueType.VOCAL_MASKING, "Lead Vocal") in _keys(decision)

    def test_two_distinct_new_issues_both_suggested(self):
        agent = SuggestionPacingAgent()
        decision = agent.tick(
            [issue(IssueType.FEEDBACK, channel=None),
             issue(IssueType.CLIPPING, channel=None)],
            now_ms=0,
        )
        assert len(decision.to_suggest) == 2


# ===========================================================================
# SECTION 2 — Ongoing issue => throttled repeat (10-30s window)
# ===========================================================================

class TestOngoingRepeat:

    def test_ongoing_issue_is_not_repeated_before_interval(self):
        agent = SuggestionPacingAgent(repeat_interval_ms=15_000)
        agent.tick([issue()], now_ms=1_000)                      # new -> suggested
        decision = agent.tick([issue()], now_ms=6_000)           # +5s, too soon
        assert decision.to_suggest == []

    def test_ongoing_issue_repeats_after_interval(self):
        agent = SuggestionPacingAgent(repeat_interval_ms=15_000)
        agent.tick([issue()], now_ms=1_000)
        decision = agent.tick([issue()], now_ms=17_000)          # +16s, due
        assert (IssueType.VOCAL_MASKING, "Lead Vocal") in _keys(decision)

    def test_llm_is_called_only_on_new_and_repeat_boundaries(self):
        """Spec performance rule: not every loop. Steady issue ticked each
        second for 20s yields exactly two emissions (new + one repeat)."""
        agent = SuggestionPacingAgent(repeat_interval_ms=15_000)
        emissions = 0
        for t in range(0, 21):  # 0s .. 20s
            decision = agent.tick([issue()], now_ms=t * 1_000)
            emissions += len(decision.to_suggest)
        assert emissions == 2


# ===========================================================================
# SECTION 3 — No issues => "Your mix is sounding good"
# ===========================================================================

class TestAllClear:

    def test_no_issues_reports_all_clear_message(self):
        agent = SuggestionPacingAgent()
        decision = agent.tick([], now_ms=1_000)

        assert decision.all_clear is True
        assert decision.status_message == ALL_CLEAR_MESSAGE
        assert decision.to_suggest == []

    def test_all_clear_message_text(self):
        assert ALL_CLEAR_MESSAGE == "Your mix is sounding good"


# ===========================================================================
# SECTION 4 — Resolved issue that reappears counts as new again
# ===========================================================================

class TestResolveAndReappear:

    def test_reappearing_issue_is_immediate_again(self):
        agent = SuggestionPacingAgent(repeat_interval_ms=15_000)
        agent.tick([issue()], now_ms=1_000)        # new -> suggested
        agent.tick([], now_ms=2_000)               # resolved (all clear)

        decision = agent.tick([issue()], now_ms=2_500)  # back within interval
        assert (IssueType.VOCAL_MASKING, "Lead Vocal") in _keys(decision)


# ===========================================================================
# SECTION 5 — Pause after a user change (2-4s)
# ===========================================================================

class TestUserChangePause:

    def test_new_issue_suppressed_during_channel_pause(self):
        agent = SuggestionPacingAgent(pause_ms=3_000)
        agent.notify_user_change(now_ms=1_000, channel="Lead Vocal")

        decision = agent.tick([issue(channel="Lead Vocal")], now_ms=1_500)
        assert decision.to_suggest == []  # within the 3s pause window

    def test_issue_resumes_after_pause_expires(self):
        agent = SuggestionPacingAgent(pause_ms=3_000)
        agent.notify_user_change(now_ms=1_000, channel="Lead Vocal")
        agent.tick([issue(channel="Lead Vocal")], now_ms=1_500)   # paused

        decision = agent.tick([issue(channel="Lead Vocal")], now_ms=4_500)
        assert (IssueType.VOCAL_MASKING, "Lead Vocal") in _keys(decision)

    def test_channel_pause_does_not_block_other_channels(self):
        agent = SuggestionPacingAgent(pause_ms=3_000)
        agent.notify_user_change(now_ms=1_000, channel="Lead Vocal")

        decision = agent.tick(
            [issue(IssueType.CLIPPING, channel="Kick")], now_ms=1_500
        )
        assert (IssueType.CLIPPING, "Kick") in _keys(decision)

    def test_global_pause_blocks_mix_wide_issue(self):
        agent = SuggestionPacingAgent(pause_ms=3_000)
        agent.notify_user_change(now_ms=1_000)  # global (no channel)

        decision = agent.tick(
            [issue(IssueType.LOUDNESS, channel=None)], now_ms=1_500
        )
        assert decision.to_suggest == []

    def test_is_paused_helper(self):
        agent = SuggestionPacingAgent(pause_ms=3_000)
        agent.notify_user_change(now_ms=1_000, channel="Lead Vocal")
        assert agent.is_paused("Lead Vocal", now_ms=2_000) is True
        assert agent.is_paused("Lead Vocal", now_ms=5_000) is False
        assert agent.is_paused("Kick", now_ms=2_000) is False
