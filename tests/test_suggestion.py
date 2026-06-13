"""
tests/test_suggestion.py

AI Sound Advisor — Agent 4: Suggestion Generator (LLM — Mistral 7B)
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — AGENT 4):
  - "Convert structured issues into clear, instructor-style guidance"
  - "Generate short, actionable suggestions" / "Maintain instructor tone"
  - "Provide optional detailed explanation ('More')"
  - "Provide step-by-step instructions ('Tell me how')"
  - "Use channel labels when available"
  Output format (spec):
      [Priority Icon] Message
      Confidence: High / Medium / Low
  Failsafe: "If AI model fails -> fallback to basic alerts"

The local LLM (Mistral 7B via Ollama / LM Studio) sits behind the injectable
LLMClient interface so the agent is tested with NO model running.

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.suggestion  →  SuggestionGenerator, LLMClient, Suggestion,
                     SuggestionSource

Run with:
  pytest tests/test_suggestion.py -v
"""

import pytest

from src.detection import Issue, IssueType, Priority, ConfidenceLevel, priority_for
from src.state_manager import Scene

from src.suggestion import (
    SuggestionGenerator,
    LLMClient,
    Suggestion,
    SuggestionSource,
)


# ===========================================================================
# Test doubles — in-memory LLM client (no model, no network)
# ===========================================================================

class FakeLLMClient(LLMClient):
    """Returns a canned response and records the last prompt it received."""

    def __init__(self, response="Bring the lead vocal up a couple of dB."):
        self.response = response
        self.last_prompt = None
        self.call_count = 0

    def generate(self, prompt):
        self.call_count += 1
        self.last_prompt = prompt
        return self.response


class RaisingLLMClient(LLMClient):
    """Simulates an unavailable / failing local model."""

    def generate(self, prompt):
        raise ConnectionError("ollama not running")


def make_issue(
    issue_type=IssueType.VOCAL_MASKING,
    channel="Lead Vocal",
    priority=Priority.HIGH,
    confidence=0.9,
):
    return Issue(
        issue=issue_type, channel=channel, priority=priority, confidence=confidence
    )


# ===========================================================================
# SECTION 1 — Suggestion model rendering (spec output format)
# ===========================================================================

class TestSuggestionRendering:

    def test_render_matches_spec_output_format(self):
        suggestion = Suggestion(
            issue=IssueType.VOCAL_MASKING,
            channel="Channel 3",
            priority=Priority.HIGH,
            confidence=0.9,
            message="Channel 3 vocals are slightly buried under music",
            source=SuggestionSource.LLM,
        )
        rendered = suggestion.render()
        assert rendered == (
            "🔴 Channel 3 vocals are slightly buried under music\n"
            "Confidence: High"
        )

    @pytest.mark.parametrize(
        "priority, icon",
        [
            (Priority.CRITICAL, "🔴"),
            (Priority.HIGH, "🔴"),
            (Priority.MEDIUM, "🟡"),
            (Priority.LOW, "🔵"),
        ],
    )
    def test_priority_icon_mapping(self, priority, icon):
        suggestion = Suggestion(
            issue=IssueType.LOUDNESS,
            channel=None,
            priority=priority,
            confidence=0.6,
            message="x",
            source=SuggestionSource.FALLBACK,
        )
        assert suggestion.priority_icon == icon

    @pytest.mark.parametrize(
        "confidence, label",
        [(0.95, "High"), (0.6, "Medium"), (0.2, "Low")],
    )
    def test_render_confidence_label(self, confidence, label):
        suggestion = Suggestion(
            issue=IssueType.CLIPPING,
            channel=None,
            priority=Priority.HIGH,
            confidence=confidence,
            message="msg",
            source=SuggestionSource.FALLBACK,
        )
        assert suggestion.render().endswith(f"Confidence: {label}")


# ===========================================================================
# SECTION 2 — LLM-backed generation (happy path)
# ===========================================================================

class TestLlmGeneration:

    def test_generate_uses_llm_text_when_available(self):
        client = FakeLLMClient(response="Raise the Lead Vocal fader slightly.")
        generator = SuggestionGenerator(client)

        suggestion = generator.generate(make_issue())

        assert suggestion.message == "Raise the Lead Vocal fader slightly."
        assert suggestion.source == SuggestionSource.LLM

    def test_generate_carries_issue_metadata(self):
        client = FakeLLMClient()
        generator = SuggestionGenerator(client)

        issue = make_issue(channel="Lead Vocal", priority=Priority.HIGH, confidence=0.9)
        suggestion = generator.generate(issue)

        assert suggestion.issue == IssueType.VOCAL_MASKING
        assert suggestion.channel == "Lead Vocal"
        assert suggestion.priority == Priority.HIGH
        assert suggestion.confidence == 0.9

    def test_prompt_includes_channel_label_and_issue(self):
        """Spec: 'Use channel labels when available.'"""
        client = FakeLLMClient()
        generator = SuggestionGenerator(client)

        generator.generate(make_issue(channel="Lead Vocal"))

        assert client.last_prompt is not None
        assert "Lead Vocal" in client.last_prompt

    def test_llm_response_is_stripped(self):
        client = FakeLLMClient(response="  trimmed text  \n")
        generator = SuggestionGenerator(client)

        assert generator.generate(make_issue()).message == "trimmed text"


# ===========================================================================
# SECTION 3 — Failsafe fallback ("AI model fails -> basic alerts")
# ===========================================================================

class TestFallback:

    def test_no_client_uses_fallback(self):
        generator = SuggestionGenerator()  # no LLM configured

        suggestion = generator.generate(make_issue())

        assert suggestion.source == SuggestionSource.FALLBACK
        assert suggestion.message != ""

    def test_failing_llm_falls_back_without_raising(self):
        generator = SuggestionGenerator(RaisingLLMClient())

        try:
            suggestion = generator.generate(make_issue())
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"generate() raised when LLM failed: {exc!r}")

        assert suggestion.source == SuggestionSource.FALLBACK
        assert suggestion.message != ""

    def test_empty_llm_response_falls_back(self):
        generator = SuggestionGenerator(FakeLLMClient(response="   "))

        suggestion = generator.generate(make_issue())

        assert suggestion.source == SuggestionSource.FALLBACK
        assert suggestion.message != ""

    def test_fallback_uses_channel_label_when_present(self):
        generator = SuggestionGenerator()
        suggestion = generator.generate(make_issue(channel="Lead Vocal"))
        assert "Lead Vocal" in suggestion.message

    @pytest.mark.parametrize("issue_type", list(IssueType))
    def test_every_issue_type_has_a_fallback_message(self, issue_type):
        """Basic-alert coverage: no issue type may render an empty suggestion."""
        generator = SuggestionGenerator()
        issue = Issue(
            issue=issue_type,
            channel="Ch 1",
            priority=priority_for(issue_type, Scene.SERMON),
            confidence=0.6,
        )
        suggestion = generator.generate(issue)
        assert suggestion.message.strip() != ""
        assert suggestion.render().startswith(suggestion.priority_icon)


# ===========================================================================
# SECTION 4 — "More" (explain) and "Tell me how" (how_to)
# ===========================================================================

class TestExplainAndHowTo:

    def test_explain_uses_llm_when_available(self):
        client = FakeLLMClient(response="Detailed explanation of vocal masking.")
        generator = SuggestionGenerator(client)

        text = generator.explain(make_issue())
        assert text == "Detailed explanation of vocal masking."

    def test_explain_falls_back_without_client(self):
        generator = SuggestionGenerator()
        text = generator.explain(make_issue())
        assert isinstance(text, str) and text.strip() != ""

    def test_how_to_uses_llm_when_available(self):
        client = FakeLLMClient(response="1. Select channel\n2. Raise fader")
        generator = SuggestionGenerator(client)

        assert generator.how_to(make_issue()) == "1. Select channel\n2. Raise fader"

    def test_how_to_fallback_is_step_by_step(self):
        """Spec example: a 'Tell me how' answer is a numbered step list."""
        generator = SuggestionGenerator()
        steps = generator.how_to(make_issue(channel="Lead Vocal"))

        assert "1." in steps and "2." in steps
        assert "Lead Vocal" in steps

    def test_failing_llm_how_to_falls_back(self):
        generator = SuggestionGenerator(RaisingLLMClient())
        steps = generator.how_to(make_issue())
        assert "1." in steps  # fell back to template steps, no exception


# ===========================================================================
# SECTION 5 — Batch generation
# ===========================================================================

class TestGenerateAll:

    def test_generate_all_preserves_order_and_count(self):
        generator = SuggestionGenerator(FakeLLMClient())
        issues = [
            make_issue(issue_type=IssueType.FEEDBACK, channel=None,
                       priority=Priority.CRITICAL, confidence=0.95),
            make_issue(issue_type=IssueType.VOCAL_MASKING, channel="Lead Vocal"),
            make_issue(issue_type=IssueType.CLIPPING, channel=None,
                       priority=Priority.HIGH, confidence=0.85),
        ]

        suggestions = generator.generate_all(issues)

        assert [s.issue for s in suggestions] == [i.issue for i in issues]
        assert all(isinstance(s, Suggestion) for s in suggestions)

    def test_generate_all_on_empty_list_returns_empty(self):
        generator = SuggestionGenerator(FakeLLMClient())
        assert generator.generate_all([]) == []


# ===========================================================================
# SECTION — Conversational chat (AI Prompt -> real LLM chat)
# ===========================================================================

class TestChat:

    def test_chat_uses_the_llm_when_available(self):
        client = FakeLLMClient(response="Try trimming 300 Hz on the vocal.")
        generator = SuggestionGenerator(client)
        reply, source = generator.chat("Why does it sound muddy?")
        assert reply == "Try trimming 300 Hz on the vocal."
        assert source == SuggestionSource.LLM
        # The operator's question is in the prompt sent to the model.
        assert "Why does it sound muddy?" in client.last_prompt

    def test_chat_prompt_includes_history_and_mix_context(self):
        client = FakeLLMClient(response="ok")
        generator = SuggestionGenerator(client)
        generator.chat(
            "And now?",
            history=[{"role": "user", "content": "Is the vocal too quiet?"},
                     {"role": "assistant", "content": "A little — try +2 dB."}],
            mix_summary="active issues: vocal_masking on Lead Vocal",
        )
        prompt = client.last_prompt
        assert "vocal_masking on Lead Vocal" in prompt
        assert "Is the vocal too quiet?" in prompt
        assert "And now?" in prompt

    def test_chat_falls_back_without_a_model(self):
        generator = SuggestionGenerator()  # no client
        reply, source = generator.chat("How do I fix feedback?")
        assert source == SuggestionSource.FALLBACK
        assert reply                                  # a helpful, non-empty message

    def test_chat_falls_back_when_the_model_errors(self):
        generator = SuggestionGenerator(RaisingLLMClient())
        reply, source = generator.chat("Anything?")
        assert source == SuggestionSource.FALLBACK
        assert reply


# ===========================================================================
# SECTION — Channel wording: a channel-less issue reads as the main mix
# ===========================================================================

class TestChannelWording:

    def test_channel_less_issue_refers_to_the_main_mix(self):
        generator = SuggestionGenerator()  # fallback templates
        suggestion = generator.generate(
            make_issue(issue_type=IssueType.CLIPPING, channel=None,
                       priority=Priority.HIGH, confidence=0.9)
        )
        text = suggestion.message.lower()
        assert "main mix" in text
        assert "the channel" not in text

    def test_channel_issue_still_uses_the_channel_label(self):
        generator = SuggestionGenerator()
        suggestion = generator.generate(
            make_issue(issue_type=IssueType.VOCAL_MASKING, channel="Lead Vocal")
        )
        assert "Lead Vocal" in suggestion.message
