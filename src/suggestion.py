"""
src/suggestion.py

Agent 4: Suggestion Generator (LLM — Mistral 7B) for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — AGENT 4):
  - "Convert structured issues into clear, instructor-style guidance"
  - "Generate short, actionable suggestions" / "Maintain instructor tone"
  - "Provide optional detailed explanation ('More')"  → explain()
  - "Provide step-by-step instructions ('Tell me how')"  → how_to()
  - "Use channel labels when available"
  Output format:
      [Priority Icon] Message
      Confidence: High / Medium / Low
  Failsafe: "If AI model fails -> fallback to basic alerts"
  AI behaviour rules: advice only (never controls the mixer), concise,
                      instructor tone, use channel labels.

Design
------
The local Mistral 7B model is reached through the injectable ``LLMClient``
interface (Ollama or LM Studio in production). Every model call is wrapped so
that any failure — model down, timeout, empty response — degrades gracefully
to a built-in basic-alert template. The agent therefore always returns usable
guidance, satisfying the never-crash failsafe.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from src.detection import (
    ConfidenceLevel,
    Issue,
    IssueType,
    Priority,
    confidence_level,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Priority -> status icon (spec: Red = urgent, Yellow = warning, Blue = info).
PRIORITY_ICONS: Dict[Priority, str] = {
    Priority.CRITICAL: "🔴",
    Priority.HIGH: "🔴",
    Priority.MEDIUM: "🟡",
    Priority.LOW: "🔵",
}

#: System prompt encoding the spec's AI behaviour rules for the LLM.
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert live-sound engineer coaching an operator during a "
    "service. Speak in a calm, encouraging instructor tone. Give advice only "
    "— never claim to change the mixer yourself. Be concise and actionable, "
    "and refer to channels by their label when one is provided."
)

#: Human-readable phrasing for each issue type (used in prompts + fallbacks).
ISSUE_PHRASES: Dict[IssueType, str] = {
    IssueType.FEEDBACK: "feedback risk (a ringing/howling tone building up)",
    IssueType.VOCAL_MASKING: "vocal masking (the vocal is buried under the music)",
    IssueType.CLIPPING: "clipping (the signal is hitting digital full scale)",
    IssueType.LOUDNESS: "a loudness level that is off the target",
    IssueType.EQ_MUD: "muddiness (too much low-mid energy)",
    IssueType.EQ_HARSHNESS: "harshness (too much high-frequency energy)",
    IssueType.BALANCE: "a balance problem between channels",
}


# ===========================================================================
# Output model
# ===========================================================================

class SuggestionSource(str, Enum):
    """Where a suggestion's text came from."""

    LLM = "llm"
    FALLBACK = "fallback"


class Suggestion(BaseModel):
    """A user-facing suggestion derived from a detected Issue."""

    issue: IssueType
    channel: Optional[str]
    priority: Priority
    confidence: float
    message: str
    source: SuggestionSource

    @property
    def confidence_label(self) -> ConfidenceLevel:
        return confidence_level(self.confidence)

    @property
    def priority_icon(self) -> str:
        return PRIORITY_ICONS.get(self.priority, "🔵")

    def render(self) -> str:
        """Render in the spec output format: icon + message, then confidence."""
        label = self.confidence_label.value.capitalize()
        return f"{self.priority_icon} {self.message}\nConfidence: {label}"


# ===========================================================================
# LLM client interface
# ===========================================================================

class LLMClient(ABC):
    """Interface for a local LLM backend (Mistral 7B via Ollama / LM Studio).

    Concrete, HTTP-backed implementations (``OllamaLLMClient``,
    ``LMStudioLLMClient``) live in :mod:`src.llm_client`; build one with
    ``src.llm_client.build_llm_client()`` and pass it to ``SuggestionGenerator``.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's completion for ``prompt`` (raises on failure)."""


# ===========================================================================
# Generator
# ===========================================================================

class SuggestionGenerator:
    """Turns Issue objects into instructor-style Suggestions via the LLM.

    Pass an ``LLMClient`` to use the local model; omit it (or let a call fail)
    and the generator falls back to built-in basic alerts.
    """

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, issue: Issue) -> Suggestion:
        """Produce a short, actionable Suggestion for a single issue."""
        message, source = self._invoke(
            self._build_prompt(issue, "suggest"),
            lambda: self._fallback_message(issue),
        )
        return Suggestion(
            issue=issue.issue,
            channel=issue.channel,
            priority=issue.priority,
            confidence=issue.confidence,
            message=message,
            source=source,
        )

    def generate_all(self, issues: List[Issue]) -> List[Suggestion]:
        """Generate suggestions for a list of issues, preserving order."""
        return [self.generate(issue) for issue in issues]

    def llm_available(self) -> bool:
        """Whether a local LLM is configured *and* reachable (for the status dot).

        ``False`` when no client is configured (the app runs on basic alerts) or
        when the configured backend's health probe reports it unreachable. A
        client without an ``is_available`` probe is assumed available once set.
        Never raises — a failing probe simply reads as unavailable.
        """
        client = self._client
        if client is None:
            return False
        probe = getattr(client, "is_available", None)
        if callable(probe):
            try:
                return bool(probe())
            except Exception:  # noqa: BLE001 — failsafe: treat as unavailable
                return False
        return True

    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        mix_summary: str = "",
    ) -> Tuple[str, SuggestionSource]:
        """Answer a free-text operator question conversationally (the AI chat).

        Grounds the reply in the current mix (``mix_summary``) and the prior
        turns (``history``: a list of ``{"role", "content"}`` dicts). Uses the
        local LLM when available; otherwise returns a helpful fallback. Returns
        ``(reply_text, source)`` and never raises (failsafe).
        """
        return self._invoke(
            self._build_chat_prompt(message, history or [], mix_summary),
            lambda: self._chat_fallback(),
        )

    def explain(self, issue: Issue) -> str:
        """The "More" action: a fuller explanation of the issue."""
        message, _ = self._invoke(
            self._build_prompt(issue, "explain"),
            lambda: self._fallback_explanation(issue),
        )
        return message

    def how_to(self, issue: Issue) -> str:
        """The "Tell me how" action: numbered, step-by-step instructions."""
        message, _ = self._invoke(
            self._build_prompt(issue, "how_to"),
            lambda: self._fallback_steps(issue),
        )
        return message

    # ------------------------------------------------------------------
    # LLM invocation with failsafe fallback
    # ------------------------------------------------------------------

    def _invoke(
        self, prompt: str, fallback: Callable[[], str]
    ) -> Tuple[str, SuggestionSource]:
        """Call the LLM; on absence/failure/empty output use ``fallback``."""
        if self._client is None:
            return fallback(), SuggestionSource.FALLBACK
        try:
            text = self._client.generate(prompt).strip()
            if not text:
                raise ValueError("empty LLM response")
            return text, SuggestionSource.LLM
        except Exception:  # noqa: BLE001 — failsafe: fall back to basic alerts
            return fallback(), SuggestionSource.FALLBACK

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, issue: Issue, mode: str) -> str:
        phrase = ISSUE_PHRASES.get(issue.issue, issue.issue.value)
        channel = issue.channel or "the mix"
        header = (
            f"Detected issue: {phrase}.\n"
            f"Affected channel/area: {channel}.\n"
            f"Priority: {issue.priority.value}. "
            f"Confidence: {confidence_level(issue.confidence).value}.\n\n"
        )
        if mode == "explain":
            ask = (
                "Explain in 2-3 sentences what this means and why it matters, "
                "in plain language for a volunteer operator."
            )
        elif mode == "how_to":
            ask = (
                "Give concise numbered, step-by-step instructions to fix this "
                "on a Behringer X32. Keep it to 3-5 short steps."
            )
        else:  # suggest
            ask = (
                "Give one short, actionable suggestion (one or two sentences) "
                "to address this. Refer to the channel by its label."
            )
        return header + ask

    def _build_chat_prompt(
        self, message: str, history: List[Dict[str, str]], mix_summary: str
    ) -> str:
        """Compose a conversational prompt from mix context + history + message."""
        lines: List[str] = []
        if mix_summary:
            lines.append(f"Current mix status: {mix_summary}.")
            lines.append("")
        for turn in history:
            role = (turn.get("role") or "").lower() if isinstance(turn, dict) else ""
            content = turn.get("content", "") if isinstance(turn, dict) else ""
            if not content:
                continue
            who = "Operator" if role == "user" else "Coach"
            lines.append(f"{who}: {content}")
        lines.append(f"Operator: {message}")
        lines.append("Coach:")
        return "\n".join(lines)

    @staticmethod
    def _chat_fallback() -> str:
        """Reply used when no local model is reachable."""
        return (
            "I can't reach the local AI model right now, so I can't chat freely. "
            "The live suggestions are still running — connect Ollama or LM Studio "
            "to enable conversational help."
        )

    # ------------------------------------------------------------------
    # Fallback (basic alert) templates
    # ------------------------------------------------------------------

    @staticmethod
    def _target(issue: Issue) -> str:
        return issue.channel if issue.channel else "the main mix"

    def _fallback_message(self, issue: Issue) -> str:
        target = self._target(issue)
        messages: Dict[IssueType, str] = {
            IssueType.FEEDBACK: (
                "Feedback risk detected. Ease down the master or the contributing "
                "channel now, then find and cut the ringing frequency."
            ),
            IssueType.VOCAL_MASKING: (
                f"{target} is getting buried under the music. Bring the vocal up "
                "a little, or ease the band down."
            ),
            IssueType.CLIPPING: (
                f"{target} is clipping. Reduce its gain or fader until the peaks "
                "stop hitting the top."
            ),
            IssueType.LOUDNESS: (
                "Overall loudness is off target. Nudge the master toward your "
                "loudness goal."
            ),
            IssueType.EQ_MUD: (
                f"{target} sounds muddy. Try trimming a few dB around 250-500 Hz."
            ),
            IssueType.EQ_HARSHNESS: (
                f"{target} sounds harsh. Gently cut a little around 3-5 kHz."
            ),
            IssueType.BALANCE: (
                "The balance is a little off. Adjust faders to even things out."
            ),
        }
        return messages.get(issue.issue, f"Check {target}.")

    def _fallback_explanation(self, issue: Issue) -> str:
        phrase = ISSUE_PHRASES.get(issue.issue, issue.issue.value)
        target = self._target(issue)
        return (
            f"This alert is about {phrase} on {target}. It was flagged with "
            f"{confidence_level(issue.confidence).value} confidence. Addressing it "
            "helps keep the mix clear and comfortable for the room."
        )

    def _fallback_steps(self, issue: Issue) -> str:
        target = self._target(issue)
        steps: Dict[IssueType, List[str]] = {
            IssueType.FEEDBACK: [
                f"Pull down the master or {target} immediately.",
                "Listen for the ringing frequency.",
                "Cut that frequency with a narrow EQ band.",
            ],
            IssueType.VOCAL_MASKING: [
                f"Select {target}.",
                "Raise its fader by 2-4 dB.",
                "If still unclear, gently reduce the band's low-mids.",
            ],
            IssueType.CLIPPING: [
                f"Select {target}.",
                "Lower the gain/trim until peaks stop hitting 0 dB.",
                "Re-check the meters.",
            ],
            IssueType.LOUDNESS: [
                "Watch the main loudness (LUFS) meter.",
                "Move the master fader toward the target level.",
                "Re-check after a few seconds.",
            ],
            IssueType.EQ_MUD: [
                f"Select {target}.",
                "Cut 2-4 dB around 250-500 Hz.",
                "Listen for improved clarity.",
            ],
            IssueType.EQ_HARSHNESS: [
                f"Select {target}.",
                "Cut 2-3 dB around 3-5 kHz.",
                "Listen for reduced harshness.",
            ],
            IssueType.BALANCE: [
                "Compare the channel levels.",
                "Adjust faders to even the balance.",
                "Re-check the overall mix.",
            ],
        }
        chosen = steps.get(issue.issue, [f"Select {target}.", "Adjust as needed."])
        return "\n".join(f"{n}. {step}" for n, step in enumerate(chosen, start=1))
