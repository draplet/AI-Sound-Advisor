"""
tests/test_llm_client.py

AI Sound Advisor — Real LLM backend client for Agent 4 (Suggestion Generator)
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt):
  - AGENT 4 / AI: "Mistral 7B (local model) — Run via: Ollama OR LM Studio API"
  - FAILSAFE BEHAVIOR: "If AI model fails -> fallback to basic alerts" and
    "System must never crash during service"
  - CONNECTION STATUS INDICATOR: "AI/LLM availability ... Must update in real time"
  - PERFORMANCE RULES: "AI (LLM) must NOT run every loop" (unchanged: pacing owns this)

Design under test
-----------------
The real HTTP call to the local model server (Ollama / LM Studio) sits behind
an injectable ``HttpTransport`` ABC, so the clients are exercised here with an
in-memory fake — no network, no running model. Each concrete client implements
the existing ``src.suggestion.LLMClient`` interface (``generate(prompt) -> str``)
and adds a never-raising ``is_available()`` health check. Any backend failure
raises ``LLMUnavailableError`` from ``generate`` so the SuggestionGenerator's
existing ``except Exception`` path degrades to basic alerts.

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.llm_client  →  HttpTransport, UrllibHttpTransport, LLMUnavailableError,
                     OllamaLLMClient, LMStudioLLMClient,
                     LLMBackend, LLMConfig, build_llm_client

Run with:
  pytest tests/test_llm_client.py -v
"""

import pytest

from src.detection import Issue, IssueType, Priority
from src.suggestion import LLMClient, SuggestionGenerator, SuggestionSource

from src.llm_client import (
    HttpTransport,
    UrllibHttpTransport,
    LLMUnavailableError,
    OllamaLLMClient,
    LMStudioLLMClient,
    LLMBackend,
    LLMConfig,
    build_llm_client,
    DEFAULT_OLLAMA_URL,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_LMSTUDIO_MODEL,
    DEFAULT_LLM_TIMEOUT,
)


# ===========================================================================
# Test doubles — in-memory HTTP transport (no network)
# ===========================================================================

class FakeTransport(HttpTransport):
    """Records requests; returns a canned body or raises a canned error."""

    def __init__(self, body=None, error=None):
        self._body = body if body is not None else {}
        self._error = error
        self.calls = []  # list of (method, url, payload, timeout)

    def post_json(self, url, payload, timeout):
        self.calls.append(("POST", url, payload, timeout))
        if self._error is not None:
            raise self._error
        return self._body

    def get_json(self, url, timeout):
        self.calls.append(("GET", url, None, timeout))
        if self._error is not None:
            raise self._error
        return self._body


def ollama_body(text):
    return {"response": text}


def lmstudio_body(text):
    return {"choices": [{"message": {"content": text}}]}


def make_issue():
    return Issue(
        issue=IssueType.VOCAL_MASKING,
        channel="Lead Vocal",
        priority=Priority.HIGH,
        confidence=0.9,
    )


# ===========================================================================
# SECTION 1 — Module contracts (interfaces & defaults)
# ===========================================================================

class TestContracts:

    def test_concrete_clients_are_llmclients(self):
        """Both clients must satisfy the existing Agent 4 LLMClient interface."""
        assert issubclass(OllamaLLMClient, LLMClient)
        assert issubclass(LMStudioLLMClient, LLMClient)

    def test_http_transport_is_abstract(self):
        with pytest.raises(TypeError):
            HttpTransport()  # cannot instantiate the ABC directly

    def test_real_transport_exists_and_is_a_transport(self):
        assert issubclass(UrllibHttpTransport, HttpTransport)

    def test_unavailable_error_is_an_exception(self):
        assert issubclass(LLMUnavailableError, Exception)

    def test_default_constants_present(self):
        assert DEFAULT_OLLAMA_URL.startswith("http")
        assert DEFAULT_LMSTUDIO_URL.startswith("http")
        assert DEFAULT_OLLAMA_MODEL
        assert DEFAULT_LMSTUDIO_MODEL
        assert DEFAULT_LLM_TIMEOUT > 0


# ===========================================================================
# SECTION 2 — OllamaLLMClient
# ===========================================================================

class TestOllamaClient:

    def test_generate_returns_model_text(self):
        transport = FakeTransport(body=ollama_body("Raise the Lead Vocal fader."))
        client = OllamaLLMClient(transport=transport)

        assert client.generate("a prompt") == "Raise the Lead Vocal fader."

    def test_generate_strips_whitespace(self):
        transport = FakeTransport(body=ollama_body("  trimmed \n"))
        client = OllamaLLMClient(transport=transport)

        assert client.generate("p") == "trimmed"

    def test_generate_posts_to_api_generate_with_payload(self):
        transport = FakeTransport(body=ollama_body("ok"))
        client = OllamaLLMClient(
            model="mistral", base_url="http://host:11434", transport=transport
        )

        client.generate("MY PROMPT")

        method, url, payload, timeout = transport.calls[0]
        assert method == "POST"
        assert url == "http://host:11434/api/generate"
        assert payload["model"] == "mistral"
        assert payload["prompt"] == "MY PROMPT"
        assert payload["stream"] is False
        assert "system" in payload  # instructor-tone system prompt is sent

    def test_base_url_trailing_slash_is_normalised(self):
        transport = FakeTransport(body=ollama_body("ok"))
        client = OllamaLLMClient(base_url="http://host:11434/", transport=transport)

        client.generate("p")

        assert transport.calls[0][1] == "http://host:11434/api/generate"

    def test_transport_failure_raises_unavailable(self):
        transport = FakeTransport(error=ConnectionError("refused"))
        client = OllamaLLMClient(transport=transport)

        with pytest.raises(LLMUnavailableError):
            client.generate("p")

    def test_empty_response_raises_unavailable(self):
        transport = FakeTransport(body=ollama_body("   "))
        client = OllamaLLMClient(transport=transport)

        with pytest.raises(LLMUnavailableError):
            client.generate("p")

    def test_missing_key_raises_unavailable(self):
        transport = FakeTransport(body={"unexpected": "shape"})
        client = OllamaLLMClient(transport=transport)

        with pytest.raises(LLMUnavailableError):
            client.generate("p")

    def test_is_available_true_when_health_succeeds(self):
        transport = FakeTransport(body={"models": []})
        client = OllamaLLMClient(transport=transport)

        assert client.is_available() is True
        assert transport.calls[0][0] == "GET"  # health uses a GET probe

    def test_is_available_false_and_never_raises_on_failure(self):
        transport = FakeTransport(error=ConnectionError("down"))
        client = OllamaLLMClient(transport=transport)

        assert client.is_available() is False  # swallowed, returns bool


# ===========================================================================
# SECTION 3 — LMStudioLLMClient
# ===========================================================================

class TestLMStudioClient:

    def test_generate_returns_chat_completion_text(self):
        transport = FakeTransport(body=lmstudio_body("Ease the band down a touch."))
        client = LMStudioLLMClient(transport=transport)

        assert client.generate("p") == "Ease the band down a touch."

    def test_generate_posts_chat_completions_with_messages(self):
        transport = FakeTransport(body=lmstudio_body("ok"))
        client = LMStudioLLMClient(
            model="mistral-7b-instruct",
            base_url="http://host:1234/v1",
            transport=transport,
        )

        client.generate("USER PROMPT")

        method, url, payload, timeout = transport.calls[0]
        assert method == "POST"
        assert url == "http://host:1234/v1/chat/completions"
        assert payload["model"] == "mistral-7b-instruct"
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["system", "user"]
        assert payload["messages"][1]["content"] == "USER PROMPT"

    def test_transport_failure_raises_unavailable(self):
        transport = FakeTransport(error=TimeoutError("slow"))
        client = LMStudioLLMClient(transport=transport)

        with pytest.raises(LLMUnavailableError):
            client.generate("p")

    def test_malformed_response_raises_unavailable(self):
        transport = FakeTransport(body={"choices": []})
        client = LMStudioLLMClient(transport=transport)

        with pytest.raises(LLMUnavailableError):
            client.generate("p")

    def test_is_available_true_when_models_listed(self):
        transport = FakeTransport(body={"data": []})
        client = LMStudioLLMClient(transport=transport)

        assert client.is_available() is True

    def test_is_available_false_on_failure(self):
        transport = FakeTransport(error=ConnectionError("down"))
        client = LMStudioLLMClient(transport=transport)

        assert client.is_available() is False


# ===========================================================================
# SECTION 4 — Integration with the existing SuggestionGenerator (Agent 4)
# ===========================================================================

class TestSuggestionGeneratorIntegration:

    def test_real_client_text_is_used_as_llm_source(self):
        transport = FakeTransport(body=ollama_body("Bring the vocal up 2 dB."))
        generator = SuggestionGenerator(OllamaLLMClient(transport=transport))

        suggestion = generator.generate(make_issue())

        assert suggestion.message == "Bring the vocal up 2 dB."
        assert suggestion.source == SuggestionSource.LLM

    def test_failing_client_falls_back_to_basic_alert(self):
        """Failsafe: a dead model server must not raise — basic alert instead."""
        transport = FakeTransport(error=ConnectionError("ollama not running"))
        generator = SuggestionGenerator(OllamaLLMClient(transport=transport))

        suggestion = generator.generate(make_issue())

        assert suggestion.source == SuggestionSource.FALLBACK
        assert suggestion.message.strip() != ""


# ===========================================================================
# SECTION 5 — Configuration & factory (env-driven, app runs without a model)
# ===========================================================================

class TestConfigAndFactory:

    def test_default_backend_is_none(self):
        cfg = LLMConfig.from_env({})
        assert cfg.backend == LLMBackend.NONE

    def test_build_returns_none_when_backend_none(self):
        """Default behaviour: no LLM wired, app still runs on basic alerts."""
        assert build_llm_client(LLMConfig(backend=LLMBackend.NONE)) is None

    def test_from_env_selects_ollama(self):
        cfg = LLMConfig.from_env({"SOUND_ADVISOR_LLM_BACKEND": "ollama"})
        assert cfg.backend == LLMBackend.OLLAMA
        client = build_llm_client(cfg)
        assert isinstance(client, OllamaLLMClient)

    def test_from_env_selects_lmstudio_and_overrides(self):
        cfg = LLMConfig.from_env(
            {
                "SOUND_ADVISOR_LLM_BACKEND": "lmstudio",
                "SOUND_ADVISOR_LLM_MODEL": "my-model",
                "SOUND_ADVISOR_LLM_BASE_URL": "http://box:1234/v1",
                "SOUND_ADVISOR_LLM_TIMEOUT": "12.5",
            }
        )
        assert cfg.backend == LLMBackend.LMSTUDIO
        assert cfg.model == "my-model"
        assert cfg.base_url == "http://box:1234/v1"
        assert cfg.timeout == 12.5
        assert isinstance(build_llm_client(cfg), LMStudioLLMClient)

    def test_unknown_backend_falls_back_to_none(self):
        cfg = LLMConfig.from_env({"SOUND_ADVISOR_LLM_BACKEND": "gpt-9000"})
        assert cfg.backend == LLMBackend.NONE
        assert build_llm_client(cfg) is None

    def test_invalid_timeout_falls_back_to_default(self):
        cfg = LLMConfig.from_env(
            {"SOUND_ADVISOR_LLM_BACKEND": "ollama",
             "SOUND_ADVISOR_LLM_TIMEOUT": "not-a-number"}
        )
        assert cfg.timeout == DEFAULT_LLM_TIMEOUT

    def test_backend_parsing_is_case_insensitive(self):
        cfg = LLMConfig.from_env({"SOUND_ADVISOR_LLM_BACKEND": "OLLAMA"})
        assert cfg.backend == LLMBackend.OLLAMA
