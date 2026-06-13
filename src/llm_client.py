"""
src/llm_client.py

Real local-LLM backends for Agent 4 (Suggestion Generator) of the AI Sound
Advisor.

Spec requirements implemented here (project_master_plan.txt):
  - AI / LLM: "Mistral 7B (local model) — Run via: Ollama OR LM Studio API"
  - FAILSAFE BEHAVIOR: "If AI model fails -> fallback to basic alerts" and
    "System must never crash during service"
  - CONNECTION STATUS INDICATOR: "AI/LLM availability ... Must update in real time"
  - PERFORMANCE RULES: "AI (LLM) must NOT run every loop" — unchanged; the pacing
    agent (Agent 6) still owns *when* a call happens. This module only owns *how*.

Design
------
Each concrete client implements the existing ``src.suggestion.LLMClient``
interface (``generate(prompt) -> str``), so it drops straight into
``SuggestionGenerator`` with no other code changes.

The actual HTTP call to the local model server lives behind an injectable
``HttpTransport`` ABC (the project's standard pattern for external I/O). In
production the default ``UrllibHttpTransport`` uses only the standard library —
no extra dependency. In tests an in-memory fake is injected, so the clients are
exercised with no network and no running model.

Failure handling honours the failsafe contract: ``generate`` raises
``LLMUnavailableError`` on any backend problem (connection refused, timeout,
HTTP/JSON error, unexpected shape, empty output). ``SuggestionGenerator`` already
catches ``Exception`` and degrades to its built-in basic alerts, so a dead model
server can never interrupt a live service. ``is_available()`` is a never-raising
health probe for the dashboard's AI/LLM status indicator.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from enum import Enum
from typing import Mapping, Optional

from pydantic import BaseModel

from src.suggestion import DEFAULT_SYSTEM_PROMPT, LLMClient

# ---------------------------------------------------------------------------
# Defaults & environment configuration
# ---------------------------------------------------------------------------

#: Default Ollama HTTP endpoint (local Mistral 7B via `ollama serve`).
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: Default LM Studio OpenAI-compatible endpoint.
DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"

#: Default model tags for each backend.
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_LMSTUDIO_MODEL = "mistral-7b-instruct"

#: Request timeout (seconds). Generous because local generation can be slow.
DEFAULT_LLM_TIMEOUT = 30.0

#: Shared dashboard env-var prefix (mirrors web_server.ENV_PREFIX).
ENV_PREFIX = "SOUND_ADVISOR_"


# ===========================================================================
# Errors
# ===========================================================================

class LLMUnavailableError(RuntimeError):
    """Raised by ``generate`` when the local model cannot serve a request.

    A subclass of ``RuntimeError`` (hence ``Exception``) so the existing
    ``SuggestionGenerator`` failsafe catch falls back to basic alerts.
    """


# ===========================================================================
# HTTP transport interface + standard-library implementation
# ===========================================================================

class HttpTransport(ABC):
    """Minimal JSON-over-HTTP transport, injected so clients need no network."""

    @abstractmethod
    def post_json(self, url: str, payload: dict, timeout: float) -> dict:
        """POST ``payload`` as JSON to ``url`` and return the decoded JSON body."""

    @abstractmethod
    def get_json(self, url: str, timeout: float) -> dict:
        """GET ``url`` and return the decoded JSON body (used by health probes)."""


class UrllibHttpTransport(HttpTransport):
    """Production transport built on the standard-library ``urllib`` (no deps)."""

    def post_json(self, url: str, payload: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request, timeout)

    def get_json(self, url: str, timeout: float) -> dict:
        request = urllib.request.Request(url, method="GET")
        return self._send(request, timeout)

    @staticmethod
    def _send(request: "urllib.request.Request", timeout: float) -> dict:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


# ===========================================================================
# Concrete clients
# ===========================================================================

class _BaseHttpLLMClient(LLMClient):
    """Shared plumbing for HTTP-backed local LLM clients."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout: float,
        system: str,
        transport: Optional[HttpTransport],
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.system = system
        self._transport = transport or UrllibHttpTransport()

    # -- subclass hooks -------------------------------------------------
    def _generate_url(self) -> str:  # pragma: no cover - trivial
        raise NotImplementedError

    def _health_url(self) -> str:  # pragma: no cover - trivial
        raise NotImplementedError

    def _build_payload(self, prompt: str) -> dict:  # pragma: no cover - trivial
        raise NotImplementedError

    def _extract_text(self, body: dict) -> str:  # pragma: no cover - trivial
        raise NotImplementedError

    # -- public API -----------------------------------------------------
    def generate(self, prompt: str) -> str:
        """Return the model completion, or raise ``LLMUnavailableError``."""
        try:
            body = self._transport.post_json(
                self._generate_url(), self._build_payload(prompt), self.timeout
            )
            text = self._extract_text(body).strip()
        except LLMUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalise every backend failure
            raise LLMUnavailableError(
                f"{type(self).__name__} request failed: {exc}"
            ) from exc
        if not text:
            raise LLMUnavailableError(f"{type(self).__name__} returned no text")
        return text

    def is_available(self) -> bool:
        """Never-raising health probe for the AI/LLM status indicator."""
        try:
            self._transport.get_json(self._health_url(), self.timeout)
            return True
        except Exception:  # noqa: BLE001 — failsafe: report unavailable, don't crash
            return False


class OllamaLLMClient(_BaseHttpLLMClient):
    """Local Mistral 7B served by Ollama (``POST /api/generate``)."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_LLM_TIMEOUT,
        system: str = DEFAULT_SYSTEM_PROMPT,
        transport: Optional[HttpTransport] = None,
    ) -> None:
        super().__init__(
            model=model, base_url=base_url, timeout=timeout,
            system=system, transport=transport,
        )

    def _generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    def _health_url(self) -> str:
        return f"{self.base_url}/api/tags"

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "system": self.system,
            "stream": False,
        }

    def _extract_text(self, body: dict) -> str:
        if "response" not in body:
            raise LLMUnavailableError("Ollama response missing 'response' field")
        return str(body["response"])


class LMStudioLLMClient(_BaseHttpLLMClient):
    """Local Mistral 7B served by LM Studio's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = DEFAULT_LMSTUDIO_MODEL,
        base_url: str = DEFAULT_LMSTUDIO_URL,
        timeout: float = DEFAULT_LLM_TIMEOUT,
        system: str = DEFAULT_SYSTEM_PROMPT,
        transport: Optional[HttpTransport] = None,
    ) -> None:
        super().__init__(
            model=model, base_url=base_url, timeout=timeout,
            system=system, transport=transport,
        )

    def _generate_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _health_url(self) -> str:
        return f"{self.base_url}/models"

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "stream": False,
        }

    def _extract_text(self, body: dict) -> str:
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(
                "LM Studio response had an unexpected shape"
            ) from exc


# ===========================================================================
# Env-driven selection (app runs fine with no model: backend "none")
# ===========================================================================

class LLMBackend(str, Enum):
    """Which local model server to talk to (or NONE to stay on basic alerts)."""

    NONE = "none"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"

    @classmethod
    def parse(cls, raw: Optional[str]) -> "LLMBackend":
        """Case-insensitive parse; anything unrecognised -> NONE (safe default)."""
        if not raw:
            return cls.NONE
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.NONE


class LLMConfig(BaseModel):
    """Environment-sourced selection for the local LLM backend.

    Environment variables (all optional):
      SOUND_ADVISOR_LLM_BACKEND   none | ollama | lmstudio  (default none)
      SOUND_ADVISOR_LLM_MODEL     model tag       (default per backend)
      SOUND_ADVISOR_LLM_BASE_URL  server base URL (default per backend)
      SOUND_ADVISOR_LLM_TIMEOUT   request timeout seconds (default 30.0)
    """

    backend: LLMBackend = LLMBackend.NONE
    model: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = DEFAULT_LLM_TIMEOUT

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "LLMConfig":
        """Build a config from a mapping (defaults to ``os.environ``)."""
        import os

        env = os.environ if environ is None else environ
        raw_timeout = env.get(f"{ENV_PREFIX}LLM_TIMEOUT")
        try:
            timeout = (
                float(raw_timeout) if raw_timeout not in (None, "")
                else DEFAULT_LLM_TIMEOUT
            )
        except (TypeError, ValueError):
            timeout = DEFAULT_LLM_TIMEOUT
        return cls(
            backend=LLMBackend.parse(env.get(f"{ENV_PREFIX}LLM_BACKEND")),
            model=env.get(f"{ENV_PREFIX}LLM_MODEL") or None,
            base_url=env.get(f"{ENV_PREFIX}LLM_BASE_URL") or None,
            timeout=timeout,
        )


def build_llm_client(config: Optional[LLMConfig] = None) -> Optional[LLMClient]:
    """Construct the configured local LLM client, or ``None`` for basic alerts.

    Returning ``None`` (the default) keeps the app fully operational with no
    model running — ``SuggestionGenerator`` simply uses its built-in alerts.
    """
    cfg = config or LLMConfig.from_env()
    if cfg.backend is LLMBackend.OLLAMA:
        return OllamaLLMClient(
            model=cfg.model or DEFAULT_OLLAMA_MODEL,
            base_url=cfg.base_url or DEFAULT_OLLAMA_URL,
            timeout=cfg.timeout,
        )
    if cfg.backend is LLMBackend.LMSTUDIO:
        return LMStudioLLMClient(
            model=cfg.model or DEFAULT_LMSTUDIO_MODEL,
            base_url=cfg.base_url or DEFAULT_LMSTUDIO_URL,
            timeout=cfg.timeout,
        )
    return None
