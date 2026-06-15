"""
src/network_config.py

Persistent network settings for the X32 (spec: NETWORKING INTERFACE).

Spec requirements implemented here (project_master_plan.txt):
  1. ARCHITECTURE & STORAGE
       - "Create a persistent JSON storage system (`config.json`) to save
          network settings."
       - "If the file is missing, automatically initialize it with the factory
          default shipping settings of a Behringer X32:
            * Default IP: 192.168.1.128
            * Default Subnet: 255.255.255.0
            * Default Gateway: 192.168.1.1
            * Default UDP Target Port: 10023
            * Default Local Timeout: 1500ms"

Design
------
:class:`NetworkSettings` is the validated shape of those five values.
:class:`NetworkConfigStore` reads/writes them to ``config.json``. It is a
failsafe component — a missing file is created from the factory defaults, and a
corrupt/unreadable file falls back to the factory defaults rather than raising,
so the dashboard always starts (spec: "System must never crash during service").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from src.mixer_state import X32_OSC_PORT

# ---------------------------------------------------------------------------
# Factory defaults — the shipping settings of a Behringer X32.
# ---------------------------------------------------------------------------

FACTORY_DEFAULT_IP = "192.168.1.128"
FACTORY_DEFAULT_SUBNET = "255.255.255.0"
FACTORY_DEFAULT_GATEWAY = "192.168.1.1"
FACTORY_DEFAULT_PORT = X32_OSC_PORT          # 10023
FACTORY_DEFAULT_TIMEOUT_MS = 1500

#: Default filename for the persisted settings (project root).
DEFAULT_CONFIG_FILENAME = "config.json"


# ===========================================================================
# Settings model
# ===========================================================================

class NetworkSettings(BaseModel):
    """The five persisted X32 network values, seeded with factory defaults."""

    ip: str = FACTORY_DEFAULT_IP
    subnet: str = FACTORY_DEFAULT_SUBNET
    gateway: str = FACTORY_DEFAULT_GATEWAY
    port: int = Field(default=FACTORY_DEFAULT_PORT, ge=1, le=65535)
    #: Connection / reply timeout in milliseconds.
    timeout_ms: int = Field(default=FACTORY_DEFAULT_TIMEOUT_MS, ge=1)

    @property
    def timeout_s(self) -> float:
        """Timeout expressed in seconds, for the OSC layer."""
        return self.timeout_ms / 1000.0


# ===========================================================================
# Persistent store
# ===========================================================================

class NetworkConfigStore:
    """Loads/saves :class:`NetworkSettings` to ``config.json``.

    The store is failsafe: construction never touches the disk, and both
    :meth:`load` and :meth:`save` degrade to the in-memory factory defaults
    instead of raising. A missing file is auto-created on first load.
    """

    def __init__(self, path: str | Path = DEFAULT_CONFIG_FILENAME) -> None:
        self._path = Path(path)
        self._settings = NetworkSettings()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def settings(self) -> NetworkSettings:
        """The currently held settings (call :meth:`load` to read from disk)."""
        return self._settings

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self) -> NetworkSettings:
        """Read settings from disk, creating the file from defaults if missing.

        On a missing file: write the factory defaults and return them. On a
        corrupt/partial/unreadable file: return the factory defaults without
        overwriting the bad file (so the operator can recover it manually).
        """
        if not self._path.exists():
            self._settings = NetworkSettings()
            self.save()  # materialise factory defaults to disk
            return self._settings

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._settings = NetworkSettings.model_validate(raw)
        except (OSError, ValueError, ValidationError):
            # Corrupt or unreadable -> fall back to defaults, leave file intact.
            self._settings = NetworkSettings()
        return self._settings

    def save(self, settings: Optional[NetworkSettings] = None) -> NetworkSettings:
        """Persist ``settings`` (or the held settings) to ``config.json``.

        Returns the settings now held. Never raises: a write failure leaves the
        in-memory settings updated so the running session still reflects them.
        """
        if settings is not None:
            self._settings = settings
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._settings.model_dump(), indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass  # failsafe: keep the in-memory value even if the disk write fails
        return self._settings
