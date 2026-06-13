"""
src/setup_assistant.py

Hardware connection verification for the AI Sound Advisor.

Spec requirements implemented here:
  - "If X32 disconnects -> show warning"
  - "If Audio input lost -> show warning"
  - "System must never crash during service" / "Continue operating where possible"
  - "Connection Status Indicator: Must update in real time"

verify_connections() is guaranteed never to raise: any unexpected internal
error is swallowed and reported as an operational-but-degraded SetupReport.
"""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ConnectionStatus(str, Enum):
    """Real-time connection state for a single hardware component."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class HardwareWarning(BaseModel):
    """A validated warning payload describing one disconnected component."""

    component: str = Field(min_length=1)
    message: str = Field(min_length=1)


class SetupReport(BaseModel):
    """
    Structured, validated result of a connection check.

    is_operational reflects the failsafe guarantee: the system reports that
    it is still running even when hardware is missing.
    """

    x32_status: ConnectionStatus
    audio_status: ConnectionStatus
    warnings: List[HardwareWarning] = Field(default_factory=list)
    is_operational: bool = True


class SystemSetupAssistant:
    """
    Inspects the (mocked) hardware connection flags and produces a SetupReport.

    The connection flags are injected at construction time so the assistant is
    fully testable without real hardware.
    """

    def __init__(self, x32_connected: bool, audio_input_connected: bool) -> None:
        self.x32_connected = x32_connected
        self.audio_input_connected = audio_input_connected

    def verify_connections(self) -> SetupReport:
        """
        Build a SetupReport for the current connection flags.

        Never raises: on any unexpected failure it degrades gracefully to an
        operational report so the live service is not interrupted.
        """
        try:
            warnings: List[HardwareWarning] = []

            x32_status = (
                ConnectionStatus.CONNECTED
                if self.x32_connected
                else ConnectionStatus.DISCONNECTED
            )
            if not self.x32_connected:
                warnings.append(
                    HardwareWarning(
                        component="X32",
                        message="X32 mixer is disconnected. Check the network/USB link.",
                    )
                )

            audio_status = (
                ConnectionStatus.CONNECTED
                if self.audio_input_connected
                else ConnectionStatus.DISCONNECTED
            )
            if not self.audio_input_connected:
                warnings.append(
                    HardwareWarning(
                        component="AudioInput",
                        message="Audio input signal lost. Check the audio interface.",
                    )
                )

            return SetupReport(
                x32_status=x32_status,
                audio_status=audio_status,
                warnings=warnings,
                is_operational=True,
            )
        except Exception:  # noqa: BLE001 — failsafe: never crash during service
            return SetupReport(
                x32_status=ConnectionStatus.DISCONNECTED,
                audio_status=ConnectionStatus.DISCONNECTED,
                warnings=[
                    HardwareWarning(
                        component="System",
                        message="Connection verification failed; running in degraded mode.",
                    )
                ],
                is_operational=True,
            )
