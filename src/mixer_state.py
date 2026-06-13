"""
src/mixer_state.py

Agent 2: Mixer State Agent (OSC) for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — AGENT 2):
  - "Monitor and track X32 mixer state via OSC"
  - "Read channel levels (faders, gain)"
  - "Read mute states"
  - "Read channel labels"
  - Output Example:
        {"channel_3": {"name": "Lead Vocal", "fader": -10, "mute": false}}

Failsafe (system-wide spec):
  - "If X32 disconnects -> show warning"
  - "System must never crash during service"
  - "Continue operating where possible"

Design
------
The Behringer X32 is queried over UDP/OSC. All network I/O sits behind the
``X32OscTransport`` interface so the agent is fully testable with an in-memory
fake — no sockets, no hardware. ``UdpOscTransport`` is the real python-osc
implementation used in production.

X32 OSC value conventions handled here:
  - Fader/level is a normalised float 0.0..1.0 → converted to dB.
  - ``/ch/NN/mix/on`` is 1 when the channel is ON (audible); muted == (on == 0).
  - Head-amp gain is a normalised float 0.0..1.0 over the -12..+60 dB range.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of input channels on an X32 (full-size console).
X32_CHANNEL_COUNT = 32

#: dB value reported for a fully-closed fader (X32 shows -oo).
FADER_MIN_DB: float = -90.0

#: Head-amp gain range (dB) corresponding to the normalised 0.0..1.0 value.
GAIN_MIN_DB: float = -12.0
GAIN_MAX_DB: float = 60.0

#: Default UDP port the X32 listens on for OSC.
X32_OSC_PORT = 10023


# ===========================================================================
# Value conversions (X32 OSC protocol)
# ===========================================================================

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fader_float_to_db(value: float) -> float:
    """Convert an X32 normalised fader position (0.0..1.0) to dB.

    Piecewise mapping from the X32 OSC protocol:
        f >= 0.5     -> 40f - 30   (  0.5 -> -10 dB, 1.0 -> +10 dB)
        f >= 0.25    -> 80f - 50   ( 0.25 -> -30 dB)
        f >= 0.0625  -> 160f - 70  (0.0625 -> -60 dB)
        f >  0.0     -> 480f - 90
        f == 0.0     -> FADER_MIN_DB (-oo floor)
    """
    f = _clamp(float(value), 0.0, 1.0)
    if f >= 0.5:
        return f * 40.0 - 30.0
    if f >= 0.25:
        return f * 80.0 - 50.0
    if f >= 0.0625:
        return f * 160.0 - 70.0
    if f > 0.0:
        return f * 480.0 - 90.0
    return FADER_MIN_DB


def fader_db_to_float(db: float) -> float:
    """Inverse of :func:`fader_float_to_db` (dB -> normalised 0.0..1.0)."""
    d = float(db)
    if d >= -10.0:
        return _clamp((d + 30.0) / 40.0, 0.0, 1.0)
    if d >= -30.0:
        return (d + 50.0) / 80.0
    if d >= -60.0:
        return (d + 70.0) / 160.0
    if d > FADER_MIN_DB:
        return (d + 90.0) / 480.0
    return 0.0


def gain_float_to_db(value: float) -> float:
    """Convert an X32 normalised head-amp gain (0.0..1.0) to dB (-12..+60)."""
    f = _clamp(float(value), 0.0, 1.0)
    return f * (GAIN_MAX_DB - GAIN_MIN_DB) + GAIN_MIN_DB


# ===========================================================================
# OSC address builders
# ===========================================================================

def _validate_index(index: int) -> int:
    if not isinstance(index, int) or not (1 <= index <= X32_CHANNEL_COUNT):
        raise ValueError(
            f"channel index must be an int in 1..{X32_CHANNEL_COUNT}, got {index!r}"
        )
    return index


def channel_name_address(index: int) -> str:
    """OSC address for a channel's scribble-strip name (e.g. /ch/03/config/name)."""
    return f"/ch/{_validate_index(index):02d}/config/name"


def channel_fader_address(index: int) -> str:
    """OSC address for a channel's fader level (e.g. /ch/03/mix/fader)."""
    return f"/ch/{_validate_index(index):02d}/mix/fader"


def channel_mute_address(index: int) -> str:
    """OSC address for a channel's on/off state (e.g. /ch/03/mix/on)."""
    return f"/ch/{_validate_index(index):02d}/mix/on"


def channel_gain_address(index: int) -> str:
    """OSC address for a channel's head-amp gain (zero-based headamp index)."""
    return f"/headamp/{_validate_index(index) - 1:03d}/gain"


# ===========================================================================
# Structured output models
# ===========================================================================

class ChannelState(BaseModel):
    """Validated state of a single mixer channel (spec output shape)."""

    index: int
    name: str
    fader: float  # dB
    mute: bool
    gain: float = 0.0  # dB (head-amp)


class MixerState(BaseModel):
    """Snapshot of the tracked mixer state plus connection health.

    ``connected`` and ``warnings`` carry the failsafe signal: when the X32 is
    unreachable the snapshot is still returned (never an exception), with
    ``connected=False`` and a human-readable warning.
    """

    connected: bool
    channels: Dict[str, ChannelState] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


# ===========================================================================
# OSC transport interface + real implementation
# ===========================================================================

class X32OscTransport(ABC):
    """Interface for querying the X32 over OSC.

    ``query`` sends an OSC address and returns the mixer's response value.
    Implementations raise on connection/timeout failure; the agent's
    ``read_state`` translates that into a graceful disconnected snapshot.
    """

    @abstractmethod
    def query(self, address: str) -> Any:
        """Return the X32's response value for ``address`` (raises on failure)."""


class UdpOscTransport(X32OscTransport):
    """Real UDP/OSC transport for the X32, built on python-osc.

    The X32 replies to a parameter query on the same socket it was queried
    from, so a single UDP socket is used to send the request and receive the
    matching reply within ``timeout`` seconds.
    """

    def __init__(self, ip: str, port: int = X32_OSC_PORT, timeout: float = 1.0) -> None:
        # Imported lazily so the module loads even where python-osc is absent.
        from pythonosc.udp_client import SimpleUDPClient

        self._ip = ip
        self._port = port
        self._client = SimpleUDPClient(ip, port)
        self._client._sock.settimeout(timeout)

    def query(self, address: str) -> Any:
        import socket

        from pythonosc.osc_message import OscMessage
        from pythonosc.osc_message_builder import OscMessageBuilder

        request = OscMessageBuilder(address=address).build()
        sock = self._client._sock
        try:
            sock.sendto(request.dgram, (self._ip, self._port))
            data, _ = sock.recvfrom(65535)
        except socket.timeout as exc:
            raise TimeoutError(f"X32 did not respond to {address}") from exc
        except OSError as exc:
            raise ConnectionError(f"X32 OSC query failed for {address}: {exc}") from exc

        params = list(OscMessage(data).params)
        if len(params) == 1:
            return params[0]
        return params


# ===========================================================================
# Agent
# ===========================================================================

def _first(value: Any) -> Any:
    """Unwrap a single OSC argument that may arrive inside a list/tuple."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


class MixerStateAgent:
    """Reads and tracks X32 channel state via an injected OSC transport."""

    def __init__(self, transport: X32OscTransport) -> None:
        self._transport = transport

    def read_channel(self, index: int) -> ChannelState:
        """Query and build the state of a single channel.

        Low-level call: it propagates transport errors. Use :meth:`read_state`
        for the failsafe, never-raising aggregation used by the live loop.
        """
        name = _first(self._transport.query(channel_name_address(index)))
        fader_raw = _first(self._transport.query(channel_fader_address(index)))
        on_raw = _first(self._transport.query(channel_mute_address(index)))
        gain_raw = _first(self._transport.query(channel_gain_address(index)))

        return ChannelState(
            index=index,
            name="" if name is None else str(name),
            fader=fader_float_to_db(float(fader_raw)),
            mute=int(on_raw) == 0,  # X32: on==1 audible, on==0 muted
            gain=gain_float_to_db(float(gain_raw)),
        )

    def read_state(self, channels: Optional[Iterable[int]] = None) -> MixerState:
        """Read the requested channels into a MixerState snapshot.

        Failsafe: this never raises. If the X32 is unreachable it returns a
        snapshot with ``connected=False`` and a warning so the rest of the
        system can keep operating and surface a status indicator.
        """
        if channels is None:
            channels = range(1, X32_CHANNEL_COUNT + 1)

        result: Dict[str, ChannelState] = {}
        try:
            for index in channels:
                result[f"channel_{index}"] = self.read_channel(index)
        except Exception as exc:  # noqa: BLE001 — failsafe: never crash during service
            return MixerState(
                connected=False,
                channels={},
                warnings=[
                    "X32 mixer is not responding. Check the network connection. "
                    f"({type(exc).__name__}: {exc})"
                ],
            )

        return MixerState(connected=True, channels=result, warnings=[])
