"""
src/x32_connection.py

X32 master-connection workflow for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt):
  - "OSC / X32 Communication: python-osc (UDP/OSC communication with X32)"
  - "CONNECTION STATUS INDICATOR — Displays X32 connection (Green/Red) ...
     Must update in real time. If connection fails: show warning, continue
     operating where possible."
  - NETWORK CONNECTION / X32 CONNECT WORKFLOW (added to the master plan):
      1. On Connect, send an OSC ``/info`` packet to the configured IP/Port.
      2. Parse the reply and show the console model + firmware in a status
         label (e.g. "Connected to: X32 Firmware 4.06").
      3. While connected, a background thread re-sends ``/xremote`` every 9 s
         so the console keeps streaming parameter changes to us.
      4. Connection drops are handled gracefully — never crashes the service.

Design
------
All network I/O sits behind the small :class:`OscChannel` interface, so the
manager is fully testable with an in-memory fake — no sockets, no hardware.
:class:`UdpOscChannel` is the real python-osc implementation used live.

X32 protocol notes
------------------
  - ``/info`` replies with four strings:
        [server_version, server_name, console_model, console_firmware]
    e.g. ["V2.07", "osc-server", "X32", "4.06"].
  - ``/xremote`` subscribes this client to all parameter changes for ~10 s,
    so it must be re-sent on a shorter cycle (9 s here) to stay subscribed.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field

from src.mixer_state import X32_OSC_PORT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: How often to re-send /xremote (seconds). The X32 keeps a client subscribed
#: for ~10 s, so 9 s leaves a safety margin against drops.
XREMOTE_INTERVAL_S: float = 9.0

#: Default timeout (seconds) waiting for the /info reply.
INFO_TIMEOUT_S: float = 1.0


# ===========================================================================
# Structured /info result
# ===========================================================================

class X32Info(BaseModel):
    """Parsed result of an X32 ``/info`` query."""

    server_version: str = ""
    server_name: str = ""
    console_model: str = ""
    console_firmware: str = ""
    #: The raw OSC arguments, kept for diagnostics.
    raw: List[str] = Field(default_factory=list)

    @property
    def status_label(self) -> str:
        """Human label for the UI, e.g. 'Connected to: X32 Firmware 4.06'."""
        model = self.console_model or "console"
        if self.console_firmware:
            return f"Connected to: {model} Firmware {self.console_firmware}"
        return f"Connected to: {model}"


def parse_info_reply(params: Any) -> X32Info:
    """Build an :class:`X32Info` from the raw OSC ``/info`` arguments.

    Tolerant of short/odd replies: missing fields are left blank rather than
    raising, so a partial response still yields a usable status.
    """
    if params is None:
        items: List[str] = []
    elif isinstance(params, (list, tuple)):
        items = [("" if p is None else str(p)) for p in params]
    else:
        items = [str(params)]

    def at(i: int) -> str:
        return items[i] if i < len(items) else ""

    return X32Info(
        server_version=at(0),
        server_name=at(1),
        console_model=at(2),
        console_firmware=at(3),
        raw=items,
    )


# ===========================================================================
# OSC channel interface + real implementation
# ===========================================================================

class OscChannel(ABC):
    """A bidirectional OSC channel to one X32 (send + request/reply)."""

    @abstractmethod
    def query(self, address: str, timeout: float) -> Any:
        """Send ``address`` and return the console's reply args (raises on failure)."""

    @abstractmethod
    def send(self, address: str) -> None:
        """Fire-and-forget an OSC message (no reply expected)."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any underlying socket. Safe to call more than once."""


class UdpOscChannel(OscChannel):
    """Real UDP/OSC channel for the X32, built on python-osc.

    The X32 replies on the same socket it was queried from, so a single UDP
    socket handles both the ``/info`` request/reply and the ``/xremote`` sends.
    """

    def __init__(self, ip: str, port: int = X32_OSC_PORT) -> None:
        # Imported lazily so the module loads even where python-osc is absent.
        from pythonosc.udp_client import SimpleUDPClient

        self._ip = ip
        self._port = port
        self._client = SimpleUDPClient(ip, port)

    def query(self, address: str, timeout: float) -> Any:
        import socket

        from pythonosc.osc_message import OscMessage
        from pythonosc.osc_message_builder import OscMessageBuilder

        request = OscMessageBuilder(address=address).build()
        sock = self._client._sock
        previous = sock.gettimeout()
        sock.settimeout(timeout)
        try:
            sock.sendto(request.dgram, (self._ip, self._port))
            data, _ = sock.recvfrom(65535)
        except socket.timeout as exc:
            raise TimeoutError(f"X32 did not respond to {address}") from exc
        except OSError as exc:
            raise ConnectionError(f"X32 OSC query failed for {address}: {exc}") from exc
        finally:
            sock.settimeout(previous)

        return list(OscMessage(data).params)

    def send(self, address: str) -> None:
        from pythonosc.osc_message_builder import OscMessageBuilder

        message = OscMessageBuilder(address=address).build()
        self._client._sock.sendto(message.dgram, (self._ip, self._port))

    def close(self) -> None:
        try:
            self._client._sock.close()
        except Exception:  # noqa: BLE001 — failsafe: closing must never raise
            pass


def _default_channel_factory(ip: str, port: int) -> OscChannel:
    """Build a real UDP channel (the live default)."""
    return UdpOscChannel(ip, port)


# ===========================================================================
# Connection status snapshot
# ===========================================================================

class ConnectionState(BaseModel):
    """Snapshot of the master connection, shaped for the dashboard."""

    connected: bool = False
    ip: Optional[str] = None
    port: Optional[int] = None
    #: UI status label ("Connected to: X32 Firmware 4.06" / a warning / idle).
    status_label: str = "Not connected"
    info: Optional[X32Info] = None
    warning: Optional[str] = None


# ===========================================================================
# Connection manager
# ===========================================================================

class X32ConnectionManager:
    """Owns the master X32 connection: ``/info`` handshake + ``/xremote`` keepalive.

    Thread-safe. ``connect`` is synchronous (it sends ``/info`` and waits for the
    reply); on success it starts a daemon thread that re-sends ``/xremote`` every
    :data:`XREMOTE_INTERVAL_S` seconds. A failed handshake or a dropped keepalive
    leaves the manager in a clean disconnected state with a human-readable
    warning — it never raises into the caller.
    """

    def __init__(
        self,
        channel_factory: Callable[[str, int], OscChannel] = _default_channel_factory,
        *,
        keepalive_interval: float = XREMOTE_INTERVAL_S,
        info_timeout: float = INFO_TIMEOUT_S,
    ) -> None:
        self._channel_factory = channel_factory
        self._keepalive_interval = keepalive_interval
        self._info_timeout = info_timeout

        self._lock = threading.RLock()
        self._channel: Optional[OscChannel] = None
        self._state = ConnectionState()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self, ip: str, port: int = X32_OSC_PORT) -> ConnectionState:
        """Open a channel, send ``/info``, parse the reply, start the keepalive.

        Never raises: any failure returns a disconnected snapshot carrying a
        warning the UI can display.
        """
        self.disconnect()  # tear down any previous connection first

        try:
            channel = self._channel_factory(ip, port)
        except Exception as exc:  # noqa: BLE001 — failsafe
            return self._set_failed(ip, port, f"Could not open OSC channel: {exc}")

        try:
            reply = channel.query("/info", self._info_timeout)
        except Exception as exc:  # noqa: BLE001 — failsafe (timeout / network)
            try:
                channel.close()
            except Exception:  # noqa: BLE001
                pass
            return self._set_failed(
                ip, port,
                f"No response from {ip}:{port}. Check the network connection. "
                f"({type(exc).__name__})",
            )

        info = parse_info_reply(reply)
        with self._lock:
            self._channel = channel
            self._stop = threading.Event()
            self._state = ConnectionState(
                connected=True, ip=ip, port=port,
                status_label=info.status_label, info=info, warning=None,
            )
            self._start_keepalive_locked()
        return self._state

    def disconnect(self) -> ConnectionState:
        """Stop the keepalive thread, close the channel, go idle. Idempotent."""
        with self._lock:
            self._stop.set()
            thread = self._keepalive_thread
            channel = self._channel
            self._keepalive_thread = None
            self._channel = None

        # Join outside the lock so the keepalive thread can acquire it to exit.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._keepalive_interval + 1.0)
        if channel is not None:
            try:
                channel.close()
            except Exception:  # noqa: BLE001 — failsafe
                pass

        with self._lock:
            # Preserve a drop warning if one was recorded; otherwise go idle.
            if self._state.connected:
                self._state = ConnectionState()
            return self._state

    def status(self) -> ConnectionState:
        """Return the current connection snapshot (a copy is unnecessary — frozen read)."""
        with self._lock:
            return self._state

    # ------------------------------------------------------------------
    # Keepalive (background /xremote subscription)
    # ------------------------------------------------------------------

    def _start_keepalive_locked(self) -> None:
        """Start the daemon keepalive thread. Caller must hold the lock."""
        thread = threading.Thread(
            target=self._keepalive_loop,
            args=(self._channel, self._stop),
            name="x32-xremote-keepalive",
            daemon=True,
        )
        self._keepalive_thread = thread
        thread.start()

    def _keepalive_loop(self, channel: OscChannel, stop: threading.Event) -> None:
        """Re-send ``/xremote`` until stopped or the link drops.

        Sends immediately on start (to subscribe right away), then every
        ``keepalive_interval`` seconds. A send failure is treated as a dropped
        connection: it records a warning and exits cleanly.
        """
        while not stop.is_set():
            try:
                channel.send("/xremote")
            except Exception as exc:  # noqa: BLE001 — failsafe: a drop must not crash
                self._record_drop(stop, f"X32 connection lost: {exc}")
                return
            # Wait returns True if signalled to stop, False on timeout.
            if stop.wait(self._keepalive_interval):
                return

    def _record_drop(self, stop: threading.Event, message: str) -> None:
        """Mark the connection as dropped (graceful) if this loop still owns it."""
        with self._lock:
            if self._stop is not stop:
                return  # a newer connection superseded us; stay quiet
            previous_ip = self._state.ip
            previous_port = self._state.port
            self._state = ConnectionState(
                connected=False, ip=previous_ip, port=previous_port,
                status_label=message, warning=message,
            )
            self._channel = None
            self._keepalive_thread = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_failed(self, ip: str, port: int, message: str) -> ConnectionState:
        with self._lock:
            self._state = ConnectionState(
                connected=False, ip=ip, port=port,
                status_label=message, warning=message,
            )
            return self._state
