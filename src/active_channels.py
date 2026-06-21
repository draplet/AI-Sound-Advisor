"""
src/active_channels.py

"Active Channels" monitor for the AI Sound Advisor dashboard.

Gives per-channel visibility using data the X32 already exposes over OSC — the
channel scribble-strip name, fader level and mute — without needing per-channel
audio. It periodically scans all channels and reports the ones that are actually
in use: unmuted and with the fader above a threshold (default -60 dB).

Why a separate monitor (not the per-tick loop)
----------------------------------------------
Reading every channel is ~4 OSC round-trips x 32 channels, which is far too slow
to do on the 3 Hz analysis loop. So this monitor scans at most once per
``refresh_interval_ms`` (default 60 s) and caches the result. The scan is gated
on the master connection being up, so it does no network I/O (and never blocks
on timeouts) while disconnected.

Design
------
The monitor takes an injected :class:`~src.mixer_state.MixerStateAgent` (whose
transport is the real X32 in production, or an in-memory fake in tests) and an
``is_connected`` callable. It is failsafe: ``MixerStateAgent.read_state`` never
raises, and a disconnected/unreachable mixer yields an ``available=False``
snapshot with a warning rather than an error.
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from src.mixer_state import X32_CHANNEL_COUNT, MixerStateAgent

#: Faders above this level (dB), when unmuted, count as "active". A fully closed
#: fader sits at FADER_MIN_DB (-90 dB), so this excludes parked channels.
ACTIVE_FADER_THRESHOLD_DB: float = -60.0

#: How often the list is allowed to re-scan the console (ms). 30 s is fresh
#: enough to track fader moves while keeping OSC traffic to ~2 scans/min.
DEFAULT_REFRESH_INTERVAL_MS: int = 30_000

#: Per-query OSC timeout (seconds) to use *for the scan transport*. Kept short so
#: a network hiccup mid-scan can't make a scan drag on — read_state aborts on the
#: first slow query. Use when building the scanner's UdpOscTransport.
SCAN_QUERY_TIMEOUT_S: float = 0.25


class ActiveChannel(BaseModel):
    """One in-use channel for the dashboard list."""

    index: int
    name: str
    fader_db: float
    gain_db: float = 0.0


class ActiveChannelsSnapshot(BaseModel):
    """Cached result of the most recent scan, shaped for the UI."""

    available: bool = False
    channels: List[ActiveChannel] = Field(default_factory=list)
    #: Clock value (ms) of the last scan; None before the first one.
    scanned_ms: Optional[int] = None
    warning: Optional[str] = None
    threshold_db: float = ACTIVE_FADER_THRESHOLD_DB


class ActiveChannelsMonitor:
    """Scans the X32 (lazily, gated on connection) for in-use channels."""

    def __init__(
        self,
        mixer_agent: MixerStateAgent,
        is_connected: Callable[[], bool],
        *,
        channel_count: int = X32_CHANNEL_COUNT,
        threshold_db: float = ACTIVE_FADER_THRESHOLD_DB,
        refresh_interval_ms: int = DEFAULT_REFRESH_INTERVAL_MS,
    ) -> None:
        self._agent = mixer_agent
        self._is_connected = is_connected
        self._channels = list(range(1, channel_count + 1))
        self._threshold = threshold_db
        self._interval = refresh_interval_ms
        self._lock = threading.Lock()
        #: Held while a scan is in flight, so concurrent requests don't kick off
        #: overlapping console scans (acquired non-blocking).
        self._scan_lock = threading.Lock()
        self._snap = ActiveChannelsSnapshot(threshold_db=threshold_db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> ActiveChannelsSnapshot:
        """Return the cached snapshot without scanning."""
        with self._lock:
            return self._snap

    def maybe_refresh(self, now_ms: int) -> ActiveChannelsSnapshot:
        """Re-scan if the cache is stale (older than the refresh interval).

        Returns the current snapshot either way. Cheap to call on every request:
        the actual OSC scan happens at most once per ``refresh_interval_ms``.
        """
        with self._lock:
            last = self._snap.scanned_ms
        if last is not None and (now_ms - last) < self._interval:
            return self.snapshot()
        return self.refresh(now_ms)

    def refresh(self, now_ms: int) -> ActiveChannelsSnapshot:
        """Scan the console now and cache the result (failsafe, never raises).

        If a scan is already in flight (another request triggered one), this
        returns the cached snapshot instead of starting a second overlapping
        console scan.
        """
        if not self._scan_lock.acquire(blocking=False):
            return self.snapshot()
        try:
            snap = self._scan(now_ms)
            with self._lock:
                self._snap = snap
            return snap
        finally:
            self._scan_lock.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _scan(self, now_ms: int) -> ActiveChannelsSnapshot:
        # Gate on the master connection so we never block on UDP timeouts while
        # the console is not connected.
        try:
            connected = bool(self._is_connected())
        except Exception:  # noqa: BLE001 — failsafe: treat as not connected
            connected = False
        if not connected:
            return ActiveChannelsSnapshot(
                available=False, scanned_ms=now_ms,
                warning="X32 not connected — connect to see active channels.",
                threshold_db=self._threshold,
            )

        state = self._agent.read_state(self._channels)  # never raises
        if not state.connected:
            warning = state.warnings[0] if state.warnings else "X32 not responding."
            return ActiveChannelsSnapshot(
                available=False, scanned_ms=now_ms, warning=warning,
                threshold_db=self._threshold,
            )

        active = [
            ActiveChannel(
                index=ch.index, name=ch.name, fader_db=ch.fader, gain_db=ch.gain,
            )
            for ch in state.channels.values()
            if not ch.mute and ch.fader > self._threshold
        ]
        # Loudest (highest fader) first.
        active.sort(key=lambda a: a.fader_db, reverse=True)
        return ActiveChannelsSnapshot(
            available=True, channels=active, scanned_ms=now_ms,
            threshold_db=self._threshold,
        )
