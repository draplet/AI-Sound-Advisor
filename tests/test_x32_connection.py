"""
tests/test_x32_connection.py

AI Sound Advisor — X32 master-connection workflow.

Specification reference (project_master_plan.txt):
  - "OSC / X32 Communication: python-osc"
  - "CONNECTION STATUS INDICATOR — Displays X32 connection ... Must update in
     real time. If connection fails: show warning, continue operating."
  - NETWORK CONNECTION / X32 CONNECT WORKFLOW:
      send /info, parse console name + firmware into a status label, and run a
      background /xremote keepalive every 9 s with graceful drop handling.

Design note:
  The X32 is reached over an injectable ``OscChannel``. These tests use an
  in-memory fake channel so the workflow is verified with NO real hardware and
  NO network — mirroring how the rest of the agents are tested.

Run with:
  pytest tests/test_x32_connection.py -v
"""

import threading
import time

import pytest

from src.x32_connection import (
    OscChannel,
    X32ConnectionManager,
    X32Info,
    parse_info_reply,
)


# ===========================================================================
# Fakes
# ===========================================================================

class FakeChannel(OscChannel):
    """In-memory OSC channel: scripted /info reply, records every send."""

    def __init__(self, info_reply, *, fail_query=False, fail_send_after=None):
        self._info_reply = info_reply
        self._fail_query = fail_query
        self._fail_send_after = fail_send_after
        self.sends = []                 # addresses passed to send()
        self.closed = False
        self._send_event = threading.Event()

    def query(self, address, timeout):
        if self._fail_query:
            raise TimeoutError("no response")
        if address == "/info":
            return self._info_reply
        return []

    def send(self, address):
        self.sends.append(address)
        self._send_event.set()
        if self._fail_send_after is not None and len(self.sends) >= self._fail_send_after:
            raise ConnectionError("link dropped")

    def close(self):
        self.closed = True

    def wait_for_send(self, timeout=1.0):
        return self._send_event.wait(timeout)


X32_INFO = ["V2.07", "osc-server", "X32", "4.06"]


# ===========================================================================
# parse_info_reply / X32Info
# ===========================================================================

def test_parse_info_reply_extracts_model_and_firmware():
    info = parse_info_reply(X32_INFO)
    assert info.server_version == "V2.07"
    assert info.console_model == "X32"
    assert info.console_firmware == "4.06"
    assert info.raw == ["V2.07", "osc-server", "X32", "4.06"]


def test_status_label_matches_spec_example():
    info = parse_info_reply(X32_INFO)
    assert info.status_label == "Connected to: X32 Firmware 4.06"


def test_parse_info_reply_tolerates_short_reply():
    info = parse_info_reply(["X32RACK"])
    # First arg lands in server_version; missing fields stay blank, no raise.
    assert info.server_version == "X32RACK"
    assert info.console_model == ""
    assert info.console_firmware == ""


def test_parse_info_reply_handles_none():
    info = parse_info_reply(None)
    assert info == X32Info()


# ===========================================================================
# connect(): /info handshake
# ===========================================================================

def test_connect_sends_info_and_reports_connected():
    channel = FakeChannel(X32_INFO)
    mgr = X32ConnectionManager(channel_factory=lambda ip, port: channel,
                               keepalive_interval=0.05)
    state = mgr.connect("192.168.0.2", 10023)
    try:
        assert state.connected is True
        assert state.ip == "192.168.0.2"
        assert state.port == 10023
        assert state.status_label == "Connected to: X32 Firmware 4.06"
        assert state.info.console_firmware == "4.06"
    finally:
        mgr.disconnect()


def test_connect_failure_returns_warning_not_exception():
    channel = FakeChannel(X32_INFO, fail_query=True)
    mgr = X32ConnectionManager(channel_factory=lambda ip, port: channel)
    state = mgr.connect("10.0.0.9", 10023)
    assert state.connected is False
    assert state.warning is not None
    assert "10.0.0.9" in state.status_label
    # A failed handshake closes the channel it opened.
    assert channel.closed is True


def test_connect_passes_configured_ip_port_to_factory():
    captured = {}

    def factory(ip, port):
        captured["ip"] = ip
        captured["port"] = port
        return FakeChannel(X32_INFO)

    mgr = X32ConnectionManager(channel_factory=factory, keepalive_interval=0.05)
    mgr.connect("172.16.5.4", 10024)
    try:
        assert captured == {"ip": "172.16.5.4", "port": 10024}
    finally:
        mgr.disconnect()


# ===========================================================================
# /xremote keepalive
# ===========================================================================

def test_keepalive_sends_xremote_after_connect():
    channel = FakeChannel(X32_INFO)
    mgr = X32ConnectionManager(channel_factory=lambda ip, port: channel,
                               keepalive_interval=0.02)
    mgr.connect("192.168.0.2", 10023)
    try:
        assert channel.wait_for_send(timeout=1.0)
        # Give the loop a moment to send a few more.
        time.sleep(0.1)
        assert channel.sends.count("/xremote") >= 1
        assert all(addr == "/xremote" for addr in channel.sends)
    finally:
        mgr.disconnect()


def test_disconnect_stops_keepalive_and_closes_channel():
    channel = FakeChannel(X32_INFO)
    mgr = X32ConnectionManager(channel_factory=lambda ip, port: channel,
                               keepalive_interval=0.02)
    mgr.connect("192.168.0.2", 10023)
    assert channel.wait_for_send(timeout=1.0)
    state = mgr.disconnect()

    assert state.connected is False
    assert channel.closed is True
    sends_at_disconnect = len(channel.sends)
    time.sleep(0.1)
    # No further /xremote sends after disconnect.
    assert len(channel.sends) == sends_at_disconnect


def test_dropped_keepalive_is_handled_gracefully():
    # Channel raises on the 2nd send -> simulates the X32 dropping off the net.
    channel = FakeChannel(X32_INFO, fail_send_after=2)
    mgr = X32ConnectionManager(channel_factory=lambda ip, port: channel,
                               keepalive_interval=0.02)
    mgr.connect("192.168.0.2", 10023)
    try:
        # Poll until the manager notices the drop (never raises).
        deadline = time.time() + 1.0
        while time.time() < deadline and mgr.status().connected:
            time.sleep(0.02)
        state = mgr.status()
        assert state.connected is False
        assert state.warning is not None
        assert "lost" in state.warning.lower()
    finally:
        mgr.disconnect()


# ===========================================================================
# status() / reconnect
# ===========================================================================

def test_status_starts_idle():
    mgr = X32ConnectionManager(channel_factory=lambda ip, port: FakeChannel(X32_INFO))
    state = mgr.status()
    assert state.connected is False
    assert state.status_label == "Not connected"


def test_reconnect_tears_down_previous_channel():
    first = FakeChannel(X32_INFO)
    second = FakeChannel(X32_INFO)
    channels = iter([first, second])
    mgr = X32ConnectionManager(channel_factory=lambda ip, port: next(channels),
                               keepalive_interval=0.05)
    mgr.connect("192.168.0.2", 10023)
    mgr.connect("192.168.0.3", 10023)   # reconnect -> first channel closed
    try:
        assert first.closed is True
        assert mgr.status().ip == "192.168.0.3"
    finally:
        mgr.disconnect()
