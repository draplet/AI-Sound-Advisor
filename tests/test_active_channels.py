"""
tests/test_active_channels.py

Tests for the Active Channels monitor (src.active_channels): it lists the
in-use X32 channels (unmuted, fader above threshold), scans at most once per
interval, and is gated/failsafe on the connection.
"""
from src.active_channels import (
    ACTIVE_FADER_THRESHOLD_DB,
    ActiveChannelsMonitor,
)
from src.mixer_state import MixerStateAgent, X32OscTransport
from src.mixer_state import (
    channel_fader_address,
    channel_gain_address,
    channel_mute_address,
    channel_name_address,
    fader_db_to_float,
)


class FakeX32Transport(X32OscTransport):
    """In-memory X32: serves name/fader/on/gain per channel from a dict.

    ``channels`` maps index -> (name, fader_db, on, gain_db). Unknown channels
    answer as a closed, named-empty channel.
    """

    def __init__(self, channels):
        self._channels = channels

    def query(self, address):
        for index, (name, fader_db, on, gain_db) in self._channels.items():
            if address == channel_name_address(index):
                return name
            if address == channel_fader_address(index):
                return fader_db_to_float(fader_db)
            if address == channel_mute_address(index):
                return on
            if address == channel_gain_address(index):
                return 0.5
        # Defaults for any channel not configured: closed + on.
        return "" if address.endswith("/config/name") else (1 if address.endswith("/mix/on") else 0.0)


def make_monitor(channels, connected=True, **kw):
    agent = MixerStateAgent(FakeX32Transport(channels))
    return ActiveChannelsMonitor(
        agent, is_connected=lambda: connected, channel_count=4, **kw
    )


class TestActiveChannelsFiltering:

    def test_lists_only_unmuted_above_threshold(self):
        # ch1: up + on -> active; ch2: muted -> out; ch3: down low -> out;
        # ch4: up + on -> active.
        channels = {
            1: ("Lead Vocal", -5.0, 1, 0.0),
            2: ("Drums", -3.0, 0, 0.0),      # muted
            3: ("Spare", -80.0, 1, 0.0),     # below -60 dB
            4: ("Acoustic", -12.0, 1, 0.0),
        }
        snap = make_monitor(channels).refresh(now_ms=1000)
        assert snap.available is True
        names = [c.name for c in snap.channels]
        assert names == ["Lead Vocal", "Acoustic"]  # loudest first

    def test_threshold_is_configurable(self):
        channels = {1: ("A", -50.0, 1, 0.0), 2: ("B", -5.0, 1, 0.0)}
        snap = make_monitor(channels, threshold_db=-20.0).refresh(now_ms=0)
        assert [c.name for c in snap.channels] == ["B"]

    def test_default_threshold(self):
        assert ACTIVE_FADER_THRESHOLD_DB == -60.0


class TestGatingAndFailsafe:

    def test_not_connected_reports_unavailable(self):
        snap = make_monitor({1: ("A", -5.0, 1, 0.0)}, connected=False).refresh(0)
        assert snap.available is False
        assert snap.channels == []
        assert snap.warning


class TestRefreshCadence:

    def test_maybe_refresh_caches_within_interval(self):
        calls = {"n": 0}

        class CountingTransport(FakeX32Transport):
            def query(self, address):
                calls["n"] += 1
                return super().query(address)

        agent = MixerStateAgent(CountingTransport({1: ("A", -5.0, 1, 0.0)}))
        mon = ActiveChannelsMonitor(
            agent, is_connected=lambda: True, channel_count=2,
            refresh_interval_ms=60_000,
        )
        mon.maybe_refresh(now_ms=0)
        after_first = calls["n"]
        assert after_first > 0
        # Within the interval -> no new scan.
        mon.maybe_refresh(now_ms=30_000)
        assert calls["n"] == after_first
        # Past the interval -> re-scan.
        mon.maybe_refresh(now_ms=61_000)
        assert calls["n"] > after_first
