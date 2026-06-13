"""
tests/test_mixer_state.py

AI Sound Advisor — Agent 2: Mixer State Agent (OSC)
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — AGENT 2):
  - "Monitor and track X32 mixer state via OSC"
  - "Read channel levels (faders, gain)"
  - "Read mute states"
  - "Read channel labels"
  - Output Example:
        {
          "channel_3": {
            "name": "Lead Vocal",
            "fader": -10,
            "mute": false
          }
        }
  Failsafe (system-wide spec):
  - "If X32 disconnects -> show warning"
  - "System must never crash during service"
  - "Continue operating where possible"

Design note:
  The X32 is reached over UDP/OSC via an injectable ``X32OscTransport``.
  Tests use an in-memory fake transport so the agent is verified with NO real
  hardware and NO network, exactly mirroring how Agent 1's strategies and the
  SetupAssistant are tested.

Modules under test (DO NOT EXIST YET — intended TDD red state):
  src.mixer_state  →  MixerStateAgent, X32OscTransport, ChannelState,
                      MixerState, fader_float_to_db, fader_db_to_float,
                      gain_float_to_db, channel_name_address,
                      channel_fader_address, channel_mute_address,
                      channel_gain_address, FADER_MIN_DB

Run with:
  pytest tests/test_mixer_state.py -v
"""

import pytest

from src.mixer_state import (
    MixerStateAgent,
    X32OscTransport,
    ChannelState,
    MixerState,
    fader_float_to_db,
    fader_db_to_float,
    gain_float_to_db,
    channel_name_address,
    channel_fader_address,
    channel_mute_address,
    channel_gain_address,
    FADER_MIN_DB,
)


# ===========================================================================
# Test doubles — in-memory OSC transport (no sockets, no hardware)
# ===========================================================================

def make_channel_responses(index, name, fader_f, on, gain_f):
    """Build the four OSC address->value entries the agent queries per channel."""
    return {
        channel_name_address(index): name,
        channel_fader_address(index): fader_f,
        channel_mute_address(index): on,
        channel_gain_address(index): gain_f,
    }


class FakeX32Transport(X32OscTransport):
    """A scripted OSC transport.

    Returns canned values per address, or raises a preconfigured exception to
    simulate a disconnected / unresponsive mixer.
    """

    def __init__(self, responses=None, raise_exc=None):
        self.responses = responses or {}
        self.raise_exc = raise_exc
        self.queried = []

    def query(self, address):
        if self.raise_exc is not None:
            raise self.raise_exc
        self.queried.append(address)
        return self.responses[address]


# ===========================================================================
# SECTION 1 — X32 fader float<->dB conversion
#
# X32 normalises fader position to 0.0..1.0; the spec output uses dB.
# Reference mapping points (X32 OSC protocol):
#   1.0000 -> +10 dB,  0.7500 -> 0 dB,  0.5000 -> -10 dB,
#   0.2500 -> -30 dB,  0.0625 -> -60 dB, 0.0000 -> -oo (floor)
# ===========================================================================

class TestFaderConversion:

    @pytest.mark.parametrize(
        "float_value, expected_db",
        [
            (1.0, 10.0),
            (0.75, 0.0),
            (0.5, -10.0),     # the spec's "fader": -10 example
            (0.25, -30.0),
            (0.0625, -60.0),
        ],
    )
    def test_fader_float_to_db_known_points(self, float_value, expected_db):
        assert fader_float_to_db(float_value) == pytest.approx(expected_db, abs=1e-9)

    def test_fader_zero_maps_to_floor(self):
        assert fader_float_to_db(0.0) == pytest.approx(FADER_MIN_DB)

    def test_fader_input_is_clamped_above_one(self):
        """Out-of-range floats must not produce nonsense; >1.0 clamps to +10 dB."""
        assert fader_float_to_db(1.5) == pytest.approx(10.0)

    @pytest.mark.parametrize(
        "db_value, expected_float",
        [
            (10.0, 1.0),
            (0.0, 0.75),
            (-10.0, 0.5),
            (-30.0, 0.25),
            (-60.0, 0.0625),
        ],
    )
    def test_fader_db_to_float_known_points(self, db_value, expected_float):
        assert fader_db_to_float(db_value) == pytest.approx(expected_float, abs=1e-9)

    @pytest.mark.parametrize("float_value", [0.05, 0.2, 0.5, 0.73, 0.95, 1.0])
    def test_fader_round_trip_float_to_db_to_float(self, float_value):
        round_tripped = fader_db_to_float(fader_float_to_db(float_value))
        assert round_tripped == pytest.approx(float_value, abs=1e-9)


# ===========================================================================
# SECTION 2 — X32 head-amp gain float->dB conversion (range -12..+60 dB)
# ===========================================================================

class TestGainConversion:

    @pytest.mark.parametrize(
        "float_value, expected_db",
        [(0.0, -12.0), (1.0, 60.0), (0.5, 24.0)],
    )
    def test_gain_float_to_db_endpoints(self, float_value, expected_db):
        assert gain_float_to_db(float_value) == pytest.approx(expected_db, abs=1e-9)


# ===========================================================================
# SECTION 3 — OSC address builders (X32 protocol addressing)
# ===========================================================================

class TestAddressBuilders:

    def test_channel_addresses_are_two_digit_padded(self):
        assert channel_name_address(3) == "/ch/03/config/name"
        assert channel_fader_address(3) == "/ch/03/mix/fader"
        assert channel_mute_address(3) == "/ch/03/mix/on"

    def test_channel_one_is_zero_padded(self):
        assert channel_fader_address(1) == "/ch/01/mix/fader"

    def test_gain_address_uses_zero_based_three_digit_headamp(self):
        # Channel 1 -> headamp 000, channel 32 -> headamp 031
        assert channel_gain_address(1) == "/headamp/000/gain"
        assert channel_gain_address(32) == "/headamp/031/gain"

    @pytest.mark.parametrize("bad_index", [0, -1, 33, 100])
    def test_invalid_channel_index_raises(self, bad_index):
        with pytest.raises(ValueError):
            channel_fader_address(bad_index)


# ===========================================================================
# SECTION 4 — ChannelState / MixerState models (Pydantic validation)
# ===========================================================================

class TestModels:

    def test_channel_state_holds_core_fields(self):
        ch = ChannelState(index=3, name="Lead Vocal", fader=-10.0, mute=False, gain=12.0)
        assert ch.name == "Lead Vocal"
        assert ch.fader == -10.0
        assert ch.mute is False

    def test_mixer_state_defaults_to_empty_and_disconnected_fields(self):
        state = MixerState(connected=True, channels={})
        assert state.connected is True
        assert state.channels == {}
        assert state.warnings == []


# ===========================================================================
# SECTION 5 — MixerStateAgent reads channel state via OSC
# ===========================================================================

class TestMixerStateAgentReads:

    def test_read_channel_returns_populated_channel_state(self):
        """SPEC PRIMARY: channel 3 = "Lead Vocal", fader 0.5 (-10 dB), unmuted."""
        transport = FakeX32Transport(
            make_channel_responses(3, name="Lead Vocal", fader_f=0.5, on=1, gain_f=0.5)
        )
        agent = MixerStateAgent(transport)

        channel = agent.read_channel(3)

        assert channel.name == "Lead Vocal"
        assert channel.fader == pytest.approx(-10.0)
        assert channel.mute is False
        assert channel.gain == pytest.approx(24.0)

    def test_mute_is_true_when_x32_channel_is_off(self):
        """X32 '/mix/on' == 0 means the channel is OFF, i.e. muted."""
        transport = FakeX32Transport(
            make_channel_responses(5, name="Choir", fader_f=0.75, on=0, gain_f=0.3)
        )
        agent = MixerStateAgent(transport)

        assert agent.read_channel(5).mute is True

    def test_mute_is_false_when_x32_channel_is_on(self):
        transport = FakeX32Transport(
            make_channel_responses(5, name="Choir", fader_f=0.75, on=1, gain_f=0.3)
        )
        agent = MixerStateAgent(transport)

        assert agent.read_channel(5).mute is False

    def test_transport_value_wrapped_in_list_is_unwrapped(self):
        """python-osc may deliver a single value inside a 1-tuple/list."""
        transport = FakeX32Transport(
            {
                channel_name_address(2): ["Kick"],
                channel_fader_address(2): [0.5],
                channel_mute_address(2): [1],
                channel_gain_address(2): [0.0],
            }
        )
        agent = MixerStateAgent(transport)

        channel = agent.read_channel(2)
        assert channel.name == "Kick"
        assert channel.fader == pytest.approx(-10.0)
        assert channel.mute is False


# ===========================================================================
# SECTION 6 — read_state aggregates channels (spec output shape)
# ===========================================================================

class TestReadState:

    def _two_channel_transport(self):
        responses = {}
        responses.update(
            make_channel_responses(3, "Lead Vocal", fader_f=0.5, on=1, gain_f=0.5)
        )
        responses.update(
            make_channel_responses(4, "Acoustic", fader_f=0.75, on=0, gain_f=0.25)
        )
        return FakeX32Transport(responses)

    def test_read_state_keys_channels_as_channel_n(self):
        agent = MixerStateAgent(self._two_channel_transport())

        state = agent.read_state([3, 4])

        assert state.connected is True
        assert "channel_3" in state.channels
        assert "channel_4" in state.channels

    def test_read_state_matches_spec_example_shape(self):
        agent = MixerStateAgent(self._two_channel_transport())

        state = agent.read_state([3])
        dumped = state.channels["channel_3"].model_dump()

        # Spec example fields must all be present with the expected values.
        assert dumped["name"] == "Lead Vocal"
        assert dumped["fader"] == pytest.approx(-10.0)
        assert dumped["mute"] is False

    def test_read_state_reads_each_requested_channel(self):
        agent = MixerStateAgent(self._two_channel_transport())

        state = agent.read_state([3, 4])

        assert state.channels["channel_4"].name == "Acoustic"
        assert state.channels["channel_4"].mute is True  # on==0


# ===========================================================================
# SECTION 7 — Failsafe: a disconnected X32 must never crash the system
# ===========================================================================

class TestFailsafe:

    def test_read_state_does_not_raise_when_transport_fails(self):
        """SPEC: 'System must never crash during service.'"""
        transport = FakeX32Transport(raise_exc=ConnectionError("no route to host"))
        agent = MixerStateAgent(transport)

        try:
            agent.read_state([1, 2, 3])
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"read_state raised on transport failure: {exc!r}")

    def test_read_state_reports_disconnected_and_warns_on_failure(self):
        """SPEC: 'If X32 disconnects -> show warning' / 'Continue operating'."""
        transport = FakeX32Transport(raise_exc=TimeoutError("timed out"))
        agent = MixerStateAgent(transport)

        state = agent.read_state([1, 2, 3])

        assert state.connected is False
        assert len(state.warnings) >= 1
        assert state.channels == {}

    def test_returned_object_is_always_a_mixer_state(self):
        transport = FakeX32Transport(raise_exc=OSError("socket error"))
        agent = MixerStateAgent(transport)

        assert isinstance(agent.read_state([1]), MixerState)
