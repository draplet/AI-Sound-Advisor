"""
Tests for Fake X32 OSC Server (Phase 1).

Tests verify that the OSC server:
- Listens on UDP 10023
- Responds to /info with correct X32 identifier
- Responds to /xremote keepalive
- Doesn't crash on unknown messages
- Displays local IP on startup
"""
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
from pythonosc import osc_message_builder


# Helper: Build and send OSC message via UDP
def send_osc(address: str, args: list = None, host: str = "127.0.0.1", port: int = 10023) -> bytes:
    """Send OSC message and receive response."""
    builder = osc_message_builder.OscMessageBuilder(address=address)
    if args:
        for arg in args:
            builder.add_arg(arg)
    msg = builder.build()

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2.0)
    client.sendto(msg.dgram, (host, port))
    try:
        response, _ = client.recvfrom(1024)
        return response
    finally:
        client.close()


@pytest.fixture(scope="module")
def fake_x32_server():
    """Start Fake X32 OSC server in background thread."""
    from fake_x32.fake_x32_osc import FakeX32OscServer

    server = FakeX32OscServer(host="127.0.0.1", port=10023)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()

    # Wait for server to be ready
    time.sleep(0.5)

    yield server

    # Cleanup
    server.stop()
    thread.join(timeout=2.0)


class TestFakeX32Info:
    """Test /info OSC message response."""

    def test_info_response_format(self, fake_x32_server):
        """Query /info and verify response is valid OSC message."""
        response = send_osc("/info")
        assert response is not None
        assert len(response) > 0

    def test_info_response_contains_firmware(self, fake_x32_server):
        """Parse /info response and verify firmware identifier."""
        from pythonosc import osc_message
        response = send_osc("/info")
        msg = osc_message.OscMessage(response)
        params = msg.params

        # Expected: ["V2.07", "fake-x32-simulator", "X32", "4.06"]
        assert len(params) >= 4
        assert params[0] == "V2.07"
        assert params[1] == "fake-x32-simulator"
        assert params[2] == "X32"
        assert params[3] == "4.06"

    def test_info_response_identifies_as_x32(self, fake_x32_server):
        """Verify /info identifies as X32 so Audio Pilot recognizes it."""
        from pythonosc import osc_message
        response = send_osc("/info")
        msg = osc_message.OscMessage(response)
        params = msg.params

        # Audio Pilot looks for "X32" in position 2
        assert params[2] == "X32"


class TestFakeX32Keepalive:
    """Test /xremote OSC message (keepalive)."""

    def test_xremote_acknowledged(self, fake_x32_server):
        """Query /xremote and verify server responds."""
        response = send_osc("/xremote")
        assert response is not None

    def test_xremote_multiple_sends(self, fake_x32_server):
        """Send /xremote multiple times (like Audio Pilot's 9s keepalive)."""
        for _ in range(3):
            response = send_osc("/xremote")
            assert response is not None
            time.sleep(0.1)


class TestFakeX32Robustness:
    """Test server doesn't crash on invalid/unknown messages."""

    def test_unknown_address_ignored(self, fake_x32_server):
        """Send unknown OSC address, server should ignore gracefully."""
        # Send unknown address (should not crash)
        try:
            send_osc("/unknown/address")
        except socket.timeout:
            # Timeout is OK - server may not respond to unknown addresses
            pass

        # Server should still be responsive
        response = send_osc("/info")
        assert response is not None

    def test_malformed_message_ignored(self, fake_x32_server):
        """Send raw bytes (malformed OSC), server should ignore gracefully."""
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(2.0)
        try:
            # Send garbage bytes
            client.sendto(b"garbage data that is not OSC", ("127.0.0.1", 10023))
        except Exception:
            pass  # Ignore send errors
        finally:
            client.close()

        # Server should still be responsive
        time.sleep(0.1)
        response = send_osc("/info")
        assert response is not None


class TestFakeX32Startup:
    """Test server startup behavior."""

    def test_startup_script_exists(self):
        """Verify run_fake_x32.py exists."""
        script_path = Path("fake_x32/run_fake_x32.py")
        assert script_path.exists(), f"{script_path} not found"

    def test_startup_prints_ip(self):
        """Run startup script and verify it prints IP address to stdout."""
        script_path = Path("fake_x32/run_fake_x32.py")

        # Run script with timeout (will block on server.start(), but stdout appears first)
        proc = subprocess.Popen(
            ["python", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Give it 1 second to startup and print
        time.sleep(1.0)

        # Kill the process
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=2.0)

        # Verify output contains IP address
        assert "Your IP address is:" in stdout, f"IP not in output: {stdout}"
        assert "Starting Fake X32 OSC server" in stdout, f"Startup msg not in output: {stdout}"
