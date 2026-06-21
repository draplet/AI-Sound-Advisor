"""
Fake X32 Simulator - Entry point.

Starts the OSC server and displays the computer's local IP address.

Usage:
    python run_fake_x32.py

Then in Audio Pilot:
    - Go to X32 Network settings
    - Enter the displayed IP address
    - Port: 10023
    - Click Connect
"""
import logging
import socket
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from fake_x32.fake_x32_osc import FakeX32OscServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """
    Detect local IP address by connecting to a public DNS.
    This handles multi-homed systems correctly.
    """
    try:
        # Connect to a public DNS (doesn't actually send data)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        # Fallback: use localhost if DNS method fails
        logger.warning("Could not detect local IP, using localhost")
        return "127.0.0.1"


def main():
    """Start Fake X32 server and display connection info."""
    try:
        ip = get_local_ip()

        print("=" * 60)
        print("Fake X32 Simulator - Starting")
        print("=" * 60)
        print(f"Your IP address is: {ip}")
        print(f"OSC Port: 10023")
        print()
        print("To connect Audio Pilot:")
        print(f"  1. Enter IP: {ip}")
        print(f"  2. Port: 10023")
        print(f"  3. Click Connect")
        print()
        print("Starting Fake X32 OSC server on 0.0.0.0:10023")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        print()

        server = FakeX32OscServer()
        try:
            server.start()
        except KeyboardInterrupt:
            print("\nShutdown complete.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()


if __name__ == "__main__":
    main()
