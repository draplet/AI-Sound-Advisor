"""
Fake X32 OSC Server.

Minimal OSC server that impersonates a Behringer X32 mixer.
Listens on UDP 10023 and responds to /info and /xremote messages.
"""
import logging
import socket
from typing import Optional

from pythonosc import osc_message, osc_message_builder

logger = logging.getLogger(__name__)


class FakeX32OscServer:
    """Simple UDP-based OSC server mimicking X32 behavior."""

    X32_INFO = ["V2.07", "fake-x32-simulator", "X32", "4.06"]
    PORT = 10023
    BUFFER_SIZE = 1024

    def __init__(self, host: str = "0.0.0.0", port: int = PORT):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.running = False

    def start(self) -> None:
        """Bind socket and enter receive loop. Blocking call."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

        logger.info(f"Fake X32 OSC server listening on {self.host}:{self.port}")
        self.running = True

        try:
            while self.running:
                self._receive_and_dispatch()
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        except Exception as e:
            logger.error(f"OSC server error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully close socket."""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _receive_and_dispatch(self) -> None:
        """Receive one OSC message and dispatch to handler."""
        try:
            data, addr = self.sock.recvfrom(self.BUFFER_SIZE)
            self._dispatch_message(data, addr)
        except socket.timeout:
            pass
        except Exception as e:
            logger.warning(f"Error processing message: {e}")

    def _dispatch_message(self, data: bytes, addr: tuple) -> None:
        """Parse OSC message and route to handler."""
        try:
            msg = osc_message.OscMessage(data)
            address = msg.address

            if address == "/info":
                response = self._handle_info()
            elif address == "/xremote":
                response = self._handle_xremote()
            else:
                logger.debug(f"Ignoring unknown OSC address: {address}")
                return

            # Send response back to client
            self.sock.sendto(response, addr)

        except Exception as e:
            logger.warning(f"Failed to parse/dispatch message: {e}")

    def _handle_info(self) -> bytes:
        """Return /info response: ["V2.07", "fake-x32-simulator", "X32", "4.06"]."""
        builder = osc_message_builder.OscMessageBuilder(address="/info")
        for param in self.X32_INFO:
            builder.add_arg(param)
        return builder.build().dgram

    def _handle_xremote(self) -> bytes:
        """Return /xremote acknowledgement (empty message)."""
        builder = osc_message_builder.OscMessageBuilder(address="/xremote")
        return builder.build().dgram
