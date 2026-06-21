"""
tests/test_network_config.py

AI Sound Advisor — persistent X32 network settings (spec NETWORKING INTERFACE).

Specification reference (project_master_plan.txt — NETWORKING INTERFACE):
  1. ARCHITECTURE & STORAGE
       - persistent JSON storage (config.json) for the network settings
       - if the file is missing, initialise it with the X32 factory defaults
         (IP 192.168.1.128, subnet 255.255.255.0, gateway 192.168.1.1,
          UDP port 10023, timeout 1500ms)
  2. SETTINGS UI WINDOW: save edited settings to disk
  3. TEST MECHANISM: a one-shot /info probe that does not disturb the master link

These tests cover the storage layer plus the three HTTP endpoints that the
Network Settings window talks to — all hardware-free (an in-memory OSC channel).

Run with:
  pytest tests/test_network_config.py -v
"""

import json

from fastapi.testclient import TestClient

from src.network_config import (
    FACTORY_DEFAULT_GATEWAY,
    FACTORY_DEFAULT_IP,
    FACTORY_DEFAULT_PORT,
    FACTORY_DEFAULT_SUBNET,
    FACTORY_DEFAULT_TIMEOUT_MS,
    NetworkConfigStore,
    NetworkSettings,
)

from tests.test_x32_connection import FakeChannel, X32_INFO


# ===========================================================================
# NetworkSettings model
# ===========================================================================

def test_settings_defaults_are_the_x32_factory_values():
    s = NetworkSettings()
    assert s.ip == FACTORY_DEFAULT_IP == "192.168.1.128"
    assert s.subnet == FACTORY_DEFAULT_SUBNET == "255.255.255.0"
    assert s.gateway == FACTORY_DEFAULT_GATEWAY == "192.168.1.1"
    assert s.port == FACTORY_DEFAULT_PORT == 10023
    assert s.timeout_ms == FACTORY_DEFAULT_TIMEOUT_MS == 1500


def test_timeout_s_converts_ms_to_seconds():
    assert NetworkSettings(timeout_ms=1500).timeout_s == 1.5


# ===========================================================================
# NetworkConfigStore: load / save / failsafe
# ===========================================================================

def test_load_creates_file_with_factory_defaults_when_missing(tmp_path):
    path = tmp_path / "config.json"
    store = NetworkConfigStore(path)
    assert not path.exists()

    settings = store.load()

    assert path.exists()                      # file auto-created
    assert settings == NetworkSettings()      # with factory defaults
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["ip"] == "192.168.1.128"
    assert on_disk["port"] == 10023


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "config.json"
    store = NetworkConfigStore(path)
    store.save(NetworkSettings(
        ip="10.1.2.3", subnet="255.255.0.0", gateway="10.1.0.1",
        port=10024, timeout_ms=2000,
    ))

    reloaded = NetworkConfigStore(path).load()
    assert reloaded.ip == "10.1.2.3"
    assert reloaded.subnet == "255.255.0.0"
    assert reloaded.gateway == "10.1.0.1"
    assert reloaded.port == 10024
    assert reloaded.timeout_ms == 2000


def test_corrupt_file_falls_back_to_defaults_without_overwriting(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ this is not valid json", encoding="utf-8")

    store = NetworkConfigStore(path)
    settings = store.load()

    assert settings == NetworkSettings()          # graceful fallback
    # The bad file is left intact so the operator can recover it manually.
    assert path.read_text(encoding="utf-8") == "{ this is not valid json"


# ===========================================================================
# HTTP endpoints (Network Settings window)
# ===========================================================================

def _app(tmp_path, *, channel=None):
    """Build a minimal dashboard app wired to a temp store + fake OSC channel."""
    from types import SimpleNamespace

    from src.web_server import create_app

    channel = channel or FakeChannel(X32_INFO)

    # A throwaway orchestrator/agents stub is enough for the network endpoints,
    # but create_app expects real-ish collaborators, so reuse the web test env.
    from tests.test_web import make_env
    env = make_env()
    # Swap in a real store + a fake-channel connection manager.
    from src.x32_connection import X32ConnectionManager
    store = NetworkConfigStore(tmp_path / "config.json")
    env.app.state.network_store = store
    env.app.state.x32_connection = X32ConnectionManager(
        channel_factory=lambda ip, port: channel
    )
    return SimpleNamespace(client=TestClient(env.app), store=store, channel=channel)


def test_get_settings_returns_factory_defaults_first_time(tmp_path):
    ctx = _app(tmp_path)
    resp = ctx.client.get("/api/network/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ip"] == "192.168.1.128"
    assert body["timeout_ms"] == 1500


def test_post_settings_persists_to_disk(tmp_path):
    ctx = _app(tmp_path)
    payload = {
        "ip": "192.168.0.55", "subnet": "255.255.255.0",
        "gateway": "192.168.0.1", "port": 10023, "timeout_ms": 1200,
    }
    resp = ctx.client.post("/api/network/settings", json=payload)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Persisted and re-loadable.
    reloaded = NetworkConfigStore(ctx.store.path).load()
    assert reloaded.ip == "192.168.0.55"
    assert reloaded.timeout_ms == 1200


def test_post_settings_rejects_bad_port(tmp_path):
    ctx = _app(tmp_path)
    payload = {
        "ip": "192.168.0.55", "subnet": "255.255.255.0",
        "gateway": "192.168.0.1", "port": 99999, "timeout_ms": 1200,
    }
    resp = ctx.client.post("/api/network/settings", json=payload)
    assert resp.status_code == 422   # FastAPI validation, not a crash


def test_test_endpoint_reports_console_on_success(tmp_path):
    ctx = _app(tmp_path)
    resp = ctx.client.post("/api/network/test", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["status_label"] == "Connected to: X32 Firmware 4.06"


def test_test_endpoint_reports_warning_on_no_response(tmp_path):
    ctx = _app(tmp_path, channel=FakeChannel(X32_INFO, fail_query=True))
    resp = ctx.client.post("/api/network/test", json={"ip": "10.0.0.9"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is False
    assert "10.0.0.9" in body["status_label"]


def test_saved_settings_become_connect_form_default(tmp_path):
    ctx = _app(tmp_path)
    ctx.client.post("/api/network/settings", json={
        "ip": "192.168.7.7", "subnet": "255.255.255.0",
        "gateway": "192.168.7.1", "port": 10023, "timeout_ms": 1500,
    })
    status = ctx.client.get("/api/x32/status").json()
    assert status["default_ip"] == "192.168.7.7"
