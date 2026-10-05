"""Device enumeration and the live-capture WebSocket path.

Uses a fake capture backend throughout, so these run on any machine.
"""

import pytest
from fastapi.testclient import TestClient

from backend.tests.fake_capture import FakeCaptureBackend

START_CAPTURE = {
    "type": "start",
    "session_id": "cap-001",
    "source": "system_audio",
    "context": {"channel": "zoom-demo", "caller_type": "executive",
                "contact_known": True, "transaction_value": 2_500_000},
}


@pytest.fixture()
def capture_app(monkeypatch):
    """App with capture enabled and a hardware-free backend installed."""
    monkeypatch.setenv("DETECTOR_PROVIDER", "mock")
    monkeypatch.setenv("MOCK_LATENCY_MS", "0")
    monkeypatch.setenv("VAD_ENABLED", "false")
    monkeypatch.setenv("SYSTEM_AUDIO_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_AUDIO_BLOCK_MS", "20")
    from backend.app.config import get_settings
    get_settings.cache_clear()
    from backend.app.main import create_app
    app = create_app()
    yield app
    get_settings.cache_clear()


@pytest.fixture()
def capture_client(capture_app):
    with TestClient(capture_app) as client:
        capture_app.state.capture_backend = FakeCaptureBackend()
        capture_app.state.capture_unavailable_reason = None
        yield client


# ---- GET /audio/devices ----------------------------------------------
def test_devices_reports_disabled_by_default(client):
    body = client.get("/audio/devices").json()
    assert body["enabled"] is False
    assert body["devices"] == []
    assert "SYSTEM_AUDIO_ENABLED" in body["reason"]


def test_devices_lists_loopback_only(capture_client):
    body = capture_client.get("/audio/devices").json()
    assert body["enabled"] is True
    assert body["backend"] == "fake"
    assert body["devices"], "expected at least one loopback device"
    assert all(d["is_loopback"] for d in body["devices"])
    assert all("Microphone" not in d["label"] for d in body["devices"])


def test_devices_never_leak_the_backend_id(capture_client):
    raw = capture_client.get("/audio/devices").text
    assert "fake.monitor" not in raw
    for device in capture_client.get("/audio/devices").json()["devices"]:
        assert device["id"].startswith("dev-")
        assert set(device) == {"id", "label", "channels", "is_loopback"}


def test_devices_reports_reason_when_backend_missing(capture_app):
    with TestClient(capture_app) as client:
        capture_app.state.capture_backend = None
        capture_app.state.capture_unavailable_reason = "SoundCard is not installed"
        body = client.get("/audio/devices").json()
    assert body["enabled"] is False
    assert body["reason"] == "SoundCard is not installed"


def test_config_advertises_the_capability(capture_client):
    assert capture_client.get("/config").json()["system_audio_enabled"] is True


# ---- WebSocket capture sessions --------------------------------------
def drain_until(ws, kind, limit=400):
    events = []
    for _ in range(limit):
        msg = ws.receive_json()
        events.append(msg)
        if msg.get("type") == kind:
            return events
    raise AssertionError(f"never received a {kind!r} event")


def test_capture_session_runs_end_to_end(capture_client):
    with capture_client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START_CAPTURE)
        assert ws.receive_json()["type"] == "session_started"

        started = ws.receive_json()
        assert started["type"] == "capture_started"
        assert started["source"] == "system_audio"
        assert started["device_label"] == "Fake Output (monitor)"
        assert started["sample_rate"] == 48000

        events = drain_until(ws, "analysis")
        analysis = events[-1]
        assert analysis["windows_analyzed"] >= 1
        assert "synthetic_probability" in analysis

        ws.send_json({"type": "stop"})
        final = drain_until(ws, "final")[-1]

    assert final["session_id"] == "cap-001"
    assert final["windows_analyzed"] >= 1
    assert 0.0 <= final["final_score"] <= 1.0


def test_client_audio_is_rejected_during_capture(capture_client):
    """The server is pulling audio; client frames are a protocol error."""
    with capture_client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START_CAPTURE)
        ws.receive_json(); ws.receive_json(); ws.receive_json()
        ws.send_bytes(b"\x00\x01" * 100)
        codes = []
        for _ in range(40):
            msg = ws.receive_json()
            if msg.get("type") == "error":
                codes.append(msg["code"])
                break
        assert codes == ["INVALID_MESSAGE"]


def test_capture_refused_when_feature_disabled(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START_CAPTURE)
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "CAPTURE_UNAVAILABLE"


def test_capture_refused_when_backend_unavailable(capture_app):
    with TestClient(capture_app) as client:
        capture_app.state.capture_backend = None
        capture_app.state.capture_unavailable_reason = "SoundCard is not installed"
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json(START_CAPTURE)
            error = ws.receive_json()
    assert error["code"] == "CAPTURE_UNAVAILABLE"
    assert "SoundCard" in error["message"]


def test_unknown_device_id_is_refused(capture_client):
    with capture_client.websocket_connect("/ws/stream") as ws:
        ws.send_json({**START_CAPTURE, "device_id": "dev-deadbeef"})
        error = ws.receive_json()
    assert error["code"] == "CAPTURE_UNAVAILABLE"
    assert "unknown capture device" in error["message"]


def test_failed_device_open_leaves_no_session(capture_app):
    with TestClient(capture_app) as client:
        capture_app.state.capture_backend = FakeCaptureBackend(
            open_error=RuntimeError("device busy"))
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json(START_CAPTURE)
            assert ws.receive_json()["type"] == "session_started"
            error = ws.receive_json()
            assert error["code"] == "CAPTURE_FAILED"
            assert "device busy" in error["message"]
        assert capture_app.state.sessions.active_count == 0


def test_capture_session_is_released_on_disconnect(capture_client, capture_app):
    """An abrupt disconnect must still release the device and the session.

    Server-side teardown runs after the client socket closes, so this waits for
    release rather than asserting it has already happened.
    """
    import time

    with capture_client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START_CAPTURE)
        ws.receive_json(); ws.receive_json(); ws.receive_json()
        assert capture_app.state.sessions.active_count == 1

    deadline = time.monotonic() + 5.0
    while capture_app.state.sessions.active_count and time.monotonic() < deadline:
        time.sleep(0.02)
    assert capture_app.state.sessions.active_count == 0


def test_two_capture_sessions_in_a_row(capture_client):
    """The device must be released cleanly so a second session can start."""
    for session_id in ("cap-a", "cap-b"):
        with capture_client.websocket_connect("/ws/stream") as ws:
            ws.send_json({**START_CAPTURE, "session_id": session_id})
            assert ws.receive_json()["type"] == "session_started"
            assert ws.receive_json()["type"] == "capture_started"
            drain_until(ws, "analysis")
            ws.send_json({"type": "stop"})
            final = drain_until(ws, "final")[-1]
        assert final["session_id"] == session_id
        assert final["windows_analyzed"] >= 1


def test_file_upload_still_works_when_capture_is_enabled(capture_client, speech_16k):
    """Enabling capture must not disturb the original upload path."""
    from backend.app.ingestion.normalizer import float_to_pcm16

    with capture_client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "session_id": "upload-1", "source": "demo_web",
                      "audio_format": {"encoding": "pcm_s16le",
                                       "sample_rate": 16000, "channels": 1}})
        ws.receive_json(); ws.receive_json()
        for i in range(0, speech_16k.size, 4000):
            ws.send_bytes(float_to_pcm16(speech_16k[i:i + 4000]))
        ws.send_json({"type": "stop"})
        final = drain_until(ws, "final")[-1]
    assert final["windows_analyzed"] >= 1


# ---- session limit (audit M3) ----------------------------------------
def test_capture_actually_stops_at_the_session_limit(capture_app):
    """"Capture stopped" must mean capture stopped (audit M3).

    Exceeding MAX_SESSION_SECONDS only broke out of the pump, so the server
    stopped *reading* while the capture thread kept recording the machine's
    output until the client disconnected — with the UI still showing its
    "CAPTURING SYSTEM AUDIO" banner.
    """
    import time

    capture_app.state.settings.max_session_seconds = 1
    backend = FakeCaptureBackend()
    with TestClient(capture_app) as client:
        capture_app.state.capture_backend = backend
        capture_app.state.capture_unavailable_reason = None
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json(START_CAPTURE)
            ws.receive_json(); ws.receive_json(); ws.receive_json()

            codes = []
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                msg = ws.receive_json()
                codes.append(msg.get("code") or msg.get("type"))
                if msg.get("code") == "SESSION_TOO_LONG":
                    break
            assert "SESSION_TOO_LONG" in codes, codes

            # Capture must be winding down, not still recording.
            time.sleep(0.3)
            served = backend.recorder.blocks_served
            time.sleep(0.3)
            assert backend.recorder.blocks_served == served, (
                "the capture thread kept recording after reporting it had stopped"
            )
