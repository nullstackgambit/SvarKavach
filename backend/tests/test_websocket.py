"""End-to-end transport tests: frontend request -> ... -> response (plan §30)."""

import json
import threading
import time

import numpy as np
import pytest

from backend.app.ingestion.normalizer import float_to_pcm16

START = {
    "type": "start",
    "session_id": "test-001",
    "source": "demo_web",
    "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
    "context": {"channel": "zoom-demo", "caller_type": "executive",
                "contact_known": True, "transaction_value": 2_500_000},
}


def stream_audio(ws, audio, chunk=4000):
    for i in range(0, audio.size, chunk):
        ws.send_bytes(float_to_pcm16(audio[i:i + chunk]))


def drain(ws):
    """Collect events until the final message arrives."""
    events = []
    while True:
        msg = ws.receive_json()
        events.append(msg)
        if msg.get("type") == "final":
            return events


def test_session_starts_and_reports_detector(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        started = ws.receive_json()
        assert started["type"] == "session_started"
        assert started["session_id"] == "test-001"
        assert started["status"] == "ready"
        assert started["is_mock"] is True
        assert started["window_seconds"] == pytest.approx(4.0375)
        assert ws.receive_json() == {"type": "status", "status": "processing"}


def test_full_session_produces_analysis_and_final(client, speech_16k):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        stream_audio(ws, speech_16k)
        ws.send_json({"type": "stop"})
        events = drain(ws)

    analyses = [e for e in events if e["type"] == "analysis"]
    final = events[-1]
    assert analyses, "expected at least one analysis event"
    assert final["type"] == "final"
    assert final["windows_analyzed"] >= 1
    assert 0.0 <= final["final_score"] <= 1.0
    assert final["risk_state"] in {
        "NONE", "WATCH", "VERIFY", "ESCALATE", "INSUFFICIENT_EVIDENCE"}
    assert final["recommendation"].strip()
    assert final["is_mock"] is True


def test_analysis_events_carry_the_documented_fields(client, speech_16k):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        stream_audio(ws, speech_16k)
        ws.send_json({"type": "stop"})
        events = drain(ws)

    scored = [e for e in events if e["type"] == "analysis" and "synthetic_probability" in e]
    assert scored
    first = scored[0]
    for field in ("session_id", "window_start", "window_end", "synthetic_probability",
                  "smoothed_score", "risk_state", "confidence", "model", "inference_ms"):
        assert field in first, f"missing {field}"
    assert first["window_end"] > first["window_start"]
    assert first["inference_ms"] >= 0


def test_recommendation_event_follows_state(client, speech_16k):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        stream_audio(ws, speech_16k)
        ws.send_json({"type": "stop"})
        events = drain(ws)
    recs = [e for e in events if e["type"] == "recommendation"]
    assert recs and all(r["message"].strip() for r in recs)


def test_malformed_json_is_reported_and_survivable(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text("{not json")
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "INVALID_JSON"
        # the connection must still be usable
        ws.send_json(START)
        assert ws.receive_json()["type"] == "session_started"


def test_non_object_json_is_rejected(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps([1, 2, 3]))
        assert ws.receive_json()["code"] == "INVALID_MESSAGE"


def test_unknown_message_type_is_rejected(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "explode"})
        assert ws.receive_json()["code"] == "INVALID_MESSAGE"


def test_audio_before_start_is_rejected(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(b"\x00\x01" * 100)
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "SESSION_NOT_STARTED"


def test_stop_before_start_is_rejected(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "stop"})
        assert ws.receive_json()["code"] == "SESSION_NOT_STARTED"


def test_duplicate_start_is_rejected(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        ws.send_json(START)
        assert ws.receive_json()["code"] == "SESSION_ALREADY_STARTED"


def test_invalid_start_payload_is_rejected(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "session_id": "../../etc/passwd"})
        assert ws.receive_json()["code"] == "INVALID_MESSAGE"


def test_oversized_chunk_is_rejected(client, app):
    limit = app.state.settings.max_chunk_bytes
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        ws.send_bytes(b"\x00" * (limit + 2))
        assert ws.receive_json()["code"] == "CHUNK_TOO_LARGE"


def test_odd_length_pcm_is_reported(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        ws.send_bytes(b"\x00\x01\x02")     # not a whole number of PCM16 samples
        assert ws.receive_json()["code"] == "INVALID_AUDIO"


def test_empty_chunks_are_ignored(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        ws.send_bytes(b"")
        ws.send_json({"type": "stop"})
        assert drain(ws)[-1]["type"] == "final"


def test_silence_yields_insufficient_evidence(client):
    """Low-quality audio must never become a confident synthetic verdict (§15)."""
    silence = np.zeros(16000 * 10, dtype=np.float32)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        stream_audio(ws, silence)
        ws.send_json({"type": "stop"})
        final = drain(ws)[-1]
    assert final["risk_state"] == "INSUFFICIENT_EVIDENCE"
    assert final["windows_analyzed"] == 0


def test_sessions_are_independent(client, speech_16k):
    """A second session must not inherit the first one's state (§46 step 16).

    Note: the exact number of windows a session analyses is timing-dependent by
    design — when inference is slower than the hop, the pipeline keeps the newest
    window and drops the rest. So this asserts *isolation*, not score equality:
    the second session's smoother must start from scratch.
    """
    runs = []
    for session_id in ("iso-1", "iso-2"):
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({**START, "session_id": session_id})
            ws.receive_json(); ws.receive_json()
            stream_audio(ws, speech_16k)
            ws.send_json({"type": "stop"})
            runs.append(drain(ws))

    for session_id, events in zip(("iso-1", "iso-2"), runs):
        final = events[-1]
        assert final["session_id"] == session_id
        assert final["windows_analyzed"] >= 1
        assert 0.0 <= final["final_score"] <= 1.0

        # The decisive check: the first scored window of each session seeds the
        # EWMA with its own value. A leaked smoother would show a different one.
        scored = [e for e in events
                  if e["type"] == "analysis" and "synthetic_probability" in e]
        assert scored[0]["smoothed_score"] == pytest.approx(
            scored[0]["synthetic_probability"], abs=1e-6), \
            f"{session_id} inherited smoothing state from a previous session"

        # Window numbering must restart from the beginning of the new stream.
        assert scored[0]["window_start"] == pytest.approx(0.0)


def test_no_session_leak_after_completion(client, speech_16k, app):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        stream_audio(ws, speech_16k)
        ws.send_json({"type": "stop"})
        drain(ws)
    assert app.state.sessions.active_count == 0


def test_abstaining_session_still_reports_how_much_audio_was_heard(client):
    """A session where every window abstains must not look like it heard nothing.

    Showing "0 s" after a long capture reads as a broken microphone rather than
    as "captured, but not scoreable".
    """
    silence = np.zeros(16000 * 10, dtype=np.float32)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_json(); ws.receive_json()
        stream_audio(ws, silence)
        ws.send_json({"type": "stop"})
        final = drain(ws)[-1]
    assert final["windows_analyzed"] == 0
    assert final["risk_state"] == "INSUFFICIENT_EVIDENCE"
    assert final["audio_seconds"] >= 9.0, "must report the audio it actually processed"
    assert "device" in final


# ======================================================================
# Audit regressions
# ======================================================================

def _strict_json(text: str):
    """Parse exactly as a browser's JSON.parse does.

    Python's json.loads accepts NaN/Infinity/-Infinity; JSON.parse does not, and
    none of them are legal JSON. Reading server frames with receive_json() hid a
    real client-breaking bug, so these assertions parse strictly.
    """
    def reject(constant):
        raise AssertionError(f"payload contains non-JSON constant {constant!r}")
    return json.loads(text, parse_constant=reject)


def drain_text(ws):
    """Like drain(), but returns raw frames so they can be parsed strictly."""
    frames = []
    while True:
        text = ws.receive_text()
        frames.append(text)
        if _strict_json(text).get("type") == "final":
            return frames


def test_every_frame_is_valid_json_for_a_browser(client):
    """Digital silence must not put `-Infinity` on the wire (audit H2).

    An all-zero window has an RMS of 0, whose dBFS is -inf. Serialised, that
    became the literal `-Infinity`, and the dashboard's JSON.parse threw and
    dropped the event. Silent lead-ins are ordinary, so this broke real files.
    """
    silence = np.zeros(16000 * 10, dtype=np.float32)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(START)
        ws.receive_text(); ws.receive_text()
        stream_audio(ws, silence)
        ws.send_json({"type": "stop"})
        frames = drain_text(ws)

    assert any('"type":"analysis"' in f for f in frames), "expected analysis events"
    for frame in frames:
        _strict_json(frame)          # raises if any non-JSON constant is present
        assert "Infinity" not in frame and "NaN" not in frame


def test_what_gets_scored_does_not_depend_on_client_chunk_size(client, speech_16k):
    """Windows must be dropped by inference speed, never by batching (audit H3).

    A single `push_audio` that completed several windows used to keep only the
    last, so a client sending large-but-legal frames lost most of its audio:
    at the 8.19 s MAX_CHUNK_BYTES limit, 93% of windows never reached the model.
    """
    counts = {}
    for chunk_samples in (4000, 16000, 64000):       # 0.25 s, 1 s, 4 s
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({**START, "session_id": f"chunk-{chunk_samples}"})
            ws.receive_json(); ws.receive_json()
            stream_audio(ws, speech_16k, chunk=chunk_samples)
            ws.send_json({"type": "stop"})
            final = drain(ws)[-1]
        counts[chunk_samples] = final["windows_analyzed"]

    assert min(counts.values()) > 0, counts
    # Before the fix these differed by roughly the chunk-size ratio (16x).
    assert max(counts.values()) - min(counts.values()) <= 2, (
        f"windows analysed still depends on how the client batches bytes: {counts}"
    )


def test_a_reused_session_id_cannot_unregister_a_live_session(client, app):
    """One connection's teardown must not evict another's session (audit M2).

    `session_id` is client-supplied. The registry used to let the newest
    connection overwrite the entry, so when the *first* socket closed it deleted
    the *second* one's record — under-counting active sessions and letting a
    client walk straight past MAX_CONCURRENT_SESSIONS.
    """
    with client.websocket_connect("/ws/stream") as first:
        first.send_json(START)
        first.receive_json(); first.receive_json()
        assert app.state.sessions.active_count == 1

        with client.websocket_connect("/ws/stream") as second:
            second.send_json(START)              # same session_id, deliberately
            reply = second.receive_json()
            assert reply["type"] == "error"
            assert reply["code"] == "SESSION_ALREADY_STARTED"
            # The duplicate was refused, so the original is untouched.
            assert app.state.sessions.active_count == 1

    deadline = time.monotonic() + 5.0
    while app.state.sessions.active_count and time.monotonic() < deadline:
        time.sleep(0.02)
    assert app.state.sessions.active_count == 0


@pytest.fixture()
def vad_app(monkeypatch):
    """App with VAD switched on — the default `app` fixture disables it.

    That default is why the shared-VAD defect went unnoticed: with no VAD there
    is no per-session state left to corrupt.
    """
    monkeypatch.setenv("DETECTOR_PROVIDER", "mock")
    monkeypatch.setenv("MOCK_LATENCY_MS", "0")
    monkeypatch.setenv("VAD_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_AUDIO_ENABLED", "false")
    from backend.app.config import get_settings
    get_settings.cache_clear()
    from backend.app.main import create_app
    application = create_app()
    yield application
    get_settings.cache_clear()


@pytest.fixture()
def vad_client(vad_app):
    from fastapi.testclient import TestClient
    with TestClient(vad_app) as c:
        yield c


def test_each_live_session_is_given_its_own_vad(vad_client, vad_app):
    """The handler must build a VAD per session, not share one (audit H1)."""
    built = []
    original = vad_app.state.build_vad
    vad_app.state.build_vad = lambda: built.append(original()) or built[-1]

    with vad_client.websocket_connect("/ws/stream") as first:
        first.send_json({**START, "session_id": "vad-a"})
        first.receive_json(); first.receive_json()
        with vad_client.websocket_connect("/ws/stream") as second:
            second.send_json({**START, "session_id": "vad-b"})
            second.receive_json(); second.receive_json()
            assert vad_app.state.sessions.active_count == 2

    assert len(built) == 2, "expected one VAD per session"
    assert built[0] is not built[1], (
        "both sessions were handed the same VAD; Silero's LSTM state would be shared"
    )


def test_concurrent_sessions_score_exactly_as_they_would_alone(vad_client, real_speech_16k):
    """Running alongside another session must not change a session's results.

    Silero carries LSTM state across frames. With one app-wide instance, both
    concurrent sessions' frames fed the same recurrent state: every one of the
    16 windows in each session came out with different speech accounting than
    the same audio scored on its own.
    """
    results = {}

    def run(session_id, barrier):
        with vad_client.websocket_connect("/ws/stream") as ws:
            ws.send_json({**START, "session_id": session_id})
            ws.receive_json(); ws.receive_json()
            barrier.wait(timeout=30)         # line both sessions up before streaming
            stream_audio(ws, real_speech_16k)
            ws.send_json({"type": "stop"})
            events = drain(ws)
        results[session_id] = [
            e["metrics"]["speech_seconds"] for e in events if e["type"] == "analysis"
        ]

    run("solo", threading.Barrier(1))
    assert results["solo"], "expected at least one analysis window"

    barrier = threading.Barrier(2)
    threads = [threading.Thread(target=run, args=(f"conc-{i}", barrier)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    for session_id in ("conc-1", "conc-2"):
        assert results[session_id] == results["solo"], (
            f"{session_id} produced different speech accounting than the same audio "
            f"scored alone — session state is leaking between connections"
        )
