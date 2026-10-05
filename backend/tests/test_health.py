def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "svarkavach"
    assert body["model_loaded"] is True
    assert body["model"] == "mock"
    assert body["is_mock"] is True
    assert body["device"] == "cpu"


def test_ready_is_true_when_mock_is_requested(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "problems": []}


def test_config_exposes_thresholds_and_windowing(client):
    cfg = client.get("/config").json()
    assert cfg["window_samples"] == 64600
    assert cfg["hop_samples"] == 8000
    assert cfg["thresholds"] == {"watch": 0.40, "verify": 0.65, "escalate": 0.85}
    assert cfg["detector"]["model"] == "mock"


def test_config_leaks_no_secrets_or_paths(client):
    raw = client.get("/config").text.lower()
    for forbidden in ("secret", "password", "token", "api_key", "/home/", "model_path"):
        assert forbidden not in raw


def test_ready_is_false_when_real_detector_is_requested_but_mocked(client, app):
    """A mock standing in for a requested real detector must report not-ready."""
    app.state.settings.detector_provider = "xlsr_aasist_onnx"
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False
    finally:
        app.state.settings.detector_provider = "mock"
