"""Health, readiness, and safe configuration endpoints (plan §18)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state
    detector = getattr(state, "detector", None)
    return {
        "status": "ok",
        "service": state.settings.app_name.lower(),
        "model_loaded": detector is not None,
        "model": detector.name if detector else None,
        "is_mock": detector.is_mock if detector else None,
        "device": detector.device if detector else None,
        "vad": getattr(state, "vad_name", None),
        "vad_fallback": getattr(state, "vad_fallback", None),
        "active_sessions": state.sessions.active_count,
    }


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict:
    """Not-ready when a real detector was requested but is unavailable (§18)."""
    state = request.app.state
    detector = getattr(state, "detector", None)
    settings = state.settings

    problems: list[str] = []
    if detector is None:
        problems.append("detector not loaded")
    elif settings.uses_real_detector and detector.is_mock:
        problems.append(
            f"requested detector '{settings.detector_provider}' is not active; "
            f"mock inference is running instead"
        )
    if getattr(state, "startup_error", None):
        problems.append(str(state.startup_error))

    if problems:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False, "problems": problems}
    return {"ready": True, "problems": []}


@router.get("/config")
async def config(request: Request) -> dict:
    """Non-secret configuration only. Never exposes environment variables."""
    state = request.app.state
    detector = getattr(state, "detector", None)
    payload = state.settings.public_config()
    payload["detector"] = detector.describe() if detector else None
    payload["calibration"] = state.calibration_info
    payload["vad"] = {
        "name": getattr(state, "vad_name", None),
        "is_fallback": getattr(state, "vad_fallback", None),
    }
    return payload
