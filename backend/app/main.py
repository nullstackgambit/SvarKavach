"""SvarKavach application entry point.

Startup order (plan §36): configuration -> logging -> device detection ->
VAD -> detector -> warm-up -> readiness.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import audio_devices, health, websocket
from .config import REPO_ROOT, get_settings
from .detector.base import DetectorError
from .detector.factory import build_detector
from .ingestion.session_manager import SessionManager
from .ingestion.adapters.system_audio import build_backend
from .preprocessing.vad import VadFactory
from .scoring.calibration import Calibrator
from .utils.logging import configure_logging, get_logger

FRONTEND_DIR = REPO_ROOT / "frontend"
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    app.state.sessions = SessionManager(max_concurrent=settings.max_concurrent_sessions)
    app.state.startup_error = None
    app.state.calibration_info = Calibrator().describe()

    # A factory, not an instance. Silero carries LSTM state across frames, so
    # every session needs its own; sharing one interleaves concurrent sessions'
    # audio into a single recurrent state and corrupts all of them. The factory
    # resolves Silero-vs-fallback once here, so /config and session_started can
    # still report which implementation is in use.
    vad_factory = VadFactory(
        enabled=settings.vad_enabled,
        model_path=settings.vad_model_path or None,
        sample_rate=settings.target_sample_rate,
    )
    app.state.build_vad = vad_factory
    app.state.vad_name = vad_factory.name
    app.state.vad_fallback = vad_factory.is_fallback

    # ---- system audio capture (opt-in) -------------------------------
    app.state.capture_backend = None
    app.state.capture_unavailable_reason = None
    if settings.system_audio_enabled:
        backend, reason = build_backend()
        app.state.capture_backend = backend
        app.state.capture_unavailable_reason = reason
        if backend is None:
            log.warning("system audio capture unavailable", extra={"reason": reason})
        else:
            log.info("system audio capture ready", extra={"backend": backend.name})
            # Capture is a privileged capability and this app has no auth. Warn
            # unconditionally: the real bind address comes from uvicorn's --host
            # flag, not from settings.host, so a check on the latter can be
            # silently wrong.
            log.warning(
                "SYSTEM AUDIO CAPTURE IS ENABLED. Any client that can reach this "
                "server can start recording this machine's audio output, and this "
                "application has no authentication. Serve it on 127.0.0.1 only, or "
                "put authentication and TLS in front of it.",
                extra={"configured_host": settings.host},
            )
            if settings.binds_publicly:
                log.warning(
                    "Configured HOST is not a loopback address. Note the actual "
                    "bind address is whatever uvicorn --host was given.",
                    extra={"configured_host": settings.host},
                )

    try:
        detector = build_detector(settings)
    except DetectorError as exc:
        # Fail fast rather than silently degrade a requested real detector.
        log.error("detector initialisation failed", extra={"error": str(exc)})
        raise

    app.state.detector = detector
    log.info(
        "detector ready",
        extra={"model": detector.name, "device": detector.device,
               "is_mock": detector.is_mock},
    )

    try:
        detector.warmup()
        log.info("model warm-up complete", extra={"model": detector.name})
    except Exception as exc:  # warm-up must never block startup
        log.warning("model warm-up failed", extra={"error": str(exc)})

    log.info(
        "%s ready on %s:%s", settings.app_name, settings.host, settings.port,
        extra={"detector": detector.name, "vad": app.state.vad_name},
    )
    try:
        yield
    finally:
        detector.close()
        log.info("shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        description="Real-time synthetic voice detection platform (MVP)",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(audio_devices.router)
    app.include_router(websocket.router)

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.middleware("http")
        async def no_store_for_static(request, call_next):
            """Serve the demo frontend uncached.

            Without this the browser keeps a stale app.js/styles.css after an
            edit, which looks exactly like the change not working. This is a
            local demo server, so correctness beats caching.
            """
            response = await call_next(request)
            if request.url.path.startswith("/static") or request.url.path == "/":
                response.headers["Cache-Control"] = "no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
            return response

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
