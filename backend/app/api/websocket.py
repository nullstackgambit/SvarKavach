"""The /ws/stream endpoint — the single ingestion surface for the MVP.

Protocol (plan §7):
  client -> {"type":"start", ...}      server -> {"type":"session_started", ...}
  client -> binary PCM16 frames        server -> {"type":"analysis", ...}
                                       server -> {"type":"recommendation", ...}
  client -> {"type":"stop"}            server -> {"type":"final", ...}

A malformed or failing session is closed cleanly; it must never take down the
application (plan §24).
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..ingestion.adapters.base import CaptureError
from ..ingestion.adapters.system_audio import SystemAudioAdapter, resolve_device
from ..ingestion.session_manager import SessionExistsError, SessionLimitError
from ..models.messages import (
    AudioChunk,
    ErrorCode,
    SessionState,
    StartMessage,
    error_event,
    status_event,
)
from ..ingestion.normalizer import InvalidAudioError, validate_format
from ..pipeline.stream_processor import StreamProcessor
from ..utils.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


async def _pump_capture(adapter, processor, session, settings, emit) -> None:
    """Feed captured audio into the pipeline until capture stops.

    Live capture has no natural end, so MAX_SESSION_SECONDS is enforced here —
    the upload path enforces it per incoming frame, and no frames arrive in this
    mode.
    """
    try:
        async for chunk in adapter.stream():
            if session.elapsed_seconds > settings.max_session_seconds:
                # Actually stop it. Breaking out of the pump only stops us
                # *reading* the audio — the capture thread keeps recording the
                # machine's output until the client disconnects, which makes the
                # message below a lie and keeps the UI's capture banner honest
                # about nothing. request_stop() is the cancellation-safe path.
                adapter.request_stop()
                await emit(error_event(
                    ErrorCode.SESSION_TOO_LONG,
                    f"Capture stopped after the {settings.max_session_seconds}s limit."))
                await emit({"type": "capture_stopped",
                            "reason": "session_limit_reached"})
                break
            try:
                await processor.push_audio(chunk)
            except InvalidAudioError as exc:
                await emit(error_event(ErrorCode.INVALID_AUDIO, str(exc)))
                break
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("capture pump failed",
                      extra={"session_id": session.session_id})
        try:
            await emit(error_event(ErrorCode.CAPTURE_FAILED,
                                   f"System audio capture stopped: {exc}"))
        except Exception:
            pass


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    state = websocket.app.state
    settings = state.settings

    session = None
    processor: StreamProcessor | None = None
    audio_format = None
    adapter: SystemAudioAdapter | None = None
    pump: asyncio.Task | None = None
    stream_seq = 0

    async def stop_capture() -> None:
        nonlocal adapter, pump
        if pump is not None:
            pump.cancel()
            try:
                await pump
            except (asyncio.CancelledError, Exception):
                pass
            pump = None
        if adapter is not None:
            await adapter.close()
            adapter = None

    async def emit(payload: dict) -> None:
        await websocket.send_json(payload)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # ---- control messages ------------------------------------
            if (text := message.get("text")) is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await emit(error_event(ErrorCode.INVALID_JSON,
                                           "Control message was not valid JSON."))
                    continue

                if not isinstance(payload, dict):
                    await emit(error_event(ErrorCode.INVALID_MESSAGE,
                                           "Control message must be a JSON object."))
                    continue

                kind = payload.get("type")

                if kind == "start":
                    if session is not None:
                        await emit(error_event(ErrorCode.SESSION_ALREADY_STARTED,
                                               "This connection already has a session."))
                        continue
                    try:
                        start = StartMessage.model_validate(payload)
                        validate_format(start.audio_format.sample_rate,
                                        start.audio_format.channels)
                    except ValidationError as exc:
                        await emit(error_event(ErrorCode.INVALID_MESSAGE,
                                               f"Invalid start message: {exc.error_count()} "
                                               f"field(s) failed validation."))
                        continue
                    except InvalidAudioError as exc:
                        await emit(error_event(ErrorCode.UNSUPPORTED_FORMAT, str(exc)))
                        continue

                    # Resolve the capture device first: a failure here must
                    # not leave a half-built session behind.
                    capture_device = None
                    if start.source == "system_audio":
                        if not settings.system_audio_enabled:
                            await emit(error_event(
                                ErrorCode.CAPTURE_UNAVAILABLE,
                                "System audio capture is disabled. Set "
                                "SYSTEM_AUDIO_ENABLED=true in .env to enable it."))
                            continue
                        backend = getattr(state, "capture_backend", None)
                        if backend is None:
                            await emit(error_event(
                                ErrorCode.CAPTURE_UNAVAILABLE,
                                getattr(state, "capture_unavailable_reason",
                                        "audio capture backend unavailable")))
                            continue
                        try:
                            capture_device = resolve_device(backend, start.device_id)
                        except CaptureError as exc:
                            await emit(error_event(ErrorCode.CAPTURE_UNAVAILABLE, str(exc)))
                            continue

                    try:
                        session = state.sessions.create(
                            session_id=start.session_id,
                            source=start.source,
                            context=start.context,
                            detector_name=state.detector.name,
                        )
                    except SessionExistsError as exc:
                        await emit(error_event(ErrorCode.SESSION_ALREADY_STARTED,
                                               str(exc)))
                        continue
                    except SessionLimitError as exc:
                        await emit(error_event(ErrorCode.SERVER_BUSY, str(exc)))
                        await websocket.close(code=1013)
                        return

                    session.state = SessionState.STREAMING
                    audio_format = start.audio_format
                    processor = StreamProcessor(
                        settings=settings,
                        detector=state.detector,
                        # Per session: a shared VAD would interleave this
                        # session's frames with every other live session's.
                        vad=state.build_vad(),
                        session=session,
                        emit=emit,
                    )
                    await processor.start()

                    await emit({
                        "type": "session_started",
                        "session_id": session.session_id,
                        "status": "ready",
                        "detector": state.detector.name,
                        "is_mock": state.detector.is_mock,
                        "device": state.detector.device,
                        "vad": state.vad_name,
                        "vad_fallback": state.vad_fallback,
                        "window_seconds": settings.window_seconds,
                        "hop_seconds": settings.hop_seconds,
                        "thresholds": {
                            "watch": settings.watch_threshold,
                            "verify": settings.verify_threshold,
                            "escalate": settings.escalate_threshold,
                        },
                    })
                    if start.source == "system_audio" and capture_device is not None:
                        adapter = SystemAudioAdapter(
                            backend=state.capture_backend,
                            device=capture_device,
                            session_id=session.session_id,
                            block_ms=settings.system_audio_block_ms,
                            queue_blocks=settings.system_audio_queue_blocks,
                        )
                        try:
                            await adapter.open()
                        except CaptureError as exc:
                            adapter = None
                            await emit(error_event(ErrorCode.CAPTURE_FAILED, str(exc)))
                            await processor.close()
                            state.sessions.remove(session)
                            session, processor, audio_format = None, None, None
                            continue
                        await emit({"type": "capture_started", **adapter.describe()})
                        pump = asyncio.create_task(
                            _pump_capture(adapter, processor, session, settings, emit))

                    await emit(status_event("processing"))
                    continue

                if kind == "stop":
                    if session is None or processor is None:
                        await emit(error_event(ErrorCode.SESSION_NOT_STARTED,
                                               "No active session to stop."))
                        continue
                    await stop_capture()
                    final = await processor.finish()
                    await emit(final)
                    await processor.close()
                    state.sessions.remove(session)
                    session, processor, audio_format = None, None, None
                    continue

                await emit(error_event(ErrorCode.INVALID_MESSAGE,
                                       f"Unknown message type: {kind!r}"))
                continue

            # ---- binary audio ----------------------------------------
            if (payload := message.get("bytes")) is not None:
                if session is None or processor is None:
                    await emit(error_event(ErrorCode.SESSION_NOT_STARTED,
                                           "Send a 'start' message before audio."))
                    continue
                if adapter is not None:
                    await emit(error_event(
                        ErrorCode.INVALID_MESSAGE,
                        "This session captures system audio server-side; "
                        "do not send audio frames."))
                    continue
                if len(payload) > settings.max_chunk_bytes:
                    await emit(error_event(
                        ErrorCode.CHUNK_TOO_LARGE,
                        f"Chunk of {len(payload)} bytes exceeds the "
                        f"{settings.max_chunk_bytes}-byte limit."))
                    continue
                if session.elapsed_seconds > settings.max_session_seconds:
                    await emit(error_event(
                        ErrorCode.SESSION_TOO_LONG,
                        f"Session exceeded {settings.max_session_seconds}s."))
                    await websocket.close(code=1009)
                    return
                if not payload:
                    continue

                stream_seq += 1
                chunk = AudioChunk(
                    session_id=session.session_id,
                    stream_id=f"{session.session_id}:{stream_seq}",
                    timestamp_ms=int(session.elapsed_seconds * 1000),
                    audio_pcm16=payload,
                    sample_rate=audio_format.sample_rate,
                    channels=audio_format.channels,
                    codec_hint=audio_format.encoding,
                )
                try:
                    await processor.push_audio(chunk)
                except InvalidAudioError as exc:
                    await emit(error_event(ErrorCode.INVALID_AUDIO, str(exc)))
                continue

    except WebSocketDisconnect:
        log.info("client disconnected",
                 extra={"session_id": session.session_id if session else None})
    except Exception:
        log.exception("websocket session failed",
                      extra={"session_id": session.session_id if session else None})
        if session is not None:
            session.state = SessionState.ERROR
        try:
            await emit(error_event(ErrorCode.INTERNAL_ERROR,
                                   "The analysis session ended unexpectedly."))
        except Exception:
            pass
    finally:
        # Synchronous cleanup first. A client disconnect cancels this task, so
        # any `await` below may raise CancelledError immediately — anything
        # that MUST happen has to happen before the first await, or the session
        # leaks and the capture device is never released.
        if adapter is not None:
            adapter.request_stop()
        if session is not None:
            state.sessions.remove(session)

        # Best-effort async teardown. Each step is guarded so a cancellation
        # part-way through cannot skip the remaining ones.
        try:
            await stop_capture()
        except BaseException:
            pass
        try:
            if processor is not None:
                await processor.close()
        except BaseException:
            pass
        try:
            await websocket.close()
        except BaseException:
            pass
