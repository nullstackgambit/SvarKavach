"""WebSocket protocol models and the source-agnostic internal audio model.

The `AudioChunk` here is the contract every future audio-source adapter
(Zoom, SIPREC, WebRTC, contact-centre) must produce. Nothing downstream of
this point knows or cares where the audio came from (plan §3, §41).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# Internal representation
# --------------------------------------------------------------------------
@dataclass(slots=True)
class AudioChunk:
    """A normalised slice of audio entering the detection pipeline."""

    session_id: str
    stream_id: str
    timestamp_ms: int
    audio_pcm16: bytes
    sample_rate: int
    channels: int
    codec_hint: str | None = None


class SessionState(str, Enum):
    """Explicit session lifecycle (plan §23)."""

    CREATED = "CREATED"
    CONNECTING = "CONNECTING"
    READY = "READY"
    STREAMING = "STREAMING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


# --------------------------------------------------------------------------
# Client -> server
# --------------------------------------------------------------------------
class AudioFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate: Annotated[int, Field(ge=8000, le=192000)] = 16000
    channels: Annotated[int, Field(ge=1, le=2)] = 1


class SessionContext(BaseModel):
    """Demo transaction context. NOT identity verification (plan §20)."""

    model_config = ConfigDict(extra="ignore")

    channel: Annotated[str, Field(max_length=64)] = "demo_web"
    caller_type: Annotated[str, Field(max_length=64)] = "unknown"
    contact_known: bool = True
    transaction_value: Annotated[float, Field(ge=0, le=1e15)] = 0.0


class StartMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["start"]
    session_id: Annotated[str, Field(min_length=1, max_length=64,
                                     pattern=r"^[A-Za-z0-9._:-]+$")]
    #: "demo_web"    - the client pushes binary PCM16 frames (file upload).
    #: "system_audio" - the server captures the host's audio output itself.
    source: Literal["demo_web", "system_audio"] = "demo_web"
    audio_format: AudioFormat = Field(default_factory=AudioFormat)
    context: SessionContext = Field(default_factory=SessionContext)
    #: Opaque capture device id from GET /audio/devices. system_audio only.
    device_id: Annotated[str | None, Field(max_length=64,
                                           pattern=r"^[A-Za-z0-9._:-]+$")] = None


class StopMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["stop"]


# --------------------------------------------------------------------------
# Server -> client
# --------------------------------------------------------------------------
class ErrorCode(str, Enum):
    INVALID_JSON = "INVALID_JSON"
    INVALID_MESSAGE = "INVALID_MESSAGE"
    SESSION_NOT_STARTED = "SESSION_NOT_STARTED"
    SESSION_ALREADY_STARTED = "SESSION_ALREADY_STARTED"
    INVALID_AUDIO = "INVALID_AUDIO"
    CHUNK_TOO_LARGE = "CHUNK_TOO_LARGE"
    SESSION_TOO_LONG = "SESSION_TOO_LONG"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    DETECTOR_ERROR = "DETECTOR_ERROR"
    CAPTURE_UNAVAILABLE = "CAPTURE_UNAVAILABLE"
    CAPTURE_FAILED = "CAPTURE_FAILED"
    SERVER_BUSY = "SERVER_BUSY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def error_event(code: ErrorCode, message: str) -> dict:
    return {"type": "error", "code": code.value, "message": message}


def status_event(status: str, **extra) -> dict:
    return {"type": "status", "status": status, **extra}
