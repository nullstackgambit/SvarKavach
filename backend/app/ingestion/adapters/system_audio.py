"""Live capture of the machine's own audio output (loopback).

This is what makes the demo reflect the real product story: instead of replaying
a file, the dashboard analyses whatever is currently playing — a Zoom desktop
call, a softphone, a browser meeting.

Platform support comes from SoundCard (BSD-3):
  * Linux   - PulseAudio/PipeWire "monitor" sources
  * Windows - WASAPI loopback devices

SoundCard is imported lazily: the application, and the whole file-upload demo,
must keep working when it is not installed.

Capture is inherently blocking, so it runs on a worker thread and hands blocks
to the event loop through a bounded queue — the same discipline the pipeline
already applies to model inference (plan §13). When the consumer falls behind,
the oldest block is dropped rather than letting memory grow without bound.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Protocol

import numpy as np

from ...models.messages import AudioChunk
from ...utils.logging import get_logger
from ..normalizer import float_to_pcm16
from .base import AudioSourceAdapter, CaptureError

log = get_logger(__name__)

#: Capture rate. 48 kHz is the near-universal native rate for output devices;
#: the existing normalizer resamples to 16 kHz, so there is one resampling path.
CAPTURE_SAMPLE_RATE = 48_000
#: How long to wait for the device to open before giving up.
OPEN_TIMEOUT_SECONDS = 10.0
#: How long close() waits for the capture thread to notice the stop flag.
THREAD_JOIN_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class CaptureDevice:
    """A capture device. `backend_id` never leaves the server (plan §35)."""

    id: str
    label: str
    backend_id: str
    channels: int
    is_loopback: bool

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "channels": self.channels,
            "is_loopback": self.is_loopback,
        }


def opaque_id(backend_id: str) -> str:
    """Stable, non-reversible client-facing id for a backend device."""
    digest = hashlib.blake2s(backend_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"dev-{digest}"


class CaptureBackend(Protocol):
    """Minimal surface the adapter needs. Lets tests run without a device."""

    name: str

    def list_devices(self) -> list[CaptureDevice]: ...

    def open_recorder(
        self, device: CaptureDevice, sample_rate: int, channels: int, blocksize: int
    ): ...


#: PulseAudio client name. Also the value the argv shim below installs, so the
#: capture shows up as "svarkavach" in pavucontrol rather than "python".
CLIENT_NAME = "svarkavach"


def _import_soundcard():
    """Import soundcard, working around an upstream crash on minimal argv.

    SoundCard 0.4.6's PulseAudio backend infers a client name from `sys.argv`
    and reads `sys.argv[1]` whenever `sys.argv[0] == "-c"`. Python actually
    leaves `sys.argv == ["-c"]` for `python -c "..."`, so the import raises
    IndexError before any of our code runs. Pad argv across the import only.
    """
    import sys

    needs_shim = len(sys.argv) < 2 and (not sys.argv or sys.argv[0] in ("-c", "-"))
    saved = list(sys.argv)
    if needs_shim:
        sys.argv = [CLIENT_NAME, CLIENT_NAME]
    try:
        import soundcard

        return soundcard
    finally:
        if needs_shim:
            sys.argv = saved


class SoundCardBackend:
    """SoundCard-backed capture. Constructed only when the import succeeds."""

    name = "soundcard"

    def __init__(self) -> None:
        self._sc = _import_soundcard()

    def list_devices(self) -> list[CaptureDevice]:
        devices: list[CaptureDevice] = []
        for mic in self._sc.all_microphones(include_loopback=True):
            backend_id = str(getattr(mic, "id", mic.name))
            is_loopback = bool(getattr(mic, "isloopback", False))
            devices.append(
                CaptureDevice(
                    id=opaque_id(backend_id),
                    label=mic.name,
                    backend_id=backend_id,
                    channels=int(getattr(mic, "channels", 2) or 2),
                    is_loopback=is_loopback,
                )
            )
        # Loopback devices first: they are what this feature is for.
        devices.sort(key=lambda d: (not d.is_loopback, d.label))
        return devices

    @contextmanager
    def open_recorder(
        self, device: CaptureDevice, sample_rate: int, channels: int, blocksize: int
    ) -> Iterator:
        mic = self._sc.get_microphone(device.backend_id, include_loopback=True)
        with mic.recorder(
            samplerate=sample_rate, channels=channels, blocksize=blocksize
        ) as recorder:
            yield recorder


def build_backend() -> tuple[CaptureBackend | None, str | None]:
    """Return (backend, reason_unavailable). Never raises."""
    try:
        return SoundCardBackend(), None
    except ImportError:
        return None, (
            "the 'SoundCard' package is not installed. "
            "Install it with: pip install -r requirements-audio.txt"
        )
    except Exception as exc:  # e.g. no PulseAudio/PipeWire server running
        return None, f"audio backend unavailable: {exc}"


def list_capture_devices(backend: CaptureBackend, loopback_only: bool = True) -> list[CaptureDevice]:
    devices = backend.list_devices()
    return [d for d in devices if d.is_loopback] if loopback_only else devices


def resolve_device(
    backend: CaptureBackend, device_id: str | None, loopback_only: bool = True
) -> CaptureDevice:
    """Map a client-supplied opaque id to a real device.

    Only ids produced by enumeration are accepted, so a client cannot name an
    arbitrary device.
    """
    devices = list_capture_devices(backend, loopback_only=loopback_only)
    if not devices:
        raise CaptureError(
            "no loopback capture device was found. On Linux this needs a "
            "PulseAudio or PipeWire server with a monitor source; on Windows it "
            "needs a WASAPI loopback device."
        )
    if not device_id:
        return devices[0]
    for device in devices:
        if device.id == device_id:
            return device
    raise CaptureError(f"unknown capture device id {device_id!r}")


class SystemAudioAdapter(AudioSourceAdapter):
    """Streams the host's audio output as `AudioChunk`s."""

    name = "system_audio"

    def __init__(
        self,
        backend: CaptureBackend,
        device: CaptureDevice,
        session_id: str,
        *,
        sample_rate: int = CAPTURE_SAMPLE_RATE,
        block_ms: int = 100,
        queue_blocks: int = 50,
    ) -> None:
        self._backend = backend
        self._device = device
        self._session_id = session_id
        self._sample_rate = sample_rate
        self._channels = max(1, min(int(device.channels or 2), 2))
        self._blocksize = max(1, int(sample_rate * block_ms / 1000))
        self._queue_blocks = max(1, queue_blocks)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._opened = asyncio.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._open_error: Exception | None = None
        self._dropped_blocks = 0
        self._closed = False
        self._joined = False

    # -- properties ----------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def device(self) -> CaptureDevice:
        return self._device

    @property
    def dropped_blocks(self) -> int:
        return self._dropped_blocks

    def describe(self) -> dict:
        return {
            "source": self.name,
            "device_label": self._device.label,
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "backend": getattr(self._backend, "name", "unknown"),
        }

    # -- lifecycle -----------------------------------------------------
    async def open(self) -> None:
        """Open the device. Raises CaptureError if it cannot be started."""
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._queue_blocks)
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"capture-{self._session_id}", daemon=True
        )
        self._thread.start()

        try:
            await asyncio.wait_for(self._opened.wait(), timeout=OPEN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            self._stop.set()
            raise CaptureError(
                f"capture device {self._device.label!r} did not open within "
                f"{OPEN_TIMEOUT_SECONDS:.0f}s"
            ) from exc

        if self._open_error is not None:
            raise CaptureError(
                f"could not open capture device {self._device.label!r}: {self._open_error}"
            )

        log.info("system audio capture started", extra={
            "session_id": self._session_id,
            "device": self._device.label,
            "sample_rate": self._sample_rate,
            "channels": self._channels,
        })

    def request_stop(self) -> None:
        """Synchronous, cancellation-safe stop.

        Sets the stop flag and unblocks any consumer. Safe to call from a
        `finally` block during task cancellation, where awaiting is not.
        """
        self._closed = True
        self._stop.set()
        self._put_sentinel()

    async def close(self) -> None:
        self.request_stop()
        # Guard on whether the join has already happened, NOT on whether a stop
        # was already requested. The WebSocket teardown path calls
        # request_stop() first (it is cancellation-safe, unlike awaiting), so a
        # `_closed` guard here made close() short-circuit on every client
        # disconnect and never wait for the capture thread — leaving the device
        # held after the session was reported closed.
        if self._joined:
            return
        self._joined = True
        # Wait for the capture thread cooperatively rather than through
        # asyncio.to_thread(). close() runs during connection teardown, and
        # handing a blocking join to the default executor deadlocks there when
        # the surrounding event loop is already winding down. Polling yields to
        # the loop, needs no executor, and is still bounded.
        if self._thread is not None:
            deadline = time.monotonic() + THREAD_JOIN_TIMEOUT_SECONDS
            while self._thread.is_alive() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            if self._thread.is_alive():
                log.warning("capture thread did not exit in time",
                            extra={"session_id": self._session_id})
            self._thread = None
        log.info("system audio capture stopped", extra={
            "session_id": self._session_id,
            "dropped_blocks": self._dropped_blocks,
        })

    # -- capture thread -------------------------------------------------
    def _signal_opened(self, error: Exception | None) -> None:
        self._open_error = error
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._opened.set)

    def _capture_loop(self) -> None:
        try:
            with self._backend.open_recorder(
                self._device, self._sample_rate, self._channels, self._blocksize
            ) as recorder:
                self._signal_opened(None)
                while not self._stop.is_set():
                    data = recorder.record(numframes=self._blocksize)
                    if data is None:
                        break
                    self._publish(data)
        except Exception as exc:
            if not self._opened.is_set():
                self._signal_opened(exc)
            else:
                log.error("capture failed mid-stream", extra={
                    "session_id": self._session_id, "error": str(exc)})
        finally:
            if self._loop is not None and self._queue is not None:
                self._loop.call_soon_threadsafe(self._put_sentinel)

    def _publish(self, data: np.ndarray) -> None:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 2:
            array = array.reshape(-1)      # -> interleaved, as PCM16 expects
        payload = float_to_pcm16(array)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._enqueue, payload)

    def _enqueue(self, payload: bytes) -> None:
        """Runs on the event loop thread. Drops oldest when saturated."""
        queue = self._queue
        if queue is None:
            return
        if queue.full():
            try:
                queue.get_nowait()
                self._dropped_blocks += 1
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(payload)

    def _put_sentinel(self) -> None:
        """Signal end-of-stream, evicting a block if the queue is saturated.

        A plain put_nowait() would raise QueueFull here and the consumer would
        never learn that capture had ended.
        """
        queue = self._queue
        if queue is None:
            return
        while queue.full():
            try:
                queue.get_nowait()
                self._dropped_blocks += 1
            except asyncio.QueueEmpty:
                break
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:  # pragma: no cover - defensive
            pass

    # -- consumption ----------------------------------------------------
    async def stream(self) -> AsyncIterator[AudioChunk]:
        if self._queue is None:
            raise CaptureError("open() must be called before stream()")
        seq = 0
        elapsed_ms = 0
        block_ms = int(1000 * self._blocksize / self._sample_rate)
        while True:
            payload = await self._queue.get()
            if payload is None:          # sentinel: capture ended
                break
            seq += 1
            yield AudioChunk(
                session_id=self._session_id,
                stream_id=f"{self._session_id}:sys:{seq}",
                timestamp_ms=elapsed_ms,
                audio_pcm16=payload,
                sample_rate=self._sample_rate,
                channels=self._channels,
                codec_hint="pcm_s16le",
            )
            elapsed_ms += block_ms
