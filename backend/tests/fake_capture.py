"""A capture backend that needs no audio hardware.

Lets the adapter and WebSocket paths be tested on any machine, including CI.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as np

from backend.app.ingestion.adapters.system_audio import CaptureDevice, opaque_id


#: A real capture device blocks for the duration of the block it returns. The
#: fake must pace itself too, or it spins at ~16,000 blocks/sec and floods the
#: event loop with cross-thread callbacks — which starves session teardown.
#: 0.5 ms keeps tests fast (~40x real time) while staying rate-limited.
DEFAULT_BLOCK_DELAY = 0.0005


class FakeRecorder:
    def __init__(self, channels: int, amplitude: float = 0.2,
                 delay: float = DEFAULT_BLOCK_DELAY):
        self._channels = channels
        self._amplitude = amplitude
        self._delay = delay
        self._phase = 0
        self.blocks_served = 0

    def record(self, numframes: int) -> np.ndarray:
        if self._delay:
            threading.Event().wait(self._delay)
        t = (np.arange(numframes) + self._phase) / 48000.0
        self._phase += numframes
        mono = (self._amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        self.blocks_served += 1
        return np.repeat(mono[:, None], self._channels, axis=1)


class FakeCaptureBackend:
    name = "fake"

    def __init__(self, devices=None, open_error: Exception | None = None,
                 channels: int = 2, delay: float = DEFAULT_BLOCK_DELAY,
                 release_delay: float = 0.0):
        self._open_error = open_error
        self._channels = channels
        self._delay = delay
        #: How long releasing the device takes. A real device does not come free
        #: instantly, and a fake that releases in zero time cannot show whether
        #: teardown actually waited for the capture thread.
        self._release_delay = release_delay
        #: Set once the recorder context has fully exited.
        self.released = threading.Event()
        self.recorder = None
        self._devices = devices if devices is not None else [
            CaptureDevice(
                id=opaque_id("fake.monitor"), label="Fake Output (monitor)",
                backend_id="fake.monitor", channels=channels, is_loopback=True,
            ),
            CaptureDevice(
                id=opaque_id("fake.mic"), label="Fake Microphone",
                backend_id="fake.mic", channels=1, is_loopback=False,
            ),
        ]

    def list_devices(self):
        return list(self._devices)

    @contextmanager
    def open_recorder(self, device, sample_rate, channels, blocksize):
        if self._open_error is not None:
            raise self._open_error
        self.recorder = FakeRecorder(channels, delay=self._delay)
        try:
            yield self.recorder
        finally:
            if self._release_delay:
                threading.Event().wait(self._release_delay)
            self.released.set()
