"""SystemAudioAdapter contract, exercised without any audio hardware."""

import asyncio

import pytest

from backend.app.ingestion.adapters.base import AudioSourceAdapter, CaptureError
from backend.app.ingestion.adapters.system_audio import (
    CaptureDevice, SystemAudioAdapter, list_capture_devices, opaque_id, resolve_device,
)
from backend.tests.fake_capture import FakeCaptureBackend


@pytest.fixture()
def backend():
    return FakeCaptureBackend()


async def collect(adapter, count, timeout=5.0):
    chunks = []

    async def _read():
        async for chunk in adapter.stream():
            chunks.append(chunk)
            if len(chunks) >= count:
                return

    await asyncio.wait_for(_read(), timeout=timeout)
    return chunks


# ---- device discovery ------------------------------------------------
def test_opaque_id_is_stable_and_hides_backend_name():
    assert opaque_id("alsa_output.pci.monitor") == opaque_id("alsa_output.pci.monitor")
    assert opaque_id("a") != opaque_id("b")
    assert "alsa" not in opaque_id("alsa_output.pci.monitor")


def test_public_payload_never_leaks_backend_id(backend):
    device = backend.list_devices()[0]
    assert "backend_id" not in device.public()
    assert device.backend_id not in str(device.public())


def test_loopback_filter_excludes_microphones(backend):
    loopback = list_capture_devices(backend, loopback_only=True)
    assert loopback and all(d.is_loopback for d in loopback)
    assert len(list_capture_devices(backend, loopback_only=False)) > len(loopback)


def test_resolve_defaults_to_first_loopback_device(backend):
    assert resolve_device(backend, None).is_loopback is True


def test_resolve_accepts_a_known_id(backend):
    target = list_capture_devices(backend)[0]
    assert resolve_device(backend, target.id).backend_id == target.backend_id


def test_resolve_rejects_an_unknown_id(backend):
    with pytest.raises(CaptureError):
        resolve_device(backend, "dev-deadbeef")


def test_resolve_rejects_a_raw_backend_name(backend):
    """A client must not be able to name an arbitrary device."""
    with pytest.raises(CaptureError):
        resolve_device(backend, "fake.monitor")


def test_no_loopback_devices_is_a_clear_error():
    empty = FakeCaptureBackend(devices=[])
    with pytest.raises(CaptureError, match="no loopback capture device"):
        resolve_device(empty, None)


# ---- streaming --------------------------------------------------------
@pytest.mark.asyncio
async def test_adapter_emits_wellformed_chunks(backend):
    device = resolve_device(backend, None)
    adapter = SystemAudioAdapter(backend, device, "sess-1", block_ms=20)
    await adapter.open()
    try:
        chunks = await collect(adapter, 3)
    finally:
        await adapter.close()

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.session_id == "sess-1"
        assert chunk.sample_rate == 48000
        assert chunk.channels == 2
        assert chunk.codec_hint == "pcm_s16le"
        # 20 ms at 48 kHz, stereo, 2 bytes per sample
        assert len(chunk.audio_pcm16) == 960 * 2 * 2
    assert chunks[0].timestamp_ms == 0
    assert chunks[1].timestamp_ms == 20


@pytest.mark.asyncio
async def test_adapter_is_a_registered_source_adapter(backend):
    adapter = SystemAudioAdapter(backend, resolve_device(backend, None), "s")
    assert isinstance(adapter, AudioSourceAdapter)
    assert adapter.name == "system_audio"
    assert adapter.describe()["device_label"] == "Fake Output (monitor)"


@pytest.mark.asyncio
async def test_chunks_survive_normalisation(backend):
    """The adapter's output must be consumable by the existing normalizer."""
    from backend.app.ingestion.normalizer import normalize_chunk

    adapter = SystemAudioAdapter(backend, resolve_device(backend, None), "s", block_ms=20)
    await adapter.open()
    try:
        chunk = (await collect(adapter, 1))[0]
    finally:
        await adapter.close()

    samples, stats = normalize_chunk(chunk, 16000)
    assert samples.size == 320          # 20 ms at 16 kHz mono
    assert stats.rms_dbfs > -60         # the fake tone carries real energy


@pytest.mark.asyncio
async def test_open_failure_raises_capture_error():
    failing = FakeCaptureBackend(open_error=RuntimeError("device busy"))
    adapter = SystemAudioAdapter(failing, resolve_device(failing, None), "s")
    with pytest.raises(CaptureError, match="device busy"):
        await adapter.open()
    await adapter.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_ends_the_stream(backend):
    adapter = SystemAudioAdapter(backend, resolve_device(backend, None), "s", block_ms=20)
    await adapter.open()
    await collect(adapter, 1)
    await adapter.close()
    await adapter.close()          # must not raise
    # The stream terminates rather than hanging.
    remaining = [c async for c in adapter.stream()]
    assert isinstance(remaining, list)


@pytest.mark.asyncio
async def test_stream_before_open_is_rejected(backend):
    adapter = SystemAudioAdapter(backend, resolve_device(backend, None), "s")
    with pytest.raises(CaptureError, match="open\\(\\)"):
        async for _ in adapter.stream():
            break


@pytest.mark.asyncio
async def test_backlog_is_bounded_and_drops_oldest(backend):
    """A slow consumer must not grow memory without bound."""
    adapter = SystemAudioAdapter(
        backend, resolve_device(backend, None), "s", block_ms=10, queue_blocks=3
    )
    await adapter.open()
    try:
        await asyncio.sleep(0.4)          # let capture outrun the (idle) consumer
        assert adapter._queue.qsize() <= 3
        assert adapter.dropped_blocks > 0
    finally:
        await adapter.close()


# ---- teardown (audit M1) ---------------------------------------------
@pytest.mark.asyncio
async def test_close_joins_the_capture_thread_even_after_request_stop():
    """close() must wait for the device, whatever happened before it.

    The WebSocket teardown calls request_stop() first, because awaiting is
    unsafe during cancellation. close() then guarded on "was a stop already
    requested?" and returned immediately, so on every client disconnect it
    skipped the thread join and left the device held after the session was
    reported closed. The guard belongs on whether the join has happened.
    """
    backend = FakeCaptureBackend(release_delay=0.3)
    adapter = SystemAudioAdapter(backend, resolve_device(backend, None), "s", block_ms=20)
    await adapter.open()
    await collect(adapter, 1)

    adapter.request_stop()          # exactly what the WebSocket `finally` does
    await adapter.close()

    assert backend.released.is_set(), (
        "close() returned while the capture device was still held"
    )


@pytest.mark.asyncio
async def test_close_stays_idempotent_after_joining():
    backend = FakeCaptureBackend(release_delay=0.05)
    adapter = SystemAudioAdapter(backend, resolve_device(backend, None), "s", block_ms=20)
    await adapter.open()
    await collect(adapter, 1)
    await adapter.close()
    await adapter.close()           # second call must be a cheap no-op, not a hang
    assert backend.released.is_set()
