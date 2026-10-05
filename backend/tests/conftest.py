import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Where the suite looks for genuine recorded speech, in priority order.
#
# Some checks need REAL speech and cannot be faked: Silero VAD only fires on
# actual voice, and the detector's spoof/bona-fide index convention can only be
# confirmed against a known-human recording. Synthetic tones do not work — a
# frequency-modulated chirp scores 0 speech frames out of 437.
#
# 1. sample_audio/  — the project's own documented location. Drop real
#    recordings there (see sample_audio/README.md) and these tests get stronger
#    automatically.
# 2. The speech clips shipped with ALSA on most Linux desktops. A convenience
#    for Linux contributors only; NOT a project dependency, and absent on
#    Windows and macOS.
# 3. Otherwise the real-speech tests skip. Nothing in the suite fails for want
#    of an audio file.
_ALSA_CLIPS = ("Front_Center", "Front_Left", "Rear_Center", "Side_Left")
_AUDIO_SUFFIXES = (".wav", ".flac", ".ogg")


def _read_mono_16k(path: Path) -> np.ndarray | None:
    from backend.app.ingestion.normalizer import resample
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    return resample(data.mean(axis=1), rate, 16000)


def _load_project_speech() -> np.ndarray | None:
    """Real speech the user supplied under sample_audio/."""
    root = REPO_ROOT / "sample_audio"
    if not root.is_dir():
        return None
    parts = []
    for path in sorted(root.rglob("*")):
        # Only bona fide audio is usable as a "known human speech" fixture.
        if "synthetic" in path.parts or "spoof" in path.parts or "fake" in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES:
            chunk = _read_mono_16k(path)
            if chunk is not None and chunk.size:
                parts.append(chunk)
    if not parts:
        return None
    audio = np.concatenate(parts).astype("float32")
    # Ensure enough length for several analysis windows.
    while audio.size < 14 * 16000:
        audio = np.concatenate([audio, audio])
    return audio.astype("float32")


def _load_alsa_speech() -> np.ndarray | None:
    """Linux desktop convenience only. Never required."""
    import os

    if os.environ.get("SVARKAVACH_NO_SYSTEM_AUDIO"):
        return None
    parts = []
    for name in _ALSA_CLIPS:
        path = Path(f"/usr/share/sounds/alsa/{name}.wav")
        if path.is_file():
            chunk = _read_mono_16k(path)
            if chunk is not None and chunk.size:
                parts.append(chunk)
    if not parts:
        return None
    return np.concatenate(parts * 2).astype("float32")


def _find_real_speech() -> np.ndarray | None:
    # Explicit None checks: numpy arrays have no unambiguous truth value, so
    # `a or b` raises ValueError once sample_audio/ actually contains audio.
    project = _load_project_speech()
    if project is not None:
        return project
    return _load_alsa_speech()


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch):
    """Keep the suite hermetic.

    Settings normally load the developer's `.env`, so a local override such as
    SYSTEM_AUDIO_ENABLED=true would silently change what the tests assert.
    Tests must depend only on what they set explicitly.
    """
    from backend.app.config import Settings
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    from backend.app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def settings():
    from backend.app.config import Settings
    return Settings(_env_file=None, detector_provider="mock",
                    mock_latency_ms=0.0, vad_enabled=True)


@pytest.fixture()
def app(monkeypatch):
    """App pinned to the mock detector with VAD disabled.

    VAD is off here on purpose. Silero only fires on genuine speech, and no
    speech recording is guaranteed to exist on a contributor's machine — with
    VAD on, these tests would pass on Linux and fail on Windows purely because
    of a missing fixture. The VAD itself is covered by `test_vad.py`, which
    skips when real speech is unavailable.
    """
    monkeypatch.setenv("DETECTOR_PROVIDER", "mock")
    monkeypatch.setenv("MOCK_LATENCY_MS", "0")
    monkeypatch.setenv("VAD_ENABLED", "false")
    monkeypatch.setenv("SYSTEM_AUDIO_ENABLED", "false")
    from backend.app.config import get_settings
    get_settings.cache_clear()
    from backend.app.main import create_app
    application = create_app()
    yield application
    get_settings.cache_clear()


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def speech_16k() -> np.ndarray:
    """~14 s of 16 kHz audio with speech-like energy.

    Prefers real recorded speech; otherwise returns band-limited noise, which
    carries enough energy to clear the quality gates on any platform.
    """
    real = _find_real_speech()
    if real is not None:
        return real

    rng = np.random.default_rng(1234)
    n = 14 * 16000
    noise = rng.standard_normal(n).astype(np.float32)
    # Cheap band-pass toward the speech band so the signal is not pure hiss.
    kernel = np.hanning(65).astype(np.float32)
    kernel /= kernel.sum()
    shaped = np.convolve(noise, kernel, mode="same").astype(np.float32)
    envelope = (0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * np.arange(n) / 16000)).astype(np.float32)
    out = (shaped * envelope)
    return (0.25 * out / (np.abs(out).max() + 1e-9)).astype(np.float32)


@pytest.fixture(scope="session")
def real_speech_16k() -> np.ndarray:
    """Genuine recorded speech, or skip. Only for tests that need real speech."""
    real = _find_real_speech()
    if real is None:
        pytest.skip(
            "no real speech available. Add a .wav under sample_audio/bona_fide/ "
            "to enable the tests that need genuine recorded voice."
        )
    return real
