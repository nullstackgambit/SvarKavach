"""Audio normalisation: anything in, mono float32 @ target rate out.

Deliberately does NOT denoise, apply AGC, or otherwise "clean" the signal.
Those algorithms remove exactly the low-energy spectral irregularities that
anti-spoofing detectors key on (plan §9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from ..models.messages import AudioChunk

# PCM16 full scale. Used for both decoding and clipping detection.
_PCM16_SCALE = 32768.0
_MAX_REASONABLE_RATE = 192_000
#: dBFS reported for digital silence. A true -inf is not representable in JSON
#: (json.dumps emits the literal `-Infinity`, which JSON.parse rejects), so
#: silence is floored here instead. Far below any usable quality threshold, so
#: every comparison against MIN_RMS_DBFS keeps its meaning.
SILENCE_DBFS = -120.0


class InvalidAudioError(ValueError):
    """Raised when a chunk cannot be decoded or normalised."""


@dataclass(slots=True)
class SignalStats:
    """Cheap signal statistics used by the quality gate."""

    rms: float
    peak: float
    rms_dbfs: float
    clipping_ratio: float
    num_samples: int

    def as_dict(self) -> dict:
        return {
            "rms_dbfs": round(self.rms_dbfs, 2),
            "peak": round(self.peak, 4),
            "clipping_ratio": round(self.clipping_ratio, 5),
            "num_samples": self.num_samples,
        }


def validate_format(sample_rate: int, channels: int) -> None:
    """Reject impossible declarations before any allocation happens (plan §35)."""
    if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= _MAX_REASONABLE_RATE:
        raise InvalidAudioError(
            f"unsupported sample_rate {sample_rate!r}; expected 8000..{_MAX_REASONABLE_RATE}"
        )
    if channels not in (1, 2):
        raise InvalidAudioError(f"unsupported channel count {channels!r}; expected 1 or 2")


def pcm16_to_float(raw: bytes) -> np.ndarray:
    """Decode little-endian signed 16-bit PCM into float32 in [-1, 1)."""
    if len(raw) % 2 != 0:
        raise InvalidAudioError(
            f"PCM16 payload length {len(raw)} is not a whole number of samples"
        )
    if not raw:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return samples / _PCM16_SCALE


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Encode float samples back to PCM16, clamping rather than wrapping."""
    clamped = np.clip(samples, -1.0, 1.0 - 1.0 / _PCM16_SCALE)
    return (clamped * _PCM16_SCALE).astype("<i2").tobytes()


def downmix_to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels down to mono."""
    if channels == 1:
        return samples
    if samples.size % channels != 0:
        # Drop a trailing partial frame rather than misalign every channel.
        samples = samples[: samples.size - (samples.size % channels)]
    if samples.size == 0:
        return samples
    return samples.reshape(-1, channels).mean(axis=1)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resample. BSD-licensed scipy; avoids an LGPL soxr dependency."""
    if source_rate == target_rate or samples.size == 0:
        return samples
    gcd = np.gcd(source_rate, target_rate)
    return resample_poly(samples, target_rate // gcd, source_rate // gcd).astype(np.float32)


def sanitize(samples: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with silence rather than letting them poison the model."""
    if samples.size and not np.all(np.isfinite(samples)):
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return samples


def compute_stats(samples: np.ndarray) -> SignalStats:
    if samples.size == 0:
        return SignalStats(0.0, 0.0, SILENCE_DBFS, 0.0, 0)
    abs_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak = float(abs_samples.max())
    # A sample within 1 LSB of full scale is treated as clipped.
    clipped = int(np.count_nonzero(abs_samples >= 1.0 - (1.0 / _PCM16_SCALE)))
    return SignalStats(
        rms=rms,
        peak=peak,
        rms_dbfs=max(20.0 * np.log10(rms), SILENCE_DBFS) if rms > 0 else SILENCE_DBFS,
        clipping_ratio=clipped / samples.size,
        num_samples=int(samples.size),
    )


def normalize_chunk(chunk: AudioChunk, target_rate: int) -> tuple[np.ndarray, SignalStats]:
    """Full normalisation path: decode -> mono -> resample -> sanitise."""
    validate_format(chunk.sample_rate, chunk.channels)
    samples = pcm16_to_float(chunk.audio_pcm16)
    samples = downmix_to_mono(samples, chunk.channels)
    samples = sanitize(samples)
    samples = resample(samples, chunk.sample_rate, target_rate)
    samples = sanitize(samples).astype(np.float32, copy=False)
    return samples, compute_stats(samples)
