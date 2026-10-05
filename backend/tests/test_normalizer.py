import numpy as np
import pytest

from backend.app.ingestion.normalizer import (
    InvalidAudioError, compute_stats, downmix_to_mono, float_to_pcm16,
    normalize_chunk, pcm16_to_float, resample, sanitize, validate_format,
)
from backend.app.models.messages import AudioChunk


def test_stereo_to_mono_averages_channels():
    interleaved = np.array([1.0, 0.0, 0.5, 0.5, -1.0, 1.0], dtype=np.float32)
    assert np.allclose(downmix_to_mono(interleaved, 2), [0.5, 0.5, 0.0])


def test_stereo_drops_trailing_partial_frame():
    assert downmix_to_mono(np.array([1.0, 1.0, 0.5], dtype=np.float32), 2).size == 1


def test_resample_48k_to_16k_changes_length_by_a_third():
    x = np.zeros(48000, dtype=np.float32)
    assert abs(resample(x, 48000, 16000).size - 16000) <= 1


def test_resample_is_noop_at_matching_rate():
    x = np.arange(10, dtype=np.float32)
    assert resample(x, 16000, 16000) is x


def test_float_pcm16_roundtrip_is_near_lossless():
    x = np.linspace(-0.9, 0.9, 500, dtype=np.float32)
    assert np.max(np.abs(pcm16_to_float(float_to_pcm16(x)) - x)) < 1e-4


def test_pcm16_clamps_instead_of_wrapping():
    loud = np.array([5.0, -5.0], dtype=np.float32)
    assert np.all(np.abs(pcm16_to_float(float_to_pcm16(loud))) <= 1.0)


def test_odd_length_payload_is_rejected():
    with pytest.raises(InvalidAudioError):
        pcm16_to_float(b"\x00\x01\x02")


def test_empty_payload_is_allowed():
    assert pcm16_to_float(b"").size == 0


def test_nan_and_inf_are_replaced():
    out = sanitize(np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32))
    assert np.all(np.isfinite(out)) and out[3] == np.float32(0.5)


@pytest.mark.parametrize("rate,channels", [(99, 1), (1_000_000, 1), (16000, 0), (16000, 3)])
def test_impossible_formats_are_rejected(rate, channels):
    with pytest.raises(InvalidAudioError):
        validate_format(rate, channels)


def test_clipping_ratio_detects_full_scale():
    assert compute_stats(np.ones(100, dtype=np.float32)).clipping_ratio == 1.0
    assert compute_stats(np.zeros(100, dtype=np.float32)).clipping_ratio == 0.0


def test_rms_dbfs_of_half_amplitude_sine():
    t = np.arange(16000, dtype=np.float32) / 16000
    stats = compute_stats((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    assert -9.5 < stats.rms_dbfs < -8.5      # 0.5/sqrt(2) -> about -9.03 dBFS


def test_full_chunk_normalisation_produces_mono_16k():
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    mono = 0.4 * np.sin(2 * np.pi * 300 * t)
    inter = np.empty(96000, dtype=np.float32)
    inter[0::2] = mono
    inter[1::2] = mono
    chunk = AudioChunk("s", "s:1", 0, float_to_pcm16(inter), 48000, 2)
    samples, stats = normalize_chunk(chunk, 16000)
    assert abs(samples.size - 16000) <= 1
    assert samples.dtype == np.float32
    assert stats.num_samples == samples.size
