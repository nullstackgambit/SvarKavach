import numpy as np
import pytest

from backend.app.preprocessing.vad import EnergyVad
from backend.app.preprocessing.windowing import SlidingWindowBuffer

SR, W, H = 16000, 64600, 8000


def make(vad=None):
    return SlidingWindowBuffer(W, H, SR, vad=vad)


def test_no_window_before_enough_audio():
    b = make()
    b.push(np.zeros(W - 1, dtype=np.float32))
    assert list(b.pop_windows()) == []


def test_first_window_is_exact():
    b = make()
    b.push(np.arange(W, dtype=np.float32))
    (win,) = list(b.pop_windows())
    assert win.start_sample == 0 and win.end_sample == W
    assert win.samples.size == W
    assert win.samples[0] == 0 and win.samples[-1] == W - 1


def test_hop_is_exact_and_windows_overlap():
    b = make()
    b.push(np.arange(W + 3 * H, dtype=np.float32))
    wins = list(b.pop_windows())
    assert len(wins) == 4
    starts = [w.start_sample for w in wins]
    assert starts == [0, H, 2 * H, 3 * H]
    assert wins[0].end_sample - wins[1].start_sample == W - H


def test_window_count_matches_formula():
    total = 200_000
    b = make()
    b.push(np.zeros(total, dtype=np.float32))
    assert len(list(b.pop_windows())) == (total - W) // H + 1


def test_chunked_push_matches_single_push():
    a, b = make(), make()
    data = np.arange(W + 2 * H, dtype=np.float32)
    a.push(data)
    for i in range(0, data.size, 1600):
        b.push(data[i:i + 1600])
    assert [w.start_sample for w in a.pop_windows()] == [w.start_sample for w in b.pop_windows()]


def test_memory_stays_bounded_on_long_stream():
    b = make()
    peak = 0
    for _ in range(400):
        b.push(np.zeros(H, dtype=np.float32))
        list(b.pop_windows())
        peak = max(peak, b.buffered_samples)
    assert b.total_samples == 400 * H
    assert peak <= W + 2 * H


def test_flush_pads_short_tail_to_full_length():
    b = make()
    b.push(np.ones(16000, dtype=np.float32))
    assert list(b.pop_windows()) == []
    tail = b.flush()
    assert tail is not None and tail.is_final
    assert tail.samples.size == W
    assert tail.samples[-1] == 1.0 and tail.samples[0] == 0.0   # right-aligned


def test_flush_returns_none_when_drained():
    b = make()
    b.push(np.zeros(W, dtype=np.float32))
    list(b.pop_windows())
    b2 = make()
    assert b2.flush() is None


def test_speech_accounting_with_vad():
    b = make(vad=EnergyVad())
    loud = np.ones(W, dtype=np.float32) * 0.5
    b.push(loud)
    (win,) = list(b.pop_windows())
    assert win.speech_ratio > 0.9
    assert win.speech_seconds > 3.5


def test_silence_reports_no_speech():
    b = make(vad=EnergyVad())
    b.push(np.zeros(W, dtype=np.float32))
    (win,) = list(b.pop_windows())
    assert win.speech_ratio == 0.0 and win.speech_seconds == 0.0


def test_without_vad_everything_counts_as_speech():
    b = make()
    b.push(np.zeros(W, dtype=np.float32))
    (win,) = list(b.pop_windows())
    assert win.speech_ratio == 1.0


def test_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        SlidingWindowBuffer(100, 200, SR)
    with pytest.raises(ValueError):
        SlidingWindowBuffer(0, 10, SR)
