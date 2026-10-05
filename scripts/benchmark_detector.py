"""Measure real inference latency for the configured detector (plan §31).

    python -m scripts.benchmark_detector --provider xlsr_aasist_onnx --runs 20

Reports measured numbers only. Run this on the machine you intend to demo on —
latency differs by an order of magnitude between CPU and CUDA, and a figure
measured elsewhere is not evidence about your box.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import Settings                   # noqa: E402
from backend.app.detector.factory import build_detector   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark detector inference latency")
    parser.add_argument("--provider", default=None, help="override DETECTOR_PROVIDER")
    parser.add_argument("--device", default=None, help="override MODEL_DEVICE (auto|cpu|cuda)")
    parser.add_argument("--runs", type=int, default=10, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=2, help="untimed warm-up iterations")
    args = parser.parse_args()

    overrides = {}
    if args.provider:
        overrides["detector_provider"] = args.provider
    if args.device:
        overrides["model_device"] = args.device
    settings = Settings(**overrides)

    print(f"Loading detector '{settings.detector_provider}' ...")
    load_start = time.perf_counter()
    detector = build_detector(settings)
    load_seconds = time.perf_counter() - load_start

    window_samples = detector.expected_samples or settings.window_samples
    window_seconds = window_samples / settings.target_sample_rate
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(window_samples) * 0.05).astype(np.float32)

    for _ in range(args.warmup):
        detector.predict(audio, settings.target_sample_rate)

    timings = []
    for _ in range(args.runs):
        start = time.perf_counter()
        detector.predict(audio, settings.target_sample_rate)
        timings.append((time.perf_counter() - start) * 1000.0)

    timings.sort()
    mean = statistics.fmean(timings)
    p50 = timings[len(timings) // 2]
    p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]

    print("\n" + "=" * 58)
    print(f"Detector:            {detector.name}")
    print(f"Device:              {detector.device}")
    print(f"Mock:                {detector.is_mock}")
    print(f"Model load:          {load_seconds:.2f} s")
    print(f"Window:              {window_samples} samples ({window_seconds:.3f} s)")
    print(f"Runs:                {args.runs}")
    print(f"Mean latency:        {mean:.1f} ms")
    print(f"p50 / p95:           {p50:.1f} / {p95:.1f} ms")
    print(f"Min / max:           {timings[0]:.1f} / {timings[-1]:.1f} ms")
    print(f"Real-time factor:    {mean / 1000 / window_seconds:.3f}x of window duration")

    hop_ms = settings.hop_seconds * 1000
    if mean <= hop_ms:
        print(f"Keeps up with the {settings.hop_seconds}s hop: YES "
              f"({mean:.0f} ms <= {hop_ms:.0f} ms)")
    else:
        print(f"Keeps up with the {settings.hop_seconds}s hop: NO "
              f"({mean:.0f} ms > {hop_ms:.0f} ms)")
        print("  The pipeline will drop intermediate windows (latest-window-wins) "
              "and report them as windows_skipped.")
        print(f"  Sustainable hop on this machine: ~{mean / 1000:.2f} s")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
