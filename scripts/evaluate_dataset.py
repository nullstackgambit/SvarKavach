"""Offline evaluation over a labelled directory of audio files (plan §29).

    python -m scripts.evaluate_dataset --input sample_audio
    python -m scripts.evaluate_dataset --input sample_audio --provider xlsr_aasist_onnx

Expected layout:

    sample_audio/
    ├── bona_fide/   *.wav|flac|mp3|m4a|ogg
    └── synthetic/   *.wav|flac|mp3|m4a|ogg

Reports accuracy, precision, recall, F1 and EER at the file level. If the
directory holds no labelled audio, the script says so — it never invents
numbers (plan §47.6).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import Settings                        # noqa: E402
from backend.app.detector.factory import build_detector        # noqa: E402
from backend.app.ingestion.normalizer import resample, sanitize  # noqa: E402
from backend.app.preprocessing.vad import build_vad            # noqa: E402
from backend.app.preprocessing.windowing import SlidingWindowBuffer  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}
LABEL_DIRS = {"bona_fide": 0, "bonafide": 0, "real": 0, "human": 0,
              "synthetic": 1, "spoof": 1, "fake": 1}


def load_audio(path: Path, target_rate: int) -> np.ndarray | None:
    """Load via soundfile, falling back to ffmpeg for compressed formats."""
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    except Exception:
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            return None
        try:
            raw = subprocess.run(
                ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le",
                 "-ac", "1", "-ar", str(target_rate), "-"],
                capture_output=True, check=True, timeout=120,
            ).stdout
        except Exception:
            return None
        return sanitize(np.frombuffer(raw, dtype="<f4").copy())
    return sanitize(resample(mono, rate, target_rate))


def score_file(audio, settings, detector, vad) -> tuple[float | None, int, float]:
    """Mean synthetic score across the file's analysis windows."""
    buffer = SlidingWindowBuffer(
        settings.window_samples, settings.hop_samples,
        settings.target_sample_rate, vad=vad, vad_threshold=settings.vad_threshold,
    )
    buffer.push(audio)
    windows = list(buffer.pop_windows())
    if not windows:
        tail = buffer.flush()
        if tail is not None:
            windows = [tail]

    scores, total_ms = [], 0.0
    for window in windows:
        if vad is not None and window.speech_seconds < settings.min_speech_seconds:
            continue
        result = detector.predict(window.samples, window.sample_rate)
        scores.append(result.synthetic_probability)
        total_ms += result.inference_ms

    if not scores:
        return None, 0, 0.0
    return float(np.mean(scores)), len(scores), total_ms / len(scores)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold achieving it.

    Sweeps every candidate threshold and returns the point where the false
    accept rate (bona fide scored as synthetic) and the false reject rate
    (synthetic scored as bona fide) are closest.
    """
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan"), float("nan")

    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best_gap = np.inf
    best = (float("nan"), float("nan"))
    for threshold in candidates:
        far = float((negatives >= threshold).mean())   # bona fide called synthetic
        frr = float((positives < threshold).mean())    # synthetic called bona fide
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best = ((far + frr) / 2.0, float(threshold))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a labelled audio directory")
    parser.add_argument("--input", default="sample_audio", help="dataset root")
    parser.add_argument("--provider", default=None,
                        help="override DETECTOR_PROVIDER (mock|xlsr_aasist_onnx|xlsr_aasist_torch)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="decision threshold on the synthetic score")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    overrides = {"detector_provider": args.provider} if args.provider else {}
    settings = Settings(**overrides)
    detector = build_detector(settings)
    vad = build_vad(settings.vad_enabled, settings.vad_model_path or None,
                    settings.target_sample_rate)

    files: list[tuple[Path, int]] = []
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not directory.is_dir():
            continue
        label = LABEL_DIRS.get(directory.name.lower())
        if label is None:
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                files.append((path, label))

    if not files:
        print(f"No labelled audio found under {root}/.")
        print("Expected sub-directories named 'bona_fide/' and 'synthetic/'.")
        print("Nothing was evaluated; no metrics are reported.")
        return 0

    print(f"Detector: {detector.name} (device={detector.device}, mock={detector.is_mock})")
    if detector.is_mock:
        print("WARNING: the mock detector produces simulated scores. The metrics below "
              "measure the harness, not a real model.\n")

    scores, labels, latencies, skipped = [], [], [], []
    started = time.perf_counter()
    for path, label in files:
        audio = load_audio(path, settings.target_sample_rate)
        if audio is None or audio.size == 0:
            skipped.append((path, "could not decode"))
            continue
        score, n_windows, mean_ms = score_file(audio, settings, detector, vad)
        if score is None:
            skipped.append((path, "no scoreable speech windows"))
            continue
        scores.append(score); labels.append(label); latencies.append(mean_ms)
        print(f"  {path.relative_to(root)!s:<48} "
              f"label={'synthetic' if label else 'bona_fide':<9} "
              f"score={score:.4f} windows={n_windows}")

    if not scores:
        print("\nNo file produced a scoreable window; no metrics reported.")
        for path, reason in skipped:
            print(f"  skipped {path.name}: {reason}")
        return 0

    s = np.array(scores); y = np.array(labels)
    predicted = (s >= args.threshold).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())

    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and precision + recall else float("nan"))

    print("\n" + "=" * 60)
    print(f"Total files:            {len(files)}")
    print(f"Evaluated:              {len(scores)}")
    print(f"Bona fide:              {int((y == 0).sum())}")
    print(f"Synthetic:              {int((y == 1).sum())}")
    print(f"Decision threshold:     {args.threshold}")
    print(f"Confusion (TP/TN/FP/FN):{tp} / {tn} / {fp} / {fn}")
    print(f"Accuracy:               {accuracy:.4f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall:                 {recall:.4f}")
    print(f"F1:                     {f1:.4f}")
    if (y == 0).any() and (y == 1).any():
        eer, eer_threshold = compute_eer(s, y)
        print(f"EER:                    {eer:.4f} (at threshold {eer_threshold:.4f})")
    else:
        print("EER:                    not computable (needs both classes)")
    print(f"Mean inference time:    {np.mean(latencies):.1f} ms per window")
    print(f"Wall clock:             {time.perf_counter() - started:.1f} s")
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for path, reason in skipped:
            print(f"  {path.name}: {reason}")
    print("\nNote: thresholds are demonstration defaults and have not been "
          "validated on representative data.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
