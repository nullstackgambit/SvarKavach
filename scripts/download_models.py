"""Download the model weights SvarKavach needs.

    python -m scripts.download_models              # VAD + XLS-R+AASIST (ONNX)
    python -m scripts.download_models --vad-only
    python -m scripts.download_models --torch      # also the PyTorch checkpoint

Nothing here is bundled with the repository; every file is fetched from its
upstream source so provenance and licence stay traceable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_BASE = "https://huggingface.co/lab260/Spectra-AASIST/resolve/main"
SILERO_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)

ONNX_TARGET = REPO_ROOT / "models" / "spectra_aasist" / "spectra-aasist.onnx"
VAD_TARGET = REPO_ROOT / "models" / "silero_vad" / "silero_vad.onnx"

TORCH_FILES = {
    "model.safetensors": REPO_ROOT / "models" / "spectra_aasist" / "model.safetensors",
    "model.py": REPO_ROOT / "models" / "spectra_aasist" / "model.py",
    "config.json": REPO_ROOT / "models" / "spectra_aasist" / "config.json",
}


def download(url: str, target: Path, min_bytes: int = 1024) -> bool:
    if target.is_file() and target.stat().st_size >= min_bytes:
        print(f"  OK {target.name} already present "
              f"({target.stat().st_size / 1e6:.1f} MB) — skipping")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"  -> {target.name}\n    from {url}")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            written = 0
            with tmp.open("wb") as handle:
                for block in response.iter_bytes(1 << 20):
                    handle.write(block)
                    written += len(block)
                    if total:
                        pct = 100 * written / total
                        print(f"\r    {written/1e6:7.1f} / {total/1e6:.1f} MB "
                              f"({pct:5.1f}%)", end="", flush=True)
            print()
    except httpx.HTTPError as exc:
        print(f"    FAILED: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    if written < min_bytes:
        print(f"    FAILED: only {written} bytes received", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(target)
    print(f"    saved to {target}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SvarKavach model weights")
    parser.add_argument("--vad-only", action="store_true",
                        help="download only the Silero VAD model (~2 MB)")
    parser.add_argument("--torch", action="store_true",
                        help="also download the PyTorch checkpoint (~1.2 GB extra)")
    args = parser.parse_args()

    print("Silero VAD (MIT)")
    ok = download(SILERO_URL, VAD_TARGET, min_bytes=100_000)

    if not args.vad_only:
        print("\nSpectra-AASIST - XLS-R-300M + AASIST (Apache-2.0), ONNX, ~1.2 GB")
        ok &= download(f"{HF_BASE}/spectra-aasist.onnx", ONNX_TARGET, min_bytes=10_000_000)

    if args.torch:
        print("\nSpectra-AASIST PyTorch checkpoint, ~1.2 GB")
        for name, target in TORCH_FILES.items():
            min_size = 10_000_000 if name.endswith(".safetensors") else 2
            ok &= download(f"{HF_BASE}/{name}", target, min_bytes=min_size)

    print("\nDone." if ok else "\nOne or more downloads failed.", file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
