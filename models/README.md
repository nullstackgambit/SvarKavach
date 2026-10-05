# Models

Model weights are **not** committed to this repository. Fetch them with:

```bash
python -m scripts.download_models
```

## Silero VAD — `silero_vad/silero_vad.onnx`

| | |
|---|---|
| Source | https://github.com/snakers4/silero-vad |
| Licence | MIT |
| Size | ~2.3 MB |
| Role | Speech/non-speech gating so the detector is not run on silence |

At 16 kHz the model consumes **512 new samples per call** and internally
prepends **64 samples of context** from the previous frame, so the tensor fed to
ONNX Runtime is 576 wide. Omitting that context makes the model return ~0.001
for every input, including clear speech — the failure is silent, so
`backend/app/preprocessing/vad.py` handles the context explicitly.

If this file is missing the application falls back to an RMS energy gate and
labels itself `energy_rms_fallback` in `/health`, `/config`, and the dashboard.

## Spectra-AASIST — `spectra_aasist/`

| | |
|---|---|
| Source | https://huggingface.co/lab260/Spectra-AASIST |
| Licence | Apache-2.0 (repository metadata; the README text says MIT — both permissive) |
| Architecture | `facebook/wav2vec2-xls-r-300m` SSL front-end → MLP bridge → AASIST graph-attention back-end |
| Parameters | ~300 M |
| Files | `spectra-aasist.onnx` (~1.21 GB), or `model.safetensors` + `model.py` for the PyTorch path |

### Contract

Verified against the upstream source, not assumed:

```
input   wav     float32 [batch, 64600]   mono, 16 kHz
output  logits  float32 [batch, 2]       index 0 = spoof, index 1 = bona fide
```

64,600 samples is 4.0375 s at 16 kHz, which is why `WINDOW_SECONDS` defaults to
that value rather than a round 4.0.

Preprocessing, per the upstream model card: **pre-emphasis** (coefficient 0.97,
matching `torchaudio.functional.preemphasis`) then pad/truncate to 64,600
samples. Implemented in `backend/app/detector/xlsr_aasist.py`.

### Verifying the class-index convention

The index convention is easy to get backwards, and inverting it inverts every
score the product reports. It was confirmed empirically by running recorded
human speech through the model:

| Input | p_spoof | p_bona_fide |
|---|---|---|
| Recorded human speech (3 clips) | 0.019 / 0.025 / 0.027 | 0.98 / 0.98 / 0.97 |
| White noise | 0.737 | 0.263 |

Re-run this check after any model change:

```bash
python -m scripts.benchmark_detector --provider xlsr_aasist_onnx
```

### Known limitations

Anti-spoofing models generalise poorly outside their training distribution.
Published EERs for this model family are reported on public benchmarks; they are
**not** evidence about your audio, your languages, or your telephony codecs.
Treat every threshold in this repository as a demonstration default until you
have measured performance on representative data. See `README.md` § Limitations.
