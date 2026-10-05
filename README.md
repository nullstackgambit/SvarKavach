# SvarKavach

**Real-time synthetic voice detection platform — MVP**

SvarKavach receives a near-real-time voice stream, normalises and segments the
audio, applies an AI anti-spoofing detector, smooths and contextualises the
detector output, and presents an actionable voice-impersonation risk state on a
live security dashboard.

The system is **advisory**. It never blocks a transaction, and abstaining is a
first-class output.

---

## 1. What it does

```
Browser demo frontend
        │  WebSocket: binary PCM16 + JSON control
        ▼
FastAPI ingestion gateway ──► session manager
        ▼
Audio normalizer            mono · PCM16 · 16 kHz
        ▼
Silero VAD + sliding windows        4.0375 s window, 0.5 s hop
        ▼
Quality gate                 abstain rather than guess
        ▼
XLS-R-300M + AASIST detector        behind a swappable interface
        ▼
Calibration → EWMA smoothing → k-of-m sustained evidence
        ▼
Risk engine + demo transaction context
        ▼
NONE · WATCH · VERIFY · ESCALATE · INSUFFICIENT_EVIDENCE
        ▼
Live browser dashboard
```

The browser is only one **audio-source adapter**. Everything downstream of the
gateway consumes a normalised `AudioChunk` and has no idea where the audio came
from, so Zoom, SIP/SIPREC, WebRTC, or contact-centre media can be added later
without touching the detection or risk layers.

## 2. MVP scope

**Included:** WebSocket ingestion · browser file upload streamed at ~real time ·
decoding and normalisation to mono 16 kHz PCM16 · Silero VAD with an energy
fallback · 4 s/0.5 s sliding windows · XLS-R-300M + AASIST detection behind a
provider interface · a mock detector · calibration, EWMA smoothing and k-of-m
sustained evidence · a context-aware risk engine · live WebSocket results ·
a dashboard with timeline · health/readiness/config endpoints · structured
logging · tests · an offline evaluation script.

**Deliberately excluded:** prosody analysis · speaker consistency or identity
verification · diarisation · face verification · caller-ID verification · Zoom,
WhatsApp, SIPREC and PSTN integrations · production auth/SSO · multi-tenancy ·
cloud deployment automation · distributed inference · persistent storage of raw
audio.

## 3. Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Transport | `backend/app/api/websocket.py` | Protocol framing and validation. No model logic. |
| Ingestion | `backend/app/ingestion/` | Session registry, decode, downmix, resample, sanitise. |
| Preprocessing | `backend/app/preprocessing/` | VAD, sliding windows, quality gates. |
| Detection | `backend/app/detector/` | `DetectorProvider` interface + implementations. |
| Scoring | `backend/app/scoring/` | Calibration, smoothing, risk policy. |
| Pipeline | `backend/app/pipeline/` | Wires the layers, owns async inference scheduling. |
| Presentation | `frontend/` | Display only — performs **no** security decisions. |

Boundaries that are enforced, not merely suggested: no risk threshold lives
inside the detector, no model-specific logic lives in the WebSocket handler, and
the frontend never decides a risk state.

## 4. Installation

> Setting this up fresh on Windows or Ubuntu, with or without a GPU? [`SETUP.md`](SETUP.md) is a copy-paste runbook for all four combinations.

Requires **Python 3.11+**. Verified on Python 3.12.3 / Linux.

**Linux / macOS**

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Windows (PowerShell)**

```
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`uvicorn[standard]` skips `uvloop` on Windows automatically (it carries a
`sys_platform != 'win32'` marker), so the install works unchanged. Everything
else in `requirements.txt` ships Windows wheels.

`ffmpeg` is optional — it is used only by the offline evaluation script to
decode compressed formats libsndfile cannot read. The live path never needs it,
because the browser decodes the audio itself.

Check the environment at any time:

```bash
python -m scripts.check_environment
```

## 5. Model setup

```bash
python -m scripts.download_models
```

This fetches Silero VAD (~2.3 MB) and the XLS-R-300M + AASIST ONNX model
(~1.21 GB) into `models/`. For the VAD alone:

```bash
python -m scripts.download_models --vad-only
```

| Model | Source | Licence |
|---|---|---|
| Silero VAD | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT |
| Spectra-AASIST (XLS-R-300M + AASIST) | [lab260/Spectra-AASIST](https://huggingface.co/lab260/Spectra-AASIST) | Apache-2.0 |

Details and the verified I/O contract are in [`models/README.md`](models/README.md).

## 6. Running the backend

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

If `backend` is not importable, run it from the repository root with
`PYTHONPATH=.` prefixed.

> **The detector defaults to `mock` until you create a `.env`.** With no `.env`
> present the dashboard shows a `MOCK — DEMO ONLY` badge and a yellow banner
> reading *"Scores are simulated by a deterministic stand-in"*. That is the
> system reporting its configuration honestly, not a missing feature — the real
> XLS-R-300M + AASIST detector is fully implemented. Turn it on in § 9.
>
> The default is `mock` because a fresh clone has no model weights, and the
> design refuses to silently substitute simulated inference for a requested real
> detector. Check what is actually running at any time:
>
> ```bash
> curl -s localhost:8000/health
> ```

## 7. Opening the frontend

Open <http://localhost:8000>. The backend serves the dashboard itself — there is
no separate frontend server or build step.

## 8. Testing with mock inference

The default `DETECTOR_PROVIDER=mock` needs no model download:

```bash
uvicorn backend.app.main:app --port 8000
```

The mock produces deterministic, content-derived scores so demos are
reproducible. It is **not** a detector. The dashboard shows a permanent
`MOCK — DEMO ONLY` badge and a warning banner whenever it is active.

To force a high-risk demo path:

```bash
MOCK_BASE_SCORE=0.93 MOCK_SPREAD=0.05 uvicorn backend.app.main:app --port 8000
```

## 9. Enabling the real model

```bash
cp .env.example .env
```

Then edit `.env` and set:

```
DETECTOR_PROVIDER=xlsr_aasist_onnx
```

Restart the server, then verify:

```bash
curl -s localhost:8000/health
```

`"model": "xlsr_aasist_onnx"` and `"is_mock": false` confirm it is live.

If the real detector cannot load, startup **fails loudly**. It never silently
degrades to simulated inference. Set `ALLOW_DETECTOR_FALLBACK=true` only if you
explicitly want mock fallback.

### CUDA

```bash
pip uninstall -y onnxruntime && pip install -r requirements-gpu.txt
```

With `MODEL_DEVICE=auto` the CUDA execution provider is used when available.
Measure your own latency:

```bash
python -m scripts.benchmark_detector --provider xlsr_aasist_onnx --runs 20
```

### PyTorch provider (optional)

```bash
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
```

```bash
python -m scripts.download_models --torch
```

Then set `DETECTOR_PROVIDER=xlsr_aasist_torch`.

## 10. Live system audio capture

Instead of replaying a file, SvarKavach can analyse whatever is currently
playing through this machine's output — a Zoom desktop call, a softphone, a
browser meeting. This is the first real `AudioSourceAdapter`; the detection,
scoring and risk layers are untouched.

> ### Security: read before enabling
>
> This application has **no authentication**. With capture enabled, any client
> that can reach the dashboard can start recording this machine's audio output.
> That is a materially different exposure from "a user uploads their own file".
>
> - It is **off by default** (`SYSTEM_AUDIO_ENABLED=false`).
> - Serve it on `127.0.0.1` only, or put authentication and TLS in front of it.
> - The server logs a warning on every start while capture is enabled.
>
> Note the warning reflects the *configured* `HOST`; the address actually bound
> is whatever `uvicorn --host` was given.

### Setup

```bash
pip install -r requirements-audio.txt
```

Then set `SYSTEM_AUDIO_ENABLED=true` in `.env` and restart.

| Platform | Mechanism | Requirement |
|---|---|---|
| Linux | PulseAudio/PipeWire monitor source | `pipewire-pulse` (standard on modern desktops) or PulseAudio |
| Windows | WASAPI loopback | nothing extra |
| macOS | — | no native loopback; needs a virtual device such as BlackHole. Not supported. |

Check what the server can see:

```bash
curl -s localhost:8000/audio/devices
```

### Using it

Open the dashboard, switch the Audio Source toggle to **System audio (live)**,
pick a capture device, and press Start. A red **CAPTURING SYSTEM AUDIO** bar
stays visible for the whole session. Press Stop to end it and produce the final
verdict; `MAX_SESSION_SECONDS` (default 900) is a hard safety cap so a forgotten
session cannot record indefinitely.

Device ids shown to the browser are opaque and server-assigned — the underlying
device name never leaves the server, and a client cannot name an arbitrary
device.

### Tuning for live audio

Live audio contains silence between utterances, and **that is what most affects
the score**. See § 16 for the measurements; in short, raise
`MIN_SPEECH_SECONDS` if you see high scores on quiet stretches, and re-check
that you are not simply rejecting every window.

## 11. WebSocket protocol

One endpoint: `ws://<host>/ws/stream`.

**Client → server, session start (JSON):**

```json
{
  "type": "start",
  "session_id": "demo-001",
  "source": "demo_web",
  "audio_format": { "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1 },
  "context": {
    "channel": "zoom-demo",
    "caller_type": "executive",
    "contact_known": true,
    "transaction_value": 2500000
  }
}
```

**Client → server, live capture instead of upload:**

```json
{ "type": "start", "source": "system_audio", "device_id": "dev-6233185e",
  "context": { ... } }
```

The server then captures the audio itself and replies with:

```json
{ "type": "capture_started", "source": "system_audio",
  "device_label": "Monitor of Built-in Audio Analog Stereo",
  "sample_rate": 48000, "channels": 2, "backend": "soundcard" }
```

In this mode the client sends **no** audio frames; doing so is a protocol error.
`{"type":"stop"}` ends the session identically to the upload path.

**Client → server, audio (upload mode only):** binary frames of signed 16-bit
little-endian PCM.
Send 100–500 ms per frame; never the whole recording in one frame. Frames larger
than `MAX_CHUNK_BYTES` are rejected.

**Client → server, end:** `{"type": "stop"}`

## 12. Example server events

```json
{ "type": "session_started", "session_id": "demo-001", "status": "ready",
  "detector": "xlsr_aasist_onnx", "is_mock": false, "device": "cuda" }
```

```json
{ "type": "analysis", "session_id": "demo-001",
  "window_start": 2.0, "window_end": 6.037,
  "synthetic_probability": 0.81, "smoothed_score": 0.77, "risk_score": 0.847,
  "risk_state": "VERIFY", "confidence": 0.62, "sustained": true,
  "model": "xlsr_aasist_onnx", "inference_ms": 41.3, "quality_flags": [] }
```

```json
{ "type": "recommendation", "risk_state": "VERIFY",
  "message": "Elevated synthetic-voice risk detected. Verify the caller using an independent trusted channel before authorizing sensitive actions." }
```

```json
{ "type": "final", "session_id": "demo-001", "final_score": 0.874,
  "risk_state": "VERIFY", "windows_analyzed": 42, "windows_skipped": 0 }
```

```json
{ "type": "error", "code": "INVALID_AUDIO",
  "message": "Audio stream could not be decoded or normalized." }
```

Error codes: `INVALID_JSON`, `INVALID_MESSAGE`, `SESSION_NOT_STARTED`,
`SESSION_ALREADY_STARTED`, `INVALID_AUDIO`, `CHUNK_TOO_LARGE`,
`SESSION_TOO_LONG`, `UNSUPPORTED_FORMAT`, `DETECTOR_ERROR`, `SERVER_BUSY`,
`CAPTURE_UNAVAILABLE`, `CAPTURE_FAILED`, `INTERNAL_ERROR`.

## 13. Scoring, in three distinct numbers

The UI shows three values deliberately, because conflating them is how these
systems mislead people:

| Name | Meaning |
|---|---|
| `detector_score` | Raw model output for one window. Not a validated probability. |
| `smoothed_score` | EWMA over windows (`alpha = 0.35`). |
| `risk_score` | `smoothed_score × context multiplier`, clamped to `[0, 1]`. |

A state above `WATCH` additionally requires **sustained evidence** — k of the
last m windows above the verify threshold (default 3 of 5). A single spiking
window raises awareness but never triggers an operator action.

Default thresholds — `WATCH 0.40`, `VERIFY 0.65`, `ESCALATE 0.85` — are
**demonstration defaults**, configurable via `.env`, and not validated against
representative data.

## 14. Tests

```bash
pytest
```

105 tests covering normalisation (stereo→mono, 48k→16k, float↔PCM16, malformed
input), windowing (first window, hop, overlap, short streams, memory bounds),
the risk engine (threshold boundaries, context multiplier, clamping, abstention),
smoothing (seeding, k-of-m), the detector interface and calibration, VAD behaviour, and full WebSocket
sessions including malformed messages and session isolation.

A few checks need **genuine recorded speech** and cannot be faked — Silero VAD
only fires on real voice, and the detector's spoof/bona-fide index convention can
only be confirmed against a known-human recording. A synthetic chirp scores zero
speech frames, so it proves nothing. The suite resolves real speech in this
order:

1. **`sample_audio/`** — anything you drop under `bona_fide/` is used first, so
   the tests get stronger as soon as you add your own recordings.
2. **ALSA's desktop speech clips** (`/usr/share/sounds/alsa/*.wav`) — a
   convenience for Linux contributors. Not a dependency, and absent on Windows
   and macOS.
3. **Neither** — the real-speech tests skip. Nothing fails for want of audio.

So the suite passes identically on Linux, Windows and macOS. Reproduce the
no-system-audio path on Linux with:

```bash
SVARKAVACH_NO_SYSTEM_AUDIO=1 pytest
```

## 15. Offline evaluation

Drop labelled audio into `sample_audio/bona_fide/` and `sample_audio/synthetic/`
(see [`sample_audio/README.md`](sample_audio/README.md)), then:

```bash
python -m scripts.evaluate_dataset --input sample_audio --provider xlsr_aasist_onnx
```

Reports accuracy, precision, recall, F1, EER and mean inference time. If the
directory holds no labelled audio, it says so rather than reporting numbers.

## 16. Platform notes

Developed and tested on Linux (Python 3.12). The application code is
platform-neutral — `pathlib` throughout, no POSIX-only imports, no shelling out
on the request path — but only the Linux path has been executed end to end.

**Windows.** Expected to work. Specifically checked:

| Concern | Status |
|---|---|
| `uvloop` (POSIX-only) | Correctly gated behind a `sys_platform != 'win32'` marker |
| Path handling | `pathlib` everywhere; no `os.path.join`, no hardcoded POSIX paths in app code |
| Console output | All script output is cp1252-safe, so `python -m scripts.… > log.txt` will not raise `UnicodeEncodeError` |
| Test suite | Passes with no system audio present (`SVARKAVACH_NO_SYSTEM_AUDIO=1` reproduces that path on Linux) |
| Dependencies | numpy, scipy, soundfile, onnxruntime all publish Windows wheels |

Two Windows-specific reminders:

- Use `.venv\Scripts\Activate.ps1`, not `source .venv/bin/activate`.
- The README's shell snippets are bash. On PowerShell, set environment variables
  with `$env:DETECTOR_PROVIDER="xlsr_aasist_onnx"` rather than the inline
  `VAR=value command` form — or just use `.env`, which works identically on all
  platforms.

`ffmpeg` remains optional everywhere; it is used only by the offline evaluation
script for formats libsndfile cannot decode.

**macOS.** Same position as Windows: no known blockers, not executed end to end.
The system-speech test fixture does not exist there either, which is why the
suite no longer depends on it.

## 17. Limitations

Read this section before quoting any accuracy figure.

1. **The detector generalises poorly out of domain, and there is no engineering
   fix.** Anti-spoofing models degrade sharply on unseen vocoders, new TTS
   releases, and telephony codecs. Benchmark EERs published for this model
   family are not evidence about your audio, your languages, or your channels.
2. **Thresholds are unvalidated defaults.** They were chosen to demonstrate
   state transitions, not fitted to data. Calibration is currently an explicit
   identity pass-through; `CALIBRATION_TEMPERATURE` is the hook for fixing that
   once you have a held-out set.
3. **No number in this repository is a measured accuracy claim.** The evaluation
   script reports only what it computes on files you supply.
4. **Latency depends entirely on the host — measure it, don't assume it.**
   When inference is slower than the hop, the pipeline keeps the newest window
   and reports the rest as `windows_skipped` — visible in the dashboard, never
   hidden. The only figures below are ones actually measured on the development
   VM (Intel i7-14700, 8 vCPU, **no AVX/AVX2 exposed by the hypervisor**,
   ONNX Runtime CPU provider):

   | Metric | Measured |
   |---|---|
   | Model load | 2.6 s |
   | Mean inference / 4.0375 s window | 697 ms |
   | p50 / p95 | 691 / 1123 ms |
   | Real-time factor | 0.173× of window duration |
   | Keeps up with a 0.5 s hop | No — sustainable hop ~0.70 s here |

   That machine has no GPU, and its CPU exposes only SSE4.2, so these are close
   to a worst case for a 300M-parameter model. **No CUDA number is quoted here
   because none was measured.** Run this on your target box:

   ```bash
   python -m scripts.benchmark_detector --provider xlsr_aasist_onnx --runs 20
   ```

   If the reported mean exceeds `HOP_SECONDS × 1000`, either accept the skipped
   windows or raise `HOP_SECONDS` to the sustainable value the script prints.
5. **Silence inside a window drives the score up — this is the single biggest
   gotcha for live capture.** The detector was trained on continuous speech, and
   reads digital-silence padding as a spoofing artefact. Measured on this
   machine, streaming genuine recorded human speech through loopback:

   Controlled measurement — one clean recording of genuine human speech, the
   only variable being how much of the window is digital silence (no capture,
   no resampling, no editing of the speech itself):

   | Speech inside the 4.0375 s window | p_synthetic |
   |---|---|
   | 4.04 s (full) | 0.0344 |
   | 3.50 s | 0.3182 |
   | 3.00 s | 0.2136 |
   | 2.50 s | 0.3971 |
   | 2.00 s | 0.7167 |
   | 1.50 s | 0.8549 |

   Nothing is wrong with the capture path: audio captured through loopback
   scores the same as the source file (0.0015–0.0143 captured vs 0.0012–0.0161
   direct) when the window is full of speech. What moves the score is silence.

   The default `MIN_SPEECH_SECONDS=1.5` therefore admits windows that are ~62%
   silence, and those score as synthetic. Raising it filters them out, but set
   it too high (3.0 s on a 4.0375 s window) and **every** window abstains and you
   get `INSUFFICIENT_EVIDENCE` for the whole session. The usable value depends on
   your audio; tune it against representative recordings, not against a demo
   clip. This is a threshold-tuning problem, not something to fix in code.

   The k-of-m sustained-evidence rule is the safety net that stops one such
   window from escalating on its own — which is exactly why the bug that
   bypassed it (fixed, see `test_final_verdict.py`) mattered.

6. **Contextual gating is a demo mechanism, not a fraud model.** The transaction
   multiplier illustrates policy shaping; it is not risk-validated.
7. **Audio-only.** A multimodal video deepfake is only partially addressed.
8. **This is an arms race.** Attackers adopt new open TTS models for free.
   Budget for retraining, or plan for decay.

## 18. Future integration with Zoom / telephony

The ingestion contract is already source-agnostic:

```
Zoom bot / SIPREC / WebRTC / contact-centre media
                ↓
        AudioSourceAdapter
                ↓
    normalised AudioChunk (16 kHz mono PCM16)
                ↓
      existing detection pipeline (unchanged)
```

Adding an adapter means producing `AudioChunk` objects. The preprocessing,
detection, scoring, and risk layers need no changes. The browser demo is a
controlled simulation of exactly this adapter layer.

## 19. Privacy

- Audio is held in memory only and is **never written to disk** by default.
- No raw recordings are retained; no voiceprints are computed or stored.
- Only session metrics (counters, scores, timings) are kept, in memory, for the
  life of the connection.
- Raw audio samples are never logged. Neither are secrets.
- `PERSIST_AUDIO_DEBUG` is opt-in and defaults to `false`.
- Live system-audio capture is opt-in (`SYSTEM_AUDIO_ENABLED`, default `false`),
  is announced in the UI by a persistent red banner for the whole session, and
  writes nothing to disk. It records the machine's output, so anything else
  playing — music, notifications — is captured too while a session runs.
- `/config` exposes only non-secret settings and never environment variables.

**Production deployments** require TLS and authentication on the WebSocket
endpoint — neither is implemented here, because this is a local MVP. Voice is
personal data under the DPDP Act 2023; deploy inside your own network and
complete a DPIA before processing real calls.

## 20. Licence note

SvarKavach integrates third-party models under their own licences: Silero VAD
(MIT) and Spectra-AASIST (Apache-2.0). Neither is redistributed in this
repository; both are downloaded from their upstream sources at setup time.
