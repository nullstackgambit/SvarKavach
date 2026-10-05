# Sample audio

Audio files are **not** committed (see `.gitignore`). Drop your own recordings
here.

## Layout for the evaluation script

```
sample_audio/
├── bona_fide/          # genuine human speech
│   ├── human_01.wav
│   └── human_02.wav
└── synthetic/          # AI-generated / cloned speech
    ├── synthetic_01.wav
    └── synthetic_02.wav
```

Directory names `bona_fide/`, `bonafide/`, `real/`, `human/` map to label 0;
`synthetic/`, `spoof/`, `fake/` map to label 1. Anything else is ignored.

Supported: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus`, `.webm`.
Compressed formats need `ffmpeg` on the host. Any sample rate and channel count
works — the pipeline resamples to 16 kHz mono.

Run:

```bash
python -m scripts.evaluate_dataset --input sample_audio --provider xlsr_aasist_onnx
```

For the **live dashboard** you do not need this layout — just pick any audio
file in the browser.

## Demo cases worth preparing

**Case A — bona fide.** A genuine recording. Expect a low detector score,
`NONE` or `WATCH`.

**Case B — synthetic.** Output from a TTS/voice-cloning system. Expect a high
detector score, `VERIFY` or `ESCALATE`.

**Case C — poor quality.** Near-silence, heavy clipping, or under ~1.5 s of
speech per window. Expect `INSUFFICIENT_EVIDENCE` — the system abstains rather
than guessing. This case matters: it demonstrates the system refusing to
convert bad audio into a confident accusation.

> Do not describe a sample as genuinely human or genuinely synthetic unless you
> actually know its provenance. A file's label is a claim about where it came
> from, not a model output.

## These files also strengthen the test suite

Anything you place under `bona_fide/` is picked up automatically by the tests
that need genuine recorded speech (VAD firing, detector index convention). With
nothing here, those tests fall back to the speech clips ALSA ships on Linux
desktops, and skip entirely on Windows and macOS.

Files under `synthetic/` are deliberately **not** used as a "known human speech"
fixture, for obvious reasons.

Nothing here is committed — the whole directory is gitignored except this README.
