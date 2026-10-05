/* SvarKavach demo frontend.
 *
 * Decodes a local audio file, converts it to mono 16 kHz PCM16, and streams it
 * over a WebSocket at approximately real time so the backend behaves exactly as
 * it would with a live call. The browser performs no risk logic whatsoever —
 * every score and state shown here comes from the server (plan §43).
 */
'use strict';

const TARGET_RATE   = 16000;
const CHUNK_SECONDS = 0.25;                       // 250 ms per WS frame
const CHUNK_SAMPLES = TARGET_RATE * CHUNK_SECONDS;

const $ = (id) => document.getElementById(id);

const ui = {
  fileInput: $('fileInput'), fileName: $('fileName'),
  fileDuration: $('fileDuration'), fileFormat: $('fileFormat'),
  startBtn: $('startBtn'), stopBtn: $('stopBtn'), resetBtn: $('resetBtn'),
  streamBar: $('streamBar'), streamPos: $('streamPos'),
  connDot: $('connDot'), connText: $('connText'),
  detectorBadge: $('detectorBadge'), detectorName: $('detectorName'),
  mockBanner: $('mockBanner'),
  riskBanner: $('riskBanner'), riskState: $('riskState'), riskScore: $('riskScore'),
  statCurrent: $('statCurrent'), statSmoothed: $('statSmoothed'),
  statRisk: $('statRisk'), statWindows: $('statWindows'),
  recommendation: $('recommendation'),
  finalCard: $('finalCard'), finalScore: $('finalScore'), finalState: $('finalState'),
  finalWindows: $('finalWindows'), finalAudio: $('finalAudio'), finalRec: $('finalRec'),
  errorCard: $('errorCard'), errorList: $('errorList'),
  canvas: $('timeline'),
  srcFile: $('srcFile'), srcSystem: $('srcSystem'),
  filePane: $('filePane'), systemPane: $('systemPane'),
  deviceSelect: $('deviceSelect'), captureHint: $('captureHint'),
  captureBanner: $('captureBanner'), streamVerb: $('streamVerb'),
  tsWs: $('tsWs'), tsAudio: $('tsAudio'), tsVad: $('tsVad'),
  tsModel: $('tsModel'), tsDevice: $('tsDevice'), tsInfer: $('tsInfer'),
  tsSkipped: $('tsSkipped'),
};

const state = {
  pcm: null,          // Float32Array, mono @ 16 kHz
  duration: 0,
  ws: null,
  streaming: false,
  timer: null,
  offset: 0,
  points: [],         // { t, raw, smooth, risk, state }
  thresholds: { watch: 0.40, verify: 0.65, escalate: 0.85 },
  mode: 'file',       // 'file' | 'system'
  captureReady: false,
  elapsedTimer: null,
  captureStartedAt: 0,
};

// A live capture has no end, so the timeline must not grow without bound.
const MAX_POINTS = 600;

/* ── connection indicator ─────────────────────────────────── */
function setConn(kind, text) {
  ui.connDot.className = 'dot ' + kind;
  ui.connText.textContent = text;
  ui.tsWs.textContent = text;
}

function addError(message) {
  ui.errorCard.hidden = false;
  const li = document.createElement('li');
  li.textContent = message;                     // textContent: never innerHTML
  ui.errorList.appendChild(li);
}

/* ── audio decoding ───────────────────────────────────────── */
async function decodeToMono16k(file) {
  const bytes = await file.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(bytes.slice(0));
  } finally {
    ctx.close();
  }

  // Downmix to mono by averaging channels.
  const n = decoded.length;
  const mono = new Float32Array(n);
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += data[i];
  }
  if (decoded.numberOfChannels > 1) {
    for (let i = 0; i < n; i++) mono[i] /= decoded.numberOfChannels;
  }

  if (decoded.sampleRate === TARGET_RATE) {
    return { pcm: mono, sourceRate: decoded.sampleRate, duration: decoded.duration };
  }

  // Resample with an OfflineAudioContext — the browser's own resampler.
  const outLength = Math.ceil(n * TARGET_RATE / decoded.sampleRate);
  const off = new OfflineAudioContext(1, outLength, TARGET_RATE);
  const buf = off.createBuffer(1, n, decoded.sampleRate);
  buf.copyToChannel(mono, 0);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return {
    pcm: rendered.getChannelData(0).slice(),
    sourceRate: decoded.sampleRate,
    duration: decoded.duration,
  };
}

function floatToPcm16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/* ── file selection ───────────────────────────────────────── */
ui.fileInput.addEventListener('change', async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  resetAnalysis();
  ui.fileName.textContent = file.name;          // textContent sanitises by design
  ui.fileFormat.textContent = file.type || 'unknown';
  ui.fileDuration.textContent = 'decoding…';
  ui.startBtn.disabled = true;

  try {
    const { pcm, sourceRate, duration } = await decodeToMono16k(file);
    state.pcm = pcm;
    state.duration = duration;
    ui.fileDuration.textContent = duration.toFixed(2) + ' s';
    ui.fileFormat.textContent =
      `${file.type || 'audio'} · ${sourceRate} Hz → ${TARGET_RATE} Hz mono`;
    updateStartAvailability();
  } catch (err) {
    ui.fileDuration.textContent = 'decode failed';
    addError('Could not decode this audio file: ' + err.message);
  }
});

/* ── streaming ────────────────────────────────────────────── */
function readContext() {
  return {
    channel: $('ctxChannel').value,
    caller_type: $('ctxCaller').value,
    contact_known: $('ctxKnown').value === 'true',
    transaction_value: Number($('ctxValue').value) || 0,
  };
}

ui.startBtn.addEventListener('click', () => {
  if (state.mode === 'file' && !state.pcm) return;
  if (state.mode === 'system' && !state.captureReady) return;
  resetAnalysis();

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  setConn('connecting', 'Connecting');
  ui.startBtn.disabled = true;

  ws.onopen = () => {
    setConn('connected', 'Connected');
    const message = {
      type: 'start',
      session_id: 'demo-' + Date.now(),
      source: state.mode === 'system' ? 'system_audio' : 'demo_web',
      context: readContext(),
    };
    if (state.mode === 'system') {
      message.device_id = ui.deviceSelect.value;
    } else {
      message.audio_format = {
        encoding: 'pcm_s16le', sample_rate: TARGET_RATE, channels: 1,
      };
    }
    ws.send(JSON.stringify(message));
  };

  // Every handler below ignores events from a socket that is no longer the
  // current one. Without this, a previous session's late `onclose` tears down
  // the session that replaced it (Reset-then-Start, or a fast double Start).
  ws.onmessage = (event) => {
    if (state.ws !== ws) return;
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      // Never fail silently: a malformed frame is a server bug worth surfacing.
      addError('Could not parse a message from the server: ' + err.message);
      return;
    }
    handleServerEvent(msg);
  };

  ws.onerror = () => {
    if (state.ws !== ws) return;
    setConn('error', 'Connection error'); addError('WebSocket error.');
  };

  ws.onclose = () => {
    if (state.ws !== ws) return;
    stopTimer();
    stopElapsedClock();
    state.streaming = false;
    ui.captureBanner.hidden = true;
    setConn('', 'Disconnected');
    updateStartAvailability();
    ui.stopBtn.disabled = true;
    if (ui.tsAudio.textContent !== 'Complete') ui.tsAudio.textContent = 'Idle';
  };
});

ui.stopBtn.addEventListener('click', () => finishStream());
ui.resetBtn.addEventListener('click', () => {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.close();
  resetAnalysis();
  ui.fileInput.value = '';
  state.pcm = null;
  ui.fileName.textContent = ui.fileDuration.textContent = ui.fileFormat.textContent = '—';
  ui.startBtn.disabled = true;
});

function beginStreaming() {
  state.offset = 0;
  state.streaming = true;
  ui.stopBtn.disabled = false;
  ui.tsAudio.textContent = 'Streaming';
  setConn('analyzing', 'Analyzing');

  // Pace against the wall clock so we approximate real time rather than
  // drifting with setInterval.
  const startedAt = performance.now();
  let sent = 0;

  const tick = () => {
    if (!state.streaming || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;

    const elapsed = (performance.now() - startedAt) / 1000;
    const due = Math.floor(elapsed / CHUNK_SECONDS) + 1;

    while (sent < due && state.offset < state.pcm.length) {
      const end = Math.min(state.offset + CHUNK_SAMPLES, state.pcm.length);
      const pcm16 = floatToPcm16(state.pcm.subarray(state.offset, end));
      state.ws.send(pcm16.buffer);
      state.offset = end;
      sent++;
      const secs = state.offset / TARGET_RATE;
      ui.streamPos.textContent = secs.toFixed(1);
      ui.streamBar.style.width = (100 * state.offset / state.pcm.length) + '%';
    }

    if (state.offset >= state.pcm.length) { finishStream(); return; }
    state.timer = setTimeout(tick, 40);
  };
  tick();
}

function stopTimer() {
  if (state.timer) { clearTimeout(state.timer); state.timer = null; }
}

function finishStream() {
  if (!state.streaming) return;
  state.streaming = false;
  stopTimer();
  stopElapsedClock();
  ui.stopBtn.disabled = true;
  ui.tsAudio.textContent = 'Finalising';
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'stop' }));
  }
}

/* ── server events ────────────────────────────────────────── */
function handleServerEvent(msg) {
  switch (msg.type) {
    case 'session_started': {
      state.thresholds = msg.thresholds || state.thresholds;
      applyDetectorIdentity({
        model: msg.detector, is_mock: msg.is_mock, device: msg.device,
        vad: msg.vad, vad_fallback: msg.vad_fallback,
      });
      // In live mode the server pulls audio; we wait for capture_started.
      if (state.mode === 'file') beginStreaming();
      break;
    }
    case 'capture_started': {
      state.streaming = true;
      ui.stopBtn.disabled = false;
      ui.captureBanner.hidden = false;
      ui.tsAudio.textContent = `Capturing · ${msg.device_label}`;
      ui.captureHint.textContent =
        `Capturing ${msg.device_label} at ${msg.sample_rate} Hz, ${msg.channels} ch.`;
      setConn('analyzing', 'Analyzing');
      startElapsedClock();
      break;
    }
    case 'analysis': {
      if (typeof msg.synthetic_probability === 'number') {
        ui.statCurrent.textContent  = pct(msg.synthetic_probability);
        ui.tsInfer.textContent = msg.inference_ms != null ? msg.inference_ms + ' ms' : '—';
      } else {
        ui.statCurrent.textContent = '—';
      }
      ui.statSmoothed.textContent = pct(msg.smoothed_score);
      ui.statRisk.textContent     = pct(msg.risk_score);
      ui.statWindows.textContent  = msg.windows_analyzed;
      ui.tsSkipped.textContent    = msg.windows_skipped ?? 0;
      applyRiskState(msg.risk_state, msg.risk_score);

      state.points.push({
        t: msg.window_end,
        raw: typeof msg.synthetic_probability === 'number' ? msg.synthetic_probability : null,
        smooth: msg.smoothed_score,
        risk: msg.risk_score,
      });
      if (state.points.length > MAX_POINTS) state.points.shift();
      drawTimeline();
      break;
    }
    case 'capture_stopped': {
      // Server-side capture has actually ended; the banner must not keep
      // claiming this machine's audio is being recorded.
      state.streaming = false;
      ui.captureBanner.hidden = true;
      ui.stopBtn.disabled = true;
      ui.tsAudio.textContent = 'Capture stopped';
      stopElapsedClock();
      break;
    }
    case 'recommendation':
      ui.recommendation.textContent = msg.message;
      break;
    case 'final': {
      ui.finalCard.hidden = false;
      ui.finalScore.textContent   = pct(msg.final_score);
      ui.finalState.textContent   = msg.risk_state;
      ui.finalWindows.textContent = msg.windows_analyzed;
      ui.finalAudio.textContent   = (msg.audio_seconds ?? 0) + ' s';
      ui.finalRec.textContent     = msg.recommendation;
      applyRiskState(msg.risk_state, msg.final_score);
      ui.tsAudio.textContent = 'Complete';
      ui.captureBanner.hidden = true;
      stopElapsedClock();
      setConn('connected', 'Complete');
      break;
    }
    case 'status':
      break;
    case 'error':
      addError(`[${msg.code}] ${msg.message}`);
      break;
  }
}

function pct(v) { return (v == null) ? '—' : (v * 100).toFixed(1) + '%'; }

function applyRiskState(riskState, score) {
  ui.riskBanner.className = 'risk-banner state-' + riskState;
  ui.riskState.textContent = riskState === 'INSUFFICIENT_EVIDENCE'
    ? 'INSUFFICIENT EVIDENCE' : riskState;
  ui.riskScore.textContent = riskState === 'INSUFFICIENT_EVIDENCE' ? 'n/a' : pct(score);
}

function resetAnalysis() {
  stopTimer();
  stopElapsedClock();
  ui.captureBanner.hidden = true;
  state.streaming = false;
  state.points = [];
  state.offset = 0;
  ui.statCurrent.textContent = ui.statSmoothed.textContent = ui.statRisk.textContent = '—';
  ui.statWindows.textContent = '0';
  ui.streamBar.style.width = '0%';
  ui.streamPos.textContent = '0.0';
  ui.tsInfer.textContent = '—';
  ui.tsSkipped.textContent = '0';
  ui.finalCard.hidden = true;
  ui.errorCard.hidden = true;
  ui.errorList.replaceChildren();
  ui.recommendation.textContent = 'Select an audio file and start the analysis.';
  ui.riskBanner.className = 'risk-banner state-idle';
  ui.riskState.textContent = 'AWAITING AUDIO';
  ui.riskScore.textContent = '—';
  drawTimeline();
}

/* ── timeline ─────────────────────────────────────────────── */
function drawTimeline() {
  const cv = ui.canvas;
  const ratio = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || 600;
  const cssH = 220;
  if (cv.width !== cssW * ratio || cv.height !== cssH * ratio) {
    cv.width = cssW * ratio; cv.height = cssH * ratio;
  }
  const g = cv.getContext('2d');
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  g.clearRect(0, 0, cssW, cssH);

  const pad = { l: 40, r: 12, t: 12, b: 24 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;
  const y = (v) => pad.t + h * (1 - v);

  // horizontal grid + axis labels
  g.font = '10px ui-monospace, monospace';
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (let v = 0; v <= 1.0001; v += 0.25) {
    g.strokeStyle = 'rgba(148,163,196,.14)';
    g.beginPath(); g.moveTo(pad.l, y(v)); g.lineTo(pad.l + w, y(v)); g.stroke();
    g.fillStyle = '#6b7a9e';
    g.fillText((v * 100).toFixed(0) + '%', pad.l - 7, y(v));
  }

  // threshold lines, always labelled with text
  const bands = [
    ['watch', state.thresholds.watch, '#eab308'],
    ['verify', state.thresholds.verify, '#f97316'],
    ['escalate', state.thresholds.escalate, '#ef4444'],
  ];
  g.setLineDash([4, 4]); g.textAlign = 'left';
  for (const [name, v, colour] of bands) {
    g.strokeStyle = colour; g.globalAlpha = .55;
    g.beginPath(); g.moveTo(pad.l, y(v)); g.lineTo(pad.l + w, y(v)); g.stroke();
    g.globalAlpha = 1; g.fillStyle = colour;
    g.fillText(name.toUpperCase(), pad.l + 4, y(v) - 7);
  }
  g.setLineDash([]);

  if (state.points.length === 0) {
    g.fillStyle = '#6b7a9e'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('Awaiting analysis windows…', pad.l + w / 2, pad.t + h / 2);
    return;
  }

  const tMax = Math.max(state.duration, state.points[state.points.length - 1].t, 1);
  const x = (t) => pad.l + w * Math.min(t / tMax, 1);

  const series = [
    ['risk',   '#fb923c', 2],
    ['smooth', '#a78bfa', 2],
    ['raw',    '#60a5fa', 1.5],
  ];
  for (const [key, colour, width] of series) {
    g.strokeStyle = colour; g.lineWidth = width;
    g.beginPath();
    let started = false;
    for (const p of state.points) {
      if (p[key] == null) continue;
      const px = x(p.t), py = y(p[key]);
      if (!started) { g.moveTo(px, py); started = true; } else { g.lineTo(px, py); }
    }
    g.stroke();
    // mark the latest sample so single-point sessions are still visible
    const last = [...state.points].reverse().find((p) => p[key] != null);
    if (last) {
      g.fillStyle = colour;
      g.beginPath(); g.arc(x(last.t), y(last[key]), 3, 0, Math.PI * 2); g.fill();
    }
  }

  // x axis
  g.fillStyle = '#6b7a9e'; g.textAlign = 'center'; g.textBaseline = 'top';
  g.fillText('0 s', pad.l, pad.t + h + 6);
  g.fillText(tMax.toFixed(1) + ' s', pad.l + w, pad.t + h + 6);
}


/* ── audio source mode ────────────────────────────────────── */
function setMode(mode) {
  if (state.streaming) return;                 // never switch mid-session
  state.mode = mode;
  const live = mode === 'system';
  ui.srcFile.classList.toggle('active', !live);
  ui.srcSystem.classList.toggle('active', live);
  ui.srcFile.setAttribute('aria-selected', String(!live));
  ui.srcSystem.setAttribute('aria-selected', String(live));
  ui.filePane.hidden = live;
  ui.systemPane.hidden = !live;
  ui.streamVerb.textContent = live ? 'captured' : 'streamed';
  resetAnalysis();
  updateStartAvailability();
  if (live) loadCaptureDevices();
}

function updateStartAvailability() {
  ui.startBtn.disabled = state.mode === 'file'
    ? !state.pcm
    : !state.captureReady;
}

async function loadCaptureDevices() {
  ui.deviceSelect.replaceChildren(new Option('Loading…'));
  state.captureReady = false;
  updateStartAvailability();
  try {
    const info = await (await fetch('/audio/devices')).json();
    ui.deviceSelect.replaceChildren();
    if (!info.enabled || !info.devices.length) {
      ui.deviceSelect.replaceChildren(new Option('Unavailable'));
      ui.deviceSelect.disabled = true;
      ui.captureHint.textContent = info.reason
        || 'No loopback capture device was found on this machine.';
      state.captureReady = false;
    } else {
      for (const device of info.devices) {
        ui.deviceSelect.appendChild(new Option(device.label, device.id));
      }
      ui.deviceSelect.disabled = false;
      ui.captureHint.textContent =
        "Captures whatever is playing through this machine's output — a Zoom "
        + 'call, a softphone, a browser meeting.';
      state.captureReady = true;
    }
  } catch (err) {
    ui.deviceSelect.replaceChildren(new Option('Unavailable'));
    ui.deviceSelect.disabled = true;
    ui.captureHint.textContent = 'Could not reach the server to list devices.';
    state.captureReady = false;
  }
  updateStartAvailability();
}

function startElapsedClock() {
  state.captureStartedAt = performance.now();
  stopElapsedClock();
  state.elapsedTimer = setInterval(() => {
    const secs = (performance.now() - state.captureStartedAt) / 1000;
    ui.streamPos.textContent = secs.toFixed(1);
    // No known total for a live capture, so show motion rather than progress.
    ui.streamBar.style.width = ((secs * 10) % 100) + '%';
  }, 200);
}

function stopElapsedClock() {
  if (state.elapsedTimer) { clearInterval(state.elapsedTimer); state.elapsedTimer = null; }
}

ui.srcFile.addEventListener('click', () => setMode('file'));
ui.srcSystem.addEventListener('click', () => setMode('system'));

/* ── detector identity ────────────────────────────────────── */
function applyDetectorIdentity({ model, is_mock, device, vad, vad_fallback }) {
  ui.detectorName.textContent = is_mock ? 'MOCK — DEMO ONLY' : model;
  ui.detectorBadge.className = 'detector-badge ' + (is_mock ? 'mock' : 'real');
  ui.mockBanner.hidden = !is_mock;
  ui.tsModel.textContent = model || '—';
  ui.tsDevice.textContent = device || '—';
  if (vad) ui.tsVad.textContent = vad + (vad_fallback ? ' (fallback)' : '');
}

/* Show which detector is active before the operator runs anything. Without
 * this the first thing they learn about the detector is a mock warning
 * mid-analysis, which reads as a missing feature rather than a setting. */
async function loadServerConfig() {
  try {
    const cfg = await (await fetch('/config')).json();
    if (cfg.thresholds) state.thresholds = cfg.thresholds;
    if (cfg.detector) {
      applyDetectorIdentity({
        model: cfg.detector.model,
        is_mock: cfg.detector.is_mock,
        device: cfg.detector.device,
        vad: cfg.vad && cfg.vad.name,
        vad_fallback: cfg.vad && cfg.vad.is_fallback,
      });
    }
    drawTimeline();
  } catch (err) {
    // The dashboard still works if /config is unreachable; the session_started
    // event fills the same fields in a moment.
  }
}

window.addEventListener('resize', drawTimeline);
drawTimeline();
loadServerConfig();
