import {
  DEFAULTS, DriverState, Level, LEFT_EYE_CONTOUR, LEFT_IRIS, MOUTH_CONTOUR, RIGHT_EYE_CONTOUR, RIGHT_IRIS,
  AlarmPolicy, cueFor, handAtEar, phoneNearFace, readFace,
} from "./engine.js";
import { TESSELATION } from "./mesh.js";

const PHONE_EVERY = 8; // phone and hand models run in their own worker on every Nth frame

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const stage = $("stage");

const cfg = DEFAULTS;
let state = new DriverState(cfg);
let ready = false;
let mode = "demo";
let chapters = [];
let chapterIdx = -1;
let lastVideoT = 0;
let stream = null;
let camT0 = 0;
let frameNo = 0;
let phone = { t: -1, phones: [], hands: [] };
let lastKey = "";
let fps = 0, lastFrameAt = 0, lastUi = 0;
let running = false;
const voice = new AlarmPolicy();

// ---- performance budget (open with ?perf) ------------------------------------
// Per analysed frame, on the main thread: p95 under 16 ms in total and no task over 50 ms,
// so the page keeps up with 30 fps video and never janks. window.__perf() reports it.
const PERF = new URLSearchParams(location.search).has("perf");
const perf = { face: [], phone: [], draw: [], ui: [], total: [], long: [], gaps: [] };
if (PERF) {
  try {
    new PerformanceObserver((l) => l.getEntries().forEach((e) => perf.long.push(Math.round(e.duration)))).observe({ type: "longtask", buffered: true });
  } catch {}
  const pct = (a, q) => (a.length ? [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(a.length * q))] : 0);
  window.__perf = () => Object.fromEntries(Object.entries(perf).map(([k, a]) => [k,
    k === "long" ? { count: a.length, max: Math.max(0, ...a) } : { n: a.length, p50: +pct(a, 0.5).toFixed(1), p95: +pct(a, 0.95).toFixed(1), max: +Math.max(0, ...a).toFixed(1) }]));
  window.__perfReset = () => Object.values(perf).forEach((a) => (a.length = 0));
}
let soundOn = readSoundPref();
let alarmOn = readPref("dm-alarm", "on") !== "off"; // chimes and the siren, separate from the spoken voice
let view = readPref("dm-view2", "light"); // "light": outlines only, "dots": every landmark as a dot, "mesh": the full triangulated face mesh

// ---- sound: reel audio plus spoken alerts ------------------------------------

function readSoundPref() {
  try { return localStorage.getItem("dm-sound") !== "off"; } catch { return true; }
}

// Pre-rendered neural voice (driver_monitor/voice.py) for every phrase it still matches;
// the browser's own speech, a female English voice where there is one, for anything else.
let clips = {};
let playing = null, queued = null;
async function loadVoice() {
  const manifest = await fetch("/voice/manifest.json").then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
  for (const [key, phrase] of Object.entries(manifest)) {
    const a = new Audio(`/voice/${key}.mp3`);
    a.preload = "auto";
    a.addEventListener("ended", () => {
      playing = null;
      if (queued) playClip(...queued.splice(0));
    });
    clips[key] = { audio: a, phrase };
  }
}

function playClip(audio) {
  queued = null;
  playing = audio;
  audio.currentTime = 0;
  audio.play().catch(() => (playing = null));
}

let speaker = null;
function pickVoice() {
  const en = (window.speechSynthesis?.getVoices() || []).filter((v) => /^en/i.test(v.lang));
  const female = /aria|jenny|ava|emma|michelle|sonia|libby|zira|samantha|serena|karen|moira|google us english|female/i;
  speaker = en.find((v) => female.test(v.name) && /natural|online|neural|google/i.test(v.name))
    || en.find((v) => female.test(v.name)) || en[0] || null;
}
if ("speechSynthesis" in window) {
  pickVoice();
  speechSynthesis.addEventListener?.("voiceschanged", pickVoice);
}

// An urgent alert interrupts; a caution waits behind the current phrase (one deep).
let voiceTimer = 0;
function say(cue, urgent) {
  const lead = alarmOn ? chime(urgent) : 0; // the attention tone comes first, then the words
  if (!soundOn) return;
  clearTimeout(voiceTimer);
  if (lead) voiceTimer = setTimeout(() => speak(cue, urgent), lead * 1000);
  else speak(cue, urgent);
}

function speak(cue, urgent) {
  const clip = clips[cue.key];
  if (clip && clip.phrase === cue.phrase) {
    if (playing && !playing.paused) {
      if (!urgent) return void (queued = [clip.audio]);
      playing.pause();
    }
    return playClip(clip.audio);
  }
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(cue.phrase);
  if (speaker) u.voice = speaker;
  u.rate = 1.0;
  if (urgent) speechSynthesis.cancel();
  else if (speechSynthesis.pending) return;
  speechSynthesis.speak(u);
}

// Closed eyes for 2 s after the first phrase: a loud two-tone alarm that speeds up until they open.
let ac = null, sirenTimer = 0, sirenGain = null;
function siren(hz) {
  if (!alarmOn || hz <= 0) {
    if (sirenTimer) { clearTimeout(sirenTimer); sirenTimer = 0; }
    return;
  }
  if (sirenTimer) return void (siren.hz = hz); // running: the next beat picks up the new rate
  siren.hz = hz;
  try {
    ac ||= new AudioContext();
    if (ac.state === "suspended") ac.resume();
    sirenGain ||= ac.createGain(); // gain 1 = full scale
    sirenGain.connect(ac.destination);
  } catch { return; }
  hush(); // the voice stops: the alarm owns the speakers now
  let high = true;
  const beat = () => {
    const half = 0.5 / siren.hz;
    const o = ac.createOscillator();
    o.type = "square";
    o.frequency.value = high ? 1900 : 1250;
    o.connect(sirenGain);
    o.start();
    o.stop(ac.currentTime + half * 0.92);
    high = !high;
    sirenTimer = setTimeout(beat, half * 1000);
  };
  beat();
}

// Attention tone before the words: a soft two-note chime for a caution, three sharp beeps for an alert.
// Returns how long it lasts, in seconds.
function chime(urgent) {
  try {
    ac ||= new AudioContext();
    if (ac.state === "suspended") ac.resume();
  } catch { return 0; }
  const notes = urgent ? [[1480, 0], [1480, 0.17], [1480, 0.34]] : [[660, 0], [880, 0.16]];
  const len = urgent ? 0.1 : 0.22, peak = urgent ? 0.32 : 0.22;
  const t0 = ac.currentTime + 0.01;
  for (const [f, at] of notes) {
    const o = ac.createOscillator(), g = ac.createGain();
    o.type = urgent ? "square" : "sine";
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t0 + at);
    g.gain.exponentialRampToValueAtTime(peak, t0 + at + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + at + len);
    o.connect(g).connect(ac.destination);
    o.start(t0 + at);
    o.stop(t0 + at + len + 0.02);
  }
  try { navigator.vibrate?.(urgent ? [180, 70, 180, 70, 360] : [120]); } catch {}
  return notes[notes.length - 1][1] + len + 0.12;
}

function hush() {
  clearTimeout(voiceTimer);
  queued = null;
  playing?.pause();
  playing = null;
  window.speechSynthesis?.cancel();
}

function setSound(on) {
  soundOn = on;
  try { localStorage.setItem("dm-sound", on ? "on" : "off"); } catch {}
  if (!on) hush();
  video.muted = !on || mode !== "demo";
  const btn = $("sound");
  btn.classList.toggle("is-off", !on);
  btn.setAttribute("aria-pressed", String(on));
  btn.setAttribute("aria-label", on ? "Mute the voice and reel sound" : "Turn the voice and reel sound on");
  btn.title = on ? "Voice on" : "Voice off";
}

function setAlarm(on) {
  alarmOn = on;
  try { localStorage.setItem("dm-alarm", on ? "on" : "off"); } catch {}
  if (!on) siren(0);
  const btn = $("alarm");
  btn.classList.toggle("is-off", !on);
  btn.setAttribute("aria-pressed", String(on));
  btn.setAttribute("aria-label", on ? "Turn alarm tones off" : "Turn alarm tones on");
  btn.title = on ? "Alarm tones on" : "Alarm tones off";
}

// ---- models -----------------------------------------------------------------

// Inference runs in worker.js. One frame is in flight at a time; frames that arrive while it
// is busy are skipped, so the page never queues work it cannot finish.
let worker = null, phoneWorker = null, phoneReady = false, busy = false, phoneBusy = false, nextId = 0, inflight = null;
let phoneJob = null;

// Real download progress: each worker announces the files it will fetch (with their sizes) and
// reports bytes as they arrive. Only the face pipeline blocks the start; phone models follow.
const MB = 1048576;
const dl = { plan: new Map(), got: new Map() };
function onProgress(d) {
  if (d.type === "plan") {
    if (d.role === "face") for (const a of d.assets) dl.plan.set(a.key, a);
  } else if (d.type === "progress" && d.role === "face") {
    dl.got.set(d.key, d.loaded);
    showLoad();
  }
}

function showLoad() {
  let total = 0, loaded = 0, current = null;
  for (const [k, a] of dl.plan) {
    total += a.bytes;
    const got = dl.got.get(k) || 0;
    loaded += got;
    if (got < a.bytes && !current) current = a;
  }
  if (!total) return;
  const pct = Math.min(100, Math.round((loaded / total) * 100));
  if (!ready) { $("status-label").textContent = "Starting detection"; $("status-sub").textContent = `Loading models, ${pct}%`; }
  $("veil-progress").hidden = false;
  $("veil-fill").style.width = `${pct}%`;
  $("veil-bar").setAttribute("aria-valuenow", String(pct));
  $("veil-text").textContent = current ? `Loading ${current.label.toLowerCase()}` : "Starting the face model";
  $("veil-detail").textContent = `${(loaded / MB).toFixed(1)} of ${(total / MB).toFixed(1)} MB`;
}

const LOAD_TIMEOUT_MS = 20000;

function loadModels() {
  worker?.terminate();
  phoneWorker?.terminate();
  dl.plan.clear();
  dl.got.clear();
  phoneReady = false;
  return new Promise((resolve, reject) => {
    // Fail only when nothing has moved for LOAD_TIMEOUT_MS: a slow link that keeps downloading is fine.
    let timer;
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => reject(new Error("timeout")), LOAD_TIMEOUT_MS); };
    const done = (fn, v) => { clearTimeout(timer); fn(v); };
    arm();
    worker = new Worker("/worker.js");
    worker.onerror = (e) => done(reject, new Error(e.message || "worker failed"));
    worker.onmessage = (e) => {
      const d = e.data;
      if (d.type === "ready") { engineDelegate = d.delegate; done(resolve); phoneStarter?.(); }
      else if (d.type === "error") done(reject, new Error(d.message));
      else if (d.type === "result") onResult(d);
      else { arm(); onProgress(d); }
    };
    worker.postMessage({ type: "init", role: "face" });
    // Phone evidence is optional and loads in its own worker: the face pipeline starts without it.
    // It starts once the face pipeline is ready, so its 15 MB never competes with the first load.
    const startPhone = () => {
    phoneWorker = new Worker("/worker.js");
    phoneWorker.onerror = (e) => phoneOff(e.message || "worker failed");
    phoneWorker.onmessage = (e) => {
      const d = e.data;
      if (d.type === "phone-ready") phoneReady = true;
      else if (d.type === "phone-error") phoneOff(d.message);
      else if (d.type === "result") onPhone(d);
    };
    phoneWorker.postMessage({ type: "init", role: "phone" });
    };
    phoneStarter = startPhone;
  });
}
let engineDelegate = "GPU", phoneStarter = null;

// Phone evidence is a bonus: if its models fail, say so (HUD and log) instead of silently reading "No".
let phoneBroken = false;
function phoneOff(why) {
  if (phoneBroken) return;
  phoneBroken = true;
  phoneReady = false;
  console.warn("Phone detection off:", why);
  logLine(mode === "demo" ? video.currentTime : 0, "Phone detection off", 1);
}

// ---- veil -------------------------------------------------------------------

function veil({ text, action, busy = false } = {}) {
  const v = $("veil");
  if (!text) {
    v.classList.add("is-hidden");
    return;
  }
  v.classList.remove("is-hidden");
  v.classList.toggle("is-idle", !busy);
  if (!busy || text !== "Loading models") $("veil-progress").hidden = !(busy && dl.plan.size);
  $("veil-text").textContent = text;
  const btn = $("veil-action");
  btn.hidden = !action;
  if (action) {
    btn.textContent = action.label;
    btn.onclick = action.run;
  }
}

// ---- inputs -----------------------------------------------------------------

async function startDemo() {
  stopCamera();
  mode = "demo";
  stage.classList.remove("is-mirrored");
  $("transport").classList.remove("is-hidden");
  video.srcObject = null;
  if (!video.src.endsWith("/demo/reel.mp4")) {
    video.src = "/demo/reel.mp4";
    video.loop = true;
  }
  resetState();
  chapterIdx = -1;
  veil({ text: "An AI-generated driver acts out each scenario. The monitor sees the footage the way it would see a dashboard camera.", action: { label: "Play test reel", run: playDemo } });
}

async function playDemo() {
  veil(); // the reel plays at once; detection joins in when the models land
  video.muted = !soundOn; // a click got us here, so sound is allowed
  try {
    await video.play();
  } catch {
    veil({ text: "Playback was blocked by the browser.", action: { label: "Try again", run: playDemo } });
  }
}

async function startCamera() {
  mode = "camera";
  video.pause();
  video.muted = true;
  video.removeAttribute("src");
  video.load();
  stage.classList.add("is-mirrored");
  $("transport").classList.add("is-hidden");
  resetState();
  veil({ text: "Your camera feed stays on this device. Nothing is recorded or uploaded.", action: { label: "Turn on camera", run: openCamera } });
}

async function openCamera() {
  veil({ text: "Waiting for camera permission", busy: true });
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (err) {
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    veil({
      text: denied ? "Camera access was blocked. Allow it in your browser's site settings, then try again." : "No camera was found on this device.",
      action: { label: "Try again", run: openCamera },
    });
    return;
  }
  if (mode !== "camera") return stopCamera();
  video.srcObject = stream;
  await video.play();
  camT0 = performance.now();
  resetState();
  veil();
}

function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}

function resetState() {
  state = new DriverState(cfg);
  phone = { t: -1, phones: [], hands: [] };
  lastKey = "";
  voice.reset();
  siren(0);
  hush();
}

// ---- chapters (demo reel) ---------------------------------------------------

async function loadChapters() {
  chapters = await fetch("/demo/chapters.json").then((r) => r.json()).catch(() => []);
  const host = $("chapters");
  host.innerHTML = "";
  chapters.forEach((ch, i) => {
    const b = document.createElement("button");
    b.className = "chapter";
    b.type = "button";
    b.innerHTML = `<span class="chapter-track"><span class="chapter-fill"></span></span><span class="chapter-label"></span>`;
    b.querySelector(".chapter-label").textContent = ch.label;
    b.title = ch.expect;
    b.onclick = () => {
      video.currentTime = ch.start + 0.01;
      if (video.paused) playDemo();
    };
    host.appendChild(b);
    ch.el = b;
  });
}

function chapterAt(t) {
  for (let i = chapters.length - 1; i >= 0; i--) if (t >= chapters[i].start) return i;
  return 0;
}

// ---- main loop --------------------------------------------------------------

// Analyse each decoded video frame once. rAF runs at the display rate (60 Hz or more),
// twice the video's 30 fps, and the reel's clock moves between frames, so without this
// the same picture was run through the models two times over.
const FRAME_CB = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
let newFrame = !FRAME_CB;
if (FRAME_CB) {
  const onFrame = () => {
    newFrame = true;
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
}

// The models see a 640 px wide copy of the frame: landmarks come back normalised, so the
// precision is the same, and uploading a quarter of the pixels is most of the per-frame cost.
const INPUT_W = new URLSearchParams(location.search).has("fullres") ? 0 : 640;

function tick() {
  requestAnimationFrame(tick);
  if (!ready || busy || video.readyState < 2 || video.paused || video.ended) return;
  if (video.videoWidth === 0) return;
  if (!newFrame) return;
  newFrame = !FRAME_CB;

  const now = performance.now();
  let t;
  if (mode === "demo") {
    t = video.currentTime;
    if (t === lastVideoT) return; // no new frame yet
    const idx = chapterAt(t);
    // The reel is one drive, calibrated once. Only a loop or a scrub backwards starts over.
    if (t < lastVideoT - 0.5) {
      resetState();
      chapterIdx = -1;
    }
    if (idx !== chapterIdx) {
      chapterIdx = idx;
      if (chapters[idx]) logDivider(chapters[idx].label);
    }
    lastVideoT = t;
  } else {
    t = (now - camT0) / 1000;
  }

  if (canvas.width !== video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  const w = canvas.width, h = canvas.height;
  const iw = INPUT_W && w > INPUT_W ? INPUT_W : w, ih = Math.round((h / w) * iw);
  busy = true;
  const id = ++nextId;
  frameNo++;
  inflight = { id, t, w, h, mode };
  const size = { resizeWidth: iw, resizeHeight: ih, resizeQuality: "low" };
  createImageBitmap(video, size).then(
    (bitmap) => worker.postMessage({ type: "frame", id, bitmap, ts: now, role: "face" }, [bitmap]),
    () => { busy = false; },
  );
  if (phoneReady && !phoneBusy && frameNo % PHONE_EVERY === 0) {
    phoneBusy = true;
    phoneJob = { id, t, w, h, mode };
    createImageBitmap(video, size).then(
      (bitmap) => phoneWorker.postMessage({ type: "frame", id, bitmap, ts: now, role: "phone" }, [bitmap]),
      () => { phoneBusy = false; },
    );
  }
}

function onPhone(d) {
  phoneBusy = false;
  const job = phoneJob;
  if (!job || job.id !== d.id || job.mode !== mode || !d.ph) return;
  const { t, w, h } = job, k = w / d.bw;
  phone = {
    t,
    phones: d.ph.phones.map((p) => ({ box: p.box.map((v) => v * k), score: p.score })),
    hands: d.ph.hands.map((l) => l.map(([x, y]) => [x * w, y * h])),
  };
  if (PERF) perf.phone.push(d.ms2 ?? d.msAll);
}

function onResult(d) {
  busy = false;
  const job = inflight;
  if (!job || job.id !== d.id || job.mode !== mode) return;
  const { t, w, h } = job;
  const p0 = performance.now();

  let reading = null;
  if (d.face) {
    const lm = [];
    for (let i = 0; i < d.face.xyz.length; i += 3) lm.push({ x: d.face.xyz[i], y: d.face.xyz[i + 1], z: d.face.xyz[i + 2] });
    const categories = Object.entries(d.face.shapes).map(([categoryName, score]) => ({ categoryName, score }));
    reading = readFace({ faceLandmarks: [lm], faceBlendshapes: [{ categories }] }, w, h);
  }
  const fresh = Math.abs(t - phone.t) < 1;

  const f = { t, face: !!reading, ear: 0, mar: 0, pitch: 0, yaw: 0, gaze: 0, blink: null, jawOpen: null, phone: fresh && phone.phones.length > 0 };
  if (reading) {
    Object.assign(f, { ear: reading.ear, mar: reading.mar, pitch: reading.pitch, yaw: reading.yaw, gaze: reading.gaze, blink: reading.blink, jawOpen: reading.jawOpen });
    if (fresh) {
      f.phoneNearFace = phoneNearFace(phone.phones, reading.box);
      f.handAtEar = phone.hands.some((hd) => handAtEar(hd, reading.box));
    }
  }
  const s = state.update(f);

  const done = performance.now();
  const dt = done - lastFrameAt;
  lastFrameAt = done;
  if (dt > 0 && dt < 1000) fps = fps ? fps * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
  // On a slow device the reel would outrun the models; slow it down instead of skipping evidence.
  if (mode === "demo" && fps) video.playbackRate = Math.max(0.35, Math.min(1, fps / 22));

  draw(reading, s, fresh ? phone : null);
  const p3 = performance.now();
  logState(s, t);
  const alarm = voice.update(s.reasons, t, s.level === Level.ALERT, s.yawns); // live during calibration too
  if (alarm.cue) say(alarm.cue, alarm.preempt);
  siren(alarm.sirenHz);
  if (done - lastUi > 66) {
    lastUi = done;
    renderPanel(reading, s, fresh ? phone : null);
  }
  if (PERF) {
    const p4 = performance.now();
    perf.face.push(d.ms); // worker time, off the main thread
    perf.draw.push(p3 - p0); perf.ui.push(p4 - p3); perf.total.push(p4 - p0); // main-thread cost only
    if (dt > 0 && dt < 1000) perf.gaps.push(dt);
  }
}

// ---- drawing ----------------------------------------------------------------

function readPref(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}

function setView(v) {
  view = v;
  for (const b of document.querySelectorAll(".view-toggle [data-view]")) b.setAttribute("aria-checked", String(b.dataset.view === v));
  try { localStorage.setItem("dm-view2", v); } catch {}
}
document.querySelector(".view-toggle").addEventListener("click", (e) => {
  const b = e.target.closest("[data-view]");
  if (b) setView(b.dataset.view);
});
setView(view);

const COLORS = { ok: "#5fd08a", warn: "#f5b546", alert: "#ef4444", line: "rgba(255,255,255,0.85)", soft: "rgba(255,255,255,0.35)" };

function path(points, idx) {
  ctx.beginPath();
  idx.forEach((i, k) => (k ? ctx.lineTo(points[i][0], points[i][1]) : ctx.moveTo(points[i][0], points[i][1])));
  ctx.closePath();
  ctx.stroke();
}

function draw(r, s, ph) {
  const w = canvas.width, h = canvas.height;
  const u = Math.max(1, w / 900);
  ctx.clearRect(0, 0, w, h);
  ctx.lineJoin = "round";

  if (ph) {
    ctx.lineWidth = 2 * u;
    for (const p of ph.phones) {
      ctx.strokeStyle = COLORS.alert;
      ctx.strokeRect(p.box[0], p.box[1], p.box[2] - p.box[0], p.box[3] - p.box[1]);
    }
    ctx.strokeStyle = "rgba(120,190,255,0.8)";
    ctx.lineWidth = 1.5 * u;
    for (const hd of ph.hands) {
      for (const [a, b] of HAND_LINKS) {
        ctx.beginPath();
        ctx.moveTo(hd[a][0], hd[a][1]);
        ctx.lineTo(hd[b][0], hd[b][1]);
        ctx.stroke();
      }
    }
  }
  if (!r) return;

  // Face brackets
  const [x0, y0, x1, y1] = r.box;
  const pad = (x1 - x0) * 0.08, len = (x1 - x0) * 0.14;
  ctx.strokeStyle = s.level === Level.ALERT ? COLORS.alert : s.level === Level.WARN ? COLORS.warn : COLORS.soft;
  ctx.lineWidth = 2 * u;
  for (const [cx, cy, dx, dy] of [[x0 - pad, y0 - pad, 1, 1], [x1 + pad, y0 - pad, -1, 1], [x0 - pad, y1 + pad, 1, -1], [x1 + pad, y1 + pad, -1, -1]]) {
    ctx.beginPath();
    ctx.moveTo(cx + dx * len, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * len);
    ctx.stroke();
  }

  if (view === "mesh") {
    // The full triangulated mesh (2556 edges) in one path, tinted by the current level.
    ctx.lineWidth = Math.max(0.8, 0.8 * u);
    ctx.strokeStyle = s.level === Level.ALERT ? "rgba(239,68,68,0.5)" : s.level === Level.WARN ? "rgba(245,181,70,0.5)" : "rgba(200,215,210,0.55)";
    ctx.beginPath();
    const pts = r.points;
    for (let i = 0; i < TESSELATION.length; i += 2) {
      const a = pts[TESSELATION[i]], b = pts[TESSELATION[i + 1]];
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
  } else if (view === "dots") {
    // Every landmark as a dot, tinted by the current level; one path keeps it cheap.
    const rad = 1.6 * u;
    ctx.fillStyle = s.level === Level.ALERT ? "rgba(239,68,68,0.9)" : s.level === Level.WARN ? "rgba(245,181,70,0.9)" : "rgba(95,208,138,0.9)";
    ctx.beginPath();
    for (const [px, py] of r.points) {
      ctx.moveTo(px + rad, py);
      ctx.arc(px, py, rad, 0, Math.PI * 2);
    }
    ctx.fill();
  }

  ctx.lineWidth = 1.5 * u;
  ctx.strokeStyle = s.eyesClosed ? COLORS.alert : COLORS.ok;
  path(r.points, RIGHT_EYE_CONTOUR);
  path(r.points, LEFT_EYE_CONTOUR);
  ctx.strokeStyle = s.yawning ? COLORS.warn : COLORS.line;
  path(r.points, MOUTH_CONTOUR);
  ctx.fillStyle = "#fff";
  for (const i of [RIGHT_IRIS, LEFT_IRIS]) {
    ctx.beginPath();
    ctx.arc(r.points[i][0], r.points[i][1], 2.2 * u, 0, Math.PI * 2);
    ctx.fill();
  }
}

const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

// ---- panel ------------------------------------------------------------------

const READINGS = [
  { key: "ear", label: "Eye openness", lo: 0, hi: 0.45, fmt: (v) => v.toFixed(2) },
  { key: "jaw", label: "Jaw open", lo: 0, hi: 1, fmt: (v) => v.toFixed(2), marks: [cfg.yawn_jaw] },
  { key: "yaw", label: "Head turn", lo: -60, hi: 60, fmt: (v) => `${v.toFixed(0)}°`, marks: [-cfg.yaw_limit_deg, cfg.yaw_limit_deg] },
  { key: "pitch", label: "Head tilt", lo: -45, hi: 45, fmt: (v) => `${v.toFixed(0)}°`, marks: [-cfg.pitch_down_limit_deg, cfg.pitch_up_limit_deg] },
  { key: "gaze", label: "Gaze", lo: -1, hi: 1, fmt: (v) => v.toFixed(2), marks: [-cfg.gaze_limit, cfg.gaze_limit] },
  { key: "perclos", label: "PERCLOS, 60 s", lo: 0, hi: 0.4, fmt: (v) => `${Math.round(v * 100)}%`, marks: [cfg.perclos_warn] },
];

function buildPanel() {
  const host = $("readings");
  for (const r of READINGS) {
    const row = document.createElement("div");
    row.className = "reading";
    row.innerHTML = `<dt></dt><dd>–</dd><div class="meter"><div class="meter-fill"></div></div>`;
    row.querySelector("dt").textContent = r.label;
    const meter = row.querySelector(".meter");
    const pos = (v) => `${((v - r.lo) / (r.hi - r.lo)) * 100}%`;
    if (r.lo < 0) meter.insertAdjacentHTML("beforeend", `<span class="meter-zero" style="left:${pos(0)}"></span>`);
    for (const m of r.marks || []) meter.insertAdjacentHTML("beforeend", `<span class="meter-mark" style="left:${pos(m)}"></span>`);
    r.row = row;
    r.dd = row.querySelector("dd");
    r.fill = row.querySelector(".meter-fill");
    r.pos = (v) => Math.max(0, Math.min(1, (v - r.lo) / (r.hi - r.lo)));
    host.appendChild(row);
  }
}

function setReading(key, value, hot = false, alert = false) {
  const r = READINGS.find((x) => x.key === key);
  if (value === null) {
    r.dd.textContent = "–";
    r.fill.style.left = r.fill.style.width = "0";
    return;
  }
  r.dd.textContent = r.fmt(value);
  const p = r.pos(value);
  if (r.lo < 0) {
    const z = r.pos(0);
    r.fill.style.left = `${Math.min(p, z) * 100}%`;
    r.fill.style.width = `${Math.abs(p - z) * 100}%`;
  } else {
    r.fill.style.left = "0";
    r.fill.style.width = `${p * 100}%`;
  }
  r.row.classList.toggle("is-hot", hot);
  r.row.classList.toggle("is-alert", alert);
}

const LABEL = { 0: "Attentive", 1: "Caution", 2: "Alert" };

function renderPanel(r, s, ph) {
  const learning = s.calibrating && s.level === Level.OK; // alerts take over the status while calibrating
  const level = learning ? -1 : s.level;
  $("status-dot").dataset.level = level;
  stage.dataset.level = Math.max(0, level);
  $("status-label").textContent = learning ? "Calibrating" : LABEL[s.level];
  $("status-sub").textContent = learning
    ? (s.calibrationHint || "Hold your normal driving pose")
    : s.reasons.length ? s.reasons.join(" · ") : "Eyes on the road";
  $("calib").hidden = !s.calibrating;
  $("calib-fill").style.width = `${s.calibrationProgress * 100}%`;

  setReading("ear", r ? r.ear : null, s.eyesClosed, s.closedFor >= cfg.microsleep_s);
  setReading("jaw", r && r.jawOpen !== null ? r.jawOpen : null, s.yawning);
  setReading("yaw", r ? s.yaw : null, Math.abs(s.yaw) > cfg.yaw_limit_deg);
  setReading("pitch", r ? s.pitch : null, s.pitch < -cfg.pitch_down_limit_deg || s.pitch > cfg.pitch_up_limit_deg);
  setReading("gaze", r ? s.gaze : null, Math.abs(s.gaze) > cfg.gaze_limit);
  setReading("perclos", s.perclos, s.perclos >= cfg.perclos_warn);

  $("yawns").textContent = s.yawns;
  $("phone").textContent = phoneBroken ? "Off" : ph && ph.phones.length ? "Yes" : "No";
  $("phone").title = phoneBroken ? "Phone detection off" : "";
  $("fps").textContent = fps ? Math.round(fps) : "–";

  const banner = $("banner");
  if (s.level > Level.OK && s.reasons.length) {
    const cue = s.reasons.map(cueFor).find(Boolean);
    const headline = cue ? cue.headline : s.reasons[0];
    let detail = s.reasons.join("  ·  ");
    if (mode === "demo" && s.reasons.some((x) => x.startsWith("Fatigue: 3 yawns"))) detail += "  ·  Demo: three yawns in seconds. In real driving the window is 5 minutes.";
    if (banner.hidden || banner.dataset.level !== String(s.level)) banner.dataset.level = s.level;
    if ($("banner-title").textContent !== headline) $("banner-title").textContent = headline;
    if ($("banner-detail").textContent !== detail) $("banner-detail").textContent = detail;
    banner.hidden = false;
    const tag = s.level === Level.ALERT ? "⚠ " : "";
    if (document.hidden || s.level === Level.ALERT) document.title = `${tag}${headline} · Driver Monitor`;
  } else {
    banner.hidden = true;
    document.title = "Driver Monitor";
  }

  if (mode === "demo") {
    const t = video.currentTime;
    $("clock").textContent = clock(t);
    chapters.forEach((ch, i) => {
      const frac = Math.max(0, Math.min(1, (t - ch.start) / (ch.end - ch.start)));
      ch.el.querySelector(".chapter-fill").style.width = `${frac * 100}%`;
      ch.el.classList.toggle("is-current", i === chapterIdx);
    });
  }
}

const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

// ---- event log --------------------------------------------------------------

function logState(s, t) {
  if (s.calibrating && s.level === Level.OK) return;
  const key = `${s.level}|${s.reasons.map((r) => r.replace(/ [\d.]+s?%?$/, "")).join("|")}`;
  if (key === lastKey) return;
  const first = lastKey === "";
  lastKey = key;
  if (first && s.level === Level.OK) return logLine(t, "Calibrated. Watching.", 0);
  logLine(t, s.level === Level.OK ? "Back to attentive" : s.reasons.join(" · "), s.level);
}

function logLine(t, text, level) {
  const log = $("log");
  log.querySelector(".log-empty")?.remove();
  const li = document.createElement("li");
  li.innerHTML = `<time></time><span class="lvl-${level}"></span>`;
  li.querySelector("time").textContent = `${t.toFixed(1)}s`;
  li.querySelector("span").textContent = text;
  log.prepend(li);
  while (log.children.length > 40) log.lastChild.remove();
}

function logDivider(label) {
  logLine(mode === "demo" ? video.currentTime : 0, `New scenario: ${label}`, 0);
}

// ---- wiring -----------------------------------------------------------------

document.querySelectorAll(".mode").forEach((b) => {
  b.addEventListener("click", () => {
    if (b.dataset.mode === mode) return;
    document.querySelectorAll(".mode").forEach((x) => {
      const on = x === b;
      x.classList.toggle("is-active", on);
      x.setAttribute("aria-selected", String(on));
    });
    if (b.dataset.mode === "camera") startCamera();
    else startDemo();
  });
});

$("play").addEventListener("click", () => (video.paused ? playDemo() : video.pause()));
$("sound").addEventListener("click", () => setSound(!soundOn));
$("alarm").addEventListener("click", () => setAlarm(!alarmOn));
video.addEventListener("play", () => $("play").classList.add("is-playing"));
video.addEventListener("pause", () => $("play").classList.remove("is-playing"));
$("recal").addEventListener("click", () => {
  resetState();
  logLine(mode === "demo" ? video.currentTime : (performance.now() - camT0) / 1000, "Recalibrating", 0);
});
window.addEventListener("keydown", (e) => {
  if (e.key === "c" && !e.metaKey && !e.ctrlKey) $("recal").click();
  if (e.key === " " && mode === "demo" && e.target === document.body) {
    e.preventDefault();
    $("play").click();
  }
});

buildPanel();
setSound(soundOn);
setAlarm(alarmOn);
function boot() {
  return Promise.all([loadModels(), loadChapters(), loadVoice()]);
}
function start() {
  // The page is usable at once: the poster and Play button show while the models download.
  const wanted = new URLSearchParams(location.search).get("mode");
  if (!mode || mode === "demo") {
    if (wanted === "camera") document.querySelector('[data-mode="camera"]').click();
    else startDemo();
  }
  boot()
  .then(() => {
    ready = true;
    if (!running) {
      running = true;
      requestAnimationFrame(tick);
    }
    if (mode === "demo" && !video.paused && video.currentTime > 0.3) {
      // The reel started before the models were ready: rewind so calibration sees the calm opener.
      video.currentTime = 0;
      resetState();
      chapterIdx = -1;
    }
  })
  .catch((err) => {
    console.error(err);
    const slow = err && err.message === "timeout";
    veil({
      text: slow ? "Loading is taking too long. Your connection may be slow." : "The models couldn't load.",
      action: { label: "Try again", run: start },
    });
  });
}
start();
