import { FaceLandmarker, FilesetResolver, HandLandmarker, ObjectDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import {
  DEFAULTS, DriverState, Level, LEFT_EYE_CONTOUR, LEFT_IRIS, MOUTH_CONTOUR, RIGHT_EYE_CONTOUR, RIGHT_IRIS,
  VoicePolicy, cueFor, handAtEar, phoneNearFace, readFace,
} from "./engine.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODELS = "https://storage.googleapis.com/mediapipe-models";
const PHONE_EVERY = 4; // run the heavier phone/hand models on every Nth frame

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const stage = $("stage");

const cfg = DEFAULTS;
let state = new DriverState(cfg);
let face = null, objects = null, hands = null;
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
const voice = new VoicePolicy();
let soundOn = readSoundPref();
let view = readPref("dm-view", "light"); // "light": outlines only, "mesh": every landmark as a dot

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
function say(cue, urgent) {
  if (!soundOn) return;
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

function hush() {
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
  btn.setAttribute("aria-label", on ? "Mute alerts and reel sound" : "Turn sound on");
  btn.title = on ? "Sound on" : "Sound off";
}

// ---- models -----------------------------------------------------------------

async function loadModels() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const delegate = "GPU";
  face = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${MODELS}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`, delegate },
    runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true,
  });
  // Phone evidence is optional and loads in the background: the face pipeline starts without it.
  loadPhoneModels(fileset, delegate);
}

async function loadPhoneModels(fileset, delegate) {
  try {
    [objects, hands] = await Promise.all([
      ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${MODELS}/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite`, delegate },
        runningMode: "VIDEO", scoreThreshold: 0.35, categoryAllowlist: ["cell phone"], maxResults: 3,
      }),
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${MODELS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`, delegate },
        runningMode: "VIDEO", numHands: 2,
      }),
    ]);
  } catch (err) {
    console.warn("Phone detection unavailable", err);
  }
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
  veil();
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

function tick() {
  requestAnimationFrame(tick);
  if (!face || video.readyState < 2 || video.paused || video.ended) return;
  if (video.videoWidth === 0) return;

  const now = performance.now();
  let t;
  if (mode === "demo") {
    t = video.currentTime;
    if (t === lastVideoT) return; // no new frame yet
    const idx = chapterAt(t);
    // New scenario, or the reel looped / was scrubbed backwards: a new driver, so recalibrate.
    if (idx !== chapterIdx || t < lastVideoT - 0.5) {
      chapterIdx = idx;
      resetState();
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
  const reading = readFace(face.detectForVideo(video, now), w, h);

  frameNo++;
  if (objects && hands && frameNo % PHONE_EVERY === 0) {
    const det = objects.detectForVideo(video, now);
    const hr = hands.detectForVideo(video, now);
    phone = {
      t,
      phones: (det.detections || []).map((d) => ({
        box: [d.boundingBox.originX, d.boundingBox.originY, d.boundingBox.originX + d.boundingBox.width, d.boundingBox.originY + d.boundingBox.height],
        score: d.categories[0]?.score || 0,
      })),
      hands: (hr.landmarks || []).map((lm) => lm.map((p) => [p.x * w, p.y * h])),
    };
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

  const dt = now - lastFrameAt;
  lastFrameAt = now;
  if (dt > 0 && dt < 1000) fps = fps ? fps * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
  // On a slow device the reel would outrun the models; slow it down instead of skipping evidence.
  if (mode === "demo" && fps) video.playbackRate = Math.max(0.35, Math.min(1, fps / 22));

  draw(reading, s, fresh ? phone : null);
  logState(s, t);
  if (!s.calibrating) {
    const cue = voice.update(s.reasons, t, s.yawns);
    if (cue) say(cue, s.level === Level.ALERT);
  }
  if (now - lastUi > 66) {
    lastUi = now;
    renderPanel(reading, s, fresh ? phone : null);
  }
}

// ---- drawing ----------------------------------------------------------------

function readPref(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}

function setView(v) {
  view = v;
  for (const b of document.querySelectorAll(".view-toggle [data-view]")) b.setAttribute("aria-checked", String(b.dataset.view === v));
  try { localStorage.setItem("dm-view", v); } catch {}
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
  const level = s.calibrating ? -1 : s.level;
  $("status-dot").dataset.level = level;
  stage.dataset.level = Math.max(0, level);
  $("status-label").textContent = s.calibrating ? "Calibrating" : LABEL[s.level];
  $("status-sub").textContent = s.calibrating
    ? (r ? "Look at the road as you normally would" : "Looking for a face")
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
  $("phone").textContent = ph && ph.phones.length ? "Yes" : "No";
  $("fps").textContent = fps ? Math.round(fps) : "–";

  const banner = $("banner");
  if (!s.calibrating && s.level > Level.OK && s.reasons.length) {
    const cue = s.reasons.map(cueFor).find(Boolean);
    const headline = cue ? cue.headline : s.reasons[0];
    const detail = s.reasons.join("  ·  ");
    if (banner.hidden || banner.dataset.level !== String(s.level)) banner.dataset.level = s.level;
    if ($("banner-title").textContent !== headline) $("banner-title").textContent = headline;
    if ($("banner-detail").textContent !== detail) $("banner-detail").textContent = detail;
    banner.hidden = false;
  } else {
    banner.hidden = true;
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
  if (s.calibrating) return;
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
veil({ text: "Loading the face model, about 4 MB", busy: true });
Promise.all([loadModels(), loadChapters(), loadVoice()])
  .then(() => {
    const wanted = new URLSearchParams(location.search).get("mode");
    if (wanted === "camera") document.querySelector('[data-mode="camera"]').click();
    else startDemo();
    if (!running) {
      running = true;
      requestAnimationFrame(tick);
    }
  })
  .catch((err) => {
    console.error(err);
    veil({ text: "The models couldn't load. Check your connection and reload the page." });
  });
