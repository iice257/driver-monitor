// Browser port of driver_monitor/metrics.py and driver_monitor/state.py.
// Same signals, same thresholds (config.toml defaults), same temporal rules.

export const Level = { OK: 0, WARN: 1, ALERT: 2 };

export const DEFAULTS = {
  calibration_s: 3.0,
  smoothing: 0.35,
  ear_ratio: 0.72,
  ear_floor: 0.14,
  blink_score: 0.65,
  microsleep_s: 1.2,
  perclos_window_s: 60.0,
  perclos_warn: 0.15,
  perclos_min_s: 20.0,
  yawn_jaw: 0.45,
  yawn_mar: 0.45,
  yawn_min_s: 1.0,
  yawn_notice_s: 3.0,
  yawn_window_s: 300.0,
  yawn_warn_count: 3,
  yawn_squint_s: 6.0,
  yaw_limit_deg: 25.0,
  pitch_down_limit_deg: 15.0,
  pitch_up_limit_deg: 25.0,
  gaze_limit: 0.45,
  hysteresis: 0.75,
  distraction_s: 2.0,
  no_face_s: 2.0,
  phone_s: 1.0,
  phone_down_s: 1.0,
  hand_at_ear_s: 2.0,
  phone_grace_s: 0.6,
};

// ---- metrics ---------------------------------------------------------------

export const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
export const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE_CORNERS = [33, 133];
const LEFT_EYE_CORNERS = [362, 263];
export const RIGHT_IRIS = 468;
export const LEFT_IRIS = 473;
const MOUTH_CORNERS = [61, 291];
const MOUTH_PAIRS = [[81, 178], [13, 14], [311, 402]];
const FOREHEAD = 10;
const CHIN = 152;
export const RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
export const LEFT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382];
export const MOUTH_CONTOUR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const deg = (r) => (r * 180) / Math.PI;

function eyeAspectRatio(p, idx) {
  const [p1, p2, p3, p4, p5, p6] = idx.map((i) => p[i]);
  const w = dist(p1, p4);
  return w < 1e-6 ? 0 : (dist(p2, p6) + dist(p3, p5)) / (2 * w);
}

function mouthAspectRatio(p) {
  const w = dist(p[MOUTH_CORNERS[0]], p[MOUTH_CORNERS[1]]);
  if (w < 1e-6) return 0;
  return MOUTH_PAIRS.reduce((s, [a, b]) => s + dist(p[a], p[b]), 0) / MOUTH_PAIRS.length / w;
}

function headPose(p) {
  const r = [0, 1, 2].map((k) => p[LEFT_EYE_CORNERS[1]][k] - p[RIGHT_EYE_CORNERS[0]][k]);
  const d = [0, 1, 2].map((k) => p[CHIN][k] - p[FOREHEAD][k]);
  const f = [d[1] * r[2] - d[2] * r[1], d[2] * r[0] - d[0] * r[2], d[0] * r[1] - d[1] * r[0]];
  const n = Math.hypot(...f);
  if (n < 1e-9) return { pitch: 0, yaw: 0, roll: 0 };
  const [fx, fy, fz] = f.map((v) => v / n);
  return {
    yaw: deg(Math.atan2(fx, -fz)),
    pitch: deg(Math.atan2(-fy, Math.hypot(fx, fz))),
    roll: deg(Math.atan2(r[1], r[0])),
  };
}

function gazeRatio(p) {
  if (p.length < 478) return 0;
  const vals = [];
  for (const [[a, b], iris] of [[RIGHT_EYE_CORNERS, RIGHT_IRIS], [LEFT_EYE_CORNERS, LEFT_IRIS]]) {
    const ax = [p[b][0] - p[a][0], p[b][1] - p[a][1]];
    const den = ax[0] * ax[0] + ax[1] * ax[1];
    if (den < 1e-6) continue;
    const t = ((p[iris][0] - p[a][0]) * ax[0] + (p[iris][1] - p[a][1]) * ax[1]) / den;
    vals.push((t - 0.5) * 2);
  }
  if (!vals.length) return 0;
  return Math.max(-1, Math.min(1, vals.reduce((s, v) => s + v, 0) / vals.length));
}

export function faceBox(p) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of p) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  return [x0, y0, x1, y1];
}

/** MediaPipe FaceLandmarker result -> reading in pixel space (z scaled like x). */
export function readFace(result, w, h) {
  if (!result.faceLandmarks?.length) return null;
  const pts = result.faceLandmarks[0].map((l) => [l.x * w, l.y * h, l.z * w]);
  const shapes = {};
  for (const c of result.faceBlendshapes?.[0]?.categories || []) shapes[c.categoryName] = c.score;
  const has = Object.keys(shapes).length > 0;
  return {
    points: pts,
    ear: (eyeAspectRatio(pts, RIGHT_EYE) + eyeAspectRatio(pts, LEFT_EYE)) / 2,
    mar: mouthAspectRatio(pts),
    ...headPose(pts),
    gaze: gazeRatio(pts),
    blink: has ? ((shapes.eyeBlinkLeft || 0) + (shapes.eyeBlinkRight || 0)) / 2 : null,
    jawOpen: has ? shapes.jawOpen || 0 : null,
    box: faceBox(pts),
  };
}

// ---- phone heuristics ------------------------------------------------------

export function handAtEar(hand, face) {
  const [x0, y0, x1, y1] = face;
  const fw = x1 - x0, fh = y1 - y0;
  const palm = [0, 5, 9, 13, 17].reduce((s, i) => [s[0] + hand[i][0] / 5, s[1] + hand[i][1] / 5], [0, 0]);
  const beside = (x0 - 0.6 * fw < palm[0] && palm[0] < x0 + 0.25 * fw) || (x1 - 0.25 * fw < palm[0] && palm[0] < x1 + 0.6 * fw);
  const earHeight = y0 + 0.15 * fh < palm[1] && palm[1] < y1 + 0.15 * fh;
  return beside && earHeight;
}

export function phoneNearFace(phones, face) {
  const [x0, y0, x1, y1] = face;
  const pad = 0.5 * (x1 - x0);
  const zone = [x0 - pad, y0, x1 + pad, y1 + pad];
  return phones.some(({ box: b }) => {
    const ix = Math.max(0, Math.min(b[2], zone[2]) - Math.max(b[0], zone[0]));
    const iy = Math.max(0, Math.min(b[3], zone[3]) - Math.max(b[1], zone[1]));
    const area = (b[2] - b[0]) * (b[3] - b[1]);
    return area > 0 && (ix * iy) / area > 0.3;
  });
}

// ---- temporal state --------------------------------------------------------

class Timer {
  constructor(grace = 0) {
    this.grace = grace;
    this.since = null;
    this.lastTrue = null;
  }
  update(active, t) {
    if (active) {
      if (this.since === null) this.since = t;
      this.lastTrue = t;
    } else if (this.since !== null && (this.lastTrue === null || t - this.lastTrue > this.grace)) {
      this.since = this.lastTrue = null;
    }
    return this.since === null ? 0 : t - this.since;
  }
}

class Ema {
  constructor(alpha) {
    this.alpha = alpha;
    this.value = null;
  }
  push(x) {
    this.value = this.value === null ? x : this.value + this.alpha * (x - this.value);
    return this.value;
  }
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export class DriverState {
  constructor(cfg = DEFAULTS) {
    this.cfg = { ...DEFAULTS, ...cfg };
    this.reset();
  }

  reset() {
    const c = this.cfg;
    this.start = null;
    this.calib = [];
    this.baselineEar = null;
    this.baselinePitch = this.baselineYaw = this.baselineGaze = 0;
    this.closed = new Timer();
    this.away = new Timer(0.3);
    this.mouth = new Timer(0.2);
    this.missing = new Timer();
    this.phone = new Timer(c.phone_grace_s);
    this.phoneDown = new Timer(c.phone_grace_s);
    this.earHand = new Timer(c.phone_grace_s);
    this.yawE = new Ema(c.smoothing);
    this.pitchE = new Ema(c.smoothing);
    this.gazeE = new Ema(c.smoothing);
    this.isAway = false;
    this.yawnCounted = false;
    this.closedSamples = [];
    this.yawns = [];
  }

  threshold() {
    if (this.baselineEar === null) return 0.21;
    return Math.max(this.cfg.ear_floor, this.baselineEar * this.cfg.ear_ratio);
  }

  calibrate(f) {
    if (this.start === null) this.start = f.t;
    if (f.face) this.calib.push(f);
    if (f.t - this.start < this.cfg.calibration_s || this.calib.length < 10) return true;
    const ears = this.calib.map((c) => c.ear).sort((a, b) => a - b);
    this.baselineEar = median(ears.slice(ears.length >> 1));
    this.baselinePitch = median(this.calib.map((c) => c.pitch));
    this.baselineYaw = median(this.calib.map((c) => c.yaw));
    this.baselineGaze = median(this.calib.map((c) => c.gaze));
    return false;
  }

  perclos(t, closed) {
    const s = this.closedSamples;
    s.push([t, closed]);
    while (s.length && t - s[0][0] > this.cfg.perclos_window_s) s.shift();
    let total = 0, shut = 0;
    for (let i = 1; i < s.length; i++) {
      const dt = s[i][0] - s[i - 1][0];
      total += dt;
      if (s[i - 1][1]) shut += dt;
    }
    if (total < Math.min(this.cfg.perclos_min_s, this.cfg.perclos_window_s)) return 0;
    return shut / total;
  }

  attention(f, calibrating) {
    const c = this.cfg;
    if (!f.face) return [0, 0, 0, this.away.update(false, f.t)];
    const dYaw = this.yawE.push(f.yaw) - this.baselineYaw;
    const dPitch = this.pitchE.push(f.pitch) - this.baselinePitch;
    const dGaze = this.gazeE.push(f.gaze) - this.baselineGaze;
    if (calibrating) return [dYaw, dPitch, dGaze, 0];
    const k = this.isAway ? c.hysteresis : 1;
    this.isAway =
      Math.abs(dYaw) > c.yaw_limit_deg * k ||
      dPitch < -c.pitch_down_limit_deg * k ||
      dPitch > c.pitch_up_limit_deg * k ||
      Math.abs(dGaze) > c.gaze_limit * k;
    return [dYaw, dPitch, dGaze, this.away.update(this.isAway, f.t)];
  }

  /** f: {t, face, ear, mar, pitch, yaw, gaze, blink, jawOpen, phone, phoneNearFace, handAtEar} */
  update(f) {
    const c = this.cfg;
    const calibrating = this.baselineEar === null && this.calibrate(f);
    const missingFor = this.missing.update(!f.face, f.t);

    const jaw = f.jawOpen ?? null;
    const mouthOpen = f.face && (jaw !== null ? jaw > c.yawn_jaw : f.mar > c.yawn_mar);
    const openFor = this.mouth.update(mouthOpen, f.t);
    const opening = f.face && (jaw !== null ? jaw > c.yawn_jaw * 0.6 : f.mar > c.yawn_mar * 0.6);
    const squint = opening && openFor < c.yawn_squint_s;

    const thr = this.threshold();
    const closed = f.face && !squint && (f.ear < thr || (f.blink !== null && f.blink > c.blink_score));
    const closedFor = this.closed.update(closed, f.t);
    const perclos = this.perclos(f.t, closed);

    const yawning = openFor >= c.yawn_min_s;
    if (yawning && !this.yawnCounted) {
      this.yawns.push(f.t);
      this.yawnCounted = true;
    } else if (openFor === 0) {
      this.yawnCounted = false;
    }
    while (this.yawns.length && f.t - this.yawns[0] > c.yawn_window_s) this.yawns.shift();
    const recentYawn = this.yawns.length > 0 && f.t - this.yawns[this.yawns.length - 1] < c.yawn_notice_s;

    const [dYaw, dPitch, dGaze, awayFor] = this.attention(f, calibrating);

    const phoneFor = this.phone.update(!!f.phoneNearFace, f.t);
    const phoneDownFor = this.phoneDown.update(!!f.phone && dPitch < -c.pitch_down_limit_deg * 0.6, f.t);
    const earHandFor = this.earHand.update(!!f.handAtEar, f.t);

    let level = Level.OK;
    const reasons = [];
    const raise = (lvl, why) => {
      level = Math.max(level, lvl);
      reasons.push(why);
    };

    if (!calibrating) {
      if (closedFor >= c.microsleep_s) raise(Level.ALERT, `Eyes closed ${closedFor.toFixed(1)}s`);
      if (phoneFor >= c.phone_s || phoneDownFor >= c.phone_down_s) raise(Level.ALERT, "Phone in use");
      else if (earHandFor >= c.hand_at_ear_s) raise(Level.ALERT, "Phone call");
      if (awayFor >= c.distraction_s && !reasons.includes("Phone in use")) raise(Level.ALERT, "Eyes off road");
      if (missingFor >= c.no_face_s) raise(Level.WARN, "Driver not visible");
      if (perclos >= c.perclos_warn) raise(Level.WARN, `Fatigue: PERCLOS ${Math.round(perclos * 100)}%`);
      if (this.yawns.length >= c.yawn_warn_count) raise(Level.WARN, `Fatigue: ${this.yawns.length} yawns`);
      else if (recentYawn) raise(Level.WARN, "Yawn");
    }

    return {
      level, reasons, calibrating,
      eyesClosed: closed, earThreshold: thr, perclos,
      yawns: this.yawns.length, closedFor, awayFor,
      yaw: dYaw, pitch: dPitch, gaze: dGaze, yawning,
      calibrationProgress: calibrating && this.start !== null ? Math.min(1, (f.t - this.start) / c.calibration_s) : 1,
    };
  }
}

// ---- spoken alerts (port of driver_monitor/alerts.py) -------------------------

// Matched by prefix against DriverState reasons, most specific first.
export const CUES = [
  ["Eyes closed", { key: "eyes", headline: "Eyes closed", phrase: "Eyes closed. Wake up!", repeatS: 4 }],
  ["Phone call", { key: "call", headline: "Phone call detected", phrase: "Phone call detected", repeatS: 8 }],
  ["Phone in use", { key: "phone", headline: "Phone detected", phrase: "Phone detected", repeatS: 8 }],
  ["Eyes off road", { key: "away", headline: "Eyes off the road", phrase: "Eyes off the road", repeatS: 5 }],
  ["Driver not visible", { key: "absent", headline: "Driver not visible", phrase: "Driver not visible", repeatS: 10 }],
  ["Fatigue", { key: "fatigue", headline: "Fatigue detected", phrase: "Fatigue detected. Take a break", repeatS: 30 }],
  ["Yawn", { key: "yawn", headline: "Yawning detected", phrase: "Yawning detected", repeatS: 30 }],
];

export function cueFor(reason) {
  for (const [prefix, cue] of CUES) if (reason.startsWith(prefix)) return cue;
  return null;
}

// Says a cue when its reason first appears (at most once per minGapS) and again every
// repeatS while it holds; one phrase per update, in priority order. A rising yawn count
// makes each new yawn its own announcement.
export class VoicePolicy {
  constructor(minGapS = 2.5) {
    this.minGapS = minGapS;
    this.reset();
  }

  reset() {
    this.last = new Map();
    this.spoken = new Set();
    this.yawns = 0;
  }

  update(reasons, t, yawns = null) {
    const active = reasons.map(cueFor).filter(Boolean);
    const keys = new Set(active.map((c) => c.key));
    for (const k of [...this.spoken]) if (!keys.has(k)) this.spoken.delete(k);
    if (yawns !== null) {
      if (yawns > this.yawns) this.spoken.delete("yawn");
      this.yawns = yawns;
    }
    for (const cue of active) {
      const last = this.last.get(cue.key);
      const since = last === undefined ? Infinity : t - last;
      if (since >= (this.spoken.has(cue.key) ? cue.repeatS : this.minGapS)) {
        this.last.set(cue.key, t);
        this.spoken.add(cue.key);
        return cue;
      }
    }
    return null;
  }
}
