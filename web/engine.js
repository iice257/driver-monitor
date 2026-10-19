// Browser port of driver_monitor/metrics.py and driver_monitor/state.py.
// Same signals, same thresholds (config.toml defaults), same temporal rules.

export const Level = { OK: 0, WARN: 1, ALERT: 2 };

export const DEFAULTS = {
  smoothing: 0.35,
  calib_good_s: 2.0,
  calib_window_s: 8.0,
  calib_ear_min: 0.2,
  calib_blink_max: 0.3,
  calib_jaw_max: 0.2,
  calib_pose_deg: 15.0,
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
  yawn_squint_s: 2.0,
  eye_pose_yaw_deg: 30.0,
  eye_pose_pitch_deg: 25.0,
  yaw_limit_deg: 25.0,
  pitch_down_limit_deg: 15.0,
  pitch_up_limit_deg: 25.0,
  gaze_limit: 0.45,
  hysteresis: 0.75,
  distraction_s: 2.0,
  no_face_s: 2.0,
  no_face_alert_s: 5.0,
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
    this.calibLast = null;
    this.calib = []; // good frames: [t, dt, frame]
    this.calibGood = 0;
    this.calibHint = null;
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

  /** Why this frame can't teach us what an attentive driver looks like, or null if it can. */
  calibProblem(f) {
    const c = this.cfg;
    if (!f.face) return "Face the camera";
    const blink = f.blink ?? null;
    if (!(f.ear > c.calib_ear_min || (blink !== null && blink < c.calib_blink_max))) return "Open your eyes";
    const jaw = f.jawOpen ?? null;
    if (jaw !== null ? jaw > c.calib_jaw_max : f.mar > c.yawn_mar) return "Close your mouth";
    if (Math.abs(f.yaw) > c.calib_pose_deg || Math.abs(f.pitch) > c.calib_pose_deg) return "Look at the road";
    return null;
  }

  calibrate(f) {
    const c = this.cfg;
    const dt = this.calibLast === null ? 0 : Math.min(Math.max(f.t - this.calibLast, 0), 0.25);
    this.calibLast = f.t;
    this.calibHint = this.calibProblem(f);
    if (this.calibHint === null) this.calib.push([f.t, dt, f]);
    while (this.calib.length && f.t - this.calib[0][0] > c.calib_window_s) this.calib.shift();
    this.calibGood = this.calib.reduce((sum, x) => sum + x[1], 0);
    if (this.calibGood < c.calib_good_s || this.calib.length < 10) return true;
    const frames = this.calib.map((x) => x[2]);
    const ears = frames.map((x) => x.ear).sort((a, b) => a - b);
    this.baselineEar = median(ears.slice(ears.length >> 1));
    this.baselinePitch = median(frames.map((x) => x.pitch));
    this.baselineYaw = median(frames.map((x) => x.yaw));
    this.baselineGaze = median(frames.map((x) => x.gaze));
    this.calibHint = null;
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

  attention(f) {
    const c = this.cfg;
    if (!f.face) return [0, 0, 0, this.away.update(false, f.t)];
    const dYaw = this.yawE.push(f.yaw) - this.baselineYaw;
    const dPitch = this.pitchE.push(f.pitch) - this.baselinePitch;
    const dGaze = this.gazeE.push(f.gaze) - this.baselineGaze;
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
    // Only a real yawn (jaw past the yawn threshold) earns the squint exemption, and only for
    // yawn_squint_s: a slack jaw with shut eyes is a driver falling asleep.
    const squint = mouthOpen && openFor < c.yawn_squint_s;

    // With the head turned or tipped far from the camera EAR is unreliable: blink blendshape only.
    const thr = this.threshold();
    const blinkClosed = f.blink != null && f.blink > c.blink_score;
    const shut = Math.abs(f.yaw) > c.eye_pose_yaw_deg || f.pitch < -c.eye_pose_pitch_deg ? blinkClosed : f.ear < thr || blinkClosed;
    const closed = f.face && !squint && shut;
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

    const [dYaw, dPitch, dGaze, awayFor] = this.attention(f);

    const phoneFor = this.phone.update(!!f.phoneNearFace, f.t);
    const phoneDownFor = this.phoneDown.update(!!f.phone && dPitch < -c.pitch_down_limit_deg * 0.6, f.t);
    const earHandFor = this.earHand.update(!!f.handAtEar, f.t);

    let level = Level.OK;
    const reasons = [];
    const raise = (lvl, why) => {
      level = Math.max(level, lvl);
      reasons.push(why);
    };

    // Alerts are live from the first frame: until the baseline is learned they run on the textbook
    // thresholds, so a driver who is already asleep when the monitor starts is not ignored.
    if (closedFor >= c.microsleep_s) raise(Level.ALERT, `Eyes closed ${closedFor.toFixed(1)}s`);
    if (phoneFor >= c.phone_s || phoneDownFor >= c.phone_down_s) raise(Level.ALERT, "Phone in use");
    else if (earHandFor >= c.hand_at_ear_s) raise(Level.ALERT, "Phone call");
    if (awayFor >= c.distraction_s && !reasons.includes("Phone in use")) raise(Level.ALERT, "Eyes off road");
    if (missingFor >= c.no_face_alert_s) raise(Level.ALERT, "Driver not visible");
    else if (missingFor >= c.no_face_s) raise(Level.WARN, "Driver not visible");
    if (perclos >= c.perclos_warn) raise(Level.WARN, `Fatigue: PERCLOS ${Math.round(perclos * 100)}%`);
    if (this.yawns.length >= c.yawn_warn_count) raise(Level.WARN, `Fatigue: ${this.yawns.length} yawns`);
    else if (recentYawn) raise(Level.WARN, "Yawn");

    return {
      level, reasons, calibrating,
      eyesClosed: closed, earThreshold: thr, perclos,
      yawns: this.yawns.length, closedFor, awayFor,
      yaw: dYaw, pitch: dPitch, gaze: dGaze, yawning,
      calibrationProgress: calibrating ? Math.min(1, this.calibGood / c.calib_good_s) : 1,
      calibrationHint: calibrating ? this.calibHint : null,
    };
  }
}

// ---- spoken alerts (port of driver_monitor/alerts.py) -------------------------

// Matched by prefix against DriverState reasons, most specific first.
export const CUES = [
  ["Eyes closed", { key: "eyes", headline: "Eyes closed", phrase: "Eyes closed, please stay awake.", repeatS: 4 }],
  ["Phone call", { key: "call", headline: "Phone call detected", phrase: "Phone call detected, please keep your eyes on the road.", repeatS: 8 }],
  ["Phone in use", { key: "phone", headline: "Phone detected", phrase: "Phone detected, please put your phone away.", repeatS: 8 }],
  ["Eyes off road", { key: "away", headline: "Driver distracted", phrase: "Driver distracted, please focus on the road.", repeatS: 5 }],
  ["Driver not visible", { key: "absent", headline: "Driver not visible", phrase: "Driver not visible, please pay attention.", repeatS: 10 }],
  ["Fatigue", { key: "fatigue", headline: "Fatigue detected", phrase: "Fatigue detected, please take a break soon.", repeatS: 30 }],
  ["Yawn", { key: "yawn", headline: "Yawning detected", phrase: "Yawning detected, please drive carefully.", repeatS: 30 }],
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

/** VoicePolicy plus the safety rules: ALERT cues pre-empt; closed eyes escalate to a siren. Mirrors alerts.py. */
export class AlarmPolicy {
  constructor({ sirenAfterS = 2.0, sirenStartHz = 2.0, sirenMaxHz = 6.0, sirenRampHz = 0.5 } = {}) {
    Object.assign(this, { sirenAfterS, sirenStartHz, sirenMaxHz, sirenRampHz });
    this.voice = new VoicePolicy();
    this.eyesSince = null;
  }

  reset() {
    this.voice.reset();
    this.eyesSince = null;
  }

  update(reasons, t, urgent, yawns = null) {
    const eyes = reasons.some((r) => r.startsWith("Eyes closed"));
    if (!eyes) this.eyesSince = null;
    else if (this.eyesSince === null) this.eyesSince = t;
    if (this.eyesSince !== null && t - this.eyesSince >= this.sirenAfterS) {
      const hz = Math.min(this.sirenMaxHz, this.sirenStartHz + this.sirenRampHz * (t - this.eyesSince - this.sirenAfterS));
      this.voice.update(reasons.filter((r) => !r.startsWith("Eyes closed")), t, yawns);
      return { cue: null, preempt: true, sirenHz: hz };
    }
    const cue = this.voice.update(reasons, t, yawns);
    return { cue, preempt: Boolean(cue) && urgent, sirenHz: 0 };
  }
}
