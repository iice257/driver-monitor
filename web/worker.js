// Face, hand and phone inference, off the main thread so the page never janks.
// A classic worker: MediaPipe's WASM loader needs importScripts, which module workers lack.
//
// Everything is self-hosted and pinned: the tasks-vision runtime (0.10.14) under /vendor and the
// models under /models. These are the same model files the desktop app uses (see the README).
const VERSION = "0.10.14";
const VENDOR = `/vendor/tasks-vision-${VERSION}`;
const MODEL_DIR = "/models";

// Sizes are what the progress bar counts against (the server may compress, so Content-Length lies);
// tests/test_web_assets.py keeps them equal to the files on disk.
const ASSETS = {
  bundle: { url: `${VENDOR}/vision_bundle.cjs`, bytes: 137025, label: "Runtime" },
  wasmSimd: { url: `${VENDOR}/wasm/vision_wasm_internal.wasm`, bytes: 9423986, label: "Runtime" },
  wasmPlain: { url: `${VENDOR}/wasm/vision_wasm_nosimd_internal.wasm`, bytes: 9294247, label: "Runtime" },
  face: { url: `${MODEL_DIR}/face_landmarker.task`, bytes: 3758596, label: "Face model" },
  objects: { url: `${MODEL_DIR}/efficientdet_lite0.tflite`, bytes: 7254339, label: "Phone model" },
  hands: { url: `${MODEL_DIR}/hand_landmarker.task`, bytes: 7819105, label: "Hand model" },
};
const GPU_TIMEOUT_MS = 3000;

let FaceLandmarker, FilesetResolver, HandLandmarker, ObjectDetector;
let face = null, objects = null, hands = null;
let role = "";
let delegateUsed = "GPU";

// Download with byte progress. Returns the bytes; the browser's HTTP cache makes repeat visits instant.
async function fetchBytes(key) {
  const a = ASSETS[key];
  const res = await fetch(a.url);
  if (!res.ok) throw new Error(`${a.label}: ${a.url} returned ${res.status}`);
  const parts = [];
  let loaded = 0, last = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - last > 80) {
      last = now;
      postMessage({ type: "progress", role, key, label: a.label, loaded: Math.min(loaded, a.bytes), total: a.bytes });
    }
  }
  postMessage({ type: "progress", role, key, label: a.label, loaded: a.bytes, total: a.bytes });
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

const simd = () => WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));

// The bundle is a CommonJS file; fetch it and run it directly instead of importScripts.
async function loadRuntime() {
  const src = new TextDecoder().decode(await fetchBytes("bundle"));
  const mod = { exports: {} };
  new Function("module", "exports", src)(mod, mod.exports);
  ({ FaceLandmarker, FilesetResolver, HandLandmarker, ObjectDetector } = mod.exports);
  await fetchBytes(simd() ? "wasmSimd" : "wasmPlain"); // lands in the HTTP cache; the loader's own fetch is then instant
  return FilesetResolver.forVisionTasks(`${VENDOR}/wasm`);
}

const withTimeout = (p, ms, what) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${what} took longer than ${ms / 1000} s`)), ms))]);

// Try the GPU first; if it fails or stalls past GPU_TIMEOUT_MS, build the same task on the CPU.
async function create(build, warm) {
  const blank = new OffscreenCanvas(640, 360);
  blank.getContext("2d").fillRect(0, 0, 640, 360);
  const attempt = async (delegate) => {
    const task = await build(delegate);
    warm(task, blank); // the first inference compiles the graphs: do it before the video plays
    return task;
  };
  try {
    const task = await withTimeout(attempt("GPU"), GPU_TIMEOUT_MS, "GPU start-up");
    delegateUsed = "GPU";
    return task;
  } catch (err) {
    console.warn("GPU unavailable, using the CPU:", err && err.message);
    delegateUsed = "CPU";
    return attempt("CPU");
  }
}

// Two workers run this file: "face" (every frame) and "phone" (hands and phone, every few
// frames). Phone inference takes ~100 ms, so sharing a thread would stall the face pipeline.
async function init(which) {
  role = which;
  const keys = ["bundle", simd() ? "wasmSimd" : "wasmPlain", ...(role === "face" ? ["face"] : ["objects", "hands"])];
  postMessage({ type: "plan", role, assets: keys.map((k) => ({ key: k, label: ASSETS[k].label, bytes: ASSETS[k].bytes })) });
  const modelP = role === "face" ? fetchBytes("face") : null; // downloads alongside the runtime, not after it
  const fileset = await loadRuntime();
  if (role === "face") {
    const model = await modelP;
    face = await create(
      (delegate) => FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: model, delegate },
        runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true,
      }),
      (task, blank) => task.detectForVideo(blank, 1),
    );
    postMessage({ type: "ready", delegate: delegateUsed });
    return;
  }
  const [objBytes, handBytes] = await Promise.all([fetchBytes("objects"), fetchBytes("hands")]);
  objects = await create(
    (delegate) => ObjectDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: objBytes, delegate },
      runningMode: "VIDEO", scoreThreshold: 0.35, categoryAllowlist: ["cell phone"], maxResults: 3,
    }),
    (task, blank) => task.detectForVideo(blank, 1),
  );
  hands = await create(
    (delegate) => HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: handBytes, delegate },
      runningMode: "VIDEO", numHands: 2,
    }),
    (task, blank) => task.detectForVideo(blank, 1),
  );
  postMessage({ type: "phone-ready", delegate: delegateUsed });
}

let phoneDead = false;
function phoneFailed(err) {
  if (phoneDead) return;
  phoneDead = true;
  console.warn("Phone detection off", err);
  postMessage({ type: "phone-error", message: String((err && err.message) || err) });
}

function analyse({ bitmap, ts, id, role: kind }) {
  const t0 = performance.now();
  const fr = kind === "face" ? face.detectForVideo(bitmap, ts) : {};
  const t1 = performance.now();
  let found = null;
  if (fr.faceLandmarks?.length) {
    const lm = fr.faceLandmarks[0];
    const xyz = new Float32Array(lm.length * 3);
    lm.forEach((p, i) => { xyz[i * 3] = p.x; xyz[i * 3 + 1] = p.y; xyz[i * 3 + 2] = p.z; });
    const shapes = {};
    for (const c of fr.faceBlendshapes?.[0]?.categories || []) {
      if (c.categoryName === "eyeBlinkLeft" || c.categoryName === "eyeBlinkRight" || c.categoryName === "jawOpen") shapes[c.categoryName] = c.score;
    }
    found = { xyz, shapes };
  }
  let ph = null;
  if (kind === "phone" && objects && hands && !phoneDead) {
    try {
      const det = objects.detectForVideo(bitmap, ts);
      const hr = hands.detectForVideo(bitmap, ts);
      ph = {
        phones: (det.detections || []).map((d) => ({
          box: [d.boundingBox.originX, d.boundingBox.originY, d.boundingBox.originX + d.boundingBox.width, d.boundingBox.originY + d.boundingBox.height],
          score: d.categories[0]?.score || 0,
        })),
        hands: (hr.landmarks || []).map((l) => l.map((p) => [p.x, p.y])),
      };
    } catch (err) {
      phoneFailed(err);
    }
  }
  const bw = bitmap.width;
  bitmap.close();
  postMessage({ type: "result", id, face: found, ph, bw, ms: t1 - t0, msAll: performance.now() - t0 }, found ? [found.xyz.buffer] : []);
}

const fail = (err) => {
  const message = String((err && err.message) || err);
  if (role === "phone") phoneFailed(err);
  else postMessage({ type: "error", message });
};

self.onmessage = (e) => {
  const d = e.data;
  if (d.type === "init") init(d.role).catch(fail);
  else if (d.type === "frame") {
    try { analyse(d); } catch (err) { fail(err); }
  }
};
