// Face, hand and phone inference, off the main thread so the page never janks.
// A classic worker: MediaPipe's WASM loader needs importScripts, which module workers lack.
const BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODELS = "https://storage.googleapis.com/mediapipe-models";

let FaceLandmarker, FilesetResolver, HandLandmarker, ObjectDetector;
let face = null, objects = null, hands = null;

// jsDelivr serves .cjs with a MIME type importScripts refuses, so the bundle is fetched and run directly.
async function loadBundle() {
  const src = await (await fetch(`${BASE}/vision_bundle.cjs`)).text();
  const mod = { exports: {} };
  new Function("module", "exports", src)(mod, mod.exports);
  ({ FaceLandmarker, FilesetResolver, HandLandmarker, ObjectDetector } = mod.exports);
}

// Two workers run this file: "face" (every frame) and "phone" (hands and phone, every few
// frames). Phone inference takes ~100 ms, so sharing a thread would stall the face pipeline.
async function init(role) {
  await loadBundle();
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}/wasm`);
  const delegate = "GPU";
  const blank = new OffscreenCanvas(640, 360);
  blank.getContext("2d").fillRect(0, 0, 640, 360);
  if (role === "face") {
    face = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: new URL("/models/face_landmarker.task", self.location.origin).href, delegate },
      runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true,
    });
    face.detectForVideo(blank, 1); // the first inference compiles the GPU graphs: do it before the video plays
    postMessage({ type: "ready" });
    return;
  }
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
    objects.detectForVideo(blank, 1);
    hands.detectForVideo(blank, 1);
    postMessage({ type: "phone-ready" });
  } catch (err) {
    phoneFailed(err);
  }
}

let phoneDead = false;
function phoneFailed(err) {
  if (phoneDead) return;
  phoneDead = true;
  console.warn("Phone detection off", err);
  postMessage({ type: "phone-error", message: String((err && err.message) || err) });
}

function analyse({ bitmap, ts, id, role }) {
  const t0 = performance.now();
  const fr = role === "face" ? face.detectForVideo(bitmap, ts) : {};
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
  if (role === "phone" && objects && hands && !phoneDead) {
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

self.onmessage = (e) => {
  const d = e.data;
  if (d.type === "init") init(d.role).catch((err) => (d.role === "phone" ? phoneFailed(err) : postMessage({ type: "error", message: String((err && err.message) || err) })));
  else if (d.type === "frame") analyse(d);
};
