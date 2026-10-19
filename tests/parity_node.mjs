// Runs web/engine.js over the recorded signal trace and prints [level, reasons] per frame as JSON.
// No dependencies: used by tests/test_parity.py to prove the browser and Python monitors agree.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, "$1")), "..");
const { DriverState, Level } = await import(pathToFileURL(path.join(root, "web", "engine.js")).href);
const traces = JSON.parse(readFileSync(path.join(root, "tests", "fixtures", "signal_trace.json"), "utf8"));

const out = {};
for (const [name, frames] of Object.entries(traces)) {
  const state = new DriverState();
  out[name] = frames.map((f) => {
    const s = state.update({
      t: f.t, face: f.face, ear: f.ear, mar: f.mar, pitch: f.pitch, yaw: f.yaw, gaze: f.gaze,
      blink: f.blink ?? null, jawOpen: f.jaw_open ?? null,
      phone: f.phone, phoneNearFace: f.phone_near_face, handAtEar: f.hand_at_ear,
    });
    return [s.level, s.reasons];
  });
}
console.log(JSON.stringify(out));
