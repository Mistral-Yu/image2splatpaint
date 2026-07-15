import assert from "node:assert/strict";
import { inspectIndependentPly } from "./independent_ply_parser.mjs";

const properties = [
  "x", "y", "z", "nx", "ny", "nz",
  "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
  "scale_0", "scale_1", "scale_2",
  "rot_0", "rot_1", "rot_2", "rot_3",
];
const header = [
  "ply",
  "format binary_little_endian 1.0",
  "comment image2gaussianpaint_frame 4 3",
  "comment image2gaussianpaint_blend standard_alpha",
  "comment image2gaussianpaint_layer_order flat_z0 0.01",
  "element vertex 2",
  ...properties.map((name) => `property float ${name}`),
  "end_header",
  "",
].join("\n");
const headerBytes = new TextEncoder().encode(header);
const payload = new ArrayBuffer(2 * properties.length * 4);
const view = new DataView(payload);
for (let vertex = 0; vertex < 2; vertex += 1) {
  const base = vertex * properties.length;
  view.setFloat32((base + 0) * 4, vertex ? 0.5 : -0.5, true);
  view.setFloat32((base + 1) * 4, 0, true);
  view.setFloat32((base + 2) * 4, 0, true);
  view.setFloat32((base + 9) * 4, 0, true);
  view.setFloat32((base + 10) * 4, Math.log(0.1), true);
  view.setFloat32((base + 11) * 4, Math.log(0.05), true);
  view.setFloat32((base + 12) * 4, Math.log(0.0001), true);
  view.setFloat32((base + 13) * 4, 1, true);
}
const fixture = new Uint8Array(headerBytes.length + payload.byteLength);
fixture.set(headerBytes);
fixture.set(new Uint8Array(payload), headerBytes.length);

const report = inspectIndependentPly(fixture);
assert.equal(report.valid, true);
assert.equal(report.vertices, 2);
assert.equal(report.frameAspect, 4 / 3);
assert.equal(report.standardAlpha, true);
assert.equal(report.zContract, true);

const nonPlanar = fixture.slice();
new DataView(nonPlanar.buffer).setFloat32(headerBytes.length + 14 * 4, 0.25, true);
assert.equal(inspectIndependentPly(nonPlanar).valid, false);

console.log(JSON.stringify({ ok: true, report }, null, 2));
