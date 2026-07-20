import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  FIBONACCI_HEMISPHERE_POSE_COUNT,
  TILT_FOV_DEGREES,
  TILT_MAX_ANGLE_DEGREES,
  cameraDiagnostics,
  canonicalOrbitRadius,
  clampOrbitAngles,
  fibonacciHemispherePoses,
  fitOrbitRadius,
  frameWorldCorners,
  inverseProjectPlanarSplatPoint,
  orbitProjectionContract,
  orbitCameraPose,
  projectPlanarSplatPoint,
  projectWorldPoint,
  trainingCameraMarkerGeometry,
} from "../web/tilt-camera.mjs";

const appSource = await readFile(resolve(import.meta.dirname, "../web/app.js"), "utf8");
const htmlSource = await readFile(resolve(import.meta.dirname, "../web/index.html"), "utf8");
const viewerSource = await readFile(resolve(import.meta.dirname, "../web/tilt-viewer.mjs"), "utf8");
const bundleSource = await readFile(resolve(import.meta.dirname, "../web/tilt-viewer.bundle.js"), "utf8");
assert.ok(appSource.includes("Math.round(pitch * 1000) / 1000"));
assert.ok(appSource.includes("Math.round(yaw * 1000) / 1000"));
assert.ok(htmlSource.includes('data-testid="tilt-radius-mode"'));
assert.ok(htmlSource.includes('value="training"'));
assert.ok(htmlSource.includes('value="fit"'));
assert.ok(viewerSource.includes("setRadiusMode(mode)"));
assert.ok(viewerSource.includes('radiusMode === "fit" ? fitAllOrbitRadius : orbitRadius'));
assert.ok(appSource.includes("dataset.tiltRadiusMode"));
assert.ok(htmlSource.includes('<script src="./tilt-viewer.bundle.js?v=camera-summary-v2"></script>'));
assert.ok(appSource.includes("globalThis.Image2SplatPaintTilt"));
assert.ok(!appSource.includes("Tilt rendering requires GitHub Pages or a local HTTP server."));
assert.ok(bundleSource.includes("PlayCanvas Engine v2.20.6"));

const frame = { x: 1, y: 0.5625 };
const viewport = { width: 1280, height: 720 };
const poses = fibonacciHemispherePoses();
const radius = fitOrbitRadius(frame, viewport, { poses });
const canonicalRadius = canonicalOrbitRadius(frame, { poses });
const angles = [[0, 0], [5, 0], [-5, 0], [0, 15], [0, -15], [30, 0], [-30, 0], [75, 0], [0, -75]];

assert.equal(TILT_MAX_ANGLE_DEGREES, 75);
assert.equal(FIBONACCI_HEMISPHERE_POSE_COUNT, 49);
assert.equal(poses.length, 49);
assert.equal(poses[0].polarDegrees, 0);
assert.ok(Math.abs(poses.at(-1).polarDegrees - 75) < 1e-10);
assert.equal(new Set(poses.map((pose) => pose.id)).size, poses.length);
assert.ok(Math.abs(canonicalRadius - radius) < 1e-10);
assert.ok(Math.abs(clampOrbitAngles(75, 75).polarDegrees - 75) < 1e-10);
for (const pose of poses) {
  assert.ok(pose.polarDegrees >= 0 && pose.polarDegrees <= 75 + 1e-10);
  assert.ok(pose.direction[2] > 0, `pose ${pose.index} left the front hemisphere`);
  assert.ok(Math.abs(Math.hypot(...pose.direction) - 1) < 1e-10);
}

for (const [pitch, yaw] of angles) {
  const contract = orbitProjectionContract({ width: 1280, height: 720 }, viewport, pitch, yaw, radius);
  const pose = orbitCameraPose(pitch, yaw, radius);
  for (const z of [-0.005, 0, 0.005]) {
    for (const point of [[-0.8, -0.7], [-0.2, 0.5], [0, 0], [0.75, -0.25]]) {
      const projected = projectPlanarSplatPoint(point, contract, z);
      const world = [contract.frameX * point[0], -contract.frameY * point[1], z];
      const reference = projectWorldPoint(world, pose, viewport, TILT_FOV_DEGREES);
      assert.ok(Math.abs(projected.point[0] - reference.ndcX) < 1e-10);
      assert.ok(Math.abs(projected.point[1] + reference.ndcY) < 1e-10);
      assert.ok(Math.abs(projected.depth - reference.depth) < 1e-10);
      const restored = inverseProjectPlanarSplatPoint(projected.point, contract, z);
      assert.ok(restored.valid);
      assert.ok(Math.hypot(restored.point[0] - point[0], restored.point[1] - point[1]) < 1e-9);
    }
  }
}

for (const [pitch, yaw] of angles) {
  const diagnostics = cameraDiagnostics(frame, viewport, pitch, yaw, radius);
  assert.ok(Math.abs(Math.hypot(...diagnostics.position) - radius) < 1e-10, `radius drift at ${pitch}/${yaw}`);
  assert.ok(Math.abs(diagnostics.center.x - viewport.width / 2) < 1e-8, `center x drift at ${pitch}/${yaw}`);
  assert.ok(Math.abs(diagnostics.center.y - viewport.height / 2) < 1e-8, `center y drift at ${pitch}/${yaw}`);
  assert.ok(diagnostics.corners.every((corner) => corner.valid && Math.abs(corner.ndcX) <= 0.900001 && Math.abs(corner.ndcY) <= 0.900001));
}

for (const pose of poses) {
  const diagnostics = cameraDiagnostics(frame, viewport, pose.pitchDegrees, pose.yawDegrees, radius);
  assert.ok(Math.abs(Math.hypot(...diagnostics.position) - radius) < 1e-10);
  assert.ok(Math.hypot(diagnostics.center.x - viewport.width / 2, diagnostics.center.y - viewport.height / 2) < 1e-8);
  assert.ok(diagnostics.corners.every((corner) => corner.valid && Math.abs(corner.ndcX) <= 0.900001 && Math.abs(corner.ndcY) <= 0.900001));
}

function bounds(corners) {
  return {
    minX: Math.min(...corners.map((corner) => corner.x)),
    maxX: Math.max(...corners.map((corner) => corner.x)),
    minY: Math.min(...corners.map((corner) => corner.y)),
    maxY: Math.max(...corners.map((corner) => corner.y)),
  };
}

const yawPositive = bounds(cameraDiagnostics(frame, viewport, 0, 30, radius).corners);
const yawNegative = bounds(cameraDiagnostics(frame, viewport, 0, -30, radius).corners);
assert.ok(Math.abs((yawPositive.maxX - yawPositive.minX) - (yawNegative.maxX - yawNegative.minX)) < 1e-8);
assert.ok(Math.abs(yawPositive.minX - (viewport.width - yawNegative.maxX)) < 1e-8);

// Orbiting the camera by +yaw is the same relative pose as rotating the plane
// by -yaw in front of a fixed camera at (0, 0, radius).
const yaw = 30 * Math.PI / 180;
const orbitPose = orbitCameraPose(0, 30, radius);
const fixedPose = orbitCameraPose(0, 0, radius);
for (const point of frameWorldCorners(frame)) {
  const orbitProjection = projectWorldPoint(point, orbitPose, viewport, TILT_FOV_DEGREES);
  const rotatedPoint = [
    Math.cos(yaw) * point[0] - Math.sin(yaw) * point[2],
    point[1],
    Math.sin(yaw) * point[0] + Math.cos(yaw) * point[2],
  ];
  const fixedProjection = projectWorldPoint(rotatedPoint, fixedPose, viewport, TILT_FOV_DEGREES);
  assert.ok(Math.hypot(orbitProjection.x - fixedProjection.x, orbitProjection.y - fixedProjection.y) < 1e-8);
  assert.ok(Math.abs(orbitProjection.depth - fixedProjection.depth) < 1e-8);
}

const markerSnapshot = {
  cameras: [
    { id: "front-a", kind: "front", pitch_degrees: 0, yaw_degrees: 0 },
    { id: "front-b", kind: "front", pitch_degrees: 0, yaw_degrees: 0 },
    {
      id: "pitch-plus-5",
      kind: "virtual",
      pitch_degrees: 5,
      yaw_degrees: 0,
      intrinsics: { fov_degrees: 60, fx: 590, fy: 590, cx: 639.5, cy: 359.5 },
    },
    { id: "yaw-minus-15", kind: "virtual", pitch_degrees: 0, yaw_degrees: -15 },
  ],
  active_camera_id: "pitch-plus-5",
  orbit_radius: radius,
  fov_degrees: TILT_FOV_DEGREES,
  target: [0, 0, 0],
  image_aspect: 16 / 9,
};
const markers = trainingCameraMarkerGeometry(markerSnapshot);
assert.equal(markers.length, 3);
assert.equal(markers.find((marker) => marker.kind === "front").multiplicity, 2);
assert.equal(markers.filter((marker) => marker.kind === "front").length, 1);
assert.equal(markers.filter((marker) => marker.kind === "virtual").length, 2);
assert.equal(markers.reduce((total, marker) => total + marker.multiplicity, 0), 4);
assert.equal(markers.filter((marker) => marker.active).length, 1);
assert.equal(markers.find((marker) => marker.active).fovDegrees, 60);
for (const marker of markers) {
  assert.ok(Math.abs(Math.hypot(...marker.position) - radius) < 1e-10);
  assert.ok(Math.hypot(...marker.target) < 1e-12);
  assert.ok(Math.abs(Math.hypot(...marker.forward) - 1) < 1e-10);
  assert.equal(marker.frustumCorners.length, 4);
  for (const corner of marker.frustumCorners) assert.ok(corner.every(Number.isFinite));
}

console.log(JSON.stringify({ ok: true, radius, canonicalRadius, fovDegrees: TILT_FOV_DEGREES, angles, fibonacciPoses: poses.length, maxPolarDegrees: poses.at(-1).polarDegrees }, null, 2));
