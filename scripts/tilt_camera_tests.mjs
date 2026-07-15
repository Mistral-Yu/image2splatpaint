import assert from "node:assert/strict";
import {
  FIBONACCI_HEMISPHERE_POSE_COUNT,
  TILT_FOV_DEGREES,
  TILT_MAX_ANGLE_DEGREES,
  cameraDiagnostics,
  clampOrbitAngles,
  fibonacciHemispherePoses,
  fitOrbitRadius,
  frameWorldCorners,
  orbitCameraPose,
  projectWorldPoint,
} from "../web/tilt-camera.mjs";

const frame = { x: 1, y: 0.5625 };
const viewport = { width: 1280, height: 720 };
const poses = fibonacciHemispherePoses();
const radius = fitOrbitRadius(frame, viewport, { poses });
const angles = [[0, 0], [5, 0], [-5, 0], [0, 15], [0, -15], [30, 0], [-30, 0], [75, 0], [0, -75]];

assert.equal(TILT_MAX_ANGLE_DEGREES, 75);
assert.equal(FIBONACCI_HEMISPHERE_POSE_COUNT, 49);
assert.equal(poses.length, 49);
assert.equal(poses[0].polarDegrees, 0);
assert.ok(Math.abs(poses.at(-1).polarDegrees - 75) < 1e-10);
assert.equal(new Set(poses.map((pose) => pose.id)).size, poses.length);
assert.ok(Math.abs(clampOrbitAngles(75, 75).polarDegrees - 75) < 1e-10);
for (const pose of poses) {
  assert.ok(pose.polarDegrees >= 0 && pose.polarDegrees <= 75 + 1e-10);
  assert.ok(pose.direction[2] > 0, `pose ${pose.index} left the front hemisphere`);
  assert.ok(Math.abs(Math.hypot(...pose.direction) - 1) < 1e-10);
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

console.log(JSON.stringify({ ok: true, radius, fovDegrees: TILT_FOV_DEGREES, angles, fibonacciPoses: poses.length, maxPolarDegrees: poses.at(-1).polarDegrees }, null, 2));
