import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const app = await readFile(new URL("../web/app.js", import.meta.url), "utf8");

function functionSource(name) {
  const start = app.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} missing`);
  const parametersStart = app.indexOf("(", start);
  let parameterDepth = 0;
  let brace = -1;
  for (let index = parametersStart; index < app.length; index += 1) {
    if (app[index] === "(") parameterDepth += 1;
    if (app[index] === ")") parameterDepth -= 1;
    if (parameterDepth === 0) {
      brace = app.indexOf("{", index);
      break;
    }
  }
  assert.ok(brace >= 0, `${name} body missing`);
  let depth = 0;
  for (let index = brace; index < app.length; index += 1) {
    if (app[index] === "{") depth += 1;
    if (app[index] === "}") depth -= 1;
    if (depth === 0) return app.slice(start, index + 1);
  }
  throw new Error(`${name} body is incomplete`);
}

const helpers = new Function(`
  const DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE = 4;
  const CURRICULUM_GROWTH_FRACTION = 6 / 7;
  const VIRTUAL_TILT_DIRECTIONS = [[1,0],[-1,0],[0,1],[0,-1]];
  ${functionSource("planarTiltRotation")}
  ${functionSource("projectPlanarPoint")}
  ${functionSource("inverseProjectPlanarPoint")}
  ${functionSource("virtualTiltStepSpec")}
  return { planarTiltRotation, projectPlanarPoint, inverseProjectPlanarPoint, virtualTiltStepSpec };
`)();

for (const frame of [{ width: 1, height: 1 }, { width: 1600, height: 900 }, { width: 900, height: 1600 }]) {
  for (const [pitchDegrees, yawDegrees] of [[0, 0], [2, 0], [-2, 0], [0, 5], [0, -5], [5, -5]]) {
    const pitch = pitchDegrees * Math.PI / 180;
    const yaw = yawDegrees * Math.PI / 180;
    for (const point of [[-1, -1], [-0.4, 0.7], [0, 0], [0.9, -0.2], [1, 1]]) {
      const projected = helpers.projectPlanarPoint(point, pitch, yaw, 4, 0, frame);
      const restored = helpers.inverseProjectPlanarPoint(projected.point, pitch, yaw, 4, frame);
      assert.ok(projected.valid && restored.valid);
      assert.ok(Math.abs(restored.point[0] - point[0]) < 1e-12);
      assert.ok(Math.abs(restored.point[1] - point[1]) < 1e-12);
    }
  }
}

const variants = { enabled: true, interval: 32, weight: 0.25, cameraDistance: 4 };
assert.equal(helpers.virtualTiltStepSpec(31, "mid", 7000, variants).enabled, false);
assert.equal(helpers.virtualTiltStepSpec(32, "coarse", 7000, variants).enabled, false);
assert.deepEqual(
  [32, 64, 96, 128].map((step) => {
    const value = helpers.virtualTiltStepSpec(step, "mid", 7000, variants);
    return [value.pitchDegrees, value.yawDegrees];
  }),
  [[5, 0], [-5, 0], [0, 5], [0, -5]],
);
assert.equal(helpers.virtualTiltStepSpec(4000, "full", 7000, variants).angleDegrees, 15);
assert.equal(helpers.virtualTiltStepSpec(6016, "full", 7000, variants).angleDegrees, 30);

const depthX = Math.sin(5 * Math.PI / 180) * 0.25;
const depthY = 0;
const sigmaDepth = Math.hypot(depthX, depthY);
const supportDepth = 4 * sigmaDepth;
const excess = Math.max(0, supportDepth / 0.01 - 1);
const coefficient = 2 * 0.01 * 0.5 * excess / 0.01;
const scaleGradientX = coefficient * 4 * depthX ** 2 / sigmaDepth;
assert.ok(scaleGradientX > 0);

const contracts = {
  configCapacity: app.includes('add("config", 64 * 4, true)') && app.includes("size: 64 * 4"),
  qaSwitch:
    app.includes('qaOverrides("__image2GaussianTiltTraining")') &&
    app.includes('query.has("virtual-tilt")') &&
    app.includes('query.has("tilt-robust-split")') &&
    app.includes('queryNumber("virtual-order-weight"'),
  stagedSchedule: app.includes('stage !== "coarse"') &&
    app.includes('stage === "mid" ? 5') &&
    app.includes("progress < CURRICULUM_GROWTH_FRACTION ? 15 : 30") &&
    app.includes("step % variants.interval === 0"),
  plyFrameCoordinates: app.includes("let frame = vec2<f32>(cfg(0u), cfg(1u))") &&
    app.includes("-frame.y * point.y") && app.includes("-distance * camera.y / (frame.y * denominator)"),
  inverseHomography: app.includes("fn virtual_inverse_point") && app.includes("let determinant = a00 * a11 - a01 * a10"),
  projectedTileBounds: app.includes("let p0 = virtual_project_point(center + vec2<f32>(-radius.x, -radius.y)") &&
    app.includes("!virtual_tilt_enabled() && !tile_intersects_footprint"),
  cameraDepthSort: app.includes("virtual_camera_depth(xy[a], transform[a].w)") &&
    app.includes("virtual_layer_depth(packedTag)"),
  warpedTarget: app.includes("target_rgb_at(inversePoint.xy") && app.includes("target_color_at(inversePoint.xy"),
  validMask: app.includes("if (inversePoint.z < 0.5)") && app.includes("valid = 1.0"),
  exactBackwardMapping: app.includes('"  let p = inversePoint.xy;"') && app.includes('"    points[i] = inversePoint.xy;"'),
  weightedTiltLoss: app.includes("cfg(61u)") && app.includes("dColor *= viewWeight"),
  scaleOnlyOrderPenalty: app.includes("fn virtual_order_penalty") &&
    app.includes("residual * excess * influence") &&
    app.includes("atomic_add_f32(base + 2u, coefficient * dSupportX)") &&
    app.includes("atomic_add_f32(base + 3u, coefficient * dSupportY)") &&
    !app.slice(app.indexOf("fn virtual_order_penalty"), app.indexOf("const optimizerShader")).includes("base + 4u"),
  penaltyBeforeOptimizer: app.indexOf('profilePassDescriptor(profileSample, "virtual-order-penalty")') <
    app.indexOf('profilePassDescriptor(profileSample, "optimizer")', app.indexOf('profilePassDescriptor(profileSample, "virtual-order-penalty")')),
  frontPathIdentity: app.includes("if (!virtual_tilt_enabled()) { return vec3<f32>(point, 1.0); }"),
  telemetry: app.includes("virtual_tilt_steps_completed") && app.includes("last_virtual_tilt"),
};

assert.ok(Object.values(contracts).every(Boolean), JSON.stringify(contracts));
console.log(JSON.stringify({ ok: true, contracts }, null, 2));
