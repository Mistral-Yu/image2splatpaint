import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  fibonacciHemispherePoses,
  orbitProjectionContract,
  projectPlanarSplatPoint,
  trainingCameraMarkerGeometry,
} from "../web/tilt-camera.mjs";

const app = await readFile(new URL("../web/app.js", import.meta.url), "utf8");
const index = await readFile(new URL("../web/index.html", import.meta.url), "utf8");
const tiltViewer = await readFile(new URL("../web/tilt-viewer.mjs", import.meta.url), "utf8");

assert.match(app, /virtual_camera_evaluation: m\.virtual_camera_evaluation/);

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
  const DEFAULT_SHARED_CAMERA_FOV_DEGREES = 50;
  const MIN_SHARED_CAMERA_FOV_DEGREES = 25;
  const MAX_SHARED_CAMERA_FOV_DEGREES = 55;
  const VIRTUAL_TILT_FOV_DEGREES = 50;
  const CURRICULUM_GROWTH_FRACTION = 6 / 7;
  const VIRTUAL_TILT_DIRECTIONS = [[1,0],[-1,0],[0,1],[0,-1]];
  const DEFAULT_VIRTUAL_CAMERA_COUNT = 100;
  const MAX_VIRTUAL_CAMERA_COUNT = 128;
  const DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES = 60;
  const VIRTUAL_CAMERA_GOLDEN_ANGLE_RADIANS = Math.PI * (3 - Math.sqrt(5));
  const sharedTiltOrbitRadiusCache = new Map();
  ${functionSource("planarTiltRotation")}
  ${functionSource("clampSharedCameraFov")}
  ${functionSource("projectPlanarPoint")}
  ${functionSource("inverseProjectPlanarPoint")}
  ${functionSource("virtualTiltStepSpec")}
  ${functionSource("virtualCameraShuffle")}
  ${functionSource("virtualCameraBag")}
  ${functionSource("virtualCameraFibonacciPose")}
  ${functionSource("virtualCameraFibonacciPoses")}
  ${functionSource("sharedTiltOrbitRadius")}
  ${functionSource("virtualTeacherCoverage")}
  ${functionSource("virtualCameraCoverageStats")}
  ${functionSource("virtualCameraIntrinsics")}
  ${functionSource("virtualFrontIntrinsics")}
  ${functionSource("virtualCameraSamplingStepSpec")}
  ${functionSource("virtualCameraTrainingStepSpec")}
  ${functionSource("virtualCameraSamplingCountsThroughStep")}
  ${functionSource("virtualCameraCatalog")}
  return {
    planarTiltRotation,
    projectPlanarPoint,
    inverseProjectPlanarPoint,
    virtualTiltStepSpec,
    virtualCameraBag,
    virtualCameraFibonacciPose,
    virtualCameraFibonacciPoses,
    sharedTiltOrbitRadius,
    virtualTeacherCoverage,
    virtualCameraCoverageStats,
    virtualCameraIntrinsics,
    virtualFrontIntrinsics,
    virtualCameraSamplingStepSpec,
    virtualCameraTrainingStepSpec,
    virtualCameraSamplingCountsThroughStep,
    virtualCameraCatalog,
  };
`)();

for (const frame of [{ width: 1, height: 1 }, { width: 1600, height: 900 }, { width: 900, height: 1600 }]) {
  for (const [pitchDegrees, yawDegrees] of [[0, 0], [2, 0], [-2, 0], [0, 5], [0, -5], [5, -5]]) {
    const pitch = pitchDegrees * Math.PI / 180;
    const yaw = yawDegrees * Math.PI / 180;
    for (const z of [-0.005, 0, 0.005]) {
      for (const point of [[-1, -1], [-0.4, 0.7], [0, 0], [0.9, -0.2], [1, 1]]) {
        const projected = helpers.projectPlanarPoint(point, pitch, yaw, 4, z, frame);
        const restored = helpers.inverseProjectPlanarPoint(projected.point, pitch, yaw, 4, frame, z);
        assert.ok(projected.valid && restored.valid);
        assert.ok(Math.abs(restored.point[0] - point[0]) < 1e-11);
        assert.ok(Math.abs(restored.point[1] - point[1]) < 1e-11);
      }
    }
  }
}

const variants = { enabled: true, interval: 32, weight: 0.25, cameraDistance: 4, autoCameraDistance: true };
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

const samplingVariants = {
  enabled: true,
  slots: 128,
  virtualSlots: 4,
  cameraCount: 12,
  maxAngleDegrees: 30,
  fovDegrees: 50,
  seed: 49734321,
  cameraDistance: 4,
  autoCameraDistance: true,
  orderPenaltyWeight: 0,
};
const bag = helpers.virtualCameraBag(0, samplingVariants);
assert.equal(bag.length, 128);
assert.equal(bag.filter((value) => value < 0).length, 124);
assert.deepEqual([0, 1, 2, 3].map((camera) => bag.filter((value) => value === camera).length), [1, 1, 1, 1]);
assert.deepEqual(helpers.virtualCameraBag(0, samplingVariants), bag);
assert.notDeepEqual(helpers.virtualCameraBag(1, samplingVariants), bag);
const oneSixteenthBag = helpers.virtualCameraBag(0, { ...samplingVariants, virtualSlots: 8 });
assert.equal(oneSixteenthBag.length, 128);
assert.equal(oneSixteenthBag.filter((value) => value >= 0).length, 8);
assert.deepEqual(Array.from({ length: 8 }, (_, camera) => oneSixteenthBag.filter((value) => value === camera).length), new Array(8).fill(1));
const virtualSlots = bag.map((camera, slot) => ({ camera, slot })).filter(({ camera }) => camera >= 0);
const sampledCameras = virtualSlots.map(({ slot }) => helpers.virtualCameraSamplingStepSpec(
  1 + slot,
  "coarse",
  7000,
  1000,
  samplingVariants,
));
assert.equal(sampledCameras.every((sample) => sample.kind === "virtual" && sample.angleDegrees > 0 && sample.weight === 1), true);
assert.equal(new Set(sampledCameras.map((sample) => sample.cameraId)).size, 4);
const retryStep = 1 + virtualSlots[0].slot;
assert.deepEqual(
  helpers.virtualCameraSamplingStepSpec(retryStep, "coarse", 7000, 1000, samplingVariants),
  helpers.virtualCameraSamplingStepSpec(retryStep, "coarse", 7000, 1000, samplingVariants),
);
for (const logicalStep of [1, 1000, 1001, 1128, 1129, 7000]) {
  const counts = helpers.virtualCameraSamplingCountsThroughStep(logicalStep, 1000, samplingVariants);
  const samples = Array.from({ length: logicalStep }, (_, index) => helpers.virtualCameraSamplingStepSpec(
    index + 1,
    index + 1 <= 1000 ? "coarse" : index + 1 < 3000 ? "mid" : "full",
    7000,
    1000,
    samplingVariants,
  ));
  assert.equal(counts.front, samples.filter((sample) => sample.kind === "front").length);
  assert.equal(counts.virtual, samples.filter((sample) => sample.kind === "virtual").length);
}
const lateCycle = 40;
const lateBag = helpers.virtualCameraBag(lateCycle, samplingVariants);
const lateSlot = lateBag.findIndex((direction) => direction >= 0);
const lateSample = helpers.virtualCameraSamplingStepSpec(1 + lateCycle * 128 + lateSlot, "full", 7000, 1000, samplingVariants);
assert.equal(lateSample.kind, "virtual");
assert.ok(lateSample.angleDegrees > 0 && lateSample.angleDegrees <= 30);

const additiveVariants = {
  ...samplingVariants,
  objectiveMode: "additive",
  planeConstrained: true,
  regularizationWeight: 0.1,
  regularizationRampSteps: 200,
  midAngleDegrees: 2,
  fullAngleDegrees: 5,
};
const additiveMid = helpers.virtualCameraTrainingStepSpec(
  { ...lateSample, angleDegrees: 30, pitchDegrees: 18, yawDegrees: 24 },
  1100,
  "mid",
  1000,
  additiveVariants,
);
assert.ok(Math.abs(additiveMid.angleDegrees - 2) < 1e-12);
assert.ok(Math.abs(additiveMid.weight - 0.1) < 1e-12);
assert.ok(Math.abs(Math.hypot(additiveMid.pitchDegrees, additiveMid.yawDegrees) - 2) < 1e-12);
const additiveFull = helpers.virtualCameraTrainingStepSpec(
  { ...lateSample, angleDegrees: 30, pitchDegrees: 18, yawDegrees: 24 },
  1400,
  "full",
  1000,
  additiveVariants,
);
assert.ok(Math.abs(additiveFull.angleDegrees - 5) < 1e-12);
assert.ok(Math.abs(additiveFull.weight - 0.1) < 1e-12);

const appFibonacci = helpers.virtualCameraFibonacciPoses(12, 30);
const sharedFibonacci = fibonacciHemispherePoses(13, 30).slice(1);
assert.equal(appFibonacci.length, 12);
for (let index = 0; index < appFibonacci.length; index += 1) {
  assert.ok(Math.abs(appFibonacci[index].pitchDegrees - sharedFibonacci[index].pitchDegrees) < 1e-12);
  assert.ok(Math.abs(appFibonacci[index].yawDegrees - sharedFibonacci[index].yawDegrees) < 1e-12);
  assert.ok(Math.abs(appFibonacci[index].polarDegrees - sharedFibonacci[index].polarDegrees) < 1e-12);
}

const teacherCoverage = (frame, pose, radius, fovDegrees = 50, sampleSide = 96) => {
  let valid = 0;
  for (let y = 0; y < sampleSide; y += 1) {
    const v = y / (sampleSide - 1) * 2 - 1;
    for (let x = 0; x < sampleSide; x += 1) {
      const u = x / (sampleSide - 1) * 2 - 1;
      const source = helpers.inverseProjectPlanarPoint(
        [u, v],
        pose.pitchDegrees * Math.PI / 180,
        pose.yawDegrees * Math.PI / 180,
        radius,
        frame,
        0,
        fovDegrees,
      );
      if (
        source.valid &&
        source.point[0] >= -1 && source.point[0] <= 1 &&
        source.point[1] >= -1 && source.point[1] <= 1
      ) valid += 1;
    }
  }
  return valid / (sampleSide * sampleSide);
};

const cameraContractResults = [];
for (const frame of [{ width: 1024, height: 682 }, { width: 768, height: 1024 }]) {
  const radius30 = helpers.sharedTiltOrbitRadius(frame.width, frame.height, 30, 30);
  const radius75 = helpers.sharedTiltOrbitRadius(frame.width, frame.height, 75, 49);
  assert.ok(Number.isFinite(radius30) && radius30 > 1 && radius30 < 10);
  assert.ok(Math.abs(radius30 - radius75) < 1e-12, "all cameras must share one front-identity orbit radius");
  const expectedRadius = frame.height / Math.max(frame.width, frame.height) / Math.tan(25 * Math.PI / 180);
  assert.ok(Math.abs(radius30 - expectedRadius) < 1e-12);
  for (const corner of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) {
    const front = helpers.projectPlanarPoint(corner, 0, 0, radius30, 0, frame, 50);
    assert.ok(front.valid);
    assert.ok(Math.hypot(front.point[0] - corner[0], front.point[1] - corner[1]) < 1e-12);
  }
  const poses = helpers.virtualCameraFibonacciPoses(30, 30);
  const coverages = [];
  for (const pose of poses) {
    for (const corner of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) {
      const projected = helpers.projectPlanarPoint(
        corner,
        pose.pitchDegrees * Math.PI / 180,
        pose.yawDegrees * Math.PI / 180,
        radius30,
        0,
        frame,
        50,
      );
      assert.ok(projected.valid);
      const restored = helpers.inverseProjectPlanarPoint(
        projected.point,
        pose.pitchDegrees * Math.PI / 180,
        pose.yawDegrees * Math.PI / 180,
        radius30,
        frame,
        0,
        50,
      );
      assert.ok(restored.valid);
      assert.ok(Math.hypot(restored.point[0] - corner[0], restored.point[1] - corner[1]) < 1e-10);
    }
    coverages.push(teacherCoverage(frame, pose, radius30, 50));
  }
  const minimumCoverage = Math.min(...coverages);
  const requiredCoverage = frame.width > frame.height ? 0.70 : 0.75;
  assert.ok(minimumCoverage >= requiredCoverage, `virtual teacher coverage too small: ${minimumCoverage}`);
  const radius35 = helpers.sharedTiltOrbitRadius(frame.width, frame.height, 30, 30, 35);
  const coverage35 = Math.min(...poses.map((pose) => teacherCoverage(frame, pose, radius35, 35)));
  assert.ok(coverage35 > minimumCoverage, "smaller shared FOV should increase teacher coverage");
  cameraContractResults.push({ frame, radius30, radius75, minimumCoverage, maximumCoverage: Math.max(...coverages) });
}

const landscapeIntrinsics = helpers.virtualCameraIntrinsics(1024, 682, 50);
assert.ok(Math.abs(landscapeIntrinsics.fx - 682 * 0.5 / Math.tan(25 * Math.PI / 180)) < 1e-12);
assert.equal(landscapeIntrinsics.fx, landscapeIntrinsics.fy);
assert.equal(landscapeIntrinsics.cx, 511.5);
assert.equal(landscapeIntrinsics.cy, 340.5);
assert.deepEqual(helpers.virtualFrontIntrinsics(1024, 682, 50), {
  ...landscapeIntrinsics,
  projection: "identity-reference",
});

const uniformVariants = {
  ...samplingVariants,
  slots: 13,
  virtualSlots: 12,
  uniformCameras: true,
};
const uniformBag = helpers.virtualCameraBag(0, uniformVariants);
assert.equal(uniformBag.length, 13);
assert.equal(uniformBag.filter((camera) => camera < 0).length, 1);
assert.deepEqual(
  Array.from({ length: 12 }, (_, camera) => uniformBag.filter((value) => value === camera).length),
  new Array(12).fill(1),
);
const uniformPhaseOne = Array.from({ length: 1000 }, (_, index) => helpers.virtualCameraSamplingStepSpec(
  1 + index,
  "coarse",
  7000,
  1000,
  uniformVariants,
));
assert.equal(uniformPhaseOne.some((sample) => sample.kind === "virtual"), true);
assert.equal(new Set(uniformPhaseOne.map((sample) => sample.cameraId)).size, 13);
const uniformSamples = Array.from({ length: 130 }, (_, index) => helpers.virtualCameraSamplingStepSpec(
  1 + index,
  "coarse",
  7000,
  1000,
  uniformVariants,
));
const uniformCameraCounts = new Map();
for (const sample of uniformSamples) {
  uniformCameraCounts.set(sample.cameraId, (uniformCameraCounts.get(sample.cameraId) || 0) + 1);
}
assert.equal(uniformCameraCounts.size, 13);
assert.ok(Math.max(...uniformCameraCounts.values()) - Math.min(...uniformCameraCounts.values()) <= 1);
const directlyCountSamples = (step, variants) => {
  const samples = Array.from({ length: step }, (_, index) => helpers.virtualCameraSamplingStepSpec(
    index + 1,
    index + 1 <= 1000 ? "coarse" : index + 1 < 3000 ? "mid" : "full",
    7000,
    1000,
    variants,
  ));
  const virtual = samples.filter((sample) => sample.kind === "virtual").length;
  return { front: step - virtual, virtual };
};
assert.deepEqual(
  helpers.virtualCameraSamplingCountsThroughStep(1000, 1000, uniformVariants),
  directlyCountSamples(1000, uniformVariants),
);
assert.deepEqual(
  helpers.virtualCameraSamplingCountsThroughStep(1130, 1000, uniformVariants),
  directlyCountSamples(1130, uniformVariants),
);
const expectedSevenThousandCounts = [
  { share: 3.125, slots: 128, virtualSlots: 4 },
  { share: 6.25, slots: 128, virtualSlots: 8 },
  { share: 12.5, slots: 128, virtualSlots: 16 },
  { share: 25, slots: 128, virtualSlots: 32 },
  { share: 50, slots: 128, virtualSlots: 64 },
];
for (const candidate of expectedSevenThousandCounts) {
  const variants = { ...samplingVariants, slots: candidate.slots, virtualSlots: candidate.virtualSlots };
  assert.deepEqual(
    helpers.virtualCameraSamplingCountsThroughStep(7000, 1000, variants),
    directlyCountSamples(7000, variants),
    `7000-step count mismatch for share ${candidate.share}%`,
  );
}
assert.deepEqual(
  helpers.virtualCameraSamplingCountsThroughStep(7000, 1000, uniformVariants),
  directlyCountSamples(7000, uniformVariants),
);

const catalog = helpers.virtualCameraCatalog(1, 1, 4, samplingVariants);
assert.equal(catalog.length, 13);
assert.equal(new Set(catalog.map((camera) => camera.id)).size, 13);
const sharedCatalogRadius = helpers.sharedTiltOrbitRadius(1024, 682, 30, 12, 50);
const sizedCatalog = helpers.virtualCameraCatalog(1024, 682, sharedCatalogRadius, samplingVariants);
assert.equal(sizedCatalog[0].intrinsics.model, "pinhole");
assert.equal(sizedCatalog.slice(1).every((camera) => camera.intrinsics.model === "pinhole"), true);
assert.equal(sizedCatalog.every((camera) => camera.intrinsics.fx === camera.intrinsics.fy), true);
assert.equal(sizedCatalog.every((camera) => camera.intrinsics.fov_degrees === 50), true);
assert.equal(sizedCatalog.every((camera) => camera.intrinsics.fx === sizedCatalog[0].intrinsics.fx), true);
assert.equal(sizedCatalog.every((camera) => camera.intrinsics.cx === sizedCatalog[0].intrinsics.cx), true);
assert.equal(sizedCatalog.every((camera) => camera.intrinsics.cy === sizedCatalog[0].intrinsics.cy), true);
assert.ok(sizedCatalog[1].intrinsics.fx > 700 && sizedCatalog[1].intrinsics.fx < 740);
const cameraSnapshot = {
  cameras: sizedCatalog.map((camera) => ({ ...camera, multiplicity: 1 })),
  active_camera_id: "virtual-fib-12",
  orbit_radius: sharedCatalogRadius,
  fov_degrees: 50,
  target: [0, 0, 0],
  image_aspect: 1024 / 682,
};
const cameraMarkers = trainingCameraMarkerGeometry(cameraSnapshot);
assert.equal(cameraMarkers.length, sizedCatalog.length);
for (const camera of sizedCatalog) {
  const marker = cameraMarkers.find((candidate) => candidate.ids.includes(camera.id));
  assert.ok(marker, `marker missing for ${camera.id}`);
  assert.equal(marker.fovDegrees, camera.intrinsics.fov_degrees);
  const contract = orbitProjectionContract(
    { width: 1024, height: 682 },
    { width: 1024, height: 682 },
    camera.pitch_degrees,
    camera.yaw_degrees,
    cameraSnapshot.orbit_radius,
    camera.intrinsics.fov_degrees,
  );
  for (const point of [[-0.7, -0.4], [0, 0], [0.65, 0.3]]) {
    const viewerProjection = projectPlanarSplatPoint(point, contract, 0);
    const trainingProjection = camera.kind === "front"
      ? { point, valid: true }
      : helpers.projectPlanarPoint(
        point,
        camera.pitch_degrees * Math.PI / 180,
        camera.yaw_degrees * Math.PI / 180,
        cameraSnapshot.orbit_radius,
        0,
        { width: 1024, height: 682 },
      );
    assert.ok(viewerProjection.valid && trainingProjection.valid);
    assert.ok(
      Math.hypot(
        viewerProjection.point[0] - trainingProjection.point[0],
        viewerProjection.point[1] - trainingProjection.point[1],
      ) < 1e-10,
      `training/viewer projection mismatch for ${camera.id}`,
    );
  }
}

const depthX = Math.sin(5 * Math.PI / 180) * 0.25;
const depthY = 0;
const sigmaDepth = Math.hypot(depthX, depthY);
const supportDepth = 4 * sigmaDepth;
const excess = Math.max(0, supportDepth / 0.01 - 1);
const coefficient = 2 * 0.01 * 0.5 * excess / 0.01;
const scaleGradientX = coefficient * 4 * depthX ** 2 / sigmaDepth;
assert.ok(scaleGradientX > 0);

// A z=0 source board still has position-dependent camera-space depth when tilted.
// This is the depth contract used by virtual training; no learned per-splat z is needed.
const boardFrame = { width: 1600, height: 900 };
const boardYaw = 30 * Math.PI / 180;
const boardDistance = 4;
const boardLeft = helpers.projectPlanarPoint([-0.75, 0], 0, boardYaw, boardDistance, 0, boardFrame);
const boardRight = helpers.projectPlanarPoint([0.75, 0], 0, boardYaw, boardDistance, 0, boardFrame);
assert.ok(boardLeft.valid && boardRight.valid);
assert.ok(Math.abs(boardLeft.depth - boardRight.depth) > 0.1, "tilted board depth must come from camera pose and planar x/y");
for (const point of [[-0.75, 0], [0, 0], [0.75, 0]]) {
  const projected = helpers.projectPlanarPoint(point, 0, boardYaw, boardDistance, 0, boardFrame);
  const restored = helpers.inverseProjectPlanarPoint(projected.point, 0, boardYaw, boardDistance, boardFrame, 0);
  assert.ok(projected.valid && restored.valid);
  assert.ok(Math.hypot(restored.point[0] - point[0], restored.point[1] - point[1]) < 1e-11);
}

const contracts = {
  configCapacity: app.includes("const TRAIN_CONFIG_FLOATS = 84") &&
    app.includes("add(\"config\", TRAIN_CONFIG_BYTES, true)") &&
    app.includes("size: TRAIN_CONFIG_BYTES"),
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
    app.includes("-frame.y * point.y") && app.includes("depth * tanHalfFov * aspect"),
  inverseHomography: app.includes("fn virtual_inverse_point_at_z") && app.includes("let determinant = a00 * a11 - a01 * a10"),
  projectedTileBounds: app.includes("let p0 = virtual_project_point(center + vec2<f32>(-radius.x, -radius.y)") &&
    app.includes("!virtual_tilt_enabled() && !tile_intersects_footprint"),
  cameraDepthSort: app.includes("virtual_camera_depth(xy[a].center, transform[a].w, xy[a].rawDepth)") &&
    app.includes("virtual_pass_layer_depth(packedTag: f32, rawDepth: f32)"),
  planeConstrainedVirtualTraining:
    app.includes("fn virtual_pass_layer_depth(packedTag: f32, rawDepth: f32)") &&
    app.includes("virtual_tilt_enabled() && cfg(65u) > 0.5") &&
    app.includes("config[65] = tiltStep.enabled && tiltStep.planeConstrained !== false ? 1 : 0") &&
    app.includes("plane_constrained: Boolean(tiltStep.enabled && tiltStep.planeConstrained !== false)") &&
    app.includes('"  let layerZ = virtual_pass_layer_depth(t.w, rawDepth);"'),
  selectedViewAllRouting:
    !index.includes('data-testid="virtual-front-sharpness"') &&
    !index.includes('data-testid="virtual-front-canonical"') &&
    !index.includes('data-testid="virtual-source-domain"') &&
    app.includes("threeDgsMultiview: true") &&
    app.includes('gradientChannels: { geometry: 1, appearance: 1, density: 1, depth: 1 }') &&
    app.includes('virtual_camera_3dgs_multiview: true') &&
    app.includes('"selected-view-all"') &&
    app.includes('"camera-projected-3d-covariance"') &&
    app.includes('config[74] = tiltStep.enabled && cameraCovariance3d ? 1 : 0') &&
    app.includes('fn camera_covariance_3d_enabled() -> bool') &&
    app.includes('fn project_planar_gaussian(center: vec2<f32>, layerZ: f32, t: vec4<f32>)'),
  virtualInvalidRegionVariant:
    app.includes('requestedInvalidRegionMode === "black-loss" ? "black-loss" : "mask"') &&
    app.includes('config[66] = tiltStep.enabled && tiltStep.invalidRegionMode === "black-loss" ? 1 : 0') &&
    app.includes('fn training_sample_valid(sourcePoint: vec3<f32>) -> bool') &&
    app.includes('!source_domain_reprojection_enabled() && cfg(66u) > 0.5') &&
    app.includes('virtual_camera_invalid_region_mode'),
  sharedOrbitRadius: app.includes("function sharedTiltOrbitRadius") &&
    app.includes("requestedTiltStep.maxAngleDegrees") &&
    app.includes("runVirtualCameraSampling.maxAngleDegrees") &&
    app.includes("const radius = frameY / Math.tan(safeFov * Math.PI / 360)") &&
    app.includes("requestedTiltStep.fovDegrees") &&
    app.includes("fov_degrees: runVirtualCameraSampling.fovDegrees") &&
    app.includes("config[64] = tiltStep.fovDegrees") &&
    app.includes("sharedTiltOrbitRadiusCache") &&
    tiltViewer.includes("cameraPool?.orbit_radius") &&
    tiltViewer.includes("sharedFovDegrees") &&
    tiltViewer.includes("fitOrbitRadius(supportFrame") &&
    tiltViewer.includes("trainingOrbitRadius: orbitRadius"),
  additiveFrontTilt: app.includes('viewOverride: "front"') &&
    app.includes("clearExactGradient: false") &&
    app.includes("gradientNormalization: 1 / (1 + Math.max(0, tiltStep.weight))") &&
    app.includes("let gradientNormalization = max(cfg(63u), 0.0001)") &&
    app.includes("const effectiveExactBackward = qualityRecoveryVariants().exactBackward || !this.renderGradientPipeline") &&
    app.includes("tiltStep.enabled && effectiveExactBackward && variants.ewa2x2"),
  sampledSingleCamera:
    app.includes('Virtual camera sampling requires exact WebGPU backward and finite-pixel EWA.') &&
    app.includes('viewOverride: virtualStep ? trainingCamera : "front"') &&
    app.includes("gradientNormalization: 1") &&
    app.includes("virtual_camera_sample") &&
    app.includes("virtualCameraSampling: runVirtualCameraSampling") &&
    app.includes("requested_share_percent: runVirtualCameraSampling.requestedSharePercent") &&
    app.includes("effective_share_percent: runVirtualCameraSampling.effectiveSharePercent") &&
    app.includes('mode: uniformCameras ? "uniform-all-cameras" : "weighted-virtual-share"'),
  phaseOneVirtualSampling:
    app.includes("Camera sampling is independent of the resolution curriculum") &&
    app.includes("const virtualCameraWarmupSteps = 0") &&
    app.includes("warmup_steps: 0") &&
    !app.includes('if (!variants.enabled || stage === "coarse") return front'),
  sameSrgbTeacherContract:
    app.includes('contract: "front and virtual teachers use the same sRGB signal values"') &&
    app.includes('teacher_sampling: "bilinear interpolation in sRGB signal space"') &&
    app.includes("function trainingColorSpaceAudit") &&
    app.includes("state.metrics.color_space_audit = trainingColorSpaceAudit") &&
    app.includes("color_space_audit: m.color_space_audit ? structuredClone(m.color_space_audit) : null"),
  frontOnlyLivePreview:
    app.includes('virtualCameraSample?.kind === "virtual"') &&
    app.includes("await renderer.prepareTileLists(previewImage, state.params, { sync: true })") &&
    app.includes("await renderer.refreshRenderState(previewImage, state.params)") &&
    app.indexOf("await renderer.refreshRenderState(previewImage, state.params)") <
      app.indexOf("renderer.presentTrainState(state.image)", app.indexOf("async function presentTrainingPreview")),
  fibonacciSampling:
    app.includes("function virtualCameraFibonacciPoses") &&
    app.includes("virtual_camera_count: runVirtualCameraSampling.cameraCount") &&
    app.includes("max_angle_degrees: runVirtualCameraSampling.maxAngleDegrees") &&
    app.includes("const uniformCameras = !hasSlotOverride && requestedUiSharePercent >= 100"),
  rejectedPairwiseAbsent: !app.includes("pairwise-order-flip") &&
    !app.includes("pairwise_order_flip") &&
    !app.includes("pairwiseOrderFlip"),
  minimumStorageLimit: app.includes("REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 8") &&
    app.includes("PREFERRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 9") &&
    app.includes("!legacyGradientSupported || (this.renderGradientPipeline && this.parallelRenderGradientPipeline)") &&
    app.includes("qualityVariants.exactBackward || !this.renderGradientPipeline"),
  warpedTarget: app.includes("target_rgb_at(inversePoint.xy") && app.includes("target_color_at(inversePoint.xy"),
  validMask: app.includes("if (inversePoint.z < 0.5)") && app.includes("valid = 1.0"),
  exactBackwardMapping: app.includes('"  let splatPoint = virtual_inverse_point_at_z(outputPoint, layerZ).xy;"') &&
    app.includes('"  let layerZ = virtual_pass_layer_depth(t.w, rawDepth);"'),
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
  obliqueOverlapDiagnostics:
    app.includes("const OVERLAP_METRIC_STRIDE = 19") &&
    app.includes("async computeObliqueDiagnostics(image, params)") &&
    app.includes("state.metrics?.virtual_camera_sampling?.orbit_radius") &&
    app.includes("yawDegrees: 0, cameraDistance") &&
    app.includes("for (const angle of [15, 30, 45, 60, 75])") &&
    app.includes("adjacent_order_flip_ratio") &&
    app.includes("async obliqueOverlapDiagnostics()"),
  obliqueProjectionParity:
    app.includes("let samplePoint = virtual_inverse_point_at_z(outputPoint, virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy") &&
    app.includes("target_rgb_at(inversePoint.xy, width, height)") &&
    app.includes("await this.refreshRenderState(image, params)"),
  allVirtualTeacherMetrics:
    app.includes("const VIRTUAL_CAMERA_METRIC_TILE_STRIDE = 15") &&
    app.includes("async ensureVirtualCameraMetricsPipeline()") &&
    app.includes("async computeVirtualCameraEvaluation(image, params, frontMetrics)") &&
    app.includes('target: "known planar source reprojected per camera"') &&
    app.includes('aggregation: "equal-camera macro; p10/min retain weak virtual views"') &&
    app.includes("virtual_camera_evaluation = await state.webgpu.renderer.computeVirtualCameraEvaluation") &&
    app.includes("virtual teacher evaluation cameras=") &&
    app.includes("data.virtualCameraEvaluationSummary = virtualEvaluation") &&
    app.includes("data.virtualCameraTiltSummary = obliqueViews"),
  displayedCamerasWereTrained:
    app.includes("function tiltCameraCounts(sampling)") &&
    app.includes('source: "runtime-step-history"') &&
    app.includes('source: "deterministic-schedule"') &&
    app.includes(".filter((camera) => camera.multiplicity > 0)") &&
    tiltViewer.includes("sampleCount: cameraMarkers.reduce") &&
    app.includes("dataset.tiltCameraSampleCount"),
};

assert.ok(app.includes("for (const angle of [15, 30, 45, 60, 75])"));
assert.ok(app.includes("gpu_training_memory: m.gpu_training_memory ? structuredClone(m.gpu_training_memory) : null"));

assert.ok(Object.values(contracts).every(Boolean), JSON.stringify(contracts));
console.log(JSON.stringify({ ok: true, contracts, cameraContractResults }, null, 2));
