import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const [app, index, tiltViewer] = await Promise.all([
  readFile(new URL("../web/app.js", import.meta.url), "utf8"),
  readFile(new URL("../web/index.html", import.meta.url), "utf8"),
  readFile(new URL("../web/tilt-viewer.mjs", import.meta.url), "utf8"),
]);

const ANGLE = 5 * Math.PI / 180;
const DEPTH_THRESHOLD = 0.01;
const COLOR_THRESHOLD = 0.08;

function profile({ width, height, scale, theta, opacity, color, samples }) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const axisX = [c * scale[0], s * scale[0]];
  const axisY = [-s * scale[1], c * scale[1]];
  const longSide = Math.max(width, height);
  const frame = [width / longSide, height / longSide];
  const worldX = [axisX[0] * frame[0], axisX[1] * frame[1]];
  const worldY = [axisY[0] * frame[0], axisY[1] * frame[1]];
  const supportDepth = 4 * Math.sin(ANGLE) * Math.max(
    Math.hypot(worldX[0], worldY[0]),
    Math.hypot(worldX[1], worldY[1]),
  );
  const mismatch = Math.max(...samples.map((sample) => (
    sample.reduce((sum, value, channel) => sum + Math.abs(value - color[channel]), 0) / 3
  )));
  const risk = opacity < 0.007 ? 0 : opacity
    * Math.max(0, supportDepth / DEPTH_THRESHOLD - 1)
    * Math.max(0, mismatch / COLOR_THRESHOLD - 1);
  return { supportDepth, mismatch, risk: Math.min(64, risk) };
}

const largeContrasting = profile({
  width: 1024,
  height: 512,
  scale: [0.25, 0.05],
  theta: 0,
  opacity: 0.8,
  color: [0.9, 0.3, 0.1],
  samples: [[0.05, 0.05, 0.05], [0.9, 0.3, 0.1], [0.1, 0.2, 0.8], [0.9, 0.3, 0.1]],
});
assert.ok(largeContrasting.supportDepth > DEPTH_THRESHOLD);
assert.ok(largeContrasting.risk > 0);

const smallContrasting = profile({
  width: 1024,
  height: 512,
  scale: [0.005, 0.003],
  theta: 0.4,
  opacity: 0.8,
  color: [0.9, 0.3, 0.1],
  samples: [[0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0]],
});
assert.equal(smallContrasting.risk, 0);

const largeUniform = profile({
  width: 512,
  height: 1024,
  scale: [0.3, 0.08],
  theta: 1.1,
  opacity: 0.95,
  color: [0.2, 0.4, 0.6],
  samples: Array.from({ length: 4 }, () => [0.2, 0.4, 0.6]),
});
assert.ok(Number.isFinite(largeUniform.supportDepth));
assert.equal(largeUniform.risk, 0);

const lowOpacity = profile({
  width: 1024,
  height: 512,
  scale: [0.25, 0.05],
  theta: 0,
  opacity: 0.005,
  color: [1, 0, 0],
  samples: Array.from({ length: 8 }, () => [0, 1, 1]),
});
assert.equal(lowOpacity.risk, 0);

const parentArea = 0.25 * 0.05;
const splitShrink = 0.8;
const childArea = parentArea * splitShrink * splitShrink;
const alphaFixtures = [];
for (const parentOpacity of [0.007, 0.1, 0.5, 0.73, 0.995]) {
  const childOpacity = parentOpacity * parentArea / (2 * childArea);
  assert.ok(childOpacity >= 0.005 && childOpacity <= 0.99);
  assert.ok(Math.abs(2 * childOpacity * childArea - parentOpacity * parentArea) < 1e-12);
  let squaredError = 0;
  let maximumError = 0;
  let samples = 0;
  for (let x = -4; x <= 4; x += 0.01) {
    const parent = parentOpacity * Math.exp(-0.5 * x ** 2);
    const left = childOpacity * Math.exp(-0.5 * ((x + 0.34) / splitShrink) ** 2);
    const right = childOpacity * Math.exp(-0.5 * ((x - 0.34) / splitShrink) ** 2);
    const replacement = 1 - (1 - left) * (1 - right);
    const error = Math.abs(parent - replacement);
    squaredError += error ** 2;
    maximumError = Math.max(maximumError, error);
    samples += 1;
  }
  const rmse = Math.sqrt(squaredError / samples);
  assert.ok(rmse < 0.05);
  assert.ok(maximumError < 0.1);
  alphaFixtures.push({ parentOpacity, childOpacity, rmse, maximumError });
}

const sourceContracts = {
  densityOnlyProfile: app.includes("fn tilt_split_profile(g: u32") &&
    app.includes("normal optimizer iterations do not pay for it"),
  aspectAwareExportSpace: app.includes("let frameScale = vec2<f32>(config[0] / longSide, config[1] / longSide)"),
  covarianceSupportDepth: app.includes("length(vec2<f32>(worldX.x, worldY.x))") &&
    app.includes("Math.hypot(worldX[0], worldY[0])"),
  fixedFiveDegreeProbe: app.includes("const DEFAULT_TILT_SPLIT_ANGLE_DEGREES = 5"),
  qaBaselineSwitch: app.includes('tiltRobustSplit: typeof overrides.tiltRobustSplit === "boolean"') &&
    app.includes("phase39.tiltRobustSplit ? 1 : 0"),
  colorAndDepthGate: app.includes("supportDepth / depthThreshold") && app.includes("colorMismatch / colorThreshold"),
  fourSigmaColorGate: app.includes("center - axisX * 4.0") && app.includes("center + axisY * 4.0"),
  singleSourceRequired: app.includes("var tiltTrueSplit = mode == 1u && tiltProfile.x > 0.0 && config[42] > 0.5"),
  sourceReplaced: app.includes("xy[source].center = replacementSourcePos") &&
    app.includes("transform[source] = vec4<f32>(replacementSourceScale"),
  adcOnlyGrowthGate: app.includes("let tiltRisk = adc && tiltProfile.x > 0.0"),
  symmetricOffsets: app.includes("xy[source].center - axis * major * 0.34") &&
    app.includes("major * splitOffset * select(side, 1.0, tiltTrueSplit)"),
  broadSplitShrinksBothAxes: app.includes("nextScale = select(axisShrink, sourceT.xy * splitShrink, tiltTrueSplit)") &&
    app.includes("splitScale = select(axisShrink, sourceT.xy * splitShrink, tiltTrueSplit)"),
  appearancePreservingSplit: app.includes("const DEFAULT_TILT_SPLIT_SHRINK = 0.8") &&
    app.includes("color[source] = vec4<f32>(sourceC.rgb, childOpacity)"),
  runtimePlanarScaleCap: app.includes("maxPlanarScale: clampNumber(els.maxPlanarScale.value") &&
    app.includes('const optimizerShader = `') &&
    app.includes("struct Config { values: array<vec4<f32>, 21>, };") &&
    app.includes("let phaseMaxPlanarScale = mix(max(cfg(62u), ${PHASE_ONE_MAX_PLANAR_SCALE}), max(cfg(62u), minScale), phaseOneProgress)") &&
    app.includes("nextScale = min(nextScale, vec2<f32>(phaseMaxPlanarScale))") &&
    app.includes("vec2<f32>(max(config[56], ${MIN_SPLAT_SCALE}))"),
  safeDefaults: app.includes("const DEFAULT_MAX_PLANAR_SCALE = 0.1") &&
    index.includes('id="maxPlanarScale"') && index.includes('value="0.1"') &&
    app.includes('query.get("tilt-robust-split") !== "0"\n        : false') &&
    app.includes('query.get("virtual-tilt") !== "0"\n        : false') &&
    app.includes("const DEFAULT_VIRTUAL_ORDER_PENALTY_WEIGHT = 0"),
  viewerFrontOrientation: !tiltViewer.includes("splat.setEulerAngles") &&
    tiltViewer.includes('const camera = new Entity("Tilt Camera")'),
  viewerRuntimeSafety: tiltViewer.includes("canonicalOrbitRadius(frame") &&
    !tiltViewer.includes("plySupportHalfExtents") &&
    tiltViewer.includes("const sortTimeoutMs = Math.min(30000") &&
    tiltViewer.includes("requestedPitch === pitch && requestedYaw === yaw") &&
    tiltViewer.includes('from "./tilt-camera.mjs"') &&
    tiltViewer.includes("orbitCameraPose(viewPitch, viewYaw, viewRadius)") &&
    tiltViewer.includes('viewMode: cameraMarkersVisible ? "camera-pool-overview" : "center-orbit"') &&
    tiltViewer.includes("fitOrbitRadius(supportFrame") &&
    app.includes("supportFrame: renderFootprintSupportFrame(state.image, state.params)") &&
    tiltViewer.includes('blobUrl = ""') &&
    app.includes("function assertTiltViewerCapacity") &&
    index.includes('data-testid="tilt-pitch" type="range" min="-75" max="75"') &&
    index.includes('data-testid="tilt-frame-overlay"'),
  integratedOpacityPreserved: app.includes("massShare / max(0.00000001, replacementArea)"),
  boundedMassSaturation: app.includes("childOpacity = clamp(replacementOpacity, 0.005, 0.99)") &&
    app.includes("tilt_opacity_saturations: values[21]"),
  sourceIndependentFallback: app.includes("if (mode == 0u) {") &&
    app.indexOf("if (mode == 0u) {") < app.indexOf("let source = (encoded & SOURCE_MASK) - 1u"),
  splitStatsConserved: app.includes("stats[index] = select(sourceStats, sourceStats * 0.5, tiltTrueSplit)") &&
    app.includes("stats[source] = sourceStats * 0.5"),
  capacityRecycleSplit: app.includes("let sourceTiltRisk = adcRecycle && sourceTiltProfile.x > 0.0") &&
    app.includes("let sourceRole = select(ROLE_SOURCE_OTHER, ROLE_SOURCE_SPLIT, sourceTiltRisk)") &&
    app.includes("var tiltTrueSplit = adcRecycle && (sourceClaim & ROLE_MASK) == ROLE_SOURCE_SPLIT"),
  optimizerStateReset: app.includes("ROLE_SOURCE_SPLIT") && app.includes("optimizerSourceResetPipeline"),
  telemetry: app.includes("tilt_risk_candidates: values[19]") && app.includes("tilt_true_splits: values[20]"),
  finalTelemetry: app.includes("state.metrics.tilt_risk = summarizeTiltRisk(state.params, state.image)"),
};

assert.ok(Object.values(sourceContracts).every(Boolean), JSON.stringify(sourceContracts));
console.log(JSON.stringify({ ok: true, largeContrasting, smallContrasting, largeUniform, alphaFixtures, sourceContracts }, null, 2));
