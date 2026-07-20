import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const app = await readFile(new URL("../web/app.js", import.meta.url), "utf8");
const html = await readFile(new URL("../web/index.html", import.meta.url), "utf8");

function compose(layers, background = [0, 0, 0]) {
  const rgb = [0, 0, 0];
  let transmittance = 1;
  for (const layer of layers) {
    const alpha = Math.min(0.99, Math.max(0, layer.opacity * layer.kernel * (layer.mip ?? 1)));
    if (alpha < 1 / 255 || transmittance < 1e-4) continue;
    for (let channel = 0; channel < 3; channel += 1) {
      rgb[channel] += transmittance * alpha * layer.color[channel];
    }
    transmittance *= 1 - alpha;
  }
  for (let channel = 0; channel < 3; channel += 1) rgb[channel] += transmittance * background[channel];
  return { rgb, alpha: 1 - transmittance };
}

function close(actual, expected, tolerance = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

function loss(layers, targetRgb, targetAlpha, background = [0, 0, 0]) {
  const rendered = compose(layers, background);
  const rgb = rendered.rgb.reduce((sum, value, channel) => sum + 0.5 * (value - targetRgb[channel]) ** 2, 0);
  return rgb + 0.5 * (rendered.alpha - targetAlpha) ** 2;
}

function exactReverse(layers, targetRgb, targetAlpha, background = [0, 0, 0]) {
  const rendered = compose(layers, background);
  const dColor = rendered.rgb.map((value, channel) => value - targetRgb[channel]);
  const dCompositeAlpha = rendered.alpha - targetAlpha;
  let transAfter = 1 - rendered.alpha;
  let gradTransmittance = dColor.reduce((sum, value, channel) => sum + value * background[channel], 0) - dCompositeAlpha;
  const gradients = Array.from({ length: layers.length }, () => ({ color: [0, 0, 0], opacity: 0 }));
  for (let index = layers.length - 1; index >= 0; index -= 1) {
    const layer = layers[index];
    const alpha = layer.opacity * layer.kernel * (layer.mip ?? 1);
    const transBefore = transAfter / (1 - alpha);
    gradients[index].color = dColor.map((value) => value * transBefore * alpha);
    const dAlpha = transBefore * dColor.reduce((sum, value, channel) => sum + value * layer.color[channel], 0)
      - gradTransmittance * transBefore;
    gradients[index].opacity = dAlpha * layer.kernel * (layer.mip ?? 1);
    gradTransmittance = dColor.reduce((sum, value, channel) => sum + value * alpha * layer.color[channel], 0)
      + gradTransmittance * (1 - alpha);
    transAfter = transBefore;
  }
  return gradients;
}

function gaussianWithDerivatives(point, center, scale, theta, pixelSigma = 0.003) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const dx = point[0] - center[0];
  const dy = point[1] - center[1];
  const rx = c * dx + s * dy;
  const ry = -s * dx + c * dy;
  const effective = scale.map((value) => Math.hypot(value, pixelSigma));
  const inverse = effective.map((value) => 1 / value ** 2);
  const kernel = Math.exp(-0.5 * (rx ** 2 * inverse[0] + ry ** 2 * inverse[1]));
  const mip = Math.sqrt((scale[0] * scale[1]) / (effective[0] * effective[1]));
  const weighted = kernel * mip;
  const dCenterLog = [
    rx * c * inverse[0] - ry * s * inverse[1],
    rx * s * inverse[0] + ry * c * inverse[1],
  ];
  const dLogScale = [
    rx ** 2 * scale[0] ** 2 / effective[0] ** 4 + 0.5 * (1 - scale[0] ** 2 / effective[0] ** 2),
    ry ** 2 * scale[1] ** 2 / effective[1] ** 4 + 0.5 * (1 - scale[1] ** 2 / effective[1] ** 2),
  ];
  const dThetaLog = -rx * ry * (inverse[0] - inverse[1]);
  return {
    weighted,
    center: dCenterLog.map((value) => value * weighted),
    logScale: dLogScale.map((value) => value * weighted),
    theta: dThetaLog * weighted,
  };
}

const red = { color: [1, 0, 0], opacity: 0.5, kernel: 1 };
const blue = { color: [0, 0, 1], opacity: 0.5, kernel: 1 };
const one = compose([red], [0, 0, 0]);
close(one.alpha, 0.5);
close(one.rgb[0], 0.5);

const redBlue = compose([red, blue]);
const blueRed = compose([blue, red]);
close(redBlue.alpha, 0.75);
close(blueRed.alpha, 0.75);
assert.deepEqual(redBlue.rgb, [0.5, 0, 0.25]);
assert.deepEqual(blueRed.rgb, [0.25, 0, 0.5]);

const overWhite = compose([red], [1, 1, 1]);
assert.deepEqual(overWhite.rgb, [1, 0.5, 0.5]);
const derivedWhite = one.rgb.map((value) => value + (1 - one.alpha));
assert.deepEqual(derivedWhite, overWhite.rgb);

const layers = [
  { color: [0.8, 0.2, 0.1], opacity: 0.4, kernel: 0.7 },
  { color: [0.1, 0.5, 0.9], opacity: 0.6, kernel: 0.8 },
];
const epsilon = 1e-5;
const base = compose(layers).rgb[0];
const opacityPlus = structuredClone(layers);
opacityPlus[0].opacity += epsilon;
const numericOpacity = (compose(opacityPlus).rgb[0] - base) / epsilon;
const alpha0 = layers[0].opacity * layers[0].kernel;
const behindRed = layers[1].opacity * layers[1].kernel * layers[1].color[0];
const analyticOpacity = layers[0].kernel * (layers[0].color[0] - behindRed);
close(numericOpacity, analyticOpacity, 1e-5);

const colorPlus = structuredClone(layers);
colorPlus[0].color[0] += epsilon;
const numericColor = (compose(colorPlus).rgb[0] - base) / epsilon;
close(numericColor, alpha0, 1e-8);

for (const orderedLayers of [layers, [...layers].reverse()]) {
  const targetRgb = [0.3, 0.4, 0.2];
  const targetAlpha = 0.93;
  const analytic = exactReverse(orderedLayers, targetRgb, targetAlpha);
  for (let index = 0; index < orderedLayers.length; index += 1) {
    const opacityVariant = structuredClone(orderedLayers);
    opacityVariant[index].opacity += epsilon;
    const numeric = (loss(opacityVariant, targetRgb, targetAlpha) - loss(orderedLayers, targetRgb, targetAlpha)) / epsilon;
    close(numeric, analytic[index].opacity, 2e-5);
    for (let channel = 0; channel < 3; channel += 1) {
      const colorVariant = structuredClone(orderedLayers);
      colorVariant[index].color[channel] += epsilon;
      const numericChannel = (loss(colorVariant, targetRgb, targetAlpha) - loss(orderedLayers, targetRgb, targetAlpha)) / epsilon;
      close(numericChannel, analytic[index].color[channel], 2e-5);
    }
  }
}

{
  const point = [0.17, -0.09];
  const center = [0.03, -0.02];
  const scale = [0.21, 0.08];
  const theta = 0.37;
  const analytic = gaussianWithDerivatives(point, center, scale, theta);
  for (let axis = 0; axis < 2; axis += 1) {
    const plus = [...center];
    const minus = [...center];
    plus[axis] += epsilon;
    minus[axis] -= epsilon;
    const numeric = (gaussianWithDerivatives(point, plus, scale, theta).weighted - gaussianWithDerivatives(point, minus, scale, theta).weighted) / (2 * epsilon);
    close(numeric, analytic.center[axis], 2e-5);
    const scalePlus = [...scale];
    const scaleMinus = [...scale];
    scalePlus[axis] *= Math.exp(epsilon);
    scaleMinus[axis] *= Math.exp(-epsilon);
    const numericLogScale = (gaussianWithDerivatives(point, center, scalePlus, theta).weighted - gaussianWithDerivatives(point, center, scaleMinus, theta).weighted) / (2 * epsilon);
    close(numericLogScale, analytic.logScale[axis], 2e-5);
  }
  const numericTheta = (gaussianWithDerivatives(point, center, scale, theta + epsilon).weighted - gaussianWithDerivatives(point, center, scale, theta - epsilon).weighted) / (2 * epsilon);
  close(numericTheta, analytic.theta, 2e-5);
}

const sourceContracts = {
  standardForward: (app.match(/rendered \+= transmittance \* alpha \*/g) || []).length >= 2,
  compositeAlphaState: (app.match(/let compositeAlpha = 1\.0 - transmittance/g) || []).length >= 2,
  transparentBackgroundAlpha: app.includes("let compositeAlpha = 1.0 - transmittance") && !/compositeAlpha\s*=\s*1\.0\s*;/.test(app),
  graphdecoCutoffs: app.includes("alpha >= 0.0039215686") && app.includes("transmittance < 0.0001"),
  deterministicTileOrder: app.includes("fn sort_tiles(") && app.includes("sift_down(base, count, start)"),
  directSplatBuffers:
    app.includes("let bounds = tile_bounds(g);") &&
    app.includes("var center = xy[g].center;") &&
    app.includes("@group(0) @binding(8) var<storage, read_write> stats: array<vec4<f32>>;"),
  blackTrainingBackground: app.includes("const bg = new Float32Array([0, 0, 0])"),
  trainedPlyOpacity: app.includes("comment image2gaussianpaint_blend standard_alpha"),
  trainedPlyLayerOrder: app.includes("comment image2gaussianpaint_layer_order"),
  externalSortDepthSpan: app.includes("const PLY_LAYER_DEPTH_SPAN = 1e-2"),
  packedLayerTraining: app.includes("var layerOrder = clamp(min(fract(t.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0)"),
  deterministicLayerSort: app.includes("fn tile_less("),
  standaloneLayerOrder:
    app.includes("const useSplatPreviewOrder = Boolean(options.splatSmallFirstOrder)") &&
    app.includes("const useGlobalOrder = !useTileOrder && (Boolean(params.layerOrderEnabled) || useSplatPreviewOrder)") &&
    app.includes("function frontRenderLayerDepth(index, params)") &&
    app.includes("return order * LAYER_CODE_RANGE") &&
    app.includes("frontRenderLayerDepth(b, params) - frontRenderLayerDepth(a, params)") &&
    app.includes("return Math.abs(delta) > 1e-7 ? delta : a - b;"),
  finalFrontConfigReset:
    app.includes("const config = new Float32Array(TRAIN_CONFIG_FLOATS);") &&
    app.includes('async refreshRenderState(image, params, { view = null } = {})') &&
    app.includes('const requestedView = view === "front" ? null : view;') &&
    app.includes("config[56] = virtualView ? 1 : 0;") &&
    app.includes("config[67] = params.virtualDepthEnabled ? 1 : 0;"),
  sharedEwaQuadrature:
    app.includes("useGaussLegendre: f32") &&
    app.includes("uniforms.useGaussLegendre > 0.5"),
  sharedAlphaCutoff: (app.match(/alpha >= 0\.0039215686/g) || []).length >= 3,
  renderSurfaceParity:
    app.includes('source: "training-pixel-state-vs-standalone-rgba"') &&
    app.includes("...displayRgbaParity(trainingFrame.rgba, standaloneRgba)") &&
    !app.includes("rgbaParity(tileRgba, standaloneRgba)"),
  rgbaExportParity:
    app.includes("return vec4f(rgb, 1.0 - transmittance);") &&
    app.includes("f32(px) / f32(width - 1u) * 2.0 - 1.0") &&
    app.includes("rgba[di + 3] = source[si + 3]") &&
    app.includes("rgba[target + 3] = clampByte(pixels[source + 3] * 255)") &&
    app.includes('frame.getContext("2d", { alpha: true })') &&
    app.includes("const pngRgbaParity = displayRgbaParity(rgba, decodedRgba)") &&
    app.includes("alphaMaximum === 0 && premultipliedMaximum <= 1") &&
    app.includes("PNG display RGBA round-trip mismatch"),
  persistedPlyLayerMode: app.includes("const layerOrderEnabled = Boolean(params.layerOrderEnabled)"),
  flatDensifyLayer: app.includes("config[35] > 0.5"),
  noForceMode: !html.includes("forceOpaqueAlpha") && !app.includes("FORCED_ALPHA_TARGET"),
  noClosedFormCalibration: !html.includes("calibratePlyCoverage") && !app.includes("plyCoverageCalibration"),
  imageGsTopKBackwardRetired:
    !/top.?k|image.?gs/i.test(html + app),
  exactBackwardDefault: app.includes('exactBackward: booleanVariant("exactBackward", true)'),
  cooperativeExactBackward: !app.includes('replayCursor: booleanVariant') &&
    app.includes("struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };") &&
    (app.match(/alphaState: array<AlphaState>/g) || []).length >= 5 &&
    app.includes("AlphaState(1.0 - transmittance, acceptedEnd, 0.0, contributorCount)") &&
    app.includes("contributorCount += 1u") &&
    app.includes("alphaState[pixel].acceptedEnd") &&
    app.includes("alphaState[pixel].compositeAlpha") &&
    !app.includes("bitcast<f32>(acceptedEnd)") &&
    app.includes("@compute @workgroup_size(8, 8, 1)") &&
    app.includes("var<workgroup> reduceGeom: array<vec4<f32>, 64>") &&
    app.includes("fn exact_alpha_backward_quad") &&
    app.includes('entryPoint: this.quadExactBackwardEnabled ? "exact_alpha_backward_quad" : "exact_alpha_backward"') &&
    app.includes("this.quadExactBackwardEnabled ? TILE_SIZE : 8"),
  selectiveAnisotropy: app.includes("const DEFAULT_SURFACE_ANISOTROPY = 8") &&
    (app.match(/floor\(t\.w\) >= 2\.0/g) || []).length >= 2 &&
    (app.match(/max\(config\[55\], 1\.0\)/g) || []).length >= 2,
  selectiveAnisotropyReadback: app.includes("surface_anisotropy_max") && app.includes("detail_anisotropy_max"),
  adjustedAnisotropyPolicy: app.includes("anisotropyLimitForTag(baseline.detailTags?.[i], baseline)") &&
    app.includes("anisotropy_limit_violations") &&
    app.includes("plyContract.anisotropy_limit_violations === 0"),
  adaptiveCurriculum: app.includes("const CURRICULUM_COARSE_FRACTION = 1 / 7") &&
    app.includes("const CURRICULUM_DENSITY_FRACTION = 3 / 7") &&
    app.includes("const CURRICULUM_GROWTH_FRACTION = 6 / 7") &&
    app.includes("experimentalCoarseSteps(requestedSteps, variants.coarseSteps)") &&
    app.includes("return curriculumStageStep(steps, CURRICULUM_DENSITY_FRACTION)"),
  reverseTransmittance: app.includes("let transBefore = transAfter / max(1.0 - alpha, 0.01)"),
  exactAtomicGradient: app.includes("atomicCompareExchangeWeak(&exactGradient[index]"),
  exactLossGradient: app.includes('entryPoint: "loss_gradient"'),
  exactOptimizer: app.includes('entryPoint: "optimize_exact"'),
  configurableAlphaWeight: html.includes('id="alphaLossWeight"') && app.includes("DEFAULT_ALPHA_LOSS_WEIGHT = 0.2"),
  dualBackgroundRetired: !html.includes('id="dualBackgroundToggle"') &&
    !html.includes('id="dualBackgroundWeight"') &&
    !app.includes("dTransmittanceExtra") &&
    !app.includes("dualBackgroundGradient"),
  exposureRatioPrimary: html.includes("background / outside") && app.includes("metrics.coverage.background_exposure_ratio * 100"),
  renderParityQa:
    app.includes('reason: "render_parity_mismatch"') &&
    app.includes("typeof parity.display_equivalent") &&
    app.includes("if (!parity.display_equivalent)"),
  productExportIgnoresParity: !app
    .slice(app.indexOf("async function saveExport("), app.indexOf("function inspectPlyContract("))
    .includes("exportCoverageStatus()"),
  profileQueryCapacityGuard: app.includes("PERFORMANCE_PROFILE_QUERY_CAPACITY = 32") &&
    app.includes("Performance profile query capacity exceeded"),
};

assert.ok(Object.values(sourceContracts).every(Boolean), JSON.stringify(sourceContracts));
console.log(JSON.stringify({ ok: true, sourceContracts }, null, 2));
