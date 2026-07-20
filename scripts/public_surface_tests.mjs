import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const [app, index, styles, readme, ignore, license, notices, tiltViewer, tiltBundle, tiltCamera, vendorLicense, buildScript, sourceImagesReadme] = await Promise.all([
  readFile(new URL("web/app.js", root), "utf8"),
  readFile(new URL("web/index.html", root), "utf8"),
  readFile(new URL("web/styles.css", root), "utf8"),
  readFile(new URL("README.md", root), "utf8"),
  readFile(new URL(".gitignore", root), "utf8"),
  readFile(new URL("LICENSE", root), "utf8"),
  readFile(new URL("THIRD_PARTY_NOTICES.md", root), "utf8"),
  readFile(new URL("web/tilt-viewer.mjs", root), "utf8"),
  readFile(new URL("web/tilt-viewer.bundle.js", root), "utf8"),
  readFile(new URL("web/tilt-camera.mjs", root), "utf8"),
  readFile(new URL("web/vendor/PLAYCANVAS-LICENSE.txt", root), "utf8"),
  readFile(new URL("scripts/build-pages.mjs", root), "utf8"),
  readFile(new URL("assets/source-images/README.md", root), "utf8"),
]);

const retired = [
  "Reduce oversized splats", "Confirmed tangent chains", "Symmetric long-axis split",
  "Late subpixel detail", "Reset density stats per batch", "ADC protect fine detail",
  "ADC prioritize fine detail", "areaRegularization", "confirmed_chain_at",
  "apply_symmetric_parents", "adcProtectDetail", "adcPrioritizeDetail",
];

const legacyPaths = ["pyproject.toml", "src", "tests", "viewer", "build", "web/export-formats.mjs"];
const absent = async (path) => access(new URL(path, root)).then(() => false, () => true);

function functionSource(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} missing`);
  const brace = source.indexOf("{", start);
  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    if (source[i] === "{") depth += 1;
    if (source[i] === "}") depth -= 1;
    if (depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`${name} body is incomplete`);
}

const transformPlanarSplatForPly = new Function(
  `${functionSource(app, "plyFrameScale")}\n${functionSource(app, "transformPlanarSplatForPly")}\nreturn transformPlanarSplatForPly;`,
)();
const landscapeProbe = transformPlanarSplatForPly(0.5, 0.5, 0.2, 0.1, 0, { width: 800, height: 400 });
assert.ok(Math.abs(landscapeProbe.x - 0.5) < 1e-12);
assert.ok(Math.abs(landscapeProbe.y + 0.25) < 1e-12);
assert.ok(Math.abs(landscapeProbe.sx - 0.2) < 1e-12);
assert.ok(Math.abs(landscapeProbe.sy - 0.05) < 1e-12);

const theta = Math.PI / 4;
const rotatedProbe = transformPlanarSplatForPly(0, 0, 0.2, 0.1, theta, { width: 800, height: 400 });
const reconstructed = {
  xx: Math.cos(rotatedProbe.theta) ** 2 * rotatedProbe.sx ** 2 + Math.sin(rotatedProbe.theta) ** 2 * rotatedProbe.sy ** 2,
  yy: Math.sin(rotatedProbe.theta) ** 2 * rotatedProbe.sx ** 2 + Math.cos(rotatedProbe.theta) ** 2 * rotatedProbe.sy ** 2,
  xy: Math.cos(rotatedProbe.theta) * Math.sin(rotatedProbe.theta) * (rotatedProbe.sx ** 2 - rotatedProbe.sy ** 2),
};
const expected = {
  xx: 0.5 * (0.2 ** 2 + 0.1 ** 2),
  yy: 0.25 * 0.5 * (0.2 ** 2 + 0.1 ** 2),
  xy: -0.5 * 0.5 * (0.2 ** 2 - 0.1 ** 2),
};
assert.ok(Math.max(
  Math.abs(reconstructed.xx - expected.xx),
  Math.abs(reconstructed.yy - expected.yy),
  Math.abs(reconstructed.xy - expected.xy),
) < 1e-12);

const scaleSplatLocalAspectRatio = new Function(
  `const MIN_SPLAT_SCALE = 0.0015;\n${functionSource(app, "scaleSplatLocalAspectRatio")}\nreturn scaleSplatLocalAspectRatio;`,
)();
const aspectProbe = scaleSplatLocalAspectRatio(0.2, 0.1, 4);
assert.ok(Math.abs(aspectProbe.sx - 0.4) < 1e-12);
assert.ok(Math.abs(aspectProbe.sy - 0.05) < 1e-12);
assert.ok(Math.abs(aspectProbe.sx * aspectProbe.sy - 0.02) < 1e-12);

const splatPreviewOrderComparator = new Function(
  `const MIN_SPLAT_SCALE = 0.0015; const PLY_LAYER_DEPTH_SPAN = 1e-2; const DEFAULT_VIRTUAL_DEPTH_THICKNESS = 0.0025;\n${functionSource(app, "boundedVirtualDepth")}\n${functionSource(app, "plyLayerDepth")}\n${functionSource(app, "layerOrderComparator")}\n${functionSource(app, "splatPreviewOrderComparator")}\nreturn splatPreviewOrderComparator;`,
)();
const orderProbe = {
  count: 2,
  opacity: new Float32Array([0.5, 0.5]),
  scale: new Float32Array([0.1, 0.1, 0.3, 0.3]),
  depthOrder: new Float32Array([0.5, 0.5]),
  layerOrderEnabled: false,
  virtualDepthEnabled: false,
};
assert.ok(splatPreviewOrderComparator(0, 1, orderProbe) < 0, "equal opacity must place the smaller splat first");
orderProbe.opacity[1] = 0.9;
assert.ok(splatPreviewOrderComparator(1, 0, orderProbe) < 0, "higher opacity must be composited first");

const initialSplatOrientation = new Function(
  `${functionSource(app, "initialSplatOrientation")}\nreturn initialSplatOrientation;`,
)();
const orientationBins = new Array(8).fill(0);
for (let index = 0; index < 2048; index += 1) {
  const angle = initialSplatOrientation(index);
  assert.ok(Number.isFinite(angle) && angle >= -Math.PI / 2 && angle < Math.PI / 2);
  orientationBins[Math.min(7, Math.floor(((angle + Math.PI / 2) / Math.PI) * 8))] += 1;
}
assert.ok(Math.max(...orientationBins) - Math.min(...orientationBins) <= 2);
const spatialOrientationBins = new Array(8).fill(0);
for (let index = 0; index < 128; index += 1) {
  const angle = initialSplatOrientation(index, 14);
  spatialOrientationBins[Math.min(7, Math.floor(((angle + Math.PI / 2) / Math.PI) * 8))] += 1;
}
assert.ok(Math.max(...spatialOrientationBins) - Math.min(...spatialOrientationBins) <= 2);

const contracts = {
  defaultIterations:
    /id="stepCount"[^>]*value="3000"/.test(index) &&
    app.includes("const DEFAULT_ITERATIONS = 3000") &&
    (app.match(/fallback: DEFAULT_ITERATIONS/g)?.length || 0) >= 3,
  retiredProductCodeAbsent: retired.every((term) => !app.includes(term) && !index.includes(term) && !readme.includes(term)),
  retainedControls: index.includes("Tile/subtile culling"),
  experimentalGpuPerformanceControls:
    /id="opacitySupportAggressive"[^>]*type="checkbox"(?![^>]*checked)/.test(index) &&
    /id="subgroupSyncReduction"[^>]*type="checkbox"(?![^>]*checked)/.test(index) &&
    !index.includes("opacitySupportFourSigma") &&
    !index.includes("Opacity modes are mutually exclusive") &&
    app.includes("syncExperimentalPerformanceControls") &&
    app.includes("configureExperimentalPerformance") &&
    app.includes("subgroupSyncReductionLines") &&
    app.includes("opacity_support_q"),
  brandAboveTabs:
    index.indexOf('<div class="brand controls-brand">') <
      index.indexOf('<nav class="log-tabs"') &&
    index.indexOf('<nav class="log-tabs"') <
      index.indexOf('id="trainingLogPanel"') &&
    index.match(/<h1>Image2SplatPaint<\/h1>/g)?.length === 1 &&
    styles.includes("grid-template-rows: auto 28px minmax(0, 1fr)"),
  adaptiveDetailAlwaysOn:
    !index.includes("adaptiveDetailToggle") &&
    !index.includes("Adaptive thin details") &&
    !app.includes("els.adaptiveDetailToggle") &&
    app.includes("adaptiveDetail: true"),
  publicLicenses:
    license.includes("MIT License") &&
    license.includes("Copyright (c) 2026 Mistral-Yu") &&
    readme.includes("[MIT License](LICENSE)") &&
    readme.includes("PlayCanvas Engine") &&
    readme.includes("PlayCanvas Engine](https://github.com/playcanvas/engine) 2.20.6") &&
    readme.includes("[Third-Party Notices](THIRD_PARTY_NOTICES.md)") &&
    notices.includes("PlayCanvas Engine 2.20.6") &&
    notices.includes("MIT License") &&
    vendorLicense.includes("Copyright (c) 2011-2026 PlayCanvas Ltd.") &&
    index.includes('href="../LICENSE"') &&
    index.includes('href="../THIRD_PARTY_NOTICES.md"'),
  exportGuidanceAndSharedFooter:
    index.includes("PLY uses standard SH0 Gaussian Splatting fields.") &&
    index.includes("For conventional multi-angle 3DGS viewing, train with GS Virtual Camera Sampling") &&
    app.includes("For conventional multi-angle 3DGS viewing, train with GS Virtual Camera Sampling") &&
    !readme.includes("sRGB colors") &&
    !readme.includes("Initial splat colors come from the image") &&
    !readme.includes("With `GS Virtual Camera Sampling`, inspect the in-memory PLY") &&
    !readme.includes("Use `GS Virtual Camera Sampling` for conventional multi-angle 3DGS viewing") &&
    !/Graphdeco/i.test(`${index}\n${app}\n${readme}`) &&
    !/[ぁ-んァ-ヶ一-龯]/.test(`${index}\n${app}`) &&
    (index.match(/data-testid="app-footer"/g)?.length || 0) === 1 &&
    index.indexOf('data-testid="app-footer"') > index.indexOf('id="exportPanel"') &&
    !index.slice(index.indexOf('id="exportPanel"'), index.indexOf('data-testid="app-footer"')).includes('class="license-links"') &&
    index.includes("Copyright (c) 2026 Mistral-Yu"),
  localEvidenceIgnored: ["checkpoints/", "doc/", "runbooks/", "outputs/", "references/", "AGENTS.md"]
    .every((entry) => ignore.split(/\r?\n/).includes(entry)),
  publicImageAllowlist:
    ignore.includes("assets/source-images/*") &&
    [
      "!assets/source-images/README.md",
      "!assets/source-images/ramen-photo.jpg",
      "!assets/source-images/generated-geometric-sample.jpg",
    ].every((entry) => ignore.split(/\r?\n/).includes(entry)) &&
    buildScript.includes('["README.md", "ramen-photo.jpg", "generated-geometric-sample.jpg"]') &&
    !/hair|portrait|attached-photo|写真1/i.test(`${buildScript}\n${sourceImagesReadme}`),
  imageSizeStatus:
    index.includes('title="Training image size / original image size"') &&
    app.includes('`${state.image.width}x${state.image.height} / ${originalWidth}x${originalHeight}`') &&
    app.includes("state.image.originalWidth || state.image.width") &&
    app.includes("state.image.originalHeight || state.image.height") &&
    /@media \(max-width: 520px\)[\s\S]*?\.progress\s*\{\s*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/.test(styles),
  privateIdentityAbsent:
    !/\/Users\/mistralyu|\/home\/mistralyu|@[a-z0-9.-]+\.(?:com|jp)\b/i.test(`${app}\n${index}\n${styles}\n${readme}\n${buildScript}`) &&
    !/(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*["'][^"']+/i.test(`${app}\n${index}\n${readme}\n${buildScript}`),
  experimentalScriptsIgnored: [
    "scripts/*",
    "!scripts/build-pages.mjs",
    "!scripts/pages_artifact_tests.mjs",
    "!scripts/public_surface_tests.mjs",
    "!scripts/standard_alpha_tests.mjs",
    "!scripts/tilt_camera_tests.mjs",
    "!scripts/virtual_tilt_tests.mjs",
    "!scripts/adaptive_capacity_tests.mjs",
    "!scripts/export_cross_platform_tests.mjs",
    "!scripts/gpu_memory_status_tests.mjs",
    "!scripts/independent_ply_parser.mjs",
    "!scripts/independent_ply_parser_tests.mjs",
    "!scripts/large_dispatch_tests.mjs",
    "!scripts/p2_budget_contract_tests.mjs",
    "!scripts/p2_export_memory_tests.mjs",
    "!scripts/p2_optimizer_input_tests.mjs",
    "!scripts/static-server-lib.mjs",
    "!scripts/static-server.mjs",
  ].every((entry) => ignore.split(/\r?\n/).includes(entry)),
  gaussianFusionRetired:
    !/gaussianFusion|Gaussian fusion|gaussian_fusion|ensureFusionPipelines|fuseExperimentalGpu|syncExperimentalOptimizationDependency/.test(`${app}\n${index}`),
  cachePrecomputeRetired:
    !/geometryPrecompute|geometry-cache|geometry_precompute|GEOMETRY_PRECOMPUTE|geometryBuffer|SplatGeometry/.test(app),
  legacyPathsRemoved: (await Promise.all(legacyPaths.map(absent))).every(Boolean),
  pagesPublicSurface:
    readme.includes("generated geometric Sample image") &&
    readme.includes("author-owned ramen sample") &&
    readme.includes("assets/source-images/README.md") &&
    readme.includes("https://mistral-yu.github.io/image2splatpaint/") &&
    !readme.includes("## Compatibility") &&
    readme.includes("## TODO") &&
    readme.includes("Improve training methods for paint-oriented effects.") &&
    readme.includes("Improve compatibility with conventional 3D Gaussian Splatting workflows.") &&
    sourceImagesReadme.includes("`ramen-photo.jpg` as an additional sample"),
  directFileLaunch:
    index.includes("img-src 'self' file: data: blob:") &&
    index.includes('<script src="./tilt-viewer.bundle.js?v=camera-summary-v2"></script>') &&
    index.includes('<script src="./sample-image-data.js?v=sample-v1"></script>') &&
    index.includes('<script src="./app.js?') &&
    app.includes("EMBEDDED_SAMPLE_IMAGE_URL") &&
    app.includes("globalThis.Image2SplatPaintTilt") &&
    !app.includes("Tilt rendering requires GitHub Pages or a local HTTP server.") &&
    tiltBundle.includes("PlayCanvas Engine v2.20.6") &&
    tiltBundle.includes("Image2SplatPaintTilt") &&
    app.includes("runVirtualCameraSampling.enabled && runVirtualCameraSampling.autoCameraDistance") &&
    app.includes("await state.previewRefreshPromise") &&
    app.includes("Training failed:") &&
    !index.includes('data-testid="training-status"') &&
    app.includes('if (status && els.statusText.textContent !== status) setStatus(status)'),
  reliableTrainingControls:
    index.includes('data-testid="training-log-tab"') &&
    index.includes('aria-selected="true">Train</button>') &&
    app.includes("state.startPending") &&
    app.includes("state.sampleLoading") &&
    app.includes("Image load failed:") &&
    !styles.includes(".training-status"),
  splatPreviewOrdering:
    /id="splatSmallFirstOrder"[^>]*type="checkbox"(?![^>]*checked)/.test(index) &&
    app.includes("function splatPreviewOrderComparator") &&
    app.includes("splatSmallFirstOrder: effectsEnabled && Boolean(els.splatSmallFirstOrder?.checked)") &&
    app.includes("? splatPreviewOrderComparator(a, b, params)"),
  outsidePreviewToggleDuringAdjustment:
    app.includes("const outsidePreviewReady = Boolean(") &&
    app.includes("state.image &&") &&
    app.includes("state.webgpu.renderer &&") &&
    app.includes("els.outsidePreviewToggle.disabled = !outsidePreviewReady;") &&
    !app.includes("els.outsidePreviewToggle.disabled = state.previewRefreshPending || !outsidePreviewReady;"),
  adaptiveGridInitialization:
    /id="adaptiveGridInitializationFraction"[^>]*type="number"[^>]*min="0"[^>]*max="50"[^>]*value="25"/.test(index) &&
    index.includes("Adaptive placement (%)") &&
    index.indexOf('id="adaptiveGridInitializationFraction"') > index.indexOf('<summary>Learning rates</summary>') &&
    app.includes("function adaptiveGridInitializationVariants") &&
    app.includes("requested = typeof overrides.enabled === \"boolean\" ? overrides.enabled : true") &&
    app.includes("Number(els.adaptiveGridInitializationFraction?.value) / 100") &&
    app.includes("async applyAdaptiveGridInitialization(image, params, variants)") &&
    app.includes("parameter_scope: \"center-and-image-rgb-only\"") &&
    app.includes("const runAdaptiveGridInitialization = adaptiveGridInitializationVariants") &&
    app.includes("await state.webgpu.renderer.applyAdaptiveGridInitialization(") &&
    app.includes("initialization_adaptive") &&
    app.includes("benchmarkSummary()"),
  splatKernelFalloffPreviewOnly:
    index.includes('data-testid="splat-kernel-falloff"') &&
    app.includes("kernelFalloff: effectsEnabled ? clampNumber(els.splatKernelFalloff?.value, 0, 2, 1) : 1") &&
    app.includes("kernel = pow(kernel, clamp(uniforms.kernelFalloff, 0.0, 2.0));") &&
    app.includes("Keep the trained 4σ footprint") &&
    !app.includes("kernelFalloff: learningRates"),
  splatShapePreviewOnly:
    /id="splatParameterEffects"[^>]*type="checkbox"[^>]*checked/.test(index) &&
    /id="splatShapeGaussian"[^>]*class="active"[^>]*aria-pressed="true"/.test(index) &&
    /id="splatShapeRectangle"[^>]*aria-pressed="false"/.test(index) &&
    /id="splatSmallFirstOrder"[^>]*type="checkbox"(?![^>]*checked)/.test(index) &&
    app.includes('splatShape: effectsEnabled ? state.splatPreviewShape : "gaussian"') &&
    app.includes("uniforms.shapeMode > 0.5") &&
    app.includes("let radius = max(local.x, local.y)") &&
    app.includes("exp(-0.5 * radius * radius), radius <= 4.0") &&
    app.includes("Shape changes only the footprint kernel") &&
    app.includes('const shapeMode = options.splatShape === "rectangle" ? 1 : 0;') &&
    app.includes('data.splatPreviewShape = els.splatParameterEffects?.checked ? state.splatPreviewShape : "gaussian";') &&
    !app.includes("splatShape: learningRates"),
  schedulingProfile:
    app.includes("function summarizeTrainingScheduling") &&
    app.includes("timestamp_gpu_passes_ms") &&
    app.includes("runtime_wall_by_phase_ms") &&
    app.includes("state.metrics.scheduling_profile = summarizeTrainingScheduling(") &&
    app.includes("scheduling_profile: m.scheduling_profile || null"),
  responsiveMobileViewer:
    styles.includes("@media (max-width: 880px)") &&
    styles.includes("height: clamp(420px, 75dvh, 760px)"),
  unbiasedInitialOrientation:
    app.includes("function initialSplatOrientation") &&
    app.includes('scheme: "deterministic-spatially-decorrelated-world-angle"') &&
    app.includes("initialSplatOrientation(index, layout.cols)") &&
    app.includes("const shape = initialSplatShape(image, layout, i)") &&
    app.includes("theta[i] = shape.theta") &&
    app.includes("initial_orientation: m.initial_orientation") &&
    !app.includes("theta[i] = 0") &&
    Math.max(...orientationBins) - Math.min(...orientationBins) <= 2 &&
    Math.max(...spatialOrientationBins) - Math.min(...spatialOrientationBins) <= 2,
  phaseOneFullGeometry:
    app.includes("const PHASE_ONE_MAX_PLANAR_SCALE = 0.32") &&
    app.includes("const PHASE_ONE_SHAPE_LR_MULTIPLIER = 2.5") &&
    app.includes('config[79] = trainingStage === "coarse" ? PHASE_ONE_SHAPE_LR_MULTIPLIER : 1') &&
    app.match(/scaleLr = cfg\(13u\).*phaseOneShapeBoost/g)?.length === 2 &&
    app.match(/rotationLr = cfg\(14u\).*phaseOneShapeBoost/g)?.length === 2,
  algorithmRegistry:
    index.includes('id="algorithmSelect"') &&
    index.includes('<option value="planar-gaussian" selected>Planar Gaussian</option>') &&
    index.includes('<option value="gs-virtual-camera-sampling">GS Virtual Camera Sampling</option>') &&
    app.includes("const ALGORITHM_REGISTRY") &&
    app.includes("train: trainPlanarGaussian") &&
    app.includes("train: trainGsVirtualCameraSampling") &&
    app.includes("trainGaussianAlgorithm(false)") &&
    app.includes("trainGaussianAlgorithm(true)") &&
    app.includes("await selectedAlgorithm().train()") &&
    app.includes("algorithm.initialize(state.image, initialCount)") &&
    app.includes("algorithmSupportsExport(formatKey)") &&
    app.includes('data.runAlgorithm = state.metrics.algorithm || ""') &&
    app.includes('data.runAlgorithm = ""') &&
    !index.includes("ADC original") &&
    !index.includes("3DGS-MCMC"),
  publicQaRestricted:
    app.includes("const QA_RUNTIME_ENABLED") &&
    app.includes('qaOverrides("__flatPhotoPhase40")') &&
    app.includes("if (QA_RUNTIME_ENABLED) window.__flatPhotoTest"),
  automaticSplatFloor:
    app.includes("DEFAULT_INITIAL_SPLATS = 128") &&
    app.includes("AUTO_INITIAL_SPLATS_MIN = 128") &&
    app.includes("AUTO_FINAL_SPLATS_MIN = 3000") &&
    app.includes("previousEstimate / 2") &&
    app.includes("Math.min(final, AUTO_INITIAL_SPLATS_MIN)"),
  splatParameterUi:
    app.includes("const DEFAULT_MAX_SIDE = 512") &&
    index.includes('data-testid="train-size" type="number" min="32" max="3200" step="64" value="512"') &&
    index.includes("<strong>Splat parameters</strong>") &&
    index.includes('data-testid="splat-scale" type="range" min="0" max="5"') &&
    index.includes('data-testid="splat-aspect-ratio" type="range" min="0.1" max="10"') &&
    app.includes("function scaleSplatLocalAspectRatio") &&
    !index.includes('id="splatRotation"') &&
    !app.includes("els.splatRotation") &&
    !app.includes("rotationDegrees"),
  imageGsTopKBackwardRetired:
    !/top.?k|image.?gs/i.test(app + index),
  finalSplatCommitBehavior:
    app.includes('els.finalSplatCount.addEventListener("input"') &&
    app.includes("reconcileSplatCounts: false") &&
    app.includes('els.finalSplatCount.addEventListener("change"'),
  allocationAlwaysValidated:
    app.includes('this.device.pushErrorScope("out-of-memory")') &&
    !app.includes('if (verifyAllocation) {\n      this.device.pushErrorScope("out-of-memory")'),
  aspectPreservingPly:
    app.includes("transformPlanarSplatForPly") &&
    app.includes("single NDC-to-isotropic-world conversion") &&
    app.includes("comment image2gaussianpaint_frame") &&
    app.includes("aspect_ratio_preserved") &&
    app.includes("opacity_error_max") &&
    app.includes("color_error_max"),
  standardAlphaProduct:
    app.includes("comment image2gaussianpaint_blend standard_alpha") &&
    app.includes("comment image2gaussianpaint_layer_order") &&
    app.includes("standard_alpha_blend") &&
    app.includes("rendered += transmittance * alpha") &&
    app.includes("let compositeAlpha = 1.0 - transmittance") &&
    app.includes("fn sort_tiles(") &&
    app.includes("alpha < 0.99") === false &&
    !app.includes("comment image2gaussianpaint_blend normalized_weighted") &&
    !index.includes("forceOpaqueAlpha") &&
    !index.includes("calibratePlyCoverage") &&
    !app.includes("function plyCoverageCalibration"),
  trainableLayerOrder:
    index.includes('data-testid="train-layer-order"') &&
    app.includes("var layerOrder = clamp(min(fract(t.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0)") &&
    app.includes("fn tile_less(") &&
    app.includes('source: "training-pixel-state-vs-standalone-rgba"') &&
    app.includes("...displayRgbaParity(trainingFrame.rgba, standaloneRgba)") &&
    app.includes("Boolean(params.layerOrderEnabled)"),
  virtualCameraSampling:
    index.includes('data-testid="virtual-camera-panel"') &&
    index.includes('data-testid="virtual-camera-settings"') &&
    !index.includes('data-testid="virtual-camera-sampling"') &&
    index.includes('data-testid="virtual-camera-share"') &&
    index.includes('data-testid="virtual-camera-max-angle"') &&
    index.includes('data-testid="virtual-camera-count"') &&
    index.includes('data-testid="virtual-camera-fov"') &&
    index.includes('data-testid="virtual-gof-density" type="checkbox" checked') &&
    !index.includes('data-testid="virtual-front-sharpness"') &&
    !index.includes('data-testid="virtual-front-canonical"') &&
    !index.includes('data-testid="virtual-source-domain"') &&
    index.includes('data-testid="virtual-camera-coverage"') &&
    index.includes('Virtual camera share (%)') &&
    index.includes('GS Virtual Camera Sampling settings') &&
    index.includes('data-testid="virtual-camera-share" type="number" min="1" max="100" step="0.125" value="100"') &&
    index.includes('data-testid="virtual-camera-max-angle" type="number" min="5" max="75" step="5" value="60"') &&
    index.includes('data-testid="virtual-camera-count" type="number" min="4" max="128" step="1" value="50"') &&
    index.indexOf('data-testid="lr-panel"') < index.indexOf('data-testid="virtual-camera-panel"') &&
    index.indexOf('data-testid="virtual-camera-panel"') < index.indexOf('data-testid="capacity-mode"') &&
    !index.includes('id="virtualCameraSampling"') &&
    app.includes("algorithmUsesVirtualCameras()") &&
    app.includes('return trainGaussianAlgorithm(false)') &&
    app.includes('return trainGaussianAlgorithm(true)') &&
    app.includes('Tilt is available only for GS Virtual Camera Sampling.') &&
    app.includes("const DEFAULT_VIRTUAL_CAMERA_SHARE_PERCENT = 100") &&
    app.includes("const DEFAULT_VIRTUAL_CAMERA_COUNT = 50") &&
    app.includes("const DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES = 60") &&
    app.includes("const MAX_VIRTUAL_CAMERA_COUNT = 128") &&
    app.includes('mode: runVirtualCameraSampling.enabled ? runVirtualCameraSampling.mode : "off"') &&
    app.includes("virtualCameraSamplingStepSpec") &&
    app.includes('virtual_camera_gradient_routing: "selected-view-all"') &&
    app.includes('virtual_camera_3dgs_multiview: true') &&
    app.includes('gradientChannels: { geometry: 1, appearance: 1, density: 1, depth: 1 }') &&
    app.includes('cameraCovariance3d: virtualStep') &&
    app.includes('fn project_planar_gaussian(center: vec2<f32>, layerZ: f32, t: vec4<f32>)') &&
    app.includes("data.virtualCameraFrontSteps") &&
    app.includes("data.virtualCameraVirtualSteps") &&
    app.includes("virtualCameraCoverageStats") &&
    index.includes('data-testid="tilt-camera-markers"') &&
    index.includes('data-testid="tilt-camera-summary"') &&
    tiltViewer.includes("setCameraMarkersVisible") &&
    tiltViewer.includes("frontSampleCount") &&
    tiltViewer.includes("virtualSampleCount") &&
    app.includes("tiltCameraVirtualMarkerCount") &&
    app.includes("tiltCameraRunSamplingEnabled") &&
    app.includes("runtime-step-history") &&
    app.includes("deterministic-schedule") &&
    app.includes("Virtual sampling is ON for the next Train.") &&
    app.includes("Virtual camera sampling was off for this run.") &&
    tiltCamera.includes("trainingCameraMarkerGeometry"),
  trainingResultTabsLocked:
    app.includes("const resultTabsLocked = state.running") &&
    app.includes("els.splatsTab.disabled = resultTabsLocked") &&
    app.includes("els.exportTab.disabled = resultTabsLocked") &&
    app.includes('state.running && (name === "splats" || name === "export")') &&
    app.includes('data.resultTabsLocked = String(resultTabsLocked)'),
  aspectAwareTrainingGrid:
    app.includes("function aspectAwareGridEnabled") &&
    app.includes("const INITIAL_SPLAT_COVERAGE_MULTIPLIER = 2.0") &&
    app.includes("INITIAL_SPLAT_COVERAGE_MULTIPLIER / cols") &&
    app.includes("INITIAL_SPLAT_COVERAGE_MULTIPLIER / rows") &&
    app.includes("layout.baseScaleX") &&
    app.includes("layout.baseScaleY") &&
    app.includes("size: TRAIN_CONFIG_BYTES"),
  metricDisplayParity:
    app.includes("function syncDisplayedSsimMetrics") &&
    app.includes("state.metrics?.final_global_ssim") &&
    app.includes("state.metrics?.final_regional_ssim?.p10") &&
    app.includes("syncDisplayedSsimMetrics();"),
  simpleExports:
    !index.includes("Compressed PLY") &&
    !index.includes("PlayCanvas SOG") &&
    !app.includes("compressed-ply") &&
    !app.includes('formatKey !== "ply"') &&
    index.indexOf('id="savePlyButton"') < index.indexOf('id="savePngButton"') &&
    index.includes("Save Splat PNG") &&
    app.includes('filename: `image2splatpaint-splats-${shape}.png`') &&
    app.includes("splat_shape: spec.shape") &&
    app.includes("outside_image: Boolean(spec.renderOptions.outside)"),
  tiltViewer:
    index.indexOf('id="eventLogTab"') < index.indexOf('id="tiltTab"') &&
    !index.includes('aria-controls="tiltPanel" aria-selected="false" disabled') &&
    index.includes('id="tiltCanvas"') &&
    index.includes('<script src="./tilt-viewer.bundle.js?v=camera-summary-v2"></script>') &&
    app.includes("const { createTiltViewer } = tiltViewerRuntime()") &&
    app.includes('destroyTiltViewer({ restoreCanvas: true })') &&
    tiltViewer.includes('from "./vendor/playcanvas-2.20.6.min.mjs"') &&
    tiltViewer.includes('new Application(canvas') &&
    tiltViewer.includes('splat.addComponent("gsplat"') &&
    !tiltViewer.includes("splat.setEulerAngles") &&
    tiltViewer.includes('scene.once("gsplat:sorted"') &&
    tiltViewer.includes("setTiltAndWait") &&
    tiltViewer.includes("plyDigest") &&
    app.includes("PLY bytes differ from the current Tilt view") &&
    app.includes("state.tilt.verifiedRevision === currentTiltRevision()") &&
    app.includes("els.tiltTab.disabled = !tiltAvailable") &&
    app.includes('state.metrics?.algorithm === GS_VIRTUAL_CAMERA_ALGORITHM_ID') &&
    app.includes('name === "tilt" && !algorithmUsesVirtualCameras()') &&
    app.includes("function tiltViewerAvailabilityMessage") &&
    !tiltViewer.includes("app.autoRender = true") &&
    tiltViewer.includes("gammaCorrection: GAMMA_SRGB") &&
    tiltViewer.includes("toneMapping: TONEMAP_NONE") &&
    !tiltViewer.includes("app.scene.gammaCorrection") &&
    !tiltViewer.includes("app.scene.toneMapping"),
  fibonacciHemisphere:
    tiltCamera.includes("TILT_MAX_ANGLE_DEGREES = 75") &&
    tiltCamera.includes("FIBONACCI_HEMISPHERE_POSE_COUNT = 49") &&
    tiltCamera.includes("FIBONACCI_GOLDEN_ANGLE_RADIANS") &&
    index.includes('data-testid="tilt-training-views-button"') &&
    index.includes('data-testid="tilt-teacher-canvas"') &&
    index.includes('data-testid="tilt-original-view"') &&
    index.includes('data-testid="tilt-overlay-view"') &&
    index.includes('data-testid="tilt-splats-view"') &&
    index.includes('data-testid="tilt-contact-sheet"') &&
    !index.includes("Stop sweep") &&
    app.includes("showTiltTrainingViews") &&
    app.includes("rasterizeVirtualTeacher") &&
    app.includes("setTiltDisplayMode") &&
    !app.includes("startTiltHemisphereSweep"),
  renamedProduct:
    index.includes("<title>Image2SplatPaint</title>") &&
    app.includes('filename: "image2splatpaint.ply"') &&
    app.includes('filename: "image2splatpaint-splats.png"') &&
    app.includes('window.__image2SplatPaint =') &&
    app.includes("window.__image2GaussianPaint = window.__image2SplatPaint") &&
    app.includes("comment image2gaussianpaint_frame"),
};

assert.ok(Object.values(contracts).every(Boolean), JSON.stringify(contracts));
console.log(JSON.stringify({ ok: true, contracts }, null, 2));
