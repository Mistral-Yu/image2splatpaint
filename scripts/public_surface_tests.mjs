import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const [app, index, styles, readme, ignore, license, notices, tiltViewer, tiltCamera, vendorLicense] = await Promise.all([
  readFile(new URL("web/app.js", root), "utf8"),
  readFile(new URL("web/index.html", root), "utf8"),
  readFile(new URL("web/styles.css", root), "utf8"),
  readFile(new URL("README.md", root), "utf8"),
  readFile(new URL(".gitignore", root), "utf8"),
  readFile(new URL("LICENSE", root), "utf8"),
  readFile(new URL("THIRD_PARTY_NOTICES.md", root), "utf8"),
  readFile(new URL("web/tilt-viewer.mjs", root), "utf8"),
  readFile(new URL("web/tilt-camera.mjs", root), "utf8"),
  readFile(new URL("web/vendor/PLAYCANVAS-LICENSE.txt", root), "utf8"),
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

const contracts = {
  retiredProductCodeAbsent: retired.every((term) => !app.includes(term) && !index.includes(term) && !readme.includes(term)),
  retainedControls: index.includes("Tile/subtile culling"),
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
  localEvidenceIgnored: ["checkpoints/", "doc/", "runbooks/", "outputs/", "references/", "AGENTS.md"]
    .every((entry) => ignore.split(/\r?\n/).includes(entry)),
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
    "!scripts/p1_webgpu_probe.html",
    "!scripts/p1_webgpu_probe.mjs",
    "!scripts/static-server-lib.mjs",
    "!scripts/static-server.mjs",
  ].every((entry) => ignore.split(/\r?\n/).includes(entry)),
  legacyPathsRemoved: (await Promise.all(legacyPaths.map(absent))).every(Boolean),
  pagesPublicSurface:
    readme.includes("author-owned ramen sample") &&
    readme.includes("assets/source-images/README.md") &&
    readme.includes("https://mistral-yu.github.io/image2splatpaint/"),
  algorithmRegistry:
    index.includes('id="algorithmSelect"') &&
    index.includes('<option value="planar-gaussian" selected>Planar Gaussian</option>') &&
    app.includes("const ALGORITHM_REGISTRY") &&
    app.includes("return selectedAlgorithm().train()") &&
    app.includes("algorithm.initialize(state.image, initialCount)") &&
    app.includes("algorithmSupportsExport(formatKey)") &&
    !index.includes("ADC original") &&
    !index.includes("3DGS-MCMC"),
  publicQaRestricted:
    app.includes("const QA_RUNTIME_ENABLED") &&
    app.includes('qaOverrides("__flatPhotoPhase40")') &&
    app.includes("if (QA_RUNTIME_ENABLED) window.__flatPhotoTest"),
  automaticSplatFloor:
    app.includes("AUTO_INITIAL_SPLATS_MIN = 500") &&
    app.includes("AUTO_FINAL_SPLATS_MIN = 3000") &&
    app.includes("previousEstimate / 2") &&
    app.includes("previousInitial * 2"),
  splatParameterUi:
    app.includes("const DEFAULT_MAX_SIDE = 1600") &&
    index.includes('data-testid="train-size" type="number" min="32" max="3200" step="64" value="1600"') &&
    index.includes("<strong>Splat parameters</strong>") &&
    index.includes('data-testid="splat-scale-x" type="range" min="0" max="5"') &&
    index.includes('data-testid="splat-scale-y" type="range" min="0" max="5"') &&
    !index.includes('id="splatRotation"') &&
    !app.includes("els.splatRotation") &&
    !app.includes("rotationDegrees"),
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
    app.includes("var layerOrder = clamp(fract(t.w)") &&
    app.includes("fn tile_less(") &&
    app.includes('source: "training-pixel-state-vs-standalone-rgba"') &&
    app.includes("...displayRgbaParity(trainingFrame.rgba, standaloneRgba)") &&
    app.includes("Boolean(params.layerOrderEnabled)"),
  aspectAwareTrainingGrid:
    app.includes("function aspectAwareGridEnabled") &&
    app.includes("const baseScaleX = aspectAwareGridEnabled() ? 1.6 / cols : baseScale") &&
    app.includes("const baseScaleY = aspectAwareGridEnabled() ? 1.6 / rows : baseScale") &&
    app.includes("layout.baseScaleX") &&
    app.includes("layout.baseScaleY") &&
    app.includes("size: 64 * 4"),
  simpleExports:
    !index.includes("Compressed PLY") &&
    !index.includes("PlayCanvas SOG") &&
    !app.includes("compressed-ply") &&
    !app.includes('formatKey !== "ply"') &&
    index.indexOf('id="savePlyButton"') < index.indexOf('id="savePngButton"'),
  tiltViewer:
    index.indexOf('id="eventLogTab"') < index.indexOf('id="tiltTab"') &&
    index.includes('id="tiltCanvas"') &&
    app.includes('await import(TILT_VIEWER_MODULE_URL)') &&
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
    !tiltViewer.includes("app.autoRender = true") &&
    tiltViewer.includes("gammaCorrection: GAMMA_SRGB") &&
    tiltViewer.includes("toneMapping: TONEMAP_NONE") &&
    !tiltViewer.includes("app.scene.gammaCorrection") &&
    !tiltViewer.includes("app.scene.toneMapping"),
  fibonacciHemisphere:
    tiltCamera.includes("TILT_MAX_ANGLE_DEGREES = 75") &&
    tiltCamera.includes("FIBONACCI_HEMISPHERE_POSE_COUNT = 49") &&
    tiltCamera.includes("FIBONACCI_GOLDEN_ANGLE_RADIANS") &&
    index.includes('data-testid="tilt-sweep-button"') &&
    index.includes('data-testid="tilt-contact-sheet"') &&
    app.includes("startTiltHemisphereSweep") &&
    app.includes("captureFrameBlob") &&
    app.includes("clearTiltSweepResults"),
  renamedProduct:
    index.includes("<title>Image2SplatPaint</title>") &&
    app.includes('filename: "image2splatpaint.ply"') &&
    app.includes('filename: "image2splatpaint.png"') &&
    app.includes('window.__image2SplatPaint =') &&
    app.includes("window.__image2GaussianPaint = window.__image2SplatPaint") &&
    app.includes("comment image2gaussianpaint_frame"),
};

assert.ok(Object.values(contracts).every(Boolean), JSON.stringify(contracts));
console.log(JSON.stringify({ ok: true, contracts }, null, 2));
