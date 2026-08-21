import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

async function loadInstalledGlobal(path, name) {
  const context = vm.createContext({
    ArrayBuffer,
    DataView,
    Math,
    Number,
    Object,
    Set,
    TextEncoder,
    Uint8Array,
  });
  context.globalThis = context;
  vm.runInContext(await readFile(new URL(`../${path}`, import.meta.url), "utf8"), context, { filename: path });
  return context[name];
}

test("numeric helpers preserve clamping and step normalization contracts", async () => {
  const numeric = await loadInstalledGlobal("web/core/numeric-utils.js", "Image2SplatPaintNumeric");
  assert.equal(numeric.clampNumber("12", 0, 10, 4), 10);
  assert.equal(numeric.clampNumber("invalid", 0, 10, 4), 4);
  assert.equal(numeric.normalizeStepInteger(18, { min: 1, max: 20, fallback: 5, step: 4 }), 17);
  assert.deepEqual(Array.from(numeric.hexColorToRgb("#ff8000")), [1, 128 / 255, 0]);
});

test("image metadata preserves dimensions and EXIF axis semantics", async () => {
  const metadata = await loadInstalledGlobal("web/input/image-metadata.js", "Image2SplatPaintImageMetadata");
  assert.equal(metadata.orientationSwapsImageAxes(6), true);
  assert.equal(metadata.orientationSwapsImageAxes(3), false);
  assert.deepEqual(Array.from(metadata.displayOrientedImageSize(4032, 3024, 6)), [3024, 4032]);

  const png = new Uint8Array(24);
  png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const view = new DataView(png.buffer);
  view.setUint32(16, 2048);
  view.setUint32(20, 1024);
  assert.deepEqual(
    { ...metadata.parseImageDimensions(png, "image/png") },
    { width: 2048, height: 1024, format: "png" },
  );
});

test("PLY serializer preserves frame, standard-alpha header, and 17-float rows", async () => {
  const serializer = await loadInstalledGlobal("web/export/ply-serializer.js", "Image2SplatPaintPlySerializer");
  assert.deepEqual(
    { ...serializer.plyFrameScale({ width: 800, height: 400 }) },
    { x: 1, y: 0.5, width: 800, height: 400, aspect: 2 },
  );
  const header = serializer.createPlyHeader({
    count: 1,
    image: { width: 800, height: 400 },
    boundarySigma: 3,
    layerOrderEnabled: true,
    layerDepthSpan: 0.01,
  });
  assert.match(header, /comment image2gaussianpaint_blend standard_alpha/);
  assert.equal((header.match(/^property float /gm) || []).length, 17);

  const rowBytes = 17 * 4;
  const result = serializer.serializeBinaryPly({
    header,
    count: 1,
    rowBytes,
    shC0: 0.28209479177387814,
    geometryAt: () => ({ x: 0, y: 0, sx: 0.1, sy: 0.05, theta: 0 }),
    depthAt: () => 0,
    rgbAt: () => [0.5, 0.5, 0.5],
    opacityAt: () => 0.5,
    logit: (value) => Math.log(value / (1 - value)),
  });
  assert.equal(result.byteLength, new TextEncoder().encode(header).byteLength + rowBytes);
});

test("GPU profile reducer preserves histogram percentile bins", async () => {
  const metrics = await loadInstalledGlobal("web/gpu/metrics.js", "Image2SplatPaintGpuMetrics");
  const summary = metrics.profileDistributionSummary([1, 1, 2, 0, 0, 0, 0, 0], 4, 8);
  assert.equal(summary.mean, 2);
  assert.equal(summary.p50_bin, "1");
  assert.equal(summary.p90_bin, "2-3");
  assert.equal(summary.histogram["2-3"], 2);
});

test("WGSL factories preserve byte-stable tile and optimizer reset sources", async () => {
  const context = vm.createContext({
    ILLUSTRATIVE_OIL_WGSL: "IO",
    MIP_PIXEL_SIGMA: 0.5,
    Object,
    RECTANGLE_TRAPEZOID_WGSL: "RT",
    RENDER_SIGMA: 4,
    TILE_SIZE: 16,
    VIRTUAL_TILT_WGSL: "VT",
  });
  context.globalThis = context;
  for (const path of ["web/gpu/shaders/tile-pipeline.js", "web/gpu/shaders/optimizer-reset.js"]) {
    vm.runInContext(await readFile(new URL(`../${path}`, import.meta.url), "utf8"), context, { filename: path });
  }
  const tile = context.Image2SplatPaintTilePipelineShader.create({
    opacitySupportDeclaration: "OD",
    opacitySupportFunction: "OF",
    paintSupportFunctions: "PF",
    supportSigmaExpression: "SS",
    exactTileIntersectionFunction: "EI",
    opacitySupportEarlyExit: "OE",
    exactTileIntersectionGuard: "EG",
    sortTilesFunction: "ST",
  });
  const optimizerReset = context.Image2SplatPaintOptimizerResetShader.create();
  assert.equal(tile.length, 6317);
  assert.equal(createHash("sha256").update(tile).digest("hex"), "d3c6136fb57c6f0f005de5c2e110bdbbcfe8ce5a26f084c878db86d3d92e29ba");
  assert.equal(optimizerReset.length, 1866);
  assert.equal(createHash("sha256").update(optimizerReset).digest("hex"), "946e15a4b94356a399ddae7ea2b48da581472575ec70484f897d31d0672ca2d1");
});

test("preview WGSL factories preserve generated source fingerprints", async () => {
  const context = vm.createContext({
    ILLUSTRATIVE_OIL_WGSL: "IO",
    LAYERED_OPAQUE_BRUSH_EDGE_SOFTNESS: 2,
    LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT: 1,
    LAYER_CODE_RANGE: 6,
    MAX_DISCRETE_LAYER_COUNT: 8,
    MIN_DISCRETE_LAYER_COUNT: 7,
    Object,
    RECTANGLE_EDGE_SOFTNESS: 4,
    RECTANGLE_FLAG_ASYMMETRIC_SOFTNESS: 5,
    RECTANGLE_KERNEL_EXTENT: 3,
    RECTANGLE_TRAPEZOID_WGSL: "RT",
    RENDER_SIGMA: 4,
    TILE_SIZE: 16,
  });
  context.globalThis = context;
  vm.runInContext(
    await readFile(new URL("../web/gpu/shaders/preview-pipelines.js", import.meta.url), "utf8"),
    context,
  );
  const shaders = Object.keys(context.Image2SplatPaintPreviewShaders)
    .map((name) => context.Image2SplatPaintPreviewShaders[name]());
  assert.deepEqual(shaders.map((shader) => shader.length), [9565, 3557, 7465, 375, 809]);
  assert.equal(
    createHash("sha256").update(shaders.join("\u0000")).digest("hex"),
    "cb624d452f9cb10e95b80910f00dc6f22c973a25f20b8c6bd64b363a09cb54e1",
  );
});

test("metric WGSL factories preserve generated source fingerprints", async () => {
  const context = vm.createContext({
    BACKGROUND_EXPOSURE_EPSILON: 0.01,
    DEFAULT_ALPHA_TARGET: 1,
    METRIC_TILE_STRIDE: 2,
    MIP_PIXEL_SIGMA: 0.5,
    Object,
    OVERLAP_METRIC_STRIDE: 4,
    RENDER_SIGMA: 5,
    TILE_SIZE: 16,
    VIRTUAL_CAMERA_METRIC_TILE_STRIDE: 3,
    VIRTUAL_TILT_WGSL: "VT",
  });
  context.globalThis = context;
  vm.runInContext(await readFile(new URL("../web/gpu/shaders/metric-pipelines.js", import.meta.url), "utf8"), context);
  const shaders = Object.keys(context.Image2SplatPaintMetricShaders).map((name) =>
    name === "overlapMetrics"
      ? context.Image2SplatPaintMetricShaders[name]({ hiddenRgbBinding: "HB", hiddenRgbShader: "HS" })
      : context.Image2SplatPaintMetricShaders[name](),
  );
  assert.deepEqual(shaders.map((shader) => shader.length), [1862, 8933, 7701, 10818, 4788]);
  assert.equal(
    createHash("sha256").update(shaders.join("\u0000")).digest("hex"),
    "96f3b98c1226ef0eab07d6ade90cffa57be4d8217f664f81e72b4a81747618ed",
  );
});
