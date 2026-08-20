const {
  clampNumber,
  normalizeStepInteger,
  hexColorToRgb,
  limitNumber,
} = globalThis.Image2SplatPaintNumeric;
const { canvasToBlob } = globalThis.Image2SplatPaintCanvasBlob;
const {
  createPlyHeader,
  plyFrameScale: exportedPlyFrameScale,
  serializeBinaryPly,
  transformPlanarSplat,
} = globalThis.Image2SplatPaintPlySerializer;
const { inspectPlyContract: inspectSerializedPlyContract } = globalThis.Image2SplatPaintPlyInspector;
const {
  displayOrientedImageSize,
  orientationSwapsImageAxes,
  parseImageDimensions,
} = globalThis.Image2SplatPaintImageMetadata;
const { profileDistributionSummary } = globalThis.Image2SplatPaintGpuMetrics;
const { requestWebGpuDevice } = globalThis.Image2SplatPaintGpuDevice;

const APP_SCRIPT_URL = document.currentScript?.src || new URL("./app.js", location.href).href;
const SAMPLE_IMAGE_URL = new URL("../assets/source-images/generated-geometric-sample.jpg", APP_SCRIPT_URL).href;
const EMBEDDED_SAMPLE_IMAGE_URL = globalThis.__IMAGE2SPLATPAINT_SAMPLE_JPEG || "";
const ALGORITHM_REGISTRY = Object.freeze({
  [PLANAR_GAUSSIAN_ALGORITHM_ID]: Object.freeze({
    id: PLANAR_GAUSSIAN_ALGORITHM_ID,
    label: "Planar Gaussian",
    backend: "custom-webgpu",
    initialize: initGaussians,
    train: trainPlanarGaussian,
    exports: Object.freeze(["png", "ply"]),
    capabilities: Object.freeze({ render: true, metrics: true, png: true, ply: true, tilt: false, virtualCameras: false }),
  }),
  [RECTANGLE_SPLATS_ALGORITHM_ID]: Object.freeze({
    id: RECTANGLE_SPLATS_ALGORITHM_ID,
    label: "Rectangle Splats",
    backend: "custom-webgpu",
    initialize: initRectangles,
    train: trainRectangleSplats,
    exports: Object.freeze(["png"]),
    capabilities: Object.freeze({
      render: true,
      metrics: true,
      png: true,
      ply: false,
      tilt: false,
      virtualCameras: false,
      kernelShape: "rectangle",
      opaqueLayeredPaint: true,
      minimumOpacity: true,
      variableTopWidth: true,
      requiresLayerOrder: true,
      configurableLayerCount: true,
      contributionCleanup: true,
    }),
  }),
  [LAYERED_OPAQUE_BRUSH_ALGORITHM_ID]: Object.freeze({
    id: LAYERED_OPAQUE_BRUSH_ALGORITHM_ID,
    label: "Brush Splats",
    backend: "custom-webgpu",
    initialize: initLayeredOpaqueBrush,
    train: trainLayeredOpaqueBrush,
    exports: Object.freeze(["png"]),
    capabilities: Object.freeze({
      render: true,
      metrics: true,
      png: true,
      ply: false,
      tilt: false,
      virtualCameras: false,
      kernelShape: "opaque-brush",
      opaqueLayeredPaint: true,
      minimumOpacity: true,
      requiresLayerOrder: true,
      configurableLayerCount: true,
      contributionCleanup: true,
    }),
  }),
  [GS_VIRTUAL_CAMERA_ALGORITHM_ID]: Object.freeze({
    id: GS_VIRTUAL_CAMERA_ALGORITHM_ID,
    label: "GS Virtual Camera Sampling",
    backend: "custom-webgpu",
    initialize: initGaussians,
    train: trainGsVirtualCameraSampling,
    exports: Object.freeze(["png", "ply"]),
    capabilities: Object.freeze({ render: true, metrics: true, png: true, ply: true, tilt: true, virtualCameras: true }),
  }),
});
const LIMITS = {
  trainSizeMin: 32,
  trainSizeMax: 3200,
  splatsMin: 4,
  splatsMax: MANUAL_SPLATS_MAX,
  stepsMin: 1,
  stepsMax: Number.MAX_SAFE_INTEGER,
  lrMin: 0,
  lrDefaultMax: 1,
  opacityLrMax: 1,
  scaleLrMax: 1,
  thetaAlignLrMax: 1,
  maxAnisotropyMin: 1,
  maxAnisotropyMax: 32,
  boundarySigmaMin: 0,
  boundarySigmaMax: 4,
  maxPlanarScaleMin: 0.02,
  maxPlanarScaleMax: 2,
  detailCoherenceMin: 0,
  detailCoherenceMax: 1,
};

const QA_RUNTIME_ENABLED =
  location.protocol === "file:" ||
  location.hostname === "localhost" ||
  location.hostname === "127.0.0.1" ||
  location.hostname === "[::1]";

function liveQualityMetricsEnabled() {
  return Boolean(document.querySelector("#liveQualityMetrics")?.checked);
}

function qaPeriodicTrainingEvaluationEnabled() {
  if (!QA_RUNTIME_ENABLED) return false;
  const query = new URLSearchParams(globalThis.location?.search || "");
  return query.get("periodic-metrics") === "1" || query.get("cdp-check") === "1";
}

function periodicTrainingEvaluationEnabled() {
  return liveQualityMetricsEnabled() || qaPeriodicTrainingEvaluationEnabled();
}

function finalRenderAuditEnabled() {
  if (!QA_RUNTIME_ENABLED) return false;
  const query = new URLSearchParams(globalThis.location?.search || "");
  return query.get("final-render-audit") === "1" || query.get("cdp-check") === "1";
}

function previewInvariantHashEnabled() {
  if (!QA_RUNTIME_ENABLED) return false;
  const query = new URLSearchParams(globalThis.location?.search || "");
  return query.get("preview-hash") === "1";
}

function selectedAlgorithm() {
  const id = document.querySelector("#algorithmSelect")?.value || PLANAR_GAUSSIAN_ALGORITHM_ID;
  const algorithm = ALGORITHM_REGISTRY[id];
  if (!algorithm) throw new Error(`Algorithm is not available: ${id}`);
  return algorithm;
}

function stageAwareGrowthDefaultForAlgorithm(algorithm = selectedAlgorithm()) {
  void algorithm;
  return true;
}

function trainedResultAlgorithm() {
  if (!state.params || !state.metrics?.algorithm) return null;
  return ALGORITHM_REGISTRY[state.metrics.algorithm] || null;
}

function displayedResultAlgorithm() {
  return trainedResultAlgorithm() || selectedAlgorithm();
}

function algorithmSupportsExport(formatKey, algorithm = displayedResultAlgorithm()) {
  if (!algorithm?.exports.includes(formatKey)) return false;
  // The selector configures the next run. Never let changing it reinterpret an
  // existing analytic paint result as a Gaussian PLY.
  if (formatKey === "ply" && state.params) {
    return normalizedKernelShape(state.params.kernelShape) === "gaussian";
  }
  return true;
}

function algorithmUsesVirtualCameras(algorithm = selectedAlgorithm()) {
  return Boolean(algorithm.capabilities.virtualCameras);
}

function algorithmUsesRectangleKernel(algorithm = selectedAlgorithm()) {
  return algorithm.capabilities.kernelShape === "rectangle";
}

function algorithmUsesLayeredOpaqueBrush(algorithm = selectedAlgorithm()) {
  return algorithm.capabilities.kernelShape === "opaque-brush";
}

function algorithmUsesOpaqueLayeredPaint(algorithm = selectedAlgorithm()) {
  return Boolean(algorithm.capabilities.opaqueLayeredPaint);
}

function algorithmUsesPaintKernel(algorithm = selectedAlgorithm()) {
  return algorithmUsesRectangleKernel(algorithm) || algorithmUsesLayeredOpaqueBrush(algorithm);
}

function normalizedKernelShape(value) {
  return ["rectangle", "opaque-brush"].includes(value) ? value : "gaussian";
}

function configurePaintKernel(config, params = state.params) {
  const shape = normalizedKernelShape(params?.kernelShape);
  config[40] = shape === "opaque-brush"
    ? 4
    : shape === "rectangle" ? 1 : 0;
  config[41] = shape === "opaque-brush"
    ? LAYERED_OPAQUE_BRUSH_EDGE_SOFTNESS
    : RECTANGLE_EDGE_SOFTNESS;
  config[85] = params?.opaqueLayered ? 1 : 0;
  config[88] = clampNumber(
    params?.minimumOpacity,
    MIN_LEARNED_PAINT_OPACITY,
    MAX_LEARNED_PAINT_OPACITY,
    MAX_LEARNED_PAINT_OPACITY,
  );
  // Child visibility/order is independent from the physical v1/v2 compaction
  // choice.  v2 must not change the growth rule it is measuring.
  config[86] = params?.currentVisibilityChildPolicyEnabled === false ? 0 : 1;
  config[87] = params?.brushWidthTaperEnabled ? DEFAULT_LAYERED_BRUSH_TAPER_LR : 0;
  // Opaque Paint algorithms learn opacity from RGB error above this floor.
  config[90] = params?.minimumOpacityEnabled ? 1 : 0;
  config[91] = clampNumber(
    params?.rectangleMaxAspectRatio,
    MIN_RECTANGLE_ASPECT_RATIO,
    MAX_RECTANGLE_ASPECT_RATIO,
    DEFAULT_RECTANGLE_ASPECT_RATIO,
  );
  config[92] = rectangleOrientationCode(params?.rectangleOrientation);
  // CIELAB-L* monochrome underpainting until slot 100, followed by the
  // standard signal-sRGB objective.
  config[93] = params?.monochromeUnderpaintingEnabled ? 1 : 0;
  config[94] = params?.brushOpacityGradientEnabled ? 1 : 0;
  config[95] = params?.brushWidthTaperEnabled ? 1 : 0;
  config[96] = clampNumber(
    params?.rectangleTopRatio,
    MIN_RECTANGLE_TOP_RATIO,
    MAX_RECTANGLE_TOP_RATIO,
    DEFAULT_RECTANGLE_TOP_RATIO,
  );
  config[97] = rectangleShapeFlags(params);
  config[98] = clampNumber(
    params?.rectangleTopRatioMax,
    config[96],
    MAX_RECTANGLE_TOP_RATIO,
    DEFAULT_RECTANGLE_TOP_RATIO_MAX,
  );
  // Density shaders own slot 99 for their paint-repair-only dispatch.
  config[99] = 0;
  config[100] = Math.max(
    0,
    Math.round(Number(params?.colorFinishStartStep) || 0),
  );
  config[101] = clampNumber(params?.brushOpacityGradientStart, 0, 1, 0);
  config[102] = clampNumber(params?.brushOpacityGradientEnd, 0, 1, 1);
  config[103] = clampNumber(params?.brushWidthTaperStart, 0, 1, 1);
  config[104] = clampNumber(params?.brushWidthTaperEnd, 0, 1, 0);
  // Density passes do not mutate the size-ordered layer assignment. The
  // optimizer sets slot 105 only on an explicitly scheduled full-sort step.
  config[105] = 0;
  config[106] = clampNumber(
    params?.surfaceLayerPriorLayers,
    MIN_DISCRETE_LAYER_COUNT,
    MAX_DISCRETE_LAYER_COUNT,
    DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
  );
  // Low values: 1 = QA full-run parent replacement, 2 = monochrome-to-color
  // transition only. Bit 4 is a QA-only front-child probe. Bit 8 folds color
  // repair into the scheduled size-layer promotion. Bit 16 guards ordinary
  // trained forward layer motion until stale Paint RGB has been repaired.
  config[107] = (params?.harmfulRectangleParentSplitEnabled
    ? params?.harmfulRectangleParentSplitTransitionOnly ? 2 : 1
    : 0) +
    (params?.frontSplitChildrenEnabled ? 4 : 0) +
    (params?.surfaceLayerPriorColorAwarePromotion === false ? 0 : 8) +
    (params?.trainLayerColorGuardEnabled ? 16 : 0);
  config[111] = params?.brushLocalColorFlowEnabled
    ? BRUSH_LOCAL_COLOR_FLOW_STRENGTH
    : 0;
  config[112] = params?.brushStrokePersistenceEnabled ? 1 : 0;
  config[113] = clampNumber(params?.brushRibbonAspectFloor, 1, LIMITS.maxAnisotropyMax, BRUSH_STROKE_PERSISTENCE_RIBBON_MIN_RATIO);
  config[114] = clampNumber(params?.brushAccentAspectFloor, 1, LIMITS.maxAnisotropyMax, BRUSH_STROKE_PERSISTENCE_ACCENT_MIN_RATIO);
  config[115] = clampNumber(
    params?.rectangleMinAspectRatio,
    MIN_RECTANGLE_ASPECT_RATIO,
    config[91],
    DEFAULT_RECTANGLE_MIN_ASPECT_RATIO,
  );
  config[116] = clampNumber(
    params?.rectangleOpacityGradientMin,
    0,
    1,
    1,
  );
  config[117] = clampNumber(
    params?.rectangleOpacityGradientMax,
    config[116],
    1,
    1,
  );
  config[118] = clampNumber(
    params?.maximumOpacity,
    config[88],
    MAX_LEARNED_PAINT_OPACITY,
    MAX_LEARNED_PAINT_OPACITY,
  );
  config[119] = clampNumber(
    params?.brushMinAspectRatio,
    LIMITS.maxAnisotropyMin,
    clampNumber(
      params?.brushMaxAspectRatio,
      LIMITS.maxAnisotropyMin,
      LIMITS.maxAnisotropyMax,
      DEFAULT_MAX_ANISOTROPY,
    ),
    LIMITS.maxAnisotropyMin,
  );
  config[123] = clampNumber(params?.rectangleCenterOpacityGradientMin, 0, 1, 1);
  config[124] = clampNumber(
    params?.rectangleCenterOpacityGradientMax,
    config[123],
    1,
    1,
  );
  config[125] = clampNumber(params?.brushCenterOpacityGradientMin, 0, 1, 1);
  config[126] = clampNumber(
    params?.brushCenterOpacityGradientMax,
    config[125],
    1,
    1,
  );
  config[127] = 0;
  return config;
}

function selectedLearnedOpacityRange(minimumSelector, maximumSelector) {
  const min = clampNumber(
    document.querySelector(minimumSelector)?.value,
    MIN_LEARNED_PAINT_OPACITY,
    MAX_LEARNED_PAINT_OPACITY,
    MAX_LEARNED_PAINT_OPACITY,
  );
  const max = clampNumber(
    document.querySelector(maximumSelector)?.value,
    min,
    MAX_LEARNED_PAINT_OPACITY,
    MAX_LEARNED_PAINT_OPACITY,
  );
  return { min, max };
}

function selectedRectangleLearnedOpacity() {
  return selectedLearnedOpacityRange(
    "#rectangleLearnedOpacityMin",
    "#rectangleLearnedOpacityMax",
  );
}

function selectedLayeredBrushLearnedOpacity() {
  return selectedLearnedOpacityRange(
    "#layeredBrushLearnedOpacityMin",
    "#layeredBrushLearnedOpacityMax",
  );
}

function selectedRectangleOpacityGradient() {
  const minimum = clampNumber(
    document.querySelector("#rectangleOpacityGradientMin")?.value,
    0,
    1,
    1,
  );
  const maximum = clampNumber(
    document.querySelector("#rectangleOpacityGradientMax")?.value,
    minimum,
    1,
    1,
  );
  return { min: minimum, max: maximum };
}

function selectedOpacityMultiplierRange(minimumSelector, maximumSelector) {
  const min = clampNumber(document.querySelector(minimumSelector)?.value, 0, 1, 1);
  const max = clampNumber(document.querySelector(maximumSelector)?.value, min, 1, 1);
  return { min, max };
}

function selectedRectangleCenterOpacityGradient() {
  return selectedOpacityMultiplierRange(
    "#rectangleCenterOpacityGradientMin",
    "#rectangleCenterOpacityGradientMax",
  );
}

function selectedLayeredBrushCenterOpacityGradient() {
  return selectedOpacityMultiplierRange(
    "#layeredBrushCenterOpacityGradientMin",
    "#layeredBrushCenterOpacityGradientMax",
  );
}

function selectedLayeredBrushDirectionalEffects() {
  const opacityStart = clampNumber(
    document.querySelector("#layeredBrushOpacityGradientStart")?.value,
    0,
    1,
    1,
  );
  const opacityEnd = clampNumber(
    document.querySelector("#layeredBrushOpacityGradientEnd")?.value,
    opacityStart,
    1,
    1,
  );
  const widthStart = clampNumber(document.querySelector("#layeredBrushWidthTaperStart")?.value, 0, 1, 1);
  const widthEnd = clampNumber(document.querySelector("#layeredBrushWidthTaperEnd")?.value, 0, 1, 1);
  return {
    opacity: Math.abs(1 - opacityStart) > 0.000001 || Math.abs(1 - opacityEnd) > 0.000001,
    opacityStart,
    opacityEnd,
    widthTaper: Math.abs(widthEnd - widthStart) > 0.000001,
    widthStart,
    widthEnd,
  };
}

function selectedSharedColorWorkflow() {
  return {
    monochromeUnderpainting: Boolean(document.querySelector("#monochromeUnderpainting")?.checked),
    colorFinishStartPercent: clampNumber(
      document.querySelector("#colorFinishStart")?.value,
      MIN_COLOR_FINISH_START_PERCENT,
      MAX_COLOR_FINISH_START_PERCENT,
      DEFAULT_COLOR_FINISH_START_PERCENT,
    ),
  };
}

function colorFinishStartStep(totalIterations, percentage) {
  const total = Math.max(1, Math.round(Number(totalIterations) || 1));
  const percent = clampNumber(
    percentage,
    MIN_COLOR_FINISH_START_PERCENT,
    MAX_COLOR_FINISH_START_PERCENT,
    DEFAULT_COLOR_FINISH_START_PERCENT,
  );
  if (percent <= 0) return 1;
  if (percent >= 100) return total + 1;
  return Math.max(1, Math.ceil(total * percent / 100));
}

function selectedRectangleTopRatio() {
  return clampNumber(
    document.querySelector("#rectangleTopRatio")?.value,
    MIN_RECTANGLE_TOP_RATIO,
    MAX_RECTANGLE_TOP_RATIO,
    DEFAULT_RECTANGLE_TOP_RATIO,
  );
}

function selectedRectangleTopRatioMax(minimum = selectedRectangleTopRatio()) {
  return clampNumber(
    document.querySelector("#rectangleTopRatioMax")?.value,
    minimum,
    MAX_RECTANGLE_TOP_RATIO,
    DEFAULT_RECTANGLE_TOP_RATIO_MAX,
  );
}

function selectedRectangleMinAspectRatio() {
  return clampNumber(
    document.querySelector("#rectangleMinAspectRatio")?.value,
    MIN_RECTANGLE_ASPECT_RATIO,
    MAX_RECTANGLE_ASPECT_RATIO,
    DEFAULT_RECTANGLE_MIN_ASPECT_RATIO,
  );
}

function selectedRectangleMaxAspectRatio(minimum = selectedRectangleMinAspectRatio()) {
  return clampNumber(
    document.querySelector("#rectangleMaxAspectRatio")?.value,
    minimum,
    MAX_RECTANGLE_ASPECT_RATIO,
    DEFAULT_RECTANGLE_ASPECT_RATIO,
  );
}

function normalizedRectangleOrientation(value) {
  return ["vertical", "horizontal"].includes(value)
    ? value
    : DEFAULT_RECTANGLE_ORIENTATION;
}

function selectedRectangleOrientation() {
  return normalizedRectangleOrientation(
    document.querySelector("#rectangleOrientation")?.value,
  );
}

function rectangleOrientationCode(value) {
  const orientation = normalizedRectangleOrientation(value);
  return orientation === "vertical" ? 1 : orientation === "horizontal" ? 2 : 0;
}

function constrainedRectangleTheta(theta, sx, sy, orientation = DEFAULT_RECTANGLE_ORIENTATION) {
  const normalized = normalizedRectangleOrientation(orientation);
  if (normalized === "free") return theta;
  const longAxisIsX = sx >= sy;
  if (normalized === "vertical") return longAxisIsX ? Math.PI * 0.5 : 0;
  return longAxisIsX ? 0 : Math.PI * 0.5;
}

function rectangleConstraintProbe(params = state.params) {
  if (!params?.scale || !params?.theta || params.count <= 0) {
    return {
      count: 0,
      max_aspect_ratio: null,
      orientation: normalizedRectangleOrientation(params?.rectangleOrientation),
      max_orientation_error: null,
    };
  }
  const orientation = normalizedRectangleOrientation(params.rectangleOrientation);
  let maxAspectRatio = 1;
  let maxOrientationError = 0;
  for (let i = 0; i < params.count; i += 1) {
    const sx = Math.max(1e-12, Math.abs(params.scale[i * 2]));
    const sy = Math.max(1e-12, Math.abs(params.scale[i * 2 + 1]));
    maxAspectRatio = Math.max(maxAspectRatio, Math.max(sx, sy) / Math.min(sx, sy));
    if (orientation === "free") continue;
    const longAxisAngle = params.theta[i] + (sx >= sy ? 0 : Math.PI * 0.5);
    const orientationError = orientation === "vertical"
      ? Math.abs(Math.cos(longAxisAngle))
      : Math.abs(Math.sin(longAxisAngle));
    maxOrientationError = Math.max(maxOrientationError, orientationError);
  }
  return {
    count: params.count,
    max_aspect_ratio: maxAspectRatio,
    orientation,
    max_orientation_error: orientation === "free" ? null : maxOrientationError,
  };
}

function selectedRectangleShapeSettings(params = null) {
  const fromParams = (key, fallback) =>
    typeof params?.[key] === "boolean"
      ? params[key]
      : Boolean(document.querySelector(`#${key}`)?.checked ?? fallback);
  return {
    preserveArea: fromParams("rectanglePreserveArea", DEFAULT_RECTANGLE_PRESERVE_AREA),
    edgeDirectedTaper: fromParams(
      "rectangleEdgeDirectedTaper",
      DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER,
    ),
    structureAwareRatio: fromParams(
      "rectangleStructureAwareRatio",
      DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO,
    ),
    asymmetricSoftness: fromParams(
      "rectangleAsymmetricSoftness",
      DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS,
    ),
  };
}

function rectangleShapeFlags(params = null) {
  const settings = selectedRectangleShapeSettings(params);
  return (
    (settings.preserveArea ? RECTANGLE_FLAG_PRESERVE_AREA : 0) +
    (settings.structureAwareRatio ? RECTANGLE_FLAG_STRUCTURE_AWARE_RATIO : 0) +
    (settings.asymmetricSoftness ? RECTANGLE_FLAG_ASYMMETRIC_SOFTNESS : 0) +
    (settings.edgeDirectedTaper ? RECTANGLE_FLAG_EDGE_DIRECTED_TAPER : 0)
  );
}

function trainedSplatShape(params = state.params) {
  const shape = normalizedKernelShape(params?.kernelShape || selectedAlgorithm().capabilities.kernelShape);
  if (shape === "rectangle") return "box";
  return shape;
}

function planarTiltRotation(pitchRadians, yawRadians) {
  const cp = Math.cos(pitchRadians);
  const sp = Math.sin(pitchRadians);
  const cy = Math.cos(yawRadians);
  const sy = Math.sin(yawRadians);
  return [
    cy, 0, -sy,
    -sy * sp, cp, -cy * sp,
    -sy * cp, -sp, -cy * cp,
  ];
}

function clampSharedCameraFov(value) {
  return Math.max(
    MIN_SHARED_CAMERA_FOV_DEGREES,
    Math.min(MAX_SHARED_CAMERA_FOV_DEGREES, Number(value) || DEFAULT_SHARED_CAMERA_FOV_DEGREES),
  );
}

function projectPlanarPoint(
  point,
  pitchRadians,
  yawRadians,
  cameraDistance = DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE,
  z = 0,
  frame = { width: 1, height: 1 },
  fovDegrees = DEFAULT_SHARED_CAMERA_FOV_DEGREES,
) {
  const r = planarTiltRotation(pitchRadians, yawRadians);
  const longSide = Math.max(1, Number(frame.width), Number(frame.height));
  const frameX = Math.max(1, Number(frame.width)) / longSide;
  const frameY = Math.max(1, Number(frame.height)) / longSide;
  const x = Number(point[0]);
  const y = Number(point[1]);
  const worldX = frameX * x;
  const worldY = -frameY * y;
  const cameraX = r[0] * worldX + r[1] * worldY + r[2] * z;
  const cameraY = r[3] * worldX + r[4] * worldY + r[5] * z;
  const depth = cameraDistance + r[6] * worldX + r[7] * worldY + r[8] * z;
  const tanHalfFov = Math.tan(clampSharedCameraFov(fovDegrees) * Math.PI / 360);
  const aspect = Math.max(1, Number(frame.width)) / Math.max(1, Number(frame.height));
  return {
    point: [cameraX / (depth * tanHalfFov * aspect), -cameraY / (depth * tanHalfFov)],
    depth,
    valid: Number.isFinite(depth) && depth > 1e-6,
  };
}

function inverseProjectPlanarPoint(
  point,
  pitchRadians,
  yawRadians,
  cameraDistance = DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE,
  frame = { width: 1, height: 1 },
  z = 0,
  fovDegrees = DEFAULT_SHARED_CAMERA_FOV_DEGREES,
  outputAspect = null,
) {
  const r = planarTiltRotation(pitchRadians, yawRadians);
  const longSide = Math.max(1, Number(frame.width), Number(frame.height));
  const frameX = Math.max(1, Number(frame.width)) / longSide;
  const frameY = Math.max(1, Number(frame.height)) / longSide;
  const u = Number(point[0]);
  const v = Number(point[1]);
  const tanHalfFov = Math.tan(clampSharedCameraFov(fovDegrees) * Math.PI / 360);
  const aspect = Number.isFinite(Number(outputAspect)) && Number(outputAspect) > 0
    ? Number(outputAspect)
    : Math.max(1, Number(frame.width)) / Math.max(1, Number(frame.height));
  const focalX = 1 / (tanHalfFov * aspect);
  const focalY = 1 / tanHalfFov;
  const depthBase = cameraDistance + r[8] * z;
  const a00 = u * frameX * r[6] - focalX * frameX * r[0];
  const a01 = -u * frameY * r[7] + focalX * frameY * r[1];
  const a10 = v * frameX * r[6] + focalY * frameX * r[3];
  const a11 = -v * frameY * r[7] - focalY * frameY * r[4];
  const determinant = a00 * a11 - a01 * a10;
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-8) {
    return { point: [0, 0], valid: false };
  }
  const rightX = focalX * r[2] * z - u * depthBase;
  const rightY = -focalY * r[5] * z - v * depthBase;
  const x = (rightX * a11 - a01 * rightY) / determinant;
  const y = (a00 * rightY - rightX * a10) / determinant;
  return { point: [x, y], valid: Number.isFinite(x) && Number.isFinite(y) };
}

const VIRTUAL_TILT_WGSL = `
struct VirtualTiltRotation {
  row0: vec3<f32>,
  row1: vec3<f32>,
  row2: vec3<f32>,
};

fn virtual_tilt_enabled() -> bool { return cfg(56u) > 0.5; }

fn virtual_tilt_rotation() -> VirtualTiltRotation {
  let pitch = cfg(57u);
  let yaw = cfg(58u);
  let cp = cos(pitch);
  let sp = sin(pitch);
  let cy = cos(yaw);
  let sy = sin(yaw);
  return VirtualTiltRotation(
    vec3<f32>(cy, 0.0, -sy),
    vec3<f32>(-sy * sp, cp, -cy * sp),
    vec3<f32>(-sy * cp, -sp, -cy * cp)
  );
}

fn virtual_project_point(point: vec2<f32>, z: f32) -> vec3<f32> {
  if (!virtual_tilt_enabled()) { return vec3<f32>(point, z); }
  let rotation = virtual_tilt_rotation();
  let longSide = max(cfg(0u), cfg(1u));
  let frame = vec2<f32>(cfg(0u), cfg(1u)) / max(longSide, 1.0);
  let source = vec3<f32>(frame.x * point.x, -frame.y * point.y, z);
  let camera = vec3<f32>(dot(rotation.row0, source), dot(rotation.row1, source), dot(rotation.row2, source));
  let distance = max(cfg(59u), 0.01);
  let depth = max(distance + camera.z, 0.000001);
  let fovDegrees = clamp(cfg(64u), ${MIN_SHARED_CAMERA_FOV_DEGREES}.0, ${MAX_SHARED_CAMERA_FOV_DEGREES}.0);
  let tanHalfFov = tan(fovDegrees * 3.141592653589793 / 360.0);
  let aspect = cfg(0u) / max(cfg(1u), 1.0);
  return vec3<f32>(camera.x / (depth * tanHalfFov * aspect), -camera.y / (depth * tanHalfFov), depth);
}

fn virtual_inverse_point_at_z(point: vec2<f32>, z: f32) -> vec3<f32> {
  if (!virtual_tilt_enabled()) { return vec3<f32>(point, 1.0); }
  let rotation = virtual_tilt_rotation();
  let distance = max(cfg(59u), 0.01);
  let longSide = max(cfg(0u), cfg(1u));
  let frame = vec2<f32>(cfg(0u), cfg(1u)) / max(longSide, 1.0);
  let fovDegrees = clamp(cfg(64u), ${MIN_SHARED_CAMERA_FOV_DEGREES}.0, ${MAX_SHARED_CAMERA_FOV_DEGREES}.0);
  let tanHalfFov = tan(fovDegrees * 3.141592653589793 / 360.0);
  let aspect = cfg(0u) / max(cfg(1u), 1.0);
  let focalX = 1.0 / (tanHalfFov * aspect);
  let focalY = 1.0 / tanHalfFov;
  let depthBase = distance + rotation.row2.z * z;
  let a00 = point.x * frame.x * rotation.row2.x - focalX * frame.x * rotation.row0.x;
  let a01 = -point.x * frame.y * rotation.row2.y + focalX * frame.y * rotation.row0.y;
  let a10 = point.y * frame.x * rotation.row2.x + focalY * frame.x * rotation.row1.x;
  let a11 = -point.y * frame.y * rotation.row2.y - focalY * frame.y * rotation.row1.y;
  let determinant = a00 * a11 - a01 * a10;
  if (abs(determinant) < 0.00000001) { return vec3<f32>(0.0, 0.0, 0.0); }
  let rhs = vec2<f32>(
    focalX * rotation.row0.z * z - point.x * depthBase,
    -focalY * rotation.row1.z * z - point.y * depthBase
  );
  let source = vec2<f32>(
    (rhs.x * a11 - a01 * rhs.y) / determinant,
    (a00 * rhs.y - rhs.x * a10) / determinant
  );
  let valid = all(source >= vec2<f32>(-1.0)) && all(source <= vec2<f32>(1.0));
  return vec3<f32>(source, select(0.0, 1.0, valid));
}

fn virtual_inverse_point(point: vec2<f32>) -> vec3<f32> {
  return virtual_inverse_point_at_z(point, 0.0);
}

fn source_domain_reprojection_enabled() -> bool {
  return virtual_tilt_enabled() && cfg(72u) > 0.5;
}

fn training_output_point(gridPoint: vec2<f32>) -> vec3<f32> {
  if (!source_domain_reprojection_enabled()) { return vec3<f32>(gridPoint, 1.0); }
  let projected = virtual_project_point(gridPoint, 0.0);
  let valid = projected.z > 0.000001 && all(projected.xy >= vec2<f32>(-1.0)) && all(projected.xy <= vec2<f32>(1.0));
  return vec3<f32>(projected.xy, select(0.0, 1.0, valid));
}

fn training_source_point(gridPoint: vec2<f32>, projectedPoint: vec3<f32>) -> vec3<f32> {
  if (source_domain_reprojection_enabled()) { return vec3<f32>(gridPoint, projectedPoint.z); }
  return virtual_inverse_point(projectedPoint.xy);
}

fn training_output_pixel(
  sourcePixel: vec2<u32>,
  outputPoint: vec2<f32>,
  width: u32,
  height: u32
) -> vec2<u32> {
  // Front-view pixels already have exact integer coordinates. Converting them
  // to NDC and back can round a tile-boundary pixel (for example x=32) down to
  // the previous tile, while the standalone fragment renderer uses x=32.
  if (!source_domain_reprojection_enabled()) { return sourcePixel; }
  let outputPixel = clamp(
    (outputPoint * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)),
    vec2<f32>(0.0),
    vec2<f32>(f32(width - 1u), f32(height - 1u))
  );
  return vec2<u32>(u32(outputPixel.x), u32(outputPixel.y));
}

fn training_sample_valid(sourcePoint: vec3<f32>) -> bool {
  return sourcePoint.z > 0.5 || (!source_domain_reprojection_enabled() && cfg(66u) > 0.5);
}

fn source_domain_area_weight(gridPoint: vec2<f32>) -> f32 {
  if (!source_domain_reprojection_enabled()) { return 1.0; }
  let stepX = select(1.0, 2.0 / f32(u32(cfg(0u)) - 1u), u32(cfg(0u)) > 1u);
  let stepY = select(1.0, 2.0 / f32(u32(cfg(1u)) - 1u), u32(cfg(1u)) > 1u);
  let directionX = select(stepX, -stepX, gridPoint.x + stepX > 1.0);
  let directionY = select(stepY, -stepY, gridPoint.y + stepY > 1.0);
  let base = virtual_project_point(gridPoint, 0.0).xy;
  let projectedX = virtual_project_point(gridPoint + vec2<f32>(directionX, 0.0), 0.0).xy;
  let projectedY = virtual_project_point(gridPoint + vec2<f32>(0.0, directionY), 0.0).xy;
  let derivativeX = (projectedX - base) / directionX;
  let derivativeY = (projectedY - base) / directionY;
  return clamp(abs(derivativeX.x * derivativeY.y - derivativeX.y * derivativeY.x), 0.05, 4.0);
}

fn packed_layer_order(packedTag: f32) -> f32 {
  return clamp(min(fract(packedTag), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
}

fn virtual_layer_depth(packedTag: f32) -> f32 {
  if (cfg(45u) <= 0.5) { return 0.0; }
  // Keep this shared projection fragment self-contained: several preview and
  // metrics shaders include it without the optimizer's packing helpers.
  let order = clamp(min(fract(packedTag), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  return (order - 0.5) * ${PLY_LAYER_DEPTH_SPAN};
}

fn virtual_bounded_depth(rawDepth: f32) -> f32 {
  return select(0.0, cfg(68u) * tanh(rawDepth), cfg(67u) > 0.5);
}

fn virtual_pass_layer_depth(packedTag: f32, rawDepth: f32) -> f32 {
  if (cfg(67u) > 0.5) { return virtual_layer_depth(packedTag) + virtual_bounded_depth(rawDepth); }
  // Hard-plane baseline: virtual teachers describe one physical board.
  return select(virtual_layer_depth(packedTag), 0.0, virtual_tilt_enabled() && cfg(65u) > 0.5);
}

fn camera_covariance_3d_enabled() -> bool {
  return virtual_tilt_enabled() && cfg(74u) > 0.5;
}

struct ProjectedPlanarGaussian {
  center: vec2<f32>,
  scale: vec2<f32>,
  theta: f32,
  valid: f32,
};

struct ProjectedPointJacobian {
  point: vec3<f32>,
  sourceX: vec2<f32>,
  sourceY: vec2<f32>,
  sourceZ: vec2<f32>,
};

struct CovarianceGradient {
  xx: f32,
  yy: f32,
  xy: f32,
};

struct SourceProjectionGradient {
  center: vec2<f32>,
  logScale: vec2<f32>,
  theta: f32,
  layerZ: f32,
};

fn virtual_project_derivative(
  camera: vec3<f32>,
  depth: f32,
  focalX: f32,
  focalY: f32,
  dCamera: vec3<f32>,
) -> vec2<f32> {
  let invDepthSquared = 1.0 / max(depth * depth, 0.000000000001);
  return vec2<f32>(
    (dCamera.x * depth - camera.x * dCamera.z) * invDepthSquared / max(focalX, 0.000001),
    (-dCamera.y * depth + camera.y * dCamera.z) * invDepthSquared / max(focalY, 0.000001)
  );
}

fn virtual_project_point_with_jacobian(point: vec2<f32>, z: f32) -> ProjectedPointJacobian {
  if (!virtual_tilt_enabled()) {
    return ProjectedPointJacobian(vec3<f32>(point, z), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0));
  }
  let rotation = virtual_tilt_rotation();
  let longSide = max(cfg(0u), cfg(1u));
  let frame = vec2<f32>(cfg(0u), cfg(1u)) / max(longSide, 1.0);
  let source = vec3<f32>(frame.x * point.x, -frame.y * point.y, z);
  let camera = vec3<f32>(dot(rotation.row0, source), dot(rotation.row1, source), dot(rotation.row2, source));
  let rawDepth = max(cfg(59u), 0.01) + camera.z;
  let depth = max(rawDepth, 0.000001);
  let fovDegrees = clamp(cfg(64u), ${MIN_SHARED_CAMERA_FOV_DEGREES}.0, ${MAX_SHARED_CAMERA_FOV_DEGREES}.0);
  let tanHalfFov = tan(fovDegrees * 3.141592653589793 / 360.0);
  let focalX = tanHalfFov * (cfg(0u) / max(cfg(1u), 1.0));
  let focalY = tanHalfFov;
  let sourceCameraX = vec3<f32>(rotation.row0.x * frame.x, rotation.row1.x * frame.x, rotation.row2.x * frame.x);
  let sourceCameraY = vec3<f32>(-rotation.row0.y * frame.y, -rotation.row1.y * frame.y, -rotation.row2.y * frame.y);
  let sourceCameraZ = vec3<f32>(rotation.row0.z, rotation.row1.z, rotation.row2.z);
  return ProjectedPointJacobian(
    vec3<f32>(camera.x / (depth * max(focalX, 0.000001)), -camera.y / (depth * max(focalY, 0.000001)), depth),
    virtual_project_derivative(camera, depth, focalX, focalY, sourceCameraX),
    virtual_project_derivative(camera, depth, focalX, focalY, sourceCameraY),
    virtual_project_derivative(camera, depth, focalX, focalY, sourceCameraZ)
  );
}

fn projected_covariance_gradient(
  scale: vec2<f32>,
  covariance: vec3<f32>,
  logScaleGradient: vec2<f32>,
  thetaGradient: f32,
) -> CovarianceGradient {
  let delta = covariance.x - covariance.y;
  let discriminantSquared = delta * delta + 4.0 * covariance.z * covariance.z;
  let discriminant = sqrt(max(discriminantSquared, 0.000000000001));
  let majorLambdaGradient = logScaleGradient.x / max(2.0 * scale.x * scale.x, 0.000000000001);
  let minorLambdaGradient = logScaleGradient.y / max(2.0 * scale.y * scale.y, 0.000000000001);
  let rootX = delta / discriminant;
  let rootXY = 2.0 * covariance.z / discriminant;
  var xx = 0.5 * (majorLambdaGradient * (1.0 + rootX) + minorLambdaGradient * (1.0 - rootX));
  var yy = 0.5 * (majorLambdaGradient * (1.0 - rootX) + minorLambdaGradient * (1.0 + rootX));
  var xy = (majorLambdaGradient - minorLambdaGradient) * rootXY;
  if (discriminantSquared > 0.0000000001) {
    xx -= thetaGradient * covariance.z / discriminantSquared;
    yy += thetaGradient * covariance.z / discriminantSquared;
    xy += thetaGradient * delta / discriminantSquared;
  }
  return CovarianceGradient(xx, yy, xy);
}

fn covariance_directional_gradient(
  axisX: vec2<f32>,
  axisY: vec2<f32>,
  axisZ: vec2<f32>,
  dAxisX: vec2<f32>,
  dAxisY: vec2<f32>,
  dAxisZ: vec2<f32>,
  gradient: CovarianceGradient,
) -> f32 {
  let dXX = 2.0 * (axisX.x * dAxisX.x + axisY.x * dAxisY.x + axisZ.x * dAxisZ.x);
  let dYY = 2.0 * (axisX.y * dAxisX.y + axisY.y * dAxisY.y + axisZ.y * dAxisZ.y);
  let dXY = axisX.y * dAxisX.x + axisX.x * dAxisX.y + axisY.y * dAxisY.x + axisY.x * dAxisY.y + axisZ.y * dAxisZ.x + axisZ.x * dAxisZ.y;
  return gradient.xx * dXX + gradient.yy * dYY + gradient.xy * dXY;
}

fn source_projection_gradient(
  center: vec2<f32>,
  layerZ: f32,
  t: vec4<f32>,
  screenCenterGradient: vec2<f32>,
  projectedLogScaleGradient: vec2<f32>,
  projectedThetaGradient: f32,
) -> SourceProjectionGradient {
  let scale = max(t.xy, vec2<f32>(0.0001));
  let c = cos(t.z);
  let s = sin(t.z);
  let axisXSource = vec2<f32>(c, s) * scale.x;
  let axisYSource = vec2<f32>(-s, c) * scale.y;
  let axisXTheta = vec2<f32>(-s, c) * scale.x;
  let axisYTheta = vec2<f32>(-c, -s) * scale.y;
  let thickness = max(cfg(75u), 0.000001);
  let base = virtual_project_point_with_jacobian(center, layerZ);
  let xEnd = virtual_project_point_with_jacobian(center + axisXSource, layerZ);
  let yEnd = virtual_project_point_with_jacobian(center + axisYSource, layerZ);
  let zEnd = virtual_project_point_with_jacobian(center, layerZ + thickness);
  let axisX = xEnd.point.xy - base.point.xy;
  let axisY = yEnd.point.xy - base.point.xy;
  let axisZ = zEnd.point.xy - base.point.xy;
  let covariance = vec3<f32>(
    axisX.x * axisX.x + axisY.x * axisY.x + axisZ.x * axisZ.x,
    axisX.y * axisX.y + axisY.y * axisY.y + axisZ.y * axisZ.y,
    axisX.x * axisX.y + axisY.x * axisY.y + axisZ.x * axisZ.y
  );
  let projected = project_planar_gaussian(center, layerZ, t);
  let covarianceGradient = projected_covariance_gradient(max(projected.scale, vec2<f32>(0.0001)), covariance, projectedLogScaleGradient, projectedThetaGradient);
  let centerX = dot(screenCenterGradient, base.sourceX) + covariance_directional_gradient(
    axisX, axisY, axisZ,
    xEnd.sourceX - base.sourceX,
    yEnd.sourceX - base.sourceX,
    zEnd.sourceX - base.sourceX,
    covarianceGradient
  );
  let centerY = dot(screenCenterGradient, base.sourceY) + covariance_directional_gradient(
    axisX, axisY, axisZ,
    xEnd.sourceY - base.sourceY,
    yEnd.sourceY - base.sourceY,
    zEnd.sourceY - base.sourceY,
    covarianceGradient
  );
  let logScaleX = covariance_directional_gradient(
    axisX, axisY, axisZ,
    xEnd.sourceX * axisXSource.x + xEnd.sourceY * axisXSource.y,
    vec2<f32>(0.0),
    vec2<f32>(0.0),
    covarianceGradient
  );
  let logScaleY = covariance_directional_gradient(
    axisX, axisY, axisZ,
    vec2<f32>(0.0),
    yEnd.sourceX * axisYSource.x + yEnd.sourceY * axisYSource.y,
    vec2<f32>(0.0),
    covarianceGradient
  );
  let theta = covariance_directional_gradient(
    axisX, axisY, axisZ,
    xEnd.sourceX * axisXTheta.x + xEnd.sourceY * axisXTheta.y,
    yEnd.sourceX * axisYTheta.x + yEnd.sourceY * axisYTheta.y,
    vec2<f32>(0.0),
    covarianceGradient
  );
  let layerZGradient = dot(screenCenterGradient, base.sourceZ) + covariance_directional_gradient(
    axisX, axisY, axisZ,
    xEnd.sourceZ - base.sourceZ,
    yEnd.sourceZ - base.sourceZ,
    zEnd.sourceZ - base.sourceZ,
    covarianceGradient
  );
  return SourceProjectionGradient(vec2<f32>(centerX, centerY), vec2<f32>(logScaleX, logScaleY), theta, layerZGradient);
}

fn project_planar_gaussian(center: vec2<f32>, layerZ: f32, t: vec4<f32>) -> ProjectedPlanarGaussian {
  if (!camera_covariance_3d_enabled()) {
    return ProjectedPlanarGaussian(center, max(t.xy, vec2<f32>(0.0001)), t.z, 1.0);
  }
  let base = virtual_project_point(center, layerZ);
  let c = cos(t.z);
  let s = sin(t.z);
  let scale = max(t.xy, vec2<f32>(0.0001));
  let axisX = vec2<f32>(c, s) * scale.x;
  let axisY = vec2<f32>(-s, c) * scale.y;
  let projectedX = virtual_project_point(center + axisX, layerZ);
  let projectedY = virtual_project_point(center + axisY, layerZ);
  let projectedNormal = virtual_project_point(center, layerZ + max(cfg(75u), 0.000001));
  let dx = projectedX.xy - base.xy;
  let dy = projectedY.xy - base.xy;
  let dz = projectedNormal.xy - base.xy;
  let covarianceX = dx.x * dx.x + dy.x * dy.x + dz.x * dz.x;
  let covarianceY = dx.y * dx.y + dy.y * dy.y + dz.y * dz.y;
  let covarianceXY = dx.x * dx.y + dy.x * dy.y + dz.x * dz.y;
  let trace = covarianceX + covarianceY;
  let discriminant = sqrt(max(0.0, (covarianceX - covarianceY) * (covarianceX - covarianceY) + 4.0 * covarianceXY * covarianceXY));
  let lambdaMajor = max(0.00000001, 0.5 * (trace + discriminant));
  let lambdaMinor = max(0.00000001, 0.5 * (trace - discriminant));
  let projectedTheta = 0.5 * atan2(2.0 * covarianceXY, covarianceX - covarianceY);
  let valid = base.z > 0.000001 && projectedX.z > 0.000001 && projectedY.z > 0.000001 && projectedNormal.z > 0.000001;
  return ProjectedPlanarGaussian(
    base.xy,
    vec2<f32>(sqrt(lambdaMajor), sqrt(lambdaMinor)),
    projectedTheta,
    select(0.0, 1.0, valid)
  );
}

fn virtual_camera_depth(center: vec2<f32>, packedTag: f32, rawDepth: f32) -> f32 {
  if (!virtual_tilt_enabled()) {
    return select(fract(packedTag), virtual_pass_layer_depth(packedTag, rawDepth), cfg(67u) > 0.5);
  }
  return virtual_project_point(center, virtual_pass_layer_depth(packedTag, rawDepth)).z;
}
`;

const RECTANGLE_TRAPEZOID_WGSL = `
struct RectangleTrapezoidSample {
  kernel: f32,
  gradient: vec2<f32>,
};

fn rectangle_flag_enabled(flags: f32, bit: u32) -> bool {
  return (u32(round(flags)) & bit) != 0u;
}

fn rectangle_effective_width_ratios(
  minimumRatioInput: f32,
  maximumRatioInput: f32,
  packedTag: f32,
  flags: f32
) -> vec2<f32> {
  let minimumRatio = clamp(minimumRatioInput, 0.0, 1.0);
  let maximumRatio = clamp(max(maximumRatioInput, minimumRatio), 0.0, 1.0);
  // Rectangle paint uses eight deterministic depth layers. Permute those
  // layer buckets into a stable 0..1 selector so Min/Max describe a range of
  // short-edge ratios rather than the two opposing edges of every splat.
  let layerOrder = clamp(
    min(fract(packedTag), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE},
    0.0,
    1.0
  );
  let layerBucket = min(7u, u32(round(layerOrder * 7.0)));
  let rangeSelector = f32((layerBucket * 5u + 3u) % 8u) / 7.0;
  let flatStructure = floor(packedTag) < 1.5;
  let shortEdgeRatio = select(
    mix(minimumRatio, maximumRatio, rangeSelector),
    maximumRatio,
    rectangle_flag_enabled(flags, ${RECTANGLE_FLAG_STRUCTURE_AWARE_RATIO}u) &&
      flatStructure
  );
  return vec2<f32>(shortEdgeRatio, 1.0);
}

fn rectangle_area_compensation(widthRatios: vec2<f32>, flags: f32) -> f32 {
  return select(
    1.0,
    sqrt(2.0 / max(widthRatios.x + widthRatios.y, 0.0001)),
    rectangle_flag_enabled(flags, ${RECTANGLE_FLAG_PRESERVE_AREA}u)
  );
}

fn rectangle_trapezoid_kernel_sample(
  normalized: vec2<f32>,
  featherInput: f32,
  minimumRatioInput: f32,
  maximumRatioInput: f32,
  asymmetricSoftness: bool,
  opacityMinimumInput: f32,
  opacityMaximumInput: f32,
  centerOpacityMinimumInput: f32,
  centerOpacityMaximumInput: f32
) -> RectangleTrapezoidSample {
  let feather = clamp(featherInput, 0.01, 0.49);
  let minimumRatio = clamp(minimumRatioInput, 0.0, 1.0);
  let maximumRatio = clamp(max(maximumRatioInput, minimumRatio), 0.0, 1.0);
  // Local -Y is the selected short edge and local +Y is the full-width edge.
  let verticalProgress = clamp(normalized.y * 0.5 + 0.5, 0.0, 1.0);
  let halfWidth = max(0.0001, mix(minimumRatio, maximumRatio, verticalProgress));
  let axisCoordinate = vec2<f32>(
    abs(normalized.x) / halfWidth,
    abs(normalized.y)
  );
  let horizontalRawT =
    (axisCoordinate.x - (1.0 - feather)) / (2.0 * feather);
  let horizontalT = clamp(horizontalRawT, 0.0, 1.0);
  let horizontalAxis =
    1.0 - horizontalT * horizontalT * (3.0 - 2.0 * horizontalT);
  let dHorizontalAxisDu =
    -6.0 * horizontalT * (1.0 - horizontalT) / (2.0 * feather);
  let verticalFeather = select(
    feather,
    mix(feather * 0.70, feather * 1.25, verticalProgress),
    asymmetricSoftness
  );
  let verticalRawT =
    (axisCoordinate.y - (1.0 - verticalFeather)) /
    (2.0 * verticalFeather);
  let verticalT = clamp(verticalRawT, 0.0, 1.0);
  let verticalAxis =
    1.0 - verticalT * verticalT * (3.0 - 2.0 * verticalT);
  let dVerticalFeatherDy = select(
    0.0,
    0.275 * feather,
    asymmetricSoftness && normalized.y > -1.0 && normalized.y < 1.0
  );
  let verticalNumerator = axisCoordinate.y - 1.0 + verticalFeather;
  let verticalDenominator = 2.0 * verticalFeather;
  let dVerticalNumeratorDy = sign(normalized.y) + dVerticalFeatherDy;
  let dVerticalDenominatorDy = 2.0 * dVerticalFeatherDy;
  let dVerticalTDy = select(
    0.0,
    (
      dVerticalNumeratorDy * verticalDenominator -
      verticalNumerator * dVerticalDenominatorDy
    ) / max(verticalDenominator * verticalDenominator, 0.00000001),
    verticalRawT > 0.0 && verticalRawT < 1.0
  );
  let dVerticalAxisDy =
    -6.0 * verticalT * (1.0 - verticalT) * dVerticalTDy;
  let dHalfWidthDy = select(
    0.0,
    0.5 * (maximumRatio - minimumRatio),
    normalized.y > -1.0 && normalized.y < 1.0
  );
  let dHorizontalDy =
    -abs(normalized.x) * dHalfWidthDy / max(halfWidth * halfWidth, 0.00000001);
  let baseGradient = vec2<f32>(
    dHorizontalAxisDu * sign(normalized.x) * verticalAxis / halfWidth,
    horizontalAxis * dVerticalAxisDy +
      dHorizontalAxisDu * verticalAxis * dHorizontalDy
  );
  let baseKernel = horizontalAxis * verticalAxis;
  let opacityMaximum = clamp(opacityMaximumInput, 0.0, 1.0);
  let opacityMinimum = clamp(opacityMinimumInput, 0.0, opacityMaximum);
  let opacityFactor = mix(opacityMinimum, opacityMaximum, verticalProgress);
  let dOpacityFactorDy = select(
    0.0,
    0.5 * (opacityMaximum - opacityMinimum),
    normalized.y > -1.0 && normalized.y < 1.0
  );
  // Squared shape-relative distance is 0 at the center and 1 along every
  // trapezoid edge. It avoids a circular mask cutting across the paint shape.
  let radialCoordinate = max(axisCoordinate.x, axisCoordinate.y);
  let radialProgress = clamp(radialCoordinate * radialCoordinate, 0.0, 1.0);
  let dRadialCoordinate = select(
    vec2<f32>(0.0, sign(normalized.y)),
    vec2<f32>(sign(normalized.x) / halfWidth, dHorizontalDy),
    axisCoordinate.x >= axisCoordinate.y
  );
  let dRadialProgress = select(
    vec2<f32>(0.0),
    2.0 * radialCoordinate * dRadialCoordinate,
    radialCoordinate > 0.0 && radialCoordinate < 1.0
  );
  let centerOpacityMaximum = clamp(centerOpacityMaximumInput, 0.0, 1.0);
  let centerOpacityMinimum = clamp(centerOpacityMinimumInput, 0.0, centerOpacityMaximum);
  let centerOpacityFactor = mix(
    centerOpacityMaximum,
    centerOpacityMinimum,
    radialProgress
  );
  let centerOpacityGradient =
    (centerOpacityMinimum - centerOpacityMaximum) * dRadialProgress;
  let combinedOpacityFactor = opacityFactor * centerOpacityFactor;
  let combinedOpacityGradient =
    vec2<f32>(0.0, dOpacityFactorDy) * centerOpacityFactor +
    centerOpacityGradient * opacityFactor;
  let gradient =
    baseGradient * combinedOpacityFactor + baseKernel * combinedOpacityGradient;
  return RectangleTrapezoidSample(baseKernel * combinedOpacityFactor, gradient);
}
`;

const ILLUSTRATIVE_OIL_WGSL = `
struct OilKernelSample {
  kernel: f32,
  gradient: vec2<f32>,
  taperGradient: f32,
};

fn brush_directional_progress(normalized: vec2<f32>, majorIsX: bool) -> f32 {
  let longitudinal = select(normalized.y, normalized.x, majorIsX);
  return clamp(0.5 + 0.5 * longitudinal, 0.0, 1.0);
}

fn illustrative_oil_family(scale: vec2<f32>, packedTag: f32) -> f32 {
  let minor = max(0.0001, min(scale.x, scale.y));
  let anisotropy = max(scale.x, scale.y) / minor;
  if (floor(packedTag) >= 2.0) { return 2.0; }
  return select(0.0, 1.0, anisotropy >= ${ILLUSTRATIVE_OIL_RIBBON_ANISOTROPY});
}

fn illustrative_oil_kernel_sample(
  normalized: vec2<f32>,
  majorIsX: bool,
  feather: f32,
  family: f32,
  taperAmount: f32,
  opacityGradientEnabled: bool,
  widthTaperEnabled: bool,
  opacityStart: f32,
  opacityEnd: f32,
  centerOpacityMinimumInput: f32,
  centerOpacityMaximumInput: f32,
  widthStart: f32,
  widthEnd: f32
) -> OilKernelSample {
  let longitudinal = select(normalized.y, normalized.x, majorIsX);
  let transverse = select(normalized.x, normalized.y, majorIsX);
  // One connected contour per family. Local structure chooses the family;
  // there are no repeated lobes, grooves, random shape switches, or jaggedness.
  let isRibbon = family >= 0.5 && family < 1.5;
  let isAccent = family >= 1.5;
  let lengthScale = select(select(0.90, 1.00, isRibbon), 0.74, isAccent);
  let widthBase = select(select(0.94, 0.82, isRibbon), 0.68, isAccent);
  let shoulder = select(select(0.10, 0.17, isRibbon), 0.26, isAccent);
  let widthBias = select(select(0.02, -0.025, isRibbon), -0.08, isAccent);
  let bendAmount = select(select(0.025, 0.055, isRibbon), 0.035, isAccent);
  let u = longitudinal / lengthScale;
  let u2 = u * u;
  let directionalProgress = clamp(0.5 + 0.5 * longitudinal, 0.0, 1.0);
  let dProgressDLongitudinal = select(
    0.0,
    0.5,
    longitudinal > -1.0 && longitudinal < 1.0
  );
  let dProgressDU = dProgressDLongitudinal * lengthScale;
  let bend = bendAmount * u * (1.0 - u2);
  let bendGradient = bendAmount * (1.0 - 3.0 * u2);
  let bentTransverse = transverse - bend;
  let baseWidth = widthBase * (1.0 - shoulder * u2 + widthBias * u);
  let baseWidthGradient = widthBase * (-2.0 * shoulder * u + widthBias);
  let taper = select(0.0, clamp(taperAmount, 0.0, 1.0), widthTaperEnabled);
  let configuredWidth = mix(widthStart, widthEnd, directionalProgress);
  let taperFactor = mix(widthStart, configuredWidth, taper);
  let taperFactorGradient = taper * (widthEnd - widthStart) * dProgressDU;
  let rawWidth = baseWidth * taperFactor;
  let rawWidthGradient = baseWidthGradient * taperFactor + baseWidth * taperFactorGradient;
  let rawWidthTaperGradient = select(
    0.0,
    baseWidth * (configuredWidth - widthStart),
    widthTaperEnabled
  );
  // Zero is the public minimum. Keep only an internal epsilon for finite WGSL
  // division at the exact tip where a full width gradient closes to a point.
  let width = max(0.0001, rawWidth);
  let widthGradient = select(0.0, rawWidthGradient, rawWidth > 0.0001);
  let widthTaperGradient = select(0.0, rawWidthTaperGradient, rawWidth > 0.0001);
  let v = bentTransverse / width;
  let v2 = v * v;
  let q = u2 * u2 + v2 * v2;
  let denominator = max(0.0001, 2.0 * feather);
  let t = clamp((q - (1.0 - feather)) / denominator, 0.0, 1.0);
  let baseKernel = 1.0 - t * t * (3.0 - 2.0 * t);
  if (baseKernel <= 0.00000001) {
    return OilKernelSample(0.0, vec2<f32>(0.0), 0.0);
  }
  let dKernelDq = -6.0 * t * (1.0 - t) / denominator;
  let dVdU = -bendGradient / width - v * widthGradient / width;
  let dVdTaper = -v * widthTaperGradient / width;
  let dQdU = 4.0 * u * u2 + 4.0 * v * v2 * dVdU;
  let dQdTaper = 4.0 * v * v2 * dVdTaper;
  let dQdLongitudinal = dQdU / lengthScale;
  let dQdTransverse = 4.0 * v * v2 / width;
  let gradientLongTrans = vec2<f32>(
    dKernelDq * dQdLongitudinal,
    dKernelDq * dQdTransverse
  );
  let baseGradient = select(gradientLongTrans.yx, gradientLongTrans, majorIsX);
  let opacityFactor = select(
    1.0,
    mix(opacityStart, opacityEnd, directionalProgress),
    opacityGradientEnabled
  );
  let dOpacityDLongitudinal = select(
    0.0,
    (opacityEnd - opacityStart) * dProgressDLongitudinal,
    opacityGradientEnabled
  );
  let opacityGradientLongTrans = vec2<f32>(dOpacityDLongitudinal, 0.0);
  let opacityGradient = select(opacityGradientLongTrans.yx, opacityGradientLongTrans, majorIsX);
  // q=1 is the connected Brush contour. sqrt(q) is squared distance in its
  // quartic shape coordinates, giving a finite zero derivative at the center.
  let radialProgress = clamp(sqrt(max(q, 0.0)), 0.0, 1.0);
  let dRadialProgressDq = select(
    0.0,
    0.5 / max(radialProgress, 0.000001),
    q > 0.000000000001 && q < 1.0
  );
  let centerOpacityMaximum = clamp(centerOpacityMaximumInput, 0.0, 1.0);
  let centerOpacityMinimum = clamp(centerOpacityMinimumInput, 0.0, centerOpacityMaximum);
  let centerOpacityFactor = mix(
    centerOpacityMaximum,
    centerOpacityMinimum,
    radialProgress
  );
  let dCenterOpacityDq =
    (centerOpacityMinimum - centerOpacityMaximum) * dRadialProgressDq;
  let centerOpacityGradientLongTrans = dCenterOpacityDq * vec2<f32>(
    dQdLongitudinal,
    dQdTransverse
  );
  let centerOpacityGradient = select(
    centerOpacityGradientLongTrans.yx,
    centerOpacityGradientLongTrans,
    majorIsX
  );
  let combinedOpacityFactor = opacityFactor * centerOpacityFactor;
  let combinedOpacityGradient =
    opacityGradient * centerOpacityFactor + centerOpacityGradient * opacityFactor;
  let kernel = baseKernel * combinedOpacityFactor;
  let gradient =
    baseGradient * combinedOpacityFactor + baseKernel * combinedOpacityGradient;
  let taperGradient =
    dKernelDq * dQdTaper * combinedOpacityFactor +
    baseKernel * opacityFactor * dCenterOpacityDq * dQdTaper;
  return OilKernelSample(kernel, gradient, taperGradient);
}

`;

// Monochrome underpainting preserves perceptual CIELAB L* while splat storage,
// compositing, the color finish, Preview, and final metrics remain signal-sRGB.
const MONOCHROME_LAB_L_WGSL = `
fn srgb_decode_channel(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

fn srgb_decode_derivative(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  let derivative = select(
    (2.4 / 1.055) * pow((c + 0.055) / 1.055, 1.4),
    1.0 / 12.92,
    c <= 0.04045
  );
  // Keep the one-sided derivative at stored sRGB bounds. Color updates are
  // clamped after Adam, so a zero derivative here would trap full-Lab fitting
  // at exactly 0 or 1 even when the target lies inside the valid range.
  return derivative;
}

fn lab_curve(value: f32) -> f32 {
  let delta = 6.0 / 29.0;
  let delta3 = delta * delta * delta;
  return select(value / (3.0 * delta * delta) + 4.0 / 29.0, pow(max(value, 0.0), 1.0 / 3.0), value > delta3);
}

fn lab_curve_derivative(value: f32) -> f32 {
  let delta = 6.0 / 29.0;
  let delta3 = delta * delta * delta;
  return select(1.0 / (3.0 * delta * delta), 1.0 / (3.0 * pow(max(value, delta3), 2.0 / 3.0)), value > delta3);
}

fn srgb_to_normalized_lab(rgb: vec3<f32>) -> vec3<f32> {
  let linear = vec3<f32>(
    srgb_decode_channel(rgb.r),
    srgb_decode_channel(rgb.g),
    srgb_decode_channel(rgb.b)
  );
  let xyz = vec3<f32>(
    dot(linear, vec3<f32>(0.4124564, 0.3575761, 0.1804375)),
    dot(linear, vec3<f32>(0.2126729, 0.7151522, 0.0721750)),
    dot(linear, vec3<f32>(0.0193339, 0.1191920, 0.9503041))
  );
  let f = vec3<f32>(
    lab_curve(xyz.x / 0.95047),
    lab_curve(xyz.y),
    lab_curve(xyz.z / 1.08883)
  );
  return vec3<f32>(
    (116.0 * f.y - 16.0) / 100.0,
    500.0 * (f.x - f.y) / 128.0,
    200.0 * (f.y - f.z) / 128.0
  );
}

fn normalized_lab_gradient_srgb(rgb: vec3<f32>, dLab: vec3<f32>) -> vec3<f32> {
  let linear = vec3<f32>(
    srgb_decode_channel(rgb.r),
    srgb_decode_channel(rgb.g),
    srgb_decode_channel(rgb.b)
  );
  let xyz = vec3<f32>(
    dot(linear, vec3<f32>(0.4124564, 0.3575761, 0.1804375)),
    dot(linear, vec3<f32>(0.2126729, 0.7151522, 0.0721750)),
    dot(linear, vec3<f32>(0.0193339, 0.1191920, 0.9503041))
  );
  let normalizedXyz = vec3<f32>(xyz.x / 0.95047, xyz.y, xyz.z / 1.08883);
  let dF = vec3<f32>(
    dLab.y * (500.0 / 128.0),
    dLab.x * (116.0 / 100.0) - dLab.y * (500.0 / 128.0) + dLab.z * (200.0 / 128.0),
    -dLab.z * (200.0 / 128.0)
  );
  let dXyz = dF * vec3<f32>(
    lab_curve_derivative(normalizedXyz.x) / 0.95047,
    lab_curve_derivative(normalizedXyz.y),
    lab_curve_derivative(normalizedXyz.z) / 1.08883
  );
  let dLinear = vec3<f32>(
    dot(dXyz, vec3<f32>(0.4124564, 0.2126729, 0.0193339)),
    dot(dXyz, vec3<f32>(0.3575761, 0.7151522, 0.1191920)),
    dot(dXyz, vec3<f32>(0.1804375, 0.0721750, 0.9503041))
  );
  return dLinear * vec3<f32>(
    srgb_decode_derivative(rgb.r),
    srgb_decode_derivative(rgb.g),
    srgb_decode_derivative(rgb.b)
  );
}

fn normalized_lab_l_only_gray_gradient_srgb(
  rgb: vec3<f32>,
  targetRgb: vec3<f32>
) -> vec3<f32> {
  let lab = srgb_to_normalized_lab(rgb);
  let targetLab = srgb_to_normalized_lab(targetRgb);
  let unconstrained = normalized_lab_gradient_srgb(
    rgb,
    vec3<f32>(sign(lab.x - targetLab.x), 0.0, 0.0)
  );
  // P1/P2 constrain R=G=B. Share the derivative of that one gray scalar
  // equally across the three stored RGB parameters so Adam preserves neutrality.
  return vec3<f32>((unconstrained.r + unconstrained.g + unconstrained.b) / 3.0);
}
`;

function qaOverrides(name) {
  return QA_RUNTIME_ENABLED && globalThis[name] && typeof globalThis[name] === "object"
    ? globalThis[name]
    : {};
}

function trainLayerColorGuardEnabled() {
  const override = qaOverrides("__image2SplatTrainLayerColorGuard");
  if (typeof override.enabled === "boolean") return override.enabled;
  return true;
}

function opaqueBrushLocalColorFlowEnabled() {
  const override = qaOverrides("__image2SplatBrushLocalColorFlow");
  if (Object.hasOwn(override, "enabled")) return override.enabled !== false;
  return Boolean(document.querySelector("#layeredBrushLocalColorFlow")?.checked);
}

function opaqueBrushStrokePersistenceEnabled() {
  const override = qaOverrides("__image2SplatBrushStrokePersistence");
  if (Object.hasOwn(override, "enabled")) return override.enabled !== false;
  return Boolean(document.querySelector("#layeredBrushStrokePersistence")?.checked);
}

function selectedBrushAspectFloors() {
  const maximum = selectedBrushMaxAspectRatio();
  return {
    ribbon: clampNumber(document.querySelector("#layeredBrushRibbonAspectFloor")?.value, 1, maximum, Math.min(maximum, BRUSH_STROKE_PERSISTENCE_RIBBON_MIN_RATIO)),
    accent: clampNumber(document.querySelector("#layeredBrushAccentAspectFloor")?.value, 1, maximum, Math.min(maximum, BRUSH_STROKE_PERSISTENCE_ACCENT_MIN_RATIO)),
  };
}

function selectedBrushMinAspectRatio() {
  return clampNumber(
    document.querySelector("#layeredBrushMinAspectRatio")?.value,
    LIMITS.maxAnisotropyMin,
    LIMITS.maxAnisotropyMax,
    LIMITS.maxAnisotropyMin,
  );
}

function selectedBrushMaxAspectRatio() {
  const minimum = selectedBrushMinAspectRatio();
  return clampNumber(
    document.querySelector("#layeredBrushMaxAspectRatio")?.value,
    minimum,
    LIMITS.maxAnisotropyMax,
    DEFAULT_MAX_ANISOTROPY,
  );
}

function brushContributionDiagnosticsSettings(algorithm = selectedAlgorithm()) {
  if (!QA_RUNTIME_ENABLED || algorithm?.id !== LAYERED_OPAQUE_BRUSH_ALGORITHM_ID) {
    return {
      enabled: false,
      strength: 0,
      flatLinearGradient: BRUSH_CONTRIBUTION_FLAT_LINEAR_GRADIENT,
    };
  }
  const override = qaOverrides("__image2SplatBrushContributionDiagnostics");
  const query = new URLSearchParams(globalThis.location?.search || "");
  const queryStrength = query.get("brush-neff-strength");
  const queryFlatGradient = query.get("brush-neff-flat-gradient");
  const strength = clampNumber(
    override.strength ?? (queryStrength === null ? 0 : queryStrength),
    0,
    1,
    0,
  );
  return {
    enabled: strength > 0,
    strength,
    flatLinearGradient: clampNumber(
      override.flatLinearGradient ?? (
        queryFlatGradient === null
          ? BRUSH_CONTRIBUTION_FLAT_LINEAR_GRADIENT
          : queryFlatGradient
      ),
      0.0001,
      0.25,
      BRUSH_CONTRIBUTION_FLAT_LINEAR_GRADIENT,
    ),
  };
}

function opaquePaintCurrentVisibilityChildPolicyEnabled() {
  const override = qaOverrides("__image2SplatCurrentVisibilityCompaction");
  if (typeof override.enabled === "boolean") return override.enabled;
  return Boolean(document.querySelector("#currentContributionCompaction")?.checked);
}

function opaquePaintCurrentVisibilityCompactionEnabled() {
  // The retired v1 physical compaction must not silently replace v2 when the
  // public Current-Visibility checkbox is off. It remains QA-addressable only.
  const override = qaOverrides("__image2SplatCurrentVisibilityCompaction");
  return override.physicalCompaction === true &&
    opaquePaintCurrentVisibilityChildPolicyEnabled() &&
    !currentContributionCompactionSettings().enabled;
}

function currentContributionCompactionSettings(algorithm = selectedAlgorithm()) {
  const override = qaOverrides("__image2SplatCurrentContributionCompactionV2");
  const requested = typeof override.enabled === "boolean"
    ? override.enabled
    : Boolean(document.querySelector("#currentContributionCompaction")?.checked);
  const readPercent = (selector, fallback, minimum, maximum) => clampNumber(
    document.querySelector(selector)?.value,
    minimum,
    maximum,
    fallback * 100,
  ) / 100;
  const virtual = algorithmUsesVirtualCameras(algorithm);
  const virtualSampling = virtual ? virtualCameraSamplingVariants(true) : null;
  const requestedWindowSteps = Math.round(clampNumber(
    override.windowSteps,
    1,
    CURRENT_CONTRIBUTION_MAX_WINDOW_STEPS,
    clampNumber(
      document.querySelector("#currentContributionCompactionWindow")?.value,
      1,
      CURRENT_CONTRIBUTION_MAX_WINDOW_STEPS,
      1,
    ),
  ));
  // A virtual run must observe every slot in its deterministic bag before it
  // can call a splat zero-contribution. The UI value may make the window
  // longer, never shorter than one full camera-pool cycle.
  const measurementWindowSteps = virtual
    ? Math.max(requestedWindowSteps, Math.max(1, Math.round(virtualSampling?.slots || 1)))
    : requestedWindowSteps;
  const requestedIntervalSteps = Math.round(clampNumber(
    override.intervalSteps,
    1,
    CURRENT_CONTRIBUTION_MAX_INTERVAL,
    clampNumber(
      document.querySelector("#currentContributionCompactionInterval")?.value,
      1,
      CURRENT_CONTRIBUTION_MAX_INTERVAL,
      CURRENT_CONTRIBUTION_COMPACTION_INTERVAL,
    ),
  ));
  return {
    enabled: requested,
    algorithm: algorithm.id,
    virtual,
    startFraction: clampNumber(
      override.startFraction,
      CURRENT_CONTRIBUTION_MIN_COMPACTION_FRACTION,
      CURRENT_CONTRIBUTION_MAX_COMPACTION_FRACTION,
      readPercent(
        "#currentContributionCompactionStart",
        CURRENT_CONTRIBUTION_COMPACTION_FRACTION,
        CURRENT_CONTRIBUTION_MIN_COMPACTION_FRACTION * 100,
        CURRENT_CONTRIBUTION_MAX_COMPACTION_FRACTION * 100,
      ),
    ),
    maxRemovalFraction: clampNumber(
      override.maxRemovalFraction,
      0,
      CURRENT_CONTRIBUTION_MAX_FRACTION,
      readPercent(
        "#currentContributionCompactionMaxRemoval",
        CURRENT_CONTRIBUTION_MAX_FRACTION,
        0,
        CURRENT_CONTRIBUTION_MAX_FRACTION * 100,
      ),
    ),
    nearZeroMaxFraction: clampNumber(
      override.nearZeroMaxFraction,
      0,
      CURRENT_CONTRIBUTION_MAX_FRACTION,
      readPercent(
        "#currentContributionCompactionNearZero",
        CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_FRACTION,
        0,
        CURRENT_CONTRIBUTION_MAX_FRACTION * 100,
      ),
    ),
    requestedWindowSteps,
    measurementWindowSteps,
    requestedIntervalSteps,
    intervalSteps: Math.max(requestedIntervalSteps, measurementWindowSteps),
    virtualCameraPoolSlots: virtual ? Math.max(1, Math.round(virtualSampling?.slots || 1)) : 0,
  };
}

function scaleBiasedSurfaceLayerPriorSettings(algorithm = selectedAlgorithm()) {
  const override = qaOverrides("__image2SplatSurfaceLayerPrior");
  const requested = typeof override.enabled === "boolean"
    ? override.enabled
    : Boolean(els.scaleBiasedSurfaceLayerPrior?.checked);
  const interval = (name, element) => Math.round(clampNumber(
    override[name],
    0,
    MAX_SCALE_BIASED_SURFACE_LAYER_SORT_INTERVAL,
    clampNumber(
      element?.value,
      0,
      MAX_SCALE_BIASED_SURFACE_LAYER_SORT_INTERVAL,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_INTERVAL,
    ),
  ));
  return {
    enabled: requested,
    algorithm: algorithm.id,
    colorAwarePromotion: override.colorAwarePromotion !== false,
    layers: Math.round(clampNumber(
      override.layers,
      MIN_DISCRETE_LAYER_COUNT,
      MAX_DISCRETE_LAYER_COUNT,
      clampNumber(
        els.scaleBiasedSurfaceLayerPriorLayers?.value,
        MIN_DISCRETE_LAYER_COUNT,
        MAX_DISCRETE_LAYER_COUNT,
        DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
      ),
    )),
    p1Interval: interval("p1Interval", els.scaleBiasedSurfaceLayerPriorP1Interval),
    p2Interval: interval("p2Interval", els.scaleBiasedSurfaceLayerPriorP2Interval),
    p3Interval: interval("p3Interval", els.scaleBiasedSurfaceLayerPriorP3Interval),
    untilFraction: clampNumber(
      override.untilFraction,
      0,
      1,
      clampNumber(
        els.scaleBiasedSurfaceLayerPriorUntil?.value,
        0,
        100,
        DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_UNTIL * 100,
      ) / 100,
    ),
  };
}

function scaleBiasedSurfaceLayerSortSchedule(step, steps, settings) {
  const total = Math.max(1, Math.round(Number(steps) || 1));
  const current = Math.max(1, Math.min(total, Math.round(Number(step) || 1)));
  const p1End = experimentalCoarseSteps(total);
  const p2End = experimentalDensifySteps(total);
  const phase = current <= p1End ? "P1" : current <= p2End ? "P2" : "P3";
  const phaseStart = phase === "P1" ? 1 : phase === "P2" ? p1End + 1 : p2End + 1;
  const interval = Math.max(0, Math.round(Number(
    phase === "P1"
      ? settings?.p1Interval
      : phase === "P2"
        ? settings?.p2Interval
        : settings?.p3Interval,
  ) || 0));
  const untilStep = Math.floor(total * clampNumber(settings?.untilFraction, 0, 1, 0));
  const active = Boolean(settings?.enabled) && interval > 0 && untilStep > 0 && current <= untilStep;
  return {
    enabled: Boolean(settings?.enabled),
    phase,
    phaseStart,
    interval,
    untilStep,
    active,
    due: active && (current - phaseStart + 1) % interval === 0,
  };
}

function harmfulRectangleParentSplitSettings(algorithm = selectedAlgorithm()) {
  const override = qaOverrides("__image2SplatHarmfulRectangleParentSplit");
  const requested = typeof override.enabled === "boolean" ? override.enabled : true;
  return {
    // Every algorithm shares the bounded density-event replacement primitive.
    // Its opacity/layer/depth handling remains algorithm-specific in WGSL.
    enabled: requested,
  };
}

function frontSplitChildrenSettings() {
  const override = qaOverrides("__image2SplatFrontSplitChildren");
  return { enabled: override.enabled === true };
}

function opaquePaintLateSettleFraction() {
  const overrides = qaOverrides("__image2SplatPaintLateSettle");
  const query = new URLSearchParams(globalThis.location?.search || "");
  const queryValue = Number(query.get("paint-settle-fraction"));
  const requested = Number.isFinite(Number(overrides.fraction))
    ? Number(overrides.fraction)
    : QA_RUNTIME_ENABLED && query.has("paint-settle-fraction") && Number.isFinite(queryValue)
      ? queryValue
      : OPAQUE_PAINT_LATE_SETTLE_FRACTION;
  return Math.max(0, Math.min(MAX_OPAQUE_PAINT_LATE_SETTLE_FRACTION, requested));
}

function opaquePaintLateSettleStartStep(steps, fraction = OPAQUE_PAINT_LATE_SETTLE_FRACTION) {
  const total = Math.max(1, Math.round(Number(steps) || 1));
  const boundedFraction = Math.max(0, Math.min(MAX_OPAQUE_PAINT_LATE_SETTLE_FRACTION, Number(fraction) || 0));
  if (boundedFraction <= 0) return total + 1;
  return Math.max(1, Math.floor(total * (1 - boundedFraction)) + 1);
}

function opaquePaintStructuralMutationAllowed(
  step,
  steps,
  opaqueLayered,
  fraction = OPAQUE_PAINT_LATE_SETTLE_FRACTION,
) {
  return !opaqueLayered || Math.round(step) < opaquePaintLateSettleStartStep(steps, fraction);
}

function effectiveGrowthApplyUntilFraction(
  steps,
  requestedFraction,
  opaqueLayered,
  opaquePaintSettleFraction = OPAQUE_PAINT_LATE_SETTLE_FRACTION,
) {
  const total = Math.max(1, Math.round(steps));
  const requested = Math.max(0, Math.min(1, Number(requestedFraction) || 0));
  if (!opaqueLayered || requested <= 0) return requested;
  const lastMutableStep = Math.max(
    0,
    opaquePaintLateSettleStartStep(total, opaquePaintSettleFraction) - 1,
  );
  return Math.min(requested, lastMutableStep / total);
}

const sharedTiltOrbitRadiusCache = new Map();

function sharedTiltOrbitRadius(
  width,
  height,
  maxAngleDegrees = DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES,
  cameraCount = 49,
  fovDegrees = DEFAULT_SHARED_CAMERA_FOV_DEGREES,
) {
  const safeWidth = Math.max(1, Math.round(Number(width) || 1));
  const safeHeight = Math.max(1, Math.round(Number(height) || 1));
  const safeMaxAngle = Math.max(5, Math.min(
    MAX_VIRTUAL_CAMERA_ANGLE_DEGREES,
    Number(maxAngleDegrees) || DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES,
  ));
  const safeCameraCount = Math.max(4, Math.min(MAX_VIRTUAL_CAMERA_COUNT, Math.round(Number(cameraCount) || 49)));
  const safeFov = clampSharedCameraFov(fovDegrees);
  const key = `${safeWidth}x${safeHeight}:${safeMaxAngle}:${safeCameraCount}:${safeFov}`;
  if (!sharedTiltOrbitRadiusCache.has(key)) {
    const longSide = Math.max(safeWidth, safeHeight);
    const frameY = safeHeight / longSide;
    const radius = frameY / Math.tan(safeFov * Math.PI / 360);
    sharedTiltOrbitRadiusCache.set(key, radius);
  }
  return sharedTiltOrbitRadiusCache.get(key);
}

function virtualTeacherCoverage(
  frame,
  pose,
  cameraDistance,
  fovDegrees = DEFAULT_SHARED_CAMERA_FOV_DEGREES,
  sampleSide = 48,
) {
  const side = Math.max(16, Math.min(96, Math.round(Number(sampleSide) || 48)));
  let valid = 0;
  for (let y = 0; y < side; y += 1) {
    const v = y / (side - 1) * 2 - 1;
    for (let x = 0; x < side; x += 1) {
      const u = x / (side - 1) * 2 - 1;
      const source = inverseProjectPlanarPoint(
        [u, v],
        pose.pitchDegrees * Math.PI / 180,
        pose.yawDegrees * Math.PI / 180,
        cameraDistance,
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
  return valid / (side * side);
}

function virtualCameraCoverageStats(width, height, variants, sampleSide = 48) {
  const frame = {
    width: Math.max(1, Math.round(Number(width) || 1)),
    height: Math.max(1, Math.round(Number(height) || 1)),
  };
  const fovDegrees = clampSharedCameraFov(variants?.fovDegrees);
  const cameraCount = Math.max(4, Math.min(MAX_VIRTUAL_CAMERA_COUNT, Math.round(Number(variants?.cameraCount) || DEFAULT_VIRTUAL_CAMERA_COUNT)));
  const maxAngleDegrees = Math.max(5, Math.min(
    MAX_VIRTUAL_CAMERA_ANGLE_DEGREES,
    Number(variants?.maxAngleDegrees) || DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES,
  ));
  const orbitRadius = sharedTiltOrbitRadius(
    frame.width,
    frame.height,
    maxAngleDegrees,
    cameraCount,
    fovDegrees,
  );
  const coverages = virtualCameraFibonacciPoses(cameraCount, maxAngleDegrees)
    .map((pose) => virtualTeacherCoverage(frame, pose, orbitRadius, fovDegrees, sampleSide));
  const total = coverages.reduce((sum, value) => sum + value, 0);
  return {
    sample_side: Math.max(16, Math.min(96, Math.round(Number(sampleSide) || 48))),
    minimum: Math.min(...coverages),
    mean: total / Math.max(1, coverages.length),
    maximum: Math.max(...coverages),
    orbit_radius: orbitRadius,
    fov_degrees: fovDegrees,
  };
}

function qaTileIndexCapacityOverride() {
  if (!QA_RUNTIME_ENABLED) return null;
  const query = new URLSearchParams(globalThis.location?.search || "");
  if (query.get("qa") !== "1" || !query.has("tile-index-capacity")) return null;
  const requested = Number(query.get("tile-index-capacity"));
  if (!Number.isFinite(requested)) return null;
  return Math.max(1, Math.min(TILE_OFFSET_VALUE_MASK, Math.floor(requested)));
}

function qaTileOverflowFixtureEnabled() {
  return qaTileIndexCapacityOverride() !== null;
}

function performanceProfileRequested() {
  const query = new URLSearchParams(globalThis.location?.search || "");
  return QA_RUNTIME_ENABLED && query.get("qa") === "1" && query.get("profile") === "1";
}

function residualDestinationOracleRequested() {
  const query = new URLSearchParams(globalThis.location?.search || "");
  return QA_RUNTIME_ENABLED && query.get("qa") === "1" && query.get("residual-oracle") === "1";
}

function hiddenRgbAttributionRequested() {
  const query = new URLSearchParams(globalThis.location?.search || "");
  return QA_RUNTIME_ENABLED && query.get("qa") === "1" && (
    query.get("hidden-rgb-attribution") === "1" || layerEfficiencyVariants().diagnostics
  );
}

function layerEfficiencyVariants() {
  const overrides = qaOverrides("__image2SplatLayerEfficiency");
  const query = new URLSearchParams(globalThis.location?.search || "");
  const qa = QA_RUNTIME_ENABLED && query.get("qa") === "1";
  const enabled = (name, queryName, fallback = false) => {
    if (typeof overrides[name] === "boolean") return overrides[name];
    if (qa && query.has(queryName)) return query.get(queryName) !== "0";
    return fallback;
  };
  return {
    diagnostics: enabled("diagnostics", "layer-efficiency-diagnostics"),
    deepRelocation: enabled("deepRelocation", "deep-layer-relocation"),
    deepFraction: Math.max(
      0.05,
      Math.min(0.75, Number(overrides.deepFraction) || LAYER_DIAGNOSTIC_DEEP_FRACTION),
    ),
    influenceThreshold: Math.max(0.05, Math.min(1.5, Number(overrides.influenceThreshold) || 0.45)),
  };
}

function layerEfficiencyDiagnosticsRequested() {
  return layerEfficiencyVariants().diagnostics;
}

function overlapDiagnosticsBindingPlan(maxStorageBuffersPerShaderStage, attributionRequested) {
  const availableStorageBuffers = Math.max(0, Math.floor(Number(maxStorageBuffersPerShaderStage) || 0));
  const hiddenRgbEnabled = Boolean(attributionRequested) && availableStorageBuffers >= 9;
  return {
    availableStorageBuffers,
    hiddenRgbEnabled,
    hiddenRgbUnavailable: Boolean(attributionRequested) && !hiddenRgbEnabled,
    storageBufferCount: hiddenRgbEnabled ? 9 : 8,
  };
}

function residualTileCdfEnabled() {
  const query = new URLSearchParams(globalThis.location?.search || "");
  if (QA_RUNTIME_ENABLED && query.get("qa") === "1" && query.has("residual-cdf")) {
    return query.get("residual-cdf") !== "0";
  }
  return true;
}

function residualTileControlWords(image) {
  if (!residualTileCdfEnabled() || !image) return 0;
  const tileCount = Math.ceil(image.width / TILE_SIZE) * Math.ceil(image.height / TILE_SIZE);
  const blocks = Math.ceil(tileCount / 256);
  return tileCount + 1 + blocks * 2;
}

function aspectAwareGridEnabled() {
  const query = new URLSearchParams(globalThis.location?.search || "");
  if (QA_RUNTIME_ENABLED && query.get("qa") === "1" && query.has("aspect-grid")) {
    return query.get("aspect-grid") !== "0";
  }
  return true;
}

function adaptiveGridInitializationVariants(virtualCameraSamplingEnabled = false) {
  const overrides = qaOverrides("__image2SplatAdaptiveInitialization");
  const requested = typeof overrides.enabled === "boolean" ? overrides.enabled : true;
  const rawFraction = Number.isFinite(Number(overrides.fraction))
    ? Number(overrides.fraction)
    : Number(els.adaptiveGridInitializationFraction?.value) / 100;
  const fraction = Math.max(0, Math.min(0.5, Number.isFinite(rawFraction) ? rawFraction : 0.25));
  const candidateCount = Number.isFinite(Number(overrides.candidateCount))
    ? Number(overrides.candidateCount)
    : 12;
  return {
    requested,
    enabled: requested && fraction > 0 && !virtualCameraSamplingEnabled,
    fraction,
    candidateCount: Math.max(4, Math.min(24, Math.round(candidateCount))),
    gridMargin: 0.94,
    reason: !requested || fraction <= 0
      ? "not-requested"
      : virtualCameraSamplingEnabled
        ? "planar-front-only"
        : "enabled",
  };
}

function performanceVariants() {
  const overrides = qaOverrides("__image2GaussianPerformance");
  const query = new URLSearchParams(globalThis.location?.search || "");
  const quadBackwardQuery = query.get("quad-backward");
  const exactTileQuery = query.get("exact-tile-intersection");
  const subgroupQuery = query.get("subgroup-backward");
  const bindGroupCacheQuery = query.get("bind-group-cache");
  const paintShapeCullingQuery = query.get("paint-shape-culling");
  const virtualExactCullingQuery = query.get("virtual-exact-culling");
  const gpuBatchQuery = query.get("gpu-batch");
  const asyncPresentationQuery = query.get("async-presentation");
  const metricReuseQuery = query.get("metric-render-reuse");
  const segmentedBackwardQuery = query.get("segmented-backward");
  const fixedPointGradientQuery = query.get("fixed-point-gradient");
  const inverseScaleQuery = query.get("inverse-scale-optimization");
  const adaptiveGpuBatch = typeof overrides.adaptiveGpuBatch === "boolean"
    ? overrides.adaptiveGpuBatch
    : QA_RUNTIME_ENABLED && query.has("gpu-batch")
      ? gpuBatchQuery !== "0"
      : false;
  const asyncPresentation = typeof overrides.asyncPresentation === "boolean"
    ? overrides.asyncPresentation
    : QA_RUNTIME_ENABLED && query.has("async-presentation")
      ? asyncPresentationQuery !== "0"
      : false;
  const metricTileReuse = typeof overrides.metricTileReuse === "boolean"
    ? overrides.metricTileReuse
    : QA_RUNTIME_ENABLED && query.has("metric-render-reuse")
      ? metricReuseQuery !== "0"
      : false;
  return {
    tileCooperativeRenderer: overrides.tileCooperativeRenderer === true,
    quadExactBackward: typeof overrides.quadExactBackward === "boolean"
      ? overrides.quadExactBackward
      : QA_RUNTIME_ENABLED && query.has("quad-backward")
        ? quadBackwardQuery !== "0"
        : true,
    exactTileIntersection: typeof overrides.exactTileIntersection === "boolean"
      ? overrides.exactTileIntersection
      : QA_RUNTIME_ENABLED && query.has("exact-tile-intersection")
        ? exactTileQuery !== "0"
        : true,
    subgroupExactBackward: typeof overrides.subgroupExactBackward === "boolean"
      ? overrides.subgroupExactBackward
      : QA_RUNTIME_ENABLED && query.has("subgroup-backward")
        ? subgroupQuery !== "0"
        : true,
    bindGroupCache: typeof overrides.bindGroupCache === "boolean"
      ? overrides.bindGroupCache
      : QA_RUNTIME_ENABLED && query.has("bind-group-cache")
        ? bindGroupCacheQuery !== "0"
        : true,
    shapeAwarePaintCulling: typeof overrides.shapeAwarePaintCulling === "boolean"
      ? overrides.shapeAwarePaintCulling
      : QA_RUNTIME_ENABLED && query.has("paint-shape-culling")
        ? paintShapeCullingQuery !== "0"
        : true,
    projectedVirtualExactCulling: typeof overrides.projectedVirtualExactCulling === "boolean"
      ? overrides.projectedVirtualExactCulling
      : QA_RUNTIME_ENABLED && query.has("virtual-exact-culling")
        ? virtualExactCullingQuery !== "0"
        : true,
    segmentedExactBackward: typeof overrides.segmentedExactBackward === "boolean"
      ? overrides.segmentedExactBackward
      : QA_RUNTIME_ENABLED && query.has("segmented-backward")
        ? segmentedBackwardQuery !== "0"
        : false,
    fixedPointExactGradient: typeof overrides.fixedPointExactGradient === "boolean"
      ? overrides.fixedPointExactGradient
      : QA_RUNTIME_ENABLED && query.has("fixed-point-gradient")
        ? fixedPointGradientQuery !== "0"
        : false,
    inverseScaleOptimization: typeof overrides.inverseScaleOptimization === "boolean"
      ? overrides.inverseScaleOptimization
      : QA_RUNTIME_ENABLED && query.has("inverse-scale-optimization")
        ? inverseScaleQuery !== "0"
        : false,
    opacityAwareSupportMode: "off",
    adaptiveGpuThroughput: false,
    adaptiveGpuBatch,
    gpuSchedulingMode: adaptiveGpuBatch ? "adaptive" : "compatible",
    asyncPresentation,
    metricTileReuse,
  };
}

function phase33Variants() {
  const overrides = qaOverrides("__flatPhotoPhase33");
  const enabled = (name, fallback) => (typeof overrides[name] === "boolean" ? overrides[name] : fallback);
  const finite = (name, fallback) => (Number.isFinite(Number(overrides[name])) ? Number(overrides[name]) : fallback);
  const inputNumber = (selector, fallback) => {
    const value = Number(document.querySelector(selector)?.value);
    return Number.isFinite(value) ? value : fallback;
  };
  const coarseStepsOverride = Number(overrides.coarseSteps);
  return {
    importanceRecycle: enabled("importanceRecycle", false),
    adcEligibility: enabled("adcEligibility", false),
    coverageDensity: enabled("coverageDensity", false),
    structureTensor: enabled("structureTensor", true),
    coverageLoss: enabled("coverageLoss", false),
    coarseToFull: enabled("coarseToFull", true),
    threeStageCurriculum: enabled("threeStageCurriculum", true),
    adaptiveCurriculum: enabled("adaptiveCurriculum", true),
    ewa2x2: enabled("ewa2x2", true),
    importanceEma: Math.max(0.001, Math.min(1, finite("importanceEma", PHASE33_IMPORTANCE_EMA))),
    coverageTarget: Math.max(0.001, Math.min(1, finite("coverageTarget", PHASE33_COVERAGE_TARGET))),
    coverageLossWeight: Math.max(0, Math.min(1, finite("coverageLossWeight", PHASE33_COVERAGE_LOSS_WEIGHT))),
    coverageDensityStrength: Math.max(0, Math.min(1, finite("coverageDensityStrength", PHASE33_COVERAGE_DENSITY_STRENGTH))),
    coarseMaxSide: Math.max(64, Math.round(finite("coarseMaxSide", PHASE33_COARSE_MAX_SIDE))),
    coarseSteps: Number.isFinite(coarseStepsOverride) ? Math.max(0, Math.round(coarseStepsOverride)) : null,
    stageMinScaleRatio: Math.max(0, Math.min(0.25, finite("stageMinScaleRatio", DEFAULT_STAGE_MIN_SCALE_RATIO))),
    phaseRelativeScaleGuard: enabled(
      "phaseRelativeScaleGuard",
      Boolean(document.querySelector("#phaseRelativeScaleGuard")?.checked),
    ),
    p1RelativeScaleFloorRatio: Math.max(0, Math.min(1, finite(
      "p1RelativeScaleFloorRatio",
      inputNumber("#p1RelativeScaleFloorRatio", DEFAULT_P1_RELATIVE_SCALE_FLOOR_RATIO),
    ))),
    p2RelativeScaleFloorRatio: Math.max(0, Math.min(1, finite(
      "p2RelativeScaleFloorRatio",
      inputNumber("#p2RelativeScaleFloorRatio", DEFAULT_P2_RELATIVE_SCALE_FLOOR_RATIO),
    ))),
    p3RelativeScaleFloorRatio: Math.max(0, Math.min(1, finite(
      "p3RelativeScaleFloorRatio",
      inputNumber("#p3RelativeScaleFloorRatio", DEFAULT_P3_RELATIVE_SCALE_FLOOR_RATIO),
    ))),
  };
}

function phase37Variants() {
  const overrides = qaOverrides("__flatPhotoPhase37");
  const enabled = (name, fallback) => (typeof overrides[name] === "boolean" ? overrides[name] : fallback);
  return {
    absGradient: enabled("absGradient", false),
    gradientCoherence: enabled("gradientCoherence", true),
    adamRowAge: enabled("adamRowAge", false),
    secondAdcReset: enabled("secondAdcReset", true),
    ewaGaussLegendre: enabled("ewaGaussLegendre", false),
    parallelOptimizer: enabled("parallelOptimizer", true),
    edgeErrorDensity: enabled("edgeErrorDensity", false),
    progressiveGradientLoss: enabled("progressiveGradientLoss", false),
    significanceRecycle: enabled("significanceRecycle", false),
    structureAnisotropy: enabled("structureAnisotropy", false),
    structuralErrorMap: enabled("structuralErrorMap", false),
  };
}

function phase38Variants() {
  const overrides = qaOverrides("__flatPhotoPhase38");
  const finite = (name, fallback) => (Number.isFinite(Number(overrides[name])) ? Number(overrides[name]) : fallback);
  return {
    colorTailSteps: 100,
    adcSplitInterval: Math.max(100, Math.round(finite("adcSplitInterval", Number(document.querySelector("#adcSplitInterval")?.value) || EXPERIMENTAL_ADC_INTERVAL_FOR_7000))),
    adcResetInterval: Math.max(100, Math.round(finite("adcResetInterval", Number(document.querySelector("#adcResetInterval")?.value) || EXPERIMENTAL_ADC_INTERVAL_FOR_7000))),
    adcWindowEvents: Math.max(1, Math.min(32, Math.round(finite("adcWindowEvents", DEFAULT_ADC_WINDOW_EVENTS)))),
  };
}

function phase39Variants() {
  const overrides = qaOverrides("__flatPhotoPhase39");
  const query = new URLSearchParams(location.search);
  const finite = (name, fallback) => (Number.isFinite(Number(overrides[name])) ? Number(overrides[name]) : fallback);
  const queryNumber = (name, fallback) => {
    const value = Number(query.get(name));
    return QA_RUNTIME_ENABLED && query.has(name) && Number.isFinite(value) ? value : fallback;
  };
  const controlNumber = (selector, fallback) => {
    const value = Number(document.querySelector(selector)?.value);
    return Number.isFinite(value) ? value : fallback;
  };
  const choice = (name, allowed, fallback) => (
    allowed.includes(String(overrides[name])) ? String(overrides[name]) : fallback
  );
  const requestedStageShares = {
    p1: Math.max(0, finite("stageGrowthP1", controlNumber("#stageGrowthP1", DEFAULT_STAGE_GROWTH_SHARES.p1))),
    p2: Math.max(0, finite("stageGrowthP2", controlNumber("#stageGrowthP2", DEFAULT_STAGE_GROWTH_SHARES.p2))),
    p3: Math.max(0, finite("stageGrowthP3", controlNumber("#stageGrowthP3", DEFAULT_STAGE_GROWTH_SHARES.p3))),
  };
  const stageShareTotal = requestedStageShares.p1 + requestedStageShares.p2 + requestedStageShares.p3;
  const stageGrowthShares = stageShareTotal > 0
    ? Object.fromEntries(Object.entries(requestedStageShares).map(([key, value]) => [key, value / stageShareTotal]))
    : Object.fromEntries(Object.entries(DEFAULT_STAGE_GROWTH_SHARES).map(([key, value]) => [key, value / 100]));
  return {
    // Density relocation mutates a destination while reading a source. Role
    // reservation is therefore a correctness invariant.
    singleSourceClaim: true,
    densifyInterval: Math.max(1, Math.min(1000, Math.round(finite("densifyInterval", Number(document.querySelector("#densifyInterval")?.value) || 100)))),
    growthFraction: Math.max(0.001, Math.min(1, finite("growthFraction", controlNumber("#growthPercentage", DEFAULT_GROWTH_FRACTION * 100) / 100))),
    growthApplyUntilFraction: Math.max(0, Math.min(1, finite(
      "growthApplyUntilFraction",
      queryNumber(
        "growth-apply-until",
        controlNumber("#growthApplyUntil", DEFAULT_GROWTH_APPLY_UNTIL_FRACTION * 100) / 100,
      ),
    ))),
    growthSignalThreshold: Math.max(0, Math.min(1000, finite("growthSignalThreshold", controlNumber("#growthSignalThreshold", DEFAULT_GROWTH_SIGNAL_THRESHOLD)))),
    stageAwareGrowth: typeof overrides.stageAwareGrowth === "boolean"
      ? overrides.stageAwareGrowth
      : Boolean(document.querySelector("#stageAwareGrowth")?.checked),
    stageGrowthShares,
    requestedStageGrowthShares: requestedStageShares,
    qaGrowthComparisons: QA_RUNTIME_ENABLED && overrides.qaGrowthComparisons === true,
    structureGuidedAllocation: typeof overrides.structureGuidedAllocation === "boolean"
      ? overrides.structureGuidedAllocation
      : Boolean(document.querySelector("#structureGuidedAllocation")?.checked),
    structureLumaSpace: choice(
      "structureLumaSpace",
      ["linear-srgb", "srgb-baseline"],
      "srgb-baseline",
    ),
    structureRegionGrid: Math.max(4, Math.min(MAX_STRUCTURE_REGION_GRID, Math.round(finite(
      "structureRegionGrid",
      controlNumber("#structureRegionGrid", DEFAULT_STRUCTURE_REGION_GRID),
    )))),
    midTrainingOverdensityCorrection: typeof overrides.midTrainingOverdensityCorrection === "boolean"
      ? overrides.midTrainingOverdensityCorrection
      : Boolean(document.querySelector("#midTrainingOverdensityCorrection")?.checked),
    overdensityCorrectionInterval: Math.max(100, Math.min(100000, Math.round(finite(
      "overdensityCorrectionInterval",
      controlNumber("#overdensityCorrectionInterval", DEFAULT_OVERDENSITY_CORRECTION_INTERVAL),
    )))),
    overdensityCorrectionSchedule: choice(
      "overdensityCorrectionSchedule",
      ["interval", "p2-p3-start"],
      ["interval", "p2-p3-start"].includes(
        document.querySelector("#overdensityCorrectionSchedule")?.value,
      ) ? document.querySelector("#overdensityCorrectionSchedule").value : "interval",
    ),
    overdensityDonorFraction: Math.max(0, Math.min(1, finite(
      "overdensityDonorFraction",
      controlNumber("#overdensityDonorPercent", DEFAULT_OVERDENSITY_DONOR_FRACTION * 100) / 100,
    ))),
    // The ADC-specialized split window and recycle reset were retired after
    // paired 7000-step front/virtual tests showed no repeatable net benefit.
    adcSplitEnabled: false,
    adcRecycleEnabled: false,
    mcmcRelocationEnabled: true,
    // QA-only fixed-count diagnostic: skip split, recycle, and relocation.
    densityEventsEnabled: typeof overrides.densityEventsEnabled === "boolean"
      ? overrides.densityEventsEnabled
      : !(QA_RUNTIME_ENABLED && query.get("density-events") === "0"),
    tiltRobustSplit: typeof overrides.tiltRobustSplit === "boolean"
      ? overrides.tiltRobustSplit
      : QA_RUNTIME_ENABLED && query.has("tilt-robust-split")
        ? query.get("tilt-robust-split") !== "0"
        : false,
    tiltSplitAngleDegrees: Math.max(0, Math.min(30, finite(
      "tiltSplitAngleDegrees",
      queryNumber("tilt-split-angle", DEFAULT_TILT_SPLIT_ANGLE_DEGREES),
    ))),
  };
}

function virtualTiltVariants() {
  const overrides = qaOverrides("__image2GaussianTiltTraining");
  const query = new URLSearchParams(location.search);
  const finite = (name, fallback) => (Number.isFinite(Number(overrides[name])) ? Number(overrides[name]) : fallback);
  const queryNumber = (name, fallback) => {
    const value = Number(query.get(name));
    return QA_RUNTIME_ENABLED && query.has(name) && Number.isFinite(value) ? value : fallback;
  };
  const explicitDistance = Number.isFinite(Number(overrides.cameraDistance)) ||
    (QA_RUNTIME_ENABLED && query.has("virtual-tilt-distance") && Number.isFinite(Number(query.get("virtual-tilt-distance"))));
  return {
    enabled: typeof overrides.enabled === "boolean"
      ? overrides.enabled
      : QA_RUNTIME_ENABLED && query.has("virtual-tilt")
        ? query.get("virtual-tilt") !== "0"
        : false,
    interval: Math.max(1, Math.min(1000, Math.round(finite("interval", queryNumber("virtual-tilt-interval", DEFAULT_VIRTUAL_TILT_INTERVAL))))),
    weight: Math.max(0, Math.min(1, finite("weight", queryNumber("virtual-tilt-weight", DEFAULT_VIRTUAL_TILT_WEIGHT)))),
    cameraDistance: Math.max(2, Math.min(32, finite("cameraDistance", queryNumber("virtual-tilt-distance", DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE)))),
    autoCameraDistance: !explicitDistance,
    orderPenaltyWeight: Math.max(0, Math.min(0.1, finite(
      "orderPenaltyWeight",
      queryNumber("virtual-order-weight", DEFAULT_VIRTUAL_ORDER_PENALTY_WEIGHT),
    ))),
  };
}

function virtualTiltStepSpec(step, stage, steps = 7000, variants = null) {
  if (steps && typeof steps === "object") {
    variants = steps;
    steps = 7000;
  }
  variants ||= virtualTiltVariants();
  const total = Math.max(1, Math.round(steps));
  const progress = Math.max(0, Math.min(1, step / total));
  const due = variants.enabled && stage !== "coarse" && step > 0 && step % variants.interval === 0;
  if (!due) {
    return {
      enabled: false,
      pitchRadians: 0,
      yawRadians: 0,
      pitchDegrees: 0,
      yawDegrees: 0,
      cameraDistance: variants.cameraDistance,
      autoCameraDistance: variants.autoCameraDistance,
      weight: 1,
      orderPenaltyWeight: variants.orderPenaltyWeight,
      progress,
      angleDegrees: 0,
    };
  }
  const angleDegrees = stage === "mid" ? 5 : progress < CURRICULUM_TILT_LATE_FRACTION ? 15 : 30;
  const event = Math.max(0, Math.floor(step / variants.interval) - 1);
  const [pitchDirection, yawDirection] = VIRTUAL_TILT_DIRECTIONS[event % VIRTUAL_TILT_DIRECTIONS.length];
  const pitchDegrees = pitchDirection * angleDegrees;
  const yawDegrees = yawDirection * angleDegrees;
  const angleWeight = angleDegrees >= 30 ? 0.5 : angleDegrees >= 15 ? 0.75 : 1;
  return {
    enabled: true,
    pitchRadians: pitchDegrees * Math.PI / 180,
    yawRadians: yawDegrees * Math.PI / 180,
    pitchDegrees,
    yawDegrees,
    cameraDistance: variants.cameraDistance,
    autoCameraDistance: variants.autoCameraDistance,
    weight: variants.weight * angleWeight,
    orderPenaltyWeight: variants.orderPenaltyWeight,
    progress,
    angleDegrees,
  };
}

function virtualCameraSamplingVariants(enabledOverride = null) {
  const overrides = qaOverrides("__image2SplatVirtualCameraSampling");
  const query = new URLSearchParams(location.search);
  const queryNumber = (name, fallback) => {
    const value = Number(query.get(name));
    return QA_RUNTIME_ENABLED && query.has(name) && Number.isFinite(value) ? value : fallback;
  };
  const shareInput = document.querySelector("#virtualCameraShare");
  const cameraCountInput = document.querySelector("#virtualCameraCount");
  const maxAngleInput = document.querySelector("#virtualCameraMaxAngle");
  const fovInput = document.querySelector("#virtualCameraFov");
  const gofDensityInput = document.querySelector("#virtualGofDensity");
  const requestedUiSharePercent = Math.max(1, Math.min(100,
    Number.isFinite(Number(overrides.sharePercent))
      ? Number(overrides.sharePercent)
      : queryNumber("virtual-camera-share", Number(shareInput?.value) || DEFAULT_VIRTUAL_CAMERA_SHARE_PERCENT)));
  const cameraCount = Math.max(4, Math.min(MAX_VIRTUAL_CAMERA_COUNT, Math.round(Number.isFinite(Number(overrides.cameraCount))
    ? Number(overrides.cameraCount)
    : queryNumber("virtual-camera-count", Number(cameraCountInput?.value) || DEFAULT_VIRTUAL_CAMERA_COUNT))));
  const maxAngleDegrees = Math.max(5, Math.min(MAX_VIRTUAL_CAMERA_ANGLE_DEGREES, Number.isFinite(Number(overrides.maxAngleDegrees))
    ? Number(overrides.maxAngleDegrees)
    : queryNumber("virtual-camera-max-angle", Number(maxAngleInput?.value) || DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES)));
  const fovDegrees = clampSharedCameraFov(Number.isFinite(Number(overrides.fovDegrees))
    ? Number(overrides.fovDegrees)
    : queryNumber("virtual-camera-fov", Number(fovInput?.value) || DEFAULT_SHARED_CAMERA_FOV_DEGREES));
  const hasSlotOverride = Number.isFinite(Number(overrides.virtualSlots)) ||
    (QA_RUNTIME_ENABLED && query.has("virtual-camera-slots"));
  const uniformCameras = !hasSlotOverride && requestedUiSharePercent >= 100;
  const slots = uniformCameras ? cameraCount + 1 : DEFAULT_VIRTUAL_CAMERA_POOL_SLOTS;
  const requestedVirtualSlots = Number.isFinite(Number(overrides.virtualSlots))
    ? Number(overrides.virtualSlots)
    : QA_RUNTIME_ENABLED && query.has("virtual-camera-slots")
      ? queryNumber("virtual-camera-slots", DEFAULT_VIRTUAL_CAMERA_SLOTS)
      : requestedUiSharePercent / 100 * slots;
  const requestedSharePercent = hasSlotOverride
    ? requestedVirtualSlots / slots * 100
    : requestedUiSharePercent;
  const virtualSlots = uniformCameras
    ? cameraCount
    : Math.max(1, Math.min(slots - 1, Math.round(requestedVirtualSlots)));
  const gradientBalance = virtualCameraGradientBalance(virtualSlots, slots);
  const tilt = virtualTiltVariants();
  const planeConstrained = typeof overrides.planeConstrained === "boolean"
    ? overrides.planeConstrained
    : !(QA_RUNTIME_ENABLED && query.get("virtual-camera-plane") === "0");
  const requestedInvalidRegionMode = String(
    overrides.invalidRegionMode ?? (QA_RUNTIME_ENABLED ? query.get("virtual-camera-invalid-region") : "") ?? "mask",
  ).toLowerCase();
  const invalidRegionMode = requestedInvalidRegionMode === "black-loss" ? "black-loss" : "mask";
  return {
    enabled: typeof enabledOverride === "boolean"
      ? enabledOverride
      : typeof overrides.enabled === "boolean"
        ? overrides.enabled
        : algorithmUsesVirtualCameras(),
    boundedDepth: Boolean(document.querySelector("#virtualBoundedDepth")?.checked),
    threeDgsMultiview: true,
    depthThickness: Math.max(0.0001, Math.min(0.05, Number.isFinite(Number(overrides.depthThickness))
      ? Number(overrides.depthThickness)
      : DEFAULT_VIRTUAL_DEPTH_THICKNESS)),
    depthCenterWeight: Math.max(0, Math.min(10, Number.isFinite(Number(overrides.depthCenterWeight))
      ? Number(overrides.depthCenterWeight)
      : DEFAULT_VIRTUAL_DEPTH_CENTER_WEIGHT)),
    depthSmoothnessWeight: Math.max(0, Math.min(10, Number.isFinite(Number(overrides.depthSmoothnessWeight))
      ? Number(overrides.depthSmoothnessWeight)
      : DEFAULT_VIRTUAL_DEPTH_SMOOTHNESS_WEIGHT)),
    depthLearningRate: Math.max(0, Math.min(1, Number.isFinite(Number(overrides.depthLearningRate))
      ? Number(overrides.depthLearningRate)
      : DEFAULT_VIRTUAL_DEPTH_LEARNING_RATE)),
    softDepthConstraint: typeof overrides.softDepthConstraint === "boolean"
      ? overrides.softDepthConstraint
      : DEFAULT_VIRTUAL_DEPTH_SOFT_CONSTRAINT,
    depthPriorDelta: Math.max(0.00001, Math.min(0.05, Number.isFinite(Number(overrides.depthPriorDelta))
      ? Number(overrides.depthPriorDelta)
      : DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA)),
    depthUpdateInterval: Math.max(1, Math.min(128, Math.round(Number.isFinite(Number(overrides.depthUpdateInterval))
      ? Number(overrides.depthUpdateInterval)
      : queryNumber("virtual-camera-depth-update-interval", DEFAULT_VIRTUAL_DEPTH_UPDATE_INTERVAL)))),
    // GOF affects density ranking only; optimizer gradients remain signed.
    gofDensity: typeof overrides.gofDensity === "boolean"
      ? overrides.gofDensity
      : QA_RUNTIME_ENABLED && query.has("virtual-camera-gof-density")
        ? query.get("virtual-camera-gof-density") === "1"
        : Boolean(gofDensityInput?.checked),
    slots,
    virtualSlots,
    uniformCameras,
    mode: uniformCameras ? "uniform-all-cameras" : "weighted-virtual-share",
    cameraCount,
    maxAngleDegrees,
    fovDegrees,
    requestedSharePercent,
    effectiveSharePercent: virtualSlots / slots * 100,
    frontGradientAnchorWeight: gradientBalance.frontGradientAnchorWeight,
    effectiveGradientSharePercent: gradientBalance.effectiveVirtualShare * 100,
    seed: Math.max(0, Math.floor(Number.isFinite(Number(overrides.seed))
      ? Number(overrides.seed)
      : queryNumber("virtual-camera-seed", DEFAULT_VIRTUAL_CAMERA_SEED))) >>> 0,
    cameraDistance: tilt.cameraDistance,
    autoCameraDistance: tilt.autoCameraDistance,
    orderPenaltyWeight: tilt.orderPenaltyWeight,
    objectiveMode: "replacement",
    planeConstrained,
    normalizeAdditive: false,
    invalidRegionMode,
    regularizationWeight: Math.max(0, Math.min(1, Number.isFinite(Number(overrides.regularizationWeight))
      ? Number(overrides.regularizationWeight)
      : queryNumber("virtual-camera-regularization-weight", DEFAULT_VIRTUAL_CAMERA_REGULARIZATION_WEIGHT))),
    regularizationRampSteps: Math.max(1, Math.min(2000, Math.round(Number.isFinite(Number(overrides.regularizationRampSteps))
      ? Number(overrides.regularizationRampSteps)
      : queryNumber("virtual-camera-regularization-ramp", DEFAULT_VIRTUAL_CAMERA_REGULARIZATION_RAMP_STEPS)))),
    midAngleDegrees: Math.max(0.5, Math.min(15, Number.isFinite(Number(overrides.midAngleDegrees))
      ? Number(overrides.midAngleDegrees)
      : queryNumber("virtual-camera-mid-angle", DEFAULT_VIRTUAL_CAMERA_MID_ANGLE_DEGREES))),
    fullAngleDegrees: Math.max(0.5, Math.min(30, Number.isFinite(Number(overrides.fullAngleDegrees))
      ? Number(overrides.fullAngleDegrees)
      : queryNumber("virtual-camera-full-angle", DEFAULT_VIRTUAL_CAMERA_FULL_ANGLE_DEGREES))),
  };
}

function virtualCameraGradientBalance(virtualSlots, slots) {
  const safeSlots = Math.max(1, Math.round(Number(slots) || 1));
  const sampledVirtualShare = Math.max(
    0,
    Math.min(1, Number(virtualSlots) / safeSlots || 0),
  );
  // A single source image does not provide independent virtual observations.
  // Above a 50% sampled-view share, pair each virtual gradient with enough
  // canonical-front gradient to keep the aggregate objective at most 50%
  // virtual. The virtual camera still runs at the requested frequency.
  const frontGradientAnchorWeight = Math.max(0, Math.min(1, sampledVirtualShare * 2 - 1));
  return {
    sampledVirtualShare,
    frontGradientAnchorWeight,
    effectiveVirtualShare: sampledVirtualShare / (1 + frontGradientAnchorWeight),
  };
}

function virtualCameraShuffle(values, seed) {
  const shuffled = [...values];
  let stateValue = seed >>> 0;
  const random = () => {
    stateValue = (stateValue + 0x6d2b79f5) >>> 0;
    let value = stateValue;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const selected = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[selected]] = [shuffled[selected], shuffled[index]];
  }
  return shuffled;
}

function virtualCameraBag(cycle, variants) {
  const entries = new Array(variants.slots - variants.virtualSlots).fill(-1);
  const cameraCount = Math.max(1, Math.round(Number(variants.cameraCount) || DEFAULT_VIRTUAL_CAMERA_COUNT));
  const cameraOffset = Math.max(0, Math.round(cycle * variants.virtualSlots)) % cameraCount;
  for (let virtualSlot = 0; virtualSlot < variants.virtualSlots; virtualSlot += 1) {
    entries.push((cameraOffset + virtualSlot) % cameraCount);
  }
  const cycleSeed = (variants.seed ^ Math.imul((cycle + 1) >>> 0, 0x9e3779b9)) >>> 0;
  return virtualCameraShuffle(entries, cycleSeed);
}

function virtualCameraFibonacciPose(index, count = DEFAULT_VIRTUAL_CAMERA_COUNT, maxAngleDegrees = DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES) {
  const safeCount = Math.max(4, Math.min(MAX_VIRTUAL_CAMERA_COUNT, Math.round(Number(count) || DEFAULT_VIRTUAL_CAMERA_COUNT)));
  const safeMaxAngle = Math.max(5, Math.min(
    MAX_VIRTUAL_CAMERA_ANGLE_DEGREES,
    Number(maxAngleDegrees) || DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES,
  ));
  const minCosine = Math.cos(safeMaxAngle * Math.PI / 180);
  const safeIndex = Math.max(0, Math.min(safeCount - 1, Math.round(Number(index) || 0)));
  const sourceIndex = safeIndex + 1;
  const fraction = sourceIndex / safeCount;
  const cosine = 1 - fraction * (1 - minCosine);
  const polar = Math.acos(Math.max(-1, Math.min(1, cosine)));
  const azimuth = sourceIndex * VIRTUAL_CAMERA_GOLDEN_ANGLE_RADIANS % (Math.PI * 2);
  const sine = Math.sin(polar);
  const direction = [sine * Math.cos(azimuth), sine * Math.sin(azimuth), cosine];
  const pitchRadians = Math.asin(Math.max(-1, Math.min(1, direction[1])));
  const yawRadians = Math.atan2(direction[0], direction[2]);
  return {
    index: safeIndex,
    id: `virtual-fib-${String(sourceIndex).padStart(2, "0")}`,
    fraction,
    polarDegrees: polar * 180 / Math.PI,
    azimuthDegrees: azimuth * 180 / Math.PI,
    pitchDegrees: pitchRadians * 180 / Math.PI,
    yawDegrees: yawRadians * 180 / Math.PI,
    direction,
  };
}

function virtualCameraFibonacciPoses(count = DEFAULT_VIRTUAL_CAMERA_COUNT, maxAngleDegrees = DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES) {
  const safeCount = Math.max(4, Math.min(MAX_VIRTUAL_CAMERA_COUNT, Math.round(Number(count) || DEFAULT_VIRTUAL_CAMERA_COUNT)));
  return Array.from({ length: safeCount }, (_, index) => virtualCameraFibonacciPose(index, safeCount, maxAngleDegrees));
}

function virtualCameraIntrinsics(width = 1, height = 1, fovDegrees = DEFAULT_SHARED_CAMERA_FOV_DEGREES) {
  const imageWidth = Math.max(1, Number(width) || 1);
  const imageHeight = Math.max(1, Number(height) || 1);
  const safeFov = clampSharedCameraFov(fovDegrees);
  const focalPixels = imageHeight * 0.5 / Math.tan(safeFov * Math.PI / 360);
  return {
    model: "pinhole",
    width: imageWidth,
    height: imageHeight,
    fov_degrees: safeFov,
    fx: focalPixels,
    fy: focalPixels,
    cx: (imageWidth - 1) * 0.5,
    cy: (imageHeight - 1) * 0.5,
  };
}

function virtualFrontIntrinsics(width = 1, height = 1, fovDegrees = DEFAULT_SHARED_CAMERA_FOV_DEGREES) {
  return {
    ...virtualCameraIntrinsics(width, height, fovDegrees),
    projection: "identity-reference",
  };
}

function virtualCameraSamplingStepSpec(step, stage, steps, coarseStepLimit, variants) {
  const total = Math.max(1, Math.round(steps));
  const front = {
    samplingEnabled: Boolean(variants.enabled),
    kind: "front",
    cameraId: "front",
    enabled: false,
    pitchRadians: 0,
    yawRadians: 0,
    pitchDegrees: 0,
    yawDegrees: 0,
    angleDegrees: 0,
    fovDegrees: variants.fovDegrees,
    cameraDistance: variants.cameraDistance,
    autoCameraDistance: variants.autoCameraDistance,
    orderPenaltyWeight: variants.orderPenaltyWeight,
    cameraCount: variants.cameraCount,
    maxAngleDegrees: variants.maxAngleDegrees,
    planeConstrained: variants.planeConstrained !== false,
    invalidRegionMode: variants.invalidRegionMode,
    weight: 1,
    poolCycle: null,
    poolSlot: null,
  };
  if (!variants.enabled) return front;
  // Camera sampling is independent of the resolution curriculum. Phase 1 uses
  // the coarse target, but follows the configured front/virtual share from step 1.
  const eligibleIndex = Math.max(0, step - 1);
  const poolCycle = Math.floor(eligibleIndex / variants.slots);
  const poolSlot = eligibleIndex % variants.slots;
  const cameraIndex = virtualCameraBag(poolCycle, variants)[poolSlot];
  if (cameraIndex < 0) return { ...front, poolCycle, poolSlot };
  const camera = virtualCameraFibonacciPose(cameraIndex, variants.cameraCount, variants.maxAngleDegrees);
  const pitchDegrees = camera.pitchDegrees;
  const yawDegrees = camera.yawDegrees;
  return {
    ...front,
    kind: "virtual",
    cameraId: camera.id,
    enabled: true,
    pitchRadians: pitchDegrees * Math.PI / 180,
    yawRadians: yawDegrees * Math.PI / 180,
    pitchDegrees,
    yawDegrees,
    angleDegrees: camera.polarDegrees,
    polarDegrees: camera.polarDegrees,
    azimuthDegrees: camera.azimuthDegrees,
    poolCycle,
    poolSlot,
  };
}

function virtualCameraTrainingStepSpec(sample, step, stage, coarseStepLimit, variants) {
  if (sample.kind !== "virtual" || variants.objectiveMode !== "additive") return sample;
  const angleLimit = stage === "mid" ? variants.midAngleDegrees : variants.fullAngleDegrees;
  const angleScale = sample.angleDegrees > angleLimit ? angleLimit / sample.angleDegrees : 1;
  const rampProgress = Math.max(0, Math.min(1, step / variants.regularizationRampSteps));
  const pitchDegrees = sample.pitchDegrees * angleScale;
  const yawDegrees = sample.yawDegrees * angleScale;
  return {
    ...sample,
    pitchDegrees,
    yawDegrees,
    pitchRadians: pitchDegrees * Math.PI / 180,
    yawRadians: yawDegrees * Math.PI / 180,
    angleDegrees: Math.min(sample.angleDegrees, angleLimit),
    regularizationRampProgress: rampProgress,
    weight: variants.regularizationWeight * rampProgress,
  };
}

function virtualCameraSamplingCountsThroughStep(step, coarseStepLimit, variants) {
  const completed = Math.max(0, Math.round(Number(step) || 0));
  if (!variants.enabled) return { front: completed, virtual: 0 };
  const eligible = completed;
  const completeCycles = Math.floor(eligible / variants.slots);
  const remainder = eligible % variants.slots;
  const virtual = completeCycles * variants.virtualSlots + virtualCameraBag(completeCycles, variants)
    .slice(0, remainder)
    .filter((direction) => direction >= 0).length;
  return { front: completed - virtual, virtual };
}

function virtualCameraCatalog(
  width = 1,
  height = 1,
  orbitRadius = DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE,
  variants = virtualCameraSamplingVariants(false),
) {
  const frame = {
    width: Math.max(1, Number(width) || 1),
    height: Math.max(1, Number(height) || 1),
  };
  const intrinsics = virtualCameraIntrinsics(width, height, variants.fovDegrees);
  const cameras = [{
    id: "front",
    kind: "front",
    pitch_degrees: 0,
    yaw_degrees: 0,
    teacher_coverage: 1,
    intrinsics: virtualFrontIntrinsics(width, height, variants.fovDegrees),
  }];
  for (const pose of virtualCameraFibonacciPoses(variants.cameraCount, variants.maxAngleDegrees)) {
    cameras.push({
      id: pose.id,
      kind: "virtual",
      pitch_degrees: pose.pitchDegrees,
      yaw_degrees: pose.yawDegrees,
      polar_degrees: pose.polarDegrees,
      azimuth_degrees: pose.azimuthDegrees,
      teacher_coverage: virtualTeacherCoverage(
        frame,
        pose,
        orbitRadius,
        variants.fovDegrees,
      ),
      intrinsics: { ...intrinsics, projection: "perspective-virtual" },
    });
  }
  return cameras;
}

function virtualDepthCameraConfidence(camera, teacherCoverage = 1) {
  if (camera?.kind !== "virtual") return 0;
  const coverage = Math.max(0, Math.min(1, Number(teacherCoverage) || 0));
  const polarDegrees = Math.max(0, Math.min(89.9,
    Number(camera.polarDegrees ?? camera.polar_degrees ?? camera.angleDegrees) || 0));
  const angleConfidence = Math.max(0, Math.cos(polarDegrees * Math.PI / 180));
  return Math.max(
    MIN_VIRTUAL_DEPTH_CAMERA_CONFIDENCE,
    Math.min(1, Math.sqrt(coverage) * angleConfidence),
  );
}

function phase40Variants() {
  const overrides = qaOverrides("__flatPhotoPhase40");
  const query = new URLSearchParams(location.search);
  const qa = QA_RUNTIME_ENABLED && query.has("qa");
  const enabled = (name, fallback, queryName) => {
    if (typeof overrides[name] === "boolean") return overrides[name];
    if (qa && query.has(queryName)) return query.get(queryName) !== "0";
    return fallback;
  };
  const finite = (name, fallback, queryName) => {
    if (Number.isFinite(Number(overrides[name]))) return Number(overrides[name]);
    if (qa && query.has(queryName) && Number.isFinite(Number(query.get(queryName)))) return Number(query.get(queryName));
    return fallback;
  };
  const localColorAnchor = enabled("localColorAnchor", true, "phase40Anchor");
  const localColorAnchorDefault = algorithmUsesVirtualCameras()
    ? DEFAULT_VIRTUAL_LOCAL_COLOR_ANCHOR_WEIGHT
    : DEFAULT_LOCAL_COLOR_ANCHOR_WEIGHT;
  const alphaLoss = !algorithmUsesLayeredOpaqueBrush() && enabled("alphaLoss", true, "phase40Alpha");
  const alphaLossInput = Number(document.querySelector("#alphaLossWeight")?.value);
  return {
    localColorAnchor,
    localColorAnchorWeight: localColorAnchor
      ? Math.max(0, Math.min(0.2, finite("localColorAnchorWeight", localColorAnchorDefault, "phase40AnchorWeight")))
      : 0,
    alphaLoss,
    alphaLossWeight: alphaLoss
      ? Math.max(0, Math.min(1, finite("alphaLossWeight", Number.isFinite(alphaLossInput) ? alphaLossInput : DEFAULT_ALPHA_LOSS_WEIGHT, "phase40AlphaWeight")))
      : 0,
    overlapDiagnostics: enabled("overlapDiagnostics", true, "phase40Diagnostics"),
  };
}

function phase45Variants() {
  const overrides = qaOverrides("__flatPhotoPhase45");
  const enabled = (name, fallback) => (typeof overrides[name] === "boolean" ? overrides[name] : fallback);
  const finite = (name, fallback) => (Number.isFinite(Number(overrides[name])) ? Number(overrides[name]) : fallback);
  return {
    // Telemetry-only: this flag may collect a report, but must never gate ADC behavior.
    telemetry: enabled("telemetry", false),
    donorEligibility: enabled("donorEligibility", false),
    donorQuantile: Math.max(0, Math.min(1, finite("donorQuantile", 1 / 32))),
    recipientScore: enabled("recipientScore", false),
    recipientStrength: Math.max(0, Math.min(1, finite("recipientStrength", 0))),
    quotaFloor: Math.max(0, Math.min(1, finite("quotaFloor", 0))),
    firstResetOnly: enabled("firstResetOnly", false),
    seedOffset: Math.max(0, Math.min(1024, Math.floor(finite("seedOffset", 0)))),
  };
}

function phase46Variants() {
  const overrides = qaOverrides("__flatPhotoPhase46");
  const enabled = (name, fallback) => (typeof overrides[name] === "boolean" ? overrides[name] : fallback);
  const finite = (name, fallback) => (Number.isFinite(Number(overrides[name])) ? Number(overrides[name]) : fallback);
  const intervalInput = Number(document.querySelector("#layerUpdateInterval")?.value);
  return {
    layerUpdateInterval: Math.max(1, Math.min(3000, Math.round(finite(
      "layerUpdateInterval",
      Number.isFinite(intervalInput) ? intervalInput : LAYER_TRAIN_INTERVAL,
    )))),
    layerUpdateRate: Math.max(0, Math.min(1, finite("layerUpdateRate", 0.01))),
    stageAwareRate: enabled("stageAwareRate", false),
    freezeFraction: Math.max(0, Math.min(1, finite("freezeFraction", 1))),
  };
}

function discreteLayerSettings() {
  const opaqueLayered = algorithmUsesOpaqueLayeredPaint();
  const requested = Boolean(document.querySelector("#discreteLayers")?.checked);
  const accumulationRequested = Boolean(document.querySelector("#layerAwareAccumulation")?.checked);
  const rawCount = Number(document.querySelector("#discreteLayerCount")?.value);
  const rawMoveRadius = Number(document.querySelector("#discreteLayerMoveRadius")?.value);
  const count = Math.max(
    MIN_DISCRETE_LAYER_COUNT,
    Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(Number.isFinite(rawCount) ? rawCount : DEFAULT_DISCRETE_LAYER_COUNT)),
  );
  return {
    opaqueLayered,
    currentVisibilityChildPolicyEnabled:
      opaqueLayered && opaquePaintCurrentVisibilityChildPolicyEnabled(),
    currentVisibilityCompactionEnabled:
      opaqueLayered && opaquePaintCurrentVisibilityCompactionEnabled(),
    opaquePaintSettleFraction: opaqueLayered ? opaquePaintLateSettleFraction() : 0,
    enabled: (opaqueLayered || requested) && Boolean(document.querySelector("#trainLayerOrder")?.checked) && !algorithmUsesVirtualCameras(),
    accumulationEnabled: (opaqueLayered || accumulationRequested) && Boolean(document.querySelector("#trainLayerOrder")?.checked) && !algorithmUsesVirtualCameras(),
    count,
    moveRadius: opaqueLayered
      ? Math.max(
        LAYERED_OPAQUE_BRUSH_LAYER_MOVE_RADIUS,
        Math.min(count - 1, Math.round(
          Number.isFinite(rawMoveRadius) ? rawMoveRadius : LAYERED_OPAQUE_BRUSH_LAYER_MOVE_RADIUS,
        )),
      )
      : Math.max(0, Math.min(count - 1, Math.round(
        Number.isFinite(rawMoveRadius) ? rawMoveRadius : DEFAULT_DISCRETE_LAYER_MOVE_RADIUS,
      ))),
  };
}

function quantizeLayerOrder(order, layerCount) {
  const count = Math.max(MIN_DISCRETE_LAYER_COUNT, Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(layerCount)));
  const normalized = Math.max(0, Math.min(1 - Number.EPSILON, order));
  const scaled = normalized * count;
  const layer = Math.min(count - 1, Math.floor(scaled));
  const inLayerOrder = scaled - layer;
  return (layer + inLayerOrder * 0.999999) / count;
}

function layerOptimizationSettings(
  step,
  steps,
  stage,
  variants = phase46Variants(),
  opaqueLayered = false,
  settleFraction = OPAQUE_PAINT_LATE_SETTLE_FRACTION,
) {
  const stageMultiplier = variants.stageAwareRate
    ? stage === "coarse" ? 1 : stage === "mid" ? 0.5 : 0.2
    : 1;
  const freezeStep = Math.round(steps * variants.freezeFraction);
  const scheduled = variants.freezeFraction > 0 && step <= freezeStep && step % variants.layerUpdateInterval === 0;
  const settleAllowsMutation = opaquePaintStructuralMutationAllowed(
    step,
    steps,
    opaqueLayered,
    settleFraction,
  );
  const enabled = variants.freezeFraction > 0 && step <= freezeStep && settleAllowsMutation;
  return {
    interval: variants.layerUpdateInterval,
    rate: variants.layerUpdateRate * stageMultiplier,
    enabled,
    due: enabled && step % variants.layerUpdateInterval === 0,
    scheduled,
    suppressedByLateSettle: scheduled && !settleAllowsMutation,
  };
}

function qualityRecoveryVariants() {
  const overrides = qaOverrides("__image2GaussianQuality");
  const query = new URLSearchParams(location.search);
  const booleanVariant = (name, fallback) => {
    if (typeof overrides[name] === "boolean") return overrides[name];
    if (QA_RUNTIME_ENABLED && query.has(name)) return query.get(name) !== "0";
    return fallback;
  };
  const numericVariant = (name, fallback) => {
    const override = Number(overrides[name]);
    if (Number.isFinite(override)) return override;
    const queryValue = Number(query.get(name));
    if (QA_RUNTIME_ENABLED && query.has(name) && Number.isFinite(queryValue)) return queryValue;
    return fallback;
  };
  return {
    exactBackward: booleanVariant("exactBackward", true),
    surfaceAnisotropy: Math.max(1, Math.min(DEFAULT_MAX_ANISOTROPY, numericVariant("surfaceAnisotropy", DEFAULT_SURFACE_ANISOTROPY))),
  };
}

function experimentalVariants() {
  return {
    densitySignal: "gradient-error",
    massLaw: "legacy-area",
    mipGradient: "full",
    sgldNoiseLr: DEFAULT_SGLD_NOISE_LR,
    sgldTail: false,
    adcRecoveryDecaySteps: ADC_RECOVERY_DECAY_STEPS,
    phase33: phase33Variants(),
    phase37: phase37Variants(),
    phase38: phase38Variants(),
    phase39: phase39Variants(),
    phase40: phase40Variants(),
    phase45: phase45Variants(),
    virtualTilt: virtualTiltVariants(),
    virtualCameraSampling: virtualCameraSamplingVariants(),
    qualityRecovery: qualityRecoveryVariants(),
  };
}

const els = {
  appRoot: document.querySelector("main.app"),
  viewer: document.querySelector(".viewer"),
  viewControls: document.querySelector(".view-controls"),
  previewCanvas: document.querySelector("#previewCanvas"),
  gpuCanvas: document.querySelector("#gpuCanvas"),
  tiltCanvas: document.querySelector("#tiltCanvas"),
  tiltTeacherCanvas: document.querySelector("#tiltTeacherCanvas"),
  tiltFrameOverlay: document.querySelector("#tiltFrameOverlay"),
  previewImageFrame: document.querySelector("#previewImageFrame"),
  actualSizeButton: document.querySelector("#actualSizeButton"),
  fitViewButton: document.querySelector("#fitViewButton"),
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  trainSize: document.querySelector("#trainSize"),
  initialSplatCount: document.querySelector("#initialSplatCount"),
  adaptiveGridInitializationFraction: document.querySelector("#adaptiveGridInitializationFraction"),
  finalSplatCount: document.querySelector("#finalSplatCount"),
  capacityMode: document.querySelector("#capacityMode"),
  algorithmSelect: document.querySelector("#algorithmSelect"),
  stepCount: document.querySelector("#stepCount"),
  previewRefresh: document.querySelector("#previewRefresh"),
  liveQualityMetrics: document.querySelector("#liveQualityMetrics"),
  memoryLimiterUnlock: document.querySelector("#memoryLimiterUnlock"),
  currentContributionCompaction: document.querySelector("#currentContributionCompaction"),
  currentContributionCompactionStart: document.querySelector("#currentContributionCompactionStart"),
  currentContributionCompactionInterval: document.querySelector("#currentContributionCompactionInterval"),
  currentContributionCompactionMaxRemoval: document.querySelector("#currentContributionCompactionMaxRemoval"),
  currentContributionCompactionNearZero: document.querySelector("#currentContributionCompactionNearZero"),
  currentContributionCompactionWindow: document.querySelector("#currentContributionCompactionWindow"),
  tileCullingToggle: document.querySelector("#tileCullingToggle"),
  trainLayerOrder: document.querySelector("#trainLayerOrder"),
  layerAwareAccumulation: document.querySelector("#layerAwareAccumulation"),
  discreteLayers: document.querySelector("#discreteLayers"),
  discreteLayerCount: document.querySelector("#discreteLayerCount"),
  discreteLayerMoveRadius: document.querySelector("#discreteLayerMoveRadius"),
  layerUpdateInterval: document.querySelector("#layerUpdateInterval"),
  phaseRelativeScaleGuard: document.querySelector("#phaseRelativeScaleGuard"),
  p1RelativeScaleFloorRatio: document.querySelector("#p1RelativeScaleFloorRatio"),
  p2RelativeScaleFloorRatio: document.querySelector("#p2RelativeScaleFloorRatio"),
  p3RelativeScaleFloorRatio: document.querySelector("#p3RelativeScaleFloorRatio"),
  positionLearningRate: document.querySelector("#positionLearningRate"),
  colorLearningRate: document.querySelector("#colorLearningRate"),
  opacityLearningRate: document.querySelector("#opacityLearningRate"),
  alphaLossWeight: document.querySelector("#alphaLossWeight"),
  opaquePaintPanel: document.querySelector("#opaquePaintPanel"),
  opaquePaintSummary: document.querySelector("#opaquePaintSummary"),
  rectangleOpacityGradientMin: document.querySelector("#rectangleOpacityGradientMin"),
  rectangleOpacityGradientMax: document.querySelector("#rectangleOpacityGradientMax"),
  rectangleCenterOpacityGradientMin: document.querySelector("#rectangleCenterOpacityGradientMin"),
  rectangleCenterOpacityGradientMax: document.querySelector("#rectangleCenterOpacityGradientMax"),
  rectangleLearnedOpacityMin: document.querySelector("#rectangleLearnedOpacityMin"),
  rectangleLearnedOpacityMax: document.querySelector("#rectangleLearnedOpacityMax"),
  layeredBrushOpacityGradientStart: document.querySelector("#layeredBrushOpacityGradientStart"),
  layeredBrushOpacityGradientEnd: document.querySelector("#layeredBrushOpacityGradientEnd"),
  layeredBrushCenterOpacityGradientMin: document.querySelector("#layeredBrushCenterOpacityGradientMin"),
  layeredBrushCenterOpacityGradientMax: document.querySelector("#layeredBrushCenterOpacityGradientMax"),
  layeredBrushLearnedOpacityMin: document.querySelector("#layeredBrushLearnedOpacityMin"),
  layeredBrushLearnedOpacityMax: document.querySelector("#layeredBrushLearnedOpacityMax"),
  layeredBrushWidthTaperStart: document.querySelector("#layeredBrushWidthTaperStart"),
  layeredBrushWidthTaperEnd: document.querySelector("#layeredBrushWidthTaperEnd"),
  layeredBrushLocalColorFlow: document.querySelector("#layeredBrushLocalColorFlow"),
  layeredBrushStrokePersistence: document.querySelector("#layeredBrushStrokePersistence"),
  layeredBrushMinAspectRatio: document.querySelector("#layeredBrushMinAspectRatio"),
  layeredBrushMaxAspectRatio: document.querySelector("#layeredBrushMaxAspectRatio"),
  layeredBrushRibbonAspectFloor: document.querySelector("#layeredBrushRibbonAspectFloor"),
  layeredBrushAccentAspectFloor: document.querySelector("#layeredBrushAccentAspectFloor"),
  monochromeUnderpainting: document.querySelector("#monochromeUnderpainting"),
  colorFinishStart: document.querySelector("#colorFinishStart"),
  rectangleTopRatio: document.querySelector("#rectangleTopRatio"),
  rectangleTopRatioMax: document.querySelector("#rectangleTopRatioMax"),
  rectangleMinAspectRatio: document.querySelector("#rectangleMinAspectRatio"),
  rectangleMaxAspectRatio: document.querySelector("#rectangleMaxAspectRatio"),
  rectangleOrientation: document.querySelector("#rectangleOrientation"),
  rectanglePreserveArea: document.querySelector("#rectanglePreserveArea"),
  rectangleEdgeDirectedTaper: document.querySelector("#rectangleEdgeDirectedTaper"),
  rectangleStructureAwareRatio: document.querySelector("#rectangleStructureAwareRatio"),
  rectangleAsymmetricSoftness: document.querySelector("#rectangleAsymmetricSoftness"),
  scaleBiasedSurfaceLayerPrior: document.querySelector("#scaleBiasedSurfaceLayerPrior"),
  scaleBiasedSurfaceLayerPriorLayers: document.querySelector("#scaleBiasedSurfaceLayerPriorLayers"),
  scaleBiasedSurfaceLayerPriorP1Interval: document.querySelector("#scaleBiasedSurfaceLayerPriorP1Interval"),
  scaleBiasedSurfaceLayerPriorP2Interval: document.querySelector("#scaleBiasedSurfaceLayerPriorP2Interval"),
  scaleBiasedSurfaceLayerPriorP3Interval: document.querySelector("#scaleBiasedSurfaceLayerPriorP3Interval"),
  scaleBiasedSurfaceLayerPriorUntil: document.querySelector("#scaleBiasedSurfaceLayerPriorUntil"),
  virtualBoundedDepth: document.querySelector("#virtualBoundedDepth"),
  virtualGofDensity: document.querySelector("#virtualGofDensity"),
  virtualCameraShare: document.querySelector("#virtualCameraShare"),
  virtualCameraMaxAngle: document.querySelector("#virtualCameraMaxAngle"),
  virtualCameraCount: document.querySelector("#virtualCameraCount"),
  virtualCameraFov: document.querySelector("#virtualCameraFov"),
  virtualCameraCoverageEstimate: document.querySelector("#virtualCameraCoverageEstimate"),
  stageAwareGrowth: document.querySelector("#stageAwareGrowth"),
  stageGrowthP1: document.querySelector("#stageGrowthP1"),
  stageGrowthP2: document.querySelector("#stageGrowthP2"),
  stageGrowthP3: document.querySelector("#stageGrowthP3"),
  structureGuidedAllocation: document.querySelector("#structureGuidedAllocation"),
  structureRegionGrid: document.querySelector("#structureRegionGrid"),
  midTrainingOverdensityCorrection: document.querySelector("#midTrainingOverdensityCorrection"),
  overdensityCorrectionSchedule: document.querySelector("#overdensityCorrectionSchedule"),
  overdensityCorrectionInterval: document.querySelector("#overdensityCorrectionInterval"),
  overdensityDonorPercent: document.querySelector("#overdensityDonorPercent"),
  scaleLearningRate: document.querySelector("#scaleLearningRate"),
  rotationLearningRate: document.querySelector("#rotationLearningRate"),
  thetaAlignRate: document.querySelector("#thetaAlignRate"),
  maxAnisotropy: document.querySelector("#maxAnisotropy"),
  maxPlanarScale: document.querySelector("#maxPlanarScale"),
  boundarySigma: document.querySelector("#boundarySigma"),
  detailCoherence: document.querySelector("#detailCoherence"),
  densifyInterval: document.querySelector("#densifyInterval"),
  growthPercentage: document.querySelector("#growthPercentage"),
  growthApplyUntil: document.querySelector("#growthApplyUntil"),
  growthSignalThreshold: document.querySelector("#growthSignalThreshold"),
  retryWebGpuButton: document.querySelector("#retryWebGpuButton"),
  gpuLimitText: document.querySelector("#gpuLimitText"),
  memoryEstimateText: document.querySelector("#memoryEstimateText"),
  headroomText: document.querySelector("#headroomText"),
  limiterText: document.querySelector("#limiterText"),
  recommendationText: document.querySelector("#recommendationText"),
  speedEstimateText: document.querySelector("#speedEstimateText"),
  detailCapText: document.querySelector("#detailCapText"),
  capacityProbeText: document.querySelector("#capacityProbeText"),
  tileReserveText: document.querySelector("#tileReserveText"),
  measuredSpeedText: document.querySelector("#measuredSpeedText"),
  budgetNote: document.querySelector("#budgetNote"),
  pathInput: document.querySelector("#pathInput"),
  loadImageButton: document.querySelector("#loadImageButton"),
  clearImageButton: document.querySelector("#clearImageButton"),
  sampleButton: document.querySelector("#sampleButton"),
  pathButton: document.querySelector("#pathButton"),
  splatsPreviewButton: document.querySelector("#splatsPreviewButton"),
  originalPreviewButton: document.querySelector("#originalPreviewButton"),
  outsidePreviewToggle: document.querySelector("#outsidePreviewToggle"),
  startButton: document.querySelector("#startButton"),
  pauseButton: document.querySelector("#pauseButton"),
  stopButton: document.querySelector("#stopButton"),
  resetButton: document.querySelector("#resetButton"),
  savePngButton: document.querySelector("#savePngButton"),
  savePlyButton: document.querySelector("#savePlyButton"),
  pngExportResolution: document.querySelector("#pngExportResolution"),
  pngExportLongSide: document.querySelector("#pngExportLongSide"),
  pngExportResolutionStatus: document.querySelector("#pngExportResolutionStatus"),
  stepText: document.querySelector("#stepText"),
  splatText: document.querySelector("#splatText"),
  imageSizeText: document.querySelector("#imageSizeText"),
  lossText: document.querySelector("#lossText"),
  psnrText: document.querySelector("#psnrText"),
  ssimText: document.querySelector("#ssimText"),
  regionalSsimText: document.querySelector("#regionalSsimText"),
  coverageText: document.querySelector("#coverageText"),
  trainingTimingText: document.querySelector("#trainingTimingText"),
  gpuMemoryText: document.querySelector("#gpuMemoryText"),
  boundaryText: document.querySelector("#boundaryText"),
  backendText: document.querySelector("#backendText"),
  webGpuNotice: document.querySelector("#webGpuNotice"),
  webGpuNoticeText: document.querySelector("#webGpuNoticeText"),
  statusText: document.querySelector("#statusText"),
  trainingStatusDetails: document.querySelector("#trainingStatusDetails"),
  log: document.querySelector("#log"),
  clearLogButton: document.querySelector("#clearLogButton"),
  logTabs: document.querySelector("#logTabs"),
  trainingLogTab: document.querySelector("#trainingLogTab"),
  eventLogTab: document.querySelector("#eventLogTab"),
  splatsTab: document.querySelector("#splatsTab"),
  exportTab: document.querySelector("#exportTab"),
  tiltTab: document.querySelector("#tiltTab"),
  trainingLogPanel: document.querySelector("#trainingLogPanel"),
  eventLogPanel: document.querySelector("#eventLogPanel"),
  splatsPanel: document.querySelector("#splatsPanel"),
  exportPanel: document.querySelector("#exportPanel"),
  tiltPanel: document.querySelector("#tiltPanel"),
  tiltPitch: document.querySelector("#tiltPitch"),
  tiltPitchValue: document.querySelector("#tiltPitchValue"),
  tiltYaw: document.querySelector("#tiltYaw"),
  tiltYawValue: document.querySelector("#tiltYawValue"),
  tiltFrontButton: document.querySelector("#tiltFrontButton"),
  tiltRefreshButton: document.querySelector("#tiltRefreshButton"),
  tiltStatus: document.querySelector("#tiltStatus"),
  tiltCameraSummary: document.querySelector("#tiltCameraSummary"),
  tiltCameraMode: document.querySelector("#tiltCameraMode"),
  tiltPositionValue: document.querySelector("#tiltPositionValue"),
  tiltRadiusValue: document.querySelector("#tiltRadiusValue"),
  tiltFovValue: document.querySelector("#tiltFovValue"),
  tiltProjectionError: document.querySelector("#tiltProjectionError"),
  tiltOriginalView: document.querySelector("#tiltOriginalView"),
  tiltOverlayView: document.querySelector("#tiltOverlayView"),
  tiltSplatsView: document.querySelector("#tiltSplatsView"),
  tiltTrainingViewsButton: document.querySelector("#tiltTrainingViewsButton"),
  tiltTrainingViewsProgress: document.querySelector("#tiltTrainingViewsProgress"),
  tiltTrainingViewsSummary: document.querySelector("#tiltTrainingViewsSummary"),
  tiltContactSheet: document.querySelector("#tiltContactSheet"),
  tiltCameraMarkers: document.querySelector("#tiltCameraMarkers"),
  tiltRadiusMode: document.querySelector("#tiltRadiusMode"),
  splatsEmpty: document.querySelector("#splatsEmpty"),
  splatsContent: document.querySelector("#splatsContent"),
  splatsMeta: document.querySelector("#splatsMeta"),
  splatParameterEffects: document.querySelector("#splatParameterEffects"),
  splatAlphaBackground: document.querySelector("#splatAlphaBackground"),
  splatSmallFirstOrder: document.querySelector("#splatSmallFirstOrder"),
  splatShapeGaussian: document.querySelector("#splatShapeGaussian"),
  splatShapeRectangle: document.querySelector("#splatShapeRectangle"),
  splatShapeOpaqueBrush: document.querySelector("#splatShapeOpaqueBrush"),
  splatOpacity: document.querySelector("#splatOpacity"),
  splatOpacityValue: document.querySelector("#splatOpacityValue"),
  splatKernelFalloff: document.querySelector("#splatKernelFalloff"),
  splatKernelFalloffValue: document.querySelector("#splatKernelFalloffValue"),
  splatScale: document.querySelector("#splatScale"),
  splatScaleValue: document.querySelector("#splatScaleValue"),
  splatAspectRatio: document.querySelector("#splatAspectRatio"),
  splatAspectRatioValue: document.querySelector("#splatAspectRatioValue"),
  resetSplatAdjustments: document.querySelector("#resetSplatAdjustments"),
  splatAdjustStatus: document.querySelector("#splatAdjustStatus"),
  exportDescription: document.querySelector("#exportDescription"),
  exportCount: document.querySelector("#exportCount"),
  exportStatus: document.querySelector("#exportStatus"),
};

const previewCtx = els.previewCanvas.getContext("2d", { willReadFrequently: true });

function beginTrainingRun() {
  const run = {
    generation: ++state.trainingGeneration,
    renderer: state.webgpu.renderer,
    image: state.image,
    params: null,
    metrics: null,
    cancelled: false,
  };
  state.trainingRun = run;
  return run;
}

function updateTrainingRunOwnership(run, { image, params, metrics } = {}) {
  if (!run || state.trainingRun !== run) return;
  if (image !== undefined) run.image = image;
  if (params !== undefined) run.params = params;
  if (metrics !== undefined) run.metrics = metrics;
}

function ownsTrainingRun(run) {
  return Boolean(
    run &&
    !run.cancelled &&
    state.trainingRun === run &&
    state.trainingGeneration === run.generation &&
    state.webgpu.renderer === run.renderer &&
    state.image === run.image &&
    (!run.params || state.params === run.params) &&
    (!run.metrics || state.metrics === run.metrics) &&
    !run.renderer?.deviceLost,
  );
}

function trainingRunCancelledError() {
  const error = new Error("Training run was cancelled because its WebGPU lifecycle changed.");
  error.trainingRunCancelled = true;
  return error;
}

function assertTrainingRun(run) {
  if (run && !ownsTrainingRun(run)) throw trainingRunCancelledError();
}

async function awaitTrainingRun(run, promise) {
  assertTrainingRun(run);
  const value = await promise;
  assertTrainingRun(run);
  return value;
}

function invalidateTrainingRun(reason = "lifecycle change") {
  const run = state.trainingRun;
  if (!run) return false;
  run.cancelled = true;
  state.trainingRun = null;
  state.trainingGeneration += 1;
  state.startPending = false;
  log(`training lifecycle invalidated: ${reason}`);
  return true;
}

function trainingLifecycleInputLocked() {
  return state.running || state.startPending || state.imageLoading || state.webGpuRecoveryPending;
}

function previewModeInputLocked() {
  // Original/Splats is display-only and is safe while the optimizer owns the
  // training buffers. startPending intentionally spans the whole async run,
  // so only its pre-running upload window lacks a stable presentation source.
  return state.webGpuRecoveryPending || (state.startPending && !state.running);
}

function isExpectedPreviewCancellation(error) {
  if (error?.trainingRunCancelled || state.webGpuRecoveryPending || state.webgpu.renderer?.deviceLost) return true;
  const message = String(error?.message || error || "").toLowerCase();
  return message.includes("device lost") ||
    message.includes("instance dropped") ||
    message.includes("webgpu lifecycle changed");
}

function log(message) {
  els.log.textContent = `${new Date().toLocaleTimeString()}  ${message}\n${els.log.textContent}`.slice(0, 4000);
}

function eventLog(message) {
  log(`[event] ${message}`);
}

function setStatus(status) {
  els.statusText.textContent = status;
  publishState();
}

function formatTrainingElapsed(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor((Number(milliseconds) || 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function resetTrainingTiming(blank = true) {
  state.trainingTiming = {
    elapsedMs: 0,
    iterationsPerSecond: 0,
    sampleElapsedMs: 0,
    sampleSteps: 0,
    lastStep: 0,
  };
  els.trainingTimingText.textContent = blank ? "- / -" : `- / ${formatTrainingElapsed(0)}`;
}

function recordTrainingTiming(step, elapsedMs) {
  const timing = state.trainingTiming;
  const stepDelta = Math.max(0, step - timing.lastStep);
  const activeMs = Math.max(0, Number(elapsedMs) || 0);
  timing.elapsedMs += activeMs;
  timing.sampleElapsedMs += activeMs;
  timing.sampleSteps += stepDelta;
  timing.lastStep = step;
  if (timing.sampleElapsedMs >= 250 || step === state.metrics?.steps_requested || state.stopRequested) {
    const currentRate = timing.sampleElapsedMs > 0
      ? (timing.sampleSteps * 1000) / timing.sampleElapsedMs
      : 0;
    timing.iterationsPerSecond = timing.iterationsPerSecond > 0
      ? timing.iterationsPerSecond * 0.65 + currentRate * 0.35
      : currentRate;
    timing.sampleElapsedMs = 0;
    timing.sampleSteps = 0;
  }
}

function syncTrainingTimingDisplay() {
  const timing = state.trainingTiming;
  if (!state.running && timing.lastStep === 0) return;
  const rate = timing.iterationsPerSecond > 0 ? `${timing.iterationsPerSecond.toFixed(1)} it/s` : "-";
  els.trainingTimingText.textContent = `${rate} / ${formatTrainingElapsed(timing.elapsedMs)}`;
}

function setTrainingMessage(message, kind = "info") {
  const text = String(message || "").toLowerCase();
  let status = "";
  if (kind === "error") status = text.includes("safety") || text.includes("blocked") ? "safety stopped" : "error";
  else if (text.includes("evaluating")) status = "evaluating";
  else if (text.includes("waiting")) status = "waiting";
  else if (text.includes("preparing")) status = "preparing";
  else if (text.includes("loading")) status = "loading";
  else if (text.includes("training on")) status = "running";
  else if (text.includes("complete")) status = "done";
  else if (text.includes("stopped")) status = "stopped";
  else if (text.includes("loaded") || text.includes("prepared") || text.includes("reset")) status = "image loaded";
  else if (text === "ready.") status = "idle";
  if (status && els.statusText.textContent !== status) setStatus(status);
}

function webGpuUnavailableMessage(reason = "") {
  const detail = String(reason || "").toLowerCase();
  if (detail.includes("navigator.gpu")) {
    return "WebGPU is unavailable in this browser. Training requires a current desktop browser with WebGPU enabled.";
  }
  if (detail.includes("adapter unavailable")) {
    return "No compatible WebGPU adapter was found. Check browser WebGPU support and the active GPU, then retry.";
  }
  return "WebGPU could not be initialized. Training is disabled, but image preview remains available.";
}

function syncDisplayedSsimMetrics() {
  const psnr = state.metrics?.latest_psnr ?? state.metrics?.final_psnr;
  const global = state.metrics?.latest_global_ssim ?? state.metrics?.final_global_ssim;
  const localP10 = state.metrics?.latest_regional_ssim?.p10 ?? state.metrics?.final_regional_ssim?.p10;
  if (Number.isFinite(psnr)) els.psnrText.textContent = `${psnr.toFixed(2)} dB`;
  if (Number.isFinite(global)) els.ssimText.textContent = global.toFixed(4);
  if (Number.isFinite(localP10)) els.regionalSsimText.textContent = localP10.toFixed(4);
}

function resetEvaluationStatusForNewTraining() {
  state.splatInspectorNonfiniteCache = null;
  state.metrics = null;
  state.lastGpuLoss = null;
  els.lossText.textContent = "-";
  els.psnrText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.coverageText.textContent = "- / -";
}

function createIdempotentDatasetProxy(dataset) {
  return new Proxy(dataset, {
    set(target, property, value) {
      // DOMStringMap coerces assigned values to strings. Compare its live
      // value rather than retaining a shadow cache, so external mutations are
      // still observed on the next synchronous publish.
      if (target[property] === String(value)) return true;
      target[property] = value;
      return true;
    },
  });
}

const publishedStateDataset = createIdempotentDatasetProxy(document.documentElement.dataset);

function publishState() {
  updateGpuMemoryStatus();
  updateCapacityStatus();
  syncTrainingTimingDisplay();
  const data = publishedStateDataset;
  data.status = els.statusText.textContent;
  data.productName = PRODUCT_NAME;
  data.algorithm = selectedAlgorithm().id;
  data.algorithmLabel = selectedAlgorithm().label;
  data.resultAlgorithm = trainedResultAlgorithm()?.id || "";
  data.backend = els.backendText.textContent;
  data.running = String(state.running);
  data.startPending = String(state.startPending);
  data.trainingGeneration = String(state.trainingGeneration);
  data.sampleLoading = String(state.sampleLoading);
  data.imageLoading = String(state.imageLoading);
  data.paused = String(state.paused);
  data.runtimeSettingsRevision = String(state.runtimeSettingsRevision);
  data.stopRequested = String(state.stopRequested);
  data.downloadsEnabled = String(state.exportReady && !state.exporting);
  data.exporting = String(state.exporting);
  data.aspectAwareGrid = String(aspectAwareGridEnabled());
  data.imageLoaded = String(Boolean(state.image));
  data.previewMode = state.previewMode;
  data.outsidePreviewEnabled = String(Boolean(els.outsidePreviewToggle.checked));
  data.outsidePreviewActive = String(Boolean(
    els.outsidePreviewToggle.checked &&
    !state.running &&
    (state.previewPadding.x > 0 || state.previewPadding.y > 0),
  ));
  data.outsidePreviewPending = String(state.previewRefreshPending);
  data.previewRequestedRevision = String(state.previewRequestedRevision);
  data.previewAppliedRevision = String(state.previewAppliedRevision);
  data.previewAppliedAlphaBackground = state.previewAppliedAlphaBackground;
  data.webGpuRecoveryPending = String(state.webGpuRecoveryPending);
  data.webGpuRecoveryAttempts = String(state.webGpuRecoveryAttempts);
  data.postTrainingGpuRecoveries = String(state.metrics?.post_training_gpu_recoveries || 0);
  data.splatParameterEffectsEnabled = String(Boolean(els.splatParameterEffects?.checked));
  data.splatPreviewShape = els.splatParameterEffects?.checked ? state.splatPreviewShape : trainedSplatShape();
  data.previewPaddingX = String(state.previewPadding.x);
  data.previewPaddingY = String(state.previewPadding.y);
  data.previewCanvasWidth = String(state.previewPadding.width || els.gpuCanvas.width);
  data.previewCanvasHeight = String(state.previewPadding.height || els.gpuCanvas.height);
  data.previewOnlyBytes = String(state.previewPadding.bytes);
  data.previewTileMode = state.webgpu.renderer?.lastPreviewStats?.tile_mode || "";
  data.previewTileReferences = String(state.webgpu.renderer?.lastPreviewStats?.tile_references || 0);
  data.canvasViewMode = state.canvasView.mode;
  data.canvasViewScale = String(state.canvasView.scale);
  updateCanvasViewControls();
  updateTiltControlState();
  const lifecycleLocked = trainingLifecycleInputLocked();
  els.appRoot.inert = state.imageLoading;
  els.appRoot.setAttribute("aria-busy", String(state.imageLoading));
  const resultTabsLocked = lifecycleLocked;
  const splatsTabLocked = previewModeInputLocked() || !state.params;
  els.splatsTab.disabled = splatsTabLocked;
  els.exportTab.disabled = resultTabsLocked;
  els.splatsTab.setAttribute("aria-disabled", String(splatsTabLocked));
  els.exportTab.setAttribute("aria-disabled", String(resultTabsLocked));
  data.resultTabsLocked = String(resultTabsLocked);
  data.splatsTabLocked = String(splatsTabLocked);
  els.startButton.disabled = lifecycleLocked || !state.image || !state.webgpu.supported;
  els.sampleButton.disabled = lifecycleLocked || state.sampleLoading;
  els.loadImageButton.disabled = lifecycleLocked;
  els.pathButton.disabled = lifecycleLocked;
  els.fileInput.disabled = lifecycleLocked;
  els.resetButton.disabled = lifecycleLocked || state.previewRefreshPending || !state.image;
  els.clearImageButton.disabled = lifecycleLocked || state.previewRefreshPending || !state.image;
  const outsidePreviewReady = Boolean(
    state.image &&
    state.params &&
    state.webgpu.renderer &&
    !lifecycleLocked,
  );
  // Display-only toggles are safe while a slider refresh is pending: requests
  // are coalesced by the preview revision loop and do not change parameters.
  els.outsidePreviewToggle.disabled = !outsidePreviewReady;
  els.pauseButton.disabled = !state.running;
  const webGpuChecking = state.webgpu.reason === "checking";
  const webGpuUnavailable = !state.webgpu.supported && !webGpuChecking && !state.webGpuRecoveryPending;
  els.webGpuNotice.hidden = !webGpuUnavailable;
  if (webGpuUnavailable) {
    els.webGpuNoticeText.textContent = webGpuUnavailableMessage(state.webgpu.reason);
  }
  els.retryWebGpuButton.disabled = lifecycleLocked || state.webgpu.supported;
  els.pauseButton.textContent = state.paused ? "Resume" : "Pause";
  data.resetEnabled = String(!els.resetButton.disabled);
  data.clearImageEnabled = String(!els.clearImageButton.disabled);
  data.pauseEnabled = String(!els.pauseButton.disabled);
  data.step = els.stepText.textContent;
  data.loss = els.lossText.textContent;
  data.psnr = els.psnrText.textContent;
  data.ssim = els.ssimText.textContent;
  data.regionalSsimP10 = els.regionalSsimText.textContent;
  data.imageSize = els.imageSizeText.textContent;
  data.coverageStatus = els.coverageText.textContent;
  data.backgroundPixels = Number.isFinite(state.metrics?.coverage_stats?.background_exposure_ratio)
    ? `${(state.metrics.coverage_stats.background_exposure_ratio * 100).toFixed(2)}%`
    : "-";
  data.outsideSplats = Number.isFinite(state.metrics?.outside_render_splat_count)
    ? Number(state.metrics.outside_render_splat_count).toLocaleString()
    : "-";
  data.trainingTiming = els.trainingTimingText.textContent;
  data.currentIterationsPerSecond = String(state.trainingTiming.iterationsPerSecond || "");
  data.currentTrainingElapsedMs = String(state.trainingTiming.elapsedMs || 0);
  data.gpuActiveBytes = String(state.gpuMemory.activeBytes);
  data.gpuReservedBytes = String(state.gpuMemory.reservedBytes);
  data.gpuActiveMb = bytesToMB(state.gpuMemory.activeBytes).toFixed(1);
  data.gpuReservedMb = bytesToMB(state.gpuMemory.reservedBytes).toFixed(1);
  data.gpuTrainingActiveBytes = String(state.gpuMemory.trainingActiveBytes);
  data.gpuTrainingReservedBytes = String(state.gpuMemory.trainingReservedBytes);
  data.gpuResultActiveBytes = String(state.gpuMemory.resultActiveBytes);
  data.gpuResultReservedBytes = String(state.gpuMemory.resultReservedBytes);
  data.gpuMemoryScope = "tracked-persistent-app-buffers";
  data.gpuTransientResourcesIncluded = "false";
  const resultRenderMemory =
    state.webgpu.renderer?.resultRenderMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
  data.resultRenderCacheBytes = String(resultRenderMemory.activeBytes || 0);
  data.resultRenderCacheMb = bytesToMB(resultRenderMemory.activeBytes || 0).toFixed(1);
  data.resultRenderCacheSource = state.metrics?.result_render_cache?.source || "";
  data.resultRenderCacheOrder = state.metrics?.result_render_cache?.order_mode || "";
  data.splatBytes = state.params ? String(state.params.count * ROW_BYTES) : "0";
  data.brushTaperMinimum = Number.isFinite(state.metrics?.brush_taper_stats?.minimum)
    ? String(state.metrics.brush_taper_stats.minimum)
    : "";
  data.brushTaperMean = Number.isFinite(state.metrics?.brush_taper_stats?.mean)
    ? String(state.metrics.brush_taper_stats.mean)
    : "";
  data.brushTaperMaximum = Number.isFinite(state.metrics?.brush_taper_stats?.maximum)
    ? String(state.metrics.brush_taper_stats.maximum)
    : "";
  data.lastDownload = state.lastDownload;
  data.lastGpuLoss = state.lastGpuLoss === null ? "" : String(state.lastGpuLoss);
  // Keep the DOM on the same WebGPU metric snapshot used by logs and exports.
  // This repairs a stale label after a preview-only canvas redraw.
  syncDisplayedSsimMetrics();
  if (state.metrics) {
    els.splatText.textContent = `${state.metrics.num_gaussians} / ${state.metrics.final_splats}`;
  } else if (state.params) {
    els.splatText.textContent = String(state.params.count);
  } else {
    els.splatText.textContent = "-";
  }
  data.inputMode = state.lastInputMode;
  data.resizeMode = "max-side";
  data.initialSplatInput = els.initialSplatCount.value;
  data.finalSplatInput = els.finalSplatCount.value;
  data.capacityMode = els.capacityMode.value;
  data.capacityProbeStatus = state.capacityProbe.status;
  data.capacityProbeRequested = String(state.capacityProbe.requested || 0);
  data.capacityProbeSelected = String(state.capacityProbe.selected || 0);
  data.capacityProbeFastPath = String(Boolean(state.capacityProbe.fastPath));
  data.capacityProbeAttempts = String(state.capacityProbe.attempts.length);
  data.initializationMode = state.metrics?.initialization || "image-rgb-grid";
  data.adaptiveGridInitializationFraction = String(adaptiveGridInitializationVariants().fraction * 100);
  data.previewRefreshInput = els.previewRefresh.value;
  data.liveQualityMetricsInput = String(Boolean(els.liveQualityMetrics.checked));
  const contributionCompaction = currentContributionCompactionSettings();
  data.currentContributionCompactionInput = String(contributionCompaction.enabled);
  data.currentContributionCompactionStartInput = String(contributionCompaction.startFraction * 100);
  data.currentContributionCompactionIntervalInput = String(contributionCompaction.requestedIntervalSteps);
  data.currentContributionCompactionMaxRemovalInput = String(contributionCompaction.maxRemovalFraction * 100);
  data.currentContributionCompactionNearZeroInput = String(contributionCompaction.nearZeroMaxFraction * 100);
  data.currentContributionCompactionWindowInput = String(contributionCompaction.requestedWindowSteps);
  data.currentContributionCompactionEffectiveWindow = String(contributionCompaction.measurementWindowSteps);
  data.blendMode = "standard-alpha";
  data.gpuDensifyEnabled = "true";
  data.tileCullingEnabled = String(Boolean(els.tileCullingToggle?.checked));
  data.trainLayerOrderInput = String(Boolean(els.trainLayerOrder?.checked));
  data.layerAwareAccumulationInput = String(Boolean(els.layerAwareAccumulation?.checked));
  data.discreteLayersInput = String(Boolean(els.discreteLayers?.checked));
  data.discreteLayerCountInput = els.discreteLayerCount?.value || String(DEFAULT_DISCRETE_LAYER_COUNT);
  data.discreteLayerMoveRadiusInput = els.discreteLayerMoveRadius?.value || String(DEFAULT_DISCRETE_LAYER_MOVE_RADIUS);
  data.layerUpdateIntervalInput = els.layerUpdateInterval.value;
  data.positionLearningRateInput = els.positionLearningRate.value;
  data.colorLearningRateInput = els.colorLearningRate.value;
  data.opacityLearningRateInput = els.opacityLearningRate.value;
  data.alphaLossWeightInput = els.alphaLossWeight.value;
  const rectangleLearnedOpacity = selectedRectangleLearnedOpacity();
  const brushLearnedOpacity = selectedLayeredBrushLearnedOpacity();
  data.rectangleLearnedOpacityRange = `${rectangleLearnedOpacity.min},${rectangleLearnedOpacity.max}`;
  data.layeredBrushLearnedOpacityRange = `${brushLearnedOpacity.min},${brushLearnedOpacity.max}`;
  const brushDirectionalEffects = selectedLayeredBrushDirectionalEffects();
  const sharedColorWorkflow = selectedSharedColorWorkflow();
  data.layeredBrushOpacityGradientInput = String(brushDirectionalEffects.opacity);
  data.layeredBrushOpacityGradientRange = `${brushDirectionalEffects.opacityStart},${brushDirectionalEffects.opacityEnd}`;
  data.layeredBrushWidthTaperInput = String(brushDirectionalEffects.widthTaper);
  data.layeredBrushWidthTaperRange = `${brushDirectionalEffects.widthStart},${brushDirectionalEffects.widthEnd}`;
  data.layeredBrushLocalColorFlowInput = String(
    Boolean(els.layeredBrushLocalColorFlow?.checked),
  );
  data.layeredBrushStrokePersistenceInput = String(
    Boolean(els.layeredBrushStrokePersistence?.checked),
  );
  const brushAspectFloors = selectedBrushAspectFloors();
  data.layeredBrushMinAspectRatioInput = String(selectedBrushMinAspectRatio());
  data.layeredBrushMaxAspectRatioInput = String(selectedBrushMaxAspectRatio());
  data.layeredBrushRibbonAspectFloorInput = String(brushAspectFloors.ribbon);
  data.layeredBrushAccentAspectFloorInput = String(brushAspectFloors.accent);
  data.monochromeUnderpaintingInput = String(sharedColorWorkflow.monochromeUnderpainting);
  data.colorFinishStartInput = String(sharedColorWorkflow.colorFinishStartPercent);
  data.rectangleTopRatioInput = String(selectedRectangleTopRatio());
  data.rectangleTopRatioMaxInput =
    String(selectedRectangleTopRatioMax(selectedRectangleTopRatio()));
  data.rectangleMinAspectRatioInput = String(selectedRectangleMinAspectRatio());
  data.rectangleMaxAspectRatioInput =
    String(selectedRectangleMaxAspectRatio(selectedRectangleMinAspectRatio()));
  data.rectangleOrientationInput = selectedRectangleOrientation();
  const rectangleShape = selectedRectangleShapeSettings();
  data.rectanglePreserveAreaInput = String(rectangleShape.preserveArea);
  data.rectangleEdgeDirectedTaperInput = String(rectangleShape.edgeDirectedTaper);
  data.rectangleStructureAwareRatioInput = String(rectangleShape.structureAwareRatio);
  data.rectangleAsymmetricSoftnessInput = String(rectangleShape.asymmetricSoftness);
  const surfaceLayerPrior = scaleBiasedSurfaceLayerPriorSettings();
  data.scaleBiasedSurfaceLayerPriorInput = String(surfaceLayerPrior.enabled);
  data.scaleBiasedSurfaceLayerPriorLayersInput = String(surfaceLayerPrior.layers);
  data.scaleBiasedSurfaceLayerPriorP1IntervalInput = String(surfaceLayerPrior.p1Interval);
  data.scaleBiasedSurfaceLayerPriorP2IntervalInput = String(surfaceLayerPrior.p2Interval);
  data.scaleBiasedSurfaceLayerPriorP3IntervalInput = String(surfaceLayerPrior.p3Interval);
  data.scaleBiasedSurfaceLayerPriorUntilInput = String(surfaceLayerPrior.untilFraction * 100);
  data.virtualCameraSamplingInput = String(algorithmUsesVirtualCameras());
  data.virtual3dgsMultiviewInput = "true";
  data.virtualGofDensityInput = String(Boolean(els.virtualGofDensity.checked));
  data.virtualCameraShareInput = els.virtualCameraShare.value;
  data.virtualCameraMaxAngleInput = els.virtualCameraMaxAngle.value;
  data.virtualCameraCountInput = els.virtualCameraCount.value;
  data.virtualCameraFovInput = els.virtualCameraFov.value;
  data.virtualCameraCoverageMinimum = String(state.metrics?.virtual_camera_sampling?.teacher_coverage?.minimum ?? "");
  data.virtualCameraSamplingActive = String(Boolean(state.metrics?.virtual_camera_sampling?.enabled));
  data.virtualCameraActiveId = state.metrics?.virtual_camera_sampling?.active_camera_id || "";
  data.virtualCameraSamplingMode = state.metrics?.virtual_camera_sampling?.mode || "off";
  data.virtualCameraFrontSteps = String(state.metrics?.virtual_camera_sampling?.front_steps ?? 0);
  data.virtualCameraVirtualSteps = String(state.metrics?.virtual_camera_sampling?.virtual_steps ?? 0);
  data.virtualCameraEvaluationCount = String(state.metrics?.virtual_camera_evaluation?.virtual_views?.camera_count ?? 0);
  data.virtualCameraVirtualPsnr = String(state.metrics?.virtual_camera_evaluation?.virtual_views?.rgb_psnr_macro ?? "");
  data.virtualCameraVirtualSsim = String(state.metrics?.virtual_camera_evaluation?.virtual_views?.rgb_ssim_macro ?? "");
  data.virtualCameraVirtualSsimP10 = String(state.metrics?.virtual_camera_evaluation?.virtual_views?.rgb_ssim_p10 ?? "");
  data.virtualCameraAllViewSsim = String(state.metrics?.virtual_camera_evaluation?.all_views?.rgb_ssim_macro ?? "");
  data.virtualCameraAllViewPsnr = String(state.metrics?.virtual_camera_evaluation?.all_views?.rgb_psnr_macro ?? "");
  data.virtualCameraAllViewSsimP10 = String(state.metrics?.virtual_camera_evaluation?.all_views?.rgb_ssim_p10 ?? "");
  const virtualEvaluation = state.metrics?.virtual_camera_evaluation;
  data.virtualCameraEvaluationSummary = virtualEvaluation
    ? JSON.stringify({
      backend: virtualEvaluation.backend,
      target: virtualEvaluation.target,
      aggregation: virtualEvaluation.aggregation,
      front_view: virtualEvaluation.front_view,
      virtual_views: virtualEvaluation.virtual_views,
      all_views: virtualEvaluation.all_views,
    })
    : "";
  const obliqueViews = state.metrics?.oblique_overlap_diagnostics?.views;
  data.virtualCameraTiltSummary = obliqueViews
    ? JSON.stringify(Object.fromEntries(
      ["pitch-30", "yaw-30", "pitch-60", "yaw-60"].map((key) => [key, obliqueViews[key]?.["1"] ?? null]),
    ))
    : "";
  data.colorSpaceContract = state.metrics?.color_space_audit?.contract || "front and virtual teachers use the same sRGB signal values";
  data.frontLumaDelta = String(state.metrics?.color_space_audit?.training_minus_source_luma ?? "");
  data.renderParityAlphaMax = String(state.metrics?.render_surface_parity?.alpha_max_abs ?? "");
  data.renderParityPremultipliedMax = String(state.metrics?.render_surface_parity?.premultiplied_max_abs ?? "");
  data.renderParityMismatchPixels = String(state.metrics?.render_surface_parity?.premultiplied_mismatch_pixels ?? "");
  data.layerAwareParityPremultipliedMax = String(
    state.metrics?.layer_aware_render_parity?.premultiplied_max_abs ?? "",
  );
  data.renderParityMaximumPixel = state.metrics?.render_surface_parity?.maximum_pixel
    ? JSON.stringify(state.metrics.render_surface_parity.maximum_pixel)
    : "";
  data.stageAwareGrowthInput = String(Boolean(els.stageAwareGrowth.checked));
  const stageShares = phase39Variants().stageGrowthShares;
  data.stageGrowthShares = `${(stageShares.p1 * 100).toFixed(2)},${(stageShares.p2 * 100).toFixed(2)},${(stageShares.p3 * 100).toFixed(2)}`;
  data.structureGuidedAllocationInput = String(Boolean(els.structureGuidedAllocation.checked));
  data.structureRegionGridInput = els.structureRegionGrid.value;
  data.midTrainingOverdensityCorrectionInput = String(Boolean(els.midTrainingOverdensityCorrection.checked));
  data.overdensityCorrectionScheduleInput = els.overdensityCorrectionSchedule.value;
  data.overdensityCorrectionIntervalInput = els.overdensityCorrectionInterval.value;
  data.overdensityDonorPercentInput = els.overdensityDonorPercent.value;
  data.scaleLearningRateInput = els.scaleLearningRate.value;
  data.rotationLearningRateInput = els.rotationLearningRate.value;
  data.thetaAlignRateInput = els.thetaAlignRate.value;
  data.maxAnisotropyInput = els.maxAnisotropy.value;
  data.maxPlanarScaleInput = els.maxPlanarScale.value;
  data.boundarySigmaInput = els.boundarySigma.value;
  data.adaptiveDetailInput = "true";
  data.detailCoherenceInput = els.detailCoherence.value;
  data.densifyIntervalInput = els.densifyInterval.value;
  data.growthPercentageInput = els.growthPercentage.value;
  data.growthApplyUntilInput = els.growthApplyUntil.value;
  data.growthSignalThresholdInput = els.growthSignalThreshold.value;
  data.subgroupExactBackward = String(Boolean(state.webgpu.renderer?.subgroupExactBackwardEnabled));
  data.subgroupsAvailable = String(Boolean(state.webgpu.subgroups));
  data.safetyMode = state.safety.mode;
  data.safetyStopReason = state.safety.lastStopReason;
  data.safetyStopEstimateMb = state.safety.lastStopEstimateMB;
  data.safetyStopBudgetMb = state.safety.lastStopBudgetMB;
  data.safetyRecommended = state.safety.lastRecommended;
  data.memoryLimiterUnlocked = String(state.safety.memoryLimiterUnlocked);
  if (state.recommendation) {
    data.memoryBudgetMb = String(state.recommendation.budgetMB);
    data.memoryBudgetMode = String(state.recommendation.memoryBudgetMode);
    data.memoryBudgetMultiplier = String(state.recommendation.memoryBudgetMultiplier);
    data.memoryEstimatedMb = String(state.recommendation.estimatedMB);
    data.memoryHeadroomMb = String(state.recommendation.headroomMB);
    data.memoryHintSource = String(state.recommendation.memoryHintSource);
    data.memoryHintMb = String(state.recommendation.memoryHintMB);
    data.autoBudgetSource = String(state.recommendation.autoBudgetSource);
    data.autoBudgetReservedMb = String(state.recommendation.autoReservedMB);
    data.autoBudgetEnvelopeMb = String(state.recommendation.autoEnvelopeMB);
    data.deviceLimiterTrainSize = String(state.recommendation.limiterTrainSize);
    data.deviceLimiterFinalSplats = String(state.recommendation.limiterFinalSplats);
    data.recommendedTrainSize = String(state.recommendation.recommendedTrainSize);
    data.recommendedFinalSplats = String(state.recommendation.recommendedFinalSplats);
    data.metricsInterval = String(state.recommendation.metricInterval);
    data.speedWork = String(state.recommendation.previewWork);
    data.vramExact = "false";
  } else {
    data.memoryBudgetMb = "";
    data.memoryBudgetMode = "";
    data.memoryBudgetMultiplier = "";
    data.memoryEstimatedMb = "";
    data.memoryHeadroomMb = "";
    data.memoryHintSource = "";
    data.memoryHintMb = "";
    data.autoBudgetSource = "";
    data.autoBudgetReservedMb = "";
    data.autoBudgetEnvelopeMb = "";
    data.deviceLimiterTrainSize = "";
    data.deviceLimiterFinalSplats = "";
    data.recommendedTrainSize = "";
    data.recommendedFinalSplats = "";
    data.metricsInterval = "";
    data.speedWork = "";
    data.vramExact = "false";
  }
  if (state.metrics) {
    const lastExport = state.metrics.export_history?.[state.metrics.export_history.length - 1];
    const lastPerformance = state.metrics.performance_trace?.[state.metrics.performance_trace.length - 1];
    data.lastExportFormat = String(lastExport?.format ?? "");
    data.lastExportFilename = String(lastExport?.filename ?? "");
    data.lastExportBytes = String(lastExport?.bytes ?? "");
    data.lastExportWidth = String(lastExport?.width ?? "");
    data.lastExportHeight = String(lastExport?.height ?? "");
    data.lastExportNonblackPixels = String(lastExport?.nonblack_pixels ?? "");
    data.lastExportMeanRgb = String(lastExport?.mean_rgb ?? "");
    data.lastExportRoundTripCount = String(lastExport?.round_trip?.count ?? lastExport?.round_trip?.vertices ?? "");
    data.lastExportRoundTripShDegree = String(lastExport?.round_trip?.sh_degree ?? (lastExport?.round_trip?.sh_degree_0 ? 0 : ""));
    data.lastExportRoundTripZAbsMax = String(lastExport?.round_trip?.z_abs_max ?? (lastExport?.round_trip?.all_z_zero ? 0 : ""));
    data.lastExportRoundTripNonfinite = String(lastExport?.round_trip?.nonfinite?.length ?? (lastExport?.round_trip?.all_finite ? 0 : ""));
    data.initialL1 = String(state.metrics.initial_l1);
    data.finalL1 = String(state.metrics.final_l1);
    data.initialRgbMse = String(state.metrics.initial_rgb_mse ?? "");
    data.finalRgbMse = String(state.metrics.final_rgb_mse ?? "");
    data.initialPsnr = String(state.metrics.initial_psnr ?? "");
    data.finalPsnr = String(state.metrics.final_psnr ?? "");
    data.psnrUnit = state.metrics.quality_metric_contract?.psnr_unit || "dB";
    data.psnrSignalSpace = state.metrics.quality_metric_contract?.signal_space || "sRGB signal values";
    data.finalGlobalSsim = String(state.metrics.final_global_ssim ?? "");
    data.finalWindowedSsim = String(state.metrics.final_windowed_ssim ?? state.metrics.final_ssim ?? "");
    data.finalLocalP10 = String(state.metrics.final_regional_ssim?.p10 ?? "");
    data.finalAlphaL1 = String(state.metrics.final_alpha_l1 ?? "");
    data.finalAlphaSsim = String(state.metrics.final_alpha_ssim ?? "");
    data.trainingElapsedMs = String(lastPerformance?.elapsed_ms ?? "");
    data.trainingIterationsPerSecond = String(lastPerformance?.iterations_per_second ?? "");
    data.trainingEvaluationMode = state.metrics.training_evaluation?.mode || "";
    data.trainingEvaluationSource = state.metrics.training_evaluation?.source || "";
    data.trainingPeriodicEvaluations = String(
      state.metrics.training_evaluation?.periodic_full_image_evaluations ?? 0,
    );
    data.trainingResidualRefreshes = String(state.metrics.training_residual_map?.refresh_count ?? 0);
    data.trainingResidualCpuReadbacks = String(state.metrics.training_residual_map?.cpu_readbacks ?? 0);
    data.finalParameterHash = String(state.metrics.final_parameter_hash ?? "");
    data.stepsDone = String(state.metrics.steps_done);
    data.stepsRequested = String(state.metrics.steps_requested);
    data.stopped = String(state.metrics.stopped);
    data.gaussians = String(state.metrics.num_gaussians);
    data.initialSplats = String(state.metrics.initial_splats);
    data.initialization = String(state.metrics.initialization ?? "");
    data.finalSplats = String(state.metrics.final_splats);
    data.lossBackend = state.metrics.loss_backend || "";
    data.webgpuComputeLoss = String(Boolean(state.metrics.webgpu_compute_loss));
    data.webgpuTrainExecuted = String(Boolean(state.metrics.webgpu_train_executed));
    data.webgpuTrainUpdate = String(Boolean(state.metrics.webgpu_train_update));
    data.webgpuTrainError = state.metrics.webgpu_train_error || "";
    data.virtualTiltSteps = String(state.metrics.webgpu_train_stats?.virtual_tilt_steps_completed ?? 0);
    data.virtualTiltLastStep = String(state.metrics.webgpu_train_stats?.last_virtual_tilt?.step ?? "");
    data.virtualTiltLastPitch = String(state.metrics.webgpu_train_stats?.last_virtual_tilt?.pitchDegrees ?? "");
    data.virtualTiltLastYaw = String(state.metrics.webgpu_train_stats?.last_virtual_tilt?.yawDegrees ?? "");
    data.virtualTiltEnabled = String(Boolean(state.metrics.webgpu_train_stats?.virtual_tilt?.enabled));
    data.virtualTiltInterval = String(state.metrics.webgpu_train_stats?.virtual_tilt?.interval ?? "");
    data.curriculumCoarseSteps = String(state.metrics.webgpu_train_stats?.coarse_steps_completed ?? 0);
    data.curriculumMidSteps = String(state.metrics.webgpu_train_stats?.mid_steps_completed ?? 0);
    data.curriculumTrainingStage = String(state.metrics.webgpu_train_stats?.training_stage ?? "");
    data.ewa2x2 = String(Boolean(state.metrics.phase33_variants?.ewa2x2));
    data.exactBackward = String(Boolean(state.metrics.phase46_variants?.qualityRecovery?.exactBackward ?? qualityRecoveryVariants().exactBackward));
    data.tiltRiskCount = String(state.metrics.tilt_risk?.risky_count ?? 0);
    data.tiltRiskRatio = String(state.metrics.tilt_risk?.risky_ratio ?? 0);
    data.tiltRiskSupportP99 = String(state.metrics.tilt_risk?.support_depth_p99 ?? 0);
    data.experimentalPrefixPreserved = String(state.metrics.experimental_prefix_preserved !== false);
    data.densifyEvents = String(state.metrics.densify_events?.length || 0);
    data.growthScheduleMode = String(state.metrics.growth_schedule?.mode || "");
    data.growthPercentage = String(state.metrics.growth_schedule?.percentage ?? "");
    data.growthSignalThreshold = String(state.metrics.growth_schedule?.signal_threshold ?? "");
    data.growthThresholdSkips = String(state.metrics.growth_schedule?.threshold_skips ?? 0);
    data.stageAwareGrowth = String(Boolean(state.metrics.growth_schedule?.stage_aware));
    data.growthCapReachedStep = String(state.metrics.growth_schedule?.cap_reached_step ?? "");
    data.layerUpdateInterval = String(state.metrics.layer_update_interval ?? "");
    data.layerUpdateCount = String(state.metrics.layer_update_count ?? 0);
    data.layerUpdateFirstSteps = String(state.metrics.layer_update_first_steps ?? "");
    data.layerUpdateLastStep = String(state.metrics.layer_update_last_step ?? "");
    data.fusionAdcSplitEvents = String(state.metrics.fusion_events?.adc_split || 0);
    data.fusionAdcDuplicateEvents = String(state.metrics.fusion_events?.adc_duplicate || 0);
    data.fusionMcmcTeleportEvents = String(state.metrics.fusion_events?.mcmc_teleport || 0);
    data.fusionMcmcReseedEvents = String(state.metrics.fusion_events?.mcmc_reseed || 0);
    data.fusionPruneEvents = String(state.metrics.fusion_events?.prune || 0);
    data.fusionOpacityResetEvents = String(state.metrics.fusion_events?.opacity_reset || 0);
    data.tiltRiskCandidates = String(state.metrics.fusion_events?.tilt_risk_candidates || 0);
    data.tiltTrueSplits = String(state.metrics.fusion_events?.tilt_true_splits || 0);
    data.tiltOpacitySaturations = String(state.metrics.fusion_events?.tilt_opacity_saturations || 0);
    data.tileIndexTotal = String(state.metrics.tile_counters?.total ?? "");
    data.tileIndexCapacity = String(state.metrics.tile_counters?.capacity ?? "");
    data.tileIndexOverflow = String(state.metrics.tile_counters?.overflow ?? "");
    data.tileRetrySteps = String(state.metrics.tile_retry_steps ?? 0);
    data.tileRetryEvents = String(state.metrics.tile_retry_events?.length ?? 0);
    data.tileRetryParameterHashMatches = String(state.metrics.tile_retry_parameter_hash?.matches ?? "");
    data.qaTileIndexCapacity = String(state.metrics.qa_tile_index_capacity ?? "");
    data.tileAverageCandidates = String(state.metrics.tile_counters?.average_candidates ?? "");
    data.activeSplats = String(state.metrics.tile_counters?.active_count ?? state.metrics.num_gaussians ?? "");
    data.freeSplats = String(state.metrics.tile_counters?.free_count ?? "");
    data.ssimTrend = state.metrics.ssim_trend || "";
    data.globalSsimTrend = state.metrics.global_ssim_trend || "";
    data.psnrTrend = state.metrics.psnr_trend || "";
    data.initialSsim = String(state.metrics.initial_ssim);
    data.finalSsim = String(state.metrics.final_ssim);
    data.initialGlobalSsim = String(state.metrics.initial_global_ssim);
    data.finalGlobalSsim = String(state.metrics.final_global_ssim);
    data.initialWindowedSsim = String(state.metrics.initial_windowed_ssim);
    data.finalWindowedSsim = String(state.metrics.final_windowed_ssim);
    data.initialRegionalSsimP10 = String(state.metrics.initial_regional_ssim?.p10 ?? "");
    data.finalRegionalSsimP10 = String(state.metrics.final_regional_ssim?.p10 ?? "");
    data.finalRegionalSsimMinimum = String(state.metrics.final_regional_ssim?.minimum ?? "");
    data.finalRegionalSsimMedian = String(state.metrics.final_regional_ssim?.median ?? "");
    data.initialAlphaSsim = String(state.metrics.initial_alpha_ssim ?? "");
    data.finalAlphaSsim = String(state.metrics.final_alpha_ssim ?? "");
    data.initialAlphaL1 = String(state.metrics.initial_alpha_l1 ?? "");
    data.finalAlphaL1 = String(state.metrics.final_alpha_l1 ?? "");
    data.lrScale = String(state.metrics.lr_scale);
    data.positionLearningRate = String(state.metrics.learning_rates?.position ?? "");
    data.colorLearningRate = String(state.metrics.learning_rates?.color ?? "");
    data.opacityLearningRate = String(state.metrics.learning_rates?.opacity ?? "");
    data.alphaLossWeight = String(state.metrics.alpha_loss_weight ?? "");
    data.scaleLearningRate = String(state.metrics.learning_rates?.scale ?? "");
    data.rotationLearningRate = String(state.metrics.learning_rates?.rotation ?? "");
    data.thetaAlignRate = String(state.metrics.learning_rates?.thetaAlign ?? "");
    data.maxAnisotropy = String(state.metrics.learning_rates?.maxAnisotropy ?? "");
    data.boundarySigma = String(state.metrics.boundary_sigma ?? "");
    data.outsideRenderSplatCount = String(state.metrics.outside_render_splat_count ?? "");
    data.adaptiveDetail = String(Boolean(state.metrics.learning_rates?.adaptiveDetail));
    data.detailCoherence = String(state.metrics.learning_rates?.detailCoherence ?? "");
    data.detailSplatCount = String(state.metrics.detail_splat_count ?? "");
    data.detailSplatRatio = String(state.metrics.detail_splat_ratio ?? "");
    data.paramDeltaPosition = String(state.metrics.param_delta?.position ?? "");
    data.paramDeltaColor = String(state.metrics.param_delta?.color ?? "");
    data.paramDeltaOpacity = String(state.metrics.param_delta?.opacity ?? "");
    data.paramDeltaScale = String(state.metrics.param_delta?.scale ?? "");
    data.paramDeltaRotation = String(state.metrics.param_delta?.rotation ?? "");
    data.initialOrientationBins = JSON.stringify(state.metrics.initial_orientation?.bins ?? []);
    data.phaseOneShapeLrMultiplier = String(state.webgpu.renderer?.lastTrainStats?.phase_one_shape_lr_multiplier ?? "");
    data.phaseOneMaxPlanarScale = String(state.webgpu.renderer?.lastTrainStats?.phase_one_max_planar_scale ?? "");
    data.boundaryLeakCount = String(state.metrics.boundary_leak_count ?? "");
    data.boundaryMaxLeak = String(state.metrics.boundary_max_leak ?? "");
    data.backgroundExposureCount = String(state.metrics.coverage_stats?.background_exposure_count ?? "");
    data.backgroundExposureRatio = String(state.metrics.coverage_stats?.background_exposure_ratio ?? "");
    data.alphaDarkMean = String(state.metrics.coverage_stats?.luminance_buckets?.dark?.mean_alpha ?? "");
    data.alphaMidMean = String(state.metrics.coverage_stats?.luminance_buckets?.mid?.mean_alpha ?? "");
    data.alphaLightMean = String(state.metrics.coverage_stats?.luminance_buckets?.light?.mean_alpha ?? "");
    data.defaultOutput = state.metrics.default_output || "";
    data.lossCount = String(state.metrics.losses?.length || 0);
    data.ssimCount = String(state.metrics.ssim?.length || 0);
    data.windowedSsimCount = String(state.metrics.windowed_ssim?.length || 0);
    data.runAlgorithm = state.metrics.algorithm || "";
    data.inputOriginalWidth = String(state.metrics.input_original_size?.[0] || "");
    data.inputOriginalHeight = String(state.metrics.input_original_size?.[1] || "");
    data.resizeScale = String(state.metrics.resize_scale ?? "");
    data.previewRefresh = String(state.metrics.preview_refresh ?? "");
    data.previewFrames = String(state.metrics.preview_frames ?? 0);
    data.lastPreviewStep = String(state.metrics.last_preview_step ?? "");
    data.paramsRevision = String(state.metrics.params_revision ?? 0);
    data.coverageRevision = String(state.metrics.coverage_revision ?? "");
  } else {
    data.lastExportFormat = "";
    data.lastExportFilename = "";
    data.lastExportBytes = "";
    data.lastExportWidth = "";
    data.lastExportHeight = "";
    data.lastExportNonblackPixels = "";
    data.lastExportMeanRgb = "";
    data.lastExportRoundTripCount = "";
    data.lastExportRoundTripShDegree = "";
    data.lastExportRoundTripZAbsMax = "";
    data.lastExportRoundTripNonfinite = "";
    data.initialL1 = "";
    data.finalL1 = "";
    data.trainingEvaluationMode = "";
    data.trainingEvaluationSource = "";
    data.trainingPeriodicEvaluations = "";
    data.trainingResidualRefreshes = "";
    data.trainingResidualCpuReadbacks = "";
    data.finalParameterHash = "";
    data.stepsDone = "";
    data.stepsRequested = "";
    data.stopped = "";
    data.gaussians = "";
    data.initialSplats = "";
    data.initialization = "";
    data.finalSplats = "";
    data.lossBackend = "";
    data.webgpuComputeLoss = "";
    data.webgpuTrainExecuted = "";
    data.webgpuTrainUpdate = "";
    data.webgpuTrainError = "";
    data.experimentalPrefixPreserved = "";
    data.densifyEvents = "";
    data.fusionAdcSplitEvents = "";
    data.fusionAdcDuplicateEvents = "";
    data.fusionMcmcTeleportEvents = "";
    data.fusionMcmcReseedEvents = "";
    data.fusionPruneEvents = "";
    data.fusionOpacityResetEvents = "";
    data.tileIndexTotal = "";
    data.tileIndexCapacity = "";
    data.tileIndexOverflow = "";
    data.tileAverageCandidates = "";
    data.activeSplats = "";
    data.freeSplats = "";
    data.ssimTrend = "";
    data.globalSsimTrend = "";
    data.initialSsim = "";
    data.finalSsim = "";
    data.initialGlobalSsim = "";
    data.finalGlobalSsim = "";
    data.initialWindowedSsim = "";
    data.finalWindowedSsim = "";
    data.initialRegionalSsimP10 = "";
    data.finalRegionalSsimP10 = "";
    data.finalRegionalSsimMinimum = "";
    data.finalRegionalSsimMedian = "";
    data.lrScale = "";
    data.positionLearningRate = "";
    data.colorLearningRate = "";
    data.opacityLearningRate = "";
    data.scaleLearningRate = "";
    data.rotationLearningRate = "";
    data.thetaAlignRate = "";
    data.maxAnisotropy = "";
    data.boundarySigma = "";
    data.outsideRenderSplatCount = "";
    data.adaptiveDetail = "";
    data.detailCoherence = "";
    data.paramDeltaPosition = "";
    data.paramDeltaColor = "";
    data.paramDeltaOpacity = "";
    data.paramDeltaScale = "";
    data.paramDeltaRotation = "";
    data.initialOrientationBins = "";
    data.phaseOneShapeLrMultiplier = "";
    data.phaseOneMaxPlanarScale = "";
    data.boundaryLeakCount = "";
    data.boundaryMaxLeak = "";
    data.backgroundExposureCount = "";
    data.backgroundExposureRatio = "";
    data.defaultOutput = "";
    data.lossCount = "";
    data.ssimCount = "";
    data.windowedSsimCount = "";
    data.runAlgorithm = "";
    data.inputOriginalWidth = "";
    data.inputOriginalHeight = "";
    data.resizeScale = "";
    data.previewRefresh = "";
    data.previewFrames = "";
    data.lastPreviewStep = "";
    data.paramsRevision = "";
    data.coverageRevision = "";
  }
  updateExportPanel();
}

function resizedSize(width, height, longSide, maximumLongSide = LIMITS.trainSizeMax) {
  const safeLongSide = clampNumber(longSide, 1, maximumLongSide, Math.min(DEFAULT_MAX_SIDE, maximumLongSide));
  const scale = Math.min(1, safeLongSide / Math.max(width, height));
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

function resizeTargetSize(width, height) {
  const [targetWidth, targetHeight] = resizedSize(width, height, Number(els.trainSize.value));
  return {
    width: targetWidth,
    height: targetHeight,
    mode: "max-side",
    scale: Math.max(targetWidth / width, targetHeight / height),
  };
}

function syncTrainSizeUi() {
  els.trainSize.disabled = state.running;
  publishState();
}

function normalizeUiSplatCount(value, fallback = DEFAULT_FINAL_SPLATS, max = LIMITS.splatsMax) {
  return normalizeStepInteger(value, {
    min: LIMITS.splatsMin,
    max: Math.min(LIMITS.splatsMax, max),
    fallback,
    step: 4,
  });
}

function normalizeActiveSplatCount(value, fallback = DEFAULT_INITIAL_SPLATS, max = LIMITS.splatsMax) {
  return normalizeStepInteger(value, {
    min: LIMITS.splatsMin,
    max: Math.min(LIMITS.splatsMax, max),
    fallback,
    step: 1,
  });
}

function assertSplatCountContract(params, context) {
  const count = Number(params?.count);
  if (!Number.isSafeInteger(count) || count < LIMITS.splatsMin || count > LIMITS.splatsMax) {
    throw new Error(`${context}: invalid splat count ${String(params?.count)}`);
  }
  const requiredLengths = { xy: count * 2, scale: count * 2, rgb: count * 3, opacity: count, theta: count };
  for (const [name, length] of Object.entries(requiredLengths)) {
    if (!params[name] || params[name].length !== length) {
      throw new Error(`${context}: ${name} length ${params[name]?.length ?? "missing"} != ${length}`);
    }
  }
  for (const name of ["depthOrder", "virtualDepth", "detailTags"]) {
    if (params[name] && params[name].length !== count) {
      throw new Error(`${context}: ${name} length ${params[name].length} != ${count}`);
    }
  }
  return count;
}

function bytesToMB(bytes) {
  return bytes / MB;
}

function formatMB(bytes) {
  return `${bytesToMB(bytes).toFixed(bytes < 10 * MB ? 1 : 0)} MB`;
}

function updateGpuMemoryStatus() {
  const training = state.webgpu.renderer?.trainingMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
  const result = state.webgpu.renderer?.resultRenderMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
  state.gpuMemory.trainingActiveBytes = Math.max(0, Math.round(training.activeBytes || 0));
  state.gpuMemory.trainingReservedBytes = Math.max(
    state.gpuMemory.trainingActiveBytes,
    Math.round(training.reservedBytes || 0),
  );
  state.gpuMemory.resultActiveBytes = Math.max(0, Math.round(result.activeBytes || 0));
  state.gpuMemory.resultReservedBytes = Math.max(
    state.gpuMemory.resultActiveBytes,
    Math.round(result.reservedBytes || 0),
  );
  state.gpuMemory.activeBytes = Math.max(
    0,
    state.gpuMemory.trainingActiveBytes + state.gpuMemory.resultActiveBytes,
  );
  state.gpuMemory.reservedBytes = Math.max(
    state.gpuMemory.activeBytes,
    state.gpuMemory.trainingReservedBytes + state.gpuMemory.resultReservedBytes,
  );
  const activeMB = bytesToMB(state.gpuMemory.activeBytes);
  const reservedMB = bytesToMB(state.gpuMemory.reservedBytes);
  const digits = Math.max(activeMB, reservedMB) < 10 ? 1 : 0;
  els.gpuMemoryText.textContent = `${activeMB.toFixed(digits)} / ${reservedMB.toFixed(digits)} MB`;
}

function updateCapacityStatus() {
  const probe = state.capacityProbe;
  if (probe.status === "probing") {
    els.capacityProbeText.textContent = "Testing GPU capacity...";
  } else if (probe.fastPath && probe.selected > 0) {
    els.capacityProbeText.textContent = "Auto · fast path";
  } else if (probe.status === "passed" && probe.selected > 0) {
    els.capacityProbeText.textContent = `Auto · ${compactNumber(probe.selected)} verified`;
  } else if (probe.status === "fallback" && probe.selected > 0) {
    els.capacityProbeText.textContent = `Auto · reduced to ${compactNumber(probe.selected)}`;
  } else if (probe.status === "failed") {
    els.capacityProbeText.textContent = "Auto · failed";
  } else {
    els.capacityProbeText.textContent = els.capacityMode.value === "auto-probe" ? "Auto · ready" : "Use Max splats";
  }

  const trainState = state.webgpu.renderer?.trainState;
  if (trainState?.tileIndexCapacity) {
    const used = Number(state.metrics?.tile_counters?.total ?? trainState.tileIndexInitialReferences ?? 0);
    const ratio = used / Math.max(1, trainState.tileIndexCapacity);
    els.tileReserveText.textContent = `${compactNumber(used)} / ${compactNumber(trainState.tileIndexCapacity)} (${Math.round(ratio * 100)}%)`;
  } else {
    els.tileReserveText.textContent = "-";
  }

  const samples = state.metrics?.performance_trace || [];
  const measured = samples.find((sample) => Number.isFinite(sample.iterations_per_second) && sample.iterations_per_second > 0);
  if (measured) {
    const remaining = Math.max(0, (state.metrics.steps_requested || 0) - measured.step);
    const seconds = remaining / measured.iterations_per_second;
    els.measuredSpeedText.textContent = `${measured.iterations_per_second.toFixed(1)} it/s / ${formatDuration(seconds)}`;
  } else {
    els.measuredSpeedText.textContent = "-";
  }
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "-";
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function roundDownStep(value, step) {
  return Math.max(step, Math.floor(value / step) * step);
}

function pixelBytes() {
  // RGB target, source/render alpha, metrics/preview state, and three vec4 loss-gradient records.
  return 3 * 4 + 2 * 4 + 6 * 4 * 2 + 4 * 4 + 12 * 4;
}

function splatBytes(splats = 1) {
  // Ping-pong params, Adam/density state, exact backward gradient, and tile index allowance.
  return 256 + EXACT_GRADIENT_STRIDE * 4 + TILE_INDEX_FACTOR * 4;
}

function tileReferenceCountForParams(image, params) {
  if (!image || !params?.count) return 0;
  const tileCols = Math.ceil(image.width / TILE_SIZE);
  const tileRows = Math.ceil(image.height / TILE_SIZE);
  const useEwa = phase33Variants().ewa2x2;
  const pixelSigma = (MIP_PIXEL_SIGMA * 2) / Math.max(image.width, image.height);
  const pixelPaddingX = useEwa && image.width > 1 ? 0.5 / (image.width - 1) : 0;
  const pixelPaddingY = useEwa && image.height > 1 ? 0.5 / (image.height - 1) : 0;
  let total = 0;
  for (let i = 0; i < params.count; i += 1) {
    const sx = Math.max(0.0001, params.scale[i * 2]);
    const sy = Math.max(0.0001, params.scale[i * 2 + 1]);
    const effectiveX = useEwa ? sx : Math.hypot(sx, pixelSigma);
    const effectiveY = useEwa ? sy : Math.hypot(sy, pixelSigma);
    const theta = params.theta?.[i] || 0;
    const c = Math.abs(Math.cos(theta));
    const s = Math.abs(Math.sin(theta));
    const radiusX = RENDER_SIGMA * (c * effectiveX + s * effectiveY) + pixelPaddingX;
    const radiusY = RENDER_SIGMA * (s * effectiveX + c * effectiveY) + pixelPaddingY;
    const centerX = params.xy[i * 2];
    const centerY = params.xy[i * 2 + 1];
    const minNormX = Math.max(-1, centerX - radiusX);
    const minNormY = Math.max(-1, centerY - radiusY);
    const maxNormX = Math.min(1, centerX + radiusX);
    const maxNormY = Math.min(1, centerY + radiusY);
    const minPxX = Math.floor((minNormX * 0.5 + 0.5) * (image.width - 1));
    const minPxY = Math.floor((minNormY * 0.5 + 0.5) * (image.height - 1));
    const maxPxX = Math.ceil((maxNormX * 0.5 + 0.5) * (image.width - 1));
    const maxPxY = Math.ceil((maxNormY * 0.5 + 0.5) * (image.height - 1));
    const minTileX = Math.max(0, Math.min(tileCols - 1, Math.floor(minPxX / TILE_SIZE)));
    const minTileY = Math.max(0, Math.min(tileRows - 1, Math.floor(minPxY / TILE_SIZE)));
    const maxTileX = Math.max(0, Math.min(tileCols - 1, Math.floor(maxPxX / TILE_SIZE)));
    const maxTileY = Math.max(0, Math.min(tileRows - 1, Math.floor(maxPxY / TILE_SIZE)));
    total += (maxTileX - minTileX + 1) * (maxTileY - minTileY + 1);
  }
  return total;
}

function tileIndexCapacityLimit(device) {
  const limits = device?.limits || state.webgpu.limits || {};
  const maxBuffer = limitNumber(limits, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(limits, "maxStorageBufferBindingSize", 128 * MB);
  const budgetShare = Math.max(4, Math.floor(memoryBudgetBytes() * 0.25));
  return Math.max(1, Math.min(TILE_OFFSET_VALUE_MASK, Math.floor(Math.min(maxBuffer, maxStorage, budgetShare) / 4)));
}

function plannedTileIndexCapacity(image, params, bufferCapacity, device) {
  const tileCount = Math.ceil(image.width / TILE_SIZE) * Math.ceil(image.height / TILE_SIZE);
  const observed = tileReferenceCountForParams(image, params);
  const observedPerSplat = observed / Math.max(1, params.count);
  const projectedPerSplat = Math.max(TILE_INDEX_FACTOR, observedPerSplat * TILE_INDEX_INITIAL_HEADROOM);
  const requested = Math.ceil(Math.min(bufferCapacity * tileCount, bufferCapacity * projectedPerSplat));
  const normalCapacity = Math.max(1, Math.min(requested, tileIndexCapacityLimit(device)));
  const qaCapacity = qaTileIndexCapacityOverride();
  return {
    capacity: qaCapacity === null ? normalCapacity : Math.min(normalCapacity, qaCapacity),
    observed,
    observedPerSplat,
    requested,
    qaForcedCapacity: qaCapacity,
  };
}

function trainingBufferDescriptors(
  image,
  params,
  capacity,
  device = state.webgpu.renderer?.device,
  prepared = {},
) {
  const tilePlan = prepared.tilePlan || plannedTileIndexCapacity(image, params, capacity, device);
  const tileCount = Math.ceil(image.width / TILE_SIZE) * Math.ceil(image.height / TILE_SIZE);
  const variants = phase33Variants();
  const preparedStages = Object.hasOwn(prepared, "coarseImage");
  const stageDimensions = preparedStages
    ? null
    : curriculumStageDimensions(image.width, image.height, variants);
  const coarseImage = preparedStages ? prepared.coarseImage || null : null;
  const midImage = preparedStages ? prepared.midImage || null : null;
  const stageRgbBytes = (stageImage, dimensions) => stageImage?.rgb?.byteLength ||
    (dimensions ? dimensions.width * dimensions.height * 3 * 4 : 0);
  const stageAlphaBytes = (stageImage, dimensions) => stageImage?.alpha?.byteLength ||
    (dimensions ? dimensions.width * dimensions.height * 4 : 0);
  const coarseRgbBytes = stageRgbBytes(coarseImage, stageDimensions?.coarse);
  const midRgbBytes = stageRgbBytes(midImage, stageDimensions?.mid);
  const coarseAlphaBytes = stageAlphaBytes(coarseImage, stageDimensions?.coarse);
  const midAlphaBytes = stageAlphaBytes(midImage, stageDimensions?.mid);
  const optimizerStride = 96;
  const profileEnabled = Boolean(state.webgpu.renderer?.performanceProfile?.timestampQuery);
  const descriptors = [];
  const add = (name, size, storage = false) => {
    if (size > 0) descriptors.push({ name, size: Math.max(4, Math.ceil(size / 4) * 4), storage });
  };
  add("config", TRAIN_CONFIG_BYTES, true);
  add("batch-config", TRAIN_BATCH_CONFIG_BYTES);
  add("present-config", 16);
  add("target-rgb", image.rgb.byteLength, true);
  add("coarse-target-rgb", coarseRgbBytes, true);
  add("mid-target-rgb", midRgbBytes, true);
  add("target-alpha", image.alpha?.byteLength || image.width * image.height * 4, true);
  add("coarse-target-alpha", coarseAlphaBytes, true);
  add("mid-target-alpha", midAlphaBytes, true);
  add("error-map", image.width * image.height * 4, true);
  add("stats", capacity * 2 * 4 * 4, true);
  add("density-control", (
    capacity * 4 +
    DENSITY_EVENT_SLOTS + 1 +
    Math.ceil(capacity / 256) * 2 +
    PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE +
    residualTileControlWords(image)
  ) * 4, true);
  add("tile-counts", tileCount * 4, true);
  add("tile-offsets", (tileCount + 1) * 4, true);
  add("tile-cursors", tileCount * 4, true);
  add("tile-indices", tilePlan.capacity * 4, true);
  add("tile-control", 16, true);
  add("pixel-state", image.width * image.height * 16, true);
  add("alpha-state", image.width * image.height * ALPHA_STATE_BYTES_PER_PIXEL, true);
  add("loss-gradient", image.width * image.height * 48, true);
  add("exact-gradient", capacity * EXACT_GRADIENT_STRIDE * 4, true);
  add("ssim-tiles", ssimWorkingBufferBytes(image), true);
  add("optimizer-state", capacity * optimizerStride, true);
  add("xy-depth", capacity * 4 * 4, true);
  add("transform", capacity * 4 * 4, true);
  add("color", capacity * 4 * 4, true);
  add("readback", capacity * 12 * 4);
  add("growth-signal-readback", 4);
  if (profileEnabled) {
    add("exact-backward-telemetry", EXACT_BACKWARD_TELEMETRY_BYTES, true);
    add("exact-backward-telemetry-readback", EXACT_BACKWARD_TELEMETRY_BYTES);
    add("profile-resolve", PERFORMANCE_PROFILE_QUERY_CAPACITY * 8);
    add("profile-readback", PERFORMANCE_PROFILE_QUERY_CAPACITY * 8);
  }
  const segmentedPartialBytes = tilePlan.capacity * EXACT_GRADIENT_STRIDE * 4;
  const segmentedAuxiliaryBytes =
    capacity * 4 +
    (capacity + 1) * 4 +
    capacity * 4 +
    tilePlan.capacity * 4;
  const baseReservedBytes = descriptors.reduce((sum, item) => sum + item.size, 0);
  const segmentedRequested = performanceVariants().segmentedExactBackward;
  const segmentedSupported =
    segmentedRequested &&
    Number(device?.limits?.maxStorageBuffersPerShaderStage || 8) >= 9 &&
    segmentedPartialBytes <= SEGMENTED_EXACT_BACKWARD_MAX_BYTES &&
    baseReservedBytes + segmentedPartialBytes + segmentedAuxiliaryBytes <= memoryBudgetBytes() * 0.9;
  const fixedPointRequested = performanceVariants().fixedPointExactGradient;
  const fixedPointSupported = fixedPointRequested && !segmentedSupported;
  if (segmentedSupported) {
    add("segmented-partial-gradient", segmentedPartialBytes, true);
    add("segmented-reference-counts", capacity * 4, true);
    add("segmented-reference-offsets", (capacity + 1) * 4, true);
    add("segmented-reference-cursors", capacity * 4, true);
    add("segmented-references", tilePlan.capacity * 4, true);
  }
  if (fixedPointSupported) {
    add("fixed-point-gradient-control", FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES, true);
    add("fixed-point-gradient-readback", FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES);
  }
  return {
    descriptors,
    tilePlan,
    coarseImage,
    midImage,
    segmentedExactBackward: {
      requested: segmentedRequested,
      enabled: segmentedSupported,
      partialBytes: segmentedSupported ? segmentedPartialBytes : 0,
      auxiliaryBytes: segmentedSupported ? segmentedAuxiliaryBytes : 0,
      reason: !segmentedRequested
        ? "not-requested"
        : Number(device?.limits?.maxStorageBuffersPerShaderStage || 8) < 9
          ? "requires-9-storage-buffers"
          : segmentedPartialBytes > SEGMENTED_EXACT_BACKWARD_MAX_BYTES
            ? "partial-buffer-cap"
            : baseReservedBytes + segmentedPartialBytes + segmentedAuxiliaryBytes > memoryBudgetBytes() * 0.9
              ? "training-memory-budget"
              : "enabled",
    },
    fixedPointExactGradient: {
      requested: fixedPointRequested,
      enabled: fixedPointSupported,
      scale: FIXED_POINT_EXACT_GRADIENT_SCALE,
      reason: !fixedPointRequested
        ? "not-requested"
        : segmentedSupported
          ? "mutually-exclusive-with-segmented-backward"
          : "enabled",
    },
  };
}

function trainingAllocationPlan(
  image,
  params,
  capacity,
  device = state.webgpu.renderer?.device,
  prepared = {},
) {
  const descriptorPlan = trainingBufferDescriptors(image, params, capacity, device, prepared);
  const storageSizes = descriptorPlan.descriptors.filter((item) => item.storage).map((item) => item.size);
  const allSizes = descriptorPlan.descriptors.map((item) => item.size);
  const maxBuffer = limitNumber(device?.limits || state.webgpu.limits || {}, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(device?.limits || state.webgpu.limits || {}, "maxStorageBufferBindingSize", 128 * MB);
  const largestStorageBytes = Math.max(0, ...storageSizes);
  const largestBufferBytes = Math.max(0, ...allSizes);
  const reservedBytes = allSizes.reduce((sum, size) => sum + size, 0);
  return {
    capacity,
    tilePlan: descriptorPlan.tilePlan,
    descriptors: descriptorPlan.descriptors,
    reservedBytes,
    largestStorageBytes,
    largestBufferBytes,
    maxStorage,
    maxBuffer,
    withinBufferLimits: largestStorageBytes <= maxStorage && largestBufferBytes <= maxBuffer,
    withinBudget: reservedBytes <= memoryBudgetBytes() * 0.9,
    segmentedExactBackward: descriptorPlan.segmentedExactBackward,
    fixedPointExactGradient: descriptorPlan.fixedPointExactGradient,
  };
}

function trainStateAllocatedDescriptors(trainState) {
  if (!trainState) return [];
  const entries = [
    ["config", trainState.configBuffer],
    ["batch-config", trainState.batchConfigBuffer],
    ["present-config", trainState.presentConfigBuffer],
    ["target-rgb", trainState.targetBuffer],
    ["coarse-target-rgb", trainState.coarseTargetBuffer],
    ["mid-target-rgb", trainState.midTargetBuffer],
    ["target-alpha", trainState.targetAlphaBuffer],
    ["coarse-target-alpha", trainState.coarseTargetAlphaBuffer],
    ["mid-target-alpha", trainState.midTargetAlphaBuffer],
    ["error-map", trainState.errorMapBuffer],
    ["stats", trainState.statsBuffer],
    ["density-control", trainState.densityControlBuffer],
    ["tile-counts", trainState.tileCountsBuffer],
    ["tile-offsets", trainState.tileOffsetsBuffer],
    ["tile-cursors", trainState.tileCursorsBuffer],
    ["tile-indices", trainState.tileIndicesBuffer],
    ["tile-control", trainState.tileControlBuffer],
    ["pixel-state", trainState.pixelStateBuffer],
    ["alpha-state", trainState.alphaStateBuffer],
    ["loss-gradient", trainState.lossGradientBuffer],
    ["exact-gradient", trainState.exactGradientBuffer],
    ["fixed-point-gradient-control", trainState.fixedPointGradientControlBuffer],
    ["fixed-point-gradient-readback", trainState.fixedPointGradientReadbackBuffer],
    ["segmented-partial-gradient", trainState.segmentedPartialGradientBuffer],
    ["segmented-reference-counts", trainState.segmentedReferenceCountsBuffer],
    ["segmented-reference-offsets", trainState.segmentedReferenceOffsetsBuffer],
    ["segmented-reference-cursors", trainState.segmentedReferenceCursorsBuffer],
    ["segmented-references", trainState.segmentedReferencesBuffer],
    ["ssim-tiles", trainState.ssimTileBuffer],
    ["optimizer-state", trainState.optimizerStateBuffer],
    ["xy-depth", trainState.xyBuffers?.[0]],
    ["transform", trainState.transformBuffers?.[0]],
    ["color", trainState.colorBuffers?.[0]],
    ["readback", trainState.readbackBuffer],
    ["growth-signal-readback", trainState.growthSignalReadbackBuffer],
    ["exact-backward-telemetry", trainState.exactBackwardTelemetryBuffer],
    ["exact-backward-telemetry-readback", trainState.exactBackwardTelemetryReadbackBuffer],
    ["profile-resolve", trainState.profileResolveBuffer],
    ["profile-readback", trainState.profileReadbackBuffer],
  ];
  return entries
    .filter(([, buffer]) => Boolean(buffer))
    .map(([name, buffer]) => ({ name, size: Number(buffer.size || 0) }));
}

function allocationDescriptorMismatch(plannedDescriptors, allocatedDescriptors) {
  const planned = new Map(plannedDescriptors.map(({ name, size }) => [name, Number(size)]));
  const allocated = new Map(allocatedDescriptors.map(({ name, size }) => [name, Number(size)]));
  const names = [...new Set([...planned.keys(), ...allocated.keys()])].sort();
  return names
    .filter((name) => planned.get(name) !== allocated.get(name))
    .map((name) => ({ name, planned: planned.get(name) ?? null, allocated: allocated.get(name) ?? null }));
}

function tileGrowthMemoryPlan({
  currentReservedBytes,
  currentTileBytes,
  nextTileBytes,
  budgetBytes = memoryBudgetBytes() * 0.9,
}) {
  const current = Math.max(0, Number(currentReservedBytes) || 0);
  const previous = Math.max(0, Number(currentTileBytes) || 0);
  const next = Math.max(0, Number(nextTileBytes) || 0);
  const budget = Math.max(0, Number(budgetBytes) || 0);
  const finalReservedBytes = Math.max(0, current - previous) + next;
  const transientReservedBytes = current + next;
  return {
    currentReservedBytes: current,
    currentTileBytes: previous,
    nextTileBytes: next,
    finalReservedBytes,
    transientReservedBytes,
    budgetBytes: budget,
    withinBudget: finalReservedBytes <= budget && transientReservedBytes <= budget,
  };
}

function capacityProbeCandidates(requested) {
  const capped = normalizeUiSplatCount(requested, DEFAULT_FINAL_SPLATS, MANUAL_SPLATS_MAX);
  const candidates = CAPACITY_PROBE_TIERS.filter((value) => value <= capped);
  if (!candidates.includes(capped)) candidates.push(capped);
  return [...new Set(candidates)].sort((a, b) => b - a);
}

function estimatedImageSizeFor(trainSize) {
  if (!state.image) return { width: trainSize, height: trainSize };
  const sourceWidth = state.image.cacheWidth || state.image.width;
  const sourceHeight = state.image.cacheHeight || state.image.height;
  const [width, height] = resizedSize(sourceWidth, sourceHeight, trainSize);
  return { width, height };
}

function imagePixelEstimate(trainSize) {
  const size = estimatedImageSizeFor(trainSize);
  return size.width * size.height;
}

function sideFromPixelBudget(pixelBudget, trainSize) {
  if (!state.image) return Math.sqrt(pixelBudget);
  const sourceWidth = state.image.cacheWidth || state.image.width;
  const sourceHeight = state.image.cacheHeight || state.image.height;
  const side = Math.max(1, sourceWidth, sourceHeight);
  const aspectPixels = Math.max(1, sourceWidth * sourceHeight) / (side * side);
  return Math.sqrt(pixelBudget / Math.max(0.01, aspectPixels));
}

function estimateGpuMemory(trainSize, splats) {
  const fullSize = estimatedImageSizeFor(trainSize);
  const pixels = fullSize.width * fullSize.height;
  const variants = phase33Variants();
  const curriculum = curriculumStageDimensions(fullSize.width, fullSize.height, variants);
  const coarseTargetBytes = curriculum.coarse ? curriculum.coarse.width * curriculum.coarse.height * 4 * 4 : 0;
  const midTargetBytes = curriculum.mid ? curriculum.mid.width * curriculum.mid.height * 4 * 4 : 0;
  // Ping-pong params, Adam moments, active/density state, and compact tile indices.
  const trainStateBytes = splats * splatBytes(splats);
  const targetBytes = pixels * 4 * 4;
  const metricsBytes = pixels * 6 * 4;
  const previewBytes = pixels * 4 * 4;
  const alphaStateBytes = pixels * ALPHA_STATE_BYTES_PER_PIXEL;
  const lossGradientBytes = pixels * 48;
  const exactGradientBytes = splats * EXACT_GRADIENT_STRIDE * 4;
  const tileCount = Math.ceil(Math.sqrt(pixels) / TILE_SIZE) ** 2;
  const tileScratchBytes = tileCount * 3 * 4 + (tileCount + 1) * 4 + 16;
  const overlapDiagnosticBytes = phase40Variants().overlapDiagnostics ? pixels * 8 : 0;
  const peakBytes = trainStateBytes + targetBytes + coarseTargetBytes + midTargetBytes + alphaStateBytes + lossGradientBytes + metricsBytes * 2 + previewBytes * 2 + overlapDiagnosticBytes + tileScratchBytes + splats * 32;
  return { pixels, trainStateBytes, targetBytes, coarseTargetBytes, midTargetBytes, alphaStateBytes, lossGradientBytes, exactGradientBytes, metricsBytes, previewBytes, overlapDiagnosticBytes, peakBytes };
}

function browserMemoryHintBytes() {
  const deviceMemoryGB = Number(navigator.deviceMemory);
  if (Number.isFinite(deviceMemoryGB) && deviceMemoryGB > 0) {
    return { bytes: deviceMemoryGB * GB, source: `deviceMemory ${deviceMemoryGB}GB`, exact: false };
  }
  const heapLimit = Number(performance?.memory?.jsHeapSizeLimit);
  if (Number.isFinite(heapLimit) && heapLimit > 0) {
    return { bytes: heapLimit, source: "jsHeapSizeLimit", exact: false };
  }
  return { bytes: 0, source: "WebGPU limits only", exact: false };
}

function autoBudgetInfo(limits) {
  const maxBuffer = limitNumber(limits, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(limits, "maxStorageBufferBindingSize", 128 * MB);
  const browserMemory = browserMemoryHintBytes();
  const webgpuEnvelope = Math.min(2 * GB, maxBuffer * 8, maxStorage * 16);
  const hintedWorkingSet = browserMemory.bytes > 0 ? browserMemory.bytes * 0.125 : 256 * MB;
  const budgetBytes = clampNumber(Math.min(hintedWorkingSet, webgpuEnvelope), 128 * MB, 2 * GB);
  const reservedBytes = browserMemory.bytes > 0 ? Math.max(0, browserMemory.bytes - budgetBytes) : 0;
  return {
    budgetBytes,
    reservedBytes,
    exact: false,
    source: browserMemory.source,
    webgpuEnvelope,
  };
}

function memoryBudgetInfo(limits) {
  const automatic = autoBudgetInfo(limits);
  if (!state.safety.memoryLimiterUnlocked) {
    return { ...automatic, mode: "automatic", multiplier: 1 };
  }
  const budgetBytes = Math.min(
    MEMORY_LIMITER_UNLOCK_MAX_BYTES,
    automatic.budgetBytes * MEMORY_LIMITER_UNLOCK_MULTIPLIER,
  );
  const memoryHint = browserMemoryHintBytes();
  return {
    ...automatic,
    budgetBytes,
    reservedBytes: memoryHint.bytes > 0 ? Math.max(0, memoryHint.bytes - budgetBytes) : 0,
    webgpuEnvelope: Math.min(
      MEMORY_LIMITER_UNLOCK_MAX_BYTES,
      automatic.webgpuEnvelope * MEMORY_LIMITER_UNLOCK_MULTIPLIER,
    ),
    mode: "unlocked",
    multiplier: MEMORY_LIMITER_UNLOCK_MULTIPLIER,
  };
}

function memoryBudgetBytes() {
  return memoryBudgetInfo(state.webgpu.limits).budgetBytes;
}

function computeBudgetFor(trainSize, finalSplats, steps) {
  const limits = state.webgpu.limits || {};
  trainSize = Math.round(clampNumber(trainSize, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE));
  finalSplats = normalizeUiSplatCount(finalSplats, DEFAULT_FINAL_SPLATS);
  steps = normalizeStepInteger(steps, { min: LIMITS.stepsMin, max: LIMITS.stepsMax, fallback: DEFAULT_ITERATIONS });
  const budgetInfo = memoryBudgetInfo(limits);
  const budgetBytes = budgetInfo.budgetBytes;
  const memoryHint = browserMemoryHintBytes();
  const autoBudget = autoBudgetInfo(limits);
  const current = estimateGpuMemory(trainSize, finalSplats);
  const maxBuffer = limitNumber(limits, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(limits, "maxStorageBufferBindingSize", 128 * MB);
  const maxTexture = limitNumber(limits, "maxTextureDimension2D", LIMITS.trainSizeMax);
  const hardBufferBytes = Math.min(maxBuffer, maxStorage);
  const largestPixelStorageBytes = Math.max(
    current.targetBytes,
    current.alphaStateBytes,
    current.lossGradientBytes,
    current.metricsBytes,
    current.previewBytes,
    current.overlapDiagnosticBytes,
  );
  const largestPixelBytesPerPixel = Math.max(1, Math.ceil(largestPixelStorageBytes / Math.max(1, current.pixels)));
  const hardPixelLimit = Math.floor(hardBufferBytes / largestPixelBytesPerPixel);
  const softPixelLimit = Math.floor((budgetBytes - finalSplats * splatBytes()) / pixelBytes());
  const pixelBudget = Math.max(1, Math.min(hardPixelLimit, softPixelLimit));
  const limiterTrainSize = Math.min(
    LIMITS.trainSizeMax,
    maxTexture,
    Math.max(LIMITS.trainSizeMin, roundDownStep(sideFromPixelBudget(Math.max(1, pixelBudget), trainSize), 16)),
  );
  const splatBudget = Math.floor((budgetBytes - current.pixels * pixelBytes()) / splatBytes());
  const limiterFinalSplats = Math.max(
    LIMITS.splatsMin,
    Math.min(LIMITS.splatsMax, roundDownStep(Math.max(LIMITS.splatsMin, splatBudget), 4)),
  );
  const recommendedTrainSize = Math.min(trainSize, limiterTrainSize);
  const recommendedFinalSplats = Math.min(finalSplats, limiterFinalSplats);
  const previewWork = current.pixels * finalSplats;
  const trainWork = finalSplats * steps;
  const rawMetricInterval = Math.ceil((steps / 60) * Math.max(1, previewWork / 70_000_000));
  const metricInterval = clampNumber(rawMetricInterval, 1, DEFAULT_MAX_METRIC_INTERVAL, 1);
  const headroomBytes = budgetBytes - current.peakBytes;
  return {
    budgetBytes,
    budgetMB: bytesToMB(budgetBytes).toFixed(0),
    headroomBytes,
    headroomMB: bytesToMB(headroomBytes).toFixed(1),
    memoryHintSource: memoryHint.source,
    memoryHintMB: memoryHint.bytes > 0 ? bytesToMB(memoryHint.bytes).toFixed(0) : "",
    memoryBudgetMode: budgetInfo.mode,
    memoryBudgetMultiplier: budgetInfo.multiplier,
    autoBudgetExact: false,
    autoBudgetSource: autoBudget.source,
    autoReservedMB: autoBudget.reservedBytes > 0 ? bytesToMB(autoBudget.reservedBytes).toFixed(0) : "",
    autoEnvelopeMB: bytesToMB(autoBudget.webgpuEnvelope).toFixed(0),
    estimatedBytes: current.peakBytes,
    estimatedMB: bytesToMB(current.peakBytes).toFixed(1),
    metricsBytes: current.metricsBytes,
    targetBytes: current.targetBytes,
    trainStateBytes: current.trainStateBytes,
    maxBuffer,
    maxStorage,
    maxTexture,
    limiterTrainSize,
    limiterFinalSplats,
    recommendedTrainSize,
    recommendedFinalSplats,
    metricInterval,
    previewWork,
    trainWork,
    overBudget: current.peakBytes > budgetBytes,
    overHardLimit: largestPixelStorageBytes > hardBufferBytes,
  };
}

function updateImageSizeStatus() {
  if (!state.image) {
    els.imageSizeText.textContent = "-";
    return;
  }
  const cacheWidth = state.image.cacheWidth || state.image.width;
  const cacheHeight = state.image.cacheHeight || state.image.height;
  els.imageSizeText.textContent = `${cacheWidth}x${cacheHeight}`;
}

function computeRecommendation() {
  const trainSize = Math.round(clampNumber(els.trainSize.value, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE));
  const finalSplats = normalizeUiSplatCount(els.finalSplatCount.value, DEFAULT_FINAL_SPLATS);
  const steps = normalizeStepInteger(els.stepCount.value, { min: LIMITS.stepsMin, max: LIMITS.stepsMax, fallback: DEFAULT_ITERATIONS });
  return computeBudgetFor(trainSize, finalSplats, steps);
}

function safetyFailure(rec, context) {
  if (rec.overHardLimit) return { context, reason: "safety_stop_hard_limit", rec };
  if (rec.overBudget) return { context, reason: "safety_stop_budget", rec };
  return null;
}

function setSafetyStop(failure) {
  const rec = failure.rec;
  state.safety.lastStopReason = failure.reason;
  state.safety.lastStopEstimateMB = rec.estimatedMB;
  state.safety.lastStopBudgetMB = rec.budgetMB;
  state.safety.lastRecommended = `${rec.recommendedTrainSize}px/${rec.recommendedFinalSplats}`;
  els.budgetNote.textContent = `Safety stop: ${failure.reason}. ${rec.estimatedMB}/${rec.budgetMB} MB, try ${state.safety.lastRecommended}.`;
  log(`safety stop ${failure.context}: ${failure.reason} estimate=${rec.estimatedMB}MB budget=${rec.budgetMB}MB recommended=${state.safety.lastRecommended}`);
  publishState();
}

function runtimeSafetyError(reason, context, details = {}) {
  const rec = state.recommendation || computeRecommendation();
  const failure = { reason, context, rec };
  if (state.metrics) {
    state.metrics.stopped = true;
    state.metrics.safety_stop = {
      reason,
      context,
      estimated_mb: rec.estimatedMB,
      budget_mb: rec.budgetMB,
      ...details,
    };
  }
  setSafetyStop(failure);
  const error = new Error(`${reason}: ${context}`);
  error.safetyStop = true;
  return error;
}

function clearSafetyStop() {
  state.safety.lastStopReason = "";
  state.safety.lastStopEstimateMB = "";
  state.safety.lastStopBudgetMB = "";
  state.safety.lastRecommended = "";
}

function setLimiterAttributes(rec) {
  els.trainSize.max = String(LIMITS.trainSizeMax);
  const splatLimit = state.image
    ? normalizeUiSplatCount(Math.min(MANUAL_SPLATS_MAX, rec.limiterFinalSplats), DEFAULT_FINAL_SPLATS)
    : MANUAL_SPLATS_MAX;
  els.initialSplatCount.max = String(Math.min(CAPACITY_PROBE_FAST_PATH_MAX, splatLimit));
  els.finalSplatCount.max = String(splatLimit);
}

function applyDeviceLimiter(rec, { reconcileSplatCounts = true } = {}) {
  setLimiterAttributes(rec);
  if (state.running) return false;
  if (!state.image) return false;

  const effectiveLimit = normalizeUiSplatCount(
    Math.min(MANUAL_SPLATS_MAX, rec.limiterFinalSplats),
    DEFAULT_FINAL_SPLATS,
  );
  if (Number(els.finalSplatCount.value) > effectiveLimit) {
    els.finalSplatCount.value = String(effectiveLimit);
    return true;
  }
  if (reconcileSplatCounts && Number(els.initialSplatCount.value) > Number(els.finalSplatCount.value)) {
    els.initialSplatCount.value = String(els.finalSplatCount.value);
    return true;
  }
  return false;
}

function syncMemoryLimiterUnlockUi() {
  const unlocked = state.safety.memoryLimiterUnlocked;
  els.memoryLimiterUnlock.setAttribute("aria-pressed", String(unlocked));
  els.memoryLimiterUnlock.textContent = `Memory limiter unlock: ${unlocked ? "ON" : "OFF"}`;
}

function updateMemoryRecommendation({ reconcileSplatCounts = true } = {}) {
  syncMemoryLimiterUnlockUi();
  let rec = computeRecommendation();
  for (let i = 0; i < 3 && applyDeviceLimiter(rec, { reconcileSplatCounts }); i += 1) {
    rec = computeRecommendation();
  }
  setLimiterAttributes(rec);
  state.recommendation = rec;
  els.gpuLimitText.textContent = `buf ${formatMB(rec.maxBuffer)} / stor ${formatMB(rec.maxStorage)} / tex ${rec.maxTexture}`;
  els.memoryEstimateText.textContent = `${rec.estimatedMB}/${rec.budgetMB} MB`;
  els.headroomText.textContent = `${rec.headroomMB} MB`;
  els.limiterText.textContent = `${rec.limiterTrainSize}px / ${rec.limiterFinalSplats}`;
  els.recommendationText.textContent = `${rec.recommendedTrainSize}px / ${rec.recommendedFinalSplats}`;
  els.speedEstimateText.textContent = `${compactNumber(rec.previewWork)} / metrics ${rec.metricInterval}`;
  const detailCap = recommendedDetailShape(state.image || { width: Number(els.trainSize.value) || 1024, height: Number(els.trainSize.value) || 1024 });
  els.detailCapText.textContent = `${detailCap.minorPx.toFixed(1)} x ${detailCap.majorPx.toFixed(1)}px / ${detailCap.maxAnisotropy.toFixed(1)}:1`;
  els.budgetNote.textContent = rec.overHardLimit
    ? "Current settings exceed a WebGPU buffer limit."
    : rec.overBudget
      ? "Current settings exceed the safety budget; apply recommended or lower values."
      : state.safety.memoryLimiterUnlocked
        ? `Unlocked mode uses a ${rec.memoryBudgetMultiplier}x app working-set budget (${rec.budgetMB} MB). WebGPU hard limits still apply; browser or device termination is more likely.`
        : `Auto uses ${rec.autoBudgetSource} and WebGPU limits; ${rec.autoReservedMB ? `${rec.autoReservedMB} MB remains outside the app estimate` : "a conservative fallback is active"}. Exact free VRAM is unavailable.`;
  publishState();
  return rec;
}

function recommendedDetailShape(image) {
  const longSide = Math.max(1, Number(image?.width) || 1, Number(image?.height) || 1);
  const minorPx = 1;
  const majorPx = Math.max(8, Math.min(64, longSide * 0.02));
  return { minorPx, majorPx, maxAnisotropy: Math.max(6, Math.min(32, majorPx / minorPx)) };
}

function compactNumber(value) {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(Math.round(value));
}

function releaseImageSource(image) {
  const source = image?.sourceBitmap;
  if (!source) return;
  source.close?.();
  releaseCanvasBackingStore(source);
  image.sourceBitmap = null;
}

function releaseCanvasBackingStore(source) {
  // Canvas has no close(). Shrinking it releases the backing store while the
  // Float32 source cache remains available only on the replacement image.
  if (typeof HTMLCanvasElement !== "undefined" && source instanceof HTMLCanvasElement) {
    source.width = 1;
    source.height = 1;
  }
}


function showCanvas(kind) {
  els.previewCanvas.hidden = kind !== "preview";
  els.gpuCanvas.hidden = kind !== "gpu";
  els.tiltCanvas.hidden = kind !== "tilt";
  els.tiltTeacherCanvas.hidden = kind !== "tilt" || state.tilt.viewMode === "splats";
  els.tiltTeacherCanvas.style.opacity = state.tilt.viewMode === "overlay" ? "0.5" : "1";
  els.tiltCanvas.style.opacity = kind === "tilt" && state.tilt.viewMode === "original" ? "0" : "1";
  els.tiltFrameOverlay.hidden = kind !== "tilt" || !state.tilt.controller;
  els.viewControls.hidden = kind === "tilt";
  // The padded canvas already communicates the outside-image extent. Keep the
  // legacy frame node for QA selectors, but never draw a competing white box.
  els.previewImageFrame.hidden = true;
  if (kind !== "tilt") applyCanvasView();
}

function activePreviewCanvas() {
  return state.previewMode === "splats" && !els.gpuCanvas.hidden ? els.gpuCanvas : els.previewCanvas;
}

function fittedCanvasScale(width = activePreviewCanvas().width, height = activePreviewCanvas().height) {
  const rect = els.viewer.getBoundingClientRect();
  const maxWidth = Math.max(160, rect.width - 44);
  const maxHeight = Math.max(160, rect.height - 44);
  return Math.min(maxWidth / Math.max(1, width), maxHeight / Math.max(1, height));
}

function applyCanvasView() {
  const activeCanvas = activePreviewCanvas();
  const width = activeCanvas.width;
  const height = activeCanvas.height;
  if (!width || !height) return;
  const scale = state.canvasView.mode === "fit" ? fittedCanvasScale(width, height) : state.canvasView.scale;
  state.canvasView.scale = Math.max(0.02, Math.min(32, scale));
  const transform = `translate(-50%, -50%) translate(${state.canvasView.panX}px, ${state.canvasView.panY}px)`;
  for (const canvas of [els.previewCanvas, els.gpuCanvas]) {
    canvas.style.width = `${Math.max(1, Math.round(canvas.width * state.canvasView.scale))}px`;
    canvas.style.height = `${Math.max(1, Math.round(canvas.height * state.canvasView.scale))}px`;
    canvas.style.transform = transform;
  }
  if (!els.previewImageFrame.hidden && state.image) {
    els.previewImageFrame.style.width = `${Math.max(1, Math.round(state.image.width * state.canvasView.scale))}px`;
    els.previewImageFrame.style.height = `${Math.max(1, Math.round(state.image.height * state.canvasView.scale))}px`;
    els.previewImageFrame.style.transform = transform;
  }
  document.documentElement.dataset.canvasViewMode = state.canvasView.mode;
  document.documentElement.dataset.canvasViewScale = String(state.canvasView.scale);
  document.documentElement.dataset.canvasPanX = String(Math.round(state.canvasView.panX));
  document.documentElement.dataset.canvasPanY = String(Math.round(state.canvasView.panY));
}

function setCanvasView(mode) {
  if (canvasViewInputLocked()) return false;
  state.canvasView.mode = mode;
  state.canvasView.scale = mode === "actual" ? 1 : fittedCanvasScale();
  state.canvasView.panX = 0;
  state.canvasView.panY = 0;
  applyCanvasView();
  publishState();
  return true;
}

function fitCanvases(width = activePreviewCanvas().width, height = activePreviewCanvas().height) {
  state.canvasView.mode = "fit";
  state.canvasView.scale = fittedCanvasScale(width, height);
  state.canvasView.panX = 0;
  state.canvasView.panY = 0;
  applyCanvasView();
}

function canvasViewInputLocked() {
  return trainingLifecycleInputLocked();
}

function cancelCanvasViewGesture() {
  const pointerIds = [...state.canvasPointers.keys()];
  state.canvasPointers.clear();
  state.canvasPinch = null;
  state.canvasView.pointerId = null;
  els.viewer.classList.remove("is-panning");
  for (const pointerId of pointerIds) {
    try {
      if (els.viewer.hasPointerCapture(pointerId)) els.viewer.releasePointerCapture(pointerId);
    } catch {
      // Pointer capture may already have ended between lifecycle updates.
    }
  }
}

function updateCanvasViewControls() {
  const locked = canvasViewInputLocked();
  if (locked && (state.canvasPointers.size > 0 || state.canvasPinch || state.canvasView.pointerId !== null)) {
    cancelCanvasViewGesture();
  }
  const disabled = !state.image || locked;
  els.actualSizeButton.disabled = disabled;
  els.fitViewButton.disabled = disabled;
  els.viewer.classList.toggle("canvas-view-locked", locked);
  els.viewer.dataset.canvasViewLocked = String(locked);
  els.viewer.title = locked ? "Canvas zoom and pan are locked while training." : "";
  document.documentElement.dataset.canvasViewLocked = String(locked);
}

function zoomCanvasAt(clientX, clientY, deltaY) {
  if (!state.image || canvasViewInputLocked()) return false;
  const rect = els.viewer.getBoundingClientRect();
  const oldScale = state.canvasView.scale;
  const nextScale = Math.max(0.02, Math.min(32, oldScale * Math.exp(-deltaY * 0.0015)));
  const centerX = rect.left + rect.width * 0.5 + state.canvasView.panX;
  const centerY = rect.top + rect.height * 0.5 + state.canvasView.panY;
  const imageX = (clientX - centerX) / oldScale;
  const imageY = (clientY - centerY) / oldScale;
  state.canvasView.mode = "custom";
  state.canvasView.scale = nextScale;
  state.canvasView.panX = clientX - (rect.left + rect.width * 0.5) - imageX * nextScale;
  state.canvasView.panY = clientY - (rect.top + rect.height * 0.5) - imageY * nextScale;
  applyCanvasView();
  return true;
}

function drawRgbToCanvas(rgb, width, height) {
  previewCtx.putImageData(rgbToImageData(rgb, width, height), 0, 0);
}

function drawOriginalToCanvas() {
  if (!state.image) return false;
  const { width, height, sourceBitmap } = state.image;
  if (els.previewCanvas.width !== width) els.previewCanvas.width = width;
  if (els.previewCanvas.height !== height) els.previewCanvas.height = height;
  previewCtx.clearRect(0, 0, width, height);
  if (sourceBitmap) {
    // Reuse the bounded source cache instead of converting the full Float32
    // image back to ImageData on every Original/Splats tab switch.
    previewCtx.imageSmoothingEnabled = true;
    previewCtx.imageSmoothingQuality = "high";
    previewCtx.drawImage(sourceBitmap, 0, 0, width, height);
  } else {
    drawRgbToCanvas(state.image.rgb, width, height);
  }
  return true;
}

function updatePreviewModeControls() {
  const hasImage = Boolean(state.image);
  const hasSplats = Boolean(state.params);
  const locked = previewModeInputLocked();
  els.originalPreviewButton.disabled = !hasImage || locked;
  els.splatsPreviewButton.disabled = !hasSplats || locked;
  for (const [button, mode] of [
    [els.splatsPreviewButton, "splats"],
    [els.originalPreviewButton, "original"],
  ]) {
    const active = state.previewMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  document.documentElement.dataset.previewMode = state.previewMode;
}

function setPreviewMode(mode) {
  if (previewModeInputLocked()) return false;
  if (mode === "original") {
    if (!state.image) return false;
    state.previewMode = "original";
    drawOriginalToCanvas();
    showCanvas("preview");
  } else if (mode === "splats") {
    if (!state.params) return false;
    state.previewMode = "splats";
    if (state.running) {
      state.previewPadding = previewPaddingSpec(state.image, state.params, false);
      state.webgpu.renderer?.presentTrainState(state.image);
    } else {
      refreshOutsidePreview().catch((error) => log(`splat preview failed: ${error.message}`));
    }
    showCanvas("gpu");
  } else {
    return false;
  }
  updatePreviewModeControls();
  publishState();
  return true;
}

function rgbToImageData(rgb, width, height, alpha = null) {
  const imageData = new ImageData(width, height);
  for (let i = 0, j = 0, p = 0; j < rgb.length; i += 4, j += 3, p += 1) {
    imageData.data[i] = clampByte(rgb[j] * 255);
    imageData.data[i + 1] = clampByte(rgb[j + 1] * 255);
    imageData.data[i + 2] = clampByte(rgb[j + 2] * 255);
    imageData.data[i + 3] = clampByte((alpha?.[p] ?? 1) * 255);
  }
  return imageData;
}

function resizeFloatImageBilinear(image, width, height) {
  const sourceWidth = Math.max(1, Math.round(image.width));
  const sourceHeight = Math.max(1, Math.round(image.height));
  const targetWidth = Math.max(1, Math.round(width));
  const targetHeight = Math.max(1, Math.round(height));
  if (sourceWidth === targetWidth && sourceHeight === targetHeight) {
    return {
      width: targetWidth,
      height: targetHeight,
      rgb: image.rgb,
      alpha: image.alpha || null,
    };
  }
  const rgb = new Float32Array(targetWidth * targetHeight * 3);
  const alpha = new Float32Array(targetWidth * targetHeight);
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = Math.max(0, Math.min(sourceHeight - 1, (y + 0.5) * scaleY - 0.5));
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    const fy = sourceY - y0;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = Math.max(0, Math.min(sourceWidth - 1, (x + 0.5) * scaleX - 0.5));
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const fx = sourceX - x0;
      const p00 = y0 * sourceWidth + x0;
      const p10 = y0 * sourceWidth + x1;
      const p01 = y1 * sourceWidth + x0;
      const p11 = y1 * sourceWidth + x1;
      const target = y * targetWidth + x;
      for (let channel = 0; channel < 3; channel += 1) {
        const top =
          image.rgb[p00 * 3 + channel] * (1 - fx) +
          image.rgb[p10 * 3 + channel] * fx;
        const bottom =
          image.rgb[p01 * 3 + channel] * (1 - fx) +
          image.rgb[p11 * 3 + channel] * fx;
        rgb[target * 3 + channel] = top * (1 - fy) + bottom * fy;
      }
      const a00 = image.alpha?.[p00] ?? 1;
      const a10 = image.alpha?.[p10] ?? 1;
      const a01 = image.alpha?.[p01] ?? 1;
      const a11 = image.alpha?.[p11] ?? 1;
      alpha[target] =
        (a00 * (1 - fx) + a10 * fx) * (1 - fy) +
        (a01 * (1 - fx) + a11 * fx) * fy;
    }
  }
  return { width: targetWidth, height: targetHeight, rgb, alpha };
}

function makeCoarseTrainingImage(image, maxSide) {
  const currentSide = Math.max(image.width, image.height);
  if (currentSide <= maxSide) return null;
  const [width, height] = resizedSize(image.width, image.height, maxSide);
  return resizeFloatImageBilinear(image, width, height);
}

function curriculumCoarseMaxSide(fullSide, variants = phase33Variants()) {
  const boundedFullSide = Math.max(1, Math.round(fullSide));
  if (!variants.adaptiveCurriculum) return Math.min(boundedFullSide, variants.coarseMaxSide);
  // The curriculum is defined relative to the effective full training image,
  // including a 512px run: full / 4 -> full / 2 -> full.
  return Math.min(
    boundedFullSide,
    Math.max(CURRICULUM_COARSE_MIN_SIDE, Math.round(boundedFullSide / CURRICULUM_COARSE_DIVISOR)),
  );
}

function curriculumMidMaxSideForFullSide(fullSide, coarseMaxSide = PHASE33_COARSE_MAX_SIDE) {
  const boundedFullSide = Math.max(1, Math.round(fullSide));
  const coarseSide = Math.min(boundedFullSide, Math.max(1, Math.round(coarseMaxSide)));
  return Math.min(boundedFullSide, Math.max(coarseSide + 1, Math.round(boundedFullSide / 2)));
}

function curriculumMidMaxSide(image, coarseMaxSide = PHASE33_COARSE_MAX_SIDE) {
  return curriculumMidMaxSideForFullSide(Math.max(image.width, image.height), coarseMaxSide);
}

function curriculumStageDimensions(width, height, variants = phase33Variants()) {
  const full = { width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(height)) };
  const fullSide = Math.max(full.width, full.height);
  if (!variants.coarseToFull) {
    return { full, coarse: null, mid: null, coarseMaxSide: fullSide, midMaxSide: fullSide };
  }
  const coarseMaxSide = curriculumCoarseMaxSide(fullSide, variants);
  const midMaxSide = curriculumMidMaxSideForFullSide(fullSide, coarseMaxSide);
  const dimensionsAt = (maxSide) => {
    if (maxSide >= fullSide) return null;
    const [stageWidth, stageHeight] = resizedSize(full.width, full.height, maxSide);
    return { width: stageWidth, height: stageHeight };
  };
  return {
    full,
    coarse: dimensionsAt(coarseMaxSide),
    mid: variants.threeStageCurriculum ? dimensionsAt(midMaxSide) : null,
    coarseMaxSide,
    midMaxSide,
  };
}

function makeCurriculumImages(image, variants = phase33Variants()) {
  if (!variants.coarseToFull) return { coarseImage: null, midImage: null };
  const dimensions = curriculumStageDimensions(image.width, image.height, variants);
  const coarseImage = dimensions.coarse ? makeCoarseTrainingImage(image, dimensions.coarseMaxSide) : null;
  const midImage = dimensions.mid ? makeCoarseTrainingImage(image, dimensions.midMaxSide) : null;
  return { coarseImage, midImage, coarseMaxSide: dimensions.coarseMaxSide, midMaxSide: dimensions.midMaxSide };
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function splatGridLayout(image, count) {
  const cols = Math.max(1, Math.round(Math.sqrt((count * image.width) / image.height)));
  const rows = Math.max(1, Math.ceil(count / cols));
  const baseScale = INITIAL_SPLAT_COVERAGE_MULTIPLIER / Math.max(rows, cols);
  // Parameters remain in per-axis image NDC; this only matches initial coverage to each grid cell.
  const baseScaleX = aspectAwareGridEnabled() ? INITIAL_SPLAT_COVERAGE_MULTIPLIER / cols : baseScale;
  const baseScaleY = aspectAwareGridEnabled() ? INITIAL_SPLAT_COVERAGE_MULTIPLIER / rows : baseScale;
  return {
    rows,
    cols,
    baseScaleX,
    baseScaleY,
    baseScale,
  };
}

function stageMinimumScale(image, initialCount, trainingStage, ratio) {
  if (trainingStage === "full" || ratio <= 0) return MIN_SPLAT_SCALE;
  const referenceLayout = splatGridLayout(image, Math.max(1, initialCount));
  return Math.max(
    MIN_SPLAT_SCALE,
    Math.min(referenceLayout.baseScaleX, referenceLayout.baseScaleY) * ratio,
  );
}

function stageBaseScaleFloorRatio(trainingStage) {
  // Growth/reseed birth size is independent of the optional optimizer guard.
  // The optimizer may shrink useful detail later, but a phase transition must
  // not make new children start at the absolute minimum scale.
  if (trainingStage === "coarse") return DEFAULT_P1_BASE_SCALE_FLOOR_RATIO;
  if (trainingStage === "mid") return DEFAULT_P2_BASE_SCALE_FLOOR_RATIO;
  return DEFAULT_P3_BASE_SCALE_FLOOR_RATIO;
}

function stageRelativeScaleFloorRatio(trainingStage, variants = phase33Variants()) {
  if (trainingStage === "coarse") return variants.p1RelativeScaleFloorRatio;
  if (trainingStage === "mid") return variants.p2RelativeScaleFloorRatio;
  return variants.p3RelativeScaleFloorRatio;
}

function geometricMeanScaleMedian(params) {
  const values = [];
  for (let i = 0; i < (params?.count || 0); i += 1) {
    const sx = Math.max(MIN_SPLAT_SCALE, Number(params.scale[i * 2]) || MIN_SPLAT_SCALE);
    const sy = Math.max(MIN_SPLAT_SCALE, Number(params.scale[i * 2 + 1]) || MIN_SPLAT_SCALE);
    values.push(Math.sqrt(sx * sy));
  }
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)] || MIN_SPLAT_SCALE;
}

function splatGridAt(layout, index) {
  const col = index % layout.cols;
  const row = Math.floor(index / layout.cols);
  return constrainSplat(
    layout.cols === 1 ? 0 : -0.95 + (1.9 * col) / (layout.cols - 1),
    layout.rows === 1 ? 0 : -0.95 + (1.9 * row) / (layout.rows - 1),
    layout.baseScaleX,
    layout.baseScaleY,
  );
}

function initialSplatOrientation(index, columns = 1) {
  const safeIndex = Math.max(0, Math.round(index));
  const safeColumns = Math.max(1, Math.round(columns));
  const column = safeIndex % safeColumns;
  const row = Math.floor(safeIndex / safeColumns);
  let bits = (Math.imul(column + 1, 0x9e3779b1) ^ Math.imul(row + 1, 0x85ebca6b)) >>> 0;
  bits = ((bits >>> 16) | (bits << 16)) >>> 0;
  bits = (((bits & 0x55555555) << 1) | ((bits >>> 1) & 0x55555555)) >>> 0;
  bits = (((bits & 0x33333333) << 2) | ((bits >>> 2) & 0x33333333)) >>> 0;
  bits = (((bits & 0x0f0f0f0f) << 4) | ((bits >>> 4) & 0x0f0f0f0f)) >>> 0;
  bits = (((bits & 0x00ff00ff) << 8) | ((bits >>> 8) & 0x00ff00ff)) >>> 0;
  const unit = (bits + 0.5) / 4294967296;
  return (unit - 0.5) * Math.PI;
}

function initialSplatShape(image, layout, index) {
  const longSide = Math.max(1, image.width, image.height);
  const frameX = image.width / longSide;
  const frameY = image.height / longSide;
  const radiusX = frameX * layout.baseScaleX;
  const radiusY = frameY * layout.baseScaleY;
  const major = Math.max(radiusX, radiusY);
  const minor = Math.max(MIN_SPLAT_SCALE * Math.min(frameX, frameY), Math.min(radiusX, radiusY));
  const worldTheta = initialSplatOrientation(index, layout.cols);
  const c = Math.cos(worldTheta);
  const s = Math.sin(worldTheta);
  const major2 = major * major;
  const minor2 = minor * minor;
  const covarianceWorldX = c * c * major2 + s * s * minor2;
  const covarianceWorldY = s * s * major2 + c * c * minor2;
  const covarianceWorldXY = c * s * (major2 - minor2);
  const covarianceX = covarianceWorldX / (frameX * frameX);
  const covarianceY = covarianceWorldY / (frameY * frameY);
  const covarianceXY = -covarianceWorldXY / (frameX * frameY);
  const trace = covarianceX + covarianceY;
  const delta = Math.hypot(covarianceX - covarianceY, 2 * covarianceXY);
  return {
    sx: Math.sqrt(Math.max(MIN_SPLAT_SCALE ** 2, 0.5 * (trace + delta))),
    sy: Math.sqrt(Math.max(MIN_SPLAT_SCALE ** 2, 0.5 * (trace - delta))),
    theta: 0.5 * Math.atan2(2 * covarianceXY, covarianceX - covarianceY),
  };
}

function adaptiveBspImportanceGrid(image, count) {
  const longSide = Math.max(1, image.width, image.height);
  const targetLongSide = Math.min(
    longSide,
    Math.max(64, Math.min(384, Math.ceil(Math.sqrt(Math.max(1, count)) * 2))),
  );
  const width = Math.max(1, Math.round(image.width * targetLongSide / longSide));
  const height = Math.max(1, Math.round(image.height * targetLongSide / longSide));
  const luma = new Float32Array(width * height);
  const color = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(image.height - 1, Math.round((y + 0.5) * image.height / height - 0.5));
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.min(image.width - 1, Math.round((x + 0.5) * image.width / width - 0.5));
      const source = (sourceY * image.width + sourceX) * 3;
      const destination = (y * width + x) * 3;
      const r = image.rgb[source];
      const g = image.rgb[source + 1];
      const b = image.rgb[source + 2];
      color[destination] = r;
      color[destination + 1] = g;
      color[destination + 2] = b;
      luma[y * width + x] = r * 0.299 + g * 0.587 + b * 0.114;
    }
  }
  const importance = new Float32Array(width * height);
  let detailMaximum = 1e-6;
  for (let y = 0; y < height; y += 1) {
    const ym = Math.max(0, y - 1);
    const yp = Math.min(height - 1, y + 1);
    for (let x = 0; x < width; x += 1) {
      const xm = Math.max(0, x - 1);
      const xp = Math.min(width - 1, x + 1);
      const center = y * width + x;
      const gx = 0.5 * (luma[y * width + xp] - luma[y * width + xm]);
      const gy = 0.5 * (luma[yp * width + x] - luma[ym * width + x]);
      const laplacian = Math.abs(
        luma[y * width + xm] +
        luma[y * width + xp] +
        luma[ym * width + x] +
        luma[yp * width + x] -
        4 * luma[center],
      );
      let colorDifference = 0;
      const colorCenter = center * 3;
      for (const neighbor of [y * width + xm, y * width + xp, ym * width + x, yp * width + x]) {
        const offset = neighbor * 3;
        colorDifference += Math.hypot(
          color[colorCenter] - color[offset],
          color[colorCenter + 1] - color[offset + 1],
          color[colorCenter + 2] - color[offset + 2],
        );
      }
      const detail = 0.5 * Math.hypot(gx, gy) + 0.3 * laplacian + 0.05 * colorDifference;
      importance[center] = detail;
      detailMaximum = Math.max(detailMaximum, detail);
    }
  }
  for (let i = 0; i < importance.length; i += 1) {
    const normalized = Math.min(1, importance[i] / detailMaximum);
    importance[i] = 0.12 + 0.88 * Math.sqrt(normalized);
  }
  return { width, height, importance };
}

const structureGuidedRegionProfileCache = new WeakMap();

// Project-native allocator: it keeps the existing standard-alpha renderer,
// layer model, growth schedule, and optimizer, and only adds a soft regional
// destination budget when the user enables the checkbox.
function computeStructureGuidedRegionProfile(
  image,
  { lumaSpace = "srgb-baseline", regionGrid = DEFAULT_STRUCTURE_REGION_GRID } = {},
) {
  const started = performance.now();
  const lumaAt = (x, y) => {
    const px = Math.max(0, Math.min(image.width - 1, x));
    const py = Math.max(0, Math.min(image.height - 1, y));
    const offset = (py * image.width + px) * 3;
    if (lumaSpace === "srgb-baseline") {
      return 0.299 * image.rgb[offset] +
        0.587 * image.rgb[offset + 1] +
        0.114 * image.rgb[offset + 2];
    }
    return 0.2126729 * srgbSignalToLinear(image.rgb[offset]) +
      0.7151522 * srgbSignalToLinear(image.rgb[offset + 1]) +
      0.0721750 * srgbSignalToLinear(image.rgb[offset + 2]);
  };
  const smoothLumaAt = (x, y) => (
    lumaAt(x - 1, y - 1) + 2 * lumaAt(x, y - 1) + lumaAt(x + 1, y - 1) +
    2 * lumaAt(x - 1, y) + 4 * lumaAt(x, y) + 2 * lumaAt(x + 1, y) +
    lumaAt(x - 1, y + 1) + 2 * lumaAt(x, y + 1) + lumaAt(x + 1, y + 1)
  ) / 16;
  const measureGrid = (gridSize) => {
    const raw = new Float64Array(gridSize * gridSize);
    const sampleGrid = 8;
    for (let regionY = 0; regionY < gridSize; regionY += 1) {
      const y0 = Math.floor(regionY * image.height / gridSize);
      const y1 = Math.max(y0 + 1, Math.floor((regionY + 1) * image.height / gridSize));
      for (let regionX = 0; regionX < gridSize; regionX += 1) {
        const x0 = Math.floor(regionX * image.width / gridSize);
        const x1 = Math.max(x0 + 1, Math.floor((regionX + 1) * image.width / gridSize));
        let sum = 0;
        let sumSquared = 0;
        let samples = 0;
        for (let sy = 0; sy < sampleGrid; sy += 1) {
          const y = Math.min(image.height - 1, Math.floor(y0 + (sy + 0.5) * (y1 - y0) / sampleGrid));
          for (let sx = 0; sx < sampleGrid; sx += 1) {
            const x = Math.min(image.width - 1, Math.floor(x0 + (sx + 0.5) * (x1 - x0) / sampleGrid));
            const gx = 0.5 * (smoothLumaAt(x + 1, y) - smoothLumaAt(x - 1, y));
            const gy = 0.5 * (smoothLumaAt(x, y + 1) - smoothLumaAt(x, y - 1));
            const magnitude = Math.hypot(gx, gy);
            sum += magnitude;
            sumSquared += magnitude * magnitude;
            samples += 1;
          }
        }
        const mean = sum / Math.max(1, samples);
        raw[regionY * gridSize + regionX] = Math.sqrt(Math.max(
          0,
          sumSquared / Math.max(1, samples) - mean * mean,
        ));
      }
    }
    const sorted = Array.from(raw).sort((a, b) => a - b);
    const low = percentileSorted(sorted, 0.1);
    const high = percentileSorted(sorted, 0.9);
    const demand = Float64Array.from(raw, (value) => Math.max(
      0,
      Math.min(1, (Math.log1p(value * 4096) - Math.log1p(low * 4096)) /
        Math.max(1e-9, Math.log1p(high * 4096) - Math.log1p(low * 4096))),
    ));
    return { raw, demand, percentile10: low, percentile90: high };
  };
  const child = measureGrid(regionGrid);
  const profile = {
    ...child,
    lumaSpace,
    regionGrid,
    regionMode: `${regionGrid}x${regionGrid}`,
    processingMs: performance.now() - started,
  };
  return profile;
}

function structureGuidedRegionProfile(image, options = null) {
  const variants = options || phase39Variants();
  const lumaSpace = variants.structureLumaSpace || "srgb-baseline";
  const regionGrid = variants.structureRegionGrid || DEFAULT_STRUCTURE_REGION_GRID;
  let cachedByVariant = structureGuidedRegionProfileCache.get(image);
  if (!cachedByVariant) {
    cachedByVariant = new Map();
    structureGuidedRegionProfileCache.set(image, cachedByVariant);
  }
  const key = `${lumaSpace}:${regionGrid}`;
  if (!cachedByVariant.has(key)) {
    cachedByVariant.set(key, computeStructureGuidedRegionProfile(image, { lumaSpace, regionGrid }));
  }
  return cachedByVariant.get(key);
}

function structureGuidedRegionQuotas(image, targetCount, finalCount, options = null) {
  const profile = structureGuidedRegionProfile(image, options);
  const regionCount = profile.regionGrid * profile.regionGrid;
  const progress = Math.max(0, Math.min(1, targetCount / Math.max(1, finalCount)));
  const structureStrength = 0.88 - 0.06 * progress;
  const detailMass = Array.from(profile.demand, (value) => 0.02 + value ** 1.75);
  const detailTotal = detailMass.reduce((sum, value) => sum + value, 0);
  const exact = detailMass.map((value) => targetCount * (
    (1 - structureStrength) / regionCount +
    structureStrength * value / Math.max(1e-9, detailTotal)
  ));
  const quotas = Uint32Array.from(exact, Math.floor);
  let remaining = targetCount - quotas.reduce((sum, value) => sum + value, 0);
  const fractions = exact.map((value, region) => ({ region, fraction: value - Math.floor(value) }))
    .sort((a, b) => b.fraction - a.fraction || a.region - b.region);
  for (let i = 0; i < remaining; i += 1) quotas[fractions[i].region] += 1;
  return { ...profile, quotas, structureStrength };
}

function structureGuidedRegionControl(image, targetCount, finalCount) {
  const allocation = structureGuidedRegionQuotas(image, targetCount, finalCount);
  const control = new Uint32Array(PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE);
  const regionCount = allocation.regionGrid * allocation.regionGrid;
  for (let region = 0; region < regionCount; region += 1) {
    const base = region * PHASE45_REGION_STRIDE;
    control[base + 1] = Math.round(Math.min(1, allocation.demand[region]) * 65535);
    control[base + 9] = allocation.quotas[region];
  }
  return { control, allocation };
}

function structureGuidedProfileBenchmark(image = state.image, repetitions = 25) {
  if (!image) throw new Error("Load an image before benchmarking structure profiles.");
  const variants = ["linear-srgb", "srgb-baseline"].map((lumaSpace) => ({ lumaSpace }));
  return variants.map((variant) => {
    const samples = [];
    for (let i = 0; i < Math.max(1, Math.round(repetitions)); i += 1) {
      samples.push(computeStructureGuidedRegionProfile(image, variant).processingMs);
    }
    samples.sort((a, b) => a - b);
    return {
      ...variant,
      repetitions: samples.length,
      median_ms: percentileSorted(samples, 0.5),
      p10_ms: percentileSorted(samples, 0.1),
      p90_ms: percentileSorted(samples, 0.9),
    };
  });
}

function sourceDetailSplatDistribution(image, params) {
  const profile = structureGuidedRegionProfile(image);
  const regionGrid = profile.regionGrid;
  const regionCount = regionGrid * regionGrid;
  const demand = Array.from(profile.demand);
  const counts = new Uint32Array(regionCount);
  for (let index = 0; index < params.count; index += 1) {
    const x = Math.max(0, Math.min(0.999999, params.xy[index * 2] * 0.5 + 0.5));
    const y = Math.max(0, Math.min(0.999999, params.xy[index * 2 + 1] * 0.5 + 0.5));
    const regionX = Math.min(regionGrid - 1, Math.floor(x * regionGrid));
    const regionY = Math.min(regionGrid - 1, Math.floor(y * regionGrid));
    counts[regionY * regionGrid + regionX] += 1;
  }
  const regions = demand.map((value, region) => ({ region, demand: value, count: counts[region] }))
    .sort((a, b) => a.demand - b.demand || a.region - b.region);
  const quartileSize = Math.max(1, Math.floor(regions.length / 4));
  const bottom = regions.slice(0, quartileSize);
  const top = regions.slice(-quartileSize);
  const meanCount = (items) => items.reduce((sum, item) => sum + item.count, 0) / items.length;
  const flatMean = meanCount(bottom);
  const detailMean = meanCount(top);
  const countMean = params.count / regionCount;
  const demandMean = demand.reduce((sum, value) => sum + value, 0) / demand.length;
  let covariance = 0;
  let countVariance = 0;
  let demandVariance = 0;
  for (let region = 0; region < regionCount; region += 1) {
    const countDelta = counts[region] - countMean;
    const demandDelta = demand[region] - demandMean;
    covariance += countDelta * demandDelta;
    countVariance += countDelta * countDelta;
    demandVariance += demandDelta * demandDelta;
  }
  return {
    grid: [regionGrid, regionGrid],
    count: params.count,
    flat_quartile_mean_count: flatMean,
    detail_quartile_mean_count: detailMean,
    detail_to_flat_count_ratio: detailMean / Math.max(1e-9, flatMean),
    demand_count_correlation: covariance / Math.max(1e-9, Math.sqrt(countVariance * demandVariance)),
    minimum_region_count: Math.min(...counts),
    maximum_region_count: Math.max(...counts),
  };
}

function adaptiveBspIntegral(grid, valueAt) {
  const stride = grid.width + 1;
  const values = new Float64Array(stride * (grid.height + 1));
  for (let y = 0; y < grid.height; y += 1) {
    let row = 0;
    for (let x = 0; x < grid.width; x += 1) {
      row += valueAt(x, y, grid.importance[y * grid.width + x]);
      values[(y + 1) * stride + x + 1] = values[y * stride + x + 1] + row;
    }
  }
  return { values, stride };
}

function adaptiveBspSum(integral, x0, y0, x1, y1) {
  const { values, stride } = integral;
  return values[y1 * stride + x1] -
    values[y0 * stride + x1] -
    values[y1 * stride + x0] +
    values[y0 * stride + x0];
}

function adaptiveBspRegions(image, count) {
  const grid = adaptiveBspImportanceGrid(image, count);
  const massIntegral = adaptiveBspIntegral(grid, (_x, _y, importance) => importance);
  const xIntegral = adaptiveBspIntegral(grid, (x, _y, importance) => importance * (x + 0.5));
  const yIntegral = adaptiveBspIntegral(grid, (_x, y, importance) => importance * (y + 0.5));
  const x2Integral = adaptiveBspIntegral(grid, (x, _y, importance) => importance * (x + 0.5) ** 2);
  const y2Integral = adaptiveBspIntegral(grid, (_x, y, importance) => importance * (y + 0.5) ** 2);
  let serial = 0;
  const describe = (x0, y0, x1, y1) => {
    const mass = Math.max(1e-8, adaptiveBspSum(massIntegral, x0, y0, x1, y1));
    const meanX = adaptiveBspSum(xIntegral, x0, y0, x1, y1) / mass;
    const meanY = adaptiveBspSum(yIntegral, x0, y0, x1, y1) / mass;
    const varianceX = Math.max(0, adaptiveBspSum(x2Integral, x0, y0, x1, y1) / mass - meanX ** 2);
    const varianceY = Math.max(0, adaptiveBspSum(y2Integral, x0, y0, x1, y1) / mass - meanY ** 2);
    const splittable = x1 - x0 > 1 || y1 - y0 > 1;
    return {
      x0, y0, x1, y1, mass, meanX, meanY, varianceX, varianceY,
      serial: serial += 1,
      priority: splittable ? mass * Math.sqrt((x1 - x0) * (y1 - y0)) : -Infinity,
    };
  };
  const heap = [];
  const push = (region) => {
    heap.push(region);
    let index = heap.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (heap[parent].priority >= region.priority) break;
      heap[index] = heap[parent];
      index = parent;
    }
    heap[index] = region;
  };
  const pop = () => {
    const root = heap[0];
    const tail = heap.pop();
    if (heap.length && tail) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        const right = left + 1;
        if (left >= heap.length) break;
        const child = right < heap.length && heap[right].priority > heap[left].priority ? right : left;
        if (heap[child].priority <= tail.priority) break;
        heap[index] = heap[child];
        index = child;
      }
      heap[index] = tail;
    }
    return root;
  };
  push(describe(0, 0, grid.width, grid.height));
  while (heap.length < count) {
    const region = pop();
    if (!region || !Number.isFinite(region.priority)) {
      if (region) push(region);
      break;
    }
    const width = region.x1 - region.x0;
    const height = region.y1 - region.y0;
    const splitX = width > 1 && (height <= 1 || region.varianceX / Math.max(1, width ** 2) >= region.varianceY / Math.max(1, height ** 2));
    const start = splitX ? region.x0 : region.y0;
    const end = splitX ? region.x1 : region.y1;
    const targetMass = region.mass * 0.5;
    let cut = start + 1;
    let bestDifference = Infinity;
    for (let candidate = start + 1; candidate < end; candidate += 1) {
      const partialMass = splitX
        ? adaptiveBspSum(massIntegral, region.x0, region.y0, candidate, region.y1)
        : adaptiveBspSum(massIntegral, region.x0, region.y0, region.x1, candidate);
      const difference = Math.abs(partialMass - targetMass);
      if (difference < bestDifference) {
        bestDifference = difference;
        cut = candidate;
      }
    }
    if (splitX) {
      push(describe(region.x0, region.y0, cut, region.y1));
      push(describe(cut, region.y0, region.x1, region.y1));
    } else {
      push(describe(region.x0, region.y0, region.x1, cut));
      push(describe(region.x0, cut, region.x1, region.y1));
    }
  }
  const regions = heap.sort((a, b) => a.meanY - b.meanY || a.meanX - b.meanX || a.serial - b.serial);
  return { grid, regions };
}

function applyAdaptiveBspPaintInitialization(image, params, kernelShape) {
  const { grid, regions } = adaptiveBspRegions(image, params.count);
  const extent = kernelShape === "rectangle" ? RECTANGLE_KERNEL_EXTENT : LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT;
  let regionAreaTotal = 0;
  for (let i = 0; i < params.count; i += 1) {
    const region = regions[i % Math.max(1, regions.length)] || {
      x0: 0, y0: 0, x1: grid.width, y1: grid.height,
      meanX: grid.width * 0.5, meanY: grid.height * 0.5,
    };
    const x = -1 + 2 * region.meanX / grid.width;
    const y = -1 + 2 * region.meanY / grid.height;
    const regionWidth = 2 * (region.x1 - region.x0) / grid.width;
    const regionHeight = 2 * (region.y1 - region.y0) / grid.height;
    const structure = strokeStructureAt(image, x, y, params.theta[i]);
    const theta = kernelShape === "rectangle"
      ? rectangleDirectedTaperTheta(
          image,
          x,
          y,
          structure.theta,
          params.rectangleEdgeDirectedTaper &&
            params.rectangleTopRatio < 1 - 0.000001,
        )
      : structure.theta;
    const areaScale = Math.sqrt(Math.max(MIN_SPLAT_SCALE ** 2, regionWidth * regionHeight)) * 0.62 / extent;
    const anisotropy = 1.25 + 3.25 * structure.coherence;
    const stretch = Math.sqrt(anisotropy);
    const constrained = constrainSplat(
      x,
      y,
      areaScale * stretch,
      areaScale / stretch,
      theta,
      params.boundarySigma,
      Math.max(4, anisotropy),
    );
    params.xy[i * 2] = constrained.x;
    params.xy[i * 2 + 1] = constrained.y;
    params.scale[i * 2] = constrained.sx;
    params.scale[i * 2 + 1] = constrained.sy;
    params.theta[i] = theta;
    if (kernelShape === "rectangle" && params.rectangleStructureAwareRatio) {
      params.detailTags[i] =
        structure.coherence >= RECTANGLE_STRUCTURE_MIN_COHERENCE &&
        structure.energy >= RECTANGLE_STRUCTURE_MIN_ENERGY
          ? 2
          : 1;
    }
    regionAreaTotal += regionWidth * regionHeight;
  }
  params.initializationScheme = "image-importance-bsp";
  params.initializationStats = {
    map_width: grid.width,
    map_height: grid.height,
    unique_region_count: regions.length,
    splat_count: params.count,
    mean_region_area_ndc: regionAreaTotal / Math.max(1, regions.length),
  };
  return params;
}

function initialOrientationStats(params, image) {
  const bins = new Array(8).fill(0);
  let anisotropyTotal = 0;
  let anisotropyMax = 1;
  let finite = true;
  for (let index = 0; index < params.count; index += 1) {
    const world = transformPlanarSplatForPly(
      params.xy[index * 2],
      params.xy[index * 2 + 1],
      params.scale[index * 2],
      params.scale[index * 2 + 1],
      params.theta[index],
      image,
    );
    const angle = ((world.theta % Math.PI) + Math.PI) % Math.PI;
    bins[Math.min(bins.length - 1, Math.floor(angle / Math.PI * bins.length))] += 1;
    const ratio = world.sx / Math.max(1e-12, world.sy);
    anisotropyTotal += ratio;
    anisotropyMax = Math.max(anisotropyMax, ratio);
    finite &&= Number.isFinite(angle + world.sx + world.sy + ratio);
  }
  return {
    scheme: "deterministic-spatially-decorrelated-world-angle",
    bins,
    bin_spread: Math.max(...bins) - Math.min(...bins),
    world_anisotropy_mean: anisotropyTotal / Math.max(1, params.count),
    world_anisotropy_max: anisotropyMax,
    finite,
  };
}

function selectedBoundarySigma() {
  return clampNumber(
    els.boundarySigma?.value,
    LIMITS.boundarySigmaMin,
    LIMITS.boundarySigmaMax,
    DEFAULT_BOUNDARY_SIGMA,
  );
}

function clampSplatCenter(value, margin = selectedBoundarySigma() * MIN_SPLAT_SCALE) {
  return Math.max(-1 + margin, Math.min(1 - margin, value));
}

function currentMaxAnisotropy() {
  return clampNumber(els.maxAnisotropy?.value, LIMITS.maxAnisotropyMin, LIMITS.maxAnisotropyMax, DEFAULT_MAX_ANISOTROPY);
}

function anisotropyLimitsForParams(params = null) {
  const detail = clampNumber(
    params?.maxAnisotropy ?? state.metrics?.learning_rates?.maxAnisotropy ?? currentMaxAnisotropy(),
    LIMITS.maxAnisotropyMin,
    LIMITS.maxAnisotropyMax,
    DEFAULT_MAX_ANISOTROPY,
  );
  const surface = Math.min(
    detail,
    clampNumber(
      params?.surfaceAnisotropy ?? state.metrics?.learning_rates?.surfaceAnisotropy ?? qualityRecoveryVariants().surfaceAnisotropy,
      LIMITS.maxAnisotropyMin,
      LIMITS.maxAnisotropyMax,
      DEFAULT_SURFACE_ANISOTROPY,
    ),
  );
  return { surface, detail };
}

function anisotropyLimitForTag(tag, params = null) {
  const limits = anisotropyLimitsForParams(params);
  return Math.floor(Number(tag) || 1) >= 2 ? limits.detail : limits.surface;
}

function capScaleAnisotropy(sx, sy, maxRatio = currentMaxAnisotropy()) {
  const safeRatio = Math.max(1, maxRatio);
  const safeSx = Math.max(MIN_SPLAT_SCALE, sx);
  const safeSy = Math.max(MIN_SPLAT_SCALE, sy);
  const major = Math.max(safeSx, safeSy);
  const minor = Math.max(MIN_SPLAT_SCALE, Math.min(safeSx, safeSy));
  if (major / minor <= safeRatio) return { sx: safeSx, sy: safeSy };
  const cappedMajor = minor * safeRatio;
  return safeSx >= safeSy ? { sx: cappedMajor, sy: minor } : { sx: minor, sy: cappedMajor };
}

function rotatedExtentAtSigma(sx, sy, theta = 0, sigma = BOUNDARY_SIGMA) {
  const c = Math.abs(Math.cos(theta));
  const s = Math.abs(Math.sin(theta));
  return {
    x: sigma * Math.hypot(c * sx, s * sy),
    y: sigma * Math.hypot(s * sx, c * sy),
  };
}

function rotatedSplatExtent(sx, sy, theta = 0, sigma = selectedBoundarySigma()) {
  return rotatedExtentAtSigma(sx, sy, theta, sigma);
}

function previewPaddingSpec(image, params, enabled = els.outsidePreviewToggle.checked) {
  if (!enabled || !image || !params?.count) {
    return { x: 0, y: 0, width: image?.width || 1, height: image?.height || 1, scaleX: 1, scaleY: 1, bytes: 0 };
  }
  let outsideX = 0;
  let outsideY = 0;
  for (let i = 0; i < params.count; i += 1) {
    const extent = rotatedExtentAtSigma(params.scale[i * 2], params.scale[i * 2 + 1], params.theta?.[i] || 0, RENDER_SIGMA);
    outsideX = Math.max(outsideX, Math.abs(params.xy[i * 2]) + extent.x - 1);
    outsideY = Math.max(outsideY, Math.abs(params.xy[i * 2 + 1]) + extent.y - 1);
  }
  const limitX = Math.min(MAX_PREVIEW_PADDING_PX, Math.round(image.width * MAX_PREVIEW_PADDING_FRACTION));
  const limitY = Math.min(MAX_PREVIEW_PADDING_PX, Math.round(image.height * MAX_PREVIEW_PADDING_FRACTION));
  const x = Math.min(limitX, Math.max(0, Math.ceil(outsideX * image.width * 0.5)));
  const y = Math.min(limitY, Math.max(0, Math.ceil(outsideY * image.height * 0.5)));
  const width = image.width + x * 2;
  const height = image.height + y * 2;
  return {
    x,
    y,
    width,
    height,
    scaleX: image.width / width,
    scaleY: image.height / height,
    bytes: Math.max(0, (width * height - image.width * image.height) * 4),
  };
}

function buildPreviewTileIndexData(image, params, options = {}) {
  const preview = previewPaddingSpec(image, params, Boolean(options.outside));
  const tileCols = Math.ceil(preview.width / TILE_SIZE);
  const tileRows = Math.ceil(preview.height / TILE_SIZE);
  const tileCount = tileCols * tileRows;
  const maxTileReferences = Number.isFinite(Number(options.maxTileReferences))
    ? Math.max(1, Math.floor(Number(options.maxTileReferences)))
    : Number.MAX_SAFE_INTEGER;
  const counts = new Uint32Array(tileCount);
  const bounds = new Int32Array(params.count * 4);
  const aspectStretch = Math.sqrt(Math.max(0.000001, Number(options.localAspectRatio) || 1));
  const scaleMultiplier = Number.isFinite(Number(options.splatScaleMultiplier))
    ? Math.max(0, Number(options.splatScaleMultiplier))
    : 1;
  const useEwa = phase33Variants().ewa2x2;
  const pixelSigma = MIP_PIXEL_SIGMA * 2 / Math.max(image.width, image.height);
  const pixelPadX = useEwa && image.width > 1 ? 0.5 / (image.width - 1) : 0;
  const pixelPadY = useEwa && image.height > 1 ? 0.5 / (image.height - 1) : 0;
  let referenceCount = 0;
  for (let index = 0; index < params.count; index += 1) {
    const theta = params.theta?.[index] || 0;
    const c = Math.abs(Math.cos(theta));
    const s = Math.abs(Math.sin(theta));
    const baseX = Math.max(0.0001, params.scale[index * 2] * scaleMultiplier * aspectStretch);
    const baseY = Math.max(0.0001, params.scale[index * 2 + 1] * scaleMultiplier / aspectStretch);
    const effectiveX = useEwa ? baseX : Math.hypot(baseX, pixelSigma);
    const effectiveY = useEwa ? baseY : Math.hypot(baseY, pixelSigma);
    const radiusX = (RENDER_SIGMA * (c * effectiveX + s * effectiveY) + pixelPadX) * preview.scaleX;
    const radiusY = (RENDER_SIGMA * (s * effectiveX + c * effectiveY) + pixelPadY) * preview.scaleY;
    const centerX = params.xy[index * 2] * preview.scaleX;
    const centerY = params.xy[index * 2 + 1] * preview.scaleY;
    const minX = Math.max(0, Math.min(preview.width - 1, Math.floor(((centerX - radiusX) * 0.5 + 0.5) * Math.max(0, preview.width - 1))));
    const maxX = Math.max(0, Math.min(preview.width - 1, Math.ceil(((centerX + radiusX) * 0.5 + 0.5) * Math.max(0, preview.width - 1))));
    const minY = Math.max(0, Math.min(preview.height - 1, Math.floor(((centerY - radiusY) * 0.5 + 0.5) * Math.max(0, preview.height - 1))));
    const maxY = Math.max(0, Math.min(preview.height - 1, Math.ceil(((centerY + radiusY) * 0.5 + 0.5) * Math.max(0, preview.height - 1))));
    const minTileX = Math.floor(minX / TILE_SIZE);
    const maxTileX = Math.floor(maxX / TILE_SIZE);
    const minTileY = Math.floor(minY / TILE_SIZE);
    const maxTileY = Math.floor(maxY / TILE_SIZE);
    bounds.set([minTileX, maxTileX, minTileY, maxTileY], index * 4);
    for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
      for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
        referenceCount += 1;
        if (referenceCount > maxTileReferences) {
          throw new Error("Preview creates too many tile references; reduce Splat scale or Local aspect ratio.");
        }
        counts[tileY * tileCols + tileX] += 1;
      }
    }
  }
  const offsets = new Uint32Array(tileCount + 1);
  for (let tile = 0; tile < tileCount; tile += 1) offsets[tile + 1] = offsets[tile] + counts[tile];
  const indices = new Uint32Array(offsets[tileCount]);
  const cursors = offsets.slice(0, tileCount);
  for (let index = 0; index < params.count; index += 1) {
    const offset = index * 4;
    for (let tileY = bounds[offset + 2]; tileY <= bounds[offset + 3]; tileY += 1) {
      for (let tileX = bounds[offset]; tileX <= bounds[offset + 1]; tileX += 1) {
        const tile = tileY * tileCols + tileX;
        indices[cursors[tile]++] = index;
      }
    }
  }
  const comparator = options.splatSmallFirstOrder
    ? (a, b) => splatPreviewOrderComparator(a, b, params)
    : params.layerOrderEnabled
      ? (a, b) => layerOrderComparator(a, b, params)
      : null;
  if (comparator) {
    for (let tile = 0; tile < tileCount; tile += 1) {
      if (offsets[tile + 1] - offsets[tile] > 1) {
        indices.subarray(offsets[tile], offsets[tile + 1]).sort(comparator);
      }
    }
  }
  return { preview, offsets, indices };
}

function constrainSplat(
  x,
  y,
  sx,
  sy,
  theta = 0,
  boundarySigma = selectedBoundarySigma(),
  maxAnisotropy = currentMaxAnisotropy(),
) {
  if (boundarySigma <= 0) {
    const capped = capScaleAnisotropy(sx, sy, maxAnisotropy);
    return {
      x: clampSplatCenter(x, 0),
      y: clampSplatCenter(y, 0),
      sx: capped.sx,
      sy: capped.sy,
    };
  }
  const minimumExtent = rotatedSplatExtent(MIN_SPLAT_SCALE, MIN_SPLAT_SCALE, theta, boundarySigma);
  let cx = clampSplatCenter(x, minimumExtent.x);
  let cy = clampSplatCenter(y, minimumExtent.y);
  const capped = capScaleAnisotropy(sx, sy, maxAnisotropy);
  const extent = rotatedSplatExtent(capped.sx, capped.sy, theta, boundarySigma);
  const fit = Math.min(
    1,
    (1 - Math.abs(cx)) / Math.max(extent.x, 1e-8),
    (1 - Math.abs(cy)) / Math.max(extent.y, 1e-8),
  );
  let fitted = capScaleAnisotropy(
    Math.max(MIN_SPLAT_SCALE, capped.sx * fit),
    Math.max(MIN_SPLAT_SCALE, capped.sy * fit),
    maxAnisotropy,
  );
  let finalExtent = rotatedSplatExtent(fitted.sx, fitted.sy, theta, boundarySigma);
  const globalFit = Math.min(1, 0.999 / Math.max(finalExtent.x, finalExtent.y));
  fitted = capScaleAnisotropy(
    Math.max(MIN_SPLAT_SCALE, fitted.sx * globalFit),
    Math.max(MIN_SPLAT_SCALE, fitted.sy * globalFit),
    maxAnisotropy,
  );
  finalExtent = rotatedSplatExtent(fitted.sx, fitted.sy, theta, boundarySigma);
  cx = clampSplatCenter(cx, finalExtent.x);
  cy = clampSplatCenter(cy, finalExtent.y);
  return {
    x: cx,
    y: cy,
    sx: fitted.sx,
    sy: fitted.sy,
  };
}

function snapshotParams(params) {
  const anisotropyLimits = anisotropyLimitsForParams(params);
  return {
    kernelShape: normalizedKernelShape(params.kernelShape),
    rectangleTopRatio: clampNumber(
      params.rectangleTopRatio,
      MIN_RECTANGLE_TOP_RATIO,
      MAX_RECTANGLE_TOP_RATIO,
      DEFAULT_RECTANGLE_TOP_RATIO,
    ),
    rectangleTopRatioMax: clampNumber(
      params.rectangleTopRatioMax,
      params.rectangleTopRatio,
      MAX_RECTANGLE_TOP_RATIO,
      DEFAULT_RECTANGLE_TOP_RATIO_MAX,
    ),
    rectangleOpacityGradientMin: clampNumber(
      params.rectangleOpacityGradientMin,
      0,
      1,
      1,
    ),
    rectangleOpacityGradientMax: clampNumber(
      params.rectangleOpacityGradientMax,
      0,
      1,
      1,
    ),
    rectangleCenterOpacityGradientMin: clampNumber(
      params.rectangleCenterOpacityGradientMin,
      0,
      1,
      1,
    ),
    rectangleCenterOpacityGradientMax: clampNumber(
      params.rectangleCenterOpacityGradientMax,
      clampNumber(params.rectangleCenterOpacityGradientMin, 0, 1, 1),
      1,
      1,
    ),
    rectangleMinAspectRatio: clampNumber(
      params.rectangleMinAspectRatio,
      MIN_RECTANGLE_ASPECT_RATIO,
      params.rectangleMaxAspectRatio,
      DEFAULT_RECTANGLE_MIN_ASPECT_RATIO,
    ),
    rectangleMaxAspectRatio: clampNumber(
      params.rectangleMaxAspectRatio,
      MIN_RECTANGLE_ASPECT_RATIO,
      MAX_RECTANGLE_ASPECT_RATIO,
      DEFAULT_RECTANGLE_ASPECT_RATIO,
    ),
    rectangleOrientation: normalizedRectangleOrientation(params.rectangleOrientation),
    rectanglePreserveArea:
      params.rectanglePreserveArea ?? DEFAULT_RECTANGLE_PRESERVE_AREA,
    rectangleEdgeDirectedTaper:
      params.rectangleEdgeDirectedTaper ?? DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER,
    rectangleStructureAwareRatio:
      params.rectangleStructureAwareRatio ?? DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO,
    rectangleAsymmetricSoftness:
      params.rectangleAsymmetricSoftness ?? DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS,
    opaqueLayered: Boolean(params.opaqueLayered),
    minimumOpacityEnabled: Boolean(params.minimumOpacityEnabled),
    minimumOpacity: clampNumber(
      params.minimumOpacity,
      MIN_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
    ),
    maximumOpacity: clampNumber(
      params.maximumOpacity,
      params.minimumOpacity,
      MAX_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
    ),
    brushMinAspectRatio: clampNumber(
      params.brushMinAspectRatio,
      LIMITS.maxAnisotropyMin,
      params.brushMaxAspectRatio,
      LIMITS.maxAnisotropyMin,
    ),
    brushMaxAspectRatio: clampNumber(
      params.brushMaxAspectRatio,
      LIMITS.maxAnisotropyMin,
      LIMITS.maxAnisotropyMax,
      DEFAULT_MAX_ANISOTROPY,
    ),
    illustrativeOilVersion: Math.max(0, Math.round(Number(params.illustrativeOilVersion) || 0)),
    brushLocalColorFlowEnabled: Boolean(params.brushLocalColorFlowEnabled),
    brushStrokePersistenceEnabled: Boolean(params.brushStrokePersistenceEnabled),
    brushRibbonAspectFloor: clampNumber(
      params.brushRibbonAspectFloor,
      1,
      LIMITS.maxAnisotropyMax,
      BRUSH_STROKE_PERSISTENCE_RIBBON_MIN_RATIO,
    ),
    brushAccentAspectFloor: clampNumber(
      params.brushAccentAspectFloor,
      1,
      LIMITS.maxAnisotropyMax,
      BRUSH_STROKE_PERSISTENCE_ACCENT_MIN_RATIO,
    ),
    surfaceLayerPriorEnabled: Boolean(params.surfaceLayerPriorEnabled),
    surfaceLayerPriorColorAwarePromotion:
      params.surfaceLayerPriorColorAwarePromotion !== false,
    trainLayerColorGuardEnabled: Boolean(params.trainLayerColorGuardEnabled),
    surfaceLayerPriorLayers: Math.round(clampNumber(
      params.surfaceLayerPriorLayers,
      MIN_DISCRETE_LAYER_COUNT,
      MAX_DISCRETE_LAYER_COUNT,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
    )),
    surfaceLayerPriorP1Interval: Math.max(0, Math.round(params.surfaceLayerPriorP1Interval || 0)),
    surfaceLayerPriorP2Interval: Math.max(0, Math.round(params.surfaceLayerPriorP2Interval || 0)),
    surfaceLayerPriorP3Interval: Math.max(0, Math.round(params.surfaceLayerPriorP3Interval || 0)),
    surfaceLayerPriorUntilFraction: clampNumber(
      params.surfaceLayerPriorUntilFraction,
      0,
      1,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_UNTIL,
    ),
    harmfulRectangleParentSplitEnabled: Boolean(params.harmfulRectangleParentSplitEnabled),
    harmfulRectangleParentSplitTransitionOnly: Boolean(
      params.harmfulRectangleParentSplitTransitionOnly,
    ),
    frontSplitChildrenEnabled: Boolean(params.frontSplitChildrenEnabled),
    brushOpacityGradientEnabled: Boolean(params.brushOpacityGradientEnabled),
    brushOpacityGradientStart: clampNumber(params.brushOpacityGradientStart, 0, 1, 0),
    brushOpacityGradientEnd: clampNumber(params.brushOpacityGradientEnd, 0, 1, 1),
    brushCenterOpacityGradientMin: clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1),
    brushCenterOpacityGradientMax: clampNumber(
      params.brushCenterOpacityGradientMax,
      clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1),
      1,
      1,
    ),
    brushWidthTaperEnabled: Boolean(params.brushWidthTaperEnabled),
    brushWidthTaperStart: clampNumber(params.brushWidthTaperStart, 0, 1, 1),
    brushWidthTaperEnd: clampNumber(params.brushWidthTaperEnd, 0, 1, 0),
    monochromeUnderpaintingEnabled: Boolean(params.monochromeUnderpaintingEnabled),
    colorFinishStartPercent: clampNumber(
      params.colorFinishStartPercent,
      MIN_COLOR_FINISH_START_PERCENT,
      MAX_COLOR_FINISH_START_PERCENT,
      DEFAULT_COLOR_FINISH_START_PERCENT,
    ),
    colorFinishStartStep: Math.max(
      0,
      Math.round(Number(params.colorFinishStartStep) || 0),
    ),
    currentVisibilityChildPolicyEnabled: params.currentVisibilityChildPolicyEnabled !== false,
    currentVisibilityCompactionEnabled: params.currentVisibilityCompactionEnabled !== false,
    illustrativeOilFamilyStats: params.illustrativeOilFamilyStats
      ? structuredClone(params.illustrativeOilFamilyStats)
      : null,
    count: params.count,
    xy: new Float32Array(params.xy),
    scale: new Float32Array(params.scale),
    rgb: new Float32Array(params.rgb),
    opacity: new Float32Array(params.opacity),
    theta: new Float32Array(params.theta),
    depthOrder: params.depthOrder ? new Float32Array(params.depthOrder) : initialDepthOrder(params.count),
    virtualDepth: params.virtualDepth ? new Float32Array(params.virtualDepth) : new Float32Array(params.count),
    brushTaper: params.brushTaper
      ? new Float32Array(params.brushTaper)
      : new Float32Array(params.count).fill(DEFAULT_LAYERED_BRUSH_TAPER),
    virtualDepthEnabled: Boolean(params.virtualDepthEnabled),
    virtualDepthThickness: Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS,
    virtualDepthSoftConstraintEnabled: params.virtualDepthSoftConstraintEnabled !== false,
    virtualDepthPriorDelta: Number(params.virtualDepthPriorDelta) || DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA,
    detailTags: params.detailTags ? new Float32Array(params.detailTags) : new Float32Array(params.count).fill(1),
    boundarySigma: Number.isFinite(params.boundarySigma) ? params.boundarySigma : selectedBoundarySigma(),
    layerOrderEnabled: Boolean(params.layerOrderEnabled),
    layerAwareAccumulationEnabled: Boolean(params.layerAwareAccumulationEnabled),
    discreteLayersEnabled: Boolean(params.discreteLayersEnabled),
    discreteLayerCount: Math.max(MIN_DISCRETE_LAYER_COUNT, Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(params.discreteLayerCount || DEFAULT_DISCRETE_LAYER_COUNT))),
    discreteLayerMoveRadius: Math.max(0, Math.round(params.discreteLayerMoveRadius ?? DEFAULT_DISCRETE_LAYER_MOVE_RADIUS)),
    maxAnisotropy: anisotropyLimits.detail,
    surfaceAnisotropy: anisotropyLimits.surface,
    rows: params.rows,
    cols: params.cols,
    bg: params.bg ? new Float32Array(params.bg) : new Float32Array([0, 0, 0]),
  };
}

function nonfiniteParamCount(params) {
  let count = 0;
  for (const values of [params?.xy, params?.scale, params?.rgb, params?.opacity, params?.theta, params?.depthOrder, params?.virtualDepth, params?.brushTaper, params?.detailTags]) {
    if (!values) continue;
    for (let i = 0; i < values.length; i += 1) {
      if (!Number.isFinite(values[i])) count += 1;
    }
  }
  return count;
}

function assertFiniteParams(params, context) {
  const count = nonfiniteParamCount(params);
  if (count > 0) throw runtimeSafetyError("safety_stop_nonfinite_params", context, { nonfinite_values: count });
}

function finalSplatInspectorNonfiniteCount(params, metrics) {
  const finalReadbackStep = metrics?.final_readback_step;
  const immutableFinal = Boolean(
    params &&
    metrics?.cpu_mirror_current &&
    Number.isFinite(finalReadbackStep) &&
    finalReadbackStep === metrics?.steps_done,
  );
  if (!immutableFinal) return nonfiniteParamCount(params);
  const cache = state.splatInspectorNonfiniteCache;
  if (
    cache?.params === params &&
    cache.finalReadbackStep === finalReadbackStep &&
    cache.count === params.count
  ) {
    return cache.value;
  }
  const value = nonfiniteParamCount(params);
  state.splatInspectorNonfiniteCache = { params, finalReadbackStep, count: params.count, value };
  return value;
}

function meanAbsDelta(a, b, length) {
  let total = 0;
  for (let i = 0; i < length; i += 1) total += Math.abs(b[i] - a[i]);
  return length > 0 ? total / length : 0;
}

function paramDeltaFromSnapshot(snapshot, params) {
  if (!snapshot) return null;
  const count = Math.min(snapshot.count, params.count);
  return {
    scope: "initial-prefix",
    count,
    position: meanAbsDelta(snapshot.xy, params.xy, count * 2),
    color: meanAbsDelta(snapshot.rgb, params.rgb, count * 3),
    opacity: meanAbsDelta(snapshot.opacity, params.opacity, count),
    scale: meanAbsDelta(snapshot.scale, params.scale, count * 2),
    rotation: meanAbsDelta(snapshot.theta, params.theta, count),
    layerOrder: meanAbsDelta(snapshot.depthOrder, params.depthOrder, count),
    virtualDepth: meanAbsDelta(snapshot.virtualDepth, params.virtualDepth, count),
  };
}

function boundaryLeakStats(params, sigma = params?.boundarySigma ?? selectedBoundarySigma()) {
  let count = 0;
  let maxLeak = 0;
  for (let i = 0; i < params.count; i += 1) {
    const extent = rotatedSplatExtent(params.scale[i * 2], params.scale[i * 2 + 1], params.theta?.[i] || 0, sigma);
    const leakX = Math.max(0, Math.abs(params.xy[i * 2]) + extent.x - 1);
    const leakY = Math.max(0, Math.abs(params.xy[i * 2 + 1]) + extent.y - 1);
    const leak = Math.max(leakX, leakY);
    if (leak > 1e-6) count += 1;
    maxLeak = Math.max(maxLeak, leak);
  }
  return { count, maxLeak };
}

function outsideRenderFootprintStats(params) {
  return boundaryLeakStats(params, RENDER_SIGMA);
}

function renderFootprintSupportFrame(image, params) {
  const frame = plyFrameScale(image);
  let supportX = 1;
  let supportY = 1;
  for (let i = 0; i < (params?.count || 0); i += 1) {
    const extent = rotatedExtentAtSigma(
      params.scale[i * 2],
      params.scale[i * 2 + 1],
      params.theta?.[i] || 0,
      RENDER_SIGMA,
    );
    supportX = Math.max(supportX, Math.abs(params.xy[i * 2]) + extent.x);
    supportY = Math.max(supportY, Math.abs(params.xy[i * 2 + 1]) + extent.y);
  }
  return {
    x: frame.x * supportX,
    y: frame.y * supportY,
    normalized_x: supportX,
    normalized_y: supportY,
  };
}

function optimizerFootprintHistogram(image = state.image, params = state.params) {
  if (!image || !params) return null;
  const labels = ["1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1023", "1024-2047", "2048-4095", "4096+"];
  const bins = Object.fromEntries(labels.map((label) => [label, 0]));
  const areas = [];
  let totalWaves = 0;
  let totalLaneCapacity = 0;
  const useEwa = phase33Variants().ewa2x2;
  const pixelSigma = MIP_PIXEL_SIGMA * 2 / Math.max(image.width, image.height);
  for (let i = 0; i < params.count; i += 1) {
    const theta = params.theta[i] || 0;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const baseX = Math.max(0.0001, params.scale[i * 2]);
    const baseY = Math.max(0.0001, params.scale[i * 2 + 1]);
    const scaleX = useEwa ? baseX : Math.sqrt(baseX * baseX + pixelSigma * pixelSigma);
    const scaleY = useEwa ? baseY : Math.sqrt(baseY * baseY + pixelSigma * pixelSigma);
    const padX = useEwa && image.width > 1 ? 0.5 / (image.width - 1) : 0;
    const padY = useEwa && image.height > 1 ? 0.5 / (image.height - 1) : 0;
    const radiusX = RENDER_SIGMA * (Math.abs(c) * scaleX + Math.abs(s) * scaleY) + padX;
    const radiusY = RENDER_SIGMA * (Math.abs(s) * scaleX + Math.abs(c) * scaleY) + padY;
    const centerX = params.xy[i * 2];
    const centerY = params.xy[i * 2 + 1];
    const minX = Math.floor((Math.max(-1, centerX - radiusX) * 0.5 + 0.5) * Math.max(0, image.width - 1));
    const maxX = Math.ceil((Math.min(1, centerX + radiusX) * 0.5 + 0.5) * Math.max(0, image.width - 1));
    const minY = Math.floor((Math.max(-1, centerY - radiusY) * 0.5 + 0.5) * Math.max(0, image.height - 1));
    const maxY = Math.ceil((Math.min(1, centerY + radiusY) * 0.5 + 0.5) * Math.max(0, image.height - 1));
    const area = Math.max(1, maxX - minX + 1) * Math.max(1, maxY - minY + 1);
    areas.push(area);
    const exponent = Math.max(0, Math.ceil(Math.log2(area + 1)) - 1);
    const label = exponent >= labels.length - 1 ? labels[labels.length - 1] : labels[exponent];
    bins[label] += 1;
    const waves = Math.ceil(area / 64);
    totalWaves += waves;
    totalLaneCapacity += waves * 64;
  }
  areas.sort((a, b) => a - b);
  const percentile = (q) => areas[Math.min(areas.length - 1, Math.floor(Math.max(0, areas.length - 1) * q))] ?? null;
  return {
    source: "qa-final-cpu-mirror",
    count: areas.length,
    bins,
    minimum: areas[0] ?? null,
    median: percentile(0.5),
    p90: percentile(0.9),
    p99: percentile(0.99),
    maximum: areas[areas.length - 1] ?? null,
    total_64_lane_waves: totalWaves,
    lane_utilization: areas.reduce((sum, area) => sum + area, 0) / Math.max(1, totalLaneCapacity),
  };
}

function distributionStats(values, binCount = 8, fixedRange = null) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) {
    return { count: 0, min: 0, median: 0, mean: 0, max: 0, bins: new Array(binCount).fill(0), range: [0, 0] };
  }
  const min = fixedRange?.[0] ?? finite[0];
  const max = fixedRange?.[1] ?? finite[finite.length - 1];
  const span = Math.max(max - min, Number.EPSILON);
  const bins = new Array(binCount).fill(0);
  let sum = 0;
  for (const value of finite) {
    sum += value;
    const normalized = Math.max(0, Math.min(1, (value - min) / span));
    bins[Math.min(binCount - 1, Math.floor(normalized * binCount))] += 1;
  }
  return {
    count: finite.length,
    min: finite[0],
    median: finite[Math.floor(finite.length / 2)],
    mean: sum / finite.length,
    max: finite[finite.length - 1],
    bins,
    range: [min, max],
  };
}

function splatShapeStats(params, image) {
  const count = params?.count || 0;
  if (!count) return null;
  const sampling = finalDiagnosticSampling(count);
  const maxSide = Math.max(image?.width || 1, image?.height || 1);
  const pixelScale = maxSide * 0.5 * RENDER_SIGMA;
  const values = [];
  const bins = [0, 0, 0, 0, 0];
  let minScale = Infinity;
  let maxScale = 0;
  let sumScale = 0;
  let tinySplatCount = 0;
  let boundaryTinySplatCount = 0;
  let interiorTinySplatCount = 0;
  let anisotropySum = 0;
  let anisotropyMax = 1;
  let elongatedCount = 0;
  let boundarySplatCount = 0;
  let nonfiniteCount = 0;
  const opacityValues = [];
  const sxValues = [];
  const syValues = [];
  const geometricMeanScaleValues = [];
  const radiusValues = [];
  const anisotropyValues = [];
  const rotationValues = [];
  for (let i = 0; i < count; i += 1) {
    const rawValues = [
      params.xy[i * 2],
      params.xy[i * 2 + 1],
      params.scale[i * 2],
      params.scale[i * 2 + 1],
      params.opacity[i],
      params.theta?.[i] || 0,
    ];
    if (rawValues.some((value) => !Number.isFinite(value))) nonfiniteCount += 1;
    const sx = Math.max(MIN_SPLAT_SCALE, params.scale[i * 2]);
    const sy = Math.max(MIN_SPLAT_SCALE, params.scale[i * 2 + 1]);
    const opacity = params.opacity[i];
    const rotation = Math.atan2(Math.sin(params.theta?.[i] || 0), Math.cos(params.theta?.[i] || 0));
    const scale = (sx + sy) * 0.5;
    const major = Math.max(sx, sy);
    const minor = Math.max(MIN_SPLAT_SCALE, Math.min(sx, sy));
    const ratio = major / minor;
    const radiusPx = major * pixelScale;
    const areaScale = Math.sqrt(sx * sy);
    const edgeBandX = 8 * 2 / Math.max(1, image?.width || 1);
    const edgeBandY = 8 * 2 / Math.max(1, image?.height || 1);
    const boundaryAnchored =
      Math.abs(params.xy[i * 2]) >= 1 - edgeBandX ||
      Math.abs(params.xy[i * 2 + 1]) >= 1 - edgeBandY;
    if (i % sampling.stride === 0) {
      opacityValues.push(opacity);
      sxValues.push(sx);
      syValues.push(sy);
      geometricMeanScaleValues.push(areaScale);
      radiusValues.push(radiusPx);
      anisotropyValues.push(ratio);
      rotationValues.push(rotation);
      values.push(scale);
    }
    if (boundaryAnchored) boundarySplatCount += 1;
    minScale = Math.min(minScale, scale);
    maxScale = Math.max(maxScale, scale);
    sumScale += scale;
    anisotropySum += ratio;
    anisotropyMax = Math.max(anisotropyMax, ratio);
    if (areaScale <= MIN_SPLAT_SCALE * 1.05) {
      tinySplatCount += 1;
      if (boundaryAnchored) boundaryTinySplatCount += 1;
      else interiorTinySplatCount += 1;
    }
    if (ratio >= 1.8) elongatedCount += 1;
    if (radiusPx < 1) bins[0] += 1;
    else if (radiusPx < 2) bins[1] += 1;
    else if (radiusPx < 4) bins[2] += 1;
    else if (radiusPx < 8) bins[3] += 1;
    else bins[4] += 1;
  }
  values.sort((a, b) => a - b);
  const medianScale = values[Math.floor(values.length / 2)] || 0;
  return {
    count,
    inspection_sample_count: values.length,
    inspection_sample_stride: sampling.stride,
    min_scale: minScale,
    median_scale: medianScale,
    mean_scale: sumScale / count,
    max_scale: maxScale,
    scale_histogram: {
      radius_px_lt_1: bins[0],
      radius_px_1_2: bins[1],
      radius_px_2_4: bins[2],
      radius_px_4_8: bins[3],
      radius_px_gte_8: bins[4],
    },
    tiny_splat_count: tinySplatCount,
    tiny_splat_ratio: tinySplatCount / count,
    boundary_tiny_splat_count: boundaryTinySplatCount,
    boundary_tiny_splat_ratio: boundaryTinySplatCount / count,
    interior_tiny_splat_count: interiorTinySplatCount,
    interior_tiny_splat_ratio: interiorTinySplatCount / count,
    anisotropy_ratio_mean: anisotropySum / count,
    anisotropy_ratio_max: anisotropyMax,
    elongated_splat_count: elongatedCount,
    boundary_splat_count: boundarySplatCount,
    nonfinite_splat_count: nonfiniteCount,
    inspection: {
      opacity: distributionStats(opacityValues, 8, [0, 1]),
      scale_x: distributionStats(sxValues),
      scale_y: distributionStats(syValues),
      geometric_mean_scale: distributionStats(geometricMeanScaleValues),
      radius_px: distributionStats(radiusValues),
      anisotropy: distributionStats(anisotropyValues, 8, [1, Math.max(1, anisotropyMax)]),
      rotation: distributionStats(rotationValues, 8, [-Math.PI, Math.PI]),
    },
  };
}

function renderSplatInspector() {
  const current = Boolean(
    state.splatBaseline &&
      state.params &&
      state.metrics?.cpu_mirror_current &&
      state.metrics?.final_readback_step === state.metrics?.steps_done,
  );
  els.splatsEmpty.hidden = current;
  els.splatsContent.hidden = !current;
  const disabled = !current || trainingLifecycleInputLocked();
  const effectsEnabled = Boolean(els.splatParameterEffects?.checked);
  els.splatParameterEffects.disabled = disabled;
  for (const control of [
    els.splatOpacity,
    els.splatKernelFalloff,
    els.splatScale,
    els.splatAspectRatio,
    els.splatSmallFirstOrder,
    els.resetSplatAdjustments,
  ]) {
    control.disabled = disabled || !effectsEnabled;
  }
  for (const control of [els.splatShapeGaussian, els.splatShapeRectangle, els.splatShapeOpaqueBrush]) {
    control.disabled = disabled || !effectsEnabled;
  }
  els.splatAlphaBackground.disabled = disabled;
  updateSplatShapeControls();
  document.documentElement.dataset.splatsInspectionReady = String(current);
  document.documentElement.dataset.splatsAdjustmentReady = String(current && !disabled);
  if (!current) return;

  const adjustment = state.metrics?.post_train_adjustments;
  els.splatsMeta.textContent = `${state.params.count.toLocaleString()} splats · step ${state.metrics.steps_done}`;
  els.splatAdjustStatus.textContent = !effectsEnabled
    ? "Effects off · trained Gaussian preview."
    : state.adjustingSplats
      ? "Updating the preview..."
      : adjustment
        ? `Live preview · revision ${state.metrics.params_revision}`
        : "Move a control to preview the result.";
  const data = document.documentElement.dataset;
  data.splatsInspectionStep = String(state.metrics.steps_done);
  data.splatsInspectionCount = String(state.params.count);
  data.splatsInspectionNonfinite = String(finalSplatInspectorNonfiniteCount(state.params, state.metrics));
  data.splatsParamsRevision = String(state.metrics.params_revision ?? 0);
  data.splatsCoverageRevision = String(state.metrics.coverage_revision ?? "");
}

function publishLifecycleInteractionState() {
  updatePreviewModeControls();
  renderSplatInspector();
  publishState();
}

function updateSplatAdjustmentLabels() {
  els.splatOpacityValue.textContent = `${Number(els.splatOpacity.value).toFixed(2)}x`;
  els.splatKernelFalloffValue.textContent = `${Number(els.splatKernelFalloff.value).toFixed(2)}x`;
  els.splatScaleValue.textContent = `${Number(els.splatScale.value).toFixed(2)}x`;
  els.splatAspectRatioValue.textContent = `${Number(els.splatAspectRatio.value).toFixed(2)}x`;
}

function updateSplatShapeControls() {
  for (const [button, shape] of [
    [els.splatShapeGaussian, "gaussian"],
    [els.splatShapeRectangle, "rectangle"],
    [els.splatShapeOpaqueBrush, "opaque-brush"],
  ]) {
    const active = state.splatPreviewShape === shape;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function setSplatPreviewShape(shape) {
  if (trainingLifecycleInputLocked()) return;
  state.splatPreviewShape = shape === "opaque-brush"
    ? "opaque-brush"
    : normalizedKernelShape(shape);
  updateSplatShapeControls();
  refreshOutsidePreview().catch((error) => log(error.message));
  publishState();
}

function resetSplatAdjustmentControls() {
  els.splatOpacity.value = "1";
  els.splatKernelFalloff.value = "1";
  els.splatScale.value = "1";
  els.splatAspectRatio.value = "1";
  els.splatSmallFirstOrder.checked = false;
  const trainedShape = trainedSplatShape();
  state.splatPreviewShape = trainedShape === "box" ? "rectangle" : trainedShape;
  updateSplatAdjustmentLabels();
  updateSplatShapeControls();
}

function clearSplatAdjustmentBaseline() {
  window.clearTimeout(state.splatAdjustmentValidationTimer);
  if (state.splatAdjustmentFrame) window.cancelAnimationFrame(state.splatAdjustmentFrame);
  state.splatAdjustmentEpoch += 1;
  state.splatInspectorNonfiniteCache = null;
  state.splatBaseline = null;
  state.adjustingSplats = false;
  state.splatAdjustmentVersion = 0;
  state.splatAdjustmentFrame = 0;
  state.splatAdjustmentRenderedVersion = 0;
  state.splatAdjustmentValidationVersion = 0;
  state.splatAdjustmentStats = { requests: 0, renders: 0, staleDropped: 0, yields: 0, lastMs: 0 };
  els.splatParameterEffects.checked = true;
  els.outsidePreviewToggle.checked = false;
  resetSplatAdjustmentControls();
  els.splatAdjustStatus.textContent = "Ready.";
  renderSplatInspector();
}

function captureSplatAdjustmentBaseline() {
  if (!state.params || !state.metrics?.cpu_mirror_current) return;
  state.splatAdjustmentEpoch += 1;
  state.splatInspectorNonfiniteCache = null;
  // Final readback already owns an immutable CPU mirror. Live adjustments are
  // shader uniforms, so a second full parameter clone is unnecessary.
  state.splatBaseline = state.params;
  state.metrics.params_revision = 0;
  state.metrics.coverage_revision = 0;
  state.metrics.post_train_adjustments = null;
  state.splatAdjustmentVersion = 0;
  state.splatAdjustmentRenderedVersion = 0;
  state.splatAdjustmentValidationVersion = 0;
  state.splatAdjustmentStats = { requests: 0, renders: 0, staleDropped: 0, yields: 0, lastMs: 0 };
  els.splatParameterEffects.checked = true;
  resetSplatAdjustmentControls();
  renderSplatInspector();
}

function scaleSplatLocalAspectRatio(sx, sy, aspectRatio) {
  const stretch = Math.sqrt(Math.max(1e-6, aspectRatio));
  return {
    sx: Math.max(MIN_SPLAT_SCALE, sx * stretch),
    sy: Math.max(MIN_SPLAT_SCALE, sy / stretch),
  };
}

async function adjustedParamsFromBaseline(version, epoch) {
  const baseline = state.splatBaseline;
  if (!baseline) throw new Error("Finish training before adjusting splats.");
  const values = currentSplatAdjustmentValues();
  const { opacityMultiplier, splatScaleMultiplier, localAspectRatio } = values;
  if (opacityMultiplier === 1 && splatScaleMultiplier === 1 && localAspectRatio === 1) {
    return { params: baseline, values };
  }
  const params = snapshotParams(baseline);
  for (let i = 0; i < params.count; i += 1) {
    if (i > 0 && i % 8192 === 0) {
      state.splatAdjustmentStats.yields += 1;
      await nextFrame();
      if (
        epoch !== state.splatAdjustmentEpoch ||
        baseline !== state.splatBaseline ||
        version !== state.splatAdjustmentVersion
      ) return null;
    }
    const localScale = scaleSplatLocalAspectRatio(
      baseline.scale[i * 2] * splatScaleMultiplier,
      baseline.scale[i * 2 + 1] * splatScaleMultiplier,
      localAspectRatio,
    );
    const constrained = constrainSplat(
      baseline.xy[i * 2],
      baseline.xy[i * 2 + 1],
      localScale.sx,
      localScale.sy,
      baseline.theta[i],
      baseline.boundarySigma,
      anisotropyLimitForTag(baseline.detailTags?.[i], baseline),
    );
    params.xy[i * 2] = constrained.x;
    params.xy[i * 2 + 1] = constrained.y;
    params.scale[i * 2] = constrained.sx;
    params.scale[i * 2 + 1] = constrained.sy;
    params.theta[i] = baseline.theta[i];
    params.opacity[i] = opacityMultiplier >= 10
      ? 0.999999
      : Math.min(0.999999, Math.max(1e-6, baseline.opacity[i] * opacityMultiplier));
  }
  assertFiniteParams(params, "post-training-adjustment");
  return { params, values };
}

function captureSplatAdjustmentSnapshot() {
  const baseline = state.splatBaseline;
  return {
    baseline,
    sourceParams: baseline || state.params,
    epoch: state.splatAdjustmentEpoch,
    version: state.splatAdjustmentVersion,
  };
}

function splatAdjustmentSnapshotIsCurrent(snapshot) {
  return Boolean(
    snapshot &&
    snapshot.epoch === state.splatAdjustmentEpoch &&
    snapshot.version === state.splatAdjustmentVersion &&
    snapshot.baseline === state.splatBaseline &&
    snapshot.sourceParams === (state.splatBaseline || state.params),
  );
}

async function materializeCurrentSplatAdjustmentSnapshot() {
  const snapshot = captureSplatAdjustmentSnapshot();
  if (!snapshot.sourceParams) return null;
  const materialized = snapshot.baseline
    ? await adjustedParamsFromBaseline(snapshot.version, snapshot.epoch)
    : { params: snapshot.sourceParams };
  if (!materialized?.params || !splatAdjustmentSnapshotIsCurrent(snapshot)) return null;
  // Keep this detached from the optimizer and baseline. It is an ephemeral
  // export/viewer payload, not a new trained parameter state.
  return { ...materialized, snapshot };
}

function currentSplatAdjustmentValues() {
  const enabled = Boolean(els.splatParameterEffects?.checked);
  return {
    enabled,
    opacityMultiplier: enabled ? clampNumber(els.splatOpacity.value, 0, 10, 1) : 1,
    kernelFalloff: enabled ? clampNumber(els.splatKernelFalloff.value, 0, 2, 1) : 1,
    splatScaleMultiplier: enabled ? clampNumber(els.splatScale.value, 0, 5, 1) : 1,
    localAspectRatio: enabled ? clampNumber(els.splatAspectRatio.value, 0.1, 10, 1) : 1,
  };
}

function splatAlphaRenderOptions() {
  const effectsEnabled = Boolean(els.splatParameterEffects?.checked);
  const trainedShape = trainedSplatShape();
  const adjustments = currentSplatAdjustmentValues();
  return {
    alphaBackground: hexColorToRgb(els.splatAlphaBackground?.value, [0, 0, 0]),
    splatSmallFirstOrder: effectsEnabled && Boolean(els.splatSmallFirstOrder?.checked),
    kernelFalloff: adjustments.kernelFalloff,
    opacityMultiplier: adjustments.opacityMultiplier,
    splatScaleMultiplier: adjustments.splatScaleMultiplier,
    localAspectRatio: adjustments.localAspectRatio,
    splatShape: effectsEnabled
      ? state.splatPreviewShape === "rectangle" && trainedShape === "box"
        ? "box"
        : state.splatPreviewShape
      : trainedShape,
  };
}

function lockAdjustedExport() {
  state.metrics.coverage_revision = null;
  updateDownloads(false);
  state.exportMessage = "Updating the splat preview...";
  updateExportPanel();
}

async function renderLiveSplatAdjustments(version, epoch) {
  if (
    epoch !== state.splatAdjustmentEpoch ||
    !state.splatBaseline ||
    !state.image ||
    !state.metrics ||
    trainingLifecycleInputLocked() ||
    version !== state.splatAdjustmentVersion
  ) return;
  const startedAt = performance.now();
  const values = currentSplatAdjustmentValues();
  if (epoch !== state.splatAdjustmentEpoch || version !== state.splatAdjustmentVersion) {
    state.splatAdjustmentStats.staleDropped += 1;
    return;
  }
  // Live adjustments are render uniforms over the immutable trained buffers.
  // Materialize modified CPU arrays only for formats that require parameters.
  state.params = state.splatBaseline;
  state.metrics.params_revision = (state.metrics.params_revision ?? 0) + 1;
  state.metrics.post_train_adjustments = values;
  lockAdjustedExport();
  state.previewMode = "splats";
  updatePreviewModeControls();
  await refreshOutsidePreview();
  if (
    epoch !== state.splatAdjustmentEpoch ||
    version !== state.splatAdjustmentVersion
  ) return;
  state.splatAdjustmentRenderedVersion = version;
  state.splatAdjustmentStats.renders += 1;
  state.splatAdjustmentStats.lastMs = performance.now() - startedAt;
  state.metrics.splat_adjustment_scheduler = { ...state.splatAdjustmentStats };
  renderSplatInspector();
  publishState();
}

async function validateLiveSplatAdjustments(version, epoch) {
  if (
    epoch !== state.splatAdjustmentEpoch ||
    !state.splatBaseline ||
    !state.image ||
    !state.metrics ||
    trainingLifecycleInputLocked() ||
    version !== state.splatAdjustmentVersion
  ) return;
  state.adjustingSplats = true;
  renderSplatInspector();
  try {
    // Splat controls are display/export adjustments. Re-uploading a complete
    // optimizer state here caused large allocations to race tab-driven preview
    // renders after training buffers had already been released. The live
    // preview above is the validation surface; training metrics stay unchanged.
    await nextFrame();
    if (
      epoch !== state.splatAdjustmentEpoch ||
      version !== state.splatAdjustmentVersion ||
      trainingLifecycleInputLocked()
    ) return;
    state.metrics.post_adjustment_overlap_diagnostics = null;
    state.metrics.post_adjustment_overlap_revision = null;
    // These controls are display/export uniforms over the immutable trained
    // parameters. Training coverage and quality stay valid; PNG export runs
    // its own RGBA round-trip check for the adjusted display surface.
    state.metrics.coverage_revision = state.metrics.params_revision ?? 0;
    updateDownloads(!state.metrics?.safety_stop);
    state.exportMessage = "Preview ready; training quality and coverage metrics are unchanged.";
    updateExportPanel();
  } catch (error) {
    if (isExpectedPreviewCancellation(error)) return;
    updateDownloads(false);
    state.exportMessage = `Adjustment failed: ${error.message}`;
    updateExportPanel();
    log(state.exportMessage);
  } finally {
    state.adjustingSplats = false;
    renderSplatInspector();
    publishState();
  }
}

async function processSplatAdjustmentQueue() {
  if (state.splatAdjustmentProcessing) return;
  state.splatAdjustmentProcessing = true;
  try {
    while (state.splatBaseline && state.image && state.metrics && !trainingLifecycleInputLocked()) {
      const epoch = state.splatAdjustmentEpoch;
      const version = state.splatAdjustmentVersion;
      if (state.splatAdjustmentRenderedVersion !== version) {
        await renderLiveSplatAdjustments(version, epoch);
        if (
          epoch !== state.splatAdjustmentEpoch ||
          version !== state.splatAdjustmentVersion
        ) continue;
      }
      if (state.splatAdjustmentValidationVersion === version) {
        state.splatAdjustmentValidationVersion = 0;
        await validateLiveSplatAdjustments(version, epoch);
        if (
          epoch !== state.splatAdjustmentEpoch ||
          version !== state.splatAdjustmentVersion
        ) continue;
      }
      break;
    }
  } catch (error) {
    if (isExpectedPreviewCancellation(error)) return;
    state.exportMessage = `Adjustment failed: ${error.message}`;
    updateExportPanel();
    log(state.exportMessage);
  } finally {
    state.splatAdjustmentProcessing = false;
    if (
      state.splatBaseline &&
      !trainingLifecycleInputLocked() &&
      (state.splatAdjustmentRenderedVersion !== state.splatAdjustmentVersion ||
        state.splatAdjustmentValidationVersion === state.splatAdjustmentVersion)
    ) {
      queueMicrotask(() => processSplatAdjustmentQueue());
    }
  }
}

function queueSplatAdjustments({ immediate = false } = {}) {
  updateSplatAdjustmentLabels();
  if (!state.splatBaseline || !state.image || !state.metrics || trainingLifecycleInputLocked()) return;
  const version = state.splatAdjustmentVersion + 1;
  state.splatAdjustmentVersion = version;
  state.splatAdjustmentStats.requests += 1;
  window.clearTimeout(state.splatAdjustmentValidationTimer);
  const enqueuePreview = () => {
    state.splatAdjustmentFrame = 0;
    processSplatAdjustmentQueue();
  };
  if (immediate) {
    if (state.splatAdjustmentFrame) window.cancelAnimationFrame(state.splatAdjustmentFrame);
    state.splatAdjustmentFrame = 0;
    enqueuePreview();
  } else if (!state.splatAdjustmentFrame) {
    state.splatAdjustmentFrame = window.requestAnimationFrame(enqueuePreview);
  }
  state.splatAdjustmentValidationTimer = window.setTimeout(() => {
    if (version !== state.splatAdjustmentVersion) return;
    state.splatAdjustmentValidationVersion = version;
    processSplatAdjustmentQueue();
  }, immediate ? 0 : 240);
}

function resetSplatAdjustments() {
  resetSplatAdjustmentControls();
  queueSplatAdjustments({ immediate: true });
}

function initGaussians(image, count) {
  count = normalizeActiveSplatCount(count, DEFAULT_INITIAL_SPLATS);
  const boundarySigma = selectedBoundarySigma();
  const layout = splatGridLayout(image, count);
  const bg = new Float32Array([0, 0, 0]);
  const xy = new Float32Array(count * 2);
  const scale = new Float32Array(count * 2);
  const rgb = new Float32Array(count * 3);
  const opacity = new Float32Array(count);
  const theta = new Float32Array(count);
  const depthOrder = initialDepthOrder(count);
  const virtualDepth = new Float32Array(count);
  const detailTags = new Float32Array(count).fill(1);

  for (let i = 0; i < count; i += 1) {
    const grid = splatGridAt(layout, i);
    const shape = initialSplatShape(image, layout, i);
    const c = constrainSplat(grid.x, grid.y, shape.sx, shape.sy, shape.theta, boundarySigma);
    xy[i * 2] = c.x;
    xy[i * 2 + 1] = c.y;
    scale[i * 2] = c.sx;
    scale[i * 2 + 1] = c.sy;
    sampleImageAt(image, xy[i * 2], xy[i * 2 + 1], rgb, i * 3);
    opacity[i] = 0.98;
    theta[i] = shape.theta;
  }
  return {
    kernelShape: "gaussian",
    xy, scale, rgb, opacity, theta, depthOrder, virtualDepth, detailTags, count,
    rows: layout.rows, cols: layout.cols, bg,
    boundarySigma,
    layerOrderEnabled: Boolean(els.trainLayerOrder?.checked),
    discreteLayersEnabled: discreteLayerSettings().enabled,
    discreteLayerCount: discreteLayerSettings().count,
    virtualDepthEnabled: false,
    virtualDepthThickness: DEFAULT_VIRTUAL_DEPTH_THICKNESS,
    virtualDepthSoftConstraintEnabled: DEFAULT_VIRTUAL_DEPTH_SOFT_CONSTRAINT,
    virtualDepthPriorDelta: DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA,
  };
}

function initRectangles(image, count) {
  return initOpaqueLayeredPaint(image, count, "rectangle");
}

function strokeTensorAt(image, x, y, radius = 1) {
  const sampleRadius = Math.max(1, Math.round(radius));
  const dx = 2 * sampleRadius / Math.max(1, image.width - 1);
  const dy = 2 * sampleRadius / Math.max(1, image.height - 1);
  const sample = new Float32Array(3);
  const lumaAt = (px, py) => {
    sampleImageRgbBilinear(image, px, py, sample);
    return sample[0] * 0.299 + sample[1] * 0.587 + sample[2] * 0.114;
  };
  let jxx = 0;
  let jxy = 0;
  let jyy = 0;
  const derivativeScale = 0.5 / sampleRadius;
  for (let oy = -1; oy <= 1; oy += 1) {
    for (let ox = -1; ox <= 1; ox += 1) {
      const px = x + ox * dx;
      const py = y + oy * dy;
      const gx = derivativeScale * (lumaAt(px + dx, py) - lumaAt(px - dx, py));
      const gy = derivativeScale * (lumaAt(px, py + dy) - lumaAt(px, py - dy));
      jxx += gx * gx;
      jxy += gx * gy;
      jyy += gy * gy;
    }
  }
  return { jxx: jxx / 9, jxy: jxy / 9, jyy: jyy / 9 };
}

function strokeStructureFromTensor(tensor, image, fallbackTheta = 0) {
  const { jxx, jxy, jyy } = tensor;
  const dx = 2 / Math.max(1, image.width - 1);
  const dy = 2 / Math.max(1, image.height - 1);
  const trace = jxx + jyy;
  if (!Number.isFinite(trace) || trace < 1e-10) {
    return { theta: fallbackTheta, coherence: 0, energy: 0 };
  }
  const separation = Math.hypot(jxx - jyy, 2 * jxy);
  const normalPixelTheta = 0.5 * Math.atan2(2 * jxy, jxx - jyy);
  const tangentPixelX = -Math.sin(normalPixelTheta);
  const tangentPixelY = Math.cos(normalPixelTheta);
  return {
    theta: Math.atan2(tangentPixelY * dy, tangentPixelX * dx),
    coherence: Math.max(0, Math.min(1, separation / Math.max(trace, 1e-10))),
    energy: trace,
  };
}

function strokeStructureAt(image, x, y, fallbackTheta = 0) {
  return strokeStructureFromTensor(strokeTensorAt(image, x, y, 1), image, fallbackTheta);
}

function rectangleDirectedTaperTheta(image, x, y, theta, enabled = true) {
  if (!enabled) return theta;
  const normalX = -Math.sin(theta);
  const normalY = Math.cos(theta);
  const offset = 4 / Math.max(1, Math.max(image.width, image.height) - 1);
  const positive = strokeStructureAt(
    image,
    x + normalX * offset,
    y + normalY * offset,
    theta,
  ).energy;
  const negative = strokeStructureAt(
    image,
    x - normalX * offset,
    y - normalY * offset,
    theta,
  ).energy;
  // Local -Y is the narrow edge. A pi flip preserves the axial direction
  // while putting that narrow edge on the stronger-detail side.
  return positive > negative + 1e-10 ? theta + Math.PI : theta;
}

function oilStrokeStructureAt(image, x, y, fallbackTheta = 0) {
  const fine = strokeTensorAt(image, x, y, 1);
  const medium = strokeTensorAt(image, x, y, 3);
  const coarse = strokeTensorAt(image, x, y, 7);
  return strokeStructureFromTensor({
    jxx: fine.jxx * 0.55 + medium.jxx * 0.30 + coarse.jxx * 0.15,
    jxy: fine.jxy * 0.55 + medium.jxy * 0.30 + coarse.jxy * 0.15,
    jyy: fine.jyy * 0.55 + medium.jyy * 0.30 + coarse.jyy * 0.15,
  }, image, fallbackTheta);
}

function blendAxialAngle(from, to, amount) {
  const t = Math.max(0, Math.min(1, amount));
  const x = (1 - t) * Math.cos(2 * from) + t * Math.cos(2 * to);
  const y = (1 - t) * Math.sin(2 * from) + t * Math.sin(2 * to);
  return 0.5 * Math.atan2(y, x);
}

function oilStrokeHierarchy(structure, fallbackTheta) {
  const detail = Math.max(0, Math.min(1, Math.sqrt(Math.max(0, structure.energy) / 0.0025)));
  const directionWeight = Math.max(0, Math.min(1, (structure.coherence - 0.08) / 0.37));
  return {
    detail,
    scaleMultiplier: 1.65 + (0.62 - 1.65) * detail,
    anisotropy: 1.45 + 5.55 * structure.coherence * (0.70 + 0.30 * detail),
    theta: blendAxialAngle(fallbackTheta, structure.theta, directionWeight * directionWeight * (3 - 2 * directionWeight)),
  };
}

function illustrativeOilFamily(detail, anisotropy) {
  if (detail >= ILLUSTRATIVE_OIL_DETAIL_THRESHOLD) return 2;
  return anisotropy >= ILLUSTRATIVE_OIL_RIBBON_ANISOTROPY ? 1 : 0;
}

function gaussianBlurScalar(values, width, height, sigma) {
  const boundedSigma = Math.max(0.6, Math.min(4, Number(sigma) || 1));
  const radius = Math.max(1, Math.min(10, Math.ceil(boundedSigma * 3)));
  const kernel = new Float32Array(radius * 2 + 1);
  let kernelSum = 0;
  for (let offset = -radius; offset <= radius; offset += 1) {
    const value = Math.exp(-(offset * offset) / (2 * boundedSigma * boundedSigma));
    kernel[offset + radius] = value;
    kernelSum += value;
  }
  for (let index = 0; index < kernel.length; index += 1) kernel[index] /= kernelSum;
  const horizontal = new Float32Array(values.length);
  const result = new Float32Array(values.length);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        sum += values[row + Math.max(0, Math.min(width - 1, x + offset))] * kernel[offset + radius];
      }
      horizontal[row + x] = sum;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sampleY = Math.max(0, Math.min(height - 1, y + offset));
        sum += horizontal[sampleY * width + x] * kernel[offset + radius];
      }
      result[y * width + x] = sum;
    }
  }
  return result;
}

function initLayeredOpaqueBrush(image, count) {
  return initOpaqueLayeredPaint(image, count, "opaque-brush");
}

function initOpaqueLayeredPaint(image, count, kernelShape) {
  const params = initGaussians(image, count);
  const seededTheta = params.theta.slice();
  params.kernelShape = kernelShape;
  params.rectangleTopRatio = kernelShape === "rectangle"
    ? selectedRectangleTopRatio()
    : DEFAULT_RECTANGLE_TOP_RATIO;
  params.rectangleTopRatioMax = kernelShape === "rectangle"
    ? selectedRectangleTopRatioMax(params.rectangleTopRatio)
    : DEFAULT_RECTANGLE_TOP_RATIO_MAX;
  const rectangleOpacityGradient = selectedRectangleOpacityGradient();
  params.rectangleOpacityGradientMin = rectangleOpacityGradient.min;
  params.rectangleOpacityGradientMax = rectangleOpacityGradient.max;
  const rectangleCenterOpacityGradient = selectedRectangleCenterOpacityGradient();
  params.rectangleCenterOpacityGradientMin = rectangleCenterOpacityGradient.min;
  params.rectangleCenterOpacityGradientMax = rectangleCenterOpacityGradient.max;
  const learnedOpacity = kernelShape === "rectangle"
    ? selectedRectangleLearnedOpacity()
    : selectedLayeredBrushLearnedOpacity();
  params.rectangleMinAspectRatio = kernelShape === "rectangle"
    ? selectedRectangleMinAspectRatio()
    : DEFAULT_RECTANGLE_MIN_ASPECT_RATIO;
  params.rectangleMaxAspectRatio = kernelShape === "rectangle"
    ? selectedRectangleMaxAspectRatio(params.rectangleMinAspectRatio)
    : DEFAULT_RECTANGLE_ASPECT_RATIO;
  params.rectangleOrientation = kernelShape === "rectangle"
    ? selectedRectangleOrientation()
    : DEFAULT_RECTANGLE_ORIENTATION;
  const rectangleShape = selectedRectangleShapeSettings();
  params.rectanglePreserveArea =
    kernelShape === "rectangle" ? rectangleShape.preserveArea : DEFAULT_RECTANGLE_PRESERVE_AREA;
  params.rectangleEdgeDirectedTaper =
    kernelShape === "rectangle"
      ? rectangleShape.edgeDirectedTaper
      : DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER;
  params.rectangleStructureAwareRatio =
    kernelShape === "rectangle"
      ? rectangleShape.structureAwareRatio
      : DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO;
  params.rectangleAsymmetricSoftness =
    kernelShape === "rectangle"
      ? rectangleShape.asymmetricSoftness
      : DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS;
  params.opaqueLayered = true;
  params.currentVisibilityChildPolicyEnabled = opaquePaintCurrentVisibilityChildPolicyEnabled();
  params.currentVisibilityCompactionEnabled = opaquePaintCurrentVisibilityCompactionEnabled();
  const surfaceLayerPrior = scaleBiasedSurfaceLayerPriorSettings();
  params.surfaceLayerPriorEnabled = surfaceLayerPrior.enabled;
  params.surfaceLayerPriorColorAwarePromotion = surfaceLayerPrior.colorAwarePromotion;
  params.trainLayerColorGuardEnabled = trainLayerColorGuardEnabled();
  params.surfaceLayerPriorLayers = surfaceLayerPrior.layers;
  params.surfaceLayerPriorP1Interval = surfaceLayerPrior.p1Interval;
  params.surfaceLayerPriorP2Interval = surfaceLayerPrior.p2Interval;
  params.surfaceLayerPriorP3Interval = surfaceLayerPrior.p3Interval;
  params.surfaceLayerPriorUntilFraction = surfaceLayerPrior.untilFraction;
  params.harmfulRectangleParentSplitEnabled =
    harmfulRectangleParentSplitSettings().enabled;
  params.frontSplitChildrenEnabled = frontSplitChildrenSettings().enabled;
  const directionalEffects = selectedLayeredBrushDirectionalEffects();
  const brushCenterOpacityGradient = selectedLayeredBrushCenterOpacityGradient();
  params.minimumOpacityEnabled = true;
  params.minimumOpacity = learnedOpacity.min;
  params.maximumOpacity = learnedOpacity.max;
  params.boundarySigma = Math.min(
    params.boundarySigma,
    kernelShape === "rectangle" ? RECTANGLE_KERNEL_EXTENT : LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT,
  );
  params.layerOrderEnabled = true;
  params.layerAwareAccumulationEnabled = true;
  params.discreteLayersEnabled = true;
  params.discreteLayerMoveRadius = LAYERED_OPAQUE_BRUSH_LAYER_MOVE_RADIUS;
  applyAdaptiveBspPaintInitialization(image, params, kernelShape);
  const baseCount = params.count;
  params.opacity.fill(params.maximumOpacity);
  const illustrativeOil = kernelShape === "opaque-brush";
  params.brushOpacityGradientEnabled = illustrativeOil && directionalEffects.opacity;
  params.brushOpacityGradientStart = directionalEffects.opacityStart;
  params.brushOpacityGradientEnd = directionalEffects.opacityEnd;
  params.brushCenterOpacityGradientMin = brushCenterOpacityGradient.min;
  params.brushCenterOpacityGradientMax = brushCenterOpacityGradient.max;
  params.brushWidthTaperEnabled = illustrativeOil && directionalEffects.widthTaper;
  params.brushWidthTaperStart = directionalEffects.widthStart;
  params.brushWidthTaperEnd = directionalEffects.widthEnd;
  params.brushTaper = new Float32Array(baseCount).fill(DEFAULT_LAYERED_BRUSH_TAPER);
  params.illustrativeOilVersion = illustrativeOil ? 1 : 0;
  params.brushLocalColorFlowEnabled = illustrativeOil && opaqueBrushLocalColorFlowEnabled();
  params.brushStrokePersistenceEnabled = illustrativeOil && opaqueBrushStrokePersistenceEnabled();
  params.brushMinAspectRatio = illustrativeOil
    ? selectedBrushMinAspectRatio()
    : LIMITS.maxAnisotropyMin;
  params.brushMaxAspectRatio = illustrativeOil
    ? selectedBrushMaxAspectRatio()
    : DEFAULT_MAX_ANISOTROPY;
  const brushAspectFloors = selectedBrushAspectFloors();
  params.brushRibbonAspectFloor = brushAspectFloors.ribbon;
  params.brushAccentAspectFloor = brushAspectFloors.accent;
  const illustrativeOilFamilyCounts = [0, 0, 0];
  for (let i = 0; i < baseCount; i += 1) {
    const x = params.xy[i * 2];
    const y = params.xy[i * 2 + 1];
    const structure = illustrativeOil
      ? oilStrokeStructureAt(image, x, y, seededTheta[i])
      : strokeStructureAt(image, x, y, seededTheta[i]);
    const areaScale = Math.sqrt(Math.max(MIN_SPLAT_SCALE ** 2, params.scale[i * 2] * params.scale[i * 2 + 1]));
    const oilHierarchy = illustrativeOil
      ? oilStrokeHierarchy(structure, seededTheta[i])
      : null;
    const anisotropy = oilHierarchy?.anisotropy ?? (1.35 + 2.65 * structure.coherence);
    // Preserve Rectangle's established image-structure initialization. The
    // configurable minimum is an optimizer constraint, not a reason to change
    // the starting footprint distribution.
    const stretch = Math.sqrt(anisotropy);
    const scaleMultiplier = oilHierarchy?.scaleMultiplier ?? 1;
    const unconstrainedTheta = illustrativeOil
      ? oilHierarchy.theta
      : rectangleDirectedTaperTheta(
          image,
          x,
          y,
          structure.theta,
          params.rectangleEdgeDirectedTaper &&
            params.rectangleTopRatio < 1 - 0.000001,
        );
    const nextSx = areaScale * scaleMultiplier * stretch;
    const nextSy = areaScale * scaleMultiplier / stretch;
    const theta = kernelShape === "rectangle"
      ? constrainedRectangleTheta(
          unconstrainedTheta,
          nextSx,
          nextSy,
          params.rectangleOrientation,
        )
      : unconstrainedTheta;
    const constrained = constrainSplat(
      x,
      y,
      nextSx,
      nextSy,
      theta,
      params.boundarySigma,
      kernelShape === "rectangle"
        ? Math.min(params.rectangleMaxAspectRatio, Math.max(4, anisotropy))
        : Math.max(7, anisotropy),
    );
    params.xy[i * 2] = constrained.x;
    params.xy[i * 2 + 1] = constrained.y;
    params.scale[i * 2] = constrained.sx;
    params.scale[i * 2 + 1] = constrained.sy;
    params.theta[i] = theta;
    if (illustrativeOil) {
      const constrainedAnisotropy =
        Math.max(constrained.sx, constrained.sy) / Math.max(MIN_SPLAT_SCALE, Math.min(constrained.sx, constrained.sy));
      const family = illustrativeOilFamily(oilHierarchy.detail, constrainedAnisotropy);
      params.detailTags[i] = family === 2 ? 2 : 1;
      illustrativeOilFamilyCounts[family] += 1;
    } else if (params.rectangleStructureAwareRatio) {
      params.detailTags[i] =
        structure.coherence >= RECTANGLE_STRUCTURE_MIN_COHERENCE &&
        structure.energy >= RECTANGLE_STRUCTURE_MIN_ENERGY
          ? 2
          : 1;
    }
  }
  params.illustrativeOilFamilyStats = illustrativeOil
    ? {
        version: 1,
        selection: "deterministic-local-structure",
        base_patch: illustrativeOilFamilyCounts[0],
        directional_ribbon: illustrativeOilFamilyCounts[1],
        edge_accent: illustrativeOilFamilyCounts[2],
      }
    : null;
  for (let i = 0; i < baseCount; i += 1) {
    footprintWeightedTargetColor(image, params, i, params.rgb.subarray(i * 3, i * 3 + 3));
  }
  return params;
}

function initialDepthOrder(count) {
  const values = new Float32Array(Math.max(0, count));
  const denominator = Math.max(1, count - 1);
  for (let i = 0; i < count; i += 1) values[i] = 1 - i / denominator;
  return values;
}

function packedLayerOrder(packedTag) {
  const fraction = packedTag - Math.floor(packedTag);
  return Math.max(0, Math.min(1, Math.min(fraction, LAYER_CODE_RANGE) / LAYER_CODE_RANGE));
}

function summarizeScaleBiasedSurfaceLayerSort(params) {
  const count = Math.max(0, Math.round(Number(params?.count) || 0));
  const layers = Math.round(clampNumber(
    params?.surfaceLayerPriorLayers,
    MIN_DISCRETE_LAYER_COUNT,
    MAX_DISCRETE_LAYER_COUNT,
    DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
  ));
  const actualCounts = Array(layers).fill(0);
  const expectedCounts = Array(layers).fill(0);
  const minimumArea = MIN_SPLAT_SCALE ** 2;
  const maximumAxis = Math.max(
    PHASE_ONE_MAX_PLANAR_SCALE,
    clampNumber(params?.maxPlanarScale, MIN_SPLAT_SCALE, 1, DEFAULT_MAX_PLANAR_SCALE),
  );
  const maximumArea = Math.max(minimumArea * 1.0001, maximumAxis ** 2);
  let matched = 0;
  for (let index = 0; index < count; index += 1) {
    const area = clampNumber(
      params.scale[index * 2] * params.scale[index * 2 + 1],
      minimumArea,
      maximumArea,
      minimumArea,
    );
    const sizeRank = clampNumber(
      (Math.log(area) - Math.log(minimumArea)) /
        Math.max(1e-6, Math.log(maximumArea) - Math.log(minimumArea)),
      0,
      1,
      0,
    );
    const expected = Math.min(layers - 1, Math.floor(Math.min(0.999999, 1 - sizeRank) * layers));
    const actual = Math.min(layers - 1, Math.floor(
      clampNumber(params.depthOrder?.[index], 0, 0.999999, 0) * layers,
    ));
    expectedCounts[expected] += 1;
    actualCounts[actual] += 1;
    if (actual === expected) matched += 1;
  }
  return {
    checked_splats: count,
    matching_splats: matched,
    match_ratio: matched / Math.max(1, count),
    expected_layer_counts: expectedCounts,
    actual_layer_counts: actualCounts,
  };
}

function boundedVirtualDepth(params, index) {
  if (!params?.virtualDepthEnabled) return 0;
  const thickness = Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS;
  return thickness * Math.tanh(params.virtualDepth?.[index] || 0);
}

function finalDiagnosticSampling(count, limit = MAX_FINAL_DIAGNOSTIC_SAMPLES) {
  const sourceCount = Math.max(0, Math.round(Number(count) || 0));
  const sampleLimit = Math.max(1, Math.round(Number(limit) || 1));
  const stride = Math.max(1, Math.ceil(sourceCount / sampleLimit));
  return {
    sourceCount,
    stride,
    sampleCount: sourceCount > 0 ? Math.ceil(sourceCount / stride) : 0,
  };
}

function virtualDepthDistribution(params) {
  if (!params?.count) return null;
  const sampling = finalDiagnosticSampling(params.count);
  const raw = new Float32Array(sampling.sampleCount);
  const virtual = new Float32Array(sampling.sampleCount);
  const composite = new Float32Array(sampling.sampleCount);
  let sampleIndex = 0;
  for (let index = 0; index < params.count; index += sampling.stride) {
    raw[sampleIndex] = Number(params.virtualDepth?.[index]) || 0;
    virtual[sampleIndex] = boundedVirtualDepth(params, index);
    composite[sampleIndex] = plyLayerDepth(index, params);
    sampleIndex += 1;
  }
  const summarize = (values) => {
    const sorted = values.slice().sort();
    const at = (fraction) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * fraction)))];
    return {
      minimum: sorted[0],
      p01: at(0.01),
      p50: at(0.5),
      p99: at(0.99),
      maximum: sorted.at(-1),
      mean_abs: values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length,
    };
  };
  return {
    enabled: Boolean(params.virtualDepthEnabled),
    thickness: Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS,
    source_count: sampling.sourceCount,
    sample_count: sampling.sampleCount,
    sample_stride: sampling.stride,
    raw: summarize(raw),
    virtual_z: summarize(virtual),
    composite_z: summarize(composite),
  };
}

function sampleImageAt(image, x, y, out, offset) {
  const px = Math.max(0, Math.min(image.width - 1, Math.round(((x + 1) * 0.5) * (image.width - 1))));
  const py = Math.max(0, Math.min(image.height - 1, Math.round(((y + 1) * 0.5) * (image.height - 1))));
  const source = (py * image.width + px) * 3;
  out[offset] = image.rgb[source];
  out[offset + 1] = image.rgb[source + 1];
  out[offset + 2] = image.rgb[source + 2];
}

function srgbSignalToLinear(value) {
  const channel = Math.max(0, Math.min(1, Number(value) || 0));
  return channel <= 0.04045
    ? channel / 12.92
    : ((channel + 0.055) / 1.055) ** 2.4;
}

function linearToSrgbSignal(value) {
  const channel = Math.max(0, Math.min(1, Number(value) || 0));
  return channel <= 0.0031308
    ? channel * 12.92
    : 1.055 * channel ** (1 / 2.4) - 0.055;
}

function neutralSrgbPreservingLabL(r, g, b) {
  const relativeY =
    0.2126729 * srgbSignalToLinear(r) +
    0.7151522 * srgbSignalToLinear(g) +
    0.0721750 * srgbSignalToLinear(b);
  return linearToSrgbSignal(relativeY);
}

function convertRgbToNeutralLabL(rgb) {
  for (let offset = 0; offset + 2 < rgb.length; offset += 3) {
    const gray = neutralSrgbPreservingLabL(
      rgb[offset],
      rgb[offset + 1],
      rgb[offset + 2],
    );
    rgb[offset] = gray;
    rgb[offset + 1] = gray;
    rgb[offset + 2] = gray;
  }
  return rgb;
}

function brushColorNeutrality(params = state.params) {
  const rgb = params?.rgb;
  const count = Math.min(params?.count || 0, Math.floor((rgb?.length || 0) / 3));
  let maximumChannelDelta = 0;
  let meanChannelDelta = 0;
  for (let index = 0; index < count; index += 1) {
    const offset = index * 3;
    const delta = Math.max(
      Math.abs(rgb[offset] - rgb[offset + 1]),
      Math.abs(rgb[offset + 1] - rgb[offset + 2]),
      Math.abs(rgb[offset + 2] - rgb[offset]),
    );
    maximumChannelDelta = Math.max(maximumChannelDelta, delta);
    meanChannelDelta += delta;
  }
  return {
    count,
    maximum_channel_delta: maximumChannelDelta,
    mean_channel_delta: meanChannelDelta / Math.max(1, count),
    completely_neutral: maximumChannelDelta <= 1e-6,
  };
}

function densifyWarmupSteps(steps) {
  return steps >= 10 ? Math.min(DENSIFY_WARMUP_MAX_STEPS, Math.floor(steps * DENSIFY_WARMUP_FRACTION)) : 0;
}

function curriculumStageStep(steps, fraction) {
  const total = Math.max(1, Math.round(steps));
  return Math.min(total, Math.max(1, Math.round(total * fraction)));
}

function experimentalCoarseSteps(steps, override = null) {
  const total = Math.max(1, Math.round(steps));
  if (Number.isFinite(override)) return Math.min(total, Math.max(0, Math.round(override)));
  return curriculumStageStep(total, CURRICULUM_COARSE_FRACTION);
}

function experimentalDensifySteps(steps) {
  return curriculumStageStep(steps, CURRICULUM_DENSITY_FRACTION);
}

function experimentalGrowthSteps(steps, fraction = phase39Variants().growthApplyUntilFraction) {
  const boundedFraction = Math.max(0, Math.min(1, Number(fraction) || 0));
  return boundedFraction > 0 ? curriculumStageStep(steps, boundedFraction) : 0;
}

function overdensityCorrectionScheduleSteps(steps, growthSteps, settings) {
  const horizon = Math.max(0, Math.round(growthSteps));
  if (!settings?.midTrainingOverdensityCorrection || horizon <= 0) return [];
  if (settings.overdensityCorrectionSchedule === "p2-p3-start") {
    return [
      experimentalCoarseSteps(steps) + 1,
      experimentalDensifySteps(steps) + 1,
    ].filter((step, index, values) => step <= horizon && values.indexOf(step) === index);
  }
  const interval = Math.max(100, Math.round(
    settings.overdensityCorrectionInterval || DEFAULT_OVERDENSITY_CORRECTION_INTERVAL,
  ));
  const scheduled = [];
  for (let step = interval; step <= horizon; step += interval) scheduled.push(step);
  return scheduled;
}

function growthEventScheduled(step, densitySteps, growthSteps, settings) {
  const current = Math.max(1, Math.round(step));
  const horizon = Math.max(0, Math.round(growthSteps));
  if (!settings?.densityEventsEnabled || horizon <= 0 || current > horizon) return false;
  if (current === horizon) return true;
  return current > densifyWarmupSteps(densitySteps) &&
    current % Math.max(1, Math.round(settings?.densifyInterval || 1)) === 0;
}

function curriculumTrainingStage(step, steps, variants, coarseImage, midImage) {
  const coarseEnd = experimentalCoarseSteps(steps, variants.coarseSteps);
  if (variants.coarseToFull && coarseImage && step <= coarseEnd) return "coarse";
  if (variants.threeStageCurriculum && midImage && step <= experimentalDensifySteps(steps)) return "mid";
  return "full";
}

function opaquePaintVisibilityCompactionStep(
  steps,
  fraction = OPAQUE_PAINT_LATE_SETTLE_FRACTION,
) {
  return Math.max(1, opaquePaintLateSettleStartStep(steps, fraction) - 1);
}

function opaquePaintVisibilityGraceSteps(steps) {
  return Math.min(
    OPAQUE_PAINT_VISIBILITY_GRACE_STEPS,
    Math.max(OPAQUE_PAINT_VISIBILITY_MIN_GAP_STEPS, Math.round(Math.max(1, steps) * 0.02)),
  );
}

function opaquePaintVisibilityCompactionDue(step, steps, settings) {
  if (
    !settings?.opaqueLayered ||
    settings?.currentVisibilityCompactionEnabled === false ||
    settings?.currentContributionCompactionEnabled === true
  ) return false;
  const compactionStep = opaquePaintVisibilityCompactionStep(
    steps,
    settings?.opaquePaintSettleFraction,
  );
  const growthEnd = experimentalGrowthSteps(steps, settings?.growthApplyUntilFraction);
  if (growthEnd > 0 && compactionStep >= growthEnd) return false;
  const grace = compactionStep - growthEnd;
  return grace >= opaquePaintVisibilityGraceSteps(steps) && Math.round(step) === compactionStep;
}

function currentContributionCompactionStep(steps, settings) {
  const total = Math.max(1, Math.round(steps));
  const measurementWindowSteps = Math.max(1, Math.round(settings?.measurementWindowSteps || 1));
  const deadline = currentContributionCompactionDeadline(total, settings);
  const requestedReset = Math.max(1, Math.floor(total * clampNumber(
    settings?.startFraction,
    CURRENT_CONTRIBUTION_MIN_COMPACTION_FRACTION,
    CURRENT_CONTRIBUTION_MAX_COMPACTION_FRACTION,
    CURRENT_CONTRIBUTION_COMPACTION_FRACTION,
  )) + 1);
  const firstEvent = requestedReset + measurementWindowSteps - 1;
  return firstEvent <= deadline ? firstEvent : 0;
}

function currentContributionCompactionResetStep(steps, settings) {
  const target = currentContributionCompactionStep(steps, settings);
  if (target <= 0) return 0;
  const window = Math.max(1, Math.round(settings?.measurementWindowSteps || 1));
  return Math.max(1, target - window + 1);
}

function currentContributionCompactionDeadline(steps, settings) {
  const total = Math.max(1, Math.round(steps));
  const structuralDeadline = settings?.opaqueLayered
    ? opaquePaintVisibilityCompactionStep(total, settings?.opaquePaintSettleFraction)
    : total - 1;
  const growthEnd = experimentalGrowthSteps(total, settings?.growthApplyUntilFraction);
  // The final growth event closes active cardinality. Do not physically remove
  // splats at or after that milestone; the remaining tail optimizes a fixed set.
  return growthEnd > 0
    ? Math.max(0, Math.min(structuralDeadline, growthEnd - 1))
    : structuralDeadline;
}

function currentContributionCompactionSchedule(step, steps, settings) {
  const current = Math.max(1, Math.round(Number(step) || 1));
  const firstEvent = currentContributionCompactionStep(steps, settings);
  const firstReset = currentContributionCompactionResetStep(steps, settings);
  const window = Math.max(1, Math.round(settings?.measurementWindowSteps || 1));
  const interval = Math.max(
    window,
    Math.round(settings?.intervalSteps || CURRENT_CONTRIBUTION_COMPACTION_INTERVAL),
  );
  const deadline = currentContributionCompactionDeadline(steps, settings);
  const resetOffset = current - firstReset;
  const eventOffset = current - firstEvent;
  return {
    enabled: Boolean(settings?.enabled) && firstEvent > 0,
    firstEvent,
    firstReset,
    interval,
    window,
    deadline,
    resetDue: Boolean(settings?.enabled) && firstReset > 0 && current <= deadline &&
      resetOffset >= 0 && resetOffset % interval === 0,
    compactionDue: Boolean(settings?.enabled) && firstEvent > 0 && current <= deadline &&
      eventOffset >= 0 && eventOffset % interval === 0,
  };
}

function currentContributionCompactionResetDue(step, steps, settings) {
  return currentContributionCompactionSchedule(step, steps, settings).resetDue;
}

function currentContributionCompactionDue(step, steps, settings) {
  return currentContributionCompactionSchedule(step, steps, settings).compactionDue;
}

function opaquePaintDetailRecoveryDue(step, steps, interval = OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL) {
  const cadence = Math.max(2, Math.round(interval || OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL));
  const offset = Math.floor(cadence / 2);
  // Keep ownership moves on their established cadence and leave a short
  // settling window before the final result is presented.
  return step > 0 && step < Math.max(1, steps - 16) && step % cadence === offset;
}

function performanceProfileSchedule(steps) {
  const total = Math.max(1, Math.round(steps));
  const b1 = experimentalCoarseSteps(total, phase33Variants().coarseSteps);
  const b2 = experimentalDensifySteps(total);
  const horizon = experimentalGrowthSteps(total);
  const schedule = new Map();
  for (const [label, rawStep] of [
    ["B1-100", b1 - 100],
    ["B1", b1],
    ["B1+1", b1 + 1],
    ["B1+100", b1 + 100],
    ["B2", b2],
    ["H", horizon],
    ["S", total],
  ]) {
    const step = Math.max(1, Math.min(total, Math.round(rawStep)));
    const labels = schedule.get(step) || [];
    labels.push(label);
    schedule.set(step, labels);
  }
  return schedule;
}

function performanceProfileLabels(step, steps) {
  return performanceProfileSchedule(steps).get(Math.round(step)) || [];
}

function summarizeTrainingScheduling(profileSamples = [], traceSamples = []) {
  const gpuPasses = {};
  const profileIo = {
    queue_wait_count: 0,
    queue_wait_wall_ms: 0,
    readback_count: 0,
    readback_bytes: 0,
    readback_wall_ms: 0,
  };
  for (const sample of profileSamples) {
    for (const [name, milliseconds] of Object.entries(sample.stages_ms || {})) {
      if (!Number.isFinite(milliseconds)) continue;
      const entry = gpuPasses[name] || { samples: 0, total_ms: 0, mean_ms: 0, max_ms: 0 };
      entry.samples += 1;
      entry.total_ms += milliseconds;
      entry.max_ms = Math.max(entry.max_ms, milliseconds);
      entry.mean_ms = entry.total_ms / entry.samples;
      gpuPasses[name] = entry;
    }
    profileIo.queue_wait_count += Math.max(0, Number(sample.queue_wait_count) || 0);
    profileIo.queue_wait_wall_ms += Math.max(0, Number(sample.queue_wait_wall_ms) || 0);
    profileIo.readback_count += Math.max(0, Number(sample.readback_count) || 0);
    profileIo.readback_bytes += Math.max(0, Number(sample.readback_bytes) || 0);
    profileIo.readback_wall_ms += Math.max(0, Number(sample.readback_wall_ms) || 0);
  }
  const runtime = {};
  for (const sample of traceSamples) {
    const phase = sample.phase || "unknown";
    const entry = runtime[phase] || {
      samples: 0,
      steps: 0,
      train_ms: 0,
      density_ms: 0,
      relocation_ms: 0,
      presentation_ms: 0,
      wall_ms: 0,
    };
    entry.samples += 1;
    entry.steps += Math.max(0, Number(sample.interval_steps) || 0);
    entry.train_ms += Math.max(0, Number(sample.train_ms) || 0);
    entry.density_ms += Math.max(0, Number(sample.density_ms) || 0);
    entry.relocation_ms += Math.max(0, Number(sample.relocation_ms) || 0);
    entry.presentation_ms += Math.max(0, Number(sample.presentation_ms) || 0);
    entry.wall_ms += Math.max(0, Number(sample.interval_ms) || 0);
    runtime[phase] = entry;
  }
  for (const entry of Object.values(runtime)) {
    entry.iterations_per_second = entry.steps > 0 && entry.wall_ms > 0
      ? (entry.steps * 1000) / entry.wall_ms
      : 0;
  }
  return {
    timestamp_samples: profileSamples.length,
    timestamp_gpu_passes_ms: gpuPasses,
    timestamp_profile_io: profileIo,
    runtime_wall_by_phase_ms: runtime,
  };
}

function experimentalAdcInterval(steps) {
  if (steps >= 3000) return EXPERIMENTAL_ADC_INTERVAL_FOR_7000;
  return Math.max(300, Math.round(steps / 4));
}

function experimentalAdcWindow(steps, interval, densifyInterval, minimumEvents) {
  const proportional = Math.max(24, Math.min(160, Math.round(steps * 0.02)));
  const eventAligned = Math.max(1, Math.round(densifyInterval)) * Math.max(1, Math.round(minimumEvents));
  return Math.min(Math.max(1, Math.round(interval)), Math.max(proportional, eventAligned));
}

function experimentalSchedule(steps) {
  const densityHorizon = experimentalDensifySteps(steps);
  const warmup = densifyWarmupSteps(densityHorizon);
  const phase38 = phase38Variants();
  const adcInterval = phase38.adcSplitInterval || experimentalAdcInterval(densityHorizon);
  const adcWindow = experimentalAdcWindow(
    densityHorizon,
    adcInterval,
    phase39Variants().densifyInterval,
    phase38.adcWindowEvents,
  );
  const resetInterval = phase38.adcResetInterval || EXPERIMENTAL_ADC_INTERVAL_FOR_7000;
  const resetHorizon = experimentalGrowthSteps(steps);
  return { steps, densityHorizon, warmup, adcInterval, adcWindow, resetInterval, resetHorizon };
}

function densityGpuConfig({ image, count, targetCount, step, steps, layout, maxAnisotropy, capacity, mode }) {
  const schedule = experimentalSchedule(steps);
  const variants = phase33Variants();
  const phase37 = phase37Variants();
  const phase38 = phase38Variants();
  const phase39 = phase39Variants();
  const phase45 = phase45Variants();
  const layerEfficiency = layerEfficiencyVariants();
  const productOverdensityCorrection = mode === 3 && phase39.midTrainingOverdensityCorrection;
  const phase45DonorActive = productOverdensityCorrection || (
    phase45.donorEligibility && (!phase45.firstResetOnly || step <= schedule.resetInterval)
  );
  const detail = selectedLearningRates();
  const trainState = state.webgpu.renderer?.trainState;
  const trainingStage = curriculumTrainingStage(step, steps, variants, trainState?.coarseImage, trainState?.midImage);
  const stageImage = trainingStage === "coarse"
    ? trainState?.coarseImage
    : trainingStage === "mid"
      ? trainState?.midImage
      : image;
  const stageMinScale = stageMinimumScale(
    stageImage || image,
    state.metrics?.initial_splats || count,
    trainingStage,
    variants.stageMinScaleRatio,
  );
  const baseScaleFloorRatio = stageBaseScaleFloorRatio(trainingStage);
  const shaderStepLimit = mode === 3 ? schedule.densityHorizon : steps;
  const config = new Float32Array(TRAIN_CONFIG_FLOATS);
  config.set([
      image.width,
      image.height,
      count,
      targetCount,
      step,
      shaderStepLimit,
      layout.cols,
      layout.rows,
      layout.baseScale,
      maxAnisotropy,
      capacity,
      mode,
      schedule.adcInterval,
      schedule.adcWindow,
      schedule.warmup,
      schedule.densityHorizon,
      variants.importanceRecycle ? 1 : 0,
      variants.adcEligibility ? 1 : 0,
      variants.coverageDensity ? 1 : 0,
      variants.structureTensor ? 1 : 0,
      variants.importanceEma,
      variants.coverageTarget,
      variants.coverageLossWeight,
      variants.coverageDensityStrength,
      phase37.absGradient ? 1 : 0,
      phase37.gradientCoherence ? 1 : 0,
      phase37.edgeErrorDensity ? 1 : 0,
      phase37.significanceRecycle ? 1 : 0,
      phase37.structureAnisotropy ? 1 : 0,
      detail.adaptiveDetail ? 1 : 0,
      detail.maxAnisotropy,
      detail.detailCoherence,
      0,
      phase45.seedOffset,
      phase39.growthSignalThreshold,
      els.trainLayerOrder.checked ? 1 : 0,
      mode === 3 ? step + 1 : step,
      phase39.tiltRobustSplit ? 1 : 0,
      phase39.tiltSplitAngleDegrees * Math.PI / 180,
      PLY_LAYER_DEPTH_SPAN,
      DEFAULT_TILT_SPLIT_COLOR_THRESHOLD,
      DEFAULT_TILT_SPLIT_SHRINK,
      phase39.singleSourceClaim ? 1 : 0,
      phase39.qaGrowthComparisons ? 1 : 0,
      phase45.telemetry || phase45DonorActive || phase45.recipientScore ? 1 : 0,
      phase45DonorActive ? 1 : 0,
      phase45.donorQuantile,
      phase45.recipientScore ? phase45.recipientStrength : 0,
      phase39.growthSignalThreshold,
      0,
      0,
      0,
      layout.baseScaleX,
      layout.baseScaleY,
      detail.boundarySigma,
      detail.surfaceAnisotropy,
      detail.maxPlanarScale,
      phase39.adcSplitEnabled ? 1 : 0,
      phase39.adcRecycleEnabled ? 1 : 0,
      phase39.mcmcRelocationEnabled ? 1 : 0,
      stageMinScale,
      baseScaleFloorRatio,
      layerEfficiency.deepRelocation ? 1 : 0,
      layerEfficiency.deepFraction,
      layerEfficiency.influenceThreshold,
      state.params?.opaqueLayered ? 1 : 0,
    ], 0);
  config[49] = experimentalDensifySteps(steps);
  configurePaintKernel(config, state.params);
  // Density-only algorithm flags. Training config reuses these slots under a
  // separate shader contract and rewrites them before optimizer dispatch.
  config[66] = state.params?.virtualCameraSamplingEnabled ? 1 : 0;
  config[67] = state.params?.virtualDepthEnabled ? 1 : 0;
  config[68] = phase39.structureGuidedAllocation ? 1 : 0;
  config[69] = phase39.structureRegionGrid;
  if (productOverdensityCorrection) {
    // The region quantile is already the move cap. Do not apply a second
    // random ADC sampling gate to the independently safe donor cohort.
    config[46] = phase39.overdensityDonorFraction;
    config[47] = 1;
    config[50] = 1;
    config[51] = 1;
  }
  return { schedule, config };
}

function experimentalDensityPhase(step, steps) {
  const schedule = experimentalSchedule(steps);
  if (step > schedule.densityHorizon) return "settle";
  if (step <= schedule.warmup) return "warmup";
  return "growth";
}

function splatTargetForGrowth(currentCount, finalCount, growthFraction = DEFAULT_GROWTH_FRACTION) {
  if (finalCount <= currentCount) return currentCount;
  const added = Math.max(1, Math.ceil(currentCount * Math.max(0.001, growthFraction)));
  return normalizeActiveSplatCount(Math.min(finalCount, currentCount + added), currentCount);
}

function remainingGrowthEventCount(step, growthEnd, densifyInterval) {
  const current = Math.max(1, Math.round(step));
  const end = Math.max(current, Math.round(growthEnd));
  const interval = Math.max(1, Math.round(densifyInterval));
  if (current >= end) return 1;
  const regularAfterCurrent = Math.max(0, Math.floor((end - 1) / interval) - Math.floor(current / interval));
  return 1 + regularAfterCurrent + 1; // current event + regular events + terminal H event
}

function growthSchedulePlan({
  step,
  steps,
  initialCount,
  currentCount,
  finalCount,
  growthFraction,
  growthApplyUntilFraction = phase39Variants().growthApplyUntilFraction,
  densifyInterval,
  stageAware,
  stageGrowthShares = Object.fromEntries(Object.entries(DEFAULT_STAGE_GROWTH_SHARES).map(([key, value]) => [key, value / 100])),
}) {
  const normalTarget = splatTargetForGrowth(currentCount, finalCount, growthFraction);
  const normalIncrement = Math.max(0, normalTarget - currentCount);
  const densityEnd = experimentalDensifySteps(steps);
  const growthEnd = experimentalGrowthSteps(steps, growthApplyUntilFraction);
  if (!stageAware) {
    const terminalClosure = growthEnd > 0 && Math.round(step) === growthEnd;
    return {
      mode: "threshold-percentage-target-closure",
      desiredCount: terminalClosure ? finalCount : normalTarget,
      previousDesired: currentCount,
      normalIncrement,
      catchUpLimit: finalCount,
      requestedCount: terminalClosure ? finalCount : normalTarget,
      densityEnd,
      growthEnd,
      remainingGrowthEvents: terminalClosure ? 1 : null,
      scheduleDebt: 0,
      repairIncrement: 0,
      terminalClosure,
    };
  }

  const warmup = densifyWarmupSteps(densityEnd);
  const p1End = experimentalCoarseSteps(steps);
  const range = Math.max(0, finalCount - initialCount);
  const p1Share = Math.max(0, Math.min(1, Number(stageGrowthShares.p1) || 0));
  const p2Share = Math.max(0, Math.min(1 - p1Share, Number(stageGrowthShares.p2) || 0));
  const p2Cumulative = p1Share + p2Share;
  const desiredAt = (targetStep) => {
    const segmentProgress = targetStep <= p1End
      ? Math.max(0, Math.min(1, (targetStep - warmup) / Math.max(1, p1End - warmup))) * p1Share
      : targetStep <= densityEnd
        ? p1Share + Math.max(0, Math.min(1, (targetStep - p1End) / Math.max(1, densityEnd - p1End))) * p2Share
        : p2Cumulative + Math.max(0, Math.min(1, (targetStep - densityEnd) / Math.max(1, growthEnd - densityEnd))) * (1 - p2Cumulative);
    return Math.min(finalCount, initialCount + Math.round(range * segmentProgress));
  };
  const terminalClosure = growthEnd > 0 && Math.round(step) === growthEnd;
  const desiredCount = terminalClosure ? finalCount : desiredAt(step);
  const previousDesired = desiredAt(Math.max(warmup, step - Math.max(1, densifyInterval)));
  const stageIncrement = Math.max(0, desiredCount - previousDesired);
  const remainingGrowthEvents = remainingGrowthEventCount(step, growthEnd, densifyInterval);
  const scheduleDebt = Math.max(0, previousDesired - currentCount);
  const repairIncrement = Math.ceil(scheduleDebt / remainingGrowthEvents);
  const catchUpLimit = Math.min(finalCount, currentCount + stageIncrement + repairIncrement);
  const requestedCount = normalizeActiveSplatCount(
    Math.max(currentCount, Math.min(desiredCount, catchUpLimit)),
    currentCount,
  );
  return {
    mode: "stage-aware-percentage-cap",
    desiredCount,
    previousDesired,
    normalIncrement: stageIncrement,
    catchUpLimit,
    requestedCount,
    densityEnd,
    growthEnd,
    remainingGrowthEvents,
    scheduleDebt,
    repairIncrement,
    terminalClosure,
    stageGrowthShares: { p1: p1Share, p2: p2Share, p3: Math.max(0, 1 - p2Cumulative) },
  };
}

function referenceGrowthTargets(currentCount, finalCount, eligibleSourceCount, enabled) {
  if (!enabled) return null;
  const capReached = currentCount >= finalCount;
  const eligible = Number.isFinite(eligibleSourceCount) ? eligibleSourceCount : null;
  return {
    reference_only: true,
    changes_requested_count: false,
    status: capReached ? "cap-reached" : eligible === null ? "not-sampled" : "sampled",
    default_like_eligible_count: eligible,
    default_like_target: capReached ? finalCount : eligible === null ? null : Math.min(finalCount, currentCount + eligible),
    mcmc_like_fraction: 0.05,
    mcmc_like_target: capReached
      ? finalCount
      : Math.min(finalCount, currentCount + Math.max(1, Math.ceil(currentCount * 0.05))),
  };
}

function growParamPlaceholders(params, targetCount) {
  if (targetCount <= params.count) return params;
  const oldCount = params.count;
  const xy = new Float32Array(targetCount * 2);
  const scale = new Float32Array(targetCount * 2);
  const rgb = new Float32Array(targetCount * 3);
  const opacity = new Float32Array(targetCount);
  const theta = new Float32Array(targetCount);
  const depthOrder = new Float32Array(targetCount);
  const virtualDepth = new Float32Array(targetCount);
  const brushTaper = new Float32Array(targetCount);
  const detailTags = new Float32Array(targetCount);
  xy.set(params.xy);
  scale.set(params.scale);
  rgb.set(params.rgb);
  opacity.set(params.opacity);
  theta.set(params.theta);
  depthOrder.set(params.depthOrder || initialDepthOrder(oldCount));
  virtualDepth.set(params.virtualDepth || new Float32Array(oldCount));
  brushTaper.set(
    params.brushTaper || new Float32Array(oldCount).fill(DEFAULT_LAYERED_BRUSH_TAPER),
  );
  detailTags.set(params.detailTags || new Float32Array(oldCount).fill(1));
  for (let i = oldCount; i < targetCount; i += 1) {
    const source = i % oldCount;
    xy[i * 2] = params.xy[source * 2];
    xy[i * 2 + 1] = params.xy[source * 2 + 1];
    scale[i * 2] = params.scale[source * 2];
    scale[i * 2 + 1] = params.scale[source * 2 + 1];
    rgb[i * 3] = params.rgb[source * 3];
    rgb[i * 3 + 1] = params.rgb[source * 3 + 1];
    rgb[i * 3 + 2] = params.rgb[source * 3 + 2];
    opacity[i] = params.opacity[source];
    theta[i] = params.theta[source];
    depthOrder[i] = params.depthOrder?.[source] ?? (1 - source / Math.max(1, oldCount - 1));
    virtualDepth[i] = params.virtualDepth?.[source] ?? 0;
    brushTaper[i] = params.brushTaper?.[source] ?? DEFAULT_LAYERED_BRUSH_TAPER;
    detailTags[i] = params.detailTags?.[source] ?? 1;
  }
  return {
    kernelShape: normalizedKernelShape(params.kernelShape),
    rectangleTopRatio: clampNumber(
      params.rectangleTopRatio,
      MIN_RECTANGLE_TOP_RATIO,
      MAX_RECTANGLE_TOP_RATIO,
      DEFAULT_RECTANGLE_TOP_RATIO,
    ),
    rectangleTopRatioMax: clampNumber(
      params.rectangleTopRatioMax,
      params.rectangleTopRatio,
      MAX_RECTANGLE_TOP_RATIO,
      DEFAULT_RECTANGLE_TOP_RATIO_MAX,
    ),
    rectangleOpacityGradientMin: clampNumber(
      params.rectangleOpacityGradientMin,
      0,
      1,
      1,
    ),
    rectangleOpacityGradientMax: clampNumber(
      params.rectangleOpacityGradientMax,
      0,
      1,
      1,
    ),
    rectangleCenterOpacityGradientMin: clampNumber(
      params.rectangleCenterOpacityGradientMin,
      0,
      1,
      1,
    ),
    rectangleCenterOpacityGradientMax: clampNumber(
      params.rectangleCenterOpacityGradientMax,
      clampNumber(params.rectangleCenterOpacityGradientMin, 0, 1, 1),
      1,
      1,
    ),
    rectangleMinAspectRatio: clampNumber(
      params.rectangleMinAspectRatio,
      MIN_RECTANGLE_ASPECT_RATIO,
      params.rectangleMaxAspectRatio,
      DEFAULT_RECTANGLE_MIN_ASPECT_RATIO,
    ),
    rectangleMaxAspectRatio: clampNumber(
      params.rectangleMaxAspectRatio,
      MIN_RECTANGLE_ASPECT_RATIO,
      MAX_RECTANGLE_ASPECT_RATIO,
      DEFAULT_RECTANGLE_ASPECT_RATIO,
    ),
    rectangleOrientation: normalizedRectangleOrientation(params.rectangleOrientation),
    rectanglePreserveArea:
      params.rectanglePreserveArea ?? DEFAULT_RECTANGLE_PRESERVE_AREA,
    rectangleEdgeDirectedTaper:
      params.rectangleEdgeDirectedTaper ?? DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER,
    rectangleStructureAwareRatio:
      params.rectangleStructureAwareRatio ?? DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO,
    rectangleAsymmetricSoftness:
      params.rectangleAsymmetricSoftness ?? DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS,
    illustrativeOilVersion: Math.max(0, Math.round(Number(params.illustrativeOilVersion) || 0)),
    brushLocalColorFlowEnabled: Boolean(params.brushLocalColorFlowEnabled),
    brushStrokePersistenceEnabled: Boolean(params.brushStrokePersistenceEnabled),
    brushRibbonAspectFloor: clampNumber(
      params.brushRibbonAspectFloor,
      1,
      LIMITS.maxAnisotropyMax,
      BRUSH_STROKE_PERSISTENCE_RIBBON_MIN_RATIO,
    ),
    brushAccentAspectFloor: clampNumber(
      params.brushAccentAspectFloor,
      1,
      LIMITS.maxAnisotropyMax,
      BRUSH_STROKE_PERSISTENCE_ACCENT_MIN_RATIO,
    ),
    surfaceLayerPriorEnabled: Boolean(params.surfaceLayerPriorEnabled),
    surfaceLayerPriorColorAwarePromotion:
      params.surfaceLayerPriorColorAwarePromotion !== false,
    trainLayerColorGuardEnabled: Boolean(params.trainLayerColorGuardEnabled),
    surfaceLayerPriorLayers: Math.round(clampNumber(
      params.surfaceLayerPriorLayers,
      MIN_DISCRETE_LAYER_COUNT,
      MAX_DISCRETE_LAYER_COUNT,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
    )),
    surfaceLayerPriorP1Interval: Math.max(0, Math.round(params.surfaceLayerPriorP1Interval || 0)),
    surfaceLayerPriorP2Interval: Math.max(0, Math.round(params.surfaceLayerPriorP2Interval || 0)),
    surfaceLayerPriorP3Interval: Math.max(0, Math.round(params.surfaceLayerPriorP3Interval || 0)),
    surfaceLayerPriorUntilFraction: clampNumber(
      params.surfaceLayerPriorUntilFraction,
      0,
      1,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_UNTIL,
    ),
    harmfulRectangleParentSplitEnabled: Boolean(params.harmfulRectangleParentSplitEnabled),
    harmfulRectangleParentSplitTransitionOnly: Boolean(
      params.harmfulRectangleParentSplitTransitionOnly,
    ),
    frontSplitChildrenEnabled: Boolean(params.frontSplitChildrenEnabled),
    brushOpacityGradientEnabled: Boolean(params.brushOpacityGradientEnabled),
    brushOpacityGradientStart: clampNumber(params.brushOpacityGradientStart, 0, 1, 0),
    brushOpacityGradientEnd: clampNumber(params.brushOpacityGradientEnd, 0, 1, 1),
    brushCenterOpacityGradientMin: clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1),
    brushCenterOpacityGradientMax: clampNumber(
      params.brushCenterOpacityGradientMax,
      clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1),
      1,
      1,
    ),
    brushMinAspectRatio: clampNumber(params.brushMinAspectRatio, LIMITS.maxAnisotropyMin, params.brushMaxAspectRatio, LIMITS.maxAnisotropyMin),
    brushMaxAspectRatio: clampNumber(params.brushMaxAspectRatio, LIMITS.maxAnisotropyMin, LIMITS.maxAnisotropyMax, DEFAULT_MAX_ANISOTROPY),
    brushWidthTaperEnabled: Boolean(params.brushWidthTaperEnabled),
    brushWidthTaperStart: clampNumber(params.brushWidthTaperStart, 0, 1, 1),
    brushWidthTaperEnd: clampNumber(params.brushWidthTaperEnd, 0, 1, 0),
    monochromeUnderpaintingEnabled: Boolean(params.monochromeUnderpaintingEnabled),
    colorFinishStartPercent: clampNumber(
      params.colorFinishStartPercent,
      MIN_COLOR_FINISH_START_PERCENT,
      MAX_COLOR_FINISH_START_PERCENT,
      DEFAULT_COLOR_FINISH_START_PERCENT,
    ),
    colorFinishStartStep: Math.max(
      0,
      Math.round(Number(params.colorFinishStartStep) || 0),
    ),
    currentVisibilityChildPolicyEnabled: params.currentVisibilityChildPolicyEnabled !== false,
    currentVisibilityCompactionEnabled: params.currentVisibilityCompactionEnabled !== false,
    illustrativeOilFamilyStats: params.illustrativeOilFamilyStats
      ? structuredClone(params.illustrativeOilFamilyStats)
      : null,
    opaqueLayered: Boolean(params.opaqueLayered),
    minimumOpacityEnabled: Boolean(params.minimumOpacityEnabled),
    minimumOpacity: clampNumber(
      params.minimumOpacity,
      MIN_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
    ),
    maximumOpacity: clampNumber(params.maximumOpacity, params.minimumOpacity, MAX_LEARNED_PAINT_OPACITY, MAX_LEARNED_PAINT_OPACITY),
    xy, scale, rgb, opacity, theta, depthOrder, virtualDepth, brushTaper, detailTags, count: targetCount,
    rows: params.rows, cols: params.cols, bg: params.bg,
    boundarySigma: params.boundarySigma,
    layerOrderEnabled: Boolean(params.layerOrderEnabled),
    layerAwareAccumulationEnabled: Boolean(params.layerAwareAccumulationEnabled),
    discreteLayersEnabled: Boolean(params.discreteLayersEnabled),
    discreteLayerCount: params.discreteLayerCount,
    discreteLayerMoveRadius: params.discreteLayerMoveRadius,
    virtualDepthEnabled: Boolean(params.virtualDepthEnabled),
    virtualDepthThickness: Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS,
    virtualDepthSoftConstraintEnabled: params.virtualDepthSoftConstraintEnabled !== false,
    virtualDepthPriorDelta: Number(params.virtualDepthPriorDelta) || DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA,
  };
}

function resizeParamPlaceholders(params, targetCount) {
  if (targetCount >= params.count) return growParamPlaceholders(params, targetCount);
  const slice = (values, length) => values?.slice?.(0, length) || null;
  return {
    ...params,
    count: targetCount,
    xy: slice(params.xy, targetCount * 2),
    scale: slice(params.scale, targetCount * 2),
    rgb: slice(params.rgb, targetCount * 3),
    opacity: slice(params.opacity, targetCount),
    theta: slice(params.theta, targetCount),
    depthOrder: slice(params.depthOrder, targetCount),
    virtualDepth: slice(params.virtualDepth, targetCount),
    brushTaper: slice(params.brushTaper, targetCount),
    detailTags: slice(params.detailTags, targetCount),
  };
}

function compactSplatParams(params, keepIndices) {
  const count = keepIndices.length;
  const copy = (source, stride, fallback = 0) => {
    const output = new Float32Array(count * stride);
    for (let target = 0; target < count; target += 1) {
      const sourceIndex = keepIndices[target];
      for (let component = 0; component < stride; component += 1) {
        output[target * stride + component] = source?.[sourceIndex * stride + component] ?? fallback;
      }
    }
    return output;
  };
  return {
    ...params,
    count,
    xy: copy(params.xy, 2),
    scale: copy(params.scale, 2),
    rgb: copy(params.rgb, 3),
    opacity: copy(params.opacity, 1),
    theta: copy(params.theta, 1),
    depthOrder: copy(params.depthOrder, 1),
    virtualDepth: copy(params.virtualDepth, 1),
    brushTaper: copy(params.brushTaper, 1, DEFAULT_LAYERED_BRUSH_TAPER),
    detailTags: copy(params.detailTags, 1, 1),
  };
}

const FOOTPRINT_COLOR_SAMPLES = [
  [0, 0, 1],
  [-1, 0, Math.exp(-0.5)],
  [1, 0, Math.exp(-0.5)],
  [0, -1, Math.exp(-0.5)],
  [0, 1, Math.exp(-0.5)],
  [-1, -1, Math.exp(-1)],
  [1, -1, Math.exp(-1)],
  [-1, 1, Math.exp(-1)],
  [1, 1, Math.exp(-1)],
];

function sampleImageRgbBilinear(image, x, y, out) {
  const fx = Math.max(0, Math.min(image.width - 1, ((x + 1) * 0.5) * (image.width - 1)));
  const fy = Math.max(0, Math.min(image.height - 1, ((y + 1) * 0.5) * (image.height - 1)));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const row = image.width * 3;
  const i00 = y0 * row + x0 * 3;
  const i10 = y0 * row + x1 * 3;
  const i01 = y1 * row + x0 * 3;
  const i11 = y1 * row + x1 * 3;
  for (let channel = 0; channel < 3; channel += 1) {
    const top = image.rgb[i00 + channel] * (1 - tx) + image.rgb[i10 + channel] * tx;
    const bottom = image.rgb[i01 + channel] * (1 - tx) + image.rgb[i11 + channel] * tx;
    out[channel] = top * (1 - ty) + bottom * ty;
  }
}

function footprintWeightedTargetColor(image, params, index, out = new Float32Array(3)) {
  out.fill(0);
  if (!image?.rgb || !params?.xy || !params?.scale) return out;
  const centerX = Number(params.xy[index * 2]);
  const centerY = Number(params.xy[index * 2 + 1]);
  const sx = Math.max(1e-6, Number(params.scale[index * 2]));
  const sy = Math.max(1e-6, Number(params.scale[index * 2 + 1]));
  const theta = Number(params.theta?.[index]) || 0;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);
  const sample = new Float32Array(3);
  const target = new Float64Array(3);
  let weightSum = 0;
  for (const [localX, localY, weight] of FOOTPRINT_COLOR_SAMPLES) {
    const x = centerX + localX * sx * cosTheta - localY * sy * sinTheta;
    const y = centerY + localX * sx * sinTheta + localY * sy * cosTheta;
    sampleImageRgbBilinear(image, x, y, sample);
    for (let channel = 0; channel < 3; channel += 1) target[channel] += sample[channel] * weight;
    weightSum += weight;
  }
  for (let channel = 0; channel < 3; channel += 1) out[channel] = target[channel] / Math.max(weightSum, 1e-8);
  return out;
}

function hardZeroContributionPlan(
  params,
  importanceData,
  maxPruneFraction = OPAQUE_PAINT_HARD_ZERO_MAX_FRACTION,
) {
  const count = params?.count || 0;
  if (
    count < MIN_COMPACTION_SPLATS ||
    !importanceData?.observedCoverage ||
    !importanceData?.integratedInfluence
  ) {
    return { applied: false, reason: "insufficient-current-visibility", before: count, after: count, removed: 0 };
  }
  // v1 uses its 10% default. v2 passes its user-selected total removal cap;
  // both remain bounded by the active-population floor below.
  const boundedFraction = Math.max(0, Math.min(1, maxPruneFraction));
  const maxRemove = Math.max(0, Math.min(
    Math.floor(count * boundedFraction),
    count - MIN_COMPACTION_SPLATS,
  ));
  const depthBinCount = 4096;
  const depthBins = new Uint32Array(depthBinCount);
  const removalMask = new Uint8Array(count);
  let candidateCount = 0;
  for (let index = 0; index < count; index += 1) {
    const observed = Math.max(0, Number(importanceData.observedCoverage[index]) || 0);
    const integratedInfluence = Math.max(0, Number(importanceData.integratedInfluence[index]) || 0);
    if (
      observed <= OPAQUE_PAINT_HARD_ZERO_EPSILON &&
      integratedInfluence <= OPAQUE_PAINT_HARD_ZERO_EPSILON
    ) {
      candidateCount += 1;
      const depth = Math.max(0, Math.min(1, Number(params.depthOrder?.[index]) || 0));
      depthBins[Math.min(depthBinCount - 1, Math.floor(depth * depthBinCount))] += 1;
    }
  }
  if (!candidateCount) {
    return { applied: false, reason: "no-current-hard-zero", before: count, after: count, removed: 0 };
  }
  const selectedCount = Math.min(candidateCount, maxRemove);
  if (selectedCount <= 0) {
    return { applied: false, reason: "minimum-count-guard", before: count, after: count, removed: 0 };
  }
  let thresholdBin = 0;
  let selectedBeforeThreshold = 0;
  while (
    thresholdBin < depthBinCount - 1 &&
    selectedBeforeThreshold + depthBins[thresholdBin] < selectedCount
  ) {
    selectedBeforeThreshold += depthBins[thresholdBin];
    thresholdBin += 1;
  }
  let thresholdQuota = selectedCount - selectedBeforeThreshold;
  for (let index = 0; index < count; index += 1) {
    const observed = Math.max(0, Number(importanceData.observedCoverage[index]) || 0);
    const integratedInfluence = Math.max(0, Number(importanceData.integratedInfluence[index]) || 0);
    if (
      observed > OPAQUE_PAINT_HARD_ZERO_EPSILON ||
      integratedInfluence > OPAQUE_PAINT_HARD_ZERO_EPSILON
    ) continue;
    const depth = Math.max(0, Math.min(1, Number(params.depthOrder?.[index]) || 0));
    const bin = Math.min(depthBinCount - 1, Math.floor(depth * depthBinCount));
    if (bin < thresholdBin || (bin === thresholdBin && thresholdQuota-- > 0)) removalMask[index] = 1;
  }
  const keepIndices = new Uint32Array(count - selectedCount);
  let keepIndex = 0;
  for (let index = 0; index < count; index += 1) {
    if (!removalMask[index]) keepIndices[keepIndex++] = index;
  }
  return {
    applied: true,
    reason: "current-hard-zero",
    before: count,
    after: keepIndices.length,
    removed: selectedCount,
    removed_ratio: selectedCount / count,
    hard_zero_candidates: candidateCount,
    hard_zero_candidate_ratio: candidateCount / count,
    max_prune_fraction: boundedFraction,
    epsilon: OPAQUE_PAINT_HARD_ZERO_EPSILON,
    keepIndices,
  };
}

function currentContributionCompactionPlan(
  params,
  importanceData,
  {
    maxRemovalFraction = CURRENT_CONTRIBUTION_MAX_FRACTION,
    nearZeroMaxFraction = CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_FRACTION,
  } = {},
) {
  const count = params?.count || 0;
  if (
    count < MIN_COMPACTION_SPLATS ||
    !importanceData?.observedCoverage ||
    !importanceData?.integratedInfluence
  ) {
    return { applied: false, reason: "insufficient-current-contribution", before: count, after: count, removed: 0 };
  }
  const boundedMaxRemovalFraction = clampNumber(
    maxRemovalFraction,
    0,
    CURRENT_CONTRIBUTION_MAX_FRACTION,
    CURRENT_CONTRIBUTION_MAX_FRACTION,
  );
  const boundedNearZeroMaxFraction = clampNumber(
    nearZeroMaxFraction,
    0,
    boundedMaxRemovalFraction,
    CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_FRACTION,
  );
  const maxRemove = Math.max(0, Math.min(
    Math.floor(count * boundedMaxRemovalFraction),
    count - MIN_COMPACTION_SPLATS,
  ));
  const exactPlan = hardZeroContributionPlan(
    params,
    importanceData,
    boundedMaxRemovalFraction,
  );
  const exactKeep = exactPlan.applied ? exactPlan.keepIndices : null;
  const removeMask = new Uint8Array(count);
  let exactZeroCandidates = exactPlan.hard_zero_candidates || 0;
  let exactZeroRemoved = 0;
  if (exactKeep) {
    removeMask.fill(1);
    for (const index of exactKeep) removeMask[index] = 0;
    exactZeroRemoved = exactPlan.removed;
  }
  const remainingBudget = Math.max(0, maxRemove - exactZeroRemoved);
  const maxNearZero = Math.max(0, Math.min(
    Math.floor(count * boundedNearZeroMaxFraction),
    remainingBudget,
  ));
  let nearZeroCandidates = 0;
  let selectedNearZero = 0;
  let nearZeroDetailProtected = 0;
  for (let index = 0; index < count; index += 1) {
    const observed = Math.max(0, Number(importanceData.observedCoverage[index]) || 0);
    const influence = Math.max(0, Number(importanceData.integratedInfluence[index]) || 0);
    if (removeMask[index]) continue;
    if (
      observed <= CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_COVERAGE &&
      influence <= CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_INFLUENCE
    ) {
      nearZeroCandidates += 1;
      if (Math.floor(Number(params.detailTags?.[index]) || 0) >= 2) {
        nearZeroDetailProtected += 1;
        continue;
      }
      if (selectedNearZero < maxNearZero) {
        selectedNearZero += 1;
        removeMask[index] = 1;
      }
    }
  }
  const requestedRemoved = exactZeroRemoved + selectedNearZero;
  if (!requestedRemoved) {
    return {
      applied: false,
      reason: "no-current-zero-or-near-zero",
      before: count,
      after: count,
      removed: 0,
      exact_zero_candidates: 0,
      near_zero_candidates: nearZeroCandidates,
      near_zero_detail_protected: nearZeroDetailProtected,
    };
  }
  const keepIndices = new Uint32Array(count - requestedRemoved);
  let keepIndex = 0;
  for (let index = 0; index < count; index += 1) {
    if (!removeMask[index]) keepIndices[keepIndex++] = index;
  }
  return {
    applied: requestedRemoved > 0,
    reason: "current-contribution-exact-and-near-zero",
    before: count,
    after: keepIndices.length,
    removed: requestedRemoved,
    removed_ratio: requestedRemoved / count,
    max_remove_fraction: boundedMaxRemovalFraction,
    exact_zero_candidates: exactZeroCandidates,
    exact_zero_removed: exactZeroRemoved,
    near_zero_candidates: nearZeroCandidates,
    near_zero_removed: selectedNearZero,
    near_zero_detail_protected: nearZeroDetailProtected,
    near_zero_max_fraction: boundedNearZeroMaxFraction,
    near_zero_max_coverage: CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_COVERAGE,
    near_zero_max_influence: CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_INFLUENCE,
    epsilon: OPAQUE_PAINT_HARD_ZERO_EPSILON,
    keepIndices,
  };
}

function emptyFusionEvents() {
  return {
    adc_split: 0,
    adc_duplicate: 0,
    adc_recycle: 0,
    mcmc_teleport: 0,
    mcmc_reseed: 0,
    inactive_reused: 0,
    prune: 0,
    opacity_reset: 0,
    importance_protected: 0,
    adc_eligible: 0,
    adc_fallback: 0,
    structure_guided: 0,
    nonfinite_stats: 0,
    adc_low_to_high: 0,
    adc_high_to_low: 0,
    adc_same_band: 0,
    source_claim_conflicts: 0,
    source_claims: 0,
    tilt_risk_candidates: 0,
    tilt_true_splits: 0,
    tilt_opacity_saturations: 0,
    paint_outlier_recycle: 0,
    paint_outlier_recolor: 0,
    paint_outlier_trim: 0,
    surface_layer_candidates: 0,
    surface_layer_promotions: 0,
    harmful_rectangle_candidate_selections: 0,
    harmful_rectangle_front_oversized_selections: 0,
    harmful_rectangle_high_contribution_selections: 0,
    harmful_rectangle_high_deviation_selections: 0,
    harmful_rectangle_parent_replacements: 0,
    harmful_rectangle_children_created: 0,
  };
}

function ssimFromMoments(meanA, meanB, varA, varB, cov) {
  const c1 = 0.01 ** 2;
  const c2 = 0.03 ** 2;
  return ((2 * meanA * meanB + c1) * (2 * cov + c2)) / ((meanA ** 2 + meanB ** 2 + c1) * (varA + varB + c2));
}

function percentileSorted(values, fraction) {
  if (!values.length) return null;
  const position = clampNumber(fraction, 0, 1, 0) * (values.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const mix = position - lower;
  return values[lower] * (1 - mix) + values[upper] * mix;
}

function psnrFromRgbMse(mse) {
  if (!Number.isFinite(mse) || mse < 0) return Number.NaN;
  return 10 * Math.log10(1 / Math.max(PSNR_MSE_FLOOR, mse));
}

function ssimWorkingBufferBytes(image) {
  return image.width * image.height * SSIM_WORKING_BYTES_PER_PIXEL;
}

function summarizeVirtualCameraMetricSet(entries) {
  const finite = entries.filter((entry) => Number.isFinite(entry.metrics?.ssim));
  const values = (key) => finite
    .map((entry) => Number(entry.metrics?.[key]))
    .filter(Number.isFinite);
  const average = (key) => {
    const set = values(key);
    return set.length ? set.reduce((sum, value) => sum + value, 0) / set.length : null;
  };
  const sortedSsim = values("ssim").sort((a, b) => a - b);
  return {
    camera_count: finite.length,
    valid_pixel_count: finite.reduce((sum, entry) => sum + (entry.metrics.valid_pixel_count || 0), 0),
    valid_pixel_ratio_mean: average("valid_pixel_ratio"),
    rgb_ssim_macro: average("ssim"),
    rgb_ssim_p10: percentileSorted(sortedSsim, 0.1),
    rgb_ssim_min: sortedSsim[0] ?? null,
    rgb_l1_macro: average("loss"),
    rgb_mse_macro: average("mse"),
    rgb_psnr_macro: average("psnr"),
    alpha_ssim_macro: average("alphaSsim"),
    alpha_l1_macro: average("alphaL1"),
    coverage_mean: average("coverage_mean"),
    background_exposure_ratio_mean: average("background_exposure_ratio"),
    rendered_mean_srgb_signal: average("rendered_mean_srgb_signal"),
    target_mean_srgb_signal: average("target_mean_srgb_signal"),
    rendered_minus_target_signal: average("rendered_minus_target_signal"),
    rendered_signal_stddev: average("rendered_signal_stddev"),
    target_signal_stddev: average("target_signal_stddev"),
    rendered_mean_srgb_chroma: average("rendered_mean_srgb_chroma"),
    target_mean_srgb_chroma: average("target_mean_srgb_chroma"),
    rendered_minus_target_chroma: average("rendered_minus_target_chroma"),
  };
}

function targetTangentAt(image, x, y) {
  const px = Math.max(1, Math.min(image.width - 2, Math.round((x * 0.5 + 0.5) * (image.width - 1))));
  const py = Math.max(1, Math.min(image.height - 2, Math.round((y * 0.5 + 0.5) * (image.height - 1))));
  const luma = (ix, iy) => {
    const offset = (iy * image.width + ix) * 3;
    return image.rgb[offset] * 0.299 + image.rgb[offset + 1] * 0.587 + image.rgb[offset + 2] * 0.114;
  };
  const gx = 0.5 * (luma(px + 1, py) - luma(px - 1, py));
  const gy = 0.5 * (luma(px, py + 1) - luma(px, py - 1));
  return { angle: Math.atan2(gy, gx) + Math.PI * 0.5, energy: gx * gx + gy * gy };
}

function phase39PixelLengths(width, height, sx, sy, theta) {
  const pixelScale = (angle) => 0.5 * Math.hypot(
    Math.cos(angle) * Math.max(1, width - 1),
    Math.sin(angle) * Math.max(1, height - 1),
  );
  const xLength = sx * pixelScale(theta);
  const yLength = sy * pixelScale(theta + Math.PI * 0.5);
  return {
    xLength,
    yLength,
    major: Math.max(xLength, yLength),
    minor: Math.min(xLength, yLength),
    majorAxis: xLength >= yLength ? "x" : "y",
  };
}

function phase39ContractProbe(width = 1024, height = 512, sx = 0.02, sy = 0.004, theta = Math.PI * 0.25) {
  const lengths = phase39PixelLengths(width, height, sx, sy, theta);
  const maxSearchPx = Math.min(64, Math.max(width, height) * 0.02);
  return {
    ...lengths,
    maxSearchPx,
    maxCellRadius: 8,
    maxBucketsPerSide: 17,
    singleSourceClaim: phase39Variants().singleSourceClaim,
  };
}

function computeThinLineMetrics(image, params) {
  const tags = params.detailTags;
  if (!tags || !params.count) return null;
  const sampling = finalDiagnosticSampling(params.count, MAX_THIN_LINE_DIAGNOSTIC_SAMPLES);
  const points = [];
  const cellSize = 8;
  const cells = new Map();
  const key = (x, y) => `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)}`;
  for (let i = 0; i < params.count; i += sampling.stride) {
    if (tags[i] <= 1.5) continue;
    const sx = params.scale[i * 2];
    const sy = params.scale[i * 2 + 1];
    const theta = params.theta[i];
    const lengths = phase39PixelLengths(image.width, image.height, sx, sy, theta);
    const useX = lengths.majorAxis === "x";
    const angle = theta + (useX ? 0 : Math.PI * 0.5);
    const point = {
      i,
      x: (params.xy[i * 2] * 0.5 + 0.5) * (image.width - 1),
      y: (params.xy[i * 2 + 1] * 0.5 + 0.5) * (image.height - 1),
      angle,
      majorPx: lengths.major,
      minorPx: lengths.minor,
    };
    points.push(point);
    const bucketKey = key(point.x, point.y);
    if (!cells.has(bucketKey)) cells.set(bucketKey, []);
    cells.get(bucketKey).push(point);
  }
  if (!points.length) {
    return {
      detail_count: 0,
      sampled_splats: sampling.sampleCount,
      source_splats: sampling.sourceCount,
      sample_stride: sampling.stride,
      gap_ratio: null,
      isolated_detail_ratio: null,
      off_ridge_streak_ratio: null,
    };
  }
  const hasSupport = (point, side) => {
    const maxSearchPx = Math.min(64, Math.max(image.width, image.height) * 0.02);
    const reach = Math.min(maxSearchPx, Math.max(2, point.majorPx * 1.25));
    const targetX = point.x + Math.cos(point.angle) * reach * side;
    const targetY = point.y + Math.sin(point.angle) * reach * side;
    const radius = Math.min(maxSearchPx, Math.max(3, point.minorPx * 3, point.majorPx * 0.45));
    const cellRadius = Math.min(8, Math.max(1, Math.ceil(radius / cellSize)));
    const cx = Math.floor(targetX / cellSize);
    const cy = Math.floor(targetY / cellSize);
    let checkedCandidates = 0;
    const maxCandidateChecks = 256;
    for (let oy = -cellRadius; oy <= cellRadius; oy += 1) {
      for (let ox = -cellRadius; ox <= cellRadius; ox += 1) {
        for (const candidate of cells.get(`${cx + ox},${cy + oy}`) || []) {
          checkedCandidates += 1;
          if (checkedCandidates > maxCandidateChecks) return false;
          if (candidate.i === point.i || Math.abs(Math.cos(candidate.angle - point.angle)) < 0.8) continue;
          if (Math.hypot(candidate.x - targetX, candidate.y - targetY) <= radius) return true;
        }
      }
    }
    return false;
  };
  let missingSides = 0;
  let isolated = 0;
  let offRidge = 0;
  for (const point of points) {
    const forward = hasSupport(point, 1);
    const backward = hasSupport(point, -1);
    missingSides += Number(!forward) + Number(!backward);
    if (!forward && !backward) isolated += 1;
    const target = targetTangentAt(image, params.xy[point.i * 2], params.xy[point.i * 2 + 1]);
    if (target.energy < 0.0004 || Math.abs(Math.cos(target.angle - point.angle)) < 0.8) offRidge += 1;
  }
  return {
    detail_count: points.length,
    sampled_splats: sampling.sampleCount,
    source_splats: sampling.sourceCount,
    sample_stride: sampling.stride,
    gap_ratio: missingSides / (points.length * 2),
    isolated_detail_ratio: isolated / points.length,
    off_ridge_streak_ratio: offRidge / points.length,
  };
}

function regionalSsimFromTileMetrics(values, width, height, tileSize = 8, columns = 4, rows = 4) {
  const regions = Array.from({ length: columns * rows }, (_, index) => ({
    index,
    column: index % columns,
    row: Math.floor(index / columns),
    count: 0,
    loss: 0,
    renderedY: 0,
    targetY: 0,
    renderedY2: 0,
    targetY2: 0,
    renderedTargetY: 0,
    gradientError: 0,
    targetGradientEnergy: 0,
    gradientCount: 0,
    ssimSum: 0,
  }));
  const tileColumns = Math.ceil(width / tileSize);
  const tileCount = Math.ceil(height / tileSize) * tileColumns;
  for (let tileIndex = 0; tileIndex < tileCount; tileIndex += 1) {
    const source = tileIndex * METRIC_TILE_STRIDE;
    const count = values[source + 7];
    if (count <= 0) continue;
    const tileX = tileIndex % tileColumns;
    const tileY = Math.floor(tileIndex / tileColumns);
    const centerX = Math.min(width - 1, tileX * tileSize + (Math.min(tileSize, width - tileX * tileSize) - 1) * 0.5);
    const centerY = Math.min(height - 1, tileY * tileSize + (Math.min(tileSize, height - tileY * tileSize) - 1) * 0.5);
    const column = Math.min(columns - 1, Math.floor((centerX / Math.max(1, width)) * columns));
    const row = Math.min(rows - 1, Math.floor((centerY / Math.max(1, height)) * rows));
    const region = regions[row * columns + column];
    region.count += count;
    region.loss += values[source];
    region.renderedY += values[source + 1];
    region.targetY += values[source + 2];
    region.renderedY2 += values[source + 3];
    region.targetY2 += values[source + 4];
    region.renderedTargetY += values[source + 5];
    region.gradientError += values[source + 12];
    region.targetGradientEnergy += values[source + 13];
    region.gradientCount += values[source + 14];
    region.ssimSum += values[source + 34];
  }

  const measured = regions.filter((region) => region.count > 0).map((region) => {
    const ssim = region.ssimSum / region.count;
    return {
      index: region.index,
      column: region.column,
      row: region.row,
      bounds: [
        Math.floor((region.column * width) / columns),
        Math.floor((region.row * height) / rows),
        Math.floor(((region.column + 1) * width) / columns),
        Math.floor(((region.row + 1) * height) / rows),
      ],
      pixels: region.count,
      ssim,
      l1: region.loss / region.count,
      gradient_l1: region.gradientError / Math.max(1, region.gradientCount),
      target_gradient_energy: region.targetGradientEnergy / Math.max(1, region.gradientCount),
      gradient_fidelity: 1 - region.gradientError / Math.max(0.000001, region.targetGradientEnergy),
    };
  });
  const sorted = measured.map((region) => region.ssim).sort((a, b) => a - b);
  const worst = measured.reduce((current, region) => (!current || region.ssim < current.ssim ? region : current), null);
  const highFrequencyRegions = measured
    .slice()
    .sort((a, b) => b.target_gradient_energy - a.target_gradient_energy)
    .slice(0, Math.max(1, Math.ceil(measured.length * 0.25)));
  return {
    grid: [columns, rows],
    tile_size: tileSize,
    mean: sorted.reduce((sum, value) => sum + value, 0) / Math.max(1, sorted.length),
    minimum: sorted[0] ?? null,
    p10: percentileSorted(sorted, 0.1),
    median: percentileSorted(sorted, 0.5),
    p90: percentileSorted(sorted, 0.9),
    maximum: sorted[sorted.length - 1] ?? null,
    worst_region: worst,
    high_frequency_regions: highFrequencyRegions,
    regions: measured,
  };
}

async function detectWebGpu() {
  if (!("gpu" in navigator)) {
    state.webgpu = { supported: false, renderer: null, reason: "navigator.gpu unavailable", limits: null, adapterInfo: null };
    els.backendText.textContent = "webgpu unavailable";
    updateMemoryRecommendation();
    log("WebGPU unavailable; training disabled");
    return false;
  }
  try {
    const profileRequested = performanceProfileRequested();
    const requested = await requestWebGpuDevice(navigator.gpu, {
      profileRequested,
      subgroupExactBackward: performanceVariants().subgroupExactBackward,
      requiredStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
      preferredStorageBuffersPerShaderStage: PREFERRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
    });
    const { device } = requested;
    const renderer = new WebGpuPreview(device, els.gpuCanvas, {
      profileRequested,
      timestampQuery: requested.timestampQuery,
      subgroupExactBackward: requested.subgroupExactBackward,
    });
    state.webgpu = {
      supported: true,
      renderer,
      reason: "available",
      limits: requested.limits,
      adapterInfo: requested.adapterInfo,
      adapterFeatures: requested.adapterFeatures,
      profile: requested.profile,
      subgroups: requested.subgroups,
    };
    device.lost.then((info) => handleWebGpuDeviceLoss(renderer, info));
    els.backendText.textContent = "webgpu available";
    updateMemoryRecommendation();
    log("WebGPU available for train and preview");
    return true;
  } catch (error) {
    state.webgpu = { supported: false, renderer: null, reason: error.message, limits: null, adapterInfo: null };
    els.backendText.textContent = "webgpu unavailable";
    updateMemoryRecommendation();
    log(`WebGPU failed; training disabled: ${error.message}`);
    return false;
  }
}

function completedResultStatus() {
  if (!state.params || !state.image || !state.metrics) return "";
  if (!state.metrics.finished_at && !state.metrics.final_cpu_result_ready_at) return "";
  if (state.metrics.safety_stop) return "safety stopped";
  return state.metrics.stopped ? "stopped" : "done";
}

function cancelCompletedResultGpuRecovery() {
  if (state.webGpuRecoveryPending) invalidatePreviewRefresh();
  state.webGpuRecoveryGeneration += 1;
  state.webGpuRecoveryPending = false;
  state.webGpuRecoveryAttempts = 0;
}

async function recoverCompletedResultWebGpu(generation, terminalStatus) {
  const expectedParams = state.params;
  const expectedMetrics = state.metrics;
  const recoveryIsCurrent = () => (
    generation === state.webGpuRecoveryGeneration &&
    state.params === expectedParams &&
    state.metrics === expectedMetrics
  );
  state.webGpuRecoveryPending = true;
  publishLifecycleInteractionState();
  try {
    let recovered = false;
    for (let attempt = 1; attempt <= 2 && recoveryIsCurrent(); attempt += 1) {
      state.webGpuRecoveryAttempts = attempt;
      recovered = await detectWebGpu();
      if (!recoveryIsCurrent()) return;
      if (recovered && state.webgpu.renderer && !state.webgpu.renderer.deviceLost) break;
      await nextFrame();
      if (!recoveryIsCurrent()) return;
    }
    if (!recoveryIsCurrent()) return;
    if (!recovered || !state.webgpu.renderer || state.webgpu.renderer.deviceLost) {
      els.backendText.textContent = "webgpu unavailable";
      setStatus("gpu unavailable");
      setTrainingMessage("Training result preserved, but the GPU preview could not be restored.", "info");
      updateDownloads(false);
      log("GPU preview recovery failed; completed CPU result was preserved");
      return;
    }
    state.previewGeneration += 1;
    state.previewAppliedRevision = 0;
    const recoveryRenderer = state.webgpu.renderer;
    const uploaded = await recoveryRenderer.uploadResultRenderState(expectedParams);
    if (!recoveryIsCurrent()) {
      recoveryRenderer.disposeResultRenderState();
      return;
    }
    if (!uploaded) throw new Error("The preserved result could not be uploaded to the recovered GPU.");
    await refreshOutsidePreview({ recovery: true });
    if (!recoveryIsCurrent()) {
      recoveryRenderer.disposeResultRenderState();
      return;
    }
    state.webGpuRecoveryAttempts = 0;
    setStatus(terminalStatus);
    setTrainingMessage(
      terminalStatus === "stopped" ? "Training stopped. GPU preview restored." : "Training complete. GPU preview restored.",
      terminalStatus === "safety stopped" ? "error" : "success",
    );
    updateDownloads(!state.metrics?.safety_stop);
    if (state.metrics) {
      state.metrics.post_training_gpu_recoveries =
        (state.metrics.post_training_gpu_recoveries || 0) + 1;
    }
    log("GPU preview restored; completed training result preserved");
  } finally {
    if (recoveryIsCurrent()) {
      state.webGpuRecoveryPending = false;
      publishLifecycleInteractionState();
    }
  }
}

function handleWebGpuDeviceLoss(renderer, info = {}) {
  if (state.webgpu.renderer !== renderer) return;
  const terminalStatus = completedResultStatus();
  const reason = `device lost: ${info.reason || "unknown"}`;
  renderer.deviceLost = true;
  state.lastGpuLoss = { reason, at: new Date().toISOString(), after_training: Boolean(terminalStatus) };
  invalidatePreviewRefresh();
  state.splatAdjustmentEpoch += 1;
  window.clearTimeout(state.splatAdjustmentValidationTimer);
  if (state.splatAdjustmentFrame) window.cancelAnimationFrame(state.splatAdjustmentFrame);
  state.splatAdjustmentFrame = 0;
  state.splatAdjustmentValidationVersion = 0;
  invalidateTrainingRun("WebGPU device lost");
  state.paused = false;
  if (!terminalStatus) state.stopRequested = true;
  // Dependency synchronizers consult state.running when controls are enabled.
  // Clear it first so a device loss cannot leave next-run settings disabled.
  state.running = false;
  try {
    renderer.disposeTrainState();
    renderer.disposeResultRenderState();
  } catch {
    // A lost device may reject cleanup; CPU parameters remain authoritative.
  }
  state.webgpu = { supported: false, renderer: null, reason, limits: null, adapterInfo: null };
  els.backendText.textContent = terminalStatus ? "webgpu recovering" : "webgpu unavailable";
  setInputControlsDisabled(false);
  setPausedRuntimeControlsEnabled(false);
  els.pauseButton.disabled = true;
  els.pauseButton.textContent = "Pause";
  els.stopButton.disabled = true;
  updateDownloads(false);
  renderSplatInspector();
  updateMemoryRecommendation();
  if (terminalStatus) {
    state.stopRequested = false;
    const generation = ++state.webGpuRecoveryGeneration;
    setStatus("recovering gpu");
    setTrainingMessage("Training result preserved. Restoring the GPU preview...", "info");
    log("GPU preview connection lost after training; restoring it without discarding the result");
    publishState();
    recoverCompletedResultWebGpu(generation, terminalStatus).catch((error) => {
      if (generation !== state.webGpuRecoveryGeneration) return;
      state.webGpuRecoveryPending = false;
      setStatus("gpu unavailable");
      setTrainingMessage("Training result preserved, but the GPU preview could not be restored.", "info");
      log(`GPU preview recovery failed: ${error.message}`);
      publishLifecycleInteractionState();
    });
    return;
  }
  setTrainingMessage(`Training failed: ${reason}`, "error");
  setStatus("error");
  log(reason);
  publishState();
}


function rgbaParity(a, b) {
  if (a.length !== b.length) return { exact: false, max_abs: 255, mean_abs: Number.POSITIVE_INFINITY };
  let total = 0;
  let maximum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const delta = Math.abs(a[i] - b[i]);
    total += delta;
    maximum = Math.max(maximum, delta);
  }
  return { exact: maximum === 0, max_abs: maximum, mean_abs: total / Math.max(1, a.length) };
}

function displayRgbaParity(a, b) {
  const raw = rgbaParity(a, b);
  if (a.length !== b.length) {
    return {
      exact: false,
      display_equivalent: false,
      ...raw,
      alpha_max_abs: 255,
      premultiplied_max_abs: 255,
      premultiplied_mean_abs: Number.POSITIVE_INFINITY,
      premultiplied_tolerance: 1,
    };
  }
  let alphaMaximum = 0;
  let premultipliedMaximum = 0;
  let premultipliedTotal = 0;
  let premultipliedMismatchPixels = 0;
  let maximumPixel = null;
  let channels = 0;
  for (let i = 0; i < a.length; i += 4) {
    alphaMaximum = Math.max(alphaMaximum, Math.abs(a[i + 3] - b[i + 3]));
    let pixelMaximum = 0;
    for (let channel = 0; channel < 3; channel += 1) {
      const source = Math.round(a[i + channel] * a[i + 3] / 255);
      const decoded = Math.round(b[i + channel] * b[i + 3] / 255);
      const delta = Math.abs(source - decoded);
      pixelMaximum = Math.max(pixelMaximum, delta);
      if (delta > premultipliedMaximum) {
        maximumPixel = {
          pixel: i / 4,
          channel,
          training_rgba: Array.from(a.slice(i, i + 4)),
          standalone_rgba: Array.from(b.slice(i, i + 4)),
        };
      }
      premultipliedMaximum = Math.max(premultipliedMaximum, delta);
      premultipliedTotal += delta;
      channels += 1;
    }
    if (pixelMaximum > 1) premultipliedMismatchPixels += 1;
  }
  // The training buffer and the standalone render target quantize the same
  // float alpha through different 8-bit paths. A one-code alpha difference
  // can add one more code of premultiplied-color round-off, so allow two only
  // when that alpha difference is present. An alpha-identical two-code color
  // difference remains a real render mismatch and still blocks export.
  const premultipliedTolerance = alphaMaximum > 0 ? 2 : 1;
  const displayEquivalent =
    alphaMaximum <= 1 && premultipliedMaximum <= premultipliedTolerance;
  return {
    exact: displayEquivalent,
    display_equivalent: displayEquivalent,
    max_abs: raw.max_abs,
    mean_abs: raw.mean_abs,
    raw_exact: raw.exact,
    alpha_max_abs: alphaMaximum,
    premultiplied_max_abs: premultipliedMaximum,
    premultiplied_mean_abs: premultipliedTotal / Math.max(1, channels),
    premultiplied_tolerance: premultipliedTolerance,
    premultiplied_mismatch_pixels: premultipliedMismatchPixels,
    maximum_pixel: maximumPixel,
  };
}

function srgbSignalStatisticsFromRgb(rgb, alpha = null) {
  const pixelCount = Math.floor(rgb.length / 3);
  const channelSum = [0, 0, 0];
  let lumaSum = 0;
  let alphaSum = 0;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const r = Math.max(0, Math.min(1, Number(rgb[pixel * 3]) || 0));
    const g = Math.max(0, Math.min(1, Number(rgb[pixel * 3 + 1]) || 0));
    const b = Math.max(0, Math.min(1, Number(rgb[pixel * 3 + 2]) || 0));
    channelSum[0] += r;
    channelSum[1] += g;
    channelSum[2] += b;
    lumaSum += r * 0.2126 + g * 0.7152 + b * 0.0722;
    alphaSum += Math.max(0, Math.min(1, Number(alpha?.[pixel] ?? 1)));
  }
  return {
    pixels: pixelCount,
    mean_rgb: channelSum.map((sum) => sum / Math.max(1, pixelCount)),
    mean_srgb_signal_luma: lumaSum / Math.max(1, pixelCount),
    mean_alpha: alphaSum / Math.max(1, pixelCount),
  };
}

function srgbSignalStatisticsFromRgba(rgba) {
  const pixelCount = Math.floor(rgba.length / 4);
  const rgb = new Float32Array(pixelCount * 3);
  const alpha = new Float32Array(pixelCount);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    rgb[pixel * 3] = rgba[pixel * 4] / 255;
    rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1] / 255;
    rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2] / 255;
    alpha[pixel] = rgba[pixel * 4 + 3] / 255;
  }
  return srgbSignalStatisticsFromRgb(rgb, alpha);
}

function trainingColorSpaceAudit(image, trainState, trainingRgba, standaloneRgba) {
  const source = srgbSignalStatisticsFromRgb(image.rgb, image.alpha);
  const trainingFront = srgbSignalStatisticsFromRgba(trainingRgba);
  const standaloneFront = srgbSignalStatisticsFromRgba(standaloneRgba);
  const stageStats = (stageImage) => stageImage
    ? srgbSignalStatisticsFromRgb(stageImage.rgb, stageImage.alpha)
    : null;
  return {
    contract: "front and virtual teachers use the same sRGB signal values",
    source_decode: "Canvas 2D RGBA8 -> float / 255",
    teacher_sampling: "bilinear interpolation in sRGB signal space",
    training_display: "direct unorm output; no extra gamma encode",
    source,
    coarse_teacher: stageStats(trainState?.coarseImage),
    mid_teacher: stageStats(trainState?.midImage),
    training_front: trainingFront,
    standalone_front: standaloneFront,
    training_minus_source_luma: trainingFront.mean_srgb_signal_luma - source.mean_srgb_signal_luma,
    standalone_minus_source_luma: standaloneFront.mean_srgb_signal_luma - source.mean_srgb_signal_luma,
    training_minus_standalone_luma:
      trainingFront.mean_srgb_signal_luma - standaloneFront.mean_srgb_signal_luma,
    render_parity: displayRgbaParity(trainingRgba, standaloneRgba),
  };
}

function layerTelemetryEnabled() {
  return QA_RUNTIME_ENABLED && (
    globalThis.__flatPhotoLayerTelemetry === true || layerEfficiencyDiagnosticsRequested()
  );
}

function lastEventAtStep(events, step) {
  if (!Array.isArray(events)) return null;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index]?.step === step) return events[index];
  }
  return null;
}

function sampledFrameTelemetry(snapshot, previous) {
  const pixelCount = snapshot.width * snapshot.height;
  const stride = Math.max(1, Math.ceil(Math.sqrt(pixelCount / 4096)));
  const luma = [];
  const rgbDelta = [];
  const signedDelta = [];
  for (let y = 0; y < snapshot.height; y += stride) {
    for (let x = 0; x < snapshot.width; x += stride) {
      const offset = (y * snapshot.width + x) * 4;
      const value = snapshot.rgba[offset] * 0.299 + snapshot.rgba[offset + 1] * 0.587 + snapshot.rgba[offset + 2] * 0.114;
      luma.push(value);
      if (previous?.width === snapshot.width && previous?.height === snapshot.height) {
        const priorOffset = offset;
        const delta = value - previous.luma[luma.length - 1];
        signedDelta.push(delta);
        rgbDelta.push((
          Math.abs(snapshot.rgba[offset] - previous.rgba[priorOffset]) +
          Math.abs(snapshot.rgba[offset + 1] - previous.rgba[priorOffset + 1]) +
          Math.abs(snapshot.rgba[offset + 2] - previous.rgba[priorOffset + 2])
        ) / 3);
      }
    }
  }
  if (!signedDelta.length) return { luma: Float32Array.from(luma), signedDelta: Float32Array.from(signedDelta) };
  const previousLuma = previous.luma;
  const meanCurrent = luma.reduce((sum, value) => sum + value, 0) / luma.length;
  const meanPrevious = previousLuma.reduce((sum, value) => sum + value, 0) / previousLuma.length;
  let varianceCurrent = 0;
  let variancePrevious = 0;
  let covariance = 0;
  let reversalCount = 0;
  let reversalEnergy = 0;
  for (let i = 0; i < luma.length; i += 1) {
    varianceCurrent += (luma[i] - meanCurrent) ** 2;
    variancePrevious += (previousLuma[i] - meanPrevious) ** 2;
    covariance += (luma[i] - meanCurrent) * (previousLuma[i] - meanPrevious);
    if (previous.signedDelta?.length === signedDelta.length && signedDelta[i] * previous.signedDelta[i] < 0) {
      reversalCount += 1;
      reversalEnergy += Math.min(Math.abs(signedDelta[i]), Math.abs(previous.signedDelta[i]));
    }
  }
  const denominator = Math.max(1, luma.length - 1);
  rgbDelta.sort((a, b) => a - b);
  return {
    luma: Float32Array.from(luma),
    signedDelta: Float32Array.from(signedDelta),
    frame_luma_mad: signedDelta.reduce((sum, value) => sum + Math.abs(value), 0) / signedDelta.length / 255,
    frame_rgb_p95: percentileSorted(rgbDelta, 0.95) / 255,
    temporal_ssim: ssimFromMoments(
      meanCurrent / 255,
      meanPrevious / 255,
      varianceCurrent / denominator / (255 ** 2),
      variancePrevious / denominator / (255 ** 2),
      covariance / denominator / (255 ** 2),
    ),
    reversal_ratio: previous.signedDelta?.length === signedDelta.length ? reversalCount / signedDelta.length : null,
    reversal_energy: previous.signedDelta?.length === signedDelta.length ? reversalEnergy / signedDelta.length / 255 : null,
  };
}

function layerPairDirection(indexA, layerA, indexB, layerB) {
  if (layerA !== layerB) return layerA > layerB ? 1 : -1;
  return indexA < indexB ? 1 : -1;
}

function layerOrderComparatorProbe() {
  return {
    lower_depth_loses: layerPairDirection(2, 0.25, 7, 0.75) === -1,
    higher_depth_wins: layerPairDirection(2, 0.75, 7, 0.25) === 1,
    equal_depth_uses_low_index: layerPairDirection(2, 0.5, 7, 0.5) === 1,
    equal_depth_reverses_for_high_index: layerPairDirection(7, 0.5, 2, 0.5) === -1,
  };
}

function localOverlapPairOrders(geometry, count) {
  const gridSize = 16;
  const sampleLimit = 32;
  const cells = new Map();
  const layers = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const x = geometry.xy[index * 2];
    const y = geometry.xy[index * 2 + 1];
    const cellX = Math.max(0, Math.min(gridSize - 1, Math.floor((x * 0.5 + 0.5) * gridSize)));
    const cellY = Math.max(0, Math.min(gridSize - 1, Math.floor((y * 0.5 + 0.5) * gridSize)));
    const key = cellY * gridSize + cellX;
    const cell = cells.get(key) || { items: [], seen: 0 };
    cell.seen += 1;
    if (cell.items.length < sampleLimit) {
      cell.items.push(index);
    } else {
      const hash = Math.imul((index + 1) ^ key, 0x9e3779b1) >>> 0;
      const replacement = hash % cell.seen;
      if (replacement < sampleLimit) cell.items[replacement] = index;
    }
    cells.set(key, cell);
    const packed = geometry.transform[index * 4 + 3];
    layers[index] = packedLayerOrder(packed);
  }
  const pairs = new Map();
  const addPair = (ia, ib) => {
      if (ia === ib) return;
      const ax = geometry.xy[ia * 2];
      const ay = geometry.xy[ia * 2 + 1];
      const asx = geometry.transform[ia * 4];
      const asy = geometry.transform[ia * 4 + 1];
      const at = geometry.transform[ia * 4 + 2];
      const arx = RENDER_SIGMA * (Math.abs(Math.cos(at)) * asx + Math.abs(Math.sin(at)) * asy);
      const ary = RENDER_SIGMA * (Math.abs(Math.sin(at)) * asx + Math.abs(Math.cos(at)) * asy);
      const bx = geometry.xy[ib * 2];
      const by = geometry.xy[ib * 2 + 1];
      const bsx = geometry.transform[ib * 4];
      const bsy = geometry.transform[ib * 4 + 1];
      const bt = geometry.transform[ib * 4 + 2];
      const brx = RENDER_SIGMA * (Math.abs(Math.cos(bt)) * bsx + Math.abs(Math.sin(bt)) * bsy);
      const bry = RENDER_SIGMA * (Math.abs(Math.sin(bt)) * bsx + Math.abs(Math.cos(bt)) * bsy);
      if (Math.abs(ax - bx) > arx + brx || Math.abs(ay - by) > ary + bry) return;
      const low = Math.min(ia, ib);
      const high = Math.max(ia, ib);
      pairs.set(`${low}:${high}`, layerPairDirection(low, layers[low], high, layers[high]));
  };
  const neighborOffsets = [[0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]];
  for (const key of [...cells.keys()].sort((a, b) => a - b)) {
    const cellX = key % gridSize;
    const cellY = Math.floor(key / gridSize);
    const items = cells.get(key).items;
    for (const [dx, dy] of neighborOffsets) {
      const neighborX = cellX + dx;
      const neighborY = cellY + dy;
      if (neighborX < 0 || neighborX >= gridSize || neighborY < 0 || neighborY >= gridSize) continue;
      const neighborKey = neighborY * gridSize + neighborX;
      const neighborItems = cells.get(neighborKey)?.items;
      if (!neighborItems) continue;
      if (neighborKey === key) {
        for (let a = 0; a < items.length; a += 1) {
          for (let b = a + 1; b < items.length; b += 1) addPair(items[a], items[b]);
        }
      } else {
        for (const ia of items) {
          for (const ib of neighborItems) addPair(ia, ib);
        }
      }
    }
  }
  return pairs;
}

async function recordLayerTelemetry(step, run = null) {
  assertTrainingRun(run);
  if (!layerTelemetryEnabled() || !state.webgpu.renderer?.trainState) return;
  const previousRecord = state.metrics.layer_telemetry[state.metrics.layer_telemetry.length - 1];
  if (previousRecord?.step === step) return;
  const count = state.params.count;
  const [snapshot, geometry] = await awaitTrainingRun(run, Promise.all([
    state.webgpu.renderer.capturePresentedStateRgba(),
    state.webgpu.renderer.readLayerTelemetryGeometry(count),
  ]));
  if (!snapshot || !geometry) return;
  const depth = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const packed = geometry.transform[index * 4 + 3];
    depth[index] = packedLayerOrder(packed);
  }
  const order = Array.from({ length: count }, (_, index) => index)
    .sort((a, b) => depth[a] - depth[b] || a - b);
  const ranks = new Int32Array(count);
  for (let rank = 0; rank < order.length; rank += 1) ranks[order[rank]] = rank;
  const previous = state.layerTelemetryState;
  const common = Math.min(count, previous?.depth.length || 0);
  let layerDeltaTotal = 0;
  let layerDeltaMax = 0;
  let rankFlipCount = 0;
  let rankReflipCount = 0;
  const movementSigns = new Int8Array(count);
  for (let index = 0; index < common; index += 1) {
    const delta = Math.abs(depth[index] - previous.depth[index]);
    layerDeltaTotal += delta;
    layerDeltaMax = Math.max(layerDeltaMax, delta);
    const movement = Math.sign(ranks[index] - previous.ranks[index]);
    movementSigns[index] = movement;
    if (movement !== 0) rankFlipCount += 1;
    if (movement !== 0 && previous.movementSigns[index] !== 0 && movement !== previous.movementSigns[index]) {
      rankReflipCount += 1;
    }
  }
  const pairOrders = localOverlapPairOrders(geometry, count);
  const densify = lastEventAtStep(state.metrics.densify_events, step);
  const previousPairState = previous;
  let persistentPairs = 0;
  let flippedPairs = 0;
  if (previousPairState?.pairOrders) {
    for (const [key, direction] of pairOrders) {
      if (!previousPairState.pairOrders.has(key)) continue;
      persistentPairs += 1;
      if (previousPairState.pairOrders.get(key) !== direction) flippedPairs += 1;
    }
  }
  const pairUnion = previousPairState?.pairOrders
    ? pairOrders.size + previousPairState.pairOrders.size - persistentPairs
    : 0;
  const frameStats = sampledFrameTelemetry(snapshot, previous?.snapshot);
  const frameDelta = previous?.snapshot?.rgba?.length === snapshot.rgba.length
    ? rgbaParity(previous.snapshot.rgba, snapshot.rgba)
    : null;
  const stage = snapshot.kind;
  const triggers = [];
  if (!previous || previous.stage !== stage) triggers.push("stage-change");
  if (densify) triggers.push("growth");
  if (layerEfficiencyDiagnosticsRequested()) {
    const steps = Math.max(1, state.metrics?.steps_requested || step || 1);
    const checkpointBucket = Math.min(4, Math.floor(step / Math.max(1, steps / 4)));
    const checkpoints = state.layerEfficiencyCheckpoints || [];
    if (checkpoints.at(-1)?.bucket !== checkpointBucket || step >= steps) {
      const deepCount = Math.max(1, Math.ceil(count * layerEfficiencyVariants().deepFraction));
      checkpoints.push({
        step,
        bucket: checkpointBucket,
        splats: count,
        deepest_indices: order.slice(0, deepCount),
      });
      state.layerEfficiencyCheckpoints = checkpoints.slice(-5);
    }
  }
  state.metrics.layer_telemetry.push({
    step,
    stage,
    triggers,
    splats: count,
    new_splats: Math.max(0, count - common),
    layer_delta_mean: common > 0 ? layerDeltaTotal / common : null,
    layer_delta_max: common > 0 ? layerDeltaMax : null,
    rank_flip_count: common > 0 ? rankFlipCount : null,
    rank_flip_ratio: common > 0 ? rankFlipCount / common : null,
    rank_reflip_count: common > 0 ? rankReflipCount : null,
    rank_reflip_ratio: common > 0 ? rankReflipCount / common : null,
    local_pair_count: pairOrders.size,
    local_pair_persistent_count: persistentPairs,
    local_order_flip_count: flippedPairs,
    local_order_flip_ratio: persistentPairs > 0 ? flippedPairs / persistentPairs : null,
    local_pair_membership_churn: pairUnion > 0
      ? (pairOrders.size + (previousPairState?.pairOrders?.size || 0) - 2 * persistentPairs) / pairUnion
      : null,
    frame_width: snapshot.width,
    frame_height: snapshot.height,
    frame_rgba_mean_abs: frameDelta?.mean_abs ?? null,
    frame_rgba_max_abs: frameDelta?.max_abs ?? null,
    frame_luma_mad: frameStats.frame_luma_mad ?? null,
    frame_rgb_p95: frameStats.frame_rgb_p95 ?? null,
    temporal_ssim: frameStats.temporal_ssim ?? null,
    reversal_ratio: frameStats.reversal_ratio ?? null,
    reversal_energy: frameStats.reversal_energy ?? null,
  });
  state.layerTelemetryState = {
    depth,
    ranks,
    movementSigns,
    pairOrders,
    snapshot: { ...snapshot, luma: frameStats.luma, signedDelta: frameStats.signedDelta },
    stage,
  };
}

function summarizeLayerEfficiency(params, attribution, tileDiagnostics, checkpoints, initialCount) {
  if (!params?.depthOrder || !attribution || !tileDiagnostics) {
    return {
      supported: false,
      reason: !attribution ? "hidden-rgb-attribution-unavailable" : "tile-diagnostics-unavailable",
    };
  }
  const deepFraction = layerEfficiencyVariants().deepFraction;
  const deepThreshold = deepFraction;
  const layerCount = 10;
  const bins = Array.from({ length: layerCount }, (_, index) => ({
    layer: index,
    splats: 0,
    initial_splats: 0,
    positive_harm: 0,
    contribution_mass: 0,
    deep_occluded_pixel_hits: 0,
    deep_occluded_weight_mass: 0,
    tile_references: 0,
  }));
  const attributionByIndex = new Map(attribution.rows.map((row) => [row.index, row]));
  let totalHarm = 0;
  let deepHarm = 0;
  let totalContribution = 0;
  let deepContribution = 0;
  let totalDeepHits = 0;
  let deepestDeepHits = 0;
  let totalTileReferences = 0;
  let deepTileReferences = 0;
  const finalDeep = new Set();
  for (let index = 0; index < params.count; index += 1) {
    const layer = Math.max(0, Math.min(1, params.depthOrder[index]));
    const bin = bins[Math.min(layerCount - 1, Math.floor(layer * layerCount))];
    const row = attributionByIndex.get(index);
    const harm = row?.positive_leave_one_out_l1_harm || 0;
    const contribution = row?.contribution_mass || 0;
    const deepHits = row?.deep_occluded_pixel_hits || 0;
    const deepWeight = row?.deep_occluded_weight_mass || 0;
    const tileReferences = tileDiagnostics.per_splat_reference_count[index] || 0;
    bin.splats += 1;
    if (index < initialCount) bin.initial_splats += 1;
    bin.positive_harm += harm;
    bin.contribution_mass += contribution;
    bin.deep_occluded_pixel_hits += deepHits;
    bin.deep_occluded_weight_mass += deepWeight;
    bin.tile_references += tileReferences;
    totalHarm += harm;
    totalContribution += contribution;
    totalDeepHits += deepHits;
    totalTileReferences += tileReferences;
    if (layer <= deepThreshold) {
      finalDeep.add(index);
      deepHarm += harm;
      deepContribution += contribution;
      deepestDeepHits += deepHits;
      deepTileReferences += tileReferences;
    }
  }
  const persistence = (checkpoints || []).map((checkpoint) => {
    const prior = new Set(checkpoint.deepest_indices || []);
    let retained = 0;
    for (const index of finalDeep) if (prior.has(index)) retained += 1;
    return {
      step: checkpoint.step,
      splats: checkpoint.splats,
      final_deep_retained_ratio: retained / Math.max(1, Math.min(finalDeep.size, prior.size)),
    };
  });
  const persistentDeepRatio = persistence.length > 1
    ? persistence.slice(0, -1).reduce((sum, item) => sum + item.final_deep_retained_ratio, 0) / (persistence.length - 1)
    : null;
  const deepHarmRatio = deepHarm / Math.max(1e-12, totalHarm);
  return {
    supported: true,
    contract: "QA-only depth attribution; no training or export contract changes",
    deep_fraction: deepFraction,
    deep_harm_ratio: deepHarmRatio,
    deep_contribution_ratio: deepContribution / Math.max(1e-12, totalContribution),
    deep_occluded_hit_ratio: deepestDeepHits / Math.max(1, totalDeepHits),
    deep_tile_reference_ratio: deepTileReferences / Math.max(1, totalTileReferences),
    persistent_deep_ratio: persistentDeepRatio,
    relocation_gate_passed: deepHarmRatio >= 0.4 && persistentDeepRatio !== null && persistentDeepRatio >= 0.5,
    relocation_candidate_enabled: layerEfficiencyVariants().deepRelocation,
    bins,
    persistence,
    tile_quantization: tileDiagnostics.quantization,
    tile_quantization_gate: {
      ten_layers_passed: (tileDiagnostics.quantization?.[10]?.estimated_comparison_reduction_ratio || 0) >= 0.5,
      five_layers_passed: (tileDiagnostics.quantization?.[5]?.estimated_comparison_reduction_ratio || 0) >= 0.5,
      note: "comparison reduction is theoretical; no runtime speed claim without an end-to-end A/B",
    },
  };
}

function destroyBuffers(...buffers) {
  for (const buffer of buffers) {
    if (buffer) buffer.destroy();
  }
}

function makeBuffer(device, data, usage, track = null) {
  const buffer = device.createBuffer({
    size: Math.ceil(data.byteLength / 4) * 4,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  track?.(buffer);
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

function makeSizedBuffer(device, data, sizeBytes, usage, track = null) {
  const buffer = device.createBuffer({
    size: Math.ceil(Math.max(sizeBytes, data.byteLength) / 4) * 4,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  track?.(buffer);
  if (data.byteLength) {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }
  return buffer;
}

function packColors(params) {
  const color = new Float32Array(params.count * 4);
  for (let i = 0; i < params.count; i += 1) {
    color[i * 4] = params.rgb[i * 3];
    color[i * 4 + 1] = params.rgb[i * 3 + 1];
    color[i * 4 + 2] = params.rgb[i * 3 + 2];
    color[i * 4 + 3] = params.opacity[i];
  }
  return color;
}

function packPositions(params) {
  const positions = new Float32Array(params.count * 4);
  for (let i = 0; i < params.count; i += 1) {
    positions[i * 4] = params.xy[i * 2];
    positions[i * 4 + 1] = params.xy[i * 2 + 1];
    positions[i * 4 + 2] = params.virtualDepthEnabled
      ? Math.max(-VIRTUAL_DEPTH_RAW_LIMIT, Math.min(VIRTUAL_DEPTH_RAW_LIMIT, params.virtualDepth?.[i] || 0))
      : params.brushWidthTaperEnabled
        ? Math.max(0, Math.min(1, params.brushTaper?.[i] ?? DEFAULT_LAYERED_BRUSH_TAPER))
        : 0;
    positions[i * 4 + 3] = 0;
  }
  return positions;
}

function packTransforms(params) {
  const transform = new Float32Array(params.count * 4);
  for (let i = 0; i < params.count; i += 1) {
    transform[i * 4] = params.scale[i * 2];
    transform[i * 4 + 1] = params.scale[i * 2 + 1];
    transform[i * 4 + 2] = params.theta?.[i] || 0;
    const tag = Math.max(1, Math.floor(params.detailTags?.[i] || 1));
    let layer = params.layerOrderEnabled
      ? Math.max(0, Math.min(1, params.depthOrder?.[i] ?? (1 - i / Math.max(1, params.count - 1))))
      : 0;
    if (params.discreteLayersEnabled) layer = quantizeLayerOrder(layer, params.discreteLayerCount);
    transform[i * 4 + 3] = tag + layer * LAYER_CODE_RANGE;
  }
  return transform;
}

function selectedBackend() {
  return state.webgpu.supported && state.webgpu.renderer ? "webgpu-train+render" : "webgpu-unavailable";
}

function selectedLearningRates() {
  const maxAnisotropy = clampNumber(els.maxAnisotropy.value, LIMITS.maxAnisotropyMin, LIMITS.maxAnisotropyMax, DEFAULT_MAX_ANISOTROPY);
  const boundarySigma = selectedBoundarySigma();
  return {
    scale: DEFAULT_LR_SCALE,
    position: clampNumber(els.positionLearningRate.value, LIMITS.lrMin, LIMITS.lrDefaultMax, DEFAULT_POSITION_LR),
    color: clampNumber(els.colorLearningRate.value, LIMITS.lrMin, LIMITS.lrDefaultMax, DEFAULT_COLOR_LR),
    opacity: clampNumber(els.opacityLearningRate.value, LIMITS.lrMin, LIMITS.opacityLrMax, DEFAULT_OPACITY_LR),
    scaleParam: clampNumber(els.scaleLearningRate.value, LIMITS.lrMin, LIMITS.scaleLrMax, DEFAULT_SCALE_LR),
    rotation: clampNumber(els.rotationLearningRate.value, LIMITS.lrMin, LIMITS.lrDefaultMax, DEFAULT_ROTATION_LR),
    thetaAlign: clampNumber(els.thetaAlignRate.value, LIMITS.lrMin, LIMITS.thetaAlignLrMax, DEFAULT_THETA_ALIGN_LR),
    maxAnisotropy,
    maxPlanarScale: clampNumber(els.maxPlanarScale.value, LIMITS.maxPlanarScaleMin, LIMITS.maxPlanarScaleMax, DEFAULT_MAX_PLANAR_SCALE),
    surfaceAnisotropy: qualityRecoveryVariants().surfaceAnisotropy,
    boundarySigma,
    adaptiveDetail: true,
    detailCoherence: clampNumber(els.detailCoherence.value, LIMITS.detailCoherenceMin, LIMITS.detailCoherenceMax, 0.8),
  };
}

function selectedPreviewRefresh() {
  return ["frame", "10", "final"].includes(els.previewRefresh.value) ? els.previewRefresh.value : "10";
}

function shouldPresentTrainingStep(step, refresh) {
  if (refresh === "frame") return true;
  if (refresh === "10") return step % 10 === 0;
  return false;
}

function trainSyncInterval() {
  const override = Number(window.__flatPhotoTrainSyncInterval);
  const interval = Number.isFinite(override) && override > 0 ? override : DEFAULT_TRAIN_SYNC_INTERVAL;
  return Math.max(1, Math.min(64, Math.round(interval)));
}

function effectiveTrainSyncInterval(activeSplats, baseInterval = trainSyncInterval()) {
  const count = Math.max(0, Math.round(Number(activeSplats) || 0));
  if (count >= VERY_HIGH_SPLAT_SYNC_THRESHOLD) return Math.min(baseInterval, 1);
  if (count >= HIGH_SPLAT_SYNC_THRESHOLD) return Math.min(baseInterval, 2);
  return baseInterval;
}

function plannedAdaptiveGpuBatch({
  step,
  steps,
  desiredSize,
  metricInterval,
  previewRefresh,
  structuralStep,
  qaHashPending,
  densitySteps,
  growthSteps,
  growthSettings,
  runLayerSettings,
  paintSettings,
  currentContributionSettings,
}) {
  if (performanceVariants().gpuSchedulingMode !== "adaptive") return 1;
  const safeMetricInterval = Math.max(1, Math.round(metricInterval) || 1);
  const startStage = curriculumTrainingStage(
    step,
    steps,
    phase33Variants(),
    state.webgpu.renderer?.trainState?.coarseImage,
    state.webgpu.renderer?.trainState?.midImage,
  );
  const startPaintMutationAllowed = opaquePaintStructuralMutationAllowed(
    step,
    steps,
    Boolean(state.params?.opaqueLayered),
    paintSettings?.opaquePaintSettleFraction,
  );
  const startPaintDetailRecoveryScheduled =
    Boolean(state.params?.opaqueLayered) &&
    opaquePaintDetailRecoveryDue(
      step,
      steps,
      OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL,
    );
  const startBrushSurfaceRecoveryScheduled =
    normalizedKernelShape(state.params?.kernelShape) === "opaque-brush" &&
    startPaintDetailRecoveryScheduled;
  const startLayerSchedule = layerOptimizationSettings(
    step,
    steps,
    startStage,
    runLayerSettings,
    Boolean(state.params?.opaqueLayered),
    paintSettings?.opaquePaintSettleFraction,
  );
  const startDiscreteLayerScheduled = Boolean(
    state.params?.discreteLayersEnabled &&
    state.params?.discreteLayerMoveRadius > 0 &&
    step >= densitySteps &&
    startPaintDetailRecoveryScheduled,
  );
  const surfaceLayerSortSettings = {
    enabled: state.params?.surfaceLayerPriorEnabled,
    p1Interval: state.params?.surfaceLayerPriorP1Interval,
    p2Interval: state.params?.surfaceLayerPriorP2Interval,
    p3Interval: state.params?.surfaceLayerPriorP3Interval,
    untilFraction: state.params?.surfaceLayerPriorUntilFraction,
  };
  const startSurfaceLayerSortDue = scaleBiasedSurfaceLayerSortSchedule(
    step,
    steps,
    surfaceLayerSortSettings,
  ).due;
  if (
    previewRefresh === "frame" ||
    structuralStep ||
    startLayerSchedule.due ||
    (startPaintMutationAllowed && startPaintDetailRecoveryScheduled) ||
    startDiscreteLayerScheduled ||
    startSurfaceLayerSortDue ||
    step % safeMetricInterval === 0 ||
    shouldPresentTrainingStep(step, previewRefresh) ||
    step % 32 === 0 ||
    step === steps ||
    qaHashPending ||
    performanceProfileLabels(step, steps).length > 0 ||
    virtualTiltStepSpec(step, startStage, steps).enabled
  ) return 1;
  const requested = Math.max(1, Math.min(MAX_TRAIN_BATCH_SIZE, Math.round(desiredSize) || 1));
  let count = 1;
  for (let candidate = step + 1; candidate <= steps && count < requested; candidate += 1) {
    const paintMutationAllowed = opaquePaintStructuralMutationAllowed(
      candidate,
      steps,
      Boolean(state.params?.opaqueLayered),
      paintSettings?.opaquePaintSettleFraction,
    );
    const candidateStage = curriculumTrainingStage(
      candidate,
      steps,
      phase33Variants(),
      state.webgpu.renderer?.trainState?.coarseImage,
      state.webgpu.renderer?.trainState?.midImage,
    );
    const densityScheduled = growthEventScheduled(
      candidate,
      densitySteps,
      growthSteps,
      growthSettings,
    );
    const densityDue = paintMutationAllowed && densityScheduled;
    const paintDetailRecoveryScheduled =
      Boolean(state.params?.opaqueLayered) &&
      opaquePaintDetailRecoveryDue(
        candidate,
        steps,
        OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL,
      );
    const brushSurfaceRecoveryScheduled =
      normalizedKernelShape(state.params?.kernelShape) === "opaque-brush" &&
      paintDetailRecoveryScheduled;
    const discreteLayerScheduled = Boolean(
      state.params?.discreteLayersEnabled &&
      state.params?.discreteLayerMoveRadius > 0 &&
      candidate >= densitySteps &&
      paintDetailRecoveryScheduled,
    );
    const relocationScheduled =
      growthSettings.densityEventsEnabled &&
      growthSettings.mcmcRelocationEnabled &&
      candidate > densifyWarmupSteps(densitySteps) &&
      (
        (
          candidate <= Math.floor(densitySteps * 0.85) &&
          candidate % EXPERIMENTAL_REFINE_EVERY === 0
        ) ||
        brushSurfaceRecoveryScheduled
      );
    const relocationDue = paintMutationAllowed && relocationScheduled;
    const layerSchedule = layerOptimizationSettings(
      candidate,
      steps,
      candidateStage,
      runLayerSettings,
      Boolean(state.params?.opaqueLayered),
      paintSettings?.opaquePaintSettleFraction,
    );
    const layerDue = layerSchedule.due;
    const visibilityCompactionDue = opaquePaintVisibilityCompactionDue(
      candidate,
      steps,
      paintSettings,
    );
    const contributionSettings = {
      ...currentContributionSettings,
      opaqueLayered: paintSettings?.opaqueLayered,
      opaquePaintSettleFraction: paintSettings?.opaquePaintSettleFraction,
    };
    const contributionCompactionResetDue = currentContributionCompactionResetDue(
      candidate,
      steps,
      contributionSettings,
    );
    const contributionCompactionDue = currentContributionCompactionDue(
      candidate,
      steps,
      contributionSettings,
    );
    const surfaceLayerSortDue = scaleBiasedSurfaceLayerSortSchedule(
      candidate,
      steps,
      surfaceLayerSortSettings,
    ).due;
    const suppressedPaintMutationScheduled = !paintMutationAllowed && (
      densityScheduled ||
      relocationScheduled ||
      layerSchedule.scheduled
    );
    if (
      candidateStage !== startStage ||
      densityDue ||
      relocationDue ||
      layerDue ||
      (paintMutationAllowed && paintDetailRecoveryScheduled) ||
      discreteLayerScheduled ||
      visibilityCompactionDue ||
      contributionCompactionResetDue ||
      contributionCompactionDue ||
      surfaceLayerSortDue ||
      suppressedPaintMutationScheduled ||
      performanceProfileLabels(candidate, steps).length > 0 ||
      virtualTiltStepSpec(candidate, candidateStage, steps).enabled ||
      candidate % safeMetricInterval === 0 ||
      shouldPresentTrainingStep(candidate, previewRefresh) ||
      candidate % 32 === 0 ||
      candidate === steps
    ) break;
    count += 1;
  }
  return count;
}

function setInputControlsDisabled(disabled) {
  for (const element of [
    els.fileInput,
    els.algorithmSelect,
    els.trainSize,
    els.initialSplatCount,
    els.adaptiveGridInitializationFraction,
    els.finalSplatCount,
    els.capacityMode,
    els.memoryLimiterUnlock,
    els.stepCount,
    els.previewRefresh,
    els.liveQualityMetrics,
    els.currentContributionCompaction,
    els.currentContributionCompactionStart,
    els.currentContributionCompactionMaxRemoval,
    els.currentContributionCompactionNearZero,
    els.currentContributionCompactionWindow,
    els.monochromeUnderpainting,
    els.colorFinishStart,
    els.tileCullingToggle,
    els.trainLayerOrder,
    els.layerAwareAccumulation,
    els.discreteLayers,
    els.discreteLayerCount,
    els.discreteLayerMoveRadius,
    els.layerUpdateInterval,
    els.scaleBiasedSurfaceLayerPrior,
    els.scaleBiasedSurfaceLayerPriorLayers,
    els.scaleBiasedSurfaceLayerPriorP1Interval,
    els.scaleBiasedSurfaceLayerPriorP2Interval,
    els.scaleBiasedSurfaceLayerPriorP3Interval,
    els.scaleBiasedSurfaceLayerPriorUntil,
    els.positionLearningRate,
    els.colorLearningRate,
    els.opacityLearningRate,
    els.alphaLossWeight,
    els.rectangleLearnedOpacityMin,
    els.rectangleLearnedOpacityMax,
    els.rectangleOpacityGradientMin,
    els.rectangleOpacityGradientMax,
    els.rectangleCenterOpacityGradientMin,
    els.rectangleCenterOpacityGradientMax,
    els.layeredBrushLearnedOpacityMin,
    els.layeredBrushLearnedOpacityMax,
    els.layeredBrushOpacityGradientStart,
    els.layeredBrushOpacityGradientEnd,
    els.layeredBrushCenterOpacityGradientMin,
    els.layeredBrushCenterOpacityGradientMax,
    els.layeredBrushWidthTaperStart,
    els.layeredBrushWidthTaperEnd,
    els.layeredBrushLocalColorFlow,
    els.layeredBrushStrokePersistence,
    els.layeredBrushMinAspectRatio,
    els.layeredBrushMaxAspectRatio,
    els.layeredBrushRibbonAspectFloor,
    els.layeredBrushAccentAspectFloor,
    els.rectangleTopRatio,
    els.rectangleTopRatioMax,
    els.rectangleMinAspectRatio,
    els.rectangleMaxAspectRatio,
    els.rectangleOrientation,
    els.rectanglePreserveArea,
    els.rectangleEdgeDirectedTaper,
    els.rectangleStructureAwareRatio,
    els.rectangleAsymmetricSoftness,
    els.virtualBoundedDepth,
    els.virtualGofDensity,
    els.virtualCameraShare,
    els.virtualCameraMaxAngle,
    els.virtualCameraCount,
    els.virtualCameraFov,
    els.stageAwareGrowth,
    els.stageGrowthP1,
    els.stageGrowthP2,
    els.stageGrowthP3,
    els.structureGuidedAllocation,
    els.structureRegionGrid,
    els.midTrainingOverdensityCorrection,
    els.overdensityCorrectionSchedule,
    els.overdensityCorrectionInterval,
    els.overdensityDonorPercent,
    els.scaleLearningRate,
    els.rotationLearningRate,
    els.thetaAlignRate,
    els.maxAnisotropy,
    els.maxPlanarScale,
    els.boundarySigma,
    els.detailCoherence,
    els.phaseRelativeScaleGuard,
    els.p1RelativeScaleFloorRatio,
    els.p2RelativeScaleFloorRatio,
    els.p3RelativeScaleFloorRatio,
    els.densifyInterval,
    els.growthPercentage,
    els.growthApplyUntil,
    els.growthSignalThreshold,
    els.retryWebGpuButton,
    els.pathInput,
    els.loadImageButton,
    els.clearImageButton,
    els.sampleButton,
    els.pathButton,
  ]) {
    element.disabled = disabled;
  }
  syncTrainSizeUi();
  if (!disabled) {
    syncLayerOrderDependency();
    syncVirtualCameraDependency();
  }
}

function syncAlgorithmRequirements() {
  const opaqueLayered = algorithmUsesOpaqueLayeredPaint();
  const brushSelected = algorithmUsesLayeredOpaqueBrush();
  const rectangleSelected = algorithmUsesRectangleKernel();
  document.documentElement.dataset.opaquePaintSettings = "visible";
  if (brushSelected) {
    if (!els.discreteLayerMoveRadius.dataset.nonBrushValue) {
      els.discreteLayerMoveRadius.dataset.nonBrushValue =
        els.discreteLayerMoveRadius.value || String(DEFAULT_DISCRETE_LAYER_MOVE_RADIUS);
    }
    els.discreteLayerMoveRadius.value = String(LAYERED_OPAQUE_BRUSH_LAYER_MOVE_RADIUS);
  } else if (els.discreteLayerMoveRadius.dataset.nonBrushValue) {
    els.discreteLayerMoveRadius.value = els.discreteLayerMoveRadius.dataset.nonBrushValue;
    delete els.discreteLayerMoveRadius.dataset.nonBrushValue;
  }
  for (const [control, fallback] of [
    [els.alphaLossWeight, DEFAULT_ALPHA_LOSS_WEIGHT],
  ]) {
    if (brushSelected) {
      if (!control.dataset.nonOpaqueValue) control.dataset.nonOpaqueValue = control.value || String(fallback);
      control.value = "0";
    } else if (control.dataset.nonOpaqueValue) {
      control.value = control.dataset.nonOpaqueValue;
      delete control.dataset.nonOpaqueValue;
    }
  }
  if (opaqueLayered) {
    els.trainLayerOrder.checked = true;
    els.layerAwareAccumulation.checked = true;
    els.discreteLayers.checked = true;
  }
  els.opacityLearningRate.disabled = state.running;
  els.structureRegionGrid.disabled = state.running || (
    !els.structureGuidedAllocation.checked && !els.midTrainingOverdensityCorrection.checked
  );
  els.overdensityDonorPercent.disabled = state.running || !els.midTrainingOverdensityCorrection.checked;
  els.overdensityCorrectionSchedule.disabled = state.running || !els.midTrainingOverdensityCorrection.checked;
  els.overdensityCorrectionInterval.disabled = state.running ||
    !els.midTrainingOverdensityCorrection.checked ||
    phase39Variants().overdensityCorrectionSchedule !== "interval";
  els.alphaLossWeight.disabled = state.running || brushSelected;
  els.adaptiveGridInitializationFraction.disabled = state.running || algorithmUsesPaintKernel();
  els.adaptiveGridInitializationFraction.title = algorithmUsesPaintKernel()
    ? "Paint algorithms use deterministic image-importance BSP placement."
    : "Moves this percentage of the initial grid centers toward multiscale image structure on WebGPU before iteration 1.";
  els.rectangleLearnedOpacityMin.disabled = state.running || !rectangleSelected;
  els.rectangleLearnedOpacityMax.disabled = state.running || !rectangleSelected;
  els.rectangleOpacityGradientMin.disabled = state.running || !rectangleSelected;
  els.rectangleOpacityGradientMax.disabled = state.running || !rectangleSelected;
  els.rectangleCenterOpacityGradientMin.disabled = state.running || !rectangleSelected;
  els.rectangleCenterOpacityGradientMax.disabled = state.running || !rectangleSelected;
  els.layeredBrushLearnedOpacityMin.disabled = state.running || !brushSelected;
  els.layeredBrushLearnedOpacityMax.disabled = state.running || !brushSelected;
  els.layeredBrushOpacityGradientStart.disabled = state.running || !brushSelected;
  els.layeredBrushOpacityGradientEnd.disabled = state.running || !brushSelected;
  els.layeredBrushCenterOpacityGradientMin.disabled = state.running || !brushSelected;
  els.layeredBrushCenterOpacityGradientMax.disabled = state.running || !brushSelected;
  els.layeredBrushWidthTaperStart.disabled = state.running || !brushSelected;
  els.layeredBrushWidthTaperEnd.disabled = state.running || !brushSelected;
  els.layeredBrushLocalColorFlow.disabled = state.running || !brushSelected;
  els.layeredBrushStrokePersistence.disabled = state.running || !brushSelected;
  els.layeredBrushMinAspectRatio.disabled = state.running || !brushSelected;
  els.layeredBrushMaxAspectRatio.disabled = state.running || !brushSelected;
  els.layeredBrushRibbonAspectFloor.disabled =
    state.running || !brushSelected || !els.layeredBrushStrokePersistence.checked;
  els.layeredBrushAccentAspectFloor.disabled =
    state.running || !brushSelected || !els.layeredBrushStrokePersistence.checked;
  els.monochromeUnderpainting.disabled = state.running;
  els.colorFinishStart.disabled = state.running || !els.monochromeUnderpainting.checked;
  const contributionCompactionDisabled = state.running || !els.currentContributionCompaction.checked;
  els.currentContributionCompaction.disabled = state.running;
  els.currentContributionCompactionStart.disabled = contributionCompactionDisabled;
  els.currentContributionCompactionInterval.disabled = contributionCompactionDisabled;
  els.currentContributionCompactionMaxRemoval.disabled = contributionCompactionDisabled;
  els.currentContributionCompactionNearZero.disabled = contributionCompactionDisabled;
  els.currentContributionCompactionWindow.disabled = contributionCompactionDisabled;
  els.rectangleTopRatio.disabled = state.running || !rectangleSelected;
  els.rectangleTopRatioMax.disabled = state.running || !rectangleSelected;
  els.rectangleMinAspectRatio.disabled = state.running || !rectangleSelected;
  els.rectangleMaxAspectRatio.disabled = state.running || !rectangleSelected;
  els.rectangleOrientation.disabled = state.running || !rectangleSelected;
  els.scaleBiasedSurfaceLayerPrior.disabled = state.running;
  for (const control of [
    els.scaleBiasedSurfaceLayerPriorLayers,
    els.scaleBiasedSurfaceLayerPriorP1Interval,
    els.scaleBiasedSurfaceLayerPriorP2Interval,
    els.scaleBiasedSurfaceLayerPriorP3Interval,
    els.scaleBiasedSurfaceLayerPriorUntil,
  ]) {
    control.disabled = state.running || !els.scaleBiasedSurfaceLayerPrior.checked;
  }
  els.rectangleTopRatio.title = rectangleSelected
    ? "Minimum top-edge / base-edge ratio in the deterministic Rectangle shape range; 0 allows triangle tips."
    : "Available for Rectangle Splats.";
  els.rectangleTopRatioMax.title = rectangleSelected
    ? "Maximum top-edge / base-edge ratio in the deterministic Rectangle shape range; 1 makes the parallel edges equal."
    : "Available for Rectangle Splats.";
  els.rectangleMinAspectRatio.title = rectangleSelected
    ? "Minimum long-side / short-side aspect ratio; 1 allows square footprints."
    : "Available for Rectangle Splats.";
  els.rectangleMaxAspectRatio.title = rectangleSelected
    ? "Maximum long-side / short-side aspect ratio; this Rectangle value overrides Shared Max anisotropy."
    : "Available for Rectangle Splats.";
  els.rectangleOrientation.title = rectangleSelected
    ? "Free keeps structure-guided rotation; Vertical or Horizontal locks the long axis."
    : "Available for Rectangle Splats.";
  for (const control of [
    els.rectanglePreserveArea,
    els.rectangleEdgeDirectedTaper,
    els.rectangleStructureAwareRatio,
    els.rectangleAsymmetricSoftness,
  ]) {
    control.disabled = state.running || !rectangleSelected;
  }
  for (const control of [els.stageGrowthP1, els.stageGrowthP2, els.stageGrowthP3]) {
    control.disabled = state.running || !els.stageAwareGrowth.checked;
  }
  els.structureGuidedAllocation.disabled = state.running;
}

function syncLayerOrderDependency() {
  syncAlgorithmRequirements();
  if (els.trainLayerOrder.checked) els.tileCullingToggle.checked = true;
  els.tileCullingToggle.disabled = state.running || els.trainLayerOrder.checked;
  els.layerUpdateInterval.disabled = state.running || !els.trainLayerOrder.checked;
  const discreteAvailable = els.trainLayerOrder.checked && !algorithmUsesVirtualCameras();
  const opaqueLayered = algorithmUsesOpaqueLayeredPaint();
  els.trainLayerOrder.disabled = state.running || opaqueLayered;
  els.layerAwareAccumulation.disabled = state.running || !discreteAvailable || opaqueLayered;
  els.discreteLayers.disabled = state.running || !discreteAvailable || opaqueLayered;
  els.discreteLayerCount.disabled = state.running || !discreteAvailable || (!els.discreteLayers.checked && !els.layerAwareAccumulation.checked);
  els.discreteLayerMoveRadius.disabled = state.running || !discreteAvailable || !els.discreteLayers.checked;
}

function updateVirtualCameraCoverageEstimate() {
  if (!state.image) {
    els.virtualCameraCoverageEstimate.textContent = "Teacher coverage: load an image to estimate.";
    delete document.documentElement.dataset.virtualCameraCoverageEstimate;
    return null;
  }
  if (!algorithmUsesVirtualCameras()) {
    els.virtualCameraCoverageEstimate.textContent = "Teacher coverage is evaluated only for GS Virtual Camera Sampling.";
    delete document.documentElement.dataset.virtualCameraCoverageEstimate;
    return null;
  }
  const variants = virtualCameraSamplingVariants(algorithmUsesVirtualCameras());
  const stats = virtualCameraCoverageStats(state.image.width, state.image.height, variants);
  const percent = (value) => `${(value * 100).toFixed(1)}%`;
  els.virtualCameraCoverageEstimate.textContent = `Teacher coverage min / mean / max: ${percent(stats.minimum)} / ${percent(stats.mean)} / ${percent(stats.maximum)}`;
  document.documentElement.dataset.virtualCameraCoverageEstimate = JSON.stringify(stats);
  return stats;
}

function syncVirtualCameraDependency() {
  const virtualAlgorithm = algorithmUsesVirtualCameras();
  const panel = document.querySelector(".virtual-camera-panel");
  if (panel) panel.hidden = !virtualAlgorithm;
  const disabled = state.running || !virtualAlgorithm;
  els.virtualBoundedDepth.disabled = disabled;
  els.virtualGofDensity.disabled = disabled;
  els.virtualCameraShare.disabled = disabled;
  els.virtualCameraMaxAngle.disabled = disabled;
  els.virtualCameraCount.disabled = disabled;
  els.virtualCameraFov.disabled = disabled;
  updateVirtualCameraCoverageEstimate();
}

function pausedRuntimeControls() {
  return [
    els.previewRefresh,
    els.tileCullingToggle,
    els.positionLearningRate,
    els.colorLearningRate,
    els.opacityLearningRate,
    els.alphaLossWeight,
    els.scaleLearningRate,
    els.rotationLearningRate,
    els.thetaAlignRate,
    els.maxAnisotropy,
    els.maxPlanarScale,
    els.boundarySigma,
    els.detailCoherence,
    els.densifyInterval,
    els.growthPercentage,
    els.growthSignalThreshold,
  ];
}

function setPausedRuntimeControlsEnabled(enabled) {
  for (const element of pausedRuntimeControls()) element.disabled = !enabled;
  if (algorithmUsesLayeredOpaqueBrush()) {
    els.alphaLossWeight.disabled = true;
  }
  document.documentElement.dataset.pausedRuntimeControls = String(enabled);
}

function syncRuntimeMetrics(learningRates, previewRefresh) {
  if (!state.metrics) return;
  const growth = phase39Variants();
  const phase40 = phase40Variants();
  const runStageAware = Boolean(state.metrics.growth_schedule?.stage_aware);
  state.metrics.preview_refresh = previewRefresh;
  state.metrics.runtime_settings_revision = state.runtimeSettingsRevision;
  state.metrics.growth_schedule.percentage = growth.growthFraction * 100;
  state.metrics.growth_schedule.mode = runStageAware
    ? "stage-aware-percentage-cap"
    : "threshold-percentage-cap";
  state.metrics.growth_schedule.stage_aware = runStageAware;
  state.metrics.growth_schedule.signal_threshold = growth.growthSignalThreshold;
  state.metrics.learning_rates = {
    ...state.metrics.learning_rates,
    position: learningRates.position,
    color: learningRates.color,
    opacity: learningRates.opacity,
    scale: learningRates.scaleParam,
    rotation: learningRates.rotation,
    thetaAlign: learningRates.thetaAlign,
    maxAnisotropy: learningRates.maxAnisotropy,
    maxPlanarScale: learningRates.maxPlanarScale,
    boundarySigma: learningRates.boundarySigma,
    adaptiveDetail: learningRates.adaptiveDetail,
    detailCoherence: learningRates.detailCoherence,
  };
  state.metrics.alpha_loss_weight = phase40.alphaLossWeight;
  state.metrics.boundary_sigma = learningRates.boundarySigma;
  if (state.params) state.params.boundarySigma = learningRates.boundarySigma;
  const data = document.documentElement.dataset;
  data.runtimeAppliedRevision = String(state.runtimeSettingsRevision);
  data.runtimeAppliedPosition = String(learningRates.position);
  data.runtimeAppliedMaxPlanarScale = String(learningRates.maxPlanarScale);
  data.runtimeAppliedBoundarySigma = String(learningRates.boundarySigma);
  data.runtimeAppliedAlphaLossWeight = String(state.metrics.alpha_loss_weight);
  data.runtimeAppliedPreviewRefresh = previewRefresh;
}

async function presentTrainingPreview(step, run = null) {
  assertTrainingRun(run);
  if (layerTelemetryEnabled()) await awaitTrainingRun(run, recordLayerTelemetry(step, run));
  const renderer = state.webgpu.renderer;
  const virtualCameraSample = renderer?.lastTrainStats?.virtual_camera_sample;
  if (virtualCameraSample?.kind === "virtual") {
    const trainingStage = renderer.lastTrainStats?.training_stage;
    const previewImage = trainingStage === "coarse"
      ? renderer.trainState?.coarseImage || state.image
      : trainingStage === "mid"
        ? renderer.trainState?.midImage || state.image
        : state.image;
    if (els.tileCullingToggle.checked) {
      await awaitTrainingRun(run, renderer.prepareTileLists(previewImage, state.params, { sync: true }));
    }
    await awaitTrainingRun(run, renderer.refreshRenderState(previewImage, state.params));
    state.metrics.virtual_camera_sampling.preview_front_restores += 1;
  }
  let presented = false;
  if (els.outsidePreviewToggle.checked && !state.running && state.webgpu.renderer) {
    const buffers = state.webgpu.renderer.currentTrainBuffers(state.params);
    await awaitTrainingRun(run, state.webgpu.renderer.render(state.image, state.params, buffers));
    presented = true;
  } else {
    state.previewPadding = previewPaddingSpec(state.image, state.params, false);
    presented = renderer?.presentTrainState(state.image);
  }
  if (!presented) return false;
  if (state.previewMode === "splats") showCanvas("gpu");
  state.metrics.preview_frames += 1;
  state.metrics.last_preview_step = step;
  els.stepText.textContent = `${step} / ${state.metrics.steps_requested}`;
  publishState();
  return true;
}

async function resolveTileOverflowRetry(parameterHashBefore = null, run = null) {
  assertTrainingRun(run);
  const renderer = state.webgpu.renderer;
  if (!els.tileCullingToggle.checked || !renderer?.trainState?.tileReady) return 0;
  const tileCounters = await awaitTrainingRun(run, renderer.readTileCounters());
  if (!tileCounters) return 0;
  state.metrics.tile_counters = tileCounters;
  const noopSteps = Math.max(0, Math.floor(Number(tileCounters.noop_steps) || 0));
  if (tileCounters.overflow === 0 && noopSteps === 0) return 0;

  if (parameterHashBefore !== null) {
    const parameterHashAfter = await awaitTrainingRun(run, renderer.hashTrainParameters(state.params));
    const hashMatches = parameterHashAfter === parameterHashBefore;
    state.metrics.tile_retry_parameter_hash = {
      before: parameterHashBefore,
      after: parameterHashAfter,
      matches: hashMatches,
    };
    if (!hashMatches) {
      throw new Error(`tile overflow mutated parameters before retry: ${parameterHashBefore}/${parameterHashAfter}`);
    }
  }

  const previousCapacity = tileCounters.capacity;
  const expanded = tileCounters.overflow > 0
    ? await awaitTrainingRun(run, renderer.growTileIndexCapacity(tileCounters.total))
    : true;
  if (!expanded) {
    const rec = state.recommendation || updateMemoryRecommendation();
    const failure = { context: "tile-culling", reason: "safety_stop_tile_index_overflow", rec };
    state.metrics.stopped = true;
    state.metrics.safety_stop = {
      reason: failure.reason,
      context: failure.context,
      tile_indices: tileCounters.total,
      tile_capacity: tileCounters.capacity,
      tile_overflow: tileCounters.overflow,
      estimated_mb: rec.estimatedMB,
      budget_mb: rec.budgetMB,
    };
    setSafetyStop(failure);
    const error = new Error(`tile index capacity unavailable: ${tileCounters.total}/${tileCounters.capacity}`);
    error.safetyStop = true;
    throw error;
  }

  const retrySteps = Math.max(1, noopSteps);
  await awaitTrainingRun(run, renderer.clearTileNoopCounter());
  state.metrics.tile_retry_steps += retrySteps;
  state.metrics.tile_retry_events.push({
    after_step: state.metrics.steps_done,
    retry_steps: retrySteps,
    required_indices: tileCounters.total,
    previous_capacity: previousCapacity,
    next_capacity: renderer.trainState.tileIndexCapacity,
  });
  log(`tile overflow no-op: retrying ${retrySteps} iteration(s) after capacity ${previousCapacity} -> ${renderer.trainState.tileIndexCapacity}`);
  eventLog(`tile overflow retried ${retrySteps} iteration(s)`);
  return retrySteps;
}

function densityMetricSnapshot(metrics = null) {
  const coverage = metrics?.coverage || state.metrics?.coverage_stats;
  const regional = metrics?.regionalSsim || state.metrics?.latest_regional_ssim || state.metrics?.final_regional_ssim;
  return {
    step: metrics ? (state.metrics?.steps_done ?? 0) : (coverage?.step ?? state.metrics?.last_preview_step ?? 0),
    psnr: Number.isFinite(metrics?.psnr) ? metrics.psnr : state.metrics?.latest_psnr ?? state.metrics?.final_psnr ?? null,
    global_ssim: Number.isFinite(metrics?.ssim) ? metrics.ssim : state.metrics?.latest_global_ssim ?? state.metrics?.final_global_ssim ?? null,
    local_p10: Number.isFinite(regional?.p10) ? regional.p10 : regional?.p10 ?? null,
    alpha_ssim: Number.isFinite(metrics?.alphaSsim) ? metrics.alphaSsim : state.metrics?.latest_alpha_ssim ?? state.metrics?.final_alpha_ssim ?? null,
    alpha_l1: Number.isFinite(metrics?.alphaL1) ? metrics.alphaL1 : state.metrics?.latest_alpha_l1 ?? state.metrics?.final_alpha_l1 ?? null,
    background_exposure_ratio: Number.isFinite(coverage?.background_exposure_ratio)
      ? coverage.background_exposure_ratio
      : null,
  };
}

async function refreshTrainingResidualSignal(step, reason, run = null) {
  assertTrainingRun(run);
  const renderer = state.webgpu.renderer;
  if (!renderer?.trainState || !state.image || !state.params) {
    throw new Error("Training residual refresh requires active WebGPU state.");
  }
  const started = performance.now();
  const report = await awaitTrainingRun(run, renderer.refreshTrainingResidualMap(
    state.image,
    state.params,
    { step, reason },
  ));
  const telemetry = state.metrics?.training_residual_map;
  if (telemetry) {
    telemetry.refresh_count += 1;
    if (reason === "growth") telemetry.growth_refreshes += 1;
    if (reason === "relocation") telemetry.relocation_refreshes += 1;
    telemetry.last_step = step;
    telemetry.last_reason = reason;
    telemetry.last_wall_ms = performance.now() - started;
    if (telemetry.events.length < 160) {
      telemetry.events.push({
        ...report,
        wall_ms: telemetry.last_wall_ms,
      });
    }
  }
  return report;
}

function linkDensityEventMetricSnapshot(step, metrics) {
  const snapshot = densityMetricSnapshot(metrics);
  for (const event of state.metrics?.densify_events || []) {
    if (event.step === step && !event.metrics_after) event.metrics_after = snapshot;
  }
}

async function applyCurrentVisibilityCompaction(step, steps, run = null) {
  assertTrainingRun(run);
  const renderer = state.webgpu.renderer;
  const params = state.params;
  if (!params?.opaqueLayered || state.metrics?.stopped || !renderer?.trainState) return false;
  if ((state.metrics?.current_visibility_compaction_events || []).some((event) => event.step === step)) {
    return false;
  }
  // Layer priority uses the current trained order. The visibility path skips
  // percentile summaries and the planner uses fixed-size depth bins, avoiding
  // the previous per-candidate object allocation and O(N log N) sort.
  await awaitTrainingRun(run, renderer.readTrainedColors(params));
  assertFiniteParams(params, "current-visibility-compaction-readback");
  const importanceData = await awaitTrainingRun(
    run,
    renderer.readImportanceData(params.count, { includeSummary: false }),
  );
  const plan = hardZeroContributionPlan(params, importanceData);
  const report = { ...plan };
  delete report.keepIndices;
  delete report.pruneIndices;
  report.step = step;
  report.phase = "pre-settle";
  report.policy = "current-visibility-hard-zero";
  report.visibility_window = {
    reset_step: renderer.trainState?.currentVisibilityWindow?.reset_step ?? step - 1,
    measured_steps: 1,
    signal: "accepted-pixels-and-sum-t-before-alpha",
    scope: "all-opaque-paint-layers",
  };
  report.metrics_evaluation = "deferred-to-final";
  if (!plan.applied) {
    state.metrics.current_visibility_compaction = report;
    state.metrics.current_visibility_compaction_events.push(report);
    return false;
  }
  const compactResult = await awaitTrainingRun(run, renderer.compactTrainStateGpu(plan.keepIndices));
  if (!compactResult.compacted) {
    report.applied = false;
    report.reason = compactResult.reason || "gpu-compaction-skipped";
    report.gpu_compaction_allocation = compactResult.allocation || null;
    state.metrics.current_visibility_compaction = report;
    state.metrics.current_visibility_compaction_events.push(report);
    return false;
  }
  state.params = compactSplatParams(params, plan.keepIndices);
  updateTrainingRunOwnership(run, { params: state.params });
  if (els.tileCullingToggle.checked) {
    await awaitTrainingRun(run, renderer.prepareTileLists(state.image, state.params, { sync: true }));
  }
  report.gpu_compaction_ms = compactResult.gpu_ms;
  report.gpu_transient_bytes = compactResult.transient_bytes || 0;
  report.gpu_compaction_allocation = compactResult.allocation || null;
  report.optimizer_state_preserved = true;
  report.params_compacted = true;
  state.metrics.current_visibility_compaction_removed_total += plan.removed;
  state.metrics.current_visibility_compaction = report;
  state.metrics.current_visibility_compaction_events.push(report);
  state.metrics.num_gaussians = state.params.count;
  state.metrics.params_revision = (state.metrics.params_revision || 0) + 1;
  state.metrics.cpu_mirror_step = step;
  state.metrics.cpu_mirror_count = state.params.count;
  return true;
}

async function applyCurrentContributionCompaction(step, steps, settings, run = null) {
  assertTrainingRun(run);
  const renderer = state.webgpu.renderer;
  const params = state.params;
  if (!settings?.enabled || state.metrics?.stopped || !renderer?.trainState) return false;
  if ((state.metrics?.current_contribution_compaction_events || []).some((event) => event.step === step)) {
    return false;
  }
  // The planner does not need CPU RGB, XY, or scale data: its input is the
  // current exact-backward contribution buffer.  Avoiding that readback keeps
  // this late event from causing the post-training UI stalls it is intended to
  // reduce.
  const layerMirrorCurrent = await awaitTrainingRun(
    run,
    renderer.readTrainedLayerOrder(params),
  );
  if (!layerMirrorCurrent) throw new Error("Current Contribution Compaction requires a current GPU layer mirror.");
  const importanceData = await awaitTrainingRun(
    run,
    renderer.readImportanceData(params.count, { includeSummary: false }),
  );
  const plan = currentContributionCompactionPlan(
    params,
    importanceData,
    settings,
  );
  const report = { ...plan };
  delete report.keepIndices;
  report.step = step;
  report.phase = "periodic-visibility-recovery";
  report.policy = "current-contribution-compaction-v2";
  report.algorithm = state.metrics?.algorithm || settings.algorithm;
  report.event_index = (state.metrics?.current_contribution_compaction_events?.length || 0) + 1;
  report.interval_steps = settings.intervalSteps;
  report.visibility_window = {
    reset_step: renderer.trainState?.currentVisibilityWindow?.reset_step ?? step - 1,
    measured_steps: settings.measurementWindowSteps,
    signal: "accepted-pixels-and-sum-t-before-alpha",
    scope: settings.virtual
      ? "all-active-splats-across-one-complete-virtual-camera-pool"
      : "all-active-splats-in-current-training-view",
  };
  report.layer_order_source = "current-gpu-transform";
  report.metrics_evaluation = "deferred-to-final";
  if (!plan.applied) {
    state.metrics.current_contribution_compaction = report;
    state.metrics.current_contribution_compaction_events.push(report);
    return false;
  }
  const compactResult = await awaitTrainingRun(run, renderer.compactTrainStateGpu(plan.keepIndices));
  if (!compactResult.compacted) {
    report.applied = false;
    report.reason = compactResult.reason || "gpu-compaction-skipped";
    report.gpu_compaction_allocation = compactResult.allocation || null;
    state.metrics.current_contribution_compaction = report;
    state.metrics.current_contribution_compaction_events.push(report);
    return false;
  }
  state.params = compactSplatParams(params, plan.keepIndices);
  updateTrainingRunOwnership(run, { params: state.params });
  if (els.tileCullingToggle.checked) {
    await awaitTrainingRun(run, renderer.prepareTileLists(state.image, state.params, { sync: true }));
  }
  report.gpu_compaction_ms = compactResult.gpu_ms;
  report.gpu_transient_bytes = compactResult.transient_bytes || 0;
  report.gpu_compaction_allocation = compactResult.allocation || null;
  report.optimizer_state_preserved = true;
  report.params_compacted = true;
  state.metrics.current_contribution_compaction_removed_total += plan.removed;
  state.metrics.current_contribution_compaction = report;
  state.metrics.current_contribution_compaction_events.push(report);
  state.metrics.num_gaussians = state.params.count;
  state.metrics.params_revision = (state.metrics.params_revision || 0) + 1;
  // GPU compaction did not read XY/scale/RGB back.  Keep this CPU mirror
  // explicitly stale so Stop/finalization must obtain the authoritative state.
  state.metrics.cpu_mirror_step = null;
  state.metrics.cpu_mirror_count = null;
  state.metrics.cpu_mirror_current = false;
  state.metrics.cpu_layer_mirror_step = step;
  return true;
}

async function updatePreview(step, final = false, { present = true, readOnlyPeriodic = false } = {}, run = null) {
  assertTrainingRun(run);
  const backend = selectedBackend();
  if (!backend.startsWith("webgpu")) throw new Error(`WebGPU required: ${state.webgpu.reason}`);
  const safety = safetyFailure(computeBudgetFor(Number(els.trainSize.value), state.params.count, state.metrics?.steps_requested || 1), "metrics");
  if (safety) {
    setSafetyStop(safety);
    throw new Error(`${safety.reason}: metrics/readback skipped before budget overflow`);
  }
  if (layerTelemetryEnabled()) await awaitTrainingRun(run, recordLayerTelemetry(step, run));
  const reusableMetricRender = Boolean(
    !final && state.webgpu.renderer?.canReuseMetricRender(state.image)
  );
  const reuseMetricRender = Boolean(
    reusableMetricRender &&
    (readOnlyPeriodic || performanceVariants().metricTileReuse)
  );
  if (readOnlyPeriodic && !reuseMetricRender && els.tileCullingToggle.checked) {
    await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(
      state.image,
      state.params,
      { sync: true },
    ));
  }
  if (!readOnlyPeriodic && els.tileCullingToggle.checked && state.webgpu.renderer?.trainState) {
    const includeTileDistribution = Boolean(
      performanceProfileRequested() &&
      performanceProfileLabels(step, state.metrics?.steps_requested || step).length > 0
    );
    if (!reuseMetricRender) {
      await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(state.image, state.params, { sync: true }));
    }
    let tileCounters = await awaitTrainingRun(run, state.webgpu.renderer.readTileCounters({ includeDistribution: includeTileDistribution }));
    const reserveRatio = tileCounters ? tileCounters.total / Math.max(1, tileCounters.capacity) : 0;
    const qaOverflowPending =
      qaTileOverflowFixtureEnabled() &&
      (state.metrics?.steps_done || 0) === 0 &&
      !state.metrics?.tile_retry_parameter_hash?.matches;
    const reserveLevel = reserveRatio >= 0.9 ? 90 : reserveRatio >= 0.8 ? 80 : reserveRatio >= 0.7 ? 70 : 0;
    if (reserveLevel > (state.webgpu.renderer.trainState.tileReserveLevel || 0)) {
      state.webgpu.renderer.trainState.tileReserveLevel = reserveLevel;
      eventLog(`tile reserve ${reserveLevel}% threshold: ${tileCounters.total}/${tileCounters.capacity}`);
    }
    const shouldExpandTileReserve = Boolean(
      !qaOverflowPending && tileCounters && (tileCounters.overflow > 0 || reserveRatio >= 0.8),
    );
    const expandedTileReserve = shouldExpandTileReserve
      ? await awaitTrainingRun(run, state.webgpu.renderer.growTileIndexCapacity(tileCounters.total, { proactive: tileCounters.overflow === 0 }))
      : false;
    if (expandedTileReserve) {
      const previousCapacity = tileCounters.capacity;
      await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(state.image, state.params, { sync: true }));
      tileCounters = await awaitTrainingRun(run, state.webgpu.renderer.readTileCounters({ includeDistribution: includeTileDistribution }));
      log(`tile index capacity expanded ${previousCapacity} -> ${tileCounters.capacity} for ${tileCounters.total} references`);
      eventLog(`tile index capacity expanded ${previousCapacity} -> ${tileCounters.capacity}`);
    }
    if (tileCounters) {
      state.metrics.tile_counters = tileCounters;
      if (!qaOverflowPending && (tileCounters.overflow > 0 || (reserveRatio >= 0.9 && shouldExpandTileReserve && !expandedTileReserve))) {
        const rec = state.recommendation || updateMemoryRecommendation();
        const failure = { context: "tile-culling", reason: "safety_stop_tile_index_overflow", rec };
        state.metrics.stopped = true;
        state.metrics.safety_stop = {
          reason: failure.reason,
          context: failure.context,
          tile_indices: tileCounters.total,
          tile_capacity: tileCounters.capacity,
          tile_overflow: tileCounters.overflow,
          estimated_mb: rec.estimatedMB,
          budget_mb: rec.budgetMB,
        };
        setSafetyStop(failure);
        const error = new Error(`tile index capacity unavailable: ${tileCounters.total}/${tileCounters.capacity}`);
        error.safetyStop = true;
        throw error;
      }
    }
  }
  let trainBuffers = state.webgpu.renderer?.currentTrainBuffers(state.params);
  if (final && state.webgpu.renderer?.trainState) {
    const cpuMirrorAlreadyCurrent =
      state.metrics?.cpu_mirror_step === step &&
      state.metrics?.cpu_mirror_count === state.params.count;
    if (!cpuMirrorAlreadyCurrent) {
      await awaitTrainingRun(run, state.webgpu.renderer.readTrainedColors(state.params));
    }
    assertFiniteParams(state.params, "final-readback");
    if (previewInvariantHashEnabled()) {
      state.metrics.final_parameter_hash = hashParams(state.params);
    }
    if (final) {
      state.metrics.thin_line_metrics = computeThinLineMetrics(state.image, state.params);
      state.metrics.tilt_risk = summarizeTiltRisk(state.params, state.image);
      state.metrics.virtual_depth_stats = virtualDepthDistribution(state.params);
      if (state.params.surfaceLayerPriorEnabled && state.metrics.surface_layer_prior) {
        state.metrics.surface_layer_prior.final_assignment =
          summarizeScaleBiasedSurfaceLayerSort(state.params);
      }
      state.metrics.final_readback_step = step;
    }
  }
  const cpuMirrorCurrent = final || step === 0;
  state.metrics.cpu_mirror_current = cpuMirrorCurrent;
  if (cpuMirrorCurrent) {
    state.metrics.param_delta = paramDeltaFromSnapshot(state.metrics.initial_param_snapshot, state.params);
  }
  let metrics = { loss: Number.NaN, ssim: Number.NaN };
  els.backendText.textContent = backend;
  try {
    // Padded preview is display-only and never runs while the optimizer owns the live buffers.
    const outsidePreviewActive = present && els.outsidePreviewToggle.checked && !state.running;
    if (outsidePreviewActive) {
      await awaitTrainingRun(run, state.webgpu.renderer.render(state.image, state.params, trainBuffers));
    }
    metrics = await awaitTrainingRun(run, state.webgpu.renderer.computeMetrics(
      state.image,
      state.params,
      trainBuffers,
      { reuseCurrentRender: reuseMetricRender },
    ));
    const previousMetricReuse = state.metrics.metric_tile_reuse || {};
    state.metrics.metric_tile_reuse = {
      requested: readOnlyPeriodic || performanceVariants().metricTileReuse,
      last_applied: Boolean(metrics.reusedRender),
      applied_count: (previousMetricReuse.applied_count || 0) + (metrics.reusedRender ? 1 : 0),
      fallback_count: (previousMetricReuse.fallback_count || 0) + (metrics.reusedRender ? 0 : 1),
      lag_iterations: metrics.reusedRender ? 1 : previousMetricReuse.lag_iterations || 0,
    };
    const restoreTrainingStage = !final
      ? state.webgpu.renderer.lastTrainStats?.training_stage
      : "full";
    const restoreStageImage = restoreTrainingStage === "coarse"
      ? state.webgpu.renderer.trainState?.coarseImage
      : restoreTrainingStage === "mid"
        ? state.webgpu.renderer.trainState?.midImage
        : null;
    if (restoreStageImage && !metrics.reusedRender) {
      if (els.tileCullingToggle.checked) {
        await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(restoreStageImage, state.params, { sync: true }));
      }
      await awaitTrainingRun(run, state.webgpu.renderer.refreshRenderState(restoreStageImage, state.params));
      state.metrics.preview_resolution_restores += 1;
    }
    if (present && !outsidePreviewActive) state.webgpu.renderer.presentTrainState(state.image);
    if (present && state.previewMode === "splats") showCanvas("gpu");
    if (present) {
      state.metrics.preview_frames += 1;
      state.metrics.last_preview_step = step;
    }
    const finiteMetrics = [
      metrics.loss,
      metrics.mse,
      metrics.psnr,
      metrics.alphaL1,
      metrics.alphaSsim,
      metrics.alphaObjective,
      metrics.objectiveLoss,
      metrics.ssim,
      metrics.windowedSsim,
      metrics.regionalSsim?.minimum,
      metrics.regionalSsim?.p10,
      metrics.coverage?.mean,
      metrics.coverage?.minimum,
    ];
    if (finiteMetrics.some((value) => !Number.isFinite(value))) {
      throw runtimeSafetyError("safety_stop_nonfinite_metrics", `metrics-step-${step}`);
    }
    if (final) {
      const brushContributionSettings = brushContributionDiagnosticsSettings(
        ALGORITHM_REGISTRY[state.metrics?.algorithm],
      );
      if (brushContributionSettings.enabled) {
        const report = await awaitTrainingRun(
          run,
          state.webgpu.renderer.computeBrushContributionDiagnostics(
            state.image,
            state.params,
            brushContributionSettings,
          ),
        );
        const repeatedReport = await awaitTrainingRun(
          run,
          state.webgpu.renderer.computeBrushContributionDiagnostics(
            state.image,
            state.params,
            brushContributionSettings,
          ),
        );
        const deterministicFields = [
          "valid_pixels",
          "flat_pixels",
          "flat_contributing_pixels",
          "high_neff_pixels",
          "mean_accepted_contributors",
          "mean_physical_effective_contributors",
          "mean_renyi2_entropy",
          "mean_composited_alpha",
        ];
        report.repeat_deltas = Object.fromEntries(deterministicFields.map((key) => [
          key,
          Number(repeatedReport?.[key]) - Number(report?.[key]),
        ]));
        report.repeat_deterministic = deterministicFields.every(
          (key) => Object.is(repeatedReport?.[key], report?.[key]),
        );
        report.repeat_parameter_hash_matches = Boolean(
          report.parameter_hash_matches && repeatedReport?.parameter_hash_matches &&
          repeatedReport?.parameter_hash_after === report.parameter_hash_after,
        );
        const coverageMean = Number(metrics.coverage?.mean);
        if (report && Number.isFinite(coverageMean)) {
          report.final_metric_coverage_mean = coverageMean;
          report.final_metric_alpha_mean_delta = report.mean_composited_alpha - coverageMean;
        }
        state.metrics.brush_contribution_diagnostics = report;
      }
      if (state.metrics.virtual_camera_sampling?.enabled) {
        setTrainingMessage(`Evaluating ${state.metrics.virtual_camera_sampling.virtual_camera_count} virtual camera teachers on WebGPU...`);
        state.metrics.virtual_camera_evaluation = await awaitTrainingRun(run, state.webgpu.renderer.computeVirtualCameraEvaluation(
          state.image,
          state.params,
          metrics,
        ));
      }
      const explicitFinalAudit = finalRenderAuditEnabled();
      state.metrics.overlap_diagnostics =
        !algorithmUsesPaintKernel() &&
        (explicitFinalAudit || layerEfficiencyDiagnosticsRequested())
        ? await awaitTrainingRun(run, state.webgpu.renderer.computeOverlapDiagnostics(state.image, state.params))
        : null;
      if (!algorithmUsesPaintKernel() && layerEfficiencyDiagnosticsRequested()) {
        if (els.tileCullingToggle.checked) {
          await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(state.image, state.params, { sync: true }));
        }
        const tileDiagnostics = await awaitTrainingRun(run, state.webgpu.renderer.readLayerTileDiagnostics(state.params));
        state.metrics.layer_efficiency = summarizeLayerEfficiency(
          state.params,
          state.metrics.overlap_diagnostics?.hidden_rgb_attribution || null,
          tileDiagnostics,
          state.layerEfficiencyCheckpoints,
          state.metrics.initial_splats,
        );
      }
      if (QA_RUNTIME_ENABLED && state.metrics.virtual_camera_sampling?.enabled) {
        state.metrics.oblique_overlap_diagnostics = await awaitTrainingRun(run, state.webgpu.renderer.computeObliqueDiagnostics(state.image, state.params));
      }
      // Export is result-bound, so product finalization verifies the training
      // surface against the standalone renderer once. This is the only product
      // full-frame quality readback beyond final metrics.
      const trainingFrame = await awaitTrainingRun(run, state.webgpu.renderer.capturePresentedStateRgba());
      if (!trainingFrame || trainingFrame.width !== state.image.width || trainingFrame.height !== state.image.height) {
        throw new Error("Final training RGBA readback has the wrong resolution.");
      }
      const standaloneRgba = await awaitTrainingRun(run, state.webgpu.renderer.captureFrameRgba(
        state.image,
        state.params,
        trainBuffers,
      ));
      state.metrics.render_surface_parity = {
        ...displayRgbaParity(trainingFrame.rgba, standaloneRgba),
        source: "training-pixel-state-vs-standalone-rgba",
        width: trainingFrame.width,
        height: trainingFrame.height,
      };
      if (explicitFinalAudit) {
        const auditParams = state.params;
        if (auditParams.layerAwareAccumulationEnabled) {
          auditParams.layerAwareAccumulationEnabled = false;
          try {
            const ordinaryAlphaRgba = await awaitTrainingRun(run, state.webgpu.renderer.captureFrameRgba(
              state.image,
              auditParams,
              trainBuffers,
            ));
            state.metrics.layer_aware_render_parity = {
              ...displayRgbaParity(ordinaryAlphaRgba, standaloneRgba),
              source: "same-params-standard-alpha-off-vs-layer-aware-on",
            };
          } finally {
            auditParams.layerAwareAccumulationEnabled = true;
          }
        } else {
          state.metrics.layer_aware_render_parity = null;
        }
        state.metrics.color_space_audit = trainingColorSpaceAudit(
          state.image,
          state.webgpu.renderer.trainState,
          trainingFrame.rgba,
          standaloneRgba,
        );
        if (!state.metrics.importance_stats) {
          state.metrics.importance_stats = await awaitTrainingRun(run, state.webgpu.renderer.readImportanceSummary(state.params.count));
        }
        if (state.metrics.importance_stats?.nonfinite_count > 0) {
          throw runtimeSafetyError("safety_stop_nonfinite_importance", "final-importance-readback", {
            nonfinite_stats: state.metrics.importance_stats.nonfinite_count,
          });
        }
      } else {
        state.metrics.layer_aware_render_parity = null;
        state.metrics.color_space_audit = {
          contract: "front and virtual teachers use the same sRGB signal values",
          skipped: true,
          reason: "extended-color-audit-is-qa-only",
        };
      }
    }
    const densityCounters = await awaitTrainingRun(run, state.webgpu.renderer.readDensityCounters());
    if (densityCounters) {
      const previous = state.metrics.last_density_counters || emptyFusionEvents();
      const relocated = densityCounters.mcmc_teleport - (previous.mcmc_teleport || 0);
      state.metrics.density_counters = densityCounters;
      state.metrics.last_density_counters = { ...densityCounters };
      state.metrics.fusion_events.adc_duplicate = densityCounters.adc_duplicate;
      state.metrics.fusion_events.adc_split = densityCounters.adc_split;
      state.metrics.fusion_events.adc_recycle = densityCounters.adc_recycle;
      state.metrics.fusion_events.mcmc_teleport = densityCounters.mcmc_teleport;
      state.metrics.fusion_events.mcmc_reseed = densityCounters.mcmc_reseed + densityCounters.mcmc_teleport;
      state.metrics.fusion_events.inactive_reused = densityCounters.inactive_reused;
      state.metrics.fusion_events.opacity_reset = densityCounters.opacity_reset;
      state.metrics.fusion_events.prune = densityCounters.prune;
      state.metrics.fusion_events.importance_protected = densityCounters.importance_protected;
      state.metrics.fusion_events.adc_eligible = densityCounters.adc_eligible;
      state.metrics.fusion_events.adc_fallback = densityCounters.adc_fallback;
      state.metrics.fusion_events.structure_guided = densityCounters.structure_guided;
      state.metrics.fusion_events.nonfinite_stats = densityCounters.nonfinite_stats;
      state.metrics.fusion_events.adc_low_to_high = densityCounters.adc_low_to_high;
      state.metrics.fusion_events.adc_high_to_low = densityCounters.adc_high_to_low;
      state.metrics.fusion_events.adc_same_band = densityCounters.adc_same_band;
      state.metrics.fusion_events.source_claim_conflicts = densityCounters.source_claim_conflicts;
      state.metrics.fusion_events.source_claims = densityCounters.source_claims;
      state.metrics.fusion_events.tilt_risk_candidates = densityCounters.tilt_risk_candidates;
      state.metrics.fusion_events.tilt_true_splits = densityCounters.tilt_true_splits;
      state.metrics.fusion_events.tilt_opacity_saturations = densityCounters.tilt_opacity_saturations;
      state.metrics.fusion_events.paint_outlier_recycle = densityCounters.paint_outlier_recycle;
      state.metrics.fusion_events.paint_outlier_recolor = densityCounters.paint_outlier_recolor;
      state.metrics.fusion_events.paint_outlier_trim = densityCounters.paint_outlier_trim;
      state.metrics.fusion_events.surface_layer_candidates = densityCounters.surface_layer_candidates;
      state.metrics.fusion_events.surface_layer_promotions = densityCounters.surface_layer_promotions;
      state.metrics.fusion_events.harmful_rectangle_candidate_selections =
        densityCounters.harmful_rectangle_candidate_selections;
      state.metrics.fusion_events.harmful_rectangle_front_oversized_selections =
        densityCounters.harmful_rectangle_front_oversized_selections;
      state.metrics.fusion_events.harmful_rectangle_high_contribution_selections =
        densityCounters.harmful_rectangle_high_contribution_selections;
      state.metrics.fusion_events.harmful_rectangle_high_deviation_selections =
        densityCounters.harmful_rectangle_high_deviation_selections;
      state.metrics.fusion_events.harmful_rectangle_parent_replacements =
        densityCounters.harmful_rectangle_parent_replacements;
      state.metrics.fusion_events.harmful_rectangle_children_created =
        densityCounters.harmful_rectangle_children_created;
      if (state.metrics.front_footprint_refinement_v2) {
        // Growth events reset the shared density counters. The report above is
        // cumulative; a later relocation/finalization read must not erase it.
        state.metrics.front_footprint_refinement_v2.last_density_counters = {
          candidate_selections: densityCounters.harmful_rectangle_candidate_selections,
          parent_replacements: densityCounters.harmful_rectangle_parent_replacements,
          children_created: densityCounters.harmful_rectangle_children_created,
        };
      }
      if (densityCounters.nonfinite_stats > 0) {
        throw runtimeSafetyError("safety_stop_nonfinite_density", `density-step-${step}`, {
          nonfinite_stats: densityCounters.nonfinite_stats,
        });
      }
      if (relocated > 0) {
        const event = { step, moved: relocated, cumulative: densityCounters.mcmc_teleport };
        if (state.metrics.webgpu_relocation_events.length < 96) state.metrics.webgpu_relocation_events.push(event);
        if (state.metrics.webgpu_refine_events.length < 96) state.metrics.webgpu_refine_events.push(event);
      }
    }
    state.lastGpuLoss = metrics.loss;
    state.metrics.webgpu_compute_loss = true;
    state.metrics.last_gpu_loss = metrics.loss;
    state.metrics.last_alpha_l1 = metrics.alphaL1;
    state.metrics.last_alpha_ssim = metrics.alphaSsim;
    state.metrics.last_objective_loss = metrics.objectiveLoss;
    state.metrics.webgpu_loss_stats = state.webgpu.renderer.lastLossStats;
    state.metrics.webgpu_train_executed = Boolean(state.webgpu.renderer.lastTrainStats);
    state.metrics.webgpu_train_update = Boolean(state.webgpu.renderer.lastTrainStats?.updated);
    state.metrics.webgpu_train_stats = state.webgpu.renderer.lastTrainStats;
    state.metrics.stage_profile = state.webgpu.renderer.trainState?.stageProfile?.map((sample) => ({ ...sample })) || [];
    state.metrics.stage_profile_backend = state.webgpu.profile?.timing_backend || "off";
    state.metrics.scheduling_profile = summarizeTrainingScheduling(
      state.metrics.stage_profile,
      state.metrics.performance_trace,
    );
  } catch (error) {
    if (error.trainingRunCancelled || (run && !ownsTrainingRun(run))) throw error;
    if (error.safetyStop) throw error;
    state.lastGpuLoss = null;
    state.metrics.webgpu_compute_loss = false;
    state.metrics.webgpu_compute_error = error.message;
    throw new Error(`WebGPU preview/metrics failed: ${error.message}`);
  }
  state.metrics.losses.push(metrics.loss);
  state.metrics.rgb_mse.push(metrics.mse);
  state.metrics.psnr.push(metrics.psnr);
  state.metrics.alpha_losses.push(metrics.alphaL1);
  state.metrics.alpha_ssim.push(metrics.alphaSsim);
  state.metrics.objective_losses.push(metrics.objectiveLoss);
  state.metrics.ssim.push(metrics.windowedSsim);
  state.metrics.global_ssim.push(metrics.ssim);
  state.metrics.windowed_ssim.push(metrics.windowedSsim);
  state.metrics.regional_ssim_p10.push(metrics.regionalSsim.p10);
  Object.assign(state.metrics, {
    latest_evaluation_step: step,
    latest_l1: metrics.loss,
    latest_rgb_mse: metrics.mse,
    latest_psnr: metrics.psnr,
    latest_alpha_l1: metrics.alphaL1,
    latest_alpha_ssim: metrics.alphaSsim,
    latest_alpha_objective: metrics.alphaObjective,
    latest_objective_loss: metrics.objectiveLoss,
    latest_ssim: metrics.windowedSsim,
    latest_global_ssim: metrics.ssim,
    latest_windowed_ssim: metrics.windowedSsim,
    latest_regional_ssim: metrics.regionalSsim,
    latest_high_frequency: metrics.highFrequency,
  });
  if (final) {
    Object.assign(state.metrics, {
      final_evaluation_step: step,
      final_metrics_complete: true,
      final_l1: metrics.loss,
      final_rgb_mse: metrics.mse,
      final_psnr: metrics.psnr,
      final_alpha_l1: metrics.alphaL1,
      final_alpha_ssim: metrics.alphaSsim,
      final_alpha_objective: metrics.alphaObjective,
      final_objective_loss: metrics.objectiveLoss,
      final_ssim: metrics.windowedSsim,
      final_global_ssim: metrics.ssim,
      final_windowed_ssim: metrics.windowedSsim,
      final_regional_ssim: metrics.regionalSsim,
      final_high_frequency: metrics.highFrequency,
    });
  }
  state.metrics.coverage_stats = metrics.coverage ? { ...metrics.coverage, step } : null;
  state.metrics.coverage_revision = state.metrics.params_revision ?? 0;
  linkDensityEventMetricSnapshot(step, metrics);
  if (state.metrics.initial_ssim !== null) {
    const delta = metrics.windowedSsim - state.metrics.initial_ssim;
    state.metrics.ssim_trend = delta > 0.0005 ? "up" : delta < -0.0005 ? "down" : "flat";
  }
  if (state.metrics.initial_global_ssim !== null) {
    const delta = metrics.ssim - state.metrics.initial_global_ssim;
    state.metrics.global_ssim_trend = delta > 0.0005 ? "up" : delta < -0.0005 ? "down" : "flat";
  }
  if (state.metrics.initial_psnr !== null) {
    const delta = metrics.psnr - state.metrics.initial_psnr;
    state.metrics.psnr_trend = delta > 0.05 ? "up" : delta < -0.05 ? "down" : "flat";
  }
  const boundary = cpuMirrorCurrent ? boundaryLeakStats(state.params) : null;
  const outsideRender = cpuMirrorCurrent ? outsideRenderFootprintStats(state.params) : null;
  const shape = cpuMirrorCurrent ? splatShapeStats(state.params, state.image) : null;
  state.metrics.boundary_sigma = state.params.boundarySigma ?? selectedBoundarySigma();
  if (cpuMirrorCurrent) {
    state.metrics.boundary_leak_count = boundary.count;
    state.metrics.boundary_max_leak = boundary.maxLeak;
    state.metrics.outside_render_splat_count = outsideRender.count;
    state.metrics.outside_render_max_extent = outsideRender.maxLeak;
    state.metrics.shape_stats = shape;
    if (final) {
      state.metrics.source_detail_splat_distribution = sourceDetailSplatDistribution(
        state.image,
        state.params,
      );
    }
    state.metrics.scale_histogram = shape?.scale_histogram || null;
    state.metrics.tiny_splat_count = shape?.tiny_splat_count ?? null;
    state.metrics.tiny_splat_ratio = shape?.tiny_splat_ratio ?? null;
    state.metrics.boundary_tiny_splat_count = shape?.boundary_tiny_splat_count ?? null;
    state.metrics.boundary_tiny_splat_ratio = shape?.boundary_tiny_splat_ratio ?? null;
    state.metrics.interior_tiny_splat_count = shape?.interior_tiny_splat_count ?? null;
    state.metrics.interior_tiny_splat_ratio = shape?.interior_tiny_splat_ratio ?? null;
    state.metrics.anisotropy_ratio = shape
      ? { mean: shape.anisotropy_ratio_mean, max: shape.anisotropy_ratio_max, elongated_count: shape.elongated_splat_count }
      : null;
  }
  state.metrics.steps_done = step;
  state.metrics.num_gaussians = state.params.count;
  state.metrics.loss_backend = "webgpu-compute";
  if (state.metrics.trend_checkpoints.length < 256) {
    state.metrics.trend_checkpoints.push({
      step,
      loss: metrics.loss,
      rgb_mse: metrics.mse,
      psnr: metrics.psnr,
      alpha_l1: metrics.alphaL1,
      alpha_ssim: metrics.alphaSsim,
      alpha_objective: metrics.alphaObjective,
      objective_loss: metrics.objectiveLoss,
      ssim: metrics.windowedSsim,
      global_ssim: metrics.ssim,
      regional_ssim: {
        minimum: metrics.regionalSsim.minimum,
        p10: metrics.regionalSsim.p10,
        median: metrics.regionalSsim.median,
        mean: metrics.regionalSsim.mean,
        worst_region: metrics.regionalSsim.worst_region,
      },
      high_frequency: metrics.highFrequency,
      boundary_leak_count: boundary?.count ?? null,
      boundary_max_leak: boundary?.maxLeak ?? null,
      outside_render_splat_count: outsideRender?.count ?? null,
      tiny_splat_count: shape?.tiny_splat_count ?? null,
      anisotropy_ratio_max: shape?.anisotropy_ratio_max ?? null,
      coverage_under_ratio: metrics.coverage?.under_ratio ?? null,
      background_exposure_count: metrics.coverage?.background_exposure_count ?? null,
      gaussians: state.params.count,
    });
  }
  els.stepText.textContent = `${step} / ${state.metrics.steps_requested}`;
  els.lossText.textContent = metrics.loss.toFixed(6);
  syncDisplayedSsimMetrics();
  els.boundaryText.textContent = boundary ? `${boundary.count} / ${boundary.maxLeak.toFixed(6)}` : "-";
  const coveragePairReady = Boolean(boundary && Number.isFinite(metrics.coverage?.background_exposure_count));
  els.coverageText.textContent = coveragePairReady
    ? `${(metrics.coverage.background_exposure_ratio * 100).toFixed(2)}% / ${outsideRender.count.toLocaleString()}`
    : "- / -";
  renderSplatInspector();
  publishState();
  if (final) {
    const label = state.metrics?.stopped ? "stopped" : "finished";
    const worst = metrics.regionalSsim.worst_region;
    log(`${label} loss=${metrics.loss.toFixed(6)} psnr=${metrics.psnr.toFixed(2)}dB alpha_l1=${metrics.alphaL1.toFixed(6)} alpha_ssim=${metrics.alphaSsim.toFixed(6)} objective=${metrics.objectiveLoss.toFixed(6)} global_ssim=${metrics.ssim.toFixed(6)} windowed_ssim=${metrics.windowedSsim.toFixed(6)} local_p10=${metrics.regionalSsim.p10.toFixed(6)} worst_region=${worst?.column ?? "-"},${worst?.row ?? "-"}`);
    const virtualEvaluation = state.metrics.virtual_camera_evaluation;
    if (virtualEvaluation) {
      const virtual = virtualEvaluation.virtual_views;
      const allViews = virtualEvaluation.all_views;
      log(`virtual teacher evaluation cameras=${virtual.camera_count} virtual_psnr=${virtual.rgb_psnr_macro?.toFixed(2) ?? "-"}dB virtual_ssim=${virtual.rgb_ssim_macro?.toFixed(6) ?? "-"} virtual_p10=${virtual.rgb_ssim_p10?.toFixed(6) ?? "-"} all_view_psnr=${allViews.rgb_psnr_macro?.toFixed(2) ?? "-"}dB all_view_ssim=${allViews.rgb_ssim_macro?.toFixed(6) ?? "-"} all_view_p10=${allViews.rgb_ssim_p10?.toFixed(6) ?? "-"}`);
    }
  }
  return metrics.loss;
}

async function startTraining() {
  if (!state.image) {
    setTrainingMessage("Load an image before training.", "error");
    setStatus("no image");
    return false;
  }
  if (state.running || state.startPending || state.webGpuRecoveryPending) return false;
  const run = beginTrainingRun();
  state.startPending = true;
  setInputControlsDisabled(true);
  setTrainingMessage(state.previewRefreshPending ? "Waiting for the preview to finish..." : "Preparing WebGPU training...");
  publishLifecycleInteractionState();
  try {
    if (state.previewRefreshPending) {
      eventLog("Train requested while preview refresh was pending; waiting for the preview contract.");
      await awaitTrainingRun(run, state.previewRefreshPromise);
    }
    if (!ownsTrainingRun(run) || !state.image || state.running) return false;
    await awaitTrainingRun(run, selectedAlgorithm().train(run));
    return ownsTrainingRun(run);
  } catch (error) {
    if (error.trainingRunCancelled || !ownsTrainingRun(run)) return false;
    run.renderer?.disposeTrainState();
    state.running = false;
    state.paused = false;
    setInputControlsDisabled(false);
    setPausedRuntimeControlsEnabled(false);
    setStatus("error");
    setTrainingMessage(`Training failed: ${error.message}`, "error");
    log(`training start failed: ${error.message}`);
    eventLog(`training start failed: ${error.message}`);
    return false;
  } finally {
    if (state.trainingRun === run) {
      state.startPending = false;
      state.trainingRun = null;
      publishLifecycleInteractionState();
      if (
        state.previewMode === "splats" &&
        state.image &&
        state.params &&
        state.metrics?.final_cpu_result_ready_at &&
        state.webgpu.renderer &&
        !state.webgpu.renderer.deviceLost
      ) {
        try {
          await refreshOutsidePreview();
        } catch (error) {
          log(`post-training preview refresh failed: ${error.message}`);
        }
      }
    }
  }
}

async function trainPlanarGaussian(run) {
  return trainGaussianAlgorithm(false, run);
}

async function trainRectangleSplats(run) {
  return trainGaussianAlgorithm(false, run);
}

async function trainLayeredOpaqueBrush(run) {
  return trainGaussianAlgorithm(false, run);
}

async function trainGsVirtualCameraSampling(run) {
  return trainGaussianAlgorithm(true, run);
}


function stopTraining() {
  if (!state.running) return;
  state.stopRequested = true;
  state.paused = false;
  setPausedRuntimeControlsEnabled(false);
  setStatus("stopping");
  eventLog("stop requested");
}

function togglePause() {
  if (!state.running) return;
  state.paused = !state.paused;
  state.visibilityPaused = false;
  if (state.paused) {
    setPausedRuntimeControlsEnabled(true);
  } else {
    setPausedRuntimeControlsEnabled(false);
    state.runtimeSettingsRevision += 1;
  }
  setStatus(state.paused ? "paused" : "running");
  eventLog(state.paused ? "training paused" : "training resumed");
}

function resetTrainingState() {
  if (!state.image || state.running) return;
  invalidateTrainingRun("training reset");
  cancelCompletedResultGpuRecovery();
  destroyTiltViewer({ restoreCanvas: true });
  state.webgpu.renderer?.disposeTrainState();
  state.webgpu.renderer?.disposeResultRenderState();
  state.params = null;
  state.metrics = null;
  state.previewPadding = previewPaddingSpec(state.image, null, false);
  state.paused = false;
  state.stopRequested = false;
  state.lastGpuLoss = null;
  els.stepText.textContent = "0 / 0";
  els.lossText.textContent = "-";
  els.psnrText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.coverageText.textContent = "- / -";
  resetTrainingTiming();
  clearSplatAdjustmentBaseline();
  setPreviewMode("original");
  updateDownloads(false);
  updateMemoryRecommendation();
  updateVirtualCameraCoverageEstimate();
  setStatus("image loaded");
  setTrainingMessage("Reset to the loaded image.", "success");
  log("reset splats to loaded image");
}

function clearImage() {
  if (state.running) return;
  invalidateTrainingRun("image cleared");
  cancelCompletedResultGpuRecovery();
  destroyTiltViewer({ restoreCanvas: true });
  state.imageLoadGeneration += 1;
  state.imageLoading = false;
  state.webgpu.renderer?.disposeTrainState();
  state.webgpu.renderer?.disposeResultRenderState();
  releaseImageSource(state.image);
  state.image = null;
  state.params = null;
  state.metrics = null;
  state.paused = false;
  state.stopRequested = false;
  state.lastGpuLoss = null;
  state.previewMode = "original";
  state.previewPadding = { x: 0, y: 0, width: 640, height: 420, bytes: 0 };
  els.fileInput.value = "";
  els.dropZone.classList.remove("ready", "dragover");
  els.previewCanvas.width = 640;
  els.previewCanvas.height = 420;
  els.gpuCanvas.width = 640;
  els.gpuCanvas.height = 420;
  previewCtx.clearRect(0, 0, els.previewCanvas.width, els.previewCanvas.height);
  els.gpuCanvas.getContext("2d")?.clearRect(0, 0, els.gpuCanvas.width, els.gpuCanvas.height);
  showCanvas("preview");
  updatePreviewModeControls();
  fitCanvases();
  els.stepText.textContent = "0 / 0";
  els.lossText.textContent = "-";
  els.psnrText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.coverageText.textContent = "- / -";
  resetTrainingTiming();
  els.splatText.textContent = "-";
  updateImageSizeStatus();
  clearSplatAdjustmentBaseline();
  updateDownloads(false);
  updateMemoryRecommendation();
  updateVirtualCameraCoverageEstimate();
  syncAlgorithmRequirements();
  setStatus("idle");
  setTrainingMessage("Ready.");
  eventLog("image cleared");
}

function confirmClearImage() {
  if (!state.image || state.running) return;
  if (!window.confirm("Clear the loaded image and its training result?")) return;
  clearImage();
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function updateDownloads(enabled) {
  const requested = Boolean(enabled);
  const coverage = requested ? exportCoverageStatus() : null;
  state.exportReady = Boolean(requested && coverage?.ok);
  if (state.exportReady && !state.exporting && !state.exportMessage.startsWith("Exported")) {
    state.exportMessage = coverage?.warning
      ? `Ready to export with warning: ${coverage.message}.`
      : "Ready to export.";
  } else if (requested && !state.exporting && coverage && !coverage.ok) {
    state.exportMessage = `Export unavailable: ${coverage.message}.`;
  } else if (!state.exportReady && !state.exporting) {
    state.exportMessage = "Finish training before export.";
  }
  updateExportPanel();
  publishState();
}

const EXPORT_FORMATS = {
  ply: {
    label: "PLY",
    filename: "image2splatpaint.ply",
    description: "PLY uses standard SH0 Gaussian Splatting fields. For conventional multi-angle 3DGS viewing, train with GS Virtual Camera Sampling; Planar Gaussian is front-view only.",
  },
  png: {
    label: "Splat PNG",
    filename: "image2splatpaint-splats.png",
    description: "Current Splats preview without UI zoom, pan, or controls.",
  },
};

function selectedPngExportResolutionMode() {
  const value = els.pngExportResolution?.value;
  return ["training", "2048", "4096", "custom"].includes(value)
    ? value
    : DEFAULT_PNG_EXPORT_RESOLUTION;
}

function pngExportFrameSpec(image = state.image) {
  const trainingWidth = Math.max(1, Math.round(Number(image?.width) || Number(els.trainSize?.value) || DEFAULT_MAX_SIDE));
  const trainingHeight = Math.max(1, Math.round(Number(image?.height) || trainingWidth));
  const trainingLongSide = Math.max(trainingWidth, trainingHeight);
  const mode = selectedPngExportResolutionMode();
  const requestedLongSide = mode === "training"
    ? trainingLongSide
    : mode === "custom"
      ? Number(els.pngExportLongSide?.value)
      : Number(mode);
  const longSide = Math.round(clampNumber(
    requestedLongSide,
    MIN_PNG_EXPORT_LONG_SIDE,
    MAX_PNG_EXPORT_LONG_SIDE,
    trainingLongSide,
  ));
  if (mode === "training") {
    return { mode, longSide: trainingLongSide, width: trainingWidth, height: trainingHeight };
  }
  const scale = longSide / trainingLongSide;
  return {
    mode,
    longSide,
    width: Math.max(1, Math.round(trainingWidth * scale)),
    height: Math.max(1, Math.round(trainingHeight * scale)),
  };
}

function syncPngExportResolutionUi() {
  const spec = pngExportFrameSpec();
  if (selectedPngExportResolutionMode() !== "custom") {
    els.pngExportLongSide.value = String(spec.longSide);
  } else {
    els.pngExportLongSide.value = String(Math.round(clampNumber(
      els.pngExportLongSide.value,
      MIN_PNG_EXPORT_LONG_SIDE,
      MAX_PNG_EXPORT_LONG_SIDE,
      spec.longSide,
    )));
  }
  els.pngExportResolution.disabled = state.exporting;
  els.pngExportLongSide.disabled = state.exporting;
  els.pngExportResolutionStatus.textContent = `${spec.width} x ${spec.height}px`;
  return spec;
}

function currentSplatPngSpec() {
  const renderOptions = {
    ...splatAlphaRenderOptions(),
    outside: Boolean(els.outsidePreviewToggle?.checked),
  };
  const shape = renderOptions.splatShape === "opaque-brush"
    ? "opaque-brush"
    : ["rectangle", "box"].includes(renderOptions.splatShape)
      ? "rectangle"
      : "gaussian";
  return {
    filename: `image2splatpaint-splats-${shape}.png`,
    shape,
    renderOptions,
  };
}

function updateExportPanel() {
  const pngFrame = syncPngExportResolutionUi();
  const enabled = state.exportReady && !state.exporting;
  const algorithm = displayedResultAlgorithm();
  const plySupported = algorithmSupportsExport("ply", algorithm);
  const pngSupported = algorithmSupportsExport("png", algorithm);
  const plyPlan = plySupported && state.params && state.image
    ? plyExportMemoryPlan(state.params, state.image, { download: true })
    : null;
  const plyEnabled = enabled && plySupported && Boolean(plyPlan?.ok);
  const pngEnabled = enabled && pngSupported;
  els.savePngButton.disabled = !pngEnabled;
  els.savePlyButton.disabled = !plyEnabled;
  els.savePngButton.textContent = state.exporting ? "Saving..." : "Save Splat PNG";
  els.savePlyButton.textContent = state.exporting ? "Exporting..." : "Export PLY";
  els.exportDescription.textContent = algorithm.capabilities.kernelShape === "opaque-brush"
    ? "Brush Splats export exactly as PNG. Standard Gaussian Splatting PLY cannot represent the analytic brush kernel."
    : algorithm.capabilities.kernelShape === "rectangle"
      ? "Rectangle Splats export exactly as PNG. Standard Gaussian Splatting PLY cannot represent the rectangular kernel."
      : EXPORT_FORMATS.ply.description;
  els.exportCount.textContent = state.params ? state.params.count.toLocaleString() : "-";
  els.exportStatus.textContent = enabled && plyPlan && !plyPlan.ok
    ? `PNG is ready. PLY needs ${plyPlan.estimatedPeakMB} MB peak memory; ${plyPlan.reason}.`
    : state.exportMessage;
  const data = document.documentElement.dataset;
  data.exportReady = String(enabled);
  data.exportResultAlgorithm = algorithm.id;
  data.pngExportReady = String(pngEnabled);
  data.pngExportResolution = pngFrame.mode;
  data.pngExportLongSide = String(pngFrame.longSide);
  data.pngExportWidth = String(pngFrame.width);
  data.pngExportHeight = String(pngFrame.height);
  data.plyExportReady = String(plyEnabled);
  data.plyExportPeakMb = plyPlan?.estimatedPeakMB || "";
  data.plyExportBudgetMb = plyPlan?.budgetMB || "";
}

function exportCoverageStatus(metrics = state.metrics) {
  if (metrics?.safety_stop) {
    return { ok: false, reason: "safety_stop", message: "training ended with a safety stop" };
  }
  const coverage = metrics?.coverage_stats;
  const exposedPixels = coverage?.background_exposure_count;
  const coverageCurrent = coverage?.step === metrics?.steps_done;
  const revisionCurrent =
    metrics?.params_revision === undefined || metrics?.coverage_revision === metrics?.params_revision;
  if (!revisionCurrent) {
    return { ok: false, reason: "coverage_stale", message: "coverage is stale for the current splat revision" };
  }
  if (!coverageCurrent || typeof exposedPixels !== "number" || !Number.isFinite(exposedPixels)) {
    return { ok: false, reason: "coverage_missing", message: "final coverage was not measured" };
  }
  const parity = metrics?.render_surface_parity;
  if (
    !parity ||
    !Number.isFinite(parity.max_abs) ||
    !Number.isFinite(parity.mean_abs) ||
    typeof parity.display_equivalent !== "boolean"
  ) {
    return { ok: false, reason: "render_parity_missing", message: "final render parity was not measured" };
  }
  if (!parity.display_equivalent) {
    return {
      ok: false,
      reason: "render_parity_mismatch",
      message:
        `training and export render surfaces differ: alpha ${parity.alpha_max_abs}, ` +
        `premultiplied max ${parity.premultiplied_max_abs}`,
    };
  }
  if (exposedPixels !== 0) {
    return {
      ok: true,
      warning: true,
      reason: "background_exposure",
      exposedPixels,
      message: `background exposed at ${exposedPixels} pixels`,
    };
  }
  return { ok: true, warning: false, reason: "verified", exposedPixels: 0, message: "final coverage verified" };
}

function downloadBlob(name, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  state.lastDownload = `${name}:${blob.size}`;
  log(`download ${name} bytes=${blob.size}`);
  publishState();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function decodeImageBlobRgba(blob, width, height) {
  let source = null;
  let objectUrl = "";
  try {
    if (typeof createImageBitmap === "function") {
      source = await createImageBitmap(blob);
    } else {
      objectUrl = URL.createObjectURL(blob);
      source = await new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("PNG round-trip image decode failed."));
        image.src = objectUrl;
      });
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: true, willReadFrequently: true });
    if (!context) throw new Error("2D canvas is unavailable for PNG round-trip validation.");
    context.clearRect(0, 0, width, height);
    context.drawImage(source, 0, 0, width, height);
    return new Uint8ClampedArray(context.getImageData(0, 0, width, height).data);
  } finally {
    source?.close?.();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}

async function makeSplatPreviewPngBlob() {
  if (!state.image || !state.params || !state.webgpu.renderer) {
    throw new Error("No trained splat result to export.");
  }
  const spec = currentSplatPngSpec();
  const exportFrame = pngExportFrameSpec(state.image);
  const renderImage = exportFrame.mode === "training"
    ? state.image
    : { ...state.image, width: exportFrame.width, height: exportFrame.height };
  const displayParams = state.params;
  const renderBuffers = state.webgpu.renderer.currentResultBuffers(displayParams);
  const capture = await state.webgpu.renderer.captureRenderedRgba(
    renderImage,
    displayParams,
    renderBuffers,
    {
      ...spec.renderOptions,
      rebuildTiles: exportFrame.width !== state.image.width || exportFrame.height !== state.image.height,
    },
  );
  const { rgba, width, height } = capture;
  let nonblackPixels = 0;
  let rgbSum = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    const sum = rgba[i] + rgba[i + 1] + rgba[i + 2];
    rgbSum += sum;
    if (sum > 0) nonblackPixels += 1;
  }
  let sourceNonblackPixels = 0;
  for (let i = 0; i < state.image.rgb.length; i += 3) {
    if (state.image.rgb[i] + state.image.rgb[i + 1] + state.image.rgb[i + 2] > 0) sourceNonblackPixels += 1;
  }
  if (sourceNonblackPixels > 0 && nonblackPixels === 0) {
    throw new Error("PNG frame readback is unexpectedly all black.");
  }
  const frame = document.createElement("canvas");
  frame.width = width;
  frame.height = height;
  const context = frame.getContext("2d", { alpha: true });
  if (!context) throw new Error("2D canvas is unavailable for PNG export.");
  context.putImageData(new ImageData(rgba, width, height), 0, 0);
  const blob = await canvasToBlob(frame);
  const decodedRgba = await decodeImageBlobRgba(blob, width, height);
  const pngRgbaParity = displayRgbaParity(rgba, decodedRgba);
  if (!pngRgbaParity.exact) {
    throw new Error(
      `PNG display RGBA round-trip mismatch: alpha ${pngRgbaParity.alpha_max_abs}, ` +
      `premultiplied max ${pngRgbaParity.premultiplied_max_abs}, mean ${pngRgbaParity.premultiplied_mean_abs}`,
    );
  }
  return {
    blob,
    width,
    height,
    exportFrame,
    spec,
    padding: capture.padding,
    nonblackPixels,
    meanRgb: rgbSum / Math.max(1, width * height * 3 * 255),
    pngRgbaParity,
  };
}

function logit(value) {
  const v = Math.min(Math.max(value, 1e-6), 1 - 1e-6);
  return Math.log(v / (1 - v));
}

async function sha256Hex(buffer) {
  if (!globalThis.crypto?.subtle) return "";
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

const PLY_ROW_BYTES = 17 * 4;
const GPU_READBACK_ROW_BYTES = 12 * 4;
const CPU_PARAMETER_ROW_BYTES = 52;

function parameterArrayBytes(params) {
  if (!params) return 0;
  let bytes = 0;
  let found = false;
  for (const values of [
    params.xy,
    params.scale,
    params.rgb,
    params.opacity,
    params.theta,
    params.depthOrder,
    params.virtualDepth,
    params.brushTaper,
    params.detailTags,
    params.bg,
  ]) {
    if (!ArrayBuffer.isView(values)) continue;
    bytes += values.byteLength;
    found = true;
  }
  return found ? bytes : Math.max(0, Number(params.count) || 0) * CPU_PARAMETER_ROW_BYTES;
}

function imageCpuResidentBytes(image) {
  if (!image) return 0;
  const floatCacheBytes =
    Math.max(0, Number(image.rgb?.byteLength) || 0) +
    Math.max(0, Number(image.alpha?.byteLength) || 0);
  // The bounded canvas is intentionally retained for Original preview and a
  // later training resize. Count it in CPU-side export headroom as well.
  const sourceWidth = Math.max(0, Math.round(Number(image.sourceBitmap?.width) || 0));
  const sourceHeight = Math.max(0, Math.round(Number(image.sourceBitmap?.height) || 0));
  return floatCacheBytes + sourceWidth * sourceHeight * 4;
}

function browserCpuMemoryInfo() {
  const heapLimit = Number(performance?.memory?.jsHeapSizeLimit);
  const heapUsed = Number(performance?.memory?.usedJSHeapSize);
  if (Number.isFinite(heapLimit) && heapLimit > 0 && Number.isFinite(heapUsed) && heapUsed >= 0) {
    return { source: "jsHeapSizeLimit", budgetBytes: heapLimit, usedBytes: heapUsed, exactFree: true };
  }
  const deviceMemoryGB = Number(navigator.deviceMemory);
  if (Number.isFinite(deviceMemoryGB) && deviceMemoryGB > 0) {
    return {
      source: `deviceMemory ${deviceMemoryGB}GB`,
      budgetBytes: clampNumber(deviceMemoryGB * GB * 0.125, 256 * MB, 1024 * MB, 256 * MB),
      usedBytes: 0,
      exactFree: false,
    };
  }
  return { source: "conservative fallback", budgetBytes: 256 * MB, usedBytes: 0, exactFree: false };
}

function plyHeaderText(params, image) {
  const boundarySigma = Number.isFinite(params.boundarySigma) ? params.boundarySigma : selectedBoundarySigma();
  return createPlyHeader({
    count: params.count,
    image,
    boundarySigma,
    layerOrderEnabled: Boolean(params.layerOrderEnabled),
    layerDepthSpan: PLY_LAYER_DEPTH_SPAN,
  });
}

function plyExportMemoryPlan(
  params = state.params,
  image = state.image,
  { download = true, memoryInfo = browserCpuMemoryInfo(), baseline = state.splatBaseline } = {},
) {
  const count = Math.max(0, Number(params?.count) || 0);
  const parameterBytes = parameterArrayBytes(params);
  const baselineBytes = baseline === params
    ? 0
    : baseline === true
      ? count * CPU_PARAMETER_ROW_BYTES
      : parameterArrayBytes(baseline);
  const imageBytes = imageCpuResidentBytes(image);
  const readbackBytes = count * GPU_READBACK_ROW_BYTES;
  const headerBytes = params && image ? new TextEncoder().encode(plyHeaderText(params, image)).byteLength : 1024;
  const plyBytes = headerBytes + count * PLY_ROW_BYTES;
  const blobCopyBytes = download ? plyBytes : 0;
  const residentBytes = parameterBytes + baselineBytes + imageBytes;
  return exportPeakMemoryPlan({
    count,
    parameterBytes,
    baselineBytes,
    imageBytes,
    readbackBytes,
    plyBytes,
    blobCopyBytes,
    residentBytes,
    memoryInfo,
  });
}

function exportPeakMemoryPlan({
  count,
  parameterBytes,
  baselineBytes,
  imageBytes,
  readbackBytes,
  plyBytes,
  blobCopyBytes,
  residentBytes = parameterBytes + baselineBytes + imageBytes,
  memoryInfo,
}) {
  const requiredIncrementBytes = Math.max(readbackBytes, plyBytes + blobCopyBytes);
  const estimatedPeakBytes = residentBytes + requiredIncrementBytes;
  const budgetBytes = Math.max(1, Number(memoryInfo?.budgetBytes) || 256 * MB);
  const availableBytes = Math.max(0, budgetBytes - Math.max(0, Number(memoryInfo?.usedBytes) || 0));
  const ok = memoryInfo?.exactFree
    ? requiredIncrementBytes <= availableBytes * 0.75
    : estimatedPeakBytes <= budgetBytes * 0.9;
  const reason = ok
    ? `${memoryInfo?.source || "memory estimate"} has sufficient headroom`
    : memoryInfo?.exactFree
      ? `available JS heap is ${bytesToMB(availableBytes).toFixed(0)} MB`
      : `${memoryInfo?.source || "memory estimate"} budget is ${bytesToMB(budgetBytes * 0.9).toFixed(0)} MB`;
  return {
    ok,
    count,
    parameterBytes,
    baselineBytes,
    imageBytes,
    readbackBytes,
    plyBytes,
    blobCopyBytes,
    residentBytes,
    requiredIncrementBytes,
    estimatedPeakBytes,
    estimatedPeakMB: bytesToMB(estimatedPeakBytes).toFixed(1),
    budgetBytes,
    budgetMB: bytesToMB(budgetBytes).toFixed(0),
    memorySource: memoryInfo?.source || "unknown",
    reason,
  };
}

function assertPlyExportCapacity(params = state.params, image = state.image, download = true) {
  const plan = plyExportMemoryPlan(params, image, { download });
  if (!plan.ok) {
    throw Object.assign(
      new Error(`PLY export needs ${plan.estimatedPeakMB} MB peak memory; ${plan.reason}`),
      { exportOnly: true, exportMemoryPlan: plan },
    );
  }
  return plan;
}

function assertTiltViewerCapacity(params = state.params, image = state.image) {
  const plan = plyExportMemoryPlan(params, image, { download: false });
  const viewerBytes = plan.plyBytes * 4;
  const requiredIncrementBytes = Math.max(plan.readbackBytes, viewerBytes);
  const estimatedPeakBytes = plan.residentBytes + requiredIncrementBytes;
  const memoryInfo = browserCpuMemoryInfo();
  const availableBytes = Math.max(0, plan.budgetBytes - Math.max(0, Number(memoryInfo.usedBytes) || 0));
  const ok = memoryInfo.exactFree
    ? requiredIncrementBytes <= availableBytes * 0.6
    : estimatedPeakBytes <= plan.budgetBytes * 0.8;
  if (!ok) {
    throw Object.assign(
      new Error(`Tilt view needs about ${bytesToMB(estimatedPeakBytes).toFixed(1)} MB peak memory; reduce Max splats.`),
      { exportOnly: true, tiltMemoryPlan: { ...plan, viewerBytes, requiredIncrementBytes, estimatedPeakBytes } },
    );
  }
  return { ...plan, viewerBytes, requiredIncrementBytes, estimatedPeakBytes };
}

function plyFrameScale(image = state.image) {
  return exportedPlyFrameScale(image);
}

function transformPlanarSplatForPly(x, y, sx, sy, theta, image = state.image) {
  // This is the single NDC-to-isotropic-world conversion, independent of grid initialization.
  return transformPlanarSplat(x, y, sx, sy, theta, image);
}

function sourceRgbAtNdc(image, x, y) {
  const px = Math.min(image.width - 1, Math.max(0, Math.round((Math.min(1, Math.max(-1, x)) * 0.5 + 0.5) * (image.width - 1))));
  const py = Math.min(image.height - 1, Math.max(0, Math.round((Math.min(1, Math.max(-1, y)) * 0.5 + 0.5) * (image.height - 1))));
  const offset = (py * image.width + px) * 3;
  return [image.rgb[offset], image.rgb[offset + 1], image.rgb[offset + 2]];
}

function tiltRiskProfileForSplat(params, image, index, angleDegrees = DEFAULT_TILT_SPLIT_ANGLE_DEGREES) {
  const theta = params.theta?.[index] || 0;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const sx = params.scale[index * 2];
  const sy = params.scale[index * 2 + 1];
  const frame = plyFrameScale(image);
  const worldX = [c * sx * frame.x, s * sx * frame.y];
  const worldY = [-s * sy * frame.x, c * sy * frame.y];
  const angleSin = Math.sin(Math.max(0, angleDegrees) * Math.PI / 180);
  const yawDepth = 4 * angleSin * Math.hypot(worldX[0], worldY[0]);
  const pitchDepth = 4 * angleSin * Math.hypot(worldX[1], worldY[1]);
  const supportDepth = Math.max(yawDepth, pitchDepth);
  const centerX = params.xy[index * 2];
  const centerY = params.xy[index * 2 + 1];
  const axisX = [c * sx, s * sx];
  const axisY = [-s * sy, c * sy];
  const sourceColor = [params.rgb[index * 3], params.rgb[index * 3 + 1], params.rgb[index * 3 + 2]];
  const samples = [
    sourceRgbAtNdc(image, centerX - axisX[0], centerY - axisX[1]),
    sourceRgbAtNdc(image, centerX + axisX[0], centerY + axisX[1]),
    sourceRgbAtNdc(image, centerX - axisY[0], centerY - axisY[1]),
    sourceRgbAtNdc(image, centerX + axisY[0], centerY + axisY[1]),
    sourceRgbAtNdc(image, centerX - axisX[0] * 4, centerY - axisX[1] * 4),
    sourceRgbAtNdc(image, centerX + axisX[0] * 4, centerY + axisX[1] * 4),
    sourceRgbAtNdc(image, centerX - axisY[0] * 4, centerY - axisY[1] * 4),
    sourceRgbAtNdc(image, centerX + axisY[0] * 4, centerY + axisY[1] * 4),
  ];
  const colorMismatch = samples.reduce((maximum, sample) => Math.max(
    maximum,
    (Math.abs(sample[0] - sourceColor[0]) + Math.abs(sample[1] - sourceColor[1]) + Math.abs(sample[2] - sourceColor[2])) / 3,
  ), 0);
  const opacity = params.opacity[index];
  const risk = opacity < 0.007 ? 0 : Math.min(64,
    opacity
      * Math.max(0, supportDepth / PLY_LAYER_DEPTH_SPAN - 1)
      * Math.max(0, colorMismatch / DEFAULT_TILT_SPLIT_COLOR_THRESHOLD - 1),
  );
  const useX = yawDepth >= pitchDepth
    ? Math.abs(worldX[0]) >= Math.abs(worldY[0])
    : Math.abs(worldX[1]) >= Math.abs(worldY[1]);
  return { index, risk, supportDepth, colorMismatch, splitAxis: useX ? "x" : "y" };
}

function summarizeTiltRisk(params, image) {
  const angleDegrees = phase39Variants().tiltSplitAngleDegrees;
  const sampling = finalDiagnosticSampling(params.count);
  const profiles = [];
  for (let index = 0; index < params.count; index += sampling.stride) {
    profiles.push(tiltRiskProfileForSplat(params, image, index, angleDegrees));
  }
  const support = profiles.map((profile) => profile.supportDepth).sort((a, b) => a - b);
  const percentile = (fraction) => support.length
    ? support[Math.min(support.length - 1, Math.round((support.length - 1) * fraction))]
    : 0;
  const risky = profiles.filter((profile) => profile.risk > 0).sort((a, b) => b.risk - a.risk);
  const riskyRatio = risky.length / Math.max(1, profiles.length);
  return {
    angle_degrees: angleDegrees,
    depth_threshold: PLY_LAYER_DEPTH_SPAN,
    color_threshold: DEFAULT_TILT_SPLIT_COLOR_THRESHOLD,
    source_count: sampling.sourceCount,
    sample_count: sampling.sampleCount,
    sample_stride: sampling.stride,
    risky_sample_count: risky.length,
    risky_count: Math.min(sampling.sourceCount, Math.round(riskyRatio * sampling.sourceCount)),
    risky_ratio: riskyRatio,
    support_depth_p95: percentile(0.95),
    support_depth_p99: percentile(0.99),
    support_depth_max: support.at(-1) || 0,
    top: risky.slice(0, 16),
  };
}

function plyLayerDepth(index, params, enabled = Boolean(params.layerOrderEnabled)) {
  const fallback = 1 - index / Math.max(1, params.count - 1);
  const order = Math.max(0, Math.min(1, params.depthOrder?.[index] ?? fallback));
  const layerDepth = enabled ? (order - 0.5) * PLY_LAYER_DEPTH_SPAN : 0;
  return layerDepth + boundedVirtualDepth(params, index);
}

function frontRenderLayerDepth(index, params) {
  if (params.virtualDepthEnabled) return plyLayerDepth(index, params);
  if (!params.layerOrderEnabled) return 0;
  const fallback = 1 - index / Math.max(1, params.count - 1);
  const order = Math.max(0, Math.min(1, params.depthOrder?.[index] ?? fallback));
  return order * LAYER_CODE_RANGE;
}

function layerOrderComparator(a, b, params) {
  const delta = frontRenderLayerDepth(b, params) - frontRenderLayerDepth(a, params);
  // Match tile_less() in the unit used by the active front renderer. PLY z is
  // a separate export-space mapping and must not change near-tie decisions.
  return Math.abs(delta) > 1e-7 ? delta : a - b;
}

function splatPreviewOrderComparator(a, b, params) {
  const opacityDelta = (params.opacity[b] || 0) - (params.opacity[a] || 0);
  if (Math.abs(opacityDelta) > 1e-4) return opacityDelta;
  const areaA = Math.max(MIN_SPLAT_SCALE, params.scale[a * 2]) * Math.max(MIN_SPLAT_SCALE, params.scale[a * 2 + 1]);
  const areaB = Math.max(MIN_SPLAT_SCALE, params.scale[b * 2]) * Math.max(MIN_SPLAT_SCALE, params.scale[b * 2 + 1]);
  const areaDelta = areaA - areaB;
  if (Math.abs(areaDelta) > 1e-10) return areaDelta;
  return layerOrderComparator(a, b, params);
}

function buildSplatPreviewOrder(params) {
  const ordered = new Uint32Array(params.count);
  for (let index = 0; index < params.count; index += 1) ordered[index] = index;
  ordered.sort((a, b) => splatPreviewOrderComparator(a, b, params));
  return ordered;
}

function cachedResultSmallFirstOrder(resultState, params) {
  if (!resultState || resultState.sourceParams !== params || resultState.count !== params.count) return null;
  const cache = resultState.smallFirstOrderCache;
  if (cache?.params === params && cache.count === params.count) return cache.order;
  const order = buildSplatPreviewOrder(params);
  resultState.smallFirstOrderCache = { params, count: params.count, order };
  return order;
}

function makePly(params = state.params, image = state.image) {
  if (!params) throw new Error("No trained splats to export.");
  if (!image) throw new Error("No source image is available for aspect-preserving PLY export.");
  assertSplatCountContract(params, "ply-export");
  assertFiniteParams(params, "ply-export");
  const layerOrderEnabled = Boolean(params.layerOrderEnabled);
  const header = plyHeaderText(params, image);
  return serializeBinaryPly({
    header,
    count: params.count,
    rowBytes: PLY_ROW_BYTES,
    shC0: SH_C0,
    geometryAt: (i) => transformPlanarSplatForPly(
      params.xy[i * 2],
      params.xy[i * 2 + 1],
      params.scale[i * 2],
      params.scale[i * 2 + 1],
      params.theta?.[i] || 0,
      image,
    ),
    depthAt: (i) => plyLayerDepth(i, params, layerOrderEnabled),
    rgbAt: (i) => [params.rgb[i * 3], params.rgb[i * 3 + 1], params.rgb[i * 3 + 2]],
    opacityAt: (i) => params.opacity[i],
    logit,
  });
}

function currentTiltRevision() {
  if (!state.params || !state.image) return "";
  return [
    state.splatAdjustmentEpoch,
    state.metrics?.params_revision ?? 0,
    state.metrics?.final_readback_step ?? -1,
    state.metrics?.steps_done ?? -1,
    state.splatAdjustmentVersion,
    state.params.count,
    state.image.width,
    state.image.height,
  ].join(":");
}

function tiltViewerRuntime() {
  const runtime = globalThis.Image2SplatPaintTilt;
  if (typeof runtime?.createTiltViewer !== "function") {
    throw new Error("Tilt renderer bundle did not load.");
  }
  return runtime;
}

function tiltViewerReady() {
  const resultAlgorithm = trainedResultAlgorithm();
  return Boolean(
    resultAlgorithm?.capabilities.virtualCameras &&
    typeof globalThis.Image2SplatPaintTilt?.createTiltViewer === "function" &&
    state.params &&
    state.metrics &&
    !state.running &&
    state.metrics.cpu_mirror_current &&
    state.metrics.final_readback_step === state.metrics.steps_done,
  );
}

function tiltViewerAvailabilityMessage() {
  const resultAlgorithm = trainedResultAlgorithm();
  if (resultAlgorithm && !resultAlgorithm.capabilities.virtualCameras) {
    return "Train with GS Virtual Camera Sampling before opening Tilt.";
  }
  if (!resultAlgorithm && !algorithmUsesVirtualCameras()) {
    return "Tilt is available only for GS Virtual Camera Sampling.";
  }
  if (typeof globalThis.Image2SplatPaintTilt?.createTiltViewer !== "function") return "Tilt renderer failed to load.";
  if (state.running) return "Stop or finish training before opening the Tilt viewer.";
  if (!state.params || !state.metrics) return "Finish training to inspect the PLY.";
  if (!state.metrics.cpu_mirror_current || state.metrics.final_readback_step !== state.metrics.steps_done) {
    return "Waiting for the final training result before opening the Tilt viewer.";
  }
  return "Open Tilt to build a fresh in-memory PLY view.";
}

function updateTiltControlState() {
  const ready = tiltViewerReady();
  const resultAlgorithm = trainedResultAlgorithm();
  const tiltAvailable = resultAlgorithm
    ? Boolean(resultAlgorithm.capabilities.virtualCameras)
    : algorithmUsesVirtualCameras();
  els.tiltTab.disabled = !tiltAvailable;
  els.tiltTab.setAttribute("aria-disabled", String(!tiltAvailable));
  const interactive = Boolean(state.tilt.controller) && !state.tilt.teacherViewsLoading;
  const inspectingCameraPool = interactive && state.tilt.cameraMarkersVisible;
  els.tiltPitch.disabled = !interactive || inspectingCameraPool;
  els.tiltYaw.disabled = !interactive || inspectingCameraPool;
  els.tiltFrontButton.disabled = !interactive || inspectingCameraPool;
  els.tiltRefreshButton.disabled = !ready || state.tilt.loading || state.tilt.teacherViewsLoading;
  els.tiltTrainingViewsButton.disabled = !interactive || state.tilt.teacherViewsLoading;
  for (const button of [els.tiltOriginalView, els.tiltOverlayView, els.tiltSplatsView]) {
    button.disabled = !interactive;
  }
  els.tiltCameraMarkers.disabled = !interactive;
  els.tiltRadiusMode.disabled = !interactive;
  document.documentElement.dataset.tiltReady = String(ready);
  document.documentElement.dataset.tiltLoaded = String(Boolean(state.tilt.controller));
  document.documentElement.dataset.tiltLoading = String(state.tilt.loading);
  document.documentElement.dataset.tiltRevision = state.tilt.revision;
  document.documentElement.dataset.tiltPlySha256 = state.tilt.plyDigest || state.tilt.verifiedPlyDigest;
  document.documentElement.dataset.tiltPlyBytes = String(state.tilt.plyByteLength || state.tilt.verifiedPlyByteLength || 0);
  document.documentElement.dataset.tiltPlyVertices = String(state.tilt.vertices || 0);
  document.documentElement.dataset.tiltTrainingViewsLoading = String(state.tilt.teacherViewsLoading);
  document.documentElement.dataset.tiltTrainingViewCount = String(state.tilt.teacherViews.length);
  document.documentElement.dataset.tiltDisplayMode = state.tilt.viewMode;
  document.documentElement.dataset.tiltCameraMarkers = String(state.tilt.cameraMarkersVisible);
  const cameraMarkers = state.tilt.controller?.diagnostics?.().cameraMarkers;
  const markerCount = cameraMarkers?.count || 0;
  const sampleCount = cameraMarkers?.sampleCount || 0;
  const frontCount = cameraMarkers?.frontCount || 0;
  const virtualCount = cameraMarkers?.virtualCount || 0;
  const frontSampleCount = cameraMarkers?.frontSampleCount || 0;
  const virtualSampleCount = cameraMarkers?.virtualSampleCount || 0;
  const runSampling = state.metrics?.virtual_camera_sampling;
  const runSamplingEnabled = Boolean(runSampling?.enabled);
  // This describes the next Train selection; existing Tilt availability above
  // remains bound to the completed result.
  const currentSamplingEnabled = algorithmUsesVirtualCameras();
  document.documentElement.dataset.tiltCameraMarkerCount = String(markerCount);
  document.documentElement.dataset.tiltCameraSampleCount = String(sampleCount);
  document.documentElement.dataset.tiltCameraFrontMarkerCount = String(frontCount);
  document.documentElement.dataset.tiltCameraVirtualMarkerCount = String(virtualCount);
  document.documentElement.dataset.tiltCameraFrontSampleCount = String(frontSampleCount);
  document.documentElement.dataset.tiltCameraVirtualSampleCount = String(virtualSampleCount);
  document.documentElement.dataset.tiltCameraRunSamplingEnabled = String(runSamplingEnabled);
  document.documentElement.dataset.tiltCameraCurrentSamplingEnabled = String(currentSamplingEnabled);
  if (!state.tilt.controller) {
    els.tiltCameraSummary.textContent = "Finish training to inspect the cameras used in this run.";
  } else if (virtualCount > 0) {
    els.tiltCameraSummary.textContent =
      `${markerCount} camera poses: ${frontCount} front + ${virtualCount} virtual` +
      ` · ${frontSampleCount} front / ${virtualSampleCount} virtual selections.`;
  } else if (runSamplingEnabled || cameraMarkers?.samplingEnabled) {
    const stepsDone = Number(state.metrics?.steps_done) || 0;
    const warmupSteps = Number(runSampling?.warmup_steps) || 0;
    els.tiltCameraSummary.textContent =
      stepsDone <= warmupSteps
        ? `${frontCount || 1} front camera · This run ended during the front-only warmup (${stepsDone} / ${warmupSteps}).`
        : `${frontCount || 1} front camera · No virtual camera was selected in this run.`;
  } else if (currentSamplingEnabled) {
    els.tiltCameraSummary.textContent =
      `${frontCount || 1} front camera · This run used front only. Virtual sampling is ON for the next Train.`;
  } else {
    els.tiltCameraSummary.textContent =
      `${frontCount || 1} front camera · Virtual camera sampling was off for this run.`;
  }
}

function restorePrimaryCanvas() {
  showCanvas(state.previewMode === "splats" && state.params ? "gpu" : "preview");
}

function positionTiltFrameSegment(element, start, end) {
  const length = Math.hypot(end.x - start.x, end.y - start.y);
  const angle = Math.atan2(end.y - start.y, end.x - start.x) * 180 / Math.PI;
  element.style.left = `${start.x}px`;
  element.style.top = `${start.y}px`;
  element.style.width = `${length}px`;
  element.style.transform = `rotate(${angle}deg)`;
}

function updateTiltCameraDiagnostics(diagnostics) {
  if (!diagnostics?.corners?.every((corner) => corner.valid)) {
    els.tiltFrameOverlay.hidden = true;
    els.tiltProjectionError.textContent = "invalid";
    return;
  }
  const segments = [...els.tiltFrameOverlay.querySelectorAll("span")];
  diagnostics.corners.forEach((corner, index) => {
    positionTiltFrameSegment(segments[index], corner, diagnostics.corners[(index + 1) % diagnostics.corners.length]);
  });
  const centerMarker = els.tiltFrameOverlay.querySelector("i");
  centerMarker.style.left = `${diagnostics.center.x}px`;
  centerMarker.style.top = `${diagnostics.center.y}px`;
  const trainingOrbitRadius = diagnostics.trainingOrbitRadius ?? diagnostics.radius;
  const viewerOrbitRadius = diagnostics.viewerOrbitRadius ?? diagnostics.radius;
  const displayedRadius = diagnostics.viewMode === "camera-pool-overview"
    ? trainingOrbitRadius
    : viewerOrbitRadius;
  els.tiltRadiusValue.textContent = displayedRadius.toFixed(4);
  els.tiltPositionValue.textContent = diagnostics.position.map((value) => value.toFixed(3)).join(", ");
  els.tiltFovValue.textContent = `${diagnostics.fovDegrees.toFixed(0)}\u00b0`;
  els.tiltProjectionError.textContent = `${diagnostics.cornerErrorMaxPx.toFixed(3)} px`;
  els.tiltCameraMode.textContent = diagnostics.viewMode === "camera-pool-overview"
    ? "camera pool overview"
    : diagnostics.radiusMode === "fit" ? "center orbit / fit splats" : "center orbit / training radius";
  els.tiltFrameOverlay.hidden = els.tiltCanvas.hidden;
  document.documentElement.dataset.tiltOrbitRadius = String(displayedRadius);
  document.documentElement.dataset.tiltTrainingOrbitRadius = String(trainingOrbitRadius);
  document.documentElement.dataset.tiltViewerOrbitRadius = String(viewerOrbitRadius);
  document.documentElement.dataset.tiltFitAllOrbitRadius = String(diagnostics.fitAllOrbitRadius ?? viewerOrbitRadius);
  document.documentElement.dataset.tiltRadiusMode = diagnostics.radiusMode || "training";
  document.documentElement.dataset.tiltViewMode = diagnostics.viewMode || "center-orbit";
  document.documentElement.dataset.tiltCameraPosition = diagnostics.position.join(",");
  document.documentElement.dataset.tiltFov = String(diagnostics.fovDegrees);
  document.documentElement.dataset.tiltProjectionError = String(diagnostics.cornerErrorMaxPx);
  scheduleTiltTeacherFrame(diagnostics);
}

function setTiltDisplayMode(mode, { focusCamera = false } = {}) {
  state.tilt.viewMode = ["original", "overlay", "splats"].includes(mode) ? mode : "splats";
  if (focusCamera && state.tilt.controller && state.tilt.cameraMarkersVisible) {
    state.tilt.cameraMarkersVisible = false;
    els.tiltCameraMarkers.checked = false;
    state.tilt.controller.setCameraMarkersVisible(false);
  }
  for (const [button, value] of [
    [els.tiltOriginalView, "original"],
    [els.tiltOverlayView, "overlay"],
    [els.tiltSplatsView, "splats"],
  ]) {
    const active = state.tilt.viewMode === value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  if (!els.tiltCanvas.hidden) showCanvas("tilt");
  const diagnostics = state.tilt.controller?.diagnostics?.().camera;
  if (diagnostics && state.tilt.viewMode !== "splats") scheduleTiltTeacherFrame(diagnostics);
  publishState();
}

function rasterizeVirtualTeacher(canvas, image, camera, { maxSide = 512, outputAspect = null } = {}) {
  const aspect = Number.isFinite(Number(outputAspect)) && Number(outputAspect) > 0
    ? Number(outputAspect)
    : image.width / image.height;
  const width = aspect >= 1 ? Math.max(1, Math.round(maxSide)) : Math.max(1, Math.round(maxSide * aspect));
  const height = aspect >= 1 ? Math.max(1, Math.round(maxSide / aspect)) : Math.max(1, Math.round(maxSide));
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  const pixels = context.createImageData(width, height);
  const pitch = Number(camera.pitchDegrees ?? camera.pitch_degrees) || 0;
  const yaw = Number(camera.yawDegrees ?? camera.yaw_degrees) || 0;
  const distance = Math.max(0.01, Number(camera.cameraDistance ?? camera.radius) || DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE);
  const fovDegrees = clampSharedCameraFov(camera.fovDegrees ?? camera.fov_degrees);
  const rotation = planarTiltRotation(pitch * Math.PI / 180, yaw * Math.PI / 180);
  const longSide = Math.max(1, image.width, image.height);
  const frameX = image.width / longSide;
  const frameY = image.height / longSide;
  const tanHalfFov = Math.tan(fovDegrees * Math.PI / 360);
  const focalX = 1 / (tanHalfFov * aspect);
  const focalY = 1 / tanHalfFov;
  const sourceWidth = image.width;
  const sourceHeight = image.height;
  for (let py = 0, target = 0; py < height; py += 1) {
    const v = height > 1 ? py / (height - 1) * 2 - 1 : 0;
    const a10 = v * frameX * rotation[6] + focalY * frameX * rotation[3];
    const a11 = -v * frameY * rotation[7] - focalY * frameY * rotation[4];
    const rightY = -v * distance;
    for (let px = 0; px < width; px += 1, target += 4) {
      const u = width > 1 ? px / (width - 1) * 2 - 1 : 0;
      const a00 = u * frameX * rotation[6] - focalX * frameX * rotation[0];
      const a01 = -u * frameY * rotation[7] + focalX * frameY * rotation[1];
      const determinant = a00 * a11 - a01 * a10;
      const rightX = -u * distance;
      if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-8) {
        pixels.data[target + 3] = 255;
        continue;
      }
      const sourceX = (rightX * a11 - a01 * rightY) / determinant;
      const sourceY = (a00 * rightY - rightX * a10) / determinant;
      if (!Number.isFinite(sourceX + sourceY) || sourceX < -1 || sourceX > 1 || sourceY < -1 || sourceY > 1) {
        pixels.data[target + 3] = 255;
        continue;
      }
      const sx = (sourceX * 0.5 + 0.5) * (sourceWidth - 1);
      const sy = (sourceY * 0.5 + 0.5) * (sourceHeight - 1);
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const y1 = Math.min(sourceHeight - 1, y0 + 1);
      const fx = sx - x0;
      const fy = sy - y0;
      const i00 = (y0 * sourceWidth + x0) * 3;
      const i10 = (y0 * sourceWidth + x1) * 3;
      const i01 = (y1 * sourceWidth + x0) * 3;
      const i11 = (y1 * sourceWidth + x1) * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        const top = image.rgb[i00 + channel] * (1 - fx) + image.rgb[i10 + channel] * fx;
        const bottom = image.rgb[i01 + channel] * (1 - fx) + image.rgb[i11 + channel] * fx;
        pixels.data[target + channel] = clampByte((top * (1 - fy) + bottom * fy) * 255);
      }
      pixels.data[target + 3] = 255;
    }
  }
  context.putImageData(pixels, 0, 0);
  return { width, height };
}

function scheduleTiltTeacherFrame(diagnostics) {
  if (!state.image || state.tilt.viewMode === "splats" || els.tiltCanvas.hidden) return;
  const request = ++state.tilt.teacherFrameRequest;
  requestAnimationFrame(() => {
    if (request !== state.tilt.teacherFrameRequest || !state.image) return;
    const aspect = Math.max(1, els.tiltCanvas.clientWidth) / Math.max(1, els.tiltCanvas.clientHeight);
    rasterizeVirtualTeacher(els.tiltTeacherCanvas, state.image, {
      pitchDegrees: diagnostics.pitchDegrees,
      yawDegrees: diagnostics.yawDegrees,
      cameraDistance: diagnostics.radius,
      fovDegrees: diagnostics.fovDegrees,
    }, { maxSide: Math.min(768, Math.max(256, els.tiltCanvas.clientWidth)), outputAspect: aspect });
  });
}

function clearTiltTrainingViews() {
  state.tilt.teacherViews = [];
  els.tiltContactSheet.replaceChildren();
  els.tiltTrainingViewsProgress.max = 1;
  els.tiltTrainingViewsProgress.value = 0;
  els.tiltTrainingViewsSummary.textContent = "Virtual teacher images used in this run.";
}

function appendTiltTrainingView(result, canvas) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.cameraId = result.id;
  button.title = `${result.id}: pitch ${result.pitchDegrees.toFixed(1)}°, yaw ${result.yawDegrees.toFixed(1)}°`;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", `${result.id} virtual teacher image`);
  const label = document.createElement("span");
  label.textContent = `${result.polarDegrees.toFixed(1)}° · ${result.multiplicity}x`;
  button.append(canvas, label);
  button.addEventListener("click", () => {
    if (!state.tilt.controller || state.tilt.teacherViewsLoading) return;
    try {
      state.tilt.cameraMarkersVisible = false;
      els.tiltCameraMarkers.checked = false;
      state.tilt.controller.setCameraMarkersVisible(false);
      const pitch = Math.round(result.pitchDegrees * 10) / 10;
      const yaw = Math.round(result.yawDegrees * 10) / 10;
      els.tiltPitch.value = String(pitch);
      els.tiltYaw.value = String(yaw);
      els.tiltPitchValue.textContent = `${pitch.toFixed(1)}\u00b0`;
      els.tiltYawValue.textContent = `${yaw.toFixed(1)}\u00b0`;
      document.documentElement.dataset.tiltPitch = String(pitch);
      document.documentElement.dataset.tiltYaw = String(yaw);
      updateTiltControlState();
      publishState();
      applyTiltInputs();
    } catch (error) {
      els.tiltStatus.textContent = `Training camera view failed: ${error.message}`;
      log(error.message);
    }
  });
  els.tiltContactSheet.append(button);
}

async function showTiltTrainingViews() {
  if (state.tilt.teacherViewsLoading) return;
  const controller = await loadTiltViewer();
  if (!controller) return;
  const pool = await tiltCameraPoolSnapshot();
  const cameras = pool.cameras.filter((camera) => camera.kind === "virtual" && camera.multiplicity > 0);
  clearTiltTrainingViews();
  state.tilt.teacherViewsLoading = true;
  els.tiltTrainingViewsProgress.max = Math.max(1, cameras.length);
  els.tiltStatus.textContent = cameras.length
    ? `Generating virtual teacher images 0 / ${cameras.length}...`
    : "This run did not use a virtual camera.";
  updateTiltControlState();
  publishState();
  try {
    for (let index = 0; index < cameras.length; index += 1) {
      const camera = cameras[index];
      const canvas = document.createElement("canvas");
      rasterizeVirtualTeacher(canvas, state.image, {
        pitchDegrees: camera.pitch_degrees,
        yawDegrees: camera.yaw_degrees,
        cameraDistance: pool.orbit_radius,
        fovDegrees: pool.fov_degrees,
      }, { maxSide: 240, outputAspect: state.image.width / state.image.height });
      const result = {
        id: camera.id,
        pitchDegrees: camera.pitch_degrees,
        yawDegrees: camera.yaw_degrees,
        polarDegrees: Number(camera.polar_degrees) || Math.hypot(camera.pitch_degrees, camera.yaw_degrees),
        multiplicity: camera.multiplicity,
      };
      state.tilt.teacherViews.push(result);
      appendTiltTrainingView(result, canvas);
      els.tiltTrainingViewsProgress.value = index + 1;
      els.tiltTrainingViewsSummary.textContent = `${index + 1} / ${cameras.length} virtual teacher images`;
      els.tiltStatus.textContent = `Generating virtual teacher images ${index + 1} / ${cameras.length}...`;
      if (index % 4 === 3) await nextFrame();
    }
    // The gallery can change the side panel scrollbar and therefore the viewer
    // size. Reconcile PlayCanvas and analytic projection after layout settles.
    await nextFrame();
    await controller.refreshCameraDiagnostics();
    els.tiltTrainingViewsSummary.textContent = cameras.length
      ? `${cameras.length} virtual teacher images used in this run`
      : "No virtual teacher images in this run";
    els.tiltStatus.textContent = cameras.length
      ? "Training views ready. Select one to inspect its camera pose."
      : "This run used only the front camera.";
  } finally {
    state.tilt.teacherViewsLoading = false;
    updateTiltControlState();
    publishState();
  }
}

function destroyTiltViewer({ restoreCanvas = false } = {}) {
  if (tiltInputFrame) {
    cancelAnimationFrame(tiltInputFrame);
    tiltInputFrame = 0;
  }
  state.tilt.teacherFrameRequest += 1;
  state.tilt.generation += 1;
  state.tilt.abortController?.abort();
  state.tilt.abortController = null;
  state.tilt.controller?.destroy?.();
  state.tilt.controller = null;
  state.tilt.revision = "";
  state.tilt.loading = false;
  state.tilt.plyDigest = "";
  state.tilt.plyByteLength = 0;
  state.tilt.vertices = 0;
  state.tilt.teacherViewsLoading = false;
  clearTiltTrainingViews();
  els.tiltFrameOverlay.hidden = true;
  els.tiltPositionValue.textContent = "-";
  els.tiltRadiusValue.textContent = "-";
  els.tiltProjectionError.textContent = "-";
  els.tiltPitch.value = "0";
  els.tiltYaw.value = "0";
  els.tiltPitchValue.textContent = "0\u00b0";
  els.tiltYawValue.textContent = "0\u00b0";
  document.documentElement.dataset.tiltPitch = "0";
  document.documentElement.dataset.tiltYaw = "0";
  if (restoreCanvas && !els.tiltCanvas.hidden) restorePrimaryCanvas();
  els.tiltStatus.textContent = tiltViewerAvailabilityMessage();
  updateTiltControlState();
}

function applyTiltInputs() {
  let pitch = Number(els.tiltPitch.value) || 0;
  let yaw = Number(els.tiltYaw.value) || 0;
  const applied = state.tilt.controller?.setTilt(pitch, yaw);
  if (applied) {
    pitch = applied.pitch;
    yaw = applied.yaw;
    els.tiltPitch.value = String(pitch);
    els.tiltYaw.value = String(yaw);
  }
  const displayedPitch = String(Object.is(Math.round(pitch * 1000) / 1000, -0) ? 0 : Math.round(pitch * 1000) / 1000);
  const displayedYaw = String(Object.is(Math.round(yaw * 1000) / 1000, -0) ? 0 : Math.round(yaw * 1000) / 1000);
  els.tiltPitchValue.textContent = `${displayedPitch}\u00b0`;
  els.tiltYawValue.textContent = `${displayedYaw}\u00b0`;
  document.documentElement.dataset.tiltPitch = displayedPitch;
  document.documentElement.dataset.tiltYaw = displayedYaw;
}

let tiltInputFrame = 0;

function scheduleTiltInputs() {
  if (tiltInputFrame) return;
  tiltInputFrame = requestAnimationFrame(() => {
    tiltInputFrame = 0;
    applyTiltInputs();
  });
}

function tiltCameraCounts(sampling) {
  const expectedSteps = Math.max(0, Math.round(Number(state.metrics?.steps_done) || 0));
  const stored = Object.fromEntries(
    Object.entries(sampling?.camera_counts || {}).map(([id, count]) => [id, Math.max(0, Number(count) || 0)]),
  );
  const storedTotal = Object.values(stored).reduce((total, count) => total + count, 0);
  if (!sampling?.enabled || storedTotal >= expectedSteps) return { counts: stored, source: "metrics" };

  const history = state.virtualCameraByStep;
  if (history instanceof Map && history.size === expectedSteps) {
    const counts = {};
    let complete = true;
    for (let step = 1; step <= expectedSteps; step += 1) {
      const cameraId = history.get(step);
      if (!cameraId) {
        complete = false;
        break;
      }
      counts[cameraId] = (counts[cameraId] || 0) + 1;
    }
    if (complete) return { counts, source: "runtime-step-history" };
  }

  const variants = {
    enabled: true,
    slots: Math.max(2, Math.round(Number(sampling.pool_slots) || DEFAULT_VIRTUAL_CAMERA_POOL_SLOTS)),
    virtualSlots: Math.max(1, Math.round(Number(sampling.virtual_slots) || DEFAULT_VIRTUAL_CAMERA_SLOTS)),
    cameraCount: Math.max(4, Math.min(MAX_VIRTUAL_CAMERA_COUNT, Math.round(Number(sampling.virtual_camera_count) || DEFAULT_VIRTUAL_CAMERA_COUNT))),
    maxAngleDegrees: Number(sampling.max_angle_degrees) || DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES,
    fovDegrees: clampSharedCameraFov(sampling.fov_degrees),
    seed: Math.max(0, Math.floor(
      Number.isFinite(Number(sampling.seed)) ? Number(sampling.seed) : DEFAULT_VIRTUAL_CAMERA_SEED,
    )) >>> 0,
    cameraDistance: Number(sampling.orbit_radius) || DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE,
    autoCameraDistance: true,
    orderPenaltyWeight: 0,
    planeConstrained: true,
    invalidRegionMode: sampling.invalid_region_mode || "mask",
  };
  const warmupSteps = Math.max(0, Math.round(Number(sampling.warmup_steps) || 0));
  const counts = {};
  for (let step = 1; step <= expectedSteps; step += 1) {
    const stage = step <= warmupSteps ? "coarse" : "full";
    const camera = virtualCameraSamplingStepSpec(step, stage, expectedSteps, warmupSteps, variants);
    counts[camera.cameraId] = (counts[camera.cameraId] || 0) + 1;
  }
  return { counts, source: "deterministic-schedule" };
}

async function tiltCameraPoolSnapshot() {
  const sampling = state.metrics?.virtual_camera_sampling;
  const fovDegrees = clampSharedCameraFov(sampling?.fov_degrees);
  const radius = Number.isFinite(Number(sampling?.orbit_radius))
    ? Number(sampling.orbit_radius)
    : sharedTiltOrbitRadius(
      state.image.width,
      state.image.height,
      sampling?.max_angle_degrees,
      49,
      fovDegrees,
    );
  const cameraHistory = tiltCameraCounts(sampling);
  const cameras = sampling?.enabled
    ? sampling.cameras
      .map((camera) => ({
        ...camera,
        multiplicity: Math.max(0, Number(cameraHistory.counts[camera.id]) || 0),
      }))
      .filter((camera) => camera.multiplicity > 0)
    : [{
      id: "front",
      kind: "front",
      pitch_degrees: 0,
      yaw_degrees: 0,
      intrinsics: virtualFrontIntrinsics(state.image.width, state.image.height, fovDegrees),
    }];
  return {
    cameras: structuredClone(cameras),
    sampling_enabled: Boolean(sampling?.enabled),
    camera_counts_source: cameraHistory.source,
    active_camera_id: sampling?.enabled ? sampling.active_camera_id : "front",
    orbit_radius: radius,
    max_angle_degrees: Math.max(5, Math.min(
      MAX_VIRTUAL_CAMERA_ANGLE_DEGREES,
      Number(sampling?.max_angle_degrees) || DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES,
    )),
    fov_degrees: fovDegrees,
    target: Array.isArray(sampling?.target) ? [...sampling.target] : [0, 0, 0],
    image_aspect: state.image.width / state.image.height,
  };
}

async function loadTiltViewer({ force = false } = {}) {
  if (!tiltViewerReady()) throw new Error("Finish training before opening the Tilt viewer.");
  const revision = currentTiltRevision();
  if (!force && state.tilt.controller && state.tilt.revision === revision) {
    showCanvas("tilt");
    applyTiltInputs();
    return state.tilt.controller;
  }

  destroyTiltViewer();
  const generation = state.tilt.generation + 1;
  state.tilt.generation = generation;
  const abortController = new AbortController();
  state.tilt.abortController = abortController;
  state.tilt.loading = true;
  els.tiltStatus.textContent = "Building the PlayCanvas PLY view...";
  showCanvas("tilt");
  updateTiltControlState();
  try {
    const adjusted = await materializeCurrentSplatAdjustmentSnapshot();
    if (!adjusted || revision !== currentTiltRevision()) {
      if (generation === state.tilt.generation) {
        state.tilt.abortController = null;
        state.tilt.loading = false;
        els.tiltStatus.textContent = "Tilt source changed; reopen Tilt to use the latest splats.";
        updateTiltControlState();
      }
      return null;
    }
    const tiltParams = adjusted.params;
    assertTiltViewerCapacity(tiltParams, state.image);
    const plyBuffer = makePly(tiltParams, state.image);
    const cameraPool = await tiltCameraPoolSnapshot();
    const { createTiltViewer } = tiltViewerRuntime();
    const controller = await createTiltViewer({
      canvas: els.tiltCanvas,
      plyBuffer,
      frame: plyFrameScale(state.image),
      supportFrame: renderFootprintSupportFrame(state.image, tiltParams),
      signal: abortController.signal,
      onCameraChange: updateTiltCameraDiagnostics,
      cameraPool,
    });
    if (
      generation !== state.tilt.generation ||
      !splatAdjustmentSnapshotIsCurrent(adjusted.snapshot) ||
      revision !== currentTiltRevision()
    ) {
      controller.destroy();
      if (generation === state.tilt.generation) {
        state.tilt.abortController = null;
        state.tilt.loading = false;
        els.tiltStatus.textContent = "Tilt source changed; reopen Tilt to use the latest splats.";
        updateTiltControlState();
      }
      return null;
    }
    state.tilt.controller = controller;
    state.tilt.abortController = null;
    state.tilt.revision = revision;
    state.tilt.loading = false;
    state.tilt.plyDigest = controller.plyDigest;
    state.tilt.plyByteLength = controller.plyByteLength;
    state.tilt.vertices = controller.vertices;
    state.tilt.verifiedRevision = revision;
    state.tilt.verifiedPlyDigest = controller.plyDigest;
    state.tilt.verifiedPlyByteLength = controller.plyByteLength;
    controller.setCameraMarkersVisible(state.tilt.cameraMarkersVisible);
    controller.setRadiusMode(state.tilt.radiusMode);
    applyTiltInputs();
    setTiltDisplayMode(state.tilt.viewMode);
    els.tiltStatus.textContent = `PlayCanvas ${controller.engineVersion} (${controller.backend}). The camera orbits the image center at a fixed radius.`;
    eventLog(
      `Tilt viewer loaded from memory PLY: ${tiltParams.count} splats` +
      ` sha256=${controller.plyDigest.slice(0, 12)}`,
    );
    publishState();
    return controller;
  } catch (error) {
    if (error?.name === "AbortError") return null;
    if (generation === state.tilt.generation) {
      state.tilt.loading = false;
      restorePrimaryCanvas();
      els.tiltStatus.textContent = `Tilt viewer failed: ${error.message}`;
      updateTiltControlState();
    }
    throw error;
  }
}

async function saveExport({ download = true, formatKey = "ply" } = {}) {
  if (state.exporting) return;
  const algorithm = displayedResultAlgorithm();
  if (!algorithmSupportsExport(formatKey, algorithm)) {
    throw new Error(`${algorithm.label} does not support ${formatKey.toUpperCase()} export.`);
  }
  if (!state.exportReady || state.metrics?.safety_stop) throw new Error("Export is not ready.");
  const coverage = exportCoverageStatus();
  if (!coverage.ok) throw new Error(`Export is not ready: ${coverage.message}.`);
  const format = EXPORT_FORMATS[formatKey] || EXPORT_FORMATS.ply;
  const pngSpec = formatKey === "png" ? currentSplatPngSpec() : null;
  const filename = pngSpec?.filename || format.filename;
  if (formatKey === "ply") assertPlyExportCapacity(state.params, state.image, download);

  state.exporting = true;
  state.exportMessage = `Preparing ${format.label}...`;
  updateExportPanel();
  publishState();
  try {
    if (formatKey === "png") {
      const { blob, width, height, exportFrame, spec, padding, nonblackPixels, meanRgb, pngRgbaParity } = await makeSplatPreviewPngBlob();
      const bytes = new Uint8Array(await blob.arrayBuffer());
      if (download) downloadBlob(filename, blob);
      state.exportMessage = `${filename} ${download ? "saved" : "validated"} (${(bytes.byteLength / 1024).toFixed(1)} KiB).`;
      state.metrics.export_history ||= [];
      state.metrics.export_history.push({
        format: formatKey,
        filename,
        bytes: bytes.byteLength,
        width,
        height,
        resolution_mode: exportFrame.mode,
        requested_long_side: exportFrame.longSide,
        splat_shape: spec.shape,
        splat_parameter_effects: Boolean(els.splatParameterEffects?.checked),
        splat_small_first_order: Boolean(spec.renderOptions.splatSmallFirstOrder),
        kernel_falloff: Number(spec.renderOptions.kernelFalloff),
        alpha_background: [...spec.renderOptions.alphaBackground],
        outside_image: Boolean(spec.renderOptions.outside),
        padding: { x: padding.x, y: padding.y },
        nonblack_pixels: nonblackPixels,
        mean_rgb: meanRgb,
        rgba_roundtrip: pngRgbaParity,
        step: state.metrics.steps_done,
      });
      eventLog(`${download ? "exported" : "validated"} ${filename} ${width}x${height} bytes=${bytes.byteLength}`);
      return {
        format: formatKey,
        filename,
        bytes,
        width,
        height,
        exportFrame,
        spec,
        padding,
        nonblackPixels,
        meanRgb,
        pngRgbaParity,
      };
    }

    const exportParams = await materializeCurrentSplatAdjustmentSnapshot();
    if (!exportParams?.params) throw new Error("Splat adjustments changed while preparing the PLY.");
    const plyParams = exportParams.params;
    const plyBuffer = makePly(plyParams, state.image);
    const plyContract = inspectPlyContract(plyBuffer, plyParams, state.image);
    const plyValid =
      plyContract.vertices === plyParams.count &&
      plyContract.row_bytes === 68 &&
      (plyContract.layer_order_enabled ? plyContract.layer_depth_match : plyContract.all_z_zero) &&
      plyContract.all_finite &&
      plyContract.sh_degree_0 &&
      plyContract.planar_rotation &&
      plyContract.standard_alpha_blend &&
      plyContract.aspect_ratio_preserved &&
      plyContract.geometry_match_error_max <= 1e-5 &&
      plyContract.opacity_error_max <= 1e-5 &&
      plyContract.color_error_max <= 1e-5 &&
      plyContract.anisotropy_limit_violations === 0 &&
      plyContract.boundary_leak_count === 0;
    if (!plyValid) throw new Error(`Canonical PLY contract failed: ${JSON.stringify(plyContract)}`);

    const bytes = new Uint8Array(plyBuffer);
    const plyDigest = await sha256Hex(plyBuffer);
    if (!splatAdjustmentSnapshotIsCurrent(exportParams.snapshot)) {
      throw new Error("Splat adjustments changed while preparing the PLY.");
    }
    const currentTiltDigest = state.tilt.verifiedRevision === currentTiltRevision()
      ? state.tilt.verifiedPlyDigest
      : "";
    if (plyDigest && currentTiltDigest && plyDigest !== currentTiltDigest) {
      throw new Error("PLY bytes differ from the current Tilt view.");
    }

    if (download) downloadBlob(format.filename, new Blob([bytes], { type: "application/octet-stream" }));
    state.exportMessage = `${format.filename} ${download ? "saved" : "validated"} (${(bytes.byteLength / 1024).toFixed(1)} KiB).`;
    state.metrics.export_history ||= [];
    state.metrics.export_history.push({
      format: formatKey,
      filename: format.filename,
      bytes: bytes.byteLength,
      sha256: plyDigest,
      step: state.metrics.steps_done,
      contract: plyContract,
    });
    document.documentElement.dataset.lastPlySha256 = plyDigest;
    document.documentElement.dataset.lastPlyBytes = String(bytes.byteLength);
    eventLog(
      `${download ? "exported" : "validated"} ${format.filename} bytes=${bytes.byteLength}` +
      ` sha256=${plyDigest ? plyDigest.slice(0, 12) : "unavailable"}` +
      ` aspect=${plyContract.frame_aspect.toFixed(6)}` +
      ` geometry_error=${plyContract.geometry_match_error_max.toExponential(2)}` +
      ` opacity_error=${plyContract.opacity_error_max.toExponential(2)}` +
      ` color_error=${plyContract.color_error_max.toExponential(2)}`,
    );
    return {
      format: formatKey,
      filename: format.filename,
      bytes,
      sha256: plyDigest,
      plyContract,
    };
  } catch (error) {
    state.exportMessage = `Export failed: ${error.message}`;
    eventLog(state.exportMessage);
    throw error;
  } finally {
    state.exporting = false;
    updateExportPanel();
    publishState();
  }
}

function inspectPlyContract(buffer = makePly(), sourceParams = state.params, sourceImage = state.image) {
  return inspectSerializedPlyContract(buffer, sourceParams, sourceImage, {
    plyFrameScale,
    selectedBoundarySigma,
    anisotropyLimitsForParams,
    plyLayerDepth,
    transformPlanarSplatForPly,
    rotatedSplatExtent,
    shC0: SH_C0,
    minSplatScale: MIN_SPLAT_SCALE,
    renderSigma: RENDER_SIGMA,
    plyLayerDepthSpan: PLY_LAYER_DEPTH_SPAN,
    defaultVirtualDepthThickness: DEFAULT_VIRTUAL_DEPTH_THICKNESS,
  });
}

function hashBytes(bytes, seed = 2166136261) {
  let hash = seed >>> 0;
  for (let i = 0; i < bytes.byteLength; i += 1) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function hashParams(params = state.params) {
  let hash = 2166136261;
  for (const values of [params?.xy, params?.scale, params?.rgb, params?.opacity, params?.theta, params?.depthOrder]) {
    if (!values) continue;
    hash = hashBytes(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), hash);
  }
  return hash >>> 0;
}

function previewInvariantSnapshot() {
  if (!state.params) return null;
  const metrics = state.metrics;
  const hashEnabled = previewInvariantHashEnabled();
  const paramsHash = hashEnabled ? hashParams() : null;
  const plySignature = new TextEncoder().encode([
    paramsHash ?? "deferred",
    state.params.count,
    state.image?.width || 0,
    state.image?.height || 0,
    state.params.boundarySigma,
    Boolean(state.params.layerOrderEnabled),
    PLY_LAYER_DEPTH_SPAN,
  ].join("|"));
  return {
    params_hash: paramsHash,
    ply_hash: hashBytes(plySignature),
    hash_mode: hashEnabled ? "full-qa" : "revision-only",
    splats: state.params.count,
    steps: metrics?.steps_done ?? 0,
    params_revision: metrics?.params_revision ?? 0,
    l1: metrics?.final_l1 ?? null,
    psnr: metrics?.final_psnr ?? null,
    global_ssim: metrics?.final_global_ssim ?? null,
    windowed_ssim: metrics?.final_windowed_ssim ?? null,
    regional_p10: metrics?.final_regional_ssim?.p10 ?? null,
  };
}

async function runPreviewRefreshLoop() {
  const refreshEpoch = state.previewRefreshEpoch;
  let rendered = false;
  state.previewRefreshPending = true;
  publishState();
  try {
    while (state.previewAppliedRevision < state.previewRequestedRevision) {
      if (refreshEpoch !== state.previewRefreshEpoch) return false;
      const requestedRevision = state.previewRequestedRevision;
      if (state.running) {
        state.previewPadding = previewPaddingSpec(state.image, state.params, false);
        state.previewAppliedAlphaBackground = "";
        state.webgpu.renderer?.presentTrainState(state.image);
        if (state.previewMode === "splats") showCanvas("gpu");
        state.previewAppliedRevision = requestedRevision;
        rendered = true;
        break;
      }
      const renderer = state.webgpu.renderer;
      if (!state.image || !state.params || !renderer || renderer.deviceLost) {
        state.previewPadding = previewPaddingSpec(state.image, state.params, false);
        state.previewAppliedAlphaBackground = "";
        state.previewAppliedRevision = requestedRevision;
        break;
      }

      const generation = state.previewGeneration;
      const image = state.image;
      const params = state.params;
      const displayParams = params;
      const before = previewInvariantSnapshot();
      const buffers = renderer.currentTrainBuffers(params) || renderer.currentResultBuffers(params);
      const splatOptionsActive =
        document.documentElement.dataset.activeDetailTab === "splats" &&
        state.previewMode === "splats";
      const alphaOptions = splatOptionsActive ? splatAlphaRenderOptions() : {};
      const appliedAlphaBackground = splatOptionsActive ? els.splatAlphaBackground.value.toLowerCase() : "";
      try {
        await renderer.render(image, displayParams, buffers, null, alphaOptions);
      } catch (error) {
        if (refreshEpoch !== state.previewRefreshEpoch || isExpectedPreviewCancellation(error)) return rendered;
        throw error;
      }
      if (
        refreshEpoch !== state.previewRefreshEpoch ||
        generation !== state.previewGeneration ||
        renderer !== state.webgpu.renderer ||
        renderer.deviceLost ||
        state.running ||
        image !== state.image ||
        params !== state.params
      ) {
        continue;
      }

      const after = previewInvariantSnapshot();
      const invariant = JSON.stringify(before) === JSON.stringify(after);
      if (state.metrics) {
        state.metrics.preview_only_contract = {
          invariant,
          before,
          after,
          padding: { ...state.previewPadding },
        };
      }
      document.documentElement.dataset.previewContractInvariant = String(invariant);
      document.documentElement.dataset.previewContractParamsHash = String(after?.params_hash ?? "");
      document.documentElement.dataset.previewContractPlyHash = String(after?.ply_hash ?? "");
      if (!invariant) throw new Error("preview-only contract changed training parameters, metrics, or PLY payload");
      if (state.previewMode === "splats") showCanvas("gpu");
      state.previewAppliedAlphaBackground = appliedAlphaBackground;
      state.previewAppliedRevision = requestedRevision;
      rendered = true;
    }
    return rendered;
  } finally {
    if (refreshEpoch === state.previewRefreshEpoch) {
      state.previewRefreshPending = false;
      publishState();
    }
  }
}

function invalidatePreviewRefresh() {
  state.previewGeneration += 1;
  state.previewRefreshEpoch += 1;
  state.previewRequestedRevision += 1;
  state.previewAppliedRevision = state.previewRequestedRevision;
  state.previewRefreshPending = false;
  state.previewRefreshPromise = Promise.resolve(false);
}

async function refreshOutsidePreview({ recovery = false } = {}) {
  if (state.webGpuRecoveryPending && !recovery) return false;
  state.previewRequestedRevision += 1;
  if (state.running) {
    state.previewPadding = previewPaddingSpec(state.image, state.params, false);
    state.previewAppliedAlphaBackground = "";
    state.webgpu.renderer?.presentTrainState(state.image);
    if (state.previewMode === "splats") showCanvas("gpu");
    state.previewAppliedRevision = state.previewRequestedRevision;
    publishState();
    return false;
  }
  if (!state.image || !state.params || !state.webgpu.renderer) {
    state.previewPadding = previewPaddingSpec(state.image, state.params, false);
    state.previewAppliedAlphaBackground = "";
    state.previewAppliedRevision = state.previewRequestedRevision;
    publishState();
    return false;
  }
  if (!state.previewRefreshPending) {
    state.previewRefreshPromise = runPreviewRefreshLoop();
  }
  publishState();
  return state.previewRefreshPromise;
}

function setImageDragover(active) {
  els.dropZone.classList.toggle("dragover", active && !state.image);
  els.viewer.classList.toggle("image-dragover", active && Boolean(state.image));
}


function activateDetailTab(name) {
  const tabs = {
    training: [els.trainingLogTab, els.trainingLogPanel],
    event: [els.eventLogTab, els.eventLogPanel],
    splats: [els.splatsTab, els.splatsPanel],
    export: [els.exportTab, els.exportPanel],
    tilt: [els.tiltTab, els.tiltPanel],
  };
  if (previewModeInputLocked() && name === "splats") name = "training";
  if (trainingLifecycleInputLocked() && name === "export") name = "training";
  const resultAlgorithm = trainedResultAlgorithm();
  const tiltAvailable = resultAlgorithm
    ? Boolean(resultAlgorithm.capabilities.virtualCameras)
    : algorithmUsesVirtualCameras();
  if (name === "tilt" && !tiltAvailable) name = "training";
  if (name !== "tilt" && !els.tiltCanvas.hidden) {
    restorePrimaryCanvas();
    destroyTiltViewer();
  }
  for (const [key, [tab, panel]] of Object.entries(tabs)) {
    const active = key === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  }
  document.documentElement.dataset.activeDetailTab = name;
}


document.documentElement.dataset.algorithm = selectedAlgorithm().id;
// Keep the existing desktop-first status view while saving vertical space on
// phones. This is an initial default only; later resizes preserve user choice.
if (window.matchMedia("(max-width: 520px)").matches) {
  els.trainingStatusDetails.open = false;
}
activateDetailTab("training");
syncLayerOrderDependency();
syncVirtualCameraDependency();
updatePreviewModeControls();
resetSplatAdjustmentControls();
renderSplatInspector();

window.addEventListener("resize", applyCanvasView);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.running && !state.paused) {
    state.paused = true;
    state.visibilityPaused = true;
    setPausedRuntimeControlsEnabled(true);
    setStatus("paused");
    eventLog("training paused because the page became hidden");
  } else if (!document.hidden && state.visibilityPaused) {
    state.visibilityPaused = false;
    eventLog("page visible; training remains paused until Resume");
  }
  publishState();
});
window.addEventListener("pagehide", () => destroyTiltViewer());

if (QA_RUNTIME_ENABLED) {
  window.__image2SplatPaint = {
    snapshot() {
      return { ...document.documentElement.dataset };
    },
  };
  window.__image2GaussianPaint = window.__image2SplatPaint;
}

if (QA_RUNTIME_ENABLED && new URLSearchParams(location.search).has("qa")) {
  els.pathInput.type = "text";
  els.pathInput.hidden = false;
  els.pathInput.setAttribute("aria-label", "QA image path");
  els.pathInput.style.cssText = "grid-column:1/-1;width:100%;";
  els.pathButton.hidden = false;
  els.pathButton.textContent = "Load QA path";
  const qaMetricsData = document.createElement("textarea");
  qaMetricsData.id = "qaMetricsData";
  qaMetricsData.dataset.testid = "qa-metrics-data";
  qaMetricsData.hidden = true;
  const qaMetricsButton = document.createElement("button");
  qaMetricsButton.id = "qaMetricsButton";
  qaMetricsButton.dataset.testid = "qa-metrics-button";
  qaMetricsButton.type = "button";
  qaMetricsButton.textContent = "QA Metrics";
  qaMetricsButton.style.cssText =
    "position:fixed;left:16px;bottom:80px;z-index:20;width:96px;height:30px;font-size:12px;opacity:0.35;";
  qaMetricsButton.addEventListener("click", () => {
    try {
      qaMetricsData.value = JSON.stringify({
        snapshot: window.__image2SplatPaint.snapshot(),
        metrics: window.__flatPhotoTest.metricsSummary(),
        benchmark: window.__flatPhotoTest.benchmarkSummary(),
        ply_contract: state.params ? inspectPlyContract() : null,
        edge_containment_probe: window.__flatPhotoTest.edgeContainmentProbe(),
      });
      document.documentElement.dataset.qaMetricsBytes = String(qaMetricsData.value.length);
      document.documentElement.dataset.qaMetricsError = "";
    } catch (error) {
      qaMetricsData.value = "";
      document.documentElement.dataset.qaMetricsBytes = "";
      document.documentElement.dataset.qaMetricsError = error.message;
    }
  });
  const qaObliqueButton = document.createElement("button");
  qaObliqueButton.id = "qaObliqueButton";
  qaObliqueButton.dataset.testid = "qa-oblique-button";
  qaObliqueButton.type = "button";
  qaObliqueButton.textContent = "QA Oblique";
  qaObliqueButton.style.cssText =
    "position:fixed;left:116px;bottom:80px;z-index:20;width:96px;height:30px;font-size:12px;opacity:0.35;";
  qaObliqueButton.addEventListener("click", async () => {
    qaObliqueButton.disabled = true;
    document.documentElement.dataset.qaObliqueStatus = "running";
    try {
      const report = state.metrics?.oblique_overlap_diagnostics || (
        state.image && state.params && state.webgpu.renderer?.trainState
          ? await state.webgpu.renderer.computeObliqueDiagnostics(state.image, state.params)
          : null
      );
      if (!report) throw new Error("Run a QA virtual-camera training before measuring oblique overlap.");
      if (state.metrics) state.metrics.oblique_overlap_diagnostics = report;
      qaMetricsData.value = JSON.stringify(report);
      document.documentElement.dataset.qaObliqueStatus = "complete";
      document.documentElement.dataset.qaObliqueBytes = String(qaMetricsData.value.length);
      document.documentElement.dataset.qaObliqueError = "";
    } catch (error) {
      qaMetricsData.value = "";
      document.documentElement.dataset.qaObliqueStatus = "failed";
      document.documentElement.dataset.qaObliqueError = error.message;
    } finally {
      qaObliqueButton.disabled = false;
    }
  });
  const qaDeviceLossButton = document.createElement("button");
  qaDeviceLossButton.id = "qaDeviceLossButton";
  qaDeviceLossButton.dataset.testid = "qa-device-loss-button";
  qaDeviceLossButton.type = "button";
  qaDeviceLossButton.textContent = "QA Device Loss";
  qaDeviceLossButton.style.cssText =
    "position:fixed;left:216px;bottom:80px;z-index:20;width:110px;height:30px;font-size:12px;opacity:0.35;";
  qaDeviceLossButton.addEventListener("click", () => {
    if (!state.webgpu.renderer?.device) return;
    state.webgpu.renderer.device.destroy();
  });
  document.body.append(qaMetricsData, qaMetricsButton, qaObliqueButton, qaDeviceLossButton);
}

// Compatibility aliases exist only for local QA scripts and checkpoints.
if (QA_RUNTIME_ENABLED) window.__flatPhoto3dgs = window.__image2SplatPaint;

detectWebGpu()
  .then((available) => setStatus(available ? "idle" : "gpu unavailable"))
  .catch((error) => {
    state.webgpu = { supported: false, renderer: null, reason: error.message, limits: null, adapterInfo: null };
    els.backendText.textContent = "webgpu unavailable";
    setStatus("gpu unavailable");
    log(`webgpu check failed: ${error.message}`);
  })
  .finally(() => {
    syncTrainSizeUi();
    updateMemoryRecommendation();
    publishState();
  });

if (QA_RUNTIME_ENABLED) window.__flatPhotoTest = {
  loadGeneratedSample,
  structureGuidedProfileBenchmark,
  initialSplatOrientation,
  initialSplatShape,
  initialOrientationStats,
  rectangleConstraintProbe,
  opaquePaintLateSettleFraction,
  opaquePaintLateSettleStartStep,
  opaquePaintStructuralMutationAllowed,
  sharedTiltOrbitRadius,
  optimizerFootprintHistogram,
  brushColorNeutrality,
  colorFinishStartStep,
  phase39ContractProbe,
  layerOrderComparatorProbe,
  layerEfficiencyVariants,
  summarizeLayerEfficiency,
  planarTiltRotation,
  projectPlanarPoint,
  inverseProjectPlanarPoint,
  virtualTiltVariants,
  virtualTiltStepSpec,
  virtualCameraSamplingVariants,
  virtualCameraBag,
  virtualCameraSamplingStepSpec,
  virtualCameraSamplingCountsThroughStep,
  virtualCameraCatalog,
  tiltRiskProfileForSplat,
  summarizeTiltRisk,
  async tiltDefaultRendererProbe(includeFrames = false) {
    const controller = await loadTiltViewer();
    if (!controller) throw new Error("Tilt viewer did not initialize.");
    controller.setCameraMarkersVisible(false);
    await new Promise((resolve) => setTimeout(resolve, 100));
    const before = controller.diagnostics();
    const capturePng = async () => {
      if (!includeFrames) return null;
      const blob = await controller.captureFrameBlob("image/png", 1);
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
        reader.onerror = () => reject(reader.error || new Error("Tilt PNG read failed."));
        reader.readAsDataURL(blob);
      });
    };
    const frontFrame = await capturePng();
    const drive = async (yaw) => {
      els.tiltPitch.value = "0";
      els.tiltYaw.value = String(yaw);
      const applied = await controller.setTiltAndWait(0, yaw);
      els.tiltPitch.value = String(applied.pitch);
      els.tiltYaw.value = String(applied.yaw);
      return {
        diagnostics: controller.diagnostics(),
        frame: await capturePng(),
      };
    };
    const positive = await drive(2);
    const negative = await drive(-2);
    return {
      contract: "playcanvas-default-sort-settled-views",
      before: before.presentation,
      positive: {
        yaw: positive.diagnostics.yaw,
        presentation: positive.diagnostics.presentation,
      },
      negative: {
        yaw: negative.diagnostics.yaw,
        presentation: negative.diagnostics.presentation,
      },
      final_camera: negative.diagnostics.camera,
      frames: includeFrames
        ? { front: frontFrame, positive: positive.frame, negative: negative.frame }
        : null,
    };
  },
  startTraining,
  stopTraining,
  resetTrainingState,
  async loadPathImage(path) {
    els.pathInput.value = path;
    state.lastInputMode = "path";
    await loadPathImage();
  },
  clearImage,
  getState() {
    const requestedDiscreteLayers = Math.max(
      MIN_DISCRETE_LAYER_COUNT,
      Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(state.params?.discreteLayerCount || DEFAULT_DISCRETE_LAYER_COUNT)),
    );
    const discreteDepthValues = state.params?.depthOrder
      ? [...new Set(Array.from(state.params.depthOrder, (value) => Math.min(
          requestedDiscreteLayers - 1,
          Math.floor(Math.max(0, Math.min(1 - Number.EPSILON, value)) * requestedDiscreteLayers),
        )))].sort((a, b) => a - b)
      : [];
    return {
      image: state.image ? {
        width: state.image.width,
        height: state.image.height,
        originalWidth: state.image.originalWidth,
        originalHeight: state.image.originalHeight,
        cacheWidth: state.image.cacheWidth,
        cacheHeight: state.image.cacheHeight,
        inputCacheMaxSide: state.image.inputCacheMaxSide,
        inputCacheResized: state.image.inputCacheResized,
        inputDecodeMode: state.image.inputDecodeMode,
        inputOrientation: state.image.inputOrientation,
        fileName: state.image.fileName,
      } : null,
      running: state.running,
      stopRequested: state.stopRequested,
      metrics: state.metrics,
      webgpu: {
        supported: state.webgpu.supported,
        reason: state.webgpu.reason,
        limits: state.webgpu.limits,
        adapterFeatures: state.webgpu.adapterFeatures || [],
        profile: state.webgpu.profile || { requested: false, timing_backend: "off" },
      },
      recommendation: state.recommendation,
      downloadsEnabled: !els.savePlyButton.disabled && !els.savePngButton.disabled,
      status: els.statusText.textContent,
      backend: els.backendText.textContent,
      algorithm: selectedAlgorithm().id,
      algorithmLabel: selectedAlgorithm().label,
      gpuDensifyEnabled: true,
      tileCullingEnabled: Boolean(els.tileCullingToggle.checked),
      opacityAwareSupport: performanceVariants().opacityAwareSupportMode,
      adaptiveGpuThroughput: performanceVariants().adaptiveGpuThroughput,
      adaptiveGpuBatch: performanceVariants().adaptiveGpuBatch,
      gpuSchedulingMode: performanceVariants().gpuSchedulingMode,
      asyncPresentation: performanceVariants().asyncPresentation,
      metricTileReuse: performanceVariants().metricTileReuse,
      segmentedExactBackward: performanceVariants().segmentedExactBackward,
      fixedPointExactGradient: performanceVariants().fixedPointExactGradient,
      inverseScaleOptimization: performanceVariants().inverseScaleOptimization,
      stageGrowthShares: phase39Variants().stageGrowthShares,
      discreteLayers: {
        enabled: Boolean(state.params?.discreteLayersEnabled),
        layerAwareAccumulation: Boolean(state.params?.layerAwareAccumulationEnabled),
        requested: Number(state.params?.discreteLayerCount || 0),
        moveRadius: Number(state.params?.discreteLayerMoveRadius || 0),
        occupied: discreteDepthValues.length,
        values: discreteDepthValues,
      },
      loss: els.lossText.textContent,
      step: els.stepText.textContent,
      previewRefresh: els.previewRefresh.value,
      previewFrames: state.metrics?.preview_frames || 0,
      lastPreviewStep: state.metrics?.last_preview_step ?? null,
      splatBytes: state.params ? state.params.count * ROW_BYTES : 0,
      capacityProbe: structuredClone(state.capacityProbe),
      gpuMemory: { ...state.gpuMemory },
      tileCounters: state.metrics?.tile_counters ? { ...state.metrics.tile_counters } : null,
    };
  },
  forceDeviceLoss() {
    if (!state.webgpu.renderer?.device) return false;
    state.webgpu.renderer.device.destroy();
    return true;
  },
  capacityCandidates: capacityProbeCandidates,
  capacityPlan(capacity) {
    if (!state.image) throw new Error("Load an image before requesting a capacity plan.");
    const params = state.params || initGaussians(
      state.image,
      Math.min(Number(els.initialSplatCount.value) || DEFAULT_INITIAL_SPLATS, CAPACITY_PROBE_FAST_PATH_MAX),
    );
    return trainingAllocationPlan(state.image, params, capacity);
  },
  activateDetailTab,
  renderSplatInspector,
  resizeCapProbe(width, height, maxSide = DEFAULT_MAX_SIDE) {
    const [resizedWidth, resizedHeight] = resizedSize(width, height, maxSide);
    return { width: resizedWidth, height: resizedHeight, maxSide };
  },
  inputCacheProbe(width, height) {
    const [cacheWidth, cacheHeight] = resizedSize(
      width,
      height,
      INPUT_CACHE_MAX_SIDE,
      INPUT_CACHE_MAX_SIDE,
    );
    return {
      sourceWidth: width,
      sourceHeight: height,
      cacheWidth,
      cacheHeight,
      maxSide: INPUT_CACHE_MAX_SIDE,
      resized: cacheWidth !== width || cacheHeight !== height,
    };
  },
  async exportCurrent(format = "ply", download = false) {
    if (!EXPORT_FORMATS[format]) throw new Error(`Unknown export format: ${format}`);
    updateExportPanel();
    return saveExport({ download, formatKey: format });
  },
  async tiltDiagnostics() {
    const controller = await loadTiltViewer();
    return controller?.diagnostics?.() || null;
  },
  async setTiltAndWait(pitch, yaw) {
    const controller = await loadTiltViewer();
    const result = await controller.setTiltAndWait(pitch, yaw);
    els.tiltPitch.value = String(result.pitch);
    els.tiltYaw.value = String(result.yaw);
    applyTiltInputs();
    return { ...result, diagnostics: controller.diagnostics() };
  },
  async showTrainingViews() {
    await showTiltTrainingViews();
    return structuredClone(state.tilt.teacherViews);
  },
  async obliqueOverlapDiagnostics() {
    if (!state.image || !state.params || !state.webgpu.renderer?.trainState) {
      throw new Error("Finish a WebGPU training run before measuring oblique overlap.");
    }
    const report = await state.webgpu.renderer.computeObliqueDiagnostics(state.image, state.params);
    if (state.metrics) state.metrics.oblique_overlap_diagnostics = report;
    return structuredClone(report);
  },
  trainingViews() {
    return structuredClone(state.tilt.teacherViews);
  },
  algorithmRegistry() {
    return Object.values(ALGORITHM_REGISTRY).map((algorithm) => ({
      id: algorithm.id,
      label: algorithm.label,
      backend: algorithm.backend,
      exports: [...algorithm.exports],
      capabilities: { ...algorithm.capabilities },
    }));
  },
  benchmarkSummary() {
    const m = state.metrics;
    if (!m) return null;
    const performance = m.performance_trace?.at(-1) || null;
    const overlap = m.overlap_diagnostics?.scales?.["1"] || null;
    const hiddenRgbAttribution = m.overlap_diagnostics?.hidden_rgb_attribution || null;
    return {
      contract: {
        algorithm: m.algorithm,
        seed: "deterministic-webgpu",
        image_size: m.image_size,
        steps_requested: m.steps_requested,
        steps_done: m.steps_done,
        initialization: m.initialization,
        initialization_adaptive: m.initialization_adaptive || null,
      },
      quality: {
        psnr_rgb_db: m.final_psnr,
        global_ssim: m.final_global_ssim,
        local_ssim_p10: m.final_regional_ssim?.p10 ?? null,
        l1: m.final_l1,
      },
      alpha: {
        mean_composited: overlap?.mean_composited_alpha ?? m.coverage_stats?.mean ?? null,
        l1: m.final_alpha_l1,
        ssim: m.final_alpha_ssim,
      },
      coverage: {
        background_exposure_ratio: m.coverage_stats?.background_exposure_ratio ?? null,
        background_exposure_count: m.coverage_stats?.background_exposure_count ?? null,
      },
      shape: {
        scale_histogram: m.scale_histogram || m.shape_stats?.scale_histogram || null,
        anisotropy_ratio: m.anisotropy_ratio ?? m.shape_stats?.anisotropy_ratio_mean ?? null,
        axis_px: m.shape_stats?.inspection?.radius_px || null,
      },
      overlap: {
        effective_contributors: overlap?.mean_effective_contributors ?? null,
        color_variance: overlap?.mean_contributor_color_variance ?? null,
      },
      hidden_rgb_noise_check: {
        method: hiddenRgbAttribution ? "WebGPU positive leave-one-out attribution at 0.25x" : "Splats tab: inspect at 0.25x scale on a non-image alpha background",
        automated_metric: hiddenRgbAttribution?.total_positive_harm ?? null,
        top_32_positive_harm: hiddenRgbAttribution?.top_32_positive_harm ?? null,
        attribution: hiddenRgbAttribution,
      },
      execution: {
        elapsed_ms: performance?.elapsed_ms ?? null,
        iterations_per_second: performance?.iterations_per_second ?? null,
        gpu_memory: m.gpu_training_memory || null,
        scheduling_profile: m.scheduling_profile || null,
      },
    };
  },
  metricsSummary() {
    const m = state.metrics;
    if (!m) return null;
    const checkpoints = m.trend_checkpoints || [];
    let ssimDownCount = 0;
    let ssimUpCount = 0;
    let ssimMaxDrop = 0;
    let globalDownCount = 0;
    let globalUpCount = 0;
    let globalMaxDrop = 0;
    let regionalDownCount = 0;
    let regionalUpCount = 0;
    let regionalMaxDrop = 0;
    for (let i = 1; i < checkpoints.length; i += 1) {
      const delta = checkpoints[i].ssim - checkpoints[i - 1].ssim;
      if (delta > 0.0005) ssimUpCount += 1;
      if (delta < -0.0005) {
        ssimDownCount += 1;
        ssimMaxDrop = Math.min(ssimMaxDrop, delta);
      }
      const globalDelta = checkpoints[i].global_ssim - checkpoints[i - 1].global_ssim;
      if (globalDelta > 0.0005) globalUpCount += 1;
      if (globalDelta < -0.0005) {
        globalDownCount += 1;
        globalMaxDrop = Math.min(globalMaxDrop, globalDelta);
      }
      const regionalDelta = checkpoints[i].regional_ssim?.p10 - checkpoints[i - 1].regional_ssim?.p10;
      if (Number.isFinite(regionalDelta) && regionalDelta > 0.0005) regionalUpCount += 1;
      if (Number.isFinite(regionalDelta) && regionalDelta < -0.0005) {
        regionalDownCount += 1;
        regionalMaxDrop = Math.min(regionalMaxDrop, regionalDelta);
      }
    }
    return {
      algorithm: m.algorithm,
      algorithm_label: m.algorithm_label,
      initial_orientation: m.initial_orientation,
      final_evaluation_step: m.final_evaluation_step,
      final_metrics_complete: m.final_metrics_complete,
      initial_l1: m.initial_l1,
      final_l1: m.final_l1,
      initial_rgb_mse: m.initial_rgb_mse,
      final_rgb_mse: m.final_rgb_mse,
      initial_psnr: m.initial_psnr,
      final_psnr: m.final_psnr,
      psnr_trend: m.psnr_trend,
      quality_metric_contract: m.quality_metric_contract,
      initial_alpha_l1: m.initial_alpha_l1,
      final_alpha_l1: m.final_alpha_l1,
      initial_alpha_ssim: m.initial_alpha_ssim,
      final_alpha_ssim: m.final_alpha_ssim,
      initial_alpha_objective: m.initial_alpha_objective,
      final_alpha_objective: m.final_alpha_objective,
      initial_objective_loss: m.initial_objective_loss,
      final_objective_loss: m.final_objective_loss,
      color_objective: m.color_objective,
      initial_ssim: m.initial_ssim,
      final_ssim: m.final_ssim,
      initial_global_ssim: m.initial_global_ssim,
      final_global_ssim: m.final_global_ssim,
      initial_windowed_ssim: m.initial_windowed_ssim,
      final_windowed_ssim: m.final_windowed_ssim,
      initial_regional_ssim: m.initial_regional_ssim,
      final_regional_ssim: m.final_regional_ssim,
      initial_high_frequency: m.initial_high_frequency,
      final_high_frequency: m.final_high_frequency,
      ssim_trend: m.ssim_trend,
      global_ssim_trend: m.global_ssim_trend,
      steps_done: m.steps_done,
      steps_requested: m.steps_requested,
      train_sync_interval: m.train_sync_interval,
      preview_refresh: m.preview_refresh,
      preview_frames: m.preview_frames,
      preview_resolution_restores: m.preview_resolution_restores,
      last_preview_step: m.last_preview_step,
      params_revision: m.params_revision,
      coverage_revision: m.coverage_revision,
      post_train_adjustments: m.post_train_adjustments,
      num_gaussians: m.num_gaussians,
      boundary_leak_count: m.boundary_leak_count,
      boundary_max_leak: m.boundary_max_leak,
      boundary_sigma: m.boundary_sigma,
      outside_render_splat_count: m.outside_render_splat_count,
      outside_render_max_extent: m.outside_render_max_extent,
      shape_stats: m.shape_stats,
      scale_histogram: m.scale_histogram,
      tiny_splat_count: m.tiny_splat_count,
      tiny_splat_ratio: m.tiny_splat_ratio,
      boundary_tiny_splat_count: m.boundary_tiny_splat_count,
      boundary_tiny_splat_ratio: m.boundary_tiny_splat_ratio,
      interior_tiny_splat_count: m.interior_tiny_splat_count,
      interior_tiny_splat_ratio: m.interior_tiny_splat_ratio,
      anisotropy_ratio: m.anisotropy_ratio,
      detail_splat_count: m.detail_splat_count,
      detail_splat_ratio: m.detail_splat_ratio,
      detail_anisotropy_max: m.detail_anisotropy_max,
      surface_anisotropy_max: m.surface_anisotropy_max,
      thin_line_metrics: m.thin_line_metrics,
      finalization_wall_ms: m.finalization_wall_ms,
      final_diagnostic_sample_limit: m.final_diagnostic_sample_limit,
      final_parameter_hash: m.final_parameter_hash ?? null,
      training_evaluation: m.training_evaluation ? structuredClone(m.training_evaluation) : null,
      training_residual_map: m.training_residual_map ? structuredClone(m.training_residual_map) : null,
      param_delta: m.param_delta,
      fusion_events: m.fusion_events,
      refine_events: m.fusion_refine_events,
      relocation_events: m.fusion_refine_events,
      webgpu_relocation_requested: Boolean(m.webgpu_relocation_requested),
      webgpu_relocation: Boolean(m.webgpu_relocation),
      webgpu_relocation_events: m.webgpu_relocation_events,
      webgpu_refine_requested: Boolean(m.webgpu_refine_requested),
      webgpu_refine: Boolean(m.webgpu_refine),
      webgpu_refine_events: m.webgpu_refine_events,
      densify_events: m.densify_events,
      growth_schedule: m.growth_schedule,
      source_detail_splat_distribution: m.source_detail_splat_distribution || null,
      gpu_densify_requested: Boolean(m.webgpu_densify_requested),
      gpu_densify: Boolean(m.webgpu_densify),
      density_counters: m.density_counters,
      render_aware_density: Boolean(m.render_aware_density),
      weighted_mass_redistribution: Boolean(m.weighted_mass_redistribution),
      sgld_2d: Boolean(m.sgld_2d),
      density_horizon: m.density_horizon,
      experimental_variants: m.experimental_variants,
      phase33_variants: m.phase33_variants,
      phase37_variants: m.phase37_variants,
      phase38_variants: m.phase38_variants,
      phase39_variants: m.phase39_variants,
      phase40_variants: m.phase40_variants,
      phase45_variants: m.phase45_variants,
      phase46_variants: m.phase46_variants,
      phase_relative_scale_guard: m.phase_relative_scale_guard
        ? structuredClone(m.phase_relative_scale_guard)
        : null,
      phase45_region_report: m.phase45_region_report,
      mid_training_overdensity_correction: m.mid_training_overdensity_correction
        ? structuredClone(m.mid_training_overdensity_correction)
        : null,
      ...(m.brush_contribution_diagnostics
        ? { brush_contribution_diagnostics: structuredClone(m.brush_contribution_diagnostics) }
        : {}),
      overlap_diagnostics: m.overlap_diagnostics,
      oblique_overlap_diagnostics: m.oblique_overlap_diagnostics || null,
      render_surface_parity: m.render_surface_parity || null,
      color_space_audit: m.color_space_audit ? structuredClone(m.color_space_audit) : null,
      performance_trace: m.performance_trace,
      performance_profile_schedule: m.performance_profile_schedule || {},
      stage_profile: m.stage_profile || [],
      stage_profile_backend: m.stage_profile_backend || "off",
      scheduling_profile: m.scheduling_profile || null,
      gpu_scheduling: m.gpu_scheduling ? structuredClone(m.gpu_scheduling) : null,
      metric_tile_reuse: m.metric_tile_reuse ? structuredClone(m.metric_tile_reuse) : null,
      importance_stats: m.importance_stats,
      residual_destination_oracle: m.residual_destination_oracle
        ? structuredClone(m.residual_destination_oracle)
        : null,
      coverage_stats: m.coverage_stats,
      density_gpu_ms: m.density_gpu_ms,
      relocation_gpu_ms: m.relocation_gpu_ms,
      post_density_annealing: Boolean(m.post_density_annealing),
      tile_culling_enabled: Boolean(m.tile_culling_enabled),
      train_layer_order: Boolean(m.train_layer_order),
      virtual_camera_sampling: m.virtual_camera_sampling
        ? structuredClone(m.virtual_camera_sampling)
        : null,
      virtual_camera_evaluation: m.virtual_camera_evaluation
        ? structuredClone(m.virtual_camera_evaluation)
        : null,
      virtual_depth_stats: m.virtual_depth_stats ? structuredClone(m.virtual_depth_stats) : null,
      gpu_training_memory: m.gpu_training_memory ? structuredClone(m.gpu_training_memory) : null,
      layer_update_interval: m.layer_update_interval,
      layer_update_rate: m.layer_update_rate,
      layer_stage_aware_rate: Boolean(m.layer_stage_aware_rate),
      layer_freeze_fraction: m.layer_freeze_fraction,
      layer_update_count: m.layer_update_count,
      layer_update_first_steps: m.layer_update_first_steps || [],
      layer_update_last_step: m.layer_update_last_step,
      layer_telemetry_enabled: Boolean(m.layer_telemetry_enabled),
      layer_telemetry: m.layer_telemetry || [],
      layer_efficiency: m.layer_efficiency ? structuredClone(m.layer_efficiency) : null,
      surface_layer_prior: m.surface_layer_prior
        ? structuredClone(m.surface_layer_prior)
        : null,
      front_split_children: m.front_split_children
        ? structuredClone(m.front_split_children)
        : null,
      front_footprint_refinement_v2: m.front_footprint_refinement_v2
        ? structuredClone(m.front_footprint_refinement_v2)
        : null,
      brush_detail_v1: m.brush_detail_v1
        ? structuredClone(m.brush_detail_v1)
        : null,
      brush_detail_layer_policy: m.brush_detail_layer_policy
        ? structuredClone(m.brush_detail_layer_policy)
        : null,
      train_layer_color_guard: m.train_layer_color_guard
        ? structuredClone(m.train_layer_color_guard)
        : null,
      brush_local_color_flow: m.brush_local_color_flow
        ? structuredClone(m.brush_local_color_flow)
        : null,
      brush_stroke_persistence: m.brush_stroke_persistence
        ? structuredClone(m.brush_stroke_persistence)
        : null,
      result_render_cache: m.result_render_cache
        ? structuredClone(m.result_render_cache)
        : null,
      current_visibility_compaction: m.current_visibility_compaction
        ? structuredClone(m.current_visibility_compaction)
        : null,
      current_visibility_compaction_events: m.current_visibility_compaction_events
        ? structuredClone(m.current_visibility_compaction_events)
        : [],
      current_visibility_compaction_removed_total: m.current_visibility_compaction_removed_total || 0,
      current_contribution_compaction: m.current_contribution_compaction
        ? structuredClone(m.current_contribution_compaction)
        : null,
      current_contribution_compaction_events: m.current_contribution_compaction_events
        ? structuredClone(m.current_contribution_compaction_events)
        : [],
      current_contribution_compaction_removed_total:
        m.current_contribution_compaction_removed_total || 0,
      current_contribution_compaction_deferred_count:
        m.current_contribution_compaction_deferred_count || 0,
      current_contribution_compaction_deferred_steps:
        m.current_contribution_compaction_deferred_steps
          ? structuredClone(m.current_contribution_compaction_deferred_steps)
          : [],
      current_contribution_compaction_settings: m.current_contribution_compaction_settings
        ? structuredClone(m.current_contribution_compaction_settings)
        : null,
      opaque_paint_late_settle: m.opaque_paint_late_settle
        ? structuredClone(m.opaque_paint_late_settle)
        : null,
      layer_order_delta: m.param_delta?.layerOrder ?? null,
      tile_counters: m.tile_counters || null,
      tile_retry_steps: m.tile_retry_steps || 0,
      tile_retry_events: m.tile_retry_events || [],
      tile_retry_parameter_hash: m.tile_retry_parameter_hash || null,
      cpu_mirror_current: Boolean(m.cpu_mirror_current),
      webgpu_train_stats: m.webgpu_train_stats || null,
      checkpoint_count: checkpoints.length,
      trend_series: checkpoints.map((checkpoint) => ({
        step: checkpoint.step,
        loss: checkpoint.loss,
        alpha_l1: checkpoint.alpha_l1,
        objective_loss: checkpoint.objective_loss,
        global_ssim: checkpoint.global_ssim,
        windowed_ssim: checkpoint.ssim,
        regional_p10: checkpoint.regional_ssim?.p10 ?? null,
        regional_minimum: checkpoint.regional_ssim?.minimum ?? null,
        gradient_l1: checkpoint.high_frequency?.gradient_l1 ?? null,
        gradient_fidelity: checkpoint.high_frequency?.gradient_fidelity ?? null,
        gaussians: checkpoint.gaussians,
      })),
      ssim_checkpoint_stats: {
        count: checkpoints.length,
        up_count: ssimUpCount,
        down_count: ssimDownCount,
        up_ratio: checkpoints.length > 1 ? ssimUpCount / (checkpoints.length - 1) : null,
        max_drop: ssimMaxDrop,
      },
      global_ssim_checkpoint_stats: {
        count: checkpoints.length,
        up_count: globalUpCount,
        down_count: globalDownCount,
        up_ratio: checkpoints.length > 1 ? globalUpCount / (checkpoints.length - 1) : null,
        max_drop: globalMaxDrop,
      },
      regional_p10_checkpoint_stats: {
        count: checkpoints.length,
        up_count: regionalUpCount,
        down_count: regionalDownCount,
        up_ratio: checkpoints.length > 1 ? regionalUpCount / (checkpoints.length - 1) : null,
        max_drop: regionalMaxDrop,
      },
      first_checkpoint: checkpoints[0] || null,
      mid_checkpoint: checkpoints[Math.floor(checkpoints.length / 2)] || null,
      last_checkpoint: checkpoints[checkpoints.length - 1] || null,
    };
  },
  makePlyBytes() {
    return makePly().byteLength;
  },
  inspectPlyContract,
  rotatedBoundaryProbe() {
    const theta = Math.PI / 2;
    const constrained = constrainSplat(0.9, 0, 0.04, 0.07, theta);
    const params = {
      count: 1,
      xy: new Float32Array([constrained.x, constrained.y]),
      scale: new Float32Array([constrained.sx, constrained.sy]),
      theta: new Float32Array([theta]),
    };
    return { constrained, ...boundaryLeakStats(params) };
  },
  edgeContainmentProbe() {
    const theta = 0.31;
    const source = { x: 0.98, y: -0.96, sx: 0.08, sy: 0.025 };
    const zero = constrainSplat(source.x, source.y, source.sx, source.sy, theta, 0);
    const medium = constrainSplat(source.x, source.y, source.sx, source.sy, theta, 1);
    const strict = constrainSplat(source.x, source.y, source.sx, source.sy, theta, 2.5);
    const image = { width: 640, height: 480 };
    const params = {
      count: 4,
      xy: new Float32Array([zero.x, zero.y, zero.x, zero.y, zero.x, zero.y, zero.x, zero.y]),
      scale: new Float32Array([zero.sx, zero.sy, zero.sx, zero.sy, zero.sx, zero.sy, zero.sx, zero.sy]),
      theta: new Float32Array([theta, theta, theta, theta]),
      rgb: new Float32Array([0.2, 0.4, 0.6, 0.2, 0.4, 0.6, 0.2, 0.4, 0.6, 0.2, 0.4, 0.6]),
      opacity: new Float32Array([0.8, 0.8, 0.8, 0.8]),
      boundarySigma: 0,
      layerOrderEnabled: false,
    };
    const ply = inspectPlyContract(makePly(params, image), params, image);
    return {
      source,
      zero,
      medium,
      strict,
      zero_scale_preserved: Math.abs(zero.sx - source.sx) < 1e-8 && Math.abs(zero.sy - source.sy) < 1e-8,
      medium_scale_reduced: medium.sx < zero.sx || medium.sy < zero.sy,
      strict_scale_reduced: strict.sx < medium.sx || strict.sy < medium.sy,
      selected_containment: boundaryLeakStats(params, 0),
      render_footprint: outsideRenderFootprintStats(params),
      ply,
    };
  },
  plyReflectionProbe() {
    const theta = 0.37;
    const constrained = constrainSplat(0.2, -0.1, 0.04, 0.02, theta);
    const params = {
      count: 4,
      xy: new Float32Array([constrained.x, constrained.y, constrained.x, constrained.y, constrained.x, constrained.y, constrained.x, constrained.y]),
      scale: new Float32Array([constrained.sx, constrained.sy, constrained.sx, constrained.sy, constrained.sx, constrained.sy, constrained.sx, constrained.sy]),
      theta: new Float32Array([theta, theta, theta, theta]),
      rgb: new Float32Array([0.2, 0.4, 0.6, 0.2, 0.4, 0.6, 0.2, 0.4, 0.6, 0.2, 0.4, 0.6]),
      opacity: new Float32Array([0.8, 0.8, 0.8, 0.8]),
    };
    return inspectPlyContract(makePly(params), params);
  },
  exportCoverageProbe() {
    const parity = {
      exact: true,
      display_equivalent: true,
      max_abs: 0,
      mean_abs: 0,
      alpha_max_abs: 0,
      premultiplied_max_abs: 0,
    };
    return {
      missing: exportCoverageStatus({ steps_done: 4, coverage_stats: null }),
      stale: exportCoverageStatus({ steps_done: 4, coverage_stats: { step: 3, background_exposure_count: 0 } }),
      safety: exportCoverageStatus({
        steps_done: 4,
        safety_stop: { reason: "probe" },
        coverage_stats: { step: 4, background_exposure_count: 0 },
      }),
      parityMissing: exportCoverageStatus({ steps_done: 4, coverage_stats: { step: 4, background_exposure_count: 0 } }),
      parityMismatch: exportCoverageStatus({
        steps_done: 4,
        coverage_stats: { step: 4, background_exposure_count: 0 },
        render_surface_parity: {
          exact: false,
          display_equivalent: false,
          max_abs: 1,
          mean_abs: 0.001,
          alpha_max_abs: 1,
          premultiplied_max_abs: 1,
        },
      }),
      exposed: exportCoverageStatus({
        steps_done: 4,
        coverage_stats: { step: 4, background_exposure_count: 1 },
        render_surface_parity: parity,
      }),
      verified: exportCoverageStatus({
        steps_done: 4,
        coverage_stats: { step: 4, background_exposure_count: 0 },
        render_surface_parity: parity,
      }),
    };
  },
  async exportBlockedCoverageProbe() {
    const originalMetrics = state.metrics;
    const originalMessage = state.exportMessage;
    const originalExportReady = state.exportReady;
    const formatKey = algorithmSupportsExport("ply") ? "ply" : "png";
    const parity = {
      exact: true,
      display_equivalent: true,
      max_abs: 0,
      mean_abs: 0,
      alpha_max_abs: 0,
      premultiplied_max_abs: 0,
    };
    const probes = {
      missing: { steps_done: 4, coverage_stats: null },
      stale: { steps_done: 4, coverage_stats: { step: 3, background_exposure_count: 0 } },
      safety: {
        steps_done: 4,
        safety_stop: { reason: "probe" },
        coverage_stats: { step: 4, background_exposure_count: 0 },
      },
      parityMissing: { steps_done: 4, coverage_stats: { step: 4, background_exposure_count: 0 } },
      parityMismatch: {
        steps_done: 4,
        coverage_stats: { step: 4, background_exposure_count: 0 },
        render_surface_parity: {
          exact: false,
          display_equivalent: false,
          max_abs: 1,
          mean_abs: 0.001,
          alpha_max_abs: 1,
          premultiplied_max_abs: 1,
        },
      },
      exposed: {
        steps_done: 4,
        coverage_stats: { step: 4, background_exposure_count: 1 },
        render_surface_parity: parity,
      },
    };
    const result = {};
    try {
      for (const [name, metrics] of Object.entries(probes)) {
        state.metrics = metrics;
        state.exportReady = true;
        try {
          await saveExport({ download: false, formatKey });
          result[name] = { blocked: false, message: "export unexpectedly succeeded" };
        } catch (error) {
          result[name] = {
            blocked: /Export is not ready/.test(error.message),
            message: error.message,
          };
        }
      }
    } finally {
      state.metrics = originalMetrics;
      state.exportMessage = originalMessage;
      state.exportReady = originalExportReady;
      updateExportPanel();
      publishState();
    }
    return result;
  },
};
