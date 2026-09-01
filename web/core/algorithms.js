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
  if (id === FLOW_SPLAT_FUSION_ALGORITHM_ID && Image2SplatPaintFlowBirthLinks.selectedPath() === "internal-bend") {
    return {...algorithm, initialize: Image2SplatPaintInternalBend.initialize, backend: "shared-brush-webgpu",
      capabilities: {...algorithm.capabilities, kernelShape: "opaque-brush", opaqueLayeredPaint: true,
        minimumOpacity: true, requiresLayerOrder: true, internalBend: true}};
  }
  if (id === FLOW_SPLAT_FUSION_ALGORITHM_ID && Image2SplatPaintFlowBirthLinks.selectedPath() === "birth-linked") {
    return {
      ...algorithm,
      initialize: Image2SplatPaintFlowBirthLinks.initialize,
      backend: "shared-brush-webgpu",
      capabilities: { ...algorithm.capabilities, kernelShape: "opaque-brush", opaqueLayeredPaint: true,
        minimumOpacity: true, requiresLayerOrder: true, configurableLayerCount: true, contributionCleanup: true, flowBirthLinked: true },
    };
  }
  return algorithm;
}

function stageAwareGrowthDefaultForAlgorithm(algorithm = selectedAlgorithm()) {
  void algorithm;
  return true;
}

function trainedResultAlgorithm() {
  if ((!state.params && !state.flowSplatResult) || !state.metrics?.algorithm) return null;
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

function selectedRectanglePaintShape() {
  const value = document.querySelector("#rectanglePaintShape")?.value;
  return value === "opaque-brush" ? "opaque-brush" : "rectangle";
}

function algorithmUsesRectangleKernel(algorithm = selectedAlgorithm()) {
  if (algorithm?.id === RECTANGLE_SPLATS_ALGORITHM_ID) {
    return selectedRectanglePaintShape() === "rectangle";
  }
  return algorithm.capabilities.kernelShape === "rectangle";
}

function algorithmUsesLayeredOpaqueBrush(algorithm = selectedAlgorithm()) {
  if (algorithm?.id === RECTANGLE_SPLATS_ALGORITHM_ID) {
    return selectedRectanglePaintShape() === "opaque-brush";
  }
  return algorithm.capabilities.kernelShape === "opaque-brush";
}

function algorithmUsesFlowSplatFusion(algorithm = selectedAlgorithm()) {
  return Boolean(algorithm.capabilities.flowSplatFusion);
}

function algorithmUsesCurveSplatChain(algorithm = selectedAlgorithm()) {
  return Boolean(algorithm.capabilities.curveSplatChain);
}

function algorithmUsesFlowStrokeTraining(algorithm = selectedAlgorithm()) {
  return algorithmUsesFlowSplatFusion(algorithm) || algorithmUsesCurveSplatChain(algorithm);
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
  // Rectangle long-axis allowance in radians. Missing saved params retain the old exact lock.
  config[127] = shape === "rectangle"
    ? normalizedRectangleOrientationTolerance(params?.rectangleOrientationTolerance) * Math.PI / 200
    : 0;
  if (params?.internalBendKey) config[54] = 0; // Fixed layer ownership; rawDepth learns bend only.
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

function normalizedRectangleOrientationTolerance(value, fallback = 0) {
  return clampNumber(value, 0, 100, fallback);
}

function selectedRectangleOrientationTolerance() {
  // Keep the existing run/snapshot percentage contract; only the UI uses degrees.
  return selectedRectangleOrientationToleranceDegrees() / 90 * 100;
}

function selectedRectangleOrientationToleranceDegrees() {
  return clampNumber(
    document.querySelector("#rectangleOrientationTolerance")?.value,
    0,
    90,
    DEFAULT_RECTANGLE_ORIENTATION_TOLERANCE * 0.9,
  );
}

function constrainedRectangleTheta(theta, sx, sy, orientation = DEFAULT_RECTANGLE_ORIENTATION, tolerancePercent = 0) {
  const normalized = normalizedRectangleOrientation(orientation);
  if (normalized === "free") return theta;
  const longAxisIsX = sx >= sy;
  const target = normalized === "vertical"
    ? (longAxisIsX ? Math.PI * 0.5 : 0)
    : (longAxisIsX ? 0 : Math.PI * 0.5);
  const tolerance = normalizedRectangleOrientationTolerance(tolerancePercent) * Math.PI / 200;
  if (tolerance <= 0) return target;
  if (tolerance >= Math.PI * 0.5) return theta;
  // Long-axis orientation repeats every half turn. Keep the nearest half turn
  // so a tapered Rectangle does not flip its narrow edge when it is already allowed.
  const offset = theta - target;
  const delta = offset - Math.floor((offset + Math.PI * 0.5) / Math.PI) * Math.PI;
  return theta + Math.min(tolerance, Math.max(-tolerance, delta)) - delta;
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
  const toleranceDegrees = normalizedRectangleOrientationTolerance(params.rectangleOrientationTolerance) * 0.9;
  let maxAspectRatio = 1;
  let maxOrientationError = 0;
  let violationCount = 0;
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
    if (Math.asin(Math.min(1, orientationError)) > toleranceDegrees * Math.PI / 180 + 0.00001) violationCount += 1;
  }
  return {
    count: params.count,
    max_aspect_ratio: maxAspectRatio,
    orientation,
    max_orientation_error: orientation === "free" ? null : maxOrientationError,
    tolerance_degrees: orientation === "free" ? null : toleranceDegrees,
    max_deviation_degrees: orientation === "free" ? null : Math.asin(Math.min(1, maxOrientationError)) * 180 / Math.PI,
    violation_count: violationCount,
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
  // A balanced flat-headed Brush contour. Most of the stable quartic body is
  // retained while a sixth-order long-axis component reduces ellipse-like ends.
  let u4 = u2 * u2;
  let q = 0.55 * u4 + 0.45 * u4 * u2 + v2 * v2;
  let denominator = max(0.0001, 2.0 * feather);
  let t = clamp((q - (1.0 - feather)) / denominator, 0.0, 1.0);
  let baseKernel = 1.0 - t * t * (3.0 - 2.0 * t);
  if (baseKernel <= 0.00000001) {
    return OilKernelSample(0.0, vec2<f32>(0.0), 0.0);
  }
  let dKernelDq = -6.0 * t * (1.0 - t) / denominator;
  let dVdU = -bendGradient / width - v * widthGradient / width;
  let dVdTaper = -v * widthTaperGradient / width;
  let dQdU = 0.55 * 4.0 * u * u2 +
    0.45 * 6.0 * u * u2 * u2 + 4.0 * v * v2 * dVdU;
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
  // q=1 is the connected Brush contour. sqrt(q) is implicit contour progress,
  // with a finite zero derivative at the center.
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
  const frontlessSampling = Boolean(QA_RUNTIME_ENABLED && (
    overrides.frontlessSampling === true || query.get("virtual-camera-frontless") === "1"
  ));
  const uniformCameras = !hasSlotOverride && requestedUiSharePercent >= 100;
  const slots = frontlessSampling
    ? cameraCount
    : uniformCameras
      ? cameraCount + 1
      : DEFAULT_VIRTUAL_CAMERA_POOL_SLOTS;
  const requestedVirtualSlots = Number.isFinite(Number(overrides.virtualSlots))
    ? Number(overrides.virtualSlots)
    : QA_RUNTIME_ENABLED && query.has("virtual-camera-slots")
      ? queryNumber("virtual-camera-slots", DEFAULT_VIRTUAL_CAMERA_SLOTS)
      : requestedUiSharePercent / 100 * slots;
  const requestedSharePercent = hasSlotOverride
    ? requestedVirtualSlots / slots * 100
    : requestedUiSharePercent;
  const virtualSlots = frontlessSampling
    ? slots
    : uniformCameras
      ? cameraCount
      : Math.max(1, Math.min(slots - 1, Math.round(requestedVirtualSlots)));
  const gradientBalance = virtualCameraGradientBalance(virtualSlots, slots);
  const requestedFrontAnchorWeight = Number.isFinite(Number(overrides.frontGradientAnchorWeight))
    ? Number(overrides.frontGradientAnchorWeight)
    : queryNumber("virtual-camera-front-anchor", NaN);
  const frontGradientAnchorWeight = QA_RUNTIME_ENABLED && Number.isFinite(requestedFrontAnchorWeight)
    ? Math.max(0, Math.min(1, requestedFrontAnchorWeight))
    : gradientBalance.frontGradientAnchorWeight;
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
      : queryNumber("virtual-camera-depth-center-weight", DEFAULT_VIRTUAL_DEPTH_CENTER_WEIGHT))),
    depthSmoothnessWeight: Math.max(0, Math.min(10, Number.isFinite(Number(overrides.depthSmoothnessWeight))
      ? Number(overrides.depthSmoothnessWeight)
      : queryNumber("virtual-camera-depth-smoothness-weight", DEFAULT_VIRTUAL_DEPTH_SMOOTHNESS_WEIGHT))),
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
    frontlessSampling,
    uniformCameras,
    mode: frontlessSampling
      ? "frontless-virtual-sampling"
      : uniformCameras
        ? "uniform-all-cameras"
        : "weighted-virtual-share",
    cameraCount,
    maxAngleDegrees,
    fovDegrees,
    requestedSharePercent,
    effectiveSharePercent: virtualSlots / slots * 100,
    frontGradientAnchorWeight,
    effectiveGradientSharePercent:
      gradientBalance.sampledVirtualShare / (1 + frontGradientAnchorWeight) * 100,
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
  // Give every virtual update a weak canonical-front anchor so alternating
  // views cannot drive isolated splats toward saturated RGB. Above a 50%
  // sampled-view share, strengthen that anchor enough to keep the aggregate
  // objective at most 50% virtual. Camera sampling frequency is unchanged.
  const lowShareAnchor = sampledVirtualShare * 0.5;
  const highShareBalance = sampledVirtualShare * 2 - 1;
  const frontGradientAnchorWeight = Math.max(
    0,
    Math.min(1, Math.max(lowShareAnchor, highShareBalance)),
  );
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
