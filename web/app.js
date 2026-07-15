const ROW_BYTES = 32;
const SH_C0 = 0.28209479177387814;
const MB = 1024 * 1024;
const GB = 1024 * MB;
const MAX_INPUT_FILE_BYTES = 128 * MB;
const MAX_INPUT_DECODED_PIXELS = 64_000_000;
const DEFAULT_MAX_SIDE = 1600;
const DEFAULT_INITIAL_SPLATS = 2048;
const DEFAULT_FINAL_SPLATS = 8192;
const AUTO_SPLATS_MAX = 65535;
const AUTO_INITIAL_SPLATS_MIN = 500;
const AUTO_FINAL_SPLATS_MIN = 3000;
const MANUAL_SPLATS_MAX = 1048576;
const CAPACITY_PROBE_FAST_PATH_MAX = 262144;
const CAPACITY_PROBE_TIERS = [262144, 524288, 786432, 1048576];
const DEFAULT_LR_SCALE = 1;
const DEFAULT_POSITION_LR = 0.00016;
const DEFAULT_COLOR_LR = 0.0025;
const DEFAULT_OPACITY_LR = 0.05;
const DEFAULT_SCALE_LR = 0.002;
const DEFAULT_ROTATION_LR = 0.001;
const DEFAULT_THETA_ALIGN_LR = 0.005;
const DEFAULT_MAX_ANISOTROPY = 8;
const DEFAULT_SURFACE_ANISOTROPY = 6;
const DEFAULT_BOUNDARY_SIGMA = 0;
const DEFAULT_DSSIM_WEIGHT = 0.2;
const DEFAULT_SGLD_NOISE_LR = 0.02;
const BOUNDARY_SIGMA = 2.5;
const RENDER_SIGMA = 4;
const MIN_SPLAT_SCALE = 0.0015;
const BACKGROUND_EXPOSURE_EPSILON = 1e-8;
const MIP_PIXEL_SIGMA = 0.35;
const DENSITY_EVENT_SLOTS = 22;
const PHASE33_IMPORTANCE_EMA = 0.05;
const PHASE33_COVERAGE_TARGET = 0.05;
const PHASE33_COVERAGE_LOSS_WEIGHT = 0.02;
const PHASE33_COVERAGE_DENSITY_STRENGTH = 0.15;
const PHASE33_COARSE_MAX_SIDE = 512;
const CURRICULUM_COARSE_MIN_SIDE = 256;
const CURRICULUM_COARSE_DIVISOR = 4;
const CURRICULUM_COARSE_FRACTION = 1 / 7;
const CURRICULUM_DENSITY_FRACTION = 3 / 7;
const CURRICULUM_GROWTH_FRACTION = 6 / 7;
const DEFAULT_ADC_RECYCLE_RATE = 0.25;
const DEFAULT_ADC_LATE_RECYCLE_RATE = 0.10;
const DEFAULT_ADC_SPLIT_SIGNAL_THRESHOLD = 0.0003;
const DEFAULT_ADC_SPLIT_RESIDUAL_THRESHOLD = 0.0025;
const ADC_RECOVERY_DECAY_STEPS = 250;
const EXPERIMENTAL_REFINE_EVERY = 50;
const EXPERIMENTAL_ADC_INTERVAL_FOR_7000 = 3000;
const DENSIFY_WARMUP_FRACTION = 0.1;
const DENSIFY_WARMUP_MAX_STEPS = 700;
const DEFAULT_TRAIN_SYNC_INTERVAL = 8;
const REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 9;
const DEFAULT_MAX_METRIC_INTERVAL = 100;
const MAX_PREVIEW_PADDING_PX = 256;
const MAX_PREVIEW_PADDING_FRACTION = 0.2;
const DEFAULT_LOCAL_COLOR_ANCHOR_WEIGHT = 0.01;
const DEFAULT_ALPHA_LOSS_WEIGHT = 0.2;
const DEFAULT_ALPHA_TARGET = 0.99;
const LAYER_CODE_RANGE = 0.24;
// Large enough to survive common 3DGS depth-sort quantization, but only 0.5%
// of the exported plane's two-unit long side at the extrema.
const PLY_LAYER_DEPTH_SPAN = 1e-2;
const DEFAULT_TILT_SPLIT_ANGLE_DEGREES = 5;
const DEFAULT_TILT_SPLIT_COLOR_THRESHOLD = 0.08;
const DEFAULT_TILT_SPLIT_SHRINK = 0.8;
const DEFAULT_MAX_PLANAR_SCALE = 0.03;
const DEFAULT_VIRTUAL_TILT_INTERVAL = 32;
const DEFAULT_VIRTUAL_TILT_WEIGHT = 0.25;
const DEFAULT_VIRTUAL_ORDER_PENALTY_WEIGHT = 0;
const DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE = 4;
const VIRTUAL_TILT_DIRECTIONS = Object.freeze([
  [1, 0], [-1, 0], [0, 1], [0, -1],
]);
const LAYER_TRAIN_INTERVAL = 500;
const EXACT_GRADIENT_STRIDE = 16;
const OVERLAP_METRIC_STRIDE = 16;
const TILE_SIZE = 16;
const TILE_INDEX_FACTOR = 64;
const TILE_INDEX_INITIAL_HEADROOM = 1.5;
const TILE_INDEX_GROWTH_HEADROOM = 1.25;
const TILE_OFFSET_OVERFLOW_BIT = 0x80000000;
const TILE_OFFSET_VALUE_MASK = 0x7fffffff;
const DEFAULT_GROWTH_FRACTION = 0.15;
const DEFAULT_GROWTH_SIGNAL_THRESHOLD = 0.0003;
const STAGE_AWARE_GROWTH_RESERVE = 0.30;
const GEOMETRY_PRECOMPUTE_STRIDE_BYTES = 80;
const PACKED_OPTIMIZER_STRIDE_BYTES = 128;
const METRIC_TILE_STRIDE = 33;
const PHASE45_REGION_GRID = 8;
const PHASE45_REGION_COUNT = PHASE45_REGION_GRID * PHASE45_REGION_GRID;
const PHASE45_REGION_STRIDE = 24;
const HIGH_ITERATION_CONFIRM = 50000;
const PERFORMANCE_PROFILE_QUERY_CAPACITY = 32;
const APP_ASSET_BASE_URL = new URL(".", document.currentScript?.src || location.href);
const TILT_VIEWER_MODULE_URL = new URL("tilt-viewer.mjs", APP_ASSET_BASE_URL).href;
const TILT_CAMERA_MODULE_URL = new URL("tilt-camera.mjs", APP_ASSET_BASE_URL).href;
const PRODUCT_NAME = "Image2SplatPaint";
const PRODUCT_FORMAT = "image2splatpaint-web";
const PLANAR_GAUSSIAN_ALGORITHM_ID = "planar-gaussian";
const ALGORITHM_REGISTRY = Object.freeze({
  [PLANAR_GAUSSIAN_ALGORITHM_ID]: Object.freeze({
    id: PLANAR_GAUSSIAN_ALGORITHM_ID,
    label: "Planar Gaussian",
    backend: "custom-webgpu",
    initialize: initGaussians,
    train: trainPlanarGaussian,
    exports: Object.freeze(["png", "ply"]),
    capabilities: Object.freeze({ render: true, metrics: true, png: true, ply: true, tilt: true }),
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

function selectedAlgorithm() {
  const id = document.querySelector("#algorithmSelect")?.value || PLANAR_GAUSSIAN_ALGORITHM_ID;
  const algorithm = ALGORITHM_REGISTRY[id];
  if (!algorithm) throw new Error(`Algorithm is not available: ${id}`);
  return algorithm;
}

function algorithmSupportsExport(formatKey) {
  return selectedAlgorithm().exports.includes(formatKey);
}

function planarTiltRotation(pitchRadians, yawRadians) {
  const cp = Math.cos(pitchRadians);
  const sp = Math.sin(pitchRadians);
  const cy = Math.cos(yawRadians);
  const sy = Math.sin(yawRadians);
  return [
    cy, 0, sy,
    sp * sy, cp, -sp * cy,
    -cp * sy, sp, cp * cy,
  ];
}

function projectPlanarPoint(point, pitchRadians, yawRadians, cameraDistance = DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE, z = 0, frame = { width: 1, height: 1 }) {
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
  const cameraZ = r[6] * worldX + r[7] * worldY + r[8] * z;
  const denominator = cameraDistance - cameraZ;
  return {
    point: [cameraDistance * cameraX / (frameX * denominator), -cameraDistance * cameraY / (frameY * denominator)],
    depth: cameraZ,
    valid: Number.isFinite(denominator) && denominator > 1e-6,
  };
}

function inverseProjectPlanarPoint(point, pitchRadians, yawRadians, cameraDistance = DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE, frame = { width: 1, height: 1 }) {
  const r = planarTiltRotation(pitchRadians, yawRadians);
  const longSide = Math.max(1, Number(frame.width), Number(frame.height));
  const frameX = Math.max(1, Number(frame.width)) / longSide;
  const frameY = Math.max(1, Number(frame.height)) / longSide;
  const u = Number(point[0]);
  const v = Number(point[1]);
  const h00 = cameraDistance * r[0];
  const h01 = -cameraDistance * (frameY / frameX) * r[1];
  const h10 = -cameraDistance * (frameX / frameY) * r[3];
  const h11 = cameraDistance * r[4];
  const h20 = -frameX * r[6];
  const h21 = frameY * r[7];
  const a00 = u * h20 - h00;
  const a01 = u * h21 - h01;
  const a10 = v * h20 - h10;
  const a11 = v * h21 - h11;
  const determinant = a00 * a11 - a01 * a10;
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-8) {
    return { point: [0, 0], valid: false };
  }
  const rightX = -u * cameraDistance;
  const rightY = -v * cameraDistance;
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
    vec3<f32>(cy, 0.0, sy),
    vec3<f32>(sp * sy, cp, -sp * cy),
    vec3<f32>(-cp * sy, sp, cp * cy)
  );
}

fn virtual_project_point(point: vec2<f32>, z: f32) -> vec3<f32> {
  if (!virtual_tilt_enabled()) { return vec3<f32>(point, z); }
  let rotation = virtual_tilt_rotation();
  let longSide = max(cfg(0u), cfg(1u));
  let frame = vec2<f32>(cfg(0u), cfg(1u)) / max(longSide, 1.0);
  let source = vec3<f32>(frame.x * point.x, -frame.y * point.y, z);
  let camera = vec3<f32>(dot(rotation.row0, source), dot(rotation.row1, source), dot(rotation.row2, source));
  let distance = max(cfg(59u), 2.0);
  let denominator = max(distance - camera.z, 0.000001);
  return vec3<f32>(distance * camera.x / (frame.x * denominator), -distance * camera.y / (frame.y * denominator), camera.z);
}

fn virtual_inverse_point(point: vec2<f32>) -> vec3<f32> {
  if (!virtual_tilt_enabled()) { return vec3<f32>(point, 1.0); }
  let rotation = virtual_tilt_rotation();
  let distance = max(cfg(59u), 2.0);
  let longSide = max(cfg(0u), cfg(1u));
  let frame = vec2<f32>(cfg(0u), cfg(1u)) / max(longSide, 1.0);
  let h00 = distance * rotation.row0.x;
  let h01 = -distance * (frame.y / frame.x) * rotation.row0.y;
  let h10 = -distance * (frame.x / frame.y) * rotation.row1.x;
  let h11 = distance * rotation.row1.y;
  let h20 = -frame.x * rotation.row2.x;
  let h21 = frame.y * rotation.row2.y;
  let a00 = point.x * h20 - h00;
  let a01 = point.x * h21 - h01;
  let a10 = point.y * h20 - h10;
  let a11 = point.y * h21 - h11;
  let determinant = a00 * a11 - a01 * a10;
  if (abs(determinant) < 0.00000001) { return vec3<f32>(0.0, 0.0, 0.0); }
  let rhs = -point * distance;
  let source = vec2<f32>(
    (rhs.x * a11 - a01 * rhs.y) / determinant,
    (a00 * rhs.y - rhs.x * a10) / determinant
  );
  let valid = all(source >= vec2<f32>(-1.0)) && all(source <= vec2<f32>(1.0));
  return vec3<f32>(source, select(0.0, 1.0, valid));
}

fn virtual_layer_depth(packedTag: f32) -> f32 {
  if (cfg(45u) <= 0.5) { return 0.0; }
  let order = clamp(fract(packedTag) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  return (order - 0.5) * ${PLY_LAYER_DEPTH_SPAN};
}

fn virtual_camera_depth(center: vec2<f32>, packedTag: f32) -> f32 {
  if (!virtual_tilt_enabled()) { return fract(packedTag); }
  return virtual_project_point(center, virtual_layer_depth(packedTag)).z;
}
`;

function qaOverrides(name) {
  return QA_RUNTIME_ENABLED && globalThis[name] && typeof globalThis[name] === "object"
    ? globalThis[name]
    : {};
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

function aspectAwareGridEnabled() {
  const query = new URLSearchParams(globalThis.location?.search || "");
  if (QA_RUNTIME_ENABLED && query.get("qa") === "1" && query.has("aspect-grid")) {
    return query.get("aspect-grid") !== "0";
  }
  return true;
}

function performanceVariants() {
  const overrides = qaOverrides("__image2GaussianPerformance");
  const query = new URLSearchParams(globalThis.location?.search || "");
  const geometryQuery = query.get("geometry-cache");
  const quadBackwardQuery = query.get("quad-backward");
  const exactTileQuery = query.get("exact-tile-intersection");
  return {
    tileCooperativeRenderer: overrides.tileCooperativeRenderer === true,
    geometryPrecompute: typeof overrides.geometryPrecompute === "boolean"
      ? overrides.geometryPrecompute
      : QA_RUNTIME_ENABLED && geometryQuery === "1",
    quadExactBackward: typeof overrides.quadExactBackward === "boolean"
      ? overrides.quadExactBackward
      : QA_RUNTIME_ENABLED && query.has("quad-backward")
        ? quadBackwardQuery !== "0"
        : false,
    exactTileIntersection: typeof overrides.exactTileIntersection === "boolean"
      ? overrides.exactTileIntersection
      : QA_RUNTIME_ENABLED && query.has("exact-tile-intersection")
        ? exactTileQuery !== "0"
        : false,
  };
}

function phase33Variants() {
  const overrides = qaOverrides("__flatPhotoPhase33");
  const enabled = (name, fallback) => (typeof overrides[name] === "boolean" ? overrides[name] : fallback);
  const finite = (name, fallback) => (Number.isFinite(Number(overrides[name])) ? Number(overrides[name]) : fallback);
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
  return {
    // ADC relocation always mutates a destination while reading a source. Role
    // reservation is therefore a correctness invariant, not a quality variant.
    singleSourceClaim: true,
    densifyInterval: Math.max(1, Math.min(1000, Math.round(finite("densifyInterval", Number(document.querySelector("#densifyInterval")?.value) || 100)))),
    growthFraction: Math.max(0.001, Math.min(1, finite("growthFraction", controlNumber("#growthPercentage", DEFAULT_GROWTH_FRACTION * 100) / 100))),
    growthSignalThreshold: Math.max(0, Math.min(1000, finite("growthSignalThreshold", controlNumber("#growthSignalThreshold", DEFAULT_GROWTH_SIGNAL_THRESHOLD)))),
    adcSplitSignalThreshold: Math.max(0, Math.min(1000, finite("adcSplitSignalThreshold", controlNumber("#adcSplitSignalThreshold", DEFAULT_ADC_SPLIT_SIGNAL_THRESHOLD)))),
    adcSplitResidualThreshold: Math.max(0, Math.min(1000, finite("adcSplitResidualThreshold", controlNumber("#adcSplitResidualThreshold", DEFAULT_ADC_SPLIT_RESIDUAL_THRESHOLD)))),
    adcRecycleRate: Math.max(0, Math.min(1, finite("adcRecycleRate", controlNumber("#adcRecyclePercentage", DEFAULT_ADC_RECYCLE_RATE * 100) / 100))),
    adcLateRecycleRate: Math.max(0, Math.min(1, finite("adcLateRecycleRate", controlNumber("#adcLateRecyclePercentage", DEFAULT_ADC_LATE_RECYCLE_RATE * 100) / 100))),
    stageAwareGrowth: typeof overrides.stageAwareGrowth === "boolean"
      ? overrides.stageAwareGrowth
      : Boolean(document.querySelector("#stageAwareGrowth")?.checked),
    qaGrowthComparisons: QA_RUNTIME_ENABLED && overrides.qaGrowthComparisons === true,
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
  return {
    enabled: typeof overrides.enabled === "boolean"
      ? overrides.enabled
      : QA_RUNTIME_ENABLED && query.has("virtual-tilt")
        ? query.get("virtual-tilt") !== "0"
        : false,
    interval: Math.max(4, Math.min(1000, Math.round(finite("interval", queryNumber("virtual-tilt-interval", DEFAULT_VIRTUAL_TILT_INTERVAL))))),
    weight: Math.max(0, Math.min(1, finite("weight", queryNumber("virtual-tilt-weight", DEFAULT_VIRTUAL_TILT_WEIGHT)))),
    cameraDistance: Math.max(2, Math.min(32, finite("cameraDistance", queryNumber("virtual-tilt-distance", DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE)))),
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
      weight: 1,
      orderPenaltyWeight: variants.orderPenaltyWeight,
      progress,
      angleDegrees: 0,
    };
  }
  const angleDegrees = stage === "mid" ? 5 : progress < CURRICULUM_GROWTH_FRACTION ? 15 : 30;
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
    weight: variants.weight * angleWeight,
    orderPenaltyWeight: variants.orderPenaltyWeight,
    progress,
    angleDegrees,
  };
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
  const alphaLoss = enabled("alphaLoss", true, "phase40Alpha");
  const alphaLossInput = Number(document.querySelector("#alphaLossWeight")?.value);
  const dualBackgroundInput = Boolean(document.querySelector("#dualBackgroundToggle")?.checked);
  const dualBackgroundWeightInput = Number(document.querySelector("#dualBackgroundWeight")?.value);
  return {
    localColorAnchor,
    localColorAnchorWeight: localColorAnchor
      ? Math.max(0, Math.min(0.2, finite("localColorAnchorWeight", DEFAULT_LOCAL_COLOR_ANCHOR_WEIGHT, "phase40AnchorWeight")))
      : 0,
    alphaLoss,
    alphaLossWeight: alphaLoss
      ? Math.max(0, Math.min(1, finite("alphaLossWeight", Number.isFinite(alphaLossInput) ? alphaLossInput : DEFAULT_ALPHA_LOSS_WEIGHT, "phase40AlphaWeight")))
      : 0,
    dualBackground: enabled("dualBackground", dualBackgroundInput, "dualBackground"),
    dualBackgroundWeight: Math.max(0, Math.min(1, finite("dualBackgroundWeight", Number.isFinite(dualBackgroundWeightInput) ? dualBackgroundWeightInput : 0.25, "dualBackgroundWeight"))),
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

function layerOptimizationSettings(step, steps, stage, variants = phase46Variants()) {
  const stageMultiplier = variants.stageAwareRate
    ? stage === "coarse" ? 1 : stage === "mid" ? 0.5 : 0.2
    : 1;
  const freezeStep = Math.round(steps * variants.freezeFraction);
  const enabled = variants.freezeFraction > 0 && step <= freezeStep;
  return {
    interval: variants.layerUpdateInterval,
    rate: variants.layerUpdateRate * stageMultiplier,
    enabled,
    due: enabled && step % variants.layerUpdateInterval === 0,
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
    qualityRecovery: qualityRecoveryVariants(),
  };
}

const els = {
  viewer: document.querySelector(".viewer"),
  viewControls: document.querySelector(".view-controls"),
  previewCanvas: document.querySelector("#previewCanvas"),
  gpuCanvas: document.querySelector("#gpuCanvas"),
  tiltCanvas: document.querySelector("#tiltCanvas"),
  tiltFrameOverlay: document.querySelector("#tiltFrameOverlay"),
  previewImageFrame: document.querySelector("#previewImageFrame"),
  actualSizeButton: document.querySelector("#actualSizeButton"),
  fitViewButton: document.querySelector("#fitViewButton"),
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  trainSize: document.querySelector("#trainSize"),
  initialSplatCount: document.querySelector("#initialSplatCount"),
  finalSplatCount: document.querySelector("#finalSplatCount"),
  capacityMode: document.querySelector("#capacityMode"),
  algorithmSelect: document.querySelector("#algorithmSelect"),
  stepCount: document.querySelector("#stepCount"),
  previewRefresh: document.querySelector("#previewRefresh"),
  tileCullingToggle: document.querySelector("#tileCullingToggle"),
  trainLayerOrder: document.querySelector("#trainLayerOrder"),
  layerUpdateInterval: document.querySelector("#layerUpdateInterval"),
  positionLearningRate: document.querySelector("#positionLearningRate"),
  colorLearningRate: document.querySelector("#colorLearningRate"),
  opacityLearningRate: document.querySelector("#opacityLearningRate"),
  alphaLossWeight: document.querySelector("#alphaLossWeight"),
  dualBackgroundToggle: document.querySelector("#dualBackgroundToggle"),
  stageAwareGrowth: document.querySelector("#stageAwareGrowth"),
  dualBackgroundWeight: document.querySelector("#dualBackgroundWeight"),
  scaleLearningRate: document.querySelector("#scaleLearningRate"),
  rotationLearningRate: document.querySelector("#rotationLearningRate"),
  thetaAlignRate: document.querySelector("#thetaAlignRate"),
  maxAnisotropy: document.querySelector("#maxAnisotropy"),
  maxPlanarScale: document.querySelector("#maxPlanarScale"),
  boundarySigma: document.querySelector("#boundarySigma"),
  detailCoherence: document.querySelector("#detailCoherence"),
  adcSplitInterval: document.querySelector("#adcSplitInterval"),
  adcResetInterval: document.querySelector("#adcResetInterval"),
  densifyInterval: document.querySelector("#densifyInterval"),
  growthPercentage: document.querySelector("#growthPercentage"),
  growthSignalThreshold: document.querySelector("#growthSignalThreshold"),
  adcSplitSignalThreshold: document.querySelector("#adcSplitSignalThreshold"),
  adcSplitResidualThreshold: document.querySelector("#adcSplitResidualThreshold"),
  adcRecyclePercentage: document.querySelector("#adcRecyclePercentage"),
  adcLateRecyclePercentage: document.querySelector("#adcLateRecyclePercentage"),
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
  stepText: document.querySelector("#stepText"),
  splatText: document.querySelector("#splatText"),
  imageSizeText: document.querySelector("#imageSizeText"),
  lossText: document.querySelector("#lossText"),
  ssimText: document.querySelector("#ssimText"),
  regionalSsimText: document.querySelector("#regionalSsimText"),
  backgroundPixelText: document.querySelector("#backgroundPixelText"),
  outsideSplatText: document.querySelector("#outsideSplatText"),
  gpuMemoryText: document.querySelector("#gpuMemoryText"),
  boundaryText: document.querySelector("#boundaryText"),
  backendText: document.querySelector("#backendText"),
  statusText: document.querySelector("#statusText"),
  log: document.querySelector("#log"),
  clearLogButton: document.querySelector("#clearLogButton"),
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
  tiltCameraMode: document.querySelector("#tiltCameraMode"),
  tiltPositionValue: document.querySelector("#tiltPositionValue"),
  tiltRadiusValue: document.querySelector("#tiltRadiusValue"),
  tiltFovValue: document.querySelector("#tiltFovValue"),
  tiltProjectionError: document.querySelector("#tiltProjectionError"),
  tiltSweepButton: document.querySelector("#tiltSweepButton"),
  tiltSweepStopButton: document.querySelector("#tiltSweepStopButton"),
  tiltSweepProgress: document.querySelector("#tiltSweepProgress"),
  tiltSweepSummary: document.querySelector("#tiltSweepSummary"),
  tiltContactSheet: document.querySelector("#tiltContactSheet"),
  splatsEmpty: document.querySelector("#splatsEmpty"),
  splatsContent: document.querySelector("#splatsContent"),
  splatsMeta: document.querySelector("#splatsMeta"),
  splatAlphaBackground: document.querySelector("#splatAlphaBackground"),
  splatOpacity: document.querySelector("#splatOpacity"),
  splatOpacityValue: document.querySelector("#splatOpacityValue"),
  splatScaleX: document.querySelector("#splatScaleX"),
  splatScaleXValue: document.querySelector("#splatScaleXValue"),
  splatScaleY: document.querySelector("#splatScaleY"),
  splatScaleYValue: document.querySelector("#splatScaleYValue"),
  resetSplatAdjustments: document.querySelector("#resetSplatAdjustments"),
  splatAdjustStatus: document.querySelector("#splatAdjustStatus"),
  exportDescription: document.querySelector("#exportDescription"),
  exportCount: document.querySelector("#exportCount"),
  exportStatus: document.querySelector("#exportStatus"),
};

const state = {
  image: null,
  params: null,
  metrics: null,
  running: false,
  paused: false,
  stopRequested: false,
  webgpu: { supported: false, renderer: null, reason: "checking", limits: null, adapterInfo: null },
  recommendation: null,
  safety: { mode: "on", lastStopReason: "", lastStopEstimateMB: "", lastStopBudgetMB: "", lastRecommended: "" },
  lastPreview: null,
  lastDownload: "",
  lastGpuLoss: null,
  lastInputMode: "",
  imageLoadGeneration: 0,
  exportReady: false,
  exporting: false,
  adjustingSplats: false,
  splatBaseline: null,
  splatAdjustmentVersion: 0,
  splatAdjustmentFrame: 0,
  splatAdjustmentValidationTimer: 0,
  splatAdjustmentChain: Promise.resolve(),
  runtimeSettingsRevision: 0,
  previewGeneration: 0,
  previewRequestedRevision: 0,
  previewAppliedRevision: 0,
  previewAppliedAlphaBackground: "",
  previewRefreshPending: false,
  previewRefreshPromise: Promise.resolve(false),
  previewMode: "original",
  previewPadding: { x: 0, y: 0, width: 0, height: 0, bytes: 0 },
  gpuMemory: { activeBytes: 0, reservedBytes: 0 },
  capacityProbe: { status: "manual", requested: 0, selected: 0, attempts: [], fastPath: true },
  canvasView: { mode: "fit", scale: 1, panX: 0, panY: 0, pointerId: null, lastX: 0, lastY: 0 },
  canvasPointers: new Map(),
  canvasPinch: null,
  visibilityPaused: false,
  layerTelemetryState: null,
  tilt: {
    controller: null,
    abortController: null,
    revision: "",
    loading: false,
    generation: 0,
    plyDigest: "",
    plyByteLength: 0,
    vertices: 0,
    verifiedRevision: "",
    verifiedPlyDigest: "",
    verifiedPlyByteLength: 0,
    sweepRunning: false,
    sweepStopRequested: false,
    sweepResults: [],
    sweepObjectUrls: [],
    sweepStartedAt: 0,
  },
  exportMessage: "Train or stop with verified coverage before export.",
};

const previewCtx = els.previewCanvas.getContext("2d", { willReadFrequently: true });

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

function publishState() {
  updateGpuMemoryStatus();
  updateCapacityStatus();
  const data = document.documentElement.dataset;
  data.status = els.statusText.textContent;
  data.productName = PRODUCT_NAME;
  data.algorithm = selectedAlgorithm().id;
  data.algorithmLabel = selectedAlgorithm().label;
  data.backend = els.backendText.textContent;
  data.running = String(state.running);
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
  data.previewPaddingX = String(state.previewPadding.x);
  data.previewPaddingY = String(state.previewPadding.y);
  data.previewCanvasWidth = String(state.previewPadding.width || els.gpuCanvas.width);
  data.previewCanvasHeight = String(state.previewPadding.height || els.gpuCanvas.height);
  data.previewOnlyBytes = String(state.previewPadding.bytes);
  data.canvasViewMode = state.canvasView.mode;
  data.canvasViewScale = String(state.canvasView.scale);
  updateCanvasViewControls();
  updateTiltControlState();
  els.startButton.disabled = state.running || state.previewRefreshPending || !state.image || !state.webgpu.supported;
  els.resetButton.disabled = state.running || state.previewRefreshPending || !state.image;
  els.clearImageButton.disabled = state.running || state.previewRefreshPending || !state.image;
  const outsidePreviewReady = Boolean(
    state.params &&
    state.metrics?.cpu_mirror_current &&
    state.metrics?.final_readback_step === state.metrics?.steps_done &&
    !state.running,
  );
  els.outsidePreviewToggle.disabled = state.previewRefreshPending || !outsidePreviewReady;
  els.pauseButton.disabled = !state.running;
  els.retryWebGpuButton.disabled = state.running || state.webgpu.supported;
  els.retryWebGpuButton.hidden = state.webgpu.supported;
  els.pauseButton.textContent = state.paused ? "Resume" : "Pause";
  data.resetEnabled = String(!els.resetButton.disabled);
  data.clearImageEnabled = String(!els.clearImageButton.disabled);
  data.pauseEnabled = String(!els.pauseButton.disabled);
  data.step = els.stepText.textContent;
  data.loss = els.lossText.textContent;
  data.ssim = els.ssimText.textContent;
  data.regionalSsimP10 = els.regionalSsimText.textContent;
  data.imageSize = els.imageSizeText.textContent;
  data.backgroundPixels = els.backgroundPixelText.textContent;
  data.outsideSplats = els.outsideSplatText.textContent;
  data.gpuActiveBytes = String(state.gpuMemory.activeBytes);
  data.gpuReservedBytes = String(state.gpuMemory.reservedBytes);
  data.gpuActiveMb = bytesToMB(state.gpuMemory.activeBytes).toFixed(1);
  data.gpuReservedMb = bytesToMB(state.gpuMemory.reservedBytes).toFixed(1);
  data.splatBytes = state.params ? String(state.params.count * ROW_BYTES) : "0";
  data.lastDownload = state.lastDownload;
  data.lastGpuLoss = state.lastGpuLoss === null ? "" : String(state.lastGpuLoss);
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
  data.initializationMode = "image-rgb";
  data.previewRefreshInput = els.previewRefresh.value;
  data.blendMode = "standard-alpha";
  data.gpuDensifyEnabled = "true";
  data.tileCullingEnabled = String(Boolean(els.tileCullingToggle?.checked));
  data.trainLayerOrderInput = String(Boolean(els.trainLayerOrder?.checked));
  data.layerUpdateIntervalInput = els.layerUpdateInterval.value;
  data.positionLearningRateInput = els.positionLearningRate.value;
  data.colorLearningRateInput = els.colorLearningRate.value;
  data.opacityLearningRateInput = els.opacityLearningRate.value;
  data.alphaLossWeightInput = els.alphaLossWeight.value;
  data.dualBackgroundInput = String(Boolean(els.dualBackgroundToggle.checked));
  data.stageAwareGrowthInput = String(Boolean(els.stageAwareGrowth.checked));
  data.dualBackgroundWeightInput = els.dualBackgroundWeight.value;
  data.scaleLearningRateInput = els.scaleLearningRate.value;
  data.rotationLearningRateInput = els.rotationLearningRate.value;
  data.thetaAlignRateInput = els.thetaAlignRate.value;
  data.maxAnisotropyInput = els.maxAnisotropy.value;
  data.maxPlanarScaleInput = els.maxPlanarScale.value;
  data.boundarySigmaInput = els.boundarySigma.value;
  data.adaptiveDetailInput = "true";
  data.detailCoherenceInput = els.detailCoherence.value;
  data.adcSplitIntervalInput = els.adcSplitInterval.value;
  data.adcResetIntervalInput = els.adcResetInterval.value;
  data.densifyIntervalInput = els.densifyInterval.value;
  data.growthPercentageInput = els.growthPercentage.value;
  data.growthSignalThresholdInput = els.growthSignalThreshold.value;
  data.adcSplitSignalThresholdInput = els.adcSplitSignalThreshold.value;
  data.adcSplitResidualThresholdInput = els.adcSplitResidualThreshold.value;
  data.adcRecyclePercentageInput = els.adcRecyclePercentage.value;
  data.adcLateRecyclePercentageInput = els.adcLateRecyclePercentage.value;
  data.safetyMode = state.safety.mode;
  data.safetyStopReason = state.safety.lastStopReason;
  data.safetyStopEstimateMb = state.safety.lastStopEstimateMB;
  data.safetyStopBudgetMb = state.safety.lastStopBudgetMB;
  data.safetyRecommended = state.safety.lastRecommended;
  if (state.recommendation) {
    data.memoryBudgetMb = String(state.recommendation.budgetMB);
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
    const lastAdcReset = state.metrics.adc_reset_events?.[state.metrics.adc_reset_events.length - 1];
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
    data.finalGlobalSsim = String(state.metrics.final_global_ssim ?? "");
    data.finalWindowedSsim = String(state.metrics.final_windowed_ssim ?? state.metrics.final_ssim ?? "");
    data.finalLocalP10 = String(state.metrics.final_regional_ssim?.p10 ?? "");
    data.finalAlphaL1 = String(state.metrics.final_alpha_l1 ?? "");
    data.finalAlphaSsim = String(state.metrics.final_alpha_ssim ?? "");
    data.trainingElapsedMs = String(lastPerformance?.elapsed_ms ?? "");
    data.trainingIterationsPerSecond = String(lastPerformance?.iterations_per_second ?? "");
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
    data.geometryCacheRequested = String(Boolean(state.metrics.webgpu_train_stats?.geometry_precompute_requested));
    data.geometryCacheEnabled = String(Boolean(state.metrics.webgpu_train_stats?.geometry_precompute_enabled));
    data.geometryCacheReason = String(state.metrics.webgpu_train_stats?.geometry_precompute_reason ?? "");
    data.geometryCacheRequiredBindings = String(state.metrics.webgpu_train_stats?.geometry_precompute_limits?.requiredStorageBuffersPerShaderStage ?? "");
    data.geometryCacheAvailableBindings = String(state.metrics.webgpu_train_stats?.geometry_precompute_limits?.maxStorageBuffersPerShaderStage ?? "");
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
    data.adcSplitSignalThreshold = String(state.metrics.adc_controls?.split_signal_threshold ?? "");
    data.adcSplitResidualThreshold = String(state.metrics.adc_controls?.split_residual_threshold ?? "");
    data.adcRecyclePercentage = String(state.metrics.adc_controls?.recycle_percentage ?? "");
    data.adcLateRecyclePercentage = String(state.metrics.adc_controls?.late_recycle_percentage ?? "");
    data.lastAdcRecycled = String(lastAdcReset?.recycled ?? "");
    data.lastAdcRecyclePercentage = String(lastAdcReset?.adc_recycle_percentage ?? "");
    data.lastAdcLateRecyclePercentage = String(lastAdcReset?.adc_late_recycle_percentage ?? "");
    data.adcResetEvents = String(state.metrics.adc_reset_events?.length || 0);
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
    data.dualBackground = String(Boolean(state.metrics.dual_background));
    data.dualBackgroundWeight = String(state.metrics.dual_background_weight ?? "");
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
    data.algorithm = state.metrics.algorithm || "";
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
    data.boundaryLeakCount = "";
    data.boundaryMaxLeak = "";
    data.backgroundExposureCount = "";
    data.backgroundExposureRatio = "";
    data.defaultOutput = "";
    data.lossCount = "";
    data.ssimCount = "";
    data.windowedSsimCount = "";
    data.algorithm = "";
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

function resizedSize(width, height, longSide) {
  const safeLongSide = clampNumber(longSide, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE);
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

function clampNumber(value, min, max, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

function normalizeStepInteger(value, { min, max, fallback, step = 1 }) {
  const safeMin = Number.isSafeInteger(min) ? min : 0;
  const safeMax = Number.isSafeInteger(max) && max >= safeMin ? max : Number.MAX_SAFE_INTEGER;
  const safeStep = Number.isSafeInteger(step) && step > 0 ? step : 1;
  const fallbackNumber = Number(fallback);
  const safeFallback = Number.isFinite(fallbackNumber) ? fallbackNumber : safeMin;
  const parsed = Number(value);
  const finite = Number.isFinite(parsed) ? parsed : safeFallback;
  const clamped = Math.min(safeMax, Math.max(safeMin, finite));
  const snapped = safeMin + Math.round((clamped - safeMin) / safeStep) * safeStep;
  return Math.min(safeMax, Math.max(safeMin, Math.trunc(snapped)));
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
  for (const name of ["depthOrder", "detailTags"]) {
    if (params[name] && params[name].length !== count) {
      throw new Error(`${context}: ${name} length ${params[name].length} != ${count}`);
    }
  }
  return count;
}

function hexColorToRgb(value, fallback = [0, 0, 0]) {
  const match = /^#([0-9a-f]{6})$/i.exec(String(value || ""));
  if (!match) return [...fallback];
  const packed = Number.parseInt(match[1], 16);
  return [((packed >> 16) & 255) / 255, ((packed >> 8) & 255) / 255, (packed & 255) / 255];
}

function limitNumber(limits, name, fallback) {
  const value = limits?.[name];
  return Number.isFinite(value) ? value : fallback;
}

function bytesToMB(bytes) {
  return bytes / MB;
}

function formatMB(bytes) {
  return `${bytesToMB(bytes).toFixed(bytes < 10 * MB ? 1 : 0)} MB`;
}

function updateGpuMemoryStatus() {
  const snapshot = state.webgpu.renderer?.trainingMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
  state.gpuMemory.activeBytes = Math.max(0, Math.round(snapshot.activeBytes || 0));
  state.gpuMemory.reservedBytes = Math.max(state.gpuMemory.activeBytes, Math.round(snapshot.reservedBytes || 0));
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
  const geometrySupported = Boolean(state.webgpu.renderer?.geometryPrecomputeSupport?.(splats).supported);
  return 256 + EXACT_GRADIENT_STRIDE * 4 + TILE_INDEX_FACTOR * 4 + (geometrySupported ? 32 + GEOMETRY_PRECOMPUTE_STRIDE_BYTES : 0);
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
  const ssimTileCount = Math.ceil(image.width / 8) * Math.ceil(image.height / 8);
  const variants = phase33Variants();
  const stages = Object.hasOwn(prepared, "coarseImage")
    ? prepared
    : makeCurriculumImages(image, variants);
  const coarseImage = stages.coarseImage || null;
  const midImage = stages.midImage || null;
  const geometrySupport =
    prepared.geometrySupport ||
    state.webgpu.renderer?.geometryPrecomputeSupport?.(capacity) ||
    { supported: false, bytes: 0 };
  const optimizerStride = geometrySupport.supported ? PACKED_OPTIMIZER_STRIDE_BYTES : 96;
  const profileEnabled = Boolean(state.webgpu.renderer?.performanceProfile?.timestampQuery);
  const descriptors = [];
  const add = (name, size, storage = false) => {
    if (size > 0) descriptors.push({ name, size: Math.max(4, Math.ceil(size / 4) * 4), storage });
  };
  add("config", 64 * 4, true);
  add("present-config", 16);
  add("packed-stats-config", 16);
  add("target-rgb", image.rgb.byteLength, true);
  add("coarse-target-rgb", coarseImage?.rgb.byteLength || 0, true);
  add("mid-target-rgb", midImage?.rgb.byteLength || 0, true);
  add("target-alpha", image.alpha?.byteLength || image.width * image.height * 4, true);
  add("coarse-target-alpha", coarseImage?.alpha.byteLength || 0, true);
  add("mid-target-alpha", midImage?.alpha.byteLength || 0, true);
  add("error-map", image.width * image.height * 4, true);
  add("stats", capacity * 2 * 4 * 4, true);
  add("density-control", (capacity * 4 + DENSITY_EVENT_SLOTS + 1 + Math.ceil(capacity / 256) * 2 + PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE) * 4, true);
  add("tile-counts", tileCount * 4, true);
  add("tile-offsets", (tileCount + 1) * 4, true);
  add("tile-cursors", tileCount * 4, true);
  add("tile-indices", tilePlan.capacity * 4, true);
  add("tile-control", 16, true);
  add("pixel-state", image.width * image.height * 16, true);
  add("alpha-state", image.width * image.height * 8, true);
  add("loss-gradient", image.width * image.height * 48, true);
  add("exact-gradient", capacity * EXACT_GRADIENT_STRIDE * 4, true);
  add("ssim-tiles", ssimTileCount * 64, true);
  add("optimizer-state", capacity * optimizerStride, true);
  add("xy", capacity * 2 * 4, true);
  add("transform", capacity * 4 * 4, true);
  add("color", capacity * 4 * 4, true);
  add("geometry", geometrySupport.supported ? geometrySupport.bytes : 0, true);
  add("readback", capacity * 10 * 4);
  add("growth-signal-readback", 4);
  if (profileEnabled) {
    add("exact-backward-telemetry", 32, true);
    add("exact-backward-telemetry-readback", 32);
    add("profile-resolve", PERFORMANCE_PROFILE_QUERY_CAPACITY * 8);
    add("profile-readback", PERFORMANCE_PROFILE_QUERY_CAPACITY * 8);
  }
  return { descriptors, tilePlan, coarseImage, midImage, geometrySupport };
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
  };
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
  const sourceWidth = state.image.originalWidth || state.image.width;
  const sourceHeight = state.image.originalHeight || state.image.height;
  const [width, height] = resizedSize(sourceWidth, sourceHeight, trainSize);
  return { width, height };
}

function imagePixelEstimate(trainSize) {
  const size = estimatedImageSizeFor(trainSize);
  return size.width * size.height;
}

function sideFromPixelBudget(pixelBudget, trainSize) {
  if (!state.image) return Math.sqrt(pixelBudget);
  const sourceWidth = state.image.originalWidth || state.image.width;
  const sourceHeight = state.image.originalHeight || state.image.height;
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
  const alphaStateBytes = pixels * 8;
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

function autoBudgetBytes(limits) {
  return autoBudgetInfo(limits).budgetBytes;
}

function memoryBudgetBytes() {
  return autoBudgetBytes(state.webgpu.limits);
}

function computeBudgetFor(trainSize, finalSplats, steps) {
  const limits = state.webgpu.limits || {};
  trainSize = Math.round(clampNumber(trainSize, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE));
  finalSplats = normalizeUiSplatCount(finalSplats, DEFAULT_FINAL_SPLATS);
  steps = normalizeStepInteger(steps, { min: LIMITS.stepsMin, max: LIMITS.stepsMax, fallback: 7000 });
  const budgetBytes = memoryBudgetBytes();
  const memoryHint = browserMemoryHintBytes();
  const autoBudget = autoBudgetInfo(limits);
  const current = estimateGpuMemory(trainSize, finalSplats);
  const maxBuffer = limitNumber(limits, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(limits, "maxStorageBufferBindingSize", 128 * MB);
  const maxTexture = limitNumber(limits, "maxTextureDimension2D", LIMITS.trainSizeMax);
  const hardPixelLimit = Math.floor(Math.min(maxBuffer, maxStorage) / 24);
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
    overHardLimit: current.metricsBytes > Math.min(maxBuffer, maxStorage),
  };
}

function imageBasedSplatCounts(image = state.image) {
  if (!image) return { initial: DEFAULT_INITIAL_SPLATS, final: DEFAULT_FINAL_SPLATS };
  const pixels = Math.max(1, image.width * image.height);
  const previousEstimate = clampNumber(pixels / 96, 1024, AUTO_SPLATS_MAX, DEFAULT_FINAL_SPLATS);
  const final = roundDownStep(Math.max(AUTO_FINAL_SPLATS_MIN, previousEstimate / 2), 4);
  const previousInitial = roundDownStep(
    clampNumber(final / 16, AUTO_INITIAL_SPLATS_MIN, 4096, DEFAULT_INITIAL_SPLATS),
    4,
  );
  const initial = roundDownStep(Math.min(final, previousInitial * 2), 4);
  return { initial: Math.min(initial, final), final };
}

function applyLoadedImageSplatEstimate() {
  const counts = imageBasedSplatCounts();
  els.initialSplatCount.value = String(counts.initial);
  els.finalSplatCount.value = String(counts.final);
}

function updateImageSizeStatus() {
  els.imageSizeText.textContent = state.image ? `${state.image.width} x ${state.image.height}` : "-";
}

function computeRecommendation() {
  const trainSize = Math.round(clampNumber(els.trainSize.value, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE));
  const finalSplats = normalizeUiSplatCount(els.finalSplatCount.value, DEFAULT_FINAL_SPLATS);
  const steps = normalizeStepInteger(els.stepCount.value, { min: LIMITS.stepsMin, max: LIMITS.stepsMax, fallback: 7000 });
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

function updateMemoryRecommendation({ reconcileSplatCounts = true } = {}) {
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

function resizeLoadedImageToMaxSide(maxSide) {
  if (!state.image) return false;
  const sourceWidth = state.image.originalWidth || state.image.width;
  const sourceHeight = state.image.originalHeight || state.image.height;
  const targetSide = Math.round(clampNumber(maxSide, LIMITS.trainSizeMin, LIMITS.trainSizeMax, Math.max(sourceWidth, sourceHeight)));
  const [width, height] = resizedSize(sourceWidth, sourceHeight, targetSide);
  if (width === state.image.width && height === state.image.height) return false;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (state.image.sourceBitmap) {
    ctx.drawImage(state.image.sourceBitmap, 0, 0, width, height);
  } else {
    const source = document.createElement("canvas");
    source.width = state.image.width;
    source.height = state.image.height;
    source.getContext("2d").putImageData(rgbToImageData(state.image.rgb, state.image.width, state.image.height, state.image.alpha), 0, 0);
    ctx.drawImage(source, 0, 0, width, height);
  }
  const imageData = ctx.getImageData(0, 0, width, height);
  const rgb = new Float32Array(width * height * 3);
  const alpha = new Float32Array(width * height);
  for (let i = 0, j = 0, p = 0; i < imageData.data.length; i += 4, j += 3, p += 1) {
    rgb[j] = imageData.data[i] / 255;
    rgb[j + 1] = imageData.data[i + 1] / 255;
    rgb[j + 2] = imageData.data[i + 2] / 255;
    alpha[p] = imageData.data[i + 3] / 255;
  }

  state.webgpu.renderer?.disposeTrainState();
  state.image = {
    ...state.image,
    width,
    height,
    resizeMode: "max-side",
    resizeScale: Math.max(width / state.image.originalWidth, height / state.image.originalHeight),
    rgb,
    alpha,
  };
  updateImageSizeStatus();
  state.params = null;
  state.metrics = null;
  state.previewPadding = previewPaddingSpec(state.image, null, false);
  state.lastDownload = "";
  els.stepText.textContent = "0 / 0";
  els.splatText.textContent = "-";
  els.lossText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.backgroundPixelText.textContent = "-";
  els.outsideSplatText.textContent = "-";
  clearSplatAdjustmentBaseline();
  els.previewCanvas.width = width;
  els.previewCanvas.height = height;
  els.gpuCanvas.width = width;
  els.gpuCanvas.height = height;
  fitCanvases(width, height);
  setPreviewMode("original");
  updateDownloads(false);
  setStatus("image loaded");
  updateMemoryRecommendation();
  return true;
}

async function loadFile(file) {
  if (state.running) {
    throw new Error("Stop training before loading another image.");
  }
  destroyTiltViewer({ restoreCanvas: true });
  if (!file?.type?.startsWith("image/")) {
    throw new Error("Choose an image file.");
  }
  if (file.size > MAX_INPUT_FILE_BYTES) {
    throw new Error(`Image file is too large (maximum ${Math.round(MAX_INPUT_FILE_BYTES / MB)} MB).`);
  }
  const loadGeneration = state.imageLoadGeneration + 1;
  state.imageLoadGeneration = loadGeneration;
  state.webgpu.renderer?.disposeTrainState();
  const bitmap = await decodeImageFile(file);
  try {
    if (loadGeneration !== state.imageLoadGeneration) return false;
    if (bitmap.width * bitmap.height > MAX_INPUT_DECODED_PIXELS) {
      throw new Error(`Decoded image is too large (maximum ${MAX_INPUT_DECODED_PIXELS.toLocaleString()} pixels).`);
    }
    els.trainSize.value = String(clampNumber(els.trainSize.value, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE));
    const [width, height] = resizedSize(bitmap.width, bitmap.height, LIMITS.trainSizeMax);
    const loadScale = Math.max(width / bitmap.width, height / bitmap.height);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    const rgb = new Float32Array(width * height * 3);
    const alpha = new Float32Array(width * height);
    for (let i = 0, j = 0, p = 0; i < imageData.data.length; i += 4, j += 3, p += 1) {
      rgb[j] = imageData.data[i] / 255;
      rgb[j + 1] = imageData.data[i + 1] / 255;
      rgb[j + 2] = imageData.data[i + 2] / 255;
      alpha[p] = imageData.data[i + 3] / 255;
    }
    state.image?.sourceBitmap?.close?.();
    state.image = {
      fileName: file.name,
      width,
      height,
      originalWidth: bitmap.width,
      originalHeight: bitmap.height,
      resizeMode: "source-cap",
      resizeScale: loadScale,
      rgb,
      alpha,
      sourceBitmap: canvas,
    };
    if (loadGeneration !== state.imageLoadGeneration) return false;
    applyLoadedImageSplatEstimate();
    updateImageSizeStatus();
    state.params = null;
    state.metrics = null;
    state.previewPadding = previewPaddingSpec(state.image, null, false);
    state.lastDownload = "";
    els.stepText.textContent = "0 / 0";
    els.splatText.textContent = "-";
    els.lossText.textContent = "-";
    els.ssimText.textContent = "-";
    els.regionalSsimText.textContent = "-";
    els.boundaryText.textContent = "-";
    els.backgroundPixelText.textContent = "-";
    els.outsideSplatText.textContent = "-";
    clearSplatAdjustmentBaseline();
    els.dropZone.classList.add("ready");
    els.previewCanvas.width = width;
    els.previewCanvas.height = height;
    els.gpuCanvas.width = width;
    els.gpuCanvas.height = height;
    fitCanvases(width, height);
    setPreviewMode("original");
    updateDownloads(false);
    updateMemoryRecommendation();
    setStatus("image loaded");
    log(`loaded ${file.name} ${width}x${height} from ${bitmap.width}x${bitmap.height} scale=${loadScale.toFixed(3)} sRGB; training resize deferred`);
    return true;
  } finally {
    bitmap.close?.();
  }
}

async function decodeImageFile(file) {
  if (typeof createImageBitmap === "function") {
    try {
      return await createImageBitmap(file);
    } catch (error) {
      log(`createImageBitmap decode failed; using Canvas image decode: ${error.message}`);
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImageElement(url);
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("2D canvas is unavailable for image decode.");
    context.drawImage(image, 0, 0);
    return canvas;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function imagePathCandidates(value) {
  const path = value.trim().replaceAll("\\", "/");
  if (!path) throw new Error("Enter an image path.");
  if (/^file:\/\//.test(path)) {
    if (location.protocol !== "file:") throw new Error("file:// images only work when the app itself is opened with file://.");
    return [path];
  }
  if (/^[a-zA-Z]:\//.test(path)) {
    if (location.protocol === "file:") return [`file:///${path}`];
    throw new Error("Local absolute paths need file:// launch, file input/drop, or a local server path.");
  }
  if (path.startsWith("/Users/") || path.startsWith("/home/")) {
    if (location.protocol === "file:") return [`file://${path}`];
    throw new Error("Local absolute paths need file:// launch, file input/drop, or a local server path.");
  }
  if (/^https?:\/\//.test(path) || path.startsWith("/")) return [path];
  const clean = path.replace(/^\.\//, "");
  return [`../${clean}`, clean, `./${clean}`];
}

async function loadPathImage() {
  const path = els.pathInput.value;
  const name = path.split(/[\\/]/).filter(Boolean).pop() || "path-image";
  const file = await loadFirstImagePath(imagePathCandidates(path), name);
  state.lastInputMode = "path";
  await loadFile(file);
}

async function loadFirstImagePath(urls, name) {
  const errors = [];
  for (const url of urls) {
    try {
      return await loadImageUrlAsFile(url, name);
    } catch (error) {
      errors.push(`${url}: ${error.message}`);
    }
  }
  throw new Error(`image load failed. Tried ${errors.join(" | ")}`);
}

async function loadImageUrlAsFile(url, name) {
  if (location.protocol !== "file:") {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`image fetch failed: ${response.status}`);
    const blob = await response.blob();
    return new File([blob], name, { type: blob.type || "image/jpeg" });
  }
  const image = await loadImageElement(url);
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("image canvas export failed");
  return new File([blob], name.replace(/\.[^.]+$/, ".png"), { type: "image/png" });
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`image load failed: ${url}`));
    image.src = url;
  });
}

async function loadGeneratedSample() {
  const path = "assets/source-images/ramen-photo.jpg";
  const file = await loadFirstImagePath(imagePathCandidates(path), "ramen-photo.jpg");
  state.lastInputMode = "sample";
  await loadFile(file);
}

function showCanvas(kind) {
  els.previewCanvas.hidden = kind !== "preview";
  els.gpuCanvas.hidden = kind !== "gpu";
  els.tiltCanvas.hidden = kind !== "tilt";
  els.tiltFrameOverlay.hidden = kind !== "tilt" || !state.tilt.controller;
  els.viewControls.hidden = kind === "tilt";
  els.previewImageFrame.hidden =
    kind !== "gpu" ||
    state.running ||
    !els.outsidePreviewToggle.checked ||
    state.previewPadding.x <= 0 && state.previewPadding.y <= 0;
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
  state.canvasView.mode = mode;
  state.canvasView.scale = mode === "actual" ? 1 : fittedCanvasScale();
  state.canvasView.panX = 0;
  state.canvasView.panY = 0;
  applyCanvasView();
  publishState();
}

function fitCanvases(width = activePreviewCanvas().width, height = activePreviewCanvas().height) {
  state.canvasView.mode = "fit";
  state.canvasView.scale = fittedCanvasScale(width, height);
  state.canvasView.panX = 0;
  state.canvasView.panY = 0;
  applyCanvasView();
}

function updateCanvasViewControls() {
  const disabled = !state.image;
  els.actualSizeButton.disabled = disabled;
  els.fitViewButton.disabled = disabled;
}

function zoomCanvasAt(clientX, clientY, deltaY) {
  if (!state.image) return;
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
}

function drawRgbToCanvas(rgb, width, height) {
  previewCtx.putImageData(rgbToImageData(rgb, width, height), 0, 0);
  state.lastPreview = { rgb: new Float32Array(rgb), width, height };
}

function updatePreviewModeControls() {
  const hasImage = Boolean(state.image);
  const hasSplats = Boolean(state.params);
  els.originalPreviewButton.disabled = !hasImage;
  els.splatsPreviewButton.disabled = !hasSplats;
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
  if (mode === "original") {
    if (!state.image) return false;
    state.previewMode = "original";
    drawRgbToCanvas(state.image.rgb, state.image.width, state.image.height);
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

function makeCoarseTrainingImage(image, maxSide) {
  const currentSide = Math.max(image.width, image.height);
  if (currentSide <= maxSide) return null;
  const [width, height] = resizedSize(image.width, image.height, maxSide);
  const source = document.createElement("canvas");
  source.width = image.width;
  source.height = image.height;
  source.getContext("2d").putImageData(rgbToImageData(image.rgb, image.width, image.height, image.alpha), 0, 0);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const rgb = new Float32Array(width * height * 3);
  const alpha = new Float32Array(width * height);
  for (let i = 0, j = 0, p = 0; i < pixels.length; i += 4, j += 3, p += 1) {
    rgb[j] = pixels[i] / 255;
    rgb[j + 1] = pixels[i + 1] / 255;
    rgb[j + 2] = pixels[i + 2] / 255;
    alpha[p] = pixels[i + 3] / 255;
  }
  return { width, height, rgb, alpha };
}

function curriculumCoarseMaxSide(fullSide, variants = phase33Variants()) {
  const boundedFullSide = Math.max(1, Math.round(fullSide));
  if (!variants.adaptiveCurriculum) return Math.min(boundedFullSide, variants.coarseMaxSide);
  if (boundedFullSide <= PHASE33_COARSE_MAX_SIDE) return boundedFullSide;
  return Math.min(
    boundedFullSide,
    Math.max(CURRICULUM_COARSE_MIN_SIDE, Math.round(boundedFullSide / CURRICULUM_COARSE_DIVISOR)),
  );
}

function curriculumMidMaxSideForFullSide(fullSide, coarseMaxSide = PHASE33_COARSE_MAX_SIDE) {
  if (fullSide <= coarseMaxSide) return fullSide;
  return Math.max(coarseMaxSide + 1, Math.min(fullSide, Math.round(Math.sqrt(coarseMaxSide * fullSide))));
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
  const baseScale = 1.6 / Math.max(rows, cols);
  // Parameters remain in per-axis image NDC; this only matches initial coverage to each grid cell.
  const baseScaleX = aspectAwareGridEnabled() ? 1.6 / cols : baseScale;
  const baseScaleY = aspectAwareGridEnabled() ? 1.6 / rows : baseScale;
  return {
    rows,
    cols,
    baseScaleX,
    baseScaleY,
    baseScale,
  };
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
    count: params.count,
    xy: new Float32Array(params.xy),
    scale: new Float32Array(params.scale),
    rgb: new Float32Array(params.rgb),
    opacity: new Float32Array(params.opacity),
    theta: new Float32Array(params.theta),
    depthOrder: params.depthOrder ? new Float32Array(params.depthOrder) : initialDepthOrder(params.count),
    detailTags: params.detailTags ? new Float32Array(params.detailTags) : new Float32Array(params.count).fill(1),
    boundarySigma: Number.isFinite(params.boundarySigma) ? params.boundarySigma : selectedBoundarySigma(),
    layerOrderEnabled: Boolean(params.layerOrderEnabled),
    maxAnisotropy: anisotropyLimits.detail,
    surfaceAnisotropy: anisotropyLimits.surface,
    rows: params.rows,
    cols: params.cols,
    bg: params.bg ? new Float32Array(params.bg) : new Float32Array([0, 0, 0]),
  };
}

function nonfiniteParamCount(params) {
  let count = 0;
  for (const values of [params?.xy, params?.scale, params?.rgb, params?.opacity, params?.theta, params?.depthOrder, params?.detailTags]) {
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
    opacityValues.push(opacity);
    sxValues.push(sx);
    syValues.push(sy);
    radiusValues.push(radiusPx);
    anisotropyValues.push(ratio);
    rotationValues.push(rotation);
    if (boundaryAnchored) boundarySplatCount += 1;
    values.push(scale);
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
  const disabled = !current || state.running;
  for (const control of [
    els.splatOpacity,
    els.splatScaleX,
    els.splatScaleY,
    els.splatAlphaBackground,
    els.resetSplatAdjustments,
  ]) {
    control.disabled = disabled;
  }
  document.documentElement.dataset.splatsInspectionReady = String(current);
  document.documentElement.dataset.splatsAdjustmentReady = String(current && !disabled);
  if (!current) return;

  const adjustment = state.metrics?.post_train_adjustments;
  els.splatsMeta.textContent = `${state.params.count.toLocaleString()} splats · step ${state.metrics.steps_done}`;
  els.splatAdjustStatus.textContent = state.adjustingSplats
    ? "Updating the GPU preview and rechecking coverage..."
    : adjustment
      ? `Live preview · revision ${state.metrics.params_revision}`
      : "Move a slider to preview the result.";
  const data = document.documentElement.dataset;
  data.splatsInspectionStep = String(state.metrics.steps_done);
  data.splatsInspectionCount = String(state.params.count);
  data.splatsInspectionNonfinite = String(nonfiniteParamCount(state.params));
  data.splatsParamsRevision = String(state.metrics.params_revision ?? 0);
  data.splatsCoverageRevision = String(state.metrics.coverage_revision ?? "");
}

function updateSplatAdjustmentLabels() {
  els.splatOpacityValue.textContent = `${Number(els.splatOpacity.value).toFixed(2)}x`;
  els.splatScaleXValue.textContent = `${Number(els.splatScaleX.value).toFixed(2)}x`;
  els.splatScaleYValue.textContent = `${Number(els.splatScaleY.value).toFixed(2)}x`;
}

function resetSplatAdjustmentControls() {
  els.splatOpacity.value = "1";
  els.splatScaleX.value = "1";
  els.splatScaleY.value = "1";
  updateSplatAdjustmentLabels();
}

function clearSplatAdjustmentBaseline() {
  window.clearTimeout(state.splatAdjustmentValidationTimer);
  if (state.splatAdjustmentFrame) window.cancelAnimationFrame(state.splatAdjustmentFrame);
  state.splatBaseline = null;
  state.adjustingSplats = false;
  state.splatAdjustmentVersion = 0;
  state.splatAdjustmentFrame = 0;
  els.outsidePreviewToggle.checked = false;
  resetSplatAdjustmentControls();
  els.splatAdjustStatus.textContent = "Ready.";
  renderSplatInspector();
}

function captureSplatAdjustmentBaseline() {
  if (!state.params || !state.metrics?.cpu_mirror_current) return;
  state.splatBaseline = snapshotParams(state.params);
  state.metrics.params_revision = 0;
  state.metrics.coverage_revision = 0;
  state.metrics.post_train_adjustments = null;
  state.splatAdjustmentVersion = 0;
  resetSplatAdjustmentControls();
  renderSplatInspector();
}

function adjustedParamsFromBaseline() {
  const baseline = state.splatBaseline;
  if (!baseline) throw new Error("Finish training before adjusting splats.");
  const opacityMultiplier = clampNumber(els.splatOpacity.value, 0, 2, 1);
  const scaleXMultiplier = clampNumber(els.splatScaleX.value, 0, 5, 1);
  const scaleYMultiplier = clampNumber(els.splatScaleY.value, 0, 5, 1);
  const params = snapshotParams(baseline);
  for (let i = 0; i < params.count; i += 1) {
    const theta = baseline.theta[i];
    const constrained = constrainSplat(
      baseline.xy[i * 2],
      baseline.xy[i * 2 + 1],
      baseline.scale[i * 2] * scaleXMultiplier,
      baseline.scale[i * 2 + 1] * scaleYMultiplier,
      theta,
      baseline.boundarySigma,
      anisotropyLimitForTag(baseline.detailTags?.[i], baseline),
    );
    params.xy[i * 2] = constrained.x;
    params.xy[i * 2 + 1] = constrained.y;
    params.scale[i * 2] = constrained.sx;
    params.scale[i * 2 + 1] = constrained.sy;
    params.theta[i] = theta;
    params.opacity[i] = Math.min(0.999999, Math.max(1e-6, baseline.opacity[i] * opacityMultiplier));
  }
  assertFiniteParams(params, "post-training-adjustment");
  return {
    params,
    values: { opacityMultiplier, scaleXMultiplier, scaleYMultiplier },
  };
}

function splatAlphaRenderOptions() {
  return {
    alphaBackground: hexColorToRgb(els.splatAlphaBackground?.value, [0, 0, 0]),
  };
}

function lockAdjustedExport() {
  state.metrics.coverage_revision = null;
  updateDownloads(false);
  state.exportMessage = "Export is locked until the live adjustment passes coverage validation.";
  updateExportPanel();
}

async function renderLiveSplatAdjustments(version) {
  if (!state.splatBaseline || !state.image || !state.metrics || state.running || version !== state.splatAdjustmentVersion) return;
  const adjusted = adjustedParamsFromBaseline();
  state.params = adjusted.params;
  state.metrics.params_revision = (state.metrics.params_revision ?? 0) + 1;
  state.metrics.post_train_adjustments = adjusted.values;
  lockAdjustedExport();
  state.previewMode = "splats";
  updatePreviewModeControls();
  await refreshOutsidePreview();
  if (version !== state.splatAdjustmentVersion) return;
  renderSplatInspector();
  publishState();
}

async function validateLiveSplatAdjustments(version) {
  if (!state.splatBaseline || !state.image || !state.metrics || state.running || version !== state.splatAdjustmentVersion) return;
  state.adjustingSplats = true;
  renderSplatInspector();
  try {
    await state.webgpu.renderer.uploadTrainState(state.image, state.params, state.params.count);
    await updatePreview(state.metrics.steps_done, true, { present: false });
    if (version !== state.splatAdjustmentVersion) return;
    state.metrics.post_adjustment_overlap_diagnostics = await state.webgpu.renderer.computeOverlapDiagnostics(state.image, state.params);
    state.metrics.post_adjustment_overlap_revision = state.metrics.params_revision;
    state.metrics.coverage_revision = state.metrics.params_revision;
    updateDownloads(exportCoverageStatus().ok);
    state.exportMessage = "Ready to export the adjusted and verified splats.";
    updateExportPanel();
  } catch (error) {
    updateDownloads(false);
    state.exportMessage = `Adjustment failed: ${error.message}`;
    updateExportPanel();
    log(state.exportMessage);
  } finally {
    state.webgpu.renderer?.disposeTrainState();
    state.adjustingSplats = false;
    renderSplatInspector();
    publishState();
    await refreshOutsidePreview();
  }
}

function queueSplatAdjustments({ immediate = false } = {}) {
  updateSplatAdjustmentLabels();
  if (!state.splatBaseline || !state.image || !state.metrics || state.running) return;
  const version = state.splatAdjustmentVersion + 1;
  state.splatAdjustmentVersion = version;
  window.clearTimeout(state.splatAdjustmentValidationTimer);
  const enqueuePreview = () => {
    state.splatAdjustmentFrame = 0;
    state.splatAdjustmentChain = state.splatAdjustmentChain
      .catch(() => {})
      .then(() => renderLiveSplatAdjustments(state.splatAdjustmentVersion))
      .catch((error) => {
        state.exportMessage = `Adjustment failed: ${error.message}`;
        updateExportPanel();
        log(state.exportMessage);
      });
  };
  if (immediate) enqueuePreview();
  else if (!state.splatAdjustmentFrame) state.splatAdjustmentFrame = window.requestAnimationFrame(enqueuePreview);
  state.splatAdjustmentValidationTimer = window.setTimeout(() => {
    state.splatAdjustmentChain = state.splatAdjustmentChain
      .catch(() => {})
      .then(() => validateLiveSplatAdjustments(version));
  }, 240);
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
  const detailTags = new Float32Array(count).fill(1);

  for (let i = 0; i < count; i += 1) {
    const grid = splatGridAt(layout, i);
    const c = constrainSplat(grid.x, grid.y, grid.sx, grid.sy, 0, boundarySigma);
    xy[i * 2] = c.x;
    xy[i * 2 + 1] = c.y;
    scale[i * 2] = c.sx;
    scale[i * 2 + 1] = c.sy;
    sampleImageAt(image, xy[i * 2], xy[i * 2 + 1], rgb, i * 3);
    opacity[i] = 0.98;
    theta[i] = 0;
  }
  return {
    xy, scale, rgb, opacity, theta, depthOrder, detailTags, count,
    rows: layout.rows, cols: layout.cols, bg,
    boundarySigma,
    layerOrderEnabled: Boolean(els.trainLayerOrder?.checked),
  };
}

function initialDepthOrder(count) {
  const values = new Float32Array(Math.max(0, count));
  const denominator = Math.max(1, count - 1);
  for (let i = 0; i < count; i += 1) values[i] = 1 - i / denominator;
  return values;
}

function sampleImageAt(image, x, y, out, offset) {
  const px = Math.max(0, Math.min(image.width - 1, Math.round(((x + 1) * 0.5) * (image.width - 1))));
  const py = Math.max(0, Math.min(image.height - 1, Math.round(((y + 1) * 0.5) * (image.height - 1))));
  const source = (py * image.width + px) * 3;
  out[offset] = image.rgb[source];
  out[offset + 1] = image.rgb[source + 1];
  out[offset + 2] = image.rgb[source + 2];
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

function experimentalGrowthSteps(steps) {
  return curriculumStageStep(steps, CURRICULUM_GROWTH_FRACTION);
}

function curriculumTrainingStage(step, steps, variants, coarseImage, midImage) {
  const coarseEnd = experimentalCoarseSteps(steps, variants.coarseSteps);
  if (variants.coarseToFull && coarseImage && step <= coarseEnd) return "coarse";
  if (variants.threeStageCurriculum && midImage && step <= experimentalDensifySteps(steps)) return "mid";
  return "full";
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

function experimentalAdcInterval(steps) {
  if (steps >= 3000) return EXPERIMENTAL_ADC_INTERVAL_FOR_7000;
  return Math.max(300, Math.round(steps / 4));
}

function experimentalAdcWindow(steps) {
  return Math.max(24, Math.min(160, Math.round(steps * 0.02)));
}

function experimentalSchedule(steps) {
  const densityHorizon = experimentalDensifySteps(steps);
  const warmup = densifyWarmupSteps(densityHorizon);
  const phase38 = phase38Variants();
  const adcInterval = phase38.adcSplitInterval || experimentalAdcInterval(densityHorizon);
  const adcWindow = experimentalAdcWindow(densityHorizon);
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
  const phase45DonorActive = phase45.donorEligibility && (!phase45.firstResetOnly || step <= schedule.resetInterval);
  const detail = selectedLearningRates();
  const shaderStepLimit = mode === 3 ? schedule.densityHorizon : steps;
  return {
    schedule,
    config: new Float32Array([
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
      phase39.adcSplitSignalThreshold,
      phase39.adcSplitResidualThreshold,
      phase39.adcRecycleRate,
      phase39.adcLateRecycleRate,
      layout.baseScaleX,
      layout.baseScaleY,
      detail.boundarySigma,
      detail.surfaceAnisotropy,
      detail.maxPlanarScale,
    ]),
  };
}

function experimentalDensityPhase(step, steps) {
  const schedule = experimentalSchedule(steps);
  if (step > schedule.densityHorizon) return "settle";
  if (step <= schedule.warmup) return "warmup";
  const interval = schedule.adcInterval;
  const phaseOffset = step % interval;
  const inWindow = phaseOffset === 0 || phaseOffset >= interval - schedule.adcWindow;
  return inWindow ? "adc" : "mcmc";
}

function experimentalAdcResetStep(step, steps) {
  const schedule = experimentalSchedule(steps);
  const scheduled = step <= schedule.resetHorizon && step > schedule.warmup && step % schedule.resetInterval === 0;
  if (!scheduled) return false;
  return phase37Variants().secondAdcReset || step <= schedule.densityHorizon;
}

function splatTargetForGrowth(currentCount, finalCount, growthFraction = DEFAULT_GROWTH_FRACTION) {
  if (finalCount <= currentCount) return currentCount;
  const added = Math.max(1, Math.ceil(currentCount * Math.max(0.001, growthFraction)));
  return normalizeActiveSplatCount(Math.min(finalCount, currentCount + added), currentCount);
}

function growthSchedulePlan({
  step,
  steps,
  initialCount,
  currentCount,
  finalCount,
  growthFraction,
  densifyInterval,
  stageAware,
}) {
  const normalTarget = splatTargetForGrowth(currentCount, finalCount, growthFraction);
  const normalIncrement = Math.max(0, normalTarget - currentCount);
  const densityEnd = experimentalDensifySteps(steps);
  const growthEnd = experimentalGrowthSteps(steps);
  if (!stageAware) {
    return {
      mode: "threshold-percentage-cap",
      desiredCount: normalTarget,
      previousDesired: currentCount,
      normalIncrement,
      catchUpLimit: finalCount,
      requestedCount: normalTarget,
      densityEnd,
      growthEnd,
    };
  }

  const warmup = densifyWarmupSteps(densityEnd);
  const range = Math.max(0, finalCount - initialCount);
  const desiredAt = (targetStep) => {
    const segmentProgress = targetStep <= densityEnd
      ? Math.max(0, Math.min(1, (targetStep - warmup) / Math.max(1, densityEnd - warmup))) * (1 - STAGE_AWARE_GROWTH_RESERVE)
      : (1 - STAGE_AWARE_GROWTH_RESERVE) +
        Math.max(0, Math.min(1, (targetStep - densityEnd) / Math.max(1, growthEnd - densityEnd))) * STAGE_AWARE_GROWTH_RESERVE;
    return Math.min(finalCount, initialCount + Math.round(range * segmentProgress));
  };
  const desiredCount = desiredAt(step);
  const previousDesired = desiredAt(Math.max(warmup, step - Math.max(1, densifyInterval)));
  const stageIncrement = Math.max(0, desiredCount - previousDesired);
  const catchUpLimit = Math.min(finalCount, currentCount + stageIncrement * 2);
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
  const detailTags = new Float32Array(targetCount);
  xy.set(params.xy);
  scale.set(params.scale);
  rgb.set(params.rgb);
  opacity.set(params.opacity);
  theta.set(params.theta);
  depthOrder.set(params.depthOrder || initialDepthOrder(oldCount));
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
    detailTags[i] = params.detailTags?.[source] ?? 1;
  }
  return {
    xy, scale, rgb, opacity, theta, depthOrder, detailTags, count: targetCount,
    rows: params.rows, cols: params.cols, bg: params.bg,
    boundarySigma: params.boundarySigma,
    layerOrderEnabled: Boolean(params.layerOrderEnabled),
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
  const points = [];
  const cellSize = 8;
  const cells = new Map();
  const key = (x, y) => `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)}`;
  for (let i = 0; i < params.count; i += 1) {
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
  if (!points.length) return { detail_count: 0, gap_ratio: null, isolated_detail_ratio: null, off_ridge_streak_ratio: null };
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
  }

  const measured = regions.filter((region) => region.count > 0).map((region) => {
    const meanA = region.renderedY / region.count;
    const meanB = region.targetY / region.count;
    const ssim = ssimFromMoments(
      meanA,
      meanB,
      Math.max(0, region.renderedY2 / region.count - meanA ** 2),
      Math.max(0, region.targetY2 / region.count - meanB ** 2),
      region.renderedTargetY / region.count - meanA * meanB,
    );
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
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("adapter unavailable");
    const adapterInfo = adapter.info || null;
    const adapterFeatures = Array.from(adapter.features || []).sort();
    const profileRequested = performanceProfileRequested();
    const timestampQuery = profileRequested && adapter.features.has("timestamp-query");
    const adapterStorageBuffers = Number(adapter.limits?.maxStorageBuffersPerShaderStage || 8);
    if (adapterStorageBuffers < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE) {
      throw new Error(
        `adapter supports ${adapterStorageBuffers} storage buffers per shader stage; ${REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE} required`,
      );
    }
    const requiredLimits = {
      maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
    };
    let device;
    try {
      device = await adapter.requestDevice({
        requiredFeatures: timestampQuery ? ["timestamp-query"] : [],
        requiredLimits,
      });
    } catch (error) {
      if (!timestampQuery) throw error;
      device = await adapter.requestDevice({ requiredLimits });
    }
    const renderer = new WebGpuPreview(device, els.gpuCanvas, {
      profileRequested,
      timestampQuery: timestampQuery && device.features.has("timestamp-query"),
    });
    state.webgpu = {
      supported: true,
      renderer,
      reason: "available",
      limits: device.limits || adapter.limits || null,
      adapterInfo,
      adapterFeatures,
      profile: {
        requested: profileRequested,
        timing_backend: timestampQuery && device.features.has("timestamp-query") ? "timestamp-query" : profileRequested ? "unavailable" : "off",
      },
    };
    device.lost.then((info) => {
      if (state.webgpu.renderer !== renderer) return;
      renderer.deviceLost = true;
      state.stopRequested = true;
      state.paused = false;
      renderer.disposeTrainState();
      state.webgpu = { supported: false, renderer: null, reason: `device lost: ${info.reason || "unknown"}`, limits: null, adapterInfo: null };
      els.backendText.textContent = "webgpu unavailable";
      updateMemoryRecommendation();
      setStatus("error");
      log(state.webgpu.reason);
    });
    els.backendText.textContent = "webgpu available";
    updateMemoryRecommendation();
    log("WebGPU available for train and preview");
  } catch (error) {
    state.webgpu = { supported: false, renderer: null, reason: error.message, limits: null, adapterInfo: null };
    els.backendText.textContent = "webgpu unavailable";
    updateMemoryRecommendation();
    log(`WebGPU failed; training disabled: ${error.message}`);
  }
}

class WebGpuPreview {
  constructor(device, canvas, profile = {}) {
    this.device = device;
    this.canvas = canvas;
    this.context = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: "opaque" });
    this.pipeline = null;
    this.lastLossStats = null;
    this.vertexBuffer = null;
    this.densityBindGroupLayout = null;
    this.growSelectPipeline = null;
    this.distributionPipeline = null;
    this.distributionBlockScanPipeline = null;
    this.distributionBlockSumsPipeline = null;
    this.distributionOffsetPipeline = null;
    this.growRedistributePipeline = null;
    this.growApplyPipeline = null;
    this.relocationSelectPipeline = null;
    this.relocationApplyPipeline = null;
    this.phase45RegionTelemetryPipeline = null;
    this.phase45RegionFinalizePipeline = null;
    this.phase45DonorSafetyPipeline = null;
    this.optimizerResetPipeline = null;
    this.optimizerSourceResetPipeline = null;
    this.geometryPrecomputePipeline = null;
    this.packedStatsExportPipeline = null;
    this.packedStatsImportPipeline = null;
    this.tileCountPipeline = null;
    this.tilePrefixPipeline = null;
    this.tileFillPipeline = null;
    this.tileSortPipeline = null;
    this.renderStatePipeline = null;
    this.tileCooperativeRenderPipeline = null;
    this.ssimTilePipeline = null;
    this.renderGradientPipeline = null;
    this.parallelRenderGradientPipeline = null;
    this.lossGradientPipeline = null;
    this.exactAlphaBackwardPipeline = null;
    this.virtualOrderPenaltyPipeline = null;
    this.exactBackwardTelemetryPipeline = null;
    this.exactOptimizerPipeline = null;
    const performance = performanceVariants();
    this.quadExactBackwardEnabled = performance.quadExactBackward;
    this.exactTileIntersectionEnabled = performance.exactTileIntersection;
    this.pixelMetricsPipeline = null;
    this.overlapMetricsPipeline = null;
    this.alphaLossPipeline = null;
    this.presentPipeline = null;
    this.performanceProfile = {
      requested: Boolean(profile.profileRequested),
      timestampQuery: Boolean(profile.timestampQuery),
    };
  }

  profilePassDescriptor(profileSample, name) {
    if (!profileSample) return undefined;
    if (profileSample.queryCount + 2 > PERFORMANCE_PROFILE_QUERY_CAPACITY) {
      throw new Error(`Performance profile query capacity exceeded before ${name}.`);
    }
    const beginningOfPassWriteIndex = profileSample.queryCount;
    const endOfPassWriteIndex = beginningOfPassWriteIndex + 1;
    profileSample.queryCount += 2;
    profileSample.stages.push({ name, beginningOfPassWriteIndex, endOfPassWriteIndex });
    return {
      timestampWrites: {
        querySet: this.trainState.profileQuerySet,
        beginningOfPassWriteIndex,
        endOfPassWriteIndex,
      },
    };
  }

  dispatchLinear(pass, workgroupCount) {
    const count = Math.max(1, Math.ceil(workgroupCount));
    const limit = Math.max(1, Number(this.device.limits?.maxComputeWorkgroupsPerDimension || 65535));
    const x = Math.min(count, limit);
    const y = Math.ceil(count / x);
    if (y > limit) throw new Error(`Linear compute dispatch ${count} exceeds the 2D WebGPU workgroup limit ${limit}x${limit}.`);
    pass.dispatchWorkgroups(x, y);
  }

  geometryPrecomputeSupport(capacity) {
    const requested = performanceVariants().geometryPrecompute;
    const limits = this.device.limits || {};
    const bytes = Math.max(GEOMETRY_PRECOMPUTE_STRIDE_BYTES, capacity * GEOMETRY_PRECOMPUTE_STRIDE_BYTES);
    const maxBufferSize = Number(limits.maxBufferSize || 0);
    const maxStorageBufferBindingSize = Number(limits.maxStorageBufferBindingSize || 0);
    const maxStorageBuffersPerShaderStage = Number(limits.maxStorageBuffersPerShaderStage || 0);
    const supported =
      requested &&
      bytes <= maxBufferSize &&
      bytes <= maxStorageBufferBindingSize &&
      maxStorageBuffersPerShaderStage >= 8;
    let reason = requested ? "supported" : "not-requested";
    if (requested && bytes > maxBufferSize) reason = "maxBufferSize";
    else if (requested && bytes > maxStorageBufferBindingSize) reason = "maxStorageBufferBindingSize";
    else if (requested && maxStorageBuffersPerShaderStage < 8) reason = "maxStorageBuffersPerShaderStage";
    return {
      requested,
      supported,
      bytes,
      reason,
      limits: {
        requiredStorageBuffersPerShaderStage: 8,
        maxStorageBuffersPerShaderStage,
        maxBufferSize,
        maxStorageBufferBindingSize,
      },
    };
  }

  geometryPrecomputeEnabled() {
    return Boolean(this.trainState?.geometryPrecomputeEnabled && this.trainState?.geometryBuffer);
  }

  renderStatePipelineChoice() {
    const limits = this.device.limits || {};
    const supported =
      Number(limits.maxComputeInvocationsPerWorkgroup || 0) >= TILE_SIZE * TILE_SIZE &&
      Number(limits.maxComputeWorkgroupSizeX || 0) >= TILE_SIZE &&
      Number(limits.maxComputeWorkgroupSizeY || 0) >= TILE_SIZE &&
      Number(limits.maxComputeWorkgroupStorageSize || 0) >= 12288;
    const cooperative =
      performanceVariants().tileCooperativeRenderer &&
      Boolean(els.tileCullingToggle.checked) &&
      supported;
    return {
      pipeline: cooperative ? this.tileCooperativeRenderPipeline : this.renderStatePipeline,
      cooperative,
      supported,
    };
  }

  ensurePipeline(count) {
    const shader = `
struct Uniforms {
  width: f32,
  height: f32,
  count: f32,
  bgR: f32,
  bgG: f32,
  bgB: f32,
  useTiles: f32,
  useEwa: f32,
  sourceWidth: f32,
  sourceHeight: f32,
  viewScaleX: f32,
  viewScaleY: f32,
  reserved: f32,
  alphaBgR: f32,
  alphaBgG: f32,
  alphaBgB: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> xy: array<vec2f>;
@group(0) @binding(2) var<storage, read> transform: array<vec4f>;
@group(0) @binding(3) var<storage, read> color: array<vec4f>;
@group(0) @binding(4) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(5) var<storage, read> tileIndices: array<u32>;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@location(0) position: vec2f) -> VertexOut {
  var out: VertexOut;
  out.position = vec4f(position, 0.0, 1.0);
  out.uv = position;
  return out;
}

fn mip_weight_scale(baseScale: vec2f) -> vec3f {
  let pixelSigma = 0.35 * 2.0 / max(uniforms.sourceWidth, uniforms.sourceHeight);
  let effectiveScale = sqrt(baseScale * baseScale + vec2f(pixelSigma * pixelSigma));
  let compensation = sqrt((baseScale.x * baseScale.y) / max(effectiveScale.x * effectiveScale.y, 0.00000001));
  return vec3f(effectiveScale.x, effectiveScale.y, compensation);
}

fn gaussian_kernel(d: vec2f, c: f32, s: f32, scale: vec2f) -> f32 {
  let r = vec2f(c * d.x + s * d.y, -s * d.x + c * d.y);
  let q = dot(r / scale, r / scale);
  return exp(-0.5 * q);
}

fn tile_offset(index: u32) -> u32 {
  return tileOffsets[index] & 0x7fffffffu;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let viewScale = max(vec2f(uniforms.viewScaleX, uniforms.viewScaleY), vec2f(0.000001));
  let width = max(1u, u32(uniforms.width));
  let height = max(1u, u32(uniforms.height));
  let px = min(width - 1u, u32(in.position.x));
  let py = min(height - 1u, u32(in.position.y));
  let p = vec2f(
    select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u) / viewScale.x,
    select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u) / viewScale.y
  );
  var transmittance = 1.0;
  var rgb = vec3f(0.0);
  let tileCols = max(1u, (u32(uniforms.width) + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u);
  let tileX = min(tileCols - 1u, u32(in.position.x) / ${TILE_SIZE}u);
  let tileRows = max(1u, (u32(uniforms.height) + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u);
  let tileY = min(tileRows - 1u, u32(in.position.y) / ${TILE_SIZE}u);
  let tile = tileY * tileCols + tileX;
  let useOrdered = uniforms.useTiles > 0.5;
  let usePerTile = uniforms.useTiles > 0.5 && uniforms.useTiles < 1.5;
  let tileCapacity = arrayLength(&tileIndices);
  let safeTile = min(tile, max(1u, arrayLength(&tileOffsets)) - 2u);
  let start = select(0u, min(tile_offset(safeTile), tileCapacity), usePerTile);
  let end = select(u32(uniforms.count), min(tile_offset(safeTile + 1u), tileCapacity), usePerTile);
  var cursor = start;
  loop {
    if (cursor >= end) { break; }
    var i = cursor;
    if (useOrdered) { i = tileIndices[cursor]; }
    let d = p - xy[i];
    let t = transform[i];
    if (t.w < 0.5) {
      cursor = cursor + 1u;
      continue;
    }
    let c = cos(t.z);
    let sT = sin(t.z);
    let baseScale = max(t.xy, vec2f(0.0001));
    let mip = mip_weight_scale(baseScale);
    let useEwa = uniforms.useEwa > 0.5;
    var kernel = gaussian_kernel(d, c, sT, mip.xy);
    var compensation = mip.z;
    if (useEwa) {
      let ox = select(0.0, 0.5 / (uniforms.sourceWidth - 1.0), uniforms.sourceWidth > 1.0);
      let oy = select(0.0, 0.5 / (uniforms.sourceHeight - 1.0), uniforms.sourceHeight > 1.0);
      let clampToImage = viewScale.x >= 0.999999 && viewScale.y >= 0.999999;
      let p00 = select(p + vec2f(-ox, -oy), clamp(p + vec2f(-ox, -oy), vec2f(-1.0), vec2f(1.0)), clampToImage);
      let p10 = select(p + vec2f( ox, -oy), clamp(p + vec2f( ox, -oy), vec2f(-1.0), vec2f(1.0)), clampToImage);
      let p01 = select(p + vec2f(-ox,  oy), clamp(p + vec2f(-ox,  oy), vec2f(-1.0), vec2f(1.0)), clampToImage);
      let p11 = select(p + vec2f( ox,  oy), clamp(p + vec2f( ox,  oy), vec2f(-1.0), vec2f(1.0)), clampToImage);
      kernel = 0.25 * (
        gaussian_kernel(p00 - xy[i], c, sT, baseScale) +
        gaussian_kernel(p10 - xy[i], c, sT, baseScale) +
        gaussian_kernel(p01 - xy[i], c, sT, baseScale) +
        gaussian_kernel(p11 - xy[i], c, sT, baseScale)
      );
      compensation = 1.0;
    }
    if (kernel >= 0.0003354626) {
      let alpha = clamp(kernel * color[i].a * compensation, 0.0, 0.99);
      if (alpha >= 0.0039215686) {
        rgb += transmittance * alpha * color[i].rgb;
        transmittance *= 1.0 - alpha;
      }
    }
    cursor = cursor + 1u;
  }
  rgb += transmittance * vec3f(uniforms.alphaBgR, uniforms.alphaBgG, uniforms.alphaBgB);
  return vec4f(rgb, 1.0 - transmittance);
}`;
    if (this.pipeline) return;
    const module = this.device.createShaderModule({ code: shader });
    this.pipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }],
      },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-strip" },
    });
    const verts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this.vertexBuffer = this.device.createBuffer({
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, verts);
  }

  async ensureGeometryPrecomputePipeline() {
    if (!this.geometryPrecomputeEnabled() || this.geometryPrecomputePipeline) return;
    const shader = `
struct Config { values: array<vec4<f32>, 14>, };
struct SplatGeometry {
  centerTrig: vec4<f32>,
  baseEffective: vec4<f32>,
  sampleMip: vec4<f32>,
  pixelBounds: vec4<u32>,
  tileBounds: vec4<u32>,
};
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> geometry: array<SplatGeometry>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

@compute @workgroup_size(64)
fn precompute_geometry(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  if (g >= u32(cfg(2u))) { return; }
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let center = xy[g];
  let t = transform[g];
  let c = cos(t.z);
  let s = sin(t.z);
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
  let useEwa = cfg(26u) > 0.5;
  let sampleScale = select(effective, baseScale, useEwa);
  let mip = select(
    sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001)),
    1.0,
    useEwa
  );
  let pixelPadding = select(
    vec2<f32>(0.0),
    vec2<f32>(
      select(0.0, 0.5 / f32(width - 1u), width > 1u),
      select(0.0, 0.5 / f32(height - 1u), height > 1u)
    ),
    useEwa
  );
  let radius = vec2<f32>(
    ${RENDER_SIGMA} * (abs(c) * sampleScale.x + abs(s) * sampleScale.y),
    ${RENDER_SIGMA} * (abs(s) * sampleScale.x + abs(c) * sampleScale.y)
  ) + pixelPadding;
  let minNorm = max(vec2<f32>(-1.0), center - radius);
  let maxNorm = min(vec2<f32>(1.0), center + radius);
  let pixelSpan = vec2<f32>(f32(width - 1u), f32(height - 1u));
  let minPx = vec2<u32>(floor((minNorm * 0.5 + 0.5) * pixelSpan));
  let maxPx = vec2<u32>(ceil((maxNorm * 0.5 + 0.5) * pixelSpan));
  geometry[g] = SplatGeometry(
    vec4<f32>(center, c, s),
    vec4<f32>(baseScale, effective),
    vec4<f32>(sampleScale, mip, select(0.0, 1.0, t.w >= 0.5)),
    vec4<u32>(minPx, maxPx),
    vec4<u32>(minPx / ${TILE_SIZE}u, maxPx / ${TILE_SIZE}u)
  );
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.geometryPrecomputePipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "precompute_geometry" },
    });
  }

  async encodeGeometryPrecompute(encoder, params, profileSample) {
    if (!this.geometryPrecomputeEnabled()) return false;
    await this.ensureGeometryPrecomputePipeline();
    const front = this.trainState.front;
    const bindGroup = this.device.createBindGroup({
      layout: this.geometryPrecomputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
        { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
        { binding: 3, resource: { buffer: this.trainState.geometryBuffer } },
      ],
    });
    const pass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "geometry"));
    pass.setPipeline(this.geometryPrecomputePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.end();
    return true;
  }

  async ensureTilePipelines() {
    if (this.tileCountPipeline && this.tilePrefixPipeline && this.tileFillPipeline && this.tileSortPipeline) return;
    const geometryEnabled = this.geometryPrecomputeEnabled();
    const exactTileIntersectionEnabled = this.exactTileIntersectionEnabled;
    const geometryDeclaration = geometryEnabled
      ? `struct SplatGeometry {
  centerTrig: vec4<f32>,
  baseEffective: vec4<f32>,
  sampleMip: vec4<f32>,
  pixelBounds: vec4<u32>,
  tileBounds: vec4<u32>,
};
@group(0) @binding(8) var<storage, read_write> geometry: array<SplatGeometry>;`
      : "";
    const geometryBuildFunction = geometryEnabled
      ? `fn build_geometry(g: u32) -> vec4<u32> {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let center = xy[g];
  let t = transform[g];
  let c = cos(t.z);
  let s = sin(t.z);
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
  let useEwa = cfg(26u) > 0.5;
  let sampleScale = select(effective, baseScale, useEwa);
  let mip = select(sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001)), 1.0, useEwa);
  let pixelPadding = select(
    vec2<f32>(0.0),
    vec2<f32>(
      select(0.0, 0.5 / f32(width - 1u), width > 1u),
      select(0.0, 0.5 / f32(height - 1u), height > 1u)
    ),
    useEwa
  );
  let radius = vec2<f32>(
    ${RENDER_SIGMA} * (abs(c) * sampleScale.x + abs(s) * sampleScale.y),
    ${RENDER_SIGMA} * (abs(s) * sampleScale.x + abs(c) * sampleScale.y)
  ) + pixelPadding;
  let minNorm = max(vec2<f32>(-1.0), center - radius);
  let maxNorm = min(vec2<f32>(1.0), center + radius);
  let pixelSpan = vec2<f32>(f32(width - 1u), f32(height - 1u));
  let minPx = vec2<u32>(floor((minNorm * 0.5 + 0.5) * pixelSpan));
  let maxPx = vec2<u32>(ceil((maxNorm * 0.5 + 0.5) * pixelSpan));
  let tileBounds = vec4<u32>(minPx / ${TILE_SIZE}u, maxPx / ${TILE_SIZE}u);
  geometry[g] = SplatGeometry(
    vec4<f32>(center, c, s),
    vec4<f32>(baseScale, effective),
    vec4<f32>(sampleScale, mip, select(0.0, 1.0, t.w >= 0.5)),
    vec4<u32>(minPx, maxPx),
    tileBounds
  );
  return tileBounds;
}`
      : "";
    const countBoundsStatement = geometryEnabled
      ? "var bounds = vec4<u32>(0u); if (cfg(47u) > 0.5 && !virtual_tilt_enabled()) { bounds = build_geometry(g); } else { bounds = tile_bounds(g); }"
      : "let bounds = tile_bounds(g);";
    const fillActiveExpression = geometryEnabled
      ? "select(transform[g].w >= 0.5, geometry[g].sampleMip.w > 0.5, cfg(47u) > 0.5 && !virtual_tilt_enabled())"
      : "transform[g].w >= 0.5";
    const fillBoundsStatement = geometryEnabled
      ? "var bounds = tile_bounds(g); if (cfg(47u) > 0.5 && !virtual_tilt_enabled()) { bounds = geometry[g].tileBounds; }"
      : "let bounds = tile_bounds(g);";
    const exactTileIntersectionFunction = exactTileIntersectionEnabled
      ? `
fn normalized_coordinate(pixel: u32, size: u32) -> f32 {
  return select(0.0, f32(pixel) / f32(size - 1u) * 2.0 - 1.0, size > 1u);
}

fn quadratic_value(dx: f32, dy: f32, a: f32, b: f32, c: f32) -> f32 {
  return a * dx * dx + 2.0 * b * dx * dy + c * dy * dy;
}

fn minimum_quadratic_on_rect(dMin: vec2<f32>, dMax: vec2<f32>, a: f32, b: f32, c: f32) -> f32 {
  if (dMin.x <= 0.0 && dMax.x >= 0.0 && dMin.y <= 0.0 && dMax.y >= 0.0) {
    return 0.0;
  }
  var best = 1.0e30;
  var dx = dMin.x;
  var dy = clamp(-b * dx / c, dMin.y, dMax.y);
  best = min(best, quadratic_value(dx, dy, a, b, c));
  dx = dMax.x;
  dy = clamp(-b * dx / c, dMin.y, dMax.y);
  best = min(best, quadratic_value(dx, dy, a, b, c));
  dy = dMin.y;
  dx = clamp(-b * dy / a, dMin.x, dMax.x);
  best = min(best, quadratic_value(dx, dy, a, b, c));
  dy = dMax.y;
  dx = clamp(-b * dy / a, dMin.x, dMax.x);
  return min(best, quadratic_value(dx, dy, a, b, c));
}

fn tile_intersects_footprint(g: u32, tx: u32, ty: u32) -> bool {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let t = transform[g];
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
  let useEwa = cfg(26u) > 0.5;
  let sampleScale = select(effective, baseScale, useEwa);
  let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
  let padding = select(
    vec2<f32>(0.0),
    vec2<f32>(
      select(0.0, sampleOffset / f32(width - 1u), width > 1u),
      select(0.0, sampleOffset / f32(height - 1u), height > 1u)
    ),
    useEwa
  );
  let minPixel = vec2<u32>(tx * ${TILE_SIZE}u, ty * ${TILE_SIZE}u);
  let maxPixel = min(vec2<u32>((tx + 1u) * ${TILE_SIZE}u - 1u, (ty + 1u) * ${TILE_SIZE}u - 1u), vec2<u32>(width - 1u, height - 1u));
  let rectMin = max(vec2<f32>(-1.0), vec2<f32>(normalized_coordinate(minPixel.x, width), normalized_coordinate(minPixel.y, height)) - padding);
  let rectMax = min(vec2<f32>(1.0), vec2<f32>(normalized_coordinate(maxPixel.x, width), normalized_coordinate(maxPixel.y, height)) + padding);
  let cTheta = cos(t.z);
  let sTheta = sin(t.z);
  let invScale2 = 1.0 / (sampleScale * sampleScale);
  let a = cTheta * cTheta * invScale2.x + sTheta * sTheta * invScale2.y;
  let b = cTheta * sTheta * (invScale2.x - invScale2.y);
  let c = sTheta * sTheta * invScale2.x + cTheta * cTheta * invScale2.y;
  let center = xy[g];
  let minimumQ = minimum_quadratic_on_rect(rectMin - center, rectMax - center, a, b, c);
  return minimumQ <= ${RENDER_SIGMA * RENDER_SIGMA + 0.0001};
}
`
      : "";
    const exactTileIntersectionGuard = exactTileIntersectionEnabled
      ? "if (!virtual_tilt_enabled() && !tile_intersects_footprint(g, tx, ty)) { continue; }"
      : "";
    const shader = `
struct Config { values: array<vec4<f32>, 16>, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tileCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> tileOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> tileCursors: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> tileIndices: array<u32>;
@group(0) @binding(7) var<storage, read_write> control: array<atomic<u32>>;
${geometryDeclaration}

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

${VIRTUAL_TILT_WGSL}

${geometryBuildFunction}

fn tile_bounds(g: u32) -> vec4<u32> {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let center = xy[g];
  let t = transform[g];
  let c = cos(t.z);
  let s = sin(t.z);
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  let useEwa = cfg(26u) > 0.5;
  var effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
  var pixelPadding = vec2<f32>(0.0);
  if (useEwa) {
    effective = baseScale;
    pixelPadding = vec2<f32>(
      select(0.0, 0.5 / f32(u32(cfg(0u)) - 1u), u32(cfg(0u)) > 1u),
      select(0.0, 0.5 / f32(u32(cfg(1u)) - 1u), u32(cfg(1u)) > 1u)
    );
  }
  let radius = vec2<f32>(
    ${RENDER_SIGMA} * (abs(c) * effective.x + abs(s) * effective.y),
    ${RENDER_SIGMA} * (abs(s) * effective.x + abs(c) * effective.y)
  ) + pixelPadding;
  var minNorm = center - radius;
  var maxNorm = center + radius;
  if (virtual_tilt_enabled()) {
    let p0 = virtual_project_point(center + vec2<f32>(-radius.x, -radius.y), 0.0).xy;
    let p1 = virtual_project_point(center + vec2<f32>( radius.x, -radius.y), 0.0).xy;
    let p2 = virtual_project_point(center + vec2<f32>(-radius.x,  radius.y), 0.0).xy;
    let p3 = virtual_project_point(center + vec2<f32>( radius.x,  radius.y), 0.0).xy;
    minNorm = min(min(p0, p1), min(p2, p3));
    maxNorm = max(max(p0, p1), max(p2, p3));
  }
  minNorm = max(vec2<f32>(-1.0), minNorm);
  maxNorm = min(vec2<f32>(1.0), maxNorm);
  let minPx = vec2<u32>(floor((minNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
  let maxPx = vec2<u32>(ceil((maxNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
  return vec4<u32>(minPx / ${TILE_SIZE}u, maxPx / ${TILE_SIZE}u);
}

${exactTileIntersectionFunction}

@compute @workgroup_size(64)
fn count_tiles(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  if (g >= u32(cfg(2u)) || transform[g].w < 0.5) { return; }
  let tileCols = (u32(cfg(0u)) + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  ${countBoundsStatement}
  for (var ty = bounds.y; ty <= bounds.w; ty = ty + 1u) {
    for (var tx = bounds.x; tx <= bounds.z; tx = tx + 1u) {
      ${exactTileIntersectionGuard}
      atomicAdd(&tileCounts[ty * tileCols + tx], 1u);
    }
  }
}

@compute @workgroup_size(1)
fn prefix_tiles(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x != 0u) { return; }
  let tileCount = arrayLength(&tileCounts);
  let capacity = min(arrayLength(&tileIndices), 0x7fffffffu);
  var acceptedTotal = 0u;
  var requiredTotal = 0u;
  for (var tile = 0u; tile < tileCount; tile = tile + 1u) {
    tileOffsets[tile] = acceptedTotal;
    let count = atomicLoad(&tileCounts[tile]);
    if (count > 0xffffffffu - requiredTotal) {
      requiredTotal = 0xffffffffu;
    } else {
      requiredTotal += count;
    }
    acceptedTotal += min(count, capacity - acceptedTotal);
  }
  let overflow = requiredTotal > capacity;
  tileOffsets[tileCount] = acceptedTotal | select(0u, 0x80000000u, overflow);
  atomicStore(&control[0], requiredTotal);
  var overflowAmount = 0u;
  if (overflow) { overflowAmount = requiredTotal - capacity; }
  atomicStore(&control[1], overflowAmount);
}

@compute @workgroup_size(64)
fn fill_tiles(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  if (g >= u32(cfg(2u)) || !(${fillActiveExpression})) { return; }
  let tileCols = (u32(cfg(0u)) + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  ${fillBoundsStatement}
  let capacity = arrayLength(&tileIndices);
  for (var ty = bounds.y; ty <= bounds.w; ty = ty + 1u) {
    for (var tx = bounds.x; tx <= bounds.z; tx = tx + 1u) {
      ${exactTileIntersectionGuard}
      let tile = ty * tileCols + tx;
      let slot = atomicAdd(&tileCursors[tile], 1u);
      let outIndex = tileOffsets[tile] + slot;
      if (outIndex < capacity) {
        tileIndices[outIndex] = g;
      }
    }
  }
}

fn sift_down(base: u32, count: u32, initialRoot: u32) {
  var root = initialRoot;
  loop {
    let left = root * 2u + 1u;
    if (left >= count) { break; }
    var largest = root;
    if (tile_less(tileIndices[base + largest], tileIndices[base + left])) { largest = left; }
    let right = left + 1u;
    if (right < count && tile_less(tileIndices[base + largest], tileIndices[base + right])) { largest = right; }
    if (largest == root) { break; }
    let value = tileIndices[base + root];
    tileIndices[base + root] = tileIndices[base + largest];
    tileIndices[base + largest] = value;
    root = largest;
  }
}

fn tile_less(a: u32, b: u32) -> bool {
  let depthA = virtual_camera_depth(xy[a], transform[a].w);
  let depthB = virtual_camera_depth(xy[b], transform[b].w);
  if (abs(depthA - depthB) > 0.0000001) {
    // Heap-sort ascending by inverse depth, so the final list is front-to-back.
    return depthA > depthB;
  }
  return a < b;
}

@compute @workgroup_size(1)
fn sort_tiles(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let tile = id.x + id.y * workgroups.x;
  let tileCount = arrayLength(&tileOffsets) - 1u;
  if (tile >= tileCount) { return; }
  let base = min(tileOffsets[tile] & 0x7fffffffu, arrayLength(&tileIndices));
  let tileEnd = min(tileOffsets[tile + 1u] & 0x7fffffffu, arrayLength(&tileIndices));
  let count = tileEnd - base;
  if (count < 2u) { return; }
  var start = count / 2u;
  loop {
    if (start == 0u) { break; }
    start -= 1u;
    sift_down(base, count, start);
  }
  var end = count;
  loop {
    if (end <= 1u) { break; }
    end -= 1u;
    let value = tileIndices[base];
    tileIndices[base] = tileIndices[base + end];
    tileIndices[base + end] = value;
    sift_down(base, end, 0u);
  }
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    [this.tileCountPipeline, this.tilePrefixPipeline, this.tileFillPipeline, this.tileSortPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "count_tiles" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "prefix_tiles" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "fill_tiles" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "sort_tiles" } }),
    ]);
  }

  async prepareTileLists(
    image,
    params,
    { sync = false, encoder = null, profileSample = null, geometryPrecomputed = false, writeConfig = true } = {},
  ) {
    if (!this.trainState || this.trainState.capacity < params.count) return false;
    await this.ensureTilePipelines();
    const config = new Float32Array(64);
    config.set([image.width, image.height, params.count, params.bg[0], params.bg[1], params.bg[2]], 0);
    config[17] = currentMaxAnisotropy();
    config[18] = experimentalDensifySteps(state.metrics?.steps_requested || 1);
    config[19] = els.tileCullingToggle.checked ? 1 : 0;
    config[26] = phase33Variants().ewa2x2 ? 1 : 0;
    config[47] = geometryPrecomputed && this.geometryPrecomputeEnabled() ? 1 : 0;
    if (writeConfig) this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    const front = this.trainState.front;
    const common = [
      { binding: 0, resource: { buffer: this.trainState.configBuffer } },
      { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
      { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
    ];
    const geometryEntry = this.geometryPrecomputeEnabled()
      ? [{ binding: 8, resource: { buffer: this.trainState.geometryBuffer } }]
      : [];
    const countBindGroup = this.device.createBindGroup({
      layout: this.tileCountPipeline.getBindGroupLayout(0),
      entries: [...common, { binding: 3, resource: { buffer: this.trainState.tileCountsBuffer } }, ...geometryEntry],
    });
    const prefixBindGroup = this.device.createBindGroup({
      layout: this.tilePrefixPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 3, resource: { buffer: this.trainState.tileCountsBuffer } },
        { binding: 4, resource: { buffer: this.trainState.tileOffsetsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
        { binding: 7, resource: { buffer: this.trainState.tileControlBuffer } },
      ],
    });
    const fillBindGroup = this.device.createBindGroup({
      layout: this.tileFillPipeline.getBindGroupLayout(0),
      entries: [
        ...common,
        { binding: 4, resource: { buffer: this.trainState.tileOffsetsBuffer } },
        { binding: 5, resource: { buffer: this.trainState.tileCursorsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
        ...geometryEntry,
      ],
    });
    const sortBindGroup = this.device.createBindGroup({
      layout: this.tileSortPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
        { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
        { binding: 4, resource: { buffer: this.trainState.tileOffsetsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
      ],
    });
    const commandEncoder = encoder || this.device.createCommandEncoder();
    commandEncoder.clearBuffer(this.trainState.tileCountsBuffer);
    commandEncoder.clearBuffer(this.trainState.tileCursorsBuffer);
    commandEncoder.clearBuffer(this.trainState.tileControlBuffer, 0, 8);
    const countPass = commandEncoder.beginComputePass(this.profilePassDescriptor(profileSample, "tile_count"));
    countPass.setPipeline(this.tileCountPipeline);
    countPass.setBindGroup(0, countBindGroup);
    countPass.dispatchWorkgroups(Math.ceil(params.count / 64));
    countPass.end();
    const prefixPass = commandEncoder.beginComputePass(this.profilePassDescriptor(profileSample, "tile_prefix"));
    prefixPass.setPipeline(this.tilePrefixPipeline);
    prefixPass.setBindGroup(0, prefixBindGroup);
    prefixPass.dispatchWorkgroups(1);
    prefixPass.end();
    const fillPass = commandEncoder.beginComputePass(this.profilePassDescriptor(profileSample, "tile_fill"));
    fillPass.setPipeline(this.tileFillPipeline);
    fillPass.setBindGroup(0, fillBindGroup);
    fillPass.dispatchWorkgroups(Math.ceil(params.count / 64));
    fillPass.end();
    const sortPass = commandEncoder.beginComputePass(this.profilePassDescriptor(profileSample, "tile_sort"));
    sortPass.setPipeline(this.tileSortPipeline);
    sortPass.setBindGroup(0, sortBindGroup);
    this.dispatchLinear(sortPass, this.trainState.tileCount);
    sortPass.end();
    if (!encoder) {
      this.device.queue.submit([commandEncoder.finish()]);
      if (sync) await this.device.queue.onSubmittedWorkDone();
    }
    this.trainState.tileReady = true;
    this.trainState.tileBuilds = (this.trainState.tileBuilds || 0) + 1;
    return true;
  }

  async readTileCounters() {
    if (!this.trainState?.tileReady) return null;
    const readBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.tileControlBuffer, 0, readBuffer, 0, 16);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();
      const total = values[0];
      const overflow = values[1];
      return {
        total,
        overflow,
        noop_steps: values[2],
        capacity: this.trainState.tileIndexCapacity,
        reserve_ratio: total / Math.max(1, this.trainState.tileIndexCapacity),
        tile_count: this.trainState.tileCount,
        average_candidates: total / Math.max(1, this.trainState.tileCount),
        active_count: this.trainState.count,
        free_count: Math.max(0, this.trainState.capacity - this.trainState.count),
      };
    } finally {
      readBuffer.destroy();
    }
  }

  async clearTileNoopCounter() {
    if (!this.trainState?.tileControlBuffer) return;
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.trainState.tileControlBuffer, 8, 8);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  async hashTrainParameters(params) {
    if (!this.trainState || this.trainState.count !== params.count) return null;
    const front = this.trainState.front;
    const xyBytes = params.count * 2 * 4;
    const transformBytes = params.count * 4 * 4;
    const colorBytes = params.count * 4 * 4;
    const totalBytes = xyBytes + transformBytes + colorBytes;
    const readBuffer = this.device.createBuffer({
      size: Math.max(4, totalBytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.xyBuffers[front], 0, readBuffer, 0, xyBytes);
      encoder.copyBufferToBuffer(this.trainState.transformBuffers[front], 0, readBuffer, xyBytes, transformBytes);
      encoder.copyBufferToBuffer(this.trainState.colorBuffers[front], 0, readBuffer, xyBytes + transformBytes, colorBytes);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ, 0, totalBytes);
      const bytes = new Uint8Array(readBuffer.getMappedRange(0, totalBytes));
      let hash = 0x811c9dc5;
      for (let index = 0; index < bytes.length; index += 1) {
        hash = Math.imul(hash ^ bytes[index], 0x01000193) >>> 0;
      }
      readBuffer.unmap();
      return hash.toString(16).padStart(8, "0");
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
      readBuffer.destroy();
    }
  }

  async growTileIndexCapacity(requiredTotal, { proactive = false } = {}) {
    if (!this.trainState) return false;
    if (!proactive && requiredTotal <= this.trainState.tileIndexCapacity) return true;
    const limit = tileIndexCapacityLimit(this.device);
    const requested = Math.ceil(
      Math.max(
        requiredTotal * TILE_INDEX_GROWTH_HEADROOM,
        this.trainState.tileIndexCapacity * TILE_INDEX_INITIAL_HEADROOM,
      ) / 256,
    ) * 256;
    const nextCapacity = Math.min(requested, limit);
    if (nextCapacity < requiredTotal || nextCapacity <= this.trainState.tileIndexCapacity) return false;
    const previous = this.trainState.tileIndicesBuffer;
    const growthPlan = tileGrowthMemoryPlan({
      currentReservedBytes: this.trainingMemorySnapshot().reservedBytes,
      currentTileBytes: previous?.size,
      nextTileBytes: nextCapacity * 4,
    });
    if (!growthPlan.withinBudget) {
      if (state.metrics) state.metrics.tile_growth_budget_rejection = growthPlan;
      return false;
    }
    let nextBuffer = null;
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    try {
      nextBuffer = this.device.createBuffer({
        size: nextCapacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const encoder = this.device.createCommandEncoder();
      encoder.clearBuffer(nextBuffer);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const validationError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      if (validationError || oomError) {
        nextBuffer.destroy();
        return false;
      }
      this.trainState.tileIndicesBuffer = nextBuffer;
      this.trainState.tileIndexCapacity = nextCapacity;
      this.trainState.tileReady = false;
      previous.destroy();
      updateGpuMemoryStatus();
      return true;
    } catch (error) {
      nextBuffer?.destroy();
      await this.device.popErrorScope().catch(() => null);
      await this.device.popErrorScope().catch(() => null);
      return false;
    }
  }

  currentTrainBuffers(params) {
    if (!this.trainState || this.trainState.count !== params.count) return null;
    const front = this.trainState.front;
    return {
      xyBuffer: this.trainState.xyBuffers[front],
      transformBuffer: this.trainState.transformBuffers[front],
      colorBuffer: this.trainState.colorBuffers[front],
      targetBuffer: this.trainState.targetBuffer,
      errorMapBuffer: this.trainState.errorMapBuffer,
      statsBuffer: this.trainState.statsBuffer,
      tileOffsetsBuffer: this.trainState.tileOffsetsBuffer,
      tileIndicesBuffer: this.trainState.tileIndicesBuffer,
      pixelStateBuffer: this.trainState.pixelStateBuffer,
      ssimTileBuffer: this.trainState.ssimTileBuffer,
    };
  }

  async render(image, params, sourceBuffers = null, targetView = null, options = {}) {
    this.ensurePipeline(params.count);
    const preview = previewPaddingSpec(image, params, options.outside ?? els.outsidePreviewToggle.checked);
    state.previewPadding = preview;
    this.canvas.width = preview.width;
    this.canvas.height = preview.height;
    const padded = preview.x > 0 || preview.y > 0;
    const alphaBackground = Array.isArray(options.alphaBackground)
      ? options.alphaBackground
      : [params.bg[0], params.bg[1], params.bg[2]];
    const useTileOrder = !padded && els.tileCullingToggle.checked && Boolean(sourceBuffers?.tileOffsetsBuffer);
    const useGlobalOrder = !useTileOrder && Boolean(params.layerOrderEnabled);
    const uniform = new Float32Array([
      preview.width,
      preview.height,
      params.count,
      params.bg[0],
      params.bg[1],
      params.bg[2],
      useTileOrder ? 1 : useGlobalOrder ? 2 : 0,
      phase33Variants().ewa2x2 ? 1 : 0,
      image.width,
      image.height,
      preview.scaleX,
      preview.scaleY,
      0,
      alphaBackground[0] || 0,
      alphaBackground[1] || 0,
      alphaBackground[2] || 0,
    ]);
    const uniformBuffer = makeBuffer(this.device, uniform, GPUBufferUsage.UNIFORM);
    const buffers = [uniformBuffer];
    let xyBuffer = sourceBuffers?.xyBuffer;
    let transformBuffer = sourceBuffers?.transformBuffer;
    let colorBuffer = sourceBuffers?.colorBuffer;
    let tileOffsetsBuffer = sourceBuffers?.tileOffsetsBuffer;
    let tileIndicesBuffer = sourceBuffers?.tileIndicesBuffer;
    if (!xyBuffer || !transformBuffer || !colorBuffer) {
      const color = packColors(params);
      const transform = packTransforms(params);
      xyBuffer = makeBuffer(this.device, params.xy, GPUBufferUsage.STORAGE);
      transformBuffer = makeBuffer(this.device, transform, GPUBufferUsage.STORAGE);
      colorBuffer = makeBuffer(this.device, color, GPUBufferUsage.STORAGE);
      buffers.push(xyBuffer, transformBuffer, colorBuffer);
    }
    if (useGlobalOrder) {
      const ordered = new Uint32Array(params.count);
      for (let i = 0; i < params.count; i += 1) ordered[i] = i;
      ordered.sort((a, b) => {
        const delta = (params.depthOrder?.[b] ?? 0) - (params.depthOrder?.[a] ?? 0);
        return Math.abs(delta) > 1e-7 ? delta : a - b;
      });
      tileOffsetsBuffer = makeBuffer(this.device, new Uint32Array([0, params.count]), GPUBufferUsage.STORAGE);
      tileIndicesBuffer = makeBuffer(this.device, ordered, GPUBufferUsage.STORAGE);
      buffers.push(tileOffsetsBuffer, tileIndicesBuffer);
    } else if (!tileOffsetsBuffer || !tileIndicesBuffer) {
      tileOffsetsBuffer = makeBuffer(this.device, new Uint32Array([0, params.count]), GPUBufferUsage.STORAGE);
      tileIndicesBuffer = makeBuffer(this.device, new Uint32Array([0]), GPUBufferUsage.STORAGE);
      buffers.push(tileOffsetsBuffer, tileIndicesBuffer);
    }
    try {
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: xyBuffer } },
          { binding: 2, resource: { buffer: transformBuffer } },
          { binding: 3, resource: { buffer: colorBuffer } },
          { binding: 4, resource: { buffer: tileOffsetsBuffer } },
          { binding: 5, resource: { buffer: tileIndicesBuffer } },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: targetView || this.context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 1, g: 1, b: 1, a: 1 },
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      pass.draw(4);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    } finally {
      destroyBuffers(...buffers);
    }
  }

  async captureFrameRgba(image, params, sourceBuffers = null) {
    const preview = previewPaddingSpec(image, params, false);
    const texture = this.device.createTexture({
      size: [preview.width, preview.height, 1],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const bytesPerPixel = 4;
    const bytesPerRow = Math.ceil((preview.width * bytesPerPixel) / 256) * 256;
    const readBuffer = this.device.createBuffer({
      size: bytesPerRow * preview.height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      await this.render(image, params, sourceBuffers, texture.createView(), { outside: false });
      const encoder = this.device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture },
        { buffer: readBuffer, bytesPerRow, rowsPerImage: preview.height },
        { width: preview.width, height: preview.height, depthOrArrayLayers: 1 },
      );
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const source = new Uint8Array(readBuffer.getMappedRange());
      const rgba = new Uint8ClampedArray(image.width * image.height * 4);
      const bgra = this.format.startsWith("bgra");
      for (let y = 0; y < image.height; y += 1) {
        const sourceRow = (preview.y + y) * bytesPerRow + preview.x * bytesPerPixel;
        const targetRow = y * image.width * bytesPerPixel;
        for (let x = 0; x < image.width; x += 1) {
          const si = sourceRow + x * bytesPerPixel;
          const di = targetRow + x * bytesPerPixel;
          rgba[di] = source[si + (bgra ? 2 : 0)];
          rgba[di + 1] = source[si + 1];
          rgba[di + 2] = source[si + (bgra ? 0 : 2)];
          rgba[di + 3] = source[si + 3];
        }
      }
      return rgba;
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
      readBuffer.destroy();
      texture.destroy();
    }
  }

  async capturePresentedStateRgba() {
    if (!this.trainState?.pixelStateBuffer) return null;
    const resolution = this.trainState.pixelStateResolution || [this.trainState.width, this.trainState.height];
    const width = Math.max(1, Math.round(resolution[0]));
    const height = Math.max(1, Math.round(resolution[1]));
    const bytes = width * height * 4 * 4;
    const readBuffer = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.pixelStateBuffer, 0, readBuffer, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const pixels = new Float32Array(readBuffer.getMappedRange());
      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let source = 0, target = 0; source < pixels.length; source += 4, target += 4) {
        rgba[target] = clampByte(pixels[source] * 255);
        rgba[target + 1] = clampByte(pixels[source + 1] * 255);
        rgba[target + 2] = clampByte(pixels[source + 2] * 255);
        rgba[target + 3] = clampByte(pixels[source + 3] * 255);
      }
      readBuffer.unmap();
      return { rgba, width, height, kind: this.trainState.pixelStateKind || "full" };
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
      readBuffer.destroy();
    }
  }

  async readLayerTelemetryGeometry(count) {
    if (!this.trainState || count <= 0) return null;
    const xyBytes = count * 2 * 4;
    const transformBytes = count * 4 * 4;
    const readBuffer = this.device.createBuffer({
      size: xyBytes + transformBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const front = this.trainState.front;
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.xyBuffers[front], 0, readBuffer, 0, xyBytes);
      encoder.copyBufferToBuffer(this.trainState.transformBuffers[front], 0, readBuffer, xyBytes, transformBytes);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const mapped = readBuffer.getMappedRange();
      const xy = new Float32Array(mapped, 0, count * 2).slice();
      const transform = new Float32Array(mapped, xyBytes, count * 4).slice();
      readBuffer.unmap();
      return { xy, transform };
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
      readBuffer.destroy();
    }
  }

  ensurePresentPipeline() {
    if (this.presentPipeline) return;
    const shader = `
struct PresentConfig {
  width: u32,
  height: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<uniform> config: PresentConfig;
@group(0) @binding(1) var<storage, read> pixelState: array<vec4<f32>>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var out: VertexOut;
  out.position = vec4<f32>(positions[index], 0.0, 1.0);
  return out;
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let x = min(config.width - 1u, u32(position.x));
  let y = min(config.height - 1u, u32(position.y));
  return vec4<f32>(pixelState[y * config.width + x].rgb, 1.0);
}`;
    const module = this.device.createShaderModule({ code: shader });
    this.presentPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
  }

  presentTrainState(image) {
    if (!this.trainState?.pixelStateBuffer || !this.trainState.presentConfigBuffer) return false;
    this.ensurePresentPipeline();
    const resolution = this.trainState.pixelStateResolution || [image.width, image.height];
    const width = Math.max(1, Math.round(resolution[0]));
    const height = Math.max(1, Math.round(resolution[1]));
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    document.documentElement.dataset.previewBufferWidth = String(width);
    document.documentElement.dataset.previewBufferHeight = String(height);
    document.documentElement.dataset.previewBufferKind = this.trainState.pixelStateKind || "full";
    this.device.queue.writeBuffer(this.trainState.presentConfigBuffer, 0, new Uint32Array([width, height, 0, 0]));
    const bindGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.presentConfigBuffer } },
        { binding: 1, resource: { buffer: this.trainState.pixelStateBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 1, g: 1, b: 1, a: 1 },
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.presentPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  async computeMetrics(image, params, sourceBuffers = null) {
    if (!sourceBuffers?.pixelStateBuffer) {
      throw new Error("Experimental metrics require active WebGPU render state");
    }
    return this.computeTrainStateMetrics(image, params);
  }

  async ensureRenderGradientPipelines() {
    if (
      this.renderStatePipeline &&
      this.tileCooperativeRenderPipeline &&
      this.ssimTilePipeline &&
      this.renderGradientPipeline &&
      this.parallelRenderGradientPipeline &&
      this.lossGradientPipeline &&
      this.exactAlphaBackwardPipeline &&
      this.virtualOrderPenaltyPipeline &&
      this.exactBackwardTelemetryPipeline &&
      this.exactOptimizerPipeline
    ) return;
    const geometryEnabled = this.geometryPrecomputeEnabled();
    const geometryStruct = `struct SplatGeometry {
  centerTrig: vec4<f32>,
  baseEffective: vec4<f32>,
  sampleMip: vec4<f32>,
  pixelBounds: vec4<u32>,
  tileBounds: vec4<u32>,
};`;
    const renderGeometryDeclaration = geometryEnabled
      ? `${geometryStruct}\n@group(0) @binding(9) var<storage, read> geometry: array<SplatGeometry>;`
      : "";
    const optimizerGeometryDeclaration = geometryEnabled
      ? `${geometryStruct}\n@group(0) @binding(8) var<storage, read> geometry: array<SplatGeometry>;`
      : "";
    const optimizerAdamExtraFields = geometryEnabled
      ? "stats: vec4<f32>,\n  importance: vec4<f32>,"
      : "";
    const optimizerStatsDeclaration = geometryEnabled
      ? ""
      : "@group(0) @binding(8) var<storage, read_write> stats: array<vec4<f32>>;";
    const optimizerStatsUpdate = geometryEnabled
      ? `let previous = opt.stats;
  let meanError = errorSum / max(observed, 1.0);
  let signedGradientSignal = length(gradCenter) * normalizer;
  let absoluteGradientSignal = length(gradCenterAbs) * normalizer;
  let densityGradientMode = u32(cfg(29u));
  let gradientSignal = select(signedGradientSignal, absoluteGradientSignal, densityGradientMode == 1u);
  let meanContribution = influenceSum / max(observed, 1.0);
  opt.stats = vec4<f32>(previous.x + gradientSignal, max(previous.y, meanError), max(previous.z, meanContribution), previous.w + 1.0);
  let previousImportance = opt.importance;
  let importanceAlpha = clamp(cfg(27u), 0.001, 1.0);
  let measuredImportance = vec3<f32>(observed, influenceSum, errorSum);
  let coherenceNorm = gradCenterNorm * normalizer;
  let importanceW = select(previousImportance.w + 1.0, previousImportance.w + coherenceNorm, densityGradientMode == 2u);
  opt.importance = vec4<f32>(mix(previousImportance.xyz, measuredImportance, importanceAlpha), importanceW);
  adam[g] = opt;`
      : `adam[g] = opt;
  let previous = stats[g];
  let meanError = errorSum / max(observed, 1.0);
  let signedGradientSignal = length(gradCenter) * normalizer;
  let absoluteGradientSignal = length(gradCenterAbs) * normalizer;
  let densityGradientMode = u32(cfg(29u));
  let gradientSignal = select(signedGradientSignal, absoluteGradientSignal, densityGradientMode == 1u);
  let meanContribution = influenceSum / max(observed, 1.0);
  stats[g] = vec4<f32>(previous.x + gradientSignal, max(previous.y, meanError), max(previous.z, meanContribution), previous.w + 1.0);
  let previousImportance = stats[capacity + g];
  let importanceAlpha = clamp(cfg(27u), 0.001, 1.0);
  let measuredImportance = vec3<f32>(observed, influenceSum, errorSum);
  let coherenceNorm = gradCenterNorm * normalizer;
  let importanceW = select(previousImportance.w + 1.0, previousImportance.w + coherenceNorm, densityGradientMode == 2u);
  stats[capacity + g] = vec4<f32>(mix(previousImportance.xyz, measuredImportance, importanceAlpha), importanceW);`;
    const renderGeometryOverride = geometryEnabled
      ? `if (cfg(47u) > 0.5 && !virtual_tilt_enabled()) {
        let geom = geometry[g];
        center = geom.centerTrig.xy;
        c = geom.centerTrig.z;
        s = geom.centerTrig.w;
        baseScale = geom.baseEffective.xy;
        effective = geom.baseEffective.zw;
        sampleScale = geom.sampleMip.xy;
        mip = geom.sampleMip.z;
      }`
      : "";
    const optimizerGeometryOverride = geometryEnabled
      ? `if (cfg(47u) > 0.5) {
    let geom = geometry[g];
    center = geom.centerTrig.xy;
    c = geom.centerTrig.z;
    s = geom.centerTrig.w;
    baseScale = geom.baseEffective.xy;
    effective = geom.baseEffective.zw;
    sampleScale = geom.sampleMip.xy;
    mip = geom.sampleMip.z;
    minPx = geom.pixelBounds.xy;
    maxPx = geom.pixelBounds.zw;
    isActive = geom.sampleMip.w > 0.5;
  }`
      : "";
    const renderShader = `
struct Config { values: array<vec4<f32>, 16>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(6) var<storage, read> tileIndices: array<u32>;
@group(0) @binding(7) var<storage, read_write> pixelState: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> alphaState: array<AlphaState>;
${renderGeometryDeclaration}

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

${VIRTUAL_TILT_WGSL}

fn gaussian_kernel(d: vec2<f32>, c: f32, s: f32, scale: vec2<f32>) -> f32 {
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  let q = dot(r / scale, r / scale);
  return exp(-0.5 * q);
}

fn tile_offset(index: u32) -> u32 {
  return tileOffsets[index] & 0x7fffffffu;
}

fn tile_list_overflow() -> bool {
  let finalOffset = tileOffsets[arrayLength(&tileOffsets) - 1u];
  return (finalOffset & 0x80000000u) != 0u;
}

var<workgroup> tileShared0: array<vec4<f32>, ${TILE_SIZE * TILE_SIZE}>;
var<workgroup> tileShared1: array<vec4<f32>, ${TILE_SIZE * TILE_SIZE}>;
var<workgroup> tileSharedColor: array<vec4<f32>, ${TILE_SIZE * TILE_SIZE}>;

@compute @workgroup_size(64)
fn render_state(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let pixel = id.x + id.y * workgroups.x * 64u;
  if (pixel >= width * height) { return; }
  let px = pixel % width;
  let py = pixel / width;
  let outputPoint = vec2<f32>(
    select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
    select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
  );
  let bg = vec3<f32>(cfg(3u), cfg(4u), cfg(5u));
  if (cfg(19u) > 0.5 && tile_list_overflow()) {
    pixelState[pixel] = vec4<f32>(bg, -1.0);
    alphaState[pixel] = AlphaState(-1.0, 0u);
    return;
  }
  let useTiles = cfg(19u) > 0.5;
  let tileCols = (width + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  let tile = (py / ${TILE_SIZE}u) * tileCols + (px / ${TILE_SIZE}u);
  let tileCapacity = arrayLength(&tileIndices);
  let start = select(0u, min(tile_offset(tile), tileCapacity), useTiles);
  let end = select(u32(cfg(2u)), min(tile_offset(tile + 1u), tileCapacity), useTiles);
  let inversePoint = virtual_inverse_point(outputPoint);
  if (inversePoint.z < 0.5) {
    pixelState[pixel] = vec4<f32>(bg, 0.0);
    alphaState[pixel] = AlphaState(0.0, start);
    return;
  }
  let p = inversePoint.xy;
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  var rendered = vec3<f32>(0.0);
  var transmittance = 1.0;
  var cursor = start;
  var acceptedEnd = start;
  loop {
    if (cursor >= end) { break; }
    var g = cursor;
    if (useTiles) { g = tileIndices[cursor]; }
    let t = transform[g];
    if (t.w >= 0.5) {
      var center = xy[g];
      var c = cos(t.z);
      var s = sin(t.z);
      var baseScale = max(t.xy, vec2<f32>(0.0001));
      var effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
      var sampleScale = effective;
      var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
      let useEwa = cfg(26u) > 0.5;
      if (useEwa) {
        sampleScale = baseScale;
        mip = 1.0;
      }
      ${renderGeometryOverride}
      let d = p - center;
      var kernel = gaussian_kernel(d, c, s, sampleScale);
      if (useEwa) {
        let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
        let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
        let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
        let q00 = virtual_inverse_point(clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;
        let q10 = virtual_inverse_point(clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;
        let q01 = virtual_inverse_point(clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;
        let q11 = virtual_inverse_point(clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;
        kernel = 0.25 * (
          gaussian_kernel(q00 - center, c, s, baseScale) +
          gaussian_kernel(q10 - center, c, s, baseScale) +
          gaussian_kernel(q01 - center, c, s, baseScale) +
          gaussian_kernel(q11 - center, c, s, baseScale)
        );
      }
      if (kernel >= 0.0003354626) {
        let alpha = clamp(kernel * color[g].a * mip, 0.0, 0.99);
        if (alpha >= 0.0039215686) {
          rendered += transmittance * alpha * color[g].rgb;
          transmittance *= 1.0 - alpha;
          acceptedEnd = cursor + 1u;
        }
      }
    }
    cursor += 1u;
    if (transmittance < 0.0001) { break; }
  }
  rendered += transmittance * bg;
  let compositeAlpha = 1.0 - transmittance;
  pixelState[pixel] = vec4<f32>(rendered, compositeAlpha);
  alphaState[pixel] = AlphaState(1.0 - transmittance, acceptedEnd);
}

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE}, 1)
fn render_state_tile(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let tileCols = (width + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  let tile = wid.y * tileCols + wid.x;
  let tileCapacity = arrayLength(&tileIndices);
  let px = wid.x * ${TILE_SIZE}u + lid.x;
  let py = wid.y * ${TILE_SIZE}u + lid.y;
  let screenInside = px < width && py < height;
  let bg = vec3<f32>(cfg(3u), cfg(4u), cfg(5u));
  if (cfg(19u) > 0.5 && tile_list_overflow()) {
    if (screenInside) {
      let pixel = py * width + px;
      pixelState[pixel] = vec4<f32>(bg, -1.0);
      alphaState[pixel] = AlphaState(-1.0, 0u);
    }
    return;
  }
  let start = min(tile_offset(tile), tileCapacity);
  let end = min(tile_offset(tile + 1u), tileCapacity);
  var p = vec2<f32>(0.0);
  var outputPoint = vec2<f32>(0.0);
  var inside = false;
  if (screenInside) {
    outputPoint = vec2<f32>(
      select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
      select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
    );
    let inversePoint = virtual_inverse_point(outputPoint);
    p = inversePoint.xy;
    inside = inversePoint.z > 0.5;
  }
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  let useEwa = cfg(26u) > 0.5;
  let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
  let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
  let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
  var rendered = vec3<f32>(0.0);
  var transmittance = 1.0;
  var batchStart = start;
  var acceptedEnd = start;
  loop {
    if (batchStart >= end) { break; }
    let batchCount = min(${TILE_SIZE * TILE_SIZE}u, end - batchStart);
    if (localIndex < batchCount) {
      let g = tileIndices[batchStart + localIndex];
      let t = transform[g];
      var center = xy[g];
      var c = cos(t.z);
      var s = sin(t.z);
      var baseScale = max(t.xy, vec2<f32>(0.0001));
      var effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
      var sampleScale = effective;
      var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
      let isActive = t.w >= 0.5;
      if (useEwa) {
        sampleScale = baseScale;
        mip = 1.0;
      }
      ${renderGeometryOverride}
      tileShared0[localIndex] = vec4<f32>(center, c, s);
      tileShared1[localIndex] = vec4<f32>(sampleScale, mip, select(0.0, 1.0, isActive));
      tileSharedColor[localIndex] = color[g];
    }
    workgroupBarrier();
    if (inside) {
      for (var j = 0u; j < batchCount; j += 1u) {
        if (tileShared1[j].w > 0.5) {
          let center = tileShared0[j].xy;
          let c = tileShared0[j].z;
          let s = tileShared0[j].w;
          let sampleScale = tileShared1[j].xy;
          var kernel = gaussian_kernel(p - center, c, s, sampleScale);
          if (useEwa) {
            let q00 = virtual_inverse_point(clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;
            let q10 = virtual_inverse_point(clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;
            let q01 = virtual_inverse_point(clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;
            let q11 = virtual_inverse_point(clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;
            kernel = 0.25 * (
              gaussian_kernel(q00 - center, c, s, sampleScale) +
              gaussian_kernel(q10 - center, c, s, sampleScale) +
              gaussian_kernel(q01 - center, c, s, sampleScale) +
              gaussian_kernel(q11 - center, c, s, sampleScale)
            );
          }
          if (kernel >= 0.0003354626) {
            let rgba = tileSharedColor[j];
            let alpha = clamp(kernel * rgba.a * tileShared1[j].z, 0.0, 0.99);
            if (alpha >= 0.0039215686 && transmittance >= 0.0001) {
              rendered += transmittance * alpha * rgba.rgb;
              transmittance *= 1.0 - alpha;
              acceptedEnd = batchStart + j + 1u;
            }
          }
        }
      }
    }
    workgroupBarrier();
    batchStart += batchCount;
  }
  if (screenInside) {
    let pixel = py * width + px;
    rendered = select(bg, rendered + transmittance * bg, inside);
    let compositeAlpha = 1.0 - transmittance;
    let storedAlpha = select(0.0, compositeAlpha, inside);
    pixelState[pixel] = vec4<f32>(rendered, storedAlpha);
    alphaState[pixel] = AlphaState(storedAlpha, select(start, acceptedEnd, inside));
  }
}`;

    const ssimShader = `
struct Config { values: array<vec4<f32>, 16>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ssimTiles: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(5) var<storage, read> alphaState: array<AlphaState>;
var<workgroup> sx: array<f32, 64>;
var<workgroup> sy: array<f32, 64>;
var<workgroup> sx2: array<f32, 64>;
var<workgroup> sy2: array<f32, 64>;
var<workgroup> sxy: array<f32, 64>;
var<workgroup> sc: array<f32, 64>;
var<workgroup> sax: array<f32, 64>;
var<workgroup> say: array<f32, 64>;
var<workgroup> sax2: array<f32, 64>;
var<workgroup> say2: array<f32, 64>;
var<workgroup> saxy: array<f32, 64>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

${VIRTUAL_TILT_WGSL}

fn target_rgb_at(point: vec2<f32>, width: u32, height: u32) -> vec3<f32> {
  let source = clamp((point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)), vec2<f32>(0.0), vec2<f32>(f32(width - 1u), f32(height - 1u)));
  let p0 = vec2<u32>(floor(source));
  let p1 = min(p0 + vec2<u32>(1u), vec2<u32>(width - 1u, height - 1u));
  let f = fract(source);
  let i00 = (p0.y * width + p0.x) * 3u;
  let i10 = (p0.y * width + p1.x) * 3u;
  let i01 = (p1.y * width + p0.x) * 3u;
  let i11 = (p1.y * width + p1.x) * 3u;
  let c00 = vec3<f32>(targetRgb[i00], targetRgb[i00 + 1u], targetRgb[i00 + 2u]);
  let c10 = vec3<f32>(targetRgb[i10], targetRgb[i10 + 1u], targetRgb[i10 + 2u]);
  let c01 = vec3<f32>(targetRgb[i01], targetRgb[i01 + 1u], targetRgb[i01 + 2u]);
  let c11 = vec3<f32>(targetRgb[i11], targetRgb[i11 + 1u], targetRgb[i11 + 2u]);
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn target_alpha_at(point: vec2<f32>, width: u32, height: u32) -> f32 {
  let source = clamp((point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)), vec2<f32>(0.0), vec2<f32>(f32(width - 1u), f32(height - 1u)));
  let p0 = vec2<u32>(floor(source));
  let p1 = min(p0 + vec2<u32>(1u), vec2<u32>(width - 1u, height - 1u));
  let f = fract(source);
  let a00 = targetAlpha[p0.y * width + p0.x];
  let a10 = targetAlpha[p0.y * width + p1.x];
  let a01 = targetAlpha[p1.y * width + p0.x];
  let a11 = targetAlpha[p1.y * width + p1.x];
  return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
}

@compute @workgroup_size(64)
fn ssim_tiles(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let tileCols = (width + 7u) / 8u;
  let tileIndex = wid.y * workgroups.x + wid.x;
  let tileCount = tileCols * ((height + 7u) / 8u);
  if (tileIndex >= tileCount) { return; }
  let tileX = tileIndex % tileCols;
  let tileY = tileIndex / tileCols;
  let px = tileX * 8u + lid.x % 8u;
  let py = tileY * 8u + lid.x / 8u;
  var x = 0.0;
  var y = 0.0;
  var ax = 0.0;
  var ay = 0.0;
  var valid = 0.0;
  if (px < width && py < height) {
    let pixel = py * width + px;
    let outputPoint = vec2<f32>(select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u));
    let inversePoint = virtual_inverse_point(outputPoint);
    if (inversePoint.z > 0.5) {
      x = dot(pixelState[pixel].rgb, vec3<f32>(1.0 / 3.0));
      y = dot(target_rgb_at(inversePoint.xy, width, height), vec3<f32>(1.0 / 3.0));
      ax = alphaState[pixel].compositeAlpha;
      ay = target_alpha_at(inversePoint.xy, width, height);
      valid = 1.0;
    }
  }
  sx[lid.x] = x;
  sy[lid.x] = y;
  sx2[lid.x] = x * x;
  sy2[lid.x] = y * y;
  sxy[lid.x] = x * y;
  sc[lid.x] = valid;
  sax[lid.x] = ax;
  say[lid.x] = ay;
  sax2[lid.x] = ax * ax;
  say2[lid.x] = ay * ay;
  saxy[lid.x] = ax * ay;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      sx[lid.x] += sx[lid.x + stride];
      sy[lid.x] += sy[lid.x + stride];
      sx2[lid.x] += sx2[lid.x + stride];
      sy2[lid.x] += sy2[lid.x + stride];
      sxy[lid.x] += sxy[lid.x + stride];
      sc[lid.x] += sc[lid.x + stride];
      sax[lid.x] += sax[lid.x + stride];
      say[lid.x] += say[lid.x + stride];
      sax2[lid.x] += sax2[lid.x + stride];
      say2[lid.x] += say2[lid.x + stride];
      saxy[lid.x] += saxy[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let count = max(sc[0], 1.0);
    let mux = sx[0] / count;
    let muy = sy[0] / count;
    let vx = max(0.0, sx2[0] / count - mux * mux);
    let vy = max(0.0, sy2[0] / count - muy * muy);
    let cov = sxy[0] / count - mux * muy;
    let c1 = 0.0001;
    let c2 = 0.0009;
    let ssim = ((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / max(0.00000001, (mux * mux + muy * muy + c1) * (vx + vy + c2));
    let amux = sax[0] / count;
    let amuy = say[0] / count;
    let avx = max(0.0, sax2[0] / count - amux * amux);
    let avy = max(0.0, say2[0] / count - amuy * amuy);
    let acov = saxy[0] / count - amux * amuy;
    let alphaSsim = ((2.0 * amux * amuy + c1) * (2.0 * acov + c2)) / max(0.00000001, (amux * amux + amuy * amuy + c1) * (avx + avy + c2));
    let base = tileIndex * 4u;
    ssimTiles[base] = vec4<f32>(mux, muy, vx, vy);
    ssimTiles[base + 1u] = vec4<f32>(cov, ssim, count, 0.0);
    ssimTiles[base + 2u] = vec4<f32>(amux, amuy, avx, avy);
    ssimTiles[base + 3u] = vec4<f32>(acov, alphaSsim, count, 0.0);
  }
}`;

    const lossGradientShader = [
      "struct Config { values: array<vec4<f32>, 16>, };",
      "@group(0) @binding(0) var<uniform> config: Config;",
      "@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;",
      "@group(0) @binding(2) var<storage, read> targetAlpha: array<f32>;",
      "@group(0) @binding(3) var<storage, read> pixelState: array<vec4<f32>>;",
      "@group(0) @binding(4) var<storage, read> ssimTiles: array<vec4<f32>>;",
      "@group(0) @binding(5) var<storage, read_write> lossGradient: array<vec4<f32>>;",
      "fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }",
      VIRTUAL_TILT_WGSL,
      "fn target_color_at(point: vec2<f32>, width: u32, height: u32) -> vec3<f32> {",
      "  let source = clamp((point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)), vec2<f32>(0.0), vec2<f32>(f32(width - 1u), f32(height - 1u)));",
      "  let p0 = vec2<u32>(floor(source));",
      "  let p1 = min(p0 + vec2<u32>(1u), vec2<u32>(width - 1u, height - 1u));",
      "  let f = fract(source);",
      "  let i00 = (p0.y * width + p0.x) * 3u;",
      "  let i10 = (p0.y * width + p1.x) * 3u;",
      "  let i01 = (p1.y * width + p0.x) * 3u;",
      "  let i11 = (p1.y * width + p1.x) * 3u;",
      "  let c00 = vec3<f32>(targetRgb[i00], targetRgb[i00 + 1u], targetRgb[i00 + 2u]);",
      "  let c10 = vec3<f32>(targetRgb[i10], targetRgb[i10 + 1u], targetRgb[i10 + 2u]);",
      "  let c01 = vec3<f32>(targetRgb[i01], targetRgb[i01 + 1u], targetRgb[i01 + 2u]);",
      "  let c11 = vec3<f32>(targetRgb[i11], targetRgb[i11 + 1u], targetRgb[i11 + 2u]);",
      "  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);",
      "}",
      "fn target_alpha_at(point: vec2<f32>, width: u32, height: u32) -> f32 {",
      "  let source = clamp((point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)), vec2<f32>(0.0), vec2<f32>(f32(width - 1u), f32(height - 1u)));",
      "  let p0 = vec2<u32>(floor(source));",
      "  let p1 = min(p0 + vec2<u32>(1u), vec2<u32>(width - 1u, height - 1u));",
      "  let f = fract(source);",
      "  return mix(mix(targetAlpha[p0.y * width + p0.x], targetAlpha[p0.y * width + p1.x], f.x), mix(targetAlpha[p1.y * width + p0.x], targetAlpha[p1.y * width + p1.x], f.x), f.y);",
      "}",
      "fn safe_signed(v: f32) -> f32 {",
      "  if (abs(v) >= 0.0000001) { return v; }",
      "  return select(-0.0000001, 0.0000001, v >= 0.0);",
      "}",
      "fn rendered_luma(px: u32, py: u32, width: u32) -> f32 {",
      "  return dot(pixelState[py * width + px].rgb, vec3<f32>(1.0 / 3.0));",
      "}",
      "fn target_luma(px: u32, py: u32, width: u32) -> f32 {",
      "  let height = u32(cfg(1u));",
      "  let outputPoint = vec2<f32>(select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u));",
      "  let inversePoint = virtual_inverse_point(outputPoint);",
      "  return dot(target_color_at(inversePoint.xy, width, height), vec3<f32>(1.0 / 3.0));",
      "}",
      "@compute @workgroup_size(64)",
      "fn loss_gradient(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) {",
      "  let width = u32(cfg(0u));",
      "  let height = u32(cfg(1u));",
      "  let pixel = id.x + id.y * workgroups.x * 64u;",
      "  if (pixel >= width * height) { return; }",
      "  let px = pixel % width;",
      "  let py = pixel / width;",
      "  let outputPoint = vec2<f32>(select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u));",
      "  let inversePoint = virtual_inverse_point(outputPoint);",
      "  if (inversePoint.z < 0.5) {",
      "    lossGradient[pixel * 3u] = vec4<f32>(0.0);",
      "    lossGradient[pixel * 3u + 1u] = vec4<f32>(0.0);",
      "    lossGradient[pixel * 3u + 2u] = vec4<f32>(0.0);",
      "    return;",
      "  }",
      "  let renderedState = pixelState[pixel];",
      "  let targetColor = target_color_at(inversePoint.xy, width, height);",
      "  let residual = renderedState.rgb - targetColor;",
      "  var dColor = sign(residual) * ((1.0 - 0.2) / 3.0);",
      "  let ssimTileCols = (width + 7u) / 8u;",
      "  let tile = (py / 8u) * ssimTileCols + (px / 8u);",
      "  let moments = ssimTiles[tile * 4u];",
      "  let extra = ssimTiles[tile * 4u + 1u];",
      "  let mux = moments.x;",
      "  let muy = moments.y;",
      "  let vx = moments.z;",
      "  let vy = moments.w;",
      "  let cov = extra.x;",
      "  let ssim = extra.y;",
      "  let n = max(extra.z, 1.0);",
      "  let x = dot(renderedState.rgb, vec3<f32>(1.0 / 3.0));",
      "  let y = dot(targetColor, vec3<f32>(1.0 / 3.0));",
      "  let a = safe_signed(2.0 * mux * muy + 0.0001);",
      "  let b = safe_signed(2.0 * cov + 0.0009);",
      "  let cc = safe_signed(mux * mux + muy * muy + 0.0001);",
      "  let dd = safe_signed(vx + vy + 0.0009);",
      "  let dSsim = ssim * ((2.0 * muy / n) / a + (2.0 * (y - muy) / n) / b - (2.0 * mux / n) / cc - (2.0 * (x - mux) / n) / dd);",
      "  dColor += vec3<f32>(-0.5 * 0.2 * dSsim / 3.0);",
      "  if (cfg(15u) > 0.5) {",
      "    var gradientDerivative = 0.0;",
      "    var gradientTerms = 0.0;",
      "    if (px > 0u) {",
      "      gradientDerivative += sign((x - rendered_luma(px - 1u, py, width)) - (y - target_luma(px - 1u, py, width)));",
      "      gradientTerms += 1.0;",
      "    }",
      "    if (px + 1u < width) {",
      "      gradientDerivative += sign((x - rendered_luma(px + 1u, py, width)) - (y - target_luma(px + 1u, py, width)));",
      "      gradientTerms += 1.0;",
      "    }",
      "    if (py > 0u) {",
      "      gradientDerivative += sign((x - rendered_luma(px, py - 1u, width)) - (y - target_luma(px, py - 1u, width)));",
      "      gradientTerms += 1.0;",
      "    }",
      "    if (py + 1u < height) {",
      "      gradientDerivative += sign((x - rendered_luma(px, py + 1u, width)) - (y - target_luma(px, py + 1u, width)));",
      "      gradientTerms += 1.0;",
      "    }",
      "    let frequencyRamp = clamp((cfg(8u) / max(cfg(9u), 1.0) - 0.2) / 0.3, 0.0, 1.0);",
      "    dColor += vec3<f32>(cfg(16u) * frequencyRamp * gradientDerivative / max(1.0, gradientTerms) / 3.0);",
      "  }",
      "  let alphaMoments = ssimTiles[tile * 4u + 2u];",
      "  let alphaExtra = ssimTiles[tile * 4u + 3u];",
      "  let alphaMux = alphaMoments.x;",
      "  let alphaMuy = alphaMoments.y;",
      "  let alphaVx = alphaMoments.z;",
      "  let alphaVy = alphaMoments.w;",
      "  let alphaCov = alphaExtra.x;",
      "  let alphaSsim = alphaExtra.y;",
      "  let alphaN = max(alphaExtra.z, 1.0);",
      "  let alphaX = renderedState.a;",
      "  let alphaY = target_alpha_at(inversePoint.xy, width, height);",
      "  let alphaA = safe_signed(2.0 * alphaMux * alphaMuy + 0.0001);",
      "  let alphaB = safe_signed(2.0 * alphaCov + 0.0009);",
      "  let alphaC = safe_signed(alphaMux * alphaMux + alphaMuy * alphaMuy + 0.0001);",
      "  let alphaD = safe_signed(alphaVx + alphaVy + 0.0009);",
      "  let dAlphaSsim = alphaSsim * ((2.0 * alphaMuy / alphaN) / alphaA + (2.0 * (alphaY - alphaMuy) / alphaN) / alphaB - (2.0 * alphaMux / alphaN) / alphaC - (2.0 * (alphaX - alphaMux) / alphaN) / alphaD);",
      "  var dAlpha = cfg(46u) * ((1.0 - 0.2) * sign(alphaX - alphaY) - 0.5 * 0.2 * dAlphaSsim);",
      "  var dTransmittanceExtra = 0.0;",
      "  if (cfg(48u) > 0.5 && cfg(49u) > 0.0) {",
      "    let background = vec3<f32>(cfg(3u), cfg(4u), cfg(5u));",
      "    let alternateBackground = vec3<f32>(1.0);",
      "    let backgroundDelta = alternateBackground - background;",
      "    let renderedAlternate = renderedState.rgb + (1.0 - renderedState.a) * backgroundDelta;",
      "    let targetAlternate = targetColor * alphaY + (1.0 - alphaY) * alternateBackground;",
      "    let alternateGradient = cfg(49u) * sign(renderedAlternate - targetAlternate) * ((1.0 - 0.2) / 3.0);",
      "    dColor += alternateGradient;",
      "    dTransmittanceExtra = dot(alternateGradient, backgroundDelta);",
      "  }",
      "  let residualMagnitude = (abs(residual.r) + abs(residual.g) + abs(residual.b)) / 3.0;",
      "  if (cfg(21u) > 0.5 && residualMagnitude > 0.02) {",
      "    dAlpha += -2.0 * cfg(23u) * max(0.0, cfg(22u) - renderedState.a);",
      "  }",
      "  let viewWeight = select(1.0, clamp(cfg(61u), 0.0, 1.0), virtual_tilt_enabled());",
      "  dColor *= viewWeight;",
      "  dAlpha *= viewWeight;",
      "  dTransmittanceExtra *= viewWeight;",
      "  lossGradient[pixel * 3u] = vec4<f32>(dColor, dAlpha);",
      "  lossGradient[pixel * 3u + 1u] = vec4<f32>(targetColor, residualMagnitude);",
      "  lossGradient[pixel * 3u + 2u] = vec4<f32>(dTransmittanceExtra, 0.0, 0.0, 0.0);",
      "}",
    ].join("\n");

    const exactBackwardShader = [
      "struct Config { values: array<vec4<f32>, 16>, };",
      "struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, };",
      "struct KernelSample { kernel: f32, dCenter: vec2<f32>, dLogScale: vec2<f32>, dTheta: f32, };",
      "struct KernelEvaluation { rawKernel: f32, weightedKernel: f32, dCenter: vec2<f32>, dLogScale: vec2<f32>, dTheta: f32, };",
      "@group(0) @binding(0) var<uniform> config: Config;",
      "@group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;",
      "@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;",
      "@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;",
      "@group(0) @binding(4) var<storage, read> tileOffsets: array<u32>;",
      "@group(0) @binding(5) var<storage, read> tileIndices: array<u32>;",
      "@group(0) @binding(6) var<storage, read> lossGradient: array<vec4<f32>>;",
      "@group(0) @binding(7) var<storage, read_write> exactGradient: array<atomic<u32>>;",
      "@group(0) @binding(8) var<storage, read> alphaState: array<AlphaState>;",
      "fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }",
      VIRTUAL_TILT_WGSL,
      "fn tile_offset(index: u32) -> u32 { return tileOffsets[index] & 0x7fffffffu; }",
      "fn tile_list_overflow() -> bool { return cfg(19u) > 0.5 && (tileOffsets[arrayLength(&tileOffsets) - 1u] & 0x80000000u) != 0u; }",
      "fn kernel_sample(d: vec2<f32>, c: f32, s: f32, baseScale: vec2<f32>, sampleScale: vec2<f32>, includeMipGradient: bool) -> KernelSample {",
      "  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);",
      "  let invS2 = 1.0 / (sampleScale * sampleScale);",
      "  let q = dot(r * r, invS2);",
      "  if (q > 16.0) { return KernelSample(0.0, vec2<f32>(0.0), vec2<f32>(0.0), 0.0); }",
      "  let kernel = exp(-0.5 * q);",
      "  let dCenter = vec2<f32>(r.x * c * invS2.x - r.y * s * invS2.y, r.x * s * invS2.x + r.y * c * invS2.y);",
      "  var dLogScale = vec2<f32>(r.x * r.x * baseScale.x * baseScale.x / pow(sampleScale.x, 4.0), r.y * r.y * baseScale.y * baseScale.y / pow(sampleScale.y, 4.0));",
      "  if (includeMipGradient) {",
      "    let ratio = (baseScale * baseScale) / (sampleScale * sampleScale);",
      "    dLogScale += 0.5 * (vec2<f32>(1.0) - ratio);",
      "  }",
      "  let dTheta = -r.x * r.y * (invS2.x - invS2.y);",
      "  return KernelSample(kernel, dCenter, dLogScale, dTheta);",
      "}",
      "fn evaluate_kernel(p: vec2<f32>, outputPoint: vec2<f32>, center: vec2<f32>, t: vec4<f32>, width: u32, height: u32) -> KernelEvaluation {",
      "  let baseScale = max(t.xy, vec2<f32>(0.0001));",
      "  let c = cos(t.z);",
      "  let s = sin(t.z);",
      "  let pixelSigma = 0.35 * 2.0 / max(cfg(0u), cfg(1u));",
      "  let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));",
      "  let useEwa = cfg(26u) > 0.5;",
      "  if (!useEwa) {",
      "    let sample = kernel_sample(p - center, c, s, baseScale, effective, true);",
      "    let mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));",
      "    return KernelEvaluation(sample.kernel, sample.kernel * mip, sample.kernel * mip * sample.dCenter, sample.kernel * mip * sample.dLogScale, sample.kernel * mip * sample.dTheta);",
      "  }",
      "  let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);",
      "  let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);",
      "  let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);",
      "  let q0 = virtual_inverse_point(clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;",
      "  let q1 = virtual_inverse_point(clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;",
      "  let q2 = virtual_inverse_point(clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;",
      "  let q3 = virtual_inverse_point(clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0))).xy;",
      "  let sample0 = kernel_sample(q0 - center, c, s, baseScale, baseScale, false);",
      "  let sample1 = kernel_sample(q1 - center, c, s, baseScale, baseScale, false);",
      "  let sample2 = kernel_sample(q2 - center, c, s, baseScale, baseScale, false);",
      "  let sample3 = kernel_sample(q3 - center, c, s, baseScale, baseScale, false);",
      "  let rawKernel = 0.25 * (sample0.kernel + sample1.kernel + sample2.kernel + sample3.kernel);",
      "  let dCenter = 0.25 * (sample0.kernel * sample0.dCenter + sample1.kernel * sample1.dCenter + sample2.kernel * sample2.dCenter + sample3.kernel * sample3.dCenter);",
      "  let dLogScale = 0.25 * (sample0.kernel * sample0.dLogScale + sample1.kernel * sample1.dLogScale + sample2.kernel * sample2.dLogScale + sample3.kernel * sample3.dLogScale);",
      "  let dTheta = 0.25 * (sample0.kernel * sample0.dTheta + sample1.kernel * sample1.dTheta + sample2.kernel * sample2.dTheta + sample3.kernel * sample3.dTheta);",
      "  return KernelEvaluation(rawKernel, rawKernel, dCenter, dLogScale, dTheta);",
      "}",
      "fn atomic_add_f32(index: u32, value: f32) {",
      "  if (abs(value) < 0.00000000000000000001) { return; }",
      "  var oldBits = atomicLoad(&exactGradient[index]);",
      "  loop {",
      "    let oldValue = bitcast<f32>(oldBits);",
      "    let nextBits = bitcast<u32>(oldValue + value);",
      "    let exchanged = atomicCompareExchangeWeak(&exactGradient[index], oldBits, nextBits);",
      "    if (exchanged.exchanged) { break; }",
      "    oldBits = exchanged.old_value;",
      "  }",
      "}",
      "fn add_gradient(g: u32, geom: vec4<f32>, appearance: vec4<f32>, misc: vec4<f32>, density: vec4<f32>) {",
      "  let base = g * 16u;",
      "  atomic_add_f32(base, geom.x);",
      "  atomic_add_f32(base + 1u, geom.y);",
      "  atomic_add_f32(base + 2u, geom.z);",
      "  atomic_add_f32(base + 3u, geom.w);",
      "  atomic_add_f32(base + 4u, appearance.x);",
      "  atomic_add_f32(base + 5u, appearance.y);",
      "  atomic_add_f32(base + 6u, appearance.z);",
      "  atomic_add_f32(base + 7u, appearance.w);",
      "  atomic_add_f32(base + 8u, misc.x);",
      "  atomic_add_f32(base + 9u, misc.y);",
      "  atomic_add_f32(base + 10u, misc.z);",
      "  atomic_add_f32(base + 11u, misc.w);",
      "  atomic_add_f32(base + 12u, density.x);",
      "  atomic_add_f32(base + 13u, density.y);",
      "  atomic_add_f32(base + 14u, density.z);",
      "}",
      "var<workgroup> reduceGeom: array<vec4<f32>, 64>;",
      "var<workgroup> reduceAppearance: array<vec4<f32>, 64>;",
      "var<workgroup> reduceMisc: array<vec4<f32>, 64>;",
      "var<workgroup> reduceDensity: array<vec4<f32>, 64>;",
      "fn add_subtile_gradient(localIndex: u32, g: u32, geom: vec4<f32>, appearance: vec4<f32>, misc: vec4<f32>, density: vec4<f32>) {",
      "  reduceGeom[localIndex] = geom;",
      "  reduceAppearance[localIndex] = appearance;",
      "  reduceMisc[localIndex] = misc;",
      "  reduceDensity[localIndex] = density;",
      "  workgroupBarrier();",
      "  var stride = 32u;",
      "  loop {",
      "    if (localIndex < stride) {",
      "      reduceGeom[localIndex] += reduceGeom[localIndex + stride];",
      "      reduceAppearance[localIndex] += reduceAppearance[localIndex + stride];",
      "      reduceMisc[localIndex] += reduceMisc[localIndex + stride];",
      "      reduceDensity[localIndex] += reduceDensity[localIndex + stride];",
      "    }",
      "    workgroupBarrier();",
      "    if (stride == 1u) { break; }",
      "    stride /= 2u;",
      "  }",
      "  if (localIndex == 0u) {",
      "    add_gradient(g, reduceGeom[0], reduceAppearance[0], reduceMisc[0], reduceDensity[0]);",
      "  }",
      "  workgroupBarrier();",
      "}",
      "@compute @workgroup_size(8, 8, 1)",
      "fn exact_alpha_backward(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) localIndex: u32, @builtin(workgroup_id) wid: vec3<u32>) {",
      "  if (tile_list_overflow()) { return; }",
      "  let width = u32(cfg(0u));",
      "  let height = u32(cfg(1u));",
      "  let px = id.x;",
      "  let py = id.y;",
      "  let screenValid = px < width && py < height;",
      "  let pixel = min(px, width - 1u) + min(py, height - 1u) * width;",
      "  let outputPoint = vec2<f32>(select(0.0, f32(min(px, width - 1u)) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(min(py, height - 1u)) / f32(height - 1u) * 2.0 - 1.0, height > 1u));",
      "  let inversePoint = virtual_inverse_point(outputPoint);",
      "  let validPixel = screenValid && inversePoint.z > 0.5;",
      "  let p = inversePoint.xy;",
      "  let useTiles = cfg(19u) > 0.5;",
      "  let tileCols = (width + 15u) / 16u;",
      "  let tile = (wid.y / 2u) * tileCols + (wid.x / 2u);",
      "  let tileCapacity = arrayLength(&tileIndices);",
      "  let start = select(0u, min(tile_offset(tile), tileCapacity), useTiles);",
      "  let end = select(u32(cfg(2u)), min(tile_offset(tile + 1u), tileCapacity), useTiles);",
      "  var acceptedEnd = start;",
      "  var dColor = vec3<f32>(0.0);",
      "  var targetAndError = vec4<f32>(0.0);",
      "  var gradTransmittance = 0.0;",
      "  var transAfter = 1.0;",
      "  if (validPixel) {",
      "    acceptedEnd = min(end, max(start, alphaState[pixel].acceptedEnd));",
      "    let packedLoss = lossGradient[pixel * 3u];",
      "    targetAndError = lossGradient[pixel * 3u + 1u];",
      "    let dualBackgroundGradient = lossGradient[pixel * 3u + 2u].x;",
      "    dColor = packedLoss.rgb;",
      "    gradTransmittance = dot(dColor, vec3<f32>(cfg(3u), cfg(4u), cfg(5u))) + dualBackgroundGradient - packedLoss.a;",
      "    transAfter = clamp(1.0 - alphaState[pixel].compositeAlpha, 0.0, 1.0);",
      "  }",
      "  var reverseCursor = end;",
      "  loop {",
      "    if (reverseCursor <= start) { break; }",
      "    reverseCursor -= 1u;",
      "    let g = select(reverseCursor, tileIndices[reverseCursor], useTiles);",
      "    var geom = vec4<f32>(0.0);",
      "    var appearance = vec4<f32>(0.0);",
      "    var misc = vec4<f32>(0.0);",
      "    var density = vec4<f32>(0.0);",
      "    if (validPixel && reverseCursor < acceptedEnd) {",
      "      let t = transform[g];",
      "      if (t.w >= 0.5) {",
      "        let evaluation = evaluate_kernel(p, outputPoint, xy[g], t, width, height);",
      "        if (evaluation.rawKernel >= 0.0003354626) {",
      "          let rgba = color[g];",
      "          let unclampedAlpha = evaluation.weightedKernel * rgba.a;",
      "          let alpha = clamp(unclampedAlpha, 0.0, 0.99);",
      "          if (alpha >= 0.0039215686) {",
      "            let transBefore = transAfter / max(1.0 - alpha, 0.01);",
      "            let dAlpha = transBefore * dot(dColor, rgba.rgb) - gradTransmittance * transBefore;",
      "            let differentiableAlpha = select(0.0, dAlpha, unclampedAlpha > 0.0 && unclampedAlpha < 0.99);",
      "            let dWeightedKernel = differentiableAlpha * rgba.a;",
      "            let gradCenter = dWeightedKernel * evaluation.dCenter;",
      "            let gradLogScale = dWeightedKernel * evaluation.dLogScale;",
      "            let gradTheta = dWeightedKernel * evaluation.dTheta;",
      "            let influence = transBefore * alpha;",
      "            let anchor = sign(rgba.rgb - targetAndError.rgb) * (cfg(44u) * influence / 3.0);",
      "            let gradColor = dColor * influence + anchor;",
      "            let gradLogit = differentiableAlpha * evaluation.weightedKernel * rgba.a * (1.0 - rgba.a);",
      "            geom = vec4<f32>(gradCenter, gradLogScale);",
      "            appearance = vec4<f32>(gradColor, gradLogit);",
      "            misc = vec4<f32>(gradTheta, influence, targetAndError.a, 1.0);",
      "            density = vec4<f32>(abs(gradCenter), length(gradCenter), 0.0);",
      "            gradTransmittance = dot(dColor, alpha * rgba.rgb) + gradTransmittance * (1.0 - alpha);",
      "            transAfter = transBefore;",
      "          }",
      "        }",
      "      }",
      "    }",
      "    add_subtile_gradient(localIndex, g, geom, appearance, misc, density);",
      "  }",
      "}",
      "@compute @workgroup_size(8, 8, 1)",
      "fn exact_alpha_backward_quad(@builtin(local_invocation_index) localIndex: u32, @builtin(workgroup_id) wid: vec3<u32>) {",
      "  if (tile_list_overflow()) { return; }",
      "  let width = u32(cfg(0u));",
      "  let height = u32(cfg(1u));",
      "  let localX = localIndex % 8u;",
      "  let localY = localIndex / 8u;",
      "  let baseX = wid.x * 16u + localX * 2u;",
      "  let baseY = wid.y * 16u + localY * 2u;",
      "  let offsets = array<vec2<u32>, 4>(vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 1u));",
      "  let useTiles = cfg(19u) > 0.5;",
      "  let tileCols = (width + 15u) / 16u;",
      "  let tile = wid.y * tileCols + wid.x;",
      "  let tileCapacity = arrayLength(&tileIndices);",
      "  let start = select(0u, min(tile_offset(tile), tileCapacity), useTiles);",
      "  let end = select(u32(cfg(2u)), min(tile_offset(tile + 1u), tileCapacity), useTiles);",
      "  var validPixels: array<u32, 4>;",
      "  var pixelIndices: array<u32, 4>;",
      "  var points: array<vec2<f32>, 4>;",
      "  var outputPoints: array<vec2<f32>, 4>;",
      "  var acceptedEnds: array<u32, 4>;",
      "  var dColors: array<vec3<f32>, 4>;",
      "  var targetsAndErrors: array<vec4<f32>, 4>;",
      "  var gradTransmittances: array<f32, 4>;",
      "  var transAfters: array<f32, 4>;",
      "  for (var i = 0u; i < 4u; i += 1u) {",
      "    let px = baseX + offsets[i].x;",
      "    let py = baseY + offsets[i].y;",
      "    let screenValid = px < width && py < height;",
      "    let safeX = min(px, width - 1u);",
      "    let safeY = min(py, height - 1u);",
      "    let pixel = safeX + safeY * width;",
      "    let outputPoint = vec2<f32>(select(0.0, f32(safeX) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(safeY) / f32(height - 1u) * 2.0 - 1.0, height > 1u));",
      "    let inversePoint = virtual_inverse_point(outputPoint);",
      "    let validPixel = screenValid && inversePoint.z > 0.5;",
      "    validPixels[i] = select(0u, 1u, validPixel);",
      "    pixelIndices[i] = pixel;",
      "    points[i] = inversePoint.xy;",
      "    outputPoints[i] = outputPoint;",
      "    acceptedEnds[i] = start;",
      "    dColors[i] = vec3<f32>(0.0);",
      "    targetsAndErrors[i] = vec4<f32>(0.0);",
      "    gradTransmittances[i] = 0.0;",
      "    transAfters[i] = 1.0;",
      "    if (validPixel) {",
      "      acceptedEnds[i] = min(end, max(start, alphaState[pixel].acceptedEnd));",
      "      let packedLoss = lossGradient[pixel * 3u];",
      "      targetsAndErrors[i] = lossGradient[pixel * 3u + 1u];",
      "      let dualBackgroundGradient = lossGradient[pixel * 3u + 2u].x;",
      "      dColors[i] = packedLoss.rgb;",
      "      gradTransmittances[i] = dot(packedLoss.rgb, vec3<f32>(cfg(3u), cfg(4u), cfg(5u))) + dualBackgroundGradient - packedLoss.a;",
      "      transAfters[i] = clamp(1.0 - alphaState[pixel].compositeAlpha, 0.0, 1.0);",
      "    }",
      "  }",
      "  var reverseCursor = end;",
      "  loop {",
      "    if (reverseCursor <= start) { break; }",
      "    reverseCursor -= 1u;",
      "    let g = select(reverseCursor, tileIndices[reverseCursor], useTiles);",
      "    let t = transform[g];",
      "    let rgba = color[g];",
      "    var geom = vec4<f32>(0.0);",
      "    var appearance = vec4<f32>(0.0);",
      "    var misc = vec4<f32>(0.0);",
      "    var density = vec4<f32>(0.0);",
      "    if (t.w >= 0.5) {",
      "      for (var i = 0u; i < 4u; i += 1u) {",
      "        if (validPixels[i] != 0u && reverseCursor < acceptedEnds[i]) {",
      "          let evaluation = evaluate_kernel(points[i], outputPoints[i], xy[g], t, width, height);",
      "          if (evaluation.rawKernel >= 0.0003354626) {",
      "            let unclampedAlpha = evaluation.weightedKernel * rgba.a;",
      "            let alpha = clamp(unclampedAlpha, 0.0, 0.99);",
      "            if (alpha >= 0.0039215686) {",
      "              let transBefore = transAfters[i] / max(1.0 - alpha, 0.01);",
      "              let dAlpha = transBefore * dot(dColors[i], rgba.rgb) - gradTransmittances[i] * transBefore;",
      "              let differentiableAlpha = select(0.0, dAlpha, unclampedAlpha > 0.0 && unclampedAlpha < 0.99);",
      "              let dWeightedKernel = differentiableAlpha * rgba.a;",
      "              let gradCenter = dWeightedKernel * evaluation.dCenter;",
      "              let gradLogScale = dWeightedKernel * evaluation.dLogScale;",
      "              let gradTheta = dWeightedKernel * evaluation.dTheta;",
      "              let influence = transBefore * alpha;",
      "              let anchor = sign(rgba.rgb - targetsAndErrors[i].rgb) * (cfg(44u) * influence / 3.0);",
      "              geom += vec4<f32>(gradCenter, gradLogScale);",
      "              appearance += vec4<f32>(dColors[i] * influence + anchor, differentiableAlpha * evaluation.weightedKernel * rgba.a * (1.0 - rgba.a));",
      "              misc += vec4<f32>(gradTheta, influence, targetsAndErrors[i].a, 1.0);",
      "              density += vec4<f32>(abs(gradCenter), length(gradCenter), 0.0);",
      "              gradTransmittances[i] = dot(dColors[i], alpha * rgba.rgb) + gradTransmittances[i] * (1.0 - alpha);",
      "              transAfters[i] = transBefore;",
      "            }",
      "          }",
      "        }",
      "      }",
      "    }",
      "    add_subtile_gradient(localIndex, g, geom, appearance, misc, density);",
      "  }",
      "}",
    ].join("\n");

    const exactBackwardTelemetryShader = `
struct Config { values: array<vec4<f32>, 14>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> tileIndices: array<u32>;
@group(0) @binding(5) var<storage, read> alphaState: array<AlphaState>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
var<workgroup> acceptedEndMax: array<u32, 64>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

fn tile_offset(index: u32) -> u32 { return tileOffsets[index] & 0x7fffffffu; }
fn tile_list_overflow() -> bool {
  return cfg(19u) > 0.5 && (tileOffsets[arrayLength(&tileOffsets) - 1u] & 0x80000000u) != 0u;
}

fn normalized_pixel(pixel: u32, size: u32) -> f32 {
  return select(0.0, f32(pixel) / f32(size - 1u) * 2.0 - 1.0, size > 1u);
}

fn footprint_intersects_subtile(g: u32, wid: vec3<u32>, width: u32, height: u32) -> bool {
  let t = transform[g];
  if (t.w < 0.5) { return false; }
  let center = xy[g];
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  let sampleScale = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
  let c = cos(t.z);
  let s = sin(t.z);
  let radius = ${RENDER_SIGMA}.0 * vec2<f32>(
    abs(c) * sampleScale.x + abs(s) * sampleScale.y,
    abs(s) * sampleScale.x + abs(c) * sampleScale.y
  );
  let minPx = vec2<u32>(wid.xy * 8u);
  let maxPx = min(minPx + vec2<u32>(7u), vec2<u32>(width - 1u, height - 1u));
  let pixelMargin = vec2<f32>(select(0.0, 1.0 / f32(width - 1u), width > 1u), select(0.0, 1.0 / f32(height - 1u), height > 1u));
  let subtileMin = vec2<f32>(normalized_pixel(minPx.x, width), normalized_pixel(minPx.y, height)) - pixelMargin;
  let subtileMax = vec2<f32>(normalized_pixel(maxPx.x, width), normalized_pixel(maxPx.y, height)) + pixelMargin;
  let footprintMin = center - radius;
  let footprintMax = center + radius;
  return footprintMax.x >= subtileMin.x && footprintMin.x <= subtileMax.x && footprintMax.y >= subtileMin.y && footprintMin.y <= subtileMax.y;
}

@compute @workgroup_size(8, 8, 1)
fn measure_exact_backward(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) localIndex: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  if (tile_list_overflow()) { return; }
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let validPixel = id.x < width && id.y < height;
  let pixel = min(id.x, width - 1u) + min(id.y, height - 1u) * width;
  let useTiles = cfg(19u) > 0.5;
  let tileCols = (width + 15u) / 16u;
  let tile = (wid.y / 2u) * tileCols + (wid.x / 2u);
  let tileCapacity = arrayLength(&tileIndices);
  let start = select(0u, min(tile_offset(tile), tileCapacity), useTiles);
  let end = select(u32(cfg(2u)), min(tile_offset(tile + 1u), tileCapacity), useTiles);
  acceptedEndMax[localIndex] = select(start, min(end, max(start, alphaState[pixel].acceptedEnd)), validPixel);
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (localIndex < stride) { acceptedEndMax[localIndex] = max(acceptedEndMax[localIndex], acceptedEndMax[localIndex + stride]); }
    workgroupBarrier();
  }
  if (localIndex != 0u) { return; }
  let acceptedEnd = acceptedEndMax[0];
  var rejected = 0u;
  for (var cursor = start; cursor < acceptedEnd; cursor += 1u) {
    let g = select(cursor, tileIndices[cursor], useTiles);
    if (!footprint_intersects_subtile(g, wid, width, height)) { rejected += 1u; }
  }
  atomicAdd(&counters[0], end - start);
  atomicAdd(&counters[1], end - acceptedEnd);
  atomicAdd(&counters[2], acceptedEnd - start);
  atomicAdd(&counters[3], rejected);
  atomicAdd(&counters[4], 1u);
}
`;

    const virtualOrderPenaltyShader = `
struct Config { values: array<vec4<f32>, 16>, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> exactGradient: array<atomic<u32>>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

${VIRTUAL_TILT_WGSL}

fn load_f32(index: u32) -> f32 { return bitcast<f32>(atomicLoad(&exactGradient[index])); }

fn atomic_add_f32(index: u32, value: f32) {
  if (abs(value) < 0.00000000000000000001) { return; }
  var oldBits = atomicLoad(&exactGradient[index]);
  loop {
    let oldValue = bitcast<f32>(oldBits);
    let exchanged = atomicCompareExchangeWeak(&exactGradient[index], oldBits, bitcast<u32>(oldValue + value));
    if (exchanged.exchanged) { break; }
    oldBits = exchanged.old_value;
  }
}

@compute @workgroup_size(64)
fn virtual_order_penalty(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let g = id.x + id.y * workgroups.x * 64u;
  if (!virtual_tilt_enabled() || cfg(60u) <= 0.0 || g >= u32(cfg(2u))) { return; }
  let t = transform[g];
  let rgba = color[g];
  if (t.w < 0.5 || rgba.a < 0.007) { return; }
  let longSide = max(cfg(0u), cfg(1u));
  let frame = vec2<f32>(cfg(0u), cfg(1u)) / max(longSide, 1.0);
  let c = cos(t.z);
  let s = sin(t.z);
  let axisX = vec3<f32>(frame.x * c * t.x, -frame.y * s * t.x, 0.0);
  let axisY = vec3<f32>(-frame.x * s * t.y, -frame.y * c * t.y, 0.0);
  let rotation = virtual_tilt_rotation();
  let depthX = dot(rotation.row2, axisX);
  let depthY = dot(rotation.row2, axisY);
  let sigmaDepth = sqrt(depthX * depthX + depthY * depthY + 0.000000000001);
  let supportDepth = ${RENDER_SIGMA}.0 * sigmaDepth;
  let excess = max(0.0, supportDepth / ${PLY_LAYER_DEPTH_SPAN} - 1.0);
  if (excess <= 0.0) { return; }
  let base = g * ${EXACT_GRADIENT_STRIDE}u;
  let influence = max(abs(load_f32(base + 9u)), 0.01);
  let residual = max(0.0, load_f32(base + 10u)) / influence;
  let coefficient = 2.0 * cfg(60u) * residual * excess * influence / ${PLY_LAYER_DEPTH_SPAN};
  let dSupportX = ${RENDER_SIGMA}.0 * depthX * depthX / sigmaDepth;
  let dSupportY = ${RENDER_SIGMA}.0 * depthY * depthY / sigmaDepth;
  atomic_add_f32(base + 2u, coefficient * dSupportX);
  atomic_add_f32(base + 3u, coefficient * dSupportY);
}
`;

    const optimizerShader = `
struct Config { values: array<vec4<f32>, 16>, };
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
  ${optimizerAdamExtraFields}
};
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> xy: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(5) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> ssimTiles: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> adam: array<AdamState>;
@group(0) @binding(9) var<storage, read> exactGradients: array<f32>;
@group(0) @binding(10) var<storage, read_write> tileControl: array<atomic<u32>>;
${optimizerStatsDeclaration}
${optimizerGeometryDeclaration}

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn safe_signed(v: f32) -> f32 {
  if (abs(v) >= 0.0000001) { return v; }
  return select(-0.0000001, 0.0000001, v >= 0.0);
}
fn hash_unit(seed: f32) -> f32 {
  let value = sin(seed * 12.9898) * 43758.5453;
  return value - floor(value);
}

fn rendered_luma(px: u32, py: u32, width: u32) -> f32 {
  return dot(pixelState[py * width + px].rgb, vec3<f32>(1.0 / 3.0));
}

fn target_luma(px: u32, py: u32, width: u32) -> f32 {
  let index = (py * width + px) * 3u;
  return (targetRgb[index] + targetRgb[index + 1u] + targetRgb[index + 2u]) / 3.0;
}

struct KernelSample {
  kernel: f32,
  dCenter: vec2<f32>,
  dLogScale: vec2<f32>,
  dTheta: f32,
};

struct PixelGradient {
  geom: vec4<f32>,
  appearance: vec4<f32>,
  misc: vec4<f32>,
  density: vec4<f32>,
};

var<workgroup> reduceGeom: array<vec4<f32>, 64>;
var<workgroup> reduceAppearance: array<vec4<f32>, 64>;
var<workgroup> reduceMisc: array<vec4<f32>, 64>;
var<workgroup> reduceDensity: array<vec4<f32>, 64>;

fn kernel_sample(
  d: vec2<f32>,
  c: f32,
  s: f32,
  baseScale: vec2<f32>,
  sampleScale: vec2<f32>,
  includeMipGradient: bool,
) -> KernelSample {
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  let invS2 = 1.0 / (sampleScale * sampleScale);
  let q = dot(r * r, invS2);
  if (q > ${RENDER_SIGMA * RENDER_SIGMA}) {
    return KernelSample(0.0, vec2<f32>(0.0), vec2<f32>(0.0), 0.0);
  }
  let kernel = exp(-0.5 * q);
  let dCenter = vec2<f32>(
    r.x * c * invS2.x - r.y * s * invS2.y,
    r.x * s * invS2.x + r.y * c * invS2.y
  );
  var dLogScale = vec2<f32>(
    r.x * r.x * baseScale.x * baseScale.x / pow(sampleScale.x, 4.0),
    r.y * r.y * baseScale.y * baseScale.y / pow(sampleScale.y, 4.0)
  );
  if (includeMipGradient) {
    let k = (baseScale * baseScale) / (sampleScale * sampleScale);
    dLogScale += 0.5 * (vec2<f32>(1.0) - k);
  }
  let dTheta = -r.x * r.y * (invS2.x - invS2.y);
  return KernelSample(kernel, dCenter, dLogScale, dTheta);
}

fn pixel_gradient(
  px: u32,
  py: u32,
  width: u32,
  height: u32,
  center: vec2<f32>,
  rgba: vec4<f32>,
  baseScale: vec2<f32>,
  sampleScale: vec2<f32>,
  mip: f32,
  useEwa: bool,
  c: f32,
  s: f32,
) -> PixelGradient {
  let zero = PixelGradient(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
  let p = vec2<f32>(
    select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
    select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
  );
  let d = p - center;
  var kernelSum = 0.0;
  var centerNumerator = vec2<f32>(0.0);
  var scaleNumerator = vec2<f32>(0.0);
  var thetaNumerator = 0.0;
  if (useEwa) {
    let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
    let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
    let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
    let sample0 = kernel_sample(clamp(p + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale, sampleScale, false);
    let sample1 = kernel_sample(clamp(p + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale, sampleScale, false);
    let sample2 = kernel_sample(clamp(p + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale, sampleScale, false);
    let sample3 = kernel_sample(clamp(p + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale, sampleScale, false);
    kernelSum = sample0.kernel + sample1.kernel + sample2.kernel + sample3.kernel;
    centerNumerator = sample0.kernel * sample0.dCenter + sample1.kernel * sample1.dCenter + sample2.kernel * sample2.dCenter + sample3.kernel * sample3.dCenter;
    scaleNumerator = sample0.kernel * sample0.dLogScale + sample1.kernel * sample1.dLogScale + sample2.kernel * sample2.dLogScale + sample3.kernel * sample3.dLogScale;
    thetaNumerator = sample0.kernel * sample0.dTheta + sample1.kernel * sample1.dTheta + sample2.kernel * sample2.dTheta + sample3.kernel * sample3.dTheta;
  } else {
    let sample = kernel_sample(d, c, s, baseScale, sampleScale, true);
    kernelSum = sample.kernel;
    centerNumerator = sample.kernel * sample.dCenter;
    scaleNumerator = sample.kernel * sample.dLogScale;
    thetaNumerator = sample.kernel * sample.dTheta;
  }
  if (kernelSum <= 0.00000001) { return zero; }
  let quadratureScale = select(1.0, 0.25, useEwa);
  let weight = kernelSum * quadratureScale * rgba.a * mip;
  let pixel = py * width + px;
  let renderedState = pixelState[pixel];
  if (renderedState.a <= 0.00000001) { return zero; }
  let targetIndex = pixel * 3u;
  let targetColor = vec3<f32>(targetRgb[targetIndex], targetRgb[targetIndex + 1u], targetRgb[targetIndex + 2u]);
  let residual = renderedState.rgb - targetColor;
  let residualMagnitude = (abs(residual.r) + abs(residual.g) + abs(residual.b)) / 3.0;
  var dLoss = sign(residual) * ((1.0 - ${DEFAULT_DSSIM_WEIGHT}) / 3.0);
  let ssimTileCols = (width + 7u) / 8u;
  let tile = (py / 8u) * ssimTileCols + (px / 8u);
  let moments = ssimTiles[tile * 4u];
  let extra = ssimTiles[tile * 4u + 1u];
  let mux = moments.x;
  let muy = moments.y;
  let vx = moments.z;
  let vy = moments.w;
  let cov = extra.x;
  let ssim = extra.y;
  let n = max(extra.z, 1.0);
  let x = dot(renderedState.rgb, vec3<f32>(1.0 / 3.0));
  let y = dot(targetColor, vec3<f32>(1.0 / 3.0));
  let a = safe_signed(2.0 * mux * muy + 0.0001);
  let b = safe_signed(2.0 * cov + 0.0009);
  let cc = safe_signed(mux * mux + muy * muy + 0.0001);
  let dd = safe_signed(vx + vy + 0.0009);
  let dSsim = ssim * ((2.0 * muy / n) / a + (2.0 * (y - muy) / n) / b - (2.0 * mux / n) / cc - (2.0 * (x - mux) / n) / dd);
  dLoss += vec3<f32>(-0.5 * ${DEFAULT_DSSIM_WEIGHT} * dSsim / 3.0);
  if (cfg(15u) > 0.5) {
    var gradientDerivative = 0.0;
    var gradientTerms = 0.0;
    if (px > 0u) {
      gradientDerivative += sign((x - rendered_luma(px - 1u, py, width)) - (y - target_luma(px - 1u, py, width)));
      gradientTerms += 1.0;
    }
    if (px + 1u < width) {
      gradientDerivative += sign((x - rendered_luma(px + 1u, py, width)) - (y - target_luma(px + 1u, py, width)));
      gradientTerms += 1.0;
    }
    if (py > 0u) {
      gradientDerivative += sign((x - rendered_luma(px, py - 1u, width)) - (y - target_luma(px, py - 1u, width)));
      gradientTerms += 1.0;
    }
    if (py + 1u < height) {
      gradientDerivative += sign((x - rendered_luma(px, py + 1u, width)) - (y - target_luma(px, py + 1u, width)));
      gradientTerms += 1.0;
    }
    let frequencyRamp = clamp((cfg(8u) / max(cfg(9u), 1.0) - 0.2) / 0.3, 0.0, 1.0);
    dLoss += vec3<f32>(cfg(16u) * frequencyRamp * gradientDerivative / max(1.0, gradientTerms) / 3.0);
  }
  let alpha = clamp(weight, 0.0, 0.99);
  let otherTransmittance = clamp((1.0 - renderedState.a) / max(0.01, 1.0 - alpha), 0.0001, 1.0);
  let influence = alpha * sqrt(otherTransmittance);
  let anchorGradient = sign(rgba.rgb - targetColor) * (cfg(44u) * influence / 3.0);
  var weightSignal = dot(dLoss, rgba.rgb - renderedState.rgb) * influence;
  if (cfg(21u) > 0.5 && residualMagnitude > 0.02) {
    let coverageDeficit = max(0.0, cfg(22u) - renderedState.a);
    weightSignal += -2.0 * cfg(23u) * coverageDeficit * weight;
  }
  let alphaRegularizer = -cfg(46u) * (1.0 - renderedState.a) * alpha * (1.0 - rgba.a) / max(0.01, 1.0 - alpha);
  let centerPixelGrad = weightSignal * centerNumerator / kernelSum;
  return PixelGradient(
    vec4<f32>(centerPixelGrad, weightSignal * scaleNumerator / kernelSum),
    vec4<f32>(dLoss * influence + anchorGradient, weightSignal * (1.0 - rgba.a) + alphaRegularizer),
    vec4<f32>(weightSignal * thetaNumerator / kernelSum, influence, residualMagnitude, 1.0),
    vec4<f32>(abs(centerPixelGrad), length(centerPixelGrad), 0.0)
  );
}

fn apply_optimizer(
  g: u32,
  center: vec2<f32>,
  t: vec4<f32>,
  rgba: vec4<f32>,
  baseScale: vec2<f32>,
  c: f32,
  s: f32,
  geomSum: vec4<f32>,
  appearanceSum: vec4<f32>,
  miscSum: vec4<f32>,
  densitySum: vec4<f32>,
) {
  let capacity = u32(cfg(28u));
  let gradCenter = geomSum.xy;
  let gradLogScale = geomSum.zw;
  let gradColor = appearanceSum.xyz;
  let gradLogit = appearanceSum.w;
  let gradTheta = miscSum.x;
  let influenceSum = miscSum.y;
  let errorSum = miscSum.z;
  let observed = miscSum.w;
  let gradCenterAbs = densitySum.xy;
  let gradCenterNorm = densitySum.z;
  let normalizer = 1.0 / max(influenceSum, 0.01);
  let gradGeom = vec4<f32>(gradCenter, gradLogScale) * normalizer;
  let gradAppearance = vec4<f32>(gradColor, gradLogit) * normalizer;
  let gradRotation = vec4<f32>(gradTheta * normalizer, 0.0, 0.0, 0.0);
  let beta1 = 0.9;
  let beta2 = 0.999;
  var opt = adam[g];
  let adcResetStep = opt.mTheta.w;
  opt.mGeom = beta1 * opt.mGeom + (1.0 - beta1) * gradGeom;
  opt.vGeom = beta2 * opt.vGeom + (1.0 - beta2) * gradGeom * gradGeom;
  opt.mColor = beta1 * opt.mColor + (1.0 - beta1) * gradAppearance;
  opt.vColor = beta2 * opt.vColor + (1.0 - beta2) * gradAppearance * gradAppearance;
  opt.mTheta = beta1 * opt.mTheta + (1.0 - beta1) * gradRotation;
  opt.vTheta = beta2 * opt.vTheta + (1.0 - beta2) * gradRotation * gradRotation;
  opt.mTheta.w = adcResetStep;
  let step = max(cfg(8u), 1.0);
  let useRowAge = cfg(30u) > 0.5 && adcResetStep > 0.0 && step >= adcResetStep;
  let optimizerAge = select(step, max(1.0, step - adcResetStep + 1.0), useRowAge);
  let bias1 = max(0.000001, 1.0 - pow(beta1, optimizerAge));
  let bias2 = max(0.000001, 1.0 - pow(beta2, optimizerAge));
  let geomAdam = (opt.mGeom / bias1) / (sqrt(opt.vGeom / bias2) + vec4<f32>(0.00000001));
  let colorAdam = (opt.mColor / bias1) / (sqrt(opt.vColor / bias2) + vec4<f32>(0.00000001));
  let thetaAdam = (opt.mTheta / bias1) / (sqrt(opt.vTheta / bias2) + vec4<f32>(0.00000001));
  let horizon = max(cfg(9u), 1.0);
  let afterDensity = max(0.0, step - horizon);
  let progress = min(1.0, step / horizon);
  let densityAnneal = max(0.05, 1.0 - progress);
  var adcRecovery = 0.0;
  if (adcResetStep > 0.0 && step > adcResetStep) {
    adcRecovery = exp(-(step - adcResetStep) / ${ADC_RECOVERY_DECAY_STEPS}.0);
  }
  let settleAnneal = max(exp(-afterDensity / 100.0), adcRecovery);
  let colorSettleAnneal = max(exp(-afterDensity / max(cfg(33u), 1.0)), adcRecovery);
  let spatialGate = densityAnneal * settleAnneal;
  let colorGate = densityAnneal * colorSettleAnneal;
  let lrScale = cfg(6u);
  let positionLr = cfg(10u) * lrScale * spatialGate;
  let colorLr = cfg(11u) * lrScale * colorGate;
  let opacityLr = cfg(12u) * lrScale * spatialGate;
  let scaleLr = cfg(13u) * lrScale * spatialGate;
  let rotationLr = cfg(14u) * lrScale * spatialGate;
  var nextCenter = center - geomAdam.xy * positionLr;
  var nextScale = exp(log(baseScale) - geomAdam.zw * scaleLr);
  var nextColor = clamp(rgba.rgb - colorAdam.rgb * colorLr, vec3<f32>(0.0), vec3<f32>(1.0));
  let currentLogit = log(clamp(rgba.a, 0.005, 0.995) / (1.0 - clamp(rgba.a, 0.005, 0.995)));
  var nextOpacity = 1.0 / (1.0 + exp(-(currentLogit - colorAdam.w * opacityLr)));
  var nextTheta = t.z - thetaAdam.x * rotationLr;
  let u1 = max(0.000001, hash_unit(f32(g) * 17.13 + step * 0.73));
  let u2 = hash_unit(f32(g) * 31.71 + step * 1.37);
  let normal = sqrt(-2.0 * log(u1)) * vec2<f32>(cos(6.28318530718 * u2), sin(6.28318530718 * u2));
  let covarianceNoise = vec2<f32>(c * normal.x * baseScale.x - s * normal.y * baseScale.y, s * normal.x * baseScale.x + c * normal.y * baseScale.y) / max(max(baseScale.x, baseScale.y), 0.0001);
  let noiseStep = (1.0 - progress) * positionLr;
  let defaultNoiseGate = 1.0 - rgba.a;
  let sigmoidNoiseGate = 1.0 / (1.0 + exp((rgba.a - 0.2) * 20.0));
  let noiseGate = select(defaultNoiseGate, sigmoidNoiseGate, cfg(34u) > 0.5);
  nextCenter += covarianceNoise * noiseStep * noiseGate * ${DEFAULT_SGLD_NOISE_LR};
  let minScale = ${MIN_SPLAT_SCALE};
  nextTheta = clamp(nextTheta, -3.14159265, 3.14159265);
  nextScale = max(nextScale, vec2<f32>(minScale));
  nextScale = min(nextScale, vec2<f32>(max(cfg(62u), minScale)));
  let major = max(nextScale.x, nextScale.y);
  let minor = max(minScale, min(nextScale.x, nextScale.y));
  let baseMaxAnisotropy = max(cfg(17u), 1.0);
  let surfaceMaxAnisotropy = max(cfg(51u), 1.0);
  let detailTagged = floor(t.w) >= 2.0;
  let maxAnisotropy = select(min(baseMaxAnisotropy, surfaceMaxAnisotropy), baseMaxAnisotropy, detailTagged);
  if (major / minor > maxAnisotropy) {
    let capped = minor * maxAnisotropy;
    nextScale = select(vec2<f32>(minor, capped), vec2<f32>(capped, minor), nextScale.x >= nextScale.y);
  }
  let nextCos = abs(cos(nextTheta));
  let nextSin = abs(sin(nextTheta));
  let boundarySigma = max(cfg(50u), 0.0);
  nextCenter = clamp(nextCenter, vec2<f32>(-1.0), vec2<f32>(1.0));
  if (boundarySigma > 0.0) {
    let minimumExtent = boundarySigma * minScale;
    nextCenter = clamp(nextCenter, vec2<f32>(-1.0 + minimumExtent), vec2<f32>(1.0 - minimumExtent));
    var extent = boundarySigma * vec2<f32>(
      length(vec2<f32>(nextCos * nextScale.x, nextSin * nextScale.y)),
      length(vec2<f32>(nextSin * nextScale.x, nextCos * nextScale.y))
    );
    let available = max(vec2<f32>(minimumExtent), vec2<f32>(1.0) - abs(nextCenter));
    let fit = min(1.0, min(available.x / max(extent.x, 0.00000001), available.y / max(extent.y, 0.00000001)));
    nextScale = max(vec2<f32>(minScale), nextScale * fit);
    extent = boundarySigma * vec2<f32>(
      length(vec2<f32>(nextCos * nextScale.x, nextSin * nextScale.y)),
      length(vec2<f32>(nextSin * nextScale.x, nextCos * nextScale.y))
    );
    let globalFit = min(1.0, 0.999 / max(extent.x, extent.y));
    nextScale = max(vec2<f32>(minScale), nextScale * globalFit);
    extent = boundarySigma * vec2<f32>(
      length(vec2<f32>(nextCos * nextScale.x, nextSin * nextScale.y)),
      length(vec2<f32>(nextSin * nextScale.x, nextCos * nextScale.y))
    );
    nextCenter = clamp(nextCenter, vec2<f32>(-1.0) + extent, vec2<f32>(1.0) - extent);
  }
  xy[g] = nextCenter;
  let layerTag = floor(t.w);
  var layerOrder = clamp(fract(t.w) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  if (cfg(45u) > 0.5 && cfg(54u) > 0.5) {
    let meanError = errorSum / max(observed, 1.0);
    let meanInfluence = influenceSum / max(observed, 1.0);
    let stableBias = (hash_unit(f32(g) * 0.754877666) - 0.5) * 0.02;
    let targetLayer = clamp(0.5 + meanInfluence * 0.35 - meanError * 0.8 + stableBias, 0.0, 1.0);
    layerOrder = mix(layerOrder, targetLayer, clamp(cfg(53u), 0.0, 1.0));
  }
  let packedLayer = layerTag + select(0.0, layerOrder * ${LAYER_CODE_RANGE}, cfg(45u) > 0.5);
  transform[g] = vec4<f32>(nextScale, nextTheta, packedLayer);
  color[g] = vec4<f32>(nextColor, clamp(nextOpacity, 0.005, 0.995));
  ${optimizerStatsUpdate}
}

@compute @workgroup_size(64)
fn optimize(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  let count = u32(cfg(2u));
  if (pixelState[0].a < 0.0) {
    if (g == 0u) { atomicAdd(&tileControl[2], 1u); }
    return;
  }
  if (g >= count || transform[g].w < 0.5) { return; }
  let capacity = u32(cfg(28u));
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let center = xy[g];
  let t = transform[g];
  let rgba = color[g];
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
  let useEwa = cfg(26u) > 0.5;
  var sampleScale = effective;
  var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
  var pixelPadding = vec2<f32>(0.0);
  if (useEwa) {
    sampleScale = baseScale;
    mip = 1.0;
    pixelPadding = vec2<f32>(
      select(0.0, 0.5 / f32(width - 1u), width > 1u),
      select(0.0, 0.5 / f32(height - 1u), height > 1u)
    );
  }
  let c = cos(t.z);
  let s = sin(t.z);
  let radius = vec2<f32>(
    ${RENDER_SIGMA} * (abs(c) * sampleScale.x + abs(s) * sampleScale.y),
    ${RENDER_SIGMA} * (abs(s) * sampleScale.x + abs(c) * sampleScale.y)
  ) + pixelPadding;
  let minNorm = max(vec2<f32>(-1.0), center - radius);
  let maxNorm = min(vec2<f32>(1.0), center + radius);
  let minPx = vec2<u32>(floor((minNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
  let maxPx = vec2<u32>(ceil((maxNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
  var gradColor = vec3<f32>(0.0);
  var gradLogit = 0.0;
  var gradCenter = vec2<f32>(0.0);
  var gradCenterAbs = vec2<f32>(0.0);
  var gradCenterNorm = 0.0;
  var gradLogScale = vec2<f32>(0.0);
  var gradTheta = 0.0;
  var influenceSum = 0.0;
  var errorSum = 0.0;
  var observed = 0.0;
  for (var py = minPx.y; py <= maxPx.y; py += 1u) {
    for (var px = minPx.x; px <= maxPx.x; px += 1u) {
      let contribution = pixel_gradient(px, py, width, height, center, rgba, baseScale, sampleScale, mip, useEwa, c, s);
      gradCenter += contribution.geom.xy;
      gradLogScale += contribution.geom.zw;
      gradColor += contribution.appearance.xyz;
      gradLogit += contribution.appearance.w;
      gradTheta += contribution.misc.x;
      influenceSum += contribution.misc.y;
      errorSum += contribution.misc.z;
      observed += contribution.misc.w;
      gradCenterAbs += contribution.density.xy;
      gradCenterNorm += contribution.density.z;
    }
  }
  let normalizer = 1.0 / max(influenceSum, 0.01);
  let gradGeom = vec4<f32>(gradCenter, gradLogScale) * normalizer;
  let gradAppearance = vec4<f32>(gradColor, gradLogit) * normalizer;
  let gradRotation = vec4<f32>(gradTheta * normalizer, 0.0, 0.0, 0.0);
  let beta1 = 0.9;
  let beta2 = 0.999;
  var opt = adam[g];
  let adcResetStep = opt.mTheta.w;
  opt.mGeom = beta1 * opt.mGeom + (1.0 - beta1) * gradGeom;
  opt.vGeom = beta2 * opt.vGeom + (1.0 - beta2) * gradGeom * gradGeom;
  opt.mColor = beta1 * opt.mColor + (1.0 - beta1) * gradAppearance;
  opt.vColor = beta2 * opt.vColor + (1.0 - beta2) * gradAppearance * gradAppearance;
  opt.mTheta = beta1 * opt.mTheta + (1.0 - beta1) * gradRotation;
  opt.vTheta = beta2 * opt.vTheta + (1.0 - beta2) * gradRotation * gradRotation;
  opt.mTheta.w = adcResetStep;
  let step = max(cfg(8u), 1.0);
  let useRowAge = cfg(30u) > 0.5 && adcResetStep > 0.0 && step >= adcResetStep;
  let optimizerAge = select(step, max(1.0, step - adcResetStep + 1.0), useRowAge);
  let bias1 = max(0.000001, 1.0 - pow(beta1, optimizerAge));
  let bias2 = max(0.000001, 1.0 - pow(beta2, optimizerAge));
  let geomAdam = (opt.mGeom / bias1) / (sqrt(opt.vGeom / bias2) + vec4<f32>(0.00000001));
  let colorAdam = (opt.mColor / bias1) / (sqrt(opt.vColor / bias2) + vec4<f32>(0.00000001));
  let thetaAdam = (opt.mTheta / bias1) / (sqrt(opt.vTheta / bias2) + vec4<f32>(0.00000001));
  let horizon = max(cfg(9u), 1.0);
  let afterDensity = max(0.0, step - horizon);
  let progress = min(1.0, step / horizon);
  let densityAnneal = max(0.05, 1.0 - progress);
  var adcRecovery = 0.0;
  if (adcResetStep > 0.0 && step > adcResetStep) {
    adcRecovery = exp(-(step - adcResetStep) / ${ADC_RECOVERY_DECAY_STEPS}.0);
  }
  let settleAnneal = max(exp(-afterDensity / 100.0), adcRecovery);
  let colorSettleAnneal = max(exp(-afterDensity / max(cfg(33u), 1.0)), adcRecovery);
  let spatialGate = densityAnneal * settleAnneal;
  let colorGate = densityAnneal * colorSettleAnneal;
  let lrScale = cfg(6u);
  let positionLr = cfg(10u) * lrScale * spatialGate;
  let colorLr = cfg(11u) * lrScale * colorGate;
  let opacityLr = cfg(12u) * lrScale * spatialGate;
  let scaleLr = cfg(13u) * lrScale * spatialGate;
  let rotationLr = cfg(14u) * lrScale * spatialGate;
  var nextCenter = center - geomAdam.xy * positionLr;
  var nextScale = exp(log(baseScale) - geomAdam.zw * scaleLr);
  var nextColor = clamp(rgba.rgb - colorAdam.rgb * colorLr, vec3<f32>(0.0), vec3<f32>(1.0));
  let currentLogit = log(clamp(rgba.a, 0.005, 0.995) / (1.0 - clamp(rgba.a, 0.005, 0.995)));
  var nextOpacity = 1.0 / (1.0 + exp(-(currentLogit - colorAdam.w * opacityLr)));
  var nextTheta = t.z - thetaAdam.x * rotationLr;
  let u1 = max(0.000001, hash_unit(f32(g) * 17.13 + step * 0.73));
  let u2 = hash_unit(f32(g) * 31.71 + step * 1.37);
  let normal = sqrt(-2.0 * log(u1)) * vec2<f32>(cos(6.28318530718 * u2), sin(6.28318530718 * u2));
  let covarianceNoise = vec2<f32>(c * normal.x * baseScale.x - s * normal.y * baseScale.y, s * normal.x * baseScale.x + c * normal.y * baseScale.y) / max(max(baseScale.x, baseScale.y), 0.0001);
  let noiseStep = (1.0 - progress) * positionLr;
  let defaultNoiseGate = 1.0 - rgba.a;
  let sigmoidNoiseGate = 1.0 / (1.0 + exp((rgba.a - 0.2) * 20.0));
  let noiseGate = select(defaultNoiseGate, sigmoidNoiseGate, cfg(34u) > 0.5);
  nextCenter += covarianceNoise * noiseStep * noiseGate * ${DEFAULT_SGLD_NOISE_LR};
  let minScale = ${MIN_SPLAT_SCALE};
  nextTheta = clamp(nextTheta, -3.14159265, 3.14159265);
  nextScale = max(nextScale, vec2<f32>(minScale));
  nextScale = min(nextScale, vec2<f32>(max(cfg(62u), minScale)));
  let major = max(nextScale.x, nextScale.y);
  let minor = max(minScale, min(nextScale.x, nextScale.y));
  let baseMaxAnisotropy = max(cfg(17u), 1.0);
  let surfaceMaxAnisotropy = max(cfg(51u), 1.0);
  let detailTagged = floor(t.w) >= 2.0;
  let maxAnisotropy = select(min(baseMaxAnisotropy, surfaceMaxAnisotropy), baseMaxAnisotropy, detailTagged);
  if (major / minor > maxAnisotropy) {
    let capped = minor * maxAnisotropy;
    nextScale = select(vec2<f32>(minor, capped), vec2<f32>(capped, minor), nextScale.x >= nextScale.y);
  }
  let nextCos = abs(cos(nextTheta));
  let nextSin = abs(sin(nextTheta));
  let boundarySigma = max(cfg(50u), 0.0);
  nextCenter = clamp(nextCenter, vec2<f32>(-1.0), vec2<f32>(1.0));
  if (boundarySigma > 0.0) {
    let minimumExtent = boundarySigma * minScale;
    nextCenter = clamp(nextCenter, vec2<f32>(-1.0 + minimumExtent), vec2<f32>(1.0 - minimumExtent));
    var extent = boundarySigma * vec2<f32>(
      length(vec2<f32>(nextCos * nextScale.x, nextSin * nextScale.y)),
      length(vec2<f32>(nextSin * nextScale.x, nextCos * nextScale.y))
    );
    let available = max(vec2<f32>(minimumExtent), vec2<f32>(1.0) - abs(nextCenter));
    let fit = min(1.0, min(available.x / max(extent.x, 0.00000001), available.y / max(extent.y, 0.00000001)));
    nextScale = max(vec2<f32>(minScale), nextScale * fit);
    extent = boundarySigma * vec2<f32>(
      length(vec2<f32>(nextCos * nextScale.x, nextSin * nextScale.y)),
      length(vec2<f32>(nextSin * nextScale.x, nextCos * nextScale.y))
    );
    let globalFit = min(1.0, 0.999 / max(extent.x, extent.y));
    nextScale = max(vec2<f32>(minScale), nextScale * globalFit);
    extent = boundarySigma * vec2<f32>(
      length(vec2<f32>(nextCos * nextScale.x, nextSin * nextScale.y)),
      length(vec2<f32>(nextSin * nextScale.x, nextCos * nextScale.y))
    );
    nextCenter = clamp(nextCenter, vec2<f32>(-1.0) + extent, vec2<f32>(1.0) - extent);
  }
  xy[g] = nextCenter;
  let layerTag = floor(t.w);
  var layerOrder = clamp(fract(t.w) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  if (cfg(45u) > 0.5 && cfg(54u) > 0.5) {
    let meanError = errorSum / max(observed, 1.0);
    let meanInfluence = influenceSum / max(observed, 1.0);
    let stableBias = (hash_unit(f32(g) * 0.754877666) - 0.5) * 0.02;
    let targetLayer = clamp(0.5 + meanInfluence * 0.35 - meanError * 0.8 + stableBias, 0.0, 1.0);
    layerOrder = mix(layerOrder, targetLayer, clamp(cfg(53u), 0.0, 1.0));
  }
  let packedLayer = layerTag + select(0.0, layerOrder * ${LAYER_CODE_RANGE}, cfg(45u) > 0.5);
  transform[g] = vec4<f32>(nextScale, nextTheta, packedLayer);
  color[g] = vec4<f32>(nextColor, clamp(nextOpacity, 0.005, 0.995));
  ${optimizerStatsUpdate}
}

@compute @workgroup_size(64)
fn optimize_parallel(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let g = wid.y * workgroups.x + wid.x;
  let count = u32(cfg(2u));
  if (pixelState[0].a < 0.0) {
    if (g == 0u && lid.x == 0u) { atomicAdd(&tileControl[2], 1u); }
    return;
  }
  if (g >= count) { return; }
  var isActive = g < count && transform[g].w >= 0.5;
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  var center = xy[g];
  let t = transform[g];
  let rgba = color[g];
  var baseScale = max(t.xy, vec2<f32>(0.0001));
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  var effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
  let useEwa = cfg(26u) > 0.5;
  var sampleScale = effective;
  var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
  var pixelPadding = vec2<f32>(0.0);
  if (useEwa) {
    sampleScale = baseScale;
    mip = 1.0;
    pixelPadding = vec2<f32>(
      select(0.0, 0.5 / f32(width - 1u), width > 1u),
      select(0.0, 0.5 / f32(height - 1u), height > 1u)
    );
  }
  var c = cos(t.z);
  var s = sin(t.z);
  let radius = vec2<f32>(
    ${RENDER_SIGMA} * (abs(c) * sampleScale.x + abs(s) * sampleScale.y),
    ${RENDER_SIGMA} * (abs(s) * sampleScale.x + abs(c) * sampleScale.y)
  ) + pixelPadding;
  let minNorm = max(vec2<f32>(-1.0), center - radius);
  let maxNorm = min(vec2<f32>(1.0), center + radius);
  var minPx = vec2<u32>(floor((minNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
  var maxPx = vec2<u32>(ceil((maxNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
  ${optimizerGeometryOverride}
  let spanX = maxPx.x - minPx.x + 1u;
  let pixelCount = (maxPx.y - minPx.y + 1u) * spanX;
  var geomSum = vec4<f32>(0.0);
  var appearanceSum = vec4<f32>(0.0);
  var miscSum = vec4<f32>(0.0);
  var densitySum = vec4<f32>(0.0);
  if (isActive) {
    for (var pixelOffset = lid.x; pixelOffset < pixelCount; pixelOffset += 64u) {
      let px = minPx.x + pixelOffset % spanX;
      let py = minPx.y + pixelOffset / spanX;
      let contribution = pixel_gradient(px, py, width, height, center, rgba, baseScale, sampleScale, mip, useEwa, c, s);
      geomSum += contribution.geom;
      appearanceSum += contribution.appearance;
      miscSum += contribution.misc;
      densitySum += contribution.density;
    }
  }
  reduceGeom[lid.x] = geomSum;
  reduceAppearance[lid.x] = appearanceSum;
  reduceMisc[lid.x] = miscSum;
  reduceDensity[lid.x] = densitySum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      reduceGeom[lid.x] += reduceGeom[lid.x + stride];
      reduceAppearance[lid.x] += reduceAppearance[lid.x + stride];
      reduceMisc[lid.x] += reduceMisc[lid.x + stride];
      reduceDensity[lid.x] += reduceDensity[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u && isActive) {
    apply_optimizer(g, center, t, rgba, baseScale, c, s, reduceGeom[0], reduceAppearance[0], reduceMisc[0], reduceDensity[0]);
  }
}

@compute @workgroup_size(64)
fn optimize_exact(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let g = id.x + id.y * workgroups.x * 64u;
  let count = u32(cfg(2u));
  if (pixelState[0].a < 0.0) {
    if (g == 0u) { atomicAdd(&tileControl[2], 1u); }
    return;
  }
  if (g >= count || transform[g].w < 0.5) { return; }
  let base = g * 16u;
  let geomSum = vec4<f32>(exactGradients[base], exactGradients[base + 1u], exactGradients[base + 2u], exactGradients[base + 3u]);
  let appearanceSum = vec4<f32>(exactGradients[base + 4u], exactGradients[base + 5u], exactGradients[base + 6u], exactGradients[base + 7u]);
  let miscSum = vec4<f32>(exactGradients[base + 8u], exactGradients[base + 9u], exactGradients[base + 10u], exactGradients[base + 11u]);
  let densitySum = vec4<f32>(exactGradients[base + 12u], exactGradients[base + 13u], exactGradients[base + 14u], 0.0);
  let center = xy[g];
  let t = transform[g];
  let rgba = color[g];
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  apply_optimizer(g, center, t, rgba, baseScale, cos(t.z), sin(t.z), geomSum, appearanceSum, miscSum, densitySum);
}`;

    const renderModule = this.device.createShaderModule({ code: renderShader });
    const ssimModule = this.device.createShaderModule({ code: ssimShader });
    const lossGradientModule = this.device.createShaderModule({ code: lossGradientShader });
    const exactBackwardModule = this.device.createShaderModule({ code: exactBackwardShader });
    const virtualOrderPenaltyModule = this.device.createShaderModule({ code: virtualOrderPenaltyShader });
    const exactBackwardTelemetryModule = this.device.createShaderModule({ code: exactBackwardTelemetryShader });
    const optimizerModule = this.device.createShaderModule({ code: optimizerShader });
    for (const module of [renderModule, ssimModule, lossGradientModule, exactBackwardModule, virtualOrderPenaltyModule, exactBackwardTelemetryModule, optimizerModule]) {
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === "error");
      if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    }
    [
      this.renderStatePipeline,
      this.tileCooperativeRenderPipeline,
      this.ssimTilePipeline,
      this.renderGradientPipeline,
      this.parallelRenderGradientPipeline,
      this.lossGradientPipeline,
      this.exactAlphaBackwardPipeline,
      this.virtualOrderPenaltyPipeline,
      this.exactBackwardTelemetryPipeline,
      this.exactOptimizerPipeline,
    ] = await Promise.all([
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: renderModule, entryPoint: "render_state" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: renderModule, entryPoint: "render_state_tile" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: ssimModule, entryPoint: "ssim_tiles" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: optimizerModule, entryPoint: "optimize" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: optimizerModule, entryPoint: "optimize_parallel" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: lossGradientModule, entryPoint: "loss_gradient" } }),
      this.device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: exactBackwardModule,
          entryPoint: this.quadExactBackwardEnabled ? "exact_alpha_backward_quad" : "exact_alpha_backward",
        },
      }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: virtualOrderPenaltyModule, entryPoint: "virtual_order_penalty" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: exactBackwardTelemetryModule, entryPoint: "measure_exact_backward" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: optimizerModule, entryPoint: "optimize_exact" } }),
    ]);
  }

  async ensurePixelMetricsPipeline() {
    if (this.pixelMetricsPipeline) return;
    const shader = `
struct Config { values: array<vec4<f32>, 8>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metricsOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> errorMap: array<f32>;
@group(0) @binding(5) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(6) var<storage, read> alphaState: array<AlphaState>;
var<workgroup> wgLoss: array<f32, 64>;
var<workgroup> wgX: array<f32, 64>;
var<workgroup> wgY: array<f32, 64>;
var<workgroup> wgX2: array<f32, 64>;
var<workgroup> wgY2: array<f32, 64>;
var<workgroup> wgXY: array<f32, 64>;
var<workgroup> wgMax: array<f32, 64>;
var<workgroup> wgCount: array<f32, 64>;
var<workgroup> wgCoverage: array<f32, 64>;
var<workgroup> wgCoverageMin: array<f32, 64>;
var<workgroup> wgCoverageUnder: array<f32, 64>;
var<workgroup> wgBackgroundExposure: array<f32, 64>;
var<workgroup> wgGradientError: array<f32, 64>;
var<workgroup> wgTargetGradientEnergy: array<f32, 64>;
var<workgroup> wgGradientCount: array<f32, 64>;
var<workgroup> wgAlphaError: array<f32, 64>;
var<workgroup> wgAlphaDark: array<vec4<f32>, 64>;
var<workgroup> wgAlphaMid: array<vec4<f32>, 64>;
var<workgroup> wgAlphaLight: array<vec4<f32>, 64>;
var<workgroup> wgAlphaMoments: array<vec4<f32>, 64>;
var<workgroup> wgAlphaCross: array<f32, 64>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

@compute @workgroup_size(64)
fn metrics(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let tileCols = (width + 7u) / 8u;
  let tileIndex = wid.y * workgroups.x + wid.x;
  let tileCount = tileCols * ((height + 7u) / 8u);
  if (tileIndex >= tileCount) { return; }
  let tileX = tileIndex % tileCols;
  let tileY = tileIndex / tileCols;
  let px = tileX * 8u + lid.x % 8u;
  let py = tileY * 8u + lid.x / 8u;
  var loss = 0.0;
  var x = 0.0;
  var y = 0.0;
  var valid = 0.0;
  var coverage = 0.0;
  var coverageUnder = 0.0;
  var backgroundExposure = 0.0;
  var gradientError = 0.0;
  var targetGradientEnergy = 0.0;
  var gradientCount = 0.0;
  var alphaError = 0.0;
  var alphaDark = vec4<f32>(0.0);
  var alphaMid = vec4<f32>(0.0);
  var alphaLight = vec4<f32>(0.0);
  var alphaMoments = vec4<f32>(0.0);
  var alphaCross = 0.0;
  if (px < width && py < height) {
    let pixel = py * width + px;
    let rendered = pixelState[pixel].rgb;
    alphaError = abs(alphaState[pixel].compositeAlpha - targetAlpha[pixel]);
    coverage = pixelState[pixel].a;
    alphaMoments = vec4<f32>(coverage, targetAlpha[pixel], coverage * coverage, targetAlpha[pixel] * targetAlpha[pixel]);
    alphaCross = coverage * targetAlpha[pixel];
    coverageUnder = select(0.0, 1.0, coverage < cfg(22u));
    backgroundExposure = select(0.0, 1.0, coverage < ${DEFAULT_ALPHA_TARGET});
    let targetIndex = pixel * 3u;
    let targetColor = vec3<f32>(targetRgb[targetIndex], targetRgb[targetIndex + 1u], targetRgb[targetIndex + 2u]);
    loss = (abs(rendered.r - targetColor.r) + abs(rendered.g - targetColor.g) + abs(rendered.b - targetColor.b)) / 3.0;
    x = dot(rendered, vec3<f32>(1.0 / 3.0));
    y = dot(targetColor, vec3<f32>(1.0 / 3.0));
    let alphaBucket = vec4<f32>(coverage, alphaError, backgroundExposure, 1.0);
    if (y < 0.25) {
      alphaDark = alphaBucket;
    } else if (y < 0.75) {
      alphaMid = alphaBucket;
    } else {
      alphaLight = alphaBucket;
    }
    if (px + 1u < width) {
      let rightPixel = pixel + 1u;
      let rightRendered = dot(pixelState[rightPixel].rgb, vec3<f32>(1.0 / 3.0));
      let rightTargetIndex = rightPixel * 3u;
      let rightTarget = (targetRgb[rightTargetIndex] + targetRgb[rightTargetIndex + 1u] + targetRgb[rightTargetIndex + 2u]) / 3.0;
      gradientError += abs((rightRendered - x) - (rightTarget - y));
      targetGradientEnergy += abs(rightTarget - y);
      gradientCount += 1.0;
    }
    if (py + 1u < height) {
      let downPixel = pixel + width;
      let downRendered = dot(pixelState[downPixel].rgb, vec3<f32>(1.0 / 3.0));
      let downTargetIndex = downPixel * 3u;
      let downTarget = (targetRgb[downTargetIndex] + targetRgb[downTargetIndex + 1u] + targetRgb[downTargetIndex + 2u]) / 3.0;
      gradientError += abs((downRendered - x) - (downTarget - y));
      targetGradientEnergy += abs(downTarget - y);
      gradientCount += 1.0;
    }
    errorMap[pixel] = loss + select(0.0, 0.2 * gradientError / max(1.0, gradientCount), cfg(20u) > 0.5);
    valid = 1.0;
  }
  wgLoss[lid.x] = loss;
  wgX[lid.x] = x;
  wgY[lid.x] = y;
  wgX2[lid.x] = x * x;
  wgY2[lid.x] = y * y;
  wgXY[lid.x] = x * y;
  wgMax[lid.x] = loss;
  wgCount[lid.x] = valid;
  wgCoverage[lid.x] = coverage;
  wgCoverageMin[lid.x] = select(1000000000.0, coverage, valid > 0.5);
  wgCoverageUnder[lid.x] = coverageUnder;
  wgBackgroundExposure[lid.x] = backgroundExposure;
  wgGradientError[lid.x] = gradientError;
  wgTargetGradientEnergy[lid.x] = targetGradientEnergy;
  wgGradientCount[lid.x] = gradientCount;
  wgAlphaError[lid.x] = alphaError;
  wgAlphaDark[lid.x] = alphaDark;
  wgAlphaMid[lid.x] = alphaMid;
  wgAlphaLight[lid.x] = alphaLight;
  wgAlphaMoments[lid.x] = alphaMoments;
  wgAlphaCross[lid.x] = alphaCross;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      wgLoss[lid.x] += wgLoss[lid.x + stride];
      wgX[lid.x] += wgX[lid.x + stride];
      wgY[lid.x] += wgY[lid.x + stride];
      wgX2[lid.x] += wgX2[lid.x + stride];
      wgY2[lid.x] += wgY2[lid.x + stride];
      wgXY[lid.x] += wgXY[lid.x + stride];
      wgMax[lid.x] = max(wgMax[lid.x], wgMax[lid.x + stride]);
      wgCount[lid.x] += wgCount[lid.x + stride];
      wgCoverage[lid.x] += wgCoverage[lid.x + stride];
      wgCoverageMin[lid.x] = min(wgCoverageMin[lid.x], wgCoverageMin[lid.x + stride]);
      wgCoverageUnder[lid.x] += wgCoverageUnder[lid.x + stride];
      wgBackgroundExposure[lid.x] += wgBackgroundExposure[lid.x + stride];
      wgGradientError[lid.x] += wgGradientError[lid.x + stride];
      wgTargetGradientEnergy[lid.x] += wgTargetGradientEnergy[lid.x + stride];
      wgGradientCount[lid.x] += wgGradientCount[lid.x + stride];
      wgAlphaError[lid.x] += wgAlphaError[lid.x + stride];
      wgAlphaDark[lid.x] += wgAlphaDark[lid.x + stride];
      wgAlphaMid[lid.x] += wgAlphaMid[lid.x + stride];
      wgAlphaLight[lid.x] += wgAlphaLight[lid.x + stride];
      wgAlphaMoments[lid.x] += wgAlphaMoments[lid.x + stride];
      wgAlphaCross[lid.x] += wgAlphaCross[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let out = tileIndex * ${METRIC_TILE_STRIDE}u;
    metricsOut[out] = wgLoss[0];
    metricsOut[out + 1u] = wgX[0];
    metricsOut[out + 2u] = wgY[0];
    metricsOut[out + 3u] = wgX2[0];
    metricsOut[out + 4u] = wgY2[0];
    metricsOut[out + 5u] = wgXY[0];
    metricsOut[out + 6u] = wgMax[0];
    metricsOut[out + 7u] = wgCount[0];
    metricsOut[out + 8u] = wgCoverage[0];
    metricsOut[out + 9u] = wgCoverageMin[0];
    metricsOut[out + 10u] = wgCoverageUnder[0];
    metricsOut[out + 11u] = wgBackgroundExposure[0];
    metricsOut[out + 12u] = wgGradientError[0];
    metricsOut[out + 13u] = wgTargetGradientEnergy[0];
    metricsOut[out + 14u] = wgGradientCount[0];
    metricsOut[out + 15u] = wgAlphaError[0];
    metricsOut[out + 16u] = wgAlphaDark[0].x;
    metricsOut[out + 17u] = wgAlphaDark[0].y;
    metricsOut[out + 18u] = wgAlphaDark[0].z;
    metricsOut[out + 19u] = wgAlphaDark[0].w;
    metricsOut[out + 20u] = wgAlphaMid[0].x;
    metricsOut[out + 21u] = wgAlphaMid[0].y;
    metricsOut[out + 22u] = wgAlphaMid[0].z;
    metricsOut[out + 23u] = wgAlphaMid[0].w;
    metricsOut[out + 24u] = wgAlphaLight[0].x;
    metricsOut[out + 25u] = wgAlphaLight[0].y;
    metricsOut[out + 26u] = wgAlphaLight[0].z;
    metricsOut[out + 27u] = wgAlphaLight[0].w;
    metricsOut[out + 28u] = wgAlphaMoments[0].x;
    metricsOut[out + 29u] = wgAlphaMoments[0].y;
    metricsOut[out + 30u] = wgAlphaMoments[0].z;
    metricsOut[out + 31u] = wgAlphaMoments[0].w;
    metricsOut[out + 32u] = wgAlphaCross[0];
  }
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.pixelMetricsPipeline = await this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "metrics" } });
  }

  async ensureOverlapMetricsPipeline() {
    if (this.overlapMetricsPipeline) return;
    const shader = `
struct Config { values: array<vec4<f32>, 12>, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(5) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(6) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(7) var<storage, read> tileIndices: array<u32>;
@group(0) @binding(8) var<storage, read_write> metricsOut: array<f32>;
var<workgroup> reduceA: array<vec4<f32>, 64>;
var<workgroup> reduceB: array<vec4<f32>, 64>;
var<workgroup> reduceC: array<vec4<f32>, 64>;
var<workgroup> reduceD: array<vec4<f32>, 64>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn gaussian_kernel(d: vec2<f32>, c: f32, s: f32, scale: vec2<f32>) -> f32 {
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  let q = dot(r / scale, r / scale);
  return exp(-0.5 * q);
}

@compute @workgroup_size(64)
fn overlap_metrics(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let tileCols8 = (width + 7u) / 8u;
  let tileIndex = wid.y * workgroups.x + wid.x;
  let tileCount = tileCols8 * ((height + 7u) / 8u);
  if (tileIndex >= tileCount) { return; }
  let tileX8 = tileIndex % tileCols8;
  let tileY8 = tileIndex / tileCols8;
  let px = tileX8 * 8u + lid.x % 8u;
  let py = tileY8 * 8u + lid.x / 8u;
  var a = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  var cc = vec4<f32>(0.0);
  var dOut = vec4<f32>(0.0);
  if (px < width && py < height) {
    let pixel = py * width + px;
    let p = vec2<f32>(
      select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
      select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
    );
    let targetIndex = pixel * 3u;
    let targetColor = vec3<f32>(targetRgb[targetIndex], targetRgb[targetIndex + 1u], targetRgb[targetIndex + 2u]);
    let useTiles = cfg(19u) > 0.5;
    let tileCols = (width + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
    let tile = (py / ${TILE_SIZE}u) * tileCols + (px / ${TILE_SIZE}u);
    let capacity = arrayLength(&tileIndices);
    let start = select(0u, min(tileOffsets[tile] & 0x7fffffffu, capacity), useTiles);
    let end = select(u32(cfg(2u)), min(tileOffsets[tile + 1u] & 0x7fffffffu, capacity), useTiles);
    let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
    let scaleFactor = clamp(cfg(45u), 0.01, 1.0);
    var numerator = vec3<f32>(0.0);
    var colorSecond = vec3<f32>(0.0);
    var denom = 0.0;
    var sumW2 = 0.0;
    var maxW = 0.0;
    var sumWLogW = 0.0;
    var targetDistance = 0.0;
    var transmittance = 1.0;
    var compositedRgb = vec3<f32>(0.0);
    var cursor = start;
    loop {
      if (cursor >= end) { break; }
      var g = cursor;
      if (useTiles) { g = tileIndices[cursor]; }
      let t = transform[g];
      if (t.w >= 0.5) {
        let center = xy[g];
        let c = cos(t.z);
        let s = sin(t.z);
        let baseScale = max(t.xy * scaleFactor, vec2<f32>(0.0001));
        let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
        var kernel = gaussian_kernel(p - center, c, s, effective);
        var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
        if (cfg(26u) > 0.5) {
          let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
          let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
          let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
          kernel = 0.25 * (
            gaussian_kernel(clamp(p + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
            gaussian_kernel(clamp(p + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
            gaussian_kernel(clamp(p + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
            gaussian_kernel(clamp(p + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale)
          );
          mip = 1.0;
        }
        let weight = clamp(kernel * color[g].a * mip, 0.0, 0.99);
        if (kernel >= 0.0003354626 && weight >= 0.0039215686) {
          let rgb = color[g].rgb;
          numerator += weight * rgb;
          colorSecond += weight * rgb * rgb;
          denom += weight;
          sumW2 += weight * weight;
          maxW = max(maxW, weight);
          sumWLogW += weight * log(max(weight, 0.00000001));
          targetDistance += weight * dot(abs(rgb - targetColor), vec3<f32>(1.0 / 3.0));
          if (transmittance >= 0.0001) {
            compositedRgb += transmittance * weight * rgb;
            transmittance *= 1.0 - weight;
          }
        }
      }
      cursor += 1u;
    }
    let covered = denom > ${BACKGROUND_EXPOSURE_EPSILON};
    let weightedMean = select(vec3<f32>(0.0), numerator / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered);
    let rendered = compositedRgb + transmittance * vec3<f32>(cfg(3u), cfg(4u), cfg(5u));
    let variance = max(vec3<f32>(0.0), colorSecond / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}) - weightedMean * weightedMean);
    let effectiveContributors = select(0.0, denom * denom / max(sumW2, 0.0000000000000001), covered);
    let maxShare = select(0.0, maxW / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered);
    let entropy = select(0.0, max(0.0, log(max(denom, 0.00000001)) - sumWLogW / max(denom, ${BACKGROUND_EXPOSURE_EPSILON})), covered);
    let alpha = 1.0 - transmittance;
    let rgbError = abs(rendered - targetColor);
    let l1 = dot(rgbError, vec3<f32>(1.0 / 3.0));
    let maxChannel = max(rgbError.r, max(rgbError.g, rgbError.b));
    a = vec4<f32>(1.0, denom, effectiveContributors, maxShare);
    b = vec4<f32>(entropy, alpha, abs(alpha - targetAlpha[pixel]), dot(variance, vec3<f32>(1.0 / 3.0)));
    cc = vec4<f32>(select(0.0, targetDistance / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered), l1, maxChannel, select(0.0, 1.0, maxChannel > 0.10));
    dOut = vec4<f32>(select(0.0, 1.0, alpha < ${DEFAULT_ALPHA_TARGET}), sumW2, maxW, maxChannel);
  }
  reduceA[lid.x] = a;
  reduceB[lid.x] = b;
  reduceC[lid.x] = cc;
  reduceD[lid.x] = dOut;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      reduceA[lid.x] += reduceA[lid.x + stride];
      reduceB[lid.x] += reduceB[lid.x + stride];
      reduceC[lid.x] += reduceC[lid.x + stride];
      reduceD[lid.x] = vec4<f32>(
        reduceD[lid.x].xyz + reduceD[lid.x + stride].xyz,
        max(reduceD[lid.x].w, reduceD[lid.x + stride].w)
      );
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let out = tileIndex * ${OVERLAP_METRIC_STRIDE}u;
    metricsOut[out] = reduceA[0].x;
    metricsOut[out + 1u] = reduceA[0].y;
    metricsOut[out + 2u] = reduceA[0].z;
    metricsOut[out + 3u] = reduceA[0].w;
    metricsOut[out + 4u] = reduceB[0].x;
    metricsOut[out + 5u] = reduceB[0].y;
    metricsOut[out + 6u] = reduceB[0].z;
    metricsOut[out + 7u] = reduceB[0].w;
    metricsOut[out + 8u] = reduceC[0].x;
    metricsOut[out + 9u] = reduceC[0].y;
    metricsOut[out + 10u] = reduceC[0].z;
    metricsOut[out + 11u] = reduceC[0].w;
    metricsOut[out + 12u] = reduceD[0].x;
    metricsOut[out + 13u] = reduceD[0].y;
    metricsOut[out + 14u] = reduceD[0].z;
    metricsOut[out + 15u] = reduceD[0].w;
  }
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.overlapMetricsPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "overlap_metrics" },
    });
  }

  async ensureAlphaLossPipeline() {
    if (this.alphaLossPipeline) return;
    const shader = `
struct Config { values: array<vec4<f32>, 12>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(5) var<storage, read> alphaState: array<AlphaState>;
var<workgroup> reduceGradient: array<f32, 64>;
var<workgroup> reduceWeight: array<f32, 64>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn gaussian_kernel(d: vec2<f32>, c: f32, s: f32, scale: vec2<f32>) -> f32 {
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  return exp(-0.5 * dot(r / scale, r / scale));
}
@compute @workgroup_size(64)
fn alpha_loss(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let g = wid.y * workgroups.x + wid.x;
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let alphaActive = cfg(46u) > 0.0;
  let isActive = g < u32(cfg(2u)) && transform[g].w >= 0.5 && alphaActive;
  var gradient = 0.0;
  var weightSum = 0.0;
  if (isActive) {
    let center = xy[g];
    let t = transform[g];
    let rgba = color[g];
    let c = cos(t.z);
    let s = sin(t.z);
    let baseScale = max(t.xy, vec2<f32>(0.0001));
    let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
    let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
    let useEwa = cfg(26u) > 0.5;
    let radius = vec2<f32>(
      ${RENDER_SIGMA} * (abs(c) * effective.x + abs(s) * effective.y),
      ${RENDER_SIGMA} * (abs(s) * effective.x + abs(c) * effective.y)
    );
    let minNorm = max(vec2<f32>(-1.0), center - radius);
    let maxNorm = min(vec2<f32>(1.0), center + radius);
    let minPx = vec2<u32>(floor((minNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
    let maxPx = vec2<u32>(ceil((maxNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
    let spanX = maxPx.x - minPx.x + 1u;
    let pixelCount = (maxPx.y - minPx.y + 1u) * spanX;
    for (var offset = lid.x; offset < pixelCount; offset += 64u) {
      let px = minPx.x + offset % spanX;
      let py = minPx.y + offset / spanX;
      let p = vec2<f32>(
        select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
        select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
      );
      var kernel = gaussian_kernel(p - center, c, s, effective);
      var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
      if (useEwa) {
        let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
        let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
        let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
        kernel = 0.25 * (
          gaussian_kernel(clamp(p + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
          gaussian_kernel(clamp(p + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
          gaussian_kernel(clamp(p + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
          gaussian_kernel(clamp(p + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale)
        );
        mip = 1.0;
      }
      let rawWeight = clamp(kernel * rgba.a * mip, 0.0, 0.99);
      if (kernel >= 0.0003354626 && rawWeight >= 0.0039215686) {
        let pixel = py * width + px;
        let alpha = alphaState[pixel].compositeAlpha;
        let alphaGoal = targetAlpha[pixel];
        let derivative = (1.0 - alpha) * rawWeight * (1.0 - rgba.a) / max(0.01, 1.0 - rawWeight);
        gradient += sign(alpha - alphaGoal) * derivative;
        weightSum += rawWeight;
      }
    }
  }
  reduceGradient[lid.x] = gradient;
  reduceWeight[lid.x] = weightSum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      reduceGradient[lid.x] += reduceGradient[lid.x + stride];
      reduceWeight[lid.x] += reduceWeight[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u && isActive) {
    let rgba = color[g];
    let logit = log(clamp(rgba.a, 0.005, 0.995) / (1.0 - clamp(rgba.a, 0.005, 0.995)));
    let learningRate = min(0.05, cfg(12u) * cfg(46u));
    let nextOpacity = 1.0 / (1.0 + exp(-(logit - learningRate * reduceGradient[0] / max(reduceWeight[0], 0.01))));
    color[g] = vec4<f32>(rgba.rgb, clamp(nextOpacity, 0.005, 0.995));
  }
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.alphaLossPipeline = await this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "alpha_loss" } });
  }

  async computeOverlapDiagnostics(image, params) {
    if (!this.trainState || !phase40Variants().overlapDiagnostics) return null;
    await this.ensureOverlapMetricsPipeline();
    const partialCount = Math.ceil(image.width / 8) * Math.ceil(image.height / 8);
    const outputBytes = partialCount * OVERLAP_METRIC_STRIDE * 4;
    const configBuffer = this.device.createBuffer({ size: 48 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const outputBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const front = this.trainState.front;
    const summarize = (values, scale) => {
      const totals = new Float64Array(OVERLAP_METRIC_STRIDE);
      let maxChannelError = 0;
      for (let i = 0; i < values.length; i += OVERLAP_METRIC_STRIDE) {
        for (let j = 0; j < OVERLAP_METRIC_STRIDE - 1; j += 1) totals[j] += values[i + j];
        maxChannelError = Math.max(maxChannelError, values[i + 15]);
      }
      const pixels = Math.max(1, totals[0]);
      const coveredPixels = Math.max(1, totals[0] - totals[12]);
      return {
        scale,
        pixels: totals[0],
        covered_pixels: totals[0] - totals[12],
        covered_ratio: (totals[0] - totals[12]) / pixels,
        mean_weight_sum: totals[1] / pixels,
        mean_effective_contributors: totals[2] / coveredPixels,
        mean_max_weight_share: totals[3] / coveredPixels,
        mean_weight_entropy: totals[4] / coveredPixels,
        mean_composited_alpha: totals[5] / pixels,
        mean_alpha_error: totals[6] / pixels,
        mean_contributor_color_variance: totals[7] / coveredPixels,
        mean_contributor_target_distance: totals[8] / coveredPixels,
        l1: totals[9] / pixels,
        mean_max_channel_error: totals[10] / pixels,
        bad_pixel_count_0_10: totals[11],
        bad_pixel_ratio_0_10: totals[11] / pixels,
        background_exposure_count: totals[12],
        mean_weight_square_sum: totals[13] / pixels,
        mean_max_weight: totals[14] / pixels,
        max_channel_error: maxChannelError,
      };
    };
    try {
      const results = {};
      for (const scale of [1, 0.5, 0.25]) {
        const config = new Float32Array(48);
        config.set([image.width, image.height, params.count, params.bg[0], params.bg[1], params.bg[2]], 0);
        config[19] = els.tileCullingToggle.checked ? 1 : 0;
        config[26] = phase33Variants().ewa2x2 ? 1 : 0;
        config[31] = phase37Variants().ewaGaussLegendre ? 1 : 0;
        config[45] = scale;
        this.device.queue.writeBuffer(configBuffer, 0, config);
        const bindGroup = this.device.createBindGroup({
          layout: this.overlapMetricsPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: configBuffer } },
            { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
            { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
            { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
            { binding: 4, resource: { buffer: this.trainState.targetBuffer } },
            { binding: 5, resource: { buffer: this.trainState.targetAlphaBuffer } },
            { binding: 6, resource: { buffer: this.trainState.tileOffsetsBuffer } },
            { binding: 7, resource: { buffer: this.trainState.tileIndicesBuffer } },
            { binding: 8, resource: { buffer: outputBuffer } },
          ],
        });
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.overlapMetricsPipeline);
        pass.setBindGroup(0, bindGroup);
        this.dispatchLinear(pass, partialCount);
        pass.end();
        encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBytes);
        this.device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(readBuffer.getMappedRange()).slice();
        readBuffer.unmap();
        results[String(scale)] = summarize(values, scale);
      }
      return {
        backend: "webgpu-final-only",
        standard_alpha_blend: true,
        source_alpha_preserved: Boolean(image.alpha),
        scales: results,
      };
    } finally {
      configBuffer.destroy();
      outputBuffer.destroy();
      readBuffer.destroy();
    }
  }

  async refreshRenderState(image, params) {
    await this.ensureRenderGradientPipelines();
    const variants = phase33Variants();
    const config = new Float32Array(32);
    config.set([image.width, image.height, params.count, params.bg[0], params.bg[1], params.bg[2]], 0);
    config[17] = currentMaxAnisotropy();
    config[18] = experimentalDensifySteps(state.metrics?.steps_requested || 1);
    config[19] = els.tileCullingToggle.checked ? 1 : 0;
    config[20] = phase37Variants().structuralErrorMap ? 1 : 0;
    config[22] = variants.coverageTarget;
    config[26] = variants.ewa2x2 ? 1 : 0;
    this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    const front = this.trainState.front;
    const renderChoice = this.renderStatePipelineChoice();
    const renderBindGroup = this.device.createBindGroup({
      layout: renderChoice.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
        { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
        { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
        { binding: 5, resource: { buffer: this.trainState.tileOffsetsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
        { binding: 7, resource: { buffer: this.trainState.pixelStateBuffer } },
        { binding: 8, resource: { buffer: this.trainState.alphaStateBuffer } },
      ],
    });
    const stageKind = image === this.trainState.coarseImage
      ? "coarse"
      : image === this.trainState.midImage
        ? "mid"
        : "full";
    const targetBuffer = stageKind === "coarse"
      ? this.trainState.coarseTargetBuffer
      : stageKind === "mid"
        ? this.trainState.midTargetBuffer
        : this.trainState.targetBuffer;
    const targetAlphaBuffer = stageKind === "coarse"
      ? this.trainState.coarseTargetAlphaBuffer
      : stageKind === "mid"
        ? this.trainState.midTargetAlphaBuffer
        : this.trainState.targetAlphaBuffer;
    const ssimBindGroup = this.device.createBindGroup({
      layout: this.ssimTilePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: targetBuffer } },
        { binding: 2, resource: { buffer: this.trainState.pixelStateBuffer } },
        { binding: 3, resource: { buffer: this.trainState.ssimTileBuffer } },
        { binding: 4, resource: { buffer: targetAlphaBuffer } },
        { binding: 5, resource: { buffer: this.trainState.alphaStateBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const renderPass = encoder.beginComputePass();
    renderPass.setPipeline(renderChoice.pipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    if (renderChoice.cooperative) {
      renderPass.dispatchWorkgroups(Math.ceil(image.width / TILE_SIZE), Math.ceil(image.height / TILE_SIZE));
    } else {
      this.dispatchLinear(renderPass, Math.ceil((image.width * image.height) / 64));
    }
    renderPass.end();
    const ssimPass = encoder.beginComputePass();
    ssimPass.setPipeline(this.ssimTilePipeline);
    ssimPass.setBindGroup(0, ssimBindGroup);
    this.dispatchLinear(ssimPass, Math.ceil(image.width / 8) * Math.ceil(image.height / 8));
    ssimPass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    this.trainState.pixelStateResolution = [image.width, image.height];
    this.trainState.pixelStateKind = stageKind;
  }

  async computeTrainStateMetrics(image, params) {
    await this.ensurePixelMetricsPipeline();
    await this.refreshRenderState(image, params);
    const partialCount = Math.ceil(image.width / 8) * Math.ceil(image.height / 8);
    const outputBytes = partialCount * METRIC_TILE_STRIDE * 4;
    const lossBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const bindGroup = this.device.createBindGroup({
        layout: this.pixelMetricsPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.targetBuffer } },
          { binding: 2, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 3, resource: { buffer: lossBuffer } },
          { binding: 4, resource: { buffer: this.trainState.errorMapBuffer } },
          { binding: 5, resource: { buffer: this.trainState.targetAlphaBuffer } },
          { binding: 6, resource: { buffer: this.trainState.alphaStateBuffer } },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pixelMetricsPipeline);
      pass.setBindGroup(0, bindGroup);
      this.dispatchLinear(pass, partialCount);
      pass.end();
      encoder.copyBufferToBuffer(lossBuffer, 0, readBuffer, 0, outputBytes);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();
      let lossTotal = 0;
      let renderedY = 0;
      let targetY = 0;
      let renderedY2 = 0;
      let targetY2 = 0;
      let renderedTargetY = 0;
      let maxLoss = 0;
      let windowedTotal = 0;
      let windowedCount = 0;
      let coverageTotal = 0;
      let coverageMinimum = Number.POSITIVE_INFINITY;
      let coverageUnder = 0;
      let backgroundExposure = 0;
      let gradientError = 0;
      let targetGradientEnergy = 0;
      let gradientCount = 0;
      let alphaError = 0;
      const alphaBuckets = {
        dark: { alpha: 0, error: 0, under: 0, count: 0 },
        mid: { alpha: 0, error: 0, under: 0, count: 0 },
        light: { alpha: 0, error: 0, under: 0, count: 0 },
      };
      const alphaMoments = { rendered: 0, target: 0, rendered2: 0, target2: 0, cross: 0 };
      for (let i = 0; i < values.length; i += METRIC_TILE_STRIDE) {
        const count = values[i + 7];
        if (count <= 0) continue;
        lossTotal += values[i];
        renderedY += values[i + 1];
        targetY += values[i + 2];
        renderedY2 += values[i + 3];
        targetY2 += values[i + 4];
        renderedTargetY += values[i + 5];
        maxLoss = Math.max(maxLoss, values[i + 6]);
        coverageTotal += values[i + 8];
        coverageMinimum = Math.min(coverageMinimum, values[i + 9]);
        coverageUnder += values[i + 10];
        backgroundExposure += values[i + 11];
        gradientError += values[i + 12];
        targetGradientEnergy += values[i + 13];
        gradientCount += values[i + 14];
        alphaError += values[i + 15];
        for (const [name, offset] of [["dark", 16], ["mid", 20], ["light", 24]]) {
          alphaBuckets[name].alpha += values[i + offset];
          alphaBuckets[name].error += values[i + offset + 1];
          alphaBuckets[name].under += values[i + offset + 2];
          alphaBuckets[name].count += values[i + offset + 3];
        }
        alphaMoments.rendered += values[i + 28];
        alphaMoments.target += values[i + 29];
        alphaMoments.rendered2 += values[i + 30];
        alphaMoments.target2 += values[i + 31];
        alphaMoments.cross += values[i + 32];
        const meanX = values[i + 1] / count;
        const meanY = values[i + 2] / count;
        windowedTotal += ssimFromMoments(meanX, meanY, Math.max(0, values[i + 3] / count - meanX ** 2), Math.max(0, values[i + 4] / count - meanY ** 2), values[i + 5] / count - meanX * meanY);
        windowedCount += 1;
      }
      const pixelCount = image.width * image.height;
      const loss = lossTotal / pixelCount;
      const alphaL1 = alphaError / pixelCount;
      const alphaMean = alphaMoments.rendered / pixelCount;
      const alphaTargetMean = alphaMoments.target / pixelCount;
      const alphaSsim = ssimFromMoments(
        alphaMean,
        alphaTargetMean,
        Math.max(0, alphaMoments.rendered2 / pixelCount - alphaMean ** 2),
        Math.max(0, alphaMoments.target2 / pixelCount - alphaTargetMean ** 2),
        alphaMoments.cross / pixelCount - alphaMean * alphaTargetMean,
      );
      const alphaWeight = phase40Variants().alphaLossWeight;
      const alphaObjective = (1 - DEFAULT_DSSIM_WEIGHT) * alphaL1 + DEFAULT_DSSIM_WEIGHT * (1 - alphaSsim) * 0.5;
      const objectiveLoss = loss + alphaWeight * alphaObjective;
      const meanX = renderedY / pixelCount;
      const meanY = targetY / pixelCount;
      const ssim = ssimFromMoments(meanX, meanY, Math.max(0, renderedY2 / pixelCount - meanX ** 2), Math.max(0, targetY2 / pixelCount - meanY ** 2), renderedTargetY / pixelCount - meanX * meanY);
      const windowedSsim = windowedTotal / Math.max(1, windowedCount);
      const regionalSsim = regionalSsimFromTileMetrics(values, image.width, image.height);
      const highFrequency = {
        gradient_l1: gradientError / Math.max(1, gradientCount),
        target_gradient_energy: targetGradientEnergy / Math.max(1, gradientCount),
        gradient_fidelity: 1 - gradientError / Math.max(0.000001, targetGradientEnergy),
        high_energy_regions: regionalSsim.high_frequency_regions,
      };
      const coverage = {
        target: phase33Variants().coverageTarget,
        mean: coverageTotal / Math.max(1, pixelCount),
        minimum: Number.isFinite(coverageMinimum) ? coverageMinimum : null,
        under_count: coverageUnder,
        under_ratio: coverageUnder / Math.max(1, pixelCount),
        background_exposure_count: backgroundExposure,
        background_exposure_ratio: backgroundExposure / Math.max(1, pixelCount),
        background_exposure_alpha_threshold: DEFAULT_ALPHA_TARGET,
        luminance_buckets: Object.fromEntries(Object.entries(alphaBuckets).map(([name, bucket]) => {
          const count = Math.max(1, bucket.count);
          return [name, {
            count: bucket.count,
            mean_alpha: bucket.alpha / count,
            mean_alpha_error: bucket.error / count,
            under_0_99_count: bucket.under,
            under_0_99_ratio: bucket.under / count,
          }];
        })),
      };
      this.lastLossStats = { loss, alphaL1, alphaSsim, alphaObjective, alphaWeight, objectiveLoss, ssim, windowedSsim, regionalSsim, highFrequency, coverage, max: maxLoss, count: pixelCount, partial_count: partialCount, bytes: outputBytes, reduction: "tile-8x8-from-compact-render", compact_tile_candidates: Boolean(els.tileCullingToggle.checked) };
      return { loss, alphaL1, alphaSsim, alphaObjective, alphaWeight, objectiveLoss, ssim, windowedSsim, regionalSsim, highFrequency, coverage };
    } finally {
      lossBuffer.destroy();
      readBuffer.destroy();
    }
  }

  async ensureDensityPipelines() {
    if (this.growSelectPipeline && this.distributionOffsetPipeline && this.relocationApplyPipeline && this.phase45RegionTelemetryPipeline && this.phase45RegionFinalizePipeline && this.phase45DonorSafetyPipeline) {
      await this.ensureOptimizerResetPipeline();
      return;
    }
    if (!this.densityBindGroupLayout) {
      const compute = GPUShaderStage.COMPUTE;
      this.densityBindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: compute, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: compute, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: compute, buffer: { type: "storage" } },
          { binding: 3, visibility: compute, buffer: { type: "storage" } },
          { binding: 4, visibility: compute, buffer: { type: "storage" } },
          { binding: 5, visibility: compute, buffer: { type: "storage" } },
          { binding: 6, visibility: compute, buffer: { type: "storage" } },
          { binding: 7, visibility: compute, buffer: { type: "read-only-storage" } },
        ],
      });
    }
    const shader = `
@group(0) @binding(0) var<storage, read> config: array<f32>;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read_write> xy: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> transform: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> color: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> stats: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> errorMap: array<f32>;

const SOURCE_MASK = 0x3fffffffu;
const ROLE_DESTINATION = 0x80000000u;
const ROLE_SOURCE_SPLIT = 0x40000000u;
const ROLE_SOURCE_OTHER = 0x20000000u;
const ROLE_MASK = 0xe0000000u;
const ROLE_TOKEN_MASK = 0x1fffffffu;
const CDF_BLOCK_SIZE = 256u;
const EVENT_SLOTS = ${DENSITY_EVENT_SLOTS}u;
const PHASE45_REGION_GRID = ${PHASE45_REGION_GRID}u;
const PHASE45_REGION_COUNT = ${PHASE45_REGION_COUNT}u;
const PHASE45_REGION_STRIDE = ${PHASE45_REGION_STRIDE}u;
const PHASE45_ENERGY_QUANTIZATION = 65536.0;
const PHASE45_RESIDUAL_QUANTIZATION = 4096.0;
const PHASE45_NORMALIZED_QUANTIZATION = 256.0;
const PHASE45_DONOR_ELIGIBLE = 1u;
var<workgroup> cdfScratch: array<f32, 256>;
var<workgroup> phase45Demand: array<u32, ${PHASE45_REGION_COUNT}>;

fn hash_unit(seed: f32) -> f32 {
  let x = sin((seed + config[33] * 104729.0) * 12.9898) * 43758.5453123;
  return x - floor(x);
}

fn target_at(pos: vec2<f32>, width: u32, height: u32) -> vec3<f32> {
  let safePos = min(max(pos, vec2<f32>(-1.0)), vec2<f32>(1.0));
  let px = min(width - 1u, u32(floor((safePos.x * 0.5 + 0.5) * f32(width - 1u) + 0.5)));
  let py = min(height - 1u, u32(floor((safePos.y * 0.5 + 0.5) * f32(height - 1u) + 0.5)));
  let index = (py * width + px) * 3u;
  return vec3<f32>(targetRgb[index], targetRgb[index + 1u], targetRgb[index + 2u]);
}

// Returns risk, split-axis (1 = local X), projected depth span, and color mismatch.
// This runs only during density events; normal optimizer iterations do not pay for it.
fn tilt_split_profile(g: u32, width: u32, height: u32) -> vec4<f32> {
  if (config[37] <= 0.5) { return vec4<f32>(0.0); }
  let t = transform[g];
  let sourceColor = color[g];
  if (t.w < 0.5 || sourceColor.a < 0.007) { return vec4<f32>(0.0); }
  let c = cos(t.z);
  let s = sin(t.z);
  let axisX = vec2<f32>(c, s) * t.x;
  let axisY = vec2<f32>(-s, c) * t.y;
  let longSide = max(config[0], config[1]);
  let frameScale = vec2<f32>(config[0] / longSide, config[1] / longSide);
  let worldX = axisX * frameScale;
  let worldY = axisY * frameScale;
  let angleSin = sin(max(0.0, config[38]));
  let yawDepth = 4.0 * angleSin * length(vec2<f32>(worldX.x, worldY.x));
  let pitchDepth = 4.0 * angleSin * length(vec2<f32>(worldX.y, worldY.y));
  let supportDepth = max(yawDepth, pitchDepth);
  let depthThreshold = max(0.000001, config[39]);
  if (supportDepth <= depthThreshold) {
    return vec4<f32>(0.0, select(0.0, 1.0, max(abs(worldX.x), abs(worldX.y)) >= max(abs(worldY.x), abs(worldY.y))), supportDepth, 0.0);
  }
  let center = xy[g];
  let sampleX0 = target_at(center - axisX, width, height);
  let sampleX1 = target_at(center + axisX, width, height);
  let sampleY0 = target_at(center - axisY, width, height);
  let sampleY1 = target_at(center + axisY, width, height);
  let sampleXFar0 = target_at(center - axisX * 4.0, width, height);
  let sampleXFar1 = target_at(center + axisX * 4.0, width, height);
  let sampleYFar0 = target_at(center - axisY * 4.0, width, height);
  let sampleYFar1 = target_at(center + axisY * 4.0, width, height);
  let mismatchX = max(
    dot(abs(sampleX0 - sourceColor.rgb), vec3<f32>(0.3333333333)),
    dot(abs(sampleX1 - sourceColor.rgb), vec3<f32>(0.3333333333))
  );
  let mismatchY = max(
    dot(abs(sampleY0 - sourceColor.rgb), vec3<f32>(0.3333333333)),
    dot(abs(sampleY1 - sourceColor.rgb), vec3<f32>(0.3333333333))
  );
  let mismatchFar = max(
    max(
      dot(abs(sampleXFar0 - sourceColor.rgb), vec3<f32>(0.3333333333)),
      dot(abs(sampleXFar1 - sourceColor.rgb), vec3<f32>(0.3333333333))
    ),
    max(
      dot(abs(sampleYFar0 - sourceColor.rgb), vec3<f32>(0.3333333333)),
      dot(abs(sampleYFar1 - sourceColor.rgb), vec3<f32>(0.3333333333))
    )
  );
  let colorMismatch = max(max(mismatchX, mismatchY), mismatchFar);
  let colorThreshold = max(0.000001, config[40]);
  let risk = sourceColor.a
    * max(0.0, supportDepth / depthThreshold - 1.0)
    * max(0.0, colorMismatch / colorThreshold - 1.0);
  let useX = select(
    abs(worldX.y) >= abs(worldY.y),
    abs(worldX.x) >= abs(worldY.x),
    yawDepth >= pitchDepth
  );
  return vec4<f32>(min(risk, 64.0), select(0.0, 1.0, useX), supportDepth, colorMismatch);
}

fn target_luma_pixel(px: i32, py: i32, width: u32, height: u32) -> f32 {
  let safeX = u32(clamp(px, 0, i32(width) - 1));
  let safeY = u32(clamp(py, 0, i32(height) - 1));
  let index = (safeY * width + safeX) * 3u;
  let rgb = vec3<f32>(targetRgb[index], targetRgb[index + 1u], targetRgb[index + 2u]);
  return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

fn structure_at(pos: vec2<f32>, width: u32, height: u32) -> vec4<f32> {
  let px = i32(round((clamp(pos.x, -1.0, 1.0) * 0.5 + 0.5) * f32(width - 1u)));
  let py = i32(round((clamp(pos.y, -1.0, 1.0) * 0.5 + 0.5) * f32(height - 1u)));
  var jxx = 0.0;
  var jxy = 0.0;
  var jyy = 0.0;
  for (var oy = -1; oy <= 1; oy += 1) {
    for (var ox = -1; ox <= 1; ox += 1) {
      let gx = 0.5 * (target_luma_pixel(px + ox + 1, py + oy, width, height) - target_luma_pixel(px + ox - 1, py + oy, width, height));
      let gy = 0.5 * (target_luma_pixel(px + ox, py + oy + 1, width, height) - target_luma_pixel(px + ox, py + oy - 1, width, height));
      jxx += gx * gx;
      jxy += gx * gy;
      jyy += gy * gy;
    }
  }
  jxx /= 9.0;
  jxy /= 9.0;
  jyy /= 9.0;
  let trace = jxx + jyy;
  let separation = sqrt(max(0.0, (jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy));
  let coherence = separation / max(trace, 0.00000001);
  let normalAngle = 0.5 * atan2(2.0 * jxy, jxx - jyy);
  let tangentAngle = normalAngle + 1.57079632679;
  return vec4<f32>(atan2(sin(tangentAngle), cos(tangentAngle)), coherence, trace, 0.0);
}

fn phase45_structure_energy(pos: vec2<f32>, radius: i32, width: u32, height: u32) -> f32 {
  let px = i32(round((clamp(pos.x, -1.0, 1.0) * 0.5 + 0.5) * f32(width - 1u)));
  let py = i32(round((clamp(pos.y, -1.0, 1.0) * 0.5 + 0.5) * f32(height - 1u)));
  let gx = 0.5 * (target_luma_pixel(px + radius, py, width, height) - target_luma_pixel(px - radius, py, width, height));
  let gy = 0.5 * (target_luma_pixel(px, py + radius, width, height) - target_luma_pixel(px, py - radius, width, height));
  return (gx * gx + gy * gy) / f32(radius * radius);
}

fn phase45_multiscale_energy(pos: vec2<f32>, width: u32, height: u32) -> f32 {
  let fine = phase45_structure_energy(pos, 1, width, height);
  let medium = phase45_structure_energy(pos, 2, width, height);
  let coarse = phase45_structure_energy(pos, 4, width, height);
  return max(fine, max(medium, coarse));
}

fn phase45_sample_energy(g: u32, width: u32, height: u32) -> vec2<f32> {
  let t = transform[g];
  let useX = t.x >= t.y;
  let majorAngle = t.z + select(1.57079632679, 0.0, useX);
  let minorAngle = majorAngle + 1.57079632679;
  let footprintMajor = vec2<f32>(cos(majorAngle), sin(majorAngle)) * max(t.x, t.y) * ${BOUNDARY_SIGMA};
  let footprintMinor = vec2<f32>(cos(minorAngle), sin(minorAngle)) * min(t.x, t.y) * ${BOUNDARY_SIGMA};
  let fixedPixel = vec2<f32>(8.0 / max(1.0, f32(width - 1u)), 8.0 / max(1.0, f32(height - 1u)));
  let center = xy[g];
  var sum = phase45_multiscale_energy(center, width, height);
  var maximum = sum;
  var sampleCount = 1.0;
  for (var gy = -4; gy <= 4; gy += 1) {
    for (var gx = -4; gx <= 4; gx += 1) {
      if (gx == 0 && gy == 0) { continue; }
      let offset = footprintMajor * (f32(gx) * 0.25) + footprintMinor * (f32(gy) * 0.25);
      let sample = phase45_multiscale_energy(constrain_xy(center + offset), width, height);
      sum += sample;
      maximum = max(maximum, sample);
      sampleCount += 1.0;
    }
  }
  let fixedOffsets = array<vec2<f32>, 8>(
    vec2<f32>(fixedPixel.x, 0.0), vec2<f32>(-fixedPixel.x, 0.0),
    vec2<f32>(0.0, fixedPixel.y), vec2<f32>(0.0, -fixedPixel.y),
    fixedPixel, -fixedPixel, vec2<f32>(fixedPixel.x, -fixedPixel.y), vec2<f32>(-fixedPixel.x, fixedPixel.y)
  );
  for (var i = 0u; i < 8u; i += 1u) {
    let sample = phase45_multiscale_energy(constrain_xy(center + fixedOffsets[i]), width, height);
    sum += sample;
    maximum = max(maximum, sample);
    sampleCount += 1.0;
  }
  return vec2<f32>(sum / sampleCount, maximum);
}

fn phase45_encode_energy(value: f32) -> u32 { return u32(round(clamp(value, 0.0, 0.0625) * PHASE45_ENERGY_QUANTIZATION)); }
fn phase45_encode_residual(value: f32) -> u32 { return u32(round(clamp(value, 0.0, 1.0) * PHASE45_RESIDUAL_QUANTIZATION)); }
fn phase45_encode_normalized(value: f32) -> u32 { return u32(round(clamp(value, 0.0, 16.0) * PHASE45_NORMALIZED_QUANTIZATION)); }

fn phase45_utility_bin(energyMaximum: f32, residual: f32, influence: f32) -> u32 {
  let utility = max(clamp(energyMaximum / 0.0625, 0.0, 1.0), max(clamp(residual / 0.02, 0.0, 1.0), clamp(influence, 0.0, 1.0)));
  return min(7u, u32(floor(utility * 8.0)));
}

fn constrain_xy(pos: vec2<f32>) -> vec2<f32> {
  let margin = max(config[54], 0.0) * ${MIN_SPLAT_SCALE};
  return min(max(pos, vec2<f32>(-1.0 + margin)), vec2<f32>(1.0 - margin));
}

fn cap_anisotropy(scale: vec2<f32>, maxAnisotropy: f32) -> vec2<f32> {
  let minor = max(0.0015, min(scale.x, scale.y));
  let major = max(scale.x, scale.y);
  if (major / minor <= maxAnisotropy) { return max(scale, vec2<f32>(0.0015)); }
  let capped = minor * maxAnisotropy;
  return select(vec2<f32>(minor, capped), vec2<f32>(capped, minor), scale.x >= scale.y);
}

fn rotated_extent(scale: vec2<f32>, theta: f32) -> vec2<f32> {
  let c = abs(cos(theta));
  let s = abs(sin(theta));
  return max(config[54], 0.0) * vec2<f32>(
    length(vec2<f32>(c * scale.x, s * scale.y)),
    length(vec2<f32>(s * scale.x, c * scale.y))
  );
}

fn constrain_scale(pos: vec2<f32>, scale: vec2<f32>, theta: f32, maxAnisotropy: f32) -> vec2<f32> {
  let capped = cap_anisotropy(
    min(max(vec2<f32>(${MIN_SPLAT_SCALE}), scale), vec2<f32>(max(config[56], ${MIN_SPLAT_SCALE}))),
    maxAnisotropy
  );
  if (config[54] <= 0.0) { return capped; }
  let extent = rotated_extent(capped, theta);
  let available = max(vec2<f32>(0.0), vec2<f32>(1.0) - abs(pos));
  let fit = min(1.0, min(available.x / max(extent.x, 0.00000001), available.y / max(extent.y, 0.00000001)));
  var fitted = cap_anisotropy(max(vec2<f32>(${MIN_SPLAT_SCALE}), capped * fit), maxAnisotropy);
  let fittedExtent = rotated_extent(fitted, theta);
  let globalFit = min(1.0, 0.999 / max(fittedExtent.x, fittedExtent.y));
  fitted = max(vec2<f32>(${MIN_SPLAT_SCALE}), fitted * globalFit);
  return fitted;
}

fn constrain_position(pos: vec2<f32>, scale: vec2<f32>, theta: f32) -> vec2<f32> {
  if (config[54] <= 0.0) { return clamp(pos, vec2<f32>(-1.0), vec2<f32>(1.0)); }
  let extent = rotated_extent(scale, theta);
  return clamp(pos, vec2<f32>(-1.0) + extent, vec2<f32>(1.0) - extent);
}

fn try_claim_role(index: u32, role: u32) -> bool {
  for (var attempt = 0u; attempt < 4u; attempt += 1u) {
    let claim = atomicCompareExchangeWeak(&control[index], 0u, role);
    if (claim.exchanged) { return true; }
    if (claim.old_value != 0u) { return false; }
  }
  return false;
}

fn rollback_role(index: u32, role: u32) {
  loop {
    let rollback = atomicCompareExchangeWeak(&control[index], role, 0u);
    if (rollback.exchanged || rollback.old_value != role) { return; }
  }
}

fn is_adc_step(step: u32) -> bool {
  let interval = max(1u, u32(config[12]));
  let window = u32(config[13]);
  let warmup = u32(config[14]);
  let densityHorizon = u32(config[15]);
  if (step <= warmup || step > densityHorizon) { return false; }
  let phase = step % interval;
  return phase == 0u || phase >= interval - min(window, interval);
}

fn normalized_stats(g: u32) -> vec3<f32> {
  let st = stats[g];
  return vec3<f32>(st.x / max(st.w, 1.0), st.y, st.z);
}

fn importance_at(g: u32) -> vec4<f32> {
  return stats[u32(config[10]) + g];
}

fn importance_score(g: u32) -> f32 {
  let im = importance_at(g);
  let pixelCount = max(1.0, config[0] * config[1]);
  let count = max(1.0, config[2]);
  let expectedInfluence = pixelCount / count;
  let relativeInfluence = clamp(im.y / max(expectedInfluence, 0.000001), 0.0, 4.0);
  let coverageFactor = sqrt(clamp(im.x / 16.0, 0.0, 4.0));
  let t = transform[g];
  let areaPixels = 3.14159265 * 6.25 * max(0.00000001, t.x * t.y) * max(1.0, (config[0] - 1.0) * (config[1] - 1.0) * 0.25);
  let areaFactor = sqrt(clamp(areaPixels / 64.0, 0.0, 4.0));
  return relativeInfluence * 0.65 + coverageFactor * 0.2 + areaFactor * 0.15;
}

fn importance_residual(g: u32) -> f32 {
  let im = importance_at(g);
  return im.z / max(im.x, 1.0);
}

fn distribution_weight(g: u32, adc: bool) -> f32 {
  let t = transform[g];
  let c = color[g];
  if (t.w < 0.5 || c.a < 0.005) { return 0.0; }
  let tiltProfile = tilt_split_profile(g, u32(config[0]), u32(config[1]));
  let signal = normalized_stats(g);
  let areaMass = c.a * sqrt(max(0.00000001, t.x * t.y));
  let coverageRaw = clamp(sqrt(max(importance_at(g).x, 1.0) / 16.0), 0.5, 3.0);
  let coverageGain = select(1.0, mix(1.0, coverageRaw, clamp(config[23], 0.0, 1.0)), config[18] > 0.5);
  let residualSupport = max(signal.y, importance_residual(g));
  var densityResidual = residualSupport;
  var densityGradient = signal.x;
  if (u32(config[11]) == 1u) {
    let growthSignal = densityGradient + densityResidual * 0.5;
    let adcSplit = is_adc_step(u32(config[4]));
    let signalThreshold = select(config[34], config[48], adcSplit);
    let residualThreshold = select(0.0, config[49], adcSplit);
    var growthEligible = growthSignal > signalThreshold && densityResidual >= residualThreshold;
    if (config[17] > 0.5) {
      growthEligible = growthEligible && importance_at(g).x >= 4.0 && densityResidual >= 0.01;
    }
    if (!growthEligible && tiltProfile.x <= 0.0) { return 0.0; }
  }
  if (config[26] > 0.5) {
    let localStructure = structure_at(xy[g], u32(config[0]), u32(config[1]));
    let edgeScore = clamp(sqrt(max(0.0, localStructure.z) / 0.0004), 0.0, 4.0);
    densityResidual *= 1.0 + 0.25 * edgeScore;
  }
  if (adc && config[25] > 0.5) {
    let denominator = importance_at(g).w;
    let coherence = select(1.0, clamp(stats[g].x / denominator, 0.0, 1.0), denominator > 0.00000001);
    let directionWeight = 0.8 + 25.0 * pow(1.0 - coherence, 15.0);
    let projectedMajor = max(t.x, t.y) * max(config[0], config[1]) * 0.5 * ${BOUNDARY_SIGMA};
    densityGradient *= select(1.0 / directionWeight, directionWeight, projectedMajor > 3.0);
  }
  let base = areaMass * 0.15 + signal.z * 0.2 + densityResidual * 0.3 + densityGradient * coverageGain * 0.35;
  let adcBoost = select(0.0, densityGradient * coverageGain * 0.45 + densityResidual * 0.35, adc);
  var combined = base + adcBoost;
  if (tiltProfile.x > 0.0) {
    combined += min(8.0, tiltProfile.x) * (0.25 + areaMass);
  }
  if (adc && config[47] > 0.0) {
    let capacity = u32(config[10]);
    let uv = clamp(xy[g] * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(0.999999));
    let cell = min(vec2<u32>(PHASE45_REGION_GRID - 1u), vec2<u32>(uv * f32(PHASE45_REGION_GRID)));
    let region = cell.y * PHASE45_REGION_GRID + cell.x;
    let regionBase = phase45_region_base(capacity) + region * PHASE45_REGION_STRIDE;
    let regionCount = f32(atomicLoad(&control[regionBase]));
    let quota = f32(atomicLoad(&control[regionBase + 9u]));
    let meanEnergy = f32(atomicLoad(&control[regionBase + 1u])) / (PHASE45_ENERGY_QUANTIZATION * max(1.0, regionCount));
    let maxEnergy = f32(atomicLoad(&control[regionBase + 2u])) / PHASE45_ENERGY_QUANTIZATION;
    let meanResidual = f32(atomicLoad(&control[regionBase + 3u])) / (PHASE45_RESIDUAL_QUANTIZATION * max(1.0, regionCount));
    let deficit = clamp((quota - regionCount) / max(1.0, quota), 0.0, 1.0);
    let recipientDemand = clamp(max(meanEnergy * 8.0, maxEnergy * 4.0) + meanResidual * 4.0, 0.0, 1.0);
    combined *= 1.0 + clamp(config[47], 0.0, 1.0) * deficit * recipientDemand;
  }
  if (config[32] > 0.5) {
    let areaPixels = 3.14159265 * 6.25 * max(0.00000001, t.x * t.y) * max(1.0, (config[0] - 1.0) * (config[1] - 1.0) * 0.25);
    let footprintCost = sqrt(max(1.0, areaPixels / 64.0));
    combined /= mix(1.0, footprintCost, 0.35);
  }
  if (combined != combined || abs(combined) > 100000000000000000000.0) { return combined; }
  return max(0.00000001, combined);
}

fn cdf_base(capacity: u32) -> u32 { return capacity * 2u + EVENT_SLOTS; }
fn cdf_max_blocks(capacity: u32) -> u32 { return (capacity + CDF_BLOCK_SIZE - 1u) / CDF_BLOCK_SIZE; }
fn cdf_block_sum_base(capacity: u32) -> u32 { return cdf_base(capacity) + capacity + 1u; }
fn cdf_block_offset_base(capacity: u32) -> u32 { return cdf_block_sum_base(capacity) + cdf_max_blocks(capacity); }
fn phase45_region_base(capacity: u32) -> u32 { return cdf_block_offset_base(capacity) + cdf_max_blocks(capacity); }
fn phase45_donor_base(capacity: u32) -> u32 { return phase45_region_base(capacity) + PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE; }

@compute @workgroup_size(64)
fn phase45_collect_region_telemetry(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  if (u32(config[11]) != 3u || config[44] <= 0.5 || g >= count) { return; }
  let width = u32(config[0]);
  let height = u32(config[1]);
  let uv = clamp(xy[g] * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(0.999999));
  let cell = min(vec2<u32>(PHASE45_REGION_GRID - 1u), vec2<u32>(uv * f32(PHASE45_REGION_GRID)));
  let region = cell.y * PHASE45_REGION_GRID + cell.x;
  let base = phase45_region_base(capacity) + region * PHASE45_REGION_STRIDE;
  let energy = phase45_sample_energy(g, width, height);
  let im = importance_at(g);
  let expectedInfluence = max(1.0, config[0] * config[1] / max(1.0, config[2]));
  let residual = clamp(im.z / max(im.x, 1.0), 0.0, 1.0);
  let influence = clamp(im.y / expectedInfluence, 0.0, 16.0);
  let support = clamp(im.x / 16.0, 0.0, 16.0);
  let locallyProtected = energy.y >= 0.0004;
  let lowUtility = residual < 0.02 && influence < 0.75 && support < 0.75 && energy.x < 0.0004 && !locallyProtected;
  let utilityBin = phase45_utility_bin(energy.y, residual, influence);
  atomicAdd(&control[base], 1u);
  atomicAdd(&control[base + 1u], phase45_encode_energy(energy.x));
  atomicMax(&control[base + 2u], phase45_encode_energy(energy.y));
  atomicAdd(&control[base + 3u], phase45_encode_residual(residual));
  atomicMax(&control[base + 4u], phase45_encode_residual(residual));
  atomicAdd(&control[base + 5u], phase45_encode_normalized(influence));
  atomicMax(&control[base + 6u], phase45_encode_normalized(influence));
  atomicAdd(&control[base + 7u], phase45_encode_normalized(support));
  atomicMax(&control[base + 8u], phase45_encode_normalized(support));
  if (lowUtility) { atomicAdd(&control[base + 10u], 1u); }
  if (locallyProtected) { atomicAdd(&control[base + 11u], 1u); }
  atomicAdd(&control[base + 12u + utilityBin], 1u);
}

@compute @workgroup_size(64)
fn phase45_finalize_region_telemetry(@builtin(local_invocation_id) localId: vec3u) {
  let region = localId.x;
  let capacity = u32(config[10]);
  let base = phase45_region_base(capacity) + region * PHASE45_REGION_STRIDE;
  let regionSplats = atomicLoad(&control[base]);
  let meanEnergy = f32(atomicLoad(&control[base + 1u])) / (PHASE45_ENERGY_QUANTIZATION * max(1.0, f32(regionSplats)));
  let maxEnergy = f32(atomicLoad(&control[base + 2u])) / PHASE45_ENERGY_QUANTIZATION;
  let meanResidual = f32(atomicLoad(&control[base + 3u])) / (PHASE45_RESIDUAL_QUANTIZATION * max(1.0, f32(regionSplats)));
  let meanSupport = f32(atomicLoad(&control[base + 7u])) / (PHASE45_NORMALIZED_QUANTIZATION * max(1.0, f32(regionSplats)));
  // The max term is an explicit guard for small high-frequency islands in a smooth region.
  let multiscaleMaxGuard = max(meanEnergy, maxEnergy * 0.5);
  let demand = max(1u, u32(round((0.0625 + multiscaleMaxGuard * 8.0 + meanResidual * 4.0 + sqrt(meanSupport)) * 4096.0)));
  phase45Demand[region] = demand;
  let donorTarget = max(1u, u32(ceil(f32(regionSplats) * clamp(config[46], 0.0, 1.0))));
  var donorCumulative = 0u;
  var donorCutoff = 7u;
  var donorCutoffFraction = 1.0;
  var donorCutoffFound = false;
  for (var bin = 0u; bin < 8u; bin += 1u) {
    let binCount = atomicLoad(&control[base + 12u + bin]);
    if (!donorCutoffFound && donorCumulative + binCount >= donorTarget) {
      donorCutoff = bin;
      donorCutoffFraction = clamp(f32(donorTarget - donorCumulative) / max(1.0, f32(binCount)), 0.0, 1.0);
      donorCutoffFound = true;
    }
    donorCumulative += binCount;
  }
  let packedCutoff = min(7u, donorCutoff) | (u32(round(donorCutoffFraction * 65535.0)) << 8u);
  atomicStore(&control[base + 20u], packedCutoff);
  workgroupBarrier();
  if (region != 0u) { return; }
  var totalDemand = 0u;
  for (var i = 0u; i < PHASE45_REGION_COUNT; i += 1u) { totalDemand += phase45Demand[i]; }
  var allocated = 0u;
  for (var i = 0u; i < PHASE45_REGION_COUNT; i += 1u) {
    let quota = u32(floor(f32(u32(config[2])) * f32(phase45Demand[i]) / max(1.0, f32(totalDemand))));
    atomicStore(&control[phase45_region_base(capacity) + i * PHASE45_REGION_STRIDE + 9u], quota);
    allocated += quota;
  }
  let remainder = u32(config[2]) - allocated;
  for (var extra = 0u; extra < remainder; extra += 1u) {
    var bestRegion = 0u;
    var bestFraction = -1.0;
    for (var i = 0u; i < PHASE45_REGION_COUNT; i += 1u) {
      let rawQuota = f32(u32(config[2])) * f32(phase45Demand[i]) / max(1.0, f32(totalDemand));
      let fraction = rawQuota - floor(rawQuota);
      if (fraction > bestFraction) {
        bestFraction = fraction;
        bestRegion = i;
      }
    }
    atomicAdd(&control[phase45_region_base(capacity) + bestRegion * PHASE45_REGION_STRIDE + 9u], 1u);
    phase45Demand[bestRegion] = 0u;
  }
}

fn phase45_pixel_state_at(pos: vec2<f32>, width: u32, height: u32) -> vec4<f32> {
  let safe = clamp(pos, vec2<f32>(-1.0), vec2<f32>(1.0));
  let px = min(width - 1u, u32(round((safe.x * 0.5 + 0.5) * f32(width - 1u))));
  let py = min(height - 1u, u32(round((safe.y * 0.5 + 0.5) * f32(height - 1u))));
  let base = (py * width + px) * 4u;
  return vec4<f32>(errorMap[base], errorMap[base + 1u], errorMap[base + 2u], errorMap[base + 3u]);
}

fn phase45_gaussian_kernel(d: vec2<f32>, theta: f32, scale: vec2<f32>) -> f32 {
  let c = cos(theta);
  let s = sin(theta);
  let rotated = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  let q = dot(rotated / scale, rotated / scale);
  return exp(-0.5 * q);
}

fn phase45_donor_weight(g: u32, pos: vec2<f32>, width: u32, height: u32) -> f32 {
  let t = transform[g];
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  let ox = select(0.0, 0.5 / f32(width - 1u), width > 1u);
  let oy = select(0.0, 0.5 / f32(height - 1u), height > 1u);
  let center = xy[g];
  let kernel = 0.25 * (
    phase45_gaussian_kernel(clamp(pos + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, t.z, baseScale) +
    phase45_gaussian_kernel(clamp(pos + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, t.z, baseScale) +
    phase45_gaussian_kernel(clamp(pos + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, t.z, baseScale) +
    phase45_gaussian_kernel(clamp(pos + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, t.z, baseScale)
  );
  return kernel * color[g].a;
}

@compute @workgroup_size(64)
fn phase45_evaluate_donor_safety(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  if (g >= count || config[45] <= 0.5) { return; }
  let width = u32(config[0]);
  let height = u32(config[1]);
  let uv = clamp(xy[g] * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(0.999999));
  let cell = min(vec2<u32>(PHASE45_REGION_GRID - 1u), vec2<u32>(uv * f32(PHASE45_REGION_GRID)));
  let region = cell.y * PHASE45_REGION_GRID + cell.x;
  let regionBase = phase45_region_base(capacity) + region * PHASE45_REGION_STRIDE;
  let energy = phase45_sample_energy(g, width, height);
  let im = importance_at(g);
  let expectedInfluence = max(1.0, config[0] * config[1] / max(1.0, config[2]));
  let residual = clamp(im.z / max(im.x, 1.0), 0.0, 1.0);
  let influence = clamp(im.y / expectedInfluence, 0.0, 16.0);
  let utilityBin = phase45_utility_bin(energy.y, residual, influence);
  let packedCutoff = atomicLoad(&control[regionBase + 20u]);
  let cutoffBin = packedCutoff & 7u;
  let cutoffFraction = f32(packedCutoff >> 8u) / 65535.0;
  let regionSurplus = atomicLoad(&control[regionBase]) > atomicLoad(&control[regionBase + 9u]);
  let localDetailSafe = energy.y < 0.0004;
  let cutoffSample = hash_unit(f32(g) * 29.17 + config[4] * 0.019);
  let lowQuantile = utilityBin < cutoffBin || (utilityBin == cutoffBin && cutoffSample < cutoffFraction);
  let t = transform[g];
  let c = cos(t.z);
  let s = sin(t.z);
  let axisX = vec2<f32>(c, s) * t.x;
  let axisY = vec2<f32>(-s, c) * t.y;
  let center = xy[g];
  let samples = array<vec2<f32>, 9>(
    center,
    center + axisX, center - axisX, center + axisY, center - axisY,
    center + (axisX + axisY) * 0.70710678, center + (axisX - axisY) * 0.70710678,
    center + (-axisX + axisY) * 0.70710678, center - (axisX + axisY) * 0.70710678
  );
  var supportSafe = true;
  var nonfinite = false;
  for (var i = 0u; i < 9u; i += 1u) {
    let pos = clamp(samples[i], vec2<f32>(-1.0), vec2<f32>(1.0));
    let state = phase45_pixel_state_at(pos, width, height);
    let weight = phase45_donor_weight(g, pos, width, height);
    if (state.a != state.a || weight != weight || state.a < 0.0 || weight < 0.0) {
      nonfinite = true;
      supportSafe = false;
      continue;
    }
    if (weight <= 0.000001) { continue; }
    let remaining = state.a - weight;
    let share = weight / max(state.a, 0.00000001);
    if (remaining <= max(0.01, state.a * 0.10) || share >= 0.80) {
      supportSafe = false;
      continue;
    }
    let targetColor = target_at(pos, width, height);
    let currentError = dot(abs(state.rgb - targetColor), vec3<f32>(1.0 / 3.0));
    let without = (state.rgb * state.a - color[g].rgb * weight) / remaining;
    let removalRisk = dot(abs(without - targetColor), vec3<f32>(1.0 / 3.0)) - currentError;
    if (removalRisk > 0.003 || without.x != without.x || without.y != without.y || without.z != without.z) {
      supportSafe = false;
    }
  }
  var flags = (region << 8u) | (utilityBin << 16u);
  if (supportSafe) { flags |= 2u; atomicAdd(&control[regionBase + 21u], 1u); }
  if (!localDetailSafe) { flags |= 4u; }
  if (regionSurplus) { flags |= 8u; }
  if (lowQuantile) { flags |= 16u; }
  if (nonfinite) { flags |= 32u; }
  let eligible = supportSafe && localDetailSafe && regionSurplus && lowQuantile && !nonfinite;
  if (eligible) { flags |= PHASE45_DONOR_ELIGIBLE; atomicAdd(&control[regionBase + 22u], 1u); }
  atomicStore(&control[phase45_donor_base(capacity) + g], flags);
}

@compute @workgroup_size(256)
fn build_distribution(@builtin(global_invocation_id) id: vec3u) {
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  let g = id.x;
  if (g >= count) { return; }
  let adc = is_adc_step(u32(config[4])) || u32(config[11]) == 3u;
  var weight = distribution_weight(g, adc);
  if (weight != weight || abs(weight) > 100000000000000000000.0) {
    atomicAdd(&control[capacity * 2u + 12u], 1u);
    weight = 0.0;
  }
  if (weight > 0.0 && config[43] > 0.5) {
    atomicAdd(&control[capacity * 2u + 18u], 1u);
  }
  atomicStore(&control[cdf_base(capacity) + g], bitcast<u32>(weight));
}

@compute @workgroup_size(256)
fn scan_distribution_blocks(
  @builtin(local_invocation_id) localId: vec3u,
  @builtin(workgroup_id) groupId: vec3u,
) {
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  let g = groupId.x * CDF_BLOCK_SIZE + localId.x;
  var weight = 0.0;
  if (g < count) { weight = bitcast<f32>(atomicLoad(&control[cdf_base(capacity) + g])); }
  cdfScratch[localId.x] = weight;
  workgroupBarrier();
  for (var offset = 1u; offset < CDF_BLOCK_SIZE; offset = offset * 2u) {
    var addend = 0.0;
    if (localId.x >= offset) { addend = cdfScratch[localId.x - offset]; }
    workgroupBarrier();
    cdfScratch[localId.x] += addend;
    workgroupBarrier();
  }
  if (g < count) { atomicStore(&control[cdf_base(capacity) + g], bitcast<u32>(cdfScratch[localId.x])); }
  let blockLast = min(count, (groupId.x + 1u) * CDF_BLOCK_SIZE) - 1u;
  if (g == blockLast) {
    atomicStore(&control[cdf_block_sum_base(capacity) + groupId.x], bitcast<u32>(cdfScratch[localId.x]));
  }
}

@compute @workgroup_size(1)
fn scan_distribution_block_sums() {
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  let blocks = (count + CDF_BLOCK_SIZE - 1u) / CDF_BLOCK_SIZE;
  var prefix = 0.0;
  for (var block = 0u; block < blocks; block += 1u) {
    atomicStore(&control[cdf_block_offset_base(capacity) + block], bitcast<u32>(prefix));
    prefix += bitcast<f32>(atomicLoad(&control[cdf_block_sum_base(capacity) + block]));
  }
  atomicStore(&control[cdf_base(capacity) + capacity], bitcast<u32>(prefix));
}

@compute @workgroup_size(256)
fn add_distribution_block_offsets(@builtin(global_invocation_id) id: vec3u) {
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  let g = id.x;
  if (g >= count) { return; }
  let block = g / CDF_BLOCK_SIZE;
  let value = bitcast<f32>(atomicLoad(&control[cdf_base(capacity) + g]));
  let offset = bitcast<f32>(atomicLoad(&control[cdf_block_offset_base(capacity) + block]));
  atomicStore(&control[cdf_base(capacity) + g], bitcast<u32>(value + offset));
}

fn pick_source(seedIndex: u32, count: u32, adc: bool) -> u32 {
  let capacity = u32(config[10]);
  let cdfBase = capacity * 2u + EVENT_SLOTS;
  let total = bitcast<f32>(atomicLoad(&control[cdfBase + capacity]));
  if (total <= 0.00000001) { return seedIndex % max(count, 1u); }
  let step = config[4];
  let sample = hash_unit(f32(seedIndex) * 13.0 + step * 0.31 + select(17.0, 41.0, adc)) * total;
  var low = 0u;
  var high = count;
  loop {
    if (low >= high) { break; }
    let mid = low + (high - low) / 2u;
    let value = bitcast<f32>(atomicLoad(&control[cdfBase + mid]));
    if (value < sample) { low = mid + 1u; } else { high = mid; }
  }
  return min(count - 1u, low);
}

fn encode_selection(source: u32, mode: u32) -> u32 {
  return ((mode & 3u) << 30u) | ((source + 1u) & SOURCE_MASK);
}

fn pick_error_position(seedIndex: u32, width: u32, height: u32) -> vec2<f32> {
  let pixelCount = width * height;
  var bestPixel = 0u;
  var bestScore = -1.0;
  for (var n = 0u; n < 32u; n = n + 1u) {
    let u = hash_unit(f32(seedIndex) * 17.17 + f32(n) * 91.73 + config[4] * 0.37);
    let pixel = min(pixelCount - 1u, u32(u * f32(pixelCount)));
    let score = errorMap[pixel] + hash_unit(f32(pixel + seedIndex + n)) * 0.0001;
    if (score > bestScore) {
      bestPixel = pixel;
      bestScore = score;
    }
  }
  let px = bestPixel % width;
  let py = bestPixel / width;
  let x = select(0.0, (f32(px) / f32(width - 1u)) * 2.0 - 1.0, width > 1u);
  let y = select(0.0, (f32(py) / f32(height - 1u)) * 2.0 - 1.0, height > 1u);
  return constrain_xy(vec2<f32>(x, y));
}

fn error_at_position(pos: vec2<f32>, width: u32, height: u32) -> f32 {
  let px = min(width - 1u, u32(floor((pos.x * 0.5 + 0.5) * f32(width - 1u) + 0.5)));
  let py = min(height - 1u, u32(floor((pos.y * 0.5 + 0.5) * f32(height - 1u) + 0.5)));
  return errorMap[py * width + px];
}

fn grid_position(seedIndex: u32, targetCount: u32, cols: u32, rows: u32) -> vec2<f32> {
  let gridIndex = min(targetCount - 1u, u32(hash_unit(f32(seedIndex) * 0.754877666 + 19.19) * f32(targetCount)));
  let col = gridIndex % cols;
  let row = gridIndex / cols;
  let x = select(0.0, -0.95 + 1.9 * f32(col) / f32(cols - 1u), cols > 1u);
  let y = select(0.0, -0.95 + 1.9 * f32(row) / f32(rows - 1u), rows > 1u);
  return constrain_xy(vec2<f32>(x, y));
}

@compute @workgroup_size(64)
fn select_grow(@builtin(global_invocation_id) id: vec3u) {
  let local = id.x;
  let oldCount = u32(config[2]);
  let targetCount = u32(config[3]);
  let step = u32(config[4]);
  let steps = u32(config[5]);
  let baseScale = vec2<f32>(config[52], config[53]);
  let capacity = u32(config[10]);
  let index = oldCount + local;
  if (index >= targetCount) { return; }
  let adc = is_adc_step(step);
  let source = pick_source(index, oldCount, adc);
  let sourceSignal = normalized_stats(source);
  let sourceImportance = importance_at(source);
  let tiltProfile = tilt_split_profile(source, u32(config[0]), u32(config[1]));
  // Tilt-risk replacement is an ADC operation. Running it on every ordinary
  // growth event repeatedly duplicates broad splats and inflates tile work.
  let tiltRisk = adc && tiltProfile.x > 0.0;
  let residualSupport = max(sourceSignal.y, importance_residual(source));
  let major = max(transform[source].x, transform[source].y);
  let projectedMajor = major * max(config[0], config[1]) * 0.5 * ${BOUNDARY_SIGMA};
  let signalThreshold = select(config[34], config[48], adc);
  let residualThreshold = select(0.0, config[49], adc);
  let highSignal = sourceSignal.x + residualSupport * 0.5 > signalThreshold && residualSupport >= residualThreshold;
  var eligible = highSignal || tiltRisk;
  if (config[17] > 0.5) {
    eligible = (highSignal && sourceImportance.x >= 4.0 && residualSupport >= 0.01) || tiltRisk;
  }
  let mode = select(2u, 1u, projectedMajor > 3.0 || tiltRisk);
  var finalMode = select(0u, mode, eligible);
  let eventBase = capacity * 2u;
  if (config[42] > 0.5 && finalMode != 0u) {
    let token = (local + 1u) & ROLE_TOKEN_MASK;
    let claimValue = select(ROLE_SOURCE_OTHER, ROLE_SOURCE_SPLIT, finalMode == 1u) | token;
    if (try_claim_role(source, claimValue)) {
      atomicAdd(&control[eventBase + 17u], 1u);
      if (tiltRisk) { atomicAdd(&control[eventBase + 19u], 1u); }
    } else {
      atomicAdd(&control[eventBase + 16u], 1u);
      finalMode = 0u;
    }
  }
  atomicStore(&control[capacity + local], encode_selection(source, finalMode));
  if (config[42] <= 0.5) {
    if (tiltRisk && finalMode != 0u) { atomicAdd(&control[eventBase + 19u], 1u); }
    let occurrence = select(1u, 0x00010001u, finalMode != 0u);
    atomicAdd(&control[source], occurrence);
  }
  if (adc && config[17] > 0.5) {
    if (eligible) { atomicAdd(&control[eventBase + 9u], 1u); }
    else { atomicAdd(&control[eventBase + 10u], 1u); }
  }
}

@compute @workgroup_size(64)
fn redistribute_sources(@builtin(global_invocation_id) id: vec3u) {
  let source = id.x;
  let count = u32(config[2]);
  if (source >= count) { return; }
  let packed = atomicLoad(&control[source]);
  if (config[42] > 0.5) { return; }
  let copies = select(packed & 0xffffu, packed >> 16u, u32(config[11]) == 1u);
  if (copies == 0u) { return; }
  let c = color[source];
  let share = c.a / f32(copies + 1u);
  color[source] = vec4<f32>(c.rgb, max(0.005, share));
}

@compute @workgroup_size(64)
fn apply_grow(@builtin(global_invocation_id) id: vec3u) {
  let local = id.x;
  let width = u32(config[0]);
  let height = u32(config[1]);
  let oldCount = u32(config[2]);
  let targetCount = u32(config[3]);
  let step = u32(config[4]);
  let baseScale = vec2<f32>(config[52], config[53]);
  let maxAnisotropy = max(config[9], 1.0);
  let cols = u32(config[6]);
  let rows = u32(config[7]);
  let capacity = u32(config[10]);
  let index = oldCount + local;
  if (index >= targetCount) { return; }
  let encoded = atomicLoad(&control[capacity + local]);
  let mode = encoded >> 30u;
  let eventBase = capacity * 2u;

  // A claim conflict becomes a source-independent reseed. It must not read a
  // source that another workgroup may be replacing in this dispatch.
  if (mode == 0u) {
    let gridPos = grid_position(index, targetCount, cols, rows);
    let residualPos = pick_error_position(index, width, height);
    let materiallyWorse = error_at_position(residualPos, width, height) > error_at_position(gridPos, width, height) + 0.04;
    let residualPriority = config[17] > 0.5 || config[18] > 0.5;
    let useResidual = materiallyWorse && (residualPriority || hash_unit(f32(index) * 29.7 + f32(step) * 0.11) < 0.15);
    var nextPos = constrain_xy(select(gridPos, residualPos, useResidual));
    var nextScale = baseScale * 0.35;
    var nextTheta = 0.0;
    let localStructure = structure_at(nextPos, width, height);
    let localError = error_at_position(nextPos, width, height);
    let structureGuided = config[19] > 0.5 && localStructure.y > 0.45 && localStructure.z > 0.0004 && localError > 0.02;
    let adaptiveDetail = config[29] > 0.5 && localStructure.y >= config[31] && localStructure.z > 0.0004 && localError > 0.02;
    let surfaceMaxAnisotropy = max(config[55], 1.0);
    let localMaxAnisotropy = select(min(maxAnisotropy, surfaceMaxAnisotropy), maxAnisotropy, adaptiveDetail);
    if (structureGuided) {
      let areaRadius = sqrt(max(0.00000001, nextScale.x * nextScale.y));
      let ratio = min(localMaxAnisotropy, select(1.6, localMaxAnisotropy, config[28] > 0.5 || adaptiveDetail));
      nextScale = vec2<f32>(areaRadius * sqrt(ratio), areaRadius / sqrt(ratio));
      nextTheta = localStructure.x;
    }
    nextScale = constrain_scale(nextPos, max(min(nextScale, baseScale * 0.9), baseScale * 0.35), nextTheta, localMaxAnisotropy);
    nextPos = constrain_position(nextPos, nextScale, nextTheta);
    let reseedLayer = select(0.0, hash_unit(f32(index) * 0.61803398875) * ${LAYER_CODE_RANGE}, config[35] > 0.5);
    xy[index] = nextPos;
    transform[index] = vec4<f32>(nextScale, nextTheta, select(1.0, 2.0, adaptiveDetail) + reseedLayer);
    color[index] = vec4<f32>(target_at(nextPos, width, height), 0.005);
    stats[index] = vec4<f32>(0.0);
    stats[capacity + index] = vec4<f32>(0.0);
    atomicAdd(&control[eventBase + 2u], 1u);
    if (structureGuided) { atomicAdd(&control[eventBase + 11u], 1u); }
    return;
  }

  let source = (encoded & SOURCE_MASK) - 1u;
  let sourceT = transform[source];
  let sourceC = color[source];
  let sourceStats = stats[source];
  let sourceImportance = importance_at(source);
  let useX = sourceT.x >= sourceT.y;
  let tiltProfile = tilt_split_profile(source, width, height);
  var tiltTrueSplit = mode == 1u && tiltProfile.x > 0.0 && config[42] > 0.5;
  let splitUseX = select(useX, tiltProfile.y > 0.5, tiltTrueSplit);
  let sourceLongAngle = sourceT.z + select(1.57079632679, 0.0, splitUseX);
  let axis = vec2<f32>(cos(sourceLongAngle), sin(sourceLongAngle));
  let perp = vec2<f32>(-axis.y, axis.x);
  let side = select(-1.0, 1.0, hash_unit(f32(index) * 53.0 + f32(step) * 1.7) > 0.5);
  let major = select(sourceT.y, sourceT.x, splitUseX);
  let minor = max(0.0015, min(sourceT.x, sourceT.y));
  let jitter = (hash_unit(f32(index) * 71.0 + f32(step) * 2.3) - 0.5) * minor * 0.35;
  var nextPos = xy[source] + axis * major * 0.48 * side + perp * jitter;
  var nextScale = sourceT.xy * 0.98;
  if (mode == 1u) {
    let splitOffset = select(0.55, 0.34, tiltTrueSplit);
    nextPos = xy[source] + axis * major * splitOffset * select(side, 1.0, tiltTrueSplit);
    let splitShrink = select(0.72, clamp(config[41], 0.5, 0.85), tiltTrueSplit);
    let axisShrink = sourceT.xy * vec2<f32>(select(0.94, splitShrink, splitUseX), select(splitShrink, 0.94, splitUseX));
    nextScale = select(axisShrink, sourceT.xy * splitShrink, tiltTrueSplit);
  } else if (mode == 2u) {
    nextPos = xy[source] + axis * major * 0.24 * side + perp * jitter;
    nextScale = sourceT.xy * 0.96;
  }
  nextPos = constrain_xy(nextPos);
  var nextTheta = sourceT.z;
  let localStructure = structure_at(nextPos, width, height);
  let localError = error_at_position(nextPos, width, height);
  let structureGuided = !tiltTrueSplit && config[19] > 0.5 && localStructure.y > 0.45 && localStructure.z > 0.0004 && localError > 0.02;
  let adaptiveDetail = config[29] > 0.5 && localStructure.y >= config[31] && localStructure.z > 0.0004 && localError > 0.02;
  let inheritedDetail = sourceT.w > 1.5;
  let detailTagged = adaptiveDetail || inheritedDetail;
  let surfaceMaxAnisotropy = max(config[55], 1.0);
  let localMaxAnisotropy = select(min(maxAnisotropy, surfaceMaxAnisotropy), maxAnisotropy, detailTagged);
  if (structureGuided) {
    let areaRadius = sqrt(max(0.00000001, nextScale.x * nextScale.y));
    let ratio = min(localMaxAnisotropy, select(1.6, localMaxAnisotropy, config[28] > 0.5 || adaptiveDetail));
    nextScale = vec2<f32>(areaRadius * sqrt(ratio), areaRadius / sqrt(ratio));
    nextTheta = localStructure.x;
  }
  if (!tiltTrueSplit) { nextScale = min(nextScale, baseScale * 0.9); }
  let scaleFloor = select(baseScale * 0.35, vec2<f32>(${MIN_SPLAT_SCALE}), tiltTrueSplit);
  nextScale = constrain_scale(nextPos, max(nextScale, scaleFloor), nextTheta, localMaxAnisotropy);
  nextPos = constrain_position(nextPos, nextScale, nextTheta);

  var replacementSourcePos = xy[source];
  var replacementSourceScale = sourceT.xy;
  if (tiltTrueSplit) {
    replacementSourcePos = constrain_xy(xy[source] - axis * major * 0.34);
    replacementSourceScale = constrain_scale(
      replacementSourcePos,
      max(nextScale, vec2<f32>(${MIN_SPLAT_SCALE})),
      sourceT.z,
      localMaxAnisotropy
    );
    replacementSourcePos = constrain_position(replacementSourcePos, replacementSourceScale, sourceT.z);
  }

  let massShare = sourceC.a * max(0.00000001, sourceT.x * sourceT.y);
  var childOpacity = min(0.99, max(0.005, massShare / max(0.00000001, nextScale.x * nextScale.y)));
  if (tiltTrueSplit) {
    let replacementArea = nextScale.x * nextScale.y + replacementSourceScale.x * replacementSourceScale.y;
    let replacementOpacity = massShare / max(0.00000001, replacementArea);
    // Opaque parents commonly need a value just above 0.99 after shrinking.
    // Rejecting that case disabled every true split. Keep the symmetric split,
    // clamp the standard per-splat alpha, and report the small mass shortfall.
    if (replacementOpacity > 0.99) { atomicAdd(&control[eventBase + 21u], 1u); }
    childOpacity = clamp(replacementOpacity, 0.005, 0.99);
  }
  let targetColor = target_at(nextPos, width, height);
  let childColor = select(sourceC.rgb * 0.25 + targetColor * 0.75, sourceC.rgb, tiltTrueSplit);
  xy[index] = nextPos;
  let inheritedLayer = fract(sourceT.w);
  transform[index] = vec4<f32>(nextScale, nextTheta, select(1.0, 2.0, detailTagged) + inheritedLayer);
  color[index] = vec4<f32>(childColor, childOpacity);
  stats[index] = select(sourceStats, sourceStats * 0.5, tiltTrueSplit);
  var childImportance = sourceImportance * 0.5;
  if (config[25] > 0.5 && !tiltTrueSplit) { childImportance.w = sourceImportance.w; }
  stats[capacity + index] = childImportance;
  if (tiltTrueSplit) {
    xy[source] = replacementSourcePos;
    transform[source] = vec4<f32>(replacementSourceScale, sourceT.z, sourceT.w);
    color[source] = vec4<f32>(sourceC.rgb, childOpacity);
    stats[source] = sourceStats * 0.5;
    stats[capacity + source] = sourceImportance * 0.5;
    atomicAdd(&control[eventBase + 20u], 1u);
  }
  if (mode == 1u) { atomicAdd(&control[eventBase + 1u], 1u); }
  else if (mode == 2u) { atomicAdd(&control[eventBase], 1u); }
  if (structureGuided) { atomicAdd(&control[eventBase + 11u], 1u); }
}

@compute @workgroup_size(64)
fn select_relocation(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let width = u32(config[0]);
  let height = u32(config[1]);
  let count = u32(config[2]);
  let step = config[4];
  let capacity = u32(config[10]);
  let adcRecycle = u32(config[11]) == 3u;
  if (g >= count) { return; }
  let t = transform[g];
  let c = color[g];
  let st = stats[g];
  let signal = normalized_stats(g);
  let candidateImportance = importance_score(g);
  let radiusPx = max(t.x, t.y) * max(f32(width), f32(height)) * 1.25;
  let inactiveMcmc = t.w < 0.5 || c.a < 0.006 || radiusPx < 0.55 || (st.w > 32.0 && signal.y < 0.012 && signal.z < 0.00008);
  let lowImportanceNoise = config[16] > 0.5 && st.w > 32.0 && candidateImportance < 0.45 && importance_residual(g) < 0.035;
  let candidateStructure = structure_at(xy[g], width, height);
  let lowSignificanceSmooth = config[27] > 0.5 && st.w > 32.0 && candidateImportance < 0.75 && importance_residual(g) < 0.02 && candidateStructure.z < 0.0004;
  let inactiveAdc = t.w < 0.5 || c.a < 0.025 || radiusPx < 0.65 || (st.w > 32.0 && signal.z < 0.00002 && signal.y < 0.025) || lowImportanceNoise || lowSignificanceSmooth;
  var inactive = select(inactiveMcmc, inactiveAdc, adcRecycle);
  let phase45DonorRecord = atomicLoad(&control[phase45_donor_base(capacity) + g]);
  if (adcRecycle && config[45] > 0.5) { inactive = (phase45DonorRecord & PHASE45_DONOR_ELIGIBLE) != 0u; }
  let lateRecycle = adcRecycle && u32(step) > u32(config[15]);
  let adcSelectionRate = select(config[50], config[51], lateRecycle);
  let selectedRate = select(0.02, adcSelectionRate, adcRecycle);
  let selected = hash_unit(f32(g) * 37.0 + step * 0.137) < selectedRate;
  if (!inactive || !selected) { return; }
  if (adcRecycle && config[16] > 0.5 && candidateImportance > 1.5) {
    atomicAdd(&control[capacity * 2u + 8u], 1u);
    return;
  }
  let source = pick_source(g + 104729u, count, adcRecycle);
  if (source == g || color[source].a < 0.02) { return; }
  let sourceT = transform[source];
  let sourceC = color[source];
  let sourceTiltProfile = tilt_split_profile(source, width, height);
  let sourceTiltRisk = adcRecycle && sourceTiltProfile.x > 0.0;
  let sourceSignal = normalized_stats(source);
  let sourceRadiusPx = max(sourceT.x, sourceT.y) * max(f32(width), f32(height)) * 1.25;
  let sourceInactiveMcmc = sourceT.w < 0.5 || sourceC.a < 0.006 || sourceRadiusPx < 0.55 || (stats[source].w > 32.0 && sourceSignal.y < 0.012 && sourceSignal.z < 0.00008);
  let sourceLowImportanceNoise = config[16] > 0.5 && stats[source].w > 32.0 && importance_score(source) < 0.45 && importance_residual(source) < 0.035;
  let sourceInactiveAdc = sourceT.w < 0.5 || sourceC.a < 0.025 || sourceRadiusPx < 0.65 || (stats[source].w > 32.0 && sourceSignal.z < 0.00002 && sourceSignal.y < 0.025) || sourceLowImportanceNoise;
  let sourceInactive = select(sourceInactiveMcmc, sourceInactiveAdc, adcRecycle);
  if (sourceInactive) { return; }
  if (adcRecycle && sourceSignal.x + sourceSignal.y * 0.5 <= 0.00015 && !sourceTiltRisk) { return; }
  let token = (g + 1u) & ROLE_TOKEN_MASK;
  let destinationRole = ROLE_DESTINATION | token;
  if (!try_claim_role(g, destinationRole)) {
    atomicAdd(&control[capacity * 2u + 16u], 1u);
    return;
  }
  let sourceRole = select(ROLE_SOURCE_OTHER, ROLE_SOURCE_SPLIT, sourceTiltRisk) | token;
  if (!try_claim_role(source, sourceRole)) {
    rollback_role(g, destinationRole);
    atomicAdd(&control[capacity * 2u + 16u], 1u);
    return;
  }
  atomicAdd(&control[capacity * 2u + 17u], 1u);
  if (sourceTiltRisk) { atomicAdd(&control[capacity * 2u + 19u], 1u); }
  atomicStore(&control[capacity + g], encode_selection(source, select(0u, 3u, adcRecycle)));
  if (adcRecycle && config[45] > 0.5) {
    let donorRegion = (phase45DonorRecord >> 8u) & 63u;
    atomicAdd(&control[phase45_region_base(capacity) + donorRegion * PHASE45_REGION_STRIDE + 23u], 1u);
  }
}

@compute @workgroup_size(64)
fn apply_relocation(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let width = u32(config[0]);
  let height = u32(config[1]);
  let count = u32(config[2]);
  let step = config[4];
  let maxAnisotropy = max(config[9], 1.0);
  let baseScale = vec2<f32>(config[52], config[53]);
  let capacity = u32(config[10]);
  if (g >= count) { return; }
  let encoded = atomicLoad(&control[capacity + g]);
  if ((encoded & SOURCE_MASK) == 0u) { return; }
  let source = (encoded & SOURCE_MASK) - 1u;
  let selectionMode = encoded >> 30u;
  let adcRecycle = selectionMode == 3u;
  let sourceT = transform[source];
  let sourceC = color[source];
  let sourceStats = stats[source];
  let sourceImportance = importance_at(source);
  let tiltProfile = tilt_split_profile(source, width, height);
  let sourceClaim = atomicLoad(&control[source]);
  var tiltTrueSplit = adcRecycle && (sourceClaim & ROLE_MASK) == ROLE_SOURCE_SPLIT && tiltProfile.x > 0.0;
  let destinationStructure = structure_at(xy[g], width, height);
  let useX = select(sourceT.x >= sourceT.y, tiltProfile.y > 0.5, tiltTrueSplit);
  let sourceStructure = structure_at(xy[source], width, height);
  let sourceLongAngle = sourceT.z + select(1.57079632679, 0.0, useX);
  let axis = vec2<f32>(cos(sourceLongAngle), sin(sourceLongAngle));
  let perp = vec2<f32>(-axis.y, axis.x);
  let side = select(-1.0, 1.0, hash_unit(f32(g) * 53.0 + step * 1.7) > 0.5);
  let jitter = (hash_unit(f32(g) * 71.0 + step * 2.3) - 0.5) * min(sourceT.x, sourceT.y) * 0.35;
  let major = select(sourceT.y, sourceT.x, useX);
  var nextPos = constrain_xy(xy[source] + axis * major * select(select(0.52, 0.55, adcRecycle) * side, 0.34, tiltTrueSplit) + perp * select(jitter, 0.0, tiltTrueSplit));
  var splitScale = min(sourceT.xy, baseScale * 0.9);
  if (adcRecycle) {
    let splitShrink = select(0.72, clamp(config[41], 0.5, 0.85), tiltTrueSplit);
    let axisShrink = sourceT.xy * vec2<f32>(select(0.94, splitShrink, useX), select(splitShrink, 0.94, useX));
    splitScale = select(axisShrink, sourceT.xy * splitShrink, tiltTrueSplit);
  }
  var nextTheta = sourceT.z;
  let localStructure = structure_at(nextPos, width, height);
  let localError = error_at_position(nextPos, width, height);
  let structureGuided = !tiltTrueSplit && config[19] > 0.5 && localStructure.y > 0.45 && localStructure.z > 0.0004 && localError > 0.02;
  let adaptiveDetail = config[29] > 0.5 && localStructure.y >= config[31] && localStructure.z > 0.0004 && localError > 0.02;
  let inheritedDetail = sourceT.w > 1.5;
  let detailTagged = adaptiveDetail || inheritedDetail;
  let surfaceMaxAnisotropy = max(config[55], 1.0);
  let localMaxAnisotropy = select(min(maxAnisotropy, surfaceMaxAnisotropy), maxAnisotropy, detailTagged);
  if (structureGuided) {
    let areaRadius = sqrt(max(0.00000001, splitScale.x * splitScale.y));
    let ratio = min(localMaxAnisotropy, select(1.6, localMaxAnisotropy, config[28] > 0.5 || adaptiveDetail));
    splitScale = vec2<f32>(areaRadius * sqrt(ratio), areaRadius / sqrt(ratio));
    nextTheta = localStructure.x;
  }
  let nextScale = constrain_scale(nextPos, splitScale, nextTheta, localMaxAnisotropy);
  nextPos = constrain_position(nextPos, nextScale, nextTheta);
  var replacementSourcePos = xy[source];
  var replacementSourceScale = sourceT.xy;
  if (tiltTrueSplit) {
    replacementSourcePos = constrain_xy(xy[source] - axis * major * 0.34);
    replacementSourceScale = constrain_scale(replacementSourcePos, nextScale, sourceT.z, localMaxAnisotropy);
    replacementSourcePos = constrain_position(replacementSourcePos, replacementSourceScale, sourceT.z);
  }
  let massShare = sourceC.a * max(0.00000001, sourceT.x * sourceT.y);
  var childOpacity = min(0.99, max(0.005, massShare / max(0.00000001, nextScale.x * nextScale.y)));
  if (tiltTrueSplit) {
    let replacementArea = nextScale.x * nextScale.y + replacementSourceScale.x * replacementSourceScale.y;
    let replacementOpacity = massShare / max(0.00000001, replacementArea);
    if (replacementOpacity > 0.99) { atomicAdd(&control[capacity * 2u + 21u], 1u); }
    childOpacity = clamp(replacementOpacity, 0.005, 0.99);
  }
  let targetColor = target_at(nextPos, width, height);
  xy[g] = nextPos;
  transform[g] = vec4<f32>(nextScale, nextTheta, select(1.0, 2.0, detailTagged) + fract(sourceT.w));
  let nextColor = select(
    sourceC.rgb * select(0.7, 0.6, adcRecycle) + targetColor * select(0.3, 0.4, adcRecycle),
    sourceC.rgb,
    tiltTrueSplit
  );
  color[g] = vec4<f32>(nextColor, childOpacity);
  stats[g] = select(sourceStats, sourceStats * 0.5, tiltTrueSplit);
  stats[capacity + g] = select(
    vec4<f32>(0.0, 0.0, 0.0, select(0.0, sourceImportance.w, config[25] > 0.5)),
    sourceImportance * 0.5,
    tiltTrueSplit
  );
  let eventBase = capacity * 2u;
  if (tiltTrueSplit) {
    xy[source] = replacementSourcePos;
    transform[source] = vec4<f32>(replacementSourceScale, sourceT.z, sourceT.w);
    color[source] = vec4<f32>(sourceC.rgb, childOpacity);
    stats[source] = sourceStats * 0.5;
    stats[capacity + source] = sourceImportance * 0.5;
    atomicAdd(&control[eventBase + 20u], 1u);
  }
  if (adcRecycle) {
    atomicAdd(&control[eventBase + 1u], 1u);
    atomicAdd(&control[eventBase + 7u], 1u);
    let destinationHigh = destinationStructure.z >= 0.0004;
    let sourceHigh = sourceStructure.z >= 0.0004;
    if (!destinationHigh && sourceHigh) {
      atomicAdd(&control[eventBase + 13u], 1u);
    } else if (destinationHigh && !sourceHigh) {
      atomicAdd(&control[eventBase + 14u], 1u);
    } else {
      atomicAdd(&control[eventBase + 15u], 1u);
    }
  } else {
    atomicAdd(&control[eventBase + 3u], 1u);
  }
  if (structureGuided) { atomicAdd(&control[eventBase + 11u], 1u); }
  atomicAdd(&control[eventBase + 4u], 1u);
}

@compute @workgroup_size(64)
fn reset_density_aux(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  if (g >= count || config[25] <= 0.5) { return; }
  stats[capacity + g].w = 0.0;
}

`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.densityBindGroupLayout] });
    const make = (entryPoint) => this.device.createComputePipelineAsync({ layout, compute: { module, entryPoint } });
    [
      this.distributionPipeline,
      this.distributionBlockScanPipeline,
      this.distributionBlockSumsPipeline,
      this.distributionOffsetPipeline,
      this.growSelectPipeline,
      this.growRedistributePipeline,
      this.growApplyPipeline,
      this.relocationSelectPipeline,
      this.relocationApplyPipeline,
      this.densityAuxResetPipeline,
      this.phase45RegionTelemetryPipeline,
      this.phase45RegionFinalizePipeline,
      this.phase45DonorSafetyPipeline,
    ] = await Promise.all([
      make("build_distribution"),
      make("scan_distribution_blocks"),
      make("scan_distribution_block_sums"),
      make("add_distribution_block_offsets"),
      make("select_grow"),
      make("redistribute_sources"),
      make("apply_grow"),
      make("select_relocation"),
      make("apply_relocation"),
      make("reset_density_aux"),
      make("phase45_collect_region_telemetry"),
      make("phase45_finalize_region_telemetry"),
      make("phase45_evaluate_donor_safety"),
    ]);
    await this.ensureOptimizerResetPipeline();
  }

  async ensureOptimizerResetPipeline() {
    if (this.optimizerResetPipeline && this.optimizerSourceResetPipeline) return;
    const packedStats = this.geometryPrecomputeEnabled();
    const resetAdamExtraFields = packedStats ? "stats: vec4<f32>,\n  importance: vec4<f32>," : "";
    const resetAdamExtraValues = packedStats ? ", vec4<f32>(0.0), vec4<f32>(0.0)" : "";
    const shader = `
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
  ${resetAdamExtraFields}
};
@group(0) @binding(0) var<storage, read> config: array<f32>;
@group(0) @binding(1) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> adam: array<AdamState>;
const SOURCE_MASK = 0x3fffffffu;
const ROLE_SOURCE_MASK = 0x60000000u;

fn reset_state(adcResetStep: f32) -> AdamState {
  return AdamState(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, adcResetStep), vec4<f32>(0.0)${resetAdamExtraValues});
}

@compute @workgroup_size(64)
fn reset_selected(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let oldCount = u32(config[2]);
  let targetCount = u32(config[3]);
  let capacity = u32(config[10]);
  let mode = u32(config[11]);
  var destination = g;
  var selection = capacity + g;
  if (mode == 1u) {
    if (g >= targetCount - oldCount) { return; }
    destination = oldCount + g;
  } else if (g >= oldCount) {
    return;
  }

  let encoded = atomicLoad(&control[selection]);
  if ((encoded & SOURCE_MASK) == 0u) { return; }
  let adcResetStep = select(0.0, config[36], mode == 3u);
  adam[destination] = reset_state(adcResetStep);
}

@compute @workgroup_size(64)
fn reset_sources(@builtin(global_invocation_id) id: vec3u) {
  let source = id.x;
  let oldCount = u32(config[2]);
  let mode = u32(config[11]);
  if (source >= oldCount) { return; }
  let packed = atomicLoad(&control[source]);
  let roleSelected = (packed & ROLE_SOURCE_MASK) != 0u;
  let legacySelected = select(packed > 0u, (packed >> 16u) > 0u, mode == 1u);
  let selected = select(legacySelected, roleSelected, config[42] > 0.5);
  if (!selected) { return; }
  let adcResetStep = select(0.0, config[36], mode == 3u);
  adam[source] = reset_state(adcResetStep);
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    [this.optimizerResetPipeline, this.optimizerSourceResetPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "reset_selected" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "reset_sources" } }),
    ]);
  }

  async ensurePackedStatsSyncPipelines() {
    if (this.packedStatsExportPipeline && this.packedStatsImportPipeline) return;
    const shader = `
struct SyncConfig { count: u32, capacity: u32, pad0: u32, pad1: u32, };
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
  stats: vec4<f32>,
  importance: vec4<f32>,
};
@group(0) @binding(0) var<uniform> sync: SyncConfig;
@group(0) @binding(1) var<storage, read_write> adam: array<AdamState>;
@group(0) @binding(2) var<storage, read_write> densityStats: array<vec4<f32>>;

@compute @workgroup_size(64)
fn export_stats(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  if (g >= sync.count) { return; }
  densityStats[g] = adam[g].stats;
  densityStats[sync.capacity + g] = adam[g].importance;
}

@compute @workgroup_size(64)
fn import_stats(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  if (g >= sync.count) { return; }
  var value = adam[g];
  value.stats = densityStats[g];
  value.importance = densityStats[sync.capacity + g];
  adam[g] = value;
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    [this.packedStatsExportPipeline, this.packedStatsImportPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "export_stats" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "import_stats" } }),
    ]);
  }

  async syncPackedStats(direction, count = this.trainState?.count || 0) {
    if (!this.trainState || !this.geometryPrecomputeEnabled() || count <= 0) return;
    await this.ensurePackedStatsSyncPipelines();
    const pipeline = direction === "import" ? this.packedStatsImportPipeline : this.packedStatsExportPipeline;
    this.device.queue.writeBuffer(
      this.trainState.packedStatsConfigBuffer,
      0,
      new Uint32Array([count, this.trainState.capacity, 0, 0]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.packedStatsConfigBuffer } },
        { binding: 1, resource: { buffer: this.trainState.optimizerStateBuffer } },
        { binding: 2, resource: { buffer: this.trainState.statsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  optimizerResetBindGroup(pipeline = this.optimizerResetPipeline) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.densityControlBuffer } },
        { binding: 2, resource: { buffer: this.trainState.optimizerStateBuffer } },
      ],
    });
  }

  densityBindGroup(front = this.trainState.front) {
    return this.device.createBindGroup({
      layout: this.densityBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.targetBuffer } },
        { binding: 2, resource: { buffer: this.trainState.xyBuffers[front] } },
        { binding: 3, resource: { buffer: this.trainState.transformBuffers[front] } },
        { binding: 4, resource: { buffer: this.trainState.colorBuffers[front] } },
        { binding: 5, resource: { buffer: this.trainState.statsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.densityControlBuffer } },
        { binding: 7, resource: { buffer: this.trainState.errorMapBuffer } },
      ],
    });
  }

  phase45DonorBindGroup(front = this.trainState.front) {
    return this.device.createBindGroup({
      layout: this.densityBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.targetBuffer } },
        { binding: 2, resource: { buffer: this.trainState.xyBuffers[front] } },
        { binding: 3, resource: { buffer: this.trainState.transformBuffers[front] } },
        { binding: 4, resource: { buffer: this.trainState.colorBuffers[front] } },
        { binding: 5, resource: { buffer: this.trainState.statsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.densityControlBuffer } },
        { binding: 7, resource: { buffer: this.trainState.pixelStateBuffer } },
      ],
    });
  }

  async uploadTrainState(image, params, capacity = params.count, { verifyAllocation = false } = {}) {
    const activeCount = assertSplatCountContract(params, "gpu-upload");
    const previousGeometryEnabled = this.geometryPrecomputeEnabled();
    const capacityError = (kind, message) => Object.assign(new Error(message), { capacityFailure: kind });
    const variants = phase33Variants();
    const { coarseImage, midImage } = makeCurriculumImages(image, variants);
    const bufferCapacity = Math.max(activeCount, normalizeUiSplatCount(capacity, activeCount));
    const geometrySupport = this.geometryPrecomputeSupport(bufferCapacity);
    const tilePlan = plannedTileIndexCapacity(image, params, bufferCapacity, this.device);
    const allocationPlan = trainingAllocationPlan(image, params, bufferCapacity, this.device, {
      coarseImage,
      midImage,
      geometrySupport,
      tilePlan,
    });
    if (!allocationPlan.withinBufferLimits) {
      throw capacityError(
        "validation",
        `GPU capacity rejected before allocation: largest buffer ${formatMB(allocationPlan.largestBufferBytes)} exceeds WebGPU limits`,
      );
    }
    if (!allocationPlan.withinBudget) {
      throw capacityError(
        "out-of-memory",
        `GPU capacity rejected before allocation: ${formatMB(allocationPlan.reservedBytes)} exceeds the 90% working-set budget`,
      );
    }

    this.disposeTrainState();
    const allocatedResources = [];
    const track = (resource) => {
      allocatedResources.push(resource);
      return resource;
    };
    const allocationDevice = {
      queue: this.device.queue,
      createBuffer: (descriptor) => track(this.device.createBuffer(descriptor)),
      createQuerySet: (descriptor) => track(this.device.createQuerySet(descriptor)),
    };
    let scopesOpen = false;
    const popAllocationScopes = async () => {
      if (!scopesOpen) return { validationError: null, oomError: null, popErrors: [] };
      let validationError = null;
      let oomError = null;
      const popErrors = [];
      try {
        validationError = await this.device.popErrorScope();
      } catch (error) {
        popErrors.push(error);
      }
      try {
        oomError = await this.device.popErrorScope();
      } catch (error) {
        popErrors.push(error);
      }
      scopesOpen = false;
      return { validationError, oomError, popErrors };
    };
    if (previousGeometryEnabled !== geometrySupport.supported) {
      this.geometryPrecomputePipeline = null;
      this.tileCountPipeline = null;
      this.tilePrefixPipeline = null;
      this.tileFillPipeline = null;
      this.tileSortPipeline = null;
      this.renderStatePipeline = null;
      this.tileCooperativeRenderPipeline = null;
      this.renderGradientPipeline = null;
      this.parallelRenderGradientPipeline = null;
      this.lossGradientPipeline = null;
      this.exactAlphaBackwardPipeline = null;
      this.exactBackwardTelemetryPipeline = null;
      this.exactOptimizerPipeline = null;
      this.optimizerResetPipeline = null;
      this.optimizerSourceResetPipeline = null;
      this.packedStatsExportPipeline = null;
      this.packedStatsImportPipeline = null;
    }
    const tileCols = Math.ceil(image.width / TILE_SIZE);
    const tileRows = Math.ceil(image.height / TILE_SIZE);
    const tileCount = tileCols * tileRows;
    const tileIndexCapacity = tilePlan.capacity;
    const ssimTileCount = Math.ceil(image.width / 8) * Math.ceil(image.height / 8);
    const color = packColors(params);
    const transform = packTransforms(params);
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    scopesOpen = true;
    try {
    const candidate = {
      width: image.width,
      height: image.height,
      count: params.count,
      capacity: bufferCapacity,
      front: 0,
      geometryPrecomputeRequested: geometrySupport.requested,
      geometryPrecomputeEnabled: geometrySupport.supported,
      geometryPrecomputeReason: geometrySupport.reason,
      geometryPrecomputeLimits: geometrySupport.limits,
      geometryBuffer: geometrySupport.supported
        ? allocationDevice.createBuffer({
            size: geometrySupport.bytes,
            usage: GPUBufferUsage.STORAGE,
          })
        : null,
      configBuffer: allocationDevice.createBuffer({
        size: 64 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      presentConfigBuffer: allocationDevice.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      packedStatsConfigBuffer: allocationDevice.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      targetBuffer: makeBuffer(allocationDevice, image.rgb, GPUBufferUsage.STORAGE),
      coarseTargetBuffer: coarseImage ? makeBuffer(allocationDevice, coarseImage.rgb, GPUBufferUsage.STORAGE) : null,
      midTargetBuffer: midImage ? makeBuffer(allocationDevice, midImage.rgb, GPUBufferUsage.STORAGE) : null,
      targetAlphaBuffer: makeBuffer(allocationDevice, image.alpha || new Float32Array(image.width * image.height).fill(1), GPUBufferUsage.STORAGE),
      coarseTargetAlphaBuffer: coarseImage ? makeBuffer(allocationDevice, coarseImage.alpha, GPUBufferUsage.STORAGE) : null,
      midTargetAlphaBuffer: midImage ? makeBuffer(allocationDevice, midImage.alpha, GPUBufferUsage.STORAGE) : null,
      coarseImage,
      midImage,
      coarseTrainingSteps: 0,
      midTrainingSteps: 0,
      virtualTiltSteps: 0,
      lastVirtualTilt: null,
      pixelStateResolution: null,
      pixelStateKind: "uninitialized",
      errorMapBuffer: allocationDevice.createBuffer({
        size: Math.max(4, image.width * image.height * 4),
        usage: GPUBufferUsage.STORAGE,
      }),
      statsBuffer: allocationDevice.createBuffer({
        size: Math.max(32, bufferCapacity * 2 * 4 * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      densityControlBuffer: allocationDevice.createBuffer({
        size: Math.max(32, (bufferCapacity * 4 + DENSITY_EVENT_SLOTS + 1 + Math.ceil(bufferCapacity / 256) * 2 + PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE) * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      tileCountsBuffer: allocationDevice.createBuffer({
        size: Math.max(4, tileCount * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      tileOffsetsBuffer: allocationDevice.createBuffer({
        size: Math.max(8, (tileCount + 1) * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      tileCursorsBuffer: allocationDevice.createBuffer({
        size: Math.max(4, tileCount * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      tileIndicesBuffer: allocationDevice.createBuffer({
        size: tileIndexCapacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      tileControlBuffer: allocationDevice.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      pixelStateBuffer: allocationDevice.createBuffer({
        size: Math.max(16, image.width * image.height * 16),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      alphaStateBuffer: allocationDevice.createBuffer({
        size: Math.max(8, image.width * image.height * 8),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      lossGradientBuffer: allocationDevice.createBuffer({
        size: Math.max(48, image.width * image.height * 48),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      exactGradientBuffer: allocationDevice.createBuffer({
        size: Math.max(EXACT_GRADIENT_STRIDE * 4, bufferCapacity * EXACT_GRADIENT_STRIDE * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      exactBackwardTelemetryBuffer: this.performanceProfile.timestampQuery
        ? allocationDevice.createBuffer({
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          })
        : null,
      exactBackwardTelemetryReadbackBuffer: this.performanceProfile.timestampQuery
        ? allocationDevice.createBuffer({
            size: 32,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          })
        : null,
      ssimTileBuffer: allocationDevice.createBuffer({
        size: Math.max(64, ssimTileCount * 64),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      optimizerStateBuffer: allocationDevice.createBuffer({
        size: Math.max(
          geometrySupport.supported ? PACKED_OPTIMIZER_STRIDE_BYTES : 96,
          bufferCapacity * (geometrySupport.supported ? PACKED_OPTIMIZER_STRIDE_BYTES : 96),
        ),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      tileCols,
      tileRows,
      tileCount,
      tileIndexCapacity,
      tileIndexInitialReferences: tilePlan.observed,
      tileIndexInitialReferencesPerSplat: tilePlan.observedPerSplat,
      tileIndexRequestedCapacity: tilePlan.requested,
      allocationPlan,
      tileReady: false,
      tileBuilds: 0,
      tileReserveLevel: 0,
      stageProfile: [],
      profileQuerySet: this.performanceProfile.timestampQuery
        ? allocationDevice.createQuerySet({ type: "timestamp", count: PERFORMANCE_PROFILE_QUERY_CAPACITY })
        : null,
      profileResolveBuffer: this.performanceProfile.timestampQuery
        ? allocationDevice.createBuffer({
            size: PERFORMANCE_PROFILE_QUERY_CAPACITY * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
          })
        : null,
      profileReadbackBuffer: this.performanceProfile.timestampQuery
        ? allocationDevice.createBuffer({
            size: PERFORMANCE_PROFILE_QUERY_CAPACITY * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          })
        : null,
      growthSignalReadbackBuffer: allocationDevice.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      zeroDensityScratch: new Uint32Array(bufferCapacity * 2),
      zeroDensityEvents: new Uint32Array(DENSITY_EVENT_SLOTS),
      zeroPhase45RegionTelemetry: new Uint32Array(PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE),
      zeroPhase45DonorTelemetry: new Uint32Array(bufferCapacity),
      zeroStats: new Float32Array(bufferCapacity * 4),
      xyBuffers: [
        makeSizedBuffer(allocationDevice, params.xy, bufferCapacity * 2 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
      ],
      transformBuffers: [
        makeSizedBuffer(allocationDevice, transform, bufferCapacity * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
      ],
      colorBuffers: [
        makeSizedBuffer(allocationDevice, color, bufferCapacity * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
      ],
      readbackBuffer: allocationDevice.createBuffer({
        size: bufferCapacity * (2 + 4 + 4) * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
    };
    this.device.queue.writeBuffer(
      candidate.densityControlBuffer,
      0,
      new Uint32Array(bufferCapacity * 3 + DENSITY_EVENT_SLOTS + 1),
    );
    const encoder = this.device.createCommandEncoder();
    for (const buffer of [
      candidate.statsBuffer,
      candidate.densityControlBuffer,
      candidate.tileIndicesBuffer,
      candidate.pixelStateBuffer,
      candidate.alphaStateBuffer,
      candidate.lossGradientBuffer,
      candidate.exactGradientBuffer,
      candidate.tileControlBuffer,
      candidate.ssimTileBuffer,
      candidate.optimizerStateBuffer,
      candidate.readbackBuffer,
    ]) encoder.clearBuffer(buffer);
    if (candidate.exactBackwardTelemetryBuffer) encoder.clearBuffer(candidate.exactBackwardTelemetryBuffer);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const scoped = await popAllocationScopes();
    if (scoped.popErrors.length) throw capacityError("internal", `GPU error-scope failure: ${scoped.popErrors.map((error) => error.message).join(" | ")}`);
    if (scoped.validationError) throw capacityError("validation", `GPU capacity validation failed: ${scoped.validationError.message}`);
    if (scoped.oomError) throw capacityError("out-of-memory", `GPU capacity allocation failed: ${scoped.oomError.message}`);
    if (verifyAllocation) {
      const actualReservedBytes = allocatedResources.reduce(
        (total, resource) => total + Number(resource?.size || 0),
        0,
      );
      if (actualReservedBytes !== allocationPlan.reservedBytes) {
        throw capacityError(
          "internal",
          `GPU allocation descriptor drift: planned ${allocationPlan.reservedBytes} bytes, allocated ${actualReservedBytes} bytes`,
        );
      }
    }
    this.trainState = candidate;
    allocatedResources.length = 0;
    updateGpuMemoryStatus();
    } catch (error) {
      const scoped = await popAllocationScopes();
      for (const resource of allocatedResources) resource?.destroy?.();
      this.trainState = null;
      updateGpuMemoryStatus();
      if (error.capacityFailure) throw error;
      if (scoped.popErrors.length) throw capacityError("internal", `${error.message}; ${scoped.popErrors.map((item) => item.message).join(" | ")}`);
      if (scoped.validationError) throw capacityError("validation", `${error.message}; ${scoped.validationError.message}`);
      if (scoped.oomError) throw capacityError("out-of-memory", `${error.message}; ${scoped.oomError.message}`);
      throw error;
    }
  }

  async probeTrainingCapacity(image, params, requestedCapacity) {
    const attempts = [];
    state.capacityProbe = {
      status: "probing",
      requested: requestedCapacity,
      selected: 0,
      attempts,
      fastPath: false,
    };
    publishState();
    for (const capacity of capacityProbeCandidates(requestedCapacity)) {
      if (capacity <= CAPACITY_PROBE_FAST_PATH_MAX) continue;
      const plan = trainingAllocationPlan(image, params, capacity, this.device);
      if (!plan.withinBufferLimits || !plan.withinBudget) {
        attempts.push({
          capacity,
          status: "rejected",
          reason: !plan.withinBufferLimits ? "per-buffer-limit" : "working-set-budget",
          estimated_reserved_bytes: plan.reservedBytes,
        });
        continue;
      }
      const started = performance.now();
      try {
        await this.uploadTrainState(image, params, capacity, { verifyAllocation: true });
        const actual = this.trainingMemorySnapshot();
        if (actual.reservedBytes > memoryBudgetBytes() * 0.9) {
          this.disposeTrainState();
          attempts.push({
            capacity,
            status: "rejected",
            duration_ms: performance.now() - started,
            reason: "actual-working-set-budget",
            estimated_reserved_bytes: plan.reservedBytes,
            actual_reserved_bytes: actual.reservedBytes,
          });
          continue;
        }
        attempts.push({
          capacity,
          status: "passed",
          duration_ms: performance.now() - started,
          estimated_reserved_bytes: plan.reservedBytes,
          actual_reserved_bytes: actual.reservedBytes,
          tile_capacity: this.trainState.tileIndexCapacity,
        });
        state.capacityProbe.status = "passed";
        state.capacityProbe.selected = capacity;
        publishState();
        return { capacity, attempts, plan, fastPath: false };
      } catch (error) {
        attempts.push({ capacity, status: "failed", duration_ms: performance.now() - started, reason: error.message });
        log(`capacity probe ${compactNumber(capacity)} failed: ${error.message}`);
        if (!state.webgpu.supported || state.webgpu.renderer !== this) throw error;
        if (error.capacityFailure !== "out-of-memory") {
          state.capacityProbe.status = "failed";
          publishState();
          throw error;
        }
      }
    }

    const fallback = Math.min(requestedCapacity, CAPACITY_PROBE_FAST_PATH_MAX);
    try {
      await this.uploadTrainState(image, params, fallback);
    } catch (error) {
      state.capacityProbe.status = "failed";
      publishState();
      throw error;
    }
    attempts.push({ capacity: fallback, status: "fallback" });
    state.capacityProbe.status = "fallback";
    state.capacityProbe.selected = fallback;
    publishState();
    return { capacity: fallback, attempts, plan: trainingAllocationPlan(image, params, fallback, this.device), fastPath: true };
  }

  async growExperimentalGpu(image, params, targetCount, step, steps) {
    if (!this.trainState || this.trainState.capacity < targetCount || targetCount <= params.count) return false;
    await this.ensureDensityPipelines();
    const oldCount = params.count;
    const layout = splatGridLayout(image, targetCount);
    const { config } = densityGpuConfig({
      image,
      count: oldCount,
      targetCount,
      step,
      steps,
      layout,
      maxAnisotropy: currentMaxAnisotropy(),
      capacity: this.trainState.capacity,
      mode: 1,
    });
    this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    this.device.queue.writeBuffer(this.trainState.densityControlBuffer, 0, this.trainState.zeroDensityScratch);
    this.device.queue.writeBuffer(
      this.trainState.densityControlBuffer,
      this.trainState.capacity * 2 * 4,
      this.trainState.zeroDensityEvents,
    );
    await this.syncPackedStats("export", oldCount);
    const front = this.trainState.front;
    const bindGroup = this.densityBindGroup(front);
    const distributionEncoder = this.device.createCommandEncoder();
    const distributionPass = distributionEncoder.beginComputePass();
    distributionPass.setPipeline(this.distributionPipeline);
    distributionPass.setBindGroup(0, bindGroup);
    distributionPass.dispatchWorkgroups(Math.ceil(oldCount / 256));
    distributionPass.setPipeline(this.distributionBlockScanPipeline);
    distributionPass.dispatchWorkgroups(Math.ceil(oldCount / 256));
    distributionPass.setPipeline(this.distributionBlockSumsPipeline);
    distributionPass.dispatchWorkgroups(1);
    distributionPass.setPipeline(this.distributionOffsetPipeline);
    distributionPass.dispatchWorkgroups(Math.ceil(oldCount / 256));
    distributionPass.end();
    const cdfTotalOffset = (this.trainState.capacity * 3 + DENSITY_EVENT_SLOTS) * 4;
    distributionEncoder.copyBufferToBuffer(
      this.trainState.densityControlBuffer,
      cdfTotalOffset,
      this.trainState.growthSignalReadbackBuffer,
      0,
      4,
    );
    this.device.queue.submit([distributionEncoder.finish()]);
    await this.trainState.growthSignalReadbackBuffer.mapAsync(GPUMapMode.READ);
    const candidateMass = new Float32Array(this.trainState.growthSignalReadbackBuffer.getMappedRange())[0];
    this.trainState.growthSignalReadbackBuffer.unmap();
    if (!Number.isFinite(candidateMass) || candidateMass <= 1e-8) {
      const qaCounters = phase39Variants().qaGrowthComparisons
        ? await this.readDensityCounters()
        : null;
      this.lastTrainStats = {
        ...(this.lastTrainStats || {}),
        gpu_densify: false,
        growth_threshold_skipped: true,
        growth_candidate_mass: Number.isFinite(candidateMass) ? candidateMass : null,
        active_count: oldCount,
      };
      return {
        grown: false,
        count: oldCount,
        candidateMass,
        operations: qaCounters ? { eligible_sources: qaCounters.growth_eligible_sources } : {},
      };
    }
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bindGroup);
    pass.setPipeline(this.growSelectPipeline);
    pass.dispatchWorkgroups(Math.ceil((targetCount - oldCount) / 64));
    pass.setPipeline(this.growRedistributePipeline);
    pass.dispatchWorkgroups(Math.ceil(oldCount / 64));
    pass.setPipeline(this.growApplyPipeline);
    pass.dispatchWorkgroups(Math.ceil((targetCount - oldCount) / 64));
    pass.setPipeline(this.optimizerResetPipeline);
    pass.setBindGroup(0, this.optimizerResetBindGroup());
    pass.dispatchWorkgroups(Math.ceil((targetCount - oldCount) / 64));
    pass.setPipeline(this.optimizerSourceResetPipeline);
    pass.setBindGroup(0, this.optimizerResetBindGroup(this.optimizerSourceResetPipeline));
    pass.dispatchWorkgroups(Math.ceil(oldCount / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.syncPackedStats("import", targetCount);
    const operationCounters = await this.readDensityCounters();
    const operations = {
      split: Math.max(0, operationCounters?.adc_split || 0),
      duplicate: Math.max(0, operationCounters?.adc_duplicate || 0),
      reseed: Math.max(0, operationCounters?.mcmc_reseed || 0),
      source_claims: Math.max(0, operationCounters?.source_claims || 0),
      source_claim_conflicts: Math.max(0, operationCounters?.source_claim_conflicts || 0),
      eligible_sources: Math.max(0, operationCounters?.growth_eligible_sources || 0),
      tilt_risk_candidates: Math.max(0, operationCounters?.tilt_risk_candidates || 0),
      tilt_true_splits: Math.max(0, operationCounters?.tilt_true_splits || 0),
    };
    this.trainState.count = targetCount;
    this.lastTrainStats = {
      ...(this.lastTrainStats || {}),
      gpu_densify: true,
      render_aware_density: true,
      weighted_mass_redistribution: true,
      preallocated_capacity: this.trainState.capacity,
      active_count: targetCount,
      gpu_densify_sync: true,
      growth_signal_readback_bytes: 4,
      growth_candidate_mass: candidateMass,
      growth_threshold_skipped: false,
      growth_operations: operations,
      density_stats_reset_after_batch: false,
    };
    return { grown: true, count: targetCount, candidateMass, operations };
  }

  async relocateExperimentalGpu(image, params, step, learningRates = selectedLearningRates()) {
    if (!this.trainState || this.trainState.capacity < params.count) return false;
    await this.ensureDensityPipelines();
    const layout = splatGridLayout(image, params.count);
    const { config } = densityGpuConfig({
      image,
      count: params.count,
      targetCount: params.count,
      step,
      steps: state.metrics?.steps_requested || step,
      layout,
      maxAnisotropy: learningRates.maxAnisotropy,
      capacity: this.trainState.capacity,
      mode: 2,
    });
    this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    this.device.queue.writeBuffer(this.trainState.densityControlBuffer, 0, this.trainState.zeroDensityScratch);
    this.device.queue.writeBuffer(
      this.trainState.densityControlBuffer,
      this.trainState.capacity * 2 * 4,
      this.trainState.zeroDensityEvents,
    );
    await this.syncPackedStats("export", params.count);
    const front = this.trainState.front;
    const bindGroup = this.densityBindGroup(front);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bindGroup);
    pass.setPipeline(this.distributionPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    pass.setPipeline(this.distributionBlockScanPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    pass.setPipeline(this.distributionBlockSumsPipeline);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.distributionOffsetPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    pass.setPipeline(this.relocationSelectPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.growRedistributePipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.relocationApplyPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.optimizerResetPipeline);
    pass.setBindGroup(0, this.optimizerResetBindGroup());
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.optimizerSourceResetPipeline);
    pass.setBindGroup(0, this.optimizerResetBindGroup(this.optimizerSourceResetPipeline));
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await this.syncPackedStats("import", params.count);
    this.lastTrainStats = {
      ...(this.lastTrainStats || {}),
      gpu_relocation: true,
      gpu_relocation_step: step,
      weighted_mass_redistribution: true,
      active_count: params.count,
    };
    return true;
  }

  async adcResetExperimentalGpu(image, params, step, learningRates = selectedLearningRates()) {
    if (!this.trainState || this.trainState.capacity < params.count) return null;
    await this.ensureDensityPipelines();
    const phase45 = phase45Variants();
    const adcControls = phase39Variants();
    const phase45Active = phase45.telemetry || phase45.donorEligibility || phase45.recipientScore;
    const phase45DonorActive = phase45.donorEligibility && (!phase45.firstResetOnly || step <= experimentalSchedule(state.metrics?.steps_requested || step).resetInterval);
    if (phase45DonorActive) {
      if (els.tileCullingToggle.checked) await this.prepareTileLists(image, params);
      await this.refreshRenderState(image, params);
    }
    const layout = splatGridLayout(image, params.count);
    const { config } = densityGpuConfig({
      image,
      count: params.count,
      targetCount: params.count,
      step,
      steps: state.metrics?.steps_requested || step,
      layout,
      maxAnisotropy: learningRates.maxAnisotropy,
      capacity: this.trainState.capacity,
      mode: 3,
    });
    this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    await this.syncPackedStats("export", params.count);
    this.device.queue.writeBuffer(this.trainState.densityControlBuffer, 0, this.trainState.zeroDensityScratch);
    this.device.queue.writeBuffer(
      this.trainState.densityControlBuffer,
      this.trainState.capacity * 2 * 4,
      this.trainState.zeroDensityEvents,
    );
    if (phase45Active) {
      this.device.queue.writeBuffer(
        this.trainState.densityControlBuffer,
        this.phase45RegionTelemetryOffsetBytes(),
        this.trainState.zeroPhase45RegionTelemetry,
      );
    }
    if (phase45DonorActive) {
      this.device.queue.writeBuffer(
        this.trainState.densityControlBuffer,
        this.phase45DonorTelemetryOffsetBytes(),
        this.trainState.zeroPhase45DonorTelemetry,
      );
    }
    const bindGroup = this.densityBindGroup(this.trainState.front);
    const donorBindGroup = phase45DonorActive ? this.phase45DonorBindGroup(this.trainState.front) : null;
    const optimizerResetBindGroup = this.optimizerResetBindGroup();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bindGroup);
    if (phase45Active) {
      pass.setPipeline(this.phase45RegionTelemetryPipeline);
      pass.dispatchWorkgroups(Math.ceil(params.count / 64));
      pass.setPipeline(this.phase45RegionFinalizePipeline);
      pass.dispatchWorkgroups(1);
    }
    pass.setPipeline(this.distributionPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    pass.setPipeline(this.distributionBlockScanPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    pass.setPipeline(this.distributionBlockSumsPipeline);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.distributionOffsetPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    if (phase45DonorActive) {
      pass.setPipeline(this.phase45DonorSafetyPipeline);
      pass.setBindGroup(0, donorBindGroup);
      pass.dispatchWorkgroups(Math.ceil(params.count / 64));
      pass.setBindGroup(0, bindGroup);
    }
    pass.setPipeline(this.relocationSelectPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.growRedistributePipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.relocationApplyPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.optimizerResetPipeline);
    pass.setBindGroup(0, optimizerResetBindGroup);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.optimizerSourceResetPipeline);
    pass.setBindGroup(0, this.optimizerResetBindGroup(this.optimizerSourceResetPipeline));
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.end();
    encoder.clearBuffer(this.trainState.statsBuffer, 0, this.trainState.capacity * 4 * 4);
    if (phase37Variants().gradientCoherence) {
      const auxPass = encoder.beginComputePass();
      auxPass.setPipeline(this.densityAuxResetPipeline);
      auxPass.setBindGroup(0, bindGroup);
      auxPass.dispatchWorkgroups(Math.ceil(params.count / 64));
      auxPass.end();
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await this.syncPackedStats("import", params.count);
    const countersAfter = await this.readDensityCounters();
    const phase45RegionReport = phase45Active ? await this.readPhase45RegionReport(params.count) : null;
    const operationCount = (name) => Math.max(0, countersAfter?.[name] || 0);
    const recycled = operationCount("adc_recycle");
    this.lastTrainStats = {
      ...(this.lastTrainStats || {}),
      gpu_adc_reset: true,
      gpu_adc_reset_step: step,
      gpu_adc_reset_mode: "recycle-denoise",
      gpu_adc_recycled: recycled,
      gpu_adc_frequency_moves: {
        low_to_high: operationCount("adc_low_to_high"),
        high_to_low: operationCount("adc_high_to_low"),
        same_band: operationCount("adc_same_band"),
      },
      optimizer_state_reset: "selected-source-destination",
      source_claims: operationCount("source_claims"),
      source_claim_conflicts: operationCount("source_claim_conflicts"),
      adc_recycle_percentage: adcControls.adcRecycleRate * 100,
      adc_late_recycle_percentage: adcControls.adcLateRecycleRate * 100,
      adc_split_signal_threshold: adcControls.adcSplitSignalThreshold,
      adc_split_residual_threshold: adcControls.adcSplitResidualThreshold,
      cpu_full_state_readback: false,
      phase45_region_report: phase45RegionReport,
    };
    return {
      step,
      gpu: true,
      mode: "recycle-denoise",
      recycled,
      frequency_moves: {
        low_to_high: operationCount("adc_low_to_high"),
        high_to_low: operationCount("adc_high_to_low"),
        same_band: operationCount("adc_same_band"),
      },
      optimizer_state_reset: "selected-source-destination",
      source_claims: operationCount("source_claims"),
      source_claim_conflicts: operationCount("source_claim_conflicts"),
      adc_recycle_percentage: adcControls.adcRecycleRate * 100,
      adc_late_recycle_percentage: adcControls.adcLateRecycleRate * 100,
      adc_split_signal_threshold: adcControls.adcSplitSignalThreshold,
      adc_split_residual_threshold: adcControls.adcSplitResidualThreshold,
      cpu_full_state_readback: false,
      phase45_region_report: phase45RegionReport,
    };
  }

  phase45RegionTelemetryOffsetBytes() {
    const capacity = this.trainState.capacity;
    return (capacity * 3 + DENSITY_EVENT_SLOTS + 1 + Math.ceil(capacity / 256) * 2) * 4;
  }

  phase45DonorTelemetryOffsetBytes() {
    return this.phase45RegionTelemetryOffsetBytes() + PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE * 4;
  }

  async readPhase45RegionReport(currentSplatCount) {
    if (!this.trainState) return null;
    const bytes = PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE * 4;
    const readBuffer = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.densityControlBuffer, this.phase45RegionTelemetryOffsetBytes(), readBuffer, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();
      const decodeEnergy = (value) => value / 65536;
      const decodeResidual = (value) => value / 4096;
      const decodeNormalized = (value) => value / 256;
      const regions = [];
      let populatedRegionCount = 0;
      let provisionalQuota = 0;
      let lowUtilityCount = 0;
      let locallyProtectedCount = 0;
      let supportSafeCount = 0;
      let donorEligibleCount = 0;
      let movedOutCount = 0;
      for (let region = 0; region < PHASE45_REGION_COUNT; region += 1) {
        const base = region * PHASE45_REGION_STRIDE;
        const splatCount = values[base];
        if (splatCount > 0) populatedRegionCount += 1;
        provisionalQuota += values[base + 9];
        lowUtilityCount += values[base + 10];
        locallyProtectedCount += values[base + 11];
        supportSafeCount += values[base + 21];
        donorEligibleCount += values[base + 22];
        movedOutCount += values[base + 23];
        regions.push({
          region,
          x: region % PHASE45_REGION_GRID,
          y: Math.floor(region / PHASE45_REGION_GRID),
          splat_count: splatCount,
          multiscale_structure_energy: {
            mean: decodeEnergy(values[base + 1]) / Math.max(1, splatCount),
            maximum: decodeEnergy(values[base + 2]),
          },
          residual: { mean: decodeResidual(values[base + 3]) / Math.max(1, splatCount), maximum: decodeResidual(values[base + 4]) },
          influence: { mean: decodeNormalized(values[base + 5]) / Math.max(1, splatCount), maximum: decodeNormalized(values[base + 6]) },
          support: { mean: decodeNormalized(values[base + 7]) / Math.max(1, splatCount), maximum: decodeNormalized(values[base + 8]) },
          provisional_quota: values[base + 9],
          low_utility_count: values[base + 10],
          locally_protected_count: values[base + 11],
          utility_histogram: Array.from(values.slice(base + 12, base + 20)),
          donor_cutoff_bin: values[base + 20] & 7,
          donor_cutoff_fraction: (values[base + 20] >>> 8) / 65535,
          support_safe_count: values[base + 21],
          donor_eligible_count: values[base + 22],
          moved_out_count: values[base + 23],
        });
      }
      return {
        grid: [PHASE45_REGION_GRID, PHASE45_REGION_GRID],
        region_count: PHASE45_REGION_COUNT,
        populated_region_count: populatedRegionCount,
        current_splat_count: currentSplatCount,
        provisional_quota: provisionalQuota,
        low_utility_count: lowUtilityCount,
        locally_protected_count: locallyProtectedCount,
        support_safe_count: supportSafeCount,
        donor_eligible_count: donorEligibleCount,
        moved_out_count: movedOutCount,
        regions,
      };
    } finally {
      readBuffer.destroy();
    }
  }

  async readDensityCounters() {
    if (!this.trainState) return null;
    const counterCount = DENSITY_EVENT_SLOTS;
    const bytes = counterCount * 4;
    const readBuffer = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(
        this.trainState.densityControlBuffer,
        this.trainState.capacity * 2 * 4,
        readBuffer,
        0,
        bytes,
      );
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();
      return {
        adc_duplicate: values[0],
        adc_split: values[1],
        adc_recycle: values[7],
        mcmc_reseed: values[2],
        mcmc_teleport: values[3],
        inactive_reused: values[4],
        opacity_reset: values[5],
        prune: values[6],
        importance_protected: values[8],
        adc_eligible: values[9],
        adc_fallback: values[10],
        structure_guided: values[11],
        nonfinite_stats: values[12],
        adc_low_to_high: values[13],
        adc_high_to_low: values[14],
        adc_same_band: values[15],
        source_claim_conflicts: values[16],
        source_claims: values[17],
        growth_eligible_sources: values[18],
        tilt_risk_candidates: values[19],
        tilt_true_splits: values[20],
        tilt_opacity_saturations: values[21],
      };
    } finally {
      readBuffer.destroy();
    }
  }

  async readImportanceSummary(count) {
    if (!this.trainState?.statsBuffer || count <= 0) return null;
    await this.syncPackedStats("export", count);
    const bytes = count * 4 * 4;
    const readBuffer = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.statsBuffer, this.trainState.capacity * 4 * 4, readBuffer, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();
      const coverage = [];
      const influence = [];
      const residual = [];
      let nonfiniteCount = 0;
      for (let i = 0; i < values.length; i += 4) {
        if (![values[i], values[i + 1], values[i + 2], values[i + 3]].every(Number.isFinite)) nonfiniteCount += 1;
        const observed = Number.isFinite(values[i]) ? Math.max(0, values[i]) : 0;
        const integratedInfluence = Number.isFinite(values[i + 1]) ? Math.max(0, values[i + 1]) : 0;
        const residualMass = Number.isFinite(values[i + 2]) ? Math.max(0, values[i + 2]) : 0;
        coverage.push(observed);
        influence.push(integratedInfluence);
        residual.push(residualMass / Math.max(observed, 1));
      }
      const summarize = (items) => {
        const sorted = items.sort((a, b) => a - b);
        const total = sorted.reduce((sum, value) => sum + value, 0);
        return {
          mean: total / Math.max(1, sorted.length),
          p10: percentileSorted(sorted, 0.1),
          median: percentileSorted(sorted, 0.5),
          p90: percentileSorted(sorted, 0.9),
          maximum: sorted[sorted.length - 1] ?? null,
        };
      };
      return { count, nonfinite_count: nonfiniteCount, coverage: summarize(coverage), influence: summarize(influence), residual: summarize(residual) };
    } finally {
      readBuffer.destroy();
    }
  }

  trainingBuffers() {
    if (!this.trainState) return [];
    return [...new Set([
      this.trainState.configBuffer,
      this.trainState.presentConfigBuffer,
      this.trainState.packedStatsConfigBuffer,
      this.trainState.targetBuffer,
      this.trainState.coarseTargetBuffer,
      this.trainState.midTargetBuffer,
      this.trainState.targetAlphaBuffer,
      this.trainState.coarseTargetAlphaBuffer,
      this.trainState.midTargetAlphaBuffer,
      this.trainState.errorMapBuffer,
      this.trainState.statsBuffer,
      this.trainState.densityControlBuffer,
      this.trainState.tileCountsBuffer,
      this.trainState.tileOffsetsBuffer,
      this.trainState.tileCursorsBuffer,
      this.trainState.tileIndicesBuffer,
      this.trainState.tileControlBuffer,
      this.trainState.pixelStateBuffer,
      this.trainState.alphaStateBuffer,
      this.trainState.lossGradientBuffer,
      this.trainState.exactGradientBuffer,
      this.trainState.exactBackwardTelemetryBuffer,
      this.trainState.exactBackwardTelemetryReadbackBuffer,
      this.trainState.ssimTileBuffer,
      this.trainState.optimizerStateBuffer,
      this.trainState.exactGradientBuffer,
      this.trainState.geometryBuffer,
      this.trainState.readbackBuffer,
      this.trainState.profileResolveBuffer,
      this.trainState.profileReadbackBuffer,
      this.trainState.growthSignalReadbackBuffer,
      ...this.trainState.xyBuffers,
      ...this.trainState.transformBuffers,
      ...this.trainState.colorBuffers,
    ].filter(Boolean))];
  }

  trainingMemorySnapshot() {
    if (!this.trainState) return { activeBytes: 0, reservedBytes: 0 };
    const reservedBytes = this.trainingBuffers().reduce((total, buffer) => total + Number(buffer.size || 0), 0);
    const capacityBuffers = [
      this.trainState.statsBuffer,
      this.trainState.densityControlBuffer,
      this.trainState.optimizerStateBuffer,
      this.trainState.geometryBuffer,
      this.trainState.readbackBuffer,
      ...this.trainState.xyBuffers,
      ...this.trainState.transformBuffers,
      ...this.trainState.colorBuffers,
    ].filter(Boolean);
    const capacityBytes = capacityBuffers.reduce((total, buffer) => total + Number(buffer.size || 0), 0);
    const tileReservedBytes = Number(this.trainState.tileIndicesBuffer?.size || 0);
    const fixedBytes = Math.max(0, reservedBytes - capacityBytes - tileReservedBytes);
    const activeRatio = Math.min(1, Math.max(0, this.trainState.count / Math.max(1, this.trainState.capacity)));
    const observedTileReferences = Number(state.metrics?.tile_counters?.total ?? this.trainState.tileIndexInitialReferences ?? 0);
    const activeTileBytes = Math.min(tileReservedBytes, Math.max(0, observedTileReferences) * 4);
    const activeBytes = Math.min(reservedBytes, fixedBytes + capacityBytes * activeRatio + activeTileBytes);
    return { activeBytes, reservedBytes };
  }

  disposeTrainState() {
    if (!this.trainState) {
      updateGpuMemoryStatus();
      return;
    }
    const buffers = this.trainingBuffers();
    for (const buffer of buffers) {
      buffer?.destroy();
    }
    this.trainState.profileQuerySet?.destroy();
    this.trainState = null;
    updateGpuMemoryStatus();
  }

  async trainStepRenderGradientGpu(image, params, learningRates, { sync = true } = {}) {
    await this.ensureRenderGradientPipelines();
    const variants = phase33Variants();
    const phase37 = phase37Variants();
    const phase38 = phase38Variants();
    const phase39 = phase39Variants();
    const phase40 = phase40Variants();
    const currentStep = (state.metrics?.steps_done || 0) + 1;
    const requestedSteps = state.metrics?.steps_requested || 1;
    const profileLabels = this.performanceProfile.timestampQuery
      ? performanceProfileLabels(currentStep, requestedSteps)
      : [];
    const profileSample = profileLabels.length > 0
      ? { step: currentStep, labels: profileLabels, queryCount: 0, stages: [] }
      : null;
    const effectiveSync = sync || Boolean(profileSample);
    const coarseStepLimit = experimentalCoarseSteps(requestedSteps, variants.coarseSteps);
    const midStepLimit = experimentalDensifySteps(requestedSteps);
    const trainingStage = curriculumTrainingStage(
      currentStep,
      requestedSteps,
      variants,
      this.trainState.coarseImage,
      this.trainState.midImage,
    );
    const useCoarse = trainingStage === "coarse";
    const useMid = trainingStage === "mid";
    const workImage = useCoarse ? this.trainState.coarseImage : useMid ? this.trainState.midImage : image;
    const targetBuffer = useCoarse
      ? this.trainState.coarseTargetBuffer
      : useMid
        ? this.trainState.midTargetBuffer
        : this.trainState.targetBuffer;
    const targetAlphaBuffer = useCoarse
      ? this.trainState.coarseTargetAlphaBuffer
      : useMid
        ? this.trainState.midTargetAlphaBuffer
        : this.trainState.targetAlphaBuffer;
    const qualityVariants = qualityRecoveryVariants();
    const useExactBackward = qualityVariants.exactBackward;
    const layerSettings = layerOptimizationSettings(
      currentStep,
      requestedSteps,
      trainingStage,
      state.metrics?.phase46_variants || phase46Variants(),
    );
    const requestedTiltStep = virtualTiltStepSpec(currentStep, trainingStage, requestedSteps);
    const tiltStep = useExactBackward && variants.ewa2x2
      ? requestedTiltStep
      : { ...requestedTiltStep, enabled: false, pitchRadians: 0, yawRadians: 0, pitchDegrees: 0, yawDegrees: 0, weight: 1 };
    const config = new Float32Array([
      workImage.width,
      workImage.height,
      params.count,
      params.bg[0],
      params.bg[1],
      params.bg[2],
      learningRates.scale,
      learningRates.maxAnisotropy,
      currentStep,
      state.metrics?.steps_requested || 1,
      learningRates.position,
      learningRates.color,
      learningRates.opacity,
      learningRates.scaleParam,
      learningRates.rotation,
      phase37.progressiveGradientLoss ? 1 : 0,
      phase37.progressiveGradientLoss ? 0.02 : 0,
      learningRates.maxAnisotropy,
      experimentalDensifySteps(state.metrics?.steps_requested || 1),
      els.tileCullingToggle.checked ? 1 : 0,
      variants.importanceRecycle ? 1 : 0,
      variants.coverageLoss ? 1 : 0,
      variants.coverageTarget,
      variants.coverageLossWeight,
      variants.coarseToFull ? 1 : 0,
      variants.structureTensor ? 1 : 0,
      variants.ewa2x2 ? 1 : 0,
      variants.importanceEma,
      this.trainState.capacity,
      phase37.gradientCoherence ? 2 : phase37.absGradient ? 1 : 0,
      phase37.adamRowAge ? 1 : 0,
      phase37.ewaGaussLegendre ? 1 : 0,
      0,
      phase38.colorTailSteps,
      0,
      0,
      0,
      0,
      1,
      coarseStepLimit,
      0,
      0,
      0,
      0,
      phase40.localColorAnchorWeight,
      els.trainLayerOrder.checked ? 1 : 0,
      phase40.alphaLoss ? phase40.alphaLossWeight : 0,
      this.geometryPrecomputeEnabled() ? 1 : 0,
      phase40.dualBackground ? 1 : 0,
      phase40.dualBackground ? phase40.dualBackgroundWeight : 0,
      learningRates.boundarySigma,
      learningRates.surfaceAnisotropy,
      layerSettings.interval,
      layerSettings.rate,
      layerSettings.due ? 1 : 0,
      0,
      tiltStep.enabled ? 1 : 0,
      tiltStep.pitchRadians,
      tiltStep.yawRadians,
      tiltStep.cameraDistance,
      tiltStep.enabled ? tiltStep.orderPenaltyWeight : 0,
      tiltStep.weight,
      learningRates.maxPlanarScale,
      0,
    ]);
    let errorScopeOpen = false;
    const profileWallStarted = profileSample ? performance.now() : 0;
    let profileEncodeSubmitMs = 0;
    let profileSyncWaitMs = 0;
    let profileReadbackMs = 0;
    let profileCandidateStats = null;
    if (effectiveSync) {
      this.device.pushErrorScope("validation");
      errorScopeOpen = true;
    }
    try {
      const encoder = this.device.createCommandEncoder();
      this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
      if (els.tileCullingToggle.checked) {
        await this.prepareTileLists(workImage, params, {
          encoder,
          profileSample,
          geometryPrecomputed: this.geometryPrecomputeEnabled(),
          writeConfig: false,
        });
      } else {
        await this.encodeGeometryPrecompute(encoder, params, profileSample);
      }
      const front = this.trainState.front;
      const renderGeometryEntry = this.geometryPrecomputeEnabled()
        ? [{ binding: 9, resource: { buffer: this.trainState.geometryBuffer } }]
        : [];
      const optimizerGeometryEntry = this.geometryPrecomputeEnabled()
        ? [{ binding: 8, resource: { buffer: this.trainState.geometryBuffer } }]
        : [];
      const optimizerStatsEntry = this.geometryPrecomputeEnabled()
        ? []
        : [{ binding: 8, resource: { buffer: this.trainState.statsBuffer } }];
      const renderChoice = this.renderStatePipelineChoice();
      const renderBindGroup = this.device.createBindGroup({
        layout: renderChoice.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
          { binding: 5, resource: { buffer: this.trainState.tileOffsetsBuffer } },
          { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
          { binding: 7, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 8, resource: { buffer: this.trainState.alphaStateBuffer } },
          ...renderGeometryEntry,
        ],
      });
      const ssimBindGroup = this.device.createBindGroup({
        layout: this.ssimTilePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: targetBuffer } },
          { binding: 2, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 3, resource: { buffer: this.trainState.ssimTileBuffer } },
          { binding: 4, resource: { buffer: targetAlphaBuffer } },
          { binding: 5, resource: { buffer: this.trainState.alphaStateBuffer } },
        ],
      });
      const optimizerPipeline = useExactBackward
        ? null
        : (phase37.parallelOptimizer ? this.parallelRenderGradientPipeline : this.renderGradientPipeline);
      const optimizerBindGroup = useExactBackward ? null : this.device.createBindGroup({
        layout: optimizerPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
          { binding: 4, resource: { buffer: targetBuffer } },
          { binding: 5, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 6, resource: { buffer: this.trainState.ssimTileBuffer } },
          { binding: 7, resource: { buffer: this.trainState.optimizerStateBuffer } },
          ...optimizerStatsEntry,
          ...(phase37.parallelOptimizer ? optimizerGeometryEntry : []),
        ],
      });
      const lossGradientBindGroup = useExactBackward ? this.device.createBindGroup({
        layout: this.lossGradientPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: targetBuffer } },
          { binding: 2, resource: { buffer: targetAlphaBuffer } },
          { binding: 3, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 4, resource: { buffer: this.trainState.ssimTileBuffer } },
          { binding: 5, resource: { buffer: this.trainState.lossGradientBuffer } },
        ],
      }) : null;
      const exactBackwardBindGroup = useExactBackward ? this.device.createBindGroup({
        layout: this.exactAlphaBackwardPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
          { binding: 4, resource: { buffer: this.trainState.tileOffsetsBuffer } },
          { binding: 5, resource: { buffer: this.trainState.tileIndicesBuffer } },
          { binding: 6, resource: { buffer: this.trainState.lossGradientBuffer } },
          { binding: 7, resource: { buffer: this.trainState.exactGradientBuffer } },
          { binding: 8, resource: { buffer: this.trainState.alphaStateBuffer } },
        ],
      }) : null;
      const virtualOrderPenaltyBindGroup = useExactBackward && tiltStep.enabled && tiltStep.orderPenaltyWeight > 0
        ? this.device.createBindGroup({
            layout: this.virtualOrderPenaltyPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.trainState.configBuffer } },
              { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
              { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
              { binding: 4, resource: { buffer: this.trainState.exactGradientBuffer } },
            ],
          })
        : null;
      const exactBackwardTelemetryBindGroup = useExactBackward && profileSample ? this.device.createBindGroup({
        layout: this.exactBackwardTelemetryPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.tileOffsetsBuffer } },
          { binding: 4, resource: { buffer: this.trainState.tileIndicesBuffer } },
          { binding: 5, resource: { buffer: this.trainState.alphaStateBuffer } },
          { binding: 6, resource: { buffer: this.trainState.exactBackwardTelemetryBuffer } },
        ],
      }) : null;
      const exactOptimizerBindGroup = useExactBackward ? this.device.createBindGroup({
        layout: this.exactOptimizerPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
          { binding: 5, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 7, resource: { buffer: this.trainState.optimizerStateBuffer } },
          ...optimizerStatsEntry,
          { binding: 9, resource: { buffer: this.trainState.exactGradientBuffer } },
          { binding: 10, resource: { buffer: this.trainState.tileControlBuffer } },
        ],
      }) : null;
      const renderPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "render"));
      renderPass.setPipeline(renderChoice.pipeline);
      renderPass.setBindGroup(0, renderBindGroup);
      if (renderChoice.cooperative) {
        renderPass.dispatchWorkgroups(Math.ceil(workImage.width / TILE_SIZE), Math.ceil(workImage.height / TILE_SIZE));
      } else {
        this.dispatchLinear(renderPass, Math.ceil((workImage.width * workImage.height) / 64));
      }
      renderPass.end();
      const ssimPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "ssim"));
      ssimPass.setPipeline(this.ssimTilePipeline);
      ssimPass.setBindGroup(0, ssimBindGroup);
      this.dispatchLinear(ssimPass, Math.ceil(workImage.width / 8) * Math.ceil(workImage.height / 8));
      ssimPass.end();
      if (useExactBackward) {
        encoder.clearBuffer(this.trainState.exactGradientBuffer);
        const lossPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "loss-gradient"));
        lossPass.setPipeline(this.lossGradientPipeline);
        lossPass.setBindGroup(0, lossGradientBindGroup);
        this.dispatchLinear(lossPass, Math.ceil((workImage.width * workImage.height) / 64));
        lossPass.end();
        if (profileSample) {
          encoder.clearBuffer(this.trainState.exactBackwardTelemetryBuffer);
          const telemetryPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "exact-backward-telemetry"));
          telemetryPass.setPipeline(this.exactBackwardTelemetryPipeline);
          telemetryPass.setBindGroup(0, exactBackwardTelemetryBindGroup);
          telemetryPass.dispatchWorkgroups(Math.ceil(workImage.width / 8), Math.ceil(workImage.height / 8));
          telemetryPass.end();
        }
        const backwardPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "exact-backward"));
        backwardPass.setPipeline(this.exactAlphaBackwardPipeline);
        backwardPass.setBindGroup(0, exactBackwardBindGroup);
        backwardPass.dispatchWorkgroups(
          Math.ceil(workImage.width / (this.quadExactBackwardEnabled ? TILE_SIZE : 8)),
          Math.ceil(workImage.height / (this.quadExactBackwardEnabled ? TILE_SIZE : 8)),
        );
        backwardPass.end();
        if (virtualOrderPenaltyBindGroup) {
          const orderPenaltyPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "virtual-order-penalty"));
          orderPenaltyPass.setPipeline(this.virtualOrderPenaltyPipeline);
          orderPenaltyPass.setBindGroup(0, virtualOrderPenaltyBindGroup);
          this.dispatchLinear(orderPenaltyPass, Math.ceil(params.count / 64));
          orderPenaltyPass.end();
        }
        const optimizerPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "optimizer"));
        optimizerPass.setPipeline(this.exactOptimizerPipeline);
        optimizerPass.setBindGroup(0, exactOptimizerBindGroup);
        this.dispatchLinear(optimizerPass, Math.ceil(params.count / 64));
        optimizerPass.end();
      } else {
        const optimizerPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "optimizer"));
        optimizerPass.setPipeline(optimizerPipeline);
        optimizerPass.setBindGroup(0, optimizerBindGroup);
        if (phase37.parallelOptimizer) this.dispatchLinear(optimizerPass, params.count);
        else optimizerPass.dispatchWorkgroups(Math.ceil(params.count / 64));
        optimizerPass.end();
      }
      if (profileSample && useExactBackward) {
        encoder.copyBufferToBuffer(
          this.trainState.exactBackwardTelemetryBuffer,
          0,
          this.trainState.exactBackwardTelemetryReadbackBuffer,
          0,
          32,
        );
      }
      if (profileSample?.queryCount) {
        encoder.resolveQuerySet(
          this.trainState.profileQuerySet,
          0,
          profileSample.queryCount,
          this.trainState.profileResolveBuffer,
          0,
        );
        encoder.copyBufferToBuffer(
          this.trainState.profileResolveBuffer,
          0,
          this.trainState.profileReadbackBuffer,
          0,
          profileSample.queryCount * 8,
        );
      }
      this.device.queue.submit([encoder.finish()]);
      if (profileSample) profileEncodeSubmitMs = performance.now() - profileWallStarted;
      this.trainState.pixelStateResolution = [workImage.width, workImage.height];
      this.trainState.pixelStateKind = trainingStage;
      if (useCoarse) this.trainState.coarseTrainingSteps += 1;
      if (useMid) this.trainState.midTrainingSteps += 1;
      if (tiltStep.enabled) {
        this.trainState.virtualTiltSteps += 1;
        this.trainState.lastVirtualTilt = { ...tiltStep, step: currentStep };
      }
      if (effectiveSync) {
        const syncStarted = profileSample ? performance.now() : 0;
        await this.device.queue.onSubmittedWorkDone();
        if (profileSample) profileSyncWaitMs = performance.now() - syncStarted;
        const error = await this.device.popErrorScope();
        errorScopeOpen = false;
        if (error) throw new Error(error.message);
      }
      if (profileSample?.queryCount) {
        const readbackStarted = performance.now();
        const bytes = profileSample.queryCount * 8;
        const readBuffer = this.trainState.profileReadbackBuffer;
        await readBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
        const timestamps = new BigUint64Array(readBuffer.getMappedRange(0, bytes)).slice();
        readBuffer.unmap();
        profileReadbackMs = performance.now() - readbackStarted;
        if (useExactBackward) {
          const telemetryStarted = performance.now();
          const telemetryReadback = this.trainState.exactBackwardTelemetryReadbackBuffer;
          await telemetryReadback.mapAsync(GPUMapMode.READ, 0, 32);
          const values = new Uint32Array(telemetryReadback.getMappedRange(0, 32)).slice();
          telemetryReadback.unmap();
          const candidateTotal = values[0];
          const suffixCandidates = values[1];
          const prefixCandidates = values[2];
          const subtileRejectedCandidates = values[3];
          profileCandidateStats = {
            candidate_total: candidateTotal,
            accepted_prefix_candidates: prefixCandidates,
            uniform_suffix_candidates: suffixCandidates,
            uniform_suffix_ratio: suffixCandidates / Math.max(1, candidateTotal),
            subtile_rejected_candidates: subtileRejectedCandidates,
            subtile_rejected_ratio_of_prefix: subtileRejectedCandidates / Math.max(1, prefixCandidates),
            workgroups: values[4],
          };
          profileReadbackMs += performance.now() - telemetryStarted;
        }
        const stages = Object.fromEntries(profileSample.stages.map((stage) => [
          stage.name,
          Number(timestamps[stage.endOfPassWriteIndex] - timestamps[stage.beginningOfPassWriteIndex]) / 1e6,
        ]));
        this.trainState.stageProfile.push({
          step: currentStep,
          labels: [...profileSample.labels],
          backend: "timestamp-query",
          resolution: [workImage.width, workImage.height],
          active_splats: params.count,
          stages_ms: stages,
          total_profiled_ms: Object.values(stages).reduce((sum, value) => sum + value, 0),
          exact_backward_candidates: profileCandidateStats,
          encode_submit_wall_ms: profileEncodeSubmitMs,
          sync_wait_wall_ms: profileSyncWaitMs,
          profile_readback_wall_ms: profileReadbackMs,
          total_wall_ms: performance.now() - profileWallStarted,
        });
      }
      this.lastTrainStats = {
        backend: "webgpu-render-gradient-adam",
        count: params.count,
        mode: useExactBackward ? "experimental-standard-alpha-exact-backward" : "experimental-render-l1-dssim-adam",
        exact_alpha_backward: useExactBackward,
        quad_exact_backward: this.quadExactBackwardEnabled,
        exact_tile_intersection: this.exactTileIntersectionEnabled,
        render_gradient_optimizer: true,
        dssim_weight: DEFAULT_DSSIM_WEIGHT,
        compact_tile_candidates: Boolean(els.tileCullingToggle.checked),
        tile_builds: this.trainState.tileBuilds,
        sgld_2d: true,
        experimental_variants: experimentalVariants(),
        training_resolution: [workImage.width, workImage.height],
        training_stage: trainingStage,
        coarse_to_full: variants.coarseToFull,
        three_stage_curriculum: variants.threeStageCurriculum,
        coarse_active: useCoarse,
        mid_active: useMid,
        coarse_steps_completed: this.trainState.coarseTrainingSteps,
        mid_steps_completed: this.trainState.midTrainingSteps,
        virtual_tilt_steps_completed: this.trainState.virtualTiltSteps,
        last_virtual_tilt: this.trainState.lastVirtualTilt,
        coarse_step_limit: coarseStepLimit,
        mid_step_limit: midStepLimit,
        mid_resolution: this.trainState.midImage ? [this.trainState.midImage.width, this.trainState.midImage.height] : null,
        coarse_resolution: this.trainState.coarseImage ? [this.trainState.coarseImage.width, this.trainState.coarseImage.height] : null,
        full_resolution: [image.width, image.height],
        adaptive_curriculum: variants.adaptiveCurriculum,
        layer_update_interval: layerSettings.interval,
        layer_update_rate: layerSettings.rate,
        layer_update_enabled: layerSettings.enabled,
        layer_update_due: layerSettings.due,
        virtual_tilt: {
          ...tiltStep,
          interval: virtualTiltVariants().interval,
        },
        coarse_transition_steps: 0,
        density_horizon: experimentalDensifySteps(state.metrics?.steps_requested || 1),
        learningRates,
        stage_profile_samples: this.trainState.stageProfile.length,
        tile_cooperative_renderer: renderChoice.cooperative,
        tile_cooperative_supported: renderChoice.supported,
        geometry_precompute_requested: this.trainState.geometryPrecomputeRequested,
        geometry_precompute_enabled: this.geometryPrecomputeEnabled(),
        geometry_precompute_reason: this.trainState.geometryPrecomputeReason,
        geometry_precompute_limits: { ...this.trainState.geometryPrecomputeLimits },
        geometry_precompute_stride_bytes: GEOMETRY_PRECOMPUTE_STRIDE_BYTES,
        sync: effectiveSync,
        updated: true,
      };
    } finally {
      if (errorScopeOpen) this.device.popErrorScope().catch(() => {});
    }
  }

  async trainStepGpu(image, params, learningRates, { sync = true } = {}) {
    if (
      !this.trainState ||
      this.trainState.width !== image.width ||
      this.trainState.height !== image.height ||
      this.trainState.capacity < params.count
    ) {
      await this.uploadTrainState(image, params);
    }
    this.trainState.count = params.count;
    return this.trainStepRenderGradientGpu(image, params, learningRates, { sync });
  }

  async readTrainedColors(params) {
    if (!this.trainState || this.trainState.count !== params.count) return;
    const front = this.trainState.front;
    const xyBytes = params.xy.byteLength;
    const transformBytes = params.count * 4 * 4;
    const colorBytes = params.count * 4 * 4;
    const totalBytes = xyBytes + transformBytes + colorBytes;
    const readBuffer = this.trainState.readbackBuffer;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.trainState.xyBuffers[front], 0, readBuffer, 0, xyBytes);
    encoder.copyBufferToBuffer(this.trainState.transformBuffers[front], 0, readBuffer, xyBytes, transformBytes);
    encoder.copyBufferToBuffer(this.trainState.colorBuffers[front], 0, readBuffer, xyBytes + transformBytes, colorBytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readBuffer.mapAsync(GPUMapMode.READ, 0, totalBytes);
      const mapped = readBuffer.getMappedRange(0, totalBytes);
      params.xy.set(new Float32Array(mapped, 0, params.xy.length));
      const transforms = new Float32Array(mapped, xyBytes, params.count * 4);
      params.detailTags = new Float32Array(params.count);
      params.depthOrder = new Float32Array(params.count);
      let detailSplatCount = 0;
      let detailAnisotropyMax = 1;
      let surfaceAnisotropyMax = 1;
      for (let i = 0; i < params.count; i += 1) {
        params.scale[i * 2] = transforms[i * 4];
        params.scale[i * 2 + 1] = transforms[i * 4 + 1];
        params.theta[i] = transforms[i * 4 + 2];
        const packedTag = transforms[i * 4 + 3];
        params.detailTags[i] = Math.floor(packedTag);
        params.depthOrder[i] = Math.max(0, Math.min(1, (packedTag - Math.floor(packedTag)) / LAYER_CODE_RANGE));
        if (Math.floor(packedTag) > 1.5) {
          detailSplatCount += 1;
          const minor = Math.max(MIN_SPLAT_SCALE, Math.min(transforms[i * 4], transforms[i * 4 + 1]));
          detailAnisotropyMax = Math.max(detailAnisotropyMax, Math.max(transforms[i * 4], transforms[i * 4 + 1]) / minor);
        } else {
          const minor = Math.max(MIN_SPLAT_SCALE, Math.min(transforms[i * 4], transforms[i * 4 + 1]));
          surfaceAnisotropyMax = Math.max(surfaceAnisotropyMax, Math.max(transforms[i * 4], transforms[i * 4 + 1]) / minor);
        }
      }
      if (state.metrics) {
        state.metrics.detail_splat_count = detailSplatCount;
        state.metrics.detail_splat_ratio = detailSplatCount / Math.max(1, params.count);
        state.metrics.detail_anisotropy_max = detailAnisotropyMax;
        state.metrics.surface_anisotropy_max = surfaceAnisotropyMax;
      }
      const colors = new Float32Array(mapped, xyBytes + transformBytes, params.count * 4);
      for (let i = 0; i < params.count; i += 1) {
        params.rgb[i * 3] = colors[i * 4];
        params.rgb[i * 3 + 1] = colors[i * 4 + 1];
        params.rgb[i * 3 + 2] = colors[i * 4 + 2];
        params.opacity[i] = colors[i * 4 + 3];
      }
      readBuffer.unmap();
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
    }
  }
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
    };
  }
  let alphaMaximum = 0;
  let premultipliedMaximum = 0;
  let premultipliedTotal = 0;
  let channels = 0;
  for (let i = 0; i < a.length; i += 4) {
    alphaMaximum = Math.max(alphaMaximum, Math.abs(a[i + 3] - b[i + 3]));
    for (let channel = 0; channel < 3; channel += 1) {
      const source = Math.round(a[i + channel] * a[i + 3] / 255);
      const decoded = Math.round(b[i + channel] * b[i + 3] / 255);
      const delta = Math.abs(source - decoded);
      premultipliedMaximum = Math.max(premultipliedMaximum, delta);
      premultipliedTotal += delta;
      channels += 1;
    }
  }
  const displayEquivalent = alphaMaximum === 0 && premultipliedMaximum <= 1;
  return {
    exact: displayEquivalent,
    display_equivalent: displayEquivalent,
    max_abs: raw.max_abs,
    mean_abs: raw.mean_abs,
    raw_exact: raw.exact,
    alpha_max_abs: alphaMaximum,
    premultiplied_max_abs: premultipliedMaximum,
    premultiplied_mean_abs: premultipliedTotal / Math.max(1, channels),
  };
}

function layerTelemetryEnabled() {
  return QA_RUNTIME_ENABLED && globalThis.__flatPhotoLayerTelemetry === true;
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
    layers[index] = Math.max(0, Math.min(1, (packed - Math.floor(packed)) / LAYER_CODE_RANGE));
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

async function recordLayerTelemetry(step) {
  if (!layerTelemetryEnabled() || !state.webgpu.renderer?.trainState) return;
  const previousRecord = state.metrics.layer_telemetry[state.metrics.layer_telemetry.length - 1];
  if (previousRecord?.step === step) return;
  const count = state.params.count;
  const [snapshot, geometry] = await Promise.all([
    state.webgpu.renderer.capturePresentedStateRgba(),
    state.webgpu.renderer.readLayerTelemetryGeometry(count),
  ]);
  if (!snapshot || !geometry) return;
  const depth = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const packed = geometry.transform[index * 4 + 3];
    depth[index] = Math.max(0, Math.min(1, (packed - Math.floor(packed)) / LAYER_CODE_RANGE));
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
  const recycle = lastEventAtStep(state.metrics.adc_reset_events, step);
  const previousPairState = recycle ? null : previous;
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
  if (recycle) triggers.push("recycle");
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

function packTransforms(params) {
  const transform = new Float32Array(params.count * 4);
  for (let i = 0; i < params.count; i += 1) {
    transform[i * 4] = params.scale[i * 2];
    transform[i * 4 + 1] = params.scale[i * 2 + 1];
    transform[i * 4 + 2] = params.theta?.[i] || 0;
    const tag = Math.max(1, Math.floor(params.detailTags?.[i] || 1));
    const layer = params.layerOrderEnabled
      ? Math.max(0, Math.min(1, params.depthOrder?.[i] ?? (1 - i / Math.max(1, params.count - 1))))
      : 0;
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
  return ["frame", "10", "metrics"].includes(els.previewRefresh.value) ? els.previewRefresh.value : "10";
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

function setInputControlsDisabled(disabled) {
  for (const element of [
    els.fileInput,
    els.algorithmSelect,
    els.trainSize,
    els.initialSplatCount,
    els.finalSplatCount,
    els.capacityMode,
    els.stepCount,
    els.previewRefresh,
    els.tileCullingToggle,
    els.trainLayerOrder,
    els.layerUpdateInterval,
    els.positionLearningRate,
    els.colorLearningRate,
    els.opacityLearningRate,
    els.alphaLossWeight,
    els.dualBackgroundToggle,
    els.stageAwareGrowth,
    els.dualBackgroundWeight,
    els.scaleLearningRate,
    els.rotationLearningRate,
    els.thetaAlignRate,
    els.maxAnisotropy,
    els.maxPlanarScale,
    els.boundarySigma,
    els.detailCoherence,
    els.adcSplitInterval,
    els.adcResetInterval,
    els.densifyInterval,
    els.growthPercentage,
    els.growthSignalThreshold,
    els.adcSplitSignalThreshold,
    els.adcSplitResidualThreshold,
    els.adcRecyclePercentage,
    els.adcLateRecyclePercentage,
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
  if (!disabled) syncLayerOrderDependency();
}

function syncLayerOrderDependency() {
  if (els.trainLayerOrder.checked) els.tileCullingToggle.checked = true;
  els.tileCullingToggle.disabled = state.running || els.trainLayerOrder.checked;
  els.layerUpdateInterval.disabled = state.running || !els.trainLayerOrder.checked;
}

function pausedRuntimeControls() {
  return [
    els.previewRefresh,
    els.tileCullingToggle,
    els.positionLearningRate,
    els.colorLearningRate,
    els.opacityLearningRate,
    els.alphaLossWeight,
    els.dualBackgroundToggle,
    els.dualBackgroundWeight,
    els.scaleLearningRate,
    els.rotationLearningRate,
    els.thetaAlignRate,
    els.maxAnisotropy,
    els.maxPlanarScale,
    els.boundarySigma,
    els.detailCoherence,
    els.adcSplitInterval,
    els.adcResetInterval,
    els.densifyInterval,
    els.growthPercentage,
    els.growthSignalThreshold,
    els.adcSplitSignalThreshold,
    els.adcSplitResidualThreshold,
    els.adcRecyclePercentage,
    els.adcLateRecyclePercentage,
  ];
}

function setPausedRuntimeControlsEnabled(enabled) {
  for (const element of pausedRuntimeControls()) element.disabled = !enabled;
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
  state.metrics.adc_controls.split_signal_threshold = growth.adcSplitSignalThreshold;
  state.metrics.adc_controls.split_residual_threshold = growth.adcSplitResidualThreshold;
  state.metrics.adc_controls.recycle_percentage = growth.adcRecycleRate * 100;
  state.metrics.adc_controls.late_recycle_percentage = growth.adcLateRecycleRate * 100;
  state.metrics.learning_rates = {
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
  state.metrics.dual_background = phase40.dualBackground;
  state.metrics.dual_background_weight = phase40.dualBackgroundWeight;
  const data = document.documentElement.dataset;
  data.runtimeAppliedRevision = String(state.runtimeSettingsRevision);
  data.runtimeAppliedPosition = String(learningRates.position);
  data.runtimeAppliedMaxPlanarScale = String(learningRates.maxPlanarScale);
  data.runtimeAppliedBoundarySigma = String(learningRates.boundarySigma);
  data.runtimeAppliedAlphaLossWeight = String(state.metrics.alpha_loss_weight);
  data.runtimeAppliedDualBackground = String(state.metrics.dual_background);
  data.runtimeAppliedDualBackgroundWeight = String(state.metrics.dual_background_weight);
  data.runtimeAppliedPreviewRefresh = previewRefresh;
}

async function presentTrainingPreview(step) {
  if (layerTelemetryEnabled()) await recordLayerTelemetry(step);
  let presented = false;
  if (els.outsidePreviewToggle.checked && !state.running && state.webgpu.renderer) {
    const buffers = state.webgpu.renderer.currentTrainBuffers(state.params);
    await state.webgpu.renderer.render(state.image, state.params, buffers);
    presented = true;
  } else {
    state.previewPadding = previewPaddingSpec(state.image, state.params, false);
    presented = state.webgpu.renderer?.presentTrainState(state.image);
  }
  if (!presented) return false;
  if (state.previewMode === "splats") showCanvas("gpu");
  state.metrics.preview_frames += 1;
  state.metrics.last_preview_step = step;
  els.stepText.textContent = `${step} / ${state.metrics.steps_requested}`;
  publishState();
  return true;
}

async function resolveTileOverflowRetry(parameterHashBefore = null) {
  const renderer = state.webgpu.renderer;
  if (!els.tileCullingToggle.checked || !renderer?.trainState?.tileReady) return 0;
  const tileCounters = await renderer.readTileCounters();
  if (!tileCounters) return 0;
  state.metrics.tile_counters = tileCounters;
  const noopSteps = Math.max(0, Math.floor(Number(tileCounters.noop_steps) || 0));
  if (tileCounters.overflow === 0 && noopSteps === 0) return 0;

  if (parameterHashBefore !== null) {
    const parameterHashAfter = await renderer.hashTrainParameters(state.params);
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
    ? await renderer.growTileIndexCapacity(tileCounters.total)
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
  await renderer.clearTileNoopCounter();
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

async function updatePreview(step, final = false, { present = true } = {}) {
  const backend = selectedBackend();
  if (!backend.startsWith("webgpu")) throw new Error(`WebGPU required: ${state.webgpu.reason}`);
  const safety = safetyFailure(computeBudgetFor(Number(els.trainSize.value), state.params.count, state.metrics?.steps_requested || 1), "metrics");
  if (safety) {
    setSafetyStop(safety);
    throw new Error(`${safety.reason}: metrics/readback skipped before budget overflow`);
  }
  if (layerTelemetryEnabled()) await recordLayerTelemetry(step);
  if (els.tileCullingToggle.checked && state.webgpu.renderer?.trainState) {
    await state.webgpu.renderer.prepareTileLists(state.image, state.params, { sync: true });
    let tileCounters = await state.webgpu.renderer.readTileCounters();
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
      ? await state.webgpu.renderer.growTileIndexCapacity(tileCounters.total, { proactive: tileCounters.overflow === 0 })
      : false;
    if (expandedTileReserve) {
      const previousCapacity = tileCounters.capacity;
      await state.webgpu.renderer.prepareTileLists(state.image, state.params, { sync: true });
      tileCounters = await state.webgpu.renderer.readTileCounters();
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
  const trainBuffers = state.webgpu.renderer?.currentTrainBuffers(state.params);
  if (final && state.webgpu.renderer?.trainState) {
    await state.webgpu.renderer.readTrainedColors(state.params);
    assertFiniteParams(state.params, "final-readback");
    if (final) {
      state.metrics.thin_line_metrics = computeThinLineMetrics(state.image, state.params);
      state.metrics.tilt_risk = summarizeTiltRisk(state.params, state.image);
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
      await state.webgpu.renderer.render(state.image, state.params, trainBuffers);
    }
    metrics = await state.webgpu.renderer.computeMetrics(state.image, state.params, trainBuffers);
    const restoreTrainingStage = !final
      ? state.webgpu.renderer.lastTrainStats?.training_stage
      : "full";
    const restoreStageImage = restoreTrainingStage === "coarse"
      ? state.webgpu.renderer.trainState?.coarseImage
      : restoreTrainingStage === "mid"
        ? state.webgpu.renderer.trainState?.midImage
        : null;
    if (restoreStageImage) {
      if (els.tileCullingToggle.checked) {
        await state.webgpu.renderer.prepareTileLists(restoreStageImage, state.params, { sync: true });
      }
      await state.webgpu.renderer.refreshRenderState(restoreStageImage, state.params);
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
      state.metrics.overlap_diagnostics = await state.webgpu.renderer.computeOverlapDiagnostics(state.image, state.params);
      const trainingFrame = await state.webgpu.renderer.capturePresentedStateRgba();
      if (!trainingFrame || trainingFrame.width !== state.image.width || trainingFrame.height !== state.image.height) {
        throw new Error("Final training RGBA readback has the wrong resolution.");
      }
      const standaloneRgba = await state.webgpu.renderer.captureFrameRgba(state.image, state.params);
      state.metrics.render_surface_parity = {
        ...displayRgbaParity(trainingFrame.rgba, standaloneRgba),
        source: "training-pixel-state-vs-standalone-rgba",
        width: trainingFrame.width,
        height: trainingFrame.height,
      };
      state.metrics.importance_stats = await state.webgpu.renderer.readImportanceSummary(state.params.count);
      if (state.metrics.importance_stats?.nonfinite_count > 0) {
        throw runtimeSafetyError("safety_stop_nonfinite_importance", "final-importance-readback", {
          nonfinite_stats: state.metrics.importance_stats.nonfinite_count,
        });
      }
    }
    const densityCounters = await state.webgpu.renderer.readDensityCounters();
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
  } catch (error) {
    if (error.safetyStop) throw error;
    state.lastGpuLoss = null;
    state.metrics.webgpu_compute_loss = false;
    state.metrics.webgpu_compute_error = error.message;
    throw new Error(`WebGPU preview/metrics failed: ${error.message}`);
  }
  state.metrics.losses.push(metrics.loss);
  state.metrics.alpha_losses.push(metrics.alphaL1);
  state.metrics.alpha_ssim.push(metrics.alphaSsim);
  state.metrics.objective_losses.push(metrics.objectiveLoss);
  state.metrics.ssim.push(metrics.windowedSsim);
  state.metrics.global_ssim.push(metrics.ssim);
  state.metrics.windowed_ssim.push(metrics.windowedSsim);
  state.metrics.regional_ssim_p10.push(metrics.regionalSsim.p10);
  state.metrics.final_l1 = metrics.loss;
  state.metrics.final_alpha_l1 = metrics.alphaL1;
  state.metrics.final_alpha_ssim = metrics.alphaSsim;
  state.metrics.final_alpha_objective = metrics.alphaObjective;
  state.metrics.final_objective_loss = metrics.objectiveLoss;
  state.metrics.final_ssim = metrics.windowedSsim;
  state.metrics.final_global_ssim = metrics.ssim;
  state.metrics.final_windowed_ssim = metrics.windowedSsim;
  state.metrics.final_regional_ssim = metrics.regionalSsim;
  state.metrics.final_high_frequency = metrics.highFrequency;
  state.metrics.coverage_stats = metrics.coverage ? { ...metrics.coverage, step } : null;
  state.metrics.coverage_revision = state.metrics.params_revision ?? 0;
  if (state.metrics.initial_ssim !== null) {
    const delta = metrics.windowedSsim - state.metrics.initial_ssim;
    state.metrics.ssim_trend = delta > 0.0005 ? "up" : delta < -0.0005 ? "down" : "flat";
  }
  if (state.metrics.initial_global_ssim !== null) {
    const delta = metrics.ssim - state.metrics.initial_global_ssim;
    state.metrics.global_ssim_trend = delta > 0.0005 ? "up" : delta < -0.0005 ? "down" : "flat";
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
  els.ssimText.textContent = metrics.ssim.toFixed(6);
  els.regionalSsimText.textContent = metrics.regionalSsim.p10.toFixed(6);
  els.boundaryText.textContent = boundary ? `${boundary.count} / ${boundary.maxLeak.toFixed(6)}` : "-";
  const coveragePairReady = Boolean(boundary && Number.isFinite(metrics.coverage?.background_exposure_count));
  els.backgroundPixelText.textContent = coveragePairReady
    ? `${(metrics.coverage.background_exposure_ratio * 100).toFixed(2)}%`
    : "-";
  els.outsideSplatText.textContent = coveragePairReady ? outsideRender.count.toLocaleString() : "-";
  renderSplatInspector();
  publishState();
  if (final) {
    const label = state.metrics?.stopped ? "stopped" : "finished";
    const worst = metrics.regionalSsim.worst_region;
    log(`${label} loss=${metrics.loss.toFixed(6)} alpha_l1=${metrics.alphaL1.toFixed(6)} alpha_ssim=${metrics.alphaSsim.toFixed(6)} objective=${metrics.objectiveLoss.toFixed(6)} global_ssim=${metrics.ssim.toFixed(6)} windowed_ssim=${metrics.windowedSsim.toFixed(6)} local_p10=${metrics.regionalSsim.p10.toFixed(6)} worst_region=${worst?.column ?? "-"},${worst?.row ?? "-"}`);
  }
  return metrics.loss;
}

async function startTraining() {
  return selectedAlgorithm().train();
}

async function trainPlanarGaussian() {
  if (!state.image || state.running || state.previewRefreshPending) return;
  const algorithm = selectedAlgorithm();
  if (algorithm.id !== PLANAR_GAUSSIAN_ALGORITHM_ID) throw new Error(`Unsupported training algorithm: ${algorithm.id}`);
  destroyTiltViewer({ restoreCanvas: true });
  activateDetailTab("training");
  if (!state.webgpu.supported || !state.webgpu.renderer) {
    setStatus("error");
    els.backendText.textContent = "webgpu required";
    log(`WebGPU required; training not started: ${state.webgpu.reason}`);
    return;
  }
  const requestedSteps = normalizeStepInteger(els.stepCount.value, {
    min: LIMITS.stepsMin,
    max: LIMITS.stepsMax,
    fallback: 7000,
  });
  if (requestedSteps > HIGH_ITERATION_CONFIRM) {
    const ok = window.confirm(`Run ${requestedSteps.toLocaleString()} iterations? This may take a long time.`);
    if (!ok) {
      setStatus("idle");
      log(`training cancelled before start: ${requestedSteps} iterations`);
      return;
    }
  }
  const resizedForTraining = resizeLoadedImageToMaxSide(Number(els.trainSize.value));
  if (resizedForTraining) {
    syncTrainSizeUi();
    log(`applied training image size ${state.image.width}x${state.image.height}`);
  }
  clearSafetyStop();
  const preflight = updateMemoryRecommendation();
  const preflightFailure = safetyFailure(preflight, "start");
  const wantsHighCapacityProbe =
    els.capacityMode.value === "auto-probe" &&
    Number(els.finalSplatCount.value) > CAPACITY_PROBE_FAST_PATH_MAX;
  if (preflightFailure && (!wantsHighCapacityProbe || preflightFailure.reason === "safety_stop_hard_limit")) {
    setSafetyStop(preflightFailure);
    setStatus("safety stopped");
    updateDownloads(false);
    return;
  }
  if (preflightFailure) log(`Auto probe will search below the rejected ${els.finalSplatCount.value} splat request.`);
  state.running = true;
  state.previewGeneration += 1;
  state.paused = false;
  state.runtimeSettingsRevision = 0;
  state.layerTelemetryState = null;
  state.stopRequested = false;
  clearSplatAdjustmentBaseline();
  updateDownloads(false);
  setStatus("running");
  els.startButton.disabled = true;
  els.stopButton.disabled = false;
  setInputControlsDisabled(true);
  setPausedRuntimeControlsEnabled(false);

  const initialCount = normalizeUiSplatCount(
    els.initialSplatCount.value,
    DEFAULT_INITIAL_SPLATS,
    CAPACITY_PROBE_FAST_PATH_MAX,
  );
  let finalCount = Math.max(initialCount, normalizeUiSplatCount(els.finalSplatCount.value, DEFAULT_FINAL_SPLATS));
  const steps = requestedSteps;
  let learningRates = selectedLearningRates();
  let previewRefresh = selectedPreviewRefresh();
  const runStageAwareGrowth = phase39Variants().stageAwareGrowth;
  const runLayerSettings = phase46Variants();
  const gpuDensifyEnabled = true;
  const gpuRelocationEnabled = true;
  const useAutoCapacityProbe = els.capacityMode.value === "auto-probe" && finalCount > CAPACITY_PROBE_FAST_PATH_MAX;
  state.capacityProbe = {
    status: useAutoCapacityProbe ? "ready" : finalCount <= CAPACITY_PROBE_FAST_PATH_MAX ? "fast" : "manual",
    requested: finalCount,
    selected: useAutoCapacityProbe ? 0 : finalCount,
    attempts: [],
    fastPath: finalCount <= CAPACITY_PROBE_FAST_PATH_MAX,
  };
  els.initialSplatCount.value = String(initialCount);
  els.finalSplatCount.value = String(finalCount);
  els.stepCount.value = String(steps);
  els.previewRefresh.value = previewRefresh;
  els.layerUpdateInterval.value = String(runLayerSettings.layerUpdateInterval);
  els.positionLearningRate.value = String(learningRates.position);
  els.colorLearningRate.value = String(learningRates.color);
  els.opacityLearningRate.value = String(learningRates.opacity);
  els.scaleLearningRate.value = String(learningRates.scaleParam);
  els.rotationLearningRate.value = String(learningRates.rotation);
  els.thetaAlignRate.value = String(learningRates.thetaAlign);
  els.maxAnisotropy.value = String(learningRates.maxAnisotropy);
  els.maxPlanarScale.value = String(learningRates.maxPlanarScale);
  els.boundarySigma.value = String(learningRates.boundarySigma);
  els.detailCoherence.value = String(learningRates.detailCoherence);
  const budget = updateMemoryRecommendation();
  if (budget.overBudget) {
    log(`settings exceed safety budget ${budget.estimatedMB}MB > ${budget.budgetMB}MB; recommended ${budget.recommendedTrainSize}px ${budget.recommendedFinalSplats} splats`);
  }
  const initialization = "image-rgb";
  state.params = algorithm.initialize(state.image, initialCount);
  if (!els.trainLayerOrder.checked) state.params.depthOrder.fill(0);
  state.previewMode = "splats";
  fitCanvases(state.image.width, state.image.height);
  updatePreviewModeControls();
  state.metrics = {
    format: PRODUCT_FORMAT,
    version: 1,
    input_name: state.image.fileName,
    image_size: [state.image.width, state.image.height],
    input_original_size: [state.image.originalWidth, state.image.originalHeight],
    resize_mode: state.image.resizeMode,
    resize_scale: state.image.resizeScale,
    train_size: Math.max(state.image.width, state.image.height),
    algorithm: algorithm.id,
    algorithm_label: algorithm.label,
    algorithm_backend: algorithm.backend,
    algorithm_capabilities: { ...algorithm.capabilities },
    initialization,
    initial_splats: initialCount,
    final_splats: finalCount,
    num_gaussians: initialCount,
    default_output: "ply",
    steps_requested: steps,
    steps_done: 0,
    preview_refresh: previewRefresh,
    preview_frames: 0,
    preview_resolution_restores: 0,
    last_preview_step: null,
    params_revision: 0,
    coverage_revision: null,
    post_train_adjustments: null,
    runtime_settings_revision: 0,
    lr_scale: learningRates.scale,
    learning_rates: {
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
      trainLayerOrder: Boolean(els.trainLayerOrder.checked),
      layerUpdateInterval: runLayerSettings.layerUpdateInterval,
    },
    initial_param_snapshot: snapshotParams(state.params),
    param_delta: null,
    train_sync_interval: trainSyncInterval(),
    backend: selectedBackend(),
    webgpu_supported: state.webgpu.supported,
    webgpu_reason: state.webgpu.reason,
    gpu_budget: {
      exact_vram: false,
      budget_mb: budget.budgetMB,
      estimated_mb: budget.estimatedMB,
      headroom_mb: budget.headroomMB,
      memory_hint_source: budget.memoryHintSource,
      memory_hint_mb: budget.memoryHintMB,
      limiter_train_size: budget.limiterTrainSize,
      limiter_final_splats: budget.limiterFinalSplats,
      recommended_train_size: budget.recommendedTrainSize,
      recommended_final_splats: budget.recommendedFinalSplats,
      metric_interval: budget.metricInterval,
    },
    safety_stop: null,
    webgpu_compute_loss: false,
    webgpu_train_executed: false,
    webgpu_train_update: false,
    webgpu_densify_requested: gpuDensifyEnabled,
    webgpu_relocation_requested: gpuRelocationEnabled,
    webgpu_relocation: false,
    webgpu_refine_requested: gpuRelocationEnabled,
    webgpu_refine: false,
    loss_backend: "webgpu-compute",
    initial_l1: null,
    final_l1: null,
    initial_alpha_l1: null,
    final_alpha_l1: null,
    initial_alpha_ssim: null,
    final_alpha_ssim: null,
    initial_alpha_objective: null,
    final_alpha_objective: null,
    initial_objective_loss: null,
    final_objective_loss: null,
    initial_ssim: null,
    final_ssim: null,
    initial_global_ssim: null,
    final_global_ssim: null,
    initial_windowed_ssim: null,
    final_windowed_ssim: null,
    initial_regional_ssim: null,
    final_regional_ssim: null,
    initial_high_frequency: null,
    final_high_frequency: null,
    boundary_sigma: learningRates.boundarySigma,
    boundary_leak_count: null,
    boundary_max_leak: null,
    shape_stats: null,
    final_readback_step: null,
    scale_histogram: null,
    tiny_splat_count: null,
    tiny_splat_ratio: null,
    boundary_tiny_splat_count: null,
    boundary_tiny_splat_ratio: null,
    interior_tiny_splat_count: null,
    interior_tiny_splat_ratio: null,
    anisotropy_ratio: null,
    detail_splat_count: null,
    detail_splat_ratio: null,
    detail_anisotropy_max: null,
    surface_anisotropy_max: null,
    thin_line_metrics: null,
    fusion_events: emptyFusionEvents(),
    fusion_refine_events: [],
    webgpu_relocation_events: [],
    webgpu_refine_events: [],
    adc_reset_events: [],
    density_counters: null,
    last_density_counters: null,
    render_aware_density: true,
    weighted_mass_redistribution: true,
    sgld_2d: true,
    experimental_variants: experimentalVariants(),
    phase33_variants: phase33Variants(),
    phase37_variants: phase37Variants(),
    phase38_variants: phase38Variants(),
    phase39_variants: { ...phase39Variants(), stageAwareGrowth: runStageAwareGrowth },
    phase40_variants: phase40Variants(),
    phase45_variants: phase45Variants(),
    phase46_variants: runLayerSettings,
    phase45_region_report: null,
    overlap_diagnostics: null,
    performance_trace: [],
    performance_profile_schedule: Object.fromEntries(performanceProfileSchedule(steps)),
    importance_stats: null,
    coverage_stats: null,
    density_gpu_ms: 0,
    relocation_gpu_ms: 0,
    adc_reset_gpu_ms: 0,
    density_horizon: experimentalDensifySteps(steps),
    post_density_annealing: true,
    tile_culling_enabled: Boolean(els.tileCullingToggle.checked),
    tile_retry_steps: 0,
    tile_retry_events: [],
    tile_retry_parameter_hash: null,
    qa_tile_index_capacity: qaTileIndexCapacityOverride(),
    train_layer_order: Boolean(els.trainLayerOrder.checked),
    layer_update_interval: runLayerSettings.layerUpdateInterval,
    layer_update_rate: runLayerSettings.layerUpdateRate,
    layer_stage_aware_rate: runLayerSettings.stageAwareRate,
    layer_freeze_fraction: runLayerSettings.freezeFraction,
    layer_update_count: 0,
    layer_update_first_steps: [],
    layer_update_last_step: null,
    layer_telemetry_enabled: layerTelemetryEnabled(),
    layer_telemetry: [],
    experimental_prefix_preserved: true,
    trend_checkpoints: [],
    ssim_trend: "",
    global_ssim_trend: "",
    losses: [],
    alpha_losses: [],
    alpha_ssim: [],
    objective_losses: [],
    ssim: [],
    global_ssim: [],
    windowed_ssim: [],
    regional_ssim_p10: [],
    densify_events: [],
    growth_schedule: {
      mode: runStageAwareGrowth
        ? "stage-aware-percentage-cap"
        : "threshold-percentage-cap",
      final_is_cap: true,
      percentage: phase39Variants().growthFraction * 100,
      signal_threshold: phase39Variants().growthSignalThreshold,
      stage_aware: runStageAwareGrowth,
      detail_reserve_percentage: STAGE_AWARE_GROWTH_RESERVE * 100,
      density_stage_target: runStageAwareGrowth
        ? Math.min(finalCount, state.params.count + Math.round((finalCount - state.params.count) * (1 - STAGE_AWARE_GROWTH_RESERVE)))
        : null,
      growth_stage_target: finalCount,
      cap_reached_step: state.params.count >= finalCount ? 0 : null,
      training_early_stop: false,
      threshold_skips: 0,
    },
    adc_controls: {
      split_signal_threshold: phase39Variants().adcSplitSignalThreshold,
      split_residual_threshold: phase39Variants().adcSplitResidualThreshold,
      recycle_percentage: phase39Variants().adcRecycleRate * 100,
      late_recycle_percentage: phase39Variants().adcLateRecycleRate * 100,
    },
    stopped: false,
    started_at: new Date().toISOString(),
  };
  syncRuntimeMetrics(learningRates, previewRefresh);
  publishState();

  let trainingError = null;
  try {
    let allocationResult = null;
    if (useAutoCapacityProbe) {
      allocationResult = await state.webgpu.renderer.probeTrainingCapacity(state.image, state.params, finalCount);
      finalCount = allocationResult.capacity;
      els.finalSplatCount.value = String(finalCount);
      state.metrics.final_splats = finalCount;
      state.metrics.capacity_probe = {
        mode: "auto-probe",
        requested: state.capacityProbe.requested,
        selected: finalCount,
        fast_path: allocationResult.fastPath,
        attempts: allocationResult.attempts,
      };
    } else {
      await state.webgpu.renderer.uploadTrainState(
        state.image,
        state.params,
        finalCount,
        { verifyAllocation: finalCount > CAPACITY_PROBE_FAST_PATH_MAX },
      );
      state.metrics.capacity_probe = {
        mode: els.capacityMode.value,
        requested: finalCount,
        selected: finalCount,
        fast_path: finalCount <= CAPACITY_PROBE_FAST_PATH_MAX,
        attempts: [],
      };
    }
    const allocatedMemory = state.webgpu.renderer.trainingMemorySnapshot();
    state.metrics.gpu_training_memory = {
      accounting: "app-created-buffers",
      exact_device_vram: false,
      active_bytes_at_start: Math.round(allocatedMemory.activeBytes),
      reserved_bytes: Math.round(allocatedMemory.reservedBytes),
    };
    log(`GPU training buffers reserved ${formatMB(allocatedMemory.reservedBytes)}; active estimate ${formatMB(allocatedMemory.activeBytes)}`);
    await updatePreview(0, false);
    state.metrics.initial_l1 = state.metrics.final_l1;
    state.metrics.initial_alpha_l1 = state.metrics.final_alpha_l1;
    state.metrics.initial_alpha_ssim = state.metrics.final_alpha_ssim;
    state.metrics.initial_alpha_objective = state.metrics.final_alpha_objective;
    state.metrics.initial_objective_loss = state.metrics.final_objective_loss;
    state.metrics.initial_ssim = state.metrics.final_ssim;
    state.metrics.initial_global_ssim = state.metrics.final_global_ssim;
    state.metrics.initial_windowed_ssim = state.metrics.final_windowed_ssim;
    state.metrics.initial_regional_ssim = state.metrics.final_regional_ssim;
    state.metrics.initial_high_frequency = state.metrics.final_high_frequency;
    log(
      `training start algorithm=${state.metrics.algorithm} backend=${state.metrics.backend} initial_loss=${state.metrics.initial_l1.toFixed(6)} initial_alpha_l1=${state.metrics.initial_alpha_l1.toFixed(6)} initial_objective=${state.metrics.initial_objective_loss.toFixed(6)} initial_global_ssim=${state.metrics.initial_global_ssim.toFixed(6)} initial_windowed_ssim=${state.metrics.initial_ssim.toFixed(6)} initial_local_p10=${state.metrics.initial_regional_ssim.p10.toFixed(6)} growth=${state.metrics.growth_schedule.percentage}% threshold=${state.metrics.growth_schedule.signal_threshold} cap=${finalCount}`,
    );

    const metricInterval = Math.max(1, Math.min(DEFAULT_MAX_METRIC_INTERVAL, state.recommendation?.metricInterval || Math.floor(steps / 60)));
    let appliedRuntimeSettingsRevision = state.runtimeSettingsRevision;
    const trainingPerfStarted = performance.now();
    let traceLastTime = trainingPerfStarted;
    let traceLastStep = 0;
    for (let step = 1; step <= steps; step += 1) {
      const stepWallStarted = performance.now();
      const traceProfileLabels = state.webgpu.profile?.timing_backend === "timestamp-query"
        ? performanceProfileLabels(step, steps)
        : [];
      let stepDensityMs = 0;
      let stepTrainMs = 0;
      let stepAdcMs = 0;
      let stepRelocationMs = 0;
      let stepPresentationMs = 0;
      let presentation = "none";
      while (state.paused && !state.stopRequested) {
        await nextFrame();
      }
      if (appliedRuntimeSettingsRevision !== state.runtimeSettingsRevision) {
        learningRates = selectedLearningRates();
        previewRefresh = selectedPreviewRefresh();
        syncRuntimeMetrics(learningRates, previewRefresh);
        appliedRuntimeSettingsRevision = state.runtimeSettingsRevision;
        eventLog(`runtime parameters applied revision=${appliedRuntimeSettingsRevision}`);
      }
      if (state.stopRequested) {
        state.metrics.stopped = true;
        log(`stopped at step ${step - 1}`);
        break;
      }
      const densitySteps = experimentalDensifySteps(steps);
      const growthSteps = experimentalGrowthSteps(steps);
      const growthSettings = { ...phase39Variants(), stageAwareGrowth: runStageAwareGrowth };
      const densifyInterval = growthSettings.densifyInterval;
      const densifyDue =
        step > densifyWarmupSteps(densitySteps) &&
        step <= growthSteps &&
        (step % densifyInterval === 0 || step === growthSteps);
      const growthPlan = densifyDue
        ? growthSchedulePlan({
          step,
          steps,
          initialCount: state.metrics.initial_splats,
          currentCount: state.params.count,
          finalCount,
          growthFraction: growthSettings.growthFraction,
          densifyInterval,
          stageAware: growthSettings.stageAwareGrowth,
        })
        : null;
      const requestedTargetCount = growthPlan?.requestedCount ?? state.params.count;
      let targetCount = state.params.count;
      let growthResult = null;
      let growthStartCount = state.params.count;
      let densityPhase = experimentalDensityPhase(step, steps);
      let densityGpuMs = 0;
      if (requestedTargetCount > state.params.count) {
        const densifyFailure = safetyFailure(computeBudgetFor(Number(els.trainSize.value), requestedTargetCount, steps), "densify");
        if (densifyFailure) {
          state.metrics.stopped = true;
          state.metrics.safety_stop = {
            reason: densifyFailure.reason,
            context: densifyFailure.context,
            estimated_mb: densifyFailure.rec.estimatedMB,
            budget_mb: densifyFailure.rec.budgetMB,
            recommended_train_size: densifyFailure.rec.recommendedTrainSize,
            recommended_final_splats: densifyFailure.rec.recommendedFinalSplats,
          };
          setSafetyStop(densifyFailure);
          break;
        }
        const densityStarted = performance.now();
        growthResult = await state.webgpu.renderer.growExperimentalGpu(state.image, state.params, requestedTargetCount, step, steps);
        densityGpuMs = performance.now() - densityStarted;
        stepDensityMs += densityGpuMs;
        state.metrics.density_gpu_ms += densityGpuMs;
        if (!growthResult) {
          throw new Error("Experimental GPU densify failed; CPU fallback is disabled");
        }
        if (growthResult.grown) {
          targetCount = growthResult.count;
          state.params = growParamPlaceholders(state.params, targetCount);
          state.metrics.webgpu_densify = true;
          state.metrics.num_gaussians = state.params.count;
          if (state.params.count >= finalCount && state.metrics.growth_schedule.cap_reached_step === null) {
            state.metrics.growth_schedule.cap_reached_step = step;
          }
        } else {
          state.metrics.growth_schedule.threshold_skips += 1;
        }
      }
      if (densifyDue) {
        const operations = growthResult?.operations || {};
        state.metrics.densify_events.push({
          step,
          stage: curriculumTrainingStage(
            step,
            steps,
            phase33Variants(),
            state.webgpu.renderer.trainState?.coarseImage,
            state.webgpu.renderer.trainState?.midImage,
          ),
          schedule_mode: growthPlan.mode,
          count_before: growthStartCount,
          count: state.params.count,
          desired_count: growthPlan.desiredCount,
          previous_desired_count: growthPlan.previousDesired,
          requested_count: requestedTargetCount,
          actual_count: state.params.count,
          added: growthResult?.grown ? targetCount - growthStartCount : 0,
          normal_increment: growthPlan.normalIncrement,
          catch_up_limit: growthPlan.catchUpLimit,
          headroom: Math.max(0, finalCount - state.params.count),
          candidate_mass: Number.isFinite(growthResult?.candidateMass) ? growthResult.candidateMass : null,
          eligible_source_count: (operations.source_claims || 0) + (operations.source_claim_conflicts || 0),
          qa_growth_comparisons: referenceGrowthTargets(
            growthStartCount,
            finalCount,
            operations.eligible_sources,
            growthSettings.qaGrowthComparisons,
          ),
          applied_source_count: (operations.split || 0) + (operations.duplicate || 0),
          split_count: operations.split || 0,
          duplicate_count: operations.duplicate || 0,
          reseed_count: operations.reseed || 0,
          source_claims: operations.source_claims || 0,
          source_claim_conflicts: operations.source_claim_conflicts || 0,
          threshold_skipped: Boolean(growthResult && !growthResult.grown),
          skipped_reason: requestedTargetCount <= growthStartCount
            ? "schedule-no-growth"
            : growthResult?.grown
              ? null
              : "candidate-threshold",
          all_or_none: true,
          algorithm: state.metrics.algorithm,
          density_phase: densityPhase,
          gpu_ms: densityGpuMs,
        });
      }
      const structuralStep = densifyDue || experimentalAdcResetStep(step, steps);
      const qaHashPending = qaTileOverflowFixtureEnabled() && !state.metrics.tile_retry_parameter_hash?.matches;
      const shouldSyncTrain = qaHashPending || structuralStep || step % state.metrics.train_sync_interval === 0 || step % metricInterval === 0 || step === steps;
      const parameterHashBefore = qaHashPending
        ? await state.webgpu.renderer.hashTrainParameters(state.params)
        : null;
      const trainStarted = performance.now();
      await state.webgpu.renderer.trainStepGpu(state.image, state.params, learningRates, { sync: shouldSyncTrain });
      stepTrainMs = performance.now() - trainStarted;
      state.metrics.webgpu_train_executed = true;
      state.metrics.webgpu_train_update = Boolean(state.webgpu.renderer.lastTrainStats?.updated);
      if (shouldSyncTrain) {
        const retrySteps = await resolveTileOverflowRetry(parameterHashBefore);
        if (retrySteps > 0) {
          state.metrics.webgpu_train_update = false;
          const resumedStep = Math.max(0, step - retrySteps);
          state.metrics.steps_done = resumedStep;
          step = resumedStep;
          continue;
        }
      }
      if (state.webgpu.renderer.lastTrainStats?.layer_update_due) {
        state.metrics.layer_update_count += 1;
        state.metrics.layer_update_last_step = step;
        if (state.metrics.layer_update_first_steps.length < 16) state.metrics.layer_update_first_steps.push(step);
      }
      state.metrics.steps_done = step;
      const adcResetThisStep = experimentalAdcResetStep(step, steps);
      if (adcResetThisStep) {
        const resetStarted = performance.now();
        const resetEvent = await state.webgpu.renderer.adcResetExperimentalGpu(state.image, state.params, step, learningRates);
        stepAdcMs = performance.now() - resetStarted;
        state.metrics.adc_reset_gpu_ms += stepAdcMs;
        if (resetEvent) {
          state.metrics.adc_reset_events.push(resetEvent);
          state.metrics.phase45_region_report = resetEvent.phase45_region_report;
        }
      }
      if (
        !adcResetThisStep &&
        gpuRelocationEnabled &&
        step > densifyWarmupSteps(densitySteps) &&
        step <= Math.floor(densitySteps * 0.85) &&
        step % EXPERIMENTAL_REFINE_EVERY === 0
      ) {
        const relocationStarted = performance.now();
        const relocatedOnGpu = await state.webgpu.renderer.relocateExperimentalGpu(state.image, state.params, step, learningRates);
        stepRelocationMs = performance.now() - relocationStarted;
        state.metrics.relocation_gpu_ms += stepRelocationMs;
        if (relocatedOnGpu) {
          state.metrics.webgpu_relocation = true;
          state.metrics.webgpu_refine = true;
        }
      }
      if (state.stopRequested) {
        state.metrics.stopped = true;
        log(`stopped at step ${step}`);
        break;
      }
      if (step % metricInterval === 0 || step === steps || traceProfileLabels.length > 0) {
        const presentationStarted = performance.now();
        presentation = "metrics";
        const metricsFailure = safetyFailure(computeBudgetFor(Number(els.trainSize.value), state.params.count, steps), "metrics");
        if (metricsFailure) {
          state.metrics.stopped = true;
          state.metrics.safety_stop = {
            reason: metricsFailure.reason,
            context: metricsFailure.context,
            estimated_mb: metricsFailure.rec.estimatedMB,
            budget_mb: metricsFailure.rec.budgetMB,
            recommended_train_size: metricsFailure.rec.recommendedTrainSize,
            recommended_final_splats: metricsFailure.rec.recommendedFinalSplats,
          };
          setSafetyStop(metricsFailure);
          break;
        }
        await updatePreview(step, false);
        await nextFrame();
        stepPresentationMs = performance.now() - presentationStarted;
      } else if (shouldPresentTrainingStep(step, previewRefresh)) {
        const presentationStarted = performance.now();
        presentation = "preview";
        if (!(await presentTrainingPreview(step))) throw new Error("WebGPU live preview state is unavailable");
        await nextFrame();
        stepPresentationMs = performance.now() - presentationStarted;
      } else if (step % 32 === 0) {
        const presentationStarted = performance.now();
        presentation = "status";
        els.stepText.textContent = `${step} / ${state.metrics.steps_requested}`;
        publishState();
        await nextFrame();
        stepPresentationMs = performance.now() - presentationStarted;
      }
      if (step % metricInterval === 0 || step === steps) {
        const now = performance.now();
        const intervalSteps = step - traceLastStep;
        const intervalMs = now - traceLastTime;
        if (intervalSteps > 0 && intervalMs >= 0.05) {
          const tracePoint = {
          step,
          phase: experimentalDensityPhase(step, steps),
          active_splats: state.params.count,
          requested_splats: requestedTargetCount,
          splat_cap: finalCount,
          growth_horizon: growthSteps,
          growth_fraction: growthSettings.growthFraction,
          growth_signal_threshold: growthSettings.growthSignalThreshold,
          elapsed_ms: now - trainingPerfStarted,
          interval_ms: intervalMs,
          interval_steps: intervalSteps,
          iterations_per_second: (intervalSteps * 1000) / intervalMs,
          step_wall_ms: now - stepWallStarted,
          train_ms: stepTrainMs,
          density_ms: stepDensityMs,
          adc_ms: stepAdcMs,
          relocation_ms: stepRelocationMs,
          presentation_ms: stepPresentationMs,
          presentation,
          profile_labels: [...traceProfileLabels],
          sync: shouldSyncTrain,
          tile_builds: state.webgpu.renderer?.lastTrainStats?.tile_builds ?? null,
          tile_candidates: state.metrics.tile_counters?.total ?? null,
          tile_capacity: state.metrics.tile_counters?.capacity ?? null,
          };
          if (state.metrics.performance_trace.length < 160) state.metrics.performance_trace.push(tracePoint);
          else state.metrics.performance_trace[159] = tracePoint;
          traceLastTime = now;
          traceLastStep = step;
        }
      }
    }
    if (!state.metrics.safety_stop) {
      await updatePreview(state.metrics.steps_done, true);
      captureSplatAdjustmentBaseline();
    }
    state.metrics.finished_at = new Date().toISOString();
  } catch (error) {
    if (error.safetyStop) {
      state.metrics.stopped = true;
    } else {
      trainingError = error;
      state.metrics.webgpu_train_error = error.message;
    }
    const rec = state.recommendation || updateMemoryRecommendation();
    log(`WebGPU training stopped: ${error.message}; estimate=${rec.estimatedMB}MB recommended=${rec.recommendedTrainSize}px/${rec.recommendedFinalSplats} splats`);
  } finally {
    const reservedBeforeRelease = state.webgpu.renderer?.trainingMemorySnapshot?.().reservedBytes || 0;
    state.webgpu.renderer?.disposeTrainState();
    if (reservedBeforeRelease > 0) log(`GPU training buffers released ${formatMB(reservedBeforeRelease)}`);
    state.running = false;
    state.paused = false;
    if (state.previewMode === "splats" && state.params) showCanvas("gpu");
    els.startButton.disabled = false;
    els.pauseButton.disabled = true;
    els.pauseButton.textContent = "Pause";
    els.stopButton.disabled = true;
    setPausedRuntimeControlsEnabled(false);
    setInputControlsDisabled(false);
    renderSplatInspector();
    const deviceLost = !state.webgpu.supported && String(state.webgpu.reason).startsWith("device lost:");
    if (trainingError || deviceLost) {
      setStatus("error");
      updateDownloads(false);
    } else {
      const exportCoverage = exportCoverageStatus();
      if (!exportCoverage.ok) {
        state.metrics.background_exposure_violation = false;
        setStatus(state.metrics.safety_stop ? "safety stopped" : "coverage failed");
        updateDownloads(false);
        log(`export blocked: ${exportCoverage.message}`);
      } else {
        state.metrics.background_exposure_violation = exportCoverage.warning === true;
        setStatus(
          state.metrics.safety_stop
            ? "safety stopped"
            : state.metrics.stopped
              ? "stopped"
              : "done",
        );
        updateDownloads(true);
        if (exportCoverage.warning) log(`export warning: ${exportCoverage.message}`);
      }
    }
    publishState();
  }
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
  destroyTiltViewer({ restoreCanvas: true });
  state.webgpu.renderer?.disposeTrainState();
  state.params = null;
  state.metrics = null;
  state.previewPadding = previewPaddingSpec(state.image, null, false);
  state.paused = false;
  state.stopRequested = false;
  state.lastGpuLoss = null;
  els.stepText.textContent = "0 / 0";
  els.lossText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.backgroundPixelText.textContent = "-";
  els.outsideSplatText.textContent = "-";
  clearSplatAdjustmentBaseline();
  setPreviewMode("original");
  updateDownloads(false);
  updateMemoryRecommendation();
  setStatus("image loaded");
  log("reset splats to loaded image");
}

function clearImage() {
  if (state.running) return;
  destroyTiltViewer({ restoreCanvas: true });
  state.imageLoadGeneration += 1;
  state.webgpu.renderer?.disposeTrainState();
  state.image?.sourceBitmap?.close?.();
  state.image = null;
  state.params = null;
  state.metrics = null;
  state.paused = false;
  state.stopRequested = false;
  state.lastGpuLoss = null;
  state.lastPreview = null;
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
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.backgroundPixelText.textContent = "-";
  els.outsideSplatText.textContent = "-";
  els.splatText.textContent = "-";
  updateImageSizeStatus();
  clearSplatAdjustmentBaseline();
  updateDownloads(false);
  updateMemoryRecommendation();
  setStatus("idle");
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
  state.exportReady = Boolean(enabled);
  if (state.exportReady && !state.exporting && !state.exportMessage.startsWith("Exported")) {
    const coverage = exportCoverageStatus();
    state.exportMessage = coverage.warning
      ? `Ready to export. Warning: ${coverage.message}.`
      : "Ready to export the verified final splats.";
  } else if (!state.exportReady && !state.exporting) {
    state.exportMessage = "Train or stop with verified coverage before export.";
  }
  updateExportPanel();
  publishState();
}

const EXPORT_FORMATS = {
  ply: {
    label: "PLY",
    filename: "image2splatpaint.ply",
    description: "Graphdeco-style SH0 PLY with aspect-preserving planar geometry and trained opacity.",
  },
  png: {
    label: "PNG image",
    filename: "image2splatpaint.png",
    description: "Rendered Gaussian result cropped to the source image frame.",
  },
};

function updateExportPanel() {
  const enabled = state.exportReady && !state.exporting;
  const algorithm = selectedAlgorithm();
  const plyPlan = state.params && state.image
    ? plyExportMemoryPlan(state.params, state.image, { download: true })
    : null;
  const plyEnabled = enabled && algorithm.exports.includes("ply") && Boolean(plyPlan?.ok);
  els.savePngButton.disabled = !enabled || !algorithm.exports.includes("png");
  els.savePlyButton.disabled = !plyEnabled;
  els.savePngButton.textContent = state.exporting ? "Saving..." : "Save PNG";
  els.savePlyButton.textContent = state.exporting ? "Exporting..." : "Export PLY";
  els.exportDescription.textContent = EXPORT_FORMATS.ply.description;
  els.exportCount.textContent = state.params ? state.params.count.toLocaleString() : "-";
  els.exportStatus.textContent = enabled && plyPlan && !plyPlan.ok
    ? `PNG is ready. PLY needs ${plyPlan.estimatedPeakMB} MB peak memory; ${plyPlan.reason}.`
    : state.exportMessage;
  const data = document.documentElement.dataset;
  data.exportReady = String(enabled);
  data.pngExportReady = String(enabled);
  data.plyExportReady = String(plyEnabled);
  data.plyExportPeakMb = plyPlan?.estimatedPeakMB || "";
  data.plyExportBudgetMb = plyPlan?.budgetMB || "";
  data.exportCoverage = exportCoverageStatus().reason;
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

function canvasToBlob(canvas, type = "image/png") {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Canvas image encoding failed."));
    }, type);
  });
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

async function makeFramePngBlob() {
  if (!state.image || !state.params || !state.webgpu.renderer) {
    throw new Error("No trained Gaussian result to export.");
  }
  const width = state.image.width;
  const height = state.image.height;
  const rgba = await state.webgpu.renderer.captureFrameRgba(state.image, state.params);
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
const GPU_READBACK_ROW_BYTES = 10 * 4;
const CPU_PARAMETER_ROW_BYTES = 44;

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
    params.detailTags,
    params.bg,
  ]) {
    if (!ArrayBuffer.isView(values)) continue;
    bytes += values.byteLength;
    found = true;
  }
  return found ? bytes : Math.max(0, Number(params.count) || 0) * CPU_PARAMETER_ROW_BYTES;
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
  const props = [
    "property float x",
    "property float y",
    "property float z",
    "property float nx",
    "property float ny",
    "property float nz",
    "property float f_dc_0",
    "property float f_dc_1",
    "property float f_dc_2",
    "property float opacity",
    "property float scale_0",
    "property float scale_1",
    "property float scale_2",
    "property float rot_0",
    "property float rot_1",
    "property float rot_2",
    "property float rot_3",
  ];
  const frame = plyFrameScale(image);
  const layerOrderEnabled = Boolean(params.layerOrderEnabled);
  const boundarySigma = Number.isFinite(params.boundarySigma) ? params.boundarySigma : selectedBoundarySigma();
  return `ply\nformat binary_little_endian 1.0\ncomment image2gaussianpaint_frame ${frame.width} ${frame.height}\ncomment image2gaussianpaint_blend standard_alpha\ncomment image2gaussianpaint_edge_containment ${boundarySigma}\ncomment image2gaussianpaint_layer_order ${layerOrderEnabled ? "micro_z" : "flat_z0"} ${PLY_LAYER_DEPTH_SPAN}\nelement vertex ${params.count}\n${props.join("\n")}\nend_header\n`;
}

function plyExportMemoryPlan(
  params = state.params,
  image = state.image,
  { download = true, memoryInfo = browserCpuMemoryInfo(), baseline = state.splatBaseline } = {},
) {
  const count = Math.max(0, Number(params?.count) || 0);
  const parameterBytes = parameterArrayBytes(params);
  const baselineBytes = baseline === true ? count * CPU_PARAMETER_ROW_BYTES : parameterArrayBytes(baseline);
  const imageBytes = (image?.rgb?.byteLength || 0) + (image?.alpha?.byteLength || 0);
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
  const width = Math.max(1, Number(image?.width) || 1);
  const height = Math.max(1, Number(image?.height) || 1);
  const longSide = Math.max(width, height);
  return {
    x: width / longSide,
    y: height / longSide,
    width,
    height,
    aspect: width / height,
  };
}

function transformPlanarSplatForPly(x, y, sx, sy, theta, image = state.image) {
  // This is the single NDC-to-isotropic-world conversion, independent of grid initialization.
  const frame = plyFrameScale(image);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const sx2 = sx * sx;
  const sy2 = sy * sy;
  const covarianceX = frame.x * frame.x * (c * c * sx2 + s * s * sy2);
  const covarianceY = frame.y * frame.y * (s * s * sx2 + c * c * sy2);
  const covarianceXY = -frame.x * frame.y * c * s * (sx2 - sy2);
  const trace = covarianceX + covarianceY;
  const delta = Math.hypot(covarianceX - covarianceY, 2 * covarianceXY);
  const lambda0 = Math.max(1e-12, 0.5 * (trace + delta));
  const lambda1 = Math.max(1e-12, 0.5 * (trace - delta));
  return {
    x: x * frame.x,
    y: -y * frame.y,
    sx: Math.sqrt(lambda0),
    sy: Math.sqrt(lambda1),
    theta: 0.5 * Math.atan2(2 * covarianceXY, covarianceX - covarianceY),
    frame,
  };
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
  const profiles = Array.from({ length: params.count }, (_, index) => tiltRiskProfileForSplat(params, image, index, angleDegrees));
  const support = profiles.map((profile) => profile.supportDepth).sort((a, b) => a - b);
  const percentile = (fraction) => support.length
    ? support[Math.min(support.length - 1, Math.round((support.length - 1) * fraction))]
    : 0;
  const risky = profiles.filter((profile) => profile.risk > 0).sort((a, b) => b.risk - a.risk);
  return {
    angle_degrees: angleDegrees,
    depth_threshold: PLY_LAYER_DEPTH_SPAN,
    color_threshold: DEFAULT_TILT_SPLIT_COLOR_THRESHOLD,
    risky_count: risky.length,
    risky_ratio: risky.length / Math.max(1, params.count),
    support_depth_p95: percentile(0.95),
    support_depth_p99: percentile(0.99),
    support_depth_max: support.at(-1) || 0,
    top: risky.slice(0, 16),
  };
}

function plyLayerDepth(index, params, enabled = Boolean(params.layerOrderEnabled)) {
  if (!enabled) return 0;
  const fallback = 1 - index / Math.max(1, params.count - 1);
  const order = Math.max(0, Math.min(1, params.depthOrder?.[index] ?? fallback));
  return (order - 0.5) * PLY_LAYER_DEPTH_SPAN;
}

function makePly(params = state.params, image = state.image) {
  if (!params) throw new Error("No trained splats to export.");
  if (!image) throw new Error("No source image is available for aspect-preserving PLY export.");
  assertSplatCountContract(params, "ply-export");
  assertFiniteParams(params, "ply-export");
  const layerOrderEnabled = Boolean(params.layerOrderEnabled);
  const header = plyHeaderText(params, image);
  const headerBytes = new TextEncoder().encode(header);
  const rowBytes = params.count * PLY_ROW_BYTES;
  const buffer = new ArrayBuffer(headerBytes.byteLength + rowBytes);
  const bytes = new Uint8Array(buffer);
  bytes.set(headerBytes, 0);
  const view = new DataView(buffer, headerBytes.byteLength);
  let o = 0;
  for (let i = 0; i < params.count; i += 1) {
    const geometry = transformPlanarSplatForPly(
      params.xy[i * 2],
      params.xy[i * 2 + 1],
      params.scale[i * 2],
      params.scale[i * 2 + 1],
      params.theta?.[i] || 0,
      image,
    );
    const halfTheta = geometry.theta * 0.5;
    const values = [
      geometry.x,
      geometry.y,
      plyLayerDepth(i, params, layerOrderEnabled),
      0,
      0,
      0,
      (params.rgb[i * 3] - 0.5) / SH_C0,
      (params.rgb[i * 3 + 1] - 0.5) / SH_C0,
      (params.rgb[i * 3 + 2] - 0.5) / SH_C0,
      logit(params.opacity[i]),
      Math.log(Math.max(geometry.sx, 1e-6)),
      Math.log(Math.max(geometry.sy, 1e-6)),
      Math.log(1e-4),
      Math.cos(halfTheta),
      0,
      0,
      Math.sin(halfTheta),
    ];
    for (const value of values) {
      view.setFloat32(o, value, true);
      o += 4;
    }
  }
  return buffer;
}

function currentTiltRevision() {
  if (!state.params || !state.image) return "";
  return [
    state.metrics?.params_revision ?? 0,
    state.metrics?.final_readback_step ?? -1,
    state.metrics?.steps_done ?? -1,
    state.splatAdjustmentVersion,
    state.params.count,
    state.image.width,
    state.image.height,
  ].join(":");
}

function tiltViewerReady() {
  return Boolean(
    location.protocol !== "file:" &&
    state.params &&
    state.metrics &&
    !state.running &&
    state.metrics.cpu_mirror_current &&
    state.metrics.final_readback_step === state.metrics.steps_done,
  );
}

function updateTiltControlState() {
  const ready = tiltViewerReady();
  els.tiltTab.disabled = !ready || state.tilt.loading;
  const interactive = Boolean(state.tilt.controller) && !state.tilt.sweepRunning;
  els.tiltPitch.disabled = !interactive;
  els.tiltYaw.disabled = !interactive;
  els.tiltFrontButton.disabled = !interactive;
  els.tiltRefreshButton.disabled = !ready || state.tilt.loading || state.tilt.sweepRunning;
  els.tiltSweepButton.disabled = !interactive;
  els.tiltSweepStopButton.disabled = !state.tilt.sweepRunning;
  document.documentElement.dataset.tiltReady = String(ready);
  document.documentElement.dataset.tiltLoaded = String(Boolean(state.tilt.controller));
  document.documentElement.dataset.tiltLoading = String(state.tilt.loading);
  document.documentElement.dataset.tiltRevision = state.tilt.revision;
  document.documentElement.dataset.tiltPlySha256 = state.tilt.plyDigest || state.tilt.verifiedPlyDigest;
  document.documentElement.dataset.tiltPlyBytes = String(state.tilt.plyByteLength || state.tilt.verifiedPlyByteLength || 0);
  document.documentElement.dataset.tiltPlyVertices = String(state.tilt.vertices || 0);
  document.documentElement.dataset.tiltSweepRunning = String(state.tilt.sweepRunning);
  document.documentElement.dataset.tiltSweepCompleted = String(state.tilt.sweepResults.length);
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
  els.tiltRadiusValue.textContent = diagnostics.radius.toFixed(4);
  els.tiltPositionValue.textContent = diagnostics.position.map((value) => value.toFixed(3)).join(", ");
  els.tiltFovValue.textContent = `${diagnostics.fovDegrees.toFixed(0)}\u00b0`;
  els.tiltProjectionError.textContent = `${diagnostics.cornerErrorMaxPx.toFixed(3)} px`;
  els.tiltCameraMode.textContent = "center orbit";
  els.tiltFrameOverlay.hidden = els.tiltCanvas.hidden;
  document.documentElement.dataset.tiltOrbitRadius = String(diagnostics.radius);
  document.documentElement.dataset.tiltCameraPosition = diagnostics.position.join(",");
  document.documentElement.dataset.tiltFov = String(diagnostics.fovDegrees);
  document.documentElement.dataset.tiltProjectionError = String(diagnostics.cornerErrorMaxPx);
}

function clearTiltSweepResults() {
  for (const url of state.tilt.sweepObjectUrls) URL.revokeObjectURL(url);
  state.tilt.sweepObjectUrls = [];
  state.tilt.sweepResults = [];
  state.tilt.sweepStartedAt = 0;
  els.tiltContactSheet.replaceChildren();
  els.tiltSweepProgress.value = 0;
  els.tiltSweepSummary.textContent = "49-pose Fibonacci cap · 0–75°";
  delete document.documentElement.dataset.tiltSweepElapsedMs;
  delete document.documentElement.dataset.tiltSweepMaxProjectionError;
  delete document.documentElement.dataset.tiltSweepRetainedBytes;
  delete document.documentElement.dataset.tiltSweepWorstCamera;
}

function appendTiltSweepFrame(result, objectUrl) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.poseIndex = String(result.index);
  button.dataset.stress = String(result.polarDegrees >= 74.999);
  button.title = `Polar ${result.polarDegrees.toFixed(1)}°, azimuth ${result.azimuthDegrees.toFixed(1)}°`;
  const image = document.createElement("img");
  image.src = objectUrl;
  image.loading = "lazy";
  image.decoding = "async";
  image.alt = `Fibonacci camera ${result.index + 1} at ${result.polarDegrees.toFixed(1)} degrees`;
  const label = document.createElement("span");
  label.textContent = `${String(result.index + 1).padStart(2, "0")} · ${result.polarDegrees.toFixed(1)}°`;
  button.append(image, label);
  button.addEventListener("click", async () => {
    if (!state.tilt.controller || state.tilt.sweepRunning) return;
    await state.tilt.controller.setTiltAndWait(result.pitchDegrees, result.yawDegrees);
    els.tiltPitch.value = String(result.pitchDegrees);
    els.tiltYaw.value = String(result.yawDegrees);
    applyTiltInputs();
  });
  els.tiltContactSheet.append(button);
}

async function startTiltHemisphereSweep() {
  if (state.tilt.sweepRunning) return;
  const controller = await loadTiltViewer();
  if (!controller) return;
  const { fibonacciHemispherePoses } = await import(TILT_CAMERA_MODULE_URL);
  const poses = fibonacciHemispherePoses(49, 75);
  clearTiltSweepResults();
  state.tilt.sweepRunning = true;
  state.tilt.sweepStopRequested = false;
  state.tilt.sweepStartedAt = performance.now();
  els.tiltSweepProgress.max = poses.length;
  els.tiltStatus.textContent = `Rendering Fibonacci hemisphere 0 / ${poses.length}...`;
  updateTiltControlState();
  publishState();
  await nextFrame();
  await nextFrame();
  let capturedBytes = 0;
  try {
    for (const pose of poses) {
      if (state.tilt.sweepStopRequested) break;
      const rendered = await controller.setTiltAndWait(pose.pitchDegrees, pose.yawDegrees);
      const blob = await controller.captureFrameBlob();
      const objectUrl = URL.createObjectURL(blob);
      state.tilt.sweepObjectUrls.push(objectUrl);
      capturedBytes += blob.size;
      const provisional = { ...pose, sortMs: rendered.sortMs, captureBytes: blob.size };
      appendTiltSweepFrame(provisional, objectUrl);
      await nextFrame();
      await nextFrame();
      const diagnostics = await controller.refreshCameraDiagnostics();
      if (!diagnostics?.corners?.every((corner) => corner.valid)) {
        throw new Error(`Fibonacci pose ${pose.index} produced an invalid projection.`);
      }
      const result = {
        ...provisional,
        position: [...diagnostics.position],
        target: [...diagnostics.target],
        radius: diagnostics.radius,
        fovDegrees: diagnostics.fovDegrees,
        projectionErrorPx: diagnostics.cornerErrorMaxPx,
      };
      state.tilt.sweepResults.push(result);
      els.tiltSweepProgress.value = state.tilt.sweepResults.length;
      els.tiltSweepSummary.textContent = `${state.tilt.sweepResults.length} / ${poses.length} · ${pose.polarDegrees.toFixed(1)}°`;
      els.tiltStatus.textContent = `Rendering Fibonacci hemisphere ${state.tilt.sweepResults.length} / ${poses.length}...`;
      document.documentElement.dataset.tiltSweepCompleted = String(state.tilt.sweepResults.length);
    }
    const elapsedMs = performance.now() - state.tilt.sweepStartedAt;
    const results = state.tilt.sweepResults;
    const maxProjectionErrorPx = Math.max(0, ...results.map((result) => result.projectionErrorPx));
    const projectionWorst = results.reduce((current, result) => !current || result.projectionErrorPx > current.projectionErrorPx ? result : current, null);
    const slowest = results.reduce((current, result) => !current || result.sortMs > current.sortMs ? result : current, null);
    const completed = results.length === poses.length;
    const summary = {
      camera_distribution: "fibonacci-spherical-cap",
      requested_poses: poses.length,
      completed_poses: results.length,
      max_polar_degrees: Math.max(0, ...results.map((result) => result.polarDegrees)),
      fixed_radius: results.every((result) => Math.abs(result.radius - results[0].radius) <= 1e-8),
      max_projection_error_px: maxProjectionErrorPx,
      elapsed_ms: elapsedMs,
      milliseconds_per_pose: results.length ? elapsedMs / results.length : null,
      captured_bytes: capturedBytes,
      ply_bytes: state.tilt.plyByteLength,
      app_retained_bytes: state.tilt.plyByteLength + capturedBytes,
      exact_device_vram: false,
      slowest_pose: slowest ? { index: slowest.index, polar_degrees: slowest.polarDegrees, azimuth_degrees: slowest.azimuthDegrees, sort_ms: slowest.sortMs } : null,
      projection_worst_pose: projectionWorst ? {
        index: projectionWorst.index,
        polar_degrees: projectionWorst.polarDegrees,
        azimuth_degrees: projectionWorst.azimuthDegrees,
        position: projectionWorst.position,
        target: projectionWorst.target,
        radius: projectionWorst.radius,
        fov_degrees: projectionWorst.fovDegrees,
        error_px: projectionWorst.projectionErrorPx,
      } : null,
      stopped: !completed,
    };
    if (state.metrics) state.metrics.tilt_hemisphere_sweep = summary;
    document.documentElement.dataset.tiltSweepElapsedMs = String(elapsedMs);
    document.documentElement.dataset.tiltSweepMaxProjectionError = String(maxProjectionErrorPx);
    document.documentElement.dataset.tiltSweepRetainedBytes = String(summary.app_retained_bytes);
    document.documentElement.dataset.tiltSweepWorstCamera = JSON.stringify(summary.projection_worst_pose);
    els.tiltSweepSummary.textContent = completed
      ? `${results.length} poses · ${(elapsedMs / 1000).toFixed(1)} s · max error ${maxProjectionErrorPx.toFixed(3)} px`
      : `Stopped at ${results.length} / ${poses.length}`;
    els.tiltStatus.textContent = completed
      ? "Fibonacci hemisphere complete. Select a frame to inspect that camera."
      : "Fibonacci hemisphere stopped.";
    eventLog(`Tilt hemisphere ${completed ? "completed" : "stopped"} poses=${results.length}/${poses.length} elapsed_ms=${elapsedMs.toFixed(1)} capture=${formatMB(capturedBytes)}`);
  } finally {
    state.tilt.sweepRunning = false;
    state.tilt.sweepStopRequested = false;
    if (state.tilt.controller) {
      await state.tilt.controller.setTiltAndWait(0, 0).catch(() => null);
      els.tiltPitch.value = "0";
      els.tiltYaw.value = "0";
      applyTiltInputs();
    }
    updateTiltControlState();
    publishState();
  }
}

function destroyTiltViewer({ restoreCanvas = false } = {}) {
  state.tilt.sweepStopRequested = true;
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
  state.tilt.sweepRunning = false;
  clearTiltSweepResults();
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
  els.tiltStatus.textContent = tiltViewerReady()
    ? "Open Tilt to build a fresh in-memory PLY view."
    : location.protocol === "file:"
      ? "Tilt requires GitHub Pages or a local HTTP server."
      : "Finish training to inspect the PLY.";
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
  els.tiltPitchValue.textContent = `${pitch}\u00b0`;
  els.tiltYawValue.textContent = `${yaw}\u00b0`;
  document.documentElement.dataset.tiltPitch = String(pitch);
  document.documentElement.dataset.tiltYaw = String(yaw);
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
    assertTiltViewerCapacity(state.params, state.image);
    const plyBuffer = makePly(state.params, state.image);
    const { createTiltViewer } = await import(TILT_VIEWER_MODULE_URL);
    const controller = await createTiltViewer({
      canvas: els.tiltCanvas,
      plyBuffer,
      frame: plyFrameScale(state.image),
      signal: abortController.signal,
      onCameraChange: updateTiltCameraDiagnostics,
    });
    if (generation !== state.tilt.generation) {
      controller.destroy();
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
    applyTiltInputs();
    els.tiltStatus.textContent = `PlayCanvas ${controller.engineVersion} (${controller.backend}). The camera orbits the image center at a fixed radius.`;
    eventLog(
      `Tilt viewer loaded from memory PLY: ${state.params.count} splats` +
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
  if (!algorithmSupportsExport(formatKey)) {
    throw new Error(`${selectedAlgorithm().label} does not support ${formatKey.toUpperCase()} export.`);
  }
  const coverage = exportCoverageStatus();
  if (!coverage.ok) throw new Error(`Export blocked: ${coverage.message}`);
  if (coverage.warning) log(`export warning: ${coverage.message}`);
  const format = EXPORT_FORMATS[formatKey] || EXPORT_FORMATS.ply;
  if (formatKey === "ply") assertPlyExportCapacity(state.params, state.image, download);

  state.exporting = true;
  state.exportMessage = `Preparing ${format.label}...`;
  updateExportPanel();
  publishState();
  try {
    if (formatKey === "png") {
      const { blob, nonblackPixels, meanRgb, pngRgbaParity } = await makeFramePngBlob();
      const bytes = new Uint8Array(await blob.arrayBuffer());
      if (download) downloadBlob(format.filename, blob);
      state.exportMessage = `${format.filename} ${download ? "saved" : "validated"} (${(bytes.byteLength / 1024).toFixed(1)} KiB).`;
      state.metrics.export_history ||= [];
      state.metrics.export_history.push({
        format: formatKey,
        filename: format.filename,
        bytes: bytes.byteLength,
        width: state.image.width,
        height: state.image.height,
        nonblack_pixels: nonblackPixels,
        mean_rgb: meanRgb,
        rgba_roundtrip: pngRgbaParity,
        step: state.metrics.steps_done,
      });
      eventLog(`${download ? "exported" : "validated"} ${format.filename} ${state.image.width}x${state.image.height} bytes=${bytes.byteLength}`);
      return {
        format: formatKey,
        filename: format.filename,
        bytes,
        width: state.image.width,
        height: state.image.height,
        nonblackPixels,
        meanRgb,
        pngRgbaParity,
      };
    }

    const plyBuffer = makePly(state.params, state.image);
    const plyContract = inspectPlyContract(plyBuffer, state.params, state.image);
    const plyValid =
      plyContract.vertices === state.params.count &&
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
  const bytes = new Uint8Array(buffer);
  const marker = new TextEncoder().encode("end_header\n");
  let markerOffset = -1;
  for (let i = 0; i <= bytes.length - marker.length; i += 1) {
    if (marker.every((value, index) => bytes[i + index] === value)) {
      markerOffset = i;
      break;
    }
  }
  if (markerOffset < 0) throw new Error("PLY end_header marker missing");
  const dataOffset = markerOffset + marker.length;
  const header = new TextDecoder("ascii").decode(bytes.subarray(0, dataOffset));
  const vertexMatch = header.match(/element vertex (\d+)/);
  const frameMatch = header.match(/comment image2gaussianpaint_frame (\d+) (\d+)/);
  const boundaryMatch = header.match(/comment image2gaussianpaint_edge_containment ([0-9.eE+-]+)/);
  const layerMatch = header.match(/comment image2gaussianpaint_layer_order (micro_z|flat_z0) ([0-9.eE+-]+)/);
  const properties = [...header.matchAll(/^property float (\S+)$/gm)].map((match) => match[1]);
  const vertices = Number(vertexMatch?.[1] || 0);
  const frameWidth = Number(frameMatch?.[1] || 0);
  const frameHeight = Number(frameMatch?.[2] || 0);
  const expectedFrame = plyFrameScale(sourceImage);
  const boundarySigma = Math.max(0, Number(boundaryMatch?.[1] ?? sourceParams?.boundarySigma ?? selectedBoundarySigma()));
  const frameAspect = frameHeight > 0 ? frameWidth / frameHeight : Number.NaN;
  const aspectRatioError = Number.isFinite(frameAspect) ? Math.abs(frameAspect - expectedFrame.aspect) : Number.POSITIVE_INFINITY;
  const rowBytes = properties.length * 4;
  const view = new DataView(buffer, dataOffset);
  let allZZero = true;
  let zAbsMax = 0;
  let layerDepthErrorMax = 0;
  let allFinite = true;
  let planarSh0 = !properties.some((name) => name.startsWith("f_rest_"));
  let planarRotation = true;
  let geometryMatchErrorMax = 0;
  let opacityErrorMax = 0;
  let colorErrorMax = 0;
  let boundaryLeakCount = 0;
  let boundaryMaxLeak = 0;
  let renderOutsideCount = 0;
  let renderOutsideMaxExtent = 0;
  const anisotropyLimits = anisotropyLimitsForParams(sourceParams);
  let surfaceAnisotropyMax = 1;
  let detailAnisotropyMax = 1;
  let anisotropyLimitViolations = 0;
  for (let i = 0; i < vertices; i += 1) {
    const row = i * rowBytes;
    for (let p = 0; p < properties.length; p += 1) {
      if (!Number.isFinite(view.getFloat32(row + p * 4, true))) allFinite = false;
    }
    const z = view.getFloat32(row + 8, true);
    if (z !== 0) allZZero = false;
    zAbsMax = Math.max(zAbsMax, Math.abs(z));
    if (view.getFloat32(row + 14 * 4, true) !== 0 || view.getFloat32(row + 15 * 4, true) !== 0) planarRotation = false;
    const x = view.getFloat32(row, true);
    const y = view.getFloat32(row + 4, true);
    const sx = Math.exp(view.getFloat32(row + 10 * 4, true));
    const sy = Math.exp(view.getFloat32(row + 11 * 4, true));
    const theta = 2 * Math.atan2(view.getFloat32(row + 16 * 4, true), view.getFloat32(row + 13 * 4, true));
    if (sourceParams && i < sourceParams.count) {
      const expectedZ = plyLayerDepth(i, sourceParams, layerMatch?.[1] === "micro_z");
      layerDepthErrorMax = Math.max(layerDepthErrorMax, Math.abs(z - expectedZ));
      const expected = transformPlanarSplatForPly(
        sourceParams.xy[i * 2],
        sourceParams.xy[i * 2 + 1],
        sourceParams.scale[i * 2],
        sourceParams.scale[i * 2 + 1],
        sourceParams.theta?.[i] || 0,
        sourceImage,
      );
      const angleError = 0.5 * Math.abs(Math.atan2(Math.sin(2 * (theta - expected.theta)), Math.cos(2 * (theta - expected.theta))));
      geometryMatchErrorMax = Math.max(
        geometryMatchErrorMax,
        Math.abs(x - expected.x),
        Math.abs(y - expected.y),
        Math.abs(sx - expected.sx),
        Math.abs(sy - expected.sy),
        angleError,
      );
      const exportedOpacity = 1 / (1 + Math.exp(-view.getFloat32(row + 9 * 4, true)));
      opacityErrorMax = Math.max(opacityErrorMax, Math.abs(exportedOpacity - sourceParams.opacity[i]));
      for (let channel = 0; channel < 3; channel += 1) {
        const exportedColor = view.getFloat32(row + (6 + channel) * 4, true) * SH_C0 + 0.5;
        colorErrorMax = Math.max(colorErrorMax, Math.abs(exportedColor - sourceParams.rgb[i * 3 + channel]));
      }
      const sourceMinor = Math.max(
        MIN_SPLAT_SCALE,
        Math.min(sourceParams.scale[i * 2], sourceParams.scale[i * 2 + 1]),
      );
      const sourceRatio = Math.max(sourceParams.scale[i * 2], sourceParams.scale[i * 2 + 1]) / sourceMinor;
      const detail = Math.floor(Number(sourceParams.detailTags?.[i]) || 1) >= 2;
      const limit = detail ? anisotropyLimits.detail : anisotropyLimits.surface;
      if (detail) detailAnisotropyMax = Math.max(detailAnisotropyMax, sourceRatio);
      else surfaceAnisotropyMax = Math.max(surfaceAnisotropyMax, sourceRatio);
      if (sourceRatio > limit + 1e-5) anisotropyLimitViolations += 1;
    }
    const extent = rotatedSplatExtent(sx, sy, theta, boundarySigma);
    const leak = Math.max(
      0,
      Math.abs(x) + extent.x - expectedFrame.x,
      Math.abs(y) + extent.y - expectedFrame.y,
    );
    if (leak > 1e-6) boundaryLeakCount += 1;
    boundaryMaxLeak = Math.max(boundaryMaxLeak, leak);
    const renderExtent = rotatedSplatExtent(sx, sy, theta, RENDER_SIGMA);
    const renderLeak = Math.max(
      0,
      Math.abs(x) + renderExtent.x - expectedFrame.x,
      Math.abs(y) + renderExtent.y - expectedFrame.y,
    );
    if (renderLeak > 1e-6) renderOutsideCount += 1;
    renderOutsideMaxExtent = Math.max(renderOutsideMaxExtent, renderLeak);
  }
  return {
    format: header.includes("format binary_little_endian 1.0") ? "binary_little_endian" : "unknown",
    vertices,
    properties,
    row_bytes: rowBytes,
    payload_bytes: buffer.byteLength - dataOffset,
    all_z_zero: allZZero,
    z_abs_max: zAbsMax,
    layer_order_enabled: layerMatch?.[1] === "micro_z",
    layer_depth_span: Number(layerMatch?.[2] || 0),
    layer_depth_error_max: layerDepthErrorMax,
    layer_depth_match: layerDepthErrorMax <= 1e-8 && zAbsMax <= PLY_LAYER_DEPTH_SPAN * 0.501,
    all_finite: allFinite,
    sh_degree_0: planarSh0,
    planar_rotation: planarRotation,
    frame_width: frameWidth,
    frame_height: frameHeight,
    frame_aspect: frameAspect,
    expected_frame_aspect: expectedFrame.aspect,
    aspect_ratio_error: aspectRatioError,
    aspect_ratio_preserved: aspectRatioError <= 1e-6,
    standard_alpha_blend: header.includes("comment image2gaussianpaint_blend standard_alpha"),
    geometry_match_error_max: geometryMatchErrorMax,
    y_reflection_rotation: geometryMatchErrorMax <= 1e-5,
    y_reflection_rotation_error_max: geometryMatchErrorMax,
    opacity_error_max: opacityErrorMax,
    color_error_max: colorErrorMax,
    surface_anisotropy_limit: anisotropyLimits.surface,
    surface_anisotropy_max: surfaceAnisotropyMax,
    detail_anisotropy_limit: anisotropyLimits.detail,
    detail_anisotropy_max: detailAnisotropyMax,
    anisotropy_limit_violations: anisotropyLimitViolations,
    boundary_sigma: boundarySigma,
    boundary_leak_count: boundaryLeakCount,
    boundary_max_leak: boundaryMaxLeak,
    render_footprint_outside_count: renderOutsideCount,
    render_footprint_outside_max_extent: renderOutsideMaxExtent,
  };
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
  const paramsHash = hashParams();
  const plySignature = new TextEncoder().encode([
    paramsHash,
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
    splats: state.params.count,
    steps: metrics?.steps_done ?? 0,
    l1: metrics?.final_l1 ?? null,
    global_ssim: metrics?.final_global_ssim ?? null,
    windowed_ssim: metrics?.final_windowed_ssim ?? null,
    regional_p10: metrics?.final_regional_ssim?.p10 ?? null,
  };
}

async function runPreviewRefreshLoop() {
  let rendered = false;
  state.previewRefreshPending = true;
  publishState();
  try {
    while (state.previewAppliedRevision < state.previewRequestedRevision) {
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
      if (!state.image || !state.params || !state.webgpu.renderer) {
        state.previewPadding = previewPaddingSpec(state.image, state.params, false);
        state.previewAppliedAlphaBackground = "";
        state.previewAppliedRevision = requestedRevision;
        break;
      }

      const generation = state.previewGeneration;
      const image = state.image;
      const params = state.params;
      const before = previewInvariantSnapshot();
      const buffers = state.webgpu.renderer.currentTrainBuffers(params);
      const alphaOptions = document.documentElement.dataset.activeDetailTab === "splats" && state.previewMode === "splats"
        ? splatAlphaRenderOptions()
        : undefined;
      const appliedAlphaBackground = alphaOptions ? els.splatAlphaBackground.value.toLowerCase() : "";
      await state.webgpu.renderer.render(image, params, buffers, null, alphaOptions);
      if (generation !== state.previewGeneration || state.running || image !== state.image || params !== state.params) {
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
    state.previewRefreshPending = false;
    publishState();
  }
}

async function refreshOutsidePreview() {
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

els.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  els.dropZone.classList.add("dragover");
});

els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("dragover"));

els.dropZone.addEventListener("drop", async (event) => {
  event.preventDefault();
  els.dropZone.classList.remove("dragover");
  try {
    state.lastInputMode = "drop";
    await loadFile(event.dataTransfer.files[0]);
  } catch (error) {
    setStatus("error");
    log(error.message);
  }
});

els.fileInput.addEventListener("change", async () => {
  try {
    state.lastInputMode = "file";
    await loadFile(els.fileInput.files[0]);
    eventLog(`loaded image from file picker: ${els.fileInput.files[0]?.name || "unknown"}`);
  } catch (error) {
    setStatus("error");
    log(error.message);
    eventLog(error.message);
  }
});

els.loadImageButton.addEventListener("click", () => {
  if (!state.running) els.fileInput.click();
});

els.clearImageButton.addEventListener("click", confirmClearImage);

els.sampleButton.addEventListener("click", () => {
  state.lastInputMode = "sample";
  loadGeneratedSample().catch((error) => {
    setStatus("error");
    log(error.message);
  });
});
els.splatsPreviewButton.addEventListener("click", () => setPreviewMode("splats"));
els.originalPreviewButton.addEventListener("click", () => setPreviewMode("original"));
els.outsidePreviewToggle.addEventListener("change", () => {
  // Padded preview is a display-layer toggle. Preserve the user's current
  // zoom and pan instead of fitting the newly sized canvas automatically.
  state.canvasView.mode = "custom";
  refreshOutsidePreview().catch((error) => {
    setStatus("error");
    log(error.message);
  });
});
els.pathButton.addEventListener("click", () => {
  loadPathImage().catch((error) => {
    setStatus("error");
    log(error.message);
    eventLog(error.message);
  });
});
for (const element of [els.trainSize, els.initialSplatCount, els.capacityMode, els.stepCount]) {
  element.addEventListener("input", updateMemoryRecommendation);
}
els.finalSplatCount.addEventListener("input", () => {
  updateMemoryRecommendation({ reconcileSplatCounts: false });
});
const commitFinalSplatCount = () => {
  updateMemoryRecommendation();
};
els.finalSplatCount.addEventListener("change", commitFinalSplatCount);
els.finalSplatCount.addEventListener("blur", commitFinalSplatCount);
els.tileCullingToggle.addEventListener("change", publishState);
els.trainLayerOrder.addEventListener("change", () => {
  syncLayerOrderDependency();
  publishState();
});
els.layerUpdateInterval.addEventListener("input", publishState);
els.stageAwareGrowth.addEventListener("change", publishState);
els.detailCoherence.addEventListener("input", publishState);
els.adcSplitInterval.addEventListener("input", publishState);
els.adcResetInterval.addEventListener("input", publishState);
els.densifyInterval.addEventListener("input", publishState);
els.growthPercentage.addEventListener("input", publishState);
els.growthSignalThreshold.addEventListener("input", publishState);
els.adcSplitSignalThreshold.addEventListener("input", publishState);
els.adcSplitResidualThreshold.addEventListener("input", publishState);
els.adcRecyclePercentage.addEventListener("input", publishState);
els.adcLateRecyclePercentage.addEventListener("input", publishState);
els.retryWebGpuButton.addEventListener("click", async () => {
  if (state.running || state.webgpu.supported) return;
  els.retryWebGpuButton.disabled = true;
  try {
    state.webgpu.renderer?.disposeTrainState();
    await detectWebGpu();
  } finally {
    publishState();
  }
});
els.startButton.addEventListener("click", startTraining);
els.algorithmSelect.addEventListener("change", () => {
  const algorithm = selectedAlgorithm();
  document.documentElement.dataset.algorithm = algorithm.id;
  updateExportPanel();
  eventLog(`algorithm selected ${algorithm.label}`);
  publishState();
});
els.pauseButton.addEventListener("click", togglePause);
els.stopButton.addEventListener("click", stopTraining);
els.resetButton.addEventListener("click", resetTrainingState);
els.savePlyButton.addEventListener("click", () => {
  saveExport().catch((error) => log(error.message));
});
els.savePngButton.addEventListener("click", () => {
  saveExport({ formatKey: "png" }).catch((error) => log(error.message));
});

function activateDetailTab(name) {
  const tabs = {
    training: [els.trainingLogTab, els.trainingLogPanel],
    event: [els.eventLogTab, els.eventLogPanel],
    splats: [els.splatsTab, els.splatsPanel],
    export: [els.exportTab, els.exportPanel],
    tilt: [els.tiltTab, els.tiltPanel],
  };
  if (name !== "tilt" && !els.tiltCanvas.hidden) {
    restorePrimaryCanvas();
    destroyTiltViewer();
  }
  for (const [key, [tab, panel]] of Object.entries(tabs)) {
    const active = key === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    panel.classList.toggle("active", active);
  }
  document.documentElement.dataset.activeDetailTab = name;
}

els.trainingLogTab.addEventListener("click", () => {
  activateDetailTab("training");
  refreshOutsidePreview().catch((error) => log(error.message));
});
els.eventLogTab.addEventListener("click", () => activateDetailTab("event"));
els.clearLogButton.addEventListener("click", () => {
  els.log.textContent = "";
});
els.splatsTab.addEventListener("click", () => {
  activateDetailTab("splats");
  refreshOutsidePreview().catch((error) => log(error.message));
});
els.exportTab.addEventListener("click", () => activateDetailTab("export"));
els.tiltTab.addEventListener("click", () => {
  activateDetailTab("tilt");
  loadTiltViewer().catch((error) => {
    log(error.message);
    eventLog(error.message);
  });
});
for (const control of [els.tiltPitch, els.tiltYaw]) {
  control.addEventListener("input", applyTiltInputs);
}
els.tiltFrontButton.addEventListener("click", () => {
  els.tiltPitch.value = "0";
  els.tiltYaw.value = "0";
  applyTiltInputs();
});
els.tiltRefreshButton.addEventListener("click", () => {
  loadTiltViewer({ force: true }).catch((error) => {
    log(error.message);
    eventLog(error.message);
  });
});
els.tiltSweepButton.addEventListener("click", () => {
  startTiltHemisphereSweep().catch((error) => {
    els.tiltStatus.textContent = `Hemisphere sweep failed: ${error.message}`;
    log(error.message);
    eventLog(error.message);
    state.tilt.sweepRunning = false;
    updateTiltControlState();
  });
});
els.tiltSweepStopButton.addEventListener("click", () => {
  state.tilt.sweepStopRequested = true;
  els.tiltStatus.textContent = "Stopping hemisphere sweep...";
});
for (const control of [els.splatOpacity, els.splatScaleX, els.splatScaleY]) {
  control.addEventListener("input", () => queueSplatAdjustments());
}
els.splatAlphaBackground.addEventListener("input", () => {
  refreshOutsidePreview().catch((error) => log(error.message));
});
els.resetSplatAdjustments.addEventListener("click", () => {
  resetSplatAdjustments();
});
els.actualSizeButton.addEventListener("click", () => setCanvasView("actual"));
els.fitViewButton.addEventListener("click", () => setCanvasView("fit"));
els.viewer.addEventListener(
  "wheel",
  (event) => {
    if (!state.image || event.target === els.tiltCanvas || !(event.target instanceof HTMLCanvasElement)) return;
    event.preventDefault();
    zoomCanvasAt(event.clientX, event.clientY, event.deltaY);
  },
  { passive: false },
);
els.viewer.addEventListener("contextmenu", (event) => {
  if (event.target instanceof HTMLCanvasElement) event.preventDefault();
});
els.viewer.addEventListener("pointerdown", (event) => {
  const directPointer = event.pointerType !== "mouse";
  if (!state.image || event.target === els.tiltCanvas || (!directPointer && event.button !== 2) || !(event.target instanceof HTMLCanvasElement)) return;
  event.preventDefault();
  state.canvasView.pointerId = event.pointerId;
  state.canvasView.lastX = event.clientX;
  state.canvasView.lastY = event.clientY;
  state.canvasPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (state.canvasPointers.size === 2) {
    const points = [...state.canvasPointers.values()];
    state.canvasPinch = {
      distance: Math.max(1, Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y)),
      centerX: (points[0].x + points[1].x) * 0.5,
      centerY: (points[0].y + points[1].y) * 0.5,
      scale: state.canvasView.scale,
      panX: state.canvasView.panX,
      panY: state.canvasView.panY,
    };
  }
  els.viewer.classList.add("is-panning");
  els.viewer.setPointerCapture(event.pointerId);
});
els.viewer.addEventListener("pointermove", (event) => {
  if (!state.canvasPointers.has(event.pointerId)) return;
  state.canvasPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  state.canvasView.mode = "custom";
  if (state.canvasPointers.size >= 2 && state.canvasPinch) {
    const points = [...state.canvasPointers.values()].slice(0, 2);
    const distance = Math.max(1, Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y));
    const centerX = (points[0].x + points[1].x) * 0.5;
    const centerY = (points[0].y + points[1].y) * 0.5;
    state.canvasView.scale = Math.max(0.02, Math.min(32, state.canvasPinch.scale * distance / state.canvasPinch.distance));
    state.canvasView.panX = state.canvasPinch.panX + centerX - state.canvasPinch.centerX;
    state.canvasView.panY = state.canvasPinch.panY + centerY - state.canvasPinch.centerY;
  } else {
    state.canvasView.panX += event.clientX - state.canvasView.lastX;
    state.canvasView.panY += event.clientY - state.canvasView.lastY;
  }
  state.canvasView.lastX = event.clientX;
  state.canvasView.lastY = event.clientY;
  applyCanvasView();
});
function endCanvasPan(event) {
  if (!state.canvasPointers.has(event.pointerId)) return;
  state.canvasPointers.delete(event.pointerId);
  state.canvasPinch = null;
  const remaining = state.canvasPointers.entries().next().value;
  state.canvasView.pointerId = remaining?.[0] ?? null;
  if (remaining) {
    state.canvasView.lastX = remaining[1].x;
    state.canvasView.lastY = remaining[1].y;
  } else {
    els.viewer.classList.remove("is-panning");
  }
  if (els.viewer.hasPointerCapture(event.pointerId)) els.viewer.releasePointerCapture(event.pointerId);
  publishState();
}
els.viewer.addEventListener("pointerup", endCanvasPan);
els.viewer.addEventListener("pointercancel", endCanvasPan);
activateDetailTab("training");
syncLayerOrderDependency();
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
  document.body.append(qaMetricsData, qaMetricsButton);
}

// Compatibility aliases exist only for local QA scripts and checkpoints.
if (QA_RUNTIME_ENABLED) window.__flatPhoto3dgs = window.__image2SplatPaint;

detectWebGpu()
  .then(() => setStatus("idle"))
  .catch((error) => {
    state.webgpu = { supported: false, renderer: null, reason: error.message, limits: null, adapterInfo: null };
    els.backendText.textContent = "webgpu unavailable";
    setStatus("idle");
    log(`webgpu check failed: ${error.message}`);
  })
  .finally(() => {
    syncTrainSizeUi();
    updateMemoryRecommendation();
    publishState();
  });

if (QA_RUNTIME_ENABLED) window.__flatPhotoTest = {
  loadGeneratedSample,
  optimizerFootprintHistogram,
  phase39ContractProbe,
  layerOrderComparatorProbe,
  planarTiltRotation,
  projectPlanarPoint,
  inverseProjectPlanarPoint,
  virtualTiltVariants,
  virtualTiltStepSpec,
  tiltRiskProfileForSplat,
  summarizeTiltRisk,
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
    return {
      image: state.image ? { width: state.image.width, height: state.image.height, fileName: state.image.fileName } : null,
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
  async runHemisphereSweep() {
    await startTiltHemisphereSweep();
    return structuredClone(state.metrics?.tilt_hemisphere_sweep || null);
  },
  hemisphereResults() {
    return structuredClone(state.tilt.sweepResults);
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
      initial_l1: m.initial_l1,
      final_l1: m.final_l1,
      initial_alpha_l1: m.initial_alpha_l1,
      final_alpha_l1: m.final_alpha_l1,
      initial_alpha_ssim: m.initial_alpha_ssim,
      final_alpha_ssim: m.final_alpha_ssim,
      initial_alpha_objective: m.initial_alpha_objective,
      final_alpha_objective: m.final_alpha_objective,
      initial_objective_loss: m.initial_objective_loss,
      final_objective_loss: m.final_objective_loss,
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
      adc_reset_events: m.adc_reset_events,
      densify_events: m.densify_events,
      growth_schedule: m.growth_schedule,
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
      phase45_region_report: m.phase45_region_report,
      overlap_diagnostics: m.overlap_diagnostics,
      render_surface_parity: m.render_surface_parity || null,
      performance_trace: m.performance_trace,
      performance_profile_schedule: m.performance_profile_schedule || {},
      stage_profile: m.stage_profile || [],
      stage_profile_backend: m.stage_profile_backend || "off",
      importance_stats: m.importance_stats,
      coverage_stats: m.coverage_stats,
      density_gpu_ms: m.density_gpu_ms,
      relocation_gpu_ms: m.relocation_gpu_ms,
      adc_reset_gpu_ms: m.adc_reset_gpu_ms,
      post_density_annealing: Boolean(m.post_density_annealing),
      tile_culling_enabled: Boolean(m.tile_culling_enabled),
      train_layer_order: Boolean(m.train_layer_order),
      layer_update_interval: m.layer_update_interval,
      layer_update_rate: m.layer_update_rate,
      layer_stage_aware_rate: Boolean(m.layer_stage_aware_rate),
      layer_freeze_fraction: m.layer_freeze_fraction,
      layer_update_count: m.layer_update_count,
      layer_update_first_steps: m.layer_update_first_steps || [],
      layer_update_last_step: m.layer_update_last_step,
      layer_telemetry_enabled: Boolean(m.layer_telemetry_enabled),
      layer_telemetry: m.layer_telemetry || [],
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
        try {
          await saveExport({ download: false });
          result[name] = { blocked: false, message: "export unexpectedly succeeded" };
        } catch (error) {
          result[name] = { blocked: error.message.startsWith("Export blocked:"), message: error.message };
        }
      }
    } finally {
      state.metrics = originalMetrics;
      state.exportMessage = originalMessage;
      updateExportPanel();
      publishState();
    }
    return result;
  },
};
