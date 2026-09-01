(function installDensityShaderFactory(global) {
  function create({ protectedPrefix = 0 } = {}) {
    protectedPrefix = Math.max(0, Math.floor(Number(protectedPrefix) || 0));
    return `
@group(0) @binding(0) var<storage, read> config: array<f32>;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(2) var<storage, read_write> xy: array<SplatPosition>;
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
const RESIDUAL_ORACLE_ENABLED = ${residualDestinationOracleRequested() ? "true" : "false"};
const RESIDUAL_TILE_CDF_ENABLED = ${residualTileCdfEnabled() ? "true" : "false"};
const RESIDUAL_TILE_SIZE = ${TILE_SIZE}u;
var<workgroup> cdfScratch: array<f32, 256>;
var<workgroup> residualCdfScratch: array<u32, 256>;
var<workgroup> phase45Demand: array<u32, ${PHASE45_REGION_COUNT}>;

fn active_region_grid() -> u32 {
  return clamp(u32(config[69]), 4u, PHASE45_REGION_GRID);
}

fn active_region_count() -> u32 {
  let grid = active_region_grid();
  return grid * grid;
}

fn hash_unit(seed: f32) -> f32 {
  let x = sin((seed + config[33] * 104729.0) * 12.9898) * 43758.5453123;
  return x - floor(x);
}

fn underpaint_decode_srgb(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

fn underpaint_encode_srgb(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
}

fn underpaint_neutral_lab_l(rgb: vec3<f32>) -> vec3<f32> {
  let relativeY = dot(
    vec3<f32>(
      underpaint_decode_srgb(rgb.r),
      underpaint_decode_srgb(rgb.g),
      underpaint_decode_srgb(rgb.b)
    ),
    vec3<f32>(0.2126729, 0.7151522, 0.0721750)
  );
  return vec3<f32>(underpaint_encode_srgb(relativeY));
}

fn monochrome_underpainting_active() -> bool {
  return config[93] > 0.5 &&
    config[4] < config[100];
}

fn target_at(pos: vec2<f32>, width: u32, height: u32) -> vec3<f32> {
  let safePos = min(max(pos, vec2<f32>(-1.0)), vec2<f32>(1.0));
  let px = min(width - 1u, u32(floor((safePos.x * 0.5 + 0.5) * f32(width - 1u) + 0.5)));
  let py = min(height - 1u, u32(floor((safePos.y * 0.5 + 0.5) * f32(height - 1u) + 0.5)));
  let index = (py * width + px) * 3u;
  let rgb = vec3<f32>(targetRgb[index], targetRgb[index + 1u], targetRgb[index + 2u]);
  return select(rgb, underpaint_neutral_lab_l(rgb), monochrome_underpainting_active());
}

// Paint kernels cover an oriented area, so center-pixel RGB alone is not a
// stable outlier signal. Return a center-weighted five-sample sRGB mean plus
// the mean sample deviation.
fn paint_footprint_target(g: u32, width: u32, height: u32) -> vec4<f32> {
  let t = transform[g];
  let center = xy[g].center;
  let c = cos(t.z);
  let s = sin(t.z);
  let axisX = vec2<f32>(c, s) * t.x * 0.65;
  let axisY = vec2<f32>(-s, c) * t.y * 0.65;
  let centerColor = target_at(center, width, height);
  let x0 = target_at(center - axisX, width, height);
  let x1 = target_at(center + axisX, width, height);
  let y0 = target_at(center - axisY, width, height);
  let y1 = target_at(center + axisY, width, height);
  let meanColor = (centerColor * 2.0 + x0 + x1 + y0 + y1) / 6.0;
  let deviation = (
    dot(abs(centerColor - meanColor), vec3<f32>(0.33333334)) * 2.0 +
    dot(abs(x0 - meanColor), vec3<f32>(0.33333334)) +
    dot(abs(x1 - meanColor), vec3<f32>(0.33333334)) +
    dot(abs(y0 - meanColor), vec3<f32>(0.33333334)) +
    dot(abs(y1 - meanColor), vec3<f32>(0.33333334))
  ) / 6.0;
  return vec4<f32>(meanColor, deviation);
}

fn paint_child_target(
  pos: vec2<f32>,
  scale: vec2<f32>,
  theta: f32,
  width: u32,
  height: u32
) -> vec3<f32> {
  let c = cos(theta);
  let s = sin(theta);
  let axisX = vec2<f32>(c, s) * scale.x * 0.45;
  let axisY = vec2<f32>(-s, c) * scale.y * 0.45;
  return (
    target_at(pos, width, height) * 2.0 +
    target_at(pos - axisX, width, height) +
    target_at(pos + axisX, width, height) +
    target_at(pos - axisY, width, height) +
    target_at(pos + axisY, width, height)
  ) / 6.0;
}

// Returns risk, split-axis (1 = local X), gate/action stage, and five-sample
// target-colour deviation. Stage 5 is the existing high-variance split;
// stage 4.5 is v2's direct source-footprint mismatch split/shrink/move.
// This is evaluated only during a density event and never adds a normal-step
// pass or readback.
fn harmful_rectangle_parent_profile(g: u32, width: u32, height: u32) -> vec4<f32> {
  let parentSplitMode = config[107] - floor(config[107] / 4.0) * 4.0;
  let transitionOnly = parentSplitMode > 1.5;
  let transitionWindow = max(200.0, floor(config[5] * 0.10));
  if (
    parentSplitMode <= 0.5 ||
    (config[40] > 1.5 && config[40] <= 3.5) ||
    config[42] <= 0.5 ||
    (transitionOnly && (
      config[4] < config[100] || config[4] >= config[100] + transitionWindow
    )) ||
    u32(config[4]) + 64u >= u32(config[5])
  ) { return vec4<f32>(0.0); }
  let t = transform[g];
  let c = color[g];
  let st = stats[g];
  let opaquePaint = config[65] > 0.5;
  let virtualSampling = config[66] > 0.5;
  let minimumCandidateAlpha = select(0.007, 0.5, opaquePaint);
  if (t.w < 0.5 || c.a < minimumCandidateAlpha || st.w <= 32.0) { return vec4<f32>(0.0); }
  let layerOrder = clamp(min(fract(t.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  // Paint owns an explicit surface stack. Planar and Virtual use standard
  // alpha/depth, so contribution and footprint mismatch are their front gate.
  if (opaquePaint && layerOrder < 0.625) { return vec4<f32>(0.0); }
  let major = max(t.x, t.y);
  let projectedMajor = major * max(config[0], config[1]) * 0.5 * ${BOUNDARY_SIGMA};
  let meanFootprint = sqrt(max(1.0, config[0] * config[1]) / max(1.0, config[2]));
  let oversizedThreshold = clamp(meanFootprint * 1.5, 6.0, 32.0);
  if (projectedMajor <= oversizedThreshold) { return vec4<f32>(0.0, 0.0, 1.0, 0.0); }
  let im = importance_at(g);
  let expectedInfluence = max(1.0, config[0] * config[1] / max(1.0, config[2]));
  let residual = importance_residual(g);
  let signal = normalized_stats(g);
  let footprint = paint_footprint_target(g, width, height);
  let highAcceptedContribution = im.x >= 8.0 && im.y >= expectedInfluence * 0.75;
  if (!highAcceptedContribution) { return vec4<f32>(0.0, 0.0, 2.0, footprint.a); }
  let footprintColorError = dot(abs(c.rgb - footprint.rgb), vec3<f32>(0.33333334));
  let deviationThreshold = select(0.055, 0.080, virtualSampling);
  let residualThreshold = select(0.035, 0.050, virtualSampling);
  let highError =
    footprint.a > deviationThreshold &&
    residual > residualThreshold &&
    signal.x + residual * 0.5 > max(0.0003, config[34]);
  let extremeRatio = select(select(1.65, 2.50, virtualSampling), 2.10, config[40] > 3.5);
  let directMismatch = footprintColorError > select(0.040, 0.080, virtualSampling);
  let footprintMismatch = footprint.a > select(0.020, 0.060, virtualSampling);
  let v2Mismatch =
    projectedMajor > oversizedThreshold * extremeRatio &&
    residual > 0.010 &&
    signal.x + residual * 0.5 > max(0.0003, config[34]) &&
    (directMismatch || footprintMismatch);
  if (!highError && !v2Mismatch) {
    return vec4<f32>(0.0, 0.0, select(3.0, 4.0, footprint.a > 0.055), footprint.a);
  }
  let ct = cos(t.z);
  let stheta = sin(t.z);
  let axisX = vec2<f32>(ct, stheta) * t.x * 0.70;
  let axisY = vec2<f32>(-stheta, ct) * t.y * 0.70;
  let divergenceX = dot(
    abs(target_at(xy[g].center - axisX, width, height) - target_at(xy[g].center + axisX, width, height)),
    vec3<f32>(0.33333334)
  );
  let divergenceY = dot(
    abs(target_at(xy[g].center - axisY, width, height) - target_at(xy[g].center + axisY, width, height)),
    vec3<f32>(0.33333334)
  );
  let useX = divergenceX >= divergenceY;
  let sizePressure = clamp(projectedMajor / oversizedThreshold - 1.0, 0.0, 3.0);
  let mismatchPressure =
    clamp((footprint.a - 0.055) / 0.12, 0.0, 1.0) *
    clamp((residual - 0.035) / 0.10, 0.0, 1.0);
  let directMismatchPressure =
    max(
      clamp((footprintColorError - 0.040) / 0.20, 0.0, 1.0),
      clamp((footprint.a - 0.020) / 0.12, 0.0, 1.0)
    ) * clamp((residual - 0.010) / 0.10, 0.0, 1.0);
  let risk = sizePressure * select(directMismatchPressure, mismatchPressure, highError) *
    clamp(im.y / expectedInfluence, 0.0, 2.0);
  return vec4<f32>(
    max(0.000001, risk),
    select(0.0, 1.0, useX),
    select(4.5, 5.0, highError),
    footprint.a
  );
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
  let center = xy[g].center;
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

fn structure_tensor_at(pos: vec2<f32>, radius: i32, width: u32, height: u32) -> vec3<f32> {
  let px = i32(round((clamp(pos.x, -1.0, 1.0) * 0.5 + 0.5) * f32(width - 1u)));
  let py = i32(round((clamp(pos.y, -1.0, 1.0) * 0.5 + 0.5) * f32(height - 1u)));
  var jxx = 0.0;
  var jxy = 0.0;
  var jyy = 0.0;
  let derivativeScale = 0.5 / f32(max(1, radius));
  for (var oy = -1; oy <= 1; oy += 1) {
    for (var ox = -1; ox <= 1; ox += 1) {
      let sx = px + ox * radius;
      let sy = py + oy * radius;
      let gx = derivativeScale * (
        target_luma_pixel(sx + radius, sy, width, height) -
        target_luma_pixel(sx - radius, sy, width, height)
      );
      let gy = derivativeScale * (
        target_luma_pixel(sx, sy + radius, width, height) -
        target_luma_pixel(sx, sy - radius, width, height)
      );
      jxx += gx * gx;
      jxy += gx * gy;
      jyy += gy * gy;
    }
  }
  return vec3<f32>(jxx, jxy, jyy) / 9.0;
}

fn structure_at(pos: vec2<f32>, width: u32, height: u32) -> vec4<f32> {
  let fine = structure_tensor_at(pos, 1, width, height);
  let tensor = select(
    fine,
    fine * 0.55 +
      structure_tensor_at(pos, 3, width, height) * 0.30 +
      structure_tensor_at(pos, 7, width, height) * 0.15,
    config[40] > 3.5
  );
  let jxx = tensor.x;
  let jxy = tensor.y;
  let jyy = tensor.z;
  let trace = jxx + jyy;
  let separation = sqrt(max(0.0, (jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy));
  let coherence = separation / max(trace, 0.00000001);
  let normalAngle = 0.5 * atan2(2.0 * jxy, jxx - jyy);
  let tangentAngle = normalAngle + 1.57079632679;
  return vec4<f32>(atan2(sin(tangentAngle), cos(tangentAngle)), coherence, trace, 0.0);
}

fn rectangle_directed_taper_theta(
  pos: vec2<f32>,
  theta: f32,
  width: u32,
  height: u32
) -> f32 {
  let flags = u32(round(config[97]));
  if (
    config[40] < 0.5 ||
    config[40] > 1.5 ||
    config[96] >= 1.0 - 0.000001 ||
    (flags & ${RECTANGLE_FLAG_EDGE_DIRECTED_TAPER}u) == 0u
  ) {
    return theta;
  }
  let normal = vec2<f32>(-sin(theta), cos(theta));
  let offset = 8.0 / max(1.0, f32(max(width, height) - 1u));
  let positiveEnergy = structure_at(pos + normal * offset, width, height).z;
  let negativeEnergy = structure_at(pos - normal * offset, width, height).z;
  return select(theta, theta + 3.14159265359, positiveEnergy > negativeEnergy + 0.0000000001);
}

fn rectangle_structure_detail(structure: vec4<f32>) -> bool {
  let flags = u32(round(config[97]));
  return
    config[40] > 0.5 &&
    config[40] < 1.5 &&
    (flags & ${RECTANGLE_FLAG_STRUCTURE_AWARE_RATIO}u) != 0u &&
    structure.y >= ${RECTANGLE_STRUCTURE_MIN_COHERENCE} &&
    structure.z >= ${RECTANGLE_STRUCTURE_MIN_ENERGY};
}

fn oil_structure_guided(structure: vec4<f32>) -> bool {
  return config[40] > 3.5 && structure.y > 0.12 && structure.z > 0.00008;
}

fn oil_hierarchy_scale(scale: vec2<f32>, structure: vec4<f32>) -> vec2<f32> {
  if (config[40] <= 3.5) { return scale; }
  let detail = clamp(sqrt(max(0.0, structure.z) / 0.0025), 0.0, 1.0);
  return scale * mix(1.65, 0.62, detail);
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
  let center = xy[g].center;
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

fn constrain_rectangle_orientation(scale: vec2<f32>, theta: f32) -> f32 {
  if (config[40] <= 0.5 || config[40] >= 1.5 || config[92] <= 0.5) {
    return theta;
  }
  let longAxisIsX = scale.x >= scale.y;
  let vertical = config[92] < 1.5;
  let targetTheta = select(
    select(1.57079632679, 0.0, longAxisIsX),
    select(0.0, 1.57079632679, longAxisIsX),
    vertical
  );
  let tolerance = clamp(config[127], 0.0, 1.57079632679);
  if (tolerance <= 0.0) { return targetTheta; }
  if (tolerance >= 1.57079632679) { return theta; }
  let offset = theta - targetTheta;
  let delta = offset - floor((offset + 1.57079632679) / 3.14159265359) * 3.14159265359;
  return theta + clamp(delta, -tolerance, tolerance) - delta;
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
  if (config[57] <= 0.5) { return false; }
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

fn structure_allocation_region_at(pos: vec2<f32>) -> u32 {
  let grid = active_region_grid();
  let uv = clamp(pos * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(0.999999));
  let cell = min(
    vec2<u32>(grid - 1u),
    vec2<u32>(uv * f32(grid))
  );
  return cell.y * grid + cell.x;
}

fn structure_allocation_weight_at(pos: vec2<f32>) -> f32 {
  if (config[68] <= 0.5 || u32(config[11]) != 1u) { return 1.0; }
  let capacity = u32(config[10]);
  let region = structure_allocation_region_at(pos);
  let base = phase45_region_base(capacity) + region * PHASE45_REGION_STRIDE;
  let current = f32(atomicLoad(&control[base]));
  let quota = f32(atomicLoad(&control[base + 9u]));
  let uniformQuota = max(1.0, config[3] / f32(active_region_count()));
  let deficit = clamp((quota - current) / uniformQuota, 0.0, 1.0);
  return mix(0.05, 2.5, deficit);
}

fn distribution_weight(g: u32, adc: bool) -> f32 {
  let t = transform[g];
  let c = color[g];
  if (t.w < 0.5 || c.a < 0.005) { return 0.0; }
  let tiltProfile = tilt_split_profile(g, u32(config[0]), u32(config[1]));
  let harmfulRectangleProfile = harmful_rectangle_parent_profile(
    g,
    u32(config[0]),
    u32(config[1])
  );
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
    if (!growthEligible && tiltProfile.x <= 0.0 && harmfulRectangleProfile.x <= 0.0) { return 0.0; }
  }
  if (config[26] > 0.5) {
    let localStructure = structure_at(xy[g].center, u32(config[0]), u32(config[1]));
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
  if (harmfulRectangleProfile.x > 0.0) {
    combined += min(4.0, harmfulRectangleProfile.x) * (0.5 + areaMass);
  }
  if (adc && config[47] > 0.0) {
    let capacity = u32(config[10]);
    let grid = active_region_grid();
    let uv = clamp(xy[g].center * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(0.999999));
    let cell = min(vec2<u32>(grid - 1u), vec2<u32>(uv * f32(grid)));
    let region = cell.y * grid + cell.x;
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
  combined *= structure_allocation_weight_at(xy[g].center);
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
fn collect_structure_allocation_counts(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  if (config[68] <= 0.5 || u32(config[11]) != 1u || g >= u32(config[2])) { return; }
  let region = structure_allocation_region_at(xy[g].center);
  let base = phase45_region_base(u32(config[10])) + region * PHASE45_REGION_STRIDE;
  atomicAdd(&control[base], 1u);
}

fn residual_tile_count() -> u32 {
  return ((u32(config[0]) + RESIDUAL_TILE_SIZE - 1u) / RESIDUAL_TILE_SIZE) *
    ((u32(config[1]) + RESIDUAL_TILE_SIZE - 1u) / RESIDUAL_TILE_SIZE);
}
fn residual_tile_blocks() -> u32 { return (residual_tile_count() + CDF_BLOCK_SIZE - 1u) / CDF_BLOCK_SIZE; }
fn residual_tile_base(capacity: u32) -> u32 { return phase45_donor_base(capacity) + capacity; }
fn residual_tile_block_sum_base(capacity: u32) -> u32 { return residual_tile_base(capacity) + residual_tile_count() + 1u; }
fn residual_tile_block_offset_base(capacity: u32) -> u32 { return residual_tile_block_sum_base(capacity) + residual_tile_blocks(); }

@compute @workgroup_size(256)
fn build_residual_tiles(@builtin(global_invocation_id) id: vec3u) {
  if (!RESIDUAL_TILE_CDF_ENABLED) { return; }
  let width = u32(config[0]);
  let height = u32(config[1]);
  let pixel = id.x;
  if (pixel >= width * height) { return; }
  let x = pixel % width;
  let y = pixel / width;
  let tileCols = (width + RESIDUAL_TILE_SIZE - 1u) / RESIDUAL_TILE_SIZE;
  let tile = (y / RESIDUAL_TILE_SIZE) * tileCols + x / RESIDUAL_TILE_SIZE;
  let quantized = u32(round(clamp(errorMap[pixel], 0.0, 1.0) * 256.0));
  atomicAdd(&control[residual_tile_base(u32(config[10])) + tile], quantized);
}

@compute @workgroup_size(256)
fn scan_residual_tile_blocks(
  @builtin(local_invocation_id) localId: vec3u,
  @builtin(workgroup_id) groupId: vec3u,
) {
  if (!RESIDUAL_TILE_CDF_ENABLED) { return; }
  let capacity = u32(config[10]);
  let count = residual_tile_count();
  let g = groupId.x * CDF_BLOCK_SIZE + localId.x;
  var weight = 0u;
  if (g < count) {
    weight = atomicLoad(&control[residual_tile_base(capacity) + g]);
  }
  residualCdfScratch[localId.x] = weight;
  workgroupBarrier();
  for (var offset = 1u; offset < CDF_BLOCK_SIZE; offset *= 2u) {
    var addend = 0u;
    if (localId.x >= offset) { addend = residualCdfScratch[localId.x - offset]; }
    workgroupBarrier();
    residualCdfScratch[localId.x] += addend;
    workgroupBarrier();
  }
  if (g < count) { atomicStore(&control[residual_tile_base(capacity) + g], residualCdfScratch[localId.x]); }
  let blockLast = min(count, (groupId.x + 1u) * CDF_BLOCK_SIZE) - 1u;
  if (g == blockLast) {
    atomicStore(&control[residual_tile_block_sum_base(capacity) + groupId.x], residualCdfScratch[localId.x]);
  }
}

@compute @workgroup_size(1)
fn scan_residual_tile_block_sums() {
  if (!RESIDUAL_TILE_CDF_ENABLED) { return; }
  let capacity = u32(config[10]);
  let blocks = residual_tile_blocks();
  var prefix = 0u;
  for (var block = 0u; block < blocks; block += 1u) {
    atomicStore(&control[residual_tile_block_offset_base(capacity) + block], prefix);
    prefix += atomicLoad(&control[residual_tile_block_sum_base(capacity) + block]);
  }
  atomicStore(&control[residual_tile_base(capacity) + residual_tile_count()], prefix);
}

@compute @workgroup_size(256)
fn add_residual_tile_block_offsets(@builtin(global_invocation_id) id: vec3u) {
  if (!RESIDUAL_TILE_CDF_ENABLED) { return; }
  let capacity = u32(config[10]);
  let g = id.x;
  if (g >= residual_tile_count()) { return; }
  let block = g / CDF_BLOCK_SIZE;
  let value = atomicLoad(&control[residual_tile_base(capacity) + g]);
  let offset = atomicLoad(&control[residual_tile_block_offset_base(capacity) + block]);
  atomicStore(&control[residual_tile_base(capacity) + g], value + offset);
}

@compute @workgroup_size(64)
fn phase45_collect_region_telemetry(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  if (u32(config[11]) != 3u || config[44] <= 0.5 || g >= count) { return; }
  let width = u32(config[0]);
  let height = u32(config[1]);
  let grid = active_region_grid();
  let uv = clamp(xy[g].center * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(0.999999));
  let cell = min(vec2<u32>(grid - 1u), vec2<u32>(uv * f32(grid)));
  let region = cell.y * grid + cell.x;
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

@compute @workgroup_size(${PHASE45_REGION_COUNT})
fn phase45_finalize_region_telemetry(@builtin(local_invocation_id) localId: vec3u) {
  let region = localId.x;
  let regionCount = active_region_count();
  let capacity = u32(config[10]);
  if (region < regionCount) {
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
  }
  workgroupBarrier();
  if (region != 0u) { return; }
  var totalDemand = 0u;
  for (var i = 0u; i < regionCount; i += 1u) { totalDemand += phase45Demand[i]; }
  var allocated = 0u;
  for (var i = 0u; i < regionCount; i += 1u) {
    let quota = u32(floor(f32(u32(config[2])) * f32(phase45Demand[i]) / max(1.0, f32(totalDemand))));
    atomicStore(&control[phase45_region_base(capacity) + i * PHASE45_REGION_STRIDE + 9u], quota);
    allocated += quota;
  }
  let remainder = u32(config[2]) - allocated;
  for (var extra = 0u; extra < remainder; extra += 1u) {
    var bestRegion = 0u;
    var bestFraction = -1.0;
    for (var i = 0u; i < regionCount; i += 1u) {
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
  let center = xy[g].center;
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
${protectedPrefix ? `  if (g < ${protectedPrefix}u) { atomicStore(&control[phase45_donor_base(capacity) + g], 0u); return; }\n` : ""}  let width = u32(config[0]);
  let height = u32(config[1]);
  let grid = active_region_grid();
  let uv = clamp(xy[g].center * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(0.999999));
  let cell = min(vec2<u32>(grid - 1u), vec2<u32>(uv * f32(grid)));
  let region = cell.y * grid + cell.x;
  let regionBase = phase45_region_base(capacity) + region * PHASE45_REGION_STRIDE;
  let energy = phase45_sample_energy(g, width, height);
  let im = importance_at(g);
  let expectedInfluence = max(1.0, config[0] * config[1] / max(1.0, config[2]));
  let residual = clamp(im.z / max(im.x, 1.0), 0.0, 1.0);
  let influence = clamp(im.y / expectedInfluence, 0.0, 16.0);
  let currentContributionNearZero =
    im.x <= ${CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_COVERAGE}.0 &&
    im.y <= ${CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_INFLUENCE};
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
  let center = xy[g].center;
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
  // Opaque Rectangle/Brush marks often fail the per-pixel removal simulation
  // even when they are absent from the accumulated current-view contribution.
  // Reuse only the same bounded near-zero cohort that the existing compaction
  // would otherwise delete at this midpoint.
  // A current-view near-zero row contributes no visible detail to protect and
  // is already eligible for physical removal by Contribution compaction.
  // Keep the local-detail guard for simulated-removal donors, but allow that
  // exact near-zero cohort to be reused instead of deleted.
  let contributionSafe = (supportSafe && localDetailSafe) || currentContributionNearZero;
  let eligible = contributionSafe && regionSurplus && lowQuantile && !nonfinite;
  if (eligible) { flags |= PHASE45_DONOR_ELIGIBLE; atomicAdd(&control[regionBase + 22u], 1u); }
  atomicStore(&control[phase45_donor_base(capacity) + g], flags);
}

@compute @workgroup_size(256)
fn build_distribution(@builtin(global_invocation_id) id: vec3u) {
  let count = u32(config[2]);
  let capacity = u32(config[10]);
  let g = id.x;
  if (g >= count) { return; }
${protectedPrefix ? `  if (g < ${protectedPrefix}u) { atomicStore(&control[cdf_base(capacity) + g], bitcast<u32>(0.0)); return; }\n` : ""}  let adc = is_adc_step(u32(config[4])) || u32(config[11]) == 3u;
  if (adc && config[45] > 0.5) {
    // Product over-density donors are write destinations in the following
    // pass. Exclude them from the read-only source distribution so multiple
    // destinations can safely share a live source without a read/write race.
    let donorRecord = atomicLoad(&control[phase45_donor_base(capacity) + g]);
    if ((donorRecord & PHASE45_DONOR_ELIGIBLE) != 0u) {
      atomicStore(&control[cdf_base(capacity) + g], bitcast<u32>(0.0));
      return;
    }
  }
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
  if (total <= 0.00000001) { return ${protectedPrefix ? `${protectedPrefix}u + seedIndex % max(count - ${protectedPrefix}u, 1u)` : "seedIndex % max(count, 1u)"}; }
  let step = config[4];
  let sample = hash_unit(f32(seedIndex) * 13.0 + step * 0.31 + select(17.0, 41.0, adc)) * total;
  var low = ${protectedPrefix}u;
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

fn pick_error_pixel(seedIndex: u32, width: u32, height: u32) -> u32 {
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
  return bestPixel;
}

fn pick_structure_allocation_region(seedIndex: u32) -> u32 {
  let capacity = u32(config[10]);
  let regionCount = active_region_count();
  var totalDebt = 0u;
  for (var region = 0u; region < regionCount; region += 1u) {
    let base = phase45_region_base(capacity) + region * PHASE45_REGION_STRIDE;
    let current = atomicLoad(&control[base]);
    let quota = atomicLoad(&control[base + 9u]);
    totalDebt += select(0u, quota - current, quota > current);
  }
  if (totalDebt == 0u) { return seedIndex % regionCount; }
  let sample = min(
    totalDebt - 1u,
    u32(hash_unit(f32(seedIndex) * 31.17 + config[4] * 0.79) * f32(totalDebt))
  );
  var cumulative = 0u;
  for (var region = 0u; region < regionCount; region += 1u) {
    let base = phase45_region_base(capacity) + region * PHASE45_REGION_STRIDE;
    let current = atomicLoad(&control[base]);
    let quota = atomicLoad(&control[base + 9u]);
    cumulative += select(0u, quota - current, quota > current);
    if (sample < cumulative) { return region; }
  }
  return regionCount - 1u;
}

fn pick_structure_allocation_pixel(seedIndex: u32, width: u32, height: u32) -> u32 {
  let region = pick_structure_allocation_region(seedIndex);
  let grid = active_region_grid();
  let regionX = region % grid;
  let regionY = region / grid;
  let x0 = regionX * width / grid;
  let x1 = max(x0 + 1u, (regionX + 1u) * width / grid);
  let y0 = regionY * height / grid;
  let y1 = max(y0 + 1u, (regionY + 1u) * height / grid);
  var bestPixel = min(width * height - 1u, y0 * width + x0);
  var bestScore = -1.0;
  for (var candidate = 0u; candidate < 32u; candidate += 1u) {
    let ux = hash_unit(f32(seedIndex) * 17.17 + f32(candidate) * 91.73 + config[4] * 0.37);
    let uy = hash_unit(f32(seedIndex) * 43.11 + f32(candidate) * 37.19 + config[4] * 0.61);
    let x = min(width - 1u, x0 + u32(ux * f32(max(1u, x1 - x0))));
    let y = min(height - 1u, y0 + u32(uy * f32(max(1u, y1 - y0))));
    let pixel = y * width + x;
    let score = errorMap[pixel] + hash_unit(f32(pixel + seedIndex + candidate)) * 0.0001;
    if (score > bestScore) {
      bestPixel = pixel;
      bestScore = score;
    }
  }
  return bestPixel;
}

fn error_pixel_position(pixel: u32, width: u32, height: u32) -> vec2<f32> {
  let px = pixel % width;
  let py = pixel / width;
  let x = select(0.0, (f32(px) / f32(width - 1u)) * 2.0 - 1.0, width > 1u);
  let y = select(0.0, (f32(py) / f32(height - 1u)) * 2.0 - 1.0, height > 1u);
  return constrain_xy(vec2<f32>(x, y));
}

fn pick_error_position(seedIndex: u32, width: u32, height: u32) -> vec2<f32> {
  return error_pixel_position(pick_error_pixel(seedIndex, width, height), width, height);
}

fn pick_residual_tile_pixel(seedIndex: u32, width: u32, height: u32, capacity: u32) -> u32 {
  let tileCount = residual_tile_count();
  let base = residual_tile_base(capacity);
  let total = atomicLoad(&control[base + tileCount]);
  if (total == 0u) { return pick_error_pixel(seedIndex, width, height); }
  let sample = min(total - 1u, u32(hash_unit(f32(seedIndex) * 23.47 + config[4] * 0.73 + 11.0) * f32(total)));
  var low = 0u;
  var high = tileCount;
  loop {
    if (low >= high) { break; }
    let mid = low + (high - low) / 2u;
    let value = atomicLoad(&control[base + mid]);
    if (value <= sample) { low = mid + 1u; } else { high = mid; }
  }
  let tile = min(tileCount - 1u, low);
  let tileCols = (width + RESIDUAL_TILE_SIZE - 1u) / RESIDUAL_TILE_SIZE;
  let tileX = (tile % tileCols) * RESIDUAL_TILE_SIZE;
  let tileY = (tile / tileCols) * RESIDUAL_TILE_SIZE;
  var bestPixel = min(width * height - 1u, tileY * width + tileX);
  var bestScore = -1.0;
  for (var local = 0u; local < RESIDUAL_TILE_SIZE * RESIDUAL_TILE_SIZE; local += 1u) {
    let x = tileX + local % RESIDUAL_TILE_SIZE;
    let y = tileY + local / RESIDUAL_TILE_SIZE;
    if (x >= width || y >= height) { continue; }
    let pixel = y * width + x;
    let score = errorMap[pixel] + hash_unit(f32(pixel + seedIndex + local)) * 0.0001;
    if (score > bestScore) {
      bestPixel = pixel;
      bestScore = score;
    }
  }
  return bestPixel;
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
  let sampledSource = pick_source(index, oldCount, adc);
  let sampledParentProfile = harmful_rectangle_parent_profile(
    sampledSource,
    u32(config[0]),
    u32(config[1])
  );
  // Residual CDF sampling can repeatedly miss a persistent wrong broad parent.
  // Probe one deterministic current row per requested child and prefer it only
  // when the full front/size/contribution/image-mismatch profile qualifies.
  let probeSource = ${protectedPrefix ? `${protectedPrefix}u + (local * 2654435761u + step * 2246822519u) % (oldCount - ${protectedPrefix}u)` : "(local * 2654435761u + step * 2246822519u) % oldCount"};
  let probeParentProfile = harmful_rectangle_parent_profile(
    probeSource,
    u32(config[0]),
    u32(config[1])
  );
  let sampledAllocationWeight = structure_allocation_weight_at(xy[sampledSource].center);
  let probeAllocationWeight = structure_allocation_weight_at(xy[probeSource].center);
  let useProbeSource = probeParentProfile.x > sampledParentProfile.x &&
    probeAllocationWeight >= sampledAllocationWeight * 0.5;
  let source = select(sampledSource, probeSource, useProbeSource);
  let sourceSignal = normalized_stats(source);
  let sourceImportance = importance_at(source);
  let tiltProfile = tilt_split_profile(source, u32(config[0]), u32(config[1]));
  let harmfulRectangleProfile = select(
    sampledParentProfile,
    probeParentProfile,
    useProbeSource
  );
  // Tilt-risk replacement is an ADC operation. Running it on every ordinary
  // growth event repeatedly duplicates broad splats and inflates tile work.
  let tiltRisk = adc && tiltProfile.x > 0.0;
  let residualSupport = max(sourceSignal.y, importance_residual(source));
  let major = max(transform[source].x, transform[source].y);
  let projectedMajor = major * max(config[0], config[1]) * 0.5 * ${BOUNDARY_SIGMA};
  let signalThreshold = select(config[34], config[48], adc);
  let residualThreshold = select(0.0, config[49], adc);
  let highSignal = sourceSignal.x + residualSupport * 0.5 > signalThreshold && residualSupport >= residualThreshold;
  let harmfulRectangleParent = harmfulRectangleProfile.x > 0.0;
  var eligible = highSignal || tiltRisk || harmfulRectangleParent;
  if (config[17] > 0.5) {
    eligible = (highSignal && sourceImportance.x >= 4.0 && residualSupport >= 0.01) || tiltRisk || harmfulRectangleParent;
  }
  // Scale the split decision with the mean pixel footprint represented by one
  // active splat. This keeps the decision stable across image resolutions and
  // density stages instead of baking in a 3 px threshold.
  let splitThresholdPx = clamp(0.75 * sqrt(max(1.0, config[0] * config[1]) / max(1.0, config[2])), 1.0, 32.0);
  let mode = select(2u, 1u, projectedMajor > splitThresholdPx || tiltRisk || harmfulRectangleParent);
  var finalMode = select(0u, mode, eligible);
  let eventBase = capacity * 2u;
  if (harmfulRectangleProfile.z >= 2.0) { atomicAdd(&control[eventBase + 27u], 1u); }
  if (harmfulRectangleProfile.z >= 3.0) { atomicAdd(&control[eventBase + 28u], 1u); }
  if (harmfulRectangleProfile.z >= 5.0) { atomicAdd(&control[eventBase + 29u], 1u); }
  if (harmfulRectangleParent) { atomicAdd(&control[eventBase + 30u], 1u); }
  if (config[68] > 0.5) {
    if (structure_allocation_weight_at(xy[source].center) > 1.0) {
      atomicAdd(&control[eventBase + 34u], 1u);
    } else {
      atomicAdd(&control[eventBase + 33u], 1u);
    }
  }
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
  var encodedSelection = encode_selection(source, finalMode);
  if ((RESIDUAL_ORACLE_ENABLED || RESIDUAL_TILE_CDF_ENABLED) && finalMode == 0u) {
    var selectedPixel = pick_error_pixel(index, u32(config[0]), u32(config[1]));
    if (config[68] > 0.5) {
      selectedPixel = pick_structure_allocation_pixel(index, u32(config[0]), u32(config[1]));
    } else if (RESIDUAL_TILE_CDF_ENABLED) {
      selectedPixel = pick_residual_tile_pixel(index, u32(config[0]), u32(config[1]), capacity);
    }
    encodedSelection = (selectedPixel + 1u) & SOURCE_MASK;
  }
  atomicStore(&control[capacity + local], encodedSelection);
  if (config[42] <= 0.5 && tiltRisk && finalMode != 0u) { atomicAdd(&control[eventBase + 19u], 1u); }
  if (adc && config[17] > 0.5) {
    if (eligible) { atomicAdd(&control[eventBase + 9u], 1u); }
    else { atomicAdd(&control[eventBase + 10u], 1u); }
  }
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
  let stageMinScale = vec2<f32>(max(${MIN_SPLAT_SCALE}, config[60]));
  let baseScaleFloor = baseScale * clamp(config[61], 0.0, 1.0);
  let index = oldCount + local;
  if (index >= targetCount) { return; }
  let encoded = atomicLoad(&control[capacity + local]);
  let mode = encoded >> 30u;
  let eventBase = capacity * 2u;

  // A claim conflict becomes a source-independent reseed. It must not read a
  // source that another workgroup may be replacing in this dispatch.
  if (mode == 0u) {
    let gridPos = grid_position(index, targetCount, cols, rows);
    var residualPos = select(
      pick_error_position(index, width, height),
      error_pixel_position(pick_structure_allocation_pixel(index, width, height), width, height),
      config[68] > 0.5
    );
    if (RESIDUAL_ORACLE_ENABLED || RESIDUAL_TILE_CDF_ENABLED) {
      residualPos = error_pixel_position((encoded & SOURCE_MASK) - 1u, width, height);
    }
    let residualPriority = config[17] > 0.5 || config[18] > 0.5;
    let gridResidual = error_at_position(gridPos, width, height);
    let residualError = error_at_position(residualPos, width, height);
    let materiallyWorse =
      residualError > gridResidual + 0.04;
    let baselineUseResidual = materiallyWorse &&
      (residualPriority || hash_unit(f32(index) * 29.7 + f32(step) * 0.11) < 0.15);
    let structureAllocationReseed = config[68] > 0.5 &&
      hash_unit(f32(index) * 47.3 + f32(step) * 0.19) < 0.15;
    var useResidual = baselineUseResidual || structureAllocationReseed;
    var nextPos = constrain_xy(select(gridPos, residualPos, useResidual));
    var nextScale = max(baseScaleFloor, stageMinScale);
    var nextTheta = 0.0;
    let localStructure = structure_at(nextPos, width, height);
    let localError = error_at_position(nextPos, width, height);
    let structureGuided =
      config[19] > 0.5 &&
      ((localStructure.y > 0.45 && localStructure.z > 0.0004 && localError > 0.02) || oil_structure_guided(localStructure));
    let adaptiveDetail =
      (config[29] > 0.5 && localStructure.y >= config[31] && localStructure.z > 0.0004 && localError > 0.02) ||
      rectangle_structure_detail(localStructure);
    let surfaceMaxAnisotropy = max(config[55], 1.0);
    let localMaxAnisotropy = select(min(maxAnisotropy, surfaceMaxAnisotropy), maxAnisotropy, adaptiveDetail);
    nextScale = oil_hierarchy_scale(nextScale, localStructure);
    if (structureGuided) {
      let areaRadius = sqrt(max(0.00000001, nextScale.x * nextScale.y));
      let ratio = min(localMaxAnisotropy, select(1.6, localMaxAnisotropy, config[28] > 0.5 || adaptiveDetail));
      nextScale = vec2<f32>(areaRadius * sqrt(ratio), areaRadius / sqrt(ratio));
      nextTheta = rectangle_directed_taper_theta(
        nextPos,
        localStructure.x,
        width,
        height
      );
    }
    let reseedScaleCeiling = baseScale * select(0.9, 1.15, config[40] > 3.5);
    nextTheta = constrain_rectangle_orientation(nextScale, nextTheta);
    nextScale = constrain_scale(nextPos, max(max(min(nextScale, reseedScaleCeiling), baseScaleFloor), stageMinScale), nextTheta, localMaxAnisotropy);
    nextPos = constrain_position(nextPos, nextScale, nextTheta);
    let randomizedReseedLayer = select(
      0.0,
      min(hash_unit(f32(index) * 0.61803398875) * ${LAYER_CODE_RANGE}, ${LAYER_CODE_RANGE}),
      config[35] > 0.5
    );
    // A freshly seeded Oil mark starts in the deepest paint layer. It can only
    // reach the surface after its source-footprint color has been fitted.
    let reseedLayer = select(randomizedReseedLayer, 0.0, config[40] > 3.5 && config[65] > 0.5);
    xy[index].center = nextPos;
    xy[index].rawDepth = select(
      0.0,
      ${DEFAULT_LAYERED_BRUSH_TAPER},
      config[40] > 3.5 && config[95] > 0.5
    );
    xy[index].depthGradient = 0.0;
    transform[index] = vec4<f32>(nextScale, nextTheta, select(1.0, 2.0, adaptiveDetail) + reseedLayer);
    color[index] = vec4<f32>(
      target_at(nextPos, width, height),
      select(0.005, config[118], config[65] > 0.5)
    );
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
  let harmfulRectangleProfile = harmful_rectangle_parent_profile(source, width, height);
  var tiltTrueSplit = mode == 1u && tiltProfile.x > 0.0 && config[42] > 0.5;
  let sourceLayerOrder = clamp(min(fract(sourceT.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  let sourceProjectedMajor = max(sourceT.x, sourceT.y) * max(config[0], config[1]) * 0.5 * ${BOUNDARY_SIGMA};
  let sourceMeanFootprint = sqrt(max(1.0, config[0] * config[1]) / max(1.0, config[2]));
  let brushPaintSource = config[40] > 3.5;
  let sourceOversizedThreshold = clamp(sourceMeanFootprint * 1.5, 6.0, 32.0);
  let paintResidualTrueSplit =
    mode == 1u &&
    config[65] > 0.5 &&
    (config[40] > 0.5 && (config[40] < 1.5 || config[40] > 3.5)) &&
    sourceLayerOrder >= select(0.625, 0.500, brushPaintSource) &&
    sourceProjectedMajor > sourceOversizedThreshold &&
    sourceImportance.x >= select(8.0, 4.0, brushPaintSource) &&
    importance_residual(source) > select(0.020, 0.010, brushPaintSource);
  let harmfulRectangleTrueSplit =
    mode == 1u && harmfulRectangleProfile.x > 0.0 && config[42] > 0.5;
  var paintParentTrueSplit = harmfulRectangleTrueSplit || paintResidualTrueSplit;
  if (paintParentTrueSplit) {
    let replacementCap = select(
      max(1u, min(32u, u32(max(1.0, config[2]) * 0.005))),
      max(1u, min(8u, u32(max(1.0, config[2]) * 0.001))),
      config[66] > 0.5
    );
    let replacementTicket = atomicAdd(&control[eventBase + 31u], 1u);
    if (replacementTicket >= replacementCap) {
      atomicSub(&control[eventBase + 31u], 1u);
      paintParentTrueSplit = false;
    }
  }
  let trueSplit = tiltTrueSplit || paintParentTrueSplit;
  let profileUseX = select(harmfulRectangleProfile.y, tiltProfile.y, tiltTrueSplit);
  let profileTrueSplit = tiltTrueSplit || harmfulRectangleTrueSplit;
  let splitUseX = select(useX, profileUseX > 0.5, profileTrueSplit);
  let sourceLongAngle = sourceT.z + select(1.57079632679, 0.0, splitUseX);
  let axis = vec2<f32>(cos(sourceLongAngle), sin(sourceLongAngle));
  let perp = vec2<f32>(-axis.y, axis.x);
  let side = select(-1.0, 1.0, hash_unit(f32(index) * 53.0 + f32(step) * 1.7) > 0.5);
  let major = select(sourceT.y, sourceT.x, splitUseX);
  let minor = max(0.0015, min(sourceT.x, sourceT.y));
  let jitter =
    (hash_unit(f32(index) * 71.0 + f32(step) * 2.3) - 0.5) * minor * 0.35;
  var nextPos = xy[source].center + axis * major * 0.48 * side + perp * jitter;
  var nextScale = sourceT.xy * 0.98;
  if (mode == 1u) {
    let splitOffset = select(0.55, select(0.42, 0.34, tiltTrueSplit), trueSplit);
    nextPos = xy[source].center + axis * major * splitOffset * select(side, 1.0, trueSplit);
    let trueSplitShrink = select(0.58, clamp(config[41], 0.5, 0.85), tiltTrueSplit);
    let splitShrink = select(0.72, trueSplitShrink, trueSplit);
    let axisShrink = sourceT.xy * vec2<f32>(select(0.94, splitShrink, splitUseX), select(splitShrink, 0.94, splitUseX));
    nextScale = select(axisShrink, sourceT.xy * splitShrink, tiltTrueSplit);
  } else if (mode == 2u) {
    nextPos = xy[source].center + axis * major * 0.24 * side + perp * jitter;
    nextScale = sourceT.xy * 0.96;
  }
  // Paint children inherit the parent's physical long-axis direction. A split
  // may shrink one local scale enough to swap x/y dominance; swap the scale
  // components back instead of rotating the child by 90 degrees.
  let paintChild = config[40] > 0.5 && (config[40] < 1.5 || config[40] > 3.5);
  let sourceMajorIsX = sourceT.x >= sourceT.y;
  if (paintChild && (nextScale.x >= nextScale.y) != sourceMajorIsX) {
    nextScale = nextScale.yx;
  }
  nextPos = constrain_xy(nextPos);
  var nextTheta = sourceT.z;
  let localStructure = structure_at(nextPos, width, height);
  let localError = error_at_position(nextPos, width, height);
  let structureGuided =
    !trueSplit &&
    config[40] < 0.5 &&
    config[19] > 0.5 &&
    ((localStructure.y > 0.45 && localStructure.z > 0.0004 && localError > 0.02) || oil_structure_guided(localStructure));
  let adaptiveDetail =
    (config[29] > 0.5 && localStructure.y >= config[31] && localStructure.z > 0.0004 && localError > 0.02) ||
    rectangle_structure_detail(localStructure);
  let inheritedDetail = floor(sourceT.w) >= 2.0;
  let detailTagged = adaptiveDetail || inheritedDetail;
  let surfaceMaxAnisotropy = max(config[55], 1.0);
  var localMaxAnisotropy = select(min(maxAnisotropy, surfaceMaxAnisotropy), maxAnisotropy, detailTagged);
  if (structureGuided) {
    let areaRadius = sqrt(max(0.00000001, nextScale.x * nextScale.y));
    let ratio = min(localMaxAnisotropy, select(1.6, localMaxAnisotropy, config[28] > 0.5 || adaptiveDetail));
    nextScale = vec2<f32>(areaRadius * sqrt(ratio), areaRadius / sqrt(ratio));
    nextTheta = rectangle_directed_taper_theta(
      nextPos,
      localStructure.x,
      width,
      height
    );
  }
  if (!trueSplit) { nextScale = min(nextScale, baseScale * 0.9); }
  let scaleFloor = max(select(baseScaleFloor, vec2<f32>(${MIN_SPLAT_SCALE}), trueSplit), stageMinScale);
  nextTheta = constrain_rectangle_orientation(nextScale, nextTheta);
  nextScale = constrain_scale(nextPos, max(nextScale, scaleFloor), nextTheta, localMaxAnisotropy);
  nextPos = constrain_position(nextPos, nextScale, nextTheta);

  var replacementSourcePos = xy[source].center;
  var replacementSourceScale = sourceT.xy;
  if (trueSplit) {
    let replacementOffset = select(0.42, 0.34, tiltTrueSplit);
    replacementSourcePos = constrain_xy(xy[source].center - axis * major * replacementOffset);
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
  if (trueSplit) {
    let replacementArea = nextScale.x * nextScale.y + replacementSourceScale.x * replacementSourceScale.y;
    let replacementOpacity = massShare / max(0.00000001, replacementArea);
    // Opaque parents commonly need a value just above 0.99 after shrinking.
    // Rejecting that case disabled every true split. Keep the symmetric split,
    // clamp the standard per-splat alpha, and report the small mass shortfall.
    if (replacementOpacity > 0.99) { atomicAdd(&control[eventBase + 21u], 1u); }
    childOpacity = clamp(replacementOpacity, 0.005, 0.99);
  }
  if (config[65] > 0.5) {
    childOpacity = select(
      config[88],
      clamp(childOpacity, config[88], config[118]),
      config[90] > 0.5
    );
  }
  let targetColor = target_at(nextPos, width, height);
  var childColor = select(
    sourceC.rgb * 0.25 + targetColor * 0.75,
    sourceC.rgb,
    trueSplit
  );
  if (paintParentTrueSplit) {
    childColor = select(
      paint_child_target(nextPos, nextScale, sourceT.z, width, height),
      sourceC.rgb,
      config[66] > 0.5
    );
  }
  if (config[40] > 3.5 && !tiltTrueSplit && !paintParentTrueSplit) {
    childColor = targetColor;
  }
  xy[index].center = nextPos;
  xy[index].rawDepth = xy[source].rawDepth;
  xy[index].depthGradient = 0.0;
  let inheritedLayer = min(fract(sourceT.w), ${LAYER_CODE_RANGE} * 0.999999);
  var childLayer = inheritedLayer;
  let parentLayerScaled =
    clamp(inheritedLayer / ${LAYER_CODE_RANGE}, 0.0, 0.999999) * max(2.0, config[82]);
  let parentLayerId = floor(parentLayerScaled);
  let parentInLayerOrder = fract(parentLayerScaled);
  let sameLayerChild =
    ((parentLayerId + min(0.999999, parentInLayerOrder + 0.25)) / max(2.0, config[82])) *
    ${LAYER_CODE_RANGE};
  let childTargetColorError = dot(abs(childColor - targetColor), vec3<f32>(0.33333334));
  let guardedSameLayerAdvance =
    config[86] > 0.5 &&
    config[65] > 0.5 &&
    (mode == 1u || mode == 2u) &&
    localError > 0.02 &&
    childTargetColorError <= 0.075;
  childLayer = select(childLayer, sameLayerChild, guardedSameLayerAdvance);
  // QA candidate: a mode-1 split is a new surface hypothesis. Put the child
  // at the absolute front, independent of its parent or detail tag.
  childLayer = select(
    childLayer,
    ${LAYER_CODE_RANGE} * 0.999999,
    (u32(round(config[107])) & 4u) != 0u && mode == 1u
  );
  let childTag = select(1.0, 2.0, detailTagged);
  transform[index] = vec4<f32>(nextScale, nextTheta, childTag + childLayer);
  color[index] = vec4<f32>(childColor, childOpacity);
  stats[index] = select(sourceStats, sourceStats * 0.5, trueSplit);
  // A new opaque child has not contributed yet. Inheriting the parent's
  // visibility made fully hidden children look important to prune.
  // Virtual symmetric true splits still divide the measured source history.
  var childImportance = select(
    sourceImportance * 0.5,
    vec4<f32>(0.0),
    config[86] > 0.5 && config[65] > 0.5 && !trueSplit,
  );
  if (config[25] > 0.5 && !trueSplit) { childImportance.w = sourceImportance.w; }
  stats[capacity + index] = childImportance;
  if (trueSplit) {
    var replacementColor = sourceC.rgb;
    if (paintParentTrueSplit) {
      replacementColor = select(
        paint_child_target(
          replacementSourcePos,
          replacementSourceScale,
          sourceT.z,
          width,
          height
        ),
        sourceC.rgb,
        config[66] > 0.5
      );
    }
    xy[source].center = replacementSourcePos;
    xy[source].depthGradient = 0.0;
    transform[source] = vec4<f32>(replacementSourceScale, sourceT.z, sourceT.w);
    color[source] = vec4<f32>(replacementColor, childOpacity);
    stats[source] = sourceStats * 0.5;
    stats[capacity + source] = sourceImportance * 0.5;
    atomicAdd(&control[eventBase + 20u], 1u);
  }
  if (paintParentTrueSplit) {
    atomicAdd(&control[eventBase + 32u], 1u);
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
${protectedPrefix ? `  if (g < ${protectedPrefix}u) { return; }\n` : ""}  let t = transform[g];
  let c = color[g];
  let st = stats[g];
  let signal = normalized_stats(g);
  let candidateImportance = importance_score(g);
  let radiusPx = max(t.x, t.y) * max(f32(width), f32(height)) * 1.25;
  let layerOrder = clamp(min(fract(t.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  let paintTarget = paint_footprint_target(g, width, height);
  let centerColorError = dot(abs(c.rgb - target_at(xy[g].center, width, height)), vec3<f32>(0.33333334));
  let footprintColorError = dot(abs(c.rgb - paintTarget.rgb), vec3<f32>(0.33333334));
  let paintFootprintErrorThreshold = select(0.20, 0.12, config[40] > 3.5);
  let paintFootprintConsistent =
    config[40] > 3.5 || paintTarget.a <= 0.055;
  let frontPaintOutlier =
    config[85] > 0.5 &&
    (config[40] > 3.5 || config[99] > 0.5) &&
    st.w > 32.0 &&
    layerOrder >= 0.625 &&
    importance_residual(g) > 0.045 &&
    centerColorError > 0.075 &&
    paintFootprintConsistent &&
    footprintColorError > paintFootprintErrorThreshold;
  if (config[99] > 0.5 && !frontPaintOutlier) { return; }
  let deepLowInfluence = config[62] > 0.5 && config[35] > 0.5 && st.w > 32.0 &&
    layerOrder <= config[63] && candidateImportance < config[64] && importance_residual(g) < 0.035;
  let inactiveMcmc = t.w < 0.5 || c.a < 0.006 || radiusPx < 0.55 ||
    (st.w > 32.0 && signal.y < 0.012 && signal.z < 0.00008) || deepLowInfluence || frontPaintOutlier;
  let lowImportanceNoise = config[16] > 0.5 && st.w > 32.0 && candidateImportance < 0.45 && importance_residual(g) < 0.035;
  let candidateStructure = structure_at(xy[g].center, width, height);
  let lowSignificanceSmooth = config[27] > 0.5 && st.w > 32.0 && candidateImportance < 0.75 && importance_residual(g) < 0.02 && candidateStructure.z < 0.0004;
  let inactiveAdc = t.w < 0.5 || c.a < 0.025 || radiusPx < 0.65 || (st.w > 32.0 && signal.z < 0.00002 && signal.y < 0.025) || lowImportanceNoise || lowSignificanceSmooth || frontPaintOutlier;
  var inactive = select(inactiveMcmc, inactiveAdc, adcRecycle);
  let phase45DonorRecord = atomicLoad(&control[phase45_donor_base(capacity) + g]);
  let productOverdensity = adcRecycle && config[45] > 0.5;
  if (productOverdensity) { inactive = (phase45DonorRecord & PHASE45_DONOR_ELIGIBLE) != 0u; }
  let lateRecycle = adcRecycle && u32(step) > u32(config[15]);
  let adcSelectionRate = select(config[50], config[51], lateRecycle);
  let selectedRate = select(0.02, adcSelectionRate, adcRecycle);
  let selected = frontPaintOutlier || hash_unit(f32(g) * 37.0 + step * 0.137) < selectedRate;
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
  if (adcRecycle && !productOverdensity && sourceSignal.x + sourceSignal.y * 0.5 <= 0.00015 && !sourceTiltRisk) { return; }
  let token = (g + 1u) & ROLE_TOKEN_MASK;
  let destinationRole = ROLE_DESTINATION | token;
  if (!try_claim_role(g, destinationRole)) {
    atomicAdd(&control[capacity * 2u + 16u], 1u);
    return;
  }
  if (!productOverdensity) {
    let sourceRole = select(ROLE_SOURCE_OTHER, ROLE_SOURCE_SPLIT, sourceTiltRisk) | token;
    if (!try_claim_role(source, sourceRole)) {
      rollback_role(g, destinationRole);
      atomicAdd(&control[capacity * 2u + 16u], 1u);
      return;
    }
  }
  atomicAdd(&control[capacity * 2u + 17u], 1u);
  if (sourceTiltRisk) { atomicAdd(&control[capacity * 2u + 19u], 1u); }
  var relocationMode = select(0u, 3u, adcRecycle);
  if (frontPaintOutlier) { relocationMode = 2u; }
  atomicStore(&control[capacity + g], encode_selection(source, relocationMode));
  if (frontPaintOutlier) { atomicAdd(&control[capacity * 2u + 22u], 1u); }
  if (adcRecycle && config[45] > 0.5) {
    let donorRegion = (phase45DonorRecord >> 8u) & 255u;
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
  let eventBase = capacity * 2u;
  let stageMinScale = vec2<f32>(max(${MIN_SPLAT_SCALE}, config[60]));
  let baseScaleFloor = baseScale * clamp(config[61], 0.0, 1.0);
  if (g >= count) { return; }
${protectedPrefix ? `  if (g < ${protectedPrefix}u) { return; }\n` : ""}  let encoded = atomicLoad(&control[capacity + g]);
  if ((encoded & SOURCE_MASK) == 0u) { return; }
  let source = (encoded & SOURCE_MASK) - 1u;
  let selectionMode = encoded >> 30u;
  let paintOutlierRecycle = selectionMode == 2u;
  let adcRecycle = selectionMode == 3u;
  let productOverdensity = adcRecycle && config[45] > 0.5;
  let sourceT = transform[source];
  let sourceC = color[source];
  let sourceStats = stats[source];
  let sourceImportance = importance_at(source);
  let tiltProfile = tilt_split_profile(source, width, height);
  let sourceClaim = atomicLoad(&control[source]);
  var tiltTrueSplit = adcRecycle && (sourceClaim & ROLE_MASK) == ROLE_SOURCE_SPLIT && tiltProfile.x > 0.0;
  let destinationStructure = structure_at(xy[g].center, width, height);
  let useX = select(sourceT.x >= sourceT.y, tiltProfile.y > 0.5, tiltTrueSplit);
  let sourceStructure = structure_at(xy[source].center, width, height);
  let sourceLongAngle = sourceT.z + select(1.57079632679, 0.0, useX);
  let axis = vec2<f32>(cos(sourceLongAngle), sin(sourceLongAngle));
  let perp = vec2<f32>(-axis.y, axis.x);
  let side = select(-1.0, 1.0, hash_unit(f32(g) * 53.0 + step * 1.7) > 0.5);
  let jitter =
    (hash_unit(f32(g) * 71.0 + step * 2.3) - 0.5) * min(sourceT.x, sourceT.y) * 0.35;
  let major = select(sourceT.y, sourceT.x, useX);
  var nextPos = constrain_xy(xy[source].center + axis * major * select(select(0.52, 0.55, adcRecycle) * side, 0.34, tiltTrueSplit) + perp * select(jitter, 0.0, tiltTrueSplit));
  if (productOverdensity) {
    nextPos = error_pixel_position(
      pick_structure_allocation_pixel(g + 13007u, width, height),
      width,
      height
    );
  }
  var splitScale = min(sourceT.xy, baseScale * 0.9);
  if (adcRecycle) {
    let splitShrink = select(0.72, clamp(config[41], 0.5, 0.85), tiltTrueSplit);
    let axisShrink = sourceT.xy * vec2<f32>(select(0.94, splitShrink, useX), select(splitShrink, 0.94, useX));
    splitScale = select(axisShrink, sourceT.xy * splitShrink, tiltTrueSplit);
  }
  let paintChild = config[40] > 0.5 && (config[40] < 1.5 || config[40] > 3.5);
  let sourceMajorIsX = sourceT.x >= sourceT.y;
  if (paintChild && (splitScale.x >= splitScale.y) != sourceMajorIsX) {
    splitScale = splitScale.yx;
  }
  var nextTheta = sourceT.z;
  let localStructure = structure_at(nextPos, width, height);
  let localError = error_at_position(nextPos, width, height);
  let structureGuided =
    !tiltTrueSplit &&
    config[40] < 0.5 &&
    config[19] > 0.5 &&
    ((localStructure.y > 0.45 && localStructure.z > 0.0004 && localError > 0.02) || oil_structure_guided(localStructure));
  let adaptiveDetail =
    (config[29] > 0.5 && localStructure.y >= config[31] && localStructure.z > 0.0004 && localError > 0.02) ||
    rectangle_structure_detail(localStructure);
  let inheritedDetail = floor(sourceT.w) >= 2.0;
  let detailTagged = adaptiveDetail || inheritedDetail;
  let surfaceMaxAnisotropy = max(config[55], 1.0);
  var localMaxAnisotropy = select(min(maxAnisotropy, surfaceMaxAnisotropy), maxAnisotropy, detailTagged);
  if (structureGuided) {
    let areaRadius = sqrt(max(0.00000001, splitScale.x * splitScale.y));
    let ratio = min(localMaxAnisotropy, select(1.6, localMaxAnisotropy, config[28] > 0.5 || adaptiveDetail));
    splitScale = vec2<f32>(areaRadius * sqrt(ratio), areaRadius / sqrt(ratio));
    nextTheta = rectangle_directed_taper_theta(
      nextPos,
      localStructure.x,
      width,
      height
    );
  }
  nextTheta = constrain_rectangle_orientation(splitScale, nextTheta);
  var nextScale = constrain_scale(nextPos, max(max(splitScale, baseScaleFloor), stageMinScale), nextTheta, localMaxAnisotropy);
  nextPos = constrain_position(nextPos, nextScale, nextTheta);
  var replacementSourcePos = xy[source].center;
  var replacementSourceScale = sourceT.xy;
  if (tiltTrueSplit) {
    replacementSourcePos = constrain_xy(xy[source].center - axis * major * 0.34);
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
  if (config[65] > 0.5) {
    childOpacity = select(
      config[88],
      clamp(childOpacity, config[88], config[118]),
      config[90] > 0.5
    );
  }
  let targetColor = target_at(nextPos, width, height);
  var nextColor = select(
    sourceC.rgb * select(0.7, 0.6, adcRecycle) + targetColor * select(0.3, 0.4, adcRecycle),
    sourceC.rgb,
    tiltTrueSplit
  );
  if ((config[40] > 3.5 || paintOutlierRecycle || productOverdensity) && !tiltTrueSplit) {
    nextColor = targetColor;
  }
  xy[g].center = nextPos;
  xy[g].rawDepth = xy[source].rawDepth;
  xy[g].depthGradient = 0.0;
  let inheritedLayer = min(fract(sourceT.w), ${LAYER_CODE_RANGE} * 0.999999);
  let childTag = select(1.0, 2.0, detailTagged);
  var childLayer = inheritedLayer;
  childLayer = select(
    childLayer,
    ${LAYER_CODE_RANGE} * 0.999999,
    (u32(round(config[107])) & 4u) != 0u && tiltTrueSplit
  );
  transform[g] = vec4<f32>(nextScale, nextTheta, childTag + childLayer);
  color[g] = vec4<f32>(nextColor, childOpacity);
  stats[g] = select(sourceStats, sourceStats * 0.5, tiltTrueSplit);
  stats[capacity + g] = select(
    vec4<f32>(0.0, 0.0, 0.0, select(0.0, sourceImportance.w, config[25] > 0.5)),
    sourceImportance * 0.5,
    tiltTrueSplit
  );
  if (tiltTrueSplit) {
    xy[source].center = replacementSourcePos;
    xy[source].depthGradient = 0.0;
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
fn apply_final_brush_repair(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
${protectedPrefix ? `  if (g < ${protectedPrefix}u) { return; }\n` : ""}  let count = u32(config[2]);
  let capacity = u32(config[10]);
  if (
    g >= count ||
    config[40] <= 3.5 ||
    config[85] <= 0.5 ||
    config[99] <= 0.5
  ) { return; }
  let width = u32(config[0]);
  let height = u32(config[1]);
  let t = transform[g];
  let paintTarget = paint_footprint_target(g, width, height);
  let centerColorError = dot(
    abs(color[g].rgb - target_at(xy[g].center, width, height)),
    vec3<f32>(0.33333334)
  );
  let footprintColorError = dot(
    abs(color[g].rgb - paintTarget.rgb),
    vec3<f32>(0.33333334)
  );
  let eventBase = capacity * 2u;
  if (
    paintTarget.a <= 0.055 &&
    centerColorError > 0.075 &&
    footprintColorError > 0.12
  ) {
    color[g] = vec4<f32>(paintTarget.rgb, color[g].a);
    atomicAdd(&control[eventBase + 23u], 1u);
    return;
  }
  let radiusPx = max(t.x, t.y) * max(f32(width), f32(height)) * 1.25;
  if (
    paintTarget.a > 0.055 &&
    footprintColorError > 0.12 &&
    radiusPx > 3.0 &&
    importance_residual(g) > 0.02
  ) {
    var trimmed = t;
    let stageMinScale = max(vec2<f32>(${MIN_SPLAT_SCALE}), vec2<f32>(config[60]));
    if (trimmed.x >= trimmed.y) {
      trimmed.x = max(trimmed.x * 0.65, stageMinScale.x);
    } else {
      trimmed.y = max(trimmed.y * 0.65, stageMinScale.y);
    }
    transform[g] = trimmed;
    atomicAdd(&control[eventBase + 24u], 1u);
  }
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
  }

  global.Image2SplatPaintDensityShader = Object.freeze({ create });
})(globalThis);
