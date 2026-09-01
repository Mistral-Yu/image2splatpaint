(function installTrainingPipelineShaderFactory(global) {
  function create({
    fixedPointExactGradientEnabled,
    inverseScaleOptimizationEnabled,
    optimizerStatsDeclaration,
    segmentedExactBackwardEnabled,
    protectedPrefix = 0,
  }) {
    protectedPrefix = Math.max(0, Math.floor(Number(protectedPrefix) || 0));
    // BEGIN BYTE-STABLE TRAINING SHADER DECLARATIONS
    const optimizerStatsUpdate = `adam[g] = opt;
  let previous = stats[g];
  let meanError = errorSum / max(observed, 1.0);
  let signedGradientSignal = length(gradCenter) * normalizer;
  let absoluteGradientSignal = length(gradCenterAbs) * normalizer;
  let densityGradientMode = u32(cfg(29u));
  let coherenceNorm = gradCenterNorm * normalizer;
  var gradientSignal = select(signedGradientSignal, absoluteGradientSignal, densityGradientMode == 1u);
  if (densityGradientMode == 2u && cfg(78u) > 0.5) { gradientSignal = coherenceNorm; }
  let meanContribution = influenceSum / max(observed, 1.0);
  stats[g] = vec4<f32>(previous.x + gradientSignal, max(previous.y, meanError), max(previous.z, meanContribution), previous.w + 1.0);
  let previousImportance = stats[capacity + g];
  let importanceAlpha = clamp(cfg(27u), 0.001, 1.0);
  let measuredImportance = vec3<f32>(observed, influenceSum, errorSum);
  let importanceW = select(previousImportance.w + 1.0, previousImportance.w + coherenceNorm, densityGradientMode == 2u);
  stats[capacity + g] = vec4<f32>(mix(previousImportance.xyz, measuredImportance, importanceAlpha), importanceW);`;
    const renderShader = `
struct Config { values: array<vec4<f32>, 32>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(6) var<storage, read> tileIndices: array<u32>;
@group(0) @binding(7) var<storage, read_write> pixelState: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> alphaState: array<AlphaState>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

fn accumulation_layer(packed: f32, layerCount: u32) -> u32 {
  let order = clamp(min(fract(packed), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 0.999999);
  return min(layerCount - 1u, u32(floor(order * f32(layerCount))));
}

${VIRTUAL_TILT_WGSL}
${RECTANGLE_TRAPEZOID_WGSL}
${ILLUSTRATIVE_OIL_WGSL}

fn gaussian_kernel(d: vec2<f32>, c: f32, s: f32, scale: vec2<f32>) -> f32 {
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  let q = dot(r / scale, r / scale);
  return exp(-0.5 * q);
}

fn training_kernel(
  d: vec2<f32>,
  c: f32,
  s: f32,
  scale: vec2<f32>,
  packedTag: f32,
  taperAmount: f32
) -> f32 {
  if (cfg(40u) < 0.5) {
    return gaussian_kernel(d, c, s, scale);
  }
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  if (cfg(40u) > 3.5) {
    let normalized = r / max(scale * ${LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT}, vec2<f32>(0.0001));
    return illustrative_oil_kernel_sample(
      normalized,
      scale.x >= scale.y,
      clamp(cfg(41u), 0.01, 0.49),
      illustrative_oil_family(scale, packedTag),
      taperAmount,
      cfg(94u) > 0.5,
      cfg(95u) > 0.5,
      cfg(101u),
      cfg(102u),
      cfg(125u),
      cfg(126u),
      cfg(103u),
      cfg(104u)
    ).kernel;
  }
  let widthRatios = rectangle_effective_width_ratios(
    cfg(96u),
    cfg(98u),
    packedTag,
    cfg(97u)
  );
  let areaCompensation = rectangle_area_compensation(widthRatios, cfg(97u));
  let normalized = r / max(
    scale * ${RECTANGLE_KERNEL_EXTENT} * areaCompensation,
    vec2<f32>(0.0001)
  );
  return rectangle_trapezoid_kernel_sample(
    normalized,
    cfg(41u),
    widthRatios.x,
    widthRatios.y,
    rectangle_flag_enabled(cfg(97u), ${RECTANGLE_FLAG_ASYMMETRIC_SOFTNESS}u),
    cfg(116u),
    cfg(117u),
    cfg(123u),
    cfg(124u)
  ).kernel;
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
var<workgroup> tileSharedLayer: array<f32, ${TILE_SIZE * TILE_SIZE}>;
var<workgroup> tileSharedTaper: array<f32, ${TILE_SIZE * TILE_SIZE}>;
var<workgroup> tileSharedPackedOrder: array<f32, ${TILE_SIZE * TILE_SIZE}>;
var<workgroup> tileSharedIndex: array<u32, ${TILE_SIZE * TILE_SIZE}>;

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
  let gridPoint = vec2<f32>(
    select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
    select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
  );
  let projectedPoint = training_output_point(gridPoint);
  let outputPoint = projectedPoint.xy;
  let bg = vec3<f32>(cfg(3u), cfg(4u), cfg(5u));
  if (cfg(19u) > 0.5 && tile_list_overflow()) {
    pixelState[pixel] = vec4<f32>(bg, -1.0);
    alphaState[pixel] = AlphaState(-1.0, 0u, 0.0, 0u);
    return;
  }
  let useTiles = cfg(19u) > 0.5;
  let tileCols = (width + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  let tilePixel = training_output_pixel(vec2<u32>(px, py), outputPoint, width, height);
  let tile = (tilePixel.y / ${TILE_SIZE}u) * tileCols + (tilePixel.x / ${TILE_SIZE}u);
  let tileCapacity = arrayLength(&tileIndices);
  let start = select(0u, min(tile_offset(tile), tileCapacity), useTiles);
  let end = select(u32(cfg(2u)), min(tile_offset(tile + 1u), tileCapacity), useTiles);
  let inversePoint = training_source_point(gridPoint, projectedPoint);
  if (inversePoint.z < 0.5) {
    pixelState[pixel] = vec4<f32>(bg, 0.0);
    alphaState[pixel] = AlphaState(0.0, start, 0.0, 0u);
    return;
  }
  let p = inversePoint.xy;
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  var rendered = vec3<f32>(0.0);
  var transmittance = 1.0;
  let contributionMomentsEnabled = cfg(47u) > 0.5 && cfg(40u) > 3.5;
  var contributionSquareSum = 0.0;
  let layerAware = cfg(84u) > 0.5;
  let accumulationLayerCount = clamp(u32(round(cfg(82u))), ${MIN_DISCRETE_LAYER_COUNT}u, ${MAX_DISCRETE_LAYER_COUNT}u);
  var activeLayer = accumulationLayerCount;
  var layerRendered = vec3<f32>(0.0);
  var layerTransmittance = 1.0;
  var cursor = start;
  var acceptedEnd = start;
  var contributorCount = 0u;
  loop {
    if (cursor >= end) { break; }
    var g = cursor;
    if (useTiles) { g = tileIndices[cursor]; }
    let sourceT = transform[g];
    let t = sourceT;
    if (t.w >= 0.5) {
      var center = xy[g].center;
      var c = cos(t.z);
      var s = sin(t.z);
      var baseScale = max(t.xy, vec2<f32>(0.0001));
      var effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
      var sampleScale = effective;
      var mip = select(
        sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001)),
        1.0,
        cfg(40u) > 0.5
      );
      let useEwa = cfg(26u) > 0.5;
      if (useEwa) {
        sampleScale = baseScale;
        mip = 1.0;
      }
      let layerZ = virtual_pass_layer_depth(t.w, xy[g].rawDepth);
      if (camera_covariance_3d_enabled()) {
        let projected = project_planar_gaussian(center, layerZ, t);
        center = projected.center;
        c = cos(projected.theta);
        s = sin(projected.theta);
        baseScale = projected.scale;
        effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
        sampleScale = select(effective, baseScale, useEwa);
        mip = select(
          sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001)),
          1.0,
          useEwa || cfg(40u) > 0.5
        );
      }
      let splatPoint = select(virtual_inverse_point_at_z(outputPoint, layerZ).xy, outputPoint, camera_covariance_3d_enabled());
      let d = splatPoint - center;
      var kernel = training_kernel(d, c, s, sampleScale, t.w, xy[g].rawDepth);
      if (useEwa) {
        let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
        let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
        let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
        let p00 = clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0));
        let p10 = clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0));
        let p01 = clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0));
        let p11 = clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0));
        let q00 = select(virtual_inverse_point_at_z(p00, layerZ).xy, p00, camera_covariance_3d_enabled());
        let q10 = select(virtual_inverse_point_at_z(p10, layerZ).xy, p10, camera_covariance_3d_enabled());
        let q01 = select(virtual_inverse_point_at_z(p01, layerZ).xy, p01, camera_covariance_3d_enabled());
        let q11 = select(virtual_inverse_point_at_z(p11, layerZ).xy, p11, camera_covariance_3d_enabled());
        kernel = 0.25 * (
          training_kernel(q00 - center, c, s, baseScale, t.w, xy[g].rawDepth) +
          training_kernel(q10 - center, c, s, baseScale, t.w, xy[g].rawDepth) +
          training_kernel(q01 - center, c, s, baseScale, t.w, xy[g].rawDepth) +
          training_kernel(q11 - center, c, s, baseScale, t.w, xy[g].rawDepth)
        );
      }
      if (kernel >= 0.0003354626) {
        let fixedOpaque = cfg(85u) > 0.5 && cfg(90u) < 0.5;
        let minimumOpacity = select(0.0, cfg(88u), cfg(85u) > 0.5 && cfg(90u) > 0.5);
        let effectiveOpacity = select(clamp(color[g].a, minimumOpacity, cfg(118u)) * mip, cfg(88u), fixedOpaque);
        let alphaLimit = select(0.99, cfg(88u), fixedOpaque);
        let alpha = clamp(kernel * effectiveOpacity, 0.0, alphaLimit);
        if (alpha >= 0.0039215686) {
          let surfaceRgb = color[g].rgb;
          if (contributionMomentsEnabled) {
            let contributionTransmittance = transmittance * select(1.0, layerTransmittance, layerAware);
            let contribution = contributionTransmittance * alpha;
            contributionSquareSum += contribution * contribution;
          }
          let layer = accumulation_layer(t.w, accumulationLayerCount);
          if (layerAware && activeLayer != accumulationLayerCount && layer != activeLayer) {
            rendered += transmittance * layerRendered;
            transmittance *= layerTransmittance;
            layerRendered = vec3<f32>(0.0);
            layerTransmittance = 1.0;
          }
          if (layerAware) {
            activeLayer = layer;
            layerRendered += layerTransmittance * alpha * surfaceRgb;
            layerTransmittance *= 1.0 - alpha;
          } else {
            rendered += transmittance * alpha * surfaceRgb;
            transmittance *= 1.0 - alpha;
          }
          acceptedEnd = cursor + 1u;
          contributorCount += 1u;
        }
      }
    }
    cursor += 1u;
    if (transmittance * select(1.0, layerTransmittance, layerAware) < 0.0001) { break; }
  }
  if (layerAware && activeLayer != accumulationLayerCount) {
    rendered += transmittance * layerRendered;
    transmittance *= layerTransmittance;
  }
  rendered += transmittance * bg;
  let compositeAlpha = 1.0 - transmittance;
  pixelState[pixel] = vec4<f32>(rendered, compositeAlpha);
  alphaState[pixel] = AlphaState(1.0 - transmittance, acceptedEnd, contributionSquareSum, contributorCount);
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
      alphaState[pixel] = AlphaState(-1.0, 0u, 0.0, 0u);
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
  let contributionMomentsEnabled = cfg(47u) > 0.5 && cfg(40u) > 3.5;
  var contributionSquareSum = 0.0;
  let layerAware = cfg(84u) > 0.5;
  let accumulationLayerCount = clamp(u32(round(cfg(82u))), ${MIN_DISCRETE_LAYER_COUNT}u, ${MAX_DISCRETE_LAYER_COUNT}u);
  var activeLayer = accumulationLayerCount;
  var layerRendered = vec3<f32>(0.0);
  var layerTransmittance = 1.0;
  var batchStart = start;
  var acceptedEnd = start;
  var contributorCount = 0u;
  loop {
    if (batchStart >= end) { break; }
    let batchCount = min(${TILE_SIZE * TILE_SIZE}u, end - batchStart);
    if (localIndex < batchCount) {
      let g = tileIndices[batchStart + localIndex];
      let sourceT = transform[g];
      let t = sourceT;
      var center = xy[g].center;
      var c = cos(t.z);
      var s = sin(t.z);
      var baseScale = max(t.xy, vec2<f32>(0.0001));
      var effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
      var sampleScale = effective;
      var mip = select(
        sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001)),
        1.0,
        cfg(40u) > 0.5
      );
      let isActive = t.w >= 0.5;
      if (useEwa) {
        sampleScale = baseScale;
        mip = 1.0;
      }
      let layerZ = virtual_pass_layer_depth(t.w, xy[g].rawDepth);
      if (camera_covariance_3d_enabled()) {
        let projected = project_planar_gaussian(center, layerZ, t);
        center = projected.center;
        c = cos(projected.theta);
        s = sin(projected.theta);
        baseScale = projected.scale;
        effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
        sampleScale = select(effective, baseScale, useEwa);
        mip = select(
          sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001)),
          1.0,
          useEwa || cfg(40u) > 0.5
        );
      }
      tileShared0[localIndex] = vec4<f32>(center, c, s);
      tileShared1[localIndex] = vec4<f32>(sampleScale, mip, select(0.0, 1.0, isActive));
      tileSharedColor[localIndex] = color[g];
      tileSharedLayer[localIndex] = layerZ;
      tileSharedTaper[localIndex] = xy[g].rawDepth;
      tileSharedPackedOrder[localIndex] = t.w;
      tileSharedIndex[localIndex] = g;
    }
    workgroupBarrier();
    if (inside) {
      for (var j = 0u; j < batchCount; j += 1u) {
        if (tileShared1[j].w > 0.5) {
          let center = tileShared0[j].xy;
          let c = tileShared0[j].z;
          let s = tileShared0[j].w;
          let sampleScale = tileShared1[j].xy;
          let layerZ = tileSharedLayer[j];
          let splatPoint = select(virtual_inverse_point_at_z(outputPoint, layerZ).xy, outputPoint, camera_covariance_3d_enabled());
          var kernel = training_kernel(splatPoint - center, c, s, sampleScale, tileSharedPackedOrder[j], tileSharedTaper[j]);
          if (useEwa) {
            let p00 = clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0));
            let p10 = clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0));
            let p01 = clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0));
            let p11 = clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0));
            let q00 = select(virtual_inverse_point_at_z(p00, layerZ).xy, p00, camera_covariance_3d_enabled());
            let q10 = select(virtual_inverse_point_at_z(p10, layerZ).xy, p10, camera_covariance_3d_enabled());
            let q01 = select(virtual_inverse_point_at_z(p01, layerZ).xy, p01, camera_covariance_3d_enabled());
            let q11 = select(virtual_inverse_point_at_z(p11, layerZ).xy, p11, camera_covariance_3d_enabled());
            kernel = 0.25 * (
              training_kernel(q00 - center, c, s, sampleScale, tileSharedPackedOrder[j], tileSharedTaper[j]) +
              training_kernel(q10 - center, c, s, sampleScale, tileSharedPackedOrder[j], tileSharedTaper[j]) +
              training_kernel(q01 - center, c, s, sampleScale, tileSharedPackedOrder[j], tileSharedTaper[j]) +
              training_kernel(q11 - center, c, s, sampleScale, tileSharedPackedOrder[j], tileSharedTaper[j])
            );
          }
          if (kernel >= 0.0003354626) {
            let rgba = tileSharedColor[j];
            let surfaceRgb = rgba.rgb;
            let fixedOpaque = cfg(85u) > 0.5 && cfg(90u) < 0.5;
            let minimumOpacity = select(0.0, cfg(88u), cfg(85u) > 0.5 && cfg(90u) > 0.5);
            let effectiveOpacity = select(clamp(rgba.a, minimumOpacity, cfg(118u)) * tileShared1[j].z, cfg(88u), fixedOpaque);
            let alphaLimit = select(0.99, cfg(88u), fixedOpaque);
            let alpha = clamp(kernel * effectiveOpacity, 0.0, alphaLimit);
            if (alpha >= 0.0039215686 && transmittance * select(1.0, layerTransmittance, layerAware) >= 0.0001) {
              if (contributionMomentsEnabled) {
                let contributionTransmittance = transmittance * select(1.0, layerTransmittance, layerAware);
                let contribution = contributionTransmittance * alpha;
                contributionSquareSum += contribution * contribution;
              }
              let layer = accumulation_layer(tileSharedPackedOrder[j], accumulationLayerCount);
              if (layerAware && activeLayer != accumulationLayerCount && layer != activeLayer) {
                rendered += transmittance * layerRendered;
                transmittance *= layerTransmittance;
                layerRendered = vec3<f32>(0.0);
                layerTransmittance = 1.0;
              }
              if (layerAware) {
                activeLayer = layer;
                layerRendered += layerTransmittance * alpha * surfaceRgb;
                layerTransmittance *= 1.0 - alpha;
              } else {
                rendered += transmittance * alpha * surfaceRgb;
                transmittance *= 1.0 - alpha;
              }
              acceptedEnd = batchStart + j + 1u;
              contributorCount += 1u;
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
    if (layerAware && activeLayer != accumulationLayerCount) {
      rendered += transmittance * layerRendered;
      transmittance *= layerTransmittance;
    }
    rendered = select(bg, rendered + transmittance * bg, inside);
    let compositeAlpha = 1.0 - transmittance;
    let storedAlpha = select(0.0, compositeAlpha, inside);
    pixelState[pixel] = vec4<f32>(rendered, storedAlpha);
    alphaState[pixel] = AlphaState(storedAlpha, select(start, acceptedEnd, inside), select(0.0, contributionSquareSum, inside), select(0u, contributorCount, inside));
  }
}`;

    const ssimShader = `
struct Config { values: array<vec4<f32>, 19>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ssimData: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(5) var<storage, read> alphaState: array<AlphaState>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn gaussian_weight(offset: i32) -> f32 {
  let weights = array<f32, 11>(
    0.00102838008448, 0.00759875813524, 0.0360007721284,
    0.109360689510, 0.213005537711, 0.266011724862,
    0.213005537711, 0.109360689510, 0.0360007721284,
    0.00759875813524, 0.00102838008448
  );
  return weights[u32(offset + 5)];
}
fn tile_prefix(width: u32, height: u32) -> u32 {
  return width * height * 4u;
}
fn alpha_tile_prefix(width: u32, height: u32) -> u32 {
  return tile_prefix(width, height) + width * height * 5u;
}
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
  return mix(mix(targetAlpha[p0.y * width + p0.x], targetAlpha[p0.y * width + p1.x], f.x), mix(targetAlpha[p1.y * width + p0.x], targetAlpha[p1.y * width + p1.x], f.x), f.y);
}
fn training_pair(px: u32, py: u32, width: u32, height: u32) -> array<vec4<f32>, 2> {
  let pixel = py * width + px;
  let gridPoint = vec2<f32>(select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u));
  let projectedPoint = training_output_point(gridPoint);
  let inversePoint = training_source_point(gridPoint, projectedPoint);
  let directIndex = pixel * 3u;
  let directTarget = vec3<f32>(targetRgb[directIndex], targetRgb[directIndex + 1u], targetRgb[directIndex + 2u]);
  let sampledTarget = select(target_rgb_at(inversePoint.xy, width, height), directTarget, source_domain_reprojection_enabled());
  let targetColor = select(vec3<f32>(0.0), sampledTarget, inversePoint.z > 0.5);
  let sampledAlpha = select(target_alpha_at(inversePoint.xy, width, height), targetAlpha[pixel], source_domain_reprojection_enabled());
  let targetOpacity = select(0.0, sampledAlpha, inversePoint.z > 0.5);
  let valid = training_sample_valid(inversePoint);
  let rendered = select(vec4<f32>(0.0), vec4<f32>(pixelState[pixel].rgb, alphaState[pixel].compositeAlpha), valid);
  let referenceSample = select(vec4<f32>(0.0), vec4<f32>(targetColor, targetOpacity), valid);
  return array<vec4<f32>, 2>(rendered, referenceSample);
}
@compute @workgroup_size(64)
fn ssim_horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let pixel = id.x;
  if (pixel >= width * height) { return; }
  let px = pixel % width;
  let py = pixel / width;
  var sx = vec4<f32>(0.0);
  var sy = vec4<f32>(0.0);
  var sx2 = vec4<f32>(0.0);
  var sy2 = vec4<f32>(0.0);
  var sxy = vec4<f32>(0.0);
  for (var offset = -5; offset <= 5; offset += 1) {
    let sampleX = i32(px) + offset;
    if (sampleX >= 0 && sampleX < i32(width)) {
      let pair = training_pair(u32(sampleX), py, width, height);
      let weight = gaussian_weight(offset);
      sx += weight * pair[0];
      sy += weight * pair[1];
      sx2 += weight * pair[0] * pair[0];
      sy2 += weight * pair[1] * pair[1];
      sxy += weight * pair[0] * pair[1];
    }
  }
  let base = tile_prefix(width, height) + pixel * 5u;
  ssimData[base] = sx;
  ssimData[base + 1u] = sy;
  ssimData[base + 2u] = sx2;
  ssimData[base + 3u] = sy2;
  ssimData[base + 4u] = sxy;
}
@compute @workgroup_size(64)
fn ssim_vertical(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let pixel = id.x;
  if (pixel >= width * height) { return; }
  let px = pixel % width;
  let py = pixel / width;
  var sx = vec4<f32>(0.0);
  var sy = vec4<f32>(0.0);
  var sx2 = vec4<f32>(0.0);
  var sy2 = vec4<f32>(0.0);
  var sxy = vec4<f32>(0.0);
  let prefix = tile_prefix(width, height);
  for (var offset = -5; offset <= 5; offset += 1) {
    let sampleY = i32(py) + offset;
    if (sampleY >= 0 && sampleY < i32(height)) {
      let base = prefix + (u32(sampleY) * width + px) * 5u;
      let weight = gaussian_weight(offset);
      sx += weight * ssimData[base];
      sy += weight * ssimData[base + 1u];
      sx2 += weight * ssimData[base + 2u];
      sy2 += weight * ssimData[base + 3u];
      sxy += weight * ssimData[base + 4u];
    }
  }
  let vx = max(vec4<f32>(0.0), sx2 - sx * sx);
  let vy = max(vec4<f32>(0.0), sy2 - sy * sy);
  let cov = sxy - sx * sy;
  let a = 2.0 * sx * sy + vec4<f32>(0.0001);
  let b = 2.0 * cov + vec4<f32>(0.0009);
  let c = sx * sx + sy * sy + vec4<f32>(0.0001);
  let d = vx + vy + vec4<f32>(0.0009);
  let ssim = a * b / max(vec4<f32>(0.00000001), c * d);
  // Express dSSIM/dx(sample) as a Gaussian convolution of three terms:
  // K0(center) + target(sample) * Ky(center) + render(sample) * Kx(center).
  // Two additional separable passes below apply the convolution transpose.
  let k0 = ssim * (2.0 * sy / a - 2.0 * sy / b - 2.0 * sx / c + 2.0 * sx / d);
  let ky = ssim * (2.0 / b);
  let kx = ssim * (-2.0 / d);
  let out = pixel * 4u;
  ssimData[out] = k0;
  ssimData[out + 1u] = ky;
  ssimData[out + 2u] = kx;
  ssimData[out + 3u] = ssim;
}
@compute @workgroup_size(64)
fn ssim_backward_horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let pixel = id.x;
  if (pixel >= width * height) { return; }
  let px = pixel % width;
  let py = pixel / width;
  var k0 = vec4<f32>(0.0);
  var ky = vec4<f32>(0.0);
  var kx = vec4<f32>(0.0);
  for (var offset = -5; offset <= 5; offset += 1) {
    let centerX = i32(px) + offset;
    if (centerX >= 0 && centerX < i32(width)) {
      let center = (py * width + u32(centerX)) * 4u;
      let weight = gaussian_weight(offset);
      k0 += weight * ssimData[center];
      ky += weight * ssimData[center + 1u];
      kx += weight * ssimData[center + 2u];
    }
  }
  let base = tile_prefix(width, height) + pixel * 5u;
  ssimData[base] = k0;
  ssimData[base + 1u] = ky;
  ssimData[base + 2u] = kx;
}
@compute @workgroup_size(64)
fn ssim_backward_vertical(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let pixel = id.x;
  if (pixel >= width * height) { return; }
  let px = pixel % width;
  let py = pixel / width;
  var k0 = vec4<f32>(0.0);
  var ky = vec4<f32>(0.0);
  var kx = vec4<f32>(0.0);
  let prefix = tile_prefix(width, height);
  for (var offset = -5; offset <= 5; offset += 1) {
    let centerY = i32(py) + offset;
    if (centerY >= 0 && centerY < i32(height)) {
      let base = prefix + (u32(centerY) * width + px) * 5u;
      let weight = gaussian_weight(offset);
      k0 += weight * ssimData[base];
      ky += weight * ssimData[base + 1u];
      kx += weight * ssimData[base + 2u];
    }
  }
  let pair = training_pair(px, py, width, height);
  ssimData[pixel * 4u] = k0 + pair[1] * ky + pair[0] * kx;
}
var<workgroup> alphaX: array<f32, 64>;
var<workgroup> alphaY: array<f32, 64>;
var<workgroup> alphaX2: array<f32, 64>;
var<workgroup> alphaY2: array<f32, 64>;
var<workgroup> alphaXY: array<f32, 64>;
var<workgroup> alphaCount: array<f32, 64>;
@compute @workgroup_size(64)
fn alpha_ssim_tiles(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let tileCols = (width + 7u) / 8u;
  let tileIndex = wid.y * workgroups.x + wid.x;
  if (tileIndex >= tileCols * ((height + 7u) / 8u)) { return; }
  let px = (tileIndex % tileCols) * 8u + lid.x % 8u;
  let py = (tileIndex / tileCols) * 8u + lid.x / 8u;
  var x = 0.0;
  var y = 0.0;
  var valid = 0.0;
  if (px < width && py < height) {
    let pair = training_pair(px, py, width, height);
    x = pair[0].a;
    y = pair[1].a;
    valid = 1.0;
  }
  alphaX[lid.x] = x;
  alphaY[lid.x] = y;
  alphaX2[lid.x] = x * x;
  alphaY2[lid.x] = y * y;
  alphaXY[lid.x] = x * y;
  alphaCount[lid.x] = valid;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      alphaX[lid.x] += alphaX[lid.x + stride];
      alphaY[lid.x] += alphaY[lid.x + stride];
      alphaX2[lid.x] += alphaX2[lid.x + stride];
      alphaY2[lid.x] += alphaY2[lid.x + stride];
      alphaXY[lid.x] += alphaXY[lid.x + stride];
      alphaCount[lid.x] += alphaCount[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let n = max(alphaCount[0], 1.0);
    let mux = alphaX[0] / n;
    let muy = alphaY[0] / n;
    let vx = max(0.0, alphaX2[0] / n - mux * mux);
    let vy = max(0.0, alphaY2[0] / n - muy * muy);
    let cov = alphaXY[0] / n - mux * muy;
    let a = 2.0 * mux * muy + 0.0001;
    let b = 2.0 * cov + 0.0009;
    let c = mux * mux + muy * muy + 0.0001;
    let d = vx + vy + 0.0009;
    let ssim = a * b / max(0.00000001, c * d);
    let out = alpha_tile_prefix(width, height) + tileIndex * 2u;
    ssimData[out] = vec4<f32>(mux, muy, vx, vy);
    ssimData[out + 1u] = vec4<f32>(cov, ssim, n, 0.0);
  }
}`;

    const lossGradientShader = [
      "struct Config { values: array<vec4<f32>, 32>, };",
      "@group(0) @binding(0) var<uniform> config: Config;",
      "@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;",
      "@group(0) @binding(2) var<storage, read> targetAlpha: array<f32>;",
      "@group(0) @binding(3) var<storage, read> pixelState: array<vec4<f32>>;",
      "@group(0) @binding(4) var<storage, read> ssimTiles: array<vec4<f32>>;",
      "@group(0) @binding(5) var<storage, read_write> lossGradient: array<vec4<f32>>;",
      "fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }",
      VIRTUAL_TILT_WGSL,
      MONOCHROME_LAB_L_WGSL,
      "fn safe_signed(v: f32) -> f32 {",
      "  if (abs(v) >= 0.0000001) { return v; }",
      "  return select(-0.0000001, 0.0000001, v >= 0.0);",
      "}",
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
      "fn rendered_luma(px: u32, py: u32, width: u32) -> f32 {",
      "  return dot(pixelState[py * width + px].rgb, vec3<f32>(1.0 / 3.0));",
      "}",
      "fn target_luma(px: u32, py: u32, width: u32) -> f32 {",
      "  let height = u32(cfg(1u));",
      "  let gridPoint = vec2<f32>(select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u));",
      "  let projectedPoint = training_output_point(gridPoint);",
      "  let inversePoint = training_source_point(gridPoint, projectedPoint);",
      "  let directIndex = (py * width + px) * 3u;",
      "  let directTarget = vec3<f32>(targetRgb[directIndex], targetRgb[directIndex + 1u], targetRgb[directIndex + 2u]);",
      "  let targetColor = select(target_color_at(inversePoint.xy, width, height), directTarget, source_domain_reprojection_enabled());",
      "  let sampled = dot(targetColor, vec3<f32>(1.0 / 3.0));",
      "  return select(select(sampled, 0.0, !source_domain_reprojection_enabled() && cfg(66u) > 0.5), sampled, inversePoint.z > 0.5);",
      "}",
      "@compute @workgroup_size(64)",
      "fn loss_gradient(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) {",
      "  let width = u32(cfg(0u));",
      "  let height = u32(cfg(1u));",
      "  let pixel = id.x + id.y * workgroups.x * 64u;",
      "  if (pixel >= width * height) { return; }",
      "  let px = pixel % width;",
      "  let py = pixel / width;",
      "  let gridPoint = vec2<f32>(select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u));",
      "  let projectedPoint = training_output_point(gridPoint);",
      "  let inversePoint = training_source_point(gridPoint, projectedPoint);",
      "  if (!training_sample_valid(inversePoint)) {",
      "    lossGradient[pixel * 3u] = vec4<f32>(0.0);",
      "    lossGradient[pixel * 3u + 1u] = vec4<f32>(0.0);",
      "    lossGradient[pixel * 3u + 2u] = vec4<f32>(0.0);",
      "    return;",
      "  }",
      "  let renderedState = pixelState[pixel];",
      "  let directIndex = pixel * 3u;",
      "  let directTarget = vec3<f32>(targetRgb[directIndex], targetRgb[directIndex + 1u], targetRgb[directIndex + 2u]);",
      "  let sampledTarget = select(target_color_at(inversePoint.xy, width, height), directTarget, source_domain_reprojection_enabled());",
      "  let targetColor = select(vec3<f32>(0.0), sampledTarget, inversePoint.z > 0.5);",
      "  let residual = renderedState.rgb - targetColor;",
      "  let monochromeUnderpainting = cfg(93u) > 0.5;",
      "  let underpaintingActive = monochromeUnderpainting && cfg(8u) < cfg(100u);",
      "  var dColor = sign(residual) * ((1.0 - 0.2) / 3.0);",
      "  if (underpaintingActive) {",
      "    dColor = normalized_lab_l_only_gray_gradient_srgb(renderedState.rgb, targetColor);",
      "  }",
      "  let dSsim = ssimTiles[pixel * 4u];",
      "  let x = dot(renderedState.rgb, vec3<f32>(1.0 / 3.0));",
      "  let y = dot(targetColor, vec3<f32>(1.0 / 3.0));",
      "  if (!underpaintingActive) { dColor += -0.2 * dSsim.rgb / 3.0; }",
      "  if (!underpaintingActive && cfg(15u) > 0.5) {",
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
      "  let alphaX = renderedState.a;",
      "  let sampledAlpha = select(target_alpha_at(inversePoint.xy, width, height), targetAlpha[pixel], source_domain_reprojection_enabled());",
      "  let alphaY = select(0.0, sampledAlpha, inversePoint.z > 0.5);",
      "  let alphaTileCols = (width + 7u) / 8u;",
      "  let alphaTile = (py / 8u) * alphaTileCols + (px / 8u);",
      "  let alphaBase = width * height * 9u + alphaTile * 2u;",
      "  let alphaMoments = ssimTiles[alphaBase];",
      "  let alphaExtra = ssimTiles[alphaBase + 1u];",
      "  let alphaMux = alphaMoments.x;",
      "  let alphaMuy = alphaMoments.y;",
      "  let alphaVx = alphaMoments.z;",
      "  let alphaVy = alphaMoments.w;",
      "  let alphaCov = alphaExtra.x;",
      "  let alphaSsim = alphaExtra.y;",
      "  let alphaN = max(alphaExtra.z, 1.0);",
      "  let alphaA = safe_signed(2.0 * alphaMux * alphaMuy + 0.0001);",
      "  let alphaB = safe_signed(2.0 * alphaCov + 0.0009);",
      "  let alphaC = safe_signed(alphaMux * alphaMux + alphaMuy * alphaMuy + 0.0001);",
      "  let alphaD = safe_signed(alphaVx + alphaVy + 0.0009);",
      "  let dAlphaSsim = alphaSsim * ((2.0 * alphaMuy / alphaN) / alphaA + (2.0 * (alphaY - alphaMuy) / alphaN) / alphaB - (2.0 * alphaMux / alphaN) / alphaC - (2.0 * (alphaX - alphaMux) / alphaN) / alphaD);",
      "  var dAlpha = cfg(46u) * ((1.0 - 0.2) * sign(alphaX - alphaY) - 0.5 * 0.2 * dAlphaSsim);",
      "  var residualMagnitude = (abs(residual.r) + abs(residual.g) + abs(residual.b)) / 3.0;",
      "  if (underpaintingActive) {",
      "    residualMagnitude = abs(srgb_to_normalized_lab(renderedState.rgb).x - srgb_to_normalized_lab(targetColor).x);",
      "  }",
      "  if (cfg(21u) > 0.5 && residualMagnitude > 0.02) {",
      "    dAlpha += -2.0 * cfg(23u) * max(0.0, cfg(22u) - renderedState.a);",
      "  }",
      "  let viewWeight = select(1.0, clamp(cfg(61u), 0.0, 1.0), virtual_tilt_enabled()) * source_domain_area_weight(gridPoint);",
      "  dColor *= viewWeight;",
      "  dAlpha *= viewWeight;",
      "  lossGradient[pixel * 3u] = vec4<f32>(dColor, dAlpha);",
      "  lossGradient[pixel * 3u + 1u] = vec4<f32>(targetColor, residualMagnitude);",
      "  lossGradient[pixel * 3u + 2u] = vec4<f32>(0.0);",
      "}",
    ].join("\n");

    const subgroupCounterReductionLines = [
        "var<workgroup> subgroupCounter: atomic<u32>;",
        "fn add_subtile_gradient(localIndex: u32, subgroupSize: u32, subgroupInvocation: u32, recordIndex: u32, g: u32, geom: vec4<f32>, appearance: vec4<f32>, misc: vec4<f32>, density: vec4<f32>, depth: f32) {",
        "  if (localIndex == 0u) { atomicStore(&subgroupCounter, 0u); }",
        "  workgroupBarrier();",
        "  let subgroupGeom = subgroupAdd(geom);",
        "  let subgroupAppearance = subgroupAdd(appearance);",
        "  let subgroupMisc = subgroupAdd(misc);",
        "  let subgroupDensity = subgroupAdd(density);",
        "  let subgroupDepth = subgroupAdd(depth);",
        "  var subgroupSlot = 0u;",
        "  if (subgroupElect()) {",
        "    subgroupSlot = atomicAdd(&subgroupCounter, 1u);",
        "    reduceGeom[subgroupSlot] = subgroupGeom;",
        "    reduceAppearance[subgroupSlot] = subgroupAppearance;",
        "    reduceMisc[subgroupSlot] = subgroupMisc;",
        "    reduceDensity[subgroupSlot] = subgroupDensity;",
        "    reduceDepth[subgroupSlot] = subgroupDepth;",
        "  }",
        "  subgroupSlot = subgroupBroadcastFirst(subgroupSlot);",
        "  workgroupBarrier();",
        "  let partialCount = atomicLoad(&subgroupCounter);",
        "  if (subgroupSize >= 8u) {",
        "    let partialIndex = subgroupInvocation;",
        "    let partialGeom = select(vec4<f32>(0.0), reduceGeom[partialIndex], partialIndex < partialCount);",
        "    let partialAppearance = select(vec4<f32>(0.0), reduceAppearance[partialIndex], partialIndex < partialCount);",
        "    let partialMisc = select(vec4<f32>(0.0), reduceMisc[partialIndex], partialIndex < partialCount);",
        "    let partialDensity = select(vec4<f32>(0.0), reduceDensity[partialIndex], partialIndex < partialCount);",
        "    let partialDepth = select(0.0, reduceDepth[partialIndex], partialIndex < partialCount);",
        "    let totalGeom = subgroupAdd(partialGeom);",
        "    let totalAppearance = subgroupAdd(partialAppearance);",
        "    let totalMisc = subgroupAdd(partialMisc);",
        "    let totalDensity = subgroupAdd(partialDensity);",
        "    let totalDepth = subgroupAdd(partialDepth);",
        "    let electedForFinal = subgroupElect();",
        "    if (subgroupSlot == 0u && electedForFinal) {",
        "      store_reduced_gradient(recordIndex, g, totalGeom, totalAppearance, totalMisc, totalDensity, totalDepth);",
        "    }",
        "    workgroupBarrier();",
        "    return;",
        "  }",
        "  reduceGeom[localIndex] = select(vec4<f32>(0.0), reduceGeom[localIndex], localIndex < partialCount);",
        "  reduceAppearance[localIndex] = select(vec4<f32>(0.0), reduceAppearance[localIndex], localIndex < partialCount);",
        "  reduceMisc[localIndex] = select(vec4<f32>(0.0), reduceMisc[localIndex], localIndex < partialCount);",
        "  reduceDensity[localIndex] = select(vec4<f32>(0.0), reduceDensity[localIndex], localIndex < partialCount);",
        "  reduceDepth[localIndex] = select(0.0, reduceDepth[localIndex], localIndex < partialCount);",
        "  workgroupBarrier();",
        "  for (var stride = 32u; stride > 0u; stride /= 2u) {",
        "    if (localIndex < stride) {",
        "      reduceGeom[localIndex] += reduceGeom[localIndex + stride];",
        "      reduceAppearance[localIndex] += reduceAppearance[localIndex + stride];",
        "      reduceMisc[localIndex] += reduceMisc[localIndex + stride];",
        "      reduceDensity[localIndex] += reduceDensity[localIndex + stride];",
        "      reduceDepth[localIndex] += reduceDepth[localIndex + stride];",
        "    }",
        "    workgroupBarrier();",
        "  }",
        "  if (localIndex == 0u) { store_reduced_gradient(recordIndex, g, reduceGeom[0], reduceAppearance[0], reduceMisc[0], reduceDensity[0], reduceDepth[0]); }",
        "  workgroupBarrier();",
        "}",
      ];
    const exactBackwardReductionLines = this.subgroupExactBackwardEnabled
      ? subgroupCounterReductionLines
      : [
        "fn add_subtile_gradient(localIndex: u32, recordIndex: u32, g: u32, geom: vec4<f32>, appearance: vec4<f32>, misc: vec4<f32>, density: vec4<f32>, depth: f32) {",
        "  reduceGeom[localIndex] = geom;",
        "  reduceAppearance[localIndex] = appearance;",
        "  reduceMisc[localIndex] = misc;",
        "  reduceDensity[localIndex] = density;",
        "  reduceDepth[localIndex] = depth;",
        "  workgroupBarrier();",
        "  var stride = 32u;",
        "  loop {",
        "    if (localIndex < stride) {",
        "      reduceGeom[localIndex] += reduceGeom[localIndex + stride];",
        "      reduceAppearance[localIndex] += reduceAppearance[localIndex + stride];",
        "      reduceMisc[localIndex] += reduceMisc[localIndex + stride];",
        "      reduceDensity[localIndex] += reduceDensity[localIndex + stride];",
        "      reduceDepth[localIndex] += reduceDepth[localIndex + stride];",
        "    }",
        "    workgroupBarrier();",
        "    if (stride == 1u) { break; }",
        "    stride /= 2u;",
        "  }",
        "  if (localIndex == 0u) { store_reduced_gradient(recordIndex, g, reduceGeom[0], reduceAppearance[0], reduceMisc[0], reduceDensity[0], reduceDepth[0]); }",
        "  workgroupBarrier();",
        "}",
      ];
    const exactBackwardShader = [
      this.subgroupExactBackwardEnabled ? "enable subgroups;" : "",
      "struct Config { values: array<vec4<f32>, 32>, };",
      "struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };",
      "struct KernelSample { kernel: f32, dCenter: vec2<f32>, dLogScale: vec2<f32>, dTheta: f32, dTaper: f32, };",
      "struct KernelEvaluation { rawKernel: f32, weightedKernel: f32, dCenter: vec2<f32>, dLogScale: vec2<f32>, dTheta: f32, dDepth: f32, };",
      "@group(0) @binding(0) var<uniform> config: Config;",
      "struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };",
      "@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;",
      "@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;",
      "@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;",
      "@group(0) @binding(4) var<storage, read> tileOffsets: array<u32>;",
      "@group(0) @binding(5) var<storage, read> tileIndices: array<u32>;",
      "@group(0) @binding(6) var<storage, read> lossGradient: array<vec4<f32>>;",
      fixedPointExactGradientEnabled
        ? "@group(0) @binding(7) var<storage, read_write> exactGradient: array<atomic<i32>>;"
        : "@group(0) @binding(7) var<storage, read_write> exactGradient: array<atomic<u32>>;",
      "@group(0) @binding(8) var<storage, read> alphaState: array<AlphaState>;",
      ...(segmentedExactBackwardEnabled
        ? ["@group(0) @binding(9) var<storage, read_write> partialGradient: array<f32>;"]
        : fixedPointExactGradientEnabled
          ? ["@group(0) @binding(9) var<storage, read_write> fixedPointControl: array<atomic<u32>>;"]
          : []),
      "fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }",
      VIRTUAL_TILT_WGSL,
      RECTANGLE_TRAPEZOID_WGSL,
      ILLUSTRATIVE_OIL_WGSL,
      "fn tile_offset(index: u32) -> u32 { return tileOffsets[index] & 0x7fffffffu; }",
      "fn tile_list_overflow() -> bool { return cfg(19u) > 0.5 && (tileOffsets[arrayLength(&tileOffsets) - 1u] & 0x80000000u) != 0u; }",
      "fn kernel_sample(d: vec2<f32>, c: f32, s: f32, baseScale: vec2<f32>, sampleScale: vec2<f32>, includeMipGradient: bool, packedTag: f32, taperAmount: f32) -> KernelSample {",
      "  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);",
      "  if (cfg(40u) > 3.5) {",
      `    let extent = ${LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT};`,
      "    let extentScale = max(sampleScale * extent, vec2<f32>(0.0001));",
      "    let normalized = r / extentScale;",
      "    let sample = illustrative_oil_kernel_sample(",
      "      normalized,",
      "      sampleScale.x >= sampleScale.y,",
      "      clamp(cfg(41u), 0.01, 0.49),",
      "      illustrative_oil_family(baseScale, packedTag),",
      "      taperAmount,",
      "      cfg(94u) > 0.5,",
      "      cfg(95u) > 0.5,",
      "      cfg(101u),",
      "      cfg(102u),",
      "      cfg(125u),",
      "      cfg(126u),",
      "      cfg(103u),",
      "      cfg(104u)",
      "    );",
      "    if (sample.kernel <= 0.00000001) { return KernelSample(0.0, vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0); }",
      "    let dLogDNormalized = sample.gradient / max(sample.kernel, 0.000001);",
      "    let dLogDr = dLogDNormalized / extentScale;",
      "    let dCenter = vec2<f32>(-c * dLogDr.x + s * dLogDr.y, -s * dLogDr.x - c * dLogDr.y);",
      "    let baseRatio = (baseScale * baseScale) / max(sampleScale * sampleScale, vec2<f32>(0.00000001));",
      "    var dLogScale = -dLogDNormalized * normalized;",
      "    dLogScale *= baseRatio;",
      "    let dTheta = dLogDr.x * r.y - dLogDr.y * r.x;",
      "    return KernelSample(sample.kernel, dCenter, dLogScale, dTheta, sample.taperGradient / max(sample.kernel, 0.000001));",
      "  }",
      "  if (cfg(40u) > 0.5) {",
      `    let extent = ${RECTANGLE_KERNEL_EXTENT};`,
      "    let feather = clamp(cfg(41u), 0.01, 0.49);",
      "    let widthRatios = rectangle_effective_width_ratios(cfg(96u), cfg(98u), packedTag, cfg(97u));",
      "    let areaCompensation = rectangle_area_compensation(widthRatios, cfg(97u));",
      "    let extentScale = max(sampleScale * extent * areaCompensation, vec2<f32>(0.0001));",
      "    let normalized = r / extentScale;",
      `    let asymmetricSoftness = rectangle_flag_enabled(cfg(97u), ${RECTANGLE_FLAG_ASYMMETRIC_SOFTNESS}u);`,
      "    let sample = rectangle_trapezoid_kernel_sample(normalized, feather, widthRatios.x, widthRatios.y, asymmetricSoftness, cfg(116u), cfg(117u), cfg(123u), cfg(124u));",
      "    if (sample.kernel <= 0.00000001) { return KernelSample(0.0, vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0); }",
      "    let dLogDNormalized = sample.gradient / max(sample.kernel, 0.000001);",
      "    let dLogDr = dLogDNormalized / extentScale;",
      "    let dCenter = vec2<f32>(-c * dLogDr.x + s * dLogDr.y, -s * dLogDr.x - c * dLogDr.y);",
      "    let baseRatio = (baseScale * baseScale) / max(sampleScale * sampleScale, vec2<f32>(0.00000001));",
      "    let dLogScale = -dLogDNormalized * normalized * baseRatio;",
      "    let dTheta = dLogDr.x * r.y - dLogDr.y * r.x;",
      "    return KernelSample(sample.kernel, dCenter, dLogScale, dTheta, 0.0);",
      "  }",
      "  let invS2 = 1.0 / (sampleScale * sampleScale);",
      "  let q = dot(r * r, invS2);",
      "  if (q > 16.0) { return KernelSample(0.0, vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0); }",
      "  let kernel = exp(-0.5 * q);",
      "  let dCenter = vec2<f32>(r.x * c * invS2.x - r.y * s * invS2.y, r.x * s * invS2.x + r.y * c * invS2.y);",
      "  var dLogScale = vec2<f32>(r.x * r.x * baseScale.x * baseScale.x / pow(sampleScale.x, 4.0), r.y * r.y * baseScale.y * baseScale.y / pow(sampleScale.y, 4.0));",
      "  if (includeMipGradient) {",
      "    let ratio = (baseScale * baseScale) / (sampleScale * sampleScale);",
      "    dLogScale += 0.5 * (vec2<f32>(1.0) - ratio);",
      "  }",
      "  let dTheta = -r.x * r.y * (invS2.x - invS2.y);",
      "  return KernelSample(kernel, dCenter, dLogScale, dTheta, 0.0);",
      "}",
      "fn evaluate_kernel(g: u32, p: vec2<f32>, outputPoint: vec2<f32>, sourceCenter: vec2<f32>, rawDepth: f32, sourceT: vec4<f32>, width: u32, height: u32) -> KernelEvaluation {",
      "  let center = sourceCenter;",
      "  let t = sourceT;",
      "  let baseScale = max(t.xy, vec2<f32>(0.0001));",
      "  let c = cos(t.z);",
      "  let s = sin(t.z);",
      "  let pixelSigma = 0.35 * 2.0 / max(cfg(0u), cfg(1u));",
      "  let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));",
      "  let useEwa = cfg(26u) > 0.5;",
      "  let layerZ = virtual_pass_layer_depth(t.w, rawDepth);",
      "  if (camera_covariance_3d_enabled()) {",
      "    let projected = project_planar_gaussian(center, layerZ, t);",
      "    let projectedScale = max(projected.scale, vec2<f32>(0.0001));",
      "    let projectedC = cos(projected.theta);",
      "    let projectedS = sin(projected.theta);",
      "    let projectedEffective = sqrt(projectedScale * projectedScale + vec2<f32>(pixelSigma * pixelSigma));",
      "    var rawKernel = 0.0;",
      "    var weightedKernel = 0.0;",
      "    var screenCenterDerivative = vec2<f32>(0.0);",
      "    var projectedScaleDerivative = vec2<f32>(0.0);",
      "    var projectedThetaDerivative = 0.0;",
      "    var taperDerivative = 0.0;",
      "    if (!useEwa) {",
      "      let sample = kernel_sample(outputPoint - projected.center, projectedC, projectedS, projectedScale, projectedEffective, true, t.w, rawDepth);",
      "      let mip = select(sqrt((projectedScale.x * projectedScale.y) / max(projectedEffective.x * projectedEffective.y, 0.00000001)), 1.0, cfg(40u) > 0.5);",
      "      rawKernel = sample.kernel;",
      "      weightedKernel = sample.kernel * mip;",
      "      screenCenterDerivative = sample.kernel * mip * sample.dCenter;",
      "      projectedScaleDerivative = sample.kernel * mip * sample.dLogScale;",
      "      projectedThetaDerivative = sample.kernel * mip * sample.dTheta;",
      "      taperDerivative = sample.kernel * mip * sample.dTaper;",
      "    } else {",
      "      let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);",
      "      let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);",
      "      let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);",
      "      let sample0 = kernel_sample(clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - projected.center, projectedC, projectedS, projectedScale, projectedScale, false, t.w, rawDepth);",
      "      let sample1 = kernel_sample(clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - projected.center, projectedC, projectedS, projectedScale, projectedScale, false, t.w, rawDepth);",
      "      let sample2 = kernel_sample(clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - projected.center, projectedC, projectedS, projectedScale, projectedScale, false, t.w, rawDepth);",
      "      let sample3 = kernel_sample(clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - projected.center, projectedC, projectedS, projectedScale, projectedScale, false, t.w, rawDepth);",
      "      rawKernel = 0.25 * (sample0.kernel + sample1.kernel + sample2.kernel + sample3.kernel);",
      "      weightedKernel = rawKernel;",
      "      screenCenterDerivative = 0.25 * (sample0.kernel * sample0.dCenter + sample1.kernel * sample1.dCenter + sample2.kernel * sample2.dCenter + sample3.kernel * sample3.dCenter);",
      "      projectedScaleDerivative = 0.25 * (sample0.kernel * sample0.dLogScale + sample1.kernel * sample1.dLogScale + sample2.kernel * sample2.dLogScale + sample3.kernel * sample3.dLogScale);",
      "      projectedThetaDerivative = 0.25 * (sample0.kernel * sample0.dTheta + sample1.kernel * sample1.dTheta + sample2.kernel * sample2.dTheta + sample3.kernel * sample3.dTheta);",
      "      taperDerivative = 0.25 * (sample0.kernel * sample0.dTaper + sample1.kernel * sample1.dTaper + sample2.kernel * sample2.dTaper + sample3.kernel * sample3.dTaper);",
      "    }",
      "    let sourceGradient = source_projection_gradient(center, layerZ, t, screenCenterDerivative, projectedScaleDerivative, projectedThetaDerivative);",
      "    let depthDerivative = select(select(0.0, taperDerivative, cfg(95u) > 0.5), sourceGradient.layerZ, cfg(67u) > 0.5);",
      "    return KernelEvaluation(rawKernel, weightedKernel, sourceGradient.center, sourceGradient.logScale, sourceGradient.theta, depthDerivative);",
      "  }",
      "  let splatPoint = virtual_inverse_point_at_z(outputPoint, layerZ).xy;",
      "  var inverseDepthDerivative = vec2<f32>(0.0);",
      "  if (cfg(67u) > 0.5 && virtual_tilt_enabled()) {",
      "    let epsilon = max(0.00005, cfg(68u) * 0.05);",
      "    let before = virtual_inverse_point_at_z(outputPoint, layerZ - epsilon).xy;",
      "    let after = virtual_inverse_point_at_z(outputPoint, layerZ + epsilon).xy;",
      "    inverseDepthDerivative = (after - before) / (2.0 * epsilon);",
      "  }",
      "  if (!useEwa) {",
      "    let sample = kernel_sample(splatPoint - center, c, s, baseScale, effective, true, t.w, rawDepth);",
      "    let mip = select(sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001)), 1.0, cfg(40u) > 0.5);",
      "    let weightedCenter = sample.kernel * mip * sample.dCenter;",
      "    let auxiliaryDerivative = select(select(0.0, sample.kernel * mip * sample.dTaper, cfg(95u) > 0.5), -dot(weightedCenter, inverseDepthDerivative), cfg(67u) > 0.5);",
      "    return KernelEvaluation(sample.kernel, sample.kernel * mip, weightedCenter, sample.kernel * mip * sample.dLogScale, sample.kernel * mip * sample.dTheta, auxiliaryDerivative);",
      "  }",
      "  let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);",
      "  let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);",
      "  let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);",
      "  let q0 = virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)), layerZ).xy;",
      "  let q1 = virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)), layerZ).xy;",
      "  let q2 = virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)), layerZ).xy;",
      "  let q3 = virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)), layerZ).xy;",
      "  let sample0 = kernel_sample(q0 - center, c, s, baseScale, baseScale, false, t.w, rawDepth);",
      "  let sample1 = kernel_sample(q1 - center, c, s, baseScale, baseScale, false, t.w, rawDepth);",
      "  let sample2 = kernel_sample(q2 - center, c, s, baseScale, baseScale, false, t.w, rawDepth);",
      "  let sample3 = kernel_sample(q3 - center, c, s, baseScale, baseScale, false, t.w, rawDepth);",
      "  let rawKernel = 0.25 * (sample0.kernel + sample1.kernel + sample2.kernel + sample3.kernel);",
      "  let dCenter = 0.25 * (sample0.kernel * sample0.dCenter + sample1.kernel * sample1.dCenter + sample2.kernel * sample2.dCenter + sample3.kernel * sample3.dCenter);",
      "  let dLogScale = 0.25 * (sample0.kernel * sample0.dLogScale + sample1.kernel * sample1.dLogScale + sample2.kernel * sample2.dLogScale + sample3.kernel * sample3.dLogScale);",
      "  let dTheta = 0.25 * (sample0.kernel * sample0.dTheta + sample1.kernel * sample1.dTheta + sample2.kernel * sample2.dTheta + sample3.kernel * sample3.dTheta);",
      "  let dTaper = 0.25 * (sample0.kernel * sample0.dTaper + sample1.kernel * sample1.dTaper + sample2.kernel * sample2.dTaper + sample3.kernel * sample3.dTaper);",
      "  let auxiliaryDerivative = select(select(0.0, dTaper, cfg(95u) > 0.5), -dot(dCenter, inverseDepthDerivative), cfg(67u) > 0.5);",
      "  return KernelEvaluation(rawKernel, rawKernel, dCenter, dLogScale, dTheta, auxiliaryDerivative);",
      "}",
      ...(fixedPointExactGradientEnabled
        ? [
            `const FIXED_GRADIENT_SCALE = ${FIXED_POINT_EXACT_GRADIENT_SCALE}.0;`,
            "fn atomic_add_f32(index: u32, value: f32, maximumWrites: u32) {",
            "  if (abs(value) < 0.00000000000000000001) { return; }",
            "  let safeMagnitude = max(1, i32(1073741823u / max(1u, maximumWrites)));",
            "  let raw = i32(round(clamp(value * FIXED_GRADIENT_SCALE, -2147483000.0, 2147483000.0)));",
            "  atomicMax(&fixedPointControl[1], u32(abs(raw)));",
            "  if (abs(raw) > safeMagnitude) { atomicAdd(&fixedPointControl[0], 1u); }",
            "  atomicAdd(&exactGradient[index], clamp(raw, -safeMagnitude, safeMagnitude));",
            "}",
          ]
        : [
            "fn atomic_add_f32(index: u32, value: f32, maximumWrites: u32) {",
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
          ]),
      "fn add_gradient(g: u32, geom: vec4<f32>, appearance: vec4<f32>, misc: vec4<f32>, density: vec4<f32>, depth: f32, maximumWrites: u32) {",
      "  let base = g * 16u;",
      "  atomic_add_f32(base, geom.x, maximumWrites);",
      "  atomic_add_f32(base + 1u, geom.y, maximumWrites);",
      "  atomic_add_f32(base + 2u, geom.z, maximumWrites);",
      "  atomic_add_f32(base + 3u, geom.w, maximumWrites);",
      "  atomic_add_f32(base + 4u, appearance.x, maximumWrites);",
      "  atomic_add_f32(base + 5u, appearance.y, maximumWrites);",
      "  atomic_add_f32(base + 6u, appearance.z, maximumWrites);",
      "  atomic_add_f32(base + 7u, appearance.w, maximumWrites);",
      "  atomic_add_f32(base + 8u, misc.x, maximumWrites);",
      "  atomic_add_f32(base + 9u, misc.y, maximumWrites);",
      "  atomic_add_f32(base + 10u, misc.z, maximumWrites);",
      "  atomic_add_f32(base + 11u, misc.w, maximumWrites);",
      "  atomic_add_f32(base + 12u, density.x, maximumWrites);",
      "  atomic_add_f32(base + 13u, density.y, maximumWrites);",
      "  atomic_add_f32(base + 14u, density.z, maximumWrites);",
      "  atomic_add_f32(base + 15u, depth, maximumWrites);",
      "}",
      ...(segmentedExactBackwardEnabled
        ? [
            "fn store_reduced_gradient(recordIndex: u32, g: u32, geom: vec4<f32>, appearance: vec4<f32>, misc: vec4<f32>, density: vec4<f32>, depth: f32) {",
            "  let base = recordIndex * 16u;",
            "  partialGradient[base] = geom.x;",
            "  partialGradient[base + 1u] = geom.y;",
            "  partialGradient[base + 2u] = geom.z;",
            "  partialGradient[base + 3u] = geom.w;",
            "  partialGradient[base + 4u] = appearance.x;",
            "  partialGradient[base + 5u] = appearance.y;",
            "  partialGradient[base + 6u] = appearance.z;",
            "  partialGradient[base + 7u] = appearance.w;",
            "  partialGradient[base + 8u] = misc.x;",
            "  partialGradient[base + 9u] = misc.y;",
            "  partialGradient[base + 10u] = misc.z;",
            "  partialGradient[base + 11u] = misc.w;",
            "  partialGradient[base + 12u] = density.x;",
            "  partialGradient[base + 13u] = density.y;",
            "  partialGradient[base + 14u] = density.z;",
            "  partialGradient[base + 15u] = depth;",
            "}",
          ]
        : [
            "fn store_reduced_gradient(recordIndex: u32, g: u32, geom: vec4<f32>, appearance: vec4<f32>, misc: vec4<f32>, density: vec4<f32>, depth: f32) {",
            `  let workgroupSide = ${this.quadExactBackwardEnabled ? TILE_SIZE : 8}u;`,
            "  let maximumWrites = ((u32(cfg(0u)) + workgroupSide - 1u) / workgroupSide) * ((u32(cfg(1u)) + workgroupSide - 1u) / workgroupSide) + 1u;",
            "  add_gradient(g, geom, appearance, misc, density, depth, maximumWrites);",
            "}",
          ]),
      "var<workgroup> reduceGeom: array<vec4<f32>, 64>;",
      "var<workgroup> reduceAppearance: array<vec4<f32>, 64>;",
      "var<workgroup> reduceMisc: array<vec4<f32>, 64>;",
      "var<workgroup> reduceDensity: array<vec4<f32>, 64>;",
      "var<workgroup> reduceDepth: array<f32, 64>;",
      ...exactBackwardReductionLines,
      "@compute @workgroup_size(8, 8, 1)",
      `fn exact_alpha_backward(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) localIndex: u32, @builtin(workgroup_id) wid: vec3<u32>${this.subgroupExactBackwardEnabled ? ", @builtin(subgroup_size) subgroupSize: u32, @builtin(subgroup_invocation_id) subgroupInvocation: u32" : ""}) {`,
      "  if (tile_list_overflow()) { return; }",
      "  let width = u32(cfg(0u));",
      "  let height = u32(cfg(1u));",
      "  let px = id.x;",
      "  let py = id.y;",
      "  let screenValid = px < width && py < height;",
      "  let pixel = min(px, width - 1u) + min(py, height - 1u) * width;",
      "  let outputPoint = vec2<f32>(select(0.0, f32(min(px, width - 1u)) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(min(py, height - 1u)) / f32(height - 1u) * 2.0 - 1.0, height > 1u));",
      "  let inversePoint = virtual_inverse_point(outputPoint);",
      "  let validPixel = screenValid && (inversePoint.z > 0.5 || cfg(66u) > 0.5);",
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
      "    dColor = packedLoss.rgb;",
      "    gradTransmittance = dot(dColor, vec3<f32>(cfg(3u), cfg(4u), cfg(5u))) - packedLoss.a;",
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
      "    var depth = 0.0;",
      "    if (validPixel && reverseCursor < acceptedEnd) {",
      "      let t = transform[g];",
      "      if (t.w >= 0.5) {",
      "        let evaluation = evaluate_kernel(g, p, outputPoint, xy[g].center, xy[g].rawDepth, t, width, height);",
      "        if (evaluation.rawKernel >= 0.0003354626) {",
      "          let rgba = color[g];",
      `          let fixedOpaque = cfg(85u) > 0.5 && cfg(90u) < 0.5;`,
      `          let minimumOpacity = select(0.0, cfg(88u), cfg(85u) > 0.5 && cfg(90u) > 0.5);`,
      `          let effectiveOpacity = select(clamp(rgba.a, minimumOpacity, cfg(118u)), cfg(88u), fixedOpaque);`,
      `          let alphaLimit = select(0.99, cfg(88u), fixedOpaque);`,
      "          let unclampedAlpha = evaluation.weightedKernel * effectiveOpacity;",
      "          let alpha = clamp(unclampedAlpha, 0.0, alphaLimit);",
      "          if (alpha >= 0.0039215686) {",
      "            let exactD = p - xy[g].center;",
      "            let exactC = cos(t.z);",
      "            let exactS = sin(t.z);",
      "            let surfaceRgb = rgba.rgb;",
      "            let transBefore = transAfter / max(1.0 - alpha, select(0.01, 0.005, fixedOpaque));",
      "            let dAlpha = transBefore * dot(dColor, surfaceRgb) - gradTransmittance * transBefore;",
      "            let differentiableAlpha = select(0.0, dAlpha, unclampedAlpha > 0.0 && unclampedAlpha < alphaLimit);",
      "            let dWeightedKernel = differentiableAlpha * effectiveOpacity;",
      "            var gradCenter = dWeightedKernel * evaluation.dCenter;",
      "            var gradLogScale = dWeightedKernel * evaluation.dLogScale;",
      "            var gradTheta = dWeightedKernel * evaluation.dTheta;",
      "            let influence = transBefore * alpha;",
      "            let underpaintingActive = cfg(93u) > 0.5 && cfg(8u) < cfg(100u);",
      "            let anchor = select(sign(surfaceRgb - targetAndError.rgb) * (cfg(44u) * influence / 3.0), vec3<f32>(0.0), underpaintingActive);",
      "            let surfaceGradient = dColor * influence + anchor;",
      "            let gradColor = surfaceGradient;",
      "            let gradLogit = select(differentiableAlpha * evaluation.weightedKernel * rgba.a * (1.0 - rgba.a), 0.0, fixedOpaque);",
      "            geom = vec4<f32>(gradCenter, gradLogScale) * cfg(35u);",
      "            appearance = vec4<f32>(gradColor, gradLogit) * cfg(36u);",
      "            misc = vec4<f32>(gradTheta * cfg(35u), influence, targetAndError.a * cfg(37u), cfg(37u));",
      "            density = vec4<f32>(abs(gradCenter), length(gradCenter), 0.0) * cfg(37u);",
      "            depth = dWeightedKernel * evaluation.dDepth * cfg(38u);",
      "            gradTransmittance = dot(dColor, alpha * surfaceRgb) + gradTransmittance * (1.0 - alpha);",
      "            transAfter = transBefore;",
      "          }",
      "        }",
      "      }",
      "    }",
      `    add_subtile_gradient(localIndex${this.subgroupExactBackwardEnabled ? ", subgroupSize, subgroupInvocation" : ""}, reverseCursor, g, geom, appearance, misc, density, depth);`,
      "  }",
      "}",
      "@compute @workgroup_size(64)",
      "fn exact_alpha_backward_source(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) {",
      "  if (tile_list_overflow()) { return; }",
      "  let width = u32(cfg(0u));",
      "  let height = u32(cfg(1u));",
      "  let pixel = id.x + id.y * workgroups.x * 64u;",
      "  if (pixel >= width * height) { return; }",
      "  let px = pixel % width;",
      "  let py = pixel / width;",
      "  let gridPoint = vec2<f32>(select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u), select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u));",
      "  let projectedPoint = training_output_point(gridPoint);",
      "  let outputPoint = projectedPoint.xy;",
      "  let inversePoint = training_source_point(gridPoint, projectedPoint);",
      "  if (!training_sample_valid(inversePoint)) { return; }",
      "  let useTiles = cfg(19u) > 0.5;",
      "  let tileCols = (width + 15u) / 16u;",
      "  let tilePixel = training_output_pixel(vec2<u32>(px, py), outputPoint, width, height);",
      "  let tile = (tilePixel.y / 16u) * tileCols + (tilePixel.x / 16u);",
      "  let tileCapacity = arrayLength(&tileIndices);",
      "  let start = select(0u, min(tile_offset(tile), tileCapacity), useTiles);",
      "  let end = select(u32(cfg(2u)), min(tile_offset(tile + 1u), tileCapacity), useTiles);",
      "  let acceptedEnd = min(end, max(start, alphaState[pixel].acceptedEnd));",
      "  let packedLoss = lossGradient[pixel * 3u];",
      "  let targetAndError = lossGradient[pixel * 3u + 1u];",
      "  let dColor = packedLoss.rgb;",
      "  var gradTransmittance = dot(dColor, vec3<f32>(cfg(3u), cfg(4u), cfg(5u))) - packedLoss.a;",
      "  var transAfter = clamp(1.0 - alphaState[pixel].compositeAlpha, 0.0, 1.0);",
      "  var reverseCursor = acceptedEnd;",
      "  loop {",
      "    if (reverseCursor <= start) { break; }",
      "    reverseCursor -= 1u;",
      "    let g = select(reverseCursor, tileIndices[reverseCursor], useTiles);",
      "    let t = transform[g];",
      "    if (t.w >= 0.5) {",
      "      let evaluation = evaluate_kernel(g, inversePoint.xy, outputPoint, xy[g].center, xy[g].rawDepth, t, width, height);",
      "      if (evaluation.rawKernel >= 0.0003354626) {",
      "        let rgba = color[g];",
      `        let fixedOpaque = cfg(85u) > 0.5 && cfg(90u) < 0.5;`,
      `        let minimumOpacity = select(0.0, cfg(88u), cfg(85u) > 0.5 && cfg(90u) > 0.5);`,
      `        let effectiveOpacity = select(clamp(rgba.a, minimumOpacity, cfg(118u)), cfg(88u), fixedOpaque);`,
      `        let alphaLimit = select(0.99, cfg(88u), fixedOpaque);`,
      "        let unclampedAlpha = evaluation.weightedKernel * effectiveOpacity;",
      "        let alpha = clamp(unclampedAlpha, 0.0, alphaLimit);",
      "        if (alpha >= 0.0039215686) {",
      "          let exactD = inversePoint.xy - xy[g].center;",
      "          let exactC = cos(t.z);",
      "          let exactS = sin(t.z);",
      "          let surfaceRgb = rgba.rgb;",
      "          let transBefore = transAfter / max(1.0 - alpha, select(0.01, 0.005, fixedOpaque));",
      "          let dAlpha = transBefore * dot(dColor, surfaceRgb) - gradTransmittance * transBefore;",
      "          let differentiableAlpha = select(0.0, dAlpha, unclampedAlpha > 0.0 && unclampedAlpha < alphaLimit);",
      "          let dWeightedKernel = differentiableAlpha * effectiveOpacity;",
      "          var gradCenter = dWeightedKernel * evaluation.dCenter;",
      "          var gradLogScale = dWeightedKernel * evaluation.dLogScale;",
      "          var gradTheta = dWeightedKernel * evaluation.dTheta;",
      "          let influence = transBefore * alpha;",
      "          let underpaintingActive = cfg(93u) > 0.5 && cfg(8u) < cfg(100u);",
      "          let anchor = select(sign(surfaceRgb - targetAndError.rgb) * (cfg(44u) * influence / 3.0), vec3<f32>(0.0), underpaintingActive);",
      "          let surfaceGradient = dColor * influence + anchor;",
      "          let gradColor = surfaceGradient;",
      "          let gradLogit = select(differentiableAlpha * evaluation.weightedKernel * rgba.a * (1.0 - rgba.a), 0.0, fixedOpaque);",
      "          let geom = vec4<f32>(gradCenter, gradLogScale) * cfg(35u);",
      "          let appearance = vec4<f32>(gradColor, gradLogit) * cfg(36u);",
      "          let misc = vec4<f32>(gradTheta * cfg(35u), influence, targetAndError.a * cfg(37u), cfg(37u));",
      "          let density = vec4<f32>(abs(gradCenter), length(gradCenter), 0.0) * cfg(37u);",
      "          let depth = dWeightedKernel * evaluation.dDepth * cfg(38u);",
      "          add_gradient(g, geom, appearance, misc, density, depth, width * height + 1u);",
      "          gradTransmittance = dot(dColor, alpha * surfaceRgb) + gradTransmittance * (1.0 - alpha);",
      "          transAfter = transBefore;",
      "        }",
      "      }",
      "    }",
      "  }",
      "}",
      "@compute @workgroup_size(8, 8, 1)",
      `fn exact_alpha_backward_quad(@builtin(local_invocation_index) localIndex: u32, @builtin(workgroup_id) wid: vec3<u32>${this.subgroupExactBackwardEnabled ? ", @builtin(subgroup_size) subgroupSize: u32, @builtin(subgroup_invocation_id) subgroupInvocation: u32" : ""}) {`,
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
      "    let validPixel = screenValid && (inversePoint.z > 0.5 || cfg(66u) > 0.5);",
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
      "      dColors[i] = packedLoss.rgb;",
      "      gradTransmittances[i] = dot(packedLoss.rgb, vec3<f32>(cfg(3u), cfg(4u), cfg(5u))) - packedLoss.a;",
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
      "    var depth = 0.0;",
      "    if (t.w >= 0.5) {",
      "      for (var i = 0u; i < 4u; i += 1u) {",
      "        if (validPixels[i] != 0u && reverseCursor < acceptedEnds[i]) {",
      "          let evaluation = evaluate_kernel(g, points[i], outputPoints[i], xy[g].center, xy[g].rawDepth, t, width, height);",
      "          if (evaluation.rawKernel >= 0.0003354626) {",
      `            let fixedOpaque = cfg(85u) > 0.5 && cfg(90u) < 0.5;`,
      `            let minimumOpacity = select(0.0, cfg(88u), cfg(85u) > 0.5 && cfg(90u) > 0.5);`,
      `            let effectiveOpacity = select(clamp(rgba.a, minimumOpacity, cfg(118u)), cfg(88u), fixedOpaque);`,
      `            let alphaLimit = select(0.99, cfg(88u), fixedOpaque);`,
      "            let unclampedAlpha = evaluation.weightedKernel * effectiveOpacity;",
      "            let alpha = clamp(unclampedAlpha, 0.0, alphaLimit);",
      "            if (alpha >= 0.0039215686) {",
      "              let exactD = points[i] - xy[g].center;",
      "              let exactC = cos(t.z);",
      "              let exactS = sin(t.z);",
      "              let surfaceRgb = rgba.rgb;",
      "              let transBefore = transAfters[i] / max(1.0 - alpha, select(0.01, 0.005, fixedOpaque));",
      "              let dAlpha = transBefore * dot(dColors[i], surfaceRgb) - gradTransmittances[i] * transBefore;",
      "              let differentiableAlpha = select(0.0, dAlpha, unclampedAlpha > 0.0 && unclampedAlpha < alphaLimit);",
      "              let dWeightedKernel = differentiableAlpha * effectiveOpacity;",
      "              var gradCenter = dWeightedKernel * evaluation.dCenter;",
      "              var gradLogScale = dWeightedKernel * evaluation.dLogScale;",
      "              var gradTheta = dWeightedKernel * evaluation.dTheta;",
      "              let influence = transBefore * alpha;",
      "              let underpaintingActive = cfg(93u) > 0.5 && cfg(8u) < cfg(100u);",
      "              let anchor = select(sign(surfaceRgb - targetsAndErrors[i].rgb) * (cfg(44u) * influence / 3.0), vec3<f32>(0.0), underpaintingActive);",
      "              let surfaceGradient = dColors[i] * influence + anchor;",
      "              geom += vec4<f32>(gradCenter, gradLogScale) * cfg(35u);",
      "              let gradLogit = select(differentiableAlpha * evaluation.weightedKernel * rgba.a * (1.0 - rgba.a), 0.0, fixedOpaque);",
      "              let gradColor = surfaceGradient;",
      "              appearance += vec4<f32>(gradColor, gradLogit) * cfg(36u);",
      "              misc += vec4<f32>(gradTheta * cfg(35u), influence, targetsAndErrors[i].a * cfg(37u), cfg(37u));",
      "              density += vec4<f32>(abs(gradCenter), length(gradCenter), 0.0) * cfg(37u);",
      "              depth += dWeightedKernel * evaluation.dDepth * cfg(38u);",
      "              gradTransmittances[i] = dot(dColors[i], alpha * surfaceRgb) + gradTransmittances[i] * (1.0 - alpha);",
      "              transAfters[i] = transBefore;",
      "            }",
      "          }",
      "        }",
      "      }",
      "    }",
      `    add_subtile_gradient(localIndex${this.subgroupExactBackwardEnabled ? ", subgroupSize, subgroupInvocation" : ""}, reverseCursor, g, geom, appearance, misc, density, depth);`,
      "  }",
      "}",
    ].join("\n");

    const segmentedReferenceShader = `
struct Config { values: array<vec4<f32>, 32>, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(2) var<storage, read> tileIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> referenceCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> referenceOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> referenceCursors: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> references: array<u32>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn active_reference_count() -> u32 {
  return min(
    tileOffsets[arrayLength(&tileOffsets) - 1u] & 0x7fffffffu,
    arrayLength(&tileIndices)
  );
}

@compute @workgroup_size(64)
fn count_references(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let cursor = id.x + id.y * workgroups.x * 64u;
  if (cursor >= active_reference_count()) { return; }
  let g = tileIndices[cursor];
  if (g < u32(cfg(2u))) { atomicAdd(&referenceCounts[g], 1u); }
}

@compute @workgroup_size(1)
fn prefix_references(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x != 0u) { return; }
  let count = min(u32(cfg(2u)), arrayLength(&referenceCounts));
  var total = 0u;
  for (var g = 0u; g < count; g += 1u) {
    referenceOffsets[g] = total;
    total += atomicLoad(&referenceCounts[g]);
  }
  referenceOffsets[count] = total;
}

@compute @workgroup_size(64)
fn fill_references(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let cursor = id.x + id.y * workgroups.x * 64u;
  if (cursor >= active_reference_count()) { return; }
  let g = tileIndices[cursor];
  if (g >= u32(cfg(2u))) { return; }
  let local = atomicAdd(&referenceCursors[g], 1u);
  let destination = referenceOffsets[g] + local;
  if (destination < arrayLength(&references)) { references[destination] = cursor; }
}`;

    const segmentedGradientReduceShader = `
struct Config { values: array<vec4<f32>, 27>, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> referenceOffsets: array<u32>;
@group(0) @binding(2) var<storage, read> references: array<u32>;
@group(0) @binding(3) var<storage, read> partialGradient: array<f32>;
@group(0) @binding(4) var<storage, read_write> exactGradient: array<f32>;
var<workgroup> sum0: array<vec4<f32>, 64>;
var<workgroup> sum1: array<vec4<f32>, 64>;
var<workgroup> sum2: array<vec4<f32>, 64>;
var<workgroup> sum3: array<vec4<f32>, 64>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

@compute @workgroup_size(64)
fn reduce_segmented_gradients(
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let g = wid.x + wid.y * workgroups.x;
  let count = u32(cfg(2u));
  if (g >= count) { return; }
  let start = referenceOffsets[g];
  let end = referenceOffsets[g + 1u];
  var local0 = vec4<f32>(0.0);
  var local1 = vec4<f32>(0.0);
  var local2 = vec4<f32>(0.0);
  var local3 = vec4<f32>(0.0);
  for (var cursor = start + localIndex; cursor < end; cursor += 64u) {
    let record = references[cursor] * 16u;
    local0 += vec4<f32>(
      partialGradient[record],
      partialGradient[record + 1u],
      partialGradient[record + 2u],
      partialGradient[record + 3u]
    );
    local1 += vec4<f32>(
      partialGradient[record + 4u],
      partialGradient[record + 5u],
      partialGradient[record + 6u],
      partialGradient[record + 7u]
    );
    local2 += vec4<f32>(
      partialGradient[record + 8u],
      partialGradient[record + 9u],
      partialGradient[record + 10u],
      partialGradient[record + 11u]
    );
    local3 += vec4<f32>(
      partialGradient[record + 12u],
      partialGradient[record + 13u],
      partialGradient[record + 14u],
      partialGradient[record + 15u]
    );
  }
  sum0[localIndex] = local0;
  sum1[localIndex] = local1;
  sum2[localIndex] = local2;
  sum3[localIndex] = local3;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (localIndex < stride) {
      sum0[localIndex] += sum0[localIndex + stride];
      sum1[localIndex] += sum1[localIndex + stride];
      sum2[localIndex] += sum2[localIndex + stride];
      sum3[localIndex] += sum3[localIndex + stride];
    }
    workgroupBarrier();
  }
  if (localIndex == 0u) {
    let base = g * 16u;
    exactGradient[base] = sum0[0].x;
    exactGradient[base + 1u] = sum0[0].y;
    exactGradient[base + 2u] = sum0[0].z;
    exactGradient[base + 3u] = sum0[0].w;
    exactGradient[base + 4u] = sum1[0].x;
    exactGradient[base + 5u] = sum1[0].y;
    exactGradient[base + 6u] = sum1[0].z;
    exactGradient[base + 7u] = sum1[0].w;
    exactGradient[base + 8u] = sum2[0].x;
    exactGradient[base + 9u] = sum2[0].y;
    exactGradient[base + 10u] = sum2[0].z;
    exactGradient[base + 11u] = sum2[0].w;
    exactGradient[base + 12u] = sum3[0].x;
    exactGradient[base + 13u] = sum3[0].y;
    exactGradient[base + 14u] = sum3[0].z;
    exactGradient[base + 15u] = sum3[0].w;
  }
}`;

    const exactBackwardTelemetryShader = `
struct Config { values: array<vec4<f32>, 14>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
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

fn distribution_bin(value: u32) -> u32 {
  if (value == 0u) { return 0u; }
  if (value == 1u) { return 1u; }
  if (value <= 3u) { return 2u; }
  if (value <= 7u) { return 3u; }
  if (value <= 15u) { return 4u; }
  if (value <= 31u) { return 5u; }
  if (value <= 63u) { return 6u; }
  return 7u;
}

fn footprint_intersects_subtile(g: u32, wid: vec3<u32>, width: u32, height: u32) -> bool {
  let t = transform[g];
  if (t.w < 0.5) { return false; }
  let center = xy[g].center;
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
  let pixelAcceptedEnd = select(start, min(end, max(start, alphaState[pixel].acceptedEnd)), validPixel);
  acceptedEndMax[localIndex] = pixelAcceptedEnd;
  if (validPixel) {
    let candidateCount = end - start;
    let contributorCount = alphaState[pixel].pad1;
    atomicAdd(&counters[5], candidateCount);
    atomicAdd(&counters[6], contributorCount);
    atomicAdd(&counters[7], 1u);
    atomicAdd(&counters[8u + distribution_bin(candidateCount)], 1u);
    atomicAdd(&counters[16u + distribution_bin(contributorCount)], 1u);
  }
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
struct Config { values: array<vec4<f32>, 19>, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> exactGradient: array<atomic<${fixedPointExactGradientEnabled ? "i32" : "u32"}>>;
${fixedPointExactGradientEnabled ? "@group(0) @binding(5) var<storage, read_write> fixedPointControl: array<atomic<u32>>;" : ""}

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

${VIRTUAL_TILT_WGSL}

fn load_f32(index: u32) -> f32 {
  return ${fixedPointExactGradientEnabled
    ? `f32(atomicLoad(&exactGradient[index])) / ${FIXED_POINT_EXACT_GRADIENT_SCALE}.0`
    : "bitcast<f32>(atomicLoad(&exactGradient[index]))"};
}

fn atomic_add_f32(index: u32, value: f32) {
  if (abs(value) < 0.00000000000000000001) { return; }
  ${fixedPointExactGradientEnabled
    ? `let safeMagnitude = 536870911;
  let raw = i32(round(clamp(value * ${FIXED_POINT_EXACT_GRADIENT_SCALE}.0, -2147483000.0, 2147483000.0)));
  atomicMax(&fixedPointControl[1], u32(abs(raw)));
  if (abs(raw) > safeMagnitude) { atomicAdd(&fixedPointControl[0], 1u); }
  atomicAdd(&exactGradient[index], clamp(raw, -safeMagnitude, safeMagnitude));`
    : `var oldBits = atomicLoad(&exactGradient[index]);
  loop {
    let oldValue = bitcast<f32>(oldBits);
    let exchanged = atomicCompareExchangeWeak(&exactGradient[index], oldBits, bitcast<u32>(oldValue + value));
    if (exchanged.exchanged) { break; }
    oldBits = exchanged.old_value;
  }`}
}

@compute @workgroup_size(64)
fn virtual_order_penalty(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let g = id.x + id.y * workgroups.x * 64u;
  let count = u32(cfg(2u));
  if (g >= count) { return; }
  let t = transform[g];
  let rgba = color[g];
  if (t.w < 0.5 || rgba.a < 0.007) { return; }
  let base = g * ${EXACT_GRADIENT_STRIDE}u;
  if (!virtual_tilt_enabled() || cfg(60u) <= 0.0) { return; }
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
  let influence = max(abs(load_f32(base + 9u)), 0.01);
  let residual = max(0.0, load_f32(base + 10u)) / influence;
  let coefficient = 2.0 * cfg(60u) * residual * excess * influence / ${PLY_LAYER_DEPTH_SPAN};
  let dSupportX = ${RENDER_SIGMA}.0 * depthX * depthX / sigmaDepth;
  let dSupportY = ${RENDER_SIGMA}.0 * depthY * depthY / sigmaDepth;
  atomic_add_f32(base + 2u, coefficient * dSupportX);
  atomic_add_f32(base + 3u, coefficient * dSupportY);
}
`;

    const brushLocalColorFlowShader = `
struct Config { values: array<vec4<f32>, 32>, };
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> exactGradient: array<atomic<${fixedPointExactGradientEnabled ? "i32" : "u32"}>>;
${fixedPointExactGradientEnabled ? "@group(0) @binding(5) var<storage, read_write> fixedPointControl: array<atomic<u32>>;" : ""}

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn atomic_add_theta(index: u32, value: f32) {
  if (abs(value) < 0.00000000000000000001) { return; }
  ${fixedPointExactGradientEnabled
    ? `let safeMagnitude = 536870911;
  let raw = i32(round(clamp(value * ${FIXED_POINT_EXACT_GRADIENT_SCALE}.0, -2147483000.0, 2147483000.0)));
  atomicMax(&fixedPointControl[1], u32(abs(raw)));
  if (abs(raw) > safeMagnitude) { atomicAdd(&fixedPointControl[0], 1u); }
  atomicAdd(&exactGradient[index], clamp(raw, -safeMagnitude, safeMagnitude));`
    : `var oldBits = atomicLoad(&exactGradient[index]);
  loop {
    let oldValue = bitcast<f32>(oldBits);
    let exchanged = atomicCompareExchangeWeak(&exactGradient[index], oldBits, bitcast<u32>(oldValue + value));
    if (exchanged.exchanged) { break; }
    oldBits = exchanged.old_value;
  }`}
}

@compute @workgroup_size(64)
fn brush_local_color_flow(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  let count = u32(cfg(2u));
  let strength = clamp(cfg(111u), 0.0, 0.05);
  if (g >= count || strength <= 0.0 || cfg(40u) < 3.5) { return; }
  let t = transform[g];
  let rgba = color[g];
  let minor = max(0.0001, min(t.x, t.y));
  let anisotropy = max(t.x, t.y) / minor;
  if (t.w < 0.5 || rgba.a < 0.5 || anisotropy < ${ILLUSTRATIVE_OIL_RIBBON_ANISOTROPY}) { return; }

  let center = xy[g].center;
  let imageScale = vec2<f32>(max(cfg(0u) - 1.0, 1.0), max(cfg(1u) - 1.0, 1.0));
  let radiusPx = clamp(max(t.x, t.y) * 0.5 * max(cfg(0u), cfg(1u)) * 2.0, 4.0, 18.0);
  let layerCount = max(2.0, round(cfg(82u)));
  let layer = floor(clamp(min(fract(t.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 0.999999) * layerCount);
  var axis = vec2<f32>(0.0);
  var weightSum = 0.0;
  var neighbors = 0u;
  for (var probe = 0u; probe < 12u; probe += 1u) {
    let stride = 17u + probe * 12u;
    let j = (g + 1u + stride * stride) % count;
    if (j == g) { continue; }
    let nt = transform[j];
    let nc = color[j];
    let nMinor = max(0.0001, min(nt.x, nt.y));
    if (nt.w < 0.5 || nc.a < 0.5 || max(nt.x, nt.y) / nMinor < ${ILLUSTRATIVE_OIL_RIBBON_ANISOTROPY}) { continue; }
    let nLayer = floor(clamp(min(fract(nt.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 0.999999) * layerCount);
    if (nLayer != layer) { continue; }
    let distancePx = length((xy[j].center - center) * 0.5 * imageScale);
    let colorDistance = length(nc.rgb - rgba.rgb);
    if (distancePx <= 0.25 || distancePx > radiusPx || colorDistance > 0.10) { continue; }
    let agreement = cos(2.0 * (nt.z - t.z));
    if (agreement < 0.0) { continue; }
    let weight = (1.0 - distancePx / radiusPx) * (1.0 - colorDistance / 0.10) * agreement;
    axis += weight * vec2<f32>(cos(2.0 * nt.z), sin(2.0 * nt.z));
    weightSum += weight;
    neighbors += 1u;
  }
  if (neighbors < 2u || weightSum <= 0.0001 || length(axis) / weightSum < 0.65) { return; }
  let targetTheta = 0.5 * atan2(axis.y, axis.x);
  let delta = 0.5 * atan2(sin(2.0 * (t.z - targetTheta)), cos(2.0 * (t.z - targetTheta)));
  atomic_add_theta(g * ${EXACT_GRADIENT_STRIDE}u + 8u, strength * delta);
}
`;

    const optimizerShader = `
struct Config { values: array<vec4<f32>, 32>, };
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
};
@group(0) @binding(0) var<uniform> config: Config;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read_write> xy: array<SplatPosition>;
@group(0) @binding(2) var<storage, read_write> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(5) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> ssimTiles: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> adam: array<AdamState>;
@group(0) @binding(9) var<storage, read> exactGradients: array<${fixedPointExactGradientEnabled ? "i32" : "f32"}>;
@group(0) @binding(10) var<storage, read_write> tileControl: array<atomic<u32>>;
${optimizerStatsDeclaration}

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn exact_gradient(index: u32) -> f32 {
  return ${fixedPointExactGradientEnabled
    ? `f32(exactGradients[index]) / ${FIXED_POINT_EXACT_GRADIENT_SCALE}.0`
    : "exactGradients[index]"};
}
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

fn target_rgb_at_point(point: vec2<f32>, width: u32, height: u32) -> vec3<f32> {
  let source = clamp(
    (point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)),
    vec2<f32>(0.0),
    vec2<f32>(f32(width - 1u), f32(height - 1u))
  );
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

fn paint_footprint_rgb(
  center: vec2<f32>,
  scale: vec2<f32>,
  theta: f32,
  width: u32,
  height: u32,
) -> vec4<f32> {
  let c = cos(theta);
  let s = sin(theta);
  let longOffset = vec2<f32>(c, s) * scale.x * 0.75;
  let shortOffset = vec2<f32>(-s, c) * scale.y * 0.75;
  let centerColor = target_rgb_at_point(center, width, height);
  let longA = target_rgb_at_point(center + longOffset, width, height);
  let longB = target_rgb_at_point(center - longOffset, width, height);
  let shortA = target_rgb_at_point(center + shortOffset, width, height);
  let shortB = target_rgb_at_point(center - shortOffset, width, height);
  let footprintColor = (2.0 * centerColor + longA + longB + shortA + shortB) / 6.0;
  let footprintVariation = (
    dot(abs(centerColor - footprintColor), vec3<f32>(1.0 / 3.0)) * 2.0 +
    dot(abs(longA - footprintColor), vec3<f32>(1.0 / 3.0)) +
    dot(abs(longB - footprintColor), vec3<f32>(1.0 / 3.0)) +
    dot(abs(shortA - footprintColor), vec3<f32>(1.0 / 3.0)) +
    dot(abs(shortB - footprintColor), vec3<f32>(1.0 / 3.0))
  ) / 6.0;
  return vec4<f32>(footprintColor, footprintVariation);
}

struct SurfaceLayerUpdate {
  order: f32,
  color: vec3<f32>,
  resetColorAdam: f32,
};

fn size_sorted_surface_layer(
  g: u32,
  center: vec2<f32>,
  scale: vec2<f32>,
  theta: f32,
  currentOrder: f32,
  currentColor: vec3<f32>,
  observed: f32,
  influenceSum: f32,
) -> SurfaceLayerUpdate {
  var result = SurfaceLayerUpdate(currentOrder, currentColor, 0.0);
  if (cfg(105u) <= 0.5 || observed <= 0.0 || influenceSum <= 0.000001) {
    return result;
  }
  let sortLayers = max(2.0, round(cfg(106u)));
  let minimumArea = ${MIN_SPLAT_SCALE * MIN_SPLAT_SCALE};
  let maximumAxis = max(cfg(62u), ${PHASE_ONE_MAX_PLANAR_SCALE});
  let maximumArea = max(minimumArea * 1.0001, maximumAxis * maximumAxis);
  let area = clamp(scale.x * scale.y, minimumArea, maximumArea);
  let sizeRank = clamp(
    (log(area) - log(minimumArea)) / max(log(maximumArea) - log(minimumArea), 0.000001),
    0.0,
    1.0
  );
  let frontRank = min(0.999999, 1.0 - sizeRank);
  let targetLayer = min(sortLayers - 1.0, floor(frontRank * sortLayers));
  let currentLayer = min(sortLayers - 1.0, floor(min(currentOrder, 0.999999) * sortLayers));
  let stableInLayer = fract(min(currentOrder, 0.999999) * sortLayers);

  // A backward move cannot expose a stale color. For a forward move, repair
  // the color from the same footprint before publishing the new layer.
  let surfaceFlags = u32(round(cfg(107u)));
  if (targetLayer > currentLayer && (surfaceFlags & 8u) != 0u && !(cfg(93u) > 0.5 && cfg(8u) < cfg(100u))) {
    let width = u32(cfg(0u));
    let height = u32(cfg(1u));
    let footprint = paint_footprint_rgb(center, scale, theta, width, height);
    let mismatch = dot(abs(currentColor - footprint.rgb), vec3<f32>(1.0 / 3.0));
    let paint = cfg(40u) > 0.5;
    let mismatchThreshold = select(0.08, 0.05, paint);
    let variationLimit = select(0.05, 0.08, paint);
    if (mismatch > mismatchThreshold && footprint.a > variationLimit) {
      return result;
    }
    if (mismatch > mismatchThreshold) {
      result.color = mix(currentColor, footprint.rgb, select(0.20, 0.75, paint));
      result.resetColorAdam = 1.0;
    }
  }
  result.order = (targetLayer + stableInLayer * 0.999999) / sortLayers;
  return result;
}

fn color_guarded_trained_layer(
  center: vec2<f32>,
  scale: vec2<f32>,
  theta: f32,
  currentOrder: f32,
  proposedOrder: f32,
  currentColor: vec3<f32>,
) -> SurfaceLayerUpdate {
  var result = SurfaceLayerUpdate(proposedOrder, currentColor, 0.0);
  let flags = u32(round(cfg(107u)));
  let paint = cfg(40u) > 0.5;
  let underpainting = cfg(93u) > 0.5 && cfg(8u) < cfg(100u);
  if ((flags & 16u) == 0u || !paint || underpainting || proposedOrder <= currentOrder) {
    return result;
  }
  let footprint = paint_footprint_rgb(
    center, scale, theta, u32(cfg(0u)), u32(cfg(1u))
  );
  let mismatch = dot(abs(currentColor - footprint.rgb), vec3<f32>(1.0 / 3.0));
  if (mismatch <= 0.05) { return result; }

  // Repair before publishing a forward order change. High-variation edge
  // footprints are only deferred so that the normal color optimizer can fit
  // them without replacing a deliberate Brush boundary with an average.
  result.order = currentOrder;
  if (footprint.a <= 0.08) {
    result.color = mix(currentColor, footprint.rgb, 0.75);
    result.resetColorAdam = 1.0;
  }
  return result;
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
  let dSsim = ssimTiles[pixel * 4u];
  let x = dot(renderedState.rgb, vec3<f32>(1.0 / 3.0));
  let y = dot(targetColor, vec3<f32>(1.0 / 3.0));
  dLoss += -${DEFAULT_DSSIM_WEIGHT} * dSsim.rgb / 3.0;
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
${protectedPrefix ? `  if (g < ${protectedPrefix}u) { return; }\n` : ""}  let capacity = u32(cfg(28u));
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
  let gradDepth = densitySum.w;
  let normalizer = 1.0 / max(influenceSum, 0.01);
  let optimizerScaleGradient = ${inverseScaleOptimizationEnabled
    ? "-gradLogScale * baseScale"
    : "gradLogScale"};
  let gradGeom = vec4<f32>(gradCenter, optimizerScaleGradient) * normalizer;
  let gradAppearance = vec4<f32>(gradColor, gradLogit) * normalizer;
  let gradRotation = vec4<f32>(gradTheta * normalizer, 0.0, 0.0, 0.0);
  let beta1 = 0.9;
  let beta2 = 0.999;
  var opt = adam[g];
  let adcResetStep = opt.mTheta.w;
  let previousDepthM = opt.mTheta.y;
  let previousDepthV = opt.vTheta.y;
  let previousDepthUpdates = opt.mTheta.z;
  let previousDepthObservations = opt.vTheta.w;
  opt.mGeom = beta1 * opt.mGeom + (1.0 - beta1) * gradGeom;
  opt.vGeom = beta2 * opt.vGeom + (1.0 - beta2) * gradGeom * gradGeom;
  opt.mColor = beta1 * opt.mColor + (1.0 - beta1) * gradAppearance;
  opt.vColor = beta2 * opt.vColor + (1.0 - beta2) * gradAppearance * gradAppearance;
  opt.mTheta = beta1 * opt.mTheta + (1.0 - beta1) * gradRotation;
  opt.vTheta = beta2 * opt.vTheta + (1.0 - beta2) * gradRotation * gradRotation;
  opt.mTheta.y = previousDepthM;
  opt.vTheta.y = previousDepthV;
  opt.mTheta.z = previousDepthUpdates;
  opt.vTheta.w = previousDepthObservations;
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
  let phaseOneShapeBoost = max(cfg(79u), 1.0);
  let positionLr = cfg(10u) * lrScale * spatialGate;
  let colorLr = cfg(11u) * lrScale * colorGate;
  let opacityLr = cfg(12u) * lrScale * spatialGate;
  let scaleLr = cfg(13u) * lrScale * spatialGate * phaseOneShapeBoost;
  let rotationLr = cfg(14u) * lrScale * spatialGate * phaseOneShapeBoost;
  var centerUpdate = geomAdam.xy * positionLr;
  var nextCenter = center - centerUpdate;
  var nextScale = ${inverseScaleOptimizationEnabled
    ? "1.0 / max((1.0 / baseScale) * max(vec2<f32>(0.5), vec2<f32>(1.0) - geomAdam.zw * scaleLr), vec2<f32>(0.0001))"
    : "exp(log(baseScale) - geomAdam.zw * scaleLr)"};
  var nextColor = clamp(rgba.rgb - colorAdam.rgb * colorLr, vec3<f32>(0.0), vec3<f32>(1.0));
  if (cfg(93u) > 0.5 && step < cfg(100u)) {
    nextColor = vec3<f32>(dot(nextColor, vec3<f32>(1.0 / 3.0)));
  }
  let currentLogit = log(clamp(rgba.a, 0.005, 0.995) / (1.0 - clamp(rgba.a, 0.005, 0.995)));
  var nextOpacity = select(
    1.0 / (1.0 + exp(-(currentLogit - colorAdam.w * opacityLr))),
    cfg(88u),
    cfg(85u) > 0.5 && cfg(90u) < 0.5
  );
  if (cfg(85u) > 0.5 && cfg(90u) > 0.5) {
    nextOpacity = clamp(nextOpacity, cfg(88u), cfg(118u));
  }
  var nextTheta = t.z - thetaAdam.x * rotationLr;
  var nextVirtualDepthRaw = xy[g].rawDepth;
  var accumulatedDepthGradient = xy[g].depthGradient;
  var depthObservationCount = previousDepthObservations;
  if (cfg(67u) > 0.5) {
    let boundedDepth = cfg(68u) * tanh(nextVirtualDepthRaw);
    let depthChain = cfg(68u) * (1.0 - tanh(nextVirtualDepthRaw) * tanh(nextVirtualDepthRaw));
    let depthCameraConfidence = clamp(cfg(108u), 0.0, 1.0);
    if (depthCameraConfidence > 0.0) {
      accumulatedDepthGradient += gradDepth * normalizer * depthChain * depthCameraConfidence;
      depthObservationCount += 1.0;
    }
    if (cfg(76u) > 0.5) {
      var centerPriorSlope = boundedDepth;
      if (cfg(110u) > 0.5) {
        let priorDelta = max(cfg(109u), 0.00001);
        centerPriorSlope = boundedDepth / sqrt(1.0 + boundedDepth * boundedDepth / (priorDelta * priorDelta));
      }
      var regularizationGradient = 2.0 * cfg(69u) * centerPriorSlope * depthChain;
      var neighbor = g;
      if (g + 1u < u32(cfg(2u))) { neighbor = g + 1u; }
      else if (g > 0u) { neighbor = g - 1u; }
      if (neighbor != g && transform[neighbor].w >= 0.5) {
        let delta = xy[g].center - xy[neighbor].center;
        let spatialWeight = exp(-dot(delta, delta) / 0.01);
        let neighborDepth = cfg(68u) * tanh(xy[neighbor].rawDepth);
        regularizationGradient += 2.0 * cfg(70u) * spatialWeight * (boundedDepth - neighborDepth) * depthChain;
      }
      let depthGradient = accumulatedDepthGradient / max(depthObservationCount, 1.0) + regularizationGradient;
      let depthM = beta1 * previousDepthM + (1.0 - beta1) * depthGradient;
      let depthV = beta2 * previousDepthV + (1.0 - beta2) * depthGradient * depthGradient;
      let depthUpdates = previousDepthUpdates + 1.0;
      let depthBias1 = max(0.000001, 1.0 - pow(beta1, depthUpdates));
      let depthBias2 = max(0.000001, 1.0 - pow(beta2, depthUpdates));
      let depthAdam = (depthM / depthBias1) / (sqrt(depthV / depthBias2) + 0.00000001);
      nextVirtualDepthRaw = clamp(nextVirtualDepthRaw - depthAdam * cfg(71u), -${VIRTUAL_DEPTH_RAW_LIMIT}.0, ${VIRTUAL_DEPTH_RAW_LIMIT}.0);
      opt.mTheta.y = depthM;
      opt.vTheta.y = depthV;
      opt.mTheta.z = depthUpdates;
      accumulatedDepthGradient = 0.0;
      depthObservationCount = 0.0;
    }
  } else if (cfg(40u) > 3.5 && cfg(95u) > 0.5) {
    let taperGradient = gradDepth * normalizer;
    let taperM = beta1 * previousDepthM + (1.0 - beta1) * taperGradient;
    let taperV = beta2 * previousDepthV + (1.0 - beta2) * taperGradient * taperGradient;
    let taperUpdates = previousDepthUpdates + 1.0;
    let taperBias1 = max(0.000001, 1.0 - pow(beta1, taperUpdates));
    let taperBias2 = max(0.000001, 1.0 - pow(beta2, taperUpdates));
    let taperAdam = (taperM / taperBias1) / (sqrt(taperV / taperBias2) + 0.00000001);
    nextVirtualDepthRaw = clamp(
      nextVirtualDepthRaw - taperAdam * cfg(87u) * lrScale * spatialGate,
      0.0,
      1.0
    );
    opt.mTheta.y = taperM;
    opt.vTheta.y = taperV;
    opt.mTheta.z = taperUpdates;
    accumulatedDepthGradient = 0.0;
    depthObservationCount = 0.0;
  } else {
    nextVirtualDepthRaw = 0.0;
    accumulatedDepthGradient = 0.0;
    depthObservationCount = 0.0;
  }
  opt.vTheta.w = depthObservationCount;
  let u1 = max(0.000001, hash_unit(f32(g) * 17.13 + step * 0.73));
  let u2 = hash_unit(f32(g) * 31.71 + step * 1.37);
  let normal = sqrt(-2.0 * log(u1)) * vec2<f32>(cos(6.28318530718 * u2), sin(6.28318530718 * u2));
  let covarianceNoise = vec2<f32>(c * normal.x * baseScale.x - s * normal.y * baseScale.y, s * normal.x * baseScale.x + c * normal.y * baseScale.y) / max(max(baseScale.x, baseScale.y), 0.0001);
  let noiseStep = (1.0 - progress) * positionLr *
    select(1.0, 0.0, cfg(73u) > 0.5);
  let defaultNoiseGate = 1.0 - rgba.a;
  let sigmoidNoiseGate = 1.0 / (1.0 + exp((rgba.a - 0.2) * 20.0));
  let noiseGate = select(defaultNoiseGate, sigmoidNoiseGate, cfg(34u) > 0.5);
  nextCenter += covarianceNoise * noiseStep * noiseGate * ${DEFAULT_SGLD_NOISE_LR};
  let minScale = max(${MIN_SPLAT_SCALE}, cfg(80u));
  nextTheta = clamp(nextTheta, -3.14159265, 3.14159265);
  nextScale = max(nextScale, vec2<f32>(minScale));
  if (cfg(122u) > 0.5 && cfg(120u) > minScale) {
    let geometricScale = sqrt(max(minScale * minScale, nextScale.x * nextScale.y));
    if (geometricScale < cfg(120u)) {
      let correction = pow(
        cfg(120u) / max(geometricScale, minScale),
        clamp(cfg(121u), 0.0, 1.0)
      );
      nextScale *= correction;
    }
  }
  let phaseOneProgress = clamp(step / max(cfg(39u), 1.0), 0.0, 1.0);
  let phaseMaxPlanarScale = mix(max(cfg(62u), ${PHASE_ONE_MAX_PLANAR_SCALE}), max(cfg(62u), minScale), phaseOneProgress);
  nextScale = min(nextScale, vec2<f32>(phaseMaxPlanarScale));
  let major = max(nextScale.x, nextScale.y);
  let minor = max(minScale, min(nextScale.x, nextScale.y));
  let baseMaxAnisotropy = max(cfg(17u), 1.0);
  let surfaceMaxAnisotropy = max(cfg(51u), 1.0);
  let detailTagged = floor(t.w) >= 2.0;
  var maxAnisotropy = select(min(baseMaxAnisotropy, surfaceMaxAnisotropy), baseMaxAnisotropy, detailTagged);
  if (cfg(40u) > 0.5 && cfg(40u) < 1.5) {
    maxAnisotropy = min(maxAnisotropy, max(cfg(91u), 1.0));
  }
  if (major / minor > maxAnisotropy) {
    let capped = minor * maxAnisotropy;
    nextScale = select(vec2<f32>(minor, capped), vec2<f32>(capped, minor), nextScale.x >= nextScale.y);
  }
  if (cfg(40u) > 3.5) {
    let brushMajorIsX = nextScale.x >= nextScale.y;
    let brushRatio = max(nextScale.x, nextScale.y) / max(minScale, min(nextScale.x, nextScale.y));
    let brushAreaRadius = sqrt(max(minScale * minScale, nextScale.x * nextScale.y));
    let brushMinorFeasibleRatio = brushAreaRadius * brushAreaRadius / (minScale * minScale);
    let brushMajorFeasibleRatio = phaseMaxPlanarScale * phaseMaxPlanarScale /
      max(brushAreaRadius * brushAreaRadius, minScale * minScale);
    let brushTargetRatio = min(
      max(1.0, cfg(119u)),
      min(maxAnisotropy, min(brushMinorFeasibleRatio, brushMajorFeasibleRatio))
    );
    if (brushRatio < brushTargetRatio) {
      let brushNextMajor = brushAreaRadius * sqrt(brushTargetRatio);
      let brushNextMinor = brushAreaRadius / sqrt(brushTargetRatio);
      nextScale = select(
        vec2<f32>(brushNextMinor, brushNextMajor),
        vec2<f32>(brushNextMajor, brushNextMinor),
        brushMajorIsX
      );
    }
  }
  if (cfg(40u) > 0.5 && cfg(40u) < 1.5) {
    let rectangleMajorIsX = nextScale.x >= nextScale.y;
    let rectangleMajor = max(nextScale.x, nextScale.y);
    let rectangleMinor = max(minScale, min(nextScale.x, nextScale.y));
    let rectangleRatio = rectangleMajor / rectangleMinor;
    let rectangleAreaRadius = sqrt(max(minScale * minScale, nextScale.x * nextScale.y));
    let rectangleMinorFeasibleRatio = rectangleAreaRadius * rectangleAreaRadius / (minScale * minScale);
    let rectangleMajorFeasibleRatio = phaseMaxPlanarScale * phaseMaxPlanarScale /
      max(rectangleAreaRadius * rectangleAreaRadius, minScale * minScale);
    let rectangleTargetRatio = min(
      max(1.0, cfg(115u)),
      min(maxAnisotropy, min(rectangleMinorFeasibleRatio, rectangleMajorFeasibleRatio))
    );
    if (rectangleRatio < rectangleTargetRatio) {
      let rectangleNextMajor = rectangleAreaRadius * sqrt(rectangleTargetRatio);
      let rectangleNextMinor = rectangleAreaRadius / sqrt(rectangleTargetRatio);
      nextScale = select(
        vec2<f32>(rectangleNextMinor, rectangleNextMajor),
        vec2<f32>(rectangleNextMajor, rectangleNextMinor),
        rectangleMajorIsX
      );
    }
  }
  if (cfg(40u) > 3.5 && cfg(112u) > 0.5) {
    let persistentMajorIsX = nextScale.x >= nextScale.y;
    let persistentMajor = max(nextScale.x, nextScale.y);
    let persistentMinor = max(minScale, min(nextScale.x, nextScale.y));
    let persistentRatio = persistentMajor / persistentMinor;
    let persistentAccent = floor(t.w) >= 2.0;
    let persistentDirectional =
      persistentAccent || persistentRatio >= ${BRUSH_STROKE_PERSISTENCE_DIRECTIONAL_RATIO};
    if (persistentDirectional) {
      let persistentAreaRadius = sqrt(max(minScale * minScale, nextScale.x * nextScale.y));
      let persistentFeasibleRatio = max(1.0, persistentAreaRadius * persistentAreaRadius / (minScale * minScale));
      let persistentTargetRatio = min(
        min(maxAnisotropy, persistentFeasibleRatio),
        select(
          max(1.0, cfg(113u)),
          max(1.0, cfg(114u)),
          persistentAccent
        )
      );
      let persistentNextRatio = mix(
        persistentRatio,
        max(persistentRatio, persistentTargetRatio),
        ${BRUSH_STROKE_PERSISTENCE_PROXIMAL_RATE}
      );
      let persistentNextMajor = persistentAreaRadius * sqrt(persistentNextRatio);
      let persistentNextMinor = persistentAreaRadius / sqrt(persistentNextRatio);
      nextScale = select(
        vec2<f32>(persistentNextMinor, persistentNextMajor),
        vec2<f32>(persistentNextMajor, persistentNextMinor),
        persistentMajorIsX
      );
    }
  }
  if (cfg(40u) > 0.5 && cfg(40u) < 1.5 && cfg(92u) > 0.5) {
    let longAxisIsX = nextScale.x >= nextScale.y;
    let vertical = cfg(92u) < 1.5;
    let targetTheta = select(
      select(1.57079632679, 0.0, longAxisIsX),
      select(0.0, 1.57079632679, longAxisIsX),
      vertical
    );
    let tolerance = clamp(cfg(127u), 0.0, 1.57079632679);
    if (tolerance <= 0.0) {
      nextTheta = targetTheta;
    } else if (tolerance < 1.57079632679) {
      let offset = nextTheta - targetTheta;
      let delta = offset - floor((offset + 1.57079632679) / 3.14159265359) * 3.14159265359;
      nextTheta += clamp(delta, -tolerance, tolerance) - delta;
    }
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
  xy[g].center = nextCenter;
  xy[g].rawDepth = nextVirtualDepthRaw;
  xy[g].depthGradient = accumulatedDepthGradient;
  let layerTag = floor(t.w);
  var layerOrder = clamp(min(fract(t.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  let previousLayerOrder = layerOrder;
  if (cfg(45u) > 0.5 && cfg(54u) > 0.5) {
    let meanError = errorSum / max(observed, 1.0);
    let meanInfluence = influenceSum / max(observed, 1.0);
    let stableBias = (hash_unit(f32(g) * 0.754877666) - 0.5) * 0.02;
    let targetLayer = clamp(0.5 + meanInfluence * 0.35 - meanError * 0.8 + stableBias, 0.0, 1.0);
    layerOrder = mix(layerOrder, targetLayer, clamp(cfg(53u), 0.0, 1.0));
  }
  let trainedLayerUpdate = color_guarded_trained_layer(
    nextCenter, nextScale, nextTheta, previousLayerOrder, layerOrder, nextColor
  );
  layerOrder = trainedLayerUpdate.order;
  nextColor = trainedLayerUpdate.color;
  if (trainedLayerUpdate.resetColorAdam > 0.5) {
    opt.mColor = vec4<f32>(0.0);
    opt.vColor = vec4<f32>(0.0);
  }
  // Never expose an unverified hidden row merely because it is small. The
  // post-growth contribution compactor must see those rows before any size-
  // based surface reassignment can make them visible.
  let surfaceLayerUpdate = size_sorted_surface_layer(
    g, nextCenter, nextScale, nextTheta, layerOrder, nextColor, observed, influenceSum
  );
  layerOrder = surfaceLayerUpdate.order;
  nextColor = surfaceLayerUpdate.color;
  if (surfaceLayerUpdate.resetColorAdam > 0.5) {
    opt.mColor = vec4<f32>(0.0);
    opt.vColor = vec4<f32>(0.0);
  }
  let packedLayer = layerTag + select(0.0, layerOrder * ${LAYER_CODE_RANGE}, cfg(45u) > 0.5);
  transform[g] = vec4<f32>(nextScale, nextTheta, packedLayer);
  var outputOpacity = clamp(nextOpacity, 0.005, 0.995);
  if (cfg(85u) > 0.5 && cfg(90u) < 0.5) {
    outputOpacity = cfg(88u);
  } else if (cfg(85u) > 0.5 && cfg(90u) > 0.5) {
    outputOpacity = clamp(outputOpacity, cfg(88u), cfg(118u));
  }
  color[g] = vec4<f32>(nextColor, outputOpacity);
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
  let center = xy[g].center;
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
  let optimizerScaleGradient = ${inverseScaleOptimizationEnabled
    ? "-gradLogScale * baseScale"
    : "gradLogScale"};
  let gradGeom = vec4<f32>(gradCenter, optimizerScaleGradient) * normalizer;
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
  let phaseOneShapeBoost = max(cfg(79u), 1.0);
  let positionLr = cfg(10u) * lrScale * spatialGate;
  let colorLr = cfg(11u) * lrScale * colorGate;
  let opacityLr = cfg(12u) * lrScale * spatialGate;
  let scaleLr = cfg(13u) * lrScale * spatialGate * phaseOneShapeBoost;
  let rotationLr = cfg(14u) * lrScale * spatialGate * phaseOneShapeBoost;
  var centerUpdate = geomAdam.xy * positionLr;
  var nextCenter = center - centerUpdate;
  var nextScale = ${inverseScaleOptimizationEnabled
    ? "1.0 / max((1.0 / baseScale) * max(vec2<f32>(0.5), vec2<f32>(1.0) - geomAdam.zw * scaleLr), vec2<f32>(0.0001))"
    : "exp(log(baseScale) - geomAdam.zw * scaleLr)"};
  var nextColor = clamp(rgba.rgb - colorAdam.rgb * colorLr, vec3<f32>(0.0), vec3<f32>(1.0));
  if (cfg(93u) > 0.5 && step < cfg(100u)) {
    nextColor = vec3<f32>(dot(nextColor, vec3<f32>(1.0 / 3.0)));
  }
  let currentLogit = log(clamp(rgba.a, 0.005, 0.995) / (1.0 - clamp(rgba.a, 0.005, 0.995)));
  var nextOpacity = select(
    1.0 / (1.0 + exp(-(currentLogit - colorAdam.w * opacityLr))),
    cfg(88u),
    cfg(85u) > 0.5 && cfg(90u) < 0.5
  );
  if (cfg(85u) > 0.5 && cfg(90u) > 0.5) {
    nextOpacity = clamp(nextOpacity, cfg(88u), cfg(118u));
  }
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
  let minScale = max(${MIN_SPLAT_SCALE}, cfg(80u));
  nextTheta = clamp(nextTheta, -3.14159265, 3.14159265);
  nextScale = max(nextScale, vec2<f32>(minScale));
  if (cfg(122u) > 0.5 && cfg(120u) > minScale) {
    let geometricScale = sqrt(max(minScale * minScale, nextScale.x * nextScale.y));
    if (geometricScale < cfg(120u)) {
      let correction = pow(
        cfg(120u) / max(geometricScale, minScale),
        clamp(cfg(121u), 0.0, 1.0)
      );
      nextScale *= correction;
    }
  }
  let phaseOneProgress = clamp(step / max(cfg(39u), 1.0), 0.0, 1.0);
  let phaseMaxPlanarScale = mix(max(cfg(62u), ${PHASE_ONE_MAX_PLANAR_SCALE}), max(cfg(62u), minScale), phaseOneProgress);
  nextScale = min(nextScale, vec2<f32>(phaseMaxPlanarScale));
  let major = max(nextScale.x, nextScale.y);
  let minor = max(minScale, min(nextScale.x, nextScale.y));
  let baseMaxAnisotropy = max(cfg(17u), 1.0);
  let surfaceMaxAnisotropy = max(cfg(51u), 1.0);
  let detailTagged = floor(t.w) >= 2.0;
  var maxAnisotropy = select(min(baseMaxAnisotropy, surfaceMaxAnisotropy), baseMaxAnisotropy, detailTagged);
  if (cfg(40u) > 0.5 && cfg(40u) < 1.5) {
    maxAnisotropy = min(maxAnisotropy, max(cfg(91u), 1.0));
  }
  if (major / minor > maxAnisotropy) {
    let capped = minor * maxAnisotropy;
    nextScale = select(vec2<f32>(minor, capped), vec2<f32>(capped, minor), nextScale.x >= nextScale.y);
  }
  if (cfg(40u) > 3.5) {
    let brushMajorIsX = nextScale.x >= nextScale.y;
    let brushRatio = max(nextScale.x, nextScale.y) / max(minScale, min(nextScale.x, nextScale.y));
    let brushAreaRadius = sqrt(max(minScale * minScale, nextScale.x * nextScale.y));
    let brushMinorFeasibleRatio = brushAreaRadius * brushAreaRadius / (minScale * minScale);
    let brushMajorFeasibleRatio = phaseMaxPlanarScale * phaseMaxPlanarScale /
      max(brushAreaRadius * brushAreaRadius, minScale * minScale);
    let brushTargetRatio = min(
      max(1.0, cfg(119u)),
      min(maxAnisotropy, min(brushMinorFeasibleRatio, brushMajorFeasibleRatio))
    );
    if (brushRatio < brushTargetRatio) {
      let brushNextMajor = brushAreaRadius * sqrt(brushTargetRatio);
      let brushNextMinor = brushAreaRadius / sqrt(brushTargetRatio);
      nextScale = select(
        vec2<f32>(brushNextMinor, brushNextMajor),
        vec2<f32>(brushNextMajor, brushNextMinor),
        brushMajorIsX
      );
    }
  }
  if (cfg(40u) > 0.5 && cfg(40u) < 1.5) {
    let rectangleMajorIsX = nextScale.x >= nextScale.y;
    let rectangleMajor = max(nextScale.x, nextScale.y);
    let rectangleMinor = max(minScale, min(nextScale.x, nextScale.y));
    let rectangleRatio = rectangleMajor / rectangleMinor;
    let rectangleAreaRadius = sqrt(max(minScale * minScale, nextScale.x * nextScale.y));
    let rectangleMinorFeasibleRatio = rectangleAreaRadius * rectangleAreaRadius / (minScale * minScale);
    let rectangleMajorFeasibleRatio = phaseMaxPlanarScale * phaseMaxPlanarScale /
      max(rectangleAreaRadius * rectangleAreaRadius, minScale * minScale);
    let rectangleTargetRatio = min(
      max(1.0, cfg(115u)),
      min(maxAnisotropy, min(rectangleMinorFeasibleRatio, rectangleMajorFeasibleRatio))
    );
    if (rectangleRatio < rectangleTargetRatio) {
      let rectangleNextMajor = rectangleAreaRadius * sqrt(rectangleTargetRatio);
      let rectangleNextMinor = rectangleAreaRadius / sqrt(rectangleTargetRatio);
      nextScale = select(
        vec2<f32>(rectangleNextMinor, rectangleNextMajor),
        vec2<f32>(rectangleNextMajor, rectangleNextMinor),
        rectangleMajorIsX
      );
    }
  }
  if (cfg(40u) > 3.5 && cfg(112u) > 0.5) {
    let persistentMajorIsX = nextScale.x >= nextScale.y;
    let persistentMajor = max(nextScale.x, nextScale.y);
    let persistentMinor = max(minScale, min(nextScale.x, nextScale.y));
    let persistentRatio = persistentMajor / persistentMinor;
    let persistentAccent = floor(t.w) >= 2.0;
    let persistentDirectional =
      persistentAccent || persistentRatio >= ${BRUSH_STROKE_PERSISTENCE_DIRECTIONAL_RATIO};
    if (persistentDirectional) {
      let persistentAreaRadius = sqrt(max(minScale * minScale, nextScale.x * nextScale.y));
      let persistentFeasibleRatio = max(1.0, persistentAreaRadius * persistentAreaRadius / (minScale * minScale));
      let persistentTargetRatio = min(
        min(maxAnisotropy, persistentFeasibleRatio),
        select(
          max(1.0, cfg(113u)),
          max(1.0, cfg(114u)),
          persistentAccent
        )
      );
      let persistentNextRatio = mix(
        persistentRatio,
        max(persistentRatio, persistentTargetRatio),
        ${BRUSH_STROKE_PERSISTENCE_PROXIMAL_RATE}
      );
      let persistentNextMajor = persistentAreaRadius * sqrt(persistentNextRatio);
      let persistentNextMinor = persistentAreaRadius / sqrt(persistentNextRatio);
      nextScale = select(
        vec2<f32>(persistentNextMinor, persistentNextMajor),
        vec2<f32>(persistentNextMajor, persistentNextMinor),
        persistentMajorIsX
      );
    }
  }
  if (cfg(40u) > 0.5 && cfg(40u) < 1.5 && cfg(92u) > 0.5) {
    let longAxisIsX = nextScale.x >= nextScale.y;
    let vertical = cfg(92u) < 1.5;
    let targetTheta = select(
      select(1.57079632679, 0.0, longAxisIsX),
      select(0.0, 1.57079632679, longAxisIsX),
      vertical
    );
    let tolerance = clamp(cfg(127u), 0.0, 1.57079632679);
    if (tolerance <= 0.0) {
      nextTheta = targetTheta;
    } else if (tolerance < 1.57079632679) {
      let offset = nextTheta - targetTheta;
      let delta = offset - floor((offset + 1.57079632679) / 3.14159265359) * 3.14159265359;
      nextTheta += clamp(delta, -tolerance, tolerance) - delta;
    }
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
  xy[g].center = nextCenter;
  let layerTag = floor(t.w);
  var layerOrder = clamp(min(fract(t.w), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
  let previousLayerOrder = layerOrder;
  if (cfg(45u) > 0.5 && cfg(54u) > 0.5) {
    let meanError = errorSum / max(observed, 1.0);
    let meanInfluence = influenceSum / max(observed, 1.0);
    let stableBias = (hash_unit(f32(g) * 0.754877666) - 0.5) * 0.02;
    let targetLayer = clamp(0.5 + meanInfluence * 0.35 - meanError * 0.8 + stableBias, 0.0, 1.0);
    layerOrder = mix(layerOrder, targetLayer, clamp(cfg(53u), 0.0, 1.0));
  }
  let trainedLayerUpdate = color_guarded_trained_layer(
    nextCenter, nextScale, nextTheta, previousLayerOrder, layerOrder, nextColor
  );
  layerOrder = trainedLayerUpdate.order;
  nextColor = trainedLayerUpdate.color;
  if (trainedLayerUpdate.resetColorAdam > 0.5) {
    opt.mColor = vec4<f32>(0.0);
    opt.vColor = vec4<f32>(0.0);
  }
  // Match the exact optimizer: size ordering is complete inside the currently
  // verified contributor cohort, while hidden rows retain their old layer.
  let surfaceLayerUpdate = size_sorted_surface_layer(
    g, nextCenter, nextScale, nextTheta, layerOrder, nextColor, observed, influenceSum
  );
  layerOrder = surfaceLayerUpdate.order;
  nextColor = surfaceLayerUpdate.color;
  if (surfaceLayerUpdate.resetColorAdam > 0.5) {
    opt.mColor = vec4<f32>(0.0);
    opt.vColor = vec4<f32>(0.0);
  }
  let packedLayer = layerTag + select(0.0, layerOrder * ${LAYER_CODE_RANGE}, cfg(45u) > 0.5);
  transform[g] = vec4<f32>(nextScale, nextTheta, packedLayer);
  var outputOpacity = clamp(nextOpacity, 0.005, 0.995);
  if (cfg(85u) > 0.5 && cfg(90u) < 0.5) {
    outputOpacity = cfg(88u);
  } else if (cfg(85u) > 0.5 && cfg(90u) > 0.5) {
    outputOpacity = clamp(outputOpacity, cfg(88u), cfg(118u));
  }
  color[g] = vec4<f32>(nextColor, outputOpacity);
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
  var center = xy[g].center;
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
  let gradientNormalization = max(cfg(63u), 0.0001);
  let geomSum = gradientNormalization * vec4<f32>(exact_gradient(base), exact_gradient(base + 1u), exact_gradient(base + 2u), exact_gradient(base + 3u));
  let appearanceSum = gradientNormalization * vec4<f32>(exact_gradient(base + 4u), exact_gradient(base + 5u), exact_gradient(base + 6u), exact_gradient(base + 7u));
  let miscSum = gradientNormalization * vec4<f32>(exact_gradient(base + 8u), exact_gradient(base + 9u), exact_gradient(base + 10u), exact_gradient(base + 11u));
  let densitySum = gradientNormalization * vec4<f32>(exact_gradient(base + 12u), exact_gradient(base + 13u), exact_gradient(base + 14u), exact_gradient(base + 15u));
  let center = xy[g].center;
  let t = transform[g];
  let rgba = color[g];
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  apply_optimizer(g, center, t, rgba, baseScale, cos(t.z), sin(t.z), geomSum, appearanceSum, miscSum, densitySum);
}`;

    // END BYTE-STABLE TRAINING SHADER DECLARATIONS
    return {
      renderShader,
      ssimShader,
      lossGradientShader,
      exactBackwardShader,
      segmentedReferenceShader,
      segmentedGradientReduceShader,
      exactBackwardTelemetryShader,
      virtualOrderPenaltyShader,
      brushLocalColorFlowShader,
      optimizerShader,
    };
  }

  global.Image2SplatPaintTrainingPipelineShaders = Object.freeze({ create });
})(globalThis);
