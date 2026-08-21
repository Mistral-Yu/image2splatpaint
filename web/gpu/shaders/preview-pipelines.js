(function installPreviewShaderFactories(global) {
  function renderPreview() {
    return `
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
  useGaussLegendre: f32,
  alphaBgR: f32,
  alphaBgG: f32,
  alphaBgB: f32,
  kernelFalloff: f32,
  shapeMode: f32,
  layerAware: f32,
  layerCount: f32,
  padPaint0: f32,
  padPaint1: f32,
  rectangleTopRatio: f32,
  rectangleTopRatioMax: f32,
  rectangleShapeFlags: f32,
  opacityMultiplier: f32,
  splatScaleMultiplier: f32,
  localAspectRatio: f32,
  reservedPreview0: f32,
  reservedPreview1: f32,
  reservedPreview2: f32,
  pad2: f32,
  centerOpacityMin: f32,
  centerOpacityMax: f32,
  pad3: f32,
  pad4: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct SplatPosition { center: vec2f, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
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

${RECTANGLE_TRAPEZOID_WGSL}
${ILLUSTRATIVE_OIL_WGSL}

fn preview_kernel(
  d: vec2f,
  c: f32,
  s: f32,
  scale: vec2f,
  packedTag: f32,
  taperAmount: f32,
  worldPoint: vec2f
) -> f32 {
  if (uniforms.shapeMode > 4.5) {
    let r = vec2f(c * d.x + s * d.y, -s * d.x + c * d.y);
    let normalized = r / max(scale * ${LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT}, vec2f(0.0001));
    let family = illustrative_oil_family(scale, packedTag);
    let oilKernel = illustrative_oil_kernel_sample(
      normalized,
      scale.x >= scale.y,
      ${LAYERED_OPAQUE_BRUSH_EDGE_SOFTNESS},
      family,
      taperAmount,
      uniforms.reservedPreview1 > 0.5,
      uniforms.reservedPreview2 > 0.5,
      uniforms.reservedPreview0,
      uniforms.pad2,
      uniforms.centerOpacityMin,
      uniforms.centerOpacityMax,
      uniforms.padPaint0,
      uniforms.padPaint1
    ).kernel;
    return oilKernel;
  }
  if (uniforms.shapeMode > 1.5) {
    let r = vec2f(c * d.x + s * d.y, -s * d.x + c * d.y);
    let widthRatios = rectangle_effective_width_ratios(
      uniforms.rectangleTopRatio,
      uniforms.rectangleTopRatioMax,
      packedTag,
      uniforms.rectangleShapeFlags
    );
    let areaCompensation =
      rectangle_area_compensation(widthRatios, uniforms.rectangleShapeFlags);
    let normalized = r / max(
      scale * ${RECTANGLE_KERNEL_EXTENT} * areaCompensation,
      vec2f(0.0001)
    );
    return rectangle_trapezoid_kernel_sample(
      normalized,
      ${RECTANGLE_EDGE_SOFTNESS},
      widthRatios.x,
      widthRatios.y,
      rectangle_flag_enabled(
        uniforms.rectangleShapeFlags,
        ${RECTANGLE_FLAG_ASYMMETRIC_SOFTNESS}u
      ),
      uniforms.reservedPreview0,
      uniforms.pad2,
      uniforms.centerOpacityMin,
      uniforms.centerOpacityMax
    ).kernel;
  }
  if (uniforms.shapeMode > 0.5) {
    let r = vec2f(c * d.x + s * d.y, -s * d.x + c * d.y);
    let local = abs(r / max(scale, vec2f(0.0001)));
    let radius = max(local.x, local.y);
    return select(0.0, exp(-0.5 * radius * radius), radius <= 4.0);
  }
  return gaussian_kernel(d, c, s, scale);
}

fn tile_offset(index: u32) -> u32 {
  return tileOffsets[index] & 0x7fffffffu;
}

fn preview_layer(packed: f32, layerCount: u32) -> u32 {
  let order = clamp(min(fract(packed), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 0.999999);
  return min(layerCount - 1u, u32(floor(order * f32(layerCount))));
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
  let layerAware = uniforms.layerAware > 0.5;
  let accumulationLayerCount = clamp(u32(round(uniforms.layerCount)), ${MIN_DISCRETE_LAYER_COUNT}u, ${MAX_DISCRETE_LAYER_COUNT}u);
  var activeLayer = accumulationLayerCount;
  var layerRgb = vec3f(0.0);
  var layerTransmittance = 1.0;
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
    let d = p - xy[i].center;
    let t = transform[i];
    if (t.w < 0.5) {
      cursor = cursor + 1u;
      if (transmittance * select(1.0, layerTransmittance, layerAware) < 0.0001) { break; }
      continue;
    }
    let c = cos(t.z);
    let sT = sin(t.z);
    let aspectStretch = sqrt(max(uniforms.localAspectRatio, 0.000001));
    let baseScale = max(
      t.xy *
        max(uniforms.splatScaleMultiplier, 0.0) *
        vec2f(aspectStretch, 1.0 / aspectStretch),
      vec2f(0.0001)
    );
    let mip = mip_weight_scale(baseScale);
    let useEwa = uniforms.useEwa > 0.5;
    var kernel = preview_kernel(d, c, sT, mip.xy, t.w, xy[i].rawDepth, p);
    var compensation = select(mip.z, 1.0, uniforms.shapeMode > 0.5);
    if (useEwa) {
      let sampleOffset = select(0.5, 0.28867513459481287, uniforms.useGaussLegendre > 0.5);
      let ox = select(0.0, sampleOffset / (uniforms.sourceWidth - 1.0), uniforms.sourceWidth > 1.0);
      let oy = select(0.0, sampleOffset / (uniforms.sourceHeight - 1.0), uniforms.sourceHeight > 1.0);
      let clampToImage = viewScale.x >= 0.999999 && viewScale.y >= 0.999999;
      let p00 = select(p + vec2f(-ox, -oy), clamp(p + vec2f(-ox, -oy), vec2f(-1.0), vec2f(1.0)), clampToImage);
      let p10 = select(p + vec2f( ox, -oy), clamp(p + vec2f( ox, -oy), vec2f(-1.0), vec2f(1.0)), clampToImage);
      let p01 = select(p + vec2f(-ox,  oy), clamp(p + vec2f(-ox,  oy), vec2f(-1.0), vec2f(1.0)), clampToImage);
      let p11 = select(p + vec2f( ox,  oy), clamp(p + vec2f( ox,  oy), vec2f(-1.0), vec2f(1.0)), clampToImage);
      kernel = 0.25 * (
        preview_kernel(p00 - xy[i].center, c, sT, baseScale, t.w, xy[i].rawDepth, p00) +
        preview_kernel(p10 - xy[i].center, c, sT, baseScale, t.w, xy[i].rawDepth, p10) +
        preview_kernel(p01 - xy[i].center, c, sT, baseScale, t.w, xy[i].rawDepth, p01) +
        preview_kernel(p11 - xy[i].center, c, sT, baseScale, t.w, xy[i].rawDepth, p11)
      );
      compensation = 1.0;
    }
    // Keep the trained 4σ footprint. Falloff only changes the profile inside
    // that footprint so this preview control cannot create new out-of-frame tails.
    if (kernel >= 0.0003354626) {
      let kernelFalloff = clamp(uniforms.kernelFalloff, 0.0, 2.0);
      if (kernelFalloff != 1.0) {
        kernel = pow(kernel, kernelFalloff);
      }
      // Match the optimizer contract exactly. Both opaque-paint algorithms
      // learn opacity above a floor and composite with the shared 0.99 cap.
      let alphaLimit = 0.99;
      let alpha = clamp(
        kernel * color[i].a * max(uniforms.opacityMultiplier, 0.0) * compensation,
        0.0,
        alphaLimit
      );
      if (alpha >= 0.0039215686) {
        let pigment = color[i].rgb;
        let layer = preview_layer(t.w, accumulationLayerCount);
        if (layerAware && activeLayer != accumulationLayerCount && layer != activeLayer) {
          rgb += transmittance * layerRgb;
          transmittance *= layerTransmittance;
          layerRgb = vec3f(0.0);
          layerTransmittance = 1.0;
        }
        if (layerAware) {
          activeLayer = layer;
          layerRgb += layerTransmittance * alpha * pigment;
          layerTransmittance *= 1.0 - alpha;
        } else {
          rgb += transmittance * alpha * pigment;
          transmittance *= 1.0 - alpha;
        }
      }
    }
    cursor = cursor + 1u;
    if (transmittance * select(1.0, layerTransmittance, layerAware) < 0.0001) { break; }
  }
  if (layerAware && activeLayer != accumulationLayerCount) {
    rgb += transmittance * layerRgb;
    transmittance *= layerTransmittance;
  }
  rgb += transmittance * vec3f(uniforms.alphaBgR, uniforms.alphaBgG, uniforms.alphaBgB);
  return vec4f(rgb, 1.0 - transmittance);
}`;
    if (this.pipeline) return;
  }

  function adaptiveGridInitialization() {
    return `
struct InitConfig {
  image: vec4<f32>,
  sampling: vec4<f32>,
};
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(0) var<uniform> config: InitConfig;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read_write> xy: array<SplatPosition>;
@group(0) @binding(3) var<storage, read_write> color: array<vec4<f32>>;

fn hash_unit(seed: f32) -> f32 {
  return fract(sin(seed * 12.9898 + 78.233) * 43758.5453);
}

fn target_at(p: vec2<f32>) -> vec3<f32> {
  let width = max(1u, u32(config.image.x));
  let height = max(1u, u32(config.image.y));
  let px = min(width - 1u, u32(clamp(round((p.x * 0.5 + 0.5) * f32(width - 1u)), 0.0, f32(width - 1u))));
  let py = min(height - 1u, u32(clamp(round((p.y * 0.5 + 0.5) * f32(height - 1u)), 0.0, f32(height - 1u))));
  let index = (py * width + px) * 3u;
  return vec3<f32>(targetRgb[index], targetRgb[index + 1u], targetRgb[index + 2u]);
}

fn srgb_decode_channel(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

fn srgb_encode_channel(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
}

fn neutral_srgb_preserving_lab_l(rgb: vec3<f32>) -> vec3<f32> {
  let linear = vec3<f32>(
    srgb_decode_channel(rgb.r),
    srgb_decode_channel(rgb.g),
    srgb_decode_channel(rgb.b)
  );
  let luma = dot(linear, vec3<f32>(0.2126729, 0.7151522, 0.0721750));
  return vec3<f32>(srgb_encode_channel(luma));
}

fn luminance(value: vec3<f32>) -> f32 {
  return dot(value, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn structure_score(p: vec2<f32>) -> f32 {
  let width = max(2.0, config.image.x);
  let height = max(2.0, config.image.y);
  let fine = vec2<f32>(2.0 / (width - 1.0), 2.0 / (height - 1.0));
  let coarse = fine * 4.0;
  let gxFine = luminance(target_at(p + vec2<f32>(fine.x, 0.0))) - luminance(target_at(p - vec2<f32>(fine.x, 0.0)));
  let gyFine = luminance(target_at(p + vec2<f32>(0.0, fine.y))) - luminance(target_at(p - vec2<f32>(0.0, fine.y)));
  let gxCoarse = luminance(target_at(p + vec2<f32>(coarse.x, 0.0))) - luminance(target_at(p - vec2<f32>(coarse.x, 0.0)));
  let gyCoarse = luminance(target_at(p + vec2<f32>(0.0, coarse.y))) - luminance(target_at(p - vec2<f32>(0.0, coarse.y)));
  return length(vec2<f32>(gxFine, gyFine)) * 0.75 + length(vec2<f32>(gxCoarse, gyCoarse)) * 0.25;
}

@compute @workgroup_size(64)
fn initialize_adaptive_grid(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let count = u32(config.image.z);
  if (index >= count) { return; }
  if (hash_unit(f32(index) * 19.19 + 0.37) >= config.image.w) { return; }

  let margin = config.sampling.y;
  let candidateCount = max(1u, u32(config.sampling.x));
  var best = xy[index].center;
  var bestScore = structure_score(best);
  for (var candidate = 0u; candidate < candidateCount; candidate = candidate + 1u) {
    let seed = f32(index) * 37.71 + f32(candidate) * 101.13;
    let probe = vec2<f32>(
      (hash_unit(seed + 3.1) * 2.0 - 1.0) * margin,
      (hash_unit(seed + 7.9) * 2.0 - 1.0) * margin
    );
    let score = structure_score(probe);
    if (score > bestScore) {
      best = probe;
      bestScore = score;
    }
  }
  xy[index].center = best;
  let targetColor = target_at(best);
  let teacher = select(targetColor, neutral_srgb_preserving_lab_l(targetColor), config.sampling.z > 0.5);
  color[index] = vec4<f32>(teacher, color[index].a);
}`;
  }

  function discreteLayers() {
    return `
struct Config { values: array<vec4<f32>, 32>, };
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
};
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
@group(0) @binding(2) var<storage, read_write> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> tileIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(6) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(7) var<storage, read_write> color: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> optimizerState: array<AdamState>;
@group(0) @binding(9) var<storage, read> contributionStats: array<vec4<f32>>;

fn cfg(index: u32) -> f32 { return config.values[index / 4u][index % 4u]; }
fn packed_layer(packed: f32) -> f32 {
  return clamp(min(fract(packed), ${LAYER_CODE_RANGE}) / ${LAYER_CODE_RANGE}, 0.0, 1.0);
}
fn quantized_layer_id(packed: f32, layerCount: u32) -> u32 {
  return min(layerCount - 1u, u32(floor(min(packed_layer(packed), 0.999999) * f32(layerCount))));
}
fn in_layer_order(packed: f32, layerCount: u32) -> f32 {
  return fract(min(packed_layer(packed), 0.999999) * f32(layerCount));
}
fn footprint_radius(t: vec4<f32>) -> vec2<f32> {
  let c = abs(cos(t.z));
  let s = abs(sin(t.z));
  return ${RENDER_SIGMA} * vec2<f32>(c * t.x + s * t.y, s * t.x + c * t.y);
}
fn overlap_score(centerA: vec2<f32>, tA: vec4<f32>, centerB: vec2<f32>, tB: vec4<f32>) -> f32 {
  let radiusA = footprint_radius(tA);
  let radiusB = footprint_radius(tB);
  let overlap = max(vec2<f32>(0.0), radiusA + radiusB - abs(centerA - centerB));
  let intersection = overlap.x * overlap.y;
  let minimumArea = max(0.00000001, min(4.0 * radiusA.x * radiusA.y, 4.0 * radiusB.x * radiusB.y));
  return clamp(intersection / minimumArea, 0.0, 1.0);
}

fn srgb_decode_channel(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

fn srgb_encode_channel(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
}

fn target_at(center: vec2<f32>, width: u32, height: u32) -> vec3<f32> {
  let safeCenter = clamp(center, vec2<f32>(-1.0), vec2<f32>(1.0));
  let pixel = vec2<u32>(clamp(
    round((safeCenter * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))),
    vec2<f32>(0.0),
    vec2<f32>(f32(width - 1u), f32(height - 1u))
  ));
  let offset = (pixel.y * width + pixel.x) * 3u;
  let rgb = vec3<f32>(targetRgb[offset], targetRgb[offset + 1u], targetRgb[offset + 2u]);
  let linear = vec3<f32>(
    srgb_decode_channel(rgb.r),
    srgb_decode_channel(rgb.g),
    srgb_decode_channel(rgb.b)
  );
  let luma = dot(linear, vec3<f32>(0.2126729, 0.7151522, 0.0721750));
  let neutral = vec3<f32>(srgb_encode_channel(luma));
  let underpaintingActive = cfg(93u) > 0.5 && cfg(4u) < cfg(100u);
  return select(rgb, neutral, underpaintingActive);
}

@compute @workgroup_size(64)
fn assign_layers(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  let count = u32(cfg(2u));
  if (g >= count) { return; }
  let t = transform[g];
  if (t.w < 0.5) { return; }
  let layerCount = clamp(u32(round(cfg(82u))), ${MIN_DISCRETE_LAYER_COUNT}u, ${MAX_DISCRETE_LAYER_COUNT}u);
  let baseLayer = quantized_layer_id(t.w, layerCount);
  var costs: array<f32, ${MAX_DISCRETE_LAYER_COUNT}>;
  for (var layer = 0u; layer < ${MAX_DISCRETE_LAYER_COUNT}u; layer += 1u) { costs[layer] = 0.0; }
  let width = max(1u, u32(cfg(0u)));
  let height = max(1u, u32(cfg(1u)));
  let tileCols = (width + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  let tileRows = (height + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  let radius = footprint_radius(t);
  let minPixel = vec2<u32>(clamp(
    floor(((xy[g].center - radius) * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))),
    vec2<f32>(0.0),
    vec2<f32>(f32(width - 1u), f32(height - 1u))
  ));
  let maxPixel = vec2<u32>(clamp(
    ceil(((xy[g].center + radius) * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))),
    vec2<f32>(0.0),
    vec2<f32>(f32(width - 1u), f32(height - 1u))
  ));
  let minTile = minPixel / ${TILE_SIZE}u;
  let maxTile = min(maxPixel / ${TILE_SIZE}u, vec2<u32>(tileCols - 1u, tileRows - 1u));
  let capacity = arrayLength(&tileIndices);
  for (var tileY = minTile.y; tileY <= maxTile.y; tileY += 1u) {
    for (var tileX = minTile.x; tileX <= maxTile.x; tileX += 1u) {
      let tile = tileY * tileCols + tileX;
      let begin = min(tileOffsets[tile] & 0x7fffffffu, capacity);
      let end = min(tileOffsets[tile + 1u] & 0x7fffffffu, capacity);
      for (var cursor = begin; cursor < end; cursor += 1u) {
        let other = tileIndices[cursor];
        if (other >= g || other >= count) { continue; }
        let otherTransform = transform[other];
        if (otherTransform.w < 0.5) { continue; }
        let otherRadius = footprint_radius(otherTransform);
        let otherMinPixel = vec2<u32>(clamp(
          floor(((xy[other].center - otherRadius) * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))),
          vec2<f32>(0.0),
          vec2<f32>(f32(width - 1u), f32(height - 1u))
        ));
        let otherMinTile = otherMinPixel / ${TILE_SIZE}u;
        let representativeTile = max(minTile, otherMinTile);
        if (tileX != representativeTile.x || tileY != representativeTile.y) { continue; }
        let score = overlap_score(xy[g].center, t, xy[other].center, otherTransform);
        if (score <= 0.01) { continue; }
        costs[quantized_layer_id(otherTransform.w, layerCount)] += score;
      }
    }
  }
  var selected = baseLayer;
  var bestCost = costs[baseLayer];
  let targetColorForGate = target_at(xy[g].center, width, height);
  let targetColorError = dot(abs(color[g].rgb - targetColorForGate), vec3<f32>(0.33333334));
  let observation = contributionStats[g];
  let importance = contributionStats[u32(cfg(28u)) + g];
  let brushLayerAssignment = cfg(40u) > 3.5;
  if (brushLayerAssignment && bestCost > 0.25) {
    if (targetColorError > 0.075) {
      // Fit stale hidden RGB without promoting it in the same layer event.
      color[g] = vec4<f32>(targetColorForGate, color[g].a);
      optimizerState[g].mColor = vec4<f32>(0.0);
      optimizerState[g].vColor = vec4<f32>(0.0);
    } else {
      let moveRadius = clamp(u32(round(cfg(83u))), 1u, layerCount - 1u);
      let firstCandidate = max(baseLayer, moveRadius) - moveRadius;
      let lastCandidate = min(layerCount - 1u, baseLayer + moveRadius);
      for (var layer = firstCandidate; layer <= lastCandidate; layer += 1u) {
        let candidateCost = costs[layer] + 0.03 * abs(f32(layer) - f32(baseLayer));
        if (candidateCost + 0.000001 * f32((layer + g * 17u) % layerCount) < bestCost * 0.8) {
          selected = layer;
          bestCost = candidateCost;
        }
      }
    }
  }
  let quantized = (f32(selected) + in_layer_order(t.w, layerCount) * 0.999999) / f32(layerCount);
  let out = g * 4u;
  scratch[out] = t.x;
  scratch[out + 1u] = t.y;
  scratch[out + 2u] = t.z;
  scratch[out + 3u] = floor(t.w) + quantized * ${LAYER_CODE_RANGE};
}

@compute @workgroup_size(64)
fn commit_layers(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = id.x;
  if (g >= u32(cfg(2u))) { return; }
  let offset = g * 4u;
  transform[g] = vec4<f32>(scratch[offset], scratch[offset + 1u], scratch[offset + 2u], scratch[offset + 3u]);
}`;
  }

  function presentedStatePack() {
    return `
@group(0) @binding(0) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> packedRgba: array<u32>;

@compute @workgroup_size(256)
fn pack_presented_state(@builtin(global_invocation_id) id: vec3<u32>) {
  let pixel = id.x;
  if (pixel >= arrayLength(&pixelState)) { return; }
  packedRgba[pixel] = pack4x8unorm(pixelState[pixel]);
}`;
  }

  function presentCanvas() {
    return `
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
  }

  global.Image2SplatPaintPreviewShaders = Object.freeze({
    renderPreview,
    adaptiveGridInitialization,
    discreteLayers,
    presentedStatePack,
    presentCanvas,
  });
})(globalThis);
