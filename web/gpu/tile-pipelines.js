class WebGpuTilePipelines {
  async ensureTilePipelines() {
    if (this.tileCountPipeline && this.tilePrefixPipeline && this.tileFillPipeline && this.tileSortPipeline) return;
    const exactTileIntersectionEnabled = this.exactTileIntersectionEnabled;
    const opacityAwareSupportMode = this.opacityAwareSupportMode;
    const opacityAwareSupportEnabled = opacityAwareSupportMode !== "off";
    const aggressiveOpacitySupport = opacityAwareSupportMode === "aggressive";
    const opacitySupportDeclaration = opacityAwareSupportEnabled
      ? "@group(0) @binding(8) var<storage, read> color: array<vec4<f32>>;"
      : "";
    const opacitySupportFunction = aggressiveOpacitySupport
      ? `
fn opacity_support_q(g: u32) -> f32 {
  let alphaCutoff = 0.0039215686;
  let peakAlpha = max(color[g].a, alphaCutoff);
  return clamp(-2.0 * log(alphaCutoff / peakAlpha), 0.0, 16.0);
}
`
      : "";
    const opacitySupportEarlyExit = opacityAwareSupportEnabled
      ? " || (!virtual_tilt_enabled() && color[g].a < 0.0039215686)"
      : "";
    const supportSigmaExpression = aggressiveOpacitySupport
      ? "select(4.0, sqrt(opacity_support_q(g)), !virtual_tilt_enabled())"
      : String(RENDER_SIGMA);
    const supportQExpression = aggressiveOpacitySupport
      ? "select(16.0, opacity_support_q(g), !virtual_tilt_enabled()) + 0.0001"
      : "16.0001";
    const paintSupportFunctions = this.shapeAwarePaintCullingEnabled
      ? `
fn finite_paint_support(g: u32) -> bool {
  return cfg(40u) > 0.5;
}

fn finite_paint_extent(g: u32) -> f32 {
  if (cfg(40u) > 3.5) {
    let oilFeatherMargin = pow(1.0 + clamp(cfg(41u), 0.01, 0.49), 0.25);
    return ${LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT} * oilFeatherMargin;
  }
  let widthRatios = rectangle_effective_width_ratios(
    cfg(96u),
    cfg(98u),
    transform[g].w,
    cfg(97u)
  );
  let rectangleFeatherMargin = 1.0 + 1.25 * clamp(cfg(41u), 0.01, 0.49);
  return ${RECTANGLE_KERNEL_EXTENT} *
    rectangle_area_compensation(widthRatios, cfg(97u)) *
    rectangleFeatherMargin;
}
`
      : `
fn finite_paint_support(g: u32) -> bool { return false; }
fn finite_paint_extent(g: u32) -> f32 { return 1.0; }
`;
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
  let sourceT = transform[g];
  let t = sourceT;
  let baseScale = max(t.xy, vec2<f32>(0.0001));
  let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
  let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
  let useEwa = cfg(26u) > 0.5;
  var sampleScale = select(effective, baseScale, useEwa);
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
  var footprintCenter = xy[g].center;
  var footprintTheta = t.z;
  var footprintQ = ${supportQExpression};
  if (finite_paint_support(g)) {
    sampleScale *= finite_paint_extent(g);
    footprintQ = 2.0001;
  }
  if (virtual_tilt_enabled()) {
    if (!camera_covariance_3d_enabled()) { return true; }
    let layerZ = virtual_pass_layer_depth(t.w, xy[g].rawDepth);
    let projected = project_planar_gaussian(xy[g].center, layerZ, t);
    footprintCenter = projected.center;
    footprintTheta = projected.theta;
    sampleScale = max(projected.scale, vec2<f32>(0.0001));
    footprintQ = 16.0001;
  }
  let cTheta = cos(footprintTheta);
  let sTheta = sin(footprintTheta);
  let invScale2 = 1.0 / (sampleScale * sampleScale);
  let a = cTheta * cTheta * invScale2.x + sTheta * sTheta * invScale2.y;
  let b = cTheta * sTheta * (invScale2.x - invScale2.y);
  let c = sTheta * sTheta * invScale2.x + cTheta * cTheta * invScale2.y;
  let minimumQ = minimum_quadratic_on_rect(
    rectMin - footprintCenter,
    rectMax - footprintCenter,
    a,
    b,
    c
  );
  return minimumQ <= footprintQ;
}
`
      : "";
    const exactTileIntersectionGuard = exactTileIntersectionEnabled
      ? this.projectedVirtualExactCullingEnabled
        ? "if (!tile_intersects_footprint(g, tx, ty)) { continue; }"
        : "if (!virtual_tilt_enabled() && !tile_intersects_footprint(g, tx, ty)) { continue; }"
      : "";
    const sortTilesFunction = `
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
}
`;
    const shader = `
struct Config { values: array<vec4<f32>, 32>, };
@group(0) @binding(0) var<uniform> config: Config;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tileCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> tileOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> tileCursors: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> tileIndices: array<u32>;
@group(0) @binding(7) var<storage, read_write> control: array<atomic<u32>>;
${opacitySupportDeclaration}

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

${VIRTUAL_TILT_WGSL}
${RECTANGLE_TRAPEZOID_WGSL}
${ILLUSTRATIVE_OIL_WGSL}
${opacitySupportFunction}
${paintSupportFunctions}

fn tile_bounds(g: u32) -> vec4<u32> {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let sourceT = transform[g];
  let center = xy[g].center;
  let t = sourceT;
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
  let finitePaint = finite_paint_support(g);
  let boundsScale = select(effective, effective * finite_paint_extent(g), finitePaint);
  let boundsSigma = select(${supportSigmaExpression}, 1.0, finitePaint);
  let radius = vec2<f32>(
    boundsSigma * (abs(c) * boundsScale.x + abs(s) * boundsScale.y),
    boundsSigma * (abs(s) * boundsScale.x + abs(c) * boundsScale.y)
  ) + pixelPadding;
  var minNorm = center - radius;
  var maxNorm = center + radius;
  if (virtual_tilt_enabled()) {
    let layerZ = virtual_pass_layer_depth(t.w, xy[g].rawDepth);
    if (camera_covariance_3d_enabled()) {
      let projected = project_planar_gaussian(center, layerZ, t);
      let pc = cos(projected.theta);
      let ps = sin(projected.theta);
      let projectedRadius = vec2<f32>(
        ${RENDER_SIGMA} * (abs(pc) * projected.scale.x + abs(ps) * projected.scale.y),
        ${RENDER_SIGMA} * (abs(ps) * projected.scale.x + abs(pc) * projected.scale.y)
      ) + pixelPadding;
      minNorm = projected.center - projectedRadius;
      maxNorm = projected.center + projectedRadius;
    } else {
      let p0 = virtual_project_point(center + vec2<f32>(-radius.x, -radius.y), layerZ).xy;
      let p1 = virtual_project_point(center + vec2<f32>( radius.x, -radius.y), layerZ).xy;
      let p2 = virtual_project_point(center + vec2<f32>(-radius.x,  radius.y), layerZ).xy;
      let p3 = virtual_project_point(center + vec2<f32>( radius.x,  radius.y), layerZ).xy;
      minNorm = min(min(p0, p1), min(p2, p3));
      maxNorm = max(max(p0, p1), max(p2, p3));
    }
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
  if (g >= u32(cfg(2u)) || transform[g].w < 0.5${opacitySupportEarlyExit}) { return; }
  let tileCols = (u32(cfg(0u)) + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  let bounds = tile_bounds(g);
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
  if (g >= u32(cfg(2u)) || transform[g].w < 0.5${opacitySupportEarlyExit}) { return; }
  let tileCols = (u32(cfg(0u)) + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
  let bounds = tile_bounds(g);
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
  let depthA = virtual_camera_depth(xy[a].center, transform[a].w, xy[a].rawDepth);
  let depthB = virtual_camera_depth(xy[b].center, transform[b].w, xy[b].rawDepth);
  if (abs(depthA - depthB) > 0.0000001) {
    // Heap-sort ascending by inverse depth, so the final list is front-to-back.
    return depthA > depthB;
  }
  return a < b;
}

${sortTilesFunction}`;
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

}

installWebGpuPreviewMethods(WebGpuTilePipelines.prototype, "WebGpuTilePipelines");
