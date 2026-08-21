(function installTilePipelineShaderFactory(global) {
  function create({
    opacitySupportDeclaration,
    opacitySupportFunction,
    paintSupportFunctions,
    supportSigmaExpression,
    exactTileIntersectionFunction,
    opacitySupportEarlyExit,
    exactTileIntersectionGuard,
    sortTilesFunction,
  }) {
    return `
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
  }

  global.Image2SplatPaintTilePipelineShader = Object.freeze({ create });
})(globalThis);
