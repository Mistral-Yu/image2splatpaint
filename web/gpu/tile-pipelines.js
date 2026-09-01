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
  if (cfg(40u) > 5.5) {
    // q <= 4 for the quartic soft-cell kernel, so each local axis reaches sqrt(2)x.
    return 2.12132034356;
  }
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
    const shader = this.createTileShader({
      opacitySupportDeclaration,
      opacitySupportFunction,
      paintSupportFunctions,
      supportSigmaExpression,
      exactTileIntersectionFunction,
      opacitySupportEarlyExit,
      exactTileIntersectionGuard,
      sortTilesFunction,
    });
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

registerWebGpuPreviewFeature(WebGpuTilePipelines.prototype, "WebGpuTilePipelines");
