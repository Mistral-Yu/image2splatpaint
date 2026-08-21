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
    this.residualTileBuildPipeline = null;
    this.residualTileBlockScanPipeline = null;
    this.residualTileBlockSumsPipeline = null;
    this.residualTileOffsetPipeline = null;
    this.growApplyPipeline = null;
    this.relocationSelectPipeline = null;
    this.relocationApplyPipeline = null;
    this.finalBrushRepairPipeline = null;
    this.phase45RegionTelemetryPipeline = null;
    this.phase45RegionFinalizePipeline = null;
    this.phase45DonorSafetyPipeline = null;
    this.structureAllocationCollectPipeline = null;
    this.optimizerResetPipeline = null;
    this.optimizerSourceResetPipeline = null;
    this.adaptiveGridInitializationPipeline = null;
    this.tileCountPipeline = null;
    this.tilePrefixPipeline = null;
    this.tileFillPipeline = null;
    this.tileSortPipeline = null;
    this.discreteLayerAssignPipeline = null;
    this.discreteLayerCommitPipeline = null;
    this.compactionParamPipeline = null;
    this.compactionStatePipeline = null;
    this.renderStatePipeline = null;
    this.tileCooperativeRenderPipeline = null;
    this.ssimHorizontalPipeline = null;
    this.ssimTilePipeline = null;
    this.ssimBackwardHorizontalPipeline = null;
    this.ssimBackwardVerticalPipeline = null;
    this.alphaSsimTilePipeline = null;
    this.renderGradientPipeline = null;
    this.parallelRenderGradientPipeline = null;
    this.lossGradientPipeline = null;
    this.exactAlphaBackwardPipeline = null;
    this.sourceDomainBackwardPipeline = null;
    this.virtualOrderPenaltyPipeline = null;
    this.brushLocalColorFlowPipeline = null;
    this.exactBackwardTelemetryPipeline = null;
    this.exactOptimizerPipeline = null;
    this.segmentedReferenceCountPipeline = null;
    this.segmentedReferencePrefixPipeline = null;
    this.segmentedReferenceFillPipeline = null;
    this.segmentedGradientReducePipeline = null;
    this.compiledSegmentedExactBackward = null;
    this.compiledFixedPointExactGradient = null;
    this.compiledInverseScaleOptimization = null;
    const performance = performanceVariants();
    this.quadExactBackwardEnabled = performance.quadExactBackward;
    this.exactTileIntersectionEnabled = performance.exactTileIntersection;
    this.subgroupExactBackwardEnabled = Boolean(profile.subgroupExactBackward && device.features.has("subgroups"));
    this.opacityAwareSupportMode = performance.opacityAwareSupportMode;
    this.shapeAwarePaintCullingEnabled = performance.shapeAwarePaintCulling;
    this.projectedVirtualExactCullingEnabled = performance.projectedVirtualExactCulling;
    this.segmentedExactBackwardRequested = performance.segmentedExactBackward;
    this.fixedPointExactGradientRequested = performance.fixedPointExactGradient;
    this.inverseScaleOptimizationRequested = performance.inverseScaleOptimization;
    this.trainingResidualMapPipeline = null;
    this.pixelMetricsPipeline = null;
    this.virtualCameraMetricsPipeline = null;
    this.overlapMetricsPipeline = null;
    this.overlapMetricsQaPipeline = null;
    this.alphaLossPipeline = null;
    this.presentPipeline = null;
    this.performanceProfile = {
      requested: Boolean(profile.profileRequested),
      timestampQuery: Boolean(profile.timestampQuery),
    };
    this.runtime = {
      tileCullingEnabled: typeof profile.tileCullingEnabled === "function"
        ? profile.tileCullingEnabled
        : () => true,
      trainLayerOrderEnabled: typeof profile.trainLayerOrderEnabled === "function"
        ? profile.trainLayerOrderEnabled
        : () => true,
      outsidePreviewEnabled: typeof profile.outsidePreviewEnabled === "function"
        ? profile.outsidePreviewEnabled
        : () => false,
      onPresentedState: typeof profile.onPresentedState === "function"
        ? profile.onPresentedState
        : () => {},
    };
    this.resultRenderState = null;
  }

  configureExperimentalPerformance(performance = performanceVariants()) {
    const opacityAwareSupportMode = performance.opacityAwareSupportMode === "aggressive"
      ? "aggressive"
      : "off";
    const tilePipelineChanged =
      this.opacityAwareSupportMode !== opacityAwareSupportMode ||
      this.shapeAwarePaintCullingEnabled !== performance.shapeAwarePaintCulling ||
      this.projectedVirtualExactCullingEnabled !== performance.projectedVirtualExactCulling;
    if (tilePipelineChanged) {
      this.opacityAwareSupportMode = opacityAwareSupportMode;
      this.shapeAwarePaintCullingEnabled = performance.shapeAwarePaintCulling;
      this.projectedVirtualExactCullingEnabled = performance.projectedVirtualExactCulling;
      this.tileCountPipeline = null;
      this.tilePrefixPipeline = null;
      this.tileFillPipeline = null;
      this.tileSortPipeline = null;
      if (this.trainState) {
        this.trainState.tileReady = false;
        this.trainState.bindGroupCache?.clear();
      }
    }
    if (this.segmentedExactBackwardRequested !== performance.segmentedExactBackward) {
      this.segmentedExactBackwardRequested = performance.segmentedExactBackward;
      this.exactAlphaBackwardPipeline = null;
      this.sourceDomainBackwardPipeline = null;
      this.segmentedReferenceCountPipeline = null;
      this.segmentedReferencePrefixPipeline = null;
      this.segmentedReferenceFillPipeline = null;
      this.segmentedGradientReducePipeline = null;
      this.compiledSegmentedExactBackward = null;
      this.trainState?.bindGroupCache?.clear();
    }
    if (this.fixedPointExactGradientRequested !== performance.fixedPointExactGradient) {
      this.fixedPointExactGradientRequested = performance.fixedPointExactGradient;
      this.exactAlphaBackwardPipeline = null;
      this.sourceDomainBackwardPipeline = null;
      this.virtualOrderPenaltyPipeline = null;
      this.brushLocalColorFlowPipeline = null;
      this.exactOptimizerPipeline = null;
      this.compiledFixedPointExactGradient = null;
      this.trainState?.bindGroupCache?.clear();
    }
    if (this.inverseScaleOptimizationRequested !== performance.inverseScaleOptimization) {
      this.inverseScaleOptimizationRequested = performance.inverseScaleOptimization;
      this.renderGradientPipeline = null;
      this.parallelRenderGradientPipeline = null;
      this.exactOptimizerPipeline = null;
      this.compiledInverseScaleOptimization = null;
      this.trainState?.bindGroupCache?.clear();
    }
    return {
      opacityAwareSupportMode: this.opacityAwareSupportMode,
      shapeAwarePaintCulling: this.shapeAwarePaintCullingEnabled,
      projectedVirtualExactCulling: this.projectedVirtualExactCullingEnabled,
      bindGroupCache: performance.bindGroupCache,
      segmentedExactBackward: this.segmentedExactBackwardRequested,
      fixedPointExactGradient: this.fixedPointExactGradientRequested,
      inverseScaleOptimization: this.inverseScaleOptimizationRequested,
      adaptiveGpuThroughput: performance.adaptiveGpuThroughput,
      adaptiveGpuBatch: performance.adaptiveGpuBatch,
      gpuSchedulingMode: performance.gpuSchedulingMode,
      asyncPresentation: performance.asyncPresentation,
      metricTileReuse: performance.metricTileReuse,
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

  operationProfileSample(step) {
    if (!this.performanceProfile.timestampQuery) return null;
    const labels = performanceProfileLabels(step, state.metrics?.steps_requested || step);
    return labels.length > 0 ? { step, labels, queryCount: 0, stages: [] } : null;
  }

  async submitProfiledOperation(encoder, profileSample, { resolution, activeSplats } = {}) {
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
    const wallStarted = performance.now();
    this.device.queue.submit([encoder.finish()]);
    if (!profileSample?.queryCount) return null;
    const waitStarted = performance.now();
    await this.device.queue.onSubmittedWorkDone();
    const queueWaitWallMs = performance.now() - waitStarted;
    const bytes = profileSample.queryCount * 8;
    const readbackStarted = performance.now();
    const readBuffer = this.trainState.profileReadbackBuffer;
    await readBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
    const timestamps = new BigUint64Array(readBuffer.getMappedRange(0, bytes)).slice();
    readBuffer.unmap();
    const stages = Object.fromEntries(profileSample.stages.map((stage) => [
      stage.name,
      Number(timestamps[stage.endOfPassWriteIndex] - timestamps[stage.beginningOfPassWriteIndex]) / 1e6,
    ]));
    const entry = {
      step: profileSample.step,
      labels: [...profileSample.labels],
      backend: "timestamp-query",
      resolution: resolution || [this.trainState.width, this.trainState.height],
      active_splats: activeSplats ?? this.trainState.count,
      stages_ms: stages,
      total_profiled_ms: Object.values(stages).reduce((sum, value) => sum + value, 0),
      queue_wait_count: 1,
      queue_wait_wall_ms: queueWaitWallMs,
      readback_count: 1,
      readback_bytes: bytes,
      readback_wall_ms: performance.now() - readbackStarted,
      total_wall_ms: performance.now() - wallStarted,
    };
    this.trainState.stageProfile.push(entry);
    return entry;
  }

  dispatchLinear(pass, workgroupCount) {
    const count = Math.max(1, Math.ceil(workgroupCount));
    const limit = Math.max(1, Number(this.device.limits?.maxComputeWorkgroupsPerDimension || 65535));
    const x = Math.min(count, limit);
    const y = Math.ceil(count / x);
    if (y > limit) throw new Error(`Linear compute dispatch ${count} exceeds the 2D WebGPU workgroup limit ${limit}x${limit}.`);
    pass.dispatchWorkgroups(x, y);
  }

  trainBindGroup(key, pipeline, entriesFactory) {
    const state = this.trainState;
    if (!state?.bindGroupCacheEnabled) {
      if (state?.bindGroupCacheStats) state.bindGroupCacheStats.misses += 1;
      return this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entriesFactory(),
      });
    }
    const cached = state.bindGroupCache.get(key);
    if (cached) {
      state.bindGroupCacheStats.hits += 1;
      return cached;
    }
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entriesFactory(),
    });
    state.bindGroupCache.set(key, bindGroup);
    state.bindGroupCacheStats.misses += 1;
    return bindGroup;
  }

  renderStatePipelineChoice() {
    const limits = this.device.limits || {};
    const supported =
      Number(limits.maxComputeInvocationsPerWorkgroup || 0) >= TILE_SIZE * TILE_SIZE &&
      Number(limits.maxComputeWorkgroupSizeX || 0) >= TILE_SIZE &&
      Number(limits.maxComputeWorkgroupSizeY || 0) >= TILE_SIZE &&
      Number(limits.maxComputeWorkgroupStorageSize || 0) >= 16384;
    const cooperative =
      performanceVariants().tileCooperativeRenderer &&
      this.runtime.tileCullingEnabled() &&
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

  async ensureAdaptiveGridInitializationPipeline() {
    if (this.adaptiveGridInitializationPipeline) return;
    const shader = `
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
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.adaptiveGridInitializationPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "initialize_adaptive_grid" },
    });
  }

  async applyAdaptiveGridInitialization(image, params, variants) {
    const requested = Boolean(variants?.requested);
    const monochromeUnderpainting = Boolean(variants?.monochromeUnderpainting);
    const skipped = (reason) => ({
      requested,
      applied: false,
      reason,
      fraction: Number(variants?.fraction || 0),
      candidate_count: Number(variants?.candidateCount || 0),
      moved_splats_estimate: 0,
      backend: "webgpu-compute",
    });
    if (!requested) return skipped("not-requested");
    if (!variants?.enabled) return skipped(variants?.reason || "disabled");
    if (!this.trainState || this.trainState.count !== params.count) return skipped("train-state-unavailable");
    if (params.count < 4) return skipped("too-few-splats");
    await this.ensureAdaptiveGridInitializationPipeline();
    const config = new Float32Array([
      image.width,
      image.height,
      params.count,
      variants.fraction,
      variants.candidateCount,
      variants.gridMargin,
      monochromeUnderpainting ? 1 : 0,
      0,
    ]);
    const configBuffer = this.device.createBuffer({
      size: config.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    try {
      this.device.queue.writeBuffer(configBuffer, 0, config);
      const front = this.trainState.front;
      const bindGroup = this.device.createBindGroup({
        layout: this.adaptiveGridInitializationPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.targetBuffer } },
          { binding: 2, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.adaptiveGridInitializationPipeline);
      pass.setBindGroup(0, bindGroup);
      this.dispatchLinear(pass, Math.ceil(params.count / 64));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      return {
        requested,
        applied: true,
        reason: "applied",
        fraction: variants.fraction,
        candidate_count: variants.candidateCount,
        moved_splats_estimate: Math.round(params.count * variants.fraction),
        backend: "webgpu-compute",
        parameter_scope: "center-and-image-rgb-only",
      };
    } finally {
      configBuffer.destroy();
    }
  }

  async ensureDiscreteLayerPipelines() {
    if (this.discreteLayerAssignPipeline && this.discreteLayerCommitPipeline) return;
    const shader = `
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
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    [this.discreteLayerAssignPipeline, this.discreteLayerCommitPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "assign_layers" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "commit_layers" } }),
    ]);
  }

  async prepareTileLists(
    image,
    params,
    {
      sync = false,
      encoder = null,
      profileSample = null,
      writeConfig = true,
    } = {},
  ) {
    if (!this.trainState || this.trainState.capacity < params.count) return false;
    await this.ensureTilePipelines();
    if (writeConfig) {
      const config = new Float32Array(TRAIN_CONFIG_FLOATS);
      config.set([image.width, image.height, params.count, params.bg[0], params.bg[1], params.bg[2]], 0);
      config[17] = currentMaxAnisotropy();
      config[18] = experimentalDensifySteps(state.metrics?.steps_requested || 1);
      config[19] = this.runtime.tileCullingEnabled() ? 1 : 0;
      config[26] = phase33Variants().ewa2x2 ? 1 : 0;
      config[47] = 0;
      config[45] = params.layerOrderEnabled ? 1 : 0;
      config[67] = params.virtualDepthEnabled ? 1 : 0;
      config[68] = params.virtualDepthEnabled ? Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS : 0;
      configurePaintKernel(config, params);
      this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    }
    const front = this.trainState.front;
    const tileBindGroup = (kind, pipeline, entriesFactory) => this.trainBindGroup(
      `tile-${kind}:${front}:${this.opacityAwareSupportMode}`,
      pipeline,
      entriesFactory,
    );
    const commonEntries = () => [
      { binding: 0, resource: { buffer: this.trainState.configBuffer } },
      { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
      { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
      ...(this.opacityAwareSupportMode !== "off"
        ? [{ binding: 8, resource: { buffer: this.trainState.colorBuffers[front] } }]
        : []),
    ];
    const countBindGroup = tileBindGroup("count", this.tileCountPipeline, () => [
        ...commonEntries(),
        { binding: 3, resource: { buffer: this.trainState.tileCountsBuffer } },
      ]);
    const prefixBindGroup = tileBindGroup("prefix", this.tilePrefixPipeline, () => [
        { binding: 3, resource: { buffer: this.trainState.tileCountsBuffer } },
        { binding: 4, resource: { buffer: this.trainState.tileOffsetsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
        { binding: 7, resource: { buffer: this.trainState.tileControlBuffer } },
      ]);
    const fillBindGroup = tileBindGroup("fill", this.tileFillPipeline, () => [
        ...commonEntries(),
        { binding: 4, resource: { buffer: this.trainState.tileOffsetsBuffer } },
        { binding: 5, resource: { buffer: this.trainState.tileCursorsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
      ]);
    const sortBindGroup = tileBindGroup("sort", this.tileSortPipeline, () => [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
        { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
        { binding: 4, resource: { buffer: this.trainState.tileOffsetsBuffer } },
        { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
      ]);
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

  async hashTrainParameters(params) {
    if (!this.trainState || this.trainState.count !== params.count) return null;
    const front = this.trainState.front;
    const xyBytes = params.count * 4 * 4;
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
      orderMode: "tiles",
    };
  }

  resultRenderMemorySnapshot() {
    const buffers = this.resultRenderState?.buffers || [];
    const bytes = buffers.reduce((total, buffer) => total + Math.max(0, Number(buffer?.size) || 0), 0);
    return { activeBytes: bytes, reservedBytes: bytes };
  }

  resultRenderAllocationPlan(candidateBytes) {
    const currentResultBytes = this.resultRenderMemorySnapshot().reservedBytes;
    const trainingBytes = this.trainingMemorySnapshot().reservedBytes;
    const candidate = Math.max(0, Number(candidateBytes) || 0);
    const budgetBytes = memoryBudgetBytes();
    const transientBytes = trainingBytes + currentResultBytes + candidate;
    return {
      training_bytes: trainingBytes,
      current_result_bytes: currentResultBytes,
      candidate_bytes: candidate,
      transient_bytes: transientBytes,
      budget_bytes: budgetBytes,
      within_budget: transientBytes <= budgetBytes,
    };
  }

  disposeResultRenderState() {
    if (!this.resultRenderState) return;
    this.resultRenderState.smallFirstOrderCache = null;
    destroyBuffers(...this.resultRenderState.buffers);
    this.resultRenderState = null;
    updateGpuMemoryStatus();
  }

  currentResultBuffers(params) {
    const result = this.resultRenderState;
    if (!result || result.count !== params?.count) return null;
    return {
      xyBuffer: result.xyBuffer,
      transformBuffer: result.transformBuffer,
      colorBuffer: result.colorBuffer,
      tileOffsetsBuffer: result.tileOffsetsBuffer,
      tileIndicesBuffer: result.tileIndicesBuffer,
      orderMode: result.orderMode,
    };
  }

  async preserveResultRenderState(image, params) {
    if (!this.trainState || this.trainState.count !== params.count) return false;
    const count = params.count;
    const front = this.trainState.front;
    const parameterBytes = Math.max(16, count * 4 * 4);
    const tileCount = Math.ceil(image.width / TILE_SIZE) * Math.ceil(image.height / TILE_SIZE);
    const tileReferences = Math.max(
      0,
      Math.min(
        this.trainState.tileIndexCapacity,
        Math.round(Number(state.metrics?.tile_counters?.total) || 0),
      ),
    );
    const copyTiles = Boolean(
      this.runtime.tileCullingEnabled() &&
      this.trainState.tileReady &&
      this.trainState.pixelStateKind === "full" &&
      tileReferences > 0
    );
    const tileOffsetBytes = copyTiles ? Math.max(4, (tileCount + 1) * 4) : 0;
    const tileIndexBytes = copyTiles ? Math.max(4, tileReferences * 4) : 0;
    const allocation = this.resultRenderAllocationPlan(parameterBytes * 3 + tileOffsetBytes + tileIndexBytes);
    if (!allocation.within_budget) {
      if (state.metrics) {
        state.metrics.result_render_cache = {
          source: this.resultRenderState ? "retained-previous-result-cache" : "deferred-after-train-release",
          count,
          allocation,
          deferred: true,
        };
      }
      return false;
    }
    const previous = this.resultRenderState;
    const buffers = [];
    const makeResultBuffer = (size) => {
      const buffer = this.device.createBuffer({
        size: Math.max(4, Math.ceil(size / 4) * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      buffers.push(buffer);
      return buffer;
    };
    let scopesPopped = false;
    try {
      this.device.pushErrorScope("out-of-memory");
      this.device.pushErrorScope("validation");
      const xyBuffer = makeResultBuffer(parameterBytes);
      const transformBuffer = makeResultBuffer(parameterBytes);
      const colorBuffer = makeResultBuffer(parameterBytes);
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.xyBuffers[front], 0, xyBuffer, 0, parameterBytes);
      encoder.copyBufferToBuffer(this.trainState.transformBuffers[front], 0, transformBuffer, 0, parameterBytes);
      encoder.copyBufferToBuffer(this.trainState.colorBuffers[front], 0, colorBuffer, 0, parameterBytes);
      let tileOffsetsBuffer = null;
      let tileIndicesBuffer = null;
      let orderMode = "none";
      if (copyTiles) {
        tileOffsetsBuffer = makeResultBuffer(tileOffsetBytes);
        tileIndicesBuffer = makeResultBuffer(tileIndexBytes);
        encoder.copyBufferToBuffer(this.trainState.tileOffsetsBuffer, 0, tileOffsetsBuffer, 0, tileOffsetBytes);
        encoder.copyBufferToBuffer(this.trainState.tileIndicesBuffer, 0, tileIndicesBuffer, 0, tileIndexBytes);
        orderMode = "tiles";
      }
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const validationError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      scopesPopped = true;
      if (validationError || oomError) throw validationError || oomError;
      this.resultRenderState = {
        count,
        sourceParams: params,
        smallFirstOrderCache: null,
        xyBuffer,
        transformBuffer,
        colorBuffer,
        tileOffsetsBuffer,
        tileIndicesBuffer,
        orderMode,
        buffers,
      };
      destroyBuffers(...(previous?.buffers || []));
      if (state.metrics) {
        state.metrics.result_render_cache = {
          source: "gpu-final-state-copy",
          count,
          order_mode: orderMode,
          bytes: this.resultRenderMemorySnapshot().reservedBytes,
          tile_references: orderMode === "tiles" ? tileReferences : 0,
          allocation,
        };
      }
      updateGpuMemoryStatus();
      return true;
    } catch (error) {
      destroyBuffers(...buffers);
      if (!scopesPopped) {
        await this.device.popErrorScope().catch(() => null);
        await this.device.popErrorScope().catch(() => null);
      }
      if (state.metrics) {
        state.metrics.result_render_cache = {
          source: previous ? "retained-previous-result-cache" : "deferred-after-train-release",
          count,
          allocation,
          deferred: true,
          error: String(error?.message || error),
        };
      }
      updateGpuMemoryStatus();
      return false;
    }
  }

  async uploadResultRenderState(params) {
    if (!params?.count) return false;
    const allocation = this.resultRenderAllocationPlan(params.count * (16 * 3 + 4) + 8);
    if (!allocation.within_budget) {
      if (state.metrics) {
        state.metrics.result_render_cache = {
          source: this.resultRenderState ? "retained-previous-result-cache" : "cpu-recovery-upload-deferred",
          count: params.count,
          allocation,
          deferred: true,
        };
      }
      return false;
    }
    const previous = this.resultRenderState;
    const buffers = [];
    let scopesPopped = false;
    try {
      this.device.pushErrorScope("out-of-memory");
      this.device.pushErrorScope("validation");
      const ordered = new Uint32Array(params.count);
      for (let index = 0; index < params.count; index += 1) ordered[index] = index;
      if (params.layerOrderEnabled) ordered.sort((a, b) => layerOrderComparator(a, b, params));
      const xyBuffer = makeBuffer(this.device, packPositions(params), GPUBufferUsage.STORAGE, (buffer) => buffers.push(buffer));
      const transformBuffer = makeBuffer(this.device, packTransforms(params), GPUBufferUsage.STORAGE, (buffer) => buffers.push(buffer));
      const colorBuffer = makeBuffer(this.device, packColors(params), GPUBufferUsage.STORAGE, (buffer) => buffers.push(buffer));
      const tileOffsetsBuffer = makeBuffer(
        this.device,
        new Uint32Array([0, params.count]),
        GPUBufferUsage.STORAGE,
        (buffer) => buffers.push(buffer),
      );
      const tileIndicesBuffer = makeBuffer(this.device, ordered, GPUBufferUsage.STORAGE, (buffer) => buffers.push(buffer));
      await this.device.queue.onSubmittedWorkDone();
      const validationError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      scopesPopped = true;
      if (validationError || oomError) throw validationError || oomError;
      this.resultRenderState = {
        count: params.count,
        sourceParams: params,
        smallFirstOrderCache: null,
        xyBuffer,
        transformBuffer,
        colorBuffer,
        tileOffsetsBuffer,
        tileIndicesBuffer,
        orderMode: "global",
        buffers,
      };
      destroyBuffers(...(previous?.buffers || []));
      if (state.metrics) {
        state.metrics.result_render_cache = {
          source: "cpu-recovery-upload-once",
          count: params.count,
          order_mode: "global",
          bytes: this.resultRenderMemorySnapshot().reservedBytes,
          tile_references: 0,
          allocation,
        };
      }
      updateGpuMemoryStatus();
      return true;
    } catch (error) {
      destroyBuffers(...buffers);
      if (!scopesPopped) {
        await this.device.popErrorScope().catch(() => null);
        await this.device.popErrorScope().catch(() => null);
      }
      if (state.metrics) {
        state.metrics.result_render_cache = {
          source: previous ? "retained-previous-result-cache" : "cpu-recovery-upload-deferred",
          count: params.count,
          allocation,
          deferred: true,
          error: String(error?.message || error),
        };
      }
      updateGpuMemoryStatus();
      return false;
    }
  }

  async render(image, params, sourceBuffers = null, targetView = null, options = {}) {
    this.ensurePipeline(params.count);
    const preview = previewPaddingSpec(image, params, options.outside ?? this.runtime.outsidePreviewEnabled());
    const presentingToCanvas = !targetView;
    const padded = preview.x > 0 || preview.y > 0;
    const maxStorageBinding = Math.max(
      4,
      Number(this.device.limits?.maxStorageBufferBindingSize) || 128 * MB,
    );
    // Padded/resized frames cannot reuse the training-frame tile coordinates.
    // Rebuild exact view-local tiles instead of falling back to an unsafe
    // per-pixel scan of every splat, independent of the training culling toggle.
    const rebuildPreviewTiles = padded || Boolean(options.rebuildTiles);
    let paddedTileData = null;
    if (rebuildPreviewTiles) {
      const tileCount = Math.ceil(preview.width / TILE_SIZE) * Math.ceil(preview.height / TILE_SIZE);
      if ((tileCount + 1) * 4 > maxStorageBinding) {
        throw new Error("Preview needs more tile offsets than this GPU can bind.");
      }
      paddedTileData = buildPreviewTileIndexData(image, params, {
        ...options,
        outside: Boolean(options.outside ?? this.runtime.outsidePreviewEnabled()),
        maxTileReferences: Math.floor(maxStorageBinding / 4),
      });
    }
    if (presentingToCanvas) {
      state.previewPadding = preview;
      if (this.canvas.width !== preview.width) this.canvas.width = preview.width;
      if (this.canvas.height !== preview.height) this.canvas.height = preview.height;
    }
    const alphaBackground = Array.isArray(options.alphaBackground)
      ? options.alphaBackground
      : [params.bg[0], params.bg[1], params.bg[2]];
    const kernelFalloff = Number.isFinite(Number(options.kernelFalloff))
      ? Math.max(0, Math.min(2, Number(options.kernelFalloff)))
      : 1;
    const requestedShape = options.splatShape || trainedSplatShape(params);
    const shapeMode = requestedShape === "opaque-brush"
      ? 5
      : requestedShape === "box"
          ? 2
          : requestedShape === "rectangle" ? 1 : 0;
    const useSplatPreviewOrder = Boolean(options.splatSmallFirstOrder);
    // Shape changes only the footprint kernel. Paint kernels and Gaussian
    // share the trained layer order unless the user explicitly enables the
    // preview-only small-first override.
    const useTileOrder = Boolean(paddedTileData) || (
      !useSplatPreviewOrder &&
      this.runtime.tileCullingEnabled() &&
      sourceBuffers?.orderMode === "tiles" &&
      Boolean(sourceBuffers?.tileOffsetsBuffer)
    );
    const useCachedGlobalOrder =
      !useSplatPreviewOrder &&
      !useTileOrder &&
      sourceBuffers?.orderMode === "global" &&
      Boolean(sourceBuffers?.tileOffsetsBuffer) &&
      Boolean(sourceBuffers?.tileIndicesBuffer);
    const useGlobalOrder =
      !useTileOrder &&
      !useCachedGlobalOrder &&
      (Boolean(params.layerOrderEnabled) || useSplatPreviewOrder);
    const paintKernelShape = normalizedKernelShape(params.kernelShape);
    const centerOpacityMin = paintKernelShape === "rectangle"
      ? clampNumber(params.rectangleCenterOpacityGradientMin, 0, 1, 1)
      : clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1);
    const centerOpacityMax = paintKernelShape === "rectangle"
      ? clampNumber(params.rectangleCenterOpacityGradientMax, centerOpacityMin, 1, 1)
      : clampNumber(params.brushCenterOpacityGradientMax, centerOpacityMin, 1, 1);
    const uniform = new Float32Array([
      preview.width,
      preview.height,
      params.count,
      params.bg[0],
      params.bg[1],
      params.bg[2],
      useTileOrder ? 1 : useCachedGlobalOrder || useGlobalOrder ? 2 : 0,
      phase33Variants().ewa2x2 ? 1 : 0,
      image.width,
      image.height,
      preview.scaleX,
      preview.scaleY,
      phase37Variants().ewaGaussLegendre ? 1 : 0,
      alphaBackground[0] || 0,
      alphaBackground[1] || 0,
      alphaBackground[2] || 0,
      kernelFalloff,
      shapeMode,
      params.layerAwareAccumulationEnabled ? 1 : 0,
      Math.max(MIN_DISCRETE_LAYER_COUNT, Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(params.discreteLayerCount || DEFAULT_DISCRETE_LAYER_COUNT))),
      clampNumber(params.brushWidthTaperStart, 0, 1, 1),
      clampNumber(params.brushWidthTaperEnd, 0, 1, 0),
      clampNumber(
        params.rectangleTopRatio,
        MIN_RECTANGLE_TOP_RATIO,
        MAX_RECTANGLE_TOP_RATIO,
        selectedRectangleTopRatio(),
      ),
      clampNumber(
        params.rectangleTopRatioMax,
        params.rectangleTopRatio,
        MAX_RECTANGLE_TOP_RATIO,
        selectedRectangleTopRatioMax(params.rectangleTopRatio),
      ),
      rectangleShapeFlags(params),
      Number.isFinite(Number(options.opacityMultiplier))
        ? Math.max(0, Number(options.opacityMultiplier))
        : 1,
      Number.isFinite(Number(options.splatScaleMultiplier))
        ? Math.max(0, Number(options.splatScaleMultiplier))
        : 1,
      Math.max(0.000001, Number(options.localAspectRatio) || 1),
      paintKernelShape === "rectangle"
        ? clampNumber(params.rectangleOpacityGradientMin, 0, 1, 1)
        : clampNumber(params.brushOpacityGradientStart, 0, 1, 1),
      params.brushOpacityGradientEnabled ? 1 : 0,
      params.brushWidthTaperEnabled ? 1 : 0,
      paintKernelShape === "rectangle"
        ? clampNumber(params.rectangleOpacityGradientMax, 0, 1, 1)
        : clampNumber(params.brushOpacityGradientEnd, 0, 1, 1),
      centerOpacityMin,
      centerOpacityMax,
      0,
      0,
    ]);
    const buffers = [];
    try {
      let xyBuffer = sourceBuffers?.xyBuffer;
      let transformBuffer = sourceBuffers?.transformBuffer;
      let colorBuffer = sourceBuffers?.colorBuffer;
      let tileOffsetsBuffer = sourceBuffers?.tileOffsetsBuffer;
      let tileIndicesBuffer = sourceBuffers?.tileIndicesBuffer;
      if (!xyBuffer || !transformBuffer || !colorBuffer) {
        const color = packColors(params);
        const transform = packTransforms(params);
        xyBuffer = makeBuffer(this.device, packPositions(params), GPUBufferUsage.STORAGE);
        buffers.push(xyBuffer);
        transformBuffer = makeBuffer(this.device, transform, GPUBufferUsage.STORAGE);
        buffers.push(transformBuffer);
        colorBuffer = makeBuffer(this.device, color, GPUBufferUsage.STORAGE);
        buffers.push(colorBuffer);
      }
      if (paddedTileData) {
        tileOffsetsBuffer = makeBuffer(this.device, paddedTileData.offsets, GPUBufferUsage.STORAGE);
        buffers.push(tileOffsetsBuffer);
        tileIndicesBuffer = makeBuffer(this.device, paddedTileData.indices, GPUBufferUsage.STORAGE);
        buffers.push(tileIndicesBuffer);
      }
      const resultState = this.resultRenderState;
      const resultStateMatchesSource = Boolean(
        resultState?.sourceParams === params &&
        resultState.count === params.count &&
        xyBuffer === resultState.xyBuffer &&
        transformBuffer === resultState.transformBuffer &&
        colorBuffer === resultState.colorBuffer,
      );
      if (useGlobalOrder) {
        let ordered;
        if (useSplatPreviewOrder) {
          ordered = cachedResultSmallFirstOrder(resultStateMatchesSource ? resultState : null, params) || buildSplatPreviewOrder(params);
        } else {
          ordered = new Uint32Array(params.count);
          for (let i = 0; i < params.count; i += 1) ordered[i] = i;
          ordered.sort((a, b) => layerOrderComparator(a, b, params));
        }
        tileOffsetsBuffer = makeBuffer(this.device, new Uint32Array([0, params.count]), GPUBufferUsage.STORAGE);
        buffers.push(tileOffsetsBuffer);
        tileIndicesBuffer = makeBuffer(this.device, ordered, GPUBufferUsage.STORAGE);
        buffers.push(tileIndicesBuffer);
      } else if (useCachedGlobalOrder) {
        tileOffsetsBuffer = sourceBuffers.tileOffsetsBuffer;
        tileIndicesBuffer = sourceBuffers.tileIndicesBuffer;
      } else if (!tileOffsetsBuffer || !tileIndicesBuffer) {
        tileOffsetsBuffer = makeBuffer(this.device, new Uint32Array([0, params.count]), GPUBufferUsage.STORAGE);
        buffers.push(tileOffsetsBuffer);
        tileIndicesBuffer = makeBuffer(this.device, new Uint32Array([0]), GPUBufferUsage.STORAGE);
        buffers.push(tileIndicesBuffer);
      }
      const usePersistentPreviewState =
        !useGlobalOrder &&
        resultState?.count === params.count &&
        xyBuffer === resultState.xyBuffer &&
        transformBuffer === resultState.transformBuffer &&
        colorBuffer === resultState.colorBuffer &&
        tileOffsetsBuffer === resultState.tileOffsetsBuffer &&
        tileIndicesBuffer === resultState.tileIndicesBuffer;
      this.lastPreviewStats = {
        padded,
        tile_mode: paddedTileData
          ? (padded ? "padded-tiles" : "rebuilt-tiles")
          : useTileOrder
            ? "cached-tiles"
            : useCachedGlobalOrder || useGlobalOrder
              ? "global-order"
              : "linear",
        tile_references: paddedTileData?.indices.length || 0,
      };
      let uniformBuffer;
      let bindGroup;
      if (usePersistentPreviewState) {
        if (!resultState.previewUniformBuffer) {
          resultState.previewUniformBuffer = this.device.createBuffer({
            size: Math.ceil(uniform.byteLength / 4) * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          resultState.buffers.push(resultState.previewUniformBuffer);
          resultState.previewUniformAllocations =
            (resultState.previewUniformAllocations || 0) + 1;
        }
        uniformBuffer = resultState.previewUniformBuffer;
        this.device.queue.writeBuffer(
          uniformBuffer,
          0,
          uniform.buffer,
          uniform.byteOffset,
          uniform.byteLength,
        );
        resultState.previewUniformWrites =
          (resultState.previewUniformWrites || 0) + 1;
        if (!resultState.previewBindGroup || resultState.previewPipeline !== this.pipeline) {
          resultState.previewBindGroup = this.device.createBindGroup({
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
          resultState.previewPipeline = this.pipeline;
          resultState.previewBindGroupCreations =
            (resultState.previewBindGroupCreations || 0) + 1;
        }
        bindGroup = resultState.previewBindGroup;
        if (state.metrics?.result_render_cache) {
          state.metrics.result_render_cache.preview_uniform_allocations =
            resultState.previewUniformAllocations || 0;
          state.metrics.result_render_cache.preview_uniform_writes =
            resultState.previewUniformWrites || 0;
          state.metrics.result_render_cache.preview_bind_group_creations =
            resultState.previewBindGroupCreations || 0;
          state.metrics.result_render_cache.bytes =
            this.resultRenderMemorySnapshot().reservedBytes;
        }
      } else {
        uniformBuffer = makeBuffer(this.device, uniform, GPUBufferUsage.UNIFORM);
        buffers.push(uniformBuffer);
      }
      bindGroup ||= this.device.createBindGroup({
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
      // Persistent result-cache renders own every referenced buffer and WebGPU
      // submission order preserves the latest-only preview sequence. Fence only
      // when a temporary resource must be destroyed or a capture target follows.
      if (buffers.length > 0 || targetView) {
        await this.device.queue.onSubmittedWorkDone();
      }
    } finally {
      destroyBuffers(...buffers);
    }
  }

  async captureRenderedRgba(image, params, sourceBuffers = null, options = {}) {
    const preview = previewPaddingSpec(image, params, Boolean(options.outside));
    const limits = this.device.limits || {};
    const maxTextureDimension = Math.max(1, Number(limits.maxTextureDimension2D) || MAX_PNG_EXPORT_LONG_SIDE);
    if (preview.width > maxTextureDimension || preview.height > maxTextureDimension) {
      throw new Error(`PNG resolution ${preview.width}x${preview.height} exceeds this GPU's ${maxTextureDimension}px texture limit.`);
    }
    const texture = this.device.createTexture({
      size: [preview.width, preview.height, 1],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const bytesPerPixel = 4;
    const bytesPerRow = Math.ceil((preview.width * bytesPerPixel) / 256) * 256;
    const readbackBytes = bytesPerRow * preview.height;
    const maxBufferSize = Math.max(1, Number(limits.maxBufferSize) || Number.MAX_SAFE_INTEGER);
    if (readbackBytes > maxBufferSize) {
      throw new Error(`PNG readback needs ${(readbackBytes / MB).toFixed(1)} MB, above this GPU's ${(maxBufferSize / MB).toFixed(1)} MB buffer limit.`);
    }
    const readBuffer = this.device.createBuffer({
      size: readbackBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      await this.render(image, params, sourceBuffers, texture.createView(), options);
      const encoder = this.device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture },
        { buffer: readBuffer, bytesPerRow, rowsPerImage: preview.height },
        { width: preview.width, height: preview.height, depthOrArrayLayers: 1 },
      );
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const source = new Uint8Array(readBuffer.getMappedRange());
      const rgba = new Uint8ClampedArray(preview.width * preview.height * 4);
      const bgra = this.format.startsWith("bgra");
      for (let y = 0; y < preview.height; y += 1) {
        const sourceRow = y * bytesPerRow;
        const targetRow = y * preview.width * bytesPerPixel;
        for (let x = 0; x < preview.width; x += 1) {
          const si = sourceRow + x * bytesPerPixel;
          const di = targetRow + x * bytesPerPixel;
          rgba[di] = source[si + (bgra ? 2 : 0)];
          rgba[di + 1] = source[si + 1];
          rgba[di + 2] = source[si + (bgra ? 0 : 2)];
          rgba[di + 3] = source[si + 3];
        }
      }
      return { rgba, width: preview.width, height: preview.height, padding: preview };
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
      readBuffer.destroy();
      texture.destroy();
    }
  }

  async captureFrameRgba(image, params, sourceBuffers = null) {
    const capture = await this.captureRenderedRgba(image, params, sourceBuffers, { outside: false });
    return capture.rgba;
  }

  async ensurePresentedStatePackPipeline() {
    if (this.presentedStatePackPipeline) return;
    // pack4x8unorm is specified as floor(0.5 + 255 * clamp(x, 0, 1)), which
    // is the finite-f32 equivalent of the existing clampByte(Math.round)
    // conversion used by the final training-frame parity contract.
    const shader = `
@group(0) @binding(0) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> packedRgba: array<u32>;

@compute @workgroup_size(256)
fn pack_presented_state(@builtin(global_invocation_id) id: vec3<u32>) {
  let pixel = id.x;
  if (pixel >= arrayLength(&pixelState)) { return; }
  packedRgba[pixel] = pack4x8unorm(pixelState[pixel]);
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.presentedStatePackPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "pack_presented_state" },
    });
  }

  presentedStateCaptureAllocationPlan(byteLength) {
    const trainingBytes = this.trainingMemorySnapshot().reservedBytes;
    const resultBytes = this.resultRenderMemorySnapshot().reservedBytes;
    const packedBytes = Math.max(4, Number(byteLength) || 0);
    const budgetBytes = memoryBudgetBytes();
    const transientBytes = packedBytes * 2;
    return {
      training_bytes: trainingBytes,
      result_render_bytes: resultBytes,
      packed_bytes: packedBytes,
      readback_bytes: packedBytes,
      transient_bytes: transientBytes,
      budget_bytes: budgetBytes,
      within_budget: trainingBytes + resultBytes + transientBytes <= budgetBytes,
    };
  }

  async capturePresentedStateRgba() {
    if (!this.trainState?.pixelStateBuffer) return null;
    const resolution = this.trainState.pixelStateResolution || [this.trainState.width, this.trainState.height];
    const width = Math.max(1, Math.round(resolution[0]));
    const height = Math.max(1, Math.round(resolution[1]));
    const bytes = width * height * 4;
    const allocation = this.presentedStateCaptureAllocationPlan(bytes);
    if (!allocation.within_budget) {
      throw new Error(
        `Final training RGBA capture exceeds the transient GPU budget (${formatMB(allocation.transient_bytes)} requested).`,
      );
    }
    await this.ensurePresentedStatePackPipeline();
    let packedBuffer = null;
    let readBuffer = null;
    let scopesOpen = false;
    try {
      this.device.pushErrorScope("out-of-memory");
      this.device.pushErrorScope("validation");
      scopesOpen = true;
      packedBuffer = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      readBuffer = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const bindGroup = this.device.createBindGroup({
        layout: this.presentedStatePackPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 1, resource: { buffer: packedBuffer } },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.presentedStatePackPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(width * height / 256));
      pass.end();
      encoder.copyBufferToBuffer(packedBuffer, 0, readBuffer, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const validationError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      scopesOpen = false;
      if (validationError || oomError) throw validationError || oomError;
      await readBuffer.mapAsync(GPUMapMode.READ);
      const rgba = new Uint8ClampedArray(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();
      return { rgba, width, height, kind: this.trainState.pixelStateKind || "full" };
    } finally {
      if (scopesOpen) {
        await this.device.popErrorScope().catch(() => null);
        await this.device.popErrorScope().catch(() => null);
      }
      if (readBuffer?.mapState === "mapped") readBuffer.unmap();
      readBuffer?.destroy();
      packedBuffer?.destroy();
    }
  }

  async readLayerTelemetryGeometry(count) {
    if (!this.trainState || count <= 0) return null;
    const xyBytes = count * 4 * 4;
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
      const packedPositions = new Float32Array(mapped, 0, count * 4);
      const xy = new Float32Array(count * 2);
      for (let index = 0; index < count; index += 1) {
        xy[index * 2] = packedPositions[index * 4];
        xy[index * 2 + 1] = packedPositions[index * 4 + 1];
      }
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
    this.runtime.onPresentedState({
      width,
      height,
      kind: this.trainState.pixelStateKind || "full",
    });
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

  async computeMetrics(image, params, sourceBuffers = null, options = {}) {
    if (!sourceBuffers?.pixelStateBuffer) {
      throw new Error("Experimental metrics require active WebGPU render state");
    }
    return this.computeTrainStateMetrics(image, params, options);
  }

  canReuseMetricRender(image) {
    const resolution = this.trainState?.pixelStateResolution;
    return Boolean(
      this.trainState?.tileReady &&
      this.trainState.pixelStateKind === "full" &&
      this.trainState.pixelStateViewKey === "front" &&
      resolution?.[0] === image.width &&
      resolution?.[1] === image.height
    );
  }

  async ensureRenderGradientPipelines() {
    const legacyGradientSupported = Number(this.device.limits?.maxStorageBuffersPerShaderStage || 8) >= 9;
    const segmentedExactBackwardEnabled = Boolean(this.trainState?.segmentedExactBackward?.enabled);
    const fixedPointExactGradientEnabled = Boolean(this.trainState?.fixedPointExactGradient?.enabled);
    const inverseScaleOptimizationEnabled = Boolean(this.inverseScaleOptimizationRequested);
    if (
      this.compiledSegmentedExactBackward !== null &&
      this.compiledSegmentedExactBackward !== segmentedExactBackwardEnabled
    ) {
      this.exactAlphaBackwardPipeline = null;
      this.sourceDomainBackwardPipeline = null;
      this.segmentedReferenceCountPipeline = null;
      this.segmentedReferencePrefixPipeline = null;
      this.segmentedReferenceFillPipeline = null;
      this.segmentedGradientReducePipeline = null;
      this.trainState?.bindGroupCache?.clear();
    }
    if (
      this.compiledInverseScaleOptimization !== null &&
      this.compiledInverseScaleOptimization !== inverseScaleOptimizationEnabled
    ) {
      this.renderGradientPipeline = null;
      this.parallelRenderGradientPipeline = null;
      this.exactOptimizerPipeline = null;
      this.trainState?.bindGroupCache?.clear();
    }
    if (
      this.compiledFixedPointExactGradient !== null &&
      this.compiledFixedPointExactGradient !== fixedPointExactGradientEnabled
    ) {
      this.exactAlphaBackwardPipeline = null;
      this.sourceDomainBackwardPipeline = null;
      this.virtualOrderPenaltyPipeline = null;
      this.brushLocalColorFlowPipeline = null;
      this.exactOptimizerPipeline = null;
      this.trainState?.bindGroupCache?.clear();
    }
    if (
      this.renderStatePipeline &&
      this.tileCooperativeRenderPipeline &&
      this.ssimHorizontalPipeline &&
      this.ssimTilePipeline &&
      this.ssimBackwardHorizontalPipeline &&
      this.ssimBackwardVerticalPipeline &&
      this.alphaSsimTilePipeline &&
      (!legacyGradientSupported || (this.renderGradientPipeline && this.parallelRenderGradientPipeline)) &&
      this.lossGradientPipeline &&
      this.exactAlphaBackwardPipeline &&
      this.sourceDomainBackwardPipeline &&
      this.virtualOrderPenaltyPipeline &&
      this.brushLocalColorFlowPipeline &&
      this.exactBackwardTelemetryPipeline &&
      this.exactOptimizerPipeline &&
      (!segmentedExactBackwardEnabled || (
        this.segmentedReferenceCountPipeline &&
        this.segmentedReferencePrefixPipeline &&
        this.segmentedReferenceFillPipeline &&
        this.segmentedGradientReducePipeline
      ))
    ) return;
    const optimizerStatsDeclaration = "@group(0) @binding(8) var<storage, read_write> stats: array<vec4<f32>>;";
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
    nextTheta = select(
      select(1.57079632679, 0.0, longAxisIsX),
      select(0.0, 1.57079632679, longAxisIsX),
      vertical
    );
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
    nextTheta = select(
      select(1.57079632679, 0.0, longAxisIsX),
      select(0.0, 1.57079632679, longAxisIsX),
      vertical
    );
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

    const renderModule = this.device.createShaderModule({ code: renderShader });
    const ssimModule = this.device.createShaderModule({ code: ssimShader });
    const lossGradientModule = this.device.createShaderModule({ code: lossGradientShader });
    const exactBackwardModule = this.device.createShaderModule({ code: exactBackwardShader });
    const virtualOrderPenaltyModule = this.device.createShaderModule({ code: virtualOrderPenaltyShader });
    const brushLocalColorFlowModule = this.device.createShaderModule({ code: brushLocalColorFlowShader });
    const exactBackwardTelemetryModule = this.device.createShaderModule({ code: exactBackwardTelemetryShader });
    const optimizerModule = this.device.createShaderModule({ code: optimizerShader });
    const segmentedReferenceModule = segmentedExactBackwardEnabled
      ? this.device.createShaderModule({ code: segmentedReferenceShader })
      : null;
    const segmentedGradientReduceModule = segmentedExactBackwardEnabled
      ? this.device.createShaderModule({ code: segmentedGradientReduceShader })
      : null;
    for (const module of [
      renderModule,
      ssimModule,
      lossGradientModule,
      exactBackwardModule,
      virtualOrderPenaltyModule,
      brushLocalColorFlowModule,
      exactBackwardTelemetryModule,
      optimizerModule,
      segmentedReferenceModule,
      segmentedGradientReduceModule,
    ].filter(Boolean)) {
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === "error");
      if (errors.length && module === exactBackwardModule && this.subgroupExactBackwardEnabled) {
        this.subgroupExactBackwardEnabled = false;
        eventLog(`subgroup exact backward unavailable; using portable workgroup reduction: ${errors[0].message}`);
        return this.ensureRenderGradientPipelines();
      }
      if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    }
    [
      this.renderStatePipeline,
      this.tileCooperativeRenderPipeline,
      this.ssimHorizontalPipeline,
      this.ssimTilePipeline,
      this.ssimBackwardHorizontalPipeline,
      this.ssimBackwardVerticalPipeline,
      this.alphaSsimTilePipeline,
      this.renderGradientPipeline,
      this.parallelRenderGradientPipeline,
      this.lossGradientPipeline,
      this.exactAlphaBackwardPipeline,
      this.sourceDomainBackwardPipeline,
      this.virtualOrderPenaltyPipeline,
      this.brushLocalColorFlowPipeline,
      this.exactBackwardTelemetryPipeline,
      this.exactOptimizerPipeline,
      this.segmentedReferenceCountPipeline,
      this.segmentedReferencePrefixPipeline,
      this.segmentedReferenceFillPipeline,
      this.segmentedGradientReducePipeline,
    ] = await Promise.all([
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: renderModule, entryPoint: "render_state" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: renderModule, entryPoint: "render_state_tile" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: ssimModule, entryPoint: "ssim_horizontal" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: ssimModule, entryPoint: "ssim_vertical" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: ssimModule, entryPoint: "ssim_backward_horizontal" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: ssimModule, entryPoint: "ssim_backward_vertical" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: ssimModule, entryPoint: "alpha_ssim_tiles" } }),
      legacyGradientSupported
        ? this.device.createComputePipelineAsync({ layout: "auto", compute: { module: optimizerModule, entryPoint: "optimize" } })
        : Promise.resolve(null),
      legacyGradientSupported
        ? this.device.createComputePipelineAsync({ layout: "auto", compute: { module: optimizerModule, entryPoint: "optimize_parallel" } })
        : Promise.resolve(null),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: lossGradientModule, entryPoint: "loss_gradient" } }),
      this.device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: exactBackwardModule,
          entryPoint: this.quadExactBackwardEnabled ? "exact_alpha_backward_quad" : "exact_alpha_backward",
        },
      }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: exactBackwardModule, entryPoint: "exact_alpha_backward_source" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: virtualOrderPenaltyModule, entryPoint: "virtual_order_penalty" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: brushLocalColorFlowModule, entryPoint: "brush_local_color_flow" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: exactBackwardTelemetryModule, entryPoint: "measure_exact_backward" } }),
      this.device.createComputePipelineAsync({ layout: "auto", compute: { module: optimizerModule, entryPoint: "optimize_exact" } }),
      segmentedReferenceModule
        ? this.device.createComputePipelineAsync({ layout: "auto", compute: { module: segmentedReferenceModule, entryPoint: "count_references" } })
        : Promise.resolve(null),
      segmentedReferenceModule
        ? this.device.createComputePipelineAsync({ layout: "auto", compute: { module: segmentedReferenceModule, entryPoint: "prefix_references" } })
        : Promise.resolve(null),
      segmentedReferenceModule
        ? this.device.createComputePipelineAsync({ layout: "auto", compute: { module: segmentedReferenceModule, entryPoint: "fill_references" } })
        : Promise.resolve(null),
      segmentedGradientReduceModule
        ? this.device.createComputePipelineAsync({ layout: "auto", compute: { module: segmentedGradientReduceModule, entryPoint: "reduce_segmented_gradients" } })
        : Promise.resolve(null),
    ]);
    this.compiledSegmentedExactBackward = segmentedExactBackwardEnabled;
    this.compiledFixedPointExactGradient = fixedPointExactGradientEnabled;
    this.compiledInverseScaleOptimization = inverseScaleOptimizationEnabled;
  }

  async ensureTrainingResidualMapPipeline() {
    if (this.trainingResidualMapPipeline) return;
    const shader = `
struct Config { values: array<vec4<f32>, 8>, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> errorMap: array<f32>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn mean_rgb(value: vec3<f32>) -> f32 {
  return dot(value, vec3<f32>(1.0 / 3.0));
}
fn target_mean(pixel: u32) -> f32 {
  let base = pixel * 3u;
  return (targetRgb[base] + targetRgb[base + 1u] + targetRgb[base + 2u]) / 3.0;
}

@compute @workgroup_size(64)
fn update_training_residual(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let pixel = id.x;
  if (pixel >= width * height) { return; }
  let px = pixel % width;
  let py = pixel / width;
  let rendered = pixelState[pixel].rgb;
  let targetBase = pixel * 3u;
  let targetColor = vec3<f32>(
    targetRgb[targetBase],
    targetRgb[targetBase + 1u],
    targetRgb[targetBase + 2u]
  );
  let loss = (
    abs(rendered.r - targetColor.r) +
    abs(rendered.g - targetColor.g) +
    abs(rendered.b - targetColor.b)
  ) / 3.0;
  var gradientError = 0.0;
  var gradientCount = 0.0;
  let renderedMean = mean_rgb(rendered);
  let targetMean = mean_rgb(targetColor);
  if (px + 1u < width) {
    gradientError += abs(
      (mean_rgb(pixelState[pixel + 1u].rgb) - renderedMean) -
      (target_mean(pixel + 1u) - targetMean)
    );
    gradientCount += 1.0;
  }
  if (py + 1u < height) {
    gradientError += abs(
      (mean_rgb(pixelState[pixel + width].rgb) - renderedMean) -
      (target_mean(pixel + width) - targetMean)
    );
    gradientCount += 1.0;
  }
  errorMap[pixel] = loss + select(
    0.0,
    0.2 * gradientError / max(1.0, gradientCount),
    cfg(20u) > 0.5
  );
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.trainingResidualMapPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "update_training_residual" },
    });
  }

  async refreshTrainingResidualMap(image, params, { step = 0, reason = "density" } = {}) {
    if (!this.trainState?.errorMapBuffer) {
      throw new Error("Training residual map requires active WebGPU training state.");
    }
    await this.ensureTrainingResidualMapPipeline();
    if (this.runtime.tileCullingEnabled()) {
      await this.prepareTileLists(image, params, { sync: false });
    }
    await this.refreshRenderState(image, params, { computeSsim: false });
    const bindGroup = this.device.createBindGroup({
      layout: this.trainingResidualMapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.targetBuffer } },
        { binding: 2, resource: { buffer: this.trainState.pixelStateBuffer } },
        { binding: 3, resource: { buffer: this.trainState.errorMapBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.trainingResidualMapPipeline);
    pass.setBindGroup(0, bindGroup);
    this.dispatchLinear(pass, Math.ceil((image.width * image.height) / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.trainState.errorMapStep = Math.max(0, Math.round(step));
    this.trainState.errorMapReason = String(reason);
    return {
      step: this.trainState.errorMapStep,
      reason: this.trainState.errorMapReason,
      pixels: image.width * image.height,
      backend: "webgpu-training-signal",
      cpu_readback: false,
    };
  }

  async ensurePixelMetricsPipeline() {
    if (this.pixelMetricsPipeline) return;
    const shader = `
struct Config { values: array<vec4<f32>, 8>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metricsOut: array<f32>;
@group(0) @binding(4) var<storage, read> ssimData: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(6) var<storage, read> alphaState: array<AlphaState>;
var<workgroup> wgLoss: array<f32, 64>;
var<workgroup> wgSquaredError: array<f32, 64>;
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
var<workgroup> wgSsim: array<vec2<f32>, 64>;
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
  var squaredError = 0.0;
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
  var ssim = vec2<f32>(0.0);
  if (px < width && py < height) {
    let pixel = py * width + px;
    let rendered = pixelState[pixel].rgb;
    alphaError = abs(alphaState[pixel].compositeAlpha - targetAlpha[pixel]);
    coverage = pixelState[pixel].a;
    alphaMoments = vec4<f32>(coverage, targetAlpha[pixel], coverage * coverage, targetAlpha[pixel] * targetAlpha[pixel]);
    alphaCross = coverage * targetAlpha[pixel];
    let ssimChannels = ssimData[pixel * 4u + 3u];
    ssim = vec2<f32>((ssimChannels.r + ssimChannels.g + ssimChannels.b) / 3.0, ssimChannels.a);
    coverageUnder = select(0.0, 1.0, coverage < cfg(22u));
    backgroundExposure = select(0.0, 1.0, coverage < ${DEFAULT_ALPHA_TARGET});
    let targetIndex = pixel * 3u;
    let targetColor = vec3<f32>(targetRgb[targetIndex], targetRgb[targetIndex + 1u], targetRgb[targetIndex + 2u]);
    let residual = rendered - targetColor;
    loss = (abs(rendered.r - targetColor.r) + abs(rendered.g - targetColor.g) + abs(rendered.b - targetColor.b)) / 3.0;
    squaredError = dot(residual, residual);
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
    valid = 1.0;
  }
  wgLoss[lid.x] = loss;
  wgSquaredError[lid.x] = squaredError;
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
  wgSsim[lid.x] = ssim;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      wgLoss[lid.x] += wgLoss[lid.x + stride];
      wgSquaredError[lid.x] += wgSquaredError[lid.x + stride];
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
      wgSsim[lid.x] += wgSsim[lid.x + stride];
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
    metricsOut[out + 33u] = wgSquaredError[0];
    metricsOut[out + 34u] = wgSsim[0].x;
    metricsOut[out + 35u] = wgSsim[0].y;
  }
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.pixelMetricsPipeline = await this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "metrics" } });
  }

  async ensureVirtualCameraMetricsPipeline() {
    if (this.virtualCameraMetricsPipeline) return;
    const shader = `
struct Config { values: array<vec4<f32>, 19>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metricsOut: array<f32>;
@group(0) @binding(4) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(5) var<storage, read> alphaState: array<AlphaState>;
@group(0) @binding(6) var<storage, read> ssimData: array<vec4<f32>>;
var<workgroup> wgLoss: array<f32, 64>;
var<workgroup> wgSquaredError: array<f32, 64>;
var<workgroup> wgX: array<f32, 64>;
var<workgroup> wgY: array<f32, 64>;
var<workgroup> wgX2: array<f32, 64>;
var<workgroup> wgY2: array<f32, 64>;
var<workgroup> wgXY: array<f32, 64>;
var<workgroup> wgCount: array<f32, 64>;
var<workgroup> wgAlphaL1: array<f32, 64>;
var<workgroup> wgAlphaX: array<f32, 64>;
var<workgroup> wgAlphaY: array<f32, 64>;
var<workgroup> wgAlphaX2: array<f32, 64>;
var<workgroup> wgAlphaY2: array<f32, 64>;
var<workgroup> wgAlphaXY: array<f32, 64>;
var<workgroup> wgCoverage: array<f32, 64>;
var<workgroup> wgBackground: array<f32, 64>;
var<workgroup> wgRenderedChroma: array<f32, 64>;
var<workgroup> wgTargetChroma: array<f32, 64>;
var<workgroup> wgSsim: array<vec2<f32>, 64>;
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
  return mix(mix(targetAlpha[p0.y * width + p0.x], targetAlpha[p0.y * width + p1.x], f.x), mix(targetAlpha[p1.y * width + p0.x], targetAlpha[p1.y * width + p1.x], f.x), f.y);
}

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
  var squaredError = 0.0;
  var x = 0.0;
  var y = 0.0;
  var valid = 0.0;
  var alphaL1 = 0.0;
  var alphaX = 0.0;
  var alphaY = 0.0;
  var coverage = 0.0;
  var background = 0.0;
  var renderedChroma = 0.0;
  var targetChroma = 0.0;
  var ssim = vec2<f32>(0.0);
  if (px < width && py < height) {
    let gridPoint = vec2<f32>(
      select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
      select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
    );
    let sourcePoint = virtual_inverse_point(gridPoint);
    if (sourcePoint.z > 0.5) {
      let pixel = py * width + px;
      let rendered = pixelState[pixel].rgb;
      let targetColor = target_rgb_at(sourcePoint.xy, width, height);
      coverage = alphaState[pixel].compositeAlpha;
      alphaY = target_alpha_at(sourcePoint.xy, width, height);
      alphaX = coverage;
      alphaL1 = abs(alphaX - alphaY);
      background = select(0.0, 1.0, alphaX < ${DEFAULT_ALPHA_TARGET});
      let residual = rendered - targetColor;
      loss = (abs(rendered.r - targetColor.r) + abs(rendered.g - targetColor.g) + abs(rendered.b - targetColor.b)) / 3.0;
      squaredError = dot(residual, residual);
      x = dot(rendered, vec3<f32>(1.0 / 3.0));
      y = dot(targetColor, vec3<f32>(1.0 / 3.0));
      renderedChroma = max(rendered.r, max(rendered.g, rendered.b)) - min(rendered.r, min(rendered.g, rendered.b));
      targetChroma = max(targetColor.r, max(targetColor.g, targetColor.b)) - min(targetColor.r, min(targetColor.g, targetColor.b));
      let ssimChannels = ssimData[pixel * 4u + 3u];
      ssim = vec2<f32>((ssimChannels.r + ssimChannels.g + ssimChannels.b) / 3.0, ssimChannels.a);
      valid = 1.0;
    }
  }
  wgLoss[lid.x] = loss;
  wgSquaredError[lid.x] = squaredError;
  wgX[lid.x] = x;
  wgY[lid.x] = y;
  wgX2[lid.x] = x * x;
  wgY2[lid.x] = y * y;
  wgXY[lid.x] = x * y;
  wgCount[lid.x] = valid;
  wgAlphaL1[lid.x] = alphaL1;
  wgAlphaX[lid.x] = alphaX;
  wgAlphaY[lid.x] = alphaY;
  wgAlphaX2[lid.x] = alphaX * alphaX;
  wgAlphaY2[lid.x] = alphaY * alphaY;
  wgAlphaXY[lid.x] = alphaX * alphaY;
  wgCoverage[lid.x] = coverage;
  wgBackground[lid.x] = background;
  wgRenderedChroma[lid.x] = renderedChroma;
  wgTargetChroma[lid.x] = targetChroma;
  wgSsim[lid.x] = ssim;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      wgLoss[lid.x] += wgLoss[lid.x + stride];
      wgSquaredError[lid.x] += wgSquaredError[lid.x + stride];
      wgX[lid.x] += wgX[lid.x + stride];
      wgY[lid.x] += wgY[lid.x + stride];
      wgX2[lid.x] += wgX2[lid.x + stride];
      wgY2[lid.x] += wgY2[lid.x + stride];
      wgXY[lid.x] += wgXY[lid.x + stride];
      wgCount[lid.x] += wgCount[lid.x + stride];
      wgAlphaL1[lid.x] += wgAlphaL1[lid.x + stride];
      wgAlphaX[lid.x] += wgAlphaX[lid.x + stride];
      wgAlphaY[lid.x] += wgAlphaY[lid.x + stride];
      wgAlphaX2[lid.x] += wgAlphaX2[lid.x + stride];
      wgAlphaY2[lid.x] += wgAlphaY2[lid.x + stride];
      wgAlphaXY[lid.x] += wgAlphaXY[lid.x + stride];
      wgCoverage[lid.x] += wgCoverage[lid.x + stride];
      wgBackground[lid.x] += wgBackground[lid.x + stride];
      wgRenderedChroma[lid.x] += wgRenderedChroma[lid.x + stride];
      wgTargetChroma[lid.x] += wgTargetChroma[lid.x + stride];
      wgSsim[lid.x] += wgSsim[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let out = tileIndex * ${VIRTUAL_CAMERA_METRIC_TILE_STRIDE}u;
    metricsOut[out] = wgLoss[0];
    metricsOut[out + 1u] = wgX[0];
    metricsOut[out + 2u] = wgY[0];
    metricsOut[out + 3u] = wgX2[0];
    metricsOut[out + 4u] = wgY2[0];
    metricsOut[out + 5u] = wgXY[0];
    metricsOut[out + 6u] = wgCount[0];
    metricsOut[out + 7u] = wgAlphaL1[0];
    metricsOut[out + 8u] = wgAlphaX[0];
    metricsOut[out + 9u] = wgAlphaY[0];
    metricsOut[out + 10u] = wgAlphaX2[0];
    metricsOut[out + 11u] = wgAlphaY2[0];
    metricsOut[out + 12u] = wgAlphaXY[0];
    metricsOut[out + 13u] = wgCoverage[0];
    metricsOut[out + 14u] = wgBackground[0];
    metricsOut[out + 15u] = wgSquaredError[0];
    metricsOut[out + 16u] = wgRenderedChroma[0];
    metricsOut[out + 17u] = wgTargetChroma[0];
    metricsOut[out + 18u] = wgSsim[0].x;
    metricsOut[out + 19u] = wgSsim[0].y;
  }
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.virtualCameraMetricsPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "metrics" },
    });
  }

  async ensureOverlapMetricsPipeline({ hiddenRgb = false } = {}) {
    const pipelineKey = hiddenRgb ? "overlapMetricsQaPipeline" : "overlapMetricsPipeline";
    if (this[pipelineKey]) return this[pipelineKey];
    if (hiddenRgb && Number(this.device.limits?.maxStorageBuffersPerShaderStage || 8) < 9) return null;
    const hiddenRgbBinding = hiddenRgb
      ? "@group(0) @binding(9) var<storage, read_write> hiddenRgbAttribution: array<atomic<u32>>;"
      : "";
    const hiddenRgbShader = hiddenRgb ? `
    if (abs(scaleFactor - 0.25) < 0.0001 && !virtual_tilt_enabled() && validPixel) {
      var suffixTransmittance = 1.0;
      var suffixColor = vec3<f32>(cfg(3u), cfg(4u), cfg(5u));
      var reverse = acceptedEnd;
      loop {
        if (reverse <= start) { break; }
        reverse -= 1u;
        var g = reverse;
        if (useTiles) { g = tileIndices[reverse]; }
        let t = transform[g];
        if (t.w < 0.5) { continue; }
        let center = xy[g].center;
        let c = cos(t.z);
        let s = sin(t.z);
        let samplePoint = virtual_inverse_point_at_z(outputPoint, virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy;
        let baseScale = max(t.xy * scaleFactor, vec2<f32>(0.0001));
        let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
        var kernel = gaussian_kernel(samplePoint - center, c, s, effective);
        var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
        if (cfg(26u) > 0.5) {
          let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
          let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
          let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
          kernel = 0.25 * (
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale)
          );
          mip = 1.0;
        }
        let weight = clamp(kernel * color[g].a * mip, 0.0, 0.99);
        if (kernel < 0.0003354626 || weight < 0.0039215686) { continue; }
        let denominator = max(0.00000001, (1.0 - weight) * suffixTransmittance);
        let transmittanceBefore = clamp(transmittance / denominator, 0.0, 1.0);
        let contribution = transmittanceBefore * weight;
        let withoutRgb = rendered + contribution * (suffixColor - color[g].rgb);
        let withoutError = dot(abs(withoutRgb - targetColor), vec3<f32>(1.0 / 3.0));
        let positiveHarm = max(0.0, l1 - withoutError);
        let attributionQuantization = min(
          HIDDEN_RGB_ATTRIBUTION_QUANTIZATION,
          floor(4000000000.0 / max(1.0, f32(width) * f32(height) * 0.25))
        );
        let out = g * HIDDEN_RGB_ATTRIBUTION_STRIDE;
        atomicAdd(&hiddenRgbAttribution[out], u32(round(clamp(positiveHarm, 0.0, 0.25) * attributionQuantization)));
        atomicAdd(&hiddenRgbAttribution[out + 1u], u32(round(clamp(contribution, 0.0, 0.25) * attributionQuantization)));
        atomicMax(&hiddenRgbAttribution[out + 2u], u32(round(clamp(positiveHarm, 0.0, 0.25) * attributionQuantization)));
        suffixColor = weight * color[g].rgb + (1.0 - weight) * suffixColor;
        suffixTransmittance *= 1.0 - weight;
      }
      if (transmittance < 0.0001) {
        var deepCursor = acceptedEnd;
        loop {
          if (deepCursor >= end) { break; }
          var deepG = deepCursor;
          if (useTiles) { deepG = tileIndices[deepCursor]; }
          deepCursor += 1u;
          let deepT = transform[deepG];
          if (deepT.w < 0.5) { continue; }
          let deepCenter = xy[deepG].center;
          let deepC = cos(deepT.z);
          let deepS = sin(deepT.z);
          let deepPoint = virtual_inverse_point_at_z(outputPoint, virtual_pass_layer_depth(deepT.w, xy[deepG].rawDepth)).xy;
          let deepBaseScale = max(deepT.xy * scaleFactor, vec2<f32>(0.0001));
          let deepEffective = sqrt(deepBaseScale * deepBaseScale + vec2<f32>(pixelSigma * pixelSigma));
          var deepKernel = gaussian_kernel(deepPoint - deepCenter, deepC, deepS, deepEffective);
          var deepMip = sqrt((deepBaseScale.x * deepBaseScale.y) / max(deepEffective.x * deepEffective.y, 0.00000001));
          if (cfg(26u) > 0.5) {
            let deepSampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
            let deepOx = select(0.0, deepSampleOffset / f32(width - 1u), width > 1u);
            let deepOy = select(0.0, deepSampleOffset / f32(height - 1u), height > 1u);
            deepKernel = 0.25 * (
              gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-deepOx, -deepOy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(deepT.w, xy[deepG].rawDepth)).xy - deepCenter, deepC, deepS, deepBaseScale) +
              gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( deepOx, -deepOy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(deepT.w, xy[deepG].rawDepth)).xy - deepCenter, deepC, deepS, deepBaseScale) +
              gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-deepOx,  deepOy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(deepT.w, xy[deepG].rawDepth)).xy - deepCenter, deepC, deepS, deepBaseScale) +
              gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( deepOx,  deepOy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(deepT.w, xy[deepG].rawDepth)).xy - deepCenter, deepC, deepS, deepBaseScale)
            );
            deepMip = 1.0;
          }
          let deepWeight = clamp(deepKernel * color[deepG].a * deepMip, 0.0, 0.99);
          if (deepKernel < 0.0003354626 || deepWeight < 0.0039215686) { continue; }
          let deepQuantization = min(
            HIDDEN_RGB_ATTRIBUTION_QUANTIZATION,
            floor(4000000000.0 / max(1.0, f32(width) * f32(height) * 0.25))
          );
          let deepOut = deepG * HIDDEN_RGB_ATTRIBUTION_STRIDE;
          atomicAdd(&hiddenRgbAttribution[deepOut + 3u], 1u);
          atomicAdd(&hiddenRgbAttribution[deepOut + 4u], u32(round(clamp(deepWeight, 0.0, 0.25) * deepQuantization)));
        }
      }
    }` : "";
    const shader = `
struct Config { values: array<vec4<f32>, 19>, };
@group(0) @binding(0) var<uniform> config: Config;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(5) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(6) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(7) var<storage, read> tileIndices: array<u32>;
@group(0) @binding(8) var<storage, read_write> metricsOut: array<f32>;
${hiddenRgbBinding}
const HIDDEN_RGB_ATTRIBUTION_STRIDE = 5u;
const HIDDEN_RGB_ATTRIBUTION_QUANTIZATION = 4096.0;
var<workgroup> reduceA: array<vec4<f32>, 64>;
var<workgroup> reduceB: array<vec4<f32>, 64>;
var<workgroup> reduceC: array<vec4<f32>, 64>;
var<workgroup> reduceD: array<vec4<f32>, 64>;
var<workgroup> reduceOrder: array<vec4<f32>, 64>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
${VIRTUAL_TILT_WGSL}

fn gaussian_kernel(d: vec2<f32>, c: f32, s: f32, scale: vec2<f32>) -> f32 {
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  let q = dot(r / scale, r / scale);
  return exp(-0.5 * q);
}

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
  var orderOut = vec4<f32>(0.0);
  if (px < width && py < height) {
    let pixel = py * width + px;
    let outputPoint = vec2<f32>(
      select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
      select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
    );
    let inversePoint = virtual_inverse_point(outputPoint);
    let targetColor = target_rgb_at(inversePoint.xy, width, height);
    let targetAlphaValue = target_alpha_at(inversePoint.xy, width, height);
    let useTiles = cfg(19u) > 0.5;
    let tileCols = (width + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
    let tile = (py / ${TILE_SIZE}u) * tileCols + (px / ${TILE_SIZE}u);
    let capacity = arrayLength(&tileIndices);
    let start = select(0u, min(tileOffsets[tile] & 0x7fffffffu, capacity), useTiles);
    let end = select(u32(cfg(2u)), min(tileOffsets[tile + 1u] & 0x7fffffffu, capacity), useTiles);
    let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
    let scaleFactor = clamp(cfg(63u), 0.01, 1.0);
    var numerator = vec3<f32>(0.0);
    var colorSecond = vec3<f32>(0.0);
    var denom = 0.0;
    var sumW2 = 0.0;
    var maxW = 0.0;
    var sumWLogW = 0.0;
    var targetDistance = 0.0;
    var transmittance = 1.0;
    var compositedRgb = vec3<f32>(0.0);
    var previousFrontOrder = 0.0;
    var hasPreviousFrontOrder = false;
    var adjacentOrderPairs = 0.0;
    var adjacentOrderFlips = 0.0;
    var acceptedEnd = start;
    var cursor = start;
    loop {
      if (cursor >= end) { break; }
      var g = cursor;
      if (useTiles) { g = tileIndices[cursor]; }
      let t = transform[g];
      if (t.w >= 0.5) {
        let center = xy[g].center;
        let c = cos(t.z);
        let s = sin(t.z);
        let samplePoint = virtual_inverse_point_at_z(outputPoint, virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy;
        let baseScale = max(t.xy * scaleFactor, vec2<f32>(0.0001));
        let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
        var kernel = gaussian_kernel(samplePoint - center, c, s, effective);
        var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
        if (cfg(26u) > 0.5) {
          let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
          let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
          let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
          kernel = 0.25 * (
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale)
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
            let frontOrder = fract(t.w);
            if (hasPreviousFrontOrder) {
              adjacentOrderPairs += 1.0;
              adjacentOrderFlips += select(0.0, 1.0, frontOrder > previousFrontOrder + 0.0000001);
            }
            previousFrontOrder = frontOrder;
            hasPreviousFrontOrder = true;
            compositedRgb += transmittance * weight * rgb;
            transmittance *= 1.0 - weight;
            acceptedEnd = cursor + 1u;
          }
        }
      }
      cursor += 1u;
    }
    let validPixel = inversePoint.z > 0.5;
    let covered = denom > ${BACKGROUND_EXPOSURE_EPSILON} && validPixel;
    let weightedMean = select(vec3<f32>(0.0), numerator / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered);
    let rendered = compositedRgb + transmittance * vec3<f32>(cfg(3u), cfg(4u), cfg(5u));
    let variance = max(vec3<f32>(0.0), colorSecond / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}) - weightedMean * weightedMean);
    let effectiveContributors = select(0.0, denom * denom / max(sumW2, 0.0000000000000001), covered);
    let maxShare = select(0.0, maxW / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered);
    let entropy = select(0.0, max(0.0, log(max(denom, 0.00000001)) - sumWLogW / max(denom, ${BACKGROUND_EXPOSURE_EPSILON})), covered);
    let alpha = 1.0 - transmittance;
    let rgbError = select(vec3<f32>(0.0), abs(rendered - targetColor), validPixel);
    let l1 = dot(rgbError, vec3<f32>(1.0 / 3.0));
    let maxChannel = max(rgbError.r, max(rgbError.g, rgbError.b));
${hiddenRgbShader}
    a = vec4<f32>(select(0.0, 1.0, validPixel), select(0.0, denom, validPixel), effectiveContributors, maxShare);
    b = vec4<f32>(entropy, select(0.0, alpha, validPixel), select(0.0, abs(alpha - targetAlphaValue), validPixel), dot(variance, vec3<f32>(1.0 / 3.0)));
    cc = select(vec4<f32>(0.0), vec4<f32>(select(0.0, targetDistance / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered), l1, maxChannel, select(0.0, 1.0, maxChannel > 0.10)), validPixel);
    dOut = select(vec4<f32>(0.0), vec4<f32>(select(0.0, 1.0, alpha < ${DEFAULT_ALPHA_TARGET}), sumW2, maxW, maxChannel), validPixel);
    orderOut = select(vec4<f32>(0.0), vec4<f32>(adjacentOrderPairs, adjacentOrderFlips, select(0.0, 1.0, covered), 0.0), validPixel);
  }
  reduceA[lid.x] = a;
  reduceB[lid.x] = b;
  reduceC[lid.x] = cc;
  reduceD[lid.x] = dOut;
  reduceOrder[lid.x] = orderOut;
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
      reduceOrder[lid.x] += reduceOrder[lid.x + stride];
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
    metricsOut[out + 16u] = reduceOrder[0].x;
    metricsOut[out + 17u] = reduceOrder[0].y;
    metricsOut[out + 18u] = reduceOrder[0].z;
  }
}`;
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this[pipelineKey] = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "overlap_metrics" },
    });
    return this[pipelineKey];
  }

  async ensureAlphaLossPipeline() {
    if (this.alphaLossPipeline) return;
    const shader = `
struct Config { values: array<vec4<f32>, 12>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
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
    let center = xy[g].center;
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

  async computeOverlapDiagnostics(
    image,
    params,
    { views = null, scales = [1, 0.5, 0.25], hiddenRgb = false } = {},
  ) {
    if (!this.trainState || !phase40Variants().overlapDiagnostics) return null;
    const partialCount = Math.ceil(image.width / 8) * Math.ceil(image.height / 8);
    const outputBytes = partialCount * OVERLAP_METRIC_STRIDE * 4;
    const hiddenRgbRequested = (hiddenRgb || hiddenRgbAttributionRequested()) && !views?.length;
    const bindingPlan = overlapDiagnosticsBindingPlan(
      this.device.limits?.maxStorageBuffersPerShaderStage,
      hiddenRgbRequested,
    );
    const hiddenRgbSupported = !bindingPlan.hiddenRgbUnavailable;
    const hiddenRgbEnabled = bindingPlan.hiddenRgbEnabled;
    const overlapMetricsPipeline = await this.ensureOverlapMetricsPipeline({ hiddenRgb: hiddenRgbEnabled });
    if (!overlapMetricsPipeline) throw new Error("Hidden RGB attribution pipeline unavailable on this WebGPU device.");
    const hiddenRgbStride = 5;
    const hiddenRgbBytes = Math.max(hiddenRgbStride * 4, params.count * hiddenRgbStride * 4);
    const configBuffer = this.device.createBuffer({ size: TRAIN_CONFIG_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const outputBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const hiddenRgbBuffer = hiddenRgbEnabled
      ? this.device.createBuffer({
          size: hiddenRgbBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
      : null;
    const hiddenRgbReadBuffer = hiddenRgbEnabled
      ? this.device.createBuffer({ size: hiddenRgbBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
      : null;
    const front = this.trainState.front;
    const summarize = (values, scale) => {
      const totals = new Float64Array(OVERLAP_METRIC_STRIDE);
      let maxChannelError = 0;
      for (let i = 0; i < values.length; i += OVERLAP_METRIC_STRIDE) {
        for (let j = 0; j < OVERLAP_METRIC_STRIDE; j += 1) {
          if (j !== 15) totals[j] += values[i + j];
        }
        maxChannelError = Math.max(maxChannelError, values[i + 15]);
      }
      const pixels = Math.max(1, totals[0]);
      const coveredPixels = Math.max(0, totals[0] - totals[12]);
      const contributingPixels = Math.max(1, totals[18]);
      return {
        scale,
        pixels: totals[0],
        covered_pixels: totals[0] - totals[12],
        covered_ratio: (totals[0] - totals[12]) / pixels,
        mean_weight_sum: totals[1] / pixels,
        contributing_pixels: totals[18],
        contributing_ratio: totals[18] / pixels,
        mean_effective_contributors: totals[2] / contributingPixels,
        mean_max_weight_share: totals[3] / contributingPixels,
        mean_weight_entropy: totals[4] / contributingPixels,
        mean_composited_alpha: totals[5] / pixels,
        mean_alpha_error: totals[6] / pixels,
        mean_contributor_color_variance: totals[7] / contributingPixels,
        mean_contributor_target_distance: totals[8] / contributingPixels,
        l1: totals[9] / pixels,
        mean_max_channel_error: totals[10] / pixels,
        bad_pixel_count_0_10: totals[11],
        bad_pixel_ratio_0_10: totals[11] / pixels,
        background_exposure_count: totals[12],
        mean_weight_square_sum: totals[13] / pixels,
        mean_max_weight: totals[14] / pixels,
        max_channel_error: maxChannelError,
        adjacent_order_pair_count: totals[16],
        adjacent_order_flip_count: totals[17],
        adjacent_order_flip_ratio: totals[16] > 0 ? totals[17] / totals[16] : 0,
      };
    };
    const viewSpecs = views?.length ? views : [{ key: "front", pitchDegrees: 0, yawDegrees: 0 }];
    const buildConfig = async (view, scale) => {
      const pitchDegrees = Number(view.pitchDegrees) || 0;
      const yawDegrees = Number(view.yawDegrees) || 0;
      const enabled = Math.abs(pitchDegrees) > 0.0001 || Math.abs(yawDegrees) > 0.0001;
      const fovDegrees = clampSharedCameraFov(
        view.fovDegrees || state.metrics?.virtual_camera_sampling?.fov_degrees,
      );
      const cameraDistance = enabled
        ? Number(view.cameraDistance) || sharedTiltOrbitRadius(
          image.width,
          image.height,
          state.metrics?.virtual_camera_sampling?.max_angle_degrees,
          49,
          fovDegrees,
        )
        : DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE;
      const config = new Float32Array(TRAIN_CONFIG_FLOATS);
      config.set([image.width, image.height, params.count, params.bg[0], params.bg[1], params.bg[2]], 0);
      config[19] = this.runtime.tileCullingEnabled() ? 1 : 0;
      config[26] = phase33Variants().ewa2x2 ? 1 : 0;
      config[31] = phase37Variants().ewaGaussLegendre ? 1 : 0;
      config[45] = this.runtime.trainLayerOrderEnabled() ? 1 : 0;
      config[56] = enabled ? 1 : 0;
      config[57] = pitchDegrees * Math.PI / 180;
      config[58] = yawDegrees * Math.PI / 180;
      config[59] = cameraDistance;
      config[63] = scale;
      config[64] = fovDegrees;
      config[67] = params.virtualDepthEnabled ? 1 : 0;
      config[68] = params.virtualDepthEnabled ? Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS : 0;
      config[81] = params.discreteLayersEnabled ? 1 : 0;
      config[82] = Math.max(MIN_DISCRETE_LAYER_COUNT, Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(params.discreteLayerCount || DEFAULT_DISCRETE_LAYER_COUNT)));
      config[84] = params.layerAwareAccumulationEnabled ? 1 : 0;
      configurePaintKernel(config, params);
      return { config, pitchDegrees, yawDegrees, cameraDistance, fovDegrees };
    };
    let operationError = null;
    let hiddenRgbAttribution = null;
    try {
      const results = {};
      for (const view of viewSpecs) {
        const viewResults = {};
        for (const scale of scales) {
        const { config, pitchDegrees, yawDegrees, cameraDistance } = await buildConfig(view, scale);
        this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
        if (this.runtime.tileCullingEnabled()) {
          await this.prepareTileLists(image, params, {
            sync: true,
            writeConfig: false,
          });
        }
        this.device.queue.writeBuffer(configBuffer, 0, config);
        const entries = [
            { binding: 0, resource: { buffer: configBuffer } },
            { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
            { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
            { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
            { binding: 4, resource: { buffer: this.trainState.targetBuffer } },
            { binding: 5, resource: { buffer: this.trainState.targetAlphaBuffer } },
            { binding: 6, resource: { buffer: this.trainState.tileOffsetsBuffer } },
            { binding: 7, resource: { buffer: this.trainState.tileIndicesBuffer } },
            { binding: 8, resource: { buffer: outputBuffer } },
          ];
        if (hiddenRgbEnabled) entries.push({ binding: 9, resource: { buffer: hiddenRgbBuffer } });
        const bindGroup = this.device.createBindGroup({
          layout: overlapMetricsPipeline.getBindGroupLayout(0),
          entries,
        });
        const encoder = this.device.createCommandEncoder();
        const attributionPass = hiddenRgbEnabled && Math.abs(scale - 0.25) < 0.0001 && Math.abs(pitchDegrees) < 0.0001 && Math.abs(yawDegrees) < 0.0001;
        if (attributionPass) encoder.clearBuffer(hiddenRgbBuffer);
        const pass = encoder.beginComputePass();
        pass.setPipeline(overlapMetricsPipeline);
        pass.setBindGroup(0, bindGroup);
        this.dispatchLinear(pass, partialCount);
        pass.end();
        encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBytes);
        if (attributionPass) encoder.copyBufferToBuffer(hiddenRgbBuffer, 0, hiddenRgbReadBuffer, 0, hiddenRgbBytes);
        this.device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(readBuffer.getMappedRange()).slice();
        readBuffer.unmap();
        if (attributionPass) {
          await hiddenRgbReadBuffer.mapAsync(GPUMapMode.READ);
          const attributionValues = new Uint32Array(hiddenRgbReadBuffer.getMappedRange()).slice();
          hiddenRgbReadBuffer.unmap();
          const initialCount = Number(state.metrics?.initial_splats) || 0;
          const attributionQuantization = Math.min(
            4096,
            Math.floor(4_000_000_000 / Math.max(1, image.width * image.height * 0.25)),
          );
          const rows = [];
          for (let index = 0; index < params.count; index += 1) {
            const attributionBase = index * hiddenRgbStride;
            const positiveHarm = attributionValues[attributionBase] / attributionQuantization;
            const deepPixelHits = attributionValues[attributionBase + 3];
            if (positiveHarm <= 0 && deepPixelHits <= 0) continue;
            const centerX = params.xy[index * 2];
            const centerY = params.xy[index * 2 + 1];
            rows.push({
              index,
              positive_leave_one_out_l1_harm: positiveHarm,
              contribution_mass: attributionValues[attributionBase + 1] / attributionQuantization,
              maximum_pixel_harm: attributionValues[attributionBase + 2] / attributionQuantization,
              deep_occluded_pixel_hits: deepPixelHits,
              deep_occluded_weight_mass: attributionValues[attributionBase + 4] / attributionQuantization,
              initial_cohort: index < initialCount,
              center: [centerX, centerY],
              region_16x16: [
                Math.min(15, Math.max(0, Math.floor((centerX * 0.5 + 0.5) * 16))),
                Math.min(15, Math.max(0, Math.floor((centerY * 0.5 + 0.5) * 16))),
              ],
              rgb: Array.from(params.rgb.slice(index * 3, index * 3 + 3)),
              opacity: params.opacity[index],
            });
          }
          rows.sort((a, b) => b.positive_leave_one_out_l1_harm - a.positive_leave_one_out_l1_harm || a.index - b.index);
          const top = rows.slice(0, 32);
          hiddenRgbAttribution = {
            backend: "webgpu-final-only",
            scale: 0.25,
            metric: "positive exact leave-one-out RGB L1 harm",
            quantization: attributionQuantization,
            attributed_splat_count: rows.filter((row) => row.positive_leave_one_out_l1_harm > 0).length,
            deep_occluded_splat_count: rows.filter((row) => row.deep_occluded_pixel_hits > 0).length,
            deep_occluded_pixel_hits: rows.reduce((sum, row) => sum + row.deep_occluded_pixel_hits, 0),
            total_positive_harm: rows.reduce((sum, row) => sum + row.positive_leave_one_out_l1_harm, 0),
            top_32_positive_harm: top.reduce((sum, row) => sum + row.positive_leave_one_out_l1_harm, 0),
            top_initial_cohort_fraction: top.filter((row) => row.initial_cohort).length / Math.max(1, top.length),
            fingerprint: top.slice(0, 12).map((row) => `${row.index}:${row.region_16x16.join(",")}`).join("|"),
            top,
            rows,
          };
        }
        viewResults[String(scale)] = {
          ...summarize(values, scale),
          pitch_degrees: pitchDegrees,
          yaw_degrees: yawDegrees,
          camera_distance: cameraDistance,
        };
        }
        results[String(view.key || `pitch-${Number(view.pitchDegrees) || 0}-yaw-${Number(view.yawDegrees) || 0}`)] = viewResults;
      }
      const report = {
        backend: "webgpu-final-only",
        standard_alpha_blend: true,
        source_alpha_preserved: Boolean(image.alpha),
        views: results,
      };
      if (!views?.length) report.scales = results.front;
      if (hiddenRgbAttribution) report.hidden_rgb_attribution = hiddenRgbAttribution;
      if (hiddenRgbRequested && !hiddenRgbSupported) {
        report.hidden_rgb_attribution_unavailable = {
          reason: "maxStorageBuffersPerShaderStage < 9",
          available_storage_buffers: bindingPlan.availableStorageBuffers,
          required_storage_buffers: 9,
        };
      }
      return report;
    } catch (error) {
      operationError = error;
      throw error;
    } finally {
      let restorationError = null;
      try {
        const { config } = await buildConfig({ pitchDegrees: 0, yawDegrees: 0 }, 1);
        this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
        if (this.runtime.tileCullingEnabled()) {
          await this.prepareTileLists(image, params, { sync: true, writeConfig: false });
        }
        await this.refreshRenderState(image, params);
      } catch (error) {
        restorationError = error;
      } finally {
        configBuffer.destroy();
        outputBuffer.destroy();
        readBuffer.destroy();
        hiddenRgbBuffer?.destroy();
        hiddenRgbReadBuffer?.destroy();
      }
      if (restorationError) {
        if (operationError) {
          console.warn("Overlap diagnostics render-state restoration also failed", restorationError);
        } else {
          throw restorationError;
        }
      }
    }
  }

  async computeObliqueDiagnostics(image, params) {
    const views = [{ key: "front", pitchDegrees: 0, yawDegrees: 0 }];
    const trainingOrbitRadius = Number(state.metrics?.virtual_camera_sampling?.orbit_radius);
    const trainingFovDegrees = clampSharedCameraFov(state.metrics?.virtual_camera_sampling?.fov_degrees);
    const cameraDistance = Number.isFinite(trainingOrbitRadius) && trainingOrbitRadius > 0
      ? trainingOrbitRadius
      : sharedTiltOrbitRadius(
        image.width,
        image.height,
        state.metrics?.virtual_camera_sampling?.max_angle_degrees,
        49,
        trainingFovDegrees,
      );
    for (const angle of [15, 30, 45, 60, 75, 89]) {
      views.push(
        { key: `pitch-${angle}`, pitchDegrees: angle, yawDegrees: 0, cameraDistance, fovDegrees: trainingFovDegrees },
        { key: `yaw-${angle}`, pitchDegrees: 0, yawDegrees: angle, cameraDistance, fovDegrees: trainingFovDegrees },
      );
    }
    return this.computeOverlapDiagnostics(image, params, { views, scales: [1] });
  }

  async refreshRenderState(
    image,
    params,
    { view = null, computeSsim = true, contributionMoments = false } = {},
  ) {
    await this.ensureRenderGradientPipelines();
    const variants = phase33Variants();
    const requestedView = view === "front" ? null : view;
    const pitchDegrees = Number(requestedView?.pitchDegrees ?? requestedView?.pitch_degrees) || 0;
    const yawDegrees = Number(requestedView?.yawDegrees ?? requestedView?.yaw_degrees) || 0;
    const virtualView = Boolean(requestedView) && (Math.abs(pitchDegrees) > 0.0001 || Math.abs(yawDegrees) > 0.0001);
    const fovDegrees = clampSharedCameraFov(
      requestedView?.fovDegrees ?? requestedView?.fov_degrees ?? state.metrics?.virtual_camera_sampling?.fov_degrees,
    );
    const cameraDistance = virtualView
      ? Number(requestedView?.cameraDistance) || Number(state.metrics?.virtual_camera_sampling?.orbit_radius) || sharedTiltOrbitRadius(
        image.width,
        image.height,
        state.metrics?.virtual_camera_sampling?.max_angle_degrees,
        state.metrics?.virtual_camera_sampling?.virtual_camera_count,
        fovDegrees,
      )
      : DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE;
    const config = new Float32Array(TRAIN_CONFIG_FLOATS);
    config.set([image.width, image.height, params.count, params.bg[0], params.bg[1], params.bg[2]], 0);
    config[17] = currentMaxAnisotropy();
    config[18] = experimentalDensifySteps(state.metrics?.steps_requested || 1);
    config[19] = this.runtime.tileCullingEnabled() ? 1 : 0;
    config[20] = phase37Variants().structuralErrorMap ? 1 : 0;
    config[22] = variants.coverageTarget;
    config[26] = variants.ewa2x2 ? 1 : 0;
    config[31] = phase37Variants().ewaGaussLegendre ? 1 : 0;
    config[45] = params.layerOrderEnabled ? 1 : 0;
    // Density passes use slot 47 for recipient scoring.  A final-only Brush QA
    // render reuses it as a read-only contribution-moment switch; ordinary
    // training/front refreshes always clear it.
    config[47] = contributionMoments ? 1 : 0;
    config[56] = virtualView ? 1 : 0;
    config[57] = pitchDegrees * Math.PI / 180;
    config[58] = yawDegrees * Math.PI / 180;
    config[59] = cameraDistance;
    config[64] = virtualView ? fovDegrees : DEFAULT_SHARED_CAMERA_FOV_DEGREES;
    config[65] = virtualView && requestedView?.planeConstrained !== false ? 1 : 0;
    config[67] = params.virtualDepthEnabled ? 1 : 0;
    config[68] = params.virtualDepthEnabled
      ? Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS
      : 0;
    config[74] = virtualView && requestedView?.cameraCovariance3d ? 1 : 0;
    config[81] = params.discreteLayersEnabled ? 1 : 0;
    config[82] = Math.max(MIN_DISCRETE_LAYER_COUNT, Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(params.discreteLayerCount || DEFAULT_DISCRETE_LAYER_COUNT)));
    config[84] = params.layerAwareAccumulationEnabled ? 1 : 0;
    configurePaintKernel(config, params);
    this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    if (virtualView && this.runtime.tileCullingEnabled()) {
      await this.prepareTileLists(image, params, {
        sync: true,
        writeConfig: false,
      });
    }
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
    const ssimFullEntries = () => [
            { binding: 0, resource: { buffer: this.trainState.configBuffer } },
            { binding: 1, resource: { buffer: targetBuffer } },
            { binding: 2, resource: { buffer: this.trainState.pixelStateBuffer } },
            { binding: 3, resource: { buffer: this.trainState.ssimTileBuffer } },
            { binding: 4, resource: { buffer: targetAlphaBuffer } },
            { binding: 5, resource: { buffer: this.trainState.alphaStateBuffer } },
          ];
    const ssimFilterEntries = () => [
      { binding: 0, resource: { buffer: this.trainState.configBuffer } },
      { binding: 3, resource: { buffer: this.trainState.ssimTileBuffer } },
    ];
    const ssimBindGroups = computeSsim
      ? [
          [this.ssimHorizontalPipeline, ssimFullEntries],
          [this.ssimTilePipeline, ssimFilterEntries],
          [this.ssimBackwardHorizontalPipeline, ssimFilterEntries],
          [this.ssimBackwardVerticalPipeline, ssimFullEntries],
          [this.alphaSsimTilePipeline, ssimFullEntries],
        ].map(([pipeline, entries]) => this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: entries(),
        }))
      : null;
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
    if (ssimBindGroups) {
      const horizontalPass = encoder.beginComputePass();
      horizontalPass.setPipeline(this.ssimHorizontalPipeline);
      horizontalPass.setBindGroup(0, ssimBindGroups[0]);
      this.dispatchLinear(horizontalPass, Math.ceil((image.width * image.height) / 64));
      horizontalPass.end();
      const verticalPass = encoder.beginComputePass();
      verticalPass.setPipeline(this.ssimTilePipeline);
      verticalPass.setBindGroup(0, ssimBindGroups[1]);
      this.dispatchLinear(verticalPass, Math.ceil((image.width * image.height) / 64));
      verticalPass.end();
      const backwardHorizontalPass = encoder.beginComputePass();
      backwardHorizontalPass.setPipeline(this.ssimBackwardHorizontalPipeline);
      backwardHorizontalPass.setBindGroup(0, ssimBindGroups[2]);
      this.dispatchLinear(backwardHorizontalPass, Math.ceil((image.width * image.height) / 64));
      backwardHorizontalPass.end();
      const backwardVerticalPass = encoder.beginComputePass();
      backwardVerticalPass.setPipeline(this.ssimBackwardVerticalPipeline);
      backwardVerticalPass.setBindGroup(0, ssimBindGroups[3]);
      this.dispatchLinear(backwardVerticalPass, Math.ceil((image.width * image.height) / 64));
      backwardVerticalPass.end();
      const alphaSsimPass = encoder.beginComputePass();
      alphaSsimPass.setPipeline(this.alphaSsimTilePipeline);
      alphaSsimPass.setBindGroup(0, ssimBindGroups[4]);
      alphaSsimPass.dispatchWorkgroups(Math.ceil(image.width / 8), Math.ceil(image.height / 8));
      alphaSsimPass.end();
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    this.trainState.pixelStateResolution = [image.width, image.height];
    this.trainState.pixelStateKind = stageKind;
    this.trainState.pixelStateViewKey = virtualView ? "virtual" : "front";
  }

  async computeBrushContributionDiagnostics(image, params, settings) {
    if (!settings?.enabled || normalizedKernelShape(params.kernelShape) !== "opaque-brush") return null;
    const parameterHashBefore = hashParams(params);
    const pixelCount = image.width * image.height;
    const readBytes = Math.max(ALPHA_STATE_BYTES_PER_PIXEL, pixelCount * ALPHA_STATE_BYTES_PER_PIXEL);
    const readBuffer = this.device.createBuffer({
      size: readBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let operationError = null;
    let report = null;
    try {
      if (this.runtime.tileCullingEnabled()) {
        await this.prepareTileLists(image, params, { sync: true });
      }
      await this.refreshRenderState(image, params, {
        computeSsim: false,
        contributionMoments: true,
      });
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(
        this.trainState.alphaStateBuffer,
        0,
        readBuffer,
        0,
        readBytes,
      );
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const linearLuma = new Float32Array(pixelCount);
      for (let pixel = 0; pixel < pixelCount; pixel += 1) {
        const rgb = pixel * 3;
        linearLuma[pixel] =
          0.2126729 * srgbSignalToLinear(image.rgb[rgb]) +
          0.7151522 * srgbSignalToLinear(image.rgb[rgb + 1]) +
          0.0721750 * srgbSignalToLinear(image.rgb[rgb + 2]);
      }
      const flatThreshold = settings.flatLinearGradient;
      let validPixels = 0;
      let flatPixels = 0;
      let flatContributingPixels = 0;
      let highNeffPixels = 0;
      let contributorCountTotal = 0;
      let effectiveContributorTotal = 0;
      let renyi2EntropyTotal = 0;
      let validCompositeAlphaTotal = 0;
      let flatContributingAlphaTotal = 0;
      for (let pixel = 0; pixel < pixelCount; pixel += 1) {
        const targetAlpha = image.alpha?.[pixel] ?? 1;
        if (!(targetAlpha >= 0.5)) continue;
        validPixels += 1;
        const base = pixel * 4;
        const compositeAlpha = floats[base];
        if (Number.isFinite(compositeAlpha) && compositeAlpha >= 0) {
          validCompositeAlphaTotal += compositeAlpha;
        }
        const x = pixel % image.width;
        const y = Math.floor(pixel / image.width);
        const center = linearLuma[pixel];
        let localGradient = 0;
        if (x > 0) localGradient = Math.max(localGradient, Math.abs(center - linearLuma[pixel - 1]));
        if (x + 1 < image.width) localGradient = Math.max(localGradient, Math.abs(center - linearLuma[pixel + 1]));
        if (y > 0) localGradient = Math.max(localGradient, Math.abs(center - linearLuma[pixel - image.width]));
        if (y + 1 < image.height) localGradient = Math.max(localGradient, Math.abs(center - linearLuma[pixel + image.width]));
        if (localGradient > flatThreshold) continue;
        flatPixels += 1;
        const contributionSquareSum = floats[base + 2];
        const contributorCount = words[base + 3];
        if (
          !(compositeAlpha >= BRUSH_CONTRIBUTION_MIN_COMPOSITE_ALPHA) ||
          !(contributionSquareSum > 1e-12) ||
          contributorCount === 0
        ) continue;
        const effectiveContributors = Math.max(
          1,
          Math.min(contributorCount, (compositeAlpha * compositeAlpha) / contributionSquareSum),
        );
        flatContributingPixels += 1;
        contributorCountTotal += contributorCount;
        effectiveContributorTotal += effectiveContributors;
        renyi2EntropyTotal += Math.log(effectiveContributors);
        flatContributingAlphaTotal += compositeAlpha;
        if (effectiveContributors > BRUSH_CONTRIBUTION_HIGH_NEFF) highNeffPixels += 1;
      }
      const highNeffRatioValid = highNeffPixels / Math.max(1, validPixels);
      report = {
        candidate: "BR-CAND-01",
        backend: "webgpu-final-only-readback",
        read_only: true,
        standard_alpha_blend: true,
        signal_space: "signal-sRGB storage; exact sRGB decode to Linear-sRGB Rec.709 luma for flatness",
        strength: settings.strength,
        flat_linear_gradient_threshold: flatThreshold,
        minimum_composite_alpha: BRUSH_CONTRIBUTION_MIN_COMPOSITE_ALPHA,
        high_neff_threshold: BRUSH_CONTRIBUTION_HIGH_NEFF,
        valid_pixels: validPixels,
        flat_pixels: flatPixels,
        flat_ratio: flatPixels / Math.max(1, validPixels),
        flat_contributing_pixels: flatContributingPixels,
        mean_accepted_contributors: contributorCountTotal / Math.max(1, flatContributingPixels),
        mean_physical_effective_contributors: effectiveContributorTotal / Math.max(1, flatContributingPixels),
        mean_renyi2_entropy: renyi2EntropyTotal / Math.max(1, flatContributingPixels),
        high_neff_pixels: highNeffPixels,
        high_neff_ratio_valid: highNeffRatioValid,
        high_neff_ratio_flat_contributing: highNeffPixels / Math.max(1, flatContributingPixels),
        mean_composited_alpha: validCompositeAlphaTotal / Math.max(1, validPixels),
        mean_flat_contributing_alpha: flatContributingAlphaTotal / Math.max(1, flatContributingPixels),
        precondition_minimum_ratio: BRUSH_CONTRIBUTION_MIN_HIGH_NEFF_FRACTION,
        precondition_pass: highNeffRatioValid >= BRUSH_CONTRIBUTION_MIN_HIGH_NEFF_FRACTION,
      };
      return report;
    } catch (error) {
      operationError = error;
      throw error;
    } finally {
      let restorationError = null;
      try {
        await this.refreshRenderState(image, params, {
          computeSsim: true,
          contributionMoments: false,
        });
      } catch (error) {
        restorationError = error;
      } finally {
        readBuffer.destroy();
      }
      if (report) {
        report.parameter_hash_before = parameterHashBefore;
        report.parameter_hash_after = hashParams(params);
        report.parameter_hash_matches = report.parameter_hash_before === report.parameter_hash_after;
      }
      if (restorationError) {
        if (operationError) {
          console.warn("Brush contribution diagnostic render-state restoration also failed", restorationError);
        } else {
          throw restorationError;
        }
      }
    }
  }

  async computeTrainStateMetrics(image, params, { reuseCurrentRender = false } = {}) {
    await this.ensurePixelMetricsPipeline();
    const reusedRender = Boolean(reuseCurrentRender && this.canReuseMetricRender(image));
    if (!reusedRender) await this.refreshRenderState(image, params);
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
          { binding: 4, resource: { buffer: this.trainState.ssimTileBuffer } },
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
      let squaredErrorTotal = 0;
      let renderedY = 0;
      let targetY = 0;
      let renderedY2 = 0;
      let targetY2 = 0;
      let renderedTargetY = 0;
      let maxLoss = 0;
      let windowedTotal = 0;
      let alphaWindowedTotal = 0;
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
        squaredErrorTotal += values[i + 33];
        windowedTotal += values[i + 34];
        alphaWindowedTotal += values[i + 35];
      }
      const pixelCount = image.width * image.height;
      const loss = lossTotal / pixelCount;
      const mse = squaredErrorTotal / Math.max(1, pixelCount * 3);
      const psnr = psnrFromRgbMse(mse);
      const alphaL1 = alphaError / pixelCount;
      const alphaMean = alphaMoments.rendered / pixelCount;
      const alphaTargetMean = alphaMoments.target / pixelCount;
      const alphaSsim = alphaWindowedTotal / Math.max(1, pixelCount);
      const alphaWeight = phase40Variants().alphaLossWeight;
      const alphaObjective = (1 - DEFAULT_DSSIM_WEIGHT) * alphaL1 + DEFAULT_DSSIM_WEIGHT * (1 - alphaSsim) * 0.5;
      const objectiveLoss = loss + alphaWeight * alphaObjective;
      const meanX = renderedY / pixelCount;
      const meanY = targetY / pixelCount;
      const momentSsim = ssimFromMoments(meanX, meanY, Math.max(0, renderedY2 / pixelCount - meanX ** 2), Math.max(0, targetY2 / pixelCount - meanY ** 2), renderedTargetY / pixelCount - meanX * meanY);
      const ssim = windowedTotal / Math.max(1, pixelCount);
      const windowedSsim = ssim;
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
      this.lastLossStats = {
        loss,
        mse,
        psnr,
        psnr_metric: {
          channels: "RGB",
          signal_space: "sRGB signal values",
          data_range: 1,
          mse_floor: PSNR_MSE_FLOOR,
          unit: "dB",
        },
        alphaL1,
        alphaSsim,
        alphaObjective,
        alphaWeight,
        objectiveLoss,
        ssim,
        windowedSsim,
        momentSsim,
        regionalSsim,
        highFrequency,
        coverage,
        max: maxLoss,
        count: pixelCount,
        partial_count: partialCount,
        bytes: outputBytes,
        reduction: "tile-8x8-from-compact-render",
        compact_tile_candidates: this.runtime.tileCullingEnabled(),
        reused_render: reusedRender,
      };
      return { loss, mse, psnr, alphaL1, alphaSsim, alphaObjective, alphaWeight, objectiveLoss, ssim, windowedSsim, regionalSsim, highFrequency, coverage, reusedRender };
    } finally {
      lossBuffer.destroy();
      readBuffer.destroy();
    }
  }

  async computeVirtualCameraViewMetrics(image, params, view, outputBuffer, readBuffer) {
    await this.refreshRenderState(image, params, { view });
    const partialCount = Math.ceil(image.width / 8) * Math.ceil(image.height / 8);
    const bindGroup = this.device.createBindGroup({
      layout: this.virtualCameraMetricsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.targetBuffer } },
        { binding: 2, resource: { buffer: this.trainState.pixelStateBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: this.trainState.targetAlphaBuffer } },
        { binding: 5, resource: { buffer: this.trainState.alphaStateBuffer } },
        { binding: 6, resource: { buffer: this.trainState.ssimTileBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.virtualCameraMetricsPipeline);
    pass.setBindGroup(0, bindGroup);
    this.dispatchLinear(pass, partialCount);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, partialCount * VIRTUAL_CAMERA_METRIC_TILE_STRIDE * 4);
    this.device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    let loss = 0;
    let squaredError = 0;
    let rendered = 0;
    let target = 0;
    let rendered2 = 0;
    let target2 = 0;
    let cross = 0;
    let pixels = 0;
    let alphaL1 = 0;
    let alpha = 0;
    let targetAlpha = 0;
    let alpha2 = 0;
    let targetAlpha2 = 0;
    let alphaCross = 0;
    let coverage = 0;
    let background = 0;
    let renderedChroma = 0;
    let targetChroma = 0;
    let ssimTotal = 0;
    let alphaSsimTotal = 0;
    const tileSsim = [];
    for (let index = 0; index < values.length; index += VIRTUAL_CAMERA_METRIC_TILE_STRIDE) {
      const count = values[index + 6];
      if (count <= 0) continue;
      loss += values[index];
      rendered += values[index + 1];
      target += values[index + 2];
      rendered2 += values[index + 3];
      target2 += values[index + 4];
      cross += values[index + 5];
      pixels += count;
      alphaL1 += values[index + 7];
      alpha += values[index + 8];
      targetAlpha += values[index + 9];
      alpha2 += values[index + 10];
      targetAlpha2 += values[index + 11];
      alphaCross += values[index + 12];
      coverage += values[index + 13];
      background += values[index + 14];
      squaredError += values[index + 15];
      renderedChroma += values[index + 16];
      targetChroma += values[index + 17];
      ssimTotal += values[index + 18];
      alphaSsimTotal += values[index + 19];
      tileSsim.push(values[index + 18] / count);
    }
    if (pixels <= 0) {
      throw new Error(`Virtual camera ${view.id || "unknown"} has no valid teacher pixels.`);
    }
    const mean = (sum) => sum / pixels;
    const rgbMean = mean(rendered);
    const targetMean = mean(target);
    const alphaMean = mean(alpha);
    const targetAlphaMean = mean(targetAlpha);
    tileSsim.sort((a, b) => a - b);
    return {
      id: view.id,
      pitch_degrees: Number(view.pitchDegrees ?? view.pitch_degrees) || 0,
      yaw_degrees: Number(view.yawDegrees ?? view.yaw_degrees) || 0,
      polar_degrees: Number(view.polarDegrees ?? view.polar_degrees) || null,
      valid_pixel_count: pixels,
      valid_pixel_ratio: pixels / Math.max(1, image.width * image.height),
      loss: mean(loss),
      mse: squaredError / Math.max(1, pixels * 3),
      psnr: psnrFromRgbMse(squaredError / Math.max(1, pixels * 3)),
      ssim: mean(ssimTotal),
      windowedSsim: mean(ssimTotal),
      local_p10: percentileSorted(tileSsim, 0.1),
      alphaL1: mean(alphaL1),
      alphaSsim: mean(alphaSsimTotal),
      coverage_mean: mean(coverage),
      background_exposure_ratio: mean(background),
      rendered_mean_srgb_signal: rgbMean,
      target_mean_srgb_signal: targetMean,
      rendered_minus_target_signal: rgbMean - targetMean,
      rendered_signal_stddev: Math.sqrt(Math.max(0, mean(rendered2) - rgbMean ** 2)),
      target_signal_stddev: Math.sqrt(Math.max(0, mean(target2) - targetMean ** 2)),
      rendered_mean_srgb_chroma: mean(renderedChroma),
      target_mean_srgb_chroma: mean(targetChroma),
      rendered_minus_target_chroma: mean(renderedChroma - targetChroma),
    };
  }

  async computeVirtualCameraEvaluation(image, params, frontMetrics) {
    const sampling = state.metrics?.virtual_camera_sampling;
    if (!sampling?.enabled) return null;
    const catalog = Array.isArray(sampling.cameras) ? sampling.cameras.filter((camera) => camera.kind === "virtual") : [];
    if (!catalog.length) return null;
    await this.ensureVirtualCameraMetricsPipeline();
    const partialCount = Math.ceil(image.width / 8) * Math.ceil(image.height / 8);
    const outputBytes = partialCount * VIRTUAL_CAMERA_METRIC_TILE_STRIDE * 4;
    const outputBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuffer = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const orbitRadius = Number(sampling.orbit_radius) || sharedTiltOrbitRadius(
      image.width,
      image.height,
      sampling.max_angle_degrees,
      sampling.virtual_camera_count,
      sampling.fov_degrees,
    );
    const virtualEntries = [];
    try {
      for (let cameraIndex = 0; cameraIndex < catalog.length; cameraIndex += 1) {
        const camera = catalog[cameraIndex];
        const view = {
          id: camera.id,
          pitchDegrees: camera.pitch_degrees,
          yawDegrees: camera.yaw_degrees,
          polarDegrees: camera.polar_degrees,
          fovDegrees: camera.intrinsics?.fov_degrees || sampling.fov_degrees,
          cameraDistance: orbitRadius,
          planeConstrained: true,
          cameraCovariance3d: Boolean(sampling.three_dgs_multiview),
        };
        virtualEntries.push({ camera: view, metrics: await this.computeVirtualCameraViewMetrics(image, params, view, outputBuffer, readBuffer) });
        if ((cameraIndex + 1) % 4 === 0 || cameraIndex + 1 === catalog.length) {
          setTrainingMessage(`Evaluating virtual camera teachers ${cameraIndex + 1} / ${catalog.length}...`);
          publishState();
          await nextFrame();
        }
      }
    } finally {
      outputBuffer.destroy();
      readBuffer.destroy();
      if (this.runtime.tileCullingEnabled()) {
        await this.prepareTileLists(image, params, { sync: true });
      }
      await this.refreshRenderState(image, params);
    }
    const frontEntry = {
      camera: { id: "front", kind: "front", pitchDegrees: 0, yawDegrees: 0 },
      metrics: {
        valid_pixel_count: image.width * image.height,
        valid_pixel_ratio: 1,
        loss: frontMetrics.loss,
        mse: frontMetrics.mse,
        psnr: frontMetrics.psnr,
        ssim: frontMetrics.ssim,
        windowedSsim: frontMetrics.windowedSsim,
        local_p10: frontMetrics.regionalSsim?.p10 ?? null,
        alphaL1: frontMetrics.alphaL1,
        alphaSsim: frontMetrics.alphaSsim,
        coverage_mean: frontMetrics.coverage?.mean ?? null,
        background_exposure_ratio: frontMetrics.coverage?.background_exposure_ratio ?? null,
      },
    };
    return {
      backend: "webgpu-final-only",
      metric_space: "sRGB signal values",
      psnr_contract: {
        channels: "RGB",
        data_range: 1,
        mse_floor: PSNR_MSE_FLOOR,
        unit: "dB",
      },
      target: "known planar source reprojected per camera",
      aggregation: "equal-camera macro for SSIM, L1, MSE, and PSNR; p10/min retain weak virtual views",
      front_view: frontEntry.metrics,
      virtual_views: summarizeVirtualCameraMetricSet(virtualEntries),
      all_views: summarizeVirtualCameraMetricSet([frontEntry, ...virtualEntries]),
      cameras: virtualEntries.map(({ camera, metrics }) => ({
        id: camera.id,
        pitch_degrees: metrics.pitch_degrees,
        yaw_degrees: metrics.yaw_degrees,
        polar_degrees: metrics.polar_degrees,
        ...metrics,
      })),
    };
  }

  async ensureDensityPipelines() {
    if (this.growSelectPipeline && this.distributionOffsetPipeline && this.residualTileOffsetPipeline && this.relocationApplyPipeline && this.finalBrushRepairPipeline && this.phase45RegionTelemetryPipeline && this.phase45RegionFinalizePipeline && this.phase45DonorSafetyPipeline && this.structureAllocationCollectPipeline) {
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
  return select(
    select(1.57079632679, 0.0, longAxisIsX),
    select(0.0, 1.57079632679, longAxisIsX),
    vertical
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
  let width = u32(config[0]);
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
  let adc = is_adc_step(u32(config[4])) || u32(config[11]) == 3u;
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
  let probeSource = (local * 2654435761u + step * 2246822519u) % oldCount;
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
  let t = transform[g];
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
  let encoded = atomicLoad(&control[capacity + g]);
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
  let count = u32(config[2]);
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
      this.residualTileBuildPipeline,
      this.residualTileBlockScanPipeline,
      this.residualTileBlockSumsPipeline,
      this.residualTileOffsetPipeline,
      this.growSelectPipeline,
      this.growApplyPipeline,
      this.relocationSelectPipeline,
      this.relocationApplyPipeline,
      this.finalBrushRepairPipeline,
      this.densityAuxResetPipeline,
      this.phase45RegionTelemetryPipeline,
      this.phase45RegionFinalizePipeline,
      this.phase45DonorSafetyPipeline,
      this.structureAllocationCollectPipeline,
    ] = await Promise.all([
      make("build_distribution"),
      make("scan_distribution_blocks"),
      make("scan_distribution_block_sums"),
      make("add_distribution_block_offsets"),
      make("build_residual_tiles"),
      make("scan_residual_tile_blocks"),
      make("scan_residual_tile_block_sums"),
      make("add_residual_tile_block_offsets"),
      make("select_grow"),
      make("apply_grow"),
      make("select_relocation"),
      make("apply_relocation"),
      make("apply_final_brush_repair"),
      make("reset_density_aux"),
      make("phase45_collect_region_telemetry"),
      make("phase45_finalize_region_telemetry"),
      make("phase45_evaluate_donor_safety"),
      make("collect_structure_allocation_counts"),
    ]);
    await this.ensureOptimizerResetPipeline();
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
    this.disposeResultRenderState();
    const capacityError = (kind, message) => Object.assign(new Error(message), { capacityFailure: kind });
    const variants = phase33Variants();
    const { coarseImage, midImage } = makeCurriculumImages(image, variants);
    // UI base counts use a step of four, but additive Line capacity can be any
    // integer. Rounding the effective capacity back to the UI grid could
    // allocate one slot too few and make the final growth event return false.
    const bufferCapacity = Math.max(activeCount, normalizeActiveSplatCount(capacity, activeCount));
    const tilePlan = plannedTileIndexCapacity(image, params, bufferCapacity, this.device);
  const allocationPlan = trainingAllocationPlan(image, params, bufferCapacity, this.device, {
      coarseImage,
      midImage,
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
    const tileCols = Math.ceil(image.width / TILE_SIZE);
    const tileRows = Math.ceil(image.height / TILE_SIZE);
    const tileCount = tileCols * tileRows;
    const tileIndexCapacity = tilePlan.capacity;
    const color = packColors(params);
    const transform = packTransforms(params);
    const positions = packPositions(params);
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
      bindGroupCacheEnabled: performanceVariants().bindGroupCache,
      bindGroupCache: new Map(),
      bindGroupCacheStats: { hits: 0, misses: 0 },
      configBuffer: allocationDevice.createBuffer({
        size: TRAIN_CONFIG_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      batchConfigBuffer: allocationDevice.createBuffer({
        size: TRAIN_BATCH_CONFIG_BYTES,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      presentConfigBuffer: allocationDevice.createBuffer({
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
      virtualCameraFrontSteps: 0,
      virtualCameraVirtualSteps: 0,
      lastVirtualCameraSample: null,
      virtualCameraOrbitRadius: null,
      pixelStateResolution: null,
      pixelStateKind: "uninitialized",
      errorMapBuffer: allocationDevice.createBuffer({
        size: Math.max(4, image.width * image.height * 4),
        usage: GPUBufferUsage.STORAGE | (residualDestinationOracleRequested() ? GPUBufferUsage.COPY_SRC : 0),
      }),
      errorMapStep: null,
      errorMapReason: "",
      statsBuffer: allocationDevice.createBuffer({
        size: Math.max(32, bufferCapacity * 2 * 4 * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      densityControlBuffer: allocationDevice.createBuffer({
        size: Math.max(32, (
          bufferCapacity * 4 +
          DENSITY_EVENT_SLOTS + 1 +
          Math.ceil(bufferCapacity / 256) * 2 +
          PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE +
          residualTileControlWords(image)
        ) * 4),
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
        size: Math.max(ALPHA_STATE_BYTES_PER_PIXEL, image.width * image.height * ALPHA_STATE_BYTES_PER_PIXEL),
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
      segmentedExactBackward: allocationPlan.segmentedExactBackward,
      fixedPointExactGradient: allocationPlan.fixedPointExactGradient,
      fixedPointGradientControlBuffer: allocationPlan.fixedPointExactGradient?.enabled
        ? allocationDevice.createBuffer({
            size: FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          })
        : null,
      fixedPointGradientReadbackBuffer: allocationPlan.fixedPointExactGradient?.enabled
        ? allocationDevice.createBuffer({
            size: FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          })
        : null,
      segmentedPartialGradientBuffer: allocationPlan.segmentedExactBackward?.enabled
        ? allocationDevice.createBuffer({
            size: allocationPlan.segmentedExactBackward.partialBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          })
        : null,
      segmentedReferenceCountsBuffer: allocationPlan.segmentedExactBackward?.enabled
        ? allocationDevice.createBuffer({
            size: Math.max(4, bufferCapacity * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          })
        : null,
      segmentedReferenceOffsetsBuffer: allocationPlan.segmentedExactBackward?.enabled
        ? allocationDevice.createBuffer({
            size: Math.max(8, (bufferCapacity + 1) * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          })
        : null,
      segmentedReferenceCursorsBuffer: allocationPlan.segmentedExactBackward?.enabled
        ? allocationDevice.createBuffer({
            size: Math.max(4, bufferCapacity * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          })
        : null,
      segmentedReferencesBuffer: allocationPlan.segmentedExactBackward?.enabled
        ? allocationDevice.createBuffer({
            size: Math.max(4, tileIndexCapacity * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          })
        : null,
      exactBackwardTelemetryBuffer: this.performanceProfile.timestampQuery
        ? allocationDevice.createBuffer({
            size: EXACT_BACKWARD_TELEMETRY_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          })
        : null,
      exactBackwardTelemetryReadbackBuffer: this.performanceProfile.timestampQuery
        ? allocationDevice.createBuffer({
            size: EXACT_BACKWARD_TELEMETRY_BYTES,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          })
        : null,
      ssimTileBuffer: allocationDevice.createBuffer({
        size: Math.max(64, ssimWorkingBufferBytes(image)),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      optimizerStateBuffer: allocationDevice.createBuffer({
        size: Math.max(96, bufferCapacity * 96),
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
      residualTileControlWords: residualTileControlWords(image),
      zeroResidualTileControl: new Uint32Array(residualTileControlWords(image)),
      residualOracleEvents: [],
      residualOracleRatios: [],
      xyBuffers: [makeSizedBuffer(allocationDevice, positions, bufferCapacity * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)],
      transformBuffers: [makeSizedBuffer(allocationDevice, transform, bufferCapacity * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)],
      colorBuffers: [makeSizedBuffer(allocationDevice, color, bufferCapacity * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)],
      readbackBuffer: allocationDevice.createBuffer({
        size: bufferCapacity * (4 + 4 + 4) * 4,
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
      candidate.fixedPointGradientControlBuffer,
      candidate.tileControlBuffer,
      candidate.ssimTileBuffer,
      candidate.optimizerStateBuffer,
      candidate.readbackBuffer,
    ]) {
      if (buffer) encoder.clearBuffer(buffer);
    }
    if (candidate.exactBackwardTelemetryBuffer) encoder.clearBuffer(candidate.exactBackwardTelemetryBuffer);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const scoped = await popAllocationScopes();
    if (scoped.popErrors.length) throw capacityError("internal", `GPU error-scope failure: ${scoped.popErrors.map((error) => error.message).join(" | ")}`);
    if (scoped.validationError) throw capacityError("validation", `GPU capacity validation failed: ${scoped.validationError.message}`);
    if (scoped.oomError) throw capacityError("out-of-memory", `GPU capacity allocation failed: ${scoped.oomError.message}`);
    if (verifyAllocation) {
      const allocatedDescriptors = trainStateAllocatedDescriptors(candidate);
      const mismatches = allocationDescriptorMismatch(allocationPlan.descriptors, allocatedDescriptors);
      const actualReservedBytes = allocatedDescriptors.reduce((total, descriptor) => total + descriptor.size, 0);
      if (mismatches.length || actualReservedBytes !== allocationPlan.reservedBytes) {
        throw capacityError(
          "internal",
          `GPU allocation descriptor drift: planned ${allocationPlan.reservedBytes} bytes, allocated ${actualReservedBytes} bytes; ${JSON.stringify(mismatches)}`,
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

  async growExperimentalGpu(
    image,
    params,
    targetCount,
    step,
    steps,
    { forceZeroMassFallback = false } = {},
  ) {
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
    let structureAllocation = null;
    if (config[68] > 0.5) {
      const prepared = structureGuidedRegionControl(
        image,
        targetCount,
        this.trainState.capacity,
      );
      structureAllocation = prepared.allocation;
      this.device.queue.writeBuffer(
        this.trainState.densityControlBuffer,
        this.phase45RegionTelemetryOffsetBytes(),
        prepared.control,
      );
    }
    if (this.trainState.residualTileControlWords > 0) {
      this.device.queue.writeBuffer(
        this.trainState.densityControlBuffer,
        this.residualTileControlOffsetBytes(),
        this.trainState.zeroResidualTileControl,
      );
    }
    const front = this.trainState.front;
    const bindGroup = this.densityBindGroup(front);
    const distributionProfileSample = this.operationProfileSample(step);
    const distributionEncoder = this.device.createCommandEncoder();
    const distributionPass = distributionEncoder.beginComputePass(
      this.profilePassDescriptor(distributionProfileSample, "density_distribution"),
    );
    distributionPass.setBindGroup(0, bindGroup);
    if (structureAllocation) {
      distributionPass.setPipeline(this.structureAllocationCollectPipeline);
      distributionPass.dispatchWorkgroups(Math.ceil(oldCount / 64));
    }
    distributionPass.setPipeline(this.distributionPipeline);
    distributionPass.dispatchWorkgroups(Math.ceil(oldCount / 256));
    distributionPass.setPipeline(this.distributionBlockScanPipeline);
    distributionPass.dispatchWorkgroups(Math.ceil(oldCount / 256));
    distributionPass.setPipeline(this.distributionBlockSumsPipeline);
    distributionPass.dispatchWorkgroups(1);
    distributionPass.setPipeline(this.distributionOffsetPipeline);
    distributionPass.dispatchWorkgroups(Math.ceil(oldCount / 256));
    if (this.trainState.residualTileControlWords > 0) {
      const residualTileCount = Math.ceil(image.width / TILE_SIZE) * Math.ceil(image.height / TILE_SIZE);
      distributionPass.setPipeline(this.residualTileBuildPipeline);
      distributionPass.dispatchWorkgroups(Math.ceil((image.width * image.height) / 256));
      distributionPass.setPipeline(this.residualTileBlockScanPipeline);
      distributionPass.dispatchWorkgroups(Math.ceil(residualTileCount / 256));
      distributionPass.setPipeline(this.residualTileBlockSumsPipeline);
      distributionPass.dispatchWorkgroups(1);
      distributionPass.setPipeline(this.residualTileOffsetPipeline);
      distributionPass.dispatchWorkgroups(Math.ceil(residualTileCount / 256));
    }
    distributionPass.end();
    const cdfTotalOffset = (this.trainState.capacity * 3 + DENSITY_EVENT_SLOTS) * 4;
    distributionEncoder.copyBufferToBuffer(
      this.trainState.densityControlBuffer,
      cdfTotalOffset,
      this.trainState.growthSignalReadbackBuffer,
      0,
      4,
    );
    const distributionProfile = await this.submitProfiledOperation(distributionEncoder, distributionProfileSample, {
      resolution: [image.width, image.height],
      activeSplats: oldCount,
    });
    const candidateReadbackStarted = performance.now();
    await this.trainState.growthSignalReadbackBuffer.mapAsync(GPUMapMode.READ);
    const candidateMass = new Float32Array(this.trainState.growthSignalReadbackBuffer.getMappedRange())[0];
    this.trainState.growthSignalReadbackBuffer.unmap();
    if (distributionProfile) {
      distributionProfile.readback_count += 1;
      distributionProfile.readback_bytes += 4;
      distributionProfile.readback_wall_ms += performance.now() - candidateReadbackStarted;
      distributionProfile.total_wall_ms = performance.now() - candidateReadbackStarted + distributionProfile.total_wall_ms;
    }
    const finiteCandidateMass = Number.isFinite(candidateMass);
    const zeroCandidateMass = finiteCandidateMass && candidateMass <= 1e-8;
    if (!finiteCandidateMass || (zeroCandidateMass && !forceZeroMassFallback)) {
      const qaCounters = phase39Variants().qaGrowthComparisons
        ? await this.readDensityCounters()
        : null;
      this.lastTrainStats = {
        ...(this.lastTrainStats || {}),
        gpu_densify: false,
        growth_threshold_skipped: true,
        growth_candidate_mass: Number.isFinite(candidateMass) ? candidateMass : null,
        growth_zero_mass_fallback: false,
        active_count: oldCount,
      };
      return {
        grown: false,
        count: oldCount,
        candidateMass,
        residualOracle: null,
        operations: qaCounters ? { eligible_sources: qaCounters.growth_eligible_sources } : {},
      };
    }
    const applyProfileSample = this.operationProfileSample(step);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass(this.profilePassDescriptor(applyProfileSample, "density_apply"));
    pass.setBindGroup(0, bindGroup);
    pass.setPipeline(this.growSelectPipeline);
    pass.dispatchWorkgroups(Math.ceil((targetCount - oldCount) / 64));
    pass.setPipeline(this.growApplyPipeline);
    pass.dispatchWorkgroups(Math.ceil((targetCount - oldCount) / 64));
    pass.setPipeline(this.optimizerResetPipeline);
    pass.setBindGroup(0, this.optimizerResetBindGroup());
    pass.dispatchWorkgroups(Math.ceil((targetCount - oldCount) / 64));
    pass.setPipeline(this.optimizerSourceResetPipeline);
    pass.setBindGroup(0, this.optimizerResetBindGroup(this.optimizerSourceResetPipeline));
    pass.dispatchWorkgroups(Math.ceil(oldCount / 64));
    pass.end();
    const applyProfile = await this.submitProfiledOperation(encoder, applyProfileSample, {
      resolution: [image.width, image.height],
      activeSplats: targetCount,
    });
    const counterReadbackStarted = performance.now();
    const operationCounters = await this.readDensityCounters();
    if (applyProfile) {
      applyProfile.readback_count += 1;
      applyProfile.readback_bytes += DENSITY_EVENT_SLOTS * 4;
      applyProfile.readback_wall_ms += performance.now() - counterReadbackStarted;
    }
    const residualOracle = residualDestinationOracleRequested()
      ? await this.readResidualDestinationOracle(image, oldCount, targetCount, step, phase45Variants().seedOffset)
      : null;
    const operations = {
      split: Math.max(0, operationCounters?.adc_split || 0),
      duplicate: Math.max(0, operationCounters?.adc_duplicate || 0),
      reseed: Math.max(0, operationCounters?.mcmc_reseed || 0),
      adc_eligible: Math.max(0, operationCounters?.adc_eligible || 0),
      adc_fallback: Math.max(0, operationCounters?.adc_fallback || 0),
      source_claims: Math.max(0, operationCounters?.source_claims || 0),
      source_claim_conflicts: Math.max(0, operationCounters?.source_claim_conflicts || 0),
      eligible_sources: Math.max(0, operationCounters?.growth_eligible_sources || 0),
      tilt_risk_candidates: Math.max(0, operationCounters?.tilt_risk_candidates || 0),
      tilt_true_splits: Math.max(0, operationCounters?.tilt_true_splits || 0),
      surface_layer_candidates: Math.max(0, operationCounters?.surface_layer_candidates || 0),
      surface_layer_promotions: Math.max(0, operationCounters?.surface_layer_promotions || 0),
      harmful_rectangle_candidate_selections: Math.max(
        0,
        operationCounters?.harmful_rectangle_candidate_selections || 0,
      ),
      harmful_rectangle_front_oversized_selections: Math.max(
        0,
        operationCounters?.harmful_rectangle_front_oversized_selections || 0,
      ),
      harmful_rectangle_high_contribution_selections: Math.max(
        0,
        operationCounters?.harmful_rectangle_high_contribution_selections || 0,
      ),
      harmful_rectangle_high_deviation_selections: Math.max(
        0,
        operationCounters?.harmful_rectangle_high_deviation_selections || 0,
      ),
      harmful_rectangle_parent_replacements: Math.max(
        0,
        operationCounters?.harmful_rectangle_parent_replacements || 0,
      ),
      harmful_rectangle_children_created: Math.max(
        0,
        operationCounters?.harmful_rectangle_children_created || 0,
      ),
      structure_allocation_over_budget_source_selections: Math.max(
        0,
        operationCounters?.structure_allocation_over_budget_source_selections || 0,
      ),
      structure_allocation_under_budget_source_selections: Math.max(
        0,
        operationCounters?.structure_allocation_under_budget_source_selections || 0,
      ),
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
      growth_zero_mass_fallback: zeroCandidateMass,
      structure_guided_allocation: structureAllocation ? {
        enabled: true,
        grid: [structureAllocation.regionGrid, structureAllocation.regionGrid],
        luma_space: structureAllocation.lumaSpace,
        region_mode: structureAllocation.regionMode,
        profile_processing_ms: structureAllocation.processingMs,
        structure_strength: structureAllocation.structureStrength,
        gradient_variance_p10: structureAllocation.percentile10,
        gradient_variance_p90: structureAllocation.percentile90,
        target_quotas: Array.from(structureAllocation.quotas),
      } : { enabled: false, baseline: true },
      growth_operations: operations,
      density_profiled: Boolean(distributionProfile || applyProfile),
      density_stats_reset_after_batch: false,
      residual_tile_cdf: residualTileCdfEnabled(),
    };
    return {
      grown: true,
      count: targetCount,
      candidateMass,
      operations,
      residualOracle,
      zeroMassFallback: zeroCandidateMass,
    };
  }

  async readResidualDestinationOracle(image, startIndex, targetCount, step, seedOffset = 0) {
    if (!this.trainState || targetCount <= startIndex) return null;
    const pixelCount = image.width * image.height;
    const destinationCount = targetCount - startIndex;
    const selectionBytes = destinationCount * 4;
    const errorMapBytes = pixelCount * 4;
    const bytes = Math.max(4, selectionBytes + errorMapBytes);
    const readBuffer = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(
        this.trainState.densityControlBuffer,
        this.trainState.capacity * 4,
        readBuffer,
        0,
        selectionBytes,
      );
      encoder.copyBufferToBuffer(this.trainState.errorMapBuffer, 0, readBuffer, selectionBytes, errorMapBytes);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const mapped = readBuffer.getMappedRange();
      const selections = new Uint32Array(mapped, 0, destinationCount).slice();
      const errorMap = new Float32Array(mapped, selectionBytes, pixelCount).slice();
      readBuffer.unmap();

      const tileSize = 16;
      const tileCols = Math.ceil(image.width / tileSize);
      const tileRows = Math.ceil(image.height / tileSize);
      const tileWeights = new Float64Array(tileCols * tileRows);
      const tileMaximums = new Float32Array(tileCols * tileRows);
      let totalWeight = 0;
      for (let y = 0; y < image.height; y += 1) {
        const tileRow = Math.floor(y / tileSize) * tileCols;
        const row = y * image.width;
        for (let x = 0; x < image.width; x += 1) {
          const value = Number.isFinite(errorMap[row + x]) ? Math.max(0, errorMap[row + x]) : 0;
          const tile = tileRow + Math.floor(x / tileSize);
          tileWeights[tile] += value;
          tileMaximums[tile] = Math.max(tileMaximums[tile], value);
          totalWeight += value;
        }
      }
      for (let tile = 1; tile < tileWeights.length; tile += 1) tileWeights[tile] += tileWeights[tile - 1];

      const hashUnit = (seed) => {
        const value = Math.sin((seed + seedOffset * 104729) * 12.9898) * 43758.5453123;
        return value - Math.floor(value);
      };
      const chooseTile = (sample) => {
        let low = 0;
        let high = tileWeights.length - 1;
        while (low < high) {
          const mid = low + Math.floor((high - low) / 2);
          if (tileWeights[mid] < sample) low = mid + 1;
          else high = mid;
        }
        return low;
      };
      const modeZeroSelections = [];
      for (let local = 0; local < selections.length; local += 1) {
        const encoded = selections[local];
        if ((encoded >>> 30) !== 0 || (encoded & 0x3fffffff) === 0) continue;
        modeZeroSelections.push({
          index: startIndex + local,
          pixel: (encoded & 0x3fffffff) - 1,
        });
      }
      const sampleCount = Math.min(256, modeZeroSelections.length);
      const ratios = [];
      let selectedTotal = 0;
      let oracleTotal = 0;
      let zeroOracleCount = 0;
      for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
        const selection = modeZeroSelections[Math.floor((sampleIndex + 0.5) * modeZeroSelections.length / sampleCount)];
        const index = selection.index;
        const selected = selection.pixel < pixelCount && Number.isFinite(errorMap[selection.pixel])
          ? Math.max(0, errorMap[selection.pixel])
          : 0;
        const tileSample = totalWeight > 0
          ? hashUnit(index * 23.47 + step * 0.73 + 11) * totalWeight
          : 0;
        const oracle = totalWeight > 0 ? tileMaximums[chooseTile(tileSample)] : selected;
        if (oracle > 1e-8) ratios.push(Math.min(1, selected / oracle));
        else zeroOracleCount += 1;
        selectedTotal += selected;
        oracleTotal += oracle;
      }
      ratios.sort((a, b) => a - b);
      const belowHalfCount = ratios.filter((value) => value < 0.5).length;
      const event = {
        step,
        error_map_step: this.trainState?.errorMapStep ?? 0,
        error_map_age: Math.max(0, step - (this.trainState?.errorMapStep ?? 0)),
        tile_size: tileSize,
        tile_count: tileCols * tileRows,
        mode_zero_count: modeZeroSelections.length,
        sample_count: ratios.length,
        zero_oracle_count: zeroOracleCount,
        selected_oracle_median: ratios.length ? percentileSorted(ratios, 0.5) : null,
        selected_oracle_p10: ratios.length ? percentileSorted(ratios, 0.1) : null,
        below_half_count: belowHalfCount,
        below_half_fraction: belowHalfCount / Math.max(1, ratios.length),
        selected_residual_mean: selectedTotal / Math.max(1, sampleCount),
        oracle_residual_mean: oracleTotal / Math.max(1, sampleCount),
      };
      this.trainState.residualOracleEvents.push(event);
      this.trainState.residualOracleRatios.push(...ratios);
      return event;
    } finally {
      readBuffer.destroy();
    }
  }

  residualDestinationOracleSummary() {
    const ratios = [...(this.trainState?.residualOracleRatios || [])].sort((a, b) => a - b);
    const events = [...(this.trainState?.residualOracleEvents || [])];
    const belowHalfCount = ratios.filter((value) => value < 0.5).length;
    const median = ratios.length ? percentileSorted(ratios, 0.5) : null;
    const belowHalfFraction = belowHalfCount / Math.max(1, ratios.length);
    return {
      enabled: residualDestinationOracleRequested(),
      contract: "best-of-32-vs-residual-sum-cdf-tile-max",
      event_count: events.length,
      sample_count: ratios.length,
      selected_oracle_median: median,
      selected_oracle_p10: ratios.length ? percentileSorted(ratios, 0.1) : null,
      below_half_count: belowHalfCount,
      below_half_fraction: belowHalfFraction,
      gate_triggered: ratios.length > 0 && (median < 0.9 || belowHalfFraction > 0.1),
      events,
    };
  }

  async relocateExperimentalGpu(
    image,
    params,
    step,
    learningRates = selectedLearningRates(),
    { paintRepairOnly = false } = {},
  ) {
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
    config[99] = paintRepairOnly ? 1 : 0;
    this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    this.device.queue.writeBuffer(this.trainState.densityControlBuffer, 0, this.trainState.zeroDensityScratch);
    this.device.queue.writeBuffer(
      this.trainState.densityControlBuffer,
      this.trainState.capacity * 2 * 4,
      this.trainState.zeroDensityEvents,
    );
    const front = this.trainState.front;
    const bindGroup = this.densityBindGroup(front);
    const dedicatedBrushRepair =
      paintRepairOnly &&
      normalizedKernelShape(params.kernelShape) === "opaque-brush";
    const relocationProfileSample = this.operationProfileSample(step);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass(this.profilePassDescriptor(relocationProfileSample, "relocation"));
    pass.setBindGroup(0, bindGroup);
    if (dedicatedBrushRepair) {
      pass.setPipeline(this.finalBrushRepairPipeline);
      pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    } else {
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
      pass.setPipeline(this.relocationApplyPipeline);
      pass.dispatchWorkgroups(Math.ceil(params.count / 64));
      pass.setPipeline(this.optimizerResetPipeline);
      pass.setBindGroup(0, this.optimizerResetBindGroup());
      pass.dispatchWorkgroups(Math.ceil(params.count / 64));
      pass.setPipeline(this.optimizerSourceResetPipeline);
      pass.setBindGroup(0, this.optimizerResetBindGroup(this.optimizerSourceResetPipeline));
      pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    }
    pass.end();
    const relocationProfile = await this.submitProfiledOperation(encoder, relocationProfileSample, {
      resolution: [image.width, image.height],
      activeSplats: params.count,
    });
    let queueWaitWallMs = relocationProfile?.queue_wait_wall_ms || 0;
    if (!relocationProfile) {
      const waitStarted = performance.now();
      await this.device.queue.onSubmittedWorkDone();
      queueWaitWallMs = performance.now() - waitStarted;
    }
    this.lastTrainStats = {
      ...(this.lastTrainStats || {}),
      gpu_relocation: true,
      gpu_relocation_step: step,
      paint_outlier_repair_only: Boolean(paintRepairOnly),
      paint_outlier_dedicated_pass: dedicatedBrushRepair,
      weighted_mass_redistribution: true,
      active_count: params.count,
      relocation_queue_wait_count: 1,
      relocation_queue_wait_wall_ms: queueWaitWallMs,
      relocation_profiled: Boolean(relocationProfile),
    };
    return true;
  }

  async correctMidTrainingOverdensityGpu(
    image,
    params,
    step,
    learningRates = selectedLearningRates(),
  ) {
    if (!this.trainState || this.trainState.capacity < params.count) return null;
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
      mode: 3,
    });
    this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
    this.device.queue.writeBuffer(this.trainState.densityControlBuffer, 0, this.trainState.zeroDensityScratch);
    this.device.queue.writeBuffer(
      this.trainState.densityControlBuffer,
      this.trainState.capacity * 2 * 4,
      this.trainState.zeroDensityEvents,
    );
    const front = this.trainState.front;
    const densityBindGroup = this.densityBindGroup(front);
    const donorBindGroup = this.phase45DonorBindGroup(front);
    const profileSample = this.operationProfileSample(step);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "overdensity_correction"));

    pass.setBindGroup(0, densityBindGroup);
    pass.setPipeline(this.phase45RegionTelemetryPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));
    pass.setPipeline(this.phase45RegionFinalizePipeline);
    pass.dispatchWorkgroups(1);

    pass.setBindGroup(0, donorBindGroup);
    pass.setPipeline(this.phase45DonorSafetyPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 64));

    pass.setBindGroup(0, densityBindGroup);
    pass.setPipeline(this.distributionPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    pass.setPipeline(this.distributionBlockScanPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    pass.setPipeline(this.distributionBlockSumsPipeline);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.distributionOffsetPipeline);
    pass.dispatchWorkgroups(Math.ceil(params.count / 256));
    pass.setPipeline(this.relocationSelectPipeline);
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

    const operationProfile = await this.submitProfiledOperation(encoder, profileSample, {
      resolution: [image.width, image.height],
      activeSplats: params.count,
    });
    if (!operationProfile) await this.device.queue.onSubmittedWorkDone();
    const [counters, regions] = await Promise.all([
      this.readDensityCounters(),
      this.readPhase45RegionReport(params.count),
    ]);
    const report = {
      enabled: true,
      step,
      grid: regions?.grid || [phase39Variants().structureRegionGrid, phase39Variants().structureRegionGrid],
      total_splats: params.count,
      donor_quantile: phase39Variants().overdensityDonorFraction,
      eligible_donors: regions?.donor_eligible_count || 0,
      moved_splats: regions?.moved_out_count || counters?.adc_recycle || 0,
      provisional_quota: regions?.provisional_quota || 0,
      fixed_count: true,
      support_guard: "nine-point leave-one-out-or-current-contribution-near-zero",
      source_strategy: "shared-read-only-live-source",
      destination_rgb: "teacher-at-relocated-position",
      profile_ms: operationProfile?.gpu_ms ?? null,
      regions,
    };
    this.lastTrainStats = {
      ...(this.lastTrainStats || {}),
      gpu_relocation: report.moved_splats > 0,
      gpu_relocation_step: step,
      weighted_mass_redistribution: true,
      active_count: params.count,
      mid_training_overdensity_correction: report,
    };
    return report;
  }

  phase45RegionTelemetryOffsetBytes() {
    const capacity = this.trainState.capacity;
    return (capacity * 3 + DENSITY_EVENT_SLOTS + 1 + Math.ceil(capacity / 256) * 2) * 4;
  }

  phase45DonorTelemetryOffsetBytes() {
    return this.phase45RegionTelemetryOffsetBytes() + PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE * 4;
  }

  residualTileControlOffsetBytes() {
    return this.phase45DonorTelemetryOffsetBytes() + this.trainState.capacity * 4;
  }

  async readPhase45RegionReport(currentSplatCount) {
    if (!this.trainState) return null;
    const regionGrid = phase39Variants().structureRegionGrid;
    const regionCount = regionGrid * regionGrid;
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
      for (let region = 0; region < regionCount; region += 1) {
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
          x: region % regionGrid,
          y: Math.floor(region / regionGrid),
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
        grid: [regionGrid, regionGrid],
        region_count: regionCount,
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
        paint_outlier_recycle: values[22],
        paint_outlier_recolor: values[23],
        paint_outlier_trim: values[24],
        surface_layer_candidates: values[25],
        surface_layer_promotions: values[26],
        harmful_rectangle_front_oversized_selections: values[27],
        harmful_rectangle_high_contribution_selections: values[28],
        harmful_rectangle_high_deviation_selections: values[29],
        harmful_rectangle_candidate_selections: values[30],
        harmful_rectangle_parent_replacements: values[31],
        harmful_rectangle_children_created: values[32],
        structure_allocation_over_budget_source_selections: values[33],
        structure_allocation_under_budget_source_selections: values[34],
      };
    } finally {
      readBuffer.destroy();
    }
  }

  async ensureCompactionPipelines() {
    if (this.compactionParamPipeline && this.compactionStatePipeline) return;
    const parameterShader = `
struct CompactConfig { oldCount: u32, newCount: u32, capacity: u32, _padding: u32, };
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(0) var<uniform> config: CompactConfig;
@group(0) @binding(1) var<storage, read> keepIndices: array<u32>;
@group(0) @binding(2) var<storage, read> sourceXy: array<SplatPosition>;
@group(0) @binding(3) var<storage, read> sourceTransform: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> sourceColor: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> outputXy: array<SplatPosition>;
@group(0) @binding(6) var<storage, read_write> outputTransform: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> outputColor: array<vec4<f32>>;

@compute @workgroup_size(64)
fn compact_parameters(@builtin(global_invocation_id) id: vec3u) {
  let destination = id.x;
  if (destination >= config.newCount) { return; }
  let source = keepIndices[destination];
  if (source >= config.oldCount) { return; }
  outputXy[destination] = sourceXy[source];
  outputTransform[destination] = sourceTransform[source];
  outputColor[destination] = sourceColor[source];
}`;
    const stateShader = `
struct CompactConfig { oldCount: u32, newCount: u32, capacity: u32, _padding: u32, };
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
};
@group(0) @binding(0) var<uniform> config: CompactConfig;
@group(0) @binding(1) var<storage, read> keepIndices: array<u32>;
@group(0) @binding(2) var<storage, read> sourceAdam: array<AdamState>;
@group(0) @binding(3) var<storage, read> sourceStats: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outputAdam: array<AdamState>;
@group(0) @binding(5) var<storage, read_write> outputStats: array<vec4<f32>>;

@compute @workgroup_size(64)
fn compact_state(@builtin(global_invocation_id) id: vec3u) {
  let destination = id.x;
  if (destination >= config.newCount) { return; }
  let source = keepIndices[destination];
  if (source >= config.oldCount) { return; }
  outputAdam[destination] = sourceAdam[source];
  outputStats[destination] = sourceStats[source];
  outputStats[config.newCount + destination] = sourceStats[config.capacity + source];
}`;
    const compile = async (code, entryPoint) => {
      const module = this.device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === "error");
      if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
      return this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint } });
    };
    [this.compactionParamPipeline, this.compactionStatePipeline] = await Promise.all([
      compile(parameterShader, "compact_parameters"),
      compile(stateShader, "compact_state"),
    ]);
  }

  async compactTrainStateGpu(keepIndices) {
    const trainState = this.trainState;
    if (!trainState) throw new Error("Compaction requires active WebGPU training buffers.");
    const oldCount = trainState.count;
    const newCount = keepIndices.length;
    if (newCount <= 0 || newCount >= oldCount) return { compacted: false, oldCount, newCount: oldCount, gpu_ms: 0 };
    await this.ensureCompactionPipelines();
    const started = performance.now();
    const transientBytes = 16 + newCount * (4 + 16 * 3 + 96 + 16 * 2);
    const trainingBytes = this.trainingMemorySnapshot().reservedBytes;
    const resultBytes = this.resultRenderMemorySnapshot().reservedBytes;
    const budgetBytes = memoryBudgetBytes();
    const allocation = {
      training_bytes: trainingBytes,
      result_render_bytes: resultBytes,
      candidate_bytes: transientBytes,
      transient_bytes: trainingBytes + resultBytes + transientBytes,
      budget_bytes: budgetBytes,
      within_budget: trainingBytes + resultBytes + transientBytes <= budgetBytes,
    };
    if (!allocation.within_budget) {
      return {
        compacted: false,
        oldCount,
        newCount: oldCount,
        gpu_ms: 0,
        transient_bytes: transientBytes,
        allocation,
        reason: "gpu-transient-budget",
      };
    }
    let configBuffer = null;
    let keepBuffer = null;
    let outputXy = null;
    let outputTransform = null;
    let outputColor = null;
    let outputAdam = null;
    let outputStats = null;
    const createOutput = (size) => this.device.createBuffer({
      size: Math.max(4, size),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const front = trainState.front;
    let scopesPopped = false;
    try {
      this.device.pushErrorScope("out-of-memory");
      this.device.pushErrorScope("validation");
      configBuffer = makeBuffer(
        this.device,
        new Uint32Array([oldCount, newCount, trainState.capacity, 0]),
        GPUBufferUsage.UNIFORM,
      );
      keepBuffer = makeBuffer(
        this.device,
        keepIndices instanceof Uint32Array ? keepIndices : new Uint32Array(keepIndices),
        GPUBufferUsage.STORAGE,
      );
      outputXy = createOutput(newCount * 16);
      outputTransform = createOutput(newCount * 16);
      outputColor = createOutput(newCount * 16);
      outputAdam = createOutput(newCount * 96);
      outputStats = createOutput(newCount * 2 * 16);
      const parameterBindGroup = this.device.createBindGroup({
        layout: this.compactionParamPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: configBuffer } },
          { binding: 1, resource: { buffer: keepBuffer } },
          { binding: 2, resource: { buffer: trainState.xyBuffers[front] } },
          { binding: 3, resource: { buffer: trainState.transformBuffers[front] } },
          { binding: 4, resource: { buffer: trainState.colorBuffers[front] } },
          { binding: 5, resource: { buffer: outputXy } },
          { binding: 6, resource: { buffer: outputTransform } },
          { binding: 7, resource: { buffer: outputColor } },
        ],
      });
      const stateBindGroup = this.device.createBindGroup({
        layout: this.compactionStatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: configBuffer } },
          { binding: 1, resource: { buffer: keepBuffer } },
          { binding: 2, resource: { buffer: trainState.optimizerStateBuffer } },
          { binding: 3, resource: { buffer: trainState.statsBuffer } },
          { binding: 4, resource: { buffer: outputAdam } },
          { binding: 5, resource: { buffer: outputStats } },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      const parameterPass = encoder.beginComputePass();
      parameterPass.setPipeline(this.compactionParamPipeline);
      parameterPass.setBindGroup(0, parameterBindGroup);
      this.dispatchLinear(parameterPass, newCount);
      parameterPass.end();
      const statePass = encoder.beginComputePass();
      statePass.setPipeline(this.compactionStatePipeline);
      statePass.setBindGroup(0, stateBindGroup);
      this.dispatchLinear(statePass, newCount);
      statePass.end();
      for (const target of trainState.xyBuffers) encoder.copyBufferToBuffer(outputXy, 0, target, 0, newCount * 16);
      for (const target of trainState.transformBuffers) encoder.copyBufferToBuffer(outputTransform, 0, target, 0, newCount * 16);
      for (const target of trainState.colorBuffers) encoder.copyBufferToBuffer(outputColor, 0, target, 0, newCount * 16);
      encoder.copyBufferToBuffer(outputAdam, 0, trainState.optimizerStateBuffer, 0, newCount * 96);
      encoder.copyBufferToBuffer(outputStats, 0, trainState.statsBuffer, 0, newCount * 16);
      encoder.copyBufferToBuffer(
        outputStats,
        newCount * 16,
        trainState.statsBuffer,
        trainState.capacity * 16,
        newCount * 16,
      );
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const validationError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      scopesPopped = true;
      if (validationError || oomError) throw validationError || oomError;
      if (this.trainState !== trainState) throw new Error("Training state changed during GPU compaction.");
      trainState.count = newCount;
      trainState.tileReady = false;
      trainState.bindGroupCache?.clear?.();
      return {
        compacted: true,
        oldCount,
        newCount,
        gpu_ms: performance.now() - started,
        transient_bytes: transientBytes,
        allocation,
      };
    } finally {
      if (!scopesPopped) {
        await this.device.popErrorScope().catch(() => null);
        await this.device.popErrorScope().catch(() => null);
      }
      destroyBuffers(configBuffer, keepBuffer, outputXy, outputTransform, outputColor, outputAdam, outputStats);
    }
  }

  async resetImportanceWindowGpu(count) {
    if (!this.trainState?.statsBuffer || count <= 0) return false;
    const boundedCount = Math.min(this.trainState.count, Math.max(0, Math.round(count)));
    if (boundedCount <= 0) return false;
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(
      this.trainState.statsBuffer,
      this.trainState.capacity * 4 * 4,
      boundedCount * 4 * 4,
    );
    this.device.queue.submit([encoder.finish()]);
    this.trainState.currentVisibilityWindow = {
      reset_step: state.metrics?.steps_done || 0,
      count: boundedCount,
      buffer: "importance-stats",
    };
    return true;
  }

  async readTrainedLayerOrder(params) {
    if (!this.trainState || this.trainState.count !== params?.count) return false;
    const count = params.count;
    const bytes = count * 4 * 4;
    const readBuffer = this.device.createBuffer({
      size: Math.max(4, bytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(
        this.trainState.transformBuffers[this.trainState.front],
        0,
        readBuffer,
        0,
        bytes,
      );
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const transforms = new Float32Array(readBuffer.getMappedRange()).slice();
      params.detailTags = new Float32Array(count);
      params.depthOrder = new Float32Array(count);
      for (let index = 0; index < count; index += 1) {
        const packedTag = transforms[index * 4 + 3];
        params.detailTags[index] = Math.floor(packedTag);
        params.depthOrder[index] = packedLayerOrder(packedTag);
      }
      return true;
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
      readBuffer.destroy();
    }
  }

  async readImportanceData(count, { includeSummary = true } = {}) {
    if (!this.trainState?.statsBuffer || count <= 0) return null;
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
      const coverage = includeSummary ? [] : null;
      const influence = includeSummary ? [] : null;
      const residual = includeSummary ? [] : null;
      const normalizedInfluence = new Float32Array(count);
      const integratedInfluenceBySplat = new Float32Array(count);
      const observedCoverage = new Float32Array(count);
      let nonfiniteCount = 0;
      for (let i = 0; i < values.length; i += 4) {
        if (![values[i], values[i + 1], values[i + 2], values[i + 3]].every(Number.isFinite)) nonfiniteCount += 1;
        const observed = Number.isFinite(values[i]) ? Math.max(0, values[i]) : 0;
        const integratedInfluence = Number.isFinite(values[i + 1]) ? Math.max(0, values[i + 1]) : 0;
        const residualMass = Number.isFinite(values[i + 2]) ? Math.max(0, values[i + 2]) : 0;
        if (includeSummary) {
          coverage.push(observed);
          influence.push(integratedInfluence);
          residual.push(residualMass / Math.max(observed, 1));
        }
        normalizedInfluence[i / 4] = integratedInfluence / Math.max(observed, 1);
        observedCoverage[i / 4] = observed;
        integratedInfluenceBySplat[i / 4] = integratedInfluence;
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
      return {
        normalizedInfluence,
        integratedInfluence: integratedInfluenceBySplat,
        observedCoverage,
        summary: includeSummary ? {
          count,
          nonfinite_count: nonfiniteCount,
          coverage: summarize(coverage),
          influence: summarize(influence),
          residual: summarize(residual),
        } : null,
      };
    } finally {
      readBuffer.destroy();
    }
  }

  async readImportanceSummary(count) {
    return (await this.readImportanceData(count))?.summary || null;
  }

  trainingBuffers() {
    if (!this.trainState) return [];
    return [...new Set([
      this.trainState.configBuffer,
      this.trainState.batchConfigBuffer,
      this.trainState.presentConfigBuffer,
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
      this.trainState.fixedPointGradientControlBuffer,
      this.trainState.fixedPointGradientReadbackBuffer,
      this.trainState.segmentedPartialGradientBuffer,
      this.trainState.segmentedReferenceCountsBuffer,
      this.trainState.segmentedReferenceOffsetsBuffer,
      this.trainState.segmentedReferenceCursorsBuffer,
      this.trainState.segmentedReferencesBuffer,
      this.trainState.exactBackwardTelemetryBuffer,
      this.trainState.exactBackwardTelemetryReadbackBuffer,
      this.trainState.ssimTileBuffer,
      this.trainState.optimizerStateBuffer,
      this.trainState.exactGradientBuffer,
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

  async readTrainedColors(params) {
    if (!this.trainState || this.trainState.count !== params.count) return;
    const front = this.trainState.front;
    const xyBytes = params.count * 4 * 4;
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
      const positions = new Float32Array(mapped, 0, params.count * 4);
      if (!params.virtualDepth || params.virtualDepth.length !== params.count) params.virtualDepth = new Float32Array(params.count);
      if (!params.brushTaper || params.brushTaper.length !== params.count) {
        params.brushTaper = new Float32Array(params.count).fill(DEFAULT_LAYERED_BRUSH_TAPER);
      }
      for (let i = 0; i < params.count; i += 1) {
        params.xy[i * 2] = positions[i * 4];
        params.xy[i * 2 + 1] = positions[i * 4 + 1];
        params.virtualDepth[i] = params.virtualDepthEnabled ? positions[i * 4 + 2] : 0;
        if (params.brushWidthTaperEnabled) {
          params.brushTaper[i] = Math.max(0, Math.min(1, positions[i * 4 + 2]));
        }
      }
      const transforms = new Float32Array(mapped, xyBytes, params.count * 4);
      if (!params.detailTags || params.detailTags.length !== params.count) {
        params.detailTags = new Float32Array(params.count);
      }
      if (!params.depthOrder || params.depthOrder.length !== params.count) {
        params.depthOrder = new Float32Array(params.count);
      }
      let detailSplatCount = 0;
      let detailAnisotropyMax = 1;
      let surfaceAnisotropyMax = 1;
      for (let i = 0; i < params.count; i += 1) {
        params.scale[i * 2] = transforms[i * 4];
        params.scale[i * 2 + 1] = transforms[i * 4 + 1];
        params.theta[i] = transforms[i * 4 + 2];
        const packedTag = transforms[i * 4 + 3];
        params.detailTags[i] = Math.floor(packedTag);
        params.depthOrder[i] = packedLayerOrder(packedTag);
        const minor = Math.max(MIN_SPLAT_SCALE, Math.min(transforms[i * 4], transforms[i * 4 + 1]));
        const anisotropy = Math.max(transforms[i * 4], transforms[i * 4 + 1]) / minor;
        if (Math.floor(packedTag) > 1.5) {
          detailSplatCount += 1;
          detailAnisotropyMax = Math.max(detailAnisotropyMax, anisotropy);
        } else {
          surfaceAnisotropyMax = Math.max(surfaceAnisotropyMax, anisotropy);
        }
      }
      if (state.metrics) {
        state.metrics.detail_splat_count = detailSplatCount;
        state.metrics.detail_splat_ratio = detailSplatCount / Math.max(1, params.count);
        state.metrics.detail_anisotropy_max = detailAnisotropyMax;
        state.metrics.surface_anisotropy_max = surfaceAnisotropyMax;
        if (params.brushWidthTaperEnabled) {
          let minimumTaper = 1;
          let maximumTaper = 0;
          let taperSum = 0;
          for (let i = 0; i < params.count; i += 1) {
            const taper = params.brushTaper[i];
            minimumTaper = Math.min(minimumTaper, taper);
            maximumTaper = Math.max(maximumTaper, taper);
            taperSum += taper;
          }
          state.metrics.brush_taper_stats = {
            minimum: minimumTaper,
            mean: taperSum / Math.max(1, params.count),
            maximum: maximumTaper,
            learned_on_gpu: true,
          };
        } else {
          state.metrics.brush_taper_stats = null;
        }
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
