const webGpuPreviewFeatureDefinitions = new Map();

class WebGpuPreviewFeatureAdapter {
  constructor(owner, sourceName, methods) {
    this.owner = owner;
    this.sourceName = sourceName;
    this.methods = methods;
  }

  invoke(name, args) {
    const descriptor = this.methods.get(name);
    if (!descriptor) throw new Error(`WebGpuPreview feature method missing: ${this.sourceName}.${name}`);
    return descriptor.value.apply(this.owner, args);
  }
}

function registerWebGpuPreviewFeature(sourcePrototype, sourceName) {
  if (webGpuPreviewFeatureDefinitions.has(sourceName)) {
    throw new Error(`WebGpuPreview feature already registered: ${sourceName}`);
  }
  const methods = new Map();
  for (const name of Object.getOwnPropertyNames(sourcePrototype)) {
    if (name === "constructor") continue;
    if (Object.prototype.hasOwnProperty.call(WebGpuPreview.prototype, name)) {
      throw new Error(`WebGpuPreview method collision: ${sourceName}.${name}`);
    }
    const descriptor = Object.getOwnPropertyDescriptor(sourcePrototype, name);
    if (typeof descriptor?.value !== "function") {
      throw new Error(`WebGpuPreview feature method must be callable: ${sourceName}.${name}`);
    }
    methods.set(name, descriptor);
    Object.defineProperty(
      WebGpuPreview.prototype,
      name,
      {
        configurable: descriptor.configurable,
        enumerable: descriptor.enumerable,
        writable: descriptor.writable,
        value(...args) {
          const feature = this.webGpuFeatureAdapters?.get(sourceName);
          if (!feature) throw new Error(`WebGpuPreview feature unavailable: ${sourceName}`);
          return feature.invoke(name, args);
        },
      },
    );
  }
  webGpuPreviewFeatureDefinitions.set(sourceName, methods);
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
    this.webGpuFeatureAdapters = new Map(
      Array.from(webGpuPreviewFeatureDefinitions, ([sourceName, methods]) => [
        sourceName,
        new WebGpuPreviewFeatureAdapter(this, sourceName, methods),
      ]),
    );
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
    const shader = Image2SplatPaintPreviewShaders.renderPreview();
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
    const shader = Image2SplatPaintPreviewShaders.adaptiveGridInitialization();
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
    const shader = Image2SplatPaintPreviewShaders.discreteLayers();
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
    const shader = Image2SplatPaintPreviewShaders.presentedStatePack();
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
    const shader = Image2SplatPaintPreviewShaders.presentCanvas();
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
    const {
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
    } = Image2SplatPaintTrainingPipelineShaders.create.call(this, {
      fixedPointExactGradientEnabled,
      inverseScaleOptimizationEnabled,
      optimizerStatsDeclaration,
      segmentedExactBackwardEnabled,
    });
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
    const shader = Image2SplatPaintMetricShaders.trainingResidualMap();
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
    const shader = Image2SplatPaintMetricShaders.pixelMetrics();
    const module = this.device.createShaderModule({ code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join(" | "));
    this.pixelMetricsPipeline = await this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "metrics" } });
  }

  async ensureVirtualCameraMetricsPipeline() {
    if (this.virtualCameraMetricsPipeline) return;
    const shader = Image2SplatPaintMetricShaders.virtualCameraMetrics();
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
    const shader = Image2SplatPaintMetricShaders.overlapMetrics({ hiddenRgbBinding, hiddenRgbShader });
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
    const shader = Image2SplatPaintMetricShaders.alphaLoss();
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
    const shader = Image2SplatPaintDensityShader.create();
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
