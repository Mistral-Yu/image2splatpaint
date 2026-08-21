class WebGpuOptimizerRuntime {
  async ensureOptimizerResetPipeline() {
    if (this.optimizerResetPipeline && this.optimizerSourceResetPipeline) return;
    const shader = `
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> config: array<f32>;
@group(0) @binding(1) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> adam: array<AdamState>;
const SOURCE_MASK = 0x3fffffffu;
const ROLE_SOURCE_MASK = 0x60000000u;

fn reset_state(adcResetStep: f32) -> AdamState {
  return AdamState(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, adcResetStep), vec4<f32>(0.0));
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

  async trainStepRenderGradientGpu(image, params, learningRates, {
    sync = true,
    currentStepOverride = null,
    batchEncoder = null,
    batchConfigSlot = -1,
    viewOverride = null,
    clearExactGradient = true,
    applyOptimizer = true,
    recordTrainingStep = true,
    gradientNormalization = 1,
    gradientChannels = null,
    frontDetailLoss = false,
    sourceDomainReprojection = false,
    suppressSgldNoise = false,
    cameraCovariance3d = false,
    virtualDepthUpdateDue = false,
    virtualDepthUpdateInterval = DEFAULT_VIRTUAL_DEPTH_UPDATE_INTERVAL,
    virtualDepthCameraConfidence = 0,
    gofDensity = false,
  } = {}) {
    await this.ensureRenderGradientPipelines();
    if (params.discreteLayersEnabled) await this.ensureDiscreteLayerPipelines();
    const variants = phase33Variants();
    const phase37 = phase37Variants();
    const phase38 = phase38Variants();
    const phase39 = phase39Variants();
    const phase40 = phase40Variants();
    const currentStep = Number.isFinite(currentStepOverride)
      ? Math.max(1, Math.round(currentStepOverride))
      : (state.metrics?.steps_done || 0) + 1;
    const requestedSteps = state.metrics?.steps_requested || 1;
    const profileLabels = !batchEncoder && recordTrainingStep && this.performanceProfile.timestampQuery
      ? performanceProfileLabels(currentStep, requestedSteps)
      : [];
    const profileSample = profileLabels.length > 0
      ? { step: currentStep, labels: profileLabels, queryCount: 0, stages: [] }
      : null;
    const effectiveSync = !batchEncoder && (sync || Boolean(profileSample));
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
    const useExactBackward =
      qualityVariants.exactBackward ||
      algorithmUsesPaintKernel() ||
      !this.renderGradientPipeline;
    const layerSettings = layerOptimizationSettings(
      currentStep,
      requestedSteps,
      trainingStage,
      state.metrics?.phase46_variants || phase46Variants(),
      Boolean(params.opaqueLayered),
      state.metrics?.opaque_paint_late_settle?.fraction,
    );
    const scheduledTiltStep = virtualTiltStepSpec(currentStep, trainingStage, requestedSteps);
    const requestedTiltStep = viewOverride === "front"
      ? { ...scheduledTiltStep, enabled: false, pitchRadians: 0, yawRadians: 0, pitchDegrees: 0, yawDegrees: 0, weight: 1 }
      : viewOverride || scheduledTiltStep;
    const sharedCameraDistance = requestedTiltStep.enabled && requestedTiltStep.autoCameraDistance
      ? sharedTiltOrbitRadius(
        workImage.width,
        workImage.height,
        requestedTiltStep.maxAngleDegrees,
        requestedTiltStep.cameraCount,
        requestedTiltStep.fovDegrees,
      )
      : requestedTiltStep.cameraDistance;
    const tiltStep = useExactBackward && variants.ewa2x2
      ? { ...requestedTiltStep, cameraDistance: sharedCameraDistance }
      : { ...requestedTiltStep, enabled: false, pitchRadians: 0, yawRadians: 0, pitchDegrees: 0, yawDegrees: 0, weight: 1 };
    const boundedDepthEnabled = Boolean(
      useExactBackward &&
      params.virtualDepthEnabled &&
      state.metrics?.virtual_camera_sampling?.enabled,
    );
    const config = new Float32Array(TRAIN_CONFIG_FLOATS);
    config.set([
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
      phase37.progressiveGradientLoss || frontDetailLoss ? 1 : 0,
      phase37.progressiveGradientLoss || frontDetailLoss ? 0.02 : 0,
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
      Math.max(0, Number(gradientChannels?.geometry ?? 1)),
      Math.max(0, Number(gradientChannels?.appearance ?? 1)),
      Math.max(0, Number(gradientChannels?.density ?? 1)),
      Math.max(0, Number(gradientChannels?.depth ?? 1)),
      coarseStepLimit,
      0,
      0,
      0,
      0,
      phase40.localColorAnchorWeight,
      els.trainLayerOrder.checked ? 1 : 0,
      phase40.alphaLoss ? phase40.alphaLossWeight : 0,
      0,
      0,
      0,
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
      Math.max(0.0001, Math.min(1, Number(gradientNormalization) || 1)),
    ], 0);
    config[64] = tiltStep.fovDegrees || DEFAULT_SHARED_CAMERA_FOV_DEGREES;
    config[65] = tiltStep.enabled && tiltStep.planeConstrained !== false ? 1 : 0;
    config[66] = tiltStep.enabled && tiltStep.invalidRegionMode === "black-loss" ? 1 : 0;
    config[67] = boundedDepthEnabled ? 1 : 0;
    config[68] = boundedDepthEnabled ? Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS : 0;
    config[69] = boundedDepthEnabled
      ? Number(params.virtualDepthCenterWeight ?? DEFAULT_VIRTUAL_DEPTH_CENTER_WEIGHT)
      : 0;
    config[70] = boundedDepthEnabled
      ? Number(params.virtualDepthSmoothnessWeight ?? DEFAULT_VIRTUAL_DEPTH_SMOOTHNESS_WEIGHT)
      : 0;
    config[71] = boundedDepthEnabled
      ? Number(params.virtualDepthLearningRate ?? DEFAULT_VIRTUAL_DEPTH_LEARNING_RATE)
      : 0;
    config[72] = tiltStep.enabled && sourceDomainReprojection ? 1 : 0;
    config[73] = suppressSgldNoise ? 1 : 0;
    config[74] = tiltStep.enabled && cameraCovariance3d ? 1 : 0;
    config[75] = 1e-4;
    config[76] = boundedDepthEnabled && virtualDepthUpdateDue ? 1 : 0;
    config[77] = Math.max(1, Math.round(virtualDepthUpdateInterval));
    config[78] = gofDensity ? 1 : 0;
    config[79] = trainingStage === "coarse" ? PHASE_ONE_SHAPE_LR_MULTIPLIER : 1;
    config[80] = stageMinimumScale(
      workImage,
      state.metrics?.initial_splats || params.count,
      trainingStage,
      variants.stageMinScaleRatio,
    );
    config[81] = params.discreteLayersEnabled ? 1 : 0;
    config[82] = Math.max(MIN_DISCRETE_LAYER_COUNT, Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(params.discreteLayerCount || DEFAULT_DISCRETE_LAYER_COUNT)));
    config[83] = Math.max(0, Math.min(config[82] - 1, Math.round(params.discreteLayerMoveRadius ?? DEFAULT_DISCRETE_LAYER_MOVE_RADIUS)));
    config[84] = params.layerAwareAccumulationEnabled ? 1 : 0;
    config[49] = experimentalDensifySteps(state.metrics?.steps_requested || 1);
    configurePaintKernel(config, params);
    const surfaceLayerSort = scaleBiasedSurfaceLayerSortSchedule(
      currentStep,
      requestedSteps,
      {
        enabled: params.surfaceLayerPriorEnabled,
        p1Interval: params.surfaceLayerPriorP1Interval,
        p2Interval: params.surfaceLayerPriorP2Interval,
        p3Interval: params.surfaceLayerPriorP3Interval,
        untilFraction: params.surfaceLayerPriorUntilFraction,
      },
    );
    config[45] = params.layerOrderEnabled ? 1 : 0;
    config[105] = surfaceLayerSort.due ? 1 : 0;
    config[106] = clampNumber(
      params.surfaceLayerPriorLayers,
      MIN_DISCRETE_LAYER_COUNT,
      MAX_DISCRETE_LAYER_COUNT,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
    );
    config[108] = boundedDepthEnabled
      ? Math.max(0, Math.min(1, Number(virtualDepthCameraConfidence) || 0))
      : 0;
    config[109] = boundedDepthEnabled
      ? Math.max(0.00001, Number(params.virtualDepthPriorDelta) || DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA)
      : DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA;
    config[110] = boundedDepthEnabled && params.virtualDepthSoftConstraintEnabled !== false ? 1 : 0;
    const relativeScaleGuard = this.trainState.phaseRelativeScaleGuard;
    config[120] = relativeScaleGuard?.enabled ? relativeScaleGuard.floor : 0;
    config[121] = relativeScaleGuard?.enabled ? relativeScaleGuard.strength : 0;
    config[122] = relativeScaleGuard?.enabled ? 1 : 0;
    const sourceDomainActive = config[72] > 0.5;
    const segmentedExactBackwardActive = Boolean(
      useExactBackward &&
      !sourceDomainActive &&
      els.tileCullingToggle.checked &&
      this.trainState.segmentedExactBackward?.enabled,
    );
    const fixedPointExactGradientActive = Boolean(
      useExactBackward &&
      !segmentedExactBackwardActive &&
      this.trainState.fixedPointExactGradient?.enabled,
    );
    let errorScopeOpen = false;
    const profileWallStarted = profileSample ? performance.now() : 0;
    let profileEncodeSubmitMs = 0;
    let profileSyncWaitMs = 0;
    let profileReadbackMs = 0;
    let profileQueueWaitCount = 0;
    let profileReadbackCount = 0;
    let profileReadbackBytes = 0;
    let profileCandidateStats = null;
    let profileFixedPointGradientStats = null;
    if (effectiveSync) {
      this.device.pushErrorScope("validation");
      errorScopeOpen = true;
    }
    try {
      const encoder = batchEncoder || this.device.createCommandEncoder();
      if (batchEncoder) {
        if (batchConfigSlot < 0 || batchConfigSlot >= MAX_TRAIN_BATCH_SIZE) {
          throw new Error(`Invalid GPU batch config slot ${batchConfigSlot}.`);
        }
        const configOffset = batchConfigSlot * TRAIN_CONFIG_BYTES;
        this.device.queue.writeBuffer(this.trainState.batchConfigBuffer, configOffset, config);
        encoder.copyBufferToBuffer(
          this.trainState.batchConfigBuffer,
          configOffset,
          this.trainState.configBuffer,
          0,
          TRAIN_CONFIG_BYTES,
        );
      } else {
        this.device.queue.writeBuffer(this.trainState.configBuffer, 0, config);
      }
      if (els.tileCullingToggle.checked) {
        await this.prepareTileLists(workImage, params, {
          encoder,
          profileSample,
          writeConfig: false,
        });
      }
      const front = this.trainState.front;
      const optimizerStatsEntries = () => [{ binding: 8, resource: { buffer: this.trainState.statsBuffer } }];
      const defaultRenderChoice = this.renderStatePipelineChoice();
      const renderChoice = sourceDomainActive
        ? { pipeline: this.renderStatePipeline, cooperative: false, supported: defaultRenderChoice.supported }
        : defaultRenderChoice;
      const exactBackwardPipeline = sourceDomainActive
        ? this.sourceDomainBackwardPipeline
        : this.exactAlphaBackwardPipeline;
      const bindGroupKeyBase = [
        front,
        trainingStage,
        sourceDomainActive ? 1 : 0,
      ].join(":");
      const cachedBindGroup = (kind, pipeline, entriesFactory, variant = "") => this.trainBindGroup(
        `${kind}:${bindGroupKeyBase}:${variant}`,
        pipeline,
        entriesFactory,
      );
      const renderBindGroup = cachedBindGroup("render", renderChoice.pipeline, () => [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
          { binding: 5, resource: { buffer: this.trainState.tileOffsetsBuffer } },
          { binding: 6, resource: { buffer: this.trainState.tileIndicesBuffer } },
          { binding: 7, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 8, resource: { buffer: this.trainState.alphaStateBuffer } },
        ], renderChoice.cooperative ? "cooperative" : "linear");
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
      const ssimHorizontalBindGroup = cachedBindGroup("ssim-horizontal", this.ssimHorizontalPipeline, ssimFullEntries);
      const ssimBindGroup = cachedBindGroup("ssim-vertical", this.ssimTilePipeline, ssimFilterEntries);
      const ssimBackwardHorizontalBindGroup = cachedBindGroup(
        "ssim-backward-horizontal",
        this.ssimBackwardHorizontalPipeline,
        ssimFilterEntries,
      );
      const ssimBackwardVerticalBindGroup = cachedBindGroup(
        "ssim-backward-vertical",
        this.ssimBackwardVerticalPipeline,
        ssimFullEntries,
      );
      const alphaSsimTileBindGroup = cachedBindGroup("alpha-ssim-tiles", this.alphaSsimTilePipeline, ssimFullEntries);
      const optimizerPipeline = useExactBackward
        ? null
        : (phase37.parallelOptimizer ? this.parallelRenderGradientPipeline : this.renderGradientPipeline);
      const optimizerBindGroup = useExactBackward ? null : cachedBindGroup("legacy-optimizer", optimizerPipeline, () => [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
          { binding: 4, resource: { buffer: targetBuffer } },
          { binding: 5, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 6, resource: { buffer: this.trainState.ssimTileBuffer } },
          { binding: 7, resource: { buffer: this.trainState.optimizerStateBuffer } },
          ...optimizerStatsEntries(),
        ], phase37.parallelOptimizer ? "parallel" : "serial");
      const lossGradientBindGroup = useExactBackward ? cachedBindGroup("loss-gradient", this.lossGradientPipeline, () => [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: targetBuffer } },
          { binding: 2, resource: { buffer: targetAlphaBuffer } },
          { binding: 3, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 4, resource: { buffer: this.trainState.ssimTileBuffer } },
          { binding: 5, resource: { buffer: this.trainState.lossGradientBuffer } },
        ]) : null;
      const exactBackwardBindGroup = useExactBackward ? cachedBindGroup("exact-backward", exactBackwardPipeline, () => [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
          { binding: 4, resource: { buffer: this.trainState.tileOffsetsBuffer } },
          { binding: 5, resource: { buffer: this.trainState.tileIndicesBuffer } },
          { binding: 6, resource: { buffer: this.trainState.lossGradientBuffer } },
          ...(segmentedExactBackwardActive
            ? []
            : [{ binding: 7, resource: { buffer: this.trainState.exactGradientBuffer } }]),
          { binding: 8, resource: { buffer: this.trainState.alphaStateBuffer } },
          ...(segmentedExactBackwardActive
            ? [{ binding: 9, resource: { buffer: this.trainState.segmentedPartialGradientBuffer } }]
            : fixedPointExactGradientActive
              ? [{ binding: 9, resource: { buffer: this.trainState.fixedPointGradientControlBuffer } }]
              : []),
        ], sourceDomainActive ? "source" : "alpha") : null;
      const segmentedReferenceCountBindGroup = segmentedExactBackwardActive
        ? cachedBindGroup("segmented-ref-count", this.segmentedReferenceCountPipeline, () => [
            { binding: 0, resource: { buffer: this.trainState.configBuffer } },
            { binding: 1, resource: { buffer: this.trainState.tileOffsetsBuffer } },
            { binding: 2, resource: { buffer: this.trainState.tileIndicesBuffer } },
            { binding: 3, resource: { buffer: this.trainState.segmentedReferenceCountsBuffer } },
          ])
        : null;
      const segmentedReferencePrefixBindGroup = segmentedExactBackwardActive
        ? cachedBindGroup("segmented-ref-prefix", this.segmentedReferencePrefixPipeline, () => [
            { binding: 0, resource: { buffer: this.trainState.configBuffer } },
            { binding: 3, resource: { buffer: this.trainState.segmentedReferenceCountsBuffer } },
            { binding: 4, resource: { buffer: this.trainState.segmentedReferenceOffsetsBuffer } },
          ])
        : null;
      const segmentedReferenceFillBindGroup = segmentedExactBackwardActive
        ? cachedBindGroup("segmented-ref-fill", this.segmentedReferenceFillPipeline, () => [
            { binding: 0, resource: { buffer: this.trainState.configBuffer } },
            { binding: 1, resource: { buffer: this.trainState.tileOffsetsBuffer } },
            { binding: 2, resource: { buffer: this.trainState.tileIndicesBuffer } },
            { binding: 4, resource: { buffer: this.trainState.segmentedReferenceOffsetsBuffer } },
            { binding: 5, resource: { buffer: this.trainState.segmentedReferenceCursorsBuffer } },
            { binding: 6, resource: { buffer: this.trainState.segmentedReferencesBuffer } },
          ])
        : null;
      const segmentedGradientReduceBindGroup = segmentedExactBackwardActive
        ? cachedBindGroup("segmented-gradient-reduce", this.segmentedGradientReducePipeline, () => [
            { binding: 0, resource: { buffer: this.trainState.configBuffer } },
            { binding: 1, resource: { buffer: this.trainState.segmentedReferenceOffsetsBuffer } },
            { binding: 2, resource: { buffer: this.trainState.segmentedReferencesBuffer } },
            { binding: 3, resource: { buffer: this.trainState.segmentedPartialGradientBuffer } },
            { binding: 4, resource: { buffer: this.trainState.exactGradientBuffer } },
          ])
        : null;
      const virtualOrderPenaltyBindGroup = useExactBackward && tiltStep.enabled && tiltStep.orderPenaltyWeight > 0
        ? cachedBindGroup("virtual-order", this.virtualOrderPenaltyPipeline, () => [
              { binding: 0, resource: { buffer: this.trainState.configBuffer } },
              { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
              { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
              { binding: 4, resource: { buffer: this.trainState.exactGradientBuffer } },
              ...(fixedPointExactGradientActive
                ? [{ binding: 5, resource: { buffer: this.trainState.fixedPointGradientControlBuffer } }]
                : []),
            ])
        : null;
      const brushLocalColorFlowBindGroup = useExactBackward && config[111] > 0
        ? cachedBindGroup("brush-local-color-flow", this.brushLocalColorFlowPipeline, () => [
            { binding: 0, resource: { buffer: this.trainState.configBuffer } },
            { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
            { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
            { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
            { binding: 4, resource: { buffer: this.trainState.exactGradientBuffer } },
            ...(fixedPointExactGradientActive
              ? [{ binding: 5, resource: { buffer: this.trainState.fixedPointGradientControlBuffer } }]
              : []),
          ])
        : null;
      const exactBackwardTelemetryBindGroup = useExactBackward && profileSample && !sourceDomainActive ? cachedBindGroup("exact-telemetry", this.exactBackwardTelemetryPipeline, () => [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.tileOffsetsBuffer } },
          { binding: 4, resource: { buffer: this.trainState.tileIndicesBuffer } },
          { binding: 5, resource: { buffer: this.trainState.alphaStateBuffer } },
          { binding: 6, resource: { buffer: this.trainState.exactBackwardTelemetryBuffer } },
        ]) : null;
      const exactOptimizerBindGroup = useExactBackward ? cachedBindGroup("exact-optimizer", this.exactOptimizerPipeline, () => [
          { binding: 0, resource: { buffer: this.trainState.configBuffer } },
          { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
          { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
          { binding: 3, resource: { buffer: this.trainState.colorBuffers[front] } },
          { binding: 4, resource: { buffer: targetBuffer } },
          { binding: 5, resource: { buffer: this.trainState.pixelStateBuffer } },
          { binding: 7, resource: { buffer: this.trainState.optimizerStateBuffer } },
          ...optimizerStatsEntries(),
          { binding: 9, resource: { buffer: this.trainState.exactGradientBuffer } },
          { binding: 10, resource: { buffer: this.trainState.tileControlBuffer } },
        ]) : null;
      const discreteLayerPassDue = Boolean(
        params.discreteLayersEnabled &&
        params.discreteLayerMoveRadius > 0 &&
        normalizedKernelShape(params.kernelShape) === "opaque-brush" &&
        applyOptimizer &&
        opaquePaintStructuralMutationAllowed(
          currentStep,
          requestedSteps,
          Boolean(params.opaqueLayered),
          state.metrics?.opaque_paint_late_settle?.fraction,
        ) &&
        Boolean(params.opaqueLayered) &&
        currentStep >= experimentalDensifySteps(requestedSteps) &&
        opaquePaintDetailRecoveryDue(
          currentStep,
          requestedSteps,
          OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL,
        )
      );
      const discreteLayerEntries = () => [
        { binding: 0, resource: { buffer: this.trainState.configBuffer } },
        { binding: 1, resource: { buffer: this.trainState.xyBuffers[front] } },
        { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
        { binding: 3, resource: { buffer: this.trainState.tileOffsetsBuffer } },
        { binding: 4, resource: { buffer: this.trainState.tileIndicesBuffer } },
        { binding: 5, resource: { buffer: this.trainState.exactGradientBuffer } },
        { binding: 6, resource: { buffer: targetBuffer } },
        { binding: 7, resource: { buffer: this.trainState.colorBuffers[front] } },
        { binding: 8, resource: { buffer: this.trainState.optimizerStateBuffer } },
        { binding: 9, resource: { buffer: this.trainState.statsBuffer } },
      ];
      const discreteLayerAssignBindGroup = discreteLayerPassDue
        ? cachedBindGroup("discrete-layer-assign", this.discreteLayerAssignPipeline, discreteLayerEntries)
        : null;
      const discreteLayerCommitBindGroup = discreteLayerPassDue
        ? cachedBindGroup("discrete-layer-commit", this.discreteLayerCommitPipeline, () => [
            { binding: 0, resource: { buffer: this.trainState.configBuffer } },
            { binding: 2, resource: { buffer: this.trainState.transformBuffers[front] } },
            { binding: 5, resource: { buffer: this.trainState.exactGradientBuffer } },
          ])
        : null;
      const renderPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "render"));
      renderPass.setPipeline(renderChoice.pipeline);
      renderPass.setBindGroup(0, renderBindGroup);
      if (renderChoice.cooperative) {
        renderPass.dispatchWorkgroups(Math.ceil(workImage.width / TILE_SIZE), Math.ceil(workImage.height / TILE_SIZE));
      } else {
        this.dispatchLinear(renderPass, Math.ceil((workImage.width * workImage.height) / 64));
      }
      renderPass.end();
      const ssimHorizontalPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "ssim-horizontal"));
      ssimHorizontalPass.setPipeline(this.ssimHorizontalPipeline);
      ssimHorizontalPass.setBindGroup(0, ssimHorizontalBindGroup);
      this.dispatchLinear(ssimHorizontalPass, Math.ceil((workImage.width * workImage.height) / 64));
      ssimHorizontalPass.end();
      const ssimPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "ssim"));
      ssimPass.setPipeline(this.ssimTilePipeline);
      ssimPass.setBindGroup(0, ssimBindGroup);
      this.dispatchLinear(ssimPass, Math.ceil((workImage.width * workImage.height) / 64));
      ssimPass.end();
      const ssimBackwardHorizontalPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "ssim-backward-horizontal"));
      ssimBackwardHorizontalPass.setPipeline(this.ssimBackwardHorizontalPipeline);
      ssimBackwardHorizontalPass.setBindGroup(0, ssimBackwardHorizontalBindGroup);
      this.dispatchLinear(ssimBackwardHorizontalPass, Math.ceil((workImage.width * workImage.height) / 64));
      ssimBackwardHorizontalPass.end();
      const ssimBackwardVerticalPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "ssim-backward-vertical"));
      ssimBackwardVerticalPass.setPipeline(this.ssimBackwardVerticalPipeline);
      ssimBackwardVerticalPass.setBindGroup(0, ssimBackwardVerticalBindGroup);
      this.dispatchLinear(ssimBackwardVerticalPass, Math.ceil((workImage.width * workImage.height) / 64));
      ssimBackwardVerticalPass.end();
      const alphaSsimPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "alpha-ssim"));
      alphaSsimPass.setPipeline(this.alphaSsimTilePipeline);
      alphaSsimPass.setBindGroup(0, alphaSsimTileBindGroup);
      alphaSsimPass.dispatchWorkgroups(Math.ceil(workImage.width / 8), Math.ceil(workImage.height / 8));
      alphaSsimPass.end();
      if (useExactBackward) {
        if (clearExactGradient) encoder.clearBuffer(this.trainState.exactGradientBuffer);
        const lossPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "loss-gradient"));
        lossPass.setPipeline(this.lossGradientPipeline);
        lossPass.setBindGroup(0, lossGradientBindGroup);
        this.dispatchLinear(lossPass, Math.ceil((workImage.width * workImage.height) / 64));
        lossPass.end();
        if (profileSample && !sourceDomainActive) {
          encoder.clearBuffer(this.trainState.exactBackwardTelemetryBuffer);
          const telemetryPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "exact-backward-telemetry"));
          telemetryPass.setPipeline(this.exactBackwardTelemetryPipeline);
          telemetryPass.setBindGroup(0, exactBackwardTelemetryBindGroup);
          telemetryPass.dispatchWorkgroups(Math.ceil(workImage.width / 8), Math.ceil(workImage.height / 8));
          telemetryPass.end();
        }
        if (segmentedExactBackwardActive) {
          encoder.clearBuffer(this.trainState.segmentedReferenceCountsBuffer);
          encoder.clearBuffer(this.trainState.segmentedReferenceCursorsBuffer);
          const referenceCountPass = encoder.beginComputePass(
            this.profilePassDescriptor(profileSample, "segmented-reference-count"),
          );
          referenceCountPass.setPipeline(this.segmentedReferenceCountPipeline);
          referenceCountPass.setBindGroup(0, segmentedReferenceCountBindGroup);
          this.dispatchLinear(
            referenceCountPass,
            Math.ceil(this.trainState.tileIndexCapacity / 64),
          );
          referenceCountPass.end();
          const referencePrefixPass = encoder.beginComputePass(
            this.profilePassDescriptor(profileSample, "segmented-reference-prefix"),
          );
          referencePrefixPass.setPipeline(this.segmentedReferencePrefixPipeline);
          referencePrefixPass.setBindGroup(0, segmentedReferencePrefixBindGroup);
          referencePrefixPass.dispatchWorkgroups(1);
          referencePrefixPass.end();
          const referenceFillPass = encoder.beginComputePass(
            this.profilePassDescriptor(profileSample, "segmented-reference-fill"),
          );
          referenceFillPass.setPipeline(this.segmentedReferenceFillPipeline);
          referenceFillPass.setBindGroup(0, segmentedReferenceFillBindGroup);
          this.dispatchLinear(
            referenceFillPass,
            Math.ceil(this.trainState.tileIndexCapacity / 64),
          );
          referenceFillPass.end();
        }
        const backwardPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "exact-backward"));
        backwardPass.setPipeline(exactBackwardPipeline);
        backwardPass.setBindGroup(0, exactBackwardBindGroup);
        if (sourceDomainActive) {
          this.dispatchLinear(backwardPass, Math.ceil((workImage.width * workImage.height) / 64));
        } else {
          backwardPass.dispatchWorkgroups(
            Math.ceil(workImage.width / (this.quadExactBackwardEnabled ? TILE_SIZE : 8)),
            Math.ceil(workImage.height / (this.quadExactBackwardEnabled ? TILE_SIZE : 8)),
          );
        }
        backwardPass.end();
        if (segmentedExactBackwardActive) {
          const segmentedReducePass = encoder.beginComputePass(
            this.profilePassDescriptor(profileSample, "segmented-gradient-reduce"),
          );
          segmentedReducePass.setPipeline(this.segmentedGradientReducePipeline);
          segmentedReducePass.setBindGroup(0, segmentedGradientReduceBindGroup);
          this.dispatchLinear(segmentedReducePass, params.count);
          segmentedReducePass.end();
        }
        if (virtualOrderPenaltyBindGroup) {
          const orderPenaltyPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "virtual-order-penalty"));
          orderPenaltyPass.setPipeline(this.virtualOrderPenaltyPipeline);
          orderPenaltyPass.setBindGroup(0, virtualOrderPenaltyBindGroup);
          this.dispatchLinear(orderPenaltyPass, Math.ceil(params.count / 64));
          orderPenaltyPass.end();
        }
        if (brushLocalColorFlowBindGroup) {
          const colorFlowPass = encoder.beginComputePass(
            this.profilePassDescriptor(profileSample, "brush-local-color-flow"),
          );
          colorFlowPass.setPipeline(this.brushLocalColorFlowPipeline);
          colorFlowPass.setBindGroup(0, brushLocalColorFlowBindGroup);
          this.dispatchLinear(colorFlowPass, Math.ceil(params.count / 64));
          colorFlowPass.end();
        }
        if (applyOptimizer) {
          const optimizerPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "optimizer"));
          optimizerPass.setPipeline(this.exactOptimizerPipeline);
          optimizerPass.setBindGroup(0, exactOptimizerBindGroup);
          this.dispatchLinear(optimizerPass, Math.ceil(params.count / 64));
          optimizerPass.end();
        }
      } else {
        const optimizerPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "optimizer"));
        optimizerPass.setPipeline(optimizerPipeline);
        optimizerPass.setBindGroup(0, optimizerBindGroup);
        if (phase37.parallelOptimizer) this.dispatchLinear(optimizerPass, params.count);
        else optimizerPass.dispatchWorkgroups(Math.ceil(params.count / 64));
        optimizerPass.end();
      }
      if (discreteLayerPassDue) {
        const assignPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "discrete-layer-assign"));
        assignPass.setPipeline(this.discreteLayerAssignPipeline);
        assignPass.setBindGroup(0, discreteLayerAssignBindGroup);
        this.dispatchLinear(assignPass, Math.ceil(params.count / 64));
        assignPass.end();
        const commitPass = encoder.beginComputePass(this.profilePassDescriptor(profileSample, "discrete-layer-commit"));
        commitPass.setPipeline(this.discreteLayerCommitPipeline);
        commitPass.setBindGroup(0, discreteLayerCommitBindGroup);
        this.dispatchLinear(commitPass, Math.ceil(params.count / 64));
        commitPass.end();
      }
      if (profileSample && useExactBackward && !sourceDomainActive) {
        encoder.copyBufferToBuffer(
          this.trainState.exactBackwardTelemetryBuffer,
          0,
          this.trainState.exactBackwardTelemetryReadbackBuffer,
          0,
          EXACT_BACKWARD_TELEMETRY_BYTES,
        );
      }
      if (profileSample && fixedPointExactGradientActive) {
        encoder.copyBufferToBuffer(
          this.trainState.fixedPointGradientControlBuffer,
          0,
          this.trainState.fixedPointGradientReadbackBuffer,
          0,
          FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES,
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
      if (!batchEncoder) {
        this.device.queue.submit([encoder.finish()]);
        if (profileSample) profileEncodeSubmitMs = performance.now() - profileWallStarted;
      }
      this.trainState.pixelStateResolution = [workImage.width, workImage.height];
      this.trainState.pixelStateKind = trainingStage;
      this.trainState.pixelStateViewKey = tiltStep.enabled || sourceDomainActive ? "virtual" : "front";
      if (recordTrainingStep && useCoarse) this.trainState.coarseTrainingSteps += 1;
      if (recordTrainingStep && useMid) this.trainState.midTrainingSteps += 1;
      if (recordTrainingStep && tiltStep.enabled) {
        this.trainState.virtualTiltSteps += 1;
        this.trainState.lastVirtualTilt = { ...tiltStep, step: currentStep };
      }
      if (effectiveSync) {
        const syncStarted = profileSample ? performance.now() : 0;
        await this.device.queue.onSubmittedWorkDone();
        if (profileSample) {
          profileSyncWaitMs = performance.now() - syncStarted;
          profileQueueWaitCount += 1;
        }
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
        profileReadbackCount += 1;
        profileReadbackBytes += bytes;
        if (useExactBackward && !sourceDomainActive) {
          const telemetryStarted = performance.now();
          const telemetryReadback = this.trainState.exactBackwardTelemetryReadbackBuffer;
          await telemetryReadback.mapAsync(GPUMapMode.READ, 0, EXACT_BACKWARD_TELEMETRY_BYTES);
          const values = new Uint32Array(telemetryReadback.getMappedRange(0, EXACT_BACKWARD_TELEMETRY_BYTES)).slice();
          telemetryReadback.unmap();
          profileReadbackCount += 1;
          profileReadbackBytes += EXACT_BACKWARD_TELEMETRY_BYTES;
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
            per_pixel_overdraw: profileDistributionSummary(values.slice(8, 16), values[7], values[5]),
            per_pixel_contributors: profileDistributionSummary(values.slice(16, 24), values[7], values[6]),
          };
          profileReadbackMs += performance.now() - telemetryStarted;
        }
        if (fixedPointExactGradientActive) {
          const fixedPointStarted = performance.now();
          const fixedPointReadback = this.trainState.fixedPointGradientReadbackBuffer;
          await fixedPointReadback.mapAsync(
            GPUMapMode.READ,
            0,
            FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES,
          );
          const fixedPointValues = new Uint32Array(
            fixedPointReadback.getMappedRange(0, FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES),
          ).slice();
          fixedPointReadback.unmap();
          profileReadbackCount += 1;
          profileReadbackBytes += FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES;
          profileFixedPointGradientStats = {
            active: true,
            scale: FIXED_POINT_EXACT_GRADIENT_SCALE,
            safety_clamps: fixedPointValues[0],
            maximum_abs_quantized_contribution: fixedPointValues[1],
          };
          profileReadbackMs += performance.now() - fixedPointStarted;
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
          fixed_point_exact_gradient: profileFixedPointGradientStats,
          encode_submit_wall_ms: profileEncodeSubmitMs,
          sync_wait_wall_ms: profileSyncWaitMs,
          profile_readback_wall_ms: profileReadbackMs,
          queue_wait_count: profileQueueWaitCount,
          queue_wait_wall_ms: profileSyncWaitMs,
          readback_count: profileReadbackCount,
          readback_bytes: profileReadbackBytes,
          readback_wall_ms: profileReadbackMs,
          total_wall_ms: performance.now() - profileWallStarted,
        });
      }
      if (recordTrainingStep) this.lastTrainStats = {
        backend: "webgpu-render-gradient-adam",
        count: params.count,
        mode: useExactBackward ? "experimental-standard-alpha-exact-backward" : "experimental-render-l1-dssim-adam",
        exact_alpha_backward: useExactBackward,
        quad_exact_backward: this.quadExactBackwardEnabled && !sourceDomainActive,
        subgroup_exact_backward: this.subgroupExactBackwardEnabled && !sourceDomainActive,
        source_domain_reprojection: sourceDomainActive,
        source_domain_jacobian_weighting: sourceDomainActive,
        sgld_noise_suppressed: Boolean(suppressSgldNoise),
        exact_tile_intersection: this.exactTileIntersectionEnabled,
        shape_aware_paint_culling: this.shapeAwarePaintCullingEnabled,
        projected_virtual_exact_culling: this.projectedVirtualExactCullingEnabled,
        segmented_exact_backward: {
          ...this.trainState.segmentedExactBackward,
          active: segmentedExactBackwardActive,
        },
        fixed_point_exact_gradient: {
          ...this.trainState.fixedPointExactGradient,
          active: fixedPointExactGradientActive,
          scale: FIXED_POINT_EXACT_GRADIENT_SCALE,
          profile: profileFixedPointGradientStats,
        },
        inverse_scale_optimization: {
          requested: this.inverseScaleOptimizationRequested,
          active: Boolean(this.inverseScaleOptimizationRequested),
          parameter: this.inverseScaleOptimizationRequested ? "inverse-scale" : "log-scale",
          renderer_unchanged: true,
        },
        opacity_aware_support: this.opacityAwareSupportMode,
        render_gradient_optimizer: true,
        dssim_weight: DEFAULT_DSSIM_WEIGHT,
        compact_tile_candidates: Boolean(els.tileCullingToggle.checked),
        tile_builds: this.trainState.tileBuilds,
        sgld_2d: true,
        experimental_variants: experimentalVariants(),
        training_resolution: [workImage.width, workImage.height],
        training_stage: trainingStage,
        phase_one_shape_lr_multiplier: trainingStage === "coarse" ? PHASE_ONE_SHAPE_LR_MULTIPLIER : 1,
        phase_one_max_planar_scale: trainingStage === "coarse" ? PHASE_ONE_MAX_PLANAR_SCALE : learningRates.maxPlanarScale,
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
        discrete_layers: Boolean(params.discreteLayersEnabled),
        discrete_layer_count: params.discreteLayersEnabled ? params.discreteLayerCount : 0,
        discrete_layer_move_radius: params.discreteLayersEnabled ? params.discreteLayerMoveRadius : 0,
        layer_aware_accumulation: Boolean(params.layerAwareAccumulationEnabled),
        discrete_overlap_assignment_due: discreteLayerPassDue,
        virtual_depth_update_interval: boundedDepthEnabled ? Math.max(1, Math.round(virtualDepthUpdateInterval)) : 0,
        virtual_depth_update_due: Boolean(boundedDepthEnabled && virtualDepthUpdateDue),
        virtual_tilt: {
          ...tiltStep,
          interval: virtualTiltVariants().interval,
          fov_degrees: tiltStep.fovDegrees || DEFAULT_SHARED_CAMERA_FOV_DEGREES,
          plane_constrained: Boolean(tiltStep.enabled && tiltStep.planeConstrained !== false),
          source_domain_reprojection: Boolean(tiltStep.enabled && sourceDomainReprojection),
        },
        coarse_transition_steps: 0,
        density_horizon: experimentalDensifySteps(state.metrics?.steps_requested || 1),
        learningRates,
        stage_profile_samples: this.trainState.stageProfile.length,
        tile_cooperative_renderer: renderChoice.cooperative,
        tile_cooperative_supported: renderChoice.supported,
        stage_min_scale_ratio: trainingStage === "full" ? 0 : variants.stageMinScaleRatio,
        stage_base_scale_floor_ratio: stageBaseScaleFloorRatio(trainingStage),
        stage_min_scale_normalized: config[80],
        bind_group_cache_enabled: this.trainState.bindGroupCacheEnabled,
        bind_group_cache_entries: this.trainState.bindGroupCache.size,
        bind_group_cache_hits: this.trainState.bindGroupCacheStats.hits,
        bind_group_cache_misses: this.trainState.bindGroupCacheStats.misses,
        sync: effectiveSync,
        batched: Boolean(batchEncoder),
        batch_config_slot: batchEncoder ? batchConfigSlot : null,
        updated: true,
      };
    } finally {
      if (errorScopeOpen) this.device.popErrorScope().catch(() => {});
    }
  }

  async trainStepGpu(image, params, learningRates, {
    sync = true,
    virtualCameraSampling = null,
    currentStepOverride = null,
    batchEncoder = null,
    batchConfigSlot = -1,
  } = {}) {
    if (
      !this.trainState ||
      this.trainState.width !== image.width ||
      this.trainState.height !== image.height ||
      this.trainState.capacity < params.count
    ) {
      await this.uploadTrainState(image, params);
    }
    this.trainState.count = params.count;
    const variants = phase33Variants();
    const currentStep = Number.isFinite(currentStepOverride)
      ? Math.max(1, Math.round(currentStepOverride))
      : (state.metrics?.steps_done || 0) + 1;
    const requestedSteps = state.metrics?.steps_requested || 1;
    const stage = curriculumTrainingStage(
      currentStep,
      requestedSteps,
      variants,
      this.trainState.coarseImage,
      this.trainState.midImage,
    );
    const samplingVariants = virtualCameraSampling || virtualCameraSamplingVariants(false);
    const virtualCameraWarmupSteps = 0;
    const sampledCamera = virtualCameraSamplingStepSpec(
      currentStep,
      stage,
      requestedSteps,
      virtualCameraWarmupSteps,
      samplingVariants,
    );
    const trainingCamera = virtualCameraTrainingStepSpec(
      sampledCamera,
      currentStep,
      stage,
      virtualCameraWarmupSteps,
      samplingVariants,
    );
    const tiltStep = virtualTiltStepSpec(currentStep, stage, requestedSteps);
    const effectiveExactBackward = qualityRecoveryVariants().exactBackward || !this.renderGradientPipeline;
    if (samplingVariants.enabled) {
      if (!effectiveExactBackward || !variants.ewa2x2) {
        throw new Error("Virtual camera sampling requires exact WebGPU backward and finite-pixel EWA.");
      }
      let result;
      const virtualStep = trainingCamera.kind === "virtual";
      const gradientBalance = virtualCameraGradientBalance(
        samplingVariants.virtualSlots,
        samplingVariants.slots,
      );
      const frontGradientAnchorWeight = virtualStep
        ? gradientBalance.frontGradientAnchorWeight
        : 0;
      const cameraCounts = virtualCameraSamplingCountsThroughStep(
        currentStep,
        virtualCameraWarmupSteps,
        samplingVariants,
      );
      const virtualDepthUpdateDue = virtualStep && cameraCounts.virtual > 0 &&
        cameraCounts.virtual % samplingVariants.depthUpdateInterval === 0;
      const cameraMetadata = state.metrics?.virtual_camera_sampling?.cameras?.find?.(
        (camera) => camera.id === trainingCamera.cameraId,
      );
      const virtualDepthConfidence = samplingVariants.softDepthConstraint
        ? virtualDepthCameraConfidence(trainingCamera, cameraMetadata?.teacher_coverage)
        : 1;
      if (frontGradientAnchorWeight > 0) {
        if (batchEncoder) {
          throw new Error("Balanced virtual-camera front anchors require sequential GPU steps.");
        }
        await this.trainStepRenderGradientGpu(image, params, learningRates, {
          sync: false,
          currentStepOverride: currentStep,
          viewOverride: "front",
          clearExactGradient: true,
          applyOptimizer: false,
          recordTrainingStep: false,
          gradientNormalization: 1,
          gradientChannels: {
            geometry: frontGradientAnchorWeight,
            appearance: frontGradientAnchorWeight,
            density: frontGradientAnchorWeight,
            depth: 0,
          },
          sourceDomainReprojection: false,
          cameraCovariance3d: false,
        });
      }
      result = await this.trainStepRenderGradientGpu(image, params, learningRates, {
        sync,
        currentStepOverride: currentStep,
        batchEncoder,
        batchConfigSlot,
        viewOverride: virtualStep ? trainingCamera : "front",
        clearExactGradient: frontGradientAnchorWeight <= 0,
        applyOptimizer: true,
        recordTrainingStep: true,
        gradientNormalization: 1 / (1 + frontGradientAnchorWeight),
        gradientChannels: { geometry: 1, appearance: 1, density: 1, depth: 1 },
        sourceDomainReprojection: false,
        cameraCovariance3d: virtualStep,
        virtualDepthUpdateDue,
        virtualDepthUpdateInterval: samplingVariants.depthUpdateInterval,
        virtualDepthCameraConfidence: virtualDepthConfidence,
        gofDensity: samplingVariants.gofDensity,
      });
      this.trainState.virtualCameraFrontSteps = cameraCounts.front;
      this.trainState.virtualCameraVirtualSteps = cameraCounts.virtual;
      const effectiveRadius = trainingCamera.kind === "virtual"
        ? this.lastTrainStats?.virtual_tilt?.cameraDistance
        : this.trainState.virtualCameraOrbitRadius;
      if (Number.isFinite(effectiveRadius)) this.trainState.virtualCameraOrbitRadius = effectiveRadius;
      this.trainState.lastVirtualCameraSample = { ...trainingCamera, step: currentStep };
      this.lastTrainStats = {
        ...this.lastTrainStats,
        virtual_camera_sample: this.trainState.lastVirtualCameraSample,
        virtual_camera_gradient_normalized: false,
        virtual_camera_invalid_region_mode: samplingVariants.invalidRegionMode,
        virtual_camera_front_steps: this.trainState.virtualCameraFrontSteps,
        virtual_camera_virtual_steps: this.trainState.virtualCameraVirtualSteps,
        virtual_camera_warmup_steps: virtualCameraWarmupSteps,
        virtual_camera_orbit_radius: this.trainState.virtualCameraOrbitRadius,
        virtual_camera_requested_objective_mode: "3dgs-multiview",
        virtual_camera_objective_mode: "3dgs-multiview",
        virtual_camera_gradient_routing: frontGradientAnchorWeight > 0
          ? "sampled-view-all+balanced-front-anchor"
          : "selected-view-all",
        virtual_camera_front_anchor_weight: frontGradientAnchorWeight,
        virtual_camera_effective_gradient_share_percent:
          gradientBalance.effectiveVirtualShare * 100,
        virtual_camera_front_anchor_passes: frontGradientAnchorWeight > 0
          ? cameraCounts.virtual
          : 0,
        virtual_camera_3dgs_multiview: true,
        virtual_camera_channel_mask: { position: 1, color: 1, opacity: 1, scale: 1, rotation: 1, depth: 1, density: 1 },
        virtual_camera_projection_mode: "camera-projected-3d-covariance",
        virtual_camera_gof_density: Boolean(samplingVariants.gofDensity),
        virtual_camera_depth_confidence: virtualDepthConfidence,
        virtual_camera_front_gradient_steps: cameraCounts.front,
        virtual_camera_virtual_gradient_steps: cameraCounts.virtual,
      };
      return result;
    }
    if (tiltStep.enabled && effectiveExactBackward && variants.ewa2x2) {
      await this.trainStepRenderGradientGpu(image, params, learningRates, {
        sync: false,
        currentStepOverride: currentStep,
        viewOverride: "front",
        clearExactGradient: true,
        applyOptimizer: false,
        recordTrainingStep: false,
      });
      return this.trainStepRenderGradientGpu(image, params, learningRates, {
        sync,
        currentStepOverride: currentStep,
        viewOverride: tiltStep,
        clearExactGradient: false,
        applyOptimizer: true,
        recordTrainingStep: true,
        gradientNormalization: 1 / (1 + Math.max(0, tiltStep.weight)),
      });
    }
    return this.trainStepRenderGradientGpu(image, params, learningRates, {
      sync,
      currentStepOverride: currentStep,
      batchEncoder,
      batchConfigSlot,
    });
  }

  async trainStepsGpu(image, params, learningRates, steps, { virtualCameraSampling = null } = {}) {
    const batchSteps = steps.slice(0, MAX_TRAIN_BATCH_SIZE);
    if (batchSteps.length === 0) return null;
    const encoder = this.device.createCommandEncoder();
    const virtualCameraSamples = [];
    this.device.pushErrorScope("validation");
    let errorScopeOpen = true;
    try {
      for (let slot = 0; slot < batchSteps.length; slot += 1) {
        await this.trainStepGpu(image, params, learningRates, {
          sync: false,
          virtualCameraSampling,
          currentStepOverride: batchSteps[slot],
          batchEncoder: encoder,
          batchConfigSlot: slot,
        });
        if (this.lastTrainStats?.virtual_camera_sample) {
          virtualCameraSamples.push({
            ...this.lastTrainStats.virtual_camera_sample,
            step: batchSteps[slot],
          });
        }
      }
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const error = await this.device.popErrorScope();
      errorScopeOpen = false;
      if (error) throw new Error(error.message);
      if (this.lastTrainStats) {
        this.lastTrainStats = {
          ...this.lastTrainStats,
          sync: true,
          batched: true,
          batch_size: batchSteps.length,
          batch_first_step: batchSteps[0],
          batch_last_step: batchSteps[batchSteps.length - 1],
          virtual_camera_samples: virtualCameraSamples,
        };
      }
      return this.lastTrainStats;
    } finally {
      if (errorScopeOpen) this.device.popErrorScope().catch(() => {});
    }
  }

}

registerWebGpuPreviewFeature(WebGpuOptimizerRuntime.prototype, "WebGpuOptimizerRuntime");
