let qaFlowRibbonPreviewCache = null;

function presentQaFlowRibbonResult(result) {
  state.flowSplatResult = { sourceImage: state.image, ...result };
  presentFlowSplatFusionResult(state.flowSplatResult);
}

if (QA_RUNTIME_ENABLED) window.__flatPhotoTest = {
  flowStrokeWorkloadProfile(options = {}) {
    if (!state.image) throw new Error("Load an image before profiling flow-stroke training.");
    const representation = options.representation === "fused-ribbon"
      ? "fused-ribbon"
      : "curve-splat-chain";
    const maxSide = Math.max(32, Math.min(512, Math.round(Number(options.maxSide) || 512)));
    const requestedSplats = Math.max(256, Math.min(14000, Math.round(Number(options.maxSplats) || 8192)));
    const splatsPerChain = Image2SplatPaintFlowRibbonTrainer.constants.CURVE_SAMPLES;
    const maxStrokes = representation === "curve-splat-chain"
      ? Math.max(1, Math.floor(requestedSplats / splatsPerChain))
      : requestedSplats;
    const [width, height] = resizedSize(state.image.width, state.image.height, maxSide);
    const trainingImage = width === state.image.width && height === state.image.height
      ? state.image
      : { ...state.image, ...resizeFloatImageBilinear(state.image, width, height) };
    const reference = Image2SplatPaintFlowPaintReference.createFlowPaintReference(trainingImage, {
      seed: 240825,
      strength: 1,
      profile: "connected-ribbon-v1",
      maxStrokes,
      minimumStrokes: representation === "curve-splat-chain" ? 1 : 256,
      includeStrokePlan: true,
    });
    const prepared = Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
      trainingImage,
      reference.strokePlan,
      { maxStrokes: reference.strokePlan.length, representation },
    );
    const curveSampleEvaluationsPerCandidate = representation === "curve-splat-chain" ? 3 : 84;
    const tileSampleStride = representation === "curve-splat-chain" ? 8 : 1;
    return {
      representation,
      width,
      height,
      pixels: width * height,
      requested_splats: requestedSplats,
      curve_count: prepared.strokeCount,
      physical_splat_count: representation === "curve-splat-chain"
        ? prepared.strokeCount * splatsPerChain
        : null,
      tile_size: Image2SplatPaintFlowRibbonTrainer.constants.TILE_SIZE,
      tile_candidate_entries: prepared.tileCandidateEntries,
      tile_candidates_mean: prepared.meanTileCandidates,
      tile_candidates_max: prepared.maxTileCandidates,
      pixel_candidate_pairs_per_iteration: prepared.pixelCandidatePairs,
      tile_sample_stride: tileSampleStride,
      sampled_pixel_candidate_pairs_per_iteration:
        Math.ceil(prepared.pixelCandidatePairs / tileSampleStride),
      curve_sample_evaluations_per_candidate: curveSampleEvaluationsPerCandidate,
      curve_sample_evaluations_per_iteration:
        Math.ceil(prepared.pixelCandidatePairs * curveSampleEvaluationsPerCandidate / tileSampleStride),
      preview_readback_bytes: width * height * 16,
    };
  },
  async trainFlowRibbon(options = {}) {
    if (!state.image) throw new Error("Load an image before training the Flow Paint ribbon candidate.");
    const sourceImage = state.image;
    const requestedSide = Math.max(64, Math.min(256, Math.round(Number(options.maxSide) || 256)));
    const [width, height] = resizedSize(state.image.width, state.image.height, requestedSide);
    const trainingImage = width === state.image.width && height === state.image.height
      ? state.image
      : { ...state.image, ...resizeFloatImageBilinear(state.image, width, height) };
    const reference = Image2SplatPaintFlowPaintReference.createFlowPaintReference(trainingImage, {
      seed: 240825,
      strength: 1,
      profile: "connected-ribbon-v1",
      maxStrokes: Math.max(256, Math.min(14000, Math.round(Number(options.maxStrokes) || 14000))),
      includeStrokePlan: true,
    });
    const result = await Image2SplatPaintFlowRibbonTrainer.train(
      trainingImage,
      reference.strokePlan,
      {
        iterations: Math.max(1, Math.min(300, Math.round(Number(options.iterations) || 300))),
        maxStrokes: reference.strokePlan.length,
        colorAnchor: Number(options.colorAnchor ?? 0.0035),
        geometryAnchor: Number(options.geometryAnchor ?? 0.00035),
        widthAnchor: Number(options.widthAnchor ?? 0.0008),
        opacityAnchor: Number(options.opacityAnchor ?? 0.0008),
        shouldStop: options.shouldStop,
        onProgress(progress) {
          document.documentElement.dataset.qaFlowTrainIteration = String(progress.iteration);
          document.documentElement.dataset.qaFlowTrainIterations = String(progress.iterations);
          options.onProgress?.(progress);
        },
      },
    );
    if (state.image !== sourceImage) throw new Error("The loaded image changed during Flow Paint training.");
    qaFlowRibbonPreviewCache = { sourceImage, result };
    presentQaFlowRibbonResult(result);
    window.__lastFlowRibbonTraining = structuredClone(result.metadata);
    return structuredClone(result.metadata);
  },
  hasFlowRibbonResult() {
    return Boolean(qaFlowRibbonPreviewCache?.sourceImage === state.image);
  },
  showFlowRibbonResult() {
    if (qaFlowRibbonPreviewCache?.sourceImage !== state.image) return null;
    presentQaFlowRibbonResult(qaFlowRibbonPreviewCache.result);
    return structuredClone(qaFlowRibbonPreviewCache.result.metadata);
  },
  getLastFlowRibbonTraining() {
    return structuredClone(window.__lastFlowRibbonTraining || null);
  },
  showFlowPaintReference(options = {}) {
    if (!state.image) throw new Error("Load an image before previewing the flow-painted reference.");
    const result = Image2SplatPaintFlowPaintReference.createFlowPaintReference(state.image, options);
    fitCanvases(result.image.width, result.image.height);
    previewCtx.putImageData(
      rgbToImageData(result.image.rgb, result.image.width, result.image.height, result.image.alpha),
      0,
      0,
    );
    showCanvas("preview");
    return structuredClone(result.metadata);
  },
  loadGeneratedSample,
  structureGuidedProfileBenchmark,
  initialSplatOrientation,
  initialSplatShape,
  initialOrientationStats,
  rectangleConstraintProbe,
  opaquePaintLateSettleFraction,
  opaquePaintLateSettleStartStep,
  opaquePaintStructuralMutationAllowed,
  sharedTiltOrbitRadius,
  optimizerFootprintHistogram,
  brushColorNeutrality,
  colorFinishStartStep,
  phase39ContractProbe,
  layerOrderComparatorProbe,
  layerEfficiencyVariants,
  summarizeLayerEfficiency,
  planarTiltRotation,
  projectPlanarPoint,
  inverseProjectPlanarPoint,
  virtualTiltVariants,
  virtualTiltStepSpec,
  virtualCameraSamplingVariants,
  virtualCameraBag,
  virtualCameraSamplingStepSpec,
  virtualCameraSamplingCountsThroughStep,
  virtualCameraCatalog,
  tiltRiskProfileForSplat,
  summarizeTiltRisk,
  async tiltDefaultRendererProbe(includeFrames = false) {
    const controller = await loadTiltViewer();
    if (!controller) throw new Error("Tilt viewer did not initialize.");
    controller.setCameraMarkersVisible(false);
    await new Promise((resolve) => setTimeout(resolve, 100));
    const before = controller.diagnostics();
    const capturePng = async () => {
      if (!includeFrames) return null;
      const blob = await controller.captureFrameBlob("image/png", 1);
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
        reader.onerror = () => reject(reader.error || new Error("Tilt PNG read failed."));
        reader.readAsDataURL(blob);
      });
    };
    const frontFrame = await capturePng();
    const drive = async (yaw) => {
      els.tiltPitch.value = "0";
      els.tiltYaw.value = String(yaw);
      const applied = await controller.setTiltAndWait(0, yaw);
      els.tiltPitch.value = String(applied.pitch);
      els.tiltYaw.value = String(applied.yaw);
      return {
        diagnostics: controller.diagnostics(),
        frame: await capturePng(),
      };
    };
    const positive = await drive(2);
    const negative = await drive(-2);
    return {
      contract: "playcanvas-default-sort-settled-views",
      before: before.presentation,
      positive: {
        yaw: positive.diagnostics.yaw,
        presentation: positive.diagnostics.presentation,
      },
      negative: {
        yaw: negative.diagnostics.yaw,
        presentation: negative.diagnostics.presentation,
      },
      final_camera: negative.diagnostics.camera,
      frames: includeFrames
        ? { front: frontFrame, positive: positive.frame, negative: negative.frame }
        : null,
    };
  },
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
    const requestedDiscreteLayers = Math.max(
      MIN_DISCRETE_LAYER_COUNT,
      Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(state.params?.discreteLayerCount || DEFAULT_DISCRETE_LAYER_COUNT)),
    );
    const discreteDepthValues = state.params?.depthOrder
      ? [...new Set(Array.from(state.params.depthOrder, (value) => Math.min(
          requestedDiscreteLayers - 1,
          Math.floor(Math.max(0, Math.min(1 - Number.EPSILON, value)) * requestedDiscreteLayers),
        )))].sort((a, b) => a - b)
      : [];
    return {
      image: state.image ? {
        width: state.image.width,
        height: state.image.height,
        originalWidth: state.image.originalWidth,
        originalHeight: state.image.originalHeight,
        cacheWidth: state.image.cacheWidth,
        cacheHeight: state.image.cacheHeight,
        inputCacheMaxSide: state.image.inputCacheMaxSide,
        inputCacheResized: state.image.inputCacheResized,
        inputDecodeMode: state.image.inputDecodeMode,
        inputOrientation: state.image.inputOrientation,
        fileName: state.image.fileName,
      } : null,
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
      opacityAwareSupport: performanceVariants().opacityAwareSupportMode,
      adaptiveGpuThroughput: performanceVariants().adaptiveGpuThroughput,
      adaptiveGpuBatch: performanceVariants().adaptiveGpuBatch,
      gpuSchedulingMode: performanceVariants().gpuSchedulingMode,
      asyncPresentation: performanceVariants().asyncPresentation,
      metricTileReuse: performanceVariants().metricTileReuse,
      segmentedExactBackward: performanceVariants().segmentedExactBackward,
      fixedPointExactGradient: performanceVariants().fixedPointExactGradient,
      inverseScaleOptimization: performanceVariants().inverseScaleOptimization,
      stageGrowthShares: phase39Variants().stageGrowthShares,
      discreteLayers: {
        enabled: Boolean(state.params?.discreteLayersEnabled),
        layerAwareAccumulation: Boolean(state.params?.layerAwareAccumulationEnabled),
        requested: Number(state.params?.discreteLayerCount || 0),
        moveRadius: Number(state.params?.discreteLayerMoveRadius || 0),
        occupied: discreteDepthValues.length,
        values: discreteDepthValues,
      },
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
  forceDeviceLoss() {
    if (!state.webgpu.renderer?.device) return false;
    state.webgpu.renderer.device.destroy();
    return true;
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
  inputCacheProbe(width, height) {
    const [cacheWidth, cacheHeight] = resizedSize(
      width,
      height,
      INPUT_CACHE_MAX_SIDE,
      INPUT_CACHE_MAX_SIDE,
    );
    return {
      sourceWidth: width,
      sourceHeight: height,
      cacheWidth,
      cacheHeight,
      maxSide: INPUT_CACHE_MAX_SIDE,
      resized: cacheWidth !== width || cacheHeight !== height,
    };
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
  async showTrainingViews() {
    await showTiltTrainingViews();
    return structuredClone(state.tilt.teacherViews);
  },
  async obliqueOverlapDiagnostics() {
    if (!state.image || !state.params || !state.webgpu.renderer?.trainState) {
      throw new Error("Finish a WebGPU training run before measuring oblique overlap.");
    }
    const report = await state.webgpu.renderer.computeObliqueDiagnostics(state.image, state.params);
    if (state.metrics) state.metrics.oblique_overlap_diagnostics = report;
    return structuredClone(report);
  },
  trainingViews() {
    return structuredClone(state.tilt.teacherViews);
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
  benchmarkSummary() {
    const m = state.metrics;
    if (!m) return null;
    const performance = m.performance_trace?.at(-1) || null;
    const overlap = m.overlap_diagnostics?.scales?.["1"] || null;
    const hiddenRgbAttribution = m.overlap_diagnostics?.hidden_rgb_attribution || null;
    return {
      contract: {
        algorithm: m.algorithm,
        seed: "deterministic-webgpu",
        image_size: m.image_size,
        steps_requested: m.steps_requested,
        steps_done: m.steps_done,
        initialization: m.initialization,
        initialization_adaptive: m.initialization_adaptive || null,
      },
      quality: {
        psnr_rgb_db: m.final_psnr,
        global_ssim: m.final_global_ssim,
        local_ssim_p10: m.final_regional_ssim?.p10 ?? null,
        l1: m.final_l1,
      },
      alpha: {
        mean_composited: overlap?.mean_composited_alpha ?? m.coverage_stats?.mean ?? null,
        l1: m.final_alpha_l1,
        ssim: m.final_alpha_ssim,
      },
      coverage: {
        background_exposure_ratio: m.coverage_stats?.background_exposure_ratio ?? null,
        background_exposure_count: m.coverage_stats?.background_exposure_count ?? null,
      },
      shape: {
        scale_histogram: m.scale_histogram || m.shape_stats?.scale_histogram || null,
        anisotropy_ratio: m.anisotropy_ratio ?? m.shape_stats?.anisotropy_ratio_mean ?? null,
        axis_px: m.shape_stats?.inspection?.radius_px || null,
      },
      overlap: {
        effective_contributors: overlap?.mean_effective_contributors ?? null,
        color_variance: overlap?.mean_contributor_color_variance ?? null,
      },
      hidden_rgb_noise_check: {
        method: hiddenRgbAttribution ? "WebGPU positive leave-one-out attribution at 0.25x" : "Splats tab: inspect at 0.25x scale on a non-image alpha background",
        automated_metric: hiddenRgbAttribution?.total_positive_harm ?? null,
        top_32_positive_harm: hiddenRgbAttribution?.top_32_positive_harm ?? null,
        attribution: hiddenRgbAttribution,
      },
      execution: {
        elapsed_ms: performance?.elapsed_ms ?? null,
        iterations_per_second: performance?.iterations_per_second ?? null,
        gpu_memory: m.gpu_training_memory || null,
        scheduling_profile: m.scheduling_profile || null,
      },
    };
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
      algorithm: m.algorithm,
      algorithm_label: m.algorithm_label,
      initial_orientation: m.initial_orientation,
      final_evaluation_step: m.final_evaluation_step,
      final_metrics_complete: m.final_metrics_complete,
      initial_l1: m.initial_l1,
      final_l1: m.final_l1,
      initial_rgb_mse: m.initial_rgb_mse,
      final_rgb_mse: m.final_rgb_mse,
      initial_psnr: m.initial_psnr,
      final_psnr: m.final_psnr,
      psnr_trend: m.psnr_trend,
      quality_metric_contract: m.quality_metric_contract,
      initial_alpha_l1: m.initial_alpha_l1,
      final_alpha_l1: m.final_alpha_l1,
      initial_alpha_ssim: m.initial_alpha_ssim,
      final_alpha_ssim: m.final_alpha_ssim,
      initial_alpha_objective: m.initial_alpha_objective,
      final_alpha_objective: m.final_alpha_objective,
      initial_objective_loss: m.initial_objective_loss,
      final_objective_loss: m.final_objective_loss,
      color_objective: m.color_objective,
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
      finalization_wall_ms: m.finalization_wall_ms,
      final_diagnostic_sample_limit: m.final_diagnostic_sample_limit,
      final_parameter_hash: m.final_parameter_hash ?? null,
      training_evaluation: m.training_evaluation ? structuredClone(m.training_evaluation) : null,
      training_residual_map: m.training_residual_map ? structuredClone(m.training_residual_map) : null,
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
      densify_events: m.densify_events,
      growth_schedule: m.growth_schedule,
      source_detail_splat_distribution: m.source_detail_splat_distribution || null,
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
      phase_relative_scale_guard: m.phase_relative_scale_guard
        ? structuredClone(m.phase_relative_scale_guard)
        : null,
      phase45_region_report: m.phase45_region_report,
      mid_training_overdensity_correction: m.mid_training_overdensity_correction
        ? structuredClone(m.mid_training_overdensity_correction)
        : null,
      ...(m.brush_contribution_diagnostics
        ? { brush_contribution_diagnostics: structuredClone(m.brush_contribution_diagnostics) }
        : {}),
      overlap_diagnostics: m.overlap_diagnostics,
      oblique_overlap_diagnostics: m.oblique_overlap_diagnostics || null,
      render_surface_parity: m.render_surface_parity || null,
      color_space_audit: m.color_space_audit ? structuredClone(m.color_space_audit) : null,
      performance_trace: m.performance_trace,
      performance_profile_schedule: m.performance_profile_schedule || {},
      stage_profile: m.stage_profile || [],
      stage_profile_backend: m.stage_profile_backend || "off",
      scheduling_profile: m.scheduling_profile || null,
      gpu_scheduling: m.gpu_scheduling ? structuredClone(m.gpu_scheduling) : null,
      metric_tile_reuse: m.metric_tile_reuse ? structuredClone(m.metric_tile_reuse) : null,
      importance_stats: m.importance_stats,
      residual_destination_oracle: m.residual_destination_oracle
        ? structuredClone(m.residual_destination_oracle)
        : null,
      coverage_stats: m.coverage_stats,
      density_gpu_ms: m.density_gpu_ms,
      relocation_gpu_ms: m.relocation_gpu_ms,
      post_density_annealing: Boolean(m.post_density_annealing),
      tile_culling_enabled: Boolean(m.tile_culling_enabled),
      train_layer_order: Boolean(m.train_layer_order),
      virtual_camera_sampling: m.virtual_camera_sampling
        ? structuredClone(m.virtual_camera_sampling)
        : null,
      virtual_camera_evaluation: m.virtual_camera_evaluation
        ? structuredClone(m.virtual_camera_evaluation)
        : null,
      virtual_depth_stats: m.virtual_depth_stats ? structuredClone(m.virtual_depth_stats) : null,
      gpu_training_memory: m.gpu_training_memory ? structuredClone(m.gpu_training_memory) : null,
      layer_update_interval: m.layer_update_interval,
      layer_update_rate: m.layer_update_rate,
      layer_stage_aware_rate: Boolean(m.layer_stage_aware_rate),
      layer_freeze_fraction: m.layer_freeze_fraction,
      layer_update_count: m.layer_update_count,
      layer_update_first_steps: m.layer_update_first_steps || [],
      layer_update_last_step: m.layer_update_last_step,
      layer_telemetry_enabled: Boolean(m.layer_telemetry_enabled),
      layer_telemetry: m.layer_telemetry || [],
      layer_efficiency: m.layer_efficiency ? structuredClone(m.layer_efficiency) : null,
      surface_layer_prior: m.surface_layer_prior
        ? structuredClone(m.surface_layer_prior)
        : null,
      front_split_children: m.front_split_children
        ? structuredClone(m.front_split_children)
        : null,
      front_footprint_refinement_v2: m.front_footprint_refinement_v2
        ? structuredClone(m.front_footprint_refinement_v2)
        : null,
      brush_detail_v1: m.brush_detail_v1
        ? structuredClone(m.brush_detail_v1)
        : null,
      brush_detail_layer_policy: m.brush_detail_layer_policy
        ? structuredClone(m.brush_detail_layer_policy)
        : null,
      train_layer_color_guard: m.train_layer_color_guard
        ? structuredClone(m.train_layer_color_guard)
        : null,
      brush_local_color_flow: m.brush_local_color_flow
        ? structuredClone(m.brush_local_color_flow)
        : null,
      brush_stroke_persistence: m.brush_stroke_persistence
        ? structuredClone(m.brush_stroke_persistence)
        : null,
      result_render_cache: m.result_render_cache
        ? structuredClone(m.result_render_cache)
        : null,
      current_visibility_compaction: m.current_visibility_compaction
        ? structuredClone(m.current_visibility_compaction)
        : null,
      current_visibility_compaction_events: m.current_visibility_compaction_events
        ? structuredClone(m.current_visibility_compaction_events)
        : [],
      current_visibility_compaction_removed_total: m.current_visibility_compaction_removed_total || 0,
      current_contribution_compaction: m.current_contribution_compaction
        ? structuredClone(m.current_contribution_compaction)
        : null,
      current_contribution_compaction_events: m.current_contribution_compaction_events
        ? structuredClone(m.current_contribution_compaction_events)
        : [],
      current_contribution_compaction_removed_total:
        m.current_contribution_compaction_removed_total || 0,
      current_contribution_compaction_deferred_count:
        m.current_contribution_compaction_deferred_count || 0,
      current_contribution_compaction_deferred_steps:
        m.current_contribution_compaction_deferred_steps
          ? structuredClone(m.current_contribution_compaction_deferred_steps)
          : [],
      current_contribution_compaction_settings: m.current_contribution_compaction_settings
        ? structuredClone(m.current_contribution_compaction_settings)
        : null,
      opaque_paint_late_settle: m.opaque_paint_late_settle
        ? structuredClone(m.opaque_paint_late_settle)
        : null,
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
    const originalExportReady = state.exportReady;
    const formatKey = algorithmSupportsExport("ply") ? "ply" : "png";
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
        state.exportReady = true;
        try {
          await saveExport({ download: false, formatKey });
          result[name] = { blocked: false, message: "export unexpectedly succeeded" };
        } catch (error) {
          result[name] = {
            blocked: /Export is not ready/.test(error.message),
            message: error.message,
          };
        }
      }
    } finally {
      state.metrics = originalMetrics;
      state.exportMessage = originalMessage;
      state.exportReady = originalExportReady;
      updateExportPanel();
      publishState();
    }
    return result;
  },
};
