function beginTrainingRun() {
  const run = new Image2SplatPaintTrainingSession({
    generation: ++state.trainingGeneration,
    renderer: state.webgpu.renderer,
    image: state.image,
  });
  state.trainingRun = run;
  return run;
}

function updateTrainingRunOwnership(run, { image, params, metrics } = {}) {
  if (!run || state.trainingRun !== run) return;
  run.updateOwnership({ image, params, metrics });
}

function ownsTrainingRun(run) {
  return Boolean(run?.owns(state));
}

function assertTrainingRun(run) {
  if (run) run.assertCurrent(state);
}

async function awaitTrainingRun(run, promise) {
  return run ? run.awaitCurrent(state, promise) : promise;
}

function invalidateTrainingRun(reason = "lifecycle change") {
  const run = state.trainingRun;
  if (!run) return false;
  run.cancel();
  state.trainingRun = null;
  state.trainingGeneration += 1;
  state.startPending = false;
  log(`training lifecycle invalidated: ${reason}`);
  return true;
}

function trainingLifecycleInputLocked() {
  return state.running || state.startPending || state.imageLoading || state.webGpuRecoveryPending;
}

function previewModeInputLocked() {
  // Original/Splats is display-only and is safe while the optimizer owns the
  // training buffers. startPending intentionally spans the whole async run,
  // so only its pre-running upload window lacks a stable presentation source.
  return state.webGpuRecoveryPending || (state.startPending && !state.running);
}

function isExpectedPreviewCancellation(error) {
  if (error?.trainingRunCancelled || state.webGpuRecoveryPending || state.webgpu.renderer?.deviceLost) return true;
  const message = String(error?.message || error || "").toLowerCase();
  return message.includes("device lost") ||
    message.includes("instance dropped") ||
    message.includes("webgpu lifecycle changed");
}

function log(message) {
  els.log.textContent = `${new Date().toLocaleTimeString()}  ${message}\n${els.log.textContent}`.slice(0, 4000);
}

function eventLog(message) {
  log(`[event] ${message}`);
}

function setStatus(status) {
  els.statusText.textContent = status;
  publishState();
}

function formatTrainingElapsed(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor((Number(milliseconds) || 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function resetTrainingTiming(blank = true) {
  state.trainingTiming = {
    elapsedMs: 0,
    iterationsPerSecond: 0,
    sampleElapsedMs: 0,
    sampleSteps: 0,
    lastStep: 0,
  };
  els.trainingTimingText.textContent = blank ? "- / -" : `- / ${formatTrainingElapsed(0)}`;
}

function recordTrainingTiming(step, elapsedMs) {
  const timing = state.trainingTiming;
  const stepDelta = Math.max(0, step - timing.lastStep);
  const activeMs = Math.max(0, Number(elapsedMs) || 0);
  timing.elapsedMs += activeMs;
  timing.sampleElapsedMs += activeMs;
  timing.sampleSteps += stepDelta;
  timing.lastStep = step;
  if (timing.sampleElapsedMs >= 250 || step === state.metrics?.steps_requested || state.stopRequested) {
    const currentRate = timing.sampleElapsedMs > 0
      ? (timing.sampleSteps * 1000) / timing.sampleElapsedMs
      : 0;
    timing.iterationsPerSecond = timing.iterationsPerSecond > 0
      ? timing.iterationsPerSecond * 0.65 + currentRate * 0.35
      : currentRate;
    timing.sampleElapsedMs = 0;
    timing.sampleSteps = 0;
  }
}

function syncTrainingTimingDisplay() {
  const timing = state.trainingTiming;
  if (!state.running && timing.lastStep === 0) return;
  const rate = timing.iterationsPerSecond > 0 ? `${timing.iterationsPerSecond.toFixed(1)} it/s` : "-";
  els.trainingTimingText.textContent = `${rate} / ${formatTrainingElapsed(timing.elapsedMs)}`;
}

function setTrainingMessage(message, kind = "info") {
  const text = String(message || "").toLowerCase();
  let status = "";
  if (kind === "error") status = text.includes("safety") || text.includes("blocked") ? "safety stopped" : "error";
  else if (text.includes("evaluating")) status = "evaluating";
  else if (text.includes("waiting")) status = "waiting";
  else if (text.includes("preparing")) status = "preparing";
  else if (text.includes("loading")) status = "loading";
  else if (text.includes("training on")) status = "running";
  else if (text.includes("complete")) status = "done";
  else if (text.includes("stopped")) status = "stopped";
  else if (text.includes("loaded") || text.includes("prepared") || text.includes("reset")) status = "image loaded";
  else if (text === "ready.") status = "idle";
  if (status && els.statusText.textContent !== status) setStatus(status);
}

function webGpuUnavailableMessage(reason = "") {
  const detail = String(reason || "").toLowerCase();
  if (detail.includes("navigator.gpu")) {
    return "WebGPU is unavailable in this browser. Training requires a current desktop browser with WebGPU enabled.";
  }
  if (detail.includes("adapter unavailable")) {
    return "No compatible WebGPU adapter was found. Check browser WebGPU support and the active GPU, then retry.";
  }
  return "WebGPU could not be initialized. Training is disabled, but image preview remains available.";
}

function syncDisplayedSsimMetrics() {
  const psnr = state.metrics?.latest_psnr ?? state.metrics?.final_psnr;
  const global = state.metrics?.latest_global_ssim ?? state.metrics?.final_global_ssim;
  const localP10 = state.metrics?.latest_regional_ssim?.p10 ?? state.metrics?.final_regional_ssim?.p10;
  if (Number.isFinite(psnr)) els.psnrText.textContent = `${psnr.toFixed(2)} dB`;
  if (Number.isFinite(global)) els.ssimText.textContent = global.toFixed(4);
  if (Number.isFinite(localP10)) els.regionalSsimText.textContent = localP10.toFixed(4);
}

function resetEvaluationStatusForNewTraining() {
  state.splatInspectorNonfiniteCache = null;
  state.metrics = null;
  state.lastGpuLoss = null;
  els.lossText.textContent = "-";
  els.psnrText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.coverageText.textContent = "- / -";
}

function createIdempotentDatasetProxy(dataset) {
  return new Proxy(dataset, {
    set(target, property, value) {
      // DOMStringMap coerces assigned values to strings. Compare its live
      // value rather than retaining a shadow cache, so external mutations are
      // still observed on the next synchronous publish.
      if (target[property] === String(value)) return true;
      target[property] = value;
      return true;
    },
  });
}

const publishedStateDataset = createIdempotentDatasetProxy(document.documentElement.dataset);

function publishState() {
  updateGpuMemoryStatus();
  updateCapacityStatus();
  syncTrainingTimingDisplay();
  const data = publishedStateDataset;
  data.status = els.statusText.textContent;
  data.productName = PRODUCT_NAME;
  data.algorithm = selectedAlgorithm().id;
  data.algorithmLabel = selectedAlgorithm().label;
  data.resultAlgorithm = trainedResultAlgorithm()?.id || "";
  data.backend = els.backendText.textContent;
  data.running = String(state.running);
  data.startPending = String(state.startPending);
  data.trainingGeneration = String(state.trainingGeneration);
  data.sampleLoading = String(state.sampleLoading);
  data.imageLoading = String(state.imageLoading);
  data.paused = String(state.paused);
  data.runtimeSettingsRevision = String(state.runtimeSettingsRevision);
  data.stopRequested = String(state.stopRequested);
  data.downloadsEnabled = String(state.exportReady && !state.exporting);
  data.exporting = String(state.exporting);
  data.aspectAwareGrid = String(aspectAwareGridEnabled());
  data.imageLoaded = String(Boolean(state.image));
  data.previewMode = state.previewMode;
  data.outsidePreviewEnabled = String(Boolean(els.outsidePreviewToggle.checked));
  data.outsidePreviewActive = String(Boolean(
    els.outsidePreviewToggle.checked &&
    !state.running &&
    (state.previewPadding.x > 0 || state.previewPadding.y > 0),
  ));
  data.outsidePreviewPending = String(state.previewRefreshPending);
  data.previewRequestedRevision = String(state.previewRequestedRevision);
  data.previewAppliedRevision = String(state.previewAppliedRevision);
  data.previewAppliedAlphaBackground = state.previewAppliedAlphaBackground;
  data.webGpuRecoveryPending = String(state.webGpuRecoveryPending);
  data.webGpuRecoveryAttempts = String(state.webGpuRecoveryAttempts);
  data.postTrainingGpuRecoveries = String(state.metrics?.post_training_gpu_recoveries || 0);
  data.splatParameterEffectsEnabled = String(Boolean(els.splatParameterEffects?.checked));
  data.splatPreviewShape = els.splatParameterEffects?.checked ? state.splatPreviewShape : trainedSplatShape();
  data.previewPaddingX = String(state.previewPadding.x);
  data.previewPaddingY = String(state.previewPadding.y);
  data.previewCanvasWidth = String(state.previewPadding.width || els.gpuCanvas.width);
  data.previewCanvasHeight = String(state.previewPadding.height || els.gpuCanvas.height);
  data.previewOnlyBytes = String(state.previewPadding.bytes);
  data.previewTileMode = state.webgpu.renderer?.lastPreviewStats?.tile_mode || "";
  data.previewTileReferences = String(state.webgpu.renderer?.lastPreviewStats?.tile_references || 0);
  data.canvasViewMode = state.canvasView.mode;
  data.canvasViewScale = String(state.canvasView.scale);
  updateCanvasViewControls();
  updateTiltControlState();
  const lifecycleLocked = trainingLifecycleInputLocked();
  els.appRoot.inert = state.imageLoading;
  els.appRoot.setAttribute("aria-busy", String(state.imageLoading));
  const resultTabsLocked = lifecycleLocked;
  const splatsTabLocked = previewModeInputLocked() || !state.params;
  els.splatsTab.disabled = splatsTabLocked;
  els.exportTab.disabled = resultTabsLocked;
  els.splatsTab.setAttribute("aria-disabled", String(splatsTabLocked));
  els.exportTab.setAttribute("aria-disabled", String(resultTabsLocked));
  data.resultTabsLocked = String(resultTabsLocked);
  data.splatsTabLocked = String(splatsTabLocked);
  els.startButton.disabled = lifecycleLocked || !state.image || !state.webgpu.supported;
  els.sampleButton.disabled = lifecycleLocked || state.sampleLoading;
  els.loadImageButton.disabled = lifecycleLocked;
  els.pathButton.disabled = lifecycleLocked;
  els.fileInput.disabled = lifecycleLocked;
  els.resetButton.disabled = lifecycleLocked || state.previewRefreshPending || !state.image;
  els.clearImageButton.disabled = lifecycleLocked || state.previewRefreshPending || !state.image;
  const outsidePreviewReady = Boolean(
    state.image &&
    state.params &&
    state.webgpu.renderer &&
    !lifecycleLocked,
  );
  // Display-only toggles are safe while a slider refresh is pending: requests
  // are coalesced by the preview revision loop and do not change parameters.
  els.outsidePreviewToggle.disabled = !outsidePreviewReady;
  els.pauseButton.disabled = !state.running;
  const webGpuChecking = state.webgpu.reason === "checking";
  const webGpuUnavailable = !state.webgpu.supported && !webGpuChecking && !state.webGpuRecoveryPending;
  els.webGpuNotice.hidden = !webGpuUnavailable;
  if (webGpuUnavailable) {
    els.webGpuNoticeText.textContent = webGpuUnavailableMessage(state.webgpu.reason);
  }
  els.retryWebGpuButton.disabled = lifecycleLocked || state.webgpu.supported;
  els.pauseButton.textContent = state.paused ? "Resume" : "Pause";
  data.resetEnabled = String(!els.resetButton.disabled);
  data.clearImageEnabled = String(!els.clearImageButton.disabled);
  data.pauseEnabled = String(!els.pauseButton.disabled);
  data.step = els.stepText.textContent;
  data.loss = els.lossText.textContent;
  data.psnr = els.psnrText.textContent;
  data.ssim = els.ssimText.textContent;
  data.regionalSsimP10 = els.regionalSsimText.textContent;
  data.imageSize = els.imageSizeText.textContent;
  data.coverageStatus = els.coverageText.textContent;
  data.backgroundPixels = Number.isFinite(state.metrics?.coverage_stats?.background_exposure_ratio)
    ? `${(state.metrics.coverage_stats.background_exposure_ratio * 100).toFixed(2)}%`
    : "-";
  data.outsideSplats = Number.isFinite(state.metrics?.outside_render_splat_count)
    ? Number(state.metrics.outside_render_splat_count).toLocaleString()
    : "-";
  data.trainingTiming = els.trainingTimingText.textContent;
  data.currentIterationsPerSecond = String(state.trainingTiming.iterationsPerSecond || "");
  data.currentTrainingElapsedMs = String(state.trainingTiming.elapsedMs || 0);
  data.gpuActiveBytes = String(state.gpuMemory.activeBytes);
  data.gpuReservedBytes = String(state.gpuMemory.reservedBytes);
  data.gpuActiveMb = bytesToMB(state.gpuMemory.activeBytes).toFixed(1);
  data.gpuReservedMb = bytesToMB(state.gpuMemory.reservedBytes).toFixed(1);
  data.gpuTrainingActiveBytes = String(state.gpuMemory.trainingActiveBytes);
  data.gpuTrainingReservedBytes = String(state.gpuMemory.trainingReservedBytes);
  data.gpuResultActiveBytes = String(state.gpuMemory.resultActiveBytes);
  data.gpuResultReservedBytes = String(state.gpuMemory.resultReservedBytes);
  data.gpuMemoryScope = "tracked-persistent-app-buffers";
  data.gpuTransientResourcesIncluded = "false";
  const resultRenderMemory =
    state.webgpu.renderer?.resultRenderMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
  data.resultRenderCacheBytes = String(resultRenderMemory.activeBytes || 0);
  data.resultRenderCacheMb = bytesToMB(resultRenderMemory.activeBytes || 0).toFixed(1);
  data.resultRenderCacheSource = state.metrics?.result_render_cache?.source || "";
  data.resultRenderCacheOrder = state.metrics?.result_render_cache?.order_mode || "";
  data.splatBytes = state.params ? String(state.params.count * ROW_BYTES) : "0";
  data.brushTaperMinimum = Number.isFinite(state.metrics?.brush_taper_stats?.minimum)
    ? String(state.metrics.brush_taper_stats.minimum)
    : "";
  data.brushTaperMean = Number.isFinite(state.metrics?.brush_taper_stats?.mean)
    ? String(state.metrics.brush_taper_stats.mean)
    : "";
  data.brushTaperMaximum = Number.isFinite(state.metrics?.brush_taper_stats?.maximum)
    ? String(state.metrics.brush_taper_stats.maximum)
    : "";
  data.lastDownload = state.lastDownload;
  data.lastGpuLoss = state.lastGpuLoss === null ? "" : String(state.lastGpuLoss);
  // Keep the DOM on the same WebGPU metric snapshot used by logs and exports.
  // This repairs a stale label after a preview-only canvas redraw.
  syncDisplayedSsimMetrics();
  if (state.metrics) {
    els.splatText.textContent = `${state.metrics.num_gaussians} / ${state.metrics.final_splats}`;
  } else if (state.params) {
    els.splatText.textContent = String(state.params.count);
  } else {
    els.splatText.textContent = "-";
  }
  data.inputMode = state.lastInputMode;
  data.resizeMode = "max-side";
  data.initialSplatInput = els.initialSplatCount.value;
  data.finalSplatInput = els.finalSplatCount.value;
  data.capacityMode = els.capacityMode.value;
  data.capacityProbeStatus = state.capacityProbe.status;
  data.capacityProbeRequested = String(state.capacityProbe.requested || 0);
  data.capacityProbeSelected = String(state.capacityProbe.selected || 0);
  data.capacityProbeFastPath = String(Boolean(state.capacityProbe.fastPath));
  data.capacityProbeAttempts = String(state.capacityProbe.attempts.length);
  data.initializationMode = state.metrics?.initialization || "image-rgb-grid";
  data.adaptiveGridInitializationFraction = String(adaptiveGridInitializationVariants().fraction * 100);
  data.previewRefreshInput = els.previewRefresh.value;
  data.liveQualityMetricsInput = String(Boolean(els.liveQualityMetrics.checked));
  const contributionCompaction = currentContributionCompactionSettings();
  data.currentContributionCompactionInput = String(contributionCompaction.enabled);
  data.currentContributionCompactionStartInput = String(contributionCompaction.startFraction * 100);
  data.currentContributionCompactionIntervalInput = String(contributionCompaction.requestedIntervalSteps);
  data.currentContributionCompactionMaxRemovalInput = String(contributionCompaction.maxRemovalFraction * 100);
  data.currentContributionCompactionNearZeroInput = String(contributionCompaction.nearZeroMaxFraction * 100);
  data.currentContributionCompactionWindowInput = String(contributionCompaction.requestedWindowSteps);
  data.currentContributionCompactionEffectiveWindow = String(contributionCompaction.measurementWindowSteps);
  data.blendMode = "standard-alpha";
  data.flowStrokeCoherenceInput = els.flowStrokeCoherence?.value || "0";
  data.gpuDensifyEnabled = "true";
  data.tileCullingEnabled = String(Boolean(els.tileCullingToggle?.checked));
  data.trainLayerOrderInput = String(Boolean(els.trainLayerOrder?.checked));
  data.layerAwareAccumulationInput = String(Boolean(els.layerAwareAccumulation?.checked));
  data.discreteLayersInput = String(Boolean(els.discreteLayers?.checked));
  data.discreteLayerCountInput = els.discreteLayerCount?.value || String(DEFAULT_DISCRETE_LAYER_COUNT);
  data.discreteLayerMoveRadiusInput = els.discreteLayerMoveRadius?.value || String(DEFAULT_DISCRETE_LAYER_MOVE_RADIUS);
  data.layerUpdateIntervalInput = els.layerUpdateInterval.value;
  data.positionLearningRateInput = els.positionLearningRate.value;
  data.colorLearningRateInput = els.colorLearningRate.value;
  data.opacityLearningRateInput = els.opacityLearningRate.value;
  data.alphaLossWeightInput = els.alphaLossWeight.value;
  const rectangleLearnedOpacity = selectedRectangleLearnedOpacity();
  const brushLearnedOpacity = selectedLayeredBrushLearnedOpacity();
  data.rectangleLearnedOpacityRange = `${rectangleLearnedOpacity.min},${rectangleLearnedOpacity.max}`;
  data.layeredBrushLearnedOpacityRange = `${brushLearnedOpacity.min},${brushLearnedOpacity.max}`;
  const brushDirectionalEffects = selectedLayeredBrushDirectionalEffects();
  const sharedColorWorkflow = selectedSharedColorWorkflow();
  data.layeredBrushOpacityGradientInput = String(brushDirectionalEffects.opacity);
  data.layeredBrushOpacityGradientRange = `${brushDirectionalEffects.opacityStart},${brushDirectionalEffects.opacityEnd}`;
  data.layeredBrushWidthTaperInput = String(brushDirectionalEffects.widthTaper);
  data.layeredBrushWidthTaperRange = `${brushDirectionalEffects.widthStart},${brushDirectionalEffects.widthEnd}`;
  data.layeredBrushLocalColorFlowInput = String(
    Boolean(els.layeredBrushLocalColorFlow?.checked),
  );
  data.layeredBrushStrokePersistenceInput = String(
    Boolean(els.layeredBrushStrokePersistence?.checked),
  );
  const brushAspectFloors = selectedBrushAspectFloors();
  data.layeredBrushMinAspectRatioInput = String(selectedBrushMinAspectRatio());
  data.layeredBrushMaxAspectRatioInput = String(selectedBrushMaxAspectRatio());
  data.layeredBrushRibbonAspectFloorInput = String(brushAspectFloors.ribbon);
  data.layeredBrushAccentAspectFloorInput = String(brushAspectFloors.accent);
  data.monochromeUnderpaintingInput = String(sharedColorWorkflow.monochromeUnderpainting);
  data.colorFinishStartInput = String(sharedColorWorkflow.colorFinishStartPercent);
  data.rectangleTopRatioInput = String(selectedRectangleTopRatio());
  data.rectangleTopRatioMaxInput =
    String(selectedRectangleTopRatioMax(selectedRectangleTopRatio()));
  data.rectangleMinAspectRatioInput = String(selectedRectangleMinAspectRatio());
  data.rectangleMaxAspectRatioInput =
    String(selectedRectangleMaxAspectRatio(selectedRectangleMinAspectRatio()));
  data.rectangleOrientationInput = selectedRectangleOrientation();
  data.rectangleOrientationToleranceInput = String(selectedRectangleOrientationToleranceDegrees());
  data.rectangleOrientationToleranceUnit = "degrees";
  data.rectangleOrientationConstraints = state.metrics?.rectangle_orientation_constraints
    ? JSON.stringify(state.metrics.rectangle_orientation_constraints)
    : "";
  const rectangleShape = selectedRectangleShapeSettings();
  data.rectanglePreserveAreaInput = String(rectangleShape.preserveArea);
  data.rectangleEdgeDirectedTaperInput = String(rectangleShape.edgeDirectedTaper);
  data.rectangleStructureAwareRatioInput = String(rectangleShape.structureAwareRatio);
  data.rectangleAsymmetricSoftnessInput = String(rectangleShape.asymmetricSoftness);
  const surfaceLayerPrior = scaleBiasedSurfaceLayerPriorSettings();
  data.scaleBiasedSurfaceLayerPriorInput = String(surfaceLayerPrior.enabled);
  data.scaleBiasedSurfaceLayerPriorLayersInput = String(surfaceLayerPrior.layers);
  data.scaleBiasedSurfaceLayerPriorP1IntervalInput = String(surfaceLayerPrior.p1Interval);
  data.scaleBiasedSurfaceLayerPriorP2IntervalInput = String(surfaceLayerPrior.p2Interval);
  data.scaleBiasedSurfaceLayerPriorP3IntervalInput = String(surfaceLayerPrior.p3Interval);
  data.scaleBiasedSurfaceLayerPriorUntilInput = String(surfaceLayerPrior.untilFraction * 100);
  data.virtualCameraSamplingInput = String(algorithmUsesVirtualCameras());
  data.virtual3dgsMultiviewInput = "true";
  data.virtualGofDensityInput = String(Boolean(els.virtualGofDensity.checked));
  data.virtualCameraShareInput = els.virtualCameraShare.value;
  data.virtualCameraMaxAngleInput = els.virtualCameraMaxAngle.value;
  data.virtualCameraCountInput = els.virtualCameraCount.value;
  data.virtualCameraFovInput = els.virtualCameraFov.value;
  data.virtualCameraCoverageMinimum = String(state.metrics?.virtual_camera_sampling?.teacher_coverage?.minimum ?? "");
  data.virtualCameraSamplingActive = String(Boolean(state.metrics?.virtual_camera_sampling?.enabled));
  data.virtualCameraActiveId = state.metrics?.virtual_camera_sampling?.active_camera_id || "";
  data.virtualCameraSamplingMode = state.metrics?.virtual_camera_sampling?.mode || "off";
  data.virtualCameraFrontSteps = String(state.metrics?.virtual_camera_sampling?.front_steps ?? 0);
  data.virtualCameraVirtualSteps = String(state.metrics?.virtual_camera_sampling?.virtual_steps ?? 0);
  data.virtualCameraEvaluationCount = String(state.metrics?.virtual_camera_evaluation?.virtual_views?.camera_count ?? 0);
  data.virtualCameraVirtualPsnr = String(state.metrics?.virtual_camera_evaluation?.virtual_views?.rgb_psnr_macro ?? "");
  data.virtualCameraVirtualSsim = String(state.metrics?.virtual_camera_evaluation?.virtual_views?.rgb_ssim_macro ?? "");
  data.virtualCameraVirtualSsimP10 = String(state.metrics?.virtual_camera_evaluation?.virtual_views?.rgb_ssim_p10 ?? "");
  data.virtualCameraAllViewSsim = String(state.metrics?.virtual_camera_evaluation?.all_views?.rgb_ssim_macro ?? "");
  data.virtualCameraAllViewPsnr = String(state.metrics?.virtual_camera_evaluation?.all_views?.rgb_psnr_macro ?? "");
  data.virtualCameraAllViewSsimP10 = String(state.metrics?.virtual_camera_evaluation?.all_views?.rgb_ssim_p10 ?? "");
  const virtualEvaluation = state.metrics?.virtual_camera_evaluation;
  data.virtualCameraEvaluationSummary = virtualEvaluation
    ? JSON.stringify({
      backend: virtualEvaluation.backend,
      target: virtualEvaluation.target,
      aggregation: virtualEvaluation.aggregation,
      front_view: virtualEvaluation.front_view,
      virtual_views: virtualEvaluation.virtual_views,
      all_views: virtualEvaluation.all_views,
    })
    : "";
  const obliqueViews = state.metrics?.oblique_overlap_diagnostics?.views;
  data.virtualCameraTiltSummary = obliqueViews
    ? JSON.stringify(Object.fromEntries(
      ["pitch-30", "yaw-30", "pitch-60", "yaw-60"].map((key) => [key, obliqueViews[key]?.["1"] ?? null]),
    ))
    : "";
  data.colorSpaceContract = state.metrics?.color_space_audit?.contract || "front and virtual teachers use the same sRGB signal values";
  data.frontLumaDelta = String(state.metrics?.color_space_audit?.training_minus_source_luma ?? "");
  data.renderParityAlphaMax = String(state.metrics?.render_surface_parity?.alpha_max_abs ?? "");
  data.renderParityPremultipliedMax = String(state.metrics?.render_surface_parity?.premultiplied_max_abs ?? "");
  data.renderParityMismatchPixels = String(state.metrics?.render_surface_parity?.premultiplied_mismatch_pixels ?? "");
  data.layerAwareParityPremultipliedMax = String(
    state.metrics?.layer_aware_render_parity?.premultiplied_max_abs ?? "",
  );
  data.renderParityMaximumPixel = state.metrics?.render_surface_parity?.maximum_pixel
    ? JSON.stringify(state.metrics.render_surface_parity.maximum_pixel)
    : "";
  data.stageAwareGrowthInput = String(Boolean(els.stageAwareGrowth.checked));
  const stageShares = phase39Variants().stageGrowthShares;
  data.stageGrowthShares = `${(stageShares.p1 * 100).toFixed(2)},${(stageShares.p2 * 100).toFixed(2)},${(stageShares.p3 * 100).toFixed(2)}`;
  data.structureGuidedAllocationInput = String(Boolean(els.structureGuidedAllocation.checked));
  data.structureRegionGridInput = els.structureRegionGrid.value;
  data.midTrainingOverdensityCorrectionInput = String(Boolean(els.midTrainingOverdensityCorrection.checked));
  data.overdensityCorrectionScheduleInput = els.overdensityCorrectionSchedule.value;
  data.overdensityCorrectionIntervalInput = els.overdensityCorrectionInterval.value;
  data.overdensityDonorPercentInput = els.overdensityDonorPercent.value;
  data.scaleLearningRateInput = els.scaleLearningRate.value;
  data.rotationLearningRateInput = els.rotationLearningRate.value;
  data.thetaAlignRateInput = els.thetaAlignRate.value;
  data.maxAnisotropyInput = els.maxAnisotropy.value;
  data.maxPlanarScaleInput = els.maxPlanarScale.value;
  data.boundarySigmaInput = els.boundarySigma.value;
  data.adaptiveDetailInput = "true";
  data.detailCoherenceInput = els.detailCoherence.value;
  data.densifyIntervalInput = els.densifyInterval.value;
  data.growthPercentageInput = els.growthPercentage.value;
  data.growthApplyUntilInput = els.growthApplyUntil.value;
  data.growthSignalThresholdInput = els.growthSignalThreshold.value;
  data.subgroupExactBackward = String(Boolean(state.webgpu.renderer?.subgroupExactBackwardEnabled));
  data.subgroupsAvailable = String(Boolean(state.webgpu.subgroups));
  data.safetyMode = state.safety.mode;
  data.safetyStopReason = state.safety.lastStopReason;
  data.safetyStopEstimateMb = state.safety.lastStopEstimateMB;
  data.safetyStopBudgetMb = state.safety.lastStopBudgetMB;
  data.safetyRecommended = state.safety.lastRecommended;
  data.memoryLimiterUnlocked = String(state.safety.memoryLimiterUnlocked);
  if (state.recommendation) {
    data.memoryBudgetMb = String(state.recommendation.budgetMB);
    data.memoryBudgetMode = String(state.recommendation.memoryBudgetMode);
    data.memoryBudgetMultiplier = String(state.recommendation.memoryBudgetMultiplier);
    data.memoryEstimatedMb = String(state.recommendation.estimatedMB);
    data.memoryHeadroomMb = String(state.recommendation.headroomMB);
    data.memoryHintSource = String(state.recommendation.memoryHintSource);
    data.memoryHintMb = String(state.recommendation.memoryHintMB);
    data.autoBudgetSource = String(state.recommendation.autoBudgetSource);
    data.autoBudgetReservedMb = String(state.recommendation.autoReservedMB);
    data.autoBudgetEnvelopeMb = String(state.recommendation.autoEnvelopeMB);
    data.deviceLimiterTrainSize = String(state.recommendation.limiterTrainSize);
    data.deviceLimiterFinalSplats = String(state.recommendation.limiterFinalSplats);
    data.recommendedTrainSize = String(state.recommendation.recommendedTrainSize);
    data.recommendedFinalSplats = String(state.recommendation.recommendedFinalSplats);
    data.metricsInterval = String(state.recommendation.metricInterval);
    data.speedWork = String(state.recommendation.previewWork);
    data.vramExact = "false";
  } else {
    data.memoryBudgetMb = "";
    data.memoryBudgetMode = "";
    data.memoryBudgetMultiplier = "";
    data.memoryEstimatedMb = "";
    data.memoryHeadroomMb = "";
    data.memoryHintSource = "";
    data.memoryHintMb = "";
    data.autoBudgetSource = "";
    data.autoBudgetReservedMb = "";
    data.autoBudgetEnvelopeMb = "";
    data.deviceLimiterTrainSize = "";
    data.deviceLimiterFinalSplats = "";
    data.recommendedTrainSize = "";
    data.recommendedFinalSplats = "";
    data.metricsInterval = "";
    data.speedWork = "";
    data.vramExact = "false";
  }
  if (state.metrics) {
    const lastExport = state.metrics.export_history?.[state.metrics.export_history.length - 1];
    const lastPerformance = state.metrics.performance_trace?.[state.metrics.performance_trace.length - 1];
    data.lastExportFormat = String(lastExport?.format ?? "");
    data.lastExportFilename = String(lastExport?.filename ?? "");
    data.lastExportBytes = String(lastExport?.bytes ?? "");
    data.lastExportWidth = String(lastExport?.width ?? "");
    data.lastExportHeight = String(lastExport?.height ?? "");
    data.lastExportNonblackPixels = String(lastExport?.nonblack_pixels ?? "");
    data.lastExportMeanRgb = String(lastExport?.mean_rgb ?? "");
    data.lastExportRoundTripCount = String(lastExport?.round_trip?.count ?? lastExport?.round_trip?.vertices ?? "");
    data.lastExportRoundTripShDegree = String(lastExport?.round_trip?.sh_degree ?? (lastExport?.round_trip?.sh_degree_0 ? 0 : ""));
    data.lastExportRoundTripZAbsMax = String(lastExport?.round_trip?.z_abs_max ?? (lastExport?.round_trip?.all_z_zero ? 0 : ""));
    data.lastExportRoundTripNonfinite = String(lastExport?.round_trip?.nonfinite?.length ?? (lastExport?.round_trip?.all_finite ? 0 : ""));
    data.initialL1 = String(state.metrics.initial_l1);
    data.finalL1 = String(state.metrics.final_l1);
    data.initialRgbMse = String(state.metrics.initial_rgb_mse ?? "");
    data.finalRgbMse = String(state.metrics.final_rgb_mse ?? "");
    data.initialPsnr = String(state.metrics.initial_psnr ?? "");
    data.finalPsnr = String(state.metrics.final_psnr ?? "");
    data.psnrUnit = state.metrics.quality_metric_contract?.psnr_unit || "dB";
    data.psnrSignalSpace = state.metrics.quality_metric_contract?.signal_space || "sRGB signal values";
    data.finalGlobalSsim = String(state.metrics.final_global_ssim ?? "");
    data.finalWindowedSsim = String(state.metrics.final_windowed_ssim ?? state.metrics.final_ssim ?? "");
    data.finalLocalP10 = String(state.metrics.final_regional_ssim?.p10 ?? "");
    data.finalAlphaL1 = String(state.metrics.final_alpha_l1 ?? "");
    data.finalAlphaSsim = String(state.metrics.final_alpha_ssim ?? "");
    data.trainingElapsedMs = String(lastPerformance?.elapsed_ms ?? "");
    data.trainingIterationsPerSecond = String(lastPerformance?.iterations_per_second ?? "");
    data.trainingEvaluationMode = state.metrics.training_evaluation?.mode || "";
    data.trainingEvaluationSource = state.metrics.training_evaluation?.source || "";
    data.trainingPeriodicEvaluations = String(
      state.metrics.training_evaluation?.periodic_full_image_evaluations ?? 0,
    );
    data.trainingResidualRefreshes = String(state.metrics.training_residual_map?.refresh_count ?? 0);
    data.trainingResidualCpuReadbacks = String(state.metrics.training_residual_map?.cpu_readbacks ?? 0);
    data.finalParameterHash = String(state.metrics.final_parameter_hash ?? "");
    data.stepsDone = String(state.metrics.steps_done);
    data.stepsRequested = String(state.metrics.steps_requested);
    data.stopped = String(state.metrics.stopped);
    data.gaussians = String(state.metrics.num_gaussians);
    data.initialSplats = String(state.metrics.initial_splats);
    data.initialization = String(state.metrics.initialization ?? "");
    data.finalSplats = String(state.metrics.final_splats);
    data.lossBackend = state.metrics.loss_backend || "";
    data.webgpuComputeLoss = String(Boolean(state.metrics.webgpu_compute_loss));
    data.webgpuTrainExecuted = String(Boolean(state.metrics.webgpu_train_executed));
    data.webgpuTrainUpdate = String(Boolean(state.metrics.webgpu_train_update));
    data.webgpuTrainError = state.metrics.webgpu_train_error || "";
    data.virtualTiltSteps = String(state.metrics.webgpu_train_stats?.virtual_tilt_steps_completed ?? 0);
    data.virtualTiltLastStep = String(state.metrics.webgpu_train_stats?.last_virtual_tilt?.step ?? "");
    data.virtualTiltLastPitch = String(state.metrics.webgpu_train_stats?.last_virtual_tilt?.pitchDegrees ?? "");
    data.virtualTiltLastYaw = String(state.metrics.webgpu_train_stats?.last_virtual_tilt?.yawDegrees ?? "");
    data.virtualTiltEnabled = String(Boolean(state.metrics.webgpu_train_stats?.virtual_tilt?.enabled));
    data.virtualTiltInterval = String(state.metrics.webgpu_train_stats?.virtual_tilt?.interval ?? "");
    data.curriculumCoarseSteps = String(state.metrics.webgpu_train_stats?.coarse_steps_completed ?? 0);
    data.curriculumMidSteps = String(state.metrics.webgpu_train_stats?.mid_steps_completed ?? 0);
    data.curriculumTrainingStage = String(state.metrics.webgpu_train_stats?.training_stage ?? "");
    data.ewa2x2 = String(Boolean(state.metrics.phase33_variants?.ewa2x2));
    data.exactBackward = String(Boolean(state.metrics.phase46_variants?.qualityRecovery?.exactBackward ?? qualityRecoveryVariants().exactBackward));
    data.tiltRiskCount = String(state.metrics.tilt_risk?.risky_count ?? 0);
    data.tiltRiskRatio = String(state.metrics.tilt_risk?.risky_ratio ?? 0);
    data.tiltRiskSupportP99 = String(state.metrics.tilt_risk?.support_depth_p99 ?? 0);
    data.experimentalPrefixPreserved = String(state.metrics.experimental_prefix_preserved !== false);
    data.densifyEvents = String(state.metrics.densify_events?.length || 0);
    data.growthScheduleMode = String(state.metrics.growth_schedule?.mode || "");
    data.growthPercentage = String(state.metrics.growth_schedule?.percentage ?? "");
    data.growthSignalThreshold = String(state.metrics.growth_schedule?.signal_threshold ?? "");
    data.growthThresholdSkips = String(state.metrics.growth_schedule?.threshold_skips ?? 0);
    data.stageAwareGrowth = String(Boolean(state.metrics.growth_schedule?.stage_aware));
    data.growthCapReachedStep = String(state.metrics.growth_schedule?.cap_reached_step ?? "");
    data.layerUpdateInterval = String(state.metrics.layer_update_interval ?? "");
    data.layerUpdateCount = String(state.metrics.layer_update_count ?? 0);
    data.layerUpdateFirstSteps = String(state.metrics.layer_update_first_steps ?? "");
    data.layerUpdateLastStep = String(state.metrics.layer_update_last_step ?? "");
    data.fusionAdcSplitEvents = String(state.metrics.fusion_events?.adc_split || 0);
    data.fusionAdcDuplicateEvents = String(state.metrics.fusion_events?.adc_duplicate || 0);
    data.fusionMcmcTeleportEvents = String(state.metrics.fusion_events?.mcmc_teleport || 0);
    data.fusionMcmcReseedEvents = String(state.metrics.fusion_events?.mcmc_reseed || 0);
    data.fusionPruneEvents = String(state.metrics.fusion_events?.prune || 0);
    data.fusionOpacityResetEvents = String(state.metrics.fusion_events?.opacity_reset || 0);
    data.tiltRiskCandidates = String(state.metrics.fusion_events?.tilt_risk_candidates || 0);
    data.tiltTrueSplits = String(state.metrics.fusion_events?.tilt_true_splits || 0);
    data.tiltOpacitySaturations = String(state.metrics.fusion_events?.tilt_opacity_saturations || 0);
    data.tileIndexTotal = String(state.metrics.tile_counters?.total ?? "");
    data.tileIndexCapacity = String(state.metrics.tile_counters?.capacity ?? "");
    data.tileIndexOverflow = String(state.metrics.tile_counters?.overflow ?? "");
    data.tileRetrySteps = String(state.metrics.tile_retry_steps ?? 0);
    data.tileRetryEvents = String(state.metrics.tile_retry_events?.length ?? 0);
    data.tileRetryParameterHashMatches = String(state.metrics.tile_retry_parameter_hash?.matches ?? "");
    data.qaTileIndexCapacity = String(state.metrics.qa_tile_index_capacity ?? "");
    data.tileAverageCandidates = String(state.metrics.tile_counters?.average_candidates ?? "");
    data.activeSplats = String(state.metrics.tile_counters?.active_count ?? state.metrics.num_gaussians ?? "");
    data.freeSplats = String(state.metrics.tile_counters?.free_count ?? "");
    data.ssimTrend = state.metrics.ssim_trend || "";
    data.globalSsimTrend = state.metrics.global_ssim_trend || "";
    data.psnrTrend = state.metrics.psnr_trend || "";
    data.initialSsim = String(state.metrics.initial_ssim);
    data.finalSsim = String(state.metrics.final_ssim);
    data.initialGlobalSsim = String(state.metrics.initial_global_ssim);
    data.finalGlobalSsim = String(state.metrics.final_global_ssim);
    data.initialWindowedSsim = String(state.metrics.initial_windowed_ssim);
    data.finalWindowedSsim = String(state.metrics.final_windowed_ssim);
    data.initialRegionalSsimP10 = String(state.metrics.initial_regional_ssim?.p10 ?? "");
    data.finalRegionalSsimP10 = String(state.metrics.final_regional_ssim?.p10 ?? "");
    data.finalRegionalSsimMinimum = String(state.metrics.final_regional_ssim?.minimum ?? "");
    data.finalRegionalSsimMedian = String(state.metrics.final_regional_ssim?.median ?? "");
    data.initialAlphaSsim = String(state.metrics.initial_alpha_ssim ?? "");
    data.finalAlphaSsim = String(state.metrics.final_alpha_ssim ?? "");
    data.initialAlphaL1 = String(state.metrics.initial_alpha_l1 ?? "");
    data.finalAlphaL1 = String(state.metrics.final_alpha_l1 ?? "");
    data.lrScale = String(state.metrics.lr_scale);
    data.positionLearningRate = String(state.metrics.learning_rates?.position ?? "");
    data.colorLearningRate = String(state.metrics.learning_rates?.color ?? "");
    data.opacityLearningRate = String(state.metrics.learning_rates?.opacity ?? "");
    data.alphaLossWeight = String(state.metrics.alpha_loss_weight ?? "");
    data.scaleLearningRate = String(state.metrics.learning_rates?.scale ?? "");
    data.rotationLearningRate = String(state.metrics.learning_rates?.rotation ?? "");
    data.thetaAlignRate = String(state.metrics.learning_rates?.thetaAlign ?? "");
    data.maxAnisotropy = String(state.metrics.learning_rates?.maxAnisotropy ?? "");
    data.boundarySigma = String(state.metrics.boundary_sigma ?? "");
    data.outsideRenderSplatCount = String(state.metrics.outside_render_splat_count ?? "");
    data.adaptiveDetail = String(Boolean(state.metrics.learning_rates?.adaptiveDetail));
    data.detailCoherence = String(state.metrics.learning_rates?.detailCoherence ?? "");
    data.detailSplatCount = String(state.metrics.detail_splat_count ?? "");
    data.detailSplatRatio = String(state.metrics.detail_splat_ratio ?? "");
    data.paramDeltaPosition = String(state.metrics.param_delta?.position ?? "");
    data.paramDeltaColor = String(state.metrics.param_delta?.color ?? "");
    data.paramDeltaOpacity = String(state.metrics.param_delta?.opacity ?? "");
    data.paramDeltaScale = String(state.metrics.param_delta?.scale ?? "");
    data.paramDeltaRotation = String(state.metrics.param_delta?.rotation ?? "");
    data.initialOrientationBins = JSON.stringify(state.metrics.initial_orientation?.bins ?? []);
    data.phaseOneShapeLrMultiplier = String(state.webgpu.renderer?.lastTrainStats?.phase_one_shape_lr_multiplier ?? "");
    data.phaseOneMaxPlanarScale = String(state.webgpu.renderer?.lastTrainStats?.phase_one_max_planar_scale ?? "");
    data.boundaryLeakCount = String(state.metrics.boundary_leak_count ?? "");
    data.boundaryMaxLeak = String(state.metrics.boundary_max_leak ?? "");
    data.backgroundExposureCount = String(state.metrics.coverage_stats?.background_exposure_count ?? "");
    data.backgroundExposureRatio = String(state.metrics.coverage_stats?.background_exposure_ratio ?? "");
    data.alphaDarkMean = String(state.metrics.coverage_stats?.luminance_buckets?.dark?.mean_alpha ?? "");
    data.alphaMidMean = String(state.metrics.coverage_stats?.luminance_buckets?.mid?.mean_alpha ?? "");
    data.alphaLightMean = String(state.metrics.coverage_stats?.luminance_buckets?.light?.mean_alpha ?? "");
    data.defaultOutput = state.metrics.default_output || "";
    data.lossCount = String(state.metrics.losses?.length || 0);
    data.ssimCount = String(state.metrics.ssim?.length || 0);
    data.windowedSsimCount = String(state.metrics.windowed_ssim?.length || 0);
    data.runAlgorithm = state.metrics.algorithm || "";
    data.inputOriginalWidth = String(state.metrics.input_original_size?.[0] || "");
    data.inputOriginalHeight = String(state.metrics.input_original_size?.[1] || "");
    data.resizeScale = String(state.metrics.resize_scale ?? "");
    data.previewRefresh = String(state.metrics.preview_refresh ?? "");
    data.previewFrames = String(state.metrics.preview_frames ?? 0);
    data.lastPreviewStep = String(state.metrics.last_preview_step ?? "");
    data.paramsRevision = String(state.metrics.params_revision ?? 0);
    data.coverageRevision = String(state.metrics.coverage_revision ?? "");
  } else {
    data.lastExportFormat = "";
    data.lastExportFilename = "";
    data.lastExportBytes = "";
    data.lastExportWidth = "";
    data.lastExportHeight = "";
    data.lastExportNonblackPixels = "";
    data.lastExportMeanRgb = "";
    data.lastExportRoundTripCount = "";
    data.lastExportRoundTripShDegree = "";
    data.lastExportRoundTripZAbsMax = "";
    data.lastExportRoundTripNonfinite = "";
    data.initialL1 = "";
    data.finalL1 = "";
    data.trainingEvaluationMode = "";
    data.trainingEvaluationSource = "";
    data.trainingPeriodicEvaluations = "";
    data.trainingResidualRefreshes = "";
    data.trainingResidualCpuReadbacks = "";
    data.finalParameterHash = "";
    data.stepsDone = "";
    data.stepsRequested = "";
    data.stopped = "";
    data.gaussians = "";
    data.initialSplats = "";
    data.initialization = "";
    data.finalSplats = "";
    data.lossBackend = "";
    data.webgpuComputeLoss = "";
    data.webgpuTrainExecuted = "";
    data.webgpuTrainUpdate = "";
    data.webgpuTrainError = "";
    data.experimentalPrefixPreserved = "";
    data.densifyEvents = "";
    data.fusionAdcSplitEvents = "";
    data.fusionAdcDuplicateEvents = "";
    data.fusionMcmcTeleportEvents = "";
    data.fusionMcmcReseedEvents = "";
    data.fusionPruneEvents = "";
    data.fusionOpacityResetEvents = "";
    data.tileIndexTotal = "";
    data.tileIndexCapacity = "";
    data.tileIndexOverflow = "";
    data.tileAverageCandidates = "";
    data.activeSplats = "";
    data.freeSplats = "";
    data.ssimTrend = "";
    data.globalSsimTrend = "";
    data.initialSsim = "";
    data.finalSsim = "";
    data.initialGlobalSsim = "";
    data.finalGlobalSsim = "";
    data.initialWindowedSsim = "";
    data.finalWindowedSsim = "";
    data.initialRegionalSsimP10 = "";
    data.finalRegionalSsimP10 = "";
    data.finalRegionalSsimMinimum = "";
    data.finalRegionalSsimMedian = "";
    data.lrScale = "";
    data.positionLearningRate = "";
    data.colorLearningRate = "";
    data.opacityLearningRate = "";
    data.scaleLearningRate = "";
    data.rotationLearningRate = "";
    data.thetaAlignRate = "";
    data.maxAnisotropy = "";
    data.boundarySigma = "";
    data.outsideRenderSplatCount = "";
    data.adaptiveDetail = "";
    data.detailCoherence = "";
    data.paramDeltaPosition = "";
    data.paramDeltaColor = "";
    data.paramDeltaOpacity = "";
    data.paramDeltaScale = "";
    data.paramDeltaRotation = "";
    data.initialOrientationBins = "";
    data.phaseOneShapeLrMultiplier = "";
    data.phaseOneMaxPlanarScale = "";
    data.boundaryLeakCount = "";
    data.boundaryMaxLeak = "";
    data.backgroundExposureCount = "";
    data.backgroundExposureRatio = "";
    data.defaultOutput = "";
    data.lossCount = "";
    data.ssimCount = "";
    data.windowedSsimCount = "";
    data.runAlgorithm = "";
    data.inputOriginalWidth = "";
    data.inputOriginalHeight = "";
    data.resizeScale = "";
    data.previewRefresh = "";
    data.previewFrames = "";
    data.lastPreviewStep = "";
    data.paramsRevision = "";
    data.coverageRevision = "";
  }
  updateExportPanel();
}
