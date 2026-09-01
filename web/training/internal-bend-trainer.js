// Fixed topology orchestration for the opt-in analytic bend kernel. Each call
// below is one real update of the existing shared WebGPU optimizer, not Flow's
// 10-step schedule stride. No source curves or secondary optimizer are used.
async function trainInternalBend(run) {
  assertTrainingRun(run);
  if (!state.image || state.running) return;
  const renderer = run.renderer;
  if (!state.webgpu.supported || !renderer) throw Error("Internal bend requires WebGPU");
  cancelCompletedResultGpuRecovery();
  destroyTiltViewer({restoreCanvas: true});
  activateDetailTab("training");
  const steps = normalizeStepInteger(els.stepCount.value, {
    min: LIMITS.stepsMin, max: LIMITS.stepsMax, fallback: DEFAULT_ITERATIONS,
  });
  if (steps > HIGH_ITERATION_CONFIRM && !trainingUiAdapter.confirm(`Run ${steps.toLocaleString()} actual optimizer updates?`)) return;
  await resizeLoadedImageToMaxSide(Number(els.trainSize.value));
  updateTrainingRunOwnership(run, {image: state.image});
  assertTrainingRun(run);
  syncTrainSizeUi();
  clearSafetyStop();
  const count = normalizeUiSplatCount(els.finalSplatCount.value, DEFAULT_FINAL_SPLATS, 14000);
  const failure = safetyFailure(computeBudgetFor(Number(els.trainSize.value), count, steps), "start");
  if (failure) {setSafetyStop(failure); throw Error(`Internal bend safety guard: ${failure.reason}`);}
  const params = Image2SplatPaintInternalBend.initialize(state.image, count);
  const learningRates = {...selectedLearningRates(), opacity: 0};
  const previewRefresh = selectedPreviewRefresh();
  const periodicMetrics = periodicTrainingEvaluationEnabled();
  const metricInterval = Math.max(1, Math.min(DEFAULT_MAX_METRIC_INTERVAL,
    state.recommendation?.metricInterval || Math.floor(steps / 60)));
  const metrics = {
    algorithm: FLOW_SPLAT_FUSION_ALGORITHM_ID, flow_training_path: "internal-bend",
    backend: "webgpu", steps_requested: steps, steps_done: 0, num_gaussians: count,
    initial_splats: count, final_splats: count, initial_param_snapshot: snapshotParams(params),
    started_at: new Date().toISOString(), stopped: false, params_revision: 0,
    preview_frames: 0, preview_refresh: previewRefresh, tile_retry_steps: 0,
    initial_ssim: null, initial_global_ssim: null, initial_psnr: null,
    fusion_events: emptyFusionEvents(), learning_rates: learningRates,
    internal_bend: {version: 2, fixed_count: count, backcoat: params.flowBackcoatCount,
      layer_shares: [20, 40, 40], seed: 20260831, fusion: false, split: false, clone: false},
    gpu_training_memory: {peak_active_bytes: 0, peak_reserved_bytes: 0},
  };
  for (const key of ["losses", "rgb_mse", "psnr", "alpha_losses", "alpha_ssim", "objective_losses",
    "ssim", "global_ssim", "windowed_ssim", "regional_ssim_p10", "trend_checkpoints",
    "tile_retry_events", "webgpu_relocation_events", "webgpu_refine_events", "performance_trace"])
    metrics[key] = [];
  resetEvaluationStatusForNewTraining();
  state.params = params;
  state.metrics = metrics;
  state.flowSplatResult = null;
  state.running = true;
  state.paused = false;
  state.stopRequested = false;
  state.previewGeneration += 1;
  state.layerTelemetryState = null;
  state.layerEfficiencyCheckpoints = [];
  state.previewMode = previewRefresh === "final" ? "original" : "splats";
  updateTrainingRunOwnership(run, {params, metrics});
  clearSplatAdjustmentBaseline();
  resetTrainingTiming(false);
  updateDownloads(false);
  updatePreviewModeControls();
  setInputControlsDisabled(true);
  setPausedRuntimeControlsEnabled(false);
  els.startButton.disabled = true;
  els.pauseButton.disabled = false;
  els.stopButton.disabled = false;
  setStatus("running");
  let completed = false;
  try {
    // The locked v2 fixture uses ordinary floating-point gradients and AABB
    // tile bounds. Alternative gradient backends have not passed this gate.
    renderer.configureExperimentalPerformance({...performanceVariants(),
      shapeAwarePaintCulling: true, segmentedExactBackward: false, fixedPointExactGradient: false});
    await awaitTrainingRun(run, renderer.uploadTrainState(state.image, params, count));
    await awaitTrainingRun(run, renderer.prepareFlowBirthLinks(state.image, params));
    await updatePreview(0, false, {}, run);
    metrics.initial_ssim = metrics.latest_ssim;
    metrics.initial_global_ssim = metrics.latest_global_ssim;
    metrics.initial_psnr = metrics.latest_psnr;
    let retries = 0;
    for (let step = 1; step <= steps && !state.stopRequested;) {
      while (state.paused && !state.stopRequested) await awaitTrainingRun(run, nextFrame());
      if (state.stopRequested) break;
      const started = performance.now();
      await awaitTrainingRun(run, renderer.trainStepRenderGradientGpu(state.image, params, learningRates,
        {sync: true, currentStepOverride: step, suppressSgldNoise: true}));
      if (await resolveTileOverflowRetry(null, run)) {
        if (++retries > 4) throw Error("Internal bend tile allocation failed repeatedly");
        continue;
      }
      retries = 0;
      metrics.steps_done = step;
      metrics.params_revision += 1;
      recordTrainingTiming(step, performance.now() - started);
      if (periodicMetrics && step % metricInterval === 0) {
        await updatePreview(step, false, {readOnlyPeriodic: true, present: shouldPresentTrainingStep(step, previewRefresh)}, run);
        await awaitTrainingRun(run, nextFrame());
      } else if (shouldPresentTrainingStep(step, previewRefresh)) {
        await presentTrainingPreview(step, run);
        await awaitTrainingRun(run, nextFrame());
      } else if (step % 32 === 0) {
        els.stepText.textContent = `${step} / ${steps}`;
        publishState();
        await awaitTrainingRun(run, nextFrame());
      }
      const memory = renderer.trainingMemorySnapshot();
      metrics.gpu_training_memory.peak_active_bytes = Math.max(metrics.gpu_training_memory.peak_active_bytes, memory.activeBytes);
      metrics.gpu_training_memory.peak_reserved_bytes = Math.max(metrics.gpu_training_memory.peak_reserved_bytes, memory.reservedBytes);
      step += 1;
    }
    metrics.stopped = state.stopRequested;
    state.previewMode = "splats";
    updatePreviewModeControls();
    setStatus("finalizing");
    await updatePreview(metrics.steps_done, true, {}, run);
    metrics.final_cpu_result_ready_at = new Date().toISOString();
    metrics.final_result_cache_preserved = await awaitTrainingRun(run, renderer.preserveResultRenderState(state.image, params));
    captureSplatAdjustmentBaseline();
    completed = true;
  } finally {
    renderer.disposeTrainState();
    // Cancellation belongs to the newer lifecycle; do not unlock its controls.
    if (ownsTrainingRun(run)) {
      const memory = renderer.trainingMemorySnapshot();
      metrics.gpu_training_memory.active_bytes_after_release = memory.activeBytes;
      metrics.gpu_training_memory.reserved_bytes_after_release = memory.reservedBytes;
      metrics.finished_at = new Date().toISOString();
      state.running = false;
      state.paused = false;
      els.startButton.disabled = false;
      els.pauseButton.disabled = true;
      els.pauseButton.textContent = "Pause";
      els.stopButton.disabled = true;
      setPausedRuntimeControlsEnabled(false);
      setInputControlsDisabled(false);
      syncAlgorithmRequirements();
      updateDownloads(completed);
      if (completed) setStatus(metrics.stopped ? "stopped" : "done");
      publishState();
    }
  }
}
