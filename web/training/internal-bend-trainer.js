// Progressive active-count orchestration for the opt-in analytic bend kernel.
// Each call below is one real update of the shared WebGPU optimizer, not Flow's
// 10-step schedule stride. No source curves or secondary optimizer are used.
const INTERNAL_BEND_GROWTH_INTERVAL = 100;
const INTERNAL_BEND_GROWTH_APPLY_UNTIL = 0.90;
const INTERNAL_BEND_PHASE_GROWTH_SHARES = Object.freeze([0.20, 0.40, 0.40]);

function buildInternalBendGrowthSchedule(steps, initialCount, finalCount) {
  const safeSteps = Math.max(1, Math.round(steps));
  const start = Math.max(1, Math.min(finalCount, Math.round(initialCount)));
  const finish = Math.max(start, Math.round(finalCount));
  if (start >= finish) return {events: [], horizonStep: 0, initialCount: start, finalCount: finish};
  const horizonStep = Math.max(1, Math.round(safeSteps * INTERNAL_BEND_GROWTH_APPLY_UNTIL));
  const boundaries = [
    Math.max(1, Math.round(safeSteps / 3)),
    Math.max(1, Math.round(safeSteps * 2 / 3)),
    horizonStep,
  ].map((step) => Math.min(horizonStep, step));
  const due = new Set(boundaries);
  for (let step = INTERNAL_BEND_GROWTH_INTERVAL; step <= horizonStep; step += INTERNAL_BEND_GROWTH_INTERVAL) {
    due.add(step);
  }
  due.add(horizonStep);
  const cumulative = [0, INTERNAL_BEND_PHASE_GROWTH_SHARES[0],
    INTERNAL_BEND_PHASE_GROWTH_SHARES[0] + INTERNAL_BEND_PHASE_GROWTH_SHARES[1], 1];
  const edges = [0, ...boundaries];
  const events = [...due].filter((step) => step > 0).sort((a, b) => a - b).map((step) => {
    let segment = edges.length - 2;
    for (let i = 0; i < edges.length - 1; i += 1) {
      if (step <= edges[i + 1]) { segment = i; break; }
    }
    const span = Math.max(1, edges[segment + 1] - edges[segment]);
    const local = Math.max(0, Math.min(1, (step - edges[segment]) / span));
    const progress = cumulative[segment] + (cumulative[segment + 1] - cumulative[segment]) * local;
    const targetCount = step >= horizonStep
      ? finish
      : Math.max(start, Math.min(finish, Math.round(start + (finish - start) * progress)));
    return {step, targetCount, phase: segment + 1, terminal: step >= horizonStep};
  });
  return {events, horizonStep, initialCount: start, finalCount: finish};
}

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
  const finalCount = normalizeUiSplatCount(els.finalSplatCount.value, DEFAULT_FINAL_SPLATS, 14000);
  const initialCount = Math.min(finalCount, normalizeUiSplatCount(
    els.initialSplatCount.value,
    DEFAULT_INITIAL_SPLATS,
    finalCount,
  ));
  els.initialSplatCount.value = String(initialCount);
  els.finalSplatCount.value = String(finalCount);
  const failure = safetyFailure(computeBudgetFor(Number(els.trainSize.value), finalCount, steps), "start");
  if (failure) {setSafetyStop(failure); throw Error(`Internal bend safety guard: ${failure.reason}`);}
  let params = Image2SplatPaintInternalBend.initialize(state.image, initialCount, finalCount);
  const growthSchedule = buildInternalBendGrowthSchedule(steps, initialCount, finalCount);
  const growthByStep = new Map(growthSchedule.events.map((event) => [event.step, event]));
  const learningRates = {...selectedLearningRates(), opacity: 0};
  const previewRefresh = selectedPreviewRefresh();
  const periodicMetrics = periodicTrainingEvaluationEnabled();
  const metricInterval = Math.max(1, Math.min(DEFAULT_MAX_METRIC_INTERVAL,
    state.recommendation?.metricInterval || Math.floor(steps / 60)));
  const metrics = {
    algorithm: FLOW_SPLAT_FUSION_ALGORITHM_ID, flow_training_path: "internal-bend",
    backend: "webgpu", steps_requested: steps, steps_done: 0, num_gaussians: initialCount,
    initial_splats: initialCount, final_splats: finalCount, initial_param_snapshot: snapshotParams(params),
    started_at: new Date().toISOString(), stopped: false, params_revision: 0,
    preview_frames: 0, preview_refresh: previewRefresh, tile_retry_steps: 0,
    initial_ssim: null, initial_global_ssim: null, initial_psnr: null,
    fusion_events: emptyFusionEvents(), learning_rates: learningRates,
    density_gpu_ms: 0, densify_events: [],
    growth_schedule: {mode: "internal-bend-progressive", interval: INTERNAL_BEND_GROWTH_INTERVAL,
      apply_until: INTERNAL_BEND_GROWTH_APPLY_UNTIL, phase_shares: [20, 40, 40],
      event_steps: growthSchedule.events.map((event) => event.step), cap_reached_step: initialCount >= finalCount ? 0 : null},
    internal_bend: {version: 3, fixed_count: false, progressive_count: true,
      initial_count: initialCount, max_count: finalCount, backcoat: params.flowBackcoatCount,
      phase_growth_shares: [20, 40, 40], seed: 20260831, fusion: false},
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
    await awaitTrainingRun(run, renderer.uploadTrainState(state.image, params, finalCount));
    await awaitTrainingRun(run, renderer.prepareFlowBirthLinks(state.image, params));
    await updatePreview(0, false, {}, run);
    metrics.initial_ssim = metrics.latest_ssim;
    metrics.initial_global_ssim = metrics.latest_global_ssim;
    metrics.initial_psnr = metrics.latest_psnr;
    let retries = 0;
    for (let step = 1; step <= steps && !state.stopRequested;) {
      while (state.paused && !state.stopRequested) await awaitTrainingRun(run, nextFrame());
      if (state.stopRequested) break;
      const growthEvent = growthByStep.get(step);
      if (growthEvent?.targetCount > params.count) {
        const before = params.count;
        const growthStarted = performance.now();
        await refreshTrainingResidualSignal(step, "internal-bend-growth", run);
        const growth = await awaitTrainingRun(run, renderer.growExperimentalGpu(
          state.image,
          params,
          growthEvent.targetCount,
          step,
          steps,
          {forceZeroMassFallback: growthEvent.terminal},
        ));
        const gpuMs = performance.now() - growthStarted;
        metrics.density_gpu_ms += gpuMs;
        if (!growth) throw Error("Internal bend GPU growth failed");
        if (growth.grown) {
          params = Image2SplatPaintInternalBend.growParams(params, growth.count);
          state.params = params;
          updateTrainingRunOwnership(run, {params});
          metrics.num_gaussians = params.count;
          if (params.count >= finalCount && metrics.growth_schedule.cap_reached_step === null) {
            metrics.growth_schedule.cap_reached_step = step;
          }
          setTrainingMessage(`Training internal bend: P${growthEvent.phase}, ${params.count.toLocaleString()} / ${finalCount.toLocaleString()} Splats...`);
        }
        metrics.densify_events.push({step, phase: growthEvent.phase, count_before: before,
          requested_count: growthEvent.targetCount, actual_count: params.count,
          added: params.count - before, terminal: growthEvent.terminal,
          candidate_mass: Number.isFinite(growth.candidateMass) ? growth.candidateMass : null,
          zero_mass_fallback: Boolean(growth.zeroMassFallback), gpu_ms: gpuMs});
      }
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
    if (!metrics.stopped && params.count !== finalCount) {
      throw Error(`Internal bend growth did not reach Max splats: ${params.count} / ${finalCount}`);
    }
    metrics.final_splats = params.count;
    metrics.internal_bend.final_count = params.count;
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
