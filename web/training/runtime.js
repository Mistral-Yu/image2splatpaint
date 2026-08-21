function syncRuntimeMetrics(learningRates, previewRefresh) {
  if (!state.metrics) return;
  const growth = phase39Variants();
  const phase40 = phase40Variants();
  const runStageAware = Boolean(state.metrics.growth_schedule?.stage_aware);
  state.metrics.preview_refresh = previewRefresh;
  state.metrics.runtime_settings_revision = state.runtimeSettingsRevision;
  state.metrics.growth_schedule.percentage = growth.growthFraction * 100;
  state.metrics.growth_schedule.mode = runStageAware
    ? "stage-aware-percentage-cap"
    : "threshold-percentage-cap";
  state.metrics.growth_schedule.stage_aware = runStageAware;
  state.metrics.growth_schedule.signal_threshold = growth.growthSignalThreshold;
  state.metrics.learning_rates = {
    ...state.metrics.learning_rates,
    position: learningRates.position,
    color: learningRates.color,
    opacity: learningRates.opacity,
    scale: learningRates.scaleParam,
    rotation: learningRates.rotation,
    thetaAlign: learningRates.thetaAlign,
    maxAnisotropy: learningRates.maxAnisotropy,
    maxPlanarScale: learningRates.maxPlanarScale,
    boundarySigma: learningRates.boundarySigma,
    adaptiveDetail: learningRates.adaptiveDetail,
    detailCoherence: learningRates.detailCoherence,
  };
  state.metrics.alpha_loss_weight = phase40.alphaLossWeight;
  state.metrics.boundary_sigma = learningRates.boundarySigma;
  if (state.params) state.params.boundarySigma = learningRates.boundarySigma;
  const data = document.documentElement.dataset;
  data.runtimeAppliedRevision = String(state.runtimeSettingsRevision);
  data.runtimeAppliedPosition = String(learningRates.position);
  data.runtimeAppliedMaxPlanarScale = String(learningRates.maxPlanarScale);
  data.runtimeAppliedBoundarySigma = String(learningRates.boundarySigma);
  data.runtimeAppliedAlphaLossWeight = String(state.metrics.alpha_loss_weight);
  data.runtimeAppliedPreviewRefresh = previewRefresh;
}

async function presentTrainingPreview(step, run = null) {
  assertTrainingRun(run);
  if (layerTelemetryEnabled()) await awaitTrainingRun(run, recordLayerTelemetry(step, run));
  const renderer = state.webgpu.renderer;
  const virtualCameraSample = renderer?.lastTrainStats?.virtual_camera_sample;
  if (virtualCameraSample?.kind === "virtual") {
    const trainingStage = renderer.lastTrainStats?.training_stage;
    const previewImage = trainingStage === "coarse"
      ? renderer.trainState?.coarseImage || state.image
      : trainingStage === "mid"
        ? renderer.trainState?.midImage || state.image
        : state.image;
    if (els.tileCullingToggle.checked) {
      await awaitTrainingRun(run, renderer.prepareTileLists(previewImage, state.params, { sync: true }));
    }
    await awaitTrainingRun(run, renderer.refreshRenderState(previewImage, state.params));
    state.metrics.virtual_camera_sampling.preview_front_restores += 1;
  }
  let presented = false;
  if (els.outsidePreviewToggle.checked && !state.running && state.webgpu.renderer) {
    const buffers = state.webgpu.renderer.currentTrainBuffers(state.params);
    await awaitTrainingRun(run, state.webgpu.renderer.render(state.image, state.params, buffers));
    presented = true;
  } else {
    state.previewPadding = previewPaddingSpec(state.image, state.params, false);
    presented = renderer?.presentTrainState(state.image);
  }
  if (!presented) return false;
  if (state.previewMode === "splats") showCanvas("gpu");
  state.metrics.preview_frames += 1;
  state.metrics.last_preview_step = step;
  els.stepText.textContent = `${step} / ${state.metrics.steps_requested}`;
  publishState();
  return true;
}

async function resolveTileOverflowRetry(parameterHashBefore = null, run = null) {
  assertTrainingRun(run);
  const renderer = state.webgpu.renderer;
  if (!els.tileCullingToggle.checked || !renderer?.trainState?.tileReady) return 0;
  const tileCounters = await awaitTrainingRun(run, renderer.readTileCounters());
  if (!tileCounters) return 0;
  state.metrics.tile_counters = tileCounters;
  const noopSteps = Math.max(0, Math.floor(Number(tileCounters.noop_steps) || 0));
  if (tileCounters.overflow === 0 && noopSteps === 0) return 0;

  if (parameterHashBefore !== null) {
    const parameterHashAfter = await awaitTrainingRun(run, renderer.hashTrainParameters(state.params));
    const hashMatches = parameterHashAfter === parameterHashBefore;
    state.metrics.tile_retry_parameter_hash = {
      before: parameterHashBefore,
      after: parameterHashAfter,
      matches: hashMatches,
    };
    if (!hashMatches) {
      throw new Error(`tile overflow mutated parameters before retry: ${parameterHashBefore}/${parameterHashAfter}`);
    }
  }

  const previousCapacity = tileCounters.capacity;
  const expanded = tileCounters.overflow > 0
    ? await awaitTrainingRun(run, renderer.growTileIndexCapacity(tileCounters.total))
    : true;
  if (!expanded) {
    const rec = state.recommendation || updateMemoryRecommendation();
    const failure = { context: "tile-culling", reason: "safety_stop_tile_index_overflow", rec };
    state.metrics.stopped = true;
    state.metrics.safety_stop = {
      reason: failure.reason,
      context: failure.context,
      tile_indices: tileCounters.total,
      tile_capacity: tileCounters.capacity,
      tile_overflow: tileCounters.overflow,
      estimated_mb: rec.estimatedMB,
      budget_mb: rec.budgetMB,
    };
    setSafetyStop(failure);
    const error = new Error(`tile index capacity unavailable: ${tileCounters.total}/${tileCounters.capacity}`);
    error.safetyStop = true;
    throw error;
  }

  const retrySteps = Math.max(1, noopSteps);
  await awaitTrainingRun(run, renderer.clearTileNoopCounter());
  state.metrics.tile_retry_steps += retrySteps;
  state.metrics.tile_retry_events.push({
    after_step: state.metrics.steps_done,
    retry_steps: retrySteps,
    required_indices: tileCounters.total,
    previous_capacity: previousCapacity,
    next_capacity: renderer.trainState.tileIndexCapacity,
  });
  log(`tile overflow no-op: retrying ${retrySteps} iteration(s) after capacity ${previousCapacity} -> ${renderer.trainState.tileIndexCapacity}`);
  eventLog(`tile overflow retried ${retrySteps} iteration(s)`);
  return retrySteps;
}

function densityMetricSnapshot(metrics = null) {
  const coverage = metrics?.coverage || state.metrics?.coverage_stats;
  const regional = metrics?.regionalSsim || state.metrics?.latest_regional_ssim || state.metrics?.final_regional_ssim;
  return {
    step: metrics ? (state.metrics?.steps_done ?? 0) : (coverage?.step ?? state.metrics?.last_preview_step ?? 0),
    psnr: Number.isFinite(metrics?.psnr) ? metrics.psnr : state.metrics?.latest_psnr ?? state.metrics?.final_psnr ?? null,
    global_ssim: Number.isFinite(metrics?.ssim) ? metrics.ssim : state.metrics?.latest_global_ssim ?? state.metrics?.final_global_ssim ?? null,
    local_p10: Number.isFinite(regional?.p10) ? regional.p10 : regional?.p10 ?? null,
    alpha_ssim: Number.isFinite(metrics?.alphaSsim) ? metrics.alphaSsim : state.metrics?.latest_alpha_ssim ?? state.metrics?.final_alpha_ssim ?? null,
    alpha_l1: Number.isFinite(metrics?.alphaL1) ? metrics.alphaL1 : state.metrics?.latest_alpha_l1 ?? state.metrics?.final_alpha_l1 ?? null,
    background_exposure_ratio: Number.isFinite(coverage?.background_exposure_ratio)
      ? coverage.background_exposure_ratio
      : null,
  };
}

async function refreshTrainingResidualSignal(step, reason, run = null) {
  assertTrainingRun(run);
  const renderer = state.webgpu.renderer;
  if (!renderer?.trainState || !state.image || !state.params) {
    throw new Error("Training residual refresh requires active WebGPU state.");
  }
  const started = performance.now();
  const report = await awaitTrainingRun(run, renderer.refreshTrainingResidualMap(
    state.image,
    state.params,
    { step, reason },
  ));
  const telemetry = state.metrics?.training_residual_map;
  if (telemetry) {
    telemetry.refresh_count += 1;
    if (reason === "growth") telemetry.growth_refreshes += 1;
    if (reason === "relocation") telemetry.relocation_refreshes += 1;
    telemetry.last_step = step;
    telemetry.last_reason = reason;
    telemetry.last_wall_ms = performance.now() - started;
    if (telemetry.events.length < 160) {
      telemetry.events.push({
        ...report,
        wall_ms: telemetry.last_wall_ms,
      });
    }
  }
  return report;
}

function linkDensityEventMetricSnapshot(step, metrics) {
  const snapshot = densityMetricSnapshot(metrics);
  for (const event of state.metrics?.densify_events || []) {
    if (event.step === step && !event.metrics_after) event.metrics_after = snapshot;
  }
}

async function applyCurrentVisibilityCompaction(step, steps, run = null) {
  assertTrainingRun(run);
  const renderer = state.webgpu.renderer;
  const params = state.params;
  if (!params?.opaqueLayered || state.metrics?.stopped || !renderer?.trainState) return false;
  if ((state.metrics?.current_visibility_compaction_events || []).some((event) => event.step === step)) {
    return false;
  }
  // Layer priority uses the current trained order. The visibility path skips
  // percentile summaries and the planner uses fixed-size depth bins, avoiding
  // the previous per-candidate object allocation and O(N log N) sort.
  await awaitTrainingRun(run, renderer.readTrainedColors(params));
  assertFiniteParams(params, "current-visibility-compaction-readback");
  const importanceData = await awaitTrainingRun(
    run,
    renderer.readImportanceData(params.count, { includeSummary: false }),
  );
  const plan = hardZeroContributionPlan(params, importanceData);
  const report = { ...plan };
  delete report.keepIndices;
  delete report.pruneIndices;
  report.step = step;
  report.phase = "pre-settle";
  report.policy = "current-visibility-hard-zero";
  report.visibility_window = {
    reset_step: renderer.trainState?.currentVisibilityWindow?.reset_step ?? step - 1,
    measured_steps: 1,
    signal: "accepted-pixels-and-sum-t-before-alpha",
    scope: "all-opaque-paint-layers",
  };
  report.metrics_evaluation = "deferred-to-final";
  if (!plan.applied) {
    state.metrics.current_visibility_compaction = report;
    state.metrics.current_visibility_compaction_events.push(report);
    return false;
  }
  const compactResult = await awaitTrainingRun(run, renderer.compactTrainStateGpu(plan.keepIndices));
  if (!compactResult.compacted) {
    report.applied = false;
    report.reason = compactResult.reason || "gpu-compaction-skipped";
    report.gpu_compaction_allocation = compactResult.allocation || null;
    state.metrics.current_visibility_compaction = report;
    state.metrics.current_visibility_compaction_events.push(report);
    return false;
  }
  state.params = compactSplatParams(params, plan.keepIndices);
  updateTrainingRunOwnership(run, { params: state.params });
  if (els.tileCullingToggle.checked) {
    await awaitTrainingRun(run, renderer.prepareTileLists(state.image, state.params, { sync: true }));
  }
  report.gpu_compaction_ms = compactResult.gpu_ms;
  report.gpu_transient_bytes = compactResult.transient_bytes || 0;
  report.gpu_compaction_allocation = compactResult.allocation || null;
  report.optimizer_state_preserved = true;
  report.params_compacted = true;
  state.metrics.current_visibility_compaction_removed_total += plan.removed;
  state.metrics.current_visibility_compaction = report;
  state.metrics.current_visibility_compaction_events.push(report);
  state.metrics.num_gaussians = state.params.count;
  state.metrics.params_revision = (state.metrics.params_revision || 0) + 1;
  state.metrics.cpu_mirror_step = step;
  state.metrics.cpu_mirror_count = state.params.count;
  return true;
}

async function applyCurrentContributionCompaction(step, steps, settings, run = null) {
  assertTrainingRun(run);
  const renderer = state.webgpu.renderer;
  const params = state.params;
  if (!settings?.enabled || state.metrics?.stopped || !renderer?.trainState) return false;
  if ((state.metrics?.current_contribution_compaction_events || []).some((event) => event.step === step)) {
    return false;
  }
  // The planner does not need CPU RGB, XY, or scale data: its input is the
  // current exact-backward contribution buffer.  Avoiding that readback keeps
  // this late event from causing the post-training UI stalls it is intended to
  // reduce.
  const layerMirrorCurrent = await awaitTrainingRun(
    run,
    renderer.readTrainedLayerOrder(params),
  );
  if (!layerMirrorCurrent) throw new Error("Current Contribution Compaction requires a current GPU layer mirror.");
  const importanceData = await awaitTrainingRun(
    run,
    renderer.readImportanceData(params.count, { includeSummary: false }),
  );
  const plan = currentContributionCompactionPlan(
    params,
    importanceData,
    settings,
  );
  const report = { ...plan };
  delete report.keepIndices;
  report.step = step;
  report.phase = "periodic-visibility-recovery";
  report.policy = "current-contribution-compaction-v2";
  report.algorithm = state.metrics?.algorithm || settings.algorithm;
  report.event_index = (state.metrics?.current_contribution_compaction_events?.length || 0) + 1;
  report.interval_steps = settings.intervalSteps;
  report.visibility_window = {
    reset_step: renderer.trainState?.currentVisibilityWindow?.reset_step ?? step - 1,
    measured_steps: settings.measurementWindowSteps,
    signal: "accepted-pixels-and-sum-t-before-alpha",
    scope: settings.virtual
      ? "all-active-splats-across-one-complete-virtual-camera-pool"
      : "all-active-splats-in-current-training-view",
  };
  report.layer_order_source = "current-gpu-transform";
  report.metrics_evaluation = "deferred-to-final";
  if (!plan.applied) {
    state.metrics.current_contribution_compaction = report;
    state.metrics.current_contribution_compaction_events.push(report);
    return false;
  }
  const compactResult = await awaitTrainingRun(run, renderer.compactTrainStateGpu(plan.keepIndices));
  if (!compactResult.compacted) {
    report.applied = false;
    report.reason = compactResult.reason || "gpu-compaction-skipped";
    report.gpu_compaction_allocation = compactResult.allocation || null;
    state.metrics.current_contribution_compaction = report;
    state.metrics.current_contribution_compaction_events.push(report);
    return false;
  }
  state.params = compactSplatParams(params, plan.keepIndices);
  updateTrainingRunOwnership(run, { params: state.params });
  if (els.tileCullingToggle.checked) {
    await awaitTrainingRun(run, renderer.prepareTileLists(state.image, state.params, { sync: true }));
  }
  report.gpu_compaction_ms = compactResult.gpu_ms;
  report.gpu_transient_bytes = compactResult.transient_bytes || 0;
  report.gpu_compaction_allocation = compactResult.allocation || null;
  report.optimizer_state_preserved = true;
  report.params_compacted = true;
  state.metrics.current_contribution_compaction_removed_total += plan.removed;
  state.metrics.current_contribution_compaction = report;
  state.metrics.current_contribution_compaction_events.push(report);
  state.metrics.num_gaussians = state.params.count;
  state.metrics.params_revision = (state.metrics.params_revision || 0) + 1;
  // GPU compaction did not read XY/scale/RGB back.  Keep this CPU mirror
  // explicitly stale so Stop/finalization must obtain the authoritative state.
  state.metrics.cpu_mirror_step = null;
  state.metrics.cpu_mirror_count = null;
  state.metrics.cpu_mirror_current = false;
  state.metrics.cpu_layer_mirror_step = step;
  return true;
}

async function updatePreview(step, final = false, { present = true, readOnlyPeriodic = false } = {}, run = null) {
  assertTrainingRun(run);
  const backend = selectedBackend();
  if (!backend.startsWith("webgpu")) throw new Error(`WebGPU required: ${state.webgpu.reason}`);
  const safety = safetyFailure(computeBudgetFor(Number(els.trainSize.value), state.params.count, state.metrics?.steps_requested || 1), "metrics");
  if (safety) {
    setSafetyStop(safety);
    throw new Error(`${safety.reason}: metrics/readback skipped before budget overflow`);
  }
  if (layerTelemetryEnabled()) await awaitTrainingRun(run, recordLayerTelemetry(step, run));
  const reusableMetricRender = Boolean(
    !final && state.webgpu.renderer?.canReuseMetricRender(state.image)
  );
  const reuseMetricRender = Boolean(
    reusableMetricRender &&
    (readOnlyPeriodic || performanceVariants().metricTileReuse)
  );
  if (readOnlyPeriodic && !reuseMetricRender && els.tileCullingToggle.checked) {
    await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(
      state.image,
      state.params,
      { sync: true },
    ));
  }
  if (!readOnlyPeriodic && els.tileCullingToggle.checked && state.webgpu.renderer?.trainState) {
    const includeTileDistribution = Boolean(
      performanceProfileRequested() &&
      performanceProfileLabels(step, state.metrics?.steps_requested || step).length > 0
    );
    if (!reuseMetricRender) {
      await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(state.image, state.params, { sync: true }));
    }
    let tileCounters = await awaitTrainingRun(run, state.webgpu.renderer.readTileCounters({ includeDistribution: includeTileDistribution }));
    const reserveRatio = tileCounters ? tileCounters.total / Math.max(1, tileCounters.capacity) : 0;
    const qaOverflowPending =
      qaTileOverflowFixtureEnabled() &&
      (state.metrics?.steps_done || 0) === 0 &&
      !state.metrics?.tile_retry_parameter_hash?.matches;
    const reserveLevel = reserveRatio >= 0.9 ? 90 : reserveRatio >= 0.8 ? 80 : reserveRatio >= 0.7 ? 70 : 0;
    if (reserveLevel > (state.webgpu.renderer.trainState.tileReserveLevel || 0)) {
      state.webgpu.renderer.trainState.tileReserveLevel = reserveLevel;
      eventLog(`tile reserve ${reserveLevel}% threshold: ${tileCounters.total}/${tileCounters.capacity}`);
    }
    const shouldExpandTileReserve = Boolean(
      !qaOverflowPending && tileCounters && (tileCounters.overflow > 0 || reserveRatio >= 0.8),
    );
    const expandedTileReserve = shouldExpandTileReserve
      ? await awaitTrainingRun(run, state.webgpu.renderer.growTileIndexCapacity(tileCounters.total, { proactive: tileCounters.overflow === 0 }))
      : false;
    if (expandedTileReserve) {
      const previousCapacity = tileCounters.capacity;
      await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(state.image, state.params, { sync: true }));
      tileCounters = await awaitTrainingRun(run, state.webgpu.renderer.readTileCounters({ includeDistribution: includeTileDistribution }));
      log(`tile index capacity expanded ${previousCapacity} -> ${tileCounters.capacity} for ${tileCounters.total} references`);
      eventLog(`tile index capacity expanded ${previousCapacity} -> ${tileCounters.capacity}`);
    }
    if (tileCounters) {
      state.metrics.tile_counters = tileCounters;
      if (!qaOverflowPending && (tileCounters.overflow > 0 || (reserveRatio >= 0.9 && shouldExpandTileReserve && !expandedTileReserve))) {
        const rec = state.recommendation || updateMemoryRecommendation();
        const failure = { context: "tile-culling", reason: "safety_stop_tile_index_overflow", rec };
        state.metrics.stopped = true;
        state.metrics.safety_stop = {
          reason: failure.reason,
          context: failure.context,
          tile_indices: tileCounters.total,
          tile_capacity: tileCounters.capacity,
          tile_overflow: tileCounters.overflow,
          estimated_mb: rec.estimatedMB,
          budget_mb: rec.budgetMB,
        };
        setSafetyStop(failure);
        const error = new Error(`tile index capacity unavailable: ${tileCounters.total}/${tileCounters.capacity}`);
        error.safetyStop = true;
        throw error;
      }
    }
  }
  let trainBuffers = state.webgpu.renderer?.currentTrainBuffers(state.params);
  if (final && state.webgpu.renderer?.trainState) {
    const cpuMirrorAlreadyCurrent =
      state.metrics?.cpu_mirror_step === step &&
      state.metrics?.cpu_mirror_count === state.params.count;
    if (!cpuMirrorAlreadyCurrent) {
      await awaitTrainingRun(run, state.webgpu.renderer.readTrainedColors(state.params));
    }
    assertFiniteParams(state.params, "final-readback");
    if (previewInvariantHashEnabled()) {
      state.metrics.final_parameter_hash = hashParams(state.params);
    }
    if (final) {
      state.metrics.thin_line_metrics = computeThinLineMetrics(state.image, state.params);
      state.metrics.tilt_risk = summarizeTiltRisk(state.params, state.image);
      state.metrics.virtual_depth_stats = virtualDepthDistribution(state.params);
      if (state.params.surfaceLayerPriorEnabled && state.metrics.surface_layer_prior) {
        state.metrics.surface_layer_prior.final_assignment =
          summarizeScaleBiasedSurfaceLayerSort(state.params);
      }
      state.metrics.final_readback_step = step;
    }
  }
  const cpuMirrorCurrent = final || step === 0;
  state.metrics.cpu_mirror_current = cpuMirrorCurrent;
  if (cpuMirrorCurrent) {
    state.metrics.param_delta = paramDeltaFromSnapshot(state.metrics.initial_param_snapshot, state.params);
  }
  let metrics = { loss: Number.NaN, ssim: Number.NaN };
  els.backendText.textContent = backend;
  try {
    // Padded preview is display-only and never runs while the optimizer owns the live buffers.
    const outsidePreviewActive = present && els.outsidePreviewToggle.checked && !state.running;
    if (outsidePreviewActive) {
      await awaitTrainingRun(run, state.webgpu.renderer.render(state.image, state.params, trainBuffers));
    }
    metrics = await awaitTrainingRun(run, state.webgpu.renderer.computeMetrics(
      state.image,
      state.params,
      trainBuffers,
      { reuseCurrentRender: reuseMetricRender },
    ));
    const previousMetricReuse = state.metrics.metric_tile_reuse || {};
    state.metrics.metric_tile_reuse = {
      requested: readOnlyPeriodic || performanceVariants().metricTileReuse,
      last_applied: Boolean(metrics.reusedRender),
      applied_count: (previousMetricReuse.applied_count || 0) + (metrics.reusedRender ? 1 : 0),
      fallback_count: (previousMetricReuse.fallback_count || 0) + (metrics.reusedRender ? 0 : 1),
      lag_iterations: metrics.reusedRender ? 1 : previousMetricReuse.lag_iterations || 0,
    };
    const restoreTrainingStage = !final
      ? state.webgpu.renderer.lastTrainStats?.training_stage
      : "full";
    const restoreStageImage = restoreTrainingStage === "coarse"
      ? state.webgpu.renderer.trainState?.coarseImage
      : restoreTrainingStage === "mid"
        ? state.webgpu.renderer.trainState?.midImage
        : null;
    if (restoreStageImage && !metrics.reusedRender) {
      if (els.tileCullingToggle.checked) {
        await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(restoreStageImage, state.params, { sync: true }));
      }
      await awaitTrainingRun(run, state.webgpu.renderer.refreshRenderState(restoreStageImage, state.params));
      state.metrics.preview_resolution_restores += 1;
    }
    if (present && !outsidePreviewActive) state.webgpu.renderer.presentTrainState(state.image);
    if (present && state.previewMode === "splats") showCanvas("gpu");
    if (present) {
      state.metrics.preview_frames += 1;
      state.metrics.last_preview_step = step;
    }
    const finiteMetrics = [
      metrics.loss,
      metrics.mse,
      metrics.psnr,
      metrics.alphaL1,
      metrics.alphaSsim,
      metrics.alphaObjective,
      metrics.objectiveLoss,
      metrics.ssim,
      metrics.windowedSsim,
      metrics.regionalSsim?.minimum,
      metrics.regionalSsim?.p10,
      metrics.coverage?.mean,
      metrics.coverage?.minimum,
    ];
    if (finiteMetrics.some((value) => !Number.isFinite(value))) {
      throw runtimeSafetyError("safety_stop_nonfinite_metrics", `metrics-step-${step}`);
    }
    if (final) {
      const brushContributionSettings = brushContributionDiagnosticsSettings(
        ALGORITHM_REGISTRY[state.metrics?.algorithm],
      );
      if (brushContributionSettings.enabled) {
        const report = await awaitTrainingRun(
          run,
          state.webgpu.renderer.computeBrushContributionDiagnostics(
            state.image,
            state.params,
            brushContributionSettings,
          ),
        );
        const repeatedReport = await awaitTrainingRun(
          run,
          state.webgpu.renderer.computeBrushContributionDiagnostics(
            state.image,
            state.params,
            brushContributionSettings,
          ),
        );
        const deterministicFields = [
          "valid_pixels",
          "flat_pixels",
          "flat_contributing_pixels",
          "high_neff_pixels",
          "mean_accepted_contributors",
          "mean_physical_effective_contributors",
          "mean_renyi2_entropy",
          "mean_composited_alpha",
        ];
        report.repeat_deltas = Object.fromEntries(deterministicFields.map((key) => [
          key,
          Number(repeatedReport?.[key]) - Number(report?.[key]),
        ]));
        report.repeat_deterministic = deterministicFields.every(
          (key) => Object.is(repeatedReport?.[key], report?.[key]),
        );
        report.repeat_parameter_hash_matches = Boolean(
          report.parameter_hash_matches && repeatedReport?.parameter_hash_matches &&
          repeatedReport?.parameter_hash_after === report.parameter_hash_after,
        );
        const coverageMean = Number(metrics.coverage?.mean);
        if (report && Number.isFinite(coverageMean)) {
          report.final_metric_coverage_mean = coverageMean;
          report.final_metric_alpha_mean_delta = report.mean_composited_alpha - coverageMean;
        }
        state.metrics.brush_contribution_diagnostics = report;
      }
      if (state.metrics.virtual_camera_sampling?.enabled) {
        setTrainingMessage(`Evaluating ${state.metrics.virtual_camera_sampling.virtual_camera_count} virtual camera teachers on WebGPU...`);
        state.metrics.virtual_camera_evaluation = await awaitTrainingRun(run, state.webgpu.renderer.computeVirtualCameraEvaluation(
          state.image,
          state.params,
          metrics,
        ));
      }
      const explicitFinalAudit = finalRenderAuditEnabled();
      state.metrics.overlap_diagnostics =
        !algorithmUsesPaintKernel() &&
        (explicitFinalAudit || layerEfficiencyDiagnosticsRequested())
        ? await awaitTrainingRun(run, state.webgpu.renderer.computeOverlapDiagnostics(state.image, state.params))
        : null;
      if (!algorithmUsesPaintKernel() && layerEfficiencyDiagnosticsRequested()) {
        if (els.tileCullingToggle.checked) {
          await awaitTrainingRun(run, state.webgpu.renderer.prepareTileLists(state.image, state.params, { sync: true }));
        }
        const tileDiagnostics = await awaitTrainingRun(run, state.webgpu.renderer.readLayerTileDiagnostics(state.params));
        state.metrics.layer_efficiency = summarizeLayerEfficiency(
          state.params,
          state.metrics.overlap_diagnostics?.hidden_rgb_attribution || null,
          tileDiagnostics,
          state.layerEfficiencyCheckpoints,
          state.metrics.initial_splats,
        );
      }
      if (QA_RUNTIME_ENABLED && state.metrics.virtual_camera_sampling?.enabled) {
        state.metrics.oblique_overlap_diagnostics = await awaitTrainingRun(run, state.webgpu.renderer.computeObliqueDiagnostics(state.image, state.params));
      }
      // Export is result-bound, so product finalization verifies the training
      // surface against the standalone renderer once. This is the only product
      // full-frame quality readback beyond final metrics.
      const trainingFrame = await awaitTrainingRun(run, state.webgpu.renderer.capturePresentedStateRgba());
      if (!trainingFrame || trainingFrame.width !== state.image.width || trainingFrame.height !== state.image.height) {
        throw new Error("Final training RGBA readback has the wrong resolution.");
      }
      const standaloneRgba = await awaitTrainingRun(run, state.webgpu.renderer.captureFrameRgba(
        state.image,
        state.params,
        trainBuffers,
      ));
      state.metrics.render_surface_parity = {
        ...displayRgbaParity(trainingFrame.rgba, standaloneRgba),
        source: "training-pixel-state-vs-standalone-rgba",
        width: trainingFrame.width,
        height: trainingFrame.height,
      };
      if (explicitFinalAudit) {
        const auditParams = state.params;
        if (auditParams.layerAwareAccumulationEnabled) {
          auditParams.layerAwareAccumulationEnabled = false;
          try {
            const ordinaryAlphaRgba = await awaitTrainingRun(run, state.webgpu.renderer.captureFrameRgba(
              state.image,
              auditParams,
              trainBuffers,
            ));
            state.metrics.layer_aware_render_parity = {
              ...displayRgbaParity(ordinaryAlphaRgba, standaloneRgba),
              source: "same-params-standard-alpha-off-vs-layer-aware-on",
            };
          } finally {
            auditParams.layerAwareAccumulationEnabled = true;
          }
        } else {
          state.metrics.layer_aware_render_parity = null;
        }
        state.metrics.color_space_audit = trainingColorSpaceAudit(
          state.image,
          state.webgpu.renderer.trainState,
          trainingFrame.rgba,
          standaloneRgba,
        );
        if (!state.metrics.importance_stats) {
          state.metrics.importance_stats = await awaitTrainingRun(run, state.webgpu.renderer.readImportanceSummary(state.params.count));
        }
        if (state.metrics.importance_stats?.nonfinite_count > 0) {
          throw runtimeSafetyError("safety_stop_nonfinite_importance", "final-importance-readback", {
            nonfinite_stats: state.metrics.importance_stats.nonfinite_count,
          });
        }
      } else {
        state.metrics.layer_aware_render_parity = null;
        state.metrics.color_space_audit = {
          contract: "front and virtual teachers use the same sRGB signal values",
          skipped: true,
          reason: "extended-color-audit-is-qa-only",
        };
      }
    }
    const densityCounters = await awaitTrainingRun(run, state.webgpu.renderer.readDensityCounters());
    if (densityCounters) {
      const previous = state.metrics.last_density_counters || emptyFusionEvents();
      const relocated = densityCounters.mcmc_teleport - (previous.mcmc_teleport || 0);
      state.metrics.density_counters = densityCounters;
      state.metrics.last_density_counters = { ...densityCounters };
      state.metrics.fusion_events.adc_duplicate = densityCounters.adc_duplicate;
      state.metrics.fusion_events.adc_split = densityCounters.adc_split;
      state.metrics.fusion_events.adc_recycle = densityCounters.adc_recycle;
      state.metrics.fusion_events.mcmc_teleport = densityCounters.mcmc_teleport;
      state.metrics.fusion_events.mcmc_reseed = densityCounters.mcmc_reseed + densityCounters.mcmc_teleport;
      state.metrics.fusion_events.inactive_reused = densityCounters.inactive_reused;
      state.metrics.fusion_events.opacity_reset = densityCounters.opacity_reset;
      state.metrics.fusion_events.prune = densityCounters.prune;
      state.metrics.fusion_events.importance_protected = densityCounters.importance_protected;
      state.metrics.fusion_events.adc_eligible = densityCounters.adc_eligible;
      state.metrics.fusion_events.adc_fallback = densityCounters.adc_fallback;
      state.metrics.fusion_events.structure_guided = densityCounters.structure_guided;
      state.metrics.fusion_events.nonfinite_stats = densityCounters.nonfinite_stats;
      state.metrics.fusion_events.adc_low_to_high = densityCounters.adc_low_to_high;
      state.metrics.fusion_events.adc_high_to_low = densityCounters.adc_high_to_low;
      state.metrics.fusion_events.adc_same_band = densityCounters.adc_same_band;
      state.metrics.fusion_events.source_claim_conflicts = densityCounters.source_claim_conflicts;
      state.metrics.fusion_events.source_claims = densityCounters.source_claims;
      state.metrics.fusion_events.tilt_risk_candidates = densityCounters.tilt_risk_candidates;
      state.metrics.fusion_events.tilt_true_splits = densityCounters.tilt_true_splits;
      state.metrics.fusion_events.tilt_opacity_saturations = densityCounters.tilt_opacity_saturations;
      state.metrics.fusion_events.paint_outlier_recycle = densityCounters.paint_outlier_recycle;
      state.metrics.fusion_events.paint_outlier_recolor = densityCounters.paint_outlier_recolor;
      state.metrics.fusion_events.paint_outlier_trim = densityCounters.paint_outlier_trim;
      state.metrics.fusion_events.surface_layer_candidates = densityCounters.surface_layer_candidates;
      state.metrics.fusion_events.surface_layer_promotions = densityCounters.surface_layer_promotions;
      state.metrics.fusion_events.harmful_rectangle_candidate_selections =
        densityCounters.harmful_rectangle_candidate_selections;
      state.metrics.fusion_events.harmful_rectangle_front_oversized_selections =
        densityCounters.harmful_rectangle_front_oversized_selections;
      state.metrics.fusion_events.harmful_rectangle_high_contribution_selections =
        densityCounters.harmful_rectangle_high_contribution_selections;
      state.metrics.fusion_events.harmful_rectangle_high_deviation_selections =
        densityCounters.harmful_rectangle_high_deviation_selections;
      state.metrics.fusion_events.harmful_rectangle_parent_replacements =
        densityCounters.harmful_rectangle_parent_replacements;
      state.metrics.fusion_events.harmful_rectangle_children_created =
        densityCounters.harmful_rectangle_children_created;
      if (state.metrics.front_footprint_refinement_v2) {
        // Growth events reset the shared density counters. The report above is
        // cumulative; a later relocation/finalization read must not erase it.
        state.metrics.front_footprint_refinement_v2.last_density_counters = {
          candidate_selections: densityCounters.harmful_rectangle_candidate_selections,
          parent_replacements: densityCounters.harmful_rectangle_parent_replacements,
          children_created: densityCounters.harmful_rectangle_children_created,
        };
      }
      if (densityCounters.nonfinite_stats > 0) {
        throw runtimeSafetyError("safety_stop_nonfinite_density", `density-step-${step}`, {
          nonfinite_stats: densityCounters.nonfinite_stats,
        });
      }
      if (relocated > 0) {
        const event = { step, moved: relocated, cumulative: densityCounters.mcmc_teleport };
        if (state.metrics.webgpu_relocation_events.length < 96) state.metrics.webgpu_relocation_events.push(event);
        if (state.metrics.webgpu_refine_events.length < 96) state.metrics.webgpu_refine_events.push(event);
      }
    }
    state.lastGpuLoss = metrics.loss;
    state.metrics.webgpu_compute_loss = true;
    state.metrics.last_gpu_loss = metrics.loss;
    state.metrics.last_alpha_l1 = metrics.alphaL1;
    state.metrics.last_alpha_ssim = metrics.alphaSsim;
    state.metrics.last_objective_loss = metrics.objectiveLoss;
    state.metrics.webgpu_loss_stats = state.webgpu.renderer.lastLossStats;
    state.metrics.webgpu_train_executed = Boolean(state.webgpu.renderer.lastTrainStats);
    state.metrics.webgpu_train_update = Boolean(state.webgpu.renderer.lastTrainStats?.updated);
    state.metrics.webgpu_train_stats = state.webgpu.renderer.lastTrainStats;
    state.metrics.stage_profile = state.webgpu.renderer.trainState?.stageProfile?.map((sample) => ({ ...sample })) || [];
    state.metrics.stage_profile_backend = state.webgpu.profile?.timing_backend || "off";
    state.metrics.scheduling_profile = summarizeTrainingScheduling(
      state.metrics.stage_profile,
      state.metrics.performance_trace,
    );
  } catch (error) {
    if (error.trainingRunCancelled || (run && !ownsTrainingRun(run))) throw error;
    if (error.safetyStop) throw error;
    state.lastGpuLoss = null;
    state.metrics.webgpu_compute_loss = false;
    state.metrics.webgpu_compute_error = error.message;
    throw new Error(`WebGPU preview/metrics failed: ${error.message}`);
  }
  state.metrics.losses.push(metrics.loss);
  state.metrics.rgb_mse.push(metrics.mse);
  state.metrics.psnr.push(metrics.psnr);
  state.metrics.alpha_losses.push(metrics.alphaL1);
  state.metrics.alpha_ssim.push(metrics.alphaSsim);
  state.metrics.objective_losses.push(metrics.objectiveLoss);
  state.metrics.ssim.push(metrics.windowedSsim);
  state.metrics.global_ssim.push(metrics.ssim);
  state.metrics.windowed_ssim.push(metrics.windowedSsim);
  state.metrics.regional_ssim_p10.push(metrics.regionalSsim.p10);
  Object.assign(state.metrics, {
    latest_evaluation_step: step,
    latest_l1: metrics.loss,
    latest_rgb_mse: metrics.mse,
    latest_psnr: metrics.psnr,
    latest_alpha_l1: metrics.alphaL1,
    latest_alpha_ssim: metrics.alphaSsim,
    latest_alpha_objective: metrics.alphaObjective,
    latest_objective_loss: metrics.objectiveLoss,
    latest_ssim: metrics.windowedSsim,
    latest_global_ssim: metrics.ssim,
    latest_windowed_ssim: metrics.windowedSsim,
    latest_regional_ssim: metrics.regionalSsim,
    latest_high_frequency: metrics.highFrequency,
  });
  if (final) {
    Object.assign(state.metrics, {
      final_evaluation_step: step,
      final_metrics_complete: true,
      final_l1: metrics.loss,
      final_rgb_mse: metrics.mse,
      final_psnr: metrics.psnr,
      final_alpha_l1: metrics.alphaL1,
      final_alpha_ssim: metrics.alphaSsim,
      final_alpha_objective: metrics.alphaObjective,
      final_objective_loss: metrics.objectiveLoss,
      final_ssim: metrics.windowedSsim,
      final_global_ssim: metrics.ssim,
      final_windowed_ssim: metrics.windowedSsim,
      final_regional_ssim: metrics.regionalSsim,
      final_high_frequency: metrics.highFrequency,
    });
  }
  state.metrics.coverage_stats = metrics.coverage ? { ...metrics.coverage, step } : null;
  state.metrics.coverage_revision = state.metrics.params_revision ?? 0;
  linkDensityEventMetricSnapshot(step, metrics);
  if (state.metrics.initial_ssim !== null) {
    const delta = metrics.windowedSsim - state.metrics.initial_ssim;
    state.metrics.ssim_trend = delta > 0.0005 ? "up" : delta < -0.0005 ? "down" : "flat";
  }
  if (state.metrics.initial_global_ssim !== null) {
    const delta = metrics.ssim - state.metrics.initial_global_ssim;
    state.metrics.global_ssim_trend = delta > 0.0005 ? "up" : delta < -0.0005 ? "down" : "flat";
  }
  if (state.metrics.initial_psnr !== null) {
    const delta = metrics.psnr - state.metrics.initial_psnr;
    state.metrics.psnr_trend = delta > 0.05 ? "up" : delta < -0.05 ? "down" : "flat";
  }
  const boundary = cpuMirrorCurrent ? boundaryLeakStats(state.params) : null;
  const outsideRender = cpuMirrorCurrent ? outsideRenderFootprintStats(state.params) : null;
  const shape = cpuMirrorCurrent ? splatShapeStats(state.params, state.image) : null;
  state.metrics.boundary_sigma = state.params.boundarySigma ?? selectedBoundarySigma();
  if (cpuMirrorCurrent) {
    state.metrics.boundary_leak_count = boundary.count;
    state.metrics.boundary_max_leak = boundary.maxLeak;
    state.metrics.outside_render_splat_count = outsideRender.count;
    state.metrics.outside_render_max_extent = outsideRender.maxLeak;
    state.metrics.shape_stats = shape;
    if (final) {
      state.metrics.source_detail_splat_distribution = sourceDetailSplatDistribution(
        state.image,
        state.params,
      );
    }
    state.metrics.scale_histogram = shape?.scale_histogram || null;
    state.metrics.tiny_splat_count = shape?.tiny_splat_count ?? null;
    state.metrics.tiny_splat_ratio = shape?.tiny_splat_ratio ?? null;
    state.metrics.boundary_tiny_splat_count = shape?.boundary_tiny_splat_count ?? null;
    state.metrics.boundary_tiny_splat_ratio = shape?.boundary_tiny_splat_ratio ?? null;
    state.metrics.interior_tiny_splat_count = shape?.interior_tiny_splat_count ?? null;
    state.metrics.interior_tiny_splat_ratio = shape?.interior_tiny_splat_ratio ?? null;
    state.metrics.anisotropy_ratio = shape
      ? { mean: shape.anisotropy_ratio_mean, max: shape.anisotropy_ratio_max, elongated_count: shape.elongated_splat_count }
      : null;
  }
  state.metrics.steps_done = step;
  state.metrics.num_gaussians = state.params.count;
  state.metrics.loss_backend = "webgpu-compute";
  if (state.metrics.trend_checkpoints.length < 256) {
    state.metrics.trend_checkpoints.push({
      step,
      loss: metrics.loss,
      rgb_mse: metrics.mse,
      psnr: metrics.psnr,
      alpha_l1: metrics.alphaL1,
      alpha_ssim: metrics.alphaSsim,
      alpha_objective: metrics.alphaObjective,
      objective_loss: metrics.objectiveLoss,
      ssim: metrics.windowedSsim,
      global_ssim: metrics.ssim,
      regional_ssim: {
        minimum: metrics.regionalSsim.minimum,
        p10: metrics.regionalSsim.p10,
        median: metrics.regionalSsim.median,
        mean: metrics.regionalSsim.mean,
        worst_region: metrics.regionalSsim.worst_region,
      },
      high_frequency: metrics.highFrequency,
      boundary_leak_count: boundary?.count ?? null,
      boundary_max_leak: boundary?.maxLeak ?? null,
      outside_render_splat_count: outsideRender?.count ?? null,
      tiny_splat_count: shape?.tiny_splat_count ?? null,
      anisotropy_ratio_max: shape?.anisotropy_ratio_max ?? null,
      coverage_under_ratio: metrics.coverage?.under_ratio ?? null,
      background_exposure_count: metrics.coverage?.background_exposure_count ?? null,
      gaussians: state.params.count,
    });
  }
  els.stepText.textContent = `${step} / ${state.metrics.steps_requested}`;
  els.lossText.textContent = metrics.loss.toFixed(6);
  syncDisplayedSsimMetrics();
  els.boundaryText.textContent = boundary ? `${boundary.count} / ${boundary.maxLeak.toFixed(6)}` : "-";
  const coveragePairReady = Boolean(boundary && Number.isFinite(metrics.coverage?.background_exposure_count));
  els.coverageText.textContent = coveragePairReady
    ? `${(metrics.coverage.background_exposure_ratio * 100).toFixed(2)}% / ${outsideRender.count.toLocaleString()}`
    : "- / -";
  renderSplatInspector();
  publishState();
  if (final) {
    const label = state.metrics?.stopped ? "stopped" : "finished";
    const worst = metrics.regionalSsim.worst_region;
    log(`${label} loss=${metrics.loss.toFixed(6)} psnr=${metrics.psnr.toFixed(2)}dB alpha_l1=${metrics.alphaL1.toFixed(6)} alpha_ssim=${metrics.alphaSsim.toFixed(6)} objective=${metrics.objectiveLoss.toFixed(6)} global_ssim=${metrics.ssim.toFixed(6)} windowed_ssim=${metrics.windowedSsim.toFixed(6)} local_p10=${metrics.regionalSsim.p10.toFixed(6)} worst_region=${worst?.column ?? "-"},${worst?.row ?? "-"}`);
    const virtualEvaluation = state.metrics.virtual_camera_evaluation;
    if (virtualEvaluation) {
      const virtual = virtualEvaluation.virtual_views;
      const allViews = virtualEvaluation.all_views;
      log(`virtual teacher evaluation cameras=${virtual.camera_count} virtual_psnr=${virtual.rgb_psnr_macro?.toFixed(2) ?? "-"}dB virtual_ssim=${virtual.rgb_ssim_macro?.toFixed(6) ?? "-"} virtual_p10=${virtual.rgb_ssim_p10?.toFixed(6) ?? "-"} all_view_psnr=${allViews.rgb_psnr_macro?.toFixed(2) ?? "-"}dB all_view_ssim=${allViews.rgb_ssim_macro?.toFixed(6) ?? "-"} all_view_p10=${allViews.rgb_ssim_p10?.toFixed(6) ?? "-"}`);
    }
  }
  return metrics.loss;
}
