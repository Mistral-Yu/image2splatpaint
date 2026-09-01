function initialDepthOrder(count) {
  const values = new Float32Array(Math.max(0, count));
  const denominator = Math.max(1, count - 1);
  for (let i = 0; i < count; i += 1) values[i] = 1 - i / denominator;
  return values;
}

function packedLayerOrder(packedTag) {
  const fraction = packedTag - Math.floor(packedTag);
  return Math.max(0, Math.min(1, Math.min(fraction, LAYER_CODE_RANGE) / LAYER_CODE_RANGE));
}

function summarizeScaleBiasedSurfaceLayerSort(params) {
  const count = Math.max(0, Math.round(Number(params?.count) || 0));
  const layers = Math.round(clampNumber(
    params?.surfaceLayerPriorLayers,
    MIN_DISCRETE_LAYER_COUNT,
    MAX_DISCRETE_LAYER_COUNT,
    DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
  ));
  const actualCounts = Array(layers).fill(0);
  const expectedCounts = Array(layers).fill(0);
  const minimumArea = MIN_SPLAT_SCALE ** 2;
  const maximumAxis = Math.max(
    PHASE_ONE_MAX_PLANAR_SCALE,
    clampNumber(params?.maxPlanarScale, MIN_SPLAT_SCALE, 1, DEFAULT_MAX_PLANAR_SCALE),
  );
  const maximumArea = Math.max(minimumArea * 1.0001, maximumAxis ** 2);
  let matched = 0;
  for (let index = 0; index < count; index += 1) {
    const area = clampNumber(
      params.scale[index * 2] * params.scale[index * 2 + 1],
      minimumArea,
      maximumArea,
      minimumArea,
    );
    const sizeRank = clampNumber(
      (Math.log(area) - Math.log(minimumArea)) /
        Math.max(1e-6, Math.log(maximumArea) - Math.log(minimumArea)),
      0,
      1,
      0,
    );
    const expected = Math.min(layers - 1, Math.floor(Math.min(0.999999, 1 - sizeRank) * layers));
    const actual = Math.min(layers - 1, Math.floor(
      clampNumber(params.depthOrder?.[index], 0, 0.999999, 0) * layers,
    ));
    expectedCounts[expected] += 1;
    actualCounts[actual] += 1;
    if (actual === expected) matched += 1;
  }
  return {
    checked_splats: count,
    matching_splats: matched,
    match_ratio: matched / Math.max(1, count),
    expected_layer_counts: expectedCounts,
    actual_layer_counts: actualCounts,
  };
}

function boundedVirtualDepth(params, index) {
  if (!params?.virtualDepthEnabled) return 0;
  const thickness = Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS;
  return thickness * Math.tanh(params.virtualDepth?.[index] || 0);
}

function finalDiagnosticSampling(count, limit = MAX_FINAL_DIAGNOSTIC_SAMPLES) {
  const sourceCount = Math.max(0, Math.round(Number(count) || 0));
  const sampleLimit = Math.max(1, Math.round(Number(limit) || 1));
  const stride = Math.max(1, Math.ceil(sourceCount / sampleLimit));
  return {
    sourceCount,
    stride,
    sampleCount: sourceCount > 0 ? Math.ceil(sourceCount / stride) : 0,
  };
}

function virtualDepthDistribution(params) {
  if (!params?.count) return null;
  const sampling = finalDiagnosticSampling(params.count);
  const raw = new Float32Array(sampling.sampleCount);
  const virtual = new Float32Array(sampling.sampleCount);
  const composite = new Float32Array(sampling.sampleCount);
  let sampleIndex = 0;
  for (let index = 0; index < params.count; index += sampling.stride) {
    raw[sampleIndex] = Number(params.virtualDepth?.[index]) || 0;
    virtual[sampleIndex] = boundedVirtualDepth(params, index);
    composite[sampleIndex] = plyLayerDepth(index, params);
    sampleIndex += 1;
  }
  const summarize = (values) => {
    const sorted = values.slice().sort();
    const at = (fraction) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * fraction)))];
    return {
      minimum: sorted[0],
      p01: at(0.01),
      p50: at(0.5),
      p99: at(0.99),
      maximum: sorted.at(-1),
      mean_abs: values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length,
    };
  };
  return {
    enabled: Boolean(params.virtualDepthEnabled),
    thickness: Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS,
    source_count: sampling.sourceCount,
    sample_count: sampling.sampleCount,
    sample_stride: sampling.stride,
    raw: summarize(raw),
    virtual_z: summarize(virtual),
    composite_z: summarize(composite),
  };
}

function sampleImageAt(image, x, y, out, offset) {
  const px = Math.max(0, Math.min(image.width - 1, Math.round(((x + 1) * 0.5) * (image.width - 1))));
  const py = Math.max(0, Math.min(image.height - 1, Math.round(((y + 1) * 0.5) * (image.height - 1))));
  const source = (py * image.width + px) * 3;
  out[offset] = image.rgb[source];
  out[offset + 1] = image.rgb[source + 1];
  out[offset + 2] = image.rgb[source + 2];
}

function srgbSignalToLinear(value) {
  const channel = Math.max(0, Math.min(1, Number(value) || 0));
  return channel <= 0.04045
    ? channel / 12.92
    : ((channel + 0.055) / 1.055) ** 2.4;
}

function linearToSrgbSignal(value) {
  const channel = Math.max(0, Math.min(1, Number(value) || 0));
  return channel <= 0.0031308
    ? channel * 12.92
    : 1.055 * channel ** (1 / 2.4) - 0.055;
}

function neutralSrgbPreservingLabL(r, g, b) {
  const relativeY =
    0.2126729 * srgbSignalToLinear(r) +
    0.7151522 * srgbSignalToLinear(g) +
    0.0721750 * srgbSignalToLinear(b);
  return linearToSrgbSignal(relativeY);
}

function convertRgbToNeutralLabL(rgb) {
  for (let offset = 0; offset + 2 < rgb.length; offset += 3) {
    const gray = neutralSrgbPreservingLabL(
      rgb[offset],
      rgb[offset + 1],
      rgb[offset + 2],
    );
    rgb[offset] = gray;
    rgb[offset + 1] = gray;
    rgb[offset + 2] = gray;
  }
  return rgb;
}

function brushColorNeutrality(params = state.params) {
  const rgb = params?.rgb;
  const count = Math.min(params?.count || 0, Math.floor((rgb?.length || 0) / 3));
  let maximumChannelDelta = 0;
  let meanChannelDelta = 0;
  for (let index = 0; index < count; index += 1) {
    const offset = index * 3;
    const delta = Math.max(
      Math.abs(rgb[offset] - rgb[offset + 1]),
      Math.abs(rgb[offset + 1] - rgb[offset + 2]),
      Math.abs(rgb[offset + 2] - rgb[offset]),
    );
    maximumChannelDelta = Math.max(maximumChannelDelta, delta);
    meanChannelDelta += delta;
  }
  return {
    count,
    maximum_channel_delta: maximumChannelDelta,
    mean_channel_delta: meanChannelDelta / Math.max(1, count),
    completely_neutral: maximumChannelDelta <= 1e-6,
  };
}

function densifyWarmupSteps(steps) {
  return steps >= 10 ? Math.min(DENSIFY_WARMUP_MAX_STEPS, Math.floor(steps * DENSIFY_WARMUP_FRACTION)) : 0;
}

function curriculumStageStep(steps, fraction) {
  const total = Math.max(1, Math.round(steps));
  return Math.min(total, Math.max(1, Math.round(total * fraction)));
}

function experimentalCoarseSteps(steps, override = null) {
  const total = Math.max(1, Math.round(steps));
  if (Number.isFinite(override)) return Math.min(total, Math.max(0, Math.round(override)));
  return curriculumStageStep(total, CURRICULUM_COARSE_FRACTION);
}

function experimentalDensifySteps(steps) {
  return curriculumStageStep(steps, CURRICULUM_DENSITY_FRACTION);
}

function experimentalGrowthSteps(steps, fraction = phase39Variants().growthApplyUntilFraction) {
  const boundedFraction = Math.max(0, Math.min(1, Number(fraction) || 0));
  return boundedFraction > 0 ? curriculumStageStep(steps, boundedFraction) : 0;
}

function overdensityCorrectionScheduleSteps(steps, growthSteps, settings) {
  const horizon = Math.max(0, Math.round(growthSteps));
  if (!settings?.midTrainingOverdensityCorrection || horizon <= 0) return [];
  if (settings.overdensityCorrectionSchedule === "p2-p3-start") {
    return [
      experimentalCoarseSteps(steps) + 1,
      experimentalDensifySteps(steps) + 1,
    ].filter((step, index, values) => step <= horizon && values.indexOf(step) === index);
  }
  const interval = Math.max(100, Math.round(
    settings.overdensityCorrectionInterval || DEFAULT_OVERDENSITY_CORRECTION_INTERVAL,
  ));
  const scheduled = [];
  for (let step = interval; step <= horizon; step += interval) scheduled.push(step);
  return scheduled;
}

function growthEventScheduled(step, densitySteps, growthSteps, settings) {
  const current = Math.max(1, Math.round(step));
  const horizon = Math.max(0, Math.round(growthSteps));
  if (!settings?.densityEventsEnabled || horizon <= 0 || current > horizon) return false;
  if (current === horizon) return true;
  return current > densifyWarmupSteps(densitySteps) &&
    current % Math.max(1, Math.round(settings?.densifyInterval || 1)) === 0;
}

function curriculumTrainingStage(step, steps, variants, coarseImage, midImage) {
  const coarseEnd = experimentalCoarseSteps(steps, variants.coarseSteps);
  if (variants.coarseToFull && coarseImage && step <= coarseEnd) return "coarse";
  if (variants.threeStageCurriculum && midImage && step <= experimentalDensifySteps(steps)) return "mid";
  return "full";
}

function opaquePaintVisibilityCompactionStep(
  steps,
  fraction = OPAQUE_PAINT_LATE_SETTLE_FRACTION,
) {
  return Math.max(1, opaquePaintLateSettleStartStep(steps, fraction) - 1);
}

function opaquePaintVisibilityGraceSteps(steps) {
  return Math.min(
    OPAQUE_PAINT_VISIBILITY_GRACE_STEPS,
    Math.max(OPAQUE_PAINT_VISIBILITY_MIN_GAP_STEPS, Math.round(Math.max(1, steps) * 0.02)),
  );
}

function opaquePaintVisibilityCompactionDue(step, steps, settings) {
  if (
    !settings?.opaqueLayered ||
    settings?.currentVisibilityCompactionEnabled === false ||
    settings?.currentContributionCompactionEnabled === true
  ) return false;
  const compactionStep = opaquePaintVisibilityCompactionStep(
    steps,
    settings?.opaquePaintSettleFraction,
  );
  const growthEnd = experimentalGrowthSteps(steps, settings?.growthApplyUntilFraction);
  if (growthEnd > 0 && compactionStep >= growthEnd) return false;
  const grace = compactionStep - growthEnd;
  return grace >= opaquePaintVisibilityGraceSteps(steps) && Math.round(step) === compactionStep;
}

function currentContributionCompactionStep(steps, settings) {
  const total = Math.max(1, Math.round(steps));
  const measurementWindowSteps = Math.max(1, Math.round(settings?.measurementWindowSteps || 1));
  const deadline = currentContributionCompactionDeadline(total, settings);
  const requestedReset = Math.max(1, Math.floor(total * clampNumber(
    settings?.startFraction,
    CURRENT_CONTRIBUTION_MIN_COMPACTION_FRACTION,
    CURRENT_CONTRIBUTION_MAX_COMPACTION_FRACTION,
    CURRENT_CONTRIBUTION_COMPACTION_FRACTION,
  )) + 1);
  const firstEvent = requestedReset + measurementWindowSteps - 1;
  return firstEvent <= deadline ? firstEvent : 0;
}

function currentContributionCompactionResetStep(steps, settings) {
  const target = currentContributionCompactionStep(steps, settings);
  if (target <= 0) return 0;
  const window = Math.max(1, Math.round(settings?.measurementWindowSteps || 1));
  return Math.max(1, target - window + 1);
}

function currentContributionCompactionDeadline(steps, settings) {
  const total = Math.max(1, Math.round(steps));
  const structuralDeadline = settings?.opaqueLayered
    ? opaquePaintVisibilityCompactionStep(total, settings?.opaquePaintSettleFraction)
    : total - 1;
  const growthEnd = experimentalGrowthSteps(total, settings?.growthApplyUntilFraction);
  // The final growth event closes active cardinality. Do not physically remove
  // splats at or after that milestone; the remaining tail optimizes a fixed set.
  return growthEnd > 0
    ? Math.max(0, Math.min(structuralDeadline, growthEnd - 1))
    : structuralDeadline;
}

function currentContributionCompactionSchedule(step, steps, settings) {
  const current = Math.max(1, Math.round(Number(step) || 1));
  const firstEvent = currentContributionCompactionStep(steps, settings);
  const firstReset = currentContributionCompactionResetStep(steps, settings);
  const window = Math.max(1, Math.round(settings?.measurementWindowSteps || 1));
  const interval = Math.max(
    window,
    Math.round(settings?.intervalSteps || CURRENT_CONTRIBUTION_COMPACTION_INTERVAL),
  );
  const deadline = currentContributionCompactionDeadline(steps, settings);
  const resetOffset = current - firstReset;
  const eventOffset = current - firstEvent;
  return {
    enabled: Boolean(settings?.enabled) && firstEvent > 0,
    firstEvent,
    firstReset,
    interval,
    window,
    deadline,
    resetDue: Boolean(settings?.enabled) && firstReset > 0 && current <= deadline &&
      resetOffset >= 0 && resetOffset % interval === 0,
    compactionDue: Boolean(settings?.enabled) && firstEvent > 0 && current <= deadline &&
      eventOffset >= 0 && eventOffset % interval === 0,
  };
}

function currentContributionCompactionResetDue(step, steps, settings) {
  return currentContributionCompactionSchedule(step, steps, settings).resetDue;
}

function currentContributionCompactionDue(step, steps, settings) {
  return currentContributionCompactionSchedule(step, steps, settings).compactionDue;
}

function opaquePaintDetailRecoveryDue(step, steps, interval = OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL) {
  const cadence = Math.max(2, Math.round(interval || OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL));
  const offset = Math.floor(cadence / 2);
  // Keep ownership moves on their established cadence and leave a short
  // settling window before the final result is presented.
  return step > 0 && step < Math.max(1, steps - 16) && step % cadence === offset;
}

function performanceProfileSchedule(steps) {
  const total = Math.max(1, Math.round(steps));
  const b1 = experimentalCoarseSteps(total, phase33Variants().coarseSteps);
  const b2 = experimentalDensifySteps(total);
  const horizon = experimentalGrowthSteps(total);
  const schedule = new Map();
  for (const [label, rawStep] of [
    ["B1-100", b1 - 100],
    ["B1", b1],
    ["B1+1", b1 + 1],
    ["B1+100", b1 + 100],
    ["B2", b2],
    ["H", horizon],
    ["S", total],
  ]) {
    const step = Math.max(1, Math.min(total, Math.round(rawStep)));
    const labels = schedule.get(step) || [];
    labels.push(label);
    schedule.set(step, labels);
  }
  return schedule;
}

function performanceProfileLabels(step, steps) {
  return performanceProfileSchedule(steps).get(Math.round(step)) || [];
}

function summarizeTrainingScheduling(profileSamples = [], traceSamples = []) {
  const gpuPasses = {};
  const profileIo = {
    queue_wait_count: 0,
    queue_wait_wall_ms: 0,
    readback_count: 0,
    readback_bytes: 0,
    readback_wall_ms: 0,
  };
  for (const sample of profileSamples) {
    for (const [name, milliseconds] of Object.entries(sample.stages_ms || {})) {
      if (!Number.isFinite(milliseconds)) continue;
      const entry = gpuPasses[name] || { samples: 0, total_ms: 0, mean_ms: 0, max_ms: 0 };
      entry.samples += 1;
      entry.total_ms += milliseconds;
      entry.max_ms = Math.max(entry.max_ms, milliseconds);
      entry.mean_ms = entry.total_ms / entry.samples;
      gpuPasses[name] = entry;
    }
    profileIo.queue_wait_count += Math.max(0, Number(sample.queue_wait_count) || 0);
    profileIo.queue_wait_wall_ms += Math.max(0, Number(sample.queue_wait_wall_ms) || 0);
    profileIo.readback_count += Math.max(0, Number(sample.readback_count) || 0);
    profileIo.readback_bytes += Math.max(0, Number(sample.readback_bytes) || 0);
    profileIo.readback_wall_ms += Math.max(0, Number(sample.readback_wall_ms) || 0);
  }
  const runtime = {};
  for (const sample of traceSamples) {
    const phase = sample.phase || "unknown";
    const entry = runtime[phase] || {
      samples: 0,
      steps: 0,
      train_ms: 0,
      density_ms: 0,
      relocation_ms: 0,
      presentation_ms: 0,
      wall_ms: 0,
    };
    entry.samples += 1;
    entry.steps += Math.max(0, Number(sample.interval_steps) || 0);
    entry.train_ms += Math.max(0, Number(sample.train_ms) || 0);
    entry.density_ms += Math.max(0, Number(sample.density_ms) || 0);
    entry.relocation_ms += Math.max(0, Number(sample.relocation_ms) || 0);
    entry.presentation_ms += Math.max(0, Number(sample.presentation_ms) || 0);
    entry.wall_ms += Math.max(0, Number(sample.interval_ms) || 0);
    runtime[phase] = entry;
  }
  for (const entry of Object.values(runtime)) {
    entry.iterations_per_second = entry.steps > 0 && entry.wall_ms > 0
      ? (entry.steps * 1000) / entry.wall_ms
      : 0;
  }
  return {
    timestamp_samples: profileSamples.length,
    timestamp_gpu_passes_ms: gpuPasses,
    timestamp_profile_io: profileIo,
    runtime_wall_by_phase_ms: runtime,
  };
}

function experimentalAdcInterval(steps) {
  if (steps >= 3000) return EXPERIMENTAL_ADC_INTERVAL_FOR_7000;
  return Math.max(300, Math.round(steps / 4));
}

function experimentalAdcWindow(steps, interval, densifyInterval, minimumEvents) {
  const proportional = Math.max(24, Math.min(160, Math.round(steps * 0.02)));
  const eventAligned = Math.max(1, Math.round(densifyInterval)) * Math.max(1, Math.round(minimumEvents));
  return Math.min(Math.max(1, Math.round(interval)), Math.max(proportional, eventAligned));
}

function experimentalSchedule(steps) {
  const densityHorizon = experimentalDensifySteps(steps);
  const warmup = densifyWarmupSteps(densityHorizon);
  const phase38 = phase38Variants();
  const adcInterval = phase38.adcSplitInterval || experimentalAdcInterval(densityHorizon);
  const adcWindow = experimentalAdcWindow(
    densityHorizon,
    adcInterval,
    phase39Variants().densifyInterval,
    phase38.adcWindowEvents,
  );
  const resetInterval = phase38.adcResetInterval || EXPERIMENTAL_ADC_INTERVAL_FOR_7000;
  const resetHorizon = experimentalGrowthSteps(steps);
  return { steps, densityHorizon, warmup, adcInterval, adcWindow, resetInterval, resetHorizon };
}

function densityGpuConfig({ image, count, targetCount, step, steps, layout, maxAnisotropy, capacity, mode }) {
  const schedule = experimentalSchedule(steps);
  const variants = phase33Variants();
  const phase37 = phase37Variants();
  const phase38 = phase38Variants();
  const phase39 = phase39Variants();
  const phase45 = phase45Variants();
  const layerEfficiency = layerEfficiencyVariants();
  const productOverdensityCorrection = mode === 3 && phase39.midTrainingOverdensityCorrection;
  const phase45DonorActive = productOverdensityCorrection || (
    phase45.donorEligibility && (!phase45.firstResetOnly || step <= schedule.resetInterval)
  );
  const detail = selectedLearningRates();
  const trainState = state.webgpu.renderer?.trainState;
  const trainingStage = curriculumTrainingStage(step, steps, variants, trainState?.coarseImage, trainState?.midImage);
  const stageImage = trainingStage === "coarse"
    ? trainState?.coarseImage
    : trainingStage === "mid"
      ? trainState?.midImage
      : image;
  const stageMinScale = stageMinimumScale(
    stageImage || image,
    state.metrics?.initial_splats || count,
    trainingStage,
    variants.stageMinScaleRatio,
  );
  const baseScaleFloorRatio = stageBaseScaleFloorRatio(trainingStage);
  const shaderStepLimit = mode === 3 ? schedule.densityHorizon : steps;
  const config = new Float32Array(TRAIN_CONFIG_FLOATS);
  config.set([
      image.width,
      image.height,
      count,
      targetCount,
      step,
      shaderStepLimit,
      layout.cols,
      layout.rows,
      layout.baseScale,
      maxAnisotropy,
      capacity,
      mode,
      schedule.adcInterval,
      schedule.adcWindow,
      schedule.warmup,
      schedule.densityHorizon,
      variants.importanceRecycle ? 1 : 0,
      variants.adcEligibility ? 1 : 0,
      variants.coverageDensity ? 1 : 0,
      variants.structureTensor ? 1 : 0,
      variants.importanceEma,
      variants.coverageTarget,
      variants.coverageLossWeight,
      variants.coverageDensityStrength,
      phase37.absGradient ? 1 : 0,
      phase37.gradientCoherence ? 1 : 0,
      phase37.edgeErrorDensity ? 1 : 0,
      phase37.significanceRecycle ? 1 : 0,
      phase37.structureAnisotropy ? 1 : 0,
      detail.adaptiveDetail ? 1 : 0,
      detail.maxAnisotropy,
      detail.detailCoherence,
      0,
      phase45.seedOffset,
      phase39.growthSignalThreshold,
      els.trainLayerOrder.checked ? 1 : 0,
      mode === 3 ? step + 1 : step,
      phase39.tiltRobustSplit ? 1 : 0,
      phase39.tiltSplitAngleDegrees * Math.PI / 180,
      PLY_LAYER_DEPTH_SPAN,
      DEFAULT_TILT_SPLIT_COLOR_THRESHOLD,
      DEFAULT_TILT_SPLIT_SHRINK,
      phase39.singleSourceClaim ? 1 : 0,
      phase39.qaGrowthComparisons ? 1 : 0,
      phase45.telemetry || phase45DonorActive || phase45.recipientScore ? 1 : 0,
      phase45DonorActive ? 1 : 0,
      phase45.donorQuantile,
      phase45.recipientScore ? phase45.recipientStrength : 0,
      phase39.growthSignalThreshold,
      0,
      0,
      0,
      layout.baseScaleX,
      layout.baseScaleY,
      detail.boundarySigma,
      detail.surfaceAnisotropy,
      detail.maxPlanarScale,
      phase39.adcSplitEnabled ? 1 : 0,
      phase39.adcRecycleEnabled ? 1 : 0,
      phase39.mcmcRelocationEnabled ? 1 : 0,
      stageMinScale,
      baseScaleFloorRatio,
      layerEfficiency.deepRelocation ? 1 : 0,
      layerEfficiency.deepFraction,
      layerEfficiency.influenceThreshold,
      state.params?.opaqueLayered ? 1 : 0,
    ], 0);
  config[49] = experimentalDensifySteps(steps);
  configurePaintKernel(config, state.params);
  // Density-only algorithm flags. Training config reuses these slots under a
  // separate shader contract and rewrites them before optimizer dispatch.
  config[66] = state.params?.virtualCameraSamplingEnabled ? 1 : 0;
  config[67] = state.params?.virtualDepthEnabled ? 1 : 0;
  config[68] = phase39.structureGuidedAllocation ? 1 : 0;
  config[69] = phase39.structureRegionGrid;
  if (productOverdensityCorrection) {
    // The region quantile is already the move cap. Do not apply a second
    // random ADC sampling gate to the independently safe donor cohort.
    config[46] = phase39.overdensityDonorFraction;
    config[47] = 1;
    config[50] = 1;
    config[51] = 1;
  }
  return { schedule, config };
}

function experimentalDensityPhase(step, steps) {
  const schedule = experimentalSchedule(steps);
  if (step > schedule.densityHorizon) return "settle";
  if (step <= schedule.warmup) return "warmup";
  return "growth";
}

function splatTargetForGrowth(currentCount, finalCount, growthFraction = DEFAULT_GROWTH_FRACTION) {
  if (finalCount <= currentCount) return currentCount;
  const added = Math.max(1, Math.ceil(currentCount * Math.max(0.001, growthFraction)));
  return normalizeActiveSplatCount(Math.min(finalCount, currentCount + added), currentCount);
}

function remainingGrowthEventCount(step, growthEnd, densifyInterval) {
  const current = Math.max(1, Math.round(step));
  const end = Math.max(current, Math.round(growthEnd));
  const interval = Math.max(1, Math.round(densifyInterval));
  if (current >= end) return 1;
  const regularAfterCurrent = Math.max(0, Math.floor((end - 1) / interval) - Math.floor(current / interval));
  return 1 + regularAfterCurrent + 1; // current event + regular events + terminal H event
}

function growthSchedulePlan({
  step,
  steps,
  initialCount,
  currentCount,
  finalCount,
  growthFraction,
  growthApplyUntilFraction = phase39Variants().growthApplyUntilFraction,
  densifyInterval,
  stageAware,
  stageGrowthShares = Object.fromEntries(Object.entries(DEFAULT_STAGE_GROWTH_SHARES).map(([key, value]) => [key, value / 100])),
}) {
  const normalTarget = splatTargetForGrowth(currentCount, finalCount, growthFraction);
  const normalIncrement = Math.max(0, normalTarget - currentCount);
  const densityEnd = experimentalDensifySteps(steps);
  const growthEnd = experimentalGrowthSteps(steps, growthApplyUntilFraction);
  if (!stageAware) {
    const terminalClosure = growthEnd > 0 && Math.round(step) === growthEnd;
    return {
      mode: "threshold-percentage-target-closure",
      desiredCount: terminalClosure ? finalCount : normalTarget,
      previousDesired: currentCount,
      normalIncrement,
      catchUpLimit: finalCount,
      requestedCount: terminalClosure ? finalCount : normalTarget,
      densityEnd,
      growthEnd,
      remainingGrowthEvents: terminalClosure ? 1 : null,
      scheduleDebt: 0,
      repairIncrement: 0,
      terminalClosure,
    };
  }

  const warmup = densifyWarmupSteps(densityEnd);
  const p1End = experimentalCoarseSteps(steps);
  const range = Math.max(0, finalCount - initialCount);
  const p1Share = Math.max(0, Math.min(1, Number(stageGrowthShares.p1) || 0));
  const p2Share = Math.max(0, Math.min(1 - p1Share, Number(stageGrowthShares.p2) || 0));
  const p2Cumulative = p1Share + p2Share;
  const desiredAt = (targetStep) => {
    const segmentProgress = targetStep <= p1End
      ? Math.max(0, Math.min(1, (targetStep - warmup) / Math.max(1, p1End - warmup))) * p1Share
      : targetStep <= densityEnd
        ? p1Share + Math.max(0, Math.min(1, (targetStep - p1End) / Math.max(1, densityEnd - p1End))) * p2Share
        : p2Cumulative + Math.max(0, Math.min(1, (targetStep - densityEnd) / Math.max(1, growthEnd - densityEnd))) * (1 - p2Cumulative);
    return Math.min(finalCount, initialCount + Math.round(range * segmentProgress));
  };
  const terminalClosure = growthEnd > 0 && Math.round(step) === growthEnd;
  const desiredCount = terminalClosure ? finalCount : desiredAt(step);
  const previousDesired = desiredAt(Math.max(warmup, step - Math.max(1, densifyInterval)));
  const stageIncrement = Math.max(0, desiredCount - previousDesired);
  const remainingGrowthEvents = remainingGrowthEventCount(step, growthEnd, densifyInterval);
  const scheduleDebt = Math.max(0, previousDesired - currentCount);
  const repairIncrement = Math.ceil(scheduleDebt / remainingGrowthEvents);
  const catchUpLimit = Math.min(finalCount, currentCount + stageIncrement + repairIncrement);
  const requestedCount = normalizeActiveSplatCount(
    Math.max(currentCount, Math.min(desiredCount, catchUpLimit)),
    currentCount,
  );
  return {
    mode: "stage-aware-percentage-cap",
    desiredCount,
    previousDesired,
    normalIncrement: stageIncrement,
    catchUpLimit,
    requestedCount,
    densityEnd,
    growthEnd,
    remainingGrowthEvents,
    scheduleDebt,
    repairIncrement,
    terminalClosure,
    stageGrowthShares: { p1: p1Share, p2: p2Share, p3: Math.max(0, 1 - p2Cumulative) },
  };
}

function referenceGrowthTargets(currentCount, finalCount, eligibleSourceCount, enabled) {
  if (!enabled) return null;
  const capReached = currentCount >= finalCount;
  const eligible = Number.isFinite(eligibleSourceCount) ? eligibleSourceCount : null;
  return {
    reference_only: true,
    changes_requested_count: false,
    status: capReached ? "cap-reached" : eligible === null ? "not-sampled" : "sampled",
    default_like_eligible_count: eligible,
    default_like_target: capReached ? finalCount : eligible === null ? null : Math.min(finalCount, currentCount + eligible),
    mcmc_like_fraction: 0.05,
    mcmc_like_target: capReached
      ? finalCount
      : Math.min(finalCount, currentCount + Math.max(1, Math.ceil(currentCount * 0.05))),
  };
}

function growParamPlaceholders(params, targetCount) {
  if (targetCount <= params.count) return params;
  const oldCount = params.count;
  const xy = new Float32Array(targetCount * 2);
  const scale = new Float32Array(targetCount * 2);
  const rgb = new Float32Array(targetCount * 3);
  const opacity = new Float32Array(targetCount);
  const theta = new Float32Array(targetCount);
  const depthOrder = new Float32Array(targetCount);
  const virtualDepth = new Float32Array(targetCount);
  const brushTaper = new Float32Array(targetCount);
  const detailTags = new Float32Array(targetCount);
  xy.set(params.xy);
  scale.set(params.scale);
  rgb.set(params.rgb);
  opacity.set(params.opacity);
  theta.set(params.theta);
  depthOrder.set(params.depthOrder || initialDepthOrder(oldCount));
  virtualDepth.set(params.virtualDepth || new Float32Array(oldCount));
  brushTaper.set(
    params.brushTaper || new Float32Array(oldCount).fill(DEFAULT_LAYERED_BRUSH_TAPER),
  );
  detailTags.set(params.detailTags || new Float32Array(oldCount).fill(1));
  for (let i = oldCount; i < targetCount; i += 1) {
    const source = i % oldCount;
    xy[i * 2] = params.xy[source * 2];
    xy[i * 2 + 1] = params.xy[source * 2 + 1];
    scale[i * 2] = params.scale[source * 2];
    scale[i * 2 + 1] = params.scale[source * 2 + 1];
    rgb[i * 3] = params.rgb[source * 3];
    rgb[i * 3 + 1] = params.rgb[source * 3 + 1];
    rgb[i * 3 + 2] = params.rgb[source * 3 + 2];
    opacity[i] = params.opacity[source];
    theta[i] = params.theta[source];
    depthOrder[i] = params.depthOrder?.[source] ?? (1 - source / Math.max(1, oldCount - 1));
    virtualDepth[i] = params.virtualDepth?.[source] ?? 0;
    brushTaper[i] = params.brushTaper?.[source] ?? DEFAULT_LAYERED_BRUSH_TAPER;
    detailTags[i] = params.detailTags?.[source] ?? 1;
  }
  return {
    kernelShape: normalizedKernelShape(params.kernelShape),
    ...(params.flowBirthLinksEnabled ? {
      flowBirthLinksEnabled: true,
      flowBirthLinkStrength: params.flowBirthLinkStrength,
      flowLinkedSplatMin: params.flowLinkedSplatMin,
      flowLinkedSplatMax: params.flowLinkedSplatMax,
      flowBackcoatCount: params.flowBackcoatCount,
      flowTrainingSize: params.flowTrainingSize?.slice(),
    } : {}),
    rectangleTopRatio: clampNumber(
      params.rectangleTopRatio,
      MIN_RECTANGLE_TOP_RATIO,
      MAX_RECTANGLE_TOP_RATIO,
      DEFAULT_RECTANGLE_TOP_RATIO,
    ),
    rectangleTopRatioMax: clampNumber(
      params.rectangleTopRatioMax,
      params.rectangleTopRatio,
      MAX_RECTANGLE_TOP_RATIO,
      DEFAULT_RECTANGLE_TOP_RATIO_MAX,
    ),
    rectangleOpacityGradientMin: clampNumber(
      params.rectangleOpacityGradientMin,
      0,
      1,
      1,
    ),
    rectangleOpacityGradientMax: clampNumber(
      params.rectangleOpacityGradientMax,
      0,
      1,
      1,
    ),
    rectangleCenterOpacityGradientMin: clampNumber(
      params.rectangleCenterOpacityGradientMin,
      0,
      1,
      1,
    ),
    rectangleCenterOpacityGradientMax: clampNumber(
      params.rectangleCenterOpacityGradientMax,
      clampNumber(params.rectangleCenterOpacityGradientMin, 0, 1, 1),
      1,
      1,
    ),
    rectangleMinAspectRatio: clampNumber(
      params.rectangleMinAspectRatio,
      MIN_RECTANGLE_ASPECT_RATIO,
      params.rectangleMaxAspectRatio,
      DEFAULT_RECTANGLE_MIN_ASPECT_RATIO,
    ),
    rectangleMaxAspectRatio: clampNumber(
      params.rectangleMaxAspectRatio,
      MIN_RECTANGLE_ASPECT_RATIO,
      MAX_RECTANGLE_ASPECT_RATIO,
      DEFAULT_RECTANGLE_ASPECT_RATIO,
    ),
    rectangleOrientation: normalizedRectangleOrientation(params.rectangleOrientation),
    rectangleOrientationTolerance: normalizedRectangleOrientationTolerance(params.rectangleOrientationTolerance),
    rectanglePreserveArea:
      params.rectanglePreserveArea ?? DEFAULT_RECTANGLE_PRESERVE_AREA,
    rectangleEdgeDirectedTaper:
      params.rectangleEdgeDirectedTaper ?? DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER,
    rectangleStructureAwareRatio:
      params.rectangleStructureAwareRatio ?? DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO,
    rectangleAsymmetricSoftness:
      params.rectangleAsymmetricSoftness ?? DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS,
    illustrativeOilVersion: Math.max(0, Math.round(Number(params.illustrativeOilVersion) || 0)),
    brushLocalColorFlowEnabled: Boolean(params.brushLocalColorFlowEnabled),
    brushStrokePersistenceEnabled: Boolean(params.brushStrokePersistenceEnabled),
    brushRibbonAspectFloor: clampNumber(
      params.brushRibbonAspectFloor,
      1,
      LIMITS.maxAnisotropyMax,
      BRUSH_STROKE_PERSISTENCE_RIBBON_MIN_RATIO,
    ),
    brushAccentAspectFloor: clampNumber(
      params.brushAccentAspectFloor,
      1,
      LIMITS.maxAnisotropyMax,
      BRUSH_STROKE_PERSISTENCE_ACCENT_MIN_RATIO,
    ),
    surfaceLayerPriorEnabled: Boolean(params.surfaceLayerPriorEnabled),
    surfaceLayerPriorColorAwarePromotion:
      params.surfaceLayerPriorColorAwarePromotion !== false,
    trainLayerColorGuardEnabled: Boolean(params.trainLayerColorGuardEnabled),
    surfaceLayerPriorLayers: Math.round(clampNumber(
      params.surfaceLayerPriorLayers,
      MIN_DISCRETE_LAYER_COUNT,
      MAX_DISCRETE_LAYER_COUNT,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
    )),
    surfaceLayerPriorP1Interval: Math.max(0, Math.round(params.surfaceLayerPriorP1Interval || 0)),
    surfaceLayerPriorP2Interval: Math.max(0, Math.round(params.surfaceLayerPriorP2Interval || 0)),
    surfaceLayerPriorP3Interval: Math.max(0, Math.round(params.surfaceLayerPriorP3Interval || 0)),
    surfaceLayerPriorUntilFraction: clampNumber(
      params.surfaceLayerPriorUntilFraction,
      0,
      1,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_UNTIL,
    ),
    harmfulRectangleParentSplitEnabled: Boolean(params.harmfulRectangleParentSplitEnabled),
    harmfulRectangleParentSplitTransitionOnly: Boolean(
      params.harmfulRectangleParentSplitTransitionOnly,
    ),
    frontSplitChildrenEnabled: Boolean(params.frontSplitChildrenEnabled),
    brushOpacityGradientEnabled: Boolean(params.brushOpacityGradientEnabled),
    brushOpacityGradientStart: clampNumber(params.brushOpacityGradientStart, 0, 1, 0),
    brushOpacityGradientEnd: clampNumber(params.brushOpacityGradientEnd, 0, 1, 1),
    brushCenterOpacityGradientMin: clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1),
    brushCenterOpacityGradientMax: clampNumber(
      params.brushCenterOpacityGradientMax,
      clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1),
      1,
      1,
    ),
    brushMinAspectRatio: clampNumber(params.brushMinAspectRatio, LIMITS.maxAnisotropyMin, params.brushMaxAspectRatio, LIMITS.maxAnisotropyMin),
    brushMaxAspectRatio: clampNumber(params.brushMaxAspectRatio, LIMITS.maxAnisotropyMin, LIMITS.maxAnisotropyMax, DEFAULT_MAX_ANISOTROPY),
    brushWidthTaperEnabled: Boolean(params.brushWidthTaperEnabled),
    brushWidthTaperStart: clampNumber(params.brushWidthTaperStart, 0, 1, 1),
    brushWidthTaperEnd: clampNumber(params.brushWidthTaperEnd, 0, 1, 0),
    monochromeUnderpaintingEnabled: Boolean(params.monochromeUnderpaintingEnabled),
    colorFinishStartPercent: clampNumber(
      params.colorFinishStartPercent,
      MIN_COLOR_FINISH_START_PERCENT,
      MAX_COLOR_FINISH_START_PERCENT,
      DEFAULT_COLOR_FINISH_START_PERCENT,
    ),
    colorFinishStartStep: Math.max(
      0,
      Math.round(Number(params.colorFinishStartStep) || 0),
    ),
    currentVisibilityChildPolicyEnabled: params.currentVisibilityChildPolicyEnabled !== false,
    currentVisibilityCompactionEnabled: params.currentVisibilityCompactionEnabled !== false,
    illustrativeOilFamilyStats: params.illustrativeOilFamilyStats
      ? structuredClone(params.illustrativeOilFamilyStats)
      : null,
    opaqueLayered: Boolean(params.opaqueLayered),
    minimumOpacityEnabled: Boolean(params.minimumOpacityEnabled),
    minimumOpacity: clampNumber(
      params.minimumOpacity,
      MIN_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
    ),
    maximumOpacity: clampNumber(params.maximumOpacity, params.minimumOpacity, MAX_LEARNED_PAINT_OPACITY, MAX_LEARNED_PAINT_OPACITY),
    xy, scale, rgb, opacity, theta, depthOrder, virtualDepth, brushTaper, detailTags, count: targetCount,
    rows: params.rows, cols: params.cols, bg: params.bg,
    boundarySigma: params.boundarySigma,
    layerOrderEnabled: Boolean(params.layerOrderEnabled),
    layerAwareAccumulationEnabled: Boolean(params.layerAwareAccumulationEnabled),
    discreteLayersEnabled: Boolean(params.discreteLayersEnabled),
    discreteLayerCount: params.discreteLayerCount,
    discreteLayerMoveRadius: params.discreteLayerMoveRadius,
    virtualDepthEnabled: Boolean(params.virtualDepthEnabled),
    virtualDepthThickness: Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS,
    virtualDepthSoftConstraintEnabled: params.virtualDepthSoftConstraintEnabled !== false,
    virtualDepthPriorDelta: Number(params.virtualDepthPriorDelta) || DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA,
  };
}

function compactSplatParams(params, keepIndices) {
  const count = keepIndices.length;
  const copy = (source, stride, fallback = 0) => {
    const output = new Float32Array(count * stride);
    for (let target = 0; target < count; target += 1) {
      const sourceIndex = keepIndices[target];
      for (let component = 0; component < stride; component += 1) {
        output[target * stride + component] = source?.[sourceIndex * stride + component] ?? fallback;
      }
    }
    return output;
  };
  return {
    ...params,
    count,
    xy: copy(params.xy, 2),
    scale: copy(params.scale, 2),
    rgb: copy(params.rgb, 3),
    opacity: copy(params.opacity, 1),
    theta: copy(params.theta, 1),
    depthOrder: copy(params.depthOrder, 1),
    virtualDepth: copy(params.virtualDepth, 1),
    brushTaper: copy(params.brushTaper, 1, DEFAULT_LAYERED_BRUSH_TAPER),
    detailTags: copy(params.detailTags, 1, 1),
  };
}

const FOOTPRINT_COLOR_SAMPLES = [
  [0, 0, 1],
  [-1, 0, Math.exp(-0.5)],
  [1, 0, Math.exp(-0.5)],
  [0, -1, Math.exp(-0.5)],
  [0, 1, Math.exp(-0.5)],
  [-1, -1, Math.exp(-1)],
  [1, -1, Math.exp(-1)],
  [-1, 1, Math.exp(-1)],
  [1, 1, Math.exp(-1)],
];

function sampleImageRgbBilinear(image, x, y, out) {
  const fx = Math.max(0, Math.min(image.width - 1, ((x + 1) * 0.5) * (image.width - 1)));
  const fy = Math.max(0, Math.min(image.height - 1, ((y + 1) * 0.5) * (image.height - 1)));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const row = image.width * 3;
  const i00 = y0 * row + x0 * 3;
  const i10 = y0 * row + x1 * 3;
  const i01 = y1 * row + x0 * 3;
  const i11 = y1 * row + x1 * 3;
  for (let channel = 0; channel < 3; channel += 1) {
    const top = image.rgb[i00 + channel] * (1 - tx) + image.rgb[i10 + channel] * tx;
    const bottom = image.rgb[i01 + channel] * (1 - tx) + image.rgb[i11 + channel] * tx;
    out[channel] = top * (1 - ty) + bottom * ty;
  }
}

function footprintWeightedTargetColor(image, params, index, out = new Float32Array(3)) {
  out.fill(0);
  if (!image?.rgb || !params?.xy || !params?.scale) return out;
  const centerX = Number(params.xy[index * 2]);
  const centerY = Number(params.xy[index * 2 + 1]);
  const sx = Math.max(1e-6, Number(params.scale[index * 2]));
  const sy = Math.max(1e-6, Number(params.scale[index * 2 + 1]));
  const theta = Number(params.theta?.[index]) || 0;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);
  const sample = new Float32Array(3);
  const target = new Float64Array(3);
  let weightSum = 0;
  for (const [localX, localY, weight] of FOOTPRINT_COLOR_SAMPLES) {
    const x = centerX + localX * sx * cosTheta - localY * sy * sinTheta;
    const y = centerY + localX * sx * sinTheta + localY * sy * cosTheta;
    sampleImageRgbBilinear(image, x, y, sample);
    for (let channel = 0; channel < 3; channel += 1) target[channel] += sample[channel] * weight;
    weightSum += weight;
  }
  for (let channel = 0; channel < 3; channel += 1) out[channel] = target[channel] / Math.max(weightSum, 1e-8);
  return out;
}

function hardZeroContributionPlan(
  params,
  importanceData,
  maxPruneFraction = OPAQUE_PAINT_HARD_ZERO_MAX_FRACTION,
) {
  const count = params?.count || 0;
  if (
    count < MIN_COMPACTION_SPLATS ||
    !importanceData?.observedCoverage ||
    !importanceData?.integratedInfluence
  ) {
    return { applied: false, reason: "insufficient-current-visibility", before: count, after: count, removed: 0 };
  }
  // v1 uses its 10% default. v2 passes its user-selected total removal cap;
  // both remain bounded by the active-population floor below.
  const boundedFraction = Math.max(0, Math.min(1, maxPruneFraction));
  const maxRemove = Math.max(0, Math.min(
    Math.floor(count * boundedFraction),
    count - MIN_COMPACTION_SPLATS,
  ));
  const depthBinCount = 4096;
  const depthBins = new Uint32Array(depthBinCount);
  const removalMask = new Uint8Array(count);
  let candidateCount = 0;
  for (let index = 0; index < count; index += 1) {
    if (index < (params.flowBackcoatCount || 0)) continue;
    const observed = Math.max(0, Number(importanceData.observedCoverage[index]) || 0);
    const integratedInfluence = Math.max(0, Number(importanceData.integratedInfluence[index]) || 0);
    if (
      observed <= OPAQUE_PAINT_HARD_ZERO_EPSILON &&
      integratedInfluence <= OPAQUE_PAINT_HARD_ZERO_EPSILON
    ) {
      candidateCount += 1;
      const depth = Math.max(0, Math.min(1, Number(params.depthOrder?.[index]) || 0));
      depthBins[Math.min(depthBinCount - 1, Math.floor(depth * depthBinCount))] += 1;
    }
  }
  if (!candidateCount) {
    return { applied: false, reason: "no-current-hard-zero", before: count, after: count, removed: 0 };
  }
  const selectedCount = Math.min(candidateCount, maxRemove);
  if (selectedCount <= 0) {
    return { applied: false, reason: "minimum-count-guard", before: count, after: count, removed: 0 };
  }
  let thresholdBin = 0;
  let selectedBeforeThreshold = 0;
  while (
    thresholdBin < depthBinCount - 1 &&
    selectedBeforeThreshold + depthBins[thresholdBin] < selectedCount
  ) {
    selectedBeforeThreshold += depthBins[thresholdBin];
    thresholdBin += 1;
  }
  let thresholdQuota = selectedCount - selectedBeforeThreshold;
  for (let index = 0; index < count; index += 1) {
    if (index < (params.flowBackcoatCount || 0)) continue;
    const observed = Math.max(0, Number(importanceData.observedCoverage[index]) || 0);
    const integratedInfluence = Math.max(0, Number(importanceData.integratedInfluence[index]) || 0);
    if (
      observed > OPAQUE_PAINT_HARD_ZERO_EPSILON ||
      integratedInfluence > OPAQUE_PAINT_HARD_ZERO_EPSILON
    ) continue;
    const depth = Math.max(0, Math.min(1, Number(params.depthOrder?.[index]) || 0));
    const bin = Math.min(depthBinCount - 1, Math.floor(depth * depthBinCount));
    if (bin < thresholdBin || (bin === thresholdBin && thresholdQuota-- > 0)) removalMask[index] = 1;
  }
  const keepIndices = new Uint32Array(count - selectedCount);
  let keepIndex = 0;
  for (let index = 0; index < count; index += 1) {
    if (!removalMask[index]) keepIndices[keepIndex++] = index;
  }
  return {
    applied: true,
    reason: "current-hard-zero",
    before: count,
    after: keepIndices.length,
    removed: selectedCount,
    removed_ratio: selectedCount / count,
    hard_zero_candidates: candidateCount,
    hard_zero_candidate_ratio: candidateCount / count,
    max_prune_fraction: boundedFraction,
    epsilon: OPAQUE_PAINT_HARD_ZERO_EPSILON,
    keepIndices,
  };
}

function currentContributionCompactionPlan(
  params,
  importanceData,
  {
    maxRemovalFraction = CURRENT_CONTRIBUTION_MAX_FRACTION,
    nearZeroMaxFraction = CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_FRACTION,
  } = {},
) {
  const count = params?.count || 0;
  if (
    count < MIN_COMPACTION_SPLATS ||
    !importanceData?.observedCoverage ||
    !importanceData?.integratedInfluence
  ) {
    return { applied: false, reason: "insufficient-current-contribution", before: count, after: count, removed: 0 };
  }
  const boundedMaxRemovalFraction = clampNumber(
    maxRemovalFraction,
    0,
    CURRENT_CONTRIBUTION_MAX_FRACTION,
    CURRENT_CONTRIBUTION_MAX_FRACTION,
  );
  const boundedNearZeroMaxFraction = clampNumber(
    nearZeroMaxFraction,
    0,
    boundedMaxRemovalFraction,
    CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_FRACTION,
  );
  const maxRemove = Math.max(0, Math.min(
    Math.floor(count * boundedMaxRemovalFraction),
    count - MIN_COMPACTION_SPLATS,
  ));
  const exactPlan = hardZeroContributionPlan(
    params,
    importanceData,
    boundedMaxRemovalFraction,
  );
  const exactKeep = exactPlan.applied ? exactPlan.keepIndices : null;
  const removeMask = new Uint8Array(count);
  let exactZeroCandidates = exactPlan.hard_zero_candidates || 0;
  let exactZeroRemoved = 0;
  if (exactKeep) {
    removeMask.fill(1);
    for (const index of exactKeep) removeMask[index] = 0;
    exactZeroRemoved = exactPlan.removed;
  }
  const remainingBudget = Math.max(0, maxRemove - exactZeroRemoved);
  const maxNearZero = Math.max(0, Math.min(
    Math.floor(count * boundedNearZeroMaxFraction),
    remainingBudget,
  ));
  let nearZeroCandidates = 0;
  let selectedNearZero = 0;
  let nearZeroDetailProtected = 0;
  for (let index = 0; index < count; index += 1) {
    if (index < (params.flowBackcoatCount || 0)) continue;
    const observed = Math.max(0, Number(importanceData.observedCoverage[index]) || 0);
    const influence = Math.max(0, Number(importanceData.integratedInfluence[index]) || 0);
    if (removeMask[index]) continue;
    if (
      observed <= CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_COVERAGE &&
      influence <= CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_INFLUENCE
    ) {
      nearZeroCandidates += 1;
      if (Math.floor(Number(params.detailTags?.[index]) || 0) >= 2) {
        nearZeroDetailProtected += 1;
        continue;
      }
      if (selectedNearZero < maxNearZero) {
        selectedNearZero += 1;
        removeMask[index] = 1;
      }
    }
  }
  const requestedRemoved = exactZeroRemoved + selectedNearZero;
  if (!requestedRemoved) {
    return {
      applied: false,
      reason: "no-current-zero-or-near-zero",
      before: count,
      after: count,
      removed: 0,
      exact_zero_candidates: 0,
      near_zero_candidates: nearZeroCandidates,
      near_zero_detail_protected: nearZeroDetailProtected,
    };
  }
  const keepIndices = new Uint32Array(count - requestedRemoved);
  let keepIndex = 0;
  for (let index = 0; index < count; index += 1) {
    if (!removeMask[index]) keepIndices[keepIndex++] = index;
  }
  return {
    applied: requestedRemoved > 0,
    reason: "current-contribution-exact-and-near-zero",
    before: count,
    after: keepIndices.length,
    removed: requestedRemoved,
    removed_ratio: requestedRemoved / count,
    max_remove_fraction: boundedMaxRemovalFraction,
    exact_zero_candidates: exactZeroCandidates,
    exact_zero_removed: exactZeroRemoved,
    near_zero_candidates: nearZeroCandidates,
    near_zero_removed: selectedNearZero,
    near_zero_detail_protected: nearZeroDetailProtected,
    near_zero_max_fraction: boundedNearZeroMaxFraction,
    near_zero_max_coverage: CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_COVERAGE,
    near_zero_max_influence: CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_INFLUENCE,
    epsilon: OPAQUE_PAINT_HARD_ZERO_EPSILON,
    keepIndices,
  };
}
