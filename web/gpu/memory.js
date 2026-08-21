function bytesToMB(bytes) {
  return bytes / MB;
}

function formatMB(bytes) {
  return `${bytesToMB(bytes).toFixed(bytes < 10 * MB ? 1 : 0)} MB`;
}

function updateGpuMemoryStatus() {
  const training = state.webgpu.renderer?.trainingMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
  const result = state.webgpu.renderer?.resultRenderMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
  state.gpuMemory.trainingActiveBytes = Math.max(0, Math.round(training.activeBytes || 0));
  state.gpuMemory.trainingReservedBytes = Math.max(
    state.gpuMemory.trainingActiveBytes,
    Math.round(training.reservedBytes || 0),
  );
  state.gpuMemory.resultActiveBytes = Math.max(0, Math.round(result.activeBytes || 0));
  state.gpuMemory.resultReservedBytes = Math.max(
    state.gpuMemory.resultActiveBytes,
    Math.round(result.reservedBytes || 0),
  );
  state.gpuMemory.activeBytes = Math.max(
    0,
    state.gpuMemory.trainingActiveBytes + state.gpuMemory.resultActiveBytes,
  );
  state.gpuMemory.reservedBytes = Math.max(
    state.gpuMemory.activeBytes,
    state.gpuMemory.trainingReservedBytes + state.gpuMemory.resultReservedBytes,
  );
  const activeMB = bytesToMB(state.gpuMemory.activeBytes);
  const reservedMB = bytesToMB(state.gpuMemory.reservedBytes);
  const digits = Math.max(activeMB, reservedMB) < 10 ? 1 : 0;
  els.gpuMemoryText.textContent = `${activeMB.toFixed(digits)} / ${reservedMB.toFixed(digits)} MB`;
}

function updateCapacityStatus() {
  const probe = state.capacityProbe;
  if (probe.status === "probing") {
    els.capacityProbeText.textContent = "Testing GPU capacity...";
  } else if (probe.fastPath && probe.selected > 0) {
    els.capacityProbeText.textContent = "Auto · fast path";
  } else if (probe.status === "passed" && probe.selected > 0) {
    els.capacityProbeText.textContent = `Auto · ${compactNumber(probe.selected)} verified`;
  } else if (probe.status === "fallback" && probe.selected > 0) {
    els.capacityProbeText.textContent = `Auto · reduced to ${compactNumber(probe.selected)}`;
  } else if (probe.status === "failed") {
    els.capacityProbeText.textContent = "Auto · failed";
  } else {
    els.capacityProbeText.textContent = els.capacityMode.value === "auto-probe" ? "Auto · ready" : "Use Max splats";
  }

  const trainState = state.webgpu.renderer?.trainState;
  if (trainState?.tileIndexCapacity) {
    const used = Number(state.metrics?.tile_counters?.total ?? trainState.tileIndexInitialReferences ?? 0);
    const ratio = used / Math.max(1, trainState.tileIndexCapacity);
    els.tileReserveText.textContent = `${compactNumber(used)} / ${compactNumber(trainState.tileIndexCapacity)} (${Math.round(ratio * 100)}%)`;
  } else {
    els.tileReserveText.textContent = "-";
  }

  const samples = state.metrics?.performance_trace || [];
  const measured = samples.find((sample) => Number.isFinite(sample.iterations_per_second) && sample.iterations_per_second > 0);
  if (measured) {
    const remaining = Math.max(0, (state.metrics.steps_requested || 0) - measured.step);
    const seconds = remaining / measured.iterations_per_second;
    els.measuredSpeedText.textContent = `${measured.iterations_per_second.toFixed(1)} it/s / ${formatDuration(seconds)}`;
  } else {
    els.measuredSpeedText.textContent = "-";
  }
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "-";
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function roundDownStep(value, step) {
  return Math.max(step, Math.floor(value / step) * step);
}

function pixelBytes() {
  // RGB target, source/render alpha, metrics/preview state, and three vec4 loss-gradient records.
  return 3 * 4 + 2 * 4 + 6 * 4 * 2 + 4 * 4 + 12 * 4;
}

function splatBytes(splats = 1) {
  // Ping-pong params, Adam/density state, exact backward gradient, and tile index allowance.
  return 256 + EXACT_GRADIENT_STRIDE * 4 + TILE_INDEX_FACTOR * 4;
}

function tileReferenceCountForParams(image, params) {
  if (!image || !params?.count) return 0;
  const tileCols = Math.ceil(image.width / TILE_SIZE);
  const tileRows = Math.ceil(image.height / TILE_SIZE);
  const useEwa = phase33Variants().ewa2x2;
  const pixelSigma = (MIP_PIXEL_SIGMA * 2) / Math.max(image.width, image.height);
  const pixelPaddingX = useEwa && image.width > 1 ? 0.5 / (image.width - 1) : 0;
  const pixelPaddingY = useEwa && image.height > 1 ? 0.5 / (image.height - 1) : 0;
  let total = 0;
  for (let i = 0; i < params.count; i += 1) {
    const sx = Math.max(0.0001, params.scale[i * 2]);
    const sy = Math.max(0.0001, params.scale[i * 2 + 1]);
    const effectiveX = useEwa ? sx : Math.hypot(sx, pixelSigma);
    const effectiveY = useEwa ? sy : Math.hypot(sy, pixelSigma);
    const theta = params.theta?.[i] || 0;
    const c = Math.abs(Math.cos(theta));
    const s = Math.abs(Math.sin(theta));
    const radiusX = RENDER_SIGMA * (c * effectiveX + s * effectiveY) + pixelPaddingX;
    const radiusY = RENDER_SIGMA * (s * effectiveX + c * effectiveY) + pixelPaddingY;
    const centerX = params.xy[i * 2];
    const centerY = params.xy[i * 2 + 1];
    const minNormX = Math.max(-1, centerX - radiusX);
    const minNormY = Math.max(-1, centerY - radiusY);
    const maxNormX = Math.min(1, centerX + radiusX);
    const maxNormY = Math.min(1, centerY + radiusY);
    const minPxX = Math.floor((minNormX * 0.5 + 0.5) * (image.width - 1));
    const minPxY = Math.floor((minNormY * 0.5 + 0.5) * (image.height - 1));
    const maxPxX = Math.ceil((maxNormX * 0.5 + 0.5) * (image.width - 1));
    const maxPxY = Math.ceil((maxNormY * 0.5 + 0.5) * (image.height - 1));
    const minTileX = Math.max(0, Math.min(tileCols - 1, Math.floor(minPxX / TILE_SIZE)));
    const minTileY = Math.max(0, Math.min(tileRows - 1, Math.floor(minPxY / TILE_SIZE)));
    const maxTileX = Math.max(0, Math.min(tileCols - 1, Math.floor(maxPxX / TILE_SIZE)));
    const maxTileY = Math.max(0, Math.min(tileRows - 1, Math.floor(maxPxY / TILE_SIZE)));
    total += (maxTileX - minTileX + 1) * (maxTileY - minTileY + 1);
  }
  return total;
}

function tileIndexCapacityLimit(device) {
  const limits = device?.limits || state.webgpu.limits || {};
  const maxBuffer = limitNumber(limits, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(limits, "maxStorageBufferBindingSize", 128 * MB);
  const budgetShare = Math.max(4, Math.floor(memoryBudgetBytes() * 0.25));
  return Math.max(1, Math.min(TILE_OFFSET_VALUE_MASK, Math.floor(Math.min(maxBuffer, maxStorage, budgetShare) / 4)));
}

function plannedTileIndexCapacity(image, params, bufferCapacity, device) {
  const tileCount = Math.ceil(image.width / TILE_SIZE) * Math.ceil(image.height / TILE_SIZE);
  const observed = tileReferenceCountForParams(image, params);
  const observedPerSplat = observed / Math.max(1, params.count);
  const projectedPerSplat = Math.max(TILE_INDEX_FACTOR, observedPerSplat * TILE_INDEX_INITIAL_HEADROOM);
  const requested = Math.ceil(Math.min(bufferCapacity * tileCount, bufferCapacity * projectedPerSplat));
  const normalCapacity = Math.max(1, Math.min(requested, tileIndexCapacityLimit(device)));
  const qaCapacity = qaTileIndexCapacityOverride();
  return {
    capacity: qaCapacity === null ? normalCapacity : Math.min(normalCapacity, qaCapacity),
    observed,
    observedPerSplat,
    requested,
    qaForcedCapacity: qaCapacity,
  };
}

function trainingBufferDescriptors(
  image,
  params,
  capacity,
  device = state.webgpu.renderer?.device,
  prepared = {},
) {
  const tilePlan = prepared.tilePlan || plannedTileIndexCapacity(image, params, capacity, device);
  const tileCount = Math.ceil(image.width / TILE_SIZE) * Math.ceil(image.height / TILE_SIZE);
  const variants = phase33Variants();
  const preparedStages = Object.hasOwn(prepared, "coarseImage");
  const stageDimensions = preparedStages
    ? null
    : curriculumStageDimensions(image.width, image.height, variants);
  const coarseImage = preparedStages ? prepared.coarseImage || null : null;
  const midImage = preparedStages ? prepared.midImage || null : null;
  const stageRgbBytes = (stageImage, dimensions) => stageImage?.rgb?.byteLength ||
    (dimensions ? dimensions.width * dimensions.height * 3 * 4 : 0);
  const stageAlphaBytes = (stageImage, dimensions) => stageImage?.alpha?.byteLength ||
    (dimensions ? dimensions.width * dimensions.height * 4 : 0);
  const coarseRgbBytes = stageRgbBytes(coarseImage, stageDimensions?.coarse);
  const midRgbBytes = stageRgbBytes(midImage, stageDimensions?.mid);
  const coarseAlphaBytes = stageAlphaBytes(coarseImage, stageDimensions?.coarse);
  const midAlphaBytes = stageAlphaBytes(midImage, stageDimensions?.mid);
  const optimizerStride = 96;
  const profileEnabled = Boolean(state.webgpu.renderer?.performanceProfile?.timestampQuery);
  const descriptors = [];
  const add = (name, size, storage = false) => {
    if (size > 0) descriptors.push({ name, size: Math.max(4, Math.ceil(size / 4) * 4), storage });
  };
  add("config", TRAIN_CONFIG_BYTES, true);
  add("batch-config", TRAIN_BATCH_CONFIG_BYTES);
  add("present-config", 16);
  add("target-rgb", image.rgb.byteLength, true);
  add("coarse-target-rgb", coarseRgbBytes, true);
  add("mid-target-rgb", midRgbBytes, true);
  add("target-alpha", image.alpha?.byteLength || image.width * image.height * 4, true);
  add("coarse-target-alpha", coarseAlphaBytes, true);
  add("mid-target-alpha", midAlphaBytes, true);
  add("error-map", image.width * image.height * 4, true);
  add("stats", capacity * 2 * 4 * 4, true);
  add("density-control", (
    capacity * 4 +
    DENSITY_EVENT_SLOTS + 1 +
    Math.ceil(capacity / 256) * 2 +
    PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE +
    residualTileControlWords(image)
  ) * 4, true);
  add("tile-counts", tileCount * 4, true);
  add("tile-offsets", (tileCount + 1) * 4, true);
  add("tile-cursors", tileCount * 4, true);
  add("tile-indices", tilePlan.capacity * 4, true);
  add("tile-control", 16, true);
  add("pixel-state", image.width * image.height * 16, true);
  add("alpha-state", image.width * image.height * ALPHA_STATE_BYTES_PER_PIXEL, true);
  add("loss-gradient", image.width * image.height * 48, true);
  add("exact-gradient", capacity * EXACT_GRADIENT_STRIDE * 4, true);
  add("ssim-tiles", ssimWorkingBufferBytes(image), true);
  add("optimizer-state", capacity * optimizerStride, true);
  add("xy-depth", capacity * 4 * 4, true);
  add("transform", capacity * 4 * 4, true);
  add("color", capacity * 4 * 4, true);
  add("readback", capacity * 12 * 4);
  add("growth-signal-readback", 4);
  if (profileEnabled) {
    add("exact-backward-telemetry", EXACT_BACKWARD_TELEMETRY_BYTES, true);
    add("exact-backward-telemetry-readback", EXACT_BACKWARD_TELEMETRY_BYTES);
    add("profile-resolve", PERFORMANCE_PROFILE_QUERY_CAPACITY * 8);
    add("profile-readback", PERFORMANCE_PROFILE_QUERY_CAPACITY * 8);
  }
  const segmentedPartialBytes = tilePlan.capacity * EXACT_GRADIENT_STRIDE * 4;
  const segmentedAuxiliaryBytes =
    capacity * 4 +
    (capacity + 1) * 4 +
    capacity * 4 +
    tilePlan.capacity * 4;
  const baseReservedBytes = descriptors.reduce((sum, item) => sum + item.size, 0);
  const segmentedRequested = performanceVariants().segmentedExactBackward;
  const segmentedSupported =
    segmentedRequested &&
    Number(device?.limits?.maxStorageBuffersPerShaderStage || 8) >= 9 &&
    segmentedPartialBytes <= SEGMENTED_EXACT_BACKWARD_MAX_BYTES &&
    baseReservedBytes + segmentedPartialBytes + segmentedAuxiliaryBytes <= memoryBudgetBytes() * 0.9;
  const fixedPointRequested = performanceVariants().fixedPointExactGradient;
  const fixedPointSupported = fixedPointRequested && !segmentedSupported;
  if (segmentedSupported) {
    add("segmented-partial-gradient", segmentedPartialBytes, true);
    add("segmented-reference-counts", capacity * 4, true);
    add("segmented-reference-offsets", (capacity + 1) * 4, true);
    add("segmented-reference-cursors", capacity * 4, true);
    add("segmented-references", tilePlan.capacity * 4, true);
  }
  if (fixedPointSupported) {
    add("fixed-point-gradient-control", FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES, true);
    add("fixed-point-gradient-readback", FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES);
  }
  return {
    descriptors,
    tilePlan,
    coarseImage,
    midImage,
    segmentedExactBackward: {
      requested: segmentedRequested,
      enabled: segmentedSupported,
      partialBytes: segmentedSupported ? segmentedPartialBytes : 0,
      auxiliaryBytes: segmentedSupported ? segmentedAuxiliaryBytes : 0,
      reason: !segmentedRequested
        ? "not-requested"
        : Number(device?.limits?.maxStorageBuffersPerShaderStage || 8) < 9
          ? "requires-9-storage-buffers"
          : segmentedPartialBytes > SEGMENTED_EXACT_BACKWARD_MAX_BYTES
            ? "partial-buffer-cap"
            : baseReservedBytes + segmentedPartialBytes + segmentedAuxiliaryBytes > memoryBudgetBytes() * 0.9
              ? "training-memory-budget"
              : "enabled",
    },
    fixedPointExactGradient: {
      requested: fixedPointRequested,
      enabled: fixedPointSupported,
      scale: FIXED_POINT_EXACT_GRADIENT_SCALE,
      reason: !fixedPointRequested
        ? "not-requested"
        : segmentedSupported
          ? "mutually-exclusive-with-segmented-backward"
          : "enabled",
    },
  };
}

function trainingAllocationPlan(
  image,
  params,
  capacity,
  device = state.webgpu.renderer?.device,
  prepared = {},
) {
  const descriptorPlan = trainingBufferDescriptors(image, params, capacity, device, prepared);
  const storageSizes = descriptorPlan.descriptors.filter((item) => item.storage).map((item) => item.size);
  const allSizes = descriptorPlan.descriptors.map((item) => item.size);
  const maxBuffer = limitNumber(device?.limits || state.webgpu.limits || {}, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(device?.limits || state.webgpu.limits || {}, "maxStorageBufferBindingSize", 128 * MB);
  const largestStorageBytes = Math.max(0, ...storageSizes);
  const largestBufferBytes = Math.max(0, ...allSizes);
  const reservedBytes = allSizes.reduce((sum, size) => sum + size, 0);
  return {
    capacity,
    tilePlan: descriptorPlan.tilePlan,
    descriptors: descriptorPlan.descriptors,
    reservedBytes,
    largestStorageBytes,
    largestBufferBytes,
    maxStorage,
    maxBuffer,
    withinBufferLimits: largestStorageBytes <= maxStorage && largestBufferBytes <= maxBuffer,
    withinBudget: reservedBytes <= memoryBudgetBytes() * 0.9,
    segmentedExactBackward: descriptorPlan.segmentedExactBackward,
    fixedPointExactGradient: descriptorPlan.fixedPointExactGradient,
  };
}

function trainStateAllocatedDescriptors(trainState) {
  if (!trainState) return [];
  const entries = [
    ["config", trainState.configBuffer],
    ["batch-config", trainState.batchConfigBuffer],
    ["present-config", trainState.presentConfigBuffer],
    ["target-rgb", trainState.targetBuffer],
    ["coarse-target-rgb", trainState.coarseTargetBuffer],
    ["mid-target-rgb", trainState.midTargetBuffer],
    ["target-alpha", trainState.targetAlphaBuffer],
    ["coarse-target-alpha", trainState.coarseTargetAlphaBuffer],
    ["mid-target-alpha", trainState.midTargetAlphaBuffer],
    ["error-map", trainState.errorMapBuffer],
    ["stats", trainState.statsBuffer],
    ["density-control", trainState.densityControlBuffer],
    ["tile-counts", trainState.tileCountsBuffer],
    ["tile-offsets", trainState.tileOffsetsBuffer],
    ["tile-cursors", trainState.tileCursorsBuffer],
    ["tile-indices", trainState.tileIndicesBuffer],
    ["tile-control", trainState.tileControlBuffer],
    ["pixel-state", trainState.pixelStateBuffer],
    ["alpha-state", trainState.alphaStateBuffer],
    ["loss-gradient", trainState.lossGradientBuffer],
    ["exact-gradient", trainState.exactGradientBuffer],
    ["fixed-point-gradient-control", trainState.fixedPointGradientControlBuffer],
    ["fixed-point-gradient-readback", trainState.fixedPointGradientReadbackBuffer],
    ["segmented-partial-gradient", trainState.segmentedPartialGradientBuffer],
    ["segmented-reference-counts", trainState.segmentedReferenceCountsBuffer],
    ["segmented-reference-offsets", trainState.segmentedReferenceOffsetsBuffer],
    ["segmented-reference-cursors", trainState.segmentedReferenceCursorsBuffer],
    ["segmented-references", trainState.segmentedReferencesBuffer],
    ["ssim-tiles", trainState.ssimTileBuffer],
    ["optimizer-state", trainState.optimizerStateBuffer],
    ["xy-depth", trainState.xyBuffers?.[0]],
    ["transform", trainState.transformBuffers?.[0]],
    ["color", trainState.colorBuffers?.[0]],
    ["readback", trainState.readbackBuffer],
    ["growth-signal-readback", trainState.growthSignalReadbackBuffer],
    ["exact-backward-telemetry", trainState.exactBackwardTelemetryBuffer],
    ["exact-backward-telemetry-readback", trainState.exactBackwardTelemetryReadbackBuffer],
    ["profile-resolve", trainState.profileResolveBuffer],
    ["profile-readback", trainState.profileReadbackBuffer],
  ];
  return entries
    .filter(([, buffer]) => Boolean(buffer))
    .map(([name, buffer]) => ({ name, size: Number(buffer.size || 0) }));
}

function allocationDescriptorMismatch(plannedDescriptors, allocatedDescriptors) {
  const planned = new Map(plannedDescriptors.map(({ name, size }) => [name, Number(size)]));
  const allocated = new Map(allocatedDescriptors.map(({ name, size }) => [name, Number(size)]));
  const names = [...new Set([...planned.keys(), ...allocated.keys()])].sort();
  return names
    .filter((name) => planned.get(name) !== allocated.get(name))
    .map((name) => ({ name, planned: planned.get(name) ?? null, allocated: allocated.get(name) ?? null }));
}

function tileGrowthMemoryPlan({
  currentReservedBytes,
  currentTileBytes,
  nextTileBytes,
  budgetBytes = memoryBudgetBytes() * 0.9,
}) {
  const current = Math.max(0, Number(currentReservedBytes) || 0);
  const previous = Math.max(0, Number(currentTileBytes) || 0);
  const next = Math.max(0, Number(nextTileBytes) || 0);
  const budget = Math.max(0, Number(budgetBytes) || 0);
  const finalReservedBytes = Math.max(0, current - previous) + next;
  const transientReservedBytes = current + next;
  return {
    currentReservedBytes: current,
    currentTileBytes: previous,
    nextTileBytes: next,
    finalReservedBytes,
    transientReservedBytes,
    budgetBytes: budget,
    withinBudget: finalReservedBytes <= budget && transientReservedBytes <= budget,
  };
}

function capacityProbeCandidates(requested) {
  const capped = normalizeUiSplatCount(requested, DEFAULT_FINAL_SPLATS, MANUAL_SPLATS_MAX);
  const candidates = CAPACITY_PROBE_TIERS.filter((value) => value <= capped);
  if (!candidates.includes(capped)) candidates.push(capped);
  return [...new Set(candidates)].sort((a, b) => b - a);
}

function estimatedImageSizeFor(trainSize) {
  if (!state.image) return { width: trainSize, height: trainSize };
  const sourceWidth = state.image.cacheWidth || state.image.width;
  const sourceHeight = state.image.cacheHeight || state.image.height;
  const [width, height] = resizedSize(sourceWidth, sourceHeight, trainSize);
  return { width, height };
}

function imagePixelEstimate(trainSize) {
  const size = estimatedImageSizeFor(trainSize);
  return size.width * size.height;
}

function sideFromPixelBudget(pixelBudget, trainSize) {
  if (!state.image) return Math.sqrt(pixelBudget);
  const sourceWidth = state.image.cacheWidth || state.image.width;
  const sourceHeight = state.image.cacheHeight || state.image.height;
  const side = Math.max(1, sourceWidth, sourceHeight);
  const aspectPixels = Math.max(1, sourceWidth * sourceHeight) / (side * side);
  return Math.sqrt(pixelBudget / Math.max(0.01, aspectPixels));
}

function estimateGpuMemory(trainSize, splats) {
  const fullSize = estimatedImageSizeFor(trainSize);
  const pixels = fullSize.width * fullSize.height;
  const variants = phase33Variants();
  const curriculum = curriculumStageDimensions(fullSize.width, fullSize.height, variants);
  const coarseTargetBytes = curriculum.coarse ? curriculum.coarse.width * curriculum.coarse.height * 4 * 4 : 0;
  const midTargetBytes = curriculum.mid ? curriculum.mid.width * curriculum.mid.height * 4 * 4 : 0;
  // Ping-pong params, Adam moments, active/density state, and compact tile indices.
  const trainStateBytes = splats * splatBytes(splats);
  const targetBytes = pixels * 4 * 4;
  const metricsBytes = pixels * 6 * 4;
  const previewBytes = pixels * 4 * 4;
  const alphaStateBytes = pixels * ALPHA_STATE_BYTES_PER_PIXEL;
  const lossGradientBytes = pixels * 48;
  const exactGradientBytes = splats * EXACT_GRADIENT_STRIDE * 4;
  const tileCount = Math.ceil(Math.sqrt(pixels) / TILE_SIZE) ** 2;
  const tileScratchBytes = tileCount * 3 * 4 + (tileCount + 1) * 4 + 16;
  const overlapDiagnosticBytes = phase40Variants().overlapDiagnostics ? pixels * 8 : 0;
  const peakBytes = trainStateBytes + targetBytes + coarseTargetBytes + midTargetBytes + alphaStateBytes + lossGradientBytes + metricsBytes * 2 + previewBytes * 2 + overlapDiagnosticBytes + tileScratchBytes + splats * 32;
  return { pixels, trainStateBytes, targetBytes, coarseTargetBytes, midTargetBytes, alphaStateBytes, lossGradientBytes, exactGradientBytes, metricsBytes, previewBytes, overlapDiagnosticBytes, peakBytes };
}

function browserMemoryHintBytes() {
  const deviceMemoryGB = Number(navigator.deviceMemory);
  if (Number.isFinite(deviceMemoryGB) && deviceMemoryGB > 0) {
    return { bytes: deviceMemoryGB * GB, source: `deviceMemory ${deviceMemoryGB}GB`, exact: false };
  }
  const heapLimit = Number(performance?.memory?.jsHeapSizeLimit);
  if (Number.isFinite(heapLimit) && heapLimit > 0) {
    return { bytes: heapLimit, source: "jsHeapSizeLimit", exact: false };
  }
  return { bytes: 0, source: "WebGPU limits only", exact: false };
}

function autoBudgetInfo(limits) {
  const maxBuffer = limitNumber(limits, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(limits, "maxStorageBufferBindingSize", 128 * MB);
  const browserMemory = browserMemoryHintBytes();
  const webgpuEnvelope = Math.min(2 * GB, maxBuffer * 8, maxStorage * 16);
  const hintedWorkingSet = browserMemory.bytes > 0 ? browserMemory.bytes * 0.125 : 256 * MB;
  const budgetBytes = clampNumber(Math.min(hintedWorkingSet, webgpuEnvelope), 128 * MB, 2 * GB);
  const reservedBytes = browserMemory.bytes > 0 ? Math.max(0, browserMemory.bytes - budgetBytes) : 0;
  return {
    budgetBytes,
    reservedBytes,
    exact: false,
    source: browserMemory.source,
    webgpuEnvelope,
  };
}

function memoryBudgetInfo(limits) {
  const automatic = autoBudgetInfo(limits);
  if (!state.safety.memoryLimiterUnlocked) {
    return { ...automatic, mode: "automatic", multiplier: 1 };
  }
  const budgetBytes = Math.min(
    MEMORY_LIMITER_UNLOCK_MAX_BYTES,
    automatic.budgetBytes * MEMORY_LIMITER_UNLOCK_MULTIPLIER,
  );
  const memoryHint = browserMemoryHintBytes();
  return {
    ...automatic,
    budgetBytes,
    reservedBytes: memoryHint.bytes > 0 ? Math.max(0, memoryHint.bytes - budgetBytes) : 0,
    webgpuEnvelope: Math.min(
      MEMORY_LIMITER_UNLOCK_MAX_BYTES,
      automatic.webgpuEnvelope * MEMORY_LIMITER_UNLOCK_MULTIPLIER,
    ),
    mode: "unlocked",
    multiplier: MEMORY_LIMITER_UNLOCK_MULTIPLIER,
  };
}

function memoryBudgetBytes() {
  return memoryBudgetInfo(state.webgpu.limits).budgetBytes;
}

function computeBudgetFor(trainSize, finalSplats, steps) {
  const limits = state.webgpu.limits || {};
  trainSize = Math.round(clampNumber(trainSize, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE));
  finalSplats = normalizeUiSplatCount(finalSplats, DEFAULT_FINAL_SPLATS);
  steps = normalizeStepInteger(steps, { min: LIMITS.stepsMin, max: LIMITS.stepsMax, fallback: DEFAULT_ITERATIONS });
  const budgetInfo = memoryBudgetInfo(limits);
  const budgetBytes = budgetInfo.budgetBytes;
  const memoryHint = browserMemoryHintBytes();
  const autoBudget = autoBudgetInfo(limits);
  const current = estimateGpuMemory(trainSize, finalSplats);
  const maxBuffer = limitNumber(limits, "maxBufferSize", 256 * MB);
  const maxStorage = limitNumber(limits, "maxStorageBufferBindingSize", 128 * MB);
  const maxTexture = limitNumber(limits, "maxTextureDimension2D", LIMITS.trainSizeMax);
  const hardBufferBytes = Math.min(maxBuffer, maxStorage);
  const largestPixelStorageBytes = Math.max(
    current.targetBytes,
    current.alphaStateBytes,
    current.lossGradientBytes,
    current.metricsBytes,
    current.previewBytes,
    current.overlapDiagnosticBytes,
  );
  const largestPixelBytesPerPixel = Math.max(1, Math.ceil(largestPixelStorageBytes / Math.max(1, current.pixels)));
  const hardPixelLimit = Math.floor(hardBufferBytes / largestPixelBytesPerPixel);
  const softPixelLimit = Math.floor((budgetBytes - finalSplats * splatBytes()) / pixelBytes());
  const pixelBudget = Math.max(1, Math.min(hardPixelLimit, softPixelLimit));
  const limiterTrainSize = Math.min(
    LIMITS.trainSizeMax,
    maxTexture,
    Math.max(LIMITS.trainSizeMin, roundDownStep(sideFromPixelBudget(Math.max(1, pixelBudget), trainSize), 16)),
  );
  const splatBudget = Math.floor((budgetBytes - current.pixels * pixelBytes()) / splatBytes());
  const limiterFinalSplats = Math.max(
    LIMITS.splatsMin,
    Math.min(LIMITS.splatsMax, roundDownStep(Math.max(LIMITS.splatsMin, splatBudget), 4)),
  );
  const recommendedTrainSize = Math.min(trainSize, limiterTrainSize);
  const recommendedFinalSplats = Math.min(finalSplats, limiterFinalSplats);
  const previewWork = current.pixels * finalSplats;
  const trainWork = finalSplats * steps;
  const rawMetricInterval = Math.ceil((steps / 60) * Math.max(1, previewWork / 70_000_000));
  const metricInterval = clampNumber(rawMetricInterval, 1, DEFAULT_MAX_METRIC_INTERVAL, 1);
  const headroomBytes = budgetBytes - current.peakBytes;
  return {
    budgetBytes,
    budgetMB: bytesToMB(budgetBytes).toFixed(0),
    headroomBytes,
    headroomMB: bytesToMB(headroomBytes).toFixed(1),
    memoryHintSource: memoryHint.source,
    memoryHintMB: memoryHint.bytes > 0 ? bytesToMB(memoryHint.bytes).toFixed(0) : "",
    memoryBudgetMode: budgetInfo.mode,
    memoryBudgetMultiplier: budgetInfo.multiplier,
    autoBudgetExact: false,
    autoBudgetSource: autoBudget.source,
    autoReservedMB: autoBudget.reservedBytes > 0 ? bytesToMB(autoBudget.reservedBytes).toFixed(0) : "",
    autoEnvelopeMB: bytesToMB(autoBudget.webgpuEnvelope).toFixed(0),
    estimatedBytes: current.peakBytes,
    estimatedMB: bytesToMB(current.peakBytes).toFixed(1),
    metricsBytes: current.metricsBytes,
    targetBytes: current.targetBytes,
    trainStateBytes: current.trainStateBytes,
    maxBuffer,
    maxStorage,
    maxTexture,
    limiterTrainSize,
    limiterFinalSplats,
    recommendedTrainSize,
    recommendedFinalSplats,
    metricInterval,
    previewWork,
    trainWork,
    overBudget: current.peakBytes > budgetBytes,
    overHardLimit: largestPixelStorageBytes > hardBufferBytes,
  };
}

function updateImageSizeStatus() {
  if (!state.image) {
    els.imageSizeText.textContent = "-";
    return;
  }
  const cacheWidth = state.image.cacheWidth || state.image.width;
  const cacheHeight = state.image.cacheHeight || state.image.height;
  els.imageSizeText.textContent = `${cacheWidth}x${cacheHeight}`;
}

function computeRecommendation() {
  const trainSize = Math.round(clampNumber(els.trainSize.value, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE));
  const finalSplats = normalizeUiSplatCount(els.finalSplatCount.value, DEFAULT_FINAL_SPLATS);
  const steps = normalizeStepInteger(els.stepCount.value, { min: LIMITS.stepsMin, max: LIMITS.stepsMax, fallback: DEFAULT_ITERATIONS });
  return computeBudgetFor(trainSize, finalSplats, steps);
}

function safetyFailure(rec, context) {
  if (rec.overHardLimit) return { context, reason: "safety_stop_hard_limit", rec };
  if (rec.overBudget) return { context, reason: "safety_stop_budget", rec };
  return null;
}

function setSafetyStop(failure) {
  const rec = failure.rec;
  state.safety.lastStopReason = failure.reason;
  state.safety.lastStopEstimateMB = rec.estimatedMB;
  state.safety.lastStopBudgetMB = rec.budgetMB;
  state.safety.lastRecommended = `${rec.recommendedTrainSize}px/${rec.recommendedFinalSplats}`;
  els.budgetNote.textContent = `Safety stop: ${failure.reason}. ${rec.estimatedMB}/${rec.budgetMB} MB, try ${state.safety.lastRecommended}.`;
  log(`safety stop ${failure.context}: ${failure.reason} estimate=${rec.estimatedMB}MB budget=${rec.budgetMB}MB recommended=${state.safety.lastRecommended}`);
  publishState();
}

function runtimeSafetyError(reason, context, details = {}) {
  const rec = state.recommendation || computeRecommendation();
  const failure = { reason, context, rec };
  if (state.metrics) {
    state.metrics.stopped = true;
    state.metrics.safety_stop = {
      reason,
      context,
      estimated_mb: rec.estimatedMB,
      budget_mb: rec.budgetMB,
      ...details,
    };
  }
  setSafetyStop(failure);
  const error = new Error(`${reason}: ${context}`);
  error.safetyStop = true;
  return error;
}

function clearSafetyStop() {
  state.safety.lastStopReason = "";
  state.safety.lastStopEstimateMB = "";
  state.safety.lastStopBudgetMB = "";
  state.safety.lastRecommended = "";
}

function setLimiterAttributes(rec) {
  els.trainSize.max = String(LIMITS.trainSizeMax);
  const splatLimit = state.image
    ? normalizeUiSplatCount(Math.min(MANUAL_SPLATS_MAX, rec.limiterFinalSplats), DEFAULT_FINAL_SPLATS)
    : MANUAL_SPLATS_MAX;
  els.initialSplatCount.max = String(Math.min(CAPACITY_PROBE_FAST_PATH_MAX, splatLimit));
  els.finalSplatCount.max = String(splatLimit);
}

function applyDeviceLimiter(rec, { reconcileSplatCounts = true } = {}) {
  setLimiterAttributes(rec);
  if (state.running) return false;
  if (!state.image) return false;

  const effectiveLimit = normalizeUiSplatCount(
    Math.min(MANUAL_SPLATS_MAX, rec.limiterFinalSplats),
    DEFAULT_FINAL_SPLATS,
  );
  if (Number(els.finalSplatCount.value) > effectiveLimit) {
    els.finalSplatCount.value = String(effectiveLimit);
    return true;
  }
  if (reconcileSplatCounts && Number(els.initialSplatCount.value) > Number(els.finalSplatCount.value)) {
    els.initialSplatCount.value = String(els.finalSplatCount.value);
    return true;
  }
  return false;
}

function syncMemoryLimiterUnlockUi() {
  const unlocked = state.safety.memoryLimiterUnlocked;
  els.memoryLimiterUnlock.setAttribute("aria-pressed", String(unlocked));
  els.memoryLimiterUnlock.textContent = `Memory limiter unlock: ${unlocked ? "ON" : "OFF"}`;
}

function updateMemoryRecommendation({ reconcileSplatCounts = true } = {}) {
  syncMemoryLimiterUnlockUi();
  let rec = computeRecommendation();
  for (let i = 0; i < 3 && applyDeviceLimiter(rec, { reconcileSplatCounts }); i += 1) {
    rec = computeRecommendation();
  }
  setLimiterAttributes(rec);
  state.recommendation = rec;
  els.gpuLimitText.textContent = `buf ${formatMB(rec.maxBuffer)} / stor ${formatMB(rec.maxStorage)} / tex ${rec.maxTexture}`;
  els.memoryEstimateText.textContent = `${rec.estimatedMB}/${rec.budgetMB} MB`;
  els.headroomText.textContent = `${rec.headroomMB} MB`;
  els.limiterText.textContent = `${rec.limiterTrainSize}px / ${rec.limiterFinalSplats}`;
  els.recommendationText.textContent = `${rec.recommendedTrainSize}px / ${rec.recommendedFinalSplats}`;
  els.speedEstimateText.textContent = `${compactNumber(rec.previewWork)} / metrics ${rec.metricInterval}`;
  const detailCap = recommendedDetailShape(state.image || { width: Number(els.trainSize.value) || 1024, height: Number(els.trainSize.value) || 1024 });
  els.detailCapText.textContent = `${detailCap.minorPx.toFixed(1)} x ${detailCap.majorPx.toFixed(1)}px / ${detailCap.maxAnisotropy.toFixed(1)}:1`;
  els.budgetNote.textContent = rec.overHardLimit
    ? "Current settings exceed a WebGPU buffer limit."
    : rec.overBudget
      ? "Current settings exceed the safety budget; apply recommended or lower values."
      : state.safety.memoryLimiterUnlocked
        ? `Unlocked mode uses a ${rec.memoryBudgetMultiplier}x app working-set budget (${rec.budgetMB} MB). WebGPU hard limits still apply; browser or device termination is more likely.`
        : `Auto uses ${rec.autoBudgetSource} and WebGPU limits; ${rec.autoReservedMB ? `${rec.autoReservedMB} MB remains outside the app estimate` : "a conservative fallback is active"}. Exact free VRAM is unavailable.`;
  publishState();
  return rec;
}

