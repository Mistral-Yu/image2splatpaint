function emptyFusionEvents() {
  return {
    adc_split: 0,
    adc_duplicate: 0,
    adc_recycle: 0,
    mcmc_teleport: 0,
    mcmc_reseed: 0,
    inactive_reused: 0,
    prune: 0,
    opacity_reset: 0,
    importance_protected: 0,
    adc_eligible: 0,
    adc_fallback: 0,
    structure_guided: 0,
    nonfinite_stats: 0,
    adc_low_to_high: 0,
    adc_high_to_low: 0,
    adc_same_band: 0,
    source_claim_conflicts: 0,
    source_claims: 0,
    tilt_risk_candidates: 0,
    tilt_true_splits: 0,
    tilt_opacity_saturations: 0,
    paint_outlier_recycle: 0,
    paint_outlier_recolor: 0,
    paint_outlier_trim: 0,
    surface_layer_candidates: 0,
    surface_layer_promotions: 0,
    harmful_rectangle_candidate_selections: 0,
    harmful_rectangle_front_oversized_selections: 0,
    harmful_rectangle_high_contribution_selections: 0,
    harmful_rectangle_high_deviation_selections: 0,
    harmful_rectangle_parent_replacements: 0,
    harmful_rectangle_children_created: 0,
  };
}

async function detectWebGpu() {
  if (!("gpu" in navigator)) {
    state.webgpu = { supported: false, renderer: null, reason: "navigator.gpu unavailable", limits: null, adapterInfo: null };
    els.backendText.textContent = "webgpu unavailable";
    updateMemoryRecommendation();
    log("WebGPU unavailable; training disabled");
    return false;
  }
  try {
    const profileRequested = performanceProfileRequested();
    const requested = await requestWebGpuDevice(navigator.gpu, {
      profileRequested,
      subgroupExactBackward: performanceVariants().subgroupExactBackward,
      requiredStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
      preferredStorageBuffersPerShaderStage: PREFERRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
    });
    const { device } = requested;
    const renderer = new WebGpuPreview(device, els.gpuCanvas, {
      profileRequested,
      timestampQuery: requested.timestampQuery,
      subgroupExactBackward: requested.subgroupExactBackward,
      tileCullingEnabled: () => Boolean(els.tileCullingToggle.checked),
      trainLayerOrderEnabled: () => Boolean(els.trainLayerOrder.checked),
      outsidePreviewEnabled: () => Boolean(els.outsidePreviewToggle.checked),
      onPresentedState: ({ width, height, kind }) => {
        document.documentElement.dataset.previewBufferWidth = String(width);
        document.documentElement.dataset.previewBufferHeight = String(height);
        document.documentElement.dataset.previewBufferKind = kind;
      },
    });
    state.webgpu = {
      supported: true,
      renderer,
      reason: "available",
      limits: requested.limits,
      adapterInfo: requested.adapterInfo,
      adapterFeatures: requested.adapterFeatures,
      profile: requested.profile,
      subgroups: requested.subgroups,
    };
    device.lost.then((info) => handleWebGpuDeviceLoss(renderer, info));
    els.backendText.textContent = "webgpu available";
    updateMemoryRecommendation();
    log("WebGPU available for train and preview");
    return true;
  } catch (error) {
    state.webgpu = { supported: false, renderer: null, reason: error.message, limits: null, adapterInfo: null };
    els.backendText.textContent = "webgpu unavailable";
    updateMemoryRecommendation();
    log(`WebGPU failed; training disabled: ${error.message}`);
    return false;
  }
}

function completedResultStatus() {
  if (!state.params || !state.image || !state.metrics) return "";
  if (!state.metrics.finished_at && !state.metrics.final_cpu_result_ready_at) return "";
  if (state.metrics.safety_stop) return "safety stopped";
  return state.metrics.stopped ? "stopped" : "done";
}

function cancelCompletedResultGpuRecovery() {
  if (state.webGpuRecoveryPending) invalidatePreviewRefresh();
  state.webGpuRecoveryGeneration += 1;
  state.webGpuRecoveryPending = false;
  state.webGpuRecoveryAttempts = 0;
}

async function recoverCompletedResultWebGpu(generation, terminalStatus) {
  const expectedParams = state.params;
  const expectedMetrics = state.metrics;
  const recoveryIsCurrent = () => (
    generation === state.webGpuRecoveryGeneration &&
    state.params === expectedParams &&
    state.metrics === expectedMetrics
  );
  state.webGpuRecoveryPending = true;
  publishLifecycleInteractionState();
  try {
    let recovered = false;
    for (let attempt = 1; attempt <= 2 && recoveryIsCurrent(); attempt += 1) {
      state.webGpuRecoveryAttempts = attempt;
      recovered = await detectWebGpu();
      if (!recoveryIsCurrent()) return;
      if (recovered && state.webgpu.renderer && !state.webgpu.renderer.deviceLost) break;
      await nextFrame();
      if (!recoveryIsCurrent()) return;
    }
    if (!recoveryIsCurrent()) return;
    if (!recovered || !state.webgpu.renderer || state.webgpu.renderer.deviceLost) {
      els.backendText.textContent = "webgpu unavailable";
      setStatus("gpu unavailable");
      setTrainingMessage("Training result preserved, but the GPU preview could not be restored.", "info");
      updateDownloads(false);
      log("GPU preview recovery failed; completed CPU result was preserved");
      return;
    }
    state.previewGeneration += 1;
    state.previewAppliedRevision = 0;
    const recoveryRenderer = state.webgpu.renderer;
    const uploaded = await recoveryRenderer.uploadResultRenderState(expectedParams);
    if (!recoveryIsCurrent()) {
      recoveryRenderer.disposeResultRenderState();
      return;
    }
    if (!uploaded) throw new Error("The preserved result could not be uploaded to the recovered GPU.");
    await refreshOutsidePreview({ recovery: true });
    if (!recoveryIsCurrent()) {
      recoveryRenderer.disposeResultRenderState();
      return;
    }
    state.webGpuRecoveryAttempts = 0;
    setStatus(terminalStatus);
    setTrainingMessage(
      terminalStatus === "stopped" ? "Training stopped. GPU preview restored." : "Training complete. GPU preview restored.",
      terminalStatus === "safety stopped" ? "error" : "success",
    );
    updateDownloads(!state.metrics?.safety_stop);
    if (state.metrics) {
      state.metrics.post_training_gpu_recoveries =
        (state.metrics.post_training_gpu_recoveries || 0) + 1;
    }
    log("GPU preview restored; completed training result preserved");
  } finally {
    if (recoveryIsCurrent()) {
      state.webGpuRecoveryPending = false;
      publishLifecycleInteractionState();
    }
  }
}

function handleWebGpuDeviceLoss(renderer, info = {}) {
  if (state.webgpu.renderer !== renderer) return;
  const terminalStatus = completedResultStatus();
  const reason = `device lost: ${info.reason || "unknown"}`;
  renderer.deviceLost = true;
  state.lastGpuLoss = { reason, at: new Date().toISOString(), after_training: Boolean(terminalStatus) };
  invalidatePreviewRefresh();
  state.splatAdjustmentEpoch += 1;
  window.clearTimeout(state.splatAdjustmentValidationTimer);
  if (state.splatAdjustmentFrame) window.cancelAnimationFrame(state.splatAdjustmentFrame);
  state.splatAdjustmentFrame = 0;
  state.splatAdjustmentValidationVersion = 0;
  invalidateTrainingRun("WebGPU device lost");
  state.paused = false;
  if (!terminalStatus) state.stopRequested = true;
  // Dependency synchronizers consult state.running when controls are enabled.
  // Clear it first so a device loss cannot leave next-run settings disabled.
  state.running = false;
  try {
    renderer.disposeTrainState();
    renderer.disposeResultRenderState();
  } catch {
    // A lost device may reject cleanup; CPU parameters remain authoritative.
  }
  state.webgpu = { supported: false, renderer: null, reason, limits: null, adapterInfo: null };
  els.backendText.textContent = terminalStatus ? "webgpu recovering" : "webgpu unavailable";
  setInputControlsDisabled(false);
  setPausedRuntimeControlsEnabled(false);
  els.pauseButton.disabled = true;
  els.pauseButton.textContent = "Pause";
  els.stopButton.disabled = true;
  updateDownloads(false);
  renderSplatInspector();
  updateMemoryRecommendation();
  if (terminalStatus) {
    state.stopRequested = false;
    const generation = ++state.webGpuRecoveryGeneration;
    setStatus("recovering gpu");
    setTrainingMessage("Training result preserved. Restoring the GPU preview...", "info");
    log("GPU preview connection lost after training; restoring it without discarding the result");
    publishState();
    recoverCompletedResultWebGpu(generation, terminalStatus).catch((error) => {
      if (generation !== state.webGpuRecoveryGeneration) return;
      state.webGpuRecoveryPending = false;
      setStatus("gpu unavailable");
      setTrainingMessage("Training result preserved, but the GPU preview could not be restored.", "info");
      log(`GPU preview recovery failed: ${error.message}`);
      publishLifecycleInteractionState();
    });
    return;
  }
  setTrainingMessage(`Training failed: ${reason}`, "error");
  setStatus("error");
  log(reason);
  publishState();
}


function rgbaParity(a, b) {
  if (a.length !== b.length) return { exact: false, max_abs: 255, mean_abs: Number.POSITIVE_INFINITY };
  let total = 0;
  let maximum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const delta = Math.abs(a[i] - b[i]);
    total += delta;
    maximum = Math.max(maximum, delta);
  }
  return { exact: maximum === 0, max_abs: maximum, mean_abs: total / Math.max(1, a.length) };
}

function displayRgbaParity(a, b) {
  const raw = rgbaParity(a, b);
  if (a.length !== b.length) {
    return {
      exact: false,
      display_equivalent: false,
      ...raw,
      alpha_max_abs: 255,
      premultiplied_max_abs: 255,
      premultiplied_mean_abs: Number.POSITIVE_INFINITY,
      premultiplied_tolerance: 1,
    };
  }
  let alphaMaximum = 0;
  let premultipliedMaximum = 0;
  let premultipliedTotal = 0;
  let premultipliedMismatchPixels = 0;
  let maximumPixel = null;
  let channels = 0;
  for (let i = 0; i < a.length; i += 4) {
    alphaMaximum = Math.max(alphaMaximum, Math.abs(a[i + 3] - b[i + 3]));
    let pixelMaximum = 0;
    for (let channel = 0; channel < 3; channel += 1) {
      const source = Math.round(a[i + channel] * a[i + 3] / 255);
      const decoded = Math.round(b[i + channel] * b[i + 3] / 255);
      const delta = Math.abs(source - decoded);
      pixelMaximum = Math.max(pixelMaximum, delta);
      if (delta > premultipliedMaximum) {
        maximumPixel = {
          pixel: i / 4,
          channel,
          training_rgba: Array.from(a.slice(i, i + 4)),
          standalone_rgba: Array.from(b.slice(i, i + 4)),
        };
      }
      premultipliedMaximum = Math.max(premultipliedMaximum, delta);
      premultipliedTotal += delta;
      channels += 1;
    }
    if (pixelMaximum > 1) premultipliedMismatchPixels += 1;
  }
  // The training buffer and the standalone render target quantize the same
  // float alpha through different 8-bit paths. A one-code alpha difference
  // can add one more code of premultiplied-color round-off, so allow two only
  // when that alpha difference is present. An alpha-identical two-code color
  // difference remains a real render mismatch and still blocks export.
  const premultipliedTolerance = alphaMaximum > 0 ? 2 : 1;
  const displayEquivalent =
    alphaMaximum <= 1 && premultipliedMaximum <= premultipliedTolerance;
  return {
    exact: displayEquivalent,
    display_equivalent: displayEquivalent,
    max_abs: raw.max_abs,
    mean_abs: raw.mean_abs,
    raw_exact: raw.exact,
    alpha_max_abs: alphaMaximum,
    premultiplied_max_abs: premultipliedMaximum,
    premultiplied_mean_abs: premultipliedTotal / Math.max(1, channels),
    premultiplied_tolerance: premultipliedTolerance,
    premultiplied_mismatch_pixels: premultipliedMismatchPixels,
    maximum_pixel: maximumPixel,
  };
}

function srgbSignalStatisticsFromRgb(rgb, alpha = null) {
  const pixelCount = Math.floor(rgb.length / 3);
  const channelSum = [0, 0, 0];
  let lumaSum = 0;
  let alphaSum = 0;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const r = Math.max(0, Math.min(1, Number(rgb[pixel * 3]) || 0));
    const g = Math.max(0, Math.min(1, Number(rgb[pixel * 3 + 1]) || 0));
    const b = Math.max(0, Math.min(1, Number(rgb[pixel * 3 + 2]) || 0));
    channelSum[0] += r;
    channelSum[1] += g;
    channelSum[2] += b;
    lumaSum += r * 0.2126 + g * 0.7152 + b * 0.0722;
    alphaSum += Math.max(0, Math.min(1, Number(alpha?.[pixel] ?? 1)));
  }
  return {
    pixels: pixelCount,
    mean_rgb: channelSum.map((sum) => sum / Math.max(1, pixelCount)),
    mean_srgb_signal_luma: lumaSum / Math.max(1, pixelCount),
    mean_alpha: alphaSum / Math.max(1, pixelCount),
  };
}

function srgbSignalStatisticsFromRgba(rgba) {
  const pixelCount = Math.floor(rgba.length / 4);
  const rgb = new Float32Array(pixelCount * 3);
  const alpha = new Float32Array(pixelCount);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    rgb[pixel * 3] = rgba[pixel * 4] / 255;
    rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1] / 255;
    rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2] / 255;
    alpha[pixel] = rgba[pixel * 4 + 3] / 255;
  }
  return srgbSignalStatisticsFromRgb(rgb, alpha);
}

function trainingColorSpaceAudit(image, trainState, trainingRgba, standaloneRgba) {
  const source = srgbSignalStatisticsFromRgb(image.rgb, image.alpha);
  const trainingFront = srgbSignalStatisticsFromRgba(trainingRgba);
  const standaloneFront = srgbSignalStatisticsFromRgba(standaloneRgba);
  const stageStats = (stageImage) => stageImage
    ? srgbSignalStatisticsFromRgb(stageImage.rgb, stageImage.alpha)
    : null;
  return {
    contract: "front and virtual teachers use the same sRGB signal values",
    source_decode: "Canvas 2D RGBA8 -> float / 255",
    teacher_sampling: "bilinear interpolation in sRGB signal space",
    training_display: "direct unorm output; no extra gamma encode",
    source,
    coarse_teacher: stageStats(trainState?.coarseImage),
    mid_teacher: stageStats(trainState?.midImage),
    training_front: trainingFront,
    standalone_front: standaloneFront,
    training_minus_source_luma: trainingFront.mean_srgb_signal_luma - source.mean_srgb_signal_luma,
    standalone_minus_source_luma: standaloneFront.mean_srgb_signal_luma - source.mean_srgb_signal_luma,
    training_minus_standalone_luma:
      trainingFront.mean_srgb_signal_luma - standaloneFront.mean_srgb_signal_luma,
    render_parity: displayRgbaParity(trainingRgba, standaloneRgba),
  };
}

function layerTelemetryEnabled() {
  return QA_RUNTIME_ENABLED && (
    globalThis.__flatPhotoLayerTelemetry === true || layerEfficiencyDiagnosticsRequested()
  );
}

function lastEventAtStep(events, step) {
  if (!Array.isArray(events)) return null;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index]?.step === step) return events[index];
  }
  return null;
}

function sampledFrameTelemetry(snapshot, previous) {
  const pixelCount = snapshot.width * snapshot.height;
  const stride = Math.max(1, Math.ceil(Math.sqrt(pixelCount / 4096)));
  const luma = [];
  const rgbDelta = [];
  const signedDelta = [];
  for (let y = 0; y < snapshot.height; y += stride) {
    for (let x = 0; x < snapshot.width; x += stride) {
      const offset = (y * snapshot.width + x) * 4;
      const value = snapshot.rgba[offset] * 0.299 + snapshot.rgba[offset + 1] * 0.587 + snapshot.rgba[offset + 2] * 0.114;
      luma.push(value);
      if (previous?.width === snapshot.width && previous?.height === snapshot.height) {
        const priorOffset = offset;
        const delta = value - previous.luma[luma.length - 1];
        signedDelta.push(delta);
        rgbDelta.push((
          Math.abs(snapshot.rgba[offset] - previous.rgba[priorOffset]) +
          Math.abs(snapshot.rgba[offset + 1] - previous.rgba[priorOffset + 1]) +
          Math.abs(snapshot.rgba[offset + 2] - previous.rgba[priorOffset + 2])
        ) / 3);
      }
    }
  }
  if (!signedDelta.length) return { luma: Float32Array.from(luma), signedDelta: Float32Array.from(signedDelta) };
  const previousLuma = previous.luma;
  const meanCurrent = luma.reduce((sum, value) => sum + value, 0) / luma.length;
  const meanPrevious = previousLuma.reduce((sum, value) => sum + value, 0) / previousLuma.length;
  let varianceCurrent = 0;
  let variancePrevious = 0;
  let covariance = 0;
  let reversalCount = 0;
  let reversalEnergy = 0;
  for (let i = 0; i < luma.length; i += 1) {
    varianceCurrent += (luma[i] - meanCurrent) ** 2;
    variancePrevious += (previousLuma[i] - meanPrevious) ** 2;
    covariance += (luma[i] - meanCurrent) * (previousLuma[i] - meanPrevious);
    if (previous.signedDelta?.length === signedDelta.length && signedDelta[i] * previous.signedDelta[i] < 0) {
      reversalCount += 1;
      reversalEnergy += Math.min(Math.abs(signedDelta[i]), Math.abs(previous.signedDelta[i]));
    }
  }
  const denominator = Math.max(1, luma.length - 1);
  rgbDelta.sort((a, b) => a - b);
  return {
    luma: Float32Array.from(luma),
    signedDelta: Float32Array.from(signedDelta),
    frame_luma_mad: signedDelta.reduce((sum, value) => sum + Math.abs(value), 0) / signedDelta.length / 255,
    frame_rgb_p95: percentileSorted(rgbDelta, 0.95) / 255,
    temporal_ssim: ssimFromMoments(
      meanCurrent / 255,
      meanPrevious / 255,
      varianceCurrent / denominator / (255 ** 2),
      variancePrevious / denominator / (255 ** 2),
      covariance / denominator / (255 ** 2),
    ),
    reversal_ratio: previous.signedDelta?.length === signedDelta.length ? reversalCount / signedDelta.length : null,
    reversal_energy: previous.signedDelta?.length === signedDelta.length ? reversalEnergy / signedDelta.length / 255 : null,
  };
}

function layerPairDirection(indexA, layerA, indexB, layerB) {
  if (layerA !== layerB) return layerA > layerB ? 1 : -1;
  return indexA < indexB ? 1 : -1;
}

function layerOrderComparatorProbe() {
  return {
    lower_depth_loses: layerPairDirection(2, 0.25, 7, 0.75) === -1,
    higher_depth_wins: layerPairDirection(2, 0.75, 7, 0.25) === 1,
    equal_depth_uses_low_index: layerPairDirection(2, 0.5, 7, 0.5) === 1,
    equal_depth_reverses_for_high_index: layerPairDirection(7, 0.5, 2, 0.5) === -1,
  };
}

function localOverlapPairOrders(geometry, count) {
  const gridSize = 16;
  const sampleLimit = 32;
  const cells = new Map();
  const layers = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const x = geometry.xy[index * 2];
    const y = geometry.xy[index * 2 + 1];
    const cellX = Math.max(0, Math.min(gridSize - 1, Math.floor((x * 0.5 + 0.5) * gridSize)));
    const cellY = Math.max(0, Math.min(gridSize - 1, Math.floor((y * 0.5 + 0.5) * gridSize)));
    const key = cellY * gridSize + cellX;
    const cell = cells.get(key) || { items: [], seen: 0 };
    cell.seen += 1;
    if (cell.items.length < sampleLimit) {
      cell.items.push(index);
    } else {
      const hash = Math.imul((index + 1) ^ key, 0x9e3779b1) >>> 0;
      const replacement = hash % cell.seen;
      if (replacement < sampleLimit) cell.items[replacement] = index;
    }
    cells.set(key, cell);
    const packed = geometry.transform[index * 4 + 3];
    layers[index] = packedLayerOrder(packed);
  }
  const pairs = new Map();
  const addPair = (ia, ib) => {
      if (ia === ib) return;
      const ax = geometry.xy[ia * 2];
      const ay = geometry.xy[ia * 2 + 1];
      const asx = geometry.transform[ia * 4];
      const asy = geometry.transform[ia * 4 + 1];
      const at = geometry.transform[ia * 4 + 2];
      const arx = RENDER_SIGMA * (Math.abs(Math.cos(at)) * asx + Math.abs(Math.sin(at)) * asy);
      const ary = RENDER_SIGMA * (Math.abs(Math.sin(at)) * asx + Math.abs(Math.cos(at)) * asy);
      const bx = geometry.xy[ib * 2];
      const by = geometry.xy[ib * 2 + 1];
      const bsx = geometry.transform[ib * 4];
      const bsy = geometry.transform[ib * 4 + 1];
      const bt = geometry.transform[ib * 4 + 2];
      const brx = RENDER_SIGMA * (Math.abs(Math.cos(bt)) * bsx + Math.abs(Math.sin(bt)) * bsy);
      const bry = RENDER_SIGMA * (Math.abs(Math.sin(bt)) * bsx + Math.abs(Math.cos(bt)) * bsy);
      if (Math.abs(ax - bx) > arx + brx || Math.abs(ay - by) > ary + bry) return;
      const low = Math.min(ia, ib);
      const high = Math.max(ia, ib);
      pairs.set(`${low}:${high}`, layerPairDirection(low, layers[low], high, layers[high]));
  };
  const neighborOffsets = [[0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]];
  for (const key of [...cells.keys()].sort((a, b) => a - b)) {
    const cellX = key % gridSize;
    const cellY = Math.floor(key / gridSize);
    const items = cells.get(key).items;
    for (const [dx, dy] of neighborOffsets) {
      const neighborX = cellX + dx;
      const neighborY = cellY + dy;
      if (neighborX < 0 || neighborX >= gridSize || neighborY < 0 || neighborY >= gridSize) continue;
      const neighborKey = neighborY * gridSize + neighborX;
      const neighborItems = cells.get(neighborKey)?.items;
      if (!neighborItems) continue;
      if (neighborKey === key) {
        for (let a = 0; a < items.length; a += 1) {
          for (let b = a + 1; b < items.length; b += 1) addPair(items[a], items[b]);
        }
      } else {
        for (const ia of items) {
          for (const ib of neighborItems) addPair(ia, ib);
        }
      }
    }
  }
  return pairs;
}

async function recordLayerTelemetry(step, run = null) {
  assertTrainingRun(run);
  if (!layerTelemetryEnabled() || !state.webgpu.renderer?.trainState) return;
  const previousRecord = state.metrics.layer_telemetry[state.metrics.layer_telemetry.length - 1];
  if (previousRecord?.step === step) return;
  const count = state.params.count;
  const [snapshot, geometry] = await awaitTrainingRun(run, Promise.all([
    state.webgpu.renderer.capturePresentedStateRgba(),
    state.webgpu.renderer.readLayerTelemetryGeometry(count),
  ]));
  if (!snapshot || !geometry) return;
  const depth = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const packed = geometry.transform[index * 4 + 3];
    depth[index] = packedLayerOrder(packed);
  }
  const order = Array.from({ length: count }, (_, index) => index)
    .sort((a, b) => depth[a] - depth[b] || a - b);
  const ranks = new Int32Array(count);
  for (let rank = 0; rank < order.length; rank += 1) ranks[order[rank]] = rank;
  const previous = state.layerTelemetryState;
  const common = Math.min(count, previous?.depth.length || 0);
  let layerDeltaTotal = 0;
  let layerDeltaMax = 0;
  let rankFlipCount = 0;
  let rankReflipCount = 0;
  const movementSigns = new Int8Array(count);
  for (let index = 0; index < common; index += 1) {
    const delta = Math.abs(depth[index] - previous.depth[index]);
    layerDeltaTotal += delta;
    layerDeltaMax = Math.max(layerDeltaMax, delta);
    const movement = Math.sign(ranks[index] - previous.ranks[index]);
    movementSigns[index] = movement;
    if (movement !== 0) rankFlipCount += 1;
    if (movement !== 0 && previous.movementSigns[index] !== 0 && movement !== previous.movementSigns[index]) {
      rankReflipCount += 1;
    }
  }
  const pairOrders = localOverlapPairOrders(geometry, count);
  const densify = lastEventAtStep(state.metrics.densify_events, step);
  const previousPairState = previous;
  let persistentPairs = 0;
  let flippedPairs = 0;
  if (previousPairState?.pairOrders) {
    for (const [key, direction] of pairOrders) {
      if (!previousPairState.pairOrders.has(key)) continue;
      persistentPairs += 1;
      if (previousPairState.pairOrders.get(key) !== direction) flippedPairs += 1;
    }
  }
  const pairUnion = previousPairState?.pairOrders
    ? pairOrders.size + previousPairState.pairOrders.size - persistentPairs
    : 0;
  const frameStats = sampledFrameTelemetry(snapshot, previous?.snapshot);
  const frameDelta = previous?.snapshot?.rgba?.length === snapshot.rgba.length
    ? rgbaParity(previous.snapshot.rgba, snapshot.rgba)
    : null;
  const stage = snapshot.kind;
  const triggers = [];
  if (!previous || previous.stage !== stage) triggers.push("stage-change");
  if (densify) triggers.push("growth");
  if (layerEfficiencyDiagnosticsRequested()) {
    const steps = Math.max(1, state.metrics?.steps_requested || step || 1);
    const checkpointBucket = Math.min(4, Math.floor(step / Math.max(1, steps / 4)));
    const checkpoints = state.layerEfficiencyCheckpoints || [];
    if (checkpoints.at(-1)?.bucket !== checkpointBucket || step >= steps) {
      const deepCount = Math.max(1, Math.ceil(count * layerEfficiencyVariants().deepFraction));
      checkpoints.push({
        step,
        bucket: checkpointBucket,
        splats: count,
        deepest_indices: order.slice(0, deepCount),
      });
      state.layerEfficiencyCheckpoints = checkpoints.slice(-5);
    }
  }
  state.metrics.layer_telemetry.push({
    step,
    stage,
    triggers,
    splats: count,
    new_splats: Math.max(0, count - common),
    layer_delta_mean: common > 0 ? layerDeltaTotal / common : null,
    layer_delta_max: common > 0 ? layerDeltaMax : null,
    rank_flip_count: common > 0 ? rankFlipCount : null,
    rank_flip_ratio: common > 0 ? rankFlipCount / common : null,
    rank_reflip_count: common > 0 ? rankReflipCount : null,
    rank_reflip_ratio: common > 0 ? rankReflipCount / common : null,
    local_pair_count: pairOrders.size,
    local_pair_persistent_count: persistentPairs,
    local_order_flip_count: flippedPairs,
    local_order_flip_ratio: persistentPairs > 0 ? flippedPairs / persistentPairs : null,
    local_pair_membership_churn: pairUnion > 0
      ? (pairOrders.size + (previousPairState?.pairOrders?.size || 0) - 2 * persistentPairs) / pairUnion
      : null,
    frame_width: snapshot.width,
    frame_height: snapshot.height,
    frame_rgba_mean_abs: frameDelta?.mean_abs ?? null,
    frame_rgba_max_abs: frameDelta?.max_abs ?? null,
    frame_luma_mad: frameStats.frame_luma_mad ?? null,
    frame_rgb_p95: frameStats.frame_rgb_p95 ?? null,
    temporal_ssim: frameStats.temporal_ssim ?? null,
    reversal_ratio: frameStats.reversal_ratio ?? null,
    reversal_energy: frameStats.reversal_energy ?? null,
  });
  state.layerTelemetryState = {
    depth,
    ranks,
    movementSigns,
    pairOrders,
    snapshot: { ...snapshot, luma: frameStats.luma, signedDelta: frameStats.signedDelta },
    stage,
  };
}

function summarizeLayerEfficiency(params, attribution, tileDiagnostics, checkpoints, initialCount) {
  if (!params?.depthOrder || !attribution || !tileDiagnostics) {
    return {
      supported: false,
      reason: !attribution ? "hidden-rgb-attribution-unavailable" : "tile-diagnostics-unavailable",
    };
  }
  const deepFraction = layerEfficiencyVariants().deepFraction;
  const deepThreshold = deepFraction;
  const layerCount = 10;
  const bins = Array.from({ length: layerCount }, (_, index) => ({
    layer: index,
    splats: 0,
    initial_splats: 0,
    positive_harm: 0,
    contribution_mass: 0,
    deep_occluded_pixel_hits: 0,
    deep_occluded_weight_mass: 0,
    tile_references: 0,
  }));
  const attributionByIndex = new Map(attribution.rows.map((row) => [row.index, row]));
  let totalHarm = 0;
  let deepHarm = 0;
  let totalContribution = 0;
  let deepContribution = 0;
  let totalDeepHits = 0;
  let deepestDeepHits = 0;
  let totalTileReferences = 0;
  let deepTileReferences = 0;
  const finalDeep = new Set();
  for (let index = 0; index < params.count; index += 1) {
    const layer = Math.max(0, Math.min(1, params.depthOrder[index]));
    const bin = bins[Math.min(layerCount - 1, Math.floor(layer * layerCount))];
    const row = attributionByIndex.get(index);
    const harm = row?.positive_leave_one_out_l1_harm || 0;
    const contribution = row?.contribution_mass || 0;
    const deepHits = row?.deep_occluded_pixel_hits || 0;
    const deepWeight = row?.deep_occluded_weight_mass || 0;
    const tileReferences = tileDiagnostics.per_splat_reference_count[index] || 0;
    bin.splats += 1;
    if (index < initialCount) bin.initial_splats += 1;
    bin.positive_harm += harm;
    bin.contribution_mass += contribution;
    bin.deep_occluded_pixel_hits += deepHits;
    bin.deep_occluded_weight_mass += deepWeight;
    bin.tile_references += tileReferences;
    totalHarm += harm;
    totalContribution += contribution;
    totalDeepHits += deepHits;
    totalTileReferences += tileReferences;
    if (layer <= deepThreshold) {
      finalDeep.add(index);
      deepHarm += harm;
      deepContribution += contribution;
      deepestDeepHits += deepHits;
      deepTileReferences += tileReferences;
    }
  }
  const persistence = (checkpoints || []).map((checkpoint) => {
    const prior = new Set(checkpoint.deepest_indices || []);
    let retained = 0;
    for (const index of finalDeep) if (prior.has(index)) retained += 1;
    return {
      step: checkpoint.step,
      splats: checkpoint.splats,
      final_deep_retained_ratio: retained / Math.max(1, Math.min(finalDeep.size, prior.size)),
    };
  });
  const persistentDeepRatio = persistence.length > 1
    ? persistence.slice(0, -1).reduce((sum, item) => sum + item.final_deep_retained_ratio, 0) / (persistence.length - 1)
    : null;
  const deepHarmRatio = deepHarm / Math.max(1e-12, totalHarm);
  return {
    supported: true,
    contract: "QA-only depth attribution; no training or export contract changes",
    deep_fraction: deepFraction,
    deep_harm_ratio: deepHarmRatio,
    deep_contribution_ratio: deepContribution / Math.max(1e-12, totalContribution),
    deep_occluded_hit_ratio: deepestDeepHits / Math.max(1, totalDeepHits),
    deep_tile_reference_ratio: deepTileReferences / Math.max(1, totalTileReferences),
    persistent_deep_ratio: persistentDeepRatio,
    relocation_gate_passed: deepHarmRatio >= 0.4 && persistentDeepRatio !== null && persistentDeepRatio >= 0.5,
    relocation_candidate_enabled: layerEfficiencyVariants().deepRelocation,
    bins,
    persistence,
    tile_quantization: tileDiagnostics.quantization,
    tile_quantization_gate: {
      ten_layers_passed: (tileDiagnostics.quantization?.[10]?.estimated_comparison_reduction_ratio || 0) >= 0.5,
      five_layers_passed: (tileDiagnostics.quantization?.[5]?.estimated_comparison_reduction_ratio || 0) >= 0.5,
      note: "comparison reduction is theoretical; no runtime speed claim without an end-to-end A/B",
    },
  };
}

function destroyBuffers(...buffers) {
  for (const buffer of buffers) {
    if (buffer) buffer.destroy();
  }
}

function makeBuffer(device, data, usage, track = null) {
  const buffer = device.createBuffer({
    size: Math.ceil(data.byteLength / 4) * 4,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  track?.(buffer);
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

function makeSizedBuffer(device, data, sizeBytes, usage, track = null) {
  const buffer = device.createBuffer({
    size: Math.ceil(Math.max(sizeBytes, data.byteLength) / 4) * 4,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  track?.(buffer);
  if (data.byteLength) {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }
  return buffer;
}

function packColors(params) {
  const color = new Float32Array(params.count * 4);
  for (let i = 0; i < params.count; i += 1) {
    color[i * 4] = params.rgb[i * 3];
    color[i * 4 + 1] = params.rgb[i * 3 + 1];
    color[i * 4 + 2] = params.rgb[i * 3 + 2];
    color[i * 4 + 3] = params.opacity[i];
  }
  return color;
}

function packPositions(params) {
  const positions = new Float32Array(params.count * 4);
  for (let i = 0; i < params.count; i += 1) {
    positions[i * 4] = params.xy[i * 2];
    positions[i * 4 + 1] = params.xy[i * 2 + 1];
    positions[i * 4 + 2] = params.virtualDepthEnabled
      ? Math.max(-VIRTUAL_DEPTH_RAW_LIMIT, Math.min(VIRTUAL_DEPTH_RAW_LIMIT, params.virtualDepth?.[i] || 0))
      : params.brushWidthTaperEnabled
        ? Math.max(0, Math.min(1, params.brushTaper?.[i] ?? DEFAULT_LAYERED_BRUSH_TAPER))
        : 0;
    positions[i * 4 + 3] = 0;
  }
  return positions;
}

function packTransforms(params) {
  const transform = new Float32Array(params.count * 4);
  for (let i = 0; i < params.count; i += 1) {
    transform[i * 4] = params.scale[i * 2];
    transform[i * 4 + 1] = params.scale[i * 2 + 1];
    transform[i * 4 + 2] = params.theta?.[i] || 0;
    const tag = Math.max(1, Math.floor(params.detailTags?.[i] || 1));
    let layer = params.layerOrderEnabled
      ? Math.max(0, Math.min(1, params.depthOrder?.[i] ?? (1 - i / Math.max(1, params.count - 1))))
      : 0;
    if (params.discreteLayersEnabled) layer = quantizeLayerOrder(layer, params.discreteLayerCount);
    transform[i * 4 + 3] = tag + layer * LAYER_CODE_RANGE;
  }
  return transform;
}

function selectedBackend() {
  return state.webgpu.supported && state.webgpu.renderer ? "webgpu-train+render" : "webgpu-unavailable";
}

function selectedLearningRates() {
  const maxAnisotropy = clampNumber(els.maxAnisotropy.value, LIMITS.maxAnisotropyMin, LIMITS.maxAnisotropyMax, DEFAULT_MAX_ANISOTROPY);
  const boundarySigma = selectedBoundarySigma();
  return {
    scale: DEFAULT_LR_SCALE,
    position: clampNumber(els.positionLearningRate.value, LIMITS.lrMin, LIMITS.lrDefaultMax, DEFAULT_POSITION_LR),
    color: clampNumber(els.colorLearningRate.value, LIMITS.lrMin, LIMITS.lrDefaultMax, DEFAULT_COLOR_LR),
    opacity: clampNumber(els.opacityLearningRate.value, LIMITS.lrMin, LIMITS.opacityLrMax, DEFAULT_OPACITY_LR),
    scaleParam: clampNumber(els.scaleLearningRate.value, LIMITS.lrMin, LIMITS.scaleLrMax, DEFAULT_SCALE_LR),
    rotation: clampNumber(els.rotationLearningRate.value, LIMITS.lrMin, LIMITS.lrDefaultMax, DEFAULT_ROTATION_LR),
    thetaAlign: clampNumber(els.thetaAlignRate.value, LIMITS.lrMin, LIMITS.thetaAlignLrMax, DEFAULT_THETA_ALIGN_LR),
    maxAnisotropy,
    maxPlanarScale: clampNumber(els.maxPlanarScale.value, LIMITS.maxPlanarScaleMin, LIMITS.maxPlanarScaleMax, DEFAULT_MAX_PLANAR_SCALE),
    surfaceAnisotropy: qualityRecoveryVariants().surfaceAnisotropy,
    boundarySigma,
    adaptiveDetail: true,
    detailCoherence: clampNumber(els.detailCoherence.value, LIMITS.detailCoherenceMin, LIMITS.detailCoherenceMax, 0.8),
  };
}

function selectedPreviewRefresh() {
  return ["frame", "10", "final"].includes(els.previewRefresh.value) ? els.previewRefresh.value : "10";
}

function shouldPresentTrainingStep(step, refresh) {
  if (refresh === "frame") return true;
  if (refresh === "10") return step % 10 === 0;
  return false;
}

function trainSyncInterval() {
  const override = Number(window.__flatPhotoTrainSyncInterval);
  const interval = Number.isFinite(override) && override > 0 ? override : DEFAULT_TRAIN_SYNC_INTERVAL;
  return Math.max(1, Math.min(64, Math.round(interval)));
}

function effectiveTrainSyncInterval(activeSplats, baseInterval = trainSyncInterval()) {
  const count = Math.max(0, Math.round(Number(activeSplats) || 0));
  if (count >= VERY_HIGH_SPLAT_SYNC_THRESHOLD) return Math.min(baseInterval, 1);
  if (count >= HIGH_SPLAT_SYNC_THRESHOLD) return Math.min(baseInterval, 2);
  return baseInterval;
}

function plannedAdaptiveGpuBatch({
  step,
  steps,
  desiredSize,
  metricInterval,
  previewRefresh,
  structuralStep,
  qaHashPending,
  densitySteps,
  growthSteps,
  growthSettings,
  runLayerSettings,
  paintSettings,
  currentContributionSettings,
}) {
  if (performanceVariants().gpuSchedulingMode !== "adaptive") return 1;
  const safeMetricInterval = Math.max(1, Math.round(metricInterval) || 1);
  const startStage = curriculumTrainingStage(
    step,
    steps,
    phase33Variants(),
    state.webgpu.renderer?.trainState?.coarseImage,
    state.webgpu.renderer?.trainState?.midImage,
  );
  const startPaintMutationAllowed = opaquePaintStructuralMutationAllowed(
    step,
    steps,
    Boolean(state.params?.opaqueLayered),
    paintSettings?.opaquePaintSettleFraction,
  );
  const startPaintDetailRecoveryScheduled =
    Boolean(state.params?.opaqueLayered) &&
    opaquePaintDetailRecoveryDue(
      step,
      steps,
      OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL,
    );
  const startBrushSurfaceRecoveryScheduled =
    normalizedKernelShape(state.params?.kernelShape) === "opaque-brush" &&
    startPaintDetailRecoveryScheduled;
  const startLayerSchedule = layerOptimizationSettings(
    step,
    steps,
    startStage,
    runLayerSettings,
    Boolean(state.params?.opaqueLayered),
    paintSettings?.opaquePaintSettleFraction,
  );
  const startDiscreteLayerScheduled = Boolean(
    state.params?.discreteLayersEnabled &&
    state.params?.discreteLayerMoveRadius > 0 &&
    step >= densitySteps &&
    startPaintDetailRecoveryScheduled,
  );
  const surfaceLayerSortSettings = {
    enabled: state.params?.surfaceLayerPriorEnabled,
    p1Interval: state.params?.surfaceLayerPriorP1Interval,
    p2Interval: state.params?.surfaceLayerPriorP2Interval,
    p3Interval: state.params?.surfaceLayerPriorP3Interval,
    untilFraction: state.params?.surfaceLayerPriorUntilFraction,
  };
  const startSurfaceLayerSortDue = scaleBiasedSurfaceLayerSortSchedule(
    step,
    steps,
    surfaceLayerSortSettings,
  ).due;
  if (
    previewRefresh === "frame" ||
    structuralStep ||
    startLayerSchedule.due ||
    (startPaintMutationAllowed && startPaintDetailRecoveryScheduled) ||
    startDiscreteLayerScheduled ||
    startSurfaceLayerSortDue ||
    step % safeMetricInterval === 0 ||
    shouldPresentTrainingStep(step, previewRefresh) ||
    step % 32 === 0 ||
    step === steps ||
    qaHashPending ||
    performanceProfileLabels(step, steps).length > 0 ||
    virtualTiltStepSpec(step, startStage, steps).enabled
  ) return 1;
  const requested = Math.max(1, Math.min(MAX_TRAIN_BATCH_SIZE, Math.round(desiredSize) || 1));
  let count = 1;
  for (let candidate = step + 1; candidate <= steps && count < requested; candidate += 1) {
    const paintMutationAllowed = opaquePaintStructuralMutationAllowed(
      candidate,
      steps,
      Boolean(state.params?.opaqueLayered),
      paintSettings?.opaquePaintSettleFraction,
    );
    const candidateStage = curriculumTrainingStage(
      candidate,
      steps,
      phase33Variants(),
      state.webgpu.renderer?.trainState?.coarseImage,
      state.webgpu.renderer?.trainState?.midImage,
    );
    const densityScheduled = growthEventScheduled(
      candidate,
      densitySteps,
      growthSteps,
      growthSettings,
    );
    const densityDue = paintMutationAllowed && densityScheduled;
    const paintDetailRecoveryScheduled =
      Boolean(state.params?.opaqueLayered) &&
      opaquePaintDetailRecoveryDue(
        candidate,
        steps,
        OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL,
      );
    const brushSurfaceRecoveryScheduled =
      normalizedKernelShape(state.params?.kernelShape) === "opaque-brush" &&
      paintDetailRecoveryScheduled;
    const discreteLayerScheduled = Boolean(
      state.params?.discreteLayersEnabled &&
      state.params?.discreteLayerMoveRadius > 0 &&
      candidate >= densitySteps &&
      paintDetailRecoveryScheduled,
    );
    const relocationScheduled =
      growthSettings.densityEventsEnabled &&
      growthSettings.mcmcRelocationEnabled &&
      candidate > densifyWarmupSteps(densitySteps) &&
      (
        (
          candidate <= Math.floor(densitySteps * 0.85) &&
          candidate % EXPERIMENTAL_REFINE_EVERY === 0
        ) ||
        brushSurfaceRecoveryScheduled
      );
    const relocationDue = paintMutationAllowed && relocationScheduled;
    const layerSchedule = layerOptimizationSettings(
      candidate,
      steps,
      candidateStage,
      runLayerSettings,
      Boolean(state.params?.opaqueLayered),
      paintSettings?.opaquePaintSettleFraction,
    );
    const layerDue = layerSchedule.due;
    const visibilityCompactionDue = opaquePaintVisibilityCompactionDue(
      candidate,
      steps,
      paintSettings,
    );
    const contributionSettings = {
      ...currentContributionSettings,
      opaqueLayered: paintSettings?.opaqueLayered,
      opaquePaintSettleFraction: paintSettings?.opaquePaintSettleFraction,
    };
    const contributionCompactionResetDue = currentContributionCompactionResetDue(
      candidate,
      steps,
      contributionSettings,
    );
    const contributionCompactionDue = currentContributionCompactionDue(
      candidate,
      steps,
      contributionSettings,
    );
    const surfaceLayerSortDue = scaleBiasedSurfaceLayerSortSchedule(
      candidate,
      steps,
      surfaceLayerSortSettings,
    ).due;
    const suppressedPaintMutationScheduled = !paintMutationAllowed && (
      densityScheduled ||
      relocationScheduled ||
      layerSchedule.scheduled
    );
    if (
      candidateStage !== startStage ||
      densityDue ||
      relocationDue ||
      layerDue ||
      (paintMutationAllowed && paintDetailRecoveryScheduled) ||
      discreteLayerScheduled ||
      visibilityCompactionDue ||
      contributionCompactionResetDue ||
      contributionCompactionDue ||
      surfaceLayerSortDue ||
      suppressedPaintMutationScheduled ||
      performanceProfileLabels(candidate, steps).length > 0 ||
      virtualTiltStepSpec(candidate, candidateStage, steps).enabled ||
      candidate % safeMetricInterval === 0 ||
      shouldPresentTrainingStep(candidate, previewRefresh) ||
      candidate % 32 === 0 ||
      candidate === steps
    ) break;
    count += 1;
  }
  return count;
}

