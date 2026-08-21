function renderSplatInspector() {
  const current = Boolean(
    state.splatBaseline &&
      state.params &&
      state.metrics?.cpu_mirror_current &&
      state.metrics?.final_readback_step === state.metrics?.steps_done,
  );
  els.splatsEmpty.hidden = current;
  els.splatsContent.hidden = !current;
  const disabled = !current || trainingLifecycleInputLocked();
  const effectsEnabled = Boolean(els.splatParameterEffects?.checked);
  els.splatParameterEffects.disabled = disabled;
  for (const control of [
    els.splatOpacity,
    els.splatKernelFalloff,
    els.splatScale,
    els.splatAspectRatio,
    els.splatSmallFirstOrder,
    els.resetSplatAdjustments,
  ]) {
    control.disabled = disabled || !effectsEnabled;
  }
  for (const control of [els.splatShapeGaussian, els.splatShapeRectangle, els.splatShapeOpaqueBrush]) {
    control.disabled = disabled || !effectsEnabled;
  }
  els.splatAlphaBackground.disabled = disabled;
  updateSplatShapeControls();
  document.documentElement.dataset.splatsInspectionReady = String(current);
  document.documentElement.dataset.splatsAdjustmentReady = String(current && !disabled);
  if (!current) return;

  const adjustment = state.metrics?.post_train_adjustments;
  els.splatsMeta.textContent = `${state.params.count.toLocaleString()} splats · step ${state.metrics.steps_done}`;
  els.splatAdjustStatus.textContent = !effectsEnabled
    ? "Effects off · trained Gaussian preview."
    : state.adjustingSplats
      ? "Updating the preview..."
      : adjustment
        ? `Live preview · revision ${state.metrics.params_revision}`
        : "Move a control to preview the result.";
  const data = document.documentElement.dataset;
  data.splatsInspectionStep = String(state.metrics.steps_done);
  data.splatsInspectionCount = String(state.params.count);
  data.splatsInspectionNonfinite = String(finalSplatInspectorNonfiniteCount(state.params, state.metrics));
  data.splatsParamsRevision = String(state.metrics.params_revision ?? 0);
  data.splatsCoverageRevision = String(state.metrics.coverage_revision ?? "");
}

function publishLifecycleInteractionState() {
  updatePreviewModeControls();
  renderSplatInspector();
  publishState();
}

function updateSplatAdjustmentLabels() {
  els.splatOpacityValue.textContent = `${Number(els.splatOpacity.value).toFixed(2)}x`;
  els.splatKernelFalloffValue.textContent = `${Number(els.splatKernelFalloff.value).toFixed(2)}x`;
  els.splatScaleValue.textContent = `${Number(els.splatScale.value).toFixed(2)}x`;
  els.splatAspectRatioValue.textContent = `${Number(els.splatAspectRatio.value).toFixed(2)}x`;
}

function updateSplatShapeControls() {
  for (const [button, shape] of [
    [els.splatShapeGaussian, "gaussian"],
    [els.splatShapeRectangle, "rectangle"],
    [els.splatShapeOpaqueBrush, "opaque-brush"],
  ]) {
    const active = state.splatPreviewShape === shape;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function setSplatPreviewShape(shape) {
  if (trainingLifecycleInputLocked()) return;
  state.splatPreviewShape = shape === "opaque-brush"
    ? "opaque-brush"
    : normalizedKernelShape(shape);
  updateSplatShapeControls();
  refreshOutsidePreview().catch((error) => log(error.message));
  publishState();
}

function resetSplatAdjustmentControls() {
  els.splatOpacity.value = "1";
  els.splatKernelFalloff.value = "1";
  els.splatScale.value = "1";
  els.splatAspectRatio.value = "1";
  els.splatSmallFirstOrder.checked = false;
  const trainedShape = trainedSplatShape();
  state.splatPreviewShape = trainedShape === "box" ? "rectangle" : trainedShape;
  updateSplatAdjustmentLabels();
  updateSplatShapeControls();
}

function clearSplatAdjustmentBaseline() {
  window.clearTimeout(state.splatAdjustmentValidationTimer);
  if (state.splatAdjustmentFrame) window.cancelAnimationFrame(state.splatAdjustmentFrame);
  state.splatAdjustmentEpoch += 1;
  state.splatInspectorNonfiniteCache = null;
  state.splatBaseline = null;
  state.adjustingSplats = false;
  state.splatAdjustmentVersion = 0;
  state.splatAdjustmentFrame = 0;
  state.splatAdjustmentRenderedVersion = 0;
  state.splatAdjustmentValidationVersion = 0;
  state.splatAdjustmentStats = { requests: 0, renders: 0, staleDropped: 0, yields: 0, lastMs: 0 };
  els.splatParameterEffects.checked = true;
  els.outsidePreviewToggle.checked = false;
  resetSplatAdjustmentControls();
  els.splatAdjustStatus.textContent = "Ready.";
  renderSplatInspector();
}

function captureSplatAdjustmentBaseline() {
  if (!state.params || !state.metrics?.cpu_mirror_current) return;
  state.splatAdjustmentEpoch += 1;
  state.splatInspectorNonfiniteCache = null;
  // Final readback already owns an immutable CPU mirror. Live adjustments are
  // shader uniforms, so a second full parameter clone is unnecessary.
  state.splatBaseline = state.params;
  state.metrics.params_revision = 0;
  state.metrics.coverage_revision = 0;
  state.metrics.post_train_adjustments = null;
  state.splatAdjustmentVersion = 0;
  state.splatAdjustmentRenderedVersion = 0;
  state.splatAdjustmentValidationVersion = 0;
  state.splatAdjustmentStats = { requests: 0, renders: 0, staleDropped: 0, yields: 0, lastMs: 0 };
  els.splatParameterEffects.checked = true;
  resetSplatAdjustmentControls();
  renderSplatInspector();
}

function scaleSplatLocalAspectRatio(sx, sy, aspectRatio) {
  const stretch = Math.sqrt(Math.max(1e-6, aspectRatio));
  return {
    sx: Math.max(MIN_SPLAT_SCALE, sx * stretch),
    sy: Math.max(MIN_SPLAT_SCALE, sy / stretch),
  };
}

async function adjustedParamsFromBaseline(version, epoch) {
  const baseline = state.splatBaseline;
  if (!baseline) throw new Error("Finish training before adjusting splats.");
  const values = currentSplatAdjustmentValues();
  const { opacityMultiplier, splatScaleMultiplier, localAspectRatio } = values;
  if (opacityMultiplier === 1 && splatScaleMultiplier === 1 && localAspectRatio === 1) {
    return { params: baseline, values };
  }
  const params = snapshotParams(baseline);
  for (let i = 0; i < params.count; i += 1) {
    if (i > 0 && i % 8192 === 0) {
      state.splatAdjustmentStats.yields += 1;
      await nextFrame();
      if (
        epoch !== state.splatAdjustmentEpoch ||
        baseline !== state.splatBaseline ||
        version !== state.splatAdjustmentVersion
      ) return null;
    }
    const localScale = scaleSplatLocalAspectRatio(
      baseline.scale[i * 2] * splatScaleMultiplier,
      baseline.scale[i * 2 + 1] * splatScaleMultiplier,
      localAspectRatio,
    );
    const constrained = constrainSplat(
      baseline.xy[i * 2],
      baseline.xy[i * 2 + 1],
      localScale.sx,
      localScale.sy,
      baseline.theta[i],
      baseline.boundarySigma,
      anisotropyLimitForTag(baseline.detailTags?.[i], baseline),
    );
    params.xy[i * 2] = constrained.x;
    params.xy[i * 2 + 1] = constrained.y;
    params.scale[i * 2] = constrained.sx;
    params.scale[i * 2 + 1] = constrained.sy;
    params.theta[i] = baseline.theta[i];
    params.opacity[i] = opacityMultiplier >= 10
      ? 0.999999
      : Math.min(0.999999, Math.max(1e-6, baseline.opacity[i] * opacityMultiplier));
  }
  assertFiniteParams(params, "post-training-adjustment");
  return { params, values };
}

function captureSplatAdjustmentSnapshot() {
  const baseline = state.splatBaseline;
  return {
    baseline,
    sourceParams: baseline || state.params,
    epoch: state.splatAdjustmentEpoch,
    version: state.splatAdjustmentVersion,
  };
}

function splatAdjustmentSnapshotIsCurrent(snapshot) {
  return Boolean(
    snapshot &&
    snapshot.epoch === state.splatAdjustmentEpoch &&
    snapshot.version === state.splatAdjustmentVersion &&
    snapshot.baseline === state.splatBaseline &&
    snapshot.sourceParams === (state.splatBaseline || state.params),
  );
}

async function materializeCurrentSplatAdjustmentSnapshot() {
  const snapshot = captureSplatAdjustmentSnapshot();
  if (!snapshot.sourceParams) return null;
  const materialized = snapshot.baseline
    ? await adjustedParamsFromBaseline(snapshot.version, snapshot.epoch)
    : { params: snapshot.sourceParams };
  if (!materialized?.params || !splatAdjustmentSnapshotIsCurrent(snapshot)) return null;
  // Keep this detached from the optimizer and baseline. It is an ephemeral
  // export/viewer payload, not a new trained parameter state.
  return { ...materialized, snapshot };
}

function currentSplatAdjustmentValues() {
  const enabled = Boolean(els.splatParameterEffects?.checked);
  return {
    enabled,
    opacityMultiplier: enabled ? clampNumber(els.splatOpacity.value, 0, 10, 1) : 1,
    kernelFalloff: enabled ? clampNumber(els.splatKernelFalloff.value, 0, 2, 1) : 1,
    splatScaleMultiplier: enabled ? clampNumber(els.splatScale.value, 0, 5, 1) : 1,
    localAspectRatio: enabled ? clampNumber(els.splatAspectRatio.value, 0.1, 10, 1) : 1,
  };
}

function splatAlphaRenderOptions() {
  const effectsEnabled = Boolean(els.splatParameterEffects?.checked);
  const trainedShape = trainedSplatShape();
  const adjustments = currentSplatAdjustmentValues();
  return {
    alphaBackground: hexColorToRgb(els.splatAlphaBackground?.value, [0, 0, 0]),
    splatSmallFirstOrder: effectsEnabled && Boolean(els.splatSmallFirstOrder?.checked),
    kernelFalloff: adjustments.kernelFalloff,
    opacityMultiplier: adjustments.opacityMultiplier,
    splatScaleMultiplier: adjustments.splatScaleMultiplier,
    localAspectRatio: adjustments.localAspectRatio,
    splatShape: effectsEnabled
      ? state.splatPreviewShape === "rectangle" && trainedShape === "box"
        ? "box"
        : state.splatPreviewShape
      : trainedShape,
  };
}

function lockAdjustedExport() {
  state.metrics.coverage_revision = null;
  updateDownloads(false);
  state.exportMessage = "Updating the splat preview...";
  updateExportPanel();
}

async function renderLiveSplatAdjustments(version, epoch) {
  if (
    epoch !== state.splatAdjustmentEpoch ||
    !state.splatBaseline ||
    !state.image ||
    !state.metrics ||
    trainingLifecycleInputLocked() ||
    version !== state.splatAdjustmentVersion
  ) return;
  const startedAt = performance.now();
  const values = currentSplatAdjustmentValues();
  if (epoch !== state.splatAdjustmentEpoch || version !== state.splatAdjustmentVersion) {
    state.splatAdjustmentStats.staleDropped += 1;
    return;
  }
  // Live adjustments are render uniforms over the immutable trained buffers.
  // Materialize modified CPU arrays only for formats that require parameters.
  state.params = state.splatBaseline;
  state.metrics.params_revision = (state.metrics.params_revision ?? 0) + 1;
  state.metrics.post_train_adjustments = values;
  lockAdjustedExport();
  state.previewMode = "splats";
  updatePreviewModeControls();
  await refreshOutsidePreview();
  if (
    epoch !== state.splatAdjustmentEpoch ||
    version !== state.splatAdjustmentVersion
  ) return;
  state.splatAdjustmentRenderedVersion = version;
  state.splatAdjustmentStats.renders += 1;
  state.splatAdjustmentStats.lastMs = performance.now() - startedAt;
  state.metrics.splat_adjustment_scheduler = { ...state.splatAdjustmentStats };
  renderSplatInspector();
  publishState();
}

async function validateLiveSplatAdjustments(version, epoch) {
  if (
    epoch !== state.splatAdjustmentEpoch ||
    !state.splatBaseline ||
    !state.image ||
    !state.metrics ||
    trainingLifecycleInputLocked() ||
    version !== state.splatAdjustmentVersion
  ) return;
  state.adjustingSplats = true;
  renderSplatInspector();
  try {
    // Splat controls are display/export adjustments. Re-uploading a complete
    // optimizer state here caused large allocations to race tab-driven preview
    // renders after training buffers had already been released. The live
    // preview above is the validation surface; training metrics stay unchanged.
    await nextFrame();
    if (
      epoch !== state.splatAdjustmentEpoch ||
      version !== state.splatAdjustmentVersion ||
      trainingLifecycleInputLocked()
    ) return;
    state.metrics.post_adjustment_overlap_diagnostics = null;
    state.metrics.post_adjustment_overlap_revision = null;
    // These controls are display/export uniforms over the immutable trained
    // parameters. Training coverage and quality stay valid; PNG export runs
    // its own RGBA round-trip check for the adjusted display surface.
    state.metrics.coverage_revision = state.metrics.params_revision ?? 0;
    updateDownloads(!state.metrics?.safety_stop);
    state.exportMessage = "Preview ready; training quality and coverage metrics are unchanged.";
    updateExportPanel();
  } catch (error) {
    if (isExpectedPreviewCancellation(error)) return;
    updateDownloads(false);
    state.exportMessage = `Adjustment failed: ${error.message}`;
    updateExportPanel();
    log(state.exportMessage);
  } finally {
    state.adjustingSplats = false;
    renderSplatInspector();
    publishState();
  }
}

async function processSplatAdjustmentQueue() {
  if (state.splatAdjustmentProcessing) return;
  state.splatAdjustmentProcessing = true;
  try {
    while (state.splatBaseline && state.image && state.metrics && !trainingLifecycleInputLocked()) {
      const epoch = state.splatAdjustmentEpoch;
      const version = state.splatAdjustmentVersion;
      if (state.splatAdjustmentRenderedVersion !== version) {
        await renderLiveSplatAdjustments(version, epoch);
        if (
          epoch !== state.splatAdjustmentEpoch ||
          version !== state.splatAdjustmentVersion
        ) continue;
      }
      if (state.splatAdjustmentValidationVersion === version) {
        state.splatAdjustmentValidationVersion = 0;
        await validateLiveSplatAdjustments(version, epoch);
        if (
          epoch !== state.splatAdjustmentEpoch ||
          version !== state.splatAdjustmentVersion
        ) continue;
      }
      break;
    }
  } catch (error) {
    if (isExpectedPreviewCancellation(error)) return;
    state.exportMessage = `Adjustment failed: ${error.message}`;
    updateExportPanel();
    log(state.exportMessage);
  } finally {
    state.splatAdjustmentProcessing = false;
    if (
      state.splatBaseline &&
      !trainingLifecycleInputLocked() &&
      (state.splatAdjustmentRenderedVersion !== state.splatAdjustmentVersion ||
        state.splatAdjustmentValidationVersion === state.splatAdjustmentVersion)
    ) {
      queueMicrotask(() => processSplatAdjustmentQueue());
    }
  }
}

function queueSplatAdjustments({ immediate = false } = {}) {
  updateSplatAdjustmentLabels();
  if (!state.splatBaseline || !state.image || !state.metrics || trainingLifecycleInputLocked()) return;
  const version = state.splatAdjustmentVersion + 1;
  state.splatAdjustmentVersion = version;
  state.splatAdjustmentStats.requests += 1;
  window.clearTimeout(state.splatAdjustmentValidationTimer);
  const enqueuePreview = () => {
    state.splatAdjustmentFrame = 0;
    processSplatAdjustmentQueue();
  };
  if (immediate) {
    if (state.splatAdjustmentFrame) window.cancelAnimationFrame(state.splatAdjustmentFrame);
    state.splatAdjustmentFrame = 0;
    enqueuePreview();
  } else if (!state.splatAdjustmentFrame) {
    state.splatAdjustmentFrame = window.requestAnimationFrame(enqueuePreview);
  }
  state.splatAdjustmentValidationTimer = window.setTimeout(() => {
    if (version !== state.splatAdjustmentVersion) return;
    state.splatAdjustmentValidationVersion = version;
    processSplatAdjustmentQueue();
  }, immediate ? 0 : 240);
}

function resetSplatAdjustments() {
  resetSplatAdjustmentControls();
  queueSplatAdjustments({ immediate: true });
}

