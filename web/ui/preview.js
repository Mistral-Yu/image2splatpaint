function recommendedDetailShape(image) {
  const longSide = Math.max(1, Number(image?.width) || 1, Number(image?.height) || 1);
  const minorPx = 1;
  const majorPx = Math.max(8, Math.min(64, longSide * 0.02));
  return { minorPx, majorPx, maxAnisotropy: Math.max(6, Math.min(32, majorPx / minorPx)) };
}

function compactNumber(value) {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(Math.round(value));
}

function releaseImageSource(image) {
  const source = image?.sourceBitmap;
  if (!source) return;
  source.close?.();
  releaseCanvasBackingStore(source);
  image.sourceBitmap = null;
}

function releaseCanvasBackingStore(source) {
  // Canvas has no close(). Shrinking it releases the backing store while the
  // Float32 source cache remains available only on the replacement image.
  if (typeof HTMLCanvasElement !== "undefined" && source instanceof HTMLCanvasElement) {
    source.width = 1;
    source.height = 1;
  }
}


function showCanvas(kind) {
  els.previewCanvas.hidden = kind !== "preview";
  els.gpuCanvas.hidden = kind !== "gpu";
  els.tiltCanvas.hidden = kind !== "tilt";
  els.tiltTeacherCanvas.hidden = kind !== "tilt" || state.tilt.viewMode === "splats";
  els.tiltTeacherCanvas.style.opacity = state.tilt.viewMode === "overlay" ? "0.5" : "1";
  els.tiltCanvas.style.opacity = kind === "tilt" && state.tilt.viewMode === "original" ? "0" : "1";
  els.tiltFrameOverlay.hidden = kind !== "tilt" || !state.tilt.controller;
  els.viewControls.hidden = kind === "tilt";
  // The padded canvas already communicates the outside-image extent. Keep the
  // legacy frame node for QA selectors, but never draw a competing white box.
  els.previewImageFrame.hidden = true;
  if (kind !== "tilt") applyCanvasView();
}

function activePreviewCanvas() {
  return state.previewMode === "splats" && !els.gpuCanvas.hidden ? els.gpuCanvas : els.previewCanvas;
}

function fittedCanvasScale(width = activePreviewCanvas().width, height = activePreviewCanvas().height) {
  const rect = els.viewer.getBoundingClientRect();
  const maxWidth = Math.max(160, rect.width - 44);
  const maxHeight = Math.max(160, rect.height - 44);
  return Math.min(maxWidth / Math.max(1, width), maxHeight / Math.max(1, height));
}

function applyCanvasView() {
  const activeCanvas = activePreviewCanvas();
  const width = activeCanvas.width;
  const height = activeCanvas.height;
  if (!width || !height) return;
  const scale = state.canvasView.mode === "fit" ? fittedCanvasScale(width, height) : state.canvasView.scale;
  state.canvasView.scale = Math.max(0.02, Math.min(32, scale));
  const transform = `translate(-50%, -50%) translate(${state.canvasView.panX}px, ${state.canvasView.panY}px)`;
  for (const canvas of [els.previewCanvas, els.gpuCanvas]) {
    canvas.style.width = `${Math.max(1, Math.round(canvas.width * state.canvasView.scale))}px`;
    canvas.style.height = `${Math.max(1, Math.round(canvas.height * state.canvasView.scale))}px`;
    canvas.style.transform = transform;
  }
  if (!els.previewImageFrame.hidden && state.image) {
    els.previewImageFrame.style.width = `${Math.max(1, Math.round(state.image.width * state.canvasView.scale))}px`;
    els.previewImageFrame.style.height = `${Math.max(1, Math.round(state.image.height * state.canvasView.scale))}px`;
    els.previewImageFrame.style.transform = transform;
  }
  document.documentElement.dataset.canvasViewMode = state.canvasView.mode;
  document.documentElement.dataset.canvasViewScale = String(state.canvasView.scale);
  document.documentElement.dataset.canvasPanX = String(Math.round(state.canvasView.panX));
  document.documentElement.dataset.canvasPanY = String(Math.round(state.canvasView.panY));
}

function setCanvasView(mode) {
  if (canvasViewInputLocked()) return false;
  state.canvasView.mode = mode;
  state.canvasView.scale = mode === "actual" ? 1 : fittedCanvasScale();
  state.canvasView.panX = 0;
  state.canvasView.panY = 0;
  applyCanvasView();
  publishState();
  return true;
}

function fitCanvases(width = activePreviewCanvas().width, height = activePreviewCanvas().height) {
  state.canvasView.mode = "fit";
  state.canvasView.scale = fittedCanvasScale(width, height);
  state.canvasView.panX = 0;
  state.canvasView.panY = 0;
  applyCanvasView();
}

function canvasViewInputLocked() {
  return trainingLifecycleInputLocked();
}

function cancelCanvasViewGesture() {
  const pointerIds = [...state.canvasPointers.keys()];
  state.canvasPointers.clear();
  state.canvasPinch = null;
  state.canvasView.pointerId = null;
  els.viewer.classList.remove("is-panning");
  for (const pointerId of pointerIds) {
    try {
      if (els.viewer.hasPointerCapture(pointerId)) els.viewer.releasePointerCapture(pointerId);
    } catch {
      // Pointer capture may already have ended between lifecycle updates.
    }
  }
}

function updateCanvasViewControls() {
  const locked = canvasViewInputLocked();
  if (locked && (state.canvasPointers.size > 0 || state.canvasPinch || state.canvasView.pointerId !== null)) {
    cancelCanvasViewGesture();
  }
  const disabled = !state.image || locked;
  els.actualSizeButton.disabled = disabled;
  els.fitViewButton.disabled = disabled;
  els.viewer.classList.toggle("canvas-view-locked", locked);
  els.viewer.dataset.canvasViewLocked = String(locked);
  els.viewer.title = locked ? "Canvas zoom and pan are locked while training." : "";
  document.documentElement.dataset.canvasViewLocked = String(locked);
}

function zoomCanvasAt(clientX, clientY, deltaY) {
  if (!state.image || canvasViewInputLocked()) return false;
  const rect = els.viewer.getBoundingClientRect();
  const oldScale = state.canvasView.scale;
  const nextScale = Math.max(0.02, Math.min(32, oldScale * Math.exp(-deltaY * 0.0015)));
  const centerX = rect.left + rect.width * 0.5 + state.canvasView.panX;
  const centerY = rect.top + rect.height * 0.5 + state.canvasView.panY;
  const imageX = (clientX - centerX) / oldScale;
  const imageY = (clientY - centerY) / oldScale;
  state.canvasView.mode = "custom";
  state.canvasView.scale = nextScale;
  state.canvasView.panX = clientX - (rect.left + rect.width * 0.5) - imageX * nextScale;
  state.canvasView.panY = clientY - (rect.top + rect.height * 0.5) - imageY * nextScale;
  applyCanvasView();
  return true;
}

function drawRgbToCanvas(rgb, width, height) {
  previewCtx.putImageData(rgbToImageData(rgb, width, height), 0, 0);
}

function drawOriginalToCanvas() {
  if (!state.image) return false;
  const { width, height, sourceBitmap } = state.image;
  if (els.previewCanvas.width !== width) els.previewCanvas.width = width;
  if (els.previewCanvas.height !== height) els.previewCanvas.height = height;
  previewCtx.clearRect(0, 0, width, height);
  if (sourceBitmap) {
    // Reuse the bounded source cache instead of converting the full Float32
    // image back to ImageData on every Original/Splats tab switch.
    previewCtx.imageSmoothingEnabled = true;
    previewCtx.imageSmoothingQuality = "high";
    previewCtx.drawImage(sourceBitmap, 0, 0, width, height);
  } else {
    drawRgbToCanvas(state.image.rgb, width, height);
  }
  return true;
}

function updatePreviewModeControls() {
  const hasImage = Boolean(state.image);
  const hasSplats = Boolean(state.params);
  const locked = previewModeInputLocked();
  els.originalPreviewButton.disabled = !hasImage || locked;
  els.splatsPreviewButton.disabled = !hasSplats || locked;
  for (const [button, mode] of [
    [els.splatsPreviewButton, "splats"],
    [els.originalPreviewButton, "original"],
  ]) {
    const active = state.previewMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  document.documentElement.dataset.previewMode = state.previewMode;
}

function setPreviewMode(mode) {
  if (previewModeInputLocked()) return false;
  if (mode === "original") {
    if (!state.image) return false;
    state.previewMode = "original";
    drawOriginalToCanvas();
    showCanvas("preview");
  } else if (mode === "splats") {
    if (!state.params) return false;
    state.previewMode = "splats";
    if (state.running) {
      state.previewPadding = previewPaddingSpec(state.image, state.params, false);
      state.webgpu.renderer?.presentTrainState(state.image);
    } else {
      refreshOutsidePreview().catch((error) => log(`splat preview failed: ${error.message}`));
    }
    showCanvas("gpu");
  } else {
    return false;
  }
  updatePreviewModeControls();
  publishState();
  return true;
}

