els.viewer.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (trainingLifecycleInputLocked()) {
    setImageDragover(false);
    return;
  }
  if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  setImageDragover(true);
});

els.viewer.addEventListener("dragleave", (event) => {
  if (event.relatedTarget instanceof Node && els.viewer.contains(event.relatedTarget)) return;
  setImageDragover(false);
});

els.viewer.addEventListener("drop", async (event) => {
  event.preventDefault();
  setImageDragover(false);
  if (trainingLifecycleInputLocked()) return;
  const file = event.dataTransfer?.files?.[0];
  if (state.image && file?.type?.startsWith("image/") && !window.confirm(
    `Replace ${state.image.fileName} with ${file.name}? The current training result will be discarded.`,
  )) {
    eventLog(`image replacement canceled: ${file.name || "unknown"}`);
    return;
  }
  try {
    state.lastInputMode = "drop";
    await loadFile(file);
    eventLog(`loaded image from drop: ${file?.name || "unknown"}`);
  } catch (error) {
    setStatus("error");
    setTrainingMessage(`Image load failed: ${error.message}`, "error");
    log(error.message);
  }
});

els.fileInput.addEventListener("change", async () => {
  const file = els.fileInput.files[0];
  if (!file) return;
  if (trainingLifecycleInputLocked()) {
    els.fileInput.value = "";
    return;
  }
  try {
    state.lastInputMode = "file";
    await loadFile(file);
    eventLog(`loaded image from file picker: ${file.name || "unknown"}`);
  } catch (error) {
    setStatus("error");
    setTrainingMessage(`Image load failed: ${error.message}`, "error");
    log(error.message);
    eventLog(error.message);
  } finally {
    // A file input does not fire change when the same path stays selected.
    // Clearing it makes retrying the same image reliable after success or error.
    els.fileInput.value = "";
  }
});

els.loadImageButton.addEventListener("click", () => {
  if (!trainingLifecycleInputLocked()) els.fileInput.click();
});

els.clearImageButton.addEventListener("click", confirmClearImage);

els.sampleButton.addEventListener("click", async () => {
  if (trainingLifecycleInputLocked() || state.sampleLoading) return;
  state.lastInputMode = "sample";
  state.sampleLoading = true;
  els.sampleButton.disabled = true;
  setTrainingMessage("Loading sample image...");
  publishState();
  try {
    await loadGeneratedSample();
  } catch (error) {
    setStatus("error");
    setTrainingMessage(`Sample failed: ${error.message}`, "error");
    log(`sample failed: ${error.message}`);
    eventLog(`sample failed: ${error.message}`);
  } finally {
    state.sampleLoading = false;
    els.sampleButton.disabled = state.running;
    publishState();
  }
});
els.splatsPreviewButton.addEventListener("click", () => setPreviewMode("splats"));
els.originalPreviewButton.addEventListener("click", () => setPreviewMode("original"));
els.outsidePreviewToggle.addEventListener("change", () => {
  if (trainingLifecycleInputLocked()) return;
  // Padded preview is a display-layer toggle. Preserve the user's current
  // zoom and pan instead of fitting the newly sized canvas automatically.
  state.canvasView.mode = "custom";
  refreshOutsidePreview().catch((error) => {
    setStatus("error");
    log(error.message);
  });
});
els.pathButton.addEventListener("click", () => {
  if (trainingLifecycleInputLocked()) return;
  loadPathImage().catch((error) => {
    setStatus("error");
    setTrainingMessage(`Image load failed: ${error.message}`, "error");
    log(error.message);
    eventLog(error.message);
  });
});
for (const element of [els.trainSize, els.initialSplatCount, els.capacityMode, els.stepCount]) {
  element.addEventListener("input", updateMemoryRecommendation);
}
els.memoryLimiterUnlock.addEventListener("click", () => {
  if (trainingLifecycleInputLocked()) return;
  state.safety.memoryLimiterUnlocked = !state.safety.memoryLimiterUnlocked;
  clearSafetyStop();
  updateMemoryRecommendation();
});
els.finalSplatCount.addEventListener("input", () => {
  updateMemoryRecommendation({ reconcileSplatCounts: false });
});
els.adaptiveGridInitializationFraction.addEventListener("input", publishState);
els.liveQualityMetrics.addEventListener("change", publishState);
els.rectanglePaintShape.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
els.flowSplatUnderpainting.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
els.flowSplatUnderpaintPercent.addEventListener("input", publishState);
els.flowSplatFusionStrokeOptimization.addEventListener("change", publishState);
els.flowSplatFusionStrokeTexture.addEventListener("change", publishState);
els.flowSplatFusionInitialization.addEventListener("change", publishState);
els.flowSplatFusionPaintCurriculum.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
for (const control of [
  els.flowSplatFusionFixedOpacity,
  els.flowSplatFusionStartingWidthDivisor,
  els.flowSplatFusionStartingLengthPercent,
  els.flowSplatFusionResidualMovePx,
  els.flowSplatFusionInitialWidthMin,
  els.flowSplatFusionInitialWidthMax,
  els.flowSplatFusionFrontWidthMax,
  els.flowSplatFusionFrontWidthLearning,
]) {
  control.addEventListener("input", publishState);
}
els.flowSplatFusionScaleMatchedResidualRepaint.addEventListener("change", publishState);
els.flowSplatFusionWidthPercent.addEventListener("input", publishState);
els.flowSplatFusionSplatSizeVariation.addEventListener("input", publishState);
els.flowSplatFusionMovementLimit.addEventListener("input", publishState);
els.currentContributionCompaction.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
for (const control of [
  els.currentContributionCompactionStart,
  els.currentContributionCompactionInterval,
  els.currentContributionCompactionMaxRemoval,
  els.currentContributionCompactionNearZero,
  els.currentContributionCompactionWindow,
]) {
  control.addEventListener("change", () => {
    const settings = currentContributionCompactionSettings();
    els.currentContributionCompactionStart.value = String(Math.round(settings.startFraction * 100));
    els.currentContributionCompactionInterval.value = String(settings.requestedIntervalSteps);
    els.currentContributionCompactionMaxRemoval.value = String(settings.maxRemovalFraction * 100);
    els.currentContributionCompactionNearZero.value = String(settings.nearZeroMaxFraction * 100);
    els.currentContributionCompactionWindow.value = String(settings.requestedWindowSteps);
    publishState();
  });
}
const commitFinalSplatCount = () => {
  updateMemoryRecommendation();
};
els.finalSplatCount.addEventListener("change", commitFinalSplatCount);
els.finalSplatCount.addEventListener("blur", commitFinalSplatCount);
els.tileCullingToggle.addEventListener("change", publishState);
els.trainLayerOrder.addEventListener("change", () => {
  syncLayerOrderDependency();
  publishState();
});
els.layerAwareAccumulation.addEventListener("change", () => {
  syncLayerOrderDependency();
  publishState();
});
for (const control of [
  els.layeredBrushLocalColorFlow,
  els.layeredBrushStrokePersistence,
]) {
  control.addEventListener("change", () => {
    syncAlgorithmRequirements();
    publishState();
  });
}
for (const control of [
  els.layeredBrushMinAspectRatio,
  els.layeredBrushMaxAspectRatio,
  els.layeredBrushRibbonAspectFloor,
  els.layeredBrushAccentAspectFloor,
]) {
  control.addEventListener("change", () => {
    const minimum = selectedBrushMinAspectRatio();
    const maximum = selectedBrushMaxAspectRatio();
    const floors = selectedBrushAspectFloors();
    els.layeredBrushMinAspectRatio.value = String(Math.min(minimum, maximum));
    els.layeredBrushMaxAspectRatio.value = String(maximum);
    els.layeredBrushRibbonAspectFloor.max = String(maximum);
    els.layeredBrushAccentAspectFloor.max = String(maximum);
    els.layeredBrushRibbonAspectFloor.value = String(floors.ribbon);
    els.layeredBrushAccentAspectFloor.value = String(floors.accent);
    publishState();
  });
}
for (const [minimumControl, maximumControl, selector] of [
  [els.rectangleLearnedOpacityMin, els.rectangleLearnedOpacityMax, selectedRectangleLearnedOpacity],
  [els.layeredBrushLearnedOpacityMin, els.layeredBrushLearnedOpacityMax, selectedLayeredBrushLearnedOpacity],
]) {
  for (const control of [minimumControl, maximumControl]) {
    control.addEventListener("change", () => {
      const range = selector();
      minimumControl.value = String(range.min);
      maximumControl.value = String(range.max);
      publishState();
    });
  }
}
for (const control of [
  els.rectangleOpacityGradientMin,
  els.rectangleOpacityGradientMax,
]) {
  control.addEventListener("change", () => {
    const gradient = selectedRectangleOpacityGradient();
    els.rectangleOpacityGradientMin.value = String(gradient.min);
    els.rectangleOpacityGradientMax.value = String(gradient.max);
    publishState();
  });
}
for (const [minimumControl, maximumControl, selector] of [
  [
    els.rectangleCenterOpacityGradientMin,
    els.rectangleCenterOpacityGradientMax,
    selectedRectangleCenterOpacityGradient,
  ],
  [
    els.layeredBrushCenterOpacityGradientMin,
    els.layeredBrushCenterOpacityGradientMax,
    selectedLayeredBrushCenterOpacityGradient,
  ],
]) {
  for (const control of [minimumControl, maximumControl]) {
    control.addEventListener("change", () => {
      const gradient = selector();
      minimumControl.value = String(gradient.min);
      maximumControl.value = String(gradient.max);
      publishState();
    });
  }
}
els.monochromeUnderpainting.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
els.colorFinishStart.addEventListener("change", () => {
  els.colorFinishStart.value = String(
    selectedSharedColorWorkflow().colorFinishStartPercent,
  );
  publishState();
});
for (const control of [
  els.layeredBrushOpacityGradientStart,
  els.layeredBrushOpacityGradientEnd,
  els.layeredBrushWidthTaperStart,
  els.layeredBrushWidthTaperEnd,
]) {
  control.addEventListener("change", () => {
    const effects = selectedLayeredBrushDirectionalEffects();
    els.layeredBrushOpacityGradientStart.value = String(effects.opacityStart);
    els.layeredBrushOpacityGradientEnd.value = String(effects.opacityEnd);
    els.layeredBrushWidthTaperStart.value = String(effects.widthStart);
    els.layeredBrushWidthTaperEnd.value = String(effects.widthEnd);
    publishState();
  });
}
els.rectangleTopRatio.addEventListener("change", () => {
  const minimum = selectedRectangleTopRatio();
  els.rectangleTopRatio.value = String(minimum);
  els.rectangleTopRatioMax.value = String(selectedRectangleTopRatioMax(minimum));
  publishState();
});
els.rectangleTopRatioMax.addEventListener("change", () => {
  const maximum = clampNumber(
    els.rectangleTopRatioMax.value,
    MIN_RECTANGLE_TOP_RATIO,
    MAX_RECTANGLE_TOP_RATIO,
    DEFAULT_RECTANGLE_TOP_RATIO_MAX,
  );
  els.rectangleTopRatioMax.value = String(maximum);
  if (selectedRectangleTopRatio() > maximum) {
    els.rectangleTopRatio.value = String(maximum);
  }
  publishState();
});
els.rectangleMinAspectRatio.addEventListener("change", () => {
  const minimum = selectedRectangleMinAspectRatio();
  els.rectangleMinAspectRatio.value = String(minimum);
  els.rectangleMaxAspectRatio.value =
    String(selectedRectangleMaxAspectRatio(minimum));
  publishState();
});
els.rectangleMaxAspectRatio.addEventListener("change", () => {
  const maximum = clampNumber(
    els.rectangleMaxAspectRatio.value,
    MIN_RECTANGLE_ASPECT_RATIO,
    MAX_RECTANGLE_ASPECT_RATIO,
    DEFAULT_RECTANGLE_ASPECT_RATIO,
  );
  els.rectangleMaxAspectRatio.value = String(maximum);
  if (selectedRectangleMinAspectRatio() > maximum) {
    els.rectangleMinAspectRatio.value = String(maximum);
  }
  publishState();
});
els.rectangleOrientation.addEventListener("change", () => {
  els.rectangleOrientation.value = selectedRectangleOrientation();
  syncAlgorithmRequirements();
  publishState();
});
els.rectangleOrientationTolerance.addEventListener("change", () => {
  els.rectangleOrientationTolerance.value = String(selectedRectangleOrientationToleranceDegrees());
  publishState();
});
for (const control of [
  els.rectanglePreserveArea,
  els.rectangleEdgeDirectedTaper,
  els.rectangleStructureAwareRatio,
  els.rectangleAsymmetricSoftness,
]) {
  control.addEventListener("change", publishState);
}
els.scaleBiasedSurfaceLayerPrior.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
for (const control of [
  els.scaleBiasedSurfaceLayerPriorLayers,
  els.scaleBiasedSurfaceLayerPriorP1Interval,
  els.scaleBiasedSurfaceLayerPriorP2Interval,
  els.scaleBiasedSurfaceLayerPriorP3Interval,
  els.scaleBiasedSurfaceLayerPriorUntil,
]) {
  control.addEventListener("change", () => {
    const settings = scaleBiasedSurfaceLayerPriorSettings();
    els.scaleBiasedSurfaceLayerPriorLayers.value = String(settings.layers);
    els.scaleBiasedSurfaceLayerPriorP1Interval.value = String(settings.p1Interval);
    els.scaleBiasedSurfaceLayerPriorP2Interval.value = String(settings.p2Interval);
    els.scaleBiasedSurfaceLayerPriorP3Interval.value = String(settings.p3Interval);
    els.scaleBiasedSurfaceLayerPriorUntil.value = String(settings.untilFraction * 100);
    publishState();
  });
}
els.discreteLayers.addEventListener("change", () => {
  syncLayerOrderDependency();
  publishState();
});
els.discreteLayerCount.addEventListener("change", () => {
  els.discreteLayerCount.value = String(discreteLayerSettings().count);
  els.discreteLayerMoveRadius.value = String(discreteLayerSettings().moveRadius);
  publishState();
});
els.discreteLayerMoveRadius.addEventListener("change", () => {
  els.discreteLayerMoveRadius.value = String(discreteLayerSettings().moveRadius);
  publishState();
});
els.layerUpdateInterval.addEventListener("input", publishState);
els.phaseRelativeScaleGuard.addEventListener("change", publishState);
els.p1RelativeScaleFloorRatio.addEventListener("input", publishState);
els.p2RelativeScaleFloorRatio.addEventListener("input", publishState);
els.p3RelativeScaleFloorRatio.addEventListener("input", publishState);
els.virtualBoundedDepth.addEventListener("change", publishState);
els.virtualGofDensity.addEventListener("change", publishState);
els.virtualCameraShare.addEventListener("input", publishState);
for (const element of [els.virtualCameraMaxAngle, els.virtualCameraCount, els.virtualCameraFov]) {
  element.addEventListener("input", () => {
    updateVirtualCameraCoverageEstimate();
    publishState();
  });
}
els.stageAwareGrowth.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
els.structureGuidedAllocation.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
els.structureRegionGrid.addEventListener("change", () => {
  els.structureRegionGrid.value = String(phase39Variants().structureRegionGrid);
  publishState();
});
els.midTrainingOverdensityCorrection.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
els.overdensityCorrectionSchedule.addEventListener("change", () => {
  syncAlgorithmRequirements();
  publishState();
});
els.overdensityCorrectionInterval.addEventListener("change", () => {
  els.overdensityCorrectionInterval.value = String(
    phase39Variants().overdensityCorrectionInterval,
  );
  publishState();
});
els.overdensityDonorPercent.addEventListener("change", () => {
  els.overdensityDonorPercent.value = String(
    phase39Variants().overdensityDonorFraction * 100,
  );
  publishState();
});
for (const element of [els.stageGrowthP1, els.stageGrowthP2, els.stageGrowthP3]) {
  element.addEventListener("input", publishState);
}
els.detailCoherence.addEventListener("input", publishState);
els.densifyInterval.addEventListener("input", publishState);
els.growthPercentage.addEventListener("input", publishState);
els.growthApplyUntil.addEventListener("input", publishState);
els.growthSignalThreshold.addEventListener("input", publishState);
els.retryWebGpuButton.addEventListener("click", async () => {
  if (trainingLifecycleInputLocked() || state.webgpu.supported) return;
  state.webgpu = { supported: false, renderer: null, reason: "checking", limits: null, adapterInfo: null };
  els.backendText.textContent = "checking";
  setStatus("checking gpu");
  els.retryWebGpuButton.disabled = true;
  try {
    const available = await detectWebGpu();
    setStatus(available ? (state.image ? "image loaded" : "idle") : "gpu unavailable");
  } finally {
    publishState();
  }
});
els.startButton.addEventListener("click", () => {
  startTraining().catch((error) => {
    setStatus("error");
    setTrainingMessage(`Training failed: ${error.message}`, "error");
    log(`training click failed: ${error.message}`);
    eventLog(`training click failed: ${error.message}`);
  });
});
els.algorithmSelect.addEventListener("change", () => {
  const algorithm = selectedAlgorithm();
  els.stageAwareGrowth.checked = stageAwareGrowthDefaultForAlgorithm(algorithm);
  const resultAlgorithm = trainedResultAlgorithm();
  const tiltAvailable = resultAlgorithm
    ? Boolean(resultAlgorithm.capabilities.virtualCameras)
    : Boolean(algorithm.capabilities.virtualCameras);
  if (!tiltAvailable && !els.tiltCanvas.hidden) {
    restorePrimaryCanvas();
    destroyTiltViewer();
  }
  if (!tiltAvailable && document.documentElement.dataset.activeDetailTab === "tilt") {
    activateDetailTab("training");
  }
  document.documentElement.dataset.algorithm = algorithm.id;
  syncAlgorithmRequirements();
  syncLayerOrderDependency();
  syncVirtualCameraDependency();
  updateTiltControlState();
  updateExportPanel();
  eventLog(`algorithm selected ${algorithm.label}`);
  publishState();
});
els.pauseButton.addEventListener("click", togglePause);
els.stopButton.addEventListener("click", stopTraining);
els.resetButton.addEventListener("click", resetTrainingState);
els.savePlyButton.addEventListener("click", () => {
  saveExport().catch((error) => log(error.message));
});
els.savePngButton.addEventListener("click", () => {
  saveExport({ formatKey: "png" }).catch((error) => log(error.message));
});
els.pngExportResolution.addEventListener("change", () => {
  syncPngExportResolutionUi();
  updateExportPanel();
  publishState();
});
els.pngExportLongSide.addEventListener("input", () => {
  els.pngExportResolution.value = "custom";
  const value = Number(els.pngExportLongSide.value);
  if (Number.isFinite(value) && value >= MIN_PNG_EXPORT_LONG_SIDE) {
    const spec = pngExportFrameSpec();
    els.pngExportResolutionStatus.textContent = `${spec.width} x ${spec.height}px`;
  }
});
els.pngExportLongSide.addEventListener("change", () => {
  els.pngExportResolution.value = "custom";
  updateExportPanel();
  publishState();
});


els.trainingLogTab.addEventListener("click", () => {
  activateDetailTab("training");
  refreshOutsidePreview().catch((error) => log(error.message));
});
els.eventLogTab.addEventListener("click", () => activateDetailTab("event"));
els.clearLogButton.addEventListener("click", () => {
  els.log.textContent = "";
});
els.splatsTab.addEventListener("click", () => {
  activateDetailTab("splats");
  // The Splats tab is an inspection surface, so entering it must not inherit
  // the Original toggle from the Train tab. setPreviewMode also refreshes the
  // retained GPU result without rebuilding optimizer state.
  if (document.documentElement.dataset.activeDetailTab === "splats") {
    setPreviewMode("splats");
  }
});
els.exportTab.addEventListener("click", () => activateDetailTab("export"));
els.tiltTab.addEventListener("click", () => {
  activateDetailTab("tilt");
  if (!tiltViewerReady()) {
    els.tiltStatus.textContent = tiltViewerAvailabilityMessage();
    updateTiltControlState();
    return;
  }
  if (state.tilt.loading) return;
  loadTiltViewer().catch((error) => {
    log(error.message);
    eventLog(error.message);
  });
});
els.logTabs.addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = [
    els.trainingLogTab,
    els.splatsTab,
    els.exportTab,
    els.eventLogTab,
    els.tiltTab,
  ].filter((tab) => !tab.disabled);
  if (!tabs.length) return;
  const current = tabs.indexOf(event.target.closest('[role="tab"]'));
  if (current < 0) return;
  let next = current;
  if (event.key === "Home") next = 0;
  else if (event.key === "End") next = tabs.length - 1;
  else next = (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  event.preventDefault();
  tabs[next].click();
  tabs[next].focus();
});
for (const control of [els.tiltPitch, els.tiltYaw]) {
  control.addEventListener("input", scheduleTiltInputs);
}
els.tiltCameraMarkers.addEventListener("change", () => {
  state.tilt.cameraMarkersVisible = els.tiltCameraMarkers.checked;
  state.tilt.controller?.setCameraMarkersVisible(state.tilt.cameraMarkersVisible);
  updateTiltControlState();
  publishState();
});
els.tiltRadiusMode.addEventListener("change", () => {
  state.tilt.radiusMode = state.tilt.controller?.setRadiusMode(els.tiltRadiusMode.value) || els.tiltRadiusMode.value;
  els.tiltRadiusMode.value = state.tilt.radiusMode;
  updateTiltControlState();
  publishState();
});
els.tiltFrontButton.addEventListener("click", () => {
  els.tiltPitch.value = "0";
  els.tiltYaw.value = "0";
  applyTiltInputs();
});
els.tiltRefreshButton.addEventListener("click", () => {
  loadTiltViewer({ force: true }).catch((error) => {
    log(error.message);
    eventLog(error.message);
  });
});
els.tiltTrainingViewsButton.addEventListener("click", () => {
  showTiltTrainingViews().catch((error) => {
    els.tiltStatus.textContent = `Training views failed: ${error.message}`;
    log(error.message);
    eventLog(error.message);
    state.tilt.teacherViewsLoading = false;
    updateTiltControlState();
  });
});
for (const [button, mode] of [
  [els.tiltOriginalView, "original"],
  [els.tiltOverlayView, "overlay"],
  [els.tiltSplatsView, "splats"],
]) button.addEventListener("click", () => setTiltDisplayMode(mode, { focusCamera: true }));
for (const control of [
  els.splatOpacity,
  els.splatKernelFalloff,
  els.splatScale,
  els.splatAspectRatio,
]) {
  control.addEventListener("input", () => queueSplatAdjustments());
  control.addEventListener("change", () => queueSplatAdjustments({ immediate: true }));
}
els.splatParameterEffects.addEventListener("change", () => {
  renderSplatInspector();
  queueSplatAdjustments({ immediate: true });
});
els.splatShapeGaussian.addEventListener("click", () => setSplatPreviewShape("gaussian"));
els.splatShapeRectangle.addEventListener("click", () => setSplatPreviewShape("rectangle"));
els.splatShapeOpaqueBrush.addEventListener("click", () => setSplatPreviewShape("opaque-brush"));
els.splatAlphaBackground.addEventListener("input", () => {
  refreshOutsidePreview().catch((error) => log(error.message));
});
els.splatSmallFirstOrder.addEventListener("change", () => {
  refreshOutsidePreview().catch((error) => log(error.message));
});
els.resetSplatAdjustments.addEventListener("click", () => {
  resetSplatAdjustments();
});
els.actualSizeButton.addEventListener("click", () => setCanvasView("actual"));
els.fitViewButton.addEventListener("click", () => setCanvasView("fit"));
els.viewer.addEventListener(
  "wheel",
  (event) => {
    if (!state.image || event.target === els.tiltCanvas || !(event.target instanceof HTMLCanvasElement)) return;
    event.preventDefault();
    if (canvasViewInputLocked()) return;
    zoomCanvasAt(event.clientX, event.clientY, event.deltaY);
  },
  { passive: false },
);
els.viewer.addEventListener("contextmenu", (event) => {
  if (event.target instanceof HTMLCanvasElement) event.preventDefault();
});
els.viewer.addEventListener("pointerdown", (event) => {
  const directPointer = event.pointerType !== "mouse";
  const mousePanButton = event.button === 0 || event.button === 2;
  if (!state.image || event.target === els.tiltCanvas || (!directPointer && !mousePanButton) || !(event.target instanceof HTMLCanvasElement)) return;
  event.preventDefault();
  if (canvasViewInputLocked()) return;
  state.canvasView.pointerId = event.pointerId;
  state.canvasView.lastX = event.clientX;
  state.canvasView.lastY = event.clientY;
  state.canvasPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (state.canvasPointers.size === 2) {
    const points = [...state.canvasPointers.values()];
    state.canvasPinch = {
      distance: Math.max(1, Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y)),
      centerX: (points[0].x + points[1].x) * 0.5,
      centerY: (points[0].y + points[1].y) * 0.5,
      scale: state.canvasView.scale,
      panX: state.canvasView.panX,
      panY: state.canvasView.panY,
    };
  }
  els.viewer.classList.add("is-panning");
  els.viewer.setPointerCapture(event.pointerId);
});
els.viewer.addEventListener("pointermove", (event) => {
  if (canvasViewInputLocked()) {
    if (state.canvasPointers.has(event.pointerId)) cancelCanvasViewGesture();
    return;
  }
  if (!state.canvasPointers.has(event.pointerId)) return;
  state.canvasPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  state.canvasView.mode = "custom";
  if (state.canvasPointers.size >= 2 && state.canvasPinch) {
    const points = [...state.canvasPointers.values()].slice(0, 2);
    const distance = Math.max(1, Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y));
    const centerX = (points[0].x + points[1].x) * 0.5;
    const centerY = (points[0].y + points[1].y) * 0.5;
    state.canvasView.scale = Math.max(0.02, Math.min(32, state.canvasPinch.scale * distance / state.canvasPinch.distance));
    state.canvasView.panX = state.canvasPinch.panX + centerX - state.canvasPinch.centerX;
    state.canvasView.panY = state.canvasPinch.panY + centerY - state.canvasPinch.centerY;
  } else {
    state.canvasView.panX += event.clientX - state.canvasView.lastX;
    state.canvasView.panY += event.clientY - state.canvasView.lastY;
  }
  state.canvasView.lastX = event.clientX;
  state.canvasView.lastY = event.clientY;
  applyCanvasView();
});
function endCanvasPan(event) {
  if (!state.canvasPointers.has(event.pointerId)) return;
  state.canvasPointers.delete(event.pointerId);
  state.canvasPinch = null;
  const remaining = state.canvasPointers.entries().next().value;
  state.canvasView.pointerId = remaining?.[0] ?? null;
  if (remaining) {
    state.canvasView.lastX = remaining[1].x;
    state.canvasView.lastY = remaining[1].y;
  } else {
    els.viewer.classList.remove("is-panning");
  }
  if (els.viewer.hasPointerCapture(event.pointerId)) els.viewer.releasePointerCapture(event.pointerId);
  publishState();
}
els.viewer.addEventListener("pointerup", endCanvasPan);
els.viewer.addEventListener("pointercancel", endCanvasPan);
