function setInputControlsDisabled(disabled) {
  for (const element of [
    els.fileInput,
    els.algorithmSelect,
    els.trainSize,
    els.initialSplatCount,
    els.adaptiveGridInitializationFraction,
    els.finalSplatCount,
    els.capacityMode,
    els.memoryLimiterUnlock,
    els.stepCount,
    els.previewRefresh,
    els.liveQualityMetrics,
    els.currentContributionCompaction,
    els.currentContributionCompactionStart,
    els.currentContributionCompactionMaxRemoval,
    els.currentContributionCompactionNearZero,
    els.currentContributionCompactionWindow,
    els.monochromeUnderpainting,
    els.colorFinishStart,
    els.tileCullingToggle,
    els.trainLayerOrder,
    els.layerAwareAccumulation,
    els.discreteLayers,
    els.discreteLayerCount,
    els.discreteLayerMoveRadius,
    els.layerUpdateInterval,
    els.scaleBiasedSurfaceLayerPrior,
    els.scaleBiasedSurfaceLayerPriorLayers,
    els.scaleBiasedSurfaceLayerPriorP1Interval,
    els.scaleBiasedSurfaceLayerPriorP2Interval,
    els.scaleBiasedSurfaceLayerPriorP3Interval,
    els.scaleBiasedSurfaceLayerPriorUntil,
    els.positionLearningRate,
    els.colorLearningRate,
    els.opacityLearningRate,
    els.alphaLossWeight,
    els.rectangleLearnedOpacityMin,
    els.rectangleLearnedOpacityMax,
    els.rectangleOpacityGradientMin,
    els.rectangleOpacityGradientMax,
    els.rectangleCenterOpacityGradientMin,
    els.rectangleCenterOpacityGradientMax,
    els.layeredBrushLearnedOpacityMin,
    els.layeredBrushLearnedOpacityMax,
    els.layeredBrushOpacityGradientStart,
    els.layeredBrushOpacityGradientEnd,
    els.layeredBrushCenterOpacityGradientMin,
    els.layeredBrushCenterOpacityGradientMax,
    els.layeredBrushWidthTaperStart,
    els.layeredBrushWidthTaperEnd,
    els.layeredBrushLocalColorFlow,
    els.layeredBrushStrokePersistence,
    els.layeredBrushMinAspectRatio,
    els.layeredBrushMaxAspectRatio,
    els.layeredBrushRibbonAspectFloor,
    els.layeredBrushAccentAspectFloor,
    els.rectangleTopRatio,
    els.rectangleTopRatioMax,
    els.rectangleMinAspectRatio,
    els.rectangleMaxAspectRatio,
    els.rectangleOrientation,
    els.rectanglePreserveArea,
    els.rectangleEdgeDirectedTaper,
    els.rectangleStructureAwareRatio,
    els.rectangleAsymmetricSoftness,
    els.virtualBoundedDepth,
    els.virtualGofDensity,
    els.virtualCameraShare,
    els.virtualCameraMaxAngle,
    els.virtualCameraCount,
    els.virtualCameraFov,
    els.stageAwareGrowth,
    els.stageGrowthP1,
    els.stageGrowthP2,
    els.stageGrowthP3,
    els.structureGuidedAllocation,
    els.structureRegionGrid,
    els.midTrainingOverdensityCorrection,
    els.overdensityCorrectionSchedule,
    els.overdensityCorrectionInterval,
    els.overdensityDonorPercent,
    els.scaleLearningRate,
    els.rotationLearningRate,
    els.thetaAlignRate,
    els.maxAnisotropy,
    els.maxPlanarScale,
    els.boundarySigma,
    els.detailCoherence,
    els.phaseRelativeScaleGuard,
    els.p1RelativeScaleFloorRatio,
    els.p2RelativeScaleFloorRatio,
    els.p3RelativeScaleFloorRatio,
    els.densifyInterval,
    els.growthPercentage,
    els.growthApplyUntil,
    els.growthSignalThreshold,
    els.retryWebGpuButton,
    els.pathInput,
    els.loadImageButton,
    els.clearImageButton,
    els.sampleButton,
    els.pathButton,
  ]) {
    element.disabled = disabled;
  }
  syncTrainSizeUi();
  if (!disabled) {
    syncLayerOrderDependency();
    syncVirtualCameraDependency();
  }
}

function syncAlgorithmRequirements() {
  const opaqueLayered = algorithmUsesOpaqueLayeredPaint();
  const brushSelected = algorithmUsesLayeredOpaqueBrush();
  const rectangleSelected = algorithmUsesRectangleKernel();
  document.documentElement.dataset.opaquePaintSettings = "visible";
  if (brushSelected) {
    if (!els.discreteLayerMoveRadius.dataset.nonBrushValue) {
      els.discreteLayerMoveRadius.dataset.nonBrushValue =
        els.discreteLayerMoveRadius.value || String(DEFAULT_DISCRETE_LAYER_MOVE_RADIUS);
    }
    els.discreteLayerMoveRadius.value = String(LAYERED_OPAQUE_BRUSH_LAYER_MOVE_RADIUS);
  } else if (els.discreteLayerMoveRadius.dataset.nonBrushValue) {
    els.discreteLayerMoveRadius.value = els.discreteLayerMoveRadius.dataset.nonBrushValue;
    delete els.discreteLayerMoveRadius.dataset.nonBrushValue;
  }
  for (const [control, fallback] of [
    [els.alphaLossWeight, DEFAULT_ALPHA_LOSS_WEIGHT],
  ]) {
    if (brushSelected) {
      if (!control.dataset.nonOpaqueValue) control.dataset.nonOpaqueValue = control.value || String(fallback);
      control.value = "0";
    } else if (control.dataset.nonOpaqueValue) {
      control.value = control.dataset.nonOpaqueValue;
      delete control.dataset.nonOpaqueValue;
    }
  }
  if (opaqueLayered) {
    els.trainLayerOrder.checked = true;
    els.layerAwareAccumulation.checked = true;
    els.discreteLayers.checked = true;
  }
  els.opacityLearningRate.disabled = state.running;
  els.structureRegionGrid.disabled = state.running || (
    !els.structureGuidedAllocation.checked && !els.midTrainingOverdensityCorrection.checked
  );
  els.overdensityDonorPercent.disabled = state.running || !els.midTrainingOverdensityCorrection.checked;
  els.overdensityCorrectionSchedule.disabled = state.running || !els.midTrainingOverdensityCorrection.checked;
  els.overdensityCorrectionInterval.disabled = state.running ||
    !els.midTrainingOverdensityCorrection.checked ||
    phase39Variants().overdensityCorrectionSchedule !== "interval";
  els.alphaLossWeight.disabled = state.running || brushSelected;
  els.adaptiveGridInitializationFraction.disabled = state.running || algorithmUsesPaintKernel();
  els.adaptiveGridInitializationFraction.title = algorithmUsesPaintKernel()
    ? "Paint algorithms use deterministic image-importance BSP placement."
    : "Moves this percentage of the initial grid centers toward multiscale image structure on WebGPU before iteration 1.";
  els.rectangleLearnedOpacityMin.disabled = state.running || !rectangleSelected;
  els.rectangleLearnedOpacityMax.disabled = state.running || !rectangleSelected;
  els.rectangleOpacityGradientMin.disabled = state.running || !rectangleSelected;
  els.rectangleOpacityGradientMax.disabled = state.running || !rectangleSelected;
  els.rectangleCenterOpacityGradientMin.disabled = state.running || !rectangleSelected;
  els.rectangleCenterOpacityGradientMax.disabled = state.running || !rectangleSelected;
  els.layeredBrushLearnedOpacityMin.disabled = state.running || !brushSelected;
  els.layeredBrushLearnedOpacityMax.disabled = state.running || !brushSelected;
  els.layeredBrushOpacityGradientStart.disabled = state.running || !brushSelected;
  els.layeredBrushOpacityGradientEnd.disabled = state.running || !brushSelected;
  els.layeredBrushCenterOpacityGradientMin.disabled = state.running || !brushSelected;
  els.layeredBrushCenterOpacityGradientMax.disabled = state.running || !brushSelected;
  els.layeredBrushWidthTaperStart.disabled = state.running || !brushSelected;
  els.layeredBrushWidthTaperEnd.disabled = state.running || !brushSelected;
  els.layeredBrushLocalColorFlow.disabled = state.running || !brushSelected;
  els.layeredBrushStrokePersistence.disabled = state.running || !brushSelected;
  els.layeredBrushMinAspectRatio.disabled = state.running || !brushSelected;
  els.layeredBrushMaxAspectRatio.disabled = state.running || !brushSelected;
  els.layeredBrushRibbonAspectFloor.disabled =
    state.running || !brushSelected || !els.layeredBrushStrokePersistence.checked;
  els.layeredBrushAccentAspectFloor.disabled =
    state.running || !brushSelected || !els.layeredBrushStrokePersistence.checked;
  els.monochromeUnderpainting.disabled = state.running;
  els.colorFinishStart.disabled = state.running || !els.monochromeUnderpainting.checked;
  const contributionCompactionDisabled = state.running || !els.currentContributionCompaction.checked;
  els.currentContributionCompaction.disabled = state.running;
  els.currentContributionCompactionStart.disabled = contributionCompactionDisabled;
  els.currentContributionCompactionInterval.disabled = contributionCompactionDisabled;
  els.currentContributionCompactionMaxRemoval.disabled = contributionCompactionDisabled;
  els.currentContributionCompactionNearZero.disabled = contributionCompactionDisabled;
  els.currentContributionCompactionWindow.disabled = contributionCompactionDisabled;
  els.rectangleTopRatio.disabled = state.running || !rectangleSelected;
  els.rectangleTopRatioMax.disabled = state.running || !rectangleSelected;
  els.rectangleMinAspectRatio.disabled = state.running || !rectangleSelected;
  els.rectangleMaxAspectRatio.disabled = state.running || !rectangleSelected;
  els.rectangleOrientation.disabled = state.running || !rectangleSelected;
  els.scaleBiasedSurfaceLayerPrior.disabled = state.running;
  for (const control of [
    els.scaleBiasedSurfaceLayerPriorLayers,
    els.scaleBiasedSurfaceLayerPriorP1Interval,
    els.scaleBiasedSurfaceLayerPriorP2Interval,
    els.scaleBiasedSurfaceLayerPriorP3Interval,
    els.scaleBiasedSurfaceLayerPriorUntil,
  ]) {
    control.disabled = state.running || !els.scaleBiasedSurfaceLayerPrior.checked;
  }
  els.rectangleTopRatio.title = rectangleSelected
    ? "Minimum top-edge / base-edge ratio in the deterministic Rectangle shape range; 0 allows triangle tips."
    : "Available for Rectangle Splats.";
  els.rectangleTopRatioMax.title = rectangleSelected
    ? "Maximum top-edge / base-edge ratio in the deterministic Rectangle shape range; 1 makes the parallel edges equal."
    : "Available for Rectangle Splats.";
  els.rectangleMinAspectRatio.title = rectangleSelected
    ? "Minimum long-side / short-side aspect ratio; 1 allows square footprints."
    : "Available for Rectangle Splats.";
  els.rectangleMaxAspectRatio.title = rectangleSelected
    ? "Maximum long-side / short-side aspect ratio; this Rectangle value overrides Shared Max anisotropy."
    : "Available for Rectangle Splats.";
  els.rectangleOrientation.title = rectangleSelected
    ? "Free keeps structure-guided rotation; Vertical or Horizontal locks the long axis."
    : "Available for Rectangle Splats.";
  for (const control of [
    els.rectanglePreserveArea,
    els.rectangleEdgeDirectedTaper,
    els.rectangleStructureAwareRatio,
    els.rectangleAsymmetricSoftness,
  ]) {
    control.disabled = state.running || !rectangleSelected;
  }
  for (const control of [els.stageGrowthP1, els.stageGrowthP2, els.stageGrowthP3]) {
    control.disabled = state.running || !els.stageAwareGrowth.checked;
  }
  els.structureGuidedAllocation.disabled = state.running;
}

function syncLayerOrderDependency() {
  syncAlgorithmRequirements();
  if (els.trainLayerOrder.checked) els.tileCullingToggle.checked = true;
  els.tileCullingToggle.disabled = state.running || els.trainLayerOrder.checked;
  els.layerUpdateInterval.disabled = state.running || !els.trainLayerOrder.checked;
  const discreteAvailable = els.trainLayerOrder.checked && !algorithmUsesVirtualCameras();
  const opaqueLayered = algorithmUsesOpaqueLayeredPaint();
  els.trainLayerOrder.disabled = state.running || opaqueLayered;
  els.layerAwareAccumulation.disabled = state.running || !discreteAvailable || opaqueLayered;
  els.discreteLayers.disabled = state.running || !discreteAvailable || opaqueLayered;
  els.discreteLayerCount.disabled = state.running || !discreteAvailable || (!els.discreteLayers.checked && !els.layerAwareAccumulation.checked);
  els.discreteLayerMoveRadius.disabled = state.running || !discreteAvailable || !els.discreteLayers.checked;
}

function updateVirtualCameraCoverageEstimate() {
  if (!state.image) {
    els.virtualCameraCoverageEstimate.textContent = "Teacher coverage: load an image to estimate.";
    delete document.documentElement.dataset.virtualCameraCoverageEstimate;
    return null;
  }
  if (!algorithmUsesVirtualCameras()) {
    els.virtualCameraCoverageEstimate.textContent = "Teacher coverage is evaluated only for GS Virtual Camera Sampling.";
    delete document.documentElement.dataset.virtualCameraCoverageEstimate;
    return null;
  }
  const variants = virtualCameraSamplingVariants(algorithmUsesVirtualCameras());
  const stats = virtualCameraCoverageStats(state.image.width, state.image.height, variants);
  const percent = (value) => `${(value * 100).toFixed(1)}%`;
  els.virtualCameraCoverageEstimate.textContent = `Teacher coverage min / mean / max: ${percent(stats.minimum)} / ${percent(stats.mean)} / ${percent(stats.maximum)}`;
  document.documentElement.dataset.virtualCameraCoverageEstimate = JSON.stringify(stats);
  return stats;
}

function syncVirtualCameraDependency() {
  const virtualAlgorithm = algorithmUsesVirtualCameras();
  const panel = document.querySelector(".virtual-camera-panel");
  if (panel) panel.hidden = !virtualAlgorithm;
  const disabled = state.running || !virtualAlgorithm;
  els.virtualBoundedDepth.disabled = disabled;
  els.virtualGofDensity.disabled = disabled;
  els.virtualCameraShare.disabled = disabled;
  els.virtualCameraMaxAngle.disabled = disabled;
  els.virtualCameraCount.disabled = disabled;
  els.virtualCameraFov.disabled = disabled;
  updateVirtualCameraCoverageEstimate();
}

function pausedRuntimeControls() {
  return [
    els.previewRefresh,
    els.tileCullingToggle,
    els.positionLearningRate,
    els.colorLearningRate,
    els.opacityLearningRate,
    els.alphaLossWeight,
    els.scaleLearningRate,
    els.rotationLearningRate,
    els.thetaAlignRate,
    els.maxAnisotropy,
    els.maxPlanarScale,
    els.boundarySigma,
    els.detailCoherence,
    els.densifyInterval,
    els.growthPercentage,
    els.growthSignalThreshold,
  ];
}

function setPausedRuntimeControlsEnabled(enabled) {
  for (const element of pausedRuntimeControls()) element.disabled = !enabled;
  if (algorithmUsesLayeredOpaqueBrush()) {
    els.alphaLossWeight.disabled = true;
  }
  document.documentElement.dataset.pausedRuntimeControls = String(enabled);
}
