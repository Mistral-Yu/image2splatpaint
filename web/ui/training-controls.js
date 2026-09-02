function setInputControlsDisabled(disabled) {
  for (const element of [
    els.fileInput,
    els.algorithmSelect,
    els.flowTrainingPath,
    els.rectanglePaintShape,
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
    els.rectangleOrientationTolerance,
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
    els.flowSplatFusionStrokeTexture,
    els.flowSplatFusionStrokeOptimization,
    els.flowSplatFusionPaintCurriculum,
    els.flowSplatFusionFixedOpacity,
    els.flowSplatFusionStartingWidthDivisor,
    els.flowSplatFusionStartingLengthPercent,
    els.flowSplatFusionResidualMovePx,
    els.flowSplatFusionScaleMatchedResidualRepaint,
    els.flowSplatFusionInitialWidthMin,
    els.flowSplatFusionInitialWidthMax,
    els.flowSplatFusionFrontWidthMax,
    els.flowSplatFusionFrontWidthLearning,
    els.flowSplatFusionColorAnchor,
    els.flowSplatFusionWidthPercent,
    els.flowSplatFusionSplatSizeVariation,
    els.flowSplatFusionEdgeAccents,
    els.flowLinkedSplatMin,
    els.flowLinkedSplatMax,
    els.flowStrokeCoherence,
    els.flowInternalBendControlPointCount,
    els.flowInternalBendControlPointPositions,
    els.flowSplatFusionMovementLimit,
    els.flowSplatUnderpainting,
    els.flowSplatBackcoatFromP1,
    els.flowSplatUnderpaintPercent,
    els.flowSplatBackcoatSizeVariation,
    els.flowSplatFusionMaxArcPercent,
  ]) {
    element.disabled = disabled;
  }
  syncTrainSizeUi();
  if (!disabled) {
    syncLayerOrderDependency();
    syncVirtualCameraDependency();
  }
}

const FLOW_BRUSH_DEFAULT_MIN_ASPECT_RATIO = 2.2;
const FLOW_STAGE_GROWTH_DEFAULTS = Object.freeze([20, 40, 40]);

function syncFlowBrushAspectDefaults(flowSelected) {
  const input = els.layeredBrushMinAspectRatio;
  const nextScope = flowSelected ? "flow" : "standard";
  const currentScope = input.dataset.valueScope || "standard";
  if (currentScope === nextScope) return;
  if (currentScope === "flow") input.dataset.flowValue = input.value;
  else input.dataset.standardValue = input.value;
  input.value = nextScope === "flow"
    ? input.dataset.flowValue || String(FLOW_BRUSH_DEFAULT_MIN_ASPECT_RATIO)
    : input.dataset.standardValue || String(LIMITS.maxAnisotropyMin);
  input.dataset.valueScope = nextScope;
}

function syncFlowStageGrowthDefaults(flowSelected) {
  const controls = [els.stageGrowthP1, els.stageGrowthP2, els.stageGrowthP3];
  const nextScope = flowSelected ? "flow" : "standard";
  const currentScope = controls[0].dataset.valueScope || "standard";
  if (currentScope === nextScope) return;
  for (let index = 0; index < controls.length; index += 1) {
    const control = controls[index];
    control.dataset[`${currentScope}Value`] = control.value;
    control.value = control.dataset[`${nextScope}Value`] || (
      nextScope === "flow" ? String(FLOW_STAGE_GROWTH_DEFAULTS[index]) : control.defaultValue
    );
    control.dataset.valueScope = nextScope;
  }
}

function syncAlgorithmRequirements() {
  const algorithm = selectedAlgorithm();
  const opaqueLayered = algorithmUsesOpaqueLayeredPaint();
  const rectangleAlgorithmSelected =
    algorithm.id === RECTANGLE_SPLATS_ALGORITHM_ID ||
    algorithm.id === LAYERED_OPAQUE_BRUSH_ALGORITHM_ID;
  const brushSelected = algorithmUsesLayeredOpaqueBrush(algorithm);
  const rectangleSelected = algorithmUsesRectangleKernel(algorithm);
  const flowSelected = algorithmUsesFlowStrokeTraining(algorithm);
  const internalBend = Boolean(algorithm.capabilities.internalBend);
  const sharedFlow = Boolean(algorithm.capabilities.flowBirthLinked || internalBend);
  syncFlowBrushAspectDefaults(sharedFlow);
  syncFlowStageGrowthDefaults(sharedFlow);
  if (flowSelected) {
    const budgetKey = "flow";
    const previousBudgetKey = els.flowSplatUnderpaintPercent.dataset.algorithm;
    if (previousBudgetKey && previousBudgetKey !== budgetKey) {
      els.flowSplatUnderpaintPercent.dataset[`${previousBudgetKey}Value`] =
        els.flowSplatUnderpaintPercent.value;
    }
    if (previousBudgetKey !== budgetKey) {
      els.flowSplatUnderpaintPercent.value =
        els.flowSplatUnderpaintPercent.dataset[`${budgetKey}Value`] ||
        "10";
      els.flowSplatUnderpaintPercent.dataset.algorithm = budgetKey;
    }
  }
  document.documentElement.dataset.flowSplatFusion = String(flowSelected);
  els.flowSplatFusionPanelSummary.textContent = "Curved Brush settings";
  els.flowSplatFusionPanelNote.textContent =
    internalBend
      ? "Experimental: one independently curved Brush Splat per row. Control-point positions are fixed settings while each Splat's bend amount learns with position, scale, rotation and RGB on the shared WebGPU optimizer. Training grows from Initial to Max splats through P1/P2/P3."
      : sharedFlow
      ? "Each Brush Splat learns position, scale and rotation. Split/clone ancestry forms linked strokes within the selected Min/Max group size; independent births remain free. Shared training controls growth, structure, layers, optimizer and visibility."
      : "Paint with curved Brush Splats using linked strokes or learned internal bends.";
  els.initialSplatCount.closest("label").classList.toggle("flow-hidden", flowSelected && !sharedFlow);
  els.initialSplatCountLabel.textContent = "Initial splats";
  els.finalSplatCountLabel.textContent = "Max splats";
  els.initialSplatCount.closest("label").title = sharedFlow
    ? "Initial trainable Brush dabs, in addition to the protected backcoat, within Max splats."
    : "Number of splats created before the first training iteration.";
  els.finalSplatCount.closest("label").title = internalBend
    ? "Final physical Splat count. Active rows grow progressively from Initial splats; one Splat owns one internal bend."
    : sharedFlow
    ? "Total Splat budget, including the protected backcoat and trainable Brush dabs."
    : flowSelected
    ? "Physical Splat budget. Detail uses complete stroke chains; the optional compact Brush backcoat uses individual Splats, so only the detail remainder rounds down."
    : "Maximum number of splats available to density growth. Loading an image does not change this value.";
  els.trainSize.min = String(LIMITS.trainSizeMin);
  els.trainSize.max = flowSelected ? "512" : String(LIMITS.trainSizeMax);
  els.finalSplatCount.min = flowSelected ? "256" : String(LIMITS.splatsMin);
  els.finalSplatCount.max = flowSelected ? "14000" : String(LIMITS.splatsMax);
  els.finalSplatCount.step = "4";
  els.stepCount.removeAttribute("max");
  els.stepCount.step = "100";
  els.stepCount.closest("label").title = flowSelected && !sharedFlow
    ? "Requested schedule limit. Flow runs one real optimizer update per 10 requested steps, including settling, while preserving growth stages. A 3000 limit normally runs 300 actual steps; the status shows actual steps."
    : "Number of training optimizer iterations.";
  els.flowSplatFusionColorAnchor.disabled = state.running || !flowSelected;
  els.flowSplatFusionStrokeTexture.disabled = state.running || !flowSelected;
  els.flowSplatFusionStrokeOptimization.disabled = state.running || !flowSelected;
  els.flowSplatFusionPaintCurriculum.disabled = state.running || !flowSelected;
  els.flowSplatFusionFixedOpacity.disabled = state.running || !flowSelected;
  const paintCurriculum = flowSelected && els.flowSplatFusionPaintCurriculum.checked;
  els.flowSplatFusionStartingWidthDivisor.disabled = state.running || !paintCurriculum;
  els.flowSplatFusionStartingLengthPercent.disabled = state.running || !paintCurriculum;
  els.flowSplatFusionResidualMovePx.disabled = state.running || !flowSelected;
  els.flowSplatFusionScaleMatchedResidualRepaint.disabled = state.running || !flowSelected;
  els.flowSplatFusionInitialWidthMin.disabled = state.running || !flowSelected;
  els.flowSplatFusionInitialWidthMax.disabled = state.running || !flowSelected;
  els.flowSplatFusionFrontWidthMax.disabled = state.running || !flowSelected;
  els.flowSplatFusionFrontWidthLearning.disabled = state.running || !flowSelected;
  els.flowSplatFusionWidthPercent.disabled = state.running || !flowSelected;
  els.flowSplatFusionSplatSizeVariation.disabled = state.running || !flowSelected;
  els.flowSplatFusionEdgeAccents.disabled = state.running || !flowSelected;
  els.flowLinkedSplatRange.classList.toggle("flow-hidden", !sharedFlow || internalBend);
  els.flowLinkedSplatMin.disabled = state.running || !sharedFlow || internalBend;
  els.flowLinkedSplatMax.disabled = state.running || !sharedFlow || internalBend;
  els.flowStrokeCoherenceLabel.classList.toggle("flow-hidden", !sharedFlow || internalBend);
  els.flowStrokeCoherence.disabled = state.running || !sharedFlow || internalBend;
  els.flowInternalBendControlPointCountLabel.classList.toggle("flow-hidden", !internalBend);
  els.flowInternalBendControlPointPositionsLabel.classList.toggle("flow-hidden", !internalBend);
  els.flowInternalBendControlPointCount.disabled = state.running || !internalBend;
  els.flowInternalBendControlPointPositions.disabled = state.running || !internalBend;
  els.flowSplatFusionMovementLimit.disabled = state.running || !flowSelected;
  els.flowSplatUnderpainting.disabled = state.running || !flowSelected;
  els.flowSplatUnderpainting.closest("label").title = sharedFlow
    ? "Adds source-mean-color Curved Brush underpainting from P1. Most rows join the normal position, scale, rotation, color, growth and linked-stroke training; only a small rear safety prefix stays protected to prevent background exposure. Included in Max splats; opacity follows Fixed stroke opacity."
    : "Adds a fixed-geometry source-colored compact Brush Splat grid behind every stroke in the final stage. Each source-opaque grid cell stays inside one flat alpha=0.99 interior. Enable Backcoat from P1 to include it throughout training. Standard alpha and the physical Max splats budget are preserved.";
  els.flowSplatBackcoatFromP1.disabled = state.running || !flowSelected || !els.flowSplatUnderpainting.checked;
  els.flowSplatBackcoatSizeVariation.disabled = state.running || !flowSelected || !els.flowSplatUnderpainting.checked;
  els.flowSplatUnderpaintPercent.disabled =
    state.running || !flowSelected || !els.flowSplatUnderpainting.checked;
  els.flowSplatFusionMaxArcLabel.hidden = false;
  els.flowSplatFusionMaxArcPercent.disabled = state.running || !flowSelected;
  els.rectanglePaintPanel.hidden = !rectangleAlgorithmSelected && !sharedFlow;
  // Its author-level display:grid can override the browser's [hidden] rule.
  els.rectanglePaintPanel.classList.toggle("flow-hidden", false);
  els.rectanglePaintPanel.querySelector("summary").textContent = sharedFlow ? "Brush dab settings" : "Rectangle Splats settings";
  els.rectanglePaintShape.disabled = state.running || !rectangleAlgorithmSelected;
  els.rectanglePaintShape.closest("label")?.classList.toggle("flow-hidden", sharedFlow);
  els.rectangleShapeSettings.hidden = !rectangleSelected;
  els.opaquePaintPanel.hidden = !brushSelected;
  document.documentElement.dataset.rectanglePaintShape = rectangleAlgorithmSelected
    ? selectedRectanglePaintShape()
    : "";
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
  els.initialSplatCount.disabled = state.running || (flowSelected && !sharedFlow);
  els.flowTrainingPath.disabled = state.running || !flowSelected;
  for (const control of [els.flowSplatFusionStrokeTexture, els.flowSplatFusionStrokeOptimization,
    els.flowSplatFusionScaleMatchedResidualRepaint, els.flowSplatFusionPaintCurriculum,
    els.flowSplatFusionStartingWidthDivisor, els.flowSplatFusionStartingLengthPercent, els.flowSplatFusionResidualMovePx,
    els.flowSplatFusionInitialWidthMin, els.flowSplatFusionInitialWidthMax, els.flowSplatFusionFrontWidthMax,
    els.flowSplatFusionFrontWidthLearning, els.flowSplatFusionColorAnchor, els.flowSplatFusionWidthPercent,
    els.flowSplatFusionEdgeAccents, els.flowSplatFusionSplatSizeVariation,
    els.flowSplatFusionMovementLimit, els.flowSplatFusionMaxArcPercent, els.flowSplatBackcoatFromP1]) {
    control.closest("label")?.classList.toggle("flow-hidden", sharedFlow);
    if (sharedFlow) control.disabled = true;
  }
  els.layeredBrushLearnedOpacityMin.closest(".rectangle-edge-ratio-row")?.classList.toggle("flow-hidden", sharedFlow);
  els.opacityLearningRate.disabled = state.running || flowSelected;
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
  els.layeredBrushLearnedOpacityMin.disabled = state.running || !brushSelected || sharedFlow;
  els.layeredBrushLearnedOpacityMax.disabled = state.running || !brushSelected || sharedFlow;
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
  els.rectangleOrientationTolerance.disabled = state.running || !rectangleSelected || selectedRectangleOrientation() === "free";
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
    ? "Free keeps unrestricted rotation; Vertical or Horizontal constrains the long axis within Orientation tolerance."
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
  for (const control of [els.flowSplatUnderpainting, els.flowSplatUnderpaintPercent,
    els.flowSplatBackcoatSizeVariation, els.flowSplatFusionFixedOpacity]) {
    control.closest("label")?.classList.toggle("flow-hidden", internalBend);
    if (internalBend) control.disabled = true;
  }
  if (internalBend) {
    for (const control of [els.layeredBrushOpacityGradientStart, els.layeredBrushOpacityGradientEnd,
      els.layeredBrushCenterOpacityGradientMin, els.layeredBrushCenterOpacityGradientMax,
      els.layeredBrushWidthTaperStart, els.layeredBrushWidthTaperEnd, els.layeredBrushLocalColorFlow]) {
      control.disabled = true;
    }
    // Internal bend owns rawDepth as its bend parameter, so generic layer-order
    // Adam and relocation-based density correction are intentionally unavailable.
    // The remaining shared layer, visibility, optimizer and growth controls are
    // live and must stay visible.
    els.layerUpdateInterval.disabled = true;
    els.layerUpdateInterval.title = "Internal-bend Brush splats keep fixed layer ownership; raw depth learns bend instead.";
    els.growthPercentage.disabled = true;
    els.growthPercentage.title = "Internal-bend Brush splats use the shared interval, end percentage and P1/P2/P3 shares with deterministic target closure.";
    for (const control of [els.midTrainingOverdensityCorrection, els.overdensityCorrectionSchedule,
      els.overdensityCorrectionInterval, els.overdensityDonorPercent]) control.disabled = true;
    els.midTrainingOverdensityCorrection.closest("label").title =
      "Unavailable for Internal-bend Brush splats because relocation of owned bend metadata is not implemented.";
  } else {
    els.layerUpdateInterval.title = "Optimize layer-order micro-depth every N iterations. 1 updates every iteration.";
    els.growthPercentage.title = "Maximum growth requested per shared density event.";
    els.midTrainingOverdensityCorrection.closest("label").title =
      "Using the selected schedule, safely reuses low-utility splats from over-budget regions in under-budget high-residual regions.";
  }
  document.querySelector('[aria-labelledby="colorWorkflowHeading"]')?.classList.toggle("flow-hidden", internalBend);
  for (const heading of ["layerVisibilityHeading", "growthDetailHeading"]) {
    document.querySelector(`[aria-labelledby="${heading}"]`)?.classList.remove("flow-hidden");
  }
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
  if (selectedAlgorithm().capabilities.internalBend) enabled = false;
  for (const element of pausedRuntimeControls()) element.disabled = !enabled;
  if (algorithmUsesLayeredOpaqueBrush()) {
    els.alphaLossWeight.disabled = true;
  }
  document.documentElement.dataset.pausedRuntimeControls = String(enabled);
}
