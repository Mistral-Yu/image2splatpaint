function applyQaFlowStrokeInitialization(params, trainingImage, strokePlan, planWidth, planHeight) {
  if (!strokePlan?.length || !params?.count) return null;
  const width = Math.max(1, Number(planWidth) || trainingImage.width);
  const height = Math.max(1, Number(planHeight) || trainingImage.height);
  const extent = LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT;
  const maxAspect = Math.max(1, Number(params.brushMaxAspectRatio) || DEFAULT_MAX_ANISOTROPY);
  let meanPath = 0;
  let meanAspect = 0;
  const layerCounts = [0, 0, 0];
  for (let index = 0; index < params.count; index += 1) {
    const planIndex = Math.min(
      strokePlan.length - 1,
      Math.floor((index + 0.5) * strokePlan.length / params.count),
    );
    const stroke = strokePlan[planIndex];
    let tangentX = stroke.end_x - stroke.start_x;
    let tangentY = stroke.end_y - stroke.start_y;
    let tangentLength = Math.hypot(tangentX, tangentY);
    if (tangentLength < 1e-6) {
      const fallback = params.theta[index] || 0;
      tangentX = Math.cos(fallback) * width;
      tangentY = Math.sin(fallback) * height;
      tangentLength = Math.hypot(tangentX, tangentY);
    }
    const unitX = tangentX / Math.max(1e-6, tangentLength);
    const unitY = tangentY / Math.max(1e-6, tangentLength);
    const tangentNdcPerPixel = Math.hypot(2 * unitX / width, 2 * unitY / height);
    const normalNdcPerPixel = Math.hypot(-2 * unitY / width, 2 * unitX / height);
    const halfPathNdc = Math.max(MIN_SPLAT_SCALE, 0.5 * stroke.path_length_px * tangentNdcPerPixel);
    const halfWidthNdc = Math.max(MIN_SPLAT_SCALE, stroke.half_width_px * normalNdcPerPixel);
    let major = Math.max(
      MIN_SPLAT_SCALE,
      halfPathNdc / extent,
      halfWidthNdc * 1.6,
    );
    let minor = Math.max(MIN_SPLAT_SCALE, halfWidthNdc / (extent * 0.82));
    if (major / minor > maxAspect) minor = major / maxAspect;
    const theta = Math.atan2(2 * unitY / height, 2 * unitX / width);
    const x = -1 + 2 * stroke.center_x / Math.max(1, width - 1);
    const y = -1 + 2 * stroke.center_y / Math.max(1, height - 1);
    const constrained = constrainSplat(
      x,
      y,
      major,
      minor,
      theta,
      params.boundarySigma,
      maxAspect,
    );
    params.xy[index * 2] = constrained.x;
    params.xy[index * 2 + 1] = constrained.y;
    params.scale[index * 2] = constrained.sx;
    params.scale[index * 2 + 1] = constrained.sy;
    params.theta[index] = theta;
    params.detailTags[index] = 1;
    params.rgb[index * 3] = clampNumber(stroke.color_r, 0, 1, 0);
    params.rgb[index * 3 + 1] = clampNumber(stroke.color_g, 0, 1, 0);
    params.rgb[index * 3 + 2] = clampNumber(stroke.color_b, 0, 1, 0);
    const layer = Math.max(0, Math.min(2, Math.round(stroke.layer) || 0));
    layerCounts[layer] += 1;
    meanPath += stroke.path_length_px;
    meanAspect += Math.max(constrained.sx, constrained.sy) /
      Math.max(MIN_SPLAT_SCALE, Math.min(constrained.sx, constrained.sy));
  }
  params.initializationScheme = "qa-flow-stroke-plan";
  params.initializationStats = {
    candidate: "BR-CAND-58",
    source: "independent-three-level-poisson-flow-plan",
    plan_count: strokePlan.length,
    applied_count: params.count,
    plan_size: [width, height],
    layer_counts: layerCounts,
    mean_path_length_px: meanPath / params.count,
    mean_aspect_ratio: meanAspect / params.count,
    brush_primitive_and_optimizer_unchanged: true,
  };
  return params.initializationStats;
}

async function trainGaussianAlgorithm(virtualCameraSamplingEnabled, run = beginTrainingRun()) {
  assertTrainingRun(run);
  if (!state.image || state.running) return;
  const algorithm = selectedAlgorithm();
  const qaQuery = QA_RUNTIME_ENABLED ? new URLSearchParams(location.search) : null;
  const flowPaintReferenceMode = qaQuery?.get("brush-flow-reference") || "";
  const runFlowPaintReference = Boolean(
    algorithmUsesLayeredOpaqueBrush(algorithm) &&
    (flowPaintReferenceMode === "1" || flowPaintReferenceMode === "fine")
  );
  const runFineFlowPaintReference = runFlowPaintReference && flowPaintReferenceMode === "fine";
  const runFlowStrokeInitialization = runFineFlowPaintReference &&
    qaQuery?.get("brush-flow-init") === "1";
  const flowPaintSourceImage = runFineFlowPaintReference ? state.image : null;
  syncAlgorithmRequirements();
  if (Boolean(algorithm.capabilities.virtualCameras) !== Boolean(virtualCameraSamplingEnabled)) {
    throw new Error(`Training entry does not match selected algorithm: ${algorithm.id}`);
  }
  cancelCompletedResultGpuRecovery();
  destroyTiltViewer({ restoreCanvas: true });
  activateDetailTab("training");
  if (!state.webgpu.supported || !state.webgpu.renderer) {
    setStatus("error");
    trainingUiAdapter.controls.backendText.textContent = "webgpu required";
    setTrainingMessage(`Training unavailable: ${state.webgpu.reason}`, "error");
    log(`WebGPU required; training not started: ${state.webgpu.reason}`);
    return;
  }
  const renderer = run.renderer;
  const performanceSelection = renderer.configureExperimentalPerformance();
  trainingUiAdapter.setDataset("opacityAwareSupport", performanceSelection.opacityAwareSupportMode);
  const requestedSteps = normalizeStepInteger(trainingUiAdapter.controls.stepCount.value, {
    min: LIMITS.stepsMin,
    max: LIMITS.stepsMax,
    fallback: DEFAULT_ITERATIONS,
  });
  if (requestedSteps > HIGH_ITERATION_CONFIRM) {
    const ok = trainingUiAdapter.confirm(`Run ${requestedSteps.toLocaleString()} iterations? This may take a long time.`);
    if (!ok) {
      setStatus("idle");
      log(`training cancelled before start: ${requestedSteps} iterations`);
      return;
    }
  }
  const resizedForTraining = await resizeLoadedImageToMaxSide(Number(trainingUiAdapter.controls.trainSize.value));
  updateTrainingRunOwnership(run, { image: state.image });
  assertTrainingRun(run);
  if (resizedForTraining) {
    syncTrainSizeUi();
    log(`applied training image size ${state.image.width}x${state.image.height}`);
  }
  clearSafetyStop();
  const preflight = updateMemoryRecommendation();
  const preflightFailure = safetyFailure(preflight, "start");
  const wantsHighCapacityProbe =
    trainingUiAdapter.controls.capacityMode.value === "auto-probe" &&
    Number(preflight.effectiveFinalSplats || trainingUiAdapter.controls.finalSplatCount.value) > CAPACITY_PROBE_FAST_PATH_MAX;
  if (preflightFailure && (!wantsHighCapacityProbe || preflightFailure.reason === "safety_stop_hard_limit")) {
    setSafetyStop(preflightFailure);
    setStatus("safety stopped");
    setTrainingMessage(`Training blocked by the GPU safety guard: ${preflightFailure.reason}`, "error");
    updateDownloads(false);
    return;
  }
  if (preflightFailure) log(`Auto probe will search below the rejected ${trainingUiAdapter.controls.finalSplatCount.value} splat request.`);
  resetEvaluationStatusForNewTraining();
  state.running = true;
  state.flowSplatResult = null;
  state.previewGeneration += 1;
  state.paused = false;
  resetTrainingTiming(false);
  state.runtimeSettingsRevision = 0;
  state.layerTelemetryState = null;
  state.layerEfficiencyCheckpoints = [];
  state.stopRequested = false;
  clearSplatAdjustmentBaseline();
  updateDownloads(false);
  setStatus("running");
  trainingUiAdapter.controls.startButton.disabled = true;
  trainingUiAdapter.controls.stopButton.disabled = false;
  setInputControlsDisabled(true);
  setPausedRuntimeControlsEnabled(false);
  setTrainingMessage("Training on WebGPU...");

  const baseInitialCount = normalizeUiSplatCount(
    trainingUiAdapter.controls.initialSplatCount.value,
    DEFAULT_INITIAL_SPLATS,
    CAPACITY_PROBE_FAST_PATH_MAX,
  );
  const baseFinalCount = Math.max(
    baseInitialCount,
    normalizeUiSplatCount(trainingUiAdapter.controls.finalSplatCount.value, DEFAULT_FINAL_SPLATS),
  );
  const steps = requestedSteps;
  let learningRates = selectedLearningRates();
  const runSharedMaxAnisotropy = learningRates.maxAnisotropy;
  if (algorithmUsesPaintKernel(algorithm)) {
    learningRates = {
      ...learningRates,
      boundarySigma: Math.min(learningRates.boundarySigma, RECTANGLE_KERNEL_EXTENT),
    };
  }
  let previewRefresh = selectedPreviewRefresh();
  const runLiveQualityMetrics = liveQualityMetricsEnabled();
  const runQaPeriodicMetrics = qaPeriodicTrainingEvaluationEnabled();
  const runPeriodicEvaluation = periodicTrainingEvaluationEnabled();
  const runPhase33 = phase33Variants();
  const runGrowthSettings = phase39Variants();
  const runStageAwareGrowth = runGrowthSettings.stageAwareGrowth;
  const runStageGrowthShares = runGrowthSettings.stageGrowthShares;
  const runLayerSettings = phase46Variants();
  const runDiscreteLayerSettings = discreteLayerSettings();
  const requestedGrowthApplyUntilFraction = runGrowthSettings.growthApplyUntilFraction;
  const runGrowthApplyUntilFraction = effectiveGrowthApplyUntilFraction(
    steps,
    requestedGrowthApplyUntilFraction,
    runDiscreteLayerSettings.opaqueLayered,
    runDiscreteLayerSettings.opaquePaintSettleFraction,
  );
  const runCurrentContributionCompaction = currentContributionCompactionSettings(algorithm);
  let initialCount = baseInitialCount;
  let finalCount = baseFinalCount;
  const runRectangleLearnedOpacity = selectedRectangleLearnedOpacity();
  const runBrushLearnedOpacity = selectedLayeredBrushLearnedOpacity();
  const runRectangleOpacityGradient = selectedRectangleOpacityGradient();
  const runRectangleCenterOpacityGradient = selectedRectangleCenterOpacityGradient();
  const runBrushDirectionalEffects = selectedLayeredBrushDirectionalEffects();
  const runBrushCenterOpacityGradient = selectedLayeredBrushCenterOpacityGradient();
  const runBrushMinAspectRatio = selectedBrushMinAspectRatio();
  const runBrushMaxAspectRatio = selectedBrushMaxAspectRatio();
  const runBrushAspectFloors = selectedBrushAspectFloors();
  const flowPaintStrength = Math.max(
    0,
    Math.min(1, Number(qaQuery?.get("brush-flow-strength") ?? 1) || 0),
  );
  let flowPaintResult = null;
  if (runFlowPaintReference) {
    if (runFineFlowPaintReference) {
      const trainingSide = Math.max(state.image.width, state.image.height);
      const sourceSide = Math.max(flowPaintSourceImage.width, flowPaintSourceImage.height);
      const referenceSide = Math.min(sourceSide, 1024, Math.max(512, trainingSide * 2));
      const [referenceWidth, referenceHeight] = resizedSize(
        flowPaintSourceImage.width,
        flowPaintSourceImage.height,
        referenceSide,
      );
      const referenceSource = referenceWidth === flowPaintSourceImage.width &&
        referenceHeight === flowPaintSourceImage.height
        ? flowPaintSourceImage
        : {
            ...flowPaintSourceImage,
            ...resizeFloatImageBilinear(flowPaintSourceImage, referenceWidth, referenceHeight),
          };
      const generated = Image2SplatPaintFlowPaintReference.createFlowPaintReference(referenceSource, {
        seed: 240825,
        strength: flowPaintStrength,
        profile: "fine-layered-v2",
        maxStrokes: 14000,
        includeStrokePlan: runFlowStrokeInitialization,
      });
      const resizedReference = resizeFloatImageBilinear(
        generated.image,
        state.image.width,
        state.image.height,
      );
      flowPaintResult = {
        image: { ...state.image, ...resizedReference },
        strokePlan: generated.strokePlan,
        metadata: {
          ...generated.metadata,
          generated_size: [referenceWidth, referenceHeight],
          training_size: [state.image.width, state.image.height],
          generated_before_training_resize: true,
        },
      };
    } else {
      flowPaintResult = Image2SplatPaintFlowPaintReference.createFlowPaintReference(state.image, {
        seed: 240825,
        strength: flowPaintStrength,
      });
    }
  }
  const trainingImage = flowPaintResult?.image || state.image;
  if (flowPaintResult) {
    const flow = flowPaintResult.metadata;
    log(
      `QA flow-painted Brush reference: ${flow.accepted_strokes}/${flow.requested_strokes} strokes, ` +
      `mean ${flow.mean_path_length_px.toFixed(2)}px, long ${(flow.long_stroke_fraction * 100).toFixed(1)}%`,
    );
  }
  const runSharedColorWorkflow = selectedSharedColorWorkflow();
  const runSurfaceLayerPrior = scaleBiasedSurfaceLayerPriorSettings();
  const runHarmfulRectangleParentSplit = harmfulRectangleParentSplitSettings();
  const runFrontSplitChildren = frontSplitChildrenSettings();
  const runLearnedOpacity = algorithmUsesLayeredOpaqueBrush(algorithm)
    ? runBrushLearnedOpacity
    : runRectangleLearnedOpacity;
  const runRectangleTopRatio = selectedRectangleTopRatio();
  const runRectangleTopRatioMax =
    selectedRectangleTopRatioMax(runRectangleTopRatio);
  const runRectangleMinAspectRatio = selectedRectangleMinAspectRatio();
  const runRectangleMaxAspectRatio =
    selectedRectangleMaxAspectRatio(runRectangleMinAspectRatio);
  const runRectangleOrientation = selectedRectangleOrientation();
  const runRectangleOrientationToleranceDegrees = selectedRectangleOrientationToleranceDegrees();
  const runRectangleOrientationTolerance = runRectangleOrientationToleranceDegrees / 90 * 100;
  if (algorithmUsesRectangleKernel(algorithm)) {
    learningRates = {
      ...learningRates,
      maxAnisotropy: runRectangleMaxAspectRatio,
    };
  }
  if (algorithmUsesLayeredOpaqueBrush(algorithm)) {
    learningRates = {
      ...learningRates,
      maxAnisotropy: runBrushMaxAspectRatio,
    };
  }
  const runRectangleShape = selectedRectangleShapeSettings();
  const runVirtualCameraSampling = virtualCameraSamplingVariants(virtualCameraSamplingEnabled);
  const adaptiveGridRequest = adaptiveGridInitializationVariants(virtualCameraSamplingEnabled);
  const runAdaptiveGridInitialization = algorithmUsesPaintKernel(algorithm)
    ? {
        ...adaptiveGridRequest,
        requested: false,
        enabled: false,
        reason: "image-importance-bsp-initialization",
      }
    : adaptiveGridRequest;
  const runVirtualCameraOrbitRadius = runVirtualCameraSampling.enabled && runVirtualCameraSampling.autoCameraDistance
    ? sharedTiltOrbitRadius(
      state.image.width,
      state.image.height,
      runVirtualCameraSampling.maxAngleDegrees,
      runVirtualCameraSampling.cameraCount,
      runVirtualCameraSampling.fovDegrees,
    )
    : runVirtualCameraSampling.cameraDistance;
  const runVirtualCameraCatalog = runVirtualCameraSampling.enabled
    ? virtualCameraCatalog(
      state.image.width,
      state.image.height,
      runVirtualCameraOrbitRadius,
      runVirtualCameraSampling,
    )
    : [];
  const runVirtualTeacherCoverage = runVirtualCameraSampling.enabled
    ? virtualCameraCoverageStats(
      state.image.width,
      state.image.height,
      runVirtualCameraSampling,
    )
    : null;
  const gpuDensifyEnabled = true;
  const gpuRelocationEnabled = true;
  const useAutoCapacityProbe = trainingUiAdapter.controls.capacityMode.value === "auto-probe" && finalCount > CAPACITY_PROBE_FAST_PATH_MAX;
  state.capacityProbe = {
    status: useAutoCapacityProbe ? "ready" : finalCount <= CAPACITY_PROBE_FAST_PATH_MAX ? "fast" : "manual",
    requested: finalCount,
    selected: useAutoCapacityProbe ? 0 : finalCount,
    attempts: [],
    fastPath: finalCount <= CAPACITY_PROBE_FAST_PATH_MAX,
  };
  trainingUiAdapter.controls.initialSplatCount.value = String(baseInitialCount);
  trainingUiAdapter.controls.finalSplatCount.value = String(baseFinalCount);
  trainingUiAdapter.controls.stageGrowthP1.value = (runStageGrowthShares.p1 * 100).toFixed(2);
  trainingUiAdapter.controls.stageGrowthP2.value = (runStageGrowthShares.p2 * 100).toFixed(2);
  trainingUiAdapter.controls.stageGrowthP3.value = (runStageGrowthShares.p3 * 100).toFixed(2);
  trainingUiAdapter.controls.stepCount.value = String(steps);
  trainingUiAdapter.controls.previewRefresh.value = previewRefresh;
  trainingUiAdapter.controls.layerUpdateInterval.value = String(runLayerSettings.layerUpdateInterval);
  trainingUiAdapter.controls.rectangleOpacityGradientMin.value = String(runRectangleOpacityGradient.min);
  trainingUiAdapter.controls.rectangleOpacityGradientMax.value = String(runRectangleOpacityGradient.max);
  trainingUiAdapter.controls.rectangleCenterOpacityGradientMin.value = String(runRectangleCenterOpacityGradient.min);
  trainingUiAdapter.controls.rectangleCenterOpacityGradientMax.value = String(runRectangleCenterOpacityGradient.max);
  trainingUiAdapter.controls.rectangleLearnedOpacityMin.value = String(runRectangleLearnedOpacity.min);
  trainingUiAdapter.controls.rectangleLearnedOpacityMax.value = String(runRectangleLearnedOpacity.max);
  trainingUiAdapter.controls.layeredBrushOpacityGradientStart.value = String(runBrushDirectionalEffects.opacityStart);
  trainingUiAdapter.controls.layeredBrushOpacityGradientEnd.value = String(runBrushDirectionalEffects.opacityEnd);
  trainingUiAdapter.controls.layeredBrushCenterOpacityGradientMin.value = String(runBrushCenterOpacityGradient.min);
  trainingUiAdapter.controls.layeredBrushCenterOpacityGradientMax.value = String(runBrushCenterOpacityGradient.max);
  trainingUiAdapter.controls.layeredBrushLearnedOpacityMin.value = String(runBrushLearnedOpacity.min);
  trainingUiAdapter.controls.layeredBrushLearnedOpacityMax.value = String(runBrushLearnedOpacity.max);
  trainingUiAdapter.controls.layeredBrushWidthTaperStart.value = String(runBrushDirectionalEffects.widthStart);
  trainingUiAdapter.controls.layeredBrushWidthTaperEnd.value = String(runBrushDirectionalEffects.widthEnd);
  trainingUiAdapter.controls.layeredBrushMinAspectRatio.value = String(runBrushMinAspectRatio);
  trainingUiAdapter.controls.layeredBrushMaxAspectRatio.value = String(runBrushMaxAspectRatio);
  trainingUiAdapter.controls.colorFinishStart.value = String(runSharedColorWorkflow.colorFinishStartPercent);
  trainingUiAdapter.controls.rectangleTopRatio.value = String(runRectangleTopRatio);
  trainingUiAdapter.controls.rectangleTopRatioMax.value = String(runRectangleTopRatioMax);
  trainingUiAdapter.controls.rectangleMinAspectRatio.value = String(runRectangleMinAspectRatio);
  trainingUiAdapter.controls.rectangleMaxAspectRatio.value = String(runRectangleMaxAspectRatio);
  trainingUiAdapter.controls.rectangleOrientation.value = runRectangleOrientation;
  trainingUiAdapter.controls.rectangleOrientationTolerance.value = String(runRectangleOrientationToleranceDegrees);
  trainingUiAdapter.controls.rectanglePreserveArea.checked = runRectangleShape.preserveArea;
  trainingUiAdapter.controls.rectangleEdgeDirectedTaper.checked = runRectangleShape.edgeDirectedTaper;
  trainingUiAdapter.controls.rectangleStructureAwareRatio.checked = runRectangleShape.structureAwareRatio;
  trainingUiAdapter.controls.rectangleAsymmetricSoftness.checked = runRectangleShape.asymmetricSoftness;
  trainingUiAdapter.controls.scaleBiasedSurfaceLayerPrior.checked = runSurfaceLayerPrior.enabled;
  trainingUiAdapter.controls.scaleBiasedSurfaceLayerPriorLayers.value = String(runSurfaceLayerPrior.layers);
  trainingUiAdapter.controls.scaleBiasedSurfaceLayerPriorP1Interval.value = String(runSurfaceLayerPrior.p1Interval);
  trainingUiAdapter.controls.scaleBiasedSurfaceLayerPriorP2Interval.value = String(runSurfaceLayerPrior.p2Interval);
  trainingUiAdapter.controls.scaleBiasedSurfaceLayerPriorP3Interval.value = String(runSurfaceLayerPrior.p3Interval);
  trainingUiAdapter.controls.scaleBiasedSurfaceLayerPriorUntil.value = String(runSurfaceLayerPrior.untilFraction * 100);
  trainingUiAdapter.controls.positionLearningRate.value = String(learningRates.position);
  trainingUiAdapter.controls.colorLearningRate.value = String(learningRates.color);
  trainingUiAdapter.controls.opacityLearningRate.value = String(learningRates.opacity);
  trainingUiAdapter.controls.scaleLearningRate.value = String(learningRates.scaleParam);
  trainingUiAdapter.controls.rotationLearningRate.value = String(learningRates.rotation);
  trainingUiAdapter.controls.thetaAlignRate.value = String(learningRates.thetaAlign);
  trainingUiAdapter.controls.maxAnisotropy.value = String(runSharedMaxAnisotropy);
  trainingUiAdapter.controls.maxPlanarScale.value = String(learningRates.maxPlanarScale);
  trainingUiAdapter.controls.boundarySigma.value = String(learningRates.boundarySigma);
  trainingUiAdapter.controls.detailCoherence.value = String(learningRates.detailCoherence);
  const budget = updateMemoryRecommendation();
  if (budget.overBudget) {
    log(`settings exceed safety budget ${budget.estimatedMB}MB > ${budget.budgetMB}MB; recommended ${budget.recommendedTrainSize}px ${budget.recommendedFinalSplats} splats`);
  }
  const initialization = runSharedColorWorkflow.monochromeUnderpainting
    ? algorithmUsesPaintKernel(algorithm)
      ? "image-lab-l-monochrome-importance-bsp"
      : "image-lab-l-monochrome-grid"
    : algorithmUsesPaintKernel(algorithm)
      ? "image-rgb-importance-bsp"
    : runAdaptiveGridInitialization.requested
      ? "image-rgb-grid-adaptive-placement"
      : "image-rgb-grid";
  state.params = algorithm.initialize(trainingImage, baseInitialCount);
  if (runFlowStrokeInitialization) {
    const flowInitialization = applyQaFlowStrokeInitialization(
      state.params,
      trainingImage,
      flowPaintResult?.strokePlan,
      flowPaintResult?.metadata?.generated_size?.[0],
      flowPaintResult?.metadata?.generated_size?.[1],
    );
    if (flowInitialization) {
      log(
        `QA flow-stroke initialization: ${flowInitialization.applied_count}/` +
        `${flowInitialization.plan_count}, mean aspect ${flowInitialization.mean_aspect_ratio.toFixed(2)}`,
      );
    }
  }
  // Training controls are disabled for the lifetime of a run. Snapshot tile
  // culling with the other run parameters so GPU optimizer code never reads
  // mutable DOM state while command buffers are being encoded.
  state.params.tileCullingEnabled = Boolean(trainingUiAdapter.controls.tileCullingToggle.checked);
  state.params.brushRibbonAspectFloor = runBrushAspectFloors.ribbon;
  state.params.brushAccentAspectFloor = runBrushAspectFloors.accent;
  state.params.brushMinAspectRatio = runBrushMinAspectRatio;
  state.params.brushMaxAspectRatio = runBrushMaxAspectRatio;
  state.params.monochromeUnderpaintingEnabled = runSharedColorWorkflow.monochromeUnderpainting;
  state.params.colorFinishStartPercent = runSharedColorWorkflow.colorFinishStartPercent;
  state.params.colorFinishStartStep = colorFinishStartStep(
    steps,
    runSharedColorWorkflow.colorFinishStartPercent,
  );
  if (state.params.monochromeUnderpaintingEnabled) convertRgbToNeutralLabL(state.params.rgb);
  initialCount = state.params.count;
  finalCount = Math.max(initialCount, finalCount);
  state.params.layerAwareAccumulationEnabled = runDiscreteLayerSettings.accumulationEnabled;
  state.params.currentVisibilityChildPolicyEnabled =
    runDiscreteLayerSettings.currentVisibilityChildPolicyEnabled;
  state.params.currentVisibilityCompactionEnabled =
    runDiscreteLayerSettings.currentVisibilityCompactionEnabled;
  state.params.currentContributionCompactionEnabled =
    runCurrentContributionCompaction.enabled;
  state.params.discreteLayersEnabled = runDiscreteLayerSettings.enabled;
  state.params.discreteLayerCount = runDiscreteLayerSettings.count;
  state.params.discreteLayerMoveRadius = runDiscreteLayerSettings.moveRadius;
  state.params.surfaceLayerPriorEnabled = runSurfaceLayerPrior.enabled;
  state.params.surfaceLayerPriorColorAwarePromotion = runSurfaceLayerPrior.colorAwarePromotion;
  state.params.surfaceLayerPriorLayers = runSurfaceLayerPrior.layers;
  state.params.surfaceLayerPriorP1Interval = runSurfaceLayerPrior.p1Interval;
  state.params.surfaceLayerPriorP2Interval = runSurfaceLayerPrior.p2Interval;
  state.params.surfaceLayerPriorP3Interval = runSurfaceLayerPrior.p3Interval;
  state.params.surfaceLayerPriorUntilFraction = runSurfaceLayerPrior.untilFraction;
  if (state.params.surfaceLayerPriorEnabled) state.params.layerOrderEnabled = true;
  state.params.harmfulRectangleParentSplitEnabled =
    runHarmfulRectangleParentSplit.enabled || (
      algorithmUsesPaintKernel(algorithm) && state.params.monochromeUnderpaintingEnabled
    );
  state.params.harmfulRectangleParentSplitTransitionOnly =
    state.params.harmfulRectangleParentSplitEnabled &&
    !runHarmfulRectangleParentSplit.enabled &&
    state.params.monochromeUnderpaintingEnabled;
  state.params.frontSplitChildrenEnabled = runFrontSplitChildren.enabled;
  if (algorithmUsesOpaqueLayeredPaint(algorithm)) {
    state.params.opaqueLayered = true;
    state.params.minimumOpacityEnabled = true;
    state.params.rectangleTopRatio = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleTopRatio
      : DEFAULT_RECTANGLE_TOP_RATIO;
    state.params.rectangleTopRatioMax = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleTopRatioMax
      : DEFAULT_RECTANGLE_TOP_RATIO_MAX;
    state.params.rectangleOpacityGradientMin = runRectangleOpacityGradient.min;
    state.params.rectangleOpacityGradientMax = runRectangleOpacityGradient.max;
    state.params.rectangleCenterOpacityGradientMin = runRectangleCenterOpacityGradient.min;
    state.params.rectangleCenterOpacityGradientMax = runRectangleCenterOpacityGradient.max;
    state.params.rectangleMinAspectRatio = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleMinAspectRatio
      : DEFAULT_RECTANGLE_MIN_ASPECT_RATIO;
    state.params.rectangleMaxAspectRatio = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleMaxAspectRatio
      : DEFAULT_RECTANGLE_ASPECT_RATIO;
    state.params.rectangleOrientation = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleOrientation
      : DEFAULT_RECTANGLE_ORIENTATION;
    state.params.rectangleOrientationTolerance = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleOrientationTolerance
      : 0;
    state.params.rectanglePreserveArea = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleShape.preserveArea
      : DEFAULT_RECTANGLE_PRESERVE_AREA;
    state.params.rectangleEdgeDirectedTaper = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleShape.edgeDirectedTaper
      : DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER;
    state.params.rectangleStructureAwareRatio = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleShape.structureAwareRatio
      : DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO;
    state.params.rectangleAsymmetricSoftness = algorithmUsesRectangleKernel(algorithm)
      ? runRectangleShape.asymmetricSoftness
      : DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS;
    state.params.minimumOpacity = runLearnedOpacity.min;
    state.params.maximumOpacity = runLearnedOpacity.max;
    state.params.opacity.fill(runLearnedOpacity.max);
    const brushSelected = algorithmUsesLayeredOpaqueBrush(algorithm);
    state.params.brushOpacityGradientEnabled =
      brushSelected && runBrushDirectionalEffects.opacity;
    state.params.brushOpacityGradientStart = runBrushDirectionalEffects.opacityStart;
    state.params.brushOpacityGradientEnd = runBrushDirectionalEffects.opacityEnd;
    state.params.brushCenterOpacityGradientMin = runBrushCenterOpacityGradient.min;
    state.params.brushCenterOpacityGradientMax = runBrushCenterOpacityGradient.max;
    state.params.brushWidthTaperEnabled =
      brushSelected && runBrushDirectionalEffects.widthTaper;
    state.params.brushWidthTaperStart = runBrushDirectionalEffects.widthStart;
    state.params.brushWidthTaperEnd = runBrushDirectionalEffects.widthEnd;
    if (!state.params.brushTaper || state.params.brushTaper.length !== state.params.count) {
      state.params.brushTaper = new Float32Array(state.params.count).fill(DEFAULT_LAYERED_BRUSH_TAPER);
    }
    state.params.layerOrderEnabled = true;
    state.params.layerAwareAccumulationEnabled = true;
    state.params.discreteLayersEnabled = true;
    state.params.discreteLayerMoveRadius = LAYERED_OPAQUE_BRUSH_LAYER_MOVE_RADIUS;
  }
  state.params.virtualDepthEnabled = Boolean(runVirtualCameraSampling.enabled && runVirtualCameraSampling.boundedDepth);
  state.params.virtualCameraSamplingEnabled = Boolean(runVirtualCameraSampling.enabled);
  state.params.virtualDepthThickness = runVirtualCameraSampling.depthThickness;
  state.params.virtualDepthCenterWeight = runVirtualCameraSampling.depthCenterWeight;
  state.params.virtualDepthSmoothnessWeight = runVirtualCameraSampling.depthSmoothnessWeight;
  state.params.virtualDepthLearningRate = runVirtualCameraSampling.depthLearningRate;
  state.params.virtualDepthSoftConstraintEnabled = runVirtualCameraSampling.softDepthConstraint;
  state.params.virtualDepthPriorDelta = runVirtualCameraSampling.depthPriorDelta;
  if (!state.params.virtualDepth || state.params.virtualDepth.length !== state.params.count) {
    state.params.virtualDepth = new Float32Array(state.params.count);
  }
  state.virtualCameraByStep = new Map();
  if (!trainingUiAdapter.controls.trainLayerOrder.checked) state.params.depthOrder.fill(0);
  state.previewMode = previewRefresh === "final" ? "original" : "splats";
  fitCanvases(state.image.width, state.image.height);
  if (previewRefresh === "final") {
    drawOriginalToCanvas();
    showCanvas("preview");
  }
  updatePreviewModeControls();
  state.metrics = {
    format: PRODUCT_FORMAT,
    version: 1,
    input_name: state.image.fileName,
    image_size: [state.image.width, state.image.height],
    input_original_size: [state.image.originalWidth, state.image.originalHeight],
    resize_mode: state.image.resizeMode,
    resize_scale: state.image.resizeScale,
    train_size: Math.max(state.image.width, state.image.height),
    algorithm: algorithm.id,
    algorithm_label: algorithm.label,
    algorithm_backend: algorithm.backend,
    algorithm_capabilities: { ...algorithm.capabilities },
    initialization: runFlowStrokeInitialization ? "qa-flow-stroke-plan" : initialization,
    initialization_adaptive: {
      requested: runAdaptiveGridInitialization.requested,
      applied: false,
      reason: runAdaptiveGridInitialization.reason,
      fraction: runAdaptiveGridInitialization.fraction,
      candidate_count: runAdaptiveGridInitialization.candidateCount,
      backend: "webgpu-compute",
    },
    initialization_bsp: state.params.initializationStats || null,
    brush_flow_paint_reference: flowPaintResult?.metadata || {
      enabled: false,
      candidate: runFineFlowPaintReference ? "BR-CAND-57" : "BR-CAND-56",
      qa_only: true,
    },
    brush_flow_stroke_initialization: runFlowStrokeInitialization
      ? state.params.initializationStats
      : {
          enabled: false,
          candidate: "BR-CAND-58",
          qa_only: true,
        },
    brush_detail_layer_policy: algorithmUsesLayeredOpaqueBrush(algorithm)
      ? {
          split_birth: "inherit-parent-layer",
          immediate_one_layer_promotion: false,
          subsequent_motion: "shared-contribution-aware-layer-training",
        }
      : null,
    train_layer_color_guard: algorithmUsesPaintKernel(algorithm)
      ? {
          enabled: Boolean(state.params.trainLayerColorGuardEnabled),
          default_enabled: true,
          mode: "repair-footprint-rgb-before-forward-order-change",
          high_variation_action: "defer-only",
        }
      : null,
    brush_local_color_flow: algorithmUsesLayeredOpaqueBrush(algorithm)
      ? {
          enabled: Boolean(state.params.brushLocalColorFlowEnabled),
          candidate: "BR-CAND-49",
          mode: "same-layer nearby-color axial-orientation regularizer",
          strength: state.params.brushLocalColorFlowEnabled
            ? BRUSH_LOCAL_COLOR_FLOW_STRENGTH
            : 0,
          constraints: {
            directional_only: true,
            minimum_neighbors: 2,
            maximum_color_distance_signal_srgb: 0.10,
            maximum_radius_px: 18,
            minimum_axial_consensus: 0.65,
          },
        }
      : null,
    brush_stroke_persistence: algorithmUsesLayeredOpaqueBrush(algorithm)
      ? {
          enabled: Boolean(state.params.brushStrokePersistenceEnabled),
          candidate: "BR-CAND-50",
          mode: "area-preserving-directional-minimum-anisotropy",
          ribbon_minimum_anisotropy: state.params.brushRibbonAspectFloor,
          accent_minimum_anisotropy: state.params.brushAccentAspectFloor,
          proximal_rate: BRUSH_STROKE_PERSISTENCE_PROXIMAL_RATE,
          parent_replacement_split_unchanged: true,
          additive_growth_unchanged: true,
          base_patch_unchanged: true,
        }
      : null,
    surface_layer_prior: {
      enabled: Boolean(state.params.surfaceLayerPriorEnabled),
      color_aware_promotion: state.params.surfaceLayerPriorColorAwarePromotion !== false,
      color_correction_strength: algorithmUsesPaintKernel(algorithm) ? 0.75 : 0.20,
      unsafe_high_variation_promotion: "deferred",
      mode: "verified-contribution-footprint-area-layer-sort",
      layers: state.params.surfaceLayerPriorLayers,
      p1_interval: state.params.surfaceLayerPriorP1Interval,
      p2_interval: state.params.surfaceLayerPriorP2Interval,
      p3_interval: state.params.surfaceLayerPriorP3Interval,
      until_fraction: state.params.surfaceLayerPriorUntilFraction,
      until_step: Math.floor(steps * state.params.surfaceLayerPriorUntilFraction),
      backend: "webgpu-optimizer",
      contract: "current-contributors-only; smaller-footprint-front; larger-footprint-rear; hidden-layer-preserved; stable-within-layer",
      event_count: 0,
      phase_counts: { P1: 0, P2: 0, P3: 0 },
      events: [],
    },
    front_footprint_refinement_v2: state.params.harmfulRectangleParentSplitEnabled
      ? {
          enabled: Boolean(state.params.harmfulRectangleParentSplitEnabled),
          transition_only: Boolean(
            state.params.harmfulRectangleParentSplitTransitionOnly,
          ),
          transition_start_step: state.params.harmfulRectangleParentSplitTransitionOnly
            ? state.params.colorFinishStartStep
            : null,
          transition_end_step: state.params.harmfulRectangleParentSplitTransitionOnly
            ? Math.min(
                steps,
                state.params.colorFinishStartStep + Math.max(200, Math.floor(steps * 0.1)),
              )
            : null,
          scope: state.params.harmfulRectangleParentSplitTransitionOnly
            ? "monochrome-to-color growth events; parent-replacement; same-layer children"
            : algorithmUsesPaintKernel(algorithm)
              ? "paint growth-events; mismatch-gated split/shrink/move; same-layer children"
              : algorithmUsesVirtualCameras(algorithm)
                ? "virtual growth-events; mismatch-gated symmetric split; depth preserved"
                : "planar growth-events; mismatch-gated standard-alpha symmetric split",
          candidate_selections: 0,
          front_oversized_selections: 0,
          high_contribution_selections: 0,
          high_deviation_selections: 0,
          parent_replacements: 0,
          children_created: 0,
        }
      : null,
    front_split_children: {
      enabled: Boolean(state.params.frontSplitChildrenEnabled),
      mode: "mode-1-growth-and-true-relocation-child-to-absolute-front",
      backend: "webgpu-density",
      default: false,
    },
    color_objective: runSharedColorWorkflow.monochromeUnderpainting
      ? {
          domain: "cielab-l-underpainting-then-signal-srgb",
          loss: "percentage-scheduled-l-star-then-rgb-l1-dssim-frequency",
          components: ["L*", "RGB"],
          schedule: {
            initialization: "neutral-srgb-preserving-cielab-l-star",
            monochrome_until_percent:
              runSharedColorWorkflow.colorFinishStartPercent,
            color_finish_start_step: state.params.colorFinishStartStep,
            before_color_finish: "L*-only; R=G=B projection",
            from_color_finish: "signal-sRGB L1/DSSIM/frequency",
            phase_independent: true,
          },
          rgb_storage_and_final_metrics_unchanged: true,
        }
      : {
          domain: "signal-srgb",
          loss: "rgb-l1-dssim-frequency",
          rgb_render_and_metrics_unchanged: true,
        },
    initial_splats: initialCount,
    final_splats: finalCount,
    num_gaussians: initialCount,
    default_output: algorithm.exports.includes("ply") ? "ply" : "png",
    steps_requested: steps,
    steps_done: 0,
    preview_refresh: previewRefresh,
    preview_frames: 0,
    preview_resolution_restores: 0,
    last_preview_step: null,
    params_revision: 0,
    coverage_revision: null,
    post_train_adjustments: null,
    runtime_settings_revision: 0,
    lr_scale: learningRates.scale,
    learning_rates: {
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
      trainLayerOrder: Boolean(trainingUiAdapter.controls.trainLayerOrder.checked),
      layerUpdateInterval: runLayerSettings.layerUpdateInterval,
      layerAwareAccumulation: runDiscreteLayerSettings.accumulationEnabled,
      currentContributionCompaction: runCurrentContributionCompaction.enabled,
      currentContributionCompactionStartPercent:
        runCurrentContributionCompaction.startFraction * 100,
      currentContributionCompactionInterval:
        runCurrentContributionCompaction.intervalSteps,
      currentContributionCompactionMaxRemovalFraction:
        runCurrentContributionCompaction.maxRemovalFraction,
      currentContributionCompactionNearZeroMaxFraction:
        runCurrentContributionCompaction.nearZeroMaxFraction,
      currentContributionCompactionMeasurementWindow:
        runCurrentContributionCompaction.measurementWindowSteps,
      illustrativeOil: algorithmUsesLayeredOpaqueBrush(algorithm),
      illustrativeOilVersion: state.params.illustrativeOilVersion || 0,
      illustrativeOilFamilyStats: state.params.illustrativeOilFamilyStats || null,
      learnedOpacityRange: algorithmUsesOpaqueLayeredPaint(algorithm)
        ? [runLearnedOpacity.min, runLearnedOpacity.max]
        : null,
      rectangleLearnedOpacityRange:
        algorithmUsesRectangleKernel(algorithm) ? [runRectangleLearnedOpacity.min, runRectangleLearnedOpacity.max] : null,
      rectangleDirectionalOpacityGradientRange:
        algorithmUsesRectangleKernel(algorithm)
          ? [runRectangleOpacityGradient.min, runRectangleOpacityGradient.max]
          : null,
      rectangleCenterOpacityGradientRange:
        algorithmUsesRectangleKernel(algorithm)
          ? [runRectangleCenterOpacityGradient.min, runRectangleCenterOpacityGradient.max]
          : null,
      layeredBrushLearnedOpacityRange:
        algorithmUsesLayeredOpaqueBrush(algorithm) ? [runBrushLearnedOpacity.min, runBrushLearnedOpacity.max] : null,
      layeredBrushMinimumAspectRatio:
        algorithmUsesLayeredOpaqueBrush(algorithm) ? runBrushMinAspectRatio : null,
      layeredBrushMaximumAspectRatio:
        algorithmUsesLayeredOpaqueBrush(algorithm) ? runBrushMaxAspectRatio : null,
      layeredBrushOpacityGradient:
        algorithmUsesLayeredOpaqueBrush(algorithm) ? runBrushDirectionalEffects.opacity : null,
      layeredBrushOpacityGradientRange:
        algorithmUsesLayeredOpaqueBrush(algorithm)
          ? [runBrushDirectionalEffects.opacityStart, runBrushDirectionalEffects.opacityEnd]
          : null,
      layeredBrushCenterOpacityGradientRange:
        algorithmUsesLayeredOpaqueBrush(algorithm)
          ? [runBrushCenterOpacityGradient.min, runBrushCenterOpacityGradient.max]
          : null,
      layeredBrushWidthTaper:
        algorithmUsesLayeredOpaqueBrush(algorithm) ? runBrushDirectionalEffects.widthTaper : null,
      layeredBrushWidthTaperRange:
        algorithmUsesLayeredOpaqueBrush(algorithm)
          ? [runBrushDirectionalEffects.widthStart, runBrushDirectionalEffects.widthEnd]
          : null,
      monochromeUnderpainting: runSharedColorWorkflow.monochromeUnderpainting,
      colorFinishStartPercent: runSharedColorWorkflow.colorFinishStartPercent,
      discreteLayers: runDiscreteLayerSettings.enabled,
      discreteLayerCount: runDiscreteLayerSettings.count,
      discreteLayerMoveRadius: runDiscreteLayerSettings.moveRadius,
    },
    live_quality_metrics: runLiveQualityMetrics,
    initial_param_snapshot: runQaPeriodicMetrics
      ? snapshotParams(state.params)
      : null,
    initial_orientation: initialOrientationStats(state.params, state.image),
    param_delta: null,
    train_sync_interval: trainSyncInterval(),
    train_sync_policy: {
      mode: "adaptive-high-splat-backpressure",
      high_splat_threshold: HIGH_SPLAT_SYNC_THRESHOLD,
      very_high_splat_threshold: VERY_HIGH_SPLAT_SYNC_THRESHOLD,
      effective_interval: trainSyncInterval(),
      readback_added: false,
    },
    gpu_scheduling: {
      adaptive_throughput: performanceSelection.adaptiveGpuThroughput,
      adaptive_gpu_batch: performanceSelection.adaptiveGpuBatch,
      mode: performanceSelection.gpuSchedulingMode,
      async_presentation: performanceSelection.asyncPresentation,
      metric_tile_reuse: performanceSelection.metricTileReuse,
      segmented_exact_backward: performanceSelection.segmentedExactBackward,
      fixed_point_exact_gradient: performanceSelection.fixedPointExactGradient,
      inverse_scale_optimization: performanceSelection.inverseScaleOptimization,
    },
    backend: selectedBackend(),
    webgpu_supported: state.webgpu.supported,
    webgpu_reason: state.webgpu.reason,
    gpu_budget: {
      exact_vram: false,
      budget_mb: budget.budgetMB,
      estimated_mb: budget.estimatedMB,
      headroom_mb: budget.headroomMB,
      memory_hint_source: budget.memoryHintSource,
      memory_hint_mb: budget.memoryHintMB,
      limiter_train_size: budget.limiterTrainSize,
      limiter_final_splats: budget.limiterFinalSplats,
      recommended_train_size: budget.recommendedTrainSize,
      recommended_final_splats: budget.recommendedFinalSplats,
      metric_interval: budget.metricInterval,
    },
    safety_stop: null,
    webgpu_compute_loss: false,
    webgpu_train_executed: false,
    webgpu_train_update: false,
    webgpu_densify_requested: gpuDensifyEnabled,
    webgpu_relocation_requested: gpuRelocationEnabled,
    webgpu_relocation: false,
    webgpu_refine_requested: gpuRelocationEnabled,
    webgpu_refine: false,
    loss_backend: "webgpu-compute",
    quality_metric_contract: {
      signal_space: "sRGB signal values",
      rgb_data_range: [0, 1],
      l1: "mean absolute error over RGB channels",
      psnr: "10*log10(1/MSE_RGB), with MSE averaged over RGB channels",
      psnr_unit: "dB",
      psnr_mse_floor: PSNR_MSE_FLOOR,
      ssim: "mean RGB-channel SSIM map; 11x11 Gaussian window, sigma 1.5, data range 1",
      objective_loss: "RGB L1 plus the configured alpha-objective weight; reported for evaluation and not fed back into training control",
    },
    latest_evaluation_step: null,
    latest_l1: null,
    latest_rgb_mse: null,
    latest_psnr: null,
    latest_alpha_l1: null,
    latest_alpha_ssim: null,
    latest_alpha_objective: null,
    latest_objective_loss: null,
    latest_ssim: null,
    latest_global_ssim: null,
    latest_windowed_ssim: null,
    latest_regional_ssim: null,
    latest_high_frequency: null,
    final_evaluation_step: null,
    final_metrics_complete: false,
    initial_l1: null,
    final_l1: null,
    initial_rgb_mse: null,
    final_rgb_mse: null,
    initial_psnr: null,
    final_psnr: null,
    initial_alpha_l1: null,
    final_alpha_l1: null,
    initial_alpha_ssim: null,
    final_alpha_ssim: null,
    initial_alpha_objective: null,
    final_alpha_objective: null,
    initial_objective_loss: null,
    final_objective_loss: null,
    initial_ssim: null,
    final_ssim: null,
    initial_global_ssim: null,
    final_global_ssim: null,
    initial_windowed_ssim: null,
    final_windowed_ssim: null,
    initial_regional_ssim: null,
    final_regional_ssim: null,
    initial_high_frequency: null,
    final_high_frequency: null,
    boundary_sigma: learningRates.boundarySigma,
    boundary_leak_count: null,
    boundary_max_leak: null,
    shape_stats: null,
    final_readback_step: null,
    scale_histogram: null,
    tiny_splat_count: null,
    tiny_splat_ratio: null,
    boundary_tiny_splat_count: null,
    boundary_tiny_splat_ratio: null,
    interior_tiny_splat_count: null,
    interior_tiny_splat_ratio: null,
    anisotropy_ratio: null,
    detail_splat_count: null,
    detail_splat_ratio: null,
    detail_anisotropy_max: null,
    surface_anisotropy_max: null,
    brush_taper_stats: null,
    thin_line_metrics: null,
    fusion_events: emptyFusionEvents(),
    fusion_refine_events: [],
    webgpu_relocation_events: [],
    webgpu_refine_events: [],
    representation: algorithmUsesLayeredOpaqueBrush(algorithm)
      ? "oriented-2d-layered-opaque-superellipse"
      : algorithmUsesRectangleKernel(algorithm)
        ? "oriented-2d-soft-rectangle"
        : "oriented-2d-gaussian",
    density_counters: null,
    last_density_counters: null,
    render_aware_density: true,
    weighted_mass_redistribution: true,
    sgld_2d: true,
    experimental_variants: experimentalVariants(),
    phase33_variants: phase33Variants(),
    phase37_variants: phase37Variants(),
    phase38_variants: phase38Variants(),
    phase39_variants: {
      ...phase39Variants(),
      stageAwareGrowth: runStageAwareGrowth,
      stageGrowthShares: runStageGrowthShares,
    },
    phase40_variants: phase40Variants(),
    phase45_variants: phase45Variants(),
    phase46_variants: runLayerSettings,
    virtual_camera_sampling: {
      enabled: runVirtualCameraSampling.enabled,
      bounded_depth: state.params.virtualDepthEnabled,
      bounded_depth_thickness: state.params.virtualDepthThickness,
      depth_center_weight: state.params.virtualDepthCenterWeight,
      depth_smoothness_weight: state.params.virtualDepthSmoothnessWeight,
      depth_learning_rate: state.params.virtualDepthLearningRate,
      soft_depth_constraint: state.params.virtualDepthSoftConstraintEnabled,
      depth_constraint_mode: state.params.virtualDepthSoftConstraintEnabled
        ? "coverage-angle-weighted-pseudo-huber"
        : "legacy-bounded-l2",
      depth_prior_delta: state.params.virtualDepthPriorDelta,
      depth_camera_confidence: "sqrt(teacher_coverage)*cos(polar_angle)",
      depth_update_interval: runVirtualCameraSampling.depthUpdateInterval,
      mode: runVirtualCameraSampling.enabled ? runVirtualCameraSampling.mode : "off",
      pool_slots: runVirtualCameraSampling.slots,
      virtual_slots: runVirtualCameraSampling.virtualSlots,
      frontless_sampling: runVirtualCameraSampling.frontlessSampling,
      requested_share_percent: runVirtualCameraSampling.requestedSharePercent,
      effective_share_percent: runVirtualCameraSampling.effectiveSharePercent,
      front_gradient_anchor_weight: runVirtualCameraSampling.frontGradientAnchorWeight,
      effective_gradient_share_percent: runVirtualCameraSampling.effectiveGradientSharePercent,
      gradient_balance_contract: runVirtualCameraSampling.frontGradientAnchorWeight > 0
        ? "virtual sampling preserved; canonical-front gradient caps virtual objective mass at 50%"
        : "selected-camera gradient",
      uniform_cameras: runVirtualCameraSampling.uniformCameras,
      virtual_camera_count: runVirtualCameraSampling.cameraCount,
      max_angle_degrees: runVirtualCameraSampling.maxAngleDegrees,
      shared_fov_degrees: runVirtualCameraSampling.fovDegrees,
      seed: runVirtualCameraSampling.seed,
      fov_degrees: runVirtualCameraSampling.fovDegrees,
      orbit_radius: runVirtualCameraOrbitRadius,
      teacher_coverage: runVirtualTeacherCoverage,
      target: [0, 0, 0],
      cameras: runVirtualCameraCatalog,
      camera_counts: Object.fromEntries(
        runVirtualCameraCatalog.map((camera) => [camera.id, 0]),
      ),
      active_camera_id: "front",
      preview_camera_id: "front",
      preview_front_restores: 0,
      front_steps: 0,
      virtual_steps: 0,
      warmup_steps: 0,
      invalid_region_mode: runVirtualCameraSampling.invalidRegionMode,
      three_dgs_multiview: runVirtualCameraSampling.threeDgsMultiview,
      projection_mode: runVirtualCameraSampling.threeDgsMultiview
        ? "camera-projected-3d-covariance"
        : "legacy-planar",
      gradient_routing: runVirtualCameraSampling.threeDgsMultiview
        ? runVirtualCameraSampling.frontGradientAnchorWeight > 0
          ? "sampled-view-all+balanced-front-anchor"
          : "selected-view-all"
        : null,
    },
    phase45_region_report: null,
    mid_training_overdensity_correction: {
      enabled: runGrowthSettings.midTrainingOverdensityCorrection,
      schedule: runGrowthSettings.overdensityCorrectionSchedule,
      interval: runGrowthSettings.overdensityCorrectionInterval,
      apply_until_step: experimentalGrowthSteps(steps, runGrowthApplyUntilFraction),
      scheduled_steps: overdensityCorrectionScheduleSteps(
        steps,
        experimentalGrowthSteps(steps, runGrowthApplyUntilFraction),
        runGrowthSettings,
      ),
      fixed_count: true,
      events: [],
    },
    overlap_diagnostics: null,
    color_space_audit: null,
    performance_trace: [],
    performance_profile_schedule: Object.fromEntries(performanceProfileSchedule(steps)),
    importance_stats: null,
    residual_destination_oracle: residualDestinationOracleRequested()
      ? {
          enabled: true,
          contract: "best-of-32-vs-residual-sum-cdf-tile-max",
          event_count: 0,
          sample_count: 0,
          selected_oracle_median: null,
          selected_oracle_p10: null,
          below_half_count: 0,
          below_half_fraction: 0,
          gate_triggered: false,
          events: [],
        }
      : null,
    coverage_stats: null,
    density_gpu_ms: 0,
    relocation_gpu_ms: 0,
    density_horizon: experimentalDensifySteps(steps),
    post_density_annealing: true,
    tile_culling_enabled: Boolean(trainingUiAdapter.controls.tileCullingToggle.checked),
    tile_retry_steps: 0,
    tile_retry_events: [],
    tile_retry_parameter_hash: null,
    qa_tile_index_capacity: qaTileIndexCapacityOverride(),
    train_layer_order: Boolean(trainingUiAdapter.controls.trainLayerOrder.checked),
    discrete_layers: runDiscreteLayerSettings.enabled,
    discrete_layer_count: runDiscreteLayerSettings.count,
    layer_update_interval: runLayerSettings.layerUpdateInterval,
    layer_update_rate: runLayerSettings.layerUpdateRate,
    layer_stage_aware_rate: runLayerSettings.stageAwareRate,
    layer_freeze_fraction: runLayerSettings.freezeFraction,
    layer_update_count: 0,
    layer_update_first_steps: [],
    layer_update_last_step: null,
    layer_telemetry_enabled: layerTelemetryEnabled(),
    layer_telemetry: [],
    layer_efficiency: null,
    current_visibility_compaction: null,
    current_visibility_compaction_events: [],
    current_visibility_compaction_removed_total: 0,
    current_contribution_compaction: null,
    current_contribution_compaction_events: [],
    current_contribution_compaction_removed_total: 0,
    current_contribution_compaction_deferred_count: 0,
    current_contribution_compaction_deferred_steps: [],
    current_contribution_compaction_settings: structuredClone(runCurrentContributionCompaction),
    opaque_paint_late_settle: {
      enabled: Boolean(runDiscreteLayerSettings.opaqueLayered),
      fraction: runDiscreteLayerSettings.opaquePaintSettleFraction,
      start_step: runDiscreteLayerSettings.opaqueLayered
        ? opaquePaintLateSettleStartStep(steps, runDiscreteLayerSettings.opaquePaintSettleFraction)
        : null,
      active: false,
      entered_step: null,
      count_at_start: null,
      count_at_end: null,
      count_stable: null,
      suppressed_total: 0,
      suppressed: {
        growth: 0,
        relocation: 0,
        layer_update: 0,
        discrete_layer: 0,
      },
      suppressed_steps: [],
      continuous_optimizer_active: true,
      final_metrics_active: true,
    },
    experimental_prefix_preserved: true,
    trend_checkpoints: [],
    ssim_trend: "",
    global_ssim_trend: "",
    psnr_trend: "",
    losses: [],
    rgb_mse: [],
    psnr: [],
    alpha_losses: [],
    alpha_ssim: [],
    objective_losses: [],
    ssim: [],
    global_ssim: [],
    windowed_ssim: [],
    regional_ssim_p10: [],
    densify_events: [],
    growth_schedule: {
      mode: runStageAwareGrowth
        ? "stage-aware-percentage-cap"
        : "threshold-percentage-target-closure",
      final_is_cap: true,
      final_is_target_at_growth_horizon: runGrowthApplyUntilFraction > 0,
      percentage: phase39Variants().growthFraction * 100,
      requested_apply_until_percentage: requestedGrowthApplyUntilFraction * 100,
      apply_until_percentage: runGrowthApplyUntilFraction * 100,
      apply_until_clamped_for_paint_settle:
        runGrowthApplyUntilFraction < requestedGrowthApplyUntilFraction,
      signal_threshold: phase39Variants().growthSignalThreshold,
      stage_aware: runStageAwareGrowth,
      phase_shares_percentage: {
        p1: runStageGrowthShares.p1 * 100,
        p2: runStageGrowthShares.p2 * 100,
        p3: runStageGrowthShares.p3 * 100,
      },
      detail_reserve_percentage: runStageGrowthShares.p3 * 100,
      density_stage_target: runStageAwareGrowth
        ? Math.min(finalCount, state.params.count + Math.round((finalCount - state.params.count) * (runStageGrowthShares.p1 + runStageGrowthShares.p2)))
        : null,
      growth_stage_target: finalCount,
      cap_reached_step: state.params.count >= finalCount ? 0 : null,
      target_reached_at_growth_horizon: state.params.count >= finalCount,
      target_reached_count: state.params.count >= finalCount ? state.params.count : null,
      training_early_stop: false,
      threshold_skips: 0,
    },
    density_controls: {
      adc_specialization_retired: true,
      mcmc_relocation_enabled: phase39Variants().mcmcRelocationEnabled,
    },
    training_residual_map: {
      mode: "webgpu-training-signal",
      evaluation_independent: true,
      cpu_readbacks: 0,
      refresh_count: 0,
      growth_refreshes: 0,
      relocation_refreshes: 0,
      last_step: null,
      last_reason: "",
      last_wall_ms: null,
      events: [],
    },
    stopped: false,
    started_at: new Date().toISOString(),
  };
  updateTrainingRunOwnership(run, { params: state.params, metrics: state.metrics });
  assertTrainingRun(run);
  syncRuntimeMetrics(learningRates, previewRefresh);
  publishState();

  let trainingError = null;
  try {
    let allocationResult = null;
    if (useAutoCapacityProbe) {
      allocationResult = await awaitTrainingRun(run, renderer.probeTrainingCapacity(trainingImage, state.params, finalCount));
      finalCount = allocationResult.capacity;
      trainingUiAdapter.controls.finalSplatCount.value = String(finalCount);
      state.metrics.final_splats = finalCount;
      state.metrics.capacity_probe = {
        mode: "auto-probe",
        requested: state.capacityProbe.requested,
        selected: finalCount,
        fast_path: allocationResult.fastPath,
        attempts: allocationResult.attempts,
      };
    } else {
      await awaitTrainingRun(run, renderer.uploadTrainState(
        trainingImage,
        state.params,
        finalCount,
        { verifyAllocation: finalCount > CAPACITY_PROBE_FAST_PATH_MAX },
      ));
      state.metrics.capacity_probe = {
        mode: trainingUiAdapter.controls.capacityMode.value,
        requested: finalCount,
        selected: finalCount,
        fast_path: finalCount <= CAPACITY_PROBE_FAST_PATH_MAX,
        attempts: [],
      };
    }
    state.metrics.growth_schedule.growth_stage_target = finalCount;
    state.metrics.growth_schedule.density_stage_target = runStageAwareGrowth
      ? Math.min(
          finalCount,
          state.metrics.initial_splats + Math.round(
            (finalCount - state.metrics.initial_splats) *
            (runStageGrowthShares.p1 + runStageGrowthShares.p2),
          ),
        )
      : null;
    const adaptiveInitializationVariants = {
      ...runAdaptiveGridInitialization,
      monochromeUnderpainting: Boolean(state.params.monochromeUnderpaintingEnabled),
    };
    const adaptiveInitialization = await awaitTrainingRun(run, renderer.applyAdaptiveGridInitialization(
      trainingImage,
      state.params,
      adaptiveInitializationVariants,
    ));
    state.metrics.initialization_adaptive = adaptiveInitialization;
    const periodicEvaluation = runPeriodicEvaluation;
    if (adaptiveInitialization.applied && runQaPeriodicMetrics) {
      await awaitTrainingRun(run, renderer.readTrainedColors(state.params));
      state.metrics.initial_param_snapshot = snapshotParams(state.params);
      state.metrics.initial_orientation = initialOrientationStats(state.params, state.image);
      log(
        `adaptive initialization applied fraction=${adaptiveInitialization.fraction.toFixed(2)} candidates=${adaptiveInitialization.candidate_count} moved~${adaptiveInitialization.moved_splats_estimate}`,
      );
    } else if (adaptiveInitialization.applied) {
      log(
        `adaptive initialization applied on WebGPU without CPU parameter readback; fraction=${adaptiveInitialization.fraction.toFixed(2)}`,
      );
    } else if (adaptiveInitialization.requested) {
      log(`adaptive initialization skipped: ${adaptiveInitialization.reason}`);
    }
    const allocatedMemory = renderer.trainingMemorySnapshot();
    state.metrics.gpu_training_memory = {
      accounting: "app-created-buffers",
      exact_device_vram: false,
      active_bytes_at_start: Math.round(allocatedMemory.activeBytes),
      reserved_bytes: Math.round(allocatedMemory.reservedBytes),
      peak_active_bytes: Math.round(allocatedMemory.activeBytes),
      peak_reserved_bytes: Math.round(allocatedMemory.reservedBytes),
      active_bytes_before_release: null,
      reserved_bytes_before_release: null,
      active_bytes_after_release: null,
      reserved_bytes_after_release: null,
    };
    log(`GPU training buffers reserved ${formatMB(allocatedMemory.reservedBytes)}; active estimate ${formatMB(allocatedMemory.activeBytes)}`);
    state.metrics.training_evaluation = {
      mode: periodicEvaluation ? "periodic-read-only" : "final-only",
      source: runLiveQualityMetrics ? "ui" : runQaPeriodicMetrics ? "qa" : "off",
      periodic_full_image_evaluations: 0,
      initial_full_image_evaluation: false,
      first_read_only_step: null,
      skipped_non_full_evaluations: 0,
    };
    log(
      `training start algorithm=${state.metrics.algorithm} backend=${state.metrics.backend} metrics=${periodicEvaluation ? "periodic-read-only" : "final-only"} growth=${state.metrics.growth_schedule.percentage}% threshold=${state.metrics.growth_schedule.signal_threshold} cap=${finalCount}`,
    );

    state.metrics.phase_relative_scale_guard = {
      enabled: runPhase33.phaseRelativeScaleGuard,
      reference: "phase-start geometric-mean scale median",
      correction: "aspect-preserving soft log-scale floor",
      strength: DEFAULT_RELATIVE_SCALE_GUARD_STRENGTH,
      ratios: {
        P1: runPhase33.p1RelativeScaleFloorRatio,
        P2: runPhase33.p2RelativeScaleFloorRatio,
        P3: runPhase33.p3RelativeScaleFloorRatio,
      },
      events: [],
    };

    const metricInterval = Math.max(1, Math.min(DEFAULT_MAX_METRIC_INTERVAL, state.recommendation?.metricInterval || Math.floor(steps / 60)));
    let appliedRuntimeSettingsRevision = state.runtimeSettingsRevision;
    resetTrainingTiming(false);
    const trainingPerfStarted = performance.now();
    let traceLastTime = trainingPerfStarted;
    let traceLastStep = 0;
    const gpuBatchTuning = {
      size: performanceSelection.gpuSchedulingMode === "adaptive" ? 8 : 1,
      targetMs: 120,
      submittedBatches: 0,
      submittedIterations: 0,
      maximumObservedSize: 1,
    };
    let relativeScaleGuardStage = null;
    for (let step = 1; step <= steps; step += 1) {
      const traceProfileLabels = state.webgpu.profile?.timing_backend === "timestamp-query"
        ? performanceProfileLabels(step, steps)
        : [];
      let stepDensityMs = 0;
      let stepTrainMs = 0;
      let stepRelocationMs = 0;
      let stepPresentationMs = 0;
      let presentation = "none";
      while (state.paused && !state.stopRequested) {
        await awaitTrainingRun(run, nextFrame());
      }
      const stepWallStarted = performance.now();
      if (appliedRuntimeSettingsRevision !== state.runtimeSettingsRevision) {
        learningRates = selectedLearningRates();
        if (algorithmUsesPaintKernel()) {
          learningRates = {
            ...learningRates,
            boundarySigma: Math.min(learningRates.boundarySigma, RECTANGLE_KERNEL_EXTENT),
          };
        }
        previewRefresh = selectedPreviewRefresh();
        syncRuntimeMetrics(learningRates, previewRefresh);
        appliedRuntimeSettingsRevision = state.runtimeSettingsRevision;
        eventLog(`runtime parameters applied revision=${appliedRuntimeSettingsRevision}`);
      }
      if (state.stopRequested) {
        state.metrics.stopped = true;
        log(`stopped at step ${step - 1}`);
        break;
      }
      const optimizerStage = curriculumTrainingStage(
        step,
        steps,
        runPhase33,
        renderer.trainState?.coarseImage,
        renderer.trainState?.midImage,
      );
      if (runPhase33.phaseRelativeScaleGuard && optimizerStage !== relativeScaleGuardStage) {
        if (relativeScaleGuardStage !== null) {
          await awaitTrainingRun(run, renderer.readTrainedColors(state.params));
        }
        const median = geometricMeanScaleMedian(state.params);
        const ratio = stageRelativeScaleFloorRatio(optimizerStage, runPhase33);
        const phase = optimizerStage === "coarse" ? "P1" : optimizerStage === "mid" ? "P2" : "P3";
        renderer.trainState.phaseRelativeScaleGuard = {
          enabled: ratio > 0,
          stage: optimizerStage,
          phase,
          median,
          ratio,
          floor: median * ratio,
          strength: DEFAULT_RELATIVE_SCALE_GUARD_STRENGTH,
        };
        state.metrics.phase_relative_scale_guard.events.push({
          step,
          phase,
          splats: state.params.count,
          median,
          ratio,
          floor: median * ratio,
        });
        relativeScaleGuardStage = optimizerStage;
      }
      const densitySteps = experimentalDensifySteps(steps);
      const growthSteps = experimentalGrowthSteps(steps, runGrowthApplyUntilFraction);
      const growthSettings = {
        ...phase39Variants(),
        stageAwareGrowth: runStageAwareGrowth,
        stageGrowthShares: runStageGrowthShares,
        growthApplyUntilFraction: runGrowthApplyUntilFraction,
      };
      const nextOverdensityCorrectionStep =
        state.metrics.mid_training_overdensity_correction.scheduled_steps[
          state.metrics.mid_training_overdensity_correction.events.length
        ] ?? null;
      const overdensityCorrectionDueAtStep = Boolean(
        nextOverdensityCorrectionStep !== null && step >= nextOverdensityCorrectionStep
      );
      const paintMutationAllowedAtStep = opaquePaintStructuralMutationAllowed(
        step,
        steps,
        Boolean(state.params?.opaqueLayered),
        runDiscreteLayerSettings.opaquePaintSettleFraction,
      );
      const densifyInterval = growthSettings.densifyInterval;
      const terminalGrowthClosureScheduledAtStep = growthSteps > 0 && step === growthSteps;
      const densifyScheduledAtStep = growthEventScheduled(
        step,
        densitySteps,
        growthSteps,
        growthSettings,
      );
      const densifyDue = paintMutationAllowedAtStep && densifyScheduledAtStep;
      const growthPlan = densifyDue
        ? growthSchedulePlan({
          step,
          steps,
          initialCount: state.metrics.initial_splats,
          currentCount: state.params.count,
          finalCount,
          growthFraction: growthSettings.growthFraction,
          growthApplyUntilFraction: growthSettings.growthApplyUntilFraction,
          densifyInterval,
          stageAware: growthSettings.stageAwareGrowth,
          stageGrowthShares: growthSettings.stageGrowthShares,
        })
        : null;
      const requestedTargetCount = growthPlan?.requestedCount ?? state.params.count;
      let targetCount = state.params.count;
      let growthResult = null;
      let growthStartCount = state.params.count;
      let densityGpuMs = 0;
      if (requestedTargetCount > state.params.count) {
        const densifyFailure = safetyFailure(computeBudgetFor(Number(trainingUiAdapter.controls.trainSize.value), requestedTargetCount, steps), "densify");
        if (densifyFailure) {
          state.metrics.stopped = true;
          state.metrics.safety_stop = {
            reason: densifyFailure.reason,
            context: densifyFailure.context,
            estimated_mb: densifyFailure.rec.estimatedMB,
            budget_mb: densifyFailure.rec.budgetMB,
            recommended_train_size: densifyFailure.rec.recommendedTrainSize,
            recommended_final_splats: densifyFailure.rec.recommendedFinalSplats,
          };
          setSafetyStop(densifyFailure);
          break;
        }
        const densityStarted = performance.now();
        await refreshTrainingResidualSignal(step, "growth", run);
        growthResult = await awaitTrainingRun(run, renderer.growExperimentalGpu(
          trainingImage,
          state.params,
          requestedTargetCount,
          step,
          steps,
          { forceZeroMassFallback: Boolean(growthPlan?.terminalClosure) },
        ));
        densityGpuMs = performance.now() - densityStarted;
        stepDensityMs += densityGpuMs;
        state.metrics.density_gpu_ms += densityGpuMs;
        if (!growthResult) {
          throw new Error("Experimental GPU densify failed; CPU fallback is disabled");
        }
        if (growthResult.grown) {
          targetCount = growthResult.count;
          state.params = growParamPlaceholders(state.params, targetCount);
          updateTrainingRunOwnership(run, { params: state.params });
          state.metrics.webgpu_densify = true;
          state.metrics.num_gaussians = state.params.count;
          if (state.params.count >= finalCount && state.metrics.growth_schedule.cap_reached_step === null) {
            state.metrics.growth_schedule.cap_reached_step = step;
          }
        } else {
          state.metrics.growth_schedule.threshold_skips += 1;
        }
      }
      if (densifyDue) {
        const operations = growthResult?.operations || {};
        const harmfulRectangleReport = state.metrics.front_footprint_refinement_v2;
        if (harmfulRectangleReport?.enabled) {
          harmfulRectangleReport.front_oversized_selections +=
            operations.harmful_rectangle_front_oversized_selections || 0;
          harmfulRectangleReport.high_contribution_selections +=
            operations.harmful_rectangle_high_contribution_selections || 0;
          harmfulRectangleReport.high_deviation_selections +=
            operations.harmful_rectangle_high_deviation_selections || 0;
          harmfulRectangleReport.candidate_selections +=
            operations.harmful_rectangle_candidate_selections || 0;
          harmfulRectangleReport.parent_replacements +=
            operations.harmful_rectangle_parent_replacements || 0;
          harmfulRectangleReport.children_created +=
            operations.harmful_rectangle_children_created || 0;
        }
        state.metrics.densify_events.push({
          step,
          stage: curriculumTrainingStage(
            step,
            steps,
            phase33Variants(),
            state.webgpu.renderer.trainState?.coarseImage,
            state.webgpu.renderer.trainState?.midImage,
          ),
          schedule_mode: growthPlan.mode,
          count_before: growthStartCount,
          count: state.params.count,
          desired_count: growthPlan.desiredCount,
          previous_desired_count: growthPlan.previousDesired,
          requested_count: requestedTargetCount,
          actual_count: state.params.count,
          added: growthResult?.grown ? targetCount - growthStartCount : 0,
          normal_increment: growthPlan.normalIncrement,
          catch_up_limit: growthPlan.catchUpLimit,
          remaining_growth_events: growthPlan.remainingGrowthEvents,
          schedule_debt: growthPlan.scheduleDebt,
          repair_increment: growthPlan.repairIncrement,
          terminal_capacity_closure: growthPlan.terminalClosure,
          zero_mass_fallback: Boolean(growthResult?.zeroMassFallback),
          headroom: Math.max(0, finalCount - state.params.count),
          candidate_mass: Number.isFinite(growthResult?.candidateMass) ? growthResult.candidateMass : null,
          eligible_source_count: (operations.source_claims || 0) + (operations.source_claim_conflicts || 0),
          qa_growth_comparisons: referenceGrowthTargets(
            growthStartCount,
            finalCount,
            operations.eligible_sources,
            growthSettings.qaGrowthComparisons,
          ),
          applied_source_count: (operations.split || 0) + (operations.duplicate || 0),
          split_count: operations.split || 0,
          duplicate_count: operations.duplicate || 0,
          reseed_count: operations.reseed || 0,
          source_claims: operations.source_claims || 0,
          source_claim_conflicts: operations.source_claim_conflicts || 0,
          surface_layer_candidates: operations.surface_layer_candidates || 0,
          surface_layer_promotions: operations.surface_layer_promotions || 0,
          harmful_rectangle_candidate_selections:
            operations.harmful_rectangle_candidate_selections || 0,
          harmful_rectangle_front_oversized_selections:
            operations.harmful_rectangle_front_oversized_selections || 0,
          harmful_rectangle_high_contribution_selections:
            operations.harmful_rectangle_high_contribution_selections || 0,
          harmful_rectangle_high_deviation_selections:
            operations.harmful_rectangle_high_deviation_selections || 0,
          harmful_rectangle_parent_replacements:
            operations.harmful_rectangle_parent_replacements || 0,
          harmful_rectangle_children_created:
            operations.harmful_rectangle_children_created || 0,
          structure_allocation_over_budget_source_selections:
            operations.structure_allocation_over_budget_source_selections || 0,
          structure_allocation_under_budget_source_selections:
            operations.structure_allocation_under_budget_source_selections || 0,
          threshold_skipped: Boolean(growthResult && !growthResult.grown),
          skipped_reason: requestedTargetCount <= growthStartCount
            ? "schedule-no-growth"
            : growthResult?.grown
              ? null
              : "candidate-threshold",
          all_or_none: true,
          algorithm: state.metrics.algorithm,
          density_phase: "growth",
          metrics_before: densityMetricSnapshot(),
          gpu_ms: densityGpuMs,
        });
        if (growthPlan.terminalClosure) {
          state.metrics.growth_schedule.target_reached_at_growth_horizon =
            state.params.count === finalCount;
          state.metrics.growth_schedule.target_reached_count = state.params.count;
        }
        if (residualDestinationOracleRequested()) {
          state.metrics.residual_destination_oracle = state.webgpu.renderer.residualDestinationOracleSummary();
        }
      }
      const paintDetailRecoveryScheduledAtStep =
        Boolean(state.params?.opaqueLayered) &&
        opaquePaintDetailRecoveryDue(
          step,
          steps,
          OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL,
        );
      const brushSurfaceRecoveryScheduledAtStep =
        normalizedKernelShape(state.params?.kernelShape) === "opaque-brush" &&
        paintDetailRecoveryScheduledAtStep;
      const brushSurfaceRecoveryDueAtStep = paintMutationAllowedAtStep && brushSurfaceRecoveryScheduledAtStep;
      const relocationScheduledAtStep =
        growthSettings.densityEventsEnabled &&
        growthSettings.mcmcRelocationEnabled &&
        gpuRelocationEnabled &&
        step > densifyWarmupSteps(densitySteps) &&
        (
          (
            step <= Math.floor(densitySteps * 0.85) &&
            step % EXPERIMENTAL_REFINE_EVERY === 0
          ) ||
          brushSurfaceRecoveryScheduledAtStep
        );
      const relocationDueAtStep = paintMutationAllowedAtStep && relocationScheduledAtStep && !overdensityCorrectionDueAtStep;
      const currentVisibilityCompactionDueAtStep = opaquePaintVisibilityCompactionDue(
        step,
        steps,
        {
          ...runDiscreteLayerSettings,
          currentContributionCompactionEnabled: runCurrentContributionCompaction.enabled,
          growthApplyUntilFraction: runGrowthApplyUntilFraction,
        },
      );
      let currentContributionCompactionDueAtStep = currentContributionCompactionDue(
        step,
        steps,
        {
          ...runCurrentContributionCompaction,
          opaqueLayered: runDiscreteLayerSettings.opaqueLayered,
          opaquePaintSettleFraction: runDiscreteLayerSettings.opaquePaintSettleFraction,
          growthApplyUntilFraction: runGrowthApplyUntilFraction,
        },
      );
      const currentContributionCompactionResetDueAtStep = currentContributionCompactionResetDue(
        step,
        steps,
        {
          ...runCurrentContributionCompaction,
          opaqueLayered: runDiscreteLayerSettings.opaqueLayered,
          opaquePaintSettleFraction: runDiscreteLayerSettings.opaquePaintSettleFraction,
          growthApplyUntilFraction: runGrowthApplyUntilFraction,
        },
      );
      const trainingStageAtStep = curriculumTrainingStage(
        step,
        steps,
        phase33Variants(),
        state.webgpu.renderer?.trainState?.coarseImage,
        state.webgpu.renderer?.trainState?.midImage,
      );
      const layerScheduleAtStep = layerOptimizationSettings(
        step,
        steps,
        trainingStageAtStep,
        runLayerSettings,
        Boolean(state.params?.opaqueLayered),
        runDiscreteLayerSettings.opaquePaintSettleFraction,
      );
      const discreteLayerScheduledAtStep = Boolean(
        state.params?.discreteLayersEnabled &&
        state.params?.discreteLayerMoveRadius > 0 &&
        step >= densitySteps &&
        paintDetailRecoveryScheduledAtStep
      );
      const surfaceLayerSortScheduleAtStep = scaleBiasedSurfaceLayerSortSchedule(
        step,
        steps,
        {
          enabled: state.params.surfaceLayerPriorEnabled,
          p1Interval: state.params.surfaceLayerPriorP1Interval,
          p2Interval: state.params.surfaceLayerPriorP2Interval,
          p3Interval: state.params.surfaceLayerPriorP3Interval,
          untilFraction: state.params.surfaceLayerPriorUntilFraction,
        },
      );
      let currentContributionCompactionDeferredBy = null;
      if (currentContributionCompactionDueAtStep && growthResult?.grown) {
        currentContributionCompactionDeferredBy = "growth";
      } else if (currentContributionCompactionDueAtStep && relocationDueAtStep) {
        currentContributionCompactionDeferredBy = "relocation";
      }
      if (currentContributionCompactionDeferredBy) {
        currentContributionCompactionDueAtStep = false;
        state.metrics.current_contribution_compaction_deferred_count += 1;
        if (state.metrics.current_contribution_compaction_deferred_steps.length < 32) {
          state.metrics.current_contribution_compaction_deferred_steps.push({
            step,
            reason: currentContributionCompactionDeferredBy,
          });
        }
      }
      const settleReport = state.metrics.opaque_paint_late_settle;
      if (settleReport?.enabled && !paintMutationAllowedAtStep) {
        if (!settleReport.active) {
          settleReport.active = true;
          settleReport.entered_step = step;
          settleReport.count_at_start = state.params.count;
        }
        const suppressedAtStep = {
          growth: densifyScheduledAtStep,
          relocation: relocationScheduledAtStep,
          layer_update: layerScheduleAtStep.scheduled,
          discrete_layer: discreteLayerScheduledAtStep,
        };
        const suppressedKinds = Object.entries(suppressedAtStep)
          .filter(([, suppressed]) => suppressed)
          .map(([kind]) => kind);
        for (const kind of suppressedKinds) {
          settleReport.suppressed[kind] += 1;
          settleReport.suppressed_total += 1;
        }
        if (suppressedKinds.length > 0 && settleReport.suppressed_steps.length < 32) {
          settleReport.suppressed_steps.push({ step, kinds: suppressedKinds });
        }
      }
      const structuralStep =
        densifyDue ||
        overdensityCorrectionDueAtStep ||
        relocationDueAtStep ||
        currentVisibilityCompactionDueAtStep ||
        currentContributionCompactionResetDueAtStep ||
        currentContributionCompactionDueAtStep ||
        layerScheduleAtStep.due ||
        surfaceLayerSortScheduleAtStep.due ||
        brushSurfaceRecoveryDueAtStep ||
        discreteLayerScheduledAtStep;
      const effectiveSyncInterval = effectiveTrainSyncInterval(
        state.params.count,
        state.metrics.train_sync_interval,
      );
      state.metrics.train_sync_policy.effective_interval = effectiveSyncInterval;
      const qaHashPending = qaTileOverflowFixtureEnabled() && !state.metrics.tile_retry_parameter_hash?.matches;
      const plannedGpuBatchSize = plannedAdaptiveGpuBatch({
        step,
        steps,
        desiredSize: gpuBatchTuning.size,
        metricInterval,
        previewRefresh,
        structuralStep,
        qaHashPending,
        densitySteps,
        growthSteps,
        growthSettings,
        runLayerSettings,
        paintSettings: runDiscreteLayerSettings,
        currentContributionSettings: runCurrentContributionCompaction,
      });
      // A balanced high-share virtual step encodes two views with distinct
      // configs into one logical optimizer update. Keep it sequential so the
      // shared config buffer cannot alias two camera passes in a GPU batch.
      const balancedFrontAnchorActive = Boolean(
        runVirtualCameraSampling.enabled &&
        runVirtualCameraSampling.frontGradientAnchorWeight > 0,
      );
      const gpuBatchSize = balancedFrontAnchorActive ? 1 : plannedGpuBatchSize;
      const shouldSyncTrain = performanceSelection.gpuSchedulingMode === "adaptive"
        ? true
        : qaHashPending ||
          structuralStep ||
          step % effectiveSyncInterval === 0 ||
          step === steps;
      const parameterHashBefore = qaHashPending
        ? await awaitTrainingRun(run, renderer.hashTrainParameters(state.params))
        : null;
      const trainStarted = performance.now();
      if (overdensityCorrectionDueAtStep) {
        const correctionStarted = performance.now();
        // Use the completed contribution window before any compaction reset at
        // the same scheduled step discards that evidence.
        await refreshTrainingResidualSignal(Math.max(1, step - 1), "overdensity correction", run);
        const correction = await awaitTrainingRun(
          run,
          renderer.correctMidTrainingOverdensityGpu(
            trainingImage,
            state.params,
            Math.max(1, step - 1),
            learningRates,
          ),
        );
        const correctionMs = performance.now() - correctionStarted;
        stepRelocationMs += correctionMs;
        state.metrics.relocation_gpu_ms += correctionMs;
        if (correction) {
          state.metrics.mid_training_overdensity_correction.events.push({ ...correction, trigger_step: step });
          state.metrics.phase45_region_report = correction.regions;
          if (correction.moved_splats > 0 && renderer.trainState) renderer.trainState.tileReady = false;
        }
      }
      if (currentVisibilityCompactionDueAtStep || currentContributionCompactionResetDueAtStep) {
        await awaitTrainingRun(run, renderer.resetImportanceWindowGpu(state.params.count));
      }
      if (gpuBatchSize > 1) {
        const batchSteps = Array.from({ length: gpuBatchSize }, (_, index) => step + index);
        await awaitTrainingRun(run, renderer.trainStepsGpu(
          trainingImage,
          state.params,
          learningRates,
          batchSteps,
          { virtualCameraSampling: runVirtualCameraSampling },
        ));
      } else {
        await awaitTrainingRun(run, renderer.trainStepGpu(trainingImage, state.params, learningRates, {
          sync: shouldSyncTrain,
          virtualCameraSampling: runVirtualCameraSampling,
        }));
      }
      stepTrainMs = performance.now() - trainStarted;
      gpuBatchTuning.submittedBatches += 1;
      gpuBatchTuning.submittedIterations += gpuBatchSize;
      gpuBatchTuning.maximumObservedSize = Math.max(gpuBatchTuning.maximumObservedSize, gpuBatchSize);
      if (performanceSelection.gpuSchedulingMode === "adaptive" && gpuBatchSize > 1 && stepTrainMs > 0.1) {
        const measuredPerIteration = stepTrainMs / gpuBatchSize;
        const proposed = Math.max(2, Math.min(MAX_TRAIN_BATCH_SIZE, Math.round(gpuBatchTuning.targetMs / measuredPerIteration)));
        gpuBatchTuning.size = Math.max(2, Math.min(MAX_TRAIN_BATCH_SIZE, Math.round((gpuBatchTuning.size + proposed) * 0.5)));
      }
      state.metrics.gpu_scheduling = {
        ...state.metrics.gpu_scheduling,
        current_batch_size: gpuBatchTuning.size,
        submitted_batches: gpuBatchTuning.submittedBatches,
        submitted_iterations: gpuBatchTuning.submittedIterations,
        maximum_observed_batch_size: gpuBatchTuning.maximumObservedSize,
        target_batch_ms: gpuBatchTuning.targetMs,
        balanced_front_anchor_sequential: balancedFrontAnchorActive,
      };
      const completedStep = step + gpuBatchSize - 1;
      step = completedStep;
      state.metrics.webgpu_train_executed = true;
      state.metrics.webgpu_train_update = Boolean(state.webgpu.renderer.lastTrainStats?.updated);
      const virtualCameraSamples = state.webgpu.renderer.lastTrainStats?.virtual_camera_samples?.length
        ? state.webgpu.renderer.lastTrainStats.virtual_camera_samples
        : state.webgpu.renderer.lastTrainStats?.virtual_camera_sample
          ? [{
              ...state.webgpu.renderer.lastTrainStats.virtual_camera_sample,
              step,
            }]
          : [];
      if (shouldSyncTrain) {
        const retrySteps = await resolveTileOverflowRetry(parameterHashBefore, run);
        if (retrySteps > 0) {
          state.metrics.webgpu_train_update = false;
          const resumedStep = Math.max(0, step - retrySteps);
          for (let revertedStep = resumedStep + 1; revertedStep <= step; revertedStep += 1) {
            const revertedCameraId = state.virtualCameraByStep.get(revertedStep);
            if (revertedCameraId) {
              state.metrics.virtual_camera_sampling.camera_counts[revertedCameraId] = Math.max(
                0,
                (state.metrics.virtual_camera_sampling.camera_counts[revertedCameraId] || 0) - 1,
              );
              state.virtualCameraByStep.delete(revertedStep);
            }
          }
          state.metrics.steps_done = resumedStep;
          step = resumedStep;
          continue;
        }
      }
      for (const virtualCameraSample of virtualCameraSamples) {
        const sampleStep = Math.max(1, Math.round(virtualCameraSample.step || step));
        const previousCameraId = state.virtualCameraByStep.get(sampleStep);
        if (previousCameraId && previousCameraId !== virtualCameraSample.cameraId) {
          state.metrics.virtual_camera_sampling.camera_counts[previousCameraId] = Math.max(
            0,
            (state.metrics.virtual_camera_sampling.camera_counts[previousCameraId] || 0) - 1,
          );
        }
        if (previousCameraId !== virtualCameraSample.cameraId) {
          state.virtualCameraByStep.set(sampleStep, virtualCameraSample.cameraId);
          state.metrics.virtual_camera_sampling.camera_counts[virtualCameraSample.cameraId] =
            (state.metrics.virtual_camera_sampling.camera_counts[virtualCameraSample.cameraId] || 0) + 1;
        }
        state.metrics.virtual_camera_sampling.active_camera_id = virtualCameraSample.cameraId;
        state.metrics.virtual_camera_sampling.last_sample = { ...virtualCameraSample };
        state.metrics.virtual_camera_sampling.front_steps = state.metrics.virtual_camera_sampling.camera_counts.front || 0;
        state.metrics.virtual_camera_sampling.virtual_steps = Object.entries(
          state.metrics.virtual_camera_sampling.camera_counts,
        ).reduce((total, [cameraId, count]) => total + (cameraId === "front" ? 0 : count), 0);
        if (Number.isFinite(state.webgpu.renderer.lastTrainStats.virtual_camera_orbit_radius)) {
          state.metrics.virtual_camera_sampling.orbit_radius = state.webgpu.renderer.lastTrainStats.virtual_camera_orbit_radius;
        }
      }
      if (state.webgpu.renderer.lastTrainStats?.layer_update_due) {
        state.metrics.layer_update_count += 1;
        state.metrics.layer_update_last_step = step;
        if (state.metrics.layer_update_first_steps.length < 16) state.metrics.layer_update_first_steps.push(step);
      }
      state.metrics.steps_done = step;
      if (surfaceLayerSortScheduleAtStep.due && state.metrics.surface_layer_prior) {
        const report = state.metrics.surface_layer_prior;
        report.event_count += 1;
        report.phase_counts[surfaceLayerSortScheduleAtStep.phase] += 1;
        if (report.events.length < 32) {
          report.events.push({
            step,
            phase: surfaceLayerSortScheduleAtStep.phase,
            layers: report.layers,
          });
        }
      }
      const relocationDue = relocationDueAtStep;
      if (relocationDue) {
        const relocationStarted = performance.now();
        await refreshTrainingResidualSignal(step, "relocation", run);
        const relocatedOnGpu = await awaitTrainingRun(run, renderer.relocateExperimentalGpu(trainingImage, state.params, step, learningRates));
        stepRelocationMs = performance.now() - relocationStarted;
        state.metrics.relocation_gpu_ms += stepRelocationMs;
        if (relocatedOnGpu) {
          state.metrics.webgpu_relocation = true;
          state.metrics.webgpu_refine = true;
          if (state.webgpu.renderer.trainState) state.webgpu.renderer.trainState.tileReady = false;
        }
      }
      if (currentVisibilityCompactionDueAtStep) {
        await applyCurrentVisibilityCompaction(step, steps, run);
      }
      if (currentContributionCompactionDueAtStep) {
        await applyCurrentContributionCompaction(
          step,
          steps,
          runCurrentContributionCompaction,
          run,
        );
      }
      if (state.stopRequested) {
        state.metrics.stopped = true;
        log(`stopped at step ${step}`);
        break;
      }
      const periodicEvaluationDue = Boolean(
        periodicEvaluation &&
        step < steps &&
        (step % metricInterval === 0 || traceProfileLabels.length > 0)
      );
      const periodicReadOnlyReady = periodicEvaluationDue;
      if (periodicReadOnlyReady) {
        const presentationStarted = performance.now();
        presentation = "metrics";
        const metricsFailure = safetyFailure(computeBudgetFor(Number(trainingUiAdapter.controls.trainSize.value), state.params.count, steps), "metrics");
        if (metricsFailure) {
          state.metrics.stopped = true;
          state.metrics.safety_stop = {
            reason: metricsFailure.reason,
            context: metricsFailure.context,
            estimated_mb: metricsFailure.rec.estimatedMB,
            budget_mb: metricsFailure.rec.budgetMB,
            recommended_train_size: metricsFailure.rec.recommendedTrainSize,
            recommended_final_splats: metricsFailure.rec.recommendedFinalSplats,
          };
          setSafetyStop(metricsFailure);
          break;
        }
        await updatePreview(step, false, { present: false, readOnlyPeriodic: true }, run);
        if (state.metrics.training_evaluation) {
          state.metrics.training_evaluation.periodic_full_image_evaluations += 1;
          if (state.metrics.training_evaluation.first_read_only_step === null) {
            state.metrics.training_evaluation.first_read_only_step = step;
            state.metrics.initial_l1 = state.metrics.latest_l1;
            state.metrics.initial_rgb_mse = state.metrics.latest_rgb_mse;
            state.metrics.initial_psnr = state.metrics.latest_psnr;
            state.metrics.initial_alpha_l1 = state.metrics.latest_alpha_l1;
            state.metrics.initial_alpha_ssim = state.metrics.latest_alpha_ssim;
            state.metrics.initial_alpha_objective = state.metrics.latest_alpha_objective;
            state.metrics.initial_objective_loss = state.metrics.latest_objective_loss;
            state.metrics.initial_ssim = state.metrics.latest_ssim;
            state.metrics.initial_global_ssim = state.metrics.latest_global_ssim;
            state.metrics.initial_windowed_ssim = state.metrics.latest_windowed_ssim;
            state.metrics.initial_regional_ssim = state.metrics.latest_regional_ssim;
            state.metrics.initial_high_frequency = state.metrics.latest_high_frequency;
          }
        }
        if (!performanceSelection.asyncPresentation) await awaitTrainingRun(run, nextFrame());
        stepPresentationMs = performance.now() - presentationStarted;
      } else if (shouldPresentTrainingStep(step, previewRefresh)) {
        const presentationStarted = performance.now();
        presentation = "preview";
        if (!(await presentTrainingPreview(step, run))) throw new Error("WebGPU live preview state is unavailable");
        if (!performanceSelection.asyncPresentation) await awaitTrainingRun(run, nextFrame());
        stepPresentationMs = performance.now() - presentationStarted;
      } else if (step % 32 === 0) {
        const presentationStarted = performance.now();
        presentation = "status";
        trainingUiAdapter.controls.stepText.textContent = `${step} / ${state.metrics.steps_requested}`;
        publishState();
        if (!performanceSelection.asyncPresentation) await awaitTrainingRun(run, nextFrame());
        stepPresentationMs = performance.now() - presentationStarted;
      }
      if (step % metricInterval === 0 || step === steps) {
        const memorySnapshot = state.webgpu.renderer?.trainingMemorySnapshot?.();
        if (memorySnapshot && state.metrics.gpu_training_memory) {
          state.metrics.gpu_training_memory.peak_active_bytes = Math.max(
            state.metrics.gpu_training_memory.peak_active_bytes,
            Math.round(memorySnapshot.activeBytes),
          );
          state.metrics.gpu_training_memory.peak_reserved_bytes = Math.max(
            state.metrics.gpu_training_memory.peak_reserved_bytes,
            Math.round(memorySnapshot.reservedBytes),
          );
        }
        const now = performance.now();
        const intervalSteps = step - traceLastStep;
        const intervalMs = now - traceLastTime;
        if (intervalSteps > 0 && intervalMs >= 0.05) {
          const tracePoint = {
          step,
          phase: experimentalDensityPhase(step, steps),
          active_splats: state.params.count,
          requested_splats: requestedTargetCount,
          splat_cap: finalCount,
          growth_horizon: growthSteps,
          growth_fraction: growthSettings.growthFraction,
          growth_signal_threshold: growthSettings.growthSignalThreshold,
          elapsed_ms: now - trainingPerfStarted,
          interval_ms: intervalMs,
          interval_steps: intervalSteps,
          iterations_per_second: (intervalSteps * 1000) / intervalMs,
          step_wall_ms: now - stepWallStarted,
          train_ms: stepTrainMs,
          density_ms: stepDensityMs,
          relocation_ms: stepRelocationMs,
          presentation_ms: stepPresentationMs,
          presentation,
          profile_labels: [...traceProfileLabels],
          sync: shouldSyncTrain,
          sync_interval: effectiveSyncInterval,
          tile_builds: state.webgpu.renderer?.lastTrainStats?.tile_builds ?? null,
          tile_candidates: state.metrics.tile_counters?.total ?? null,
          tile_capacity: state.metrics.tile_counters?.capacity ?? null,
          };
          if (state.metrics.performance_trace.length < 160) state.metrics.performance_trace.push(tracePoint);
          else state.metrics.performance_trace[159] = tracePoint;
          state.metrics.scheduling_profile = summarizeTrainingScheduling(
            state.metrics.stage_profile,
            state.metrics.performance_trace,
          );
          traceLastTime = now;
          traceLastStep = step;
        }
      }
      recordTrainingTiming(step, performance.now() - stepWallStarted);
    }
    const lateSettleReport = state.metrics.opaque_paint_late_settle;
    if (lateSettleReport?.enabled && lateSettleReport.active) {
      lateSettleReport.count_at_end = state.params.count;
      lateSettleReport.count_stable = lateSettleReport.count_at_start === state.params.count;
      if (!lateSettleReport.count_stable) {
        throw new Error(
          `Opaque paint late-settle changed splat count: ${lateSettleReport.count_at_start} -> ${state.params.count}`,
        );
      }
    }
    if (!state.metrics.safety_stop) {
      const finalizationStarted = performance.now();
      setStatus("finalizing");
      setTrainingMessage(`Finalizing ${state.params.count.toLocaleString()} splats on WebGPU...`);
      publishState();
      await awaitTrainingRun(run, nextFrame());
      if (previewRefresh === "final") {
        state.previewMode = "splats";
        updatePreviewModeControls();
      }
      await updatePreview(state.metrics.steps_done, true, {}, run);
      state.metrics.training_evaluation.final_full_image_evaluations = 1;
      if (algorithmUsesRectangleKernel(algorithm)) {
        state.metrics.rectangle_orientation_constraints = rectangleConstraintProbe(state.params);
      }
      // The final metric pass has read the trained parameters and verified the
      // visible training surface. Preserve that CPU result before allocating a
      // second GPU cache so a device loss in the copy phase stays recoverable.
      state.metrics.final_cpu_result_ready_at = new Date().toISOString();
      state.metrics.final_cpu_result_ready = {
        source: "final-metrics-readback-and-render-parity",
        count: state.params.count,
        step: state.metrics.steps_done,
      };
      const resultCachePreserved = await awaitTrainingRun(run, renderer.preserveResultRenderState(state.image, state.params));
      state.metrics.final_result_cache_preserved = resultCachePreserved;
      await awaitTrainingRun(run, nextFrame());
      captureSplatAdjustmentBaseline();
      state.metrics.finalization_wall_ms = performance.now() - finalizationStarted;
      state.metrics.final_diagnostic_sample_limit = MAX_FINAL_DIAGNOSTIC_SAMPLES;
    }
    state.metrics.finished_at = new Date().toISOString();
  } catch (error) {
    if (error.trainingRunCancelled || !ownsTrainingRun(run)) return;
    if (error.safetyStop) {
      state.metrics.stopped = true;
    } else {
      trainingError = error;
      state.metrics.webgpu_train_error = error.message;
    }
    const rec = state.recommendation || updateMemoryRecommendation();
    log(`WebGPU training stopped: ${error.message}; estimate=${rec.estimatedMB}MB recommended=${rec.recommendedTrainSize}px/${rec.recommendedFinalSplats} splats`);
  } finally {
    const runOwnsGlobalState = ownsTrainingRun(run);
    const memoryBeforeRelease = renderer?.trainingMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
    const reservedBeforeRelease = memoryBeforeRelease.reservedBytes || 0;
    if (runOwnsGlobalState && run.metrics?.gpu_training_memory) {
      run.metrics.gpu_training_memory.active_bytes_before_release = Math.round(memoryBeforeRelease.activeBytes);
      run.metrics.gpu_training_memory.reserved_bytes_before_release = Math.round(memoryBeforeRelease.reservedBytes);
      run.metrics.gpu_training_memory.peak_active_bytes = Math.max(
        run.metrics.gpu_training_memory.peak_active_bytes,
        Math.round(memoryBeforeRelease.activeBytes),
      );
      run.metrics.gpu_training_memory.peak_reserved_bytes = Math.max(
        run.metrics.gpu_training_memory.peak_reserved_bytes,
        Math.round(memoryBeforeRelease.reservedBytes),
      );
    }
    renderer?.disposeTrainState();
    const memoryAfterRelease = renderer?.trainingMemorySnapshot?.() || { activeBytes: 0, reservedBytes: 0 };
    if (runOwnsGlobalState && run.metrics?.gpu_training_memory) {
      run.metrics.gpu_training_memory.active_bytes_after_release = Math.round(memoryAfterRelease.activeBytes);
      run.metrics.gpu_training_memory.reserved_bytes_after_release = Math.round(memoryAfterRelease.reservedBytes);
    }
    if (!runOwnsGlobalState) return;
    if (
      run.metrics?.final_cpu_result_ready_at &&
      !renderer?.currentResultBuffers(run.params) &&
      state.webgpu.renderer === renderer &&
      !renderer.deviceLost
    ) {
      const uploaded = await renderer.uploadResultRenderState(run.params);
      if (!ownsTrainingRun(run)) return;
      run.metrics.final_result_cache_preserved = uploaded;
      run.metrics.final_result_cache_rebuilt_after_train_release = uploaded;
    }
    if (reservedBeforeRelease > 0) log(`GPU training buffers released ${formatMB(reservedBeforeRelease)}`);
    state.running = false;
    state.paused = false;
    if (state.previewMode === "splats" && state.params) showCanvas("gpu");
    trainingUiAdapter.controls.startButton.disabled = false;
    trainingUiAdapter.controls.pauseButton.disabled = true;
    trainingUiAdapter.controls.pauseButton.textContent = "Pause";
    trainingUiAdapter.controls.stopButton.disabled = true;
    setPausedRuntimeControlsEnabled(false);
    setInputControlsDisabled(false);
    const deviceLost = !state.webgpu.supported && String(state.webgpu.reason).startsWith("device lost:");
    if (state.webGpuRecoveryPending) {
      setStatus("recovering gpu");
      setTrainingMessage("Training result preserved. Restoring the GPU preview...", "info");
      updateDownloads(false);
    } else if (trainingError || deviceLost) {
      setStatus("error");
      setTrainingMessage(`Training failed: ${trainingError?.message || state.webgpu.reason}`, "error");
      updateDownloads(false);
    } else {
      const exportCoverage = exportCoverageStatus();
      state.metrics.background_exposure_violation = exportCoverage.warning === true;
      setStatus(
        state.metrics.safety_stop
          ? "safety stopped"
          : state.metrics.stopped
            ? "stopped"
            : "done",
      );
      setTrainingMessage(
        state.metrics.safety_stop
          ? "Training stopped by the GPU safety guard."
          : state.metrics.stopped
            ? "Training stopped."
            : "Training complete.",
        state.metrics.safety_stop ? "error" : "success",
      );
      updateDownloads(!state.metrics.safety_stop);
    }
    publishState();
  }
}
