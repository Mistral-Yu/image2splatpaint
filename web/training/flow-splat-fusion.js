const FLOW_PROGRESSIVE_GROWTH_INTERVAL = 100;
const FLOW_PROGRESSIVE_GROWTH_APPLY_UNTIL = 0.90;
const FLOW_SPLIT_APPLY_UNTIL = 0.75;

function buildFlowProgressiveGrowthSchedule(iterations, broadParentCount, fullParentCount) {
  const safeIterations = Math.max(1, Math.round(iterations));
  const safeFullCount = Math.max(1, Math.round(fullParentCount));
  const safeBroadCount = Math.max(1, Math.min(safeFullCount, Math.round(broadParentCount)));
  if (safeIterations < 3 || safeBroadCount >= safeFullCount) {
    return {
      parentCounts: [safeFullCount],
      iterationCounts: [safeIterations],
      growthIterationBudget: 0,
      settleIterations: safeIterations,
    };
  }

  // Grow a little at a time through the first 90% of training. The final 10%
  // starts at the complete Splat budget and is reserved for fixed-count settle.
  const settleIterations = Math.max(1, Math.round(
    safeIterations * (1 - FLOW_PROGRESSIVE_GROWTH_APPLY_UNTIL),
  ));
  const growthIterationBudget = Math.max(1, safeIterations - settleIterations);
  const preFullStageCount = Math.max(1, Math.ceil(
    growthIterationBudget / FLOW_PROGRESSIVE_GROWTH_INTERVAL,
  ));
  const parentCounts = [];
  const iterationCounts = [];
  let remainingGrowthIterations = growthIterationBudget;
  for (let stage = 0; stage < preFullStageCount; stage += 1) {
    const progress = stage / preFullStageCount;
    const parentCount = Math.max(safeBroadCount, Math.min(
      safeFullCount - 1,
      Math.round(safeBroadCount + (safeFullCount - safeBroadCount) * progress),
    ));
    const stageIterations = Math.min(
      FLOW_PROGRESSIVE_GROWTH_INTERVAL,
      remainingGrowthIterations,
    );
    if (parentCounts.at(-1) === parentCount) {
      iterationCounts[iterationCounts.length - 1] += stageIterations;
    } else {
      parentCounts.push(parentCount);
      iterationCounts.push(stageIterations);
    }
    remainingGrowthIterations -= stageIterations;
  }
  parentCounts.push(safeFullCount);
  iterationCounts.push(settleIterations);
  return { parentCounts, iterationCounts, growthIterationBudget, settleIterations };
}

function measureFlowControlPointDrift(strokePlan, params, detailOffset = 0) {
  if (!Array.isArray(strokePlan) || !(params instanceof Float32Array)) return 0;
  const parameterNames = [
    "start_x", "start_y",
    "control_1_x", "control_1_y",
    "control_2_x", "control_2_y",
    "end_x", "end_y",
  ];
  let squaredTotal = 0;
  let valueCount = 0;
  for (let stroke = 0; stroke < strokePlan.length; stroke += 1) {
    const base = (detailOffset + stroke) * 16;
    if (base + 7 >= params.length) break;
    for (let component = 0; component < parameterNames.length; component += 1) {
      const anchor = Number(strokePlan[stroke]?.[parameterNames[component]]);
      if (!Number.isFinite(anchor)) continue;
      const difference = params[base + component] - anchor;
      squaredTotal += difference * difference;
      valueCount += 1;
    }
  }
  return Math.sqrt(squaredTotal / Math.max(1, valueCount));
}

function createFlowGeometryAnchorParams(strokePlan) {
  const params = new Float32Array(strokePlan.length * 16);
  const parameterNames = [
    "start_x", "start_y",
    "control_1_x", "control_1_y",
    "control_2_x", "control_2_y",
    "end_x", "end_y",
  ];
  for (let stroke = 0; stroke < strokePlan.length; stroke += 1) {
    const base = stroke * 16;
    for (let component = 0; component < parameterNames.length; component += 1) {
      params[base + component] = Number(strokePlan[stroke]?.[parameterNames[component]]) || 0;
    }
  }
  return params;
}

async function trainFlowSplatFusion(run) {
  const algorithm = selectedAlgorithm();
  const compatibilityCurveAlias = algorithm.id === CURVE_SPLAT_CHAIN_ALGORITHM_ID;
  if (algorithm.id !== FLOW_SPLAT_FUSION_ALGORITHM_ID && !compatibilityCurveAlias) {
    throw new Error(`Flow-stroke trainer does not match selected algorithm: ${algorithm.id}`);
  }
  assertTrainingRun(run);
  const sourceImage = state.image;
  const flowQaParams = new URLSearchParams(location.search);
  const allowedFlowStrokeTextures = ["baseline", "fine-bristles", "brush-dabs"];
  const requestedFlowStrokeTexture = allowedFlowStrokeTextures.includes(
    trainingUiAdapter.controls.flowSplatFusionStrokeTexture.value,
  ) ? trainingUiAdapter.controls.flowSplatFusionStrokeTexture.value : "brush-dabs";
  const qaFlowStrokeTexture = flowQaParams.get("flow-stroke-texture");
  let flowStrokeTexture = allowedFlowStrokeTextures.includes(qaFlowStrokeTexture)
    ? qaFlowStrokeTexture
    : requestedFlowStrokeTexture;
  if (flowQaParams.has("flow-bristle-bundle") && !flowQaParams.has("flow-stroke-texture")) {
    flowStrokeTexture = flowQaParams.get("flow-bristle-bundle") === "1"
      ? "fine-bristles"
      : "baseline";
  }
  const flowBristleBundle = flowStrokeTexture === "fine-bristles";
  const flowBrushDabs = flowStrokeTexture === "brush-dabs";
  const flowTextureGuidedDabs = flowBrushDabs
    && flowQaParams.get("flow-texture-guided-dabs") !== "0";
  trainingUiAdapter.controls.flowSplatFusionStrokeTexture.value = flowStrokeTexture;
  // Adaptive split/merge is the product Baseline. The retired fixed-plan path
  // is preserved only in the local archive, not as a runtime mode.
  const flowTopologyMode = flowTextureGuidedDabs
    ? "adaptive-brush-dab-texture-guided"
    : "adaptive-baseline";
  const flowPaintCurriculum = flowQaParams.has("flow-paint-curriculum")
    ? flowQaParams.get("flow-paint-curriculum") !== "0"
    : trainingUiAdapter.controls.flowSplatFusionPaintCurriculum.checked;
  const flowResidualPriorityTileSampling =
    flowQaParams.get("flow-residual-priority-tiles") !== "0";
  trainingUiAdapter.controls.flowSplatFusionPaintCurriculum.checked = flowPaintCurriculum;
  const splatsPerChain = flowBrushDabs
    ? Image2SplatPaintFlowRibbonTrainer.constants.BRUSH_DAB_SAMPLES
    : flowBristleBundle
      ? Image2SplatPaintFlowRibbonTrainer.constants.BRISTLE_BUNDLE_SAMPLES
      : Image2SplatPaintFlowRibbonTrainer.constants.CURVE_SAMPLES;
  const maxSide = Math.max(LIMITS.trainSizeMin, Math.min(512, Math.round(
    Number(trainingUiAdapter.controls.trainSize.value) || DEFAULT_MAX_SIDE,
  )));
  const requestedSplatBudget = Math.max(256, Math.min(14000, Math.round(
    Number(trainingUiAdapter.controls.finalSplatCount.value) || DEFAULT_FINAL_SPLATS,
  )));
  const iterations = Math.max(1, Math.min(10000, Math.round(
    Number(trainingUiAdapter.controls.stepCount.value) || DEFAULT_ITERATIONS,
  )));
  const previewRefresh = ["frame", "10", "final"].includes(trainingUiAdapter.controls.previewRefresh.value)
    ? trainingUiAdapter.controls.previewRefresh.value
    : "10";
  const previewInterval = previewRefresh === "frame" ? 1 : previewRefresh === "10" ? 10 : 0;
  const colorAnchor = Math.max(0, Math.min(0.05, Number(trainingUiAdapter.controls.flowSplatFusionColorAnchor.value) || 0));
  const strokeWidthPercent = Math.max(25, Math.min(300, Math.round(
    Number(trainingUiAdapter.controls.flowSplatFusionWidthPercent.value) || 300,
  )));
  const qaSplatSizeVariationPercent = Number(flowQaParams.get("flow-splat-size-variation"));
  const splatSizeVariationPercent = Math.max(0, Math.min(100, Math.round(
    flowQaParams.has("flow-splat-size-variation")
      && Number.isFinite(qaSplatSizeVariationPercent)
      ? qaSplatSizeVariationPercent
      : Number(trainingUiAdapter.controls.flowSplatFusionSplatSizeVariation.value) || 0,
  )));
  const initialWidthMinimumPercent = Math.max(25, Math.min(300, Math.round(
    Number(trainingUiAdapter.controls.flowSplatFusionInitialWidthMin.value) || 55,
  )));
  const initialWidthMaximumPercent = Math.max(initialWidthMinimumPercent, Math.min(300, Math.round(
    Number(trainingUiAdapter.controls.flowSplatFusionInitialWidthMax.value) || 165,
  )));
  const qaFixedStrokeOpacity = Number(flowQaParams.get("flow-fixed-opacity"));
  const fixedStrokeOpacity = Math.max(0.05, Math.min(0.995,
    flowQaParams.has("flow-fixed-opacity") && Number.isFinite(qaFixedStrokeOpacity)
      ? qaFixedStrokeOpacity
      : Number(trainingUiAdapter.controls.flowSplatFusionFixedOpacity.value) || 0.995,
  ));
  const qaFrontWidthMaximumPercent = Number(flowQaParams.get("flow-front-width-max"));
  const frontWidthMaximumPercent = Math.max(25, Math.min(400, Math.round(
    flowQaParams.has("flow-front-width-max") && Number.isFinite(qaFrontWidthMaximumPercent)
      ? qaFrontWidthMaximumPercent
      : Number(trainingUiAdapter.controls.flowSplatFusionFrontWidthMax.value) || 300,
  )));
  const qaFrontWidthLearningPercent = Number(flowQaParams.get("flow-front-width-learning"));
  const frontWidthLearningPercent = Math.max(100, Math.min(800, Math.round(
    flowQaParams.has("flow-front-width-learning") && Number.isFinite(qaFrontWidthLearningPercent)
      ? qaFrontWidthLearningPercent
      : Number(trainingUiAdapter.controls.flowSplatFusionFrontWidthLearning.value) || 400,
  )));
  const frontWidthLearningScale = frontWidthLearningPercent / 100;
  const qaStartingWidthDivisor = Number(flowQaParams.get("flow-starting-width-divisor"));
  const startingWidthDivisor = Math.max(8, Math.min(256, Math.round(
    flowQaParams.has("flow-starting-width-divisor") && Number.isFinite(qaStartingWidthDivisor)
      ? qaStartingWidthDivisor
      : Number(trainingUiAdapter.controls.flowSplatFusionStartingWidthDivisor.value) || 32,
  )));
  const qaStartingLengthPercent = Number(flowQaParams.get("flow-starting-length-percent"));
  const startingLengthPercent = Math.max(100, Math.min(300, Math.round(
    flowQaParams.has("flow-starting-length-percent") && Number.isFinite(qaStartingLengthPercent)
      ? qaStartingLengthPercent
      : Number(trainingUiAdapter.controls.flowSplatFusionStartingLengthPercent.value) || 160,
  )));
  const qaResidualMovePerStagePx = Number(flowQaParams.get("flow-residual-move-px"));
  const residualMovePerStagePx = Math.max(0, Math.min(8,
    flowQaParams.has("flow-residual-move-px") && Number.isFinite(qaResidualMovePerStagePx)
      ? qaResidualMovePerStagePx
      : Number(trainingUiAdapter.controls.flowSplatFusionResidualMovePx.value) || 0,
  ));
  const scaleMatchedResidualRepaint = flowQaParams.has("flow-scale-matched-residual-repaint")
    ? flowQaParams.get("flow-scale-matched-residual-repaint") !== "0"
    : trainingUiAdapter.controls.flowSplatFusionScaleMatchedResidualRepaint.checked;
  trainingUiAdapter.controls.flowSplatFusionScaleMatchedResidualRepaint.checked =
    scaleMatchedResidualRepaint;
  const splatUnderpainting = flowQaParams.has("flow-coverage-backcoat")
    ? flowQaParams.get("flow-coverage-backcoat") !== "0"
    : trainingUiAdapter.controls.flowSplatUnderpainting.checked;
  trainingUiAdapter.controls.flowSplatUnderpainting.checked = splatUnderpainting;
  const defaultUnderpaintPercent = 10;
  const rawUnderpaintPercent = Number(trainingUiAdapter.controls.flowSplatUnderpaintPercent.value);
  const underpaintPercent = Math.max(0, Math.min(50, Number.isFinite(rawUnderpaintPercent)
    ? rawUnderpaintPercent
    : defaultUnderpaintPercent));
  const underpaintSplatBudget = splatUnderpainting
    ? Math.max(1, Math.min(requestedSplatBudget - 1, Math.round(
        requestedSplatBudget * underpaintPercent / 100,
      )))
    : 0;
  const detailSplatBudget = Math.max(1, requestedSplatBudget - underpaintSplatBudget);
  const detailParentBudget = Math.max(1, Math.floor(detailSplatBudget / splatsPerChain));
  const rawMaximumRibbonArcPercent = Number(
    trainingUiAdapter.controls.flowSplatFusionMaxArcPercent.value,
  );
  const maximumRibbonArcPercent = Math.max(0, Math.min(50, Number.isFinite(rawMaximumRibbonArcPercent)
    ? rawMaximumRibbonArcPercent
    : 10));
  trainingUiAdapter.controls.trainSize.value = String(maxSide);
  trainingUiAdapter.controls.finalSplatCount.value = String(requestedSplatBudget);
  trainingUiAdapter.controls.stepCount.value = String(iterations);
  trainingUiAdapter.controls.flowSplatFusionColorAnchor.value = String(colorAnchor);
  trainingUiAdapter.controls.flowSplatFusionWidthPercent.value = String(strokeWidthPercent);
  trainingUiAdapter.controls.flowSplatFusionSplatSizeVariation.value = String(
    splatSizeVariationPercent,
  );
  trainingUiAdapter.controls.flowSplatFusionInitialWidthMin.value = String(initialWidthMinimumPercent);
  trainingUiAdapter.controls.flowSplatFusionInitialWidthMax.value = String(initialWidthMaximumPercent);
  trainingUiAdapter.controls.flowSplatFusionFixedOpacity.value = String(fixedStrokeOpacity);
  trainingUiAdapter.controls.flowSplatFusionFrontWidthMax.value = String(
    frontWidthMaximumPercent,
  );
  trainingUiAdapter.controls.flowSplatFusionFrontWidthLearning.value = String(
    frontWidthLearningPercent,
  );
  trainingUiAdapter.controls.flowSplatFusionStartingWidthDivisor.value = String(startingWidthDivisor);
  trainingUiAdapter.controls.flowSplatFusionStartingLengthPercent.value = String(startingLengthPercent);
  trainingUiAdapter.controls.flowSplatFusionResidualMovePx.value = String(residualMovePerStagePx);
  trainingUiAdapter.controls.flowSplatUnderpaintPercent.value = String(underpaintPercent);
  trainingUiAdapter.controls.flowSplatFusionMaxArcPercent.value = String(maximumRibbonArcPercent);

  const [width, height] = resizedSize(sourceImage.width, sourceImage.height, maxSide);
  const trainingImage = width === sourceImage.width && height === sourceImage.height
    ? sourceImage
    : { ...sourceImage, ...resizeFloatImageBilinear(sourceImage, width, height) };
  const unscaledReference = Image2SplatPaintFlowPaintReference.createFlowPaintReference(trainingImage, {
    seed: 240825,
    strength: 1,
    profile: "connected-ribbon-v1",
    maxStrokes: detailParentBudget,
    minimumStrokes: 1,
    includeStrokePlan: true,
    planOnly: true,
    maximumRibbonArcFraction: maximumRibbonArcPercent / 100,
    textureGuidedAllocation: flowTextureGuidedDabs,
  });
  // Stroke width is a global ceiling for topology-level parent widths. It must
  // not multiply every generated stroke before the per-layer width range is
  // applied; doing both made the 300% default compound to roughly 9x for some
  // front parents before their physical Brush-dab profiles were evaluated.
  const strokeWidthMaximumFactor = strokeWidthPercent / 100;
  const reference = unscaledReference;
  // Standard and Fine use anisotropic Gaussian Splats. Brush dabs keeps the
  // same curve/topology optimizer but uses compact opaque-interior Brush
  // Splats in both forward and backward. The old continuous-ribbon path
  // remains dormant for checkpoint reproduction, not as a selectable mode.
  const representation = "curve-splat-chain";
  const detailPhysicalSplatCount = reference.strokePlan.length * splatsPerChain;
  const displayedSplatCount = underpaintSplatBudget + detailPhysicalSplatCount;
  const primitiveLabel = "splats";
  const chainQuadBackward = flowQaParams.get("chain-quad-backward") !== "0";
  const strokeOptimizationProfiles = {
    stable: {
      geometryAnchor: 0.00035,
      positionLearningRate: 0.015,
      strokeMotionCoherence: 0,
    },
    balanced: {
      geometryAnchor: 0.0000001,
      positionLearningRate: 0.07,
      strokeMotionCoherence: 0.5,
    },
    free: {
      geometryAnchor: 0,
      positionLearningRate: 0.08,
      strokeMotionCoherence: 0.75,
    },
  };
  const strokeOptimization = Object.hasOwn(
    strokeOptimizationProfiles,
    trainingUiAdapter.controls.flowSplatFusionStrokeOptimization.value,
  ) ? trainingUiAdapter.controls.flowSplatFusionStrokeOptimization.value : "balanced";
  trainingUiAdapter.controls.flowSplatFusionStrokeOptimization.value = strokeOptimization;
  const strokeOptimizationProfile = strokeOptimizationProfiles[strokeOptimization];
  const readFlowQaNumber = (name, fallback, minimum, maximum) => {
    const value = Number(flowQaParams.get(name));
    return Number.isFinite(value) && flowQaParams.has(name)
      ? Math.max(minimum, Math.min(maximum, value))
      : fallback;
  };
  const flowGeometryAnchor = readFlowQaNumber(
    "flow-geometry-anchor", strokeOptimizationProfile.geometryAnchor, 0, 0.01,
  );
  const flowPositionLearningRate = readFlowQaNumber(
    "flow-position-lr", strokeOptimizationProfile.positionLearningRate, 0, 0.5,
  );
  const uiMovementLimit = Math.max(0, Math.min(64, Number(
    trainingUiAdapter.controls.flowSplatFusionMovementLimit.value,
  ) || 0));
  const flowMaxPositionDelta = readFlowQaNumber(
    "flow-max-position-delta", uiMovementLimit, 0, 64,
  );
  trainingUiAdapter.controls.flowSplatFusionMovementLimit.value = String(flowMaxPositionDelta);
  const flowStrokeMotionCoherence = readFlowQaNumber(
    "flow-motion-coherence", strokeOptimizationProfile.strokeMotionCoherence, 0, 1,
  );
  const cumulativeLayerParents = [];
  let cumulativeParents = 0;
  for (const layer of reference.metadata.layers) {
    cumulativeParents += Math.max(0, Math.round(Number(layer.accepted) || 0));
    if (cumulativeParents > 0) cumulativeLayerParents.push(cumulativeParents);
  }
  if (cumulativeLayerParents.at(-1) !== reference.strokePlan.length) {
    cumulativeLayerParents.push(reference.strokePlan.length);
  }
  const broadParentCount = cumulativeLayerParents[0] || reference.strokePlan.length;
  const growthSchedule = buildFlowProgressiveGrowthSchedule(
    iterations,
    broadParentCount,
    reference.strokePlan.length,
  );
  const progressiveParentCounts = growthSchedule.parentCounts;
  const growthIterations = growthSchedule.iterationCounts;
  const flowTopologyOptions = {
    minimumWidthFactor: Math.min(
      initialWidthMinimumPercent / 100,
      strokeWidthMaximumFactor,
    ),
    maximumWidthFactor: Math.min(
      initialWidthMaximumPercent / 100,
      strokeWidthMaximumFactor,
    ),
    frontWidthMaximumFactor: Math.min(
      frontWidthMaximumPercent / 100,
      strokeWidthMaximumFactor,
    ),
    fixedOpacity: fixedStrokeOpacity,
    maximumWidthPx: Math.max(2, Math.min(width, height) * 0.09),
    maximumCurveArcPx: reference.metadata.maximum_ribbon_arc_px,
    imageLongSide: Math.max(width, height),
    paintCurriculumEnabled: flowPaintCurriculum,
    startingWidthDivisor,
    startingLengthPercent,
    residualMovePerEventPx: residualMovePerStagePx,
    splitFraction: 0.04,
    maximumSplitsPerEvent: 24,
    splitApplyUntil: FLOW_SPLIT_APPLY_UNTIL,
    scaleMatchedResidualRepaint,
    maximumTotalMovementPx: flowMaxPositionDelta,
    textureGuide: reference.textureGuide,
    textureGuidedAllocation: flowTextureGuidedDabs,
    curriculumProgress: 0,
  };
  let flowTopologyState = Image2SplatPaintFlowStrokeTopology.initialize(
    reference.strokePlan,
    progressiveParentCounts[0],
    flowTopologyOptions,
  );
  const initialDisplayedSplatCount = progressiveParentCounts[0] * splatsPerChain
    + (progressiveParentCounts.length === 1 ? underpaintSplatBudget : 0);

  state.running = true;
  state.paused = false;
  state.stopRequested = false;
  state.previewGeneration += 1;
  state.flowSplatResult = null;
  state.params = null;
  state.metrics = {
    algorithm: algorithm.id,
    algorithm_label: algorithm.label,
    backend: algorithm.backend,
    initialization: "three-layer flow-field Gaussian Splat chains",
    steps_requested: iterations,
    steps_done: 0,
    num_gaussians: initialDisplayedSplatCount,
    final_splats: displayedSplatCount,
    curve_count: reference.strokePlan.length,
    underpaint_parent_count: underpaintSplatBudget,
    underpaint_splat_count: underpaintSplatBudget,
    detail_splat_count: detailPhysicalSplatCount,
    splat_underpainting: splatUnderpainting,
    splat_underpainting_percent: underpaintPercent,
    coverage_backcoat: splatUnderpainting,
    coverage_backcoat_percent: underpaintPercent,
    coverage_backcoat_kernel: splatUnderpainting
      ? "compact-quartic-flat-interior-v1"
      : "disabled",
    coverage_backcoat_geometry_trainable: false,
    coverage_backcoat_pigment_trainable: false,
    preview_refresh: previewRefresh,
    fused_micro_splats_per_ribbon: 0,
    micro_splats_per_chain: splatsPerChain,
    representation,
    flow_layer_count: splatUnderpainting ? 4 : 3,
    underpaint_mode: splatUnderpainting
      ? "fixed-grid-source-colored-compact-brush-backcoat"
      : "disabled",
    maximum_ribbon_arc_percent: maximumRibbonArcPercent,
    capped_ribbon_count: reference.metadata.capped_ribbon_count,
    maximum_ribbon_arc_px: reference.metadata.maximum_ribbon_arc_px,
    maximum_initial_ribbon_arc_px: reference.metadata.maximum_final_ribbon_arc_px,
    chain_quad_backward: chainQuadBackward,
    flow_geometry_anchor: flowGeometryAnchor,
    flow_stroke_optimization: strokeOptimization,
    flow_stroke_width_percent: strokeWidthPercent,
    flow_stroke_width_mode: "global-parent-width-ceiling",
    flow_splat_size_variation_percent: splatSizeVariationPercent,
    flow_splat_size_variation_mode: "flow-xdog-thin-bristle-moderate-body-scale-families",
    flow_position_learning_rate: flowPositionLearningRate,
    flow_max_position_delta_px: flowMaxPositionDelta,
    flow_stroke_motion_coherence: flowStrokeMotionCoherence,
    flow_stroke_texture: flowStrokeTexture,
    flow_bristle_bundle: flowBristleBundle,
    flow_brush_dabs: flowBrushDabs,
    flow_brush_kernel: flowBrushDabs ? "compact-quartic-opaque-interior-v1" : "gaussian",
    flow_texture_guided_dabs: flowTextureGuidedDabs,
    flow_texture_guide_summary: reference.metadata.texture_guide,
    flow_initial_mean_parent_texture_score: reference.metadata.mean_parent_texture_score,
    flow_initial_mean_parent_edge_score: reference.metadata.mean_parent_edge_score,
    flow_initial_dark_flat_parent_fraction: reference.metadata.dark_flat_parent_fraction,
    flow_topology_mode: flowTopologyMode,
    flow_adaptive_topology: true,
    flow_paint_curriculum: flowPaintCurriculum,
    flow_fixed_stroke_opacity: fixedStrokeOpacity,
    flow_opacity_trainable: false,
    flow_front_width_maximum_percent: frontWidthMaximumPercent,
    flow_front_width_learning_percent: frontWidthLearningPercent,
    flow_front_width_learning_scale: frontWidthLearningScale,
    flow_front_width_carries_across_growth: true,
    flow_starting_width_divisor: startingWidthDivisor,
    flow_starting_full_width_px: Math.max(width, height) / startingWidthDivisor,
    flow_starting_length_percent: startingLengthPercent,
    flow_residual_move_per_stage_px: residualMovePerStagePx,
    flow_scale_matched_residual_repaint: scaleMatchedResidualRepaint,
    flow_initial_width_percent_range: [initialWidthMinimumPercent, initialWidthMaximumPercent],
    flow_topology_split_count: 0,
    flow_topology_split_fraction: flowTopologyOptions.splitFraction,
    flow_topology_maximum_splits_per_event: flowTopologyOptions.maximumSplitsPerEvent,
    flow_topology_split_apply_until: flowTopologyOptions.splitApplyUntil,
    flow_topology_merge_count: 0,
    flow_topology_residual_move_count: 0,
    flow_topology_source_added_count: flowTopologyState?.totals.sourceAdded || 0,
    flow_topology_events: [],
    flow_residual_priority_tile_sampling: flowResidualPriorityTileSampling,
    flow_tile_list_update: "growth-boundary-only",
    progressive_splat_growth: progressiveParentCounts.length > 1,
    progressive_growth_interval: FLOW_PROGRESSIVE_GROWTH_INTERVAL,
    progressive_growth_apply_until: FLOW_PROGRESSIVE_GROWTH_APPLY_UNTIL,
    progressive_growth_stage_count: progressiveParentCounts.length,
    progressive_growth_iteration_budget: growthSchedule.growthIterationBudget,
    progressive_settle_iterations: growthSchedule.settleIterations,
    progressive_growth_parent_counts: progressiveParentCounts.slice(),
    progressive_growth_iterations: growthIterations.slice(),
  };
  document.documentElement.dataset.flowMaximumRibbonArcPercent = String(maximumRibbonArcPercent);
  document.documentElement.dataset.flowStrokeWidthPercent = String(strokeWidthPercent);
  document.documentElement.dataset.flowSplatSizeVariationPercent = String(
    splatSizeVariationPercent,
  );
  document.documentElement.dataset.flowMovementLimitPx = String(flowMaxPositionDelta);
  document.documentElement.dataset.flowStrokeTexture = flowStrokeTexture;
  document.documentElement.dataset.flowBristleBundle = String(flowBristleBundle);
  document.documentElement.dataset.flowBrushDabs = String(flowBrushDabs);
  document.documentElement.dataset.flowTextureGuidedDabs = String(flowTextureGuidedDabs);
  document.documentElement.dataset.flowTextureGuideSummary = JSON.stringify(
    reference.metadata.texture_guide || null,
  );
  document.documentElement.dataset.flowTopologyMode = flowTopologyMode;
  document.documentElement.dataset.flowPaintCurriculum = String(flowPaintCurriculum);
  document.documentElement.dataset.flowFixedStrokeOpacity = String(fixedStrokeOpacity);
  document.documentElement.dataset.flowFrontWidthMaximumPercent = String(
    frontWidthMaximumPercent,
  );
  document.documentElement.dataset.flowFrontWidthLearningPercent = String(
    frontWidthLearningPercent,
  );
  document.documentElement.dataset.flowFrontWidthLearningScale = String(
    frontWidthLearningScale,
  );
  document.documentElement.dataset.flowStartingFullWidthPx = String(
    Math.max(width, height) / startingWidthDivisor,
  );
  document.documentElement.dataset.flowStartingLengthPercent = String(startingLengthPercent);
  document.documentElement.dataset.flowResidualMovePerStagePx = String(residualMovePerStagePx);
  document.documentElement.dataset.flowScaleMatchedResidualRepaint = String(
    scaleMatchedResidualRepaint,
  );
  document.documentElement.dataset.flowTopologySplitCount = "0";
  document.documentElement.dataset.flowTopologyMergeCount = "0";
  document.documentElement.dataset.flowTopologyResidualMoveCount = "0";
  document.documentElement.dataset.flowResidualPriorityTileSampling = String(
    flowResidualPriorityTileSampling,
  );
  document.documentElement.dataset.flowMaximumInitialRibbonArcPx = String(
    reference.metadata.maximum_final_ribbon_arc_px,
  );
  document.documentElement.dataset.flowCappedRibbonCount = String(
    reference.metadata.capped_ribbon_count,
  );
  document.documentElement.dataset.flowBackwardMode = chainQuadBackward
    ? "16x16-tile-2x2-lane-quad-parent-curve-reduction"
    : "8x8-per-pixel-global-atomic";
  state.previewMode = previewRefresh === "final" ? "original" : "splats";
  updateTrainingRunOwnership(run, { image: sourceImage, metrics: state.metrics });
  resetTrainingTiming(false);
  clearSplatAdjustmentBaseline();
  updateDownloads(false);
  setStatus("running");
  trainingUiAdapter.controls.startButton.disabled = true;
  trainingUiAdapter.controls.stopButton.disabled = false;
  setInputControlsDisabled(true);
  setPausedRuntimeControlsEnabled(false);
  setTrainingMessage(`Training ${algorithm.label} on WebGPU...`);
  trainingUiAdapter.controls.stepText.textContent = `0 / ${iterations}`;
  trainingUiAdapter.controls.splatText.textContent =
    `${initialDisplayedSplatCount.toLocaleString()} / ${displayedSplatCount.toLocaleString()}`;
  trainingUiAdapter.controls.lossText.textContent = "-";
  trainingUiAdapter.controls.psnrText.textContent = "-";
  trainingUiAdapter.controls.ssimText.textContent = "-";
  trainingUiAdapter.controls.regionalSsimText.textContent = "-";
  trainingUiAdapter.controls.coverageText.textContent = "- / -";
  let lastProgressElapsedMs = 0;
  let trainingError = null;
  const overallStartedAt = performance.now();
  const waitWhilePaused = async () => {
    while (state.paused && !state.stopRequested) {
      await new Promise((resolve) => window.setTimeout(resolve, 50));
      assertTrainingRun(run);
    }
  };
  const callbacksForStage = (iterationOffset, strokeCount, stageSplatCount) => ({
    shouldStop: () => state.stopRequested || !ownsTrainingRun(run),
    waitWhilePaused,
    onProgress(progress) {
      assertTrainingRun(run);
      const globalIteration = iterationOffset + progress.iteration;
      const elapsedMs = performance.now() - overallStartedAt;
      const elapsedDelta = Math.max(0, elapsedMs - lastProgressElapsedMs);
      lastProgressElapsedMs = elapsedMs;
      state.metrics.steps_done = globalIteration;
      state.metrics.num_gaussians = stageSplatCount;
      recordTrainingTiming(globalIteration, elapsedDelta);
      trainingUiAdapter.controls.stepText.textContent = `${globalIteration} / ${iterations}`;
      trainingUiAdapter.controls.splatText.textContent =
        `${stageSplatCount.toLocaleString()} / ${displayedSplatCount.toLocaleString()}`;
      setStatus(state.paused ? "paused" : state.stopRequested ? "stopping" : "running");
      publishState();
    },
    onPreview(preview) {
      assertTrainingRun(run);
      const globalIteration = iterationOffset + preview.iteration;
      state.flowSplatResult = {
        sourceImage,
        image: preview.image,
        metadata: {
          partial: true,
          iteration: globalIteration,
          requested_iterations: iterations,
          stroke_count: strokeCount,
          splat_count: stageSplatCount,
          representation,
          rgb_l1_signal: preview.rgb_l1_signal,
          psnr_signal_db: preview.psnr_signal_db,
          coverage_stats: preview.coverage_stats,
        },
      };
      state.metrics.rgb_l1 = preview.rgb_l1_signal;
      state.metrics.psnr_rgb = preview.psnr_signal_db;
      state.metrics.coverage_stats = {
        ...preview.coverage_stats,
        step: globalIteration,
      };
      trainingUiAdapter.controls.lossText.textContent = preview.rgb_l1_signal.toFixed(6);
      trainingUiAdapter.controls.psnrText.textContent = `${preview.psnr_signal_db.toFixed(2)} dB`;
      trainingUiAdapter.controls.coverageText.textContent =
        `${preview.coverage_stats.background_exposure_count.toLocaleString()} / 0`;
      if (state.previewMode === "splats") presentFlowSplatFusionResult(state.flowSplatResult);
    },
  });
  const trainerOptions = {
    colorAnchor,
    geometryAnchor: flowGeometryAnchor,
    positionLearningRate: flowPositionLearningRate,
    maxPositionDelta: flowMaxPositionDelta,
    strokeMotionCoherence: flowStrokeMotionCoherence,
    bristleBundle: flowBristleBundle,
    brushDabs: flowBrushDabs,
    textureGuidedDabs: flowTextureGuidedDabs,
    splatSizeVariation: splatSizeVariationPercent / 100,
    widthAnchor: 0.0008,
    fixedOpacity: fixedStrokeOpacity,
    frontWidthLearningScale,
    representation,
    quadBackward: chainQuadBackward,
    maximumCurveArcPx: reference.metadata.maximum_ribbon_arc_px,
    progressInterval: previewInterval || 20,
    previewInterval,
  };

  try {
    let previousStage = null;
    let result = null;
    let completedIterations = 0;
    let elapsedMs = 0;
    let completedStages = 0;
    let finalDetailPlan = flowTopologyState.plan;
    let finalUnderpaint = Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan(
      trainingImage,
      { count: 0, representation, seed: 240825 },
    );
    for (let stage = 0; stage < progressiveParentCounts.length; stage += 1) {
      const finalStage = stage === progressiveParentCounts.length - 1;
      const detailParentCount = progressiveParentCounts[stage];
      const curriculumProgress = finalStage
        ? 1
        : Math.max(0, Math.min(1, completedIterations / Math.max(1, growthSchedule.growthIterationBudget)));
      if (stage > 0) {
        flowTopologyState = Image2SplatPaintFlowStrokeTopology.evolve(
          flowTopologyState,
          previousStage?.trainingState.params,
          trainingImage,
          previousStage?.trainingState.renderedLinearRgba,
          detailParentCount,
          reference.strokePlan,
          { ...flowTopologyOptions, curriculumProgress },
        );
        state.metrics.flow_topology_split_count = flowTopologyState.totals.splits;
        state.metrics.flow_topology_merge_count = flowTopologyState.totals.merges;
        state.metrics.flow_topology_source_added_count = flowTopologyState.totals.sourceAdded;
        state.metrics.flow_topology_residual_move_count = flowTopologyState.totals.residualMoves;
        state.metrics.flow_topology_events = flowTopologyState.events.slice();
        document.documentElement.dataset.flowTopologySplitCount = String(
          flowTopologyState.totals.splits,
        );
        document.documentElement.dataset.flowTopologyMergeCount = String(
          flowTopologyState.totals.merges,
        );
        document.documentElement.dataset.flowTopologyResidualMoveCount = String(
          flowTopologyState.totals.residualMoves,
        );
      }
      const detailPlan = flowTopologyState.plan;
      finalDetailPlan = detailPlan;
      if (finalStage && splatUnderpainting) {
        finalUnderpaint = Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan(
          trainingImage,
          {
            count: underpaintSplatBudget,
            representation,
            seed: 240825,
            residualRender: previousStage?.trainingState.renderedLinearRgba,
          },
        );
      }
      const rearPlan = finalStage ? finalUnderpaint.strokePlan : [];
      const trainingStrokePlan = [...rearPlan, ...detailPlan];
      const stagePhysicalSplatCount = rearPlan.length + detailParentCount * splatsPerChain;
      const currentCurriculum = flowTopologyState.events.at(-1) || {
        curriculum_mean_half_width_px: flowTopologyState.initialCurriculum.meanHalfWidthPx,
      };
      setTrainingMessage(
        `Training ${algorithm.label}: growth ${stage + 1} / ${progressiveParentCounts.length}, ` +
        `${stagePhysicalSplatCount.toLocaleString()} Splats` +
        `, split ${flowTopologyState.totals.splits} / merge ${flowTopologyState.totals.merges}` +
        ` / move ${flowTopologyState.totals.residualMoves}, ` +
        `width ${(currentCurriculum.curriculum_mean_half_width_px * 2).toFixed(1)}px...`,
      );
      const stagePositionLearningRate = flowPositionLearningRate * (
        1 + 0.5 * (1 - curriculumProgress) ** 0.8
      );
      const stageMovementLimit = flowMaxPositionDelta > 0
        ? Math.min(flowMaxPositionDelta, Math.max(0.5, residualMovePerStagePx * 2))
        : 0;
      const stageResult = await Image2SplatPaintFlowRibbonTrainer.train(
        trainingImage,
        trainingStrokePlan,
        {
          ...trainerOptions,
          positionLearningRate: stagePositionLearningRate,
          maxPositionDelta: stageMovementLimit,
          globalIterationOffset: completedIterations,
          globalIterations: iterations,
          iterations: growthIterations[stage],
          maxStrokes: trainingStrokePlan.length,
          initialDetailParams: undefined,
          initialDetailOffset: rearPlan.length,
          detailGeometryAnchorParams: createFlowGeometryAnchorParams(detailPlan),
          residualPriorityTileSampling: flowResidualPriorityTileSampling && stage > 0,
          residualRender: previousStage?.trainingState.renderedLinearRgba,
          returnTrainingState: true,
          ...callbacksForStage(
            completedIterations,
            trainingStrokePlan.length,
            stagePhysicalSplatCount,
          ),
        },
      );
      result = stageResult;
      previousStage = stageResult;
      completedIterations += stageResult.metadata.iterations;
      elapsedMs += stageResult.metadata.elapsed_ms;
      completedStages += 1;
      assertTrainingRun(run);
      if (stageResult.metadata.stopped) break;
    }
    const finalStageControlPointRmsDrift = result.metadata.control_point_rms_drift_px;
    const fullRunControlPointRmsDrift = measureFlowControlPointDrift(
      finalDetailPlan,
      result.trainingState.params,
      finalUnderpaint.strokePlan.length,
    );
    delete result.trainingState;
    result.metadata.iterations = completedIterations;
    result.metadata.requested_iterations = iterations;
    result.metadata.coverage_stats.step = completedIterations;
    result.metadata.elapsed_ms = elapsedMs;
    result.metadata.iterations_per_second = completedIterations
      / Math.max(1e-6, elapsedMs / 1000);
    result.metadata.flow_layer_count = finalUnderpaint.strokePlan.length > 0 ? 4 : 3;
    result.metadata.underpaint_mode = finalUnderpaint.strokePlan.length > 0
      ? "fixed-grid-source-colored-compact-brush-backcoat"
      : "disabled";
    result.metadata.coverage_backcoat = finalUnderpaint.strokePlan.length > 0;
    result.metadata.coverage_backcoat_percent = underpaintPercent;
    result.metadata.coverage_backcoat_kernel = finalUnderpaint.metadata.coverage_kernel
      || "disabled";
    result.metadata.coverage_backcoat_geometry_trainable = false;
    result.metadata.coverage_backcoat_pigment_trainable = false;
    result.metadata.coverage_backcoat_guarantee = finalUnderpaint.metadata.coverage_guarantee
      || "disabled";
    result.metadata.residual_render_used = finalUnderpaint.metadata.residual_render_used;
    result.metadata.progressive_splat_growth = progressiveParentCounts.length > 1;
    result.metadata.progressive_growth_interval = FLOW_PROGRESSIVE_GROWTH_INTERVAL;
    result.metadata.progressive_growth_apply_until = FLOW_PROGRESSIVE_GROWTH_APPLY_UNTIL;
    result.metadata.progressive_growth_stage_count = progressiveParentCounts.length;
    result.metadata.progressive_growth_iteration_budget = growthSchedule.growthIterationBudget;
    result.metadata.progressive_settle_iterations = growthSchedule.settleIterations;
    result.metadata.progressive_growth_parent_counts = progressiveParentCounts.slice();
    result.metadata.progressive_growth_iterations = growthIterations.slice();
    result.metadata.progressive_growth_completed_stages = completedStages;
    result.metadata.flow_topology_mode = flowTopologyMode;
    result.metadata.flow_adaptive_topology = true;
    result.metadata.flow_topology_split_fraction = flowTopologyOptions.splitFraction;
    result.metadata.flow_topology_maximum_splits_per_event =
      flowTopologyOptions.maximumSplitsPerEvent;
    result.metadata.flow_topology_split_apply_until = flowTopologyOptions.splitApplyUntil;
    result.metadata.flow_paint_curriculum = flowPaintCurriculum;
    result.metadata.flow_stroke_texture = flowStrokeTexture;
    result.metadata.flow_bristle_bundle = flowBristleBundle;
    result.metadata.flow_brush_dabs = flowBrushDabs;
    result.metadata.flow_splat_size_variation_percent = splatSizeVariationPercent;
    result.metadata.flow_splat_size_variation_mode =
      "flow-xdog-thin-bristle-moderate-body-scale-families";
    result.metadata.flow_texture_guided_dabs = flowTextureGuidedDabs;
    result.metadata.flow_texture_guide_summary = reference.metadata.texture_guide;
    result.metadata.flow_initial_mean_parent_texture_score =
      reference.metadata.mean_parent_texture_score;
    result.metadata.flow_initial_mean_parent_edge_score =
      reference.metadata.mean_parent_edge_score;
    result.metadata.flow_initial_dark_flat_parent_fraction =
      reference.metadata.dark_flat_parent_fraction;
    result.metadata.flow_fixed_stroke_opacity = fixedStrokeOpacity;
    result.metadata.flow_opacity_trainable = false;
    result.metadata.flow_front_width_maximum_percent = frontWidthMaximumPercent;
    result.metadata.flow_front_width_learning_percent = frontWidthLearningPercent;
    result.metadata.flow_front_width_learning_scale = frontWidthLearningScale;
    result.metadata.flow_front_width_carries_across_growth = true;
    result.metadata.flow_starting_width_divisor = startingWidthDivisor;
    result.metadata.flow_starting_full_width_px = Math.max(width, height) / startingWidthDivisor;
    result.metadata.flow_starting_length_percent = startingLengthPercent;
    result.metadata.flow_residual_move_per_stage_px = residualMovePerStagePx;
    result.metadata.flow_scale_matched_residual_repaint = scaleMatchedResidualRepaint;
    result.metadata.flow_initial_width_percent_range = [
      initialWidthMinimumPercent,
      initialWidthMaximumPercent,
    ];
    result.metadata.flow_topology_split_count = flowTopologyState?.totals.splits || 0;
    result.metadata.flow_topology_merge_count = flowTopologyState?.totals.merges || 0;
    result.metadata.flow_topology_source_added_count = flowTopologyState?.totals.sourceAdded || 0;
    result.metadata.flow_topology_residual_move_count = flowTopologyState?.totals.residualMoves || 0;
    result.metadata.flow_topology_initial_distribution = flowTopologyState?.initialDistribution || null;
    result.metadata.flow_topology_events = flowTopologyState?.events.slice() || [];
    result.metadata.flow_topology_final_distribution = flowTopologyState?.events.at(-1)?.distribution
      || flowTopologyState?.initialDistribution
      || null;
    result.metadata.flow_residual_priority_tile_sampling = flowResidualPriorityTileSampling;
    result.metadata.flow_tile_list_update = "growth-boundary-only";
    result.metadata.flow_overall_elapsed_ms = performance.now() - overallStartedAt;
    result.metadata.final_stage_control_point_rms_drift_px = finalStageControlPointRmsDrift;
    result.metadata.control_point_rms_drift_px = fullRunControlPointRmsDrift;
    const coverageStats = result.metadata.coverage_stats || {};
    document.documentElement.dataset.flowControlPointRmsDriftPx = String(
      fullRunControlPointRmsDrift,
    );
    document.documentElement.dataset.flowFinalStageControlPointRmsDriftPx = String(
      finalStageControlPointRmsDrift,
    );
    document.documentElement.dataset.flowTrainingElapsedMs = String(result.metadata.elapsed_ms);
    document.documentElement.dataset.flowOverallElapsedMs = String(
      result.metadata.flow_overall_elapsed_ms,
    );
    document.documentElement.dataset.flowIterationsPerSecond = String(
      result.metadata.iterations_per_second,
    );
    document.documentElement.dataset.flowTopologyEventCount = String(
      flowTopologyState?.events.length || 0,
    );
    document.documentElement.dataset.flowTopologyResidualMoveCount = String(
      flowTopologyState?.totals.residualMoves || 0,
    );
    document.documentElement.dataset.flowTopologyInitialDistribution = JSON.stringify(
      result.metadata.flow_topology_initial_distribution,
    );
    document.documentElement.dataset.flowTopologyFinalDistribution = JSON.stringify(
      result.metadata.flow_topology_final_distribution,
    );
    document.documentElement.dataset.flowFinalMeanParentTextureScore = String(
      result.metadata.flow_topology_final_distribution?.texture_score?.mean || 0,
    );
    document.documentElement.dataset.flowFinalMeanParentEdgeScore = String(
      result.metadata.flow_topology_final_distribution?.edge_score?.mean || 0,
    );
    document.documentElement.dataset.flowFinalDarkFlatParentFraction = String(
      result.metadata.flow_topology_final_distribution?.dark_flat_parent_fraction || 0,
    );
    document.documentElement.dataset.flowBackgroundExposureRatio = String(
      Number(coverageStats.background_exposure_ratio) || 0,
    );
    document.documentElement.dataset.flowMeanTransmittance = String(
      Number(coverageStats.mean_transmittance) || 0,
    );
    document.documentElement.dataset.flowCanvasLeakLinearMean = String(
      Number(coverageStats.cream_canvas_leak_linear_mean) || 0,
    );
    document.documentElement.dataset.flowCanvasSignalSrgb = JSON.stringify(
      coverageStats.canvas_signal_srgb || [],
    );
    assertTrainingRun(run);
    state.flowSplatResult = { sourceImage, ...result };
    state.metrics = {
      ...state.metrics,
      ...result.metadata,
      algorithm: algorithm.id,
      algorithm_label: algorithm.label,
      steps_done: result.metadata.iterations,
      steps_requested: result.metadata.requested_iterations,
      num_gaussians: result.metadata.splat_count,
      final_splats: result.metadata.splat_count,
      curve_count: result.metadata.detail_parent_count,
      rgb_l1: result.metadata.rgb_l1_signal,
      psnr_rgb: result.metadata.psnr_signal_db,
      final_cpu_result_ready_at: performance.now(),
      coverage_stats: result.metadata.coverage_stats,
      render_surface_parity: {
        max_abs: 0,
        mean_abs: 0,
        alpha_max_abs: 0,
        premultiplied_max_abs: 0,
        display_equivalent: true,
      },
    };
    document.documentElement.dataset.flowMaximumFinalRibbonArcPx = String(
      result.metadata.maximum_final_curve_arc_px,
    );
    document.documentElement.dataset.flowTrainingIterationsPerSecond = String(
      result.metadata.iterations_per_second,
    );
    document.documentElement.dataset.flowFinalParameterHash = String(
      result.metadata.final_parameter_hash_fnv1a32,
    );
    document.documentElement.dataset.flowControlPointRmsDriftPx = String(
      result.metadata.control_point_rms_drift_px,
    );
    document.documentElement.dataset.flowMeanWidthDriftPx = String(
      result.metadata.mean_width_drift_px,
    );
    document.documentElement.dataset.flowMeanOpacityDrift = String(
      result.metadata.mean_opacity_drift,
    );
    document.documentElement.dataset.flowLayerParameterStats = JSON.stringify(
      result.metadata.layer_parameter_stats,
    );
    document.documentElement.dataset.flowMeanColorDrift = String(
      result.metadata.mean_color_anchor_drift_linear,
    );
    document.documentElement.dataset.flowUnderpaintSplatCount = String(
      result.metadata.underpaint_splat_count,
    );
    document.documentElement.dataset.flowCanvasExposureRatio = String(
      result.metadata.coverage_stats.background_exposure_ratio,
    );
    document.documentElement.dataset.flowCanvasLeakLinearMean = String(
      result.metadata.coverage_stats.cream_canvas_leak_linear_mean,
    );
    document.documentElement.dataset.flowMaximumTransmittance = String(
      result.metadata.coverage_stats.maximum_transmittance,
    );
    document.documentElement.dataset.flowMinimumCompositeAlpha = String(
      result.metadata.coverage_stats.minimum_composite_alpha,
    );
    updateTrainingRunOwnership(run, { metrics: state.metrics });
    trainingUiAdapter.controls.stepText.textContent = `${result.metadata.iterations} / ${result.metadata.requested_iterations}`;
    trainingUiAdapter.controls.lossText.textContent = result.metadata.rgb_l1_signal.toFixed(6);
    trainingUiAdapter.controls.psnrText.textContent = `${result.metadata.psnr_signal_db.toFixed(2)} dB`;
    trainingUiAdapter.controls.coverageText.textContent =
      `${result.metadata.coverage_stats.background_exposure_count.toLocaleString()} / 0`;
    if (state.previewMode === "splats") presentFlowSplatFusionResult(state.flowSplatResult);
    setStatus(result.metadata.stopped ? "stopped" : "done");
    setTrainingMessage(
      result.metadata.stopped
        ? `${algorithm.label} stopped after ${result.metadata.iterations} / ${result.metadata.requested_iterations} iterations.`
        : `${algorithm.label} finished: ${result.metadata.splat_count.toLocaleString()} ${primitiveLabel}.`,
      "success",
    );
    eventLog(
      `${algorithm.label} ${result.metadata.stopped ? "stopped" : "done"}` +
      ` step=${result.metadata.iterations}/${result.metadata.requested_iterations}` +
      ` curves=${result.metadata.stroke_count}` +
      ` splats=${result.metadata.splat_count}` +
      ` psnr=${result.metadata.psnr_signal_db.toFixed(2)}`,
    );
    return result;
  } catch (error) {
    trainingError = error;
    throw error;
  } finally {
    if (ownsTrainingRun(run)) {
      state.running = false;
      state.paused = false;
      state.stopRequested = false;
      trainingUiAdapter.controls.startButton.disabled = false;
      trainingUiAdapter.controls.pauseButton.disabled = true;
      trainingUiAdapter.controls.pauseButton.textContent = "Pause";
      trainingUiAdapter.controls.stopButton.disabled = true;
      setPausedRuntimeControlsEnabled(false);
      setInputControlsDisabled(false);
      updateDownloads(!trainingError && Boolean(state.flowSplatResult));
      publishState();
    }
  }
}
