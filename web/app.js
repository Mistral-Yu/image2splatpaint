const {
  clampNumber,
  normalizeStepInteger,
  hexColorToRgb,
  limitNumber,
} = globalThis.Image2SplatPaintNumeric;
const { canvasToBlob } = globalThis.Image2SplatPaintCanvasBlob;
const {
  createPlyHeader,
  plyFrameScale: exportedPlyFrameScale,
  serializeBinaryPly,
  transformPlanarSplat,
} = globalThis.Image2SplatPaintPlySerializer;
const { inspectPlyContract: inspectSerializedPlyContract } = globalThis.Image2SplatPaintPlyInspector;
const {
  displayOrientedImageSize,
  orientationSwapsImageAxes,
  parseImageDimensions,
} = globalThis.Image2SplatPaintImageMetadata;
const { profileDistributionSummary } = globalThis.Image2SplatPaintGpuMetrics;
const { requestWebGpuDevice } = globalThis.Image2SplatPaintGpuDevice;

const APP_SCRIPT_URL = document.currentScript?.src || new URL("./app.js", location.href).href;
const SAMPLE_IMAGE_URL = new URL("../assets/source-images/generated-geometric-sample.jpg", APP_SCRIPT_URL).href;
const EMBEDDED_SAMPLE_IMAGE_URL = globalThis.__IMAGE2SPLATPAINT_SAMPLE_JPEG || "";
const ALGORITHM_REGISTRY = Object.freeze({
  [PLANAR_GAUSSIAN_ALGORITHM_ID]: Object.freeze({
    id: PLANAR_GAUSSIAN_ALGORITHM_ID,
    label: "Planar Gaussian",
    backend: "custom-webgpu",
    initialize: initGaussians,
    train: trainPlanarGaussian,
    exports: Object.freeze(["png", "ply"]),
    capabilities: Object.freeze({ render: true, metrics: true, png: true, ply: true, tilt: false, virtualCameras: false }),
  }),
  [RECTANGLE_SPLATS_ALGORITHM_ID]: Object.freeze({
    id: RECTANGLE_SPLATS_ALGORITHM_ID,
    label: "Rectangle Splats",
    backend: "custom-webgpu",
    initialize: initRectangles,
    train: trainRectangleSplats,
    exports: Object.freeze(["png"]),
    capabilities: Object.freeze({
      render: true,
      metrics: true,
      png: true,
      ply: false,
      tilt: false,
      virtualCameras: false,
      kernelShape: "rectangle",
      opaqueLayeredPaint: true,
      minimumOpacity: true,
      variableTopWidth: true,
      requiresLayerOrder: true,
      configurableLayerCount: true,
      contributionCleanup: true,
    }),
  }),
  [LAYERED_OPAQUE_BRUSH_ALGORITHM_ID]: Object.freeze({
    id: LAYERED_OPAQUE_BRUSH_ALGORITHM_ID,
    label: "Brush Splats (compatibility)",
    backend: "custom-webgpu",
    initialize: initLayeredOpaqueBrush,
    train: trainLayeredOpaqueBrush,
    exports: Object.freeze(["png"]),
    capabilities: Object.freeze({
      render: true,
      metrics: true,
      png: true,
      ply: false,
      tilt: false,
      virtualCameras: false,
      kernelShape: "opaque-brush",
      opaqueLayeredPaint: true,
      minimumOpacity: true,
      requiresLayerOrder: true,
      configurableLayerCount: true,
      contributionCleanup: true,
      compatibilityAliasFor: RECTANGLE_SPLATS_ALGORITHM_ID,
    }),
  }),
  [FLOW_SPLAT_FUSION_ALGORITHM_ID]: Object.freeze({
    id: FLOW_SPLAT_FUSION_ALGORITHM_ID,
    label: "Flow Brush Fusion",
    backend: "flow-splat-chain-webgpu",
    initialize: null,
    train: trainFlowSplatFusion,
    exports: Object.freeze(["png"]),
    capabilities: Object.freeze({
      render: true,
      metrics: true,
      png: true,
      ply: false,
      tilt: false,
      virtualCameras: false,
      flowSplatFusion: true,
      curveSplatChain: true,
      microSplatsPerChain: 4,
    }),
  }),
  [CURVE_SPLAT_CHAIN_ALGORITHM_ID]: Object.freeze({
    id: CURVE_SPLAT_CHAIN_ALGORITHM_ID,
    label: "Curve Splat Chain",
    backend: "flow-splat-chain-webgpu",
    initialize: null,
    train: trainFlowSplatFusion,
    exports: Object.freeze([]),
    capabilities: Object.freeze({
      render: true,
      metrics: true,
      png: false,
      ply: false,
      tilt: false,
      virtualCameras: false,
      flowSplatFusion: true,
      curveSplatChain: true,
      microSplatsPerChain: 4,
      compatibilityAliasFor: FLOW_SPLAT_FUSION_ALGORITHM_ID,
    }),
  }),
  [GS_VIRTUAL_CAMERA_ALGORITHM_ID]: Object.freeze({
    id: GS_VIRTUAL_CAMERA_ALGORITHM_ID,
    label: "GS Virtual Camera Sampling",
    backend: "custom-webgpu",
    initialize: initGaussians,
    train: trainGsVirtualCameraSampling,
    exports: Object.freeze(["png", "ply"]),
    capabilities: Object.freeze({ render: true, metrics: true, png: true, ply: true, tilt: true, virtualCameras: true }),
  }),
});
const LIMITS = {
  trainSizeMin: 32,
  trainSizeMax: 3200,
  splatsMin: 4,
  splatsMax: MANUAL_SPLATS_MAX,
  stepsMin: 1,
  stepsMax: Number.MAX_SAFE_INTEGER,
  lrMin: 0,
  lrDefaultMax: 1,
  opacityLrMax: 1,
  scaleLrMax: 1,
  thetaAlignLrMax: 1,
  maxAnisotropyMin: 1,
  maxAnisotropyMax: 32,
  boundarySigmaMin: 0,
  boundarySigmaMax: 4,
  maxPlanarScaleMin: 0.02,
  maxPlanarScaleMax: 2,
  detailCoherenceMin: 0,
  detailCoherenceMax: 1,
};

const QA_RUNTIME_ENABLED =
  location.protocol === "file:" ||
  location.hostname === "localhost" ||
  location.hostname === "127.0.0.1" ||
  location.hostname === "[::1]";

const els = {
  appRoot: document.querySelector("main.app"),
  viewer: document.querySelector(".viewer"),
  viewControls: document.querySelector(".view-controls"),
  previewCanvas: document.querySelector("#previewCanvas"),
  gpuCanvas: document.querySelector("#gpuCanvas"),
  tiltCanvas: document.querySelector("#tiltCanvas"),
  tiltTeacherCanvas: document.querySelector("#tiltTeacherCanvas"),
  tiltFrameOverlay: document.querySelector("#tiltFrameOverlay"),
  previewImageFrame: document.querySelector("#previewImageFrame"),
  actualSizeButton: document.querySelector("#actualSizeButton"),
  fitViewButton: document.querySelector("#fitViewButton"),
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  trainSize: document.querySelector("#trainSize"),
  initialSplatCountLabel: document.querySelector("#initialSplatCountLabel"),
  initialSplatCount: document.querySelector("#initialSplatCount"),
  adaptiveGridInitializationFraction: document.querySelector("#adaptiveGridInitializationFraction"),
  finalSplatCount: document.querySelector("#finalSplatCount"),
  finalSplatCountLabel: document.querySelector("#finalSplatCountLabel"),
  capacityMode: document.querySelector("#capacityMode"),
  algorithmSelect: document.querySelector("#algorithmSelect"),
  rectanglePaintPanel: document.querySelector(".rectangle-paint-panel"),
  rectanglePaintShape: document.querySelector("#rectanglePaintShape"),
  rectangleShapeSettings: document.querySelector("#rectangleShapeSettings"),
  stepCount: document.querySelector("#stepCount"),
  previewRefresh: document.querySelector("#previewRefresh"),
  liveQualityMetrics: document.querySelector("#liveQualityMetrics"),
  memoryLimiterUnlock: document.querySelector("#memoryLimiterUnlock"),
  currentContributionCompaction: document.querySelector("#currentContributionCompaction"),
  currentContributionCompactionStart: document.querySelector("#currentContributionCompactionStart"),
  currentContributionCompactionInterval: document.querySelector("#currentContributionCompactionInterval"),
  currentContributionCompactionMaxRemoval: document.querySelector("#currentContributionCompactionMaxRemoval"),
  currentContributionCompactionNearZero: document.querySelector("#currentContributionCompactionNearZero"),
  currentContributionCompactionWindow: document.querySelector("#currentContributionCompactionWindow"),
  tileCullingToggle: document.querySelector("#tileCullingToggle"),
  trainLayerOrder: document.querySelector("#trainLayerOrder"),
  layerAwareAccumulation: document.querySelector("#layerAwareAccumulation"),
  discreteLayers: document.querySelector("#discreteLayers"),
  discreteLayerCount: document.querySelector("#discreteLayerCount"),
  discreteLayerMoveRadius: document.querySelector("#discreteLayerMoveRadius"),
  layerUpdateInterval: document.querySelector("#layerUpdateInterval"),
  phaseRelativeScaleGuard: document.querySelector("#phaseRelativeScaleGuard"),
  p1RelativeScaleFloorRatio: document.querySelector("#p1RelativeScaleFloorRatio"),
  p2RelativeScaleFloorRatio: document.querySelector("#p2RelativeScaleFloorRatio"),
  p3RelativeScaleFloorRatio: document.querySelector("#p3RelativeScaleFloorRatio"),
  positionLearningRate: document.querySelector("#positionLearningRate"),
  colorLearningRate: document.querySelector("#colorLearningRate"),
  opacityLearningRate: document.querySelector("#opacityLearningRate"),
  alphaLossWeight: document.querySelector("#alphaLossWeight"),
  opaquePaintPanel: document.querySelector("#opaquePaintPanel"),
  opaquePaintSummary: document.querySelector("#opaquePaintSummary"),
  rectangleOpacityGradientMin: document.querySelector("#rectangleOpacityGradientMin"),
  rectangleOpacityGradientMax: document.querySelector("#rectangleOpacityGradientMax"),
  rectangleCenterOpacityGradientMin: document.querySelector("#rectangleCenterOpacityGradientMin"),
  rectangleCenterOpacityGradientMax: document.querySelector("#rectangleCenterOpacityGradientMax"),
  rectangleLearnedOpacityMin: document.querySelector("#rectangleLearnedOpacityMin"),
  rectangleLearnedOpacityMax: document.querySelector("#rectangleLearnedOpacityMax"),
  layeredBrushOpacityGradientStart: document.querySelector("#layeredBrushOpacityGradientStart"),
  layeredBrushOpacityGradientEnd: document.querySelector("#layeredBrushOpacityGradientEnd"),
  layeredBrushCenterOpacityGradientMin: document.querySelector("#layeredBrushCenterOpacityGradientMin"),
  layeredBrushCenterOpacityGradientMax: document.querySelector("#layeredBrushCenterOpacityGradientMax"),
  layeredBrushLearnedOpacityMin: document.querySelector("#layeredBrushLearnedOpacityMin"),
  layeredBrushLearnedOpacityMax: document.querySelector("#layeredBrushLearnedOpacityMax"),
  layeredBrushWidthTaperStart: document.querySelector("#layeredBrushWidthTaperStart"),
  layeredBrushWidthTaperEnd: document.querySelector("#layeredBrushWidthTaperEnd"),
  layeredBrushLocalColorFlow: document.querySelector("#layeredBrushLocalColorFlow"),
  layeredBrushStrokePersistence: document.querySelector("#layeredBrushStrokePersistence"),
  layeredBrushMinAspectRatio: document.querySelector("#layeredBrushMinAspectRatio"),
  layeredBrushMaxAspectRatio: document.querySelector("#layeredBrushMaxAspectRatio"),
  layeredBrushRibbonAspectFloor: document.querySelector("#layeredBrushRibbonAspectFloor"),
  layeredBrushAccentAspectFloor: document.querySelector("#layeredBrushAccentAspectFloor"),
  monochromeUnderpainting: document.querySelector("#monochromeUnderpainting"),
  colorFinishStart: document.querySelector("#colorFinishStart"),
  rectangleTopRatio: document.querySelector("#rectangleTopRatio"),
  rectangleTopRatioMax: document.querySelector("#rectangleTopRatioMax"),
  rectangleMinAspectRatio: document.querySelector("#rectangleMinAspectRatio"),
  rectangleMaxAspectRatio: document.querySelector("#rectangleMaxAspectRatio"),
  rectangleOrientation: document.querySelector("#rectangleOrientation"),
  rectangleOrientationTolerance: document.querySelector("#rectangleOrientationTolerance"),
  rectanglePreserveArea: document.querySelector("#rectanglePreserveArea"),
  rectangleEdgeDirectedTaper: document.querySelector("#rectangleEdgeDirectedTaper"),
  rectangleStructureAwareRatio: document.querySelector("#rectangleStructureAwareRatio"),
  rectangleAsymmetricSoftness: document.querySelector("#rectangleAsymmetricSoftness"),
  scaleBiasedSurfaceLayerPrior: document.querySelector("#scaleBiasedSurfaceLayerPrior"),
  scaleBiasedSurfaceLayerPriorLayers: document.querySelector("#scaleBiasedSurfaceLayerPriorLayers"),
  scaleBiasedSurfaceLayerPriorP1Interval: document.querySelector("#scaleBiasedSurfaceLayerPriorP1Interval"),
  scaleBiasedSurfaceLayerPriorP2Interval: document.querySelector("#scaleBiasedSurfaceLayerPriorP2Interval"),
  scaleBiasedSurfaceLayerPriorP3Interval: document.querySelector("#scaleBiasedSurfaceLayerPriorP3Interval"),
  scaleBiasedSurfaceLayerPriorUntil: document.querySelector("#scaleBiasedSurfaceLayerPriorUntil"),
  virtualBoundedDepth: document.querySelector("#virtualBoundedDepth"),
  virtualGofDensity: document.querySelector("#virtualGofDensity"),
  virtualCameraShare: document.querySelector("#virtualCameraShare"),
  virtualCameraMaxAngle: document.querySelector("#virtualCameraMaxAngle"),
  virtualCameraCount: document.querySelector("#virtualCameraCount"),
  virtualCameraFov: document.querySelector("#virtualCameraFov"),
  virtualCameraCoverageEstimate: document.querySelector("#virtualCameraCoverageEstimate"),
  stageAwareGrowth: document.querySelector("#stageAwareGrowth"),
  stageGrowthP1: document.querySelector("#stageGrowthP1"),
  stageGrowthP2: document.querySelector("#stageGrowthP2"),
  stageGrowthP3: document.querySelector("#stageGrowthP3"),
  structureGuidedAllocation: document.querySelector("#structureGuidedAllocation"),
  structureRegionGrid: document.querySelector("#structureRegionGrid"),
  midTrainingOverdensityCorrection: document.querySelector("#midTrainingOverdensityCorrection"),
  overdensityCorrectionSchedule: document.querySelector("#overdensityCorrectionSchedule"),
  overdensityCorrectionInterval: document.querySelector("#overdensityCorrectionInterval"),
  overdensityDonorPercent: document.querySelector("#overdensityDonorPercent"),
  scaleLearningRate: document.querySelector("#scaleLearningRate"),
  rotationLearningRate: document.querySelector("#rotationLearningRate"),
  thetaAlignRate: document.querySelector("#thetaAlignRate"),
  maxAnisotropy: document.querySelector("#maxAnisotropy"),
  maxPlanarScale: document.querySelector("#maxPlanarScale"),
  boundarySigma: document.querySelector("#boundarySigma"),
  detailCoherence: document.querySelector("#detailCoherence"),
  densifyInterval: document.querySelector("#densifyInterval"),
  growthPercentage: document.querySelector("#growthPercentage"),
  growthApplyUntil: document.querySelector("#growthApplyUntil"),
  growthSignalThreshold: document.querySelector("#growthSignalThreshold"),
  retryWebGpuButton: document.querySelector("#retryWebGpuButton"),
  gpuLimitText: document.querySelector("#gpuLimitText"),
  memoryEstimateText: document.querySelector("#memoryEstimateText"),
  headroomText: document.querySelector("#headroomText"),
  limiterText: document.querySelector("#limiterText"),
  recommendationText: document.querySelector("#recommendationText"),
  speedEstimateText: document.querySelector("#speedEstimateText"),
  detailCapText: document.querySelector("#detailCapText"),
  capacityProbeText: document.querySelector("#capacityProbeText"),
  tileReserveText: document.querySelector("#tileReserveText"),
  measuredSpeedText: document.querySelector("#measuredSpeedText"),
  budgetNote: document.querySelector("#budgetNote"),
  pathInput: document.querySelector("#pathInput"),
  loadImageButton: document.querySelector("#loadImageButton"),
  clearImageButton: document.querySelector("#clearImageButton"),
  sampleButton: document.querySelector("#sampleButton"),
  pathButton: document.querySelector("#pathButton"),
  splatsPreviewButton: document.querySelector("#splatsPreviewButton"),
  originalPreviewButton: document.querySelector("#originalPreviewButton"),
  outsidePreviewToggle: document.querySelector("#outsidePreviewToggle"),
  startButton: document.querySelector("#startButton"),
  pauseButton: document.querySelector("#pauseButton"),
  stopButton: document.querySelector("#stopButton"),
  resetButton: document.querySelector("#resetButton"),
  flowSplatFusionPanel: document.querySelector("#flowSplatFusionPanel"),
  flowSplatFusionPanelSummary: document.querySelector("#flowSplatFusionPanelSummary"),
  flowSplatFusionPanelNote: document.querySelector("#flowSplatFusionPanelNote"),
  flowSplatFusionStrokeTexture: document.querySelector("#flowSplatFusionStrokeTexture"),
  flowSplatFusionInitialization: document.querySelector("#flowSplatFusionInitialization"),
  flowSplatFusionStrokeOptimization: document.querySelector("#flowSplatFusionStrokeOptimization"),
  flowSplatFusionPaintCurriculum: document.querySelector("#flowSplatFusionPaintCurriculum"),
  flowSplatFusionFixedOpacity: document.querySelector("#flowSplatFusionFixedOpacity"),
  flowSplatFusionStartingWidthDivisor: document.querySelector("#flowSplatFusionStartingWidthDivisor"),
  flowSplatFusionStartingLengthPercent: document.querySelector("#flowSplatFusionStartingLengthPercent"),
  flowSplatFusionResidualMovePx: document.querySelector("#flowSplatFusionResidualMovePx"),
  flowSplatFusionScaleMatchedResidualRepaint: document.querySelector("#flowSplatFusionScaleMatchedResidualRepaint"),
  flowSplatFusionInitialWidthMin: document.querySelector("#flowSplatFusionInitialWidthMin"),
  flowSplatFusionInitialWidthMax: document.querySelector("#flowSplatFusionInitialWidthMax"),
  flowSplatFusionFrontWidthMax: document.querySelector("#flowSplatFusionFrontWidthMax"),
  flowSplatFusionFrontWidthLearning: document.querySelector("#flowSplatFusionFrontWidthLearning"),
  flowSplatFusionColorAnchor: document.querySelector("#flowSplatFusionColorAnchor"),
  flowSplatFusionWidthPercent: document.querySelector("#flowSplatFusionWidthPercent"),
  flowSplatFusionSplatSizeVariation: document.querySelector("#flowSplatFusionSplatSizeVariation"),
  flowSplatFusionEdgeAccents: document.querySelector("#flowSplatFusionEdgeAccents"),
  flowSplatFusionVariableLinks: document.querySelector("#flowSplatFusionVariableLinks"),
  flowSplatFusionMovementLimit: document.querySelector("#flowSplatFusionMovementLimit"),
  flowSplatUnderpainting: document.querySelector("#flowSplatUnderpainting"),
  flowSplatBackcoatFromP1: document.querySelector("#flowSplatBackcoatFromP1"),
  flowSplatUnderpaintPercentLabel: document.querySelector("#flowSplatUnderpaintPercentLabel"),
  flowSplatUnderpaintPercent: document.querySelector("#flowSplatUnderpaintPercent"),
  flowSplatFusionMaxArcLabel: document.querySelector("#flowSplatFusionMaxArcLabel"),
  flowSplatFusionMaxArcPercent: document.querySelector("#flowSplatFusionMaxArcPercent"),
  savePngButton: document.querySelector("#savePngButton"),
  savePlyButton: document.querySelector("#savePlyButton"),
  pngExportResolution: document.querySelector("#pngExportResolution"),
  pngExportLongSide: document.querySelector("#pngExportLongSide"),
  pngExportResolutionStatus: document.querySelector("#pngExportResolutionStatus"),
  stepText: document.querySelector("#stepText"),
  splatText: document.querySelector("#splatText"),
  imageSizeText: document.querySelector("#imageSizeText"),
  lossText: document.querySelector("#lossText"),
  psnrText: document.querySelector("#psnrText"),
  ssimText: document.querySelector("#ssimText"),
  regionalSsimText: document.querySelector("#regionalSsimText"),
  coverageText: document.querySelector("#coverageText"),
  trainingTimingText: document.querySelector("#trainingTimingText"),
  gpuMemoryText: document.querySelector("#gpuMemoryText"),
  boundaryText: document.querySelector("#boundaryText"),
  backendText: document.querySelector("#backendText"),
  webGpuNotice: document.querySelector("#webGpuNotice"),
  webGpuNoticeText: document.querySelector("#webGpuNoticeText"),
  statusText: document.querySelector("#statusText"),
  trainingStatusDetails: document.querySelector("#trainingStatusDetails"),
  log: document.querySelector("#log"),
  clearLogButton: document.querySelector("#clearLogButton"),
  logTabs: document.querySelector("#logTabs"),
  trainingLogTab: document.querySelector("#trainingLogTab"),
  eventLogTab: document.querySelector("#eventLogTab"),
  splatsTab: document.querySelector("#splatsTab"),
  exportTab: document.querySelector("#exportTab"),
  tiltTab: document.querySelector("#tiltTab"),
  trainingLogPanel: document.querySelector("#trainingLogPanel"),
  eventLogPanel: document.querySelector("#eventLogPanel"),
  splatsPanel: document.querySelector("#splatsPanel"),
  exportPanel: document.querySelector("#exportPanel"),
  tiltPanel: document.querySelector("#tiltPanel"),
  tiltPitch: document.querySelector("#tiltPitch"),
  tiltPitchValue: document.querySelector("#tiltPitchValue"),
  tiltYaw: document.querySelector("#tiltYaw"),
  tiltYawValue: document.querySelector("#tiltYawValue"),
  tiltFrontButton: document.querySelector("#tiltFrontButton"),
  tiltRefreshButton: document.querySelector("#tiltRefreshButton"),
  tiltStatus: document.querySelector("#tiltStatus"),
  tiltCameraSummary: document.querySelector("#tiltCameraSummary"),
  tiltCameraMode: document.querySelector("#tiltCameraMode"),
  tiltPositionValue: document.querySelector("#tiltPositionValue"),
  tiltRadiusValue: document.querySelector("#tiltRadiusValue"),
  tiltFovValue: document.querySelector("#tiltFovValue"),
  tiltProjectionError: document.querySelector("#tiltProjectionError"),
  tiltOriginalView: document.querySelector("#tiltOriginalView"),
  tiltOverlayView: document.querySelector("#tiltOverlayView"),
  tiltSplatsView: document.querySelector("#tiltSplatsView"),
  tiltTrainingViewsButton: document.querySelector("#tiltTrainingViewsButton"),
  tiltTrainingViewsProgress: document.querySelector("#tiltTrainingViewsProgress"),
  tiltTrainingViewsSummary: document.querySelector("#tiltTrainingViewsSummary"),
  tiltContactSheet: document.querySelector("#tiltContactSheet"),
  tiltCameraMarkers: document.querySelector("#tiltCameraMarkers"),
  tiltRadiusMode: document.querySelector("#tiltRadiusMode"),
  splatsEmpty: document.querySelector("#splatsEmpty"),
  splatsContent: document.querySelector("#splatsContent"),
  splatsMeta: document.querySelector("#splatsMeta"),
  splatParameterEffects: document.querySelector("#splatParameterEffects"),
  splatAlphaBackground: document.querySelector("#splatAlphaBackground"),
  splatSmallFirstOrder: document.querySelector("#splatSmallFirstOrder"),
  splatShapeGaussian: document.querySelector("#splatShapeGaussian"),
  splatShapeRectangle: document.querySelector("#splatShapeRectangle"),
  splatShapeOpaqueBrush: document.querySelector("#splatShapeOpaqueBrush"),
  splatOpacity: document.querySelector("#splatOpacity"),
  splatOpacityValue: document.querySelector("#splatOpacityValue"),
  splatKernelFalloff: document.querySelector("#splatKernelFalloff"),
  splatKernelFalloffValue: document.querySelector("#splatKernelFalloffValue"),
  splatScale: document.querySelector("#splatScale"),
  splatScaleValue: document.querySelector("#splatScaleValue"),
  splatAspectRatio: document.querySelector("#splatAspectRatio"),
  splatAspectRatioValue: document.querySelector("#splatAspectRatioValue"),
  resetSplatAdjustments: document.querySelector("#resetSplatAdjustments"),
  splatAdjustStatus: document.querySelector("#splatAdjustStatus"),
  exportDescription: document.querySelector("#exportDescription"),
  exportCount: document.querySelector("#exportCount"),
  exportStatus: document.querySelector("#exportStatus"),
};

const trainingUiAdapter = Object.freeze({
  controls: els,
  confirm(message) {
    return window.confirm(message);
  },
  setDataset(name, value) {
    document.documentElement.dataset[name] = String(value);
  },
});

const previewCtx = els.previewCanvas.getContext("2d", { willReadFrequently: true });

function resizedSize(width, height, longSide, maximumLongSide = LIMITS.trainSizeMax) {
  const safeLongSide = clampNumber(longSide, 1, maximumLongSide, Math.min(DEFAULT_MAX_SIDE, maximumLongSide));
  const scale = Math.min(1, safeLongSide / Math.max(width, height));
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

function syncTrainSizeUi() {
  els.trainSize.disabled = state.running;
  publishState();
}

function normalizeUiSplatCount(value, fallback = DEFAULT_FINAL_SPLATS, max = LIMITS.splatsMax) {
  return normalizeStepInteger(value, {
    min: LIMITS.splatsMin,
    max: Math.min(LIMITS.splatsMax, max),
    fallback,
    step: 4,
  });
}

function normalizeActiveSplatCount(value, fallback = DEFAULT_INITIAL_SPLATS, max = LIMITS.splatsMax) {
  return normalizeStepInteger(value, {
    min: LIMITS.splatsMin,
    max: Math.min(LIMITS.splatsMax, max),
    fallback,
    step: 1,
  });
}

function assertSplatCountContract(params, context) {
  const count = Number(params?.count);
  if (!Number.isSafeInteger(count) || count < LIMITS.splatsMin || count > LIMITS.splatsMax) {
    throw new Error(`${context}: invalid splat count ${String(params?.count)}`);
  }
  const requiredLengths = { xy: count * 2, scale: count * 2, rgb: count * 3, opacity: count, theta: count };
  for (const [name, length] of Object.entries(requiredLengths)) {
    if (!params[name] || params[name].length !== length) {
      throw new Error(`${context}: ${name} length ${params[name]?.length ?? "missing"} != ${length}`);
    }
  }
  for (const name of ["depthOrder", "virtualDepth", "detailTags"]) {
    if (params[name] && params[name].length !== count) {
      throw new Error(`${context}: ${name} length ${params[name].length} != ${count}`);
    }
  }
  return count;
}

async function startTraining() {
  if (!state.image) {
    setTrainingMessage("Load an image before training.", "error");
    setStatus("no image");
    return false;
  }
  if (state.running || state.startPending || state.webGpuRecoveryPending) return false;
  const run = beginTrainingRun();
  state.startPending = true;
  setInputControlsDisabled(true);
  setTrainingMessage(state.previewRefreshPending ? "Waiting for the preview to finish..." : "Preparing WebGPU training...");
  publishLifecycleInteractionState();
  try {
    if (state.previewRefreshPending) {
      eventLog("Train requested while preview refresh was pending; waiting for the preview contract.");
      await awaitTrainingRun(run, state.previewRefreshPromise);
    }
    if (!ownsTrainingRun(run) || !state.image || state.running) return false;
    await awaitTrainingRun(run, selectedAlgorithm().train(run));
    return ownsTrainingRun(run);
  } catch (error) {
    if (error.trainingRunCancelled || !ownsTrainingRun(run)) return false;
    run.renderer?.disposeTrainState();
    state.running = false;
    state.paused = false;
    setInputControlsDisabled(false);
    setPausedRuntimeControlsEnabled(false);
    setStatus("error");
    setTrainingMessage(`Training failed: ${error.message}`, "error");
    log(`training start failed: ${error.message}`);
    eventLog(`training start failed: ${error.message}`);
    return false;
  } finally {
    if (state.trainingRun === run) {
      state.startPending = false;
      state.trainingRun = null;
      publishLifecycleInteractionState();
      if (
        state.previewMode === "splats" &&
        state.image &&
        state.params &&
        state.metrics?.final_cpu_result_ready_at &&
        state.webgpu.renderer &&
        !state.webgpu.renderer.deviceLost
      ) {
        try {
          await refreshOutsidePreview();
        } catch (error) {
          log(`post-training preview refresh failed: ${error.message}`);
        }
      }
    }
  }
}

async function trainPlanarGaussian(run) {
  return trainGaussianAlgorithm(false, run);
}

async function trainRectangleSplats(run) {
  return trainGaussianAlgorithm(false, run);
}

async function trainLayeredOpaqueBrush(run) {
  return trainGaussianAlgorithm(false, run);
}

async function trainGsVirtualCameraSampling(run) {
  return trainGaussianAlgorithm(true, run);
}


function stopTraining() {
  if (!state.running) return;
  state.stopRequested = true;
  state.paused = false;
  setPausedRuntimeControlsEnabled(false);
  setStatus("stopping");
  eventLog("stop requested");
}

function togglePause() {
  if (!state.running) return;
  state.paused = !state.paused;
  state.visibilityPaused = false;
  if (state.paused) {
    setPausedRuntimeControlsEnabled(true);
  } else {
    setPausedRuntimeControlsEnabled(false);
    state.runtimeSettingsRevision += 1;
  }
  setStatus(state.paused ? "paused" : "running");
  eventLog(state.paused ? "training paused" : "training resumed");
}

function resetTrainingState() {
  if (!state.image || state.running) return;
  invalidateTrainingRun("training reset");
  cancelCompletedResultGpuRecovery();
  destroyTiltViewer({ restoreCanvas: true });
  state.webgpu.renderer?.disposeTrainState();
  state.webgpu.renderer?.disposeResultRenderState();
  state.params = null;
  state.flowSplatResult = null;
  state.metrics = null;
  state.previewPadding = previewPaddingSpec(state.image, null, false);
  state.paused = false;
  state.stopRequested = false;
  state.lastGpuLoss = null;
  els.stepText.textContent = "0 / 0";
  els.lossText.textContent = "-";
  els.psnrText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.coverageText.textContent = "- / -";
  resetTrainingTiming();
  clearSplatAdjustmentBaseline();
  setPreviewMode("original");
  updateDownloads(false);
  updateMemoryRecommendation();
  updateVirtualCameraCoverageEstimate();
  setStatus("image loaded");
  setTrainingMessage("Reset to the loaded image.", "success");
  log("reset splats to loaded image");
}

function clearImage() {
  if (state.running) return;
  invalidateTrainingRun("image cleared");
  cancelCompletedResultGpuRecovery();
  destroyTiltViewer({ restoreCanvas: true });
  state.imageLoadGeneration += 1;
  state.imageLoading = false;
  state.webgpu.renderer?.disposeTrainState();
  state.webgpu.renderer?.disposeResultRenderState();
  releaseImageSource(state.image);
  state.image = null;
  state.params = null;
  state.flowSplatResult = null;
  state.metrics = null;
  state.paused = false;
  state.stopRequested = false;
  state.lastGpuLoss = null;
  state.previewMode = "original";
  state.previewPadding = { x: 0, y: 0, width: 640, height: 420, bytes: 0 };
  els.fileInput.value = "";
  els.dropZone.classList.remove("ready", "dragover");
  els.previewCanvas.width = 640;
  els.previewCanvas.height = 420;
  els.gpuCanvas.width = 640;
  els.gpuCanvas.height = 420;
  previewCtx.clearRect(0, 0, els.previewCanvas.width, els.previewCanvas.height);
  els.gpuCanvas.getContext("2d")?.clearRect(0, 0, els.gpuCanvas.width, els.gpuCanvas.height);
  showCanvas("preview");
  updatePreviewModeControls();
  fitCanvases();
  els.stepText.textContent = "0 / 0";
  els.lossText.textContent = "-";
  els.psnrText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.coverageText.textContent = "- / -";
  resetTrainingTiming();
  els.splatText.textContent = "-";
  updateImageSizeStatus();
  clearSplatAdjustmentBaseline();
  updateDownloads(false);
  updateMemoryRecommendation();
  updateVirtualCameraCoverageEstimate();
  syncAlgorithmRequirements();
  setStatus("idle");
  setTrainingMessage("Ready.");
  eventLog("image cleared");
}

function confirmClearImage() {
  if (!state.image || state.running) return;
  if (!window.confirm("Clear the loaded image and its training result?")) return;
  clearImage();
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function updateDownloads(enabled) {
  const requested = Boolean(enabled);
  const coverage = requested ? exportCoverageStatus() : null;
  state.exportReady = Boolean(requested && coverage?.ok);
  if (state.exportReady && !state.exporting && !state.exportMessage.startsWith("Exported")) {
    state.exportMessage = coverage?.warning
      ? `Ready to export with warning: ${coverage.message}.`
      : "Ready to export.";
  } else if (requested && !state.exporting && coverage && !coverage.ok) {
    state.exportMessage = `Export unavailable: ${coverage.message}.`;
  } else if (!state.exportReady && !state.exporting) {
    state.exportMessage = "Finish training before export.";
  }
  updateExportPanel();
  publishState();
}

const EXPORT_FORMATS = {
  ply: {
    label: "PLY",
    filename: "image2splatpaint.ply",
    description: "PLY uses standard SH0 Gaussian Splatting fields. For conventional multi-angle 3DGS viewing, train with GS Virtual Camera Sampling; Planar Gaussian is front-view only.",
  },
  png: {
    label: "Splat PNG",
    filename: "image2splatpaint-splats.png",
    description: "Current Splats preview without UI zoom, pan, or controls.",
  },
};

function activateDetailTab(name) {
  const tabs = {
    training: [els.trainingLogTab, els.trainingLogPanel],
    event: [els.eventLogTab, els.eventLogPanel],
    splats: [els.splatsTab, els.splatsPanel],
    export: [els.exportTab, els.exportPanel],
    tilt: [els.tiltTab, els.tiltPanel],
  };
  if (previewModeInputLocked() && name === "splats") name = "training";
  if (trainingLifecycleInputLocked() && name === "export") name = "training";
  const resultAlgorithm = trainedResultAlgorithm();
  const tiltAvailable = resultAlgorithm
    ? Boolean(resultAlgorithm.capabilities.virtualCameras)
    : algorithmUsesVirtualCameras();
  if (name === "tilt" && !tiltAvailable) name = "training";
  if (name !== "tilt" && !els.tiltCanvas.hidden) {
    restorePrimaryCanvas();
    destroyTiltViewer();
  }
  for (const [key, [tab, panel]] of Object.entries(tabs)) {
    const active = key === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  }
  document.documentElement.dataset.activeDetailTab = name;
}
