function selectedPngExportResolutionMode() {
  const value = els.pngExportResolution?.value;
  return ["training", "2048", "4096", "custom"].includes(value)
    ? value
    : DEFAULT_PNG_EXPORT_RESOLUTION;
}

function currentFlowPngResult() {
  return state.flowSplatResult?.sourceImage === state.image
    ? state.flowSplatResult
    : null;
}

function pngExportSourceImage() {
  const flow = currentFlowPngResult();
  if (flow) return flow.image;
  if (state.params?.flowBirthLinksEnabled && state.params.flowTrainingSize) {
    return { ...state.image, width: state.params.flowTrainingSize[0], height: state.params.flowTrainingSize[1] };
  }
  return state.image;
}

function pngExportFrameSpec(image = pngExportSourceImage()) {
  const trainingWidth = Math.max(1, Math.round(Number(image?.width) || Number(els.trainSize?.value) || DEFAULT_MAX_SIDE));
  const trainingHeight = Math.max(1, Math.round(Number(image?.height) || trainingWidth));
  const trainingLongSide = Math.max(trainingWidth, trainingHeight);
  const mode = selectedPngExportResolutionMode();
  const requestedLongSide = mode === "training"
    ? trainingLongSide
    : mode === "custom"
      ? Number(els.pngExportLongSide?.value)
      : Number(mode);
  const longSide = Math.round(clampNumber(
    requestedLongSide,
    MIN_PNG_EXPORT_LONG_SIDE,
    MAX_PNG_EXPORT_LONG_SIDE,
    trainingLongSide,
  ));
  if (mode === "training") {
    return { mode, longSide: trainingLongSide, width: trainingWidth, height: trainingHeight };
  }
  const scale = longSide / trainingLongSide;
  return {
    mode,
    longSide,
    width: Math.max(1, Math.round(trainingWidth * scale)),
    height: Math.max(1, Math.round(trainingHeight * scale)),
  };
}

function syncPngExportResolutionUi() {
  const spec = pngExportFrameSpec();
  if (selectedPngExportResolutionMode() !== "custom") {
    els.pngExportLongSide.value = String(spec.longSide);
  } else {
    els.pngExportLongSide.value = String(Math.round(clampNumber(
      els.pngExportLongSide.value,
      MIN_PNG_EXPORT_LONG_SIDE,
      MAX_PNG_EXPORT_LONG_SIDE,
      spec.longSide,
    )));
  }
  els.pngExportResolution.disabled = state.exporting;
  els.pngExportLongSide.disabled = state.exporting;
  els.pngExportResolutionStatus.textContent = `${spec.width} x ${spec.height}px`;
  return spec;
}

function currentSplatPngSpec() {
  if (currentFlowPngResult()) {
    return {
      filename: "image2splatpaint-flow-brush-fusion.png",
      shape: "flow-brush-fusion",
      renderOptions: {
        splatSmallFirstOrder: false,
        kernelFalloff: 1,
        alphaBackground: [0, 0, 0],
        outside: false,
      },
    };
  }
  const renderOptions = {
    ...splatAlphaRenderOptions(),
    outside: Boolean(els.outsidePreviewToggle?.checked),
  };
  const shape = renderOptions.splatShape === "opaque-brush"
    ? "opaque-brush"
    : ["rectangle", "box"].includes(renderOptions.splatShape)
      ? "rectangle"
      : "gaussian";
  return {
    filename: state.params?.flowBirthLinksEnabled ? "image2splatpaint-flow-brush-fusion.png" : `image2splatpaint-splats-${shape}.png`,
    shape,
    renderOptions,
  };
}

function updateExportPanel() {
  const pngFrame = syncPngExportResolutionUi();
  const enabled = state.exportReady && !state.exporting;
  const algorithm = displayedResultAlgorithm();
  const plySupported = algorithmSupportsExport("ply", algorithm);
  const pngSupported = algorithmSupportsExport("png", algorithm);
  const plyPlan = plySupported && state.params && state.image
    ? plyExportMemoryPlan(state.params, state.image, { download: true })
    : null;
  const plyEnabled = enabled && plySupported && Boolean(plyPlan?.ok);
  const pngEnabled = enabled && pngSupported;
  els.savePngButton.disabled = !pngEnabled;
  els.savePlyButton.disabled = !plyEnabled;
  els.savePngButton.textContent = state.exporting ? "Saving..." : "Save Splat PNG";
  els.savePlyButton.textContent = state.exporting ? "Exporting..." : "Export PLY";
  const resultKernelShape = normalizedKernelShape(
    state.params?.kernelShape || algorithm.capabilities.kernelShape,
  );
  els.exportDescription.textContent = currentFlowPngResult()
    ? "Curved Brush Splats exports its final opaque painted result as PNG. PLY cannot represent linked or internally bent Brush stroke structure."
    : resultKernelShape === "opaque-brush"
      ? "Brush Splats export exactly as PNG. Standard Gaussian Splatting PLY cannot represent the analytic brush kernel."
      : resultKernelShape === "rectangle"
        ? "Rectangle Splats export exactly as PNG. Standard Gaussian Splatting PLY cannot represent the rectangular kernel."
        : EXPORT_FORMATS.ply.description;
  const exportCount = state.params?.count ?? state.metrics?.num_gaussians;
  els.exportCount.textContent = Number.isFinite(exportCount) ? exportCount.toLocaleString() : "-";
  els.exportStatus.textContent = enabled && plyPlan && !plyPlan.ok
    ? `PNG is ready. PLY needs ${plyPlan.estimatedPeakMB} MB peak memory; ${plyPlan.reason}.`
    : state.exportMessage;
  const data = document.documentElement.dataset;
  data.exportReady = String(enabled);
  data.exportResultAlgorithm = algorithm.id;
  data.pngExportReady = String(pngEnabled);
  data.pngExportResolution = pngFrame.mode;
  data.pngExportLongSide = String(pngFrame.longSide);
  data.pngExportWidth = String(pngFrame.width);
  data.pngExportHeight = String(pngFrame.height);
  data.plyExportReady = String(plyEnabled);
  data.plyExportPeakMb = plyPlan?.estimatedPeakMB || "";
  data.plyExportBudgetMb = plyPlan?.budgetMB || "";
}

function exportCoverageStatus(metrics = state.metrics) {
  if (metrics?.safety_stop) {
    return { ok: false, reason: "safety_stop", message: "training ended with a safety stop" };
  }
  const coverage = metrics?.coverage_stats;
  const exposedPixels = coverage?.background_exposure_count;
  const coverageCurrent = coverage?.step === metrics?.steps_done;
  const revisionCurrent =
    metrics?.params_revision === undefined || metrics?.coverage_revision === metrics?.params_revision;
  if (!revisionCurrent) {
    return { ok: false, reason: "coverage_stale", message: "coverage is stale for the current splat revision" };
  }
  if (!coverageCurrent || typeof exposedPixels !== "number" || !Number.isFinite(exposedPixels)) {
    return { ok: false, reason: "coverage_missing", message: "final coverage was not measured" };
  }
  const parity = metrics?.render_surface_parity;
  if (
    !parity ||
    !Number.isFinite(parity.max_abs) ||
    !Number.isFinite(parity.mean_abs) ||
    typeof parity.display_equivalent !== "boolean"
  ) {
    return { ok: false, reason: "render_parity_missing", message: "final render parity was not measured" };
  }
  if (!parity.display_equivalent) {
    return {
      ok: false,
      reason: "render_parity_mismatch",
      message:
        `training and export render surfaces differ: alpha ${parity.alpha_max_abs}, ` +
        `premultiplied max ${parity.premultiplied_max_abs}`,
    };
  }
  if (exposedPixels !== 0) {
    return {
      ok: true,
      warning: true,
      reason: "background_exposure",
      exposedPixels,
      message: `background exposed at ${exposedPixels} pixels`,
    };
  }
  return { ok: true, warning: false, reason: "verified", exposedPixels: 0, message: "final coverage verified" };
}

function downloadBlob(name, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  state.lastDownload = `${name}:${blob.size}`;
  log(`download ${name} bytes=${blob.size}`);
  publishState();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function decodeImageBlobRgba(blob, width, height) {
  let source = null;
  let objectUrl = "";
  try {
    if (typeof createImageBitmap === "function") {
      source = await createImageBitmap(blob);
    } else {
      objectUrl = URL.createObjectURL(blob);
      source = await new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("PNG round-trip image decode failed."));
        image.src = objectUrl;
      });
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: true, willReadFrequently: true });
    if (!context) throw new Error("2D canvas is unavailable for PNG round-trip validation.");
    context.clearRect(0, 0, width, height);
    context.drawImage(source, 0, 0, width, height);
    return new Uint8ClampedArray(context.getImageData(0, 0, width, height).data);
  } finally {
    source?.close?.();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}

async function makeSplatPreviewPngBlob() {
  const flowResult = currentFlowPngResult();
  if (flowResult) return makeFlowPreviewPngBlob(flowResult);
  if (!state.image || !state.params || !state.webgpu.renderer) {
    throw new Error("No trained splat result to export.");
  }
  const spec = currentSplatPngSpec();
  const sourceImage = pngExportSourceImage();
  const exportFrame = pngExportFrameSpec(sourceImage);
  const renderImage = exportFrame.mode === "training"
    ? sourceImage
    : { ...sourceImage, width: exportFrame.width, height: exportFrame.height };
  const displayParams = state.params;
  const renderBuffers = state.webgpu.renderer.currentResultBuffers(displayParams);
  const capture = await state.webgpu.renderer.captureRenderedRgba(
    renderImage,
    displayParams,
    renderBuffers,
    {
      ...spec.renderOptions,
      rebuildTiles: exportFrame.width !== state.image.width || exportFrame.height !== state.image.height,
    },
  );
  const { rgba, width, height } = capture;
  let nonblackPixels = 0;
  let rgbSum = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    const sum = rgba[i] + rgba[i + 1] + rgba[i + 2];
    rgbSum += sum;
    if (sum > 0) nonblackPixels += 1;
  }
  let sourceNonblackPixels = 0;
  for (let i = 0; i < state.image.rgb.length; i += 3) {
    if (state.image.rgb[i] + state.image.rgb[i + 1] + state.image.rgb[i + 2] > 0) sourceNonblackPixels += 1;
  }
  if (sourceNonblackPixels > 0 && nonblackPixels === 0) {
    throw new Error("PNG frame readback is unexpectedly all black.");
  }
  const frame = document.createElement("canvas");
  frame.width = width;
  frame.height = height;
  const context = frame.getContext("2d", { alpha: true });
  if (!context) throw new Error("2D canvas is unavailable for PNG export.");
  context.putImageData(new ImageData(rgba, width, height), 0, 0);
  const blob = await canvasToBlob(frame);
  const decodedRgba = await decodeImageBlobRgba(blob, width, height);
  const pngRgbaParity = displayRgbaParity(rgba, decodedRgba);
  if (!pngRgbaParity.exact) {
    throw new Error(
      `PNG display RGBA round-trip mismatch: alpha ${pngRgbaParity.alpha_max_abs}, ` +
      `premultiplied max ${pngRgbaParity.premultiplied_max_abs}, mean ${pngRgbaParity.premultiplied_mean_abs}`,
    );
  }
  return {
    blob,
    width,
    height,
    exportFrame,
    spec,
    padding: capture.padding,
    nonblackPixels,
    meanRgb: rgbSum / Math.max(1, width * height * 3 * 255),
    pngRgbaParity,
  };
}

async function makeFlowPreviewPngBlob(flowResult = currentFlowPngResult()) {
  if (!flowResult?.image) throw new Error("No trained Curved Brush Splats result to export.");
  const sourceImage = flowResult.image;
  const exportFrame = pngExportFrameSpec(sourceImage);
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = sourceImage.width;
  sourceCanvas.height = sourceImage.height;
  const sourceContext = sourceCanvas.getContext("2d", { alpha: true });
  if (!sourceContext) throw new Error("2D canvas is unavailable for Flow PNG export.");
  sourceContext.putImageData(
    rgbToImageData(sourceImage.rgb, sourceImage.width, sourceImage.height, sourceImage.alpha),
    0,
    0,
  );

  const frame = document.createElement("canvas");
  frame.width = exportFrame.width;
  frame.height = exportFrame.height;
  const context = frame.getContext("2d", { alpha: true, willReadFrequently: true });
  if (!context) throw new Error("2D canvas is unavailable for Flow PNG export.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.clearRect(0, 0, frame.width, frame.height);
  context.drawImage(sourceCanvas, 0, 0, frame.width, frame.height);
  const rgba = new Uint8ClampedArray(context.getImageData(0, 0, frame.width, frame.height).data);
  const blob = await canvasToBlob(frame);
  const decodedRgba = await decodeImageBlobRgba(blob, frame.width, frame.height);
  const pngRgbaParity = displayRgbaParity(rgba, decodedRgba);
  if (!pngRgbaParity.exact) {
    throw new Error(
      `Flow PNG display RGBA round-trip mismatch: alpha ${pngRgbaParity.alpha_max_abs}, ` +
      `premultiplied max ${pngRgbaParity.premultiplied_max_abs}, mean ${pngRgbaParity.premultiplied_mean_abs}`,
    );
  }

  let nonblackPixels = 0;
  let rgbSum = 0;
  for (let index = 0; index < rgba.length; index += 4) {
    const sum = rgba[index] + rgba[index + 1] + rgba[index + 2];
    rgbSum += sum;
    if (sum > 0) nonblackPixels += 1;
  }
  return {
    blob,
    width: frame.width,
    height: frame.height,
    exportFrame,
    spec: currentSplatPngSpec(),
    padding: { x: 0, y: 0 },
    nonblackPixels,
    meanRgb: rgbSum / Math.max(1, frame.width * frame.height * 3 * 255),
    pngRgbaParity,
  };
}

function logit(value) {
  const v = Math.min(Math.max(value, 1e-6), 1 - 1e-6);
  return Math.log(v / (1 - v));
}

async function sha256Hex(buffer) {
  if (!globalThis.crypto?.subtle) return "";
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

const PLY_ROW_BYTES = 17 * 4;
const GPU_READBACK_ROW_BYTES = 12 * 4;
const CPU_PARAMETER_ROW_BYTES = 52;

function parameterArrayBytes(params) {
  if (!params) return 0;
  let bytes = 0;
  let found = false;
  for (const values of [
    params.xy,
    params.scale,
    params.rgb,
    params.opacity,
    params.theta,
    params.depthOrder,
    params.virtualDepth,
    params.brushTaper,
    params.internalBendShapes,
    params.detailTags,
    params.bg,
  ]) {
    if (!ArrayBuffer.isView(values)) continue;
    bytes += values.byteLength;
    found = true;
  }
  return found ? bytes : Math.max(0, Number(params.count) || 0) * CPU_PARAMETER_ROW_BYTES;
}

function imageCpuResidentBytes(image) {
  if (!image) return 0;
  const floatCacheBytes =
    Math.max(0, Number(image.rgb?.byteLength) || 0) +
    Math.max(0, Number(image.alpha?.byteLength) || 0);
  // The bounded canvas is intentionally retained for Original preview and a
  // later training resize. Count it in CPU-side export headroom as well.
  const sourceWidth = Math.max(0, Math.round(Number(image.sourceBitmap?.width) || 0));
  const sourceHeight = Math.max(0, Math.round(Number(image.sourceBitmap?.height) || 0));
  return floatCacheBytes + sourceWidth * sourceHeight * 4;
}

function browserCpuMemoryInfo() {
  const heapLimit = Number(performance?.memory?.jsHeapSizeLimit);
  const heapUsed = Number(performance?.memory?.usedJSHeapSize);
  if (Number.isFinite(heapLimit) && heapLimit > 0 && Number.isFinite(heapUsed) && heapUsed >= 0) {
    return { source: "jsHeapSizeLimit", budgetBytes: heapLimit, usedBytes: heapUsed, exactFree: true };
  }
  const deviceMemoryGB = Number(navigator.deviceMemory);
  if (Number.isFinite(deviceMemoryGB) && deviceMemoryGB > 0) {
    return {
      source: `deviceMemory ${deviceMemoryGB}GB`,
      budgetBytes: clampNumber(deviceMemoryGB * GB * 0.125, 256 * MB, 1024 * MB, 256 * MB),
      usedBytes: 0,
      exactFree: false,
    };
  }
  return { source: "conservative fallback", budgetBytes: 256 * MB, usedBytes: 0, exactFree: false };
}

function plyHeaderText(params, image) {
  const boundarySigma = Number.isFinite(params.boundarySigma) ? params.boundarySigma : selectedBoundarySigma();
  return createPlyHeader({
    count: params.count,
    image,
    boundarySigma,
    layerOrderEnabled: Boolean(params.layerOrderEnabled),
    layerDepthSpan: PLY_LAYER_DEPTH_SPAN,
  });
}

function plyExportMemoryPlan(
  params = state.params,
  image = state.image,
  { download = true, memoryInfo = browserCpuMemoryInfo(), baseline = state.splatBaseline } = {},
) {
  const count = Math.max(0, Number(params?.count) || 0);
  const parameterBytes = parameterArrayBytes(params);
  const baselineBytes = baseline === params
    ? 0
    : baseline === true
      ? count * CPU_PARAMETER_ROW_BYTES
      : parameterArrayBytes(baseline);
  const imageBytes = imageCpuResidentBytes(image);
  const readbackBytes = count * GPU_READBACK_ROW_BYTES;
  const headerBytes = params && image ? new TextEncoder().encode(plyHeaderText(params, image)).byteLength : 1024;
  const plyBytes = headerBytes + count * PLY_ROW_BYTES;
  const blobCopyBytes = download ? plyBytes : 0;
  const residentBytes = parameterBytes + baselineBytes + imageBytes;
  return exportPeakMemoryPlan({
    count,
    parameterBytes,
    baselineBytes,
    imageBytes,
    readbackBytes,
    plyBytes,
    blobCopyBytes,
    residentBytes,
    memoryInfo,
  });
}

function exportPeakMemoryPlan({
  count,
  parameterBytes,
  baselineBytes,
  imageBytes,
  readbackBytes,
  plyBytes,
  blobCopyBytes,
  residentBytes = parameterBytes + baselineBytes + imageBytes,
  memoryInfo,
}) {
  const requiredIncrementBytes = Math.max(readbackBytes, plyBytes + blobCopyBytes);
  const estimatedPeakBytes = residentBytes + requiredIncrementBytes;
  const budgetBytes = Math.max(1, Number(memoryInfo?.budgetBytes) || 256 * MB);
  const availableBytes = Math.max(0, budgetBytes - Math.max(0, Number(memoryInfo?.usedBytes) || 0));
  const ok = memoryInfo?.exactFree
    ? requiredIncrementBytes <= availableBytes * 0.75
    : estimatedPeakBytes <= budgetBytes * 0.9;
  const reason = ok
    ? `${memoryInfo?.source || "memory estimate"} has sufficient headroom`
    : memoryInfo?.exactFree
      ? `available JS heap is ${bytesToMB(availableBytes).toFixed(0)} MB`
      : `${memoryInfo?.source || "memory estimate"} budget is ${bytesToMB(budgetBytes * 0.9).toFixed(0)} MB`;
  return {
    ok,
    count,
    parameterBytes,
    baselineBytes,
    imageBytes,
    readbackBytes,
    plyBytes,
    blobCopyBytes,
    residentBytes,
    requiredIncrementBytes,
    estimatedPeakBytes,
    estimatedPeakMB: bytesToMB(estimatedPeakBytes).toFixed(1),
    budgetBytes,
    budgetMB: bytesToMB(budgetBytes).toFixed(0),
    memorySource: memoryInfo?.source || "unknown",
    reason,
  };
}

function assertPlyExportCapacity(params = state.params, image = state.image, download = true) {
  const plan = plyExportMemoryPlan(params, image, { download });
  if (!plan.ok) {
    throw Object.assign(
      new Error(`PLY export needs ${plan.estimatedPeakMB} MB peak memory; ${plan.reason}`),
      { exportOnly: true, exportMemoryPlan: plan },
    );
  }
  return plan;
}

function assertTiltViewerCapacity(params = state.params, image = state.image) {
  const plan = plyExportMemoryPlan(params, image, { download: false });
  const viewerBytes = plan.plyBytes * 4;
  const requiredIncrementBytes = Math.max(plan.readbackBytes, viewerBytes);
  const estimatedPeakBytes = plan.residentBytes + requiredIncrementBytes;
  const memoryInfo = browserCpuMemoryInfo();
  const availableBytes = Math.max(0, plan.budgetBytes - Math.max(0, Number(memoryInfo.usedBytes) || 0));
  const ok = memoryInfo.exactFree
    ? requiredIncrementBytes <= availableBytes * 0.6
    : estimatedPeakBytes <= plan.budgetBytes * 0.8;
  if (!ok) {
    throw Object.assign(
      new Error(`Tilt view needs about ${bytesToMB(estimatedPeakBytes).toFixed(1)} MB peak memory; reduce Max splats.`),
      { exportOnly: true, tiltMemoryPlan: { ...plan, viewerBytes, requiredIncrementBytes, estimatedPeakBytes } },
    );
  }
  return { ...plan, viewerBytes, requiredIncrementBytes, estimatedPeakBytes };
}

function plyFrameScale(image = state.image) {
  return exportedPlyFrameScale(image);
}

function transformPlanarSplatForPly(x, y, sx, sy, theta, image = state.image) {
  // This is the single NDC-to-isotropic-world conversion, independent of grid initialization.
  return transformPlanarSplat(x, y, sx, sy, theta, image);
}

function sourceRgbAtNdc(image, x, y) {
  const px = Math.min(image.width - 1, Math.max(0, Math.round((Math.min(1, Math.max(-1, x)) * 0.5 + 0.5) * (image.width - 1))));
  const py = Math.min(image.height - 1, Math.max(0, Math.round((Math.min(1, Math.max(-1, y)) * 0.5 + 0.5) * (image.height - 1))));
  const offset = (py * image.width + px) * 3;
  return [image.rgb[offset], image.rgb[offset + 1], image.rgb[offset + 2]];
}

function tiltRiskProfileForSplat(params, image, index, angleDegrees = DEFAULT_TILT_SPLIT_ANGLE_DEGREES) {
  const theta = params.theta?.[index] || 0;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const sx = params.scale[index * 2];
  const sy = params.scale[index * 2 + 1];
  const frame = plyFrameScale(image);
  const worldX = [c * sx * frame.x, s * sx * frame.y];
  const worldY = [-s * sy * frame.x, c * sy * frame.y];
  const angleSin = Math.sin(Math.max(0, angleDegrees) * Math.PI / 180);
  const yawDepth = 4 * angleSin * Math.hypot(worldX[0], worldY[0]);
  const pitchDepth = 4 * angleSin * Math.hypot(worldX[1], worldY[1]);
  const supportDepth = Math.max(yawDepth, pitchDepth);
  const centerX = params.xy[index * 2];
  const centerY = params.xy[index * 2 + 1];
  const axisX = [c * sx, s * sx];
  const axisY = [-s * sy, c * sy];
  const sourceColor = [params.rgb[index * 3], params.rgb[index * 3 + 1], params.rgb[index * 3 + 2]];
  const samples = [
    sourceRgbAtNdc(image, centerX - axisX[0], centerY - axisX[1]),
    sourceRgbAtNdc(image, centerX + axisX[0], centerY + axisX[1]),
    sourceRgbAtNdc(image, centerX - axisY[0], centerY - axisY[1]),
    sourceRgbAtNdc(image, centerX + axisY[0], centerY + axisY[1]),
    sourceRgbAtNdc(image, centerX - axisX[0] * 4, centerY - axisX[1] * 4),
    sourceRgbAtNdc(image, centerX + axisX[0] * 4, centerY + axisX[1] * 4),
    sourceRgbAtNdc(image, centerX - axisY[0] * 4, centerY - axisY[1] * 4),
    sourceRgbAtNdc(image, centerX + axisY[0] * 4, centerY + axisY[1] * 4),
  ];
  const colorMismatch = samples.reduce((maximum, sample) => Math.max(
    maximum,
    (Math.abs(sample[0] - sourceColor[0]) + Math.abs(sample[1] - sourceColor[1]) + Math.abs(sample[2] - sourceColor[2])) / 3,
  ), 0);
  const opacity = params.opacity[index];
  const risk = opacity < 0.007 ? 0 : Math.min(64,
    opacity
      * Math.max(0, supportDepth / PLY_LAYER_DEPTH_SPAN - 1)
      * Math.max(0, colorMismatch / DEFAULT_TILT_SPLIT_COLOR_THRESHOLD - 1),
  );
  const useX = yawDepth >= pitchDepth
    ? Math.abs(worldX[0]) >= Math.abs(worldY[0])
    : Math.abs(worldX[1]) >= Math.abs(worldY[1]);
  return { index, risk, supportDepth, colorMismatch, splitAxis: useX ? "x" : "y" };
}

function summarizeTiltRisk(params, image) {
  const angleDegrees = phase39Variants().tiltSplitAngleDegrees;
  const sampling = finalDiagnosticSampling(params.count);
  const profiles = [];
  for (let index = 0; index < params.count; index += sampling.stride) {
    profiles.push(tiltRiskProfileForSplat(params, image, index, angleDegrees));
  }
  const support = profiles.map((profile) => profile.supportDepth).sort((a, b) => a - b);
  const percentile = (fraction) => support.length
    ? support[Math.min(support.length - 1, Math.round((support.length - 1) * fraction))]
    : 0;
  const risky = profiles.filter((profile) => profile.risk > 0).sort((a, b) => b.risk - a.risk);
  const riskyRatio = risky.length / Math.max(1, profiles.length);
  return {
    angle_degrees: angleDegrees,
    depth_threshold: PLY_LAYER_DEPTH_SPAN,
    color_threshold: DEFAULT_TILT_SPLIT_COLOR_THRESHOLD,
    source_count: sampling.sourceCount,
    sample_count: sampling.sampleCount,
    sample_stride: sampling.stride,
    risky_sample_count: risky.length,
    risky_count: Math.min(sampling.sourceCount, Math.round(riskyRatio * sampling.sourceCount)),
    risky_ratio: riskyRatio,
    support_depth_p95: percentile(0.95),
    support_depth_p99: percentile(0.99),
    support_depth_max: support.at(-1) || 0,
    top: risky.slice(0, 16),
  };
}

function plyLayerDepth(index, params, enabled = Boolean(params.layerOrderEnabled)) {
  const fallback = 1 - index / Math.max(1, params.count - 1);
  const order = Math.max(0, Math.min(1, params.depthOrder?.[index] ?? fallback));
  const layerDepth = enabled ? (order - 0.5) * PLY_LAYER_DEPTH_SPAN : 0;
  return layerDepth + boundedVirtualDepth(params, index);
}

function frontRenderLayerDepth(index, params) {
  if (params.virtualDepthEnabled) return plyLayerDepth(index, params);
  if (!params.layerOrderEnabled) return 0;
  const fallback = 1 - index / Math.max(1, params.count - 1);
  const order = Math.max(0, Math.min(1, params.depthOrder?.[index] ?? fallback));
  return order * LAYER_CODE_RANGE;
}

function layerOrderComparator(a, b, params) {
  const delta = frontRenderLayerDepth(b, params) - frontRenderLayerDepth(a, params);
  // Match tile_less() in the unit used by the active front renderer. PLY z is
  // a separate export-space mapping and must not change near-tie decisions.
  return Math.abs(delta) > 1e-7 ? delta : a - b;
}

function splatPreviewOrderComparator(a, b, params) {
  const opacityDelta = (params.opacity[b] || 0) - (params.opacity[a] || 0);
  if (Math.abs(opacityDelta) > 1e-4) return opacityDelta;
  const areaA = Math.max(MIN_SPLAT_SCALE, params.scale[a * 2]) * Math.max(MIN_SPLAT_SCALE, params.scale[a * 2 + 1]);
  const areaB = Math.max(MIN_SPLAT_SCALE, params.scale[b * 2]) * Math.max(MIN_SPLAT_SCALE, params.scale[b * 2 + 1]);
  const areaDelta = areaA - areaB;
  if (Math.abs(areaDelta) > 1e-10) return areaDelta;
  return layerOrderComparator(a, b, params);
}

function buildSplatPreviewOrder(params) {
  const ordered = new Uint32Array(params.count);
  for (let index = 0; index < params.count; index += 1) ordered[index] = index;
  ordered.sort((a, b) => splatPreviewOrderComparator(a, b, params));
  return ordered;
}

function cachedResultSmallFirstOrder(resultState, params) {
  if (!resultState || resultState.sourceParams !== params || resultState.count !== params.count) return null;
  const cache = resultState.smallFirstOrderCache;
  if (cache?.params === params && cache.count === params.count) return cache.order;
  const order = buildSplatPreviewOrder(params);
  resultState.smallFirstOrderCache = { params, count: params.count, order };
  return order;
}

function makePly(params = state.params, image = state.image) {
  if (!params) throw new Error("No trained splats to export.");
  if (!image) throw new Error("No source image is available for aspect-preserving PLY export.");
  assertSplatCountContract(params, "ply-export");
  assertFiniteParams(params, "ply-export");
  const layerOrderEnabled = Boolean(params.layerOrderEnabled);
  const header = plyHeaderText(params, image);
  return serializeBinaryPly({
    header,
    count: params.count,
    rowBytes: PLY_ROW_BYTES,
    shC0: SH_C0,
    geometryAt: (i) => transformPlanarSplatForPly(
      params.xy[i * 2],
      params.xy[i * 2 + 1],
      params.scale[i * 2],
      params.scale[i * 2 + 1],
      params.theta?.[i] || 0,
      image,
    ),
    depthAt: (i) => plyLayerDepth(i, params, layerOrderEnabled),
    rgbAt: (i) => [params.rgb[i * 3], params.rgb[i * 3 + 1], params.rgb[i * 3 + 2]],
    opacityAt: (i) => params.opacity[i],
    logit,
  });
}
