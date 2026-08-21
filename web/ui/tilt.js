function currentTiltRevision() {
  if (!state.params || !state.image) return "";
  return [
    state.splatAdjustmentEpoch,
    state.metrics?.params_revision ?? 0,
    state.metrics?.final_readback_step ?? -1,
    state.metrics?.steps_done ?? -1,
    state.splatAdjustmentVersion,
    state.params.count,
    state.image.width,
    state.image.height,
  ].join(":");
}

function tiltViewerRuntime() {
  const runtime = globalThis.Image2SplatPaintTilt;
  if (typeof runtime?.createTiltViewer !== "function") {
    throw new Error("Tilt renderer bundle did not load.");
  }
  return runtime;
}

function tiltViewerReady() {
  const resultAlgorithm = trainedResultAlgorithm();
  return Boolean(
    resultAlgorithm?.capabilities.virtualCameras &&
    typeof globalThis.Image2SplatPaintTilt?.createTiltViewer === "function" &&
    state.params &&
    state.metrics &&
    !state.running &&
    state.metrics.cpu_mirror_current &&
    state.metrics.final_readback_step === state.metrics.steps_done,
  );
}

function tiltViewerAvailabilityMessage() {
  const resultAlgorithm = trainedResultAlgorithm();
  if (resultAlgorithm && !resultAlgorithm.capabilities.virtualCameras) {
    return "Train with GS Virtual Camera Sampling before opening Tilt.";
  }
  if (!resultAlgorithm && !algorithmUsesVirtualCameras()) {
    return "Tilt is available only for GS Virtual Camera Sampling.";
  }
  if (typeof globalThis.Image2SplatPaintTilt?.createTiltViewer !== "function") return "Tilt renderer failed to load.";
  if (state.running) return "Stop or finish training before opening the Tilt viewer.";
  if (!state.params || !state.metrics) return "Finish training to inspect the PLY.";
  if (!state.metrics.cpu_mirror_current || state.metrics.final_readback_step !== state.metrics.steps_done) {
    return "Waiting for the final training result before opening the Tilt viewer.";
  }
  return "Open Tilt to build a fresh in-memory PLY view.";
}

function updateTiltControlState() {
  const ready = tiltViewerReady();
  const resultAlgorithm = trainedResultAlgorithm();
  const tiltAvailable = resultAlgorithm
    ? Boolean(resultAlgorithm.capabilities.virtualCameras)
    : algorithmUsesVirtualCameras();
  els.tiltTab.disabled = !tiltAvailable;
  els.tiltTab.setAttribute("aria-disabled", String(!tiltAvailable));
  const interactive = Boolean(state.tilt.controller) && !state.tilt.teacherViewsLoading;
  const inspectingCameraPool = interactive && state.tilt.cameraMarkersVisible;
  els.tiltPitch.disabled = !interactive || inspectingCameraPool;
  els.tiltYaw.disabled = !interactive || inspectingCameraPool;
  els.tiltFrontButton.disabled = !interactive || inspectingCameraPool;
  els.tiltRefreshButton.disabled = !ready || state.tilt.loading || state.tilt.teacherViewsLoading;
  els.tiltTrainingViewsButton.disabled = !interactive || state.tilt.teacherViewsLoading;
  for (const button of [els.tiltOriginalView, els.tiltOverlayView, els.tiltSplatsView]) {
    button.disabled = !interactive;
  }
  els.tiltCameraMarkers.disabled = !interactive;
  els.tiltRadiusMode.disabled = !interactive;
  document.documentElement.dataset.tiltReady = String(ready);
  document.documentElement.dataset.tiltLoaded = String(Boolean(state.tilt.controller));
  document.documentElement.dataset.tiltLoading = String(state.tilt.loading);
  document.documentElement.dataset.tiltRevision = state.tilt.revision;
  document.documentElement.dataset.tiltPlySha256 = state.tilt.plyDigest || state.tilt.verifiedPlyDigest;
  document.documentElement.dataset.tiltPlyBytes = String(state.tilt.plyByteLength || state.tilt.verifiedPlyByteLength || 0);
  document.documentElement.dataset.tiltPlyVertices = String(state.tilt.vertices || 0);
  document.documentElement.dataset.tiltTrainingViewsLoading = String(state.tilt.teacherViewsLoading);
  document.documentElement.dataset.tiltTrainingViewCount = String(state.tilt.teacherViews.length);
  document.documentElement.dataset.tiltDisplayMode = state.tilt.viewMode;
  document.documentElement.dataset.tiltCameraMarkers = String(state.tilt.cameraMarkersVisible);
  const cameraMarkers = state.tilt.controller?.diagnostics?.().cameraMarkers;
  const markerCount = cameraMarkers?.count || 0;
  const sampleCount = cameraMarkers?.sampleCount || 0;
  const frontCount = cameraMarkers?.frontCount || 0;
  const virtualCount = cameraMarkers?.virtualCount || 0;
  const frontSampleCount = cameraMarkers?.frontSampleCount || 0;
  const virtualSampleCount = cameraMarkers?.virtualSampleCount || 0;
  const runSampling = state.metrics?.virtual_camera_sampling;
  const runSamplingEnabled = Boolean(runSampling?.enabled);
  // This describes the next Train selection; existing Tilt availability above
  // remains bound to the completed result.
  const currentSamplingEnabled = algorithmUsesVirtualCameras();
  document.documentElement.dataset.tiltCameraMarkerCount = String(markerCount);
  document.documentElement.dataset.tiltCameraSampleCount = String(sampleCount);
  document.documentElement.dataset.tiltCameraFrontMarkerCount = String(frontCount);
  document.documentElement.dataset.tiltCameraVirtualMarkerCount = String(virtualCount);
  document.documentElement.dataset.tiltCameraFrontSampleCount = String(frontSampleCount);
  document.documentElement.dataset.tiltCameraVirtualSampleCount = String(virtualSampleCount);
  document.documentElement.dataset.tiltCameraRunSamplingEnabled = String(runSamplingEnabled);
  document.documentElement.dataset.tiltCameraCurrentSamplingEnabled = String(currentSamplingEnabled);
  if (!state.tilt.controller) {
    els.tiltCameraSummary.textContent = "Finish training to inspect the cameras used in this run.";
  } else if (virtualCount > 0) {
    els.tiltCameraSummary.textContent =
      `${markerCount} camera poses: ${frontCount} front + ${virtualCount} virtual` +
      ` · ${frontSampleCount} front / ${virtualSampleCount} virtual selections.`;
  } else if (runSamplingEnabled || cameraMarkers?.samplingEnabled) {
    const stepsDone = Number(state.metrics?.steps_done) || 0;
    const warmupSteps = Number(runSampling?.warmup_steps) || 0;
    els.tiltCameraSummary.textContent =
      stepsDone <= warmupSteps
        ? `${frontCount || 1} front camera · This run ended during the front-only warmup (${stepsDone} / ${warmupSteps}).`
        : `${frontCount || 1} front camera · No virtual camera was selected in this run.`;
  } else if (currentSamplingEnabled) {
    els.tiltCameraSummary.textContent =
      `${frontCount || 1} front camera · This run used front only. Virtual sampling is ON for the next Train.`;
  } else {
    els.tiltCameraSummary.textContent =
      `${frontCount || 1} front camera · Virtual camera sampling was off for this run.`;
  }
}

function restorePrimaryCanvas() {
  showCanvas(state.previewMode === "splats" && state.params ? "gpu" : "preview");
}

function positionTiltFrameSegment(element, start, end) {
  const length = Math.hypot(end.x - start.x, end.y - start.y);
  const angle = Math.atan2(end.y - start.y, end.x - start.x) * 180 / Math.PI;
  element.style.left = `${start.x}px`;
  element.style.top = `${start.y}px`;
  element.style.width = `${length}px`;
  element.style.transform = `rotate(${angle}deg)`;
}

function updateTiltCameraDiagnostics(diagnostics) {
  if (!diagnostics?.corners?.every((corner) => corner.valid)) {
    els.tiltFrameOverlay.hidden = true;
    els.tiltProjectionError.textContent = "invalid";
    return;
  }
  const segments = [...els.tiltFrameOverlay.querySelectorAll("span")];
  diagnostics.corners.forEach((corner, index) => {
    positionTiltFrameSegment(segments[index], corner, diagnostics.corners[(index + 1) % diagnostics.corners.length]);
  });
  const centerMarker = els.tiltFrameOverlay.querySelector("i");
  centerMarker.style.left = `${diagnostics.center.x}px`;
  centerMarker.style.top = `${diagnostics.center.y}px`;
  const trainingOrbitRadius = diagnostics.trainingOrbitRadius ?? diagnostics.radius;
  const viewerOrbitRadius = diagnostics.viewerOrbitRadius ?? diagnostics.radius;
  const displayedRadius = diagnostics.viewMode === "camera-pool-overview"
    ? trainingOrbitRadius
    : viewerOrbitRadius;
  els.tiltRadiusValue.textContent = displayedRadius.toFixed(4);
  els.tiltPositionValue.textContent = diagnostics.position.map((value) => value.toFixed(3)).join(", ");
  els.tiltFovValue.textContent = `${diagnostics.fovDegrees.toFixed(0)}\u00b0`;
  els.tiltProjectionError.textContent = `${diagnostics.cornerErrorMaxPx.toFixed(3)} px`;
  els.tiltCameraMode.textContent = diagnostics.viewMode === "camera-pool-overview"
    ? "camera pool overview"
    : diagnostics.radiusMode === "fit" ? "center orbit / fit splats" : "center orbit / training radius";
  els.tiltFrameOverlay.hidden = els.tiltCanvas.hidden;
  document.documentElement.dataset.tiltOrbitRadius = String(displayedRadius);
  document.documentElement.dataset.tiltTrainingOrbitRadius = String(trainingOrbitRadius);
  document.documentElement.dataset.tiltViewerOrbitRadius = String(viewerOrbitRadius);
  document.documentElement.dataset.tiltFitAllOrbitRadius = String(diagnostics.fitAllOrbitRadius ?? viewerOrbitRadius);
  document.documentElement.dataset.tiltRadiusMode = diagnostics.radiusMode || "training";
  document.documentElement.dataset.tiltViewMode = diagnostics.viewMode || "center-orbit";
  document.documentElement.dataset.tiltCameraPosition = diagnostics.position.join(",");
  document.documentElement.dataset.tiltFov = String(diagnostics.fovDegrees);
  document.documentElement.dataset.tiltProjectionError = String(diagnostics.cornerErrorMaxPx);
  scheduleTiltTeacherFrame(diagnostics);
}

function setTiltDisplayMode(mode, { focusCamera = false } = {}) {
  state.tilt.viewMode = ["original", "overlay", "splats"].includes(mode) ? mode : "splats";
  if (focusCamera && state.tilt.controller && state.tilt.cameraMarkersVisible) {
    state.tilt.cameraMarkersVisible = false;
    els.tiltCameraMarkers.checked = false;
    state.tilt.controller.setCameraMarkersVisible(false);
  }
  for (const [button, value] of [
    [els.tiltOriginalView, "original"],
    [els.tiltOverlayView, "overlay"],
    [els.tiltSplatsView, "splats"],
  ]) {
    const active = state.tilt.viewMode === value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  if (!els.tiltCanvas.hidden) showCanvas("tilt");
  const diagnostics = state.tilt.controller?.diagnostics?.().camera;
  if (diagnostics && state.tilt.viewMode !== "splats") scheduleTiltTeacherFrame(diagnostics);
  publishState();
}

function rasterizeVirtualTeacher(canvas, image, camera, { maxSide = 512, outputAspect = null } = {}) {
  const aspect = Number.isFinite(Number(outputAspect)) && Number(outputAspect) > 0
    ? Number(outputAspect)
    : image.width / image.height;
  const width = aspect >= 1 ? Math.max(1, Math.round(maxSide)) : Math.max(1, Math.round(maxSide * aspect));
  const height = aspect >= 1 ? Math.max(1, Math.round(maxSide / aspect)) : Math.max(1, Math.round(maxSide));
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  const pixels = context.createImageData(width, height);
  const pitch = Number(camera.pitchDegrees ?? camera.pitch_degrees) || 0;
  const yaw = Number(camera.yawDegrees ?? camera.yaw_degrees) || 0;
  const distance = Math.max(0.01, Number(camera.cameraDistance ?? camera.radius) || DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE);
  const fovDegrees = clampSharedCameraFov(camera.fovDegrees ?? camera.fov_degrees);
  const rotation = planarTiltRotation(pitch * Math.PI / 180, yaw * Math.PI / 180);
  const longSide = Math.max(1, image.width, image.height);
  const frameX = image.width / longSide;
  const frameY = image.height / longSide;
  const tanHalfFov = Math.tan(fovDegrees * Math.PI / 360);
  const focalX = 1 / (tanHalfFov * aspect);
  const focalY = 1 / tanHalfFov;
  const sourceWidth = image.width;
  const sourceHeight = image.height;
  for (let py = 0, target = 0; py < height; py += 1) {
    const v = height > 1 ? py / (height - 1) * 2 - 1 : 0;
    const a10 = v * frameX * rotation[6] + focalY * frameX * rotation[3];
    const a11 = -v * frameY * rotation[7] - focalY * frameY * rotation[4];
    const rightY = -v * distance;
    for (let px = 0; px < width; px += 1, target += 4) {
      const u = width > 1 ? px / (width - 1) * 2 - 1 : 0;
      const a00 = u * frameX * rotation[6] - focalX * frameX * rotation[0];
      const a01 = -u * frameY * rotation[7] + focalX * frameY * rotation[1];
      const determinant = a00 * a11 - a01 * a10;
      const rightX = -u * distance;
      if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-8) {
        pixels.data[target + 3] = 255;
        continue;
      }
      const sourceX = (rightX * a11 - a01 * rightY) / determinant;
      const sourceY = (a00 * rightY - rightX * a10) / determinant;
      if (!Number.isFinite(sourceX + sourceY) || sourceX < -1 || sourceX > 1 || sourceY < -1 || sourceY > 1) {
        pixels.data[target + 3] = 255;
        continue;
      }
      const sx = (sourceX * 0.5 + 0.5) * (sourceWidth - 1);
      const sy = (sourceY * 0.5 + 0.5) * (sourceHeight - 1);
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const y1 = Math.min(sourceHeight - 1, y0 + 1);
      const fx = sx - x0;
      const fy = sy - y0;
      const i00 = (y0 * sourceWidth + x0) * 3;
      const i10 = (y0 * sourceWidth + x1) * 3;
      const i01 = (y1 * sourceWidth + x0) * 3;
      const i11 = (y1 * sourceWidth + x1) * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        const top = image.rgb[i00 + channel] * (1 - fx) + image.rgb[i10 + channel] * fx;
        const bottom = image.rgb[i01 + channel] * (1 - fx) + image.rgb[i11 + channel] * fx;
        pixels.data[target + channel] = clampByte((top * (1 - fy) + bottom * fy) * 255);
      }
      pixels.data[target + 3] = 255;
    }
  }
  context.putImageData(pixels, 0, 0);
  return { width, height };
}

function scheduleTiltTeacherFrame(diagnostics) {
  if (!state.image || state.tilt.viewMode === "splats" || els.tiltCanvas.hidden) return;
  const request = ++state.tilt.teacherFrameRequest;
  requestAnimationFrame(() => {
    if (request !== state.tilt.teacherFrameRequest || !state.image) return;
    const aspect = Math.max(1, els.tiltCanvas.clientWidth) / Math.max(1, els.tiltCanvas.clientHeight);
    rasterizeVirtualTeacher(els.tiltTeacherCanvas, state.image, {
      pitchDegrees: diagnostics.pitchDegrees,
      yawDegrees: diagnostics.yawDegrees,
      cameraDistance: diagnostics.radius,
      fovDegrees: diagnostics.fovDegrees,
    }, { maxSide: Math.min(768, Math.max(256, els.tiltCanvas.clientWidth)), outputAspect: aspect });
  });
}

function clearTiltTrainingViews() {
  state.tilt.teacherViews = [];
  els.tiltContactSheet.replaceChildren();
  els.tiltTrainingViewsProgress.max = 1;
  els.tiltTrainingViewsProgress.value = 0;
  els.tiltTrainingViewsSummary.textContent = "Virtual teacher images used in this run.";
}

function appendTiltTrainingView(result, canvas) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.cameraId = result.id;
  button.title = `${result.id}: pitch ${result.pitchDegrees.toFixed(1)}°, yaw ${result.yawDegrees.toFixed(1)}°`;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", `${result.id} virtual teacher image`);
  const label = document.createElement("span");
  label.textContent = `${result.polarDegrees.toFixed(1)}° · ${result.multiplicity}x`;
  button.append(canvas, label);
  button.addEventListener("click", () => {
    if (!state.tilt.controller || state.tilt.teacherViewsLoading) return;
    try {
      state.tilt.cameraMarkersVisible = false;
      els.tiltCameraMarkers.checked = false;
      state.tilt.controller.setCameraMarkersVisible(false);
      const pitch = Math.round(result.pitchDegrees * 10) / 10;
      const yaw = Math.round(result.yawDegrees * 10) / 10;
      els.tiltPitch.value = String(pitch);
      els.tiltYaw.value = String(yaw);
      els.tiltPitchValue.textContent = `${pitch.toFixed(1)}\u00b0`;
      els.tiltYawValue.textContent = `${yaw.toFixed(1)}\u00b0`;
      document.documentElement.dataset.tiltPitch = String(pitch);
      document.documentElement.dataset.tiltYaw = String(yaw);
      updateTiltControlState();
      publishState();
      applyTiltInputs();
    } catch (error) {
      els.tiltStatus.textContent = `Training camera view failed: ${error.message}`;
      log(error.message);
    }
  });
  els.tiltContactSheet.append(button);
}

async function showTiltTrainingViews() {
  if (state.tilt.teacherViewsLoading) return;
  const controller = await loadTiltViewer();
  if (!controller) return;
  const pool = await tiltCameraPoolSnapshot();
  const cameras = pool.cameras.filter((camera) => camera.kind === "virtual" && camera.multiplicity > 0);
  clearTiltTrainingViews();
  state.tilt.teacherViewsLoading = true;
  els.tiltTrainingViewsProgress.max = Math.max(1, cameras.length);
  els.tiltStatus.textContent = cameras.length
    ? `Generating virtual teacher images 0 / ${cameras.length}...`
    : "This run did not use a virtual camera.";
  updateTiltControlState();
  publishState();
  try {
    for (let index = 0; index < cameras.length; index += 1) {
      const camera = cameras[index];
      const canvas = document.createElement("canvas");
      rasterizeVirtualTeacher(canvas, state.image, {
        pitchDegrees: camera.pitch_degrees,
        yawDegrees: camera.yaw_degrees,
        cameraDistance: pool.orbit_radius,
        fovDegrees: pool.fov_degrees,
      }, { maxSide: 240, outputAspect: state.image.width / state.image.height });
      const result = {
        id: camera.id,
        pitchDegrees: camera.pitch_degrees,
        yawDegrees: camera.yaw_degrees,
        polarDegrees: Number(camera.polar_degrees) || Math.hypot(camera.pitch_degrees, camera.yaw_degrees),
        multiplicity: camera.multiplicity,
      };
      state.tilt.teacherViews.push(result);
      appendTiltTrainingView(result, canvas);
      els.tiltTrainingViewsProgress.value = index + 1;
      els.tiltTrainingViewsSummary.textContent = `${index + 1} / ${cameras.length} virtual teacher images`;
      els.tiltStatus.textContent = `Generating virtual teacher images ${index + 1} / ${cameras.length}...`;
      if (index % 4 === 3) await nextFrame();
    }
    // The gallery can change the side panel scrollbar and therefore the viewer
    // size. Reconcile PlayCanvas and analytic projection after layout settles.
    await nextFrame();
    await controller.refreshCameraDiagnostics();
    els.tiltTrainingViewsSummary.textContent = cameras.length
      ? `${cameras.length} virtual teacher images used in this run`
      : "No virtual teacher images in this run";
    els.tiltStatus.textContent = cameras.length
      ? "Training views ready. Select one to inspect its camera pose."
      : "This run used only the front camera.";
  } finally {
    state.tilt.teacherViewsLoading = false;
    updateTiltControlState();
    publishState();
  }
}

function destroyTiltViewer({ restoreCanvas = false } = {}) {
  if (tiltInputFrame) {
    cancelAnimationFrame(tiltInputFrame);
    tiltInputFrame = 0;
  }
  state.tilt.teacherFrameRequest += 1;
  state.tilt.generation += 1;
  state.tilt.abortController?.abort();
  state.tilt.abortController = null;
  state.tilt.controller?.destroy?.();
  state.tilt.controller = null;
  state.tilt.revision = "";
  state.tilt.loading = false;
  state.tilt.plyDigest = "";
  state.tilt.plyByteLength = 0;
  state.tilt.vertices = 0;
  state.tilt.teacherViewsLoading = false;
  clearTiltTrainingViews();
  els.tiltFrameOverlay.hidden = true;
  els.tiltPositionValue.textContent = "-";
  els.tiltRadiusValue.textContent = "-";
  els.tiltProjectionError.textContent = "-";
  els.tiltPitch.value = "0";
  els.tiltYaw.value = "0";
  els.tiltPitchValue.textContent = "0\u00b0";
  els.tiltYawValue.textContent = "0\u00b0";
  document.documentElement.dataset.tiltPitch = "0";
  document.documentElement.dataset.tiltYaw = "0";
  if (restoreCanvas && !els.tiltCanvas.hidden) restorePrimaryCanvas();
  els.tiltStatus.textContent = tiltViewerAvailabilityMessage();
  updateTiltControlState();
}

function applyTiltInputs() {
  let pitch = Number(els.tiltPitch.value) || 0;
  let yaw = Number(els.tiltYaw.value) || 0;
  const applied = state.tilt.controller?.setTilt(pitch, yaw);
  if (applied) {
    pitch = applied.pitch;
    yaw = applied.yaw;
    els.tiltPitch.value = String(pitch);
    els.tiltYaw.value = String(yaw);
  }
  const displayedPitch = String(Object.is(Math.round(pitch * 1000) / 1000, -0) ? 0 : Math.round(pitch * 1000) / 1000);
  const displayedYaw = String(Object.is(Math.round(yaw * 1000) / 1000, -0) ? 0 : Math.round(yaw * 1000) / 1000);
  els.tiltPitchValue.textContent = `${displayedPitch}\u00b0`;
  els.tiltYawValue.textContent = `${displayedYaw}\u00b0`;
  document.documentElement.dataset.tiltPitch = displayedPitch;
  document.documentElement.dataset.tiltYaw = displayedYaw;
}

let tiltInputFrame = 0;

function scheduleTiltInputs() {
  if (tiltInputFrame) return;
  tiltInputFrame = requestAnimationFrame(() => {
    tiltInputFrame = 0;
    applyTiltInputs();
  });
}

function tiltCameraCounts(sampling) {
  const expectedSteps = Math.max(0, Math.round(Number(state.metrics?.steps_done) || 0));
  const stored = Object.fromEntries(
    Object.entries(sampling?.camera_counts || {}).map(([id, count]) => [id, Math.max(0, Number(count) || 0)]),
  );
  const storedTotal = Object.values(stored).reduce((total, count) => total + count, 0);
  if (!sampling?.enabled || storedTotal >= expectedSteps) return { counts: stored, source: "metrics" };

  const history = state.virtualCameraByStep;
  if (history instanceof Map && history.size === expectedSteps) {
    const counts = {};
    let complete = true;
    for (let step = 1; step <= expectedSteps; step += 1) {
      const cameraId = history.get(step);
      if (!cameraId) {
        complete = false;
        break;
      }
      counts[cameraId] = (counts[cameraId] || 0) + 1;
    }
    if (complete) return { counts, source: "runtime-step-history" };
  }

  const variants = {
    enabled: true,
    slots: Math.max(2, Math.round(Number(sampling.pool_slots) || DEFAULT_VIRTUAL_CAMERA_POOL_SLOTS)),
    virtualSlots: Math.max(1, Math.round(Number(sampling.virtual_slots) || DEFAULT_VIRTUAL_CAMERA_SLOTS)),
    cameraCount: Math.max(4, Math.min(MAX_VIRTUAL_CAMERA_COUNT, Math.round(Number(sampling.virtual_camera_count) || DEFAULT_VIRTUAL_CAMERA_COUNT))),
    maxAngleDegrees: Number(sampling.max_angle_degrees) || DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES,
    fovDegrees: clampSharedCameraFov(sampling.fov_degrees),
    seed: Math.max(0, Math.floor(
      Number.isFinite(Number(sampling.seed)) ? Number(sampling.seed) : DEFAULT_VIRTUAL_CAMERA_SEED,
    )) >>> 0,
    cameraDistance: Number(sampling.orbit_radius) || DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE,
    autoCameraDistance: true,
    orderPenaltyWeight: 0,
    planeConstrained: true,
    invalidRegionMode: sampling.invalid_region_mode || "mask",
  };
  const warmupSteps = Math.max(0, Math.round(Number(sampling.warmup_steps) || 0));
  const counts = {};
  for (let step = 1; step <= expectedSteps; step += 1) {
    const stage = step <= warmupSteps ? "coarse" : "full";
    const camera = virtualCameraSamplingStepSpec(step, stage, expectedSteps, warmupSteps, variants);
    counts[camera.cameraId] = (counts[camera.cameraId] || 0) + 1;
  }
  return { counts, source: "deterministic-schedule" };
}

async function tiltCameraPoolSnapshot() {
  const sampling = state.metrics?.virtual_camera_sampling;
  const fovDegrees = clampSharedCameraFov(sampling?.fov_degrees);
  const radius = Number.isFinite(Number(sampling?.orbit_radius))
    ? Number(sampling.orbit_radius)
    : sharedTiltOrbitRadius(
      state.image.width,
      state.image.height,
      sampling?.max_angle_degrees,
      49,
      fovDegrees,
    );
  const cameraHistory = tiltCameraCounts(sampling);
  const cameras = sampling?.enabled
    ? sampling.cameras
      .map((camera) => ({
        ...camera,
        multiplicity: Math.max(0, Number(cameraHistory.counts[camera.id]) || 0),
      }))
      .filter((camera) => camera.multiplicity > 0)
    : [{
      id: "front",
      kind: "front",
      pitch_degrees: 0,
      yaw_degrees: 0,
      intrinsics: virtualFrontIntrinsics(state.image.width, state.image.height, fovDegrees),
    }];
  return {
    cameras: structuredClone(cameras),
    sampling_enabled: Boolean(sampling?.enabled),
    camera_counts_source: cameraHistory.source,
    active_camera_id: sampling?.enabled ? sampling.active_camera_id : "front",
    orbit_radius: radius,
    max_angle_degrees: Math.max(5, Math.min(
      MAX_VIRTUAL_CAMERA_ANGLE_DEGREES,
      Number(sampling?.max_angle_degrees) || DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES,
    )),
    fov_degrees: fovDegrees,
    target: Array.isArray(sampling?.target) ? [...sampling.target] : [0, 0, 0],
    image_aspect: state.image.width / state.image.height,
  };
}

async function loadTiltViewer({ force = false } = {}) {
  if (!tiltViewerReady()) throw new Error("Finish training before opening the Tilt viewer.");
  const revision = currentTiltRevision();
  if (!force && state.tilt.controller && state.tilt.revision === revision) {
    showCanvas("tilt");
    applyTiltInputs();
    return state.tilt.controller;
  }

  destroyTiltViewer();
  const generation = state.tilt.generation + 1;
  state.tilt.generation = generation;
  const abortController = new AbortController();
  state.tilt.abortController = abortController;
  state.tilt.loading = true;
  els.tiltStatus.textContent = "Building the PlayCanvas PLY view...";
  showCanvas("tilt");
  updateTiltControlState();
  try {
    const adjusted = await materializeCurrentSplatAdjustmentSnapshot();
    if (!adjusted || revision !== currentTiltRevision()) {
      if (generation === state.tilt.generation) {
        state.tilt.abortController = null;
        state.tilt.loading = false;
        els.tiltStatus.textContent = "Tilt source changed; reopen Tilt to use the latest splats.";
        updateTiltControlState();
      }
      return null;
    }
    const tiltParams = adjusted.params;
    assertTiltViewerCapacity(tiltParams, state.image);
    const plyBuffer = makePly(tiltParams, state.image);
    const cameraPool = await tiltCameraPoolSnapshot();
    const { createTiltViewer } = tiltViewerRuntime();
    const controller = await createTiltViewer({
      canvas: els.tiltCanvas,
      plyBuffer,
      frame: plyFrameScale(state.image),
      supportFrame: renderFootprintSupportFrame(state.image, tiltParams),
      signal: abortController.signal,
      onCameraChange: updateTiltCameraDiagnostics,
      cameraPool,
    });
    if (
      generation !== state.tilt.generation ||
      !splatAdjustmentSnapshotIsCurrent(adjusted.snapshot) ||
      revision !== currentTiltRevision()
    ) {
      controller.destroy();
      if (generation === state.tilt.generation) {
        state.tilt.abortController = null;
        state.tilt.loading = false;
        els.tiltStatus.textContent = "Tilt source changed; reopen Tilt to use the latest splats.";
        updateTiltControlState();
      }
      return null;
    }
    state.tilt.controller = controller;
    state.tilt.abortController = null;
    state.tilt.revision = revision;
    state.tilt.loading = false;
    state.tilt.plyDigest = controller.plyDigest;
    state.tilt.plyByteLength = controller.plyByteLength;
    state.tilt.vertices = controller.vertices;
    state.tilt.verifiedRevision = revision;
    state.tilt.verifiedPlyDigest = controller.plyDigest;
    state.tilt.verifiedPlyByteLength = controller.plyByteLength;
    controller.setCameraMarkersVisible(state.tilt.cameraMarkersVisible);
    controller.setRadiusMode(state.tilt.radiusMode);
    applyTiltInputs();
    setTiltDisplayMode(state.tilt.viewMode);
    els.tiltStatus.textContent = `PlayCanvas ${controller.engineVersion} (${controller.backend}). The camera orbits the image center at a fixed radius.`;
    eventLog(
      `Tilt viewer loaded from memory PLY: ${tiltParams.count} splats` +
      ` sha256=${controller.plyDigest.slice(0, 12)}`,
    );
    publishState();
    return controller;
  } catch (error) {
    if (error?.name === "AbortError") return null;
    if (generation === state.tilt.generation) {
      state.tilt.loading = false;
      restorePrimaryCanvas();
      els.tiltStatus.textContent = `Tilt viewer failed: ${error.message}`;
      updateTiltControlState();
    }
    throw error;
  }
}

