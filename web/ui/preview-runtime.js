function hashBytes(bytes, seed = 2166136261) {
  let hash = seed >>> 0;
  for (let i = 0; i < bytes.byteLength; i += 1) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function hashParams(params = state.params) {
  let hash = 2166136261;
  for (const values of [params?.xy, params?.scale, params?.rgb, params?.opacity, params?.theta, params?.depthOrder]) {
    if (!values) continue;
    hash = hashBytes(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), hash);
  }
  return hash >>> 0;
}

function previewInvariantSnapshot() {
  if (!state.params) return null;
  const metrics = state.metrics;
  const hashEnabled = previewInvariantHashEnabled();
  const paramsHash = hashEnabled ? hashParams() : null;
  const plySignature = new TextEncoder().encode([
    paramsHash ?? "deferred",
    state.params.count,
    state.image?.width || 0,
    state.image?.height || 0,
    state.params.boundarySigma,
    Boolean(state.params.layerOrderEnabled),
    PLY_LAYER_DEPTH_SPAN,
  ].join("|"));
  return {
    params_hash: paramsHash,
    ply_hash: hashBytes(plySignature),
    hash_mode: hashEnabled ? "full-qa" : "revision-only",
    splats: state.params.count,
    steps: metrics?.steps_done ?? 0,
    params_revision: metrics?.params_revision ?? 0,
    l1: metrics?.final_l1 ?? null,
    psnr: metrics?.final_psnr ?? null,
    global_ssim: metrics?.final_global_ssim ?? null,
    windowed_ssim: metrics?.final_windowed_ssim ?? null,
    regional_p10: metrics?.final_regional_ssim?.p10 ?? null,
  };
}

async function runPreviewRefreshLoop() {
  const refreshEpoch = state.previewRefreshEpoch;
  let rendered = false;
  state.previewRefreshPending = true;
  publishState();
  try {
    while (state.previewAppliedRevision < state.previewRequestedRevision) {
      if (refreshEpoch !== state.previewRefreshEpoch) return false;
      const requestedRevision = state.previewRequestedRevision;
      if (state.running) {
        state.previewPadding = previewPaddingSpec(state.image, state.params, false);
        state.previewAppliedAlphaBackground = "";
        state.webgpu.renderer?.presentTrainState(state.image);
        if (state.previewMode === "splats") showCanvas("gpu");
        state.previewAppliedRevision = requestedRevision;
        rendered = true;
        break;
      }
      const renderer = state.webgpu.renderer;
      if (!state.image || !state.params || !renderer || renderer.deviceLost) {
        state.previewPadding = previewPaddingSpec(state.image, state.params, false);
        state.previewAppliedAlphaBackground = "";
        state.previewAppliedRevision = requestedRevision;
        break;
      }

      const generation = state.previewGeneration;
      const image = state.image;
      const params = state.params;
      const displayParams = params;
      const before = previewInvariantSnapshot();
      const buffers = renderer.currentTrainBuffers(params) || renderer.currentResultBuffers(params);
      const splatOptionsActive =
        document.documentElement.dataset.activeDetailTab === "splats" &&
        state.previewMode === "splats";
      const alphaOptions = splatOptionsActive ? splatAlphaRenderOptions() : {};
      const appliedAlphaBackground = splatOptionsActive ? els.splatAlphaBackground.value.toLowerCase() : "";
      try {
        await renderer.render(image, displayParams, buffers, null, alphaOptions);
      } catch (error) {
        if (refreshEpoch !== state.previewRefreshEpoch || isExpectedPreviewCancellation(error)) return rendered;
        throw error;
      }
      if (
        refreshEpoch !== state.previewRefreshEpoch ||
        generation !== state.previewGeneration ||
        renderer !== state.webgpu.renderer ||
        renderer.deviceLost ||
        state.running ||
        image !== state.image ||
        params !== state.params
      ) {
        continue;
      }

      const after = previewInvariantSnapshot();
      const invariant = JSON.stringify(before) === JSON.stringify(after);
      if (state.metrics) {
        state.metrics.preview_only_contract = {
          invariant,
          before,
          after,
          padding: { ...state.previewPadding },
        };
      }
      document.documentElement.dataset.previewContractInvariant = String(invariant);
      document.documentElement.dataset.previewContractParamsHash = String(after?.params_hash ?? "");
      document.documentElement.dataset.previewContractPlyHash = String(after?.ply_hash ?? "");
      if (!invariant) throw new Error("preview-only contract changed training parameters, metrics, or PLY payload");
      if (state.previewMode === "splats") showCanvas("gpu");
      state.previewAppliedAlphaBackground = appliedAlphaBackground;
      state.previewAppliedRevision = requestedRevision;
      rendered = true;
    }
    return rendered;
  } finally {
    if (refreshEpoch === state.previewRefreshEpoch) {
      state.previewRefreshPending = false;
      publishState();
    }
  }
}

function invalidatePreviewRefresh() {
  state.previewGeneration += 1;
  state.previewRefreshEpoch += 1;
  state.previewRequestedRevision += 1;
  state.previewAppliedRevision = state.previewRequestedRevision;
  state.previewRefreshPending = false;
  state.previewRefreshPromise = Promise.resolve(false);
}

async function refreshOutsidePreview({ recovery = false } = {}) {
  if (state.webGpuRecoveryPending && !recovery) return false;
  state.previewRequestedRevision += 1;
  if (state.running) {
    state.previewPadding = previewPaddingSpec(state.image, state.params, false);
    state.previewAppliedAlphaBackground = "";
    state.webgpu.renderer?.presentTrainState(state.image);
    if (state.previewMode === "splats") showCanvas("gpu");
    state.previewAppliedRevision = state.previewRequestedRevision;
    publishState();
    return false;
  }
  if (!state.image || !state.params || !state.webgpu.renderer) {
    state.previewPadding = previewPaddingSpec(state.image, state.params, false);
    state.previewAppliedAlphaBackground = "";
    state.previewAppliedRevision = state.previewRequestedRevision;
    publishState();
    return false;
  }
  if (!state.previewRefreshPending) {
    state.previewRefreshPromise = runPreviewRefreshLoop();
  }
  publishState();
  return state.previewRefreshPromise;
}

function setImageDragover(active) {
  els.dropZone.classList.toggle("dragover", active && !state.image);
  els.viewer.classList.toggle("image-dragover", active && Boolean(state.image));
}


