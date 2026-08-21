document.documentElement.dataset.algorithm = selectedAlgorithm().id;
// Keep the existing desktop-first status view while saving vertical space on
// phones. This is an initial default only; later resizes preserve user choice.
if (window.matchMedia("(max-width: 520px)").matches) {
  els.trainingStatusDetails.open = false;
}
activateDetailTab("training");
syncLayerOrderDependency();
syncVirtualCameraDependency();
updatePreviewModeControls();
resetSplatAdjustmentControls();
renderSplatInspector();

window.addEventListener("resize", applyCanvasView);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.running && !state.paused) {
    state.paused = true;
    state.visibilityPaused = true;
    setPausedRuntimeControlsEnabled(true);
    setStatus("paused");
    eventLog("training paused because the page became hidden");
  } else if (!document.hidden && state.visibilityPaused) {
    state.visibilityPaused = false;
    eventLog("page visible; training remains paused until Resume");
  }
  publishState();
});
window.addEventListener("pagehide", () => destroyTiltViewer());

if (QA_RUNTIME_ENABLED) {
  window.__image2SplatPaint = {
    snapshot() {
      return { ...document.documentElement.dataset };
    },
  };
  window.__image2GaussianPaint = window.__image2SplatPaint;
}

if (QA_RUNTIME_ENABLED && new URLSearchParams(location.search).has("qa")) {
  els.pathInput.type = "text";
  els.pathInput.hidden = false;
  els.pathInput.setAttribute("aria-label", "QA image path");
  els.pathInput.style.cssText = "grid-column:1/-1;width:100%;";
  els.pathButton.hidden = false;
  els.pathButton.textContent = "Load QA path";
  const qaMetricsData = document.createElement("textarea");
  qaMetricsData.id = "qaMetricsData";
  qaMetricsData.dataset.testid = "qa-metrics-data";
  qaMetricsData.hidden = true;
  const qaMetricsButton = document.createElement("button");
  qaMetricsButton.id = "qaMetricsButton";
  qaMetricsButton.dataset.testid = "qa-metrics-button";
  qaMetricsButton.type = "button";
  qaMetricsButton.textContent = "QA Metrics";
  qaMetricsButton.style.cssText =
    "position:fixed;left:16px;bottom:80px;z-index:20;width:96px;height:30px;font-size:12px;opacity:0.35;";
  qaMetricsButton.addEventListener("click", () => {
    try {
      qaMetricsData.value = JSON.stringify({
        snapshot: window.__image2SplatPaint.snapshot(),
        metrics: window.__flatPhotoTest.metricsSummary(),
        benchmark: window.__flatPhotoTest.benchmarkSummary(),
        ply_contract: state.params ? inspectPlyContract() : null,
        edge_containment_probe: window.__flatPhotoTest.edgeContainmentProbe(),
      });
      document.documentElement.dataset.qaMetricsBytes = String(qaMetricsData.value.length);
      document.documentElement.dataset.qaMetricsError = "";
    } catch (error) {
      qaMetricsData.value = "";
      document.documentElement.dataset.qaMetricsBytes = "";
      document.documentElement.dataset.qaMetricsError = error.message;
    }
  });
  const qaObliqueButton = document.createElement("button");
  qaObliqueButton.id = "qaObliqueButton";
  qaObliqueButton.dataset.testid = "qa-oblique-button";
  qaObliqueButton.type = "button";
  qaObliqueButton.textContent = "QA Oblique";
  qaObliqueButton.style.cssText =
    "position:fixed;left:116px;bottom:80px;z-index:20;width:96px;height:30px;font-size:12px;opacity:0.35;";
  qaObliqueButton.addEventListener("click", async () => {
    qaObliqueButton.disabled = true;
    document.documentElement.dataset.qaObliqueStatus = "running";
    try {
      const report = state.metrics?.oblique_overlap_diagnostics || (
        state.image && state.params && state.webgpu.renderer?.trainState
          ? await state.webgpu.renderer.computeObliqueDiagnostics(state.image, state.params)
          : null
      );
      if (!report) throw new Error("Run a QA virtual-camera training before measuring oblique overlap.");
      if (state.metrics) state.metrics.oblique_overlap_diagnostics = report;
      qaMetricsData.value = JSON.stringify(report);
      document.documentElement.dataset.qaObliqueStatus = "complete";
      document.documentElement.dataset.qaObliqueBytes = String(qaMetricsData.value.length);
      document.documentElement.dataset.qaObliqueError = "";
    } catch (error) {
      qaMetricsData.value = "";
      document.documentElement.dataset.qaObliqueStatus = "failed";
      document.documentElement.dataset.qaObliqueError = error.message;
    } finally {
      qaObliqueButton.disabled = false;
    }
  });
  const qaDeviceLossButton = document.createElement("button");
  qaDeviceLossButton.id = "qaDeviceLossButton";
  qaDeviceLossButton.dataset.testid = "qa-device-loss-button";
  qaDeviceLossButton.type = "button";
  qaDeviceLossButton.textContent = "QA Device Loss";
  qaDeviceLossButton.style.cssText =
    "position:fixed;left:216px;bottom:80px;z-index:20;width:110px;height:30px;font-size:12px;opacity:0.35;";
  qaDeviceLossButton.addEventListener("click", () => {
    if (!state.webgpu.renderer?.device) return;
    state.webgpu.renderer.device.destroy();
  });
  document.body.append(qaMetricsData, qaMetricsButton, qaObliqueButton, qaDeviceLossButton);
}

// Compatibility aliases exist only for local QA scripts and checkpoints.
if (QA_RUNTIME_ENABLED) window.__flatPhoto3dgs = window.__image2SplatPaint;

detectWebGpu()
  .then((available) => setStatus(available ? "idle" : "gpu unavailable"))
  .catch((error) => {
    state.webgpu = { supported: false, renderer: null, reason: error.message, limits: null, adapterInfo: null };
    els.backendText.textContent = "webgpu unavailable";
    setStatus("gpu unavailable");
    log(`webgpu check failed: ${error.message}`);
  })
  .finally(() => {
    syncTrainSizeUi();
    updateMemoryRecommendation();
    publishState();
  });
