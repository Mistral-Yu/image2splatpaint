import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const [app, html] = await Promise.all([
  readFile(resolve(root, "web/app.js"), "utf8"),
  readFile(resolve(root, "web/index.html"), "utf8"),
]);

const checks = {
  million_cap:
    app.includes("const MANUAL_SPLATS_MAX = 1048576") &&
    html.includes('id="finalSplatCount"') &&
    html.includes('max="1048576"'),
  explicit_mode:
    html.includes('value="manual" selected') &&
    html.includes('value="auto-probe"') &&
    html.includes('GPU memory mode') &&
    html.includes('>Use Max splats</option>') &&
    html.includes('>Auto by GPU memory</option>') &&
    app.includes('els.capacityMode.value === "auto-probe"'),
  capacity_control_order:
    html.indexOf('data-testid="lr-panel"') < html.indexOf('data-testid="virtual-camera-panel"') &&
    html.indexOf('data-testid="virtual-camera-panel"') < html.indexOf('data-testid="budget-panel"') &&
    html.indexOf('data-testid="budget-panel"') < html.indexOf('data-testid="capacity-mode"') &&
    html.indexOf('data-testid="capacity-mode"') < html.indexOf('<div class="budget-grid">') &&
    !html.slice(html.indexOf('<div class="training-grid">'), html.indexOf('</div>', html.indexOf('<div class="training-grid">'))).includes('data-testid="capacity-mode"'),
  low_capacity_fast_path:
    app.includes("const CAPACITY_PROBE_FAST_PATH_MAX = 262144") &&
    app.includes("finalCount <= CAPACITY_PROBE_FAST_PATH_MAX") &&
    app.includes("{ verifyAllocation: finalCount > CAPACITY_PROBE_FAST_PATH_MAX }"),
  candidate_tiers:
    app.includes("[262144, 524288, 786432, 1048576]") &&
    app.includes("capacityProbeCandidates(requestedCapacity)"),
  transactional_allocation:
    app.includes("const allocatedResources = []") &&
    app.includes("allocatedResources.push(resource)") &&
    app.includes("allocatedResources.length = 0") &&
    app.includes("for (const resource of allocatedResources) resource?.destroy?.()"),
  oom_and_validation_scopes:
    app.includes('pushErrorScope("out-of-memory")') &&
    app.includes('pushErrorScope("validation")') &&
    app.includes("await this.device.queue.onSubmittedWorkDone()"),
  candidate_reused:
    app.includes("this.trainState = candidate") &&
    app.includes("return { capacity, attempts, plan, fastPath: false }"),
  per_buffer_and_budget_gate:
    app.includes("largestStorageBytes <= maxStorage") &&
    app.includes("largestBufferBytes <= maxBuffer") &&
    app.includes("reservedBytes <= memoryBudgetBytes() * 0.9"),
  atomic_tile_growth:
    app.indexOf("const validationError = await this.device.popErrorScope()", app.indexOf("async growTileIndexCapacity")) <
      app.indexOf("this.trainState.tileIndicesBuffer = nextBuffer", app.indexOf("async growTileIndexCapacity")) &&
    app.includes("reserveRatio >= 0.8") &&
    app.includes("reserveRatio >= 0.9"),
  device_loss_terminal:
    app.includes("if (state.webgpu.renderer !== renderer) return") &&
    app.includes("state.stopRequested = true") &&
    app.includes("state.paused = false") &&
    html.includes('data-testid="retry-webgpu"') &&
    html.includes('data-testid="retry-webgpu" type="button" hidden') &&
    app.includes("els.retryWebGpuButton.hidden = state.webgpu.supported"),
  read_only_memory_details:
    html.includes("GPU memory details") &&
    !html.includes('data-testid="apply-budget"') &&
    !html.includes('data-testid="memory-budget"') &&
    !app.includes("applyRecommendation"),
  honest_status:
    html.includes('data-testid="capacity-probe"') &&
    html.includes('<span>Splat capacity</span>') &&
    app.includes('"Use Max splats"') &&
    app.includes('"Testing GPU capacity..."') &&
    app.includes('"Auto · fast path"') &&
    html.includes('data-testid="tile-reserve"') &&
    html.includes('data-testid="measured-speed"'),
};

const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => name);
if (failed.length) throw new Error(`adaptive capacity checks failed: ${failed.join(", ")}`);
console.log(JSON.stringify({ ok: true, checks }, null, 2));
