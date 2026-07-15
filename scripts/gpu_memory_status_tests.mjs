import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const [app, html, css] = await Promise.all([
  readFile(resolve(root, "web/app.js"), "utf8"),
  readFile(resolve(root, "web/index.html"), "utf8"),
  readFile(resolve(root, "web/styles.css"), "utf8"),
]);

const checks = {
  status_fields:
    html.includes('data-testid="gpu-memory-text"') &&
    html.includes("VRAM active / reserved") &&
    html.includes("This is not system-wide, free, or total device VRAM"),
  playcanvas_style_accounting:
    app.includes("trainingBuffers()") &&
    app.includes("trainingMemorySnapshot()") &&
    app.includes("buffer.size") &&
    app.includes("app-created-buffers"),
  active_estimate:
    app.includes("this.trainState.count / Math.max(1, this.trainState.capacity)") &&
    app.includes("observedTileReferences") &&
    app.includes("activeTileBytes"),
  train_time_allocation:
    app.includes("state.webgpu.renderer.uploadTrainState(") &&
    app.includes("probeTrainingCapacity(state.image, state.params, finalCount)") &&
    app.includes("GPU training buffers reserved"),
  release_after_training:
    app.includes("reservedBeforeRelease") &&
    app.includes("GPU training buffers released") &&
    app.includes("this.trainState = null;\n    updateGpuMemoryStatus();"),
  machine_readable_state:
    app.includes("data.gpuActiveBytes") &&
    app.includes("data.gpuReservedBytes") &&
    app.includes('data.vramExact = "false"'),
  two_row_status_layout:
    css.includes("repeat(5, minmax(0, 1fr))") &&
    css.includes(".progress div:nth-child(10)"),
  no_dummy_reservation:
    !app.includes("dummyVram") &&
    !app.includes("reserveVramBuffer"),
};

const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => name);
if (failed.length) throw new Error(`GPU memory status checks failed: ${failed.join(", ")}`);
console.log(JSON.stringify({ ok: true, checks }, null, 2));
