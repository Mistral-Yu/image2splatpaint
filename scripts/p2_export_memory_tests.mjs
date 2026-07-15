import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { resolve } from "node:path";

const MB = 1024 * 1024;
const GB = 1024 * MB;
const root = resolve(import.meta.dirname, "..");
const app = await readFile(resolve(root, "web/app.js"), "utf8");
const start = app.indexOf("function exportPeakMemoryPlan(");
const end = app.indexOf("function assertPlyExportCapacity(", start);
assert.ok(start >= 0 && end > start, "pure export peak planner must be present");
const context = { Math, Number, bytesToMB: (bytes) => bytes / MB, MB };
vm.createContext(context);
vm.runInContext(app.slice(start, end), context);

const count = 1_000_000;
const fixture = {
  count,
  parameterBytes: count * 44,
  baselineBytes: count * 44,
  imageBytes: 1600 * 1066 * 16,
  readbackBytes: count * 40,
  plyBytes: 1024 + count * 68,
  blobCopyBytes: 1024 + count * 68,
};
fixture.residentBytes = fixture.parameterBytes + fixture.baselineBytes + fixture.imageBytes;

const ample = context.exportPeakMemoryPlan({
  ...fixture,
  memoryInfo: { source: "test heap", budgetBytes: 2 * GB, usedBytes: 1 * GB, exactFree: true },
});
assert.equal(ample.ok, true);
assert.equal(ample.count, count);
assert.equal(ample.requiredIncrementBytes, fixture.plyBytes + fixture.blobCopyBytes);

const constrained = context.exportPeakMemoryPlan({
  ...fixture,
  memoryInfo: { source: "test heap", budgetBytes: 2 * GB, usedBytes: 1.9 * GB, exactFree: true },
});
assert.equal(constrained.ok, false);

const fallback = context.exportPeakMemoryPlan({
  ...fixture,
  memoryInfo: { source: "fallback", budgetBytes: 256 * MB, usedBytes: 0, exactFree: false },
});
assert.equal(fallback.ok, false);

const panel = app.slice(app.indexOf("function updateExportPanel()"), app.indexOf("function exportCoverageStatus("));
assert.ok(panel.includes("els.savePngButton.disabled = !enabled"));
assert.ok(panel.includes("els.savePlyButton.disabled = !plyEnabled"));
assert.ok(panel.includes("data.plyExportReady"));

const save = app.slice(app.indexOf("async function saveExport("), app.indexOf("function inspectPlyContract("));
assert.ok(save.indexOf("assertPlyExportCapacity") < save.indexOf("const plyBuffer = makePly"));
const preview = app.slice(app.indexOf("function previewInvariantSnapshot()"), app.indexOf("async function runPreviewRefreshLoop()"));
assert.equal(preview.includes("makePly("), false);

console.log(JSON.stringify({
  ok: true,
  million_fixture: {
    estimated_peak_mb: ample.estimatedPeakMB,
    required_increment_mb: (ample.requiredIncrementBytes / MB).toFixed(1),
    ample: ample.ok,
    constrained: constrained.ok,
    conservative_fallback: fallback.ok,
  },
}, null, 2));
