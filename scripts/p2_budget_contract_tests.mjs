import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const app = await readFile(resolve(root, "web/app.js"), "utf8");
const start = app.indexOf("function trainingBufferDescriptors(");
const end = app.indexOf("function capacityProbeCandidates(", start);
assert.ok(start >= 0 && end > start, "training allocation planner source must be present");

let budgetBytes = 1024 ** 3;
const context = {
  Math,
  Object,
  Number,
  state: { webgpu: { renderer: { performanceProfile: { timestampQuery: false } }, limits: {} } },
  TILE_SIZE: 16,
  DENSITY_EVENT_SLOTS: 16,
  PHASE45_REGION_COUNT: 4,
  PHASE45_REGION_STRIDE: 8,
  EXACT_GRADIENT_STRIDE: 16,
  EXACT_BACKWARD_TELEMETRY_BYTES: 96,
  PERFORMANCE_PROFILE_QUERY_CAPACITY: 16,
  TRAIN_CONFIG_FLOATS: 84,
  TRAIN_CONFIG_BYTES: 84 * 4,
  ALPHA_STATE_BYTES_PER_PIXEL: 16,
  MB: 1024 * 1024,
  phase33Variants: () => ({}),
  makeCurriculumImages: () => ({ coarseImage: null, midImage: null }),
  residualTileControlWords: () => 0,
  plannedTileIndexCapacity: () => ({ capacity: 1000 }),
  limitNumber: (limits, name, fallback) => Number.isFinite(Number(limits?.[name])) ? Number(limits[name]) : fallback,
  memoryBudgetBytes: () => budgetBytes,
};
vm.createContext(context);
vm.runInContext(app.slice(start, end), context);

const image = {
  width: 100,
  height: 50,
  rgb: new Float32Array(100 * 50 * 3),
  alpha: new Float32Array(100 * 50),
};
const params = { count: 256 };
const prepared = {
  tilePlan: { capacity: 1000 },
  coarseImage: null,
  midImage: null,
};
const generousDevice = {
  limits: {
    maxBufferSize: 1024 ** 3,
    maxStorageBufferBindingSize: 1024 ** 3,
  },
};

const descriptorPlan = context.trainingBufferDescriptors(image, params, 256, generousDevice, prepared);
const descriptor = (name) => descriptorPlan.descriptors.find((item) => item.name === name);
assert.equal(descriptor("alpha-state").size, image.width * image.height * 16);
assert.equal(descriptor("packed-stats-config"), undefined);
assert.equal(descriptor("loss-gradient").size, image.width * image.height * 48);
assert.ok(app.includes("largestPixelStorageBytes"));
assert.ok(app.includes("current.lossGradientBytes"));
assert.ok(app.includes("overHardLimit: largestPixelStorageBytes > hardBufferBytes"));

assert.deepEqual(
  JSON.parse(JSON.stringify(context.allocationDescriptorMismatch(
    [{ name: "a", size: 4 }, { name: "b", size: 8 }],
    [{ name: "b", size: 8 }, { name: "a", size: 4 }],
  ))),
  [],
);
assert.deepEqual(
  JSON.parse(JSON.stringify(context.allocationDescriptorMismatch(
    [{ name: "a", size: 4 }],
    [{ name: "a", size: 8 }, { name: "extra", size: 4 }],
  ))),
  [
    { name: "a", planned: 4, allocated: 8 },
    { name: "extra", planned: null, allocated: 4 },
  ],
);

const fakeState = {
  configBuffer: { size: 336 },
  presentConfigBuffer: { size: 16 },
  alphaStateBuffer: { size: 80000 },
  xyBuffers: [{ size: 4096 }],
  transformBuffers: [{ size: 4096 }],
  colorBuffers: [{ size: 4096 }],
};
assert.deepEqual(
  JSON.parse(JSON.stringify(context.trainStateAllocatedDescriptors(fakeState))),
  [
    { name: "config", size: 336 },
    { name: "present-config", size: 16 },
    { name: "alpha-state", size: 80000 },
    { name: "xy-depth", size: 4096 },
    { name: "transform", size: 4096 },
    { name: "color", size: 4096 },
  ],
);

context.state.webgpu.renderer.performanceProfile.timestampQuery = true;
const profiledPlan = context.trainingBufferDescriptors(image, params, 300000, generousDevice, prepared);
for (const name of ["exact-backward-telemetry", "exact-backward-telemetry-readback", "profile-resolve", "profile-readback"]) {
  assert.ok(profiledPlan.descriptors.some((item) => item.name === name), `${name} missing from profiled plan`);
}
assert.equal(profiledPlan.descriptors.find((item) => item.name === "exact-backward-telemetry").size, 96);
assert.equal(new Set(profiledPlan.descriptors.map((item) => item.name)).size, profiledPlan.descriptors.length);
context.state.webgpu.renderer.performanceProfile.timestampQuery = false;

const generousPlan = context.trainingAllocationPlan(image, params, 256, generousDevice, prepared);
assert.equal(generousPlan.withinBufferLimits, true);
assert.equal(generousPlan.withinBudget, true);

budgetBytes = generousPlan.reservedBytes / 0.9;
assert.equal(context.trainingAllocationPlan(image, params, 256, generousDevice, prepared).withinBudget, true);
budgetBytes = (generousPlan.reservedBytes - 1) / 0.9;
assert.equal(context.trainingAllocationPlan(image, params, 256, generousDevice, prepared).withinBudget, false);

budgetBytes = 1024 ** 3;
const largestStorage = generousPlan.largestStorageBytes;
const undersizedDevice = {
  limits: {
    maxBufferSize: 1024 ** 3,
    maxStorageBufferBindingSize: largestStorage - 1,
  },
};
assert.equal(context.trainingAllocationPlan(image, params, 256, undersizedDevice, prepared).withinBufferLimits, false);

const transientRejected = context.tileGrowthMemoryPlan({
  currentReservedBytes: 100,
  currentTileBytes: 20,
  nextTileBytes: 40,
  budgetBytes: 130,
});
assert.equal(transientRejected.finalReservedBytes, 120);
assert.equal(transientRejected.transientReservedBytes, 140);
assert.equal(transientRejected.withinBudget, false);
assert.equal(context.tileGrowthMemoryPlan({
  currentReservedBytes: 100,
  currentTileBytes: 20,
  nextTileBytes: 40,
  budgetBytes: 140,
}).withinBudget, true);
assert.equal(context.tileGrowthMemoryPlan({
  currentReservedBytes: 100,
  currentTileBytes: 20,
  nextTileBytes: 70,
  budgetBytes: 140,
}).withinBudget, false);

const uploadStart = app.indexOf("async uploadTrainState(");
const uploadEnd = app.indexOf("async probeTrainingCapacity(", uploadStart);
const upload = app.slice(uploadStart, uploadEnd);
assert.ok(upload.indexOf("trainingAllocationPlan(") < upload.indexOf("this.disposeTrainState()"));
assert.ok(upload.indexOf("if (!allocationPlan.withinBudget)") < upload.indexOf("createBuffer: (descriptor)"));
assert.ok(upload.includes("GPU allocation descriptor drift"));
assert.ok(upload.includes("trainStateAllocatedDescriptors(candidate)"));
assert.ok(upload.includes("allocationDescriptorMismatch(allocationPlan.descriptors, allocatedDescriptors)"));
assert.ok(app.includes("const ALPHA_STATE_BYTES_PER_PIXEL = 16"));
assert.ok(app.includes("const alphaStateBytes = pixels * ALPHA_STATE_BYTES_PER_PIXEL"));
assert.ok(app.includes("image.width * image.height * ALPHA_STATE_BYTES_PER_PIXEL"));
assert.ok(app.includes("actual.reservedBytes > memoryBudgetBytes() * 0.9"));
assert.ok(app.includes("REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 8"));
assert.ok(app.includes("PREFERRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 9"));
assert.ok(app.includes("Math.min(adapterStorageBuffers, PREFERRED_STORAGE_BUFFERS_PER_SHADER_STAGE)"));
assert.ok(app.includes("maxStorageBuffersPerShaderStage: Math.min(adapterStorageBuffers, PREFERRED_STORAGE_BUFFERS_PER_SHADER_STAGE)"));
assert.ok(app.includes("adapterStorageBuffers < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE"));
const diagnosticsStart = app.indexOf("async computeOverlapDiagnostics(");
const diagnosticsEnd = app.indexOf("async computeObliqueDiagnostics(", diagnosticsStart);
const diagnostics = app.slice(diagnosticsStart, diagnosticsEnd);
assert.ok(diagnostics.includes("let operationError = null;"));
assert.ok(diagnostics.indexOf("configBuffer.destroy()") > diagnostics.indexOf("finally {", diagnostics.indexOf("finally {") + 1));
assert.ok(diagnostics.includes("if (operationError)"));
assert.ok(diagnostics.includes("throw restorationError"));
const growthStart = app.indexOf("async growTileIndexCapacity(");
const growthEnd = app.indexOf("currentTrainBuffers(", growthStart);
const growth = app.slice(growthStart, growthEnd);
assert.ok(growth.indexOf("tileGrowthMemoryPlan(") < growth.indexOf("this.device.createBuffer("));
assert.ok(growth.includes("if (!growthPlan.withinBudget)"));
assert.ok(growth.includes("tile_growth_budget_rejection"));

console.log(JSON.stringify({
  ok: true,
  reserved_bytes: generousPlan.reservedBytes,
  largest_storage_bytes: largestStorage,
  descriptors: descriptorPlan.descriptors.length,
}, null, 2));
