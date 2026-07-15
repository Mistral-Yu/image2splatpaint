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
  PACKED_OPTIMIZER_STRIDE_BYTES: 128,
  PERFORMANCE_PROFILE_QUERY_CAPACITY: 16,
  MB: 1024 * 1024,
  phase33Variants: () => ({}),
  makeCurriculumImages: () => ({ coarseImage: null, midImage: null }),
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
  geometrySupport: { supported: false, bytes: 0 },
};
const generousDevice = {
  limits: {
    maxBufferSize: 1024 ** 3,
    maxStorageBufferBindingSize: 1024 ** 3,
  },
};

const descriptorPlan = context.trainingBufferDescriptors(image, params, 256, generousDevice, prepared);
const descriptor = (name) => descriptorPlan.descriptors.find((item) => item.name === name);
assert.equal(descriptor("alpha-state").size, image.width * image.height * 8);
assert.equal(descriptor("loss-gradient").size, image.width * image.height * 48);

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
assert.ok(app.includes("actual.reservedBytes > memoryBudgetBytes() * 0.9"));
assert.ok(app.includes("REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 9"));
assert.ok(app.includes("maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE"));
assert.ok(app.includes("adapterStorageBuffers < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE"));
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
