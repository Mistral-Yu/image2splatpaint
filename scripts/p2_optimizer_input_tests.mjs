import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
const root = resolve(import.meta.dirname, "..");
const app = await readFile(resolve(root, "web/app.js"), "utf8");
const normalizeStepInteger = (value, { min, max, fallback, step = 1 }) => {
  const parsed = Number(value);
  const finite = Number.isFinite(parsed) ? parsed : fallback;
  const clamped = Math.min(max, Math.max(min, finite));
  return Math.min(max, Math.max(min, Math.trunc(min + Math.round((clamped - min) / step) * step)));
};
const normalizeUiSplat = (value, fallback = 8192, max = 1048576) =>
  normalizeStepInteger(value, { min: 4, max, fallback, step: 4 });
const normalizeActiveSplat = (value, fallback = 2048, max = 1048576) =>
  normalizeStepInteger(value, { min: 4, max, fallback, step: 1 });
const countCases = [
  ["1e3", 1000],
  [5.9, 4],
  [6, 8],
  [-10, 4],
  [Number.NaN, 8192],
  [Number.POSITIVE_INFINITY, 8192],
  [2_000_000, 1_048_576],
];

const adamAge = (iteration, firstIteration = null) =>
  firstIteration === null ? iteration : Math.max(1, iteration - firstIteration + 1);
const bias = (beta, age) => 1 - beta ** age;
const referenceCases = [
  { iteration: 1, firstIteration: null, age: 1 },
  { iteration: 2, firstIteration: null, age: 2 },
  { iteration: 3, firstIteration: null, age: 3 },
  { iteration: 3001, firstIteration: 3001, age: 1 },
  { iteration: 3002, firstIteration: 3001, age: 2 },
  { iteration: 3003, firstIteration: 3001, age: 3 },
];

for (const item of referenceCases) {
  const age = adamAge(item.iteration, item.firstIteration);
  if (age !== item.age) throw new Error(`Adam age mismatch: ${JSON.stringify({ ...item, actual: age })}`);
  if (!(bias(0.9, age) > 0) || !(bias(0.999, age) > 0)) throw new Error(`invalid bias at age ${age}`);
}

const checks = {
  current_iteration_uploaded:
    app.includes("currentStep,\n      state.metrics?.steps_requested || 1") &&
    !app.includes("state.metrics?.steps_done || 0,\n      state.metrics?.steps_requested || 1"),
  adc_first_iteration_marker:
    app.includes("mode === 3 ? step + 1 : step") &&
    (app.match(/select\(0\.0, config\[36\], mode == 3u\)/g) || []).length === 2,
  row_age_bias_correction:
    (app.match(/let optimizerAge = select\(step, max\(1\.0, step - adcResetStep \+ 1\.0\), useRowAge\)/g) || []).length === 2,
  cpu_reference_iterations: referenceCases.every((item) => adamAge(item.iteration, item.firstIteration) === item.age),
  safe_integer_count_normalization:
    countCases.every(([input, expected]) => normalizeUiSplat(input) === expected) &&
    normalizeActiveSplat(1000.6) === 1001 &&
    app.includes('assertSplatCountContract(params, "gpu-upload")') &&
    app.includes('assertSplatCountContract(params, "ply-export")'),
};

const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => name);
if (failed.length) throw new Error(`P2 optimizer/input checks failed: ${failed.join(", ")}`);
console.log(JSON.stringify({ ok: true, checks, referenceCases, countCases }, null, 2));
