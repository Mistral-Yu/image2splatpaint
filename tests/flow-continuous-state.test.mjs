import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

async function load() {
  const context = vm.createContext({ Float32Array, Int32Array, Uint32Array, Math, Number, Map, Set });
  const topology = await read("web/training/flow-stroke-topology.js");
  vm.runInContext(topology.replace(
    "global.Image2SplatPaintFlowStrokeTopology =",
    "global.continuousTopologyTest = { mapSurvivingStrokeRows, splitStroke, mergePair, normalizedOptions, applyPaintCurriculum, removePaintCurriculum }; global.Image2SplatPaintFlowStrokeTopology =",
  ), context);
  const trainer = await read("web/training/flow-ribbon-trainer.js");
  vm.runInContext(trainer.replace(
    "global.Image2SplatPaintFlowRibbonTrainer =",
    "global.continuousTrainerTest = { remapContinuousOptimizerState, UPDATE_WGSL }; global.Image2SplatPaintFlowRibbonTrainer =",
  ), context);
  return { ...context.continuousTopologyTest, ...context.continuousTrainerTest, trainerSource: trainer };
}

function stroke(id, x = 0, layer = 1) {
  return {
    topology_state_id: id,
    topology_source_index: x,
    topology_generation: 0,
    start_x: x,
    start_y: 0,
    control_1_x: x + 2,
    control_1_y: 0,
    control_2_x: x + 4,
    control_2_y: 0,
    end_x: x + 6,
    end_y: 0,
    half_width_px: 3,
    topology_target_half_width_px: 3,
    topology_curriculum_length_scale: 1,
    opacity: 0.995,
    paint_linear_r: 0.2,
    paint_linear_g: 0.3,
    paint_linear_b: 0.4,
    random: 0.2 + x * 0.01,
    layer,
  };
}

function advanceAdamScalar(state, gradient, learningRate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
  const age = state.age + 1;
  const firstMoment = beta1 * state.firstMoment + (1 - beta1) * gradient;
  const secondMoment = beta2 * state.secondMoment + (1 - beta2) * gradient * gradient;
  const bias1 = Math.max(1e-8, 1 - beta1 ** age);
  const bias2 = Math.max(1e-8, 1 - beta2 ** age);
  return {
    value: state.value - learningRate * (firstMoment / bias1) / (Math.sqrt(secondMoment / bias2) + epsilon),
    firstMoment,
    secondMoment,
    age,
  };
}

test("Flow survivor mapping retains only stable rows across reorder, growth, split, and merge", async () => {
  const { mapSurvivingStrokeRows, splitStroke, mergePair, normalizedOptions } = await load();
  const options = normalizedOptions({ maximumWidthPx: 64, imageLongSide: 128 });
  const a = stroke("source:0", 0);
  const b = stroke("source:1", 10);
  const c = stroke("source:2", 20);
  const [splitA] = splitStroke(a, options);
  const merged = mergePair(b, c, options);
  assert.notEqual(splitA.topology_state_id, a.topology_state_id);
  assert.notEqual(merged.topology_state_id, b.topology_state_id);
  assert.notEqual(merged.topology_state_id, c.topology_state_id);
  const next = [c, a, stroke("source:3", 30), splitA, merged];
  assert.deepEqual(Array.from(mapSurvivingStrokeRows([a, b, c], next)), [2, 0, -1, -1, -1]);
});

test("Flow optimizer remap keeps finite survivor m/v/age and resets all other detail rows", async () => {
  const { remapContinuousOptimizerState, trainerSource } = await load();
  const stride = 16;
  const previous = {
    firstMoment: new Float32Array(4 * stride),
    secondMoment: new Float32Array(4 * stride),
    adamSteps: new Uint32Array([0, 7, 11, 13]),
  };
  for (let row = 0; row < 4; row += 1) for (let component = 0; component < stride; component += 1) {
    previous.firstMoment[row * stride + component] = row * 100 + component;
    previous.secondMoment[row * stride + component] = row * 1000 + component;
  }
  previous.firstMoment[2 * stride] = Infinity;
  const remapped = remapContinuousOptimizerState(
    { strokeCount: 6, underpaintParentCount: 2 },
    {
      previousTrainingState: previous,
      previousDetailOffset: 1,
      survivorRows: new Int32Array([2, 0, -1, 1]),
    },
  );
  assert.deepEqual(JSON.parse(JSON.stringify(remapped.diagnostics)), {
    mode: "survivor-row-remap",
    scope: "this-stage-detail-row-initialization",
    source_available: true,
    retained_detail_rows: 2,
    reset_detail_rows: 2,
    frozen_backcoat_rows: 2,
  });
  assert.equal(remapped.firstMoment[2 * stride], 300);
  assert.equal(remapped.secondMoment[2 * stride], 3000);
  assert.equal(remapped.adamSteps[2], 13);
  assert.equal(remapped.firstMoment[3 * stride], 100);
  assert.equal(remapped.adamSteps[3], 7);
  for (const row of [0, 1, 4, 5]) {
    assert.equal(remapped.adamSteps[row], 0);
    assert.ok(Array.from(remapped.firstMoment.subarray(row * stride, (row + 1) * stride)).every((v) => v === 0));
    assert.ok(Array.from(remapped.secondMoment.subarray(row * stride, (row + 1) * stride)).every((v) => v === 0));
  }
  assert.ok(Array.from(remapped.firstMoment).every(Number.isFinite));
  assert.ok(Array.from(remapped.secondMoment).every(Number.isFinite));
  const disabled = remapContinuousOptimizerState(
    { strokeCount: 2, underpaintParentCount: 0 },
    false,
  );
  assert.equal(disabled.diagnostics.mode, "disabled");
  assert.equal(disabled.diagnostics.reset_detail_rows, 2);
  assert.match(trainerSource, /const continuousAdamCarry = options\.continuousState !== false/);
  assert.ok(trainerSource.includes("continuousAdamCarry ? 8 : momentBound"));
  // Bound includes bias correction, not just the uncorrected m/sqrt(v).
  for (let age = 1; age <= 100000; age += 1) {
    const ratio = 0.9 ** 2 / 0.999;
    const bound = Math.sqrt(0.1 ** 2 / 0.001
      * (1 - 0.999 ** age) / (1 - 0.9 ** age) ** 2
      * (1 - ratio ** age) / (1 - ratio));
    assert.ok(bound < 8);
  }
});

test("Flow survivor Adam age gives the same two-stage recurrence as one continuous run", async () => {
  const { UPDATE_WGSL } = await load();
  const gradients = [0.4, -0.15, 0.25, 0.1, -0.2];
  const learningRate = 0.02;
  let single = { value: 0.8, firstMoment: 0, secondMoment: 0, age: 0 };
  for (const gradient of gradients) single = advanceAdamScalar(single, gradient, learningRate);
  let stageOne = { value: 0.8, firstMoment: 0, secondMoment: 0, age: 0 };
  for (const gradient of gradients.slice(0, 3)) {
    stageOne = advanceAdamScalar(stageOne, gradient, learningRate);
  }
  let twoStage = { ...stageOne };
  for (const gradient of gradients.slice(3)) twoStage = advanceAdamScalar(twoStage, gradient, learningRate);
  for (const field of ["value", "firstMoment", "secondMoment"]) {
    assert.ok(Math.abs(single[field] - twoStage[field]) < 1e-12, field);
  }
  assert.equal(single.age, twoStage.age);
  assert.match(UPDATE_WGSL, /let step = f32\(prior_step \+ 1u\);/);
  assert.match(UPDATE_WGSL, /adam_steps\[stroke\] = min\(0xfffffffeu, prior_step \+ 1u\);/);
});

test("Flow curriculum carries learned width for every detail layer but not the frozen backcoat", async () => {
  const { normalizedOptions, applyPaintCurriculum, removePaintCurriculum } = await load();
  const options = normalizedOptions({
    imageLongSide: 128,
    maximumWidthPx: 64,
    paintCurriculumEnabled: true,
    startingWidthDivisor: 32,
    startingLengthPercent: 160,
  });
  for (const layer of [0, 1, 2]) {
    const applied = applyPaintCurriculum([stroke(`source:${layer}`, layer * 10, layer)], 0.25, options).plan[0];
    const restored = removePaintCurriculum({ ...applied, half_width_px: applied.half_width_px + 0.35 }, options);
    assert.ok(Math.abs(restored.topology_target_half_width_px - 3.35) < 1e-6, `layer ${layer}`);
  }
  const backcoat = { ...stroke("backcoat", 40, 3), underpaint_splat: true };
  const appliedBackcoat = applyPaintCurriculum([backcoat], 0.25, options).plan[0];
  const restoredBackcoat = removePaintCurriculum(
    { ...appliedBackcoat, half_width_px: appliedBackcoat.half_width_px + 0.35 },
    options,
  );
  assert.equal(restoredBackcoat.topology_target_half_width_px, 3);
});
