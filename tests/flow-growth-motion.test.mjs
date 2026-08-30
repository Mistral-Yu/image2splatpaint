import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const read = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

test("Flow executes one update per ten schedule steps including settle, retaining count stages", async () => {
  const source = await read("web/training/flow-splat-fusion.js");
  const context = vm.createContext({});
  vm.runInContext(source, context);
  const build = context.buildFlowProgressiveGrowthSchedule;
  const schedule = build(3000, 184, 921);
  assert.equal(schedule.parentCounts.length, 28);
  assert.equal(schedule.parentCounts[0], 184);
  assert.equal(schedule.parentCounts.at(-1), 921);
  assert.equal(schedule.growthIterationBudget, 270);
  assert.equal(schedule.settleIterations, 30);
  assert.equal(schedule.plannedIterations, 300);
  assert.equal(schedule.skippedIterations, 2700);
  for (const limit of [1, 2, 3, 19, 20, 21, 99, 100, 101, 500, 3000, 7000, 10000]) {
    for (const [broad, full] of [[1, 1], [1, 2], [3, 10], [184, 921]]) {
      const s = build(limit, broad, full);
      assert.equal(s.parentCounts.length, s.iterationCounts.length);
      assert.equal(s.parentCounts.at(-1), full);
      assert.ok(s.iterationCounts.every((n) => Number.isInteger(n) && n > 0));
      assert.ok(s.parentCounts.every((n, i, a) => n <= full && (!i || n > a[i - 1])));
      assert.ok(s.plannedIterations >= Math.ceil(limit / 10));
      assert.equal(s.iterationCounts.reduce((a, b) => a + b, 0), s.plannedIterations);
      assert.equal(s.plannedIterations + s.skippedIterations, limit);
      assert.equal(s.iterationCounts.at(-1), s.settleIterations);
      if (broad === full || limit < 3) assert.equal(s.plannedIterations, Math.ceil(limit / 10));
    }
  }
  assert.match(source, /steps_requested: iterations/);
  assert.match(source, /flow_requested_iteration_limit: requestedIterationLimit/);
  assert.match(source, /const iterations = growthSchedule\.plannedIterations/);
  assert.equal(build(3000, 921, 921).plannedIterations, 300, "fixed-count training also skips work");
});

test("Flow width phases follow real global updates and restore the original GPU config in P3", async () => {
  const source = await read("web/training/flow-ribbon-trainer.js");
  const context = vm.createContext({});
  vm.runInContext(source.replace(
    "global.Image2SplatPaintFlowRibbonTrainer =",
    "global.configBytesTest = configBytes; global.Image2SplatPaintFlowRibbonTrainer =",
  ), context);
  const settings = context.Image2SplatPaintFlowRibbonTrainer.widthTrainingSettings;
  const options = { widthTrainingPhases: true, globalIterations: 300, iterations: 10 };
  const p1 = settings(0, options);
  assert.equal(p1.phase, 1);
  assert.equal(p1.widthAnchor, 0.0002);
  assert.equal(p1.widthLearningRate, 0.015);
  assert.equal(settings(99, options).phase, 1);
  assert.equal(settings(100, options).phase, 2);
  assert.equal(settings(199, options).phase, 2);
  assert.equal(settings(200, options).phase, 3);
  assert.equal(p1.minimumWidthFactor, 1.5);
  assert.equal(p1.maximumWidthFactor, 3);
  assert.equal(settings(120, options).minimumWidthFactor, 1);
  assert.equal(settings(120, options).maximumWidthFactor, 2);
  assert.equal(settings(210, options).maximumWidthMultiplier, 1);
  assert.equal(settings(210, options).minimumWidthFactor, 0, "P3 restores the normal absolute minimum");
  assert.equal(settings(299, options).widthAnchor, 0.0008);
  assert.equal(settings(299, options).widthLearningRate, 0.010);
  assert.equal(settings(0, { ...options, globalIterations: 1 }).phase, 3);
  let previous = p1;
  for (let step = 1; step < 300; step += 1) {
    const current = settings(step, options);
    assert.ok(current.widthAnchor >= previous.widthAnchor);
    assert.ok(current.widthLearningRate <= previous.widthLearningRate);
    assert.ok(current.widthAnchor >= 0.0002 && current.widthAnchor <= 0.0008);
    assert.ok(current.widthLearningRate >= 0.010 && current.widthLearningRate <= 0.015);
    assert.equal(JSON.stringify(current), JSON.stringify(settings(step % 10, {
      ...options, globalIterationOffset: Math.floor(step / 10) * 10,
    })), "growth stages must not restart the phase schedule");
    previous = current;
  }
  // Phases change width limits, not paint layers, position bounds, opacity,
  // Adam's real local step, or the existing front-width multiplier.
  const data = {
    width: 512, height: 344, strokeCount: 921, tileCols: 32, tileRows: 22,
    sampleCount: 8, tileSampleStride: 1, canvasLinear: [0.5, 0.5, 0.5],
  };
  const config = (o) => context.configBytesTest(data, 7, o);
  const early = config(options);
  const old = config({ ...options, widthTrainingPhases: false });
  const late = config({ ...options, globalIterationOffset: 200 });
  assert.deepEqual(Array.from(late), Array.from(old), "P3 is byte-identical to old config");
  const view = new DataView(early.buffer);
  assert.equal(view.getFloat32(40, true), Math.fround(0.0002));
  assert.equal(view.getFloat32(52, true), Math.fround(0.015));
  assert.equal(view.getFloat32(80, true), Math.fround(344 * 0.09 * 3));
  assert.equal(view.getFloat32(140, true), 1.5);
  assert.equal(view.getFloat32(144, true), 3);
  for (let byte = 0; byte < old.length; byte += 1) {
    if ([40, 52, 80, 140, 144].some((start) => byte >= start && byte < start + 4)) continue;
    assert.equal(early[byte], old[byte], `unrelated GPU config byte ${byte}`);
  }
  const custom = settings(299, { ...options, widthAnchor: 0.005, widthLearningRate: 0.02 });
  assert.equal(custom.widthAnchor, 0.005);
  assert.equal(custom.widthLearningRate, 0.02);
  assert.equal(settings(0, { ...options, widthAnchor: 0, widthLearningRate: 0 }).widthAnchor, 0);
});

test("Flow phase width floors do not compound and leave fixed backcoat untouched", async () => {
  const source = await read("web/training/flow-ribbon-trainer.js");
  const context = vm.createContext({ Float32Array });
  vm.runInContext(source.replace(
    "global.Image2SplatPaintFlowRibbonTrainer =",
    "global.widthTest = { widthTrainingBounds, paramsWithoutPhaseLift }; global.Image2SplatPaintFlowRibbonTrainer =",
  ), context);
  const { widthTrainingBounds, paramsWithoutPhaseLift } = context.widthTest;
  const prepare = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData;
  const options = { widthTrainingPhases: true, globalIterations: 300, iterations: 10,
    representation: "curve-splat-chain", brushDabs: true, fixedOpacity: 0.995, maxPositionDelta: 0 };
  const image = { width: 128, height: 128, rgb: new Float32Array(128 * 128 * 3).fill(0.5) };
  const makeStroke = (layer, halfWidth = 4) => ({
    start_x: 40, start_y: 64, control_1_x: 52, control_1_y: 64,
    control_2_x: 68, control_2_y: 64, end_x: 80, end_y: 64,
    half_width_px: halfWidth, layer, random: 0.25, opacity: 0.995,
    paint_linear_r: 0.2, paint_linear_g: 0.3, paint_linear_b: 0.4,
  });
  const plan = [makeStroke(0), makeStroke(1), makeStroke(2), { ...makeStroke(3), underpaint_splat: true }];
  const p1 = prepare(image, plan, options);
  for (let i = 0; i < 3; i += 1) {
    assert.equal(p1.params[i * 16 + 8], 6);
    assert.equal(p1.anchors[i * 16 + 8], 4);
    assert.equal(p1.widthPhaseLifts[i], 2);
  }
  assert.equal(p1.widthPhaseLifts[3], 0);
  const restored = paramsWithoutPhaseLift(p1.params, p1.widthPhaseLifts);
  assert.equal(restored[8], 4);
  assert.equal(p1.params[8], 6, "normalization does not mutate rendered GPU state");
  const learned = p1.params.slice();
  learned[8] += 0.25;
  assert.equal(paramsWithoutPhaseLift(learned, p1.widthPhaseLifts)[8], 4.25);
  let currentWidth = 4;
  for (let stage = 0; stage < 10; stage += 1) {
    const data = prepare(image, [makeStroke(2, currentWidth)], { ...options, globalIterationOffset: stage * 10 });
    assert.equal(data.params[8], 6);
    currentWidth = paramsWithoutPhaseLift(data.params, data.widthPhaseLifts)[8];
  }
  const bounds = (step) => JSON.parse(JSON.stringify(widthTrainingBounds(4, 128, 128, step, options)));
  assert.deepEqual(bounds(0), { minimum: 6, maximum: 12 });
  assert.deepEqual(bounds(120), { minimum: 4, maximum: 8 });
  assert.deepEqual(bounds(210), { minimum: 0.55, maximum: 128 * 0.09 });
  const plain = prepare(image, plan, { ...options, widthTrainingPhases: false });
  const p3 = prepare(image, plan, { ...options, globalIterationOffset: 210 });
  assert.deepEqual(Array.from(p3.params), Array.from(plain.params));
  // The fixed backcoat candidate set is unchanged by the phase feature.
  for (let tile = 0; tile < plain.tileCols * plain.tileRows; tile += 1) {
    const includesBackcoat = (data) => Array.from(data.tileIndices.subarray(
      data.tileOffsets[tile], data.tileOffsets[tile + 1])).includes(3 * data.sampleCount);
    assert.equal(includesBackcoat(p1), includesBackcoat(plain));
  }
});

test("Flow selection never forces the settings disclosure open or closed", async () => {
  const source = await read("web/ui/training-controls.js");
  const html = await read("web/index.html");
  assert.doesNotMatch(source, /flowSplatFusionPanel\.open\s*=/);
  assert.doesNotMatch(html, /<details\b[^>]*id="flowSplatFusionPanel"[^>]*\sopen(?:\s|=|>)/);
});

test("Flow seeded placement is deterministic, bounded, pigment-compatible and can make substantial moves", async () => {
  const source = await read("web/training/flow-stroke-topology.js");
  const context = vm.createContext({ Float32Array });
  vm.runInContext(source.replace(
    "global.Image2SplatPaintFlowStrokeTopology =",
    "global.placementTest = { optimizeResidualPlacement, normalizedOptions }; global.Image2SplatPaintFlowStrokeTopology =",
  ), context);
  const { optimizeResidualPlacement, normalizedOptions } = context.placementTest;
  const width = 32, height = 32;
  const image = { width, height, rgb: new Float32Array(width * height * 3) };
  const rendered = new Float32Array(width * height * 4);
  for (let y = 12; y < 18; y += 1) {
    for (let x = 0; x < width; x += 1) image.rgb.fill(1, (y * width + x) * 3, (y * width + x) * 3 + 3);
  }
  const plan = Array.from({ length: 20 }, (_, index) => ({
    start_x: 8, start_y: 8, control_1_x: 11, control_1_y: 8,
    control_2_x: 15, control_2_y: 8, end_x: 18, end_y: 8,
    center_x: 13, center_y: 8, topology_origin_center_x: 13, topology_origin_center_y: 8,
    half_width_px: 1, opacity: 0.995, layer: 1, random: index / 20,
    paint_linear_r: 1, paint_linear_g: 1, paint_linear_b: 1,
  }));
  const options = normalizedOptions({ residualMovePerEventPx: 6, maximumTotalMovementPx: 12 });
  const run = (p = plan, o = options, event = 1) => optimizeResidualPlacement(p, image, rendered, 0, o, null, event);
  const a = run();
  assert.equal(JSON.stringify(a), JSON.stringify(run()));
  assert.ok(a.movedCount > 0 && a.movedCount <= 2);
  assert.ok(a.meanMovePx > 1.5 && a.meanMovePx <= 6);
  assert.ok(a.meanGain > 0);
  assert.ok(a.plan.every((stroke) => stroke.opacity === 0.995));
  assert.equal(run(plan, normalizedOptions({ maximumTotalMovementPx: 0 })).movedCount, 0);
  assert.equal(run(plan, normalizedOptions({ residualMovePerEventPx: 0 })).movedCount, 0);
  const wrongPigment = plan.map((stroke) => ({ ...stroke, paint_linear_r: 0, paint_linear_g: 0, paint_linear_b: 0 }));
  assert.equal(run(wrongPigment).movedCount, 0, "high residual is not sufficient when the pigment cannot reduce it");
  const noResidual = new Float32Array(width * height * 4);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    noResidual.set(image.rgb.subarray(pixel * 3, pixel * 3 + 3), pixel * 4);
  }
  assert.equal(optimizeResidualPlacement(plan, image, noResidual, 0, options).movedCount, 0);
  let moving = plan;
  for (let event = 1; event <= 20; event += 1) {
    moving = run(moving, options, event).plan;
    assert.ok(moving.every((s) => Math.hypot(s.center_x - 13, s.center_y - 8) <= 12 + 1e-6));
  }
});
