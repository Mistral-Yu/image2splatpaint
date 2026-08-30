import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const read = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

test("Flow compacts growth optimizer work but preserves count stages and full-count settle", async () => {
  const source = await read("web/training/flow-splat-fusion.js");
  const context = vm.createContext({});
  vm.runInContext(source, context);
  const build = context.buildFlowProgressiveGrowthSchedule;
  const schedule = build(3000, 184, 921);
  assert.equal(schedule.parentCounts.length, 28);
  assert.equal(schedule.parentCounts[0], 184);
  assert.equal(schedule.parentCounts.at(-1), 921);
  assert.equal(schedule.growthIterationBudget, 540);
  assert.equal(schedule.settleIterations, 300);
  assert.equal(schedule.plannedIterations, 840);
  assert.equal(schedule.skippedIterations, 2160);
  for (const limit of [1, 2, 3, 19, 20, 21, 99, 100, 101, 500, 3000, 7000, 10000]) {
    for (const [broad, full] of [[1, 1], [1, 2], [3, 10], [184, 921]]) {
      const s = build(limit, broad, full);
      assert.equal(s.parentCounts.length, s.iterationCounts.length);
      assert.equal(s.parentCounts.at(-1), full);
      assert.ok(s.iterationCounts.every((n) => Number.isInteger(n) && n > 0));
      assert.ok(s.parentCounts.every((n, i, a) => n <= full && (!i || n > a[i - 1])));
      assert.ok(s.iterationCounts.slice(0, -1).every((n) => n <= 20));
      assert.equal(s.iterationCounts.reduce((a, b) => a + b, 0), s.plannedIterations);
      assert.equal(s.plannedIterations + s.skippedIterations, limit);
      assert.equal(s.iterationCounts.at(-1), s.settleIterations);
    }
  }
  assert.match(source, /steps_requested: iterations/);
  assert.match(source, /flow_requested_iteration_limit: requestedIterationLimit/);
  assert.match(source, /const iterations = growthSchedule\.plannedIterations/);
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
