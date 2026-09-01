import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
async function load() {
  const context = vm.createContext({ Float32Array, Float64Array });
  vm.runInContext(await read("web/training/flow-ribbon-trainer.js"), context);
  vm.runInContext(await read("web/training/flow-paint-reference.js"), context);
  return { trainer: context.Image2SplatPaintFlowRibbonTrainer,
    reference: context.Image2SplatPaintFlowPaintReference };
}

async function loadBirthGraph() {
  const context = vm.createContext({ Int32Array, Set, Map });
  vm.runInContext(await read("web/training/flow-birth-graph.js"), context);
  return context.Image2SplatPaintFlowBirthGraph.BirthGraph;
}
const stroke = (random = 0.4) => ({
  start_x: 24, start_y: 40, control_1_x: 36, control_1_y: 28,
  control_2_x: 56, control_2_y: 48, end_x: 72, end_y: 40,
  half_width_px: 4, layer: 1, random, opacity: 0.995,
  paint_linear_r: 0.2, paint_linear_g: 0.3, paint_linear_b: 0.4,
  texture_score: 1, edge_score: 0,
});

test("Variable Brush links use stable 3–9 counts and reconcile the physical budget", async () => {
  const { trainer } = await load();
  const plan = Array.from({ length: 70 }, (_, i) => stroke((i + 0.5) / 70));
  const baseline = JSON.stringify(plan);
  for (const budget of [210, 337, 420, 523, 630]) {
    const result = trainer.allocateBrushDabCounts(plan, budget, true);
    assert.equal(result.physicalSplatCount, budget);
    assert.ok(result.plan.every((s) => s.brush_dab_count >= 3 && s.brush_dab_count <= 9));
    assert.equal(JSON.stringify(result), JSON.stringify(trainer.allocateBrushDabCounts(plan, budget, true)));
    assert.equal(JSON.stringify(result), JSON.stringify(trainer.allocateBrushDabCounts(result.plan, budget, true)));
  }
  const result = trainer.allocateBrushDabCounts(plan, 420, true);
  assert.equal(new Set(result.plan.map((s) => s.brush_dab_count)).size, 7);
  assert.equal(JSON.stringify(plan), baseline, "allocation must not mutate the reference");
  assert.throws(() => trainer.allocateBrushDabCounts(plan, 209), /at least 3/);
  assert.equal(trainer.allocateBrushDabCounts([], 10, true).physicalSplatCount, 0);
  const bounded = trainer.allocateBrushDabCounts(plan, 350, true, {minimum: 4, maximum: 6});
  assert.ok(bounded.plan.every((s) => s.brush_dab_count >= 4 && s.brush_dab_count <= 6));
  assert.equal(bounded.physicalSplatCount, 350);
  assert.throws(
    () => trainer.allocateBrushDabCounts(plan, 279, false, {minimum: 4, maximum: 6}),
    /at least 4/,
  );
});

test("Birth-linked stroke range delays continuity until Min and stops linking at Max", async () => {
  const BirthGraph = await loadBirthGraph();
  const graph = new BirthGraph(1, { minMembers: 3, maxMembers: 4 });
  const splitFrom = (row) => Uint32Array.of((1 << 30) | (row + 1));

  assert.equal(graph.grow(1, splitFrom(0), 10)[0].linked, true);
  assert.equal(graph.pack().edges.length, 0, "two members remain dormant below Min");
  assert.equal(graph.pack(() => true, { includeDormant: true }).edges.length, 1);

  assert.equal(graph.grow(2, splitFrom(1), 20)[0].linked, true);
  assert.equal(graph.pack().edges.length, 2, "three members activate continuity");
  assert.equal(graph.grow(3, splitFrom(2), 30)[0].linked, true);
  assert.equal(graph.pack().edges.length, 3);

  assert.equal(graph.grow(4, splitFrom(3), 40)[0].linked, false,
    "a fifth member must not exceed Max");
  assert.equal(graph.pack().edges.length, 3);
});

test("Birth-linked strokes can seed local P1 chains before growth", async () => {
  const BirthGraph = await loadBirthGraph();
  const graph = new BirthGraph(8, { minMembers: 2, maxMembers: 5 });
  graph.seedChains([[1, 2], [4, 5]]);
  const packed = graph.pack();
  assert.equal(packed.edges.length, 2);
  assert.deepEqual([...packed.groups].map((group) => group.length), [2, 2]);
  assert.throws(() => graph.seedChains([[2, 3]]), /Invalid initial chain row/);
  assert.throws(() => graph.seedChains([[6, 7, 0, 3, 4, 5]]), /Invalid initial chain/);
});

test("A full linked stroke starts a sibling curve fragment instead of isolated split dots", async () => {
  const BirthGraph = await loadBirthGraph();
  const graph = new BirthGraph(4, { minMembers: 2, maxMembers: 4 });
  graph.seedChains([[0, 1, 2, 3]]);
  const splitWords = Uint32Array.from([0, 1].map((row) => (1 << 30) | (row + 1)));
  const events = graph.grow(4, splitWords, 50);
  assert.ok(events.every((event) => event.linked));
  assert.deepEqual(Array.from(graph.pack().groups, (group) => group.length).sort((a, b) => a - b), [2, 4]);
});

test("Variable tile candidates contain only active dabs, always keep three bodies, and count backcoat once", async () => {
  const { trainer, reference } = await load();
  const image = { width: 96, height: 80, rgb: new Float32Array(96 * 80 * 3).fill(0.4) };
  const rear = reference.createSplatUnderpaintPlan(image, { count: 8, representation: "curve-splat-chain" });
  const details = Array.from({ length: 7 }, (_, i) => ({ ...stroke((i + 1) / 9), brush_dab_count: i + 3 }));
  const plan = [...rear.strokePlan, ...details];
  const options = { brushDabs: true, variableBrushDabs: true, textureGuidedDabs: true,
    representation: "curve-splat-chain", fixedOpacity: 0.995, splatSizeVariation: 0.4,
    widthTrainingPhases: true, globalIterations: 300, iterations: 10 };
  const data = trainer.prepareTrainingData(image, plan, options);
  assert.equal(data.sampleCount, 9);
  assert.equal(data.physicalSplatCount, 8 + 42);
  assert.equal(data.initialPhysicalSplatScaleStats.physical_splat_count, 42);
  const unique = new Set(data.tileIndices);
  assert.equal(unique.size, 50);
  for (let parent = 0; parent < plan.length; parent++) {
    const samples = [...unique].filter((v) => Math.floor(v / 9) === parent).map((v) => v % 9);
    assert.equal(samples.length, parent < 8 ? 1 : plan[parent].brush_dab_count);
    if (parent >= 8) for (const body of [5, 6, 7]) assert.ok(samples.includes(body));
  }
  assert.ok(Array.from(data.params).every(Number.isFinite));
  const fixed = trainer.prepareTrainingData(image, plan, { ...options, variableBrushDabs: false });
  assert.equal(fixed.sampleCount, 8);
  assert.equal(fixed.physicalSplatCount, 8 + 7 * 8);
  const classic = trainer.prepareTrainingData(image, plan, { ...options, brushDabs: false });
  assert.equal(classic.sampleCount, 4);
  assert.equal(classic.physicalSplatCount, 8 + 7 * 4);
});

test("P1 backcoat has the same source pigment and full-cell coverage as the final backcoat", async () => {
  const { reference } = await load();
  const image = { width: 64, height: 96, rgb: new Float32Array(64 * 96 * 3) };
  for (let i = 0; i < image.rgb.length; i++) image.rgb[i] = (i % 91) / 91;
  const options = { count: 17, representation: "curve-splat-chain", seed: 240825 };
  const early = reference.createSplatUnderpaintPlan(image, options);
  const late = reference.createSplatUnderpaintPlan(image, { ...options,
    residualRender: new Float32Array(64 * 96 * 4).fill(0.1) });
  const fields = ["start_x", "start_y", "control_1_x", "control_1_y", "half_width_px",
    "opacity", "paint_linear_r", "paint_linear_g", "paint_linear_b"];
  for (let i = 0; i < early.strokePlan.length; i++) {
    for (const field of fields) assert.equal(early.strokePlan[i][field], late.strokePlan[i][field]);
  }
  for (let y = 0; y < image.height; y++) for (let x = 0; x < image.width; x++) {
    assert.ok(early.strokePlan.some((s) => x >= s.coverage_cell_min_x && x < s.coverage_cell_max_x
      && y >= s.coverage_cell_min_y && y < s.coverage_cell_max_y));
  }
  const integration = await read("web/training/flow-splat-fusion.js");
  assert.match(integration, /const rearPlan = finalUnderpaint.strokePlan/);
  assert.match(integration, /topologyParams.subarray\(\s*previousStage.metadata.underpaint_parent_count \* Image2SplatPaintFlowRibbonTrainer.constants.PARAM_STRIDE/);
  const html = await read("web/index.html");
  assert.match(html, /id="flowLinkedSplatMin"[^>]+value="2"/);
  assert.match(html, /id="flowLinkedSplatMax"[^>]+value="9"/);
  assert.match(html, /id="flowInternalBendControlPointCount"[^>]+value="1"/);
  assert.match(html, /id="flowInternalBendControlPointPositions"[^>]+value="50"/);
  assert.doesNotMatch(html, /id="flowSplatFusionVariableLinks"/);
  assert.match(html.match(/<input id="flowSplatBackcoatFromP1"[^>]+>/)[0], /checked/);
  const controls = await read("web/ui/training-controls.js");
  assert.match(controls, /flowLinkedSplatMin.disabled = state.running \|\| !sharedFlow \|\| internalBend/);
  assert.match(controls, /flowInternalBendControlPointCount.disabled = state.running \|\| !internalBend/);
});
