import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const read = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

test("Flow Rectangle seeds reuse BSP and preserve counts, pigment and the normal curved representation", async () => {
  const context = vm.createContext({ Float32Array, Float64Array });
  for (const file of ["web/training/initialization.js", "web/training/initialization-runtime.js",
    "web/training/flow-splat-fusion.js", "web/training/flow-paint-reference.js"]) {
    vm.runInContext(await read(file), context);
  }
  // Use the actual shared BSP. A fixed tensor angle isolates the conversion
  // from per-axis NDC to image-pixel orientation for this numerical contract.
  vm.runInContext("strokeStructureAt = () => ({theta: Math.PI / 4, coherence: 0.5});", context);
  const image = { width: 64, height: 32, rgb: new Float32Array(64 * 32 * 3).fill(0.5) };
  const initializer = context.flowRectangleSeedCandidates;
  assert.equal(initializer(image, 0, 0, 240825).length, 0);
  const candidates = initializer(image, 20, 0, 240825);
  assert.equal(candidates.length, 20);
  assert.equal(JSON.stringify(candidates), JSON.stringify(initializer(image, 20, 0, 240825)));
  for (const seed of candidates) {
    assert.ok(seed.x >= 0 && seed.x < image.width && seed.y >= 0 && seed.y < image.height);
    assert.ok(seed.halfWidth > 0 && seed.halfLength > 0);
    assert.ok(Math.abs(seed.directionY / seed.directionX - 31 / 63) < 1e-10);
  }
  const create = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference;
  const options = { seed: 240825, profile: "connected-ribbon-v1", maxStrokes: 100,
    minimumStrokes: 1, includeStrokePlan: true, planOnly: true,
    maximumRibbonArcFraction: 0.1, textureGuidedAllocation: true };
  const result = create(image, { ...options, regionSeedInitializer: initializer });
  assert.equal(result.strokePlan.length, 100);
  assert.deepEqual(Array.from(result.metadata.layers, (layer) => layer.accepted), [20, 40, 40]);
  const baseline = create(image, options);
  assert.notEqual(JSON.stringify(result.strokePlan), JSON.stringify(baseline.strokePlan));
  for (const s of result.strokePlan) {
    assert.ok(s.path_length_px <= 3.2 + 1e-5);
    assert.ok(Number.isFinite(s.paint_linear_r + s.paint_linear_g + s.paint_linear_b));
    assert.ok(Math.abs(s.paint_linear_r - 0.21404114) < 1e-6, "center-sampled pigment is linear sRGB");
  }
  const rings = { width: 96, height: 96, rgb: new Float32Array(96 * 96 * 3) };
  for (let y = 0; y < 96; y += 1) for (let x = 0; x < 96; x += 1) {
    const color = 0.5 + 0.1 * Math.cos(Math.hypot(x - 48, y - 48) * 0.25);
    rings.rgb.fill(color, (y * 96 + x) * 3, (y * 96 + x) * 3 + 3);
  }
  const curved = create(rings, { ...options, regionSeedInitializer: initializer }).strokePlan;
  const bent = curved.filter((s) => {
    const dx = s.end_x - s.start_x, dy = s.end_y - s.start_y;
    return ["control_1", "control_2"].some((key) => Math.abs(
      (s[`${key}_x`] - s.start_x) * dy - (s[`${key}_y`] - s.start_y) * dx,
    ) > 0.01);
  });
  assert.ok(bent.length > 10, "curving image flow must not collapse into straight seeds");
  const topology = await read("web/training/flow-stroke-topology.js");
  const runtime = await read("web/training/flow-splat-fusion.js");
  assert.match(topology, /function mergePair\(/);
  assert.match(runtime, /const representation = "curve-splat-chain"/);
});

test("Flow retires Source-guided curves and can explicitly select the shared dab optimizer", async () => {
  const html = await read("web/index.html");
  const controls = await read("web/ui/training-controls.js");
  const runtime = await read("web/training/flow-splat-fusion.js");
  assert.ok(!html.includes('value="source-guided"'));
  assert.ok(!html.includes('flowSplatFusionInitialization'));
  assert.ok(!controls.includes('flowSplatFusionInitialization'));
  assert.match(runtime, /regionSeedInitializer: flowRectangleSeedCandidates/);
  assert.match(runtime, /createAdaptiveBrushSeeds\(trainingImage/);
  assert.match(runtime, /flowBirthLinked/);
  assert.match(runtime, /trainGaussianAlgorithm\(false, run\)/);
  assert.doesNotMatch(runtime, /createFlowPaintReference\(/);
  assert.doesNotMatch(runtime, /initRectangles\(|initOpaqueLayeredPaint\(/);
  assert.ok(!runtime.includes('rectangleSeeds ?'));
  assert.match(runtime, /geometry_rms_px: stageResult\.metadata\.control_point_rms_drift_px/);
});
