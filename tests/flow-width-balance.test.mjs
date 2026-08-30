import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
async function loadProfiles() {
  const source = await read("web/training/flow-ribbon-trainer.js");
  const context = vm.createContext({ Float32Array });
  vm.runInContext(source.replace("global.Image2SplatPaintFlowRibbonTrainer =",
    "global.profileTest = { chainSampleProfile, COMMON_WGSL }; global.Image2SplatPaintFlowRibbonTrainer ="), context);
  return { ...context.profileTest, source };
}

test("Brush widths retain two fine accents, three real middle marks, and three bodies", async () => {
  const { chainSampleProfile: profile } = await loadProfiles();
  for (const layer of [0, 1, 2]) for (const random of [0, 0.23, 0.79]) {
    for (const variation of [0, 0.4, 1]) {
      const widths = Array.from({ length: 8 }, (_, sample) =>
        profile(2, layer, sample, random, 8, 1, 1, variation).widthFactor);
      for (const sample of [0, 2]) assert.ok(widths[sample] < 0.21);
      for (const sample of [1, 3, 4]) {
        assert.ok(widths[sample] > 0.24 && widths[sample] < 0.56,
          `middle ${sample} must not compound into a hairline: ${widths[sample]}`);
      }
      for (const sample of [5, 6, 7]) assert.ok(widths[sample] > 0.55);
    }
  }
});

test("Weak and medium edges ease into thin strokes without narrowing the paint bodies", async () => {
  const { chainSampleProfile: profile } = await loadProfiles();
  for (const sample of [0, 2]) {
    const widths = [0, 1 / 3, 2 / 3, 1].map((edge) =>
      profile(2, 2, sample, 0.41, 8, 1, edge, 0.4).widthFactor);
    assert.ok(widths[1] > 0.58, "weak packed edge remains a broad accent");
    assert.ok(widths[2] > 0.30, "medium packed edge must not saturate to a hairline");
    assert.ok(widths[3] < 0.20, "strong edge can still carry a thin accent");
    for (let index = 1; index < widths.length; index++) assert.ok(widths[index] <= widths[index - 1]);
  }
  for (const sample of [1, 3, 4, 5, 6, 7]) {
    assert.equal(profile(2, 1, sample, 0.41, 8, 1, 0, 0.4).widthFactor,
      profile(2, 1, sample, 0.41, 8, 1, 1, 0.4).widthFactor);
  }
});

test("CPU tile profiles and shared forward/backward WGSL use matching width families and edge gates", async () => {
  const { source, COMMON_WGSL: wgsl } = await loadProfiles();
  const jsArray = (name) => source.match(new RegExp(`const ${name} = \\[([^\\]]+)\\]`))[1]
    .split(",").map(Number);
  const familyBlock = wgsl.split("fn chain_splat_width_variation")[1].split("fn chain_splat_length_variation")[0];
  const widthBlock = wgsl.split("fn chain_width_factor")[1].split("fn chain_length_factor")[0];
  const shaderFamily = (block, variable, fallback) => Array.from({ length: 9 }, (_, sample) => {
    const match = block.match(new RegExp(`sample == ${sample}u\\) \\{ ${variable} = ([0-9.]+);`));
    return match ? Number(match[1]) : fallback;
  });
  assert.deepEqual(shaderFamily(familyBlock, "family_scale", 1), jsArray("widthTargets"));
  assert.deepEqual(shaderFamily(widthBlock, "factor", 0.74), jsArray("baseWidthFactors"));
  assert.equal((wgsl.match(/smoothstep\(0\.30, 0\.95, chain_edge_score\(stroke\)\)/g) || []).length, 6);
  assert.ok(!/smoothstep\(0\.12, 0\.68, chain_edge_score/.test(wgsl.split("fn chain_pigment_scale")[0]),
    "all shape functions use the new gate; the established pigment formula is unchanged");
  for (const name of ["FORWARD_WGSL", "BACKWARD_WGSL", "CHAIN_QUAD_BACKWARD_WGSL"]) {
    const body = source.split(`const ${name} = String.raw`)[1].split("`;", 1)[0];
    assert.ok(body.includes("${COMMON_WGSL}"), `${name} must use the common shape functions`);
  }
});

test("Edge accents OFF skips XDoG but preserves curved geometry, pigments and texture allocation", async () => {
  const context = vm.createContext({ Float32Array, Float64Array });
  vm.runInContext(await read("web/training/flow-paint-reference.js"), context);
  const image = { width: 96, height: 96, rgb: new Float32Array(96 * 96 * 3) };
  for (let y = 0; y < 96; y++) for (let x = 0; x < 96; x++) {
    const value = 0.5 + 0.2 * Math.cos(Math.hypot(x - 48, y - 48) * 0.25);
    image.rgb.fill(value, (y * 96 + x) * 3, (y * 96 + x) * 3 + 3);
  }
  const create = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference;
  const options = { profile: "connected-ribbon-v1", seed: 240825, maxStrokes: 100,
    minimumStrokes: 1, includeStrokePlan: true, planOnly: true, textureGuidedAllocation: true };
  const on = create(image, { ...options, edgeGuidedAccents: true });
  const off = create(image, { ...options, edgeGuidedAccents: false });
  assert.ok(on.textureGuide.edgeScore.some((value) => value > 0.1));
  assert.ok(off.textureGuide.edgeScore.every((value) => value === 0));
  assert.equal(off.textureGuide.summary.flow_xdog, null);
  assert.equal(off.textureGuide.summary.edge_score_mode, "disabled");
  assert.deepEqual(off.textureGuide.score, on.textureGuide.score);
  const withoutEdge = (plan) => JSON.stringify(plan.map(({ edge_score, ...stroke }) => stroke));
  assert.equal(withoutEdge(off.strokePlan), withoutEdge(on.strokePlan));
  const html = await read("web/index.html");
  const input = html.match(/<input id="flowSplatFusionEdgeAccents"[^>]+>/)[0];
  assert.doesNotMatch(input, /checked/);
  assert.match(await read("web/ui/training-controls.js"), /flowSplatFusionEdgeAccents.disabled = state.running \|\| !flowSelected/);
});
